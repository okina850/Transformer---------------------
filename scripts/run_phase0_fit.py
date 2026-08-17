"""Phase 0 / Day 2: 次数 x 分母設計 x 近似区間 の全組合せで有理近似を作り、誤差を測る。

出力（``results/phase0/fit/<timestamp>_<tag>/``）:
- ``errors.csv``        : 全条件の誤差表
- ``coefficients.json`` : Phase 1 で使う初期化係数
- ``meta.json`` / ``config.yaml`` : 再現用メタ情報
- ``figures/``          : 誤差曲線・関数形状・裾の挙動

実行:
    python scripts/run_phase0_fit.py --config configs/phase0_fit.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.rational.denominators import make_denominator, eval_rational
from src.rational.fitting import fit_linearized, fit_nonlinear, make_inits
from src.rational.metrics import (
    error_metrics,
    eval_grid,
    get_grid,
    make_weight,
    smoothness_metrics,
)
from src.rational.pade import pade_at_zero_numeric
from src.rational.poly import vander_cond
from src.rational.targets import get_target
from src.utils.run_meta import load_yaml, new_run_dir, save_run_meta, set_seed


def _row(
    target: str,
    interval: str,
    lo: float,
    hi: float,
    m: int,
    n: int,
    denom_name: str,
    method: str,
    err: dict,
    extra: dict,
) -> dict:
    return {
        "target": target,
        "interval": interval,
        "lo": lo,
        "hi": hi,
        "m": m,
        "n": n,
        "denom": denom_name,
        "method": method,
        "sup": err["sup"],
        "argsup": err["argsup"],
        "l2": err["l2"],
        "rms": err["rms"],
        "n_nonfinite": err["n_nonfinite"],
        **extra,
    }


def run(cfg: dict, run_dir: Path) -> pd.DataFrame:
    fitcfg = cfg["fit"]
    weight_fn = make_weight(fitcfg.get("weight"))
    ext = cfg.get("extrapolation", {"lo": -15.0, "hi": 15.0})
    ext_lo, ext_hi = float(ext["lo"]), float(ext["hi"])
    rows: list[dict] = []
    coeffs: dict[str, dict] = {}

    for target_name in cfg["targets"]:
        f = get_target(target_name)

        for iv in cfg["intervals"]:
            lo, hi, iname = float(iv["lo"]), float(iv["hi"]), iv["name"]
            xf = get_grid(fitcfg.get("grid", "uniform"), lo, hi, fitcfg["n_fit"])
            xe = eval_grid(lo, hi, fitcfg["n_eval"])
            ff, fe = f(xf), f(xe)
            wf = weight_fn(xf)

            for deg in cfg["degrees"]:
                m, n = int(deg["m"]), int(deg["n"])
                t_block = time.perf_counter()
                cond = vander_cond(xf, m)

                # --- 参照1: x=0 での真の Pade 近似（論文の公式初期化の作り方） ---
                a_p, b_p = pade_at_zero_numeric(target_name, m, n)
                d_free = make_denominator("free")
                err = error_metrics(xe, fe, eval_rational(a_p, b_p, xe, d_free))
                rows.append(
                    _row(target_name, iname, lo, hi, m, n, "free", "pade@0", err,
                         {"n_params": (m + 1) + n, "cond_vander": cond,
                          "eff_deg_q": n, "tail_exp": m - n,
                          "n_sign_changes": np.nan, "n_poles": np.nan,
                          "init_spread": np.nan, "cost": np.nan})
                )

                # --- 参照2: 線形化最小二乗 (+SK)。非線形の初期値にもなる ---
                r_lin = fit_linearized(xf, ff, m, n, weights=wf,
                                       sk_iters=int(fitcfg.get("sk_iters", 0)))
                err = error_metrics(xe, fe, eval_rational(r_lin.a, r_lin.b, xe, d_free))
                rows.append(
                    _row(target_name, iname, lo, hi, m, n, "free", "linearized+sk", err,
                         {"n_params": (m + 1) + n, "cond_vander": cond,
                          "eff_deg_q": n, "tail_exp": m - n,
                          "n_sign_changes": np.nan,
                          "n_poles": len(d_free.poles(r_lin.b, lo, hi)),
                          "init_spread": np.nan, "cost": r_lin.cost})
                )

                # --- 本命: 各分母設計を非線形最小二乗で仕上げる ---
                variants = [(dspec, m) for dspec in cfg["denominators"]]
                if cfg.get("tail_matched_square"):
                    # 論点A: square で裾を abs と揃えるには m = 2n+1 が必要
                    variants.append(({"kind": "square"}, 2 * n + 1))

                for dspec, m_eff in variants:
                    d = make_denominator(dspec)
                    tail_matched = m_eff != m
                    label = f"{d.name}{'(tail-matched)' if tail_matched else ''}"

                    # 裾を揃えた変種 (m=2n+1) では Pade@0 を初期値に使わない。
                    # 分母の構造が違ううえ、m が大きく sympy の級数展開が非常に重い
                    # （n=6 なら m=13、19次まで展開することになる）。
                    if tail_matched:
                        r_lin2 = fit_linearized(xf, ff, m_eff, n, weights=wf,
                                                sk_iters=int(fitcfg.get("sk_iters", 0)))
                        pade_init = None
                    else:
                        r_lin2 = r_lin
                        pade_init = pade_at_zero_numeric(target_name, m_eff, n)

                    inits = make_inits(
                        m_eff, n, linear_fit=r_lin2, pade_coeffs=pade_init,
                        n_random=int(fitcfg.get("n_random_inits", 1)),
                        seed=int(cfg.get("seed", 0)),
                    )
                    best, allr = fit_nonlinear(xf, ff, m_eff, n, d, weights=wf, inits=inits)

                    approx = eval_rational(best.a, best.b, xe, d)
                    err = error_metrics(xe, fe, approx, weights=weight_fn(xe))
                    smooth = smoothness_metrics(xe, approx)

                    costs = [r.cost for r in allr]
                    spread = (max(costs) - min(costs)) / max(min(costs), 1e-300)

                    # 極・折れ点は **フィット区間の中と外の両方**で数える。
                    # 区間内だけ見ると「極も折れ点もない」と誤った結論になる。
                    # 実際 gelu_tanh [5,4] を [-3,3] で fit すると極は x≈±5.1、
                    # すなわち区間のすぐ外に出る。Transformer の pre-activation は
                    # そこに届きうるので、区間外こそが問題になる。
                    n_sign = len(d.kinks(best.b, lo, hi)) if d.name == "abs" else 0
                    n_poles = len(d.poles(best.b, lo, hi)) if d.can_have_poles else 0
                    n_sign_ext = len(d.kinks(best.b, ext_lo, ext_hi)) if d.name == "abs" else 0
                    n_poles_ext = len(d.poles(best.b, ext_lo, ext_hi)) if d.can_have_poles else 0
                    # 区間外での挙動を1つの数字で見るための代表点
                    probe = np.array([ext_lo / 1.5, ext_hi / 1.5])  # 既定なら x = -10, +10
                    f_probe = eval_rational(best.a, best.b, probe, d)
                    tgt_probe = f(probe)

                    rows.append(
                        _row(target_name, iname, lo, hi, m_eff, n, label, "nonlinear", err,
                             {"n_params": (m_eff + 1) + n, "cond_vander": vander_cond(xf, m_eff),
                              "eff_deg_q": d.effective_degree(n),
                              "tail_exp": d.tail_exponent(m_eff, n),
                              "n_sign_changes": n_sign, "n_poles": n_poles,
                              "n_sign_changes_ext": n_sign_ext, "n_poles_ext": n_poles_ext,
                              "f_at_neg_probe": float(f_probe[0]),
                              "f_at_pos_probe": float(f_probe[1]),
                              "target_at_neg_probe": float(tgt_probe[0]),
                              "target_at_pos_probe": float(tgt_probe[1]),
                              "init_spread": spread, "cost": best.cost,
                              "max_abs_d2": smooth["max_abs_d2"], "tv_d1": smooth["tv_d1"],
                              "best_init": best.init_source})
                    )

                    key = f"{target_name}|{iname}|m{m_eff}n{n}|{label}"
                    coeffs[key] = {
                        "target": target_name, "interval": iname, "lo": lo, "hi": hi,
                        "m": m_eff, "n": n, "denom": d.name,
                        "denom_params": d.params, "tail_matched": tail_matched,
                        "a": best.a.tolist(), "b": best.b.tolist(),
                        "sup": err["sup"], "l2": err["l2"],
                        "n_params": (m_eff + 1) + n,
                    }

                print(
                    f"  {target_name:10s} {iname:6s} [{m},{n}] done in "
                    f"{time.perf_counter() - t_block:6.2f}s",
                    flush=True,
                )

                # 論文どおりの Pade@0 初期化も Phase 1 用に保存（PAU 再現の担保）
                coeffs[f"{target_name}|{iname}|m{m}n{n}|pade@0"] = {
                    "target": target_name, "interval": iname, "lo": lo, "hi": hi,
                    "m": m, "n": n, "denom": "abs", "denom_params": {},
                    "tail_matched": False,
                    "a": a_p.tolist(), "b": b_p.tolist(),
                    "note": "true Pade approximant at x=0 (paper-style init)",
                    "n_params": (m + 1) + n,
                }

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "errors.csv", index=False, encoding="utf-8")
    (run_dir / "coefficients.json").write_text(
        json.dumps(coeffs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return df


def make_figures(cfg: dict, df: pd.DataFrame, run_dir: Path) -> None:
    """誤差曲線・関数形状・裾の挙動を描く。

    図中のラベルはすべて英語にする（Windows の matplotlib で日本語フォントが
    無いと豆腐文字になるため。図は成果物なので確実に読める方を優先する）。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = run_dir / "figures"
    figdir.mkdir(exist_ok=True)

    # --- 図1: 次数 vs 誤差（分母設計ごと・区間ごと） ---
    for target in df["target"].unique():
        for iname in df["interval"].unique():
            sub = df[(df["target"] == target) & (df["interval"] == iname)]
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            for metric, ax in zip(["sup", "l2"], axes):
                for denom in sorted(sub["denom"].unique()):
                    s = sub[(sub["denom"] == denom) & (sub["method"] == "nonlinear")]
                    if s.empty:
                        continue
                    s = s.sort_values("n")
                    ax.semilogy(s["n"], s[metric], "o-", label=denom, markersize=4)
                p = sub[sub["method"] == "pade@0"].sort_values("n")
                ax.semilogy(p["n"], p[metric], "k--x", label="pade@0", markersize=5)
                ax.set_xlabel("denominator degree n  (m = n+1)")
                ax.set_ylabel(f"{metric} error")
                ax.grid(True, alpha=0.3, which="both")
            axes[0].set_title(f"{target} / {iname}: sup error")
            axes[1].set_title(f"{target} / {iname}: L2 error")
            axes[1].legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(figdir / f"error_vs_degree_{target}_{iname}.png", dpi=140)
            plt.close(fig)

    # --- 図2: 裾の挙動（近似区間の外での外挿） ---
    ex = cfg["extrapolation"]
    xx = np.linspace(float(ex["lo"]), float(ex["hi"]), int(ex["n"]))
    coeffs = json.loads((run_dir / "coefficients.json").read_text(encoding="utf-8"))

    for target in df["target"].unique():
        f = get_target(target)
        for iname in df["interval"].unique():
            fig, ax = plt.subplots(figsize=(7.5, 4.6))
            ax.plot(xx, f(xx), "k-", lw=2.2, label=f"{target} (target)")
            for key, c in coeffs.items():
                if not key.startswith(f"{target}|{iname}|") or c["n"] != 4:
                    continue
                d = make_denominator({"kind": c["denom"], **c.get("denom_params", {})})
                y = eval_rational(np.array(c["a"]), np.array(c["b"]), xx, d)
                lbl = key.split("|")[-1] + f" [m={c['m']}]"
                ax.plot(xx, y, lw=1.3, alpha=0.9, label=lbl)
            ax.axvspan(c["lo"], c["hi"], color="gray", alpha=0.13, label="fit interval")
            ax.set_ylim(-8, 16)
            ax.set_xlabel("x")
            ax.set_ylabel("F(x)")
            ax.set_title(f"Extrapolation outside the fit interval ({target}, {iname}, n=4)")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(figdir / f"tails_{target}_{iname}.png", dpi=140)
            plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase0_fit.yaml")
    ap.add_argument("--tag", default="fit")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = set_seed(int(cfg.get("seed", 0)))
    run_dir = new_run_dir("phase0/fit", tag=args.tag)
    # 近似計算は numpy のみなので torch の import(数秒)を省く
    save_run_meta(run_dir, config=cfg, seed=seed, include_gpu=False)

    df = run(cfg, run_dir)
    if cfg.get("figures", True):
        make_figures(cfg, df, run_dir)

    print(f"wrote {len(df)} rows -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
