r"""Phase 0: 平滑分母 :math:`Q = 1+\sqrt{\varepsilon+A^2}` の :math:`\varepsilon` を決める。

:math:`\varepsilon` は2方向から挟まれる。

- **上限**: 大きすぎると :math:`\sqrt{\varepsilon}` の下駄で近似精度が頭打ちになる
- **下限**: 小さすぎると :math:`\sqrt{\varepsilon+A^2}\to|A|` となり ``abs`` に漸近し、
  Phase 1 の R3/R4 が R1/R2 と区別できなくなる（＝実験が空振りになる）

下限の判定には「平滑化が効いている x の範囲を、実際の pre-activation が
どれだけ通るか」が必要なので、**T5 の実測分布を使って重み付けする**。

出力（``results/phase0/eps/<timestamp>_<tag>/``）:
- ``eps_sweep.csv`` : eps ごとの近似精度・平滑化幅・影響を受ける活性化の割合
- ``figures/``      : 精度と影響割合のトレードオフ

実行:
    python scripts/run_phase0_eps.py --config configs/phase0_eps.yaml
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.rational.denominators import eval_rational, make_denominator
from src.rational.fitting import fit_linearized, fit_nonlinear, make_inits
from src.rational.metrics import error_metrics, eval_grid, get_grid, make_weight
from src.rational.pade import pade_at_zero_numeric
from src.rational.targets import get_target
from src.utils.run_meta import REPO_ROOT, load_yaml, new_run_dir, save_run_meta, set_seed


def load_preact_density(path: str | None) -> tuple[np.ndarray, np.ndarray]:
    """T5 の層別ヒストグラムを全層合算し、``(centers, probability)`` を返す。

    層ごとの分布はよく似ているので、分母設計の影響を1つの数字で言うには
    全層合算で十分（層別の差は T5 の成果物のほうで示す）。
    """
    if path is None:
        cands = sorted(glob.glob(str(REPO_ROOT / "results/phase0/preact/*/preact_stats.json")))
        if not cands:
            raise FileNotFoundError(
                "T5 の測定結果が見つからない。先に scripts/run_phase0_preact.py を実行すること。"
            )
        path = cands[-1]
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    centers = np.array(payload["layers"][0]["_centers"], dtype=np.float64)
    total = np.zeros_like(centers)
    for layer in payload["layers"]:
        total += np.array(layer["_hist"], dtype=np.float64)
    n = sum(layer["count"] for layer in payload["layers"])
    return centers, total / n  # ヒストグラム範囲外の質量は落ちる（極小なので許容）


def fit_one(target: str, lo: float, hi: float, m: int, n: int, denom, cfg: dict):
    fitcfg = cfg["fit"]
    f = get_target(target)
    xf = get_grid(fitcfg.get("grid", "uniform"), lo, hi, fitcfg["n_fit"])
    wf = make_weight(fitcfg.get("weight"))(xf)
    r_lin = fit_linearized(xf, f(xf), m, n, weights=wf, sk_iters=int(fitcfg.get("sk_iters", 0)))
    inits = make_inits(
        m, n, linear_fit=r_lin, pade_coeffs=pade_at_zero_numeric(target, m, n),
        n_random=int(fitcfg.get("n_random_inits", 1)), seed=int(cfg.get("seed", 0)),
    )
    best, _ = fit_nonlinear(xf, f(xf), m, n, denom, weights=wf, inits=inits)
    return best


def run(cfg: dict, run_dir: Path) -> pd.DataFrame:
    centers, prob = load_preact_density(cfg.get("preact_stats"))
    ex = cfg["extrapolation"]
    xx = np.linspace(float(ex["lo"]), float(ex["hi"]), int(ex["n"]))
    diff_tol = float(cfg["diff_tol"])
    d_abs = make_denominator("abs")
    rows: list[dict] = []

    for target in cfg["targets"]:
        f = get_target(target)
        for iv in cfg["intervals"]:
            lo, hi, iname = float(iv["lo"]), float(iv["hi"]), iv["name"]
            xe = eval_grid(lo, hi, cfg["fit"]["n_eval"])
            fe = f(xe)

            for deg in cfg["degrees"]:
                m, n = int(deg["m"]), int(deg["n"])

                # 基準となる abs のフィット
                r_abs = fit_one(target, lo, hi, m, n, d_abs, cfg)
                err_abs = error_metrics(xe, fe, eval_rational(r_abs.a, r_abs.b, xe, d_abs))
                y_abs = eval_rational(r_abs.a, r_abs.b, xx, d_abs)
                kinks, slopes = d_abs.kinks_with_severity(r_abs.b, float(ex["lo"]), float(ex["hi"]))

                for eps in cfg["eps_values"]:
                    eps = float(eps)
                    d_sq = make_denominator({"kind": "sqrt", "eps": eps})
                    r_sq = fit_one(target, lo, hi, m, n, d_sq, cfg)
                    err = error_metrics(xe, fe, eval_rational(r_sq.a, r_sq.b, xe, d_sq))

                    # --- 平滑化そのものの効果を分離して測る ---
                    # 独立にフィットした sqrt と abs を比べると、両者が別の極小解に
                    # 落ちた分の差まで混ざる（実測で max_diff が 100 を超える例が出た）。
                    # eps の効果だけを取り出すため、**abs の係数を固定したまま
                    # 分母の形だけ差し替えて**比較する。
                    y_sq_same = eval_rational(r_abs.a, r_abs.b, xx, d_sq)
                    diff = np.abs(y_sq_same - y_abs)
                    affected = diff > diff_tol
                    # その範囲を実測 pre-activation 分布で重み付けする
                    mass = float(np.interp(centers, xx, affected.astype(float)) @ prob)

                    # 参考: 独立にフィットした場合の差（解のばらつきを含む）
                    y_sq_indep = eval_rational(r_sq.a, r_sq.b, xx, d_sq)
                    diff_indep = float(np.abs(y_sq_indep - y_abs).max())

                    # 理論上の平滑化幅 sqrt(eps)/|A'(x0)|（真の折れ点があるときのみ）
                    width = float(np.sqrt(eps) / slopes.max()) if slopes.size else np.nan

                    rows.append({
                        "target": target, "interval": iname, "m": m, "n": n, "eps": eps,
                        "sup": err["sup"], "l2": err["l2"],
                        "sup_abs": err_abs["sup"], "l2_abs": err_abs["l2"],
                        "sup_ratio": err["sup"] / max(err_abs["sup"], 1e-300),
                        "n_true_kinks": int(kinks.size),
                        "kink_x": float(np.abs(kinks).min()) if kinks.size else np.nan,
                        "kink_slope": float(slopes.max()) if slopes.size else np.nan,
                        "smoothing_width_x": width,
                        "max_diff_same_coef": float(diff.max()),
                        "max_diff_indep_fit": diff_indep,
                        "affected_x_frac": float(affected.mean()),
                        "affected_preact_mass": mass,
                    })
                print(f"  {target:10s} {iname:6s} [{m},{n}] done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "eps_sweep.csv", index=False, encoding="utf-8")
    return df


def make_figures(df: pd.DataFrame, run_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = run_dir / "figures"
    figdir.mkdir(exist_ok=True)

    sub = df[df.n_true_kinks > 0]
    plot_df = sub if not sub.empty else df

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for (t, iv, m, n), g in plot_df.groupby(["target", "interval", "m", "n"]):
        g = g.sort_values("eps")
        lbl = f"{t}/{iv} [{m},{n}]"
        axes[0].loglog(g["eps"], g["sup"], "o-", markersize=4, label=lbl)
        axes[1].loglog(g["eps"], np.maximum(g["affected_preact_mass"], 1e-12), "o-",
                       markersize=4, label=lbl)
    for g_, ax, ylab, title in (
        (0, axes[0], "sup error on fit interval", "Accuracy ceiling imposed by eps"),
        (1, axes[1], "fraction of pre-activations affected",
         "How much of the real distribution the smoothing touches"),
    ):
        ax.set_xlabel("eps")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3, which="both")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "eps_tradeoff.png", dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase0_eps.yaml")
    ap.add_argument("--tag", default="eps")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = set_seed(int(cfg.get("seed", 0)))
    run_dir = new_run_dir("phase0/eps", tag=args.tag)
    save_run_meta(run_dir, config=cfg, seed=seed, include_gpu=False)

    df = run(cfg, run_dir)
    if cfg.get("figures", True):
        make_figures(df, run_dir)

    print(f"wrote {len(df)} rows -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
