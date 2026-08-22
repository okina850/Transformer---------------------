r"""Phase 0 / Day 3: torch 実装の数値パリティ試験と、float32 での条件数評価（T3・T4）。

3つを測る。

1. **パリティ**: torch(float64) 実装が numpy(float64) 参照実装と一致するか。
   ここがずれていたら Phase 1 の結果はすべて疑わしい。
2. **float32 での精度劣化**: Phase 1 は GPU の float32 で回る。
   モノミアル基底は次数と区間幅に対して急速に悪条件になるので、
   **「高次数・広区間で不安定」が最適化の問題か浮動小数点の問題か**を
   切り分けるための基準値を先に取っておく。
3. **平滑性の定量化（T3）**: :math:`F''` の大きさと1階微分の全変動を分母設計間で比較する。

**Phase 1 の R1〜R4 に対応する条件の Vandermonde 条件数を明示的に記録する**
（区間軸の比較は近似の質だけでなく数値条件も同時に動かすため）。

実行:
    python scripts/run_phase0_parity.py --config configs/phase0_parity.yaml
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
import torch

from src.rational.activation import RationalActivation
from src.rational.denominators import eval_rational, make_denominator
from src.rational.metrics import eval_grid, smoothness_metrics
from src.rational.poly import horner, polyval_naive, vander_cond
from src.utils.run_meta import REPO_ROOT, load_yaml, new_run_dir, save_run_meta, set_seed


def load_coefficients(path: str | None) -> dict:
    if path is None:
        cands = sorted(glob.glob(str(REPO_ROOT / "results/phase0/fit/*/coefficients.json")))
        if not cands:
            raise FileNotFoundError("先に scripts/run_phase0_fit.py を実行すること。")
        path = cands[-1]
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(cfg: dict, run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    coeffs = load_coefficients(cfg.get("coefficients"))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []

    for key, c in coeffs.items():
        if c.get("note") or c["denom"] not in ("abs", "sqrt", "square", "free"):
            continue
        if cfg.get("only_targets") and c["target"] not in cfg["only_targets"]:
            continue

        m, n, lo, hi = int(c["m"]), int(c["n"]), float(c["lo"]), float(c["hi"])
        a, b = np.array(c["a"]), np.array(c["b"])
        eps = float(c.get("denom_params", {}).get("eps", 1e-3))
        d = make_denominator({"kind": c["denom"], "eps": eps})

        # 評価点は近似区間と、その外側（外挿域）の両方
        x_in = eval_grid(lo, hi, int(cfg["n_eval"]))
        ext = cfg["extrapolation"]
        x_ext = np.linspace(float(ext["lo"]), float(ext["hi"]), int(ext["n"]))

        for label, x in (("in_interval", x_in), ("extrapolation", x_ext)):
            y_np = eval_rational(a, b, x, d)  # numpy float64 参照

            mod64 = RationalActivation(m, n, c["denom"], eps, a, b, dtype=torch.float64).to(device)
            mod32 = RationalActivation(m, n, c["denom"], eps, a, b, dtype=torch.float32).to(device)
            xt64 = torch.as_tensor(x, dtype=torch.float64, device=device)
            xt32 = torch.as_tensor(x, dtype=torch.float32, device=device)
            with torch.no_grad():
                y64 = mod64(xt64).cpu().numpy()
                y32 = mod32(xt32).cpu().numpy().astype(np.float64)

            scale = max(float(np.max(np.abs(y_np))), 1e-300)
            rows.append({
                "key": key, "target": c["target"], "interval": c["interval"],
                "m": m, "n": n, "denom": c["denom"], "region": label,
                "n_params": int(c["n_params"]),
                "cond_vander_fit": vander_cond(x_in, m),
                "max_abs_err_torch64_vs_numpy": float(np.max(np.abs(y64 - y_np))),
                "max_rel_err_torch64_vs_numpy": float(np.max(np.abs(y64 - y_np))) / scale,
                "max_abs_err_torch32_vs_numpy": float(np.max(np.abs(y32 - y_np))),
                "max_rel_err_torch32_vs_numpy": float(np.max(np.abs(y32 - y_np))) / scale,
            })

        # Horner 法 vs 素朴評価（float32 での桁落ちの寄与を分離する）
        #
        # 注意: src/rational/poly.py の horner / polyval_naive は
        # **意図的に内部で float64 にキャストする**（float64 の参照実装だから）。
        # したがって float32 の比較にはそのまま使えない。torch で本当に
        # float32 のまま評価して比べる。
        p_ref = horner(a, x_in)
        pscale = max(float(np.max(np.abs(p_ref))), 1e-300)
        at32 = torch.as_tensor(a, dtype=torch.float32, device=device)
        xt = torch.as_tensor(x_in, dtype=torch.float32, device=device)

        ph = torch.full_like(xt, 0.0) + at32[-1]
        for i in range(at32.numel() - 2, -1, -1):
            ph = ph * xt + at32[i]
        pn = torch.zeros_like(xt)
        for j in range(at32.numel()):
            pn = pn + at32[j] * xt.pow(j)

        rows[-1]["horner_rel_err_f32"] = (
            float(np.max(np.abs(ph.cpu().numpy().astype(np.float64) - p_ref))) / pscale
        )
        rows[-1]["naive_rel_err_f32"] = (
            float(np.max(np.abs(pn.cpu().numpy().astype(np.float64) - p_ref))) / pscale
        )

        # T3: 平滑性
        sm = smoothness_metrics(x_ext, eval_rational(a, b, x_ext, d))
        kinks = d.kinks(b, float(ext["lo"]), float(ext["hi"])) if c["denom"] == "abs" else np.array([])
        rows[-1].update({
            "max_abs_d2_ext": sm["max_abs_d2"],
            "tv_d1_ext": sm["tv_d1"],
            "n_true_kinks_ext": int(kinks.size),
        })

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "parity.csv", index=False, encoding="utf-8")

    # --- Phase 1 の R1〜R4 に対応する条件を明示的に抜き出す ---
    p1 = []
    for cond, denom, iname in cfg["phase1_conditions"]:
        key = f"{cfg['phase1_target']}|{iname}|m{cfg['phase1_m']}n{cfg['phase1_n']}|{denom}"
        c = coeffs.get(key)
        if c is None:
            p1.append({"condition": cond, "key": key, "status": "MISSING"})
            continue
        x = eval_grid(float(c["lo"]), float(c["hi"]), int(cfg["n_eval"]))
        sub = df[(df.key == key) & (df.region == "in_interval")]
        p1.append({
            "condition": cond,
            "denom": denom,
            "interval": iname,
            "lo": c["lo"], "hi": c["hi"],
            "m": c["m"], "n": c["n"], "n_params": c["n_params"],
            "sup_error": c["sup"], "l2_error": c["l2"],
            "cond_vander": vander_cond(x, int(c["m"])),
            "max_rel_err_f32": float(sub["max_rel_err_torch32_vs_numpy"].iloc[0]) if len(sub) else np.nan,
        })
    df_p1 = pd.DataFrame(p1)
    df_p1.to_csv(run_dir / "phase1_conditions.csv", index=False, encoding="utf-8")
    return df, df_p1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase0_parity.yaml")
    ap.add_argument("--tag", default="parity")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = set_seed(int(cfg.get("seed", 0)))
    run_dir = new_run_dir("phase0/parity", tag=args.tag)
    save_run_meta(run_dir, config=cfg, seed=seed, include_gpu=True)

    df, df_p1 = run(cfg, run_dir)
    print(f"wrote {len(df)} rows -> {run_dir}")
    print()
    print("=== Phase 1 の R1〜R4 に対応する条件 ===")
    print(df_p1.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
