r"""Phase 0 / Day 5: GPT-2 の FFN 活性化を有理近似に差し替え、perplexity 劣化を測る。

**学習は一切行わない。** GELU 用に学習済みの重みへ、活性化関数だけを
差し替えたときの validation perplexity を見る（research_context §6 Phase 0 の項目3・4）。

差し替え対象の目標関数は必ず ``gelu_tanh``（HF の ``gelu_new`` と同一の tanh 近似）に
揃えてある。GPT-2 は ``gelu_new`` を使うので、これがずれると差し替え時点で
「近似誤差」と「erf 版 vs tanh 版のずれ」が混ざってしまう。

出力（``results/phase0/ppl/<timestamp>_<tag>/``）:
- ``ppl_results.csv`` : 条件ごとの perplexity・劣化率
- ``figures/``        : 近似誤差 vs perplexity 劣化の関係

実行:
    python scripts/run_phase0_ppl.py --config configs/phase0_ppl.yaml
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from src.eval.data import load_token_ids, sliding_windows
from src.eval.swap import make_rational_factory, swapped_activation
from src.utils.run_meta import REPO_ROOT, load_yaml, new_run_dir, save_run_meta, set_seed


def load_coefficients(path: str | None) -> dict:
    if path is None:
        cands = sorted(glob.glob(str(REPO_ROOT / "results/phase0/fit/*/coefficients.json")))
        if not cands:
            raise FileNotFoundError("先に scripts/run_phase0_fit.py を実行すること。")
        path = cands[-1]
    return json.loads(Path(path).read_text(encoding="utf-8")), path


@torch.no_grad()
def eval_perplexity(model, ids: torch.Tensor, windows, device, batch_size: int) -> tuple[float, int]:
    nll_sum, n_tok = 0.0, 0
    for bi in range(0, len(windows), batch_size):
        batch = windows[bi : bi + batch_size]
        maxlen = max(e - b for b, e, _ in batch)
        groups = [batch] if all((e - b) == maxlen for b, e, _ in batch) else [[w] for w in batch]
        for grp in groups:
            inp = torch.stack([ids[b:e] for b, e, _ in grp]).to(device)
            logits = model(inp).logits.float()
            for k, (b, e, n_target) in enumerate(grp):
                lg, tg = logits[k, :-1], inp[k, 1:]
                start = max(lg.shape[0] - n_target, 0)
                lg, tg = lg[start:], tg[start:]
                nll = torch.nn.functional.cross_entropy(lg, tg, reduction="sum")
                nll_sum += float(nll.item())
                n_tok += int(tg.numel())
    return math.exp(nll_sum / n_tok), n_tok


def run(cfg: dict, run_dir: Path) -> pd.DataFrame:
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, cfg["eval"].get("dtype", "float32"))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=dtype).to(device).eval()

    ids = load_token_ids(tok, **cfg["dataset"])
    ev = cfg["eval"]
    windows = sliding_windows(ids, ev["n_ctx"], ev["stride"], ev.get("max_windows"))
    print(f"tokens={ids.numel()}  windows={len(windows)}  device={device}", flush=True)

    coeffs, coeff_path = load_coefficients(cfg.get("coefficients"))
    print(f"coefficients: {coeff_path}", flush=True)

    rows: list[dict] = []
    t0 = time.perf_counter()

    for cond in cfg["conditions"]:
        name, kind = cond["name"], cond["kind"]

        if kind == "original":
            ppl, n_tok = eval_perplexity(model, ids, windows, device, ev["batch_size"])
            rows.append({"condition": name, "kind": kind, "ppl": ppl, "n_target_tokens": n_tok})

        elif kind == "rational":
            key = cond["key"]
            if key not in coeffs:
                rows.append({"condition": name, "kind": kind, "key": key, "status": "MISSING_KEY"})
                continue
            c = coeffs[key]
            eps = float(cond.get("eps", c.get("denom_params", {}).get("eps", 1e-3)))
            denom = c.get("denom", "abs")
            if c.get("note"):  # pade@0 の保存形式は denom=abs 固定
                denom = "abs"

            factory = make_rational_factory(int(c["m"]), int(c["n"]), denom, eps, c["a"], c["b"])
            with swapped_activation(model, factory):
                ppl, n_tok = eval_perplexity(model, ids, windows, device, ev["batch_size"])

            rows.append({
                "condition": name, "kind": kind, "key": key,
                "target": c["target"], "interval": c["interval"],
                "lo": c["lo"], "hi": c["hi"], "m": c["m"], "n": c["n"],
                "denom": denom, "n_params": c["n_params"],
                "fit_sup_error": c.get("sup"), "fit_l2_error": c.get("l2"),
                "ppl": ppl, "n_target_tokens": n_tok,
            })
        else:
            raise ValueError(f"unknown condition kind: {kind}")

        print(f"  {name:28s} ppl={rows[-1].get('ppl', float('nan')):.4f}  "
              f"({time.perf_counter() - t0:6.1f}s elapsed)", flush=True)

    df = pd.DataFrame(rows)
    baseline = df.loc[df["condition"] == "baseline", "ppl"]
    if not baseline.empty:
        base = float(baseline.iloc[0])
        df["ppl_ratio_to_baseline"] = df["ppl"] / base
        df["ppl_delta_pct"] = (df["ppl"] / base - 1.0) * 100.0
    df.to_csv(run_dir / "ppl_results.csv", index=False, encoding="utf-8")
    return df


def make_figure(df: pd.DataFrame, run_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = run_dir / "figures"
    figdir.mkdir(exist_ok=True)
    sub = df.dropna(subset=["fit_sup_error", "ppl"])
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for _, row in sub.iterrows():
        ax.loglog(row["fit_sup_error"], row["ppl"], "o", markersize=9)
        ax.annotate(row["condition"], (row["fit_sup_error"], row["ppl"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    base = df.loc[df["condition"] == "baseline", "ppl"]
    if not base.empty:
        ax.axhline(float(base.iloc[0]), color="k", ls="--", lw=1, label="baseline (unmodified GPT-2)")
    ax.set_xlabel("approximation sup error (on fit interval)")
    ax.set_ylabel("validation perplexity")
    ax.set_title("Approximation error vs. perplexity degradation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(figdir / "error_vs_ppl.png", dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase0_ppl.yaml")
    ap.add_argument("--tag", default="ppl")
    ap.add_argument(
        "--max-windows", type=int, default=None,
        help="疎通確認用に窓数を制限する（最小構成での動作確認）。",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.max_windows is not None:
        cfg["eval"]["max_windows"] = args.max_windows
    seed = set_seed(int(cfg.get("seed", 0)))
    run_dir = new_run_dir("phase0/ppl", tag=args.tag)
    save_run_meta(run_dir, config=cfg, seed=seed, include_gpu=True)

    df = run(cfg, run_dir)
    make_figure(df, run_dir)

    print()
    print(df.to_string(index=False))
    print(f"\nwrote {len(df)} rows -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
