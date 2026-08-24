r"""README 用の要約図（3パネル）を作る。

**図の作り方の方針**: 主結果は「条件間に意味のある差はない」という否定的結果である。
y 軸を差が見える範囲まで拡大すると、実際には無視できる差が大きく見えてしまう。
そこで左パネルは**拡大表示に「全条件の広がりは 1.7%」という注記を必ず添える**。

実行:
    python scripts/make_summary_figure.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.run_meta import REPO_ROOT

CONDITIONS = [
    ("B1", "GELU", "*gelu_stage1_seed*", "tab:blue"),
    ("B2", "SwiGLU", "*swiglu_stage2_seed*", "tab:orange"),
    ("R1", "abs [-3,3]", "*R1_stage3_seed*", "tab:green"),
    ("R2", "abs [-6,6]", "*R2_stage3_seed*", "tab:red"),
    ("R3", "sqrt [-3,3]", "*R3_stage3_seed*", "tab:purple"),
    ("R4", "sqrt [-6,6]", "*R4_stage3_seed*", "tab:brown"),
]

BASE = REPO_ROOT / "results/phase1/train"
COMMON_WINDOW = 2249  # 全条件が到達しているステップ数（R1 seed2 の発散地点）


def load():
    data = {}
    for cond, label, pattern, color in CONDITIONS:
        runs = []
        for d in sorted(BASE.glob(pattern)):
            summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            evals = [json.loads(l) for l in (d / "eval_log.jsonl").read_text(encoding="utf-8").splitlines()]
            steps = [json.loads(l) for l in (d / "step_log.jsonl").read_text(encoding="utf-8").splitlines()]
            runs.append({
                "summary": summary, "evals": evals,
                "gn": [s["grad_norm_pre_clip"] for s in steps],
            })
        data[cond] = {"label": label, "color": color, "runs": runs}
    return data


def main() -> int:
    data = load()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ---------- パネル1: 最終 val perplexity ----------
    ax = axes[0]
    all_ppl = []
    for i, (cond, label, _, color) in enumerate(CONDITIONS):
        ppls = [r["evals"][-1]["val_ppl"] for r in data[cond]["runs"]
                if r["summary"]["stop_reason"] == "completed"]
        all_ppl.extend(ppls)
        ax.scatter([i] * len(ppls), ppls, s=80, color=color, zorder=3)
        if ppls:
            ax.plot([i - 0.25, i + 0.25], [min(ppls)] * 2, color=color, lw=2)
            ax.plot([i - 0.25, i + 0.25], [max(ppls)] * 2, color=color, lw=2)
            ax.plot([i, i], [min(ppls), max(ppls)], color=color, lw=1, alpha=0.5)
        n_div = sum(1 for r in data[cond]["runs"] if r["summary"]["stop_reason"] != "completed")
        if n_div:
            ax.annotate(f"{n_div}/3 diverged\n(excluded)", (i, max(all_ppl) + 0.012),
                        ha="center", fontsize=8, color="red", fontweight="bold")

    b1 = [r["evals"][-1]["val_ppl"] for r in data["B1"]["runs"]]
    ax.axhspan(min(b1), max(b1), color="tab:blue", alpha=0.10, zorder=0)
    spread = (max(all_ppl) - min(all_ppl)) / min(all_ppl) * 100
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([f"{c}\n{l}" for c, l, _, _ in CONDITIONS], fontsize=8)
    ax.set_ylabel("final validation perplexity")
    ax.set_title("RQ1: no meaningful difference between conditions", fontsize=10)
    ax.text(0.02, 0.03,
            f"NOTE: y-axis is zoomed.\nTotal spread across all 6 conditions: {spread:.1f}%\n"
            f"Blue band = GELU baseline range",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))
    ax.grid(True, alpha=0.3, axis="y")

    # ---------- パネル2: 勾配ノルム ----------
    ax = axes[1]
    for i, (cond, label, _, color) in enumerate(CONDITIONS):
        meds = [statistics.median(r["gn"][:COMMON_WINDOW]) for r in data[cond]["runs"]]
        ax.scatter([i] * len(meds), meds, s=80, color=color, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [min(meds)] * 2, color=color, lw=2)
        ax.plot([i - 0.25, i + 0.25], [max(meds)] * 2, color=color, lw=2)
        ax.plot([i, i], [min(meds), max(meds)], color=color, lw=1, alpha=0.5)

    base_med = statistics.median(
        [statistics.median(r["gn"][:COMMON_WINDOW]) for c in ("B1", "B2") for r in data[c]["runs"]]
    )
    ax.axhline(base_med, color="k", ls="--", lw=1.2)
    ax.annotate("baseline level", (5.3, base_med), fontsize=8, va="bottom", ha="right")
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([f"{c}\n{l}" for c, l, _, _ in CONDITIONS], fontsize=8)
    ax.set_ylabel("pre-clip gradient norm (median)")
    ax.set_ylim(0, None)
    ax.set_title("FINDING: rational activations produce ~2-2.5x\nlarger gradients, regardless of denominator", fontsize=10)
    ax.text(0.02, 0.03, f"measured over a common window\n(first {COMMON_WINDOW} steps, all runs reached it)",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))
    ax.grid(True, alpha=0.3, axis="y")

    # ---------- パネル3: pre-activation の裾の成長 ----------
    ax = axes[2]
    for cond, label, _, color in CONDITIONS:
        runs = data[cond]["runs"]
        evals = [e for e in runs[0]["evals"] if e.get("preact_quantiles")]
        if not evals:
            continue  # B2 は記録漏れ
        xs = [e["step"] for e in evals]
        y999 = [max(abs(q["q0.999"]) for q in e["preact_quantiles"]) for e in evals]
        ax.plot(xs, y999, color=color, lw=1.6, label=f"{cond} ({label})")
    ax.axhline(3, color="tab:red", ls=":", lw=2)
    ax.annotate("|x| = 3  (PAU / RAFT fit interval)", (200, 3.12), fontsize=8, color="tab:red")
    ax.axhline(6, color="tab:blue", ls=":", lw=2)
    ax.annotate("|x| = 6", (200, 6.12), fontsize=8, color="tab:blue")
    ax.set_xlabel("training step")
    ax.set_ylabel("99.9th percentile of |pre-activation|")
    ax.set_title("FINDING: tails grow during training and\nreach the edge of [-6,6]", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = REPO_ROOT / "results/phase1/analysis/figures/summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote -> {out}")
    print(f"total val_ppl spread across conditions: {spread:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
