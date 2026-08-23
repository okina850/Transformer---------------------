r"""Phase 1: 全条件の結果を集約し、Pre-registration の判定基準を適用する。

**基準は結果を見てから変更しない**（Pre-registration の約束）。
ただし「基準の判定」と「効果量」は必ず分けて報告する
（R2 で seed 間ばらつきが小さすぎて 0.07% の差でも非重複になる事例が出たため）。

出力: ``results/phase1/analysis/``
- ``summary.csv``       : 条件×seed の最終指標
- ``comparisons.csv``   : 全ペアの判定と効果量
- ``figures/``          : 学習曲線、勾配ノルム、符号変化の推移

実行:
    python scripts/analyze_phase1.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.train.diagnostics import ranges_overlap
from src.utils.run_meta import REPO_ROOT

#: 表示順と、条件名 -> (ラベル, 実行ディレクトリの glob パターン)
CONDITIONS = [
    ("B1", "GELU", "*gelu_stage1_seed*"),
    ("B2", "SwiGLU", "*swiglu_stage2_seed*"),
    ("R1", "abs x [-3,3]", "*R1_stage3_seed*"),
    ("R2", "abs x [-6,6]", "*R2_stage3_seed*"),
    ("R3", "sqrt x [-3,3]", "*R3_stage3_seed*"),
    ("R4", "sqrt x [-6,6]", "*R4_stage3_seed*"),
]


def load_runs() -> pd.DataFrame:
    rows = []
    base = REPO_ROOT / "results/phase1/train"
    for cond, label, pattern in CONDITIONS:
        for d in sorted(base.glob(pattern)):
            summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            evals = [json.loads(l) for l in (d / "eval_log.jsonl").read_text(encoding="utf-8").splitlines()]
            steps = [json.loads(l) for l in (d / "step_log.jsonl").read_text(encoding="utf-8").splitlines()]
            gn = [s["grad_norm_pre_clip"] for s in steps]
            last = evals[-1]
            rows.append({
                "condition": cond, "label": label, "seed": summary["seed"],
                "completed": summary["stop_reason"] == "completed",
                "stop_reason": summary["stop_reason"],
                "final_step": summary["final_step"],
                "val_ppl": last["val_ppl"], "val_loss": last["val_loss"],
                "n_loss_spikes": summary["n_loss_spikes"],
                "grad_norm_median": statistics.median(gn),
                "grad_norm_max": max(gn),
                "elapsed_sec": summary["elapsed_sec"],
                "run_dir": str(d.relative_to(REPO_ROOT)),
            })
    return pd.DataFrame(rows)


def compare(df: pd.DataFrame) -> pd.DataFrame:
    """完走した run のみで val_ppl の全ペア比較を行う。"""
    vals = {}
    for cond, _, _ in CONDITIONS:
        sub = df[(df.condition == cond) & df.completed]
        if len(sub):
            vals[cond] = sub["val_ppl"].tolist()

    rows = []
    for a, b in combinations(vals, 2):
        va, vb = vals[a], vals[b]
        overlap = ranges_overlap(va, vb)
        # 効果量: レンジ間の隙間（重なっていれば 0）と、中央値差の相対値
        gap = 0.0 if overlap else max(min(va), min(vb)) - min(max(va), max(vb))
        med_a, med_b = statistics.median(va), statistics.median(vb)
        rows.append({
            "cond_a": a, "cond_b": b,
            "n_seed_a": len(va), "n_seed_b": len(vb),
            "range_a": f"[{min(va):.4f}, {max(va):.4f}]",
            "range_b": f"[{min(vb):.4f}, {max(vb):.4f}]",
            "verdict": "差があるとは言えない" if overlap else "差がある",
            "gap_ppl": round(gap, 5),
            "gap_relative_pct": round(gap / min(med_a, med_b) * 100, 4),
            "median_diff_pct": round(abs(med_a - med_b) / min(med_a, med_b) * 100, 4),
        })
    return pd.DataFrame(rows)


def make_figures(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = out_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    base = REPO_ROOT / "results/phase1/train"
    colors = plt.get_cmap("tab10")

    # --- 図1: 学習曲線（val_ppl vs step） ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (cond, label, pattern) in enumerate(CONDITIONS):
        for j, d in enumerate(sorted(base.glob(pattern))):
            evals = [json.loads(l) for l in (d / "eval_log.jsonl").read_text(encoding="utf-8").splitlines()]
            xs = [e["step"] for e in evals]
            ys = [e["val_ppl"] for e in evals]
            ax.plot(xs, ys, color=colors(i), alpha=0.75, lw=1.2,
                    label=f"{cond} ({label})" if j == 0 else None)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("validation perplexity (log)")
    ax.set_title("Phase 1: learning curves (3 seeds per condition)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(figdir / "learning_curves.png", dpi=140)
    plt.close(fig)

    # --- 図2: 最終 val_ppl の条件別分布 ---
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, (cond, label, _) in enumerate(CONDITIONS):
        sub = df[(df.condition == cond) & df.completed]
        ax.scatter([i] * len(sub), sub["val_ppl"], s=70, color=colors(i), zorder=3)
        if len(sub):
            ax.plot([i - 0.22, i + 0.22], [sub["val_ppl"].min()] * 2, color=colors(i), lw=1.5)
            ax.plot([i - 0.22, i + 0.22], [sub["val_ppl"].max()] * 2, color=colors(i), lw=1.5)
        diverged = df[(df.condition == cond) & ~df.completed]
        if len(diverged):
            ax.scatter([i] * len(diverged), [df[df.completed]["val_ppl"].max() + 0.02] * len(diverged),
                       marker="x", s=90, color="red", zorder=4)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([f"{c}\n{l}" for c, l, _ in CONDITIONS], fontsize=8)
    ax.set_ylabel("final validation perplexity")
    ax.set_title("Final val perplexity per seed (red x = diverged run, excluded)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(figdir / "final_ppl_by_condition.png", dpi=140)
    plt.close(fig)

    # --- 図3: 勾配ノルム（クリップ前）の推移、中央値の移動窓 ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (cond, label, pattern) in enumerate(CONDITIONS):
        for j, d in enumerate(sorted(base.glob(pattern))):
            steps = [json.loads(l) for l in (d / "step_log.jsonl").read_text(encoding="utf-8").splitlines()]
            gn = np.array([s["grad_norm_pre_clip"] for s in steps])
            w = 200
            if len(gn) >= w:
                med = np.array([np.median(gn[k : k + w]) for k in range(0, len(gn) - w, w)])
                xs = np.arange(len(med)) * w
                ax.plot(xs, med, color=colors(i), alpha=0.75, lw=1.2,
                        label=f"{cond} ({label})" if j == 0 else None)
    ax.set_xlabel("step")
    ax.set_ylabel("pre-clip grad norm (median over 200-step window)")
    ax.set_title("Pre-clip gradient norm (the diagnostic clipping would have hidden)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "grad_norm_trajectory.png", dpi=140)
    plt.close(fig)

    # --- 図4: A(x) 符号変化数の推移（rational 条件のみ、RQ3） ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, (cond, label, pattern) in zip(axes.flat, CONDITIONS[2:]):
        for j, d in enumerate(sorted(base.glob(pattern))):
            evals = [json.loads(l) for l in (d / "eval_log.jsonl").read_text(encoding="utf-8").splitlines()]
            xs = [e["step"] for e in evals]
            sc = np.array([e["sign_changes"] for e in evals])  # (n_eval, n_layer)
            for layer in range(sc.shape[1]):
                ax.plot(xs, sc[:, layer], alpha=0.55, lw=1.0,
                        color=plt.get_cmap("viridis")(layer / max(sc.shape[1] - 1, 1)))
        ax.set_title(f"{cond} ({label})", fontsize=10)
        ax.grid(True, alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("step")
    for ax in axes[:, 0]:
        ax.set_ylabel("sign changes of A(x)")
    fig.suptitle("RQ3: sign changes of A(x) per layer during training (dark=layer0, bright=layer5)", fontsize=11)
    fig.tight_layout()
    fig.savefig(figdir / "sign_changes_rq3.png", dpi=140)
    plt.close(fig)

    # --- 図5: pre-activation 分位点の推移（RQ5、裾がいつ育つか） ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (cond, label, pattern) in enumerate(CONDITIONS):
        d = sorted(base.glob(pattern))[0]  # seed0 のみ
        evals = [json.loads(l) for l in (d / "eval_log.jsonl").read_text(encoding="utf-8").splitlines()]
        # B2 (SwiGLU) は preact_probe が w_gate を見ておらず空になっている
        # （実装の取りこぼし。修正済みだが本 run は再取得できないので飛ばす）
        evals = [e for e in evals if e.get("preact_quantiles")]
        if not evals:
            continue
        xs = [e["step"] for e in evals]
        # 全層の最大絶対値
        ymax = [max(max(abs(q["min"]), abs(q["max"])) for q in e["preact_quantiles"]) for e in evals]
        y999 = [max(abs(q["q0.999"]) for q in e["preact_quantiles"]) for e in evals]
        ax.plot(xs, ymax, color=colors(i), lw=1.4, label=f"{cond} max|x|")
        ax.plot(xs, y999, color=colors(i), lw=1.0, ls="--", alpha=0.6)
    ax.axhline(3, color="tab:red", ls=":", lw=1.5, label="|x|=3 (PAU/RAFT fit range)")
    ax.axhline(6, color="tab:blue", ls=":", lw=1.5, label="|x|=6")
    ax.set_xlabel("step")
    ax.set_ylabel("pre-activation magnitude")
    ax.set_title("RQ5: do the heavy tails grow during training? (solid=max, dashed=99.9%, seed0)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "preact_tails_rq5.png", dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phase1/analysis")
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_runs()
    df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8")

    cmp_df = compare(df)
    cmp_df.to_csv(out_dir / "comparisons.csv", index=False, encoding="utf-8")

    make_figures(df, out_dir)

    pd.set_option("display.width", 250)
    print("=== 条件別サマリ ===")
    agg = df.groupby(["condition", "label"]).agg(
        n_runs=("seed", "count"),
        n_completed=("completed", "sum"),
        ppl_min=("val_ppl", lambda s: round(s[df.loc[s.index, "completed"]].min(), 4) if df.loc[s.index, "completed"].any() else float("nan")),
        ppl_max=("val_ppl", lambda s: round(s[df.loc[s.index, "completed"]].max(), 4) if df.loc[s.index, "completed"].any() else float("nan")),
        spikes=("n_loss_spikes", "sum"),
        gn_median=("grad_norm_median", lambda s: round(s.median(), 3)),
        gn_max=("grad_norm_max", "max"),
    ).reset_index()
    print(agg.to_string(index=False))
    print()
    print("=== 全ペア比較（完走 run のみ） ===")
    print(cmp_df[["cond_a", "cond_b", "range_a", "range_b", "verdict", "gap_relative_pct", "median_diff_pct"]].to_string(index=False))
    print(f"\nwrote -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
