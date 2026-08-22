"""Phase 0 / T5: GPT-2 の FFN pre-activation 分布を層ごとに測定する（RQ5）。

Day 2 の結果、有理活性化の問題（絶対値の発動・極・裾の破綻）はすべて
「フィット区間の外」で起きることが分かった。本スクリプトは
**モデルが実際に区間外をどれだけ通るか**を実測し、Phase 0 全体の解釈を確定させる。

あわせて素の GPT-2 の perplexity も測る（Day 5 の差し替え実験の基準値になる。
同じ順伝播で得られるので追加コストはゼロ）。

出力（``results/phase0/preact/<timestamp>_<tag>/``）:
- ``preact_stats.csv``  : 層ごとの分位点・超過率・分布スケール
- ``preact_stats.json`` : ヒストグラム込みの全量
- ``figures/``          : 層別ヒストグラム、超過率の層依存

実行:
    python scripts/run_phase0_preact.py --config configs/phase0_preact.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from src.eval.data import load_token_ids, sliding_windows
from src.eval.preact_stats import PreActCollector, gpt2_preact_modules
from src.utils.run_meta import load_yaml, new_run_dir, save_run_meta, set_seed


@torch.no_grad()
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

    hcfg = cfg["histogram"]
    collector = PreActCollector(
        n_layers=model.config.n_layer,
        device=device,
        lo=float(hcfg["lo"]),
        hi=float(hcfg["hi"]),
        n_bins=int(hcfg["n_bins"]),
        thresholds=tuple(float(t) for t in cfg["thresholds"]),
    )
    collector.attach(gpt2_preact_modules(model))

    nll_sum, n_tok = 0.0, 0
    bs = int(ev.get("batch_size", 4))
    t0 = time.perf_counter()

    for bi in range(0, len(windows), bs):
        batch = windows[bi : bi + bs]
        maxlen = max(e - b for b, e, _ in batch)
        # 窓長が揃わないのは末尾だけなので、末尾は個別に処理して padding を避ける
        if any((e - b) != maxlen for b, e, _ in batch):
            groups = [[w] for w in batch]
        else:
            groups = [batch]

        for grp in groups:
            inp = torch.stack([ids[b:e] for b, e, _ in grp]).to(device)
            out = model(inp)
            logits = out.logits.float()

            for k, (b, e, n_target) in enumerate(grp):
                # 窓の末尾 n_target トークンだけを予測対象にする
                lg = logits[k, :-1]
                tg = inp[k, 1:]
                start = max(lg.shape[0] - n_target, 0)
                lg, tg = lg[start:], tg[start:]
                nll = torch.nn.functional.cross_entropy(lg, tg, reduction="sum")
                nll_sum += float(nll.item())
                n_tok += int(tg.numel())

        if (bi // bs) % 20 == 0:
            print(f"  {bi:5d}/{len(windows)} windows  {time.perf_counter() - t0:6.1f}s", flush=True)

    collector.remove()
    stats = collector.finalize()
    ppl = math.exp(nll_sum / n_tok)
    print(f"baseline perplexity = {ppl:.4f}  ({n_tok} target tokens, "
          f"{time.perf_counter() - t0:.1f}s)", flush=True)

    rows = []
    for i, s in enumerate(stats):
        row = {
            "layer": i,
            "count": s["count"],
            "mean": s["mean"],
            "std": s["std"],
            "min": s["min"],
            "max": s["max"],
            "iqr": s["iqr"],
            "hist_out_of_range_frac": s["hist_out_of_range_frac"],
        }
        row.update({f"q{q}": v for q, v in s["quantiles"].items()})
        row.update({k: v for k, v in s.items() if k.startswith("frac_abs_gt_")})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "preact_stats.csv", index=False, encoding="utf-8")

    payload = {
        "model": cfg["model"],
        "dataset": cfg["dataset"],
        "n_tokens": int(ids.numel()),
        "n_windows": len(windows),
        "baseline_perplexity": ppl,
        "n_target_tokens": n_tok,
        "layers": stats,
    }
    (run_dir / "preact_stats.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return df


def make_figures(cfg: dict, run_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = run_dir / "figures"
    figdir.mkdir(exist_ok=True)
    payload = json.loads((run_dir / "preact_stats.json").read_text(encoding="utf-8"))
    layers = payload["layers"]
    df = pd.read_csv(run_dir / "preact_stats.csv")

    # --- 図1: 層別ヒストグラム（対数スケール）とフィット区間の重ね描き ---
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("viridis")
    for i, s in enumerate(layers):
        c = np.array(s["_centers"])
        h = np.array(s["_hist"])
        h = h / max(h.sum(), 1)
        sel = (c >= -30) & (c <= 30)
        ax.semilogy(c[sel], np.maximum(h[sel], 1e-12), color=cmap(i / max(len(layers) - 1, 1)),
                    lw=1.0, label=f"layer {i}" if i in (0, len(layers) - 1) else None)
    ax.axvspan(-3, 3, color="tab:red", alpha=0.15, label="PAU/RAFT fit range [-3,3]")
    ax.axvspan(-6, 6, color="tab:blue", alpha=0.08, label="wide fit range [-6,6]")
    ax.set_xlabel("pre-activation value")
    ax.set_ylabel("normalized density (log)")
    ax.set_title("GPT-2 FFN pre-activation distribution by layer")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "preact_hist_by_layer.png", dpi=140)
    plt.close(fig)

    # --- 図2: 区間外に出る割合の層依存 ---
    frac_cols = [c for c in df.columns if c.startswith("frac_abs_gt_")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c in frac_cols:
        ax.semilogy(df["layer"], np.maximum(df[c], 1e-12), "o-", markersize=4,
                    label=f"P(|x| > {c.rsplit('_', 1)[-1]})")
    ax.set_xlabel("layer")
    ax.set_ylabel("fraction of pre-activations")
    ax.set_title("Fraction of pre-activations outside the fit interval")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(figdir / "preact_exceedance_by_layer.png", dpi=140)
    plt.close(fig)

    # --- 図3: 分位点レンジの層依存 ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(df["layer"], df["q0.001"], df["q0.999"], alpha=0.25, label="0.1%-99.9%")
    ax.fill_between(df["layer"], df["q0.01"], df["q0.99"], alpha=0.35, label="1%-99%")
    ax.fill_between(df["layer"], df["q0.25"], df["q0.75"], alpha=0.5, label="IQR")
    ax.plot(df["layer"], df["min"], "v--", color="k", markersize=4, lw=0.8, label="min/max")
    ax.plot(df["layer"], df["max"], "^--", color="k", markersize=4, lw=0.8)
    ax.axhline(3, color="tab:red", ls=":", label="+/-3 (PAU/RAFT)")
    ax.axhline(-3, color="tab:red", ls=":")
    ax.axhline(6, color="tab:blue", ls=":", label="+/-6")
    ax.axhline(-6, color="tab:blue", ls=":")
    ax.set_xlabel("layer")
    ax.set_ylabel("pre-activation value")
    ax.set_title("Pre-activation range by layer")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "preact_range_by_layer.png", dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase0_preact.yaml")
    ap.add_argument("--tag", default="preact")
    ap.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="設定ファイルの値を上書きする。パイプライン疎通確認（最小構成）用。",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.max_windows is not None:
        cfg["eval"]["max_windows"] = args.max_windows
        cfg["_smoke_test"] = True  # メタ情報に「本番ではない」と残す
    seed = set_seed(int(cfg.get("seed", 0)))
    run_dir = new_run_dir("phase0/preact", tag=args.tag)
    save_run_meta(run_dir, config=cfg, seed=seed, include_gpu=True)

    df = run(cfg, run_dir)
    if cfg.get("figures", True):
        make_figures(cfg, run_dir)

    print(f"wrote {len(df)} layers -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
