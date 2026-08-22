r"""Phase 1 / Stage 1: B1 (GELU) で lr_theta を探索し、全条件共通の値を決める。

承認済み方針: 絶対値ではなく **RAFT との比（7.14）とスケジュール形**を守り、
`lr_theta` の絶対値だけを B1 の探索で決める（研究の主軸である追試の
公平性を保つため。バッチサイズが RAFT の 1/64 なので絶対値をそのまま
使うのは危険 — RAFT 自身がバッチを半分にしただけで不安定化したと報告している）。

**選定基準**（探索前に固定）:
1. 発散しなかった候補の中から、探索終了時点の val_loss が最小のものを選ぶ
2. 全候補が発散した場合は最も長く安定した（発散ステップが遅い）ものを選び、
   さらに低い候補を追加探索する
3. val_loss が僅差（相対1%未満）で並んだ場合は、収束が速い方（＝高い方の lr）を選ぶ
   （本番の総ステップ数は固定なので、同じ最終到達点なら速く着く方が良い）

探索は本番より大幅に短いステップ数で行う（発散の有無と傾向を見るだけで十分）。

実行:
    python scripts/lr_search.py --config configs/phase1_production.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts.train import build_model
from src.train.trainer import train_one_run
from src.utils.run_meta import load_yaml, new_run_dir, save_run_meta, set_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1_production.yaml")
    ap.add_argument("--candidates", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    ap.add_argument("--search-steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base_cfg = load_yaml(args.config)
    base_cfg["ffn_kind"] = "gelu"  # B1 で探索する（承認済み方針）
    device = torch.device(base_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    results = []
    for lr in args.candidates:
        cfg = json.loads(json.dumps(base_cfg))  # deep copy
        cfg["train"]["total_steps"] = args.search_steps
        cfg["train"]["eval_interval"] = max(args.search_steps // 10, 50)
        cfg["train"]["eval_batches"] = 10
        cfg["train"]["lr_theta"] = lr

        seed = set_seed(args.seed)
        model = build_model(cfg)
        run_dir = new_run_dir("phase1/lr_search", tag=f"lr{lr:g}")
        save_run_meta(run_dir, config=cfg, seed=seed, include_gpu=True)

        print(f"\n=== lr_theta = {lr:g} ===", flush=True)
        summary = train_one_run(model, cfg, run_dir, device, seed)

        eval_log = [json.loads(l) for l in (run_dir / "eval_log.jsonl").read_text(encoding="utf-8").splitlines()]
        final_val_loss = eval_log[-1]["val_loss"] if eval_log else float("inf")

        results.append({
            "lr_theta": lr, "final_val_loss": final_val_loss,
            "diverged": summary["diverged"], "diverged_step": summary.get("diverged_step"),
            "final_step": summary["final_step"], "run_dir": str(run_dir),
        })
        del model
        torch.cuda.empty_cache()

    # --- 選定 ---
    stable = [r for r in results if not r["diverged"]]
    if stable:
        best = min(stable, key=lambda r: r["final_val_loss"])
        close = [r for r in stable if r["final_val_loss"] <= best["final_val_loss"] * 1.01]
        chosen = max(close, key=lambda r: r["lr_theta"])
    else:
        chosen = max(results, key=lambda r: r["diverged_step"] or 0)

    print("\n=== lr_theta search results ===")
    for r in results:
        flag = "DIVERGED" if r["diverged"] else "ok"
        print(f"  lr={r['lr_theta']:.1e}  final_val_loss={r['final_val_loss']:.4f}  "
              f"[{flag}]" + (f" at step {r['diverged_step']}" if r["diverged"] else ""))
    print(f"\nCHOSEN lr_theta = {chosen['lr_theta']:.1e}")

    out_dir = Path("results/phase1/lr_search")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "search_summary.json").write_text(
        json.dumps({"results": results, "chosen": chosen}, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
