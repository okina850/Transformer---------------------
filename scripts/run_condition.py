r"""Phase 1: 1つの条件（B1/B2/R1〜R4）を複数 seed で本番学習する。

Stage 1（B1）・Stage 2（B2）・Stage 3（R1〜R4）はすべてこのスクリプトを
条件を変えて呼ぶだけ。`lr_theta` は Stage 1 で確定した値を全条件で共通に使う
（承認済み方針）。

実行例:
    python scripts/run_condition.py --ffn-kind gelu --seeds 0 1 2 --lr-theta 1e-3
    python scripts/run_condition.py --ffn-kind rational --rational-condition R1 \
        --seeds 0 1 2 --lr-theta 1e-3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts.train import build_model
from src.train.trainer import train_one_run
from src.utils.run_meta import load_yaml, new_run_dir, save_run_meta, set_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1_production.yaml")
    ap.add_argument("--ffn-kind", required=True, choices=["gelu", "swiglu", "rational"])
    ap.add_argument("--rational-condition", default=None, help="R1/R2/R3/R4（ffn-kind=rational のとき必須）")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--lr-theta", type=float, required=True)
    ap.add_argument("--tag", default="prod")
    args = ap.parse_args()

    base_cfg = load_yaml(args.config)
    base_cfg["ffn_kind"] = args.ffn_kind
    base_cfg["rational_condition"] = args.rational_condition
    base_cfg["train"]["lr_theta"] = args.lr_theta

    device = torch.device(base_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    condition_name = args.ffn_kind if args.ffn_kind != "rational" else args.rational_condition

    t0 = time.perf_counter()
    summaries = []
    for s in args.seeds:
        cfg = json.loads(json.dumps(base_cfg))
        seed = set_seed(s)
        model = build_model(cfg)
        n_params = model.n_params()

        run_dir = new_run_dir("phase1/train", tag=f"{condition_name}_{args.tag}_seed{seed}")
        save_run_meta(run_dir, config=cfg, seed=seed, include_gpu=True, extra={"n_params": n_params})

        print(f"\n=== {condition_name} seed={seed} ===  n_params={n_params:,}", flush=True)
        summary = train_one_run(model, cfg, run_dir, device, seed)
        summary["condition"] = condition_name
        summary["run_dir"] = str(run_dir)
        summaries.append(summary)

        print(f"  -> stop_reason={summary['stop_reason']}  "
              f"final_step={summary['final_step']}  "
              f"elapsed={summary['elapsed_sec']:.1f}s", flush=True)

        del model
        torch.cuda.empty_cache()

    total_elapsed = time.perf_counter() - t0
    out_dir = Path("results/phase1/train") / f"_{condition_name}_{args.tag}_summary.json"
    out_dir.write_text(json.dumps({
        "condition": condition_name, "lr_theta": args.lr_theta,
        "seeds": args.seeds, "total_elapsed_sec": total_elapsed,
        "runs": summaries,
    }, indent=2), encoding="utf-8")

    print(f"\n=== {condition_name} 完了: {len(args.seeds)} seeds, "
          f"{total_elapsed/60:.1f}分 ===")
    for s in summaries:
        print(f"  seed={s['seed']}: {s['stop_reason']} (final_step={s['final_step']})")
    print(f"summary -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
