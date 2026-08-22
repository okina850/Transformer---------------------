r"""Phase 1: decoder-only Transformer の学習エントリポイント。

CLAUDE.md の方針に従い、``--tag smoke`` や小さい ``train.total_steps`` で
まず最小構成の疎通を確認してから、本番規模の設定ファイルで実行すること。

実行:
    python scripts/train.py --config configs/phase1_smoke.yaml --tag smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.model.gpt import GPTConfig, GPTLanguageModel
from src.train.trainer import train_one_run
from src.utils.run_meta import REPO_ROOT, load_yaml, new_run_dir, save_run_meta, set_seed


def build_model(cfg: dict) -> GPTLanguageModel:
    model_cfg = load_yaml(cfg["model"])
    ffn_kind = cfg["ffn_kind"]

    rational_coef = None
    if ffn_kind == "rational":
        cond_name = cfg["rational_condition"]
        init = json.loads((REPO_ROOT / "results/phase0/phase1_init.json").read_text(encoding="utf-8"))
        rational_coef = init["conditions"][cond_name]

    gcfg = GPTConfig(
        vocab_size=int(model_cfg["vocab_size"]),
        block_size=int(model_cfg["block_size"]),
        n_layer=int(model_cfg["n_layer"]),
        n_head=int(model_cfg["n_head"]),
        d_model=int(model_cfg["d_model"]),
        d_ff=int(model_cfg["d_ff"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
        ffn_kind=ffn_kind,
        rational_coef=rational_coef,
    )
    return GPTLanguageModel(gcfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = set_seed(int(cfg.get("seed", 0)))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    model = build_model(cfg)
    n_params = model.n_params()
    print(f"ffn_kind={cfg['ffn_kind']}  n_params={n_params:,} ({n_params/1e6:.3f}M)  device={device}", flush=True)

    condition = cfg["ffn_kind"] if cfg["ffn_kind"] != "rational" else cfg["rational_condition"]
    run_dir = new_run_dir("phase1/train", tag=f"{condition}_{args.tag}_seed{seed}")
    save_run_meta(run_dir, config=cfg, seed=seed, include_gpu=True, extra={"n_params": n_params})

    summary = train_one_run(model, cfg, run_dir, device, seed)
    print(f"\n{json.dumps(summary, indent=2)}")
    print(f"\nwrote -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
