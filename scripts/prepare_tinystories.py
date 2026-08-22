r"""Phase 1 / Stage 0: TinyStories をトークナイズし uint16 バイナリに保存する。

事前にトークン ID の配列をディスクに固定しておく（nanoGPT と同じ流儀）。
学習時は `np.memmap` で読み込むので、100M トークン（200MB）でもメモリに
全部載せる必要がない。

出力:
- ``data/tinystories/train.bin``
- ``data/tinystories/val.bin``
- ``data/tinystories/meta.json``（トークン数・vocab_size など）

実行:
    python scripts/prepare_tinystories.py --config configs/prepare_data.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.utils.run_meta import REPO_ROOT, load_yaml, set_seed


def tokenize_split(ds, tok, target_tokens: int, doc_offset: int) -> tuple[np.ndarray, int]:
    """先頭から順に文書を連結・トークナイズし、目標トークン数に達したら止める。

    ``<|endoftext|>`` を文書区切りとして挿入する（GPT-2 と同じ慣行）。
    """
    eot_id = tok.token_to_id("<|endoftext|>")
    buf: list[int] = []
    i = doc_offset
    t0 = time.perf_counter()
    while len(buf) < target_tokens and i < len(ds):
        ids = tok.encode(ds[i]["text"]).ids
        buf.extend(ids)
        buf.append(eot_id)
        i += 1
        if (i - doc_offset) % 50000 == 0:
            print(f"  {i - doc_offset} docs, {len(buf):,} tokens, "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)
    return np.array(buf[:target_tokens], dtype=np.uint16), i


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prepare_data.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 0)))

    from datasets import load_dataset
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(REPO_ROOT / cfg["tokenizer_path"]))
    vocab_size = tok.get_vocab_size()

    ds_train = load_dataset(cfg["dataset"]["name"], split="train")
    ds_val = load_dataset(cfg["dataset"]["name"], split="validation")

    out_dir = REPO_ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print("tokenizing train split...", flush=True)
    train_ids, n_train_docs = tokenize_split(ds_train, tok, int(cfg["n_train_tokens_target"]), 0)
    train_ids.tofile(out_dir / "train.bin")

    print("tokenizing val split...", flush=True)
    val_ids, n_val_docs = tokenize_split(ds_val, tok, int(cfg["n_val_tokens_target"]), 0)
    val_ids.tofile(out_dir / "val.bin")

    meta = {
        "vocab_size": vocab_size,
        "tokenizer_path": cfg["tokenizer_path"],
        "n_train_tokens": int(train_ids.size),
        "n_val_tokens": int(val_ids.size),
        "n_train_docs_used": n_train_docs,
        "n_val_docs_used": n_val_docs,
        "dtype": "uint16",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"train: {train_ids.size:,} tokens from {n_train_docs} docs")
    print(f"val:   {val_ids.size:,} tokens from {n_val_docs} docs")
    print(f"wrote -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
