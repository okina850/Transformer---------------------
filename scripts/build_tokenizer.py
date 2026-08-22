"""Phase 1 / Stage 0: TinyStories 用の BPE トークナイザを自前学習する。

**なぜ自前学習するか**: GPT-2 の既製語彙（50,257）は英語版 Wikipedia のような
多様な文章向けに作られており、幼児向けの単純な語彙しか持たない TinyStories には
過大。埋め込み層（`vocab_size × d_model`）が総パラメータを支配してしまうと、
「Transformer 本体（FFN の活性化関数）を比較する」という Phase 1 の目的が薄まる。

**BPE（Byte-Pair Encoding）**: 頻出する文字（バイト）の組を繰り返し併合して
語彙を作るサブワード分割法。GPT-2 系が使う標準的な手法。

出力: ``results/phase1/tokenizer/tokenizer.json``（Hugging Face `tokenizers` 形式）

実行:
    python scripts/build_tokenizer.py --config configs/tokenizer.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.run_meta import REPO_ROOT, load_yaml, save_run_meta, set_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tokenizer.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = set_seed(int(cfg.get("seed", 0)))

    from datasets import load_dataset
    from tokenizers import ByteLevelBPETokenizer

    ds = load_dataset(cfg["dataset"]["name"], split=cfg["dataset"]["split"])
    n = min(int(cfg["n_train_docs"]), len(ds))
    print(f"training BPE tokenizer on {n} / {len(ds)} documents...", flush=True)

    def text_iter():
        for i in range(n):
            yield ds[i]["text"]

    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(
        text_iter(),
        vocab_size=int(cfg["vocab_size"]),
        min_frequency=int(cfg["min_frequency"]),
        special_tokens=list(cfg["special_tokens"]),
    )

    out_dir = REPO_ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_dir / "tokenizer.json"))
    save_run_meta(out_dir, config=cfg, seed=seed, include_gpu=False)

    # 動作確認
    sample = ds[0]["text"][:200]
    ids = tok.encode(sample).ids
    print(f"vocab size (actual): {tok.get_vocab_size()}")
    print(f"sample text: {sample[:80]!r}...")
    print(f"encoded ({len(ids)} tokens): {ids[:20]}...")
    print(f"decoded: {tok.decode(ids)[:80]!r}...")
    print(f"wrote -> {out_dir / 'tokenizer.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
