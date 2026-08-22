"""評価用データの読み込みとトークナイズ（スライディングウィンドウ化）。"""

from __future__ import annotations

import numpy as np
import torch


def load_token_ids(
    tokenizer,
    name: str = "wikitext",
    config: str = "wikitext-2-raw-v1",
    split: str = "validation",
) -> torch.Tensor:
    """データセット全文を連結して1本のトークン列にする。

    言語モデルの perplexity 評価では、文書を連結して固定長の窓で切り出すのが標準。
    文書ごとにパディングすると、パディング部分が指標を歪めるため。
    """
    from datasets import load_dataset

    ds = load_dataset(name, config, split=split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    return ids


def sliding_windows(
    ids: torch.Tensor, n_ctx: int, stride: int, max_windows: int | None = None
) -> list[tuple[int, int, int]]:
    r"""``(begin, end, n_target)`` のリストを返す。

    スライディングウィンドウ評価では、窓を ``stride`` ずつずらしながら
    「窓の末尾 ``stride`` トークンだけ」を予測対象にする。
    こうすると各トークンが最低 ``n_ctx - stride`` トークン分の文脈を持った状態で
    評価される（文書の先頭付近を除く）。stride を小さくするほど文脈が長くなり
    perplexity は下がるが、計算量は増える。
    """
    windows: list[tuple[int, int, int]] = []
    prev_end = 0
    for begin in range(0, max(len(ids) - 1, 1), stride):
        end = min(begin + n_ctx, len(ids))
        n_target = end - prev_end
        if n_target <= 0:
            continue
        windows.append((begin, end, n_target))
        prev_end = end
        if end == len(ids):
            break
        if max_windows is not None and len(windows) >= max_windows:
            break
    return windows


def describe_tokens(ids: torch.Tensor) -> dict:
    return {"n_tokens": int(ids.numel())}
