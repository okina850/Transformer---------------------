"""GPT-2 の FFN 活性化を有理近似に差し替える。

**学習は一切行わない。** 既に学習済みの GELU 用の重みに対して、
活性化関数だけを有理近似（またはその他の代替）に差し替えたときの
perplexity 劣化を測る（Phase 0 の項目3・4）。
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn

from src.rational.activation import RationalActivation


class _ConstFn(nn.Module):
    """比較対照: 差し替えコード自体のオーバーヘッドを測るための恒等関数など。"""

    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


@contextmanager
def swapped_activation(model, new_act_fn):
    """GPT-2 の全 MLP 活性化を一時的に差し替えるコンテキストマネージャ。

    ``new_act_fn`` は ``nn.Module`` のインスタンス、または呼び出し可能な
    ``callable(int) -> nn.Module``（層番号を受け取り、層ごとに別インスタンスを作る）。
    後者は :class:`RationalActivation` のように学習可能パラメータを持つ場合に使う
    （全層で同じインスタンスを共有すると誤って重みを共有してしまうため）。
    """
    blocks = model.transformer.h
    originals = [blk.mlp.act for blk in blocks]
    try:
        for i, blk in enumerate(blocks):
            mod = new_act_fn(i) if callable(new_act_fn) and not isinstance(new_act_fn, nn.Module) else new_act_fn
            blk.mlp.act = mod.to(next(model.parameters()).device)
        yield model
    finally:
        for blk, orig in zip(blocks, originals):
            blk.mlp.act = orig


def make_rational_factory(m: int, n: int, denom: str, eps: float, a, b):
    """全層で同じ係数から独立インスタンスを作るファクトリ（PAU 流の「層ごとに1つ」）。"""

    def factory(_layer_idx: int) -> RationalActivation:
        return RationalActivation(m=m, n=n, denom=denom, eps=eps, a_init=a, b_init=b)

    return factory
