r"""FFN（Feed-Forward Network）層の4種類の実装。

**FFN とは**: Transformer の各ブロックにある、Attention の後段の2層全結合ネットワーク。
`W2 · act(W1 · x)` という形で、`act` が活性化関数。本研究が差し替える対象はここ。

**パラメータ数を揃える方針**（research_context §6 必須要件）:
`d_ff` を活性化ごとに調整し、FFN 全体のパラメータ数をほぼ一致させる。
SwiGLU はゲート行列が1枚多いため `d_ff` を 2/3 にする（Shazeer 2020 の慣行）。
有理活性化は層あたり (m+1)+n 個のスカラーパラメータが増えるが、
`d_model` に対して無視できる規模（本研究の設定で全体の0.0007%程度）。
"""

from __future__ import annotations

import torch
from torch import nn

from src.rational.activation import RationalActivation


def swiglu_dff(d_ff_base: int) -> int:
    r"""GELU 版の `d_ff` からパラメータ数を揃えた SwiGLU 用の `d_ff` を計算する。

    GELU FFN のパラメータ数（バイアスなし）: :math:`2 \cdot d_{model} \cdot d_{ff}`。
    SwiGLU FFN: :math:`3 \cdot d_{model} \cdot d_{ff}'`（ゲート・アップ・ダウンの3枚）。
    揃えるには :math:`d_{ff}' = \tfrac{2}{3} d_{ff}`。
    """
    return int(round(d_ff_base * 2 / 3))


class GELUFFN(nn.Module):
    """B1 ベースライン。バイアスなし線形層2枚 + GELU（tanh 近似 = gelu_new）。"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class SwiGLUFFN(nn.Module):
    """B2 ベースライン。`(W_gate x) \\odot SiLU(W_up x)` を `W_down` で戻す。"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_down(self.act(self.w_gate(x)) * self.w_up(x)))


class RationalFFN(nn.Module):
    r"""R1〜R4。GELUFFN と同じ骨格で、活性化だけ学習可能有理関数に差し替える。

    Parameters
    ----------
    coef
        ``results/phase0/phase1_init.json`` の ``conditions[name]`` の中身
        （``m``, ``n``, ``denom``, ``eps``, ``a``, ``b`` を持つ dict）。
    """

    def __init__(self, d_model: int, d_ff: int, coef: dict, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = RationalActivation(
            m=int(coef["m"]),
            n=int(coef["n"]),
            denom=coef["denom"],
            eps=float(coef.get("eps") or 1e-3),
            a_init=coef["a"],
            b_init=coef["b"],
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))

    def preact(self, x: torch.Tensor) -> torch.Tensor:
        """診断用: 活性化に入る直前の値（pre-activation）だけを返す。"""
        return self.fc1(x)
