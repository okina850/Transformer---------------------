r"""decoder-only Transformer のミニマル実装（nanoGPT 系）。

**decoder-only / 自己回帰**とは: 左から順に次のトークンを予測する構成。
各位置は自分より前の位置しか参照できない（causal mask）。GPT 系はこちら、
BERT 系（RAFT が使う）は前後両方を見る encoder-only。

FFN 部分だけが条件間で異なる（``src/model/ffn.py``）。Attention・埋め込み・
LayerNorm・重み初期化は全条件で共通にすることで、FFN の活性化関数の効果だけを
比較できるようにする。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    d_ff: int = 1536
    dropout: float = 0.0
    #: "gelu" | "swiglu" | "rational"
    ffn_kind: str = "gelu"
    #: ffn_kind="rational" のときの係数 dict（results/phase0/phase1_init.json 由来）
    rational_coef: dict | None = None


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_drop = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        # scaled dot-product attention（causal mask込み、Flash Attention 経由）
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


def make_ffn(cfg: GPTConfig) -> nn.Module:
    from src.model.ffn import GELUFFN, RationalFFN, SwiGLUFFN, swiglu_dff

    if cfg.ffn_kind == "gelu":
        return GELUFFN(cfg.d_model, cfg.d_ff, cfg.dropout)
    if cfg.ffn_kind == "swiglu":
        return SwiGLUFFN(cfg.d_model, swiglu_dff(cfg.d_ff), cfg.dropout)
    if cfg.ffn_kind == "rational":
        if cfg.rational_coef is None:
            raise ValueError("ffn_kind='rational' には rational_coef が必要")
        return RationalFFN(cfg.d_model, cfg.d_ff, cfg.rational_coef, cfg.dropout)
    raise ValueError(f"unknown ffn_kind: {cfg.ffn_kind!r}")


class Block(nn.Module):
    """Attention + FFN を、残差接続と Pre-LayerNorm で繋いだセット。"""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = make_ffn(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """token embedding + position embedding -> blocks -> ln_f -> lm_head。"""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # weight tying（埋め込みと出力層で重みを共有し、パラメータ数を削減）
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # 残差経路への射影層は GPT-2 の慣行に合わせて縮小初期化
        for name, p in self.named_parameters():
            if name.endswith(("proj.weight", "fc2.weight", "w_down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def n_params(self, exclude_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    # ------------------------------------------------------------------
    # RQ3 / RQ5 診断用ヘルパ
    # ------------------------------------------------------------------
    def rational_activations(self) -> list:
        """各層の RationalActivation インスタンスを層順に返す（gelu/swiglu では空）。"""
        from src.rational.activation import RationalActivation

        return [
            blk.ffn.act for blk in self.blocks if isinstance(blk.ffn.act, RationalActivation)
        ]
