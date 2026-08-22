r"""オプティマイザとパラメータ群ごとの学習率スケジュール。

**なぜパラメータ群を分けるか**: PyTorch の optimizer は「パラメータ群
（parameter group）」ごとに別の学習率を持てる。RAFT はこれを使い、
モデル本体 `lr_θ` と有理係数 `lr_RAF` を分けている（比 7.14、
θ は線形減衰・RAF は warmup 後定数）。本研究もこの比とスケジュール形を
Pre-registration どおり厳密に守る（絶対値は B1 の探索で決める）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

#: RAFT 論文 Table 10 の比。lr_RAF / lr_theta = 5e-3 / 7e-4
RAF_TO_THETA_LR_RATIO = 5e-3 / 7e-4


def split_param_groups(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """モデル本体パラメータと、有理活性化係数（a, b）を分離する。"""
    from src.rational.activation import RationalActivation

    rational_params: list[nn.Parameter] = []
    rational_ids: set[int] = set()
    for mod in model.modules():
        if isinstance(mod, RationalActivation):
            rational_params.extend([mod.a, mod.b])
            rational_ids.update(id(p) for p in (mod.a, mod.b))

    theta_params = [p for p in model.parameters() if id(p) not in rational_ids]
    return theta_params, rational_params


def make_optimizer(
    model: nn.Module,
    lr_theta: float,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.98),
) -> torch.optim.Optimizer:
    """AdamW を構築する。有理係数があれば別パラメータ群にする（比は lr_scheduler 側で管理）。

    betas=(0.9, 0.98) は RAFT の設定（論文 Table 10）。
    """
    theta_params, rational_params = split_param_groups(model)
    groups = [{"params": theta_params, "lr": lr_theta, "weight_decay": weight_decay, "name": "theta"}]
    if rational_params:
        lr_raf = lr_theta * RAF_TO_THETA_LR_RATIO
        groups.append({"params": rational_params, "lr": lr_raf, "weight_decay": 0.0, "name": "rational"})
    return torch.optim.AdamW(groups, betas=betas)


@dataclass
class LRSchedule:
    r"""RAFT の warmup + (θ: 線形減衰 / RAF: 定数) を実装する。

    ``lr(step)`` はスケーリング係数（0〜1）を返す。実際の学習率は
    ``base_lr * lr(step)`` として optimizer 側に適用する。
    """

    warmup_steps: int
    total_steps: int
    kind: str  # "linear_decay" | "constant"

    def scale(self, step: int) -> float:
        if step < self.warmup_steps:
            return step / max(self.warmup_steps, 1)
        if self.kind == "constant":
            return 1.0
        if self.kind == "linear_decay":
            remaining = max(self.total_steps - step, 0)
            span = max(self.total_steps - self.warmup_steps, 1)
            return max(remaining / span, 0.0)
        raise ValueError(f"unknown schedule kind: {self.kind}")


def apply_lr(optimizer: torch.optim.Optimizer, step: int, schedules: dict[str, LRSchedule], base_lrs: dict[str, float]) -> dict[str, float]:
    """各パラメータ群の学習率をそのステップの値に更新し、適用値を返す（ログ用）。"""
    applied = {}
    for group in optimizer.param_groups:
        name = group["name"]
        lr = base_lrs[name] * schedules[name].scale(step)
        group["lr"] = lr
        applied[name] = lr
    return applied
