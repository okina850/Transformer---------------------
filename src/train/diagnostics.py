r"""Pre-registration（`docs/experiment_log.md`）で定めた判定基準の実装。

**ここに実装した閾値は Stage 0 着手前に確定したものであり、結果を見て変更しない。**
変更が必要になった場合は理由とともに報告する（プロジェクトの手続き上のルール）。

- 損失スパイク: 直前 window=100 ステップの移動平均 + 5×移動標準偏差を超えた場合
- 発散: NaN/Inf、または loss > 2*ln(vocab_size) が10ステップ連続、
  または「クリップ前」勾配ノルムが直前100ステップの移動中央値の20倍を超えた場合
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class DivergenceMonitor:
    vocab_size: int
    window: int = 100
    spike_sigma: float = 5.0
    divergence_loss_multiplier: float = 2.0
    divergence_sustain_steps: int = 10
    grad_norm_spike_multiplier: float = 20.0

    _loss_hist: deque = field(default_factory=lambda: deque(maxlen=100), init=False)
    _gradnorm_hist: deque = field(default_factory=lambda: deque(maxlen=100), init=False)
    _high_loss_streak: int = field(default=0, init=False)
    loss_spikes: list[int] = field(default_factory=list, init=False)
    diverged_step: int | None = field(default=None, init=False)
    diverged_reason: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._loss_hist = deque(maxlen=self.window)
        self._gradnorm_hist = deque(maxlen=self.window)
        self._divergence_loss_threshold = self.divergence_loss_multiplier * math.log(self.vocab_size)

    def update(self, step: int, loss: float, grad_norm_pre_clip: float) -> dict:
        """1ステップ分の値を渡し、そのステップで検出された異常を返す。

        Returns
        -------
        dict with keys: ``is_spike``, ``is_diverged``, ``reason``
        """
        result = {"is_spike": False, "is_diverged": False, "reason": None}

        if not math.isfinite(loss) or not math.isfinite(grad_norm_pre_clip):
            self.diverged_step, self.diverged_reason = step, "nan_or_inf"
            result.update(is_diverged=True, reason="nan_or_inf")
            return result

        # 損失スパイク判定（履歴を更新する前の統計と比較）
        if len(self._loss_hist) >= 10:
            mean = sum(self._loss_hist) / len(self._loss_hist)
            var = sum((v - mean) ** 2 for v in self._loss_hist) / len(self._loss_hist)
            std = math.sqrt(var)
            if loss > mean + self.spike_sigma * std:
                self.loss_spikes.append(step)
                result["is_spike"] = True

        # 勾配ノルムの急増判定
        if len(self._gradnorm_hist) >= 10:
            sorted_gn = sorted(self._gradnorm_hist)
            median = sorted_gn[len(sorted_gn) // 2]
            if median > 0 and grad_norm_pre_clip > self.grad_norm_spike_multiplier * median:
                if self.diverged_step is None:
                    self.diverged_step, self.diverged_reason = step, "grad_norm_spike"
                result.update(is_diverged=True, reason="grad_norm_spike")

        # 持続的な高 loss（ランダム推測の2倍を10ステップ連続）
        if loss > self._divergence_loss_threshold:
            self._high_loss_streak += 1
        else:
            self._high_loss_streak = 0
        if self._high_loss_streak >= self.divergence_sustain_steps:
            if self.diverged_step is None:
                self.diverged_step = step - self.divergence_sustain_steps + 1
                self.diverged_reason = "sustained_high_loss"
            result.update(is_diverged=True, reason=result["reason"] or "sustained_high_loss")

        self._loss_hist.append(loss)
        self._gradnorm_hist.append(grad_norm_pre_clip)
        return result

    def summary(self) -> dict:
        return {
            "n_loss_spikes": len(self.loss_spikes),
            "loss_spike_steps": self.loss_spikes,
            "diverged": self.diverged_step is not None,
            "diverged_step": self.diverged_step,
            "diverged_reason": self.diverged_reason,
        }


def ranges_overlap(a_values: list[float], b_values: list[float]) -> bool:
    """Pre-registration の「差がある」基準: seed 3本のレンジが重ならないこと。

    重ならなければ「差がある」、重なれば「差があるとは言えない」。
    """
    a_lo, a_hi = min(a_values), max(a_values)
    b_lo, b_hi = min(b_values), max(b_values)
    return not (a_hi < b_lo or b_hi < a_lo)


def is_significant(a_values: list[float], b_values: list[float]) -> bool:
    """`not ranges_overlap` の読みやすい別名。"""
    return not ranges_overlap(a_values, b_values)
