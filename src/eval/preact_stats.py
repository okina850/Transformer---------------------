r"""FFN の pre-activation 分布を層ごとにストリーミング集計する（T5 / RQ5）。

**なぜ必要か**: Phase 0 Day 2 の結果、有理活性化の問題（絶対値の発動・極・裾の破綻）は
すべて「フィット区間の外」で起きることが分かった。したがって
**モデルが実際に区間外をどれだけ通るか**が、分母設計に関する問い全体の前提条件になる。
PAU と RAFT はいずれも **[-3,3]** でフィットしているので、そこが基準線である。

**pre-activation** とは活性化関数に入る直前の値のこと。GPT-2 の FFN は
``c_proj(act(c_fc(x)))`` という形なので、``c_fc`` の出力がそれにあたる。

全要素を保存すると数十 GB になるため、ヒストグラムと積算量だけを保持する
（**ストリーミング集計**）。分位点はヒストグラムから復元する。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class LayerAccumulator:
    """1層分の集計器。すべて GPU 上のテンソルで保持し、最後に CPU へ移す。"""

    n_bins: int
    lo: float
    hi: float
    device: torch.device
    thresholds: tuple[float, ...]

    hist: torch.Tensor = field(init=False)
    n_below: torch.Tensor = field(init=False)
    n_above: torch.Tensor = field(init=False)
    count: torch.Tensor = field(init=False)
    total: torch.Tensor = field(init=False)
    total_sq: torch.Tensor = field(init=False)
    vmin: torch.Tensor = field(init=False)
    vmax: torch.Tensor = field(init=False)
    over: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        z = lambda: torch.zeros((), dtype=torch.float64, device=self.device)  # noqa: E731
        self.hist = torch.zeros(self.n_bins, dtype=torch.float64, device=self.device)
        self.n_below, self.n_above = z(), z()
        self.count, self.total, self.total_sq = z(), z(), z()
        self.vmin = torch.tensor(float("inf"), dtype=torch.float64, device=self.device)
        self.vmax = torch.tensor(float("-inf"), dtype=torch.float64, device=self.device)
        # |x| > t となった要素数（閾値ごと）。ヒストグラム由来ではなく厳密に数える
        self.over = torch.zeros(len(self.thresholds), dtype=torch.float64, device=self.device)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        v = x.detach().reshape(-1).to(torch.float32)
        self.count += v.numel()
        v64 = v.to(torch.float64)
        self.total += v64.sum()
        self.total_sq += (v64 * v64).sum()
        self.vmin = torch.minimum(self.vmin, v64.min())
        self.vmax = torch.maximum(self.vmax, v64.max())

        av = v.abs()
        for i, t in enumerate(self.thresholds):
            self.over[i] += (av > t).sum()

        self.n_below += (v < self.lo).sum()
        self.n_above += (v > self.hi).sum()
        # torch.histc は範囲外を落とすので、上の2つで別途数えている
        self.hist += torch.histc(v, bins=self.n_bins, min=self.lo, max=self.hi).to(torch.float64)

    def finalize(self) -> dict:
        hist = self.hist.cpu().numpy()
        count = float(self.count.item())
        edges = np.linspace(self.lo, self.hi, self.n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        mean = float(self.total.item()) / count
        var = max(float(self.total_sq.item()) / count - mean * mean, 0.0)
        std = float(np.sqrt(var))

        below = float(self.n_below.item())
        above = float(self.n_above.item())

        # ヒストグラムからの分位点。範囲外の質量は両端に寄せて扱う
        cum = np.concatenate(([below], below + np.cumsum(hist)))
        cum = np.concatenate((cum, [below + hist.sum() + above]))
        grid = np.concatenate(([self.lo], edges[1:], [self.hi]))

        def quantile(q: float) -> float:
            target = q * count
            i = int(np.searchsorted(cum, target))
            i = min(max(i, 1), len(grid) - 1)
            c0, c1 = cum[i - 1], cum[i]
            g0, g1 = grid[i - 1], grid[i]
            if c1 <= c0:
                return float(g1)
            return float(g0 + (g1 - g0) * (target - c0) / (c1 - c0))

        qs = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
        out = {
            "count": count,
            "mean": mean,
            "std": std,
            "min": float(self.vmin.item()),
            "max": float(self.vmax.item()),
            "hist_out_of_range_frac": (below + above) / count,
            "quantiles": {str(q): quantile(q) for q in qs},
        }
        out["iqr"] = out["quantiles"]["0.75"] - out["quantiles"]["0.25"]
        for i, t in enumerate(self.thresholds):
            out[f"frac_abs_gt_{t:g}"] = float(self.over[i].item()) / count
        out["_hist"] = hist.tolist()
        out["_centers"] = centers.tolist()
        return out


class PreActCollector:
    """モデル全層の pre-activation を forward hook で集める。

    **forward hook** とは、PyTorch で順伝播の途中の値を横取りする仕組み。
    モデル本体のコードを書き換えずに中間層の出力を取り出せる。
    """

    def __init__(
        self,
        n_layers: int,
        device: torch.device,
        lo: float = -64.0,
        hi: float = 64.0,
        n_bins: int = 4096,
        thresholds: tuple[float, ...] = (3.0, 6.0, 10.0),
    ) -> None:
        self.accs = [
            LayerAccumulator(n_bins=n_bins, lo=lo, hi=hi, device=device, thresholds=thresholds)
            for _ in range(n_layers)
        ]
        self.handles: list = []

    def _hook(self, layer: int):
        def fn(_module, _inp, out):
            self.accs[layer].update(out)

        return fn

    def attach(self, mlp_input_modules: list[torch.nn.Module]) -> None:
        """各層の「pre-activation を出力するモジュール」に hook を付ける。

        GPT-2 なら ``blk.mlp.c_fc``（その出力が活性化関数への入力そのもの）。
        """
        for i, mod in enumerate(mlp_input_modules):
            self.handles.append(mod.register_forward_hook(self._hook(i)))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def finalize(self) -> list[dict]:
        return [a.finalize() for a in self.accs]

    def __enter__(self) -> PreActCollector:
        return self

    def __exit__(self, *exc) -> None:
        self.remove()


def gpt2_preact_modules(model) -> list[torch.nn.Module]:
    """GPT-2 系モデルから、pre-activation を出力するモジュールを層順に返す。"""
    blocks = model.transformer.h
    return [blk.mlp.c_fc for blk in blocks]
