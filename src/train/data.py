"""memmap ベースのバッチ読み込み（nanoGPT と同じ流儀）。

学習用トークン列を丸ごとメモリに載せず、``np.memmap`` でディスクから
必要な部分だけ読む。100M トークン（uint16, 200MB）程度なら実際には
メモリに収まるが、この流儀にしておけば規模を増やしても変更不要。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class TokenDataset:
    def __init__(self, bin_path: str | Path, block_size: int) -> None:
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def get_batch(self, batch_size: int, device: torch.device, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(self), (batch_size,), generator=generator)
        x = torch.stack([torch.from_numpy(self.data[i : i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1 : i + 1 + self.block_size].astype(np.int64)) for i in ix])
        if device.type == "cuda":
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
