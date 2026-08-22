"""学習中の pre-activation 分位点を記録する（RQ5 の時間軸情報）。

T5（Phase 0）が測ったのは学習済み GPT-2 の分布。Phase 1 はスクラッチ学習なので、
学習初期の pre-activation は LayerNorm 直後にランダム重みを掛けたものであり、
ほぼ正規分布であの重い裾は存在しない（ユーザー指摘）。裾は学習の過程で
後から育つかもしれないし、育たないかもしれない。それ自体が RQ5 の答えになる。

評価タイミングごとに軽量に呼び出す想定（毎ステップは負荷が高いため）。
"""

from __future__ import annotations

import torch


@torch.no_grad()
def collect_preact_quantiles(model, x: torch.Tensor) -> list[dict]:
    """各層の pre-activation の分位点を1回のフォワードで集める。

    Returns
    -------
    list[dict]
        層ごとに ``{"q0.001":.., "q0.01":.., "q0.5":.., "q0.99":.., "q0.999":..,
        "min":.., "max":..}``。
    """
    from src.model.ffn import RationalFFN

    captured: list[torch.Tensor] = []
    handles = []

    def make_hook(store: list):
        def hook(_module, _inp, out):
            store.append(out.detach())
        return hook

    layers_with_preact = []
    for blk in model.blocks:
        if hasattr(blk.ffn, "fc1"):
            layers_with_preact.append(blk.ffn.fc1)

    for mod in layers_with_preact:
        handles.append(mod.register_forward_hook(make_hook(captured)))

    model.eval()
    model(x)
    model.train()
    for h in handles:
        h.remove()

    qs = [0.001, 0.01, 0.5, 0.99, 0.999]
    out = []
    for t in captured:
        flat = t.reshape(-1).float()
        quant = torch.quantile(flat, torch.tensor(qs, device=flat.device))
        out.append({
            **{f"q{q}": float(v) for q, v in zip(qs, quant.tolist())},
            "min": float(flat.min()), "max": float(flat.max()),
            "std": float(flat.std()),
        })
    return out
