r"""学習可能有理活性化関数の PyTorch 実装（Phase 1 本体で使うモジュール）。

:math:`F(x) = P(x)/Q(x)`、:math:`P(x)=\sum_{j=0}^{m} a_j x^j`、
:math:`A(x)=\sum_{k=1}^{n} b_k x^k`（**定数項なし**、PAU 原論文 Eq.(2) の定義）。

分母 :math:`Q` の設計は 4 種類（``src/rational/denominators.py`` の numpy 版と同じ）:

- ``abs``    : :math:`1+|A|`            — PAU / RAFT / KAT が使う形
- ``sqrt``   : :math:`1+\sqrt{eps+A^2}` — 本研究の提案（実効次数・裾は abs と一致、C^inf）
- ``square`` : :math:`1+A^2`            — 実効次数が 2n になる（Phase 1 では使わない）
- ``free``   : :math:`1+A`              — 制約なし。極を持ちうる

**評価は Horner 法**（KAT も同じ）。素朴な :math:`\sum a_j x^j` は
:math:`x^m` を直接作るため float32 で桁落ちしやすい。

**層ごとに1つ共有する**のが PAU / RAFT の流儀（活性化のパラメータは層あたり
:math:`(m+1)+n = 10` 個だけ）。Phase 1 もこれに合わせる。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


class RationalActivation(nn.Module):
    r"""学習可能な有理活性化関数。

    Parameters
    ----------
    m, n
        分子次数と、分母内多項式 :math:`A` の次数。既定は PAU / RAFT / KAT 共通の [5,4]。
    denom
        ``"abs"`` / ``"sqrt"`` / ``"square"`` / ``"free"``。
    eps
        ``denom="sqrt"`` のときの平滑化パラメータ。
        Phase 0 の ε スイープで **1e-3** に決定（精度の上限条件と平滑化の下限条件を
        同時に満たす唯一の桁）。
    a_init, b_init
        初期化係数。省略時は 0（学習前提のテスト用途のみ。実験では必ず与えること）。
    """

    def __init__(
        self,
        m: int = 5,
        n: int = 4,
        denom: str = "abs",
        eps: float = 1e-3,
        a_init: torch.Tensor | list[float] | None = None,
        b_init: torch.Tensor | list[float] | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if denom not in ("abs", "sqrt", "square", "free"):
            raise ValueError(
                f"unknown denom {denom!r}. available: ['abs', 'sqrt', 'square', 'free']"
            )
        self.m, self.n, self.denom, self.eps = m, n, denom, float(eps)

        a = torch.zeros(m + 1, dtype=dtype) if a_init is None else torch.as_tensor(a_init, dtype=dtype)
        b = torch.zeros(n, dtype=dtype) if b_init is None else torch.as_tensor(b_init, dtype=dtype)
        if a.numel() != m + 1:
            raise ValueError(f"a_init は長さ {m + 1} である必要がある (got {a.numel()})")
        if b.numel() != n:
            raise ValueError(f"b_init は長さ {n} である必要がある (got {b.numel()})")

        self.a = nn.Parameter(a.clone())
        self.b = nn.Parameter(b.clone())

    # ------------------------------------------------------------------
    @staticmethod
    def _horner(coeffs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """昇順係数の多項式を Horner 法で評価する。"""
        out = torch.full_like(x, 0.0) + coeffs[-1]
        for i in range(coeffs.numel() - 2, -1, -1):
            out = out * x + coeffs[i]
        return out

    def _A(self, x: torch.Tensor) -> torch.Tensor:
        r""":math:`A(x)=\sum_{k=1}^{n} b_k x^k = x\sum_{j=0}^{n-1} b_{j+1} x^j`。

        定数項がないので :math:`x` をくくり出して Horner 法にかける。
        こうすると :math:`A(0)=0` が**丸め誤差なしに厳密に成り立つ**。
        これは重要で、:math:`Q(0)=1` の正規化と、
        「x=0 で A は接するだけで符号を変えない」という Phase 0 の性質が
        数値的にも保たれる。
        """
        if self.n == 0:
            return torch.zeros_like(x)
        return x * self._horner(self.b, x)

    def q(self, x: torch.Tensor) -> torch.Tensor:
        """分母 :math:`Q(x)`。設計により4通り。"""
        A = self._A(x)
        if self.denom == "abs":
            return 1.0 + A.abs()
        if self.denom == "sqrt":
            return 1.0 + torch.sqrt(A * A + self.eps)
        if self.denom == "square":
            return 1.0 + A * A
        return 1.0 + A  # free

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._horner(self.a, x) / self.q(x)

    # ------------------------------------------------------------------
    # RQ3 の診断量（Phase 1 で毎ステップ記録する）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def count_sign_changes(
        self,
        lo: float = -15.0,
        hi: float = 15.0,
        n_samples: int = 4001,
        slope_tol: float = 1e-3,
    ) -> int:
        r""":math:`A(x)` が有意な傾きで符号を変える回数を数える（RQ3 の核心）。

        ``abs`` 分母ではこれが微分不連続点の個数そのもの。
        **学習中にこれが 0 から増えるかどうか**が RQ3 の問いである。

        ``slope_tol`` が絶対閾値でよい理由: 符号変化点では :math:`A=0` すなわち
        :math:`Q=1` なので、:math:`|A'|` はそのまま微分の跳びの相対的大きさになる。
        Phase 0 実測では偽陽性が ~2e-5、真の折れ点が 1.35〜3.84 と 5 桁離れていた。
        """
        x = torch.linspace(lo, hi, n_samples, device=self.b.device, dtype=self.b.dtype)
        A = self._A(x)
        s = torch.sign(A)
        nz = s != 0
        if int(nz.sum()) < 2:
            return 0
        s, xs = s[nz], x[nz]
        idx = torch.nonzero(s[1:] != s[:-1], as_tuple=False).flatten()
        if idx.numel() == 0:
            return 0
        cand = 0.5 * (xs[idx] + xs[idx + 1])
        # A'(x) を中心差分で評価（係数から作るより実装が短く、診断用途には十分）
        h = torch.as_tensor((hi - lo) / (n_samples - 1), device=x.device, dtype=x.dtype)
        slope = ((self._A(cand + h) - self._A(cand - h)) / (2 * h)).abs()
        return int((slope > slope_tol).sum())

    @torch.no_grad()
    def extra_repr(self) -> str:
        eps = f", eps={self.eps:g}" if self.denom == "sqrt" else ""
        return f"m={self.m}, n={self.n}, denom={self.denom}{eps}"

    # ------------------------------------------------------------------
    @classmethod
    def from_coefficients_file(
        cls, path: str | Path, key: str, denom: str | None = None, **kwargs
    ) -> RationalActivation:
        """Phase 0 が出力した ``coefficients.json`` から初期化する。

        ``key`` は ``"gelu_tanh|paper|m5n4|abs"`` の形式。
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if key not in data:
            raise KeyError(
                f"key {key!r} が見つからない。候補の例: {sorted(data)[:5]}"
            )
        c = data[key]
        return cls(
            m=int(c["m"]),
            n=int(c["n"]),
            denom=denom or c["denom"],
            eps=float(c.get("denom_params", {}).get("eps", kwargs.pop("eps", 1e-3))),
            a_init=c["a"],
            b_init=c["b"],
            **kwargs,
        )
