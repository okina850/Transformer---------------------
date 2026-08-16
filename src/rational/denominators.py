r"""分母 :math:`Q(x)` の設計バリエーション。

いずれも PAU 論文の記法に合わせ、内側の多項式を
:math:`A(x) = \sum_{k=1}^{n} b_k x^k`（**定数項なし**）とする。
定数項を持たせない理由は2つ:

1. すべての設計で :math:`Q(0) = 1` に正規化され、比較の基準が揃う
2. パラメータ数が :math:`n` で揃い、CLAUDE.md の「パラメータ数を揃える」要件を満たす

各設計の性質（``n`` = A の次数、``m`` = 分子 P の次数）:

===========  ==========================  ==============  ==========  ==================
ID           Q(x)                        実効分母次数    平滑性      裾 F ~ x^(m-degQ)
===========  ==========================  ==============  ==========  ==================
``free``     :math:`1 + A(x)`            n               C^inf       m - n（**極あり**）
``abs``      :math:`1 + |A(x)|`          n               C^0（折れ）  m - n
``sqrt``     :math:`1 + \sqrt{eps+A^2}`  n               C^inf       m - n
``square``   :math:`1 + A(x)^2`          **2n**          C^inf       **m - 2n**
===========  ==========================  ==============  ==========  ==================

**この表が論点A の核心である。** ``sqrt`` は PAU (``abs``) と実効次数・裾が一致し、
違いは平滑性だけになる（＝分母設計の効果を単独で取り出せる統制された比較）。
一方 ``square`` は実効次数が2倍になるため、平滑性と裾の両方が同時に変わる。
``square`` を PAU と比較するときは、裾を揃えるために :math:`m = 2n+1` を取る
変種も併せて評価すること。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.rational.poly import horner, polyroots, real_roots_in


def _eval_A(b: ArrayLike, x: ArrayLike) -> NDArray[np.floating]:
    """:math:`A(x)=\\sum_{k=1}^{n} b_k x^k` を評価する。``b`` は b_1..b_n（昇順）。"""
    b = np.asarray(b, dtype=np.float64)
    coeffs = np.concatenate(([0.0], b))  # 定数項 0 を先頭に付けて昇順係数にする
    return horner(coeffs, x)


def _A_coeffs(b: ArrayLike) -> NDArray[np.floating]:
    """A(x) の昇順係数（定数項 0 込み）を返す。"""
    return np.concatenate(([0.0], np.asarray(b, dtype=np.float64)))


@dataclass(frozen=True)
class Denominator:
    """分母設計の共通インターフェース。"""

    name: str
    #: 実効分母次数 = degree_factor * n
    degree_factor: int
    #: 極を持ちうるか（構造的に排除できていないか）
    can_have_poles: bool
    #: 微分が至るところ連続か
    is_smooth: bool
    params: dict = field(default_factory=dict)

    def q(self, b: ArrayLike, x: ArrayLike) -> NDArray[np.floating]:
        raise NotImplementedError

    def effective_degree(self, n: int) -> int:
        return self.degree_factor * n

    def tail_exponent(self, m: int, n: int) -> int:
        """:math:`x \\to \\pm\\infty` で :math:`F(x) \\sim x^{p}` の指数 p。

        p=1 なら GELU の右裾（線形）と同じ増え方、p=0 なら有界、
        p<0 なら 0 に減衰する。
        """
        return m - self.effective_degree(n)

    def kinks(self, b: ArrayLike, lo: float, hi: float) -> NDArray[np.floating]:
        """区間内の微分不連続点。平滑な設計では空配列。"""
        return np.array([])

    def poles(self, b: ArrayLike, lo: float, hi: float) -> NDArray[np.floating]:
        """区間内の極（Q(x)=0 となる実点）。構造的に極なしの設計では空配列。"""
        return np.array([])


class FreeDenominator(Denominator):
    """:math:`Q(x) = 1 + A(x)`。制約なし（Boullé 型）。**極を持ちうる。**

    「制約を課さないと本当に極が出るのか」を実測するための対照群として置く。
    出ないなら、そもそも安全化の動機が薄いことになる。
    """

    def __init__(self) -> None:
        super().__init__(name="free", degree_factor=1, can_have_poles=True, is_smooth=True)

    def q(self, b: ArrayLike, x: ArrayLike) -> NDArray[np.floating]:
        return 1.0 + _eval_A(b, x)

    def poles(self, b: ArrayLike, lo: float, hi: float) -> NDArray[np.floating]:
        coeffs = _A_coeffs(b)
        coeffs[0] += 1.0  # 1 + A(x) = 0 の根
        return real_roots_in(coeffs, lo, hi)


class AbsDenominator(Denominator):
    r""":math:`Q(x) = 1 + |A(x)|`。PAU 原論文 Eq.(2)。極はないが折れ曲がりうる。

    **折れ点は「A(x)=0 の根」ではなく「A(x) が符号を変える点」であることに注意。**
    :math:`A` は定数項を持たないので :math:`A(0)=0` が常に成り立ち、x=0 は必ず根になる。
    しかし :math:`A(x) \approx b_2 x^2` のように接するだけなら
    :math:`|A| = |b_2|x^2` となり微分は連続で、絶対値は何もしていない。
    実際に折れるのは :math:`b_1 \neq 0` で :math:`A` が原点を横切るときである。
    そしてそのとき折れ点は **x=0 ちょうど**、すなわち pre-activation 分布の
    最頻値に一致する（Phase 1 で追跡すべき診断量）。
    """

    def __init__(self) -> None:
        super().__init__(name="abs", degree_factor=1, can_have_poles=False, is_smooth=False)

    def q(self, b: ArrayLike, x: ArrayLike) -> NDArray[np.floating]:
        return 1.0 + np.abs(_eval_A(b, x))

    def kinks(
        self, b: ArrayLike, lo: float, hi: float, n_samples: int = 200001
    ) -> NDArray[np.floating]:
        """区間内で :math:`A(x)` が符号を変える点（＝真の微分不連続点）。

        重根（接するだけ）を折れ点と誤検出しないよう、根の位置ではなく
        密なグリッド上の符号変化で判定する。判定は区間の中点で返す。
        """
        x = np.linspace(lo, hi, n_samples)
        s = np.sign(_eval_A(b, x))
        nz = s != 0
        if nz.sum() < 2:
            return np.array([])
        s, xs = s[nz], x[nz]
        idx = np.nonzero(np.diff(s) != 0)[0]
        return 0.5 * (xs[idx] + xs[idx + 1])

    def roots_of_A(self, b: ArrayLike, lo: float, hi: float) -> NDArray[np.floating]:
        """:math:`A(x)=0` の実根（符号変化を伴わないものも含む）。参考用。"""
        return real_roots_in(_A_coeffs(b), lo, hi)


class SqrtDenominator(Denominator):
    r""":math:`Q(x) = 1 + \sqrt{\varepsilon + A(x)^2}`。絶対値の平滑化。

    **PAU と実効次数・裾が一致し、違いは平滑性だけになる。**
    :math:`\varepsilon \to 0` で ``abs`` に収束するので、
    :math:`\varepsilon` を振れば「平滑化の強さ」を連続的に制御できる。
    分母設計の効果を単独で取り出す統制された比較として最も筋が良い。
    """

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__(
            name="sqrt",
            degree_factor=1,
            can_have_poles=False,
            is_smooth=True,
            params={"eps": eps},
        )

    def q(self, b: ArrayLike, x: ArrayLike) -> NDArray[np.floating]:
        eps = self.params["eps"]
        return 1.0 + np.sqrt(eps + _eval_A(b, x) ** 2)


class SquareDenominator(Denominator):
    r""":math:`Q(x) = 1 + A(x)^2`。research_context §5.2 の第一候補。

    常に :math:`\ge 1`、C^inf、実装が単純。ただし**実効分母次数が 2n になる**ため、
    同じ (m, n) では PAU と裾の挙動が変わる（論点A）。
    """

    def __init__(self) -> None:
        super().__init__(name="square", degree_factor=2, can_have_poles=False, is_smooth=True)

    def q(self, b: ArrayLike, x: ArrayLike) -> NDArray[np.floating]:
        return 1.0 + _eval_A(b, x) ** 2


def make_denominator(spec: str | dict) -> Denominator:
    """設定ファイルの記述から分母設計を作る。

    ``"abs"`` / ``{"kind": "sqrt", "eps": 1e-3}`` の形を受ける。
    """
    if isinstance(spec, str):
        spec = {"kind": spec}
    kind = spec.get("kind")
    if kind == "free":
        return FreeDenominator()
    if kind == "abs":
        return AbsDenominator()
    if kind == "sqrt":
        return SqrtDenominator(eps=float(spec.get("eps", 1e-3)))
    if kind == "square":
        return SquareDenominator()
    raise KeyError(
        f"unknown denominator {kind!r}. available: ['free', 'abs', 'sqrt', 'square']"
    )


def eval_rational(
    a: ArrayLike, b: ArrayLike, x: ArrayLike, denom: Denominator
) -> NDArray[np.floating]:
    """:math:`F(x) = P(x)/Q(x)` を評価する。``a`` は a_0..a_m、``b`` は b_1..b_n。"""
    return horner(a, x) / denom.q(b, x)
