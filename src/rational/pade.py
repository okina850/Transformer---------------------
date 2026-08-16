"""x=0 における真の Padé 近似（Taylor 係数マッチング）。

PAU 論文（Molina et al., ICLR 2020）の Appendix A.1 は、Sigmoid / Tanh / Swish の
初期化係数を「standard techniques」＝ここで実装する古典的 Padé 近似で計算している。
（ReLU 系だけは [-3,3] 上の最小二乗でフィットしている。）

**重要**: これは x=0 まわりの *局所的* 構成であり、[-6,6] のような広い区間での
近似としては最適ではない。Phase 0 ではこれを「参照・検算用」として使い、
区間フィット（``fitting.py``）と誤差を比較する。
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from numpy.typing import NDArray


def pade_at_zero(expr, x, m: int, n: int, exact: bool = True):
    """:math:`f` の [m/n] 次 Padé 近似を x=0 で求める。

    PAU の記法に合わせ、分母は :math:`Q(x) = 1 + \\sum_{k=1}^{n} b_k x^k`
    （**定数項なし**）に正規化して返す。

    Returns
    -------
    (a, b) : 係数リスト（昇順）。``a`` は長さ m+1（:math:`a_0..a_m`）、
             ``b`` は長さ n（:math:`b_1..b_n`、定数項 1 は含めない）。
             ``exact=True`` なら sympy の有理数、False なら float。

    Notes
    -----
    :math:`P(x) = f(x)Q(x) + O(x^{m+n+1})` を係数比較して解く。
    Taylor 係数を :math:`c_i` とすると
    - :math:`i = m+1, \\dots, m+n`: :math:`\\sum_{k=0}^{n} b_k c_{i-k} = 0`（:math:`b_0=1`）→ b が決まる
    - :math:`i = 0, \\dots, m`: :math:`a_i = \\sum_{k=0}^{n} b_k c_{i-k}`
    """
    import sympy as sp

    order = m + n
    series = sp.series(expr, x, 0, order + 1).removeO()
    series = sp.expand(series)
    c = [sp.nsimplify(series.coeff(x, i)) for i in range(order + 1)]

    b_syms = sp.symbols(f"b1:{n + 1}") if n > 0 else ()
    b_syms = list(b_syms) if isinstance(b_syms, tuple) else [b_syms]
    bb = [sp.Integer(1)] + b_syms

    eqs = [
        sum(bb[k] * c[i - k] for k in range(0, min(i, n) + 1))
        for i in range(m + 1, m + n + 1)
    ]
    sol = sp.solve(eqs, b_syms, dict=True)
    if not sol:
        raise RuntimeError(
            f"Pade [{m}/{n}] の連立方程式が解けなかった（分母が退化している可能性）"
        )
    bb = [sp.Integer(1)] + [sp.nsimplify(sol[0][s]) for s in b_syms]

    a = [
        sp.nsimplify(sp.simplify(sum(bb[k] * c[i - k] for k in range(0, min(i, n) + 1))))
        for i in range(m + 1)
    ]
    b = bb[1:]

    if exact:
        return a, b
    return (
        np.array([float(v) for v in a], dtype=np.float64),
        np.array([float(v) for v in b], dtype=np.float64),
    )


def to_fractions(values) -> list[Fraction]:
    """sympy の有理数を ``fractions.Fraction`` に変換する（表示・比較用）。"""
    return [Fraction(int(v.p), int(v.q)) for v in values]


def pade_at_zero_numeric(name: str, m: int, n: int) -> tuple[NDArray, NDArray]:
    """目標関数名から float 係数の Padé 近似を返す薄いラッパ。"""
    from src.rational.targets import sympy_target

    x, expr = sympy_target(name)
    return pade_at_zero(expr, x, m, n, exact=False)
