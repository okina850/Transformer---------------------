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
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray


class PadeDegenerateError(RuntimeError):
    """[m/n] の Padé 近似が一意に定まらない場合に送出される。

    Padé 表には「ブロック」と呼ばれる退化した領域が存在し、
    そこでは分母を決める線形系が特異になる。偶関数・奇関数では
    Taylor 係数が一つ飛びに 0 になるため、これが頻繁に起きる
    （例: gelu_tanh の [7/3]）。異常ではなく Padé 近似の性質である。
    """


def pade_at_zero(expr, x, m: int, n: int, exact: bool = True):
    r""":math:`f` の [m/n] 次 Padé 近似を x=0 で求める。

    PAU の記法に合わせ、分母は :math:`Q(x) = 1 + \sum_{k=1}^{n} b_k x^k`
    （**定数項なし**）に正規化して返す。

    Returns
    -------
    (a, b) : 係数リスト（昇順）。``a`` は長さ m+1（:math:`a_0..a_m`）、
             ``b`` は長さ n（:math:`b_1..b_n`、定数項 1 は含めない）。
             ``exact=True`` なら sympy の有理数、False なら float。

    Raises
    ------
    PadeDegenerateError
        ``exact=True`` で線形系が一意に解けないとき。
        ``exact=False`` のときは最小ノルム解にフォールバックするので送出されない
        （初期値として使う分にはそれで十分なため）。

    Notes
    -----
    :math:`P(x) = f(x)Q(x) + O(x^{m+n+1})` を係数比較して解く。
    Taylor 係数を :math:`c_i` とすると
    - :math:`i = m+1, \dots, m+n`: :math:`\sum_{k=0}^{n} b_k c_{i-k} = 0`（:math:`b_0=1`）→ b が決まる
    - :math:`i = 0, \dots, m`: :math:`a_i = \sum_{k=0}^{n} b_k c_{i-k}`
    """
    import sympy as sp

    order = m + n
    series = sp.expand(sp.series(expr, x, 0, order + 1).removeO())
    c = [sp.nsimplify(series.coeff(x, i)) for i in range(order + 1)]

    if n == 0:
        if exact:
            return list(c[: m + 1]), []
        return np.array([float(v) for v in c[: m + 1]]), np.array([])

    # ---- 数値経路 ----
    # sympy の solve / simplify は目標関数が sqrt(2/pi) のような無理数を含むと
    # 極端に遅くなる（gelu_tanh の高次で数分かかる）。初期値として使うだけなので
    # Taylor 係数を float に落として numpy の線形代数だけで解く。
    # lstsq は退化した系でも最小ノルム解を返すので、Padé 表のブロックも自然に扱える。
    if not exact:
        cf = np.array([float(v) for v in c], dtype=np.float64)
        M = np.zeros((n, n))
        rhs = np.zeros(n)
        for r, i in enumerate(range(m + 1, m + n + 1)):
            for k in range(1, n + 1):
                if 0 <= i - k < len(cf):
                    M[r, k - 1] = cf[i - k]
            rhs[r] = -cf[i] if i < len(cf) else 0.0
        b_num, *_ = np.linalg.lstsq(M, rhs, rcond=None)
        bb_num = np.concatenate(([1.0], b_num))
        a_num = np.array(
            [sum(bb_num[k] * cf[i - k] for k in range(0, min(i, n) + 1)) for i in range(m + 1)]
        )
        return a_num, b_num

    # ---- 厳密経路（Table 6 の検算専用） ----
    b_syms = list(sp.symbols(f"b1:{n + 1}"))
    eqs = [
        sum(([sp.Integer(1)] + b_syms)[k] * c[i - k] for k in range(0, min(i, n) + 1))
        for i in range(m + 1, m + n + 1)
    ]
    sol = sp.solve(eqs, b_syms, dict=True)
    if not (sol and all(s in sol[0] for s in b_syms)):
        raise PadeDegenerateError(
            f"Pade [{m}/{n}] は退化しており一意に定まらない（Padé 表のブロック）"
        )

    bb = [sp.Integer(1)] + [sp.nsimplify(sol[0][s]) for s in b_syms]
    a = [
        sp.nsimplify(sp.simplify(sum(bb[k] * c[i - k] for k in range(0, min(i, n) + 1))))
        for i in range(m + 1)
    ]
    return a, bb[1:]


def to_fractions(values) -> list[Fraction]:
    """sympy の有理数を ``fractions.Fraction`` に変換する（表示・比較用）。"""
    return [Fraction(int(v.p), int(v.q)) for v in values]


@lru_cache(maxsize=256)
def _pade_at_zero_numeric_cached(name: str, m: int, n: int) -> tuple[tuple, tuple]:
    from src.rational.targets import sympy_target

    x, expr = sympy_target(name)
    a, b = pade_at_zero(expr, x, m, n, exact=False)
    return tuple(np.asarray(a).tolist()), tuple(np.asarray(b).tolist())


def pade_at_zero_numeric(name: str, m: int, n: int) -> tuple[NDArray, NDArray]:
    """目標関数名から float 係数の Padé 近似を返す薄いラッパ。

    sympy の級数展開は高次で非常に重く、スイープ中に同じ (name, m, n) が
    何度も要求されるためキャッシュする。返り値は毎回新しい配列にコピーして返す
    （呼び出し側が破壊的に書き換えても壊れないようにするため）。
    """
    a, b = _pade_at_zero_numeric_cached(name, m, n)
    return np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
