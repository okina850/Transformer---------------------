"""多項式のモノミアル基底表現に関するユーティリティ。

**係数の順序は昇順で統一する**: ``coeffs[j]`` が :math:`x^j` の係数。
すなわち :math:`p(x) = \\sum_j c_j x^j`。

numpy の ``np.polyval`` は降順なので混同しないこと。昇順を採るのは
論文の記法（:math:`P(x)=\\sum_j a_j x^j`, :math:`Q(x)=1+\\sum_k b_k x^k`）と
一致させ、係数配列を論文の表とそのまま突き合わせられるようにするため。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def horner(coeffs: ArrayLike, x: ArrayLike) -> NDArray[np.floating]:
    """Horner 法で多項式を評価する。``coeffs`` は昇順。

    :math:`(((c_n x + c_{n-1})x + \\dots)x + c_0)` の形で評価するため、
    乗算回数が n 回で済み、素朴な :math:`\\sum c_j x^j` より丸め誤差も小さい。
    Phase 1 の GPU 実装（float32）でもこの形を使うので、Phase 0 の
    参照実装も同じ評価順序に揃えておく。
    """
    c = np.asarray(coeffs, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if c.size == 0:
        return np.zeros_like(x)
    out = np.full_like(x, c[-1])
    for cj in c[-2::-1]:
        out = out * x + cj
    return out


def polyval_naive(coeffs: ArrayLike, x: ArrayLike) -> NDArray[np.floating]:
    """素朴な :math:`\\sum_j c_j x^j` 評価。Horner 法との精度比較用（T4）。"""
    c = np.asarray(coeffs, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    powers = np.power.outer(x, np.arange(c.size))
    return powers @ c


def vander(x: ArrayLike, degree: int) -> NDArray[np.floating]:
    """昇順 Vandermonde 行列 :math:`V_{ij} = x_i^j`（``degree`` 次まで、列数 degree+1）。"""
    x = np.asarray(x, dtype=np.float64)
    return np.vander(x, degree + 1, increasing=True)


def vander_cond(x: ArrayLike, degree: int) -> float:
    """Vandermonde 行列の2ノルム条件数。

    モノミアル基底は次数が上がると急速に悪条件になる。この値を記録しておくことで、
    Phase 1 で高次数条件が不安定になったとき「最適化の問題」か
    「基底の悪条件による数値の問題」かを切り分ける材料にする
    （research_context §5.3 の重心形式への着手条件の判定材料）。
    """
    return float(np.linalg.cond(vander(x, degree)))


def polyder(coeffs: ArrayLike) -> NDArray[np.floating]:
    """多項式の導関数の係数（昇順）を返す。"""
    c = np.asarray(coeffs, dtype=np.float64)
    if c.size <= 1:
        return np.zeros(1)
    return c[1:] * np.arange(1, c.size)


def polyroots(coeffs: ArrayLike) -> NDArray[np.complexfloating]:
    """多項式の根を返す（昇順係数を受け取る）。

    用途は2つ:
    - 分母 ``1 + A(x)`` の実根 = 極の位置（D-free が本当に極を持つかの実測）
    - ``A(x)`` の実根 = ``1 + |A(x)|`` の微分不連続点（折れ曲がり）の位置
    """
    c = np.asarray(coeffs, dtype=np.float64)
    # 先頭（高次側）の 0 を落とさないと np.roots が次数を誤る
    nz = np.nonzero(np.abs(c) > 0)[0]
    if nz.size == 0:
        return np.array([], dtype=complex)
    c = c[: nz[-1] + 1]
    if c.size <= 1:
        return np.array([], dtype=complex)
    return np.roots(c[::-1])  # np.roots は降順


def real_roots_in(coeffs: ArrayLike, lo: float, hi: float, tol: float = 1e-9) -> NDArray[np.floating]:
    """区間 ``[lo, hi]`` 内の実根を昇順に並べて返す。

    ``tol`` は虚部をゼロとみなす閾値。数値誤差で微小な虚部が乗るため必要。
    """
    r = polyroots(coeffs)
    if r.size == 0:
        return np.array([])
    real = r[np.abs(r.imag) < tol].real
    inside = real[(real >= lo) & (real <= hi)]
    return np.sort(inside)
