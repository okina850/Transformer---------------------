r"""近似誤差の評価指標と、評価用グリッド・重み関数。

**設計方針: フィッティング用のグリッドと誤差評価用のグリッドを分ける。**
同じ点で当てはめて同じ点で誤差を測ると、点の間で誤差が大きくても見逃す
（機械学習でいう訓練データでの評価に相当する）。評価側は密なグリッドを使い、
かつフィット点と重ならないようにオフセットする。
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import simpson


# --------------------------------------------------------------------------
# グリッド
# --------------------------------------------------------------------------
def uniform_grid(lo: float, hi: float, n: int) -> NDArray[np.floating]:
    """等間隔グリッド。重みなし L2 内積を離散化するのに素直。"""
    return np.linspace(lo, hi, n)


def chebyshev_grid(lo: float, hi: float, n: int) -> NDArray[np.floating]:
    """第2種 Chebyshev 点（端点を含む）を ``[lo, hi]`` に写したもの。

    端に点が密になるため、区間端での誤差の跳ね上がり（Runge 的な挙動）を
    抑えたいときに使う。sup ノルムを重視する場合はこちら。
    """
    k = np.arange(n)
    t = np.cos(np.pi * k / (n - 1))  # [1, -1]
    return lo + (hi - lo) * (1.0 - t) / 2.0  # 昇順に直して写像


GRIDS: dict[str, Callable[[float, float, int], NDArray[np.floating]]] = {
    "uniform": uniform_grid,
    "chebyshev": chebyshev_grid,
}


def get_grid(name: str, lo: float, hi: float, n: int) -> NDArray[np.floating]:
    try:
        return GRIDS[name](lo, hi, n)
    except KeyError:
        raise KeyError(f"unknown grid {name!r}. available: {sorted(GRIDS)}") from None


def eval_grid(lo: float, hi: float, n: int, offset: bool = True) -> NDArray[np.floating]:
    """誤差評価用の密グリッド。

    ``offset=True`` のとき半格子分ずらし、フィット用の等間隔グリッドと
    点が重ならないようにする（点上だけ誤差が小さい解に騙されないため）。
    """
    x = np.linspace(lo, hi, n)
    if offset and n > 1:
        h = (hi - lo) / (n - 1)
        x = x + h / 2.0
        x = x[x <= hi]
    return x


# --------------------------------------------------------------------------
# 重み関数
# --------------------------------------------------------------------------
def weight_uniform(x: ArrayLike) -> NDArray[np.floating]:
    return np.ones_like(np.asarray(x, dtype=np.float64))


def weight_gaussian(x: ArrayLike, sigma: float = 1.0) -> NDArray[np.floating]:
    r"""正規分布密度に比例する重み :math:`\exp(-x^2/2\sigma^2)`。

    Phase 1 はスクラッチ学習であり、学習初期の pre-activation は
    LayerNorm 出力にランダム重みを掛けたものなのでおおむね正規分布に近い。
    Phase 1 用の初期化係数を作るときの重みとして使う。
    （GPT-2 の実測密度は分布が違うので Phase 1 には転用しない。Day 7 の分析用。）
    """
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-0.5 * (x / sigma) ** 2)


def make_weight(spec: dict | str | None) -> Callable[[ArrayLike], NDArray[np.floating]]:
    """設定ファイルの記述から重み関数を作る。

    ``"uniform"`` / ``{"kind": "gaussian", "sigma": 2.0}`` の形を受ける。
    """
    if spec is None or spec == "uniform":
        return weight_uniform
    if isinstance(spec, str):
        spec = {"kind": spec}
    kind = spec.get("kind", "uniform")
    if kind == "uniform":
        return weight_uniform
    if kind == "gaussian":
        sigma = float(spec.get("sigma", 1.0))
        return lambda x: weight_gaussian(x, sigma)
    raise KeyError(f"unknown weight kind {kind!r}. available: ['uniform', 'gaussian']")


# --------------------------------------------------------------------------
# 誤差指標
# --------------------------------------------------------------------------
def error_metrics(
    x: ArrayLike,
    target: ArrayLike,
    approx: ArrayLike,
    weights: ArrayLike | None = None,
) -> dict[str, float]:
    r"""近似誤差の各種指標をまとめて返す。

    Returns
    -------
    dict
        - ``sup``      : :math:`\max |f - F|`（sup ノルム / minimax 誤差）
        - ``argsup``   : sup を達成する x（区間端で出るのか内部で出るのかを見る）
        - ``l2``       : :math:`\sqrt{\int (f-F)^2 dx}`（Simpson 則）
        - ``rms``      : グリッド上の二乗平均平方根
        - ``mean_abs`` : 平均絶対誤差
        - ``sup_w`` / ``rms_w`` : 重み付き版（``weights`` 指定時のみ）

    L2 の数値積分に Simpson 則を使うのは、被積分関数が
    ``1+|A(x)|`` 型の分母により微分不連続点を持ちうるためである。
    Gauss-Legendre は滑らかさを仮定するので、折れ曲がりがあると精度が落ちる。
    """
    x = np.asarray(x, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    approx = np.asarray(approx, dtype=np.float64)

    err = approx - target
    abs_err = np.abs(err)

    # 発散・NaN があると max が NaN になるので明示的に扱う
    finite = np.isfinite(abs_err)
    n_nonfinite = int((~finite).sum())

    if finite.any():
        i = int(np.argmax(np.where(finite, abs_err, -np.inf)))
        sup = float(abs_err[i])
        argsup = float(x[i])
    else:
        sup, argsup = float("inf"), float("nan")

    safe = np.where(finite, err, 0.0)
    out: dict[str, float] = {
        "sup": sup,
        "argsup": argsup,
        "l2": float(np.sqrt(max(simpson(safe**2, x=x), 0.0))),
        "rms": float(np.sqrt(np.mean(safe**2))),
        "mean_abs": float(np.mean(np.abs(safe))),
        "n_nonfinite": n_nonfinite,
    }

    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        wsum = float(w.sum())
        out["sup_w"] = float(np.max(np.where(finite, abs_err * w, -np.inf)))
        out["rms_w"] = float(np.sqrt(np.sum(w * safe**2) / wsum)) if wsum > 0 else float("nan")

    return out


def smoothness_metrics(
    x: ArrayLike, y: ArrayLike, kink_points: ArrayLike | None = None
) -> dict[str, float]:
    r"""平滑性の定量化（T3）。

    ``1+|A(x)|`` 型の分母は :math:`A(x)=0` で微分が不連続になる（折れ曲がる）。
    その実害を数値で示すために2階差分を測る。

    Returns
    -------
    dict
        - ``max_abs_d2``  : 2階差分の絶対値の最大
        - ``tv_d1``       : 1階微分の全変動（折れがあると跳ね上がる）
        - ``n_kinks``     : ``kink_points`` の個数（区間内の折れ点の数）

    Notes
    -----
    折れ曲がりは「2階微分がデルタ関数になる」ので、離散2階差分の最大値は
    グリッド幅に依存して発散的に大きくなる。**したがってこの値はグリッドを
    揃えた条件間の相対比較にのみ使うこと。**絶対値そのものに意味はない。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    d1 = np.gradient(y, x)
    d2 = np.gradient(d1, x)
    finite2 = np.isfinite(d2)

    return {
        "max_abs_d2": float(np.max(np.abs(d2[finite2]))) if finite2.any() else float("inf"),
        "tv_d1": float(np.sum(np.abs(np.diff(d1[np.isfinite(d1)])))),
        "n_kinks": int(np.size(kink_points)) if kink_points is not None else 0,
    }
