r"""有理関数の係数フィッティング。

3つの手法を用意する。**なぜ複数必要かを明記しておく。**

1. :func:`fit_linearized` — 線形化最小二乗。
   :math:`\|P(x) - f(x)Q(x)\|` を最小化する。係数について線形なので閉形式で解ける。
   **ただし本当に最小化したいのは** :math:`\|P/Q - f\|` **であり、両者は違う**：
   線形化版は暗黙に :math:`Q(x)` で重み付けした誤差を測っているので、
   Q が大きい領域の誤差を過大評価する（＝そこを無駄に合わせにいく）。

2. :func:`fit_linearized` の ``sk_iters>0`` — Sanathanan–Koerner 反復。
   上のバイアスを消すための古典的な補正。前回の :math:`Q_{t-1}(x)` で
   各行を割ってから解き直す。数回で十分収束する。

3. :func:`fit_nonlinear` — 非線形最小二乗（Levenberg–Marquardt / trust region）。
   :math:`\|P/Q - f\|` を直接最小化する。
   **``1+|A|`` や ``1+A^2`` のような制約付きの分母形は 1・2 では扱えない**
   （線形化が成り立たない）ため、これが必須になる。
   局所解に落ちうるので複数の初期値から回して同一解に収束するかを確認する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from src.rational.denominators import Denominator, eval_rational, make_denominator
from src.rational.poly import horner, vander


@dataclass
class FitResult:
    """フィッティング結果。``a`` は a_0..a_m、``b`` は b_1..b_n（いずれも昇順）。"""

    a: NDArray[np.floating]
    b: NDArray[np.floating]
    method: str
    denom: str
    m: int
    n: int
    cost: float = float("nan")
    success: bool = True
    n_eval: int = 0
    init_source: str = ""
    info: dict = field(default_factory=dict)

    @property
    def n_params(self) -> int:
        """学習可能パラメータ数 = (m+1) + n。PAU[5,4] なら 10。"""
        return len(self.a) + len(self.b)

    def to_dict(self) -> dict:
        return {
            "a": np.asarray(self.a).tolist(),
            "b": np.asarray(self.b).tolist(),
            "method": self.method,
            "denom": self.denom,
            "m": self.m,
            "n": self.n,
            "n_params": self.n_params,
            "cost": self.cost,
            "success": self.success,
            "n_eval": self.n_eval,
            "init_source": self.init_source,
            "info": self.info,
        }


# --------------------------------------------------------------------------
# 1) 線形化最小二乗（+ Sanathanan-Koerner 反復）
# --------------------------------------------------------------------------
def fit_linearized(
    x: ArrayLike,
    f: ArrayLike,
    m: int,
    n: int,
    weights: ArrayLike | None = None,
    sk_iters: int = 0,
) -> FitResult:
    r"""線形化最小二乗で :math:`F=P/Q`、:math:`Q=1+A(x)` を当てはめる。

    残差 :math:`r_i = \sum_j a_j x_i^j - f_i\bigl(1 + \sum_k b_k x_i^k\bigr)` を
    最小二乗で解く。設計行列は :math:`[\,V_m \mid -f\odot X_n\,]`、右辺は :math:`f`。

    Parameters
    ----------
    sk_iters
        Sanathanan–Koerner 反復の回数。0 なら素の線形化解。
        1 以上なら前回の :math:`Q(x)` で各行を割って解き直す
        （:math:`\|P/Q-f\|` の最小化に近づける）。

    Notes
    -----
    得られる分母は **制約なしの** :math:`1+A(x)` である（``free`` 設計）。
    ``abs`` / ``sqrt`` / ``square`` の解はここから
    :func:`fit_nonlinear` で仕上げること。
    """
    x = np.asarray(x, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=np.float64)
    sw = np.sqrt(w)

    Vm = vander(x, m)                 # (N, m+1)
    Xn = vander(x, n)[:, 1:]          # (N, n)  x^1..x^n（定数項を除く）

    a = np.zeros(m + 1)
    b = np.zeros(n)
    scale = np.ones_like(x)
    residual_norms: list[float] = []

    for _ in range(max(sk_iters, 0) + 1):
        s = sw / scale
        design = np.hstack([Vm, -(f[:, None] * Xn)]) * s[:, None]
        rhs = f * s
        theta, *_ = np.linalg.lstsq(design, rhs, rcond=None)
        a, b = theta[: m + 1], theta[m + 1 :]
        residual_norms.append(float(np.linalg.norm(design @ theta - rhs)))
        # 次の反復用に Q(x) = 1 + A(x) を計算。0 に近いと割り算が暴れるので下限を設ける。
        q = 1.0 + (Xn @ b if n > 0 else 0.0)
        scale = np.where(np.abs(q) < 1e-8, 1e-8, np.abs(q))

    approx = horner(a, x) / (1.0 + (Xn @ b if n > 0 else 0.0))
    cost = float(0.5 * np.sum(w * (approx - f) ** 2))

    return FitResult(
        a=a,
        b=b,
        method=f"linearized(sk={sk_iters})",
        denom="free",
        m=m,
        n=n,
        cost=cost,
        info={"sk_residual_norms": residual_norms},
    )


# --------------------------------------------------------------------------
# 2) 非線形最小二乗（制約付き分母に対応）
# --------------------------------------------------------------------------
def _residual_fn(
    x: NDArray, f: NDArray, sw: NDArray, m: int, denom: Denominator
) -> Callable[[NDArray], NDArray]:
    def residual(theta: NDArray) -> NDArray:
        a, b = theta[: m + 1], theta[m + 1 :]
        q = denom.q(b, x)
        # free 設計では極を踏みうる。inf を返すと optimizer が壊れるので
        # 大きな有限値に置き換え、「その方向は悪い」とだけ伝える。
        q = np.where(np.abs(q) < 1e-8, np.sign(q) * 1e-8 + 1e-8, q)
        r = (horner(a, x) / q - f) * sw
        return np.where(np.isfinite(r), r, 1e6)

    return residual


def fit_nonlinear(
    x: ArrayLike,
    f: ArrayLike,
    m: int,
    n: int,
    denom: Denominator | str | dict,
    weights: ArrayLike | None = None,
    inits: list[tuple[NDArray, NDArray, str]] | None = None,
    max_nfev: int = 20000,
) -> tuple[FitResult, list[FitResult]]:
    r""":math:`\|P/Q-f\|` を直接最小化する。制約付き分母 (``abs``/``sqrt``/``square``) 用。

    Parameters
    ----------
    inits
        ``(a0, b0, ラベル)`` のリスト。複数与えると全部から回して、
        **最良解と全解を返す**。CLAUDE.md の「seed 最低3つ」の趣旨を
        「初期値を3通り振って同一解に収束するか確認する」形で満たすための仕組み。

    Returns
    -------
    (best, all_results)
        ``all_results`` の cost を見比べて、同一解に収束しているかを判定すること。
    """
    x = np.asarray(x, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=np.float64)
    sw = np.sqrt(w)
    d = denom if isinstance(denom, Denominator) else make_denominator(denom)

    if not inits:
        inits = [(np.zeros(m + 1), np.zeros(n), "zeros")]

    residual = _residual_fn(x, f, sw, m, d)
    results: list[FitResult] = []

    for a0, b0, label in inits:
        theta0 = np.concatenate([np.asarray(a0, dtype=np.float64), np.asarray(b0, dtype=np.float64)])
        sol = least_squares(residual, theta0, method="trf", max_nfev=max_nfev)
        a, b = sol.x[: m + 1], sol.x[m + 1 :]
        results.append(
            FitResult(
                a=a,
                b=b,
                method="nonlinear",
                denom=d.name,
                m=m,
                n=n,
                cost=float(sol.cost),
                success=bool(sol.success),
                n_eval=int(sol.nfev),
                init_source=label,
                info={"status": int(sol.status), "optimality": float(sol.optimality)},
            )
        )

    best = min(results, key=lambda r: r.cost)
    return best, results


def make_inits(
    m: int,
    n: int,
    linear_fit: FitResult | None = None,
    pade_coeffs: tuple[NDArray, NDArray] | None = None,
    n_random: int = 1,
    seed: int = 0,
    scale: float = 0.1,
) -> list[tuple[NDArray, NDArray, str]]:
    """非線形最小二乗の初期値を3系統そろえる。

    1. 線形化最小二乗の解（区間全体を見た解。最も素性が良い）
    2. x=0 での真の Padé 近似（論文の初期化。原点近傍で厳密）
    3. ランダム摂動（局所解に落ちていないかの確認用）

    出自の違う初期値から同じ解に収束すれば、大域解と見なしてよい根拠になる。
    """
    inits: list[tuple[NDArray, NDArray, str]] = []

    if linear_fit is not None:
        inits.append((linear_fit.a.copy(), linear_fit.b.copy(), "linearized"))

    if pade_coeffs is not None:
        a_p, b_p = pade_coeffs
        a_p = np.asarray(a_p, dtype=np.float64)
        b_p = np.asarray(b_p, dtype=np.float64)
        # 次数が違う場合は切り詰め／0 埋めして形を合わせる
        a0 = np.zeros(m + 1)
        a0[: min(len(a_p), m + 1)] = a_p[: m + 1]
        b0 = np.zeros(n)
        b0[: min(len(b_p), n)] = b_p[:n]
        inits.append((a0, b0, "pade@0"))

    rng = np.random.default_rng(seed)
    base_a = inits[0][0] if inits else np.zeros(m + 1)
    base_b = inits[0][1] if inits else np.zeros(n)
    for i in range(n_random):
        inits.append(
            (
                base_a + rng.normal(0.0, scale, m + 1),
                base_b + rng.normal(0.0, scale, n),
                f"random{i}(seed={seed})",
            )
        )

    return inits


def evaluate(result: FitResult, x: ArrayLike, denom: Denominator | None = None) -> NDArray:
    """フィット結果を x 上で評価する。"""
    d = denom if denom is not None else make_denominator(result.denom)
    return eval_rational(result.a, result.b, x, d)
