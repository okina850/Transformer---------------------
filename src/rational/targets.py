"""近似の目標となる活性化関数（numpy 実装）と、その記号表現。

**目標関数の選び方に注意**: Hugging Face の GPT-2 が使う GELU は
tanh を使った近似版（``gelu_new``）であり、erf を使った厳密版とは
わずかに異なる。Phase 0 の項目3（GPT-2 の活性化を差し替えて perplexity を測る）
では、モデルが実際に使っている ``gelu_tanh`` を目標関数にしないと、
「有理近似の誤差」と「erf 版 vs tanh 版のずれ」が混ざってしまう。
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import erf, expit

# gelu_tanh の定数（Hendrycks & Gimpel の近似式）
_SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)
_GELU_CUBIC = 0.044715


def sigmoid(x: ArrayLike) -> NDArray[np.floating]:
    """ロジスティックシグモイド。scipy の ``expit`` はオーバーフローに強い。"""
    return expit(np.asarray(x, dtype=np.float64))


def tanh(x: ArrayLike) -> NDArray[np.floating]:
    return np.tanh(np.asarray(x, dtype=np.float64))


def silu(x: ArrayLike) -> NDArray[np.floating]:
    """SiLU / Swish(beta=1): :math:`x\\,\\sigma(x)`。SwiGLU の中身。"""
    x = np.asarray(x, dtype=np.float64)
    return x * expit(x)


def gelu_erf(x: ArrayLike) -> NDArray[np.floating]:
    """厳密版 GELU: :math:`x\\,\\Phi(x) = \\tfrac{x}{2}\\bigl(1+\\mathrm{erf}(x/\\sqrt2)\\bigr)`。"""
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def gelu_tanh(x: ArrayLike) -> NDArray[np.floating]:
    """tanh 近似版 GELU（PyTorch の ``approximate='tanh'`` / HF の ``gelu_new``）。

    :math:`\\tfrac{x}{2}\\bigl(1+\\tanh(\\sqrt{2/\\pi}(x+0.044715x^3))\\bigr)`
    GPT-2 が使うのはこちら。
    """
    x = np.asarray(x, dtype=np.float64)
    inner = _SQRT_2_OVER_PI * (x + _GELU_CUBIC * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


#: config ファイルから名前で参照するためのレジストリ
TARGETS: dict[str, Callable[[ArrayLike], NDArray[np.floating]]] = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "silu": silu,
    "swish": silu,  # 別名
    "gelu_erf": gelu_erf,
    "gelu_tanh": gelu_tanh,
    "gelu_new": gelu_tanh,  # HF での呼び名
}


def get_target(name: str) -> Callable[[ArrayLike], NDArray[np.floating]]:
    """名前から目標関数を取得する。未知の名前は候補を添えて即エラーにする。"""
    try:
        return TARGETS[name]
    except KeyError:
        raise KeyError(
            f"unknown target {name!r}. available: {sorted(TARGETS)}"
        ) from None


def sympy_target(name: str):
    """記号版の目標関数（x=0 での真の Padé 近似を求めるのに使う）。

    numpy 版とは別に持つ必要がある。Taylor 展開を厳密有理数で扱うため。
    """
    import sympy as sp

    x = sp.symbols("x")
    table = {
        "sigmoid": 1 / (1 + sp.exp(-x)),
        "tanh": sp.tanh(x),
        "silu": x / (1 + sp.exp(-x)),
        "swish": x / (1 + sp.exp(-x)),
        "gelu_erf": x * (1 + sp.erf(x / sp.sqrt(2))) / 2,
        "gelu_tanh": x
        * (1 + sp.tanh(sp.sqrt(2 / sp.pi) * (x + sp.Rational(44715, 1000000) * x**3)))
        / 2,
        "gelu_new": x
        * (1 + sp.tanh(sp.sqrt(2 / sp.pi) * (x + sp.Rational(44715, 1000000) * x**3)))
        / 2,
    }
    try:
        return x, table[name]
    except KeyError:
        raise KeyError(
            f"unknown symbolic target {name!r}. available: {sorted(table)}"
        ) from None
