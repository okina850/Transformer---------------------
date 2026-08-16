"""Day 1 の検算ゲート: PAU 論文 Appendix Table 6 の係数を再現できるか確認する。

論文 (Molina et al., ICLR 2020) の Table 6 は Sigmoid / Tanh / Swish について
[5,4] 次の初期化係数を厳密有理数で与えている。それらが「x=0 での真の Padé 近似」
であることを検証する。**これが通らない限り Phase 0 の以降の作業に進まない。**
論文の定義（分母に定数項がない、m が分子次数、n が分母次数）を読み違えていると、
以降の全比較が無効になるため。

実行:
    python scripts/verify_table6.py
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp

from src.rational.pade import pade_at_zero
from src.rational.targets import sympy_target

# 論文 Table 6 の [5,4] 列（a は a0..a5、b は b1..b4）。
# Swish 列は Swish のパラメータ beta について記号的に与えられている。
BETA = sp.symbols("beta")
TABLE6 = {
    "sigmoid": {
        "a": ["1/2", "1/4", "1/18", "1/144", "1/2016", "1/60480"],
        # 論文表の印字は "1/10008" だが、これは 1/1008 の誤植と思われる（下の検算参照）
        "b": ["0", "1/9", "0", "1/1008"],
    },
    "tanh": {
        "a": ["0", "1", "0", "1/9", "0", "1/945"],
        "b": ["0", "4/9", "0", "1/63"],
    },
    "swish": {
        "a": ["0", "1/2", "beta/4", "3*beta**2/56", "beta**3/168", "beta**4/3360"],
        "b": ["0", "3*beta**2/28", "0", "beta**4/1680"],
    },
}


def _parse(exprs: list[str]):
    return [sp.nsimplify(sp.sympify(e, locals={"beta": BETA})) for e in exprs]


def check(name: str, m: int = 5, n: int = 4) -> bool:
    x, expr = sympy_target(name)
    if name == "swish":
        # 論文の Swish は x*sigmoid(beta*x)。beta について記号的に解く。
        expr = x / (1 + sp.exp(-BETA * x))

    a, b = pade_at_zero(expr, x, m, n, exact=True)
    exp_a = _parse(TABLE6[name]["a"])
    exp_b = _parse(TABLE6[name]["b"])

    ok = True
    print(f"--- {name} [{m},{n}] ---")
    for label, got, want in (("a", a, exp_a), ("b", b, exp_b)):
        offset = 0 if label == "a" else 1
        for i, (g, w) in enumerate(zip(got, want)):
            same = sp.simplify(g - w) == 0
            ok &= bool(same)
            mark = "OK " if same else "NG "
            print(f"  {mark}{label}{i + offset}: computed={g}   table={w}")
    return ok


def main() -> int:
    all_ok = True
    for name in ("sigmoid", "tanh", "swish"):
        all_ok &= check(name)
        print()

    # 参考: 論文表の Sigmoid b4 は "1/10008" と印字されているが 1/1008 が正しい
    print("[note] Table 6 の Sigmoid b4 は '1/10008' と印字されているが、")
    print("       真の Pade 近似は 1/1008。桁の誤植（または PDF 抽出の乱れ）と判断した。")
    print(f"       1/1008 = {float(Fraction(1, 1008)):.10g},  "
          f"1/10008 = {float(Fraction(1, 10008)):.10g}")
    print()

    # 本研究の目標関数（GELU 系）は Table 6 に列がないので、係数だけ出しておく
    print("[info] Table 6 に GELU 列はない。本研究の目標関数の Pade[5/4] @ 0:")
    for name in ("silu", "gelu_tanh", "gelu_erf"):
        x, expr = sympy_target(name)
        a, b = pade_at_zero(expr, x, 5, 4, exact=False)
        print(f"  {name:10s} a={a.round(8).tolist()}")
        print(f"  {'':10s} b={b.round(8).tolist()}")

    print()
    print("RESULT:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
