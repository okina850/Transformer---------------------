r"""Phase 0 の最終成果物: Phase 1 に渡す初期化係数を1ファイルにまとめる。

**[-3,3] と [-6,6] の両方を渡す。** Day 5 の差し替え実験は [-3,3] を
破滅的と判定したが、その条件は RAFT の設定そのものであり、
RAFT は実際には成功している。差し替え実験は係数を凍結しており、
学習を伴う Phase 1 の結果を予測しない（``docs/research_context.md`` §6 の限界を参照）。
したがって片方を落とす根拠にはならない。

出力: ``results/phase0/phase1_init.json``（リポジトリ直下の固定パス。
Phase 1 のコードがタイムスタンプ付きディレクトリを探さなくて済むように）

実行:
    python scripts/export_phase1_init.py
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.rational.denominators import make_denominator
from src.rational.poly import vander_cond
from src.utils.run_meta import REPO_ROOT, git_info

#: Phase 1 の各条件が使う係数。research_context 改訂3 の 2x2。
PHASE1_CONDITIONS = [
    ("R1", "gelu_tanh|paper|m5n4|abs", "abs", "paper", "safe PAU x [-3,3]（RAFT の完全再現）"),
    ("R2", "gelu_tanh|wide|m5n4|abs", "abs", "wide", "safe PAU x [-6,6]"),
    ("R3", "gelu_tanh|paper|m5n4|sqrt", "sqrt", "paper", "平滑分母 x [-3,3]"),
    ("R4", "gelu_tanh|wide|m5n4|sqrt", "sqrt", "wide", "平滑分母 x [-6,6]"),
]

#: 参考として残すもの（Phase 1 の主要条件ではない）
REFERENCE_KEYS = [
    ("pade_at_0_paper", "gelu_tanh|paper|m5n4|pade@0", "PAU 論文どおりの初期化（x=0 の真の Padé）"),
    ("abs_wide_m7n6", "gelu_tanh|wide|m7n6|abs", "Day 5 の切り分け条件 R5（14 パラメータ）"),
]


def latest(pattern: str) -> str:
    hits = sorted(glob.glob(str(REPO_ROOT / pattern)))
    if not hits:
        raise FileNotFoundError(f"見つからない: {pattern}")
    return hits[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "results/phase0/phase1_init.json"))
    args = ap.parse_args()

    coeff_path = latest("results/phase0/fit/*/coefficients.json")
    ppl_path = latest("results/phase0/ppl/*/ppl_results.csv")
    preact_path = latest("results/phase0/preact/*/preact_stats.csv")
    coeffs = json.loads(Path(coeff_path).read_text(encoding="utf-8"))

    import pandas as pd

    ppl = pd.read_csv(ppl_path)
    preact = pd.read_csv(preact_path)

    conditions: dict[str, dict] = {}
    for name, key, denom, iname, desc in PHASE1_CONDITIONS:
        c = coeffs[key]
        x = np.linspace(float(c["lo"]), float(c["hi"]), 2001)
        d = make_denominator({"kind": denom, "eps": 1e-3})
        row = ppl[ppl["key"] == key]
        conditions[name] = {
            "description": desc,
            "denom": denom,
            "eps": 1e-3 if denom == "sqrt" else None,
            "interval": [c["lo"], c["hi"]],
            "interval_name": iname,
            "m": c["m"],
            "n": c["n"],
            "n_params": c["n_params"],
            "a": c["a"],
            "b": c["b"],
            "fit_sup_error": c["sup"],
            "fit_l2_error": c["l2"],
            "cond_vander": vander_cond(x, int(c["m"])),
            "effective_denominator_degree": d.effective_degree(int(c["n"])),
            "tail_exponent": d.tail_exponent(int(c["m"]), int(c["n"])),
            "frozen_swap_ppl": float(row["ppl"].iloc[0]) if len(row) else None,
        }

    references: dict[str, dict] = {}
    for name, key, desc in REFERENCE_KEYS:
        c = coeffs.get(key)
        if c is None:
            continue
        references[name] = {
            "description": desc, "key": key,
            "interval": [c["lo"], c["hi"]], "m": c["m"], "n": c["n"],
            "n_params": c["n_params"], "a": c["a"], "b": c["b"],
        }

    payload = {
        "_README": (
            "Phase 0 の最終成果物。Phase 1 の R1〜R4 の初期化係数。"
            "係数は昇順 (a[j] が x^j の係数、b[k] が x^(k+1) の係数、A に定数項はない)。"
        ),
        "_LIMITATION": (
            "frozen_swap_ppl は Day 5 の差し替え実験（係数凍結・学習なし）の結果であり、"
            "Phase 1 の結果を予測しない。R1 は RAFT と同一設定で frozen_swap_ppl は "
            "破滅的だが、RAFT は実際にはこの設定で GELU を上回っている。"
            "条件選択の根拠に使わないこと。詳細は docs/research_context.md §6。"
        ),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_info().get("commit"),
        "sources": {
            "coefficients": str(Path(coeff_path).relative_to(REPO_ROOT)),
            "swap_perplexity": str(Path(ppl_path).relative_to(REPO_ROOT)),
            "preact_stats": str(Path(preact_path).relative_to(REPO_ROOT)),
        },
        "target_function": "gelu_tanh",
        "baseline_gpt2_ppl": float(ppl.loc[ppl["condition"] == "baseline", "ppl"].iloc[0]),
        "preact_reference": {
            "note": "学習済み GPT-2 small の実測。Phase 1 の学習初期はこれより裾が軽いはず。",
            "std_mean": float(preact["std"].mean()),
            "iqr_mean": float(preact["iqr"].mean()),
            "frac_abs_gt_3": float((preact["frac_abs_gt_3"] * preact["count"]).sum() / preact["count"].sum()),
            "frac_abs_gt_6": float((preact["frac_abs_gt_6"] * preact["count"]).sum() / preact["count"].sum()),
            "max_abs": float(preact[["min", "max"]].abs().max().max()),
        },
        "conditions": conditions,
        "references": references,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote -> {out}")
    for name, c in conditions.items():
        print(f"  {name}: {c['denom']:5s} {str(c['interval']):12s} "
              f"sup={c['fit_sup_error']:.3e} cond={c['cond_vander']:8.1f} "
              f"tail=x^{c['tail_exponent']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
