# experiment_log.md

> 全実験・考察・失敗の記録。CLAUDE.md の指示により、
> **没になったアイデアも理由とともに残す**（同じ轍を踏まないため）。
>
> 記法: 各エントリは `## YYYY-MM-DD | Phase X | 見出し` で始める。

---

## 2026-08-16 | Phase 0 | Day 0: 環境整備とリポジトリ初期化

### やったこと

**1. 実行環境の確認**

| 項目 | 確認結果 |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 12GB / Driver 591.86 / CUDA 13.1 / 空き約10.8GB |
| グローバル Python | 3.13.14（Microsoft Store 版）。**torch は 2.13.0+cpu で CUDA 使用不可** |
| 既存 venv `C:\Users\user\venvs\gpif` | torch 2.13.0+**cu130**（`cuda.is_available()==True`）。3.5GB、51パッケージ |

**2. venv の方針決定 → 新規作成**

既存の `gpif` venv を流用せず、`C:\Users\user\venvs\rational_activation` を新規作成した。

判断理由:
- `gpif` は GPIF Finance Awards（research_context §1 の副次的目的、本テーマとは別件で
  「現時点では応募は想定しない」と明記）用の環境。名前が別プロジェクトを指しており、
  今後そちらの都合でパッケージ構成が変わりうる。再現性をそこに依存させるのは危険。
- `gpif` には `datasets` / `matplotlib` / `ipykernel` が入っておらず、
  どのみち追加インストールが必要だった（流用のメリットが小さい）。
- venv 本体は C ドライブに置き、G ドライブ（Google Drive 同期対象）には入れない。
  venv は小ファイルが大量にできるため、Drive 同期に載せると同期が重くなる。

`torch==2.13.0+cu130` は `gpif` と同じホイールを明示指定した
（そのビルドが本機の CUDA 13.1 ドライバで動くことが実証済みのため、互換性検証の手間を省ける）。

**3. リポジトリ初期化**

- `git init` 実行（CLAUDE.md が成果物への commit hash 同梱を要求しているため事実上の前提条件）。
- `core.autocrlf false` を設定（Windows の CRLF 自動変換で差分が汚れるのを防ぐ）。
- `.gitignore` 作成。方針: **図・表・json などの成果物は追跡する**が、
  チェックポイント（`*.pt` 等）と HF キャッシュは追跡しない。
  CLAUDE.md の「学習曲線・中間ログは必ず保存」に対応するため、ログ類は git に入れる。

**4. `src/utils/run_meta.py` 作成**

再現性ルール（設定・seed・git commit hash・実行日時を成果物と同梱）を1箇所に集約。
以降の `scripts/*.py` はすべてここを経由して出力ディレクトリを作る。

設計上の判断:
- **未コミット差分を `uncommitted.diff` として保存する。**
  commit hash だけ記録しても作業ツリーが汚れていれば実際に走ったコードは再現できない。
  開発中は必ず dirty になるので、これがないとメタ情報が実質無意味になる。
- `set_seed()` で `cudnn.deterministic=True` / `benchmark=False` を設定。
  cuDNN は実行ごとに違う高速アルゴリズムを選ぶことがあり結果がぶれるため、速度より再現性を優先。
- GPU 情報の取得は `include_gpu` フラグで切れるようにした。
  Phase 0 の近似計算部分は numpy のみで完結し、torch の import（数秒）が無駄になるため。

**5. `docs/research_context.md` §12 の記述修正**

参照 PDF 名を実ファイル名に修正:
`inverse_transform_sampling.pdf` → `FunctionApproximationMethodsForTheInverseTransformSampling__arxiv.pdf`

### 決まったこと（Phase 0 のスコープ確定）

ユーザー承認により以下が確定した。

- **論点A（裾の挙動を揃えた変種を追加）: 採用。**
  `Q(x)=1+A(x)²` は分母の実効次数が `2m` になるため、同じ `(n,m)` を使うと
  PAU 型は両側で線形発散、平滑型は両側で 0 に減衰と、**近似する関数クラスが別物になる**。
  これを揃えないと Phase 1 で平滑型が負けたとき、原因が「平滑性」か「裾の形」か切り分け不能。
  → `deg P = 2m+1`（`deg P − deg Q = 1` を保つ）変種も誤差評価する。
- **論点B（GPT-2 実測 pre-activation 密度による重み付き近似）: Day 7 の余力タスクに格下げ。**
  理由: Phase 1 はスクラッチ学習であり、学習初期の pre-activation 分布は GPT-2 のそれと
  異なるため、この重みで作った係数を Phase 1 の初期化にそのまま使えない。
  1週間の成果物としての優先度が低いという判断。
- 計算コスト見積もり承認（GPU 使用は最大 30〜40分。**まず1条件で通してから全条件に上げる**）。
- seed の扱い承認: Phase 0 は学習がなく決定論的なので、
  非線形最小二乗の**初期値を3通り試して同一解に収束することを確認**する形で
  CLAUDE.md の「seed 最低3つ」の趣旨を満たす。

### 次にやること

Day 1: `targets.py` / `poly.py` / `metrics.py` と線形化最小二乗を実装し、
`docs/refs/pade_activation.pdf` Appendix Table 6 の [5,4] 係数と突き合わせて検算する。
**この検算が通らない限り先に進まない**（論文の定義の読み違いは以降の全比較を無効化するため）。

---

## アイデア置き場

> 実装せずにここに記録する。現在のフェーズを完了させることを優先する（CLAUDE.md スコープ管理）。

- **(Day 7 送り) GPT-2 実測 pre-activation 密度による重み付き有理近似。**
  近似区間 [-6,6] を一様に重み付けするのが最適とは限らない。実測分布を重みにすれば
  perplexity 劣化を減らせる可能性がある。Phase 0 の分析としては価値があるが、
  Phase 1 の初期化には転用できない（分布が違う）。
- **(条件付き) 重心座標（barycentric）表現によるパラメトライズ。**
  research_context §5.3 の通り、Phase 1 で「次数を上げると学習が不安定になる」現象が
  実際に観測された場合に限り着手する。Phase 0 では条件数と float32 での丸め誤差を
  測定しておき、着手条件の判定材料にする（測定のみ。実装はしない）。
- **(任意) 有理 minimax（Remez / AAA）による参照解。**
  線形化最小二乗の解が sup ノルム最適からどれだけ離れているかの下限を与える。
  ※ これは「参照解を作る道具」であり、活性化関数自体を重心形式でパラメトライズする
  §5.3 の項目とは別物。混同しないこと。

---

## 没になったアイデア

> まだなし。
