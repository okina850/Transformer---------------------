"""実験成果物に再現用メタ情報を同梱するためのユーティリティ。

CLAUDE.md の再現性ルール
「設定・乱数シード・git commit hash・実行日時を成果物と一緒に保存する」
に対応する。すべての scripts/*.py はここを経由して出力ディレクトリを作る。

使い方::

    from src.utils.run_meta import new_run_dir, save_run_meta, set_seed

    cfg = load_yaml("configs/phase0_fit.yaml")
    seed = set_seed(cfg["seed"])
    run_dir = new_run_dir("phase0/fit", tag="gelu_tanh")
    save_run_meta(run_dir, config=cfg, seed=seed)
    # ... 実験本体。成果物は run_dir 配下に書く ...
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# src/utils/run_meta.py -> src/utils -> src -> リポジトリルート
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"


# --------------------------------------------------------------------------
# git
# --------------------------------------------------------------------------
def _git(*args: str) -> str | None:
    """リポジトリルートで git コマンドを実行し、標準出力を返す。失敗時は None。"""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_info() -> dict[str, Any]:
    """commit hash / ブランチ / 作業ツリーが汚れているかを返す。

    ``is_dirty`` が True の場合、その実験は「どの commit か」だけでは再現できない。
    そのため :func:`save_run_meta` は未コミット差分を ``uncommitted.diff`` に保存する。
    """
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "commit_short": commit[:12] if commit else None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "is_dirty": bool(status) if status is not None else None,
        "dirty_files": status.splitlines() if status else [],
    }


# --------------------------------------------------------------------------
# 実行環境
# --------------------------------------------------------------------------
def _package_versions() -> dict[str, str]:
    """主要パッケージのバージョンを取得する（未インストールなら省略）。"""
    names = [
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "torch",
        "transformers",
        "datasets",
        "yaml",
    ]
    from importlib.metadata import PackageNotFoundError, version

    # import 名とパッケージ名が違うものの対応
    dist_name = {"yaml": "pyyaml"}

    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = version(dist_name.get(name, name))
        except PackageNotFoundError:
            continue
    return versions


def gpu_info() -> dict[str, Any]:
    """GPU 情報を返す。torch 未インストール／CPU 実行なら available=False。

    torch の import は数秒かかるため、既に import 済みのときだけ調べる……
    のではなく確実性を優先してここで import する。近似計算のみのスクリプトで
    コストが気になる場合は ``include_gpu=False`` を使うこと。
    """
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not installed"}

    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_build_version": torch.version.cuda,
        "available": bool(torch.cuda.is_available()),
    }
    if info["available"]:
        props = torch.cuda.get_device_properties(0)
        info["device_name"] = props.name
        info["total_memory_gb"] = round(props.total_memory / 1024**3, 2)
        info["capability"] = f"{props.major}.{props.minor}"
    return info


def env_info(include_gpu: bool = True) -> dict[str, Any]:
    """再現に必要な実行環境情報をまとめる。"""
    info: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": _package_versions(),
    }
    if include_gpu:
        info["gpu"] = gpu_info()
    return info


# --------------------------------------------------------------------------
# 乱数シード
# --------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True) -> int:
    """Python / numpy / torch の乱数シードをまとめて固定し、その seed を返す。

    Phase 0 は学習を伴わないため本質的に決定論的だが、非線形最小二乗の初期値生成
    などに乱数を使う。どの seed で得た係数かを必ず記録するためにここを通す。

    ``deterministic=True`` のとき cuDNN の非決定的アルゴリズム選択を禁止する。
    （cuDNN は畳み込み等で実行ごとに違う高速アルゴリズムを選ぶことがあり、
    結果が微妙にぶれる。速度は落ちるが再現性を優先する。）
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


# --------------------------------------------------------------------------
# 出力ディレクトリとメタ情報
# --------------------------------------------------------------------------
def new_run_dir(subdir: str, tag: str = "", root: Path | None = None) -> Path:
    """``results/<subdir>/<YYYYmmdd-HHMMSS>_<tag>/`` を作成して返す。

    タイムスタンプ先頭なので ``ls`` が時系列順に並ぶ。
    """
    root = root or RESULTS_ROOT
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_{tag}" if tag else stamp
    run_dir = root / subdir / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run_meta(
    run_dir: Path,
    config: dict[str, Any],
    seed: int | None = None,
    include_gpu: bool = True,
    extra: dict[str, Any] | None = None,
) -> Path:
    """``meta.json`` と ``config.yaml`` を run_dir に書き出す。

    作業ツリーが汚れている場合は ``uncommitted.diff`` も保存する。
    これがないと「commit hash は記録したが実際に走ったコードは別物」になりうる。
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    git = git_info()
    meta: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "git": git,
        "env": env_info(include_gpu=include_gpu),
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
    }
    if extra:
        meta["extra"] = extra

    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 設定ファイルは実験の主入力なので必ず現物を残す
    try:
        import yaml

        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    except ImportError:
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if git.get("is_dirty"):
        diff = _git("diff", "HEAD")
        if diff:
            (run_dir / "uncommitted.diff").write_text(diff, encoding="utf-8")

    return run_dir


def load_yaml(path: str | Path) -> dict[str, Any]:
    """YAML 設定ファイルを読む（ハードコード値で実験しないための入口）。"""
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
