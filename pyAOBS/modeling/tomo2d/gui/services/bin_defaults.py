"""默认 bin_path：环境变量 → 本仓库 src/build-tomo2d（若存在可执行文件）。"""

from __future__ import annotations

import os
from pathlib import Path

# modeling/tomo2d/gui/services/bin_defaults.py → ../../src/build-tomo2d
_PKG_BUILD = (
    Path(__file__).resolve().parents[2] / "src" / "build-tomo2d"
)


def _looks_like_bin_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    for name in ("tt_inverse", "tt_forward", "gen_smesh"):
        if (path / name).is_file() or (path / f"{name}.exe").is_file():
            return True
    return False


def default_bin_path() -> str:
    """
    解析 GUI / TomoAnd 默认二进制目录（POSIX 斜杠字符串）。

    顺序：
    1. ``PYAOBS_TOMO2D_BIN`` / ``TOMO2D_BIN``
    2. 包内 ``modeling/tomo2d/src/build-tomo2d``（含 tt_inverse 等时）
    """
    env = (os.getenv("PYAOBS_TOMO2D_BIN") or os.getenv("TOMO2D_BIN") or "").strip()
    if env:
        return Path(env).expanduser().as_posix()
    if _looks_like_bin_dir(_PKG_BUILD):
        return _PKG_BUILD.as_posix()
    # 目录存在但尚未编译时仍给出推荐路径，便于用户一眼看到该填哪
    if _PKG_BUILD.is_dir() or _PKG_BUILD.parent.is_dir():
        return _PKG_BUILD.as_posix()
    return ""
