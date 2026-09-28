# -*- coding: utf-8 -*-
"""在系统文件管理器中打开路径（兼容 Windows / WSL / macOS / Linux）。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple, Union


PathLike = Union[str, os.PathLike[str]]


def mnt_to_windows_path(path_text: str) -> Optional[str]:
    """``/mnt/d/foo`` → ``D:\\foo``；非 WSL 挂载路径返回 None。"""
    s = str(path_text).strip()
    if len(s) >= 7 and s.startswith("/mnt/") and s[5].isalpha() and s[6] == "/":
        drive = s[5].upper()
        rest = s[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    return None


def open_path_in_file_manager(path: PathLike) -> Tuple[bool, str]:
    """打开文件或目录。

    Returns:
        (ok, message) — 失败时 message 为可读说明（含路径）。
    """
    target = Path(path).expanduser()
    try:
        target = target.resolve()
    except Exception:
        target = Path(path)
    target_str = str(target)
    if not target.exists():
        return False, f"路径不存在：{target_str}"

    # 原生 Windows
    if hasattr(os, "startfile") and sys.platform.startswith("win"):
        try:
            os.startfile(target_str)  # type: ignore[attr-defined]
            return True, target_str
        except Exception as exc:
            return False, f"打开失败：{exc}\n路径：{target_str}"

    # WSL：优先用 Windows 资源管理器，避免无桌面环境下 xdg-open 刷屏
    win_path = mnt_to_windows_path(target_str)
    if win_path:
        explorer = shutil.which("explorer.exe")
        if explorer:
            try:
                # explorer 对目录常返回非 0，仍可能已打开；吞掉输出
                subprocess.Popen(
                    [explorer, win_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True, target_str
            except Exception:
                pass

    if sys.platform == "darwin":
        try:
            subprocess.Popen(["open", target_str], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True, target_str
        except Exception as exc:
            return False, f"打开失败：{exc}\n路径：{target_str}"

    opener = shutil.which("xdg-open")
    if opener:
        try:
            subprocess.Popen(
                [opener, target_str],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True, target_str
        except Exception as exc:
            return False, f"打开失败：{exc}\n路径：{target_str}"

    return False, f"当前环境无法自动打开目录，请手动打开：\n{target_str}"
