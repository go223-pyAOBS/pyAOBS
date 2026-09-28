# -*- coding: utf-8 -*-
"""Qt 平台启动辅助（Wayland / HiDPI 等）。"""

from __future__ import annotations

import os
import sys


def prefer_xcb_on_wayland(*, force: bool = False) -> bool:
    """在 Linux Wayland 会话下优先改用 xcb，规避常见协议崩溃。

    典型错误::
        xdg_wm_base: error 4: xdg_surface buffer ... does not match
        the configured maximized state ...
        The Wayland connection experienced a fatal error: Protocol error

    仅在尚未设置 ``QT_QPA_PLATFORM`` 时生效（除非 ``force=True``）。
    用户可显式 ``export QT_QPA_PLATFORM=wayland`` 保留原生 Wayland。

    Returns:
        True 表示本次写入了 ``QT_QPA_PLATFORM=xcb``。
    """
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return False
    if not force:
        existing = (os.environ.get("QT_QPA_PLATFORM") or "").strip()
        if existing:
            return False
    wayland = bool(os.environ.get("WAYLAND_DISPLAY")) or (
        (os.environ.get("XDG_SESSION_TYPE") or "").strip().lower() == "wayland"
    )
    if not wayland and not force:
        return False
    # X11 可用时才切 xcb，否则强切可能直接起不来
    if not force and not (os.environ.get("DISPLAY") or "").strip():
        return False
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    return True
