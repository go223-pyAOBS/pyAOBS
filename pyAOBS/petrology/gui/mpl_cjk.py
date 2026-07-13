"""Matplotlib / Qt 共用 CJK 字体探测与 rcParams 配置。"""

from __future__ import annotations

import os
import warnings

CJK_FONT_CANDIDATES: tuple[str, ...] = (
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Hiragino Sans GB",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "AR PL UMing CN",
    "Droid Sans Fallback",
)

_MPL_CONFIGURED = False


def _font_override() -> str:
    return (os.environ.get("PYAOBS_MPL_FONT") or os.environ.get("PYAOBS_GUI_FONT") or "").strip()


def _keyword_pick(names: set[str]) -> str | None:
    for name in sorted(names):
        nl = name.lower()
        if "noto" in nl and "cjk" in nl and "sans" in nl:
            return name
    for name in sorted(names):
        nl = name.lower()
        if "source han sans" in nl or "sourcehansans" in nl.replace(" ", ""):
            return name
    for name in sorted(names):
        nl = name.lower()
        if "wqy" in nl or "wenquanyi" in nl or "micro hei" in nl:
            return name
    return None


def pick_cjk_font_from_names(registered: set[str]) -> str | None:
    """从已注册字体族名中挑选 CJK 无衬线字体。"""
    override = _font_override()
    if override:
        if override in registered:
            return override
        low = override.lower()
        for name in registered:
            if low in name.lower():
                return name
    for candidate in CJK_FONT_CANDIDATES:
        if candidate in registered:
            return candidate
    return _keyword_pick(registered)


def pick_matplotlib_cjk_font() -> str | None:
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    return pick_cjk_font_from_names(names)


def configure_matplotlib_cjk() -> str | None:
    """在创建 Figure 之前调用，避免中文标题触发 DejaVu 缺字警告。"""
    global _MPL_CONFIGURED
    import matplotlib as mpl

    if _MPL_CONFIGURED:
        fam = mpl.rcParams.get("font.sans-serif", [])
        return fam[0] if fam else None

    chosen = pick_matplotlib_cjk_font()
    stack: list[str] = []
    if chosen:
        stack.append(chosen)
    stack.extend(c for c in CJK_FONT_CANDIDATES if c not in stack)
    stack.extend(["DejaVu Sans", "sans-serif"])
    mpl.rcParams["font.sans-serif"] = stack
    mpl.rcParams["axes.unicode_minus"] = False
    _MPL_CONFIGURED = True

    if chosen is None:
        warnings.warn(
            "未检测到 CJK 字体，matplotlib 中文可能显示为方框。"
            "Linux/WSL: sudo apt install fonts-noto-cjk；"
            "或设置环境变量 PYAOBS_MPL_FONT=字体名。",
            UserWarning,
            stacklevel=2,
        )
    return chosen


def pick_qt_font_family() -> str:
    """Qt 界面默认字体（Windows 仍优先 Segoe UI）。"""
    import sys

    if sys.platform == "win32":
        try:
            from PySide6.QtGui import QFontDatabase

            names = set(QFontDatabase.families())
            picked = pick_cjk_font_from_names(names)
            if picked:
                return picked
        except Exception:
            pass
        return "Segoe UI"

    try:
        from PySide6.QtGui import QFontDatabase

        names = set(QFontDatabase.families())
        picked = pick_cjk_font_from_names(names)
        if picked:
            return picked
        return QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).family()
    except Exception:
        return "sans-serif"
