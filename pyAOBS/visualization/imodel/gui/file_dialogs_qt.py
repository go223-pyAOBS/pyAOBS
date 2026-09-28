"""Qt file-dialog helpers（图件保存：过滤器 + 按所选类型自动补全后缀）。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

# 与主窗 Save Figure 一致；首项为默认类型
FIGURE_SAVE_FILTER = (
    "PNG (*.png);;JPEG (*.jpg *.jpeg);;PDF (*.pdf);;"
    "PostScript (*.ps);;EPS (*.eps);;TIFF (*.tif *.tiff);;SVG (*.svg);;All (*.*)"
)

_FIGURE_FILTER_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("PNG", ".png"),
    ("JPEG", ".jpg"),
    ("JPG", ".jpg"),
    ("PDF", ".pdf"),
    ("POSTSCRIPT", ".ps"),
    ("EPS", ".eps"),
    ("TIFF", ".tif"),
    ("TIF", ".tif"),
    ("SVG", ".svg"),
)

_KNOWN_FIGURE_EXTS = frozenset(
    {".png", ".jpg", ".jpeg", ".pdf", ".ps", ".eps", ".tif", ".tiff", ".svg"}
)


def suffix_from_figure_filter(selected_filter: str, *, default_ext: str = ".png") -> str:
    """从选中的保存过滤器得到应追加的后缀。"""
    filt = (selected_filter or "").upper()
    if "ALL" in filt and "*." not in (selected_filter or ""):
        return default_ext if default_ext.startswith(".") else f".{default_ext}"
    for token, suffix in _FIGURE_FILTER_SUFFIXES:
        if token in filt:
            return suffix
    # 兜底：从 *.ext 解析
    try:
        from pyAOBS.utils.qt_file_dialog import suffixes_from_filter

        found = suffixes_from_filter(selected_filter)
        if found:
            return found[0]
    except Exception:
        pass
    ds = default_ext if default_ext.startswith(".") else f".{default_ext}"
    return ds


def normalize_save_path_for_filter(
    path: str,
    selected_filter: str,
    *,
    default_ext: str = ".png",
) -> str:
    """用户未写后缀时，按所选过滤器自动补全；已有已知图件后缀则保留。"""
    text = str(path or "").strip()
    if not text:
        return text
    _base, ext = os.path.splitext(text)
    if ext and ext.lower() in _KNOWN_FIGURE_EXTS:
        return text
    if ext:
        # 非图件后缀（如 .txt）仍保留，避免误改
        return text
    return text + suffix_from_figure_filter(selected_filter, default_ext=default_ext)


def prompt_save_figure_path(
    parent: Optional[QWidget],
    *,
    caption: str = "Save Figure",
    directory: Union[str, Path, None] = None,
    default_stem: str = "figure",
    filter: str = FIGURE_SAVE_FILTER,
) -> str:
    """弹出保存对话框；取消返回空串。路径已按所选类型补全后缀。"""
    start = Path(directory) if directory else Path.cwd()
    if start.is_dir():
        start = start / default_stem
    path, selected = QFileDialog.getSaveFileName(
        parent,
        str(caption),
        str(start),
        str(filter),
    )
    if not path:
        return ""
    return normalize_save_path_for_filter(path, selected)


def save_matplotlib_figure(
    parent: Optional[QWidget],
    fig: Any,
    *,
    caption: str = "Save Figure",
    default_stem: str = "figure",
    directory: Union[str, Path, None] = None,
    dpi: int = 300,
    show_message: bool = True,
) -> str:
    """
    保存 Matplotlib Figure：支持 png/jpg/pdf/ps/eps/tif/svg；
    用户只输入文件名时按所选过滤器自动加后缀。
    成功返回路径，取消或失败返回 \"\"。
    """
    path = prompt_save_figure_path(
        parent,
        caption=caption,
        directory=directory,
        default_stem=default_stem,
    )
    if not path:
        return ""
    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    except Exception as exc:
        if show_message:
            from .styles import show_modeless_message

            show_modeless_message(
                "Error",
                f"Failed to save figure:\n{exc}",
                icon=QMessageBox.Icon.Critical,
            )
        return ""
    if show_message:
        from .styles import show_modeless_message

        show_modeless_message(
            "Saved",
            str(path),
            icon=QMessageBox.Icon.Information,
            activate=False,
        )
    return path
