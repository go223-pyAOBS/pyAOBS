"""QFileDialog 保存路径后缀补全。

原生对话框在部分平台/主题下不会按过滤器自动加扩展名；
统一在拿到路径后按选中过滤器补全。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

# 过滤器中的扩展名：*.json / *.waveop.json / *.jpg
_FILTER_EXT_RE = re.compile(r"\*\.([A-Za-z0-9][A-Za-z0-9_.]*)")


def suffixes_from_filter(selected_filter: str) -> Tuple[str, ...]:
    """从过滤器字符串提取后缀，保持出现顺序（首个即默认追加后缀）。"""
    found = _FILTER_EXT_RE.findall(selected_filter or "")
    sufs = []
    seen = set()
    for raw in found:
        s = raw.lower()
        if not s.startswith("."):
            s = "." + s
        if s == ".*" or s in seen:
            continue
        seen.add(s)
        sufs.append(s)
    return tuple(sufs)


def ensure_save_suffix(
    path: str,
    selected_filter: str = "",
    *,
    default_suffix: str = "",
    preferred_suffix: str = "",
) -> str:
    """
    若路径尚未带上过滤器允许的后缀，则追加首选后缀。

    - ``preferred_suffix``：显式首选（如 ``.out``）
    - 否则取过滤器中最长扩展名（``*.waveop.json`` 优先于 ``*.json``）
    - 再否则用 ``default_suffix``
    - 过滤器为 ``All files (*)`` 且无 default 时不改动
    """
    text = str(path or "").strip()
    if not text:
        return text

    allowed = list(suffixes_from_filter(selected_filter))
    pref = (preferred_suffix or "").strip().lower()
    if pref and not pref.startswith("."):
        pref = "." + pref
    if not pref and allowed:
        pref = allowed[0]
    if not pref:
        ds = (default_suffix or "").strip().lower()
        if ds and not ds.startswith("."):
            ds = "." + ds
        pref = ds
    if not pref:
        return text

    if pref not in allowed:
        allowed.insert(0, pref)

    lower = text.lower()
    if any(lower.endswith(s) for s in allowed):
        return text
    return text + pref


def get_save_file_name(
    parent: Any,
    caption: str,
    directory: Union[str, Path] = "",
    filter: str = "",
    *,
    options: Any = None,
    default_suffix: str = "",
    preferred_suffix: str = "",
) -> Tuple[str, str]:
    """
    包装 ``QFileDialog.getSaveFileName``，返回时自动补全后缀。

    Returns:
        (path, selected_filter)；取消时 ``("", "")``。
    """
    from PySide6.QtWidgets import QFileDialog

    kwargs = {}
    if options is not None:
        kwargs["options"] = options
    path, selected = QFileDialog.getSaveFileName(
        parent,
        str(caption),
        str(directory or ""),
        str(filter or ""),
        **kwargs,
    )
    if not path:
        return "", selected or ""
    return (
        ensure_save_suffix(
            path,
            selected or filter,
            default_suffix=default_suffix,
            preferred_suffix=preferred_suffix,
        ),
        selected or "",
    )
