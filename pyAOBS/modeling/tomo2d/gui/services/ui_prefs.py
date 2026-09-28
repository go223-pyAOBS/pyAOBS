"""tomo2d GUI 偏好：最近路径、窗口几何 / 分割条（QSettings）。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PySide6.QtCore import QByteArray, QSettings
from PySide6.QtWidgets import QSplitter, QWidget

ORG = "pyAOBS"
APP = "tomo2d"
_RECENT_KEY = "recent_paths"
_DEFAULT_RECENT = 24


def tomo2d_settings() -> QSettings:
    return QSettings(ORG, APP)


def push_recent_path(
    path: str | Path,
    *,
    settings: QSettings | None = None,
    max_items: int = _DEFAULT_RECENT,
) -> list[str]:
    """将路径置顶写入最近列表（去重）；返回更新后的列表。"""
    raw = str(path or "").strip()
    if not raw:
        return list_recent_paths(settings=settings, max_items=max_items)
    s = settings or tomo2d_settings()
    items = [raw]
    for p in list_recent_paths(settings=s, max_items=max_items * 2):
        if p != raw:
            items.append(p)
        if len(items) >= max_items:
            break
    s.setValue(_RECENT_KEY, items)
    s.sync()
    return items[:max_items]


def list_recent_paths(
    *,
    settings: QSettings | None = None,
    max_items: int = _DEFAULT_RECENT,
    existing_only: bool = False,
) -> list[str]:
    s = settings or tomo2d_settings()
    val = s.value(_RECENT_KEY, [])
    if val is None:
        return []
    if isinstance(val, str):
        items = [val] if val.strip() else []
    else:
        try:
            items = [str(x).strip() for x in list(val) if str(x).strip()]
        except TypeError:
            items = []
    out: list[str] = []
    seen: set[str] = set()
    for p in items:
        if p in seen:
            continue
        seen.add(p)
        if existing_only and not Path(p).exists():
            continue
        out.append(p)
        if len(out) >= max_items:
            break
    return out


def filter_recent_for_mode(
    paths: Iterable[str],
    *,
    mode: str,
    limit: int = 12,
) -> list[str]:
    """按 PathRow mode 过滤：dir 要目录；open/save 优先文件（也保留仍存在的路径）。"""
    out: list[str] = []
    for p in paths:
        path = Path(p)
        if mode == "dir":
            if path.is_dir() or (not path.exists() and not path.suffix):
                out.append(p)
        else:
            if path.is_file() or path.suffix or not path.exists():
                out.append(p)
        if len(out) >= limit:
            break
    return out


def save_window_layout(
    widget: QWidget,
    key: str,
    *,
    splitters: dict[str, QSplitter] | None = None,
    settings: QSettings | None = None,
) -> None:
    s = settings or tomo2d_settings()
    s.beginGroup(f"win/{key}")
    try:
        s.setValue("geometry", widget.saveGeometry())
        if splitters:
            for name, sp in splitters.items():
                if sp is not None:
                    s.setValue(f"splitter/{name}", sp.saveState())
    finally:
        s.endGroup()
        s.sync()


def restore_window_layout(
    widget: QWidget,
    key: str,
    *,
    splitters: dict[str, QSplitter] | None = None,
    settings: QSettings | None = None,
) -> bool:
    """恢复几何与分割条；有任一成功则返回 True。"""
    s = settings or tomo2d_settings()
    s.beginGroup(f"win/{key}")
    ok = False
    try:
        geo = s.value("geometry")
        if isinstance(geo, QByteArray) and not geo.isEmpty():
            ok = bool(widget.restoreGeometry(geo)) or ok
        elif geo is not None:
            # 某些平台以 bytes 返回
            ba = QByteArray(geo) if not isinstance(geo, QByteArray) else geo
            if not ba.isEmpty():
                ok = bool(widget.restoreGeometry(ba)) or ok
        if splitters:
            for name, sp in splitters.items():
                if sp is None:
                    continue
                st = s.value(f"splitter/{name}")
                if st is None:
                    continue
                ba = st if isinstance(st, QByteArray) else QByteArray(st)
                if not ba.isEmpty():
                    ok = bool(sp.restoreState(ba)) or ok
    finally:
        s.endGroup()
    return ok
