"""Qt QComboBox helpers: defer heavy slots until after popup settles."""

from __future__ import annotations

from typing import Any, Callable, Optional

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QAbstractItemView, QComboBox, QFrame, QListView, QWidget


def configure_combo_list_view(combo: QComboBox) -> QListView:
    """给 Combo 挂紧凑 QListView，避免弹层与框体之间大块空白。"""
    view = QListView(combo)
    view.setUniformItemSizes(True)
    view.setSpacing(0)
    view.setFrameShape(QFrame.Shape.NoFrame)
    view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerItem)
    view.setResizeMode(QListView.ResizeMode.Fixed)
    combo.setView(view)
    combo.setMaxVisibleItems(min(max(combo.maxVisibleItems(), 12), 24))
    return view


def restore_combo_popup(combo: Optional[QWidget]) -> None:
    """轻度修复下拉状态。

    注意：弹层关闭时 view 本来就是 Hidden，切勿因此新建 QListView，
    否则会出现「列表与下拉框之间一大块空白」。
    """
    if combo is None or not isinstance(combo, QComboBox):
        return
    try:
        combo.setEnabled(True)
        combo.setAttribute(Qt.WidgetAttribute.WA_UnderMouse, False)
    except Exception:
        pass
    try:
        view = combo.view()
    except Exception:
        view = None
    if view is None:
        try:
            configure_combo_list_view(combo)
        except Exception:
            pass
        return
    try:
        view.setEnabled(True)
    except Exception:
        pass


def hide_combo_popup(combo: Optional[QWidget]) -> None:
    """Close dropdown via hidePopup only（禁止 view().hide()）。"""
    if combo is None:
        return
    hide = getattr(combo, "hidePopup", None)
    if callable(hide):
        try:
            hide()
        except Exception:
            pass


def defer_after_combo_popup(
    fn: Callable[[], Any],
    combo: Optional[QWidget] = None,
    *,
    delay_ms: int = 40,
    hide_popup: bool = False,
) -> None:
    """Defer ``fn`` so the combo popup can finish closing first."""

    def _run() -> None:
        try:
            fn()
        finally:
            restore_combo_popup(combo)

    if hide_popup:
        hide_combo_popup(combo)
    QTimer.singleShot(max(0, int(delay_ms)), _run)


def connect_combo_deferred(
    combo: QComboBox,
    slot: Callable[..., Any],
    *,
    signal: str = "activated",
    delay_ms: int = 40,
    hide_popup: bool = False,
) -> None:
    """Connect combo so heavy ``slot`` runs after popup can settle."""
    sig = getattr(combo, signal)

    def _handler(*args: Any) -> None:
        captured = args
        defer_after_combo_popup(
            lambda: slot(*captured),
            combo,
            delay_ms=delay_ms,
            hide_popup=hide_popup,
        )

    sig.connect(_handler)

    if not getattr(combo, "_pyaobs_popup_patched", False):
        _orig_show = combo.showPopup

        def _show_popup() -> None:
            # 只保证 enabled，绝不因 isHidden 重建 view
            restore_combo_popup(combo)
            _orig_show()

        combo.showPopup = _show_popup  # type: ignore[method-assign]
        combo._pyaobs_popup_patched = True  # type: ignore[attr-defined]
