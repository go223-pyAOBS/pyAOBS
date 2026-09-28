"""Non-modal Qt dialogs for LIP Petrology GUI."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog

from pyAOBS.utils.qt_combo import (
    connect_combo_deferred,
    defer_after_combo_popup,
    hide_combo_popup,
)
from pyAOBS.utils.qt_independent_window import (
    apply_native_window_border,
    configure_independent_window,
)

_registry: list[QDialog] = []
_open_singletons: dict[type, QDialog] = {}

__all__ = [
    "configure_independent_dialog",
    "show_modeless_dialog",
    "connect_combo_deferred",
    "defer_after_combo_popup",
    "hide_combo_popup",
]


def configure_independent_dialog(dlg: QDialog) -> None:
    """Detach from parent and use a normal top-level window with a visible frame."""
    configure_independent_window(dlg)


def show_modeless_dialog(
    dlg: QDialog,
    *,
    activate: bool = False,
    singleton: bool = False,
) -> None:
    """Show a tool dialog without blocking or repeatedly raising over the main window."""
    if not isinstance(dlg, QDialog):
        raise TypeError("show_modeless_dialog expects a QDialog")

    configure_independent_dialog(dlg)

    if singleton:
        cls = type(dlg)
        existing = _open_singletons.get(cls)
        if existing is not None and existing.isVisible():
            return

    dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    _registry.append(dlg)

    if singleton:
        cls = type(dlg)
        _open_singletons[cls] = dlg

        def _clear_singleton(_code: int = 0) -> None:
            if _open_singletons.get(cls) is dlg:
                _open_singletons.pop(cls, None)

        dlg.finished.connect(_clear_singleton)

    def _on_finished(_code: int = 0) -> None:
        try:
            _registry.remove(dlg)
        except ValueError:
            pass

    dlg.finished.connect(_on_finished)
    dlg.show()
    apply_native_window_border(dlg)
    if activate:
        dlg.raise_()
        dlg.activateWindow()
