"""Non-modal Qt dialogs for LIP Petrology GUI."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog

_registry: list[QDialog] = []
_open_singletons: dict[type, QDialog] = {}


def configure_independent_dialog(dlg: QDialog) -> None:
    """Detach from parent and use a normal top-level window (no focus stealing on main window)."""
    dlg.setModal(False)
    dlg.setWindowModality(Qt.WindowModality.NonModal)
    dlg.setParent(None)
    flags = (
        Qt.WindowType.Window
        | Qt.WindowType.WindowTitleHint
        | Qt.WindowType.WindowCloseButtonHint
        | Qt.WindowType.WindowMinMaxButtonsHint
    )
    dlg.setWindowFlags(flags)


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
    if activate:
        dlg.raise_()
        dlg.activateWindow()
