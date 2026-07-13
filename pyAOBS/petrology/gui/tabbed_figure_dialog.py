"""Shared tabbed matplotlib preview dialog scaffold."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .mpl_qt import FigureCanvasQTAgg, NavigationToolbar2QT


class TabbedFigureDialog(QDialog):
    """Non-modal tabbed matplotlib preview — subclasses supply panels and figure builder."""

    def __init__(
        self,
        *,
        window_title: str,
        error_title: str,
        fig_dir: Path,
    ) -> None:
        super().__init__(None)
        self.setModal(False)
        self.setWindowTitle(window_title)
        self.resize(1020, 760)
        self.setMinimumSize(820, 560)

        self._error_title = error_title
        self._fig_dir = fig_dir
        self._figures: list = []
        self._tab_keys: list[str] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)

        bar = QHBoxLayout()
        self._extend_toolbar(bar)
        if self._show_refresh_button():
            btn_refresh = QPushButton("刷新")
            btn_refresh.clicked.connect(self._on_refresh)
            bar.addWidget(btn_refresh)
        bar.addStretch()
        btn_save = QPushButton("保存当前页…")
        btn_save.clicked.connect(self._save_current)
        bar.addWidget(btn_save)
        save_all_text = self._save_all_button_text()
        if save_all_text:
            btn_save_all = QPushButton(save_all_text)
            btn_save_all.clicked.connect(self._on_save_all)
            bar.addWidget(btn_save_all)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        bar.addWidget(btn_close)
        root.addLayout(bar)

        hint = self._build_hint()
        if hint is not None:
            root.addWidget(hint)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs, stretch=1)

        try:
            self._populate()
        except Exception as exc:
            QMessageBox.critical(self, error_title, f"生成失败:\n{exc}")
            raise

    def _panel_items(self) -> Sequence[tuple[str, str]]:
        """Return (key, tab_title) pairs."""
        raise NotImplementedError

    def _build_figure(self, key: str):
        """Build matplotlib figure for panel key."""
        raise NotImplementedError

    def _extend_toolbar(self, bar: QHBoxLayout) -> None:
        """Optional widgets before refresh (e.g. Tp input)."""

    def _show_refresh_button(self) -> bool:
        return True

    def _on_refresh(self) -> None:
        try:
            self._populate()
        except Exception as exc:
            QMessageBox.critical(self, self._error_title, f"刷新失败:\n{exc}")

    def _build_hint(self) -> QWidget | None:
        return None

    def _save_all_button_text(self) -> str | None:
        return None

    def _on_save_all(self) -> None:
        pass

    def _default_save_path(self, idx: int) -> str:
        return str(self._fig_dir / f"{self._tab_keys[idx]}.png")

    def _clear_tabs(self) -> None:
        while self._tabs.count():
            widget = self._tabs.widget(0)
            self._tabs.removeTab(0)
            widget.deleteLater()
        for fig in self._figures:
            plt.close(fig)
        self._figures.clear()
        self._tab_keys.clear()

    def _populate(self) -> None:
        self._clear_tabs()
        for key, title in self._panel_items():
            fig = self._build_figure(key)
            self._figures.append(fig)
            self._tab_keys.append(key)

            page = QWidget()
            lay = QVBoxLayout(page)
            lay.setContentsMargins(0, 0, 0, 0)
            canvas = FigureCanvasQTAgg(fig)
            toolbar = NavigationToolbar2QT(canvas, page)
            lay.addWidget(toolbar)
            lay.addWidget(canvas, stretch=1)
            self._tabs.addTab(page, title)

    def _save_current(self) -> None:
        idx = self._tabs.currentIndex()
        if idx < 0 or idx >= len(self._figures):
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存当前页",
            self._default_save_path(idx),
            "PNG (*.png);;PDF (*.pdf);;All (*.*)",
        )
        if not path:
            return
        self._figures[idx].savefig(path, dpi=150, bbox_inches="tight")
        QMessageBox.information(self, self._error_title, f"已保存:\n{path}")

    def closeEvent(self, event) -> None:  # noqa: N802
        self._clear_tabs()
        super().closeEvent(event)
