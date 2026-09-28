"""Collapsible / scrollable parameter strip for iphase."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class CollapsibleParamStrip(QWidget):
    """参数条：可折叠；按「通用 / 1D / 2D|2Dequi」分行归类。"""

    GROUP_ORDER = ("common", "1D", "fwd")
    GROUP_TITLES = {
        "common": "通用",
        "1D": "1D",
        "fwd": "2D",
    }
    GROUP_TIPS = {
        "common": "各时差模式共用的显示与配对选项",
        "1D": "1D 薄层理论 / 反演相关参数",
        "fwd": "2D 正演或 2Dequi 等效构造相关参数（同一行）",
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("IphaseParamStrip")
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(2)

        hdr = QHBoxLayout()
        self.toggle = QToolButton()
        self.toggle.setObjectName("IphaseParamsToggle")
        self.toggle.setText("▾ 参数")
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        self.toggle.setToolTip("折叠/展开参数条（按当前时差模式显示对应分组）")
        self.toggle.toggled.connect(self._on_toggle)
        hdr.addWidget(self.toggle)
        hdr.addStretch(1)
        root.addLayout(hdr)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll.setMaximumHeight(100)

        self.body = QWidget()
        self.body_lay = QVBoxLayout(self.body)
        self.body_lay.setContentsMargins(4, 2, 4, 2)
        self.body_lay.setSpacing(2)

        self._groups: dict[str, tuple[QWidget, QHBoxLayout, QLabel]] = {}
        self._equi_only: list[QWidget] = []
        for key in self.GROUP_ORDER:
            self._ensure_group(key)

        self.scroll.setWidget(self.body)
        root.addWidget(self.scroll)

    def _ensure_group(self, key: str) -> QHBoxLayout:
        if key in self._groups:
            return self._groups[key][1]
        row = QWidget()
        row.setObjectName("IphaseParamGroupRow")
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        title = QLabel(f"[{self.GROUP_TITLES.get(key, key)}]")
        title.setObjectName("IphaseParamGroupTitle")
        tip = self.GROUP_TIPS.get(key, "")
        if tip:
            title.setToolTip(tip)
        h.addWidget(title)
        self.body_lay.addWidget(row)
        self._groups[key] = (row, h, title)
        return h

    def add_group_widget(self, group: str, w: QWidget, *, equi_only: bool = False) -> None:
        self._ensure_group(group).addWidget(w)
        if equi_only:
            self._equi_only.append(w)

    def add_group_stretch(self, group: str) -> None:
        self._ensure_group(group).addStretch(1)

    def set_mode_visibility(self, mode: str) -> None:
        """按当前时差模式显示对应分组；2D/2Dequi 共用一行，标题随模式切换。"""
        m = str(mode).strip()
        visible = {
            "common": True,
            "1D": m == "1D",
            "fwd": m in ("2D", "2Dequi"),
        }
        for key, (row, _, title) in self._groups.items():
            row.setVisible(bool(visible.get(key, False)))

        if "fwd" in self._groups:
            _, _, title = self._groups["fwd"]
            if m == "2Dequi":
                title.setText("[2Dequi]")
                title.setToolTip("2Dequi：等效构造 + 2D 正演相关参数（同一行）")
            else:
                title.setText("[2D]")
                title.setToolTip("2D 正演（tx.out / r.in pois）相关参数")

        show_equi = m == "2Dequi"
        for w in self._equi_only:
            w.setVisible(show_equi)

        n = sum(1 for k, on in visible.items() if on and k in self._groups)
        self.scroll.setMaximumHeight(36 + 28 * max(n, 1))

    def clear_widgets(self) -> None:
        """清空各组内控件（保留标题），便于重建。"""
        self._equi_only.clear()
        for key, (row, h, _title) in self._groups.items():
            while h.count() > 1:
                item = h.takeAt(1)
                w = item.widget()
                if w is not None:
                    w.deleteLater()

    def _on_toggle(self, checked: bool) -> None:
        self.scroll.setVisible(checked)
        self.toggle.setText("▾ 参数" if checked else "▸ 参数")
