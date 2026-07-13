"""非模态岩性 / 预设说明窗。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from petrology.reference.lithology_reference import (
    NATIVE_BACKEND_NOTE,
    describe_lithology_key,
    describe_preset,
    format_lithology_detail_text,
    list_lithology_presets,
    peridotite_lithology_keys,
    enriched_lithology_keys,
)

from .app import UI_FONT_PT
from .dialog_utils import show_modeless_dialog
from .qt_styles import (
    COLOR_ACCENT,
    COLOR_BORDER_LIGHT,
    COLOR_BORDER_MED,
    COLOR_PANEL,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_SECONDARY,
    apply_dialog_style,
    hint_label,
    primary_button,
)


_DETAIL_QSS = f"""
QLabel#LipLithTitle {{
    font-size: {UI_FONT_PT + 1}pt;
    font-weight: bold;
    color: {COLOR_TEXT};
}}
QLabel#LipLithSection {{
    font-size: {UI_FONT_PT}pt;
    font-weight: bold;
    color: {COLOR_ACCENT};
    margin-top: 4px;
}}
QLabel#LipLithBody {{
    font-size: {UI_FONT_PT}pt;
    color: {COLOR_TEXT_SECONDARY};
    line-height: 1.45;
}}
QWidget#LipLithDetailPanel {{
    background-color: {COLOR_PANEL};
    border: 1px solid {COLOR_BORDER_LIGHT};
    border-radius: 4px;
}}
QTreeWidget {{
    border: 2px solid {COLOR_BORDER_MED};
    border-radius: 4px;
}}
"""


class LithologyReferenceDialog(QDialog):
    def __init__(
        self,
        *,
        preset: str = "(custom)",
        per: str = "",
        pyr: str = "",
        backend: str = "pymelt",
        per_h2o: float = 0.0,
        pyr_h2o: float = 0.0,
    ) -> None:
        super().__init__(None)
        self.setModal(False)
        self.setWindowTitle("岩性说明 — pMELTS / REEBOX 端元")
        self.resize(860, 620)
        self.setMinimumSize(640, 480)
        apply_dialog_style(self)
        self.setStyleSheet(self.styleSheet() + _DETAIL_QSS)

        self._context = {
            "preset": preset,
            "per": per,
            "pyr": pyr,
            "backend": backend,
            "per_h2o": per_h2o,
            "pyr_h2o": pyr_h2o,
        }

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)
        root.addWidget(
            hint_label(
                "查阅 preset 与 per/pyr 端元。"
                "橄榄石端显示的「源区 solid bulk」是未熔地幔（HZ-Dep1），"
                "不是 KKHS02 Fig.2 的起始 **熔体**；熔体见各端详情中的 Fig.2 对比。"
            )
        )

        body = QHBoxLayout()
        self._tree = QTreeWidget()
        self._tree.setHeaderLabel("岩性目录")
        self._tree.setMinimumWidth(240)
        body.addWidget(self._tree, stretch=1)

        detail_outer = QVBoxLayout()
        self._title = QLabel("")
        self._title.setObjectName("LipLithTitle")
        self._title.setWordWrap(True)
        detail_outer.addWidget(self._title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._detail_host = QWidget()
        self._detail_host.setObjectName("LipLithDetailPanel")
        self._detail_lay = QVBoxLayout(self._detail_host)
        self._detail_lay.setContentsMargins(10, 8, 10, 10)
        scroll.setWidget(self._detail_host)
        detail_outer.addWidget(scroll, stretch=1)
        body.addLayout(detail_outer, stretch=2)
        root.addLayout(body, stretch=1)

        btn_row = QHBoxLayout()
        btn_ctx = QPushButton("跳转到当前岩性面板选择")
        btn_ctx.setToolTip("定位到主界面 preset / per / pyr 当前项")
        btn_ctx.clicked.connect(self._go_context_selection)
        btn_row.addWidget(btn_ctx)
        btn_row.addStretch()
        close_btn = primary_button("关闭")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

        self._items: dict[str, QTreeWidgetItem] = {}
        self._payload: dict[str, dict] = {}
        self._populate()
        self._tree.currentItemChanged.connect(self._on_select)
        self._go_context_selection()

    def _populate(self) -> None:
        if self._context.get("backend") == "native":
            native_item = QTreeWidgetItem(["native 后端"])
            self._tree.addTopLevelItem(native_item)
            self._items["native"] = native_item
            self._payload["native"] = {
                "kind": "native",
                "title": NATIVE_BACKEND_NOTE.title_zh,
                "traits": NATIVE_BACKEND_NOTE.traits,
                "when_to_use": NATIVE_BACKEND_NOTE.when_to_use,
            }

        preset_top = QTreeWidgetItem(["命名 preset"])
        self._tree.addTopLevelItem(preset_top)
        for name in list_lithology_presets():
            leaf = QTreeWidgetItem([name])
            preset_top.addChild(leaf)
            iid = f"preset:{name}"
            self._items[iid] = leaf
            self._payload[iid] = describe_preset(name)
        preset_top.setExpanded(True)

        per_top = QTreeWidgetItem(["橄榄石端 (per)"])
        self._tree.addTopLevelItem(per_top)
        for key in peridotite_lithology_keys():
            leaf = QTreeWidgetItem([key])
            per_top.addChild(leaf)
            iid = f"lith:{key}"
            self._items[iid] = leaf
            self._payload[iid] = describe_lithology_key(key)
        per_top.setExpanded(True)

        pyr_top = QTreeWidgetItem(["辉石岩 / 榴辉岩端 (pyr)"])
        self._tree.addTopLevelItem(pyr_top)
        for key in enriched_lithology_keys():
            leaf = QTreeWidgetItem([key])
            pyr_top.addChild(leaf)
            iid = f"lith:{key}"
            self._items[iid] = leaf
            self._payload[iid] = describe_lithology_key(key)
        pyr_top.setExpanded(True)

    def _clear_detail(self) -> None:
        while self._detail_lay.count():
            item = self._detail_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _set_detail(self, payload: dict) -> None:
        self._title.setText(payload.get("title") or payload.get("id") or "")
        self._clear_detail()
        for head, text in format_lithology_detail_text(payload):
            if not str(text).strip():
                continue
            h = QLabel(head)
            h.setObjectName("LipLithSection")
            self._detail_lay.addWidget(h)
            p = QLabel(str(text))
            p.setObjectName("LipLithBody")
            p.setWordWrap(True)
            p.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self._detail_lay.addWidget(p)
        self._detail_lay.addStretch(1)

    def _on_select(self, current: QTreeWidgetItem | None, _prev) -> None:
        if current is None:
            return
        for iid, item in self._items.items():
            if item is current:
                self._set_detail(self._payload.get(iid, {}))
                return

    def _go_context_selection(self) -> None:
        ctx = self._context
        if ctx.get("backend") == "native" and "native" in self._items:
            self._tree.setCurrentItem(self._items["native"])
            return
        preset = ctx.get("preset") or "(custom)"
        if preset not in ("", "(custom)"):
            iid = f"preset:{preset}"
            if iid in self._items:
                self._tree.setCurrentItem(self._items[iid])
                return
        per = str(ctx.get("per") or "")
        if per:
            iid = f"lith:{per}"
            if iid in self._items:
                self._payload[iid] = describe_lithology_key(
                    per, h2o_wt=float(ctx.get("per_h2o") or 0.0)
                )
                self._tree.setCurrentItem(self._items[iid])
                return
        pyr = str(ctx.get("pyr") or "")
        if pyr:
            iid = f"lith:{pyr}"
            if iid in self._items:
                self._payload[iid] = describe_lithology_key(
                    pyr, h2o_wt=float(ctx.get("pyr_h2o") or 0.0)
                )
                self._tree.setCurrentItem(self._items[iid])


def show_lithology_reference(
    *,
    preset: str = "(custom)",
    per: str = "",
    pyr: str = "",
    backend: str = "pymelt",
    per_h2o: float = 0.0,
    pyr_h2o: float = 0.0,
) -> None:
    dlg = LithologyReferenceDialog(
        preset=preset,
        per=per,
        pyr=pyr,
        backend=backend,
        per_h2o=per_h2o,
        pyr_h2o=pyr_h2o,
    )
    show_modeless_dialog(dlg, singleton=False)

