"""按字段规格批量建可滚动表单（减少各 tab 样板代码）。"""

from __future__ import annotations

from typing import Literal

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..state.form_state import FormState
from .form_rows import (
    FormBinder,
    MinMaxRow,
    MultiPathRow,
    PathRow,
    labeled_combo,
    labeled_line,
)
from ...param_hints import apply_param_tooltip
from ..services.paths import resolve_work_dir
from ..services.file_filters import filters_for_form_key

FieldMode = Literal["text", "open", "save", "multi_open", "combo", "check", "minmax"]
# sections: (标题, 字段列表, 默认是否展开)
FieldSection = tuple[str, list[tuple], bool]


class CollapsibleSection(QFrame):
    """可折叠参数分组：默认收起高级项，减轻一屏字段密度。"""

    def __init__(self, title: str, *, expanded: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        self._toggle = QToolButton()
        self._toggle.setCheckable(True)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._toggle.setText(title)
        self._toggle.setChecked(expanded)
        self._toggle.toggled.connect(self._on_toggled)
        root.addWidget(self._toggle)

        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(8, 0, 0, 0)
        self.body_layout.setSpacing(4)
        self.body.setVisible(expanded)
        root.addWidget(self.body)

    def _on_toggled(self, checked: bool) -> None:
        self.body.setVisible(checked)
        self._toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

    def set_expanded(self, expanded: bool) -> None:
        if self._toggle.isChecked() != expanded:
            self._toggle.setChecked(expanded)
        else:
            self._on_toggled(expanded)


class FieldFormTab(QWidget):
    """通用参数页：字段列表 + 预览/运行按钮。

    可用 ``fields`` 平铺，或用 ``sections`` 做「常用 / 高级」分组
    （高级默认折叠）。
    """

    preview_requested = Signal()
    run_requested = Signal()
    hint_key_changed = Signal(str)

    def __init__(
        self,
        state: FormState,
        *,
        fields: list[tuple] | None = None,
        sections: list[FieldSection] | None = None,
        preview_text: str,
        run_text: str,
        defaults: dict | None = None,
        parent=None,
    ) -> None:
        """
        fields / sections 内字段项：
          ("key", "label", default, "text"|"open"|"save"|"multi_open"|"minmax")
          ("key", "label", default, "combo", [values])
          ("key", "label", default, "check")
          ("_row", [上述字段项, ...])  # 同一行横排（紧凑）
          ("_grid2", [上述字段项, ...])  # 两列网格（字段等宽）
        """
        super().__init__(parent)
        self.state = state
        self.binder = FormBinder(state)
        self._combo_keys: dict[str, object] = {}
        self._check_keys: dict[str, QCheckBox] = {}
        self._path_rows: dict[str, PathRow] = {}
        self._multi_path_rows: dict[str, MultiPathRow] = {}
        self._line_edits: dict[str, object] = {}
        self._minmax_rows: dict[str, MinMaxRow] = {}
        self._extra_widgets: list[QWidget] = []
        self._section_widgets: list[CollapsibleSection] = []

        for k, v in (defaults or {}).items():
            if not state.has(k):
                state.set(k, v)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._inner = QWidget()
        self._form = QVBoxLayout(self._inner)
        self._form.setSpacing(6)

        if sections:
            for title, specs, expanded in sections:
                sec = CollapsibleSection(title, expanded=expanded, parent=self._inner)
                self._add_specs(specs, layout=sec.body_layout, parent_w=sec.body)
                self._form.addWidget(sec)
                self._section_widgets.append(sec)
        else:
            self._add_specs(fields or [], layout=self._form, parent_w=self._inner)

        self._actions = QHBoxLayout()
        self.btn_preview = QPushButton(preview_text)
        self.btn_run = QPushButton(run_text)
        self.btn_preview.clicked.connect(self.preview_requested.emit)
        self.btn_run.clicked.connect(self.run_requested.emit)
        if preview_text:
            self._actions.addWidget(self.btn_preview)
        else:
            self.btn_preview.hide()
        if run_text:
            self._actions.addWidget(self.btn_run)
        else:
            self.btn_run.hide()
        self._actions.addStretch(1)
        self._form.addLayout(self._actions)
        self._form.addStretch(1)

        scroll.setWidget(self._inner)
        root = QVBoxLayout(self)
        root.addWidget(scroll)
        self.binder.push_to_widgets()

    def _add_specs(
        self,
        fields: list[tuple],
        *,
        layout: QVBoxLayout,
        parent_w: QWidget,
    ) -> None:
        for spec in fields:
            if spec and spec[0] == "_row":
                hbox = QHBoxLayout()
                hbox.setContentsMargins(0, 0, 0, 0)
                hbox.setSpacing(12)
                for sub in spec[1]:
                    self._add_one_spec(sub, layout=hbox, parent_w=parent_w, compact=True)
                hbox.addStretch(1)
                layout.addLayout(hbox)
                continue
            if spec and spec[0] == "_grid2":
                grid = QGridLayout()
                grid.setContentsMargins(0, 0, 0, 0)
                grid.setHorizontalSpacing(16)
                grid.setVerticalSpacing(4)
                grid.setColumnStretch(0, 1)
                grid.setColumnStretch(1, 1)
                for i, sub in enumerate(spec[1]):
                    cell = QVBoxLayout()
                    cell.setContentsMargins(0, 0, 0, 0)
                    self._add_one_spec(sub, layout=cell, parent_w=parent_w)
                    grid.addLayout(cell, i // 2, i % 2)
                layout.addLayout(grid)
                continue
            self._add_one_spec(spec, layout=layout, parent_w=parent_w)

    def _add_one_spec(
        self,
        spec: tuple,
        *,
        layout: QLayout,
        parent_w: QWidget,
        compact: bool = False,
    ) -> None:
        wd = lambda: resolve_work_dir(self.state.get_str("work_dir"))
        key = spec[0]
        label = spec[1]
        default = spec[2]
        mode = spec[3]
        if not self.state.has(key):
            self.state.set(key, default)
        if mode == "combo":
            lb, combo, row = labeled_combo(parent_w, label, values=spec[4])
            layout.addLayout(row)
            self.binder.bind_combo(key, combo)
            self._combo_keys[key] = combo
            apply_param_tooltip(lb, key)
            apply_param_tooltip(combo, key)
        elif mode == "check":
            box = QCheckBox(label)
            layout.addWidget(box)
            self.binder.bind_check(key, box)
            self._check_keys[key] = box
            apply_param_tooltip(box, key)
        elif mode == "multi_open":
            mrow = MultiPathRow(label, work_dir_getter=wd)
            layout.addWidget(mrow)
            self.binder.bind_plain(key, mrow.edit, is_file=True)
            self._multi_path_rows[key] = mrow
            apply_param_tooltip(mrow.label, key)
            apply_param_tooltip(mrow.edit, key)
            apply_param_tooltip(mrow, key)
        elif mode == "minmax":
            mrow = MinMaxRow(label, parent_w)
            layout.addWidget(mrow)
            self.binder.bind_minmax(key, mrow)
            self._minmax_rows[key] = mrow
            apply_param_tooltip(mrow.label, key)
            apply_param_tooltip(mrow.lo, key)
            apply_param_tooltip(mrow.hi, key)
            apply_param_tooltip(mrow, key)
        elif mode in ("open", "save"):
            prow = PathRow(
                label,
                mode="save_file" if mode == "save" else "open_file",
                work_dir_getter=wd,
                name_filter=filters_for_form_key(key, for_save=(mode == "save")),
            )
            layout.addWidget(prow)
            self.binder.bind_line(key, prow.edit, is_file=True)
            self._path_rows[key] = prow
            apply_param_tooltip(prow.label, key)
            apply_param_tooltip(prow.edit, key)
            apply_param_tooltip(prow, key)
        else:
            lb, edit, row = labeled_line(parent_w, label)
            if compact:
                edit.setFixedWidth(64)
                edit.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                if key == "inv.refl_stride":
                    edit.setPlaceholderText("空=1")
            layout.addLayout(row)
            self.binder.bind_line(key, edit)
            self._line_edits[key] = edit
            apply_param_tooltip(lb, key)
            apply_param_tooltip(edit, key)

    def insert_widget_at_top(self, widget: QWidget) -> None:
        """在表单最顶部插入说明等控件。"""
        self._form.insertWidget(0, widget)
        self._extra_widgets.append(widget)

    def insert_widget_before_actions(self, widget: QWidget) -> None:
        """在预览/运行按钮前插入额外控件（如折叠区）。"""
        insert_at = max(0, self._form.count() - 2)
        self._form.insertWidget(insert_at, widget)
        self._extra_widgets.append(widget)

    def pull(self) -> None:
        self.binder.pull_from_widgets()

    def set_enabled_keys(self, keys: list[str], enabled: bool) -> None:
        for k in keys:
            if k in self._line_edits:
                self._line_edits[k].setEnabled(enabled)  # type: ignore[union-attr]
            if k in self._path_rows:
                row = self._path_rows[k]
                row.setEnabled(enabled)
                # 显式同步子控件，避免曾被禁用的子项在父级重开后仍灰显
                row.edit.setEnabled(enabled)
                row.btn.setEnabled(enabled)
                if hasattr(row, "btn_recent"):
                    row.btn_recent.setEnabled(enabled)
            if k in self._multi_path_rows:
                mrow = self._multi_path_rows[k]
                mrow.setEnabled(enabled)
                mrow.edit.setEnabled(enabled)
                mrow.btn.setEnabled(enabled)
            if k in self._combo_keys:
                self._combo_keys[k].setEnabled(enabled)  # type: ignore[union-attr]
            if k in self._check_keys:
                self._check_keys[k].setEnabled(enabled)
            if k in self._minmax_rows:
                row = self._minmax_rows[k]
                row.setEnabled(enabled)
                row.lo.setEnabled(enabled)
                row.hi.setEnabled(enabled)

    def set_section_expanded(self, index: int, expanded: bool = True) -> None:
        if 0 <= index < len(self._section_widgets):
            self._section_widgets[index].set_expanded(expanded)
