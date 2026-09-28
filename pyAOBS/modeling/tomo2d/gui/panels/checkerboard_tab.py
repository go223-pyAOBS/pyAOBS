"""棋盘格分辨率测试页（常用展开，尺度可折叠）。"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QPushButton

from ..state.form_state import FormState
from ..widgets.field_form import FieldFormTab

_CORE = [
    ("cb.bg_smesh", "背景 smesh（起始/背景模型）", "", "open"),
    ("cb.geom", "geom（正演几何，-G）", "", "open"),
    ("cb.refl_file", "反射面（有同轮则自动填）", "", "open"),
    ("cb.amp", "棋盘振幅 A（%）", "3", "text"),
]

_SCALE = [
    ("cb.h_len", "水平波长 h（km）", "10", "text"),
    ("cb.v_len", "垂向波长 v（km）", "5", "text"),
]

_SECTIONS = [
    ("常用", _CORE, True),
    ("棋盘尺度", _SCALE, True),
]


class CheckerboardTab(FieldFormTab):
    preview_requested = Signal()
    run_requested = Signal()

    def __init__(self, state: FormState, parent=None) -> None:
        super().__init__(
            state,
            sections=_SECTIONS,
            preview_text="预览棋盘格测试",
            run_text="运行棋盘格测试",
            parent=parent,
        )
        btn = QPushButton("棋盘预览图…")
        btn.setToolTip(
            "弹出绘图窗：上扰动 ΔV、中棋盘后 Vp、下棋盘前 Vp；可勾选等值线与 DWS。"
        )
        btn.clicked.connect(self.open_preview_window)
        self._actions.insertWidget(1, btn)
        btn_res = QPushButton("棋盘结果图…")
        btn_res.setToolTip(
            "画出棋盘格测试：上真异常%、中恢复异常%、下残差%。"
            "窗口内可切换不同测试包。需先「运行棋盘格测试」。"
        )
        btn_res.clicked.connect(lambda: self.open_result_window())
        self._actions.insertWidget(2, btn_res)
        self._plot_win = None
        self._result_win = None
        bg_row = self._path_rows.get("cb.bg_smesh")
        if bg_row is not None:
            bg_row.edit.editingFinished.connect(self._autofill_companion_refl)
            bg_row.edit.textChanged.connect(self._autofill_companion_refl)
        self._autofill_companion_refl()

    def pull(self) -> None:
        super().pull()
        self._autofill_companion_refl()

    def on_state_pushed(self) -> None:
        self._autofill_companion_refl()

    def _autofill_companion_refl(self, *_args) -> None:
        from ..services.paths import resolve_work_dir
        from ..services.qc_workflows import fill_cb_refl_from_companion

        bg_row = self._path_rows.get("cb.bg_smesh")
        if bg_row is not None:
            self.state.set("cb.bg_smesh", bg_row.edit.text())
        try:
            work = resolve_work_dir(self.state.get_str("work_dir"))
        except Exception:
            return
        filled = fill_cb_refl_from_companion(self.state, work)
        if not filled:
            return
        row = self._path_rows.get("cb.refl_file")
        if row is None:
            return
        if row.edit.text().replace("\\", "/") != filled.replace("\\", "/"):
            row.edit.blockSignals(True)
            row.edit.setText(filled)
            row.edit.blockSignals(False)

    def open_preview_window(self) -> None:
        from ..dialogs.checkerboard_preview import open_checkerboard_preview

        prev = self._plot_win
        win = open_checkerboard_preview(
            self.state, pull=self.pull, existing=prev
        )
        if win is None:
            return
        if win is not prev:
            self._plot_win = win
            win.destroyed.connect(lambda *_a: setattr(self, "_plot_win", None))

    def open_result_window(self, run_dir=None) -> None:
        from pathlib import Path

        from ..dialogs.checkerboard_result import open_checkerboard_result

        prev = self._result_win
        win = open_checkerboard_result(
            self.state,
            pull=self.pull,
            run_dir=Path(run_dir) if run_dir is not None else None,
            existing=prev,
        )
        if win is None:
            return
        if win is not prev:
            self._result_win = win
            win.destroyed.connect(lambda *_a: setattr(self, "_result_win", None))
