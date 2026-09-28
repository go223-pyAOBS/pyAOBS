"""蒙特卡洛不确定性分析页（常用展开，其余折叠）。"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton

from ..services.mc_init_models import MC_INIT_MODE_CHOICES
from ..state.form_state import FormState
from ..widgets.field_form import FieldFormTab

_CORE = [
    ("mc.data", "走时 data（-G）", "", "open"),
    ("mc.n_runs", "实现次数 N", "10", "text"),
    ("mc.seed", "随机种子", "1", "text"),
    ("mc.chi_max", "卡方阈值（pred χ² <）", "1.8", "text"),
]

_LAYER_KEYS = (
    "mc.sed_h",
    "mc.sed_v",
    "mc.uc_h",
    "mc.uc_v",
    "mc.lc_h",
    "mc.lc_v",
    "mc.mantle_v",
)

_MESH_KEYS = ("mc.base_mesh",)
_VIN_KEYS = ("mc.v_in",)

_UNIT_FIELD_KEYS = {
    "sed": ("mc.sed_h", "mc.sed_v"),
    "uc": ("mc.uc_h", "mc.uc_v"),
    "lc": ("mc.lc_h", "mc.lc_v"),
    "mantle": ("mc.mantle_v",),
}

_INIT = [
    ("mc.init_mode", "模型方式", MC_INIT_MODE_CHOICES[0], "combo", list(MC_INIT_MODE_CHOICES)),
    ("mc.base_mesh", "初始/背景 smesh", "", "open"),
    ("mc.v_in", "v.in", "", "open"),
    ("mc.sed_h", "沉积厚度（km）", "0.2 2.5", "minmax"),
    ("mc.sed_v", "沉积顶底速度（km/s）", "1.7 3.6", "minmax"),
    ("mc.uc_h", "上地壳厚度（km）", "6 11", "minmax"),
    ("mc.uc_v", "上地壳顶底速度（km/s）", "4.0 6.5", "minmax"),
    ("mc.lc_h", "下地壳厚度（km）", "10 25", "minmax"),
    ("mc.lc_v", "下地壳顶底速度（km/s）", "6.6 7.5", "minmax"),
    ("mc.mantle_v", "地幔顶底速度（km/s）", "7.6 8.2", "minmax"),
]

_NOISE = [
    ("mc.tt_noise", "叠加走时噪声", True, "check"),
    ("mc.noise_sigma", "噪声 σ（秒；或相对 u 的倍数）", "0.01", "text"),
    ("mc.noise_relative_u", "噪声相对该行误差 u（σ×u）", False, "check"),
]

_SECTIONS = [
    ("常用", _CORE, True),
    ("起始模型", _INIT, True),
    ("走时噪声", _NOISE, False),
]


class MonteCarloTab(FieldFormTab):
    preview_requested = Signal()
    run_requested = Signal()

    def __init__(self, state: FormState, parent=None) -> None:
        super().__init__(
            state,
            sections=_SECTIONS,
            preview_text="预览蒙特卡洛",
            run_text="运行蒙特卡洛分析",
            parent=parent,
        )
        btn = QPushButton("蒙特卡洛预览图…")
        btn.setToolTip(
            "弹出绘图窗：左上 ΔV、左中第 1 次实现、左下基础模型；"
            "smesh 时左图叠各次 Moho（随海底）；"
            "右侧叠绘全部 N 条 1D（v.in 从最浅勾选层顶起：海底/基底/Conrad/莫霍），"
            "与运行同一套种子，不写盘。"
        )
        btn.clicked.connect(self.open_preview_window)
        self._actions.insertWidget(1, btn)
        btn_res = QPushButton("蒙特卡洛结果图…")
        btn_res.setToolTip(
            "画出已完成的蒙特卡洛：上均值 Vp、下误差 σ，叠界面均值 ±σ。"
            "均值/σ 已按卡方与各次 DWS 筛选。窗口内可切换不同结果包。"
        )
        btn_res.clicked.connect(lambda: self.open_result_window())
        self._actions.insertWidget(2, btn_res)
        self._plot_win = None
        self._result_win = None
        combo = self._combo_keys.get("mc.init_mode")
        if combo is not None:
            combo.currentTextChanged.connect(lambda *_a: self._sync_init_fields())
        self._vin_pick_win = None
        self._btn_vin_pick = QPushButton("选层…")
        self._btn_vin_pick.setToolTip(
            "打开独立窗口：速度底图上勾选海底/基底/Conrad/莫霍（画线），"
            "再勾选要扰动的层（蒙版）。"
        )
        self._btn_vin_pick.clicked.connect(self.open_vin_pick_window)
        self._vin_summary = QLabel("选 v.in 后点「选层…」指定界面与扰动层")
        self._vin_summary.setWordWrap(True)
        self._vin_summary.setStyleSheet("color:#475569;")
        if len(self._section_widgets) >= 2:
            body = self._section_widgets[1].body_layout
            vin_row = self._path_rows.get("mc.v_in")
            insert_at = body.count()
            if vin_row is not None:
                for i in range(body.count()):
                    item = body.itemAt(i)
                    w = item.widget() if item is not None else None
                    if w is vin_row:
                        insert_at = i + 1
                        break
            body.insertWidget(insert_at, self._btn_vin_pick)
            body.insertWidget(insert_at + 1, self._vin_summary)
        vin_row = self._path_rows.get("mc.v_in")
        if vin_row is not None:
            vin_row.edit.textChanged.connect(lambda *_a: self._update_vin_summary())
        self._sync_init_fields()

    def pull(self) -> None:
        super().pull()

    def on_state_pushed(self) -> None:
        """工区/配置写入控件时 combo 会 blockSignals，须补一次启用互锁。"""
        from ..services.mc_init_models import mc_init_mode_choice

        want = mc_init_mode_choice(self.state)
        combo = self._combo_keys.get("mc.init_mode")
        if combo is not None:
            combo.blockSignals(True)
            ix = combo.findText(want)
            combo.setCurrentIndex(max(0, ix))
            combo.blockSignals(False)
        self.state.set("mc.init_mode", want)
        self._sync_init_fields()

    def _sync_init_fields(self) -> None:
        from ..services.mc_init_models import mc_init_mode_choice, resolve_mc_init_mode

        combo = self._combo_keys.get("mc.init_mode")
        if combo is not None:
            self.state.set("mc.init_mode", str(combo.currentText() or "").strip())
        want = mc_init_mode_choice(self.state)
        if combo is not None and str(combo.currentText() or "").strip() != want:
            combo.blockSignals(True)
            ix = combo.findText(want)
            combo.setCurrentIndex(max(0, ix))
            combo.blockSignals(False)
            self.state.set("mc.init_mode", want)
        mode = resolve_mc_init_mode(self.state)
        vin_on = mode == "vinlayers"
        smesh_on = not vin_on
        self.set_enabled_keys(list(_MESH_KEYS), smesh_on)
        self.set_enabled_keys(list(_VIN_KEYS), vin_on)
        if getattr(self, "_btn_vin_pick", None) is not None:
            self._btn_vin_pick.setEnabled(vin_on)
        if getattr(self, "_vin_summary", None) is not None:
            self._vin_summary.setEnabled(vin_on)
        if vin_on:
            self._update_vin_summary()
            self._sync_unit_range_fields()
        else:
            self.set_enabled_keys(list(_LAYER_KEYS), True)

    def _update_vin_summary(self) -> None:
        from pathlib import Path

        from ..services.mc_vin_layers import (
            available_units,
            marks_from_state,
            parse_vin_units,
            resolve_mc_vin_path,
            vin_n_ifaces,
            vin_selection_summary,
        )
        from ..services.paths import resolve_work_dir

        if getattr(self, "_vin_summary", None) is None:
            return
        vin_row = self._path_rows.get("mc.v_in")
        if vin_row is not None:
            self.state.set("mc.v_in", str(vin_row.edit.text() or "").strip())
        work = resolve_work_dir(self.state.get_str("work_dir"))
        path = resolve_mc_vin_path(self.state, work)
        if path is None or not Path(path).is_file():
            self._vin_summary.setText("未找到 v.in（填本页或 gen_smesh）后再点「选层…」")
            return
        try:
            n = vin_n_ifaces(path)
        except Exception as exc:
            self._vin_summary.setText(f"解析失败：{exc}")
            return
        marks = marks_from_state(self.state, n)
        allowed = available_units(marks)
        units = parse_vin_units(self.state.get_str("mc.vin_units") or "", allowed)
        self._vin_summary.setText(f"{path.name}  ·  {vin_selection_summary(marks, units)}")

    def _sync_unit_range_fields(self) -> None:
        from ..services.mc_init_models import resolve_mc_init_mode
        from ..services.mc_vin_layers import parse_vin_units

        mode = resolve_mc_init_mode(self.state)
        if mode == "layers1d":
            self.set_enabled_keys(list(_LAYER_KEYS), True)
            return
        if mode != "vinlayers":
            self.set_enabled_keys(list(_LAYER_KEYS), False)
            return
        self.set_enabled_keys(list(_LAYER_KEYS), False)
        chosen = set(parse_vin_units(self.state.get_str("mc.vin_units") or ""))
        for unit, keys in _UNIT_FIELD_KEYS.items():
            self.set_enabled_keys(list(keys), unit in chosen)

    def open_vin_pick_window(self) -> None:
        from ..dialogs.monte_carlo_vin_pick import open_monte_carlo_vin_pick

        prev = self._vin_pick_win
        win = open_monte_carlo_vin_pick(
            self.state,
            pull=self.pull,
            on_changed=self._on_vin_pick_changed,
            existing=prev,
        )
        if win is None:
            return
        if win is not prev:
            self._vin_pick_win = win
            win.destroyed.connect(lambda *_a: setattr(self, "_vin_pick_win", None))

    def _on_vin_pick_changed(self) -> None:
        self._update_vin_summary()
        self._sync_unit_range_fields()

    def open_preview_window(self) -> None:
        from ..dialogs.monte_carlo_preview import open_monte_carlo_preview

        prev = self._plot_win
        win = open_monte_carlo_preview(
            self.state, pull=self.pull, existing=prev
        )
        if win is None:
            return
        if win is not prev:
            self._plot_win = win
            win.destroyed.connect(lambda *_a: setattr(self, "_plot_win", None))

    def open_result_window(self, run_dir=None) -> None:
        from pathlib import Path

        from ..dialogs.monte_carlo_result import open_monte_carlo_result

        prev = self._result_win
        win = open_monte_carlo_result(
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
