"""tt_inverse 参数页（常用展开，其余折叠）。"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..dialog_utils import show_modeless_text
from ..services.workflow_bridge import format_monitor_checklist
from ..state.form_state import FormState
from ..widgets.field_form import FieldFormTab

# 日常反演只需这些
_CORE = [
    ("inv.mesh", "mesh (-M)", "", "open"),
    ("inv.data", "data (-G)", "", "open"),
    ("inv.refl_file", "refl_file (-F 莫霍/反射)", "", "open"),
    ("inv.seafloor_file", "seafloor_file (-Y 海底)", "", "open"),
    ("inv.conv_file", "conv_file (-B 转换面，6/7/8)", "", "open"),
    ("inv.vsmesh", "vsmesh (-U 独立 Vs)", "", "open"),
    ("inv.kappa", "kappa (-k，无 -U 时 Vp/Vs)", "", "text"),
    ("inv.freeze_refl", "冻结 -F 界面 (-u)", False, "check"),
    (
        "_row",
        [
            ("inv.invert_water_only", "只反水 (-y)", False, "check"),
            ("inv.invert_crust_only", "只反壳 (-w)", False, "check"),
        ],
    ),
    ("inv.refl_weight", "refl_weight (-W)", "", "text"),
    (
        "_row",
        [
            ("inv.do_full_refl", "贴面反射 (-A，改路径)", False, "check"),
            ("inv.jumping", "jumping (-P)", False, "check"),
            ("inv.refl_stride", "抽稀步长", "", "text"),
        ],
    ),
    ("inv.niter", "niter (-I)", "5", "text"),
    ("inv.target_chi2", "target_chi2 (-J)", "1.0", "text"),
    ("inv.lsqr_tol", "lsqr_tol (-Q)", "1e-3", "text"),
    ("inv.crit_chi", "crit_chi (-R)", "", "text"),
    ("inv.smooth_vel", "smooth_vel (-SV)", "", "text"),
    ("inv.smooth_dep", "smooth_dep (-SD)", "", "text"),
    ("inv.damp_vel", "固定阻尼 -DV", "", "text"),
    ("inv.damp_dep", "固定阻尼 -DD", "", "text"),
    ("inv.auto_damp_max_dv", "自动阻尼 -TV %", "", "text"),
    ("inv.auto_damp_max_dd", "自动阻尼 -TD %", "", "text"),
    ("inv.out_root", "out_root (-O)", "", "save"),
    ("inv.use_repro_bundle", "可复现运行包（runs/ttinv_…）", True, "check"),
]

_RAY = [
    ("inv.xorder", "xorder (-N)", "4", "text"),
    ("inv.zorder", "zorder (-N)", "4", "text"),
    ("inv.clen", "clen (-N)", "0.8", "text"),
    ("inv.nintp", "nintp (-N)", "8", "text"),
    ("inv.bend_cg_tol", "bend_cg_tol (-N)", "1e-4", "text"),
    ("inv.bend_br_tol", "bend_br_tol (-N)", "1e-5", "text"),
]

_OUT = [
    ("inv.log_file", "log_file (-L)", "", "save"),
    ("inv.out_level", "out_level (-o，默认空=不写射线/残差)", "", "text"),
    ("inv.dws_file", "dws_file (-K，默认空)", "", "save"),
    ("inv.verbose_level", "verbose_level (-V，默认空=不刷屏)", "", "text"),
    ("inv.print_final_only", "print_final_only (-l，默认开=少写 smesh)", True, "check"),
    ("inv.bundle_run_label", "运行包目录备注（可选）", "", "text"),
]

_REG_ADV = [
    ("inv.smooth_corr_v_fn", "smooth_corr_v_fn (-CV)", "", "open"),
    ("inv.smooth_corr_d_fn", "smooth_corr_d_fn (-CD)", "", "open"),
    ("inv.damp_v_fn", "damp_v_fn (-DQ，属固定 -D)", "", "open"),
    ("inv.apply_filter", "开 2D 滤波 -s（不填边界文件则用 mesh 地形）", False, "check"),
    ("inv.filter_bound_file", "filter 边界文件（可选）", "", "open"),
    ("inv.smooth_vel_log10", "smooth_vel_log10 (-XV)", False, "check"),
    ("inv.smooth_dep_log10", "smooth_dep_log10 (-XD)", False, "check"),
]

_GRAV = [
    ("inv.grav_file", "grav_file (-ZG)", "", "open"),
    ("inv.grav_grid", "grav_grid (-ZX)", "", "text"),
    ("inv.grav_refrange", "grav_refrange (-ZR)", "", "text"),
    ("inv.grav_cont_file", "continent_up (-ZC)", "", "open"),
    ("inv.grav_cont_iconv", "continent_iconv (-ZC)", "", "text"),
    ("inv.grav_oceanU_up", "oceanU_up (-ZU)", "", "open"),
    ("inv.grav_oceanU_lo", "oceanU_lo (-ZU)", "", "open"),
    ("inv.grav_oceanU_iconv", "oceanU_iconv (-ZU)", "", "text"),
    ("inv.grav_oceanL_up", "oceanL_up (-ZL)", "", "open"),
    ("inv.grav_oceanL_iconv", "oceanL_iconv (-ZL)", "", "text"),
    ("inv.grav_sed_up", "sed_up (-ZS)", "", "open"),
    ("inv.grav_sed_lo", "sed_lo (-ZS)", "", "open"),
    ("inv.grav_sed_iconv", "sed_iconv (-ZS)", "", "text"),
    ("inv.grav_deriv", "deriv (-ZD)", "", "text"),
    ("inv.grav_weight", "weight_grav (-ZW)", "", "text"),
    ("inv.grav_z0", "z0 (-ZZ)", "", "text"),
    ("inv.grav_dws", "grav_dws (-ZK)", "", "save"),
    ("inv.grav_cutoff", "cutoff (-ZT)", "", "text"),
]

_SECTIONS = [
    ("常用（日常反演）", _CORE, True),
    ("射线弯曲 (-N)", _RAY, False),
    ("输出 / 日志", _OUT, False),
    ("正则化高级（相关/自适应阻尼等）", _REG_ADV, False),
    ("联合重力 (-Z*)", _GRAV, False),
]

_FIXED_DAMP_KEYS = ("inv.damp_vel", "inv.damp_dep", "inv.damp_v_fn")
_AUTO_DAMP_KEYS = ("inv.auto_damp_max_dv", "inv.auto_damp_max_dd")
_WATER_MUTEX = ("inv.invert_water_only", "inv.invert_crust_only")


class TtInverseTab(FieldFormTab):
    fill_upstream_requested = Signal()
    sync_ray_from_fwd_requested = Signal()
    go_damp_requested = Signal()
    go_vcorr_requested = Signal()
    go_dcorr_requested = Signal()

    def __init__(self, state: FormState, parent=None) -> None:
        super().__init__(
            state,
            sections=_SECTIONS,
            preview_text="预览 tt_inverse",
            run_text="运行 tt_inverse",
            parent=parent,
        )

        bar = QWidget()
        vl = QVBoxLayout(bar)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(4)
        btns = QWidget()
        hl = QHBoxLayout(btns)
        hl.setContentsMargins(0, 0, 0, 0)
        btn_fill = QPushButton("上游填充…")
        btn_fill.setToolTip(
            "用 gen.smesh_out / fwd 输出 / pipeline 桥接路径填充空的 mesh、data、damp、corr"
        )
        btn_fill.clicked.connect(self.fill_upstream_requested.emit)
        btn_ray = QPushButton("同步 -N（合成）")
        btn_ray.setToolTip("仅合成流程：把正演页 -N 参数覆盖到本页")
        btn_ray.clicked.connect(self.sync_ray_from_fwd_requested.emit)
        btn_damp = QPushButton("→ gen_damp")
        btn_damp.clicked.connect(self.go_damp_requested.emit)
        btn_vcorr = QPushButton("→ gen_vcorr")
        btn_vcorr.clicked.connect(self.go_vcorr_requested.emit)
        btn_dcorr = QPushButton("→ gen_dcorr")
        btn_dcorr.clicked.connect(self.go_dcorr_requested.emit)
        btn_chk = QPushButton("监视就绪检查")
        btn_chk.clicked.connect(self._show_monitor_check)
        for b in (btn_fill, btn_ray, btn_damp, btn_vcorr, btn_dcorr, btn_chk):
            hl.addWidget(b)
        hl.addStretch(1)
        vl.addWidget(btns)
        self._damp_hint = QLabel()
        self._damp_hint.setWordWrap(True)
        vl.addWidget(self._damp_hint)
        self.insert_widget_before_actions(bar)

        btn_help = QPushButton("-L 日志列说明")
        btn_help.clicked.connect(self._show_log_help)
        self._actions.insertWidget(2, btn_help)

        for key in (*_FIXED_DAMP_KEYS, *_AUTO_DAMP_KEYS):
            edit = self._line_edits.get(key)
            if edit is not None:
                edit.textChanged.connect(self._sync_damp_mutex)  # type: ignore[union-attr]
            prow = self._path_rows.get(key)
            if prow is not None:
                prow.edit.textChanged.connect(self._sync_damp_mutex)
        self._sync_damp_mutex()

        for key in _WATER_MUTEX:
            box = self._check_keys.get(key)
            if box is not None:
                box.toggled.connect(self._sync_water_mutex)

        prow = self._path_rows.get("inv.filter_bound_file")
        if prow is not None:
            prow.edit.setPlaceholderText("可空：用 mesh 海底/地形作上边界")
            prow.edit.textChanged.connect(self._on_filter_bound_changed)
        sf = self._path_rows.get("inv.seafloor_file")
        if sf is not None:
            sf.edit.setPlaceholderText("可空：只做 0/1 时不填")

    def on_state_pushed(self) -> None:
        """配置/工区写入控件后（push 会 blockSignals），补一次互锁。"""
        self._sync_damp_mutex()
        self._sync_water_mutex()

    def _field_text(self, key: str) -> str:
        edit = self._line_edits.get(key)
        if edit is not None:
            return str(edit.text() or "").strip()  # type: ignore[union-attr]
        prow = self._path_rows.get(key)
        if prow is not None:
            return str(prow.edit.text() or "").strip()
        return self.state.get_str(key)

    def _widget_for(self, key: str):
        if key in self._line_edits:
            return self._line_edits[key]
        prow = self._path_rows.get(key)
        if prow is not None:
            return prow.edit
        return None

    def _edited_damp_side(self) -> str | None:
        w = self.sender()
        if w is None:
            return None
        for k in _AUTO_DAMP_KEYS:
            if self._widget_for(k) is w:
                return "auto"
        for k in _FIXED_DAMP_KEYS:
            if self._widget_for(k) is w:
                return "fixed"
        return None

    def _set_damp_mode(self, mode: str) -> None:
        self.state.set("inv.damp_kind", mode)
        if mode == "auto":
            self.set_enabled_keys(list(_FIXED_DAMP_KEYS), False)
            self.set_enabled_keys(list(_AUTO_DAMP_KEYS), True)
        elif mode == "fixed":
            self.set_enabled_keys(list(_AUTO_DAMP_KEYS), False)
            self.set_enabled_keys(list(_FIXED_DAMP_KEYS), True)
        else:
            self.set_enabled_keys(list(_FIXED_DAMP_KEYS), True)
            self.set_enabled_keys(list(_AUTO_DAMP_KEYS), True)
        leftover_fixed = any(self._field_text(k) for k in _FIXED_DAMP_KEYS)
        leftover_auto = any(self._field_text(k) for k in _AUTO_DAMP_KEYS)
        if mode == "auto":
            extra = (
                "；固定 -D 已锁定"
                + ("（灰显框内旧值不会进命令行）" if leftover_fixed else "")
            )
            self._damp_hint.setText("当前阻尼：自动 -T" + extra)
            self._damp_hint.setStyleSheet("color: #1565c0;")
        elif mode == "fixed":
            extra = (
                "；自动 -T 已锁定"
                + ("（灰显框内旧值不会进命令行）" if leftover_auto else "")
            )
            self._damp_hint.setText("当前阻尼：固定 -D" + extra)
            self._damp_hint.setStyleSheet("color: #1565c0;")
        elif mode == "conflict":
            self._damp_hint.setText(
                "固定 -D 与自动 -T 都有值，二者互斥：请清空一侧后再运行。"
            )
            self._damp_hint.setStyleSheet("color: #c62828;")
        else:
            self._damp_hint.setText(
                "当前阻尼：未设置（-T 与 -D 互斥，填一侧即锁定另一侧）。"
            )
            self._damp_hint.setStyleSheet("")

    def _sync_damp_mutex(self, *_args) -> None:
        """-T（自动）与 -D（固定，含 -DQ）互斥：填一侧则锁定另一侧。"""
        auto_on = any(self._field_text(k) for k in _AUTO_DAMP_KEYS)
        fixed_on = any(self._field_text(k) for k in _FIXED_DAMP_KEYS)
        side = self._edited_damp_side()
        saved = (self.state.get_str("inv.damp_kind") or "").strip()
        if auto_on and not fixed_on:
            mode = "auto"
        elif fixed_on and not auto_on:
            mode = "fixed"
        elif auto_on and fixed_on:
            if side == "auto":
                mode = "auto"
            elif side == "fixed":
                mode = "fixed"
            elif saved in ("auto", "fixed"):
                mode = saved
            else:
                mode = "conflict"
        else:
            mode = "none"
        self._set_damp_mode(mode)

    def _sync_water_mutex(self, *_args) -> None:
        """-y 与 -w 互斥；默认都不勾（0/1 核与原来相同）。"""
        yw = self._check_keys.get("inv.invert_water_only")
        cw = self._check_keys.get("inv.invert_crust_only")
        if yw is None or cw is None:
            return
        if yw.isChecked() and cw.isChecked():
            other = cw if self.sender() is yw else yw
            other.blockSignals(True)
            other.setChecked(False)
            other.blockSignals(False)
            self.state.set(
                "inv.invert_crust_only" if other is cw else "inv.invert_water_only",
                False,
            )

    def _on_filter_bound_changed(self, *_args) -> None:
        """填了边界文件则自动勾上 -s；清空文件并保持勾选 = 用 mesh 地形。"""
        if not self._field_text("inv.filter_bound_file"):
            return
        box = self._check_keys.get("inv.apply_filter")
        if box is not None and not box.isChecked():
            box.setChecked(True)

    def _show_log_help(self) -> None:
        from pyAOBS.modeling.tomo2d.help_docs import TomoHelp

        show_modeless_text(
            "tt_inverse -L 日志说明",
            TomoHelp.tt_inverse_logfile_format_help(),
        )

    def _show_monitor_check(self) -> None:
        self.pull()
        show_modeless_text("监视就绪检查", format_monitor_checklist(self.state))
