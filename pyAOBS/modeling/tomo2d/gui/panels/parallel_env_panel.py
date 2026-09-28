"""顶栏：并行 / 策略环境变量（常显 OMP；加速与开发者折叠）。"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...param_hints import apply_param_tooltip
from ..services.run_env import ensure_run_env_defaults
from ..state.form_state import FormState
from ..widgets.field_form import CollapsibleSection
from ..widgets.form_rows import FormBinder

_DEFAULT_REUSE_THRESH = "1e-3"

# 预设：只改 env.* 开关，不改表单反演参数
_PRESETS: dict[str, dict] = {
    "自定义": {},
    "快速（OMP+复用）": {
        "env.inv_omp": True,
        "env.fwd_omp": True,
        "env.inv_reuse_forward": True,
        "env.inv_reuse_thresh": _DEFAULT_REUSE_THRESH,
        "env.inv_coarse2fine": False,
        "env.inv_legacy_baseline": False,
        "env.inv_lsqr_precond": False,
        "env.inv_sens_weight": False,
        "env.inv_linesearch": False,
        "env.inv_lm": False,
        "env.inv_diag": False,
        "env.graph_fs_enum": True,
    },
    "稳健（OMP+C2F）": {
        "env.inv_omp": True,
        "env.fwd_omp": True,
        "env.inv_reuse_forward": False,
        "env.inv_coarse2fine": True,
        "env.inv_legacy_baseline": False,
        "env.inv_lsqr_precond": False,
        "env.inv_sens_weight": False,
        "env.inv_linesearch": False,
        "env.inv_lm": False,
        "env.inv_diag": False,
        "env.graph_fs_enum": True,
    },
    "对拍（Legacy）": {
        "env.inv_legacy_baseline": True,
        "env.inv_reuse_forward": False,
        "env.inv_coarse2fine": False,
        "env.inv_lsqr_precond": False,
        "env.inv_sens_weight": False,
        "env.inv_linesearch": False,
        "env.inv_lm": False,
        "env.inv_diag": False,
        "env.graph_fs_enum": False,
    },
    "调试（DIAG）": {
        "env.inv_diag": True,
        "env.inv_legacy_baseline": False,
        "env.inv_linesearch": False,
        "env.inv_lm": False,
    },
}


class _HintFocusFilter(QObject):
    def __init__(self, key: str, emit_fn, parent=None) -> None:
        super().__init__(parent)
        self._key = key
        self._emit = emit_fn

    def eventFilter(self, obj, event) -> bool:  # noqa: ANN001
        if event.type() == QEvent.Type.FocusIn:
            self._emit(self._key)
        return False


def _edit(placeholder: str = "", *, width: int = 56) -> QLineEdit:
    ed = QLineEdit()
    ed.setPlaceholderText(placeholder)
    ed.setFixedWidth(width)
    ed.setFixedHeight(26)
    return ed


class ParallelEnvPanel(QWidget):
    """并行与策略：常显 OMP；加速 / 开发者默认折叠。"""

    hint_key_changed = Signal(str)

    def __init__(self, state: FormState, binder: FormBinder, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.binder = binder
        self._applying_preset = False
        ensure_run_env_defaults(state)
        self.setMinimumWidth(560)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        frame = QFrame(self)
        frame.setObjectName("tomoParallelFrame")
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(4)

        # —— 常显：OMP + 预设 ——
        r1 = QHBoxLayout()
        r1.setSpacing(8)
        title = QLabel("并行/策略")
        title.setStyleSheet("font-weight:600;color:#334;")
        apply_param_tooltip(title, "tab.parallel_env")
        lb_thr = QLabel("OMP线程")
        self.omp_threads = _edit("空=系统", width=56)
        self.ck_inv_omp = QCheckBox("tt_inverse并行")
        self.ck_fwd_omp = QCheckBox("tt_forward并行")
        self.ck_graph_fs = QCheckBox("图论FS枚举")
        lb_preset = QLabel("预设")
        self.preset = QComboBox()
        self.preset.addItems(list(_PRESETS.keys()))
        self.preset.setFixedWidth(150)
        self.preset.setToolTip(
            "一键组合：快速=OMP+前向复用+图论FS；稳健=OMP+C2F+图论FS；对拍=Legacy且图论扫C/B；调试=DIAG"
        )
        for w in (
            title,
            lb_thr,
            self.omp_threads,
            self.ck_inv_omp,
            self.ck_fwd_omp,
            self.ck_graph_fs,
            lb_preset,
            self.preset,
        ):
            r1.addWidget(w)
        r1.addStretch(1)
        lay.addLayout(r1)

        # —— 加速策略 | 开发者（同一行，默认折叠）——
        accel = CollapsibleSection("加速策略", expanded=False)
        ar = QHBoxLayout()
        ar.setSpacing(6)
        self.ck_reuse = QCheckBox("前向复用")
        lb_reuse = QLabel("阈值")
        self.reuse_thresh = _edit(_DEFAULT_REUSE_THRESH, width=52)
        self.ck_c2f = QCheckBox("分阶段C2F")
        self.ck_lsqr_precond = QCheckBox("LSQR列预条件")
        lb_precond = QLabel("上限")
        self.precond_max = _edit("10", width=40)
        self.ck_sens_weight = QCheckBox("灵敏度加权")
        lb_sens = QLabel("κ")
        self.sens_kappa = _edit("10", width=40)
        self.ck_linesearch = QCheckBox("线搜索")
        self.ck_lm = QCheckBox("LM信赖域")
        for w in (
            self.ck_reuse,
            lb_reuse,
            self.reuse_thresh,
            self.ck_c2f,
            self.ck_lsqr_precond,
            lb_precond,
            self.precond_max,
            self.ck_sens_weight,
            lb_sens,
            self.sens_kappa,
            self.ck_linesearch,
            self.ck_lm,
        ):
            ar.addWidget(w)
        ar.addStretch(1)
        accel.body_layout.addLayout(ar)

        c2f_sec = CollapsibleSection(
            "C2F 系数（留空=默认 3→1）", expanded=False, parent=accel.body
        )
        cr = QHBoxLayout()
        cr.setSpacing(4)
        self.c2f_edits: dict[str, QLineEdit] = {}
        self.c2f_labels: dict[str, QLabel] = {}
        c2f_specs = (
            ("env.inv_c2f_smooth_start", "平滑起", "3.0"),
            ("env.inv_c2f_smooth_end", "平滑终", "1.0"),
            ("env.inv_c2f_damp_start", "阻尼起", "3.0"),
            ("env.inv_c2f_damp_end", "阻尼终", "1.0"),
        )
        for key, lab, ph in c2f_specs:
            lb = QLabel(lab)
            ed = _edit(ph, width=40)
            cr.addWidget(lb)
            cr.addWidget(ed)
            self.c2f_edits[key] = ed
            self.c2f_labels[key] = lb
            binder.bind_line(key, ed)
            self._wire(ed, key)
            self._wire(lb, key)
        cr.addStretch(1)
        c2f_sec.body_layout.addLayout(cr)
        accel.body_layout.addWidget(c2f_sec)

        dev = CollapsibleSection("开发者/对拍", expanded=False)
        dr = QHBoxLayout()
        dr.setSpacing(6)
        self.ck_legacy = QCheckBox("Legacy基线")
        self.ck_diag = QCheckBox("诊断DIAG")
        self.ck_status = QCheckBox("status.jsonl")
        self.ck_status.setToolTip(
            "写 TOMO2D_INV_STATUS_JSONL（默认 outputs/status.jsonl），"
            "供反演监视读取。需较新 tt_inverse。"
        )
        dr.addWidget(self.ck_legacy)
        dr.addWidget(self.ck_diag)
        dr.addWidget(self.ck_status)
        dr.addStretch(1)
        dev.body_layout.addLayout(dr)

        row2 = QHBoxLayout()
        row2.setSpacing(6)
        row2.setContentsMargins(0, 0, 0, 0)
        accel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        dev.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        row2.addWidget(accel, stretch=3)
        row2.addWidget(dev, stretch=2)
        lay.addLayout(row2)
        lay.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame, stretch=1)

        binder.bind_line("env.omp_num_threads", self.omp_threads)
        binder.bind_check("env.inv_omp", self.ck_inv_omp)
        binder.bind_check("env.fwd_omp", self.ck_fwd_omp)
        binder.bind_check("env.inv_legacy_baseline", self.ck_legacy)
        binder.bind_check("env.inv_diag", self.ck_diag)
        binder.bind_check("env.inv_reuse_forward", self.ck_reuse)
        binder.bind_line("env.inv_reuse_thresh", self.reuse_thresh)
        binder.bind_check("env.inv_coarse2fine", self.ck_c2f)
        binder.bind_check("env.inv_lsqr_precond", self.ck_lsqr_precond)
        binder.bind_line("env.inv_lsqr_precond_max", self.precond_max)
        binder.bind_check("env.inv_sens_weight", self.ck_sens_weight)
        binder.bind_line("env.inv_sens_kappa", self.sens_kappa)
        binder.bind_check("env.inv_linesearch", self.ck_linesearch)
        binder.bind_check("env.inv_lm", self.ck_lm)
        binder.bind_check("env.graph_fs_enum", self.ck_graph_fs)
        # status.jsonl：用勾选同步路径（默认 outputs/status.jsonl）
        ensure_run_env_defaults(state)
        path_now = state.get_str("env.inv_status_jsonl_path")
        self.ck_status.setChecked(bool(path_now.strip()))
        apply_param_tooltip(self.ck_status, "env.inv_status_jsonl_path")
        self.ck_status.toggled.connect(self._on_status_jsonl_toggled)

        tip_pairs: list[tuple[QWidget, QWidget | None, str]] = [
            (title, None, "tab.parallel_env"),
            (lb_thr, self.omp_threads, "env.omp_num_threads"),
            (self.ck_inv_omp, None, "env.inv_omp"),
            (self.ck_fwd_omp, None, "env.fwd_omp"),
            (self.ck_legacy, None, "env.inv_legacy_baseline"),
            (self.ck_diag, None, "env.inv_diag"),
            (self.ck_reuse, None, "env.inv_reuse_forward"),
            (lb_reuse, self.reuse_thresh, "env.inv_reuse_thresh"),
            (self.ck_c2f, None, "env.inv_coarse2fine"),
            (self.ck_lsqr_precond, None, "env.inv_lsqr_precond"),
            (lb_precond, self.precond_max, "env.inv_lsqr_precond_max"),
            (self.ck_sens_weight, None, "env.inv_sens_weight"),
            (lb_sens, self.sens_kappa, "env.inv_sens_kappa"),
            (self.ck_linesearch, None, "env.inv_linesearch"),
            (self.ck_lm, None, "env.inv_lm"),
            (self.ck_graph_fs, None, "env.graph_fs_enum"),
        ]
        for primary, secondary, key in tip_pairs:
            self._wire(primary, key)
            if secondary is not None:
                self._wire(secondary, key)
            elif isinstance(primary, QCheckBox):
                primary.toggled.connect(
                    lambda *_, k=key: self.hint_key_changed.emit(k)
                )

        self.ck_legacy.toggled.connect(lambda *_: self._on_strategy_changed())
        self.ck_reuse.toggled.connect(lambda *_: self._on_reuse_toggled())
        self.ck_c2f.toggled.connect(lambda *_: self._on_strategy_changed())
        self.ck_lsqr_precond.toggled.connect(lambda *_: self._on_strategy_changed())
        self.ck_sens_weight.toggled.connect(lambda *_: self._on_strategy_changed())
        self.ck_linesearch.toggled.connect(lambda *_: self._on_strategy_changed())
        self.ck_lm.toggled.connect(lambda *_: self._on_strategy_changed())
        self.ck_graph_fs.toggled.connect(lambda *_: self._mark_custom_preset())
        self.precond_max.editingFinished.connect(lambda: self._mark_custom_preset())
        self.sens_kappa.editingFinished.connect(lambda: self._mark_custom_preset())
        self.ck_diag.toggled.connect(lambda *_: self._mark_custom_preset())
        self.ck_inv_omp.toggled.connect(lambda *_: self._mark_custom_preset())
        self.ck_fwd_omp.toggled.connect(lambda *_: self._mark_custom_preset())
        self.preset.currentTextChanged.connect(self._on_preset)
        self._sync_enabled()

    def _wire(self, widget: QWidget, key: str) -> None:
        apply_param_tooltip(widget, key)
        filt = _HintFocusFilter(key, self.hint_key_changed.emit, widget)
        widget.installEventFilter(filt)

    def _on_reuse_toggled(self) -> None:
        if self.ck_reuse.isChecked() and not self.reuse_thresh.text().strip():
            self.reuse_thresh.setText(_DEFAULT_REUSE_THRESH)
            self.state.set("env.inv_reuse_thresh", _DEFAULT_REUSE_THRESH)
        self._on_strategy_changed()

    def _on_status_jsonl_toggled(self, checked: bool) -> None:
        if checked:
            cur = self.state.get_str("env.inv_status_jsonl_path").strip()
            self.state.set(
                "env.inv_status_jsonl_path", cur or "outputs/status.jsonl"
            )
        else:
            self.state.set("env.inv_status_jsonl_path", "")
        self._mark_custom_preset()

    def _on_strategy_changed(self) -> None:
        self._sync_enabled()
        self._mark_custom_preset()

    def _mark_custom_preset(self) -> None:
        if self._applying_preset:
            return
        if self.preset.currentText() != "自定义":
            self.preset.blockSignals(True)
            self.preset.setCurrentText("自定义")
            self.preset.blockSignals(False)

    def _on_preset(self, name: str) -> None:
        spec = _PRESETS.get(name) or {}
        if not spec:
            return
        self._applying_preset = True
        try:
            for key, val in spec.items():
                self.state.set(key, val)
            self.binder.push_to_widgets()
            self._sync_enabled()
        finally:
            self._applying_preset = False

    def _sync_enabled(self) -> None:
        legacy = self.ck_legacy.isChecked()
        reuse_ok = self.ck_reuse.isChecked() and not legacy
        c2f_ok = self.ck_c2f.isChecked() and not legacy
        self.ck_reuse.setEnabled(not legacy)
        self.ck_c2f.setEnabled(not legacy)
        self.ck_lsqr_precond.setEnabled(not legacy)
        self.reuse_thresh.setEnabled(reuse_ok)
        self.precond_max.setEnabled((not legacy) and self.ck_lsqr_precond.isChecked())
        self.sens_kappa.setEnabled(self.ck_sens_weight.isChecked())
        for ed in self.c2f_edits.values():
            ed.setEnabled(c2f_ok)
        for lb in self.c2f_labels.values():
            lb.setEnabled(c2f_ok)
