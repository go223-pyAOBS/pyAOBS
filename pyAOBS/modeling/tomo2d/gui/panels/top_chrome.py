"""可折叠顶栏：顶行工区操作 + 路径与并行并排同高。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...param_hints import apply_param_tooltip
from ..services.paths import resolve_work_dir
from ..state.form_state import FormState
from ..widgets.form_rows import FormBinder, PathRow
from .parallel_env_panel import ParallelEnvPanel


class TopChromePanel(QToolBar):
    hint_key_changed = Signal(str)
    new_project_requested = Signal()
    open_project_requested = Signal()
    save_project_requested = Signal()
    save_profile_requested = Signal()
    load_profile_requested = Signal()
    plot_smesh_requested = Signal()
    inv_analysis_requested = Signal()
    inv_monitor_requested = Signal()
    model_picker_requested = Signal()
    model_compare_requested = Signal()
    help_requested = Signal()
    exit_requested = Signal()

    def __init__(self, state: FormState, parent=None) -> None:
        super().__init__("路径与全局", parent)
        self.setObjectName("tomoTopChrome")
        self.setMovable(False)
        self.state = state
        self.binder = FormBinder(state)

        host = QWidget()
        outer = QVBoxLayout(host)
        outer.setContentsMargins(4, 2, 4, 2)
        outer.setSpacing(4)

        # 常显顶行：工区 / 配置 / 工具 / 帮助 / 写日志 / 退出
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        for text, sig, tip in (
            ("新建工区", self.new_project_requested, "选择目录并创建工区布局与 meta/tomo2d_project.json"),
            ("打开工区", self.open_project_requested, "打开 tomo2d_project.json 工区"),
            ("保存工区", self.save_project_requested, "将当前表单写回工区 JSON"),
            ("导出配置", self.save_profile_requested, "导出独立配置 JSON（非工区）"),
            ("导入配置", self.load_profile_requested, "导入配置 JSON 或运行包 manifest"),
            ("绘制 smesh…", self.plot_smesh_requested, "按当前命令页签上的 smesh/界面路径绘图；没有则打开空图窗，再「打开…」/拖入/粘贴。也可把 .smesh / v.in / .grd 拖到主窗口"),
            ("反演结果分析…", self.inv_analysis_requested, "分析 tt_inverse -L 日志"),
            ("反演监视…", self.inv_monitor_requested, "准实时监视 χ²/RMS 与最新写出 smesh"),
            ("模型挑选…", self.model_picker_requested, "从 runs/ 选一次反演，再挑各轮 smesh 写回表单"),
            ("模型对比…", self.model_compare_requested, "选 A/B 两个 smesh 绘制差值 B−A（上 ΔV / 中 B / 下 A）"),
            ("帮助", self.help_requested, "打开帮助（F1）：GUI 说明与各程序章节"),
        ):
            btn = QPushButton(text)
            btn.setToolTip(tip)
            btn.clicked.connect(sig.emit)
            actions.addWidget(btn)

        self.write_log = QCheckBox("写 tomo2d_gui.log")
        if not state.has("gui.write_file_log"):
            state.set("gui.write_file_log", True)
        self.binder.bind_check("gui.write_file_log", self.write_log)
        apply_param_tooltip(self.write_log, "gui.write_file_log")
        actions.addWidget(self.write_log)

        actions.addStretch(1)
        btn_exit = QPushButton("退出")
        btn_exit.setObjectName("tomoExitBtn")
        btn_exit.setToolTip("退出程序（未保存工区时会提示）")
        btn_exit.clicked.connect(self.exit_requested.emit)
        actions.addWidget(btn_exit)
        outer.addLayout(actions)

        self._toggle = QToolButton()
        self._toggle.setText("▾ 折叠路径与并行")
        self._toggle.setCheckable(True)
        self._toggle.setChecked(True)
        self._toggle.toggled.connect(self._on_toggle)
        outer.addWidget(self._toggle)

        self._body = QWidget()
        body = QVBoxLayout(self._body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(4)

        # 路径（左）与并行（右）并排、垂直同高
        path_col = QWidget()
        path_lay = QVBoxLayout(path_col)
        path_lay.setContentsMargins(0, 0, 0, 0)
        path_lay.setSpacing(4)
        path_col.setFixedWidth(640)
        path_col.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding
        )

        from ..services.bin_defaults import default_bin_path

        self.bin_row = PathRow(
            "bin_path:",
            mode="dir",
            # 可执行目录必须绝对路径：子进程 cwd=work_dir，相对 bin 会找不到
            work_dir_getter=lambda: Path(
                self.state.get_str("bin_path") or default_bin_path() or "."
            ),
            keep_absolute=True,
        )
        self.work_row = PathRow(
            "work_dir:",
            mode="dir",
            work_dir_getter=lambda: resolve_work_dir(self.state.get_str("work_dir")),
            keep_absolute=True,
        )

        if not state.has("bin_path") or not state.get_str("bin_path"):
            state.set("bin_path", default_bin_path())
        if not state.has("work_dir"):
            state.set("work_dir", str(Path.cwd()))
        self.binder.bind_line("bin_path", self.bin_row.edit)
        self.binder.bind_line("work_dir", self.work_row.edit)
        # 旧工程可能存了相对 bin_path，打开时抬成绝对
        bp = state.get_str("bin_path")
        if bp and not Path(bp).expanduser().is_absolute():
            try:
                abs_bp = str(Path(bp).expanduser().resolve()).replace("\\", "/")
                state.set("bin_path", abs_bp)
                bp = abs_bp
            except OSError:
                pass
        self.bin_row.edit.setText(bp)
        self.work_row.edit.setText(state.get_str("work_dir"))
        for row, key in (
            (self.bin_row, "bin_path"),
            (self.work_row, "work_dir"),
        ):
            apply_param_tooltip(row, key)
            apply_param_tooltip(row.label, key)
            apply_param_tooltip(row.edit, key)

        from ..services.smesh_plot_core import (
            DEFAULT_SMESH_CMAP_ID,
            SMESH_CMAP_KEY,
            alias_builtin_smesh_cmap_id,
            set_smesh_cmap_id,
        )
        from ..widgets.smesh_cmap_combo import SmeshCmapCombo

        raw_cmap = state.get_str(SMESH_CMAP_KEY) if state.has(SMESH_CMAP_KEY) else ""
        aliased = alias_builtin_smesh_cmap_id(raw_cmap)
        if not raw_cmap:
            set_smesh_cmap_id(state, DEFAULT_SMESH_CMAP_ID)
        elif aliased is not None:
            set_smesh_cmap_id(state, aliased)
        self.cmap_combo = SmeshCmapCombo(state)
        self.cmap_combo._label.setFixedWidth(64)
        self.cmap_combo._label.setText("色标:")

        for prow in (self.bin_row, self.work_row):
            prow.label.setFixedWidth(64)
            prow.btn.setText("…")
            prow.btn.setFixedWidth(26)
            prow.btn.setToolTip("浏览…")

        path_lay.addWidget(self.bin_row)
        path_lay.addWidget(self.work_row)
        path_lay.addWidget(self.cmap_combo)
        path_lay.addStretch(1)

        self.parallel = ParallelEnvPanel(state, self.binder, self._body)
        self.parallel.hint_key_changed.connect(self.hint_key_changed.emit)
        self.parallel.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
        )

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.setAlignment(Qt.AlignmentFlag.AlignTop)
        row.addWidget(path_col, stretch=0)
        row.addWidget(self.parallel, stretch=0)
        row.addStretch(1)
        body.addLayout(row)

        note = QLabel(
            "工区：meta/ · inputs/ · outputs/ · runs/ · cache/；"
            "F1 帮助；并行：常显 OMP，加速/开发者折叠；绘图：滚轮缩放 · 左拖平移 · 右拖缩放 · 双击复位"
        )
        note.setStyleSheet("color:#556;")
        body.addWidget(note)

        outer.addWidget(self._body)
        self.addWidget(host)

    def _on_toggle(self, checked: bool) -> None:
        self._body.setVisible(checked)
        self._toggle.setText(
            "▾ 折叠路径与并行" if checked else "▸ 展开路径与并行"
        )
