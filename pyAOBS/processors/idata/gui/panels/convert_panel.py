# -*- coding: utf-8 -*-
"""转换阶段：OBEM→SAC / SAC→SEGY / RAW→SAC。"""

from __future__ import annotations

from PySide6.QtCore import Signal
from ..dialog_utils import show_modeless_message
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

try:
    from pyAOBS.utils.qt_combo import connect_combo_deferred
except ImportError:
    def connect_combo_deferred(combo, slot):  # type: ignore
        combo.currentIndexChanged.connect(slot)

from ..services.convert_runner import ConvertRunner
from ..services.env_context import IdataEnvContext
from ..services.segy_dataset import SegyDataset, convert_segy_to_su

# 各页说明（输入 / 输出 / config / 格式）
_HELP_OBEM = """\
【脚本】obem_tsm_to_sac_obspy.py

【输入】
• config.ini — 唯一参数；其中指定 TSM 目录、台站、采样率等
• TSM 原始块文件：位于 config 的 input_path（512MB 块，binaryformat=1/2）

【输出】
• 多道 SAC：写入 config 的 output_path
• 分量多为 Ex/Ey/Hx/Hy/FHz 等（文件名由脚本按台站与时间生成）

【config 格式】INI，示例见 processors/raw2sac/obem_config_example.ini
• [Paths] input_path / output_path / location_file(可选)
• [Station] station_name, latitude, longitude, elevation, network
• [Parameters] original_srate, target_srate, data_length(秒), et, binaryformat(1=TSM_NEW,2=TSM_OLD), year
"""

_HELP_SAC2Y = """\
【脚本】sac2y_v2_1_obspy.py

【输入】
• sac_file — 单分量连续 SAC（含时间与可选台站位置）
• ukooa_file — UKOOA 炮点表（炮号、时间、经纬度、水深等）
• config.ini — 切段长度、投影、漂移、台站位置等

【输出】
• output_segy — 标准 SEGY（3200+400 卷头 + 道；样点 IBM float；道头 big-endian）
• 几何约定：sx/sy=炮，gx/gy=OBS；scalco=scalel=-1（米）
• 「写出角色」选旧对调时，转换后交换 s*/g*（兼容历史 geom=obs 数据）

【config 格式】INI，示例见 processors/raw2sac/sac2y_config_example.ini
• [Parameters] rv(声速), length(道长秒), tcoor, lon0(投影中央经度), ex/ey, et
• [Drift] 可选 start_time/end_time/drift（时钟漂移）
• [Location] longitude/latitude 或 location_file（SAC 头无台站坐标时用）
"""

_HELP_RAW = """\
【脚本】raw2sac_v1_1_obspy.py

【输入】
• fileName — OBS 原始二进制连续记录（无 config；参数在命令行）
• sps — 采样率 Hz（如 1000）
• TC — 时间校正相关参数（与 C 版 raw2sac 一致，常用 256）

【输出】
• 当前工作目录下若干分量文件（无点扩展名）：*.shx / *.shy / *.shz [/ *.hyd]
• 实为 SAC 格式；命名取自输入文件名片段 + 三分量/四分量扩展

【说明】
• 无独立 INI；3 通道短周期或 4 通道（含 hydrophone）由文件头自动判断
• 产出在执行 cwd（Workbench 下多为 run outputs 目录）
"""

_HELP_SEGY2SU = """\
【脚本/服务】segy2su.py（idata SegyDataset.export_su）

【输入】
• input_segy — SEGY/SGY（含 3600 字节卷头；道头多为 big-endian；format=1 为 IBM 浮点）

【输出】
• output_su — Seismic Unix SU（无卷头；道头+IEEE float；默认 little-endian）
• 道头几何字段原样拷贝（炮=sx/sy，OBS=gx/gy）

【说明】
• 无 config 文件
• format=1 → IBM 转 IEEE；format=5 等按 IEEE 处理
"""

_HELP_PIPE = {
    "sac2segy": """\
【流水线】SAC → SEGY　　【脚本】sac2segy.py → sac2y

【输入】sac（SAC）+ ukooa（UKOOA 炮点）+ sac2y_config（INI）
【输出】单个 .segy 文件（炮=sx/sy，OBS=gx/gy；IBM；卷头齐全）
【config】同「SAC → SEGY」页的 sac2y_config_example.ini
【channels】本模式不用
""",
    "sac2su": """\
【流水线】SAC → SEGY → SU　　【脚本】sac2su.py

【输入】sac + ukooa + sac2y_config
【输出】单个 .su（经中间 SEGY；默认不保留 .segy，除非 CLI --keep-segy）
【config】sac2y INI（切段/投影/漂移/台站）
【channels】本模式不用
""",
    "raw2segy": """\
【流水线】RAW → SAC → SEGY　　【脚本】raw2segy.py

【输入】raw 文件 + sps + TC + ukooa + sac2y_config
【输出】目录：各分量 *.segy（如 stem_shz.segy）；中间 SAC 在 out_dir/_work_sac
【config】sac2y INI；RAW 步无 INI
【channels】all 或 shx,shy,shz,hyd（过滤分量）
""",
    "raw2su": """\
【流水线】RAW → SAC → SEGY → SU　　【脚本】raw2su.py

【输入】raw + sps + TC + ukooa + sac2y_config
【输出】目录：各分量 *.su；可选 --keep-segy 保留中间 SEGY
【config】sac2y INI；RAW 步无 INI
【channels】all 或 shx,shy,shz,hyd
""",
    "obem2segy": """\
【流水线】OBEM → SAC → SEGY　　【脚本】obem2segy.py

【输入】obem_config（INI，含 TSM 路径）+ ukooa + sac2y_config
【输出】目录：各 SAC 对应 *.segy；SAC 先写到 obem 的 output_path
【config】obem_config_example.ini + sac2y_config_example.ini
【channels】按 SAC 文件名后缀过滤，默认 all
""",
    "obem2su": """\
【流水线】OBEM → SAC → SEGY → SU　　【脚本】obem2su.py

【输入】obem_config + ukooa + sac2y_config
【输出】目录：各分量/通道 *.su
【config】obem INI + sac2y INI
【channels】默认 all；可按 SAC 名过滤
""",
}


def _help_label(text: str) -> QLabel:
    lab = QLabel(text.strip())
    lab.setWordWrap(True)
    lab.setStyleSheet(
        "QLabel {"
        " background:#f8fafc; color:#334155; border:1px solid #e2e8f0;"
        " border-radius:4px; padding:8px; font-size:12px;"
        "}"
    )
    return lab


class ConvertPanel(QWidget):
    request_load_segy = Signal(str)  # path
    fields_changed = Signal()

    def __init__(
        self,
        env: IdataEnvContext,
        runner: ConvertRunner,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.env = env
        self.runner = runner
        self._last_segy_out = ""
        self._pending_obs_remap = False
        self._build_ui()
        self.runner.job_finished.connect(self._on_job_finished)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        tip = QLabel("各标签页顶部有输入/输出/config/格式说明；后端为 raw2sac 下 CLI。")
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#475569;")
        root.addWidget(tip)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, stretch=1)

        self.obem_config = QLineEdit()
        self.sac2y_sac = QLineEdit()
        self.sac2y_ukooa = QLineEdit()
        self.sac2y_output = QLineEdit()
        self.sac2y_config = QLineEdit()
        self.raw_file = QLineEdit()
        self.raw_sps = QLineEdit("1000")
        self.raw_tc = QLineEdit("256")
        self.segy2su_in = QLineEdit()
        self.segy2su_out = QLineEdit()
        # 一键流水线字段
        self.pipe_mode = QComboBox()
        self.pipe_sac = QLineEdit()
        self.pipe_raw = QLineEdit()
        self.pipe_sps = QLineEdit("1000")
        self.pipe_tc = QLineEdit("256")
        self.pipe_obem = QLineEdit()
        self.pipe_ukooa = QLineEdit()
        self.pipe_sac2y_cfg = QLineEdit()
        self.pipe_out = QLineEdit()
        self.pipe_channels = QLineEdit("all")

        self.tabs.addTab(self._build_obem_tab(), "OBEM TSM → SAC")
        self.tabs.addTab(self._build_sac2y_tab(), "SAC → SEGY")
        self.tabs.addTab(self._build_raw_tab(), "RAW → SAC")
        self.tabs.addTab(self._build_segy2su_tab(), "SEGY → SU")
        self.tabs.addTab(self._build_pipeline_tab(), "一键流水线")

        for w, name in (
            (self.obem_config, "obem.config_file"),
            (self.sac2y_sac, "sac2y.sac_file"),
            (self.sac2y_ukooa, "sac2y.ukooa_file"),
            (self.sac2y_output, "sac2y.output_segy_file"),
            (self.sac2y_config, "sac2y.config_file"),
            (self.raw_file, "raw2sac.file_name"),
            (self.raw_sps, "raw2sac.sps"),
            (self.raw_tc, "raw2sac.tc"),
            (self.segy2su_in, "segy2su.input"),
            (self.segy2su_out, "segy2su.output"),
            (self.pipe_sac, "pipe.sac"),
            (self.pipe_raw, "pipe.raw"),
            (self.pipe_sps, "pipe.sps"),
            (self.pipe_tc, "pipe.tc"),
            (self.pipe_obem, "pipe.obem"),
            (self.pipe_ukooa, "pipe.ukooa"),
            (self.pipe_sac2y_cfg, "pipe.sac2y_cfg"),
            (self.pipe_out, "pipe.out"),
            (self.pipe_channels, "pipe.channels"),
        ):
            w.textChanged.connect(lambda _t, n=name, edit=w: self._on_field(n, edit.text()))
        connect_combo_deferred(self.pipe_mode, lambda _i: self._on_pipe_mode())

    def _build_obem_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(_help_label(_HELP_OBEM))
        form = QFormLayout()
        row = QHBoxLayout()
        row.addWidget(self.obem_config, stretch=1)
        btn = QPushButton("浏览…")
        btn.clicked.connect(
            lambda: self._browse_open(
                self.obem_config,
                "Select OBEM config file",
                "Config/INI (*.ini *.cfg *.txt);;All (*.*)",
            )
        )
        row.addWidget(btn)
        form.addRow("config_file", row)
        lay.addLayout(form)
        run = QPushButton("运行转换")
        run.clicked.connect(self._run_obem)
        lay.addWidget(run)
        lay.addStretch(1)
        return w

    def _build_sac2y_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(_help_label(_HELP_SAC2Y))
        form = QFormLayout()

        def path_row(edit: QLineEdit, title: str, save: bool = False) -> QHBoxLayout:
            row = QHBoxLayout()
            row.addWidget(edit, stretch=1)
            btn = QPushButton("浏览…")
            if save:
                btn.clicked.connect(lambda: self._browse_save(edit, title))
            else:
                btn.clicked.connect(lambda: self._browse_open(edit, title))
            row.addWidget(btn)
            return row

        form.addRow("sac_file", path_row(self.sac2y_sac, "Select SAC file"))
        form.addRow("ukooa_file", path_row(self.sac2y_ukooa, "Select UKOOA file"))
        form.addRow(
            "output_segy_file",
            path_row(self.sac2y_output, "Select output SEGY file", save=True),
        )
        form.addRow(
            "config_file",
            path_row(self.sac2y_config, "Select config file"),
        )

        self.geom_role = QComboBox()
        self.geom_role.addItem("约定：sx=炮, gx=OBS", "segy")
        self.geom_role.addItem("旧对调：sx=OBS, gx=炮", "obs")
        self.geom_role.setCurrentIndex(0)  # 默认约定
        connect_combo_deferred(self.geom_role, lambda _i: self.fields_changed.emit())
        form.addRow("写出角色", self.geom_role)
        lay.addLayout(form)

        row_btn = QHBoxLayout()
        run = QPushButton("运行转换")
        run.clicked.connect(self._run_sac2y)
        load = QPushButton("加载产出 SEGY")
        load.clicked.connect(self._load_output)
        row_btn.addWidget(run)
        row_btn.addWidget(load)
        row_btn.addStretch(1)
        lay.addLayout(row_btn)
        lay.addStretch(1)
        return w

    def _build_raw_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(_help_label(_HELP_RAW))
        form = QFormLayout()
        row = QHBoxLayout()
        row.addWidget(self.raw_file, stretch=1)
        btn = QPushButton("浏览…")
        btn.clicked.connect(lambda: self._browse_open(self.raw_file, "Select raw data file"))
        row.addWidget(btn)
        form.addRow("fileName", row)
        form.addRow("sps", self.raw_sps)
        form.addRow("TC", self.raw_tc)
        lay.addLayout(form)
        run = QPushButton("运行转换")
        run.clicked.connect(self._run_raw2sac)
        lay.addWidget(run)
        lay.addStretch(1)
        return w

    def _build_segy2su_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(_help_label(_HELP_SEGY2SU))
        form = QFormLayout()
        row_in = QHBoxLayout()
        row_in.addWidget(self.segy2su_in, stretch=1)
        btn_in = QPushButton("浏览…")
        btn_in.clicked.connect(
            lambda: self._browse_open(
                self.segy2su_in,
                "Select SEGY file",
                "SEGY (*.segy *.sgy);;All (*.*)",
            )
        )
        row_in.addWidget(btn_in)
        form.addRow("input_segy", row_in)

        row_out = QHBoxLayout()
        row_out.addWidget(self.segy2su_out, stretch=1)
        btn_out = QPushButton("浏览…")
        btn_out.clicked.connect(lambda: self._browse_save_su(self.segy2su_out))
        row_out.addWidget(btn_out)
        form.addRow("output_su", row_out)
        lay.addLayout(form)

        row_btn = QHBoxLayout()
        run = QPushButton("运行转换")
        run.clicked.connect(self._run_segy2su)
        load = QPushButton("加载产出 SU")
        load.clicked.connect(self._load_segy2su_out)
        row_btn.addWidget(run)
        row_btn.addWidget(load)
        row_btn.addStretch(1)
        lay.addLayout(row_btn)
        lay.addStretch(1)
        return w

    def _browse_save_su(self, edit: QLineEdit) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select output SU file",
            self.env.default_save_initial_dir(),
            "SU (*.su);;All (*.*)",
        )
        if path:
            target = self.env.rewrite_save_target(path)
            edit.setText(target)
            self.env.audit("browse_save", title="output_su", selected=path, rewritten=target)

    def _build_pipeline_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.pipe_help = _help_label(_HELP_PIPE["sac2segy"])
        lay.addWidget(self.pipe_help)
        form = QFormLayout()
        self.pipe_mode.addItem("SAC → SEGY", "sac2segy")
        self.pipe_mode.addItem("SAC → SU", "sac2su")
        self.pipe_mode.addItem("RAW → SEGY", "raw2segy")
        self.pipe_mode.addItem("RAW → SU", "raw2su")
        self.pipe_mode.addItem("OBEM → SEGY", "obem2segy")
        self.pipe_mode.addItem("OBEM → SU", "obem2su")
        form.addRow("流水线", self.pipe_mode)

        self.pipe_stack = QStackedWidget()
        # 0 sac
        p0 = QWidget()
        f0 = QFormLayout(p0)
        f0.addRow("sac", self._path_row(self.pipe_sac, "Select SAC", save=False))
        self.pipe_stack.addWidget(p0)
        # 1 raw
        p1 = QWidget()
        f1 = QFormLayout(p1)
        f1.addRow("raw", self._path_row(self.pipe_raw, "Select RAW", save=False))
        f1.addRow("sps", self.pipe_sps)
        f1.addRow("TC", self.pipe_tc)
        self.pipe_stack.addWidget(p1)
        # 2 obem
        p2 = QWidget()
        f2 = QFormLayout(p2)
        f2.addRow("obem_config", self._path_row(self.pipe_obem, "Select OBEM config", save=False))
        self.pipe_stack.addWidget(p2)

        form.addRow(self.pipe_stack)
        form.addRow("ukooa", self._path_row(self.pipe_ukooa, "Select UKOOA", save=False))
        form.addRow(
            "sac2y_config",
            self._path_row(self.pipe_sac2y_cfg, "Select sac2y config", save=False),
        )
        form.addRow("channels", self.pipe_channels)
        out_row = QHBoxLayout()
        out_row.addWidget(self.pipe_out, stretch=1)
        btn_out = QPushButton("浏览…")
        btn_out.clicked.connect(self._browse_pipe_out)
        out_row.addWidget(btn_out)
        form.addRow("output", out_row)
        lay.addLayout(form)
        run = QPushButton("运行一键转换")
        run.clicked.connect(self._run_pipeline)
        lay.addWidget(run)
        lay.addStretch(1)
        self._on_pipe_mode()
        return w

    def _path_row(
        self,
        edit: QLineEdit,
        title: str,
        *,
        save: bool = False,
        directory: bool = False,
    ) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(edit, stretch=1)
        btn = QPushButton("浏览…")

        def _browse() -> None:
            if directory:
                path = QFileDialog.getExistingDirectory(
                    self, title, self.env.default_save_initial_dir()
                )
                if path:
                    edit.setText(self.env.rewrite_save_target(path) if save else path)
                return
            if save:
                path, _ = QFileDialog.getSaveFileName(
                    self,
                    title,
                    self.env.default_save_initial_dir(),
                    "SU (*.su);;All (*.*)",
                )
                if path:
                    edit.setText(self.env.rewrite_save_target(path))
                return
            self._browse_open(edit, title)

        btn.clicked.connect(_browse)
        row.addWidget(btn)
        return row

    def _on_pipe_mode(self) -> None:
        mode = str(self.pipe_mode.currentData() or "sac2segy")
        if mode.startswith("sac"):
            idx = 0
        elif mode.startswith("raw"):
            idx = 1
        else:
            idx = 2
        self.pipe_stack.setCurrentIndex(idx)
        help_text = _HELP_PIPE.get(mode, _HELP_PIPE["sac2segy"])
        if hasattr(self, "pipe_help"):
            self.pipe_help.setText(help_text.strip())
        self.fields_changed.emit()

    def _browse_pipe_out(self) -> None:
        mode = str(self.pipe_mode.currentData() or "sac2segy")
        if mode == "sac2su":
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Select output SU",
                self.env.default_save_initial_dir(),
                "SU (*.su);;All (*.*)",
            )
            if path:
                self.pipe_out.setText(self.env.rewrite_save_target(path))
            return
        if mode == "sac2segy":
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Select output SEGY",
                self.env.default_save_initial_dir(),
                "SEGY (*.segy *.sgy);;All (*.*)",
            )
            if path:
                self.pipe_out.setText(self.env.rewrite_save_target(path))
            return
        path = QFileDialog.getExistingDirectory(
            self, "Select output directory", self.env.default_save_initial_dir()
        )
        if path:
            self.pipe_out.setText(path)

    def _run_pipeline(self) -> None:
        mode = str(self.pipe_mode.currentData() or "sac2segy")
        ukooa = self.pipe_ukooa.text().strip()
        cfg = self.pipe_sac2y_cfg.text().strip()
        out = self.pipe_out.text().strip()
        ch = self.pipe_channels.text().strip() or "all"
        self.runner.log.emit(f"[转换] 一键流水线 mode={mode}  channels={ch}")

        if mode == "sac2su":
            sac = self.pipe_sac.text().strip()
            if not all([sac, ukooa, cfg, out]):
                show_modeless_message("缺少参数", "sac / ukooa / sac2y_config / output.su 必填。", icon=QMessageBox.Icon.Warning)
                return
            self.runner.run_script("sac2su.py", [sac, ukooa, cfg, out])
            return
        if mode == "sac2segy":
            sac = self.pipe_sac.text().strip()
            if not all([sac, ukooa, cfg, out]):
                show_modeless_message("缺少参数", "sac / ukooa / sac2y_config / output.segy 必填。", icon=QMessageBox.Icon.Warning)
                return
            self.runner.run_script("sac2segy.py", [sac, ukooa, cfg, out])
            return
        if mode == "raw2su":
            raw = self.pipe_raw.text().strip()
            sps = self.pipe_sps.text().strip()
            tc = self.pipe_tc.text().strip()
            if not all([raw, sps, tc, ukooa, cfg, out]):
                show_modeless_message("缺少参数", "raw / sps / TC / ukooa / sac2y_config / out_dir 必填。"
                , icon=QMessageBox.Icon.Warning)
                return
            self.runner.run_script(
                "raw2su.py",
                [raw, sps, tc, ukooa, cfg, out, "--channels", ch],
            )
            return
        if mode == "raw2segy":
            raw = self.pipe_raw.text().strip()
            sps = self.pipe_sps.text().strip()
            tc = self.pipe_tc.text().strip()
            if not all([raw, sps, tc, ukooa, cfg, out]):
                show_modeless_message("缺少参数", "raw / sps / TC / ukooa / sac2y_config / out_dir 必填。"
                , icon=QMessageBox.Icon.Warning)
                return
            self.runner.run_script(
                "raw2segy.py",
                [raw, sps, tc, ukooa, cfg, out, "--channels", ch],
            )
            return
        if mode == "obem2segy":
            obem = self.pipe_obem.text().strip()
            if not all([obem, ukooa, cfg, out]):
                show_modeless_message("缺少参数", "obem_config / ukooa / sac2y_config / out_dir 必填。"
                , icon=QMessageBox.Icon.Warning)
                return
            self.runner.run_script(
                "obem2segy.py",
                [obem, ukooa, cfg, out, "--channels", ch],
            )
            return
        # obem2su
        obem = self.pipe_obem.text().strip()
        if not all([obem, ukooa, cfg, out]):
            show_modeless_message("缺少参数", "obem_config / ukooa / sac2y_config / out_dir 必填。"
            , icon=QMessageBox.Icon.Warning)
            return
        self.runner.run_script(
            "obem2su.py",
            [obem, ukooa, cfg, out, "--channels", ch],
        )

    def _on_field(self, name: str, value: str) -> None:
        self.env.note_field_change(name, value, on_changed=lambda: self.fields_changed.emit())

    def _browse_open(self, edit: QLineEdit, title: str, filt: str = "All (*.*)") -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, title, self.env.default_open_initial_dir(), filt
        )
        if path:
            backed = self.env.backup_input_file(path)
            edit.setText(backed)
            self.env.audit("browse_file", title=title, selected=path, backup=backed)

    def _browse_save(self, edit: QLineEdit, title: str) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            title,
            self.env.default_save_initial_dir(),
            "SEGY (*.segy *.sgy);;All (*.*)",
        )
        if path:
            target = self.env.rewrite_save_target(path)
            edit.setText(target)
            self.env.audit("browse_save", title=title, selected=path, rewritten=target)

    def _run_obem(self) -> None:
        self.runner.log.emit("[转换] 分步：OBEM → SAC")
        cfg = self.obem_config.text().strip()
        if not cfg:
            show_modeless_message("缺少参数", "config_file is required.", icon=QMessageBox.Icon.Warning)
            return
        self.env.audit("run_obem_clicked", config_file=cfg)
        self.runner.run_script("obem_tsm_to_sac_obspy.py", [cfg])

    def _run_sac2y(self) -> None:
        self.runner.log.emit("[转换] 分步：SAC → SEGY (sac2y)")

        args = [
            self.sac2y_sac.text().strip(),
            self.sac2y_ukooa.text().strip(),
            self.sac2y_output.text().strip(),
            self.sac2y_config.text().strip(),
        ]
        if not all(args):
            show_modeless_message(
                "缺少参数",
                "sac_file / ukooa_file / output_segy_file / config_file are required.",
                icon=QMessageBox.Icon.Warning,
            )
            return
        self._last_segy_out = args[2]
        # sac2y 原生已写约定（sx=炮,gx=OBS）；选旧对调时事后交换
        role = self.geom_role.currentData()
        self._pending_obs_remap = role in ("obs", "obs_legacy")
        self.env.audit("run_sac2y_clicked", args=args, role=role)
        self.runner.run_script("sac2y_v2_1_obspy.py", args)

    def _run_raw2sac(self) -> None:
        self.runner.log.emit("[转换] 分步：RAW → SAC")
        file_name = self.raw_file.text().strip()
        sps = self.raw_sps.text().strip()
        tc = self.raw_tc.text().strip()
        if not file_name or not sps or not tc:
            show_modeless_message("缺少参数", "fileName / sps / TC are required.", icon=QMessageBox.Icon.Warning)
            return
        self.env.audit("run_raw2sac_clicked", file_name=file_name, sps=sps, tc=tc)
        self.runner.run_script("raw2sac_v1_1_obspy.py", [file_name, sps, tc])

    def _load_output(self) -> None:
        path = self.sac2y_output.text().strip() or self._last_segy_out
        if not path:
            show_modeless_message("提示", "请先指定 output_segy_file。")
            return
        self.request_load_segy.emit(path)

    def _run_segy2su(self) -> None:
        src = self.segy2su_in.text().strip()
        dst = self.segy2su_out.text().strip()
        if not src or not dst:
            show_modeless_message("缺少参数", "input_segy / output_su are required.", icon=QMessageBox.Icon.Warning)
            return
        self.env.audit("run_segy2su_clicked", src=src, dst=dst)
        self.runner.status.emit("SEGY → SU …")
        self.runner.log.emit(f"[转换] 分步：SEGY → SU  {src} → {dst}")
        try:
            out = convert_segy_to_su(src, dst, endian="little")
            self.runner.log.emit(f"[转换] SEGY → SU 已写入 {out}")
            self.runner.status.emit("SEGY → SU done.")
            self.env.audit("run_segy2su_finished", path=str(out))
            show_modeless_message("SEGY → SU", f"已写入：\n{out}")
        except Exception as exc:
            self.runner.log.emit(f"[转换] SEGY → SU 失败: {exc}")
            self.runner.status.emit("SEGY → SU failed.")
            show_modeless_message("SEGY → SU", str(exc), icon=QMessageBox.Icon.Critical)

    def _load_segy2su_out(self) -> None:
        path = self.segy2su_out.text().strip()
        if not path:
            show_modeless_message("提示", "请先指定 output_su。")
            return
        self.request_load_segy.emit(path)

    def _on_job_finished(self, code: int, _output: str, command: list) -> None:
        if code != 0:
            return
        # sac2y 默认已是约定（sx=炮）；旧对调需交换 s*/g*
        if getattr(self, "_pending_obs_remap", False) and self._last_segy_out:
            self._pending_obs_remap = False
            try:
                ds = SegyDataset()
                ds.open(self._last_segy_out)
                n = ds.swap_source_group_slots()
                ds.recompute_offset()
                ds.save()
                self.runner.log.emit(
                    f"[obs_legacy] swapped s*/g* on {n} traces → {self._last_segy_out}"
                )
            except Exception as exc:
                self.runner.log.emit(f"[obs_legacy] remap failed: {exc}")

    def collect_state_fields(self) -> dict:
        return {
            "obem_config": self.obem_config.text(),
            "sac2y_sac": self.sac2y_sac.text(),
            "sac2y_ukooa": self.sac2y_ukooa.text(),
            "sac2y_output": self.sac2y_output.text(),
            "sac2y_config": self.sac2y_config.text(),
            "sac2y_role": str(self.geom_role.currentData() or "segy"),
            "raw_file": self.raw_file.text(),
            "raw_sps": self.raw_sps.text(),
            "raw_tc": self.raw_tc.text(),
            "segy2su_in": self.segy2su_in.text(),
            "segy2su_out": self.segy2su_out.text(),
            "pipe_mode": str(self.pipe_mode.currentData() or "sac2segy"),
            "pipe_sac": self.pipe_sac.text(),
            "pipe_raw": self.pipe_raw.text(),
            "pipe_sps": self.pipe_sps.text(),
            "pipe_tc": self.pipe_tc.text(),
            "pipe_obem": self.pipe_obem.text(),
            "pipe_ukooa": self.pipe_ukooa.text(),
            "pipe_sac2y_cfg": self.pipe_sac2y_cfg.text(),
            "pipe_out": self.pipe_out.text(),
            "pipe_channels": self.pipe_channels.text(),
        }

    def restore_state_fields(self, fields: dict) -> None:
        if not isinstance(fields, dict):
            return
        self.obem_config.setText(self.env.normalize_restored_path(str(fields.get("obem_config", ""))))
        self.sac2y_sac.setText(self.env.normalize_restored_path(str(fields.get("sac2y_sac", ""))))
        self.sac2y_ukooa.setText(self.env.normalize_restored_path(str(fields.get("sac2y_ukooa", ""))))
        self.sac2y_output.setText(self.env.normalize_restored_path(str(fields.get("sac2y_output", ""))))
        self.sac2y_config.setText(self.env.normalize_restored_path(str(fields.get("sac2y_config", ""))))
        self.raw_file.setText(self.env.normalize_restored_path(str(fields.get("raw_file", ""))))
        self.raw_sps.setText(str(fields.get("raw_sps", self.raw_sps.text())))
        self.raw_tc.setText(str(fields.get("raw_tc", self.raw_tc.text())))
        self.segy2su_in.setText(self.env.normalize_restored_path(str(fields.get("segy2su_in", ""))))
        self.segy2su_out.setText(self.env.normalize_restored_path(str(fields.get("segy2su_out", ""))))
        self.pipe_sac.setText(self.env.normalize_restored_path(str(fields.get("pipe_sac", ""))))
        self.pipe_raw.setText(self.env.normalize_restored_path(str(fields.get("pipe_raw", ""))))
        self.pipe_sps.setText(str(fields.get("pipe_sps", self.pipe_sps.text())))
        self.pipe_tc.setText(str(fields.get("pipe_tc", self.pipe_tc.text())))
        self.pipe_obem.setText(self.env.normalize_restored_path(str(fields.get("pipe_obem", ""))))
        self.pipe_ukooa.setText(self.env.normalize_restored_path(str(fields.get("pipe_ukooa", ""))))
        self.pipe_sac2y_cfg.setText(
            self.env.normalize_restored_path(str(fields.get("pipe_sac2y_cfg", "")))
        )
        self.pipe_out.setText(self.env.normalize_restored_path(str(fields.get("pipe_out", ""))))
        self.pipe_channels.setText(str(fields.get("pipe_channels", self.pipe_channels.text())))
        pm = str(fields.get("pipe_mode", "sac2segy"))
        pidx = self.pipe_mode.findData(pm)
        if pidx >= 0:
            self.pipe_mode.setCurrentIndex(pidx)
            self._on_pipe_mode()
        role = str(fields.get("sac2y_role", "segy"))
        if role == "literal_segy":
            role = "segy"
        idx = self.geom_role.findData(role)
        if idx >= 0:
            self.geom_role.setCurrentIndex(idx)

    def convert_tab_index(self) -> int:
        return int(self.tabs.currentIndex())

    def set_convert_tab_index(self, idx: int) -> None:
        self.tabs.setCurrentIndex(max(0, min(idx, self.tabs.count() - 1)))
