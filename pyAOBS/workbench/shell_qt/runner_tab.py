"""Run node tab (Qt)."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..plugins.tomo2d_templates import (
    TEMPLATE_CUSTOM,
    TEMPLATE_INVERSE_STANDARD,
    build_tomo2d_template_payload,
    list_tomo2d_templates,
)
from ..shell.logic.gui_form_commands import (
    apply_gui_quick_form_to_command,
    petrology_bridge_status,
)
from ..shell.logic.helpers import ordered_plugin_ids
from .styles import hint_label, status_panel


class RunnerTab(QWidget):
    """Plugin runner + TOMO2D template + GUI quick forms."""

    run_requested = Signal()
    apply_template = Signal()
    apply_gui_form = Signal()
    fill_example = Signal()
    save_preset = Signal()
    load_preset = Signal()
    preflight = Signal()
    cancel_run = Signal()
    launch_lip_bridge = Signal()
    refresh_petrology_bridge = Signal()
    fill_petrology_from_imodel = Signal()

    def __init__(
        self,
        plugin_ids: list[str],
        tool_registry,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._registry = tool_registry
        self._template_name_by_id = {k: v for k, v in list_tomo2d_templates()}

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        root = QVBoxLayout(inner)

        row0 = QHBoxLayout()
        row0.addWidget(QLabel("插件"))
        self._plugin = QComboBox()
        for pid in ordered_plugin_ids(plugin_ids):
            self._plugin.addItem(pid)
        self._plugin.currentIndexChanged.connect(self._on_plugin_changed)
        row0.addWidget(self._plugin)
        row0.addWidget(QLabel("Node ID"))
        self._node_id = QLineEdit("OBS_node")
        row0.addWidget(self._node_id, stretch=1)
        root.addLayout(row0)

        self._plugin_desc = hint_label("")
        root.addWidget(self._plugin_desc)

        self._tomo_group = QGroupBox("TOMO2D 模板表单（推荐）")
        tg = QVBoxLayout(self._tomo_group)
        tr = QHBoxLayout()
        tr.addWidget(QLabel("模板"))
        self._template = QComboBox()
        for k, _ in list_tomo2d_templates():
            self._template.addItem(k)
        self._template.setCurrentText(TEMPLATE_INVERSE_STANDARD)
        tr.addWidget(self._template)
        self._template_title = QLabel("")
        tr.addWidget(self._template_title, stretch=1)
        tg.addLayout(tr)

        base = QFormLayout()
        self._tpl_work_dir = QLineEdit("work")
        self._tpl_node = QLineEdit("OBS_node")
        self._tpl_verbose = QLineEdit("-1")
        self._tpl_mesh = QLineEdit("inputs/mesh.dat")
        self._tpl_data = QLineEdit("inputs/data.dat")
        base.addRow("work_dir:", self._tpl_work_dir)
        base.addRow("node_id:", self._tpl_node)
        base.addRow("verbose:", self._tpl_verbose)
        base.addRow("mesh_path:", self._tpl_mesh)
        base.addRow("data_path:", self._tpl_data)
        tg.addLayout(base)

        reg = QHBoxLayout()
        self._tpl_i = QLineEdit("5")
        self._tpl_sv = QLineEdit("100")
        self._tpl_sd = QLineEdit("10")
        self._tpl_dv = QLineEdit("1")
        self._tpl_dd = QLineEdit("20")
        for lbl, w in (
            ("I", self._tpl_i),
            ("SV", self._tpl_sv),
            ("SD", self._tpl_sd),
            ("DV", self._tpl_dv),
            ("DD", self._tpl_dd),
        ):
            reg.addWidget(QLabel(lbl))
            reg.addWidget(w)
        tg.addLayout(reg)
        self._tpl_extra = QLineEdit("")
        tg.addWidget(QLabel("extra_args"))
        tg.addWidget(self._tpl_extra)
        root.addWidget(self._tomo_group)

        # GUI quick form
        self._gui_group = QGroupBox("GUI 启动参数（插件专用）")
        gg = QVBoxLayout(self._gui_group)
        self._gui_note = hint_label("")
        gg.addWidget(self._gui_note)

        self._zplot_frame = QWidget()
        zf = QFormLayout(self._zplot_frame)
        self._zplot_itype = QComboBox()
        self._zplot_itype.addItems(["0", "1", "2", "3", "4", "-1", "-2", "-3", "-4"])
        self._zplot_irec = QLineEdit()
        self._zplot_data = QLineEdit()
        self._zplot_header = QLineEdit()
        self._zplot_record = QLineEdit()
        zf.addRow("itype:", self._zplot_itype)
        zf.addRow("irec:", self._zplot_irec)
        zf.addRow("data(.z):", self._zplot_data)
        zf.addRow("header:", self._zplot_header)
        zf.addRow("record:", self._zplot_record)
        gg.addWidget(self._zplot_frame)

        self._imodel_frame = QWidget()
        imf = QFormLayout(self._imodel_frame)
        self._imodel_node = QLineEdit("OBS_node")
        self._imodel_model = QLineEdit()
        self._imodel_aux = QLineEdit()
        imf.addRow("node_id:", self._imodel_node)
        imf.addRow("model_file:", self._imodel_model)
        imf.addRow("aux_file:", self._imodel_aux)
        imf.addRow("", hint_label("imodel（Qt）：在 GUI 内加载模型；此处登记路径并写入运行追踪。"))
        gg.addWidget(self._imodel_frame)

        self._iphase_frame = QWidget()
        ipf = QFormLayout(self._iphase_frame)
        self._iphase_node = QLineEdit("OBS_node")
        self._iphase_tx = QPlainTextEdit()
        self._iphase_tx.setMaximumHeight(80)
        self._iphase_rin = QLineEdit()
        self._iphase_txout = QLineEdit()
        ipf.addRow("node_id:", self._iphase_node)
        ipf.addRow("tx_files:", self._iphase_tx)
        ipf.addRow("rin:", self._iphase_rin)
        ipf.addRow("txout:", self._iphase_txout)
        gg.addWidget(self._iphase_frame)

        self._petrology_group = QGroupBox("LIP Petrology 桥接（imodel 观测 → LIP GUI）")
        self._petrology_group.setObjectName("WorkbenchBridgeGroup")
        pg = QVBoxLayout(self._petrology_group)
        pr = QHBoxLayout()
        self._petrology_state = QLineEdit()
        pr.addWidget(self._petrology_state, stretch=1)
        btn_browse = QPushButton("浏览…")
        btn_browse.clicked.connect(self._browse_petrology_state)
        pr.addWidget(btn_browse)
        btn_latest = QPushButton("最近 imodel 运行")
        btn_latest.clicked.connect(lambda: self.fill_petrology_from_imodel.emit())
        pr.addWidget(btn_latest)
        pg.addLayout(pr)
        self._petrology_status = status_panel("(未读取会话)")
        pg.addWidget(self._petrology_status)
        pbtn = QHBoxLayout()
        for txt, sig in (
            ("从会话读取观测", self.refresh_petrology_bridge),
            ("启动 LIP Petrology", self.launch_lip_bridge),
        ):
            b = QPushButton(txt)
            b.clicked.connect(sig.emit)
            pbtn.addWidget(b)
        pbtn.addStretch()
        pg.addLayout(pbtn)
        gg.addWidget(self._petrology_group)
        root.addWidget(self._gui_group)

        # Command form
        cmd_grp = QGroupBox("命令")
        cmd_grp.setObjectName("WorkbenchCommandGroup")
        cf = QFormLayout(cmd_grp)
        self._exec = QLineEdit("python")
        self._args = QPlainTextEdit()
        self._args.setMaximumHeight(100)
        self._cwd = QLineEdit()
        self._env = QPlainTextEdit(
            "TOMO2D_INV_OMP=1\nOMP_NUM_THREADS=4\nOMP_PLACES=threads\nOMP_PROC_BIND=spread\n"
        )
        self._env.setMaximumHeight(100)
        self._inputs = QPlainTextEdit()
        self._inputs.setMaximumHeight(100)
        cf.addRow("可执行程序:", self._exec)
        cf.addRow("参数字符串:", self._args)
        cf.addRow("工作目录(相对项目):", self._cwd)
        cf.addRow("环境变量:", self._env)
        cf.addRow("输入文件:", self._inputs)
        root.addWidget(cmd_grp)

        actions = QHBoxLayout()
        for txt, sig in (
            ("应用模板到命令", self.apply_template),
            ("应用GUI表单到命令", self.apply_gui_form),
            ("高级：运行前检查", self.preflight),
            ("填入插件启动示例", self.fill_example),
            ("保存节点预设", self.save_preset),
            ("加载节点预设", self.load_preset),
        ):
            b = QPushButton(txt)
            b.clicked.connect(sig.emit)
            actions.addWidget(b)
        actions.addStretch()
        root.addLayout(actions)

        run_row = QHBoxLayout()
        self._run_btn = QPushButton("运行节点")
        self._run_btn.setObjectName("WorkbenchPrimaryButton")
        self._run_btn.clicked.connect(self.run_requested.emit)
        self._cancel_btn = QPushButton("取消运行")
        self._cancel_btn.setObjectName("WorkbenchWarnButton")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self.cancel_run.emit)
        run_row.addWidget(self._run_btn)
        run_row.addWidget(self._cancel_btn)
        run_row.addStretch()
        root.addLayout(run_row)

        root.addStretch()
        scroll.setWidget(inner)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(scroll)

        self._template.currentIndexChanged.connect(self._update_template_title)
        self._on_plugin_changed()

    def _browse_petrology_state(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self, "选择 imodel gui_state JSON", "", "JSON (*.json);;All (*.*)"
        )
        if path:
            self._petrology_state.setText(path)
            self.refresh_petrology_bridge.emit()

    def plugin_id(self) -> str:
        return self._plugin.currentText().strip()

    def set_plugin_id(self, plugin_id: str) -> None:
        idx = self._plugin.findText(plugin_id)
        if idx >= 0:
            self._plugin.setCurrentIndex(idx)

    def set_run_buttons_enabled(self, *, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)

    def collect_run_payload(self) -> dict[str, str]:
        return {
            "node_id": self._node_id.text().strip() or "OBS_node",
            "executable": self._exec.text().strip(),
            "args": self._args.toPlainText().strip(),
            "cwd": self._cwd.text().strip(),
            "env_text": self._env.toPlainText(),
            "inputs_text": self._inputs.toPlainText(),
        }

    def set_run_payload(self, payload: dict[str, str]) -> None:
        self._node_id.setText(str(payload.get("node_id", "") or "OBS_node"))
        self._exec.setText(str(payload.get("executable", "") or ""))
        self._args.setPlainText(str(payload.get("args", "") or ""))
        self._cwd.setText(str(payload.get("cwd", "") or ""))
        self._env.setPlainText(str(payload.get("env_text", "") or ""))
        self._inputs.setPlainText(str(payload.get("inputs_text", "") or ""))

    def collect_gui_form_state(self) -> dict[str, str]:
        return {
            "zplot_itype": self._zplot_itype.currentText(),
            "zplot_irec": self._zplot_irec.text(),
            "zplot_data": self._zplot_data.text(),
            "zplot_header": self._zplot_header.text(),
            "zplot_record": self._zplot_record.text(),
            "imodel_node": self._imodel_node.text(),
            "imodel_model": self._imodel_model.text(),
            "imodel_aux": self._imodel_aux.text(),
            "iphase_node": self._iphase_node.text(),
            "iphase_tx_text": self._iphase_tx.toPlainText(),
            "iphase_rin": self._iphase_rin.text(),
            "iphase_txout": self._iphase_txout.text(),
            "petrology_state": self._petrology_state.text(),
        }

    def apply_gui_form_state(self, state: dict) -> None:
        self._zplot_itype.setCurrentText(str(state.get("zplot_itype", "0") or "0"))
        self._zplot_irec.setText(str(state.get("zplot_irec", "") or ""))
        self._zplot_data.setText(str(state.get("zplot_data", "") or ""))
        self._zplot_header.setText(str(state.get("zplot_header", "") or ""))
        self._zplot_record.setText(str(state.get("zplot_record", "") or ""))
        self._imodel_node.setText(str(state.get("imodel_node", "OBS_node") or "OBS_node"))
        self._imodel_model.setText(str(state.get("imodel_model", "") or ""))
        self._imodel_aux.setText(str(state.get("imodel_aux", "") or ""))
        self._iphase_node.setText(str(state.get("iphase_node", "OBS_node") or "OBS_node"))
        self._iphase_tx.setPlainText(str(state.get("iphase_tx_text", "") or ""))
        self._iphase_rin.setText(str(state.get("iphase_rin", "") or ""))
        self._iphase_txout.setText(str(state.get("iphase_txout", "") or ""))
        self._petrology_state.setText(str(state.get("petrology_state", "") or ""))

    def collect_template_fields(self) -> dict[str, str]:
        return {
            "work_dir": self._tpl_work_dir.text().strip(),
            "node_id": self._tpl_node.text().strip(),
            "verbose": self._tpl_verbose.text().strip(),
            "mesh_path": self._tpl_mesh.text().strip(),
            "data_path": self._tpl_data.text().strip(),
            "iterations": self._tpl_i.text().strip(),
            "sv": self._tpl_sv.text().strip(),
            "sd": self._tpl_sd.text().strip(),
            "dv": self._tpl_dv.text().strip(),
            "dd": self._tpl_dd.text().strip(),
            "extra_args": self._tpl_extra.text().strip(),
        }

    def apply_template_payload(self, payload: dict[str, str]) -> None:
        self._exec.setText(payload.get("executable", ""))
        self._cwd.setText(payload.get("cwd", ""))
        self._node_id.setText(payload.get("node_id", ""))
        self._args.setPlainText(payload.get("args", ""))
        self._env.setPlainText(payload.get("env_text", ""))
        self._inputs.setPlainText(payload.get("inputs_text", ""))

    def apply_gui_quick_form(self) -> str:
        pid = self.plugin_id()
        gui = self.collect_gui_form_state()
        nid, args, msg = apply_gui_quick_form_to_command(
            pid, node_id=self._node_id.text(), gui_form=gui
        )
        self._exec.setText("python")
        self._cwd.clear()
        self._env.clear()
        self._inputs.clear()
        self._node_id.setText(nid)
        self._args.setPlainText(args)
        return msg

    def update_petrology_bridge_label(self) -> None:
        text = self._petrology_state.text().strip()
        if not text:
            self._petrology_status.setText("(未指定 gui_state 文件)")
            return
        self._petrology_status.setText(petrology_bridge_status(text))

    def set_petrology_state_path(self, path: str) -> None:
        self._petrology_state.setText(path)
        self.update_petrology_bridge_label()

    def _update_template_title(self) -> None:
        tid = self._template.currentText()
        self._template_title.setText(self._template_name_by_id.get(tid, ""))

    def _on_plugin_changed(self) -> None:
        pid = self.plugin_id()
        try:
            plugin = self._registry.get(pid)
            desc = getattr(plugin, "description", "") if plugin else ""
        except Exception:
            desc = ""
        self._plugin_desc.setText(desc)

        is_tomo_shell = pid == "tomo2d.shell"
        is_gui = pid.endswith(".gui")
        self._tomo_group.setVisible(is_tomo_shell)
        self._gui_group.setVisible(is_gui or pid in ("imodel.gui", "petrology.lip.gui"))

        self._zplot_frame.setVisible(pid == "zplotpy.gui")
        self._imodel_frame.setVisible(pid == "imodel.gui")
        self._iphase_frame.setVisible(pid == "iphase.gui")
        self._petrology_group.setVisible(pid in ("imodel.gui", "petrology.lip.gui"))

        if is_gui:
            self._exec.setText("python")
            self._cwd.clear()
            self._cwd.setEnabled(False)
            self._inputs.setEnabled(False)
        else:
            self._cwd.setEnabled(True)
            self._inputs.setEnabled(True)

        note = ""
        if is_gui:
            note = "该 GUI 在启动后由其自身界面选择输入文件。"
        self._gui_note.setText(note)

        if is_gui and not self._node_id.text().strip():
            self._node_id.setText("OBS_node")

    def build_template_payload(self) -> dict[str, str]:
        return build_tomo2d_template_payload(
            template_id=self._template.currentText().strip(),
            fields=self.collect_template_fields(),
        )

    def ui_state_fragment(self) -> dict:
        return {
            "runner_plugin_id": self.plugin_id(),
            "runner_template_id": self._template.currentText(),
            "runner_run_payload": self.collect_run_payload(),
            "runner_gui_quick_form": self.collect_gui_form_state(),
        }

    def apply_ui_state_fragment(self, state: dict) -> None:
        pid = str(state.get("runner_plugin_id", "") or "").strip()
        if pid:
            self.set_plugin_id(pid)
        tid = str(state.get("runner_template_id", "") or "").strip()
        if tid:
            idx = self._template.findText(tid)
            if idx >= 0:
                self._template.setCurrentIndex(idx)
        payload = state.get("runner_run_payload")
        if isinstance(payload, dict):
            self.set_run_payload(payload)
        gui = state.get("runner_gui_quick_form")
        if isinstance(gui, dict):
            self.apply_gui_form_state(gui)
        self.update_petrology_bridge_label()
