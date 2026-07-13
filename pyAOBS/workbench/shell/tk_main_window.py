"""
Minimal Workbench main window (phase-4 shell skeleton).

This shell is intentionally lightweight: it wires project open/create,
run-history listing, and UI-state save/restore on top of the new core
services.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import csv
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fnmatch
import shlex
import shutil
import subprocess
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, simpledialog, ttk

from ..core.project_manager import ProjectContext, ProjectError, ProjectManager
from ..core.run_manager import RunManager
from ..petrology_launcher import (
    launch_lip_from_workbench_state,
    petrology_obs_path_from_state,
    petrology_transect_path_from_state,
)
from ..plugins.base import PluginValidationError
from ..plugins.tomo2d_templates import (
    TEMPLATE_CUSTOM,
    TEMPLATE_FORWARD_BASIC,
    TEMPLATE_INVERSE_STANDARD,
    build_tomo2d_template_payload,
    list_tomo2d_templates,
)
from ..registry.builtin_tools import create_builtin_registry
from ..core.state_store import StateStore

try:
    from ...modeling.tomo2d.help_docs import TomoHelp
except Exception:  # pragma: no cover - optional help integration
    TomoHelp = None


class _SimpleTooltip:
    """Minimal tooltip helper for Tk widgets."""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text.strip()
        self.tip: tk.Toplevel | None = None
        if not self.text:
            return
        self.widget.bind("<Enter>", self._show, add="+")
        self.widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _event: tk.Event) -> None:
        if self.tip is not None:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tip,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            padx=6,
            pady=4,
            wraplength=540,
        )
        label.pack()
        self.tip = tip

    def _hide(self, _event: tk.Event) -> None:
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


class WorkbenchMainWindowTk:
    """Tkinter Workbench shell (legacy fallback)."""

    def __init__(self, master: tk.Tk | None = None) -> None:
        self.root = master or tk.Tk()
        self.root.title("pyAOBS Workbench")
        self.root.geometry("1320x820")
        self._configure_ui_fonts()

        self.project_manager = ProjectManager()
        self.run_manager = RunManager()
        self.state_store = StateStore()
        self.tool_registry = create_builtin_registry()
        self.current_project: ProjectContext | None = None
        self._project_item_to_path: dict[str, Path] = {}
        self._run_id_to_dir: dict[str, Path] = {}
        self._run_records: list[dict[str, str | Path]] = []
        self._tooltips: list[_SimpleTooltip] = []
        self._help_map = self._build_template_help_map()
        self._batch_rerun_active = False
        self._batch_rerun_cancel_event: threading.Event | None = None
        self._batch_progress_total = 0
        self._batch_progress_done = 0
        self._batch_progress_success = 0
        self._batch_progress_failed = 0
        self._batch_progress_cancelled = 0
        self._batch_cancel_requested = False
        self._batch_queue_pending: list[str] = []
        self._batch_queue_running: list[str] = []
        self._batch_queue_recent_done: list[dict[str, str]] = []
        self._last_batch_results: list[dict[str, str]] = []
        self._last_batch_skipped: list[dict[str, str]] = []
        self._single_run_active = False
        self._single_run_cancel_event: threading.Event | None = None
        self._log_follow_after_id: str | None = None
        self._project_tree_refresh_token = 0
        self._run_history_refresh_token = 0
        self._project_tree_refresh_running = False
        self._run_history_refresh_running = False
        # Guardrails: avoid expensive full-tree recursion on large generated folders.
        self._project_tree_max_children_per_dir = 400
        self._project_tree_no_recurse_names = {
            "outputs",
            "logs",
            "__pycache__",
            "node_modules",
            ".git",
            ".venv",
            "venv",
        }
        self._run_history_scan_limit = 1200
        self._result_text_font = tkfont.nametofont("TkTextFont").copy()
        self._result_text_font.configure(size=max(13, int(self._result_text_font.cget("size")) + 2))
        self._result_tree_style = "RunResultLarge.Treeview"
        style = ttk.Style(self.root)
        style.configure(self._result_tree_style, font=("Segoe UI", 12), rowheight=24)
        style.configure(f"{self._result_tree_style}.Heading", font=("Segoe UI", 12, "bold"))

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def run(self) -> None:
        self.root.mainloop()

    def _configure_ui_fonts(self) -> None:
        """Increase default UI font size for readability."""
        base_size = 11
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            try:
                f = tkfont.nametofont(name)
                f.configure(size=base_size)
            except Exception:
                pass
        style = ttk.Style(self.root)
        style.configure(".", font=("Segoe UI", base_size))
        style.configure("Treeview.Heading", font=("Segoe UI", base_size, "bold"))

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)

        ttk.Button(top, text="新建项目", command=self._create_project).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(top, text="打开项目", command=self._open_project).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(top, text="保存界面状态", command=self._save_ui_state).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(top, text="恢复界面状态", command=self._restore_ui_state).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(top, text="刷新运行历史", command=self._refresh_all).pack(
            side=tk.LEFT, padx=(0, 6)
        )

        self.project_label_var = tk.StringVar(value="当前项目：未打开")
        ttk.Label(top, textvariable=self.project_label_var).pack(
            side=tk.LEFT, padx=(14, 0)
        )

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        left = ttk.Frame(body, padding=6)
        right = ttk.Frame(body, padding=6)
        body.add(left, weight=2)
        body.add(right, weight=3)

        ttk.Label(left, text="项目资源树").pack(anchor=tk.W, pady=(0, 4))
        self.project_tree = ttk.Treeview(left, show="tree", selectmode="browse")
        self.project_tree.pack(fill=tk.BOTH, expand=True)
        self._init_project_tree_icons()
        self.project_tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.project_tree.bind("<Double-1>", self._on_project_tree_double_click)

        self.right_notebook = ttk.Notebook(right)
        self.right_notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_history = ttk.Frame(self.right_notebook, padding=4)
        self.tab_runner = ttk.Frame(self.right_notebook, padding=4)
        self.right_notebook.add(self.tab_history, text="运行历史")
        self.right_notebook.add(self.tab_runner, text="运行节点")

        self._build_run_history_tab(self.tab_history)
        self._build_node_runner_tab(self.tab_runner)

        status_frame = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        status_frame.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor=tk.W)
        self.task_status_var = tk.StringVar(value="")
        ttk.Label(status_frame, textvariable=self.task_status_var).pack(anchor=tk.W)
        self._update_task_status_bar()

    def _build_run_history_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="节点工作区历史（runs/<node_id>/manifest.json）").pack(
            anchor=tk.W, pady=(0, 4)
        )
        filters = ttk.Frame(parent)
        filters.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(filters, text="Status").pack(side=tk.LEFT)
        self.run_filter_status_var = tk.StringVar(value="全部")
        self.run_filter_status_combo = ttk.Combobox(
            filters,
            textvariable=self.run_filter_status_var,
            values=["全部"],
            width=10,
            state="readonly",
        )
        self.run_filter_status_combo.pack(side=tk.LEFT, padx=(6, 10))
        self.run_filter_status_combo.bind("<<ComboboxSelected>>", self._on_run_filter_changed)

        ttk.Label(filters, text="Node").pack(side=tk.LEFT)
        self.run_filter_node_var = tk.StringVar(value="全部")
        self.run_filter_node_combo = ttk.Combobox(
            filters,
            textvariable=self.run_filter_node_var,
            values=["全部"],
            width=20,
            state="readonly",
        )
        self.run_filter_node_combo.pack(side=tk.LEFT, padx=(6, 10))
        self.run_filter_node_combo.bind("<<ComboboxSelected>>", self._on_run_filter_changed)

        ttk.Label(filters, text="关键词").pack(side=tk.LEFT)
        self.run_filter_keyword_var = tk.StringVar(value="")
        ent_kw = ttk.Entry(filters, textvariable=self.run_filter_keyword_var, width=28)
        ent_kw.pack(side=tk.LEFT, padx=(6, 10))
        ent_kw.bind("<KeyRelease>", self._on_run_filter_changed)

        self.run_filter_failed_only_var = tk.BooleanVar(value=False)
        chk_fail = ttk.Checkbutton(
            filters,
            text="仅失败",
            variable=self.run_filter_failed_only_var,
            command=self._on_run_filter_changed,
        )
        chk_fail.pack(side=tk.LEFT)

        ttk.Label(filters, text="批量并发").pack(side=tk.LEFT, padx=(10, 0))
        self.run_batch_concurrency_var = tk.StringVar(value="1")
        self.run_batch_concurrency_combo = ttk.Combobox(
            filters,
            textvariable=self.run_batch_concurrency_var,
            values=["1", "2", "4"],
            width=5,
            state="readonly",
        )
        self.run_batch_concurrency_combo.pack(side=tk.LEFT, padx=(6, 0))

        ttk.Button(filters, text="清空筛选", command=self._clear_run_filters).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        workspace_split = ttk.Panedwindow(parent, orient=tk.VERTICAL)
        workspace_split.pack(fill=tk.BOTH, expand=True, pady=(0, 2))
        self.workspace_history_pane = workspace_split

        workspace_frame = ttk.Frame(workspace_split)
        cols = ("run_id", "node_id", "status", "finished_at", "elapsed_s")
        self.run_tree = ttk.Treeview(
            workspace_frame,
            columns=cols,
            show="headings",
            selectmode="browse",
            height=12,
        )
        for c, title, w in (
            ("run_id", "Workspace ID", 290),
            ("node_id", "Node", 140),
            ("status", "Status", 90),
            ("finished_at", "Finished At", 180),
            ("elapsed_s", "Elapsed(s)", 90),
        ):
            self.run_tree.heading(c, text=title)
            self.run_tree.column(c, width=w, anchor=tk.W)
        self.run_tree.pack(fill=tk.BOTH, expand=True)
        self.run_tree.bind("<<TreeviewSelect>>", self._on_run_tree_select)

        history_container = ttk.Frame(workspace_split)
        actions = ttk.Frame(history_container)
        actions.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(actions, text="刷新详情", command=self._refresh_selected_run_detail).pack(
            side=tk.LEFT
        )
        ttk.Button(actions, text="打开工作区目录", command=self._open_selected_run_folder).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(actions, text="打开 GUI 审计日志", command=self._open_selected_audit_log).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(actions, text="删除选中工作区", command=self._delete_selected_run).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(actions, text="一键清理非成功", command=self._bulk_delete_non_success_runs).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(actions, text="回填到运行节点", command=self._prefill_selected_run_to_runner).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(actions, text="失败一键复跑", command=self._rerun_selected_failed).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        self.batch_rerun_button = ttk.Button(
            actions, text="批量复跑失败(筛选)", command=self._rerun_failed_filtered_batch
        )
        self.batch_rerun_button.pack(side=tk.LEFT, padx=(8, 0))
        self.batch_cancel_button = ttk.Button(
            actions, text="取消批量", command=self._cancel_batch_rerun, state=tk.DISABLED
        )
        self.batch_cancel_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(
            actions, text="导出最近批量结果 JSON", command=self._export_last_batch_results_json
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="刷新最近批量面板", command=self._refresh_last_batch_panel).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(actions, text="导出筛选结果 JSON", command=self._export_filtered_runs_json).pack(
            side=tk.LEFT, padx=(8, 0)
        )

        history_split = ttk.Panedwindow(history_container, orient=tk.VERTICAL)
        history_split.pack(fill=tk.BOTH, expand=True, pady=(0, 2))
        self.history_vertical_pane = history_split

        detail_frame = ttk.Frame(history_split)
        ttk.Label(detail_frame, text="选中运行详情").pack(anchor=tk.W, pady=(0, 4))
        self.run_detail_text = tk.Text(detail_frame, height=14, wrap=tk.WORD)
        self.run_detail_text.pack(fill=tk.BOTH, expand=True)
        self.run_detail_text.insert("1.0", "请选择一条运行记录查看详情。")
        self.run_detail_text.configure(font=self._result_text_font)
        self.run_detail_text.configure(state=tk.DISABLED)

        live_frame = ttk.Frame(history_split)
        live_title = ttk.Frame(live_frame)
        live_title.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(live_title, text="实时日志").pack(side=tk.LEFT)
        ttk.Label(live_title, text="源").pack(side=tk.LEFT, padx=(10, 0))
        self.live_log_source_var = tk.StringVar(value="stdout")
        self.live_log_source_combo = ttk.Combobox(
            live_title,
            textvariable=self.live_log_source_var,
            values=["stdout", "stderr"],
            width=8,
            state="readonly",
        )
        self.live_log_source_combo.pack(side=tk.LEFT, padx=(6, 8))
        self.live_log_source_combo.bind("<<ComboboxSelected>>", self._on_live_log_source_changed)
        self.live_log_follow_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            live_title,
            text="自动跟随",
            variable=self.live_log_follow_var,
            command=self._toggle_live_log_follow,
        ).pack(side=tk.LEFT)
        ttk.Label(live_title, text="关键词").pack(side=tk.LEFT, padx=(10, 0))
        self.live_log_keyword_var = tk.StringVar(value="")
        ent_log_kw = ttk.Entry(live_title, textvariable=self.live_log_keyword_var, width=18)
        ent_log_kw.pack(side=tk.LEFT, padx=(6, 6))
        ent_log_kw.bind("<KeyRelease>", self._on_live_log_keyword_changed)
        self.live_log_only_match_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            live_title,
            text="仅匹配行",
            variable=self.live_log_only_match_var,
            command=self._refresh_live_log_view,
        ).pack(side=tk.LEFT)
        ttk.Button(live_title, text="刷新日志", command=self._refresh_live_log_view).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        self.live_log_text = tk.Text(live_frame, height=14, wrap=tk.NONE)
        self.live_log_text.pack(fill=tk.BOTH, expand=True)
        self.live_log_text.insert("1.0", "请选择一条运行记录查看日志。")
        self.live_log_text.configure(font=self._result_text_font)
        self.live_log_text.configure(state=tk.DISABLED)

        batch_frame = ttk.Frame(history_split)
        batch_title = ttk.Frame(batch_frame)
        batch_title.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(batch_title, text="最近批量复跑结果").pack(side=tk.LEFT)
        ttk.Label(batch_title, text="状态筛选").pack(side=tk.LEFT, padx=(10, 0))
        self.last_batch_status_filter_var = tk.StringVar(value="全部")
        self.last_batch_status_filter_combo = ttk.Combobox(
            batch_title,
            textvariable=self.last_batch_status_filter_var,
            values=["全部", "success", "failed", "cancelled", "skipped"],
            width=10,
            state="readonly",
        )
        self.last_batch_status_filter_combo.pack(side=tk.LEFT, padx=(6, 0))
        self.last_batch_status_filter_combo.bind(
            "<<ComboboxSelected>>", self._on_last_batch_filter_changed
        )
        ttk.Button(batch_title, text="复制到剪贴板", command=self._copy_last_batch_panel).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        batch_content_split = ttk.Panedwindow(batch_frame, orient=tk.VERTICAL)
        batch_content_split.pack(fill=tk.BOTH, expand=True)
        self.batch_content_pane = batch_content_split

        batch_result_frame = ttk.Frame(batch_content_split)
        self.last_batch_text = tk.Text(batch_result_frame, height=12, wrap=tk.WORD)
        self.last_batch_text.pack(fill=tk.BOTH, expand=True)
        self.last_batch_text.insert("1.0", "暂无批量复跑结果。")
        self.last_batch_text.configure(font=self._result_text_font)
        self.last_batch_text.configure(state=tk.DISABLED)

        queue_wrap_frame = ttk.Frame(batch_content_split)
        queue_row = ttk.Panedwindow(queue_wrap_frame, orient=tk.HORIZONTAL)
        queue_row.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.batch_queue_pane = queue_row

        queue_panel = ttk.LabelFrame(queue_row, text="批量任务队列")
        queue_title = ttk.Frame(queue_panel)
        queue_title.pack(fill=tk.X, pady=(2, 4))
        ttk.Button(queue_title, text="刷新队列", command=self._refresh_batch_queue_panel).pack(
            side=tk.LEFT
        )
        self.batch_queue_text = tk.Text(queue_panel, height=10, wrap=tk.WORD)
        self.batch_queue_text.pack(fill=tk.BOTH, expand=False)
        self.batch_queue_text.insert("1.0", "当前无批量任务队列。")
        self.batch_queue_text.configure(font=self._result_text_font)
        self.batch_queue_text.configure(state=tk.DISABLED)

        recent_frame = ttk.LabelFrame(queue_row, text="最近完成项（双击定位运行记录）")
        self.batch_queue_recent_tree = ttk.Treeview(
            recent_frame,
            columns=("source", "status", "new_run"),
            show="headings",
            height=8,
            selectmode="browse",
            style=self._result_tree_style,
        )
        self.batch_queue_recent_tree.heading("source", text="源工作区")
        self.batch_queue_recent_tree.heading("status", text="Status")
        self.batch_queue_recent_tree.heading("new_run", text="新工作区")
        self.batch_queue_recent_tree.column("source", width=240, anchor=tk.W)
        self.batch_queue_recent_tree.column("status", width=100, anchor=tk.W)
        self.batch_queue_recent_tree.column("new_run", width=320, anchor=tk.W)
        self.batch_queue_recent_tree.pack(fill=tk.BOTH, expand=False)
        self.batch_queue_recent_tree.bind(
            "<Double-1>", self._on_batch_queue_recent_open_run
        )
        self.batch_queue_recent_tree.bind(
            "<Double-Button-1>", self._on_batch_queue_recent_open_run
        )
        self.batch_queue_recent_tree.bind(
            "<Button-3>", self._on_batch_queue_recent_context_menu
        )
        self.batch_queue_recent_tree.bind(
            "<ButtonRelease-3>", self._on_batch_queue_recent_context_menu
        )
        # Touchpad / macOS compatibility.
        self.batch_queue_recent_tree.bind(
            "<Button-2>", self._on_batch_queue_recent_context_menu
        )
        self.batch_queue_recent_tree.bind(
            "<Control-Button-1>", self._on_batch_queue_recent_context_menu
        )
        self.batch_queue_recent_menu = tk.Menu(recent_frame, tearoff=0)
        self.batch_queue_recent_menu.add_command(
            label="定位到运行详情",
            command=self._on_batch_queue_recent_open_run,
        )
        self.batch_queue_recent_menu.add_command(
            label="打开工作区目录",
            command=self._open_batch_recent_run_folder,
        )
        self.batch_queue_recent_menu.add_command(
            label="复制新工作区ID",
            command=self._copy_batch_recent_new_run_id,
        )
        self.batch_queue_recent_menu.add_command(
            label="打开实时日志",
            command=self._open_batch_recent_live_log,
        )
        batch_content_split.add(batch_result_frame, weight=1)
        batch_content_split.add(queue_wrap_frame, weight=1)
        queue_row.add(queue_panel, weight=1)
        queue_row.add(recent_frame, weight=1)
        history_split.add(detail_frame, weight=1)
        history_split.add(live_frame, weight=1)
        history_split.add(batch_frame, weight=1)
        workspace_split.add(workspace_frame, weight=1)
        workspace_split.add(history_container, weight=2)

    def _build_node_runner_tab(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(top, text="插件").grid(row=0, column=0, sticky=tk.W)
        plugin_ids = self._ordered_plugin_ids(self.tool_registry.ids())
        self.plugin_var = tk.StringVar(value=plugin_ids[0] if plugin_ids else "")
        self.plugin_combo = ttk.Combobox(
            top,
            textvariable=self.plugin_var,
            values=plugin_ids,
            width=26,
            state="readonly",
        )
        self.plugin_combo.grid(row=0, column=1, sticky=tk.W, padx=(6, 10))
        self.plugin_combo.bind("<<ComboboxSelected>>", self._on_plugin_changed)

        ttk.Label(top, text="Node ID").grid(row=0, column=2, sticky=tk.W)
        self.node_id_var = tk.StringVar(value="OBS_node")
        ttk.Entry(top, textvariable=self.node_id_var, width=24).grid(
            row=0, column=3, sticky=tk.W, padx=(6, 0)
        )

        self.plugin_desc_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.plugin_desc_var).pack(anchor=tk.W, pady=(0, 6))

        self.tomo_template_frame = ttk.LabelFrame(parent, text="TOMO2D 模板表单（推荐）")
        self.tomo_template_frame.pack(fill=tk.X, pady=(0, 6))
        self._build_template_form(self.tomo_template_frame)

        self.gui_quick_form_frame = ttk.LabelFrame(parent, text="GUI 启动参数（插件专用）")
        self._build_gui_quick_form(self.gui_quick_form_frame)

        form = ttk.Frame(parent)
        self.node_form_frame = form
        form.pack(fill=tk.BOTH, expand=True)

        ttk.Label(form, text="可执行程序").grid(row=0, column=0, sticky=tk.NW)
        self.exec_var = tk.StringVar(value="tt_inverse")
        ttk.Entry(form, textvariable=self.exec_var, width=70).grid(
            row=0, column=1, sticky=tk.EW, padx=(6, 0), pady=(0, 6)
        )

        ttk.Label(form, text="参数字符串").grid(row=1, column=0, sticky=tk.NW)
        self.args_text = tk.Text(form, height=5, wrap=tk.WORD)
        self.args_text.grid(row=1, column=1, sticky=tk.EW, padx=(6, 0), pady=(0, 6))

        ttk.Label(form, text="工作目录(相对项目)").grid(row=2, column=0, sticky=tk.NW)
        self.cwd_var = tk.StringVar(value="")
        self.cwd_entry = ttk.Entry(form, textvariable=self.cwd_var, width=70)
        self.cwd_entry.grid(
            row=2, column=1, sticky=tk.EW, padx=(6, 0), pady=(0, 6)
        )
        self.cwd_browse_button = ttk.Button(
            form, text="浏览...", command=self._browse_node_cwd
        )
        self.cwd_browse_button.grid(row=2, column=2, sticky=tk.W, padx=(6, 0), pady=(0, 6))

        split = ttk.Frame(form)
        split.grid(row=3, column=1, sticky=tk.NSEW, padx=(6, 0), pady=(0, 6))
        split.columnconfigure(0, weight=1)
        split.columnconfigure(1, weight=1)

        env_frame = ttk.LabelFrame(split, text="环境变量（每行 KEY=VALUE）")
        env_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 4))
        self.env_text = tk.Text(env_frame, height=6, wrap=tk.NONE)
        self.env_text.pack(fill=tk.BOTH, expand=True)
        self.env_text.insert(
            "1.0",
            "TOMO2D_INV_OMP=1\nOMP_NUM_THREADS=4\nOMP_PLACES=threads\nOMP_PROC_BIND=spread\n",
        )

        inputs_frame = ttk.LabelFrame(split, text="输入文件（每行一个路径）")
        inputs_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=(4, 0))
        self.inputs_text = tk.Text(inputs_frame, height=6, wrap=tk.NONE)
        self.inputs_text.pack(fill=tk.BOTH, expand=True)

        quick = ttk.Frame(form)
        quick.grid(row=4, column=1, sticky=tk.EW, padx=(6, 0), pady=(0, 6))
        ttk.Label(quick, text="快捷目录").pack(side=tk.LEFT)
        self.cwd_quick_buttons: list[ttk.Button] = []
        btn = ttk.Button(
            quick, text="raw", command=lambda: self._set_cwd_to_project_subdir("datasets/raw")
        )
        btn.pack(side=tk.LEFT, padx=(6, 0))
        self.cwd_quick_buttons.append(btn)
        btn = ttk.Button(
            quick,
            text="processed",
            command=lambda: self._set_cwd_to_project_subdir("datasets/processed"),
        )
        btn.pack(side=tk.LEFT, padx=(6, 0))
        self.cwd_quick_buttons.append(btn)
        btn = ttk.Button(
            quick, text="picks", command=lambda: self._set_cwd_to_project_subdir("picks")
        )
        btn.pack(side=tk.LEFT, padx=(6, 0))
        self.cwd_quick_buttons.append(btn)
        btn = ttk.Button(
            quick, text="models", command=lambda: self._set_cwd_to_project_subdir("models")
        )
        btn.pack(side=tk.LEFT, padx=(6, 0))
        self.cwd_quick_buttons.append(btn)
        btn = ttk.Button(
            quick,
            text="interpretation",
            command=lambda: self._set_cwd_to_project_subdir("interpretation"),
        )
        btn.pack(side=tk.LEFT, padx=(6, 0))
        self.cwd_quick_buttons.append(btn)
        btn = ttk.Button(
            quick, text="CWD=项目树选中目录", command=self._set_cwd_from_tree_selection
        )
        btn.pack(side=tk.LEFT, padx=(10, 0))
        self.cwd_quick_buttons.append(btn)
        self.inputs_quick_add_button = ttk.Button(
            quick, text="选中项加入 inputs", command=self._append_tree_selection_to_inputs
        )
        self.inputs_quick_add_button.pack(side=tk.LEFT, padx=(6, 0))

        form.columnconfigure(1, weight=1)
        form.rowconfigure(3, weight=1)

        actions = ttk.Frame(parent)
        actions.pack(fill=tk.X)
        ttk.Button(actions, text="应用模板到命令", command=self._apply_template_to_command).pack(
            side=tk.LEFT
        )
        ttk.Button(actions, text="应用GUI表单到命令", command=self._apply_gui_quick_form_to_command).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(actions, text="高级：运行前检查与预览", command=self._preflight_preview).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        self.run_node_button = ttk.Button(actions, text="运行节点", command=self._run_selected_node)
        self.run_node_button.pack(side=tk.LEFT, padx=(8, 0))
        self.cancel_run_button = ttk.Button(
            actions, text="取消运行", command=self._cancel_selected_node_run, state=tk.DISABLED
        )
        self.cancel_run_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="填入插件启动示例", command=self._fill_plugin_example).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(actions, text="保存节点预设", command=self._save_node_preset).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(actions, text="加载节点预设", command=self._load_node_preset).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(actions, text="管理节点预设", command=self._manage_node_presets).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        self._on_plugin_changed()

    @staticmethod
    def _ordered_plugin_ids(plugin_ids: list[str]) -> list[str]:
        preferred = [
            "data.gui",
            "zplotpy.gui",
            "tomo2d.gui",
            "tomo2d.shell",
            "imodel.gui",
            "iphase.gui",
            "petrology.lip.gui",
        ]
        seen: set[str] = set()
        ordered: list[str] = []
        for pid in preferred:
            if pid in plugin_ids and pid not in seen:
                ordered.append(pid)
                seen.add(pid)
        for pid in plugin_ids:
            if pid not in seen:
                ordered.append(pid)
                seen.add(pid)
        return ordered

    def _build_template_form(self, parent: ttk.LabelFrame) -> None:
        templates = list_tomo2d_templates()
        self._template_name_by_id = {k: v for k, v in templates}

        row0 = ttk.Frame(parent)
        row0.pack(fill=tk.X, pady=(4, 4), padx=6)
        ttk.Label(row0, text="模板").grid(row=0, column=0, sticky=tk.W)
        self.template_var = tk.StringVar(value=TEMPLATE_INVERSE_STANDARD)
        self.template_combo = ttk.Combobox(
            row0,
            textvariable=self.template_var,
            values=[k for k, _ in templates],
            width=24,
            state="readonly",
        )
        self.template_combo.grid(row=0, column=1, sticky=tk.W, padx=(6, 12))
        self.template_combo.bind("<<ComboboxSelected>>", self._on_template_changed)
        self.template_title_var = tk.StringVar(value="")
        ttk.Label(row0, textvariable=self.template_title_var).grid(
            row=0, column=2, sticky=tk.W
        )

        base_group = self._create_collapsible_group(parent, "基础输入")
        reg_group = self._create_collapsible_group(parent, "反演约束参数")
        run_group = self._create_collapsible_group(parent, "运行与并行")

        row1 = ttk.Frame(base_group)
        row1.pack(fill=tk.X, pady=(0, 4))
        lbl_work = ttk.Label(row1, text="work_dir")
        lbl_work.grid(row=0, column=0, sticky=tk.W)
        self.tpl_work_dir_var = tk.StringVar(value="work")
        ent_work = ttk.Entry(row1, textvariable=self.tpl_work_dir_var, width=18)
        ent_work.grid(
            row=0, column=1, sticky=tk.W, padx=(6, 10)
        )
        ttk.Button(
            row1, text="浏览...", command=self._browse_template_work_dir
        ).grid(row=0, column=2, sticky=tk.W, padx=(0, 10))
        lbl_node = ttk.Label(row1, text="node_id")
        lbl_node.grid(row=0, column=3, sticky=tk.W)
        self.tpl_node_id_var = tk.StringVar(value="OBS_node")
        ent_node = ttk.Entry(row1, textvariable=self.tpl_node_id_var, width=22)
        ent_node.grid(
            row=0, column=4, sticky=tk.W, padx=(6, 10)
        )
        lbl_verbose = ttk.Label(row1, text="verbose")
        lbl_verbose.grid(row=0, column=5, sticky=tk.W)
        self.tpl_verbose_var = tk.StringVar(value="-1")
        ent_verbose = ttk.Entry(row1, textvariable=self.tpl_verbose_var, width=8)
        ent_verbose.grid(
            row=0, column=6, sticky=tk.W, padx=(6, 0)
        )

        row2 = ttk.Frame(base_group)
        row2.pack(fill=tk.X, pady=(0, 4))
        lbl_mesh = ttk.Label(row2, text="mesh_path")
        lbl_mesh.grid(row=0, column=0, sticky=tk.W)
        self.tpl_mesh_var = tk.StringVar(value="inputs/mesh.dat")
        ent_mesh = ttk.Entry(row2, textvariable=self.tpl_mesh_var, width=48)
        ent_mesh.grid(
            row=0, column=1, sticky=tk.EW, padx=(6, 10)
        )
        ttk.Button(row2, text="项目树选取", command=self._pick_mesh_from_tree).grid(
            row=0, column=2, sticky=tk.W, padx=(0, 6)
        )
        ttk.Button(row2, text="浏览...", command=self._browse_mesh_file).grid(
            row=0, column=3, sticky=tk.W, padx=(0, 10)
        )
        lbl_data = ttk.Label(row2, text="data_path")
        lbl_data.grid(row=0, column=4, sticky=tk.W)
        self.tpl_data_var = tk.StringVar(value="inputs/data.dat")
        ent_data = ttk.Entry(row2, textvariable=self.tpl_data_var, width=48)
        ent_data.grid(
            row=0, column=5, sticky=tk.EW, padx=(6, 10)
        )
        ttk.Button(row2, text="项目树选取", command=self._pick_data_from_tree).grid(
            row=0, column=6, sticky=tk.W, padx=(0, 6)
        )
        ttk.Button(row2, text="浏览...", command=self._browse_data_file).grid(
            row=0, column=7, sticky=tk.W
        )
        row2.columnconfigure(1, weight=1)
        row2.columnconfigure(5, weight=1)

        row3 = ttk.Frame(reg_group)
        row3.pack(fill=tk.X, pady=(0, 4))
        lbl_i = ttk.Label(row3, text="I")
        lbl_i.grid(row=0, column=0, sticky=tk.W)
        self.tpl_i_var = tk.StringVar(value="5")
        ent_i = ttk.Entry(row3, textvariable=self.tpl_i_var, width=8)
        ent_i.grid(
            row=0, column=1, sticky=tk.W, padx=(6, 10)
        )
        lbl_sv = ttk.Label(row3, text="SV")
        lbl_sv.grid(row=0, column=2, sticky=tk.W)
        self.tpl_sv_var = tk.StringVar(value="100")
        ent_sv = ttk.Entry(row3, textvariable=self.tpl_sv_var, width=8)
        ent_sv.grid(
            row=0, column=3, sticky=tk.W, padx=(6, 10)
        )
        lbl_sd = ttk.Label(row3, text="SD")
        lbl_sd.grid(row=0, column=4, sticky=tk.W)
        self.tpl_sd_var = tk.StringVar(value="10")
        ent_sd = ttk.Entry(row3, textvariable=self.tpl_sd_var, width=8)
        ent_sd.grid(
            row=0, column=5, sticky=tk.W, padx=(6, 10)
        )
        lbl_dv = ttk.Label(row3, text="DV")
        lbl_dv.grid(row=0, column=6, sticky=tk.W)
        self.tpl_dv_var = tk.StringVar(value="1")
        ent_dv = ttk.Entry(row3, textvariable=self.tpl_dv_var, width=8)
        ent_dv.grid(
            row=0, column=7, sticky=tk.W, padx=(6, 10)
        )
        lbl_dd = ttk.Label(row3, text="DD")
        lbl_dd.grid(row=0, column=8, sticky=tk.W)
        self.tpl_dd_var = tk.StringVar(value="20")
        ent_dd = ttk.Entry(row3, textvariable=self.tpl_dd_var, width=8)
        ent_dd.grid(
            row=0, column=9, sticky=tk.W, padx=(6, 0)
        )

        row4 = ttk.Frame(reg_group)
        row4.pack(fill=tk.X, pady=(0, 4))
        lbl_extra = ttk.Label(row4, text="额外参数")
        lbl_extra.grid(row=0, column=0, sticky=tk.W)
        self.tpl_extra_var = tk.StringVar(value="")
        ent_extra = ttk.Entry(row4, textvariable=self.tpl_extra_var)
        ent_extra.grid(
            row=0, column=1, sticky=tk.EW, padx=(6, 0)
        )
        row4.columnconfigure(1, weight=1)

        row5 = ttk.Frame(run_group)
        row5.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(
            row5,
            text="并行建议：OMP_NUM_THREADS=源数附近；OMP_PLACES=threads；OMP_PROC_BIND=spread",
        ).pack(anchor=tk.W)

        self._attach_field_tooltips(
            {
                "work_dir": [lbl_work, ent_work],
                "node_id": [lbl_node, ent_node],
                "verbose": [lbl_verbose, ent_verbose],
                "mesh_path": [lbl_mesh, ent_mesh],
                "data_path": [lbl_data, ent_data],
                "iterations": [lbl_i, ent_i],
                "sv": [lbl_sv, ent_sv],
                "sd": [lbl_sd, ent_sd],
                "dv": [lbl_dv, ent_dv],
                "dd": [lbl_dd, ent_dd],
                "extra_args": [lbl_extra, ent_extra],
            }
        )
        self._on_template_changed()

    def _build_gui_quick_form(self, parent: ttk.LabelFrame) -> None:
        row0 = ttk.Frame(parent)
        row0.pack(fill=tk.X, pady=(4, 4), padx=6)
        ttk.Label(row0, text="插件").grid(row=0, column=0, sticky=tk.W)
        self.gui_form_plugin_var = tk.StringVar(value="")
        ttk.Label(row0, textvariable=self.gui_form_plugin_var).grid(
            row=0, column=1, sticky=tk.W, padx=(6, 10)
        )
        ttk.Label(
            row0, text="说明：填写后点击“应用GUI表单到命令”自动生成 args/inputs。"
        ).grid(row=0, column=2, sticky=tk.W)
        self.gui_form_note_var = tk.StringVar(value="")
        ttk.Label(parent, textvariable=self.gui_form_note_var).pack(
            anchor=tk.W, padx=6, pady=(0, 4)
        )

        self.gui_zplot_frame = ttk.Frame(parent)
        self.gui_imodel_frame = ttk.Frame(parent)
        self.gui_iphase_frame = ttk.Frame(parent)
        self.gui_petrology_bridge_frame = ttk.LabelFrame(
            parent, text="LIP Petrology 桥接（imodel 观测 → LIP GUI）"
        )

        # zplotpy form
        z1 = ttk.Frame(self.gui_zplot_frame)
        z1.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(z1, text="itype").grid(row=0, column=0, sticky=tk.W)
        self.gui_zplot_itype_var = tk.StringVar(value="0")
        self.gui_zplot_itype_combo = ttk.Combobox(
            z1,
            textvariable=self.gui_zplot_itype_var,
            values=["0", "1", "2", "3", "4", "-1", "-2", "-3", "-4"],
            width=8,
            state="readonly",
        )
        self.gui_zplot_itype_combo.grid(row=0, column=1, sticky=tk.W, padx=(6, 10))
        ttk.Label(z1, text="irec").grid(row=0, column=2, sticky=tk.W)
        self.gui_zplot_irec_var = tk.StringVar(value="")
        ttk.Entry(z1, textvariable=self.gui_zplot_irec_var, width=10).grid(
            row=0, column=3, sticky=tk.W, padx=(6, 0)
        )

        z2 = ttk.Frame(self.gui_zplot_frame)
        z2.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(z2, text="data(.z)").grid(row=0, column=0, sticky=tk.W)
        self.gui_zplot_data_var = tk.StringVar(value="")
        ttk.Entry(z2, textvariable=self.gui_zplot_data_var).grid(
            row=0, column=1, sticky=tk.EW, padx=(6, 6)
        )
        ttk.Button(
            z2,
            text="浏览...",
            command=lambda: self._browse_gui_file(self.gui_zplot_data_var, "选择 zplot data 文件"),
        ).grid(row=0, column=2, sticky=tk.W)
        z2.columnconfigure(1, weight=1)

        z3 = ttk.Frame(self.gui_zplot_frame)
        z3.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(z3, text="header(.hdr)").grid(row=0, column=0, sticky=tk.W)
        self.gui_zplot_header_var = tk.StringVar(value="")
        ttk.Entry(z3, textvariable=self.gui_zplot_header_var).grid(
            row=0, column=1, sticky=tk.EW, padx=(6, 6)
        )
        ttk.Button(
            z3,
            text="浏览...",
            command=lambda: self._browse_gui_file(self.gui_zplot_header_var, "选择 zplot header 文件"),
        ).grid(row=0, column=2, sticky=tk.W)
        z3.columnconfigure(1, weight=1)

        z4 = ttk.Frame(self.gui_zplot_frame)
        z4.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(z4, text="record(.rsp)").grid(row=0, column=0, sticky=tk.W)
        self.gui_zplot_record_var = tk.StringVar(value="")
        ttk.Entry(z4, textvariable=self.gui_zplot_record_var).grid(
            row=0, column=1, sticky=tk.EW, padx=(6, 6)
        )
        ttk.Button(
            z4,
            text="浏览...",
            command=lambda: self._browse_gui_file(self.gui_zplot_record_var, "选择 zplot record 文件"),
        ).grid(row=0, column=2, sticky=tk.W)
        z4.columnconfigure(1, weight=1)

        # imodel form
        i1 = ttk.Frame(self.gui_imodel_frame)
        i1.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(
            i1,
            text="imodel（Qt）：在 GUI 内加载模型即可；此处用于登记输入路径并写入运行追踪。Tk 旧版：python -m pyAOBS.visualization.imodel_gui",
        ).pack(anchor=tk.W)
        iwd = ttk.Frame(self.gui_imodel_frame)
        iwd.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(iwd, text="work_dir").grid(row=0, column=0, sticky=tk.W)
        self.gui_imodel_workdir_var = tk.StringVar(value="")
        self.gui_imodel_workdir_entry = ttk.Entry(
            iwd, textvariable=self.gui_imodel_workdir_var, width=24, state=tk.DISABLED
        )
        self.gui_imodel_workdir_entry.grid(
            row=0, column=1, sticky=tk.W, padx=(6, 10)
        )
        self.gui_imodel_workdir_browse_button = ttk.Button(
            iwd, text="浏览...", command=self._browse_gui_imodel_work_dir
        )
        self.gui_imodel_workdir_browse_button.grid(row=0, column=2, sticky=tk.W, padx=(0, 10))
        self.gui_imodel_workdir_browse_button.configure(state=tk.DISABLED)
        ttk.Label(iwd, text="node_id").grid(row=0, column=3, sticky=tk.W)
        self.gui_imodel_node_var = tk.StringVar(value="OBS_node")
        ttk.Entry(iwd, textvariable=self.gui_imodel_node_var, width=24).grid(
            row=0, column=4, sticky=tk.W, padx=(6, 0)
        )
        ires = ttk.Frame(self.gui_imodel_frame)
        ires.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(ires, text="model_file").grid(row=0, column=0, sticky=tk.W)
        self.gui_imodel_model_var = tk.StringVar(value="")
        ttk.Entry(ires, textvariable=self.gui_imodel_model_var).grid(
            row=0, column=1, sticky=tk.EW, padx=(6, 6)
        )
        ttk.Button(
            ires,
            text="浏览...",
            command=lambda: self._browse_gui_file(self.gui_imodel_model_var, "选择 imodel 模型文件"),
        ).grid(row=0, column=2, sticky=tk.W)
        ires.columnconfigure(1, weight=1)
        ires2 = ttk.Frame(self.gui_imodel_frame)
        ires2.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(ires2, text="aux_file").grid(row=0, column=0, sticky=tk.W)
        self.gui_imodel_aux_var = tk.StringVar(value="")
        ttk.Entry(ires2, textvariable=self.gui_imodel_aux_var).grid(
            row=0, column=1, sticky=tk.EW, padx=(6, 6)
        )
        ttk.Button(
            ires2,
            text="浏览...",
            command=lambda: self._browse_gui_file(self.gui_imodel_aux_var, "选择 imodel 辅助文件"),
        ).grid(row=0, column=2, sticky=tk.W)
        ires2.columnconfigure(1, weight=1)
        i2 = ttk.Frame(self.gui_imodel_frame)
        i2.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(i2, text="extra_args").grid(row=0, column=0, sticky=tk.W)
        self.gui_imodel_extra_var = tk.StringVar(value="")
        ttk.Entry(i2, textvariable=self.gui_imodel_extra_var).grid(
            row=0, column=1, sticky=tk.EW, padx=(6, 0)
        )
        i2.columnconfigure(1, weight=1)

        # iphase form
        p1 = ttk.Frame(self.gui_iphase_frame)
        p1.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(
            p1,
            text="iphase 当前主要通过 GUI 内部打开数据；此处用于登记 tx/rin/txout 资源并统一追踪。",
        ).pack(anchor=tk.W)
        pwd = ttk.Frame(self.gui_iphase_frame)
        pwd.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(pwd, text="work_dir").grid(row=0, column=0, sticky=tk.W)
        self.gui_iphase_workdir_var = tk.StringVar(value="")
        self.gui_iphase_workdir_entry = ttk.Entry(
            pwd, textvariable=self.gui_iphase_workdir_var, width=24, state=tk.DISABLED
        )
        self.gui_iphase_workdir_entry.grid(
            row=0, column=1, sticky=tk.W, padx=(6, 10)
        )
        self.gui_iphase_workdir_browse_button = ttk.Button(
            pwd, text="浏览...", command=self._browse_gui_iphase_work_dir
        )
        self.gui_iphase_workdir_browse_button.grid(row=0, column=2, sticky=tk.W, padx=(0, 10))
        self.gui_iphase_workdir_browse_button.configure(state=tk.DISABLED)
        ttk.Label(pwd, text="node_id").grid(row=0, column=3, sticky=tk.W)
        self.gui_iphase_node_var = tk.StringVar(value="OBS_node")
        ttk.Entry(pwd, textvariable=self.gui_iphase_node_var, width=24).grid(
            row=0, column=4, sticky=tk.W, padx=(6, 0)
        )
        ptx = ttk.Frame(self.gui_iphase_frame)
        ptx.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(ptx, text="tx_files").grid(row=0, column=0, sticky=tk.NW)
        self.gui_iphase_tx_text = tk.Text(ptx, height=3, wrap=tk.NONE)
        self.gui_iphase_tx_text.grid(row=0, column=1, sticky=tk.EW, padx=(6, 6))
        btn_col = ttk.Frame(ptx)
        btn_col.grid(row=0, column=2, sticky=tk.NW)
        ttk.Button(
            btn_col,
            text="多选...",
            command=self._browse_iphase_tx_files,
        ).pack(anchor=tk.W)
        ttk.Button(
            btn_col,
            text="项目树导入",
            command=self._import_iphase_tx_from_project_tree,
        ).pack(anchor=tk.W, pady=(4, 0))
        ptx.columnconfigure(1, weight=1)
        ptx2 = ttk.Frame(self.gui_iphase_frame)
        ptx2.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(ptx2, text="name_filter").grid(row=0, column=0, sticky=tk.W)
        self.gui_iphase_tx_filter_var = tk.StringVar(value="")
        ttk.Entry(ptx2, textvariable=self.gui_iphase_tx_filter_var).grid(
            row=0, column=1, sticky=tk.EW, padx=(6, 6)
        )
        ttk.Label(
            ptx2,
            text="仅导入文件名包含该关键词的 tx 文件（留空=不过滤，例如 OBS22）",
        ).grid(row=0, column=2, sticky=tk.W)
        ptx2.columnconfigure(1, weight=1)
        paux = ttk.Frame(self.gui_iphase_frame)
        paux.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(paux, text="rin_file").grid(row=0, column=0, sticky=tk.W)
        self.gui_iphase_rin_var = tk.StringVar(value="")
        ttk.Entry(paux, textvariable=self.gui_iphase_rin_var).grid(
            row=0, column=1, sticky=tk.EW, padx=(6, 6)
        )
        ttk.Button(
            paux,
            text="浏览...",
            command=lambda: self._browse_gui_file(self.gui_iphase_rin_var, "选择 iphase rin 文件"),
        ).grid(row=0, column=2, sticky=tk.W)
        paux.columnconfigure(1, weight=1)
        paux2 = ttk.Frame(self.gui_iphase_frame)
        paux2.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(paux2, text="txout_file").grid(row=0, column=0, sticky=tk.W)
        self.gui_iphase_txout_var = tk.StringVar(value="")
        ttk.Entry(paux2, textvariable=self.gui_iphase_txout_var).grid(
            row=0, column=1, sticky=tk.EW, padx=(6, 6)
        )
        ttk.Button(
            paux2,
            text="浏览...",
            command=lambda: self._browse_gui_file(self.gui_iphase_txout_var, "选择 iphase tx.out 文件"),
        ).grid(row=0, column=2, sticky=tk.W)
        paux2.columnconfigure(1, weight=1)
        p2 = ttk.Frame(self.gui_iphase_frame)
        p2.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(p2, text="extra_args").grid(row=0, column=0, sticky=tk.W)
        self.gui_iphase_extra_var = tk.StringVar(value="")
        ttk.Entry(p2, textvariable=self.gui_iphase_extra_var).grid(
            row=0, column=1, sticky=tk.EW, padx=(6, 0)
        )
        p2.columnconfigure(1, weight=1)

        # Petrology bridge (imodel session → LIP GUI)
        pb0 = ttk.Frame(self.gui_petrology_bridge_frame)
        pb0.pack(fill=tk.X, pady=(4, 4), padx=6)
        ttk.Label(
            pb0,
            text="指定 imodel 运行的 gui_state JSON（含 petrology_obs_json / petrology_transect_windows）。",
            wraplength=720,
        ).pack(anchor=tk.W)
        pb1 = ttk.Frame(self.gui_petrology_bridge_frame)
        pb1.pack(fill=tk.X, pady=(0, 4), padx=6)
        ttk.Label(pb1, text="gui_state").grid(row=0, column=0, sticky=tk.W)
        self.gui_petrology_state_var = tk.StringVar(value="")
        ttk.Entry(pb1, textvariable=self.gui_petrology_state_var).grid(
            row=0, column=1, sticky=tk.EW, padx=(6, 6)
        )
        ttk.Button(
            pb1,
            text="浏览...",
            command=self._browse_gui_petrology_state_file,
        ).grid(row=0, column=2, sticky=tk.W, padx=(0, 4))
        ttk.Button(
            pb1,
            text="最近 imodel 运行",
            command=self._fill_petrology_state_from_latest_imodel,
        ).grid(row=0, column=3, sticky=tk.W)
        pb1.columnconfigure(1, weight=1)
        self.gui_petrology_bridge_status_var = tk.StringVar(value="(未读取会话)")
        ttk.Label(
            self.gui_petrology_bridge_frame,
            textvariable=self.gui_petrology_bridge_status_var,
            wraplength=720,
        ).pack(anchor=tk.W, padx=6, pady=(0, 4))
        pb2 = ttk.Frame(self.gui_petrology_bridge_frame)
        pb2.pack(fill=tk.X, pady=(0, 6), padx=6)
        ttk.Button(
            pb2,
            text="从会话读取观测",
            command=self._refresh_petrology_bridge_status,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            pb2,
            text="启动 LIP Petrology（带 imodel 观测）",
            command=self._launch_lip_from_petrology_bridge,
        ).pack(side=tk.LEFT)

    def _create_project(self) -> None:
        selected = filedialog.askdirectory(title="选择项目目录（若不存在会自动创建）")
        if not selected:
            return
        root = Path(selected).expanduser().resolve()
        overwrite = False
        if (root / "project.yaml").exists():
            overwrite = messagebox.askyesno(
                "项目已存在",
                f"{root}\n已存在 project.yaml。\n是否覆盖初始化（保留已有文件，仅重写项目元信息）？",
            )
            if not overwrite:
                return
        try:
            self.current_project = self.project_manager.create_project(
                root, name=root.name, overwrite=overwrite
            )
        except ProjectError as exc:
            messagebox.showerror("创建项目失败", str(exc))
            return
        self._refresh_all()
        self.status_var.set(f"已创建项目：{self.current_project.root}")

    def _open_project(self) -> None:
        selected = filedialog.askdirectory(title="选择 pyAOBS 项目目录")
        if not selected:
            return
        root = Path(selected).expanduser().resolve()
        try:
            self.current_project = self.project_manager.open_project(root)
        except ProjectError as exc:
            messagebox.showerror("打开项目失败", str(exc))
            return
        self._refresh_all()
        self._restore_ui_state(silent=True, show_missing_notice=False)
        self.status_var.set(f"已打开项目：{self.current_project.root}（已自动恢复界面状态）")

    def _refresh_all(self) -> None:
        if not self.current_project:
            return
        self.project_label_var.set(f"当前项目：{self.current_project.root}")
        self._refresh_project_tree()
        self._refresh_run_history()
        self._refresh_last_batch_panel()

    def _refresh_project_tree(self) -> None:
        assert self.current_project is not None
        expanded_paths: set[str] = set()
        selected_path: str | None = None
        for item, p in list(self._project_item_to_path.items()):
            try:
                if bool(self.project_tree.item(item, "open")):
                    expanded_paths.add(str(p.resolve()))
            except Exception:
                pass
        sel = self.project_tree.selection()
        if sel:
            sp = self._project_item_to_path.get(sel[0])
            if sp is not None:
                try:
                    selected_path = str(sp.resolve())
                except Exception:
                    selected_path = str(sp)

        self._project_tree_refresh_token += 1
        token = self._project_tree_refresh_token
        root_path = self.current_project.root
        self._project_tree_refresh_running = True
        self._update_refresh_status()

        worker = threading.Thread(
            target=self._scan_project_tree_worker,
            args=(root_path, token, expanded_paths, selected_path),
            daemon=True,
        )
        worker.start()

    def _scan_project_tree_worker(
        self,
        root_path: Path,
        token: int,
        expanded_paths: set[str],
        selected_path: str | None,
    ) -> None:
        nodes = self._scan_tree_nodes(root_path, depth=0, max_depth=5)
        self.root.after(
            0,
            lambda: self._apply_project_tree_scan(
                root_path, token, nodes, expanded_paths, selected_path
            ),
        )

    def _scan_tree_nodes(
        self, parent_path: Path, *, depth: int, max_depth: int
    ) -> list[dict[str, object]]:
        if depth >= max_depth:
            return []
        try:
            children: list[Path] = []
            truncated = False
            for path in parent_path.iterdir():
                children.append(path)
                if len(children) >= self._project_tree_max_children_per_dir:
                    truncated = True
                    break
            children.sort(key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return []

        nodes: list[dict[str, object]] = []
        for path in children:
            if path.name.startswith("."):
                continue
            is_dir = path.is_dir()
            try:
                resolved = path.resolve()
            except Exception:
                resolved = path
            nodes.append({"path": resolved, "is_dir": is_dir, "depth": depth + 1})
            no_recurse = is_dir and path.name.lower() in self._project_tree_no_recurse_names
            if is_dir and not no_recurse:
                nodes.extend(self._scan_tree_nodes(path, depth=depth + 1, max_depth=max_depth))
        if truncated:
            nodes.append(
                {
                    "path": None,
                    "is_dir": False,
                    "depth": depth + 1,
                    "placeholder": f"... (仅显示前 {self._project_tree_max_children_per_dir} 项)",
                }
            )
        return nodes

    def _apply_project_tree_scan(
        self,
        root_path: Path,
        token: int,
        nodes: list[dict[str, object]],
        expanded_path_texts: set[str],
        selected_path: str | None,
    ) -> None:
        self._project_tree_refresh_running = False
        self._update_refresh_status()
        if token != self._project_tree_refresh_token:
            return
        if self.current_project is None:
            return
        if self.current_project.root.resolve() != root_path.resolve():
            return

        expanded_paths: set[str] = set(expanded_path_texts)

        self.project_tree.delete(*self.project_tree.get_children())
        self._project_item_to_path.clear()

        root_item = self.project_tree.insert(
            "",
            tk.END,
            text=f"{root_path.name}/",
            image=self._tree_icon_project,
            open=True,
        )
        self._project_item_to_path[root_item] = root_path
        parents: dict[int, str] = {0: root_item}
        for node in nodes:
            depth = int(node.get("depth", 1))
            parent_item = parents.get(max(0, depth - 1), root_item)
            placeholder = str(node.get("placeholder", "")).strip()
            if placeholder:
                self.project_tree.insert(parent_item, tk.END, text=placeholder, open=False)
                continue
            path_obj = node.get("path")
            if not isinstance(path_obj, Path):
                continue
            is_dir = bool(node.get("is_dir", False))
            label = path_obj.name + ("/" if is_dir else "")
            icon = self._tree_icon_dir if is_dir else self._tree_icon_file
            try:
                resolved = path_obj.resolve()
            except Exception:
                resolved = path_obj
            is_open = str(resolved) in expanded_paths
            item = self.project_tree.insert(
                parent_item, tk.END, text=label, image=icon, open=is_open
            )
            self._project_item_to_path[item] = resolved
            if is_dir:
                parents[depth] = item
        if selected_path is not None:
            try:
                self._select_tree_path(Path(selected_path))
            except Exception:
                pass

    def _init_project_tree_icons(self) -> None:
        # Use tiny bitmap-like icons instead of emoji so they render on all fonts.
        self._tree_icon_project = tk.PhotoImage(width=14, height=14)
        self._tree_icon_dir = tk.PhotoImage(width=14, height=14)
        self._tree_icon_file = tk.PhotoImage(width=14, height=14)

        # project/folder icon base
        for icon in (self._tree_icon_project, self._tree_icon_dir):
            icon.put("#D8A327", to=(1, 4, 13, 12))
            icon.put("#F2C14E", to=(1, 5, 13, 11))
            icon.put("#C7921E", to=(1, 3, 7, 6))
        # project icon accent
        self._tree_icon_project.put("#4F8CC9", to=(9, 2, 13, 5))

        # file icon
        self._tree_icon_file.put("#9AA4B2", to=(2, 1, 12, 13))
        self._tree_icon_file.put("#FFFFFF", to=(3, 2, 11, 12))
        self._tree_icon_file.put("#C7CED8", to=(8, 1, 12, 5))

    def _refresh_run_history(self) -> None:
        assert self.current_project is not None
        self._run_history_refresh_token += 1
        token = self._run_history_refresh_token
        project_root = self.current_project.root
        self._run_history_refresh_running = True
        self._update_refresh_status()
        worker = threading.Thread(
            target=self._scan_run_history_worker,
            args=(project_root, token),
            daemon=True,
        )
        worker.start()

    def _scan_run_history_worker(self, project_root: Path, token: int) -> None:
        runs_dir = project_root / "runs"
        records: list[dict[str, str | Path]] = []
        statuses: set[str] = set()
        nodes: set[str] = set()
        truncated = False
        if runs_dir.exists():
            scanned = 0
            try:
                run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.name, reverse=True)
            except OSError:
                run_dirs = []
            for run_dir in run_dirs:
                if scanned >= self._run_history_scan_limit:
                    truncated = True
                    break
                if not run_dir.is_dir():
                    continue
                scanned += 1
                manifest_path = run_dir / "manifest.json"
                if not manifest_path.exists():
                    continue
                try:
                    m = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                run_id = str(m.get("run_id", run_dir.name))
                node_id = str(m.get("node_id", ""))
                status = str(m.get("status", ""))
                finished_at = str(m.get("finished_at", "") or "")
                elapsed = m.get("elapsed_s", "")
                elapsed_str = f"{elapsed:.2f}" if isinstance(elapsed, (int, float)) else str(elapsed)
                command_text = " ".join([str(x) for x in m.get("command", [])]) or ""
                error_text = str(m.get("error", "") or "")
                search_blob = " ".join(
                    [run_id, node_id, status, finished_at, elapsed_str, command_text, error_text]
                ).lower()
                records.append(
                    {
                        "run_id": run_id,
                        "node_id": node_id,
                        "status": status,
                        "finished_at": finished_at,
                        "elapsed_s": elapsed_str,
                        "return_code": str(m.get("return_code", "")),
                        "created_at": str(m.get("created_at", "") or ""),
                        "command": command_text,
                        "error": error_text,
                        "run_dir": run_dir,
                        "search_blob": search_blob,
                    }
                )
                if status:
                    statuses.add(status)
                if node_id:
                    nodes.add(node_id)
        self.root.after(
            0,
            lambda: self._apply_run_history_scan(
                project_root, token, records, statuses, nodes, truncated
            ),
        )

    def _apply_run_history_scan(
        self,
        project_root: Path,
        token: int,
        records: list[dict[str, str | Path]],
        statuses: set[str],
        nodes: set[str],
        truncated: bool,
    ) -> None:
        self._run_history_refresh_running = False
        self._update_refresh_status()
        if token != self._run_history_refresh_token:
            return
        if self.current_project is None:
            return
        if self.current_project.root.resolve() != project_root.resolve():
            return

        self._run_records = records
        self.run_filter_status_combo.configure(values=["全部"] + sorted(statuses))
        if self.run_filter_status_var.get() not in self.run_filter_status_combo.cget("values"):
            self.run_filter_status_var.set("全部")
        self.run_filter_node_combo.configure(values=["全部"] + sorted(nodes))
        if self.run_filter_node_var.get() not in self.run_filter_node_combo.cget("values"):
            self.run_filter_node_var.set("全部")
        self._apply_run_history_filters()
        if truncated:
            self.status_var.set(
                f"运行历史较多：仅扫描最近 {self._run_history_scan_limit} 个 workspace 目录。"
            )

    def _update_refresh_status(self) -> None:
        parts: list[str] = []
        if self._project_tree_refresh_running:
            parts.append("项目树刷新中")
        if self._run_history_refresh_running:
            parts.append("运行历史刷新中")
        self.task_status_var.set(" | ".join(parts))

    def _apply_run_history_filters(self) -> None:
        self.run_tree.delete(*self.run_tree.get_children())
        self._run_id_to_dir.clear()
        visible_count = 0
        for rec in self._filtered_run_records():
            row = (
                str(rec["run_id"]),
                str(rec["node_id"]),
                str(rec["status"]),
                str(rec["finished_at"]),
                str(rec["elapsed_s"]),
            )
            self.run_tree.insert("", tk.END, values=row)
            self._run_id_to_dir[str(rec["run_id"])] = Path(rec["run_dir"])
            visible_count += 1
        if visible_count:
            first = self.run_tree.get_children()
            if first:
                self.run_tree.selection_set(first[0])
                self.run_tree.focus(first[0])
                self._refresh_selected_run_detail()
                self._refresh_live_log_view()
        else:
            if self._run_records:
                self._set_run_detail_text("当前筛选条件下无运行记录。")
                self._set_live_log_text("当前筛选条件下无运行记录。")
            else:
                self._set_run_detail_text("当前项目暂无运行记录。")
                self._set_live_log_text("当前项目暂无运行记录。")

    def _filtered_run_records(self) -> list[dict[str, str | Path]]:
        return [rec for rec in self._run_records if self._run_record_matches(rec)]

    def _run_record_matches(self, rec: dict[str, str | Path]) -> bool:
        status_filter = self.run_filter_status_var.get().strip()
        node_filter = self.run_filter_node_var.get().strip()
        kw = self.run_filter_keyword_var.get().strip().lower()
        only_failed = bool(self.run_filter_failed_only_var.get())

        status = str(rec.get("status", ""))
        node = str(rec.get("node_id", ""))
        if status_filter and status_filter != "全部" and status != status_filter:
            return False
        if node_filter and node_filter != "全部" and node != node_filter:
            return False
        if only_failed and status.lower() != "failed":
            return False
        if kw and kw not in str(rec.get("search_blob", "")):
            return False
        return True

    def _on_run_filter_changed(self, _event: tk.Event | None = None) -> None:
        self._apply_run_history_filters()

    def _clear_run_filters(self) -> None:
        self.run_filter_status_var.set("全部")
        self.run_filter_node_var.set("全部")
        self.run_filter_keyword_var.set("")
        self.run_filter_failed_only_var.set(False)
        self._apply_run_history_filters()

    def _export_filtered_runs_csv(self) -> None:
        if self.current_project is None:
            messagebox.showwarning("未打开项目", "请先新建或打开项目。")
            return
        records = self._filtered_run_records()
        if not records:
            messagebox.showinfo("无可导出数据", "当前筛选条件下没有可导出的运行记录。")
            return
        initial_dir = self.current_project.resolve("runs")
        target = filedialog.asksaveasfilename(
            title="导出筛选结果为 CSV",
            initialdir=str(initial_dir if initial_dir.exists() else self.current_project.root),
            defaultextension=".csv",
            filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")],
        )
        if not target:
            return
        path = Path(target)
        fieldnames = [
            "run_id",
            "return_code",
            "created_at",
            "finished_at",
            "elapsed_s",
            "command",
            "error",
            "run_dir",
        ]
        try:
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for rec in records:
                    writer.writerow(
                        {
                            "run_id": str(rec.get("run_id", "")),
                            "return_code": str(rec.get("return_code", "")),
                            "created_at": str(rec.get("created_at", "")),
                            "finished_at": str(rec.get("finished_at", "")),
                            "elapsed_s": str(rec.get("elapsed_s", "")),
                            "command": str(rec.get("command", "")),
                            "error": str(rec.get("error", "")),
                            "run_dir": str(rec.get("run_dir", "")),
                        }
                    )
        except Exception as exc:
            messagebox.showerror("导出失败", f"CSV 导出失败：{exc}")
            return
        self.status_var.set(f"已导出 {len(records)} 条记录到：{path}")

    def _export_filtered_runs_json(self) -> None:
        if self.current_project is None:
            messagebox.showwarning("未打开项目", "请先新建或打开项目。")
            return
        records = self._filtered_run_records()
        if not records:
            messagebox.showinfo("无可导出数据", "当前筛选条件下没有可导出的运行记录。")
            return
        initial_dir = self.current_project.resolve("runs")
        target = filedialog.asksaveasfilename(
            title="导出筛选结果为 JSON",
            initialdir=str(initial_dir if initial_dir.exists() else self.current_project.root),
            defaultextension=".json",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if not target:
            return
        path = Path(target)
        payload = [
            {
                "run_id": str(rec.get("run_id", "")),
                "node_id": str(rec.get("node_id", "")),
                "status": str(rec.get("status", "")),
                "return_code": str(rec.get("return_code", "")),
                "created_at": str(rec.get("created_at", "")),
                "finished_at": str(rec.get("finished_at", "")),
                "elapsed_s": str(rec.get("elapsed_s", "")),
                "command": str(rec.get("command", "")),
                "error": str(rec.get("error", "")),
                "run_dir": str(rec.get("run_dir", "")),
            }
            for rec in records
        ]
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("导出失败", f"JSON 导出失败：{exc}")
            return
        self.status_var.set(f"已导出 {len(records)} 条记录到：{path}")

    def _on_plugin_changed(self, _event: tk.Event | None = None) -> None:
        plugin_id = self.plugin_var.get().strip()
        if plugin_id == "data.shell":
            # Backward compatibility for old presets/state.
            plugin_id = "data.gui"
            self.plugin_var.set(plugin_id)
        if not plugin_id:
            self.plugin_desc_var.set("")
            return
        plugin = self.tool_registry.get(plugin_id)
        self.plugin_desc_var.set(f"{plugin.name}：{plugin.description}")
        is_tomo = plugin_id == "tomo2d.shell"
        is_gui = plugin_id.endswith(".gui")
        self.gui_form_plugin_var.set(plugin_id or "(none)")
        self._update_node_form_mode(is_gui)
        if is_tomo and not self.node_id_var.get().strip():
            self.node_id_var.set("OBS_node")
        if is_tomo:
            if not self.tomo_template_frame.winfo_manager():
                self.tomo_template_frame.pack(fill=tk.X, pady=(0, 6), before=self.node_form_frame)
            if self.gui_quick_form_frame.winfo_manager():
                self.gui_quick_form_frame.pack_forget()
        else:
            if self.tomo_template_frame.winfo_manager():
                self.tomo_template_frame.pack_forget()
            if not self.gui_quick_form_frame.winfo_manager():
                self.gui_quick_form_frame.pack(fill=tk.X, pady=(0, 6), before=self.node_form_frame)
            self.template_var.set(TEMPLATE_CUSTOM)
            self._on_template_changed()
            self._toggle_gui_quick_subforms(plugin_id)
            if is_gui:
                self.exec_var.set("python")
            elif not self.exec_var.get().strip():
                self.exec_var.set("python")
            if not self.node_id_var.get().strip():
                self.node_id_var.set("OBS_node")

    def _update_node_form_mode(self, is_gui_plugin: bool) -> None:
        if is_gui_plugin:
            self.cwd_var.set("")
            self.cwd_entry.configure(state=tk.DISABLED)
            self.cwd_browse_button.configure(state=tk.DISABLED)
            for btn in self.cwd_quick_buttons:
                btn.configure(state=tk.DISABLED)
            self._clear_inputs_text()
            self.inputs_text.configure(state=tk.DISABLED)
            self.inputs_quick_add_button.configure(state=tk.DISABLED)
            if hasattr(self, "status_var"):
                self.status_var.set(
                    "GUI 节点使用 run_id 沙箱目录，工作目录与 inputs 由 GUI 内部选择。"
                )
            return
        self.cwd_entry.configure(state=tk.NORMAL)
        self.cwd_browse_button.configure(state=tk.NORMAL)
        for btn in self.cwd_quick_buttons:
            btn.configure(state=tk.NORMAL)
        self.inputs_text.configure(state=tk.NORMAL)
        self.inputs_quick_add_button.configure(state=tk.NORMAL)

    def _toggle_gui_quick_subforms(self, plugin_id: str) -> None:
        for frm in (
            self.gui_zplot_frame,
            self.gui_imodel_frame,
            self.gui_iphase_frame,
            self.gui_petrology_bridge_frame,
        ):
            if frm.winfo_manager():
                frm.pack_forget()
        note = ""
        if plugin_id in ("zplotpy.gui", "imodel.gui", "iphase.gui", "tomo2d.gui", "petrology.lip.gui"):
            note = "该 GUI 在启动后由其自身界面选择输入文件（与 tomo2d.gui 一致）。"
        elif plugin_id == "data.gui":
            note = "data.gui 启动统一 idata 界面，无需预填输入。"
        self.gui_form_note_var.set(note)
        if plugin_id in ("imodel.gui", "petrology.lip.gui"):
            self.gui_petrology_bridge_frame.pack(
                fill=tk.X, pady=(0, 6), padx=6, before=self.node_form_frame
            )
            self._refresh_petrology_bridge_status()

    def _set_run_detail_text(self, content: str) -> None:
        self.run_detail_text.configure(state=tk.NORMAL)
        self.run_detail_text.delete("1.0", tk.END)
        self.run_detail_text.insert("1.0", self._with_text_padding(content))
        self.run_detail_text.configure(state=tk.DISABLED)

    def _set_live_log_text(self, content: str) -> None:
        self.live_log_text.configure(state=tk.NORMAL)
        self.live_log_text.delete("1.0", tk.END)
        self.live_log_text.insert("1.0", self._with_text_padding(content))
        self.live_log_text.configure(state=tk.DISABLED)

    def _update_task_status_bar(self) -> None:
        single = "运行中" if self._single_run_active else "空闲"
        if self._single_run_active and self._single_run_cancel_event is not None and self._single_run_cancel_event.is_set():
            single = "取消中"
        if self._batch_rerun_active:
            batch = (
                f"运行中 {self._batch_progress_done}/{self._batch_progress_total} "
                f"(succ={self._batch_progress_success} fail={self._batch_progress_failed} "
                f"cancelled={self._batch_progress_cancelled})"
            )
            if self._batch_cancel_requested:
                batch += " [取消中]"
        else:
            batch = "空闲"
        self.task_status_var.set(f"任务状态 | 单任务: {single} | 批量: {batch}")

    def _set_batch_progress(
        self, done: int, total: int, success: int, failed: int, cancelled: int
    ) -> None:
        self._batch_progress_done = done
        self._batch_progress_total = total
        self._batch_progress_success = success
        self._batch_progress_failed = failed
        self._batch_progress_cancelled = cancelled
        self._update_task_status_bar()

    @staticmethod
    def _read_tail_text_with_trunc(path: Path, max_chars: int = 12000) -> tuple[str, bool]:
        if not path.exists() or not path.is_file():
            return "(日志文件不存在)", False
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"(日志读取失败: {exc})", False
        if len(text) <= max_chars:
            return (text if text else "(空日志)"), False
        return (text[-max_chars:], True)

    def _set_last_batch_text(self, content: str) -> None:
        self.last_batch_text.configure(state=tk.NORMAL)
        self.last_batch_text.delete("1.0", tk.END)
        self.last_batch_text.insert("1.0", self._with_text_padding(content))
        self.last_batch_text.configure(state=tk.DISABLED)

    def _set_batch_queue_text(self, content: str) -> None:
        self.batch_queue_text.configure(state=tk.NORMAL)
        self.batch_queue_text.delete("1.0", tk.END)
        self.batch_queue_text.insert("1.0", self._with_text_padding(content))
        self.batch_queue_text.configure(state=tk.DISABLED)

    @staticmethod
    def _with_text_padding(content: str) -> str:
        text = (content or "").strip("\n")
        if not text:
            return "\n\n"
        return "\n" + text + "\n"

    def _refresh_batch_queue_panel(self) -> None:
        if not (self._batch_queue_pending or self._batch_queue_running or self._batch_queue_recent_done):
            self._set_batch_queue_text("当前无批量任务队列。")
            self.batch_queue_recent_tree.delete(*self.batch_queue_recent_tree.get_children())
            return
        lines = [
            f"pending: {len(self._batch_queue_pending)} | running: {len(self._batch_queue_running)} | recent_done: {len(self._batch_queue_recent_done)}",
            "",
            "running:",
        ]
        lines.extend([f"- {x}" for x in (self._batch_queue_running or ["(无)"])])
        lines.append("")
        lines.append("pending(top 20):")
        if self._batch_queue_pending:
            lines.extend([f"- {x}" for x in self._batch_queue_pending[:20]])
            if len(self._batch_queue_pending) > 20:
                lines.append(f"... 共 {len(self._batch_queue_pending)} 条")
        else:
            lines.append("(无)")
        self._set_batch_queue_text("\n".join(lines))
        self.batch_queue_recent_tree.delete(*self.batch_queue_recent_tree.get_children())
        for rec in self._batch_queue_recent_done[-50:]:
            self.batch_queue_recent_tree.insert(
                "",
                tk.END,
                values=(
                    str(rec.get("source_run_id", "")),
                    str(rec.get("status", "")),
                    str(rec.get("new_run_id", "")),
                ),
            )

    def _set_batch_queue_snapshot(
        self, pending: list[str], running: list[str], recent_done: list[dict[str, str]]
    ) -> None:
        self._batch_queue_pending = list(pending)
        self._batch_queue_running = list(running)
        self._batch_queue_recent_done = list(recent_done)
        self._refresh_batch_queue_panel()

    @staticmethod
    def _parse_batch_recent_values(values: object) -> dict[str, str] | None:
        if not isinstance(values, (list, tuple)) or len(values) < 3:
            return None
        return {
            "source_run_id": str(values[0]).strip(),
            "status": str(values[1]).strip(),
            "new_run_id": str(values[2]).strip(),
        }

    def _selected_batch_recent_record(self) -> dict[str, str] | None:
        sel = self.batch_queue_recent_tree.selection()
        if not sel:
            return None
        vals = self.batch_queue_recent_tree.item(sel[0], "values")
        return self._parse_batch_recent_values(vals)

    def _focus_run_from_batch_recent(self, *, show_message: bool = True) -> str | None:
        rec = self._selected_batch_recent_record()
        if rec is None:
            if show_message:
                messagebox.showinfo("未选择记录", "请先在最近完成项中选择一条记录。")
            return None
        new_run_id = str(rec.get("new_run_id", "")).strip()
        status = str(rec.get("status", "")).strip()
        if not new_run_id:
            if show_message:
                messagebox.showinfo("无法定位", f"该项状态为 {status}，没有新工作区ID可定位。")
            return None
        self._refresh_run_history()
        if not self._select_run_id(new_run_id):
            if show_message:
                messagebox.showwarning("未找到运行记录", f"未在运行历史中找到工作区ID: {new_run_id}")
            return None
        self.right_notebook.select(self.tab_history)
        return new_run_id

    def _on_batch_queue_recent_context_menu(self, event: tk.Event) -> None:
        row = self.batch_queue_recent_tree.identify_row(event.y)
        if row:
            self.batch_queue_recent_tree.selection_set(row)
            self.batch_queue_recent_tree.focus(row)
        if not self.batch_queue_recent_tree.selection():
            return
        try:
            self.batch_queue_recent_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.batch_queue_recent_menu.grab_release()

    def _open_batch_recent_run_folder(self) -> None:
        if self._focus_run_from_batch_recent() is None:
            return
        self._open_selected_run_folder()

    def _copy_batch_recent_new_run_id(self) -> None:
        rec = self._selected_batch_recent_record()
        if rec is None:
            messagebox.showinfo("未选择记录", "请先在最近完成项中选择一条记录。")
            return
        run_id = str(rec.get("new_run_id", "")).strip()
        status = str(rec.get("status", "")).strip()
        if not run_id:
            messagebox.showinfo("无工作区ID", f"该项状态为 {status}，没有新工作区ID可复制。")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(run_id)
        self.status_var.set(f"已复制新工作区ID: {run_id}")

    def _open_batch_recent_live_log(self) -> None:
        if self._focus_run_from_batch_recent() is None:
            return
        self._refresh_selected_run_detail()
        self._refresh_live_log_view()

    def _on_batch_queue_recent_open_run(self, event: tk.Event | None = None) -> None:
        if event is not None:
            row = self.batch_queue_recent_tree.identify_row(event.y)
            if row:
                self.batch_queue_recent_tree.selection_set(row)
                self.batch_queue_recent_tree.focus(row)
        if self._focus_run_from_batch_recent() is None:
            return
        self._refresh_selected_run_detail()
        self._refresh_live_log_view()

    def _filtered_last_batch_rows(self) -> list[dict[str, str]]:
        rows = self._last_batch_export_rows()
        status_filter = self.last_batch_status_filter_var.get().strip()
        if not status_filter or status_filter == "全部":
            return rows
        return [r for r in rows if str(r.get("status", "")) == status_filter]

    def _on_last_batch_filter_changed(self, _event: tk.Event | None = None) -> None:
        self._refresh_last_batch_panel()

    def _refresh_last_batch_panel(self) -> None:
        all_rows = self._last_batch_export_rows()
        if not all_rows:
            self._set_last_batch_text("暂无批量复跑结果。")
            return
        rows = self._filtered_last_batch_rows()
        success = sum(1 for r in all_rows if str(r.get("status", "")) == "success")
        failed = sum(1 for r in all_rows if str(r.get("status", "")) == "failed")
        cancelled = sum(1 for r in all_rows if str(r.get("status", "")) == "cancelled")
        skipped = sum(1 for r in all_rows if str(r.get("status", "")) == "skipped")
        current_filter = self.last_batch_status_filter_var.get().strip() or "全部"
        lines = [
            f"总计(全部): {len(all_rows)}  success={success} failed={failed} cancelled={cancelled} skipped={skipped}",
            f"当前筛选: {current_filter}  显示条数: {len(rows)}",
            "",
            "明细:",
        ]
        if rows:
            for r in rows:
                lines.append(
                    "- src={source} -> new={new} status={status} code={code} elapsed={elapsed} err={err}".format(
                        source=str(r.get("source_run_id", "")),
                        new=str(r.get("new_run_id", "")),
                        status=str(r.get("status", "")),
                        code=str(r.get("return_code", "")),
                        elapsed=str(r.get("elapsed_s", "")),
                        err=str(r.get("error", "")),
                    )
                )
        else:
            lines.append("(当前筛选下无明细)")
        self._set_last_batch_text("\n".join(lines))

    def _copy_last_batch_panel(self) -> None:
        text = self.last_batch_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("无内容", "最近批量结果面板为空。")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set("已复制最近批量结果到剪贴板。")

    def _selected_run_id(self) -> str | None:
        sel = self.run_tree.selection()
        if not sel:
            return None
        vals = self.run_tree.item(sel[0], "values")
        if not vals:
            return None
        return str(vals[0])

    def _select_run_id(self, run_id: str) -> bool:
        for item in self.run_tree.get_children():
            vals = self.run_tree.item(item, "values")
            if vals and str(vals[0]) == run_id:
                self.run_tree.selection_set(item)
                self.run_tree.focus(item)
                self.run_tree.see(item)
                return True
        return False

    def _on_run_tree_select(self, _event: tk.Event | None = None) -> None:
        self._refresh_selected_run_detail()
        self._refresh_live_log_view()

    @staticmethod
    def _read_tail_text(path: Path, max_chars: int = 3000) -> str:
        if not path.exists() or not path.is_file():
            return "(文件不存在)"
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"(读取失败: {exc})"
        if len(text) <= max_chars:
            return text.strip() or "(空)"
        return "...(截断，显示末尾)...\n" + text[-max_chars:].strip()

    @staticmethod
    def _resolve_log_paths_from_manifest(run_dir: Path, manifest: dict) -> tuple[Path, Path]:
        logs = manifest.get("logs", {}) if isinstance(manifest.get("logs"), dict) else {}
        stdout_rel = str(logs.get("stdout", ""))
        stderr_rel = str(logs.get("stderr", ""))
        stdout_path = run_dir / stdout_rel if stdout_rel else run_dir / "logs" / "stdout.log"
        stderr_path = run_dir / stderr_rel if stderr_rel else run_dir / "logs" / "stderr.log"
        return stdout_path, stderr_path

    def _refresh_selected_run_detail(self) -> None:
        run_id = self._selected_run_id()
        if not run_id:
            self._set_run_detail_text("请选择一条运行记录查看详情。")
            return
        run_dir = self._run_id_to_dir.get(run_id)
        if run_dir is None:
            self._set_run_detail_text(f"未找到工作区目录：{run_id}")
            return
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            self._set_run_detail_text(f"未找到 manifest：{manifest_path}")
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._set_run_detail_text(f"manifest 解析失败：{exc}")
            return

        stdout_path, stderr_path = self._resolve_log_paths_from_manifest(run_dir, manifest)
        audit_path = self._resolve_audit_log_path(run_dir, manifest)

        lines = [
            f"workspace_id: {manifest.get('run_id', run_id)}",
            f"node_id: {manifest.get('node_id', '')}",
            "说明: workspace_id=工作区ID（runs/<node_id>目录名），node_id=流程节点名。",
            f"status: {manifest.get('status', '')}",
            f"return_code: {manifest.get('return_code', '')}",
            f"created_at: {manifest.get('created_at', '')}",
            f"finished_at: {manifest.get('finished_at', '')}",
            f"elapsed_s: {manifest.get('elapsed_s', '')}",
            "",
            "command:",
            " ".join([str(x) for x in manifest.get("command", [])]) or "(空)",
            "",
            f"cwd: {manifest.get('cwd', '')}",
            "",
            f"stdout tail ({stdout_path}):",
            self._read_tail_text(stdout_path, max_chars=2400),
            "",
            f"stderr tail ({stderr_path}):",
            self._read_tail_text(stderr_path, max_chars=2400),
            "",
            f"audit session log ({audit_path if audit_path is not None else '(未配置)'}):",
            self._read_tail_text(audit_path, max_chars=2400) if audit_path is not None else "(无)",
        ]
        self._set_run_detail_text("\n".join(lines))

    def _on_live_log_source_changed(self, _event: tk.Event | None = None) -> None:
        self._refresh_live_log_view()

    def _on_live_log_keyword_changed(self, _event: tk.Event | None = None) -> None:
        self._refresh_live_log_view()

    def _toggle_live_log_follow(self) -> None:
        if self.live_log_follow_var.get():
            self._schedule_live_log_follow()
        else:
            self._cancel_live_log_follow()

    def _cancel_live_log_follow(self) -> None:
        if self._log_follow_after_id is not None:
            try:
                self.root.after_cancel(self._log_follow_after_id)
            except Exception:
                pass
            self._log_follow_after_id = None

    def _schedule_live_log_follow(self) -> None:
        self._cancel_live_log_follow()
        if not self.live_log_follow_var.get():
            return
        self._log_follow_after_id = self.root.after(1500, self._live_log_follow_tick)

    def _live_log_follow_tick(self) -> None:
        self._log_follow_after_id = None
        if not self.live_log_follow_var.get():
            return
        self._refresh_live_log_view()
        self._schedule_live_log_follow()

    def _refresh_live_log_view(self) -> None:
        run_id = self._selected_run_id()
        if not run_id:
            self._set_live_log_text("请选择一条运行记录查看日志。")
            return
        run_dir = self._run_id_to_dir.get(run_id)
        if run_dir is None:
            self._set_live_log_text(f"未找到工作区目录：{run_id}")
            return
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            self._set_live_log_text(f"未找到 manifest：{manifest_path}")
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._set_live_log_text(f"manifest 解析失败：{exc}")
            return
        stdout_path, stderr_path = self._resolve_log_paths_from_manifest(run_dir, manifest)
        source = self.live_log_source_var.get().strip() or "stdout"
        target = stdout_path if source == "stdout" else stderr_path
        content, truncated = self._read_tail_text_with_trunc(target, max_chars=12000)
        prefix = f"[{source}] {target}\n"
        if truncated:
            prefix += "(仅显示末尾 12000 字符)\n\n"
        else:
            prefix += "\n"
        keyword = self.live_log_keyword_var.get().strip()
        self._render_live_log_content(prefix, content, keyword, self.live_log_only_match_var.get())

    def _render_live_log_content(
        self, prefix: str, content: str, keyword: str, only_match: bool
    ) -> None:
        text_widget = self.live_log_text
        text_widget.configure(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)
        text_widget.tag_delete("log_highlight")
        text_widget.tag_configure("log_highlight", foreground="#d9480f")
        text_widget.insert("1.0", "\n" + prefix)

        body = content if content else "(空日志)"
        kw = keyword.lower()
        if kw:
            lines = body.splitlines()
            if only_match:
                lines = [ln for ln in lines if kw in ln.lower()]
            body = "\n".join(lines) if lines else "(无匹配日志行)"
        text_widget.insert("end", body + "\n")

        if kw and body and body != "(无匹配日志行)":
            start = "1.0"
            while True:
                pos = text_widget.search(keyword, start, stopindex=tk.END, nocase=True)
                if not pos:
                    break
                end = f"{pos}+{len(keyword)}c"
                text_widget.tag_add("log_highlight", pos, end)
                start = end
        text_widget.configure(state=tk.DISABLED)

    def _open_selected_run_folder(self) -> None:
        run_id = self._selected_run_id()
        if not run_id:
            messagebox.showwarning("未选择运行", "请先在运行历史中选中一条记录。")
            return
        run_dir = self._run_id_to_dir.get(run_id)
        if run_dir is None or not run_dir.exists():
            messagebox.showwarning("目录不存在", f"未找到工作区目录：{run_id}")
            return
        try:
            self._open_path_with_system(run_dir)
        except Exception as exc:
            messagebox.showerror("打开目录失败", str(exc))
            return
        self.status_var.set(f"已打开工作区目录：{run_dir}")

    @staticmethod
    def _mnt_to_windows_path(path_text: str) -> str | None:
        s = path_text.strip()
        if len(s) >= 7 and s.startswith("/mnt/") and s[5].isalpha() and s[6] == "/":
            drive = s[5].upper()
            rest = s[7:].replace("/", "\\")
            return f"{drive}:\\{rest}"
        return None

    def _open_path_with_system(self, path: Path) -> None:
        target = path.resolve()
        target_str = str(target)

        if hasattr(os, "startfile"):
            os.startfile(target_str)  # type: ignore[attr-defined]
            return

        mnt_win = self._mnt_to_windows_path(target_str)
        if mnt_win:
            try:
                subprocess.Popen(["explorer.exe", mnt_win])
                return
            except Exception:
                pass

        if sys.platform == "darwin":
            subprocess.Popen(["open", target_str])
            return

        opener = shutil.which("xdg-open")
        if opener:
            subprocess.Popen([opener, target_str])
            return

        raise RuntimeError(f"当前平台不支持自动打开路径，请手动打开：{target_str}")

    def _resolve_audit_log_path(self, run_dir: Path, manifest: dict) -> Path | None:
        audit = manifest.get("audit", {}) if isinstance(manifest.get("audit"), dict) else {}
        raw = str(audit.get("session_log", "")).strip()
        if not raw:
            return None
        p = Path(raw)
        if p.is_absolute():
            return p
        cands: list[Path] = []
        if self.current_project is not None:
            cands.append((self.current_project.root / p).resolve())
        cands.append((run_dir / p).resolve())
        for cp in cands:
            if cp.exists():
                return cp
        return cands[0] if cands else None

    def _open_selected_audit_log(self) -> None:
        selected = self._selected_run_manifest()
        if selected is None:
            return
        run_id, run_dir, manifest = selected
        audit_path = self._resolve_audit_log_path(run_dir, manifest)
        if audit_path is None:
            messagebox.showinfo("无审计日志", f"workspace_id={run_id} 未配置 GUI 审计日志。")
            return
        if not audit_path.exists():
            messagebox.showwarning("审计日志不存在", f"未找到审计日志文件：{audit_path}")
            return
        try:
            self._open_path_with_system(audit_path)
        except Exception as exc:
            messagebox.showerror("打开失败", f"无法打开审计日志：{exc}")
            return
        self.status_var.set(f"已打开审计日志：{audit_path}")

    def _delete_selected_run(self) -> None:
        run_id = self._selected_run_id()
        if not run_id:
            messagebox.showwarning("未选择运行", "请先在运行历史中选中一条记录。")
            return
        visible_run_ids = []
        selected_idx = -1
        for idx, item in enumerate(self.run_tree.get_children()):
            vals = self.run_tree.item(item, "values")
            if not vals:
                continue
            rid = str(vals[0])
            visible_run_ids.append(rid)
            if rid == run_id:
                selected_idx = idx
        run_dir = self._run_id_to_dir.get(run_id)
        if run_dir is None:
            messagebox.showwarning("未找到目录", f"未找到工作区目录：{run_id}")
            return
        if not run_dir.exists():
            messagebox.showwarning("目录不存在", f"工作区目录不存在：{run_dir}")
            self._refresh_run_history()
            return
        if self._single_run_active or self._batch_rerun_active:
            messagebox.showwarning("任务进行中", "当前有运行中的任务，请等待完成后再删除工作区记录。")
            return

        node_id = ""
        status = ""
        try:
            manifest_path = run_dir / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                node_id = str(manifest.get("node_id", ""))
                status = str(manifest.get("status", ""))
        except Exception:
            node_id = ""
            status = ""

        detail = f"workspace_id: {run_id}"
        if node_id:
            detail += f"\nnode_id: {node_id}"
        if status:
            detail += f"\nstatus: {status}"
        detail += f"\n\n目录将被永久删除：\n{run_dir}\n\n是否继续？"
        if not messagebox.askyesno("确认删除工作区记录", detail):
            return

        try:
            shutil.rmtree(run_dir)
        except Exception as exc:
            messagebox.showerror("删除失败", f"删除工作区目录失败：{exc}")
            return

        neighbor_run_id = ""
        if visible_run_ids:
            if selected_idx >= 0 and selected_idx + 1 < len(visible_run_ids):
                neighbor_run_id = visible_run_ids[selected_idx + 1]
            elif selected_idx > 0:
                neighbor_run_id = visible_run_ids[selected_idx - 1]

        runs_path = self.current_project.resolve("runs") if self.current_project else None
        selected_tree_path: Path | None = None
        if runs_path is not None:
            selected_tree_path = runs_path
        self._refresh_run_history()
        if neighbor_run_id and neighbor_run_id != run_id and self._select_run_id(neighbor_run_id):
            self._refresh_selected_run_detail()
            self._refresh_live_log_view()
        self._refresh_project_tree()
        if selected_tree_path is not None:
            self._select_tree_path(selected_tree_path)
        self.status_var.set(f"已删除工作区记录：{run_id}")

    def _bulk_delete_non_success_runs(self) -> None:
        if self._single_run_active or self._batch_rerun_active:
            messagebox.showwarning("任务进行中", "当前有运行中的任务，请等待完成后再执行批量删除。")
            return
        targets = [
            rec
            for rec in self._run_records
            if str(rec.get("status", "")).strip().lower() in {"failed", "cancelled"}
        ]
        if not targets:
            messagebox.showinfo("无可删除记录", "当前没有 failed/cancelled 的工作区记录。")
            return
        failed_count = sum(
            1 for rec in targets if str(rec.get("status", "")).strip().lower() == "failed"
        )
        cancelled_count = sum(
            1 for rec in targets if str(rec.get("status", "")).strip().lower() == "cancelled"
        )
        preview = "\n".join([f"- {str(rec.get('run_id', ''))}" for rec in targets[:12]])
        if len(targets) > 12:
            preview += f"\n... 共 {len(targets)} 条"
        ok = messagebox.askyesno(
            "确认一键清理非成功记录",
            (
                f"将删除 failed/cancelled 的工作区目录。\n"
                f"failed: {failed_count}\n"
                f"cancelled: {cancelled_count}\n"
                f"总计: {len(targets)}\n\n"
                f"{preview}\n\n"
                "是否继续？"
            ),
        )
        if not ok:
            return
        run_ids = [str(rec.get("run_id", "")).strip() for rec in targets if str(rec.get("run_id", "")).strip()]
        self._delete_runs_by_ids(run_ids, action_title="一键清理非成功记录")

    def _delete_runs_by_ids(self, run_ids: list[str], *, action_title: str) -> None:
        if not run_ids:
            messagebox.showinfo("无可删除记录", "没有可删除的工作区ID。")
            return
        rec_map: dict[str, dict[str, str | Path]] = {
            str(rec.get("run_id", "")): rec for rec in self._run_records if str(rec.get("run_id", "")).strip()
        }
        missing = [rid for rid in run_ids if rid not in rec_map]
        existing = [rid for rid in run_ids if rid in rec_map]
        if not existing:
            detail = "\n".join([f"- {rid}" for rid in missing[:20]])
            if len(missing) > 20:
                detail += f"\n... 共 {len(missing)} 条"
            messagebox.showwarning("未命中", f"未找到可删除的工作区ID：\n{detail}")
            return

        deleted = 0
        failures: list[str] = []
        for rid in existing:
            rec = rec_map[rid]
            run_dir_raw = rec.get("run_dir")
            run_dir = Path(run_dir_raw) if isinstance(run_dir_raw, Path) else Path(str(run_dir_raw))
            try:
                if run_dir.exists():
                    shutil.rmtree(run_dir)
                    deleted += 1
                else:
                    failures.append(f"{rid}: 目录不存在")
            except Exception as exc:
                failures.append(f"{rid}: {exc}")

        self._refresh_project_tree()
        self._refresh_run_history()
        if self.current_project is not None:
            self._select_tree_path(self.current_project.resolve("runs"))

        msg = f"{action_title}完成：成功删除 {deleted} 条。"
        if missing:
            msg += f"\n未命中 {len(missing)} 条。"
        if failures:
            msg += f"\n失败 {len(failures)} 条。"
        self.status_var.set(msg)
        if missing or failures:
            detail_lines = []
            if missing:
                detail_lines.append("未命中工作区ID:")
                detail_lines.extend([f"- {rid}" for rid in missing[:20]])
                if len(missing) > 20:
                    detail_lines.append(f"... 共 {len(missing)} 条")
            if failures:
                detail_lines.append("")
                detail_lines.append("删除失败:")
                detail_lines.extend([f"- {x}" for x in failures[:20]])
                if len(failures) > 20:
                    detail_lines.append(f"... 共 {len(failures)} 条")
            messagebox.showwarning(action_title, "\n".join(detail_lines))

    def _selected_run_manifest(self) -> tuple[str, Path, dict] | None:
        run_id = self._selected_run_id()
        if not run_id:
            messagebox.showwarning("未选择运行", "请先在运行历史中选中一条记录。")
            return None
        run_dir = self._run_id_to_dir.get(run_id)
        if run_dir is None:
            messagebox.showwarning("未找到目录", f"未找到工作区目录：{run_id}")
            return None
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            messagebox.showwarning("未找到 manifest", f"缺少文件：{manifest_path}")
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            messagebox.showerror("读取失败", f"manifest 解析失败：{exc}")
            return None
        return run_id, run_dir, manifest

    def _payload_from_manifest(self, manifest: dict) -> dict[str, str]:
        command = manifest.get("command", [])
        executable = ""
        args = ""
        if isinstance(command, list) and command:
            executable = str(command[0])
            args = " ".join([str(x) for x in command[1:]])

        env_overrides = manifest.get("env_overrides", {})
        env_lines: list[str] = []
        if isinstance(env_overrides, dict):
            for k, v in env_overrides.items():
                env_lines.append(f"{k}={v}")
        env_text = ("\n".join(env_lines) + "\n") if env_lines else ""

        input_files = manifest.get("input_files", [])
        input_lines: list[str] = []
        if isinstance(input_files, list):
            for item in input_files:
                if isinstance(item, dict):
                    p = str(item.get("path", "")).strip()
                    if p:
                        exists_flag = bool(item.get("exists", False))
                        if not exists_flag and not Path(p).exists():
                            continue
                        input_lines.append(p)
        inputs_text = ("\n".join(input_lines) + "\n") if input_lines else ""

        cwd_abs = str(manifest.get("cwd", "") or "")
        cwd_text = cwd_abs
        if self.current_project is not None and cwd_abs:
            try:
                cwd_rel = Path(cwd_abs).resolve().relative_to(self.current_project.root.resolve())
                cwd_text = str(cwd_rel)
            except Exception:
                cwd_text = cwd_abs

        return {
            "node_id": str(manifest.get("node_id", "") or ""),
            "executable": executable,
            "args": args,
            "cwd": cwd_text,
            "env_text": env_text,
            "inputs_text": inputs_text,
        }

    def _prefill_selected_run_to_runner(self) -> None:
        if self.current_project is None:
            messagebox.showwarning("未打开项目", "请先新建或打开项目。")
            return
        selected = self._selected_run_manifest()
        if selected is None:
            return
        run_id, _run_dir, manifest = selected
        payload = self._payload_from_manifest(manifest)
        self._set_run_payload(payload)
        if self.plugin_var.get().strip() != "tomo2d.shell" and "tomo2d.shell" in self.tool_registry.ids():
            self.plugin_var.set("tomo2d.shell")
            self._on_plugin_changed()
        self.template_var.set(TEMPLATE_CUSTOM)
        self._on_template_changed()
        self.right_notebook.select(self.tab_runner)
        self.status_var.set(f"已回填运行参数：{run_id}")

    def _rerun_selected_failed(self) -> None:
        selected = self._selected_run_manifest()
        if selected is None:
            return
        run_id, _run_dir, manifest = selected
        status = str(manifest.get("status", "") or "")
        if status.lower() != "failed":
            go_on = messagebox.askyesno(
                "非失败运行",
                f"当前记录状态为 {status or 'unknown'}，仍要作为模板复跑吗？",
            )
            if not go_on:
                return
        self._prefill_selected_run_to_runner()
        self._run_selected_node()

    def _rerun_failed_filtered_batch(self) -> None:
        if self.current_project is None:
            messagebox.showwarning("未打开项目", "请先新建或打开项目。")
            return
        if self._batch_rerun_active:
            messagebox.showinfo("任务进行中", "批量复跑任务正在执行，请稍候。")
            return

        filtered = self._filtered_run_records()
        failed = [rec for rec in filtered if str(rec.get("status", "")).lower() == "failed"]
        if not failed:
            messagebox.showinfo("无失败记录", "当前筛选结果中没有失败记录。")
            return

        try:
            max_workers = int(self.run_batch_concurrency_var.get().strip() or "1")
        except ValueError:
            max_workers = 1
        max_workers = max(1, max_workers)

        specs, runnable_lines, skipped_lines, skipped_records = self._collect_batch_rerun_specs(failed)
        if not specs:
            detail = "\n".join(skipped_lines) if skipped_lines else "未能从失败记录中构建可执行任务。"
            messagebox.showwarning("无可执行任务", detail)
            return

        go_on = self._show_batch_precheck_report(
            total_failed=len(failed),
            runnable_lines=runnable_lines,
            skipped_lines=skipped_lines,
            max_workers=max_workers,
        )
        if not go_on:
            return

        self._last_batch_results = []
        self._last_batch_skipped = skipped_records
        self._refresh_last_batch_panel()
        self._batch_rerun_active = True
        self._batch_rerun_cancel_event = threading.Event()
        self._batch_cancel_requested = False
        self._set_batch_progress(0, len(specs), 0, 0, 0)
        initial_pending = [str(s[5].get("batch_source_run_id", "")) for s in specs]
        self._set_batch_queue_snapshot(initial_pending, [], [])
        self.batch_rerun_button.configure(state=tk.DISABLED)
        self.batch_cancel_button.configure(state=tk.NORMAL)
        self.status_var.set(f"批量复跑已启动：{len(specs)} 条，并发={max_workers}")

        worker = threading.Thread(
            target=self._run_batch_specs_worker,
            args=(self.current_project, specs, max_workers, self._batch_rerun_cancel_event),
            daemon=True,
        )
        worker.start()

    def _cancel_batch_rerun(self) -> None:
        if not self._batch_rerun_active or self._batch_rerun_cancel_event is None:
            messagebox.showinfo("无批量任务", "当前没有可取消的批量任务。")
            return
        self._batch_rerun_cancel_event.set()
        self._batch_cancel_requested = True
        self._update_task_status_bar()
        self.batch_cancel_button.configure(state=tk.DISABLED)
        self.status_var.set("已请求取消批量复跑：将停止提交新任务并尝试中止运行中的任务...")

    def _collect_batch_rerun_specs(
        self, records: list[dict[str, str | Path]]
    ) -> tuple[
        list[tuple[str, list[str], dict[str, str], list[Path], str | Path | None, dict]],
        list[str],
        list[str],
        list[dict[str, str]],
    ]:
        specs: list[tuple[str, list[str], dict[str, str], list[Path], str | Path | None, dict]] = []
        runnable_lines: list[str] = []
        skipped_lines: list[str] = []
        skipped_records: list[dict[str, str]] = []
        plugin = self.tool_registry.get("tomo2d.shell")
        assert self.current_project is not None
        for rec in records:
            run_id = str(rec.get("run_id", ""))
            run_dir = rec.get("run_dir")
            if not isinstance(run_dir, Path):
                skipped_lines.append(f"- {run_id}: 缺少 run_dir 信息")
                skipped_records.append({"source_run_id": run_id, "reason": "缺少 run_dir 信息"})
                continue
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.exists():
                skipped_lines.append(f"- {run_id}: 缺少 manifest.json")
                skipped_records.append({"source_run_id": run_id, "reason": "缺少 manifest.json"})
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                payload = self._payload_from_manifest(manifest)
                spec = plugin.build_spec(self.current_project.root, payload)
            except Exception as exc:
                skipped_lines.append(f"- {run_id}: 任务构建失败（{exc}）")
                skipped_records.append({"source_run_id": run_id, "reason": f"任务构建失败: {exc}"})
                continue
            params = dict(spec.params)
            params["batch_source_run_id"] = run_id
            specs.append((spec.node_id, spec.command, spec.env, spec.inputs, spec.cwd, params))
            cmd_preview = " ".join([str(x) for x in spec.command[:6]])
            if len(spec.command) > 6:
                cmd_preview += " ..."
            runnable_lines.append(
                f"- {run_id} -> node={spec.node_id} cwd={spec.cwd or '.'} cmd={cmd_preview}"
            )
        return specs, runnable_lines, skipped_lines, skipped_records

    def _show_batch_precheck_report(
        self,
        *,
        total_failed: int,
        runnable_lines: list[str],
        skipped_lines: list[str],
        max_workers: int,
    ) -> bool:
        win = tk.Toplevel(self.root)
        win.title("批量复跑预检查报告")
        win.geometry("1020x700")

        body = ttk.Frame(win, padding=8)
        body.pack(fill=tk.BOTH, expand=True)

        header = (
            f"筛选失败总数: {total_failed}\n"
            f"可执行: {len(runnable_lines)}\n"
            f"将跳过: {len(skipped_lines)}\n"
            f"批量并发: {max_workers}\n"
        )
        ttk.Label(body, text=header, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 6))

        txt = tk.Text(body, wrap=tk.WORD)
        txt.pack(fill=tk.BOTH, expand=True)
        txt.insert("1.0", "## 可执行任务\n")
        txt.insert("end", ("\n".join(runnable_lines) if runnable_lines else "(无)") + "\n\n")
        txt.insert("end", "## 将跳过任务\n")
        txt.insert("end", ("\n".join(skipped_lines) if skipped_lines else "(无)") + "\n")
        txt.configure(state=tk.DISABLED)

        confirmed = tk.BooleanVar(value=False)

        def on_confirm() -> None:
            confirmed.set(True)
            win.destroy()

        btns = ttk.Frame(body)
        btns.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(btns, text="取消", command=win.destroy).pack(side=tk.RIGHT)
        ttk.Button(btns, text="确认启动批量复跑", command=on_confirm).pack(
            side=tk.RIGHT, padx=(0, 8)
        )

        win.transient(self.root)
        win.grab_set()
        self.root.wait_window(win)
        return bool(confirmed.get())

    def _last_batch_export_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        rows.extend(self._last_batch_results)
        for item in self._last_batch_skipped:
            rows.append(
                {
                    "source_run_id": str(item.get("source_run_id", "")),
                    "new_run_id": "",
                    "node_id": "",
                    "status": "skipped",
                    "return_code": "",
                    "elapsed_s": "",
                    "cwd": "",
                    "command": "",
                    "error": str(item.get("reason", "")),
                }
            )
        return rows

    def _export_last_batch_results_csv(self) -> None:
        if self.current_project is None:
            messagebox.showwarning("未打开项目", "请先新建或打开项目。")
            return
        rows = self._last_batch_export_rows()
        if not rows:
            messagebox.showinfo("无数据", "最近没有可导出的批量复跑结果。")
            return
        initial_dir = self.current_project.resolve("runs")
        target = filedialog.asksaveasfilename(
            title="导出最近批量结果为 CSV",
            initialdir=str(initial_dir if initial_dir.exists() else self.current_project.root),
            defaultextension=".csv",
            filetypes=[("CSV 文件", "*.csv"), ("所有文件", "*.*")],
        )
        if not target:
            return
        path = Path(target)
        fieldnames = [
            "source_run_id",
            "new_run_id",
            "return_code",
            "elapsed_s",
            "cwd",
            "command",
            "error",
        ]
        try:
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: str(row.get(k, "")) for k in fieldnames})
        except Exception as exc:
            messagebox.showerror("导出失败", f"CSV 导出失败：{exc}")
            return
        self.status_var.set(f"已导出最近批量结果 {len(rows)} 条到：{path}")

    def _export_last_batch_results_json(self) -> None:
        if self.current_project is None:
            messagebox.showwarning("未打开项目", "请先新建或打开项目。")
            return
        rows = self._last_batch_export_rows()
        if not rows:
            messagebox.showinfo("无数据", "最近没有可导出的批量复跑结果。")
            return
        initial_dir = self.current_project.resolve("runs")
        target = filedialog.asksaveasfilename(
            title="导出最近批量结果为 JSON",
            initialdir=str(initial_dir if initial_dir.exists() else self.current_project.root),
            defaultextension=".json",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if not target:
            return
        path = Path(target)
        try:
            path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("导出失败", f"JSON 导出失败：{exc}")
            return
        self.status_var.set(f"已导出最近批量结果 {len(rows)} 条到：{path}")

    def _run_batch_specs_worker(
        self,
        project: ProjectContext,
        specs: list[tuple[str, list[str], dict[str, str], list[Path], str | Path | None, dict]],
        max_workers: int,
        cancel_event: threading.Event,
    ) -> None:
        total = len(specs)
        done = 0
        success = 0
        failed = 0
        cancelled = 0
        rows: list[dict[str, str]] = []
        source_ids = [str(s[5].get("batch_source_run_id", f"item-{i+1}")) for i, s in enumerate(specs)]
        pending_ids = list(source_ids)
        running_ids: list[str] = []
        recent_done: list[dict[str, str]] = []

        def run_one(
            item: tuple[str, list[str], dict[str, str], list[Path], str | Path | None, dict]
        ) -> dict[str, str]:
            node_id, command, env, inputs, cwd, params = item
            src = str(params.get("batch_source_run_id", ""))
            run = self.run_manager.run_command(
                project=project,
                node_id=node_id,
                command=command,
                params=params,
                inputs=inputs,
                env=env,
                cwd=cwd,
                cancel_event=cancel_event,
            )
            row: dict[str, str] = {
                "source_run_id": src,
                "new_run_id": run.run_id,
                "node_id": node_id,
                "status": "failed",
                "return_code": "",
                "elapsed_s": "",
                "cwd": str(cwd or ""),
                "command": " ".join([str(x) for x in command]),
                "error": "",
            }
            try:
                manifest = json.loads(run.manifest_file.read_text(encoding="utf-8"))
                row["status"] = str(manifest.get("status", "failed"))
                row["return_code"] = str(manifest.get("return_code", ""))
                row["elapsed_s"] = str(manifest.get("elapsed_s", ""))
                row["error"] = str(manifest.get("error", "") or "")
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = f"读取新 manifest 失败: {exc}"
            return row

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            running: set = set()
            next_idx = 0

            def submit_one() -> bool:
                nonlocal next_idx
                if cancel_event.is_set() or next_idx >= total:
                    return False
                fut = ex.submit(run_one, specs[next_idx])
                running.add(fut)
                sid = source_ids[next_idx]
                if sid in pending_ids:
                    pending_ids.remove(sid)
                running_ids.append(sid)
                self.root.after(
                    0,
                    lambda p=list(pending_ids), r=list(running_ids), d=list(recent_done): self._set_batch_queue_snapshot(
                        p, r, d
                    ),
                )
                next_idx += 1
                return True

            while len(running) < max_workers and submit_one():
                pass

            while running:
                done_set, _ = wait(running, return_when=FIRST_COMPLETED)
                for fut in done_set:
                    running.remove(fut)
                    src_done = ""
                    done += 1
                    row: dict[str, str] = {"status": "failed"}
                    try:
                        row = fut.result()
                    except Exception as exc:
                        row = {
                            "source_run_id": "",
                            "new_run_id": "",
                            "node_id": "",
                            "status": "failed",
                            "return_code": "",
                            "elapsed_s": "",
                            "cwd": "",
                            "command": "",
                            "error": f"并发任务异常: {exc}",
                        }
                    src_done = str(row.get("source_run_id", ""))
                    rows.append(row)
                    status = str(row.get("status", "failed"))
                    if status == "success":
                        success += 1
                    elif status == "cancelled":
                        cancelled += 1
                    else:
                        failed += 1
                    if src_done in running_ids:
                        running_ids.remove(src_done)
                    recent_done.append(
                        {
                            "source_run_id": src_done,
                            "status": status,
                            "new_run_id": str(row.get("new_run_id", "")),
                        }
                    )
                    if len(recent_done) > 50:
                        recent_done = recent_done[-50:]
                    self.root.after(
                        0,
                        lambda d=done, t=total, s=success, f=failed, c=cancelled: (
                            self.status_var.set(
                                f"批量复跑进行中: {d}/{t}，success={s} failed={f} cancelled={c}"
                            ),
                            self._set_batch_progress(d, t, s, f, c),
                        ),
                    )
                    self.root.after(
                        0,
                        lambda p=list(pending_ids), r=list(running_ids), rd=list(recent_done): self._set_batch_queue_snapshot(
                            p, r, rd
                        ),
                    )
                    if not cancel_event.is_set():
                        submit_one()

            if cancel_event.is_set() and next_idx < total:
                for i in range(next_idx, total):
                    _node_id, command, _env, _inputs, cwd, params = specs[i]
                    rows.append(
                        {
                            "source_run_id": str(params.get("batch_source_run_id", "")),
                            "new_run_id": "",
                            "node_id": "",
                            "status": "skipped",
                            "return_code": "",
                            "elapsed_s": "",
                            "cwd": str(cwd or ""),
                            "command": " ".join([str(x) for x in command]),
                            "error": "cancelled before submit",
                        }
                    )
                    sid = source_ids[i] if i < len(source_ids) else ""
                    if sid in pending_ids:
                        pending_ids.remove(sid)
                    recent_done.append(
                        {
                            "source_run_id": sid,
                            "status": "skipped",
                            "new_run_id": "",
                        }
                    )
                    if len(recent_done) > 50:
                        recent_done = recent_done[-50:]
            self.root.after(
                0,
                lambda p=list(pending_ids), r=list(running_ids), rd=list(recent_done): self._set_batch_queue_snapshot(
                    p, r, rd
                ),
            )

        self.root.after(
            0,
            lambda: self._finish_batch_rerun(total, success, failed, cancelled, rows, cancel_event.is_set()),
        )

    def _finish_batch_rerun(
        self,
        total: int,
        success: int,
        failed: int,
        cancelled: int,
        rows: list[dict[str, str]],
        cancel_requested: bool,
    ) -> None:
        self._batch_rerun_active = False
        self._batch_rerun_cancel_event = None
        self._batch_cancel_requested = False
        self.batch_rerun_button.configure(state=tk.NORMAL)
        self.batch_cancel_button.configure(state=tk.DISABLED)
        self._last_batch_results = rows
        self._set_batch_progress(0, 0, 0, 0, 0)
        self._refresh_project_tree()
        self._refresh_run_history()
        self._refresh_last_batch_panel()
        skipped = sum(1 for r in rows if str(r.get("status", "")) == "skipped")
        title = "批量复跑已取消" if cancel_requested else "批量复跑完成"
        self.status_var.set(
            f"{title}: total={total} success={success} failed={failed} cancelled={cancelled} skipped={skipped}"
        )
        messagebox.showinfo(
            title,
            (
                f"计划总数: {total}\n"
                f"成功: {success}\n失败: {failed}\n取消: {cancelled}\n跳过: {skipped}\n"
                "可导出最近批量结果报告。"
            ),
        )

    def _on_template_changed(self, _event: tk.Event | None = None) -> None:
        template_id = self.template_var.get().strip()
        name = self._template_name_by_id.get(template_id, template_id)
        self.template_title_var.set(f"当前：{name}")
        if template_id == TEMPLATE_INVERSE_STANDARD:
            if not self.tpl_node_id_var.get().strip():
                self.tpl_node_id_var.set("OBS_node")
        elif template_id == TEMPLATE_FORWARD_BASIC:
            if not self.tpl_node_id_var.get().strip():
                self.tpl_node_id_var.set("OBS_node")

    def _create_collapsible_group(self, parent: tk.Widget, title: str) -> ttk.Frame:
        box = ttk.Frame(parent)
        box.pack(fill=tk.X, pady=(0, 4), padx=6)
        header = ttk.Frame(box)
        header.pack(fill=tk.X)
        body = ttk.Frame(box)
        body.pack(fill=tk.X, pady=(2, 0))
        expanded = tk.BooleanVar(value=True)

        def toggle() -> None:
            if expanded.get():
                body.pack_forget()
                expanded.set(False)
                btn.config(text=f"▶ {title}")
            else:
                body.pack(fill=tk.X, pady=(2, 0))
                expanded.set(True)
                btn.config(text=f"▼ {title}")

        btn = ttk.Button(header, text=f"▼ {title}", command=toggle)
        btn.pack(side=tk.LEFT)
        return body

    def _build_template_help_map(self) -> dict[str, str]:
        base = {
            "work_dir": "运行工作目录（相对项目根目录）。模板里的相对输入路径会相对于该目录解析。",
            "node_id": "该任务在 runs/ 中对应的工作区标识。相同 node_id 会复用同一目录并覆盖运行结果。",
            "verbose": "等价 -V 级别，常用 -1(静默)、0(简要)、1(详细)。",
            "mesh_path": "tt_inverse/tt_forward 的 -M 输入慢度/网格文件。",
            "data_path": "tt_inverse/tt_forward 的 -G 输入观测或几何数据文件。",
            "iterations": "tt_inverse 的 -I 迭代次数，建议 >= 1。",
            "sv": "tt_inverse 的 -SV 速度平滑权重。",
            "sd": "tt_inverse 的 -SD 深度/界面平滑权重。",
            "dv": "tt_inverse 的 -DV 全局速度阻尼。若使用 -DQ，应确保 DV>0。",
            "dd": "tt_inverse 的 -DD 深度阻尼权重。",
            "extra_args": "额外原生命令参数，会原样拼接在模板生成参数之后。",
        }
        if TomoHelp is not None:
            try:
                inv_help = TomoHelp.tt_inverse_help()
                fwd_help = TomoHelp.tt_forward_help()
                base["dv"] += "\n\n文档提示: tt_inverse_help 中说明 -DQ 与 -DV 需联动。"
                if "OMP_NUM_THREADS" in inv_help:
                    base["work_dir"] += "\n并行: 建议结合 OMP_NUM_THREADS/OMP_PLACES/OMP_PROC_BIND。"
                if "-N" in fwd_help:
                    base["extra_args"] += "\n例如可追加 -Nx/z/clen/nintp/tol1/tol2。"
            except Exception:
                pass
        return base

    def _attach_field_tooltips(self, mapping: dict[str, list[tk.Widget]]) -> None:
        for key, widgets in mapping.items():
            tip = self._help_map.get(key, "")
            for widget in widgets:
                self._tooltips.append(_SimpleTooltip(widget, tip))

    def _collect_template_fields(self) -> dict[str, str]:
        return {
            "work_dir": self.tpl_work_dir_var.get().strip(),
            "node_id": self.tpl_node_id_var.get().strip(),
            "verbose": self.tpl_verbose_var.get().strip(),
            "mesh_path": self.tpl_mesh_var.get().strip(),
            "data_path": self.tpl_data_var.get().strip(),
            "iterations": self.tpl_i_var.get().strip(),
            "sv": self.tpl_sv_var.get().strip(),
            "sd": self.tpl_sd_var.get().strip(),
            "dv": self.tpl_dv_var.get().strip(),
            "dd": self.tpl_dd_var.get().strip(),
            "extra_args": self.tpl_extra_var.get().strip(),
        }

    def _validate_template_fields(self, template_id: str, fields: dict[str, str]) -> tuple[bool, str]:
        if template_id == TEMPLATE_CUSTOM:
            return True, ""

        for key in ("mesh_path", "data_path", "work_dir", "node_id", "verbose"):
            if not fields.get(key, "").strip():
                return False, f"模板字段不能为空：{key}"

        try:
            int(fields["verbose"])
        except ValueError:
            return False, "verbose 必须是整数（例如 -1/0/1）。"

        if template_id == TEMPLATE_INVERSE_STANDARD:
            try:
                i = int(fields.get("iterations", ""))
                sv = float(fields.get("sv", ""))
                sd = float(fields.get("sd", ""))
                dv = float(fields.get("dv", ""))
                dd = float(fields.get("dd", ""))
            except ValueError:
                return False, "I/SV/SD/DV/DD 需要是合法数值。"
            if i < 1:
                return False, "I(迭代次数) 需要 >= 1。"
            if min(sv, sd, dv, dd) < 0:
                return False, "SV/SD/DV/DD 需要 >= 0。"

        if self.current_project is None:
            return True, ""

        root = self.current_project.root
        work_dir = fields.get("work_dir", "").strip()
        cwd = (root / work_dir).resolve() if work_dir else root
        if work_dir and not cwd.exists():
            return False, f"work_dir 不存在：{cwd}"

        for key in ("mesh_path", "data_path"):
            raw = fields.get(key, "").strip()
            p = Path(raw)
            candidate = p if p.is_absolute() else (cwd / p)
            if not candidate.exists():
                return False, f"{key} 文件不存在：{candidate}"
            if not candidate.is_file():
                return False, f"{key} 不是文件：{candidate}"
        return True, ""

    def _template_path_from_abs(self, abs_path: Path) -> str:
        if self.current_project is None:
            return str(abs_path)
        root = self.current_project.root.resolve()
        work_dir = self.tpl_work_dir_var.get().strip()
        base = (root / work_dir).resolve() if work_dir else root
        try:
            return str(abs_path.resolve().relative_to(base))
        except Exception:
            return str(abs_path.resolve())

    def _pick_template_file_from_tree(self, target: tk.StringVar) -> None:
        sel = self.project_tree.selection()
        if not sel:
            messagebox.showwarning("未选择项目树节点", "请先在左侧项目树选择一个文件。")
            return
        p = self._project_item_to_path.get(sel[0])
        if p is None or not p.exists() or not p.is_file():
            messagebox.showwarning("选择无效", "请选择项目树中的文件节点。")
            return
        target.set(self._template_path_from_abs(p))

    def _browse_template_file(self, target: tk.StringVar, title: str) -> None:
        if self.current_project is None:
            messagebox.showwarning("未打开项目", "请先新建或打开项目。")
            return
        root = self.current_project.root
        work_dir = self.tpl_work_dir_var.get().strip()
        initial_dir = (root / work_dir) if work_dir else root
        if not initial_dir.exists():
            initial_dir = root
        selected = filedialog.askopenfilename(title=title, initialdir=str(initial_dir))
        if not selected:
            return
        target.set(self._template_path_from_abs(Path(selected)))

    def _pick_mesh_from_tree(self) -> None:
        self._pick_template_file_from_tree(self.tpl_mesh_var)

    def _pick_data_from_tree(self) -> None:
        self._pick_template_file_from_tree(self.tpl_data_var)

    def _browse_mesh_file(self) -> None:
        self._browse_template_file(self.tpl_mesh_var, "选择 mesh 文件")

    def _browse_data_file(self) -> None:
        self._browse_template_file(self.tpl_data_var, "选择 data 文件")

    def _project_rel_or_abs(self, path: Path) -> str:
        if self.current_project is None:
            return str(path.resolve())
        root = self.current_project.root.resolve()
        target = path.resolve()
        try:
            rel = target.relative_to(root)
            return str(rel) if str(rel) != "." else ""
        except Exception:
            return str(target)

    def _browse_project_dir_to_var(
        self, target: tk.StringVar, title: str, *, fallback_subdir: str = ""
    ) -> None:
        if self.current_project is None:
            messagebox.showwarning("未打开项目", "请先新建或打开项目。")
            return
        root = self.current_project.root.resolve()
        raw = target.get().strip()
        if raw:
            candidate = Path(raw)
            initial_dir = candidate if candidate.is_absolute() else (root / candidate)
        elif fallback_subdir:
            initial_dir = root / fallback_subdir
        else:
            initial_dir = root
        if not initial_dir.exists():
            initial_dir = root
        selected = filedialog.askdirectory(title=title, initialdir=str(initial_dir))
        if not selected:
            return
        target.set(self._project_rel_or_abs(Path(selected)))

    def _browse_node_cwd(self) -> None:
        self._browse_project_dir_to_var(self.cwd_var, "选择运行工作目录", fallback_subdir="work")

    def _set_cwd_to_project_subdir(self, rel_subdir: str) -> None:
        if self.current_project is None:
            messagebox.showwarning("未打开项目", "请先新建或打开项目。")
            return
        target = self.current_project.resolve(rel_subdir)
        target.mkdir(parents=True, exist_ok=True)
        self.cwd_var.set(self._project_rel_or_abs(target))
        self.status_var.set(f"已设置工作目录：{self.cwd_var.get()}")

    def _set_cwd_from_tree_selection(self) -> None:
        if self.current_project is None:
            messagebox.showwarning("未打开项目", "请先新建或打开项目。")
            return
        sel = self.project_tree.selection()
        if not sel:
            messagebox.showwarning("未选择项目树节点", "请先在左侧项目树选择目录或文件。")
            return
        p = self._project_item_to_path.get(sel[0])
        if p is None or not p.exists():
            messagebox.showwarning("选择无效", "项目树选中项不存在。")
            return
        target_dir = p if p.is_dir() else p.parent
        self.cwd_var.set(self._project_rel_or_abs(target_dir))
        self.status_var.set(f"已设置工作目录：{self.cwd_var.get()}")

    def _append_tree_selection_to_inputs(self) -> None:
        if str(self.inputs_text.cget("state")) != str(tk.NORMAL):
            self.status_var.set("当前节点由 GUI 内部选择输入文件，已禁用手工 inputs。")
            return
        if self.current_project is None:
            messagebox.showwarning("未打开项目", "请先新建或打开项目。")
            return
        sel = self.project_tree.selection()
        if not sel:
            messagebox.showwarning("未选择项目树节点", "请先在左侧项目树选择目录或文件。")
            return
        p = self._project_item_to_path.get(sel[0])
        if p is None or not p.exists():
            messagebox.showwarning("选择无效", "项目树选中项不存在。")
            return
        val = self._project_rel_or_abs(p)
        if not val:
            messagebox.showwarning("选择无效", "项目根目录不能直接加入 inputs。")
            return
        existing = [ln.strip() for ln in self.inputs_text.get("1.0", tk.END).splitlines() if ln.strip()]
        if val in existing:
            self.status_var.set(f"inputs 已包含：{val}")
            return
        if existing:
            self.inputs_text.insert(tk.END, val + "\n")
        else:
            self.inputs_text.insert("1.0", val + "\n")
        self.status_var.set(f"已加入 inputs：{val}")

    def _browse_template_work_dir(self) -> None:
        self._browse_project_dir_to_var(
            self.tpl_work_dir_var, "选择模板 work_dir", fallback_subdir="work"
        )

    def _browse_gui_imodel_work_dir(self) -> None:
        self._browse_project_dir_to_var(
            self.gui_imodel_workdir_var, "选择 imodel work_dir", fallback_subdir="work"
        )

    def _browse_gui_iphase_work_dir(self) -> None:
        self._browse_project_dir_to_var(
            self.gui_iphase_workdir_var, "选择 iphase work_dir", fallback_subdir="work"
        )

    def _browse_gui_file(self, target: tk.StringVar, title: str) -> None:
        if self.current_project is None:
            messagebox.showwarning("未打开项目", "请先新建或打开项目。")
            return
        selected = filedialog.askopenfilename(
            title=title, initialdir=str(self.current_project.root)
        )
        if not selected:
            return
        target.set(selected)

    def _browse_iphase_tx_files(self) -> None:
        if self.current_project is None:
            messagebox.showwarning("未打开项目", "请先新建或打开项目。")
            return
        selected = filedialog.askopenfilenames(
            title="选择一个或多个 iphase tx.in 文件",
            initialdir=str(self.current_project.root),
            filetypes=[("tx 文件", "*.in *.tx *.txt"), ("所有文件", "*.*")],
        )
        if not selected:
            return
        self.gui_iphase_tx_text.delete("1.0", tk.END)
        self.gui_iphase_tx_text.insert("1.0", "\n".join(selected) + "\n")

    @staticmethod
    def _looks_like_tx_file(path: Path) -> bool:
        name = path.name.lower()
        suffix = path.suffix.lower()
        if suffix not in {".in", ".tx", ".txt"}:
            return False
        if name.startswith("tx") or "tx" in name:
            return True
        return False

    @staticmethod
    def _parse_name_filter_tokens(text: str) -> list[str]:
        raw = text.strip()
        if not raw:
            return []
        return [tok.strip().lower() for tok in raw.split(",") if tok.strip()]

    @staticmethod
    def _match_name_filter(filename: str, tokens: list[str]) -> bool:
        if not tokens:
            return True
        name = filename.lower()
        for tok in tokens:
            if "*" in tok or "?" in tok:
                if fnmatch.fnmatchcase(name, tok):
                    return True
            else:
                if tok in name:
                    return True
        return False

    def _import_iphase_tx_from_project_tree(self) -> None:
        if self.current_project is None:
            messagebox.showwarning("未打开项目", "请先新建或打开项目。")
            return
        sel = self.project_tree.selection()
        if not sel:
            messagebox.showwarning("未选择项目树节点", "请先在左侧项目树选择一个文件或目录。")
            return
        p = self._project_item_to_path.get(sel[0])
        if p is None or not p.exists():
            messagebox.showwarning("选择无效", "项目树选择路径无效。")
            return
        name_filter_raw = self.gui_iphase_tx_filter_var.get().strip()
        name_filter_tokens = self._parse_name_filter_tokens(name_filter_raw)

        files: list[Path] = []
        if p.is_file():
            if self._looks_like_tx_file(p):
                files = [p]
            else:
                messagebox.showwarning("文件类型不匹配", f"所选文件不是常见 tx 文件：{p.name}")
                return
        else:
            try:
                all_files = [x for x in p.rglob("*") if x.is_file()]
            except Exception as exc:
                messagebox.showerror("读取目录失败", str(exc))
                return
            files = sorted([x for x in all_files if self._looks_like_tx_file(x)])
            if not files:
                messagebox.showinfo("未找到文件", f"目录下未找到常见 tx 文件：{p}")
                return

        if name_filter_tokens:
            files = [f for f in files if self._match_name_filter(f.name, name_filter_tokens)]
            if not files:
                messagebox.showinfo(
                    "过滤后为空",
                    f"已应用 name_filter='{name_filter_raw}'，没有匹配的 tx 文件。",
                )
                return

        # merge with existing lines and deduplicate
        existing = [ln.strip() for ln in self.gui_iphase_tx_text.get("1.0", tk.END).splitlines()]
        merged: list[str] = []
        seen: set[str] = set()
        for raw in existing + [str(f) for f in files]:
            line = raw.strip()
            if not line:
                continue
            if line in seen:
                continue
            seen.add(line)
            merged.append(line)

        self.gui_iphase_tx_text.delete("1.0", tk.END)
        self.gui_iphase_tx_text.insert("1.0", "\n".join(merged) + "\n")
        suffix = f"，过滤规则: {name_filter_raw}" if name_filter_tokens else ""
        self.status_var.set(
            f"已从项目树导入 tx 文件 {len(files)} 个（列表共 {len(merged)} 个）{suffix}。"
        )

    @staticmethod
    def _quote_arg_if_needed(s: str) -> str:
        if not s:
            return s
        if any(ch.isspace() for ch in s) and not (s.startswith('"') and s.endswith('"')):
            return f'"{s}"'
        return s

    def _apply_gui_quick_form_to_command(self) -> bool:
        plugin_id = self.plugin_var.get().strip()
        if plugin_id == "data.shell":
            plugin_id = "data.gui"
        if plugin_id == "tomo2d.shell":
            self.status_var.set("当前是 TOMO2D 插件，请使用模板表单。")
            return True

        self.exec_var.set("python")
        self.cwd_var.set("")
        self.env_text.delete("1.0", tk.END)
        self._clear_inputs_text()
        self.args_text.delete("1.0", tk.END)

        args: list[str] = []
        inputs: list[str] = []
        if plugin_id == "zplotpy.gui":
            if not self.node_id_var.get().strip():
                self.node_id_var.set("OBS_node")
            self.status_var.set("zplotpy.gui 启动后请在 GUI 内部加载输入文件。")
            self.args_text.insert("1.0", "")
            return True
        elif plugin_id == "imodel.gui":
            node = self.gui_imodel_node_var.get().strip() or "OBS_node"
            self.node_id_var.set(node)
            self.status_var.set("imodel（Qt）启动后请在 GUI 内部加载模型与辅助文件。")
            self.args_text.insert("1.0", "")
            return True
        elif plugin_id == "iphase.gui":
            node = self.gui_iphase_node_var.get().strip() or "OBS_node"
            self.node_id_var.set(node)
            self.status_var.set("iphase.gui 启动后请在 GUI 内部加载输入文件。")
            self.args_text.insert("1.0", "")
            return True
        elif plugin_id == "data.gui":
            if not self.node_id_var.get().strip():
                self.node_id_var.set("OBS_node")
            self.args_text.insert("1.0", "")
            self.status_var.set("data.gui 直接启动 idata UI，无需额外参数。")
            return True
        elif plugin_id == "tomo2d.gui":
            if not self.node_id_var.get().strip():
                self.node_id_var.set("OBS_node")
            self.args_text.insert("1.0", "")
            self.status_var.set("tomo2d.gui 直接启动 TOMO2D GUI，无需额外参数。")
            return True
        elif plugin_id == "petrology.lip.gui":
            if not self.node_id_var.get().strip():
                self.node_id_var.set("lip_petrology")
            args = self._petrology_lip_import_args_from_state()
            self.args_text.insert("1.0", " ".join(args))
            if args:
                self.status_var.set(
                    "petrology.lip.gui 将携带 imodel 观测/沿迹 CSV 启动 LIP Petrology。"
                )
            else:
                self.status_var.set(
                    "petrology.lip.gui 启动 LIP 地幔熔融 GUI。"
                    "可在下方桥接区指定 gui_state 后重新「应用GUI表单到命令」。"
                )
            return True
        else:
            self.status_var.set("当前插件未提供 GUI 快速表单。")
            return True

        self.args_text.insert("1.0", " ".join(args))
        if inputs:
            self._set_inputs_text_value("\n".join(inputs) + "\n")
        self.status_var.set("已根据 GUI 表单生成命令参数。")
        return True

    def _apply_template_to_command(self) -> bool:
        if self.plugin_var.get().strip() != "tomo2d.shell":
            self.status_var.set("当前插件不使用 TOMO2D 模板。")
            return True
        template_id = self.template_var.get().strip()
        if template_id == TEMPLATE_CUSTOM:
            self.status_var.set("当前是自定义模板，直接编辑下方命令区即可。")
            return True
        fields = self._collect_template_fields()
        valid, msg = self._validate_template_fields(template_id, fields)
        if not valid:
            messagebox.showerror("模板校验失败", msg)
            return False
        try:
            payload = build_tomo2d_template_payload(
                template_id=template_id,
                fields=fields,
            )
        except ValueError as exc:
            messagebox.showerror("模板参数错误", str(exc))
            return False

        self.exec_var.set(payload["executable"])
        self.cwd_var.set(payload["cwd"])
        self.node_id_var.set(payload["node_id"])
        self.args_text.delete("1.0", tk.END)
        self.args_text.insert("1.0", payload["args"])
        self.env_text.delete("1.0", tk.END)
        self.env_text.insert("1.0", payload["env_text"])
        self._set_inputs_text_value(payload["inputs_text"])
        self.status_var.set("已根据模板生成命令。")
        return True

    def _fill_tomo2d_example(self) -> None:
        self.exec_var.set("tt_inverse")
        self.cwd_var.set("work")
        if not self.node_id_var.get().strip():
            self.node_id_var.set("OBS_node")
        self.args_text.delete("1.0", tk.END)
        self.args_text.insert(
            "1.0",
            "-Minputs/mesh.dat -Ginputs/data.dat -N4/4/0.8/8/0.0001/1e-05 "
            "-I5 -J1.0 -SV100 -SD10 -DV1 -DD20 -V-1",
        )
        self._set_inputs_text_value("work/inputs/mesh.dat\nwork/inputs/data.dat\n")
        self.status_var.set("已填入 TOMO2D 示例参数，请按需修改后运行。")

    def _fill_plugin_example(self) -> None:
        plugin_id = self.plugin_var.get().strip()
        if plugin_id == "data.shell":
            plugin_id = "data.gui"
        if plugin_id == "tomo2d.shell":
            template_id = self.template_var.get().strip()
            if template_id and template_id != TEMPLATE_CUSTOM:
                if self._apply_template_to_command():
                    name = self._template_name_by_id.get(template_id, template_id)
                    self.status_var.set(f"已按 TOMO2D 模板示例生成命令：{name}")
                return
            self._fill_tomo2d_example()
            return

        self.args_text.delete("1.0", tk.END)
        self.env_text.delete("1.0", tk.END)
        self._clear_inputs_text()
        self.cwd_var.set("")

        if plugin_id == "zplotpy.gui":
            self.gui_zplot_itype_var.set("0")
            self.gui_zplot_data_var.set("")
            self.gui_zplot_header_var.set("")
            self.gui_zplot_record_var.set("")
            self.gui_zplot_irec_var.set("")
            self._apply_gui_quick_form_to_command()
            self.status_var.set("已填入 zplotpy GUI 启动示例并生成参数。")
            return
        if plugin_id == "imodel.gui":
            self.gui_imodel_workdir_var.set("")
            self.gui_imodel_node_var.set("OBS_node")
            self.gui_imodel_model_var.set("")
            self.gui_imodel_aux_var.set("")
            self.gui_imodel_extra_var.set("")
            self._apply_gui_quick_form_to_command()
            self.status_var.set("已填入 imodel GUI 启动示例并生成参数。")
            return
        if plugin_id == "iphase.gui":
            self.gui_iphase_workdir_var.set("")
            self.gui_iphase_node_var.set("OBS_node")
            self.gui_iphase_tx_text.delete("1.0", tk.END)
            self.gui_iphase_tx_filter_var.set("")
            self.gui_iphase_rin_var.set("")
            self.gui_iphase_txout_var.set("")
            self.gui_iphase_extra_var.set("")
            self._apply_gui_quick_form_to_command()
            self.status_var.set("已填入 iphase GUI 启动示例并生成参数。")
            return
        if plugin_id == "data.gui":
            if not self.node_id_var.get().strip():
                self.node_id_var.set("OBS_node")
            self.args_text.insert("1.0", "")
            self.status_var.set("data.gui 启动 idata 统一转换 UI，无需参数。")
            return
        if plugin_id == "tomo2d.gui":
            if not self.node_id_var.get().strip():
                self.node_id_var.set("OBS_node")
            self.args_text.insert("1.0", "")
            self.status_var.set("tomo2d.gui 启动 TOMO2D 图形界面，无需参数。")
            return
        if plugin_id == "petrology.lip.gui":
            if not self.node_id_var.get().strip():
                self.node_id_var.set("lip_petrology")
            self.args_text.insert("1.0", "")
            self.status_var.set("petrology.lip.gui 启动 LIP 地幔熔融 GUI，无需参数。")
            return
        self.status_var.set("当前插件未提供示例。")

    def _browse_gui_petrology_state_file(self) -> None:
        initial = self.gui_petrology_state_var.get().strip()
        path = filedialog.askopenfilename(
            title="选择 imodel gui_state JSON",
            initialdir=str(Path(initial).parent) if initial else "",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        if path:
            self.gui_petrology_state_var.set(path)
            self._refresh_petrology_bridge_status()

    def _find_latest_imodel_gui_state(self) -> Path | None:
        if self.current_project is None:
            return None
        runs_dir = self.current_project.root / "runs"
        if not runs_dir.is_dir():
            return None
        try:
            run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.name, reverse=True)
        except OSError:
            return None
        for run_dir in run_dirs:
            if not run_dir.is_dir():
                continue
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            plugin = str((manifest.get("params") or {}).get("plugin", ""))
            if plugin != "imodel.gui":
                continue
            gs = manifest.get("gui_state") or {}
            rel = str(gs.get("state_file", "")).strip()
            if not rel:
                continue
            state_path = self.current_project.resolve(Path(rel))
            if state_path.is_file():
                return state_path
        return None

    def _fill_petrology_state_from_latest_imodel(self) -> None:
        state_path = self._find_latest_imodel_gui_state()
        if state_path is None:
            messagebox.showinfo(
                "未找到 imodel 会话",
                "当前项目 runs/ 下没有带 gui_state 的 imodel.gui 运行记录。",
            )
            return
        self.gui_petrology_state_var.set(str(state_path))
        self._refresh_petrology_bridge_status()
        self.status_var.set(f"已填充最近 imodel gui_state：{state_path.name}")

    def _refresh_petrology_bridge_status(self) -> None:
        state_text = self.gui_petrology_state_var.get().strip()
        if not state_text:
            self.gui_petrology_bridge_status_var.set("(未指定 gui_state 文件)")
            return
        state_path = Path(state_text)
        if not state_path.is_file():
            self.gui_petrology_bridge_status_var.set(f"文件不存在: {state_text}")
            return
        obs_path = petrology_obs_path_from_state(state_path)
        tran_path = petrology_transect_path_from_state(state_path)
        parts = [f"gui_state: {state_path.name}"]
        if obs_path:
            parts.append(f"观测 JSON: {obs_path.name}")
        else:
            parts.append("观测 JSON: (无)")
        if tran_path:
            parts.append(f"沿迹 windows: {tran_path.name}")
        else:
            parts.append("沿迹 windows: (无)")
        self.gui_petrology_bridge_status_var.set("  |  ".join(parts))

    def _petrology_lip_import_args_from_state(self) -> list[str]:
        state_text = self.gui_petrology_state_var.get().strip()
        if not state_text or not Path(state_text).is_file():
            return []
        state_path = Path(state_text)
        args: list[str] = []
        obs_path = petrology_obs_path_from_state(state_path)
        tran_path = petrology_transect_path_from_state(state_path)
        if obs_path:
            args.extend(["--import-obs", self._quote_arg_if_needed(str(obs_path))])
        if tran_path:
            args.extend(["--import-transect", self._quote_arg_if_needed(str(tran_path))])
        return args

    def _launch_lip_from_petrology_bridge(self) -> None:
        state_text = self.gui_petrology_state_var.get().strip()
        if not state_text:
            messagebox.showwarning("LIP Petrology", "请先指定 imodel gui_state JSON。")
            return
        state_path = Path(state_text)
        if not state_path.is_file():
            messagebox.showerror("LIP Petrology", f"gui_state 不存在:\n{state_text}")
            return
        try:
            launch_lip_from_workbench_state(state_path)
        except Exception as exc:
            messagebox.showerror("LIP Petrology", f"启动失败:\n{exc}")
            return
        self._refresh_petrology_bridge_status()
        self.status_var.set("已在独立进程启动 LIP Petrology（带 imodel 观测/沿迹）。")

    def _collect_run_payload(self) -> dict[str, str]:
        node_id = self.node_id_var.get().strip() or "OBS_node"
        return {
            "node_id": node_id,
            "executable": self.exec_var.get().strip(),
            "args": self.args_text.get("1.0", tk.END).strip(),
            "cwd": self.cwd_var.get().strip(),
            "env_text": self.env_text.get("1.0", tk.END),
            "inputs_text": self.inputs_text.get("1.0", tk.END),
        }

    def _collect_gui_quick_form_state(self) -> dict[str, str]:
        return {
            "zplot_itype": self.gui_zplot_itype_var.get(),
            "zplot_irec": self.gui_zplot_irec_var.get(),
            "zplot_data": self.gui_zplot_data_var.get(),
            "zplot_header": self.gui_zplot_header_var.get(),
            "zplot_record": self.gui_zplot_record_var.get(),
            "imodel_workdir": self.gui_imodel_workdir_var.get(),
            "imodel_node": self.gui_imodel_node_var.get(),
            "imodel_model": self.gui_imodel_model_var.get(),
            "imodel_aux": self.gui_imodel_aux_var.get(),
            "imodel_extra": self.gui_imodel_extra_var.get(),
            "iphase_workdir": self.gui_iphase_workdir_var.get(),
            "iphase_node": self.gui_iphase_node_var.get(),
            "iphase_tx_filter": self.gui_iphase_tx_filter_var.get(),
            "iphase_tx_text": self.gui_iphase_tx_text.get("1.0", tk.END),
            "iphase_rin": self.gui_iphase_rin_var.get(),
            "iphase_txout": self.gui_iphase_txout_var.get(),
            "iphase_extra": self.gui_iphase_extra_var.get(),
            "petrology_state": self.gui_petrology_state_var.get(),
        }

    def _apply_gui_quick_form_state(self, state: dict) -> None:
        self.gui_zplot_itype_var.set(str(state.get("zplot_itype", "0") or "0"))
        self.gui_zplot_irec_var.set(str(state.get("zplot_irec", "") or ""))
        self.gui_zplot_data_var.set(str(state.get("zplot_data", "") or ""))
        self.gui_zplot_header_var.set(str(state.get("zplot_header", "") or ""))
        self.gui_zplot_record_var.set(str(state.get("zplot_record", "") or ""))

        self.gui_imodel_workdir_var.set(str(state.get("imodel_workdir", "") or ""))
        self.gui_imodel_node_var.set(str(state.get("imodel_node", "OBS_node") or "OBS_node"))
        self.gui_imodel_model_var.set(str(state.get("imodel_model", "") or ""))
        self.gui_imodel_aux_var.set(str(state.get("imodel_aux", "") or ""))
        self.gui_imodel_extra_var.set(str(state.get("imodel_extra", "") or ""))

        self.gui_iphase_workdir_var.set(str(state.get("iphase_workdir", "") or ""))
        self.gui_iphase_node_var.set(str(state.get("iphase_node", "OBS_node") or "OBS_node"))
        self.gui_iphase_tx_filter_var.set(str(state.get("iphase_tx_filter", "") or ""))
        self.gui_iphase_tx_text.delete("1.0", tk.END)
        self.gui_iphase_tx_text.insert("1.0", str(state.get("iphase_tx_text", "") or ""))
        self.gui_iphase_rin_var.set(str(state.get("iphase_rin", "") or ""))
        self.gui_iphase_txout_var.set(str(state.get("iphase_txout", "") or ""))
        self.gui_iphase_extra_var.set(str(state.get("iphase_extra", "") or ""))
        self.gui_petrology_state_var.set(str(state.get("petrology_state", "") or ""))
        self._refresh_petrology_bridge_status()

    def _node_preset_dir(self) -> Path | None:
        if self.current_project is None:
            return None
        path = self.current_project.resolve(Path("state") / "node_presets")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _sanitize_preset_name(name: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())

    def _list_node_preset_paths(self) -> list[Path]:
        preset_dir = self._node_preset_dir()
        if preset_dir is None or not preset_dir.exists():
            return []
        files = [p for p in preset_dir.glob("*.json") if p.is_file()]
        files.sort(key=lambda p: p.name.lower())
        return files

    def _save_node_preset(self) -> None:
        if self.current_project is None:
            messagebox.showwarning("未打开项目", "请先新建或打开项目。")
            return
        preset_name = simpledialog.askstring(
            "保存节点预设",
            "请输入预设名称（例如 tomo2d_inv_fast）:",
            parent=self.root,
        )
        if not preset_name:
            return
        safe_name = self._sanitize_preset_name(preset_name)
        if not safe_name:
            messagebox.showerror("名称无效", "预设名称不能为空。")
            return
        preset_dir = self._node_preset_dir()
        assert preset_dir is not None
        path = preset_dir / f"{safe_name}.json"
        payload = {
            "schema_version": 1,
            "plugin_id": self.plugin_var.get().strip(),
            "template_id": self.template_var.get().strip(),
            "run_payload": self._collect_run_payload(),
            "gui_quick_form": self._collect_gui_quick_form_state(),
            "preset_meta": {
                "tags": [self.plugin_var.get().strip()] if self.plugin_var.get().strip() else [],
            },
        }
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))
            return
        self.status_var.set(f"已保存节点预设：{path}")

    def _load_node_preset(self) -> None:
        if self.current_project is None:
            messagebox.showwarning("未打开项目", "请先新建或打开项目。")
            return
        preset_dir = self._node_preset_dir()
        assert preset_dir is not None
        selected = filedialog.askopenfilename(
            title="选择节点预设文件",
            initialdir=str(preset_dir),
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if not selected:
            return
        self._load_node_preset_from_path(Path(selected))

    def _load_node_preset_from_path(self, path: Path) -> bool:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            messagebox.showerror("加载失败", f"预设文件解析失败：{exc}")
            return False

        plugin_id = str(obj.get("plugin_id", "") or "")
        if plugin_id and plugin_id in self.tool_registry.ids():
            self.plugin_var.set(plugin_id)
            self._on_plugin_changed()
        template_id = str(obj.get("template_id", "") or "")
        if template_id:
            self.template_var.set(template_id)
            self._on_template_changed()

        gui_state = obj.get("gui_quick_form", {})
        if isinstance(gui_state, dict):
            self._apply_gui_quick_form_state(gui_state)
        run_payload = obj.get("run_payload", {})
        if isinstance(run_payload, dict):
            cleaned: dict[str, str] = {}
            for k, v in run_payload.items():
                cleaned[str(k)] = "" if v is None else str(v)
            self._set_run_payload(cleaned)
        self.status_var.set(f"已加载节点预设：{path}")
        return True

    def _manage_node_presets(self) -> None:
        if self.current_project is None:
            messagebox.showwarning("未打开项目", "请先新建或打开项目。")
            return
        preset_dir = self._node_preset_dir()
        assert preset_dir is not None

        win = tk.Toplevel(self.root)
        win.title("节点预设管理")
        win.geometry("920x560")
        body = ttk.Frame(win, padding=8)
        body.pack(fill=tk.BOTH, expand=True)

        cols = ("name", "plugin", "tags", "mtime")
        tree = ttk.Treeview(body, columns=cols, show="headings", selectmode="browse")
        tree.heading("name", text="Preset")
        tree.heading("plugin", text="Plugin")
        tree.heading("tags", text="Tags")
        tree.heading("mtime", text="Modified")
        tree.column("name", width=320, anchor=tk.W)
        tree.column("plugin", width=180, anchor=tk.W)
        tree.column("tags", width=200, anchor=tk.W)
        tree.column("mtime", width=220, anchor=tk.W)
        tree.pack(fill=tk.BOTH, expand=True)

        item_to_path: dict[str, Path] = {}
        filter_var = tk.StringVar(value="")

        filter_row = ttk.Frame(body)
        filter_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(filter_row, text="筛选").pack(side=tk.LEFT)
        ent_filter = ttk.Entry(filter_row, textvariable=filter_var, width=36)
        ent_filter.pack(side=tk.LEFT, padx=(6, 8))

        def selected_path() -> Path | None:
            sel = tree.selection()
            if not sel:
                return None
            return item_to_path.get(sel[0])

        def refresh() -> None:
            tree.delete(*tree.get_children())
            item_to_path.clear()
            keyword = filter_var.get().strip().lower()
            for p in self._list_node_preset_paths():
                plugin = ""
                tags_text = ""
                try:
                    obj = json.loads(p.read_text(encoding="utf-8"))
                    plugin = str(obj.get("plugin_id", "") or "")
                    meta = obj.get("preset_meta", {})
                    tags = meta.get("tags", []) if isinstance(meta, dict) else []
                    if isinstance(tags, list):
                        tags_text = ",".join([str(t) for t in tags if str(t).strip()])
                except Exception:
                    plugin = "(invalid json)"
                    tags_text = ""
                if keyword:
                    blob = f"{p.stem} {plugin} {tags_text}".lower()
                    if keyword not in blob:
                        continue
                mtime = p.stat().st_mtime
                import datetime as _dt

                mtime_text = _dt.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                item = tree.insert("", tk.END, values=(p.stem, plugin, tags_text, mtime_text))
                item_to_path[item] = p

        def action_load() -> None:
            p = selected_path()
            if p is None:
                messagebox.showwarning("未选择预设", "请先选择一个预设。", parent=win)
                return
            ok = self._load_node_preset_from_path(p)
            if ok:
                win.destroy()

        def action_rename() -> None:
            p = selected_path()
            if p is None:
                messagebox.showwarning("未选择预设", "请先选择一个预设。", parent=win)
                return
            new_name = simpledialog.askstring("重命名预设", "新预设名称：", initialvalue=p.stem, parent=win)
            if not new_name:
                return
            safe = self._sanitize_preset_name(new_name)
            if not safe:
                messagebox.showerror("名称无效", "新名称不能为空。", parent=win)
                return
            target = preset_dir / f"{safe}.json"
            if target.exists() and target.resolve() != p.resolve():
                messagebox.showerror("名称冲突", f"预设已存在：{target.name}", parent=win)
                return
            try:
                p.rename(target)
            except Exception as exc:
                messagebox.showerror("重命名失败", str(exc), parent=win)
                return
            self.status_var.set(f"已重命名预设：{p.name} -> {target.name}")
            refresh()

        def action_duplicate() -> None:
            p = selected_path()
            if p is None:
                messagebox.showwarning("未选择预设", "请先选择一个预设。", parent=win)
                return
            default_name = f"{p.stem}_copy"
            new_name = simpledialog.askstring("复制预设", "新预设名称：", initialvalue=default_name, parent=win)
            if not new_name:
                return
            safe = self._sanitize_preset_name(new_name)
            if not safe:
                messagebox.showerror("名称无效", "新名称不能为空。", parent=win)
                return
            target = preset_dir / f"{safe}.json"
            if target.exists():
                messagebox.showerror("名称冲突", f"预设已存在：{target.name}", parent=win)
                return
            try:
                shutil.copy2(p, target)
            except Exception as exc:
                messagebox.showerror("复制失败", str(exc), parent=win)
                return
            self.status_var.set(f"已复制预设：{target.name}")
            refresh()

        def action_edit_tags() -> None:
            p = selected_path()
            if p is None:
                messagebox.showwarning("未选择预设", "请先选择一个预设。", parent=win)
                return
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
                messagebox.showerror("读取失败", f"预设文件解析失败：{exc}", parent=win)
                return
            meta = obj.get("preset_meta", {})
            if not isinstance(meta, dict):
                meta = {}
            tags = meta.get("tags", [])
            current = ""
            if isinstance(tags, list):
                current = ",".join([str(t) for t in tags if str(t).strip()])
            new_tags = simpledialog.askstring(
                "编辑标签",
                "请输入标签（逗号分隔，例如 tomo2d,fast,test）:",
                initialvalue=current,
                parent=win,
            )
            if new_tags is None:
                return
            parsed = [t.strip() for t in new_tags.split(",") if t.strip()]
            meta["tags"] = parsed
            obj["preset_meta"] = meta
            try:
                p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except Exception as exc:
                messagebox.showerror("保存失败", str(exc), parent=win)
                return
            self.status_var.set(f"已更新预设标签：{p.name}")
            refresh()

        def action_delete() -> None:
            p = selected_path()
            if p is None:
                messagebox.showwarning("未选择预设", "请先选择一个预设。", parent=win)
                return
            sure = messagebox.askyesno("确认删除", f"确定删除预设？\n{p.name}", parent=win)
            if not sure:
                return
            try:
                p.unlink()
            except Exception as exc:
                messagebox.showerror("删除失败", str(exc), parent=win)
                return
            self.status_var.set(f"已删除预设：{p.name}")
            refresh()

        btns = ttk.Frame(body)
        btns.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(btns, text="刷新", command=refresh).pack(side=tk.LEFT)
        ttk.Button(btns, text="加载", command=action_load).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(btns, text="重命名", command=action_rename).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(btns, text="复制", command=action_duplicate).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(btns, text="编辑标签", command=action_edit_tags).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(btns, text="删除", command=action_delete).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(btns, text="关闭", command=win.destroy).pack(side=tk.RIGHT)

        ent_filter.bind("<KeyRelease>", lambda _e: refresh())
        refresh()
        win.transient(self.root)
        win.grab_set()

    @staticmethod
    def _extract_option_value(tokens: list[str], option: str) -> str | None:
        for idx, tok in enumerate(tokens):
            if tok == option:
                if idx + 1 < len(tokens):
                    return tokens[idx + 1]
            if tok.startswith(option) and len(tok) > len(option):
                return tok[len(option) :]
        return None

    def _build_risk_warnings(
        self,
        executable: str,
        args_text: str,
        env: dict[str, str],
    ) -> list[str]:
        risks: list[str] = []
        try:
            tokens = shlex.split(args_text, posix=False)
        except ValueError:
            tokens = []

        if executable in ("tt_inverse", "tt_forward"):
            dv = self._extract_option_value(tokens, "-DV")
            dq = self._extract_option_value(tokens, "-DQ")
            if dq is not None:
                if dv is None:
                    risks.append("检测到 -DQ 但未提供 -DV，通常会导致阻尼文件不生效。")
                else:
                    try:
                        if float(dv) <= 0:
                            risks.append("检测到 -DQ 与 -DV<=0 组合，建议设置 -DV 为正值。")
                    except ValueError:
                        risks.append("检测到 -DV 非数值，建议检查参数拼写。")

        if executable == "tt_inverse" and "TOMO2D_INV_OMP" not in env:
            risks.append("未设置 TOMO2D_INV_OMP，默认可能串行。")
        if executable == "tt_forward" and "TOMO2D_FWD_OMP" not in env:
            risks.append("未设置 TOMO2D_FWD_OMP，默认可能串行。")
        if executable in ("tt_inverse", "tt_forward"):
            if "OMP_PLACES" not in env or "OMP_PROC_BIND" not in env:
                risks.append("建议显式设置 OMP_PLACES 与 OMP_PROC_BIND 以获得更稳定并行性能。")
        if executable and shutil.which(executable) is None:
            risks.append(f"当前 PATH 未找到可执行程序：{executable}（若使用绝对路径可忽略）。")
        return risks

    def _render_preflight_message(
        self,
        template_id: str,
        payload: dict[str, str],
        risks: list[str],
    ) -> str:
        lines = [
            f"模板: {template_id}",
            f"Node ID: {payload.get('node_id', '')}",
            f"CWD: {payload.get('cwd', '') or '(项目根目录)'}",
            f"Command: {payload.get('executable', '')} {payload.get('args', '')}".strip(),
            "",
            "环境变量覆盖:",
            payload.get("env_text", "").strip() or "(无)",
            "",
            "输入文件:",
            payload.get("inputs_text", "").strip() or "(无)",
            "",
            "风险检查:",
        ]
        if risks:
            lines.extend([f"- {r}" for r in risks])
        else:
            lines.append("- 未发现明显风险。")
        return "\n".join(lines)

    def _set_run_payload(self, payload: dict[str, str]) -> None:
        self.node_id_var.set(payload.get("node_id", "").strip())
        self.exec_var.set(payload.get("executable", "").strip())
        self.cwd_var.set(payload.get("cwd", "").strip())
        self.args_text.delete("1.0", tk.END)
        self.args_text.insert("1.0", payload.get("args", "").strip())
        self.env_text.delete("1.0", tk.END)
        self.env_text.insert("1.0", payload.get("env_text", ""))
        self._set_inputs_text_value(payload.get("inputs_text", ""))

    @staticmethod
    def _normalize_payload_for_plugin(plugin_id: str, payload: dict[str, str]) -> dict[str, str]:
        if not plugin_id.endswith(".gui"):
            return payload
        normalized = dict(payload)
        normalized["cwd"] = ""
        normalized["inputs_text"] = ""
        return normalized

    def _clear_inputs_text(self) -> None:
        prev_state = str(self.inputs_text.cget("state"))
        if prev_state != str(tk.NORMAL):
            self.inputs_text.configure(state=tk.NORMAL)
        self.inputs_text.delete("1.0", tk.END)
        if prev_state != str(tk.NORMAL):
            self.inputs_text.configure(state=prev_state)

    def _set_inputs_text_value(self, value: str) -> None:
        prev_state = str(self.inputs_text.cget("state"))
        if prev_state != str(tk.NORMAL):
            self.inputs_text.configure(state=tk.NORMAL)
        self.inputs_text.delete("1.0", tk.END)
        if value:
            self.inputs_text.insert("1.0", value)
        if prev_state != str(tk.NORMAL):
            self.inputs_text.configure(state=prev_state)

    def _preflight_preview(self) -> dict[str, str] | None:
        if not self.current_project:
            messagebox.showwarning("未打开项目", "请先新建或打开项目。")
            return None

        template_id = self.template_var.get().strip()
        if (
            self.plugin_var.get().strip() == "tomo2d.shell"
            and template_id != TEMPLATE_CUSTOM
            and not self._apply_template_to_command()
        ):
            return None

        plugin_id = self.plugin_var.get().strip()
        if not plugin_id:
            messagebox.showwarning("未选择插件", "请先选择插件。")
            return None
        plugin = self.tool_registry.get(plugin_id)
        payload = self._normalize_payload_for_plugin(plugin_id, self._collect_run_payload())

        preview = tk.Toplevel(self.root)
        preview.title("运行前检查与命令预览（可编辑）")
        preview.geometry("1040x760")

        frm = ttk.Frame(preview, padding=8)
        frm.pack(fill=tk.BOTH, expand=True)

        row1 = ttk.Frame(frm)
        row1.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(row1, text="Node ID").grid(row=0, column=0, sticky=tk.W)
        node_var = tk.StringVar(value=payload.get("node_id", ""))
        ttk.Entry(row1, textvariable=node_var, width=20).grid(
            row=0, column=1, sticky=tk.W, padx=(6, 10)
        )
        ttk.Label(row1, text="可执行程序").grid(row=0, column=2, sticky=tk.W)
        exec_var = tk.StringVar(value=payload.get("executable", ""))
        ttk.Entry(row1, textvariable=exec_var, width=26).grid(
            row=0, column=3, sticky=tk.W, padx=(6, 10)
        )
        ttk.Label(row1, text="CWD").grid(row=0, column=4, sticky=tk.W)
        cwd_var = tk.StringVar(value=payload.get("cwd", ""))
        cwd_entry = ttk.Entry(row1, textvariable=cwd_var, width=28)
        cwd_entry.grid(
            row=0, column=5, sticky=tk.W, padx=(6, 0)
        )

        ttk.Label(frm, text="参数字符串").pack(anchor=tk.W)
        args_txt = tk.Text(frm, height=5, wrap=tk.WORD)
        args_txt.pack(fill=tk.X, pady=(0, 6))
        args_txt.insert("1.0", payload.get("args", ""))

        lower = ttk.Panedwindow(frm, orient=tk.HORIZONTAL)
        lower.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        left = ttk.Frame(lower)
        right = ttk.Frame(lower)
        lower.add(left, weight=1)
        lower.add(right, weight=1)

        ttk.Label(left, text="环境变量覆盖（KEY=VALUE）").pack(anchor=tk.W)
        env_txt = tk.Text(left, height=10, wrap=tk.NONE)
        env_txt.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        env_txt.insert("1.0", payload.get("env_text", ""))

        ttk.Label(right, text="输入文件（每行一个）").pack(anchor=tk.W)
        input_txt = tk.Text(right, height=10, wrap=tk.NONE)
        input_txt.pack(fill=tk.BOTH, expand=True, pady=(0, 6))
        input_txt.insert("1.0", payload.get("inputs_text", ""))
        if plugin_id.endswith(".gui"):
            cwd_entry.configure(state=tk.DISABLED)
            input_txt.configure(state=tk.DISABLED)

        ttk.Label(frm, text="预检结果").pack(anchor=tk.W)
        result_txt = tk.Text(frm, height=12, wrap=tk.WORD)
        result_txt.pack(fill=tk.BOTH, expand=True)

        selected: dict[str, str] | None = None

        def collect_dialog_payload() -> dict[str, str]:
            return {
                "node_id": node_var.get().strip(),
                "executable": exec_var.get().strip(),
                "args": args_txt.get("1.0", tk.END).strip(),
                "cwd": cwd_var.get().strip(),
                "env_text": env_txt.get("1.0", tk.END),
                "inputs_text": input_txt.get("1.0", tk.END),
            }

        def recompute(show_error: bool = True) -> tuple[bool, dict[str, str]]:
            p = collect_dialog_payload()
            p = self._normalize_payload_for_plugin(plugin_id, p)
            try:
                spec = plugin.build_spec(self.current_project.root, p)
            except PluginValidationError as exc:
                if show_error:
                    messagebox.showerror("节点配置错误", str(exc))
                result_txt.configure(state=tk.NORMAL)
                result_txt.delete("1.0", tk.END)
                result_txt.insert("1.0", f"配置错误:\n{exc}")
                result_txt.configure(state=tk.DISABLED)
                return False, p
            except Exception as exc:
                if show_error:
                    messagebox.showerror("插件错误", str(exc))
                result_txt.configure(state=tk.NORMAL)
                result_txt.delete("1.0", tk.END)
                result_txt.insert("1.0", f"插件错误:\n{exc}")
                result_txt.configure(state=tk.DISABLED)
                return False, p

            risks = self._build_risk_warnings(
                executable=str(spec.command[0]) if spec.command else "",
                args_text=p.get("args", ""),
                env=spec.env,
            )
            msg = self._render_preflight_message(template_id, p, risks)
            result_txt.configure(state=tk.NORMAL)
            result_txt.delete("1.0", tk.END)
            result_txt.insert("1.0", msg)
            result_txt.configure(state=tk.DISABLED)
            return True, p

        def on_confirm() -> None:
            nonlocal selected
            ok, p = recompute(show_error=True)
            if not ok:
                return
            selected = p
            preview.destroy()

        recompute(show_error=False)

        btns = ttk.Frame(frm)
        btns.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(btns, text="重新检查", command=lambda: recompute(show_error=True)).pack(
            side=tk.LEFT
        )
        ttk.Button(btns, text="取消", command=preview.destroy).pack(side=tk.RIGHT)
        ttk.Button(btns, text="确认并回填", command=on_confirm).pack(
            side=tk.RIGHT, padx=(0, 8)
        )
        preview.transient(self.root)
        preview.grab_set()
        self.root.wait_window(preview)
        return selected

    def _run_selected_node(self) -> None:
        if not self.current_project:
            messagebox.showwarning("未打开项目", "请先新建或打开项目。")
            return
        if self._single_run_active:
            messagebox.showinfo("任务进行中", "已有节点任务正在运行，请等待完成。")
            return

        plugin_id = self.plugin_var.get().strip()
        if not plugin_id:
            messagebox.showwarning("未选择插件", "请先选择插件。")
            return

        if self._should_auto_preflight(plugin_id):
            final_payload = self._preflight_preview()
            if final_payload is None:
                self.status_var.set("已取消运行。")
                return
        else:
            # 默认直跑：减少 GUI 节点（如 imodel）的额外弹窗干扰。
            final_payload = self._collect_run_payload()

        self._set_run_payload(final_payload)

        plugin = self.tool_registry.get(plugin_id)
        payload = self._normalize_payload_for_plugin(plugin_id, final_payload)

        try:
            spec = plugin.build_spec(self.current_project.root, payload)
        except PluginValidationError as exc:
            messagebox.showerror("节点配置错误", str(exc))
            return
        except Exception as exc:
            messagebox.showerror("插件错误", str(exc))
            return

        project = self.current_project
        self._single_run_active = True
        self._single_run_cancel_event = threading.Event()
        self.run_node_button.configure(state=tk.DISABLED)
        self.cancel_run_button.configure(state=tk.NORMAL)
        self._update_task_status_bar()
        self.status_var.set(f"运行中：{' '.join(spec.command)}")
        self.root.update_idletasks()

        worker = threading.Thread(
            target=self._run_selected_node_worker,
            args=(project, spec),
            daemon=True,
        )
        worker.start()

    @staticmethod
    def _should_auto_preflight(plugin_id: str) -> bool:
        """
        默认仅对高风险 shell 节点保留自动预检。
        可通过环境变量 PYAOBS_WORKBENCH_AUTO_PREFLIGHT=1 强制全局启用。
        """
        force = str(os.environ.get("PYAOBS_WORKBENCH_AUTO_PREFLIGHT", "")).strip().lower()
        if force in {"1", "true", "yes", "on"}:
            return True
        return plugin_id == "tomo2d.shell"

    def _run_selected_node_worker(self, project: ProjectContext, spec) -> None:
        result: dict[str, str] = {
            "run_id": "",
            "status": "failed",
            "return_code": "",
            "manifest_file": "",
            "error": "",
        }
        try:
            run = self.run_manager.run_command(
                project=project,
                node_id=spec.node_id,
                command=spec.command,
                params=spec.params,
                inputs=spec.inputs,
                env=spec.env,
                cwd=spec.cwd,
                cancel_event=self._single_run_cancel_event,
            )
            result["run_id"] = run.run_id
            result["manifest_file"] = str(run.manifest_file)
            try:
                manifest = json.loads(run.manifest_file.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
            result["status"] = str(manifest.get("status", "unknown"))
            result["return_code"] = str(manifest.get("return_code", ""))
        except Exception as exc:
            result["error"] = str(exc)
        self.root.after(0, lambda r=result: self._on_selected_node_finished(r))

    def _on_selected_node_finished(self, result: dict[str, str]) -> None:
        self._single_run_active = False
        self._single_run_cancel_event = None
        self.run_node_button.configure(state=tk.NORMAL)
        self.cancel_run_button.configure(state=tk.DISABLED)
        self._update_task_status_bar()
        self._refresh_project_tree()
        self._refresh_run_history()
        if result.get("error"):
            self.status_var.set("运行失败：执行异常")
            messagebox.showerror("运行异常", result["error"])
            return
        run_id = result.get("run_id", "")
        status = result.get("status", "unknown")
        code = result.get("return_code", "")
        if status == "success":
            self.status_var.set(f"运行完成：{run_id}（return_code={code}）")
            return
        if status == "cancelled":
            self.status_var.set(f"运行已取消：{run_id}")
            return
        self.status_var.set(f"运行失败：{run_id}（return_code={code}）")
        messagebox.showwarning(
            "运行失败",
            f"节点运行失败。\nrun_id: {run_id}\nmanifest: {result.get('manifest_file', '')}",
        )

    def _cancel_selected_node_run(self) -> None:
        if not self._single_run_active or self._single_run_cancel_event is None:
            messagebox.showinfo("无运行任务", "当前没有可取消的节点任务。")
            return
        self._single_run_cancel_event.set()
        self._update_task_status_bar()
        self.status_var.set("已请求取消当前节点任务，等待进程退出...")
        self.cancel_run_button.configure(state=tk.DISABLED)

    def _save_ui_state(self) -> None:
        if not self.current_project:
            messagebox.showwarning("未打开项目", "请先新建或打开项目。")
            return
        current_tab = ""
        try:
            current_tab = str(self.right_notebook.tab(self.right_notebook.select(), "text"))
        except Exception:
            current_tab = ""
        selected_run_id = self._selected_run_id()
        state = {
            "window_geometry": self.root.geometry(),
            "selected_project_path": str(self.current_project.root),
            "run_filter_status": self.run_filter_status_var.get(),
            "run_filter_node": self.run_filter_node_var.get(),
            "run_filter_keyword": self.run_filter_keyword_var.get(),
            "run_filter_failed_only": bool(self.run_filter_failed_only_var.get()),
            "run_batch_concurrency": self.run_batch_concurrency_var.get(),
            "last_batch_status_filter": self.last_batch_status_filter_var.get(),
            "right_tab_text": current_tab,
            "selected_run_id": selected_run_id or "",
            "batch_queue_sash_pos": self._get_batch_queue_sash_pos(),
            "batch_content_sash_pos": self._get_batch_content_sash_pos(),
            "history_vertical_sash_pos": self._get_history_vertical_sash_pos(),
            "workspace_history_sash_pos": self._get_workspace_history_sash_pos(),
            "runner_plugin_id": self.plugin_var.get().strip(),
            "runner_template_id": self.template_var.get().strip(),
            "runner_run_payload": self._collect_run_payload(),
            "runner_gui_quick_form": self._collect_gui_quick_form_state(),
        }
        sel = self.project_tree.selection()
        if sel:
            p = self._project_item_to_path.get(sel[0])
            if p is not None:
                state["selected_tree_path"] = str(p)

        self.state_store.save_ui_state(self.current_project, state)
        self.status_var.set("已保存界面状态到 state/ui_state.json")

    def _get_batch_queue_sash_pos(self) -> int:
        pane = getattr(self, "batch_queue_pane", None)
        if pane is None:
            return -1
        try:
            pos = int(pane.sashpos(0))
            return pos
        except Exception:
            return -1

    def _set_batch_queue_sash_pos(self, pos: int) -> None:
        pane = getattr(self, "batch_queue_pane", None)
        if pane is None:
            return
        if pos < 0:
            return
        try:
            pane.sashpos(0, pos)
        except Exception:
            pass

    def _get_batch_content_sash_pos(self) -> int:
        pane = getattr(self, "batch_content_pane", None)
        if pane is None:
            return -1
        try:
            return int(pane.sashpos(0))
        except Exception:
            return -1

    def _set_batch_content_sash_pos(self, pos: int) -> None:
        pane = getattr(self, "batch_content_pane", None)
        if pane is None or pos < 0:
            return
        try:
            pane.sashpos(0, pos)
        except Exception:
            pass

    def _get_history_vertical_sash_pos(self) -> list[int]:
        pane = getattr(self, "history_vertical_pane", None)
        if pane is None:
            return []
        try:
            return [int(pane.sashpos(0)), int(pane.sashpos(1))]
        except Exception:
            return []

    def _set_history_vertical_sash_pos(self, positions: Sequence[int]) -> None:
        pane = getattr(self, "history_vertical_pane", None)
        if pane is None:
            return
        if len(positions) < 2:
            return
        try:
            p0 = int(positions[0])
            p1 = int(positions[1])
        except (TypeError, ValueError):
            return
        try:
            if p0 >= 0:
                pane.sashpos(0, p0)
            if p1 >= 0:
                pane.sashpos(1, p1)
        except Exception:
            pass

    def _get_workspace_history_sash_pos(self) -> int:
        pane = getattr(self, "workspace_history_pane", None)
        if pane is None:
            return -1
        try:
            return int(pane.sashpos(0))
        except Exception:
            return -1

    def _set_workspace_history_sash_pos(self, pos: int) -> None:
        pane = getattr(self, "workspace_history_pane", None)
        if pane is None or pos < 0:
            return
        try:
            pane.sashpos(0, pos)
        except Exception:
            pass

    def _restore_ui_state(
        self, *, silent: bool = False, show_missing_notice: bool = True
    ) -> None:
        if not self.current_project:
            if not silent:
                messagebox.showwarning("未打开项目", "请先新建或打开项目。")
            return
        state = self.state_store.load_ui_state(self.current_project)
        if not state:
            if show_missing_notice:
                self.status_var.set("未找到已保存的界面状态。")
            return
        geom = state.get("window_geometry")
        if isinstance(geom, str) and geom:
            try:
                self.root.geometry(geom)
            except tk.TclError:
                pass

        selected = state.get("selected_tree_path")
        if isinstance(selected, str) and selected:
            self._select_tree_path(Path(selected))
        self._restore_filter_state_from_ui_state(state)
        if not silent:
            self.status_var.set("已恢复界面状态。")

    def _restore_filter_state_from_ui_state(self, state: dict) -> None:
        status_filter = str(state.get("run_filter_status", "全部") or "全部")
        node_filter = str(state.get("run_filter_node", "全部") or "全部")
        keyword = str(state.get("run_filter_keyword", "") or "")
        failed_only = bool(state.get("run_filter_failed_only", False))
        batch_concurrency = str(state.get("run_batch_concurrency", "1") or "1")
        last_batch_filter = str(state.get("last_batch_status_filter", "全部") or "全部")
        right_tab_text = str(state.get("right_tab_text", "") or "")
        selected_run_id = str(state.get("selected_run_id", "") or "")
        try:
            batch_queue_sash_pos = int(state.get("batch_queue_sash_pos", -1) or -1)
        except (TypeError, ValueError):
            batch_queue_sash_pos = -1
        try:
            batch_content_sash_pos = int(state.get("batch_content_sash_pos", -1) or -1)
        except (TypeError, ValueError):
            batch_content_sash_pos = -1
        try:
            workspace_history_sash_pos = int(state.get("workspace_history_sash_pos", -1) or -1)
        except (TypeError, ValueError):
            workspace_history_sash_pos = -1
        history_vertical_sash_pos_raw = state.get("history_vertical_sash_pos", [])
        history_vertical_sash_pos: list[int] = []
        if isinstance(history_vertical_sash_pos_raw, list):
            for val in history_vertical_sash_pos_raw[:2]:
                try:
                    history_vertical_sash_pos.append(int(val))
                except (TypeError, ValueError):
                    continue

        run_status_values = list(self.run_filter_status_combo.cget("values"))
        self.run_filter_status_var.set(
            status_filter if status_filter in run_status_values else "全部"
        )
        run_node_values = list(self.run_filter_node_combo.cget("values"))
        self.run_filter_node_var.set(node_filter if node_filter in run_node_values else "全部")
        self.run_filter_keyword_var.set(keyword)
        self.run_filter_failed_only_var.set(failed_only)

        batch_values = list(self.run_batch_concurrency_combo.cget("values"))
        self.run_batch_concurrency_var.set(
            batch_concurrency if batch_concurrency in batch_values else "1"
        )

        last_values = list(self.last_batch_status_filter_combo.cget("values"))
        self.last_batch_status_filter_var.set(
            last_batch_filter if last_batch_filter in last_values else "全部"
        )
        if right_tab_text:
            for tab_id in self.right_notebook.tabs():
                if str(self.right_notebook.tab(tab_id, "text")) == right_tab_text:
                    self.right_notebook.select(tab_id)
                    break

        self._apply_run_history_filters()
        if selected_run_id:
            found = self._select_run_id(selected_run_id)
            if found:
                self._refresh_selected_run_detail()
        self._refresh_last_batch_panel()
        if batch_queue_sash_pos >= 0:
            self.root.after(0, lambda p=batch_queue_sash_pos: self._set_batch_queue_sash_pos(p))
        if batch_content_sash_pos >= 0:
            self.root.after(0, lambda p=batch_content_sash_pos: self._set_batch_content_sash_pos(p))
        if len(history_vertical_sash_pos) >= 2:
            self.root.after(
                0,
                lambda ps=tuple(history_vertical_sash_pos): self._set_history_vertical_sash_pos(
                    ps
                ),
            )
        if workspace_history_sash_pos >= 0:
            self.root.after(
                0, lambda p=workspace_history_sash_pos: self._set_workspace_history_sash_pos(p)
            )
        self._restore_runner_state_from_ui_state(state)

    def _restore_runner_state_from_ui_state(self, state: dict) -> None:
        plugin_id = str(state.get("runner_plugin_id", "") or "").strip()
        if plugin_id == "data.shell":
            plugin_id = "data.gui"
        if plugin_id and plugin_id in self.tool_registry.ids():
            self.plugin_var.set(plugin_id)
            self._on_plugin_changed()

        template_id = str(state.get("runner_template_id", "") or "").strip()
        if template_id:
            self.template_var.set(template_id)
            self._on_template_changed()

        gui_state = state.get("runner_gui_quick_form", {})
        if isinstance(gui_state, dict):
            self._apply_gui_quick_form_state(gui_state)

        run_payload = state.get("runner_run_payload", {})
        if isinstance(run_payload, dict):
            cleaned: dict[str, str] = {}
            for k, v in run_payload.items():
                cleaned[str(k)] = "" if v is None else str(v)
            self._set_run_payload(cleaned)

    def _select_tree_path(self, path: Path) -> None:
        target = path.resolve()
        for item, p in self._project_item_to_path.items():
            if p.resolve() == target:
                cur = item
                while cur:
                    try:
                        self.project_tree.item(cur, open=True)
                    except Exception:
                        pass
                    cur = self.project_tree.parent(cur)
                self.project_tree.selection_set(item)
                self.project_tree.focus(item)
                self.project_tree.see(item)
                return

    def _on_tree_select(self, _event: tk.Event) -> None:
        sel = self.project_tree.selection()
        if not sel:
            return
        p = self._project_item_to_path.get(sel[0])
        if p is not None:
            self.status_var.set(f"选中：{p}")

    def _on_project_tree_double_click(self, event: tk.Event) -> None:
        item = self.project_tree.identify_row(event.y)
        if not item:
            return
        p = self._project_item_to_path.get(item)
        if p is None or not p.exists():
            messagebox.showwarning("路径无效", "双击目标路径不存在。请先刷新项目树。")
            return
        if p.is_dir():
            try:
                opened = bool(self.project_tree.item(item, "open"))
                self.project_tree.item(item, open=not opened)
            except Exception:
                pass
            self.status_var.set(f"目录：{p}")
            return
        try:
            self._open_path_with_system(p)
        except Exception as exc:
            messagebox.showerror("打开文件失败", str(exc))
            return
        self.status_var.set(f"已打开文件：{p}")

    def _has_active_tasks(self) -> bool:
        return self._single_run_active or self._batch_rerun_active

    def _on_close(self) -> None:
        self._cancel_live_log_follow()
        if self._has_active_tasks():
            choice = messagebox.askyesnocancel(
                "存在运行中任务",
                "检测到仍有运行中的任务。\n\n"
                "是：不退出，等待任务完成后再关闭。\n"
                "否：请求取消任务并立即退出。\n"
                "取消：返回工作台。",
            )
            if choice is None:
                return
            if choice is True:
                self.status_var.set("仍有任务运行中，已取消关闭请求。")
                return
            # choice is False: cancel tasks then close
            if self._single_run_active and self._single_run_cancel_event is not None:
                self._single_run_cancel_event.set()
            if self._batch_rerun_active and self._batch_rerun_cancel_event is not None:
                self._batch_rerun_cancel_event.set()
        try:
            if self.current_project is not None:
                self._save_ui_state()
        except Exception:
            pass
        self.root.destroy()

