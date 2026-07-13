#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TOMO2D 参数组织与调用 GUI（最小版）

功能:
1) 流程页签: gen_smesh、tt_forward、gen_damp、gen_vcorr、tt_inverse、stat_smesh、edit_smesh_HHB、pipeline、tx.in→tomo2d
2) 参数表单 + 预览 + 执行日志（可选将摘要同步写入 work_dir/tomo2d_gui.log；子进程 stdout/stderr 正文仅显示在界面）；顶部可绘制 smesh、**反演结果分析**（读 tt_inverse ``-L`` 日志绘图，见 ``tt_inverse_log_analysis``）；复用 visualization.show_model.GridModelVisualizer 嵌入 Matplotlib，工具栏自存图
3) 参数配置保存/加载（JSON）；亦可「加载配置」选择某次运行包内的 manifest.json，恢复界面（含 gui_profile）；加载后可提示将 work_dir 同步为 manifest 记录的项目目录，并合并 python_replay 中的 inputs/ 快照路径，便于 tt_forward 等找到输入
4) 多数文件路径相对 work_dir 存储（自动去掉工作目录绝对前缀）。gen_smesh/gen_damp/gen_vcorr 中 **v.in、反射文件** 因 C 源码 ``sscanf %[^/]`` 不能与层号之间的 ``/`` 混淆，**只保存文件名**且须放在 work_dir **根目录**；TomoAnd 拼命令时也会自动取 basename。work_dir、bin_path 仍为完整路径。

运行:
    python -m pyAOBS.modeling.tomo2d.tomo2d_gui

字体:
    Linux/WSL 若中文显示为方框，请安装 Noto CJK 等字体，或设置环境变量
    PYAOBS_GUI_FONT 为 tkinter.font.families() 中存在的字体名。

路径拖放（可选）:
    将文件拖到各路径输入框上可填入路径，需 pip install tkinterdnd2；启动时已尽量使用 TkinterDnD.Tk。
界面缩放:
    右下显示当前比例与「重置」，或使用全局 Ctrl+滚轮（Linux X11 另支持 Ctrl+滚轮键 4/5）。
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

from .tomand import TomoAnd, TomoCommandError
from .param_hints import get_param_hint
from .tx2tomo2d import convert_tx_in_to_tomo2d, parse_phase_set


def _format_resolved_cmdline(args) -> str:
    """将 subprocess 实际使用的 argv 列表格式化为可复制的 shell 风格命令行。"""
    if args is None:
        return ""
    if isinstance(args, str):
        return args
    seq = [str(a) for a in args]
    if not seq:
        return ""
    try:
        return shlex.join(seq)
    except (TypeError, ValueError):
        return " ".join(seq)


# gen_smesh.cc：-C<vpath>/<ilayer>、-F<layer>/<rpath> 用 sscanf，vpath/rpath 内不得含 '/'
_TOMO2D_ZELT_TOKEN_FILE_KEYS = frozenset({"gen.v_in", "gen.refl_file", "damp.v_in", "vcorr.v_in"})


class Tomo2DGui:
    def __init__(self, master: tk.Tk | None = None):
        self.root = master or tk.Tk()
        self.root.title("TOMO2D Workflow GUI")
        self.root.geometry("1200x780")
        self._gui_state_file = (
            Path(os.environ.get("PYAOBS_GUI_STATE_FILE", "").strip())
            if os.environ.get("PYAOBS_GUI_STATE_FILE", "").strip()
            else None
        )
        try:
            self.root.resizable(True, True)
        except tk.TclError:
            pass
        self._setup_ui_scaling()

        self.vars: dict[str, tk.Variable] = {}
        self.entries: dict[str, ttk.Entry] = {}
        self.top_entries: list[ttk.Entry] = []
        self.file_buttons: dict[str, ttk.Button] = {}
        self._widget_to_var_key: dict[int, str] = {}
        self._ui_zoom_mult = 1.0
        self._zoom_pct_var: tk.StringVar | None = None
        self._file_log_lock = threading.Lock()
        self._file_log_banner_paths: set[str] = set()
        self._run_elapsed_after_id: str | None = None
        self._run_elapsed_start: float | None = None
        self._run_elapsed_title: str = ""
        self._status_bar_label: tk.Label | None = None
        # Matplotlib 嵌入图窗（反演分析 / 绘制 smesh）；主窗口关闭时需先关图并 plt.close，否则进程常无法退出
        self._mpl_embed_toplevels: list[tk.Toplevel] = []

        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close_window)
        self._bind_ctrl_wheel_zoom()
        self._bind_dynamic_controls()
        self._on_gen_smesh_mode_change()
        self._on_damp_mode_change()
        self._on_vcorr_mode_change()
        self._on_stat_mode_change()
        self._on_edit_mode_change()
        self.root.after(120, self._restore_workbench_state)

    def _register_mpl_toplevel(self, win: tk.Toplevel) -> None:
        """记录 smesh / 反演分析等嵌入图窗，便于退出时统一销毁。"""
        self._mpl_embed_toplevels.append(win)

        def _on_destroy(event: tk.Event) -> None:
            if event.widget != win:
                return
            try:
                self._mpl_embed_toplevels.remove(win)
            except ValueError:
                pass

        win.bind("<Destroy>", _on_destroy, add="+")

    def _on_close_window(self) -> None:
        """关闭主窗口：释放所有 Matplotlib 图与子 Toplevel，结束 mainloop。"""
        try:
            self._save_workbench_state()
        except Exception:
            pass
        try:
            import matplotlib.pyplot as plt

            plt.close("all")
        except Exception:
            pass
        for w in list(self._mpl_embed_toplevels):
            try:
                w.destroy()
            except tk.TclError:
                pass
        self._mpl_embed_toplevels.clear()
        try:
            self.root.quit()
        except tk.TclError:
            pass

    def _load_workbench_state(self) -> dict:
        if self._gui_state_file is None or not self._gui_state_file.exists():
            return {}
        try:
            raw = json.loads(self._gui_state_file.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return {}
            state = raw.get("tomo2d_gui", {})
            return state if isinstance(state, dict) else {}
        except Exception:
            return {}

    def _save_workbench_state(self) -> None:
        if self._gui_state_file is None:
            return
        try:
            raw: dict = {}
            if self._gui_state_file.exists():
                loaded = json.loads(self._gui_state_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    raw = loaded
            raw["tomo2d_gui"] = {"profile": self._collect_gui_profile_dict()}
            self._gui_state_file.parent.mkdir(parents=True, exist_ok=True)
            self._gui_state_file.write_text(
                json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    def _restore_workbench_state(self) -> None:
        state = self._load_workbench_state()
        if not state:
            return
        profile = state.get("profile", {})
        if not isinstance(profile, dict) or not profile:
            return
        try:
            self._apply_loaded_profile_dict(dict(profile))
            self._log("已恢复上次 TOMO2D GUI 配置", write_file=False)
        except Exception as exc:
            self._log(f"恢复 TOMO2D GUI 配置失败: {exc}", write_file=False)
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _pick_ui_sans_family(self) -> str:
        """
        选择带中文覆盖的无衬线字体。Linux/WSL 常无 Microsoft YaHei，硬编码会导致缺字方框。
        可通过环境变量 PYAOBS_GUI_FONT 指定字体名（须出现在 tkinter.font.families() 中）。
        """
        override = (os.environ.get("PYAOBS_GUI_FONT") or "").strip()
        try:
            families = set(tkfont.families(self.root))
        except Exception:
            families = set(tkfont.families())
        if override and override in families:
            return override
        candidates = [
            "Microsoft YaHei UI",
            "Microsoft YaHei",
            "PingFang SC",
            "Hiragino Sans GB",
            "Noto Sans CJK SC",
            "Noto Sans CJK JP",
            "Source Han Sans SC",
            "Source Han Sans CN",
            "WenQuanYi Micro Hei",
            "WenQuanYi Zen Hei",
            "AR PL UMing CN",
            "Droid Sans Fallback",
        ]
        for name in candidates:
            if name in families:
                return name
        # 名称与候选略有不同时（发行版差异），按关键字在已注册族名中再扫一遍
        for name in sorted(families):
            nl = name.lower()
            if "noto" in nl and "cjk" in nl and "sans" in nl:
                return name
        for name in sorted(families):
            nl = name.lower()
            if "source han sans" in nl or "sourcehansans" in nl.replace(" ", ""):
                return name
        for name in sorted(families):
            nl = name.lower()
            if "wqy" in nl or "wenquanyi" in nl or "droid sans fallback" in nl:
                return name
        try:
            return str(tkfont.nametofont("TkDefaultFont").actual("family"))
        except Exception:
            return "TkDefaultFont"

    def _pick_mono_family(self) -> str:
        """预览/日志等区等宽字体；优先常见编程字体。"""
        try:
            families = set(tkfont.families(self.root))
        except Exception:
            families = set(tkfont.families())
        for name in (
            "Microsoft YaHei Mono",  # Windows：等宽 + CJK，混排时行高更稳
            "Cascadia Mono",
            "Consolas",
            "Cascadia Code",
            "Lucida Console",
            "Courier New",
        ):
            if name in families:
                return name
        return "Courier New"

    def _apply_terminal_text_theme(self, w: tk.Text, *, mono_size: int | None = None) -> None:
        """右侧预览/日志：暗底亮字；用 spacing* 加行距，减轻等宽+中文混排时顶底「贴边/裁切」感。"""
        try:
            if mono_size is None:
                try:
                    f = tkfont.Font(root=self.root, font=w.cget("font"))
                    mono_size = int(f.cget("size"))
                except Exception:
                    mono_size = max(13, getattr(self, "_base_font_size", 14))
            gap = max(4, int(round(mono_size * 0.48)))
            wrap_gap = max(2, int(round(mono_size * 0.25)))
            w.configure(
                background=self._term_text_bg,
                foreground=self._term_text_fg,
                insertbackground=self._term_insert,
                selectbackground=self._term_select_bg,
                selectforeground=self._term_select_fg,
                highlightthickness=0,
                borderwidth=0,
                relief=tk.FLAT,
                spacing1=gap,
                spacing2=wrap_gap,
                spacing3=gap,
            )
        except tk.TclError:
            pass

    def _bind_text_copy_helpers(self, w: tk.Text) -> None:
        """预览/日志等只读式 Text：支持 Ctrl+C（无选区则复制全部）、Ctrl+A、右键菜单。"""
        def copy_all() -> None:
            t = w.get("1.0", "end-1c")
            self.root.clipboard_clear()
            self.root.clipboard_append(t)

        def copy_sel() -> None:
            try:
                t = w.get(tk.SEL_FIRST, tk.SEL_LAST)
            except tk.TclError:
                return
            self.root.clipboard_clear()
            self.root.clipboard_append(t)

        def select_all() -> None:
            w.tag_add("sel", "1.0", "end-1c")
            w.mark_set(tk.INSERT, "end-1c")
            w.see(tk.INSERT)

        def on_ctrl_c(_event):
            if w.tag_ranges("sel"):
                t = w.get(tk.SEL_FIRST, tk.SEL_LAST)
            else:
                t = w.get("1.0", "end-1c")
            self.root.clipboard_clear()
            self.root.clipboard_append(t)
            return "break"

        def on_ctrl_a(_event):
            select_all()
            return "break"

        def on_menu(event):
            menu = tk.Menu(self.root, tearoff=0)
            menu.add_command(label="复制", command=copy_sel)
            menu.add_command(label="全选", command=select_all)
            menu.add_command(label="复制全部", command=copy_all)
            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                try:
                    menu.grab_release()
                except tk.TclError:
                    pass

        try:
            w.configure(exportselection=False)
        except tk.TclError:
            pass
        w.bind("<Control-c>", on_ctrl_c)
        w.bind("<Control-a>", on_ctrl_a)
        w.bind("<Button-3>", on_menu)
        if sys.platform == "darwin":
            w.bind("<Command-c>", on_ctrl_c)
            w.bind("<Command-a>", on_ctrl_a)

    def _setup_ui_scaling(self) -> None:
        """
        提升默认字体与高 DPI 缩放，避免界面文字过小。
        """
        # 让 Tk 根据屏幕 DPI 自动缩放（在高分屏上通常 > 1.0）
        try:
            dpi = self.root.winfo_fpixels("1i")
            scale = max(1.35, dpi / 96.0)
            self.root.tk.call("tk", "scaling", scale)
        except Exception:
            scale = 1.35
        self._tk_scaling_base = float(scale)

        # 统一放大 ttk 控件字体
        base_size = 14 if scale < 1.5 else 15
        self._base_font_size = base_size
        self._ui_font_family = self._pick_ui_sans_family()
        self._mono_font_family = self._pick_mono_family()
        self._entry_font = (self._ui_font_family, base_size + 2)
        style = ttk.Style(self.root)
        style.configure(".", font=(self._ui_font_family, base_size))
        style.configure("TEntry", font=(self._ui_font_family, base_size + 1))
        style.configure("TCombobox", font=(self._ui_font_family, base_size + 1))
        style.configure("TButton", font=(self._ui_font_family, base_size))
        style.configure("TLabel", font=(self._ui_font_family, base_size))
        style.configure("TCheckbutton", font=(self._ui_font_family, base_size))
        style.configure("Treeview", rowheight=34)
        try:
            # ttk.Combobox 下拉弹出列表（Listbox）的字体
            self.root.option_add("*TCombobox*Listbox.font", f"{self._ui_font_family} {base_size + 1}")
        except tk.TclError:
            pass

        # Notebook 标签适当再大一点，便于流程识别
        style.configure("TNotebook.Tab", font=(self._ui_font_family, base_size + 1))

        # 顶部「环境与快捷操作」条 vs 下方「流程 / 命令」主区：配色反差，便于扫视分区
        self._top_chrome_bg = "#c8daf0"
        self._top_chrome_fg = "#132433"
        self._work_region_bg = "#f4f5f7"
        self._chrome_rule_bg = "#7a9ab8"
        style.configure("TopChrome.TFrame", background=self._top_chrome_bg)
        style.configure(
            "TopChrome.TLabel",
            background=self._top_chrome_bg,
            foreground=self._top_chrome_fg,
        )
        style.configure(
            "TopChrome.TCheckbutton",
            background=self._top_chrome_bg,
            foreground=self._top_chrome_fg,
        )
        style.configure("TopChrome.TButton", padding=(10, 5))
        style.configure("WorkRegion.TFrame", background=self._work_region_bg)
        style.configure("WorkRegion.TPanedwindow", background=self._work_region_bg)
        try:
            self.root.configure(bg=self._work_region_bg)
        except tk.TclError:
            pass

        # 右侧「调用预览 / 执行日志」：终端感暗色面板（前景与背景对比约 12:1 量级）
        self._term_panel_bg = "#2a3140"
        self._term_text_bg = "#1e2433"
        self._term_text_fg = "#e8eaef"
        self._term_insert = "#7ec8ff"
        self._term_select_bg = "#3d5a80"
        self._term_select_fg = "#ffffff"
        style.configure("Terminal.TLabelframe", background=self._term_panel_bg)
        style.configure(
            "Terminal.TLabelframe.Label",
            background=self._term_panel_bg,
            foreground=self._term_text_fg,
        )

        # 底部状态栏：高对比，与主工作区明显区分
        self._status_bar_rule_bg = "#0ea5e9"
        self._status_bar_bg = "#0c4a6e"
        self._status_bar_fg = "#f8fafc"

    def _build_layout(self) -> None:
        top = ttk.Frame(self.root, padding=(10, 10, 10, 8), style="TopChrome.TFrame")
        top.pack(fill=tk.X)

        lb_bin = ttk.Label(
            top, text="bin_path（tomo2d程序安装目录）:", style="TopChrome.TLabel"
        )
        lb_bin.grid(row=0, column=0, sticky=tk.W)
        self.vars["bin_path"] = tk.StringVar(value=os.getenv("PYAOBS_TOMO2D_BIN") or os.getenv("TOMO2D_BIN") or "")
        en_bin = ttk.Entry(top, textvariable=self.vars["bin_path"], width=70, font=self._entry_font)
        en_bin.grid(row=0, column=1, sticky=tk.EW, padx=4)
        self.top_entries.append(en_bin)
        self._widget_to_var_key[id(en_bin)] = "bin_path"
        bt_bin = ttk.Button(
            top, text="浏览", command=lambda: self._pick_dir("bin_path"), style="TopChrome.TButton"
        )
        bt_bin.grid(row=0, column=2)
        self._bind_hint_widgets(lb_bin, "bin_path")
        self._bind_hint_widgets(en_bin, "bin_path")
        self._bind_hint_widgets(bt_bin, "bin_path")
        self._try_register_dnd(en_bin, "bin_path")

        lb_wd = ttk.Label(top, text="work_dir(工作目录):", style="TopChrome.TLabel")
        lb_wd.grid(row=1, column=0, sticky=tk.W)
        self.vars["work_dir"] = tk.StringVar(value=str(Path.cwd()))
        en_wd = ttk.Entry(top, textvariable=self.vars["work_dir"], width=70, font=self._entry_font)
        en_wd.grid(row=1, column=1, sticky=tk.EW, padx=4)
        self.top_entries.append(en_wd)
        self._widget_to_var_key[id(en_wd)] = "work_dir"
        bt_wd = ttk.Button(
            top, text="浏览", command=lambda: self._pick_dir("work_dir"), style="TopChrome.TButton"
        )
        bt_wd.grid(row=1, column=2)
        self._bind_hint_widgets(lb_wd, "work_dir")
        self._bind_hint_widgets(en_wd, "work_dir")
        self._bind_hint_widgets(bt_wd, "work_dir")
        self._try_register_dnd(en_wd, "work_dir")

        self.vars["gui.write_file_log"] = tk.BooleanVar(value=True)
        cb_flog = ttk.Checkbutton(
            top,
            text="在工作目录写入运行摘要 tomo2d_gui.log（不含子进程 stdout/stderr 正文）",
            variable=self.vars["gui.write_file_log"],
            style="TopChrome.TCheckbutton",
        )
        cb_flog.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(4, 0))
        self._bind_hint_widgets(cb_flog, "gui.write_file_log")

        bt_inv_ana = ttk.Button(
            top,
            text="反演结果分析…",
            command=self._open_tt_inverse_analysis_dialog,
            style="TopChrome.TButton",
        )
        bt_inv_ana.grid(row=2, column=2, sticky=tk.W, padx=(12, 0), pady=(4, 0))
        self._bind_hint_widgets(bt_inv_ana, "btn.tt_inverse_analysis")

        bt_plot_smesh = ttk.Button(
            top,
            text="绘制 smesh 速度模型…",
            command=self.plot_smesh_velocity,
            style="TopChrome.TButton",
        )
        bt_plot_smesh.grid(row=2, column=3, sticky=tk.W, padx=(8, 0), pady=(4, 0))
        self._bind_hint_widgets(bt_plot_smesh, "btn.plot_smesh")

        bt_help = ttk.Button(
            top,
            text="程序帮助…",
            command=self._open_program_help_dialog,
            style="TopChrome.TButton",
        )
        bt_help.grid(row=2, column=4, sticky=tk.W, padx=(8, 0), pady=(4, 0))
        self._bind_hint_widgets(bt_help, "btn.program_help")

        lb_cmap = ttk.Label(top, text="smesh 色标（可选）:", style="TopChrome.TLabel")
        lb_cmap.grid(row=3, column=0, sticky=tk.W, pady=(6, 0))
        self.vars["gui.plot_smesh_cmap"] = tk.StringVar(value="")
        en_smesh_cmap = ttk.Entry(
            top, textvariable=self.vars["gui.plot_smesh_cmap"], width=48, font=self._entry_font
        )
        en_smesh_cmap.grid(row=3, column=1, sticky=tk.EW, padx=4, pady=(6, 0))
        self.top_entries.append(en_smesh_cmap)
        self._widget_to_var_key[id(en_smesh_cmap)] = "gui.plot_smesh_cmap"
        bt_smesh_cmap = ttk.Button(
            top, text="浏览.cpt", command=self._pick_plot_smesh_cmap, style="TopChrome.TButton"
        )
        bt_smesh_cmap.grid(row=3, column=2, sticky=tk.W, padx=(0, 4), pady=(6, 0))
        self._bind_hint_widgets(lb_cmap, "gui.plot_smesh_cmap")
        self._bind_hint_widgets(en_smesh_cmap, "gui.plot_smesh_cmap")
        self._bind_hint_widgets(bt_smesh_cmap, "gui.plot_smesh_cmap")
        self._try_register_dnd(en_smesh_cmap, "gui.plot_smesh_cmap")

        bt_save = ttk.Button(
            top, text="保存配置", command=self._save_profile, style="TopChrome.TButton"
        )
        bt_save.grid(row=0, column=3, padx=10)
        bt_load = ttk.Button(
            top, text="加载配置", command=self._load_profile, style="TopChrome.TButton"
        )
        bt_load.grid(row=1, column=3, padx=10)
        self._bind_hint_widgets(bt_save, "btn.save_profile")
        self._bind_hint_widgets(bt_load, "btn.load_profile")

        top.columnconfigure(1, weight=1)

        chrome_rule = tk.Frame(self.root, height=2, bg=self._chrome_rule_bg, borderwidth=0)
        chrome_rule.pack(fill=tk.X)

        # 先占位底部状态栏再 pack 中间区；顶边色条 + 深色底 + 加粗 tk.Label，保证运行时间等一眼可见
        self.status_var = tk.StringVar(value="就绪")
        status_strip = tk.Frame(self.root, bg=self._status_bar_rule_bg, bd=0, highlightthickness=0)
        status_strip.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Frame(status_strip, height=5, bg=self._status_bar_rule_bg, bd=0).pack(fill=tk.X)
        status_inner = tk.Frame(status_strip, bg=self._status_bar_bg, bd=0, highlightthickness=0)
        status_inner.pack(fill=tk.X)
        status_inner.columnconfigure(0, weight=1)
        sb_pt = max(12, self._base_font_size + 2)
        self._status_bar_label = tk.Label(
            status_inner,
            textvariable=self.status_var,
            anchor=tk.W,
            justify=tk.LEFT,
            bg=self._status_bar_bg,
            fg=self._status_bar_fg,
            font=(self._ui_font_family, sb_pt, "bold"),
            padx=14,
            pady=12,
            bd=0,
        )
        self._status_bar_label.grid(row=0, column=0, sticky=tk.EW)
        self._bind_hint_widgets(self._status_bar_label, "ui.status_bar")
        sg = ttk.Sizegrip(status_inner)
        sg.grid(row=0, column=1, sticky=tk.SE, padx=(0, 2), pady=(0, 2))
        corner = tk.Label(
            status_inner,
            text="◢",
            bg=self._status_bar_bg,
            fg=self._status_bar_fg,
            font=(self._ui_font_family, max(9, sb_pt - 4)),
        )
        corner.place(relx=1.0, rely=1.0, anchor=tk.SE, x=-3, y=-1)
        self._bind_hint_widgets(sg, "ui.sizegrip")
        self._bind_hint_widgets(corner, "ui.sizegrip")
        try:
            ttk.Style(self.root).configure("TSizegrip", background=self._status_bar_bg)
        except tk.TclError:
            pass

        center = ttk.Frame(self.root, style="WorkRegion.TFrame")
        center.pack(fill=tk.BOTH, expand=True, padx=8, pady=(6, 4))

        left = ttk.Frame(center, style="WorkRegion.TFrame")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(center, style="WorkRegion.TFrame")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 左侧改为上下可调分栏：上方参数页签，下方参数说明
        left_pane = ttk.Panedwindow(left, orient=tk.VERTICAL, style="WorkRegion.TPanedwindow")
        left_pane.pack(fill=tk.BOTH, expand=True)

        nb_wrap = ttk.Frame(left_pane)
        self.notebook = ttk.Notebook(nb_wrap)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.gen_tab = ttk.Frame(self.notebook)
        self.fwd_tab = ttk.Frame(self.notebook)
        self.damp_tab = ttk.Frame(self.notebook)
        self.vcorr_tab = ttk.Frame(self.notebook)
        self.inv_tab = ttk.Frame(self.notebook)
        self.stat_tab = ttk.Frame(self.notebook)
        self.edit_tab = ttk.Frame(self.notebook)
        self.pipe_tab = ttk.Frame(self.notebook)
        self.tx_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.gen_tab, text="1) gen_smesh")
        self.notebook.add(self.fwd_tab, text="2) tt_forward")
        self.notebook.add(self.damp_tab, text="3) gen_damp")
        self.notebook.add(self.vcorr_tab, text="4) gen_vcorr")
        self.notebook.add(self.inv_tab, text="5) tt_inverse")
        self.notebook.add(self.stat_tab, text="6) stat_smesh")
        self.notebook.add(self.edit_tab, text="7) edit_smesh_HHB")
        self.notebook.add(self.pipe_tab, text="8) pipeline")
        self.notebook.add(self.tx_tab, text="9) tx.in→tomo2d")

        # 所有参数页签统一使用可滚动表单，避免字体放大或参数较多时底部控件不可见
        self.gen_frame = self._make_scrollable_form_tab(self.gen_tab)
        self.fwd_frame = self._make_scrollable_form_tab(self.fwd_tab)
        self.damp_frame = self._make_scrollable_form_tab(self.damp_tab)
        self.vcorr_frame = self._make_scrollable_form_tab(self.vcorr_tab)
        self.inv_frame = self._make_scrollable_form_tab(self.inv_tab)
        self.stat_frame = self._make_scrollable_form_tab(self.stat_tab)
        self.edit_frame = self._make_scrollable_form_tab(self.edit_tab)
        self.pipe_frame = self._make_scrollable_form_tab(self.pipe_tab)
        self.tx_frame = self._make_scrollable_form_tab(self.tx_tab)

        self._build_gen_smesh_tab()
        self._build_tt_forward_tab()
        self._build_gen_damp_tab()
        self._build_gen_vcorr_tab()
        self._build_tt_inverse_tab()
        self._build_stat_smesh_tab()
        self._build_edit_smesh_tab()
        self._build_pipeline_tab()
        self._build_tx_convert_tab()

        hint_frame = ttk.LabelFrame(left_pane, text="参数说明（焦点进入输入框或鼠标悬停）", padding=8)
        hint_size = max(11, getattr(self, "_base_font_size", 13) - 1)
        self.hint_text = tk.Text(
            hint_frame,
            height=5,
            wrap=tk.WORD,
            font=(getattr(self, "_ui_font_family", self._pick_ui_sans_family()), hint_size),
        )
        self.hint_text.pack(fill=tk.BOTH, expand=True)
        self.hint_text.configure(state=tk.DISABLED)
        left_pane.add(nb_wrap, weight=4)
        left_pane.add(hint_frame, weight=1)

        self.notebook.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed)

        preview_box = ttk.LabelFrame(right, text="调用预览", padding=8, style="Terminal.TLabelframe")
        preview_box.pack(fill=tk.BOTH, expand=True)
        preview_box.columnconfigure(0, weight=1)
        preview_box.rowconfigure(0, weight=1)
        mono_pt = max(13, self._base_font_size)
        self.preview_text = tk.Text(
            preview_box,
            height=10,
            wrap=tk.WORD,
            font=(self._mono_font_family, mono_pt),
        )
        self._apply_terminal_text_theme(self.preview_text, mono_size=mono_pt)
        preview_vsb = ttk.Scrollbar(preview_box, orient=tk.VERTICAL, command=self.preview_text.yview)
        self.preview_text.configure(yscrollcommand=preview_vsb.set)
        self.preview_text.grid(row=0, column=0, sticky=tk.NSEW)
        preview_vsb.grid(row=0, column=1, sticky=tk.NS)
        self._bind_text_copy_helpers(self.preview_text)

        log_box = ttk.LabelFrame(right, text="执行日志", padding=8, style="Terminal.TLabelframe")
        log_box.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        log_box.columnconfigure(0, weight=1)
        log_box.rowconfigure(0, weight=1)
        self.log_text = tk.Text(
            log_box,
            height=8,
            wrap=tk.WORD,
            font=(self._mono_font_family, mono_pt),
        )
        self._apply_terminal_text_theme(self.log_text, mono_size=mono_pt)
        log_vsb = ttk.Scrollbar(log_box, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_vsb.set)
        self.log_text.grid(row=0, column=0, sticky=tk.NSEW)
        log_vsb.grid(row=0, column=1, sticky=tk.NS)
        self._bind_text_copy_helpers(self.log_text)

        # 按用户要求隐藏“界面缩放”面板；仍保留 Ctrl+滚轮缩放能力

        self._on_notebook_tab_changed()

    def _make_scrollable_form_tab(self, tab_parent: ttk.Frame) -> ttk.Frame:
        host = ttk.Frame(tab_parent)
        host.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(host, highlightthickness=0, borderwidth=0)
        vbar = ttk.Scrollbar(host, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas, padding=10)
        win_id = canvas.create_window((0, 0), window=inner, anchor=tk.NW)

        def on_inner_configure(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event):
            try:
                canvas.itemconfigure(win_id, width=event.width)
            except tk.TclError:
                pass

        def on_mousewheel(event):
            # Ctrl+滚轮保留给全局缩放逻辑
            if int(getattr(event, "state", 0)) & 0x0004:
                return None
            delta = int(getattr(event, "delta", 0) or 0)
            if delta != 0:
                step = -1 if delta > 0 else 1
                canvas.yview_scroll(step, "units")
                return "break"
            return None

        def on_x11_up(_event):
            canvas.yview_scroll(-1, "units")
            return "break"

        def on_x11_down(_event):
            canvas.yview_scroll(1, "units")
            return "break"

        def bind_wheel(_event=None):
            canvas.bind_all("<MouseWheel>", on_mousewheel)
            canvas.bind_all("<Button-4>", on_x11_up)
            canvas.bind_all("<Button-5>", on_x11_down)

        def unbind_wheel(_event=None):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        inner.bind("<Configure>", on_inner_configure)
        canvas.bind("<Configure>", on_canvas_configure)
        host.bind("<Enter>", bind_wheel)
        host.bind("<Leave>", unbind_wheel)
        return inner

    def _set_hint(self, key: str) -> None:
        text = get_param_hint(key)
        self.hint_text.configure(state=tk.NORMAL)
        self.hint_text.delete("1.0", tk.END)
        self.hint_text.insert(tk.END, text)
        self.hint_text.configure(state=tk.DISABLED)

    def _bind_hint_widgets(self, widget: tk.Misc, key: str) -> None:
        def show(_event=None):
            self._set_hint(key)

        widget.bind("<Enter>", show)
        widget.bind("<FocusIn>", show)

    def _toggle_inv_grav_section(self) -> None:
        """tt_inverse 页：联合重力 LabelFrame 内字段展开/折叠。"""
        self._inv_grav_expanded = not self._inv_grav_expanded
        if self._inv_grav_expanded:
            self._inv_grav_content.grid(row=1, column=0, columnspan=3, sticky=tk.EW)
            self._inv_grav_toggle_btn.configure(text="▼ 收起联合重力参数")
        else:
            self._inv_grav_content.grid_remove()
            self._inv_grav_toggle_btn.configure(text="▶ 展开联合重力参数")

    def _on_notebook_tab_changed(self, _event=None) -> None:
        try:
            idx = self.notebook.index(self.notebook.select())
        except tk.TclError:
            idx = 0
        tab_keys = [
            "tab.gen_smesh",
            "tab.tt_forward",
            "tab.gen_damp",
            "tab.gen_vcorr",
            "tab.tt_inverse",
            "tab.stat_smesh",
            "tab.edit_smesh",
            "tab.pipeline",
            "tab.tx_convert",
        ]
        self._set_hint(tab_keys[idx] if idx < len(tab_keys) else tab_keys[0])

    def _default_help_section_from_active_tab(self) -> str:
        """根据当前页签给出程序帮助默认章节。"""
        try:
            idx = self.notebook.index(self.notebook.select())
        except Exception:
            idx = 0
        tab_to_help = {
            0: "gen_smesh",
            1: "tt_forward",
            2: "gen_damp",
            3: "gen_vcorr",
            4: "tt_inverse",
            5: "stat_smesh",
            6: "edit_smesh / edit_smesh_HHB",
            7: "Python 封装总览",
            8: "Python 封装总览",
        }
        return tab_to_help.get(idx, "Python 封装总览")

    def _setup_zoom_bar(self, parent: ttk.Frame) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X)
        self._zoom_pct_var = tk.StringVar(value="100%")
        ttk.Label(row, text="当前缩放", width=8).pack(side=tk.LEFT)
        ttk.Label(row, textvariable=self._zoom_pct_var, width=6, anchor=tk.CENTER).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(row, text="重置", command=self._reset_ui_zoom).pack(side=tk.LEFT, padx=(10, 0))
        self._bind_hint_widgets(parent, "ui.scale")
        for w in row.winfo_children():
            self._bind_hint_widgets(w, "ui.scale")

    def _apply_ui_zoom(self) -> None:
        try:
            self.root.tk.call("tk", "scaling", self._tk_scaling_base * self._ui_zoom_mult)
        except tk.TclError:
            pass

        # 在 ttk 显式字体模式下，tk scaling 不会自动改变控件字体，需手动重设
        zbase = max(11, int(round(self._base_font_size * self._ui_zoom_mult)))
        self._entry_font = (self._ui_font_family, zbase + 2)
        style = ttk.Style(self.root)
        style.configure(".", font=(self._ui_font_family, zbase))
        style.configure("TEntry", font=(self._ui_font_family, zbase + 1))
        style.configure("TCombobox", font=(self._ui_font_family, zbase + 1))
        style.configure("TButton", font=(self._ui_font_family, zbase))
        style.configure("TLabel", font=(self._ui_font_family, zbase))
        style.configure("TCheckbutton", font=(self._ui_font_family, zbase))
        style.configure("TNotebook.Tab", font=(self._ui_font_family, zbase + 1))
        style.configure("Treeview", rowheight=max(26, int(round(34 * self._ui_zoom_mult))))
        try:
            self.root.option_add("*TCombobox*Listbox.font", f"{self._ui_font_family} {zbase + 1}")
        except tk.TclError:
            pass

        # 已创建控件显式 font 也要同步
        for e in self.entries.values():
            try:
                e.configure(font=self._entry_font)
            except tk.TclError:
                pass
        for e in self.top_entries:
            try:
                e.configure(font=self._entry_font)
            except tk.TclError:
                pass
        try:
            hint_size = max(11, zbase - 1)
            self.hint_text.configure(font=(self._ui_font_family, hint_size))
            mono_size = max(13, zbase)
            self.preview_text.configure(font=(self._mono_font_family, mono_size))
            self.log_text.configure(font=(self._mono_font_family, mono_size))
            self._apply_terminal_text_theme(self.preview_text, mono_size=mono_size)
            self._apply_terminal_text_theme(self.log_text, mono_size=mono_size)
        except Exception:
            pass

        try:
            lb = getattr(self, "_status_bar_label", None)
            if lb is not None:
                sb_pt = max(12, int(round(self._base_font_size * self._ui_zoom_mult)) + 2)
                lb.configure(font=(self._ui_font_family, sb_pt, "bold"))
        except Exception:
            pass

        pct = int(round(self._ui_zoom_mult * 100))
        if self._zoom_pct_var is not None:
            self._zoom_pct_var.set(f"{pct}%")

    def _nudge_ui_zoom(self, delta: float) -> None:
        self._ui_zoom_mult = max(0.75, min(1.75, self._ui_zoom_mult + delta))
        self._apply_ui_zoom()
        self.status_var.set(f"界面缩放: {int(round(self._ui_zoom_mult * 100))}%")

    def _reset_ui_zoom(self) -> None:
        self._ui_zoom_mult = 1.0
        self._apply_ui_zoom()
        self.status_var.set("界面缩放已重置为 100%")

    def _bind_ctrl_wheel_zoom(self) -> None:
        """全局 Ctrl+滚轮：与 _nudge_ui_zoom 步长一致（Linux X11 另绑 Button-4/5）。"""

        def nudge_from_wheel_delta(delta: int) -> None:
            if delta == 0:
                return
            if sys.platform == "darwin":
                step = 0.1 if delta > 0 else -0.1
            else:
                step = (delta / 120.0) * 0.1
            self._nudge_ui_zoom(step)

        def on_ctrl_mousewheel(event):
            d = int(getattr(event, "delta", 0) or 0)
            if d:
                nudge_from_wheel_delta(d)
                return "break"
            return None

        self.root.bind_all("<Control-MouseWheel>", on_ctrl_mousewheel)

        def on_x11_up(_event):
            self._nudge_ui_zoom(0.1)
            return "break"

        def on_x11_down(_event):
            self._nudge_ui_zoom(-0.1)
            return "break"

        self.root.bind_all("<Control-Button-4>", on_x11_up)
        self.root.bind_all("<Control-Button-5>", on_x11_down)

    def _parse_dnd_paths(self, data: str) -> list[str]:
        paths: list[str] = []
        try:
            parts = self.root.tk.splitlist(data)
        except tk.TclError:
            parts = [data]
        for p in parts:
            p = str(p).strip()
            if len(p) >= 2 and p[0] == "{" and p[-1] == "}":
                p = p[1:-1]
            if p:
                paths.append(p)
        return paths

    def _try_register_dnd(self, widget: tk.Misc, var_key: str) -> None:
        try:
            from tkinterdnd2 import DND_FILES
        except ImportError:
            return

        def on_drop(event):
            paths = self._parse_dnd_paths(event.data)
            if var_key and paths:
                first = paths[0]
                if var_key in self.file_buttons:
                    first = self._to_workdir_relative(first, show_warn_outside=True)
                    first = self._postprocess_zelt_file_var(var_key, first)
                self.vars[var_key].set(first)
                self._log(f"拖放填入 [{var_key}]: {first}")
                self.status_var.set(f"已填入 {var_key}")

        try:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", on_drop)
        except tk.TclError:
            pass

    def _add_standard_grid_block(self, parent: ttk.Frame, row: int, pfx: str) -> int:
        """与 gen_smesh / gen_damp / gen_vcorr 共用的网格块：nx…dx 与三个可选文件列。"""
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(2, weight=0)
        specs: list[tuple[str, str, str, str | None]] = [
            ("nx", "nx (-N)", "", None),
            ("nz", "nz (-N)", "", None),
            ("xmax", "xmax (-D)", "", None),
            ("zmax", "zmax (-D)", "", None),
            ("x_file", "x_file (-X)", "", "open"),
            ("z_file", "z_file (-Z)", "", "open"),
            ("topo_file", "topo_file (-T)", "", "open"),
            ("dx", "dx (-E)", "", None),
        ]
        for short, lbl, default, fmode in specs:
            key = f"{pfx}{short}"
            if fmode:
                row = self._entry_file_row(parent, row, lbl, key, default, save_as=False)
            else:
                row = self._entry_row(parent, row, lbl, key, default)
        return row

    def _apply_grid_file_state(self, pfx: str, grid_opt: str) -> None:
        if grid_opt == "uniform":
            self._set_enabled([f"{pfx}nx", f"{pfx}nz", f"{pfx}xmax", f"{pfx}zmax"], True)
            self._set_enabled([f"{pfx}x_file", f"{pfx}z_file", f"{pfx}topo_file", f"{pfx}dx"], False)
        elif grid_opt == "variable":
            self._set_enabled([f"{pfx}nx", f"{pfx}nz", f"{pfx}xmax", f"{pfx}zmax"], False)
            self._set_enabled([f"{pfx}x_file", f"{pfx}z_file", f"{pfx}topo_file"], True)
            self._set_enabled([f"{pfx}dx"], False)
        else:
            self._set_enabled([f"{pfx}nx", f"{pfx}nz", f"{pfx}xmax", f"{pfx}zmax"], False)
            self._set_enabled([f"{pfx}x_file", f"{pfx}topo_file"], False)
            self._set_enabled([f"{pfx}z_file", f"{pfx}dx"], True)

    def _merge_grid_vars_into(self, kwargs: dict, pfx: str, *, grid_opt: str) -> None:
        """仅合并当前 grid_opt 下界面启用的网格项，忽略灰显框里的残留文本。"""
        keys_by_grid = {
            "uniform": ("nx", "nz", "xmax", "zmax"),
            "variable": ("x_file", "z_file", "topo_file"),
            "zelt": ("dx", "z_file"),
        }
        for key in keys_by_grid.get(grid_opt, ()):
            vk = f"{pfx}{key}"
            if vk not in self.vars:
                continue
            raw = self.vars[vk].get().strip()
            if not raw:
                continue
            if key.endswith("_file"):
                kwargs[key] = raw
            else:
                kwargs[key] = self._to_number(raw)

    def _build_gen_smesh_tab(self) -> None:
        row = 0
        self.vars["gen.vel_opt"] = tk.StringVar(value="uniform")
        self.vars["gen.grid_opt"] = tk.StringVar(value="uniform")

        self.gen_frame.columnconfigure(1, weight=1)
        self.gen_frame.columnconfigure(2, weight=0)

        row = self._combo_row(self.gen_frame, row, "vel_opt (-A/-B | -C)", "gen.vel_opt", ["uniform", "zelt"])
        row = self._combo_row(
            self.gen_frame, row, "grid_opt (-N/-D | -X/-Z/-T | -E/-Z)", "gen.grid_opt", ["uniform", "variable", "zelt"]
        )

        gen_vel: list[tuple[str, str, str, str | None]] = [
            ("gen.v0", "v0 (-A)", "", None),
            ("gen.gradient", "gradient (-B)", "", None),
            ("gen.v_in", "v_in (-C 前半)", "", "open"),
            ("gen.ilayer", "ilayer (-C 后半)", "", None),
            ("gen.refl_layer", "refl_layer (-F 前半)", "", None),
            ("gen.refl_file", "refl_file (-F 后半)", "", "open"),
        ]
        for key, label, default, fmode in gen_vel:
            if fmode:
                row = self._entry_file_row(self.gen_frame, row, label, key, default, save_as=False)
            else:
                row = self._entry_row(self.gen_frame, row, label, key, default)

        row = self._add_standard_grid_block(self.gen_frame, row, "gen.")

        for key, label, default, fmode in [
            ("gen.water_col", "water_col (-W)", "", None),
            ("gen.v_water", "v_water (-Q)", "1.5", None),
            ("gen.v_air", "v_air (-R)", "0.33", None),
        ]:
            row = self._entry_row(self.gen_frame, row, label, key, default)

        row = self._entry_file_row(
            self.gen_frame, row, "zelt_dump (-d)", "gen.zelt_dump", "", save_as=True
        )
        row = self._entry_file_row(
            self.gen_frame,
            row,
            "smesh 输出文件（stdout → 磁盘，相对 work_dir）",
            "gen.smesh_out",
            "",
            save_as=True,
        )

        btns = ttk.Frame(self.gen_frame)
        btns.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))
        b_prev = ttk.Button(btns, text="预览 gen_smesh", command=self.preview_gen_smesh)
        b_prev.pack(side=tk.LEFT, padx=(0, 6))
        b_run = ttk.Button(btns, text="运行 gen_smesh", command=self.run_gen_smesh)
        b_run.pack(side=tk.LEFT)
        self._bind_hint_widgets(b_prev, "btn.preview_gen_smesh")
        self._bind_hint_widgets(b_run, "btn.run_gen_smesh")

    def _build_tt_forward_tab(self) -> None:
        row = 0
        self.fwd_frame.columnconfigure(1, weight=1)
        self.fwd_frame.columnconfigure(2, weight=0)

        # (key, label, default, file_mode)  file_mode: None | "open" | "save"
        fwd_fields: list[tuple[str, str, str, str | None]] = [
            ("fwd.smesh", "smesh (-M)", "", "open"),
            ("fwd.geom", "geom (-G)", "", "open"),
            ("fwd.refl_file", "refl_file (-F)", "", "open"),
            ("fwd.xorder", "xorder (-N)", "4", None),
            ("fwd.zorder", "zorder (-N)", "4", None),
            ("fwd.clen", "clen (-N)", "0.8", None),
            ("fwd.nintp", "nintp (-N)", "8", None),
            ("fwd.bend_cg_tol", "bend_cg_tol (-N)", "1e-4", None),
            ("fwd.bend_br_tol", "bend_br_tol (-N)", "1e-5", None),
            ("fwd.vred", "vred (-r)", "", None),
            ("fwd.out_ttime", "out_ttime (-T)", "", "save"),
            ("fwd.out_ray", "out_ray (-R)", "", "save"),
            ("fwd.out_elements", "out_elements (-E)", "", "save"),
            ("fwd.out_obs_ttime", "out_obs_ttime (-O)", "", "save"),
            ("fwd.out_source", "out_source (-S)", "", "save"),
            ("fwd.out_vgrid", "out_vgrid (-I)", "", "save"),
            ("fwd.out_diff", "out_diff (-D)", "", "save"),
            ("fwd.clock_file", "clock_file (-C)", "", "open"),
            ("fwd.sub_west", "west (-i)", "", None),
            ("fwd.sub_east", "east (-i)", "", None),
            ("fwd.sub_south", "south (-i)", "", None),
            ("fwd.sub_north", "north (-i)", "", None),
            ("fwd.sub_dx", "dx (-i)", "", None),
            ("fwd.sub_dz", "dz (-i)", "", None),
        ]
        for key, label, default, fmode in fwd_fields:
            if fmode == "open":
                row = self._entry_file_row(self.fwd_frame, row, label, key, default, save_as=False)
            elif fmode == "save":
                row = self._entry_file_row(self.fwd_frame, row, label, key, default, save_as=True)
            else:
                row = self._entry_row(self.fwd_frame, row, label, key, default)

        self.vars["fwd.verbose_level"] = tk.StringVar(value="")
        vf = ttk.Frame(self.fwd_frame)
        vf.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(2, 2))
        lb_fvl = ttk.Label(vf, text="verbose_level (-V)")
        lb_fvl.pack(side=tk.LEFT, padx=(0, 4))
        en_fvl = ttk.Entry(vf, textvariable=self.vars["fwd.verbose_level"], width=8, font=self._entry_font)
        en_fvl.pack(side=tk.LEFT, padx=(0, 10))
        self.entries["fwd.verbose_level"] = en_fvl
        self._widget_to_var_key[id(en_fvl)] = "fwd.verbose_level"
        lb_fvhint = ttk.Label(vf, text="（大于0时自动启用 -V）")
        lb_fvhint.pack(side=tk.LEFT, padx=(0, 8))
        self._bind_hint_widgets(lb_fvl, "fwd.verbose_level")
        self._bind_hint_widgets(en_fvl, "fwd.verbose_level")
        self._bind_hint_widgets(lb_fvhint, "fwd.verbose_level")
        row += 1

        self.vars["fwd.do_full_refl"] = tk.BooleanVar(value=False)
        cb_refl = ttk.Checkbutton(
            self.fwd_frame, text="do_full_refl (-A)", variable=self.vars["fwd.do_full_refl"], takefocus=1
        )
        cb_refl.grid(row=row, column=0, columnspan=2, sticky=tk.W)
        self._bind_hint_widgets(cb_refl, "fwd.do_full_refl")
        row += 1
        self.vars["fwd.graph_only"] = tk.BooleanVar(value=False)
        self.vars["fwd.omit_air_water"] = tk.BooleanVar(value=False)
        cb_go = ttk.Checkbutton(
            self.fwd_frame, text="graph_only (-g)", variable=self.vars["fwd.graph_only"], takefocus=1
        )
        cb_go.grid(row=row, column=0, columnspan=2, sticky=tk.W)
        self._bind_hint_widgets(cb_go, "fwd.graph_only")
        row += 1
        cb_naw = ttk.Checkbutton(
            self.fwd_frame,
            text="omit_air_water (-n，全网格 -I 时)",
            variable=self.vars["fwd.omit_air_water"],
            takefocus=1,
        )
        cb_naw.grid(row=row, column=0, columnspan=2, sticky=tk.W)
        self._bind_hint_widgets(cb_naw, "fwd.omit_air_water")
        row += 1

        btns = ttk.Frame(self.fwd_frame)
        btns.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))
        b_prev = ttk.Button(btns, text="预览 tt_forward", command=self.preview_tt_forward)
        b_prev.pack(side=tk.LEFT, padx=(0, 6))
        b_run = ttk.Button(btns, text="运行 tt_forward", command=self.run_tt_forward)
        b_run.pack(side=tk.LEFT)
        self._bind_hint_widgets(b_prev, "btn.preview_tt_forward")
        self._bind_hint_widgets(b_run, "btn.run_tt_forward")

    def _build_gen_damp_tab(self) -> None:
        row = 0
        self.vars["damp.vel_opt"] = tk.StringVar(value="uniform")
        self.vars["damp.grid_opt"] = tk.StringVar(value="uniform")
        row = self._combo_row(self.damp_frame, row, "vel_opt (-A | -C)", "damp.vel_opt", ["uniform", "zelt"])
        row = self._combo_row(
            self.damp_frame, row, "grid_opt (-N/-D | -X/-Z/-T | -E/-Z)", "damp.grid_opt", ["uniform", "variable", "zelt"]
        )
        for key, label, default, fmode in [
            ("damp.abnormal_damp", "abnormal_damp (-A 前半)", "", None),
            ("damp.normal_damp", "normal_damp (-A 后半)", "", None),
            ("damp.v_in", "v_in (-C 前半)", "", "open"),
            ("damp.ilayer", "ilayer (-C 后半)", "", None),
            ("damp.top_layer", "top_layer (-F 前半)", "", None),
            ("damp.bot_layer", "bot_layer (-F 后半)", "", None),
        ]:
            if fmode:
                row = self._entry_file_row(self.damp_frame, row, label, key, default, save_as=False)
            else:
                row = self._entry_row(self.damp_frame, row, label, key, default)
        row = self._add_standard_grid_block(self.damp_frame, row, "damp.")
        btns = ttk.Frame(self.damp_frame)
        btns.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))
        bpd = ttk.Button(btns, text="预览 gen_damp", command=self.preview_gen_damp)
        bpd.pack(side=tk.LEFT, padx=(0, 6))
        brd = ttk.Button(btns, text="运行 gen_damp", command=self.run_gen_damp)
        brd.pack(side=tk.LEFT)
        self._bind_hint_widgets(bpd, "btn.preview_gen_damp")
        self._bind_hint_widgets(brd, "btn.run_gen_damp")

    def _build_gen_vcorr_tab(self) -> None:
        row = 0
        self.vars["vcorr.vel_opt"] = tk.StringVar(value="uniform")
        self.vars["vcorr.grid_opt"] = tk.StringVar(value="uniform")
        row = self._combo_row(self.vcorr_frame, row, "vel_opt (-A 四段)", "vcorr.vel_opt", ["uniform", "zelt"])
        row = self._combo_row(
            self.vcorr_frame, row, "grid_opt (-N/-D | -X/-Z/-T | -E/-Z)", "vcorr.grid_opt", ["uniform", "variable", "zelt"]
        )
        for key, label, default, fmode in [
            ("vcorr.abnormal_h", "abnormal_h (-A 1/4)", "", None),
            ("vcorr.abnormal_v", "abnormal_v (-A 2/4)", "", None),
            ("vcorr.normal_h", "normal_h (-A 3/4)", "", None),
            ("vcorr.normal_v", "normal_v (-A 4/4)", "", None),
            ("vcorr.v_in", "v_in (-C 前半)", "", "open"),
            ("vcorr.ilayer", "ilayer (-C 后半)", "", None),
            ("vcorr.top_layer", "top_layer (-F 前半)", "", None),
            ("vcorr.bot_layer", "bot_layer (-F 后半)", "", None),
        ]:
            if fmode:
                row = self._entry_file_row(self.vcorr_frame, row, label, key, default, save_as=False)
            else:
                row = self._entry_row(self.vcorr_frame, row, label, key, default)
        row = self._add_standard_grid_block(self.vcorr_frame, row, "vcorr.")
        btns = ttk.Frame(self.vcorr_frame)
        btns.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))
        bpv = ttk.Button(btns, text="预览 gen_vcorr", command=self.preview_gen_vcorr)
        bpv.pack(side=tk.LEFT, padx=(0, 6))
        brv = ttk.Button(btns, text="运行 gen_vcorr", command=self.run_gen_vcorr)
        brv.pack(side=tk.LEFT)
        self._bind_hint_widgets(bpv, "btn.preview_gen_vcorr")
        self._bind_hint_widgets(brv, "btn.run_gen_vcorr")

    def _build_tt_inverse_tab(self) -> None:
        row = 0
        self.inv_frame.columnconfigure(1, weight=1)
        self.inv_frame.columnconfigure(2, weight=0)
        inv_fields: list[tuple[str, str, str, str | None]] = [
            ("inv.mesh", "mesh (-M)", "", "open"),
            ("inv.data", "data (-G)", "", "open"),
            ("inv.xorder", "xorder (-N)", "4", None),
            ("inv.zorder", "zorder (-N)", "4", None),
            ("inv.clen", "clen (-N)", "0.8", None),
            ("inv.nintp", "nintp (-N)", "8", None),
            ("inv.bend_cg_tol", "bend_cg_tol (-N)", "1e-4", None),
            ("inv.bend_br_tol", "bend_br_tol (-N)", "1e-5", None),
            ("inv.refl_file", "refl_file (-F)", "", "open"),
            ("inv.refl_weight", "refl_weight (-W)", "", None),
            ("inv.log_file", "log_file (-L)", "", "save"),
            ("inv.out_root", "out_root (-O)", "", "save"),
            ("inv.out_level", "out_level (-o)", "", None),
            ("inv.dws_file", "dws_file (-K) 输出", "", "save"),
            ("inv.crit_chi", "crit_chi (-R)", "", None),
            ("inv.lsqr_tol", "lsqr_tol (-Q)", "1e-3", None),
            ("inv.niter", "niter (-I)", "5", None),
            ("inv.target_chi2", "target_chi2 (-J)", "1.0", None),
            ("inv.smooth_vel", "smooth_vel (-SV) 单值或 wmin/wmax/dw", "", None),
            ("inv.smooth_dep", "smooth_dep (-SD) 单值或 wmin/wmax/dw", "", None),
            ("inv.smooth_corr_v_fn", "smooth_corr_v_fn (-CV)", "", "open"),
            ("inv.smooth_corr_d_fn", "smooth_corr_d_fn (-CD)", "", "open"),
            ("inv.damp_vel", "damp_vel (-DV)", "", None),
            ("inv.damp_dep", "damp_dep (-DD)", "", None),
            ("inv.damp_v_fn", "damp_v_fn (-DQ)", "", "open"),
            ("inv.auto_damp_max_dv", "auto_damp_max_dv (-TV %)", "", None),
            ("inv.auto_damp_max_dd", "auto_damp_max_dd (-TD %)", "", None),
            ("inv.filter_bound_file", "filter_bound (-s)", "", "open"),
        ]
        inv_grav_fields: list[tuple[str, str, str, str | None]] = [
            ("inv.grav_file", "grav_file (-ZG)", "", "open"),
            ("inv.grav_grid", "grav_grid (-ZX 六段/)", "", None),
            ("inv.grav_refrange", "grav_refrange (-ZR)", "", None),
            ("inv.grav_cont_file", "continent_up (-ZC 前半)", "", "open"),
            ("inv.grav_cont_iconv", "continent_iconv (-ZC 后半)", "", None),
            ("inv.grav_oceanU_up", "oceanU_up (-ZU 1/3)", "", "open"),
            ("inv.grav_oceanU_lo", "oceanU_lo (-ZU 2/3)", "", "open"),
            ("inv.grav_oceanU_iconv", "oceanU_iconv (-ZU 3/3)", "", None),
            ("inv.grav_oceanL_up", "oceanL_up (-ZL 前半)", "", "open"),
            ("inv.grav_oceanL_iconv", "oceanL_iconv (-ZL 后半)", "", None),
            ("inv.grav_sed_up", "sed_up (-ZS 1/3)", "", "open"),
            ("inv.grav_sed_lo", "sed_lo (-ZS 2/3)", "", "open"),
            ("inv.grav_sed_iconv", "sed_iconv (-ZS 3/3)", "", None),
            ("inv.grav_deriv", "deriv (-ZD 五段/)", "", None),
            ("inv.grav_weight", "weight_grav (-ZW)", "", None),
            ("inv.grav_z0", "z0 (-ZZ)", "", None),
            ("inv.grav_dws", "grav_dws (-ZK)", "", "save"),
            ("inv.grav_cutoff", "cutoff (-ZT 两段/)", "", None),
        ]
        for key, label, default, fmode in inv_fields:
            if fmode == "open":
                row = self._entry_file_row(self.inv_frame, row, label, key, default, save_as=False)
            elif fmode == "save":
                row = self._entry_file_row(self.inv_frame, row, label, key, default, save_as=True)
            else:
                row = self._entry_row(self.inv_frame, row, label, key, default)

        grav_lf = ttk.LabelFrame(self.inv_frame, text="联合重力 (-Z*)", padding=(4, 6))
        grav_lf.grid(row=row, column=0, columnspan=3, sticky=tk.EW, pady=(8, 0))
        grav_lf.columnconfigure(1, weight=1)
        grav_toolbar = ttk.Frame(grav_lf)
        grav_toolbar.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 4))
        self._inv_grav_expanded = False
        self._inv_grav_toggle_btn = ttk.Button(
            grav_toolbar,
            text="▶ 展开联合重力参数",
            command=self._toggle_inv_grav_section,
        )
        self._inv_grav_toggle_btn.pack(side=tk.LEFT, padx=(0, 8))
        self._bind_hint_widgets(self._inv_grav_toggle_btn, "inv.grav_section")
        self._inv_grav_content = ttk.Frame(grav_lf)
        self._inv_grav_content.columnconfigure(1, weight=1)
        gr = 0
        for key, label, default, fmode in inv_grav_fields:
            if fmode == "open":
                gr = self._entry_file_row(self._inv_grav_content, gr, label, key, default, save_as=False)
            elif fmode == "save":
                gr = self._entry_file_row(self._inv_grav_content, gr, label, key, default, save_as=True)
            else:
                gr = self._entry_row(self._inv_grav_content, gr, label, key, default)
        row += 1
        self.vars["inv.verbose_level"] = tk.StringVar(value="")
        self.vars["inv.do_full_refl"] = tk.BooleanVar(value=False)
        self.vars["inv.jumping"] = tk.BooleanVar(value=False)
        self.vars["inv.print_final_only"] = tk.BooleanVar(value=False)
        self.vars["inv.smooth_vel_log10"] = tk.BooleanVar(value=False)
        self.vars["inv.smooth_dep_log10"] = tk.BooleanVar(value=False)
        ctrl = ttk.Frame(self.inv_frame)
        ctrl.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(2, 2))
        r0 = ttk.Frame(ctrl)
        r0.pack(anchor=tk.W)
        lb_vl = ttk.Label(r0, text="verbose_level (-V)")
        lb_vl.pack(side=tk.LEFT, padx=(0, 4))
        en_vl = ttk.Entry(r0, textvariable=self.vars["inv.verbose_level"], width=8, font=self._entry_font)
        en_vl.pack(side=tk.LEFT, padx=(0, 10))
        self.entries["inv.verbose_level"] = en_vl
        self._widget_to_var_key[id(en_vl)] = "inv.verbose_level"
        lb_vhint = ttk.Label(r0, text="（大于0时自动启用 -V）")
        lb_vhint.pack(side=tk.LEFT, padx=(0, 8))
        cb_r = ttk.Checkbutton(r0, text="do_full_refl (-A)", variable=self.vars["inv.do_full_refl"], takefocus=1)
        cb_r.pack(side=tk.LEFT, padx=(0, 10))
        r1 = ttk.Frame(ctrl)
        r1.pack(anchor=tk.W, pady=(4, 0))
        cb_p = ttk.Checkbutton(r1, text="jumping (-P)", variable=self.vars["inv.jumping"], takefocus=1)
        cb_p.pack(side=tk.LEFT, padx=(0, 8))
        cb_l = ttk.Checkbutton(r1, text="print_final_only (-l)", variable=self.vars["inv.print_final_only"], takefocus=1)
        cb_l.pack(side=tk.LEFT, padx=(0, 8))
        cb_xv = ttk.Checkbutton(r1, text="smooth_vel_log10 (-XV)", variable=self.vars["inv.smooth_vel_log10"], takefocus=1)
        cb_xv.pack(side=tk.LEFT, padx=(0, 8))
        cb_xd = ttk.Checkbutton(r1, text="smooth_dep_log10 (-XD)", variable=self.vars["inv.smooth_dep_log10"], takefocus=1)
        cb_xd.pack(side=tk.LEFT, padx=(0, 8))
        self._bind_hint_widgets(lb_vl, "inv.verbose_level")
        self._bind_hint_widgets(en_vl, "inv.verbose_level")
        self._bind_hint_widgets(lb_vhint, "inv.verbose_level")
        self._bind_hint_widgets(cb_r, "inv.do_full_refl")
        self._bind_hint_widgets(cb_p, "inv.jumping")
        self._bind_hint_widgets(cb_l, "inv.print_final_only")
        self._bind_hint_widgets(cb_xv, "inv.smooth_vel_log10")
        self._bind_hint_widgets(cb_xd, "inv.smooth_dep_log10")
        row += 1
        self.vars["inv.use_repro_bundle"] = tk.BooleanVar(value=True)
        cb_bundle = ttk.Checkbutton(
            self.inv_frame,
            text="可复现运行包（runs/ttinv_…/：inputs、outputs、manifest；目录名含 mesh/data 主干与本地时间）",
            variable=self.vars["inv.use_repro_bundle"],
            takefocus=1,
        )
        cb_bundle.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(6, 0))
        self._bind_hint_widgets(cb_bundle, "inv.use_repro_bundle")
        row += 1
        row = self._entry_row(
            self.inv_frame,
            row,
            "运行包目录备注（可选，插入目录名）",
            "inv.bundle_run_label",
            "",
        )
        row += 1
        btns = ttk.Frame(self.inv_frame)
        btns.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))
        bpi = ttk.Button(btns, text="预览 tt_inverse", command=self.preview_tt_inverse)
        bpi.pack(side=tk.LEFT, padx=(0, 6))
        bri = ttk.Button(btns, text="运行 tt_inverse", command=self.run_tt_inverse)
        bri.pack(side=tk.LEFT)
        blg = ttk.Button(
            btns,
            text="-L 日志列说明",
            command=self._show_tt_inverse_logfile_format_help,
        )
        blg.pack(side=tk.LEFT, padx=(12, 0))
        self._bind_hint_widgets(bpi, "btn.preview_tt_inverse")
        self._bind_hint_widgets(bri, "btn.run_tt_inverse")
        self._bind_hint_widgets(blg, "btn.tt_inverse_log_help")

    def _build_stat_smesh_tab(self) -> None:
        row = 0
        self.stat_frame.columnconfigure(1, weight=1)
        self.stat_frame.columnconfigure(2, weight=0)
        self.vars["stat.mode"] = tk.StringVar(value="mesh")
        self.vars["stat.cmd_type"] = tk.StringVar(value="a")
        row = self._combo_row(self.stat_frame, row, "mode (-L | -M)", "stat.mode", ["mesh", "list"])
        row = self._combo_row(self.stat_frame, row, "cmd_type (-C / -D)", "stat.cmd_type", ["a", "b", "r"])
        for key, label, default, fmode in [
            ("stat.list_file", "list_file (-L)", "", "open"),
            ("stat.mesh_file", "mesh_file (-M)", "", "open"),
            ("stat.ave_file", "ave_file (-Cr)", "", "open"),
            ("stat.refl_nnodes", "refl_nnodes (-R)", "", None),
            ("stat.ave_x", "ave_x (-Da 1/2)", "", None),
            ("stat.xmin", "xmin (-Db 1/4)", "", None),
            ("stat.xmax", "xmax (-Db 2/4)", "", None),
            ("stat.dx", "dx (-Db 3/4)", "", None),
            ("stat.window_len", "window_len (-Da/-Db 末项)", "", None),
            ("stat.top_bound", "top_bound (-T)", "", "open"),
            ("stat.bot_bound", "bot_bound (-B)", "", "open"),
            ("stat.mid_bound", "mid_bound (-m)", "", "open"),
            ("stat.pt_corr", "pt_corr (-P 六段/)", "", None),
            ("stat.vrepl", "vrepl (-U)", "", None),
            ("stat.abs_xmin", "abs_xmin (-X)", "", None),
            ("stat.abs_xmax", "abs_xmax (-X)", "", None),
            ("stat.exclude_cxmin", "exclude_cxmin (-x 前半)", "", None),
            ("stat.exclude_cxmax", "exclude_cxmax (-x 后半)", "", None),
            ("stat.exclude_top_bound", "exclude_top (-t)", "", "open"),
            ("stat.exclude_bot_bound", "exclude_bot (-b)", "", "open"),
        ]:
            if fmode:
                row = self._entry_file_row(self.stat_frame, row, label, key, default, save_as=False)
            else:
                row = self._entry_row(self.stat_frame, row, label, key, default)
        self.vars["stat.verbose"] = tk.BooleanVar(value=False)
        cbv = ttk.Checkbutton(self.stat_frame, text="verbose (-V)", variable=self.vars["stat.verbose"], takefocus=1)
        cbv.grid(row=row, column=0, columnspan=2, sticky=tk.W)
        self._bind_hint_widgets(cbv, "stat.verbose")
        row += 1
        btns = ttk.Frame(self.stat_frame)
        btns.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))
        bps = ttk.Button(btns, text="预览 stat_smesh", command=self.preview_stat_smesh)
        bps.pack(side=tk.LEFT, padx=(0, 6))
        brs = ttk.Button(btns, text="运行 stat_smesh", command=self.run_stat_smesh)
        brs.pack(side=tk.LEFT)
        self._bind_hint_widgets(bps, "btn.preview_stat_smesh")
        self._bind_hint_widgets(brs, "btn.run_stat_smesh")

    def _build_edit_smesh_tab(self) -> None:
        row = 0
        self.edit_frame.columnconfigure(1, weight=1)
        self.edit_frame.columnconfigure(2, weight=0)
        self.vars["edit.cmd_type"] = tk.StringVar(value="a")
        row = self._combo_row(
            self.edit_frame,
            row,
            "cmd_type (-C 子命令)",
            "edit.cmd_type",
            ["a", "p", "P", "B", "s", "rm", "c", "d", "g", "l", "R", "S", "G", "m", "b"],
        )
        for key, label, default, fmode in [
            ("edit.smesh_file", "smesh_file (位置参数)", "", "open"),
            ("edit.paste_file", "paste_file (-C p…)", "", "open"),
            ("edit.prof_file", "prof_file (-C P…)", "", "open"),
            ("edit.remove_bg_file", "remove_bg_prof (-CB …)", "", "open"),
            ("edit.h_len", "h_len (-C s/c …)", "", None),
            ("edit.v_len", "v_len (-C s/c …)", "", None),
            ("edit.mx", "mx (-C rm …)", "", None),
            ("edit.mz", "mz (-C rm …)", "", None),
            ("edit.amp", "amp (-C c/d/g/R/S/G …)", "", None),
            ("edit.xmin", "xmin (-C d/S/G …)", "", None),
            ("edit.xmax", "xmax (-C d/S/G …)", "", None),
            ("edit.zmin", "zmin (-C d/S/G …)", "", None),
            ("edit.zmax", "zmax (-C d/S/G …)", "", None),
            ("edit.x0", "x0 (-C g …)", "", None),
            ("edit.z0", "z0 (-C g …)", "", None),
            ("edit.Lh", "Lh (-C g …)", "", None),
            ("edit.Lv", "Lv (-C g …)", "", None),
            ("edit.seed", "seed (-C R/S/G …)", "", None),
            ("edit.nrand", "nrand (-C R …)", "", None),
            ("edit.N", "N (-C G …)", "", None),
            ("edit.dx", "dx (-C S …)", "", None),
            ("edit.dz", "dz (-C S …)", "", None),
            ("edit.vel", "vel (-C m …)", "", None),
            ("edit.moho_file", "moho_file (-C m …)", "", "open"),
            ("edit.k", "k (-C b …)", "", None),
            ("edit.base_file", "base_file (-C b …)", "", "open"),
            ("edit.corr_file", "corr_file (-L)", "", "open"),
            ("edit.upper_bound", "upper_bound (-U)", "", "open"),
        ]:
            if fmode:
                row = self._entry_file_row(self.edit_frame, row, label, key, default, save_as=False)
            else:
                row = self._entry_row(self.edit_frame, row, label, key, default)
        btns = ttk.Frame(self.edit_frame)
        btns.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))
        bpe = ttk.Button(btns, text="预览 edit_smesh_HHB", command=self.preview_edit_smesh)
        bpe.pack(side=tk.LEFT, padx=(0, 6))
        bre = ttk.Button(btns, text="运行 edit_smesh_HHB", command=self.run_edit_smesh)
        bre.pack(side=tk.LEFT)
        self._bind_hint_widgets(bpe, "btn.preview_edit_smesh")
        self._bind_hint_widgets(bre, "btn.run_edit_smesh")

    def _build_pipeline_tab(self) -> None:
        row = 0
        self.pipe_frame.columnconfigure(1, weight=1)
        self.pipe_frame.columnconfigure(2, weight=0)
        self.vars["pipe.recipe"] = tk.StringVar(value="gen_smesh -> tt_forward")
        row = self._combo_row(
            self.pipe_frame,
            row,
            "recipe (GUI 流程)",
            "pipe.recipe",
            [
                "gen_smesh -> tt_forward",
                "gen_smesh -> tt_inverse",
                "gen_smesh -> gen_damp -> tt_inverse",
                "gen_smesh -> gen_vcorr -> tt_inverse",
            ],
        )
        note = ttk.Label(
            self.pipe_frame,
            text="说明：流程直接复用各页签当前参数；请先在对应页签填写并检查参数。",
        )
        note.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(6, 4))
        self._bind_hint_widgets(note, "pipe.recipe")
        row += 1
        self.vars["pipe.link_smesh"] = tk.StringVar(value="")
        row = self._entry_file_row(self.pipe_frame, row, "link_smesh (桥接)", "pipe.link_smesh", "", save_as=False)
        self.vars["pipe.link_damp"] = tk.StringVar(value="")
        row = self._entry_file_row(self.pipe_frame, row, "link_damp (桥接)", "pipe.link_damp", "", save_as=False)
        self.vars["pipe.link_vcorr_v"] = tk.StringVar(value="")
        row = self._entry_file_row(self.pipe_frame, row, "link_vcorr_v (桥接)", "pipe.link_vcorr_v", "", save_as=False)
        self.vars["pipe.link_vcorr_d"] = tk.StringVar(value="")
        row = self._entry_file_row(self.pipe_frame, row, "link_vcorr_d (桥接)", "pipe.link_vcorr_d", "", save_as=False)
        self.vars["pipe.auto_wire"] = tk.BooleanVar(value=True)
        cb_auto = ttk.Checkbutton(
            self.pipe_frame,
            text="自动衔接到下游参数（仅在目标参数为空时填入）",
            variable=self.vars["pipe.auto_wire"],
            takefocus=1,
        )
        cb_auto.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(2, 2))
        self._bind_hint_widgets(cb_auto, "pipe.auto_wire")
        row += 1
        btns = ttk.Frame(self.pipe_frame)
        btns.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))
        bpp = ttk.Button(btns, text="预览 pipeline", command=self.preview_pipeline)
        bpp.pack(side=tk.LEFT, padx=(0, 6))
        brp = ttk.Button(btns, text="运行 pipeline", command=self.run_pipeline)
        brp.pack(side=tk.LEFT)
        self._bind_hint_widgets(bpp, "btn.preview_pipeline")
        self._bind_hint_widgets(brp, "btn.run_pipeline")

    def _build_tx_convert_tab(self) -> None:
        row = 0
        self.tx_frame.columnconfigure(1, weight=1)
        self.tx_frame.columnconfigure(2, weight=0)
        note = ttk.Label(
            self.tx_frame,
            text=(
                "将 Zelt 系 tx.in 按震相筛选为 ttimes.dat（tt_inverse -G）与 geom.dat（tt_forward -g）；"
                "等价 Fortran tx2tomo2d.f。折射/反射拾取均输出为 r 行：第三列 0=折射、1=反射；"
                "ttimes.dat 写走时 t 与误差 u，geom.dat 同几何下 t、u 为 0。"
            ),
        )
        note.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 6))
        self._bind_hint_widgets(note, "tab.tx_convert")
        row += 1
        row = self._entry_file_row(
            self.tx_frame, row, "station.lis（炮点号 x z）", "tx.station_lis", "station.lis", save_as=False
        )
        row = self._entry_file_row(
            self.tx_frame, row, "tx.in（走时）", "tx.tx_in", "tx.in", save_as=False
        )
        row = self._entry_file_row(
            self.tx_frame,
            row,
            "输出 ttimes.dat（浏览另存或填路径；仅文件名→work_dir）",
            "tx.data_out",
            "ttimes.dat",
            save_as=True,
        )
        row = self._entry_file_row(
            self.tx_frame,
            row,
            "输出 geom.dat（浏览另存或填路径；仅文件名→work_dir）",
            "tx.geom_out",
            "geom.dat",
            save_as=True,
        )
        row = self._entry_row(
            self.tx_frame,
            row,
            "折射震相编号（逗号或空格分隔，与 Fortran 一致）",
            "tx.refr_phases",
            "1",
        )
        row = self._entry_row(
            self.tx_frame,
            row,
            "反射震相编号（逗号或空格分隔，可留空）",
            "tx.refl_phases",
            "11,12",
        )
        btns = ttk.Frame(self.tx_frame)
        btns.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))
        bpx = ttk.Button(btns, text="预览 tx 转换", command=self.preview_tx_convert)
        brx = ttk.Button(btns, text="运行 tx 转换", command=self.run_tx_convert)
        bpx.pack(side=tk.LEFT, padx=(0, 6))
        brx.pack(side=tk.LEFT)
        self._bind_hint_widgets(bpx, "btn.preview_tx_convert")
        self._bind_hint_widgets(brx, "btn.run_tx_convert")

    def _tx_convert_resolved_paths(self, work: Path):
        """返回 station、tx.in、输出 data/geom 的绝对路径。"""
        ss = self.vars["tx.station_lis"].get().strip() or "station.lis"
        tx = self.vars["tx.tx_in"].get().strip() or "tx.in"
        do = self.vars["tx.data_out"].get().strip() or "ttimes.dat"
        go = self.vars["tx.geom_out"].get().strip() or "geom.dat"

        def to_abs(pstr: str) -> Path:
            p = Path(pstr).expanduser()
            if p.is_absolute():
                return p.resolve()
            rel = self._to_workdir_relative(pstr, show_warn_outside=False)
            return (work / rel).resolve()

        return to_abs(ss), to_abs(tx), to_abs(do), to_abs(go)

    def _entry_row(self, parent: ttk.Frame, row: int, label: str, key: str, default: str = "") -> int:
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky=tk.W, padx=(0, 6), pady=2)
        self.vars[key] = tk.StringVar(value=str(default) if default is not None else "")
        e = ttk.Entry(parent, textvariable=self.vars[key], width=40, font=self._entry_font)
        e.grid(row=row, column=1, sticky=tk.EW, pady=2)
        self.entries[key] = e
        self._widget_to_var_key[id(e)] = key
        self._bind_hint_widgets(lbl, key)
        self._bind_hint_widgets(e, key)
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="").grid(row=row, column=2, sticky=tk.W)
        return row + 1

    def _entry_file_row(self, parent: ttk.Frame, row: int, label: str, key: str, default: str = "", save_as: bool = False) -> int:
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky=tk.W, padx=(0, 6), pady=2)
        self.vars[key] = tk.StringVar(value=str(default) if default is not None else "")
        e = ttk.Entry(parent, textvariable=self.vars[key], width=36, font=self._entry_font)
        e.grid(row=row, column=1, sticky=tk.EW, pady=2)
        self.entries[key] = e
        self._widget_to_var_key[id(e)] = key
        bt = ttk.Button(parent, text="浏览…", width=8, command=lambda: self._pick_file(key, save_as=save_as))
        bt.grid(row=row, column=2, sticky=tk.E, padx=(4, 0), pady=2)
        self.file_buttons[key] = bt
        self._bind_hint_widgets(lbl, key)
        self._bind_hint_widgets(e, key)
        self._bind_hint_widgets(bt, key)
        parent.columnconfigure(1, weight=1)
        self._try_register_dnd(e, key)
        return row + 1

    def _pick_file(self, var_key: str, save_as: bool = False) -> None:
        cur = self.vars[var_key].get().strip()
        wd_s = self.vars["work_dir"].get().strip() if "work_dir" in self.vars else ""
        work_base: Path | None = None
        init_dir = str(Path.cwd())
        if wd_s:
            wdp = Path(wd_s).expanduser()
            if wdp.is_dir():
                work_base = wdp.resolve()
                init_dir = str(work_base)
        if cur:
            p = Path(cur).expanduser()
            if not p.is_absolute() and work_base is not None:
                p = work_base / p
            elif not p.is_absolute():
                p = Path.cwd() / p
            if p.is_file():
                init_dir = str(p.parent)
            elif p.is_dir():
                init_dir = str(p)
            elif p.parent.is_dir():
                init_dir = str(p.parent)
        if save_as:
            init_file = ""
            if cur:
                bn = Path(cur).name
                if bn and bn not in (".", ".."):
                    init_file = bn
            if not init_file:
                if var_key == "tx.data_out":
                    init_file = "ttimes.dat"
                elif var_key == "tx.geom_out":
                    init_file = "geom.dat"
            save_kw: dict = {"title": f"另存为: {var_key}", "initialdir": init_dir}
            if init_file:
                save_kw["initialfile"] = init_file
            path = filedialog.asksaveasfilename(**save_kw)
        else:
            path = filedialog.askopenfilename(title=f"选择文件: {var_key}", initialdir=init_dir)
        if path:
            rel = self._to_workdir_relative(path, show_warn_outside=True)
            self.vars[var_key].set(self._postprocess_zelt_file_var(var_key, rel))

    def _combo_row(self, parent: ttk.Frame, row: int, label: str, key: str, values: list[str]) -> int:
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky=tk.W, padx=(0, 6), pady=2)
        cb = ttk.Combobox(
            parent,
            textvariable=self.vars[key],
            values=values,
            state="readonly",
            width=37,
            font=self._entry_font,
        )
        cb.grid(row=row, column=1, sticky=tk.W, pady=2)
        self._widget_to_var_key[id(cb)] = key
        setattr(self, f"_combo_{key.replace('.', '_')}", cb)
        self._bind_hint_widgets(lbl, key)
        self._bind_hint_widgets(cb, key)
        ttk.Label(parent, text="").grid(row=row, column=2, sticky=tk.W)
        return row + 1

    def _bind_dynamic_controls(self) -> None:
        self.vars["gen.vel_opt"].trace_add("write", lambda *_: self._on_gen_smesh_mode_change())
        self.vars["gen.grid_opt"].trace_add("write", lambda *_: self._on_gen_smesh_mode_change())
        self.vars["damp.vel_opt"].trace_add("write", lambda *_: self._on_damp_mode_change())
        self.vars["damp.grid_opt"].trace_add("write", lambda *_: self._on_damp_mode_change())
        self.vars["vcorr.vel_opt"].trace_add("write", lambda *_: self._on_vcorr_mode_change())
        self.vars["vcorr.grid_opt"].trace_add("write", lambda *_: self._on_vcorr_mode_change())
        self.vars["stat.mode"].trace_add("write", lambda *_: self._on_stat_mode_change())
        self.vars["stat.cmd_type"].trace_add("write", lambda *_: self._on_stat_mode_change())
        self.vars["edit.cmd_type"].trace_add("write", lambda *_: self._on_edit_mode_change())
        self.vars["inv.verbose_level"].trace_add("write", lambda *_: self._on_inv_verbose_level_change())

    def _on_gen_smesh_mode_change(self) -> None:
        vel_opt = self.vars["gen.vel_opt"].get()
        grid_opt = self.vars["gen.grid_opt"].get()
        # gen_smesh.cc：-C（Zelt 速度）同时计入速度与网格选项，须与 -E/-Z 配套 → grid 必须为 zelt
        if vel_opt == "zelt" and grid_opt != "zelt":
            self.vars["gen.grid_opt"].set("zelt")
            grid_opt = "zelt"
        if vel_opt == "uniform" and grid_opt == "zelt":
            self.vars["gen.grid_opt"].set("uniform")
            grid_opt = "uniform"

        cb_grid = getattr(self, "_combo_gen_grid_opt", None)
        if cb_grid is not None:
            cb_grid.configure(state="disabled" if vel_opt == "zelt" else "readonly")

        self._set_enabled(["gen.v0", "gen.gradient"], vel_opt == "uniform")
        self._set_enabled(["gen.v_in", "gen.ilayer", "gen.refl_layer", "gen.refl_file"], vel_opt == "zelt")
        self._set_enabled(["gen.zelt_dump"], vel_opt == "zelt")
        self._apply_grid_file_state("gen.", grid_opt)

    def _on_damp_mode_change(self) -> None:
        vo = self.vars["damp.vel_opt"].get()
        go = self.vars["damp.grid_opt"].get()
        if vo == "zelt" and go != "zelt":
            self.vars["damp.grid_opt"].set("zelt")
            go = "zelt"
        if vo == "uniform" and go == "zelt":
            self.vars["damp.grid_opt"].set("uniform")
            go = "uniform"
        cb_grid = getattr(self, "_combo_damp_grid_opt", None)
        if cb_grid is not None:
            cb_grid.configure(state="disabled" if vo == "zelt" else "readonly")
        self._set_enabled(["damp.abnormal_damp", "damp.normal_damp"], vo == "uniform")
        self._set_enabled(["damp.v_in", "damp.ilayer", "damp.top_layer", "damp.bot_layer"], vo == "zelt")
        self._apply_grid_file_state("damp.", go)

    def _on_vcorr_mode_change(self) -> None:
        vo = self.vars["vcorr.vel_opt"].get()
        go = self.vars["vcorr.grid_opt"].get()
        if vo == "zelt" and go != "zelt":
            self.vars["vcorr.grid_opt"].set("zelt")
            go = "zelt"
        if vo == "uniform" and go == "zelt":
            self.vars["vcorr.grid_opt"].set("uniform")
            go = "uniform"
        cb_grid = getattr(self, "_combo_vcorr_grid_opt", None)
        if cb_grid is not None:
            cb_grid.configure(state="disabled" if vo == "zelt" else "readonly")
        self._set_enabled(
            ["vcorr.abnormal_h", "vcorr.abnormal_v", "vcorr.normal_h", "vcorr.normal_v"],
            vo == "uniform",
        )
        self._set_enabled(
            ["vcorr.v_in", "vcorr.ilayer", "vcorr.top_layer", "vcorr.bot_layer"],
            vo == "zelt",
        )
        self._apply_grid_file_state("vcorr.", go)

    def _on_stat_mode_change(self) -> None:
        mode = self.vars["stat.mode"].get()
        cmd = self.vars["stat.cmd_type"].get()
        cb = getattr(self, "_combo_stat_cmd_type", None)
        if cb is not None:
            allowed = ["a", "r"] if mode == "list" else ["a", "b"]
            cb.configure(values=allowed)
        if mode == "list" and cmd == "b":
            self.vars["stat.cmd_type"].set("a")
            cmd = "a"
        if mode == "mesh" and cmd == "r":
            self.vars["stat.cmd_type"].set("a")
            cmd = "a"
        is_list = mode == "list"
        mesh = not is_list
        self._set_enabled(["stat.list_file"], is_list)
        self._set_enabled(["stat.mesh_file"], mesh)
        self._set_enabled(["stat.ave_file"], is_list and cmd == "r")
        self._set_enabled(["stat.ave_x"], mesh and cmd == "a")
        self._set_enabled(["stat.xmin", "stat.xmax", "stat.dx"], mesh and cmd == "b")
        self._set_enabled(["stat.window_len"], mesh and cmd in ("a", "b"))
        self._set_enabled(["stat.top_bound", "stat.bot_bound", "stat.mid_bound"], mesh)
        self._set_enabled(["stat.refl_nnodes"], is_list)
        for k in (
            "stat.pt_corr",
            "stat.vrepl",
            "stat.abs_xmin",
            "stat.abs_xmax",
            "stat.exclude_cxmin",
            "stat.exclude_cxmax",
            "stat.exclude_top_bound",
            "stat.exclude_bot_bound",
        ):
            self._set_enabled([k], mesh)

    def _on_edit_mode_change(self) -> None:
        cmd = self.vars["edit.cmd_type"].get()
        map_fields: dict[str, list[str]] = {
            "a": [],
            "p": ["edit.paste_file"],
            "P": ["edit.prof_file"],
            "B": ["edit.remove_bg_file"],
            "s": ["edit.h_len", "edit.v_len"],
            "rm": ["edit.mx", "edit.mz"],
            "c": ["edit.amp", "edit.h_len", "edit.v_len"],
            "d": ["edit.amp", "edit.xmin", "edit.xmax", "edit.zmin", "edit.zmax"],
            "g": ["edit.amp", "edit.x0", "edit.z0", "edit.Lh", "edit.Lv"],
            "l": [],
            "R": ["edit.seed", "edit.amp", "edit.nrand"],
            "S": ["edit.seed", "edit.amp", "edit.xmin", "edit.xmax", "edit.dx", "edit.zmin", "edit.zmax", "edit.dz"],
            "G": ["edit.seed", "edit.amp", "edit.N", "edit.xmin", "edit.xmax", "edit.zmin", "edit.zmax"],
            "m": ["edit.vel", "edit.moho_file"],
            "b": ["edit.k", "edit.base_file"],
        }
        all_optional = [
            "edit.paste_file",
            "edit.prof_file",
            "edit.remove_bg_file",
            "edit.h_len",
            "edit.v_len",
            "edit.mx",
            "edit.mz",
            "edit.amp",
            "edit.xmin",
            "edit.xmax",
            "edit.zmin",
            "edit.zmax",
            "edit.x0",
            "edit.z0",
            "edit.Lh",
            "edit.Lv",
            "edit.seed",
            "edit.nrand",
            "edit.N",
            "edit.dx",
            "edit.dz",
            "edit.vel",
            "edit.moho_file",
            "edit.k",
            "edit.base_file",
        ]
        need = set(map_fields.get(cmd, []))
        for key in all_optional:
            self._set_enabled([key], key in need)

    def _on_inv_verbose_level_change(self) -> None:
        # verbose_level 仅用于提示自动启用 -V，不再需要显式 verbose 复选框状态同步
        return

    def _set_enabled(self, keys: list[str], enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for k in keys:
            if k in self.entries:
                self.entries[k].configure(state=state)
            if k in self.file_buttons:
                self.file_buttons[k].configure(state=state)

    def _pick_dir(self, var_key: str) -> None:
        path = filedialog.askdirectory(initialdir=self.vars[var_key].get() or str(Path.cwd()))
        if path:
            self.vars[var_key].set(path)

    def _work_dir_resolved(self) -> Path:
        """当前界面 work_dir 的绝对路径（解析失败时退回 cwd）。"""
        wd = self.vars["work_dir"].get().strip() if "work_dir" in self.vars else ""
        base = Path(wd or ".").expanduser()
        try:
            return base.resolve()
        except OSError:
            return Path.cwd()

    def _to_workdir_relative(self, path_str: str, *, show_warn_outside: bool = True) -> str:
        """
        将路径转为相对于 work_dir 的字符串（POSIX 斜杠），供 tomo2d 在 cwd=work_dir 下识别。
        已为相对路径时仅规范化；绝对路径且在 work_dir 下则去掉前缀。
        """
        raw = (path_str or "").strip()
        if not raw:
            return ""
        work = self._work_dir_resolved()
        p = Path(raw).expanduser()
        if not p.is_absolute():
            norm = os.path.normpath(raw)
            return Path(norm).as_posix()
        try:
            rp = p.resolve()
        except OSError:
            rp = p
        try:
            return rp.relative_to(work).as_posix()
        except ValueError:
            rel_s = os.path.relpath(os.fspath(rp), os.fspath(work))
            if show_warn_outside and rel_s.startswith(".."):
                messagebox.showwarning(
                    "路径不在 work_dir 内",
                    "所选路径不在当前「工作目录」之下。\n"
                    "已存为相对路径（含 ..），请确认底层程序在 work_dir 下运行时能解析。\n\n"
                    f"work_dir: {work}\n路径: {rp}",
                )
            return Path(rel_s).as_posix()

    def _postprocess_zelt_file_var(self, var_key: str, workdir_relative: str) -> str:
        """
        v.in / 反射文件等填入 -C/-F 时，C 程序用 sscanf %%[^/] 解析，路径段内不能含 '/'。
        统一为仅文件名，并假定文件位于 work_dir 根目录（与 subprocess cwd 一致）。
        """
        if var_key not in _TOMO2D_ZELT_TOKEN_FILE_KEYS:
            return workdir_relative
        s = (workdir_relative or "").strip()
        if not s:
            return s
        base = os.path.basename(s.replace("\\", "/"))
        if base != s:
            messagebox.showwarning(
                "已改为仅文件名",
                "tomo2d 的 -C v.in/层号、-F 层号/反射文件 由 C 程序 sscanf 解析，"
                "文件路径中不能含目录分隔符（否则会在第一个 '/' 处被截断）。\n"
                "已自动改为仅文件名；请将该文件放在当前「工作目录」根目录下再运行。\n\n"
                f"原值: {s}\n现值: {base}",
            )
        return base

    def _normalize_all_file_path_vars(self, *, show_warn_outside: bool = False) -> None:
        """将所有「浏览…」类参数转为相对 work_dir（若可能）。"""
        for key in self.file_buttons:
            cur = self.vars[key].get().strip()
            if not cur:
                continue
            rel = self._to_workdir_relative(cur, show_warn_outside=show_warn_outside)
            self.vars[key].set(self._postprocess_zelt_file_var(key, rel))

    def _to_number(self, value: str):
        if value == "":
            return None
        try:
            if any(c in value.lower() for c in [".", "e"]):
                return float(value)
            return int(value)
        except ValueError:
            return value

    def _collect_gen_smesh_kwargs(self) -> dict:
        kwargs: dict = {
            "vel_opt": self.vars["gen.vel_opt"].get(),
            "grid_opt": self.vars["gen.grid_opt"].get(),
        }
        path_keys = frozenset({"v_in", "refl_file"})
        vo = kwargs["vel_opt"]
        if vo == "uniform":
            vel_keys = ["v0", "gradient"]
        elif vo == "zelt":
            vel_keys = ["v_in", "ilayer", "refl_layer", "refl_file"]
        else:
            vel_keys = []
        for key in vel_keys:
            raw = self.vars[f"gen.{key}"].get().strip()
            if raw:
                kwargs[key] = raw if key in path_keys else self._to_number(raw)
        self._merge_grid_vars_into(kwargs, "gen.", grid_opt=kwargs["grid_opt"])
        for key in ["water_col", "v_water", "v_air"]:
            raw = self.vars[f"gen.{key}"].get().strip()
            if raw:
                kwargs[key] = self._to_number(raw)
        if vo == "zelt":
            zd = self.vars["gen.zelt_dump"].get().strip()
            if zd:
                kwargs["zelt_dump_file"] = zd
        smesh_out = self.vars["gen.smesh_out"].get().strip()
        if smesh_out:
            kwargs["out_file"] = smesh_out
        return kwargs

    def _collect_gen_damp_kwargs(self) -> dict:
        kwargs: dict = {
            "vel_opt": self.vars["damp.vel_opt"].get(),
            "grid_opt": self.vars["damp.grid_opt"].get(),
        }
        vo = kwargs["vel_opt"]
        if vo == "uniform":
            for key in ("abnormal_damp", "normal_damp"):
                raw = self.vars[f"damp.{key}"].get().strip()
                if raw:
                    kwargs[key] = self._to_number(raw)
        elif vo == "zelt":
            for key in ("v_in", "ilayer", "top_layer", "bot_layer"):
                raw = self.vars[f"damp.{key}"].get().strip()
                if raw:
                    kwargs[key] = raw if key == "v_in" else self._to_number(raw)
        self._merge_grid_vars_into(kwargs, "damp.", grid_opt=kwargs["grid_opt"])
        return kwargs

    def _collect_gen_vcorr_kwargs(self) -> dict:
        kwargs: dict = {
            "vel_opt": self.vars["vcorr.vel_opt"].get(),
            "grid_opt": self.vars["vcorr.grid_opt"].get(),
        }
        vo = kwargs["vel_opt"]
        if vo == "uniform":
            for key in ("abnormal_h", "abnormal_v", "normal_h", "normal_v"):
                raw = self.vars[f"vcorr.{key}"].get().strip()
                if raw:
                    kwargs[key] = self._to_number(raw)
        elif vo == "zelt":
            for key in ("v_in", "ilayer", "top_layer", "bot_layer"):
                raw = self.vars[f"vcorr.{key}"].get().strip()
                if raw:
                    kwargs[key] = raw if key == "v_in" else self._to_number(raw)
        self._merge_grid_vars_into(kwargs, "vcorr.", grid_opt=kwargs["grid_opt"])
        return kwargs

    def _collect_tt_inverse_args(self) -> tuple[str, str, dict]:
        mesh = self.vars["inv.mesh"].get().strip()
        data = self.vars["inv.data"].get().strip()
        kwargs: dict = {}
        for key in ["xorder", "zorder", "clen", "nintp", "bend_cg_tol", "bend_br_tol"]:
            raw = self.vars[f"inv.{key}"].get().strip()
            if raw:
                kwargs[key] = self._to_number(raw)
        refl = self.vars["inv.refl_file"].get().strip()
        if refl:
            kwargs["refl_file"] = refl
        rw = self.vars["inv.refl_weight"].get().strip()
        if rw:
            kwargs["refl_weight"] = self._to_number(rw)
        if self.vars["inv.do_full_refl"].get():
            kwargs["do_full_refl"] = True
        if self.vars["inv.jumping"].get():
            kwargs["jumping"] = True
        if self.vars["inv.print_final_only"].get():
            kwargs["print_final_only"] = True
        fb = self.vars["inv.filter_bound_file"].get().strip()
        if fb:
            kwargs["filter_bound_file"] = fb
        for key in ["log_file", "out_root", "out_level", "dws_file"]:
            raw = self.vars[f"inv.{key}"].get().strip()
            if raw:
                kwargs[key] = raw if key != "out_level" else self._to_number(raw)
        for key in ["crit_chi", "lsqr_tol", "niter", "target_chi2"]:
            raw = self.vars[f"inv.{key}"].get().strip()
            if raw:
                kwargs[key] = self._to_number(raw)

        adv = self.vars["inv.auto_damp_max_dv"].get().strip()
        add = self.vars["inv.auto_damp_max_dd"].get().strip()
        if adv:
            kwargs["auto_damp_max_dv"] = self._to_number(adv)
        if add:
            kwargs["auto_damp_max_dd"] = self._to_number(add)

        sv = self.vars["inv.smooth_vel"].get().strip()
        sd = self.vars["inv.smooth_dep"].get().strip()
        cv = self.vars["inv.smooth_corr_v_fn"].get().strip()
        cd = self.vars["inv.smooth_corr_d_fn"].get().strip()
        xv = self.vars["inv.smooth_vel_log10"].get()
        xd = self.vars["inv.smooth_dep_log10"].get()
        if sv or sd or cv or cd or xv or xd:
            smooth: dict = {}
            if sv:
                smooth["vel"] = sv if ("/" in sv and sv.count("/") >= 2) else self._to_number(sv)
            if sd:
                smooth["dep"] = sd if ("/" in sd and sd.count("/") >= 2) else self._to_number(sd)
            if cv:
                smooth["corr_v_fn"] = cv
            if cd:
                smooth["corr_d_fn"] = cd
            if xv:
                smooth["vel_log10"] = True
            if xd:
                smooth["dep_log10"] = True
            kwargs["smooth_opts"] = smooth

        dv = self.vars["inv.damp_vel"].get().strip()
        dd = self.vars["inv.damp_dep"].get().strip()
        dq = self.vars["inv.damp_v_fn"].get().strip()
        if dv or dd or dq:
            damp: dict = {}
            if dv:
                damp["vel"] = self._to_number(dv)
            if dd:
                damp["dep"] = self._to_number(dd)
            if dq:
                damp["damp_v_fn"] = dq
            kwargs["damp_opts"] = damp

        gf = self.vars["inv.grav_file"].get().strip()
        if gf:
            g: dict = {"grav_file": gf}
            gg = self.vars["inv.grav_grid"].get().strip()
            grng = self.vars["inv.grav_refrange"].get().strip()
            if gg:
                g["grid_spec"] = gg
            if grng:
                g["refrange"] = grng
            cf = self.vars["inv.grav_cont_file"].get().strip()
            if cf:
                ic = self.vars["inv.grav_cont_iconv"].get().strip()
                if not ic:
                    raise ValueError("联合重力：已填 grav_ZC cont_up 时须填 grav_ZC iconv")
                g["continent"] = (cf, int(self._to_number(ic)))
            uu = self.vars["inv.grav_oceanU_up"].get().strip()
            ul = self.vars["inv.grav_oceanU_lo"].get().strip()
            ui = self.vars["inv.grav_oceanU_iconv"].get().strip()
            if uu or ul or ui:
                if not (uu and ul and ui):
                    raise ValueError("联合重力：grav_ZU 三项须同时填写")
                g["ocean_upper"] = (uu, ul, int(self._to_number(ui)))
            lu = self.vars["inv.grav_oceanL_up"].get().strip()
            li = self.vars["inv.grav_oceanL_iconv"].get().strip()
            if lu or li:
                if not (lu and li):
                    raise ValueError("联合重力：grav_ZL 两项须同时填写")
                g["ocean_lower"] = (lu, int(self._to_number(li)))
            su = self.vars["inv.grav_sed_up"].get().strip()
            sl = self.vars["inv.grav_sed_lo"].get().strip()
            si = self.vars["inv.grav_sed_iconv"].get().strip()
            if su or sl or si:
                if not (su and sl and si):
                    raise ValueError("联合重力：grav_ZS 三项须同时填写")
                g["sediment"] = (su, sl, int(self._to_number(si)))
            gd = self.vars["inv.grav_deriv"].get().strip()
            if gd:
                g["deriv"] = gd
            gw = self.vars["inv.grav_weight"].get().strip()
            if gw:
                g["weight_grav"] = self._to_number(gw)
            gz = self.vars["inv.grav_z0"].get().strip()
            if gz:
                g["z0"] = self._to_number(gz)
            gk = self.vars["inv.grav_dws"].get().strip()
            if gk:
                g["grav_dws"] = gk
            gcut = self.vars["inv.grav_cutoff"].get().strip()
            if gcut:
                g["cutoff"] = gcut
            kwargs["gravity_opts"] = g

        vl = self.vars["inv.verbose_level"].get().strip()
        if vl:
            val = self._to_number(vl)
            if isinstance(val, (int, float)):
                if val > 0:
                    kwargs["verbose"] = True
                    kwargs["verbose_level"] = val
            else:
                raise ValueError("inv.verbose_level 需要数值（0 表示不启用 verbose，>0 表示启用 -V[level]）")

        return mesh, data, kwargs

    def _collect_tt_forward_args(self) -> tuple[str, str | None, dict]:
        smesh = self.vars["fwd.smesh"].get().strip()
        geom = self.vars["fwd.geom"].get().strip() or None
        kwargs: dict = {}

        refl = self.vars["fwd.refl_file"].get().strip()
        if refl:
            kwargs["refl_file"] = refl

        if self.vars["fwd.do_full_refl"].get():
            kwargs["do_full_refl"] = True

        numeric = {}
        for key in ["xorder", "zorder", "clen", "nintp"]:
            raw = self.vars[f"fwd.{key}"].get().strip()
            if raw:
                numeric[key] = self._to_number(raw)
        cg = self.vars["fwd.bend_cg_tol"].get().strip()
        br = self.vars["fwd.bend_br_tol"].get().strip()
        if cg:
            numeric["tol1"] = self._to_number(cg)
        if br:
            numeric["tol2"] = self._to_number(br)
        kwargs.update(numeric)

        vred = self.vars["fwd.vred"].get().strip()
        if vred:
            kwargs["vred"] = self._to_number(vred)

        out_opts = {}
        map_out = {
            "elements": "out_elements",
            "ttime": "out_ttime",
            "obs_ttime": "out_obs_ttime",
            "ray": "out_ray",
            "source": "out_source",
            "vgrid": "out_vgrid",
            "diff": "out_diff",
        }
        for k, ui_key in map_out.items():
            raw = self.vars[f"fwd.{ui_key}"].get().strip()
            if raw:
                out_opts[k] = raw
        if out_opts:
            kwargs["out_opts"] = out_opts

        sub_keys = ("sub_west", "sub_east", "sub_south", "sub_north", "sub_dx", "sub_dz")
        sub_vals = [self.vars[f"fwd.{k}"].get().strip() for k in sub_keys]
        if any(sub_vals):
            if not all(sub_vals):
                raise ValueError("tt_forward: vgrid -i 六项 west/east/south/north/dx/dz 须同时填写")
            kwargs["vgrid_subregion"] = tuple(self._to_number(x) for x in sub_vals)

        cf = self.vars["fwd.clock_file"].get().strip()
        if cf:
            kwargs["clock_file"] = cf

        if self.vars["fwd.graph_only"].get():
            kwargs["graph_only"] = True
        if self.vars["fwd.omit_air_water"].get():
            kwargs["omit_air_water"] = True

        fvl = self.vars["fwd.verbose_level"].get().strip()
        if fvl:
            val = self._to_number(fvl)
            if isinstance(val, (int, float)):
                if val > 0:
                    kwargs["verbose"] = True
                    kwargs["verbose_level"] = val
            else:
                raise ValueError(
                    "fwd.verbose_level 需要数值（0 或留空表示不启用 verbose，>0 表示启用 -V[level]）"
                )

        return smesh, geom, kwargs

    def _collect_stat_smesh_kwargs(self) -> dict:
        kwargs: dict = {
            "mode": self.vars["stat.mode"].get(),
            "cmd_type": self.vars["stat.cmd_type"].get(),
        }
        for key in [
            "list_file",
            "mesh_file",
            "ave_file",
            "ave_x",
            "window_len",
            "xmin",
            "xmax",
            "dx",
            "top_bound",
            "bot_bound",
            "mid_bound",
            "pt_corr",
            "exclude_top_bound",
            "exclude_bot_bound",
        ]:
            raw = self.vars[f"stat.{key}"].get().strip()
            if raw:
                if key.endswith("_file") or key.endswith("_bound"):
                    kwargs[key] = raw
                else:
                    kwargs[key] = raw if key == "pt_corr" else self._to_number(raw)
        rn = self.vars["stat.refl_nnodes"].get().strip()
        if rn:
            kwargs["refl_nnodes"] = self._to_number(rn)
        vr = self.vars["stat.vrepl"].get().strip()
        if vr:
            kwargs["vrepl"] = self._to_number(vr)
        for key in ("abs_xmin", "abs_xmax", "exclude_cxmin", "exclude_cxmax"):
            raw = self.vars[f"stat.{key}"].get().strip()
            if raw:
                kwargs[key] = self._to_number(raw)
        if self.vars["stat.verbose"].get():
            kwargs["verbose"] = True
        return kwargs

    def _collect_edit_smesh_args(self) -> tuple[str, str, dict]:
        smesh_file = self.vars["edit.smesh_file"].get().strip()
        cmd_type = self.vars["edit.cmd_type"].get()
        kwargs: dict = {}
        path_keys = {
            "paste_file",
            "prof_file",
            "remove_bg_file",
            "moho_file",
            "base_file",
            "corr_file",
            "upper_bound",
        }
        for key in [
            "paste_file",
            "prof_file",
            "remove_bg_file",
            "h_len",
            "v_len",
            "mx",
            "mz",
            "amp",
            "xmin",
            "xmax",
            "zmin",
            "zmax",
            "x0",
            "z0",
            "Lh",
            "Lv",
            "seed",
            "nrand",
            "N",
            "dx",
            "dz",
            "vel",
            "moho_file",
            "k",
            "base_file",
            "corr_file",
            "upper_bound",
        ]:
            raw = self.vars[f"edit.{key}"].get().strip()
            if raw:
                kwargs[key] = raw if key in path_keys else self._to_number(raw)
        return smesh_file, cmd_type, kwargs

    def _flush_entry_focus(self) -> None:
        """点击「运行/预览」时焦点常留在 Entry 上，部分平台最后一次键入未进 StringVar；先移焦再收集参数。"""
        try:
            w = self.root.focus_get()
        except tk.TclError:
            return
        if w is not None and w is not self.root:
            try:
                self.root.focus_set()
            except tk.TclError:
                pass
        try:
            self.root.update_idletasks()
        except tk.TclError:
            pass

    def _validate_work_dir(self, *, normalize_paths: bool = True) -> Path:
        self._flush_entry_focus()
        if normalize_paths:
            self._normalize_all_file_path_vars(show_warn_outside=False)
        work_dir = Path(self.vars["work_dir"].get().strip() or str(Path.cwd())).resolve()
        if not work_dir.exists():
            raise ValueError(f"work_dir 不存在: {work_dir}")
        return work_dir

    def _get_tomo(self) -> TomoAnd:
        bin_path = self.vars["bin_path"].get().strip() or None
        return TomoAnd(bin_path=bin_path)

    def _pick_plot_smesh_cmap(self) -> None:
        try:
            work = self._validate_work_dir()
        except ValueError as e:
            messagebox.showerror("选择色标", str(e))
            return
        path = filedialog.askopenfilename(
            title="选择 GMT 色标 .cpt",
            initialdir=str(work),
            filetypes=[("CPT", "*.cpt"), ("所有文件", "*.*")],
        )
        if path:
            self.vars["gui.plot_smesh_cmap"].set(path)

    def _resolve_plot_smesh_cmap(self, work: Path) -> str:
        """返回 matplotlib 色标名，或存在的 .cpt 绝对路径；空则 seismic。"""
        try:
            s = self.vars["gui.plot_smesh_cmap"].get().strip()
        except (KeyError, tk.TclError):
            s = ""
        if not s:
            return "seismic"
        p = Path(s)
        if p.suffix.lower() == ".cpt":
            full = p if p.is_absolute() else (work / p)
            try:
                full = full.resolve()
            except OSError:
                full = p
            if full.is_file():
                return str(full)
            messagebox.showwarning(
                "smesh 色标",
                f"找不到 CPT 文件，已改用默认 seismic：\n{full}",
                parent=self.root,
            )
            return "seismic"
        return s

    def _resolve_optional_refl_path(self, work: Path) -> str | None:
        """当用户在「反射界面」对话框中选「否」时：使用 tt_forward / tt_inverse 页已填且存在的 refl_file。"""
        for key in ("fwd.refl_file", "inv.refl_file"):
            try:
                s = self.vars[key].get().strip()
            except (KeyError, tk.TclError):
                continue
            if not s:
                continue
            p = Path(s)
            full = p if p.is_absolute() else (work / p)
            if full.is_file():
                return str(full)
        return None

    def plot_smesh_velocity(self) -> None:
        """读取 smesh，在主线程弹出**单个** Matplotlib 图窗（嵌入 Tk）；并保存 PNG 到 work_dir。"""
        try:
            work = self._validate_work_dir()
        except ValueError as e:
            messagebox.showerror("绘制 smesh", str(e))
            return

        guess = ""
        try:
            guess = self.vars["fwd.smesh"].get().strip()
        except (KeyError, tk.TclError):
            pass
        if not guess:
            try:
                guess = self.vars["gen.smesh_out"].get().strip()
            except (KeyError, tk.TclError):
                pass

        initial_path: Path | None = None
        if guess:
            gp = Path(guess)
            cand = gp if gp.is_absolute() else (work / gp)
            if cand.is_file():
                initial_path = cand

        init_dir = str(initial_path.parent if initial_path else work)
        fd_kw: dict = {
            "title": "选择 smesh 文件",
            "initialdir": init_dir,
            "filetypes": [
                ("smesh / 数据", "*.smesh *.dat"),
                ("所有文件", "*.*"),
            ],
        }
        if initial_path is not None:
            fd_kw["initialfile"] = initial_path.name

        path = filedialog.askopenfilename(**fd_kw)
        if not path:
            return

        choice = messagebox.askyesnocancel(
            "反射界面",
            "是否叠加 tomo2d -F 反射界面？\n\n"
            "• 是：手动选择反射界面文本文件（每行 x z）；文件框默认打开与 smesh 相同目录\n"
            "• 否：若 tt_forward / tt_inverse 页的 refl_file 已填且文件存在，则自动叠加\n"
            "• 取消：不绘制反射界面",
            parent=self.root,
        )
        if choice is None:
            refl_path: str | None = None
        elif choice:
            smesh_dir = str(Path(path).resolve().parent)
            rp = filedialog.askopenfilename(
                title="反射界面文件（每行 x z）",
                initialdir=smesh_dir,
                filetypes=[("文本", "*.dat *.txt *.*"), ("所有文件", "*.*")],
            )
            refl_path = rp if rp else None
        else:
            refl_path = self._resolve_optional_refl_path(work)

        plot_title = "绘制 smesh"
        started_at = self._start_run_elapsed_timer(plot_title)
        self._log(f"[绘制 smesh] 读取: {path}")
        if refl_path:
            self._log(f"[绘制 smesh] 叠加反射界面: {refl_path}")
        else:
            self._log("[绘制 smesh] 未叠加反射界面（取消或未选文件 / 表单无有效 refl）")

        try:
            try:
                from ...model_building.tomoform import SlownessMesh2D, load_tomo2d_interface_file
                from ...visualization.show_model import GridModelVisualizer
            except ImportError:
                from pyAOBS.model_building.tomoform import SlownessMesh2D, load_tomo2d_interface_file
                from pyAOBS.visualization.show_model import GridModelVisualizer

            mesh = SlownessMesh2D.from_file(path)
            ds = mesh.to_xarray()
            extra: list | None = None
            if refl_path:
                rx, rz = load_tomo2d_interface_file(refl_path)
                extra = [
                    {
                        "x": rx,
                        "z": rz,
                        "label": Path(refl_path).name,
                        "color": "crimson",
                        "linewidth": 1.8,
                        "linestyle": "--",
                    }
                ]

            cmap_use = self._resolve_plot_smesh_cmap(work)
            win = tk.Toplevel(self.root)
            win.title("smesh 速度模型")
            win.geometry("900x560")
            self._register_mpl_toplevel(win)
            viz = GridModelVisualizer(output_dir=str(work))
            viz.embed_velocity_in_tk(
                win,
                ds,
                savefig_directory=str(Path(path).resolve().parent),
                figsize=(10, 5.0),
                cmap=cmap_use,
                model=mesh,
                extra_interfaces=extra,
                plot_interfaces=True,
                title="Velocity Model",
                colorbar_label="Velocity (km/s)",
                interface_linewidth=1.0,
                interface_linestyle="-",
            )
            self._cancel_run_elapsed_timer()
            self._status_set_finished(plot_title, True, started_at)
            self._log_run_elapsed(plot_title, started_at)
            self._log(
                f"[绘制 smesh] 已打开 Matplotlib 窗口（工具栏可保存 PNG）；色标: {cmap_use}"
            )
        except Exception as e:
            self._cancel_run_elapsed_timer()
            self._status_set_finished(plot_title, False, started_at)
            self._log_run_elapsed(plot_title, started_at)
            self._log(f"[绘制 smesh] 失败: {e}")
            messagebox.showerror("绘制 smesh 失败", str(e))

    def _show_tt_inverse_logfile_format_help(self) -> None:
        """弹出窗口：说明 ``tt_inverse -L`` 日志数据行各列含义（与 ``inverse.cc`` 一致）。"""
        try:
            from .help_docs import TomoHelp
        except ImportError:
            from pyAOBS.modeling.tomo2d.help_docs import TomoHelp

        body = TomoHelp.tt_inverse_logfile_format_help().strip()
        win = tk.Toplevel(self.root)
        win.title("tt_inverse -L 日志文件列说明")
        win.geometry("780x580")
        win.minsize(520, 360)
        fr = ttk.Frame(win, padding=8)
        fr.pack(fill=tk.BOTH, expand=True)
        txt_fr = ttk.Frame(fr)
        txt_fr.pack(fill=tk.BOTH, expand=True)
        mono_sz = max(11, getattr(self, "_base_font_size", 14) - 1)
        tx = tk.Text(
            txt_fr,
            wrap=tk.WORD,
            font=(self._mono_font_family, mono_sz),
            width=88,
            height=28,
        )
        sb = ttk.Scrollbar(txt_fr, orient=tk.VERTICAL, command=tx.yview)
        tx.configure(yscrollcommand=sb.set)
        tx.grid(row=0, column=0, sticky=tk.NSEW)
        sb.grid(row=0, column=1, sticky=tk.NS)
        txt_fr.columnconfigure(0, weight=1)
        txt_fr.rowconfigure(0, weight=1)
        self._apply_terminal_text_theme(tx, mono_size=mono_sz)
        tx.insert("1.0", body)
        self._bind_text_copy_helpers(tx)
        bf = ttk.Frame(fr)
        bf.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(bf, text="关闭", command=win.destroy).pack(side=tk.RIGHT)

    def _open_program_help_dialog(self) -> None:
        """弹出程序帮助文档窗口（按模块切换）。"""
        try:
            from .help_docs import TomoHelp
        except ImportError:
            from pyAOBS.modeling.tomo2d.help_docs import TomoHelp

        sections = [
            ("Python 封装总览", TomoHelp.python_wrapper_help),
            ("gen_smesh", TomoHelp.gen_smesh_help),
            ("gen_damp", TomoHelp.gen_damp_help),
            ("gen_vcorr", TomoHelp.gen_vcorr_help),
            ("tt_forward", TomoHelp.tt_forward_help),
            ("tt_inverse", TomoHelp.tt_inverse_help),
            ("tt_inverse -L 日志列说明", TomoHelp.tt_inverse_logfile_format_help),
            ("stat_smesh", TomoHelp.stat_smesh_help),
            ("edit_smesh / edit_smesh_HHB", TomoHelp.edit_smesh_help),
        ]
        sec_map = {name: fn for name, fn in sections}
        default_section = self._default_help_section_from_active_tab()
        if default_section not in sec_map:
            default_section = sections[0][0]

        win = tk.Toplevel(self.root)
        win.title("程序帮助文档")
        win.geometry("860x640")
        win.minsize(620, 420)

        fr = ttk.Frame(win, padding=8)
        fr.pack(fill=tk.BOTH, expand=True)
        row = ttk.Frame(fr)
        row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(row, text="帮助章节:").pack(side=tk.LEFT)
        v_sec = tk.StringVar(value=default_section)
        cb = ttk.Combobox(
            row,
            textvariable=v_sec,
            values=[name for name, _ in sections],
            width=34,
            state="readonly",
        )
        cb.pack(side=tk.LEFT, padx=(6, 0))

        txt_fr = ttk.Frame(fr)
        txt_fr.pack(fill=tk.BOTH, expand=True)
        mono_sz = max(11, getattr(self, "_base_font_size", 14) - 1)
        tx = tk.Text(
            txt_fr,
            wrap=tk.WORD,
            font=(self._mono_font_family, mono_sz),
            width=96,
            height=32,
        )
        sb = ttk.Scrollbar(txt_fr, orient=tk.VERTICAL, command=tx.yview)
        tx.configure(yscrollcommand=sb.set)
        tx.grid(row=0, column=0, sticky=tk.NSEW)
        sb.grid(row=0, column=1, sticky=tk.NS)
        txt_fr.columnconfigure(0, weight=1)
        txt_fr.rowconfigure(0, weight=1)
        # 帮助文档窗口使用浅色阅读主题（不复用右侧日志终端暗色主题）
        try:
            tx.configure(
                background="#ffffff",
                foreground="#1f2937",
                insertbackground="#1f2937",
                selectbackground="#cfe8ff",
                selectforeground="#111827",
                highlightthickness=1,
                highlightbackground="#d1d5db",
                highlightcolor="#93c5fd",
                borderwidth=0,
                relief=tk.FLAT,
                spacing1=max(2, int(round(mono_sz * 0.22))),
                spacing2=max(1, int(round(mono_sz * 0.12))),
                spacing3=max(2, int(round(mono_sz * 0.22))),
            )
        except tk.TclError:
            pass
        self._bind_text_copy_helpers(tx)

        def refresh_section(_event=None):
            key = v_sec.get().strip()
            fn = sec_map.get(key)
            body = ""
            try:
                body = (fn().strip() if fn else "")
            except Exception as e:
                body = f"加载帮助失败: {e}"
            tx.configure(state=tk.NORMAL)
            tx.delete("1.0", tk.END)
            tx.insert("1.0", body)
            tx.configure(state=tk.DISABLED)

        cb.bind("<<ComboboxSelected>>", refresh_section)
        refresh_section()

        bf = ttk.Frame(fr)
        bf.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(bf, text="关闭", command=win.destroy).pack(side=tk.RIGHT)

    def _resolve_path_for_analysis(self, path_str: str) -> Path:
        """相对路径则相对当前 work_dir。"""
        s = (path_str or "").strip()
        if not s:
            raise ValueError("路径为空")
        p = Path(s).expanduser()
        if p.is_absolute():
            return p.resolve()
        return (self._work_dir_resolved() / p).resolve()

    def _embed_mpl_figure(self, master: tk.Toplevel, fig, title: str) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        try:
            from . import tt_inverse_log_analysis as _tila

            _tila.ensure_matplotlib_cjk_font()
        except Exception:
            pass
        win = tk.Toplevel(master)
        win.title(title)
        win.geometry("920x680")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        self._log(f"[反演分析] 已打开图表: {title}", write_file=False)

        self._register_mpl_toplevel(win)

        def _close_embed() -> None:
            try:
                toolbar.destroy()
            except Exception:
                pass
            try:
                canvas.get_tk_widget().destroy()
            except Exception:
                pass
            try:
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception:
                pass
            try:
                self._mpl_embed_toplevels.remove(win)
            except ValueError:
                pass
            try:
                win.destroy()
            except tk.TclError:
                pass

        win.protocol("WM_DELETE_WINDOW", _close_embed)

    def _open_tt_inverse_analysis_dialog(self) -> None:
        try:
            from . import tt_inverse_log_analysis as tila
        except ImportError:
            from pyAOBS.modeling.tomo2d import tt_inverse_log_analysis as tila

        dlg = tk.Toplevel(self.root)
        dlg.title("反演结果分析（tt_inverse -L 日志）")
        dlg.geometry("700x520")
        dlg.minsize(560, 420)

        nb = ttk.Notebook(dlg, padding=6)
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        tab1 = ttk.Frame(nb, padding=6)
        tab2 = ttk.Frame(nb, padding=6)
        nb.add(tab1, text="单日志")
        nb.add(tab2, text="多日志")

        v_single = tk.StringVar(value="")
        ttk.Label(
            tab1,
            text="tt_inverse.log 路径（可相对 work_dir；运行包内多为 outputs/tt_inverse.log）：",
        ).pack(anchor=tk.W)
        row1 = ttk.Frame(tab1)
        row1.pack(fill=tk.X, pady=(4, 8))
        en1 = ttk.Entry(row1, textvariable=v_single, width=62, font=self._entry_font)
        en1.pack(side=tk.LEFT, fill=tk.X, expand=True)

        def browse_single():
            wd = str(self._work_dir_resolved())
            path = filedialog.askopenfilename(
                title="选择 tt_inverse 日志文件",
                initialdir=wd,
                filetypes=[
                    ("log", "*.log"),
                    ("文件名含 log", "*log*"),
                    ("文本", "*.txt"),
                    ("所有", "*.*"),
                ],
            )
            if path:
                try:
                    rel = self._to_workdir_relative(path, show_warn_outside=False)
                except Exception:
                    rel = path
                v_single.set(rel)

        ttk.Button(row1, text="浏览…", command=browse_single, width=6).pack(
            side=tk.LEFT, padx=(6, 0)
        )

        def run_single():
            try:
                p = self._resolve_path_for_analysis(v_single.get())
                if not p.is_file():
                    raise FileNotFoundError(f"找不到文件: {p}")
                rows = tila.parse_tt_inverse_log(p)
                fig = tila.build_figure_single_log(rows, title=str(p.name))
                self._embed_mpl_figure(dlg, fig, f"单日志: {p.name}")
            except Exception as e:
                messagebox.showerror("反演分析", str(e), parent=dlg)

        ttk.Button(tab1, text="绘制 RMS / χ² 随迭代", command=run_single).pack(
            anchor=tk.W, pady=(8, 0)
        )

        ttk.Label(
            tab2,
            text="每行一个日志路径(可相对 work_dir), 或使用下方[添加文件]按钮:",
        ).pack(anchor=tk.W)
        fm = ttk.Frame(tab2)
        fm.pack(fill=tk.BOTH, expand=True, pady=(4, 6))
        tx_m = tk.Text(fm, height=12, width=72, font=self._entry_font)
        sb_m = ttk.Scrollbar(fm, orient=tk.VERTICAL, command=tx_m.yview)
        tx_m.configure(yscrollcommand=sb_m.set)
        tx_m.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb_m.pack(side=tk.RIGHT, fill=tk.Y)

        def add_multi_files():
            wd = str(self._work_dir_resolved())
            paths = filedialog.askopenfilenames(
                title="选择多个 tt_inverse 日志",
                initialdir=wd,
                filetypes=[
                    ("log", "*.log"),
                    ("文件名含 log", "*log*"),
                    ("文本", "*.txt"),
                    ("所有", "*.*"),
                ],
            )
            for path in paths:
                try:
                    rel = self._to_workdir_relative(path, show_warn_outside=False)
                except Exception:
                    rel = path
                tx_m.insert(tk.END, rel + "\n")

        ttk.Button(tab2, text="添加文件…", command=add_multi_files).pack(anchor=tk.W)
        row_w = ttk.Frame(tab2)
        row_w.pack(fill=tk.X, pady=(8, 4))
        ttk.Label(
            row_w,
            text="粗糙度权重 w (score=pred_chi*(1+w*R), pred_chi 为末步 LSQR pred chi2):",
        ).pack(side=tk.LEFT)
        v_w = tk.StringVar(value="0.001")
        ttk.Entry(row_w, textvariable=v_w, width=10).pack(side=tk.LEFT, padx=(6, 0))

        def short_label(s: str) -> str:
            p = Path(s.strip())
            name = p.name or p.as_posix()
            return name[:28] + ("…" if len(name) > 28 else "")

        def run_multi():
            try:
                wgt = float(v_w.get().strip() or "0.001")
            except ValueError:
                messagebox.showerror("反演分析", "粗糙度权重 w 须为数字", parent=dlg)
                return
            lines = [ln.strip() for ln in tx_m.get("1.0", tk.END).splitlines() if ln.strip()]
            if len(lines) < 2:
                messagebox.showerror(
                    "反演分析", "多日志分析至少需要 2 个日志路径", parent=dlg
                )
                return
            series: dict[str, list] = {}
            for i, ln in enumerate(lines):
                try:
                    p = self._resolve_path_for_analysis(ln)
                    if not p.is_file():
                        raise FileNotFoundError(f"找不到: {p}")
                    rows = tila.parse_tt_inverse_log(p)
                    if not rows:
                        raise ValueError(f"无有效数据行: {p}")
                    key = short_label(ln)
                    if key in series:
                        key = f"{key} [{i+1}]"
                    series[key] = rows
                except Exception as e:
                    messagebox.showerror("反演分析", str(e), parent=dlg)
                    return
            try:
                fig1 = tila.build_figure_multi_last_bars(series)
                self._embed_mpl_figure(dlg, fig1, "多日志：末步 RMS/χ² 对比")
                fig2 = tila.build_figure_multi_overlay(series)
                self._embed_mpl_figure(dlg, fig2, "多日志：叠画迭代曲线")
                fig3 = tila.build_figure_multi_pareto_and_score(series, rough_weight=wgt)
                self._embed_mpl_figure(dlg, fig3, "多日志：Pareto 与综合得分")
                fig4 = tila.build_figure_multi_param_influence(series)
                self._embed_mpl_figure(dlg, fig4, "多日志：反演参数与指标（平滑/阻尼）")
                fig5 = tila.build_figure_multi_summary_table(series, rough_weight=wgt)
                self._embed_mpl_figure(dlg, fig5, "多日志：末步参数与指标汇总表")
            except Exception as e:
                messagebox.showerror("反演分析", str(e), parent=dlg)

        ttk.Button(
            tab2,
            text="生成多日志分析（末步+叠画+Pareto+参数影响+汇总表）",
            command=run_multi,
        ).pack(anchor=tk.W, pady=(4, 0))

    def _preview_append_resolved_cmdline(self, text: str, resolve) -> str:
        """在 Python 调用预览后追加与 subprocess 一致的 argv 命令行（shlex.join）。"""
        try:
            cmd = resolve()
            if cmd:
                return f"{text}\n\n# 解析后命令行:\n{_format_resolved_cmdline(cmd)}"
            return (
                f"{text}\n\n# 解析后命令行: （必选参数不齐或与「仅打印帮助」路径一致，未拼可执行 argv）"
            )
        except Exception as e:
            return f"{text}\n\n# 解析后命令行: （无法解析: {e}）"

    def _resolve_pipeline_step_cmdline(self, tomo: TomoAnd, step_name: str, spec: dict):
        if step_name == "gen_smesh":
            return tomo.resolve_cmdline_gen_smesh(**spec["kwargs"])
        if step_name == "gen_damp":
            return tomo.resolve_cmdline_gen_damp(**spec["kwargs"])
        if step_name == "gen_vcorr":
            return tomo.resolve_cmdline_gen_vcorr(**spec["kwargs"])
        if step_name == "tt_forward":
            return tomo.resolve_cmdline_tt_forward(
                smesh=spec["smesh"], geom=spec.get("geom"), **spec["kwargs"]
            )
        if step_name == "tt_inverse":
            try:
                ub = bool(self.vars["inv.use_repro_bundle"].get())
            except (KeyError, tk.TclError):
                ub = True
            m, d = spec["mesh"], spec["data"]
            kw = dict(spec["kwargs"])
            if ub:
                try:
                    from .tt_inverse_bundle import bundle_argv_preview_paths
                except ImportError:
                    from pyAOBS.modeling.tomo2d.tt_inverse_bundle import (
                        bundle_argv_preview_paths,
                    )
                m, d, kw = bundle_argv_preview_paths(m, d, kw)
            return tomo.resolve_cmdline_tt_inverse(mesh=m, data=d, **kw)
        return None

    def preview_gen_smesh(self) -> None:
        try:
            work = self._validate_work_dir()
            kwargs = self._collect_gen_smesh_kwargs()
            text = "tomo.gen_smesh(\n" + ",\n".join(f"  {k}={kwargs[k]!r}" for k in sorted(kwargs.keys())) + "\n)"
            if not str(kwargs.get("out_file") or "").strip():
                text += (
                    "\n\n# 未传 out_file：GUI 下点击「运行 gen_smesh」将被禁止；"
                    "请填写「smesh 输出文件」或仅在脚本中使用 out_file=。"
                )
            text += "\n\n" + self._gen_smesh_preview_path_audit(work, kwargs)
            tomo = self._get_tomo()
            self._set_preview(
                self._preview_append_resolved_cmdline(
                    text, lambda: tomo.resolve_cmdline_gen_smesh(**kwargs)
                )
            )
        except Exception as e:
            messagebox.showerror("预览失败", str(e))

    def preview_tt_forward(self) -> None:
        try:
            work = self._validate_work_dir()
            smesh, geom, kwargs = self._collect_tt_forward_args()
            if not smesh:
                raise ValueError("tt_forward 需要 smesh")
            pieces = [f"  smesh={smesh!r}"]
            if geom is not None:
                pieces.append(f"  geom={geom!r}")
            pieces.extend(f"  {k}={kwargs[k]!r}" for k in sorted(kwargs.keys()))
            text = (
                "tomo.tt_forward(\n"
                + ",\n".join(pieces)
                + "\n)\n\n"
                + self._tt_forward_input_path_audit(work, smesh, geom, kwargs)
            )
            tomo = self._get_tomo()
            self._set_preview(
                self._preview_append_resolved_cmdline(
                    text,
                    lambda: tomo.resolve_cmdline_tt_forward(smesh=smesh, geom=geom, **kwargs),
                )
            )
        except Exception as e:
            messagebox.showerror("预览失败", str(e))

    def run_gen_smesh(self) -> None:
        started_at: float | None = None
        title = "gen_smesh"
        try:
            work = self._validate_work_dir()
            kwargs = self._collect_gen_smesh_kwargs()
            if not str(kwargs.get("out_file") or "").strip():
                messagebox.showerror(
                    "无法运行 gen_smesh",
                    "请先填写「smesh 输出文件」（相对 work_dir）。\n\n"
                    "未指定路径时程序不会把网格写入磁盘，本界面不允许在此情况下运行。\n"
                    "仅预览参数可不填；独立脚本仍可使用 tomo.gen_smesh(..., out_file=...)。",
                )
                return
            started_at = self._start_run_elapsed_timer(title)
            base = "运行: tomo.gen_smesh(...)\n" + json.dumps(kwargs, ensure_ascii=False, indent=2)
            tomo = self._get_tomo()
            self._set_preview(
                self._preview_append_resolved_cmdline(
                    base, lambda: tomo.resolve_cmdline_gen_smesh(**kwargs)
                )
            )
            self._run_in_thread(
                title, work, lambda tomo: tomo.gen_smesh(**kwargs), started_at=started_at
            )
        except Exception as e:
            if started_at is not None:
                self._cancel_run_elapsed_timer()
                self._status_set_finished(title, False, started_at)
                self._log_run_elapsed(title, started_at)
            messagebox.showerror("运行前检查失败", str(e))

    def run_tt_forward(self) -> None:
        started_at: float | None = None
        title = "tt_forward"
        try:
            work = self._validate_work_dir()
            smesh, geom, kwargs = self._collect_tt_forward_args()
            if not smesh:
                raise ValueError("tt_forward 需要 smesh")
            started_at = self._start_run_elapsed_timer(title)
            payload = {"smesh": smesh, "geom": geom, **kwargs}
            base = "运行: tomo.tt_forward(...)\n" + json.dumps(payload, ensure_ascii=False, indent=2)
            tomo = self._get_tomo()
            self._set_preview(
                self._preview_append_resolved_cmdline(
                    base,
                    lambda: tomo.resolve_cmdline_tt_forward(smesh=smesh, geom=geom, **kwargs),
                )
            )
            self._run_in_thread(
                title,
                work,
                lambda tomo: tomo.tt_forward(smesh=smesh, geom=geom, **kwargs),
                started_at=started_at,
            )
        except Exception as e:
            if started_at is not None:
                self._cancel_run_elapsed_timer()
                self._status_set_finished(title, False, started_at)
                self._log_run_elapsed(title, started_at)
            messagebox.showerror("运行前检查失败", str(e))

    def preview_gen_damp(self) -> None:
        try:
            work = self._validate_work_dir()
            kwargs = self._collect_gen_damp_kwargs()
            text = (
                "tomo.gen_damp(\n"
                + ",\n".join(f"  {k}={kwargs[k]!r}" for k in sorted(kwargs.keys()))
                + "\n)\n\n"
                + self._mesh_family_input_path_audit(work, kwargs, "gen_damp")
            )
            tomo = self._get_tomo()
            self._set_preview(
                self._preview_append_resolved_cmdline(
                    text, lambda: tomo.resolve_cmdline_gen_damp(**kwargs)
                )
            )
        except Exception as e:
            messagebox.showerror("预览失败", str(e))

    def run_gen_damp(self) -> None:
        started_at: float | None = None
        title = "gen_damp"
        try:
            work = self._validate_work_dir()
            kwargs = self._collect_gen_damp_kwargs()
            started_at = self._start_run_elapsed_timer(title)
            base = "运行: tomo.gen_damp(...)\n" + json.dumps(kwargs, ensure_ascii=False, indent=2)
            tomo = self._get_tomo()
            self._set_preview(
                self._preview_append_resolved_cmdline(
                    base, lambda: tomo.resolve_cmdline_gen_damp(**kwargs)
                )
            )
            self._run_in_thread(
                title, work, lambda tomo: tomo.gen_damp(**kwargs), started_at=started_at
            )
        except Exception as e:
            if started_at is not None:
                self._cancel_run_elapsed_timer()
                self._status_set_finished(title, False, started_at)
                self._log_run_elapsed(title, started_at)
            messagebox.showerror("运行前检查失败", str(e))

    def preview_gen_vcorr(self) -> None:
        try:
            work = self._validate_work_dir()
            kwargs = self._collect_gen_vcorr_kwargs()
            text = (
                "tomo.gen_vcorr(\n"
                + ",\n".join(f"  {k}={kwargs[k]!r}" for k in sorted(kwargs.keys()))
                + "\n)\n\n"
                + self._mesh_family_input_path_audit(work, kwargs, "gen_vcorr")
            )
            tomo = self._get_tomo()
            self._set_preview(
                self._preview_append_resolved_cmdline(
                    text, lambda: tomo.resolve_cmdline_gen_vcorr(**kwargs)
                )
            )
        except Exception as e:
            messagebox.showerror("预览失败", str(e))

    def run_gen_vcorr(self) -> None:
        started_at: float | None = None
        title = "gen_vcorr"
        try:
            work = self._validate_work_dir()
            kwargs = self._collect_gen_vcorr_kwargs()
            started_at = self._start_run_elapsed_timer(title)
            base = "运行: tomo.gen_vcorr(...)\n" + json.dumps(kwargs, ensure_ascii=False, indent=2)
            tomo = self._get_tomo()
            self._set_preview(
                self._preview_append_resolved_cmdline(
                    base, lambda: tomo.resolve_cmdline_gen_vcorr(**kwargs)
                )
            )
            self._run_in_thread(
                title, work, lambda tomo: tomo.gen_vcorr(**kwargs), started_at=started_at
            )
        except Exception as e:
            if started_at is not None:
                self._cancel_run_elapsed_timer()
                self._status_set_finished(title, False, started_at)
                self._log_run_elapsed(title, started_at)
            messagebox.showerror("运行前检查失败", str(e))

    def preview_tt_inverse(self) -> None:
        try:
            work = self._validate_work_dir()
            mesh, data, kwargs = self._collect_tt_inverse_args()
            if not mesh or not data:
                raise ValueError("tt_inverse 需要 mesh 与 data")
            pieces = [f"  mesh={mesh!r}", f"  data={data!r}"]
            pieces.extend(f"  {k}={kwargs[k]!r}" for k in sorted(kwargs.keys()))
            text = "tomo.tt_inverse(\n" + ",\n".join(pieces) + "\n)"
            try:
                ub = bool(self.vars["inv.use_repro_bundle"].get())
            except (KeyError, tk.TclError):
                ub = True
            mesh_r, data_r, kw_r = mesh, data, kwargs
            if ub:
                try:
                    from .tt_inverse_bundle import bundle_argv_preview_paths
                except ImportError:
                    from pyAOBS.modeling.tomo2d.tt_inverse_bundle import (
                        bundle_argv_preview_paths,
                    )
                mesh_r, data_r, kw_r = bundle_argv_preview_paths(mesh, data, kwargs)
                text += (
                    "\n\n# 已勾选「可复现运行包」：运行时会新建 work_dir/runs/…/ 作为 run_dir，"
                    "输入快照到 inputs/、-L/-O/-K 等到 outputs/，并写 manifest。"
                    "下方「解析后命令行」按 proc_cwd=run_dir 拼出（与真实子进程一致；预览不创建目录）。"
                    "上表 tomo.tt_inverse(...) 仍为表单原始参数。"
                )
            else:
                text += (
                    "\n\n# 未勾选运行包：子进程 cwd = work_dir；"
                    "输出项留空则 TomoAnd 不传 -L/-O/-K（与直接运行一致）。"
                )
            text += "\n\n" + self._tt_inverse_input_path_audit(work, mesh, data, kwargs)
            tomo = self._get_tomo()
            self._set_preview(
                self._preview_append_resolved_cmdline(
                    text,
                    lambda: tomo.resolve_cmdline_tt_inverse(
                        mesh=mesh_r, data=data_r, **dict(kw_r)
                    ),
                )
            )
        except Exception as e:
            messagebox.showerror("预览失败", str(e))

    def _import_tt_inverse_bundle(self):
        try:
            from .tt_inverse_bundle import (
                build_tt_inverse_bundle,
                finalize_tt_inverse_manifest,
                write_tt_inverse_manifest,
            )
        except ImportError:
            from pyAOBS.modeling.tomo2d.tt_inverse_bundle import (
                build_tt_inverse_bundle,
                finalize_tt_inverse_manifest,
                write_tt_inverse_manifest,
            )
        return build_tt_inverse_bundle, write_tt_inverse_manifest, finalize_tt_inverse_manifest

    def _tt_inverse_bundle_run_label(self) -> str | None:
        try:
            s = self.vars["inv.bundle_run_label"].get().strip()
        except (KeyError, tk.TclError):
            return None
        return s or None

    def run_tt_inverse(self) -> None:
        started_at: float | None = None
        title = "tt_inverse"
        try:
            work = self._validate_work_dir()
            mesh, data, kwargs = self._collect_tt_inverse_args()
            if not mesh or not data:
                raise ValueError("tt_inverse 需要 mesh 与 data")
            started_at = self._start_run_elapsed_timer(title)
            payload = {"mesh": mesh, "data": data, **kwargs}
            base = "运行: tomo.tt_inverse(...)\n" + json.dumps(payload, ensure_ascii=False, indent=2)
            tomo = self._get_tomo()

            try:
                use_bundle = bool(self.vars["inv.use_repro_bundle"].get())
            except (KeyError, tk.TclError):
                use_bundle = True

            proc_cwd: Path | None = None
            manifest_path: Path | None = None
            mesh_r, data_r, kw_r = mesh, data, kwargs

            if use_bundle:
                bb, wm, fm = self._import_tt_inverse_bundle()
                br = bb(
                    work, mesh, data, kwargs, run_label=self._tt_inverse_bundle_run_label()
                )
                mesh_r, data_r, kw_r = br.mesh, br.data, br.kwargs
                argv = tomo.resolve_cmdline_tt_inverse(mesh=mesh_r, data=data_r, **kw_r)
                if not argv:
                    raise ValueError("无法解析 tt_inverse 命令行（运行包模式）")
                gui_profile_for_manifest = self._collect_gui_profile_dict()
                wm(
                    manifest_path=br.manifest_path,
                    work_dir=work,
                    run_dir=br.run_dir,
                    executable_resolved=argv[0],
                    argv=argv,
                    mesh=mesh_r,
                    data=data_r,
                    kwargs=kw_r,
                    inputs_rows=br.inputs_manifest,
                    status="running",
                    gui_profile=gui_profile_for_manifest,
                )
                proc_cwd = br.run_dir
                manifest_path = br.manifest_path
                self._log(f"[tt_inverse] 运行包目录: {br.run_dir}", write_file=False)

            if use_bundle:
                base = base + f"\n\n# 可复现运行包: proc_cwd = {proc_cwd}\n"

            self._set_preview(
                self._preview_append_resolved_cmdline(
                    base,
                    lambda: tomo.resolve_cmdline_tt_inverse(
                        mesh=mesh_r, data=data_r, **kw_r
                    ),
                )
            )

            kw_copy = dict(kw_r)

            def wf(res, exc):
                if manifest_path is not None:
                    fm(manifest_path, result=res, err=exc)

            self._run_in_thread(
                title,
                work,
                lambda t, mm=mesh_r, dd=data_r, kk=kw_copy: t.tt_inverse(
                    mesh=mm, data=dd, **kk
                ),
                proc_cwd=proc_cwd,
                worker_finally=wf if manifest_path else None,
                started_at=started_at,
            )
        except Exception as e:
            if started_at is not None:
                self._cancel_run_elapsed_timer()
                self._status_set_finished(title, False, started_at)
                self._log_run_elapsed(title, started_at)
            messagebox.showerror("运行前检查失败", str(e))

    def preview_stat_smesh(self) -> None:
        try:
            work = self._validate_work_dir()
            kwargs = self._collect_stat_smesh_kwargs()
            text = (
                "tomo.stat_smesh(\n"
                + ",\n".join(f"  {k}={kwargs[k]!r}" for k in sorted(kwargs.keys()))
                + "\n)\n\n"
                + self._stat_smesh_input_path_audit(work, kwargs)
            )
            tomo = self._get_tomo()
            self._set_preview(
                self._preview_append_resolved_cmdline(
                    text, lambda: tomo.resolve_cmdline_stat_smesh(**kwargs)
                )
            )
        except Exception as e:
            messagebox.showerror("预览失败", str(e))

    def run_stat_smesh(self) -> None:
        started_at: float | None = None
        title = "stat_smesh"
        try:
            work = self._validate_work_dir()
            kwargs = self._collect_stat_smesh_kwargs()
            started_at = self._start_run_elapsed_timer(title)
            base = "运行: tomo.stat_smesh(...)\n" + json.dumps(kwargs, ensure_ascii=False, indent=2)
            tomo = self._get_tomo()
            self._set_preview(
                self._preview_append_resolved_cmdline(
                    base, lambda: tomo.resolve_cmdline_stat_smesh(**kwargs)
                )
            )
            self._run_in_thread(
                title, work, lambda tomo: tomo.stat_smesh(**kwargs), started_at=started_at
            )
        except Exception as e:
            if started_at is not None:
                self._cancel_run_elapsed_timer()
                self._status_set_finished(title, False, started_at)
                self._log_run_elapsed(title, started_at)
            messagebox.showerror("运行前检查失败", str(e))

    def preview_edit_smesh(self) -> None:
        try:
            work = self._validate_work_dir()
            smesh_file, cmd_type, kwargs = self._collect_edit_smesh_args()
            if not smesh_file:
                raise ValueError("edit_smesh_HHB 需要 smesh_file")
            pieces = [f"  smesh_file={smesh_file!r}", f"  cmd_type={cmd_type!r}"]
            pieces.extend(f"  {k}={kwargs[k]!r}" for k in sorted(kwargs.keys()))
            text = (
                "tomo.edit_smesh(\n"
                + ",\n".join(pieces)
                + "\n)\n\n"
                + self._edit_smesh_input_path_audit(work, smesh_file, kwargs)
            )
            tomo = self._get_tomo()
            self._set_preview(
                self._preview_append_resolved_cmdline(
                    text,
                    lambda: tomo.resolve_cmdline_edit_smesh(
                        smesh_file=smesh_file, cmd_type=cmd_type, **kwargs
                    ),
                )
            )
        except Exception as e:
            messagebox.showerror("预览失败", str(e))

    def run_edit_smesh(self) -> None:
        started_at: float | None = None
        title = "edit_smesh_HHB"
        try:
            work = self._validate_work_dir()
            smesh_file, cmd_type, kwargs = self._collect_edit_smesh_args()
            if not smesh_file:
                raise ValueError("edit_smesh_HHB 需要 smesh_file")
            started_at = self._start_run_elapsed_timer(title)
            payload = {"smesh_file": smesh_file, "cmd_type": cmd_type, **kwargs}
            base = "运行: tomo.edit_smesh(...)\n" + json.dumps(payload, ensure_ascii=False, indent=2)
            tomo = self._get_tomo()
            self._set_preview(
                self._preview_append_resolved_cmdline(
                    base,
                    lambda: tomo.resolve_cmdline_edit_smesh(
                        smesh_file=smesh_file, cmd_type=cmd_type, **kwargs
                    ),
                )
            )
            self._run_in_thread(
                title,
                work,
                lambda tomo: tomo.edit_smesh(smesh_file=smesh_file, cmd_type=cmd_type, **kwargs),
                started_at=started_at,
            )
        except Exception as e:
            if started_at is not None:
                self._cancel_run_elapsed_timer()
                self._status_set_finished(title, False, started_at)
                self._log_run_elapsed(title, started_at)
            messagebox.showerror("运行前检查失败", str(e))

    def preview_tx_convert(self) -> None:
        try:
            work = self._validate_work_dir()
            station, txin, dout, gout = self._tx_convert_resolved_paths(work)
            refr = parse_phase_set(self.vars["tx.refr_phases"].get())
            refl = parse_phase_set(self.vars["tx.refl_phases"].get())
            text = (
                "from pathlib import Path\n"
                "from pyAOBS.modeling.tomo2d.tx2tomo2d import convert_tx_in_to_tomo2d\n\n"
                "convert_tx_in_to_tomo2d(\n"
                f"    Path({str(station)!r}),\n"
                f"    Path({str(txin)!r}),\n"
                f"    Path({str(dout)!r}),\n"
                f"    Path({str(gout)!r}),\n"
                f"    refr_phases={set(refr)!r},\n"
                f"    refl_phases={set(refl)!r},\n"
                ")\n\n"
                f"# work_dir: {work}\n"
                f"# 折射震相: {sorted(refr) if refr else '（空）'}\n"
                f"# 反射震相: {sorted(refl) if refl else '（空）'}\n"
            )
            text += "\n" + self._tx_convert_input_path_audit(work, station, txin, dout, gout)
            self._set_preview(text)
        except Exception as e:
            messagebox.showerror("预览失败", str(e))

    def run_tx_convert(self) -> None:
        started_at: float | None = None
        title = "tx.in→tomo2d"
        try:
            work = self._validate_work_dir()
            station, txin, dout, gout = self._tx_convert_resolved_paths(work)
            if not station.is_file():
                raise ValueError(f"台站文件不存在: {station}")
            if not txin.is_file():
                raise ValueError(f"tx.in 不存在: {txin}")
            refr = parse_phase_set(self.vars["tx.refr_phases"].get())
            refl = parse_phase_set(self.vars["tx.refl_phases"].get())
            payload = {
                "station": str(station),
                "tx_in": str(txin),
                "data_out": str(dout),
                "geom_out": str(gout),
                "refr_phases": sorted(refr),
                "refl_phases": sorted(refl),
            }
            started_at = self._start_run_elapsed_timer(title)
            self._log(f"[{title}] 开始，工作目录: {work}")
            self._set_preview("运行: tx.in→tomo2d\n" + json.dumps(payload, ensure_ascii=False, indent=2))

            def worker():
                try:
                    stats = convert_tx_in_to_tomo2d(
                        station,
                        txin,
                        dout,
                        gout,
                        refr_phases=refr,
                        refl_phases=refl,
                    )
                    self.root.after(
                        0,
                        lambda s=stats, st=started_at: self._on_tx_convert_done(
                            None, s, started_at=st
                        ),
                    )
                except Exception as e:
                    self.root.after(
                        0,
                        lambda exc=e, st=started_at: self._on_tx_convert_done(
                            exc, None, started_at=st
                        ),
                    )

            threading.Thread(target=worker, daemon=True).start()
        except Exception as e:
            if started_at is not None:
                self._cancel_run_elapsed_timer()
                self._status_set_finished(title, False, started_at)
                self._log_run_elapsed(title, started_at)
            messagebox.showerror("运行前检查失败", str(e))

    def _on_tx_convert_done(
        self, err: Exception | None, stats, *, started_at: float | None = None
    ) -> None:
        self._cancel_run_elapsed_timer()
        title = "tx.in→tomo2d"
        if err is not None:
            self._status_set_finished(title, False, started_at)
            self._log_run_elapsed(title, started_at)
            self._log(f"[{title}] 失败: {err}")
            messagebox.showerror("tx 转换失败", str(err))
            return
        assert stats is not None
        self._status_set_finished(title, True, started_at)
        self._log_run_elapsed(title, started_at)
        self._log(
            f"[{title}] 成功: station 行数={stats.station_rows}, "
            f"nshot={stats.nshot}, ntime={stats.ntime}\n"
            f"  data → {stats.data_path}\n"
            f"  geom → {stats.geom_path}"
        )

    def _pipeline_links(self) -> dict[str, str]:
        return {
            "smesh": self.vars["pipe.link_smesh"].get().strip(),
            "damp": self.vars["pipe.link_damp"].get().strip(),
            "vcorr_v": self.vars["pipe.link_vcorr_v"].get().strip(),
            "vcorr_d": self.vars["pipe.link_vcorr_d"].get().strip(),
        }

    def _auto_wire_pipeline_targets(self, recipe: str, payload: dict) -> dict:
        """
        将上游生成文件桥接到下游参数：
        - 仅在 pipe.auto_wire=True 且目标参数为空时填入
        - 已有显式参数优先，不覆盖
        """
        if not self.vars["pipe.auto_wire"].get():
            return payload
        links = self._pipeline_links()

        # 1) smesh -> tt_forward.smesh / tt_inverse.mesh
        if "tt_forward" in recipe and not payload.get("smesh") and links["smesh"]:
            payload["smesh"] = links["smesh"]
            self.vars["fwd.smesh"].set(links["smesh"])
        if "tt_inverse" in recipe and not payload.get("mesh") and links["smesh"]:
            payload["mesh"] = links["smesh"]
            self.vars["inv.mesh"].set(links["smesh"])

        # 2) gen_damp 产物 -> tt_inverse.damp_opts.damp_v_fn
        if "gen_damp" in recipe and links["damp"]:
            damp_opts = payload.get("inv_kwargs", {}).get("damp_opts", {}) or {}
            if "damp_v_fn" not in damp_opts:
                damp_opts["damp_v_fn"] = links["damp"]
                payload["inv_kwargs"]["damp_opts"] = damp_opts
                self.vars["inv.damp_v_fn"].set(links["damp"])

        # 3) gen_vcorr 产物 -> tt_inverse.smooth_opts.corr_v_fn/corr_d_fn
        if "gen_vcorr" in recipe:
            smooth_opts = payload.get("inv_kwargs", {}).get("smooth_opts", {}) or {}
            changed = False
            if links["vcorr_v"] and "corr_v_fn" not in smooth_opts:
                smooth_opts["corr_v_fn"] = links["vcorr_v"]
                self.vars["inv.smooth_corr_v_fn"].set(links["vcorr_v"])
                changed = True
            if links["vcorr_d"] and "corr_d_fn" not in smooth_opts:
                smooth_opts["corr_d_fn"] = links["vcorr_d"]
                self.vars["inv.smooth_corr_d_fn"].set(links["vcorr_d"])
                changed = True
            if changed:
                payload["inv_kwargs"]["smooth_opts"] = smooth_opts

        return payload

    def _build_pipeline_plan(self) -> list[tuple[str, dict]]:
        recipe = self.vars["pipe.recipe"].get()
        if recipe == "gen_smesh -> tt_forward":
            gen_kwargs = self._collect_gen_smesh_kwargs()
            smesh, geom, fwd_kwargs = self._collect_tt_forward_args()
            payload = self._auto_wire_pipeline_targets(
                recipe,
                {"smesh": smesh, "geom": geom, "fwd_kwargs": fwd_kwargs},
            )
            smesh = payload["smesh"]
            geom = payload["geom"]
            fwd_kwargs = payload["fwd_kwargs"]
            if not smesh:
                raise ValueError("pipeline(gen_smesh -> tt_forward) 要求 tt_forward 的 smesh 已填写")
            return [
                ("gen_smesh", {"kwargs": gen_kwargs}),
                ("tt_forward", {"smesh": smesh, "geom": geom, "kwargs": fwd_kwargs}),
            ]
        if recipe == "gen_smesh -> tt_inverse":
            gen_kwargs = self._collect_gen_smesh_kwargs()
            mesh, data, inv_kwargs = self._collect_tt_inverse_args()
            payload = self._auto_wire_pipeline_targets(
                recipe,
                {"mesh": mesh, "data": data, "inv_kwargs": inv_kwargs},
            )
            mesh = payload["mesh"]
            data = payload["data"]
            inv_kwargs = payload["inv_kwargs"]
            if not mesh or not data:
                raise ValueError("pipeline(gen_smesh -> tt_inverse) 要求 tt_inverse 的 mesh/data 已填写")
            return [
                ("gen_smesh", {"kwargs": gen_kwargs}),
                ("tt_inverse", {"mesh": mesh, "data": data, "kwargs": inv_kwargs}),
            ]
        if recipe == "gen_smesh -> gen_damp -> tt_inverse":
            gen_kwargs = self._collect_gen_smesh_kwargs()
            damp_kwargs = self._collect_gen_damp_kwargs()
            mesh, data, inv_kwargs = self._collect_tt_inverse_args()
            payload = self._auto_wire_pipeline_targets(
                recipe,
                {"mesh": mesh, "data": data, "inv_kwargs": inv_kwargs},
            )
            mesh = payload["mesh"]
            data = payload["data"]
            inv_kwargs = payload["inv_kwargs"]
            if not mesh or not data:
                raise ValueError("pipeline(gen_smesh -> gen_damp -> tt_inverse) 要求 tt_inverse 的 mesh/data 已填写")
            return [
                ("gen_smesh", {"kwargs": gen_kwargs}),
                ("gen_damp", {"kwargs": damp_kwargs}),
                ("tt_inverse", {"mesh": mesh, "data": data, "kwargs": inv_kwargs}),
            ]
        if recipe == "gen_smesh -> gen_vcorr -> tt_inverse":
            gen_kwargs = self._collect_gen_smesh_kwargs()
            vcorr_kwargs = self._collect_gen_vcorr_kwargs()
            mesh, data, inv_kwargs = self._collect_tt_inverse_args()
            payload = self._auto_wire_pipeline_targets(
                recipe,
                {"mesh": mesh, "data": data, "inv_kwargs": inv_kwargs},
            )
            mesh = payload["mesh"]
            data = payload["data"]
            inv_kwargs = payload["inv_kwargs"]
            if not mesh or not data:
                raise ValueError("pipeline(gen_smesh -> gen_vcorr -> tt_inverse) 要求 tt_inverse 的 mesh/data 已填写")
            return [
                ("gen_smesh", {"kwargs": gen_kwargs}),
                ("gen_vcorr", {"kwargs": vcorr_kwargs}),
                ("tt_inverse", {"mesh": mesh, "data": data, "kwargs": inv_kwargs}),
            ]
        raise ValueError(f"未知 pipeline recipe: {recipe}")

    def _assert_existing_file(self, path_value: str, field: str) -> None:
        if not path_value:
            return
        p = Path(path_value)
        if not p.is_absolute():
            p = Path(self.vars["work_dir"].get().strip() or str(Path.cwd())) / p
        if not p.exists():
            raise ValueError(f"pipeline 文件检查失败: {field} 不存在 -> {p}")
        if not p.is_file():
            raise ValueError(f"pipeline 文件检查失败: {field} 不是文件 -> {p}")

    def _audit_path_under_work_dir(
        self,
        work: Path,
        label: str,
        form_value: str | None,
        *,
        optional: bool = False,
    ) -> str:
        """单行说明：表单路径相对 work_dir 解析后的绝对路径及是否存在（供预览）。"""
        s = (form_value or "").strip()
        if not s:
            if optional:
                return f"{label}: （未填，不传）"
            return f"{label}: （空）"
        p = Path(s).expanduser()
        try:
            rp = p.resolve() if p.is_absolute() else (work / p).resolve()
        except OSError:
            return f"{label}: {s!r} → 解析失败"
        ok = "✓ 存在"
        bad = "✗ 不存在"
        if rp.is_file():
            st = ok
        elif rp.is_dir():
            st = "✗ 为目录（需要文件）"
        elif rp.exists():
            st = "✗ 非普通文件"
        else:
            st = bad
        return f"{label}: {s!r} → {rp}  {st}"

    def _tt_forward_input_path_audit(
        self, work: Path, smesh: str, geom: str | None, kwargs: dict
    ) -> str:
        """tt_forward 输入文件在 work_dir 下的解析结果（与运行时 cwd 一致）。"""
        lines = [
            f"# 输入路径检查（子进程 cwd = work_dir，即下方目录）",
            f"#   {work}",
            self._audit_path_under_work_dir(work, "smesh (-M)", smesh, optional=False),
            self._audit_path_under_work_dir(work, "geom (-G)", geom or "", optional=True),
        ]
        rf = kwargs.get("refl_file")
        if rf:
            lines.append(
                self._audit_path_under_work_dir(work, "refl_file (-F)", str(rf), optional=False)
            )
        cf = kwargs.get("clock_file")
        if cf:
            lines.append(
                self._audit_path_under_work_dir(work, "clock_file (-C)", str(cf), optional=False)
            )
        return "\n".join(lines)

    def _audit_output_target_note(self, work: Path, label: str, form_value: str | None) -> str:
        """输出文件路径：解析目标绝对路径并检查父目录是否存在（不必预先存在文件）。"""
        s = (form_value or "").strip()
        if not s:
            return f"{label}: （未填）"
        p = Path(s).expanduser()
        try:
            rp = p.resolve() if p.is_absolute() else (work / p).resolve()
        except OSError:
            return f"{label}（输出）: {s!r} → 解析失败"
        par = rp.parent
        try:
            par_ok = par.is_dir()
        except OSError:
            par_ok = False
        st = "✓ 父目录存在" if par_ok else "✗ 父目录不存在"
        return f"{label}（输出）: {s!r} → {rp}  {st}"

    def _mesh_family_input_path_audit(self, work: Path, kwargs: dict, title: str) -> str:
        """gen_smesh / gen_damp / gen_vcorr 共用的网格与 zelt 输入文件。"""
        lines = [
            f"# 输入路径检查（{title}，子进程 cwd = work_dir）",
            f"#   {work}",
        ]
        for key, label in (
            ("v_in", "v_in (-C)"),
            ("refl_file", "refl_file (-F)"),
            ("x_file", "x_file (-X)"),
            ("z_file", "z_file (-Z)"),
            ("topo_file", "topo_file (-T)"),
        ):
            if kwargs.get(key):
                lines.append(
                    self._audit_path_under_work_dir(work, label, str(kwargs[key]), optional=False)
                )
        zd = kwargs.get("zelt_dump_file")
        if zd:
            lines.append(
                self._audit_path_under_work_dir(work, "zelt_dump (-d)", str(zd), optional=True)
            )
        return "\n".join(lines)

    def _gen_smesh_preview_path_audit(self, work: Path, kwargs: dict) -> str:
        body = self._mesh_family_input_path_audit(work, kwargs, "gen_smesh")
        out = kwargs.get("out_file")
        return body + "\n" + self._audit_output_target_note(work, "smesh 输出文件", str(out) if out else None)

    def _tt_inverse_input_path_audit(self, work: Path, mesh: str, data: str, kwargs: dict) -> str:
        lines = [
            "# 输入路径检查（tt_inverse，预览为相对 work_dir；可复现运行包下实际 cwd 为 run_dir）",
            f"#   {work}",
            self._audit_path_under_work_dir(work, "mesh (-M)", mesh, optional=False),
            self._audit_path_under_work_dir(work, "data (-G)", data, optional=False),
        ]
        if kwargs.get("refl_file"):
            lines.append(
                self._audit_path_under_work_dir(
                    work, "refl_file (-F)", str(kwargs["refl_file"]), optional=False
                )
            )
        if kwargs.get("filter_bound_file"):
            lines.append(
                self._audit_path_under_work_dir(
                    work, "filter_bound (-s)", str(kwargs["filter_bound_file"]), optional=False
                )
            )
        sm = kwargs.get("smooth_opts") or {}
        if sm.get("corr_v_fn"):
            lines.append(
                self._audit_path_under_work_dir(
                    work, "smooth_corr_v (-CV)", str(sm["corr_v_fn"]), optional=False
                )
            )
        if sm.get("corr_d_fn"):
            lines.append(
                self._audit_path_under_work_dir(
                    work, "smooth_corr_d (-CD)", str(sm["corr_d_fn"]), optional=False
                )
            )
        dm = kwargs.get("damp_opts") or {}
        if dm.get("damp_v_fn"):
            lines.append(
                self._audit_path_under_work_dir(
                    work, "damp_v_fn (-DQ)", str(dm["damp_v_fn"]), optional=False
                )
            )
        g = kwargs.get("gravity_opts") or {}
        if g.get("grav_file"):
            lines.append(
                self._audit_path_under_work_dir(
                    work, "grav_file (-ZG)", str(g["grav_file"]), optional=False
                )
            )
        cont = g.get("continent")
        if cont and isinstance(cont, (list, tuple)) and len(cont) >= 1 and cont[0]:
            lines.append(
                self._audit_path_under_work_dir(
                    work, "grav_ZC cont_up", str(cont[0]), optional=False
                )
            )
        ou = g.get("ocean_upper")
        if ou and isinstance(ou, (list, tuple)) and len(ou) >= 2:
            if ou[0]:
                lines.append(
                    self._audit_path_under_work_dir(work, "grav_ZU up", str(ou[0]), optional=False)
                )
            if ou[1]:
                lines.append(
                    self._audit_path_under_work_dir(work, "grav_ZU lo", str(ou[1]), optional=False)
                )
        ol = g.get("ocean_lower")
        if ol and isinstance(ol, (list, tuple)) and len(ol) >= 1 and ol[0]:
            lines.append(
                self._audit_path_under_work_dir(work, "grav_ZL up", str(ol[0]), optional=False)
            )
        sed = g.get("sediment")
        if sed and isinstance(sed, (list, tuple)) and len(sed) >= 2:
            if sed[0]:
                lines.append(
                    self._audit_path_under_work_dir(work, "grav_ZS up", str(sed[0]), optional=False)
                )
            if sed[1]:
                lines.append(
                    self._audit_path_under_work_dir(work, "grav_ZS lo", str(sed[1]), optional=False)
                )
        if g.get("grav_dws"):
            lines.append(
                self._audit_output_target_note(work, "grav_dws (-ZK)", str(g["grav_dws"]))
            )
        for py_key, lbl in (
            ("log_file", "log (-L)"),
            ("out_root", "out_root (-O)"),
            ("dws_file", "dws (-K)"),
        ):
            if kwargs.get(py_key):
                lines.append(self._audit_output_target_note(work, lbl, str(kwargs[py_key])))
        return "\n".join(lines)

    def _stat_smesh_input_path_audit(self, work: Path, kwargs: dict) -> str:
        lines = [
            "# 输入路径检查（stat_smesh，子进程 cwd = work_dir）",
            f"#   {work}",
        ]
        for key in sorted(kwargs.keys()):
            if not (key.endswith("_file") or key.endswith("_bound")):
                continue
            v = kwargs[key]
            if v is None or (isinstance(v, str) and not str(v).strip()):
                continue
            lines.append(
                self._audit_path_under_work_dir(work, f"{key}", str(v), optional=False)
            )
        return "\n".join(lines)

    def _edit_smesh_input_path_audit(self, work: Path, smesh_file: str, kwargs: dict) -> str:
        lines = [
            "# 输入路径检查（edit_smesh_HHB，子进程 cwd = work_dir）",
            f"#   {work}",
            self._audit_path_under_work_dir(work, "smesh_file", smesh_file, optional=False),
        ]
        for key in sorted(kwargs.keys()):
            if key not in (
                "paste_file",
                "prof_file",
                "remove_bg_file",
                "moho_file",
                "base_file",
                "corr_file",
                "upper_bound",
            ):
                continue
            v = kwargs.get(key)
            if v is None or (isinstance(v, str) and not str(v).strip()):
                continue
            lines.append(
                self._audit_path_under_work_dir(work, key, str(v), optional=True)
            )
        return "\n".join(lines)

    def _tx_convert_input_path_audit(self, work: Path, station: Path, txin: Path, dout: Path, gout: Path) -> str:
        lines = [
            "# 输入路径检查（tx.in→tomo2d；台站与 tx.in 须已存在，输出为将写入的路径）",
            f"#   work_dir = {work}",
        ]

        def one_abs(label: str, p: Path, *, need_file: bool) -> str:
            try:
                rp = p.resolve()
            except OSError:
                return f"{label}: {p} → 解析失败"
            if need_file:
                if rp.is_file():
                    st = "✓ 存在"
                elif rp.is_dir():
                    st = "✗ 为目录"
                elif rp.exists():
                    st = "✗ 非普通文件"
                else:
                    st = "✗ 不存在"
            else:
                st = ""
            return f"{label}: {str(p)!r} → {rp}  {st}".rstrip()

        lines.append(one_abs("station_lis", station, need_file=True))
        lines.append(one_abs("tx.in", txin, need_file=True))
        lines.append(self._audit_output_target_note(work, "ttimes.dat 输出", str(dout)))
        lines.append(self._audit_output_target_note(work, "geom.dat 输出", str(gout)))
        return "\n".join(lines)

    def _pipeline_step_path_audit(self, work: Path, step_name: str, spec: dict) -> str:
        if step_name == "gen_smesh":
            return self._gen_smesh_preview_path_audit(work, spec.get("kwargs") or {})
        if step_name == "gen_damp":
            return self._mesh_family_input_path_audit(work, spec.get("kwargs") or {}, "gen_damp")
        if step_name == "gen_vcorr":
            return self._mesh_family_input_path_audit(work, spec.get("kwargs") or {}, "gen_vcorr")
        if step_name == "tt_forward":
            sm = spec.get("smesh") or ""
            gm = spec.get("geom")
            kw = dict(spec.get("kwargs") or {})
            return self._tt_forward_input_path_audit(work, str(sm), gm, kw)
        if step_name == "tt_inverse":
            return self._tt_inverse_input_path_audit(
                work,
                str(spec.get("mesh") or ""),
                str(spec.get("data") or ""),
                dict(spec.get("kwargs") or {}),
            )
        return f"# （未知步骤 {step_name}，跳过路径检查）"

    def _validate_pipeline_input_files(self, plan: list[tuple[str, dict]]) -> None:
        links = self._pipeline_links()
        for key, value in links.items():
            if value:
                self._assert_existing_file(value, f"pipe.link_{key}")

        for step_name, spec in plan:
            if step_name == "gen_smesh":
                kw = spec.get("kwargs", {}) or {}
                outf = kw.get("out_file")
                if not str(outf or "").strip():
                    raise ValueError(
                        "pipeline 中的 gen_smesh 必须在界面填写「smesh 输出文件」（out_file），"
                        "否则网格无法落盘，下游步骤无法使用。"
                    )
                for k in ("v_in", "refl_file", "x_file", "z_file", "topo_file"):
                    if k in kw and kw[k] is not None:
                        self._assert_existing_file(str(kw[k]), f"gen_smesh.{k}")
            elif step_name == "gen_damp":
                kw = spec.get("kwargs", {})
                for k in ("v_in", "x_file", "z_file", "topo_file"):
                    if k in kw and kw[k] is not None:
                        self._assert_existing_file(str(kw[k]), f"gen_damp.{k}")
            elif step_name == "gen_vcorr":
                kw = spec.get("kwargs", {})
                for k in ("v_in", "x_file", "z_file", "topo_file"):
                    if k in kw and kw[k] is not None:
                        self._assert_existing_file(str(kw[k]), f"gen_vcorr.{k}")
            elif step_name == "tt_forward":
                self._assert_existing_file(spec.get("smesh") or "", "tt_forward.smesh")
                geom = spec.get("geom")
                if geom:
                    self._assert_existing_file(str(geom), "tt_forward.geom")
                fkw = spec.get("kwargs", {}) or {}
                refl = fkw.get("refl_file")
                if refl:
                    self._assert_existing_file(str(refl), "tt_forward.refl_file")
                clk = fkw.get("clock_file")
                if clk:
                    self._assert_existing_file(str(clk), "tt_forward.clock_file")
            elif step_name == "stat_smesh":
                skw = spec.get("kwargs", {}) or {}
                mode = skw.get("mode")
                if mode == "list":
                    lf = skw.get("list_file")
                    if lf:
                        self._assert_existing_file(str(lf), "stat_smesh.list_file")
                    if skw.get("cmd_type") == "r" and skw.get("ave_file"):
                        self._assert_existing_file(str(skw["ave_file"]), "stat_smesh.ave_file")
                elif mode == "mesh":
                    mf = skw.get("mesh_file")
                    if mf:
                        self._assert_existing_file(str(mf), "stat_smesh.mesh_file")
                    for k in ("top_bound", "bot_bound", "mid_bound"):
                        p = skw.get(k)
                        if p:
                            self._assert_existing_file(str(p), f"stat_smesh.{k}")
                    et = skw.get("exclude_top_bound")
                    eb = skw.get("exclude_bot_bound")
                    if et:
                        self._assert_existing_file(str(et), "stat_smesh.exclude_top_bound")
                    if eb:
                        self._assert_existing_file(str(eb), "stat_smesh.exclude_bot_bound")
            elif step_name == "tt_inverse":
                self._assert_existing_file(spec.get("mesh") or "", "tt_inverse.mesh")
                self._assert_existing_file(spec.get("data") or "", "tt_inverse.data")
                kw = spec.get("kwargs", {}) or {}
                if kw.get("refl_file"):
                    self._assert_existing_file(str(kw["refl_file"]), "tt_inverse.refl_file")
                smooth_opts = kw.get("smooth_opts", {}) or {}
                if smooth_opts.get("corr_v_fn"):
                    self._assert_existing_file(str(smooth_opts["corr_v_fn"]), "tt_inverse.smooth_opts.corr_v_fn")
                if smooth_opts.get("corr_d_fn"):
                    self._assert_existing_file(str(smooth_opts["corr_d_fn"]), "tt_inverse.smooth_opts.corr_d_fn")
                damp_opts = kw.get("damp_opts", {}) or {}
                if damp_opts.get("damp_v_fn"):
                    self._assert_existing_file(str(damp_opts["damp_v_fn"]), "tt_inverse.damp_opts.damp_v_fn")
                if kw.get("filter_bound_file"):
                    self._assert_existing_file(str(kw["filter_bound_file"]), "tt_inverse.filter_bound_file")
                go = kw.get("gravity_opts") or {}
                if go.get("grav_file"):
                    self._assert_existing_file(str(go["grav_file"]), "tt_inverse.gravity_opts.ZG")
                cont = go.get("continent")
                if cont:
                    self._assert_existing_file(str(cont[0]), "tt_inverse.gravity_opts.ZC")
                ou = go.get("ocean_upper")
                if ou:
                    self._assert_existing_file(str(ou[0]), "tt_inverse.gravity_opts.ZU_up")
                    self._assert_existing_file(str(ou[1]), "tt_inverse.gravity_opts.ZU_lo")
                ol = go.get("ocean_lower")
                if ol:
                    self._assert_existing_file(str(ol[0]), "tt_inverse.gravity_opts.ZL")
                sed = go.get("sediment")
                if sed:
                    self._assert_existing_file(str(sed[0]), "tt_inverse.gravity_opts.ZS_up")
                    self._assert_existing_file(str(sed[1]), "tt_inverse.gravity_opts.ZS_lo")

    def preview_pipeline(self) -> None:
        try:
            work = self._validate_work_dir()
            plan = self._build_pipeline_plan()
            self._validate_pipeline_input_files(plan)
            payload = {
                "recipe": self.vars["pipe.recipe"].get(),
                "auto_wire": bool(self.vars["pipe.auto_wire"].get()),
                "links": self._pipeline_links(),
                "steps": [{"name": n, **v} for n, v in plan],
            }
            text = "pipeline plan:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
            audit_blocks = ["\n# ========== 输入路径检查（按步骤，cwd 见各段说明）=========="]
            for step_name, spec in plan:
                audit_blocks.append(f"\n### {step_name}\n")
                audit_blocks.append(self._pipeline_step_path_audit(work, step_name, spec))
            tomo = self._get_tomo()
            blocks = [text, "\n".join(audit_blocks), "\n# 各步解析后命令行:"]
            for step_name, spec in plan:
                try:
                    cmd = self._resolve_pipeline_step_cmdline(tomo, step_name, spec)
                    if cmd:
                        blocks.append(f"\n## {step_name}\n{_format_resolved_cmdline(cmd)}")
                    else:
                        blocks.append(
                            f"\n## {step_name}\n（必选参数不齐，未拼 argv）"
                        )
                except Exception as e:
                    blocks.append(f"\n## {step_name}\n（无法解析: {e}）")
            self._set_preview("\n".join(blocks))
        except Exception as e:
            messagebox.showerror("预览失败", str(e))

    def run_pipeline(self) -> None:
        started_at: float | None = None
        title = ""
        try:
            work = self._validate_work_dir()
            plan = self._build_pipeline_plan()
            self._validate_pipeline_input_files(plan)
            title = f"pipeline[{self.vars['pipe.recipe'].get()}]"
            started_at = self._start_run_elapsed_timer(title)
            payload = {
                "recipe": self.vars["pipe.recipe"].get(),
                "auto_wire": bool(self.vars["pipe.auto_wire"].get()),
                "links": self._pipeline_links(),
                "steps": [{"name": n, **v} for n, v in plan],
            }
            text = "运行: pipeline(...)\n" + json.dumps(payload, ensure_ascii=False, indent=2)
            tomo = self._get_tomo()
            blocks = [text, "\n# 各步解析后命令行:"]
            for step_name, spec in plan:
                try:
                    cmd = self._resolve_pipeline_step_cmdline(tomo, step_name, spec)
                    if cmd:
                        blocks.append(f"\n## {step_name}\n{_format_resolved_cmdline(cmd)}")
                    else:
                        blocks.append(
                            f"\n## {step_name}\n（必选参数不齐，未拼 argv）"
                        )
                except Exception as e:
                    blocks.append(f"\n## {step_name}\n（无法解析: {e}）")
            self._set_preview("\n".join(blocks))
            try:
                use_tt_inv_bundle = bool(self.vars["inv.use_repro_bundle"].get())
            except (KeyError, tk.TclError):
                use_tt_inv_bundle = True
            gui_profile_snapshot = self._collect_gui_profile_dict()
            bundle_run_label = self._tt_inverse_bundle_run_label()
            self._run_pipeline_in_thread(
                work,
                plan,
                use_tt_inverse_bundle=use_tt_inv_bundle,
                gui_profile_snapshot=gui_profile_snapshot,
                started_at=started_at,
                tt_inverse_bundle_run_label=bundle_run_label,
            )
        except Exception as e:
            if started_at is not None and title:
                self._cancel_run_elapsed_timer()
                self._status_set_finished(title, False, started_at)
                self._log_run_elapsed(title, started_at)
            messagebox.showerror("运行前检查失败", str(e))

    def _cancel_run_elapsed_timer(self) -> None:
        if self._run_elapsed_after_id is not None:
            try:
                self.root.after_cancel(self._run_elapsed_after_id)
            except tk.TclError:
                pass
            self._run_elapsed_after_id = None
        self._run_elapsed_start = None

    def _tick_run_elapsed(self) -> None:
        if self._run_elapsed_start is None:
            return
        elapsed = int(time.monotonic() - self._run_elapsed_start)
        self.status_var.set(f"执行中: {self._run_elapsed_title}（已运行 {elapsed}s）")
        self._run_elapsed_after_id = self.root.after(1000, self._tick_run_elapsed)

    def _start_run_elapsed_timer(self, title: str) -> float:
        """启动状态栏秒级计时；返回 monotonic 起点供结束时写入执行日志。"""
        self._cancel_run_elapsed_timer()
        self._run_elapsed_start = time.monotonic()
        self._run_elapsed_title = title
        self.status_var.set(f"执行中: {title}（已运行 0s）")
        self._run_elapsed_after_id = self.root.after(1000, self._tick_run_elapsed)
        return self._run_elapsed_start

    def _log_run_elapsed(self, title: str, started_at: float | None) -> None:
        """在执行日志中记录总用时（默认写入 tomo2d_gui.log）。"""
        if started_at is None:
            return
        dt = time.monotonic() - started_at
        self._log(f"[{title}] 用时 {dt:.2f}s")

    def _status_set_finished(self, title: str, ok: bool, started_at: float | None) -> None:
        """结束态状态栏，例如：完成tt_forward 用时 34.63s"""
        if started_at is not None:
            dt = time.monotonic() - started_at
            word = "完成" if ok else "失败"
            self.status_var.set(f"{word}{title} 用时 {dt:.2f}s")
        else:
            self.status_var.set(("完成" if ok else "失败") + f": {title}")

    def _run_pipeline_in_thread(
        self,
        cwd: Path,
        plan: list[tuple[str, dict]],
        *,
        use_tt_inverse_bundle: bool = False,
        gui_profile_snapshot: dict | None = None,
        started_at: float | None = None,
        tt_inverse_bundle_run_label: str | None = None,
    ) -> None:
        title = f"pipeline[{self.vars['pipe.recipe'].get()}]"
        if started_at is None:
            started_at = self._start_run_elapsed_timer(title)
        else:
            self._run_elapsed_title = title
        self._log(f"[{title}] 开始，工作目录: {cwd}")
        inv_bundle_label = tt_inverse_bundle_run_label

        def worker():
            old = Path.cwd()
            try:
                os.chdir(cwd)
                tomo = self._get_tomo()
                tomo.proc_cwd = str(cwd)
                outputs: list[tuple[str, str, str]] = []
                for step_name, spec in plan:
                    if step_name == "gen_smesh":
                        result = tomo.gen_smesh(**spec["kwargs"])
                    elif step_name == "gen_damp":
                        result = tomo.gen_damp(**spec["kwargs"])
                    elif step_name == "gen_vcorr":
                        result = tomo.gen_vcorr(**spec["kwargs"])
                    elif step_name == "tt_forward":
                        result = tomo.tt_forward(smesh=spec["smesh"], geom=spec["geom"], **spec["kwargs"])
                    elif step_name == "tt_inverse":
                        if use_tt_inverse_bundle:
                            try:
                                from .tt_inverse_bundle import (
                                    build_tt_inverse_bundle,
                                    finalize_tt_inverse_manifest,
                                    write_tt_inverse_manifest,
                                )
                            except ImportError:
                                from pyAOBS.modeling.tomo2d.tt_inverse_bundle import (
                                    build_tt_inverse_bundle,
                                    finalize_tt_inverse_manifest,
                                    write_tt_inverse_manifest,
                                )
                            br = build_tt_inverse_bundle(
                                cwd,
                                spec["mesh"],
                                spec["data"],
                                dict(spec["kwargs"]),
                                run_label=inv_bundle_label,
                            )
                            argv = tomo.resolve_cmdline_tt_inverse(
                                mesh=br.mesh, data=br.data, **br.kwargs
                            )
                            if argv:
                                write_tt_inverse_manifest(
                                    manifest_path=br.manifest_path,
                                    work_dir=cwd,
                                    run_dir=br.run_dir,
                                    executable_resolved=argv[0],
                                    argv=argv,
                                    mesh=br.mesh,
                                    data=br.data,
                                    kwargs=br.kwargs,
                                    inputs_rows=br.inputs_manifest,
                                    status="running",
                                    gui_profile=gui_profile_snapshot,
                                )
                            old_pc = tomo.proc_cwd
                            tomo.proc_cwd = str(br.run_dir)
                            err_b: BaseException | None = None
                            inv_res = None
                            try:
                                inv_res = tomo.tt_inverse(
                                    mesh=br.mesh, data=br.data, **br.kwargs
                                )
                            except BaseException as e:
                                err_b = e
                            finally:
                                tomo.proc_cwd = old_pc
                                finalize_tt_inverse_manifest(
                                    br.manifest_path, result=inv_res, err=err_b
                                )
                            if err_b is not None:
                                raise err_b
                            result = inv_res
                        else:
                            result = tomo.tt_inverse(
                                mesh=spec["mesh"], data=spec["data"], **spec["kwargs"]
                            )
                    else:
                        raise ValueError(f"未知 pipeline step: {step_name}")
                    cmd_args = getattr(result, "args", None)
                    outputs.append(
                        (
                            step_name,
                            getattr(result, "stdout", "") or "",
                            getattr(result, "stderr", "") or "",
                            cmd_args,
                        )
                    )
                self.root.after(
                    0,
                    lambda st=started_at: self._on_pipeline_done(title, outputs, None, started_at=st),
                )
            except Exception as e:
                self.root.after(
                    0,
                    lambda exc=e, st=started_at: self._on_pipeline_done(title, [], exc, started_at=st),
                )
            finally:
                os.chdir(old)

        threading.Thread(target=worker, daemon=True).start()

    def _on_pipeline_done(
        self,
        title: str,
        outputs: list[tuple[str, str, str, object | None]],
        err: Exception | None,
        *,
        started_at: float | None = None,
    ) -> None:
        self._cancel_run_elapsed_timer()
        if err is not None:
            self._status_set_finished(title, False, started_at)
            self._log_run_elapsed(title, started_at)
            if isinstance(err, TomoCommandError):
                cl = _format_resolved_cmdline(err.command)
                if cl:
                    self._log(f"[{title}] 命令: {cl}")
                self._log(f"[{title}] 失败: {err}")
                messagebox.showerror("流程执行失败", str(err))
            else:
                self._log(f"[{title}] 异常: {err}")
                messagebox.showerror("流程执行异常", str(err))
            return
        self._status_set_finished(title, True, started_at)
        self._log_run_elapsed(title, started_at)
        self._log(f"[{title}] 成功")
        for step_name, stdout, stderr, cmd_args in outputs:
            self._log(f"[{title}] step={step_name} 完成")
            cl = _format_resolved_cmdline(cmd_args)
            if cl:
                self._log(f"[{step_name}] 命令: {cl}")
            # gen_smesh 的 stdout 为完整慢度网格文本，体积大，不写入日志
            if step_name == "gen_smesh" and stdout.strip():
                self._log(
                    f"[{step_name}] stdout（慢度网格全文）已省略；"
                    "本界面在运行 pipeline 前已要求填写「smesh 输出文件」，网格应已落盘。"
                )
            elif stdout.strip():
                self._log(f"[{step_name}] stdout:\n" + stdout.strip(), write_file=False)
            if stderr.strip():
                self._log(f"[{step_name}] stderr:\n" + stderr.strip(), write_file=False)

    def _run_in_thread(
        self,
        title: str,
        cwd: Path,
        func,
        *,
        proc_cwd: Path | None = None,
        worker_finally=None,
        started_at: float | None = None,
    ):
        if started_at is None:
            started_at = self._start_run_elapsed_timer(title)
        else:
            self._run_elapsed_title = title
        self._log(f"[{title}] 开始，工作目录: {cwd}")

        def worker():
            old = Path.cwd()
            result = None
            err: BaseException | None = None
            stdout, stderr, cmd_args = "", "", None
            try:
                os.chdir(cwd)
                tomo = self._get_tomo()
                pc = proc_cwd if proc_cwd is not None else cwd
                tomo.proc_cwd = str(pc)
                result = func(tomo)
                stdout = getattr(result, "stdout", "") or ""
                stderr = getattr(result, "stderr", "") or ""
                cmd_args = getattr(result, "args", None)
            except BaseException as e:
                err = e
            finally:
                if worker_finally is not None:
                    try:
                        worker_finally(result, err)
                    except Exception as mf:
                        try:
                            self._log(f"[{title}] 运行包收尾写入 manifest 失败: {mf}", write_file=False)
                        except Exception:
                            pass
                try:
                    os.chdir(old)
                except OSError:
                    pass
            if err is None:
                self.root.after(
                    0,
                    lambda ca=cmd_args, st=started_at, so=stdout, se=stderr: self._on_run_done(
                        title, so, se, None, cmd_args=ca, started_at=st
                    ),
                )
            else:
                self.root.after(
                    0,
                    lambda exc=err, st=started_at: self._on_run_done(
                        title, "", "", exc, started_at=st
                    ),
                )

        threading.Thread(target=worker, daemon=True).start()

    def _on_run_done(
        self,
        title: str,
        stdout: str,
        stderr: str,
        err: Exception | None,
        *,
        cmd_args: object | None = None,
        started_at: float | None = None,
    ):
        self._cancel_run_elapsed_timer()
        if err is None:
            self._status_set_finished(title, True, started_at)
            self._log_run_elapsed(title, started_at)
            self._log(f"[{title}] 成功")
            cl = _format_resolved_cmdline(cmd_args)
            if cl:
                self._log(f"命令: {cl}")
            # gen_smesh 的 stdout 为网格全文，不写入执行日志（GUI 运行前已要求填写 out_file）
            if stdout.strip() and title != "gen_smesh":
                self._log("stdout:\n" + stdout.strip(), write_file=False)
            elif title == "gen_smesh" and stdout.strip():
                self._log("gen_smesh：网格已按「smesh 输出文件」落盘；stdout 正文未记入本日志。")
            if stderr.strip():
                self._log("stderr:\n" + stderr.strip(), write_file=False)
            return

        self._status_set_finished(title, False, started_at)
        self._log_run_elapsed(title, started_at)
        if isinstance(err, TomoCommandError):
            cl = _format_resolved_cmdline(err.command)
            if cl:
                self._log(f"命令: {cl}")
            self._log(f"[{title}] 失败: {err}")
            messagebox.showerror("执行失败", str(err))
        else:
            self._log(f"[{title}] 异常: {err}")
            messagebox.showerror("执行异常", str(err))

    def _set_preview(self, text: str) -> None:
        self.preview_text.delete("1.0", tk.END)
        self.preview_text.insert(tk.END, text)

    def _append_file_log_line(self, text: str) -> None:
        """将一行写入 work_dir/tomo2d_gui.log（与 _log 中带 write_file=True 的条目同步；目录须已存在）。"""
        try:
            if not self.vars["gui.write_file_log"].get():
                return
        except (KeyError, tk.TclError):
            return
        wd = self._work_dir_resolved()
        try:
            if not wd.is_dir():
                return
        except OSError:
            return
        path = wd / "tomo2d_gui.log"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        wd_key = str(wd)
        try:
            wd_key = str(wd.resolve())
        except OSError:
            pass
        with self._file_log_lock:
            try:
                with open(path, "a", encoding="utf-8", newline="\n") as f:
                    if wd_key not in self._file_log_banner_paths:
                        self._file_log_banner_paths.add(wd_key)
                        f.write(
                            f"\n======== TOMO2D GUI | 工作目录 {wd} | 开始记录 {ts} ========\n"
                        )
                    f.write(f"[{ts}] {text}\n")
            except OSError:
                pass

    def _log(self, text: str, *, write_file: bool = True) -> None:
        """写入右侧「执行日志」；write_file=False 时不追加 tomo2d_gui.log（用于子进程 stdout/stderr 正文）。"""
        now = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{now}] {text}\n")
        self.log_text.see(tk.END)
        if write_file:
            self._append_file_log_line(text)

    def _tt_inverse_replay_path_to_gui(self, p: str | None, run_dir: Path) -> str:
        """manifest 内相对路径（inputs/、outputs/）转为绝对路径填入表单。"""
        if p is None:
            return ""
        s = str(p).strip()
        if not s:
            return ""
        path = Path(s.replace("\\", "/"))
        if path.is_absolute():
            try:
                return str(path.resolve())
            except OSError:
                return s
        try:
            return str((run_dir / path).resolve())
        except OSError:
            return str(run_dir / path)

    @staticmethod
    def _json_to_entry_str(val) -> str:
        if val is None:
            return ""
        if isinstance(val, bool):
            return "1" if val else ""
        if isinstance(val, float):
            if val == int(val):
                return str(int(val))
            return repr(val)
        return str(val).strip()

    def _clear_tt_inverse_form(self) -> None:
        """清空 tt_inverse 页可编辑项（保留 inv.use_repro_bundle）。"""
        preserve = None
        try:
            preserve = self.vars["inv.use_repro_bundle"].get()
        except (KeyError, tk.TclError):
            pass
        for key in (
            "inv.mesh",
            "inv.data",
            "inv.xorder",
            "inv.zorder",
            "inv.clen",
            "inv.nintp",
            "inv.bend_cg_tol",
            "inv.bend_br_tol",
            "inv.refl_file",
            "inv.refl_weight",
            "inv.log_file",
            "inv.out_root",
            "inv.out_level",
            "inv.dws_file",
            "inv.crit_chi",
            "inv.lsqr_tol",
            "inv.niter",
            "inv.target_chi2",
            "inv.auto_damp_max_dv",
            "inv.auto_damp_max_dd",
            "inv.smooth_vel",
            "inv.smooth_dep",
            "inv.smooth_corr_v_fn",
            "inv.smooth_corr_d_fn",
            "inv.damp_vel",
            "inv.damp_dep",
            "inv.damp_v_fn",
            "inv.filter_bound_file",
            "inv.bundle_run_label",
            "inv.grav_file",
            "inv.grav_grid",
            "inv.grav_refrange",
            "inv.grav_cont_file",
            "inv.grav_cont_iconv",
            "inv.grav_oceanU_up",
            "inv.grav_oceanU_lo",
            "inv.grav_oceanU_iconv",
            "inv.grav_oceanL_up",
            "inv.grav_oceanL_iconv",
            "inv.grav_sed_up",
            "inv.grav_sed_lo",
            "inv.grav_sed_iconv",
            "inv.grav_deriv",
            "inv.grav_weight",
            "inv.grav_z0",
            "inv.grav_dws",
            "inv.grav_cutoff",
            "inv.verbose_level",
        ):
            if key in self.vars:
                self.vars[key].set("")
        for key in (
            "inv.do_full_refl",
            "inv.jumping",
            "inv.print_final_only",
            "inv.smooth_vel_log10",
            "inv.smooth_dep_log10",
        ):
            if key in self.vars:
                self.vars[key].set(False)
        if preserve is not None and "inv.use_repro_bundle" in self.vars:
            self.vars["inv.use_repro_bundle"].set(preserve)

    def _apply_tt_inverse_kwargs_to_gui(self, kwargs: dict, run_dir: Path) -> None:
        """将 TomoAnd tt_inverse 的 kwargs（与 manifest python_replay 一致）写回表单。"""
        for tk_key, py_key in (
            ("inv.xorder", "xorder"),
            ("inv.zorder", "zorder"),
            ("inv.clen", "clen"),
            ("inv.nintp", "nintp"),
            ("inv.bend_cg_tol", "bend_cg_tol"),
            ("inv.bend_br_tol", "bend_br_tol"),
            ("inv.refl_weight", "refl_weight"),
            ("inv.out_level", "out_level"),
            ("inv.crit_chi", "crit_chi"),
            ("inv.lsqr_tol", "lsqr_tol"),
            ("inv.niter", "niter"),
            ("inv.target_chi2", "target_chi2"),
            ("inv.auto_damp_max_dv", "auto_damp_max_dv"),
            ("inv.auto_damp_max_dd", "auto_damp_max_dd"),
        ):
            if py_key in kwargs and tk_key in self.vars:
                self.vars[tk_key].set(self._json_to_entry_str(kwargs[py_key]))

        if "refl_file" in kwargs and "inv.refl_file" in self.vars:
            self.vars["inv.refl_file"].set(
                self._tt_inverse_replay_path_to_gui(str(kwargs["refl_file"]), run_dir)
            )
        if "filter_bound_file" in kwargs and "inv.filter_bound_file" in self.vars:
            self.vars["inv.filter_bound_file"].set(
                self._tt_inverse_replay_path_to_gui(str(kwargs["filter_bound_file"]), run_dir)
            )
        if "log_file" in kwargs and "inv.log_file" in self.vars:
            self.vars["inv.log_file"].set(
                self._tt_inverse_replay_path_to_gui(str(kwargs["log_file"]), run_dir)
            )
        if "out_root" in kwargs and "inv.out_root" in self.vars:
            self.vars["inv.out_root"].set(
                self._tt_inverse_replay_path_to_gui(str(kwargs["out_root"]), run_dir)
            )
        if "dws_file" in kwargs and "inv.dws_file" in self.vars:
            self.vars["inv.dws_file"].set(
                self._tt_inverse_replay_path_to_gui(str(kwargs["dws_file"]), run_dir)
            )

        if kwargs.get("do_full_refl") and "inv.do_full_refl" in self.vars:
            self.vars["inv.do_full_refl"].set(True)
        if kwargs.get("jumping") and "inv.jumping" in self.vars:
            self.vars["inv.jumping"].set(True)
        if kwargs.get("print_final_only") and "inv.print_final_only" in self.vars:
            self.vars["inv.print_final_only"].set(True)

        sm = kwargs.get("smooth_opts") or {}
        if sm.get("vel") is not None and "inv.smooth_vel" in self.vars:
            self.vars["inv.smooth_vel"].set(self._json_to_entry_str(sm["vel"]))
        if sm.get("dep") is not None and "inv.smooth_dep" in self.vars:
            self.vars["inv.smooth_dep"].set(self._json_to_entry_str(sm["dep"]))
        if sm.get("corr_v_fn") and "inv.smooth_corr_v_fn" in self.vars:
            self.vars["inv.smooth_corr_v_fn"].set(
                self._tt_inverse_replay_path_to_gui(str(sm["corr_v_fn"]), run_dir)
            )
        if sm.get("corr_d_fn") and "inv.smooth_corr_d_fn" in self.vars:
            self.vars["inv.smooth_corr_d_fn"].set(
                self._tt_inverse_replay_path_to_gui(str(sm["corr_d_fn"]), run_dir)
            )
        if sm.get("vel_log10") and "inv.smooth_vel_log10" in self.vars:
            self.vars["inv.smooth_vel_log10"].set(True)
        if sm.get("dep_log10") and "inv.smooth_dep_log10" in self.vars:
            self.vars["inv.smooth_dep_log10"].set(True)

        dm = kwargs.get("damp_opts") or {}
        if dm.get("damp_v_fn") and "inv.damp_v_fn" in self.vars:
            self.vars["inv.damp_v_fn"].set(
                self._tt_inverse_replay_path_to_gui(str(dm["damp_v_fn"]), run_dir)
            )
        if dm.get("vel") is not None and "inv.damp_vel" in self.vars:
            self.vars["inv.damp_vel"].set(self._json_to_entry_str(dm["vel"]))
        if dm.get("dep") is not None and "inv.damp_dep" in self.vars:
            self.vars["inv.damp_dep"].set(self._json_to_entry_str(dm["dep"]))

        g = kwargs.get("gravity_opts") or {}
        if g.get("grav_file") and "inv.grav_file" in self.vars:
            self.vars["inv.grav_file"].set(
                self._tt_inverse_replay_path_to_gui(str(g["grav_file"]), run_dir)
            )
        if g.get("grid_spec") and "inv.grav_grid" in self.vars:
            self.vars["inv.grav_grid"].set(self._json_to_entry_str(g["grid_spec"]))
        if g.get("refrange") and "inv.grav_refrange" in self.vars:
            self.vars["inv.grav_refrange"].set(self._json_to_entry_str(g["refrange"]))
        cont = g.get("continent")
        if cont and isinstance(cont, (list, tuple)) and len(cont) >= 2:
            if "inv.grav_cont_file" in self.vars:
                self.vars["inv.grav_cont_file"].set(
                    self._tt_inverse_replay_path_to_gui(str(cont[0]), run_dir)
                )
            if "inv.grav_cont_iconv" in self.vars:
                self.vars["inv.grav_cont_iconv"].set(self._json_to_entry_str(cont[1]))
        ou = g.get("ocean_upper")
        if ou and isinstance(ou, (list, tuple)) and len(ou) >= 3:
            if "inv.grav_oceanU_up" in self.vars:
                self.vars["inv.grav_oceanU_up"].set(
                    self._tt_inverse_replay_path_to_gui(str(ou[0]), run_dir)
                )
            if "inv.grav_oceanU_lo" in self.vars:
                self.vars["inv.grav_oceanU_lo"].set(
                    self._tt_inverse_replay_path_to_gui(str(ou[1]), run_dir)
                )
            if "inv.grav_oceanU_iconv" in self.vars:
                self.vars["inv.grav_oceanU_iconv"].set(self._json_to_entry_str(ou[2]))
        ol = g.get("ocean_lower")
        if ol and isinstance(ol, (list, tuple)) and len(ol) >= 2:
            if "inv.grav_oceanL_up" in self.vars:
                self.vars["inv.grav_oceanL_up"].set(
                    self._tt_inverse_replay_path_to_gui(str(ol[0]), run_dir)
                )
            if "inv.grav_oceanL_iconv" in self.vars:
                self.vars["inv.grav_oceanL_iconv"].set(self._json_to_entry_str(ol[1]))
        sed = g.get("sediment")
        if sed and isinstance(sed, (list, tuple)) and len(sed) >= 3:
            if "inv.grav_sed_up" in self.vars:
                self.vars["inv.grav_sed_up"].set(
                    self._tt_inverse_replay_path_to_gui(str(sed[0]), run_dir)
                )
            if "inv.grav_sed_lo" in self.vars:
                self.vars["inv.grav_sed_lo"].set(
                    self._tt_inverse_replay_path_to_gui(str(sed[1]), run_dir)
                )
            if "inv.grav_sed_iconv" in self.vars:
                self.vars["inv.grav_sed_iconv"].set(self._json_to_entry_str(sed[2]))
        if g.get("deriv") and "inv.grav_deriv" in self.vars:
            self.vars["inv.grav_deriv"].set(self._json_to_entry_str(g["deriv"]))
        if g.get("weight_grav") is not None and "inv.grav_weight" in self.vars:
            self.vars["inv.grav_weight"].set(self._json_to_entry_str(g["weight_grav"]))
        if g.get("z0") is not None and "inv.grav_z0" in self.vars:
            self.vars["inv.grav_z0"].set(self._json_to_entry_str(g["z0"]))
        if g.get("grav_dws") and "inv.grav_dws" in self.vars:
            self.vars["inv.grav_dws"].set(
                self._tt_inverse_replay_path_to_gui(str(g["grav_dws"]), run_dir)
            )
        if g.get("cutoff") and "inv.grav_cutoff" in self.vars:
            self.vars["inv.grav_cutoff"].set(self._json_to_entry_str(g["cutoff"]))

        if kwargs.get("verbose") and "inv.verbose_level" in self.vars:
            vl = kwargs.get("verbose_level")
            self.vars["inv.verbose_level"].set(
                self._json_to_entry_str(vl) if vl is not None else "1"
            )

    def _apply_loaded_profile_dict(self, data: dict) -> None:
        """与「加载配置」共用：写入 self.vars 并触发各页签模式刷新。"""
        d = dict(data)
        if "fwd.bend_cg_tol" not in d and "fwd.tol1" in d:
            d["fwd.bend_cg_tol"] = d["fwd.tol1"]
        if "fwd.bend_br_tol" not in d and "fwd.tol2" in d:
            d["fwd.bend_br_tol"] = d["fwd.tol2"]
        for k, v in d.items():
            if k in self.vars:
                self.vars[k].set(v)
        self._normalize_all_file_path_vars(show_warn_outside=False)
        self._on_gen_smesh_mode_change()
        self._on_damp_mode_change()
        self._on_vcorr_mode_change()
        self._on_stat_mode_change()
        self._on_edit_mode_change()

    def _offer_sync_work_dir_from_manifest(self, manifest: dict) -> None:
        """若 manifest 含 work_dir 且目录存在，询问是否将界面工作目录改为该路径（相对路径才能解析）。"""
        wd_m = manifest.get("work_dir")
        if not wd_m:
            return
        try:
            p = Path(str(wd_m)).expanduser().resolve()
        except OSError:
            return
        if not p.is_dir():
            try:
                self._log(f"[manifest] 记录的工作目录不存在，未切换: {wd_m}", write_file=False)
            except Exception:
                pass
            return
        if "work_dir" not in self.vars:
            return
        cur = self.vars["work_dir"].get().strip()
        try:
            cur_p = Path(cur).expanduser().resolve() if cur else None
        except OSError:
            cur_p = None
        if cur_p == p:
            return
        if messagebox.askyesno(
            "同步工作目录",
            "manifest 中记录的项目根目录（work_dir）为：\n"
            f"  {p}\n\n"
            f"当前界面「工作目录」为：\n"
            f"  {cur or '（空）'}\n\n"
            "是否将界面工作目录改为上述 manifest 路径？\n\n"
            "建议选「是」：保存配置时的相对路径（gen_smesh、tt_forward 等）会相对该目录解析；"
            "若保持不一致，容易出现找不到输入文件。",
            parent=self.root,
        ):
            self.vars["work_dir"].set(str(p))

    def _merge_python_replay_paths(self, manifest: dict, source_path: str) -> None:
        """用 python_replay（相对 run_dir 的 inputs/…）覆盖 inv.* 与 kwargs 中的路径，与运行包一致。"""
        pr = manifest.get("python_replay")
        if not isinstance(pr, dict):
            return
        mesh, data = pr.get("mesh"), pr.get("data")
        if not mesh or not data:
            return
        rd = manifest.get("run_dir")
        if rd:
            run_dir = Path(str(rd)).resolve()
        else:
            run_dir = Path(source_path).resolve().parent
        if "inv.mesh" in self.vars:
            self.vars["inv.mesh"].set(self._tt_inverse_replay_path_to_gui(str(mesh), run_dir))
        if "inv.data" in self.vars:
            self.vars["inv.data"].set(self._tt_inverse_replay_path_to_gui(str(data), run_dir))
        self._apply_tt_inverse_kwargs_to_gui(dict(pr.get("kwargs") or {}), run_dir)

    def _sync_fwd_smesh_from_inv_if_missing(self) -> None:
        """若 fwd.smesh 在当前 work_dir 下找不到文件，而 inv.mesh 存在，则用 inv.mesh 填 fwd（正演常用同一 smesh）。"""
        if "fwd.smesh" not in self.vars or "inv.mesh" not in self.vars:
            return
        try:
            work = self._work_dir_resolved()
        except Exception:
            return

        def resolved_file(pstr: str) -> Path | None:
            s = (pstr or "").strip()
            if not s:
                return None
            path = Path(s).expanduser()
            if not path.is_absolute():
                path = (work / s).resolve()
            else:
                path = path.resolve()
            return path if path.is_file() else None

        if resolved_file(self.vars["fwd.smesh"].get()) is not None:
            return
        im = self.vars["inv.mesh"].get().strip()
        ip = resolved_file(im)
        if ip is None:
            return
        self.vars["fwd.smesh"].set(
            self._to_workdir_relative(str(ip), show_warn_outside=False)
        )

    def _finalize_manifest_path_resolution(
        self, manifest: dict, source_path: str, *, merge_python_replay: bool
    ) -> None:
        self._offer_sync_work_dir_from_manifest(manifest)
        self._normalize_all_file_path_vars(show_warn_outside=False)
        if merge_python_replay:
            self._merge_python_replay_paths(manifest, source_path)
            self._normalize_all_file_path_vars(show_warn_outside=False)
        self._sync_fwd_smesh_from_inv_if_missing()
        self._normalize_all_file_path_vars(show_warn_outside=False)
        wd_m = manifest.get("work_dir")
        if wd_m:
            try:
                self._log(
                    f"[manifest] 记录的项目根目录 work_dir: {wd_m}",
                    write_file=False,
                )
            except Exception:
                pass

    def _load_from_tt_inverse_manifest(self, manifest: dict, source_path: str) -> None:
        gp = manifest.get("gui_profile")
        if isinstance(gp, dict) and gp:
            self._apply_loaded_profile_dict(dict(gp))
            try:
                self._log(
                    "[manifest] 已从 gui_profile 恢复整界面；随后将解析路径（工作目录 / python_replay）",
                    write_file=False,
                )
            except Exception:
                pass
            self._finalize_manifest_path_resolution(
                manifest, source_path, merge_python_replay=True
            )
            return

        pr = manifest.get("python_replay")
        if not isinstance(pr, dict):
            raise ValueError("manifest 缺少 gui_profile 与 python_replay，无法加载")
        mesh = pr.get("mesh")
        data = pr.get("data")
        if not mesh or not data:
            raise ValueError("python_replay 缺少 mesh 或 data")
        rd = manifest.get("run_dir")
        if rd:
            run_dir = Path(str(rd)).resolve()
        else:
            run_dir = Path(source_path).resolve().parent
        kwargs = dict(pr.get("kwargs") or {})

        self._clear_tt_inverse_form()
        if "inv.mesh" in self.vars:
            self.vars["inv.mesh"].set(self._tt_inverse_replay_path_to_gui(str(mesh), run_dir))
        if "inv.data" in self.vars:
            self.vars["inv.data"].set(self._tt_inverse_replay_path_to_gui(str(data), run_dir))
        self._apply_tt_inverse_kwargs_to_gui(kwargs, run_dir)

        self._finalize_manifest_path_resolution(
            manifest, source_path, merge_python_replay=False
        )

    def _collect_gui_profile_dict(self) -> dict:
        """与「保存配置」相同：当前所有表单变量（先规范化路径）。"""
        self._normalize_all_file_path_vars(show_warn_outside=False)
        return {k: v.get() for k, v in self.vars.items()}

    def _save_profile(self) -> None:
        path = filedialog.asksaveasfilename(
            title="保存配置",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        data = self._collect_gui_profile_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self._log(f"配置已保存: {path}")

    def _load_profile(self) -> None:
        path = filedialog.askopenfilename(
            title="加载配置（全界面 JSON 或 tt_inverse 的 manifest.json）",
            filetypes=[
                ("JSON", "*.json"),
                ("manifest", "manifest.json"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("kind") == "pyaobs.tomo2d.tt_inverse":
            try:
                self._load_from_tt_inverse_manifest(data, path)
            except Exception as e:
                messagebox.showerror("加载 manifest 失败", str(e))
                return
            self._log(
                f"已从 manifest 加载: {path}"
                + (
                    "（含 gui_profile 时已恢复整界面）"
                    if isinstance(data.get("gui_profile"), dict) and data.get("gui_profile")
                    else "（仅 tt_inverse / python_replay）"
                )
            )
            return
        self._apply_loaded_profile_dict(data)
        self._log(f"配置已加载: {path}")


def launch_tomo2d_gui() -> None:
    try:
        from tkinterdnd2 import TkinterDnD

        root = TkinterDnD.Tk()
    except ImportError:
        root = tk.Tk()
    app = Tomo2DGui(master=root)
    app.root.mainloop()


if __name__ == "__main__":
    launch_tomo2d_gui()
