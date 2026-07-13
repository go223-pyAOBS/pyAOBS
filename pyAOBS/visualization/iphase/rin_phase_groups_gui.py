"""
独立的 r.in 相位组编辑 GUI（v2：单线程 + 分页列表）。

设计原则（避免 Windows/Tk 上常见卡顿与“状态悬挂”）：
- 不使用后台线程与队列；解析与写盘均在主线程同步完成（与 CLI 一致，通常很快）。
- 不用 Treeview 全量插入；左侧用 Canvas+分页行（每页固定条数），仅渲染当前页；每行 Checkbutton「保留」映射 enabled。
- 诊断与数组预览放在 Notebook 分栏，减少单屏大块 Text 同时刷新。
- “安全模式”仅不展开 rbnd/cbnd；“完整解析”同步加载全部路径数组。

用途：编辑 ray / nrbnd / rbnd / ncbnd / cbnd / nray / ivray，调试稳定后再接入 iphase_gui。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from .rin_phase_groups import (
        ConversionEvent,
        PhaseGroup,
        ReflectionEvent,
        RinMiscParams,
        apply_phase_groups_and_misc_to_rin_file,
        compile_phase_groups_to_arrays,
        describe_all_phase_groups,
        infer_phase_type_label,
        parse_phase_groups_from_arrays,
        parse_rin_phase_arrays_from_file,
        parse_rin_misc_params_from_file,
    )
except ImportError:
    from pyAOBS.visualization.iphase.rin_phase_groups import (
        ConversionEvent,
        PhaseGroup,
        ReflectionEvent,
        RinMiscParams,
        apply_phase_groups_and_misc_to_rin_file,
        compile_phase_groups_to_arrays,
        describe_all_phase_groups,
        infer_phase_type_label,
        parse_phase_groups_from_arrays,
        parse_rin_phase_arrays_from_file,
        parse_rin_misc_params_from_file,
    )

PAGE_SIZE = 100


def _debug(msg: str) -> None:
    if os.environ.get("RIN_GUI_DEBUG", "").strip() in ("1", "true", "yes"):
        log = Path.home() / "rin_phase_groups_gui.log"
        try:
            with open(log, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except OSError:
            pass


def _rbnd_list_to_text(events: list[ReflectionEvent]) -> str:
    out: list[str] = []
    for e in events:
        out.append(f"{e.boundary}" if e.direction == "down_reflect_up" else f"-{e.boundary}")
    return ",".join(out)


def _cbnd_list_to_text(events: list[ConversionEvent]) -> str:
    return ",".join(str(e.encounter_idx) for e in events)


def _parse_int_list(text: str) -> list[int]:
    clean = text.strip()
    if not clean:
        return []
    values: list[int] = []
    for token in clean.replace(" ", "").split(","):
        if not token:
            continue
        values.append(int(token))
    return values


def _parse_float_list(text: str) -> list[float]:
    clean = text.strip()
    if not clean:
        return []
    values: list[float] = []
    # 允许用户粘贴类似 "xshot=1,2,3" 的片段；对每个 token：
    # - 去空白
    # - 若含 '='，取 '=' 右侧
    # - 允许尾随逗号/空 token
    raw = clean.replace("\n", ",").replace("\t", ",")
    for token0 in raw.split(","):
        token = token0.strip()
        if not token:
            continue
        if "=" in token:
            token = token.split("=", 1)[1].strip()
        if not token:
            continue
        values.append(float(token))
    return values


def _parse_reflections(text: str) -> list[ReflectionEvent]:
    out: list[ReflectionEvent] = []
    for v in _parse_int_list(text):
        if v == 0:
            raise ValueError("rbnd 不允许 0")
        if v > 0:
            out.append(ReflectionEvent(boundary=v, direction="down_reflect_up"))
        else:
            out.append(ReflectionEvent(boundary=abs(v), direction="up_reflect_down"))
    return out


def _parse_conversions(text: str) -> list[ConversionEvent]:
    out: list[ConversionEvent] = []
    for v in _parse_int_list(text):
        if v < 0:
            raise ValueError("cbnd 不允许负数")
        out.append(ConversionEvent(encounter_idx=v))
    return out


def _group_row_label_text(global_idx: int, g: PhaseGroup) -> str:
    """列表行文字（保留状态由行首 Checkbutton 表示，此处不再写 Y/N）。"""
    nr = len(g.reflections)
    nc = len(g.conversions)
    ptype = infer_phase_type_label(g)
    cb = ",".join(str(c.encounter_idx) for c in g.conversions)
    if len(cb) > 16:
        cb = cb[:16] + "…"
    return (
        f"{global_idx + 1:4d}   ray={g.ray_code:4.1f}  "
        f"type={ptype:<4}  nray={g.nray:4d}  ivray={g.ivray:4d}  nr={nr} nc={nc}  cbnd={cb:<17}  {g.name[:12]}"
    )


class RinPhaseGroupsEditor(tk.Tk):
    def __init__(self, initial_rin_path: str | Path | None = None) -> None:
        super().__init__()
        self.title("RAYINVR r.in Phase Groups Editor (v2)")
        self.geometry("1180x780")
        self._init_styles()

        self.rin_path: Path | None = None
        self.groups: list[PhaseGroup] = []
        self._selected_idx: int | None = None
        self._page_start = 0
        self.misc = RinMiscParams()

        self.safe_mode_var = tk.BooleanVar(value=True)
        self.preview_full_var = tk.BooleanVar(value=False)

        self.status_var = tk.StringVar(value="请选择 r.in 文件")
        self.path_var = tk.StringVar(value="")
        self.page_info_var = tk.StringVar(value="")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        if initial_rin_path:
            self._open_rin_path(initial_rin_path)

    def _init_styles(self) -> None:
        # 相位组列表：白底；选中行浅色高亮
        st = ttk.Style(self)
        white = "#ffffff"
        try:
            st.configure("ListInner.TFrame", background=white)
            st.configure("Row.TFrame", background=white)
            st.configure("Row.TLabel", background=white)
            st.configure("Row.TCheckbutton", background=white)
        except Exception:
            pass

        # 选中：浅蓝底（接近 VSCode selection）
        sel_bg = "#dbeafe"
        st.configure("SelRow.TFrame", background=sel_bg)
        st.configure("SelRow.TLabel", background=sel_bg)
        st.configure("SelRow.TCheckbutton", background=sel_bg)

    def _bind_phase_list_keyboard(self, w: tk.Widget) -> None:
        """在列表区域子控件上绑定方向键；必须 return 'break' 以免被 Canvas 默认处理吞掉。"""

        def _up(_e=None) -> str:
            self._select_prev()
            return "break"

        def _down(_e=None) -> str:
            self._select_next()
            return "break"

        def _pgup(_e=None) -> str:
            self._select_prev(page=True)
            return "break"

        def _pgdn(_e=None) -> str:
            self._select_next(page=True)
            return "break"

        def _home(_e=None) -> str:
            self._select_home()
            return "break"

        def _end(_e=None) -> str:
            self._select_end()
            return "break"

        for seq, fn in (
            ("<Up>", _up),
            ("<Down>", _down),
            ("<Prior>", _pgup),
            ("<Next>", _pgdn),
            ("<Home>", _home),
            ("<End>", _end),
        ):
            w.bind(seq, fn)

    def _focus_phase_list_canvas(self) -> None:
        try:
            self.group_list_canvas.focus_set()
        except tk.TclError:
            pass

    def _build_ui(self) -> None:
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Button(top, text="打开 r.in", command=self._on_open).pack(side=tk.LEFT)
        ttk.Button(top, text="重载", command=self._on_reload).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(
            top,
            text="安全模式(仅省略尾部长度提示，仍完整加载 rbnd/cbnd)",
            variable=self.safe_mode_var,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="完整解析", command=self._on_reload_full).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="保存", command=self._on_save).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="校验", command=self._on_validate).pack(side=tk.LEFT, padx=4)
        ttk.Label(top, textvariable=self.path_var, foreground="#444").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=2)
        body.add(right, weight=3)

        # 左侧：分页 Canvas + 每行「保留」勾选（enabled）
        list_frame = ttk.LabelFrame(
            left,
            text=f"相位组列表（每页 {PAGE_SIZE} 条；勾选「保留」参与写回，取消则暂停 nray=0，数据保留可再勾选恢复）",
        )
        list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        nav = ttk.Frame(list_frame)
        nav.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)
        ttk.Button(nav, text="上一页", command=self._page_prev).pack(side=tk.LEFT)
        ttk.Button(nav, text="下一页", command=self._page_next).pack(side=tk.LEFT, padx=4)
        ttk.Label(nav, textvariable=self.page_info_var).pack(side=tk.LEFT, padx=8)
        ttk.Label(nav, text="跳到 #").pack(side=tk.LEFT)
        self.jump_var = tk.StringVar(value="1")
        ttk.Entry(nav, textvariable=self.jump_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav, text="Go", command=self._page_jump).pack(side=tk.LEFT, padx=2)

        sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_holder = ttk.Frame(list_frame)
        canvas_holder.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.group_list_canvas = tk.Canvas(
            canvas_holder,
            background="#ffffff",
            highlightthickness=0,
            yscrollcommand=sb.set,
        )
        self.group_list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=self.group_list_canvas.yview)

        self.group_list_inner = ttk.Frame(self.group_list_canvas, style="ListInner.TFrame")
        self._list_inner_window = self.group_list_canvas.create_window((0, 0), window=self.group_list_inner, anchor="nw")

        def _on_inner_configure(_event: tk.Event | None = None) -> None:
            self.group_list_canvas.configure(scrollregion=self.group_list_canvas.bbox("all"))

        self.group_list_inner.bind("<Configure>", _on_inner_configure)

        def _on_canvas_configure(event: tk.Event) -> None:
            self.group_list_canvas.itemconfigure(self._list_inner_window, width=event.width)

        self.group_list_canvas.bind("<Configure>", _on_canvas_configure)

        def _on_wheel(event: tk.Event) -> None:
            if getattr(event, "delta", 0):
                self.group_list_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.group_list_canvas.bind("<MouseWheel>", _on_wheel)
        self.group_list_canvas.bind("<Button-4>", lambda _e: self.group_list_canvas.yview_scroll(-1, "units"))
        self.group_list_canvas.bind("<Button-5>", lambda _e: self.group_list_canvas.yview_scroll(1, "units"))
        # 键盘：在 Canvas 上绑定时需 return "break"，否则会触发 Canvas 默认方向键滚动且事件不响
        self.group_list_canvas.configure(takefocus=True)
        self.group_list_canvas.bind("<Button-1>", lambda _e: self.group_list_canvas.focus_set(), add="+")
        self._bind_phase_list_keyboard(self.group_list_canvas)

        left_btn = ttk.Frame(left)
        left_btn.pack(side=tk.TOP, fill=tk.X, pady=4)
        ttk.Button(left_btn, text="新增组", command=self._on_add).pack(side=tk.LEFT)
        ttk.Button(left_btn, text="彻底删除组", command=self._on_remove).pack(side=tk.LEFT, padx=4)
        ttk.Button(left_btn, text="切换保留(当前行)", command=self._on_toggle_enabled).pack(side=tk.LEFT, padx=4)
        ttk.Button(left_btn, text="全保留", command=self._on_retain_all).pack(side=tk.LEFT, padx=4)
        ttk.Button(left_btn, text="全不保留", command=self._on_retain_none).pack(side=tk.LEFT, padx=4)
        ttk.Button(left_btn, text="复制组", command=self._on_clone).pack(side=tk.LEFT, padx=4)

        # 右侧：编辑 + Notebook
        form = ttk.LabelFrame(right, text="编辑当前组")
        form.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)

        self.var_enabled = tk.BooleanVar(value=True)
        self.var_name = tk.StringVar(value="")
        self.var_ray = tk.StringVar(value="1.1")
        self.var_nray = tk.StringVar(value="10")
        self.var_ivray = tk.StringVar(value="1")
        self.var_source_wave = tk.StringVar(value="P")
        self.var_rbnd = tk.StringVar(value="")
        self.var_cbnd = tk.StringVar(value="")
        self.var_xshot = tk.StringVar(value="")
        self.var_zshot = tk.StringVar(value="")
        self.var_vred = tk.StringVar(value="")
        self.var_imodf = tk.StringVar(value="")
        self.var_i2pt = tk.StringVar(value="")
        self.var_isrch = tk.StringVar(value="")
        self.var_aamin = tk.StringVar(value="")
        self.var_aamax = tk.StringVar(value="")
        self.var_ishot = tk.StringVar(value="")
        self.var_n2pt = tk.StringVar(value="")
        self.var_space = tk.StringVar(value="")
        self.var_x2pt = tk.StringVar(value="")

        row = 0
        ttk.Checkbutton(
            form,
            text="保留(参与写回，同左侧勾选)",
            variable=self.var_enabled,
            command=self._on_form_retain_toggle,
        ).grid(row=row, column=0, sticky="w", padx=6, pady=2)
        row += 1
        self._add_labeled_entry(form, row, "name", self.var_name)
        row += 1
        self._add_labeled_entry(form, row, "ray_code", self.var_ray)
        row += 1
        self._add_labeled_entry(form, row, "nray", self.var_nray)
        row += 1
        self._add_labeled_entry(form, row, "ivray", self.var_ivray)
        row += 1
        ttk.Label(form, text="source_wave").grid(row=row, column=0, sticky="w", padx=6, pady=2)
        ttk.Combobox(form, values=["P", "S"], state="readonly", textvariable=self.var_source_wave, width=14).grid(
            row=row, column=1, sticky="w", padx=6, pady=2
        )
        row += 1
        self._add_labeled_entry(form, row, "rbnd (如 -1,3)", self.var_rbnd)
        row += 1
        self._add_labeled_entry(form, row, "cbnd (如 0,3)", self.var_cbnd)
        row += 1
        ttk.Button(form, text="应用到当前组", command=self._on_apply_row).grid(
            row=row, column=0, columnspan=2, sticky="we", padx=6, pady=6
        )
        form.grid_columnconfigure(1, weight=1)

        global_box = ttk.LabelFrame(right, text="全局参数（来自 r.in）")
        global_box.pack(side=tk.TOP, fill=tk.X, padx=2, pady=(4, 2))
        global_box.grid_columnconfigure(1, weight=1)
        global_box.grid_columnconfigure(3, weight=1)
        global_box.grid_columnconfigure(5, weight=1)

        def add_triplet_row(
            row: int,
            l1: str,
            v1: tk.StringVar,
            l2: str,
            v2: tk.StringVar,
            l3: str,
            v3: tk.StringVar,
        ) -> None:
            ttk.Label(global_box, text=l1).grid(row=row, column=0, sticky="w", padx=6, pady=2)
            ttk.Entry(global_box, textvariable=v1, width=18).grid(row=row, column=1, sticky="we", padx=6, pady=2)
            ttk.Label(global_box, text=l2).grid(row=row, column=2, sticky="w", padx=6, pady=2)
            ttk.Entry(global_box, textvariable=v2, width=18).grid(row=row, column=3, sticky="we", padx=6, pady=2)
            ttk.Label(global_box, text=l3).grid(row=row, column=4, sticky="w", padx=6, pady=2)
            ttk.Entry(global_box, textvariable=v3, width=18).grid(row=row, column=5, sticky="we", padx=6, pady=2)

        gr = 0
        add_triplet_row(gr, "imodf", self.var_imodf, "i2pt", self.var_i2pt, "isrch", self.var_isrch)
        gr += 1
        add_triplet_row(gr, "aamin", self.var_aamin, "aamax", self.var_aamax, "space (逗号分隔)", self.var_space)
        gr += 1
        add_triplet_row(gr, "x2pt", self.var_x2pt, "pltpar.vred", self.var_vred, "ishot (逗号分隔)", self.var_ishot)
        gr += 1
        add_triplet_row(gr, "xshot (逗号分隔)", self.var_xshot, "zshot (逗号分隔)", self.var_zshot, "n2pt", self.var_n2pt)
        gr += 1
        ttk.Button(global_box, text="应用全局参数", command=self._on_apply_misc).grid(
            row=gr, column=0, columnspan=6, sticky="we", padx=6, pady=6
        )

        nb = ttk.Notebook(right)
        nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=2, pady=6)

        tab_diag = ttk.Frame(nb)
        tab_arr = ttk.Frame(nb)
        tab_explain = ttk.Frame(nb)
        nb.add(tab_diag, text="诊断")
        nb.add(tab_arr, text="数组预览")
        nb.add(tab_explain, text="文字说明")

        self.diag_text = tk.Text(tab_diag, height=14, wrap="word", font=("Segoe UI", 10))
        self.diag_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)

        arr_top = ttk.Frame(tab_arr)
        arr_top.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        ttk.Checkbutton(
            arr_top,
            text="显示全量数组",
            variable=self.preview_full_var,
            command=self._refresh_array_preview_from_groups,
        ).pack(side=tk.LEFT)

        self.arr_text = tk.Text(tab_arr, height=14, wrap="word", font=("Consolas", 9))
        self.arr_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.narrative_text = tk.Text(tab_explain, height=14, wrap="word", font=("Segoe UI", 10))
        self.narrative_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)

        status_bar = ttk.Frame(self, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(status_bar, textvariable=self.status_var, anchor=tk.W).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Sizegrip(status_bar).pack(side=tk.RIGHT)

    @staticmethod
    def _add_labeled_entry(parent: ttk.LabelFrame, row: int, label: str, var: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=2)
        ttk.Entry(parent, textvariable=var, width=28).grid(row=row, column=1, sticky="we", padx=6, pady=2)

    def _set_diag(self, lines: list[str]) -> None:
        self.diag_text.delete("1.0", tk.END)
        if not lines:
            self.diag_text.insert(tk.END, "（无诊断，可点击「校验」）\n")
            return
        self.diag_text.insert(tk.END, "\n".join(lines) + "\n")

    def _set_array_preview(self, lines: list[str]) -> None:
        self.arr_text.delete("1.0", tk.END)
        if not lines:
            self.arr_text.insert(tk.END, "暂无数组预览。\n")
            return
        self.arr_text.insert(tk.END, "\n".join(lines) + "\n")

    def _refresh_narrative_from_groups(self) -> None:
        self.narrative_text.delete("1.0", tk.END)
        if not self.groups:
            self.narrative_text.insert(tk.END, "（尚无相位组：请打开 r.in 或点击「新增组」）\n")
            return
        try:
            notes = [
                "总备注：",
                "- 若震相归类为 PPS/PSS 且出现 cbnd=1，cbnd=1 视为炮点出射设置（名义 P、即刻按 S 出射），不计入转换次数。",
                "- PSP 为两次真实界面转换，不使用 cbnd=1 做出射设置扣减。",
                "",
            ]
            lines = describe_all_phase_groups(self.groups)
            self.narrative_text.insert(tk.END, "\n".join(notes) + "\n\n".join(lines) + "\n")
        except Exception as e:
            self.narrative_text.insert(tk.END, f"生成文字说明失败: {e}\n")

    def _fmt_arr(self, name: str, vals: list[int] | list[float]) -> str:
        if not vals:
            return f"{name}=[]"
        limit = len(vals) if self.preview_full_var.get() else (32 if name in ("rbnd", "cbnd") else 20)
        if isinstance(vals[0], float):
            body = ", ".join(f"{float(v):.1f}" for v in vals[:limit])
        else:
            body = ", ".join(str(int(v)) for v in vals[:limit])
        if len(vals) > limit:
            body += f", ... (total={len(vals)})"
        return f"{name}=[{body}]"

    def _refresh_array_preview_from_groups(self) -> None:
        try:
            arr = compile_phase_groups_to_arrays(self.groups)
            lines = [
                self._fmt_arr("ray", arr["ray"]),  # type: ignore[arg-type]
                self._fmt_arr("nrbnd", arr["nrbnd"]),
                self._fmt_arr("rbnd", arr["rbnd"]),
                self._fmt_arr("ncbnd", arr["ncbnd"]),
                self._fmt_arr("cbnd", arr["cbnd"]),
                self._fmt_arr("nray", arr["nray"]),
                self._fmt_arr("ivray", arr["ivray"]),
            ]
            self._set_array_preview(lines)
        except Exception as e:
            self._set_array_preview([f"数组预览失败: {e}"])
        self._refresh_narrative_from_groups()

    def _update_page_label(self) -> None:
        n = len(self.groups)
        if n == 0:
            self.page_info_var.set("0 组")
            return
        end = min(self._page_start + PAGE_SIZE, n)
        self.page_info_var.set(f"第 {self._page_start + 1}-{end} / 共 {n} 组")

    def _refresh_list_page(self, *, keep_selection_global_idx: int | None = None) -> None:
        for w in self.group_list_inner.winfo_children():
            w.destroy()

        n = len(self.groups)
        self._update_page_label()
        if n == 0:
            self.group_list_canvas.yview_moveto(0)
            self.update_idletasks()
            return
        if self._page_start >= n:
            self._page_start = max(0, ((n - 1) // PAGE_SIZE) * PAGE_SIZE)
        end = min(self._page_start + PAGE_SIZE, n)

        sel = keep_selection_global_idx if keep_selection_global_idx is not None else self._selected_idx

        for i in range(self._page_start, end):
            is_sel = sel is not None and i == sel
            row_style = "SelRow.TFrame" if is_sel else "Row.TFrame"
            label_style = "SelRow.TLabel" if is_sel else "Row.TLabel"
            check_style = "SelRow.TCheckbutton" if is_sel else "Row.TCheckbutton"

            row = ttk.Frame(self.group_list_inner, style=row_style)
            row.pack(side=tk.TOP, fill=tk.X, pady=1)

            v = tk.BooleanVar(value=self.groups[i].enabled)

            def _ret_cmd(gi: int = i, vv: tk.BooleanVar = v) -> None:
                self._on_retain_changed(gi, vv)
                self._focus_phase_list_canvas()

            cb = ttk.Checkbutton(
                row,
                text="保留",
                variable=v,
                style=check_style,
                command=_ret_cmd,
            )
            cb.pack(side=tk.LEFT, padx=(2, 4))
            self._bind_phase_list_keyboard(cb)

            star = "* " if is_sel else "  "
            txt = star + _group_row_label_text(i, self.groups[i])
            lb = ttk.Label(row, text=txt, style=label_style, font=("Consolas", 10), anchor="w", cursor="hand2")
            lb.pack(side=tk.LEFT, fill=tk.X, expand=True)

            def _pick_row(gi: int = i) -> None:
                self._select_row(gi)

            lb.bind("<Button-1>", lambda _e, gi=i: _pick_row(gi))
            try:
                lb.configure(takefocus=True)
            except tk.TclError:
                pass
            self._bind_phase_list_keyboard(lb)

            self._bind_phase_list_keyboard(row)

        self.group_list_canvas.yview_moveto(0)
        self.group_list_canvas.update_idletasks()
        self.group_list_canvas.configure(scrollregion=self.group_list_canvas.bbox("all"))
        self.update_idletasks()

    def _select_row(self, global_idx: int) -> None:
        self._load_detail_for_index(global_idx)
        self._refresh_list_page(keep_selection_global_idx=global_idx)
        # 刷新后焦点会丢失，必须把焦点拉回列表区，方向键才生效
        self.after_idle(self._focus_phase_list_canvas)

    def _select_prev(self, *, page: bool = False) -> None:
        n = len(self.groups)
        if n == 0:
            return
        if self._selected_idx is None:
            idx = 0
        else:
            step = PAGE_SIZE if page else 1
            idx = max(0, self._selected_idx - step)
        self._page_start = (idx // PAGE_SIZE) * PAGE_SIZE
        self._select_row(idx)

    def _select_next(self, *, page: bool = False) -> None:
        n = len(self.groups)
        if n == 0:
            return
        if self._selected_idx is None:
            idx = 0
        else:
            step = PAGE_SIZE if page else 1
            idx = min(n - 1, self._selected_idx + step)
        self._page_start = (idx // PAGE_SIZE) * PAGE_SIZE
        self._select_row(idx)

    def _select_home(self) -> None:
        if not self.groups:
            return
        self._page_start = 0
        self._select_row(0)

    def _select_end(self) -> None:
        n = len(self.groups)
        if n == 0:
            return
        idx = n - 1
        self._page_start = (idx // PAGE_SIZE) * PAGE_SIZE
        self._select_row(idx)

    def _on_retain_changed(self, global_idx: int, var: tk.BooleanVar) -> None:
        if 0 <= global_idx < len(self.groups):
            self.groups[global_idx].enabled = bool(var.get())
        if self._selected_idx == global_idx:
            self.var_enabled.set(self.groups[global_idx].enabled)
        self._refresh_array_preview_from_groups()
        self._refresh_narrative_from_groups()

    def _on_form_retain_toggle(self) -> None:
        if self._selected_idx is None:
            return
        self.groups[self._selected_idx].enabled = bool(self.var_enabled.get())
        self._refresh_list_page(keep_selection_global_idx=self._selected_idx)
        self._refresh_array_preview_from_groups()
        self._refresh_narrative_from_groups()

    def _on_retain_all(self) -> None:
        for g in self.groups:
            g.enabled = True
        if self._selected_idx is not None and 0 <= self._selected_idx < len(self.groups):
            self.var_enabled.set(True)
        self._refresh_list_page(keep_selection_global_idx=self._selected_idx)
        self._refresh_array_preview_from_groups()
        self._refresh_narrative_from_groups()
        self.status_var.set("已全保留（全部组参与写回）")

    def _on_retain_none(self) -> None:
        for g in self.groups:
            g.enabled = False
        if self._selected_idx is not None and 0 <= self._selected_idx < len(self.groups):
            self.var_enabled.set(False)
        self._refresh_list_page(keep_selection_global_idx=self._selected_idx)
        self._refresh_array_preview_from_groups()
        self._refresh_narrative_from_groups()
        self.status_var.set("已全不保留（全部组暂停写回，数据保留可再勾选）")

    def _page_prev(self) -> None:
        self._page_start = max(0, self._page_start - PAGE_SIZE)
        self._refresh_list_page()

    def _page_next(self) -> None:
        n = len(self.groups)
        if n == 0:
            return
        max_start = max(0, ((n - 1) // PAGE_SIZE) * PAGE_SIZE)
        self._page_start = min(self._page_start + PAGE_SIZE, max_start)
        self._refresh_list_page()

    def _page_jump(self) -> None:
        try:
            num = int(self.jump_var.get().strip())
        except ValueError:
            messagebox.showinfo("提示", "请输入整数组号（从 1 开始）")
            return
        n = len(self.groups)
        if n == 0:
            return
        idx = num - 1
        if idx < 0 or idx >= n:
            messagebox.showinfo("提示", f"组号应在 1～{n} 之间")
            return
        self._page_start = (idx // PAGE_SIZE) * PAGE_SIZE
        self._refresh_list_page(keep_selection_global_idx=idx)
        self._load_detail_for_index(idx)

    def _load_detail_for_index(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.groups):
            self._selected_idx = None
            return
        self._selected_idx = idx
        g = self.groups[idx]
        self.var_enabled.set(g.enabled)
        self.var_name.set(g.name)
        self.var_ray.set(f"{g.ray_code:.1f}")
        self.var_nray.set(str(g.nray))
        self.var_ivray.set(str(g.ivray))
        self.var_source_wave.set(g.source_wave)
        self.var_rbnd.set(_rbnd_list_to_text(g.reflections))
        self.var_cbnd.set(_cbnd_list_to_text(g.conversions))

    def _load_file_sync(self, *, full_parse: bool) -> None:
        if self.rin_path is None:
            return
        parse_paths = full_parse or (not self.safe_mode_var.get())
        _debug(f"load start path={self.rin_path} parse_paths={parse_paths}")
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            self.status_var.set("正在解析 r.in …")
            self.update_idletasks()
            arr = parse_rin_phase_arrays_from_file(self.rin_path, parse_path_arrays=parse_paths)
            groups = parse_phase_groups_from_arrays(arr, parse_path_arrays=parse_paths)
            self.groups = groups.groups
            self.misc = parse_rin_misc_params_from_file(self.rin_path)
            self.var_xshot.set("" if self.misc.xshot is None else ",".join(f"{v:.3f}" for v in self.misc.xshot))
            self.var_zshot.set("" if self.misc.zshot is None else ",".join(f"{v:.3f}" for v in self.misc.zshot))
            self.var_vred.set("" if self.misc.vred is None else f"{float(self.misc.vred):.3f}")
            self.var_imodf.set("" if self.misc.imodf is None else str(int(self.misc.imodf)))
            self.var_i2pt.set("" if self.misc.i2pt is None else str(int(self.misc.i2pt)))
            self.var_isrch.set("" if self.misc.isrch is None else str(int(self.misc.isrch)))
            self.var_aamin.set("" if self.misc.aamin is None else f"{float(self.misc.aamin):.3f}")
            self.var_aamax.set("" if self.misc.aamax is None else f"{float(self.misc.aamax):.3f}")
            self.var_space.set("" if self.misc.space is None else ",".join(f"{v:.3f}" for v in self.misc.space))
            self.var_x2pt.set("" if self.misc.x2pt is None else f"{float(self.misc.x2pt):.3f}")
            self.var_ishot.set("" if self.misc.ishot is None else ",".join(str(int(v)) for v in self.misc.ishot))
            self.var_n2pt.set("" if self.misc.n2pt is None else str(int(self.misc.n2pt)))
            self._page_start = 0
            self._selected_idx = None
            self._set_diag(groups.diagnostics)
            lines = [
                self._fmt_arr("ray", arr.ray),
                self._fmt_arr("nrbnd", arr.nrbnd),
                self._fmt_arr("rbnd", arr.rbnd),
                self._fmt_arr("ncbnd", arr.ncbnd),
                self._fmt_arr("cbnd", arr.cbnd),
                self._fmt_arr("nray", arr.nray),
                self._fmt_arr("ivray", arr.ivray),
            ]
            self._set_array_preview(lines)
            if self.groups:
                self._refresh_list_page(keep_selection_global_idx=0)
                self._load_detail_for_index(0)
            else:
                self._refresh_list_page()
            self._refresh_narrative_from_groups()
            self.status_var.set(f"已载入 {self.rin_path.name}，共 {len(self.groups)} 组")
            _debug(f"load ok groups={len(self.groups)}")
        except Exception as e:
            _debug(f"load fail {e!r}")
            messagebox.showerror("读取失败", f"解析 r.in 失败：\n{e}")
            self.status_var.set("解析失败")
        finally:
            self.config(cursor="")

    def _on_open(self) -> None:
        fp = filedialog.askopenfilename(
            title="选择 r.in",
            filetypes=[("RAYINVR input", "r.in"), ("All files", "*.*")],
        )
        if not fp:
            return
        self._open_rin_path(fp)

    def _open_rin_path(self, rin_path: str | Path) -> None:
        p = Path(rin_path)
        if not p.exists():
            messagebox.showerror("打开失败", f"r.in 不存在：\n{p}")
            return
        self.rin_path = p
        self.path_var.set(str(self.rin_path))
        self._load_file_sync(full_parse=False)

    def _on_reload(self) -> None:
        if self.rin_path is None:
            messagebox.showinfo("提示", "请先打开 r.in")
            return
        self._load_file_sync(full_parse=False)

    def _on_reload_full(self) -> None:
        if self.rin_path is None:
            messagebox.showinfo("提示", "请先打开 r.in")
            return
        self._load_file_sync(full_parse=True)

    def _on_add(self) -> None:
        g = PhaseGroup(
            enabled=True,
            name=f"G{len(self.groups) + 1}",
            ray_code=1.1,
            nray=30,
            ivray=max((x.ivray for x in self.groups), default=0) + 1,
            source_wave="P",
        )
        self.groups.append(g)
        new_idx = len(self.groups) - 1
        self._page_start = (new_idx // PAGE_SIZE) * PAGE_SIZE
        self._refresh_list_page(keep_selection_global_idx=new_idx)
        self._load_detail_for_index(new_idx)
        self._refresh_array_preview_from_groups()
        self.status_var.set("已新增一组")

    def _on_clone(self) -> None:
        if self._selected_idx is None:
            return
        src = self.groups[self._selected_idx]
        g = PhaseGroup(
            enabled=src.enabled,
            name=(src.name + "_copy") if src.name else f"G{len(self.groups)+1}",
            ray_code=src.ray_code,
            nray=src.nray,
            ivray=src.ivray,
            source_wave=src.source_wave,
            reflections=[ReflectionEvent(r.boundary, r.direction) for r in src.reflections],
            conversions=[ConversionEvent(c.encounter_idx) for c in src.conversions],
            comment=src.comment,
        )
        ins = self._selected_idx + 1
        self.groups.insert(ins, g)
        self._page_start = (ins // PAGE_SIZE) * PAGE_SIZE
        self._refresh_list_page(keep_selection_global_idx=ins)
        self._load_detail_for_index(ins)
        self._refresh_array_preview_from_groups()
        self.status_var.set("已复制当前组")

    def _on_remove(self) -> None:
        if self._selected_idx is None:
            return
        del self.groups[self._selected_idx]
        n = len(self.groups)
        if n == 0:
            self._selected_idx = None
            self._page_start = 0
            self._refresh_list_page()
            self._refresh_array_preview_from_groups()
            self.status_var.set("已删除，列表为空")
            return
        new_idx = min(self._selected_idx, n - 1)
        self._page_start = (new_idx // PAGE_SIZE) * PAGE_SIZE
        self._refresh_list_page(keep_selection_global_idx=new_idx)
        self._load_detail_for_index(new_idx)
        self._refresh_array_preview_from_groups()
        self.status_var.set("已删除当前组")

    def _on_toggle_enabled(self) -> None:
        if self._selected_idx is None:
            return
        self.groups[self._selected_idx].enabled = not self.groups[self._selected_idx].enabled
        self.var_enabled.set(self.groups[self._selected_idx].enabled)
        self._refresh_list_page(keep_selection_global_idx=self._selected_idx)
        self._refresh_array_preview_from_groups()
        self._refresh_narrative_from_groups()

    def _on_apply_row(self) -> None:
        if self._selected_idx is None:
            return
        try:
            g = self.groups[self._selected_idx]
            g.enabled = bool(self.var_enabled.get())
            g.name = self.var_name.get().strip()
            g.ray_code = float(self.var_ray.get().strip())
            g.nray = int(self.var_nray.get().strip())
            g.ivray = int(self.var_ivray.get().strip())
            sw = self.var_source_wave.get().strip().upper()
            if sw not in ("P", "S"):
                raise ValueError("source_wave 只能为 P 或 S")
            g.source_wave = sw
            g.reflections = _parse_reflections(self.var_rbnd.get())
            g.conversions = _parse_conversions(self.var_cbnd.get())
        except Exception as e:
            messagebox.showerror("输入错误", str(e))
            return
        self._refresh_list_page(keep_selection_global_idx=self._selected_idx)
        self._refresh_array_preview_from_groups()
        self.status_var.set("当前组已更新")

    def _on_apply_misc(self) -> None:
        xs = self.var_xshot.get().strip()
        zs = self.var_zshot.get().strip()
        vr = self.var_vred.get().strip()
        imodf = self.var_imodf.get().strip()
        i2pt = self.var_i2pt.get().strip()
        isrch = self.var_isrch.get().strip()
        aamin = self.var_aamin.get().strip()
        aamax = self.var_aamax.get().strip()
        ishot = self.var_ishot.get().strip()
        n2pt = self.var_n2pt.get().strip()
        space = self.var_space.get().strip()
        x2pt = self.var_x2pt.get().strip()
        try:
            self.misc.xshot = _parse_float_list(xs) if xs else None
        except Exception as e:
            messagebox.showerror("输入错误", f"xshot 解析失败（请填逗号分隔数字，如 0,1,2）：\n{e}")
            return
        try:
            self.misc.zshot = _parse_float_list(zs) if zs else None
        except Exception as e:
            messagebox.showerror("输入错误", f"zshot 解析失败（请填逗号分隔数字，如 0,0,0）：\n{e}")
            return
        try:
            self.misc.vred = float(vr) if vr else None
        except Exception as e:
            messagebox.showerror("输入错误", f"vred 解析失败（请填单个数字，如 4.0）：\n{e}")
            return
        try:
            self.misc.imodf = int(imodf) if imodf else None
        except Exception as e:
            messagebox.showerror("输入错误", f"imodf 解析失败（请填整数，如 1）：\n{e}")
            return
        try:
            self.misc.i2pt = int(i2pt) if i2pt else None
        except Exception as e:
            messagebox.showerror("输入错误", f"i2pt 解析失败（请填整数，如 1）：\n{e}")
            return
        try:
            self.misc.isrch = int(isrch) if isrch else None
        except Exception as e:
            messagebox.showerror("输入错误", f"isrch 解析失败（请填整数，如 1）：\n{e}")
            return
        try:
            self.misc.aamin = float(aamin) if aamin else None
        except Exception as e:
            messagebox.showerror("输入错误", f"aamin 解析失败（请填数字，如 0 或 0.0）：\n{e}")
            return
        try:
            self.misc.aamax = float(aamax) if aamax else None
        except Exception as e:
            messagebox.showerror("输入错误", f"aamax 解析失败（请填数字，如 90 或 90.0）：\n{e}")
            return
        try:
            self.misc.x2pt = float(x2pt) if x2pt else None
        except Exception as e:
            messagebox.showerror("输入错误", f"x2pt 解析失败（请填数字，如 0.1）：\n{e}")
            return
        try:
            self.misc.ishot = _parse_int_list(ishot) if ishot else None
        except Exception as e:
            messagebox.showerror("输入错误", f"ishot 解析失败（请填整数列表，如 1,2,3）：\n{e}")
            return
        try:
            self.misc.n2pt = int(n2pt) if n2pt else None
        except Exception as e:
            messagebox.showerror("输入错误", f"n2pt 解析失败（请填整数，如 0）：\n{e}")
            return
        try:
            self.misc.space = _parse_float_list(space) if space else None
        except Exception as e:
            messagebox.showerror("输入错误", f"space 解析失败（请填数字列表，如 1,2,3 或 1 2 3）：\n{e}")
            return
        self._refresh_array_preview_from_groups()
        self.status_var.set("全局参数已更新（保存后写回 r.in）")

    def _on_validate(self) -> None:
        lines: list[str] = []
        for i, g in enumerate(self.groups, start=1):
            try:
                lnum = int(g.ray_code)
                frac = round(g.ray_code - lnum, 1)
                if frac not in (0.0, 0.1, 0.2, 0.3):
                    lines.append(f"组{i}: ray_code {g.ray_code} 非 L.0/L.1/L.2/L.3")
                if g.nray < 0:
                    lines.append(f"组{i}: nray 不能为负")
                if g.ivray == 0:
                    lines.append(f"组{i}: ivray 不能为 0")
                if any(c.encounter_idx < 0 for c in g.conversions):
                    lines.append(f"组{i}: cbnd 不能含负值")
                if any(r.boundary <= 0 for r in g.reflections):
                    lines.append(f"组{i}: rbnd 边界必须为正整数")
            except Exception as e:
                lines.append(f"组{i}: {e}")

        sig: dict[float, set[tuple[tuple[int, ...], tuple[int, ...]]]] = {}
        for g in self.groups:
            rb = tuple(r.boundary if r.direction == "down_reflect_up" else -r.boundary for r in g.reflections)
            cb = tuple(c.encounter_idx for c in g.conversions)
            sig.setdefault(g.ray_code, set()).add((rb, cb))
        for code, s in sig.items():
            if len(s) > 1:
                lines.append(f"ray code {code} 存在多路径，建议 r.in 里 isrch=1")

        self._set_diag(lines)
        self.status_var.set(f"校验完成：{len(lines)} 条提示" if lines else "校验通过")

    def _on_save(self) -> None:
        if self.rin_path is None:
            messagebox.showinfo("提示", "请先打开 r.in")
            return
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            ok, msg, diagnostics = apply_phase_groups_and_misc_to_rin_file(self.rin_path, self.groups, self.misc)
        finally:
            self.config(cursor="")
        if not ok:
            messagebox.showerror("保存失败", msg)
            self.status_var.set(msg)
            return
        lines = [msg, *diagnostics]
        self._set_diag(lines)
        self._refresh_array_preview_from_groups()
        self.status_var.set(f"已写回 {self.rin_path.name}")
        # 保存成功不弹窗，避免频繁打断；失败仍用弹窗提示

    def _on_close(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RAYINVR r.in 相位组编辑 GUI")
    ap.add_argument("rin", nargs="?", default="", help="可选：启动时自动打开的 r.in 路径")
    args = ap.parse_args(argv)
    RinPhaseGroupsEditor(initial_rin_path=args.rin or None).mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
