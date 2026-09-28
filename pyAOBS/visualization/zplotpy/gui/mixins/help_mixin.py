# -*- coding: utf-8 -*-
"""Help / shortcuts mixed into QtFastViewer."""

from __future__ import annotations

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as exc:  # pragma: no cover
    raise RuntimeError("未安装 PySide6") from exc


class HelpMixin:
    """帮助、快捷键与悬停提示。"""

    def _request_shortcut_quit(self) -> None:
        """Q 键退出：仅独立窗口有效；嵌入工区时忽略。"""
        if not bool(getattr(self, "_allow_shortcut_quit", True)):
            try:
                self._set_status_text(
                    "嵌入工区时 Q 不退出；请用工程主窗「文件→退出」或关闭工区窗口",
                    hold_ms=2500,
                )
            except Exception:
                pass
            return
        self.close()


    def set_allow_shortcut_quit(self, allow: bool) -> None:
        """独立启动 True；嵌入 RelocationProjectWindow 时 False。"""
        self._allow_shortcut_quit = bool(allow)


    def _setup_shortcuts(self) -> None:
        """全局窗口快捷键（避免焦点在子控件时按键失效）。"""
        shortcut_map = [
            ("Q", self._request_shortcut_quit),
            ("X", self._toggle_remove_nearest_trace),
            ("D", self._handle_delete_range_key),
            ("P", self._toggle_pick_mode),
            ("C", self._run_interp_pick),
            ("A", self._run_pick_alignment),
            ("F", self._run_adaptive_alignment),
            ("Shift+F", self._show_stacking_evaluation),
            ("S", self._save_picks_to_hdr),
            ("1", lambda: self._set_apick_from_shortcut(1)),
            ("2", lambda: self._set_apick_from_shortcut(2)),
            ("3", lambda: self._set_apick_from_shortcut(3)),
            ("[", lambda: self._shift_apick(-1)),
            ("]", lambda: self._shift_apick(1)),
            ("Ctrl+S", self._save_picks),
            ("Ctrl+Z", self._undo_last_pick_edit),
            ("Ctrl+Y", self._redo_last_pick_edit),
            ("Ctrl+R", lambda: self.request_render(immediate=True)),
            ("Left", self._prev_record),
            ("Right", self._next_record),
            ("T", self._calculate_theoretical_traveltime_dialog),
            ("Shift+T", self._clear_theoretical_traveltime),
            ("W", self._calculate_water_layer_correction_dialog),
            ("Shift+W", self._show_water_correction_curve),
            ("I", self._show_trace_info),
            ("M", self._toggle_mute_polygon_mode),
            ("Shift+M", self._toggle_mute_invert),
            ("Delete", self._delete_selected_mute_vertex),
            ("V", self._add_waveform_selection_at_cursor),
            ("Shift+V", self._remove_last_waveform_selection),
            ("Z", lambda: self._zoom_at_cursor(0.7)),
            ("O", lambda: self._zoom_at_cursor(1.3)),
            ("H", self._show_help),
            ("F1", self._show_help),
            ("Ctrl+O", self._choose_dfile),
        ]
        for key, handler in shortcut_map:
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            sc.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(handler)
            self._shortcuts.append(sc)


    def _setup_hover_help_bindings(self) -> None:
        """绑定悬停说明：默认短说明，按住 Shift 显示详细说明。"""
        widget_map: Dict[str, QtWidgets.QWidget] = {
            "open_z": self.btn_open,
            "open_hdr": self.btn_open_hdr,
            "open_r": self.btn_open_rec,
            "reload": self.btn_reload,
            "save_params": self.btn_save_params,
            "load_params": self.btn_load_params,
            "save_z": self.btn_save_z,
            "data_info": self.btn_data_info,
            "location_map": self.btn_location_map,
            "export_fig": self.btn_export_fig,
            "prev_rec": self.btn_prev_rec,
            "next_rec": self.btn_next_rec,
            "theory": self.btn_theory,
            "clear_theory": self.btn_clear_theory,
            "water_corr": self.btn_water_corr,
            "clear_water": self.btn_clear_water,
            "water_curve": self.btn_water_curve,
            "load_txin": self.btn_load_txin,
            "clear_txin": self.btn_clear_txin,
            "preview_map_txin": self.btn_preview_map_txin,
            "map_txin": self.btn_map_txin,
            "map_txin_apick_only": self.chk_map_txin_apick_only,
            "map_txin_tol": self.spin_map_txin_tol,
            "map_txin_view_only": self.chk_map_txin_view_only,
            "theme": self.btn_theme,
            "undo_pick": self.btn_undo_pick,
            "redo_pick": self.btn_redo_pick,
            "toggle_panels": self.btn_toggle_panels,
            "irec": self.spin_irec,
            "itype": self.combo_itype,
            "nskip": self.spin_nskip,
            "ndecim": self.spin_ndecim,
            "vred": self.spin_vred,
            "xmin": self.spin_xmin,
            "xmax": self.spin_xmax,
            "tmin": self.spin_tmin,
            "tmax": self.spin_tmax,
            "mode": self.combo_mode,
            "rt_shade": self.chk_rt_shade,
            "amp": self.spin_amp,
            "iscale": self.combo_iscale,
            "rcor": self.spin_rcor,
            "sf": self.spin_sf,
            "tvg": self.spin_tvg,
            "pvg": self.spin_pvg,
            "clip": self.spin_clip,
            "dscale": self.spin_dscale,
            "gain_preset_balanced": self.btn_gain_preset_balanced,
            "far_offset_boost": self.btn_far_offset_boost,
            "gain_preset_strong": self.btn_gain_preset_strong,
            "filter_on": self.chk_filter,
            "gain_on": self.chk_gain,
            "rmean": self.chk_rmean,
            "rtrend": self.chk_rtrend,
            "freqlo": self.spin_freqlo,
            "freqhi": self.spin_freqhi,
            "npoles": self.spin_npoles,
            "izerop": self.chk_zerop,
            "denoise_enabled": self.chk_denoise_enabled,
            "denoise_ab_raw": self.chk_denoise_ab_raw,
            "denoise_show_diff": self.chk_denoise_show_diff,
            "denoise_diff_gain": self.combo_denoise_diff_gain,
            "denoise_start": self.btn_denoise_start,
            "denoise_scope": self.combo_denoise_scope,
            "denoise_clear_selected": self.btn_denoise_clear_selected,
            "denoise_coh_cfg": self.btn_denoise_coh_cfg,
            "denoise_compare_plot": self.btn_denoise_compare_plot,
            "denoise_f_s": self.spin_denoise_f_s,
            "denoise_f_e": self.spin_denoise_f_e,
            "denoise_strength": self.spin_denoise_strength,
            "denoise_bwconn": self.combo_denoise_bwconn,
            "denoise_workers": self.spin_denoise_workers,
            "pick_mode": self.chk_pick_mode,
            "apick": self.spin_apick,
            "pick_size": self.spin_pick_size,
            "tcrcor": self.spin_tcrcor,
            "tlag": self.spin_tlag,
            "hilbratio": self.spin_hilbratio,
            "auto_pick": self.btn_auto_pick,
            "interp_pick": self.btn_interp_pick,
            "save_picks": self.btn_save_picks,
            "undo_pick": self.btn_undo_pick,
            "redo_pick": self.btn_redo_pick,
            "save_hdr": self.btn_save_hdr,
            "write_txin": self.btn_write_txin,
            "clear_picks": self.btn_clear_picks,
            "align_pick": self.btn_align_pick,
            "align_adaptive": self.btn_align_adaptive,
            "eval_stack": self.btn_eval_stack,
            "waveop_stack": self.btn_waveop_stack,
            "waveop_att": self.btn_waveop_att,
            "waveop_clear": self.btn_waveop_clear,
            "waveop_save": self.btn_waveop_save,
            "waveop_load": self.btn_waveop_load,
            "static_corr": self.btn_static_corr,
            "clear_static": self.btn_clear_static,
            "clear_align": self.btn_clear_align,
            "show_stack": self.chk_show_stack,
        }
        self._help_widgets: Dict[QtWidgets.QWidget, str] = {}
        for key, widget in widget_map.items():
            if widget is None:
                continue
            text = self._param_help_texts.get(key, "")
            if text:
                widget.setToolTip(text + "\n(按住 Shift 查看详细说明)")
            widget.installEventFilter(self)
            self._help_widgets[widget] = key
        # 监听 Shift 键状态变化
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self)


    def _compose_help_text(self, key: str) -> str:
        short = self._param_help_texts.get(key, "")
        app = QtWidgets.QApplication.instance()
        has_shift = bool(self._shift_pressed)
        if app is not None:
            try:
                has_shift = has_shift or bool(
                    app.keyboardModifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier
                )
            except Exception:
                pass
        if not has_shift:
            return short
        detailed = self._param_help_texts_detailed.get(key, short)
        return f"详细说明：{detailed}"


    def _group_hover_from_global_pos(self, global_pos: QtCore.QPoint) -> Tuple[Optional[QtWidgets.QGroupBox], QtCore.QPoint]:
        """根据全局坐标定位当前悬停面板及其局部坐标。"""
        for group in getattr(self, "_param_groups", []):
            try:
                local_pos = group.mapFromGlobal(global_pos)
                if group.rect().contains(local_pos):
                    return group, local_pos
            except Exception:
                continue
        return None, QtCore.QPoint()


    def _build_help_menu(self) -> None:
        """独立启动时顶部菜单「帮助」：单条目打开 HELP.md（嵌入工区则由主窗工具栏提供）。"""
        try:
            # 嵌入工区时已是普通 Widget，菜单由工程主窗提供，勿再挂本窗菜单栏
            if not bool(self.windowFlags() & QtCore.Qt.WindowType.Window):
                return
            mb = self.menuBar()
        except Exception:
            return
        m_help = None
        for act in mb.actions():
            if act.menu() is not None and act.text().replace("&", "") == "帮助":
                m_help = act.menu()
                break
        if m_help is None:
            m_help = mb.addMenu("帮助")
        else:
            m_help.clear()
        # 等菜单释放 mouse grab 后再弹窗，消除 Qt 警告
        m_help.addAction("帮助", lambda: QtCore.QTimer.singleShot(0, self._show_help))


    def _show_help(self) -> None:
        """完整帮助（含快捷键与关于）：优先打开 docs/HELP.md（非模态）。"""
        try:
            try:
                from .help_dialog import show_help_dialog
            except ImportError:
                from pyAOBS.visualization.zplotpy.gui.help_dialog import show_help_dialog

            show_help_dialog(activate=True)
            return
        except Exception:
            pass
        embedded = not bool(self.windowFlags() & QtCore.Qt.WindowType.Window)
        v_list_hint = (
            "工区底栏页签「输出 | V段」"
            if embedded
            else "剖面下方「V段」薄条；嵌入工区后迁到主窗底栏"
        )
        text = (
            "ZPLOT Qt 版帮助（摘要）\n\n"
            "完整说明见 visualization/zplotpy/docs/HELP.md\n"
            "（工区主窗：工具栏「帮助」/ F1；波形台：H / F1）\n\n"
            f"V 段列表：{v_list_hint}\n"
            "波形操作：叠加 / 清除V / 存V / 载V\n"
            "姿态联合反演请用独立工区：python -m pyAOBS.processors.relocation.gui\n"
            "常用：P 拾取，V/Shift+V 选波，A 对齐，F 自适应，S/Ctrl+S 存拾取\n"
            "H / F1 打开本帮助"
            + (
                "\n嵌入工区：Q 退出已禁用，请用主窗工具栏「退出」。"
                if not bool(getattr(self, "_allow_shortcut_quit", True))
                else ""
            )
        )
        self._show_themed_info("帮助", text)


    def _show_shortcuts(self) -> None:
        """兼容旧入口：快捷键已并入 HELP.md。"""
        self._show_help()


    def _show_about(self) -> None:
        """兼容旧入口：关于已并入 HELP.md。"""
        self._show_help()

