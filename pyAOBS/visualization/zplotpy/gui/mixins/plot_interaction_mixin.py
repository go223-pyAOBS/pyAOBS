# -*- coding: utf-8 -*-
"""Plot mouse / keyboard interaction mixed into QtFastViewer."""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as exc:  # pragma: no cover
    raise RuntimeError("未安装 PySide6") from exc

from pyAOBS.utils.qt_combo import defer_after_combo_popup, hide_combo_popup


class PlotInteractionMixin:
    """事件绑定、鼠标拾取/选道、键盘删除与缩放。"""

    def _wire_events(self) -> None:
        self._shortcuts: List[QtGui.QShortcut] = []
        theme_menu = QtWidgets.QMenu(self.btn_theme)
        theme_menu.addAction("默认主题", lambda: self._apply_optional_theme("default"))
        theme_menu.addAction("浅蓝", lambda: self._apply_optional_theme("light"))
        theme_menu.addAction("深青", lambda: self._apply_optional_theme("dark"))
        theme_menu.addSeparator()
        theme_menu.addAction("Solarized 浅色", lambda: self._apply_optional_theme("solarized_light"))
        theme_menu.addAction("Solarized 深色", lambda: self._apply_optional_theme("solarized_dark"))
        theme_menu.addAction("Nord", lambda: self._apply_optional_theme("nord"))
        theme_menu.addAction("石墨灰", lambda: self._apply_optional_theme("graphite"))
        theme_menu.addAction("森林绿", lambda: self._apply_optional_theme("forest"))
        self.btn_theme.setMenu(theme_menu)
        self.btn_open.clicked.connect(self._choose_dfile)
        self.btn_open_hdr.clicked.connect(self._choose_hfile)
        self.btn_open_rec.clicked.connect(self._choose_rfile)
        self.btn_reload.clicked.connect(lambda: self.request_render(immediate=True))
        # 保存/加载参数已取消（改由工区自动存取）；保留方法供脚本调用
        self.btn_save_z.clicked.connect(self._save_z_with_picks)
        self.btn_data_info.clicked.connect(self._show_data_information)
        self.btn_location_map.clicked.connect(self._show_location_map)
        self.btn_export_fig.clicked.connect(self._export_figure)
        self.btn_prev_rec.clicked.connect(self._prev_record)
        self.btn_next_rec.clicked.connect(self._next_record)
        self.btn_theory.clicked.connect(self._calculate_theoretical_traveltime_dialog)
        self.btn_clear_theory.clicked.connect(self._clear_theoretical_traveltime)
        self.btn_water_corr.clicked.connect(self._calculate_water_layer_correction_dialog)
        self.btn_clear_water.clicked.connect(self._clear_water_layer_correction)
        self.btn_water_curve.clicked.connect(self._show_water_correction_curve)
        self.btn_load_txin.clicked.connect(self._load_txin_overlay)
        self.btn_clear_txin.clicked.connect(self._clear_txin_overlay)
        self.btn_preview_map_txin.clicked.connect(self._preview_txin_mapping)
        self.btn_map_txin.clicked.connect(self._map_txin_to_picks)
        self.btn_mute_status.clicked.connect(self._toggle_mute_polygon_mode)
        self.chk_mute_invert.toggled.connect(self._on_mute_invert_toggled)
        self.btn_clear_mute.clicked.connect(self._clear_mute_all)
        self.btn_toggle_panels.clicked.connect(self._toggle_params_panel)
        self.btn_save_picks.clicked.connect(self._save_picks)
        self.btn_undo_pick.clicked.connect(self._undo_last_pick_edit)
        self.btn_redo_pick.clicked.connect(self._redo_last_pick_edit)
        self.btn_save_hdr.clicked.connect(self._save_picks_to_hdr)
        self.btn_write_txin.clicked.connect(self._write_txin)
        self.btn_clear_picks.clicked.connect(self._clear_picks)
        self._build_help_menu()
        self.btn_auto_pick.clicked.connect(self._run_auto_pick)
        self.btn_interp_pick.clicked.connect(self._run_interp_pick)
        self.btn_align_pick.clicked.connect(self._run_pick_alignment)
        self.btn_align_adaptive.clicked.connect(self._run_adaptive_alignment)
        self.btn_eval_stack.clicked.connect(self._show_stacking_evaluation)
        self.btn_waveop_stack.clicked.connect(self._run_waveop_stack_from_selections)
        self.btn_waveop_att.clicked.connect(self._on_waveop_att_clicked)
        self.btn_waveop_clear.clicked.connect(self._clear_waveform_selections)
        self.btn_waveop_save.clicked.connect(self._save_waveop_state)
        self.btn_waveop_load.clicked.connect(self._load_waveop_state)
        self.btn_static_corr.clicked.connect(self._calculate_static_correction_dialog)
        self.btn_clear_static.clicked.connect(self._clear_static_correction)
        self.btn_clear_align.clicked.connect(self._clear_alignment)
        self.btn_gain_preset_balanced.clicked.connect(self._apply_gain_preset_balanced)
        self.btn_far_offset_boost.clicked.connect(self._apply_far_offset_boost)
        self.btn_gain_preset_strong.clicked.connect(self._apply_gain_preset_strong)

        self.spin_irec.valueChanged.connect(lambda _: self.request_render())
        self.combo_itype.currentIndexChanged.connect(self._on_combo_request_render)
        self.spin_nskip.valueChanged.connect(lambda _: self.request_render())
        self.spin_ndecim.valueChanged.connect(lambda _: self.request_render())
        self.spin_vred.valueChanged.connect(lambda _: self.request_render())
        self.spin_vred.valueChanged.connect(lambda _: self._update_y_axis_label())
        self.spin_xmin.valueChanged.connect(lambda _: self._apply_window_from_controls())
        self.spin_xmax.valueChanged.connect(lambda _: self._apply_window_from_controls())
        self.spin_tmin.valueChanged.connect(lambda _: self._apply_window_from_controls())
        self.spin_tmax.valueChanged.connect(lambda _: self._apply_window_from_controls())
        self.spin_amp.valueChanged.connect(lambda _: self.request_render())
        self.combo_iscale.currentIndexChanged.connect(self._on_combo_request_render)
        self.spin_rcor.valueChanged.connect(lambda _: self.request_render())
        self.spin_sf.valueChanged.connect(lambda _: self.request_render())
        self.spin_tvg.valueChanged.connect(lambda _: self.request_render())
        self.spin_pvg.valueChanged.connect(lambda _: self.request_render())
        self.spin_clip.valueChanged.connect(lambda _: self.request_render())
        self.spin_dscale.valueChanged.connect(lambda _: self.request_render())
        self.chk_filter.stateChanged.connect(lambda _: self.request_render())
        self.chk_gain.stateChanged.connect(lambda _: self.request_render())
        self.chk_rmean.stateChanged.connect(lambda _: self._on_rmean_rtrend_toggled("rmean"))
        self.chk_rtrend.stateChanged.connect(lambda _: self._on_rmean_rtrend_toggled("rtrend"))
        self.spin_freqlo.valueChanged.connect(lambda _: self.request_render())
        self.spin_freqhi.valueChanged.connect(lambda _: self.request_render())
        self.spin_npoles.valueChanged.connect(lambda _: self.request_render())
        self.chk_zerop.stateChanged.connect(lambda _: self.request_render())
        self.combo_mode.currentIndexChanged.connect(self._on_combo_request_render)
        self.chk_rt_shade.stateChanged.connect(lambda _: self.request_render())
        self.spin_apick.valueChanged.connect(lambda _: self.request_render())
        self.spin_apick.valueChanged.connect(self._on_apick_changed)
        self.spin_pick_size.valueChanged.connect(lambda _: self.request_render())
        self.spin_tcrcor.valueChanged.connect(lambda _: self.request_render())
        self.spin_tlag.valueChanged.connect(lambda _: self.request_render())
        self.spin_hilbratio.valueChanged.connect(lambda _: self.request_render())
        self.chk_show_stack.stateChanged.connect(self._on_show_stack_changed)
        self.chk_denoise_enabled.stateChanged.connect(self._on_denoise_ui_changed)
        self.chk_denoise_ab_raw.stateChanged.connect(self._on_denoise_view_mode_changed)
        self.chk_denoise_show_diff.stateChanged.connect(self._on_denoise_view_mode_changed)
        self.combo_denoise_diff_gain.currentIndexChanged.connect(self._on_denoise_view_mode_changed)
        self.btn_denoise_start.clicked.connect(self._start_denoise_now)
        self.spin_denoise_f_s.valueChanged.connect(self._on_denoise_ui_changed)
        self.spin_denoise_f_e.valueChanged.connect(self._on_denoise_ui_changed)
        self.spin_denoise_strength.valueChanged.connect(self._on_denoise_ui_changed)
        self.combo_denoise_bwconn.currentIndexChanged.connect(self._on_denoise_ui_changed)
        self.spin_denoise_workers.valueChanged.connect(self._on_denoise_ui_changed)
        self.combo_denoise_scope.currentIndexChanged.connect(self._on_denoise_ui_changed)
        self.chk_denoise_pick_guidance.stateChanged.connect(self._on_denoise_ui_changed)
        self.spin_denoise_pick_hw.valueChanged.connect(self._on_denoise_ui_changed)
        self.spin_denoise_pick_floor.valueChanged.connect(self._on_denoise_ui_changed)
        self.btn_denoise_clear_selected.clicked.connect(self._clear_denoise_selected_traces)
        self.btn_denoise_coh_cfg.clicked.connect(self._open_denoise_coh_dialog)
        self.btn_denoise_compare_plot.clicked.connect(self._open_denoise_compare_plot)
        self.combo_iscale.currentIndexChanged.connect(
            lambda _i: defer_after_combo_popup(self._update_gain_effect_hint, self.combo_iscale)
        )
        self.spin_sf.valueChanged.connect(lambda _: self._update_gain_effect_hint())
        self.spin_rcor.valueChanged.connect(lambda _: self._update_gain_effect_hint())
        self.spin_amp.valueChanged.connect(lambda _: self._update_gain_effect_hint())
        self.spin_tvg.valueChanged.connect(lambda _: self._update_gain_effect_hint())
        self.spin_pvg.valueChanged.connect(lambda _: self._update_gain_effect_hint())
        self.spin_clip.valueChanged.connect(lambda _: self._update_gain_effect_hint())
        self.spin_dscale.valueChanged.connect(lambda _: self._update_gain_effect_hint())
        self._update_y_axis_label()
        self._update_denoise_hint()

        vb = self.plot.getViewBox()
        vb.sigRangeChanged.connect(self._on_view_range_changed)
        self.plot.scene().sigMouseClicked.connect(self._on_plot_mouse_clicked)
        self.plot.scene().sigMouseMoved.connect(self._on_plot_mouse_moved)
        try:
            self.plot.scene().installEventFilter(self)
        except Exception:
            pass
        try:
            self.plot.installEventFilter(self)
        except Exception:
            pass
        try:
            vp = self.plot.viewport()
            if vp is not None:
                vp.installEventFilter(self)
        except Exception:
            pass
        self._refresh_waveop_selection_list()
        self._setup_hover_help_bindings()
        self._setup_shortcuts()
        self._update_mute_status_button()


    def eventFilter(self, obj, event):
        et = event.type()
        # 图窗兜底：scene / plot / viewport 任一层收到鼠标事件都能触发手动选道
        if self._is_denoise_select_mode_active():
            is_scene_layer = obj is self.plot.scene()
            vp = None
            try:
                vp = self.plot.viewport()
            except Exception:
                vp = None
            is_plot_layer = (obj is self.plot) or (obj is vp)
            if is_scene_layer or is_plot_layer:
                is_left_btn, is_right_btn = self._event_mouse_button_flags(event)
                scene_pos = self._extract_scene_pos_from_event(obj, event)
                layer = "scene" if is_scene_layer else ("plot" if obj is self.plot else "viewport")
                if et in (QtCore.QEvent.Type.GraphicsSceneMousePress, QtCore.QEvent.Type.MouseButtonPress):
                    select_with_mod = self._is_denoise_select_modifier_active()
                    self._debug_log(
                        "DENOISE_EVT",
                        f"press layer={layer} left={int(is_left_btn)} right={int(is_right_btn)} has_pos={int(scene_pos is not None)} mod={int(select_with_mod)}",
                    )
                    if scene_pos is not None and (is_right_btn or (is_left_btn and select_with_mod)):
                        t_idx = self._nearest_render_trace_idx_by_scene_pos(scene_pos)
                        self._denoise_click_pending_trace_idx = t_idx
                        self._denoise_click_pending_remove_only = bool(is_right_btn)
                    if (not is_right_btn) and (not select_with_mod):
                        # 无修饰键左键：交给 ViewBox 默认拖曳/缩放交互
                        self._denoise_click_pending_trace_idx = None
                        return super().eventFilter(obj, event)
                    if is_left_btn and select_with_mod and self._last_render_trace_indices.size > 0 and scene_pos is not None:
                        try:
                            vb = self.plot.getViewBox()
                            mp = vb.mapSceneToView(scene_pos)
                            self._denoise_select_drag_active = True
                            self._denoise_select_drag_start_x = float(mp.x())
                            self._denoise_select_drag_last_x = float(mp.x())
                            mods = QtWidgets.QApplication.keyboardModifiers()
                            if mods & QtCore.Qt.KeyboardModifier.ControlModifier:
                                self._denoise_select_drag_mode = "replace"
                            elif mods & QtCore.Qt.KeyboardModifier.AltModifier:
                                self._denoise_select_drag_mode = "remove"
                            else:
                                self._denoise_select_drag_mode = "add"
                            self._set_plot_pan_enabled(False)
                        except Exception:
                            pass
                        return True
                elif et in (QtCore.QEvent.Type.GraphicsSceneMouseRelease, QtCore.QEvent.Type.MouseButtonRelease):
                    select_with_mod = self._is_denoise_select_modifier_active()
                    self._debug_log(
                        "DENOISE_EVT",
                        f"release layer={layer} left={int(is_left_btn)} right={int(is_right_btn)} has_pos={int(scene_pos is not None)} pending={self._denoise_click_pending_trace_idx} mod={int(select_with_mod)}",
                    )
                    did_batch_select = False
                    if self._denoise_select_drag_active:
                        did_batch_select = bool(self._finish_denoise_drag_select())
                    if did_batch_select:
                        self._sync_plot_pan_lock_state()
                        return True
                    if (not is_right_btn) and (not select_with_mod):
                        self._denoise_click_pending_trace_idx = None
                        return super().eventFilter(obj, event)
                    if scene_pos is not None and is_left_btn and select_with_mod:
                        if self._toggle_denoise_selected_trace_by_scene_pos(scene_pos, remove_only=False):
                            self._denoise_click_pending_trace_idx = None
                            return True
                    elif scene_pos is not None and is_right_btn:
                        if self._toggle_denoise_selected_trace_by_scene_pos(scene_pos, remove_only=True):
                            self._denoise_click_pending_trace_idx = None
                            return True
                    if self._denoise_click_pending_trace_idx is not None:
                        if self._toggle_denoise_selected_trace(
                            int(self._denoise_click_pending_trace_idx),
                            remove_only=bool(self._denoise_click_pending_remove_only),
                        ):
                            self._denoise_click_pending_trace_idx = None
                            return True
                    if scene_pos is not None:
                        self._set_status_text(
                            "选道事件已到达，但未命中可选渲染道（可尝试放大/减小抽稀后重试）",
                            hold_ms=900,
                        )
                    self._denoise_click_pending_trace_idx = None

        # 全局跟踪：拖拽幽灵 / 候选激活 / 调宽，即使鼠标离开原面板仍持续跟随
        if et == QtCore.QEvent.Type.MouseMove:
            gpos = self._panel_drag_global_pos_from_event(event)
            if self._panel_resize_active and self._panel_resize_group is not None:
                try:
                    gx = int(gpos.x())
                except Exception:
                    gx = int(QtGui.QCursor.pos().x())
                self._apply_panel_resize_from_global_x(gx)
            elif self._panel_drag_candidate is not None:
                self._promote_panel_drag_if_needed(gpos)
                if self._panel_drag_active:
                    self._update_panel_drag_ghost_position(gpos)
            elif (not self._panel_drag_active) and (not self._panel_resize_active):
                self._refresh_panel_hover_cursors(gpos)

        # 全局结束拖拽/调宽：在任意控件上释放左键都能完成收尾
        if et == QtCore.QEvent.Type.MouseButtonRelease:
            try:
                is_left = getattr(event, "button", lambda: None)() == QtCore.Qt.MouseButton.LeftButton
            except Exception:
                is_left = False
            if is_left:
                gpos = self._panel_drag_global_pos_from_event(event)
                if self._panel_resize_active:
                    self._finish_panel_resize(gpos)
                elif self._panel_drag_candidate is not None:
                    self._finish_panel_drag(gpos)
                    self._refresh_panel_hover_cursors(gpos)

        # 参数面板拖拽重排（标题区域按住左键拖动）与边沿调宽
        if isinstance(obj, QtWidgets.QGroupBox) and obj in getattr(self, "_param_groups", []):
            if et == QtCore.QEvent.Type.Leave:
                if not self._panel_resize_active and self._panel_drag_candidate is None:
                    try:
                        obj.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
                    except Exception:
                        pass
            if et == QtCore.QEvent.Type.MouseButtonPress and getattr(event, "button", lambda: None)() == QtCore.Qt.MouseButton.LeftButton:
                edge = self._panel_resize_hit_edge(obj, event.pos())
                if edge:
                    self._panel_resize_active = True
                    self._panel_resize_group = obj
                    self._panel_resize_edge = edge
                    self._panel_drag_candidate = None
                    self._panel_drag_active = False
                    self._destroy_panel_drag_ghost()
                    try:
                        self._panel_resize_start_x = int(event.globalPosition().x())
                    except Exception:
                        self._panel_resize_start_x = int(QtGui.QCursor.pos().x())
                    self._panel_resize_start_width = int(obj.width())
                    self._grab_panel_mouse(obj)
                    try:
                        obj.setCursor(QtCore.Qt.CursorShape.SizeHorCursor)
                    except Exception:
                        pass
                    try:
                        event.accept()
                    except Exception:
                        pass
                    return True
                elif self._is_panel_title_hit(obj, event.pos()):
                    self._panel_drag_candidate = obj
                    self._panel_drag_active = False
                    self._panel_drag_start_global = self._panel_drag_global_pos_from_event(event)
                    self._panel_drag_hotspot = QtCore.QPoint(
                        max(0, min(int(event.pos().x()), int(obj.width()) - 1)),
                        max(0, min(int(event.pos().y()), int(obj.height()) - 1)),
                    )
                    self._destroy_panel_drag_ghost()
                    self._grab_panel_mouse(obj)
                    try:
                        obj.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
                    except Exception:
                        pass
                    try:
                        event.accept()
                    except Exception:
                        pass
                    return True
            elif et == QtCore.QEvent.Type.MouseMove:
                # 全局分支已处理拖拽/调宽；此处仅刷新本面板光标
                if not self._panel_resize_active and self._panel_drag_candidate is None:
                    self._update_panel_cursor(obj, event.pos())
                elif self._panel_resize_active and self._panel_resize_group is obj:
                    self._update_panel_cursor(obj, event.pos())
                elif self._panel_drag_candidate is obj:
                    self._update_panel_cursor(obj, event.pos())
            elif et == QtCore.QEvent.Type.MouseButtonRelease and getattr(event, "button", lambda: None)() == QtCore.Qt.MouseButton.LeftButton:
                # 收尾已在全局分支处理；此处避免重复
                pass

        if et == QtCore.QEvent.Type.KeyPress and getattr(event, "key", lambda: None)() == QtCore.Qt.Key.Key_Shift:
            if not self._shift_pressed:
                self._shift_hover_pick_active = True
                self._shift_hover_pick_undo_pushed = False
                self._shift_hover_pick_updated_count = 0
                self._shift_hover_picked_traces = set()
            self._shift_pressed = True
            if self._hover_help_key:
                self.lbl_status.setText(self._compose_help_text(self._hover_help_key))
        elif et == QtCore.QEvent.Type.KeyRelease and getattr(event, "key", lambda: None)() == QtCore.Qt.Key.Key_Shift:
            self._shift_pressed = False
            if self._shift_hover_pick_active and self._shift_hover_pick_updated_count > 0:
                self._set_status_text(
                    f"Shift悬停拾取完成：更新 {self._shift_hover_pick_updated_count} 道（当前拾取字）",
                    hold_ms=1800,
                )
            self._shift_hover_pick_active = False
            self._shift_hover_pick_undo_pushed = False
            self._shift_hover_pick_updated_count = 0
            self._shift_hover_picked_traces = set()
            if self._hover_help_key:
                self.lbl_status.setText(self._compose_help_text(self._hover_help_key))
        elif et in (QtCore.QEvent.Type.ShortcutOverride, QtCore.QEvent.Type.InputMethod):
            if self._hover_help_key:
                self.lbl_status.setText(self._compose_help_text(self._hover_help_key))
        elif et == QtCore.QEvent.Type.Enter and hasattr(self, "_help_widgets") and obj in self._help_widgets:
            key = self._help_widgets[obj]
            self._hover_help_key = key
            self.lbl_status.setText(self._compose_help_text(key))
        elif et == QtCore.QEvent.Type.MouseMove and hasattr(self, "_help_widgets") and obj in self._help_widgets:
            # 某些平台上 Shift 状态变化不会触发控件重入，移动时补刷一次
            if self._hover_help_key == self._help_widgets[obj]:
                self.lbl_status.setText(self._compose_help_text(self._hover_help_key))
        elif et == QtCore.QEvent.Type.Leave and hasattr(self, "_help_widgets") and obj in self._help_widgets:
            if self._hover_help_key == self._help_widgets[obj]:
                self._hover_help_key = None
                self.lbl_status.setText("就绪")
        return super().eventFilter(obj, event)


    def _sync_plot_pan_lock_state(self) -> None:
        """统一同步 plot 平移开关，避免与手动选道/编辑模式抢事件。"""
        # 去噪手动选道不再长期锁定平移；仅在明确框选手势期间临时锁定
        lock_pan = bool(self._mute_edit_mode)
        self._set_plot_pan_enabled(not lock_pan)


    def _nearest_render_trace_idx_by_scene_pos(self, scene_pos) -> Optional[int]:
        """根据 scene 坐标返回最近渲染道号。"""
        if self._last_render_trace_indices.size == 0 or self._last_render_offsets.size == 0:
            return None
        vb = self.plot.getViewBox()
        try:
            mouse_pt = vb.mapSceneToView(scene_pos)
        except Exception:
            return None
        x = float(mouse_pt.x())
        nearest_i = int(np.argmin(np.abs(self._last_render_offsets - x)))
        return int(self._last_render_trace_indices[nearest_i])


    def _extract_scene_pos_from_event(self, obj, event):
        """从不同层级鼠标事件中提取 scene 坐标。"""
        # QGraphicsSceneMouseEvent
        try:
            sp = event.scenePos()
            if sp is not None:
                return sp
        except Exception:
            pass
        # QWidget mouse event on PlotWidget / viewport
        local_pos = None
        try:
            lp = event.position()
            if lp is not None:
                local_pos = lp.toPoint()
        except Exception:
            local_pos = None
        if local_pos is None:
            try:
                local_pos = event.pos()
            except Exception:
                local_pos = None
        if local_pos is None:
            return None
        try:
            if obj is self.plot:
                return self.plot.mapToScene(local_pos)
            vp = self.plot.viewport()
            if obj is vp:
                mapped = vp.mapTo(self.plot, local_pos)
                return self.plot.mapToScene(mapped)
        except Exception:
            return None
        return None


    def _event_mouse_button_flags(self, event) -> Tuple[bool, bool]:
        """兼容不同事件类型的左右键判定。"""
        def _enum_to_int(v) -> Optional[int]:
            if v is None:
                return None
            try:
                return int(v)
            except Exception:
                pass
            try:
                vv = getattr(v, "value", None)
                if vv is not None:
                    return int(vv)
            except Exception:
                pass
            return None

        btn = None
        try:
            btn = event.button()
        except Exception:
            btn = None
        btn_i = _enum_to_int(btn)
        left_enum = QtCore.Qt.MouseButton.LeftButton
        right_enum = QtCore.Qt.MouseButton.RightButton
        left_i = _enum_to_int(left_enum)
        right_i = _enum_to_int(right_enum)
        is_left = (btn == left_enum) or (btn_i is not None and left_i is not None and btn_i == left_i)
        is_right = (btn == right_enum) or (btn_i is not None and right_i is not None and btn_i == right_i)
        return is_left, is_right


    def _zoom_at_cursor(self, factor: float) -> None:
        if self.loaded is None:
            return
        vb = self.plot.getViewBox()
        xr, yr = vb.viewRange()
        x_center = self.mouse_x if self.mouse_x is not None else float((xr[0] + xr[1]) * 0.5)
        y_center = self.mouse_y if self.mouse_y is not None else float((yr[0] + yr[1]) * 0.5)
        x_range = float(abs(xr[1] - xr[0]))
        y_range = float(abs(yr[1] - yr[0]))
        new_x = max(1e-9, x_range * float(factor))
        new_y = max(1e-9, y_range * float(factor))
        vb.setXRange(x_center - new_x * 0.5, x_center + new_x * 0.5, padding=0.0)
        vb.setYRange(y_center - new_y * 0.5, y_center + new_y * 0.5, padding=0.0)
        self.lbl_status.setText(
            "已放大" if factor < 1.0 else "已缩小"
            + f"：中心=({x_center:.2f}, {y_center:.2f})"
        )


    def _set_plot_pan_enabled(self, enabled: bool) -> None:
        vb = self.plot.getViewBox()
        vb.setMouseEnabled(x=bool(enabled), y=bool(enabled))


    def _trace_group_key(self, trace_idx: int) -> Optional[Tuple[int, int]]:
        if self.loaded is None:
            return None
        headers = self.loaded.get("trace_headers", []) or []
        if trace_idx < 0 or trace_idx >= len(headers):
            return None
        th = headers[trace_idx]
        shot = int(getattr(th, "ishoti", 0) or 0)
        rec = int(getattr(th, "ireci", 0) or 0)
        if shot <= 0 or rec <= 0:
            return None
        return (shot, rec)


    def _trace_group_indices(self, trace_idx: int) -> List[int]:
        key = self._trace_group_key(int(trace_idx))
        if key is None or self.loaded is None:
            return [int(trace_idx)]
        headers = self.loaded.get("trace_headers", []) or []
        out: List[int] = []
        for i, th in enumerate(headers):
            if int(getattr(th, "ishoti", 0) or 0) == key[0] and int(getattr(th, "ireci", 0) or 0) == key[1]:
                out.append(int(i))
        return out if out else [int(trace_idx)]


    def _on_plot_mouse_clicked(self, ev) -> None:
        if self.loaded is None:
            return
        # 闭合后的快捷编辑：非编辑态下，左键点顶点即可进入编辑（避免“点顶点没反应”）
        if (
            (not self._mute_edit_mode)
            and len(self._mute_polygon_points) >= 3
        ):
            pos = ev.scenePos()
            vb = self.plot.getViewBox()
            if vb.sceneBoundingRect().contains(pos):
                mouse_pt = vb.mapSceneToView(pos)
                x = float(mouse_pt.x())
                y = float(mouse_pt.y())
                near_idx = self._find_near_mute_vertex(x, y)
                if near_idx is not None and ev.button() == QtCore.Qt.MouseButton.LeftButton:
                    self._mute_edit_mode = True
                    self._set_plot_pan_enabled(False)
                    self._mute_drag_vertex_idx = int(near_idx)
                    self._mute_selected_vertex_idx = int(near_idx)
                    self._refresh_mute_polygon_overlay()
                    self._update_mute_status_button()
                    self.lbl_status.setText(f"Mute编辑已开启：选中顶点 #{near_idx + 1}，拖动可调整")
                    return
        if self._mute_edit_mode:
            pos = ev.scenePos()
            vb = self.plot.getViewBox()
            if not vb.sceneBoundingRect().contains(pos):
                return
            mouse_pt = vb.mapSceneToView(pos)
            x = float(mouse_pt.x())
            y = float(mouse_pt.y())
            if ev.button() == QtCore.Qt.MouseButton.RightButton:
                near_idx = self._find_near_mute_vertex(x, y)
                if near_idx is not None:
                    self._mute_selected_vertex_idx = int(near_idx)
                    self._delete_selected_mute_vertex()
                else:
                    self._finalize_mute_polygon()
            elif ev.button() == QtCore.Qt.MouseButton.LeftButton:
                near_idx = self._find_near_mute_vertex(x, y)
                if near_idx is not None:
                    self._mute_drag_vertex_idx = int(near_idx)
                    self._mute_selected_vertex_idx = int(near_idx)
                    self._refresh_mute_polygon_overlay()
                    self.lbl_status.setText(f"Mute编辑：选中顶点 #{near_idx + 1}，拖动可调整")
                    return
                self._mute_polygon_points.append((x, y))
                self._mute_drag_vertex_idx = len(self._mute_polygon_points) - 1
                self._mute_selected_vertex_idx = int(self._mute_drag_vertex_idx)
                self._refresh_mute_polygon_overlay()
                self._update_mute_status_button()
                self.lbl_status.setText(
                    f"Mute绘制中：已添加 {len(self._mute_polygon_points)} 个顶点（右键闭合）"
                )
            elif ev.button() == QtCore.Qt.MouseButton.MiddleButton and self._mute_polygon_points:
                self._mute_polygon_points.pop()
                self._mute_drag_vertex_idx = None
                self._mute_selected_vertex_idx = (len(self._mute_polygon_points) - 1) if self._mute_polygon_points else None
                self._refresh_mute_polygon_overlay()
                self._update_mute_status_button()
                self.lbl_status.setText(
                    f"Mute绘制中：撤销一个顶点，剩余 {len(self._mute_polygon_points)} 个"
                )
            return
        if self._is_denoise_select_mode_active():
            if self._denoise_select_drag_just_finished:
                self._denoise_select_drag_just_finished = False
                return
            pos = ev.scenePos()
            is_left, is_right = self._event_mouse_button_flags(ev)
            if is_left and self._is_denoise_select_modifier_active():
                self._toggle_denoise_selected_trace_by_scene_pos(pos, remove_only=False)
                return
            if is_right:
                self._toggle_denoise_selected_trace_by_scene_pos(pos, remove_only=True)
                return
        if self.pick_manager is None:
            return
        if not self.chk_pick_mode.isChecked():
            return
        if self._last_render_trace_indices.size == 0:
            return
        pos = ev.scenePos()
        vb = self.plot.getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            return
        mouse_pt = vb.mapSceneToView(pos)
        x = float(mouse_pt.x())
        y = float(mouse_pt.y())
        nearest_i = int(np.argmin(np.abs(self._last_render_offsets - x)))
        trace_idx = int(self._last_render_trace_indices[nearest_i])
        x_trace = float(self._last_render_offsets[nearest_i])
        y_pick = float(y) - self._compute_display_tshift(trace_idx, x_trace)
        apick = int(self.spin_apick.value())
        old_pick = self._get_shared_pick(trace_idx, apick)
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            if old_pick is None:
                return
            self._push_pick_undo("鼠标删除拾取")
            self._remove_shared_pick(trace_idx, apick)
        else:
            if old_pick is not None and abs(float(old_pick) - y_pick) < 1e-9:
                return
            self._push_pick_undo("鼠标拾取/改点")
            self._set_shared_pick(trace_idx, apick, y_pick)
        self.request_render(delay_ms=10)


    def _on_plot_mouse_moved(self, pos) -> None:
        if self.loaded is None:
            return
        vb = self.plot.getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            return
        mouse_pt = vb.mapSceneToView(pos)
        self.mouse_x = float(mouse_pt.x())
        self.mouse_y = float(mouse_pt.y())
        if self._mute_edit_mode:
            buttons = QtWidgets.QApplication.mouseButtons()
            if (
                (buttons & QtCore.Qt.MouseButton.LeftButton)
            ):
                # 某些平台上 sigMouseClicked 在“释放”时触发，拖拽开始阶段拿不到 click。
                # 这里在移动事件中兜底捕获“按住左键且靠近顶点”以启动拖拽。
                if self._mute_drag_vertex_idx is None and len(self._mute_polygon_points) > 0:
                    near_idx = self._find_near_mute_vertex(float(self.mouse_x), float(self.mouse_y))
                    if near_idx is not None:
                        self._mute_drag_vertex_idx = int(near_idx)
                        self._mute_selected_vertex_idx = int(near_idx)
                if self._mute_drag_vertex_idx is None:
                    return
                if not (0 <= int(self._mute_drag_vertex_idx) < len(self._mute_polygon_points)):
                    return
                self._mute_polygon_points[int(self._mute_drag_vertex_idx)] = (float(self.mouse_x), float(self.mouse_y))
                self._mute_selected_vertex_idx = int(self._mute_drag_vertex_idx)
                self._mute_drag_active = True
                # 拖拽中使用轻量刷新，避免频繁重建文本标签导致卡顿
                self._refresh_mute_polygon_overlay(update_labels=False)
                self.request_render(delay_ms=25)
                now_ms = int(time.monotonic() * 1000.0)
                if now_ms - int(self._mute_drag_last_status_ms) >= 120:
                    self.lbl_status.setText(
                        f"Mute编辑中：拖拽顶点 #{int(self._mute_drag_vertex_idx) + 1} 到 ({self.mouse_x:.2f}, {self.mouse_y:.3f})"
                    )
                    self._mute_drag_last_status_ms = now_ms
            elif not (buttons & QtCore.Qt.MouseButton.LeftButton):
                if self._mute_drag_active:
                    self._mute_drag_active = False
                    self._refresh_mute_polygon_overlay(update_labels=True)
                    # 松手后立即重绘，确保波形立刻更新
                    self.request_render(immediate=True)
                    self.lbl_status.setText("Mute编辑：顶点拖拽完成，波形已更新")
                self._mute_drag_vertex_idx = None
            return
        if self._is_denoise_select_mode_active():
            buttons = QtWidgets.QApplication.mouseButtons()
            select_with_mod = self._is_denoise_select_modifier_active()
            if (buttons & QtCore.Qt.MouseButton.LeftButton) and select_with_mod:
                if self._last_render_trace_indices.size > 0 and self._last_render_offsets.size > 0:
                    if not self._denoise_select_drag_active:
                        self._denoise_select_drag_active = True
                        self._denoise_select_drag_start_x = float(self.mouse_x)
                        self._denoise_select_drag_last_x = float(self.mouse_x)
                        mods = QtWidgets.QApplication.keyboardModifiers()
                        if mods & QtCore.Qt.KeyboardModifier.ControlModifier:
                            self._denoise_select_drag_mode = "replace"
                        elif mods & QtCore.Qt.KeyboardModifier.AltModifier:
                            self._denoise_select_drag_mode = "remove"
                        else:
                            # 默认与 Shift 一致：追加
                            self._denoise_select_drag_mode = "add"
                        if not self._mute_edit_mode:
                            self._set_plot_pan_enabled(False)
                    else:
                        self._denoise_select_drag_last_x = float(self.mouse_x)
            elif self._denoise_select_drag_active:
                self._finish_denoise_drag_select()
                self._sync_plot_pan_lock_state()
        nearest_trace_idx: Optional[int] = None
        nearest_i_opt: Optional[int] = None
        if self._last_render_trace_indices.size > 0 and self._last_render_offsets.size > 0:
            try:
                nearest_i = int(np.argmin(np.abs(self._last_render_offsets - float(self.mouse_x))))
                nearest_i_opt = nearest_i
                nearest_trace_idx = int(self._last_render_trace_indices[nearest_i])
                self._update_location_map_cursor_for_trace(nearest_trace_idx)
            except Exception:
                nearest_trace_idx = None
                nearest_i_opt = None
        # 实时显示鼠标所在“最近道”的定位参数（拾取/非拾取模式一致）
        if self._last_render_trace_indices.size > 0 and self._last_render_offsets.size > 0:
            try:
                trace_idx = int(nearest_trace_idx) if nearest_trace_idx is not None else int(
                    self._last_render_trace_indices[int(np.argmin(np.abs(self._last_render_offsets - float(self.mouse_x))))]
                )
                if nearest_i_opt is None:
                    nearest_i_opt = int(np.argmin(np.abs(self._last_render_offsets - float(self.mouse_x))))
                x_trace = float(self._last_render_offsets[int(nearest_i_opt)])
                t_fold = float(self.mouse_y)
                t_true = float(t_fold) - float(self._compute_display_tshift(trace_idx, x_trace))
                th = None
                headers = self.loaded.get("trace_headers", []) or []
                if 0 <= trace_idx < len(headers):
                    th = headers[trace_idx]
                sxutm = float(getattr(th, "sxutm", 0.0) or 0.0) if th is not None else 0.0
                syutm = float(getattr(th, "syutm", 0.0) or 0.0) if th is not None else 0.0
                rxutm = float(getattr(th, "rxutm", 0.0) or 0.0) if th is not None else 0.0
                ryutm = float(getattr(th, "ryutm", 0.0) or 0.0) if th is not None else 0.0
                slat = float(getattr(th, "slat", 0.0) or 0.0) if th is not None else 0.0
                slong = float(getattr(th, "slong", 0.0) or 0.0) if th is not None else 0.0
                rlat = float(getattr(th, "rlat", 0.0) or 0.0) if th is not None else 0.0
                rlong = float(getattr(th, "rlong", 0.0) or 0.0) if th is not None else 0.0
                shot = int(getattr(th, "ishoti", 0) or 0) if th is not None else 0
                rec = int(getattr(th, "ireci", 0) or 0) if th is not None else 0
                _, rec_role = self._infer_rec_role_for_orientation()
                # 统一约定：S 为固定不变的震源点，R 为接收点。
                if rec_role == "sx":
                    sxy_utm = (rxutm, ryutm)
                    rxy_utm = (sxutm, syutm)
                    sxy_geo = (rlat, rlong)
                    rxy_geo = (slat, slong)
                else:
                    sxy_utm = (sxutm, syutm)
                    rxy_utm = (rxutm, ryutm)
                    sxy_geo = (slat, slong)
                    rxy_geo = (rlat, rlong)
                if abs(sxutm) > 1e-9 or abs(syutm) > 1e-9 or abs(rxutm) > 1e-9 or abs(ryutm) > 1e-9:
                    self.lbl_status.setText(
                        f"拾取定位 | 道{trace_idx} 炮{shot} 检{rec} offset={x_trace:.3f}km | "
                        f"折合={t_fold:.3f}s 真实={t_true:.3f}s | "
                        f"S=({sxy_utm[0]:.1f},{sxy_utm[1]:.1f})m R=({rxy_utm[0]:.1f},{rxy_utm[1]:.1f})m"
                    )
                else:
                    self.lbl_status.setText(
                        f"拾取定位 | 道{trace_idx} 炮{shot} 检{rec} offset={x_trace:.3f}km | "
                        f"折合={t_fold:.3f}s 真实={t_true:.3f}s | "
                        f"S=({sxy_geo[0]:.5f},{sxy_geo[1]:.5f}) R=({rxy_geo[0]:.5f},{rxy_geo[1]:.5f})"
                    )
            except Exception:
                pass
        self._apply_shift_hover_pick()


    def _toggle_remove_nearest_trace(self) -> None:
        if self.loaded is None:
            self.lbl_status.setText("当前无可操作道")
            return
        self.last_key_class = "x"
        offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
        idx = self._extract_indices(include_removed=True)
        if offsets.size == 0 or idx.size == 0:
            self.lbl_status.setText("当前无可操作道")
            return
        xref = self.mouse_x
        if xref is None:
            x_range = self.plot.getViewBox().viewRange()[0]
            xref = float((x_range[0] + x_range[1]) * 0.5)

        idx_offsets = offsets[idx]
        nearest_i = int(np.argmin(np.abs(idx_offsets - float(xref))))
        trace_idx = int(idx[nearest_i])
        trace_headers = self.loaded.get("trace_headers", [])

        if trace_idx in self._removed_traces:
            self._removed_traces.remove(trace_idx)
            if trace_idx < len(trace_headers):
                th = trace_headers[trace_idx]
                th.iflagi = abs(int(getattr(th, "iflagi", 1) or 1))
            self.lbl_status.setText(f"Trace {trace_idx} 已恢复显示")
        else:
            self._removed_traces.add(trace_idx)
            if trace_idx < len(trace_headers):
                th = trace_headers[trace_idx]
                th.iflagi = -abs(int(getattr(th, "iflagi", 1) or 1))
            self.lbl_status.setText(f"Trace {trace_idx} 已移除")

        self.request_render(delay_ms=20)


    def _handle_delete_range_key(self) -> None:
        if self.loaded is None or self.pick_manager is None:
            return
        if self.mouse_x is None:
            self.lbl_status.setText("请先将鼠标移动到绘图区，再按 d")
            return

        if self.last_key_class != "d" or self.delete_range_state == 2:
            self.delete_range_state = 0

        if self.delete_range_state == 0:
            self.delete_range_x1 = float(self.mouse_x)
            self.delete_range_state = 1
            self.lbl_status.setText(f"批量删除：已记录起点 x={self.delete_range_x1:.2f}，再按 d 记录终点")
        elif self.delete_range_state == 1:
            x2 = float(self.mouse_x)
            x1 = float(self.delete_range_x1 if self.delete_range_x1 is not None else x2)
            lo, hi = min(x1, x2), max(x1, x2)
            pick_word = int(self.spin_apick.value())
            offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
            idx = self._extract_indices()
            to_delete: List[int] = []
            for trace_idx in idx:
                gidx = int(trace_idx)
                if gidx < 0 or gidx >= offsets.size:
                    continue
                if lo <= float(offsets[gidx]) <= hi:
                    if self.pick_manager.get_pick(gidx, pick_word) is not None:
                        to_delete.append(gidx)

            deleted = len(to_delete)
            if deleted > 0:
                self._push_pick_undo("范围删除拾取")
                for gidx in to_delete:
                    self.pick_manager.remove_pick(gidx, pick_word)

            self.delete_range_state = 2
            self.delete_range_x1 = None
            self.lbl_status.setText(f"批量删除完成：删除 {deleted} 个拾取（范围 {lo:.2f}~{hi:.2f}）")
            self.request_render(delay_ms=20)

        self.last_key_class = "d"


    def keyPressEvent(self, event):
        super().keyPressEvent(event)

