# -*- coding: utf-8 -*-
"""Param panel drag/resize/height layout mixed into QtFastViewer."""

from __future__ import annotations

from typing import Optional

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as exc:  # pragma: no cover
    raise RuntimeError("未安装 PySide6") from exc


class ParamPanelMixin:
    """参数条高度统一、拖拽重排、边缘调宽、布局持久化。"""

    def _apply_param_panel_text_contrast(self, text_color: str, native_default: bool = False) -> None:
        """强制参数面板文字对比度，避免禁用态被系统调灰。"""
        panel_root = getattr(self, "params_panel_scroll", None)
        if panel_root is None:
            return
        try:
            host = panel_root.widget()
        except Exception:
            host = None
        if host is None:
            return

        labels = host.findChildren(QtWidgets.QLabel)
        checks = host.findChildren(QtWidgets.QCheckBox)
        radios = host.findChildren(QtWidgets.QRadioButton)
        for widget in labels + checks + radios:
            try:
                if native_default:
                    widget.setStyleSheet("")
                else:
                    widget.setStyleSheet(f"color: {text_color};")
            except Exception:
                pass


    def _reapply_param_group_stretches(self) -> None:
        """按面板权重分配横向 stretch，避免尾部空白。"""
        layout = getattr(self, "_params_layout", None)
        if layout is None:
            return
        weights = getattr(self, "_param_group_stretch", {}) or {}
        for group in list(getattr(self, "_param_groups", [])):
            try:
                idx = layout.indexOf(group)
                if idx < 0:
                    continue
                key = self._param_group_keys.get(group, "")
                layout.setStretch(idx, int(weights.get(key, 2)))
            except Exception:
                continue


    def _param_group_natural_height(self, group: QtWidgets.QGroupBox) -> int:
        """估算 GroupBox 边框自然高度（含标题与内边距）。"""
        try:
            group.ensurePolished()
        except Exception:
            pass
        h = 0
        try:
            lay = group.layout()
            if lay is not None:
                try:
                    lay.activate()
                except Exception:
                    pass
                h = max(int(lay.sizeHint().height()), int(lay.minimumSize().height()))
                try:
                    m = lay.contentsMargins()
                    h += int(m.top()) + int(m.bottom())
                except Exception:
                    pass
        except Exception:
            pass
        # 标题带
        try:
            fm = group.fontMetrics()
            title_h = int(fm.height()) + 10
        except Exception:
            title_h = 22
        try:
            style = group.style()
            frame = int(style.pixelMetric(QtWidgets.QStyle.PixelMetric.PM_DefaultFrameWidth, None, group))
        except Exception:
            frame = 1
        h += int(title_h) + 2 * max(1, int(frame)) + 4
        try:
            h = max(h, int(group.sizeHint().height()), int(group.minimumSizeHint().height()))
        except Exception:
            pass
        try:
            w = max(int(group.width()), int(group.minimumWidth()), int(group.sizeHint().width()), 1)
            if group.hasHeightForWidth():
                hfw = int(group.heightForWidth(w))
                if hfw > 0:
                    h = max(h, hfw)
        except Exception:
            pass
        return max(0, int(h))


    def _unify_param_group_heights(self) -> int:
        """以可见面板中最高者为准，强制统一各 GroupBox 边框高度。"""
        groups = [
            g
            for g in list(getattr(self, "_param_groups", []))
            if g is not None and (not hasattr(g, "isVisible") or g.isVisible())
        ]
        if not groups:
            return 0

        # 解除旧高度锁，按内容重新测量
        for group in groups:
            try:
                group.setMinimumHeight(0)
                group.setMaximumHeight(16777215)
                pol = group.sizePolicy()
                pol.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Expanding)
                pol.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Preferred)
                group.setSizePolicy(pol)
                group.updateGeometry()
            except Exception:
                continue

        layout = getattr(self, "_params_layout", None)
        host = getattr(self, "_params_container", None)
        try:
            if layout is not None:
                layout.activate()
        except Exception:
            pass
        try:
            if host is not None:
                host.adjustSize()
                host.updateGeometry()
        except Exception:
            pass
        try:
            QtWidgets.QApplication.sendPostedEvents(None, 0)
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
        except Exception:
            pass

        max_h = 0
        for group in groups:
            try:
                max_h = max(max_h, self._param_group_natural_height(group))
            except Exception:
                continue
        if max_h <= 0:
            return 0

        # min=max 比单靠 setFixedHeight 更稳，避免后续布局把高度改回去
        for group in groups:
            try:
                pol = group.sizePolicy()
                pol.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Expanding)
                pol.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Fixed)
                group.setSizePolicy(pol)
                group.setMinimumHeight(int(max_h))
                group.setMaximumHeight(int(max_h))
                group.resize(max(group.width(), group.minimumWidth()), int(max_h))
                group.updateGeometry()
            except Exception:
                try:
                    group.setFixedHeight(int(max_h))
                except Exception:
                    continue

        try:
            if host is not None:
                host.setMinimumHeight(int(max_h))
                host.updateGeometry()
        except Exception:
            pass

        self._param_groups_unified_height = int(max_h)
        return int(max_h)


    def _schedule_unify_param_group_heights(self, delay_ms: int = 0) -> None:
        """显示后延迟再统一一次；合并多次预约，减轻闪烁。"""
        try:
            pending = int(getattr(self, "_unify_schedule_token", 0) or 0) + 1
            self._unify_schedule_token = pending

            def _run(token: int = pending) -> None:
                if int(getattr(self, "_unify_schedule_token", 0) or 0) != int(token):
                    return
                try:
                    self.setUpdatesEnabled(False)
                    self._fit_params_strip_to_content()
                finally:
                    try:
                        self.setUpdatesEnabled(True)
                    except Exception:
                        pass

            QtCore.QTimer.singleShot(max(0, int(delay_ms)), _run)
        except Exception:
            try:
                self._fit_params_strip_to_content()
            except Exception:
                pass


    def _enforce_params_panel_fixed_height(self) -> None:
        """兼容旧名：改为可缩放约束，不再锁死高度。"""
        self._sync_params_panel_height_constraints()


    def _sync_params_panel_height_constraints(self) -> None:
        """参数区仅设最小高度，最大由垂直分割条决定。"""
        panel_root = getattr(self, "params_panel_scroll", None)
        if panel_root is None:
            return
        min_h = int(getattr(self, "_params_panel_min_height", 52) or 52)
        if min_h <= 0:
            min_h = 52
        try:
            panel_root.setMinimumHeight(min_h)
            panel_root.setMaximumHeight(16777215)
        except Exception:
            pass
        self._sync_params_panel_content_height()
        try:
            host = panel_root.widget()
        except Exception:
            host = None
        if host is not None:
            try:
                host.adjustSize()
                host.updateGeometry()
            except Exception:
                pass
        try:
            panel_root.updateGeometry()
        except Exception:
            pass
        # 若分割条存在且参数区高度被压没，恢复默认
        split = getattr(self, "_body_splitter", None)
        if split is not None and panel_root.isVisible():
            try:
                sizes = list(split.sizes())
                if len(sizes) >= 2 and int(sizes[0]) > 0 and int(sizes[0]) < min_h:
                    default_h = int(getattr(self, "_params_panel_default_height", 118) or 118)
                    delta = default_h - int(sizes[0])
                    sizes[0] = default_h
                    sizes[1] = max(1, int(sizes[1]) - delta)
                    split.setSizes(sizes)
            except Exception:
                pass


    def _sync_params_panel_content_height(self) -> None:
        """根据当前面板字号/样式，刷新参数容器最小高度。"""
        panel_root = getattr(self, "params_panel_scroll", None)
        host = getattr(self, "_params_container", None)
        if panel_root is None or host is None:
            return
        groups = getattr(self, "_param_groups", [])
        layout = getattr(self, "_params_layout", None)
        needed_h = 0
        for group in groups:
            try:
                if group is None or (hasattr(group, "isVisible") and not group.isVisible()):
                    continue
                hint_h = int(group.sizeHint().height())
                min_hint_h = int(group.minimumSizeHint().height())
                min_h = int(group.minimumHeight())
                needed_h = max(needed_h, hint_h, min_hint_h, min_h)
            except Exception:
                continue
        if layout is not None:
            try:
                m = layout.contentsMargins()
                needed_h += int(m.top()) + int(m.bottom())
            except Exception:
                pass
        if needed_h <= 0:
            needed_h = int(getattr(self, "_params_panel_fixed_height", 140) or 140)
        try:
            host.setMinimumHeight(needed_h)
            host.updateGeometry()
        except Exception:
            pass


    def _apply_params_panel_font_boost(self, step: int = 1) -> None:
        """参数面板字体整体放大一档，避免控件过小难读。"""
        panel_root = getattr(self, "params_panel_scroll", None)
        if panel_root is None:
            return
        try:
            host = panel_root.widget()
        except Exception:
            host = None
        if host is None:
            return
        widget_types = [
            QtWidgets.QGroupBox,
            QtWidgets.QLabel,
            QtWidgets.QPushButton,
            QtWidgets.QToolButton,
            QtWidgets.QCheckBox,
            QtWidgets.QRadioButton,
            QtWidgets.QSpinBox,
            QtWidgets.QDoubleSpinBox,
            QtWidgets.QComboBox,
            QtWidgets.QListWidget,
        ]
        targets = [host]
        seen_ids = {id(host)}
        for wtype in widget_types:
            try:
                children = host.findChildren(wtype)
            except Exception:
                children = []
            for child in children:
                cid = id(child)
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                targets.append(child)
        for w in targets:
            try:
                font = w.font()
                size = int(font.pointSize())
                if size <= 0:
                    size = 9
                font.setPointSize(size + int(max(0, step)))
                w.setFont(font)
            except Exception:
                pass


    def _apply_panel_frame_contrast(self, native_default: bool = False) -> None:
        """增强主参数面板分区边框；默认主题也应用。"""
        panel_groups = [
            getattr(self, "group_base", None),
            getattr(self, "group_gain", None),
            getattr(self, "group_denoise", None),
            getattr(self, "group_pick", None),
            getattr(self, "group_align", None),
            getattr(self, "group_waveop", None),
            getattr(self, "group_advcorr", None),
            getattr(self, "group_ttpl", None),
        ]
        if native_default:
            panel_border = "#5f6b7a"
            title_bg = "#e9edf4"
            title_color = "#111111"
            panel_bg = "#ffffff"
            for group in panel_groups:
                if group is None:
                    continue
                try:
                    group.setStyleSheet(
                        "QGroupBox { border: 2px solid %s; border-radius: 6px; margin-top: 8px; padding-top: 6px; background-color: %s; color: %s; }"
                        "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 1px 6px 1px 6px; background-color: %s; border-radius: 3px; color: %s; font-weight: 600; font-size: 16px; }"
                        % (panel_border, panel_bg, title_color, title_bg, title_color)
                    )
                except Exception:
                    pass
            return

        # 非默认主题由全局样式表统一控制，清除局部覆盖
        for group in panel_groups:
            if group is None:
                continue
            try:
                group.setStyleSheet("")
            except Exception:
                pass


    def _panel_title_hit_height(self, group: QtWidgets.QGroupBox) -> int:
        """按字体/样式估算标题带高度，至少保留默认命中区。"""
        base = int(getattr(self, "_panel_drag_title_height_px", 28) or 28)
        try:
            fm = group.fontMetrics()
            text_h = int(fm.height()) + 10
        except Exception:
            text_h = base
        try:
            style = group.style()
            metric = int(style.pixelMetric(QtWidgets.QStyle.PixelMetric.PM_TitleBarHeight, None, group))
        except Exception:
            metric = 0
        return max(base, text_h, metric if metric > 0 else 0)


    def _is_panel_title_hit(self, group: QtWidgets.QGroupBox, pos: QtCore.QPoint) -> bool:
        """仅在 QGroupBox 标题带内启用拖拽，避免干扰内部控件。"""
        try:
            if not group.rect().contains(pos):
                return False
        except Exception:
            return False
        return int(pos.y()) <= int(self._panel_title_hit_height(group))


    def _panel_resize_hit_edge(self, group: QtWidgets.QGroupBox, pos: QtCore.QPoint) -> str:
        """命中左右边框返回 'left'/'right'，否则返回空字符串。"""
        try:
            if not group.rect().contains(pos):
                return ""
            x = int(pos.x())
            w = int(group.width())
        except Exception:
            return ""
        margin = int(self._panel_resize_margin_px)
        if x <= margin:
            return "left"
        if x >= max(0, w - margin):
            return "right"
        return ""


    def _set_panel_group_width(self, group: QtWidgets.QGroupBox, width: int) -> None:
        """按 bounds 约束改宽（布局内用 setFixedWidth 才能生效）。"""
        min_w, max_w = self._panel_resize_bounds.get(id(group), (120, 900))
        new_w = max(int(min_w), min(int(max_w), int(width)))
        try:
            # 先放宽 max，再固定到目标宽；下次调宽仍可读 bounds 再 setFixedWidth
            group.setMaximumWidth(int(max_w))
            group.setMinimumWidth(int(min_w))
            group.setFixedWidth(new_w)
            group.updateGeometry()
        except Exception:
            pass


    def _grab_panel_mouse(self, group: QtWidgets.QGroupBox) -> None:
        """占位：不调用 grabMouse（非 popup 会触发平台警告）。

        拖拽/调宽改由 QApplication 级 eventFilter 全局跟踪。
        """
        self._panel_mouse_grabber = group


    def _release_panel_mouse(self) -> None:
        self._panel_mouse_grabber = None


    def _apply_panel_resize_from_global_x(self, gx: int) -> None:
        group = getattr(self, "_panel_resize_group", None)
        if group is None or not self._panel_resize_active:
            return
        dx = int(gx) - int(self._panel_resize_start_x)
        if self._panel_resize_edge == "left":
            dx = -dx
        new_w = int(self._panel_resize_start_width) + int(dx)
        self._set_panel_group_width(group, new_w)


    def _promote_panel_drag_if_needed(self, global_pos: QtCore.QPoint) -> None:
        """candidate 超过阈值后启动拖拽（不依赖鼠标仍在原面板上）。"""
        candidate = getattr(self, "_panel_drag_candidate", None)
        if candidate is None or self._panel_drag_active or self._panel_resize_active:
            return
        if (global_pos - self._panel_drag_start_global).manhattanLength() < int(self._panel_drag_threshold_px):
            return
        self._panel_drag_active = True
        self._create_panel_drag_ghost(candidate)


    def _finish_panel_resize(self, global_pos: Optional[QtCore.QPoint] = None) -> None:
        if not self._panel_resize_active:
            return
        group = self._panel_resize_group
        self._panel_resize_active = False
        self._panel_resize_group = None
        self._panel_resize_edge = ""
        self._release_panel_mouse()
        self._destroy_panel_drag_ghost()
        self._save_panel_layout_state()
        if group is not None:
            try:
                group.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            except Exception:
                pass
        self._refresh_panel_hover_cursors(global_pos or QtGui.QCursor.pos())


    def _update_panel_cursor(self, group: QtWidgets.QGroupBox, pos: QtCore.QPoint) -> None:
        """根据当前位置更新面板光标：边框缩放 > 标题拖拽 > 普通。"""
        if self._panel_resize_active and self._panel_resize_group is group:
            cursor = QtCore.Qt.CursorShape.SizeHorCursor
        elif self._panel_drag_candidate is group:
            cursor = QtCore.Qt.CursorShape.ClosedHandCursor
        else:
            edge = self._panel_resize_hit_edge(group, pos)
            if edge:
                cursor = QtCore.Qt.CursorShape.SizeHorCursor
            elif self._is_panel_title_hit(group, pos):
                cursor = QtCore.Qt.CursorShape.OpenHandCursor
            else:
                cursor = QtCore.Qt.CursorShape.ArrowCursor
        try:
            group.setCursor(cursor)
        except Exception:
            pass


    def _refresh_panel_hover_cursors(self, global_pos: Optional[QtCore.QPoint] = None) -> None:
        """实时刷新所有面板光标，确保离开标题/边界后立即恢复箭头。"""
        if global_pos is None:
            global_pos = QtGui.QCursor.pos()
        hover_group, hover_local_pos = self._group_hover_from_global_pos(global_pos)
        for group in getattr(self, "_param_groups", []):
            if group is hover_group:
                self._update_panel_cursor(group, hover_local_pos)
            else:
                try:
                    group.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
                except Exception:
                    continue


    def _panel_drop_target_group(self, x_global: int) -> Optional[QtWidgets.QGroupBox]:
        """按全局 X 坐标找到最近的参数面板目标组。"""
        groups = [g for g in getattr(self, "_param_groups", []) if isinstance(g, QtWidgets.QGroupBox) and g.isVisible()]
        if not groups:
            return None
        best_group = None
        best_dist = None
        for group in groups:
            try:
                center_local = group.rect().center()
                center_global = group.mapToGlobal(center_local)
                dist = abs(int(center_global.x()) - int(x_global))
            except Exception:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_group = group
        return best_group


    def _destroy_panel_drag_ghost(self) -> None:
        ghost = getattr(self, "_panel_drag_ghost", None)
        if ghost is None:
            return
        try:
            ghost.hide()
            ghost.deleteLater()
        except Exception:
            pass
        self._panel_drag_ghost = None


    def _create_panel_drag_ghost(self, group: QtWidgets.QGroupBox) -> None:
        """创建半透明悬浮面板，用于拖拽视觉反馈。"""
        self._destroy_panel_drag_ghost()
        try:
            pix = group.grab()
        except Exception:
            return
        if pix.isNull():
            return
        # 采用主窗口内浮层，避免顶层窗口在不同平台上的坐标抖动/跳位
        ghost = QtWidgets.QWidget(self)
        try:
            ghost.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.SubWindow)
            ghost.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
            ghost.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            ghost.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
            ghost.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground, False)
            ghost.resize(pix.size())
            label = QtWidgets.QLabel(ghost)
            label.setPixmap(pix)
            label.setGeometry(0, 0, pix.width(), pix.height())
            label.setStyleSheet(
                "QLabel {"
                "border: 1px solid rgba(38, 132, 255, 180);"
                "background: rgba(255, 255, 255, 45);"
                "}"
            )
            ghost.setWindowOpacity(0.72)
            # hotspot 在按下标题栏时确定，这里不再重算，避免启动拖拽瞬间跳位
            self._panel_drag_ghost = ghost
            self._update_panel_drag_ghost_position(QtGui.QCursor.pos())
            ghost.show()
            ghost.raise_()
        except Exception:
            try:
                ghost.deleteLater()
            except Exception:
                pass
            self._panel_drag_ghost = None


    def _update_panel_drag_ghost_position(self, global_pos: QtCore.QPoint) -> None:
        ghost = getattr(self, "_panel_drag_ghost", None)
        if ghost is None:
            return
        try:
            target_global = global_pos - self._panel_drag_hotspot
            target_local = self.mapFromGlobal(target_global)
            ghost.move(target_local)
            ghost.raise_()
        except Exception:
            pass


    def _panel_layout_settings(self) -> QtCore.QSettings:
        """返回用于保存面板布局的设置对象。"""
        return QtCore.QSettings("pyAOBS", "QtFastViewer")


    def _save_panel_layout_state(self) -> None:
        """保存面板顺序与宽度。"""
        try:
            settings = self._panel_layout_settings()
            group_name = str(getattr(self, "_panel_layout_settings_group", "panel_layout_v1"))
            settings.beginGroup(group_name)
            order = []
            widths: Dict[str, int] = {}
            for group in list(getattr(self, "_param_groups", [])):
                key = self._param_group_keys.get(group, "")
                if not key:
                    continue
                order.append(key)
                try:
                    widths[key] = int(group.width())
                except Exception:
                    continue
            settings.setValue("order", order)
            settings.setValue("widths", json.dumps(widths, ensure_ascii=False))
            settings.endGroup()
            settings.sync()
        except Exception:
            pass


    def _restore_panel_layout_state(self) -> None:
        """恢复面板顺序与宽度。"""
        try:
            settings = self._panel_layout_settings()
            group_name = str(getattr(self, "_panel_layout_settings_group", "panel_layout_v1"))
            settings.beginGroup(group_name)
            saved_order = settings.value("order", [])
            saved_widths_raw = settings.value("widths", "")
            settings.endGroup()
        except Exception:
            return

        groups = list(getattr(self, "_param_groups", []))
        if not groups:
            return
        key_to_group = {v: k for k, v in getattr(self, "_param_group_keys", {}).items() if v}

        if isinstance(saved_order, str):
            order_items = [x for x in saved_order.split(",") if x]
        elif isinstance(saved_order, (list, tuple)):
            order_items = [str(x) for x in saved_order if str(x)]
        else:
            order_items = []
        if order_items:
            new_groups: List[QtWidgets.QGroupBox] = []
            seen = set()
            for key in order_items:
                group = key_to_group.get(key)
                if group is None or group in seen:
                    continue
                new_groups.append(group)
                seen.add(group)
            for group in groups:
                if group not in seen:
                    new_groups.append(group)
            self._param_groups = new_groups
            layout = getattr(self, "_params_layout", None)
            if layout is not None:
                for group in self._param_groups:
                    try:
                        layout.removeWidget(group)
                    except Exception:
                        pass
                for idx, group in enumerate(self._param_groups):
                    try:
                        layout.insertWidget(idx, group, stretch=0)
                    except Exception:
                        continue
            self._reapply_param_group_stretches()

        widths: Dict[str, int] = {}
        try:
            if isinstance(saved_widths_raw, str) and saved_widths_raw.strip():
                parsed = json.loads(saved_widths_raw)
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        try:
                            widths[str(k)] = int(v)
                        except Exception:
                            continue
        except Exception:
            widths = {}

        for key, width in widths.items():
            group = key_to_group.get(key)
            if group is None:
                continue
            self._set_panel_group_width(group, int(width))


    def _move_param_group(self, source: QtWidgets.QGroupBox, target: QtWidgets.QGroupBox) -> None:
        """将 source 面板移动到 target 位置。"""
        groups = getattr(self, "_param_groups", [])
        layout = getattr(self, "_params_layout", None)
        if layout is None or source not in groups or target not in groups or source is target:
            return
        src_idx = groups.index(source)
        dst_idx = groups.index(target)
        groups.pop(src_idx)
        groups.insert(dst_idx, source)

        try:
            layout.removeWidget(source)
            layout.insertWidget(dst_idx, source, stretch=0)
        except Exception:
            return
        self._reapply_param_group_stretches()

        try:
            self._set_status_text(
                f"面板已重排：{source.title()} -> 位置 {dst_idx + 1}",
                hold_ms=1400,
            )
        except Exception:
            try:
                self.lbl_status.setText(f"面板已重排：{source.title()}")
            except Exception:
                pass
        self._save_panel_layout_state()
        self._unify_param_group_heights()


    def _panel_drag_global_pos_from_event(self, event) -> QtCore.QPoint:
        """尽量从事件中提取全局坐标，失败时回退到当前光标。"""
        # 使用光标实时全局坐标，规避不同事件类型/平台下坐标系差异导致的抖动与跳位
        return QtGui.QCursor.pos()


    def _finish_panel_drag(self, global_pos: QtCore.QPoint) -> None:
        """结束面板拖拽并按当前位置完成重排。"""
        drag_source = self._panel_drag_candidate
        was_dragging = bool(self._panel_drag_active)
        self._panel_drag_candidate = None
        self._panel_drag_active = False
        self._release_panel_mouse()
        self._destroy_panel_drag_ghost()
        if drag_source is None:
            return
        try:
            drag_source.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        except Exception:
            pass
        if not was_dragging:
            return
        gx = int(global_pos.x())
        drag_target = self._panel_drop_target_group(gx)
        if drag_target is not None:
            self._move_param_group(drag_source, drag_target)

    def _toggle_params_panel(self) -> None:
        was_visible = self.params_panel_scroll.isVisible()
        self.params_panel_scroll.setVisible(not was_visible)
        self.btn_toggle_panels.setText("显示面板" if was_visible else "隐藏面板")
        split = getattr(self, "_body_splitter", None)
        if split is None:
            return
        try:
            sizes = list(split.sizes())
            if len(sizes) < 2:
                return
            if was_visible:
                freed = int(sizes[0])
                sizes[0] = 0
                sizes[1] = max(1, int(sizes[1]) + freed)
                split.setSizes(sizes)
            else:
                default_h = int(getattr(self, "_params_panel_default_height", 118) or 118)
                if int(sizes[0]) < int(getattr(self, "_params_panel_min_height", 52) or 52):
                    sizes[1] = max(1, int(sizes[1]) - default_h)
                    sizes[0] = default_h
                    split.setSizes(sizes)
        except Exception:
            pass


    def _on_body_splitter_moved(self, *_args) -> None:
        """记录参数区当前高度，供主题切换后恢复。"""
        split = getattr(self, "_body_splitter", None)
        if split is None:
            return
        try:
            sizes = list(split.sizes())
            if sizes and int(sizes[0]) > 0:
                self._params_panel_fixed_height = int(sizes[0])
                self._params_panel_default_height = int(sizes[0])
        except Exception:
            pass
        self._sync_params_panel_content_height()


    def _estimate_action_bar_width(self) -> int:
        """估算当前工具条已显示控件的总宽度（像素）。"""
        spacing = 6
        total = 0
        for w in self._action_bar_all_widgets:
            if w.isVisible():
                total += int(w.sizeHint().width())
                total += spacing
        for sep in self._action_bar_separators:
            if sep.isVisible():
                total += int(sep.sizeHint().width()) + spacing
        return total


    def _update_action_bar_overflow(self) -> None:
        if not hasattr(self, "btn_more"):
            return
        # 用户要求“尽量全部列出，不依赖下拉”，固定隐藏“更多”
        self.btn_more.setVisible(False)


    def _fit_params_strip_to_content(self) -> None:
        """统一面板边框高度后，参数条跟齐。"""
        unified = self._unify_param_group_heights()
        self._sync_params_panel_content_height()
        host = getattr(self, "_params_container", None)
        needed = int(unified or 0)
        try:
            if host is not None:
                needed = max(needed, int(host.minimumHeight() or 0), int(host.sizeHint().height() or 0))
        except Exception:
            pass
        if needed <= 0:
            needed = int(getattr(self, "_params_panel_default_height", 118) or 118)
        needed = int(needed) + 10
        floor = int(getattr(self, "_params_panel_min_height", 52) or 52)
        cap = 320
        target = max(floor, min(cap, needed))
        self._params_panel_default_height = target
        self._params_panel_fixed_height = target
        split = getattr(self, "_body_splitter", None)
        if split is None:
            return
        try:
            sizes = list(split.sizes())
            if len(sizes) >= 2:
                # 参数条至少容纳统一后的面板高度
                target = max(target, int(unified or 0) + 10)
                delta = target - int(sizes[0])
                sizes[0] = target
                sizes[1] = max(80, int(sizes[1]) - delta)
                split.setSizes(sizes)
        except Exception:
            pass

