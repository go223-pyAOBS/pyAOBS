# -*- coding: utf-8 -*-
"""Main UI construction mixed into QtFastViewer."""

from __future__ import annotations

from typing import Optional

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as exc:  # pragma: no cover
    raise RuntimeError("未安装 PySide6") from exc

try:
    import pyqtgraph as pg
except Exception:
    pg = None  # type: ignore


class BuildUiMixin:
    """构建主界面布局、控件与绘图区。"""

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # 第一层：精简操作条（避免单行过长）
        action_bar = QtWidgets.QHBoxLayout()
        self._action_bar_layout = action_bar
        action_bar.setSpacing(3)
        root.addLayout(action_bar)

        self.btn_open = QtWidgets.QPushButton("打开 .z")
        self.btn_open_hdr = QtWidgets.QPushButton("打开 .hdr (可选)")
        self.btn_open_rec = QtWidgets.QPushButton("打开 .r (可选)")
        self.btn_open_rec.setToolTip(
            "记录文件 .rec/.rsp（可选）：炮号→模型坐标(xmod,ymod)与方位。"
            "浏览/拾取可不选；写 tx.in 时可预填 xmod，理论走时/r.in 也可用。"
        )
        self.btn_reload = QtWidgets.QPushButton("重绘")
        # 参数随工区自动存取；工具栏不再提供独立「保存/加载参数」
        self.btn_save_params = QtWidgets.QPushButton("保存参数")
        self.btn_load_params = QtWidgets.QPushButton("加载参数")
        self.btn_save_params.hide()
        self.btn_load_params.hide()
        self.btn_save_z = QtWidgets.QPushButton("保存.z")
        self.btn_data_info = QtWidgets.QPushButton("数据信息")
        self.btn_location_map = QtWidgets.QPushButton("位置Map")
        self.btn_export_fig = QtWidgets.QPushButton("导出图像")
        self.btn_prev_rec = QtWidgets.QPushButton("上一炮")
        self.btn_next_rec = QtWidgets.QPushButton("下一炮")
        self.btn_theory = QtWidgets.QPushButton("理论走时")
        self.btn_clear_theory = QtWidgets.QPushButton("清除理论")
        self.btn_water_corr = QtWidgets.QPushButton("水层校正")
        self.btn_clear_water = QtWidgets.QPushButton("清除水层")
        self.btn_water_curve = QtWidgets.QPushButton("校正曲线")
        self.btn_load_txin = QtWidgets.QPushButton("读取tx.in")
        self.btn_clear_txin = QtWidgets.QPushButton("清除tx叠加")
        self.btn_preview_map_txin = QtWidgets.QPushButton("映射预览")
        self.btn_map_txin = QtWidgets.QPushButton("tx映射拾取")
        self.chk_map_txin_apick_only = QtWidgets.QCheckBox("仅当前apick")
        self.chk_map_txin_apick_only.setChecked(False)
        self.chk_map_txin_view_only = QtWidgets.QCheckBox("仅映射当前视窗")
        self.chk_map_txin_view_only.setChecked(False)
        self.spin_map_txin_tol = QtWidgets.QDoubleSpinBox()
        self.spin_map_txin_tol.setRange(10.0, 500.0)
        self.spin_map_txin_tol.setDecimals(1)
        self.spin_map_txin_tol.setSingleStep(5.0)
        self.spin_map_txin_tol.setValue(75.0)
        self.btn_theme = QtWidgets.QToolButton()
        self.btn_theme.setText("主题")
        self.btn_theme.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_save_picks = QtWidgets.QPushButton("保存")
        self.btn_undo_pick = QtWidgets.QPushButton("撤销(Ctrl+Z)")
        self.btn_redo_pick = QtWidgets.QPushButton("重做(Ctrl+Y)")
        self.btn_save_hdr = QtWidgets.QPushButton("写入HDR")
        self.btn_write_txin = QtWidgets.QPushButton("写入tx.in")
        self.btn_clear_picks = QtWidgets.QPushButton("清空")
        self.btn_auto_pick = QtWidgets.QPushButton("自动")
        self.btn_interp_pick = QtWidgets.QPushButton("插值")
        self.btn_align_pick = QtWidgets.QPushButton("波形对齐(A)")
        self.btn_align_adaptive = QtWidgets.QPushButton("拾取更新(F)")
        self.btn_eval_stack = QtWidgets.QPushButton("叠加评价")
        self.btn_waveop_stack = QtWidgets.QPushButton("波形叠加(V段)")
        self.btn_waveop_att = QtWidgets.QPushButton("姿态校正")
        self.btn_waveop_clear = QtWidgets.QPushButton("清除V段")
        self.btn_waveop_save = QtWidgets.QPushButton("保存波形段")
        self.btn_waveop_load = QtWidgets.QPushButton("加载波形段")
        self.list_waveop_segments = QtWidgets.QListWidget()
        self.list_waveop_segments.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.list_waveop_segments.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.list_waveop_segments.setToolTip("V段列表：Shift+V 删除最近一个；已迁出参数条")
        self._waveop_list_external_host: Optional[QtWidgets.QWidget] = None
        self._waveop_list_local_dock: Optional[QtWidgets.QWidget] = None
        self._waveop_list_changed_cbs: List[Callable[[int], None]] = []
        self.btn_static_corr = QtWidgets.QPushButton("静校正")
        self.btn_clear_static = QtWidgets.QPushButton("清除静校正")
        self.btn_clear_align = QtWidgets.QPushButton("清除对齐")
        self.btn_gain_preset_balanced = QtWidgets.QPushButton("平衡")
        self.btn_far_offset_boost = QtWidgets.QPushButton("远偏增强")
        self.btn_gain_preset_strong = QtWidgets.QPushButton("强增强")
        self.btn_mute_status = QtWidgets.QPushButton("Mute: OFF")
        self.btn_clear_mute = QtWidgets.QPushButton("清空")
        self.btn_clear_mute.setEnabled(False)
        self.btn_clear_mute.setToolTip("清空所有 mute（包含顶点与编辑状态）")
        self.chk_mute_invert = QtWidgets.QCheckBox("反选")
        self.chk_mute_invert.setEnabled(False)
        self.chk_mute_invert.setToolTip("Mute 内外反选（等同 Shift+M）：勾选后保留多边形外部、压制内部")
        self.btn_toggle_panels = QtWidgets.QPushButton("隐藏面板")
        self.btn_reload.setEnabled(False)
        self.btn_save_params.setEnabled(False)
        self.btn_load_params.setEnabled(True)
        self.btn_save_z.setEnabled(False)
        self.btn_data_info.setEnabled(False)
        self.btn_location_map.setEnabled(False)
        self.btn_export_fig.setEnabled(False)
        self.btn_prev_rec.setEnabled(False)
        self.btn_next_rec.setEnabled(False)
        self.btn_theory.setEnabled(False)
        self.btn_clear_theory.setEnabled(False)
        self.btn_water_corr.setEnabled(False)
        self.btn_clear_water.setEnabled(False)
        self.btn_water_curve.setEnabled(False)
        self.btn_load_txin.setEnabled(True)
        self.btn_clear_txin.setEnabled(True)
        self.btn_preview_map_txin.setEnabled(False)
        self.btn_map_txin.setEnabled(False)
        self.chk_map_txin_apick_only.setEnabled(False)
        self.chk_map_txin_view_only.setEnabled(False)
        self.spin_map_txin_tol.setEnabled(False)
        self.btn_save_picks.setEnabled(False)
        self.btn_undo_pick.setEnabled(False)
        self.btn_redo_pick.setEnabled(False)
        self.btn_save_hdr.setEnabled(False)
        self.btn_write_txin.setEnabled(False)
        self.btn_clear_picks.setEnabled(False)
        self.btn_auto_pick.setEnabled(False)
        self.btn_interp_pick.setEnabled(False)
        self.btn_align_pick.setEnabled(False)
        self.btn_align_adaptive.setEnabled(False)
        self.btn_eval_stack.setEnabled(False)
        self.btn_waveop_stack.setEnabled(False)
        self.btn_waveop_att.setEnabled(False)
        self.btn_waveop_clear.setEnabled(False)
        self.btn_waveop_save.setEnabled(False)
        self.btn_waveop_load.setEnabled(False)
        self.btn_static_corr.setEnabled(False)
        self.btn_clear_static.setEnabled(False)
        self.btn_clear_align.setEnabled(False)
        self._action_bar_separators: List[QtWidgets.QFrame] = []

        def _add_toolbar_separator():
            sep = QtWidgets.QFrame()
            sep.setFrameShape(QtWidgets.QFrame.Shape.VLine)
            sep.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
            self._action_bar_separators.append(sep)
            action_bar.addWidget(sep)

        # 文件与工程
        action_bar.addWidget(self.btn_open)
        action_bar.addWidget(self.btn_open_hdr)
        action_bar.addWidget(self.btn_open_rec)
        action_bar.addWidget(self.btn_reload)
        action_bar.addWidget(self.btn_save_z)
        action_bar.addWidget(self.btn_undo_pick)
        action_bar.addWidget(self.btn_redo_pick)
        action_bar.addWidget(self.btn_save_hdr)
        action_bar.addWidget(self.btn_write_txin)
        action_bar.addWidget(self.btn_data_info)
        action_bar.addWidget(self.btn_location_map)
        action_bar.addWidget(self.btn_export_fig)

        _add_toolbar_separator()

        # 记录导航
        action_bar.addWidget(self.btn_prev_rec)
        action_bar.addWidget(self.btn_next_rec)

        _add_toolbar_separator()

        # 主题（帮助改挂菜单栏「帮助」单条目 / 工区工具栏单按钮）
        action_bar.addWidget(self.btn_theme)
        self.btn_more = QtWidgets.QToolButton()
        self.btn_more.setText("更多")
        self.btn_more.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_more.setVisible(False)
        action_bar.addStretch(1)
        action_bar.addWidget(self.btn_toggle_panels)
        top_buttons = [
            self.btn_open, self.btn_open_hdr, self.btn_open_rec, self.btn_reload,
            self.btn_save_z, self.btn_undo_pick, self.btn_redo_pick, self.btn_save_hdr,
            self.btn_write_txin,
            self.btn_data_info, self.btn_location_map, self.btn_export_fig, self.btn_prev_rec, self.btn_next_rec,
            self.btn_theme, self.btn_toggle_panels
        ]
        for b in top_buttons:
            b.setMinimumHeight(22)
            try:
                f = b.font()
                ps = int(f.pointSize())
                if ps <= 0:
                    ps = 9
                f.setPointSize(ps + 2)
                b.setFont(f)
            except Exception:
                pass

        # 自适应折叠：窗口变窄时将次要按钮放入“更多”
        self._action_bar_overflow_candidates: List[QtWidgets.QWidget] = []
        self._action_bar_all_widgets: List[QtWidgets.QWidget] = [
            self.btn_open,
            self.btn_open_hdr,
            self.btn_open_rec,
            self.btn_reload,
            self.btn_save_z,
            self.btn_undo_pick,
            self.btn_redo_pick,
            self.btn_save_hdr,
            self.btn_write_txin,
            self.btn_data_info,
            self.btn_location_map,
            self.btn_export_fig,
            self.btn_prev_rec,
            self.btn_next_rec,
            self.btn_theme,
            self.btn_toggle_panels,
        ]
        self._disable_toolbar_overflow = True

        # 第二层：横向参数区（高度可由垂直分割条调节，不再锁死）
        self.params_panel_scroll = QtWidgets.QScrollArea()
        self.params_panel_scroll.setWidgetResizable(True)
        # 多面板调宽后内容可能超出视口，横滚按需出现
        self.params_panel_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # 主题切换后控件尺寸可能变化，纵向滚动条按需出现可避免内容被裁切
        self.params_panel_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._params_panel_default_height = 118
        self._params_panel_min_height = 52
        self._params_panel_fixed_height = int(self._params_panel_default_height)  # 兼容旧调用名=默认高度
        self.params_panel_scroll.setMinimumHeight(int(self._params_panel_min_height))
        self.params_panel_scroll.setMaximumHeight(16777215)
        # 稍后与剖面一并加入 _body_splitter，此处先不 root.addWidget
        self._body_splitter: Optional[QtWidgets.QSplitter] = None

        params_container = QtWidgets.QWidget()
        params_layout = QtWidgets.QHBoxLayout(params_container)
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.setSpacing(2)
        # 顶对齐：矮面板与高面板上沿齐，避免为对齐被撑高
        params_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.params_panel_scroll.setWidget(params_container)
        self._params_container = params_container
        self._params_layout = params_layout
        self._param_groups: List[QtWidgets.QGroupBox] = []
        self._param_group_keys: Dict[QtWidgets.QGroupBox, str] = {}
        self._param_group_stretch: Dict[str, int] = {
            "base": 3,
            "gain": 4,
            "denoise": 3,
            "pick": 3,
            "align": 1,
            "waveop": 1,
            "advcorr": 2,
            "ttpl": 2,
        }
        self._panel_layout_settings_group = "panel_layout_v3_wideflat"
        self._panel_drag_candidate: Optional[QtWidgets.QGroupBox] = None
        self._panel_drag_active: bool = False
        self._panel_drag_start_global = QtCore.QPoint()
        self._panel_drag_title_height_px: int = 28
        self._panel_drag_threshold_px: int = 8
        self._panel_drag_hotspot = QtCore.QPoint(16, 10)
        self._panel_drag_ghost: Optional[QtWidgets.QWidget] = None
        self._panel_resize_active: bool = False
        self._panel_resize_group: Optional[QtWidgets.QGroupBox] = None
        self._panel_resize_edge: str = ""
        self._panel_resize_start_x: int = 0
        self._panel_resize_start_width: int = 0
        self._panel_resize_margin_px: int = 8
        self._panel_resize_bounds: Dict[int, Tuple[int, int]] = {}
        self._panel_mouse_grabber: Optional[QtWidgets.QGroupBox] = None

        # 基础面板（紧凑网格）
        self.group_base = QtWidgets.QGroupBox("基础显示")
        self._param_group_keys[self.group_base] = "base"
        self.group_base.setMinimumWidth(170)
        self.group_base.setMaximumWidth(16777215)
        base_page = QtWidgets.QWidget()
        base_grid = QtWidgets.QGridLayout(base_page)
        base_grid.setContentsMargins(1, 1, 1, 1)
        base_grid.setHorizontalSpacing(3)
        base_grid.setVerticalSpacing(0)

        self.spin_irec = QtWidgets.QSpinBox()
        self.spin_irec.setMinimum(0)
        self.spin_irec.setMaximum(1_000_000)
        self.spin_irec.setValue(0)
        self.spin_irec.setToolTip("0=全部记录")
        base_grid.addWidget(QtWidgets.QLabel("记录号"), 0, 0)
        base_grid.addWidget(self.spin_irec, 0, 1)

        self.combo_itype = QtWidgets.QComboBox()
        self.combo_itype.addItems(["0(全部)", "1(垂直)", "2(径向)", "3(横向)", "4(水听器)"])
        self.combo_itype.setCurrentIndex(1)  # 默认垂直分量
        base_grid.addWidget(QtWidgets.QLabel("类型"), 0, 2)
        base_grid.addWidget(self.combo_itype, 0, 3)

        self.spin_nskip = QtWidgets.QSpinBox()
        self.spin_nskip.setRange(0, 5000)
        self.spin_nskip.setValue(0)
        base_grid.addWidget(QtWidgets.QLabel("nskip"), 1, 0)
        base_grid.addWidget(self.spin_nskip, 1, 1)

        self.spin_ndecim = QtWidgets.QSpinBox()
        self.spin_ndecim.setRange(1, 500)
        self.spin_ndecim.setValue(1)
        base_grid.addWidget(QtWidgets.QLabel("ndecim"), 1, 2)
        base_grid.addWidget(self.spin_ndecim, 1, 3)

        self.spin_vred = QtWidgets.QDoubleSpinBox()
        self.spin_vred.setRange(0.0, 20.0)
        self.spin_vred.setDecimals(3)
        self.spin_vred.setSingleStep(0.1)
        self.spin_vred.setValue(float(self.params.vred if self.params.vred > 0 else 0.0))
        base_grid.addWidget(QtWidgets.QLabel("vred"), 2, 0)
        base_grid.addWidget(self.spin_vred, 2, 1)

        self.spin_xmin = QtWidgets.QDoubleSpinBox()
        self.spin_xmin.setRange(-1_000_000.0, 1_000_000.0)
        self.spin_xmin.setDecimals(3)
        self.spin_xmin.setSingleStep(1.0)
        self.spin_xmin.setValue(float(self.params.xmin))
        base_grid.addWidget(QtWidgets.QLabel("xmin"), 3, 0)
        base_grid.addWidget(self.spin_xmin, 3, 1)

        self.spin_xmax = QtWidgets.QDoubleSpinBox()
        self.spin_xmax.setRange(-1_000_000.0, 1_000_000.0)
        self.spin_xmax.setDecimals(3)
        self.spin_xmax.setSingleStep(1.0)
        self.spin_xmax.setValue(float(self.params.xmax))
        base_grid.addWidget(QtWidgets.QLabel("xmax"), 3, 2)
        base_grid.addWidget(self.spin_xmax, 3, 3)

        self.spin_tmin = QtWidgets.QDoubleSpinBox()
        self.spin_tmin.setRange(-1_000_000.0, 1_000_000.0)
        self.spin_tmin.setDecimals(4)
        self.spin_tmin.setSingleStep(0.05)
        self.spin_tmin.setValue(float(self.params.tmin))
        base_grid.addWidget(QtWidgets.QLabel("tmin"), 4, 0)
        base_grid.addWidget(self.spin_tmin, 4, 1)

        self.spin_tmax = QtWidgets.QDoubleSpinBox()
        self.spin_tmax.setRange(-1_000_000.0, 1_000_000.0)
        self.spin_tmax.setDecimals(4)
        self.spin_tmax.setSingleStep(0.05)
        self.spin_tmax.setValue(float(self.params.tmax))
        base_grid.addWidget(QtWidgets.QLabel("tmax"), 4, 2)
        base_grid.addWidget(self.spin_tmax, 4, 3)

        self.combo_mode = QtWidgets.QComboBox()
        # 前三项索引保持兼容旧配置；Density 对齐 obs_rtm GatherCanvas
        self.combo_mode.addItems(["Wiggle", "正填充", "负填充", "Density"])
        self.combo_mode.setToolTip("Density=变密度图（多道总览）；Wiggle/填充适合精细拾取")
        base_grid.addWidget(QtWidgets.QLabel("显示"), 2, 2)
        base_grid.addWidget(self.combo_mode, 2, 3)

        self.chk_rt_shade = QtWidgets.QCheckBox("交互时填充")
        self.chk_rt_shade.setChecked(False)
        self.chk_rt_shade.setToolTip("关闭可提升拖拽/缩放流畅度")
        base_grid.addWidget(self.chk_rt_shade, 5, 0, 1, 2)
        # 去掉固定 maxWidth，让控件随面板横向拉伸
        base_group_layout = QtWidgets.QVBoxLayout(self.group_base)
        base_group_layout.setContentsMargins(1, 1, 1, 1)
        base_group_layout.addWidget(base_page)
        params_layout.addWidget(self.group_base, stretch=int(self._param_group_stretch.get("base", 3)))
        self._param_groups.append(self.group_base)

        # 增益/滤波面板（紧凑网格）
        self.group_gain = QtWidgets.QGroupBox("增益/滤波")
        self._param_group_keys[self.group_gain] = "gain"
        gain_page = QtWidgets.QWidget()
        gain_grid = QtWidgets.QGridLayout(gain_page)
        gain_grid.setContentsMargins(1, 1, 1, 1)
        gain_grid.setHorizontalSpacing(3)
        gain_grid.setVerticalSpacing(0)

        self.spin_amp = QtWidgets.QDoubleSpinBox()
        self.spin_amp.setRange(0.01, 1000.0)
        self.spin_amp.setDecimals(3)
        self.spin_amp.setSingleStep(0.05)
        self.spin_amp.setValue(1.0)
        gain_grid.addWidget(QtWidgets.QLabel("amp"), 0, 0)
        gain_grid.addWidget(self.spin_amp, 0, 1)

        self.combo_iscale = QtWidgets.QComboBox()
        self.combo_iscale.addItems(["0自动", "1固定", "2变增益"])
        self.combo_iscale.setCurrentIndex(max(0, min(2, int(self.params.iscale))))
        gain_grid.addWidget(QtWidgets.QLabel("iscale"), 0, 2)
        gain_grid.addWidget(self.combo_iscale, 0, 3)

        self.spin_rcor = QtWidgets.QDoubleSpinBox()
        self.spin_rcor.setRange(-5.0, 5.0)
        self.spin_rcor.setDecimals(3)
        self.spin_rcor.setSingleStep(0.05)
        self.spin_rcor.setValue(0.0)
        gain_grid.addWidget(QtWidgets.QLabel("rcor"), 0, 4)
        gain_grid.addWidget(self.spin_rcor, 0, 5)

        self.spin_sf = QtWidgets.QDoubleSpinBox()
        self.spin_sf.setRange(0.0, 100.0)
        self.spin_sf.setDecimals(4)
        self.spin_sf.setSingleStep(0.001)
        self.spin_sf.setValue(float(self.params.sf))
        gain_grid.addWidget(QtWidgets.QLabel("sf"), 1, 0)
        gain_grid.addWidget(self.spin_sf, 1, 1)

        self.spin_tvg = QtWidgets.QDoubleSpinBox()
        self.spin_tvg.setRange(0.0, 20.0)
        self.spin_tvg.setDecimals(3)
        self.spin_tvg.setSingleStep(0.05)
        self.spin_tvg.setValue(1.0)
        gain_grid.addWidget(QtWidgets.QLabel("tvg"), 1, 2)
        gain_grid.addWidget(self.spin_tvg, 1, 3)

        self.spin_pvg = QtWidgets.QDoubleSpinBox()
        self.spin_pvg.setRange(-4.0, 4.0)
        self.spin_pvg.setDecimals(3)
        self.spin_pvg.setSingleStep(0.05)
        self.spin_pvg.setValue(1.0)
        gain_grid.addWidget(QtWidgets.QLabel("pvg"), 1, 4)
        gain_grid.addWidget(self.spin_pvg, 1, 5)
        self.spin_clip = QtWidgets.QDoubleSpinBox()
        self.spin_clip.setRange(0.0, 20.0)
        self.spin_clip.setDecimals(3)
        self.spin_clip.setSingleStep(0.1)
        self.spin_clip.setValue(float(self.params.clip))
        gain_grid.addWidget(QtWidgets.QLabel("clip"), 2, 0)
        gain_grid.addWidget(self.spin_clip, 2, 1)

        self.spin_dscale = QtWidgets.QDoubleSpinBox()
        self.spin_dscale.setRange(0.2, 5.0)
        self.spin_dscale.setDecimals(3)
        self.spin_dscale.setSingleStep(0.05)
        self.spin_dscale.setValue(1.0)
        gain_grid.addWidget(QtWidgets.QLabel("dscale"), 2, 2)
        gain_grid.addWidget(self.spin_dscale, 2, 3)

        self.chk_filter = QtWidgets.QCheckBox("滤波")
        self.chk_filter.setChecked(True)
        self.chk_gain = QtWidgets.QCheckBox("增益")
        self.chk_gain.setChecked(True)
        self.chk_gain.setToolTip(
            "增益总开关。关闭后完全跳过增益与裁剪（保留滤波后原始幅度）。"
            "注意：iscale=0 是「自动归一化」，不是关闭增益。"
        )
        self.chk_rmean = QtWidgets.QCheckBox("rmean")
        self.chk_rmean.setChecked(True)
        self.chk_rmean.setToolTip(
            "去平均（类似 SAC rmean）：滤波/增益前减去整道均值。\n"
            "注意：默认带通 fL>0 已去掉直流，勾选外观常几乎不变；"
            "请关闭「滤波」再对比，或看长记录上的基线偏移。"
        )
        self.chk_rtrend = QtWidgets.QCheckBox("rtrend")
        self.chk_rtrend.setChecked(False)
        self.chk_rtrend.setToolTip(
            "去趋势（类似 SAC rtrend）：滤波/增益前减去线性趋势。\n"
            "注意：带通开启时缓变趋势多半已被滤掉，勾选外观常几乎不变；"
            "请关闭「滤波」后再看漂移基线。"
        )

        self.spin_freqlo = QtWidgets.QDoubleSpinBox()
        self.spin_freqlo.setRange(0.1, 2000.0)
        self.spin_freqlo.setDecimals(2)
        self.spin_freqlo.setSingleStep(0.1)
        self.spin_freqlo.setValue(3.0)
        gain_grid.addWidget(QtWidgets.QLabel("fL"), 2, 4)
        gain_grid.addWidget(self.spin_freqlo, 2, 5)

        self.spin_freqhi = QtWidgets.QDoubleSpinBox()
        self.spin_freqhi.setRange(0.2, 4000.0)
        self.spin_freqhi.setDecimals(2)
        self.spin_freqhi.setSingleStep(0.1)
        self.spin_freqhi.setValue(15.0)
        gain_grid.addWidget(QtWidgets.QLabel("fH"), 3, 0)
        gain_grid.addWidget(self.spin_freqhi, 3, 1)

        self.spin_npoles = QtWidgets.QSpinBox()
        self.spin_npoles.setRange(1, 16)
        self.spin_npoles.setValue(8)
        gain_grid.addWidget(QtWidgets.QLabel("npoles"), 3, 2)
        gain_grid.addWidget(self.spin_npoles, 3, 3)
        self.chk_zerop = QtWidgets.QCheckBox("零相位")
        self.chk_zerop.setChecked(True)
        # 开关列：滤波/增益/rmean/rtr/零相位，紧挨 rcor·pvg·fL 右侧
        self.chk_rtrend.setText("rtr")
        gain_switch_col = QtWidgets.QVBoxLayout()
        gain_switch_col.setContentsMargins(2, 0, 0, 0)
        gain_switch_col.setSpacing(0)
        for chk in (
            self.chk_filter,
            self.chk_gain,
            self.chk_rmean,
            self.chk_rtrend,
            self.chk_zerop,
        ):
            gain_switch_col.addWidget(chk)
        gain_switch_col.addStretch(1)
        gain_switch_wrap = QtWidgets.QWidget()
        gain_switch_wrap.setLayout(gain_switch_col)
        gain_grid.addWidget(gain_switch_wrap, 0, 6, 4, 1)
        gain_preset_row = QtWidgets.QHBoxLayout()
        gain_preset_row.setContentsMargins(0, 0, 0, 0)
        gain_preset_row.setSpacing(1)
        gain_preset_row.addWidget(self.btn_gain_preset_balanced)
        gain_preset_row.addWidget(self.btn_far_offset_boost)
        gain_preset_row.addWidget(self.btn_gain_preset_strong)
        gain_preset_row.addStretch(1)
        gain_preset_wrap = QtWidgets.QWidget()
        gain_preset_wrap.setLayout(gain_preset_row)
        gain_grid.addWidget(gain_preset_wrap, 4, 0, 1, 7)
        self.lbl_gain_hint = QtWidgets.QLabel("")
        self.lbl_gain_hint.setStyleSheet("color:#666; font-size:10px;")
        self.lbl_gain_hint.setMaximumHeight(14)
        gain_grid.addWidget(self.lbl_gain_hint, 5, 0, 1, 7)
        for c in (1, 3, 5):
            gain_grid.setColumnStretch(c, 1)
        gain_grid.setColumnStretch(6, 0)
        gain_group_layout = QtWidgets.QVBoxLayout(self.group_gain)
        gain_group_layout.setContentsMargins(1, 1, 1, 1)
        gain_group_layout.addWidget(gain_page)
        self.group_gain.setMinimumWidth(200)
        self.group_gain.setMaximumWidth(16777215)
        params_layout.addWidget(self.group_gain, stretch=int(self._param_group_stretch.get("gain", 4)))
        self._param_groups.append(self.group_gain)

        # 去噪面板（独立预留）
        self.group_denoise = QtWidgets.QGroupBox("去噪")
        self._param_group_keys[self.group_denoise] = "denoise"
        self.group_denoise.setMinimumWidth(160)
        self.group_denoise.setMaximumWidth(16777215)
        denoise_page = QtWidgets.QWidget()
        denoise_grid = QtWidgets.QGridLayout(denoise_page)
        denoise_grid.setContentsMargins(1, 1, 1, 1)
        denoise_grid.setHorizontalSpacing(1)
        denoise_grid.setVerticalSpacing(0)

        self.chk_denoise_enabled = QtWidgets.QCheckBox("启用")
        self.chk_denoise_enabled.setChecked(bool(self._denoise_params.get("enabled", False)))
        self.lbl_denoise_title = QtWidgets.QLabel("DN")
        self.lbl_denoise_title.setStyleSheet("font-weight:600; color:#555;")
        denoise_title_row_top = QtWidgets.QHBoxLayout()
        denoise_title_row_top.setContentsMargins(0, 0, 0, 0)
        denoise_title_row_top.setSpacing(2)
        denoise_title_row_top.addWidget(self.lbl_denoise_title)
        denoise_title_row_top.addWidget(self.chk_denoise_enabled)
        self.chk_denoise_ab_raw = QtWidgets.QCheckBox("A/B")
        # UI 语义：勾选=去噪(B)；内部参数 ab_raw 语义：True=原始(A)
        self.chk_denoise_ab_raw.setChecked(not bool(self._denoise_params.get("ab_raw", False)))
        self.chk_denoise_ab_raw.setToolTip("A/B对比：勾选=去噪道(B)，取消勾选=原始道(A)")
        denoise_title_row_top.addWidget(self.chk_denoise_ab_raw)
        self.chk_denoise_show_diff = QtWidgets.QCheckBox("差值")
        self.chk_denoise_show_diff.setChecked(bool(self._denoise_params.get("show_diff", False)))
        self.chk_denoise_show_diff.setToolTip(
            "差值：每道 = (去噪后 − 去噪前) × 增益。\n"
            "若沿同相轴出现强条带/双曲能量，多为时频收缩过强（有效信号被大量改写）；"
            "可试：降低强度、提高相干「回混(cb)」、略降阈值(ct)，或 Morph 改为保守。"
        )
        denoise_title_row_top.addWidget(self.chk_denoise_show_diff)
        self.combo_denoise_diff_gain = QtWidgets.QComboBox()
        self.combo_denoise_diff_gain.addItem("x1", 1.0)
        self.combo_denoise_diff_gain.addItem("x2", 2.0)
        self.combo_denoise_diff_gain.addItem("x5", 5.0)
        self.combo_denoise_diff_gain.addItem("x10", 10.0)
        self.combo_denoise_diff_gain.addItem("x20", 20.0)
        diff_gain = float(self._denoise_params.get("diff_gain", 1.0))
        idx_diff_gain = self.combo_denoise_diff_gain.findData(diff_gain)
        if idx_diff_gain < 0:
            idx_diff_gain = 0
        self.combo_denoise_diff_gain.setCurrentIndex(idx_diff_gain)
        self.combo_denoise_diff_gain.setToolTip("差值显示放大倍数")
        self.combo_denoise_diff_gain.setMaximumWidth(52)
        denoise_title_row_top.addWidget(self.combo_denoise_diff_gain)
        self.btn_denoise_start = QtWidgets.QPushButton("开始")
        self.btn_denoise_start.setMinimumWidth(44)
        denoise_title_row_top.addWidget(self.btn_denoise_start)
        denoise_title_row_top.addStretch(1)
        denoise_title_wrap = QtWidgets.QWidget()
        denoise_title_wrap.setLayout(denoise_title_row_top)
        denoise_grid.addWidget(denoise_title_wrap, 0, 0, 1, 4)

        self.spin_denoise_f_s = QtWidgets.QDoubleSpinBox()
        self.spin_denoise_f_s.setRange(0.0, 2000.0)
        self.spin_denoise_f_s.setDecimals(2)
        self.spin_denoise_f_s.setSingleStep(0.1)
        self.spin_denoise_f_s.setValue(float(self._denoise_params.get("f_s", 3.0)))
        denoise_grid.addWidget(QtWidgets.QLabel("fL"), 1, 0)
        denoise_grid.addWidget(self.spin_denoise_f_s, 1, 1)

        self.spin_denoise_f_e = QtWidgets.QDoubleSpinBox()
        self.spin_denoise_f_e.setRange(0.1, 4000.0)
        self.spin_denoise_f_e.setDecimals(2)
        self.spin_denoise_f_e.setSingleStep(0.1)
        self.spin_denoise_f_e.setValue(float(self._denoise_params.get("f_e", 20.0)))
        denoise_grid.addWidget(QtWidgets.QLabel("fH"), 1, 2)
        denoise_grid.addWidget(self.spin_denoise_f_e, 1, 3)

        self.spin_denoise_strength = QtWidgets.QDoubleSpinBox()
        self.spin_denoise_strength.setRange(0.01, 20.0)
        self.spin_denoise_strength.setDecimals(3)
        self.spin_denoise_strength.setSingleStep(0.05)
        self.spin_denoise_strength.setValue(float(self._denoise_params.get("strength", 3.0)))
        denoise_grid.addWidget(QtWidgets.QLabel("str"), 2, 0)
        denoise_grid.addWidget(self.spin_denoise_strength, 2, 1)

        self.combo_denoise_bwconn = QtWidgets.QComboBox()
        self.combo_denoise_bwconn.addItems(["4", "8"])
        self.combo_denoise_bwconn.setCurrentText(str(int(self._denoise_params.get("bwconn", 8))))
        denoise_grid.addWidget(QtWidgets.QLabel("bw"), 2, 2)
        denoise_grid.addWidget(self.combo_denoise_bwconn, 2, 3)

        self.spin_denoise_workers = QtWidgets.QSpinBox()
        self.spin_denoise_workers.setRange(1, 64)
        self.spin_denoise_workers.setValue(max(1, int(self._denoise_params.get("workers", 1))))
        denoise_grid.addWidget(QtWidgets.QLabel("wk"), 3, 0)
        denoise_grid.addWidget(self.spin_denoise_workers, 3, 1)

        self.combo_denoise_scope = QtWidgets.QComboBox()
        self.combo_denoise_scope.addItem("渲染道", "rendered")
        self.combo_denoise_scope.addItem("视窗道", "visible")
        self.combo_denoise_scope.addItem("记录道", "record")
        self.combo_denoise_scope.addItem("手动选道", "selected")
        scope_value = str(self._denoise_params.get("scope", "rendered")).strip().lower()
        scope_idx = self.combo_denoise_scope.findData(scope_value)
        if scope_idx < 0:
            scope_idx = 0
        self.combo_denoise_scope.setCurrentIndex(scope_idx)
        denoise_grid.addWidget(QtWidgets.QLabel("范围"), 4, 0)
        denoise_grid.addWidget(self.combo_denoise_scope, 4, 1)
        self.btn_denoise_clear_selected = QtWidgets.QToolButton()
        self.btn_denoise_clear_selected.setText("清空")
        self.btn_denoise_coh_cfg = QtWidgets.QToolButton()
        self.btn_denoise_coh_cfg.setText("Coh参数")
        self.btn_denoise_compare_plot = QtWidgets.QToolButton()
        self.btn_denoise_compare_plot.setText("对比")
        self.btn_denoise_compare_plot.setToolTip(
            "Left: single-trace A/B (time, spectrum, TF). Menu ▾: all cached traces — "
            "spectrum gathers (freq × trace) and |TF| horizontal concat (single denoise_trace per trace)."
        )
        self.btn_denoise_compare_plot.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        _menu_denoise_cmp = QtWidgets.QMenu(self.btn_denoise_compare_plot)
        _act_denoise_cmp_single = QtGui.QAction("单道对比…", self.btn_denoise_compare_plot)
        _act_denoise_cmp_single.triggered.connect(self._open_denoise_compare_plot)
        _act_denoise_cmp_all = QtGui.QAction("全道：谱剖面 + 拼接|TF|…", self.btn_denoise_compare_plot)
        _act_denoise_cmp_all.triggered.connect(self._open_denoise_compare_all_tf_spectrum)
        _menu_denoise_cmp.addAction(_act_denoise_cmp_single)
        _menu_denoise_cmp.addAction(_act_denoise_cmp_all)
        self.btn_denoise_compare_plot.setMenu(_menu_denoise_cmp)
        select_row = QtWidgets.QHBoxLayout()
        select_row.setContentsMargins(0, 0, 0, 0)
        select_row.setSpacing(2)
        select_row.addWidget(self.btn_denoise_clear_selected)
        select_row.addWidget(self.btn_denoise_coh_cfg)
        select_row.addWidget(self.btn_denoise_compare_plot)
        select_row.addStretch(1)
        select_wrap = QtWidgets.QWidget()
        select_wrap.setLayout(select_row)
        denoise_grid.addWidget(select_wrap, 4, 2, 1, 2)

        self.chk_denoise_pick_guidance = QtWidgets.QCheckBox("拾取引导")
        self.chk_denoise_pick_guidance.setChecked(bool(self._denoise_params.get("pick_guidance", False)))
        self.chk_denoise_pick_guidance.setToolTip(
            "GCV 之后按「拾取模板」做 TF 软门控：模板 = 当前去噪范围内各道拾取时刻的并集，"
            "同一套时间门应用于该范围内每一道（含本道无拾取的道）。范围内完全无拾取则本项不生效。"
        )
        self.spin_denoise_pick_hw = QtWidgets.QDoubleSpinBox()
        self.spin_denoise_pick_hw.setRange(0.02, 5.0)
        self.spin_denoise_pick_hw.setDecimals(3)
        self.spin_denoise_pick_hw.setSingleStep(0.01)
        self.spin_denoise_pick_hw.setToolTip(
            "子波长度 T：时间软门控高斯包络的半高全宽 FWHM（秒），σ = T / (2√(2ln2))。"
        )
        self.spin_denoise_pick_hw.setValue(float(self._denoise_params.get("pick_wavelet_length_sec", 0.19)))
        self.spin_denoise_pick_floor = QtWidgets.QDoubleSpinBox()
        self.spin_denoise_pick_floor.setRange(0.0, 0.95)
        self.spin_denoise_pick_floor.setDecimals(3)
        self.spin_denoise_pick_floor.setSingleStep(0.02)
        self.spin_denoise_pick_floor.setValue(float(self._denoise_params.get("pick_guidance_floor", 0.12)))
        pg_row = QtWidgets.QHBoxLayout()
        pg_row.setContentsMargins(0, 0, 0, 0)
        pg_row.setSpacing(2)
        pg_row.addWidget(QtWidgets.QLabel("PG"))
        pg_row.addWidget(self.chk_denoise_pick_guidance)
        pg_row.addWidget(QtWidgets.QLabel("T(s)"))
        pg_row.addWidget(self.spin_denoise_pick_hw)
        pg_row.addWidget(QtWidgets.QLabel("外侧"))
        pg_row.addWidget(self.spin_denoise_pick_floor)
        pg_row.addStretch(1)
        pg_wrap = QtWidgets.QWidget()
        pg_wrap.setLayout(pg_row)
        denoise_grid.addWidget(pg_wrap, 5, 0, 1, 4)

        self.lbl_denoise_hint = QtWidgets.QLabel("")
        self.lbl_denoise_hint.setStyleSheet("color:#666; font-size:10px;")
        self.lbl_denoise_hint.setMaximumHeight(14)
        self.lbl_denoise_hint.setWordWrap(False)
        denoise_grid.addWidget(self.lbl_denoise_hint, 6, 0, 1, 4)
        self.progress_denoise = QtWidgets.QProgressBar()
        self.progress_denoise.setRange(0, 100)
        self.progress_denoise.setValue(0)
        self.progress_denoise.setFormat("DN %p%")
        self.progress_denoise.setTextVisible(True)
        self.progress_denoise.setFixedHeight(10)
        self.progress_denoise.setVisible(False)
        denoise_grid.addWidget(self.progress_denoise, 7, 0, 1, 4)

        denoise_group_layout = QtWidgets.QVBoxLayout(self.group_denoise)
        denoise_group_layout.setContentsMargins(1, 1, 1, 1)
        denoise_group_layout.addWidget(denoise_page)
        params_layout.addWidget(self.group_denoise, stretch=int(self._param_group_stretch.get("denoise", 3)))
        self._param_groups.append(self.group_denoise)

        # 拾取面板（紧凑网格）
        self.group_pick = QtWidgets.QGroupBox("拾取")
        self._param_group_keys[self.group_pick] = "pick"
        self.group_pick.setMinimumWidth(150)
        self.group_pick.setMaximumWidth(16777215)
        pick_page = QtWidgets.QWidget()
        pick_grid = QtWidgets.QGridLayout(pick_page)
        pick_grid.setContentsMargins(1, 1, 1, 1)
        pick_grid.setHorizontalSpacing(1)
        pick_grid.setVerticalSpacing(0)

        self.chk_pick_mode = QtWidgets.QCheckBox("拾取模式")
        self.chk_pick_mode.setChecked(False)
        self.chk_pick_mode.setToolTip("左键添加/更新拾取，右键删除当前拾取字")
        pick_grid.addWidget(QtWidgets.QLabel("拾取"), 0, 0, alignment=QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        pick_grid.addWidget(self.chk_pick_mode, 0, 1, alignment=QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)

        self.spin_apick = QtWidgets.QSpinBox()
        self.spin_apick.setRange(1, 200)
        self.spin_apick.setValue(1)
        pick_grid.addWidget(QtWidgets.QLabel("apick"), 0, 2, alignment=QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        pick_grid.addWidget(self.spin_apick, 0, 3, alignment=QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

        self.spin_pick_size = QtWidgets.QSpinBox()
        self.spin_pick_size.setRange(2, 40)
        self.spin_pick_size.setValue(8)
        pick_grid.addWidget(QtWidgets.QLabel("圆圈"), 1, 0, alignment=QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        pick_grid.addWidget(self.spin_pick_size, 1, 1, alignment=QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)

        self.spin_tcrcor = QtWidgets.QDoubleSpinBox()
        self.spin_tcrcor.setRange(0.05, 5.0)
        self.spin_tcrcor.setDecimals(3)
        self.spin_tcrcor.setSingleStep(0.05)
        self.spin_tcrcor.setValue(float(self.params.tcrcor))
        pick_grid.addWidget(QtWidgets.QLabel("tcrcor"), 1, 2, alignment=QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        pick_grid.addWidget(self.spin_tcrcor, 1, 3, alignment=QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

        self.spin_tlag = QtWidgets.QDoubleSpinBox()
        self.spin_tlag.setRange(0.005, 1.0)
        self.spin_tlag.setDecimals(3)
        self.spin_tlag.setSingleStep(0.01)
        self.spin_tlag.setValue(float(self.params.tlag))
        pick_grid.addWidget(QtWidgets.QLabel("tlag"), 2, 0, alignment=QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        pick_grid.addWidget(self.spin_tlag, 2, 1, alignment=QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)

        self.spin_hilbratio = QtWidgets.QDoubleSpinBox()
        self.spin_hilbratio.setRange(0.0, 20.0)
        self.spin_hilbratio.setDecimals(3)
        self.spin_hilbratio.setSingleStep(0.2)
        self.spin_hilbratio.setValue(float(self.params.hilbratio))
        pick_grid.addWidget(QtWidgets.QLabel("hilbr"), 2, 2, alignment=QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        pick_grid.addWidget(self.spin_hilbratio, 2, 3, alignment=QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

        # 让列随面板变宽而拉伸
        for c in range(4):
            pick_grid.setColumnStretch(c, 1 if c % 2 == 1 else 0)

        pick_btn_grid = QtWidgets.QGridLayout()
        pick_btn_grid.setSpacing(2)
        pick_btn_grid.setContentsMargins(0, 0, 0, 0)
        for btn in (self.btn_auto_pick, self.btn_interp_pick, self.btn_save_picks, self.btn_clear_picks):
            try:
                btn.setMinimumWidth(48)
                btn.setMaximumHeight(20)
            except Exception:
                pass
        pick_btn_grid.addWidget(self.btn_auto_pick, 0, 0)
        pick_btn_grid.addWidget(self.btn_interp_pick, 0, 1)
        pick_btn_grid.addWidget(self.btn_save_picks, 0, 2)
        pick_btn_grid.addWidget(self.btn_clear_picks, 0, 3)
        pick_btn_wrap = QtWidgets.QWidget()
        pick_btn_wrap.setLayout(pick_btn_grid)
        pick_grid.addWidget(pick_btn_wrap, 3, 0, 1, 4)
        # Mute 与按钮同一行高度区：并入第 3 行旁侧太挤，放第 4 行横排
        mute_row = QtWidgets.QHBoxLayout()
        mute_row.setContentsMargins(0, 0, 0, 0)
        mute_row.setSpacing(2)
        self.btn_mute_status.setMinimumWidth(72)
        self.btn_clear_mute.setMinimumWidth(44)
        mute_row.addWidget(self.btn_mute_status)
        mute_row.addWidget(self.chk_mute_invert)
        mute_row.addWidget(self.btn_clear_mute)
        mute_row.addStretch(1)
        mute_wrap = QtWidgets.QWidget()
        mute_wrap.setLayout(mute_row)
        pick_grid.addWidget(mute_wrap, 4, 0, 1, 4)
        pick_group_layout = QtWidgets.QVBoxLayout(self.group_pick)
        pick_group_layout.setContentsMargins(1, 1, 1, 1)
        pick_group_layout.addWidget(pick_page)
        params_layout.addWidget(self.group_pick, stretch=int(self._param_group_stretch.get("pick", 3)))
        self._param_groups.append(self.group_pick)

        # 对齐/叠加：单列按钮，窄面板
        self.group_align = QtWidgets.QGroupBox("对齐/叠加")
        self._param_group_keys[self.group_align] = "align"
        self.group_align.setMinimumWidth(88)
        self.group_align.setMaximumWidth(16777215)
        align_page = QtWidgets.QWidget()
        align_col = QtWidgets.QVBoxLayout(align_page)
        align_col.setContentsMargins(1, 1, 1, 1)
        align_col.setSpacing(1)

        self.chk_show_stack = QtWidgets.QCheckBox("显示叠加")
        self.chk_show_stack.setChecked(False)
        align_col.addWidget(self.chk_show_stack)
        self.btn_align_pick.setText("对齐(A)")
        self.btn_clear_align.setText("清对齐")
        self.btn_align_adaptive.setText("更新(F)")
        self.btn_eval_stack.setText("评价")
        for btn in (self.btn_align_pick, self.btn_clear_align, self.btn_align_adaptive, self.btn_eval_stack):
            btn.setMinimumWidth(72)
            btn.setMaximumHeight(20)
            align_col.addWidget(btn)
        align_col.addStretch(1)

        align_group_layout = QtWidgets.QVBoxLayout(self.group_align)
        align_group_layout.setContentsMargins(1, 1, 1, 1)
        align_group_layout.addWidget(align_page)
        params_layout.addWidget(self.group_align, stretch=int(self._param_group_stretch.get("align", 2)))
        self._param_groups.append(self.group_align)

        # 波形操作：单列按钮
        self.group_waveop = QtWidgets.QGroupBox("波形操作")
        self._param_group_keys[self.group_waveop] = "waveop"
        self.group_waveop.setMinimumWidth(80)
        self.group_waveop.setMaximumWidth(16777215)
        waveop_page = QtWidgets.QWidget()
        waveop_col = QtWidgets.QVBoxLayout(waveop_page)
        waveop_col.setContentsMargins(1, 1, 1, 1)
        waveop_col.setSpacing(1)
        self.btn_waveop_stack.setText("叠加")
        self.btn_waveop_att.setText("姿态")
        self.btn_waveop_att.setVisible(False)  # 仅 RelocationViewer 内嵌姿态时显示
        self.btn_waveop_clear.setText("清除V")
        self.btn_waveop_save.setText("存V")
        self.btn_waveop_load.setText("载V")
        for btn in (
            self.btn_waveop_stack,
            self.btn_waveop_att,
            self.btn_waveop_clear,
            self.btn_waveop_save,
            self.btn_waveop_load,
        ):
            btn.setMinimumWidth(68)
            btn.setMaximumHeight(20)
            waveop_col.addWidget(btn)
        waveop_col.addStretch(1)
        waveop_group_layout = QtWidgets.QVBoxLayout(self.group_waveop)
        waveop_group_layout.setContentsMargins(1, 1, 1, 1)
        waveop_group_layout.addWidget(waveop_page)
        params_layout.addWidget(self.group_waveop, stretch=int(self._param_group_stretch.get("waveop", 2)))
        self._param_groups.append(self.group_waveop)

        # 高级校正：2×3 扁网格
        self.group_advcorr = QtWidgets.QGroupBox("高级校正")
        self._param_group_keys[self.group_advcorr] = "advcorr"
        self.group_advcorr.setMinimumWidth(140)
        self.group_advcorr.setMaximumWidth(16777215)
        adv_page = QtWidgets.QWidget()
        adv_grid = QtWidgets.QGridLayout(adv_page)
        adv_grid.setContentsMargins(1, 1, 1, 1)
        adv_grid.setHorizontalSpacing(2)
        adv_grid.setVerticalSpacing(1)
        for btn in (
            self.btn_theory, self.btn_clear_theory, self.btn_water_corr, self.btn_clear_water,
            self.btn_water_curve, self.btn_static_corr, self.btn_clear_static,
        ):
            try:
                btn.setMaximumHeight(20)
            except Exception:
                pass
        adv_grid.addWidget(self.btn_theory, 0, 0)
        adv_grid.addWidget(self.btn_clear_theory, 0, 1)
        adv_grid.addWidget(self.btn_water_corr, 1, 0)
        adv_grid.addWidget(self.btn_clear_water, 1, 1)
        adv_grid.addWidget(self.btn_water_curve, 2, 0)
        adv_grid.addWidget(self.btn_static_corr, 2, 1)
        adv_grid.addWidget(self.btn_clear_static, 3, 0, 1, 2)
        adv_grid.setColumnStretch(0, 1)
        adv_grid.setColumnStretch(1, 1)
        adv_group_layout = QtWidgets.QVBoxLayout(self.group_advcorr)
        adv_group_layout.setContentsMargins(1, 1, 1, 1)
        adv_group_layout.addWidget(adv_page)
        params_layout.addWidget(self.group_advcorr, stretch=int(self._param_group_stretch.get("advcorr", 2)))
        self._param_groups.append(self.group_advcorr)

        # 走时模板：2×2 按钮 + 选项横排
        self.group_ttpl = QtWidgets.QGroupBox("走时模板")
        self._param_group_keys[self.group_ttpl] = "ttpl"
        self.group_ttpl.setMinimumWidth(150)
        self.group_ttpl.setMaximumWidth(16777215)
        ttpl_page = QtWidgets.QWidget()
        ttpl_grid = QtWidgets.QGridLayout(ttpl_page)
        ttpl_grid.setContentsMargins(1, 1, 1, 1)
        ttpl_grid.setHorizontalSpacing(2)
        ttpl_grid.setVerticalSpacing(1)
        for btn in (self.btn_load_txin, self.btn_clear_txin, self.btn_preview_map_txin, self.btn_map_txin):
            try:
                btn.setMaximumHeight(20)
            except Exception:
                pass
        self.btn_load_txin.setText("读tx.in")
        self.btn_clear_txin.setText("清除")
        self.btn_preview_map_txin.setText("预览")
        self.btn_map_txin.setText("映射")
        ttpl_grid.addWidget(self.btn_load_txin, 0, 0)
        ttpl_grid.addWidget(self.btn_clear_txin, 0, 1)
        ttpl_grid.addWidget(self.btn_preview_map_txin, 1, 0)
        ttpl_grid.addWidget(self.btn_map_txin, 1, 1)
        opt_row = QtWidgets.QHBoxLayout()
        opt_row.setContentsMargins(0, 0, 0, 0)
        opt_row.setSpacing(2)
        self.chk_map_txin_apick_only.setText("仅apick")
        self.chk_map_txin_view_only.setText("仅视窗")
        opt_row.addWidget(self.chk_map_txin_apick_only)
        opt_row.addWidget(self.chk_map_txin_view_only)
        opt_wrap = QtWidgets.QWidget()
        opt_wrap.setLayout(opt_row)
        ttpl_grid.addWidget(opt_wrap, 2, 0, 1, 2)
        tol_row = QtWidgets.QHBoxLayout()
        tol_row.setContentsMargins(0, 0, 0, 0)
        tol_row.setSpacing(2)
        tol_row.addWidget(QtWidgets.QLabel("容差%"))
        tol_row.addWidget(self.spin_map_txin_tol)
        tol_wrap = QtWidgets.QWidget()
        tol_wrap.setLayout(tol_row)
        ttpl_grid.addWidget(tol_wrap, 3, 0, 1, 2)
        ttpl_grid.setColumnStretch(0, 1)
        ttpl_grid.setColumnStretch(1, 1)
        ttpl_group_layout = QtWidgets.QVBoxLayout(self.group_ttpl)
        ttpl_group_layout.setContentsMargins(1, 1, 1, 1)
        ttpl_group_layout.addWidget(ttpl_page)
        params_layout.addWidget(self.group_ttpl, stretch=int(self._param_group_stretch.get("ttpl", 2)))
        self._param_groups.append(self.group_ttpl)
        # 不再尾部 stretch：横向空间全部分给各面板

        # 参数面板支持拖拽重排：在组标题区域按住左键拖动，释放后重排
        for group in self._param_groups:
            try:
                group.installEventFilter(self)
                group.setMouseTracking(True)
                group.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
                key = self._param_group_keys.get(group, "")
                min_w = max(64, int(group.minimumWidth() or 64))
                # 横向尽量拉开：上限放宽，便于均分窗口宽度
                max_w = 900
                group.setMinimumWidth(min_w)
                group.setMaximumWidth(max_w)
                try:
                    pol = group.sizePolicy()
                    pol.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Expanding)
                    pol.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Maximum)
                    group.setSizePolicy(pol)
                except Exception:
                    pass
                self._panel_resize_bounds[id(group)] = (min_w, max_w)
                # 按权重写入 stretch（恢复顺序后也会再设）
                idx = self._params_layout.indexOf(group)
                if idx >= 0:
                    self._params_layout.setStretch(idx, int(self._param_group_stretch.get(key, 2)))
            except Exception:
                pass
        self._restore_panel_layout_state()
        self._reapply_param_group_stretches()

        # 紧凑外观：统一控件最小高度，减少垂向占用
        compact_widgets = [
            self.spin_irec, self.combo_itype, self.spin_nskip, self.spin_ndecim, self.combo_mode,
            self.spin_vred, self.spin_xmin, self.spin_xmax, self.spin_tmin, self.spin_tmax,
            self.spin_amp, self.combo_iscale, self.spin_rcor, self.spin_tvg, self.spin_pvg,
            self.spin_sf, self.spin_clip, self.spin_dscale,
            self.spin_freqlo, self.spin_freqhi, self.spin_npoles,
            self.spin_denoise_f_s, self.spin_denoise_f_e, self.spin_denoise_strength, self.combo_denoise_bwconn,
            self.spin_denoise_workers,
            self.spin_apick, self.spin_pick_size
            , self.spin_tcrcor, self.spin_tlag, self.spin_hilbratio
        ]
        for w in compact_widgets:
            try:
                w.setMinimumHeight(18)
            except Exception:
                pass
        # 参数面板字体保持默认（如需再调可改 step）
        self._apply_params_panel_font_boost(step=0)
        self._sync_params_panel_height_constraints()
        self._update_gain_effect_hint()
        self._sync_denoise_params_from_ui()
        self._update_denoise_hint()
        self._fit_params_strip_to_content()
        # 嵌入时少做异步重排，避免首次切入工区页时高度连跳闪烁
        if self.parent() is None:
            self._schedule_unify_param_group_heights(50)

        self.plot = pg.PlotWidget(background="w")
        self.plot.showGrid(x=True, y=True, alpha=0.12)
        self.plot.getPlotItem().setLabels(left="Time (s)", bottom="Offset (km)")
        # 关闭 PyQtGraph 默认右键菜单（View All / X Axis / Y Axis / Mouse Mode）
        self.plot.getPlotItem().setMenuEnabled(False)
        self.plot.getViewBox().setMenuEnabled(False)
        # 地震图习惯：时间向下
        self.plot.getViewBox().invertY(True)

        # 独立运行时：V 段列表放在剖面下方薄条；嵌入工区后可 reparent 到输出页签
        self._waveop_list_local_dock = QtWidgets.QFrame()
        self._waveop_list_local_dock.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self._waveop_list_local_dock.setMinimumHeight(36)
        self._waveop_list_local_dock.setMaximumHeight(200)
        dock_lay = QtWidgets.QHBoxLayout(self._waveop_list_local_dock)
        dock_lay.setContentsMargins(4, 2, 4, 2)
        dock_lay.setSpacing(6)
        dock_tip = QtWidgets.QLabel("V段")
        dock_tip.setStyleSheet("color:#475569; font-weight:600;")
        dock_lay.addWidget(dock_tip, stretch=0)
        dock_lay.addWidget(self.list_waveop_segments, stretch=1)

        # 垂直分割：参数面板 | 绘图区 | V段条（可上下拖拽缩放）
        self._body_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self._body_splitter.setObjectName("ZplotBodySplitter")
        self._body_splitter.setChildrenCollapsible(False)
        self._body_splitter.setHandleWidth(6)
        self._body_splitter.addWidget(self.params_panel_scroll)
        self._body_splitter.addWidget(self.plot)
        self._body_splitter.addWidget(self._waveop_list_local_dock)
        self._body_splitter.setStretchFactor(0, 0)
        self._body_splitter.setStretchFactor(1, 1)
        self._body_splitter.setStretchFactor(2, 0)
        default_h = int(self._params_panel_default_height)
        self._body_splitter.setSizes([default_h, 700, 64])
        self._body_splitter.splitterMoved.connect(self._on_body_splitter_moved)
        root.addWidget(self._body_splitter, stretch=1)
        self._fit_params_strip_to_content()
        if self.parent() is None:
            self._schedule_unify_param_group_heights(100)

        self.lbl_status = QtWidgets.QLabel("就绪")
        # 姿态预览徽章仅在 RelocationViewer（mixin）中启用并显示
        self.lbl_orientation_preview = QtWidgets.QLabel("姿态校正预览: OFF")
        self.lbl_orientation_preview.setStyleSheet("color:#64748b; font-weight:600;")
        self.lbl_orientation_preview.hide()
        self.chk_orientation_preview_toggle = QtWidgets.QCheckBox("快速切换")
        self.chk_orientation_preview_toggle.setEnabled(False)
        self.chk_orientation_preview_toggle.setChecked(False)
        self.chk_orientation_preview_toggle.hide()
        self.lbl_pick_link_mode = QtWidgets.QLabel("")
        self.lbl_pick_link_mode.setStyleSheet("color:#0f766e; font-weight:600;")
        status_row = QtWidgets.QHBoxLayout()
        status_row.addWidget(self.lbl_status, stretch=1)
        status_row.addWidget(self.lbl_pick_link_mode, stretch=0)
        status_row.addWidget(self.chk_orientation_preview_toggle, stretch=0)
        status_row.addWidget(self.lbl_orientation_preview, stretch=0)
        root.addLayout(status_row)

        self._dfile: Optional[str] = None
        self._hfile: Optional[str] = None
        self._rfile: Optional[str] = None
        self._theory_model_file: Optional[str] = None
        self._update_file_open_status_label()

