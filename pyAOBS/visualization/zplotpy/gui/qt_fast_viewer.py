"""
qt_fast_viewer.py - PySide6/PyQtGraph 波形快速查看器。

本文件是薄壳：状态初始化 + show/resize/close + ``main()``。
业务方法在 ``gui/mixins/``（索引见 ``mixins/__init__.py``）。
"""

from __future__ import annotations

import os
import sys
import threading
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as exc:
    raise RuntimeError(
        "未安装 PySide6，请先安装：pip install PySide6 pyqtgraph"
    ) from exc

try:
    import pyqtgraph as pg
except Exception as exc:
    raise RuntimeError(
        "未安装 pyqtgraph，请先安装：pip install pyqtgraph"
    ) from exc

try:
    from ..core.data_loader import DataLoader
    from ..core.data_processor import DataProcessor
    from ..core.adaptive_stack import AdaptiveStacker
    from ..core.parameters import ZPlotParameters
    from ..core.pick_manager import PickManager
    from ..core.static_correction import StaticCorrector
    from ..core.stacking_evaluator import StackingEvaluator
    from ..core.theoretical_traveltime import TheoreticalTravelTimeCalculator
    from ..core.auto_picker import AutoPicker
    from ..core.interpolation_correlation_picker import InterpolationCorrelationPicker
    from ..core.src_kernel_bridge import SrcShadeKernelBridge
except ImportError:
    from pyAOBS.visualization.zplotpy.core.data_loader import DataLoader
    from pyAOBS.visualization.zplotpy.core.data_processor import DataProcessor
    from pyAOBS.visualization.zplotpy.core.adaptive_stack import AdaptiveStacker
    from pyAOBS.visualization.zplotpy.core.parameters import ZPlotParameters
    from pyAOBS.visualization.zplotpy.core.pick_manager import PickManager
    from pyAOBS.visualization.zplotpy.core.static_correction import StaticCorrector
    from pyAOBS.visualization.zplotpy.core.stacking_evaluator import StackingEvaluator
    from pyAOBS.visualization.zplotpy.core.theoretical_traveltime import TheoreticalTravelTimeCalculator
    from pyAOBS.visualization.zplotpy.core.auto_picker import AutoPicker
    from pyAOBS.visualization.zplotpy.core.interpolation_correlation_picker import InterpolationCorrelationPicker
    from pyAOBS.visualization.zplotpy.core.src_kernel_bridge import SrcShadeKernelBridge

from .mixins import (
    BuildUiMixin,
    DataInfoMixin,
    DenoiseCompareMixin,
    DenoiseMixin,
    FileIoMixin,
    HelpMixin,
    LocationMapMixin,
    LocationTerrainMixin,
    MuteMixin,
    ParamPanelMixin,
    PickMixin,
    PlotInteractionMixin,
    RenderCoreMixin,
    StackMixin,
    ThemeMixin,
    TraceGeomMixin,
    TravelStaticMixin,
    UiParamsMixin,
    WaveopMixin,
)

class QtFastViewer(
    DenoiseMixin,
    DenoiseCompareMixin,
    LocationMapMixin,
    LocationTerrainMixin,
    ParamPanelMixin,
    MuteMixin,
    WaveopMixin,
    PickMixin,
    TravelStaticMixin,
    ThemeMixin,
    HelpMixin,
    DataInfoMixin,
    StackMixin,
    BuildUiMixin,
    FileIoMixin,
    UiParamsMixin,
    PlotInteractionMixin,
    TraceGeomMixin,
    RenderCoreMixin,
    QtWidgets.QMainWindow,
):
    """PySide6 + PyQtGraph 大文件快速浏览原型。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        # 嵌入工区时作为子控件：勿按独立顶层窗尺寸初始化，避免首次切入时闪缩
        if parent is None:
            self.setWindowTitle("ZPLOT Qt Fast Viewer")
            self.resize(1500, 900)
        else:
            self.setWindowFlags(QtCore.Qt.WindowType.Widget)
            self.setWindowTitle("ZPLOT Qt Fast Viewer")
        self._active_theme_mode = "default"
        self._active_theme: Dict[str, str] = dict(self._THEME_PRESETS["default"])

        self.loader = DataLoader()
        self.processor = DataProcessor(enable_cache=True, cache_size=256)
        self.adaptive_stacker = AdaptiveStacker()
        self.params = ZPlotParameters()
        self.loaded = None
        self.pick_manager: Optional[PickManager] = None
        self.auto_picker = AutoPicker()
        self.interp_picker = InterpolationCorrelationPicker()
        self.static_corrector = StaticCorrector()
        self.stacking_evaluator = StackingEvaluator()
        self.static_correction_enabled = False
        self.static_preview_mode = False
        self.last_stacking_result: Optional[Dict[str, object]] = None
        self.theoretical_traveltime_calculator: Optional[TheoreticalTravelTimeCalculator] = None
        self.theoretical_times_data: Optional[Dict[str, np.ndarray]] = None
        self.show_theoretical_times = False
        self.txin_overlay_data: Optional[Dict[str, np.ndarray]] = None
        self.show_txin_overlay = False
        self.txin_map_preview_data: Optional[Dict[str, np.ndarray]] = None
        self.water_layer_corrections: Dict[int, float] = {}
        self.water_layer_corrected_times: Optional[Dict[str, np.ndarray]] = None
        self.show_water_layer_correction = False
        try:
            self.shade_kernel = SrcShadeKernelBridge()
        except Exception:
            self.shade_kernel = None

        self._curve_items: List[pg.PlotDataItem] = []
        self._shade_item: Optional[pg.PlotDataItem] = None
        self._density_item: Optional[pg.ImageItem] = None
        self._density_hl_item: Optional[pg.PlotDataItem] = None  # Density 下选中道竖线
        self._pick_item: Optional[pg.ScatterPlotItem] = None
        self._stack_item: Optional[pg.PlotDataItem] = None
        self._static_preview_item: Optional[pg.PlotDataItem] = None
        self._theoretical_item: Optional[pg.PlotDataItem] = None
        self._txin_item: Optional[pg.ScatterPlotItem] = None
        self._txin_map_preview_item: Optional[pg.ScatterPlotItem] = None
        self._water_corr_item: Optional[pg.PlotDataItem] = None
        self._wave_select_items: Dict[int, pg.PlotDataItem] = {}
        self._wave_select_marker_item: Optional[pg.ScatterPlotItem] = None
        self._waveop_stack_item: Optional[pg.PlotDataItem] = None
        self._mute_polygon_item: Optional[pg.PlotDataItem] = None
        self._mute_vertex_item: Optional[pg.ScatterPlotItem] = None
        self._coord_params_dialog: Optional[QtWidgets.QDialog] = None
        self._location_map_dialog: Optional[QtWidgets.QDialog] = None
        self._location_map_mode: str = "none"
        self._location_map_rec_role: str = "auto"
        self._location_map_trace_points: np.ndarray = np.empty((0, 2), dtype=float)
        self._location_map_trace_indices: np.ndarray = np.empty((0,), dtype=int)
        self._location_map_cursor_item: Optional[pg.ScatterPlotItem] = None
        self._location_map_selected_item: Optional[pg.ScatterPlotItem] = None
        self._location_map_plot_item = None
        self._location_map_plot_widget: Optional[pg.PlotWidget] = None
        self._location_map_base_bounds: Optional[Tuple[float, float, float, float]] = None
        self._location_map_terrain_item = None
        self._location_map_colorbar_gradient = None
        self._location_map_colorbar_min_label: Optional[QtWidgets.QLabel] = None
        self._location_map_colorbar_max_label: Optional[QtWidgets.QLabel] = None
        self._location_map_terrain_meta: Optional[Dict[str, object]] = None
        self._location_map_terrain_cache_key: Optional[Tuple[object, ...]] = None
        self._location_map_terrain_force_geo: bool = False
        self._location_map_terrain_manual_zone_enabled: bool = False
        self._location_map_terrain_manual_zone_value: int = 50
        self._location_map_terrain_manual_hemi: str = "auto"
        self._location_map_terrain_use_sac2y_tm: bool = False
        self._location_map_terrain_tm_lon0: float = 120.0
        self._location_map_terrain_tm_lon_wrap360: bool = True
        self._location_map_terrain_swap_lonlat: bool = False
        self._location_map_terrain_palette: str = "terrain"
        self._location_map_terrain_shade_strength: float = 0.75
        self._location_map_terrain_coast_enhance: bool = True
        self._location_map_terrain_light_alt_deg: float = 45.0
        self._location_map_terrain_light_az_deg: float = 315.0
        self._location_map_terrain_cpt_path: str = ""
        self._location_map_terrain_cpt_cache_key: Optional[Tuple[str, float]] = None
        self._location_map_terrain_cpt_cache_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._location_map_terrain_proj_text: str = ""
        self._location_map_terrain_proj_label: Optional[QtWidgets.QLabel] = None
        self._map_link_trace_idx: Optional[int] = None
        self._static_decision_box: Optional[QtWidgets.QMessageBox] = None
        self._last_render_trace_indices: np.ndarray = np.array([], dtype=int)
        self._last_render_offsets: np.ndarray = np.array([], dtype=float)
        self._last_denoise_scope_count: int = 0
        self._alignment_offsets: Dict[int, float] = {}
        self._removed_traces: set[int] = set()
        self.mouse_x: Optional[float] = None
        self.mouse_y: Optional[float] = None
        self._mute_edit_mode: bool = False
        self._mute_enabled: bool = False
        self._mute_invert: bool = False
        self._mute_polygon_points: List[Tuple[float, float]] = []
        self._mute_drag_vertex_idx: Optional[int] = None
        self._mute_selected_vertex_idx: Optional[int] = None
        self._mute_drag_active: bool = False
        self._mute_drag_last_status_ms: int = 0
        self.delete_range_state = 0
        self.delete_range_x1: Optional[float] = None
        self.last_key_class: Optional[str] = None
        self._pick_undo_stack: List[Tuple[str, Dict[int, Dict[int, float]]]] = []
        self._pick_redo_stack: List[Tuple[str, Dict[int, Dict[int, float]]]] = []
        self._pick_undo_limit = 30
        self._render_timer = QtCore.QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_now)
        self._interaction_end_timer = QtCore.QTimer(self)
        self._interaction_end_timer.setSingleShot(True)
        self._interaction_end_timer.timeout.connect(self._on_interaction_end)
        self._viewport_interacting = False
        self._did_initial_view_fit = False
        self._syncing_window_controls = False
        self._hover_help_key: Optional[str] = None
        self._shift_pressed = False
        self._shift_hover_pick_active = False
        self._shift_hover_pick_undo_pushed = False
        self._shift_hover_pick_updated_count = 0
        self._shift_hover_picked_traces: set[int] = set()
        self._status_hold_until_ms: int = 0
        self._debug_log_enabled: bool = True
        self._debug_log_path: Path = Path.cwd() / "zplotpy_denoise_debug.log"
        self._debug_last_line: str = ""
        self.waveform_selections: List[Dict[str, float]] = []
        self.waveop_stack_result: Optional[Dict[str, np.ndarray]] = None
        self._waveop_corrected_ttrue: Dict[Tuple[int, int], float] = {}
        # 姿态联合校正仅在 processors.relocation 嵌入本查看器时启用
        self._relocation_host_mode = False
        # 位置 Map / 共享地形仍用下列字段；姿态预览状态默认关闭
        self._orientation_ui_params: Dict[str, float] = {}
        self._orientation_current_solution: Dict[str, float] = {}
        self._orientation_last_applied_solution: Dict[str, float] = {}
        self._denoise_backend_stage: str = "未执行"
        self._denoise_last_applied_count: int = 0
        self._denoise_last_delta_mean_abs: float = 0.0
        self._denoise_last_delta_max_abs: float = 0.0
        self._denoise_frozen_delta_mean_abs: float = 0.0
        self._denoise_frozen_delta_max_abs: float = 0.0
        self._denoise_frozen_ready: bool = False
        self._denoise_frozen_trace_set: set[int] = set()
        self._denoise_frozen_by_trace: Dict[int, np.ndarray] = {}
        self._denoise_frozen_original_by_trace: Dict[int, np.ndarray] = {}
        self._denoise_cache_entries: "OrderedDict[Tuple[object, ...], Dict[str, object]]" = OrderedDict()
        self._denoise_cache_limit: int = 4
        self._denoise_run_armed: bool = False
        self._denoise_progress_phase: str = ""
        self._coh_gate_kernel_cache: Dict[int, np.ndarray] = {}
        self._coh_gate_lags_cache: Dict[int, np.ndarray] = {}
        self._coh_gate_smooth_kernel: np.ndarray = np.ones((5,), dtype=np.float64) / 5.0
        self._coh_thread_local = threading.local()
        self._denoise_selected_traces: set[int] = set()
        self._denoise_select_drag_active: bool = False
        self._denoise_select_drag_start_x: Optional[float] = None
        self._denoise_select_drag_last_x: Optional[float] = None
        self._denoise_select_drag_mode: str = "add"  # add/remove/replace
        self._denoise_select_drag_just_finished: bool = False
        self._denoise_click_pending_trace_idx: Optional[int] = None
        self._denoise_click_pending_remove_only: bool = False
        self._denoise_params: Dict[str, object] = {
            "enabled": bool(getattr(self.params, "denoise_enabled", 0)),
            "ab_raw": bool(getattr(self.params, "denoise_ab_raw", 1)),
            "show_diff": False,
            "diff_gain": 1.0,
            "scope": str(getattr(self.params, "denoise_scope", "rendered")),
            "f_s": float(getattr(self.params, "denoise_f_s", 3.0)),
            "f_e": float(getattr(self.params, "denoise_f_e", 20.0)),
            "bwconn": int(getattr(self.params, "denoise_bwconn", 8)),
            "strength": float(getattr(self.params, "denoise_strength", 3.0)),
            "workers": int(getattr(self.params, "denoise_workers", 1)),
            "coh_win": int(getattr(self.params, "denoise_coh_win", 11)),
            "coh_lag": int(getattr(self.params, "denoise_coh_lag", 2)),
            "coh_thr": float(getattr(self.params, "denoise_coh_thr", 0.55)),
            "coh_blend": float(getattr(self.params, "denoise_coh_blend", 0.35)),
            "coh_penalty": float(getattr(self.params, "denoise_coh_penalty", 0.08)),
            "perf_diag": bool(getattr(self.params, "denoise_perf_diag", 0)),
            "morph_enable": bool(getattr(self.params, "denoise_morph_enable", 1)),
            "morph_preset": str(getattr(self.params, "denoise_morph_preset", "balanced")),
            "morph_quantile": float(getattr(self.params, "denoise_morph_quantile", 0.70)),
            "morph_min_area": int(getattr(self.params, "denoise_morph_min_area", 24)),
            "morph_expand": int(getattr(self.params, "denoise_morph_expand", 1)),
            "morph_floor_ratio": float(getattr(self.params, "denoise_morph_floor_ratio", 0.03)),
            "morph_keep_strong_q": float(getattr(self.params, "denoise_morph_keep_strong_q", 0.95)),
            "pick_guidance": bool(getattr(self.params, "denoise_pick_guidance", 0)),
            "pick_wavelet_length_sec": float(getattr(self.params, "denoise_pick_wavelet_length", 0.19)),
            "pick_guidance_floor": float(getattr(self.params, "denoise_pick_floor", 0.12)),
            "return_debug": False,
            "return_result": False,
        }
        self._orientation_terrain_path: str = ""
        self._orientation_terrain_meta_raw: Optional[Dict[str, object]] = None
        self._orientation_terrain_meta_utm: Optional[Dict[str, object]] = None
        # 工程主窗可注入：打开位置 Map / 姿态时解析最新地形路径
        self._shared_terrain_path_provider: Optional[Callable[[], str]] = None
        # 独立窗口默认允许 Q 退出；嵌入工程主窗时应关掉（退出走工程菜单）
        self._allow_shortcut_quit: bool = True
        self._orientation_current_solution: Dict[str, float] = {
            "azimuth_deg": 0.0,
            "tilt_deg": 0.0,
            "dx": 0.0,
            "dy": 0.0,
            "dz": 0.0,
            "prior_tt_shift_sec": 0.0,
            "tt_corr_sec": 0.0,
            "time_shift_sec": 0.0,
            "objective": float("nan"),
            "accepted": 0.0,
        }
        self._orientation_preview_enabled: bool = False
        self._orientation_preview_solution: Dict[str, float] = {}
        self._orientation_preview_cache: Optional[Dict[str, object]] = None
        self._orientation_last_applied_solution: Dict[str, float] = {}
        # 工程主窗可注册：保存姿态解到工区 JSON（不写 .z/.hdr）
        self._orientation_solution_persist_cb: Optional[Callable[[Dict[str, float]], None]] = None
        # 工程主窗可注册：姿态对话框参数（含预置走时 shift）写回工区
        self._orientation_ui_persist_cb: Optional[Callable[[Dict[str, float]], None]] = None
        # 几何角色：默认本工区 obs（sx=OBS, gx/rx=炮），与 RTM --geom obs 一致
        self._orientation_geom: str = "obs"
        self._floating_dialogs: List[QtWidgets.QDialog] = []
        self._param_help_texts: Dict[str, str] = {
            "open_z": "选择并加载 .z 数据文件。",
            "open_hdr": "选择 .hdr 头文件（可选，优先拾取信息来源）。",
            "open_r": "选择 .r 记录文件（可选）。",
            "reload": "按当前参数立即重绘。",
            "save_params": "（已取消）参数随「保存工区」自动写入 outputs/viewer_params.json。",
            "load_params": "（已取消）打开工区并加载数据时自动应用 outputs/viewer_params.json。",
            "save_z": "保存当前数据为 .z（可写入当前拾取）。",
            "data_info": "打开「数据信息」窗：页签「数据概览 / 道头参数」。",
            "location_map": "打开位置Map，查看震源点与接收点空间分布。",
            "export_fig": "将当前绘图区导出为图片。",
            "prev_rec": "切到上一记录号（shot）。",
            "next_rec": "切到下一记录号（shot）。",
            "theory": "计算并叠加理论走时。",
            "clear_theory": "清除理论走时叠加。",
            "water_corr": "计算并叠加水层校正后的理论走时。",
            "clear_water": "清除水层校正叠加。",
            "water_curve": "显示水层校正的距离-校正量曲线。",
            "load_txin": "读取 tx.in 并叠加走时曲线。",
            "clear_txin": "清除 tx.in 走时叠加。",
            "preview_map_txin": "预览 tx.in 映射结果（仅高亮将新增的拾取点，不写入数据）。",
            "map_txin": "将 tx.in 走时点一键映射为当前剖面拾取（已有拾取不覆盖）。",
            "map_txin_apick_only": "仅映射当前 apick；关闭时映射 tx.in 中全部拾取字。",
            "map_txin_tol": "偏移距匹配容差百分比（相对道间距中位数）。",
            "map_txin_view_only": "仅映射当前主图视窗内可见的 tx 点（x/t）。",
            "theme": "切换界面主题（默认与多种内置配色，仅改变颜色不改尺寸）。",
            "about": "显示版本与迁移状态。",
            "toggle_panels": "显示/隐藏参数面板，给绘图区更多空间。",
            "irec": "记录号（炮集号），0 表示全部记录。",
            "itype": "分量过滤：垂直/径向/横向/水听器。",
            "nskip": "抽道间隔，增大可提升速度。",
            "ndecim": "时间采样抽取间隔，增大可提升速度。",
            "vred": "折合速度（km/s），>0 时按 t'=t-|x|/vred 显示。",
            "xmin": "显示/过滤 X 最小值（km）。",
            "xmax": "显示/过滤 X 最大值（km）。",
            "tmin": "显示时间最小值（s）。",
            "tmax": "显示时间最大值（s）。",
            "mode": "显示模式：Wiggle / 正填充 / 负填充 / Density（变密度，适合多道总览）。",
            "rt_shade": "交互时是否渲染填充。关闭可显著提升拖拽流畅度。",
            "amp": "振幅缩放系数。",
            "iscale": "增益模式：0自动，1固定，2变增益。",
            "rcor": "距离校正指数。",
            "sf": "固定缩放因子（主要在 iscale=1 下生效）。",
            "tvg": "变增益窗口长度参数。",
            "pvg": "变增益幂指数参数。",
            "clip": "振幅裁剪阈值（0=不裁剪）。",
            "dscale": "显示标度校准因子（用于匹配 Fortran 视觉振幅）。",
            "gain_preset_balanced": "应用平衡显示预设（默认浏览推荐）。",
            "far_offset_boost": "一键增强远偏移弱能量可见性。",
            "gain_preset_strong": "应用强增强预设（弱信号优先，噪声容忍更高）。",
            "filter_on": "带通滤波开关。",
            "gain_on": "增益计算开关：关闭后将跳过增益处理。",
            "rmean": "去平均（滤波前）。默认带通 fL>0 已去直流，勾选外观常不变；关滤波可对比。",
            "rtrend": "去线性趋势（滤波前）。带通开启时外观常不变；关滤波可看基线漂移。",
            "freqlo": "带通低截止频率（Hz）。",
            "freqhi": "带通高截止频率（Hz）。",
            "npoles": "滤波器阶数。",
            "izerop": "零相位滤波开关。",
            "denoise_enabled": "去噪开关：开启后在当前显示链最终波形上执行去噪（包含已启用的 mute/滤波/增益 等处理结果）。",
            "denoise_ab_raw": "A/B 对比：勾选时显示去噪道(B)；取消勾选时显示原始道(A)。",
            "denoise_show_diff": "差值显示：勾选后显示(去噪后-去噪前)残差信号，便于确认去噪实际改变量。",
            "denoise_diff_gain": "差值放大倍数：仅在差值显示下生效，用于放大残差信号便于观察。",
            "denoise_start": "开始去噪：在选道与参数确认后触发实际去噪执行。",
            "denoise_scope": "去噪范围：当前渲染道 / 当前视窗道 / 当前记录道（用于控制哪些道在当前显示链结果上执行去噪）。",
            "denoise_select_mode": "手动选道模式：开启后左键点选道切换选中状态（用于selected范围）。",
            "denoise_clear_selected": "清空已手动选中的道。",
            "denoise_coh_cfg": "相干参数窗口：设置 semblance 门控与回混参数（cw/cl/ct/cb/cp）。其中 cw/cl 单位为样点，ct/cb/cp 为无量纲。",
            "denoise_f_s": "去噪频带下限 f_s（Hz）。",
            "denoise_f_e": "去噪频带上限 f_e（Hz）。",
            "denoise_strength": "去噪强度（GCV 收缩强度，trace 路径）。",
            "denoise_bwconn": "去噪邻域连通性（4 或 8）。",
            "denoise_workers": "并行参数预留：当前版本以 trace 级路径为主，workers 用于后续扩展。",
            "pick_mode": "拾取模式：左键加点/改点，右键删点。",
            "apick": "活动拾取字编号（仅影响当前编辑/自动拾取目标）。",
            "pick_size": "拾取点圆圈大小（像素）。",
            "tcrcor": "插值相关窗口长度（秒）。",
            "tlag": "插值相关搜索半窗（秒）。",
            "hilbratio": "Hilbert 相位权重因子。",
            "interp_force": "强制拾取：即使相关性不佳也给点（不建议默认开启）。",
            "auto_pick": "对当前可见/过滤道执行自动拾取。",
            "interp_pick": "在两个种子拾取间执行插值相关拾取。",
            "save_picks": "保存拾取为 zplot.out 格式。",
            "undo_pick": "撤销上一次会改变拾取走时/集合的操作。",
            "redo_pick": "重做上一次撤销操作（撤销的反操作）。",
            "save_hdr": "将当前拾取写入头文件（.hdr）。写完后可用旁侧「写入tx.in」生成 RAYINVR 输入。",
            "write_txin": "在写入 HDR 之后：从 HDR 转换写出 tx.in（可配置各 OBS 的 xmod/tshift；有 .rec 时预填 xmod）。",
            "export_tx": "同「写入tx.in」（旧名）。",
            "clear_picks": "清空当前拾取数据。",
            "align_pick": "波形临时对齐：按当前拾取字把波形平移到共同参考时刻。",
            "align_adaptive": "拾取自适应更新：迭代估计并更新拾取时间（不平移波形）。",
            "eval_stack": "显示自适应拾取更新评价图表。",
            "static_corr": "根据当前拾取字计算短波长静校正。",
            "clear_static": "清除静校正并恢复未校正显示。",
            "clear_align": "清除所有对齐偏移。",
            "show_stack": "显示叠加道：当前活动字拾取中心 ±0.5s 窗口叠加（中心右侧）。",
            "waveop_stack": "按当前拾取字下的 V 段做自适应更新并叠加（不写回拾取）。",
            "waveop_att": "（仅在 relocation 工区嵌入时显示）内嵌姿态联合校正。",
            "waveop_clear": "清除当前 apick 的全部 V 段与高亮（其它字保留；Shift+V 只删当前字最近一段）。",
            "waveop_save": "保存 V 段与叠加校正基准为 .waveop.json（与显示参数文件分开）。",
            "waveop_load": "加载 V 段与叠加校正基准（.waveop.json）。",
        }
        self._param_help_texts_detailed: Dict[str, str] = {
            "nskip": "抽道策略在大文件下非常关键：先大 nskip 浏览，再减小做精细处理。",
            "ndecim": "ndecim 与 nskip 会共同影响渲染点数；交互时建议先提高 ndecim。",
            "vred": "折合速度会把同一速度事件拉平，便于相位连续性判断；设为0表示关闭折合。",
            "xmin": "用于限制绘图区偏移范围并减少渲染负担。",
            "xmax": "与 xmin 配对使用；若 xmin>=xmax 则该窗口无效。",
            "tmin": "时间窗可用于聚焦目标层位并减少视觉干扰。",
            "tmax": "建议与 tmin 成对设置；若 tmin>=tmax 则该窗口无效。",
            "mode": "Wiggle/填充适合精细拾取；Density 用灰度图显示整剖面，道多时更流畅。",
            "rt_shade": "关闭“交互时填充”后，平移缩放仅画主波形；停止交互后再补齐填充。Density 模式不受此项影响。",
            "iscale": "变增益(2)适合弱信号增强，但对异常噪声也更敏感，建议配合滤波与裁剪使用。",
            "gain_on": "增益总开关：取消勾选后跳过增益计算，仅保留原始幅度与滤波处理。",
            "rmean": "处理顺序：rmean → rtrend → 带通 → 增益。带通开启时直流已被滤除，勿期望波形大变。",
            "rtrend": "长记录、未滤波时去趋势最明显；与带通同时开时视觉差异通常很小。",
            "sf": "Fortran 里 sf 主要用于 iscale=1 固定比例模式；iscale=2 时影响很弱。",
            "gain_preset_balanced": "平衡远近偏移可见性，适合日常浏览与初筛。",
            "far_offset_boost": "将切换到偏向远偏移显示的参数组合（iscale=1, rcor↑, amp↑），可在此基础上微调。",
            "gain_preset_strong": "最大化弱能量可见性，可能同时放大噪声，建议配合滤波与裁剪。",
            "clip": "裁剪可抑制尖峰遮挡；常见有效区间约 1.5~4.0。",
            "dscale": "仅影响显示宽度，不改变数据本身；可用来把 Qt 观感标定到 Fortran。",
            "filter_on": "带通开关。缩放/平移时仍保持滤波（不再为流畅而临时关掉）。",
            "denoise_enabled": "去噪针对当前显示链结果执行；无需先启用 mute，若 mute 已开启则其效果会一并参与去噪输入。",
            "denoise_ab_raw": "A/B 快切：勾选=去噪(B)，取消勾选=原始(A)。建议在同一视窗下反复切换对比。",
            "denoise_show_diff": "差值模式建议与视窗道或手动选道结合使用，可快速确认哪些道被实际改写。",
            "denoise_diff_gain": "差值放大仅影响显示，不改变实际去噪结果；建议从 x2/x5 开始。",
            "denoise_start": "点击后进入执行态；若你调整了范围或参数，建议再次点击开始去噪。",
            "denoise_scope": "范围用于指定哪些道参与去噪计算；当前 trace 级实现下建议优先使用“当前渲染道”保证实时性。",
            "denoise_select_mode": "建议在固定视窗下点选目标道；selected 范围可避免对整屏道执行去噪。",
            "denoise_clear_selected": "切换记录前可先清空，避免跨记录遗留选择造成误判。",
            "denoise_coh_cfg": "cw(窗长, 样点): 时间向局部统计窗口，越大越稳但细节更平滑；cl(lag, 样点): 邻道时移搜索范围；ct(阈值, 无量纲): 门控阈值，越高越保守；cb(回混, 无量纲): 高相干区原始信号回混上限，越大越保真；cp(惩罚, 无量纲): lag 轨迹变化惩罚，越大越连续。",
            "denoise_f_s": "需满足 f_s>=0 且 f_s<f_e；建议先与带通频段保持一致再细调。",
            "denoise_f_e": "需满足 f_e>0 且 f_s<f_e；过低会压制有效信号，过高会降低抑噪收益。",
            "denoise_strength": "strength>0；值越大收缩越强，弱信号也更易被抑制，建议 0.8~2.0 起步。",
            "denoise_bwconn": "当前参数已在 denoise 校验中启用，但核心形态学连接策略仍为后续阶段预留。",
            "denoise_workers": "workers 在 section 入口已保留接口；当前实现暂未并行计算，仅用于未来兼容。",
            "align_adaptive": "该功能会直接修改当前拾取字的时间值；若要恢复请用撤销或重新载入拾取。",
            "pick_size": "圆圈太大可能遮挡细节，建议在 5~10 之间按屏幕分辨率调整。",
            "tcrcor": "窗口越大越稳但更平滑，建议 0.4~1.0 秒按数据频带调整。",
            "tlag": "搜索范围越大越鲁棒但更慢，建议 0.05~0.20 秒。",
            "hilbratio": "增大可强化相位一致性约束，过大可能抑制幅值信息。",
            "interp_force": "关闭后只保留一致性更高的结果，可避免退化成单纯线性连线。",
            "eval_stack": "叠加评价会基于最近一次 F（自适应拾取更新）结果生成统计图。",
            "static_corr": "静校正通过空间平滑提取短波长残差，校正后可提升同相轴连续性。",
            "show_stack": "仅叠加当前活动字的拾取窗（pick±0.5s），并固定显示在当前视图中心右侧，便于持续对比。",
            "save_hdr": "写入 HDR 会覆盖原头文件 picks 字段，建议先备份。随后可用「写入tx.in」。",
            "write_txin": "流程：写入HDR → 写入tx.in。转换对话框可为各 OBS 填 xmod；若已加载 .rec/.rsp 会按炮号预填。",
            "export_tx": "同 write_txin。",
            "save_params": "建议为不同数据集维护独立参数模板，便于复现实验。",
            "load_params": "加载参数后会自动触发重绘；不兼容字段会被忽略。",
            "save_z": "可将当前拾取写回 .z 的 pick 字段，便于与旧流程/程序交换。",
            "theory": "理论走时基于 RAYINVR，建议先确认 v.in 模型与当前数据坐标系一致。",
            "water_corr": "水层校正依赖理论走时与射线，建议在理论走时计算成功后再执行。",
            "water_curve": "如果曲线出现剧烈跳变，通常意味着射线覆盖不足或模型/数据坐标不一致。",
            "load_txin": "导入 tx.in（读入叠加）在「走时模板」面板；写出 tx.in 请用工具栏「写入tx.in」。",
            "preview_map_txin": "先预览即将写入的点位，再决定是否执行映射，可降低批量误操作风险。",
            "map_txin": "映射按偏移距最近道匹配；若该道该拾取字已有拾取则保留不改。",
            "map_txin_apick_only": "建议在单一震相字精修阶段开启，可避免跨拾取字误映射。",
            "map_txin_tol": "容差过小会漏映射，过大会跨道误匹配；建议 50%~120%。",
            "map_txin_view_only": "开启后将先按当前视窗过滤 tx 点，适合局部精修时避免整段批量改动。",
            "theme": "主题切换只改颜色配置，控件字体与尺寸保持一致；默认主题下参数文字使用黑色。",
            "data_info": "原「Data Info」与「道头参数」已合并为此按钮；打开后用页签切换概览与道头表。",
            "waveop_stack": "V 段列表不在参数条内：工区底栏「V段」页签，或独立启动时剖面下方薄条。",
            "waveop_att": "（仅姿态工区嵌入时显示）姿态联合校正入口。",
            "waveop_clear": "只清当前 apick；底栏/薄条中的 V 段列表同步刷新。",
            "waveop_save": "仅保存选波与叠加基准，不混入「保存参数」JSON。",
            "waveop_load": "加载前若已有 V 段会提示是否覆盖。",
        }

        self._init_debug_log_file()
        self._build_ui()
        self._wire_events()
        QtCore.QTimer.singleShot(0, self._update_action_bar_overflow)
        QtCore.QTimer.singleShot(0, lambda: self._apply_optional_theme("default"))

    def showEvent(self, event):
        super().showEvent(event)
        # 首次显示后尺寸才稳定；只预约一次，避免连续改高度闪烁
        if not bool(getattr(self, "_param_heights_unified_on_show", False)):
            self._param_heights_unified_on_show = True
            self._schedule_unify_param_group_heights(30)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_action_bar_overflow()
        # 宽度变化后部分面板内容换行，延迟重统一高度
        try:
            w = int(event.size().width())
            old = int(getattr(self, "_last_unify_width", -1))
            if abs(w - old) >= 40:
                self._last_unify_width = w
                self._schedule_unify_param_group_heights(30)
        except Exception:
            pass

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """窗口关闭前持久化面板布局。"""
        try:
            self._save_panel_layout_state()
        except Exception:
            pass
        super().closeEvent(event)


def main() -> int:
    # Matplotlib：对 FigureCanvas 使用 draw_idle() 时，实际 draw 发生在下一轮 Qt 事件循环里，
    # 局部的 warnings.catch_warnings() 无法覆盖异步绘制；去噪对比等图窗在 savefig(..., bbox_inches="tight")
    # 或重绘时可能触发「Axes 与 tight_layout 不兼容」的 UserWarning（栈顶常落在 app.exec）。
    # 该提示在此场景下可安全忽略，故在入口统一过滤，避免刷屏。
    warnings.filterwarnings(
        "ignore",
        message=r"This figure includes Axes that are not compatible with tight_layout.*",
        category=UserWarning,
    )
    # 高 DPI 提升清晰度
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    existing = QtWidgets.QApplication.instance()
    created_here = existing is None
    app = QtWidgets.QApplication(sys.argv) if created_here else existing
    pg.setConfigOptions(antialias=False, useOpenGL=True)
    win = QtFastViewer()
    win.show()
    # 若外部事件循环已在运行（如嵌入式环境），避免再次 exec 触发警告
    if created_here:
        return app.exec()
    return 0


# 兼容主入口命名：Qt 版即新的 ZPlotGUI
ZPlotGUI = QtFastViewer


if __name__ == "__main__":
    raise SystemExit(main())

