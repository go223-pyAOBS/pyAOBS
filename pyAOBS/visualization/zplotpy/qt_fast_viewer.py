"""
qt_fast_viewer.py - PySide6/PyQtGraph fast seismic viewer prototype.

目标：
- 给大文件浏览提供一个更接近原生桌面渲染性能的最小原型
- 复用现有 DataLoader / DataProcessor / ZPlotParameters
- 先聚焦“快”：加载、平移、缩放、窗口内快速重绘
"""

from __future__ import annotations

import os
import sys
import time
import json
import tempfile
import math
import colorsys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


def _tf_abs_amp(tf: object) -> np.ndarray:
    """|TF| amplitude as float64; avoids ComplexWarning if backend returns complex."""
    return np.abs(np.asarray(tf)).astype(np.float64, copy=False)


try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as exc:
    raise RuntimeError(
        "未安装 PySide6，请先安装：pip install PySide6 pyqtgraph"
    ) from exc

try:
    import pyqtgraph as pg
    import pyqtgraph.exporters as pg_exporters
except Exception as exc:
    raise RuntimeError(
        "未安装 pyqtgraph，请先安装：pip install pyqtgraph"
    ) from exc

try:
    from .data_loader import DataLoader
    from .data_processor import DataProcessor
    from .adaptive_stack import AdaptiveStacker
    from .parameters import ZPlotParameters
    from .pick_manager import PickManager
    from .hdr_to_tx import Z2TxConfig, convert_hdr_to_tx
    from .static_correction import StaticCorrector
    from .stacking_evaluator import StackingEvaluator
    from .theoretical_traveltime import TheoreticalTravelTimeCalculator
    from .auto_picker import AutoPicker
    from .interpolation_correlation_picker import InterpolationCorrelationPicker
    from .src_kernel_bridge import SrcShadeKernelBridge
    from ...processors.denoise import denoise_trace, denoise_section
    from ...processors.denoise.metrics import ab_trace_metrics, summarize_metrics
    from ...processors.denoise.ssq_backend import HAVE_SSQ as DENOISE_HAVE_SSQ
    from ...processors.relocation import (
        OrientationCorrectionInput,
        OrientationObservation,
        run_orientation_correction,
        build_bathymetry_sampler,
    )
    from ...processors.relocation.polarization_features import extract_polarization_features
    from .denoise_coh_hints import (
        COHERENCE_DIP_SLOPE_PRESET_ITEMS,
        CUSTOM_DIP_SLOPE_COMBO_MARKER,
        estimate_moveout_slope_s_per_km_from_picks,
        suggest_coherence_cb_cl_start,
    )
except ImportError:
    from data_loader import DataLoader
    from data_processor import DataProcessor
    from adaptive_stack import AdaptiveStacker
    from parameters import ZPlotParameters
    from pick_manager import PickManager
    from hdr_to_tx import Z2TxConfig, convert_hdr_to_tx
    from static_correction import StaticCorrector
    from stacking_evaluator import StackingEvaluator
    from theoretical_traveltime import TheoreticalTravelTimeCalculator
    from auto_picker import AutoPicker
    from interpolation_correlation_picker import InterpolationCorrelationPicker
    from src_kernel_bridge import SrcShadeKernelBridge
    from pyAOBS.processors.denoise import denoise_trace, denoise_section
    from pyAOBS.processors.denoise.metrics import ab_trace_metrics, summarize_metrics
    from pyAOBS.processors.denoise.ssq_backend import HAVE_SSQ as DENOISE_HAVE_SSQ
    from pyAOBS.processors.relocation import (
        OrientationCorrectionInput,
        OrientationObservation,
        run_orientation_correction,
        build_bathymetry_sampler,
    )
    from pyAOBS.processors.relocation.polarization_features import extract_polarization_features
    from pyAOBS.visualization.zplotpy.denoise_coh_hints import (
        COHERENCE_DIP_SLOPE_PRESET_ITEMS,
        CUSTOM_DIP_SLOPE_COMBO_MARKER,
        estimate_moveout_slope_s_per_km_from_picks,
        suggest_coherence_cb_cl_start,
    )

class QtFastViewer(QtWidgets.QMainWindow):
    """PySide6 + PyQtGraph 大文件快速浏览原型。"""
    _THEME_PRESETS: Dict[str, Dict[str, str]] = {
        "default": {
            "name": "默认主题",
            "window_bg": "#f2f4f8",
            "surface_bg": "#ffffff",
            "panel_bg": "#ffffff",
            "text": "#1f2937",
            "label_text": "#000000",
            "border": "#c7ccd5",
            "accent": "#2563eb",
            "accent_text": "#ffffff",
            "disabled_bg": "#edf0f4",
            "disabled_text": "#9aa1ab",
            "plot_bg": "#ffffff",
            "plot_axis": "#1f2937",
            "plot_grid_alpha": "0.12",
            "status_text": "#1f2937",
            "hint_text": "#5f6773",
            "wave_pen": "#0a0a0a",
            "shade_pen": "#2f3338",
            "pick_edge": "#505050",
            "pick_active_edge": "#ffffff",
        },
        "light": {
            "name": "浅蓝",
            "window_bg": "#eaf2fb",
            "surface_bg": "#ffffff",
            "panel_bg": "#f6faff",
            "text": "#1e293b",
            "label_text": "#1e293b",
            "border": "#b7c8de",
            "accent": "#2b6cb0",
            "accent_text": "#ffffff",
            "disabled_bg": "#e7eef7",
            "disabled_text": "#92a1b2",
            "plot_bg": "#ffffff",
            "plot_axis": "#1e293b",
            "plot_grid_alpha": "0.12",
            "status_text": "#1e293b",
            "hint_text": "#617489",
            "wave_pen": "#0f172a",
            "shade_pen": "#334155",
            "pick_edge": "#475569",
            "pick_active_edge": "#ffffff",
        },
        "dark": {
            "name": "深青",
            "window_bg": "#1f252c",
            "surface_bg": "#2a313a",
            "panel_bg": "#2f3944",
            "text": "#e5edf5",
            "label_text": "#e5edf5",
            "border": "#4b5b6b",
            "accent": "#2ea6a6",
            "accent_text": "#062b2b",
            "disabled_bg": "#3a444f",
            "disabled_text": "#8695a5",
            "plot_bg": "#1c232b",
            "plot_axis": "#dbe5ef",
            "plot_grid_alpha": "0.18",
            "status_text": "#e5edf5",
            "hint_text": "#b7c6d6",
            "wave_pen": "#e6edf5",
            "shade_pen": "#9ab2c8",
            "pick_edge": "#d5deea",
            "pick_active_edge": "#ffffff",
        },
        "solarized_light": {
            "name": "Solarized 浅色",
            "window_bg": "#fdf6e3",
            "surface_bg": "#fffdf6",
            "panel_bg": "#f8f1dd",
            "text": "#586e75",
            "label_text": "#586e75",
            "border": "#d6c7a1",
            "accent": "#268bd2",
            "accent_text": "#ffffff",
            "disabled_bg": "#efe6cf",
            "disabled_text": "#9ea99a",
            "plot_bg": "#fffdf6",
            "plot_axis": "#586e75",
            "plot_grid_alpha": "0.12",
            "status_text": "#586e75",
            "hint_text": "#6a7f86",
            "wave_pen": "#35484f",
            "shade_pen": "#5d6f74",
            "pick_edge": "#5f7378",
            "pick_active_edge": "#ffffff",
        },
        "solarized_dark": {
            "name": "Solarized 深色",
            "window_bg": "#002b36",
            "surface_bg": "#073642",
            "panel_bg": "#0a3f4d",
            "text": "#93a1a1",
            "label_text": "#93a1a1",
            "border": "#365864",
            "accent": "#b58900",
            "accent_text": "#1f1a00",
            "disabled_bg": "#274853",
            "disabled_text": "#6d8487",
            "plot_bg": "#032b36",
            "plot_axis": "#93a1a1",
            "plot_grid_alpha": "0.2",
            "status_text": "#93a1a1",
            "hint_text": "#8ca7a7",
            "wave_pen": "#d2dddd",
            "shade_pen": "#8fb0b1",
            "pick_edge": "#c7d1d1",
            "pick_active_edge": "#f8fcfc",
        },
        "nord": {
            "name": "Nord",
            "window_bg": "#2e3440",
            "surface_bg": "#3b4252",
            "panel_bg": "#434c5e",
            "text": "#eceff4",
            "label_text": "#eceff4",
            "border": "#5b657a",
            "accent": "#88c0d0",
            "accent_text": "#1f252e",
            "disabled_bg": "#4c566a",
            "disabled_text": "#9da7ba",
            "plot_bg": "#2b313d",
            "plot_axis": "#e5e9f0",
            "plot_grid_alpha": "0.18",
            "status_text": "#eceff4",
            "hint_text": "#c2cad7",
            "wave_pen": "#eceff4",
            "shade_pen": "#a7b2c4",
            "pick_edge": "#d7deea",
            "pick_active_edge": "#ffffff",
        },
        "graphite": {
            "name": "石墨灰",
            "window_bg": "#26282d",
            "surface_bg": "#30343b",
            "panel_bg": "#383c45",
            "text": "#eceff3",
            "label_text": "#eceff3",
            "border": "#596170",
            "accent": "#7aa2f7",
            "accent_text": "#111827",
            "disabled_bg": "#434955",
            "disabled_text": "#96a0b2",
            "plot_bg": "#262b33",
            "plot_axis": "#eceff3",
            "plot_grid_alpha": "0.18",
            "status_text": "#eceff3",
            "hint_text": "#bcc4d3",
            "wave_pen": "#edf1f7",
            "shade_pen": "#aeb7c9",
            "pick_edge": "#d8deea",
            "pick_active_edge": "#ffffff",
        },
        "forest": {
            "name": "森林绿",
            "window_bg": "#eef6f1",
            "surface_bg": "#ffffff",
            "panel_bg": "#f3fbf6",
            "text": "#1f3d2d",
            "label_text": "#1f3d2d",
            "border": "#b4d0c0",
            "accent": "#2f855a",
            "accent_text": "#ffffff",
            "disabled_bg": "#e6f1ea",
            "disabled_text": "#8aa294",
            "plot_bg": "#ffffff",
            "plot_axis": "#1f3d2d",
            "plot_grid_alpha": "0.12",
            "status_text": "#1f3d2d",
            "hint_text": "#557565",
            "wave_pen": "#173826",
            "shade_pen": "#395f4c",
            "pick_edge": "#365646",
            "pick_active_edge": "#ffffff",
        },
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZPLOT Qt Fast Viewer")
        self.resize(1500, 900)
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
        self._pick_item: Optional[pg.ScatterPlotItem] = None
        self._stack_item: Optional[pg.PlotDataItem] = None
        self._static_preview_item: Optional[pg.PlotDataItem] = None
        self._theoretical_item: Optional[pg.PlotDataItem] = None
        self._txin_item: Optional[pg.ScatterPlotItem] = None
        self._txin_map_preview_item: Optional[pg.ScatterPlotItem] = None
        self._water_corr_item: Optional[pg.PlotDataItem] = None
        self._wave_select_item: Optional[pg.PlotDataItem] = None
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
        self._orientation_ui_params: Dict[str, float] = {
            "wave_pre": 0.30,
            "wave_post": 0.70,
            "att_iter": 4.0,
            "att_wtt": 1.0,
            "att_wpol": 1.0,
            "att_wsym": 0.8,
        }
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
        self._orientation_current_solution: Dict[str, float] = {
            "azimuth_deg": 0.0,
            "tilt_deg": 0.0,
            "dx": 0.0,
            "dy": 0.0,
            "dz": 0.0,
            "time_shift_sec": 0.0,
            "objective": float("nan"),
            "accepted": 0.0,
        }
        self._orientation_preview_enabled: bool = False
        self._orientation_preview_solution: Dict[str, float] = {}
        self._orientation_preview_cache: Optional[Dict[str, object]] = None
        self._floating_dialogs: List[QtWidgets.QDialog] = []
        self._param_help_texts: Dict[str, str] = {
            "open_z": "选择并加载 .z 数据文件。",
            "open_hdr": "选择 .hdr 头文件（可选，优先拾取信息来源）。",
            "open_r": "选择 .r 记录文件（可选）。",
            "reload": "按当前参数立即重绘。",
            "save_params": "将当前参数保存为 JSON。",
            "load_params": "从 JSON 加载参数并应用。",
            "save_z": "保存当前数据为 .z（可写入当前拾取）。",
            "data_info": "显示当前数据规模与文件信息。",
            "coord_params": "以列表形式显示 .z/.hdr 中的道头参数统计。",
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
            "help": "显示功能帮助说明。",
            "shortcuts": "显示当前快捷键列表。",
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
            "mode": "显示模式：Wiggle / 正填充 / 负填充。",
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
            "save_hdr": "将当前拾取写入头文件（.hdr）。",
            "export_tx": "将当前拾取导出为 tx.in。",
            "clear_picks": "清空当前拾取数据。",
            "align_pick": "波形临时对齐：按当前拾取字把波形平移到共同参考时刻。",
            "align_adaptive": "拾取自适应更新：迭代估计并更新拾取时间（不平移波形）。",
            "eval_stack": "显示自适应拾取更新评价图表。",
            "static_corr": "根据当前拾取字计算短波长静校正。",
            "clear_static": "清除静校正并恢复未校正显示。",
            "clear_align": "清除所有对齐偏移。",
            "show_stack": "显示叠加道：当前活动字拾取中心 ±0.5s 窗口叠加（中心右侧）。",
            "waveop_stack": "基于 V 键标注点进行窗口选波、自适应对齐并叠加。",
            "waveop_att": "姿态校正：基于 V 段三分量波形+走时进行方位/倾斜/位置联合校正。",
            "waveop_clear": "清除所有 V 键选波窗口和高亮（Shift+V 可删除最近一个）。",
        }
        self._param_help_texts_detailed: Dict[str, str] = {
            "nskip": "抽道策略在大文件下非常关键：先大 nskip 浏览，再减小做精细处理。",
            "ndecim": "ndecim 与 nskip 会共同影响渲染点数；交互时建议先提高 ndecim。",
            "vred": "折合速度会把同一速度事件拉平，便于相位连续性判断；设为0表示关闭折合。",
            "xmin": "用于限制绘图区偏移范围并减少渲染负担。",
            "xmax": "与 xmin 配对使用；若 xmin>=xmax 则该窗口无效。",
            "tmin": "时间窗可用于聚焦目标层位并减少视觉干扰。",
            "tmax": "建议与 tmin 成对设置；若 tmin>=tmax 则该窗口无效。",
            "mode": "填充模式会触发更多线段渲染；交互阶段可先用 Wiggle，静止后再看填充。",
            "rt_shade": "关闭“交互时填充”后，平移缩放仅画主波形；停止交互后再补齐填充。",
            "iscale": "变增益(2)适合弱信号增强，但对异常噪声也更敏感，建议配合滤波与裁剪使用。",
            "gain_on": "增益总开关：取消勾选后跳过增益计算，仅保留原始幅度与滤波处理。",
            "sf": "Fortran 里 sf 主要用于 iscale=1 固定比例模式；iscale=2 时影响很弱。",
            "gain_preset_balanced": "平衡远近偏移可见性，适合日常浏览与初筛。",
            "far_offset_boost": "将切换到偏向远偏移显示的参数组合（iscale=1, rcor↑, amp↑），可在此基础上微调。",
            "gain_preset_strong": "最大化弱能量可见性，可能同时放大噪声，建议配合滤波与裁剪。",
            "clip": "裁剪可抑制尖峰遮挡；常见有效区间约 1.5~4.0。",
            "dscale": "仅影响显示宽度，不改变数据本身；可用来把 Qt 观感标定到 Fortran。",
            "filter_on": "实时交互阶段处理链会自动降级部分重计算，以保证帧率。",
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
            "save_hdr": "写入 HDR 会覆盖原头文件 picks 字段，建议先备份。",
            "export_tx": "导出 tx.in 前会先将当前拾取写入 HDR，确保转换输入为最新拾取。",
            "save_params": "建议为不同数据集维护独立参数模板，便于复现实验。",
            "load_params": "加载参数后会自动触发重绘；不兼容字段会被忽略。",
            "save_z": "可将当前拾取写回 .z 的 pick 字段，便于与旧流程/程序交换。",
            "theory": "理论走时基于 RAYINVR，建议先确认 v.in 模型与当前数据坐标系一致。",
            "water_corr": "水层校正依赖理论走时与射线，建议在理论走时计算成功后再执行。",
            "water_curve": "如果曲线出现剧烈跳变，通常意味着射线覆盖不足或模型/数据坐标不一致。",
            "load_txin": "导入 tx.in 时会自动将模型距离转换为偏移距，并叠加当前 vred 折合显示时间。",
            "preview_map_txin": "先预览即将写入的点位，再决定是否执行映射，可降低批量误操作风险。",
            "map_txin": "映射按偏移距最近道匹配；若该道该拾取字已有拾取则保留不改。",
            "map_txin_apick_only": "建议在单一震相字精修阶段开启，可避免跨拾取字误映射。",
            "map_txin_tol": "容差过小会漏映射，过大会跨道误匹配；建议 50%~120%。",
            "map_txin_view_only": "开启后将先按当前视窗过滤 tx 点，适合局部精修时避免整段批量改动。",
            "theme": "主题切换只改颜色配置，控件字体与尺寸保持一致；默认主题下参数文字使用黑色。",
        }

        self._init_debug_log_file()
        self._build_ui()
        self._wire_events()
        QtCore.QTimer.singleShot(0, self._update_action_bar_overflow)
        QtCore.QTimer.singleShot(0, lambda: self._apply_optional_theme("default"))

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
        self.btn_reload = QtWidgets.QPushButton("重绘")
        self.btn_save_params = QtWidgets.QPushButton("保存参数")
        self.btn_load_params = QtWidgets.QPushButton("加载参数")
        self.btn_save_z = QtWidgets.QPushButton("保存.z")
        self.btn_data_info = QtWidgets.QPushButton("Data Info")
        self.btn_coord_params = QtWidgets.QPushButton("道头参数")
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
        self.btn_help = QtWidgets.QPushButton("帮助")
        self.btn_shortcuts = QtWidgets.QPushButton("快捷键")
        self.btn_about = QtWidgets.QPushButton("关于")
        self.btn_theme = QtWidgets.QToolButton()
        self.btn_theme.setText("主题")
        self.btn_theme.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_save_picks = QtWidgets.QPushButton("保存")
        self.btn_undo_pick = QtWidgets.QPushButton("撤销(Ctrl+Z)")
        self.btn_redo_pick = QtWidgets.QPushButton("重做(Ctrl+Y)")
        self.btn_save_hdr = QtWidgets.QPushButton("写入HDR")
        self.btn_export_tx = QtWidgets.QPushButton("导出tx.in")
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
        self.list_waveop_segments.setMinimumHeight(52)
        self.list_waveop_segments.setMaximumHeight(66)
        self.list_waveop_segments.setToolTip("V段列表：Shift+V 删除最近一个")
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
        self.btn_toggle_panels = QtWidgets.QPushButton("隐藏面板")
        self.btn_reload.setEnabled(False)
        self.btn_save_params.setEnabled(False)
        self.btn_load_params.setEnabled(True)
        self.btn_save_z.setEnabled(False)
        self.btn_data_info.setEnabled(False)
        self.btn_coord_params.setEnabled(False)
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
        self.btn_export_tx.setEnabled(False)
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
        action_bar.addWidget(self.btn_save_params)
        action_bar.addWidget(self.btn_load_params)
        action_bar.addWidget(self.btn_save_z)
        action_bar.addWidget(self.btn_undo_pick)
        action_bar.addWidget(self.btn_redo_pick)
        action_bar.addWidget(self.btn_save_hdr)
        action_bar.addWidget(self.btn_export_tx)
        action_bar.addWidget(self.btn_data_info)
        action_bar.addWidget(self.btn_coord_params)
        action_bar.addWidget(self.btn_location_map)
        action_bar.addWidget(self.btn_export_fig)

        _add_toolbar_separator()

        # 记录导航
        action_bar.addWidget(self.btn_prev_rec)
        action_bar.addWidget(self.btn_next_rec)

        _add_toolbar_separator()

        # 帮助
        action_bar.addWidget(self.btn_theme)
        action_bar.addWidget(self.btn_help)
        action_bar.addWidget(self.btn_shortcuts)
        action_bar.addWidget(self.btn_about)
        self.btn_more = QtWidgets.QToolButton()
        self.btn_more.setText("更多")
        self.btn_more.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_more.setVisible(False)
        action_bar.addStretch(1)
        action_bar.addWidget(self.btn_toggle_panels)
        top_buttons = [
            self.btn_open, self.btn_open_hdr, self.btn_open_rec, self.btn_reload,
            self.btn_save_params, self.btn_load_params, self.btn_save_z, self.btn_undo_pick, self.btn_redo_pick, self.btn_save_hdr,
            self.btn_export_tx, self.btn_data_info, self.btn_coord_params, self.btn_location_map, self.btn_export_fig, self.btn_prev_rec, self.btn_next_rec,
            self.btn_theme, self.btn_help, self.btn_shortcuts, self.btn_about, self.btn_toggle_panels
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
            self.btn_save_params,
            self.btn_load_params,
            self.btn_save_z,
            self.btn_undo_pick,
            self.btn_redo_pick,
            self.btn_save_hdr,
            self.btn_export_tx,
            self.btn_data_info,
            self.btn_coord_params,
            self.btn_location_map,
            self.btn_export_fig,
            self.btn_prev_rec,
            self.btn_next_rec,
            self.btn_theme,
            self.btn_help,
            self.btn_shortcuts,
            self.btn_about,
            self.btn_toggle_panels,
        ]
        self._disable_toolbar_overflow = True

        # 第二层：横向参数区（参考旧版 zplotpy）
        self.params_panel_scroll = QtWidgets.QScrollArea()
        self.params_panel_scroll.setWidgetResizable(True)
        # 目标：参数区常态下不出现滚动条
        self.params_panel_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # 主题切换后控件尺寸可能变化，纵向滚动条按需出现可避免内容被裁切
        self.params_panel_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._params_panel_fixed_height = 220
        self.params_panel_scroll.setFixedHeight(int(self._params_panel_fixed_height))
        root.addWidget(self.params_panel_scroll, stretch=0)

        params_container = QtWidgets.QWidget()
        params_layout = QtWidgets.QHBoxLayout(params_container)
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.setSpacing(3)
        params_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.params_panel_scroll.setWidget(params_container)
        self._params_container = params_container
        self._params_layout = params_layout
        self._param_groups: List[QtWidgets.QGroupBox] = []
        self._param_group_keys: Dict[QtWidgets.QGroupBox, str] = {}
        self._panel_layout_settings_group = "panel_layout_v1"
        self._panel_drag_candidate: Optional[QtWidgets.QGroupBox] = None
        self._panel_drag_active: bool = False
        self._panel_drag_start_global = QtCore.QPoint()
        self._panel_drag_title_height_px: int = 24
        self._panel_drag_threshold_px: int = 8
        self._panel_drag_hotspot = QtCore.QPoint(16, 10)
        self._panel_drag_ghost: Optional[QtWidgets.QWidget] = None
        self._panel_resize_active: bool = False
        self._panel_resize_group: Optional[QtWidgets.QGroupBox] = None
        self._panel_resize_edge: str = ""
        self._panel_resize_start_x: int = 0
        self._panel_resize_start_width: int = 0
        self._panel_resize_margin_px: int = 5
        self._panel_resize_bounds: Dict[int, Tuple[int, int]] = {}

        # 基础面板（紧凑网格）
        self.group_base = QtWidgets.QGroupBox("基础显示")
        self._param_group_keys[self.group_base] = "base"
        self.group_base.setMinimumWidth(246)
        self.group_base.setMaximumWidth(262)
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
        self.combo_mode.addItems(["Wiggle", "正填充", "负填充"])
        base_grid.addWidget(QtWidgets.QLabel("显示"), 2, 2)
        base_grid.addWidget(self.combo_mode, 2, 3)

        self.chk_rt_shade = QtWidgets.QCheckBox("交互时填充")
        self.chk_rt_shade.setChecked(False)
        self.chk_rt_shade.setToolTip("关闭可提升拖拽/缩放流畅度")
        base_grid.addWidget(self.chk_rt_shade, 5, 0, 1, 2)
        for w in (
            self.spin_irec, self.spin_nskip, self.spin_ndecim, self.spin_vred,
            self.spin_xmin, self.spin_xmax, self.spin_tmin, self.spin_tmax,
            self.combo_mode,
        ):
            try:
                w.setMaximumWidth(88)
            except Exception:
                pass
        try:
            self.combo_itype.setMaximumWidth(96)
        except Exception:
            pass
        base_group_layout = QtWidgets.QVBoxLayout(self.group_base)
        base_group_layout.setContentsMargins(1, 1, 1, 1)
        base_group_layout.addWidget(base_page)
        params_layout.addWidget(self.group_base, stretch=0)
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
        gain_switch_row = QtWidgets.QHBoxLayout()
        gain_switch_row.setContentsMargins(0, 0, 0, 0)
        gain_switch_row.setSpacing(2)
        gain_switch_row.addWidget(self.chk_filter)
        gain_switch_row.addWidget(self.chk_gain)
        gain_switch_row.addWidget(self.chk_zerop)
        gain_switch_row.addStretch(1)
        gain_switch_wrap = QtWidgets.QWidget()
        gain_switch_wrap.setLayout(gain_switch_row)
        gain_grid.addWidget(gain_switch_wrap, 3, 4, 1, 2)
        gain_preset_row = QtWidgets.QHBoxLayout()
        gain_preset_row.setContentsMargins(0, 0, 0, 0)
        gain_preset_row.setSpacing(1)
        gain_preset_row.addWidget(self.btn_gain_preset_balanced)
        gain_preset_row.addWidget(self.btn_far_offset_boost)
        gain_preset_row.addWidget(self.btn_gain_preset_strong)
        gain_preset_row.addStretch(1)
        gain_preset_wrap = QtWidgets.QWidget()
        gain_preset_wrap.setLayout(gain_preset_row)
        gain_grid.addWidget(gain_preset_wrap, 4, 0, 1, 6)
        self.lbl_gain_hint = QtWidgets.QLabel("")
        self.lbl_gain_hint.setStyleSheet("color:#666; font-size:11px;")
        gain_grid.addWidget(self.lbl_gain_hint, 5, 0, 1, 6)
        gain_group_layout = QtWidgets.QVBoxLayout(self.group_gain)
        gain_group_layout.setContentsMargins(1, 1, 1, 1)
        gain_group_layout.addWidget(gain_page)
        self.group_gain.setMinimumWidth(312)
        self.group_gain.setMaximumWidth(344)
        params_layout.addWidget(self.group_gain, stretch=0)
        self._param_groups.append(self.group_gain)

        # 去噪面板（独立预留）
        self.group_denoise = QtWidgets.QGroupBox("去噪")
        self._param_group_keys[self.group_denoise] = "denoise"
        self.group_denoise.setMinimumWidth(186)
        self.group_denoise.setMaximumWidth(204)
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
        denoise_title_row_top.addStretch(1)

        denoise_title_row_bottom = QtWidgets.QHBoxLayout()
        denoise_title_row_bottom.setContentsMargins(0, 0, 0, 0)
        denoise_title_row_bottom.setSpacing(2)
        self.chk_denoise_show_diff = QtWidgets.QCheckBox("差值")
        self.chk_denoise_show_diff.setChecked(bool(self._denoise_params.get("show_diff", False)))
        self.chk_denoise_show_diff.setToolTip(
            "差值：每道 = (去噪后 − 去噪前) × 增益。\n"
            "若沿同相轴出现强条带/双曲能量，多为时频收缩过强（有效信号被大量改写）；"
            "可试：降低强度、提高相干「回混(cb)」、略降阈值(ct)，或 Morph 改为保守。"
        )
        denoise_title_row_bottom.addWidget(self.chk_denoise_show_diff)
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
        denoise_title_row_bottom.addWidget(self.combo_denoise_diff_gain)
        self.btn_denoise_start = QtWidgets.QPushButton("开始")
        self.btn_denoise_start.setMinimumWidth(54)
        self.btn_denoise_start.setMaximumWidth(58)
        denoise_title_row_bottom.addWidget(self.btn_denoise_start)
        denoise_title_row_bottom.addStretch(1)

        denoise_title_layout = QtWidgets.QVBoxLayout()
        denoise_title_layout.setContentsMargins(0, 0, 0, 0)
        denoise_title_layout.setSpacing(1)
        denoise_title_layout.addLayout(denoise_title_row_top)
        denoise_title_layout.addLayout(denoise_title_row_bottom)
        denoise_title_wrap = QtWidgets.QWidget()
        denoise_title_wrap.setLayout(denoise_title_layout)
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
        denoise_grid.addWidget(QtWidgets.QLabel("PG"), 5, 0)
        denoise_grid.addWidget(self.chk_denoise_pick_guidance, 5, 1)
        denoise_grid.addWidget(QtWidgets.QLabel("T(s)"), 5, 2)
        denoise_grid.addWidget(self.spin_denoise_pick_hw, 5, 3)
        denoise_grid.addWidget(QtWidgets.QLabel("外侧"), 6, 0)
        denoise_grid.addWidget(self.spin_denoise_pick_floor, 6, 1, 1, 3)

        for w in (
            self.spin_denoise_f_s, self.spin_denoise_f_e, self.spin_denoise_strength,
            self.combo_denoise_bwconn, self.spin_denoise_workers, self.combo_denoise_scope,
            self.spin_denoise_pick_hw, self.spin_denoise_pick_floor,
        ):
            try:
                w.setMaximumWidth(72)
            except Exception:
                pass

        self.lbl_denoise_hint = QtWidgets.QLabel("")
        self.lbl_denoise_hint.setStyleSheet("color:#666; font-size:11px;")
        self.lbl_denoise_hint.setWordWrap(True)
        denoise_grid.addWidget(self.lbl_denoise_hint, 7, 0, 1, 4)
        self.progress_denoise = QtWidgets.QProgressBar()
        self.progress_denoise.setRange(0, 100)
        self.progress_denoise.setValue(0)
        self.progress_denoise.setFormat("DN %p%")
        self.progress_denoise.setTextVisible(True)
        self.progress_denoise.setFixedHeight(12)
        self.progress_denoise.setVisible(False)
        denoise_grid.addWidget(self.progress_denoise, 8, 0, 1, 4)

        denoise_group_layout = QtWidgets.QVBoxLayout(self.group_denoise)
        denoise_group_layout.setContentsMargins(1, 1, 1, 1)
        denoise_group_layout.addWidget(denoise_page)
        params_layout.addWidget(self.group_denoise, stretch=0)
        self._param_groups.append(self.group_denoise)

        # 拾取面板（紧凑网格）
        self.group_pick = QtWidgets.QGroupBox("拾取")
        self._param_group_keys[self.group_pick] = "pick"
        self.group_pick.setMinimumWidth(220)
        self.group_pick.setMaximumWidth(228)
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

        # 让“参数名-输入框”对齐更紧凑，额外空间统一留给最右侧弹性列
        for c in range(4):
            pick_grid.setColumnStretch(c, 0)

        # 防止控件横向拉伸导致“标签-输入框”距离被动变大
        compact_width_widgets = [
            self.chk_pick_mode,
            self.spin_apick,
            self.spin_pick_size,
            self.spin_tcrcor,
            self.spin_tlag,
            self.spin_hilbratio,
        ]
        for w in compact_width_widgets:
            try:
                w.setMaximumWidth(92)
            except Exception:
                pass

        pick_btn_grid = QtWidgets.QGridLayout()
        pick_btn_grid.setSpacing(2)
        pick_btn_grid.setContentsMargins(0, 0, 0, 0)
        for btn in (self.btn_auto_pick, self.btn_interp_pick, self.btn_save_picks, self.btn_clear_picks):
            try:
                btn.setMinimumWidth(62)
            except Exception:
                pass
        pick_btn_grid.addWidget(self.btn_auto_pick, 0, 0)
        pick_btn_grid.addWidget(self.btn_interp_pick, 0, 1)
        pick_btn_grid.addWidget(self.btn_save_picks, 1, 0)
        pick_btn_grid.addWidget(self.btn_clear_picks, 1, 1)
        pick_btn_wrap = QtWidgets.QWidget()
        pick_btn_wrap.setLayout(pick_btn_grid)
        pick_grid.addWidget(pick_btn_wrap, 3, 0, 1, 4)
        # Mute状态按钮放在拾取面板，避免与顶部导航区混排
        self.btn_mute_status.setMinimumWidth(84)
        self.btn_clear_mute.setMinimumWidth(50)
        self.btn_clear_mute.setMaximumWidth(54)
        pick_grid.addWidget(self.btn_mute_status, 4, 0, 1, 2)
        pick_grid.addWidget(self.btn_clear_mute, 4, 2, 1, 2)
        pick_group_layout = QtWidgets.QVBoxLayout(self.group_pick)
        pick_group_layout.setContentsMargins(1, 1, 1, 1)
        pick_group_layout.addWidget(pick_page)
        params_layout.addWidget(self.group_pick, stretch=0)
        self._param_groups.append(self.group_pick)

        # 对齐/叠加面板（按语义分区，减少“对齐/更新”混淆）
        self.group_align = QtWidgets.QGroupBox("对齐/叠加")
        self._param_group_keys[self.group_align] = "align"
        self.group_align.setMinimumWidth(116)
        self.group_align.setMaximumWidth(122)
        align_page = QtWidgets.QWidget()
        align_layout = QtWidgets.QVBoxLayout(align_page)
        align_layout.setContentsMargins(1, 1, 1, 1)
        align_layout.setSpacing(0)

        stack_row = QtWidgets.QHBoxLayout()
        self.chk_show_stack = QtWidgets.QCheckBox("显示叠加")
        self.chk_show_stack.setChecked(False)
        stack_row.addWidget(QtWidgets.QLabel("叠加"))
        stack_row.addWidget(self.chk_show_stack)
        stack_row.addStretch(1)
        align_layout.addLayout(stack_row)

        align_layout.addWidget(self.btn_align_pick)
        align_layout.addWidget(self.btn_clear_align)
        align_layout.addWidget(self.btn_align_adaptive)
        align_layout.addWidget(self.btn_eval_stack)
        align_layout.addStretch(1)

        align_group_layout = QtWidgets.QVBoxLayout(self.group_align)
        align_group_layout.setContentsMargins(1, 1, 1, 1)
        align_group_layout.addWidget(align_page)
        params_layout.addWidget(self.group_align, stretch=0)
        self._param_groups.append(self.group_align)

        # 单列布局下统一按钮宽度，兼顾可读性与紧凑度
        for btn in (self.btn_align_pick, self.btn_clear_align, self.btn_align_adaptive, self.btn_eval_stack):
            btn.setMinimumWidth(90)

        # 波形操作面板（V 键选段后的叠加/预留校正）
        self.group_waveop = QtWidgets.QGroupBox("波形操作")
        self._param_group_keys[self.group_waveop] = "waveop"
        self.group_waveop.setMinimumWidth(116)
        self.group_waveop.setMaximumWidth(122)
        waveop_page = QtWidgets.QWidget()
        waveop_layout = QtWidgets.QVBoxLayout(waveop_page)
        waveop_layout.setContentsMargins(1, 1, 1, 1)
        waveop_layout.setSpacing(1)
        waveop_layout.addWidget(self.btn_waveop_stack)
        waveop_layout.addWidget(self.btn_waveop_att)
        waveop_layout.addWidget(self.btn_waveop_clear)
        waveop_layout.addWidget(self.btn_waveop_save)
        waveop_layout.addWidget(self.btn_waveop_load)
        waveop_layout.addWidget(self.list_waveop_segments)
        waveop_layout.addStretch(1)
        waveop_group_layout = QtWidgets.QVBoxLayout(self.group_waveop)
        waveop_group_layout.setContentsMargins(1, 1, 1, 1)
        waveop_group_layout.addWidget(waveop_page)
        params_layout.addWidget(self.group_waveop, stretch=0)
        self._param_groups.append(self.group_waveop)

        for btn in (self.btn_waveop_stack, self.btn_waveop_att, self.btn_waveop_clear, self.btn_waveop_save, self.btn_waveop_load):
            btn.setMinimumWidth(90)

        # 高级校正独立面板
        self.group_advcorr = QtWidgets.QGroupBox("高级校正")
        self._param_group_keys[self.group_advcorr] = "advcorr"
        self.group_advcorr.setMinimumWidth(182)
        self.group_advcorr.setMaximumWidth(192)
        adv_page = QtWidgets.QWidget()
        adv_grid = QtWidgets.QGridLayout(adv_page)
        adv_grid.setContentsMargins(1, 1, 1, 1)
        adv_grid.setHorizontalSpacing(4)
        adv_grid.setVerticalSpacing(0)
        adv_grid.addWidget(self.btn_theory, 0, 0)
        adv_grid.addWidget(self.btn_clear_theory, 0, 1)
        adv_grid.addWidget(self.btn_water_corr, 1, 0)
        adv_grid.addWidget(self.btn_clear_water, 1, 1)
        adv_grid.addWidget(self.btn_water_curve, 2, 0, 1, 2)
        adv_grid.addWidget(self.btn_static_corr, 3, 0)
        adv_grid.addWidget(self.btn_clear_static, 3, 1)
        adv_group_layout = QtWidgets.QVBoxLayout(self.group_advcorr)
        adv_group_layout.setContentsMargins(1, 1, 1, 1)
        adv_group_layout.addWidget(adv_page)
        params_layout.addWidget(self.group_advcorr, stretch=0)
        self._param_groups.append(self.group_advcorr)

        # 走时模板面板（tx.in 读取/预览/映射）
        self.group_ttpl = QtWidgets.QGroupBox("走时模板")
        self._param_group_keys[self.group_ttpl] = "ttpl"
        self.group_ttpl.setMinimumWidth(116)
        self.group_ttpl.setMaximumWidth(122)
        ttpl_page = QtWidgets.QWidget()
        ttpl_layout = QtWidgets.QVBoxLayout(ttpl_page)
        ttpl_layout.setContentsMargins(1, 1, 1, 1)
        ttpl_layout.setSpacing(1)
        ttpl_layout.addWidget(self.btn_load_txin)
        ttpl_layout.addWidget(self.btn_clear_txin)
        ttpl_layout.addWidget(self.btn_preview_map_txin)
        ttpl_layout.addWidget(self.btn_map_txin)
        ttpl_layout.addWidget(self.chk_map_txin_apick_only)
        ttpl_layout.addWidget(self.chk_map_txin_view_only)
        tol_row = QtWidgets.QHBoxLayout()
        tol_row.setContentsMargins(0, 0, 0, 0)
        tol_row.setSpacing(3)
        tol_row.addWidget(QtWidgets.QLabel("容差%"))
        tol_row.addWidget(self.spin_map_txin_tol)
        ttpl_layout.addLayout(tol_row)
        ttpl_layout.addStretch(1)
        ttpl_group_layout = QtWidgets.QVBoxLayout(self.group_ttpl)
        ttpl_group_layout.setContentsMargins(1, 1, 1, 1)
        ttpl_group_layout.addWidget(ttpl_page)
        params_layout.addWidget(self.group_ttpl, stretch=0)
        self._param_groups.append(self.group_ttpl)
        params_layout.addStretch(1)

        # 参数面板支持拖拽重排：在组标题区域按住左键拖动，释放后重排
        for group in self._param_groups:
            try:
                group.installEventFilter(self)
                group.setMouseTracking(True)
                group.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
                min_w = max(120, int(group.minimumWidth() or 120))
                max_w_raw = int(group.maximumWidth() or 0)
                max_w = max(max_w_raw if max_w_raw > 0 else 0, min_w + 220)
                max_w = max(max_w, 360)
                group.setMinimumWidth(min_w)
                group.setMaximumWidth(max_w)
                self._panel_resize_bounds[id(group)] = (min_w, max_w)
            except Exception:
                pass
        self._restore_panel_layout_state()

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
        self._enforce_params_panel_fixed_height()
        self._update_gain_effect_hint()
        self._sync_denoise_params_from_ui()
        self._update_denoise_hint()

        self.plot = pg.PlotWidget(background="w")
        self.plot.showGrid(x=True, y=True, alpha=0.12)
        self.plot.getPlotItem().setLabels(left="Time (s)", bottom="Offset (km)")
        # 关闭 PyQtGraph 默认右键菜单（View All / X Axis / Y Axis / Mouse Mode）
        self.plot.getPlotItem().setMenuEnabled(False)
        self.plot.getViewBox().setMenuEnabled(False)
        # 地震图习惯：时间向下
        self.plot.getViewBox().invertY(True)
        root.addWidget(self.plot, stretch=1)

        self.lbl_status = QtWidgets.QLabel("就绪")
        self.lbl_orientation_preview = QtWidgets.QLabel("姿态校正预览: OFF")
        self.lbl_orientation_preview.setStyleSheet("color:#64748b; font-weight:600;")
        self.chk_orientation_preview_toggle = QtWidgets.QCheckBox("快速切换")
        self.chk_orientation_preview_toggle.setEnabled(False)
        self.chk_orientation_preview_toggle.setChecked(False)
        self.chk_orientation_preview_toggle.toggled.connect(self._on_orientation_preview_toggle)
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

    def _toggle_params_panel(self) -> None:
        visible = self.params_panel_scroll.isVisible()
        self.params_panel_scroll.setVisible(not visible)
        self.btn_toggle_panels.setText("显示面板" if visible else "隐藏面板")

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
        self.btn_save_params.clicked.connect(self._save_parameters)
        self.btn_load_params.clicked.connect(self._load_parameters)
        self.btn_save_z.clicked.connect(self._save_z_with_picks)
        self.btn_data_info.clicked.connect(self._show_data_info)
        self.btn_coord_params.clicked.connect(self._show_coordinate_parameters)
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
        self.btn_help.clicked.connect(self._show_help)
        self.btn_shortcuts.clicked.connect(self._show_shortcuts)
        self.btn_about.clicked.connect(self._show_about)
        self.btn_mute_status.clicked.connect(self._toggle_mute_polygon_mode)
        self.btn_clear_mute.clicked.connect(self._clear_mute_all)
        self.btn_toggle_panels.clicked.connect(self._toggle_params_panel)
        self.btn_save_picks.clicked.connect(self._save_picks)
        self.btn_undo_pick.clicked.connect(self._undo_last_pick_edit)
        self.btn_redo_pick.clicked.connect(self._redo_last_pick_edit)
        self.btn_save_hdr.clicked.connect(self._save_picks_to_hdr)
        self.btn_export_tx.clicked.connect(self._export_txin)
        self.btn_clear_picks.clicked.connect(self._clear_picks)
        self.btn_auto_pick.clicked.connect(self._run_auto_pick)
        self.btn_interp_pick.clicked.connect(self._run_interp_pick)
        self.btn_align_pick.clicked.connect(self._run_pick_alignment)
        self.btn_align_adaptive.clicked.connect(self._run_adaptive_alignment)
        self.btn_eval_stack.clicked.connect(self._show_stacking_evaluation)
        self.btn_waveop_stack.clicked.connect(self._run_waveop_stack_from_selections)
        self.btn_waveop_att.clicked.connect(self._run_attitude_correction_placeholder)
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
        self.combo_itype.currentIndexChanged.connect(lambda _: self.request_render())
        self.spin_nskip.valueChanged.connect(lambda _: self.request_render())
        self.spin_ndecim.valueChanged.connect(lambda _: self.request_render())
        self.spin_vred.valueChanged.connect(lambda _: self.request_render())
        self.spin_vred.valueChanged.connect(lambda _: self._update_y_axis_label())
        self.spin_xmin.valueChanged.connect(lambda _: self._apply_window_from_controls())
        self.spin_xmax.valueChanged.connect(lambda _: self._apply_window_from_controls())
        self.spin_tmin.valueChanged.connect(lambda _: self._apply_window_from_controls())
        self.spin_tmax.valueChanged.connect(lambda _: self._apply_window_from_controls())
        self.spin_amp.valueChanged.connect(lambda _: self.request_render())
        self.combo_iscale.currentIndexChanged.connect(lambda _: self.request_render())
        self.spin_rcor.valueChanged.connect(lambda _: self.request_render())
        self.spin_sf.valueChanged.connect(lambda _: self.request_render())
        self.spin_tvg.valueChanged.connect(lambda _: self.request_render())
        self.spin_pvg.valueChanged.connect(lambda _: self.request_render())
        self.spin_clip.valueChanged.connect(lambda _: self.request_render())
        self.spin_dscale.valueChanged.connect(lambda _: self.request_render())
        self.chk_filter.stateChanged.connect(lambda _: self.request_render())
        self.chk_gain.stateChanged.connect(lambda _: self.request_render())
        self.spin_freqlo.valueChanged.connect(lambda _: self.request_render())
        self.spin_freqhi.valueChanged.connect(lambda _: self.request_render())
        self.spin_npoles.valueChanged.connect(lambda _: self.request_render())
        self.chk_zerop.stateChanged.connect(lambda _: self.request_render())
        self.combo_mode.currentIndexChanged.connect(lambda _: self.request_render())
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
        self.combo_iscale.currentIndexChanged.connect(lambda _: self._update_gain_effect_hint())
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

    def _init_debug_log_file(self) -> None:
        """初始化调试日志文件。"""
        if not bool(getattr(self, "_debug_log_enabled", False)):
            return
        try:
            self._debug_log_path = Path.cwd() / "zplotpy_denoise_debug.log"
            self._debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._debug_log_path.open("a", encoding="utf-8") as f:
                f.write("\n")
                f.write(f"===== QtFastViewer session {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        except Exception:
            try:
                self._debug_log_path = Path(tempfile.gettempdir()) / "zplotpy_denoise_debug.log"
                self._debug_log_path.parent.mkdir(parents=True, exist_ok=True)
                with self._debug_log_path.open("a", encoding="utf-8") as f:
                    f.write("\n")
                    f.write(f"===== QtFastViewer session {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            except Exception:
                self._debug_log_enabled = False

    def _debug_log(self, tag: str, message: str) -> None:
        """写入调试日志（持久化）。"""
        if not bool(getattr(self, "_debug_log_enabled", False)):
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{str(tag)}] {str(message)}"
        if line == self._debug_last_line:
            return
        self._debug_last_line = line
        try:
            with self._debug_log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _set_status_text(self, text: str, hold_ms: int = 0, force: bool = False) -> None:
        """设置状态栏文本；可选短时保留，避免被高频渲染状态覆盖。"""
        now_ms = int(time.monotonic() * 1000.0)
        if (not force) and now_ms < int(self._status_hold_until_ms):
            self._debug_log("STATUS_SKIP", str(text))
            return
        self.lbl_status.setText(str(text))
        self._debug_log("STATUS", str(text))
        if int(hold_ms) > 0:
            self._status_hold_until_ms = now_ms + int(hold_ms)
        else:
            self._status_hold_until_ms = 0

    def _apply_optional_theme(self, theme_mode: str) -> None:
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        mode = str(theme_mode).strip().lower()
        theme = self._THEME_PRESETS.get(mode)
        if not theme:
            self._set_status_text(f"未知主题模式：{theme_mode}")
            return
        try:
            effective_theme = dict(theme)
            if mode != "default":
                dark_modes = {"dark", "solarized_dark", "nord", "graphite"}
                text_color = "#ffffff" if mode in dark_modes else "#000000"
                # 非默认主题按明暗设置高对比不透明文字：浅色黑字，深色白字
                effective_theme["text"] = text_color
                effective_theme["label_text"] = text_color
                effective_theme["status_text"] = text_color
                effective_theme["hint_text"] = text_color
                effective_theme["disabled_text"] = text_color
                effective_theme["accent_text"] = text_color
                effective_theme["plot_axis"] = text_color
            self._active_theme_mode = mode
            self._active_theme = effective_theme
            if mode == "default":
                # 默认主题回归原生 Qt 外观，不套自定义样式表
                app.setStyleSheet("")
                self._apply_plot_theme(effective_theme, native_default=True)
            else:
                app.setStyleSheet(self._build_theme_stylesheet(effective_theme))
                self._apply_plot_theme(effective_theme, native_default=False)
            self._sync_params_panel_content_height()
            self._enforce_params_panel_fixed_height()
            self._set_status_text(f"主题已应用：{effective_theme['name']}")
        except Exception as exc:
            self._set_status_text(f"主题应用失败：{exc}")

    def _theme_color(self, key: str, default: str) -> str:
        return str(self._active_theme.get(key, default))

    def _file_dialog_options(self) -> QtWidgets.QFileDialog.Option:
        """非默认主题使用 Qt 文件对话框，确保文件/目录名受高对比主题控制。"""
        options = QtWidgets.QFileDialog.Option(0)
        if str(getattr(self, "_active_theme_mode", "default")).strip().lower() != "default":
            options |= QtWidgets.QFileDialog.Option.DontUseNativeDialog
        return options

    def _show_themed_info(self, title: str, text: str) -> None:
        """统一信息弹窗：非默认主题时强制高对比文本。"""
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Information)
        box.setWindowTitle(str(title))
        box.setText(str(text))
        box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        if str(getattr(self, "_active_theme_mode", "default")).strip().lower() != "default":
            txt = self._theme_color("text", "#000000")
            lbl = self._theme_color("label_text", txt)
            bg = self._theme_color("surface_bg", "#ffffff")
            panel = self._theme_color("panel_bg", bg)
            border = self._theme_color("border", "#888888")
            accent = self._theme_color("accent", "#3b82f6")
            box.setStyleSheet(
                "QMessageBox, QMessageBox QWidget { color: %s; background-color: %s; }"
                "QMessageBox QLabel { color: %s; background-color: transparent; }"
                "QMessageBox QPushButton { color: %s; background-color: %s; border: 1px solid %s; padding: 4px 10px; min-width: 70px; }"
                "QMessageBox QPushButton:hover { border-color: %s; }"
                % (txt, bg, lbl, txt, panel, border, accent)
            )
        box.exec()

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

    def _enforce_params_panel_fixed_height(self) -> None:
        """强制参数区维持固定高度，主题切换后也不外溢。"""
        panel_root = getattr(self, "params_panel_scroll", None)
        if panel_root is None:
            return
        fixed_h = int(getattr(self, "_params_panel_fixed_height", 220) or 220)
        if fixed_h <= 0:
            fixed_h = 220
        try:
            panel_root.setMinimumHeight(fixed_h)
            panel_root.setMaximumHeight(fixed_h)
            panel_root.setFixedHeight(fixed_h)
        except Exception:
            pass
        # 关键：同步内容容器最小高度，超出固定高度时走滚动而非视觉溢出
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
            needed_h = int(getattr(self, "_params_panel_fixed_height", 220) or 220)
        try:
            host.setMinimumHeight(needed_h)
            host.updateGeometry()
        except Exception:
            pass

    def _apply_file_action_text_contrast(self, text_color: str, native_default: bool = False) -> None:
        """提升文件相关操作按钮文字对比度。"""
        file_buttons = [
            self.btn_open,
            self.btn_open_hdr,
            self.btn_open_rec,
            self.btn_save_params,
            self.btn_load_params,
            self.btn_save_z,
            self.btn_save_picks,
            self.btn_save_hdr,
            self.btn_export_tx,
            self.btn_export_fig,
        ]
        for btn in file_buttons:
            try:
                if native_default:
                    btn.setStyleSheet("")
                else:
                    btn.setStyleSheet(
                        "QPushButton { color: %s; font-weight: 600; } "
                        "QPushButton:disabled { color: %s; font-weight: 600; }" % (text_color, text_color)
                    )
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

    def _apply_plot_theme(self, theme: Dict[str, str], native_default: bool = False) -> None:
        """同步图窗、状态栏和辅助提示的主题颜色。"""
        try:
            self.plot.setBackground(theme.get("plot_bg", theme.get("surface_bg", "#ffffff")))
            plot_item = self.plot.getPlotItem()
            axis_color = theme.get("plot_axis", theme.get("text", "#1f2937"))
            axis_pen = pg.mkPen(axis_color, width=1)
            for axis_name in ("left", "bottom"):
                axis = plot_item.getAxis(axis_name)
                axis.setPen(axis_pen)
                axis.setTextPen(axis_pen)
            grid_alpha = float(theme.get("plot_grid_alpha", 0.12))
            self.plot.showGrid(x=True, y=True, alpha=max(0.0, min(1.0, grid_alpha)))
            wave_pen = pg.mkPen(theme.get("wave_pen", "#0a0a0a"), width=1)
            for item in self._curve_items:
                item.setPen(wave_pen)
            if self._shade_item is not None:
                self._shade_item.setPen(pg.mkPen(theme.get("shade_pen", "#30343b"), width=1))
            if self._stack_item is not None:
                self._stack_item.setPen(pg.mkPen(theme.get("stack_pen", "#1478dc"), width=1.5))
            if self._static_preview_item is not None:
                self._static_preview_item.setPen(
                    pg.mkPen(
                        theme.get("static_preview_pen", "#c828a0"),
                        width=1.8,
                        style=QtCore.Qt.PenStyle.DashLine,
                    )
                )
            if self._theoretical_item is not None:
                self._theoretical_item.setPen(pg.mkPen(theme.get("theory_pen", "#f07814"), width=2))
            if self._txin_item is not None:
                self._txin_item.setPen(pg.mkPen(theme.get("txin_pen", "#7c3aed"), width=1))
            if self._txin_map_preview_item is not None:
                self._txin_map_preview_item.setPen(pg.mkPen(theme.get("txin_preview_pen", "#f97316"), width=1.5))
            if self._water_corr_item is not None:
                self._water_corr_item.setPen(
                    pg.mkPen(theme.get("water_pen", "#1ea0d2"), width=2, style=QtCore.Qt.PenStyle.DashLine)
                )
            if self._pick_item is not None:
                self._pick_item.setPen(pg.mkPen(theme.get("pick_pen", "#dc1e1e"), width=1))
                self._pick_item.setBrush(pg.mkBrush(theme.get("pick_brush", "#ff7878")))
        except Exception:
            pass

        if native_default:
            try:
                self.lbl_status.setStyleSheet("")
            except Exception:
                pass
            try:
                self.lbl_gain_hint.setStyleSheet("color:#666; font-size:11px;")
            except Exception:
                pass
            self._apply_panel_frame_contrast(native_default=True)
            self._apply_param_panel_text_contrast("#000000", native_default=True)
            self._apply_file_action_text_contrast("#000000", native_default=True)
            return

        try:
            self.lbl_status.setStyleSheet(
                "padding: 2px 6px; border-top: 1px solid {0}; background-color: {1}; color: {2};".format(
                    theme.get("border", "#c7ccd5"),
                    theme.get("surface_bg", "#ffffff"),
                    theme.get("status_text", theme.get("text", "#1f2937")),
                )
            )
        except Exception:
            pass

        try:
            self.lbl_gain_hint.setStyleSheet(
                "color:{0}; font-size:11px;".format(
                    theme.get("hint_text", theme.get("text", "#666666"))
                )
            )
        except Exception:
            pass
        self._apply_panel_frame_contrast(native_default=False)
        self._apply_param_panel_text_contrast(theme.get("label_text", theme.get("text", "#000000")), native_default=False)
        self._apply_file_action_text_contrast(theme.get("text", "#000000"), native_default=False)

    def _build_theme_stylesheet(self, theme: Dict[str, str]) -> str:
        """构建主题样式：仅切换颜色，不修改尺寸/间距。"""
        return f"""
QWidget {{
    background-color: {theme['window_bg']};
    color: {theme['text']};
}}
QMainWindow, QScrollArea, QMenu, QMenuBar, QStatusBar {{
    background-color: {theme['window_bg']};
    color: {theme['text']};
}}
QLabel {{
    color: {theme['label_text']};
    background: transparent;
}}
QLabel:disabled {{
    color: {theme['label_text']};
}}
QGroupBox {{
    color: {theme['label_text']};
    border-color: {theme.get('panel_border', theme['accent'])};
    background-color: {theme['panel_bg']};
}}
QGroupBox::title {{
    color: {theme['label_text']};
    background-color: {theme.get('panel_title_bg', theme['window_bg'])};
    font-weight: 600;
    font-size: 16px;
}}
QGroupBox:disabled, QGroupBox::title:disabled {{
    color: {theme['label_text']};
}}
QPushButton, QToolButton {{
    border-color: {theme['border']};
    background-color: {theme['surface_bg']};
    color: {theme['text']};
}}
QPushButton:hover, QToolButton:hover {{
    border-color: {theme['accent']};
}}
QPushButton:pressed, QToolButton:pressed {{
    background-color: {theme['panel_bg']};
}}
QPushButton:disabled, QToolButton:disabled {{
    background-color: {theme['disabled_bg']};
    color: {theme['disabled_text']};
    border-color: {theme['border']};
}}
QComboBox, QSpinBox, QDoubleSpinBox {{
    border-color: {theme['border']};
    background-color: {theme['surface_bg']};
    color: {theme['text']};
    selection-background-color: {theme['accent']};
    selection-color: {theme['accent_text']};
}}
QComboBox QAbstractItemView {{
    border: 1px solid {theme['border']};
    background-color: {theme['surface_bg']};
    color: {theme['text']};
    selection-background-color: {theme['accent']};
    selection-color: {theme['accent_text']};
}}
QCheckBox, QRadioButton {{
    color: {theme['label_text']};
}}
QCheckBox:disabled, QRadioButton:disabled {{
    color: {theme['label_text']};
}}
QFileDialog, QFileDialog QWidget {{
    color: {theme['text']};
}}
QFileDialog QTreeView, QFileDialog QListView, QFileDialog QTableView {{
    color: {theme['text']};
    background-color: {theme['surface_bg']};
    selection-color: {theme['accent_text']};
    selection-background-color: {theme['accent']};
}}
QFileDialog QLineEdit, QFileDialog QComboBox {{
    color: {theme['text']};
    background-color: {theme['surface_bg']};
    border-color: {theme['border']};
}}
QFileDialog QHeaderView::section {{
    color: {theme['label_text']};
    background-color: {theme['panel_bg']};
}}
QScrollBar::handle:vertical {{
    background: {theme['border']};
}}
QFrame {{
    color: {theme['border']};
}}
"""

    def _setup_shortcuts(self) -> None:
        """全局窗口快捷键（避免焦点在子控件时按键失效）。"""
        shortcut_map = [
            ("Q", self.close),
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
            ("F1", self._show_shortcuts),
            ("Ctrl+O", self._choose_dfile),
        ]
        for key, handler in shortcut_map:
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            sc.setContext(QtCore.Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(handler)
            self._shortcuts.append(sc)

    def _toggle_pick_mode(self) -> None:
        self.chk_pick_mode.setChecked(not self.chk_pick_mode.isChecked())
        self.lbl_status.setText("拾取模式已开启" if self.chk_pick_mode.isChecked() else "拾取模式已关闭")

    def _waveop_apick_segment_count(self, apick: Optional[int] = None) -> int:
        if apick is None:
            apick = int(self.spin_apick.value())
        cnt = 0
        for sel in self.waveform_selections:
            if int(sel.get("pick_word", apick)) == int(apick):
                cnt += 1
        return cnt

    def _waveop_apick_status_suffix(self, apick: Optional[int] = None) -> str:
        if apick is None:
            apick = int(self.spin_apick.value())
        return f"（apick={int(apick)}，V段={self._waveop_apick_segment_count(int(apick))}）"

    def _on_apick_changed(self, _value: int) -> None:
        self._refresh_waveop_selection_list()
        apick = int(self.spin_apick.value())
        self._set_status_text(f"当前拾取字: {apick}，V段: {self._waveop_apick_segment_count(apick)}", hold_ms=1200)

    def _set_apick_from_shortcut(self, pick_word: int) -> None:
        if self.spin_apick.maximum() <= 0:
            return
        val = int(max(1, min(self.spin_apick.maximum(), int(pick_word))))
        self.spin_apick.setValue(val)
        self.lbl_status.setText(f"当前拾取字: {val}，V段: {self._waveop_apick_segment_count(val)}")

    def _shift_apick(self, delta: int) -> None:
        cur = int(self.spin_apick.value())
        nxt = cur + int(delta)
        nxt = max(1, min(int(self.spin_apick.maximum()), nxt))
        if nxt != cur:
            self.spin_apick.setValue(nxt)
        apick = int(self.spin_apick.value())
        self.lbl_status.setText(f"当前拾取字: {apick}，V段: {self._waveop_apick_segment_count(apick)}")

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
            "coord_params": self.btn_coord_params,
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
            "help": self.btn_help,
            "shortcuts": self.btn_shortcuts,
            "about": self.btn_about,
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
            "export_tx": self.btn_export_tx,
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

    def _is_panel_title_hit(self, group: QtWidgets.QGroupBox, pos: QtCore.QPoint) -> bool:
        """仅在 QGroupBox 标题带内启用拖拽，避免干扰内部控件。"""
        try:
            if not group.rect().contains(pos):
                return False
        except Exception:
            return False
        return int(pos.y()) <= int(self._panel_drag_title_height_px)

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
            min_w, max_w = self._panel_resize_bounds.get(id(group), (120, 900))
            clamped = max(int(min_w), min(int(max_w), int(width)))
            try:
                group.setFixedWidth(clamped)
            except Exception:
                continue

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

        # 全局跟踪拖拽中的幽灵面板：即使鼠标离开原面板仍持续跟随
        if self._panel_drag_active and et == QtCore.QEvent.Type.MouseMove:
            self._update_panel_drag_ghost_position(self._panel_drag_global_pos_from_event(event))
        elif (not self._panel_drag_active) and (not self._panel_resize_active) and et == QtCore.QEvent.Type.MouseMove:
            self._refresh_panel_hover_cursors(self._panel_drag_global_pos_from_event(event))

        # 全局结束拖拽：在任意控件上释放左键都能完成拖拽收尾
        if self._panel_drag_candidate is not None and et == QtCore.QEvent.Type.MouseButtonRelease:
            try:
                is_left = getattr(event, "button", lambda: None)() == QtCore.Qt.MouseButton.LeftButton
            except Exception:
                is_left = False
            if is_left:
                self._finish_panel_drag(self._panel_drag_global_pos_from_event(event))
                self._refresh_panel_hover_cursors()

        # 参数面板拖拽重排（标题区域按住左键拖动）
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
                    try:
                        obj.setCursor(QtCore.Qt.CursorShape.SizeHorCursor)
                    except Exception:
                        pass
                elif self._is_panel_title_hit(obj, event.pos()):
                    self._panel_drag_candidate = obj
                    self._panel_drag_active = False
                    self._panel_drag_start_global = self._panel_drag_global_pos_from_event(event)
                    self._panel_drag_hotspot = QtCore.QPoint(
                        max(0, min(int(event.pos().x()), int(obj.width()) - 1)),
                        max(0, min(int(event.pos().y()), int(obj.height()) - 1)),
                    )
                    self._destroy_panel_drag_ghost()
                    try:
                        obj.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
                    except Exception:
                        pass
            elif et == QtCore.QEvent.Type.MouseMove:
                if self._panel_resize_active and self._panel_resize_group is obj:
                    try:
                        gx = int(event.globalPosition().x())
                    except Exception:
                        gx = int(QtGui.QCursor.pos().x())
                    dx = gx - int(self._panel_resize_start_x)
                    if self._panel_resize_edge == "left":
                        dx = -dx
                    new_w = int(self._panel_resize_start_width + dx)
                    min_w, max_w = self._panel_resize_bounds.get(id(obj), (120, 900))
                    new_w = max(int(min_w), min(int(max_w), new_w))
                    try:
                        obj.setFixedWidth(new_w)
                    except Exception:
                        pass
                elif self._panel_drag_candidate is obj:
                    now_global = self._panel_drag_global_pos_from_event(event)
                    if (now_global - self._panel_drag_start_global).manhattanLength() >= int(self._panel_drag_threshold_px):
                        if not self._panel_drag_active:
                            self._panel_drag_active = True
                            self._create_panel_drag_ghost(obj)
                    if self._panel_drag_active:
                        self._update_panel_drag_ghost_position(now_global)
                self._update_panel_cursor(obj, event.pos())
            elif et == QtCore.QEvent.Type.MouseButtonRelease and getattr(event, "button", lambda: None)() == QtCore.Qt.MouseButton.LeftButton:
                if self._panel_resize_active and self._panel_resize_group is obj:
                    self._panel_resize_active = False
                    self._panel_resize_group = None
                    self._panel_resize_edge = ""
                    self._destroy_panel_drag_ghost()
                    self._save_panel_layout_state()
                    self._refresh_panel_hover_cursors(self._panel_drag_global_pos_from_event(event))
                    return super().eventFilter(obj, event)
                self._finish_panel_drag(self._panel_drag_global_pos_from_event(event))
                self._refresh_panel_hover_cursors(self._panel_drag_global_pos_from_event(event))

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

    def _choose_dfile(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 Z 数据文件", "", "Z files (*.z);;All files (*)", options=self._file_dialog_options()
        )
        if not path:
            return
        self._dfile = path
        self._update_file_open_status_label()
        self._load_data()

    def _choose_hfile(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择头文件", "", "Header files (*.hdr *.h);;All files (*)", options=self._file_dialog_options()
        )
        if path:
            self._hfile = path
            self._update_file_open_status_label()
            if self._dfile:
                self._load_data()

    def _choose_rfile(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择记录文件", "", "Record files (*.r *.txt);;All files (*)", options=self._file_dialog_options()
        )
        if path:
            self._rfile = path
            self._update_file_open_status_label()
            if self._dfile:
                self._load_data()

    def _apply_far_offset_boost(self) -> None:
        """一键增强远偏移弱能量可见性（Fortran 风格友好参数）。"""
        # 远偏移增强：固定比例 + 距离补偿为主，避免过度放大噪声
        self.combo_iscale.setCurrentIndex(1)
        self.spin_rcor.setValue(0.8)
        self.spin_amp.setValue(1.6)
        self.spin_tvg.setValue(0.8)
        self.spin_pvg.setValue(1.2)
        self.chk_filter.setChecked(True)
        self.spin_clip.setValue(2.5)
        self.spin_dscale.setValue(1.25)
        self.request_render(delay_ms=10)
        self.lbl_status.setText("已应用远偏移增强预设：iscale=1, rcor=0.8, amp=1.6, tvg=0.8, pvg=1.2, dscale=1.25")

    def _apply_gain_preset_balanced(self) -> None:
        """平衡显示预设：兼顾近偏移与远偏移可见性。"""
        self.combo_iscale.setCurrentIndex(0)
        self.spin_rcor.setValue(0.3)
        self.spin_amp.setValue(1.2)
        self.spin_tvg.setValue(1.0)
        self.spin_pvg.setValue(1.0)
        self.chk_filter.setChecked(True)
        self.spin_clip.setValue(0.0)
        self.spin_dscale.setValue(1.0)
        self.request_render(delay_ms=10)
        self.lbl_status.setText("已应用平衡显示预设：iscale=0, rcor=0.3, amp=1.2, tvg=1.0, pvg=1.0, dscale=1.0")

    def _apply_gain_preset_strong(self) -> None:
        """强增强预设：弱能量优先，允许更高噪声。"""
        self.combo_iscale.setCurrentIndex(1)
        self.spin_rcor.setValue(1.2)
        self.spin_amp.setValue(2.2)
        self.spin_tvg.setValue(0.6)
        self.spin_pvg.setValue(1.5)
        self.chk_filter.setChecked(True)
        self.spin_clip.setValue(2.5)
        self.spin_dscale.setValue(1.6)
        self.request_render(delay_ms=10)
        self.lbl_status.setText("已应用强增强预设：iscale=1, rcor=1.2, amp=2.2, tvg=0.6, pvg=1.5, dscale=1.6")

    def _update_gain_effect_hint(self) -> None:
        iscale = int(self.combo_iscale.currentIndex())
        clip = float(self.spin_clip.value())
        dscale = float(self.spin_dscale.value())
        if iscale == 2:
            txt = "iscale=2: 主要生效 amp/tvg/pvg/clip；sf/rcor 对显示影响较弱"
        elif iscale == 1:
            txt = "iscale=1: 主要生效 sf/rcor/amp/clip（最接近 Fortran 固定比例）"
        else:
            txt = "iscale=0: 主要生效 amp/clip；sf/rcor 不参与主缩放"
        if clip > 0:
            txt += f"；当前 clip={clip:.3g} 已开启裁剪"
        txt += f"；dscale={dscale:.3g}"
        if hasattr(self, "lbl_gain_hint") and self.lbl_gain_hint is not None:
            self.lbl_gain_hint.setText(txt)

    def _sync_denoise_params_from_ui(self, *_args) -> None:
        """同步去噪参数到内存态与参数对象。"""
        bwconn = 8
        try:
            bwconn = int(self.combo_denoise_bwconn.currentText().strip())
        except Exception:
            bwconn = 8
        if bwconn not in (4, 8):
            bwconn = 8
        f_s = float(self.spin_denoise_f_s.value())
        f_e = float(self.spin_denoise_f_e.value())
        if f_e <= f_s:
            f_e = max(f_s + 0.1, 0.1)
            self.spin_denoise_f_e.blockSignals(True)
            self.spin_denoise_f_e.setValue(f_e)
            self.spin_denoise_f_e.blockSignals(False)
        scope_value = str(self.combo_denoise_scope.currentData() or "rendered").strip().lower()
        if scope_value not in ("rendered", "visible", "record", "selected"):
            scope_value = "rendered"
        diff_gain = float(self.combo_denoise_diff_gain.currentData() or 1.0)
        if (not np.isfinite(diff_gain)) or diff_gain <= 0.0:
            diff_gain = 1.0
        coh_win = int(self._denoise_params.get("coh_win", 11))
        if coh_win < 5:
            coh_win = 5
        if (coh_win % 2) == 0:
            coh_win += 1
        coh_lag = max(0, int(self._denoise_params.get("coh_lag", 2)))
        coh_thr = float(self._denoise_params.get("coh_thr", 0.55))
        coh_thr = float(min(0.95, max(0.05, coh_thr)))
        coh_blend = float(self._denoise_params.get("coh_blend", 0.35))
        coh_blend = float(min(0.90, max(0.0, coh_blend)))
        coh_penalty = float(self._denoise_params.get("coh_penalty", 0.08))
        coh_penalty = float(min(0.50, max(0.0, coh_penalty)))
        perf_diag = bool(self._denoise_params.get("perf_diag", False))
        morph_enable = bool(self._denoise_params.get("morph_enable", True))
        morph_preset = str(self._denoise_params.get("morph_preset", "balanced")).strip().lower()
        if morph_preset not in ("conservative", "balanced", "strong"):
            morph_preset = "balanced"
        morph_quantile = float(self._denoise_params.get("morph_quantile", 0.70))
        morph_quantile = float(min(0.98, max(0.05, morph_quantile)))
        morph_min_area = int(max(1, int(self._denoise_params.get("morph_min_area", 24))))
        morph_expand = int(min(6, max(0, int(self._denoise_params.get("morph_expand", 1)))))
        morph_floor_ratio = float(self._denoise_params.get("morph_floor_ratio", 0.03))
        morph_floor_ratio = float(min(0.50, max(0.0, morph_floor_ratio)))
        morph_keep_strong_q = float(self._denoise_params.get("morph_keep_strong_q", 0.95))
        morph_keep_strong_q = float(min(0.999, max(morph_quantile, morph_keep_strong_q)))
        self._denoise_params = {
            "enabled": bool(self.chk_denoise_enabled.isChecked()),
            # UI 语义与内部参数反向映射：勾选(B)->ab_raw=False，取消(A)->ab_raw=True
            "ab_raw": (not bool(self.chk_denoise_ab_raw.isChecked())),
            "show_diff": bool(getattr(self, "chk_denoise_show_diff", None) and self.chk_denoise_show_diff.isChecked()),
            "diff_gain": diff_gain,
            "scope": scope_value,
            "f_s": f_s,
            "f_e": f_e,
            "bwconn": bwconn,
            "strength": float(self.spin_denoise_strength.value()),
            "workers": max(1, int(self.spin_denoise_workers.value())),
            "coh_win": coh_win,
            "coh_lag": coh_lag,
            "coh_thr": coh_thr,
            "coh_blend": coh_blend,
            "coh_penalty": coh_penalty,
            "perf_diag": perf_diag,
            "morph_enable": morph_enable,
            "morph_preset": morph_preset,
            "morph_quantile": morph_quantile,
            "morph_min_area": morph_min_area,
            "morph_expand": morph_expand,
            "morph_floor_ratio": morph_floor_ratio,
            "morph_keep_strong_q": morph_keep_strong_q,
            "pick_guidance": bool(self.chk_denoise_pick_guidance.isChecked()),
            "pick_wavelet_length_sec": float(self.spin_denoise_pick_hw.value()),
            "pick_guidance_floor": float(self.spin_denoise_pick_floor.value()),
            "return_debug": False,
            "return_result": False,
        }
        # 与统一参数对象对齐，保证保存/加载可复现
        self.params.denoise_enabled = 1 if self._denoise_params["enabled"] else 0
        self.params.denoise_ab_raw = 1 if self._denoise_params["ab_raw"] else 0
        self.params.denoise_scope = str(self._denoise_params["scope"])
        self.params.denoise_f_s = float(self._denoise_params["f_s"])
        self.params.denoise_f_e = float(self._denoise_params["f_e"])
        self.params.denoise_bwconn = int(self._denoise_params["bwconn"])
        self.params.denoise_strength = float(self._denoise_params["strength"])
        self.params.denoise_workers = int(self._denoise_params["workers"])
        self.params.denoise_coh_win = int(self._denoise_params["coh_win"])
        self.params.denoise_coh_lag = int(self._denoise_params["coh_lag"])
        self.params.denoise_coh_thr = float(self._denoise_params["coh_thr"])
        self.params.denoise_coh_blend = float(self._denoise_params["coh_blend"])
        self.params.denoise_coh_penalty = float(self._denoise_params["coh_penalty"])
        self.params.denoise_perf_diag = 1 if bool(self._denoise_params["perf_diag"]) else 0
        self.params.denoise_morph_enable = 1 if bool(self._denoise_params["morph_enable"]) else 0
        self.params.denoise_morph_preset = str(self._denoise_params["morph_preset"])
        self.params.denoise_morph_quantile = float(self._denoise_params["morph_quantile"])
        self.params.denoise_morph_min_area = int(self._denoise_params["morph_min_area"])
        self.params.denoise_morph_expand = int(self._denoise_params["morph_expand"])
        self.params.denoise_morph_floor_ratio = float(self._denoise_params["morph_floor_ratio"])
        self.params.denoise_morph_keep_strong_q = float(self._denoise_params["morph_keep_strong_q"])
        self.params.denoise_pick_guidance = 1 if bool(self._denoise_params["pick_guidance"]) else 0
        self.params.denoise_pick_wavelet_length = float(self._denoise_params["pick_wavelet_length_sec"])
        self.params.denoise_pick_floor = float(self._denoise_params["pick_guidance_floor"])
        self.params.denoise_return_debug = 0
        self.params.denoise_return_result = 0
        self._update_denoise_hint()

    def _loaded_times_dt_seconds(self) -> Optional[float]:
        """当前已加载剖面的采样间隔 dt（秒）；无有效时间轴时为 None。"""
        loaded = getattr(self, "loaded", None)
        if not isinstance(loaded, dict):
            return None
        times = np.asarray(loaded.get("times", []), dtype=float)
        if times.size < 2:
            return None
        dt = float(abs(times[1] - times[0]))
        if not np.isfinite(dt) or dt <= 0:
            return None
        return dt

    def _open_denoise_coh_dialog(self) -> None:
        """弹出相干参数窗口，避免占用主面板高度。"""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("相干参数")
        dlg.setModal(True)

        lay = QtWidgets.QVBoxLayout(dlg)
        form = QtWidgets.QFormLayout()

        spin_win = QtWidgets.QSpinBox(dlg)
        spin_win.setRange(5, 101)
        spin_win.setSingleStep(2)
        spin_win.setValue(int(self._denoise_params.get("coh_win", 11)))
        spin_win.setToolTip("cw（单位：样点）：局部 semblance 窗长（奇数）；越大越稳、越平滑。")
        form.addRow("窗长(cw, 样点)", spin_win)

        spin_lag = QtWidgets.QSpinBox(dlg)
        spin_lag.setRange(0, 8)
        spin_lag.setSingleStep(1)
        spin_lag.setValue(int(self._denoise_params.get("coh_lag", 2)))
        spin_lag.setToolTip("cl（单位：样点）：邻道时移搜索范围；越大越能跟踪倾斜同相轴。")
        form.addRow("Lag(cl, 样点)", spin_lag)

        spin_thr = QtWidgets.QDoubleSpinBox(dlg)
        spin_thr.setRange(0.05, 0.95)
        spin_thr.setDecimals(2)
        spin_thr.setSingleStep(0.01)
        spin_thr.setValue(float(self._denoise_params.get("coh_thr", 0.55)))
        spin_thr.setToolTip("ct（单位：无量纲）：相干门控阈值；越高越保守（只保护更高相干区域）。")
        form.addRow("阈值(ct, 无量纲)", spin_thr)

        spin_blend = QtWidgets.QDoubleSpinBox(dlg)
        spin_blend.setRange(0.0, 0.90)
        spin_blend.setDecimals(2)
        spin_blend.setSingleStep(0.01)
        spin_blend.setValue(float(self._denoise_params.get("coh_blend", 0.35)))
        spin_blend.setToolTip("cb（单位：无量纲）：原始信号回混上限；越大越保真，过大可能降低抑噪。")
        form.addRow("回混(cb, 无量纲)", spin_blend)

        spin_pen = QtWidgets.QDoubleSpinBox(dlg)
        spin_pen.setRange(0.0, 0.50)
        spin_pen.setDecimals(3)
        spin_pen.setSingleStep(0.01)
        spin_pen.setValue(float(self._denoise_params.get("coh_penalty", 0.08)))
        spin_pen.setToolTip("cp（单位：无量纲）：lag 轨迹平滑惩罚；越大越连续，过大可能忽略快速变化。")
        form.addRow("平滑惩罚(cp, 无量纲)", spin_pen)

        chk_morph = QtWidgets.QCheckBox("形态学约束", dlg)
        chk_morph.setChecked(bool(self._denoise_params.get("morph_enable", True)))
        chk_morph.setToolTip("开启后在GCV后增加连通域掩膜，抑制孤立噪点")
        form.addRow("Morph", chk_morph)

        combo_morph_preset = QtWidgets.QComboBox(dlg)
        combo_morph_preset.addItem("保守(保信号)", "conservative")
        combo_morph_preset.addItem("平衡", "balanced")
        combo_morph_preset.addItem("强抑噪", "strong")
        idx_preset = combo_morph_preset.findData(str(self._denoise_params.get("morph_preset", "balanced")))
        combo_morph_preset.setCurrentIndex(idx_preset if idx_preset >= 0 else 1)
        combo_morph_preset.setToolTip("形态学约束预设档位")
        form.addRow("Morph预设", combo_morph_preset)

        lay.addLayout(form)

        lbl_coh_dt_equiv = QtWidgets.QLabel(dlg)
        lbl_coh_dt_equiv.setWordWrap(True)
        lbl_coh_dt_equiv.setStyleSheet(
            "font-family: Consolas, 'Courier New', monospace; font-size:11px; color:#246;",
        )

        def _coh_eff_win_samples(raw: int) -> int:
            w = int(max(5, raw))
            if (w % 2) == 0:
                w += 1
            return w

        def _update_coh_dt_equiv_hint() -> None:
            dt_s = self._loaded_times_dt_seconds()
            if dt_s is None:
                lbl_coh_dt_equiv.setText(
                    "等效尺度：未加载剖面或时间轴无效，无法按 dt 换算毫秒（可先加载数据后再打开此窗口核对）。",
                )
                lbl_coh_dt_equiv.setStyleSheet(
                    "font-family: Consolas, 'Courier New', monospace; font-size:11px; color:#888;",
                )
                return
            cw_raw = int(spin_win.value())
            w_eff = _coh_eff_win_samples(cw_raw)
            cl = int(spin_lag.value())
            dt_ms = float(dt_s * 1000.0)
            cw_ms = float(w_eff * dt_s * 1000.0)
            lag_half_ms = float(cl * dt_s * 1000.0)
            sr = 1.0 / dt_s
            fnyq = 0.5 * sr
            lbl_coh_dt_equiv.setStyleSheet(
                "font-family: Consolas, 'Courier New', monospace; font-size:11px; color:#246;",
            )
            w_note = ""
            if w_eff != cw_raw:
                w_note = f"（实际盒式相干窗会使用奇数 cw={w_eff}）"
            lbl_coh_dt_equiv.setText(
                f"等效尺度（由当前剖面 dt 推算）：dt≈{dt_ms:.6g} ms，采样率≈{sr:.6g} Hz，Nyquist≈{fnyq:.6g} Hz。\n"
                f"cw：盒式局部统计窗长约 {cw_ms:.6g} ms（{cw_raw}→{w_eff} 样点）{w_note}。\n"
                f"cl：邻道时移搜索为整数样点偏移 ±{cl}，单侧最大约 ±{lag_half_ms:.6g} ms。"
            )

        spin_win.valueChanged.connect(lambda _v: _update_coh_dt_equiv_hint())
        spin_lag.valueChanged.connect(lambda _v: _update_coh_dt_equiv_hint())
        _update_coh_dt_equiv_hint()
        lay.addWidget(lbl_coh_dt_equiv)

        tip = QtWidgets.QLabel(
            "参数说明：cw/cl 单位为样点；ct/cb/cp 为无量纲。\n"
            "调参方向：弱事件被削弱时可适当降 ct / 升 cb；噪声残留偏多时可升 ct / 降 cb。",
            dlg,
        )
        tip.setStyleSheet("color:#666;")
        tip.setWordWrap(True)
        lay.addWidget(tip)
        tip2 = QtWidgets.QLabel(
            "固定建议初值：cw=11, cl=2, ct=0.55, cb=0.35, cp=0.08。"
            "可用「拾取→估梯度」按某一拾取字的同相轴走时估 ms/km，再点「按 dt 与偏移距推荐 cl/cb」。",
            dlg,
        )
        tip2.setStyleSheet("color:#666;")
        tip2.setWordWrap(True)
        lay.addWidget(tip2)

        slope_form = QtWidgets.QFormLayout()
        combo_dip_slope = QtWidgets.QComboBox(dlg)
        for _lbl, slope_spkm in COHERENCE_DIP_SLOPE_PRESET_ITEMS:
            combo_dip_slope.addItem(_lbl, float(slope_spkm))
        combo_dip_slope.addItem("自定义（下方 ms/km）", CUSTOM_DIP_SLOPE_COMBO_MARKER)
        combo_dip_slope.setCurrentIndex(1)
        combo_dip_slope.setToolTip(
            "用于「推荐 cl」：假定相邻道在记录时间采样轴（times）上沿偏移的时差梯度；**未自动含折合速度**。\n"
            "折合仅在绘图纵轴平移波形、不重采样；屏上看平≠样点域已对齐，大 moveout 仍宜高档或自定义更大 ms/km。",
        )
        spin_dip_slope_custom_ms = QtWidgets.QDoubleSpinBox(dlg)
        spin_dip_slope_custom_ms.setRange(0.5, 200.0)
        spin_dip_slope_custom_ms.setDecimals(2)
        spin_dip_slope_custom_ms.setSingleStep(1.0)
        spin_dip_slope_custom_ms.setValue(18.0)
        spin_dip_slope_custom_ms.setSuffix(" ms/km")
        spin_dip_slope_custom_ms.setToolTip(
            "仅在选中「自定义」时生效：梯度=该值÷1000（s/km），指记录时间域；与折合显示 t−|x|(1/v−1/vhdr) 的纵轴画法无关。"
        )

        def _dip_slope_combo_is_custom() -> bool:
            return combo_dip_slope.currentData() == CUSTOM_DIP_SLOPE_COMBO_MARKER

        def _current_dip_slope_s_per_km() -> float:
            d = combo_dip_slope.currentData()
            if d == CUSTOM_DIP_SLOPE_COMBO_MARKER:
                return float(max(1e-6, spin_dip_slope_custom_ms.value() / 1000.0))
            try:
                return float(d)
            except (TypeError, ValueError):
                return 0.012

        def _on_dip_slope_preset_changed(_idx: int = 0) -> None:
            spin_dip_slope_custom_ms.setEnabled(_dip_slope_combo_is_custom())

        combo_dip_slope.currentIndexChanged.connect(_on_dip_slope_preset_changed)
        _on_dip_slope_preset_changed()
        slope_form.addRow("推荐 cl：梯度档位", combo_dip_slope)
        slope_form.addRow("自定义梯度", spin_dip_slope_custom_ms)
        lay.addLayout(slope_form)

        pick_row_w = QtWidgets.QWidget(dlg)
        pick_row_l = QtWidgets.QHBoxLayout(pick_row_w)
        pick_row_l.setContentsMargins(0, 0, 0, 0)
        spin_pw_for_slope = QtWidgets.QSpinBox(pick_row_w)
        npick_cap = 40
        try:
            if self.pick_manager is not None:
                npick_cap = int(max(1, min(80, int(getattr(self.pick_manager, "npick", 40) or 40))))
        except Exception:
            npick_cap = 40
        spin_pw_for_slope.setRange(1, int(npick_cap))
        try:
            spin_pw_for_slope.setValue(int(max(1, min(npick_cap, int(self.spin_apick.value())))))
        except Exception:
            spin_pw_for_slope.setValue(1)
        spin_pw_for_slope.setToolTip("仅使用该拾取字的点估计 |Δt|/|Δh|；换震相请换拾取字。")
        btn_pick_to_slope = QtWidgets.QPushButton("拾取→估梯度(ms/km)", pick_row_w)
        btn_pick_to_slope.setToolTip(
            "按该拾取字将各道走时与 offsets(km) 排序，取相邻拾取段 |Δt|/|Δh| 的 median 填入自定义梯度，"
            "并在说明里给出 p75 供保守加大 lag。再走「按 dt…推荐 cl/cb」。"
        )
        pick_row_l.addWidget(QtWidgets.QLabel("拾取字"))
        pick_row_l.addWidget(spin_pw_for_slope)
        pick_row_l.addStretch(1)
        pick_row_l.addWidget(btn_pick_to_slope)
        lay.addWidget(pick_row_w)

        def _infer_slope_from_picks() -> None:
            if self.pick_manager is None:
                lbl_cb_cl_rec.setText("拾取管理器不可用。")
                lbl_cb_cl_rec.setStyleSheet("color:#a44; font-size:11px;")
                return
            pw_i = int(spin_pw_for_slope.value())
            by_w = self.pick_manager.get_picks_by_word(pw_i)
            offs_a = None
            ld = getattr(self, "loaded", None)
            if isinstance(ld, dict):
                offs_a = ld.get("offsets")
            med, p75, msg = estimate_moveout_slope_s_per_km_from_picks(by_w, offs_a)
            if med is None or (not math.isfinite(med)) or med <= 0.0:
                lbl_cb_cl_rec.setText(msg)
                lbl_cb_cl_rec.setStyleSheet("color:#a44; font-size:11px;")
                return
            ic = combo_dip_slope.findData(CUSTOM_DIP_SLOPE_COMBO_MARKER)
            if ic >= 0:
                combo_dip_slope.setCurrentIndex(int(ic))
            spin_dip_slope_custom_ms.setValue(float(min(200.0, max(0.5, med * 1000.0))))
            extra = ""
            if p75 is not None and math.isfinite(p75) and med > 0.0 and p75 > med * 1.02:
                extra = f" 保守侧 p75≈{p75 * 1000.0:.2f} ms/km（可手改自定义接近此值以略增 cl）。"
            lbl_cb_cl_rec.setText(msg + extra)
            lbl_cb_cl_rec.setStyleSheet("color:#246; font-size:11px;")
            _on_dip_slope_preset_changed()
            _update_coh_dt_equiv_hint()

        btn_pick_to_slope.clicked.connect(_infer_slope_from_picks)

        lbl_cb_cl_rec = QtWidgets.QLabel("", dlg)
        lbl_cb_cl_rec.setWordWrap(True)
        lbl_cb_cl_rec.setStyleSheet("color:#444; font-size:11px;")

        chk_perf_diag = QtWidgets.QCheckBox("性能诊断日志", dlg)
        chk_perf_diag.setChecked(bool(self._denoise_params.get("perf_diag", False)))
        chk_perf_diag.setToolTip("开启后记录去噪各阶段耗时（写入调试日志）")

        lay.addWidget(lbl_cb_cl_rec)
        lay.addWidget(chk_perf_diag)

        btn_restore = QtWidgets.QPushButton("恢复固定初值", dlg)
        btn_restore.setToolTip("cw=11, cl=2, ct=0.55, cb=0.35, cp=0.08（与文档固定建议一致）")
        btn_suggest_cb_cl = QtWidgets.QPushButton("按 dt 与偏移距推荐 cl/cb", dlg)
        btn_suggest_cb_cl.setToolTip(
            "用当前剖面的 dt 与 offsets(km) 邻道间距中位数估算 lag(cl)，并按采样率启发回混(cb)。\n"
            "梯度取自上方档位或自定义 ms/km（乘以median Δh 与安全系数得到靶邻道时差）。",
        )

        def _apply_dt_spacing_cb_cl_hint() -> None:
            dt_s = self._loaded_times_dt_seconds()
            if dt_s is None:
                lbl_cb_cl_rec.setText("无法推荐：请先加载剖面（有效时间轴）。")
                lbl_cb_cl_rec.setStyleSheet("color:#a44; font-size:11px;")
                return
            offs = None
            ld = getattr(self, "loaded", None)
            if isinstance(ld, dict):
                offs = ld.get("offsets")
            cl_r, cb_r, detail = suggest_coherence_cb_cl_start(
                dt_s,
                offs,
                lag_max=int(spin_lag.maximum()),
                dip_slope_s_per_km=float(_current_dip_slope_s_per_km()),
            )
            spin_lag.setValue(int(cl_r))
            spin_blend.setValue(float(min(float(spin_blend.maximum()), max(float(spin_blend.minimum()), cb_r))))
            lbl_cb_cl_rec.setText(detail)
            lbl_cb_cl_rec.setStyleSheet("color:#444; font-size:11px;")
            _update_coh_dt_equiv_hint()

        btn_suggest_cb_cl.clicked.connect(_apply_dt_spacing_cb_cl_hint)

        def _restore_defaults() -> None:
            spin_win.setValue(11)
            spin_lag.setValue(2)
            spin_thr.setValue(0.55)
            spin_blend.setValue(0.35)
            spin_pen.setValue(0.08)
            chk_morph.setChecked(True)
            combo_morph_preset.setCurrentIndex(combo_morph_preset.findData("balanced"))
            lbl_cb_cl_rec.clear()
            _update_coh_dt_equiv_hint()
        btn_restore.clicked.connect(_restore_defaults)

        btn_row_preset = QtWidgets.QHBoxLayout()
        btn_row_preset.addWidget(btn_restore)
        btn_row_preset.addWidget(btn_suggest_cb_cl)
        lay.addLayout(btn_row_preset)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=dlg,
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return

        coh_win = int(spin_win.value())
        if (coh_win % 2) == 0:
            coh_win += 1
        self._denoise_params["coh_win"] = int(max(5, coh_win))
        self._denoise_params["coh_lag"] = int(max(0, spin_lag.value()))
        self._denoise_params["coh_thr"] = float(min(0.95, max(0.05, spin_thr.value())))
        self._denoise_params["coh_blend"] = float(min(0.90, max(0.0, spin_blend.value())))
        self._denoise_params["coh_penalty"] = float(min(0.50, max(0.0, spin_pen.value())))
        self._denoise_params["perf_diag"] = bool(chk_perf_diag.isChecked())
        self._denoise_params["morph_enable"] = bool(chk_morph.isChecked())
        morph_preset = str(combo_morph_preset.currentData() or "balanced").strip().lower()
        if morph_preset not in ("conservative", "balanced", "strong"):
            morph_preset = "balanced"
        self._denoise_params["morph_preset"] = morph_preset
        if morph_preset == "conservative":
            self._denoise_params["morph_quantile"] = 0.64
            self._denoise_params["morph_min_area"] = 12
            self._denoise_params["morph_expand"] = 1
            self._denoise_params["morph_floor_ratio"] = 0.02
            self._denoise_params["morph_keep_strong_q"] = 0.94
        elif morph_preset == "strong":
            self._denoise_params["morph_quantile"] = 0.82
            self._denoise_params["morph_min_area"] = 48
            self._denoise_params["morph_expand"] = 0
            self._denoise_params["morph_floor_ratio"] = 0.05
            self._denoise_params["morph_keep_strong_q"] = 0.97
        else:
            self._denoise_params["morph_quantile"] = 0.70
            self._denoise_params["morph_min_area"] = 24
            self._denoise_params["morph_expand"] = 1
            self._denoise_params["morph_floor_ratio"] = 0.03
            self._denoise_params["morph_keep_strong_q"] = 0.95

        self._on_denoise_ui_changed()

    def _on_denoise_ui_changed(self, *_args) -> None:
        """去噪控件变更：同步参数并触发重绘。"""
        self._denoise_run_armed = False
        self._clear_denoise_cache()
        self._sync_denoise_params_from_ui()
        self._sync_plot_pan_lock_state()
        self.request_render(delay_ms=10)

    def _on_denoise_view_mode_changed(self, *_args) -> None:
        """仅显示模式变更（A/B/差值）不解除已启动态。"""
        self._sync_denoise_params_from_ui()
        self._sync_plot_pan_lock_state()
        self.request_render(delay_ms=10)

    def _clear_denoise_selected_traces(self) -> None:
        """清空手动选道集合。"""
        self._denoise_run_armed = False
        self._clear_denoise_cache()
        self._denoise_selected_traces.clear()
        self._update_denoise_hint()
        self._sync_plot_pan_lock_state()
        self.request_render(delay_ms=10)

    def _is_denoise_select_mode_active(self) -> bool:
        scope_selected = str(self._denoise_params.get("scope", "rendered")).strip().lower() == "selected"
        return (
            scope_selected
            and bool(self.chk_denoise_enabled.isChecked())
            and self.loaded is not None
        )

    def _is_denoise_select_modifier_active(self) -> bool:
        """手动选道手势修饰键：Shift/Ctrl/Alt 任一按下。"""
        mods = QtWidgets.QApplication.keyboardModifiers()
        return bool(
            (mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
            or (mods & QtCore.Qt.KeyboardModifier.ControlModifier)
            or (mods & QtCore.Qt.KeyboardModifier.AltModifier)
        )

    def _clear_denoise_cache(self) -> None:
        self._denoise_cache_entries.clear()
        self._denoise_frozen_ready = False
        self._denoise_frozen_trace_set = set()
        self._denoise_frozen_by_trace = {}
        self._denoise_frozen_original_by_trace = {}
        self._denoise_frozen_delta_mean_abs = 0.0
        self._denoise_frozen_delta_max_abs = 0.0

    def _sync_plot_pan_lock_state(self) -> None:
        """统一同步 plot 平移开关，避免与手动选道/编辑模式抢事件。"""
        # 去噪手动选道不再长期锁定平移；仅在明确框选手势期间临时锁定
        lock_pan = bool(self._mute_edit_mode)
        self._set_plot_pan_enabled(not lock_pan)

    def _toggle_denoise_selected_trace(self, trace_idx: int, remove_only: bool = False) -> bool:
        """切换/移除单道选中状态，返回是否有变化。"""
        trace_idx = int(trace_idx)
        changed = False
        if remove_only:
            if trace_idx in self._denoise_selected_traces:
                self._denoise_selected_traces.remove(trace_idx)
                changed = True
        else:
            if trace_idx in self._denoise_selected_traces:
                self._denoise_selected_traces.remove(trace_idx)
            else:
                self._denoise_selected_traces.add(trace_idx)
            changed = True
        if changed:
            self._update_denoise_hint()
            self._set_status_text(
                f"选道：道 {trace_idx} -> {'选中' if trace_idx in self._denoise_selected_traces else '取消'}",
                hold_ms=900,
                force=True,
            )
            self._debug_log(
                "DENOISE_PICK",
                f"trace={int(trace_idx)} remove_only={int(bool(remove_only))} selected_count={len(self._denoise_selected_traces)}",
            )
            self.request_render(delay_ms=10)
        return changed

    def _toggle_denoise_selected_trace_by_scene_pos(self, scene_pos, remove_only: bool = False) -> bool:
        """根据 scene 坐标定位最近渲染道并执行切换。"""
        if self._last_render_trace_indices.size == 0 or self._last_render_offsets.size == 0:
            return False
        vb = self.plot.getViewBox()
        try:
            mouse_pt = vb.mapSceneToView(scene_pos)
        except Exception:
            return False
        x = float(mouse_pt.x())
        nearest_i = int(np.argmin(np.abs(self._last_render_offsets - x)))
        trace_idx = int(self._last_render_trace_indices[nearest_i])
        return self._toggle_denoise_selected_trace(trace_idx, remove_only=remove_only)

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

    def _denoise_indices_signature(self, indices: np.ndarray) -> Tuple[int, int, int, int]:
        arr = np.asarray(indices, dtype=np.int64).reshape(-1)
        if arr.size == 0:
            return (0, 0, 0, 0)
        return (
            int(arr.size),
            int(arr.min()),
            int(arr.max()),
            int(np.sum(arr, dtype=np.int64)),
        )

    def _denoise_traces_signature(self, traces: List[np.ndarray]) -> Tuple[int, int, float, float, float, float]:
        n_trace = int(len(traces))
        n_sample = 0
        acc_abs_mean = 0.0
        acc_mean = 0.0
        acc_pow = 0.0
        acc_anchor = 0.0
        for tr in traces:
            x = np.asarray(tr, dtype=np.float64).reshape(-1)
            if x.size <= 0:
                continue
            n_sample += int(x.size)
            acc_abs_mean += float(np.mean(np.abs(x)))
            acc_mean += float(np.mean(x))
            acc_pow += float(np.mean(x * x))
            mid = int(x.size // 2)
            acc_anchor += float(x[0] + x[mid] + x[-1])
        return (
            n_trace,
            int(n_sample),
            round(acc_abs_mean, 6),
            round(acc_mean, 6),
            round(acc_pow, 6),
            round(acc_anchor, 6),
        )

    def _trace_linear_corr_abs(self, a: np.ndarray, b: np.ndarray) -> float:
        """Estimate absolute linear correlation between two traces."""
        x = np.asarray(a, dtype=np.float64).reshape(-1)
        y = np.asarray(b, dtype=np.float64).reshape(-1)
        n = int(min(x.size, y.size))
        if n < 16:
            return 0.0
        x = x[:n]
        y = y[:n]
        x = x - float(np.mean(x))
        y = y - float(np.mean(y))
        sx = float(np.sqrt(np.mean(x * x)))
        sy = float(np.sqrt(np.mean(y * y)))
        if sx <= 1e-12 or sy <= 1e-12:
            return 0.0
        c = float(np.mean((x / sx) * (y / sy)))
        if not np.isfinite(c):
            return 0.0
        return float(min(1.0, max(0.0, abs(c))))

    def _shift_trace_samples(self, x: np.ndarray, shift: int) -> np.ndarray:
        """Shift trace by integer samples with zero padding."""
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        n = int(arr.size)
        if n <= 0 or int(shift) == 0:
            return arr.copy()
        y = np.zeros_like(arr)
        s = int(shift)
        if s > 0:
            y[s:] = arr[: n - s]
        else:
            k = -s
            y[: n - k] = arr[k:]
        return y

    def _coh_conv_kernel(self, win: int) -> np.ndarray:
        """Cached box kernel for local semblance window."""
        w = int(max(5, int(win)))
        if (w % 2) == 0:
            w += 1
        ker = self._coh_gate_kernel_cache.get(int(w))
        if ker is None:
            ker = np.ones((w,), dtype=np.float64)
            self._coh_gate_kernel_cache[int(w)] = ker
        return ker

    def _coh_lag_values(self, max_lag: int) -> np.ndarray:
        """Cached lag values array."""
        mlag = int(max(0, int(max_lag)))
        cached = self._coh_gate_lags_cache.get(mlag)
        if cached is not None:
            return cached
        lags_arr = np.arange(-mlag, mlag + 1, dtype=np.float64)
        if lags_arr.size <= 0:
            lags_arr = np.zeros((1,), dtype=np.float64)
        if len(self._coh_gate_lags_cache) >= 16:
            first_key = next(iter(self._coh_gate_lags_cache.keys()))
            self._coh_gate_lags_cache.pop(first_key, None)
        self._coh_gate_lags_cache[mlag] = lags_arr
        return lags_arr

    def _dp_best_prev_l1(
        self,
        prev_score: np.ndarray,
        pen: float,
        scratch: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """For each state j, solve max_i(prev[i]-pen*|i-j|) in O(k)."""
        p = np.asarray(prev_score, dtype=np.float64).reshape(-1)
        k = int(p.size)
        if k <= 0:
            return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int32)
        if pen <= 0.0:
            idx = np.arange(k, dtype=np.int32)
            return p.copy(), idx

        if scratch is not None:
            left_val = scratch["left_val"]
            right_val = scratch["right_val"]
            left_idx = scratch["left_idx"]
            right_idx = scratch["right_idx"]
            best_val = scratch["best_val"]
            best_idx = scratch["best_idx"]
        else:
            left_val = np.empty((k,), dtype=np.float64)
            right_val = np.empty((k,), dtype=np.float64)
            left_idx = np.empty((k,), dtype=np.int32)
            right_idx = np.empty((k,), dtype=np.int32)
            best_val = np.empty((k,), dtype=np.float64)
            best_idx = np.empty((k,), dtype=np.int32)

        left_val[0] = p[0]
        left_idx[0] = 0
        for j in range(1, k):
            cand = left_val[j - 1] - pen
            if p[j] >= cand:
                left_val[j] = p[j]
                left_idx[j] = j
            else:
                left_val[j] = cand
                left_idx[j] = left_idx[j - 1]

        right_val[k - 1] = p[k - 1]
        right_idx[k - 1] = k - 1
        for j in range(k - 2, -1, -1):
            cand = right_val[j + 1] - pen
            if p[j] >= cand:
                right_val[j] = p[j]
                right_idx[j] = j
            else:
                right_val[j] = cand
                right_idx[j] = right_idx[j + 1]

        sel = left_val >= right_val
        np.copyto(best_val, right_val)
        np.copyto(best_idx, right_idx)
        best_val[sel] = left_val[sel]
        best_idx[sel] = left_idx[sel]
        return best_val, best_idx.astype(np.int32, copy=False)

    def _coh_get_thread_cache(self) -> Dict[str, Dict[Tuple[int, ...], Dict[str, np.ndarray]]]:
        cache = getattr(self._coh_thread_local, "coh_cache", None)
        if cache is None:
            cache = {"work": {}, "dp": {}}
            self._coh_thread_local.coh_cache = cache
        return cache

    def _coh_get_work_buffers(self, k: int, n: int) -> Dict[str, np.ndarray]:
        cache = self._coh_get_thread_cache()["work"]
        key = (int(k), int(n))
        buf = cache.get(key)
        if buf is None:
            kk = int(max(1, k))
            nn = int(max(1, n))
            buf = {
                "a_stack": np.empty((kk, nn), dtype=np.float64),
                "c_stack": np.empty((kk, nn), dtype=np.float64),
                "s_stack": np.empty((kk, nn), dtype=np.float64),
                "e_stack": np.empty((kk, nn), dtype=np.float64),
                "num": np.empty((kk, nn), dtype=np.float64),
                "den": np.empty((kk, nn), dtype=np.float64),
                "sem_stack": np.empty((kk, nn), dtype=np.float64),
                "score": np.empty((kk, nn), dtype=np.float64),
                "parent": np.empty((kk, nn), dtype=np.int32),
                "lag_idx": np.empty((nn,), dtype=np.int32),
            }
            if len(cache) >= 4:
                first_key = next(iter(cache.keys()))
                cache.pop(first_key, None)
            cache[key] = buf
        return buf

    def _coh_get_dp_scratch(self, k: int) -> Dict[str, np.ndarray]:
        cache = self._coh_get_thread_cache()["dp"]
        key = (int(k),)
        buf = cache.get(key)
        if buf is None:
            kk = int(max(1, k))
            buf = {
                "left_val": np.empty((kk,), dtype=np.float64),
                "right_val": np.empty((kk,), dtype=np.float64),
                "left_idx": np.empty((kk,), dtype=np.int32),
                "right_idx": np.empty((kk,), dtype=np.int32),
                "best_val": np.empty((kk,), dtype=np.float64),
                "best_idx": np.empty((kk,), dtype=np.int32),
            }
            if len(cache) >= 8:
                first_key = next(iter(cache.keys()))
                cache.pop(first_key, None)
            cache[key] = buf
        return buf

    def _moving_sum_2d_same(self, x2d: np.ndarray, win: int, out: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorized moving-sum on axis=1 with zero padding ('same')."""
        x = np.asarray(x2d, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("_moving_sum_2d_same expects 2D array")
        n = int(x.shape[1])
        if n <= 0:
            return np.zeros_like(x)
        w = int(max(1, int(win)))
        if (w % 2) == 0:
            w += 1
        h = int(w // 2)
        xp = np.pad(x, ((0, 0), (h, h)), mode="constant")
        cs = np.cumsum(np.pad(xp, ((0, 0), (1, 0)), mode="constant"), axis=1)
        out_arr = cs[:, w:] - cs[:, :-w]
        out_arr = out_arr.astype(np.float64, copy=False)
        if out is not None and out.shape == out_arr.shape:
            np.copyto(out, out_arr)
            return out
        return out_arr

    def _local_semblance_gate(
        self,
        tr_prev: np.ndarray,
        tr_cur: np.ndarray,
        tr_next: np.ndarray,
        *,
        max_lag: int = 2,
        win: int = 11,
        coh_thr: float = 0.55,
        lag_penalty: float = 0.08,
        prev_shift_cache: Optional[Dict[int, np.ndarray]] = None,
        next_shift_cache: Optional[Dict[int, np.ndarray]] = None,
    ) -> np.ndarray:
        """Estimate per-sample coherence gate by local semblance and lag-path tracking."""
        a = np.asarray(tr_prev, dtype=np.float64).reshape(-1)
        b = np.asarray(tr_cur, dtype=np.float64).reshape(-1)
        c = np.asarray(tr_next, dtype=np.float64).reshape(-1)
        n = int(min(a.size, b.size, c.size))
        if n <= 16:
            return np.zeros((max(0, n),), dtype=np.float64)
        a = a[:n]
        b = b[:n]
        c = c[:n]
        ker = self._coh_conv_kernel(int(win))
        w = int(ker.size)
        eps = 1e-12
        lags_arr = self._coh_lag_values(int(max_lag))
        lags = lags_arr.astype(np.int32, copy=False)
        if lags.size <= 0:
            return np.zeros((n,), dtype=np.float64)
        k = int(lags.size)
        buf = self._coh_get_work_buffers(k, n)
        a_stack = buf["a_stack"]
        c_stack = buf["c_stack"]
        s_stack = buf["s_stack"]
        e_stack = buf["e_stack"]
        num = buf["num"]
        den = buf["den"]
        sem_stack = buf["sem_stack"]
        for li, lag in enumerate(lags):
            lag_i = int(lag)
            if prev_shift_cache is not None:
                a_al = prev_shift_cache.get(-lag_i)
                if a_al is None:
                    a_al = self._shift_trace_samples(a, -lag_i)
            else:
                a_al = self._shift_trace_samples(a, -lag_i)
            if next_shift_cache is not None:
                c_al = next_shift_cache.get(lag_i)
                if c_al is None:
                    c_al = self._shift_trace_samples(c, lag_i)
            else:
                c_al = self._shift_trace_samples(c, lag_i)
            a_stack[li, :] = np.asarray(a_al, dtype=np.float64).reshape(-1)[:n]
            c_stack[li, :] = np.asarray(c_al, dtype=np.float64).reshape(-1)[:n]

        b_row = b.reshape((1, n))
        np.add(a_stack, c_stack, out=s_stack)
        s_stack += b_row
        np.multiply(a_stack, a_stack, out=e_stack)
        e_stack += (b_row * b_row)
        e_stack += (c_stack * c_stack)
        num = self._moving_sum_2d_same(s_stack, w, out=num)
        num = num * num
        den = self._moving_sum_2d_same(e_stack, w, out=den)
        den *= 3.0
        den += eps
        np.divide(num, den, out=sem_stack)
        np.clip(sem_stack, 0.0, 1.0, out=sem_stack)

        pen = float(max(0.0, float(lag_penalty)))
        if pen <= 1e-12:
            # Fast path when no lag smoothness penalty.
            best = np.max(sem_stack, axis=0)
            thr = float(min(0.95, max(0.05, coh_thr)))
            gate = np.clip((best - thr) / max(1e-6, (1.0 - thr)), 0.0, 1.0)
            gate = np.convolve(gate, self._coh_gate_smooth_kernel, mode="same")
            return np.clip(gate, 0.0, 1.0)

        # Lag-path tracking DP with O(k*n) transition step for L1 penalty.
        score = buf["score"]
        parent = buf["parent"]
        lag_idx = buf["lag_idx"]
        score.fill(-1e18)
        parent.fill(0)
        score[:, 0] = sem_stack[:, 0]
        dp_scratch = self._coh_get_dp_scratch(k)
        for t in range(1, n):
            best_prev_val, best_prev_idx = self._dp_best_prev_l1(score[:, t - 1], pen, scratch=dp_scratch)
            score[:, t] = sem_stack[:, t] + best_prev_val
            parent[:, t] = best_prev_idx.astype(np.int32)
        lag_idx.fill(0)
        lag_idx[-1] = int(np.argmax(score[:, -1]))
        for t in range(n - 1, 0, -1):
            lag_idx[t - 1] = int(parent[int(lag_idx[t]), t])
        best = sem_stack[lag_idx, np.arange(n)]

        thr = float(min(0.95, max(0.05, coh_thr)))
        gate = np.clip((best - thr) / max(1e-6, (1.0 - thr)), 0.0, 1.0)
        # light smoothing to avoid flicker-like sample spikes
        gate = np.convolve(gate, self._coh_gate_smooth_kernel, mode="same")
        return np.clip(gate, 0.0, 1.0)

    def _denoise_pick_times_for_global_idx(self, gidx: int) -> List[float]:
        """Collect positive pick times (seconds) for one global trace index."""
        if self.pick_manager is None or gidx < 0:
            return []
        try:
            d = self.pick_manager.get_picks_for_trace(int(gidx))
        except Exception:
            return []
        out: List[float] = []
        for _pw, tv in d.items():
            try:
                v = float(tv)
            except Exception:
                continue
            if np.isfinite(v) and v > 0.0:
                out.append(v)
        return out

    def _denoise_pick_template_times_for_globals(self, gidx_list: List[int]) -> List[float]:
        """将若干全局道上的拾取合并为一套时间模板（去噪范围内共用，含未拾取道）。"""
        if self.pick_manager is None or not gidx_list:
            return []
        seen: set[float] = set()
        merged: List[float] = []
        for gi in gidx_list:
            for t in self._denoise_pick_times_for_global_idx(int(gi)):
                key = round(float(t), 4)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(float(t))
        merged.sort()
        return merged

    def _denoise_pick_guidance_cache_stamp(self) -> object:
        """Invalidate denoise cache when pick-guided mode depends on pick sets."""
        if not bool(self._denoise_params.get("pick_guidance", False)):
            return 0
        pm = self.pick_manager
        if pm is None:
            return 1
        try:
            parts: List[Tuple[int, int, float]] = []
            for tidx in sorted(pm.picks.keys()):
                row = pm.picks[tidx]
                for pw in sorted(row.keys()):
                    parts.append((int(tidx), int(pw), round(float(row[pw]), 6)))
            return hash(tuple(parts))
        except Exception:
            return 2

    def _denoise_cache_key_of(
        self,
        traces_before_denoise: List[np.ndarray],
        render_trace_indices: np.ndarray,
        denoise_trace_indices: np.ndarray,
    ) -> Tuple[object, ...]:
        return (
            self._denoise_traces_signature(traces_before_denoise),
            self._denoise_indices_signature(np.asarray(render_trace_indices, dtype=int)),
            self._denoise_indices_signature(np.asarray(denoise_trace_indices, dtype=int)),
            round(float(self._denoise_params.get("f_s", 3.0)), 6),
            round(float(self._denoise_params.get("f_e", 20.0)), 6),
            int(self._denoise_params.get("bwconn", 8)),
            round(float(self._denoise_params.get("strength", 3.0)), 6),
            int(self._denoise_params.get("coh_win", 11)),
            int(self._denoise_params.get("coh_lag", 2)),
            round(float(self._denoise_params.get("coh_thr", 0.55)), 6),
            round(float(self._denoise_params.get("coh_blend", 0.35)), 6),
            round(float(self._denoise_params.get("coh_penalty", 0.08)), 6),
            int(bool(self._denoise_params.get("morph_enable", True))),
            str(self._denoise_params.get("morph_preset", "balanced")),
            round(float(self._denoise_params.get("morph_quantile", 0.70)), 6),
            int(self._denoise_params.get("morph_min_area", 24)),
            int(self._denoise_params.get("morph_expand", 1)),
            round(float(self._denoise_params.get("morph_floor_ratio", 0.03)), 6),
            round(float(self._denoise_params.get("morph_keep_strong_q", 0.95)), 6),
            int(bool(self._denoise_params.get("pick_guidance", False))),
            round(float(self._denoise_params.get("pick_wavelet_length_sec", 0.19)), 6),
            round(float(self._denoise_params.get("pick_guidance_floor", 0.12)), 6),
            self._denoise_pick_guidance_cache_stamp(),
            int(bool(self._denoise_params.get("return_debug", False))),
            int(bool(self._denoise_params.get("return_result", False))),
        )

    def _start_denoise_now(self) -> None:
        """显式启动去噪执行。"""
        if not bool(self.chk_denoise_enabled.isChecked()):
            self._set_status_text("请先勾选去噪启用，再开始去噪", hold_ms=1400)
            return
        self._sync_denoise_params_from_ui()
        self._clear_denoise_cache()
        self._denoise_run_armed = True
        self._set_denoise_progress(0, 0)
        # Ensure progress bar paints before entering heavy render path.
        try:
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass
        self._set_status_text("去噪已启动（本次结果将缓存）", hold_ms=1400)
        self.request_render(immediate=True)

    def _set_denoise_progress(self, done: int, total: int, phase: Optional[str] = None) -> None:
        """更新去噪进度条；total<0 隐藏，total<=0 显示不定进度。"""
        if not hasattr(self, "progress_denoise") or self.progress_denoise is None:
            return
        ptxt = str(phase).strip() if phase is not None else ""
        last_ptxt = str(getattr(self, "_denoise_progress_phase", "") or "")
        if ptxt and ptxt != last_ptxt:
            self._denoise_progress_phase = ptxt
            self._set_status_text(f"DN: {ptxt}", force=True)
        d = int(max(0, done))
        t_raw = int(total)
        if t_raw < 0:
            self._denoise_progress_phase = ""
            self.progress_denoise.setVisible(False)
            return
        t = int(max(0, t_raw))
        self.progress_denoise.setVisible(True)
        if t <= 0:
            self.progress_denoise.setRange(0, 0)
            self.progress_denoise.setFormat(f"DN {ptxt}" if ptxt else "DN 计算中...")
            return
        if d > t:
            d = t
        self.progress_denoise.setRange(0, t)
        self.progress_denoise.setValue(d)
        if ptxt:
            self.progress_denoise.setFormat(f"DN {ptxt} {d}/{t} (%p%)")
        else:
            self.progress_denoise.setFormat(f"DN {d}/{t} (%p%)")

    def _finish_denoise_drag_select(self) -> bool:
        """结束框选并批量加入范围内道；返回是否执行了批量框选。"""
        if (not self._denoise_select_drag_active) or self._last_render_trace_indices.size == 0:
            self._denoise_select_drag_active = False
            self._denoise_select_drag_start_x = None
            self._denoise_select_drag_last_x = None
            self._denoise_select_drag_mode = "add"
            return False
        x0 = self._denoise_select_drag_start_x
        x1 = self._denoise_select_drag_last_x
        mode = str(self._denoise_select_drag_mode or "add")
        self._denoise_select_drag_active = False
        self._denoise_select_drag_start_x = None
        self._denoise_select_drag_last_x = None
        self._denoise_select_drag_mode = "add"
        if x0 is None or x1 is None:
            return False
        xmin = float(min(x0, x1))
        xmax = float(max(x0, x1))
        # 拖拽宽度太小视为点击，不走批量框选
        xr, _ = self.plot.getViewBox().viewRange()
        x_span = max(1e-9, abs(float(xr[1]) - float(xr[0])))
        if abs(xmax - xmin) < 0.005 * x_span:
            return False
        sel_mask = (self._last_render_offsets >= xmin) & (self._last_render_offsets <= xmax)
        picked = self._last_render_trace_indices[sel_mask]
        picked_set = {int(v) for v in picked.tolist()} if picked.size > 0 else set()
        if mode == "replace":
            self._denoise_selected_traces = set(picked_set)
        elif mode == "remove":
            if picked_set:
                self._denoise_selected_traces.difference_update(picked_set)
        else:
            if picked_set:
                self._denoise_selected_traces.update(picked_set)
        self._denoise_select_drag_just_finished = True
        self._update_denoise_hint()
        op_cn = "替换"
        if mode == "remove":
            op_cn = "移除"
        elif mode == "add":
            op_cn = "追加"
        self._set_status_text(
            f"选道框选[{op_cn}]：框内 {len(picked_set)} 道，当前已选 {len(self._denoise_selected_traces)} 道",
            hold_ms=1400,
        )
        self.request_render(delay_ms=10)
        return True

    def _update_denoise_hint(self) -> None:
        if not hasattr(self, "lbl_denoise_hint") or self.lbl_denoise_hint is None:
            return
        scope_count = int(max(0, int(getattr(self, "_last_denoise_scope_count", 0))))
        dmean = float(getattr(self, "_denoise_last_delta_mean_abs", 0.0))
        hint_text = f"道数：{scope_count} | Δmean={dmean:.3e}"
        if bool(self._denoise_params.get("show_diff", False)):
            dg = float(self._denoise_params.get("diff_gain", 1.0))
            if math.isfinite(dg) and dg > 0.0:
                hint_text += f" | 差值已×增益{dg:g}(Δmean 为增益前波形域均值)"
            else:
                hint_text += " | 差值模式(Δmean 为增益前)"
        self.lbl_denoise_hint.setText(hint_text)
        self._debug_log("DENOISE_HINT", hint_text)

    def _apply_denoise_to_render_traces(
        self,
        traces_in: List[np.ndarray],
        times: np.ndarray,
        render_trace_indices: np.ndarray,
        denoise_trace_indices: np.ndarray,
    ) -> List[np.ndarray]:
        """对当前显示链最终道执行 trace 级去噪。"""
        self._denoise_last_applied_count = 0
        self._denoise_last_delta_mean_abs = 0.0
        self._denoise_last_delta_max_abs = 0.0
        if not bool(self._denoise_params.get("enabled", False)):
            self._denoise_backend_stage = "关闭"
            self._set_denoise_progress(0, -1)
            self._debug_log("DENOISE_RUN", "skip:enabled=0")
            return traces_in
        if not bool(self._denoise_run_armed):
            if self._denoise_frozen_ready and len(self._denoise_frozen_by_trace) > 0:
                out_cached: List[np.ndarray] = []
                hit = 0
                render_ids = np.asarray(render_trace_indices, dtype=int)
                frozen_ids = self._denoise_frozen_trace_set
                for i, tr in enumerate(traces_in):
                    gidx = int(render_ids[i]) if i < render_ids.size else -1
                    if gidx in frozen_ids and gidx in self._denoise_frozen_by_trace:
                        y = np.asarray(self._denoise_frozen_by_trace[gidx], dtype=np.float64)
                        x = np.asarray(tr, dtype=np.float64)
                        if y.shape != x.shape:
                            y = np.resize(y, x.shape)
                        out_cached.append(y)
                        hit += 1
                    else:
                        out_cached.append(np.asarray(tr, dtype=np.float64))
                self._denoise_last_applied_count = int(hit)
                self._denoise_last_delta_mean_abs = float(getattr(self, "_denoise_frozen_delta_mean_abs", 0.0))
                self._denoise_last_delta_max_abs = float(getattr(self, "_denoise_frozen_delta_max_abs", 0.0))
                self._denoise_backend_stage = f"缓存回放 | {hit}/{len(out_cached)}道"
                self._set_denoise_progress(0, -1)
                self._debug_log("DENOISE_RUN", f"replay:frozen hit={hit}/{len(out_cached)}")
                return out_cached
            self._denoise_backend_stage = "待开始(点击开始去噪)"
            self._set_denoise_progress(0, -1)
            self._debug_log("DENOISE_RUN", "skip:run_armed=0")
            return traces_in
        if times.size < 2:
            self._denoise_backend_stage = "跳过: 时间轴不足"
            self._set_denoise_progress(0, -1)
            self._debug_log("DENOISE_RUN", "skip:times<2")
            return traces_in

        dt = float(abs(times[1] - times[0]))
        if (not np.isfinite(dt)) or dt <= 0.0:
            self._denoise_backend_stage = "跳过: dt无效"
            self._set_denoise_progress(0, -1)
            self._debug_log("DENOISE_RUN", f"skip:dt_invalid dt={dt}")
            return traces_in

        # 交互期间道数较大时先保持流畅，静止后自动回到去噪渲染
        if self._viewport_interacting and len(traces_in) > 300:
            self._denoise_backend_stage = f"交互中跳过({len(traces_in)}道)"
            self._set_denoise_progress(0, -1)
            self._debug_log("DENOISE_RUN", f"skip:interacting traces={len(traces_in)}")
            return traces_in

        f_s = float(self._denoise_params.get("f_s", 3.0))
        f_e = float(self._denoise_params.get("f_e", 20.0))
        bwconn = int(self._denoise_params.get("bwconn", 8))
        strength = float(self._denoise_params.get("strength", 3.0))
        coh_win = int(self._denoise_params.get("coh_win", 11))
        coh_lag = int(self._denoise_params.get("coh_lag", 2))
        coh_thr = float(self._denoise_params.get("coh_thr", 0.55))
        coh_blend = float(self._denoise_params.get("coh_blend", 0.35))
        coh_penalty = float(self._denoise_params.get("coh_penalty", 0.08))
        morph_enable = bool(self._denoise_params.get("morph_enable", True))
        morph_quantile = float(self._denoise_params.get("morph_quantile", 0.70))
        morph_min_area = int(self._denoise_params.get("morph_min_area", 24))
        morph_expand = int(self._denoise_params.get("morph_expand", 1))
        morph_floor_ratio = float(self._denoise_params.get("morph_floor_ratio", 0.03))
        morph_keep_strong_q = float(self._denoise_params.get("morph_keep_strong_q", 0.95))
        pg_enable = bool(self._denoise_params.get("pick_guidance", False))
        pick_wl = float(self._denoise_params.get("pick_wavelet_length_sec", 0.19))
        pick_fl = float(self._denoise_params.get("pick_guidance_floor", 0.12))
        times_full = np.asarray(times, dtype=np.float64).reshape(-1)
        t0_fb = float(times_full[0]) if times_full.size > 0 else 0.0
        return_debug = bool(self._denoise_params.get("return_debug", False))
        return_result = bool(self._denoise_params.get("return_result", False))
        perf_diag = bool(self._denoise_params.get("perf_diag", False))
        active_set = set(int(i) for i in np.asarray(denoise_trace_indices, dtype=int).tolist())
        if len(active_set) == 0:
            self._denoise_backend_stage = "范围内无可去噪道"
            self._set_denoise_progress(0, -1)
            self._debug_log("DENOISE_RUN", "skip:active_set=0")
            return traces_in
        progress_total = int(len(active_set))
        progress_done = 0
        progress_tick = 0
        self._set_denoise_progress(progress_done, progress_total, "去噪计算")
        self._debug_log(
            "DENOISE_RUN",
            f"start(post_ops) render={len(traces_in)} active={len(active_set)} dt={dt:.6f} "
            f"f=({f_s:.3f},{f_e:.3f}) str={strength:.3f} "
            f"coh=(w{coh_win},l{coh_lag},t{coh_thr:.2f},b{coh_blend:.2f},p{coh_penalty:.3f})",
        )

        denoised: List[np.ndarray] = []
        stage_name = "P1-trace"
        denoised_count = 0
        coh_blend_count = 0
        coh_gate_mean_list: List[float] = []
        weak_floor_hit_count = 0
        weak_floor_mean_list: List[float] = []
        delta_mean_list: List[float] = []
        delta_max = 0.0
        render_ids = np.asarray(render_trace_indices, dtype=int)
        active_positions = [i for i, gi in enumerate(render_ids.tolist()) if int(gi) in active_set]
        gidx_in_scope = [int(render_ids[int(p)]) for p in active_positions]
        pg_template_times: List[float] = []
        pick_rows_pg: Optional[List[List[float]]] = None
        if pg_enable:
            pg_template_times = self._denoise_pick_template_times_for_globals(gidx_in_scope)
            if pg_template_times:
                pick_rows_pg = [list(pg_template_times) for _ in active_positions]
        neigh_pos: Dict[int, Tuple[Optional[int], Optional[int]]] = {}
        for k, p in enumerate(active_positions):
            p_prev = active_positions[k - 1] if k > 0 else None
            p_next = active_positions[k + 1] if (k + 1) < len(active_positions) else None
            neigh_pos[int(p)] = (p_prev, p_next)
        denoise_base_by_pos: Dict[int, np.ndarray] = {}
        denoise_success_by_pos: Dict[int, bool] = {}
        stage_success_name = stage_name
        fail_type_name: Optional[str] = None
        worker_n = int(max(1, int(self._denoise_params.get("workers", 1))))
        # Adaptive parallel guard:
        # Thread parallelism can be slower on small/medium workloads due to
        # scheduling overhead and potential GIL contention.
        avg_samples = 0.0
        if active_positions:
            try:
                avg_samples = float(
                    np.mean(
                        np.asarray(
                            [np.asarray(traces_in[int(p)], dtype=np.float64).size for p in active_positions],
                            dtype=np.float64,
                        )
                    )
                )
            except Exception:
                avg_samples = 0.0
        allow_thread_parallel = (
            worker_n > 1
            and len(active_positions) >= 512
            and avg_samples >= 2048.0
            and (not self._viewport_interacting)
        )

        def _run_denoise_one(pos: int) -> Tuple[int, np.ndarray, str, bool, Optional[str]]:
            x_local = np.asarray(traces_in[int(pos)], dtype=np.float64)
            ns_loc = int(x_local.size)
            if times_full.size >= ns_loc:
                ta_loc = times_full[:ns_loc].copy()
            else:
                ta_loc = np.arange(ns_loc, dtype=np.float64) * float(dt) + float(t0_fb)
            pt_loc = list(pg_template_times) if pg_enable else []
            try:
                res_local = denoise_trace(
                    x_local,
                    dt=dt,
                    f_s=f_s,
                    f_e=f_e,
                    bwconn=bwconn,
                    strength=strength,
                    morph_enable=morph_enable,
                    morph_quantile=morph_quantile,
                    morph_min_area=morph_min_area,
                    morph_expand=morph_expand,
                    morph_floor_ratio=morph_floor_ratio,
                    morph_keep_strong_quantile=morph_keep_strong_q,
                    morph_bwconn=bwconn,
                    return_debug=return_debug,
                    return_result=return_result,
                    pick_guidance_enable=bool(pg_enable),
                    pick_times=pt_loc,
                    pick_times_axis=ta_loc,
                    pick_t0_fallback=float(t0_fb),
                    pick_wavelet_length_sec=float(pick_wl),
                    pick_guidance_floor=float(pick_fl),
                )
                if hasattr(res_local, "data"):
                    y_local = np.asarray(getattr(res_local, "data"), dtype=np.float64)
                    meta_local = getattr(res_local, "meta", {}) or {}
                    stage_local = str(meta_local.get("stage", "P1-trace"))
                else:
                    y_local = np.asarray(res_local, dtype=np.float64)
                    stage_local = "P1-trace"
                if y_local.shape != x_local.shape:
                    y_local = np.resize(y_local, x_local.shape)
                return int(pos), y_local, stage_local, True, None
            except Exception as exc_local:
                return int(pos), x_local, "", False, type(exc_local).__name__

        section_batch_used = False
        # Structural acceleration path: batch denoise in processors layer.
        if len(active_positions) > 1 and (not return_debug):
            t_batch0 = time.perf_counter() if perf_diag else 0.0
            try:
                def _section_progress(done_i: int, total_i: int) -> None:
                    self._set_denoise_progress(int(done_i), int(total_i), "去噪计算")
                    try:
                        QtWidgets.QApplication.processEvents()
                    except Exception:
                        pass

                block_in = np.asarray([np.asarray(traces_in[int(p)], dtype=np.float64) for p in active_positions], dtype=np.float64)
                self._set_denoise_progress(0, int(len(active_positions)), "去噪计算")
                try:
                    QtWidgets.QApplication.processEvents()
                except Exception:
                    pass
                block_out = denoise_section(
                    block_in,
                    dt=dt,
                    f_s=f_s,
                    f_e=f_e,
                    bwconn=bwconn,
                    strength=strength,
                    workers=worker_n,
                    morph_enable=morph_enable,
                    morph_quantile=morph_quantile,
                    morph_min_area=morph_min_area,
                    morph_expand=morph_expand,
                    morph_floor_ratio=morph_floor_ratio,
                    morph_keep_strong_quantile=morph_keep_strong_q,
                    morph_bwconn=bwconn,
                    return_debug=False,
                    progress_callback=_section_progress,
                    pick_guidance_enable=bool(pg_enable),
                    pick_times_per_row=pick_rows_pg,
                    pick_times_axis=times_full,
                    pick_t0_fallback=float(t0_fb),
                    pick_wavelet_length_sec=float(pick_wl),
                    pick_guidance_floor=float(pick_fl),
                )
                block_out = np.asarray(block_out, dtype=np.float64)
                if block_out.shape == block_in.shape:
                    for k, p in enumerate(active_positions):
                        denoise_base_by_pos[int(p)] = np.asarray(block_out[int(k)], dtype=np.float64)
                        denoise_success_by_pos[int(p)] = True
                    denoised_count = int(len(active_positions))
                    stage_success_name = "P2-section-batch"
                    progress_done = progress_total
                    self._set_denoise_progress(progress_done, progress_total, "去噪计算")
                    try:
                        QtWidgets.QApplication.processEvents()
                    except Exception:
                        pass
                    section_batch_used = True
                    self._debug_log(
                        "DENOISE_RUN",
                        f"section-batch ok active={len(active_positions)} wk={worker_n}",
                    )
                    if perf_diag:
                        self._debug_log(
                            "DENOISE_PERF",
                            f"section_batch_ms={(time.perf_counter() - t_batch0) * 1000.0:.1f} active={len(active_positions)} wk={worker_n}",
                        )
                else:
                    self._debug_log(
                        "DENOISE_RUN",
                        f"section-batch shape-mismatch in={block_in.shape} out={block_out.shape}, fallback trace-loop",
                    )
            except Exception as exc:
                self._debug_log("DENOISE_RUN", f"section-batch-fallback:{type(exc).__name__}")
                if perf_diag:
                    self._debug_log(
                        "DENOISE_PERF",
                        f"section_batch_fail_ms={(time.perf_counter() - t_batch0) * 1000.0:.1f} err={type(exc).__name__}",
                    )

        if (not section_batch_used) and ((not allow_thread_parallel) or len(active_positions) <= 1):
            if worker_n > 1 and len(active_positions) > 1:
                self._debug_log(
                    "DENOISE_RUN",
                    f"parallel->serial adaptive active={len(active_positions)} avg_samples={avg_samples:.1f} wk={worker_n}",
                )
            for pos in active_positions:
                p, y, stg, ok, err_name = _run_denoise_one(int(pos))
                denoise_base_by_pos[p] = np.asarray(y, dtype=np.float64)
                denoise_success_by_pos[p] = bool(ok)
                if ok:
                    denoised_count += 1
                    if stg:
                        stage_success_name = stg
                elif err_name:
                    fail_type_name = str(err_name)
                progress_done += 1
                progress_tick += 1
                if progress_tick >= 8 or progress_done >= progress_total:
                    progress_tick = 0
                    self._set_denoise_progress(progress_done, progress_total, "去噪计算")
                    try:
                        QtWidgets.QApplication.processEvents()
                    except Exception:
                        pass
        elif not section_batch_used:
            max_workers = int(min(worker_n, max(1, len(active_positions))))
            self._debug_log("DENOISE_RUN", f"parallel workers={max_workers} active={len(active_positions)}")
            # Chunked parallelism: reduce per-trace future overhead and context switching.
            # Use smaller chunks for better load balance, avoiding long tail at 75%/80% etc.
            chunk_size = int(math.ceil(len(active_positions) / max(1, max_workers * 4)))
            chunk_size = int(min(16, max(4, chunk_size)))
            chunks: List[List[int]] = [
                [int(v) for v in active_positions[k : k + chunk_size]]
                for k in range(0, len(active_positions), chunk_size)
            ]

            def _run_denoise_chunk(pos_list: List[int]) -> List[Tuple[int, np.ndarray, str, bool, Optional[str]]]:
                out_chunk: List[Tuple[int, np.ndarray, str, bool, Optional[str]]] = []
                for p_local in pos_list:
                    out_chunk.append(_run_denoise_one(int(p_local)))
                return out_chunk

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_run_denoise_chunk, ch): tuple(ch) for ch in chunks}
                for fut in as_completed(futures):
                    for p, y, stg, ok, err_name in fut.result():
                        denoise_base_by_pos[int(p)] = np.asarray(y, dtype=np.float64)
                        denoise_success_by_pos[int(p)] = bool(ok)
                        if ok:
                            denoised_count += 1
                            if stg:
                                stage_success_name = stg
                        elif err_name:
                            fail_type_name = str(err_name)
                        progress_done += 1
                        progress_tick += 1
                        if progress_tick >= 8 or progress_done >= progress_total:
                            progress_tick = 0
                            self._set_denoise_progress(progress_done, progress_total, "去噪计算")
                            try:
                                QtWidgets.QApplication.processEvents()
                            except Exception:
                                pass

        stage_name = stage_success_name
        if fail_type_name is not None:
            if denoised_count > 0:
                stage_name = f"{stage_name}+部分失败:{fail_type_name}"
            else:
                stage_name = f"部分失败:{fail_type_name}"

        # Layer-4 optimization: precompute shifted traces for active positions and lag set,
        # then reuse in local semblance gating to avoid repeated shift work.
        trace_shift_cache: Dict[int, Dict[int, np.ndarray]] = {}
        if int(coh_lag) > 0 and len(active_positions) > 1:
            lag_vals = list(range(-int(coh_lag), int(coh_lag) + 1))
            est_bytes = int(max(1, len(active_positions))) * int(max(1.0, avg_samples)) * int(len(lag_vals)) * 8
            # Guard memory usage: skip cache when estimated memory is too large.
            if est_bytes <= (256 * 1024 * 1024):
                for p in active_positions:
                    trp = np.asarray(traces_in[int(p)], dtype=np.float64)
                    one_cache: Dict[int, np.ndarray] = {0: trp}
                    for sv in lag_vals:
                        if sv == 0:
                            continue
                        one_cache[int(sv)] = self._shift_trace_samples(trp, int(sv))
                    trace_shift_cache[int(p)] = one_cache
                self._debug_log(
                    "DENOISE_COH",
                    f"shift-cache on active={len(active_positions)} lags={len(lag_vals)} estMB={est_bytes/1048576.0:.1f}",
                )
            else:
                self._debug_log(
                    "DENOISE_COH",
                    f"shift-cache skip active={len(active_positions)} estMB={est_bytes/1048576.0:.1f}",
                )

        def _post_blend_one(pos: int) -> Tuple[int, np.ndarray, bool, float, int, float, Optional[float], float]:
            x_local = np.asarray(traces_in[int(pos)], dtype=np.float64)
            y_local = np.asarray(denoise_base_by_pos.get(int(pos), x_local), dtype=np.float64)
            if y_local.shape != x_local.shape:
                y_local = np.resize(y_local, x_local.shape)
            local_coh_applied = False
            local_gate_mean = 0.0
            local_weak_floor_hit = 0
            local_weak_floor_mean = 0.0
            prev_i, next_i = neigh_pos.get(int(pos), (None, None))
            if prev_i is not None and next_i is not None:
                tr_prev = np.asarray(traces_in[int(prev_i)], dtype=np.float64)
                tr_next = np.asarray(traces_in[int(next_i)], dtype=np.float64)
                prev_cache = trace_shift_cache.get(int(prev_i))
                next_cache = trace_shift_cache.get(int(next_i))
                gate = self._local_semblance_gate(
                    tr_prev=tr_prev,
                    tr_cur=x_local,
                    tr_next=tr_next,
                    max_lag=coh_lag,
                    win=coh_win,
                    coh_thr=coh_thr,
                    lag_penalty=coh_penalty,
                    prev_shift_cache=prev_cache,
                    next_shift_cache=next_cache,
                )
                if gate.size > 0:
                    n_gate = int(min(gate.size, x_local.size, y_local.size))
                    if n_gate > 0:
                        max_blend = float(min(0.90, max(0.0, coh_blend)))
                        alpha = np.clip(gate[:n_gate], 0.0, 1.0) * float(max_blend)
                        # Weak-event protection floor:
                        # For coherent but locally weak samples that are over-suppressed,
                        # raise alpha floor adaptively to preserve valid weak arrivals.
                        y_head = np.asarray(y_local[:n_gate], dtype=np.float64)
                        x_head = np.asarray(x_local[:n_gate], dtype=np.float64)
                        w_e = int(max(5, coh_win))
                        if (w_e % 2) == 0:
                            w_e += 1
                        ker_e = np.ones((w_e,), dtype=np.float64) / float(w_e)
                        ex = np.convolve(x_head * x_head, ker_e, mode="same")
                        ey = np.convolve(y_head * y_head, ker_e, mode="same")
                        rx = np.sqrt(np.maximum(ex, 0.0))
                        ry = np.sqrt(np.maximum(ey, 0.0))
                        eps_e = 1e-12
                        # Low-energy score: lower local amplitude -> higher protection weight.
                        ref = float(np.percentile(rx, 70.0)) if rx.size > 0 else 0.0
                        if ref <= eps_e:
                            ref = float(np.mean(rx) + eps_e)
                        weak_score = np.clip(1.0 - (rx / max(ref, eps_e)), 0.0, 1.0)
                        # Suppression score: more attenuation after denoise -> stronger floor.
                        sup_score = np.clip((rx - ry) / np.maximum(rx, eps_e), 0.0, 1.0)
                        weak_floor_cap = min(0.25, max_blend * 0.80)
                        alpha_floor = weak_floor_cap * np.clip(gate[:n_gate], 0.0, 1.0) * weak_score * sup_score
                        alpha = np.maximum(alpha, alpha_floor)
                        if float(np.max(alpha)) > 1e-6:
                            y_local[:n_gate] = (1.0 - alpha) * y_head + alpha * x_head
                            local_coh_applied = True
                            local_gate_mean = float(np.mean(gate[:n_gate]))
                            local_weak_floor_hit = int(np.sum(alpha_floor > 1e-6))
                            local_weak_floor_mean = float(np.mean(alpha_floor))
            delta_mean_local: Optional[float] = None
            delta_max_local = 0.0
            if bool(denoise_success_by_pos.get(int(pos), False)):
                d_local = np.asarray(y_local - x_local, dtype=np.float64)
                if d_local.size > 0:
                    delta_mean_local = float(np.mean(np.abs(d_local)))
                    try:
                        delta_max_local = float(np.max(np.abs(d_local)))
                    except Exception:
                        delta_max_local = 0.0
            return (
                int(pos),
                y_local,
                bool(local_coh_applied),
                float(local_gate_mean),
                int(local_weak_floor_hit),
                float(local_weak_floor_mean),
                delta_mean_local,
                float(delta_max_local),
            )

        blend_result_by_pos: Dict[int, Tuple[np.ndarray, bool, float, int, float, Optional[float], float]] = {}
        t_blend0 = time.perf_counter() if perf_diag else 0.0
        blend_parallel = (
            worker_n > 1
            and len(active_positions) >= 64
            and (not self._viewport_interacting)
        )
        if blend_parallel:
            blend_workers = int(min(worker_n, max(1, len(active_positions))))
            with ThreadPoolExecutor(max_workers=blend_workers) as pool:
                futures = {pool.submit(_post_blend_one, int(p)): int(p) for p in active_positions}
                for fut in as_completed(futures):
                    pos, y_local, coh_applied, gate_mean, weak_hit, weak_mean, dmean_local, dmax_local = fut.result()
                    blend_result_by_pos[int(pos)] = (
                        np.asarray(y_local, dtype=np.float64),
                        bool(coh_applied),
                        float(gate_mean),
                        int(weak_hit),
                        float(weak_mean),
                        dmean_local,
                        float(dmax_local),
                    )
        else:
            for p in active_positions:
                pos, y_local, coh_applied, gate_mean, weak_hit, weak_mean, dmean_local, dmax_local = _post_blend_one(int(p))
                blend_result_by_pos[int(pos)] = (
                    np.asarray(y_local, dtype=np.float64),
                    bool(coh_applied),
                    float(gate_mean),
                    int(weak_hit),
                    float(weak_mean),
                    dmean_local,
                    float(dmax_local),
                )
        if perf_diag:
            self._debug_log(
                "DENOISE_PERF",
                f"post_blend_ms={(time.perf_counter() - t_blend0) * 1000.0:.1f} active={len(active_positions)} parallel={int(blend_parallel)}",
            )

        for i, tr in enumerate(traces_in):
            x = np.asarray(tr, dtype=np.float64)
            gidx = int(render_ids[i]) if i < render_ids.size else -1
            if gidx not in active_set:
                denoised.append(x)
                continue
            y, coh_applied, gate_mean, weak_hit, weak_mean, dmean_local, dmax_local = blend_result_by_pos.get(
                int(i),
                (np.asarray(denoise_base_by_pos.get(int(i), x), dtype=np.float64), False, 0.0, 0, 0.0, None, 0.0),
            )
            if y.shape != x.shape:
                y = np.resize(y, x.shape)
            if bool(coh_applied):
                coh_blend_count += 1
                coh_gate_mean_list.append(float(gate_mean))
                weak_floor_hit_count += int(weak_hit)
                weak_floor_mean_list.append(float(weak_mean))
            if dmean_local is not None:
                delta_mean_list.append(float(dmean_local))
            if float(dmax_local) > delta_max:
                delta_max = float(dmax_local)
            denoised.append(y)

        if coh_blend_count > 0:
            stage_name = f"{stage_name}+coh"
            gmean = float(np.mean(np.asarray(coh_gate_mean_list, dtype=np.float64))) if coh_gate_mean_list else 0.0
            wmean = float(np.mean(np.asarray(weak_floor_mean_list, dtype=np.float64))) if weak_floor_mean_list else 0.0
            self._debug_log(
                "DENOISE_COH",
                f"coh_blend_count={coh_blend_count}/{denoised_count} gate_mean={gmean:.3f} "
                f"weak_floor_hit={weak_floor_hit_count} weak_floor_mean={wmean:.4f}",
            )
        self._denoise_backend_stage = f"{stage_name} | {denoised_count}/{len(denoised)}道"
        self._denoise_last_applied_count = int(denoised_count)
        if delta_mean_list:
            self._denoise_last_delta_mean_abs = float(np.mean(np.asarray(delta_mean_list, dtype=np.float64)))
        self._denoise_last_delta_max_abs = float(delta_max)
        # 一次性计算完成后冻结结果：后续不重算，直接回放缓存，直到用户再次点击“开始去噪”
        self._denoise_frozen_ready = True
        self._denoise_frozen_trace_set = set(int(i) for i in active_set)
        frozen_map: Dict[int, np.ndarray] = {}
        original_map: Dict[int, np.ndarray] = {}
        render_ids = np.asarray(render_trace_indices, dtype=int)
        for i, tr_raw in enumerate(traces_in):
            gidx_o = int(render_ids[i]) if i < render_ids.size else -1
            if gidx_o >= 0 and gidx_o in active_set:
                original_map[gidx_o] = np.asarray(tr_raw, dtype=np.float64).copy()
        for i, y in enumerate(denoised):
            gidx = int(render_ids[i]) if i < render_ids.size else -1
            if gidx in self._denoise_frozen_trace_set:
                frozen_map[gidx] = np.asarray(y, dtype=np.float64).copy()
        self._denoise_frozen_by_trace = frozen_map
        self._denoise_frozen_original_by_trace = original_map
        self._denoise_run_armed = False
        self._denoise_frozen_delta_mean_abs = float(self._denoise_last_delta_mean_abs)
        self._denoise_frozen_delta_max_abs = float(self._denoise_last_delta_max_abs)
        self._denoise_backend_stage = f"{self._denoise_backend_stage} | 已缓存"
        self._set_denoise_progress(0, -1)
        self._debug_log("DENOISE_RUN", f"done stage={self._denoise_backend_stage}")
        self._debug_log(
            "DENOISE_DELTA",
            f"mean_abs={self._denoise_last_delta_mean_abs:.6e} max_abs={self._denoise_last_delta_max_abs:.6e}",
        )
        return denoised

    def _denoise_compare_cached_trace_ids(self) -> List[int]:
        """返回当前去噪缓存中成对存在的全局道号（已排序）。"""
        if not getattr(self, "_denoise_frozen_ready", False):
            return []
        return sorted(
            int(k)
            for k in self._denoise_frozen_by_trace.keys()
            if int(k) in self._denoise_frozen_original_by_trace
        )

    def _open_denoise_compare_plot(self) -> None:
        """弹出 Matplotlib 窗口：同一道的去噪前后时域、频谱及 TF 形态对比。"""
        if self.loaded is None:
            QtWidgets.QMessageBox.warning(self, "Denoise compare", "Load data first.")
            return
        if (not getattr(self, "_denoise_frozen_ready", False)) or len(getattr(self, "_denoise_frozen_by_trace", {}) or {}) == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Denoise compare",
                "No denoise cache yet. Enable denoise and click Start to compute, then try again.",
            )
            return
        cand = self._denoise_compare_cached_trace_ids()
        if not cand:
            QtWidgets.QMessageBox.warning(
                self,
                "Denoise compare",
                "Cache has no pre-denoise waveforms. Click Start to recompute, then try again.",
            )
            return
        items = [str(i) for i in cand]
        default_row = max(0, len(items) // 2)
        pick, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Denoise A/B figure",
            "Trace index (global):",
            items,
            default_row,
            False,
        )
        if not ok:
            return
        try:
            gidx = int(str(pick).strip())
        except Exception:
            return
        if gidx not in cand:
            return
        self._figure_denoise_ab_compare(int(gidx))

    def _figure_denoise_ab_compare(self, gidx: int) -> None:
        """绘制单独图窗：指定全局道在去噪前后的时域曲线、振幅谱（dB）及 TF（需临时重算）。 """
        try:
            import matplotlib.colors as mcolors
            from matplotlib.figure import Figure
            try:
                from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            except Exception:
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        except Exception as exc_import:
            QtWidgets.QMessageBox.warning(
                self,
                "Denoise compare",
                f"Matplotlib is not available ({type(exc_import).__name__}). Install matplotlib to plot.",
            )
            self._debug_log("DENOISE_COMPARE", f"matplotlib missing: {type(exc_import).__name__}")
            return

        x0_raw = np.asarray(self._denoise_frozen_original_by_trace.get(int(gidx)), dtype=np.float64)
        y0_raw = np.asarray(self._denoise_frozen_by_trace.get(int(gidx)), dtype=np.float64)
        if x0_raw.size == 0 or y0_raw.size == 0:
            QtWidgets.QMessageBox.warning(self, "Denoise compare", "Invalid cache for the selected trace.")
            return
        n = int(min(int(x0_raw.size), int(y0_raw.size)))
        times = np.asarray(self.loaded.get("times", []), dtype=float)
        if times.size < max(8, int(n)):
            QtWidgets.QMessageBox.warning(self, "Denoise compare", "Time axis is too short for this comparison.")
            return
        n = min(n, int(times.size))
        times = np.asarray(times[:n], dtype=float)
        x0 = np.asarray(x0_raw[:n], dtype=np.float64)
        y0 = np.asarray(y0_raw[:n], dtype=np.float64)
        if times.size >= 2:
            dt_raw = float(abs(times[1] - times[0]))
            dt_ok = dt_raw if (np.isfinite(dt_raw) and dt_raw > 0.0) else None
        else:
            dt_ok = None
        if dt_ok is None:
            QtWidgets.QMessageBox.warning(self, "Denoise compare", "Invalid sample interval dt; cannot plot.")
            return

        self._sync_denoise_params_from_ui()
        f_s = float(self._denoise_params.get("f_s", 3.0))
        f_e = float(self._denoise_params.get("f_e", 20.0))
        bwconn = int(self._denoise_params.get("bwconn", 8))
        strength = float(self._denoise_params.get("strength", 3.0))
        morph_enable = bool(self._denoise_params.get("morph_enable", True))
        morph_quantile = float(self._denoise_params.get("morph_quantile", 0.70))
        morph_min_area = int(self._denoise_params.get("morph_min_area", 24))
        morph_expand = int(self._denoise_params.get("morph_expand", 1))
        morph_floor_ratio = float(self._denoise_params.get("morph_floor_ratio", 0.03))
        morph_keep_strong_q = float(self._denoise_params.get("morph_keep_strong_q", 0.95))
        pg = bool(self._denoise_params.get("pick_guidance", False))
        cand_cmp = self._denoise_compare_cached_trace_ids()
        pt = self._denoise_pick_template_times_for_globals([int(x) for x in cand_cmp]) if (pg and cand_cmp) else []
        ta = np.asarray(times[:n], dtype=np.float64)
        t0fb = float(ta[0]) if ta.size > 0 else 0.0
        phw = float(self._denoise_params.get("pick_wavelet_length_sec", 0.19))
        pfl = float(self._denoise_params.get("pick_guidance_floor", 0.12))

        dbg_ok = False
        org_tf_amp: Optional[np.ndarray] = None
        fin_tf_amp: Optional[np.ndarray] = None
        freq_hz: Optional[np.ndarray] = None
        dbg_stage = ""
        try:
            dr = denoise_trace(
                np.asarray(x0, dtype=np.float64),
                dt=dt_ok,
                f_s=f_s,
                f_e=f_e,
                bwconn=bwconn,
                strength=strength,
                morph_enable=morph_enable,
                morph_quantile=morph_quantile,
                morph_min_area=morph_min_area,
                morph_expand=morph_expand,
                morph_floor_ratio=morph_floor_ratio,
                morph_keep_strong_quantile=morph_keep_strong_q,
                morph_bwconn=bwconn,
                return_debug=True,
                pick_guidance_enable=bool(pg),
                pick_times=pt,
                pick_times_axis=ta,
                pick_t0_fallback=t0fb,
                pick_wavelet_length_sec=phw,
                pick_guidance_floor=pfl,
            )
            if hasattr(dr, "debug") and dr.debug is not None:
                dbg = dr.debug
                org_tf_amp = _tf_abs_amp(dbg.org_tf)
                fin_tf_amp = _tf_abs_amp(dbg.final_tf)
                freq_hz = np.asarray(dbg.freq, dtype=np.float64)
                meta = getattr(dr, "meta", {}) or {}
                dbg_stage = str(meta.get("stage", "")).strip()
                dbg_ok = org_tf_amp.size > 0 and fin_tf_amp.size > 0 and freq_hz.size > 1
                if dbg_ok and (org_tf_amp.shape != fin_tf_amp.shape):
                    dbg_ok = False
        except Exception as exc_tf:
            self._debug_log("DENOISE_COMPARE", f"tf-debug fail: {type(exc_tf).__name__}: {exc_tf}")
            dbg_ok = False

        def _spectral_db(sig: np.ndarray, dt_use: float) -> Tuple[np.ndarray, np.ndarray]:
            sig_loc = np.asarray(sig, dtype=np.float64).reshape(-1)
            nn = int(sig_loc.size)
            if nn < 8:
                return np.asarray([0.0], dtype=np.float64), np.asarray([-200.0, -200.0], dtype=np.float64)
            xc = sig_loc - float(np.mean(sig_loc))
            win = np.hanning(nn)
            spec = np.fft.rfft(xc * win)
            fq = np.fft.rfftfreq(nn, d=float(dt_use))
            mag = np.abs(spec)
            ref = float(np.max(mag)) if mag.size > 0 else 0.0
            floor = float(max(1e-20 * max(ref, 1.0), 1e-30))
            db = 20.0 * np.log10(np.maximum(mag, floor))
            return fq, db

        fq_a, db_a = _spectral_db(x0, dt_ok)
        fq_b, db_b = _spectral_db(y0, dt_ok)

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Denoise compare — trace {int(gidx)}")
        dlg.setMinimumSize(940, 720)
        vbox = QtWidgets.QVBoxLayout(dlg)

        figure = Figure(figsize=(11, 9), tight_layout=False)
        canvas = FigureCanvas(figure)

        span = float(max(times[-1] - times[0], dt_ok))

        irec_lab = ""
        try:
            irec_lab = str(int(self.spin_irec.value()))
        except Exception:
            irec_lab = ""

        suptitles = []
        suptitles.append(
            f"record={irec_lab} · trace={int(gidx)} · dt={dt_ok:g} s · f_band=({f_s:g},{f_e:g}) Hz",
        )
        suptitles.append(
            "Blue/orange: time-domain A/B; light green: filled (B−A) scaled by std; "
            + (
                "TF: " + dbg_stage + " (trace B on main view may differ slightly after coherence blend). "
                if dbg_ok
                else "TF: not available (time and spectrum plots only). "
            ),
        )
        try:
            _m1 = ab_trace_metrics(x0, y0)
            suptitles.append(
                f"Metrics (A=input, B=output): SNR={_m1.snr_db:.2f} dB  RMSE={_m1.rmse:.4g}  "
                f"CC={_m1.cc:.4f}  MAE={_m1.mae:.4g}  |  SNR=10·log10(mean(A²)/MSE(A,B))"
            )
            self._debug_log(
                "DENOISE_METRICS",
                f"trace={int(gidx)} SNR_dB={_m1.snr_db:.4f} RMSE={_m1.rmse:.6e} CC={_m1.cc:.6f} MAE={_m1.mae:.6e}",
            )
        except Exception as exc_m:
            suptitles.append(f"Metrics: unavailable ({type(exc_m).__name__})")
            self._debug_log("DENOISE_METRICS", f"trace={int(gidx)} fail:{type(exc_m).__name__}")

        figure.suptitle("\n".join(suptitles), fontsize=10)

        if dbg_ok:
            gs = figure.add_gridspec(3, 2, height_ratios=[2.4, 1.05, 2.05], width_ratios=[1.0, 1.0])
            gs.update(left=0.075, right=0.975, top=0.90, bottom=0.065, hspace=0.40, wspace=0.20)
            ax_t = figure.add_subplot(gs[0, :])
            ax_sp = figure.add_subplot(gs[1, :])
            ax_tf0 = figure.add_subplot(gs[2, 0])
            ax_tf1 = figure.add_subplot(gs[2, 1])
        else:
            gs = figure.add_gridspec(2, 1, height_ratios=[2.4, 1.05])
            gs.update(left=0.075, right=0.975, top=0.90, bottom=0.065, hspace=0.36)
            ax_t = figure.add_subplot(gs[0, :])
            ax_sp = figure.add_subplot(gs[1, :])
            ax_tf0 = ax_tf1 = None  # type: ignore[misc]

        ax_t.plot(times, x0, color="#2166ac", lw=0.95, alpha=0.92, label="A pre-denoise (display pipeline)")
        ax_t.plot(times, y0, color="#ef8a62", lw=0.95, alpha=0.92, label="B final (cache, incl. coherence blend)")

        eps_d = np.maximum(np.std(x0), np.std(y0))
        diff_v = np.asarray(y0 - x0, dtype=np.float64)
        gx = np.clip(diff_v / max(8.0 * float(eps_d if eps_d > 1e-30 else 1.0), 1e-20), -1.0, 1.0)
        ax_t.fill_between(times, 0.0, gx, alpha=0.22, color="#4daf4a", label="Δ(B−A) fill (scaled to ±1)")
        ax_t.set_xlabel(f"Time (s) [{span:g} s]")
        ax_t.set_ylabel("Amplitude")
        ax_t.legend(loc="upper right", fontsize=9)
        ax_t.grid(True, alpha=0.25)

        ax_sp.plot(fq_a, db_a, color="#2166ac", lw=1.05, label="A Hanning FFT spectrum (relative dB)")
        ax_sp.plot(fq_b, db_b, color="#ef8a62", lw=1.05, label="B Hanning FFT spectrum (relative dB)")
        ax_sp.set_xlabel("Frequency (Hz)")
        ax_sp.set_ylabel("relative dB")
        f_hi = float(min(125.0, 0.5 / float(dt_ok)))
        ax_sp.set_xlim(0.0, max(f_hi, float(f_e) * 3.0, 50.0))
        ax_sp.grid(True, which="major", alpha=0.27)
        ax_sp.legend(loc="upper right", fontsize=9)

        if (
            dbg_ok
            and org_tf_amp is not None
            and fin_tf_amp is not None
            and freq_hz is not None
            and ax_tf0 is not None
            and ax_tf1 is not None
        ):
            m_tf = int(org_tf_amp.shape[1])
            t1_tf = float(times[0]) + float(dt_ok) * float(max(0, m_tf - 1))
            ext = [float(times[0]), float(min(t1_tf, float(times[-1]))), float(freq_hz[0]), float(freq_hz[-1])]
            vmin_db = float(
                max(
                    -140.0,
                    20.0 * np.log10(float(np.maximum(np.percentile(org_tf_amp, 1.5), 1e-40))),
                ),
            )
            vmax_raw = float(
                (np.percentile(org_tf_amp, 99.0) + np.percentile(fin_tf_amp, 99.0)) / 3.5 + 1e-20,
            )
            vmax_db = float(20.0 * np.log10(max(vmax_raw, 1e-30)))
            if not (np.isfinite(vmin_db) and np.isfinite(vmax_db) and vmax_db > vmin_db + 1e-3):
                vmin_db, vmax_db = -80.0, 0.0
            try:
                nrm = mcolors.Normalize(vmin=vmin_db, vmax=vmax_db)
            except Exception:
                nrm = None

            _ = ax_tf0.imshow(
                20.0 * np.log10(org_tf_amp + 1e-30),
                aspect="auto",
                origin="lower",
                cmap="inferno",
                interpolation="nearest",
                extent=list(ext),
                norm=nrm,
            )
            im1 = ax_tf1.imshow(
                20.0 * np.log10(fin_tf_amp + 1e-30),
                aspect="auto",
                origin="lower",
                cmap="inferno",
                interpolation="nearest",
                extent=list(ext),
                norm=nrm,
            )
            ax_tf0.set_title("|TF| before denoise (transform domain)")
            ax_tf1.set_title("|TF| after GCV+morph (single denoise_trace)")
            for ax_tf in (ax_tf0, ax_tf1):
                ax_tf.set_xlabel("Time (s)")
                ax_tf.set_ylabel("Freq (Hz)")
            try:
                cbar = figure.colorbar(im1, ax=[ax_tf0, ax_tf1], shrink=0.78, pad=0.02)
                cbar.ax.set_ylabel("dB")
                cbar.ax.tick_params(labelsize=8)
            except Exception:
                pass

        btn_row = QtWidgets.QHBoxLayout()
        btn_export = QtWidgets.QPushButton("Export PNG...")
        btn_close = QtWidgets.QPushButton("Close")
        btn_row.addStretch(1)
        btn_row.addWidget(btn_export)
        btn_row.addWidget(btn_close)
        btn_close.clicked.connect(dlg.close)

        path_default = Path.cwd() / f"zplotpy_denoise_compare_trace{int(gidx)}_irec{irec_lab or 'NA'}.png"

        def _export_png():
            tgt, sf = QtWidgets.QFileDialog.getSaveFileName(
                dlg,
                "Export comparison PNG",
                str(path_default),
                "PNG (*.png);;All files (*)",
            )
            if not sf:
                return
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"This figure includes Axes that are not compatible with tight_layout.*",
                        category=UserWarning,
                    )
                    figure.savefig(str(tgt), dpi=145, bbox_inches="tight")
                self._set_status_text(f"Exported comparison: {tgt}", hold_ms=2600)
                self._debug_log("DENOISE_COMPARE", f"export_png path={tgt}")
            except Exception as exc_sv:
                QtWidgets.QMessageBox.warning(dlg, "Export failed", str(exc_sv))

        btn_export.clicked.connect(_export_png)

        vbox.addWidget(canvas)
        vbox.addLayout(btn_row)

        dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.resize(980, 820)
        dlg.show()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"This figure includes Axes that are not compatible with tight_layout.*",
                    category=UserWarning,
                )
                canvas.draw_idle()
        except Exception:
            pass
        self._floating_dialogs.append(dlg)

        try:
            self._debug_log(
                "DENOISE_COMPARE",
                f"gidx={int(gidx)} n={int(n)} dbg_tf={int(dbg_ok)} stage={dbg_stage}",
            )
        except Exception:
            pass

    def _open_denoise_compare_all_tf_spectrum(self) -> None:
        """全缓存道：振幅谱剖面 (freq × trace) + 各道 |TF| 沿时间轴横向拼接（ribbon，dB）。"""
        if self.loaded is None:
            QtWidgets.QMessageBox.warning(self, "Denoise compare", "Load data first.")
            return
        if (not getattr(self, "_denoise_frozen_ready", False)) or len(getattr(self, "_denoise_frozen_by_trace", {}) or {}) == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Denoise compare",
                "No denoise cache yet. Enable denoise and click Start to compute, then try again.",
            )
            return
        cand = self._denoise_compare_cached_trace_ids()
        if not cand:
            QtWidgets.QMessageBox.warning(
                self,
                "Denoise compare",
                "Cache has no pre-denoise waveforms. Click Start to recompute, then try again.",
            )
            return
        n_tr = int(len(cand))
        if n_tr > 1200:
            ans = QtWidgets.QMessageBox.question(
                self,
                "Denoise compare",
                f"{n_tr} traces in cache. |TF| ribbon runs denoise_trace per trace and may take a long time. Continue?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if ans != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        try:
            import matplotlib.colors as mcolors
            from matplotlib.figure import Figure
            try:
                from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            except Exception:
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        except Exception as exc_import:
            QtWidgets.QMessageBox.warning(
                self,
                "Denoise compare",
                f"Matplotlib is not available ({type(exc_import).__name__}). Install matplotlib to plot.",
            )
            return

        times = np.asarray(self.loaded.get("times", []), dtype=float)
        if times.size < 8:
            QtWidgets.QMessageBox.warning(self, "Denoise compare", "Time axis is too short.")
            return
        dt_raw = float(abs(times[1] - times[0]))
        if (not np.isfinite(dt_raw)) or dt_raw <= 0.0:
            QtWidgets.QMessageBox.warning(self, "Denoise compare", "Invalid sample interval dt.")
            return
        dt_ok = float(dt_raw)

        lens = [
            int(
                min(
                    int(np.asarray(self._denoise_frozen_original_by_trace[int(g)], dtype=np.float64).size),
                    int(np.asarray(self._denoise_frozen_by_trace[int(g)], dtype=np.float64).size),
                ),
            )
            for g in cand
        ]
        n_min = int(min(lens + [int(times.size)]))
        if n_min < 8:
            QtWidgets.QMessageBox.warning(self, "Denoise compare", "Not enough samples per trace.")
            return

        self._sync_denoise_params_from_ui()
        f_s = float(self._denoise_params.get("f_s", 3.0))
        f_e = float(self._denoise_params.get("f_e", 20.0))
        bwconn = int(self._denoise_params.get("bwconn", 8))
        strength = float(self._denoise_params.get("strength", 3.0))
        morph_enable = bool(self._denoise_params.get("morph_enable", True))
        morph_quantile = float(self._denoise_params.get("morph_quantile", 0.70))
        morph_min_area = int(self._denoise_params.get("morph_min_area", 24))
        morph_expand = int(self._denoise_params.get("morph_expand", 1))
        morph_floor_ratio = float(self._denoise_params.get("morph_floor_ratio", 0.03))
        morph_keep_strong_q = float(self._denoise_params.get("morph_keep_strong_q", 0.95))
        worker_n = int(max(1, int(self._denoise_params.get("workers", 1))))
        pg_cmp = bool(self._denoise_params.get("pick_guidance", False))
        phw_cmp = float(self._denoise_params.get("pick_wavelet_length_sec", 0.19))
        pfl_cmp = float(self._denoise_params.get("pick_guidance_floor", 0.12))
        times_cmp = np.asarray(times[:n_min], dtype=np.float64)
        t0_cmp = float(times_cmp[0]) if times_cmp.size > 0 else 0.0
        cand_cmp_ids = [int(x) for x in cand]
        pg_template_cmp = (
            self._denoise_pick_template_times_for_globals(cand_cmp_ids) if pg_cmp else []
        )

        mat_a = np.zeros((n_tr, n_min), dtype=np.float64)
        mat_b = np.zeros((n_tr, n_min), dtype=np.float64)
        for j, gidx in enumerate(cand):
            xa = np.asarray(self._denoise_frozen_original_by_trace[int(gidx)], dtype=np.float64)[:n_min]
            xb = np.asarray(self._denoise_frozen_by_trace[int(gidx)], dtype=np.float64)[:n_min]
            mat_a[j, :] = xa
            mat_b[j, :] = xb

        win1 = np.hanning(n_min)
        xa_dm = mat_a - np.mean(mat_a, axis=1, keepdims=True)
        xb_dm = mat_b - np.mean(mat_b, axis=1, keepdims=True)
        spec_a = np.abs(np.fft.rfft(xa_dm * win1, axis=1))
        spec_b = np.abs(np.fft.rfft(xb_dm * win1, axis=1))
        fq = np.fft.rfftfreq(n_min, d=dt_ok)
        colmax_a = np.maximum(np.max(spec_a, axis=1, keepdims=True), 1e-30)
        colmax_b = np.maximum(np.max(spec_b, axis=1, keepdims=True), 1e-30)
        db_a_g = 20.0 * np.log10(np.maximum(spec_a / colmax_a, 1e-30))
        db_b_g = 20.0 * np.log10(np.maximum(spec_b / colmax_b, 1e-30))
        img_spec_a = np.asarray(db_a_g.T, dtype=np.float64)
        img_spec_b = np.asarray(db_b_g.T, dtype=np.float64)

        per_ab_metrics = [ab_trace_metrics(mat_a[j, :], mat_b[j, :]) for j in range(n_tr)]
        sm, sd, rmm, rmd, cm, cd, mm, md = summarize_metrics(per_ab_metrics)
        lbl_denoise_ab_metrics = QtWidgets.QLabel(
            "A/B metrics (cached traces): SNR dB = 10·log10(mean(A²)/MSE(A,B)); "
            "values are median | mean.\n"
            f"SNR: {sd:.2f} | {sm:.2f} dB   RMSE: {rmd:.4g} | {rmm:.4g}   "
            f"CC: {cd:.4f} | {cm:.4f}   MAE: {md:.4g} | {mm:.4g}"
        )
        lbl_denoise_ab_metrics.setStyleSheet(
            "font-family: Consolas, 'Courier New', monospace; font-size:11px; color:#333;",
        )
        lbl_denoise_ab_metrics.setWordWrap(True)
        try:
            self._debug_log(
                "DENOISE_METRICS",
                f"all n={n_tr} SNR_med={sd:.4f} SNR_mean={sm:.4f} RMSE_med={rmd:.6e} CC_med={cd:.6f}",
            )
        except Exception:
            pass

        prog = QtWidgets.QProgressDialog(
            "|TF| ribbon: denoise_trace (return_debug) per trace…",
            "Cancel",
            0,
            n_tr,
            self,
        )
        prog.setWindowTitle("Denoise compare — all traces")
        prog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)
        try:
            prog.show()
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass

        def _tf_job(gidx: int) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
            xv = np.asarray(self._denoise_frozen_original_by_trace[int(gidx)], dtype=np.float64)[:n_min]
            try:
                pt_j = list(pg_template_cmp) if pg_cmp else []
                dr = denoise_trace(
                    xv,
                    dt=dt_ok,
                    f_s=f_s,
                    f_e=f_e,
                    bwconn=bwconn,
                    strength=strength,
                    morph_enable=morph_enable,
                    morph_quantile=morph_quantile,
                    morph_min_area=morph_min_area,
                    morph_expand=morph_expand,
                    morph_floor_ratio=morph_floor_ratio,
                    morph_keep_strong_quantile=morph_keep_strong_q,
                    morph_bwconn=bwconn,
                    return_debug=True,
                    pick_guidance_enable=bool(pg_cmp),
                    pick_times=pt_j,
                    pick_times_axis=times_cmp,
                    pick_t0_fallback=t0_cmp,
                    pick_wavelet_length_sec=phw_cmp,
                    pick_guidance_floor=pfl_cmp,
                )
                dbg = getattr(dr, "debug", None)
                if dbg is None:
                    return int(gidx), None, None, None
                oa = _tf_abs_amp(dbg.org_tf)
                fa = _tf_abs_amp(dbg.final_tf)
                fr = np.asarray(dbg.freq, dtype=np.float64)
                if oa.shape != fa.shape or fr.size < 2:
                    return int(gidx), None, None, None
                return int(gidx), oa, fa, fr
            except Exception:
                return int(gidx), None, None, None

        tf_pairs: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        freq_hz: Optional[np.ndarray] = None
        done_tf = 0
        cancel_tf = False
        if worker_n <= 1 or n_tr <= 2:
            for gidx in cand:
                if prog.wasCanceled():
                    cancel_tf = True
                    break
                gi, oa, fa, fr = _tf_job(int(gidx))
                if fr is not None and freq_hz is None:
                    freq_hz = fr
                if oa is not None and fa is not None and fr is not None:
                    tf_pairs[int(gi)] = (oa, fa, fr)
                done_tf += 1
                prog.setValue(done_tf)
                try:
                    QtWidgets.QApplication.processEvents()
                except Exception:
                    pass
        else:
            max_w = int(min(worker_n, n_tr, 32))
            with ThreadPoolExecutor(max_workers=max_w) as pool:
                fut_to_g = {pool.submit(_tf_job, int(g)): int(g) for g in cand}
                for fut in as_completed(fut_to_g):
                    if prog.wasCanceled():
                        cancel_tf = True
                        break
                    gi, oa, fa, fr = fut.result()
                    if fr is not None and freq_hz is None:
                        freq_hz = fr
                    if oa is not None and fa is not None and fr is not None:
                        tf_pairs[int(gi)] = (oa, fa, fr)
                    done_tf += 1
                    prog.setValue(done_tf)
                    try:
                        QtWidgets.QApplication.processEvents()
                    except Exception:
                        pass
        try:
            prog.close()
        except Exception:
            pass

        stack_o: List[np.ndarray] = []
        stack_f: List[np.ndarray] = []
        ref_shape: Optional[Tuple[int, ...]] = None
        for gidx in cand:
            pr = tf_pairs.get(int(gidx))
            if not pr:
                continue
            oa, fa, _fr = pr
            if ref_shape is None:
                ref_shape = tuple(oa.shape)
            if tuple(oa.shape) == ref_shape and tuple(fa.shape) == ref_shape:
                stack_o.append(oa)
                stack_f.append(fa)

        ribbon_org: Optional[np.ndarray] = None
        ribbon_fin: Optional[np.ndarray] = None
        n_t_one = 0
        if stack_o and stack_f and len(stack_o) == len(stack_f):
            try:
                ribbon_org = np.concatenate(stack_o, axis=1)
                ribbon_fin = np.concatenate(stack_f, axis=1)
                n_t_one = int(stack_o[0].shape[1])
            except Exception:
                ribbon_org = ribbon_fin = None
                n_t_one = 0
        tf_ok = bool(
            ribbon_org is not None
            and ribbon_fin is not None
            and ribbon_org.shape == ribbon_fin.shape
            and freq_hz is not None
            and int(freq_hz.size) > 1
            and int(ribbon_org.shape[1]) > 1,
        )

        t0 = float(times[0])
        if tf_ok and ribbon_org is not None:
            t1_tf = float(t0 + dt_ok * float(max(0, int(ribbon_org.shape[1]) - 1)))
        else:
            t1_tf = float(times[int(min(n_min - 1, int(times.size) - 1))])
        ext_tf: Optional[List[float]] = None
        if tf_ok and ribbon_org is not None and freq_hz is not None:
            ext_tf = [t0, float(t1_tf), float(freq_hz[0]), float(freq_hz[-1])]

        irec_lab = ""
        try:
            irec_lab = str(int(self.spin_irec.value()))
        except Exception:
            irec_lab = ""

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Denoise — all traces (n={n_tr})")
        vbox = QtWidgets.QVBoxLayout(dlg)
        figure = Figure(figsize=(12, 9), tight_layout=False)
        canvas = FigureCanvas(figure)

        if tf_ok and ribbon_org is not None and ribbon_fin is not None and ext_tf is not None:
            gs = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.15], width_ratios=[1.0, 1.0])
            gs.update(left=0.07, right=0.98, top=0.91, bottom=0.07, hspace=0.33, wspace=0.22)
            ax_sa = figure.add_subplot(gs[0, 0])
            ax_sb = figure.add_subplot(gs[0, 1])
            ax_t0 = figure.add_subplot(gs[1, 0])
            ax_t1 = figure.add_subplot(gs[1, 1])
        else:
            gs = figure.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
            gs.update(left=0.07, right=0.98, top=0.90, bottom=0.10, wspace=0.22)
            ax_sa = figure.add_subplot(gs[0, 0])
            ax_sb = figure.add_subplot(gs[0, 1])
            ax_t0 = ax_t1 = None  # type: ignore[misc]

        x_edges = np.arange(n_tr + 1, dtype=np.float64) - 0.5
        fq0 = float(fq[0])
        fq1 = float(fq[-1])
        extent_sp = [float(x_edges[0]), float(x_edges[-1]), fq0, fq1]

        ax_sa.imshow(
            img_spec_a,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
            extent=extent_sp,
        )
        ax_sa.set_title("Spectrum gather A (per-trace peak = 0 dB)")
        ax_sa.set_xlabel("Trace column (sorted cache order)")
        ax_sa.set_ylabel("Frequency (Hz)")
        tick_idx = np.linspace(0, n_tr - 1, num=min(9, n_tr), dtype=int)
        ax_sa.set_xticks([float(i) for i in tick_idx])
        ax_sa.set_xticklabels([str(int(cand[int(i)])) for i in tick_idx], rotation=35, fontsize=7)

        ax_sb.imshow(
            img_spec_b,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
            extent=extent_sp,
        )
        ax_sb.set_title("Spectrum gather B (per-trace peak = 0 dB)")
        ax_sb.set_xlabel("Trace column (sorted cache order)")
        ax_sb.set_ylabel("Frequency (Hz)")
        ax_sb.set_xticks([float(i) for i in tick_idx])
        ax_sb.set_xticklabels([str(int(cand[int(i)])) for i in tick_idx], rotation=35, fontsize=7)
        f_hi = float(min(125.0, 0.5 / float(dt_ok)))
        for axs in (ax_sa, ax_sb):
            axs.set_ylim(0.0, max(f_hi, float(f_e) * 3.0, 50.0))

        if tf_ok and ribbon_org is not None and ribbon_fin is not None and ext_tf is not None and ax_t0 is not None and ax_t1 is not None:
            mo = ribbon_org.astype(np.float64, copy=False)
            mf = ribbon_fin.astype(np.float64, copy=False)
            lo = 20.0 * np.log10(mo + 1e-30)
            lf = 20.0 * np.log10(mf + 1e-30)
            vmin_db = float(max(-140.0, float(np.percentile(lo, 2.0))))
            vmax_db = float(min(0.0, float(np.percentile(lf, 99.5))))
            if not (np.isfinite(vmin_db) and np.isfinite(vmax_db) and vmax_db > vmin_db + 1e-3):
                vmin_db, vmax_db = -80.0, 0.0
            try:
                nrm = mcolors.Normalize(vmin=vmin_db, vmax=vmax_db)
            except Exception:
                nrm = None
            _ = ax_t0.imshow(
                lo,
                aspect="auto",
                origin="lower",
                cmap="inferno",
                interpolation="nearest",
                extent=list(ext_tf),
                norm=nrm,
            )
            im1 = ax_t1.imshow(
                lf,
                aspect="auto",
                origin="lower",
                cmap="inferno",
                interpolation="nearest",
                extent=list(ext_tf),
                norm=nrm,
            )
            n_blk = int(len(stack_o))
            if n_t_one > 0 and n_blk > 1:
                for k in range(1, n_blk):
                    xv = float(t0 + float(k) * float(n_t_one) * float(dt_ok))
                    ax_t0.axvline(xv, color="w", lw=0.45, alpha=0.55)
                    ax_t1.axvline(xv, color="w", lw=0.45, alpha=0.55)
            ax_t0.set_title(f"|TF| before — horizontal concat ({n_blk} traces)")
            ax_t1.set_title(f"|TF| after GCV+morph — horizontal concat ({n_blk} traces)")
            for ax_tf in (ax_t0, ax_t1):
                ax_tf.set_xlabel("Time (s), traces L→R in cache order")
                ax_tf.set_ylabel("Freq (Hz)")
            try:
                cbar = figure.colorbar(im1, ax=[ax_t0, ax_t1], shrink=0.82, pad=0.02)
                cbar.ax.set_ylabel("dB")
            except Exception:
                pass

        sub = (
            f"record={irec_lab} · traces={n_tr} · n={n_min} · dt={dt_ok:g}s · f=({f_s:g},{f_e:g})Hz · "
            f"TF ribbon traces={len(stack_o)} · "
            f"SNR_med={sd:.1f}dB RMSE_med={rmd:.3g} CC_med={cd:.3f}"
        )
        if cancel_tf:
            sub += " (TF ribbon may be partial: canceled)"
        elif int(len(stack_o)) < n_tr:
            sub += f" (TF skipped/failed on {n_tr - int(len(stack_o))} traces)"
        figure.suptitle(sub, fontsize=10)

        btn_row = QtWidgets.QHBoxLayout()
        btn_export = QtWidgets.QPushButton("Export PNG...")
        btn_close = QtWidgets.QPushButton("Close")
        btn_row.addStretch(1)
        btn_row.addWidget(btn_export)
        btn_row.addWidget(btn_close)
        btn_close.clicked.connect(dlg.close)
        path_default = Path.cwd() / f"zplotpy_denoise_all_tf_spec_n{n_tr}_irec{irec_lab or 'NA'}.png"

        def _export_png():
            tgt, sf = QtWidgets.QFileDialog.getSaveFileName(
                dlg,
                "Export PNG",
                str(path_default),
                "PNG (*.png);;All files (*)",
            )
            if not sf:
                return
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"This figure includes Axes that are not compatible with tight_layout.*",
                        category=UserWarning,
                    )
                    figure.savefig(str(tgt), dpi=150, bbox_inches="tight")
                self._set_status_text(f"Exported: {tgt}", hold_ms=2600)
            except Exception as exc_sv:
                QtWidgets.QMessageBox.warning(dlg, "Export failed", str(exc_sv))

        btn_export.clicked.connect(_export_png)
        vbox.addWidget(canvas)
        vbox.addWidget(lbl_denoise_ab_metrics)
        vbox.addLayout(btn_row)
        dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.resize(1100, 880)
        dlg.show()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"This figure includes Axes that are not compatible with tight_layout.*",
                    category=UserWarning,
                )
                canvas.draw_idle()
        except Exception:
            pass
        self._floating_dialogs.append(dlg)
        self._debug_log(
            "DENOISE_COMPARE",
            f"all_tf_ribbon n_tr={n_tr} n_min={n_min} tf_used={len(stack_o)} cancel={int(cancel_tf)}",
        )

    def _resolve_denoise_indices(
        self,
        idx_all: np.ndarray,
        idx_visible: np.ndarray,
        idx_render: np.ndarray,
    ) -> np.ndarray:
        """根据去噪范围配置，返回应执行去噪的全局道号集合。"""
        scope = str(self._denoise_params.get("scope", "rendered")).strip().lower()
        if scope == "record":
            out = np.asarray(idx_all, dtype=int)
            self._debug_log("DENOISE_SCOPE", f"scope=record count={int(out.size)}")
            return out
        if scope == "visible":
            out = np.asarray(idx_visible, dtype=int)
            self._debug_log("DENOISE_SCOPE", f"scope=visible count={int(out.size)}")
            return out
        if scope == "selected":
            selected = np.asarray(sorted(int(i) for i in self._denoise_selected_traces), dtype=int)
            if selected.size == 0:
                self._debug_log("DENOISE_SCOPE", "scope=selected count=0")
                return np.asarray([], dtype=int)
            all_set = set(int(i) for i in np.asarray(idx_all, dtype=int).tolist())
            selected = np.asarray([int(i) for i in selected if int(i) in all_set], dtype=int)
            self._debug_log("DENOISE_SCOPE", f"scope=selected count={int(selected.size)}")
            return selected
        out = np.asarray(idx_render, dtype=int)
        self._debug_log("DENOISE_SCOPE", f"scope=rendered count={int(out.size)}")
        return out

    def _update_y_axis_label(self) -> None:
        vred = float(self.spin_vred.value())
        if vred > 0:
            self.plot.getPlotItem().setLabels(left=f"t-x/{vred:.3g}", bottom="Offset (km)")
        else:
            self.plot.getPlotItem().setLabels(left="Time (s)", bottom="Offset (km)")

    def _apply_window_from_controls(self) -> None:
        """将 xmin/xmax/tmin/tmax 直接应用到当前视图窗口。"""
        if self.loaded is None:
            return
        if self._syncing_window_controls:
            return
        xmin = float(min(self.spin_xmin.value(), self.spin_xmax.value()))
        xmax = float(max(self.spin_xmin.value(), self.spin_xmax.value()))
        tmin = float(min(self.spin_tmin.value(), self.spin_tmax.value()))
        tmax = float(max(self.spin_tmin.value(), self.spin_tmax.value()))
        if xmin < xmax:
            self.plot.setXRange(xmin, xmax, padding=0.0)
        if tmin < tmax:
            self.plot.setYRange(tmax, tmin, padding=0.0)
        self._did_initial_view_fit = True
        self.request_render(delay_ms=10)

    def _sync_window_controls_from_view(self) -> None:
        """将当前视图范围回写到 xmin/xmax/tmin/tmax 控件。"""
        if self.loaded is None:
            return
        vb = self.plot.getViewBox()
        xr, yr = vb.viewRange()
        xmin, xmax = float(min(xr)), float(max(xr))
        tmin, tmax = float(min(yr)), float(max(yr))
        self._syncing_window_controls = True
        try:
            for w in (self.spin_xmin, self.spin_xmax, self.spin_tmin, self.spin_tmax):
                w.blockSignals(True)
            self.spin_xmin.setValue(xmin)
            self.spin_xmax.setValue(xmax)
            self.spin_tmin.setValue(tmin)
            self.spin_tmax.setValue(tmax)
        finally:
            for w in (self.spin_xmin, self.spin_xmax, self.spin_tmin, self.spin_tmax):
                w.blockSignals(False)
            self._syncing_window_controls = False

    def _estimate_auto_sf(self, trace_idx: int, sampling_rate: Optional[float], trace_gain: float) -> float:
        """为 iscale=1 且 sf=0 估计稳定 sf（避免随视窗跳变）。"""
        if self.loaded is None:
            return 0.0
        traces = self.loaded.get("traces", [])
        offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
        if trace_idx < 0 or trace_idx >= len(traces) or trace_idx >= offsets.size:
            return 0.0
        tr = np.asarray(traces[int(trace_idx)], dtype=np.float64).copy()
        if tr.size == 0:
            return 0.0
        if int(getattr(self.params, "iout", 0)) != 2:
            tr = tr - np.mean(tr)
        if int(self.params.ibndps) != 0 and sampling_rate is not None and sampling_rate > 0:
            try:
                tr = self.processor.apply_bandpass_filter(
                    tr,
                    self.params.freqlo,
                    self.params.freqhi,
                    self.params.npoles,
                    self.params.izerop,
                    float(sampling_rate),
                ).astype(np.float64, copy=False)
            except Exception:
                pass
        ampmax_ref = float(np.max(np.abs(tr))) if tr.size > 0 else 0.0
        if ampmax_ref <= 1e-20:
            return 0.0
        off = abs(float(offsets[int(trace_idx)]))
        off_for_pow = max(off * 10.0, 1e-6)
        denom = ampmax_ref * max(1.0, float(trace_gain)) * (off_for_pow ** max(0.0, float(self.params.rcor)))
        if denom <= 1e-20:
            return 0.0
        return float(self.params.amp) / denom

    def _build_processing_params(self) -> ZPlotParameters:
        """按当前开关生成处理参数（滤波/增益可独立关闭）。"""
        proc_params = ZPlotParameters()
        try:
            proc_params.from_dict(self.params.to_dict())
        except Exception:
            # 兜底：保留默认参数
            pass

        if not bool(self.chk_filter.isChecked()):
            proc_params.ibndps = 0

        if not bool(self.chk_gain.isChecked()):
            # 关闭增益：跳过 iscale/rcor/sf/tvg/pvg/clip 的增益通道
            proc_params.iscale = 0
            proc_params.amp = 1.0
            proc_params.rcor = 0.0
            proc_params.sf = 0.0
            proc_params.tvg = 1.0
            proc_params.pvg = 1.0
            proc_params.clip = 0.0

        return proc_params

    def _collect_ui_parameters(self) -> Dict[str, object]:
        return {
            "irec": int(self.spin_irec.value()),
            "itype": int(self.combo_itype.currentIndex()),
            "nskip": int(self.spin_nskip.value()),
            "ndecim": int(self.spin_ndecim.value()),
            "vred": float(self.spin_vred.value()),
            "xmin": float(self.spin_xmin.value()),
            "xmax": float(self.spin_xmax.value()),
            "tmin": float(self.spin_tmin.value()),
            "tmax": float(self.spin_tmax.value()),
            "amp": float(self.spin_amp.value()),
            "iscale": int(self.combo_iscale.currentIndex()),
            "rcor": float(self.spin_rcor.value()),
            "sf": float(self.spin_sf.value()),
            "tvg": float(self.spin_tvg.value()),
            "pvg": float(self.spin_pvg.value()),
            "clip": float(self.spin_clip.value()),
            "dscale": float(self.spin_dscale.value()),
            "filter_on": bool(self.chk_filter.isChecked()),
            "gain_on": bool(self.chk_gain.isChecked()),
            "freqlo": float(self.spin_freqlo.value()),
            "freqhi": float(self.spin_freqhi.value()),
            "npoles": int(self.spin_npoles.value()),
            "zerop": bool(self.chk_zerop.isChecked()),
            "denoise_enabled": bool(self.chk_denoise_enabled.isChecked()),
            "denoise_ab_raw": (not bool(self.chk_denoise_ab_raw.isChecked())),
            "denoise_show_diff": bool(self.chk_denoise_show_diff.isChecked()),
            "denoise_diff_gain": float(self.combo_denoise_diff_gain.currentData() or 1.0),
            "denoise_scope": str(self.combo_denoise_scope.currentData() or "rendered"),
            "denoise_select_mode": str(self.combo_denoise_scope.currentData() or "rendered") == "selected",
            "denoise_selected_traces": sorted(int(i) for i in self._denoise_selected_traces),
            "denoise_f_s": float(self.spin_denoise_f_s.value()),
            "denoise_f_e": float(self.spin_denoise_f_e.value()),
            "denoise_strength": float(self.spin_denoise_strength.value()),
            "denoise_bwconn": int(self.combo_denoise_bwconn.currentText()),
            "denoise_workers": int(self.spin_denoise_workers.value()),
            "denoise_coh_win": int(self._denoise_params.get("coh_win", 11)),
            "denoise_coh_lag": int(self._denoise_params.get("coh_lag", 2)),
            "denoise_coh_thr": float(self._denoise_params.get("coh_thr", 0.55)),
            "denoise_coh_blend": float(self._denoise_params.get("coh_blend", 0.35)),
            "denoise_coh_penalty": float(self._denoise_params.get("coh_penalty", 0.08)),
            "denoise_perf_diag": bool(self._denoise_params.get("perf_diag", False)),
            "denoise_morph_enable": bool(self._denoise_params.get("morph_enable", True)),
            "denoise_morph_preset": str(self._denoise_params.get("morph_preset", "balanced")),
            "denoise_morph_quantile": float(self._denoise_params.get("morph_quantile", 0.70)),
            "denoise_morph_min_area": int(self._denoise_params.get("morph_min_area", 24)),
            "denoise_morph_expand": int(self._denoise_params.get("morph_expand", 1)),
            "denoise_morph_floor_ratio": float(self._denoise_params.get("morph_floor_ratio", 0.03)),
            "denoise_morph_keep_strong_q": float(self._denoise_params.get("morph_keep_strong_q", 0.95)),
            "denoise_pick_guidance": bool(self.chk_denoise_pick_guidance.isChecked()),
            "denoise_pick_wavelet_length": float(self.spin_denoise_pick_hw.value()),
            "denoise_pick_floor": float(self.spin_denoise_pick_floor.value()),
            "denoise_return_debug": False,
            "denoise_return_result": False,
            "mode": int(self.combo_mode.currentIndex()),
            "rt_shade": bool(self.chk_rt_shade.isChecked()),
            "pick_mode": bool(self.chk_pick_mode.isChecked()),
            "apick": int(self.spin_apick.value()),
            "pick_size": int(self.spin_pick_size.value()),
            "tcrcor": float(self.spin_tcrcor.value()),
            "tlag": float(self.spin_tlag.value()),
            "hilbratio": float(self.spin_hilbratio.value()),
            "show_stack": bool(self.chk_show_stack.isChecked()),
            "orientation_ui_params": dict(self._orientation_ui_params),
            "orientation_current_solution": dict(self._orientation_current_solution),
        }

    def _apply_ui_parameters(self, conf: Dict[str, object]) -> None:
        if "irec" in conf:
            self.spin_irec.setValue(max(0, int(conf["irec"])))
        if "itype" in conf:
            self.combo_itype.setCurrentIndex(max(0, min(4, int(conf["itype"]))))
        if "nskip" in conf:
            self.spin_nskip.setValue(max(0, int(conf["nskip"])))
        if "ndecim" in conf:
            self.spin_ndecim.setValue(max(1, int(conf["ndecim"])))
        if "vred" in conf:
            self.spin_vred.setValue(max(0.0, float(conf["vred"])))
        if "xmin" in conf:
            self.spin_xmin.setValue(float(conf["xmin"]))
        if "xmax" in conf:
            self.spin_xmax.setValue(float(conf["xmax"]))
        if "tmin" in conf:
            self.spin_tmin.setValue(float(conf["tmin"]))
        if "tmax" in conf:
            self.spin_tmax.setValue(float(conf["tmax"]))
        if "amp" in conf:
            self.spin_amp.setValue(float(conf["amp"]))
        if "iscale" in conf:
            self.combo_iscale.setCurrentIndex(max(0, min(2, int(conf["iscale"]))))
        if "rcor" in conf:
            self.spin_rcor.setValue(float(conf["rcor"]))
        if "sf" in conf:
            self.spin_sf.setValue(max(0.0, float(conf["sf"])))
        if "tvg" in conf:
            self.spin_tvg.setValue(float(conf["tvg"]))
        if "pvg" in conf:
            self.spin_pvg.setValue(float(conf["pvg"]))
        if "clip" in conf:
            self.spin_clip.setValue(max(0.0, float(conf["clip"])))
        if "dscale" in conf:
            self.spin_dscale.setValue(max(0.2, min(5.0, float(conf["dscale"]))))
        if "filter_on" in conf:
            self.chk_filter.setChecked(bool(conf["filter_on"]))
        if "gain_on" in conf:
            self.chk_gain.setChecked(bool(conf["gain_on"]))
        if "freqlo" in conf:
            self.spin_freqlo.setValue(float(conf["freqlo"]))
        if "freqhi" in conf:
            self.spin_freqhi.setValue(float(conf["freqhi"]))
        if "npoles" in conf:
            self.spin_npoles.setValue(max(1, int(conf["npoles"])))
        if "zerop" in conf:
            self.chk_zerop.setChecked(bool(conf["zerop"]))
        if "denoise_enabled" in conf:
            self.chk_denoise_enabled.setChecked(bool(conf["denoise_enabled"]))
        if "denoise_ab_raw" in conf:
            # 兼容字段语义：denoise_ab_raw=True 表示原始(A)
            self.chk_denoise_ab_raw.setChecked(not bool(conf["denoise_ab_raw"]))
        if "denoise_show_diff" in conf:
            self.chk_denoise_show_diff.setChecked(bool(conf["denoise_show_diff"]))
        if "denoise_diff_gain" in conf:
            try:
                gain_v = float(conf["denoise_diff_gain"])
            except Exception:
                gain_v = 1.0
            idx_gain = self.combo_denoise_diff_gain.findData(gain_v)
            if idx_gain < 0:
                idx_gain = 0
            self.combo_denoise_diff_gain.setCurrentIndex(idx_gain)
        if "denoise_scope" in conf:
            scope = str(conf["denoise_scope"]).strip().lower()
            idx_scope = self.combo_denoise_scope.findData(scope)
            if idx_scope >= 0:
                self.combo_denoise_scope.setCurrentIndex(idx_scope)
        # 兼容旧字段 denoise_select_mode：当前由 scope=selected 决定选道模式，忽略该字段
        if "denoise_selected_traces" in conf:
            val_sel = conf.get("denoise_selected_traces", [])
            if isinstance(val_sel, (list, tuple)):
                self._denoise_selected_traces = {int(v) for v in val_sel if isinstance(v, (int, float))}
        if "denoise_f_s" in conf:
            self.spin_denoise_f_s.setValue(max(0.0, float(conf["denoise_f_s"])))
        if "denoise_f_e" in conf:
            self.spin_denoise_f_e.setValue(max(0.1, float(conf["denoise_f_e"])))
        if "denoise_strength" in conf:
            self.spin_denoise_strength.setValue(max(0.01, float(conf["denoise_strength"])))
        if "denoise_bwconn" in conf:
            bw = int(conf["denoise_bwconn"])
            self.combo_denoise_bwconn.setCurrentText("4" if bw == 4 else "8")
        if "denoise_workers" in conf:
            self.spin_denoise_workers.setValue(max(1, int(conf["denoise_workers"])))
        if "denoise_coh_win" in conf:
            self._denoise_params["coh_win"] = int(max(5, int(conf["denoise_coh_win"])))
        if "denoise_coh_lag" in conf:
            self._denoise_params["coh_lag"] = int(max(0, int(conf["denoise_coh_lag"])))
        if "denoise_coh_thr" in conf:
            self._denoise_params["coh_thr"] = float(min(0.95, max(0.05, float(conf["denoise_coh_thr"]))))
        if "denoise_coh_blend" in conf:
            self._denoise_params["coh_blend"] = float(min(0.90, max(0.0, float(conf["denoise_coh_blend"]))))
        if "denoise_coh_penalty" in conf:
            self._denoise_params["coh_penalty"] = float(min(0.50, max(0.0, float(conf["denoise_coh_penalty"]))))
        if "denoise_perf_diag" in conf:
            self._denoise_params["perf_diag"] = bool(conf["denoise_perf_diag"])
        if "denoise_morph_enable" in conf:
            self._denoise_params["morph_enable"] = bool(conf["denoise_morph_enable"])
        if "denoise_morph_preset" in conf:
            v = str(conf["denoise_morph_preset"]).strip().lower()
            if v in ("conservative", "balanced", "strong"):
                self._denoise_params["morph_preset"] = v
        if "denoise_morph_quantile" in conf:
            self._denoise_params["morph_quantile"] = float(min(0.98, max(0.05, float(conf["denoise_morph_quantile"]))))
        if "denoise_morph_min_area" in conf:
            self._denoise_params["morph_min_area"] = int(max(1, int(conf["denoise_morph_min_area"])))
        if "denoise_morph_expand" in conf:
            self._denoise_params["morph_expand"] = int(min(6, max(0, int(conf["denoise_morph_expand"]))))
        if "denoise_morph_floor_ratio" in conf:
            self._denoise_params["morph_floor_ratio"] = float(min(0.50, max(0.0, float(conf["denoise_morph_floor_ratio"]))))
        if "denoise_morph_keep_strong_q" in conf:
            self._denoise_params["morph_keep_strong_q"] = float(min(0.999, max(0.05, float(conf["denoise_morph_keep_strong_q"]))))
        if "denoise_pick_guidance" in conf:
            self.chk_denoise_pick_guidance.setChecked(bool(conf["denoise_pick_guidance"]))
        if "denoise_pick_wavelet_length" in conf:
            self.spin_denoise_pick_hw.setValue(float(max(0.02, min(5.0, float(conf["denoise_pick_wavelet_length"])))))
        elif "denoise_pick_half_width" in conf:
            sig_old = float(conf["denoise_pick_half_width"])
            twl = float(sig_old) * (2.0 * math.sqrt(2.0 * math.log(2.0)))
            self.spin_denoise_pick_hw.setValue(float(max(0.02, min(5.0, twl))))
        if "denoise_pick_floor" in conf:
            self.spin_denoise_pick_floor.setValue(float(min(0.95, max(0.0, float(conf["denoise_pick_floor"])))))
        # 兼容旧配置字段：保留读取但忽略 debug/result UI（已移除）
        self._sync_denoise_params_from_ui()
        if "mode" in conf:
            self.combo_mode.setCurrentIndex(max(0, min(2, int(conf["mode"]))))
        if "rt_shade" in conf:
            self.chk_rt_shade.setChecked(bool(conf["rt_shade"]))
        if "pick_mode" in conf:
            self.chk_pick_mode.setChecked(bool(conf["pick_mode"]))
        if "apick" in conf:
            apick = int(conf["apick"])
            apick = max(1, min(int(self.spin_apick.maximum()), apick))
            self.spin_apick.setValue(apick)
        if "pick_size" in conf:
            psize = max(2, min(40, int(conf["pick_size"])))
            self.spin_pick_size.setValue(psize)
        if "tcrcor" in conf:
            self.spin_tcrcor.setValue(float(conf["tcrcor"]))
        if "tlag" in conf:
            self.spin_tlag.setValue(float(conf["tlag"]))
        if "hilbratio" in conf:
            self.spin_hilbratio.setValue(float(conf["hilbratio"]))
        if "show_stack" in conf:
            self.chk_show_stack.setChecked(bool(conf["show_stack"]))
        if "orientation_ui_params" in conf:
            val = conf["orientation_ui_params"]
            if isinstance(val, dict):
                for k in ("wave_pre", "wave_post", "att_iter", "att_wtt", "att_wpol", "att_wsym"):
                    if k in val:
                        try:
                            self._orientation_ui_params[k] = float(val[k])
                        except Exception:
                            pass
        if "orientation_current_solution" in conf:
            val = conf["orientation_current_solution"]
            if isinstance(val, dict):
                for k in ("azimuth_deg", "tilt_deg", "dx", "dy", "dz", "time_shift_sec", "objective", "accepted"):
                    if k in val:
                        try:
                            self._orientation_current_solution[k] = float(val[k])
                        except Exception:
                            pass
        # V段与叠加校正基准由“波形操作”面板独立保存/加载，不与参数文件混用。

    def _save_parameters(self) -> None:
        out, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存参数配置", "", "JSON files (*.json);;All files (*)", options=self._file_dialog_options()
        )
        if not out:
            return
        payload = {
            "version": 1,
            "dfile": self._dfile,
            "hfile": self._hfile,
            "rfile": self._rfile,
            "parameters": self._collect_ui_parameters(),
        }
        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self.lbl_status.setText(f"参数已保存：{out}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "保存失败", f"参数保存失败：{exc}")

    def _load_parameters(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "加载参数配置", "", "JSON files (*.json);;All files (*)", options=self._file_dialog_options()
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            conf = payload.get("parameters", payload)
            if not isinstance(conf, dict):
                raise ValueError("参数配置格式不正确")
            self._apply_ui_parameters(conf)
            if self.loaded is not None:
                self.request_render(immediate=True)
            self.lbl_status.setText(f"参数已加载：{path}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "加载失败", f"参数加载失败：{exc}")

    def _save_waveop_state(self) -> None:
        if self.loaded is None:
            self._show_themed_info("保存V段", "请先加载数据后再保存V段。")
            return
        out, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "保存V段与校正基准",
            "",
            "WaveOp JSON (*.waveop.json *.json);;All files (*)",
            options=self._file_dialog_options(),
        )
        if not out:
            return
        wave_corr = [
            {
                "trace_idx": int(k[0]),
                "pick_word": int(k[1]),
                "t_true": float(v),
            }
            for k, v in sorted(self._waveop_corrected_ttrue.items(), key=lambda kv: (int(kv[0][1]), int(kv[0][0])))
        ]
        payload = {
            "version": 1,
            "kind": "waveop_state",
            "dfile": self._dfile,
            "hfile": self._hfile,
            "rfile": self._rfile,
            "waveform_selections": [dict(s) for s in self.waveform_selections],
            "waveop_corrected_ttrue": wave_corr,
        }
        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self.lbl_status.setText(f"V段已保存：{out} {self._waveop_apick_status_suffix()}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "保存失败", f"V段保存失败：{exc}")

    def _load_waveop_state(self) -> None:
        if self.loaded is None:
            self._show_themed_info("加载V段", "请先加载数据后再加载V段。")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "加载V段与校正基准",
            "",
            "WaveOp JSON (*.waveop.json *.json);;All files (*)",
            options=self._file_dialog_options(),
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and str(payload.get("kind", "")).strip() not in ("", "waveop_state"):
                raise ValueError("文件类型不是V段状态文件")
            restored: List[Dict[str, float]] = []
            val = payload.get("waveform_selections", [])
            if isinstance(val, list):
                for item in val:
                    if not isinstance(item, dict):
                        continue
                    try:
                        restored.append(
                            {
                                "trace_idx": float(item.get("trace_idx", -1)),
                                "offset": float(item.get("offset", 0.0)),
                                "t_display": float(item.get("t_display", 0.0)),
                                "t_true": float(item.get("t_true", 0.0)),
                                "pick_word": float(item.get("pick_word", int(self.spin_apick.value()))),
                            }
                        )
                    except Exception:
                        continue
            restored_corr: Dict[Tuple[int, int], float] = {}
            valc = payload.get("waveop_corrected_ttrue", [])
            if isinstance(valc, list):
                for item in valc:
                    if not isinstance(item, dict):
                        continue
                    try:
                        key = (int(item.get("trace_idx", -1)), int(item.get("pick_word", 1)))
                        tval = float(item.get("t_true", np.nan))
                        if key[0] >= 0 and np.isfinite(tval):
                            restored_corr[key] = tval
                    except Exception:
                        continue
            if (self.waveform_selections or self._waveop_corrected_ttrue) and (restored or restored_corr):
                ans = QtWidgets.QMessageBox.question(
                    self,
                    "加载V段",
                    "将覆盖当前V段与叠加校正基准，是否继续？",
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No,
                )
                if ans != QtWidgets.QMessageBox.StandardButton.Yes:
                    return
            self.waveform_selections = restored
            self._waveop_corrected_ttrue = restored_corr
            self._refresh_waveop_selection_list()
            self.request_render(delay_ms=10)
            self.lbl_status.setText(f"V段已加载：{path} {self._waveop_apick_status_suffix()}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "加载失败", f"V段加载失败：{exc}")

    def _show_data_info(self) -> None:
        if self.loaded is None:
            self._show_themed_info("Data Info", "当前尚未加载数据。")
            return
        header = self.loaded.get("header")
        traces = self.loaded.get("traces", [])
        offsets = self.loaded.get("offsets", [])
        times = self.loaded.get("times", [])
        trace_headers = self.loaded.get("trace_headers", []) or []
        picks_count = 0
        if self.pick_manager is not None:
            picks_count = int(self.pick_manager.count_picks())

        # 文件信息
        def _file_size_text(path: Optional[str]) -> str:
            if not path:
                return "-"
            try:
                p = Path(path)
                if not p.exists():
                    return "不存在"
                sz = float(p.stat().st_size)
                if sz < 1024:
                    return f"{int(sz)} B"
                if sz < 1024 * 1024:
                    return f"{sz / 1024.0:.1f} KB"
                if sz < 1024 * 1024 * 1024:
                    return f"{sz / (1024.0 * 1024.0):.2f} MB"
                return f"{sz / (1024.0 * 1024.0 * 1024.0):.2f} GB"
            except Exception:
                return "未知"

        # 采样与范围信息
        dt_ms = 0.0
        if len(times) >= 2:
            try:
                dt_ms = (float(times[1]) - float(times[0])) * 1000.0
            except Exception:
                dt_ms = 0.0
        tmin = float(np.min(times)) if len(times) > 0 else 0.0
        tmax = float(np.max(times)) if len(times) > 0 else 0.0
        xmin = float(np.min(offsets)) if len(offsets) > 0 else 0.0
        xmax = float(np.max(offsets)) if len(offsets) > 0 else 0.0

        # 道头统计
        rec_ids = sorted(
            {
                int(getattr(th, "ishoti", 0) or 0)
                for th in trace_headers
                if int(getattr(th, "ishoti", 0) or 0) > 0
            }
        )
        dead_count = 0
        comp_counter: Dict[int, int] = {}
        for th in trace_headers:
            try:
                if int(getattr(th, "iflagi", 1) or 1) != 1:
                    dead_count += 1
            except Exception:
                pass
            c = int(getattr(th, "itypei", 0) or 0)
            comp_counter[c] = int(comp_counter.get(c, 0)) + 1

        comp_name = {0: "全部/未标注", 1: "垂直", 2: "径向", 3: "横向", 4: "水听器"}
        comp_text = ", ".join(
            f"{comp_name.get(k, str(k))}:{v}" for k, v in sorted(comp_counter.items(), key=lambda kv: kv[0])
        ) or "-"

        # 拾取统计
        pick_word_max = int(getattr(header, "npick", 0) or 0)
        active_word = int(self.spin_apick.value()) if hasattr(self, "spin_apick") else 1
        active_word_count = 0
        if self.pick_manager is not None:
            by_word = self.pick_manager.get_picks_by_word(active_word)
            active_word_count = len(by_word)

        # 头信息中的真实参数（来自数据，不依赖当前界面控件）
        def _header_value(name: str) -> str:
            if header is None:
                return "-"
            try:
                v = getattr(header, name)
                if v is None:
                    return "-"
                if isinstance(v, float):
                    return f"{v:.6g}"
                return str(v)
            except Exception:
                return "-"

        msg = (
            "[文件]\n"
            f"dfile: {self._dfile or '-'}\n"
            f"  大小: {_file_size_text(self._dfile)}\n"
            f"hfile: {self._hfile or '-'}\n"
            f"  大小: {_file_size_text(self._hfile)}\n"
            f"rfile: {self._rfile or '-'}\n"
            f"  大小: {_file_size_text(self._rfile)}\n\n"
            "[数据规模]\n"
            f"ntraces(header): {int(getattr(header, 'ntraces', 0) or 0)}\n"
            f"npts(header): {int(getattr(header, 'npts', 0) or 0)}\n"
            f"npick(header): {pick_word_max}\n"
            f"trace arrays: {len(traces)}\n"
            f"offset count: {len(offsets)}\n"
            f"time samples: {len(times)}\n\n"
            "[采样与范围]\n"
            f"采样间隔 dt: {dt_ms:.3f} ms\n"
            f"时间范围: [{tmin:.4f}, {tmax:.4f}] s\n"
            f"偏移范围: [{xmin:.4f}, {xmax:.4f}] km\n\n"
            "[道头统计]\n"
            f"记录(shot)数量: {len(rec_ids)}\n"
            f"记录号范围: {rec_ids[0] if rec_ids else '-'} ~ {rec_ids[-1] if rec_ids else '-'}\n"
            f"死道数量: {dead_count}\n"
            f"分量统计: {comp_text}\n\n"
            "[拾取统计]\n"
            f"全部拾取点: {picks_count}\n"
            f"当前活动字(apick={active_word})拾取道数: {active_word_count}\n\n"
            "[数据头真实参数]\n"
            f"vredf(header): {_header_value('vredf')}\n"
            f"nrec(header): {_header_value('nrec')}\n"
            f"ntraces(header): {_header_value('ntraces')}\n"
            f"npts(header): {_header_value('npts')}\n"
            f"npick(header): {_header_value('npick')}\n"
            f"f1(header): {_header_value('f1')}\n"
            f"dt(header): {_header_value('dt')}"
        )
        self._show_themed_info("Data Info", msg)

    def _show_coordinate_parameters(self) -> None:
        if self.loaded is None:
            self._show_themed_info("道头参数", "当前尚未加载数据。")
            return
        header = self.loaded.get("header")
        trace_headers = self.loaded.get("trace_headers", []) or []

        # 字段说明与单位（未知字段会自动给默认描述）
        field_desc: Dict[str, str] = {
            # .z 文件头（52字节）字段说明（按 data_loader.ZFormatHeader / su2z_hhb 写入顺序）
            "ntraces": "总道数（文件中道记录数量）",
            "npts": "每道采样点数",
            "sint": "采样间隔（微秒）",
            "tstart": "起始时间（毫秒）",
            "tend": "结束时间（毫秒；若<=0可由npts与sint推算）",
            "nrec": "记录数（炮集数）",
            "npick": "每道拾取字数量（最大40）",
            "vredf": "折合速度（km/s）",
            "ifmt": "道数据格式标识（1=float32, 0=int16）",
            "xlatlong": "经纬度缩放因子（头参数）",
            "xelev": "高程缩放因子（头参数）",
            "xutm": "UTM坐标缩放因子（头参数）",
            "cm": "坐标参考参数（转换程序中默认0.0，常作保留/扩展字段）",
            "nreci": "记录号（标准化后记录索引）",
            "itsn": "记录内道序号",
            "ireci": "接收站号",
            "itypei": "分量类型编号（1垂直/2径向/3横向/4水听器）",
            "iflagi": "道有效标志（1有效）",
            "offsti": "炮检距",
            "azi": "方位角",
            "igaini": "道增益因子（来自道头）",
            "texact": "精确时间修正项",
            "slat": "震源纬度",
            "slong": "震源经度",
            "selev": "震源高程",
            "swdepth": "震源水深",
            "rlat": "接收点纬度",
            "rlong": "接收点经度",
            "relev": "接收点高程",
            "sxutm": "震源UTM X",
            "syutm": "震源UTM Y",
            "sz": "震源深度/高程Z",
            "rxutm": "接收点UTM X",
            "ryutm": "接收点UTM Y",
            "rz": "接收点深度/高程Z",
            "ishoti": "炮号（shot id）",
            "picks": "该道各拾取字走时数组",
        }
        field_unit: Dict[str, str] = {
            "ntraces": "-",
            "npts": "-",
            "sint": "us",
            "tstart": "ms",
            "tend": "ms",
            "nrec": "-",
            "npick": "-",
            "offsti": "km",
            "azi": "deg",
            "slat": "deg",
            "slong": "deg",
            "rlat": "deg",
            "rlong": "deg",
            "selev": "m",
            "swdepth": "m",
            "relev": "m",
            "sxutm": "m",
            "syutm": "m",
            "sz": "m",
            "rxutm": "m",
            "ryutm": "m",
            "rz": "m",
            "vredf": "km/s",
        }

        def _header_summary(value: object) -> str:
            if value is None:
                return "-"
            if isinstance(value, float):
                return f"{value:.6g}"
            return str(value)

        def _value_summary(values: List[object]) -> str:
            if len(values) == 0:
                return "-"
            numeric_vals: List[float] = []
            seq_count = 0
            seq_nonzero = 0
            for v in values:
                if isinstance(v, (list, tuple, np.ndarray)):
                    seq_count += 1
                    arr = np.asarray(v, dtype=float) if len(v) > 0 else np.asarray([], dtype=float)
                    if arr.size > 0:
                        seq_nonzero += int(np.sum(np.abs(arr) > 1e-12))
                    continue
                try:
                    numeric_vals.append(float(v))
                except Exception:
                    pass
            if numeric_vals:
                arr = np.asarray(numeric_vals, dtype=float)
                finite = arr[np.isfinite(arr)]
                if finite.size > 0:
                    nz = int(np.sum(np.abs(finite) > 1e-12))
                    return (
                        f"min={float(np.min(finite)):.6g}, "
                        f"max={float(np.max(finite)):.6g}, "
                        f"mean={float(np.mean(finite)):.6g}, "
                        f"非零={nz}/{int(finite.size)}"
                    )
            if seq_count > 0:
                lengths = [len(v) for v in values if isinstance(v, (list, tuple, np.ndarray))]
                avg_len = float(np.mean(lengths)) if lengths else 0.0
                return f"序列字段: 道数={seq_count}, 平均长度={avg_len:.1f}, 非零总数={seq_nonzero}"
            # 非数值字段：显示去重样本
            uniq = []
            for v in values:
                s = str(v)
                if s not in uniq:
                    uniq.append(s)
                if len(uniq) >= 6:
                    break
            return f"样本值: {', '.join(uniq)}"

        rows: List[Tuple[str, str, str, str, str]] = []

        # 文件头字段：严格按 .z 52字节头结构展示，避免混入派生/无关属性
        if header is not None:
            header_keys = [
                "ntraces", "npts", "sint", "tstart", "tend", "nrec", "npick",
                "vredf", "ifmt", "xlatlong", "xelev", "xutm", "cm"
            ]
            for key in header_keys:
                try:
                    v = getattr(header, key)
                except Exception:
                    v = None
                desc = field_desc.get(key, "文件头参数")
                unit = field_unit.get(key, "-")
                rows.append(("文件头(.z)", key, unit, desc, _header_summary(v)))

        # 所有道头字段（全量）
        header_field_values: Dict[str, List[object]] = {}
        for th in trace_headers:
            try:
                th_items = vars(th).items()
            except Exception:
                th_items = []
                for k in dir(th):
                    if str(k).startswith("_"):
                        continue
                    try:
                        v = getattr(th, k)
                    except Exception:
                        continue
                    if callable(v):
                        continue
                    th_items.append((k, v))
            for k, v in th_items:
                if str(k).startswith("_") or callable(v):
                    continue
                header_field_values.setdefault(str(k), []).append(v)

        for name in sorted(header_field_values.keys()):
            vals = header_field_values[name]
            desc = field_desc.get(name, "道头参数（自动识别）")
            unit = field_unit.get(name, "-")
            rows.append(("道头(.hdr/.z)", name, unit, desc, _value_summary(vals)))

        if self._coord_params_dialog is not None:
            try:
                self._coord_params_dialog.close()
            except Exception:
                pass
            self._coord_params_dialog = None

        dialog = QtWidgets.QDialog(self)
        self._coord_params_dialog = dialog
        dialog.setWindowTitle("道头参数列表")
        dialog.resize(1220, 650)
        dialog.setModal(False)
        dialog.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.destroyed.connect(lambda *_: setattr(self, "_coord_params_dialog", None))
        layout = QtWidgets.QVBoxLayout(dialog)
        info = QtWidgets.QLabel("以下为 .z/.hdr 的文件头参数与全部道头参数统计（含参数说明）。", dialog)
        info.setWordWrap(True)
        layout.addWidget(info)

        table = QtWidgets.QTableWidget(len(rows), 5, dialog)
        table.setHorizontalHeaderLabels(["来源", "参数名", "单位", "说明", "值/统计"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        table.setAlternatingRowColors(True)
        for r, (src, name, unit, desc, val) in enumerate(rows):
            table.setItem(r, 0, QtWidgets.QTableWidgetItem(src))
            table.setItem(r, 1, QtWidgets.QTableWidgetItem(name))
            table.setItem(r, 2, QtWidgets.QTableWidgetItem(unit))
            table.setItem(r, 3, QtWidgets.QTableWidgetItem(desc))
            table.setItem(r, 4, QtWidgets.QTableWidgetItem(val))
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(table, stretch=1)

        close_btn = QtWidgets.QPushButton("关闭", dialog)
        close_btn.clicked.connect(dialog.accept)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close_btn)
        layout.addLayout(row)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _show_location_map(self) -> None:
        if self.loaded is None:
            self._show_themed_info("位置Map", "当前尚未加载数据。")
            return
        trace_headers = self.loaded.get("trace_headers", []) or []
        if not trace_headers:
            self._show_themed_info("位置Map", "当前数据缺少道头，无法绘制位置分布。")
            return

        def _valid_xy(x: float, y: float) -> bool:
            return math.isfinite(x) and math.isfinite(y) and (abs(x) > 1e-9 or abs(y) > 1e-9)

        src_x: List[float] = []
        src_y: List[float] = []
        rec_x: List[float] = []
        rec_y: List[float] = []
        mode_utm_count = 0
        mode_geo_count = 0
        trace_points: List[Tuple[float, float]] = []
        trace_indices: List[int] = []

        use_utm = False
        for th in trace_headers:
            sxutm = float(getattr(th, "sxutm", 0.0) or 0.0)
            syutm = float(getattr(th, "syutm", 0.0) or 0.0)
            rxutm = float(getattr(th, "rxutm", 0.0) or 0.0)
            ryutm = float(getattr(th, "ryutm", 0.0) or 0.0)
            if _valid_xy(sxutm, syutm) or _valid_xy(rxutm, ryutm):
                use_utm = True
                break

        # 自动判别“哪组坐标代表接收点(随道变化)”：
        # - 若 sx/sy 变化更大，则接收点来自 sx/sy（用户当前数据场景）
        # - 否则接收点来自 rx/ry（常见场景）
        sx_series: List[Tuple[float, float]] = []
        rx_series: List[Tuple[float, float]] = []
        for th in trace_headers:
            if use_utm:
                sxv = float(getattr(th, "sxutm", 0.0) or 0.0)
                syv = float(getattr(th, "syutm", 0.0) or 0.0)
                rxv = float(getattr(th, "rxutm", 0.0) or 0.0)
                ryv = float(getattr(th, "ryutm", 0.0) or 0.0)
            else:
                sxv = float(getattr(th, "slong", 0.0) or 0.0)
                syv = float(getattr(th, "slat", 0.0) or 0.0)
                rxv = float(getattr(th, "rlong", 0.0) or 0.0)
                ryv = float(getattr(th, "rlat", 0.0) or 0.0)
            if _valid_xy(sxv, syv):
                sx_series.append((sxv, syv))
            if _valid_xy(rxv, ryv):
                rx_series.append((rxv, ryv))

        def _var_count(series: List[Tuple[float, float]], decimals: int) -> int:
            if not series:
                return 0
            uniq = {(round(float(x), decimals), round(float(y), decimals)) for x, y in series}
            return len(uniq)

        dec = 2 if use_utm else 6
        s_var = _var_count(sx_series, dec)
        r_var = _var_count(rx_series, dec)
        self._location_map_rec_role = "sx" if s_var >= r_var else "rx"

        for trace_idx, th in enumerate(trace_headers):
            px, py = self._trace_position_for_map(trace_idx, mode="utm" if use_utm else "geo")
            if _valid_xy(px, py):
                trace_points.append((px, py))
                trace_indices.append(int(trace_idx))
                rec_x.append(px)
                rec_y.append(py)
            if use_utm:
                if self._location_map_rec_role == "sx":
                    sx = float(getattr(th, "rxutm", 0.0) or 0.0)
                    sy = float(getattr(th, "ryutm", 0.0) or 0.0)
                else:
                    sx = float(getattr(th, "sxutm", 0.0) or 0.0)
                    sy = float(getattr(th, "syutm", 0.0) or 0.0)
                if _valid_xy(sx, sy):
                    src_x.append(sx)
                    src_y.append(sy)
                    mode_utm_count += 1
            else:
                if self._location_map_rec_role == "sx":
                    sx = float(getattr(th, "rlong", 0.0) or 0.0)
                    sy = float(getattr(th, "rlat", 0.0) or 0.0)
                else:
                    sx = float(getattr(th, "slong", 0.0) or 0.0)
                    sy = float(getattr(th, "slat", 0.0) or 0.0)
                if _valid_xy(sx, sy):
                    src_x.append(sx)
                    src_y.append(sy)
                    mode_geo_count += 1

        sx = np.asarray(src_x, dtype=float)
        sy = np.asarray(src_y, dtype=float)
        rx = np.asarray(rec_x, dtype=float)
        ry = np.asarray(rec_y, dtype=float)
        smode = "utm" if use_utm and mode_utm_count > 0 else ("geo" if (not use_utm and mode_geo_count > 0) else "none")
        rmode = "utm" if use_utm else "geo"

        if sx.size == 0 and rx.size == 0:
            self._show_themed_info("位置Map", "未检测到有效震源/接收点坐标。")
            return

        if self._location_map_dialog is not None:
            try:
                self._location_map_dialog.close()
            except Exception:
                pass
            self._location_map_dialog = None

        # 使用独立窗口（无 parent）确保绝不阻塞主 GUI
        dialog = QtWidgets.QDialog(None)
        self._location_map_dialog = dialog
        dialog.setWindowTitle("位置Map（震源/接收点）")
        dialog.resize(920, 640)
        dialog.setWindowFlag(QtCore.Qt.WindowType.Window, True)
        dialog.setModal(False)
        dialog.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        def _on_destroyed(*_):
            self._location_map_dialog = None
            self._location_map_cursor_item = None
            self._location_map_selected_item = None
            self._location_map_plot_item = None
            self._location_map_plot_widget = None
            self._location_map_base_bounds = None
            self._location_map_terrain_item = None
            self._location_map_terrain_cache_key = None
            self._location_map_colorbar_gradient = None
            self._location_map_colorbar_min_label = None
            self._location_map_colorbar_max_label = None
            self._location_map_terrain_proj_label = None
            self._location_map_trace_points = np.empty((0, 2), dtype=float)
            self._location_map_trace_indices = np.empty((0,), dtype=int)
            self._location_map_mode = "none"
            self._location_map_rec_role = "auto"
            self._map_link_trace_idx = None
            if self.loaded is not None:
                self.request_render(delay_ms=10)
        dialog.destroyed.connect(_on_destroyed)
        lay = QtWidgets.QVBoxLayout(dialog)

        note = QtWidgets.QLabel(
            f"震源点: {int(sx.size)}（{smode}）  |  接收点: {int(rx.size)}（{rmode}）",
            dialog,
        )
        lay.addWidget(note)
        terrain_row = QtWidgets.QHBoxLayout()
        btn_load_terrain = QtWidgets.QPushButton("加载地形", dialog)
        btn_clear_terrain = QtWidgets.QPushButton("清除地形", dialog)
        btn_dump_terrain = QtWidgets.QPushButton("查看转换样本", dialog)
        btn_save_map = QtWidgets.QPushButton("保存图像", dialog)
        chk_force_geo = QtWidgets.QCheckBox("强制按经纬度解释", dialog)
        chk_force_geo.setChecked(bool(getattr(self, "_location_map_terrain_force_geo", False)))
        terrain_hint = QtWidgets.QLabel("支持 .grd/.nc/.xyz/.txt", dialog)
        terrain_hint.setStyleSheet("color:#666;")
        terrain_row.addWidget(btn_load_terrain)
        terrain_row.addWidget(btn_clear_terrain)
        terrain_row.addWidget(btn_dump_terrain)
        terrain_row.addWidget(btn_save_map)
        terrain_row.addWidget(chk_force_geo)
        terrain_row.addWidget(terrain_hint)
        terrain_row.addStretch(1)
        lay.addLayout(terrain_row)
        proj_row = QtWidgets.QHBoxLayout()
        chk_manual_zone = QtWidgets.QCheckBox("手动UTM分带", dialog)
        chk_manual_zone.setChecked(bool(getattr(self, "_location_map_terrain_manual_zone_enabled", False)))
        proj_row.addWidget(chk_manual_zone)
        proj_row.addWidget(QtWidgets.QLabel("Zone:", dialog))
        spin_zone = QtWidgets.QSpinBox(dialog)
        spin_zone.setRange(1, 60)
        spin_zone.setValue(int(getattr(self, "_location_map_terrain_manual_zone_value", 50)))
        proj_row.addWidget(spin_zone)
        proj_row.addWidget(QtWidgets.QLabel("半球:", dialog))
        combo_hemi = QtWidgets.QComboBox(dialog)
        combo_hemi.addItem("自动", "auto")
        combo_hemi.addItem("北半球", "north")
        combo_hemi.addItem("南半球", "south")
        hemi_val = str(getattr(self, "_location_map_terrain_manual_hemi", "auto")).lower()
        hemi_idx = max(0, combo_hemi.findData(hemi_val))
        combo_hemi.setCurrentIndex(hemi_idx)
        proj_row.addWidget(combo_hemi)
        proj_row.addSpacing(8)
        chk_sac2y_tm = QtWidgets.QCheckBox("SAC2Y兼容TM", dialog)
        chk_sac2y_tm.setChecked(bool(getattr(self, "_location_map_terrain_use_sac2y_tm", False)))
        proj_row.addWidget(chk_sac2y_tm)
        proj_row.addWidget(QtWidgets.QLabel("lon0:", dialog))
        spin_tm_lon0 = QtWidgets.QDoubleSpinBox(dialog)
        spin_tm_lon0.setDecimals(6)
        spin_tm_lon0.setRange(-180.0, 360.0)
        spin_tm_lon0.setSingleStep(0.1)
        spin_tm_lon0.setValue(float(getattr(self, "_location_map_terrain_tm_lon0", 120.0)))
        proj_row.addWidget(spin_tm_lon0)
        chk_tm_wrap = QtWidgets.QCheckBox("lon<0加360", dialog)
        chk_tm_wrap.setChecked(bool(getattr(self, "_location_map_terrain_tm_lon_wrap360", True)))
        proj_row.addWidget(chk_tm_wrap)
        chk_swap_lonlat = QtWidgets.QCheckBox("经纬互换", dialog)
        chk_swap_lonlat.setChecked(bool(getattr(self, "_location_map_terrain_swap_lonlat", False)))
        proj_row.addWidget(chk_swap_lonlat)
        proj_row.addStretch(1)
        lay.addLayout(proj_row)
        vis_row = QtWidgets.QHBoxLayout()
        vis_row.addWidget(QtWidgets.QLabel("色带:", dialog))
        combo_palette = QtWidgets.QComboBox(dialog)
        combo_palette.addItem("地形", "terrain")
        combo_palette.addItem("灰度", "gray")
        combo_palette.addItem("GMT风格", "gmt")
        combo_palette.addItem("反色地形", "terrain_r")
        combo_palette.addItem("自选CPT", "custom_cpt")
        pal_val = str(getattr(self, "_location_map_terrain_palette", "terrain")).lower()
        pal_idx = max(0, combo_palette.findData(pal_val))
        combo_palette.setCurrentIndex(pal_idx)
        vis_row.addWidget(combo_palette)
        btn_pick_cpt = QtWidgets.QPushButton("选择CPT", dialog)
        vis_row.addWidget(btn_pick_cpt)
        vis_row.addWidget(QtWidgets.QLabel("光照:", dialog))
        spin_shade = QtWidgets.QDoubleSpinBox(dialog)
        spin_shade.setRange(0.0, 1.0)
        spin_shade.setSingleStep(0.05)
        spin_shade.setDecimals(2)
        spin_shade.setValue(float(getattr(self, "_location_map_terrain_shade_strength", 0.75)))
        vis_row.addWidget(spin_shade)
        vis_row.addWidget(QtWidgets.QLabel("高:", dialog))
        spin_light_alt = QtWidgets.QDoubleSpinBox(dialog)
        spin_light_alt.setRange(1.0, 89.0)
        spin_light_alt.setSingleStep(1.0)
        spin_light_alt.setDecimals(1)
        spin_light_alt.setValue(float(getattr(self, "_location_map_terrain_light_alt_deg", 45.0)))
        vis_row.addWidget(spin_light_alt)
        vis_row.addWidget(QtWidgets.QLabel("方位:", dialog))
        spin_light_az = QtWidgets.QDoubleSpinBox(dialog)
        spin_light_az.setRange(0.0, 360.0)
        spin_light_az.setSingleStep(5.0)
        spin_light_az.setDecimals(1)
        spin_light_az.setValue(float(getattr(self, "_location_map_terrain_light_az_deg", 315.0)))
        vis_row.addWidget(spin_light_az)
        chk_coast = QtWidgets.QCheckBox("海陆分界增强", dialog)
        chk_coast.setChecked(bool(getattr(self, "_location_map_terrain_coast_enhance", True)))
        vis_row.addWidget(chk_coast)
        vis_row.addStretch(1)
        lay.addLayout(vis_row)
        spin_tm_lon0.setEnabled(chk_sac2y_tm.isChecked())
        chk_tm_wrap.setEnabled(chk_sac2y_tm.isChecked())
        spin_zone.setEnabled(chk_manual_zone.isChecked() and (not chk_sac2y_tm.isChecked()))
        combo_hemi.setEnabled(chk_manual_zone.isChecked() and (not chk_sac2y_tm.isChecked()))
        proj_info = QtWidgets.QLabel(dialog)
        proj_info.setStyleSheet("color:#4b5563;")
        proj_info.setWordWrap(True)
        proj_info.setText(self._location_map_terrain_proj_text or "投影参数：未进行经纬度到UTM转换")
        self._location_map_terrain_proj_label = proj_info
        lay.addWidget(proj_info)

        pw = pg.PlotWidget(background=self._theme_color("plot_bg", "#ffffff"))
        pw.showGrid(x=True, y=True, alpha=0.15)
        pw.enableAutoRange(False)
        plot_item = pw.getPlotItem()
        self._location_map_plot_widget = pw
        self._location_map_plot_item = plot_item
        vb = plot_item.getViewBox()
        if vb is not None:
            vb.setAspectLocked(True, ratio=1.0)
        plot_item.setMenuEnabled(False)
        if use_utm:
            plot_item.setLabels(left="Y (m)", bottom="X (m)")
        else:
            plot_item.setLabels(left="Latitude (deg)", bottom="Longitude (deg)")

        legend = plot_item.addLegend(offset=(8, 8))
        if sx.size > 0:
            src_item = pg.ScatterPlotItem(
                x=sx,
                y=sy,
                size=16,
                pen=pg.mkPen("#7f1d1d", width=1.0),
                brush=pg.mkBrush("#ef4444"),
                symbol="t",
            )
            src_item.setZValue(120)
            plot_item.addItem(src_item)
            try:
                legend.addItem(src_item, "震源点")
            except Exception:
                pass
        if rx.size > 0:
            rec_item = pg.ScatterPlotItem(
                x=rx,
                y=ry,
                size=8,
                pen=pg.mkPen("#6b7280", width=1.0),
                brush=pg.mkBrush("#9ca3af"),
                symbol="o",
            )
            rec_item.setZValue(80)
            plot_item.addItem(rec_item)
            try:
                legend.addItem(rec_item, "接收点")
            except Exception:
                pass

        # 当前道实时标记（由主剖面鼠标移动驱动）
        self._location_map_cursor_item = pg.ScatterPlotItem(
            size=14,
            pen=pg.mkPen("#ffffff", width=1.6),
            brush=pg.mkBrush("#facc15"),
            symbol="o",
            pxMode=True,
        )
        self._location_map_cursor_item.setZValue(1000)
        plot_item.addItem(self._location_map_cursor_item, ignoreBounds=True)
        self._location_map_selected_item = pg.ScatterPlotItem(
            size=16,
            pen=pg.mkPen("#f3f4f6", width=2.0),
            brush=pg.mkBrush("#facc15"),
            symbol="o",
            pxMode=True,
        )
        self._location_map_selected_item.setZValue(999)
        plot_item.addItem(self._location_map_selected_item, ignoreBounds=True)

        # Map 点击联动：跳到最近道
        pw.scene().sigMouseClicked.connect(lambda ev, _pw=pw: self._on_location_map_mouse_clicked(ev, _pw))
        chk_force_geo.toggled.connect(
            lambda v: (
                setattr(self, "_location_map_terrain_force_geo", bool(v)),
                self._refresh_location_map_terrain_overlay(),
            )
        )
        chk_manual_zone.toggled.connect(
            lambda v, _spin=spin_zone, _combo=combo_hemi, _tm=chk_sac2y_tm: (
                setattr(self, "_location_map_terrain_manual_zone_enabled", bool(v)),
                _spin.setEnabled(bool(v) and (not _tm.isChecked())),
                _combo.setEnabled(bool(v) and (not _tm.isChecked())),
                self._refresh_location_map_terrain_overlay(),
            )
        )
        spin_zone.valueChanged.connect(
            lambda v: (
                setattr(self, "_location_map_terrain_manual_zone_value", int(v)),
                self._refresh_location_map_terrain_overlay(),
            )
        )
        combo_hemi.currentIndexChanged.connect(
            lambda _i, _c=combo_hemi: setattr(
                self,
                "_location_map_terrain_manual_hemi",
                str(_c.currentData() or "auto"),
            )
        )
        combo_hemi.currentIndexChanged.connect(lambda _i: self._refresh_location_map_terrain_overlay())
        chk_sac2y_tm.toggled.connect(
            lambda v, _spin=spin_tm_lon0, _wrap=chk_tm_wrap, _m=chk_manual_zone, _z=spin_zone, _h=combo_hemi: (
                setattr(self, "_location_map_terrain_use_sac2y_tm", bool(v)),
                _spin.setEnabled(bool(v)),
                _wrap.setEnabled(bool(v)),
                _z.setEnabled(_m.isChecked() and (not bool(v))),
                _h.setEnabled(_m.isChecked() and (not bool(v))),
                self._refresh_location_map_terrain_overlay(),
            )
        )
        spin_tm_lon0.valueChanged.connect(
            lambda v: (
                setattr(self, "_location_map_terrain_tm_lon0", float(v)),
                self._refresh_location_map_terrain_overlay(),
            )
        )
        chk_tm_wrap.toggled.connect(
            lambda v: (
                setattr(self, "_location_map_terrain_tm_lon_wrap360", bool(v)),
                self._refresh_location_map_terrain_overlay(),
            )
        )
        chk_swap_lonlat.toggled.connect(
            lambda v: (
                setattr(self, "_location_map_terrain_swap_lonlat", bool(v)),
                self._refresh_location_map_terrain_overlay(),
            )
        )
        combo_palette.currentIndexChanged.connect(
            lambda _i, _c=combo_palette: (
                setattr(self, "_location_map_terrain_palette", str(_c.currentData() or "terrain")),
                self._refresh_location_map_terrain_overlay(),
            )
        )
        btn_pick_cpt.clicked.connect(self._pick_location_map_cpt_file)
        spin_shade.valueChanged.connect(
            lambda v: (
                setattr(self, "_location_map_terrain_shade_strength", float(v)),
                self._refresh_location_map_terrain_overlay(),
            )
        )
        spin_light_alt.valueChanged.connect(
            lambda v: (
                setattr(self, "_location_map_terrain_light_alt_deg", float(v)),
                self._refresh_location_map_terrain_overlay(),
            )
        )
        spin_light_az.valueChanged.connect(
            lambda v: (
                setattr(self, "_location_map_terrain_light_az_deg", float(v)),
                self._refresh_location_map_terrain_overlay(),
            )
        )
        chk_coast.toggled.connect(
            lambda v: (
                setattr(self, "_location_map_terrain_coast_enhance", bool(v)),
                self._refresh_location_map_terrain_overlay(),
            )
        )
        btn_load_terrain.clicked.connect(self._load_location_map_terrain_from_dialog)
        btn_clear_terrain.clicked.connect(self._clear_location_map_terrain)
        btn_dump_terrain.clicked.connect(self._show_location_map_terrain_sample_dialog)
        btn_save_map.clicked.connect(self._save_location_map_figure)
        if self._location_map_terrain_meta is not None:
            self._apply_location_map_terrain_overlay()

        # 自动缩放到全部点
        x_all = np.concatenate([a for a in (sx, rx) if a.size > 0])
        y_all = np.concatenate([a for a in (sy, ry) if a.size > 0])
        if x_all.size > 0 and y_all.size > 0:
            x_fin = np.asarray(x_all[np.isfinite(x_all)], dtype=float)
            y_fin = np.asarray(y_all[np.isfinite(y_all)], dtype=float)
            if x_fin.size > 0 and y_fin.size > 0:
                if x_fin.size > 200:
                    xmin = float(np.nanpercentile(x_fin, 1.0))
                    xmax = float(np.nanpercentile(x_fin, 99.0))
                else:
                    xmin = float(np.min(x_fin))
                    xmax = float(np.max(x_fin))
                if y_fin.size > 200:
                    ymin = float(np.nanpercentile(y_fin, 1.0))
                    ymax = float(np.nanpercentile(y_fin, 99.0))
                else:
                    ymin = float(np.min(y_fin))
                    ymax = float(np.max(y_fin))
                if not (np.isfinite(xmin) and np.isfinite(xmax) and np.isfinite(ymin) and np.isfinite(ymax)):
                    xmin, xmax = float(np.min(x_fin)), float(np.max(x_fin))
                    ymin, ymax = float(np.min(y_fin)), float(np.max(y_fin))
                if abs(xmax - xmin) < 1e-9:
                    xmax = xmin + 1.0
                if abs(ymax - ymin) < 1e-9:
                    ymax = ymin + 1.0
                self._location_map_base_bounds = (xmin, xmax, ymin, ymax)
                self._apply_location_map_view_range()

        map_row = QtWidgets.QHBoxLayout()
        map_row.addWidget(pw, stretch=1)
        cbar_col = QtWidgets.QVBoxLayout()
        lbl_max = QtWidgets.QLabel("", dialog)
        lbl_min = QtWidgets.QLabel("", dialog)
        lbl_max.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        lbl_min.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        grad = pg.GradientWidget(orientation="right")
        grad.setMinimumWidth(26)
        grad.setMaximumWidth(26)
        cbar_col.addWidget(lbl_max)
        cbar_col.addWidget(grad, stretch=1)
        cbar_col.addWidget(lbl_min)
        cbar_wrap = QtWidgets.QWidget(dialog)
        cbar_wrap.setLayout(cbar_col)
        cbar_wrap.setVisible(False)
        map_row.addWidget(cbar_wrap)
        self._location_map_colorbar_gradient = grad
        self._location_map_colorbar_min_label = lbl_min
        self._location_map_colorbar_max_label = lbl_max
        lay.addLayout(map_row, stretch=1)

        close_btn = QtWidgets.QPushButton("关闭", dialog)
        close_btn.clicked.connect(dialog.close)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close_btn)
        lay.addLayout(row)
        if trace_points:
            self._location_map_trace_points = np.asarray(trace_points, dtype=float)
            self._location_map_trace_indices = np.asarray(trace_indices, dtype=int)
            self._location_map_mode = "utm" if use_utm else "geo"
        else:
            self._location_map_trace_points = np.empty((0, 2), dtype=float)
            self._location_map_trace_indices = np.empty((0,), dtype=int)
            self._location_map_mode = "none"
            self._location_map_rec_role = "auto"
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

        # 打开 map 后立即根据当前鼠标位置同步一次标记
        if self._last_render_trace_indices.size > 0 and self._last_render_offsets.size > 0 and self.mouse_x is not None:
            try:
                nearest_i = int(np.argmin(np.abs(self._last_render_offsets - float(self.mouse_x))))
                trace_idx = int(self._last_render_trace_indices[nearest_i])
                self._update_location_map_cursor_for_trace(trace_idx)
            except Exception:
                pass
        if self._map_link_trace_idx is not None:
            try:
                self._update_location_map_selected_for_trace(int(self._map_link_trace_idx))
            except Exception:
                pass

    def _trace_position_for_map(self, trace_idx: int, mode: str) -> Tuple[float, float]:
        if self.loaded is None:
            return (0.0, 0.0)
        trace_headers = self.loaded.get("trace_headers", []) or []
        if trace_idx < 0 or trace_idx >= len(trace_headers):
            return (0.0, 0.0)
        th = trace_headers[trace_idx]

        def _valid_xy(x: float, y: float) -> bool:
            return math.isfinite(x) and math.isfinite(y) and (abs(x) > 1e-9 or abs(y) > 1e-9)

        if str(mode).lower() == "utm":
            rec_role = str(getattr(self, "_location_map_rec_role", "rx")).lower()
            if rec_role == "sx":
                rx = float(getattr(th, "sxutm", 0.0) or 0.0)
                ry = float(getattr(th, "syutm", 0.0) or 0.0)
                sx = float(getattr(th, "rxutm", 0.0) or 0.0)
                sy = float(getattr(th, "ryutm", 0.0) or 0.0)
            else:
                rx = float(getattr(th, "rxutm", 0.0) or 0.0)
                ry = float(getattr(th, "ryutm", 0.0) or 0.0)
                sx = float(getattr(th, "sxutm", 0.0) or 0.0)
                sy = float(getattr(th, "syutm", 0.0) or 0.0)
            if _valid_xy(rx, ry):
                return (rx, ry)
            if _valid_xy(sx, sy):
                off_km = float(getattr(th, "offsti", 0.0) or 0.0)
                azi_deg = float(getattr(th, "azi", 0.0) or 0.0)
                if math.isfinite(off_km) and abs(off_km) > 1e-9 and math.isfinite(azi_deg):
                    dist_m = abs(off_km) * 1000.0
                    ang = math.radians(azi_deg)
                    ex = sx + dist_m * math.sin(ang)
                    ey = sy + dist_m * math.cos(ang)
                    if _valid_xy(ex, ey):
                        return (ex, ey)
                return (sx, sy)
            return (0.0, 0.0)

        rec_role = str(getattr(self, "_location_map_rec_role", "rx")).lower()
        if rec_role == "sx":
            lon = float(getattr(th, "slong", 0.0) or 0.0)
            lat = float(getattr(th, "slat", 0.0) or 0.0)
            slon = float(getattr(th, "rlong", 0.0) or 0.0)
            slat = float(getattr(th, "rlat", 0.0) or 0.0)
        else:
            lon = float(getattr(th, "rlong", 0.0) or 0.0)
            lat = float(getattr(th, "rlat", 0.0) or 0.0)
            slon = float(getattr(th, "slong", 0.0) or 0.0)
            slat = float(getattr(th, "slat", 0.0) or 0.0)
        if _valid_xy(lon, lat):
            return (lon, lat)
        if _valid_xy(slon, slat):
            return (slon, slat)
        return (0.0, 0.0)

    def _load_location_map_terrain_from_dialog(self) -> None:
        if self._location_map_dialog is None or self._location_map_plot_item is None:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择地形文件",
            "",
            "Terrain (*.grd *.nc *.xyz *.txt);;NetCDF (*.grd *.nc);;XYZ (*.xyz *.txt);;All files (*)",
            options=self._file_dialog_options(),
        )
        if not path:
            return
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            self._set_status_text("位置Map：正在加载并转换地形...", hold_ms=2000)
            QtWidgets.QApplication.processEvents()
            force_geo = bool(getattr(self, "_location_map_terrain_force_geo", False))
            meta = self._load_terrain_meta(path, force_geo=force_geo)
            self._location_map_terrain_meta = meta
            self._location_map_terrain_cache_key = None
            self._apply_location_map_terrain_overlay()
            self._set_status_text(f"地形已加载：{Path(path).name}", hold_ms=1800)
        except Exception as exc:
            self._show_themed_info("地形加载失败", str(exc))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _clear_location_map_terrain(self) -> None:
        self._location_map_terrain_meta = None
        self._location_map_terrain_cache_key = None
        self._location_map_terrain_proj_text = ""
        if self._location_map_terrain_proj_label is not None:
            self._location_map_terrain_proj_label.setText("投影参数：未进行经纬度到UTM转换")
        if self._location_map_plot_item is not None and self._location_map_terrain_item is not None:
            try:
                self._location_map_plot_item.removeItem(self._location_map_terrain_item)
            except Exception:
                pass
        self._location_map_terrain_item = None
        self._update_location_map_colorbar(None, None)
        self._set_status_text("地形叠加已清除", hold_ms=1200)

    def _refresh_location_map_terrain_overlay(self) -> None:
        if self._location_map_terrain_meta is None:
            return
        try:
            self._apply_location_map_terrain_overlay()
        except Exception as exc:
            self._show_themed_info("地形刷新失败", str(exc))

    def _save_location_map_figure(self) -> None:
        if self._location_map_plot_item is None:
            self._show_themed_info("保存位置图", "位置Map尚未打开。")
            return
        out, selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "保存位置Map图像",
            "location_map.png",
            "PNG Image (*.png);;SVG Vector (*.svg)",
            options=self._file_dialog_options(),
        )
        if not out:
            return
        try:
            use_svg = selected.lower().startswith("svg") or out.lower().endswith(".svg")
            if use_svg:
                if not out.lower().endswith(".svg"):
                    out += ".svg"
                exporter = pg_exporters.SVGExporter(self._location_map_plot_item)
                exporter.export(out)
            else:
                if not out.lower().endswith(".png"):
                    out += ".png"
                exporter = pg_exporters.ImageExporter(self._location_map_plot_item)
                exporter.parameters()["width"] = 1800
                exporter.export(out)
            self._set_status_text(f"位置图已保存：{Path(out).name}", hold_ms=1800)
        except Exception as exc:
            self._show_themed_info("保存位置图失败", str(exc))

    def _build_location_map_colormap_and_levels(
        self, palette: str, zmin: float, zmax: float
    ) -> Tuple[pg.ColorMap, float, float]:
        pal = str(palette or "terrain").lower()
        if pal == "custom_cpt":
            cpt = self._get_custom_cpt_colormap()
            if cpt is not None:
                z_arr, rgb_arr = cpt
                cmin = float(np.min(z_arr))
                cmax = float(np.max(z_arr))
                cspan = max(1e-12, cmax - cmin)
                pos = np.clip((z_arr - cmin) / cspan, 0.0, 1.0)
                cmap = pg.ColorMap(pos, np.clip(rgb_arr, 0, 255).astype(np.ubyte))
                return cmap, cmin, cmax
        vals = np.linspace(0.0, 1.0, 256, dtype=float)
        rgb = self._terrain_colormap_rgb(vals, palette=pal).reshape(256, 3)
        cmap = pg.ColorMap(vals, np.clip(rgb, 0, 255).astype(np.ubyte))
        return cmap, float(zmin), float(zmax)

    def _update_location_map_colorbar(self, zmin: Optional[float], zmax: Optional[float]) -> None:
        grad = self._location_map_colorbar_gradient
        lbl_min = self._location_map_colorbar_min_label
        lbl_max = self._location_map_colorbar_max_label
        if grad is None or lbl_min is None or lbl_max is None:
            return
        if zmin is None or zmax is None or (not np.isfinite(zmin)) or (not np.isfinite(zmax)):
            grad.parentWidget().setVisible(False)
            return
        pal = str(getattr(self, "_location_map_terrain_palette", "terrain"))
        cmap, lv_min, lv_max = self._build_location_map_colormap_and_levels(pal, float(zmin), float(zmax))
        if hasattr(grad, "setColorMap"):
            try:
                grad.setColorMap(cmap)
            except Exception:
                pass
        if hasattr(grad, "item"):
            try:
                pos, cols = cmap.getStops(mode="byte")
                ticks = []
                for i in range(len(pos)):
                    c = cols[i]
                    ticks.append((float(pos[i]), (int(c[0]), int(c[1]), int(c[2]), 255)))
                grad.item.restoreState({"mode": "rgb", "ticks": ticks})
            except Exception:
                pass
        lbl_min.setText(f"{lv_min:.0f}")
        lbl_max.setText(f"{lv_max:.0f}")
        grad.parentWidget().setVisible(True)

    def _pick_location_map_cpt_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择CPT文件",
            "",
            "CPT (*.cpt *.txt);;All files (*)",
            options=self._file_dialog_options(),
        )
        if not path:
            return
        self._location_map_terrain_cpt_path = str(path)
        self._location_map_terrain_cpt_cache_key = None
        self._location_map_terrain_cpt_cache_data = None
        self._location_map_terrain_palette = "custom_cpt"
        self._location_map_terrain_cache_key = None
        self._set_status_text(f"CPT已选择：{Path(path).name}", hold_ms=1800)
        self._refresh_location_map_terrain_overlay()

    def _get_custom_cpt_colormap(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        path = str(getattr(self, "_location_map_terrain_cpt_path", "") or "").strip()
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        mtime = float(p.stat().st_mtime)
        key = (str(p), mtime)
        if self._location_map_terrain_cpt_cache_key == key and self._location_map_terrain_cpt_cache_data is not None:
            return self._location_map_terrain_cpt_cache_data
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return None
        color_model = "RGB"
        pts: List[Tuple[float, float, float, float]] = []

        def _to_rgb_triplet(c1: float, c2: float, c3: float) -> Tuple[float, float, float]:
            if color_model == "HSV":
                h = float(c1) % 360.0
                s = float(c2)
                v = float(c3)
                if s > 1.0:
                    s /= 100.0
                if v > 1.0:
                    v /= 100.0
                s = max(0.0, min(1.0, s))
                v = max(0.0, min(1.0, v))
                rr, gg, bb = colorsys.hsv_to_rgb(h / 360.0, s, v)
                return (rr * 255.0, gg * 255.0, bb * 255.0)
            return (float(c1), float(c2), float(c3))

        for raw in lines:
            s = raw.strip()
            if not s:
                continue
            if "COLOR_MODEL" in s.upper():
                su = s.upper().replace(" ", "")
                if "HSV" in su:
                    color_model = "HSV"
                elif "RGB" in su:
                    color_model = "RGB"
            if s.startswith("#"):
                continue
            toks = s.split()
            if not toks:
                continue
            if toks[0].upper() in ("B", "F", "N"):
                continue
            nums: List[float] = []
            ok = True
            for t in toks[:8]:
                try:
                    nums.append(float(t))
                except Exception:
                    ok = False
                    break
            if not ok or len(nums) < 4:
                continue
            if len(nums) >= 8:
                z1, c11, c12, c13, z2, c21, c22, c23 = nums[:8]
                r1, g1, b1 = _to_rgb_triplet(c11, c12, c13)
                r2, g2, b2 = _to_rgb_triplet(c21, c22, c23)
                pts.append((z1, r1, g1, b1))
                pts.append((z2, r2, g2, b2))
            else:
                z, c1, c2, c3 = nums[:4]
                r, g, b = _to_rgb_triplet(c1, c2, c3)
                pts.append((z, r, g, b))
        if len(pts) < 2:
            return None
        pts.sort(key=lambda x: x[0])
        z_vals = np.asarray([p0 for p0, _, _, _ in pts], dtype=float)
        rgb_vals = np.asarray([[r, g, b] for _, r, g, b in pts], dtype=float)
        # 去重
        uniq_z: List[float] = []
        uniq_rgb: List[np.ndarray] = []
        for i in range(z_vals.size):
            z = float(z_vals[i])
            if len(uniq_z) == 0 or abs(z - uniq_z[-1]) > 1e-12:
                uniq_z.append(z)
                uniq_rgb.append(rgb_vals[i])
            else:
                uniq_rgb[-1] = rgb_vals[i]
        z_arr = np.asarray(uniq_z, dtype=float)
        rgb_arr = np.asarray(uniq_rgb, dtype=float)
        colors = np.clip(rgb_arr, 0.0, 255.0)
        self._location_map_terrain_cpt_cache_key = key
        self._location_map_terrain_cpt_cache_data = (z_arr, colors)
        return self._location_map_terrain_cpt_cache_data

    def _apply_location_map_view_range(
        self, terrain_bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> None:
        if self._location_map_plot_item is None:
            return
        vb = self._location_map_plot_item.getViewBox()
        if vb is None:
            return
        bounds = self._location_map_base_bounds
        if bounds is None and terrain_bounds is None:
            return
        if bounds is None:
            xmin, xmax, ymin, ymax = terrain_bounds  # type: ignore
        elif terrain_bounds is None:
            xmin, xmax, ymin, ymax = bounds
        else:
            xmin = float(min(bounds[0], terrain_bounds[0]))
            xmax = float(max(bounds[1], terrain_bounds[1]))
            ymin = float(min(bounds[2], terrain_bounds[2]))
            ymax = float(max(bounds[3], terrain_bounds[3]))
        if not (np.isfinite(xmin) and np.isfinite(xmax) and np.isfinite(ymin) and np.isfinite(ymax)):
            return
        dx = max(1.0, abs(xmax - xmin))
        dy = max(1.0, abs(ymax - ymin))
        xpad = 0.03 * dx
        ypad = 0.03 * dy
        vb.setRange(
            xRange=(xmin - xpad, xmax + xpad),
            yRange=(ymin - ypad, ymax + ypad),
            padding=0.0,
        )

    def _build_location_map_terrain_sample_text(self, max_rows: int = 80) -> str:
        meta = self._location_map_terrain_meta
        if meta is None:
            return "尚未加载地形数据。"
        mode = str(meta.get("mode", ""))
        coord_kind = str(meta.get("coord_kind", "unknown")).lower()
        map_mode = str(getattr(self, "_location_map_mode", "none")).lower()
        use_sac2y_tm = bool(getattr(self, "_location_map_terrain_use_sac2y_tm", False))

        if mode == "points":
            x_raw = np.asarray(meta.get("x", []), dtype=float)
            y_raw = np.asarray(meta.get("y", []), dtype=float)
            z_raw = np.asarray(meta.get("z", []), dtype=float)
        elif mode == "grid":
            x = np.asarray(meta.get("x", []), dtype=float)
            y = np.asarray(meta.get("y", []), dtype=float)
            z = np.asarray(meta.get("z", []), dtype=float)
            if x.size == 0 or y.size == 0 or z.size == 0:
                return "地形网格为空。"
            xx, yy = np.meshgrid(x, y, indexing="xy")
            x_raw = xx.reshape(-1)
            y_raw = yy.reshape(-1)
            z_raw = np.asarray(z, dtype=float).reshape(-1)
        else:
            return f"不支持的地形模式: {mode}"

        raw_valid = np.isfinite(x_raw) & np.isfinite(y_raw) & np.isfinite(z_raw)
        if not np.any(raw_valid):
            return "原始地形坐标无有效点。"
        x_raw = np.asarray(x_raw[raw_valid], dtype=float)
        y_raw = np.asarray(y_raw[raw_valid], dtype=float)
        z_raw = np.asarray(z_raw[raw_valid], dtype=float)

        lon_geo = np.full(x_raw.shape, np.nan, dtype=float)
        lat_geo = np.full(y_raw.shape, np.nan, dtype=float)
        x_proj = np.asarray(x_raw, dtype=float)
        y_proj = np.asarray(y_raw, dtype=float)
        geo_order = "n/a"
        proj_text = "原样坐标（未做经纬度->UTM）"

        if map_mode == "utm" and coord_kind == "geo":
            lon_geo, lat_geo, geo_order = self._normalize_geo_lonlat_order(x_raw, y_raw)
            if use_sac2y_tm:
                lon0_tm = float(getattr(self, "_location_map_terrain_tm_lon0", 120.0))
                wrap_tm = bool(getattr(self, "_location_map_terrain_tm_lon_wrap360", True))
                x_proj, y_proj, _info = self._convert_lonlat_to_sac2y_tm_arrays(
                    lon_geo, lat_geo, lon0=lon0_tm, wrap_lon360=wrap_tm
                )
                proj_text = f"SAC2Y TM(lon0={lon0_tm:.6f}, wrap360={'on' if wrap_tm else 'off'})"
            else:
                zone_override, hemi_override = self._get_manual_utm_override()
                x_proj, y_proj, _info = self._convert_lonlat_to_utm_arrays(
                    lon_geo,
                    lat_geo,
                    override_zone=zone_override,
                    override_hemisphere=hemi_override,
                )
                proj_text = "UTM(EPSG)模式"

        proj_valid = np.isfinite(x_proj) & np.isfinite(y_proj)
        if not np.any(proj_valid):
            return "投影后无有效点。请检查投影参数。"
        x_keep = x_raw[proj_valid]
        y_keep = y_raw[proj_valid]
        z_keep = z_raw[proj_valid]
        lon_keep = lon_geo[proj_valid]
        lat_keep = lat_geo[proj_valid]
        xp_keep = np.asarray(x_proj[proj_valid], dtype=float)
        yp_keep = np.asarray(y_proj[proj_valid], dtype=float)

        n = int(xp_keep.size)
        rows = max(1, min(int(max_rows), n))
        if n <= rows:
            idx = np.arange(n, dtype=int)
        else:
            idx = np.linspace(0, n - 1, rows, dtype=int)
            idx = np.unique(idx)

        lines: List[str] = []
        lines.append(f"path: {meta.get('path', '')}")
        lines.append(f"mode={mode}, coord_kind={coord_kind}, map_mode={map_mode}")
        lines.append(f"projection={proj_text}, geo_order={geo_order}")
        lines.append(f"valid_points={n}, shown={int(idx.size)}")
        lines.append("idx, raw_x, raw_y, raw_z, lon_used, lat_used, proj_x, proj_y")
        for ii in idx:
            lines.append(
                f"{int(ii):6d}, "
                f"{x_keep[ii]:.8f}, {y_keep[ii]:.8f}, {z_keep[ii]:.3f}, "
                f"{lon_keep[ii]:.8f}, {lat_keep[ii]:.8f}, "
                f"{xp_keep[ii]:.3f}, {yp_keep[ii]:.3f}"
            )
        return "\n".join(lines)

    def _show_location_map_terrain_sample_dialog(self) -> None:
        if self._location_map_dialog is None:
            self._show_themed_info("转换样本", "请先打开位置Map。")
            return
        try:
            text = self._build_location_map_terrain_sample_text(max_rows=80)
        except Exception as exc:
            self._show_themed_info("转换样本生成失败", str(exc))
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("地形转换样本")
        dlg.resize(980, 620)
        dlg.setModal(False)
        lay = QtWidgets.QVBoxLayout(dlg)
        editor = QtWidgets.QPlainTextEdit(dlg)
        editor.setReadOnly(True)
        font = QtGui.QFont("Consolas")
        font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        editor.setFont(font)
        editor.setPlainText(text)
        lay.addWidget(editor, stretch=1)
        row = QtWidgets.QHBoxLayout()
        btn_copy = QtWidgets.QPushButton("复制", dlg)
        btn_close = QtWidgets.QPushButton("关闭", dlg)
        btn_copy.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(editor.toPlainText()))
        btn_close.clicked.connect(dlg.close)
        row.addStretch(1)
        row.addWidget(btn_copy)
        row.addWidget(btn_close)
        lay.addLayout(row)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _load_terrain_meta(self, path: str, force_geo: bool = False) -> Dict[str, object]:
        def _guess_coord_kind(xv: np.ndarray, yv: np.ndarray, xname: str = "", yname: str = "") -> str:
            xn = str(xname).lower()
            yn = str(yname).lower()
            if ("lon" in xn and "lat" in yn) or ("lon" in yn and "lat" in xn):
                return "geo"
            if ("x" == xn and "y" == yn) or ("easting" in xn or "northing" in yn):
                # 仅命名提示，不强制判定
                pass
            if xv.size == 0 or yv.size == 0:
                return "unknown"
            x_abs = float(np.nanmax(np.abs(xv[np.isfinite(xv)]))) if np.any(np.isfinite(xv)) else 0.0
            y_abs = float(np.nanmax(np.abs(yv[np.isfinite(yv)]))) if np.any(np.isfinite(yv)) else 0.0
            xmin = float(np.nanmin(xv[np.isfinite(xv)])) if np.any(np.isfinite(xv)) else 0.0
            xmax = float(np.nanmax(xv[np.isfinite(xv)])) if np.any(np.isfinite(xv)) else 0.0
            ymin = float(np.nanmin(yv[np.isfinite(yv)])) if np.any(np.isfinite(yv)) else 0.0
            ymax = float(np.nanmax(yv[np.isfinite(yv)])) if np.any(np.isfinite(yv)) else 0.0
            if -180.5 <= xmin <= 180.5 and -180.5 <= xmax <= 180.5 and -90.5 <= ymin <= 90.5 and -90.5 <= ymax <= 90.5:
                return "geo"
            if x_abs > 1000.0 and y_abs > 1000.0:
                return "utm"
            return "unknown"

        p = Path(path)
        suffix = p.suffix.lower()
        if suffix in (".xyz", ".txt"):
            arr = np.loadtxt(str(p), comments="#", dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape((1, -1))
            if arr.shape[1] < 3:
                raise ValueError("xyz文本至少需要三列：x y z")
            x = np.asarray(arr[:, 0], dtype=float)
            y = np.asarray(arr[:, 1], dtype=float)
            z = np.asarray(arr[:, 2], dtype=float)
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            x, y, z = x[valid], y[valid], z[valid]
            if x.size == 0:
                raise ValueError("xyz文本没有有效数据点")
            coord_kind = "geo" if force_geo else _guess_coord_kind(x, y)
            return {"mode": "points", "x": x, "y": y, "z": z, "path": str(p), "coord_kind": coord_kind}

        if suffix in (".nc", ".grd"):
            try:
                import xarray as xr  # type: ignore
            except Exception:
                raise RuntimeError("读取 .nc/.grd 需要 xarray，请先安装：pip install xarray netCDF4")
            try:
                from ..xarray_nc import open_netcdf_like_dataset  # type: ignore
            except ImportError:  # script/非包内运行等
                from pyAOBS.visualization.xarray_nc import open_netcdf_like_dataset  # type: ignore
            ds = open_netcdf_like_dataset(p)
            try:
                data_var = None
                for name, var in ds.data_vars.items():
                    if getattr(var, "ndim", 0) >= 2:
                        data_var = name
                        break
                if data_var is None:
                    for name in ds.variables:
                        try:
                            var = ds[name]
                        except Exception:
                            continue
                        if getattr(var, "ndim", 0) >= 2:
                            data_var = name
                            break
                if data_var is None:
                    try:
                        try:
                            from ..gravity_obs_grid import gmt_nf_coards_flat_to_dataarray
                        except ImportError:
                            from pyAOBS.visualization.gravity_obs_grid import gmt_nf_coards_flat_to_dataarray

                        da_pad = gmt_nf_coards_flat_to_dataarray(ds)
                        x = np.asarray(da_pad.coords["lon"].values, dtype=float)
                        y = np.asarray(da_pad.coords["lat"].values, dtype=float)
                        z = np.asarray(da_pad.values, dtype=float)
                        coord_kind = "geo" if force_geo else _guess_coord_kind(x, y, "lon", "lat")
                        return {
                            "mode": "grid",
                            "x": x,
                            "y": y,
                            "z": z,
                            "path": str(p),
                            "coord_kind": coord_kind,
                            "x_name": "lon",
                            "y_name": "lat",
                        }
                    except Exception:
                        raise ValueError("nc/grd中未找到二维地形变量")
                da = ds[data_var].squeeze()
                if da.ndim < 2:
                    raise ValueError("地形变量维度不足（需要二维）")
                dims = list(da.dims)
                ydim, xdim = dims[-2], dims[-1]
                x = np.asarray(ds[xdim].values, dtype=float)
                y = np.asarray(ds[ydim].values, dtype=float)
                z = np.asarray(da.values, dtype=float)
                if z.ndim > 2:
                    z = z.reshape(z.shape[-2], z.shape[-1])
                if z.shape[0] != y.size or z.shape[1] != x.size:
                    # 尝试转置
                    zt = z.T
                    if zt.shape[0] == y.size and zt.shape[1] == x.size:
                        z = zt
                    else:
                        raise ValueError("nc/grd网格维度与坐标长度不一致")
                coord_kind = "geo" if force_geo else _guess_coord_kind(x, y, xdim, ydim)
                return {
                    "mode": "grid",
                    "x": x,
                    "y": y,
                    "z": z,
                    "path": str(p),
                    "coord_kind": coord_kind,
                    "x_name": str(xdim),
                    "y_name": str(ydim),
                }
            finally:
                try:
                    ds.close()
                except Exception:
                    pass
        raise ValueError("不支持的地形格式，请使用 .grd/.nc/.xyz/.txt")

    def _get_manual_utm_override(self) -> Tuple[Optional[int], Optional[str]]:
        if not bool(getattr(self, "_location_map_terrain_manual_zone_enabled", False)):
            return (None, None)
        zone = int(getattr(self, "_location_map_terrain_manual_zone_value", 0) or 0)
        if zone < 1 or zone > 60:
            return (None, None)
        hemi = str(getattr(self, "_location_map_terrain_manual_hemi", "auto") or "auto").lower()
        if hemi not in ("auto", "north", "south"):
            hemi = "auto"
        return (zone, hemi)

    def _convert_lonlat_to_utm_arrays(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        override_zone: Optional[int] = None,
        override_hemisphere: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        valid = np.isfinite(lon) & np.isfinite(lat)
        if not np.any(valid):
            raise ValueError("经纬度数据无有效点，无法转换为UTM")
        lon_valid = lon[valid]
        lat_valid = lat[valid]
        lon0 = float(np.nanmean(lon_valid))
        lat0 = float(np.nanmean(lat_valid))
        zone_auto = int(np.floor((lon0 + 180.0) / 6.0) + 1)
        zone_auto = max(1, min(60, zone_auto))
        zone = int(override_zone) if (override_zone is not None) else int(zone_auto)
        zone = max(1, min(60, zone))
        hemi_hint = str(override_hemisphere or "auto").lower()
        if hemi_hint == "north":
            lat_sign = 1.0
            hemi_src = "manual"
        elif hemi_hint == "south":
            lat_sign = -1.0
            hemi_src = "manual"
        else:
            lat_sign = 1.0 if lat0 >= 0.0 else -1.0
            hemi_src = "auto"
        epsg = (32600 + zone) if lat_sign >= 0.0 else (32700 + zone)
        central_meridian = float(-183.0 + 6.0 * float(zone))
        hemisphere = 1.0 if lat_sign >= 0.0 else -1.0
        try:
            from pyproj import Transformer  # type: ignore
        except Exception:
            raise RuntimeError("经纬度地形叠加到UTM坐标需要 pyproj，请安装：pip install pyproj")
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        x_out = np.full(lon.shape, np.nan, dtype=float)
        y_out = np.full(lat.shape, np.nan, dtype=float)
        xx, yy = transformer.transform(lon_valid, lat_valid)
        x_out[valid] = np.asarray(xx, dtype=float)
        y_out[valid] = np.asarray(yy, dtype=float)
        info: Dict[str, object] = {
            "zone": float(zone),
            "zone_auto": float(zone_auto),
            "epsg": float(epsg),
            "central_meridian": float(central_meridian),
            "lon0": float(lon0),
            "lat0": float(lat0),
            "hemisphere": float(hemisphere),
            "zone_mode": "manual" if (override_zone is not None) else "auto",
            "hemi_mode": str(hemi_src),
        }
        return x_out, y_out, info

    def _convert_lonlat_to_sac2y_tm_arrays(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        lon0: float,
        wrap_lon360: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        valid = np.isfinite(lon) & np.isfinite(lat)
        if not np.any(valid):
            raise ValueError("经纬度数据无有效点，无法转换为SAC2Y TM")
        lon_valid = lon[valid]
        lat_valid = lat[valid]
        if bool(wrap_lon360):
            lon_use = np.where(lon_valid < 0.0, lon_valid + 360.0, lon_valid)
        else:
            lon_use = lon_valid
        try:
            from pyproj import Transformer  # type: ignore
        except Exception:
            raise RuntimeError("SAC2Y兼容TM坐标转换需要 pyproj，请安装：pip install pyproj")
        proj_string = f"+proj=tmerc +lon_0={float(lon0):.10f} +datum=WGS84 +units=m +k_0=0.9996 +ellps=WGS84"
        transformer = Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)
        xx, yy = transformer.transform(lon_use, lat_valid)
        xx = np.asarray(xx, dtype=float)
        yy = np.asarray(yy, dtype=float)
        # 与 sac2y/format_utils 保持一致：x+500000，南半球 y+10000000（不加 ex/ey）
        xx = xx + 500000.0
        yy = np.where(lat_valid < 0.0, yy + 10000000.0, yy)
        x_out = np.full(lon.shape, np.nan, dtype=float)
        y_out = np.full(lat.shape, np.nan, dtype=float)
        x_out[valid] = xx
        y_out[valid] = yy
        info: Dict[str, object] = {
            "method": "sac2y_tm",
            "lon0_tm": float(lon0),
            "wrap_lon360": bool(wrap_lon360),
            "lon_mean": float(np.nanmean(lon_valid)),
            "lat_mean": float(np.nanmean(lat_valid)),
            "central_meridian": float(lon0),
            "x_false_easting": 500000.0,
            "south_y_shift": 10000000.0,
            "note": "SAC2Y兼容：不使用ex/ey",
        }
        return x_out, y_out, info

    def _normalize_geo_lonlat_order(
        self, x_geo: np.ndarray, y_geo: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        x_geo = np.asarray(x_geo, dtype=float)
        y_geo = np.asarray(y_geo, dtype=float)
        if bool(getattr(self, "_location_map_terrain_swap_lonlat", False)):
            return y_geo, x_geo, "manual_swapped"

        xv = x_geo[np.isfinite(x_geo)]
        yv = y_geo[np.isfinite(y_geo)]
        if xv.size == 0 or yv.size == 0:
            return x_geo, y_geo, "auto_default"
        x_min, x_max = float(np.min(xv)), float(np.max(xv))
        y_min, y_max = float(np.min(yv)), float(np.max(yv))
        x_is_lon = (-180.5 <= x_min <= 360.5) and (-180.5 <= x_max <= 360.5)
        x_is_lat = (-90.5 <= x_min <= 90.5) and (-90.5 <= x_max <= 90.5)
        y_is_lon = (-180.5 <= y_min <= 360.5) and (-180.5 <= y_max <= 360.5)
        y_is_lat = (-90.5 <= y_min <= 90.5) and (-90.5 <= y_max <= 90.5)
        # 典型“列顺序反了”场景：x像纬度，y像经度
        if x_is_lat and y_is_lon and (not x_is_lon or not y_is_lat):
            return y_geo, x_geo, "auto_swapped"
        return x_geo, y_geo, "auto_default"

    def _terrain_colormap_rgb(self, norm: np.ndarray, palette: str = "terrain") -> np.ndarray:
        n = np.clip(np.asarray(norm, dtype=float), 0.0, 1.0)
        pal = str(palette or "terrain").lower()
        if pal == "gray":
            stops = np.array([0.0, 1.0], dtype=float)
            colors = np.array([[30.0, 30.0, 30.0], [240.0, 240.0, 240.0]], dtype=float)
        elif pal == "custom_cpt":
            cpt = self._get_custom_cpt_colormap()
            if cpt is not None:
                z_arr, colors = cpt
                zmin = float(np.min(z_arr))
                zmax = float(np.max(z_arr))
                span = max(1e-12, zmax - zmin)
                stops = np.clip((z_arr - zmin) / span, 0.0, 1.0)
            else:
                stops = np.array([0.0, 0.35, 0.5, 0.78, 1.0], dtype=float)
                colors = np.array(
                    [
                        [26.0, 58.0, 118.0],
                        [55.0, 126.0, 184.0],
                        [102.0, 166.0, 86.0],
                        [158.0, 122.0, 76.0],
                        [240.0, 240.0, 240.0],
                    ],
                    dtype=float,
                )
        elif pal == "gmt":
            stops = np.array([0.0, 0.18, 0.35, 0.5, 0.68, 0.85, 1.0], dtype=float)
            colors = np.array(
                [
                    [20.0, 46.0, 115.0],
                    [48.0, 98.0, 168.0],
                    [94.0, 159.0, 201.0],
                    [121.0, 173.0, 98.0],
                    [171.0, 146.0, 95.0],
                    [205.0, 190.0, 160.0],
                    [248.0, 248.0, 248.0],
                ],
                dtype=float,
            )
        else:
            # 深海蓝 -> 浅海青 -> 陆地绿 -> 棕色 -> 近白高地
            stops = np.array([0.0, 0.35, 0.5, 0.78, 1.0], dtype=float)
            colors = np.array(
                [
                    [26.0, 58.0, 118.0],
                    [55.0, 126.0, 184.0],
                    [102.0, 166.0, 86.0],
                    [158.0, 122.0, 76.0],
                    [240.0, 240.0, 240.0],
                ],
                dtype=float,
            )
            if pal == "terrain_r":
                colors = colors[::-1].copy()
        rgb = np.empty((n.size, 3), dtype=float)
        flat = n.reshape(-1)
        for c in range(3):
            rgb[:, c] = np.interp(flat, stops, colors[:, c])
        return rgb.reshape(n.shape + (3,))

    def _custom_cpt_rgb_from_values(self, values: np.ndarray) -> Optional[np.ndarray]:
        cpt = self._get_custom_cpt_colormap()
        if cpt is None:
            return None
        z_arr, rgb_arr = cpt
        vals = np.asarray(values, dtype=float)
        flat = vals.reshape(-1)
        rgb = np.empty((flat.size, 3), dtype=float)
        for c in range(3):
            rgb[:, c] = np.interp(flat, z_arr, rgb_arr[:, c], left=rgb_arr[0, c], right=rgb_arr[-1, c])
        return rgb.reshape(vals.shape + (3,))

    def _terrain_rgba_from_grid(
        self,
        z_grid: np.ndarray,
        palette: str = "terrain",
        shade_strength: float = 0.75,
        coast_enhance: bool = True,
        light_alt_deg: float = 45.0,
        light_az_deg: float = 315.0,
    ) -> np.ndarray:
        zf = np.asarray(z_grid, dtype=float)
        zmin = float(np.nanpercentile(zf, 2.0))
        zmax = float(np.nanpercentile(zf, 98.0))
        span = max(1e-12, zmax - zmin)
        norm = np.clip((zf - zmin) / span, 0.0, 1.0)
        pal = str(palette or "terrain").lower()
        if pal == "custom_cpt":
            rgb_custom = self._custom_cpt_rgb_from_values(zf)
            if rgb_custom is None:
                rgb = self._terrain_colormap_rgb(norm, palette="terrain")
            else:
                rgb = rgb_custom
        elif pal == "gmt":
            # GMT风格：海陆分别拉伸色带，海岸更自然
            sea = zf < 0.0
            land = ~sea
            rgb = np.zeros(zf.shape + (3,), dtype=float)
            if np.any(sea):
                z_sea = zf[sea]
                smin = float(np.nanpercentile(z_sea, 2.0))
                smax = float(np.nanpercentile(z_sea, 98.0))
                sspan = max(1e-12, smax - smin)
                s_norm = np.clip((z_sea - smin) / sspan, 0.0, 1.0)
                s_rgb = self._terrain_colormap_rgb(s_norm, palette="gmt")
                rgb[sea] = s_rgb
            if np.any(land):
                z_land = zf[land]
                lmin = float(np.nanpercentile(z_land, 2.0))
                lmax = float(np.nanpercentile(z_land, 98.0))
                lspan = max(1e-12, lmax - lmin)
                l_norm = np.clip((z_land - lmin) / lspan, 0.0, 1.0)
                l_rgb = self._terrain_colormap_rgb(l_norm, palette="terrain")
                rgb[land] = l_rgb
        else:
            rgb = self._terrain_colormap_rgb(norm, palette=pal)
        # 改进光照：近似GMT hillshade
        gy, gx = np.gradient(zf)
        slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
        aspect = np.arctan2(-gx, gy)
        az = np.radians(float(light_az_deg))
        alt = np.radians(float(light_alt_deg))
        hill = (
            np.sin(alt) * np.sin(slope)
            + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
        )
        hill = np.clip((hill + 1.0) * 0.5, 0.0, 1.0)
        strength = float(np.clip(shade_strength, 0.0, 1.0))
        shade = (1.0 - 0.4 * strength) + (0.8 * strength) * hill
        rgb = np.clip(rgb * shade[..., None], 0.0, 255.0)
        if bool(coast_enhance):
            land = zf >= 0.0
            edge = np.zeros_like(land, dtype=bool)
            edge[1:, :] |= land[1:, :] != land[:-1, :]
            edge[:, 1:] |= land[:, 1:] != land[:, :-1]
            # 简单膨胀一圈，让海岸线在缩放时更明显
            edge2 = edge.copy()
            edge2[:-1, :] |= edge[1:, :]
            edge2[1:, :] |= edge[:-1, :]
            edge2[:, :-1] |= edge[:, 1:]
            edge2[:, 1:] |= edge[:, :-1]
            coast_color = np.array([248.0, 242.0, 150.0], dtype=float)
            blend = 0.78
            rgb[edge2] = (1.0 - blend) * rgb[edge2] + blend * coast_color
        rgba = np.zeros(zf.shape + (4,), dtype=np.uint8)
        rgba[..., :3] = rgb.astype(np.uint8)
        rgba[..., 3] = 185
        if bool(coast_enhance):
            rgba[..., 3] = np.where(edge2, 235, rgba[..., 3])
        return rgba

    def _apply_location_map_terrain_overlay(self) -> None:
        if self._location_map_plot_item is None:
            return
        meta = self._location_map_terrain_meta
        if meta is None:
            return

        # 缓存：参数未变化时不重复重建地形底图
        mode = str(meta.get("mode", ""))
        coord_kind = str(meta.get("coord_kind", "unknown")).lower()
        map_mode = str(getattr(self, "_location_map_mode", "none")).lower()
        use_sac2y_tm = bool(getattr(self, "_location_map_terrain_use_sac2y_tm", False))
        pal = str(getattr(self, "_location_map_terrain_palette", "terrain"))
        shade = float(getattr(self, "_location_map_terrain_shade_strength", 0.75))
        light_alt = float(getattr(self, "_location_map_terrain_light_alt_deg", 45.0))
        light_az = float(getattr(self, "_location_map_terrain_light_az_deg", 315.0))
        coast = bool(getattr(self, "_location_map_terrain_coast_enhance", True))
        swap_ll = bool(getattr(self, "_location_map_terrain_swap_lonlat", False))
        force_geo = bool(getattr(self, "_location_map_terrain_force_geo", False))
        cpt_path = str(getattr(self, "_location_map_terrain_cpt_path", "") or "")
        cpt_mtime = 0.0
        if cpt_path:
            try:
                cpt_mtime = float(Path(cpt_path).stat().st_mtime)
            except Exception:
                cpt_mtime = 0.0
        cache_key: Tuple[object, ...] = (
            "terrain",
            str(meta.get("path", "")),
            mode,
            coord_kind,
            map_mode,
            use_sac2y_tm,
            float(getattr(self, "_location_map_terrain_tm_lon0", 120.0)),
            bool(getattr(self, "_location_map_terrain_tm_lon_wrap360", True)),
            bool(getattr(self, "_location_map_terrain_manual_zone_enabled", False)),
            int(getattr(self, "_location_map_terrain_manual_zone_value", 0) or 0),
            str(getattr(self, "_location_map_terrain_manual_hemi", "auto")),
            pal,
            float(shade),
            float(light_alt),
            float(light_az),
            bool(coast),
            bool(swap_ll),
            bool(force_geo),
            cpt_path,
            cpt_mtime,
        )
        if (
            self._location_map_terrain_item is not None
            and self._location_map_terrain_cache_key == cache_key
        ):
            return

        self._location_map_terrain_cache_key = cache_key
        if self._location_map_terrain_item is not None:
            try:
                self._location_map_plot_item.removeItem(self._location_map_terrain_item)
            except Exception:
                pass
            self._location_map_terrain_item = None

        def _set_proj_info_text(text: str) -> None:
            self._location_map_terrain_proj_text = str(text)
            if self._location_map_terrain_proj_label is not None:
                self._location_map_terrain_proj_label.setText(self._location_map_terrain_proj_text)

        def _format_proj_info(info: Dict[str, object]) -> str:
            method = str(info.get("method", "") or "").lower()
            geo_order = str(info.get("geo_order", ""))
            order_text = ""
            if geo_order == "manual_swapped":
                order_text = "，经纬顺序=手动互换"
            elif geo_order == "auto_swapped":
                order_text = "，经纬顺序=自动互换"
            elif geo_order == "auto_default":
                order_text = "，经纬顺序=默认(x=lon,y=lat)"
            if method == "sac2y_tm":
                lon0_tm = float(info.get("lon0_tm", np.nan))
                wrap = bool(info.get("wrap_lon360", True))
                lon_mean = float(info.get("lon_mean", np.nan))
                lat_mean = float(info.get("lat_mean", np.nan))
                return (
                    "投影参数："
                    f"模式=SAC2Y兼容TM，lon0={lon0_tm:.6f}°（中央经线），"
                    f"lon<0加360={'开' if wrap else '关'}，"
                    f"参考点(均值) lon={lon_mean:.6f}°, lat={lat_mean:.6f}°，"
                    f"偏移规则=x+500000，南半球y+10000000（ex/ey未使用）{order_text}"
                )
            zone = int(round(float(info.get("zone", 0.0))))
            zone_auto = int(round(float(info.get("zone_auto", float(zone)))))
            epsg = int(round(float(info.get("epsg", 0.0))))
            lon0 = float(info.get("lon0", np.nan))
            lat0 = float(info.get("lat0", np.nan))
            cm = float(info.get("central_meridian", np.nan))
            hemi = "N" if float(info.get("hemisphere", 1.0)) >= 0.0 else "S"
            zone_mode = str(info.get("zone_mode", "auto"))
            hemi_mode = str(info.get("hemi_mode", "auto"))
            mode_text = "手动" if zone_mode == "manual" else "自动"
            extra = ""
            if zone_mode == "manual" and zone_auto != zone:
                extra = f"（自动建议Zone={zone_auto}）"
            if hemi_mode == "manual":
                extra += "（半球手动）"
            return (
                "投影参数："
                f"EPSG={epsg}，UTM Zone={zone}{hemi}，中央经线={cm:.3f}°，"
                f"参考点(均值) lon={lon0:.6f}°, lat={lat0:.6f}°，"
                f"分带模式={mode_text}{extra}{order_text}"
            )

        if mode == "grid":
            x = np.asarray(meta.get("x", []), dtype=float)
            y = np.asarray(meta.get("y", []), dtype=float)
            z = np.asarray(meta.get("z", []), dtype=float)
            if x.size < 2 or y.size < 2 or z.size == 0:
                raise ValueError("地形网格数据不足")
            if map_mode == "utm" and coord_kind == "geo":
                # 经纬度网格投影到UTM后不再是规则矩形，改为点叠加显示。
                xx, yy = np.meshgrid(x, y, indexing="xy")
                x_flat = xx.reshape(-1)
                y_flat = yy.reshape(-1)
                z_flat = np.asarray(z, dtype=float).reshape(-1)
                valid = np.isfinite(x_flat) & np.isfinite(y_flat) & np.isfinite(z_flat)
                if not np.any(valid):
                    raise ValueError("地形网格无有效点")
                x_flat = x_flat[valid]
                y_flat = y_flat[valid]
                z_flat = z_flat[valid]
                lon_geo, lat_geo, geo_order = self._normalize_geo_lonlat_order(x_flat, y_flat)
                if use_sac2y_tm:
                    lon0_tm = float(getattr(self, "_location_map_terrain_tm_lon0", 120.0))
                    wrap_tm = bool(getattr(self, "_location_map_terrain_tm_lon_wrap360", True))
                    x_utm, y_utm, proj_info = self._convert_lonlat_to_sac2y_tm_arrays(
                        lon_geo,
                        lat_geo,
                        lon0=lon0_tm,
                        wrap_lon360=wrap_tm,
                    )
                else:
                    zone_override, hemi_override = self._get_manual_utm_override()
                    x_utm, y_utm, proj_info = self._convert_lonlat_to_utm_arrays(
                        lon_geo,
                        lat_geo,
                        override_zone=zone_override,
                        override_hemisphere=hemi_override,
                    )
                proj_info["geo_order"] = geo_order
                zmin = float(np.nanmin(z_flat))
                zmax = float(np.nanmax(z_flat))
                span = max(1e-12, zmax - zmin)
                norm = np.clip((z_flat - zmin) / span, 0.0, 1.0)
                valid_xy = np.isfinite(x_utm) & np.isfinite(y_utm) & np.isfinite(norm)
                if not np.any(valid_xy):
                    raise ValueError("地形投影后无有效点（可能坐标顺序或投影参数不匹配）")
                x_utm = np.asarray(x_utm[valid_xy], dtype=float)
                y_utm = np.asarray(y_utm[valid_xy], dtype=float)
                z_plot = np.asarray(z_flat[valid_xy], dtype=float)
                xmin, xmax = float(np.min(x_utm)), float(np.max(x_utm))
                ymin, ymax = float(np.min(y_utm)), float(np.max(y_utm))
                if not (np.isfinite(xmin) and np.isfinite(xmax) and np.isfinite(ymin) and np.isfinite(ymax)):
                    raise ValueError("地形投影范围无效")
                # 将投影后的散点重采样成规则栅格，避免“离散点”观感
                nx = 700
                ny = max(220, int(round(nx * max(1e-9, (ymax - ymin)) / max(1e-9, (xmax - xmin)))))
                ny = min(ny, 900)
                x_edges = np.linspace(xmin, xmax, nx + 1)
                y_edges = np.linspace(ymin, ymax, ny + 1)
                sum_z, _, _ = np.histogram2d(y_utm, x_utm, bins=[y_edges, x_edges], weights=z_plot)
                cnt, _, _ = np.histogram2d(y_utm, x_utm, bins=[y_edges, x_edges])
                with np.errstate(invalid="ignore", divide="ignore"):
                    z_grid = sum_z / cnt
                finite_grid = np.isfinite(z_grid)
                if not np.any(finite_grid):
                    raise ValueError("地形重采样后无有效栅格")
                fill_value = float(np.nanmedian(z_plot))
                z_grid = np.where(np.isfinite(z_grid), z_grid, fill_value)
                pal = str(getattr(self, "_location_map_terrain_palette", "terrain"))
                shade = float(getattr(self, "_location_map_terrain_shade_strength", 0.75))
                coast = bool(getattr(self, "_location_map_terrain_coast_enhance", True))
                alt_deg = float(getattr(self, "_location_map_terrain_light_alt_deg", 45.0))
                az_deg = float(getattr(self, "_location_map_terrain_light_az_deg", 315.0))
                rgba = self._terrain_rgba_from_grid(
                    z_grid,
                    palette=pal,
                    shade_strength=shade,
                    coast_enhance=coast,
                    light_alt_deg=alt_deg,
                    light_az_deg=az_deg,
                )
                # 直接使用栅格顺序，保持与坐标轴方向一致
                img = pg.ImageItem(rgba, axisOrder="row-major")
                img.setRect(QtCore.QRectF(xmin, ymin, xmax - xmin, ymax - ymin))
                img.setZValue(-96)
                self._location_map_plot_item.addItem(img)
                self._location_map_terrain_item = img
                t_bounds = (
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                )
                self._apply_location_map_view_range(t_bounds)
                self._update_location_map_colorbar(float(np.nanmin(z_plot)), float(np.nanmax(z_plot)))
                text = _format_proj_info(proj_info)
                _set_proj_info_text(text)
                self._set_status_text(text, hold_ms=2800)
                return
            # pyqtgraph ImageItem 需要列优先对应 X；这里转置并设置 rect
            img = pg.ImageItem(np.asarray(z.T, dtype=float), axisOrder="col-major")
            xmin, xmax = float(np.min(x)), float(np.max(x))
            ymin, ymax = float(np.min(y)), float(np.max(y))
            img.setRect(QtCore.QRectF(xmin, ymin, xmax - xmin, ymax - ymin))
            img.setOpacity(0.45)
            self._location_map_plot_item.addItem(img)
            img.setZValue(-100)
            self._location_map_terrain_item = img
            t_bounds = (xmin, xmax, ymin, ymax)
            self._apply_location_map_view_range(t_bounds)
            self._update_location_map_colorbar(float(np.nanmin(z)), float(np.nanmax(z)))
            _set_proj_info_text("投影参数：当前地形坐标按UTM/投影坐标直接叠加（未执行经纬度转换）")
            return

        if mode == "points":
            x = np.asarray(meta.get("x", []), dtype=float)
            y = np.asarray(meta.get("y", []), dtype=float)
            z = np.asarray(meta.get("z", []), dtype=float)
            if x.size == 0:
                raise ValueError("地形点数据为空")
            if map_mode == "utm" and coord_kind == "geo":
                lon_geo, lat_geo, geo_order = self._normalize_geo_lonlat_order(x, y)
                if use_sac2y_tm:
                    lon0_tm = float(getattr(self, "_location_map_terrain_tm_lon0", 120.0))
                    wrap_tm = bool(getattr(self, "_location_map_terrain_tm_lon_wrap360", True))
                    x, y, proj_info = self._convert_lonlat_to_sac2y_tm_arrays(
                        lon_geo,
                        lat_geo,
                        lon0=lon0_tm,
                        wrap_lon360=wrap_tm,
                    )
                else:
                    zone_override, hemi_override = self._get_manual_utm_override()
                    x, y, proj_info = self._convert_lonlat_to_utm_arrays(
                        lon_geo,
                        lat_geo,
                        override_zone=zone_override,
                        override_hemisphere=hemi_override,
                    )
                proj_info["geo_order"] = geo_order
                text = _format_proj_info(proj_info)
                _set_proj_info_text(text)
                self._set_status_text(text, hold_ms=2800)
            else:
                _set_proj_info_text("投影参数：当前地形坐标按UTM/投影坐标直接叠加（未执行经纬度转换）")
            zf = np.asarray(z, dtype=float)
            zmin = float(np.nanmin(zf)) if zf.size > 0 else 0.0
            zmax = float(np.nanmax(zf)) if zf.size > 0 else 1.0
            span = max(1e-12, zmax - zmin)
            norm = np.clip((zf - zmin) / span, 0.0, 1.0)
            valid_xy = np.isfinite(x) & np.isfinite(y) & np.isfinite(norm)
            if not np.any(valid_xy):
                raise ValueError("地形点在当前投影下无有效坐标")
            x = np.asarray(x[valid_xy], dtype=float)
            y = np.asarray(y[valid_xy], dtype=float)
            norm = np.asarray(norm[valid_xy], dtype=float)
            zf = np.asarray(zf[valid_xy], dtype=float)
            # 使用全量点构建规则栅格预览，避免海量散点导致界面卡顿。
            xmin, xmax = float(np.min(x)), float(np.max(x))
            ymin, ymax = float(np.min(y)), float(np.max(y))
            nx = 700
            ny = max(220, int(round(nx * max(1e-9, (ymax - ymin)) / max(1e-9, (xmax - xmin)))))
            ny = min(ny, 900)
            x_edges = np.linspace(xmin, xmax, nx + 1)
            y_edges = np.linspace(ymin, ymax, ny + 1)
            sum_z, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges], weights=zf)
            cnt, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])
            with np.errstate(invalid="ignore", divide="ignore"):
                z_grid = sum_z / cnt
            if not np.any(np.isfinite(z_grid)):
                raise ValueError("地形点栅格化失败：无有效网格")
            fill_value = float(np.nanmedian(zf))
            z_grid = np.where(np.isfinite(z_grid), z_grid, fill_value)
            pal = str(getattr(self, "_location_map_terrain_palette", "terrain"))
            shade = float(getattr(self, "_location_map_terrain_shade_strength", 0.75))
            coast = bool(getattr(self, "_location_map_terrain_coast_enhance", True))
            alt_deg = float(getattr(self, "_location_map_terrain_light_alt_deg", 45.0))
            az_deg = float(getattr(self, "_location_map_terrain_light_az_deg", 315.0))
            rgba = self._terrain_rgba_from_grid(
                z_grid,
                palette=pal,
                shade_strength=shade,
                coast_enhance=coast,
                light_alt_deg=alt_deg,
                light_az_deg=az_deg,
            )
            img = pg.ImageItem(rgba, axisOrder="row-major")
            img.setRect(QtCore.QRectF(xmin, ymin, xmax - xmin, ymax - ymin))
            img.setZValue(-90)
            self._location_map_plot_item.addItem(img)
            self._location_map_terrain_item = img
            t_bounds = (xmin, xmax, ymin, ymax)
            self._apply_location_map_view_range(t_bounds)
            self._update_location_map_colorbar(float(np.nanmin(zf)), float(np.nanmax(zf)))
            return

    def _update_location_map_cursor_for_trace(self, trace_idx: int) -> None:
        if self._location_map_cursor_item is None:
            return
        if self._location_map_mode not in ("utm", "geo"):
            self._location_map_cursor_item.setData([], [])
            return
        x, y = self._trace_position_for_map(int(trace_idx), self._location_map_mode)
        if not (math.isfinite(x) and math.isfinite(y)) or (abs(x) < 1e-9 and abs(y) < 1e-9):
            self._location_map_cursor_item.setData([], [])
            return
        self._location_map_cursor_item.setData([float(x)], [float(y)])

    def _update_location_map_selected_for_trace(self, trace_idx: int) -> None:
        if self._location_map_selected_item is None:
            return
        if self._location_map_mode not in ("utm", "geo"):
            self._location_map_selected_item.setData([], [])
            return
        x, y = self._trace_position_for_map(int(trace_idx), self._location_map_mode)
        if not (math.isfinite(x) and math.isfinite(y)) or (abs(x) < 1e-9 and abs(y) < 1e-9):
            self._location_map_selected_item.setData([], [])
            return
        self._location_map_selected_item.setData([float(x)], [float(y)])

    def _on_location_map_mouse_clicked(self, ev, map_widget: pg.PlotWidget) -> None:
        if ev is None or map_widget is None:
            return
        if self._location_map_trace_points.size == 0 or self._location_map_trace_indices.size == 0:
            return
        vb = map_widget.getViewBox()
        pos = ev.scenePos()
        if not vb.sceneBoundingRect().contains(pos):
            return
        mouse_pt = vb.mapSceneToView(pos)
        mx = float(mouse_pt.x())
        my = float(mouse_pt.y())
        pts = self._location_map_trace_points
        idxs = self._location_map_trace_indices
        try:
            idx_filtered = self._extract_indices()
            if idx_filtered.size > 0:
                keep = np.isin(idxs, idx_filtered.astype(int))
                if np.any(keep):
                    pts = pts[keep]
                    idxs = idxs[keep]
        except Exception:
            pass
        dx = pts[:, 0] - mx
        dy = pts[:, 1] - my
        d2 = dx * dx + dy * dy
        if d2.size == 0:
            return
        nearest = int(np.argmin(d2))
        trace_idx = int(idxs[nearest])
        self._jump_to_trace_from_map(trace_idx)

    def _jump_to_trace_from_map(self, trace_idx: int) -> None:
        if self.loaded is None:
            return
        trace_headers = self.loaded.get("trace_headers", []) or []
        if trace_idx < 0 or trace_idx >= len(trace_headers):
            return
        self._map_link_trace_idx = int(trace_idx)
        self.request_render(delay_ms=10)
        th = trace_headers[trace_idx]
        shot = int(getattr(th, "ishoti", 0) or 0)
        try:
            if shot > 0 and int(self.spin_irec.value()) != shot:
                self.spin_irec.setValue(shot)
        except Exception:
            pass
        self.request_render(immediate=True)

        def _center_on_trace():
            if self.loaded is None:
                return
            offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
            if trace_idx < 0 or trace_idx >= offsets.size:
                return
            x0 = float(offsets[trace_idx])
            vb = self.plot.getViewBox()
            xr, yr = vb.viewRange()
            xspan = max(1e-9, float(abs(xr[1] - xr[0])))
            vb.setXRange(x0 - 0.5 * xspan, x0 + 0.5 * xspan, padding=0.0)
            vb.setYRange(float(yr[0]), float(yr[1]), padding=0.0)
            self._sync_window_controls_from_view()
            self._update_location_map_cursor_for_trace(trace_idx)
            self._update_location_map_selected_for_trace(trace_idx)
            self._set_status_text(f"Map定位：已跳转到道 {trace_idx}（shot={shot if shot > 0 else 'N/A'}）", hold_ms=1800)

        QtCore.QTimer.singleShot(30, _center_on_trace)

    def _export_figure(self) -> None:
        out, selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "导出图像",
            "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;PDF (*.pdf);;PostScript (*.ps)",
            options=self._file_dialog_options(),
        )
        if not out:
            return
        try:
            suffix = Path(out).suffix.lower()
            if not suffix:
                if "pdf" in selected.lower():
                    suffix = ".pdf"
                elif "postscript" in selected.lower() or "ps" in selected.lower():
                    suffix = ".ps"
                elif "jpeg" in selected.lower() or "jpg" in selected.lower():
                    suffix = ".jpg"
                elif "bmp" in selected.lower():
                    suffix = ".bmp"
                else:
                    suffix = ".png"
                out = out + suffix

            # 使用 PlotItem 导出，避免 OpenGL/抓屏导致空白图
            exporter = pg_exporters.ImageExporter(self.plot.getPlotItem())
            if suffix in (".png", ".jpg", ".jpeg", ".bmp"):
                exporter.export(out)
            elif suffix in (".pdf", ".ps"):
                # 先离屏导出到临时 png，再写入 pdf/ps
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                    tmp_png = tf.name
                exporter.export(tmp_png)
                import matplotlib.pyplot as plt
                import matplotlib.image as mpimg
                arr = mpimg.imread(tmp_png)
                h, w = int(arr.shape[0]), int(arr.shape[1])

                fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100)
                ax = fig.add_axes([0, 0, 1, 1])
                ax.imshow(arr)
                ax.axis("off")
                fig.savefig(out, format=suffix.lstrip("."), dpi=300, bbox_inches="tight", pad_inches=0)
                plt.close(fig)
                try:
                    os.remove(tmp_png)
                except Exception:
                    pass
            else:
                raise RuntimeError(f"不支持的导出格式: {suffix}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "导出失败", f"图像导出失败：{exc}")
            return
        self.lbl_status.setText(f"图像已导出：{out}")

    def _on_show_stack_changed(self, _state: int) -> None:
        self.request_render()
        if self.chk_show_stack.isChecked():
            self.lbl_status.setText("叠加道已开启：显示在主图右侧蓝色曲线")
        else:
            self.lbl_status.setText("叠加道已关闭")

    def _available_records(self) -> List[int]:
        if self.loaded is None:
            return []
        headers = self.loaded.get("trace_headers", [])
        recs = sorted({int(getattr(h, "ishoti", 0) or 0) for h in headers if int(getattr(h, "ishoti", 0) or 0) > 0})
        return recs

    def _prev_record(self) -> None:
        recs = self._available_records()
        if not recs:
            return
        cur = int(self.spin_irec.value())
        if cur <= 0:
            self.spin_irec.setValue(recs[-1])
            return
        smaller = [r for r in recs if r < cur]
        self.spin_irec.setValue(smaller[-1] if smaller else recs[-1])

    def _next_record(self) -> None:
        recs = self._available_records()
        if not recs:
            return
        cur = int(self.spin_irec.value())
        if cur <= 0:
            self.spin_irec.setValue(recs[0])
            return
        bigger = [r for r in recs if r > cur]
        self.spin_irec.setValue(bigger[0] if bigger else recs[0])

    def _show_trace_info(self) -> None:
        if self.loaded is None:
            self.lbl_status.setText("提示：请先加载数据")
            return
        offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
        if offsets.size == 0:
            self.lbl_status.setText("提示：没有可用道数据")
            return
        idx = self._extract_indices()
        if idx.size == 0:
            self.lbl_status.setText("提示：当前过滤条件下无道")
            return
        xref = self.mouse_x
        yref = self.mouse_y
        if xref is None or yref is None:
            vr = self.plot.getViewBox().viewRange()
            xref = float((vr[0][0] + vr[0][1]) * 0.5)
            yref = float((vr[1][0] + vr[1][1]) * 0.5)
        idx_offsets = offsets[idx]
        nearest_i = int(np.argmin(np.abs(idx_offsets - float(xref))))
        trace_idx = int(idx[nearest_i])
        headers = self.loaded.get("trace_headers", [])
        th = headers[trace_idx] if (trace_idx < len(headers)) else None

        shot = int(getattr(th, "ishoti", 0) or 0) if th is not None else 0
        rec = int(getattr(th, "ireci", 0) or 0) if th is not None else 0
        itsn = int(getattr(th, "itsn", trace_idx) or trace_idx) if th is not None else trace_idx
        dead = bool(int(getattr(th, "iflagi", 1) or 1) != 1) if th is not None else False
        azi = float(getattr(th, "azi", 0.0) or 0.0) if th is not None else 0.0
        off = float(offsets[trace_idx])
        cursor_time_display = float(yref)
        cursor_time_true = cursor_time_display - self._compute_display_tshift(trace_idx, off)

        pick_lines: List[str] = []
        if th is not None and getattr(th, "picks", None):
            for pi, pv in enumerate(getattr(th, "picks", []), start=1):
                if float(pv) > 0:
                    pick_lines.append(f"拾取字{pi}: {float(pv):.4f} s")
        if not pick_lines and self.pick_manager is not None:
            for pi in range(1, int(self.spin_apick.maximum()) + 1):
                pt = self.pick_manager.get_pick(trace_idx, pi)
                if pt is not None and float(pt) > 0:
                    pick_lines.append(f"拾取字{pi}: {float(pt):.4f} s")
        picks_text = "\n".join(pick_lines) if pick_lines else "无"

        msg = (
            f"道索引: {trace_idx}\n"
            f"炮站号(ishoti): {shot if shot > 0 else 'N/A'}\n"
            f"接收站号(ireci): {rec if rec > 0 else 'N/A'}\n"
            f"道序号(itsn): {itsn}\n"
            f"死道标志: {'是' if dead else '否'}\n"
            f"炮检距(offset): {off:.4f} km\n"
            f"方位角(azi): {azi:.2f}°\n"
            f"光标时间(显示): {cursor_time_display:.4f} s\n"
            f"光标时间(真实): {cursor_time_true:.4f} s\n\n"
            f"拾取时间:\n{picks_text}"
        )
        self._show_themed_info("道信息", msg)
        self.lbl_status.setText(
            f"道 {trace_idx}: offset={off:.3f}km, 炮站={shot or 'N/A'}, 显示={cursor_time_display:.3f}s, 真实={cursor_time_true:.3f}s"
        )

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

    def _show_help(self) -> None:
        text = (
            "ZPLOT Qt 版帮助\n\n"
            "基本：\n"
            "- 打开 .z/.hdr/.r 后进入交互\n"
            "- P 切换拾取模式，左键加点，右键删当前拾取字\n"
            "- 拾取模式下按住 Shift 并移动鼠标：经过即拾取（每道一次）\n"
            "- Ctrl+Z 撤销最近一次拾取修改\n"
            "- Ctrl+Y 重做最近一次撤销\n"
            "- S 写入HDR，Ctrl+S 保存 zplot.out，导出 tx.in 会先写入HDR\n\n"
            "拾取与处理：\n"
            "- C 插值相关拾取（需至少2个种子）\n"
            "- A 波形临时对齐（再次按 A 清除）\n"
            "- F 自适应拾取更新（仅改拾取时间，不移动波形）\n"
            "- V 以鼠标位置添加选波窗口（前0.3s、后0.7s，可多选）\n"
            "- Shift+V 删除最近一个选波窗口\n"
            "- M 启动多边形 mute：左键加点/拖拽顶点，右键点顶点删除，右键空白闭合；闭合后左键点顶点可直接进入编辑\n"
            "- Shift+M 切换反选（保留多边形外部）\n"
            "- Delete 删除当前选中顶点（绘制态）\n"
            "- 按 M 可启用/取消 mute（已有多边形时）；“清空Mute”会删除所有顶点并重置\n"
            "- D-D 两次按键按偏移范围批量删除当前拾取字\n"
            "- X 移除/恢复最近道\n\n"
            "视图：\n"
            "- Z 放大，O 缩小（以鼠标为中心）\n"
            "- Left/Right 上一炮/下一炮\n"
            "- I 查看最近道详细信息\n\n"
            "高级：\n"
            "- T 计算理论走时，Shift+T 清除理论走时\n"
            "- W 计算水层校正（需先有理论走时）\n"
            "- Shift+W 显示水层校正距离-校正量曲线"
        )
        self._show_themed_info("帮助", text)

    def _show_shortcuts(self) -> None:
        text = (
            "快捷键\n\n"
            "文件与界面:\n"
            "Ctrl+O 打开数据, Ctrl+R 重绘, Ctrl+S 保存拾取, Q 退出\n\n"
            "Ctrl+Z 撤销最近一次拾取修改, Ctrl+Y 重做最近一次撤销\n\n"
            "拾取:\n"
            "P 拾取模式, 1/2/3 切拾取字, [/ ] 前后拾取字, S 写HDR\n"
            "Shift+鼠标移动 连续拾取(每道一次), C 插值相关, A 波形临时对齐(二次按A清除), "
            "F 自适应拾取更新(仅改拾取时间), V 选波窗口, Shift+V 删最近V段, M 多边形mute(可拖拽顶点), Shift+M 反选, Delete 删顶点, D-D 范围删, X 移除/恢复道\n\n"
            "视图:\n"
            "Z 放大, O 缩小, Left/Right 上一炮/下一炮, I 道信息, H 帮助, F1 快捷键\n\n"
            "高级:\n"
            "T 理论走时, Shift+T 清除理论, W 水层校正, Shift+W 校正曲线"
        )
        self._show_themed_info("快捷键", text)

    def _show_about(self) -> None:
        text = (
            "ZPLOT Qt Fast Viewer\n"
            "pyAOBS.visualization.zplotpy 最终 GUI 版本\n\n"
            "当前版本已作为 zplotpy 默认图形界面。\n"
            "可通过 `python -m pyAOBS.visualization.zplotpy.gui` 启动，"
            "Workbench 的 zplotpy 节点也已统一调用此版本。\n\n"
            "核心能力：大文件快速渲染、拾取、自动/插值相关拾取、"
            "波形对齐、HDR 写入、tx.in 导出、参数保存/加载。"
        )
        self._show_themed_info("关于", text)

    def _calculate_theoretical_traveltime_dialog(self) -> None:
        if self.loaded is None or self.pick_manager is None:
            self.lbl_status.setText("请先加载数据")
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("计算理论走时")
        dialog.resize(640, 360)
        layout = QtWidgets.QFormLayout(dialog)

        model_edit = QtWidgets.QLineEdit(self._theory_model_file or "", dialog)
        browse_btn = QtWidgets.QPushButton("浏览", dialog)
        model_row = QtWidgets.QHBoxLayout()
        model_row.addWidget(model_edit)
        model_row.addWidget(browse_btn)
        model_wrap = QtWidgets.QWidget(dialog)
        model_wrap.setLayout(model_row)
        layout.addRow("模型文件(v.in)", model_wrap)

        shot_x = QtWidgets.QDoubleSpinBox(dialog)
        shot_x.setRange(-1e6, 1e6)
        shot_x.setDecimals(3)
        shot_x.setValue(0.0)
        shot_z = QtWidgets.QDoubleSpinBox(dialog)
        shot_z.setRange(-1e6, 1e6)
        shot_z.setDecimals(3)
        shot_z.setValue(0.0)
        shot_auto = QtWidgets.QCheckBox("自动从数据提取炮点", dialog)
        shot_auto.setChecked(True)
        shot_row = QtWidgets.QHBoxLayout()
        shot_row.addWidget(QtWidgets.QLabel("X"))
        shot_row.addWidget(shot_x)
        shot_row.addWidget(QtWidgets.QLabel("Z"))
        shot_row.addWidget(shot_z)
        shot_row.addWidget(shot_auto)
        shot_wrap = QtWidgets.QWidget(dialog)
        shot_wrap.setLayout(shot_row)
        layout.addRow("炮点位置(km)", shot_wrap)

        ray_edit = QtWidgets.QLineEdit("1.2", dialog)
        nray_spin = QtWidgets.QSpinBox(dialog)
        nray_spin.setRange(1, 2000)
        nray_spin.setValue(10)
        xmin_spin = QtWidgets.QDoubleSpinBox(dialog)
        xmax_spin = QtWidgets.QDoubleSpinBox(dialog)
        zmin_spin = QtWidgets.QDoubleSpinBox(dialog)
        zmax_spin = QtWidgets.QDoubleSpinBox(dialog)
        for s in (xmin_spin, xmax_spin, zmin_spin, zmax_spin):
            s.setRange(-1e6, 1e6)
            s.setDecimals(3)
        range_auto = QtWidgets.QCheckBox("自动使用模型范围", dialog)
        range_auto.setChecked(True)
        range_row = QtWidgets.QHBoxLayout()
        range_row.addWidget(QtWidgets.QLabel("xmin"))
        range_row.addWidget(xmin_spin)
        range_row.addWidget(QtWidgets.QLabel("xmax"))
        range_row.addWidget(xmax_spin)
        range_row.addWidget(QtWidgets.QLabel("zmin"))
        range_row.addWidget(zmin_spin)
        range_row.addWidget(QtWidgets.QLabel("zmax"))
        range_row.addWidget(zmax_spin)
        range_row.addWidget(range_auto)
        range_wrap = QtWidgets.QWidget(dialog)
        range_wrap.setLayout(range_row)
        layout.addRow("ray参数", ray_edit)
        layout.addRow("nray", nray_spin)
        layout.addRow("范围(km)", range_wrap)

        use_picks = QtWidgets.QCheckBox("使用观测拾取生成 tx.in", dialog)
        use_picks.setChecked(False)
        layout.addRow(use_picks)

        def _browse_model():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                dialog, "选择速度模型文件", "", "v.in files (*.in *.vin);;All files (*)", options=self._file_dialog_options()
            )
            if path:
                model_edit.setText(path)

        browse_btn.clicked.connect(_browse_model)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return

        model_file = model_edit.text().strip()
        if not model_file or not Path(model_file).exists():
            QtWidgets.QMessageBox.warning(self, "参数错误", "请提供有效的模型文件路径。")
            return
        self._theory_model_file = model_file

        shot_position = None if shot_auto.isChecked() else (float(shot_x.value()), float(shot_z.value()))
        try:
            ray_values = [float(x.strip()) for x in ray_edit.text().split(",") if x.strip()]
            if not ray_values:
                ray_values = [1.2]
        except Exception:
            QtWidgets.QMessageBox.warning(self, "参数错误", "ray 参数格式错误，请输入如 1.2 或 1.1,2.1")
            return

        ray_params: Dict[str, object] = {
            "ray": ray_values,
            "nray": int(nray_spin.value()),
        }
        if not range_auto.isChecked():
            ray_params.update(
                {
                    "xmin": float(xmin_spin.value()),
                    "xmax": float(xmax_spin.value()),
                    "zmin": float(zmin_spin.value()),
                    "zmax": float(zmax_spin.value()),
                }
            )

        self._calculate_theoretical_traveltime(
            model_file=model_file,
            shot_position=shot_position,
            ray_params=ray_params,
            use_observed_picks=bool(use_picks.isChecked()),
        )

    def _calculate_theoretical_traveltime(
        self,
        model_file: str,
        shot_position: Optional[tuple[float, float]] = None,
        ray_params: Optional[Dict[str, object]] = None,
        use_observed_picks: bool = False,
    ) -> None:
        if self.loaded is None or self.pick_manager is None:
            return
        try:
            self.lbl_status.setText("正在计算理论走时...")
            calc = TheoreticalTravelTimeCalculator(
                model_file_path=model_file,
                data_loader=self.loader,
                pick_manager=self.pick_manager,
            )
            model_info = calc.get_model_info()
            if not model_info.get("has_model"):
                self.lbl_status.setText("理论走时失败：模型加载失败")
                return
            if model_info.get("model_type") != "vin":
                self.lbl_status.setText("理论走时失败：仅支持 v.in 模型")
                return
            success = calc.calculate_travel_times(
                auto_generate_inputs=True,
                shot_position=shot_position,
                ray_params=ray_params,
                use_observed_picks=use_observed_picks,
                pick_word=int(self.spin_apick.value()),
            )
            if not success:
                self.lbl_status.setText("理论走时失败：RAYINVR 计算失败")
                return
            self.theoretical_traveltime_calculator = calc
            self.show_theoretical_times = True
            self.theoretical_times_data = None
            self.show_water_layer_correction = False
            self.water_layer_corrected_times = None
            self.request_render(immediate=True)
            self.lbl_status.setText("理论走时计算完成")
        except Exception as exc:
            self.lbl_status.setText(f"理论走时失败: {exc}")

    def _clear_theoretical_traveltime(self) -> None:
        self.theoretical_traveltime_calculator = None
        self.theoretical_times_data = None
        self.show_theoretical_times = False
        self.water_layer_corrections = {}
        self.water_layer_corrected_times = None
        self.show_water_layer_correction = False
        self._clear_theoretical_item()
        self._clear_water_corr_item()
        self.request_render(delay_ms=10)
        self.lbl_status.setText("已清除理论走时")

    def _calculate_water_layer_correction_dialog(self) -> None:
        if self.theoretical_traveltime_calculator is None:
            self.lbl_status.setText("请先计算理论走时")
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("计算水层校正")
        dialog.resize(420, 220)
        layout = QtWidgets.QFormLayout(dialog)
        water_depth = self.theoretical_traveltime_calculator.get_water_layer_depth()
        if isinstance(water_depth, (int, float)):
            depth_text = f"{float(water_depth):.3f} km"
        elif isinstance(water_depth, tuple) and len(water_depth) == 2:
            depth_vals = np.asarray(water_depth[1], dtype=float)
            depth_text = f"[{float(np.min(depth_vals)):.3f}, {float(np.max(depth_vals)):.3f}] km"
        else:
            depth_text = "无法自动提取"
        layout.addRow("水层深度", QtWidgets.QLabel(depth_text, dialog))

        vwater = QtWidgets.QDoubleSpinBox(dialog)
        vwater.setRange(0.1, 10.0)
        vwater.setDecimals(3)
        vwater.setValue(1.5)
        vrepl = QtWidgets.QDoubleSpinBox(dialog)
        vrepl.setRange(0.0, 10.0)
        vrepl.setDecimals(3)
        vrepl.setValue(0.0)
        layout.addRow("水层速度(km/s)", vwater)
        layout.addRow("替换速度(km/s,0=自动)", vrepl)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        v_replacement = float(vrepl.value()) if float(vrepl.value()) > 0 else None
        self._calculate_water_layer_correction(v_water=float(vwater.value()), v_replacement=v_replacement)

    def _calculate_water_layer_correction(
        self, v_water: float = 1.5, v_replacement: Optional[float] = None
    ) -> None:
        calc = self.theoretical_traveltime_calculator
        if calc is None:
            self.lbl_status.setText("请先计算理论走时")
            return
        try:
            rays = calc.get_all_rays(max_rays=1000)
            if not rays:
                self.lbl_status.setText("水层校正失败：未获取射线")
                return
            corrections = calc.calculate_water_layer_correction(
                rays=rays,
                water_depth=None,
                v_water=float(v_water),
                v_replacement=v_replacement,
                return_by_distance=False,
            )
            if not corrections:
                self.lbl_status.setText("水层校正失败：未得到有效校正量")
                return
            avg_correction = float(np.mean(np.asarray(list(corrections.values()), dtype=float)))
            self.water_layer_corrections = {int(k): float(v) for k, v in corrections.items()}
            # 按射线终点距离建立“距离->校正量”映射，用于逐点校正理论走时曲线
            dist_vals: List[float] = []
            corr_vals: List[float] = []
            for ray_idx, corr in self.water_layer_corrections.items():
                if ray_idx < 0 or ray_idx >= len(rays):
                    continue
                ray = rays[ray_idx]
                xarr = np.asarray(ray.get("x", []), dtype=float)
                if xarr.size == 0:
                    continue
                dist_vals.append(float(xarr[-1]))
                corr_vals.append(float(corr))

            distance_grid = np.array([], dtype=float)
            correction_grid = np.array([], dtype=float)
            if dist_vals:
                d = np.asarray(dist_vals, dtype=float)
                c = np.asarray(corr_vals, dtype=float)
                order = np.argsort(d)
                d = d[order]
                c = c[order]
                # 合并重复距离（取均值），保证后续插值单调
                uniq_d, inv = np.unique(d, return_inverse=True)
                uniq_c = np.zeros_like(uniq_d, dtype=float)
                cnt = np.zeros_like(uniq_d, dtype=float)
                for i, gid in enumerate(inv):
                    uniq_c[gid] += c[i]
                    cnt[gid] += 1.0
                cnt[cnt <= 0] = 1.0
                uniq_c = uniq_c / cnt
                distance_grid = uniq_d
                correction_grid = uniq_c

            self.show_water_layer_correction = True
            self.water_layer_corrected_times = {
                "avg_correction": np.array([avg_correction], dtype=float),
                "distances": distance_grid,
                "corrections": correction_grid,
            }
            self.request_render(immediate=True)
            self.lbl_status.setText(
                f"水层校正完成：{len(corrections)}条射线, 平均校正 {avg_correction:.6f}s, 距离映射点 {int(distance_grid.size)}"
            )
        except Exception as exc:
            self.lbl_status.setText(f"水层校正失败: {exc}")

    def _clear_water_layer_correction(self) -> None:
        self.water_layer_corrections = {}
        self.water_layer_corrected_times = None
        self.show_water_layer_correction = False
        self._clear_water_corr_item()
        self.request_render(delay_ms=10)
        self.lbl_status.setText("已清除水层校正")

    def _show_water_correction_curve(self) -> None:
        if not self.water_layer_corrected_times:
            self.lbl_status.setText("请先计算水层校正")
            return
        dmap = np.asarray(self.water_layer_corrected_times.get("distances", []), dtype=float)
        cmap = np.asarray(self.water_layer_corrected_times.get("corrections", []), dtype=float)
        if dmap.size == 0 or cmap.size == 0 or dmap.size != cmap.size:
            self.lbl_status.setText("水层校正曲线不可用：缺少有效距离映射点")
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("水层校正曲线")
        dlg.resize(760, 420)
        lay = QtWidgets.QVBoxLayout(dlg)

        plot = pg.PlotWidget(background=self._theme_color("plot_bg", "#ffffff"))
        plot.showGrid(x=True, y=True, alpha=float(self._theme_color("plot_grid_alpha", "0.15")))
        plot.getPlotItem().setLabels(left="Correction (s)", bottom="Distance (km)")
        axis_pen = pg.mkPen(self._theme_color("plot_axis", "#1f2937"), width=1)
        for axis_name in ("left", "bottom"):
            axis = plot.getPlotItem().getAxis(axis_name)
            axis.setPen(axis_pen)
            axis.setTextPen(axis_pen)
        curve = pg.PlotDataItem(
            x=dmap,
            y=cmap,
            pen=pg.mkPen(self._theme_color("water_pen", "#1ea0d2"), width=2),
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush(self._theme_color("water_pen", "#1ea0d2")),
            symbolPen=pg.mkPen(self._theme_color("water_pen", "#1ea0d2"), width=1),
        )
        plot.addItem(curve)
        lay.addWidget(plot, stretch=1)

        stats = (
            f"点数={dmap.size}  距离范围=[{float(np.min(dmap)):.3f}, {float(np.max(dmap)):.3f}] km  "
            f"校正范围=[{float(np.min(cmap)):.6f}, {float(np.max(cmap)):.6f}] s"
        )
        lay.addWidget(QtWidgets.QLabel(stats, dlg))

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close, parent=dlg)
        btns.rejected.connect(dlg.reject)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)

        dlg.exec()

    def _trace_header_source_label(self) -> str:
        """返回当前道头来源标签。"""
        if self.loaded is None:
            return "-"
        trace_headers = self.loaded.get("trace_headers", []) or []
        if not trace_headers:
            return "无道头"
        from_header_flags = [
            bool(getattr(th, "picks_from_header", False))
            for th in trace_headers
            if hasattr(th, "picks_from_header")
        ]
        if from_header_flags:
            if all(from_header_flags):
                return ".hdr"
            if not any(from_header_flags):
                return ".z内嵌"
            return "混合(.hdr/.z)"
        # 兼容缺少标记的情况：根据是否加载了有效 hfile 推断
        if self._hfile and Path(self._hfile).exists():
            return ".hdr(推断)"
        return ".z/默认(推断)"

    def _update_file_open_status_label(self) -> None:
        """更新右下角文件打开状态与道头来源（持久显示）。"""
        if not hasattr(self, "lbl_pick_link_mode"):
            return
        parts: List[str] = []
        if self._dfile and Path(self._dfile).exists():
            parts.append(f"{Path(self._dfile).name} 打开")
        elif self._dfile:
            parts.append(".z关闭")

        if self._hfile and Path(self._hfile).exists():
            parts.append(f"{Path(self._hfile).name} 打开")
        else:
            parts.append(".hdr关闭")

        # 若尚未选择任何文件，右侧留空
        if (not self._dfile) and (not self._hfile):
            self.lbl_pick_link_mode.setText("")
        else:
            self.lbl_pick_link_mode.setText(" / ".join(parts))

    def _load_data(self) -> None:
        if not self._dfile:
            return
        t0 = time.perf_counter()
        try:
            self.loaded = self.loader.load_z_format(
                self._dfile,
                hfile=self._hfile if self._hfile and Path(self._hfile).exists() else None,
                rfile=self._rfile if self._rfile and Path(self._rfile).exists() else None,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "加载失败", str(exc))
            return
        self._clear_denoise_cache()

        header = self.loaded.get("header")
        ntr = int(getattr(header, "ntraces", 0) or 0)
        npts = int(getattr(header, "npts", 0) or 0)
        # irec 过滤实际按 trace_headers.ishoti（炮号）匹配，不一定是 1..nrec 连续编号。
        # 若上限仅设为 nrec，可能导致“下一炮”目标值被 QSpinBox 截断到无效炮号，进而无波形。
        headers = self.loaded.get("trace_headers", [])
        recs = [
            int(getattr(h, "ishoti", 0) or 0)
            for h in headers
            if int(getattr(h, "ishoti", 0) or 0) > 0
        ]
        max_shot_id = max(recs) if recs else 0
        nrec_header = int(getattr(header, "nrec", 0) or 0)
        self.spin_irec.setMaximum(max(0, nrec_header, max_shot_id))
        self.btn_reload.setEnabled(True)
        self.btn_save_params.setEnabled(True)
        self.btn_save_z.setEnabled(True)
        self.btn_data_info.setEnabled(True)
        self.btn_coord_params.setEnabled(True)
        self.btn_location_map.setEnabled(True)
        self.btn_export_fig.setEnabled(True)
        self.btn_prev_rec.setEnabled(True)
        self.btn_next_rec.setEnabled(True)
        self.btn_theory.setEnabled(True)
        self.btn_clear_theory.setEnabled(True)
        self.btn_water_corr.setEnabled(True)
        self.btn_clear_water.setEnabled(True)
        self.btn_water_curve.setEnabled(True)
        self.btn_load_txin.setEnabled(True)
        self.btn_clear_txin.setEnabled(True)
        self.btn_preview_map_txin.setEnabled(True)
        self.btn_map_txin.setEnabled(True)
        self.chk_map_txin_apick_only.setEnabled(True)
        self.chk_map_txin_view_only.setEnabled(True)
        self.spin_map_txin_tol.setEnabled(True)
        self.btn_save_picks.setEnabled(True)
        self.btn_save_hdr.setEnabled(True)
        self.btn_export_tx.setEnabled(True)
        self.btn_clear_picks.setEnabled(True)
        self.btn_auto_pick.setEnabled(True)
        self.btn_interp_pick.setEnabled(True)
        self.btn_align_pick.setEnabled(True)
        self.btn_align_adaptive.setEnabled(True)
        self.btn_eval_stack.setEnabled(True)
        self.btn_waveop_stack.setEnabled(True)
        self.btn_waveop_att.setEnabled(True)
        self.btn_waveop_clear.setEnabled(True)
        self.btn_waveop_save.setEnabled(True)
        self.btn_waveop_load.setEnabled(True)
        self.btn_static_corr.setEnabled(True)
        self.btn_clear_static.setEnabled(True)
        self.btn_clear_align.setEnabled(True)
        self.plot.clear()
        self._curve_items.clear()
        self._shade_item = None
        self._pick_item = None
        self._stack_item = None
        self._static_preview_item = None
        self._theoretical_item = None
        self._txin_item = None
        self._txin_map_preview_item = None
        self._water_corr_item = None
        self._wave_select_item = None
        self._wave_select_marker_item = None
        self._waveop_stack_item = None
        self._mute_polygon_item = None
        self._mute_vertex_item = None
        self._did_initial_view_fit = False
        self._alignment_offsets = {}
        self.static_corrector.clear_corrections()
        self.static_correction_enabled = False
        self.static_preview_mode = False
        self.last_stacking_result = None
        self.theoretical_traveltime_calculator = None
        self.theoretical_times_data = None
        self.show_theoretical_times = False
        self.txin_overlay_data = None
        self.show_txin_overlay = False
        self.txin_map_preview_data = None
        self.water_layer_corrections = {}
        self.water_layer_corrected_times = None
        self.show_water_layer_correction = False
        self._removed_traces = set()
        self.waveform_selections = []
        self.waveop_stack_result = None
        self._mute_edit_mode = False
        self._mute_enabled = False
        self._mute_invert = False
        self._mute_polygon_points = []
        self._mute_drag_vertex_idx = None
        self._mute_selected_vertex_idx = None
        self._mute_drag_active = False
        self._mute_drag_last_status_ms = 0
        self._sync_plot_pan_lock_state()
        self._update_mute_status_button()
        self._set_orientation_main_preview(False, keep_cache=False)
        self._refresh_waveop_selection_list()
        npick = int(getattr(header, "npick", 10) or 10)
        self.pick_manager = PickManager(npick=npick)
        self._pick_undo_stack.clear()
        self._pick_redo_stack.clear()
        self._update_undo_button_state()
        self.spin_apick.setMaximum(max(1, npick))
        self.spin_apick.setValue(min(self.spin_apick.value(), max(1, npick)))
        offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
        times = np.asarray(self.loaded.get("times", []), dtype=float)
        if offsets.size > 0:
            self.spin_xmin.setValue(float(np.min(offsets)))
            self.spin_xmax.setValue(float(np.max(offsets)))
        if times.size > 0:
            self.spin_tmin.setValue(float(np.min(times)))
            self.spin_tmax.setValue(float(np.max(times)))
        self._sync_window_controls_from_view()
        trace_headers = self.loaded.get("trace_headers", [])
        if self.pick_manager is not None and trace_headers:
            self.pick_manager.set_trace_info_batch(trace_headers)
        self.request_render(immediate=True)
        dt = (time.perf_counter() - t0) * 1000.0
        self._update_file_open_status_label()
        self.lbl_status.setText(f"加载完成: ntr={ntr}, npts={npts}, {dt:.1f} ms")

    def _snapshot_pick_state(self) -> Dict[int, Dict[int, float]]:
        if self.pick_manager is None:
            return {}
        snap = self.pick_manager.get_all_picks()
        return {
            int(trace_idx): {int(word): float(tpk) for word, tpk in by_word.items()}
            for trace_idx, by_word in snap.items()
        }

    def _push_pick_undo(self, reason: str) -> None:
        if self.pick_manager is None:
            return
        self._pick_undo_stack.append((str(reason), self._snapshot_pick_state()))
        if len(self._pick_undo_stack) > int(self._pick_undo_limit):
            self._pick_undo_stack = self._pick_undo_stack[-int(self._pick_undo_limit):]
        self._pick_redo_stack.clear()
        self._update_undo_button_state()

    def _restore_pick_state(self, snapshot: Dict[int, Dict[int, float]]) -> None:
        if self.pick_manager is None:
            return
        self.pick_manager.clear_picks()
        for trace_idx, by_word in snapshot.items():
            for pick_word, tpk in by_word.items():
                self.pick_manager.add_pick(int(trace_idx), float(tpk), int(pick_word))

    def _update_undo_button_state(self) -> None:
        has_undo = self.pick_manager is not None and len(self._pick_undo_stack) > 0
        has_redo = self.pick_manager is not None and len(self._pick_redo_stack) > 0
        self.btn_undo_pick.setEnabled(bool(has_undo))
        self.btn_redo_pick.setEnabled(bool(has_redo))

    def _undo_last_pick_edit(self) -> None:
        if self.pick_manager is None:
            return
        if not self._pick_undo_stack:
            self.lbl_status.setText("撤销失败：没有可撤销的拾取操作")
            self._update_undo_button_state()
            return
        reason, snapshot = self._pick_undo_stack.pop()
        self._pick_redo_stack.append((reason, self._snapshot_pick_state()))
        if len(self._pick_redo_stack) > int(self._pick_undo_limit):
            self._pick_redo_stack = self._pick_redo_stack[-int(self._pick_undo_limit):]
        self._restore_pick_state(snapshot)
        self._update_undo_button_state()
        self.request_render(delay_ms=10)
        self.lbl_status.setText(f"已撤销：{reason}")

    def _redo_last_pick_edit(self) -> None:
        if self.pick_manager is None:
            return
        if not self._pick_redo_stack:
            self.lbl_status.setText("重做失败：没有可重做的拾取操作")
            self._update_undo_button_state()
            return
        reason, snapshot = self._pick_redo_stack.pop()
        self._pick_undo_stack.append((reason, self._snapshot_pick_state()))
        if len(self._pick_undo_stack) > int(self._pick_undo_limit):
            self._pick_undo_stack = self._pick_undo_stack[-int(self._pick_undo_limit):]
        self._restore_pick_state(snapshot)
        self._update_undo_button_state()
        self.request_render(delay_ms=10)
        self.lbl_status.setText(f"已重做：{reason}")

    def _on_view_range_changed(self, *_args) -> None:
        if self.loaded is None:
            return
        self._sync_window_controls_from_view()
        self._viewport_interacting = True
        self.request_render(delay_ms=20)
        self._interaction_end_timer.start(220)

    def _on_interaction_end(self) -> None:
        self._viewport_interacting = False
        self._sync_window_controls_from_view()
        self.request_render(delay_ms=30)

    def request_render(self, delay_ms: int = 60, immediate: bool = False) -> None:
        if self.loaded is None:
            return
        if immediate:
            self._render_timer.stop()
            self._render_now()
            return
        self._render_timer.start(max(0, int(delay_ms)))

    def _extract_indices(self, include_removed: bool = False) -> np.ndarray:
        traces = self.loaded.get("traces", [])
        ntr = len(traces)
        if ntr == 0:
            return np.empty((0,), dtype=int)

        idx = np.arange(ntr, dtype=int)
        headers = self.loaded.get("trace_headers", [])

        irec = int(self.spin_irec.value())
        if irec > 0 and headers:
            mask = np.array(
                [int(getattr(headers[i], "ishoti", 0) or 0) == irec for i in idx],
                dtype=bool,
            )
            idx = idx[mask]

        itype = int(self.combo_itype.currentIndex())  # 0..4
        if itype > 0 and headers and idx.size > 0:
            mask = np.array(
                [int(getattr(headers[i], "itypei", 0) or 0) == itype for i in idx],
                dtype=bool,
            )
            idx = idx[mask]

        nskip = int(self.spin_nskip.value())
        if nskip > 0 and idx.size > 0:
            step = nskip + 1
            idx = idx[::step]
        offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
        if offsets.size > 0 and idx.size > 0:
            xmin = float(min(self.spin_xmin.value(), self.spin_xmax.value()))
            xmax = float(max(self.spin_xmin.value(), self.spin_xmax.value()))
            if xmin < xmax:
                mask = np.array(
                    [xmin <= float(offsets[int(i)]) <= xmax for i in idx],
                    dtype=bool,
                )
                idx = idx[mask]
        if (not include_removed) and self._removed_traces and idx.size > 0:
            removed = self._removed_traces
            idx = np.array([int(i) for i in idx if int(i) not in removed], dtype=int)
        return idx

    def _compute_display_tshift(self, trace_idx: int, x_offset: Optional[float] = None) -> float:
        """计算某道在当前显示坐标下的时间平移（含对齐/静校正/折合速度）。"""
        tshift = float(self._alignment_offsets.get(int(trace_idx), 0.0))
        if self.static_correction_enabled:
            tshift += float(self.static_corrector.get_correction(int(trace_idx)))
        tshift += self._compute_reduction_tshift(trace_idx, x_offset)
        if self._orientation_preview_enabled and self._orientation_preview_solution:
            tshift += float(self._orientation_preview_solution.get("time_shift_sec", 0.0))
        return tshift

    def _compute_reduction_tshift(self, trace_idx: int, x_offset: Optional[float] = None) -> float:
        """仅计算折合速度对应的时间平移（不含对齐/静校正）。"""
        if self.params.vred <= 0:
            return 0.0
        x = float(x_offset) if x_offset is not None else 0.0
        if x_offset is None and self.loaded is not None:
            offsets_all = np.asarray(self.loaded.get("offsets", []), dtype=float)
            if 0 <= int(trace_idx) < offsets_all.size:
                x = float(offsets_all[int(trace_idx)])
        rvred = 1.0 / float(self.params.vred)
        rvredf = 0.0
        if self.loaded is not None:
            header = self.loaded.get("header")
            vredf = float(getattr(header, "vredf", 0.0) or 0.0) if header is not None else 0.0
            if vredf > 0.0:
                rvredf = 1.0 / vredf
        # Fortran 对齐：时间窗与绘制基于 (rvred - rvredf)
        return -abs(x) * (rvred - rvredf)

    @staticmethod
    def _orientation_len_to_km(v: float) -> float:
        d = abs(float(v))
        return d / 1000.0 if d > 50.0 else d

    def _invalidate_orientation_preview_cache(self) -> None:
        self._orientation_preview_cache = None

    def _update_orientation_preview_badge(self) -> None:
        lbl = getattr(self, "lbl_orientation_preview", None)
        chk = getattr(self, "chk_orientation_preview_toggle", None)
        if lbl is None:
            return
        has_cache = isinstance(self._orientation_preview_cache, dict) and ("traces" in self._orientation_preview_cache)
        if chk is not None:
            chk.blockSignals(True)
            chk.setEnabled(bool(has_cache))
            chk.setChecked(bool(self._orientation_preview_enabled))
            chk.blockSignals(False)
        if bool(self._orientation_preview_enabled):
            lbl.setText("姿态校正预览: ON")
            lbl.setStyleSheet("color:#059669; font-weight:700;")
        else:
            lbl.setText("姿态校正预览: OFF")
            lbl.setStyleSheet("color:#64748b; font-weight:600;")

    def _on_orientation_preview_toggle(self, checked: bool) -> None:
        if bool(checked):
            cache_ok = isinstance(self._orientation_preview_cache, dict) and ("traces" in self._orientation_preview_cache)
            if (not cache_ok) or (not self._orientation_preview_solution):
                self._set_status_text("尚无可用姿态校正预览缓存，请先在结果窗口点击“应用到主图预览（全道）”。", hold_ms=2000)
                chk = getattr(self, "chk_orientation_preview_toggle", None)
                if chk is not None:
                    chk.blockSignals(True)
                    chk.setChecked(False)
                    chk.blockSignals(False)
                return
        self._set_orientation_main_preview(bool(checked), rebuild_if_needed=False, keep_cache=True)

    def _ask_pick_save_mode(self, title: str) -> bool:
        """Return True to save orientation-corrected picks, False for original picks."""
        has_preview = bool(self._orientation_preview_solution)
        if not has_preview:
            return False
        msg = QtWidgets.QMessageBox(self)
        msg.setIcon(QtWidgets.QMessageBox.Icon.Question)
        msg.setWindowTitle(title)
        msg.setText("保存拾取走时时，选择保存模式：")
        msg.setInformativeText("“校正后走时”仅应用姿态校正的全局 dt 偏移，不写入折合显示时移。")
        btn_raw = msg.addButton("保存原始走时（默认）", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        btn_corr = msg.addButton("保存校正后走时", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        msg.addButton("取消", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(btn_raw)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked == btn_corr:
            return True
        return False

    def _build_pick_snapshot(
        self,
        use_orientation_corrected: bool = False,
    ) -> Dict[int, Dict[int, float]]:
        if self.pick_manager is None:
            return {}
        base_raw = self.pick_manager.get_all_picks()
        base: Dict[int, Dict[int, float]] = {
            int(trace_idx): {int(word): float(tpk) for word, tpk in by_word.items()}
            for trace_idx, by_word in base_raw.items()
        }
        out = dict(base)
        dt_shift = float(self._orientation_preview_solution.get("time_shift_sec", 0.0))
        if not use_orientation_corrected:
            return out
        return {
            int(trace_idx): {int(word): float(tpk) + dt_shift for word, tpk in by_word.items() if float(tpk) > 0.0}
            for trace_idx, by_word in out.items()
        }

    def _register_floating_dialog(self, dlg: QtWidgets.QDialog) -> None:
        self._floating_dialogs.append(dlg)
        dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.destroyed.connect(lambda _=None, d=dlg: self._floating_dialogs.remove(d) if d in self._floating_dialogs else None)

    def _set_orientation_main_preview(
        self,
        enabled: bool,
        solution: Optional[Dict[str, float]] = None,
        rebuild_if_needed: bool = True,
        keep_cache: bool = True,
    ) -> None:
        old_sol = dict(self._orientation_preview_solution)
        if solution is not None:
            self._orientation_preview_solution = dict(solution)
            if old_sol != self._orientation_preview_solution:
                self._invalidate_orientation_preview_cache()
        self._orientation_preview_enabled = bool(enabled)
        if (not self._orientation_preview_enabled) and (not bool(keep_cache)):
            self._orientation_preview_solution = {}
            self._invalidate_orientation_preview_cache()
        if self._orientation_preview_enabled and self.loaded is not None and bool(rebuild_if_needed):
            cache_ok = isinstance(self._orientation_preview_cache, dict) and ("traces" in self._orientation_preview_cache)
            if not cache_ok:
                try:
                    QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
                    self._apply_orientation_solution_to_all_traces(
                        traces=self.loaded.get("traces", []),
                        offsets=np.asarray(self.loaded.get("offsets", []), dtype=float),
                        force_rebuild=True,
                    )
                finally:
                    QtWidgets.QApplication.restoreOverrideCursor()
        self._update_orientation_preview_badge()
        self.request_render(delay_ms=10)

    def _apply_orientation_solution_to_all_traces(
        self,
        traces: List[np.ndarray],
        offsets: np.ndarray,
        force_rebuild: bool = False,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        if (not self._orientation_preview_enabled) or self.loaded is None:
            return traces, offsets
        sol = dict(self._orientation_preview_solution or {})
        az = float(sol.get("azimuth_deg", 0.0))
        tilt = float(sol.get("tilt_deg", 0.0))
        dx = float(sol.get("dx", 0.0))
        dy = float(sol.get("dy", 0.0))
        cache_key = (
            id(self.loaded),
            len(traces),
            float(az),
            float(tilt),
            float(dx),
            float(dy),
        )
        if (not force_rebuild) and isinstance(self._orientation_preview_cache, dict) and self._orientation_preview_cache.get("key") == cache_key:
            tr_cached = self._orientation_preview_cache.get("traces", traces)
            off_cached = self._orientation_preview_cache.get("offsets", offsets)
            return list(tr_cached), np.asarray(off_cached, dtype=float)

        out_traces: List[np.ndarray] = list(traces)
        out_offsets = np.asarray(offsets, dtype=float).copy()
        headers = self.loaded.get("trace_headers", []) or []
        if len(headers) != len(traces):
            return traces, offsets
        use_utm, rec_role = self._infer_rec_role_for_orientation()

        comp_groups: Dict[Tuple[int, int], Dict[int, int]] = {}
        for i, th in enumerate(headers):
            shot = int(getattr(th, "ishoti", 0) or 0)
            rec = int(getattr(th, "ireci", 0) or 0)
            comp = int(getattr(th, "itypei", 0) or 0)
            if shot <= 0 or rec <= 0 or comp not in (1, 2, 3):
                continue
            comp_groups.setdefault((shot, rec), {})[comp] = int(i)

        azr = np.deg2rad(float(az))
        trr = np.deg2rad(float(tilt))
        for g in comp_groups.values():
            if not (1 in g and 2 in g and 3 in g):
                continue
            iz = int(g[1])
            ir = int(g[2])
            it = int(g[3])
            z = np.asarray(traces[iz], dtype=float)
            r = np.asarray(traces[ir], dtype=float)
            t = np.asarray(traces[it], dtype=float)
            rr = np.cos(azr) * r + np.sin(azr) * t
            tt = -np.sin(azr) * r + np.cos(azr) * t
            r2 = np.cos(trr) * rr + np.sin(trr) * z
            z2 = -np.sin(trr) * rr + np.cos(trr) * z
            out_traces[iz] = np.asarray(z2, dtype=float)
            out_traces[ir] = np.asarray(r2, dtype=float)
            out_traces[it] = np.asarray(tt, dtype=float)

        # 用“沿原炮检方向的位移投影”更新偏移距，避免近零偏移被几何幅值异常放大。
        disp = np.asarray([dx, dy], dtype=float)
        conv_samples: List[float] = []
        geom0_list: List[Optional[np.ndarray]] = [None] * len(headers)
        for i, th in enumerate(headers):
            try:
                if use_utm:
                    sx = float(getattr(th, "sxutm", 0.0) or 0.0)
                    sy = float(getattr(th, "syutm", 0.0) or 0.0)
                    rx = float(getattr(th, "rxutm", 0.0) or 0.0)
                    ry = float(getattr(th, "ryutm", 0.0) or 0.0)
                else:
                    sx = float(getattr(th, "slong", 0.0) or 0.0)
                    sy = float(getattr(th, "slat", 0.0) or 0.0)
                    rx = float(getattr(th, "rlong", 0.0) or 0.0)
                    ry = float(getattr(th, "rlat", 0.0) or 0.0)
                if rec_role == "sx":
                    src_xy0 = np.asarray([rx, ry], dtype=float)
                    rec_xy0 = np.asarray([sx, sy], dtype=float)
                else:
                    src_xy0 = np.asarray([sx, sy], dtype=float)
                    rec_xy0 = np.asarray([rx, ry], dtype=float)
                v0 = rec_xy0 - src_xy0
                geom0 = float(np.linalg.norm(v0))
                if np.isfinite(geom0) and geom0 > 1e-9:
                    geom0_list[i] = np.asarray(v0, dtype=float)
                    if i < offsets.size and np.isfinite(offsets[i]) and abs(float(offsets[i])) > 1e-9:
                        conv_samples.append(abs(float(offsets[i])) / geom0)
            except Exception:
                continue
        conv_km_per_coord = float(np.median(np.asarray(conv_samples, dtype=float))) if conv_samples else 0.0
        if not np.isfinite(conv_km_per_coord) or conv_km_per_coord <= 0.0:
            conv_km_per_coord = float(self._orientation_len_to_km(np.linalg.norm(disp))) / max(float(np.linalg.norm(disp)), 1e-9)

        for i, th in enumerate(headers):
            try:
                if i >= offsets.size or (not np.isfinite(offsets[i])):
                    continue
                old_off = float(offsets[i])
                v0 = geom0_list[i]
                if v0 is None:
                    out_offsets[int(i)] = old_off
                    continue
                n0 = float(np.linalg.norm(v0))
                if not np.isfinite(n0) or n0 <= 1e-9:
                    out_offsets[int(i)] = old_off
                    continue
                u0 = np.asarray(v0, dtype=float) / n0
                delta_km = float(np.dot(disp, u0)) * float(conv_km_per_coord)
                disp_km = float(self._orientation_len_to_km(np.linalg.norm(disp)))
                delta_km = float(np.clip(delta_km, -disp_km, disp_km))
                out_offsets[int(i)] = float(old_off + delta_km)
            except Exception:
                continue

        self._orientation_preview_cache = {
            "key": cache_key,
            "traces": out_traces,
            "offsets": out_offsets,
        }
        return out_traces, out_offsets

    def _build_reduced_pick_times(self, trace_indices: List[int], pick_word: int) -> Dict[int, float]:
        """构建带折合速度修正的拾取时间（不含静校正/临时对齐）。"""
        if self.pick_manager is None:
            return {}
        by_word = self.pick_manager.get_picks_by_word(int(pick_word))
        if not by_word:
            return {}
        offsets_all = np.asarray(self.loaded.get("offsets", []), dtype=float) if self.loaded is not None else np.array([], dtype=float)
        out: Dict[int, float] = {}
        for ig in trace_indices:
            trace_idx = int(ig)
            tpk = by_word.get(trace_idx)
            if tpk is None or float(tpk) <= 0.0:
                continue
            x = float(offsets_all[trace_idx]) if 0 <= trace_idx < offsets_all.size else 0.0
            out[trace_idx] = float(tpk) + self._compute_reduction_tshift(trace_idx, x)
        return out

    def _visible_mask(self, xcoords: np.ndarray) -> np.ndarray:
        if xcoords.size == 0:
            return np.zeros((0,), dtype=bool)
        x_range = self.plot.getViewBox().viewRange()[0]
        xmin, xmax = float(min(x_range)), float(max(x_range))
        # 加一点边缘缓冲，避免拖动时闪烁
        pad = max(1e-6, 0.02 * (xmax - xmin))
        return (xcoords >= xmin - pad) & (xcoords <= xmax + pad)

    def _compute_stack_auto_shifts(self, trace_indices: np.ndarray) -> Dict[int, float]:
        """为叠加显示计算“自动按拾取对齐”的时移（不改变主波形显示）。"""
        if self.pick_manager is None or trace_indices.size == 0:
            return {}
        apick = int(self.spin_apick.value())
        by_word = self.pick_manager.get_picks_by_word(apick)
        valid_times: List[float] = []
        valid_indices: List[int] = []
        for gidx in trace_indices:
            ig = int(gidx)
            tpk = by_word.get(ig)
            if tpk is None or float(tpk) <= 0:
                continue
            tt = float(tpk)
            if self.static_correction_enabled:
                tt += float(self.static_corrector.get_correction(ig))
            valid_indices.append(ig)
            valid_times.append(tt)
        if len(valid_times) < 2:
            return {}
        ref_time = float(np.median(np.asarray(valid_times, dtype=float)))
        return {ig: (ref_time - tt) for ig, tt in zip(valid_indices, valid_times)}

    def _ensure_curve_pool(self, n: int) -> None:
        while len(self._curve_items) < n:
            item = pg.PlotDataItem(
                pen=pg.mkPen(self._theme_color("wave_pen", "#0a0a0a"), width=1)
            )
            self.plot.addItem(item)
            self._curve_items.append(item)
        for i in range(n, len(self._curve_items)):
            self._curve_items[i].setData([], [])

    def _ensure_shade_item(self) -> None:
        if self._shade_item is None:
            self._shade_item = pg.PlotDataItem(
                pen=pg.mkPen(self._theme_color("shade_pen", "#30343b"), width=1),
                connect="pairs",
            )
            self.plot.addItem(self._shade_item)

    def _clear_shade_item(self) -> None:
        if self._shade_item is not None:
            self._shade_item.setData([], [])

    def _ensure_stack_item(self) -> None:
        if self._stack_item is None:
            self._stack_item = pg.PlotDataItem(
                pen=pg.mkPen(self._theme_color("stack_pen", "#1478dc"), width=1.5)
            )
            self.plot.addItem(self._stack_item)

    def _clear_stack_item(self) -> None:
        if self._stack_item is not None:
            self._stack_item.setData([], [])

    def _ensure_static_preview_item(self) -> None:
        if self._static_preview_item is None:
            self._static_preview_item = pg.PlotDataItem(
                pen=pg.mkPen(
                    self._theme_color("static_preview_pen", "#c828a0"),
                    width=1.8,
                    style=QtCore.Qt.PenStyle.DashLine,
                )
            )
            self.plot.addItem(self._static_preview_item)

    def _clear_static_preview_item(self) -> None:
        if self._static_preview_item is not None:
            self._static_preview_item.setData([], [])

    def _ensure_theoretical_item(self) -> None:
        if self._theoretical_item is None:
            self._theoretical_item = pg.PlotDataItem(
                pen=pg.mkPen(self._theme_color("theory_pen", "#f07814"), width=2)
            )
            self.plot.addItem(self._theoretical_item)

    def _clear_theoretical_item(self) -> None:
        if self._theoretical_item is not None:
            self._theoretical_item.setData([], [])

    def _ensure_txin_item(self) -> None:
        if self._txin_item is None:
            self._txin_item = pg.ScatterPlotItem(
                size=8.0,
                pen=pg.mkPen(self._theme_color("txin_pen", "#7c3aed"), width=1),
                pxMode=True,
            )
            self.plot.addItem(self._txin_item)

    def _clear_txin_item(self) -> None:
        if self._txin_item is not None:
            self._txin_item.setData([], [])

    def _ensure_txin_map_preview_item(self) -> None:
        if self._txin_map_preview_item is None:
            self._txin_map_preview_item = pg.ScatterPlotItem(
                size=10.0,
                pen=pg.mkPen(self._theme_color("txin_preview_pen", "#f97316"), width=1.5),
                brush=pg.mkBrush(0, 0, 0, 0),
                symbol="x",
                pxMode=True,
            )
            self.plot.addItem(self._txin_map_preview_item)

    def _clear_txin_map_preview_item(self) -> None:
        if self._txin_map_preview_item is not None:
            self._txin_map_preview_item.setData([], [])

    def _ensure_water_corr_item(self) -> None:
        if self._water_corr_item is None:
            self._water_corr_item = pg.PlotDataItem(
                pen=pg.mkPen(
                    self._theme_color("water_pen", "#1ea0d2"),
                    width=2,
                    style=QtCore.Qt.PenStyle.DashLine,
                )
            )
            self.plot.addItem(self._water_corr_item)

    def _clear_water_corr_item(self) -> None:
        if self._water_corr_item is not None:
            self._water_corr_item.setData([], [])

    def _ensure_wave_select_item(self) -> None:
        if self._wave_select_item is None:
            self._wave_select_item = pg.PlotDataItem(
                pen=pg.mkPen("#e11d48", width=1.8),
            )
            self.plot.addItem(self._wave_select_item)

    def _clear_wave_select_item(self) -> None:
        if self._wave_select_item is not None:
            self._wave_select_item.setData([], [])

    def _ensure_wave_select_marker_item(self) -> None:
        if self._wave_select_marker_item is None:
            self._wave_select_marker_item = pg.ScatterPlotItem(
                size=9.0,
                pen=pg.mkPen(self._theme_color("pick_active_edge", "#ffffff"), width=1.0),
                brush=pg.mkBrush(self._theme_color("pick_brush", "#ff7878")),
                pxMode=True,
            )
            self.plot.addItem(self._wave_select_marker_item)

    def _clear_wave_select_marker_item(self) -> None:
        if self._wave_select_marker_item is not None:
            self._wave_select_marker_item.setData([], [])

    def _ensure_waveop_stack_item(self) -> None:
        if self._waveop_stack_item is None:
            self._waveop_stack_item = pg.PlotDataItem(
                pen=pg.mkPen("#dc2626", width=2.0),
            )
            self.plot.addItem(self._waveop_stack_item)

    def _clear_waveop_stack_item(self) -> None:
        if self._waveop_stack_item is not None:
            self._waveop_stack_item.setData([], [])

    def _ensure_mute_polygon_item(self) -> None:
        if self._mute_polygon_item is None:
            self._mute_polygon_item = pg.PlotDataItem(
                pen=pg.mkPen("#0ea5e9", width=2.4, style=QtCore.Qt.PenStyle.DashLine),
            )
            self.plot.addItem(self._mute_polygon_item)

    def _ensure_mute_vertex_item(self) -> None:
        if self._mute_vertex_item is None:
            self._mute_vertex_item = pg.ScatterPlotItem(pxMode=True)
            self.plot.addItem(self._mute_vertex_item)

    def _clear_mute_polygon_item(self) -> None:
        if self._mute_polygon_item is not None:
            self._mute_polygon_item.setData([], [])
        if self._mute_vertex_item is not None:
            self._mute_vertex_item.setData([], [])

    def _refresh_mute_polygon_overlay(self, update_labels: bool = True) -> None:
        if not self._mute_polygon_points:
            self._clear_mute_polygon_item()
            return
        self._ensure_mute_polygon_item()
        self._ensure_mute_vertex_item()
        pts = np.asarray(self._mute_polygon_points, dtype=float)
        if pts.shape[0] == 1:
            self._mute_polygon_item.setData([pts[0, 0]], [pts[0, 1]])
        elif self._mute_enabled and pts.shape[0] >= 3:
            closed = np.vstack([pts, pts[0]])
            self._mute_polygon_item.setData(closed[:, 0], closed[:, 1])
        else:
            self._mute_polygon_item.setData(pts[:, 0], pts[:, 1])
        # 顶点把手：常态青色，当前选中顶点高亮橙色，便于定位/拖拽
        spots = []
        for i in range(pts.shape[0]):
            selected = (self._mute_selected_vertex_idx is not None and int(self._mute_selected_vertex_idx) == i)
            spots.append(
                {
                    "pos": (float(pts[i, 0]), float(pts[i, 1])),
                    "size": 11.0 if selected else 8.5,
                    "pen": pg.mkPen("#ffffff", width=1.5),
                    "brush": pg.mkBrush("#f59e0b" if selected else "#06b6d4"),
                    "symbol": "o",
                }
            )
        self._mute_vertex_item.setData(spots=spots)

    def _update_mute_status_button(self) -> None:
        if not hasattr(self, "btn_mute_status"):
            return
        if self._mute_edit_mode:
            txt = f"Mute: DRAW({len(self._mute_polygon_points)})"
            color = "#f59e0b"
        elif self._mute_enabled:
            txt = "Mute: ON(反选)" if self._mute_invert else "Mute: ON"
            color = "#22c55e"
        else:
            txt = "Mute: OFF"
            color = "#64748b"
        self.btn_mute_status.setText(txt)
        self.btn_mute_status.setStyleSheet(
            f"QPushButton{{border:1px solid {color}; color:{color}; font-weight:600;}}"
        )
        if hasattr(self, "btn_clear_mute"):
            self.btn_clear_mute.setEnabled(len(self._mute_polygon_points) > 0 or self._mute_enabled or self._mute_edit_mode)

    def _set_plot_pan_enabled(self, enabled: bool) -> None:
        vb = self.plot.getViewBox()
        vb.setMouseEnabled(x=bool(enabled), y=bool(enabled))

    def _clear_mute_all(self) -> None:
        """关闭Mute按钮：清空所有 mute（包含顶点）。"""
        had_effect = self._mute_enabled or len(self._mute_polygon_points) > 0 or self._mute_edit_mode
        self._mute_edit_mode = False
        self._mute_enabled = False
        self._mute_invert = False
        self._mute_polygon_points = []
        self._mute_drag_vertex_idx = None
        self._mute_selected_vertex_idx = None
        self._mute_drag_active = False
        self._sync_plot_pan_lock_state()
        self._clear_mute_polygon_item()
        self._update_mute_status_button()
        if had_effect:
            self.request_render(immediate=True)
        self.lbl_status.setText("Mute已关闭并清空（再次按M可重新绘制）")

    def _find_near_mute_vertex(self, x: float, y: float) -> Optional[int]:
        if not self._mute_polygon_points:
            return None
        xr, yr = self.plot.getViewBox().viewRange()
        x_span = max(1e-6, abs(float(xr[1]) - float(xr[0])))
        y_span = max(1e-6, abs(float(yr[1]) - float(yr[0])))
        tol_x = max(1e-6, 0.03 * x_span)
        tol_y = max(1e-6, 0.03 * y_span)
        best_idx: Optional[int] = None
        best_score = float("inf")
        for i, (px, py) in enumerate(self._mute_polygon_points):
            dx = (float(x) - float(px)) / tol_x
            dy = (float(y) - float(py)) / tol_y
            score = dx * dx + dy * dy
            if score < best_score:
                best_score = score
                best_idx = int(i)
        return best_idx if best_score <= 1.0 else None

    def _ensure_pick_item(self) -> None:
        if self._pick_item is None:
            self._pick_item = pg.ScatterPlotItem(
                size=float(self.spin_pick_size.value()),
                pen=pg.mkPen(self._theme_color("pick_pen", "#dc1e1e"), width=1),
                brush=pg.mkBrush(self._theme_color("pick_brush", "#ff7878")),
                pxMode=True,
            )
            self.plot.addItem(self._pick_item)

    def _render_picks(self, offsets_all: np.ndarray, allowed_trace_indices: Optional[np.ndarray] = None) -> int:
        if self.pick_manager is None:
            return 0
        picks_all = self._build_pick_snapshot(use_orientation_corrected=False)
        if not picks_all:
            if self._pick_item is not None:
                self._pick_item.setData([], [])
            return 0

        apick = int(self.spin_apick.value())
        x_range, y_range = self.plot.getViewBox().viewRange()
        xmin, xmax = min(x_range), max(x_range)
        ymin, ymax = min(y_range), max(y_range)
        allowed_set = None
        if allowed_trace_indices is not None and len(allowed_trace_indices) > 0:
            allowed_set = set(int(i) for i in np.asarray(allowed_trace_indices, dtype=int))
        headers = self.loaded.get("trace_headers", []) if self.loaded is not None else []
        group_union: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
        if headers:
            for trace_idx, by_word in picks_all.items():
                if trace_idx < 0 or trace_idx >= len(headers):
                    continue
                th = headers[trace_idx]
                key = (int(getattr(th, "ishoti", 0) or 0), int(getattr(th, "ireci", 0) or 0))
                if key[0] <= 0 or key[1] <= 0:
                    continue
                arr = group_union.setdefault(key, [])
                for pick_word, t_raw in by_word.items():
                    if float(t_raw) > 0.0:
                        arr.append((int(pick_word), float(t_raw)))
        spots = []
        trace_iter: List[int]
        if allowed_set is not None:
            trace_iter = sorted(allowed_set)
        else:
            trace_iter = sorted(int(k) for k in picks_all.keys())
        for trace_idx in trace_iter:
            if allowed_set is not None and int(trace_idx) not in allowed_set:
                continue
            if trace_idx < 0 or trace_idx >= len(offsets_all):
                continue
            x = float(offsets_all[trace_idx])
            if x < xmin or x > xmax:
                continue
            tshift = self._compute_display_tshift(int(trace_idx), x)
            own = picks_all.get(int(trace_idx), {})
            merged_vals: List[Tuple[int, float]] = []
            for pick_word, t_raw in own.items():
                merged_vals.append((int(pick_word), float(t_raw)))
            if headers and trace_idx < len(headers):
                th = headers[trace_idx]
                key = (int(getattr(th, "ishoti", 0) or 0), int(getattr(th, "ireci", 0) or 0))
                for pick_word, t_raw in group_union.get(key, []):
                    merged_vals.append((int(pick_word), float(t_raw)))
            seen = set()
            for pick_word, t_raw in merged_vals:
                sig = (int(pick_word), round(float(t_raw), 6))
                if sig in seen:
                    continue
                seen.add(sig)
                t = float(t_raw) + tshift
                if t < ymin or t > ymax:
                    continue
                # 不同拾取字用不同颜色；当前活动字视觉上更突出
                color = pg.intColor(int(pick_word), hues=48, values=1, alpha=220)
                is_active = int(pick_word) == apick
                spots.append(
                    {
                        "pos": (x, t),
                        "data": {"trace_idx": int(trace_idx), "pick_word": int(pick_word)},
                        "brush": pg.mkBrush(color),
                        "pen": pg.mkPen(
                            self._theme_color("pick_active_edge", "#ffffff") if is_active else self._theme_color("pick_edge", "#505050"),
                            width=1.2 if is_active else 0.8,
                        ),
                        "size": 12.0 if is_active else 9.0,
                    }
                )

        self._ensure_pick_item()
        self._pick_item.setSize(float(self.spin_pick_size.value()))
        if spots:
            self._pick_item.setData(spots=spots)
        else:
            self._pick_item.setData([], [])
        return len(spots)

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

    def _get_shared_pick(self, trace_idx: int, pick_word: int) -> Optional[float]:
        if self.pick_manager is None:
            return None
        group = self._trace_group_indices(int(trace_idx))
        # Prefer current trace pick, fallback to any component in same group.
        val = self.pick_manager.get_pick(int(trace_idx), int(pick_word))
        if val is not None:
            return float(val)
        for gi in group:
            v = self.pick_manager.get_pick(int(gi), int(pick_word))
            if v is not None:
                return float(v)
        return None

    def _set_shared_pick(self, trace_idx: int, pick_word: int, t_pick: float) -> bool:
        if self.pick_manager is None:
            return False
        group = self._trace_group_indices(int(trace_idx))
        # Keep only one canonical pick per (shot,receiver,pick_word), editable from any component.
        target = int(trace_idx)
        for gi in group:
            if self.pick_manager.get_pick(int(gi), int(pick_word)) is not None:
                target = int(gi)
                break
        for gi in group:
            if int(gi) != target:
                self.pick_manager.remove_pick(int(gi), int(pick_word))
        return bool(self.pick_manager.add_pick(int(target), float(t_pick), int(pick_word)))

    def _remove_shared_pick(self, trace_idx: int, pick_word: int) -> bool:
        if self.pick_manager is None:
            return False
        group = self._trace_group_indices(int(trace_idx))
        removed = False
        for gi in group:
            removed = bool(self.pick_manager.remove_pick(int(gi), int(pick_word))) or removed
        return removed

    def _toggle_mute_polygon_mode(self) -> None:
        """M 键：无多边形时进入编辑；有多边形时切换 mute 开关。"""
        if self.loaded is None:
            self.lbl_status.setText("Mute失败：请先加载数据")
            return
        if self._mute_edit_mode:
            self._mute_edit_mode = False
            self._mute_drag_vertex_idx = None
            self._sync_plot_pan_lock_state()
            self._update_mute_status_button()
            if len(self._mute_polygon_points) >= 3:
                self.lbl_status.setText("Mute编辑已退出")
            else:
                self.lbl_status.setText("Mute编辑已退出（顶点不足3，未应用）")
            return
        # 有已闭合多边形时，M 直接启用/取消 mute 效果（保留多边形）
        if len(self._mute_polygon_points) >= 3:
            self._mute_enabled = not self._mute_enabled
            self._mute_drag_vertex_idx = None
            self._mute_selected_vertex_idx = None
            self._sync_plot_pan_lock_state()
            self._update_mute_status_button()
            self.request_render(immediate=True)
            self.lbl_status.setText("Mute已启用（按M可取消）" if self._mute_enabled else "Mute已取消（按M可恢复）")
            return
        self._mute_edit_mode = True
        self._set_plot_pan_enabled(False)
        self._mute_drag_vertex_idx = None
        # 若尚未形成有效多边形，编辑态默认禁用 mute 效果
        if len(self._mute_polygon_points) < 3:
            self._mute_enabled = False
            self._mute_selected_vertex_idx = None
        self._update_mute_status_button()
        self.lbl_status.setText("Mute编辑已开启：左键选中/拖拽或加点，右键点顶点删除，右键空白闭合应用")

    def _toggle_mute_invert(self) -> None:
        """Shift+M：切换 mute 内外反选。"""
        if not self._mute_enabled:
            self.lbl_status.setText("反选失败：请先完成 Mute 闭合并应用")
            return
        self._mute_invert = not self._mute_invert
        self._update_mute_status_button()
        self.request_render(delay_ms=10)
        self.lbl_status.setText("Mute反选已开启（保留多边形外部）" if self._mute_invert else "Mute反选已关闭（保留多边形内部）")

    def _finalize_mute_polygon(self) -> None:
        if len(self._mute_polygon_points) < 3:
            self.lbl_status.setText("Mute闭合失败：至少需要3个顶点")
            return
        self._mute_edit_mode = False
        self._sync_plot_pan_lock_state()
        self._mute_enabled = True
        self._mute_drag_vertex_idx = None
        self._mute_selected_vertex_idx = None
        self._refresh_mute_polygon_overlay()
        self._update_mute_status_button()
        self.request_render(delay_ms=10)
        self.lbl_status.setText(
            f"Mute已应用：{len(self._mute_polygon_points)} 个顶点（外部不绘制，保留区按当前参数滤波/增益）"
        )

    def _delete_selected_mute_vertex(self) -> None:
        """Del：删除当前选中（或光标附近）顶点，仅在 Mute 绘制编辑态生效。"""
        if not self._mute_edit_mode or not self._mute_polygon_points:
            return
        idx = self._mute_selected_vertex_idx
        if idx is None and self.mouse_x is not None and self.mouse_y is not None:
            idx = self._find_near_mute_vertex(float(self.mouse_x), float(self.mouse_y))
        if idx is None:
            self.lbl_status.setText("Mute删除失败：请先点选一个顶点")
            return
        i = int(idx)
        if i < 0 or i >= len(self._mute_polygon_points):
            return
        self._mute_polygon_points.pop(i)
        self._mute_drag_vertex_idx = None
        if len(self._mute_polygon_points) < 3:
            self._mute_enabled = False
        if not self._mute_polygon_points:
            self._mute_selected_vertex_idx = None
        else:
            self._mute_selected_vertex_idx = min(i, len(self._mute_polygon_points) - 1)
        self._refresh_mute_polygon_overlay()
        self._update_mute_status_button()
        # 右键删点后立即重绘波形，保证 mute 效果及时更新
        self.request_render(immediate=True)
        self.lbl_status.setText(f"Mute编辑：已删除顶点 #{i + 1}，剩余 {len(self._mute_polygon_points)} 个")

    def _build_mute_inside_mask(
        self,
        x_trace: float,
        t_display: np.ndarray,
        polygon: np.ndarray,
    ) -> np.ndarray:
        """计算纵向采样点是否位于 mute 多边形内部（显示坐标系）。"""
        n = int(polygon.shape[0])
        if n < 3 or t_display.size == 0:
            return np.zeros_like(t_display, dtype=bool)
        y_hits: List[float] = []
        for i in range(n):
            x1, y1 = float(polygon[i, 0]), float(polygon[i, 1])
            x2, y2 = float(polygon[(i + 1) % n, 0]), float(polygon[(i + 1) % n, 1])
            if abs(x2 - x1) < 1e-12:
                if abs(x_trace - x1) < 1e-9:
                    y_hits.extend([y1, y2])
                continue
            xmin, xmax = (x1, x2) if x1 <= x2 else (x2, x1)
            if x_trace < xmin or x_trace >= xmax:
                continue
            ratio = (x_trace - x1) / (x2 - x1)
            if 0.0 <= ratio <= 1.0:
                y_hits.append(y1 + ratio * (y2 - y1))
        if len(y_hits) < 2:
            return np.zeros_like(t_display, dtype=bool)
        y_hits.sort()
        mask = np.zeros_like(t_display, dtype=bool)
        for k in range(0, len(y_hits) - 1, 2):
            y0 = float(y_hits[k])
            y1 = float(y_hits[k + 1])
            lo, hi = (y0, y1) if y0 <= y1 else (y1, y0)
            mask |= (t_display >= lo) & (t_display <= hi)
        return mask

    def _apply_mute_to_raw_traces(
        self,
        raw_traces: List[np.ndarray],
        trace_indices: np.ndarray,
        render_offsets: np.ndarray,
        times: np.ndarray,
    ) -> List[np.ndarray]:
        """将多边形外部样点临时置零（仅渲染链路）。"""
        if not self._mute_enabled or len(self._mute_polygon_points) < 3:
            return raw_traces
        polygon = np.asarray(self._mute_polygon_points, dtype=float)
        out: List[np.ndarray] = []
        for li, tr in enumerate(raw_traces):
            tr_arr = np.asarray(tr, dtype=float).copy()
            ns = int(min(tr_arr.size, times.size))
            if ns <= 0:
                out.append(tr_arr)
                continue
            trace_idx = int(trace_indices[li])
            x_trace = float(render_offsets[li])
            tshift = float(self._compute_display_tshift(trace_idx, x_trace))
            t_display = np.asarray(times[:ns], dtype=float) + tshift
            inside = self._build_mute_inside_mask(x_trace, t_display, polygon)
            keep_mask = (~inside) if self._mute_invert else inside
            tr_arr[:ns][~keep_mask] = 0.0
            out.append(tr_arr)
        return out

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

    def _sync_picks_into_trace_headers(self, use_orientation_corrected: bool = False) -> bool:
        """将 pick_manager 的拾取写回当前 trace_headers.picks。"""
        if self.loaded is None or self.pick_manager is None:
            return False
        trace_headers = self.loaded.get("trace_headers", [])
        header = self.loaded.get("header")
        if not trace_headers or header is None:
            return False
        npick = int(getattr(header, "npick", 0) or 0)
        if npick <= 0:
            return False
        all_picks = self._build_pick_snapshot(use_orientation_corrected=use_orientation_corrected)
        for i, th in enumerate(trace_headers):
            picks_arr = [0.0] * npick
            by_word = all_picks.get(int(i), {})
            for pick_word, tpk in by_word.items():
                pw = int(pick_word)
                if 1 <= pw <= npick and float(tpk) > 0.0:
                    picks_arr[pw - 1] = float(tpk)
            th.picks = picks_arr
        return True

    def _save_z_with_picks(self) -> None:
        if self.loaded is None or self._dfile is None:
            self.lbl_status.setText("保存.z失败：请先加载 .z 数据")
            return
        use_corr = self._ask_pick_save_mode("保存 .z")
        if not self._sync_picks_into_trace_headers(use_orientation_corrected=bool(use_corr)):
            self.lbl_status.setText("保存.z失败：当前数据无可写入的道头/拾取信息")
            return
        msg = QtWidgets.QMessageBox(self)
        msg.setIcon(QtWidgets.QMessageBox.Icon.Question)
        msg.setWindowTitle("保存 .z")
        msg.setText("请选择保存方式：")
        btn_overwrite = msg.addButton("覆盖当前 .z（默认）", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        btn_save_as = msg.addButton("另存为新 .z（写入处理后波形）", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        btn_cancel = msg.addButton("取消", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(btn_overwrite)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked == btn_cancel or clicked is None:
            return

        out = str(self._dfile)
        use_processed_traces = False
        if clicked == btn_save_as:
            default_name = Path(self._dfile).with_name(f"{Path(self._dfile).stem}_picked.z")
            out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "另存为 .z（含拾取）", str(default_name), "Z files (*.z)", options=self._file_dialog_options()
            )
            if not out_path:
                return
            if Path(out_path).suffix.lower() != ".z":
                out_path = f"{out_path}.z"
            out = out_path
            use_processed_traces = True

        original_loader_traces = self.loader.traces
        original_loader_header = self.loader.header
        original_loader_trace_headers = self.loader.trace_headers
        try:
            if use_processed_traces:
                traces = self.loaded.get("traces", [])
                times = np.asarray(self.loaded.get("times", []), dtype=float)
                offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
                if len(traces) == 0 or times.size < 2 or offsets.size == 0:
                    self.lbl_status.setText("另存为.z失败：处理后波形不可用")
                    return
                proc_params = ZPlotParameters()
                proc_params.amp = float(self.spin_amp.value())
                proc_params.iscale = int(self.combo_iscale.currentIndex())
                proc_params.rcor = float(self.spin_rcor.value())
                proc_params.sf = float(self.spin_sf.value())
                proc_params.tvg = float(self.spin_tvg.value())
                proc_params.pvg = float(self.spin_pvg.value())
                proc_params.clip = float(self.spin_clip.value())
                proc_params.ibndps = 1 if self.chk_filter.isChecked() else 0
                proc_params.freqlo = float(self.spin_freqlo.value())
                proc_params.freqhi = float(self.spin_freqhi.value())
                proc_params.npoles = int(self.spin_npoles.value())
                proc_params.izerop = 1 if self.chk_zerop.isChecked() else 0
                proc_params.vred = float(self.spin_vred.value())
                if not bool(self.chk_gain.isChecked()):
                    proc_params.iscale = 0
                    proc_params.amp = 1.0
                    proc_params.rcor = 0.0
                    proc_params.sf = 0.0
                    proc_params.tvg = 1.0
                    proc_params.pvg = 1.0
                    proc_params.clip = 0.0
                trace_headers = self.loaded.get("trace_headers", [])
                gains = np.ones(len(traces), dtype=float)
                for i, th in enumerate(trace_headers):
                    if i < gains.size:
                        gains[i] = float(max(1, int(getattr(th, "igaini", 1) or 1)))
                sr = 1.0 / float(times[1] - times[0])
                processed_traces = self.processor.process_traces(
                    traces=[np.asarray(t) for t in traces],
                    times=times,
                    offsets=offsets,
                    params=proc_params,
                    gains=gains,
                    sampling_rate=sr,
                    realtime_interaction=False,
                )
                self.loader.traces = processed_traces

            # 统一使用当前 loaded 的头信息与道头（含已同步 picks）
            self.loader.header = self.loaded.get("header")
            self.loader.trace_headers = self.loaded.get("trace_headers", [])
            ok = self.loader.save_z_format(out, hfile=None, write_picks_to_data=True)
        except Exception as exc:
            self.lbl_status.setText(f"保存.z失败：{exc}")
            return
        finally:
            self.loader.traces = original_loader_traces
            self.loader.header = original_loader_header
            self.loader.trace_headers = original_loader_trace_headers

        mode_text = "校正后走时" if use_corr else "原始走时"
        if clicked == btn_overwrite:
            self.lbl_status.setText(f".z 已覆盖保存（{mode_text}）：{out}" if ok else "保存.z失败")
        else:
            self.lbl_status.setText(f".z 另存成功（处理后波形+{mode_text}）：{out}" if ok else "保存.z失败")

    def _save_picks(self) -> None:
        if self.pick_manager is None:
            return
        use_corr = self._ask_pick_save_mode("保存拾取")
        out, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存拾取", "", "zplot.out (*.out *.txt);;All files (*)", options=self._file_dialog_options()
        )
        if not out:
            return
        backup = self._snapshot_pick_state()
        try:
            if use_corr:
                self.pick_manager.clear_picks()
                corr = self._build_pick_snapshot(use_orientation_corrected=True)
                for trace_idx, by_word in corr.items():
                    for pick_word, tpk in by_word.items():
                        self.pick_manager.add_pick(int(trace_idx), float(tpk), int(pick_word))
            ok = self.pick_manager.save_picks(out, format="zplot")
        finally:
            self._restore_pick_state(backup)
        mode_text = "校正后走时" if use_corr else "原始走时"
        self.lbl_status.setText(f"拾取已保存（{mode_text}）" if ok else "拾取保存失败")

    def _save_picks_to_hdr(self) -> bool:
        if self.pick_manager is None or self.loaded is None:
            return False
        use_corr = self._ask_pick_save_mode("写入 HDR")
        self._sync_picks_into_trace_headers(use_orientation_corrected=bool(use_corr))
        trace_headers = self.loaded.get("trace_headers", [])
        if not trace_headers:
            self.lbl_status.setText("写入HDR失败：无道头信息")
            return False
        hfile = self._hfile
        if not hfile:
            hfile, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "选择HDR输出文件", "", "Header files (*.hdr)", options=self._file_dialog_options()
            )
            if not hfile:
                return False
            if Path(hfile).suffix.lower() != ".hdr":
                hfile = f"{hfile}.hdr"
            self._hfile = hfile
        ok = self.pick_manager.save_to_header_file(hfile, trace_headers)
        mode_text = "校正后走时" if use_corr else "原始走时"
        self.lbl_status.setText(f"拾取已写入HDR（{mode_text}）" if ok else "写入HDR失败")
        return bool(ok)

    def _export_txin(self) -> None:
        if self.loaded is None:
            return
        if not self._save_picks_to_hdr():
            return
        if not self._hfile:
            self.lbl_status.setText("导出tx.in失败：未设置HDR文件")
            return

        header = self.loaded.get("header")
        npick = int(getattr(header, "npick", 0) or 0)
        if npick <= 0:
            self.lbl_status.setText("导出tx.in失败：无有效npick")
            return

        out, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存 tx.in", "", "tx.in (*.in)", options=self._file_dialog_options()
        )
        if not out:
            return
        if Path(out).suffix.lower() != ".in":
            out = f"{out}.in"

        trace_headers = self.loaded.get("trace_headers", [])
        shots = sorted(set(int(getattr(th, "ishoti", 0) or 0) for th in trace_headers))
        cfg = self._prompt_tx_config(shots=shots, npick=npick)
        if cfg is None:
            self.lbl_status.setText("tx.in导出已取消")
            return
        ok, total = convert_hdr_to_tx(self._hfile, out, cfg, npick)
        self.lbl_status.setText(f"tx.in导出成功：{total} picks" if ok else "tx.in导出失败")

    def _load_txin_overlay(self) -> None:
        if self.loaded is None:
            QtWidgets.QMessageBox.information(self, "提示", "请先加载 .z 数据，再读取 tx.in。")
            self.lbl_status.setText("读取 tx.in 失败：尚未加载数据")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 tx.in 文件", "", "tx.in files (*.in *.tx);;All files (*)", options=self._file_dialog_options()
        )
        if not path:
            self.lbl_status.setText("读取 tx.in 已取消")
            return
        try:
            data = self._parse_txin_overlay_file(path)
            offsets = np.asarray(data.get("offsets", []), dtype=float)
            times = np.asarray(data.get("times", []), dtype=float)
            pick_words = np.asarray(data.get("pick_words", []), dtype=int)
            if offsets.size == 0 or times.size == 0:
                self.lbl_status.setText("tx.in 导入失败：无有效走时点")
                return
            self.txin_overlay_data = data
            self.show_txin_overlay = True
            self.txin_map_preview_data = None
            self._clear_txin_map_preview_item()
            self.request_render(immediate=True)
            n_words = int(np.unique(pick_words).size) if pick_words.size > 0 else 0
            self.lbl_status.setText(f"tx.in 已叠加：{offsets.size} 点，{n_words} 个拾取字（分色）")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "tx.in 导入失败", str(exc))
            self.lbl_status.setText(f"tx.in 导入失败: {exc}")

    def _clear_txin_overlay(self) -> None:
        self.txin_overlay_data = None
        self.show_txin_overlay = False
        self.txin_map_preview_data = None
        self._clear_txin_item()
        self._clear_txin_map_preview_item()
        self.request_render(delay_ms=10)
        self.lbl_status.setText("已清除 tx.in 走时叠加")

    def _build_txin_map_candidates(self) -> Tuple[Dict[Tuple[int, int], Tuple[float, float]], Dict[str, object]]:
        if self.loaded is None or self.txin_overlay_data is None:
            return {}, {"error": "请先加载数据并读取 tx.in"}
        target_idx = self._extract_indices()
        if target_idx.size == 0:
            return {}, {"error": "当前剖面无可用道"}
        offsets_all = np.asarray(self.loaded.get("offsets", []), dtype=float)
        if offsets_all.size == 0:
            return {}, {"error": "缺少偏移距信息"}
        tx_off = np.asarray(self.txin_overlay_data.get("offsets", []), dtype=float)
        tx_t = np.asarray(self.txin_overlay_data.get("times", []), dtype=float)
        tx_pw = np.asarray(self.txin_overlay_data.get("pick_words", []), dtype=int)
        if tx_off.size == 0 or tx_t.size != tx_off.size or tx_pw.size != tx_off.size:
            return {}, {"error": "tx.in 叠加数据无效"}

        target_offsets = offsets_all[target_idx]
        valid_pw_max = int(self.spin_apick.maximum())
        apick_only = bool(self.chk_map_txin_apick_only.isChecked())
        apick = int(self.spin_apick.value())
        view_only = bool(self.chk_map_txin_view_only.isChecked())
        tol_ratio = max(0.10, float(self.spin_map_txin_tol.value()) / 100.0)
        x_view = self.plot.getViewBox().viewRange()[0]
        y_view = self.plot.getViewBox().viewRange()[1]
        xmin_v, xmax_v = float(min(x_view)), float(max(x_view))
        ymin_v, ymax_v = float(min(y_view)), float(max(y_view))

        uniq_off = np.unique(np.sort(target_offsets))
        if uniq_off.size >= 2:
            spacing = float(np.median(np.diff(uniq_off)))
            tol = max(0.05, tol_ratio * abs(spacing))
        else:
            tol = float("inf")

        candidate: Dict[Tuple[int, int], Tuple[float, float]] = {}
        ignored_far = 0
        ignored_bad_pw = 0
        ignored_out_view = 0
        for i in range(int(tx_off.size)):
            pw = int(tx_pw[i])
            if pw < 1 or pw > valid_pw_max:
                ignored_bad_pw += 1
                continue
            if apick_only and pw != apick:
                continue
            x = float(tx_off[i])
            t = float(tx_t[i])
            if not (math.isfinite(x) and math.isfinite(t)):
                continue
            if view_only:
                t_disp = t + self._compute_reduction_tshift(-1, x)
                if not (xmin_v <= x <= xmax_v and ymin_v <= t_disp <= ymax_v):
                    ignored_out_view += 1
                    continue
            nearest_pos = int(np.argmin(np.abs(target_offsets - x)))
            trace_idx = int(target_idx[nearest_pos])
            dist = float(abs(target_offsets[nearest_pos] - x))
            if dist > tol:
                ignored_far += 1
                continue
            key = (trace_idx, pw)
            prev = candidate.get(key)
            if prev is None or dist < prev[0]:
                candidate[key] = (dist, t)

        return candidate, {
            "ignored_far": ignored_far,
            "ignored_bad_pw": ignored_bad_pw,
            "ignored_out_view": ignored_out_view,
            "apick_only": apick_only,
            "apick": apick,
            "view_only": view_only,
            "tol_percent": float(self.spin_map_txin_tol.value()),
        }

    def _preview_txin_mapping(self) -> None:
        if self.loaded is None or self.pick_manager is None:
            self.lbl_status.setText("请先加载数据")
            return
        if self.txin_overlay_data is None:
            self.lbl_status.setText("请先读取 tx.in")
            return
        candidate, meta = self._build_txin_map_candidates()
        err = meta.get("error")
        if err:
            self.lbl_status.setText(f"预览失败：{err}")
            return
        if not candidate:
            self.txin_map_preview_data = None
            self._clear_txin_map_preview_item()
            self.request_render(delay_ms=10)
            self.lbl_status.setText("映射预览：无可新增点")
            return

        keep_existing = 0
        preview_points: List[Tuple[int, int, float]] = []
        for (trace_idx, pw), (_, tval) in candidate.items():
            exist = self.pick_manager.get_pick(trace_idx, pw)
            if exist is not None and float(exist) > 0.0:
                keep_existing += 1
                continue
            preview_points.append((int(trace_idx), int(pw), float(tval)))

        if not preview_points:
            self.txin_map_preview_data = None
            self._clear_txin_map_preview_item()
            self.request_render(delay_ms=10)
            self.lbl_status.setText(f"映射预览：新增 0，已有拾取将保留 {keep_existing}")
            return

        arr = np.asarray(preview_points, dtype=float)
        self.txin_map_preview_data = {
            "trace_indices": arr[:, 0].astype(int),
            "pick_words": arr[:, 1].astype(int),
            "times": arr[:, 2].astype(float),
        }
        self.request_render(immediate=True)
        mode_text = f"仅apick={int(meta['apick'])}" if bool(meta["apick_only"]) else "全部拾取字"
        view_text = "仅视窗" if bool(meta["view_only"]) else "全窗口"
        self.lbl_status.setText(
            f"映射预览：将新增 {int(arr.shape[0])}，已有保留 {keep_existing}，"
            f"超距忽略 {int(meta['ignored_far'])}，视窗外忽略 {int(meta['ignored_out_view'])}，"
            f"拾取字越界忽略 {int(meta['ignored_bad_pw'])}（{mode_text}, {view_text}, 容差={float(meta['tol_percent']):.1f}%）"
        )

    def _map_txin_to_picks(self) -> None:
        """
        将 tx.in 叠加点映射为当前剖面的拾取：
        - 以当前过滤结果（irec/itype/nskip/x窗/移除道）作为目标剖面
        - 以偏移距最近道进行匹配
        - 若该道该拾取字已有拾取，则保留不覆盖
        """
        if self.loaded is None or self.pick_manager is None:
            self.lbl_status.setText("请先加载数据")
            return
        if self.txin_overlay_data is None:
            self.lbl_status.setText("请先读取 tx.in")
            return

        times_all = np.asarray(self.loaded.get("times", []), dtype=float)
        if times_all.size == 0:
            self.lbl_status.setText("映射失败：缺少时间轴")
            return
        candidate, meta = self._build_txin_map_candidates()
        err = meta.get("error")
        if err:
            self.lbl_status.setText(f"映射失败：{err}")
            return

        if not candidate:
            self.lbl_status.setText("映射完成：无可映射点（可能偏移距不匹配）")
            return

        added = 0
        kept_existing = 0
        tmin = float(np.min(times_all))
        tmax = float(np.max(times_all))
        to_add: List[Tuple[int, int, float]] = []
        for (trace_idx, pw), (_, tval) in candidate.items():
            exist = self.pick_manager.get_pick(trace_idx, pw)
            if exist is not None and float(exist) > 0.0:
                kept_existing += 1
                continue
            to_add.append((int(trace_idx), int(pw), float(np.clip(tval, tmin, tmax))))

        if to_add:
            self._push_pick_undo("tx.in 映射拾取")
            for trace_idx, pw, tval in to_add:
                self.pick_manager.add_pick(trace_idx, tval, pw)
            added = len(to_add)

        if added > 0:
            self.request_render(delay_ms=10)
        self.txin_map_preview_data = None
        self._clear_txin_map_preview_item()
        mode_text = f"仅apick={int(meta['apick'])}" if bool(meta["apick_only"]) else "全部拾取字"
        view_text = "仅视窗" if bool(meta["view_only"]) else "全窗口"
        self.lbl_status.setText(
            f"tx.in 映射完成：新增 {added}，保留已有 {kept_existing}，"
            f"超距忽略 {int(meta['ignored_far'])}，视窗外忽略 {int(meta['ignored_out_view'])}，拾取字越界忽略 {int(meta['ignored_bad_pw'])}"
            f"（{mode_text}, {view_text}, 容差={float(meta['tol_percent']):.1f}%）"
        )

    def _parse_txin_overlay_file(self, txin_path: str) -> Dict[str, np.ndarray]:
        """
        解析 tx.in 并转换为可叠加曲线：
        - 输入通常为「模型距离 x + 真实走时 t」
        - 通过分段标记行（第4列=0）中的 xmod，换算偏移距：offset = x - xmod
        - 显示时再统一做折合时间修正（在渲染阶段）
        """
        points_all: List[Tuple[float, float, int]] = []
        current_xmod: Optional[float] = None

        with open(txin_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#") or line.startswith("!"):
                    continue
                fields = line.split()
                if len(fields) < 4:
                    continue
                try:
                    col1 = float(fields[0])
                    col2 = float(fields[1])
                    _col3 = float(fields[2])
                    ipick = int(round(float(fields[3])))
                except Exception:
                    continue

                # 结束标记
                if ipick == -1:
                    break
                # 分段标记：xmod, (+/-1), 0, 0
                if ipick == 0:
                    current_xmod = float(col1)
                    continue
                # 常规拾取行
                if ipick > 0:
                    if not (math.isfinite(col1) and math.isfinite(col2)):
                        continue
                    if current_xmod is None:
                        # 无分段标记时退化：假定 x 已是偏移距
                        off = float(col1)
                    else:
                        off = float(col1 - current_xmod)
                    points_all.append((off, float(col2), int(ipick)))

        if not points_all:
            raise ValueError("未读取到可用拾取点（请检查 tx.in 文件内容）")

        arr = np.asarray(points_all, dtype=float)
        order = np.argsort(arr[:, 0])
        arr = arr[order]
        return {
            "offsets": arr[:, 0].astype(float),
            "times": arr[:, 1].astype(float),
            "pick_words": arr[:, 2].astype(int),
        }

    def _prompt_tx_config(self, shots: List[int], npick: int) -> Optional[Z2TxConfig]:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("tx.in 转换配置")
        dialog.resize(760, 420)
        layout = QtWidgets.QVBoxLayout(dialog)

        table = QtWidgets.QTableWidget(len(shots), 4, dialog)
        table.setHorizontalHeaderLabels(["OBS", "xmod", "tshift", "xshift"])
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setStretchLastSection(True)
        for row, shot in enumerate(shots):
            obs_item = QtWidgets.QTableWidgetItem(str(int(shot)))
            obs_item.setFlags(obs_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, obs_item)
            table.setItem(row, 1, QtWidgets.QTableWidgetItem("0.0"))
            table.setItem(row, 2, QtWidgets.QTableWidgetItem("0.0"))
            table.setItem(row, 3, QtWidgets.QTableWidgetItem("0.0"))
        layout.addWidget(table)

        form = QtWidgets.QFormLayout()
        edit_picku = QtWidgets.QLineEdit("0.05", dialog)
        edit_picku.setPlaceholderText("单值或逗号分隔列表，如 0.05 或 0.03,0.04,0.05")
        combo_mode = QtWidgets.QComboBox(dialog)
        combo_mode.addItems(["走时(iamp=0)", "振幅(iamp=1)"])
        form.addRow("picku", edit_picku)
        form.addRow("模式", combo_mode)
        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return None

        picku_text = edit_picku.text().strip()
        try:
            if "," in picku_text:
                picku_list = [float(x.strip()) for x in picku_text.split(",") if x.strip()]
                if not picku_list:
                    raise ValueError("picku 为空")
            else:
                picku_val = float(picku_text)
                picku_list = [picku_val] * max(1, int(npick))
        except Exception:
            QtWidgets.QMessageBox.warning(dialog, "参数错误", "picku 格式错误，请输入单值或逗号分隔浮点数。")
            return None

        obs_configs: Dict[int, Dict] = {}
        for row, shot in enumerate(shots):
            try:
                xmod_item = table.item(row, 1)
                tshift_item = table.item(row, 2)
                xshift_item = table.item(row, 3)
                xmod = float(xmod_item.text() if xmod_item is not None else 0.0)
                tshift = float(tshift_item.text() if tshift_item is not None else 0.0)
                xshift = float(xshift_item.text() if xshift_item is not None else 0.0)
            except Exception:
                QtWidgets.QMessageBox.warning(dialog, "参数错误", f"OBS {shot} 的数值格式错误。")
                return None
            obs_configs[int(shot)] = {
                "xmod": xmod,
                "tshift": tshift,
                "xshift": xshift,
                "picku": list(picku_list),
            }

        iamp = 1 if combo_mode.currentIndex() == 1 else 0
        return Z2TxConfig(obs_configs=obs_configs, iamp=iamp)

    def _clear_picks(self) -> None:
        if self.pick_manager is None:
            return
        pick_word = int(self.spin_apick.value()) if hasattr(self, "spin_apick") else 1
        by_word = self.pick_manager.get_picks_by_word(pick_word)
        if not by_word:
            self.lbl_status.setText(f"当前拾取字(apick={pick_word})无可清空拾取")
            return
        self._push_pick_undo(f"清空拾取字{pick_word}")
        removed = 0
        for trace_idx in list(by_word.keys()):
            if self.pick_manager.remove_pick(int(trace_idx), pick_word):
                removed += 1
        self.request_render(immediate=True)
        self.lbl_status.setText(f"已清空当前拾取字(apick={pick_word})：{removed} 道")

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

    def _apply_shift_hover_pick(self) -> None:
        """拾取模式下，按住 Shift 时在鼠标经过道上连续拾取（每道一次）。"""
        if not self._shift_pressed or not self._shift_hover_pick_active:
            return
        if self.loaded is None or self.pick_manager is None:
            return
        if not self.chk_pick_mode.isChecked():
            return
        if self.mouse_x is None or self.mouse_y is None:
            return
        if self._last_render_trace_indices.size == 0 or self._last_render_offsets.size == 0:
            return
        x = float(self.mouse_x)
        y = float(self.mouse_y)
        nearest_i = int(np.argmin(np.abs(self._last_render_offsets - x)))
        trace_idx = int(self._last_render_trace_indices[nearest_i])
        if trace_idx in self._shift_hover_picked_traces:
            return
        x_trace = float(self._last_render_offsets[nearest_i])
        y_pick = float(y) - self._compute_display_tshift(trace_idx, x_trace)
        apick = int(self.spin_apick.value())
        old_pick = self._get_shared_pick(trace_idx, apick)
        self._shift_hover_picked_traces.add(trace_idx)
        if old_pick is not None and abs(float(old_pick) - y_pick) < 1e-9:
            return
        if not self._shift_hover_pick_undo_pushed:
            self._push_pick_undo("Shift悬停拾取")
            self._shift_hover_pick_undo_pushed = True
        self._set_shared_pick(trace_idx, apick, y_pick)
        self._shift_hover_pick_updated_count += 1
        self.request_render(delay_ms=10)

    def _add_waveform_selection_at_cursor(self) -> None:
        """V 键：以鼠标最近道为中心，记录选波窗口（前0.3s、后0.7s）。"""
        if self.loaded is None:
            self.lbl_status.setText("V选波失败：请先加载数据")
            return
        if self.mouse_x is None or self.mouse_y is None:
            self.lbl_status.setText("V选波失败：请先将鼠标移动到剖面区域")
            return
        if self._last_render_trace_indices.size == 0 or self._last_render_offsets.size == 0:
            self.lbl_status.setText("V选波失败：当前无可选道")
            return
        x_ref = float(self.mouse_x)
        y_ref = float(self.mouse_y)
        nearest_i = int(np.argmin(np.abs(self._last_render_offsets - x_ref)))
        trace_idx = int(self._last_render_trace_indices[nearest_i])
        x_trace = float(self._last_render_offsets[nearest_i])
        tshift = float(self._compute_display_tshift(trace_idx, x_trace))
        # 优先使用“当前拾取字”在该道的拾取点作为 V 段中心；
        # 若该道该拾取字无拾取，再退回鼠标位置。
        y_center = y_ref
        center_from_pick = False
        if self.pick_manager is not None:
            apick = int(self.spin_apick.value())
            picked_t = self._get_shared_pick(trace_idx, apick)
            if picked_t is not None and float(picked_t) > 0.0:
                y_center = float(picked_t) + tshift
                center_from_pick = True
        t_true = float(y_center - tshift)
        new_sel = {
            "trace_idx": float(trace_idx),
            "offset": x_trace,
            "t_display": y_center,
            "t_true": t_true,
            "pick_word": float(int(self.spin_apick.value())),
        }
        # 每道最多保留一个 V 段：同一道再次按 V 时更新该道现有记录
        replaced = False
        key_new = (int(trace_idx), int(self.spin_apick.value()))
        for i, old in enumerate(self.waveform_selections):
            if (
                int(old.get("trace_idx", -1)) == int(trace_idx)
                and int(old.get("pick_word", int(self.spin_apick.value()))) == int(self.spin_apick.value())
            ):
                self.waveform_selections[i] = new_sel
                replaced = True
                break
        if not replaced:
            self.waveform_selections.append(new_sel)
        self._waveop_corrected_ttrue[key_new] = float(t_true)
        self._refresh_waveop_selection_list()
        self.request_render(delay_ms=10)
        src = "拾取点" if center_from_pick else "鼠标"
        action = "更新" if replaced else "添加"
        self.lbl_status.setText(
            f"V选波已{action}：道 {trace_idx}，中心={y_center:.3f}s({src})，窗口=[{y_center - 0.3:.3f}, {y_center + 0.7:.3f}]s {self._waveop_apick_status_suffix()}"
        )

    def _current_apick_waveform_selections(self) -> List[Dict[str, float]]:
        apick = int(self.spin_apick.value())
        out: List[Dict[str, float]] = []
        for sel in self.waveform_selections:
            pw = int(sel.get("pick_word", apick))
            if pw == apick:
                out.append(sel)
        return out

    def _remove_last_waveform_selection(self) -> None:
        if not self.waveform_selections:
            self.lbl_status.setText("Shift+V：当前没有可删除的 V 段")
            return
        apick = int(self.spin_apick.value())
        remove_idx = -1
        for i in range(len(self.waveform_selections) - 1, -1, -1):
            if int(self.waveform_selections[i].get("pick_word", apick)) == apick:
                remove_idx = i
                break
        if remove_idx < 0:
            self.lbl_status.setText("Shift+V：当前拾取字下没有可删除的 V 段")
            return
        last = self.waveform_selections.pop(remove_idx)
        key = (int(last.get("trace_idx", -1)), int(last.get("pick_word", int(self.spin_apick.value()))))
        if key in self._waveop_corrected_ttrue:
            del self._waveop_corrected_ttrue[key]
        self.waveop_stack_result = None
        self._refresh_waveop_selection_list()
        self.request_render(delay_ms=10)
        self.lbl_status.setText(
            f"已删除最近 V 段：道 {int(last.get('trace_idx', -1))}，中心={float(last.get('t_display', 0.0)):.3f}s {self._waveop_apick_status_suffix()}"
        )

    def _clear_waveform_selections(self) -> None:
        self.waveform_selections = []
        self._waveop_corrected_ttrue = {}
        self.waveop_stack_result = None
        self._refresh_waveop_selection_list()
        self.request_render(delay_ms=10)
        self.lbl_status.setText(f"已清除所有 V 选波窗口 {self._waveop_apick_status_suffix()}")

    def _refresh_waveop_selection_list(self) -> None:
        widget = getattr(self, "list_waveop_segments", None)
        if widget is None:
            return
        apick = int(self.spin_apick.value())
        try:
            widget.clear()
            current = self._current_apick_waveform_selections()
            if not current:
                widget.addItem("无 V 段")
                return
            widget.addItem(f"apick={apick} 共 {len(current)} 段")
            for i, sel in enumerate(current, start=1):
                trace_idx = int(sel.get("trace_idx", -1))
                t_disp = float(sel.get("t_display", 0.0))
                t0 = t_disp - 0.3
                t1 = t_disp + 0.7
                widget.addItem(f"{i:02d} 道{trace_idx} {t_disp:.3f}s [{t0:.3f},{t1:.3f}]")
        except Exception:
            pass

    def _run_waveop_stack_from_selections(self) -> None:
        """波形操作：按 V 段执行 F 同款自适应拾取更新，再叠加。"""
        if self.loaded is None:
            self.lbl_status.setText("波形叠加失败：请先加载数据")
            return
        selections = self._current_apick_waveform_selections()
        if len(selections) < 2:
            self.lbl_status.setText("波形叠加失败：当前拾取字下请先用 V 至少标注2个波形段")
            return
        traces = self.loaded.get("traces", [])
        times = np.asarray(self.loaded.get("times", []), dtype=float)
        offsets_all = np.asarray(self.loaded.get("offsets", []), dtype=float)
        if len(traces) == 0 or times.size < 4 or offsets_all.size == 0:
            self.lbl_status.setText("波形叠加失败：当前数据无效")
            return
        dt = float(times[1] - times[0]) if times.size > 1 else 0.001
        t0 = float(times[0])
        tau = np.arange(-0.3, 0.7 + 0.5 * dt, dt, dtype=np.float64)
        if tau.size < 8:
            self.lbl_status.setText("波形叠加失败：时间采样不足")
            return

        trace_indices: List[int] = []
        sel_refs: List[Dict[str, float]] = []
        for sel in selections:
            ig = int(sel.get("trace_idx", -1))
            if ig < 0 or ig >= len(traces) or ig >= offsets_all.size:
                continue
            trace_indices.append(ig)
            sel_refs.append(sel)
        if len(trace_indices) < 2:
            self.lbl_status.setText("波形叠加失败：有效 V 段不足")
            return

        proc_params = self._build_processing_params()
        raw_selected_traces = [np.asarray(traces[ig], dtype=float) for ig in trace_indices]
        selected_offsets = np.asarray([float(offsets_all[ig]) for ig in trace_indices], dtype=float)
        try:
            processed_traces = self.processor.process_traces(
                raw_selected_traces,
                times,
                selected_offsets,
                proc_params,
                realtime_interaction=False,
            )
        except Exception:
            processed_traces = raw_selected_traces

        selected_traces: List[np.ndarray] = []
        reduction_shifts: List[float] = []
        for li, ig in enumerate(trace_indices):
            red_shift = float(self._compute_reduction_tshift(int(ig), float(offsets_all[int(ig)])))
            reduction_shifts.append(red_shift)
            tr = np.asarray(processed_traces[li], dtype=float)
            tr_disp = np.interp(
                times - red_shift,
                times,
                tr,
                left=0.0,
                right=0.0,
            )
            selected_traces.append(tr_disp)

        # V 段中心作为“初始拾取”，在 F 同款自适应更新中迭代
        initial_picks: List[int] = []
        for li, sel in enumerate(sel_refs):
            t_true = float(sel.get("t_true", 0.0))
            t_display_reduced = t_true + float(reduction_shifts[li])
            ip = int(round((t_display_reduced - t0) / dt))
            initial_picks.append(ip if 0 <= ip < times.size else -1)

        try:
            result = self.adaptive_stacker.align_traces(
                traces=selected_traces,
                times=times,
                initial_picks=initial_picks,
            )
        except Exception as exc:
            self.lbl_status.setText(f"V段自适应更新失败: {exc}")
            return
        shifts = list(result.get("time_shifts", []))
        if not shifts:
            self.lbl_status.setText("V段自适应更新失败：未返回有效偏移")
            return

        updated_count = 0
        segments: List[np.ndarray] = []
        centers: List[float] = []
        for li, ig in enumerate(trace_indices):
            if li >= len(shifts) or initial_picks[li] < 0:
                continue
            old_true = float(sel_refs[li].get("t_true", 0.0))
            new_true = old_true + float(shifts[li])
            new_true = float(np.clip(new_true, t0, float(times[-1])))
            # V 段属于模板基准点：仅更新 V 基准，不写入拾取点
            sel_refs[li]["t_true"] = new_true
            sel_refs[li]["t_display"] = new_true + float(self._compute_display_tshift(int(ig), float(offsets_all[int(ig)])))
            key = (int(ig), int(sel_refs[li].get("pick_word", int(self.spin_apick.value()))))
            self._waveop_corrected_ttrue[key] = float(new_true)
            updated_count += 1

            # 按“更新后的拾取中心”截取窗口进行叠加
            t_center_display_reduced = new_true + float(reduction_shifts[li])
            seg = np.interp(
                t_center_display_reduced + tau,
                times,
                np.asarray(selected_traces[li], dtype=float),
                left=0.0,
                right=0.0,
            )
            amp = float(np.percentile(np.abs(seg), 98)) if seg.size > 0 else 0.0
            if amp > 1e-12:
                seg = seg / amp
            segments.append(np.asarray(seg, dtype=np.float64))
            centers.append(float(sel_refs[li]["t_display"]))

        if updated_count < 2 or len(segments) < 2:
            self.lbl_status.setText("波形叠加失败：可更新/可叠加的 V 段不足")
            return

        stack = np.mean(np.asarray(segments, dtype=np.float64), axis=0)
        self.waveop_stack_result = {
            "tau": tau.astype(np.float64),
            "stack": np.asarray(stack, dtype=np.float64),
            "centers": np.asarray(centers, dtype=np.float64),
        }
        self._refresh_waveop_selection_list()
        self.request_render(delay_ms=10)
        self.lbl_status.setText(
            f"V段流程完成：更新 {updated_count} 条V基准并完成叠加（处理后波形，不写入拾取）"
        )

    def _extract_trace_xyz(self, trace_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        headers = self.loaded.get("trace_headers", []) if self.loaded is not None else []
        if trace_idx < 0 or trace_idx >= len(headers):
            return np.zeros(3, dtype=float), np.zeros(3, dtype=float)
        th = headers[trace_idx]
        use_utm, rec_role = self._infer_rec_role_for_orientation()

        sxutm = float(getattr(th, "sxutm", 0.0) or 0.0)
        syutm = float(getattr(th, "syutm", 0.0) or 0.0)
        rxutm = float(getattr(th, "rxutm", 0.0) or 0.0)
        ryutm = float(getattr(th, "ryutm", 0.0) or 0.0)
        swdepth = float(getattr(th, "swdepth", 0.0) or 0.0)
        sz = float(getattr(th, "sz", swdepth) or swdepth)
        rz = float(getattr(th, "rz", 0.0) or 0.0)

        if use_utm and any(abs(v) > 1e-6 for v in (sxutm, syutm, rxutm, ryutm)):
            if rec_role == "sx":
                src = np.array([rxutm, ryutm, rz], dtype=float)
                rec = np.array([sxutm, syutm, sz], dtype=float)
            else:
                src = np.array([sxutm, syutm, sz], dtype=float)
                rec = np.array([rxutm, ryutm, rz], dtype=float)
            return src, rec

        slon = float(getattr(th, "slong", 0.0) or 0.0)
        slat = float(getattr(th, "slat", 0.0) or 0.0)
        rlon = float(getattr(th, "rlong", 0.0) or 0.0)
        rlat = float(getattr(th, "rlat", 0.0) or 0.0)
        if rec_role == "sx":
            src = np.array([rlon, rlat, rz], dtype=float)
            rec = np.array([slon, slat, sz], dtype=float)
        else:
            src = np.array([slon, slat, sz], dtype=float)
            rec = np.array([rlon, rlat, rz], dtype=float)
        return src, rec

    def _extract_trace_source_coords(self, trace_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        headers = self.loaded.get("trace_headers", []) if self.loaded is not None else []
        if trace_idx < 0 or trace_idx >= len(headers):
            return None, None
        th = headers[trace_idx]
        use_utm, rec_role = self._infer_rec_role_for_orientation()
        sxutm = float(getattr(th, "sxutm", 0.0) or 0.0)
        syutm = float(getattr(th, "syutm", 0.0) or 0.0)
        slon = float(getattr(th, "slong", 0.0) or 0.0)
        slat = float(getattr(th, "slat", 0.0) or 0.0)
        rxutm = float(getattr(th, "rxutm", 0.0) or 0.0)
        ryutm = float(getattr(th, "ryutm", 0.0) or 0.0)
        rlon = float(getattr(th, "rlong", 0.0) or 0.0)
        rlat = float(getattr(th, "rlat", 0.0) or 0.0)
        src_utm = None
        src_geo = None
        if use_utm:
            if rec_role == "sx":
                if abs(rxutm) > 1e-9 or abs(ryutm) > 1e-9:
                    src_utm = np.array([rxutm, ryutm], dtype=float)
            else:
                if abs(sxutm) > 1e-9 or abs(syutm) > 1e-9:
                    src_utm = np.array([sxutm, syutm], dtype=float)
        else:
            if rec_role == "sx":
                if abs(rlon) > 1e-9 or abs(rlat) > 1e-9:
                    src_geo = np.array([rlon, rlat], dtype=float)
            else:
                if abs(slon) > 1e-9 or abs(slat) > 1e-9:
                    src_geo = np.array([slon, slat], dtype=float)
        return src_geo, src_utm

    def _infer_rec_role_for_orientation(self) -> Tuple[bool, str]:
        trace_headers = self.loaded.get("trace_headers", []) if self.loaded is not None else []
        if not trace_headers:
            return False, "rx"

        def _valid_xy(x: float, y: float) -> bool:
            return math.isfinite(x) and math.isfinite(y) and (abs(x) > 1e-9 or abs(y) > 1e-9)

        use_utm = False
        for th in trace_headers:
            sxutm = float(getattr(th, "sxutm", 0.0) or 0.0)
            syutm = float(getattr(th, "syutm", 0.0) or 0.0)
            rxutm = float(getattr(th, "rxutm", 0.0) or 0.0)
            ryutm = float(getattr(th, "ryutm", 0.0) or 0.0)
            if _valid_xy(sxutm, syutm) or _valid_xy(rxutm, ryutm):
                use_utm = True
                break

        sx_series: List[Tuple[float, float]] = []
        rx_series: List[Tuple[float, float]] = []
        for th in trace_headers:
            if use_utm:
                sxv = float(getattr(th, "sxutm", 0.0) or 0.0)
                syv = float(getattr(th, "syutm", 0.0) or 0.0)
                rxv = float(getattr(th, "rxutm", 0.0) or 0.0)
                ryv = float(getattr(th, "ryutm", 0.0) or 0.0)
            else:
                sxv = float(getattr(th, "slong", 0.0) or 0.0)
                syv = float(getattr(th, "slat", 0.0) or 0.0)
                rxv = float(getattr(th, "rlong", 0.0) or 0.0)
                ryv = float(getattr(th, "rlat", 0.0) or 0.0)
            if _valid_xy(sxv, syv):
                sx_series.append((sxv, syv))
            if _valid_xy(rxv, ryv):
                rx_series.append((rxv, ryv))

        def _var_count(series: List[Tuple[float, float]], decimals: int) -> int:
            if not series:
                return 0
            uniq = {(round(float(x), decimals), round(float(y), decimals)) for x, y in series}
            return len(uniq)

        dec = 2 if use_utm else 6
        s_var = _var_count(sx_series, dec)
        r_var = _var_count(rx_series, dec)
        rec_role = "sx" if s_var >= r_var else "rx"
        return use_utm, rec_role

    @staticmethod
    def _normalize_depth_to_km(v: float) -> float:
        d = abs(float(v))
        return d / 1000.0 if d > 20.0 else d

    def _convert_xy_to_utm_guess(self, x: float, y: float) -> Tuple[float, float]:
        if -180.5 <= float(x) <= 360.5 and -90.5 <= float(y) <= 90.5:
            lon = np.asarray([float(x)], dtype=float)
            lat = np.asarray([float(y)], dtype=float)
            zone_override, hemi_override = self._get_manual_utm_override()
            xu, yu, _ = self._convert_lonlat_to_utm_arrays(
                lon,
                lat,
                override_zone=zone_override,
                override_hemisphere=hemi_override,
            )
            if np.isfinite(xu[0]) and np.isfinite(yu[0]):
                return float(xu[0]), float(yu[0])
        return float(x), float(y)

    def _convert_orientation_terrain_to_utm(self, terrain_meta: Dict[str, object]) -> Optional[Dict[str, object]]:
        mode = str((terrain_meta or {}).get("mode", "")).lower()
        coord_kind = str((terrain_meta or {}).get("coord_kind", "unknown")).lower()
        if mode not in ("grid", "points"):
            return None
        if coord_kind == "utm":
            out = dict(terrain_meta)
            out["coord_kind"] = "utm"
            return out

        if coord_kind != "geo":
            return None

        try:
            if mode == "grid":
                x = np.asarray(terrain_meta.get("x", []), dtype=float)
                y = np.asarray(terrain_meta.get("y", []), dtype=float)
                z = np.asarray(terrain_meta.get("z", []), dtype=float)
                if x.size < 2 or y.size < 2 or z.size == 0:
                    return None
                xx, yy = np.meshgrid(x, y, indexing="xy")
                lon, lat, _ = self._normalize_geo_lonlat_order(xx.reshape(-1), yy.reshape(-1))
                zone_override, hemi_override = self._get_manual_utm_override()
                x_utm, y_utm, _info = self._convert_lonlat_to_utm_arrays(
                    lon,
                    lat,
                    override_zone=zone_override,
                    override_hemisphere=hemi_override,
                )
                z_flat = np.asarray(z, dtype=float).reshape(-1)
                valid = np.isfinite(x_utm) & np.isfinite(y_utm) & np.isfinite(z_flat)
                if not np.any(valid):
                    return None
                x_utm = np.asarray(x_utm[valid], dtype=float)
                y_utm = np.asarray(y_utm[valid], dtype=float)
                z_flat = np.asarray(z_flat[valid], dtype=float)
                return {
                    "mode": "points",
                    "coord_kind": "utm",
                    "x": x_utm,
                    "y": y_utm,
                    "z": z_flat,
                    "path": str(terrain_meta.get("path", "")),
                }

            x = np.asarray(terrain_meta.get("x", []), dtype=float).reshape(-1)
            y = np.asarray(terrain_meta.get("y", []), dtype=float).reshape(-1)
            z = np.asarray(terrain_meta.get("z", []), dtype=float).reshape(-1)
            lon, lat, _ = self._normalize_geo_lonlat_order(x, y)
            zone_override, hemi_override = self._get_manual_utm_override()
            x_utm, y_utm, _info = self._convert_lonlat_to_utm_arrays(
                lon,
                lat,
                override_zone=zone_override,
                override_hemisphere=hemi_override,
            )
            valid = np.isfinite(x_utm) & np.isfinite(y_utm) & np.isfinite(z)
            if not np.any(valid):
                return None
            return {
                "mode": "points",
                "coord_kind": "utm",
                "x": np.asarray(x_utm[valid], dtype=float),
                "y": np.asarray(y_utm[valid], dtype=float),
                "z": np.asarray(z[valid], dtype=float),
                "path": str(terrain_meta.get("path", "")),
            }
        except Exception:
            return None

    def _collect_orientation_selected_points_utm(self) -> Tuple[np.ndarray, np.ndarray]:
        """Collect source/receiver UTM points for V-selected traces only."""
        src_pts: List[Tuple[float, float]] = []
        rec_pts: List[Tuple[float, float]] = []
        selections = self._current_apick_waveform_selections()

        for sel in selections:
            trace_idx = int(sel.get("trace_idx", -1))
            src, rec = self._extract_trace_xyz(trace_idx)
            if np.isfinite(src[:2]).all() and (abs(float(src[0])) > 1e-9 or abs(float(src[1])) > 1e-9):
                sx, sy = self._convert_xy_to_utm_guess(float(src[0]), float(src[1]))
                src_pts.append((sx, sy))
            if np.isfinite(rec[:2]).all() and (abs(float(rec[0])) > 1e-9 or abs(float(rec[1])) > 1e-9):
                rx, ry = self._convert_xy_to_utm_guess(float(rec[0]), float(rec[1]))
                rec_pts.append((rx, ry))
        src_arr = np.asarray(src_pts, dtype=float) if src_pts else np.empty((0, 2), dtype=float)
        rec_arr = np.asarray(rec_pts, dtype=float) if rec_pts else np.empty((0, 2), dtype=float)
        return src_arr, rec_arr

    def _render_orientation_terrain_preview(
        self, plot: pg.PlotWidget, terrain_meta_utm: Optional[Dict[str, object]]
    ) -> None:
        pi = plot.getPlotItem()
        pi.clear()
        pi.showGrid(x=True, y=True, alpha=0.15)
        pi.setLabels(left="Y (m)", bottom="X (m)")
        terrain_bounds: Optional[Tuple[float, float, float, float]] = None
        if terrain_meta_utm is not None:
            mode = str(terrain_meta_utm.get("mode", "")).lower()
            if mode == "grid":
                x = np.asarray(terrain_meta_utm.get("x", []), dtype=float)
                y = np.asarray(terrain_meta_utm.get("y", []), dtype=float)
                z = np.asarray(terrain_meta_utm.get("z", []), dtype=float)
                if x.size > 1 and y.size > 1 and z.size > 0:
                    img = pg.ImageItem(np.asarray(z.T, dtype=float), axisOrder="col-major")
                    xmin, xmax = float(np.min(x)), float(np.max(x))
                    ymin, ymax = float(np.min(y)), float(np.max(y))
                    img.setRect(QtCore.QRectF(xmin, ymin, xmax - xmin, ymax - ymin))
                    img.setOpacity(0.55)
                    img.setZValue(-10)
                    pi.addItem(img)
                    terrain_bounds = (xmin, xmax, ymin, ymax)
            else:
                x = np.asarray(terrain_meta_utm.get("x", []), dtype=float).reshape(-1)
                y = np.asarray(terrain_meta_utm.get("y", []), dtype=float).reshape(-1)
                z = np.asarray(terrain_meta_utm.get("z", []), dtype=float).reshape(-1)
                valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                x, y, z = x[valid], y[valid], z[valid]
                if x.size > 0:
                    xmin, xmax = float(np.min(x)), float(np.max(x))
                    ymin, ymax = float(np.min(y)), float(np.max(y))
                    terrain_bounds = (xmin, xmax, ymin, ymax)
                    # 大点云预览使用栅格化渲染，避免UI卡死；不影响原始数据采样精度。
                    if x.size > 120000:
                        nx = 700
                        ny = max(220, int(round(nx * max(1e-9, (ymax - ymin)) / max(1e-9, (xmax - xmin)))))
                        ny = min(ny, 900)
                        x_edges = np.linspace(xmin, xmax, nx + 1)
                        y_edges = np.linspace(ymin, ymax, ny + 1)
                        sum_z, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges], weights=z)
                        cnt, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])
                        with np.errstate(invalid="ignore", divide="ignore"):
                            z_grid = sum_z / cnt
                        if np.any(np.isfinite(z_grid)):
                            fill_value = float(np.nanmedian(z))
                            z_grid = np.where(np.isfinite(z_grid), z_grid, fill_value)
                            rgba = self._terrain_rgba_from_grid(
                                z_grid,
                                palette="terrain",
                                shade_strength=0.75,
                                coast_enhance=True,
                                light_alt_deg=45.0,
                                light_az_deg=315.0,
                            )
                            img = pg.ImageItem(rgba, axisOrder="row-major")
                            img.setRect(QtCore.QRectF(xmin, ymin, xmax - xmin, ymax - ymin))
                            img.setZValue(-8)
                            pi.addItem(img)
                        else:
                            # 回退：小规模散点渲染
                            sp = pg.ScatterPlotItem(x=x, y=y, size=2.5, pen=None, brush=pg.mkBrush(120, 140, 180, 90))
                            sp.setZValue(-8)
                            pi.addItem(sp)
                    else:
                        zmin = float(np.nanmin(z))
                        zmax = float(np.nanmax(z))
                        span = max(1e-12, zmax - zmin)
                        norm = np.clip((z - zmin) / span, 0.0, 1.0)
                        rgb = self._terrain_colormap_rgb(norm, palette="terrain")
                        spots = []
                        for i in range(int(x.size)):
                            c = rgb[i]
                            spots.append(
                                {
                                    "pos": (float(x[i]), float(y[i])),
                                    "size": 3.2,
                                    "brush": pg.mkBrush(int(c[0]), int(c[1]), int(c[2]), 110),
                                    "pen": pg.mkPen(int(c[0]), int(c[1]), int(c[2]), 90, width=0.3),
                                }
                            )
                        sp = pg.ScatterPlotItem(pxMode=True)
                        sp.setData(spots=spots)
                        sp.setZValue(-8)
                        pi.addItem(sp)

        src_arr, rec_arr = self._collect_orientation_selected_points_utm()
        if src_arr.size > 0:
            src_item = pg.ScatterPlotItem(
                x=src_arr[:, 0], y=src_arr[:, 1],
                size=13, pen=pg.mkPen("#7f1d1d", width=1.1), brush=pg.mkBrush("#ef4444"), symbol="t"
            )
            src_item.setZValue(20)
            pi.addItem(src_item)
        if rec_arr.size > 0:
            rec_item = pg.ScatterPlotItem(
                x=rec_arr[:, 0], y=rec_arr[:, 1],
                size=8, pen=pg.mkPen("#334155", width=1.0), brush=pg.mkBrush("#94a3b8"), symbol="o"
            )
            rec_item.setZValue(18)
            pi.addItem(rec_item)

        all_x: List[np.ndarray] = []
        all_y: List[np.ndarray] = []
        if terrain_bounds is not None:
            all_x.append(np.asarray([terrain_bounds[0], terrain_bounds[1]], dtype=float))
            all_y.append(np.asarray([terrain_bounds[2], terrain_bounds[3]], dtype=float))
        if src_arr.size > 0:
            all_x.append(src_arr[:, 0])
            all_y.append(src_arr[:, 1])
        if rec_arr.size > 0:
            all_x.append(rec_arr[:, 0])
            all_y.append(rec_arr[:, 1])
        if all_x and all_y:
            x_all = np.concatenate(all_x)
            y_all = np.concatenate(all_y)
            if x_all.size > 0 and y_all.size > 0:
                xmin, xmax = float(np.nanmin(x_all)), float(np.nanmax(x_all))
                ymin, ymax = float(np.nanmin(y_all)), float(np.nanmax(y_all))
                dx = max(1.0, xmax - xmin)
                dy = max(1.0, ymax - ymin)
                px = 0.04 * dx
                py = 0.04 * dy
                plot.setXRange(xmin - px, xmax + px, padding=0.0)
                plot.setYRange(ymin - py, ymax + py, padding=0.0)

    def _terrain_depth_sampler_from_observations(
        self, observations: List[OrientationObservation]
    ) -> Optional[Callable[[float, float], Optional[float]]]:
        terrain_meta = getattr(self, "_orientation_terrain_meta_utm", None)
        terrain_sampler = build_bathymetry_sampler(terrain_meta)
        if terrain_sampler is None:
            return None
        coord_kind = str((terrain_meta or {}).get("coord_kind", "unknown")).lower()
        geo_vals = [o.source_xy_geo for o in observations if o.source_xy_geo is not None]
        utm_vals = [o.source_xy_utm for o in observations if o.source_xy_utm is not None]
        fallback_candidates: List[Tuple[float, float]] = []
        if coord_kind == "utm":
            if utm_vals:
                u = np.asarray(utm_vals, dtype=float)
                fallback_candidates.append((float(np.median(u[:, 0])), float(np.median(u[:, 1]))))
            if geo_vals:
                g = np.asarray(geo_vals, dtype=float)
                gx, gy = float(np.median(g[:, 0])), float(np.median(g[:, 1]))
                fallback_candidates.append(self._convert_xy_to_utm_guess(gx, gy))
        else:
            if geo_vals:
                g = np.asarray(geo_vals, dtype=float)
                fallback_candidates.append((float(np.median(g[:, 0])), float(np.median(g[:, 1]))))
            if utm_vals:
                u = np.asarray(utm_vals, dtype=float)
                fallback_candidates.append((float(np.median(u[:, 0])), float(np.median(u[:, 1]))))

        def _normalize_depth_km(v: float) -> float:
            return self._normalize_depth_to_km(float(v))

        def _try_sample(cx: float, cy: float) -> Optional[float]:
            v = terrain_sampler(float(cx), float(cy))
            if v is not None and np.isfinite(float(v)):
                return float(_normalize_depth_km(float(v)))
            return None

        def _depth_sampler(x: float, y: float) -> Optional[float]:
            xx, yy = float(x), float(y)
            if coord_kind == "utm":
                # If caller passes lon/lat while terrain is UTM, convert first.
                xx, yy = self._convert_xy_to_utm_guess(xx, yy)
            tries: List[Tuple[float, float]] = [(xx, yy)]
            if coord_kind == "geo":
                tries.append((float(y), float(x)))  # 经纬顺序兜底
            if coord_kind == "unknown":
                tries.append((float(y), float(x)))
            for cx, cy in fallback_candidates:
                tries.append((cx, cy))
                if coord_kind in ("geo", "unknown"):
                    tries.append((cy, cx))
            for cx, cy in tries:
                out = _try_sample(cx, cy)
                if out is not None:
                    return out
            return None

        return _depth_sampler

    def _sample_initial_depth_km(
        self, observations: List[OrientationObservation], depth_sampler: Callable[[float, float], Optional[float]]
    ) -> Optional[float]:
        if not observations:
            return None
        for o in observations:
            for xy in (o.source_xy_utm, o.source_xy_geo):
                if xy is None:
                    continue
                d = depth_sampler(float(xy[0]), float(xy[1]))
                if d is not None and np.isfinite(float(d)) and float(d) > 0.0:
                    return float(d)
        return None

    def _show_orientation_input_preview(
        self, observations: List[OrientationObservation], depth_km: Optional[float], max_rows: int = 5
    ) -> bool:
        water_v = 1.5
        depth_ok = depth_km is not None and np.isfinite(float(depth_km)) and float(depth_km) > 0.0
        depth_text = f"{float(depth_km):.4f} km" if depth_ok else "无效/未采样到"
        # 预览优先展示偏移距最小的道，便于先核查近偏移数据质量。
        preview_obs = sorted(
            list(observations),
            key=lambda o: (abs(float(o.offset_km)), int(o.trace_idx)),
        )
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("姿态校正输入预览")
        dlg.resize(980, 460)
        lay = QtWidgets.QVBoxLayout(dlg)
        summary = QtWidgets.QLabel(
            f"水深: {depth_text}    海水波速: {water_v:.3f} km/s    "
            f"观测数量: {len(observations)}（按偏移距从小到大预览前 {min(len(preview_obs), max_rows)} 条）",
            dlg,
        )
        summary.setStyleSheet("font-weight:600;")
        lay.addWidget(summary)

        table = QtWidgets.QTableWidget(dlg)
        table.setColumnCount(9)
        table.setHorizontalHeaderLabels(
            [
                "道号",
                "震源点坐标(x,y,z)",
                "接收点坐标(x,y,z)",
                "偏移距(km)",
                "斜线距离(km)",
                "预测走时(s)",
                "基准折合走时(s)",
                "基准真实走时(s)",
                "走时残差(s)",
            ]
        )
        table.setRowCount(min(len(preview_obs), max_rows))
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        table.verticalHeader().setVisible(False)

        n = min(len(preview_obs), max_rows)
        for i in range(n):
            o = preview_obs[i]
            off = abs(float(o.offset_km))
            t_true_obs = float(o.t0)
            x_off = float(off)
            if self.loaded is not None:
                offsets_all = np.asarray(self.loaded.get("offsets", []), dtype=float)
                tidx = int(o.trace_idx)
                if offsets_all.size > tidx >= 0:
                    x_off = float(offsets_all[tidx])
            t_fold_obs = t_true_obs + float(self._compute_display_tshift(int(o.trace_idx), x_off))
            if depth_ok:
                slant = float(np.sqrt(off * off + float(depth_km) * float(depth_km)))
                t_pred = slant / water_v
                residual = t_pred - t_true_obs
                slant_text = f"{slant:.4f}"
                tpred_text = f"{t_pred:.4f}"
                res_text = f"{residual:.4f}"
            else:
                slant_text = "N/A"
                tpred_text = "N/A"
                res_text = "N/A"
            s = o.source_xyz
            r = o.receiver_xyz
            vals = [
                str(int(o.trace_idx)),
                f"({s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f})",
                f"({r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f})",
                f"{off:.4f}",
                slant_text,
                tpred_text,
                f"{t_fold_obs:.4f}",
                f"{t_true_obs:.4f}",
                res_text,
            ]
            for c, v in enumerate(vals):
                table.setItem(i, c, QtWidgets.QTableWidgetItem(v))

        table.horizontalHeader().setStretchLastSection(True)
        table.resizeColumnsToContents()
        lay.addWidget(table, stretch=1)

        if depth_ok:
            hint = QtWidgets.QLabel("确认后将按上述参数执行姿态校正。", dlg)
        else:
            hint = QtWidgets.QLabel("当前水深无效：可查看预览，但无法开始校正。请先在姿态校正窗口加载水深文件。", dlg)
        hint.setStyleSheet("color:#4b5563;")
        lay.addWidget(hint)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        btn_cancel = QtWidgets.QPushButton("取消", dlg)
        btn_ok = QtWidgets.QPushButton("确认并开始", dlg)
        btn_ok.setEnabled(bool(depth_ok))
        btn_cancel.clicked.connect(dlg.reject)
        btn_ok.clicked.connect(dlg.accept)
        row.addWidget(btn_cancel)
        row.addWidget(btn_ok)
        lay.addLayout(row)
        return dlg.exec() == int(QtWidgets.QDialog.DialogCode.Accepted)

    def _find_3c_group_for_trace(self, trace_idx: int) -> Tuple[Optional[Dict[int, int]], str]:
        if self.loaded is None:
            return None, "未加载数据"
        headers = self.loaded.get("trace_headers", []) or []
        offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
        if trace_idx < 0 or trace_idx >= len(headers):
            return None, f"无效道号 {trace_idx}"

        base = headers[trace_idx]
        shot = int(getattr(base, "ishoti", 0) or 0)
        rec = int(getattr(base, "ireci", 0) or 0)
        comp_map: Dict[int, int] = {}
        for i, th in enumerate(headers):
            if int(getattr(th, "ishoti", 0) or 0) != shot:
                continue
            if int(getattr(th, "ireci", 0) or 0) != rec:
                continue
            c = int(getattr(th, "itypei", 0) or 0)
            if c in (1, 2, 3):
                comp_map[c] = i

        # Fallback: same shot + nearest offset for missing components.
        if len(comp_map) < 3 and offsets.size == len(headers):
            x0 = float(offsets[trace_idx])
            for c in (1, 2, 3):
                if c in comp_map:
                    continue
                best_i = -1
                best_dx = float("inf")
                for i, th in enumerate(headers):
                    if int(getattr(th, "ishoti", 0) or 0) != shot:
                        continue
                    if int(getattr(th, "itypei", 0) or 0) != c:
                        continue
                    dx = abs(float(offsets[i]) - x0)
                    if dx < best_dx:
                        best_dx = dx
                        best_i = i
                if best_i >= 0:
                    comp_map[c] = best_i

        missing = [name for code, name in ((1, "垂直"), (2, "径向"), (3, "切向")) if code not in comp_map]
        if missing:
            return None, "缺少三分量: " + ", ".join(missing)
        return comp_map, ""

    def _build_orientation_observations(self) -> Tuple[Optional[List[OrientationObservation]], str]:
        if self.loaded is None:
            return None, "请先加载数据"
        selections = self._current_apick_waveform_selections()
        if not selections:
            return None, "请先使用 V 键选择至少1段波形"

        traces = self.loaded.get("traces", [])
        times = np.asarray(self.loaded.get("times", []), dtype=float)
        offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
        if len(traces) == 0 or times.size < 4:
            return None, "当前数据无效"
        dt = float(times[1] - times[0])
        if dt <= 0:
            return None, "时间采样间隔异常"

        pre_sec = float(self._orientation_ui_params.get("wave_pre", 0.30))
        post_sec = float(self._orientation_ui_params.get("wave_post", 0.70))
        obs_list: List[OrientationObservation] = []
        for sel in selections:
            trace_idx = int(sel.get("trace_idx", -1))
            group, err = self._find_3c_group_for_trace(trace_idx)
            if group is None:
                return None, f"道 {trace_idx} 无法构造三分量：{err}"

            pick_word = int(sel.get("pick_word", int(self.spin_apick.value())))
            key = (int(trace_idx), int(pick_word))
            t_ref = float(self._waveop_corrected_ttrue.get(key, float(sel.get("t_true", 0.0))))
            t0 = t_ref - pre_sec
            t1 = t_ref + post_sec
            tau = np.arange(t0, t1 + 0.5 * dt, dt, dtype=float)
            if tau.size < 8:
                return None, f"道 {trace_idx} 截窗采样点不足"

            ztr = np.asarray(traces[int(group[1])], dtype=float)
            rtr = np.asarray(traces[int(group[2])], dtype=float)
            ttr = np.asarray(traces[int(group[3])], dtype=float)
            z_win = np.interp(tau, times, ztr, left=0.0, right=0.0)
            r_win = np.interp(tau, times, rtr, left=0.0, right=0.0)
            t_win = np.interp(tau, times, ttr, left=0.0, right=0.0)
            src, rec = self._extract_trace_xyz(int(group[1]))
            src_geo, src_utm = self._extract_trace_source_coords(int(group[1]))
            off_km = float(offsets[int(group[1])]) if offsets.size > int(group[1]) else 0.0

            obs_list.append(
                OrientationObservation(
                    trace_idx=int(trace_idx),
                    pick_word=int(pick_word),
                    t0=float(t_ref),
                    dt=float(dt),
                    z=np.asarray(z_win, dtype=float),
                    r=np.asarray(r_win, dtype=float),
                    t=np.asarray(t_win, dtype=float),
                    source_xyz=np.asarray(src, dtype=float),
                    receiver_xyz=np.asarray(rec, dtype=float),
                    offset_km=float(off_km),
                    source_xy_geo=np.asarray(src_geo, dtype=float) if src_geo is not None else None,
                    source_xy_utm=np.asarray(src_utm, dtype=float) if src_utm is not None else None,
                )
            )
        return obs_list, ""

    def _open_attitude_correction_dialog(self) -> None:
        dlg = QtWidgets.QDialog(None)
        dlg.setWindowTitle("姿态校正参数")
        dlg.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        dlg.resize(980, 640)
        lay = QtWidgets.QVBoxLayout(dlg)
        form = QtWidgets.QFormLayout()
        spin_wave_pre = QtWidgets.QDoubleSpinBox(dlg)
        spin_wave_pre.setRange(0.05, 2.5)
        spin_wave_pre.setDecimals(3)
        spin_wave_pre.setSingleStep(0.05)
        spin_wave_pre.setValue(float(self._orientation_ui_params.get("wave_pre", 0.30)))
        spin_wave_post = QtWidgets.QDoubleSpinBox(dlg)
        spin_wave_post.setRange(0.05, 3.5)
        spin_wave_post.setDecimals(3)
        spin_wave_post.setSingleStep(0.05)
        spin_wave_post.setValue(float(self._orientation_ui_params.get("wave_post", 0.70)))
        spin_iter = QtWidgets.QSpinBox(dlg)
        spin_iter.setRange(1, 20)
        spin_iter.setValue(int(round(float(self._orientation_ui_params.get("att_iter", 4.0)))))
        spin_wtt = QtWidgets.QDoubleSpinBox(dlg)
        spin_wtt.setRange(0.0, 10.0)
        spin_wtt.setDecimals(2)
        spin_wtt.setSingleStep(0.1)
        spin_wtt.setValue(float(self._orientation_ui_params.get("att_wtt", 1.0)))
        spin_wpol = QtWidgets.QDoubleSpinBox(dlg)
        spin_wpol.setRange(0.0, 10.0)
        spin_wpol.setDecimals(2)
        spin_wpol.setSingleStep(0.1)
        spin_wpol.setValue(float(self._orientation_ui_params.get("att_wpol", 1.0)))
        spin_wsym = QtWidgets.QDoubleSpinBox(dlg)
        spin_wsym.setRange(0.0, 10.0)
        spin_wsym.setDecimals(2)
        spin_wsym.setSingleStep(0.1)
        spin_wsym.setValue(float(self._orientation_ui_params.get("att_wsym", 0.8)))
        form.addRow("窗前 (s)", spin_wave_pre)
        form.addRow("窗后 (s)", spin_wave_post)
        form.addRow("迭代次数", spin_iter)
        form.addRow("走时权重 wtt", spin_wtt)
        form.addRow("极化权重 wpol", spin_wpol)
        form.addRow("对称权重 wsym", spin_wsym)
        lbl_depth_preview = QtWidgets.QLabel("未计算", dlg)
        lbl_depth_preview.setStyleSheet("color:#0f172a; font-weight:600;")
        form.addRow("当前采样水深(km)", lbl_depth_preview)
        lay.addLayout(form)

        terrain_row = QtWidgets.QHBoxLayout()
        btn_load_terrain = QtWidgets.QPushButton("加载水深文件", dlg)
        btn_clear_terrain = QtWidgets.QPushButton("清除水深文件", dlg)
        lbl_terrain = QtWidgets.QLabel(dlg)
        lbl_terrain.setStyleSheet("color:#334155;")
        lbl_terrain.setWordWrap(True)
        terrain_row.addWidget(btn_load_terrain)
        terrain_row.addWidget(btn_clear_terrain)
        terrain_row.addWidget(lbl_terrain, stretch=1)
        lay.addLayout(terrain_row)

        plot = pg.PlotWidget(background=self._theme_color("plot_bg", "#ffffff"))
        lay.addWidget(plot, stretch=1)
        click_info_label = QtWidgets.QLabel("点击预览中的震源点/接收点可查看该道采样与走时信息。", dlg)
        click_info_label.setStyleSheet("color:#1f2937;")
        click_info_label.setWordWrap(True)
        lay.addWidget(click_info_label)

        def _refresh_terrain_label():
            if self._orientation_terrain_meta_utm is not None:
                p = str(self._orientation_terrain_path or self._orientation_terrain_meta_utm.get("path", ""))
                lbl_terrain.setText(f"已加载水深文件: {Path(p).name if p else '(未知)'}（已转换UTM）")
            else:
                lbl_terrain.setText("未加载水深文件")

        def _refresh_preview():
            self._render_orientation_terrain_preview(plot, self._orientation_terrain_meta_utm)
            # Overlay clickable points with per-trace metadata.
            try:
                observations, _err = self._build_orientation_observations()
            except Exception:
                observations = None
            if not observations:
                return
            depth_sampler = self._terrain_depth_sampler_from_observations(observations)
            spots = []
            for o in observations:
                src_xy = self._convert_xy_to_utm_guess(float(o.source_xyz[0]), float(o.source_xyz[1]))
                rec_xy = self._convert_xy_to_utm_guess(float(o.receiver_xyz[0]), float(o.receiver_xyz[1]))
                for role, xy in (("source", src_xy), ("receiver", rec_xy)):
                    if not (np.isfinite(xy[0]) and np.isfinite(xy[1])):
                        continue
                    data = {
                        "role": role,
                        "trace_idx": int(o.trace_idx),
                        "offset_km": float(abs(o.offset_km)),
                        "t_obs": float(o.t0),
                        "src": np.asarray(o.source_xyz, dtype=float),
                        "rec": np.asarray(o.receiver_xyz, dtype=float),
                    }
                    if depth_sampler is not None:
                        d = depth_sampler(float(xy[0]), float(xy[1]))
                        if d is not None and np.isfinite(float(d)):
                            depth_km = self._normalize_depth_to_km(float(d))
                            slant = float(np.sqrt(float(data["offset_km"]) ** 2 + depth_km ** 2))
                            t_pred = slant / 1.5
                            data["depth_km"] = float(depth_km)
                            data["slant_km"] = float(slant)
                            data["t_pred"] = float(t_pred)
                            data["residual"] = float(t_pred - float(data["t_obs"]))
                        else:
                            data["depth_km"] = float("nan")
                    else:
                        data["depth_km"] = float("nan")

                    spots.append(
                        {
                            "pos": (float(xy[0]), float(xy[1])),
                            "size": 11.0 if role == "source" else 9.0,
                            "brush": pg.mkBrush(255, 255, 255, 0),
                            "pen": pg.mkPen(255, 255, 255, 0),
                            "symbol": "t" if role == "source" else "o",
                            "data": data,
                        }
                    )
            if not spots:
                return
            pick_item = pg.ScatterPlotItem(pxMode=True)
            pick_item.setData(spots=spots)
            pick_item.setZValue(50)
            plot.addItem(pick_item)

            def _on_pick(_item, points):
                if not points:
                    return
                data = points[0].data()
                if not isinstance(data, dict):
                    return
                role = "震源" if str(data.get("role", "")) == "source" else "接收"
                tr = int(data.get("trace_idx", -1))
                off = float(data.get("offset_km", np.nan))
                t_obs = float(data.get("t_obs", np.nan))
                src = np.asarray(data.get("src", [np.nan, np.nan, np.nan]), dtype=float)
                rec = np.asarray(data.get("rec", [np.nan, np.nan, np.nan]), dtype=float)
                depth_km = float(data.get("depth_km", np.nan))
                if np.isfinite(depth_km) and depth_km > 0.0:
                    slant = float(data.get("slant_km", np.nan))
                    t_pred = float(data.get("t_pred", np.nan))
                    residual = float(data.get("residual", np.nan))
                    click_info_label.setText(
                        f"{role}点 | 道{tr} | 水深={depth_km:.4f} km | 偏移距={off:.4f} km | "
                        f"斜距={slant:.4f} km | 预测走时={t_pred:.4f} s | 观测走时={t_obs:.4f} s | 残差={residual:.4f} s\n"
                        f"S=({src[0]:.3f},{src[1]:.3f},{src[2]:.3f})  R=({rec[0]:.3f},{rec[1]:.3f},{rec[2]:.3f})"
                    )
                else:
                    click_info_label.setText(
                        f"{role}点 | 道{tr} | 未采样到有效水深 | 偏移距={off:.4f} km | 观测走时={t_obs:.4f} s\n"
                        f"S=({src[0]:.3f},{src[1]:.3f},{src[2]:.3f})  R=({rec[0]:.3f},{rec[1]:.3f},{rec[2]:.3f})"
                    )

            pick_item.sigClicked.connect(_on_pick)

        def _refresh_depth_preview():
            try:
                observations, _err = self._build_orientation_observations()
            except Exception:
                observations = None
            if not observations:
                lbl_depth_preview.setText("无V段观测")
                return
            depth_sampler = self._terrain_depth_sampler_from_observations(observations)
            if depth_sampler is None:
                lbl_depth_preview.setText("未加载水深")
                return
            depth0 = self._sample_initial_depth_km(observations, depth_sampler)
            if depth0 is None or (not np.isfinite(float(depth0))) or float(depth0) <= 0.0:
                lbl_depth_preview.setText("采样失败")
                return
            lbl_depth_preview.setText(f"{float(depth0):.4f}")

        def _load_terrain():
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "选择水深/地形文件",
                "",
                "Terrain (*.grd *.nc *.xyz *.txt);;NetCDF (*.grd *.nc);;XYZ (*.xyz *.txt);;All files (*)",
                options=self._file_dialog_options(),
            )
            if not path:
                return
            try:
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
                self._set_status_text("正在加载并转换水深文件...", hold_ms=2000)
                QtWidgets.QApplication.processEvents()
                meta_raw = self._load_terrain_meta(path, force_geo=False)
                meta_utm = self._convert_orientation_terrain_to_utm(meta_raw)
                if meta_utm is None:
                    raise ValueError("该水深文件无法转换为UTM坐标")
                self._orientation_terrain_path = str(path)
                self._orientation_terrain_meta_raw = meta_raw
                self._orientation_terrain_meta_utm = meta_utm
                _refresh_terrain_label()
                _refresh_preview()
                _refresh_depth_preview()
                self._set_status_text(f"水深文件已加载并转换UTM：{Path(path).name}", hold_ms=1800)
            except Exception as exc:
                self._show_themed_info("加载水深失败", str(exc))
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()

        def _clear_terrain():
            self._orientation_terrain_path = ""
            self._orientation_terrain_meta_raw = None
            self._orientation_terrain_meta_utm = None
            _refresh_terrain_label()
            _refresh_preview()
            _refresh_depth_preview()

        btn_load_terrain.clicked.connect(_load_terrain)
        btn_clear_terrain.clicked.connect(_clear_terrain)
        _refresh_terrain_label()
        _refresh_preview()
        _refresh_depth_preview()

        tip = QtWidgets.QLabel(
            "说明：姿态校正将优先使用“波形叠加”后更新的V段基准走时；"
            "若未执行波形叠加，则使用V段原始基准。"
            "预览图仅绘制V键选取的接收道与震源。",
            dlg,
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#4b5563;")
        lay.addWidget(tip)
        lbl_run_progress = QtWidgets.QLabel("校正进度：未开始", dlg)
        lbl_run_progress.setStyleSheet("color:#334155;")
        bar_run_progress = QtWidgets.QProgressBar(dlg)
        bar_run_progress.setRange(0, 1)
        bar_run_progress.setValue(0)
        bar_run_progress.setFormat("%v/%m")
        lay.addWidget(lbl_run_progress)
        lay.addWidget(bar_run_progress)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        btn_cancel = QtWidgets.QPushButton("取消", dlg)
        btn_preview = QtWidgets.QPushButton("姿态参数预览", dlg)
        btn_run = QtWidgets.QPushButton("开始校正", dlg)
        btn_cancel.clicked.connect(dlg.close)
        def _preview():
            self._orientation_ui_params["wave_pre"] = float(spin_wave_pre.value())
            self._orientation_ui_params["wave_post"] = float(spin_wave_post.value())
            self._orientation_ui_params["att_iter"] = float(spin_iter.value())
            self._orientation_ui_params["att_wtt"] = float(spin_wtt.value())
            self._orientation_ui_params["att_wpol"] = float(spin_wpol.value())
            self._orientation_ui_params["att_wsym"] = float(spin_wsym.value())
            _refresh_depth_preview()
            self._preview_orientation_input_from_current_params()
        def _run():
            self._orientation_ui_params["wave_pre"] = float(spin_wave_pre.value())
            self._orientation_ui_params["wave_post"] = float(spin_wave_post.value())
            self._orientation_ui_params["att_iter"] = float(spin_iter.value())
            self._orientation_ui_params["att_wtt"] = float(spin_wtt.value())
            self._orientation_ui_params["att_wpol"] = float(spin_wpol.value())
            self._orientation_ui_params["att_wsym"] = float(spin_wsym.value())
            btn_run.setEnabled(False)
            try:
                self._run_attitude_correction_with_current_params(
                    parent_dialog=dlg,
                    progress_bar=bar_run_progress,
                    progress_label=lbl_run_progress,
                )
            finally:
                btn_run.setEnabled(True)
        btn_preview.clicked.connect(_preview)
        btn_run.clicked.connect(_run)
        row.addWidget(btn_cancel)
        row.addWidget(btn_preview)
        row.addWidget(btn_run)
        lay.addLayout(row)
        self._register_floating_dialog(dlg)
        dlg.show()

    def _accept_orientation_solution(self, result) -> None:
        dx, dy, dz = result.position_correction
        t_shift = float(getattr(result, "details", {}).get("time_shift_sec", 0.0))
        self._orientation_current_solution = {
            "azimuth_deg": float(result.azimuth_deg),
            "tilt_deg": float(result.tilt_deg),
            "dx": float(dx),
            "dy": float(dy),
            "dz": float(dz),
            "time_shift_sec": float(t_shift),
            "objective": float(result.objective),
            "accepted": 1.0,
        }
        self.lbl_status.setText(
            f"已接受姿态修正：az={result.azimuth_deg:.2f}°, tilt={result.tilt_deg:.2f}°, "
            f"dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}"
        )

    def _show_orientation_diagnostics(self, result) -> None:
        history = list(getattr(result, "iteration_history", []) or [])
        if not history:
            return
        dlg = QtWidgets.QDialog(None)
        dlg.setWindowTitle("姿态校正迭代诊断")
        dlg.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        dlg.resize(900, 700)
        lay = QtWidgets.QVBoxLayout(dlg)
        plot = pg.PlotWidget(background=self._theme_color("plot_bg", "#ffffff"))
        plot.showGrid(x=True, y=True, alpha=0.16)
        plot.setLabel("left", "归一化目标函数值")
        plot.setLabel("bottom", "迭代轮次")
        lay.addWidget(plot, stretch=1)
        x = np.arange(1, len(history) + 1, dtype=float)
        j = np.asarray([float(h.get("objective", np.nan)) for h in history], dtype=float)
        jtt = np.asarray([float(h.get("J_tt_n", np.nan)) for h in history], dtype=float)
        jpol = np.asarray([float(h.get("J_pol_n", np.nan)) for h in history], dtype=float)
        jsym = np.asarray([float(h.get("J_sym_n", np.nan)) for h in history], dtype=float)
        plot.addLegend(offset=(8, 8))
        plot.plot(x, j, pen=pg.mkPen("#2563eb", width=2.0), symbol="o", symbolSize=6, name="总目标 J(归一)")
        plot.plot(x, jtt, pen=pg.mkPen("#dc2626", width=1.8), symbol="t", symbolSize=5, name="走时项 J_tt(归一)")
        plot.plot(x, jpol, pen=pg.mkPen("#059669", width=1.8), symbol="s", symbolSize=5, name="极化项 J_pol(归一)")
        plot.plot(x, jsym, pen=pg.mkPen("#7c3aed", width=1.8), symbol="d", symbolSize=5, name="对称项 J_sym(归一)")

        pol_plot = pg.PlotWidget(background=self._theme_color("plot_bg", "#ffffff"))
        pol_plot.showGrid(x=True, y=True, alpha=0.16)
        pol_plot.setLabel("left", "极化质量指标 (0-1)")
        pol_plot.setLabel("bottom", "迭代轮次")
        pol_plot.setYRange(0.0, 1.05, padding=0.02)
        lay.addWidget(pol_plot, stretch=1)
        rect_m = np.asarray([float(h.get("rectilinearity_mean", np.nan)) for h in history], dtype=float)
        dom_m = np.asarray([float(h.get("dominant_energy_ratio_mean", np.nan)) for h in history], dtype=float)
        lin_m = np.asarray([float(h.get("linearity_mean", np.nan)) for h in history], dtype=float)
        pol_plot.addLegend(offset=(8, 8))
        pol_plot.plot(x, rect_m, pen=pg.mkPen("#0f766e", width=2.0), symbol="o", symbolSize=6, name="矩形度均值")
        pol_plot.plot(x, dom_m, pen=pg.mkPen("#b45309", width=1.8), symbol="t", symbolSize=5, name="主导能量占比均值")
        pol_plot.plot(x, lin_m, pen=pg.mkPen("#1d4ed8", width=1.8), symbol="s", symbolSize=5, name="线性度均值")

        # 方向直观诊断：方位-倾角轨迹 + 相邻迭代方向变化角
        az_arr = np.asarray([float(h.get("azimuth_deg", np.nan)) for h in history], dtype=float)
        tilt_arr = np.asarray([float(h.get("tilt_deg", np.nan)) for h in history], dtype=float)
        dir_plot = pg.PlotWidget(background=self._theme_color("plot_bg", "#ffffff"))
        dir_plot.showGrid(x=True, y=True, alpha=0.16)
        dir_plot.setLabel("left", "倾角 (°)")
        dir_plot.setLabel("bottom", "方位角 (°)")
        dir_plot.addLegend(offset=(8, 8))
        dir_plot.plot(
            az_arr,
            tilt_arr,
            pen=pg.mkPen("#7c3aed", width=2.0),
            symbol="o",
            symbolSize=7,
            symbolBrush=pg.mkBrush("#7c3aed"),
            name="方位-倾角迭代轨迹",
        )
        lay.addWidget(dir_plot, stretch=1)

        def _dir_vec(az_deg: float, tilt_deg: float) -> np.ndarray:
            az = np.deg2rad(float(az_deg))
            tilt = np.deg2rad(float(tilt_deg))
            # 与校正模型一致：方位在水平面，倾角绕T轴混合R/Z。
            vx = np.cos(tilt) * np.cos(az)
            vy = np.cos(tilt) * np.sin(az)
            vz = np.sin(tilt)
            v = np.asarray([vx, vy, vz], dtype=float)
            n = float(np.linalg.norm(v))
            if not np.isfinite(n) or n <= 1e-12:
                return np.asarray([1.0, 0.0, 0.0], dtype=float)
            return v / n

        dir_change = np.full(len(history), np.nan, dtype=float)
        warn_deg = 5.0
        crit_deg = 10.0
        if len(history) >= 2:
            for i in range(1, len(history)):
                if not (
                    np.isfinite(az_arr[i - 1])
                    and np.isfinite(tilt_arr[i - 1])
                    and np.isfinite(az_arr[i])
                    and np.isfinite(tilt_arr[i])
                ):
                    continue
                v0 = _dir_vec(float(az_arr[i - 1]), float(tilt_arr[i - 1]))
                v1 = _dir_vec(float(az_arr[i]), float(tilt_arr[i]))
                c = float(np.clip(np.dot(v0, v1), -1.0, 1.0))
                dir_change[i] = float(np.rad2deg(np.arccos(c)))

        chg_plot = pg.PlotWidget(background=self._theme_color("plot_bg", "#ffffff"))
        chg_plot.showGrid(x=True, y=True, alpha=0.16)
        chg_plot.setLabel("left", "方向变化角 (°)")
        chg_plot.setLabel("bottom", "迭代轮次")
        chg_plot.addLegend(offset=(8, 8))
        chg_plot.addLine(y=warn_deg, pen=pg.mkPen("#f59e0b", width=1.2, style=QtCore.Qt.PenStyle.DashLine))
        chg_plot.addLine(y=crit_deg, pen=pg.mkPen("#ef4444", width=1.2, style=QtCore.Qt.PenStyle.DashLine))
        chg_plot.plot(
            x,
            dir_change,
            pen=pg.mkPen("#ea580c", width=2.0),
            name="相邻迭代方向变化角(曲线)",
        )
        idx_ok: List[int] = []
        idx_warn: List[int] = []
        idx_crit: List[int] = []
        for i, v in enumerate(dir_change):
            if not np.isfinite(v):
                continue
            if float(v) >= crit_deg:
                idx_crit.append(i)
            elif float(v) >= warn_deg:
                idx_warn.append(i)
            else:
                idx_ok.append(i)
        if idx_ok:
            chg_plot.plot(
                x[np.asarray(idx_ok, dtype=int)],
                dir_change[np.asarray(idx_ok, dtype=int)],
                pen=None,
                symbol="o",
                symbolSize=6,
                symbolBrush=pg.mkBrush("#10b981"),
                name=f"稳定 (<{warn_deg:.0f}°)",
            )
        if idx_warn:
            chg_plot.plot(
                x[np.asarray(idx_warn, dtype=int)],
                dir_change[np.asarray(idx_warn, dtype=int)],
                pen=None,
                symbol="o",
                symbolSize=7,
                symbolBrush=pg.mkBrush("#f59e0b"),
                name=f"警告 ({warn_deg:.0f}°~{crit_deg:.0f}°)",
            )
        if idx_crit:
            chg_plot.plot(
                x[np.asarray(idx_crit, dtype=int)],
                dir_change[np.asarray(idx_crit, dtype=int)],
                pen=None,
                symbol="o",
                symbolSize=8,
                symbolBrush=pg.mkBrush("#ef4444"),
                name=f"高风险 (≥{crit_deg:.0f}°)",
            )
        lay.addWidget(chg_plot, stretch=1)
        txt = QtWidgets.QPlainTextEdit(dlg)
        txt.setReadOnly(True)
        lines = [
            "参数说明：",
            "  J      = 总目标函数（越小越好）",
            "  J_tt   = 走时拟合误差项（越小越好）",
            "  J_pol  = 极化一致性误差项（越小越好）",
            "  J_sym  = 波形形态项（Z同相对称 + R反相对称 + 能量分布）误差（越小越好）",
            "  J_sym_shape = 纯形态对称项（Z同相、R反相）",
            "  J_energy = 能量项（R主导、T最小）",
            "  (归一) = 以初始尺度归一后参与加权，便于不同项在同一量级比较",
            "  az     = 方位角修正（度）",
            "  tilt   = 倾斜角修正（度）",
            "  dx/dy/dz = 位置修正量",
            "  矩形度/主导能量占比/线性度：极化质量指标，越接近1通常越好",
            "  相邻方向变化角：相邻两轮(az,tilt)对应三维方向夹角，越接近0说明越稳定",
            f"  稳定性分级：< {warn_deg:.0f}° 为稳定，{warn_deg:.0f}°~{crit_deg:.0f}° 为警告，≥ {crit_deg:.0f}° 为高风险",
            "",
            "迭代\t总目标J(归一)\tJ_tt(原始)\tJ_pol(原始)\tJ_sym(原始)\tJ_sym_shape\tJ_energy\tJ_tt(归一)\tJ_pol(归一)\tJ_sym(归一)\t矩形度\t主导占比\t线性度\t方位az(°)\t倾斜tilt(°)\t走时偏移dt(s)\t方向变化角(°)\t稳定性\tdx\tdy\tdz",
        ]
        for h in history:
            i0 = max(0, int(h.get("iter", 0)) - 1)
            dchg = float(dir_change[i0]) if i0 < dir_change.size else float("nan")
            if not np.isfinite(dchg):
                stability = "-"
            elif dchg >= crit_deg:
                stability = "高风险"
            elif dchg >= warn_deg:
                stability = "警告"
            else:
                stability = "稳定"
            lines.append(
                f"{int(h.get('iter', 0))}\t"
                f"{float(h.get('objective', np.nan)):.5f}\t"
                f"{float(h.get('J_tt', np.nan)):.5f}\t"
                f"{float(h.get('J_pol', np.nan)):.5f}\t"
                f"{float(h.get('J_sym', np.nan)):.5f}\t"
                f"{float(h.get('J_sym_shape', np.nan)):.5f}\t"
                f"{float(h.get('J_energy', np.nan)):.5f}\t"
                f"{float(h.get('J_tt_n', np.nan)):.5f}\t"
                f"{float(h.get('J_pol_n', np.nan)):.5f}\t"
                f"{float(h.get('J_sym_n', np.nan)):.5f}\t"
                f"{float(h.get('rectilinearity_mean', np.nan)):.4f}\t"
                f"{float(h.get('dominant_energy_ratio_mean', np.nan)):.4f}\t"
                f"{float(h.get('linearity_mean', np.nan)):.4f}\t"
                f"{float(h.get('azimuth_deg', np.nan)):.3f}\t"
                f"{float(h.get('tilt_deg', np.nan)):.3f}\t"
                f"{float(h.get('time_shift_sec', np.nan)):.3f}\t"
                f"{dchg:.3f}\t"
                f"{stability}\t"
                f"{float(h.get('dx', np.nan)):.3f}\t"
                f"{float(h.get('dy', np.nan)):.3f}\t"
                f"{float(h.get('dz', np.nan)):.3f}"
            )
        txt.setPlainText("\n".join(lines))
        lay.addWidget(txt, stretch=0)
        row = QtWidgets.QHBoxLayout()
        btn_accept = QtWidgets.QPushButton("接受为当前修正", dlg)
        btn_accept.clicked.connect(lambda: (self._accept_orientation_solution(result), dlg.close()))
        row.addWidget(btn_accept)
        row.addStretch(1)
        btn = QtWidgets.QPushButton("关闭", dlg)
        btn.clicked.connect(dlg.close)
        row.addWidget(btn)
        lay.addLayout(row)
        dlg.finished.connect(lambda _=0: self._set_orientation_main_preview(False, rebuild_if_needed=False, keep_cache=True))
        self._register_floating_dialog(dlg)
        dlg.show()

    def _show_orientation_corrected_visuals(
        self,
        observations: List[OrientationObservation],
        result,
    ) -> None:
        if not observations:
            return

        az = float(getattr(result, "azimuth_deg", 0.0))
        tilt = float(getattr(result, "tilt_deg", 0.0))
        pos = np.asarray(getattr(result, "position_correction", (0.0, 0.0, 0.0)), dtype=float)
        if pos.size != 3:
            pos = np.zeros(3, dtype=float)
        dt_shift = float(getattr(result, "details", {}).get("time_shift_sec", 0.0))

        def _len_to_km(v: float) -> float:
            d = abs(float(v))
            return d / 1000.0 if d > 50.0 else d

        def _rotate_components_local(r: np.ndarray, t: np.ndarray, z: np.ndarray, az_deg: float, tilt_deg: float):
            azr = np.deg2rad(float(az_deg))
            tr = np.deg2rad(float(tilt_deg))
            rr = np.cos(azr) * r + np.sin(azr) * t
            tt = -np.sin(azr) * r + np.cos(azr) * t
            r2 = np.cos(tr) * rr + np.sin(tr) * z
            z2 = -np.sin(tr) * rr + np.cos(tr) * z
            return r2, tt, z2

        pre_sec = float(self._orientation_ui_params.get("wave_pre", 0.30))
        post_sec = float(self._orientation_ui_params.get("wave_post", 0.70))
        disp_xy = np.asarray(pos[:2], dtype=float)

        conv_vals: List[float] = []
        dir_vecs: List[Optional[np.ndarray]] = []
        for o in observations:
            src_xy0 = np.asarray(o.source_xyz[:2], dtype=float)
            rec_xy0 = np.asarray(o.receiver_xyz[:2], dtype=float)
            v0 = rec_xy0 - src_xy0
            g0 = float(np.linalg.norm(v0))
            if np.isfinite(g0) and g0 > 1e-9:
                dir_vecs.append(np.asarray(v0, dtype=float))
                if np.isfinite(float(o.offset_km)) and abs(float(o.offset_km)) > 1e-9:
                    conv_vals.append(abs(float(o.offset_km)) / g0)
            else:
                dir_vecs.append(None)
        conv_km_per_coord = float(np.median(np.asarray(conv_vals, dtype=float))) if conv_vals else 0.0
        if not np.isfinite(conv_km_per_coord) or conv_km_per_coord <= 0.0:
            conv_km_per_coord = float(_len_to_km(np.linalg.norm(disp_xy))) / max(float(np.linalg.norm(disp_xy)), 1e-9)

        recs: List[Dict[str, object]] = []
        for iobs, o in enumerate(observations):
            n = int(min(len(o.z), len(o.r), len(o.t)))
            if n < 8:
                continue
            r2, t2, z2 = _rotate_components_local(
                np.asarray(o.r[:n], dtype=float),
                np.asarray(o.t[:n], dtype=float),
                np.asarray(o.z[:n], dtype=float),
                az,
                tilt,
            )
            off_km = float(o.offset_km)
            v0 = dir_vecs[iobs] if iobs < len(dir_vecs) else None
            if v0 is not None and np.isfinite(off_km):
                nv = float(np.linalg.norm(v0))
                if np.isfinite(nv) and nv > 1e-9:
                    u0 = np.asarray(v0, dtype=float) / nv
                    delta_km = float(np.dot(disp_xy, u0)) * float(conv_km_per_coord)
                    disp_km = float(_len_to_km(np.linalg.norm(disp_xy)))
                    delta_km = float(np.clip(delta_km, -disp_km, disp_km))
                    off_km = float(off_km + delta_km)
            tau_true = np.linspace(float(o.t0) - pre_sec, float(o.t0) + post_sec, n, dtype=float) - float(dt_shift)
            tau_fold = tau_true + float(self._compute_reduction_tshift(-1, float(off_km)))
            recs.append(
                {
                    "trace_idx": int(o.trace_idx),
                    "offset_km": float(off_km),
                    "z": np.asarray(z2, dtype=float),
                    "r": np.asarray(r2, dtype=float),
                    "t": np.asarray(t2, dtype=float),
                    "tau_true": np.asarray(tau_true, dtype=float),
                    "tau_fold": np.asarray(tau_fold, dtype=float),
                }
            )
        if not recs:
            return
        proc_params = self._build_processing_params()
        rec_offsets = np.asarray([float(r["offset_km"]) for r in recs], dtype=float)
        def _proc_comp(key: str) -> None:
            comp_traces = [np.asarray(r[key], dtype=float) for r in recs]
            nmin = int(min(len(t) for t in comp_traces))
            if nmin < 8:
                return
            times_local = np.asarray(recs[0]["tau_true"], dtype=float)[:nmin]
            comp_traces = [np.asarray(t[:nmin], dtype=float) for t in comp_traces]
            if times_local.size > 1:
                sr = 1.0 / float(times_local[1] - times_local[0])
            else:
                sr = None
            try:
                comp_proc = self.processor.process_traces(
                    traces=comp_traces,
                    times=times_local,
                    offsets=rec_offsets,
                    params=proc_params,
                    gains=np.ones(len(comp_traces), dtype=float),
                    sampling_rate=sr,
                    realtime_interaction=False,
                )
                for i, arr in enumerate(comp_proc):
                    recs[i][key] = np.asarray(arr, dtype=float)
            except Exception:
                pass
        _proc_comp("z")
        _proc_comp("r")
        _proc_comp("t")
        recs.sort(key=lambda d: float(d.get("offset_km", 0.0)))

        dlg = QtWidgets.QDialog(None)
        dlg.setWindowTitle("姿态校正后结果可视化")
        dlg.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        dlg.resize(1100, 760)
        lay = QtWidgets.QVBoxLayout(dlg)
        tab = QtWidgets.QTabWidget(dlg)
        lay.addWidget(tab)

        # Tab1: corrected 3C V-window waveforms with updated offsets and folded times.
        wave_page = QtWidgets.QWidget(dlg)
        wave_lay = QtWidgets.QVBoxLayout(wave_page)
        wave_info = QtWidgets.QLabel(
            f"显示校正后V截窗三分量：偏移距使用(dx,dy)更新；折合走时使用更新后偏移距与当前vred；真实走时应用dt={dt_shift:.3f}s。",
            wave_page,
        )
        wave_info.setWordWrap(True)
        wave_info.setStyleSheet("color:#475569;")
        wave_lay.addWidget(wave_info)
        comp_labels = [("z", "垂直 Z"), ("r", "径向 R"), ("t", "切向 T")]
        all_max = {
            k: max(1e-9, max(float(np.max(np.abs(np.asarray(r[k], dtype=float)))) for r in recs))
            for k, _ in comp_labels
        }
        wig_scale = 0.06 * max(1e-6, (float(recs[-1]["offset_km"]) - float(recs[0]["offset_km"]) + 1e-6))
        for k, cname in comp_labels:
            pw = pg.PlotWidget(background=self._theme_color("plot_bg", "#ffffff"))
            pw.showGrid(x=True, y=True, alpha=0.14)
            pw.setLabel("left", "折合走时 (s)")
            pw.setLabel("bottom", "更新后偏移距 (km)")
            pw.invertY(True)
            color = {"z": "#1d4ed8", "r": "#b45309", "t": "#6d28d9"}.get(k, "#334155")
            for recd in recs:
                x0 = float(recd["offset_km"])
                y = np.asarray(recd["tau_fold"], dtype=float)
                a = np.asarray(recd[k], dtype=float) / float(all_max[k])
                x = x0 + wig_scale * a
                pw.plot(x, y, pen=pg.mkPen(color, width=1.1))
            pw.setTitle(f"{cname}（按更新后偏移距/折合走时）")
            wave_lay.addWidget(pw, stretch=1)
        tab.addTab(wave_page, "校正后三分量波形")

        # Tab2: final polarization particle-motion + principal direction.
        pol_page = QtWidgets.QWidget(dlg)
        pol_lay = QtWidgets.QVBoxLayout(pol_page)
        pol_info = QtWidgets.QLabel("最终极化图：基于校正后三分量叠加波形（R/T/Z）的质点轨迹与主极化方向。", pol_page)
        pol_info.setWordWrap(True)
        pol_info.setStyleSheet("color:#475569;")
        pol_lay.addWidget(pol_info)

        nmin_after = int(min(len(np.asarray(r["r"], dtype=float)) for r in recs))
        r_stack_after = np.mean(np.asarray([np.asarray(r["r"], dtype=float)[:nmin_after] for r in recs], dtype=float), axis=0)
        t_stack_after = np.mean(np.asarray([np.asarray(r["t"], dtype=float)[:nmin_after] for r in recs], dtype=float), axis=0)
        z_stack_after = np.mean(np.asarray([np.asarray(r["z"], dtype=float)[:nmin_after] for r in recs], dtype=float), axis=0)
        feat_after = extract_polarization_features(z_stack_after, r_stack_after, t_stack_after)
        pv_after = np.asarray(feat_after.principal_vector, dtype=float)  # [R, T, Z]

        nmin_before = int(min(len(np.asarray(o.r, dtype=float)) for o in observations))
        r_stack_before = np.mean(np.asarray([np.asarray(o.r, dtype=float)[:nmin_before] for o in observations], dtype=float), axis=0)
        t_stack_before = np.mean(np.asarray([np.asarray(o.t, dtype=float)[:nmin_before] for o in observations], dtype=float), axis=0)
        z_stack_before = np.mean(np.asarray([np.asarray(o.z, dtype=float)[:nmin_before] for o in observations], dtype=float), axis=0)
        feat_before = extract_polarization_features(z_stack_before, r_stack_before, t_stack_before)
        pv_before = np.asarray(feat_before.principal_vector, dtype=float)  # [R, T, Z]

        grid = QtWidgets.QGridLayout()
        pol_lay.addLayout(grid, stretch=1)

        def _pm_plot(
            xa_before: np.ndarray,
            ya_before: np.ndarray,
            vx_before: float,
            vy_before: float,
            xa_after: np.ndarray,
            ya_after: np.ndarray,
            vx_after: float,
            vy_after: float,
            xlabel: str,
            ylabel: str,
            title: str,
            row: int,
            col: int,
        ) -> None:
            pw = pg.PlotWidget(background=self._theme_color("plot_bg", "#ffffff"))
            pw.showGrid(x=True, y=True, alpha=0.16)
            pw.setLabel("left", ylabel)
            pw.setLabel("bottom", xlabel)
            pw.setTitle(title)
            pw.addLegend(offset=(8, 8))
            pw.plot(xa_before, ya_before, pen=pg.mkPen("#94a3b8", width=1.3), name="校正前轨迹")
            pw.plot(xa_after, ya_after, pen=pg.mkPen("#0f766e", width=1.6), name="校正后轨迹")
            rr = float(
                max(
                    np.max(np.abs(xa_before)),
                    np.max(np.abs(ya_before)),
                    np.max(np.abs(xa_after)),
                    np.max(np.abs(ya_after)),
                    1e-6,
                )
            )
            line_x_b = np.asarray([-rr, rr], dtype=float) * float(vx_before)
            line_y_b = np.asarray([-rr, rr], dtype=float) * float(vy_before)
            pw.plot(
                line_x_b,
                line_y_b,
                pen=pg.mkPen("#64748b", width=1.6, style=QtCore.Qt.PenStyle.DashLine),
                name="校正前主方向",
            )
            line_x_a = np.asarray([-rr, rr], dtype=float) * float(vx_after)
            line_y_a = np.asarray([-rr, rr], dtype=float) * float(vy_after)
            pw.plot(
                line_x_a,
                line_y_a,
                pen=pg.mkPen("#ef4444", width=2.0, style=QtCore.Qt.PenStyle.DashLine),
                name="校正后主方向",
            )
            grid.addWidget(pw, row, col)

        _pm_plot(
            r_stack_before, z_stack_before, float(pv_before[0]), float(pv_before[2]),
            r_stack_after, z_stack_after, float(pv_after[0]), float(pv_after[2]),
            "R", "Z", "质点运动 R-Z（前后对比）", 0, 0
        )
        _pm_plot(
            r_stack_before, t_stack_before, float(pv_before[0]), float(pv_before[1]),
            r_stack_after, t_stack_after, float(pv_after[0]), float(pv_after[1]),
            "R", "T", "质点运动 R-T（前后对比）", 0, 1
        )
        _pm_plot(
            t_stack_before, z_stack_before, float(pv_before[1]), float(pv_before[2]),
            t_stack_after, z_stack_after, float(pv_after[1]), float(pv_after[2]),
            "T", "Z", "质点运动 T-Z（前后对比）", 1, 0
        )

        summary = QtWidgets.QPlainTextEdit(pol_page)
        summary.setReadOnly(True)
        summary.setPlainText(
            "\n".join(
                [
                    "最终极化指标（叠加波形）",
                    f"校正前主方向 [R,T,Z] = ({float(pv_before[0]):.4f}, {float(pv_before[1]):.4f}, {float(pv_before[2]):.4f})",
                    f"校正后主方向 [R,T,Z] = ({float(pv_after[0]):.4f}, {float(pv_after[1]):.4f}, {float(pv_after[2]):.4f})",
                    f"校正前: 线性度={float(feat_before.linearity):.4f}, 矩形度={float(feat_before.rectilinearity):.4f}, 主导占比={float(feat_before.dominant_energy_ratio):.4f}",
                    f"校正后: 线性度={float(feat_after.linearity):.4f}, 矩形度={float(feat_after.rectilinearity):.4f}, 主导占比={float(feat_after.dominant_energy_ratio):.4f}",
                ]
            )
        )
        grid.addWidget(summary, 1, 1)
        tab.addTab(pol_page, "最终极化图")

        preview_solution = {
            "azimuth_deg": float(az),
            "tilt_deg": float(tilt),
            "dx": float(pos[0]),
            "dy": float(pos[1]),
            "dz": float(pos[2]),
            "time_shift_sec": float(dt_shift),
        }

        row = QtWidgets.QHBoxLayout()
        btn_apply = QtWidgets.QPushButton("应用到主图预览（全道）", dlg)
        btn_clear = QtWidgets.QPushButton("清除主图预览", dlg)
        def _apply_preview():
            self._set_orientation_main_preview(True, solution=preview_solution)
            self.lbl_status.setText(
                "已应用姿态校正主图预览：全道使用新方位/倾角/位置与走时偏移"
            )
        def _clear_preview():
            self._set_orientation_main_preview(False)
            self.lbl_status.setText("已清除姿态校正主图预览")
        btn_apply.clicked.connect(_apply_preview)
        btn_clear.clicked.connect(_clear_preview)
        row.addWidget(btn_apply)
        row.addWidget(btn_clear)
        row.addStretch(1)
        btn = QtWidgets.QPushButton("关闭", dlg)
        btn.clicked.connect(dlg.close)
        row.addWidget(btn)
        lay.addLayout(row)
        self._register_floating_dialog(dlg)
        dlg.show()

    def _run_attitude_correction_placeholder(self) -> None:
        self._open_attitude_correction_dialog()

    def _preview_orientation_input_from_current_params(self) -> None:
        observations, err = self._build_orientation_observations()
        if observations is None:
            self._show_themed_info("姿态参数预览", f"无法生成预览：{err}")
            return
        depth_sampler = self._terrain_depth_sampler_from_observations(observations)
        depth0 = self._sample_initial_depth_km(observations, depth_sampler) if depth_sampler is not None else None
        self._show_orientation_input_preview(observations, depth0)

    def _run_attitude_correction_with_current_params(
        self,
        parent_dialog: Optional[QtWidgets.QDialog] = None,
        progress_bar: Optional[QtWidgets.QProgressBar] = None,
        progress_label: Optional[QtWidgets.QLabel] = None,
    ) -> None:
        observations, err = self._build_orientation_observations()
        if observations is None:
            self.lbl_status.setText(f"姿态校正失败：{err}")
            self._show_themed_info("姿态校正", f"姿态校正失败：{err}")
            return

        depth_sampler = self._terrain_depth_sampler_from_observations(observations)
        depth0 = self._sample_initial_depth_km(observations, depth_sampler) if depth_sampler is not None else None
        if depth_sampler is None:
            msg = "未检测到可用水深采样（请先在姿态校正窗口加载水深文件），无法执行姿态校正。"
            self.lbl_status.setText(f"姿态校正失败：{msg}")
            self._show_themed_info("姿态校正失败", msg)
            return
        if depth0 is None or not np.isfinite(depth0) or depth0 <= 0.0:
            msg = "当前地形无法采样有效水深，无法执行姿态校正。"
            self.lbl_status.setText(f"姿态校正失败：{msg}")
            self._show_themed_info("姿态校正失败", msg)
            return

        max_iters = max(1, int(round(float(self._orientation_ui_params.get("att_iter", 4.0)))))
        if progress_bar is not None:
            progress_bar.setRange(0, max_iters)
            progress_bar.setValue(0)
        if progress_label is not None:
            progress_label.setText(f"校正进度：0/{max_iters}（准备开始）")
        QtWidgets.QApplication.processEvents()

        def _on_progress(cur_iter: int, total_iter: int, stage: str) -> None:
            total_safe = max(1, int(total_iter))
            cur_safe = max(0, min(int(cur_iter), total_safe))
            if progress_bar is not None:
                progress_bar.setMaximum(total_safe)
                progress_bar.setValue(cur_safe)
            if progress_label is not None:
                progress_label.setText(f"校正进度：{cur_safe}/{total_safe}，{stage}")
            QtWidgets.QApplication.processEvents()

        try:
            result = run_orientation_correction(
                OrientationCorrectionInput(
                    observations=observations,
                    initial_azimuth_deg=float(self._orientation_current_solution.get("azimuth_deg", 0.0)),
                    initial_tilt_deg=float(self._orientation_current_solution.get("tilt_deg", 0.0)),
                    initial_position_correction=(
                        float(self._orientation_current_solution.get("dx", 0.0)),
                        float(self._orientation_current_solution.get("dy", 0.0)),
                        float(self._orientation_current_solution.get("dz", 0.0)),
                    ),
                    depth_sampler=depth_sampler,
                    max_iterations=max_iters,
                    w_tt=max(0.0, float(self._orientation_ui_params.get("att_wtt", 1.0))),
                    w_pol=max(0.0, float(self._orientation_ui_params.get("att_wpol", 1.0))),
                    w_sym=max(0.0, float(self._orientation_ui_params.get("att_wsym", 0.8))),
                    progress_callback=_on_progress,
                )
            )
        finally:
            if progress_bar is not None:
                progress_bar.setValue(max_iters)
            QtWidgets.QApplication.processEvents()
        if not result.success:
            if progress_label is not None:
                progress_label.setText("校正进度：失败")
            self.lbl_status.setText(f"姿态校正失败：{result.message}")
            self._show_themed_info("姿态校正失败", result.message)
            return

        dx, dy, dz = result.position_correction
        j_tt = float(result.details.get("J_tt", float("nan")))
        j_pol = float(result.details.get("J_pol", float("nan")))
        j_sym = float(result.details.get("J_sym", float("nan")))
        t_shift = float(result.details.get("time_shift_sec", 0.0))
        depth_info = "未使用地形采样"
        if result.source_depth_history:
            depth_info = (
                f"水深采样轮次={len(result.source_depth_history)}，"
                f"范围[{min(result.source_depth_history):.3f}, {max(result.source_depth_history):.3f}]"
            )

        msg = (
            f"方位修正: {result.azimuth_deg:.2f}°\n"
            f"倾斜修正: {result.tilt_deg:.2f}°\n"
            f"位置修正: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}\n"
            f"走时全局偏移: dt={t_shift:.3f} s（约束在 ±1.0 s）\n"
            f"目标函数: J={result.objective:.4f} (Jtt={j_tt:.4f}, Jpol={j_pol:.4f}, Jsym={j_sym:.4f})\n"
            f"{depth_info}"
        )
        self._show_themed_info("姿态校正结果", msg)
        self._show_orientation_diagnostics(result)
        self._show_orientation_corrected_visuals(observations, result)
        if progress_label is not None:
            progress_label.setText(f"校正进度：完成（{max_iters}/{max_iters}）")
        self.lbl_status.setText(
            f"姿态校正完成：az={result.azimuth_deg:.2f}°, tilt={result.tilt_deg:.2f}°, "
            f"dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}"
        )

    def _run_pick_alignment(self) -> None:
        if self.pick_manager is None:
            return
        # 与旧版交互一致：再次按 A 直接清除当前对齐
        if self._alignment_offsets:
            self._alignment_offsets = {}
            self.request_render(delay_ms=10)
            self.lbl_status.setText("已清除波形临时对齐（A 再次按下）")
            return
        pick_word = int(self.spin_apick.value())
        picks = self.pick_manager.get_picks_by_word(pick_word)
        if len(picks) < 2:
            self.lbl_status.setText("波形临时对齐失败：当前拾取字至少需要2个拾取点")
            return
        offsets_all = np.asarray(self.loaded.get("offsets", []), dtype=float) if self.loaded is not None else np.array([])
        disp_times: Dict[int, float] = {}
        for trace_idx, tpk in picks.items():
            ig = int(trace_idx)
            if ig < 0 or ig >= offsets_all.size:
                continue
            base_tshift = self._compute_display_tshift(ig, float(offsets_all[ig]))
            disp_times[ig] = float(tpk) + base_tshift
        if len(disp_times) < 2:
            self.lbl_status.setText("波形临时对齐失败：有效显示拾取不足")
            return
        ref_time = float(np.median(np.asarray(list(disp_times.values()), dtype=float)))
        self._alignment_offsets = {
            int(trace_idx): (ref_time - disp_t)
            for trace_idx, disp_t in disp_times.items()
        }
        self.request_render(delay_ms=10)
        self.lbl_status.setText(f"波形临时对齐完成：{len(self._alignment_offsets)} 道（按当前显示时间对齐）")

    def _run_adaptive_alignment(self) -> None:
        if self.loaded is None or self.pick_manager is None:
            return
        traces = self.loaded.get("traces", [])
        times = np.asarray(self.loaded.get("times", []), dtype=float)
        offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
        if len(traces) == 0 or times.size < 2 or offsets.size == 0:
            return
        idx = self._extract_indices()
        if idx.size == 0:
            self.lbl_status.setText("自适应拾取更新失败：当前过滤后无道")
            return
        xcoords = offsets[idx]
        idx_vis = idx[self._visible_mask(xcoords)]
        if idx_vis.size < 2:
            self.lbl_status.setText("自适应拾取更新失败：视窗内道数不足")
            return

        pick_word = int(self.spin_apick.value())
        by_word = self.pick_manager.get_picks_by_word(pick_word)
        # 对齐使用“折合后的显示时间域”：
        # 1) 先用当前处理参数得到用于相关的波形（含滤波）
        # 2) 按 (rvred-rvredf) 把每道重采样到统一显示时间网格
        raw_selected_traces = [np.asarray(traces[int(i)], dtype=float) for i in idx_vis]
        proc_params = self._build_processing_params()
        processed_traces = self.processor.process_traces(
            raw_selected_traces,
            times,
            offsets[idx_vis],
            proc_params,
            realtime_interaction=False,
        )
        selected_traces: List[np.ndarray] = []
        reduction_shifts: List[float] = []
        for li, gidx in enumerate(idx_vis):
            red_shift = float(self._compute_reduction_tshift(int(gidx), float(offsets[int(gidx)])))
            reduction_shifts.append(red_shift)
            tr = np.asarray(processed_traces[li], dtype=float)
            # 显示域 td = t + tshift => A(td)=A_true(td - tshift)
            # 这里将每道映射到统一 times 网格，避免 vred 下“看着对齐、算法却在真时域错配”。
            tr_disp = np.interp(
                times - red_shift,
                times,
                tr,
                left=0.0,
                right=0.0,
            )
            selected_traces.append(tr_disp)
        initial_picks: List[int] = []
        t0 = float(times[0])
        dt = float(times[1] - times[0])
        for li, gidx in enumerate(idx_vis):
            pt = by_word.get(int(gidx))
            if pt is None:
                initial_picks.append(-1)
            else:
                pt_display = float(pt) + float(reduction_shifts[li])
                initial_picks.append(int(round((pt_display - t0) / dt)))
        corr_before: Optional[float] = None
        picks_before_display: Dict[int, float] = {}
        for li, gidx in enumerate(idx_vis):
            pt = by_word.get(int(gidx))
            if pt is None:
                continue
            ptf = float(pt)
            if ptf <= 0.0:
                continue
            picks_before_display[int(li)] = ptf + float(reduction_shifts[li])
        if len(picks_before_display) >= 2:
            try:
                corr_info_before = self.adaptive_stacker.calculate_correlation(
                    traces=selected_traces,
                    times=times,
                    picks=picks_before_display,
                )
                corr_before = float(corr_info_before.get("mean_correlation", 0.0))
            except Exception:
                corr_before = None

        try:
            result = self.adaptive_stacker.align_traces(
                traces=selected_traces,
                times=times,
                initial_picks=initial_picks,
            )
        except Exception as exc:
            self.last_stacking_result = None
            self.lbl_status.setText(f"自适应拾取更新失败: {exc}")
            return
        shifts = result.get("time_shifts", [])
        if not shifts:
            self.last_stacking_result = None
            self.lbl_status.setText("自适应拾取更新失败：未返回有效偏移")
            return
        n_apply = 0
        original_picks: Dict[int, float] = {}
        updated_picks: Dict[int, float] = {}
        applied_shifts: List[float] = []
        applied_shifts_by_trace: Dict[int, float] = {}
        errors_by_trace: Dict[int, float] = {}
        corr_after: Optional[float] = None
        result_errors = list(result.get("errors", []))
        for li, gidx in enumerate(idx_vis):
            if li < len(shifts) and initial_picks[li] >= 0:
                orig = float(by_word.get(int(gidx), 0.0))
                if orig > 0:
                    if n_apply == 0:
                        self._push_pick_undo("自适应拾取更新")
                    original_picks[int(gidx)] = orig
                    # shifts 定义在“折合显示时间域”，对同一道回写到真时拾取时可直接加到原拾取。
                    new_pick = orig + float(shifts[li])
                    new_pick = float(np.clip(new_pick, t0, float(times[-1])))
                    # F 键仅更新拾取时间，不改变波形时间
                    self.pick_manager.add_pick(int(gidx), new_pick, pick_word)
                    updated_picks[int(gidx)] = new_pick
                    applied_shifts.append(float(shifts[li]))
                    applied_shifts_by_trace[int(gidx)] = float(shifts[li])
                    if li < len(result_errors):
                        errors_by_trace[int(gidx)] = float(result_errors[li])
                n_apply += 1
        picks_after_display: Dict[int, float] = {}
        for li, gidx in enumerate(idx_vis):
            orig_pt = by_word.get(int(gidx))
            if orig_pt is None:
                continue
            updated_pt = float(updated_picks.get(int(gidx), float(orig_pt)))
            if updated_pt <= 0.0:
                continue
            picks_after_display[int(li)] = updated_pt + float(reduction_shifts[li])
        if len(picks_after_display) >= 2:
            try:
                corr_info_after = self.adaptive_stacker.calculate_correlation(
                    traces=selected_traces,
                    times=times,
                    picks=picks_after_display,
                )
                corr_after = float(corr_info_after.get("mean_correlation", 0.0))
            except Exception:
                corr_after = None
        # 显式保持为空，避免 F 对波形时间产生任何影响
        self._alignment_offsets = {}
        self.last_stacking_result = {
            "original_picks": original_picks,
            "updated_picks": updated_picks,
            "time_shifts": applied_shifts,
            "errors": result_errors,
            "time_shifts_by_trace": applied_shifts_by_trace,
            "errors_by_trace": errors_by_trace,
            "quality_metric": float(result.get("quality_metric", 0.0) or 0.0),
            "mean_corr_before": corr_before,
            "mean_corr_after": corr_after,
            "offsets": offsets,
            "traces": selected_traces,
            "times": times,
        }
        self.request_render(delay_ms=10)
        msg = f"自适应拾取更新完成：更新 {n_apply} 道拾取（波形时间不变）"
        if corr_before is not None and corr_after is not None:
            delta_corr = corr_after - corr_before
            msg += f" | 一致性 mean corr: {corr_before:.3f}->{corr_after:.3f} (Δ={delta_corr:+.3f})"
        self._set_status_text(msg, hold_ms=5000)

    def _clear_alignment(self) -> None:
        self._alignment_offsets = {}
        self.request_render(delay_ms=10)
        self.lbl_status.setText("已清除波形临时对齐偏移")

    def _calculate_static_correction_dialog(self) -> None:
        if self.loaded is None or self.pick_manager is None:
            self.lbl_status.setText("提示：请先加载数据并拾取")
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("计算静校正")
        dialog.resize(420, 180)
        layout = QtWidgets.QFormLayout(dialog)
        sigma_spin = QtWidgets.QDoubleSpinBox(dialog)
        sigma_spin.setRange(0.1, 100.0)
        sigma_spin.setDecimals(2)
        sigma_spin.setValue(3.0)
        smooth_spin = QtWidgets.QDoubleSpinBox(dialog)
        smooth_spin.setRange(0.0, 1.0)
        smooth_spin.setDecimals(3)
        smooth_spin.setSingleStep(0.01)
        smooth_spin.setValue(0.1)
        layout.addRow("sigma (km)", sigma_spin)
        layout.addRow("smoothness", smooth_spin)
        info = QtWidgets.QLabel("基于当前拾取字提取短波长静校正（需至少3个有效拾取）。", dialog)
        info.setWordWrap(True)
        layout.addRow(info)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        self._calculate_static_correction(
            sigma=float(sigma_spin.value()),
            smoothness=float(smooth_spin.value()),
        )

    def _calculate_static_correction(self, sigma: float, smoothness: float) -> None:
        if self.loaded is None or self.pick_manager is None:
            return
        pick_word = int(self.spin_apick.value())
        all_picks = self.pick_manager.get_all_picks()
        if not all_picks:
            self.lbl_status.setText("静校正失败：无拾取数据")
            return
        idx = self._extract_indices()
        if idx.size < 3:
            self.lbl_status.setText("静校正失败：当前过滤后道数不足")
            return
        by_word = self.pick_manager.get_picks_by_word(pick_word)
        picked_idx = np.array(
            [int(i) for i in idx if float(by_word.get(int(i), 0.0) or 0.0) > 0.0],
            dtype=int,
        )
        if picked_idx.size < 3:
            self.lbl_status.setText("静校正失败：当前拾取字有效拾取道不足（至少3道）")
            return
        offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
        x_coords = offsets[picked_idx] if offsets.size else np.arange(picked_idx.size, dtype=float)
        trace_indices = [int(i) for i in picked_idx]
        # 在折合显示模式下，静校正应基于当前显示时间（t' = t - |x|/vred + |x|/vredf）提取。
        display_times = self._build_reduced_pick_times(trace_indices, pick_word)
        corrections = self.static_corrector.extract_short_wavelength_gaussian(
            picks=all_picks,
            trace_indices=trace_indices,
            x_coords=np.asarray(x_coords, dtype=float),
            pick_word=pick_word,
            sigma=float(sigma),
            min_picks=3,
            display_times=display_times if display_times else None,
            smoothness=float(smoothness),
        )
        if not corrections:
            self.lbl_status.setText("静校正失败：有效拾取不足或计算失败")
            return
        # 严格模式：仅对当前拾取字有有效拾取的道生效，不外推到其它道。
        strict_trace_set = set(trace_indices)
        corrections = {int(k): float(v) for k, v in corrections.items() if int(k) in strict_trace_set}
        if not corrections:
            self.lbl_status.setText("静校正失败：未生成可应用的校正道")
            return
        self.static_corrector.set_corrections(corrections)
        vals = np.asarray(list(corrections.values()), dtype=float)
        min_corr = float(np.min(vals))
        max_corr = float(np.max(vals))
        mean_corr = float(np.mean(vals))

        # 先进入预览态并立刻绘制拟合曲线，供用户评估后再决定是否应用
        self.static_correction_enabled = False
        self.static_preview_mode = True
        self.request_render(immediate=True)

        self._show_static_correction_decision(
            correction_count=len(corrections),
            min_corr=min_corr,
            max_corr=max_corr,
            mean_corr=mean_corr,
        )

    def _clear_static_correction(self) -> None:
        if self._static_decision_box is not None:
            try:
                self._static_decision_box.close()
            except Exception:
                pass
            self._static_decision_box = None
        self.static_corrector.clear_corrections()
        self.static_correction_enabled = False
        self.static_preview_mode = False
        self._clear_static_preview_item()
        self.request_render(delay_ms=10)
        self.lbl_status.setText("已清除静校正")

    def _show_static_correction_decision(
        self,
        correction_count: int,
        min_corr: float,
        max_corr: float,
        mean_corr: float,
    ) -> None:
        if self._static_decision_box is not None:
            try:
                self._static_decision_box.close()
            except Exception:
                pass
            self._static_decision_box = None

        msg = QtWidgets.QMessageBox(self)
        msg.setIcon(QtWidgets.QMessageBox.Icon.Question)
        msg.setWindowTitle("静校正拟合预览")
        msg.setText(
            f"静校正已计算完成：{correction_count} 道\n"
            f"范围 [{min_corr:.4f}, {max_corr:.4f}] s\n"
            f"平均 {mean_corr:.4f} s\n\n"
            "拟合曲线（虚线预览）已显示。\n"
            "可在主窗口缩放/平移检查后再决定是否应用。"
        )
        btn_apply = msg.addButton("应用静校正", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        btn_preview = msg.addButton("仅保留预览", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        btn_cancel = msg.addButton("取消并清除", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(btn_apply)
        msg.setModal(False)
        msg.setWindowModality(QtCore.Qt.WindowModality.NonModal)

        def _on_clicked(button: QtWidgets.QAbstractButton) -> None:
            self._static_decision_box = None
            if button == btn_cancel:
                self.static_corrector.clear_corrections()
                self.static_correction_enabled = False
                self.static_preview_mode = False
                self._clear_static_preview_item()
                self.lbl_status.setText("已取消静校正")
                self.request_render(delay_ms=10)
                return
            if button == btn_apply:
                self.static_correction_enabled = True
                self.static_preview_mode = False
                self.lbl_status.setText(
                    f"静校正已应用：{correction_count}道, 范围[{min_corr:.4f},{max_corr:.4f}]s"
                )
            else:
                self.static_correction_enabled = False
                self.static_preview_mode = True
                self.lbl_status.setText(
                    f"静校正预览：{correction_count}道, 范围[{min_corr:.4f},{max_corr:.4f}]s（虚线）"
                )
            self.request_render(delay_ms=10)

        msg.buttonClicked.connect(_on_clicked)
        self._static_decision_box = msg
        msg.show()
        QtCore.QTimer.singleShot(0, lambda: self._position_static_decision_box(msg))

    def _position_static_decision_box(self, box: QtWidgets.QWidget) -> None:
        """将静校正决策框放置到主窗口右上角，并限制在屏幕可视区内。"""
        try:
            margin = 12
            box.adjustSize()
            parent_geo = self.frameGeometry()
            box_geo = box.frameGeometry()

            x = int(parent_geo.right() - box_geo.width() - margin)
            y = int(parent_geo.top() + margin)

            screen = self.screen() or QtWidgets.QApplication.primaryScreen()
            if screen is not None:
                avail = screen.availableGeometry()
                min_x = int(avail.left() + margin)
                max_x = int(avail.right() - box_geo.width() - margin)
                min_y = int(avail.top() + margin)
                max_y = int(avail.bottom() - box_geo.height() - margin)
                if max_x < min_x:
                    max_x = min_x
                if max_y < min_y:
                    max_y = min_y
                x = max(min_x, min(x, max_x))
                y = max(min_y, min(y, max_y))

            box.move(x, y)
        except Exception:
            # 定位失败不影响主流程
            pass

    def _show_stacking_evaluation(self) -> None:
        if not self.last_stacking_result:
            self.lbl_status.setText("提示：请先执行 F（自适应拾取更新）")
            return
        data = self.last_stacking_result
        original_picks = data.get("original_picks", {})
        updated_picks = data.get("updated_picks", {})
        time_shifts = data.get("time_shifts_by_trace", data.get("time_shifts", []))
        errors = data.get("errors_by_trace", data.get("errors", []))
        traces = data.get("traces", [])
        times = data.get("times", np.array([]))
        offsets = np.asarray(data.get("offsets", []), dtype=float)
        if not isinstance(original_picks, dict) or not isinstance(updated_picks, dict):
            self.lbl_status.setText("叠加评价失败：结果数据无效")
            return
        if len(updated_picks) == 0:
            self.lbl_status.setText("叠加评价失败：无有效更新拾取")
            return
        try:
            eval_result = self.stacking_evaluator.evaluate(
                original_picks=original_picks,
                updated_picks=updated_picks,
                time_shifts=time_shifts,
                errors=errors,
                quality_metric=float(data.get("quality_metric", 0.0) or 0.0),
                traces=list(traces),
                times=np.asarray(times, dtype=float),
            )
            trace_indices = sorted(updated_picks.keys())
            self.stacking_evaluator.create_comparison_plot(eval_result, trace_indices=trace_indices)
            self.stacking_evaluator.create_shift_visualization(eval_result, offsets, trace_indices=trace_indices)
            try:
                import matplotlib.pyplot as plt
                plt.show()
            except Exception as exc:
                self.lbl_status.setText(f"叠加评价图显示失败: {exc}")
                return
            self.lbl_status.setText("叠加评价可视化已显示")
        except Exception as exc:
            self.lbl_status.setText(f"叠加评价失败: {exc}")

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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_action_bar_overflow()

    def _run_auto_pick(self) -> None:
        if self.loaded is None or self.pick_manager is None:
            return
        traces = self.loaded.get("traces", [])
        offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
        times = np.asarray(self.loaded.get("times", []), dtype=float)
        if len(traces) == 0 or times.size == 0:
            return
        idx = self._extract_indices()
        if idx.size == 0:
            return
        pick_word = int(self.spin_apick.value())
        selected_traces = [np.asarray(traces[int(i)]) for i in idx]
        selected_offsets = offsets[idx] if offsets.size else np.zeros(idx.size, dtype=float)
        results = self.auto_picker.pick_traces(
            traces=selected_traces,
            times=times,
            offsets=selected_offsets,
        )
        added = 0
        undo_pushed = False
        for local_i, res in enumerate(results):
            if not res:
                continue
            t = float(res.get("pick_time", 0.0))
            if t > 0:
                gidx = int(idx[local_i])
                old_pick = self.pick_manager.get_pick(gidx, pick_word)
                if old_pick is not None and abs(float(old_pick) - t) < 1e-9:
                    continue
                if not undo_pushed:
                    self._push_pick_undo("自动拾取")
                    undo_pushed = True
                if self.pick_manager.add_pick(gidx, t, pick_word):
                    added += 1
        self.request_render(delay_ms=10)
        self.lbl_status.setText(f"自动拾取完成：新增 {added} 个拾取")

    def _run_interp_pick(self) -> None:
        if self.loaded is None or self.pick_manager is None:
            return
        self.params.tcrcor = float(self.spin_tcrcor.value())
        self.params.tlag = float(self.spin_tlag.value())
        self.params.hilbratio = float(self.spin_hilbratio.value())
        force_pick = False
        traces = self.loaded.get("traces", [])
        offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
        times = np.asarray(self.loaded.get("times", []), dtype=float)
        if len(traces) == 0 or times.size == 0 or offsets.size == 0:
            return
        if times.size < 2:
            self.lbl_status.setText("插值相关拾取失败：时间采样点不足")
            return
        # 仅在当前过滤集合（含当前分量）内执行，避免跨分量混拾取
        idx = self._extract_indices()
        if idx.size < 3:
            self.lbl_status.setText("插值相关拾取失败：当前分量/过滤后道数不足")
            return
        idx_local_to_global = np.asarray(idx, dtype=int)
        allowed_set = set(int(i) for i in idx_local_to_global)
        global_to_local = {int(g): li for li, g in enumerate(idx_local_to_global.tolist())}
        pick_word = int(self.spin_apick.value())
        pw_picks: Dict[int, float] = self.pick_manager.get_picks_by_word(pick_word)
        pw_picks = {int(k): float(v) for k, v in pw_picks.items() if int(k) in allowed_set}
        if len(pw_picks) < 2:
            self.lbl_status.setText("插值相关拾取需要当前拾取字至少2个种子点")
            return
        # 与显示链一致：在处理后道上执行插值相关（避免“显示与拾取不一致”）
        # 性能优化：只处理当前过滤后道集合，避免全体道预处理。
        raw_traces = [np.asarray(traces[int(i)]) for i in idx_local_to_global]
        offsets_subset = offsets[idx_local_to_global]
        trace_headers = self.loaded.get("trace_headers", [])
        gains_subset = np.ones(len(raw_traces), dtype=float)
        if trace_headers:
            for li, gidx in enumerate(idx_local_to_global):
                gi = int(gidx)
                if 0 <= gi < len(trace_headers):
                    gains_subset[li] = float(max(1, int(getattr(trace_headers[gi], "igaini", 1) or 1)))
        if times.size > 1:
            sr = 1.0 / float(times[1] - times[0])
        else:
            sr = None
        try:
            proc_params = self._build_processing_params()
            traces_for_pick = self.processor.process_traces(
                traces=raw_traces,
                times=times,
                offsets=offsets_subset,
                params=proc_params,
                gains=gains_subset,
                sampling_rate=sr,
                realtime_interaction=False,
            )
        except Exception as exc:
            self.lbl_status.setText(f"插值相关拾取失败：处理后波形生成失败: {exc}")
            return

        sorted_seed = sorted(pw_picks.items(), key=lambda kv: kv[0])
        seed_pairs = []
        allowed_idx = [int(i) for i in idx_local_to_global]
        for i in range(len(sorted_seed) - 1):
            i1, t1 = sorted_seed[i]
            i2, t2 = sorted_seed[i + 1]
            if i1 == i2:
                continue
            lo_i, hi_i = min(int(i1), int(i2)), max(int(i1), int(i2))
            has_middle = any(lo_i < ai < hi_i for ai in allowed_idx)
            if has_middle:
                li1 = global_to_local.get(int(i1))
                li2 = global_to_local.get(int(i2))
                if li1 is None or li2 is None:
                    continue
                seed_pairs.append((int(li1), float(t1), int(li2), float(t2)))
        if not seed_pairs:
            self.lbl_status.setText("插值相关拾取失败：没有可插值的种子区间（需两种子之间有道）")
            return

        corr_win = max(8, int(round(float(self.params.tcrcor) / max(1e-9, float(times[1] - times[0])))))
        lag_win = max(2, int(round(float(self.params.tlag) / max(1e-9, float(times[1] - times[0])))))

        added = 0
        pair_used = 0
        undo_pushed = False
        for pick1_local_idx, pick1_time, pick2_local_idx, pick2_time in seed_pairs:
            picks = self.interp_picker.interpolation_correlation_picking(
                traces=traces_for_pick,
                times=times,
                offsets=offsets_subset,
                pick1_idx=pick1_local_idx,
                pick2_idx=pick2_local_idx,
                pick1_time=pick1_time,
                pick2_time=pick2_time,
                correlation_window=corr_win,
                search_range=lag_win,
                hilbert_ratio=float(self.params.hilbratio),
                force_pick=force_pick,
            )
            pair_used += 1
            for local_idx, tpk in picks.items():
                if int(local_idx) < 0 or int(local_idx) >= idx_local_to_global.size:
                    continue
                gidx = int(idx_local_to_global[int(local_idx)])
                if int(gidx) not in allowed_set:
                    continue
                old_pick = self.pick_manager.get_pick(int(gidx), pick_word)
                if old_pick is not None and abs(float(old_pick) - float(tpk)) < 1e-9:
                    continue
                if not undo_pushed:
                    self._push_pick_undo("插值相关拾取")
                    undo_pushed = True
                if self.pick_manager.add_pick(int(gidx), float(tpk), pick_word):
                    added += 1
        self.request_render(delay_ms=10)
        mode_text = "强制模式" if force_pick else "严格模式"
        self.lbl_status.setText(
            f"插值相关完成（{mode_text}）：区间 {pair_used} 段，新增 {added} 个拾取"
        )

    def _render_now(self) -> None:
        if self.loaded is None:
            return
        # Show progress immediately for pre-denoise preparation stage.
        dn_render_active = bool(self._denoise_run_armed) and bool(self._denoise_params.get("enabled", False))
        dn_perf_diag = bool(self._denoise_params.get("perf_diag", False)) and bool(dn_render_active)
        perf_marks: Dict[str, float] = {}
        if dn_perf_diag:
            perf_marks["t0"] = time.perf_counter()
        if dn_render_active:
            self._set_denoise_progress(0, 100, "预处理中")
            try:
                QtWidgets.QApplication.processEvents()
            except Exception:
                pass

        t0 = time.perf_counter()
        traces = self.loaded.get("traces", [])
        offsets_all = np.asarray(self.loaded.get("offsets", []), dtype=float)
        if bool(self._orientation_preview_enabled):
            cache = self._orientation_preview_cache if isinstance(self._orientation_preview_cache, dict) else None
            if cache is not None and "traces" in cache and "offsets" in cache:
                traces = cache.get("traces", traces)
                offsets_all = np.asarray(cache.get("offsets", offsets_all), dtype=float)
            else:
                traces, offsets_all = self._apply_orientation_solution_to_all_traces(
                    traces=traces,
                    offsets=offsets_all,
                    force_rebuild=True,
                )
        times = np.asarray(self.loaded.get("times", []), dtype=float)
        if len(traces) == 0 or times.size == 0:
            if dn_render_active:
                self._set_denoise_progress(0, -1)
            if dn_perf_diag:
                t1 = time.perf_counter()
                self._debug_log("DENOISE_PERF", f"render_now early=no_data total_ms={(t1 - perf_marks['t0']) * 1000.0:.1f}")
            return

        if dn_render_active:
            self._set_denoise_progress(5, 100, "索引提取")
            try:
                QtWidgets.QApplication.processEvents()
            except Exception:
                pass
        if dn_perf_diag:
            perf_marks["before_extract"] = time.perf_counter()
        idx = self._extract_indices()
        if dn_perf_diag:
            perf_marks["after_extract"] = time.perf_counter()
        if idx.size == 0:
            if dn_render_active:
                self._set_denoise_progress(0, -1)
            self._last_denoise_scope_count = 0
            self._ensure_curve_pool(0)
            self._clear_shade_item()
            self._clear_stack_item()
            self._clear_static_preview_item()
            self._clear_theoretical_item()
            self._clear_txin_item()
            self._clear_txin_map_preview_item()
            self._clear_water_corr_item()
            self._clear_wave_select_item()
            self._clear_wave_select_marker_item()
            self._clear_waveop_stack_item()
            self.lbl_status.setText("当前过滤条件下无可显示道")
            if dn_perf_diag:
                t1 = time.perf_counter()
                ext_ms = (perf_marks["after_extract"] - perf_marks["before_extract"]) * 1000.0
                self._debug_log("DENOISE_PERF", f"render_now early=no_idx extract_ms={ext_ms:.1f} total_ms={(t1 - perf_marks['t0']) * 1000.0:.1f}")
            return

        # 仅保留视窗内道，再渲染预算抽稀
        if dn_render_active:
            self._set_denoise_progress(15, 100, "视窗筛选")
            try:
                QtWidgets.QApplication.processEvents()
            except Exception:
                pass
        if dn_perf_diag:
            perf_marks["before_visible"] = time.perf_counter()
        xcoords = offsets_all[idx]
        vis_mask = self._visible_mask(xcoords)
        idx_vis = idx[vis_mask]
        if dn_perf_diag:
            perf_marks["after_visible"] = time.perf_counter()
        if idx_vis.size == 0:
            if dn_render_active:
                self._set_denoise_progress(0, -1)
            self._last_denoise_scope_count = 0
            self._ensure_curve_pool(0)
            self._clear_shade_item()
            self._clear_stack_item()
            self._clear_static_preview_item()
            self._clear_theoretical_item()
            self._clear_txin_item()
            self._clear_txin_map_preview_item()
            self._clear_water_corr_item()
            self._clear_wave_select_item()
            self._clear_wave_select_marker_item()
            self._clear_waveop_stack_item()
            self.lbl_status.setText("当前视窗内无道")
            if dn_perf_diag:
                t1 = time.perf_counter()
                vis_ms = (perf_marks["after_visible"] - perf_marks["before_visible"]) * 1000.0
                self._debug_log("DENOISE_PERF", f"render_now early=no_visible visible_ms={vis_ms:.1f} total_ms={(t1 - perf_marks['t0']) * 1000.0:.1f}")
            return

        # 渲染预算：交互中更激进
        axis_w_px = max(1.0, float(self.plot.getViewBox().width()))
        target = int(max(250, min(1400, axis_w_px * (0.9 if self._viewport_interacting else 2.0))))
        stride = int(np.ceil(idx_vis.size / target)) if idx_vis.size > target else 1
        idx_render = idx_vis[::stride] if stride > 1 else idx_vis
        if idx_render[-1] != idx_vis[-1]:
            idx_render = np.append(idx_render, idx_vis[-1])
        # 强制渲染 V 选波相关道，避免抽稀后红色选段丢失
        current_wave_selections = self._current_apick_waveform_selections()
        if current_wave_selections:
            forced = np.asarray(
                [int(s.get("trace_idx", -1)) for s in current_wave_selections],
                dtype=int,
            )
            if forced.size > 0:
                forced = forced[(forced >= 0)]
                if forced.size > 0:
                    vis_set = set(int(v) for v in idx_vis)
                    forced = np.asarray([int(v) for v in forced if int(v) in vis_set], dtype=int)
                    if forced.size > 0:
                        idx_render = np.unique(np.concatenate([idx_render, forced]))

        render_offsets = offsets_all[idx_render]
        raw_traces = [np.asarray(traces[int(i)]) for i in idx_render]
        trace_headers_all = self.loaded.get("trace_headers", [])
        render_gains = np.ones(len(idx_render), dtype=float)
        if trace_headers_all:
            for ii, gidx in enumerate(idx_render):
                ig = int(gidx)
                if 0 <= ig < len(trace_headers_all):
                    render_gains[ii] = float(max(1, int(getattr(trace_headers_all[ig], "igaini", 1) or 1)))
        self._last_render_trace_indices = np.asarray(idx_render, dtype=int)
        self._last_render_offsets = np.asarray(render_offsets, dtype=float)

        self.params.irec = int(self.spin_irec.value())
        self.params.itype = int(self.combo_itype.currentIndex())
        self.params.nskip = int(self.spin_nskip.value())
        self.params.ndecim = int(self.spin_ndecim.value())
        self.params.vred = float(self.spin_vred.value())
        self.params.xmin = float(self.spin_xmin.value())
        self.params.xmax = float(self.spin_xmax.value())
        self.params.tmin = float(self.spin_tmin.value())
        self.params.tmax = float(self.spin_tmax.value())
        self.params.amp = float(self.spin_amp.value())
        self.params.iscale = int(self.combo_iscale.currentIndex())
        self.params.rcor = float(self.spin_rcor.value())
        self.params.sf = float(self.spin_sf.value())
        self.params.tvg = float(self.spin_tvg.value())
        self.params.pvg = float(self.spin_pvg.value())
        self.params.clip = float(self.spin_clip.value())
        self.params.ibndps = 1 if self.chk_filter.isChecked() else 0
        self.params.freqlo = float(self.spin_freqlo.value())
        self.params.freqhi = float(self.spin_freqhi.value())
        self.params.npoles = int(self.spin_npoles.value())
        self.params.izerop = 1 if self.chk_zerop.isChecked() else 0
        self._sync_denoise_params_from_ui()
        self.params.tcrcor = float(self.spin_tcrcor.value())
        self.params.tlag = float(self.spin_tlag.value())
        self.params.hilbratio = float(self.spin_hilbratio.value())
        mode_idx = int(self.combo_mode.currentIndex())
        if mode_idx == 1:
            self.params.ishade = 1
        elif mode_idx == 2:
            self.params.ishade = -1
        else:
            self.params.ishade = 0

        proc_params = self._build_processing_params()

        # 可选多边形 mute：多边形外部临时置零，再进入显示处理链。
        if dn_render_active:
            self._set_denoise_progress(30, 100, "构建输入")
            try:
                QtWidgets.QApplication.processEvents()
            except Exception:
                pass
        traces_for_processing = self._apply_mute_to_raw_traces(
            raw_traces=raw_traces,
            trace_indices=np.asarray(idx_render, dtype=int),
            render_offsets=np.asarray(render_offsets, dtype=float),
            times=np.asarray(times, dtype=float),
        )
        denoise_indices = self._resolve_denoise_indices(
            idx_all=np.asarray(idx, dtype=int),
            idx_visible=np.asarray(idx_vis, dtype=int),
            idx_render=np.asarray(idx_render, dtype=int),
        )
        self._last_denoise_scope_count = int(np.asarray(denoise_indices, dtype=int).size)

        # 处理链复用现有内核桥（realtime=True 时会自动降级带通）
        if times.size > 1:
            sr = 1.0 / float(times[1] - times[0])
        else:
            sr = None
        # iscale=1 且 sf=0 时，按稳定参考道估计 sf，避免随视窗变化导致噪声跳变
        user_sf = float(proc_params.sf)
        sf_override: Optional[float] = None
        if bool(self.chk_gain.isChecked()) and int(proc_params.iscale) == 1 and user_sf <= 0.0 and idx.size > 0:
            ref_gidx = int(idx[0])
            ref_gain = 1.0
            if trace_headers_all and 0 <= ref_gidx < len(trace_headers_all):
                ref_gain = float(max(1, int(getattr(trace_headers_all[ref_gidx], "igaini", 1) or 1)))
            sf_override = self._estimate_auto_sf(ref_gidx, sr, ref_gain)
            if sf_override > 0.0:
                proc_params.sf = float(sf_override)
        if dn_render_active:
            self._set_denoise_progress(45, 100, "处理链计算")
            try:
                QtWidgets.QApplication.processEvents()
            except Exception:
                pass
        if dn_perf_diag:
            perf_marks["before_process"] = time.perf_counter()
        try:
            processed = self.processor.process_traces(
                traces=traces_for_processing,
                times=times,
                offsets=render_offsets,
                params=proc_params,
                gains=render_gains,
                sampling_rate=sr,
                realtime_interaction=self._viewport_interacting,
            )
        except Exception as exc:
            # 处理链异常时仍保留上游 mute 结果，避免渲染链回退到原始波形
            processed = traces_for_processing
            self._denoise_backend_stage = f"{self._denoise_backend_stage} | 后处理回退:{type(exc).__name__}"
        if dn_perf_diag:
            perf_marks["after_process"] = time.perf_counter()
        processed_before_denoise = [np.asarray(tr, dtype=np.float64).copy() for tr in processed]
        denoise_use_cache = (
            bool(self._denoise_params.get("enabled", False))
            and bool(self._denoise_run_armed)
        )
        denoise_cache_hit = False
        if denoise_use_cache:
            if dn_render_active:
                self._set_denoise_progress(65, 100, "去噪准备")
                try:
                    QtWidgets.QApplication.processEvents()
                except Exception:
                    pass
            cache_key = self._denoise_cache_key_of(
                traces_before_denoise=processed_before_denoise,
                render_trace_indices=np.asarray(idx_render, dtype=int),
                denoise_trace_indices=np.asarray(denoise_indices, dtype=int),
            )
            cache_entry = self._denoise_cache_entries.get(cache_key)
            cache_out = cache_entry.get("output") if isinstance(cache_entry, dict) else None
            if isinstance(cache_out, list) and len(cache_out) == len(processed_before_denoise):
                processed = [np.asarray(tr, dtype=np.float64).copy() for tr in cache_out]
                self._denoise_backend_stage = str(cache_entry.get("stage", self._denoise_backend_stage))
                self._denoise_last_applied_count = int(cache_entry.get("applied_count", 0))
                self._denoise_last_delta_mean_abs = float(cache_entry.get("delta_mean_abs", 0.0))
                self._denoise_last_delta_max_abs = float(cache_entry.get("delta_max_abs", 0.0))
                denoise_cache_hit = True
                # LRU: 命中后提升为最新
                self._denoise_cache_entries.pop(cache_key, None)
                self._denoise_cache_entries[cache_key] = cache_entry
                self._debug_log("DENOISE_CACHE", f"hit size={len(self._denoise_cache_entries)}")
            else:
                self._debug_log("DENOISE_CACHE", f"miss size={len(self._denoise_cache_entries)}")
        # 交互期间仅复用缓存，不做新的去噪重算，避免缩放/拖拽卡顿。
        if self._viewport_interacting and denoise_use_cache and (not denoise_cache_hit):
            self._denoise_backend_stage = "交互中: 缓存未命中，暂停去噪重算"
            self._debug_log("DENOISE_CACHE", "interacting-miss: skip recompute")
        elif not denoise_cache_hit:
            # 去噪放在显示链后段：对“当前操作结果（含已启用 mute/滤波/增益）”执行。
            if dn_render_active:
                self._set_denoise_progress(0, 0, "去噪计算")
                try:
                    QtWidgets.QApplication.processEvents()
                except Exception:
                    pass
            if dn_perf_diag:
                perf_marks["before_denoise"] = time.perf_counter()
            processed = self._apply_denoise_to_render_traces(
                traces_in=processed,
                times=np.asarray(times, dtype=float),
                render_trace_indices=np.asarray(idx_render, dtype=int),
                denoise_trace_indices=np.asarray(denoise_indices, dtype=int),
            )
            if dn_perf_diag:
                perf_marks["after_denoise"] = time.perf_counter()
            if denoise_use_cache:
                cache_key = self._denoise_cache_key_of(
                    traces_before_denoise=processed_before_denoise,
                    render_trace_indices=np.asarray(idx_render, dtype=int),
                    denoise_trace_indices=np.asarray(denoise_indices, dtype=int),
                )
                self._denoise_cache_entries[cache_key] = {
                    "output": [np.asarray(tr, dtype=np.float64).copy() for tr in processed],
                    "stage": str(self._denoise_backend_stage),
                    "applied_count": int(self._denoise_last_applied_count),
                    "delta_mean_abs": float(self._denoise_last_delta_mean_abs),
                    "delta_max_abs": float(self._denoise_last_delta_max_abs),
                }
                self._denoise_cache_entries.move_to_end(cache_key)
                while len(self._denoise_cache_entries) > int(self._denoise_cache_limit):
                    self._denoise_cache_entries.popitem(last=False)
                self._debug_log("DENOISE_CACHE", f"store size={len(self._denoise_cache_entries)}")
        if dn_render_active:
            self._set_denoise_progress(90, 100, "结果整理")
            try:
                QtWidgets.QApplication.processEvents()
            except Exception:
                pass
        if (
            bool(self._denoise_params.get("show_diff", False))
            and int(getattr(self, "_denoise_last_applied_count", 0)) > 0
            and len(processed_before_denoise) == len(processed)
        ):
            active_set = set(int(i) for i in np.asarray(denoise_indices, dtype=int).tolist())
            diff_gain = float(self._denoise_params.get("diff_gain", 1.0))
            if (not np.isfinite(diff_gain)) or diff_gain <= 0.0:
                diff_gain = 1.0
            diff_traces: List[np.ndarray] = []
            for i, tr_now in enumerate(processed):
                base = np.asarray(processed_before_denoise[i], dtype=np.float64)
                now = np.asarray(tr_now, dtype=np.float64)
                if base.shape != now.shape:
                    now = np.resize(now, base.shape)
                gidx = int(idx_render[i]) if i < len(idx_render) else -1
                if gidx in active_set:
                    diff_traces.append((now - base) * diff_gain)
                else:
                    # 差值模式下隐藏未参与去噪的道，避免“零线看起来像原波形未变化”
                    diff_traces.append(np.full_like(base, np.nan))
            processed = diff_traces
            self._denoise_backend_stage = f"{self._denoise_backend_stage} | 差值显示(仅目标道)"
            self._debug_log(
                "DENOISE_DIFF",
                f"enabled=1 gain={diff_gain:g} applied={int(self._denoise_last_applied_count)} active={len(active_set)} render={len(diff_traces)}",
            )
        elif bool(self._denoise_params.get("ab_raw", False)):
            # A/B 仅控制显示：A=原始(当前显示链去噪前)，不参与“是否去噪计算”的决策
            processed = [np.asarray(tr, dtype=np.float64).copy() for tr in processed_before_denoise]
            self._denoise_backend_stage = "A/B显示: 原始(A)"
            self._debug_log("DENOISE_AB", "display=A(raw)")
        self._update_denoise_hint()

        # 时间抽样：用户 ndecim + 动态 LOD
        step_user = max(1, int(self.params.ndecim))
        step_lod = 1
        if times.size > 5000:
            step_lod = 2 if self._viewport_interacting else 1
        step = max(step_user, step_lod)
        t_plot = times[::step]

        # 显示缩放（Fortran 贴近）：不对每帧再按全局振幅归一，避免抵消增益参数效果
        if render_offsets.size > 1:
            spacing = float(np.median(np.diff(np.sort(render_offsets))))
            if spacing <= 0:
                spacing = 1.0
        else:
            spacing = 1.0
        scale = 0.45 * spacing * float(self.spin_dscale.value())

        # 仅用于叠加道显示幅度估计，不参与主道二次归一
        max_amp = 0.0
        for tr in processed:
            if len(tr) == 0:
                continue
            local = float(np.percentile(np.abs(tr), 98))
            if local > max_amp:
                max_amp = local
        if max_amp <= 1e-12:
            max_amp = 1.0

        self._ensure_curve_pool(len(processed))
        self._clear_shade_item()
        self._clear_stack_item()
        shade_segment_count = 0
        can_render_shade = (
            self.params.ishade != 0
            and self.shade_kernel is not None
            and (not self._viewport_interacting or self.chk_rt_shade.isChecked())
        )
        shade_x_parts: List[np.ndarray] = []
        shade_y_parts: List[np.ndarray] = []
        fill_positive = self.params.ishade > 0
        mute_polygon_arr: Optional[np.ndarray] = None
        if self._mute_enabled and len(self._mute_polygon_points) >= 3:
            mute_polygon_arr = np.asarray(self._mute_polygon_points, dtype=float)
        row_step = 1
        if self._viewport_interacting:
            row_step = 3
        elif len(idx_render) > 1200:
            row_step = 2
        for i, tr in enumerate(processed):
            trd = np.asarray(tr)[::step]
            tshift = self._compute_display_tshift(int(idx_render[i]), float(render_offsets[i]))
            t_trace = t_plot + tshift
            if mute_polygon_arr is not None and trd.size == t_trace.size:
                keep_mask = self._build_mute_inside_mask(
                    float(render_offsets[i]),
                    np.asarray(t_trace, dtype=float),
                    mute_polygon_arr,
                )
                if self._mute_invert:
                    keep_mask = ~keep_mask
                # 对保留区做一次可见域增益归一，避免“裁剪后仍沿用全道增益”导致显示发灰。
                if int(proc_params.iscale) in (0, 2):
                    valid_gain = keep_mask & np.isfinite(trd)
                    if np.any(valid_gain):
                        local_max = float(np.max(np.abs(trd[valid_gain])))
                        if local_max > 1e-12:
                            trd = np.asarray(trd, dtype=float).copy()
                            trd[valid_gain] = trd[valid_gain] * (float(proc_params.amp) / local_max)
                            if float(proc_params.clip) > 0.0:
                                c = float(proc_params.clip)
                                trd[valid_gain] = np.clip(trd[valid_gain], -c, c)
                trd = np.asarray(trd, dtype=float).copy()
                trd[~keep_mask] = np.nan
            xw = render_offsets[i] + trd * scale
            if self._map_link_trace_idx is not None and int(idx_render[i]) == int(self._map_link_trace_idx):
                self._curve_items[i].setPen(pg.mkPen("#f59e0b", width=2.0))
            elif int(idx_render[i]) in self._denoise_selected_traces:
                self._curve_items[i].setPen(pg.mkPen("#8b5cf6", width=1.5))
            else:
                self._curve_items[i].setPen(pg.mkPen(self._theme_color("wave_pen", "#0a0a0a"), width=1))
            self._curve_items[i].setData(xw, t_trace, connect="finite")
            if can_render_shade and xw.size > 0:
                try:
                    segs = self.shade_kernel.build_segments(
                        x_values=np.asarray(xw, dtype=np.float64),
                        t_values=np.asarray(t_trace, dtype=np.float64),
                        baseline_x=float(render_offsets[i]),
                        fill_positive=fill_positive,
                        row_step=row_step,
                    )
                    if segs.shape[0] > 0:
                        shade_x_parts.append(segs[:, :, 0].reshape(-1))
                        shade_y_parts.append(segs[:, :, 1].reshape(-1))
                        shade_segment_count += int(segs.shape[0])
                except Exception:
                    # 阴影失败不影响主波形渲染
                    pass

        if can_render_shade and shade_segment_count > 0:
            self._ensure_shade_item()
            try:
                sx = np.concatenate(shade_x_parts)
                sy = np.concatenate(shade_y_parts)
                self._shade_item.setData(sx, sy, connect="pairs")
            except Exception:
                self._clear_shade_item()

        # V 选波窗口高亮（红色）与圆点标注
        if current_wave_selections:
            apick = int(self.spin_apick.value())
            sel_color = pg.intColor(max(1, apick), hues=48, values=1, alpha=230)
            self._ensure_wave_select_item()
            self._wave_select_item.setPen(pg.mkPen(sel_color, width=1.8))
            render_map = {int(gidx): i for i, gidx in enumerate(np.asarray(idx_render, dtype=int))}
            seg_x_parts: List[np.ndarray] = []
            seg_y_parts: List[np.ndarray] = []
            marker_spots = []
            for sel in current_wave_selections:
                gidx = int(sel.get("trace_idx", -1))
                row = render_map.get(gidx)
                if row is None:
                    continue
                t_center = float(sel.get("t_display", 0.0))
                x0 = float(render_offsets[row])
                trd = np.asarray(processed[row])[::step]
                if trd.size != t_plot.size:
                    continue
                tshift = self._compute_display_tshift(gidx, x0)
                t_trace = t_plot + tshift
                x_trace = x0 + trd * scale
                mask = (t_trace >= (t_center - 0.3)) & (t_trace <= (t_center + 0.7))
                if np.any(mask):
                    seg_x = np.asarray(x_trace[mask], dtype=float)
                    seg_y = np.asarray(t_trace[mask], dtype=float)
                    seg_x_parts.append(np.concatenate([seg_x, np.asarray([np.nan])]))
                    seg_y_parts.append(np.concatenate([seg_y, np.asarray([np.nan])]))
                marker_spots.append(
                    {
                        "pos": (x0, t_center),
                        "brush": pg.mkBrush(sel_color),
                        "pen": pg.mkPen(self._theme_color("pick_active_edge", "#ffffff"), width=1.0),
                        "size": 8.0,
                    }
                )
            if seg_x_parts:
                self._wave_select_item.setData(
                    np.concatenate(seg_x_parts),
                    np.concatenate(seg_y_parts),
                    connect="finite",
                )
            else:
                self._clear_wave_select_item()
            if marker_spots:
                self._ensure_wave_select_marker_item()
                self._wave_select_marker_item.setData(spots=marker_spots)
            else:
                self._clear_wave_select_marker_item()
        else:
            self._clear_wave_select_item()
            self._clear_wave_select_marker_item()

        pick_count = self._render_picks(offsets_all, allowed_trace_indices=idx)
        # 静校正曲线：当前拾取字校正后时间（虚线预览）
        if self.static_corrector.has_corrections() and self.pick_manager is not None:
            apick = int(self.spin_apick.value())
            by_word = self.pick_manager.get_picks_by_word(apick)
            px: List[float] = []
            py: List[float] = []
            for gidx in idx:
                ig = int(gidx)
                tpk = by_word.get(ig)
                if tpk is None or float(tpk) <= 0:
                    continue
                if ig < 0 or ig >= offsets_all.size:
                    continue
                corr = float(self.static_corrector.get_correction(ig))
                x_val = float(offsets_all[ig])
                tshift_base = float(self._alignment_offsets.get(ig, 0.0)) + self._compute_reduction_tshift(ig, x_val)
                px.append(float(offsets_all[ig]))
                py.append(float(tpk) + corr + tshift_base)
            if len(px) >= 2:
                order = np.argsort(np.asarray(px, dtype=float))
                xarr = np.asarray(px, dtype=float)[order]
                yarr = np.asarray(py, dtype=float)[order]
                self._ensure_static_preview_item()
                self._static_preview_item.setData(xarr, yarr)
            else:
                self._clear_static_preview_item()
        else:
            self._clear_static_preview_item()
        theory_points = 0
        txin_points = 0
        txin_preview_points = 0
        water_points = 0
        if self.show_theoretical_times and self.theoretical_traveltime_calculator is not None:
            try:
                th_data = self.theoretical_traveltime_calculator.get_theoretical_times(xcoords)
                if th_data is not None and "distance" in th_data and "time" in th_data:
                    tx = np.asarray(th_data["distance"], dtype=float)
                    tt = np.asarray(th_data["time"], dtype=float)
                    t_reduction = np.asarray(
                        [self._compute_reduction_tshift(-1, float(xv)) for xv in tx],
                        dtype=float,
                    )
                    tt_disp = tt + t_reduction
                    self.theoretical_times_data = {"distances": tx, "times": tt}
                    self._ensure_theoretical_item()
                    self._theoretical_item.setData(tx, tt_disp)
                    theory_points = int(tx.size)
                    if self.show_water_layer_correction:
                        avg_corr = 0.0
                        corr_interp = None
                        if self.water_layer_corrected_times:
                            if "avg_correction" in self.water_layer_corrected_times:
                                arr = np.asarray(self.water_layer_corrected_times["avg_correction"], dtype=float)
                                if arr.size > 0:
                                    avg_corr = float(arr[0])
                            dmap = np.asarray(self.water_layer_corrected_times.get("distances", []), dtype=float)
                            cmap = np.asarray(self.water_layer_corrected_times.get("corrections", []), dtype=float)
                            if dmap.size >= 2 and cmap.size == dmap.size:
                                # 按距离逐点插值校正；超出范围使用边界值
                                corr_interp = np.interp(tx, dmap, cmap, left=float(cmap[0]), right=float(cmap[-1]))
                            elif dmap.size == 1 and cmap.size == 1:
                                corr_interp = np.full_like(tx, float(cmap[0]), dtype=float)
                        if corr_interp is None:
                            corr_interp = np.full_like(tx, avg_corr, dtype=float)
                        wt = (tt - corr_interp) + t_reduction
                        self._ensure_water_corr_item()
                        self._water_corr_item.setData(tx, wt)
                        water_points = int(tx.size)
                    else:
                        self._clear_water_corr_item()
                else:
                    self._clear_theoretical_item()
                    self._clear_water_corr_item()
            except Exception:
                self._clear_theoretical_item()
                self._clear_water_corr_item()
        else:
            self._clear_theoretical_item()
            self._clear_water_corr_item()

        if self.show_txin_overlay and self.txin_overlay_data is not None:
            try:
                tx_off = np.asarray(self.txin_overlay_data.get("offsets", []), dtype=float)
                tx_t = np.asarray(self.txin_overlay_data.get("times", []), dtype=float)
                tx_pw = np.asarray(self.txin_overlay_data.get("pick_words", []), dtype=int)
                if tx_off.size > 0 and tx_t.size == tx_off.size and tx_pw.size == tx_off.size:
                    tx_red = np.asarray(
                        [self._compute_reduction_tshift(-1, float(xv)) for xv in tx_off],
                        dtype=float,
                    )
                    tx_disp = tx_t + tx_red
                    self._ensure_txin_item()
                    spots = []
                    for i in range(int(tx_off.size)):
                        pw = int(tx_pw[i])
                        color = pg.intColor(max(1, pw), hues=48, values=1, alpha=220)
                        spots.append(
                            {
                                "pos": (float(tx_off[i]), float(tx_disp[i])),
                                "brush": pg.mkBrush(color),
                                "pen": pg.mkPen(self._theme_color("pick_active_edge", "#ffffff"), width=0.8),
                                "size": 7.0,
                                "data": {"pick_word": pw},
                            }
                        )
                    self._txin_item.setData(spots=spots)
                    txin_points = int(tx_off.size)
                else:
                    self._clear_txin_item()
            except Exception:
                self._clear_txin_item()
        else:
            self._clear_txin_item()

        if self.txin_map_preview_data is not None:
            try:
                p_idx = np.asarray(self.txin_map_preview_data.get("trace_indices", []), dtype=int)
                p_t = np.asarray(self.txin_map_preview_data.get("times", []), dtype=float)
                p_pw = np.asarray(self.txin_map_preview_data.get("pick_words", []), dtype=int)
                if p_idx.size > 0 and p_t.size == p_idx.size and p_pw.size == p_idx.size:
                    spots = []
                    for i in range(int(p_idx.size)):
                        gidx = int(p_idx[i])
                        if gidx < 0 or gidx >= offsets_all.size:
                            continue
                        x = float(offsets_all[gidx])
                        y = float(p_t[i]) + self._compute_display_tshift(gidx, x)
                        color = pg.intColor(max(1, int(p_pw[i])), hues=48, values=1, alpha=220)
                        spots.append(
                            {
                                "pos": (x, y),
                                "brush": pg.mkBrush(0, 0, 0, 0),
                                "pen": pg.mkPen(color, width=1.4),
                                "symbol": "x",
                                "size": 10.0,
                                "data": {"pick_word": int(p_pw[i])},
                            }
                        )
                    if spots:
                        self._ensure_txin_map_preview_item()
                        self._txin_map_preview_item.setData(spots=spots)
                        txin_preview_points = len(spots)
                    else:
                        self._clear_txin_map_preview_item()
                else:
                    self._clear_txin_map_preview_item()
            except Exception:
                self._clear_txin_map_preview_item()
        else:
            self._clear_txin_map_preview_item()

        # 波形操作叠加：显示在当前视图中心右侧，避免遮挡主剖面
        if self.waveop_stack_result is not None:
            try:
                tau = np.asarray(self.waveop_stack_result.get("tau", []), dtype=float)
                stack = np.asarray(self.waveop_stack_result.get("stack", []), dtype=float)
                centers = np.asarray(self.waveop_stack_result.get("centers", []), dtype=float)
                if tau.size >= 8 and stack.size == tau.size:
                    x_view, y_view = self.plot.getViewBox().viewRange()
                    x_center = 0.5 * float(x_view[0] + x_view[1])
                    y_anchor = float(np.median(centers)) if centers.size > 0 else 0.5 * float(y_view[0] + y_view[1])
                    x_span = max(1e-6, float(abs(x_view[1] - x_view[0])))
                    x_anchor = x_center + 0.30 * x_span
                    x_max = float(max(x_view[0], x_view[1]))
                    x_anchor = min(x_anchor, x_max - 0.04 * x_span)
                    amp = float(np.percentile(np.abs(stack), 98)) if stack.size > 0 else 0.0
                    if amp <= 1e-12:
                        amp = 1.0
                    x_scale = 0.10 * x_span / amp
                    x_stack = x_anchor + stack * x_scale
                    y_stack = y_anchor + tau
                    self._ensure_waveop_stack_item()
                    self._waveop_stack_item.setData(x_stack, y_stack)
                else:
                    self._clear_waveop_stack_item()
            except Exception:
                self._clear_waveop_stack_item()
        else:
            self._clear_waveop_stack_item()

        # 叠加显示：当前活动拾取字，按“拾取时刻 ±0.5s”窗口对齐叠加
        stack_auto_flag = 0
        stack_seed_count = 0
        if self.chk_show_stack.isChecked() and len(processed) >= 2:
            try:
                apick = int(self.spin_apick.value())
                by_word = self.pick_manager.get_picks_by_word(apick) if self.pick_manager is not None else {}
                if by_word:
                    dt_plot = float(t_plot[1] - t_plot[0]) if t_plot.size > 1 else 0.001
                    half_window_sec = 0.5
                    tau = np.arange(
                        -half_window_sec,
                        half_window_sec + 0.5 * dt_plot,
                        dt_plot,
                        dtype=np.float64,
                    )
                    if tau.size >= 8:
                        stack_acc = np.zeros_like(tau, dtype=np.float64)
                        n_stack = 0
                        idx_render_arr = np.asarray(idx_render, dtype=int)
                        for i, tr in enumerate(processed):
                            gidx = int(idx_render_arr[i])
                            tpk = by_word.get(gidx)
                            if tpk is None or float(tpk) <= 0:
                                continue
                            trd = np.asarray(tr)[::step]
                            if trd.size != t_plot.size:
                                continue
                            # 叠加使用“当前显示状态”的时间基准（包含静校正），不依赖 A/F 的波形平移状态
                            pick_eff = float(tpk)
                            t_trace = t_plot
                            if self.static_correction_enabled:
                                corr = float(self.static_corrector.get_correction(gidx))
                                pick_eff += corr
                                t_trace = t_plot + corr
                            seg = np.interp(pick_eff + tau, t_trace, trd, left=0.0, right=0.0)
                            stack_acc += seg
                            n_stack += 1

                        if n_stack > 0:
                            stack_seed_count = int(n_stack)
                            stack_auto_flag = 1
                            stack = stack_acc / float(n_stack)

                            # 叠加道始终显示在当前画面中心右侧
                            x_view, y_view = self.plot.getViewBox().viewRange()
                            x_center = 0.5 * float(x_view[0] + x_view[1])
                            y_center = 0.5 * float(y_view[0] + y_view[1])
                            x_span = max(1e-6, float(abs(x_view[1] - x_view[0])))
                            x_anchor = x_center + 0.30 * x_span
                            x_max = float(max(x_view[0], x_view[1]))
                            x_anchor = min(x_anchor, x_max - 0.04 * x_span)

                            stack_amp = float(np.percentile(np.abs(stack), 98)) if stack.size > 0 else 0.0
                            if stack_amp <= 1e-12:
                                stack_amp = 1.0
                            x_scale = 0.10 * x_span / stack_amp
                            x_stack = x_anchor + stack * x_scale
                            y_stack = y_center + tau

                            self._ensure_stack_item()
                            self._stack_item.setData(x_stack, y_stack)
            except Exception:
                self._clear_stack_item()

        # 首次渲染时拟合一次；之后范围由用户窗口控制（xmin/xmax/tmin/tmax）
        if not self._viewport_interacting and (not self._did_initial_view_fit):
            xmin, xmax = float(np.min(offsets_all)), float(np.max(offsets_all))
            ymin, ymax = float(np.min(times)), float(np.max(times))
            self.plot.setXRange(xmin, xmax, padding=0.02)
            self.plot.setYRange(ymax, ymin, padding=0.02)  # 反转Y已开启，传入大->小
            vb = self.plot.getViewBox()
            vb.enableAutoRange(axis=vb.XAxis, enable=False)
            vb.enableAutoRange(axis=vb.YAxis, enable=False)
            self._did_initial_view_fit = True

        header = self.loaded.get("header")
        vredf = float(getattr(header, "vredf", 0.0) or 0.0) if header is not None else 0.0
        vred = float(self.params.vred)
        rvred = (1.0 / vred) if vred > 0 else 0.0
        rvredf = (1.0 / vredf) if vredf > 0 else 0.0
        d_rvred = rvred - rvredf

        _elapsed = (time.perf_counter() - t0) * 1000.0
        if dn_perf_diag:
            t1 = time.perf_counter()
            ext_ms = (perf_marks.get("after_extract", perf_marks["t0"]) - perf_marks.get("before_extract", perf_marks["t0"])) * 1000.0
            vis_ms = (perf_marks.get("after_visible", perf_marks["t0"]) - perf_marks.get("before_visible", perf_marks["t0"])) * 1000.0
            proc_ms = (perf_marks.get("after_process", perf_marks["t0"]) - perf_marks.get("before_process", perf_marks["t0"])) * 1000.0
            den_ms = (perf_marks.get("after_denoise", perf_marks["t0"]) - perf_marks.get("before_denoise", perf_marks["t0"])) * 1000.0
            self._debug_log(
                "DENOISE_PERF",
                f"render_now total_ms={(t1 - perf_marks['t0']) * 1000.0:.1f} extract_ms={ext_ms:.1f} "
                f"visible_ms={vis_ms:.1f} process_ms={proc_ms:.1f} denoise_ms={den_ms:.1f} "
                f"scope={int(getattr(self, '_last_denoise_scope_count', 0))} render={len(idx_render)}",
            )
        if dn_render_active:
            self._set_denoise_progress(0, -1)

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

