"""
plot_manager.py - ZPLOT 绘图管理模块

负责绘制地震剖面，包括 wiggle 图和变面积图
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import logging
import os
import math
try:
    from numba import njit
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def _wrap(func):
            return func
        return _wrap

# 配置日志
logger = logging.getLogger(__name__)

# 配置 matplotlib 字体：支持中文显示
# 尝试导入中文字体函数
try:
    from .top_toolbar import get_chinese_font
    # 获取中文字体名称
    chinese_font = get_chinese_font(13)
    chinese_font_name = chinese_font[0] if isinstance(chinese_font, tuple) else chinese_font
    # 配置 matplotlib 使用中文字体（优先），同时保留英文字体作为备选
    plt.rcParams['font.sans-serif'] = [chinese_font_name, 'Microsoft YaHei', 'SimHei', 'SimSun', 
                                       'DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
except ImportError:
    # 如果无法导入，使用系统常见中文字体
    import platform
    system = platform.system()
    if system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 
                                           'DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    elif system == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'STHeiti', 'STSong',
                                           'DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
    else:  # Linux
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC',
                                           'DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 配置日志
import logging
logger = logging.getLogger(__name__)

try:
    from .parameters import ZPlotParameters
    from .data_processor import DataProcessor
    from .src_kernel_bridge import SrcShadeKernelBridge
except ImportError:
    from parameters import ZPlotParameters
    from data_processor import DataProcessor
    from src_kernel_bridge import SrcShadeKernelBridge


@njit(cache=True)
def _build_horizontal_shade_segments_numba(x: np.ndarray, t: np.ndarray,
                                           baseline_x: float, fill_positive: bool) -> np.ndarray:
    """Numba 内核：构建水平阴影线段。"""
    n = x.shape[0]
    count = 0
    for i in range(n):
        xi = x[i]
        if fill_positive:
            if xi > baseline_x:
                count += 1
        else:
            if xi < baseline_x:
                count += 1
    if count == 0:
        return np.empty((0, 2, 2), dtype=np.float64)

    segs = np.empty((count, 2, 2), dtype=np.float64)
    k = 0
    for i in range(n):
        xi = x[i]
        keep = False
        if fill_positive:
            if xi > baseline_x:
                keep = True
        else:
            if xi < baseline_x:
                keep = True
        if keep:
            yi = t[i]
            segs[k, 0, 0] = baseline_x
            segs[k, 0, 1] = yi
            segs[k, 1, 0] = xi
            segs[k, 1, 1] = yi
            k += 1
    return segs


class PlotManager:
    """绘图管理类 - 管理地震剖面显示"""
    
    def __init__(self, ax: Axes):
        """初始化绘图管理器
        
        Args:
            ax: Matplotlib 轴对象
        """
        self.ax = ax
        self.traces_artists = []  # 存储绘制的道对象，用于更新
        self.pick_artists = []  # 单独跟踪拾取点 artist，支持轻量刷新
        self.data_processor = DataProcessor()  # 数据处理器
        self.alignment_offsets: Dict[int, float] = {}  # 对齐偏移量 {trace_idx: offset_time}
        self.alignment_overlap = 0.2  # 对齐模式下各道重叠百分比（0.2 = 20%重叠）
        self.aligned_trace_to_position: Dict[int, float] = {}  # 对齐模式下trace_idx到新位置的映射
        self.original_offsets: Dict[int, float] = {}  # 原始炮检距映射 {trace_idx: original_offset}，用于折合时间计算
        self.static_corrections: Dict[int, float] = {}  # 静校正量 {trace_idx: correction_time}
        self.static_correction_visualization = True  # 是否显示静校正可视化
        self.static_correction_preview_mode = False  # 预览模式：只显示校正后的震相曲线，不应用到波形
        self.static_correction_smoothness = 0.1  # 平滑度参数（0-1），0=通过所有点，1=最平滑
        
        # X键功能：移除的trace标记
        self.removed_traces: Dict[int, float] = {}  # {trace_idx: offset} 存储被移除的trace及其offset
        self.removed_trace_markers = []  # 存储顶部小点标记的artist对象
        self.static_correction_artists = []  # 存储静校正可视化artist对象
        
        # LOD（细节层次）机制：根据缩放级别减少绘制点数
        self.enable_lod = True  # 是否启用 LOD
        self.lod_threshold = 5000  # LOD 阈值：当点数超过此值时启用降采样
        
        # 存储当前使用的X轴坐标（用于拾取点绘制）
        self.current_x_coordinates: Optional[np.ndarray] = None
        self.current_filtered_indices: Optional[List[int]] = None
        self.last_filter_stats: Dict[str, Any] = {}
        self.shade_quality_preset: str = 'balanced'
        self._realtime_interaction_active: bool = False
        self.enable_numba_kernels: bool = bool(HAS_NUMBA)
        self.src_shade_kernel = SrcShadeKernelBridge()
        
    def plot_seismic_section(self, traces: List[np.ndarray], offsets: np.ndarray,
                            times: np.ndarray, params: ZPlotParameters,
                            trace_headers: Optional[List] = None,
                            preserve_zoom: bool = False,
                            user_xlim: Optional[Tuple[float, float]] = None,
                            user_ylim: Optional[Tuple[float, float]] = None,
                            alignment_offsets: Optional[Dict[int, float]] = None,
                            aligned_trace_indices: Optional[List[int]] = None,
                            records: Optional[List] = None,
                            realtime_interaction: bool = False) -> None:
        """绘制地震剖面
        
        Args:
            traces: 道数据列表
            offsets: 炮检距数组
            times: 时间数组
            params: 绘图参数
            trace_headers: 道头信息列表（可选）
            preserve_zoom: 是否保留用户设置的缩放
            user_xlim: 用户设置的 X 轴范围 (x_min, x_max)
            user_ylim: 用户设置的 Y 轴范围 (y_min, y_max)
            alignment_offsets: 对齐偏移量字典 {trace_idx: offset_time}（可选）
        """
        # 清空当前图形
        self._realtime_interaction_active = bool(realtime_interaction)
        self.ax.clear()
        self.traces_artists = []
        self.pick_artists = []
        # ✅ 清除理论走时曲线引用
        if hasattr(self, 'theoretical_traveltime_lines'):
            self.theoretical_traveltime_lines = []
        
        if not traces or len(traces) == 0:
            self.ax.text(0.5, 0.5, '没有可用数据', 
                        transform=self.ax.transAxes, 
                        ha='center', va='center', fontsize=14)
            return
        
        # 应用数据过滤和范围限制
        # 首先根据参数过滤（记录号、数据类型、xmin/xmax、tmin/tmax等）
        filtered_data = self._filter_data(traces, offsets, times, params, trace_headers)
        self.last_filter_stats = filtered_data.get('filter_stats', {}) if isinstance(filtered_data, dict) else {}
        
        # 过滤掉被移除的trace（X键功能）
        if self.removed_traces:
            filtered_traces = filtered_data['traces']
            filtered_offsets = filtered_data['offsets']
            filtered_indices = filtered_data.get('indices', list(range(len(filtered_traces))))
            
            # 创建新的过滤列表
            new_traces = []
            new_offsets = []
            new_indices = []
            
            for i, trace_idx in enumerate(filtered_indices):
                if trace_idx not in self.removed_traces:
                    new_traces.append(filtered_traces[i])
                    new_offsets.append(filtered_offsets[i])
                    new_indices.append(trace_idx)
            
            filtered_data['traces'] = new_traces
            filtered_data['offsets'] = np.array(new_offsets)
            filtered_data['indices'] = new_indices
        
        # 可见窗口范围（用于绘制阶段选择当前窗口内道和填充）
        # 注意：这里不截断原始数据读取，仅控制“当前窗口绘制哪些波形”
        visible_xrange: Optional[Tuple[float, float]] = None
        visible_yrange: Optional[Tuple[float, float]] = None
        if params.xmin is not None and params.xmax is not None:
            visible_xrange = (
                min(float(params.xmin), float(params.xmax)),
                max(float(params.xmin), float(params.xmax)),
            )
        if params.tmin is not None and params.tmax is not None:
            visible_yrange = (
                min(float(params.tmin), float(params.tmax)),
                max(float(params.tmin), float(params.tmax)),
            )
        
        if not filtered_data['traces']:
            # 检查是否是因为记录号不存在
            if trace_headers and len(trace_headers) > 0 and params.irec > 0:
                # 检查是否存在该记录号
                record_exists = False
                matching_traces = []  # 属于该记录的道
                for th in trace_headers:
                    if th.ishoti == params.irec:
                        record_exists = True
                        matching_traces.append(th)
                
                if not record_exists:
                    # 记录号不存在
                    error_msg = f'记录 {params.irec} 不存在\n\n请检查记录号是否正确'
                    self.ax.text(0.5, 0.5, error_msg, 
                                transform=self.ax.transAxes, 
                                ha='center', va='center', fontsize=14,
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                else:
                    # 记录存在但没有匹配的道（可能是其他过滤条件导致）
                    # 分析原因：检查数据类型过滤
                    error_msg = f'记录 {params.irec} 没有数据\n\n'

                    filter_stats = filtered_data.get('filter_stats', {})
                    drop_reasons = filter_stats.get('drop_reasons', {}) if isinstance(filter_stats, dict) else {}
                    total_candidates = int(filter_stats.get('total_candidates', 0) or 0) if isinstance(filter_stats, dict) else 0
                    itype_drops = int(drop_reasons.get('itype', 0) or 0)
                    nskip_drops = int(filter_stats.get('nskip_drop', 0) or 0) if isinstance(filter_stats, dict) else 0
                    other_drops = (
                        int(drop_reasons.get('xmin', 0) or 0) +
                        int(drop_reasons.get('xmax', 0) or 0) +
                        int(drop_reasons.get('imute', 0) or 0) +
                        int(drop_reasons.get('index_oob', 0) or 0)
                    )
                    only_itype_blocked = (
                        params.itype != 0 and
                        total_candidates > 0 and
                        itype_drops >= total_candidates and
                        other_drops == 0 and
                        nskip_drops == 0
                    )

                    if only_itype_blocked:
                        # 统计该记录中实际存在的分量类型
                        available_types = set()
                        for th in matching_traces:
                            available_types.add(th.itypei)
                        
                        # 生成分量类型名称
                        type_names = {1: '垂直', 2: '径向', 3: '横向', 4: '水听器'}
                        available_names = [f'{type_names.get(t, f"类型{t}")}({t})' for t in sorted(available_types)]
                        
                        # 生成请求的分量类型名称
                        requested_name = ''
                        if params.itype > 0:
                            requested_name = type_names.get(params.itype, f'类型{params.itype}')
                        elif params.itype == -1:
                            requested_name = '垂直+径向'
                        elif params.itype == -2:
                            requested_name = '径向+横向'
                        elif params.itype == -3:
                            requested_name = '径向+水听器'
                        elif params.itype == -4:
                            requested_name = '垂直+水听器'
                        
                        error_msg += f'请求的分量类型: {requested_name}\n'
                        error_msg += f'数据中存在的分量类型: {", ".join(available_names)}\n\n'
                        error_msg += '提示: 请将 itype 设置为 0（全部）或选择数据中存在的分量类型'
                    elif nskip_drops > 0 and (total_candidates - nskip_drops) <= 0:
                        error_msg += (
                            f'当前 nskip={params.nskip} 导致抽道后无可显示道。\n'
                            f'建议减小 nskip 或设置为 0。'
                        )
                    else:
                        error_msg += '过滤条件可能过于严格（不一定是分量类型导致）\n'
                        error_msg += f'当前参数: itype={params.itype}, nskip={params.nskip}, imute={params.imute}\n'
                        error_msg += f'窗口范围: x=[{params.xmin}, {params.xmax}], t=[{params.tmin}, {params.tmax}]'
                    
                    self.ax.text(0.5, 0.5, error_msg, 
                                transform=self.ax.transAxes, 
                                ha='center', va='center', fontsize=12,
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            else:
                # 其他原因导致无数据
                self.ax.text(0.5, 0.5, '过滤后没有数据', 
                            transform=self.ax.transAxes, 
                            ha='center', va='center', fontsize=14)
            return
        
        filtered_traces = filtered_data['traces']
        filtered_offsets = filtered_data['offsets']
        filtered_times = filtered_data['times']
        filtered_indices = filtered_data.get('indices', list(range(len(filtered_traces))))
        
        # 存储对齐偏移量（用于后续绘制时应用）
        self.alignment_offsets = alignment_offsets if alignment_offsets else {}
        
        # 存储对齐模式下的道索引映射（trace_idx -> 新的道号位置）
        self.aligned_trace_to_position: Dict[int, float] = {}
        
        # 存储原始炮检距映射（用于折合时间计算）
        self.original_offsets: Dict[int, float] = {}
        for i, trace_idx in enumerate(filtered_indices):
            if trace_idx < len(offsets):
                self.original_offsets[trace_idx] = offsets[trace_idx]
        
        # 如果启用了对齐，只显示有当前拾取字的道
        if aligned_trace_indices and len(aligned_trace_indices) > 0:
            # 过滤：只保留有拾取点的道
            aligned_filtered_traces = []
            aligned_filtered_offsets = []
            aligned_filtered_indices = []
            
            for i, trace_idx in enumerate(filtered_indices):
                if trace_idx in aligned_trace_indices:
                    aligned_filtered_traces.append(filtered_traces[i])
                    aligned_filtered_offsets.append(filtered_offsets[i])
                    aligned_filtered_indices.append(trace_idx)
            
            # 如果过滤后没有道，显示提示信息
            if len(aligned_filtered_traces) == 0:
                # 显示提示信息
                self.ax.text(0.5, 0.5, '当前拾取词没有可用于对齐的道\n\n请检查当前拾取词是否存在拾取点', 
                            transform=self.ax.transAxes, 
                            ha='center', va='center', fontsize=14,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                # 使用空数据，但保留原始过滤结果的范围信息（用于设置坐标轴）
                filtered_traces = []
                filtered_offsets = filtered_data['offsets'] if len(filtered_data['offsets']) > 0 else np.array([])
                filtered_indices = []
                # 设置坐标轴范围（如果用户已经设置了缩放，保留它）
                self._setup_axes(filtered_offsets, filtered_times, params, 
                                preserve_zoom=preserve_zoom,
                                user_xlim=user_xlim,
                                user_ylim=user_ylim)
                return
            else:
                filtered_traces = aligned_filtered_traces
                # 对齐模式下，保持使用原始的offset值作为横坐标
                # 不改变横坐标，只是对波形进行时间偏移
                filtered_offsets = np.array(aligned_filtered_offsets)
                filtered_indices = aligned_filtered_indices
                
                # 建立trace_idx到位置的映射（用于拾取点绘制）
                # 使用原始offset值
                for i, trace_idx in enumerate(filtered_indices):
                    if i < len(filtered_offsets):
                        self.aligned_trace_to_position[trace_idx] = filtered_offsets[i]
        else:
            # 未启用对齐，使用原始过滤结果
            filtered_traces = filtered_data['traces']
            filtered_offsets = filtered_data['offsets']
            filtered_indices = filtered_data.get('indices', list(range(len(filtered_traces))))

        # 保留“完整过滤结果”用于拾取点/坐标轴等叠加层；渲染可按预算抽稀
        full_filtered_traces = filtered_traces
        full_filtered_offsets = np.asarray(filtered_offsets)
        full_filtered_indices = list(filtered_indices)
        x_coordinates_full = self._calculate_x_coordinates(
            offsets, full_filtered_indices, trace_headers, records, params
        )

        # 二阶段优化：在数据处理前先按当前可见 X 窗口裁剪渲染候选，避免处理不可见道
        visible_render_indices = list(range(len(full_filtered_traces)))
        if visible_xrange is not None and len(x_coordinates_full) == len(full_filtered_traces):
            x_min_v, x_max_v = float(visible_xrange[0]), float(visible_xrange[1])
            visible_render_indices = [
                i for i, x in enumerate(x_coordinates_full)
                if x_min_v <= float(x) <= x_max_v
            ]
            if len(visible_render_indices) == 0:
                # 视窗内无道时，允许本帧直接空渲染，避免无效处理
                filtered_traces = []
                filtered_offsets = np.asarray([], dtype=float)
                filtered_indices = []
            else:
                filtered_traces = [full_filtered_traces[i] for i in visible_render_indices]
                filtered_offsets = np.asarray([full_filtered_offsets[i] for i in visible_render_indices], dtype=float)
                filtered_indices = [full_filtered_indices[i] for i in visible_render_indices]
        else:
            filtered_traces = full_filtered_traces
            filtered_offsets = full_filtered_offsets
            filtered_indices = full_filtered_indices

        # 大文件渲染预算：限制单帧参与渲染的道数（交互阶段更激进）
        render_stride = 1
        total_render_candidates = len(filtered_traces)
        if total_render_candidates > 0:
            try:
                axis_w_px = float(self.ax.bbox.width)
            except Exception:
                axis_w_px = 0.0
            axis_w_px = max(1.0, axis_w_px)
            if realtime_interaction:
                target_render_traces = int(max(300, min(1600, axis_w_px * 1.1)))
            else:
                target_render_traces = int(max(1200, min(6000, axis_w_px * 2.2)))
            if total_render_candidates > target_render_traces:
                render_stride = int(math.ceil(total_render_candidates / target_render_traces))

        if render_stride > 1 and total_render_candidates > 1:
            sample_idx = np.arange(0, total_render_candidates, render_stride, dtype=int)
            if sample_idx.size == 0:
                sample_idx = np.array([0, total_render_candidates - 1], dtype=int)
            elif sample_idx[-1] != (total_render_candidates - 1):
                sample_idx = np.append(sample_idx, total_render_candidates - 1)
            filtered_traces = [filtered_traces[k] for k in sample_idx]
            filtered_offsets = np.asarray([filtered_offsets[k] for k in sample_idx], dtype=float)
            filtered_indices = [filtered_indices[k] for k in sample_idx]
        else:
            filtered_traces = filtered_traces
            filtered_offsets = filtered_offsets
            filtered_indices = filtered_indices

        # 记录本帧渲染抽稀信息，供状态栏与日志观测
        if isinstance(self.last_filter_stats, dict):
            self.last_filter_stats["render_stride"] = int(render_stride)
            self.last_filter_stats["render_candidates"] = int(total_render_candidates)
            self.last_filter_stats["render_used"] = int(len(filtered_traces))
            self.last_filter_stats["render_visible_candidates"] = int(len(visible_render_indices))
        
        # 应用数据处理（滤波、增益、裁剪等）
        try:
            # 计算采样率
            if len(filtered_times) > 1:
                sampling_rate = 1.0 / (filtered_times[1] - filtered_times[0])
            else:
                sampling_rate = None
            
            # 处理数据
            processed_traces = self.data_processor.process_traces(
                filtered_traces,
                filtered_times,
                filtered_offsets,
                params,
                sampling_rate,
                realtime_interaction=realtime_interaction,
            )
        except Exception as e:
            logger.error(f"数据处理失败: {e}", exc_info=True)
            # 如果处理失败，使用原始数据
            processed_traces = filtered_traces
        
        # 注意：折合时间在绘制每道时单独应用，不在这里统一处理
        
        # 计算缩放因子（使用处理后的数据）
        scale = self._calculate_scale(processed_traces, filtered_offsets, params)
        
        # 获取过滤后的道头信息（用于线型设置）
        filtered_trace_headers = None
        if trace_headers and filtered_indices:
            filtered_trace_headers = [trace_headers[i] for i in filtered_indices if i < len(trace_headers)]
        
        # ✅ 根据 ixaxis 参数计算 X 轴坐标值
        # 计算X轴坐标（根据ixaxis参数）
        x_coordinates = self._calculate_x_coordinates(
            offsets, filtered_indices, trace_headers, records, params
        )
        
        # 存储X轴坐标和过滤后的索引（供plot_picks使用）
        self.current_x_coordinates = x_coordinates_full
        self.current_filtered_indices = full_filtered_indices
        
        # 根据显示模式选择绘图方法（使用处理后的数据）
        # ✅ 使用计算后的X轴坐标而不是原始的offsets
        if params.ishade == 0:
            # Wiggle 图
            self.plot_wiggle(processed_traces, x_coordinates, filtered_times, scale, params, 
                           filtered_indices, filtered_trace_headers,
                           visible_xrange=visible_xrange, visible_yrange=visible_yrange)
        elif params.ishade > 0:
            # 变面积图（正峰值填充）
            self.plot_variable_area(processed_traces, x_coordinates, filtered_times, 
                                   scale, params, fill_positive=True, 
                                   filtered_indices=filtered_indices, trace_headers=filtered_trace_headers,
                                   visible_xrange=visible_xrange, visible_yrange=visible_yrange,
                                   realtime_interaction=realtime_interaction)
        else:
            # 变面积图（负峰值填充）
            self.plot_variable_area(processed_traces, x_coordinates, filtered_times, 
                                   scale, params, fill_positive=False, 
                                   filtered_indices=filtered_indices, trace_headers=filtered_trace_headers,
                                   visible_xrange=visible_xrange, visible_yrange=visible_yrange,
                                   realtime_interaction=realtime_interaction)
        
        # 设置坐标轴（使用计算后的X轴坐标）
        # 关键修复：当启用折合时间(vred>0)时，波形绘制使用的是“折合后时间”，
        # 坐标轴也必须覆盖该时间范围，否则会出现“有数据但看不到”的现象。
        axis_times = filtered_times
        if params.vred > 0 and full_filtered_indices:
            reduction_values = []
            for trace_idx in full_filtered_indices:
                if trace_idx in self.original_offsets:
                    reduction_values.append(abs(self.original_offsets[trace_idx]) / params.vred)
            if reduction_values:
                rmin = min(reduction_values)
                rmax = max(reduction_values)
                axis_times = np.array(
                    [np.min(filtered_times) - rmax, np.max(filtered_times) - rmin],
                    dtype=float
                )

        self._setup_axes(x_coordinates_full, axis_times, params, 
                        preserve_zoom=preserve_zoom,
                        user_xlim=user_xlim,
                        user_ylim=user_ylim)
        
        # 绘制被移除trace的顶部标记（X键功能）
        self._plot_removed_trace_markers(full_filtered_offsets, filtered_times, params)
        
        # 绘制静校正可视化（在图形顶部显示静校正量）
        if self.static_corrections and not realtime_interaction:
            self.plot_static_correction_visualization(x_coordinates_full, full_filtered_indices, params)
        
    def _calculate_x_coordinates(self, offsets: np.ndarray, filtered_indices: List[int],
                                trace_headers: Optional[List], records: Optional[List],
                                params: ZPlotParameters) -> np.ndarray:
        """根据 ixaxis 参数计算 X 轴坐标值
        
        Args:
            offsets: 炮检距数组（原始）
            filtered_indices: 过滤后的道索引列表
            trace_headers: 道头信息列表
            records: 记录信息列表
            params: 绘图参数
            
        Returns:
            计算后的 X 轴坐标数组
        """
        x_coords = np.zeros(len(filtered_indices))
        
        # 获取ixaxis的绝对值（符号用于控制X轴反转，不影响坐标值计算）
        ixaxis_type = abs(params.ixaxis)
        
        for i, trace_idx in enumerate(filtered_indices):
            if ixaxis_type == 1:
                # 1: 炮检距（默认）
                x_coords[i] = offsets[trace_idx] if trace_idx < len(offsets) else 0.0
            elif ixaxis_type == 2:
                # -2: 模型位置（从记录信息中获取）
                if trace_headers and trace_idx < len(trace_headers):
                    th = trace_headers[trace_idx]
                    if records and hasattr(th, 'ishoti'):
                        shot_station = th.ishoti
                        # 查找对应的记录
                        for record in records:
                            if record.ishnum == shot_station:
                                # 使用 X 模型位置
                                x_coords[i] = record.xmod
                                break
                        else:
                            # 如果找不到记录，使用炮检距
                            x_coords[i] = offsets[trace_idx] if trace_idx < len(offsets) else 0.0
                    else:
                        x_coords[i] = offsets[trace_idx] if trace_idx < len(offsets) else 0.0
                else:
                    x_coords[i] = offsets[trace_idx] if trace_idx < len(offsets) else 0.0
            elif ixaxis_type == 3:
                # 3: 方位角
                if trace_headers and trace_idx < len(trace_headers):
                    th = trace_headers[trace_idx]
                    if hasattr(th, 'azi'):
                        x_coords[i] = th.azi
                    else:
                        x_coords[i] = offsets[trace_idx] if trace_idx < len(offsets) else 0.0
                else:
                    x_coords[i] = offsets[trace_idx] if trace_idx < len(offsets) else 0.0
            elif ixaxis_type == 4:
                # 4: 修正方位角（需要根据模型位置计算）
                if trace_headers and trace_idx < len(trace_headers):
                    th = trace_headers[trace_idx]
                    if records and hasattr(th, 'ishoti'):
                        shot_station = th.ishoti
                        # 查找对应的记录
                        for record in records:
                            if record.ishnum == shot_station:
                                # 使用记录的方位角
                                x_coords[i] = record.az
                                break
                        else:
                            # 如果找不到记录，使用道头中的方位角
                            if hasattr(th, 'azi'):
                                x_coords[i] = th.azi
                            else:
                                x_coords[i] = offsets[trace_idx] if trace_idx < len(offsets) else 0.0
                    else:
                        if hasattr(th, 'azi'):
                            x_coords[i] = th.azi
                        else:
                            x_coords[i] = offsets[trace_idx] if trace_idx < len(offsets) else 0.0
                else:
                    x_coords[i] = offsets[trace_idx] if trace_idx < len(offsets) else 0.0
            elif ixaxis_type == 5:
                # 5: 道号（使用过滤后的索引）
                x_coords[i] = float(i)
            else:
                # 默认：炮检距
                x_coords[i] = offsets[trace_idx] if trace_idx < len(offsets) else 0.0
        
        return x_coords
    
    def _apply_lod(self, data: np.ndarray, times: np.ndarray, 
                   max_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """应用 LOD（细节层次）机制：根据缩放级别减少绘制点数
        
        Args:
            data: 数据数组
            times: 时间数组
            max_points: 最大点数（如果为 None，使用 self.lod_threshold）
            
        Returns:
            (降采样后的数据, 降采样后的时间)
        """
        if not self.enable_lod:
            return data, times
        
        if max_points is None:
            max_points = self.lod_threshold
        
        if len(data) <= max_points:
            return data, times
        
        # 计算降采样步长
        step = len(data) // max_points
        if step <= 1:
            return data, times
        
        # 降采样：每隔 step 个点取一个，并保留末点（避免尾部显示被截断）
        n = min(len(data), len(times))
        if n <= 2:
            return data[:n], times[:n]
        sample_idx = np.arange(0, n, step, dtype=int)
        if sample_idx.size == 0:
            sample_idx = np.array([0, n - 1], dtype=int)
        elif sample_idx[-1] != (n - 1):
            sample_idx = np.append(sample_idx, n - 1)
        return data[sample_idx], times[sample_idx]

    def _get_dynamic_lod_points(self) -> int:
        """根据当前轴像素高度动态计算 LOD 目标点数。

        目的：
        - 在大窗口中保留更多细节
        - 在小窗口中避免无效过采样绘制
        """
        try:
            axis_h_px = float(self.ax.bbox.height)
        except Exception:
            axis_h_px = 0.0
        if axis_h_px <= 0:
            return self.lod_threshold

        # 非实时状态优先保真；实时状态适当降采样提升交互流畅
        if self._realtime_interaction_active:
            dynamic_points = int(max(4000, min(30000, axis_h_px * 6.0)))
        else:
            dynamic_points = int(max(9000, min(80000, axis_h_px * 14.0)))
        return dynamic_points

    def _build_horizontal_shade_segments(self,
                                         x_values: np.ndarray,
                                         t_values: np.ndarray,
                                         baseline_x: float,
                                         fill_positive: bool,
                                         row_step: int = 1) -> Optional[np.ndarray]:
        """构建横向阴影线段（模拟 Fortran shade 的扫描线填充策略）。

        说明：
        - 每个有效采样点只生成一条水平短线（baseline_x -> x_values[i]）
        - 用 LineCollection 批量绘制，避免 fill_betweenx 大量多边形开销
        """
        if x_values is None or t_values is None:
            return None
        if len(x_values) == 0 or len(t_values) == 0:
            return None

        n = min(len(x_values), len(t_values))
        if n <= 0:
            return None

        x = x_values[:n]
        t = t_values[:n]
        if row_step > 1:
            idx = np.arange(0, n, row_step, dtype=int)
            if idx.size == 0:
                return None
            x = x[idx]
            t = t[idx]

        segs = self.src_shade_kernel.build_segments(
            x_values=np.asarray(x, dtype=np.float64),
            t_values=np.asarray(t, dtype=np.float64),
            baseline_x=float(baseline_x),
            fill_positive=bool(fill_positive),
            row_step=int(max(1, row_step)),
        )
        if segs.shape[0] == 0:
            return None
        return segs

    def _compute_adaptive_shade_row_step(self,
                                         params: ZPlotParameters,
                                         visible_yrange: Optional[Tuple[float, float]],
                                         realtime_interaction: bool,
                                         shade_quality: str = 'balanced') -> int:
        """根据视窗与 dens 自适应计算阴影扫描行步长。"""
        row_step = 1
        try:
            dens = float(getattr(params, 'dens', 0.0) or 0.0)
        except Exception:
            dens = 0.0

        # 优先使用当前可见时间窗估算目标扫描行数，避免超密填充导致卡顿。
        if visible_yrange is not None:
            try:
                tspan = abs(float(visible_yrange[1]) - float(visible_yrange[0]))
                axis_h_px = max(1.0, float(self.ax.bbox.height))
                # 参考 Fortran：dens≈像素密度(点/mm)。这里按像素近似，dens<=0 时退化为 1.0
                effective_dens = dens if dens > 0 else 1.0
                target_rows = min(4000.0, max(600.0, tspan * effective_dens * 160.0))
                # 档位系数：快速更稀疏，高质量更致密
                if shade_quality == 'fast':
                    target_rows *= 0.9
                elif shade_quality == 'high':
                    target_rows *= 1.6
                # 以屏幕可见像素行为基准，推导行步长
                row_step = max(1, int(math.ceil(axis_h_px / target_rows)))
            except Exception:
                row_step = 1

        # 非实时状态优先保真：balanced/high 默认不跳行
        if not realtime_interaction and shade_quality in ('balanced', 'high'):
            row_step = 1
        # 实时交互阶段进一步降级，保证拖拽/缩放优先流畅
        elif realtime_interaction:
            row_step = max(row_step, 2)
        return row_step

    def _get_shade_quality(self, params: ZPlotParameters) -> str:
        """读取阴影质量档位，非法值回退为 balanced。"""
        quality = str(getattr(params, 'shade_quality_preset', self.shade_quality_preset)).strip().lower()
        if quality not in ('fast', 'balanced', 'high'):
            quality = 'balanced'
        self.shade_quality_preset = quality
        return quality

    def _add_shade_segments_collection(self,
                                       segments: np.ndarray,
                                       fill_color: str,
                                       fill_alpha: float) -> None:
        """将阴影线段批量提交为 LineCollection。"""
        if segments is None or len(segments) == 0:
            return
        shade_lc = LineCollection(
            segments,
            colors=fill_color,
            linewidths=0.3,
            linestyles='-',
            alpha=fill_alpha,
            antialiased=False
        )
        self.ax.add_collection(shade_lc)
        self.traces_artists.append(shade_lc)
    
    def plot_wiggle(self, traces: List[np.ndarray], offsets: np.ndarray,
                  times: np.ndarray, scale: float, params: ZPlotParameters,
                  filtered_indices: Optional[List[int]] = None,
                  trace_headers: Optional[List] = None,
                  visible_xrange: Optional[Tuple[float, float]] = None,
                  visible_yrange: Optional[Tuple[float, float]] = None) -> None:
        """绘制 wiggle 图（优化版本：预计算坐标，批量绘制）
        
        优化特性：
        - 预计算所有道的坐标，减少重复计算
        - 批量处理相同线型的道
        - 改进错误处理
        
        Args:
            traces: 道数据列表
            offsets: 炮检距数组
            times: 时间数组
            scale: 缩放因子
            params: 绘图参数
            filtered_indices: 过滤后的道索引列表
            trace_headers: 道头信息列表（用于确定线型）
        """
        linewidth = 0.5
        color = 'black'
        
        if not traces or len(traces) == 0:
            logger.warning("plot_wiggle: 没有道数据可绘制")
            return
        
        try:
            # 预计算所有道的坐标（优化：减少重复计算）
            n_traces = len(traces)
            x_wiggle_list = []
            plot_times_list = []
            plot_offsets_list = []
            linestyles = []
            
            for i, trace in enumerate(traces):
                # 获取绘制位置（x坐标）- 对齐模式下使用对齐后的道号位置
                plot_offset = offsets[i]
                
                # 应用 xpick 偏移（毫米转千米）
                if params.xpick != 0:
                    plot_offset = plot_offset + params.xpick / 1000.0  # 毫米转千米

                # 仅绘制当前可见 X 窗口内的道
                if visible_xrange is not None:
                    if plot_offset < visible_xrange[0] or plot_offset > visible_xrange[1]:
                        continue
                
                # 获取原始 trace_idx（用于查找对齐偏移和原始炮检距）
                trace_idx = filtered_indices[i] if filtered_indices and i < len(filtered_indices) else i
                
                # 获取原始炮检距（用于折合时间计算）
                original_offset = self.original_offsets.get(trace_idx, plot_offset)
                
                # 应用裁剪
                if params.clip > 0:
                    trace = np.clip(trace, -params.clip, params.clip)
                
                # 应用折合时间（使用原始炮检距）
                plot_times = times.copy()
                if params.vred > 0:
                    reduction = abs(original_offset) / params.vred
                    plot_times = times - reduction
                
                # 应用对齐偏移（如果该道有对齐偏移）
                # 对齐偏移量是显示时间的偏移量，直接应用到 plot_times 上
                if trace_idx in self.alignment_offsets:
                    alignment_offset = self.alignment_offsets[trace_idx]
                    plot_times = plot_times + alignment_offset  # 注意：这里是加，因为偏移量是基准时间 - 当前显示时间
                
                # 应用静校正（短波长时间变化）
                # 静校正量是短波长变化，用于增强震相相关性，不改变拾取走时
                # 只有在非预览模式下才应用到波形（预览模式下只显示校正后的震相曲线）
                if not self.static_correction_preview_mode and trace_idx in self.static_corrections:
                    static_correction = self.static_corrections[trace_idx]
                    plot_times = plot_times + static_correction
                
                # 应用 txadj 时间调整
                if params.txadj != 0:
                    plot_times = plot_times + params.txadj

                # 仅绘制与当前可见 Y 窗口有交集的道
                if visible_yrange is not None and len(plot_times) > 0:
                    tmin_i = float(np.min(plot_times))
                    tmax_i = float(np.max(plot_times))
                    if tmax_i < visible_yrange[0] or tmin_i > visible_yrange[1]:
                        continue

                # 仅绘制与当前可见 Y 窗口有交集的道
                if visible_yrange is not None and len(plot_times) > 0:
                    tmin_i = float(np.min(plot_times))
                    tmax_i = float(np.max(plot_times))
                    if tmax_i < visible_yrange[0] or tmin_i > visible_yrange[1]:
                        continue
                
                # 先应用用户抽取间隔，再做LOD，避免在过滤层抽样引起道显示异常
                trace_dec, plot_times_dec = self._apply_user_decimation(trace, plot_times, params.ndecim)
                # 用户显式设置 ndecim 时，不再叠加 LOD，避免双重抽稀
                if int(getattr(params, 'ndecim', 1) or 1) > 1:
                    trace_lod, plot_times_lod = trace_dec, plot_times_dec
                else:
                    # 应用 LOD（细节层次）：根据轴像素高度动态降采样
                    trace_lod, plot_times_lod = self._apply_lod(
                        trace_dec,
                        plot_times_dec,
                        max_points=self._get_dynamic_lod_points()
                    )
                
                # 计算 wiggle 曲线的 x 坐标（使用对齐后的道号位置）
                x_wiggle = plot_offset + trace_lod * scale
                
                # 确定线型（根据 itype 和 itypei）
                linestyle = '-'  # 默认实线
                if trace_headers and i < len(trace_headers) and params.itype < 0:
                    th = trace_headers[i]
                    if params.itype == -1:
                        # -1: 垂直(1)实线，径向(2)虚线
                        linestyle = '-' if th.itypei == 1 else '--'
                    elif params.itype == -2:
                        # -2: 径向(2)实线，横向(3)虚线
                        linestyle = '-' if th.itypei == 2 else '--'
                    elif params.itype == -3:
                        # -3: 径向(2)实线，水听器(4)虚线
                        linestyle = '-' if th.itypei == 2 else '--'
                    elif params.itype == -4:
                        # -4: 垂直(1)实线，水听器(4)虚线
                        linestyle = '-' if th.itypei == 1 else '--'
                
                # 存储预计算的坐标（使用 LOD 后的数据）
                x_wiggle_list.append(x_wiggle)
                plot_times_list.append(plot_times_lod)
                plot_offsets_list.append(plot_offset)
                linestyles.append(linestyle)
            
            if len(x_wiggle_list) == 0:
                return

            # 批量绘制（使用 LineCollection 优化性能）
            # 按线型分组，使用 LineCollection 批量绘制，性能提升 2-5 倍
            solid_indices = [i for i, ls in enumerate(linestyles) if ls == '-']
            dashed_indices = [i for i, ls in enumerate(linestyles) if ls == '--']
            
            # 批量绘制实线（使用 LineCollection）
            if solid_indices:
                try:
                    # 准备 LineCollection 数据：每条线需要 (x, y) 坐标对
                    solid_segments = []
                    for i in solid_indices:
                        # 将 x 和 y 组合成 (N, 2) 数组
                        segments = np.column_stack([x_wiggle_list[i], plot_times_list[i]])
                        solid_segments.append(segments)
                    
                    # 创建 LineCollection 并批量绘制
                    solid_lc = LineCollection(
                        solid_segments,
                        colors=color,
                        linewidths=linewidth,
                        linestyles='-',
                        antialiased=True
                    )
                    self.ax.add_collection(solid_lc)
                    self.traces_artists.append(solid_lc)
                except Exception as e:
                    logger.error(f"批量绘制实线失败: {e}", exc_info=True)
                    # 回退到逐道绘制
                    for i in solid_indices:
                        try:
                            line, = self.ax.plot(x_wiggle_list[i], plot_times_list[i], 
                                                color=color, linewidth=linewidth, linestyle='-')
                            self.traces_artists.append(line)
                        except Exception as e2:
                            logger.error(f"绘制道 {i} 失败: {e2}", exc_info=True)
            
            # 批量绘制虚线（使用 LineCollection）
            if dashed_indices:
                try:
                    # 准备 LineCollection 数据
                    dashed_segments = []
                    for i in dashed_indices:
                        segments = np.column_stack([x_wiggle_list[i], plot_times_list[i]])
                        dashed_segments.append(segments)
                    
                    # 创建 LineCollection 并批量绘制
                    dashed_lc = LineCollection(
                        dashed_segments,
                        colors=color,
                        linewidths=linewidth,
                        linestyles='--',
                        antialiased=True
                    )
                    self.ax.add_collection(dashed_lc)
                    self.traces_artists.append(dashed_lc)
                except Exception as e:
                    logger.error(f"批量绘制虚线失败: {e}", exc_info=True)
                    # 回退到逐道绘制
                    for i in dashed_indices:
                        try:
                            line, = self.ax.plot(x_wiggle_list[i], plot_times_list[i], 
                                                color=color, linewidth=linewidth, linestyle='--')
                            self.traces_artists.append(line)
                        except Exception as e2:
                            logger.error(f"绘制道 {i} 失败: {e2}", exc_info=True)
            
            # 填充区域（如果需要）
            if params.ishade != 0:
                fill_positive = params.ishade > 0
                # wiggle 路径与变面积路径统一使用内核线段填充
                fill_row_step = self._compute_adaptive_shade_row_step(
                    params=params,
                    visible_yrange=visible_yrange,
                    realtime_interaction=self._realtime_interaction_active,
                    shade_quality=self._get_shade_quality(params),
                )
                all_fill_segments = []
                pending_segments = 0
                chunk_segment_limit = 120_000 if self._realtime_interaction_active else 200_000
                for i in range(len(x_wiggle_list)):
                    try:
                        plot_offset = plot_offsets_list[i]
                        
                        x_wiggle = x_wiggle_list[i]
                        plot_times = plot_times_list[i]
                        segs = self._build_horizontal_shade_segments(
                            x_wiggle,
                            plot_times,
                            plot_offset,
                            fill_positive=fill_positive,
                            row_step=fill_row_step,
                        )
                        if segs is not None and len(segs) > 0:
                            all_fill_segments.append(segs)
                            pending_segments += len(segs)
                            if pending_segments >= chunk_segment_limit:
                                merged = np.concatenate(all_fill_segments, axis=0)
                                self._add_shade_segments_collection(
                                    merged,
                                    fill_color=color,
                                    fill_alpha=0.5,
                                )
                                all_fill_segments = []
                                pending_segments = 0
                    except Exception as e:
                        logger.error(f"填充道 {i} 失败: {e}", exc_info=True)
                if all_fill_segments:
                    try:
                        merged = np.concatenate(all_fill_segments, axis=0)
                        self._add_shade_segments_collection(
                            merged,
                            fill_color=color,
                            fill_alpha=0.5,
                        )
                    except Exception as e:
                        logger.error(f"批量绘制 wiggle 阴影线段失败: {e}", exc_info=True)
        
        except Exception as e:
            logger.error(f"plot_wiggle 执行失败: {e}", exc_info=True)
            raise
    
    def plot_variable_area(self, traces: List[np.ndarray], offsets: np.ndarray,
                          times: np.ndarray, scale: float, params: ZPlotParameters,
                          fill_positive: bool = True, filtered_indices: Optional[List[int]] = None,
                          trace_headers: Optional[List] = None,
                          visible_xrange: Optional[Tuple[float, float]] = None,
                          visible_yrange: Optional[Tuple[float, float]] = None,
                          realtime_interaction: bool = False) -> None:
        """绘制变面积图（优化版本：预计算坐标，批量绘制，LOD 支持）
        
        优化特性：
        - 预计算所有道的坐标，减少重复计算
        - 使用 LineCollection 批量绘制线
        - 应用 LOD（细节层次）机制减少绘制点数
        - 改进错误处理
        
        Args:
            traces: 道数据列表
            offsets: 炮检距数组
            times: 时间数组
            scale: 缩放因子
            params: 绘图参数
            fill_positive: 是否填充正峰值（True=正峰值，False=负峰值）
            filtered_indices: 过滤后的道索引列表（用于查找对齐偏移）
            trace_headers: 道头信息列表（用于确定线型）
        """
        linewidth = 0.3
        line_color = 'black'
        fill_color = 'black'
        fill_alpha = 0.5
        
        if not traces or len(traces) == 0:
            logger.warning("plot_variable_area: 没有道数据可绘制")
            return
        
        try:
            # 预计算所有道的坐标（优化：减少重复计算）
            x_wiggle_list = []
            plot_times_list = []
            plot_offsets_list = []
            # 填充使用更密的源点（与线条LOD解耦），避免条纹“锯齿化/稀疏化”
            x_fill_source_list = []
            t_fill_source_list = []
            linestyles = []
            
            for i, trace in enumerate(traces):
                # 获取绘制位置（x坐标）- 对齐模式下使用对齐后的道号位置
                plot_offset = offsets[i]
                
                # 应用 xpick 偏移（毫米转千米）
                if params.xpick != 0:
                    plot_offset = plot_offset + params.xpick / 1000.0  # 毫米转千米
                
                # 获取原始 trace_idx（用于查找对齐偏移和原始炮检距）
                trace_idx = filtered_indices[i] if filtered_indices and i < len(filtered_indices) else i
                
                # 获取原始炮检距（用于折合时间计算）
                original_offset = self.original_offsets.get(trace_idx, plot_offset)
                
                # 应用裁剪
                if params.clip > 0:
                    trace = np.clip(trace, -params.clip, params.clip)
                
                # 应用折合时间（使用原始炮检距）
                plot_times = times.copy()
                if params.vred > 0:
                    reduction = abs(original_offset) / params.vred
                    plot_times = times - reduction
                
                # 应用对齐偏移（如果该道有对齐偏移）
                if trace_idx in self.alignment_offsets:
                    alignment_offset = self.alignment_offsets[trace_idx]
                    plot_times = plot_times + alignment_offset
                
                # 应用静校正（短波长时间变化）
                # 静校正量是短波长变化，用于增强震相相关性，不改变拾取走时
                # 只有在非预览模式下才应用到波形（预览模式下只显示校正后的震相曲线）
                if not self.static_correction_preview_mode and trace_idx in self.static_corrections:
                    static_correction = self.static_corrections[trace_idx]
                    plot_times = plot_times + static_correction
                
                # 应用 txadj 时间调整
                if params.txadj != 0:
                    plot_times = plot_times + params.txadj
                
                # 先应用用户抽取间隔，再做LOD，避免在过滤层抽样引起道显示异常
                trace_dec, plot_times_dec = self._apply_user_decimation(trace, plot_times, params.ndecim)
                # 用户显式设置 ndecim 时，不再叠加 LOD，避免双重抽稀
                if int(getattr(params, 'ndecim', 1) or 1) > 1:
                    trace_lod, plot_times_lod = trace_dec, plot_times_dec
                else:
                    # 应用 LOD（细节层次）：根据轴像素高度动态降采样
                    trace_lod, plot_times_lod = self._apply_lod(
                        trace_dec,
                        plot_times_dec,
                        max_points=self._get_dynamic_lod_points()
                    )
                
                # 计算 wiggle 曲线的 x 坐标（使用对齐后的道号位置）
                x_wiggle = plot_offset + trace_lod * scale
                # 填充使用 decimation 后但未做 LOD 的更密轨迹
                x_fill_dense = plot_offset + trace_dec * scale
                
                # 确定线型（根据 itype 和 itypei）
                linestyle = '-'  # 默认实线
                if trace_headers and i < len(trace_headers) and params.itype < 0:
                    th = trace_headers[i]
                    if params.itype == -1:
                        linestyle = '-' if th.itypei == 1 else '--'
                    elif params.itype == -2:
                        linestyle = '-' if th.itypei == 2 else '--'
                    elif params.itype == -3:
                        linestyle = '-' if th.itypei == 2 else '--'
                    elif params.itype == -4:
                        linestyle = '-' if th.itypei == 1 else '--'
                
                # 存储预计算的坐标
                x_wiggle_list.append(x_wiggle)
                plot_times_list.append(plot_times_lod)
                plot_offsets_list.append(plot_offset)
                x_fill_source_list.append(x_fill_dense)
                t_fill_source_list.append(plot_times_dec)
                linestyles.append(linestyle)
            
            if len(x_wiggle_list) == 0:
                return

            # 批量绘制线（使用 LineCollection 优化性能）
            solid_indices = [i for i, ls in enumerate(linestyles) if ls == '-']
            dashed_indices = [i for i, ls in enumerate(linestyles) if ls == '--']
            
            # 批量绘制实线
            if solid_indices:
                try:
                    solid_segments = []
                    for i in solid_indices:
                        segments = np.column_stack([x_wiggle_list[i], plot_times_list[i]])
                        solid_segments.append(segments)
                    
                    solid_lc = LineCollection(
                        solid_segments,
                        colors=line_color,
                        linewidths=linewidth,
                        linestyles='-',
                        antialiased=True
                    )
                    self.ax.add_collection(solid_lc)
                    self.traces_artists.append(solid_lc)
                except Exception as e:
                    logger.error(f"批量绘制实线失败: {e}", exc_info=True)
                    # 回退到逐道绘制
                    for i in solid_indices:
                        try:
                            line, = self.ax.plot(x_wiggle_list[i], plot_times_list[i], 
                                                color=line_color, linewidth=linewidth, linestyle='-')
                            self.traces_artists.append(line)
                        except Exception as e2:
                            logger.error(f"绘制道 {i} 失败: {e2}", exc_info=True)
            
            # 批量绘制虚线
            if dashed_indices:
                try:
                    dashed_segments = []
                    for i in dashed_indices:
                        segments = np.column_stack([x_wiggle_list[i], plot_times_list[i]])
                        dashed_segments.append(segments)
                    
                    dashed_lc = LineCollection(
                        dashed_segments,
                        colors=line_color,
                        linewidths=linewidth,
                        linestyles='--',
                        antialiased=True
                    )
                    self.ax.add_collection(dashed_lc)
                    self.traces_artists.append(dashed_lc)
                except Exception as e:
                    logger.error(f"批量绘制虚线失败: {e}", exc_info=True)
                    # 回退到逐道绘制
                    for i in dashed_indices:
                        try:
                            line, = self.ax.plot(x_wiggle_list[i], plot_times_list[i], 
                                                color=line_color, linewidth=linewidth, linestyle='--')
                            self.traces_artists.append(line)
                        except Exception as e2:
                            logger.error(f"绘制道 {i} 失败: {e2}", exc_info=True)
            
            # 填充区域（逐道生成水平短线，最后批量绘制）
            # 策略：
            # 1) 若用户处于缩放视窗，优先保证视窗内可见道“完整填充”（不跳道/不降点）
            # 2) 其余场景下，对大文件采用自适应降级提升交互性能
            n_traces_fill = len(x_wiggle_list)
            total_fill_points = int(sum(len(x) for x in x_wiggle_list))
            fill_trace_stride = 1
            fill_point_step = 1
            shade_quality = self._get_shade_quality(params)
            fill_row_step = self._compute_adaptive_shade_row_step(
                params=params,
                visible_yrange=visible_yrange,
                realtime_interaction=realtime_interaction,
                shade_quality=shade_quality
            )
            # 质量档位影响填充预算（近似控制 traces/points 上限）
            if shade_quality == 'fast':
                target_max_fill_traces = 1500
                target_max_fill_points = 3_000_000
                chunk_segment_limit = 120_000
            elif shade_quality == 'high':
                target_max_fill_traces = 8000
                target_max_fill_points = 16_000_000
                chunk_segment_limit = 260_000
            else:
                target_max_fill_traces = 4000
                target_max_fill_points = 8_000_000
                chunk_segment_limit = 200_000
            force_full_visible_fill = visible_xrange is not None and visible_yrange is not None

            # 视窗连续拖动/缩放时，优先流畅性：加大填充降级力度
            if realtime_interaction:
                target_max_fill_traces = 1000
                target_max_fill_points = 2_000_000
                chunk_segment_limit = min(chunk_segment_limit, 120_000)
            else:
                # 静止状态优先质量：balanced/high 不主动做 trace/point 降采样
                if shade_quality in ('balanced', 'high'):
                    target_max_fill_traces = 10**9
                    target_max_fill_points = 10**9

            if not force_full_visible_fill:
                if n_traces_fill > target_max_fill_traces:
                    fill_trace_stride = int(math.ceil(n_traces_fill / target_max_fill_traces))

                effective_points = max(1, total_fill_points // fill_trace_stride)
                if effective_points > target_max_fill_points:
                    fill_point_step = int(math.ceil(effective_points / target_max_fill_points))

            zshade_msg = (
                "[ZSHADE] "
                f"traces={n_traces_fill} total_points={total_fill_points} "
                f"trace_stride={fill_trace_stride} point_step={fill_point_step} "
                f"row_step={fill_row_step} mode=horizontal_segments "
                f"quality={shade_quality} "
                f"rt={realtime_interaction} "
                f"visible_xrange={visible_xrange} visible_yrange={visible_yrange}"
            )
            print(zshade_msg)
            logger.info(zshade_msg)

            all_fill_segments = []
            pending_segments = 0
            for i in range(0, len(traces), fill_trace_stride):
                try:
                    x_wiggle = x_fill_source_list[i]
                    plot_times = t_fill_source_list[i]
                    plot_offset = plot_offsets_list[i]

                    # 仅在可见窗口内填充：X在窗口内且时间范围与窗口有交集
                    if visible_xrange is not None:
                        if plot_offset < visible_xrange[0] or plot_offset > visible_xrange[1]:
                            continue
                    if visible_yrange is not None and len(plot_times) > 0:
                        tmin_i = float(np.min(plot_times))
                        tmax_i = float(np.max(plot_times))
                        if tmax_i < visible_yrange[0] or tmin_i > visible_yrange[1]:
                            continue

                    if fill_point_step > 1:
                        x_fill = x_wiggle[::fill_point_step]
                        t_fill = plot_times[::fill_point_step]
                    else:
                        x_fill = x_wiggle
                        t_fill = plot_times

                    segs = self._build_horizontal_shade_segments(
                        x_fill, t_fill, plot_offset, fill_positive, row_step=fill_row_step
                    )
                    if segs is not None and len(segs) > 0:
                        all_fill_segments.append(segs)
                        pending_segments += len(segs)
                        # 分块提交，避免一次性 concatenate 大数组导致峰值内存/卡顿
                        if pending_segments >= chunk_segment_limit:
                            merged = np.concatenate(all_fill_segments, axis=0)
                            self._add_shade_segments_collection(merged, fill_color, fill_alpha)
                            all_fill_segments = []
                            pending_segments = 0
                except Exception as e:
                    logger.error(f"填充道 {i} 失败: {e}", exc_info=True)

            # 提交尾批次
            if all_fill_segments:
                try:
                    merged_segments = np.concatenate(all_fill_segments, axis=0)
                    self._add_shade_segments_collection(merged_segments, fill_color, fill_alpha)
                except Exception as e:
                    logger.error(f"批量绘制阴影线段失败: {e}", exc_info=True)
        
        except Exception as e:
            logger.error(f"plot_variable_area 执行失败: {e}", exc_info=True)
            raise
    
    def _filter_data(self, traces: List[np.ndarray], offsets: np.ndarray,
                    times: np.ndarray, params: ZPlotParameters,
                    trace_headers: Optional[List] = None) -> Dict[str, Any]:
        """过滤数据（根据参数）
        
        Args:
            traces: 道数据列表
            offsets: 炮检距数组
            times: 时间数组
            params: 绘图参数
            trace_headers: 道头信息列表（可选）
            
        Returns:
            过滤后的数据字典
        """
        filtered_traces = []
        filtered_offsets = []
        filtered_indices = []
        drop_reasons: Dict[str, int] = {
            "index_oob": 0,
            "xmin": 0,
            "xmax": 0,
            "itype": 0,
            "imute": 0,
        }
        nskip_drop = 0
        
        # 根据记录号过滤（如果提供了道头信息）
        start_idx = 0
        end_idx = len(traces)
        
        # 根据记录号过滤（如果提供了道头信息）
        if trace_headers and len(trace_headers) > 0:
            # 如果用户指定了记录号（irec > 0），只显示该记录的道
            if params.irec > 0:
                # 收集属于指定记录的道索引
                matching_indices = []
                for i, th in enumerate(trace_headers):
                    if i >= len(traces):
                        break
                    # 检查记录号是否匹配
                    if th.ishoti == params.irec:
                        matching_indices.append(i)
                
                # 如果找到了匹配的道，使用这些道
                if matching_indices:
                    trace_indices = matching_indices
                else:
                    # 如果没有找到匹配的道，返回空结果
                    # 调用者会显示提示信息并清空绘图区
                    msg = (
                        "[ZPLOT_FILTER] "
                        f"irec={params.irec} no-match "
                        f"input_traces={len(traces)} headers={len(trace_headers)} "
                        f"itype={params.itype} imute={params.imute} "
                        f"xmin={params.xmin} xmax={params.xmax}"
                    )
                    print(msg)
                    logger.info(msg)
                    return {
                        'traces': [],
                        'offsets': [],
                        'times': times,
                        'indices': [],
                        'filter_stats': {
                            'total_candidates': 0,
                            'drop_reasons': drop_reasons,
                            'nskip_drop': 0
                        }
                    }
            else:
                # irec <= 0 表示显示所有记录
                trace_indices = list(range(len(traces)))
        else:
            # 没有道头信息，使用所有道
            trace_indices = list(range(len(traces)))

        total_candidates = len(trace_indices)
        
        # 向量化早筛：先筛索引，再抓取 trace 引用，减少 Python 循环开销
        if len(trace_indices) > 0:
            trace_idx_arr = np.asarray(trace_indices, dtype=int)
            valid_mask = (trace_idx_arr >= 0) & (trace_idx_arr < len(traces)) & (trace_idx_arr < len(offsets))
            drop_reasons["index_oob"] += int(np.count_nonzero(~valid_mask))
            trace_idx_arr = trace_idx_arr[valid_mask]

            if trace_idx_arr.size > 0:
                pass_mask = np.ones(trace_idx_arr.shape[0], dtype=bool)

                # 有头信息时，继续按 itype/imute 预筛
                if trace_headers and len(trace_headers) > 0:
                    header_valid = trace_idx_arr < len(trace_headers)
                    if np.any(header_valid):
                        header_indices = trace_idx_arr[header_valid]
                        if params.itype != 0:
                            itype_arr = np.asarray([trace_headers[idx].itypei for idx in header_indices], dtype=int)
                            if params.itype > 0:
                                allow = (itype_arr == params.itype)
                            elif params.itype == -1:
                                allow = np.isin(itype_arr, [1, 2])
                            elif params.itype == -2:
                                allow = np.isin(itype_arr, [2, 3])
                            elif params.itype == -3:
                                allow = np.isin(itype_arr, [2, 4])
                            elif params.itype == -4:
                                allow = np.isin(itype_arr, [1, 4])
                            else:
                                allow = np.ones_like(itype_arr, dtype=bool)
                            itype_fail = ~allow
                            drop_reasons["itype"] += int(np.count_nonzero(itype_fail))
                            header_pass_idx = np.where(header_valid)[0]
                            pass_mask[header_pass_idx[itype_fail]] = False

                        if params.imute != 0:
                            iflag_arr = np.asarray([trace_headers[idx].iflagi for idx in header_indices], dtype=int)
                            if params.imute == 1:
                                allow = (iflag_arr == 1)
                            else:
                                allow = (iflag_arr != 1)
                            imute_fail = ~allow
                            drop_reasons["imute"] += int(np.count_nonzero(imute_fail))
                            header_pass_idx = np.where(header_valid)[0]
                            pass_mask[header_pass_idx[imute_fail]] = False

                passed_indices = trace_idx_arr[pass_mask]
                if passed_indices.size > 0:
                    filtered_indices = passed_indices.tolist()
                    filtered_offsets = np.asarray(offsets)[passed_indices].tolist()
                    filtered_traces = [traces[idx] for idx in filtered_indices]

        # 应用 nskip（跳过的道数）
        # 改进：先按分量/窗口/死道过滤，再进行跳道，避免 nskip 把某分量“抽没”后误判为分量错误
        profile = os.environ.get('ZPLOTPY_PARAM_PROFILE', 'modern').strip().lower()
        if params.nskip > 0 and len(filtered_traces) > 0:
            if profile == 'fortran':
                step = max(1, int(params.nskip))
            else:
                step = int(params.nskip) + 1
            before_nskip = len(filtered_traces)
            keep_indices = list(range(0, before_nskip, step))
            filtered_traces = [filtered_traces[k] for k in keep_indices]
            filtered_offsets = [filtered_offsets[k] for k in keep_indices]
            filtered_indices = [filtered_indices[k] for k in keep_indices]
            nskip_drop = before_nskip - len(filtered_traces)
        
        # 读取完整时间序列，不在过滤层按 tmin/tmax 截断；
        # 统一在绘制层基于当前坐标窗口决定显示范围。
        filtered_times = times
        
        msg = (
            "[ZPLOT_FILTER] "
            f"irec={params.irec} "
            f"in={len(traces)} candidates={total_candidates} pass={len(filtered_traces)} "
            f"drops={drop_reasons}, nskip_drop={nskip_drop} "
            f"itype={params.itype} imute={params.imute} "
            f"xmin={params.xmin} xmax={params.xmax} "
            f"trace_headers={len(trace_headers) if trace_headers else 0}"
        )
        print(msg)
        logger.info(msg)
        
        return {
            'traces': filtered_traces,
            'offsets': np.array(filtered_offsets),
            'times': filtered_times,
            'indices': filtered_indices,
            'filter_stats': {
                'total_candidates': total_candidates,
                'drop_reasons': drop_reasons,
                'nskip_drop': nskip_drop
            }
        }

    def _apply_user_decimation(self, data: np.ndarray, times: np.ndarray, ndecim: int) -> Tuple[np.ndarray, np.ndarray]:
        """应用用户设置的数据抽取间隔（仅用于绘制层，不改变过滤层数据）。

        关键点：保留首尾点，避免抽取后时间窗尾部丢失引发“道截断”视觉问题。
        """
        try:
            step = int(ndecim)
        except Exception:
            step = 1
        if step <= 1 or len(data) <= 2 or len(times) <= 2:
            return data, times

        n = min(len(data), len(times))
        if n <= 2:
            return data[:n], times[:n]

        # 按固定步长逐道抽点，并强制保留末点，避免等距取整造成局部形态异常
        sample_idx = np.arange(0, n, step, dtype=int)
        if sample_idx.size == 0:
            sample_idx = np.array([0, n - 1], dtype=int)
        elif sample_idx[-1] != (n - 1):
            sample_idx = np.append(sample_idx, n - 1)
        return data[sample_idx], times[sample_idx]
    
    def _filter_visible_range(self, filtered_data: Dict[str, Any],
                             visible_xmin: Optional[float], visible_xmax: Optional[float],
                             visible_ymin: Optional[float], visible_ymax: Optional[float]) -> Dict[str, Any]:
        """根据可见窗口范围进一步过滤数据（性能优化）
        
        Args:
            filtered_data: 已经过滤的数据字典
            visible_xmin: 可见窗口X轴最小值
            visible_xmax: 可见窗口X轴最大值
            visible_ymin: 可见窗口Y轴最小值
            visible_ymax: 可见窗口Y轴最大值
            
        Returns:
            进一步过滤后的数据字典
        """
        filtered_traces = filtered_data['traces']
        filtered_offsets = filtered_data['offsets']
        filtered_times = filtered_data['times']
        filtered_indices = filtered_data.get('indices', list(range(len(filtered_traces))))
        
        # 如果所有范围都是None，不需要进一步过滤
        if visible_xmin is None and visible_xmax is None and visible_ymin is None and visible_ymax is None:
            return filtered_data
        
        # 根据X轴范围过滤道
        visible_traces = []
        visible_offsets = []
        visible_indices = []
        
        for i, offset in enumerate(filtered_offsets):
            # X轴范围过滤
            if visible_xmin is not None and offset < visible_xmin:
                continue
            if visible_xmax is not None and offset > visible_xmax:
                continue
            
            # 添加到可见列表
            visible_traces.append(filtered_traces[i])
            visible_offsets.append(offset)
            visible_indices.append(filtered_indices[i] if i < len(filtered_indices) else i)
        
        # 根据Y轴范围过滤时间
        if len(visible_traces) == 0:
            return {
                'traces': [],
                'offsets': np.array([]),
                'times': filtered_times,
                'indices': []
            }
        
        # 时间范围过滤
        if visible_ymin is not None or visible_ymax is not None:
            time_mask = np.ones(len(filtered_times), dtype=bool)
            if visible_ymin is not None:
                time_mask &= filtered_times >= visible_ymin
            if visible_ymax is not None:
                time_mask &= filtered_times <= visible_ymax
            
            # 如果时间范围有变化，对每道数据应用时间过滤
            if np.any(~time_mask):
                visible_traces = [trace[time_mask] for trace in visible_traces]
                visible_times = filtered_times[time_mask]
            else:
                visible_times = filtered_times
        else:
            visible_times = filtered_times
        
        return {
            'traces': visible_traces,
            'offsets': np.array(visible_offsets),
            'times': visible_times,
            'indices': visible_indices
        }
    
    def _plot_removed_trace_markers(self, filtered_offsets: np.ndarray, 
                                   filtered_times: np.ndarray,
                                   params: ZPlotParameters) -> None:
        """绘制被移除trace的顶部标记（X键功能）
        
        在数据窗口顶部绘制红色圆圈，标记被移除的trace位置
        
        Args:
            filtered_offsets: 过滤后的offsets数组
            filtered_times: 时间数组
            params: 绘图参数
        """
        if not self.removed_traces:
            # 如果没有被移除的trace，清除所有标记
            for marker in self.removed_trace_markers:
                try:
                    marker.remove()
                except:
                    pass
            self.removed_trace_markers.clear()
            return
        
        try:
            # 清除之前的标记
            for marker in self.removed_trace_markers:
                try:
                    marker.remove()
                except:
                    pass
            self.removed_trace_markers.clear()
            
            # 获取Y轴范围，在顶部绘制标记
            ylim = self.ax.get_ylim()
            if ylim[0] >= ylim[1]:
                return
            
            # 计算标记的Y位置（在窗口顶部）
            # 根据itrev参数和Y轴设置确定顶部位置
            # 注意：matplotlib的get_ylim()总是返回(min, max)，其中min < max
            # 但显示位置取决于Y轴是否反转：
            # - itrev=1: 正常Y轴，ylim[1]显示在顶部
            # - itrev=0: Y轴反转，ylim[0]显示在顶部（但值较大）
            
            y_range = abs(ylim[1] - ylim[0])
            if y_range == 0:
                return
            
            if params.itrev == 1:
                # itrev=1: 正常Y轴，ylim[1]是顶部
                top_y = ylim[1]
            else:
                # itrev=0: Y轴反转，ylim[0]是顶部（值较大，但显示在顶部）
                top_y = ylim[0]
            
            # 在顶部上方2%的位置绘制标记
            marker_y = top_y + y_range * 0.02
            
            # 绘制每个被移除trace的标记
            for trace_idx, offset in self.removed_traces.items():
                # 应用 xpick 偏移（如果设置了）
                plot_offset = offset
                if params.xpick != 0:
                    plot_offset = offset + params.xpick / 1000.0  # 毫米转千米
                
                # 检查offset是否在当前显示范围内
                xlim = self.ax.get_xlim()
                if plot_offset < xlim[0] or plot_offset > xlim[1]:
                    continue  # 不在显示范围内，跳过
                
                # 绘制红色圆圈标记（使用更大的标记以便清晰可见）
                marker, = self.ax.plot(plot_offset, marker_y, 'o', 
                                      color='red', markersize=6, 
                                      markeredgewidth=1.0, markeredgecolor='darkred',
                                      fillstyle='full',  # 实心圆圈
                                      zorder=1000)  # 确保在最上层
                self.removed_trace_markers.append(marker)
        
        except Exception as e:
            logger.error(f"绘制移除trace标记失败: {e}", exc_info=True)
    
    def _calculate_scale(self, traces: List[np.ndarray], offsets: np.ndarray,
                       params: ZPlotParameters) -> float:
        """计算缩放因子
        
        Args:
            traces: 道数据列表（应该是处理后的数据）
            offsets: 炮检距数组
            params: 绘图参数
            
        Returns:
            缩放因子
        """
        if not traces or len(traces) == 0:
            return 1.0
        
        # 计算道间距（使用稳健统计，避免单个异常极小间距把振幅压扁）
        if len(offsets) > 1:
            unique_offsets = np.unique(offsets)
            if len(unique_offsets) > 1:
                unique_diffs = np.diff(np.sort(unique_offsets))
                positive_diffs = unique_diffs[unique_diffs > 0]
                if len(positive_diffs) > 0:
                    trace_spacing = float(np.median(positive_diffs))
                    if trace_spacing < 1e-4:
                        trace_spacing = float(np.percentile(positive_diffs, 75))
                    if trace_spacing < 1e-4:
                        trace_spacing = 1.0
                else:
                    trace_spacing = 1.0
            else:
                trace_spacing = 1.0
            if trace_spacing == 0:
                trace_spacing = 1.0
        else:
            trace_spacing = 1.0
        
        # 计算所有道的最大振幅（使用处理后的数据）
        # 检查是否有有效的道数据
        valid_traces = [trace for trace in traces if len(trace) > 0]
        if len(valid_traces) == 0:
            # 如果没有有效道，返回默认缩放因子
            if params.iscale == 1 and params.amp > 0:
                return params.amp
            else:
                return 1.0
        
        peak_amps = np.array([float(np.max(np.abs(trace))) for trace in valid_traces], dtype=float)
        max_amp = float(np.max(peak_amps)) if len(peak_amps) > 0 else 1.0
        robust_amp = float(np.percentile(peak_amps, 95)) if len(peak_amps) > 0 else max_amp
        if robust_amp <= 0:
            robust_amp = max_amp
        if robust_amp <= 0:
            robust_amp = 1.0
        
        if params.iscale == 0:
            # 自动缩放：根据道间距和数据范围自动计算
            # 缩放因子：使最大振幅占道间距的 80%
            scale = (trace_spacing * 0.8) / robust_amp
            
        elif params.iscale == 1:
            # 固定缩放：如果 amp 为默认值或太小，基于数据自动计算
            # 否则使用 amp 参数
            if params.amp <= 0 or params.amp > 1000:
                # amp 无效或异常大，使用自动缩放
                scale = (trace_spacing * 0.8) / robust_amp
            else:
                # 使用 amp，但确保缩放因子合理
                # 如果 amp 相对于数据值太小，进行自适应调整
                auto_scale = (trace_spacing * 0.8) / robust_amp
                # 如果 amp 和自动缩放差异很大，使用自动缩放
                if abs(params.amp - auto_scale) / max(auto_scale, 1e-10) > 10:
                    scale = auto_scale
                else:
                    scale = params.amp
            
        else:
            # 变增益：使用 amp 参数作为基础，后续可以扩展
            # 同样需要检查 amp 是否合理
            if params.amp <= 0 or params.amp > 1000:
                scale = (trace_spacing * 0.8) / robust_amp
            else:
                scale = params.amp
        
        # 应用缩放因子 sf
        if params.sf != 0:
            scale *= params.sf

        zmsg = (
            "[ZSCALE] "
            f"traces={len(valid_traces)} spacing={trace_spacing:.6g} "
            f"amp_max={max_amp:.6g} amp_p95={robust_amp:.6g} "
            f"iscale={params.iscale} amp={params.amp:.6g} sf={params.sf:.6g} scale={scale:.6g}"
        )
        print(zmsg)
        logger.info(zmsg)
        
        return scale
    
    def _apply_reduction(self, times: np.ndarray, offsets: np.ndarray, 
                        vred: float) -> np.ndarray:
        """应用折合时间
        
        Args:
            times: 时间数组
            offsets: 炮检距数组
            vred: 折合速度 (km/s)
            
        Returns:
            折合后的时间数组
        """
        if vred <= 0:
            return times
        
        # 对每道应用折合时间
        # t_reduced = t - offset / vred
        # 这里返回一个平均折合时间（简化处理）
        # 实际应用中，每道应该有不同的折合时间
        avg_offset = np.mean(np.abs(offsets)) if len(offsets) > 0 else 0.0
        reduction = avg_offset / vred
        
        return times - reduction
    
    def _setup_axes(self, offsets: np.ndarray, times: np.ndarray, 
                   params: ZPlotParameters,
                   preserve_zoom: bool = False,
                   user_xlim: Optional[Tuple[float, float]] = None,
                   user_ylim: Optional[Tuple[float, float]] = None) -> None:
        """设置坐标轴
        
        Args:
            offsets: 炮检距数组
            times: 时间数组
            params: 绘图参数
            preserve_zoom: 是否保留用户设置的缩放
            user_xlim: 用户设置的 X 轴范围 (x_min, x_max)
            user_ylim: 用户设置的 Y 轴范围 (y_min, y_max)
        """
        # 设置 X 轴标签
        # 对齐模式下仍然使用原始的X轴类型（offset等），不改变标签
        # ✅ 使用绝对值获取标签（符号只影响反转）
        ixaxis_abs = abs(params.ixaxis)
        xlabel_map = {
            1: 'Offset (km)',
            2: 'Model Position (km)',
            3: 'Azimuth (deg)',
            4: 'Corrected Azimuth (deg)',
            5: 'Trace Number'
        }
        xlabel = xlabel_map.get(ixaxis_abs, 'Offset (km)')
        self.ax.set_xlabel(xlabel, fontsize=10)
        
        # 设置 Y 轴标签
        if params.vred > 0:
            self.ax.set_ylabel(f'Reduced Time (s, v={params.vred:.1f} km/s)', fontsize=10)
        else:
            self.ax.set_ylabel('Time (s)', fontsize=10)
        
        # 设置坐标轴范围
        # 如果用户设置了缩放，优先使用用户设置的范围
        if preserve_zoom and user_xlim and user_ylim:
            # 使用用户设置的缩放范围
            # ✅ 根据ixaxis的符号决定是否反转X轴
            if params.ixaxis > 0:  # 正数表示反转X轴
                x_min, x_max = user_xlim
                self.ax.set_xlim(x_max, x_min)
            else:
                self.ax.set_xlim(user_xlim)
            # 注意：如果itrev参数改变，需要重新设置ylim以反映反转状态
            if params.itrev == 1:
                # itrev=1: 时间反转，需要反转ylim
                y_min, y_max = user_ylim
                self.ax.set_ylim(y_max, y_min)  # 反转
            else:
                self.ax.set_ylim(user_ylim)
        else:
            # 设置坐标轴范围：优先使用用户设置的参数值，如果没有设置则使用数据范围
            # 如果用户明确设置了xmin和xmax，直接使用（控制绘制区域）
            # 修复：允许负偏移距，去掉 xmin > 0 和 xmax > 0 的限制
            if params.xmin is not None and \
               params.xmax is not None and \
               params.xmin < params.xmax:
                # 用户已设置xmin/xmax，直接使用，不添加边距（保持精确范围）
                # 这确保了加载时设置的±50km范围始终生效
                # ✅ 根据ixaxis的符号决定是否反转X轴
                if params.ixaxis > 0:  # 正数表示反转X轴
                    self.ax.set_xlim(params.xmax, params.xmin)
                else:
                    self.ax.set_xlim(params.xmin, params.xmax)
            elif len(offsets) > 0:
                # 否则使用数据范围（包括负值）
                # 注意：这里的offsets参数实际上是x_coordinates（从调用处传入，已根据ixaxis计算）
                data_x_min = np.min(offsets)
                data_x_max = np.max(offsets)
                
                # 添加一些边距
                x_range = data_x_max - data_x_min
                if x_range > 0:
                    x_padding = x_range * 0.1  # 10% 边距
                    # ✅ 根据ixaxis的符号决定是否反转X轴
                    if params.ixaxis > 0:  # 正数表示反转X轴
                        self.ax.set_xlim(data_x_max + x_padding, data_x_min - x_padding)
                    else:
                        self.ax.set_xlim(data_x_min - x_padding, data_x_max + x_padding)
                else:
                    if params.ixaxis > 0:  # 正数表示反转X轴
                        self.ax.set_xlim(data_x_max + 1, data_x_min - 1)
                    else:
                        self.ax.set_xlim(data_x_min - 1, data_x_max + 1)
            
            if len(times) > 0:
                data_t_min = np.min(times)
                data_t_max = np.max(times)
                
                # 如果用户明确设置了tmin和tmax，直接使用（控制绘制区域）
                use_user_time_range = (
                    params.tmin is not None and params.tmin >= 0 and
                    params.tmax is not None and params.tmax > 0 and
                    params.tmin < params.tmax
                )

                # 关键保护：若用户时间窗与当前数据时间窗完全不重叠，
                # 则自动回退到数据范围，避免“有波形但看不到”。
                # 典型场景：启用折合时间(vred>0)后，波形整体平移到负时间区，
                # 但默认 tmin/tmax 仍是 0~20。
                if use_user_time_range:
                    user_tmin = float(params.tmin)
                    user_tmax = float(params.tmax)
                    no_overlap = user_tmax < data_t_min or user_tmin > data_t_max
                    if no_overlap:
                        use_user_time_range = False

                if use_user_time_range:
                    t_min = params.tmin
                    t_max = params.tmax
                else:
                    # 否则使用数据范围
                    t_min = data_t_min
                    t_max = data_t_max
                
                # Y轴设置：根据itrev参数决定是否反转
                t_range = t_max - t_min
                if t_range > 0:
                    t_padding = t_range * 0.05  # 5% 边距
                    if params.itrev == 1:
                        # itrev=1: 时间反转（时间向上，正常显示）
                        self.ax.set_ylim(t_min - t_padding, t_max + t_padding)
                    else:
                        # itrev=0: 默认（时间向下，地震图标准显示）
                        self.ax.set_ylim(t_max + t_padding, t_min - t_padding)
                else:
                    # 如果没有足够的数据范围，使用默认范围
                    if params.itrev == 1:
                        self.ax.set_ylim(t_min - 0.1, t_max + 0.1)
                    else:
                        self.ax.set_ylim(t_max + 0.1, t_min - 0.1)
        
        # 设置网格
        self.ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # 设置标题
        if params.title and params.title.strip():
            # 如果用户设置了title参数，使用用户设置的标题
            title = params.title.strip()
        else:
            # 否则使用默认标题
            title = f'ZPLOT - Record {params.irec}'
            if params.vred > 0:
                title += f' (Reduced, v={params.vred:.1f} km/s)'
        
        # 添加静校正状态到标题
        if self.static_corrections:
            correction_values = list(self.static_corrections.values())
            min_corr = min(correction_values)
            max_corr = max(correction_values)
            mean_corr = sum(correction_values) / len(correction_values)
            title += f' | Static Correction: {len(self.static_corrections)} traces, range [{min_corr:.3f}, {max_corr:.3f}]s, mean {mean_corr:.3f}s'
        
        self.ax.set_title(title, fontsize=12, fontweight='bold')
    
    def plot_picks(self, picks: Dict[int, Dict[int, float]], offsets: np.ndarray,
                  params: ZPlotParameters, trace_indices: Optional[List[int]] = None,
                  alignment_offsets: Optional[Dict[int, float]] = None,
                  x_coordinates: Optional[np.ndarray] = None) -> None:
        """绘制拾取点
        
        注意：被移除的trace（X键功能）的picks不会被绘制
        """
        """绘制拾取点
        
        Args:
            picks: 拾取字典 {trace_idx: {pick_word: time}}
            offsets: 炮检距数组
            params: 绘图参数
            trace_indices: 当前显示的道索引列表（可选）
        """
        # 先清理旧拾取点，避免轻量刷新时不断叠加
        if hasattr(self, 'pick_artists') and self.pick_artists:
            for artist in list(self.pick_artists):
                try:
                    artist.remove()
                except Exception:
                    pass
            self.pick_artists = []

        if not picks:
            return
        
        # 获取拾取颜色（pickc 是颜色索引列表）
        pick_colors = params.pickc if hasattr(params, 'pickc') and params.pickc else list(range(2, 42))
        
        # 定义颜色映射（使用 matplotlib 的颜色）
        # 扩展颜色映射以支持更多颜色（对应 pickc 值 2-41）
        # pickc 值从2开始，所以 color_map[0] 对应 pickc=2
        color_map = [
            'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 
            'olive', 'cyan', 'magenta', 'yellow', 'lime', 'indigo', 'violet', 'tan',
            'coral', 'teal', 'navy', 'maroon', 'gold', 'silver', 'darkgreen', 
            'darkblue', 'darkred', 'darkorange', 'darkviolet', 'darkcyan', 'darkgray',
            'lightblue', 'lightgreen', 'lightcoral', 'lightpink', 'lightyellow',
            'lightgray', 'chocolate', 'crimson', 'forestgreen', 'royalblue', 
            'sienna', 'slategray', 'khaki', 'lavender', 'salmon', 'turquoise',
            'peru', 'plum', 'steelblue', 'tomato', 'wheat', 'yellowgreen'
        ]
        
        # 计算拾取点大小（基于 spick 参数）
        # 根据原始 zplot 代码，spick 控制拾取符号大小
        # 原始代码使用 pwidth*8，其中 pwidth 基于 spick
        # 增大拾取点大小，使其更明显
        if params.spick > 0:
            # spick 单位是毫米，转换为点（points），然后放大
            # 1 毫米 ≈ 2.83465 点，原始代码使用 *8，我们使用更大的倍数
            spick_points = params.spick * 2.83465 * 8  # 放大8倍，使其更明显
        else:
            spick_points = 40.0  # 默认大小
        
        # ✅ 收集校正前的拾取点数据（用于生成平滑曲线）
        original_pick_data = {}  # {pick_word: [(x, y), ...]} - 校正前的拾取点
        corrected_pick_data = {}  # {pick_word: [(x, y), ...]} - 校正后的拾取点（用于验证）
        
        for trace_idx, trace_picks in picks.items():
            # 如果提供了 trace_indices，检查 trace_idx 是否在其中
            if trace_indices is not None and trace_idx not in trace_indices:
                continue
            
            if trace_idx >= len(offsets):
                continue
            
            # 获取原始炮检距（用于折合时间计算）
            # 注意：折合时间必须使用原始炮检距，而不是对齐后的道号位置
            original_offset = offsets[trace_idx] if trace_idx < len(offsets) else 0.0
            
            # ✅ 使用计算后的X轴坐标（如果提供了）
            # 优先使用传入的x_coordinates，否则使用存储的current_x_coordinates
            if x_coordinates is None:
                x_coordinates = self.current_x_coordinates
            if trace_indices is None:
                trace_indices = self.current_filtered_indices
            
            # 在对齐模式下，使用新的道号位置作为x坐标（用于绘制位置）
            if x_coordinates is not None and trace_indices is not None:
                # 找到trace_idx在trace_indices中的位置
                try:
                    idx_in_filtered = trace_indices.index(trace_idx)
                    if idx_in_filtered < len(x_coordinates):
                        plot_offset = x_coordinates[idx_in_filtered]
                    else:
                        # 如果索引超出范围，使用原始offset
                        if trace_idx in self.aligned_trace_to_position:
                            plot_offset = self.aligned_trace_to_position[trace_idx]
                        else:
                            plot_offset = offsets[trace_idx] if trace_idx < len(offsets) else 0.0
                except ValueError:
                    # trace_idx不在trace_indices中，使用原始offset
                    if trace_idx in self.aligned_trace_to_position:
                        plot_offset = self.aligned_trace_to_position[trace_idx]
                    else:
                        plot_offset = offsets[trace_idx] if trace_idx < len(offsets) else 0.0
            elif trace_idx in self.aligned_trace_to_position:
                plot_offset = self.aligned_trace_to_position[trace_idx]
            else:
                plot_offset = offsets[trace_idx] if trace_idx < len(offsets) else 0.0
            
            # 应用 xpick 偏移（毫米转千米）
            if params.xpick != 0:
                plot_offset = plot_offset + params.xpick / 1000.0  # 毫米转千米
            
            # 应用 xpick 偏移（毫米转千米）
            # xpick 是毫米单位，需要转换为千米
            if params.xpick != 0:
                plot_offset = plot_offset + params.xpick / 1000.0  # 毫米转千米
            
            # 跳过被移除的trace（X键功能）
            if trace_idx in self.removed_traces:
                continue
            
            for pick_word, time in trace_picks.items():
                if time <= 0:  # 无效拾取
                    continue
                
                # 确保 pick_word 是整数（可能是 numpy.float64）
                pick_word = int(pick_word)
                
                # ✅ 计算校正后的拾取时间（先应用静校正，再应用其他变换）
                # 静校正量是基于原始拾取时间计算的，所以应该先加到原始时间上
                corrected_time = time
                if trace_idx in self.static_corrections:
                    static_correction = self.static_corrections[trace_idx]
                    corrected_time = time + static_correction
                
                # 应用折合时间（使用原始炮检距）
                plot_time = time
                corrected_plot_time = corrected_time  # 校正后的时间
                
                if params.vred > 0:
                    reduction = abs(original_offset) / params.vred
                    plot_time = time - reduction
                    corrected_plot_time = corrected_time - reduction
                
                # 应用对齐偏移（如果该道有对齐偏移）
                # 对齐偏移量是显示时间的偏移量，直接应用到 plot_time 上
                if alignment_offsets and trace_idx in alignment_offsets:
                    alignment_offset = alignment_offsets[trace_idx]
                    plot_time = plot_time + alignment_offset
                    corrected_plot_time = corrected_plot_time + alignment_offset
                
                # 应用 txadj 时间调整
                if params.txadj != 0:
                    plot_time = plot_time + params.txadj
                    corrected_plot_time = corrected_plot_time + params.txadj
                
                # 应用静校正到实际拾取点（只有在非预览模式下）
                # 预览模式下，拾取点保持原位置，只绘制校正后的虚线
                if not self.static_correction_preview_mode and trace_idx in self.static_corrections:
                    # 在非预览模式下，使用校正后的时间
                    plot_time = corrected_plot_time
                
                # 选择颜色（根据 pickc 参数）
                # pick_word 从1开始，pickc 索引也从0开始
                # pickc 值从2开始（例如 pickc[0]=2 对应 pick_word=1）
                if pick_word <= len(pick_colors):
                    # 获取该拾取字对应的 pickc 值
                    pickc_value = pick_colors[pick_word - 1]
                    # pickc 值从2开始，所以 color_map[0] 对应 pickc=2
                    color_idx = pickc_value - 2
                else:
                    # 如果 pick_word 超出 pickc 范围，使用循环映射
                    color_idx = (pick_word - 1) % len(color_map)
                
                # 确保颜色索引有效
                if color_idx < 0:
                    color_idx = 0
                elif color_idx >= len(color_map):
                    color_idx = color_idx % len(color_map)
                
                color = color_map[color_idx]
                
                # 判断是否是活动拾取字
                is_active = (pick_word == params.apick)
                
                # 设置标记和大小
                if is_active:
                    # 活动拾取字：使用圆形标记，更大，更明显
                    marker = 'o'
                    marker_size = spick_points * 1.5  # 活动拾取字更大（约60点）
                    edge_width = 2.5  # 更粗的边框
                    edge_color = 'black'
                    alpha = 1.0  # 完全不透明
                    zorder = 15  # 更高的层级，确保在最上层
                else:
                    # 非活动拾取字：使用方形标记，也较大以便可见
                    marker = 's'
                    marker_size = spick_points * 1.2  # 增大非活动拾取字大小（约48点）
                    edge_width = 2.0  # 较粗的边框
                    edge_color = 'black'
                    alpha = 0.9  # 稍微透明但更明显
                    zorder = 12  # 较高的层级
                
                # 绘制拾取点
                line = self.ax.plot(plot_offset, plot_time, marker=marker, color=color, 
                                    markersize=marker_size, markeredgewidth=edge_width,
                                    markeredgecolor=edge_color, zorder=zorder, alpha=alpha,
                                    label=f'Pick {pick_word}' if is_active else None)[0]
                self.pick_artists.append(line)
                
                # ✅ 如果存在静校正量且是预览模式，收集校正前的拾取点数据（用于生成平滑曲线）
                # 只收集当前活动拾取字的数据
                if (self.static_correction_preview_mode and 
                    trace_idx in self.static_corrections and 
                    pick_word == params.apick):
                    # 收集校正前的拾取点（plot_time，不应用静校正）
                    if pick_word not in original_pick_data:
                        original_pick_data[pick_word] = []
                    original_pick_data[pick_word].append((plot_offset, plot_time))
                    
                    # 同时收集校正后的拾取点（用于验证）
                    if pick_word not in corrected_pick_data:
                        corrected_pick_data[pick_word] = []
                    corrected_pick_data[pick_word].append((plot_offset, corrected_plot_time))
        
        # ✅ 清除之前的静校正预览曲线（虚线）
        lines_before_clear = len(self.ax.lines)
        
        # 方法1：通过跟踪列表清除
        if hasattr(self, 'static_correction_curve_lines'):
            removed_count = 0
            for line in list(self.static_correction_curve_lines):  # 使用 list() 创建副本
                try:
                    if line in self.ax.lines:
                        line.remove()  # ✅ 只使用 line.remove()，不要使用 self.ax.lines.remove()
                        removed_count += 1
                except (ValueError, AttributeError):
                    pass
            self.static_correction_curve_lines = []
        
        # 方法2：通过检查所有线条清除（备用方法，确保清除所有相关曲线）
        if self.static_correction_preview_mode:
            lines_to_remove = []
            for line in list(self.ax.lines):  # 使用 list() 创建副本，避免迭代时修改
                try:
                    label = line.get_label() or ''
                    if line.get_linestyle() == '--' and '静校正后震相曲线' in label:
                        lines_to_remove.append(line)
                except:
                    pass
            
            for line in lines_to_remove:
                try:
                    if line in self.ax.lines:
                        line.remove()  # ✅ 只使用 line.remove()，不要使用 self.ax.lines.remove()
                except (ValueError, AttributeError):
                    pass
        
        # ✅ 绘制校正后的震相曲线（虚线，平滑曲线）
        # 基于校正前的拾取震相生成平滑曲线，这就是校正后的震相曲线
        if self.static_correction_preview_mode and original_pick_data:
            # 只绘制当前活动拾取字的校正后曲线
            active_pick_word = params.apick
            if active_pick_word in original_pick_data:
                # ✅ 使用校正前的拾取点数据生成平滑曲线
                original_points = original_pick_data[active_pick_word]
                if len(original_points) >= 2:
                    # 按x坐标排序
                    points_sorted = sorted(original_points, key=lambda p: p[0])
                    x_coords = np.array([p[0] for p in points_sorted])
                    y_coords_original = np.array([p[1] for p in points_sorted])  # 校正前的拾取点时间
                    
                    # 选择颜色（使用活动拾取字的颜色）
                    if active_pick_word <= len(pick_colors):
                        pickc_value = pick_colors[active_pick_word - 1]
                        color_idx = pickc_value - 2
                    else:
                        color_idx = (active_pick_word - 1) % len(color_map)
                    
                    if color_idx < 0:
                        color_idx = 0
                    elif color_idx >= len(color_map):
                        color_idx = color_idx % len(color_map)
                    
                    color = color_map[color_idx]
                    
                    # ✅ 基于校正前的拾取震相生成平滑曲线（使用多项式或样条插值）
                    # 使用平滑度参数控制曲线的平滑程度，不严格通过所有点，以提取短波长异常
                    try:
                        if len(x_coords) >= 4:
                            # 方法1：使用UnivariateSpline进行平滑（类似Beta Spline）
                            from scipy.interpolate import UnivariateSpline
                            
                            # 计算平滑参数s
                            # smoothness: 0=通过所有点(s=0)，1=最平滑(s=很大值)
                            variance = np.var(y_coords_original)
                            if variance > 0:
                                # smoothness=0时，s=0（通过所有点）
                                # smoothness=1时，s=len(x_coords)*variance（非常平滑）
                                s_param = len(x_coords) * variance * self.static_correction_smoothness
                            else:
                                s_param = 0  # 如果方差为0，使用s=0
                            
                            # 使用UnivariateSpline生成平滑曲线
                            spline = UnivariateSpline(x_coords, y_coords_original, s=s_param, k=min(3, len(x_coords)-1))
                            
                            # 生成更多的点用于绘制平滑曲线
                            x_smooth = np.linspace(min(x_coords), max(x_coords), max(100, len(x_coords) * 10))
                            y_smooth = spline(x_smooth)
                        elif len(x_coords) >= 3:
                            # 方法2：使用多项式拟合（数据点较少时）
                            # 使用2次或3次多项式
                            degree = min(2, len(x_coords) - 1)
                            coeffs = np.polyfit(x_coords, y_coords_original, degree)
                            poly_func = np.poly1d(coeffs)
                            x_smooth = np.linspace(min(x_coords), max(x_coords), max(50, len(x_coords) * 5))
                            y_smooth = poly_func(x_smooth)
                        elif len(x_coords) >= 2:
                            # 方法3：线性插值（数据点很少时）
                            from scipy.interpolate import interp1d
                            interp_func = interp1d(x_coords, y_coords_original, kind='linear', 
                                                bounds_error=False, fill_value='extrapolate')
                            x_smooth = np.linspace(min(x_coords), max(x_coords), max(50, len(x_coords) * 5))
                            y_smooth = interp_func(x_smooth)
                        else:
                            # 只有一个点，无法绘制曲线
                            x_smooth = x_coords
                            y_smooth = y_coords_original
                        
                        # 绘制平滑的虚线曲线（基于校正前的拾取震相）
                        curve_line = self.ax.plot(x_smooth, y_smooth, linestyle='--', linewidth=2.5,
                                                   color=color, alpha=0.8, zorder=14,
                                                   label=f'静校正后震相曲线 (拾取字{active_pick_word})')[0]
                        # ✅ 存储曲线引用以便后续清除
                        if not hasattr(self, 'static_correction_curve_lines'):
                            self.static_correction_curve_lines = []
                        self.static_correction_curve_lines.append(curve_line)
                        
                        # ✅ 可选：绘制校正前的拾取点连线（用于对比）
                        # self.ax.plot(x_coords, y_coords_original, linestyle=':', linewidth=1.0,
                        #            color=color, alpha=0.5, zorder=13, label='校正前拾取点连线')
                        
                    except Exception as e:
                        # 如果插值失败，直接绘制折线
                        logger.warning(f"绘制校正后震相曲线失败: {e}")
                        self.ax.plot(x_coords, y_coords_original, linestyle='--', linewidth=2.0,
                                   color=color, alpha=0.7, zorder=14,
                                   marker='o', markersize=4,
                                   label=f'静校正后震相曲线 (拾取字{active_pick_word})')
    
    def plot_theoretical_traveltime(self, 
                                   distances: np.ndarray,
                                   times: np.ndarray,
                                   params: ZPlotParameters,
                                   color: str = 'green',
                                   linestyle: str = '--',
                                   linewidth: float = 2.0,
                                   alpha: float = 0.8,
                                   label: str = '理论走时'):
        """绘制理论走时曲线
        
        Args:
            distances: 距离数组（km）
            times: 理论走时数组（s）
            params: 绘图参数
            color: 曲线颜色
            linestyle: 线型
            linewidth: 线宽
            alpha: 透明度
            label: 图例标签
        """
        if len(distances) == 0 or len(times) == 0:
            return
        
        # 应用折合速度（如果设置了）
        if params.vred > 0:
            # 计算折合时间
            reduction_times = np.abs(distances) / params.vred
            times = times - reduction_times
        
        # 应用对齐偏移（如果有）
        # 注意：理论走时通常不需要对齐偏移，但为了与观测拾取对比，可能需要应用
        
        # 应用 txadj
        if params.txadj != 0:
            times = times + params.txadj
        
        # 绘制理论走时曲线
        line = self.ax.plot(distances, times, 
                           color=color, 
                           linestyle=linestyle, 
                           linewidth=linewidth,
                           alpha=alpha,
                           label=label,
                           zorder=15)[0]  # 较高的层级，确保可见
        
        # ✅ 存储理论走时曲线引用以便后续清除
        if not hasattr(self, 'theoretical_traveltime_lines'):
            self.theoretical_traveltime_lines = []
        self.theoretical_traveltime_lines.append(line)
    
    def plot_static_correction_visualization(self, x_coordinates: np.ndarray, 
                                            trace_indices: List[int],
                                            params: ZPlotParameters):
        """绘制静校正量可视化
        
        在图形顶部绘制条形图显示静校正量
        
        Args:
            x_coordinates: X坐标数组
            trace_indices: 道索引列表
            params: 绘图参数
        """
        if not self.static_corrections or not self.static_correction_visualization:
            return
        
        if len(x_coordinates) == 0 or len(trace_indices) == 0:
            return
        
        try:
            # 获取Y轴范围，在顶部留出空间
            ylim = self.ax.get_ylim()
            y_min, y_max = ylim
            
            # 计算可视化区域（在图形顶部）
            y_range = y_max - y_min
            viz_height = y_range * 0.08  # 使用8%的高度
            viz_bottom = y_max - viz_height
            
            # 收集静校正数据
            correction_x = []
            correction_y = []
            correction_values = []
            
            for i, trace_idx in enumerate(trace_indices):
                if trace_idx in self.static_corrections and i < len(x_coordinates):
                    correction = self.static_corrections[trace_idx]
                    correction_x.append(x_coordinates[i])
                    correction_y.append(viz_bottom + viz_height / 2)  # 中心位置
                    correction_values.append(correction)
            
            if not correction_x:
                return
            
            # 归一化静校正量用于显示（映射到可视化区域的高度）
            if correction_values:
                max_abs_corr = max(abs(c) for c in correction_values)
                if max_abs_corr > 0:
                    # 将静校正量映射到可视化区域
                    normalized = [c / max_abs_corr * viz_height * 0.4 for c in correction_values]
                else:
                    normalized = [0] * len(correction_values)
                
                # 绘制条形图
                for x, y, norm_val, corr_val in zip(correction_x, correction_y, normalized, correction_values):
                    # 使用颜色表示正负：正值为红色，负值为蓝色
                    color = 'red' if corr_val >= 0 else 'blue'
                    alpha = 0.6
                    
                    # 绘制垂直线条
                    line = self.ax.plot([x, x], [y - norm_val/2, y + norm_val/2], 
                                      color=color, linewidth=1.5, alpha=alpha, zorder=20)
                    self.static_correction_artists.extend(line)
                
                # 添加标签
                label_text = f'Static Correction (max: {max_abs_corr:.3f}s)'
                self.ax.text(0.02, 0.98, label_text, transform=self.ax.transAxes,
                           fontsize=8, verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                           zorder=21)
        except Exception as e:
            logger.error(f"绘制静校正可视化失败: {e}", exc_info=True)
    
    def clear(self):
        """清空图形"""
        self.ax.clear()
        self.traces_artists = []
        # 清除移除trace的标记
        for marker in self.removed_trace_markers:
            try:
                marker.remove()
            except:
                pass
        self.removed_trace_markers.clear()
        # 清除静校正可视化
        for artist in self.static_correction_artists:
            try:
                artist.remove()
            except:
                pass
        self.static_correction_artists.clear()
