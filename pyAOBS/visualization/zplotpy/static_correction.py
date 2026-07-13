"""
static_correction.py - 静校正模块

实现短波长静校正提取和应用功能
使用高斯空间滤波提取短波长时间变化
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d, UnivariateSpline, make_interp_spline
import logging

logger = logging.getLogger(__name__)


class StaticCorrector:
    """静校正器 - 提取和应用短波长静校正"""
    
    def __init__(self):
        """初始化静校正器"""
        self.static_corrections: Dict[int, float] = {}  # {trace_idx: correction_time}
    
    def extract_short_wavelength_gaussian(
        self,
        picks: Dict[int, Dict[int, float]],
        trace_indices: List[int],
        x_coords: np.ndarray,
        pick_word: int,
        sigma: float = 3.0,
        min_picks: int = 3,
        display_times: Optional[Dict[int, float]] = None,
        smoothness: float = 0.1
    ) -> Dict[int, float]:
        """使用高斯空间滤波提取短波长静校正量
        
        Args:
            picks: 拾取数据字典 {trace_idx: {pick_word: time}}
            trace_indices: 道索引列表（过滤后的索引）
            x_coords: X坐标数组（炮检距或其他空间坐标，单位：km）
            pick_word: 拾取字编号（用于提取特定拾取）
            sigma: 高斯核标准差（单位：km），控制长波长提取的尺度
            min_picks: 最小拾取数量，少于此数量则返回空字典
            display_times: 显示时间字典 {trace_idx: display_time}，如果提供则使用显示时间而不是原始拾取时间
            
        Returns:
            静校正量字典 {trace_idx: correction_time}，单位为秒
        """
        # ✅ 收集有效的拾取数据
        # 如果提供了display_times，使用显示时间；否则使用原始拾取时间
        valid_indices = []
        valid_trace_indices = []
        valid_picks = []
        valid_x_coords = []
        
        for i, trace_idx in enumerate(trace_indices):
            if trace_idx in picks and pick_word in picks[trace_idx]:
                pick_time = picks[trace_idx][pick_word]
                if pick_time > 0:  # 有效拾取
                    # ✅ 如果提供了display_times，使用显示时间；否则使用原始拾取时间
                    if display_times and trace_idx in display_times:
                        pick_time = display_times[trace_idx]
                    
                    valid_indices.append(i)
                    valid_trace_indices.append(trace_idx)
                    valid_picks.append(pick_time)
                    if i < len(x_coords):
                        valid_x_coords.append(x_coords[i])
                    else:
                        # 如果x_coords长度不够，使用索引作为坐标
                        valid_x_coords.append(float(i))
        
        if len(valid_picks) < min_picks:
            logger.warning(f"有效拾取数量 ({len(valid_picks)}) 少于最小值 ({min_picks})，无法计算静校正")
            return {}
        
        # 转换为numpy数组
        picks_array = np.array(valid_picks)
        x_coords_array = np.array(valid_x_coords)
        
        # 检查x_coords是否有足够的空间变化
        x_range = np.max(x_coords_array) - np.min(x_coords_array)
        if x_range < 0.01:  # 如果空间范围太小（<10m），使用索引作为坐标
            logger.warning("X坐标范围太小，使用道索引作为空间坐标")
            x_coords_array = np.array(valid_indices, dtype=float)
            # 调整sigma为道数单位（假设平均道间距为1）
            # 将km转换为道数：假设平均道间距约为0.1-0.5km
            avg_trace_spacing = max(0.1, x_range / len(valid_picks)) if len(valid_picks) > 1 else 0.1
            sigma_in_traces = sigma / avg_trace_spacing
        else:
            # 使用实际空间坐标
            sigma_in_traces = sigma
        
        try:
            # 对拾取时间进行高斯平滑（提取长波长趋势）
            # 注意：gaussian_filter1d的sigma参数是样本数，不是空间单位
            # 我们需要将空间sigma转换为样本数
            if len(x_coords_array) > 1:
                # 计算平均空间采样间隔
                avg_spacing = np.mean(np.diff(np.sort(x_coords_array)))
                if avg_spacing > 0:
                    # 将空间sigma转换为样本数
                    sigma_samples = sigma_in_traces / avg_spacing
                else:
                    sigma_samples = sigma_in_traces
            else:
                sigma_samples = sigma_in_traces
            
            # 确保sigma_samples合理（至少为0.5，最大不超过数据长度的一半）
            sigma_samples = max(0.5, min(sigma_samples, len(picks_array) / 2.0))
            
            # 对拾取时间进行排序（按空间坐标）
            sort_indices = np.argsort(x_coords_array)
            sorted_picks = picks_array[sort_indices]
            sorted_x_coords = x_coords_array[sort_indices]
            
            # ✅ 使用平滑样条生成平滑曲线（与绘制曲线的方法完全一致）
            # 这是校正后的震相曲线，静校正量 = 平滑曲线值 - 原始拾取值
            from scipy.interpolate import UnivariateSpline
            
            # 计算平滑参数s（与plot_manager中的逻辑完全一致）
            variance = np.var(sorted_picks)
            if variance > 0:
                s_param = len(sorted_picks) * variance * smoothness
            else:
                s_param = 0
            
            # 生成平滑曲线（与plot_manager中的逻辑完全一致）
            if len(sorted_x_coords) >= 4:
                # 使用UnivariateSpline生成平滑曲线
                spline = UnivariateSpline(sorted_x_coords, sorted_picks, s=s_param, k=min(3, len(sorted_picks)-1))
                # 在原始x坐标上评估平滑曲线值
                smooth_curve_values = spline(sorted_x_coords)
            elif len(sorted_x_coords) >= 3:
                # 使用多项式拟合
                degree = min(2, len(sorted_x_coords) - 1)
                coeffs = np.polyfit(sorted_x_coords, sorted_picks, degree)
                poly_func = np.poly1d(coeffs)
                smooth_curve_values = poly_func(sorted_x_coords)
            else:
                # 数据点太少，使用原始值
                smooth_curve_values = sorted_picks
            
            # ✅ 计算静校正量（平滑曲线值 - 原始拾取值）
            # 如果拾取时间偏早（小于平滑曲线值），静校正量为正值，应用后时间变晚，对齐到平滑曲线
            # 如果拾取时间偏晚（大于平滑曲线值），静校正量为负值，应用后时间变早，对齐到平滑曲线
            # 这样静校正后，所有道的震相都会对齐到平滑曲线，消除短波长异常
            short_wavelength = smooth_curve_values - sorted_picks
            
            # ✅ 对静校正量进行轻微平滑处理（可选，使静校正量更平滑）
            # 使用较小的sigma进行平滑，保持静校正量的基本形状
            smooth_sigma = max(0.5, sigma_samples * 0.3)
            smoothed_short_wavelength = gaussian_filter1d(short_wavelength, sigma=smooth_sigma)
            
            # 构建静校正量字典（映射回原始trace_idx）
            static_corrections = {}
            for sort_idx, original_idx in enumerate(sort_indices):
                trace_idx = valid_trace_indices[original_idx]
                static_corrections[trace_idx] = float(smoothed_short_wavelength[sort_idx])
            
            # ✅ 对所有道使用平滑曲线插值，生成一致的静校正量
            # 方法：对所有道插值平滑曲线值，然后计算静校正量 = 平滑曲线值 - 原始拾取值
            if len(sorted_x_coords) > 1:
                # 准备插值数据：平滑曲线值（按空间坐标排序）
                interp_x = sorted_x_coords
                interp_y = smooth_curve_values  # 使用平滑曲线值，而不是静校正量
                
                # 创建平滑曲线的插值函数
                if len(interp_x) >= 4:
                    try:
                        # 使用B样条插值（B-spline），产生非常平滑的曲线
                        # 使用3次B样条（cubic B-spline）
                        bspline = make_interp_spline(interp_x, interp_y, k=3)
                        
                        # 对所有道进行插值
                        for i, trace_idx in enumerate(trace_indices):
                            if i < len(x_coords):
                                x_coord = x_coords[i]
                                # 插值平滑曲线值
                                interpolated_smooth_value = float(bspline(x_coord))
                                
                                # 获取该道的原始拾取值（如果有）
                                if trace_idx in picks and pick_word in picks[trace_idx]:
                                    pick_time = picks[trace_idx][pick_word]
                                    if display_times and trace_idx in display_times:
                                        pick_time = display_times[trace_idx]
                                    if pick_time > 0:
                                        # 有拾取：静校正量 = 平滑曲线值 - 原始拾取值
                                        # 使用插值后的平滑曲线值，但保持原始拾取点的静校正量（加权平均）
                                        original_correction = static_corrections.get(trace_idx, 0)
                                        interpolated_correction = interpolated_smooth_value - pick_time
                                        # 加权平均：70%插值 + 30%原始（保持连续性）
                                        static_corrections[trace_idx] = 0.7 * interpolated_correction + 0.3 * original_correction
                                    else:
                                        # 拾取无效，使用插值
                                        static_corrections[trace_idx] = interpolated_smooth_value - pick_time if pick_time > 0 else 0
                                else:
                                    # 没有拾取：需要估算原始拾取值
                                    # 使用相邻拾取点的平均值或插值
                                    if len(sorted_picks) >= 2:
                                        # 简单线性插值估算原始拾取值
                                        from scipy.interpolate import interp1d
                                        pick_interp = interp1d(sorted_x_coords, sorted_picks, kind='linear',
                                                              bounds_error=False, fill_value='extrapolate')
                                        estimated_pick = float(pick_interp(x_coord))
                                        static_corrections[trace_idx] = interpolated_smooth_value - estimated_pick
                                    else:
                                        static_corrections[trace_idx] = 0
                    except Exception as e:
                        logger.warning(f"B样条插值失败，使用三次样条: {e}")
                        # 回退到三次样条插值
                        interp_func = interp1d(interp_x, interp_y, kind='cubic', 
                                             bounds_error=False, fill_value='extrapolate')
                        for i, trace_idx in enumerate(trace_indices):
                            if i < len(x_coords):
                                x_coord = x_coords[i]
                                interpolated_smooth_value = float(interp_func(x_coord))
                                
                                # 获取该道的原始拾取值（如果有）
                                if trace_idx in picks and pick_word in picks[trace_idx]:
                                    pick_time = picks[trace_idx][pick_word]
                                    if display_times and trace_idx in display_times:
                                        pick_time = display_times[trace_idx]
                                    if pick_time > 0:
                                        original_correction = static_corrections.get(trace_idx, 0)
                                        interpolated_correction = interpolated_smooth_value - pick_time
                                        static_corrections[trace_idx] = 0.7 * interpolated_correction + 0.3 * original_correction
                                else:
                                    if len(sorted_picks) >= 2:
                                        from scipy.interpolate import interp1d
                                        pick_interp = interp1d(sorted_x_coords, sorted_picks, kind='linear',
                                                              bounds_error=False, fill_value='extrapolate')
                                        estimated_pick = float(pick_interp(x_coord))
                                        static_corrections[trace_idx] = interpolated_smooth_value - estimated_pick
                                    else:
                                        static_corrections[trace_idx] = 0
                else:
                    # 数据点太少，使用线性插值
                    interp_func = interp1d(interp_x, interp_y, kind='linear', 
                                         bounds_error=False, fill_value='extrapolate')
                    for i, trace_idx in enumerate(trace_indices):
                        if i < len(x_coords):
                            x_coord = x_coords[i]
                            interpolated_smooth_value = float(interp_func(x_coord))
                            
                            if trace_idx in picks and pick_word in picks[trace_idx]:
                                pick_time = picks[trace_idx][pick_word]
                                if display_times and trace_idx in display_times:
                                    pick_time = display_times[trace_idx]
                                if pick_time > 0:
                                    original_correction = static_corrections.get(trace_idx, 0)
                                    interpolated_correction = interpolated_smooth_value - pick_time
                                    static_corrections[trace_idx] = 0.7 * interpolated_correction + 0.3 * original_correction
                            else:
                                if len(sorted_picks) >= 2:
                                    from scipy.interpolate import interp1d
                                    pick_interp = interp1d(sorted_x_coords, sorted_picks, kind='linear',
                                                          bounds_error=False, fill_value='extrapolate')
                                    estimated_pick = float(pick_interp(x_coord))
                                    static_corrections[trace_idx] = interpolated_smooth_value - estimated_pick
                                else:
                                    static_corrections[trace_idx] = 0
            
            # 计算最终静校正量的统计信息
            final_corrections = list(static_corrections.values())
            logger.info(f"成功提取静校正量: {len(static_corrections)} 个道, sigma={sigma:.2f}km")
            logger.info(f"静校正量范围: [{np.min(final_corrections):.4f}, {np.max(final_corrections):.4f}] 秒")
            logger.info(f"静校正量标准差: {np.std(final_corrections):.4f} 秒")
            
            return static_corrections
            
        except Exception as e:
            logger.error(f"提取短波长静校正量失败: {e}", exc_info=True)
            return {}
    
    def clear_corrections(self):
        """清除所有静校正量"""
        self.static_corrections.clear()
        logger.info("已清除所有静校正量")
    
    def set_corrections(self, corrections: Dict[int, float]):
        """设置静校正量
        
        Args:
            corrections: 静校正量字典 {trace_idx: correction_time}
        """
        self.static_corrections = corrections.copy()
        logger.info(f"已设置 {len(self.static_corrections)} 个道的静校正量")
    
    def get_correction(self, trace_idx: int) -> float:
        """获取指定道的静校正量
        
        Args:
            trace_idx: 道索引
            
        Returns:
            静校正量（秒），如果该道没有静校正量则返回0.0
        """
        return self.static_corrections.get(trace_idx, 0.0)
    
    def has_corrections(self) -> bool:
        """检查是否有静校正量
        
        Returns:
            如果有静校正量返回True，否则返回False
        """
        return len(self.static_corrections) > 0
