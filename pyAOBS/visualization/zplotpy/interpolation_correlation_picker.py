"""
interpolation_correlation_picker.py - 插值-相关拾取模块

实现插值-相关自动拾取功能：
- 在两个已知拾取点之间，通过波形相关性自动拾取中间道的走时
- 使用线性插值创建组合参考波形
- 使用互相关方法搜索最佳匹配位置
- 通过三次互相关验证拾取可靠性

基于原始Fortran代码（zplot/zplot/main.f 约2100-2230行）
原始作者：Colin A. Zelt (1994)
修改者：Haibo Huang (2023)
Python实现：2024
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy import signal
from scipy.interpolate import interp1d

# 兼容相对导入和绝对导入
try:
    from .signal_processor import SignalProcessor
    from .src_kernel_bridge import SrcCrossCorrelationKernelBridge
except ImportError:
    from signal_processor import SignalProcessor
    from src_kernel_bridge import SrcCrossCorrelationKernelBridge


class InterpolationCorrelationPicker:
    """插值-相关拾取类
    
    实现两个已知拾取点之间的自动拾取功能
    """
    
    def __init__(self, use_spline_interpolation: bool = True,
                 use_adaptive_weighting: bool = True,
                 use_sta_lta: bool = False,
                 use_phase_correlation: bool = False):
        """初始化插值-相关拾取器
        
        Args:
            use_spline_interpolation: 是否使用样条插值（默认True）
            use_adaptive_weighting: 是否使用自适应加权（默认True）
            use_sta_lta: 是否使用STA/LTA预处理（默认False）
            use_phase_correlation: 是否使用纯相位互相关（默认False）
        """
        self.signal_processor = SignalProcessor()
        self.use_spline_interpolation = use_spline_interpolation
        self.use_adaptive_weighting = use_adaptive_weighting
        self.use_sta_lta = use_sta_lta
        self.use_phase_correlation = use_phase_correlation
        self.src_crscor_kernel = SrcCrossCorrelationKernelBridge()
    
    def extract_pilot_trace(self, trace: np.ndarray, pick_time: float,
                           times: np.ndarray, window_length: int,
                           apply_filter: bool = False,
                           filter_params: Optional[Dict] = None,
                           apply_hilbert: bool = False) -> Tuple[np.ndarray, float]:
        """提取参考波形（pilot trace）
        
        Args:
            trace: 道数据
            pick_time: 拾取时间（秒）
            times: 时间数组
            window_length: 窗口长度（样本数）
            apply_filter: 是否应用滤波
            filter_params: 滤波参数（如果为None，不滤波）
            apply_hilbert: 是否应用Hilbert变换
        
        Returns:
            (pilot_trace, max_amplitude)
            - pilot_trace: 参考波形数组
            - max_amplitude: 最大振幅（用于归一化权重）
        """
        if len(trace) == 0 or len(times) == 0:
            return np.array([]), 0.0
        
        # 计算拾取时间对应的样本索引
        dt = times[1] - times[0] if len(times) > 1 else 0.001
        pick_sample = int(round((pick_time - times[0]) / dt))
        
        # 检查边界
        if pick_sample < 0 or pick_sample >= len(trace):
            return np.array([]), 0.0
        
        # 提取窗口数据
        start_idx = pick_sample
        end_idx = min(start_idx + window_length, len(trace))
        
        if end_idx <= start_idx:
            return np.array([]), 0.0
        
        pilot = trace[start_idx:end_idx].copy()
        
        # 如果窗口长度不足，用零填充
        if len(pilot) < window_length:
            pilot = np.pad(pilot, (0, window_length - len(pilot)), mode='constant')
        
        # 去直流分量
        pilot = pilot - np.mean(pilot)
        
        # 应用滤波（如果启用）
        if apply_filter and filter_params is not None:
            # 这里可以添加滤波逻辑
            # 暂时跳过，因为需要访问DataProcessor
            pass
        
        # 应用Hilbert变换（如果启用）
        if apply_hilbert:
            pilot_phase = self.signal_processor.extract_phase(pilot)
            # 注意：Hilbert变换后，pilot存储的是相位信息
            # 但互相关时仍使用原始波形
            pass
        
        # 计算最大振幅
        max_amplitude = np.max(np.abs(pilot)) if len(pilot) > 0 else 0.0
        
        return pilot, max_amplitude
    
    def estimate_snr(self, trace: np.ndarray, signal_window: Optional[Tuple[int, int]] = None) -> float:
        """估计信噪比
        
        Args:
            trace: 道数据
            signal_window: 信号窗口 (start_idx, end_idx)，如果为None则使用整个窗口
        
        Returns:
            信噪比估计值
        """
        if len(trace) == 0:
            return 0.0
        
        if signal_window is None:
            # 使用整个窗口
            signal_power = np.var(trace)
            # 假设噪声功率为信号功率的10%（简化估计）
            noise_power = signal_power * 0.1
        else:
            start_idx, end_idx = signal_window
            signal_segment = trace[start_idx:end_idx]
            signal_power = np.var(signal_segment)
            # 使用窗口前后的数据估计噪声
            noise_segments = []
            if start_idx > 0:
                noise_segments.append(trace[:start_idx])
            if end_idx < len(trace):
                noise_segments.append(trace[end_idx:])
            if noise_segments:
                noise_power = np.mean([np.var(seg) for seg in noise_segments])
            else:
                noise_power = signal_power * 0.1
        
        if noise_power <= 0:
            return 100.0  # 高信噪比
        
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100.0
        return max(0.0, snr)  # 确保非负
    
    def waveform_similarity(self, trace1: np.ndarray, trace2: np.ndarray) -> float:
        """计算波形相似性（归一化互相关）
        
        Args:
            trace1: 第一个波形
            trace2: 第二个波形
        
        Returns:
            相似性系数（0-1）
        """
        if len(trace1) == 0 or len(trace2) == 0:
            return 0.0
        
        # 确保长度一致
        min_len = min(len(trace1), len(trace2))
        t1 = trace1[:min_len]
        t2 = trace2[:min_len]
        
        # 归一化
        t1_norm = (t1 - np.mean(t1)) / (np.std(t1) + 1e-10)
        t2_norm = (t2 - np.mean(t2)) / (np.std(t2) + 1e-10)
        
        # 计算相关系数
        correlation = np.corrcoef(t1_norm, t2_norm)[0, 1]
        if np.isnan(correlation):
            return 0.0
        
        # 转换为0-1范围
        similarity = (correlation + 1.0) / 2.0
        return max(0.0, min(1.0, similarity))
    
    def sta_lta_filter(self, trace: np.ndarray, sta_window: int = 5, lta_window: int = 50) -> np.ndarray:
        """STA/LTA滤波
        
        Args:
            trace: 道数据
            sta_window: 短时平均窗口长度（样本数）
            lta_window: 长时平均窗口长度（样本数）
        
        Returns:
            STA/LTA比值数组
        """
        if len(trace) == 0:
            return np.array([])
        
        # 计算平方（能量）
        trace_squared = trace**2
        
        # 短时平均（STA）
        sta = np.convolve(trace_squared, np.ones(sta_window) / sta_window, mode='same')
        
        # 长时平均（LTA）
        lta = np.convolve(trace_squared, np.ones(lta_window) / lta_window, mode='same')
        
        # 计算比值（避免除零）
        ratio = sta / (lta + 1e-10)
        
        return ratio
    
    def calculate_adaptive_weights(self, pilot1: np.ndarray, pilot2: np.ndarray,
                                  trace: np.ndarray, x1: float, x2: float,
                                  x_current: float, ampmx1: float, ampmx2: float) -> Tuple[float, float]:
        """计算自适应权重
        
        根据信噪比、波形相似性和距离综合计算权重
        
        Args:
            pilot1: 第一个参考波形
            pilot2: 第二个参考波形
            trace: 当前道数据（用于计算相似性）
            x1: 第一个拾取点的偏移
            x2: 第二个拾取点的偏移
            x_current: 当前道的偏移
            ampmx1: 第一个参考波形的最大振幅
            ampmx2: 第二个参考波形的最大振幅
        
        Returns:
            (w1, w2) 权重对
        """
        if not self.use_adaptive_weighting:
            # 使用简单的距离加权
            if ampmx1 > 0 and ampmx2 > 0:
                w1 = abs((x2 - x_current) / (x2 - x1)) / ampmx1
                w2 = abs((x1 - x_current) / (x2 - x1)) / ampmx2
            else:
                w1 = abs((x2 - x_current) / (x2 - x1))
                w2 = abs((x1 - x_current) / (x2 - x1))
            # 归一化
            total = w1 + w2
            if total > 0:
                w1 /= total
                w2 /= total
            return w1, w2
        
        # 1. 距离权重（基础权重）
        dist_weight1 = abs((x2 - x_current) / (x2 - x1)) if abs(x2 - x1) > 1e-6 else 0.5
        dist_weight2 = abs((x1 - x_current) / (x2 - x1)) if abs(x2 - x1) > 1e-6 else 0.5
        
        # 2. 信噪比权重
        snr1 = self.estimate_snr(pilot1)
        snr2 = self.estimate_snr(pilot2)
        snr_weight1 = snr1 / (snr1 + snr2 + 1e-10)
        snr_weight2 = snr2 / (snr1 + snr2 + 1e-10)
        
        # 3. 波形相似性权重（如果提供了当前道数据）
        similarity_weight1 = 0.5
        similarity_weight2 = 0.5
        if len(trace) > 0:
            # 提取当前道的窗口（用于相似性计算）
            # 这里简化处理，使用pilot1和pilot2的长度
            min_len = min(len(pilot1), len(pilot2), len(trace))
            if min_len > 0:
                sim1 = self.waveform_similarity(pilot1[:min_len], trace[:min_len])
                sim2 = self.waveform_similarity(pilot2[:min_len], trace[:min_len])
                total_sim = sim1 + sim2
                if total_sim > 0:
                    similarity_weight1 = sim1 / total_sim
                    similarity_weight2 = sim2 / total_sim
        
        # 4. 综合权重（加权平均）
        w1 = 0.4 * dist_weight1 + 0.3 * snr_weight1 + 0.3 * similarity_weight1
        w2 = 0.4 * dist_weight2 + 0.3 * snr_weight2 + 0.3 * similarity_weight2
        
        # 归一化
        total = w1 + w2
        if total > 0:
            w1 /= total
            w2 /= total
        else:
            w1 = 0.5
            w2 = 0.5
        
        return w1, w2
    
    def interpolate_time(self, x1: float, t1: float, x2: float, t2: float,
                        x_current: float) -> float:
        """插值走时
        
        根据设置使用线性插值或样条插值
        
        Args:
            x1: 第一个拾取点的偏移
            t1: 第一个拾取点的走时
            x2: 第二个拾取点的偏移
            t2: 第二个拾取点的走时
            x_current: 当前道的偏移
        
        Returns:
            插值得到的走时
        """
        if self.use_spline_interpolation and abs(x2 - x1) > 1e-6:
            # 使用样条插值（需要至少3个点，这里使用线性插值作为fallback）
            # 对于两个点的情况，样条插值退化为线性插值
            # 但我们可以使用三次样条的外推
            try:
                # 创建插值函数
                # 对于两个点，使用线性插值
                f = interp1d([x1, x2], [t1, t2], kind='linear', 
                           fill_value='extrapolate', bounds_error=False)
                return float(f(x_current))
            except:
                # Fallback到线性插值
                pass
        
        # 线性插值
        if abs(x2 - x1) > 1e-6:
            s = (t2 - t1) / (x2 - x1)
            b = t1 - s * x1
            return s * x_current + b
        else:
            return (t1 + t2) / 2.0
    
    def phase_cross_correlation(self, pilot_phase: np.ndarray, trace_phase: np.ndarray) -> float:
        """纯相位互相关
        
        使用相位信息进行互相关，对振幅变化不敏感
        
        Args:
            pilot_phase: 参考波形的相位
            trace_phase: 待搜索波形的相位
        
        Returns:
            相位互相关系数（-1到1）
        """
        if len(pilot_phase) == 0 or len(trace_phase) == 0:
            return 0.0
        
        # 确保长度一致
        min_len = min(len(pilot_phase), len(trace_phase))
        p1 = pilot_phase[:min_len]
        p2 = trace_phase[:min_len]
        
        # 计算相位差
        phase_diff = p1 - p2
        
        # 相位互相关：exp(i*phase_diff)的实部
        # 等价于 cos(phase_diff) 的平均值
        phase_corr = np.mean(np.cos(phase_diff))
        
        return phase_corr
    
    def cross_correlation(self, pilot: np.ndarray, trace: np.ndarray,
                         start_idx: int, window_length: int,
                         search_range: int, hilbert_ratio: float = 1.0) -> Tuple[int, float]:
        """计算互相关，返回最优滞后和相关系数
        
        使用Hilbert变换和相位一致性加权
        可选：使用纯相位互相关
        
        Args:
            pilot: 参考波形
            trace: 待搜索的道数据
            start_idx: 搜索起始位置（样本索引）
            window_length: 窗口长度（样本数）
            search_range: 搜索范围（±样本数）
            hilbert_ratio: Hilbert变换权重因子
        
        Returns:
            (最优滞后, 最大相关系数)
        """
        if len(pilot) == 0 or len(trace) == 0:
            return 0, 0.0

        if not self.use_phase_correlation:
            return self.src_crscor_kernel.cross_correlate(
                pilot=np.asarray(pilot),
                trace=np.asarray(trace),
                start_idx=int(start_idx),
                window_length=int(window_length),
                search_range=int(search_range),
                hilbert_ratio=float(hilbert_ratio),
            )

        # 纯相位模式保留 Python 实现
        pilot_window = pilot[:window_length] if len(pilot) >= window_length else pilot
        pilot_phase = self.signal_processor.extract_phase(pilot_window)
        best_lag = 0
        max_correlation = -np.inf
        for lag in range(-search_range, search_range + 1):
            trace_start = start_idx + lag
            if trace_start < 0 or trace_start + window_length > len(trace):
                continue
            trace_window = trace[trace_start:trace_start + window_length]
            trace_phase = self.signal_processor.extract_phase(trace_window)
            correlation = self.phase_cross_correlation(pilot_phase, trace_phase)
            if correlation > max_correlation:
                max_correlation = correlation
                best_lag = lag
        return best_lag, max_correlation
    
    def interpolation_correlation_picking(
        self,
        traces: List[np.ndarray],
        times: np.ndarray,
        offsets: np.ndarray,
        pick1_idx: int,
        pick2_idx: int,
        pick1_time: float,
        pick2_time: float,
        correlation_window: int,
        search_range: int,
        hilbert_ratio: float = 1.0,
        apply_filter: bool = False,
        filter_params: Optional[Dict] = None,
        apply_hilbert: bool = False,
        force_pick: bool = False
    ) -> Dict[int, float]:
        """在两个已知拾取点之间进行插值-相关自动拾取
        
        Args:
            traces: 道数据列表
            times: 时间数组
            offsets: 偏移数组
            pick1_idx: 第一个拾取点的道索引
            pick2_idx: 第二个拾取点的道索引
            pick1_time: 第一个拾取点的走时（秒）
            pick2_time: 第二个拾取点的走时（秒）
            correlation_window: 互相关窗口长度（样本数）
            search_range: 搜索范围（±样本数）
            hilbert_ratio: Hilbert变换权重因子
            apply_filter: 是否应用滤波
            filter_params: 滤波参数
            apply_hilbert: 是否应用Hilbert变换
            force_pick: 是否强制拾取（即使结果不一致也拾取）
        
        Returns:
            字典：{trace_idx: pick_time} - 成功拾取的道及其走时
        """
        # 输入验证
        if len(traces) == 0 or len(times) == 0 or len(offsets) == 0:
            return {}
        
        if pick1_idx < 0 or pick1_idx >= len(traces):
            return {}
        if pick2_idx < 0 or pick2_idx >= len(traces):
            return {}
        if pick1_idx == pick2_idx:
            return {}
        
        # 计算采样间隔
        dt = times[1] - times[0] if len(times) > 1 else 0.001
        
        # 1. 提取参考波形
        pilot1, ampmx1 = self.extract_pilot_trace(
            traces[pick1_idx], pick1_time, times,
            correlation_window, apply_filter, filter_params, apply_hilbert
        )
        pilot2, ampmx2 = self.extract_pilot_trace(
            traces[pick2_idx], pick2_time, times,
            correlation_window, apply_filter, filter_params, apply_hilbert
        )
        
        if len(pilot1) == 0 or len(pilot2) == 0:
            return {}
        
        # 2. 计算插值参数
        xp1 = offsets[pick1_idx]
        xp2 = offsets[pick2_idx]
        
        if abs(xp2 - xp1) < 1e-6:
            return {}  # 两个拾取点偏移相同，无法插值
        
        # 3. 确定中间道的范围
        min_idx = min(pick1_idx, pick2_idx)
        max_idx = max(pick1_idx, pick2_idx)
        
        # 4. 遍历中间道进行自动拾取
        picks = {}
        failed_traces = []
        
        for trace_idx in range(min_idx + 1, max_idx):
            # 检查道是否有效
            if trace_idx < 0 or trace_idx >= len(traces):
                continue
            
            # 检查偏移是否在两个拾取点之间
            x_current = offsets[trace_idx]
            if (x_current > xp1 and x_current > xp2) or \
               (x_current < xp1 and x_current < xp2):
                continue  # 不在范围内，跳过
            
            # 获取当前道数据（必须在所有使用trace的地方之前）
            trace = traces[trace_idx].copy()
            
            # 应用STA/LTA预处理（如果启用）
            if self.use_sta_lta:
                trace_sta_lta = self.sta_lta_filter(trace)
                # 使用STA/LTA比值作为权重因子（可选）
                # 这里简化处理，不修改trace本身
                pass
            
            # 计算自适应权重
            w1, w2 = self.calculate_adaptive_weights(
                pilot1, pilot2, trace, xp1, xp2, x_current, ampmx1, ampmx2
            )
            
            # 创建组合参考波形（插值）
            ccpilot = w1 * pilot1 + w2 * pilot2
            
            # 计算预期走时（使用插值方法）
            tline = self.interpolate_time(xp1, pick1_time, xp2, pick2_time, x_current)
            nline = int(round((tline - times[0]) / dt))
            
            # 检查边界
            if nline < 0 or nline >= len(times):
                failed_traces.append(trace_idx)
                continue
            
            # 去直流分量（在预期位置附近）
            dc_start = max(0, nline - search_range)
            dc_end = min(len(trace), nline + search_range + correlation_window)
            if dc_end > dc_start:
                trace[dc_start:dc_end] = trace[dc_start:dc_end] - np.mean(trace[dc_start:dc_end])
            
            # 执行三次互相关分析
            lag0, corr0 = self.cross_correlation(
                ccpilot, trace, nline, correlation_window,
                search_range, hilbert_ratio
            )
            lag1, corr1 = self.cross_correlation(
                pilot1, trace, nline, correlation_window,
                search_range, hilbert_ratio
            )
            lag2, corr2 = self.cross_correlation(
                pilot2, trace, nline, correlation_window,
                search_range, hilbert_ratio
            )
            
            # 改进的一致性检查
            # 不仅检查滞后的一致性，还检查相关系数的质量
            lagmin = min(lag0, lag1, lag2)
            lagmax = max(lag0, lag1, lag2)
            lag_diff = abs(lagmax - lagmin)
            
            # 计算平均相关系数
            avg_correlation = (corr0 + corr1 + corr2) / 3.0
            
            # 一致性条件：
            # 1. 滞后差异小于搜索范围
            # 2. 平均相关系数大于阈值（0.3）
            # 3. 或者强制拾取模式
            is_consistent = lag_diff < search_range and avg_correlation > 0.3
            
            if is_consistent or force_pick:
                # 使用插值参考波形的结果（lag0），因为它考虑了插值
                # 但如果lag0的相关系数较低，可以考虑使用lag1或lag2
                if corr0 < 0.2 and (corr1 > corr0 or corr2 > corr0):
                    # 如果插值参考波形的相关系数太低，使用相关系数更高的结果
                    if corr1 > corr2:
                        best_lag = lag1
                    else:
                        best_lag = lag2
                else:
                    best_lag = lag0

                # 防御性约束：无论内核返回什么，滞后都不允许越过搜索窗。
                # 避免异常 lag 导致走时远超 tlag 定义范围。
                if search_range > 0:
                    best_lag = int(np.clip(int(best_lag), -int(search_range), int(search_range)))
                else:
                    best_lag = 0

                tpk = tline + float(best_lag) * dt
                picks[trace_idx] = tpk
            else:
                failed_traces.append(trace_idx)
        
        return picks
    
    def get_picking_statistics(self, picks: Dict[int, float],
                              total_traces: int) -> Dict:
        """获取拾取统计信息
        
        Args:
            picks: 拾取结果字典 {trace_idx: pick_time}
            total_traces: 总道数
        
        Returns:
            统计信息字典
        """
        n_picked = len(picks)
        n_failed = total_traces - n_picked
        success_rate = n_picked / total_traces if total_traces > 0 else 0.0
        
        return {
            'n_picked': n_picked,
            'n_failed': n_failed,
            'total_traces': total_traces,
            'success_rate': success_rate
        }
