"""
adaptive_stack.py - ZPLOT 自适应叠加模块

实现自适应叠加对齐功能：
- 改进的归一化处理
- 迭代对齐优化
- 相位一致性加权叠加
- 误差估计（基于叠加功率半宽度）

基于原始Fortran代码中的tcas和pstack子程序（misc.f）
原始作者：Brian L.N. Kennett (ANU, 2002)
第一次修改：Nick Rawlinson (RSES, ANU, 2003)
本次修改：Haibo Huang (SCSIO, CAS, 2023.11)
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
import logging
try:
    from numba import njit
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def _wrap(func):
            return func
        return _wrap

# 兼容相对导入和绝对导入
try:
    from .signal_processor import SignalProcessor
except ImportError:
    from signal_processor import SignalProcessor

# 配置日志
logger = logging.getLogger(__name__)


@njit(cache=True)
def _pairwise_corrcoef_numba(segments: np.ndarray) -> np.ndarray:
    """Numba 内核：计算道间皮尔逊相关矩阵。"""
    n_traces, n_samples = segments.shape
    out = np.zeros((n_traces, n_traces), dtype=np.float64)
    for i in range(n_traces):
        out[i, i] = 1.0
        xi = segments[i]
        mean_i = 0.0
        for k in range(n_samples):
            mean_i += xi[k]
        mean_i /= n_samples
        var_i = 0.0
        for k in range(n_samples):
            d = xi[k] - mean_i
            var_i += d * d
        std_i = np.sqrt(var_i / n_samples)

        for j in range(i + 1, n_traces):
            xj = segments[j]
            mean_j = 0.0
            for k in range(n_samples):
                mean_j += xj[k]
            mean_j /= n_samples
            var_j = 0.0
            cov = 0.0
            for k in range(n_samples):
                di = xi[k] - mean_i
                dj = xj[k] - mean_j
                var_j += dj * dj
                cov += di * dj
            std_j = np.sqrt(var_j / n_samples)
            denom = std_i * std_j
            corr = 0.0
            if denom > 1e-12:
                corr = (cov / n_samples) / denom
            out[i, j] = corr
            out[j, i] = corr
    return out


class AdaptiveStacker:
    """自适应叠加类 - 实现tcas和pstack功能"""
    
    def __init__(self, nsi: int = 10, pjgl: int = 3, 
                 stkwb: float = 0.04, stkwl: float = 0.5,
                 dtcw: float = 0.1, ratio: float = 1.0):
        """初始化自适应叠加器
        
        Args:
            nsi: 叠加迭代次数（默认10）
            pjgl: Lp范数指数（默认3，L3范数）
            stkwb: 叠加窗口起始时间（秒，默认0.04）
            stkwl: 叠加窗口长度（秒，默认0.5）
            dtcw: 差分时间搜索范围（秒，默认0.1）
            ratio: 相位一致性加权比例因子（默认1.0）
        """
        self.nsi = nsi
        self.pjgl = pjgl
        self.stkwb = stkwb
        self.stkwl = stkwl
        self.dtcw = dtcw
        self.ratio = ratio
        self.signal_processor = SignalProcessor()
        
        # 误差估计参数（Haibo改进）
        self.emin = 0.025  # 最小误差（秒）
        self.emax = 0.150  # 最大误差（秒）
        self.erl = 1.25    # 误差估计比例因子（半宽度阈值）
        
        # 优化参数
        self.early_stop_threshold = 1e-6  # 早期终止阈值（质量改进小于此值时停止）
        self.min_quality_improvement = 0.01  # 最小质量改进（百分比）
        self.enable_numba_kernels = bool(HAS_NUMBA)
        
        # 优化参数
        self.early_stop_threshold = 1e-6  # 早期终止阈值（质量改进小于此值时停止）
        self.min_quality_improvement = 0.01  # 最小质量改进（百分比）
    
    def normalize_traces(self, traces: List[np.ndarray]) -> List[np.ndarray]:
        """归一化道数据（改进方法）
        
        对每条道独立归一化，使用最大绝对值
        这是Haibo改进的关键点之一，避免小振幅道被大振幅道压制
        
        Args:
            traces: 道数据列表
        
        Returns:
            归一化后的道数据列表
        """
        if not traces:
            return []
        
        normalized = []
        for trace in traces:
            if trace is None or len(trace) == 0:
                normalized.append(np.array([]))
                continue
            
            max_val = np.max(np.abs(trace))
            if max_val > 0:
                normalized.append(trace / max_val)
            else:
                normalized.append(trace.copy())
        return normalized
    
    def stack_traces(self, traces: List[np.ndarray],
                    time_shifts: List[float],
                    times: np.ndarray,
                    initial_picks: List[int],
                    window_start_idx: int,
                    window_length: int,
                    use_phase_weighting: bool = True) -> Tuple[np.ndarray, float]:
        """叠加多道数据（带相位一致性加权）
        
        实现pstack子程序的功能，包括：
        - 线性叠加和二次叠加
        - Hilbert变换相位一致性加权（Haibo新增）
        - 归一化输出
        
        Args:
            traces: 道数据列表
            time_shifts: 每道的时间偏移（秒），相对于初始拾取位置
            times: 时间数组
            initial_picks: 初始拾取样本号列表（必须提供）
            window_start_idx: 窗口起始样本索引（相对于拾取点）
            window_length: 窗口长度（样本数）
            use_phase_weighting: 是否使用相位一致性加权
        
        Returns:
            (stacked_trace, quality_metric)
            - stacked_trace: 叠加结果
            - quality_metric: 叠加质量指标（L2测度的道失配）
        """
        # 输入验证
        if not traces or len(traces) == 0:
            raise ValueError("traces列表不能为空")
        if len(time_shifts) != len(traces):
            raise ValueError(f"time_shifts长度({len(time_shifts)})必须等于traces长度({len(traces)})")
        if len(initial_picks) != len(traces):
            raise ValueError(f"initial_picks长度({len(initial_picks)})必须等于traces长度({len(traces)})")
        if window_length <= 0:
            raise ValueError(f"window_length必须大于0，当前值: {window_length}")
        
        n_samples = window_length
        stacked = np.zeros(n_samples)
        quadratic_sum = np.zeros(n_samples)
        
        # 相位一致性相关
        phase_cos_sum = np.zeros(n_samples)
        phase_sin_sum = np.zeros(n_samples)
        
        n_valid_traces = 0
        dt = times[1] - times[0] if len(times) > 1 else 0.001
        
        # 预计算以避免重复计算
        inv_dt = 1.0 / dt if dt > 0 else 1000.0
        
        for i, trace in enumerate(traces):
            # 检查是否有有效的初始拾取
            if i >= len(initial_picks) or initial_picks[i] < 0:
                continue
            
            # 关键：正确的窗口位置计算（对应Fortran的lmn）
            # Fortran: lmn = -pnkwb + nst0(i) + nint(dtcs(i)/pdts) - 1
            # 其中nst0(i)从1开始，所以-1
            # Python: imo从0开始，所以不需要-1
            # 实际对应：Python的imo = Fortran的nst0(i) - 1
            # 所以：lmn = -window_start_idx + imo + shift_samples
            imo = initial_picks[i]  # 初始拾取样本号（Python从0开始）
            shift_samples = int(round(time_shifts[i] * inv_dt))
            lmn = -window_start_idx + imo + shift_samples
            
            # 边界检查
            if lmn < 0 or lmn + n_samples > len(trace):
                continue
            
            window_data = trace[lmn:lmn + n_samples]
            
            # 线性叠加和二次叠加
            stacked += window_data
            quadratic_sum += window_data**2
            
            # Hilbert变换提取相位（Haibo新增）
            if use_phase_weighting:
                phases = self.signal_processor.extract_phase(window_data)
                phase_cos_sum += np.cos(phases)
                phase_sin_sum += np.sin(phases)
            
            n_valid_traces += 1
        
        if n_valid_traces == 0:
            return stacked, 0.0
        
        # 计算相位一致性因子（Haibo新增）
        if use_phase_weighting and n_valid_traces > 0:
            inv_n_valid = 1.0 / n_valid_traces
            phase_coherence = np.sqrt(phase_cos_sum**2 + phase_sin_sum**2) * inv_n_valid
            phase_weight = phase_coherence ** self.ratio
        else:
            phase_weight = np.ones(n_samples)
        
        # 加权叠加（Haibo改进）
        inv_n_valid = 1.0 / n_valid_traces
        stacked = stacked * phase_weight * inv_n_valid
        
        # 归一化
        max_val = np.max(np.abs(stacked))
        if max_val > 0:
            stacked = stacked / max_val
        
        # 计算质量指标（L2测度的道失配）
        quality_metric = np.sum(np.abs(quadratic_sum)) * inv_n_valid / n_samples
        
        return stacked, quality_metric
    
    def align_traces(self, traces: List[np.ndarray], 
                    times: np.ndarray,
                    initial_picks: Optional[List[int]] = None,
                    sampling_rate: Optional[float] = None) -> Dict:
        """自适应对齐多道数据
        
        实现tcas子程序的功能，包括：
        - 数据归一化（改进方法）
        - 迭代对齐优化
        - 误差估计（最后一次迭代）
        
        Args:
            traces: 道数据列表
            times: 时间数组
            initial_picks: 初始拾取样本号列表（可选）
            sampling_rate: 采样率（Hz）
        
        Returns:
            {
                'time_shifts': List[float],  # 每道的时间偏移（秒）
                'errors': List[float],        # 每道的误差估计（秒）
                'stacked_trace': np.ndarray,  # 最终叠加结果
                'quality_metric': float       # 叠加质量指标
            }
        """
        if not traces or len(traces) == 0:
            return {
                'time_shifts': [],
                'errors': [],
                'stacked_trace': np.array([]),
                'quality_metric': 0.0
            }
        
        # 1. 归一化（Haibo改进）
        normalized_traces = self.normalize_traces(traces)
        
        # 2. 计算采样率
        if sampling_rate is None:
            if len(times) > 1:
                sampling_rate = 1.0 / (times[1] - times[0])
            else:
                sampling_rate = 1000.0  # 默认采样率
        
        dt = 1.0 / sampling_rate
        
        # 3. 确定叠加窗口
        # 注意：stkwb是相对于拾取点的窗口起始偏移（秒），通常是负值（拾取点之前）
        # Fortran: nstkwb = int(stkwb/dts)
        # 如果stkwb=-0.04s，dts=0.001s，则nstkwb=-40
        # 在lmn计算中：lmn = -nstkwb + nst0(i) + ... = 40 + nst0(i) + ...
        # 所以window_start_idx应该是负值（或绝对值），用于计算偏移
        if dt <= 0:
            raise ValueError(f"采样间隔dt必须大于0，当前值: {dt}")
        
        window_start_idx = int(self.stkwb / dt)  # 可能是负值
        window_length = int(self.stkwl / dt) if self.stkwl > 0 else 500  # 默认500样本
        
        # 确保窗口长度合理（不能超过数据长度）
        # 注意：window_start_idx是相对于拾取点的偏移，不是绝对索引
        # 实际的窗口起始位置会在stack_traces中根据每道的拾取位置计算
        if window_length <= 0:
            window_length = min(500, len(times))  # 默认窗口长度，但不能超过数据长度
        if window_length > len(times):
            window_length = len(times)
        
        # 验证窗口参数
        if window_length <= 0:
            raise ValueError(f"计算得到的window_length无效: {window_length}")
        
        # 4. 检查initial_picks
        if initial_picks is None:
            # 如果没有提供initial_picks，无法进行对齐
            return {
                'time_shifts': [0.0] * len(traces),
                'errors': [self.emax] * len(traces),
                'stacked_trace': np.zeros(window_length),
                'quality_metric': 0.0
            }
        
        # 验证initial_picks长度
        if len(initial_picks) != len(traces):
            raise ValueError(
                f"initial_picks长度({len(initial_picks)})必须等于traces长度({len(traces)})"
            )
        
        # 检查是否有有效的拾取点
        valid_picks = [p for p in initial_picks if p >= 0]
        if len(valid_picks) == 0:
            return {
                'time_shifts': [0.0] * len(traces),
                'errors': [self.emax] * len(traces),
                'stacked_trace': np.zeros(window_length),
                'quality_metric': 0.0
            }
        
        # 5. 初始叠加
        initial_shifts = [0.0] * len(traces)
        stacked, _ = self.stack_traces(
            normalized_traces, initial_shifts, times,
            initial_picks, window_start_idx, window_length
        )
        
        # 6. 迭代优化
        time_shifts = [0.0] * len(traces)  # 每次迭代都相对于初始位置
        errors = [self.emax] * len(traces)  # 初始化为最大误差
        
        # 搜索范围（对应Fortran的jim1和jim2）
        jim1 = -int(round(self.dtcw / dt))
        jim2 = int(round(self.dtcw / dt))
        
        # 预计算归一化因子
        inv_stkwl = 1.0 / self.stkwl if self.stkwl > 0 else 1.0
        
        # 早期终止相关变量
        prev_quality = float('inf')
        quality_history = []
        
        for iteration in range(self.nsi):
            try:
                for i, trace in enumerate(normalized_traces):
                    if i >= len(initial_picks) or initial_picks[i] < 0:
                        continue
                    
                    imo = initial_picks[i]  # 初始拾取样本号
                    
                    # 优化：向量化计算功率曲线
                    power_curve = self._compute_power_curve_vectorized(
                        trace, stacked, imo, window_start_idx, window_length,
                        jim1, jim2, inv_stkwl
                    )
                    
                    # 找到最小值位置
                    power_array = np.array(power_curve)
                    valid_mask = np.isfinite(power_array)
                    
                    if np.any(valid_mask):
                        jm_rel = np.argmin(power_array[valid_mask])
                        valid_indices = np.where(valid_mask)[0]
                        jm_idx = valid_indices[jm_rel]
                        jm = jim1 + jm_idx  # 转换为绝对偏移
                        wm = power_array[valid_mask][jm_rel]
                    else:
                        jm = 0
                        wm = float('inf')
                    
                    # 更新时间偏移（相对于初始位置）
                    time_shifts[i] = float(jm) * dt
                    
                    # 最后一次迭代：计算误差估计
                    if iteration == self.nsi - 1:
                        errors[i] = self._estimate_error_from_power_curve(
                            power_curve, jim1, jim2, dt, jm
                        )
                
                # 使用新对齐重新叠加
                stacked, quality = self.stack_traces(
                    normalized_traces, time_shifts, times,
                    initial_picks, window_start_idx, window_length
                )
                
                quality_history.append(quality)
                
                # 早期终止检查：如果质量改进很小，提前终止
                if iteration > 0:
                    quality_improvement = abs(prev_quality - quality) / max(abs(prev_quality), 1e-10)
                    if quality_improvement < self.early_stop_threshold:
                        logger.debug(f"Early stop at iteration {iteration+1}, quality improvement: {quality_improvement:.2e}")
                        break
                    
                    # 如果质量不再改进，也考虑提前终止
                    if quality >= prev_quality * (1.0 - self.min_quality_improvement):
                        logger.debug(f"Quality not improving significantly at iteration {iteration+1}")
                        # 不立即终止，但记录
                
                prev_quality = quality
                
            except Exception as e:
                logger.error(f"Error in alignment iteration {iteration+1}: {e}")
                # 如果出错，使用当前结果
                break
        
        return {
            'time_shifts': time_shifts,
            'errors': errors,
            'stacked_trace': stacked,
            'quality_metric': quality
        }
    
    def _compute_power_curve_vectorized(self, trace: np.ndarray, stacked: np.ndarray,
                                       imo: int, window_start_idx: int, window_length: int,
                                       jim1: int, jim2: int, inv_stkwl: float) -> List[float]:
        """向量化计算功率曲线（优化版本）
        
        使用 NumPy 广播和向量化操作替代嵌套循环，大幅提升性能
        
        Args:
            trace: 归一化后的道数据
            stacked: 当前叠加结果
            imo: 初始拾取样本号
            window_start_idx: 窗口起始索引偏移
            window_length: 窗口长度
            jim1: 搜索范围下限
            jim2: 搜索范围上限
            inv_stkwl: 归一化因子（1/窗口长度）
            
        Returns:
            功率曲线列表
        """
        power_curve = []
        search_range = jim2 - jim1 + 1
        
        # 预计算基础偏移
        base_offset = -window_start_idx + imo
        
        # 创建所有可能的偏移索引
        js_values = np.arange(jim1, jim2 + 1)
        lu_start_indices = base_offset + js_values
        
        # 向量化计算：对于每个偏移，计算窗口内的差异
        for js_idx, js in enumerate(js_values):
            lu_start = lu_start_indices[js_idx]
            lu_end = lu_start + window_length
            
            # 边界检查
            if lu_start < 0 or lu_end > len(trace):
                power_curve.append(float('inf'))
                continue
            
            # 提取窗口数据
            trace_window = trace[lu_start:lu_end]
            
            # 向量化计算 Lp 范数差异
            if len(trace_window) == window_length:
                diff = np.abs(stacked - trace_window)
                ws = np.sum(diff ** self.pjgl) * inv_stkwl
                power_curve.append(ws)
            else:
                power_curve.append(float('inf'))
        
        return power_curve
    
    def _estimate_error_from_power_curve(self, power_curve: List[float],
                                        jim1: int, jim2: int,
                                        dt: float, jm: int) -> float:
        """从功率曲线估计误差（基于半宽度）
        
        对应Fortran代码中的误差估计逻辑
        
        Args:
            power_curve: 功率曲线数组
            jim1: 搜索范围下限
            jim2: 搜索范围上限
            dt: 采样间隔（秒）
            jm: 最优偏移（样本数）
        
        Returns:
            误差估计（秒）
        """
        power_array = np.array(power_curve)
        
        # 过滤无效值
        valid_mask = np.isfinite(power_array)
        if not np.any(valid_mask):
            return self.emax
        
        # 找到最小值位置（对应Fortran的jmi）
        min_idx = np.argmin(power_array[valid_mask])
        valid_indices = np.where(valid_mask)[0]
        min_idx = valid_indices[min_idx]
        min_power = power_array[min_idx]
        
        # 计算半宽度阈值
        threshold = min_power * self.erl
        
        # 左侧交叉点（对应Fortran的swl逻辑）
        err_left = 0.0
        swl = 0
        ist = len(power_array)
        jmi = min_idx + 1  # Fortran索引从1开始
        
        if jmi == 1:
            swl = 2
        else:
            l = jmi - 1
            while swl == 0:
                if l < 0:
                    swl = 2
                    break
                if power_array[l] >= threshold:
                    icl = l
                    swl = 1
                else:
                    l -= 1
            
            if swl == 1:
                den = power_array[icl + 1] - power_array[icl]
                if abs(den) > 1e-5:
                    err_left = dt * (threshold - power_array[icl]) / den
                    err_left = (jmi - icl - 1) * dt - err_left
                else:
                    err_left = (jmi - icl - 1) * dt
        
        # 右侧交叉点（对应Fortran的swr逻辑）
        err_right = 0.0
        swr = 0
        
        if jmi == ist:
            swr = 2
        else:
            l = jmi
            while swr == 0:
                if l >= ist:
                    swr = 2
                    break
                if power_array[l] >= threshold:
                    icl = l
                    swr = 1
                else:
                    l += 1
            
            if swr == 1:
                den = power_array[icl] - power_array[icl - 1]
                if abs(den) > 1e-5:
                    err_right = dt * (threshold - power_array[icl - 1]) / den
                    err_right = (icl - jmi) * dt + err_right
                else:
                    err_right = (icl - jmi) * dt
        
        # 计算平均误差
        if swr == 1 and swl == 1:
            err = (err_right + err_left) / 2.0
        elif swl == 1:
            err = err_left
        elif swr == 1:
            err = err_right
        else:
            err = self.emax
        
        # 约束误差范围
        err = max(self.emin, min(self.emax, err))
        
        return err
    
    def estimate_errors(self, traces: List[np.ndarray],
                       stacked: np.ndarray,
                       times: np.ndarray,
                       time_shifts: List[float],
                       window_start_idx: int,
                       window_length: int) -> List[float]:
        """估计每道的误差（基于叠加功率半宽度）
        
        这是Haibo改进的误差估计方法，基于叠加功率曲线的半宽度
        
        Args:
            traces: 归一化后的道数据列表
            stacked: 叠加结果
            times: 时间数组
            time_shifts: 每道的时间偏移
            window_start_idx: 窗口起始样本索引
            window_length: 窗口长度
        
        Returns:
            每道的误差估计列表（秒）
        """
        errors = []
        dt = times[1] - times[0] if len(times) > 1 else 0.001
        
        for i, trace in enumerate(traces):
            # 计算叠加功率曲线
            shift_samples = int(time_shifts[i] / dt)
            center_idx = window_start_idx + shift_samples
            
            search_range = 50  # 搜索范围（样本数）
            power_curve = []
            
            for shift in range(-search_range, search_range + 1):
                start_idx = center_idx + shift
                if start_idx < 0 or start_idx + window_length > len(trace):
                    power_curve.append(float('inf'))
                    continue
                
                window_data = trace[start_idx:start_idx + window_length]
                diff = np.abs(stacked - window_data) ** self.pjgl
                power = np.mean(diff)
                power_curve.append(power)
            
            # 找到最小值位置
            power_array = np.array(power_curve)
            # 过滤掉无穷大值
            valid_mask = np.isfinite(power_array)
            if not np.any(valid_mask):
                # 如果没有有效值，使用默认误差
                errors.append(self.emax)
                continue
            
            min_idx = np.argmin(power_array[valid_mask])
            # 将相对索引转换为绝对索引
            valid_indices = np.where(valid_mask)[0]
            min_idx = valid_indices[min_idx]
            min_power = power_array[min_idx]
            
            # 计算半宽度（左右交叉点）
            threshold = min_power * self.erl
            
            # 左侧交叉点
            left_idx = min_idx
            while left_idx > 0:
                if not np.isfinite(power_array[left_idx]) or power_array[left_idx] >= threshold:
                    break
                left_idx -= 1
            
            # 右侧交叉点
            right_idx = min_idx
            while right_idx < len(power_array) - 1:
                if not np.isfinite(power_array[right_idx]) or power_array[right_idx] >= threshold:
                    break
                right_idx += 1
            
            # 线性插值计算精确位置
            if left_idx < min_idx:
                if left_idx + 1 < len(power_array):
                    denom = power_array[left_idx + 1] - power_array[left_idx]
                    if abs(denom) > 1e-5 and not np.isnan(denom) and not np.isinf(denom):
                        err_left = (min_idx - left_idx) * dt - \
                                  (threshold - power_array[left_idx]) / denom * dt
                    else:
                        err_left = (min_idx - left_idx) * dt
                else:
                    err_left = (min_idx - left_idx) * dt
            else:
                err_left = 0.0
            
            if right_idx > min_idx:
                if right_idx > 0:
                    denom = power_array[right_idx] - power_array[right_idx - 1]
                    if abs(denom) > 1e-5 and not np.isnan(denom) and not np.isinf(denom):
                        err_right = (right_idx - min_idx - 1) * dt + \
                                   (threshold - power_array[right_idx - 1]) / denom * dt
                    else:
                        err_right = (right_idx - min_idx) * dt
                else:
                    err_right = (right_idx - min_idx) * dt
            else:
                err_right = 0.0
            
            # 检查计算结果的有效性
            if np.isnan(err_left) or np.isinf(err_left):
                err_left = 0.0
            if np.isnan(err_right) or np.isinf(err_right):
                err_right = 0.0
            
            # 平均误差
            if err_left > 0 and err_right > 0:
                err = (err_left + err_right) / 2.0
            elif err_left > 0:
                err = err_left
            elif err_right > 0:
                err = err_right
            else:
                err = self.emax
            
            # 约束误差范围
            err = max(self.emin, min(self.emax, err))
            errors.append(err)
        
        return errors
    
    def calculate_correlation(self, traces: List[np.ndarray],
                              times: np.ndarray,
                              picks: Dict[int, float],
                              window_half_length: Optional[float] = None) -> Dict:
        """计算各道以拾取点为中心的窗口内波形的相关性
        
        计算所有道之间的皮尔逊相关系数，用于验证对齐效果
        
        Args:
            traces: 道数据列表
            times: 时间数组
            picks: 拾取时间字典 {trace_idx: pick_time}
            window_half_length: 窗口半长度（秒），如果为None则使用stkwl/2
        
        Returns:
            {
                'mean_correlation': float,  # 平均相关系数
                'min_correlation': float,   # 最小相关系数
                'max_correlation': float,   # 最大相关系数
                'std_correlation': float,   # 相关系数标准差
                'pairwise_correlations': np.ndarray,  # 所有道对的相关系数矩阵
                'trace_correlations': List[float],    # 每道与其他道的平均相关系数
            }
        """
        if not traces or len(traces) == 0:
            return {
                'mean_correlation': 0.0,
                'min_correlation': 0.0,
                'max_correlation': 0.0,
                'std_correlation': 0.0,
                'pairwise_correlations': np.array([]),
                'trace_correlations': []
            }
        
        # 确定窗口长度
        if window_half_length is None:
            window_half_length = self.stkwl / 2.0
        
        # 计算采样率
        if len(times) > 1:
            dt = times[1] - times[0]
            sampling_rate = 1.0 / dt
        else:
            dt = 0.001
            sampling_rate = 1000.0
        
        window_half_samples = int(round(window_half_length / dt))
        
        # 提取各道在拾取点附近的窗口数据
        window_segments = []
        valid_trace_indices = []
        
        for trace_idx, pick_time in picks.items():
            if trace_idx >= len(traces):
                continue
            
            # 找到拾取时间对应的样本索引
            pick_sample = int(round((pick_time - times[0]) / dt))
            
            # 计算窗口范围
            start_idx = pick_sample - window_half_samples
            end_idx = pick_sample + window_half_samples
            
            # 检查边界（确保窗口在数据范围内）
            # 注意：Python 切片是左闭右开 [start_idx:end_idx)，所以 end_idx 可以等于 len(traces[trace_idx])
            trace_len = len(traces[trace_idx])
            if start_idx < 0:
                # 如果起始索引小于0，调整窗口
                start_idx = 0
            if end_idx > trace_len:
                # 如果结束索引超出范围，调整窗口
                end_idx = trace_len
            
            # 确保窗口有效（至少要有2个样本才能计算相关性）
            if end_idx <= start_idx or end_idx - start_idx < 2:
                continue
            
            # 提取窗口数据
            window_data = traces[trace_idx][start_idx:end_idx]
            
            # 归一化窗口数据（使用最大绝对值）
            max_val = np.max(np.abs(window_data))
            if max_val > 0:
                window_data = window_data / max_val
            
            window_segments.append(window_data)
            valid_trace_indices.append(trace_idx)
        
        if len(window_segments) < 2:
            return {
                'mean_correlation': 0.0,
                'min_correlation': 0.0,
                'max_correlation': 0.0,
                'std_correlation': 0.0,
                'pairwise_correlations': np.array([]),
                'trace_correlations': []
            }
        
        # 确保所有窗口长度相同（取最小长度）
        min_length = min(len(seg) for seg in window_segments)
        window_segments = [seg[:min_length] for seg in window_segments]
        
        # 计算所有道对之间的相关系数（优先 numba 内核）
        n_traces = len(window_segments)
        segment_matrix = np.asarray(window_segments, dtype=np.float64)
        if self.enable_numba_kernels and HAS_NUMBA:
            try:
                correlation_matrix = _pairwise_corrcoef_numba(segment_matrix)
            except Exception:
                correlation_matrix = np.corrcoef(segment_matrix)
        else:
            correlation_matrix = np.corrcoef(segment_matrix)

        if correlation_matrix.shape != (n_traces, n_traces):
            correlation_matrix = np.zeros((n_traces, n_traces))
        else:
            correlation_matrix = np.nan_to_num(
                correlation_matrix, nan=0.0, posinf=0.0, neginf=0.0
            )
            np.fill_diagonal(correlation_matrix, 1.0)
        
        # 提取上三角矩阵（不包括对角线）的所有相关系数
        upper_triangle = np.triu(correlation_matrix, k=1)
        pairwise_correlations = upper_triangle[upper_triangle != 0]
        
        # 计算统计量
        if len(pairwise_correlations) > 0:
            mean_corr = np.mean(pairwise_correlations)
            min_corr = np.min(pairwise_correlations)
            max_corr = np.max(pairwise_correlations)
            std_corr = np.std(pairwise_correlations)
        else:
            mean_corr = 0.0
            min_corr = 0.0
            max_corr = 0.0
            std_corr = 0.0
        
        # 计算每道与其他道的平均相关系数
        trace_correlations = []
        for i in range(n_traces):
            # 排除自己（对角线）
            other_corrs = [correlation_matrix[i, j] for j in range(n_traces) if i != j]
            if len(other_corrs) > 0:
                trace_correlations.append(np.mean(other_corrs))
            else:
                trace_correlations.append(0.0)
        
        return {
            'mean_correlation': mean_corr,
            'min_correlation': min_corr,
            'max_correlation': max_corr,
            'std_correlation': std_corr,
            'pairwise_correlations': correlation_matrix,
            'trace_correlations': trace_correlations,
            'valid_trace_indices': valid_trace_indices
        }
