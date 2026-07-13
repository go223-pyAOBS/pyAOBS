"""
auto_picker.py - ZPLOT 自动拾取模块

实现基于能量比的自动拾取算法，对应原始Fortran代码中的pick()子程序
原始作者：Brian L.N. Kennett (ANU, 2002)
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
import warnings

try:
    from .src_kernel_bridge import SrcPickKernelBridge
except ImportError:
    from src_kernel_bridge import SrcPickKernelBridge


class AutoPicker:
    """自动拾取类 - 实现基于能量比的震相拾取"""
    
    def __init__(self, window_length: float = 0.1, 
                 min_energy_ratio: float = 1.5,
                 search_start: Optional[float] = None,
                 search_end: Optional[float] = None,
                 vred: float = 0.0,
                 offset: Optional[float] = None):
        """初始化自动拾取器
        
        Args:
            window_length: 窗口长度（秒），默认0.1秒
            min_energy_ratio: 最小能量比阈值，默认1.5
            search_start: 搜索起始时间（秒，折合时间），None表示从数据开始
            search_end: 搜索结束时间（秒，折合时间），None表示到数据结束
            vred: 折合速度 (km/s)，如果>0，search_start和search_end将被视为折合时间
            offset: 炮检距 (km)，用于折合时间转换，如果None则使用平均offset
        """
        self.window_length = window_length
        self.min_energy_ratio = min_energy_ratio
        self.search_start = search_start
        self.search_end = search_end
        self.vred = vred
        self.offset = offset
        self.src_pick_kernel = SrcPickKernelBridge()
    
    def pick_trace(self, trace: np.ndarray, times: np.ndarray,
                   initial_time: Optional[float] = None,
                   search_window: Optional[float] = None,
                   offset: Optional[float] = None) -> Optional[Dict]:
        """对单道进行自动拾取
        
        Args:
            trace: 地震道数据
            times: 时间数组
            initial_time: 初始时间提示（可选），用于缩小搜索范围
            search_window: 搜索窗口（秒），如果提供initial_time，在其附近搜索
        
        Returns:
            拾取结果字典，包含：
            {
                'pick_time': float,      # 拾取时间（秒）
                'pick_index': int,        # 拾取点索引
                'energy_ratio': float,    # 能量比
                'quality': str,           # 质量等级（'high'/'medium'/'low'）
                'confidence': float       # 置信度（0-1）
            }
            如果未找到拾取点，返回None
        """
        if len(trace) == 0 or len(times) == 0:
            return None
        
        if len(trace) != len(times):
            raise ValueError("trace and times must have the same length")
        
        # 计算采样间隔
        if len(times) > 1:
            dt = times[1] - times[0]
        else:
            return None
        
        # 确定搜索范围（传入offset用于折合时间转换）
        start_idx, end_idx = self._determine_search_range(
            times, initial_time, search_window, offset
        )
        
        if start_idx >= end_idx:
            return None
        
        # 调试输出：显示搜索范围
        if start_idx < len(times) and end_idx <= len(times):
            search_start_time = times[start_idx]
            search_end_time = times[end_idx - 1] if end_idx > 0 else times[-1]
            # 如果使用了折合时间，也显示折合时间范围
            if self.vred > 0 and offset is not None:
                reduced_start = search_start_time - abs(offset) / self.vred
                reduced_end = search_end_time - abs(offset) / self.vred
                print(f"自动拾取搜索范围: 原始时间 {search_start_time:.3f}s - {search_end_time:.3f}s, "
                      f"折合时间 {reduced_start:.3f}s - {reduced_end:.3f}s "
                      f"(索引: {start_idx} - {end_idx-1}, 样本数: {end_idx-start_idx})")
            else:
                print(f"自动拾取搜索范围: {search_start_time:.3f}s - {search_end_time:.3f}s "
                      f"(索引: {start_idx} - {end_idx-1}, 样本数: {end_idx-start_idx})")
        
        # 执行能量比拾取
        pick_idx, max_ratio = self._pick_energy_ratio(
            trace, times, start_idx, end_idx
        )
        
        # 检查是否找到了有效的拾取点
        if pick_idx is None:
            print(f"未找到拾取点（能量比可能小于阈值 {self.min_energy_ratio:.3f}）")
            return None
        
        if max_ratio < self.min_energy_ratio:
            print(f"警告：找到的能量比 {max_ratio:.3f} 小于阈值 {self.min_energy_ratio:.3f}")
            return None
        
        # 调试输出：显示拾取结果
        if pick_idx < len(times):
            pick_time = times[pick_idx]
            print(f"自动拾取成功: 时间={pick_time:.3f}s, 能量比={max_ratio:.3f}, 索引={pick_idx}")
        
        # 计算拾取时间
        pick_time = times[pick_idx]
        
        # 评估拾取质量
        quality_info = self.evaluate_pick_quality(trace, times, pick_time)
        
        return {
            'pick_time': pick_time,
            'pick_index': pick_idx,
            'energy_ratio': max_ratio,
            'quality': quality_info['quality'],
            'confidence': quality_info['confidence'],
            'snr': quality_info.get('snr', 0.0)
        }
    
    def _determine_search_range(self, times: np.ndarray,
                                initial_time: Optional[float] = None,
                                search_window: Optional[float] = None,
                                offset: Optional[float] = None) -> Tuple[int, int]:
        """确定搜索范围
        
        Args:
            times: 时间数组（原始时间）
            initial_time: 初始时间提示（原始时间或折合时间，取决于vred）
            search_window: 搜索窗口（秒）
            offset: 炮检距 (km)，用于折合时间转换
        
        Returns:
            (start_idx, end_idx): 搜索范围的起始和结束索引（Python 0-based，end_idx不包含）
        """
        npts = len(times)
        
        if len(times) == 0:
            return 0, 0
        
        dt = times[1] - times[0] if len(times) > 1 else 0.001
        
        # 如果使用了折合时间，需要将折合时间转换为原始时间
        # 转换公式：t_original = t_reduced + offset / vred
        use_offset = offset if offset is not None else self.offset
        
        # 优先级1：如果提供了初始时间和搜索窗口，在其附近搜索
        if initial_time is not None and search_window is not None:
            # 如果initial_time是折合时间，需要转换为原始时间
            search_initial_time = initial_time
            if self.vred > 0 and use_offset is not None:
                search_initial_time = initial_time + abs(use_offset) / self.vred
            
            # 在初始时间附近搜索
            center_idx = np.argmin(np.abs(times - search_initial_time))
            half_window_samples = int(round(search_window / 2 / dt))
            start_idx = max(0, center_idx - half_window_samples)
            end_idx = min(npts, center_idx + half_window_samples + 1)  # +1因为end_idx不包含
        # 优先级2：如果设置了search_start或search_end，使用这些值
        elif self.search_start is not None or self.search_end is not None:
            start_idx = 0
            end_idx = npts
            
            # 将折合时间转换为原始时间（如果使用了折合时间）
            search_start_original = self.search_start
            search_end_original = self.search_end
            
            if self.vred > 0 and use_offset is not None:
                # 用户输入的是折合时间，需要转换为原始时间
                if search_start_original is not None:
                    search_start_original = search_start_original + abs(use_offset) / self.vred
                if search_end_original is not None:
                    search_end_original = search_end_original + abs(use_offset) / self.vred
            
            if search_start_original is not None:
                # 找到最接近search_start_original的索引
                # 如果search_start_original在times范围内，使用精确匹配；否则使用边界
                if search_start_original < times[0]:
                    start_idx = 0
                elif search_start_original > times[-1]:
                    start_idx = npts  # 无效范围
                else:
                    start_idx = max(0, np.argmin(np.abs(times - search_start_original)))
            if search_end_original is not None:
                # 找到最接近search_end_original的索引，+1因为end_idx不包含
                # 如果search_end_original在times范围内，使用精确匹配；否则使用边界
                if search_end_original < times[0]:
                    end_idx = 0  # 无效范围
                elif search_end_original > times[-1]:
                    end_idx = npts
                else:
                    end_idx = min(npts, np.argmin(np.abs(times - search_end_original)) + 1)
            
            # 确保范围有效
            if start_idx >= end_idx:
                print(f"警告：搜索时间范围无效: 折合时间 start={self.search_start}, end={self.search_end}, "
                      f"原始时间 start={search_start_original}, end={search_end_original}")
                return 0, 0
        else:
            # 使用全部数据
            start_idx = 0
            end_idx = npts
        
        # 确保范围有效
        if start_idx >= end_idx:
            return 0, 0
        
        return start_idx, end_idx
    
    def _pick_energy_ratio(self, trace: np.ndarray, times: np.ndarray,
                           start_idx: int, end_idx: int) -> Tuple[Optional[int], float]:
        """使用能量比方法拾取（对应Fortran的pick()子程序）
        
        Args:
            trace: 地震道数据
            times: 时间数组
            start_idx: 搜索起始索引（Python 0-based）
            end_idx: 搜索结束索引（Python 0-based，不包含）
        
        Returns:
            (pick_idx, max_ratio): 拾取点索引和最大能量比
        """
        dt = times[1] - times[0] if len(times) > 1 else 0.001
        nwind = int(round(self.window_length / dt))
        if nwind < 1:
            nwind = 1
        pick_idx, max_ratio, iflag = self.src_pick_kernel.pick_energy_ratio(
            np.asarray(trace),
            start_idx=int(start_idx),
            end_idx=int(end_idx),
            nwind=int(nwind),
            dt=float(dt),
            min_energy_ratio=float(self.min_energy_ratio),
        )
        if iflag != 1 or pick_idx < 0 or pick_idx >= len(times):
            return None, 0.0
        return int(pick_idx), float(max_ratio)
    
    def pick_traces(self, traces: List[np.ndarray], times: np.ndarray,
                    initial_times: Optional[List[float]] = None,
                    search_window: Optional[float] = None,
                    offsets: Optional[np.ndarray] = None,
                    progress_callback: Optional[callable] = None) -> List[Optional[Dict]]:
        """批量拾取多道
        
        Args:
            traces: 地震道数据列表
            times: 时间数组
            initial_times: 初始时间提示列表（可选）
            search_window: 搜索窗口（秒），如果提供initial_times，在其附近搜索
            progress_callback: 进度回调函数 callback(current, total)
        
        Returns:
            拾取结果列表，每个元素是一个拾取结果字典或None
        """
        results = []
        n_traces = len(traces)
        
        for i, trace in enumerate(traces):
            initial_time = None
            if initial_times and i < len(initial_times):
                initial_time = initial_times[i]
            
            # 获取该道的offset（用于折合时间转换）
            trace_offset = None
            if offsets is not None and i < len(offsets):
                trace_offset = offsets[i]
            elif self.offset is not None:
                trace_offset = self.offset
            
            result = self.pick_trace(trace, times, initial_time, search_window, trace_offset)
            results.append(result)
            
            # 调用进度回调
            if progress_callback:
                progress_callback(i + 1, n_traces)
        
        return results
    
    def evaluate_pick_quality(self, trace: np.ndarray, times: np.ndarray,
                              pick_time: float) -> Dict:
        """评估拾取质量
        
        Args:
            trace: 地震道数据
            times: 时间数组
            pick_time: 拾取时间
        
        Returns:
            质量评估字典，包含：
            {
                'quality': str,      # 质量等级（'high'/'medium'/'low'）
                'confidence': float,  # 置信度（0-1）
                'snr': float,        # 信噪比
                'energy_ratio': float # 能量比
            }
        """
        # 找到拾取点索引
        pick_idx = np.argmin(np.abs(times - pick_time))
        
        # 计算采样间隔
        dt = times[1] - times[0] if len(times) > 1 else 0.001
        
        # 计算窗口长度（样本数）
        nwind = int(round(self.window_length / dt))
        if nwind < 1:
            nwind = 1
        
        # 计算拾取点前后的能量
        before_start = max(0, pick_idx - nwind * 2)
        before_end = pick_idx - nwind
        after_start = pick_idx + nwind
        after_end = min(len(trace), pick_idx + nwind * 2)
        
        if before_end <= before_start or after_end <= after_start:
            return {
                'quality': 'low',
                'confidence': 0.0,
                'snr': 0.0,
                'energy_ratio': 0.0
            }
        
        # 计算噪声能量（拾取前）
        noise_energy = np.mean(trace[before_start:before_end] ** 2)
        
        # 计算信号能量（拾取后）
        signal_energy = np.mean(trace[after_start:after_end] ** 2)
        
        # 计算信噪比
        if noise_energy > 0:
            snr = signal_energy / noise_energy
        else:
            snr = float('inf') if signal_energy > 0 else 0.0
        
        # 重新计算能量比（用于质量评估）
        if before_end > before_start:
            before_sum = np.sum(trace[before_start:before_end] ** 2)
        else:
            before_sum = 0.0
        
        if after_end > after_start:
            after_sum = np.sum(trace[after_start:after_end] ** 2)
        else:
            after_sum = 0.0
        
        if before_sum > 0:
            energy_ratio = after_sum / before_sum
        else:
            energy_ratio = float('inf') if after_sum > 0 else 0.0
        
        # 质量分级
        if energy_ratio > 3.0 and snr > 5.0:
            quality = 'high'
            confidence = min(1.0, 0.7 + 0.3 * (energy_ratio - 3.0) / 5.0)
        elif energy_ratio > 2.0 and snr > 3.0:
            quality = 'medium'
            confidence = min(0.7, 0.4 + 0.3 * (energy_ratio - 2.0) / 1.0)
        elif energy_ratio > 1.5 and snr > 2.0:
            quality = 'low'
            confidence = min(0.4, 0.2 + 0.2 * (energy_ratio - 1.5) / 0.5)
        else:
            quality = 'low'
            confidence = 0.1
        
        return {
            'quality': quality,
            'confidence': confidence,
            'snr': snr,
            'energy_ratio': energy_ratio
        }
