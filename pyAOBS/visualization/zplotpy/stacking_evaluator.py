"""
stacking_evaluator.py - 自适应叠加评价和可视化模块

提供定量评价和可视化功能，展示自适应叠加方法更新后的拾取时间相对于原时间的改进
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class StackingEvaluationResult:
    """自适应叠加评价结果"""
    # 原始数据
    original_picks: Dict[int, float]  # {trace_idx: original_pick_time}
    updated_picks: Dict[int, float]   # {trace_idx: updated_pick_time}
    time_shifts: Dict[int, float]     # {trace_idx: time_shift}
    errors: Dict[int, float]          # {trace_idx: error_estimate}
    
    # 统计指标
    mean_shift: float                 # 平均时间偏移
    std_shift: float                  # 时间偏移标准差
    max_shift: float                  # 最大时间偏移
    min_shift: float                  # 最小时间偏移
    
    mean_error: float                 # 平均误差估计
    std_error: float                  # 误差估计标准差
    max_error: float                  # 最大误差估计
    min_error: float                  # 最小误差估计
    
    # 质量指标
    quality_metric: float             # 叠加质量指标
    improvement_ratio: float          # 改进比例（基于误差减少）
    
    # 一致性指标
    coherence_before: float           # 更新前的相位一致性
    coherence_after: float            # 更新后的相位一致性


class StackingEvaluator:
    """自适应叠加评价器"""
    
    def __init__(self):
        """初始化评价器"""
        pass
    
    def evaluate(self, 
                 original_picks: Dict[int, float],
                 updated_picks: Dict[int, float],
                 time_shifts: Union[List[float], Dict[int, float]],
                 errors: Union[List[float], Dict[int, float]],
                 quality_metric: float,
                 traces: Optional[List[np.ndarray]] = None,
                 times: Optional[np.ndarray] = None) -> StackingEvaluationResult:
        """评价自适应叠加结果
        
        Args:
            original_picks: 原始拾取时间字典 {trace_idx: time}
            updated_picks: 更新后的拾取时间字典 {trace_idx: time}
            time_shifts: 时间偏移列表
            errors: 误差估计列表
            quality_metric: 叠加质量指标
            traces: 道数据列表（可选，用于计算一致性）
            times: 时间数组（可选，用于计算一致性）
        
        Returns:
            评价结果对象
        """
        # 统一为 trace_idx->value 字典（兼容旧版列表输入）
        if isinstance(time_shifts, dict):
            time_shifts_dict = {int(k): float(v) for k, v in time_shifts.items()}
        else:
            time_shifts_dict = {i: float(shift) for i, shift in enumerate(time_shifts)}
        if isinstance(errors, dict):
            errors_dict = {int(k): float(v) for k, v in errors.items()}
        else:
            errors_dict = {i: float(err) for i, err in enumerate(errors)}
        
        # 计算统计指标
        shifts_array = np.array(list(time_shifts_dict.values()), dtype=float)
        errors_array = np.array(list(errors_dict.values()), dtype=float)
        
        mean_shift = np.mean(shifts_array) if len(shifts_array) > 0 else 0.0
        std_shift = np.std(shifts_array) if len(shifts_array) > 0 else 0.0
        max_shift = np.max(shifts_array) if len(shifts_array) > 0 else 0.0
        min_shift = np.min(shifts_array) if len(shifts_array) > 0 else 0.0
        
        mean_error = np.mean(errors_array) if len(errors_array) > 0 else 0.0
        std_error = np.std(errors_array) if len(errors_array) > 0 else 0.0
        max_error = np.max(errors_array) if len(errors_array) > 0 else 0.0
        min_error = np.min(errors_array) if len(errors_array) > 0 else 0.0
        
        # 计算改进比例（基于误差的减少，这里简化处理）
        # 改进比例 = (原始误差 - 更新后误差) / 原始误差
        # 如果没有原始误差数据，使用误差估计的变化
        improvement_ratio = 0.0  # 默认值，需要更多信息才能计算
        
        # 计算相位一致性（如果提供了道数据）
        coherence_before = 0.0
        coherence_after = 0.0
        
        if traces is not None and times is not None:
            coherence_before = self._calculate_coherence(
                traces, original_picks, times
            )
            coherence_after = self._calculate_coherence(
                traces, updated_picks, times
            )
        
        return StackingEvaluationResult(
            original_picks=original_picks,
            updated_picks=updated_picks,
            time_shifts=time_shifts_dict,
            errors=errors_dict,
            mean_shift=mean_shift,
            std_shift=std_shift,
            max_shift=max_shift,
            min_shift=min_shift,
            mean_error=mean_error,
            std_error=std_error,
            max_error=max_error,
            min_error=min_error,
            quality_metric=quality_metric,
            improvement_ratio=improvement_ratio,
            coherence_before=coherence_before,
            coherence_after=coherence_after
        )
    
    def _calculate_coherence(self, 
                             traces: List[np.ndarray],
                             picks: Dict[int, float],
                             times: np.ndarray) -> float:
        """计算拾取点的相位一致性
        
        Args:
            traces: 道数据列表
            picks: 拾取时间字典
            times: 时间数组
        
        Returns:
            相位一致性（0-1之间）
        """
        if len(traces) == 0 or len(picks) == 0:
            return 0.0
        
        # 提取拾取点附近的信号段
        window_length = int(0.1 / (times[1] - times[0])) if len(times) > 1 else 50
        window_length = max(10, min(window_length, len(times) // 10))
        
        segments = []
        for trace_idx, pick_time in picks.items():
            if trace_idx >= len(traces):
                continue
            
            # 找到拾取时间对应的样本索引
            pick_idx = int((pick_time - times[0]) / (times[1] - times[0])) if len(times) > 1 else 0
            start_idx = max(0, pick_idx - window_length // 2)
            end_idx = min(len(traces[trace_idx]), start_idx + window_length)
            
            if end_idx > start_idx:
                segment = traces[trace_idx][start_idx:end_idx]
                segments.append(segment)
        
        if len(segments) < 2:
            return 0.0
        
        # 计算相位一致性
        try:
            from .signal_processor import SignalProcessor
        except ImportError:
            from signal_processor import SignalProcessor
        processor = SignalProcessor()
        
        phases = []
        for segment in segments:
            phase = processor.extract_phase(segment)
            phases.append(phase)
        
        coherence = processor.phase_coherence(phases)
        return np.mean(coherence) if len(coherence) > 0 else 0.0
    
    def create_comparison_plot(self, 
                              result: StackingEvaluationResult,
                              trace_indices: Optional[List[int]] = None) -> Figure:
        """创建对比图
        
        Args:
            result: 评价结果
            trace_indices: 要显示的道索引列表（如果为None则显示所有）
        
        Returns:
            matplotlib Figure对象
        """
        fig = plt.figure(figsize=(14, 10))
        
        if trace_indices is None:
            trace_indices = sorted(result.original_picks.keys())
        
        # 子图1：拾取时间对比（散点图）
        ax1 = plt.subplot(2, 2, 1)
        original_times = [result.original_picks.get(i, 0) for i in trace_indices]
        updated_times = [result.updated_picks.get(i, 0) for i in trace_indices]
        
        ax1.scatter(trace_indices, original_times, 
                   label='Original Picks', marker='o', s=50, alpha=0.7, color='blue')
        ax1.scatter(trace_indices, updated_times, 
                   label='Updated Picks', marker='x', s=50, alpha=0.7, color='red')
        
        # 绘制连接线显示偏移
        for i, trace_idx in enumerate(trace_indices):
            if trace_idx in result.original_picks and trace_idx in result.updated_picks:
                ax1.plot([trace_idx, trace_idx], 
                        [result.original_picks[trace_idx], result.updated_picks[trace_idx]],
                        'g--', alpha=0.3, linewidth=1)
        
        ax1.set_xlabel('Trace Index')
        ax1.set_ylabel('Pick Time (s)')
        ax1.set_title('Pick Time Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2：时间偏移分布（直方图）
        ax2 = plt.subplot(2, 2, 2)
        shifts = [result.time_shifts.get(i, 0) * 1000 for i in trace_indices]  # 转换为ms
        
        ax2.hist(shifts, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(result.mean_shift * 1000, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {result.mean_shift*1000:.2f} ms')
        ax2.set_xlabel('Time Shift (ms)')
        ax2.set_ylabel('Number of Traces')
        ax2.set_title('Time Shift Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 子图3：误差估计分布
        ax3 = plt.subplot(2, 2, 3)
        error_values = [result.errors.get(i, 0) * 1000 for i in trace_indices]  # 转换为ms
        
        ax3.hist(error_values, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax3.axvline(result.mean_error * 1000, color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {result.mean_error*1000:.2f} ms')
        ax3.set_xlabel('Error Estimate (ms)')
        ax3.set_ylabel('Number of Traces')
        ax3.set_title('Error Estimate Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 子图4：统计摘要（文本）
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        
        stats_text = f"""
Adaptive Stacking Evaluation Results

Time Shift Statistics:
  Mean:              {result.mean_shift*1000:7.2f} ms
  Std Dev:           {result.std_shift*1000:7.2f} ms
  Max:               {result.max_shift*1000:7.2f} ms
  Min:               {result.min_shift*1000:7.2f} ms

Error Estimate Statistics:
  Mean:              {result.mean_error*1000:7.2f} ms
  Std Dev:           {result.std_error*1000:7.2f} ms
  Max:               {result.max_error*1000:7.2f} ms
  Min:               {result.min_error*1000:7.2f} ms

Quality Metrics:
  Stacking Quality:  {result.quality_metric:.6f}
  Coherence (Before): {result.coherence_before:.4f}
  Coherence (After):  {result.coherence_after:.4f}
  Coherence Improvement: {result.coherence_after - result.coherence_before:+.4f}

Updated Traces:      {len(result.updated_picks)}
        """
        
        ax4.text(0.1, 0.5, stats_text, fontsize=10, 
                family='monospace', verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def create_shift_visualization(self,
                                   result: StackingEvaluationResult,
                                   offsets: np.ndarray,
                                   trace_indices: Optional[List[int]] = None) -> Figure:
        """创建时间偏移可视化图（在炮检距-时间域）
        
        Args:
            result: 评价结果
            offsets: 炮检距数组
            trace_indices: 要显示的道索引列表
        
        Returns:
            matplotlib Figure对象
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        if trace_indices is None:
            trace_indices = sorted(result.original_picks.keys())
        
        # 获取对应的炮检距
        trace_offsets = [offsets[i] if i < len(offsets) else 0.0 for i in trace_indices]
        
        # 上图：原始和更新后的拾取时间
        original_times = [result.original_picks.get(i, 0) for i in trace_indices]
        updated_times = [result.updated_picks.get(i, 0) for i in trace_indices]
        
        ax1.plot(trace_offsets, original_times, 'o-', label='Original Picks', 
                color='blue', markersize=6, linewidth=1.5, alpha=0.7)
        ax1.plot(trace_offsets, updated_times, 'x-', label='Updated Picks', 
                color='red', markersize=6, linewidth=1.5, alpha=0.7)
        
        # 绘制偏移箭头
        for i, trace_idx in enumerate(trace_indices):
            if trace_idx in result.original_picks and trace_idx in result.updated_picks:
                offset = trace_offsets[i]
                orig_time = result.original_picks[trace_idx]
                upd_time = result.updated_picks[trace_idx]
                if abs(upd_time - orig_time) > 1e-6:
                    ax1.annotate('', xy=(offset, upd_time), xytext=(offset, orig_time),
                               arrowprops=dict(arrowstyle='->', color='green', 
                                             lw=1.5, alpha=0.5))
        
        ax1.set_xlabel('Offset (km)')
        ax1.set_ylabel('Pick Time (s)')
        ax1.set_title('Pick Time Comparison (Offset-Time Domain)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()  # 时间向下
        
        # 下图：时间偏移量
        shifts = [result.time_shifts.get(i, 0) * 1000 for i in trace_indices]  # 转换为ms
        
        colors = ['green' if s >= 0 else 'red' for s in shifts]
        ax2.bar(range(len(trace_indices)), shifts, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(0, color='black', linestyle='-', linewidth=1)
        ax2.axhline(result.mean_shift * 1000, color='blue', linestyle='--', 
                   linewidth=2, label=f'Mean: {result.mean_shift*1000:.2f} ms')
        
        ax2.set_xlabel('Trace Index')
        ax2.set_ylabel('Time Shift (ms)')
        ax2.set_title('Time Shift (Positive=Delay, Negative=Advance)')
        ax2.set_xticks(range(len(trace_indices)))
        ax2.set_xticklabels([f'{i}' for i in trace_indices], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def print_evaluation_report(self, result: StackingEvaluationResult):
        """打印评价报告
        
        Args:
            result: 评价结果
        """
        print("\n" + "="*70)
        print("自适应叠加评价报告 (Adaptive Stacking Evaluation Report)")
        print("="*70)
        
        print("\n【时间偏移统计】")
        print(f"  平均值 (Mean):     {result.mean_shift*1000:8.2f} ms")
        print(f"  标准差 (Std):      {result.std_shift*1000:8.2f} ms")
        print(f"  最大值 (Max):      {result.max_shift*1000:8.2f} ms")
        print(f"  最小值 (Min):      {result.min_shift*1000:8.2f} ms")
        print(f"  范围 (Range):      {(result.max_shift - result.min_shift)*1000:8.2f} ms")
        
        print("\n【误差估计统计】")
        print(f"  平均值 (Mean):     {result.mean_error*1000:8.2f} ms")
        print(f"  标准差 (Std):      {result.std_error*1000:8.2f} ms")
        print(f"  最大值 (Max):      {result.max_error*1000:8.2f} ms")
        print(f"  最小值 (Min):      {result.min_error*1000:8.2f} ms")
        
        print("\n【质量指标】")
        print(f"  叠加质量指标:      {result.quality_metric:.6f}")
        print(f"  相位一致性(前):    {result.coherence_before:.4f}")
        print(f"  相位一致性(后):    {result.coherence_after:.4f}")
        coherence_improvement = result.coherence_after - result.coherence_before
        print(f"  一致性改进:         {coherence_improvement:+.4f} ({coherence_improvement/result.coherence_before*100:+.2f}%)" 
              if result.coherence_before > 0 else "  一致性改进:         N/A")
        
        print("\n【更新统计】")
        print(f"  更新道数:           {len(result.updated_picks)}")
        print(f"  平均偏移量:         {result.mean_shift*1000:.2f} ms")
        
        # 计算偏移量分布
        shifts_array = np.array(list(result.time_shifts.values()))
        if len(shifts_array) > 0:
            positive_shifts = np.sum(shifts_array > 0)
            negative_shifts = np.sum(shifts_array < 0)
            zero_shifts = np.sum(shifts_array == 0)
            print(f"  正向偏移道数:       {positive_shifts} ({positive_shifts/len(shifts_array)*100:.1f}%)")
            print(f"  负向偏移道数:       {negative_shifts} ({negative_shifts/len(shifts_array)*100:.1f}%)")
            print(f"  无偏移道数:         {zero_shifts} ({zero_shifts/len(shifts_array)*100:.1f}%)")
        
        print("\n" + "="*70 + "\n")
