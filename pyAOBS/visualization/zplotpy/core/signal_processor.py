"""
signal_processor.py - ZPLOT 信号处理模块

实现信号处理功能：
- Hilbert变换
- 相位提取
- 包络提取
- 相位一致性计算

基于Haibo Huang 2023年11月的改进版本
"""

import numpy as np
from typing import List, Optional
from scipy import signal


class SignalProcessor:
    """信号处理类 - 实现Hilbert变换等功能
    
    基于原始Fortran代码中的hilbert子程序（bandpass.f）
    原始作者：Jonathas Maciel
    修改者：Haibo Huang (2023.11)
    """
    
    def __init__(self):
        """初始化信号处理器"""
        pass
    
    def hilbert_transform(self, x: np.ndarray, output_mode: int = 0) -> np.ndarray:
        """Hilbert变换
        
        Args:
            x: 输入信号
            output_mode: 输出模式
                0 = 标准Hilbert变换（虚部）
                1 = 输出相位
                2 = 输出包络
        
        Returns:
            处理后的信号
        """
        if len(x) == 0:
            return x
        
        # 计算解析信号（Hilbert变换）
        # scipy.signal.hilbert返回解析信号 z(t) = x(t) + i*H[x(t)]
        analytic_signal = signal.hilbert(x)
        
        if output_mode == 0:
            # 标准Hilbert变换（虚部）
            return np.imag(analytic_signal)
        elif output_mode == 1:
            # 输出相位
            return np.angle(analytic_signal)
        elif output_mode == 2:
            # 输出包络
            return np.abs(analytic_signal)
        else:
            # 默认返回标准Hilbert变换
            return np.imag(analytic_signal)
    
    def extract_phase(self, x: np.ndarray) -> np.ndarray:
        """提取瞬时相位
        
        Args:
            x: 输入信号
        
        Returns:
            瞬时相位数组（弧度，范围[-π, π]）
        """
        if len(x) == 0:
            return x
        
        analytic_signal = signal.hilbert(x)
        return np.angle(analytic_signal)
    
    def extract_envelope(self, x: np.ndarray) -> np.ndarray:
        """提取信号包络
        
        Args:
            x: 输入信号
        
        Returns:
            信号包络数组（非负）
        """
        if len(x) == 0:
            return x
        
        analytic_signal = signal.hilbert(x)
        return np.abs(analytic_signal)
    
    def phase_coherence(self, phases: List[np.ndarray]) -> np.ndarray:
        """计算相位一致性
        
        相位一致性定义为：
        C = (1/N) * |Σ e^(iφ)| = (1/N) * sqrt((Σcos(φ))² + (Σsin(φ))²)
        
        Args:
            phases: 相位数组列表（每道一个相位数组）
        
        Returns:
            相位一致性数组（长度与相位数组相同，范围[0, 1]）
            - C = 1: 完全同相
            - C = 0: 完全随机相位
        """
        if not phases or len(phases) == 0:
            return np.array([])
        
        n_traces = len(phases)
        n_samples = len(phases[0])
        
        # 验证所有相位数组长度相同
        for phase in phases:
            if len(phase) != n_samples:
                raise ValueError("All phase arrays must have the same length")
        
        # 计算相位余弦和、正弦和
        cos_sum = np.zeros(n_samples)
        sin_sum = np.zeros(n_samples)
        
        for phase in phases:
            cos_sum += np.cos(phase)
            sin_sum += np.sin(phase)
        
        # 计算相位一致性：C = |Σe^(iφ)|/N
        coherence = np.sqrt(cos_sum**2 + sin_sum**2) / n_traces
        
        return coherence
    
    def analytic_signal(self, x: np.ndarray) -> np.ndarray:
        """计算解析信号
        
        Args:
            x: 输入信号
        
        Returns:
            解析信号 z(t) = x(t) + i*H[x(t)]
        """
        if len(x) == 0:
            return x
        
        return signal.hilbert(x)
