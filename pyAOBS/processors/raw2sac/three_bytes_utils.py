#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用3字节有符号整数转换工具模块

用于OBS Raw和OBEM TSM等格式的3字节数据解码。
支持向量化批量转换，适用于高性能数据处理。

使用场景：
1. OBS Raw格式：每个通道3字节（高字节、中字节、低字节）
2. OBEM TSM格式：每个通道3字节（低字节、中字节、高字节）

注意：
- 两种格式的字节顺序不同，但都使用3字节有符号整数表示
- 本模块提供统一的接口，支持不同的字节顺序

Author: Haibo Huang, 2025/01/XX
"""

import os
import numpy as np

# 尝试导入 Numba（可选，用于性能优化）
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@jit(nopython=True, cache=True)
def _three_bytes_to_int_vectorized_numba(high_byte, mid_byte, low_byte):
    """Numba JIT编译版本的核心转换逻辑（内部函数）
    
    这个函数被设计为与Numba完全兼容，使用纯NumPy向量化操作。
    Numba可以进一步优化这些NumPy操作。
    
    Args:
        high_byte: 高字节数组（uint8）
        mid_byte: 中字节数组（uint8）
        low_byte: 低字节数组（uint8）
    
    Returns:
        有符号32位整数数组（int32）
    """
    # 转换为合适的类型（Numba要求显式类型）
    high_byte = high_byte.astype(np.uint8)
    mid_byte = mid_byte.astype(np.uint8)
    low_byte = low_byte.astype(np.uint8)
    
    # 符号扩展高字节为16位（对应C的 (short)high_byte）
    # 使用条件表达式（Numba支持）
    mask = (high_byte & 0x80) != 0
    signed_high_16 = np.where(mask, 
                              high_byte | 0xFF00,  # 负数：符号扩展
                              high_byte)            # 正数：零扩展
    
    # 组合成32位整数（little-endian格式）
    # byteval[0] = low_byte, byteval[1] = mid_byte
    # byteval[2] = signed_high_16的低字节, byteval[3] = signed_high_16的高字节
    value = (low_byte.astype(np.int32) | 
             (mid_byte.astype(np.int32) << 8) |
             ((signed_high_16 & 0xFF).astype(np.int32) << 16) |
             (((signed_high_16 >> 8) & 0xFF).astype(np.int32) << 24))
    
    # 转换为有符号32位整数
    # 如果24位整数的最高位（bit 23）是1，符号扩展到bit 24-31
    mask_24bit_negative = (value & 0x800000) != 0
    value = np.where(mask_24bit_negative, value | 0xFF000000, value)
    
    # 确保是有符号32位整数
    # 限制到32位范围并转换为有符号整数
    value_uint32 = (value & 0xFFFFFFFF).astype(np.uint32)
    
    # 使用位操作转换为有符号整数（Numba不支持view方法）
    mask_sign = (value_uint32 & 0x80000000) != 0
    value_int32 = np.where(mask_sign,
                          (value_uint32 - 0x100000000).astype(np.int32),
                          value_uint32.astype(np.int32))
    
    return value_int32


def three_bytes_to_int_vectorized(data_bytes: np.ndarray, 
                                  byte_order: str = 'hml') -> np.ndarray:
    """批量将3字节数组转换为有符号32位整数（向量化版本）
    
    通用接口，支持不同的字节顺序排列方式。
    
    字节顺序模式：
    - 'hml' (high-mid-low): OBS Raw格式，文件中字节顺序为高字节、中字节、低字节
    - 'lmh' (low-mid-high): OBEM TSM格式，文件中字节顺序为低字节、中字节、高字节
    
    使用NumPy向量化操作，如果Numba可用且启用，则使用JIT编译加速。
    
    Args:
        data_bytes: shape为 (n, 3) 的uint8数组，每行3个字节
        byte_order: 字节顺序，'hml'或'lmh'（默认：'hml'，OBS Raw格式）
    
    Returns:
        shape为 (n,) 的int32数组，包含转换后的有符号32位整数
    
    Examples:
        >>> # OBS Raw格式：文件中字节顺序为 (h, m, l)
        >>> data = np.array([[0x80, 0x00, 0x01], [0x7F, 0xFF, 0xFF]], dtype=np.uint8)
        >>> result = three_bytes_to_int_vectorized(data, byte_order='hml')
        
        >>> # OBEM TSM格式：文件中字节顺序为 (l, m, h)
        >>> data = np.array([[0x01, 0x00, 0x80], [0xFF, 0xFF, 0x7F]], dtype=np.uint8)
        >>> result = three_bytes_to_int_vectorized(data, byte_order='lmh')
    """
    if data_bytes.shape[1] != 3:
        raise ValueError(f"Expected shape (n, 3), got {data_bytes.shape}")
    
    # 根据字节顺序提取字节
    if byte_order.lower() == 'hml':
        # OBS Raw格式：文件中为 (h, m, l)，转换为little-endian格式 (l, m, h)
        high_byte = data_bytes[:, 0]
        mid_byte = data_bytes[:, 1]
        low_byte = data_bytes[:, 2]
    elif byte_order.lower() == 'lmh':
        # OBEM TSM格式：文件中为 (l, m, h)，直接使用
        low_byte = data_bytes[:, 0]
        mid_byte = data_bytes[:, 1]
        high_byte = data_bytes[:, 2]
    else:
        raise ValueError(f"Unknown byte_order: {byte_order}. Must be 'hml' or 'lmh'")
    
    # 默认使用纯NumPy向量化版本（经过优化的版本，性能已经很好）
    # 只有在明确启用时才使用Numba版本
    # 注意：Numba版本在某些情况下可能反而更慢，因此默认禁用
    use_numba = NUMBA_AVAILABLE and os.environ.get('THREE_BYTES_USE_NUMBA', '').lower() in ('1', 'true', 'yes')
    
    if use_numba:
        return _three_bytes_to_int_vectorized_numba(high_byte, mid_byte, low_byte)
    
    # 使用纯NumPy向量化版本（默认，经过优化，性能优秀）
    # 符号扩展高字节为16位（对应C的 (short)high_byte）
    # 如果最高位是1，符号扩展到16位：0x80 → 0xFF80
    signed_high_16 = np.where(high_byte & 0x80, 
                              high_byte | 0xFF00,  # 负数：符号扩展
                              high_byte)            # 正数：零扩展
    
    # 组合成32位整数（little-endian格式）
    # byteval[0] = low_byte, byteval[1] = mid_byte
    # byteval[2] = signed_high_16的低字节, byteval[3] = signed_high_16的高字节
    value = (low_byte.astype(np.int32) | 
             (mid_byte.astype(np.int32) << 8) |
             ((signed_high_16 & 0xFF).astype(np.int32) << 16) |
             (((signed_high_16 >> 8) & 0xFF).astype(np.int32) << 24))
    
    # 转换为有符号32位整数
    # 如果24位整数的最高位（bit 23）是1，符号扩展到bit 24-31
    mask_24bit_negative = (value & 0x800000) != 0
    # 如果bit 24-31还没有被设置，则设置它们（符号扩展）
    value = np.where(mask_24bit_negative, value | 0xFF000000, value)
    
    # 确保是有符号32位整数
    # 限制到32位范围并转换为有符号整数
    value_uint32 = (value & 0xFFFFFFFF).astype(np.uint32)
    # 使用view方法进行符号转换（这是最高效的方法）
    value_int32 = value_uint32.view(np.int32)
    
    return value_int32

