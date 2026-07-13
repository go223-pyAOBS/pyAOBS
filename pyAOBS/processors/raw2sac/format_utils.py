#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一格式转换工具模块

整合了OBS Raw、OBEM TSM、SAC to SEGY等格式转换的通用工具函数：
1. 十六进制转换工具（来自 fntime_v1_1.py）
2. 文件名时间解析工具（来自 fntime_v1_1.py）
3. 3字节有符号整数转换工具（来自 three_bytes_utils.py）
4. 字节交换和浮点格式转换工具（来自 sac2y_v2_1_obspy.py）
5. 时间字符串解析工具（来自 sac2y_v2_1_obspy.py）
6. 字符串工具函数（来自 sac2y_v2_1_obspy.py）
7. 文件读取工具（来自 sac2y_v2_1_obspy.py）
8. 坐标投影工具（来自 sac2y_v2_1_obspy.py，需要pyproj）

便于后续整合到 pyAOBS 代码包中。

Author: Haibo Huang, 2025/01/XX
"""

import os
import re
import numpy as np
from typing import Dict, Tuple, Optional, List
from datetime import datetime

# 尝试导入可选依赖
try:
    from pyproj import Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    Transformer = None  # 类型提示占位符

try:
    from obspy.core import UTCDateTime
    OBSPY_AVAILABLE = True
except ImportError:
    OBSPY_AVAILABLE = False
    UTCDateTime = None  # 类型提示占位符

# ============================================================================
# 十六进制转换工具（来自 fntime_v1_1.py）
# ============================================================================

def hexto_int(hex_char: str) -> int:
    """Convert a single hexadecimal character to integer.
    
    Corresponds to C function: HextoInt(unsigned char datebyte)
    
    Args:
        hex_char: Single hexadecimal character (0-9, A-F, a-f)
        
    Returns:
        Integer value (0-15)
        
    Raises:
        ValueError: If input is not a valid hexadecimal character
    """
    if len(hex_char) != 1:
        raise ValueError(f"Expected single character, got: {hex_char}")
    
    hex_char = hex_char.upper()
    if not (('0' <= hex_char <= '9') or ('A' <= hex_char <= 'F')):
        raise ValueError(f"Invalid hex character: {hex_char}")
    
    if ord(hex_char) < 58:  # '0'-'9'
        return ord(hex_char) - 48
    else:  # 'A'-'F'
        return ord(hex_char) - 55


def hexto_dec(hex_str: str) -> int:
    """Convert hexadecimal string to decimal integer.
    
    Corresponds to C function: HextoDec(unsigned int a0, unsigned char *p)
    Uses weighted conversion: sum(pow(16, a0-i) * j)
    
    Args:
        hex_str: Hexadecimal string (e.g., "5DD", "FFF")
        
    Returns:
        Decimal integer value
    """
    if not hex_str:
        return 0
    
    result = 0
    a0 = len(hex_str) - 1  # Last index (equivalent to --a1 in C code)
    
    for i, char in enumerate(hex_str):
        char_lower = char.lower()
        
        # Convert character to integer value
        if 'a' <= char_lower <= 'f':
            j = ord(char_lower) - 87
        elif 'A' <= char <= 'F':
            j = ord(char) - 55
        else:
            j = ord(char) - 48
        
        # Weighted sum: pow(16, a0 - i) * j
        weight = pow(16, a0 - i)
        result += weight * j
    
    return result


# ============================================================================
# 文件名时间解析工具（来自 fntime_v1_1.py）
# ============================================================================

def parse_datetime_from_filename(filename: str) -> Dict[str, int]:
    """Parse date and time from first 8 hexadecimal characters of filename.
    
    Args:
        filename: must match format AABBCCDD.YYY or just the first 8 hex chars
        
    Returns:
        Dictionary with keys: year, mon, day, hour, min, sec, date0
    """
    # 如果文件名包含点，只取点之前的部分
    if '.' in filename:
        date_part = filename.split('.')[0]
    else:
        date_part = filename
    
    if len(date_part) < 8:
        raise ValueError(f"Filename too short: {filename}")
    
    # Convert first 8 characters to integer list
    date = [hexto_int(date_part[i]) for i in range(8)]
    
    # Bitwise extraction (matching C code exactly)
    year = (date[0] << 2 | date[1] >> 2) & 0x0f
    mon = (date[1] << 2 | date[2] >> 2) & 0x0f
    day = (date[2] << 3 | date[3] >> 1) & 0x1f
    hour = (date[3] << 4 | date[4]) & 0x1f
    min = (date[5] << 2 | date[6] >> 2) & 0x3f
    sec = (date[6] << 4 | date[7]) & 0x3f
    
    # Year adjustment (2009-2024 range)
    year += 2000
    if year < 2009:
        year += 16
    
    return {
        'year': year,
        'mon': mon,
        'day': day,
        'hour': hour,
        'min': min,
        'sec': sec,
        'date0': date[0]  # Store for file type detection
    }


def parse_milliseconds(filename: str, pcik: float) -> float:
    """Parse milliseconds from filename.
    
    Args:
        filename: Filename (must match format AABBCCDD.YYY)
        pcik: Internal clock frequency (TC / 256.0)
        
    Returns:
        Actual milliseconds (in seconds): nzmsec * 4096 / PCIk
        
    Raises:
        ValueError: If filename doesn't match the required format
    """
    # Strict format validation: must be "XXXXXXXX.YYY" format
    # where XXXXXXXX is exactly 8 hex characters and YYY is exactly 3 hex characters
    if '.' not in filename:
        raise ValueError(f"Filename must contain a dot separator: {filename}")
    
    parts = filename.split('.', 1)
    if len(parts) != 2:
        raise ValueError(f"Filename must have exactly one dot: {filename}")
    
    date_part = parts[0]
    msec_part = parts[1]
    
    # Validate date part: must be exactly 8 hexadecimal characters
    if len(date_part) != 8:
        raise ValueError(f"Date part must be exactly 8 hexadecimal characters, got {len(date_part)}: {date_part} in {filename}")
    
    if not all(c in '0123456789ABCDEFabcdef' for c in date_part):
        raise ValueError(f"Date part must contain only hexadecimal characters: {date_part} in {filename}")
    
    # Validate msec part: must be exactly 3 hexadecimal characters
    if len(msec_part) != 3:
        raise ValueError(f"Millisecond part must be exactly 3 hexadecimal characters, got {len(msec_part)}: {msec_part} in {filename}")
    
    if not all(c in '0123456789ABCDEFabcdef' for c in msec_part):
        raise ValueError(f"Millisecond part must contain only hexadecimal characters: {msec_part} in {filename}")  
    
    # Extract msec hex part (exactly 3 characters)
    msec_hex = msec_part  
    # Convert to decimal (a0 = 2 for 3 characters)
    nzmsec = hexto_dec(msec_hex)  
    # Convert to actual milliseconds (in seconds as decimal part)
    actual_msec = nzmsec * 4096.0 / pcik   
    return actual_msec


# ============================================================================
# 3字节有符号整数转换工具（来自 three_bytes_utils.py）
# ============================================================================

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


# ============================================================================
# 字节交换工具（来自 sac2y_v2_1_obspy.py）
# ============================================================================

def swap_bytes_2(value: int) -> int:
    """2字节字节序转换（little-endian ↔ big-endian）
    
    对应C代码：up_side_down2()
    
    Args:
        value: 2字节整数值
    
    Returns:
        字节序转换后的值
    """
    return ((value >> 8) & 0x00FF) | ((value << 8) & 0xFF00)


def swap_bytes_4(value: int) -> int:
    """4字节字节序转换（little-endian ↔ big-endian）
    
    对应C代码：up_side_down4()
    
    Args:
        value: 4字节整数值
    
    Returns:
        字节序转换后的值
    """
    return ((value << 24) & 0xFF000000) | \
           ((value << 8) & 0x00FF0000) | \
           ((value >> 8) & 0x0000FF00) | \
           ((value >> 24) & 0x000000FF)


# ============================================================================
# 浮点格式转换工具（来自 sac2y_v2_1_obspy.py）
# ============================================================================

def ieee_to_ibm_float_vectorized(ieee_data: np.ndarray) -> np.ndarray:
    """向量化IEEE→IBM浮点转换（高度优化）
    
    对应C代码的float_to_ibm()函数
    
    C代码逻辑：
    fconv = from[i];  // IEEE浮点（作为int）
    if (fconv) {
        fmant = (0x007fffff & fconv) | 0x00800000;  // 尾数（添加隐含的1）
        t = ((0x7f800000 & fconv) >> 23) - 126;     // 指数
        while (t & 0x3) { ++t; fmant >>= 1; }       // 对齐到4的倍数
        fconv = (0x80000000 & fconv) | (((t>>2) + 64) << 24) | fmant;  // 组合IBM
    }
    if(endian==0) fconv = (fconv<<24) | ((fconv>>24)&0xff) | ...  // big-endian转换
    
    Args:
        ieee_data: IEEE浮点数组（float32）
    
    Returns:
        IBM浮点数组（uint32，big-endian格式）
    """
    # 转换为uint32视图（不拷贝数据，直接位操作）
    ieee_uint32 = ieee_data.view(np.uint32)
    
    # 处理零值
    zero_mask = (ieee_uint32 == 0)
    result = np.zeros_like(ieee_uint32)
    
    # 提取符号位、指数、尾数（向量化）
    sign = ieee_uint32 & 0x80000000
    exponent_raw = (ieee_uint32 & 0x7F800000) >> 23
    t = exponent_raw.astype(np.int32) - 126
    mantissa = (ieee_uint32 & 0x007FFFFF) | 0x00800000  # 添加隐含的1
    
    # 对齐到4的倍数（对应C代码：while (t & 0x3) { ++t; fmant >>= 1; }）
    # C代码逻辑：当t不是4的倍数时，t+1，mantissa右移1位，重复直到t是4的倍数
    # 也就是说：
    # - 如果 t % 4 == 1: 需要+3（+1,+1,+1），右移3位
    # - 如果 t % 4 == 2: 需要+2（+1,+1），右移2位  
    # - 如果 t % 4 == 3: 需要+1，右移1位
    # - 如果 t % 4 == 0: 不需要移位
    
    # 计算余数（t & 0x3）
    remainder = t & 0x3
    
    # 计算需要增加的次数和移位数（向量化）
    # remainder=0: shift=0
    # remainder=1: shift=3 (需要+3次，右移3位)
    # remainder=2: shift=2 (需要+2次，右移2位)
    # remainder=3: shift=1 (需要+1次，右移1位)
    shift_count = np.where(remainder == 0, 0,
                  np.where(remainder == 1, 3,
                  np.where(remainder == 2, 2, 1))).astype(np.int32)
    
    t_adjusted = t + shift_count
    mantissa_shifted = np.where(shift_count == 0, mantissa,
                       np.where(shift_count == 1, mantissa >> 1,
                       np.where(shift_count == 2, mantissa >> 2, mantissa >> 3)))
    
    # 组合IBM浮点数
    # IBM格式：符号位(1) + 指数位(7) + 尾数位(24)
    ibm_exponent = ((t_adjusted >> 2) + 64).astype(np.uint32)
    result[~zero_mask] = (sign[~zero_mask] | 
                         (ibm_exponent[~zero_mask] << 24) | 
                         (mantissa_shifted[~zero_mask] & 0x00FFFFFF))
    
    # 转换为big-endian（对应C代码endian==0的情况）
    # C代码：fconv = (fconv<<24) | ((fconv>>24)&0xff) | ((fconv&0xff00)<<8) | ((fconv&0xff0000)>>8)
    result_be = result.copy()
    result_be[~zero_mask] = (
        ((result[~zero_mask] << 24) & 0xFF000000) |
        ((result[~zero_mask] >> 24) & 0x000000FF) |
        ((result[~zero_mask] & 0x0000FF00) << 8) |
        ((result[~zero_mask] & 0x00FF0000) >> 8)
    )
    
    return result_be


# ============================================================================
# 时间字符串解析工具（来自 sac2y_v2_1_obspy.py）
# ============================================================================

def parse_time_string(time_str: str) -> Dict[str, int]:
    """解析时间字符串（格式：年/月/日 时:分:秒）
    
    支持的格式：
    - "2017/02/27 21:18:54"
    - "2017/2/27 21:18:54"
    - "2017-02-27 21:18:54"（也支持-分隔符）
    
    Args:
        time_str: 时间字符串
    
    Returns:
        包含year, month, day, hour, minute, second的字典
    
    Raises:
        ValueError: 时间字符串格式无效
    """
    # 支持多种分隔符
    time_str = time_str.strip()
    
    # 尝试解析格式：YYYY/MM/DD HH:MM:SS 或 YYYY-MM-DD HH:MM:SS
    patterns = [
        r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})\s+(\d{1,2}):(\d{1,2}):(\d{1,2})',
        r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})\s+(\d{1,2}):(\d{1,2})',  # 没有秒，默认为0
    ]
    
    for pattern in patterns:
        match = re.match(pattern, time_str)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            day = int(match.group(3))
            hour = int(match.group(4))
            minute = int(match.group(5))
            second = int(match.group(6)) if len(match.groups()) >= 6 else 0
            
            # 验证时间有效性
            try:
                datetime(year, month, day, hour, minute, second)
            except ValueError as e:
                raise ValueError(f"Invalid time string: {time_str}. {e}")
            
            return {
                'year': year,
                'month': month,
                'day': day,
                'hour': hour,
                'minute': minute,
                'second': second
            }
    
    # 如果都不匹配，尝试使用datetime解析
    try:
        dt = datetime.strptime(time_str, "%Y/%m/%d %H:%M:%S")
    except ValueError:
        try:
            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                dt = datetime.strptime(time_str, "%Y/%m/%d %H:%M")
            except ValueError:
                dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
    
    return {
        'year': dt.year,
        'month': dt.month,
        'day': dt.day,
        'hour': dt.hour,
        'minute': dt.minute,
        'second': dt.second
    }


# ============================================================================
# 字符串工具函数（来自 sac2y_v2_1_obspy.py）
# ============================================================================

def get_subs(line: str, begin: int, end: int) -> str:
    """提取字符串的子串（去除空格）
    
    对应C代码的get_subs函数
    注意：C代码中begin和end是从1开始的，这里也保持相同约定
    
    Args:
        line: 输入字符串
        begin: 起始位置（从1开始）
        end: 结束位置（从1开始，包含）
    
    Returns:
        提取的子串（去除空格）
    """
    if len(line) < end:
        return ""
    
    substring = ""
    for i in range(begin - 1, end):
        if i < len(line) and line[i] != ' ':
            substring += line[i]
    
    return substring


# ============================================================================
# 文件读取工具（来自 sac2y_v2_1_obspy.py）
# ============================================================================

def read_location_file(loc_file: str) -> Tuple[float, float]:
    """从文件读取位置信息
    
    文件格式：lon lat（空格分隔）
    
    Args:
        loc_file: 位置文件路径
    
    Returns:
        (longitude, latitude)
    
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式无效或经纬度值无效
    """
    if not os.path.exists(loc_file):
        raise FileNotFoundError(f"Cannot open the location file: {loc_file}")
    
    with open(loc_file, 'r') as f:
        line = f.readline().strip()
        values = line.split()
        
        if len(values) < 2:
            raise ValueError(f"Invalid location file format: {loc_file}. Expected 2 values (lon lat).")
        
        lon = float(values[0])
        lat = float(values[1])
        
        # 验证经纬度范围
        if lat < -90 or lat > 90:
            raise ValueError(f"Invalid latitude value in {loc_file}: {lat:.6f}° (must be between -90 and 90).")
        if lon < -180 or lon > 360:
            raise ValueError(f"Invalid longitude value in {loc_file}: {lon:.6f}° (must be between -180 and 360).")
    
    return lon, lat


# ============================================================================
# 坐标投影工具（来自 sac2y_v2_1_obspy.py，需要pyproj）
# ============================================================================

def setup_tm_projection(lon0: float) -> Optional['Transformer']:
    """设置横墨卡托投影（TM）
    
    对应C代码的vtm()函数
    使用WGS-84椭球，与C代码保持一致
    
    需要安装pyproj库：pip install pyproj
    
    Args:
        lon0: 投影中心经度
    
    Returns:
        pyproj Transformer对象，如果pyproj不可用则返回None
    
    Raises:
        ImportError: pyproj未安装
    """
    if not PYPROJ_AVAILABLE:
        raise ImportError("pyproj is required for coordinate projection. Install it with: pip install pyproj")
    
    # 使用pyproj创建TM投影
    # PROJ字符串格式：+proj=tmerc +lon_0=lon0 +datum=WGS84 +units=m +k_0=0.9996
    proj_string = f"+proj=tmerc +lon_0={lon0} +datum=WGS84 +units=m +k_0=0.9996 +ellps=WGS84"
    
    transformer = Transformer.from_crs(
        "EPSG:4326",  # WGS84经纬度
        proj_string,
        always_xy=True
    )
    
    return transformer


def lonlat_to_utm(lon: float, lat: float, transformer: Optional['Transformer'],
                  ex: float = 0.0, ey: float = 0.0) -> Tuple[float, float]:
    """经纬度转UTM坐标
    
    对应C代码的utm()函数
    - 如果lon < 0，加360度
    - X坐标加500000米（UTM偏移）
    - 南半球Y坐标加10000000米
    
    需要pyproj库和transformer对象
    
    Args:
        lon: 经度（度）
        lat: 纬度（度）
        transformer: pyproj转换器（由setup_tm_projection创建）
        ex: X方向偏移（米）
        ey: Y方向偏移（米）
    
    Returns:
        (x, y) UTM坐标（米）
    
    Raises:
        ValueError: 如果transformer为None
    """
    if transformer is None:
        raise ValueError("transformer is None. Call setup_tm_projection() first.")
    
    # 检查输入是否有效
    if not np.isfinite(lon) or not np.isfinite(lat):
        return np.nan, np.nan
    
    # 如果经度小于0，加360度（对应C代码：if (lon < 0.0) lon += 360.0）
    lon_transformed = lon if lon >= 0.0 else lon + 360.0
    
    # 转换坐标
    try:
        x, y = transformer.transform(lon_transformed, lat)
    except Exception:
        # 如果转换失败，返回NaN
        return np.nan, np.nan
    
    # 检查转换结果是否有效
    if not np.isfinite(x) or not np.isfinite(y):
        return np.nan, np.nan
    
    # 添加UTM偏移（X+500000）
    x = x + 500000.0 + ex
    
    # 南半球处理（Y+10000000）
    if lat < 0:
        y = y + 10000000.0
    
    y = y + ey
    
    # 最终检查
    if not np.isfinite(x) or not np.isfinite(y):
        return np.nan, np.nan
    
    return x, y


def lonlat_to_utm_batch(lons: np.ndarray, lats: np.ndarray,
                       transformer: Optional['Transformer'],
                       ex: float = 0.0, ey: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """批量转换坐标（向量化）
    
    需要pyproj库和transformer对象
    
    Args:
        lons: 经度数组
        lats: 纬度数组
        transformer: pyproj转换器（由setup_tm_projection创建）
        ex: X方向偏移（米）
        ey: Y方向偏移（米）
    
    Returns:
        (xs, ys) UTM坐标数组
    
    Raises:
        ValueError: 如果transformer为None
    """
    if transformer is None:
        raise ValueError("transformer is None. Call setup_tm_projection() first.")
    
    # 经度转换（向量化）
    lons_transformed = np.where(lons < 0.0, lons + 360.0, lons)
    
    # 批量转换
    xs, ys = transformer.transform(lons_transformed, lats)
    
    # 添加UTM偏移和用户偏移（向量化）
    xs = xs + 500000.0 + ex
    
    # 南半球处理（向量化）
    south_mask = lats < 0
    ys = np.where(south_mask, ys + 10000000.0, ys)
    ys = ys + ey
    
    return xs, ys


# ============================================================================
# 时间处理工具（来自 sac2y_v2_1_obspy.py）
# ============================================================================

def calculate_drift_rate(drift_params: Dict) -> Tuple[float, int, int]:
    """计算OBS漂移率
    
    对应C代码的漂移率计算：
    timegap = (edjday-stjday+1)*24*3600 + (edhour-sthour)*3600 + 
              (edmin-stmin)*60 + (edsec-stsec) + (edmsec-stmsec)/1000
    drift1 = drift / timegap (ms/s)
    
    注意：C代码中edmsec和stmsec被显式设置为0，但公式中包含这一项
    
    Args:
        drift_params: 漂移参数字典，包含：
            - styear, stmonth, stday, sthour, stmin, stsec: 开始时间
            - edyear, edmonth, edday, edhour, edmin, edsec: 结束时间
            - drift: 漂移量（毫秒）
    
    Returns:
        (drift_rate, stjday, edjday)
        drift_rate: 漂移率（ms/s）
        stjday: 开始记录儒略日（相对于年初）
        edjday: 结束记录儒略日（相对于年初）
    """
    if not OBSPY_AVAILABLE:
        raise ImportError("ObsPy is required for calculate_drift_rate function")
    
    # 获取一年中的第几天（使用ObsPy的julday属性）
    start_time = UTCDateTime(
        year=drift_params['styear'],
        month=drift_params['stmonth'],
        day=drift_params['stday']
    )
    end_time = UTCDateTime(
        year=drift_params['edyear'],
        month=drift_params['edmonth'],
        day=drift_params['edday']
    )
    
    stjday = start_time.julday  # 一年中的第几天（相对于年初）
    edjday = end_time.julday    # 一年中的第几天（相对于年初）
    
    # C代码中毫秒被设置为0：edmsec=0, stmsec=0
    edmsec = 0
    stmsec = 0
    
    # 计算时间间隔（秒）- 完全对应C代码第179行
    # 使用float64确保精度
    timegap = float((edjday - stjday + 1) * 24 * 3600 +
                    (drift_params['edhour'] - drift_params['sthour']) * 3600 +
                    (drift_params['edmin'] - drift_params['stmin']) * 60 +
                    (drift_params['edsec'] - drift_params['stsec']) +
                    (edmsec - stmsec) / 1000.0)
    
    # 检查timegap是否有效（避免除零）
    if timegap <= 0:
        raise ValueError(f"Invalid timegap: {timegap} seconds. "
                        f"Start time: {drift_params['styear']}-{drift_params['stmonth']}-{drift_params['stday']} "
                        f"{drift_params['sthour']}:{drift_params['stmin']}:{drift_params['stsec']}, "
                        f"End time: {drift_params['edyear']}-{drift_params['edmonth']}-{drift_params['edday']} "
                        f"{drift_params['edhour']}:{drift_params['edmin']}:{drift_params['edsec']}")
    
    # 计算漂移率（ms/s）- 使用float64确保精度
    # drift是毫秒，timegap是秒，结果单位是ms/s
    drift_rate = float(drift_params['drift']) / timegap
    
    return drift_rate, stjday, edjday


def calculate_shot_time(shot_info: Dict, sac_ref_time, drift_params: Dict, 
                       drift_rate: float, rv: float, offset: float, et: float,
                       sthour: int, stmin: int, stsec: int, stmsec: int,
                       stjday: int) -> float:
    """计算炮点时间（含各种校正），返回相对于参考时间的秒数
    
    对应C代码的时间校正逻辑：
    t3 = tr.hour*3600+tr.minute*60+tr.sec+tr.timbas/1000+t6+et+drift3/1000
    
    注意：C代码返回的是相对于当天0时的秒数（t3），不是绝对时间
    
    Args:
        shot_info: 炮点信息字典，包含 hour, minute, sec, msec, day
        sac_ref_time: SAC文件参考时间（UTCDateTime对象或兼容对象，需要julday属性）
        drift_params: 漂移参数字典
        drift_rate: 漂移率（ms/s）
        rv: 声速（m/s）
        offset: 偏移距（米）
        et: 时间偏移（秒）
        sthour, stmin, stsec, stmsec: 开始记录时间
        stjday: 开始记录儒略日
    
    Returns:
        校正后的炮点时间（秒，从当天0时开始）
    """
    # 基础时间（从当天0时开始的秒数）
    t3 = (shot_info['hour'] * 3600 + 
          shot_info['minute'] * 60 + 
          shot_info['sec'] + 
          shot_info['msec'] / 1000.0)
    
    # 偏移时间（根据距离和声速）
    # 对应C代码逻辑：
    # if (rv>0.001)     t6=(double)(tr.offset)/(rv*1000.);
    # if (rv>100.0 || rv<=0.001) t6=0.;
    if rv > 0.001 and rv <= 100.0:
        # 检查是否会导致无穷大
        if rv * 1000.0 == 0:
            t6 = 0.0
        else:
            t6 = abs(offset) / (rv * 1000.0)
            # 检查t6是否为有效值
            if not np.isfinite(t6):
                t6 = 0.0
    else:
        t6 = 0.0
    
    # 漂移校正
    drift2 = ((shot_info['day'] - stjday) * 24 * 3600 +
              (shot_info['hour'] - sthour) * 3600 +
              (shot_info['minute'] - stmin) * 60 +
              (shot_info['sec'] - stsec) +
              (shot_info['msec'] - stmsec) / 1000.0)
    drift3 = drift2 * drift_rate  # drift_rate已经是ms/s，drift3单位是ms
    
    # 检查drift3是否为有效值
    if not np.isfinite(drift3):
        drift3 = 0.0
    
    # 总时间校正（对应C代码：t3 = ... + t6 + et + drift3/1000）
    t3 = t3 + t6 + et + drift3 / 1000.0
    
    # 检查t3是否为有效值
    if not np.isfinite(t3):
        # 返回NaN而不是抛出异常，让调用者处理
        return np.nan
    
    # 跨天处理（如果炮点的儒略日大于SAC文件的儒略日）
    if shot_info['day'] > sac_ref_time.julday:
        day_diff = shot_info['day'] - sac_ref_time.julday
        t3 = t3 + 24 * 3600 * day_diff
        # 再次检查
        if not np.isfinite(t3):
            # 返回NaN而不是抛出异常
            return np.nan
    
    return t3


# ============================================================================
# 数据提取工具（来自 sac2y_v2_1_obspy.py）
# ============================================================================

def extract_traces_batch(sac_data: np.ndarray,
                        start_samples: np.ndarray,
                        n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """批量提取多个道的数据（高度优化）
    
    Args:
        sac_data: SAC数据数组（1D）
        start_samples: 起始采样点数组
        n_samples: 每个道的采样点数
    
    Returns:
        (traces, valid_mask)
        traces: shape (n_traces, n_samples) 的数据数组
        valid_mask: 有效道的布尔数组
    """
    n_traces = len(start_samples)
    
    # 预分配输出数组
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    
    # 过滤有效位置（向量化）
    valid_mask = (start_samples >= 0) & (start_samples + n_samples <= len(sac_data))
    
    # 批量提取有效位置
    valid_indices = np.where(valid_mask)[0]
    for idx in valid_indices:
        start = start_samples[idx]
        traces[idx] = sac_data[start:start+n_samples]
    
    # 标记无效位置
    traces[~valid_mask] = np.nan
    
    return traces, valid_mask

