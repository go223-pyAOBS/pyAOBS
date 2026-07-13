#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python implementation of raw2sac_v1.1.c using ObsPy

Convert OBS raw data to SAC format using ObsPy library.

Original C program:
    program name: raw2sac.c
    written by: XL Qiu, MH Zhao and SH Xia
    first attempt: 2008/05/30
    modified: 2016/08/31 by HY Zhang, TC is included in parameters,
                              the extend name of output SAC file is changed
                              to short period type.
    Edited by: Haibo Huang, 2025/11/21. Using ObsPy library for SAC file format handling.

Usage:
    python raw2sac_v1_1_obspy.py fileName sps TC

    fileName: Input raw data filename (format: AABBCCDD.YYY)
    sps: Sampling rate (Hz)
    TC: Clock parameter from A*.LOG file

Requirements:
    pip install obspy numpy
    pip install numba  # 可选，用于JIT编译加速（推荐）

Note:
    This implementation uses ObsPy library for SAC file format handling,
    which automatically manages the 632-byte SAC header and ensures
    compatibility with SAC software.
    
    Performance optimization:
    - Uses NumPy vectorization for fast data processing
    - Optional Numba JIT compilation for additional speedup (~2-3x faster)
"""

import os
import sys
import numpy as np
from obspy import Trace, UTCDateTime
from obspy.core import AttribDict
from typing import List, Tuple

# 统一从 format_utils.py 导入格式转换工具函数
try:
    from format_utils import (
        parse_datetime_from_filename,
        parse_milliseconds,
        hexto_dec,
        three_bytes_to_int_vectorized
    )
except ImportError:
    print("Error: Cannot import format_utils.py functions.")
    print("Please ensure format_utils.py is in the same directory.")
    sys.exit(1)


def read_raw_data(filename: str, n_channels: int) -> Tuple[List[np.ndarray], int]:
    """读取原始二进制数据文件（优化版本，使用NumPy向量化处理）
    
    性能优化：
    1. 一次性读取整个文件到内存
    2. 使用NumPy向量化操作批量转换
    3. 直接创建NumPy数组，避免Python列表的append开销
    
    Args:
        filename: 输入文件路径
        n_channels: 通道数（3或4）
    
    Returns:
        (data_channels, npts)
        data_channels: 列表，包含n_channels个NumPy数组（float32）
        npts: 数据点数
    """
    file_size = os.path.getsize(filename)
    bytes_per_point = 12 if n_channels == 4 else 9
    npts = file_size // bytes_per_point
    
    # 一次性读取整个文件到内存
    with open(filename, 'rb') as f:
        all_bytes = np.frombuffer(f.read(file_size), dtype=np.uint8)
    
    # 重塑为 (npts, bytes_per_point) 的形状，每行代表一个数据点
    # 文件中的字节顺序：对于每个通道，是 h, m, l (高字节、中字节、低字节)
    all_bytes = all_bytes[:npts * bytes_per_point].reshape(npts, bytes_per_point)
    
    # 为每个通道提取3字节数据
    data_channels = []
    for ch in range(n_channels):
        # 提取该通道的3字节数据：每行是 (h, m, l)
        # C代码中结构体：char hbyt, mbyt, lbyt
        channel_bytes = all_bytes[:, ch*3:(ch*3)+3]
        
        # 使用向量化函数批量转换
        # OBS Raw格式：文件中字节顺序为 (h, m, l)，使用'hml'模式
        int_values = three_bytes_to_int_vectorized(channel_bytes, byte_order='hml')
        
        # 转换为float32数组（对应C的 (float)）
        float_values = int_values.astype(np.float32)
        
        data_channels.append(float_values)
    
    return data_channels, npts


def create_output_filename(input_filename: str, extension: str) -> str:
    """创建输出文件名
    
    C代码逻辑：
    - fnameout[0-8] = 输入文件名的后12个字符中的前9个
    - fnameout[9-11] = 扩展名（如 "shx", "shy", "shz", "hyd"）
    - fnameout[12] = '\0'
    
    例如：输入文件名 "51A6D159.5DD" (12字符)
    - fnlength = 12
    - fnameout[0-8] = fnamein[0-8] = "51A6D159." (9字符，包含点号)
    - fnameout[9-11] = "shx" (3字符)
    - 输出： "51A6D159.shx" (12字符)
    
    Args:
        input_filename: 输入文件名（完整路径或文件名）
        extension: 扩展名（如 "shx", "shy", "shz", "hyd"）
    
    Returns:
        输出文件名（12个字符）
    """
    # 提取文件名部分（去掉路径）
    base_filename = os.path.basename(input_filename)
    fnlength = len(base_filename)
    
    # C代码：for (i=0;i<9;i++) fnameout[i] = fnamein[fnlength-12+i];
    if fnlength >= 12:
        # 从后12个字符中提取前9个
        base = base_filename[fnlength-12:fnlength-3]
    else:
        # 如果文件名不够12字符，取前9个字符（去掉扩展名后的部分）
        name_without_ext = os.path.splitext(base_filename)[0]
        base = name_without_ext[:9].ljust(9, '0')
    
    # 确保base长度为9
    base = base[:9]
    if len(base) < 9:
        base = base.ljust(9, '0')
    
    # 组合文件名（12个字符：9个基础字符 + 3个扩展名）
    output_filename = base + extension
    
    return output_filename


def convert_raw_to_sac(filename: str, sps: float, tc: float, verbose: bool = False):
    """使用ObsPy转换原始数据到SAC格式
    
    Args:
        filename: 输入文件路径
        sps: 采样率 (Hz)
        tc: 时钟参数（从A*.LOG文件读取）
        verbose: 是否输出详细信息
    """
    # 1. 解析文件名
    base_filename = os.path.basename(filename)
    if verbose:
        print("****************")
        print(f"fileName: {base_filename}")
        print(f"SPS: {sps}, TC: {tc}")
    
    try:
        dt_dict = parse_datetime_from_filename(base_filename)
    except ValueError as e:
        print(f"Error: Invalid filename format: {e}")
        print(f"Expected format: AABBCCDD.YYY (8 hex chars + dot + 3 hex chars)")
        sys.exit(1)
    
    # 计算PCIk
    pcik = tc / 256.0
    if verbose:
        print(f"PCIk: {pcik}")
    
    # 2. 解析毫秒
    try:
        actual_msec = parse_milliseconds(base_filename, pcik)
        # 转换为整数毫秒（用于SAC头文件）
        nzmsec_hex = base_filename.split('.')[1][:3]  # 后3个十六进制字符
        nzmsec_decimal = hexto_dec(nzmsec_hex)
        nzmsec_sac = int(nzmsec_decimal * 4096.0 * 1000.0 / pcik + 0.5)
    except ValueError as e:
        print(f"Error: Cannot parse milliseconds from filename: {e}")
        sys.exit(1)
    
    # 3. 创建UTCDateTime对象
    try:
        starttime = UTCDateTime(
            year=dt_dict['year'],
            month=dt_dict['mon'],
            day=dt_dict['day'],
            hour=dt_dict['hour'],
            minute=dt_dict['min'],
            second=dt_dict['sec'],
            microsecond=int(actual_msec * 1000000)
        )
    except Exception as e:
        print(f"Error: Invalid date/time: {e}")
        sys.exit(1)
    
    if verbose:
        print(f"{dt_dict['year']:4d}-{dt_dict['mon']:02d}-{dt_dict['day']:02d}")
        print(f"Starting time: {starttime}")
    
    # 4. 判断文件类型（4通道 vs 3通道）
    date0 = dt_dict['date0']
    n_channels = 4 if (date0 >> 3) == 0 else 3
    
    if verbose:
        print(f"File type: {n_channels}-channel ({'broadband + hydrophone' if n_channels == 4 else 'short period'})")
    
    # 5. 读取原始数据
    try:
        data_channels, npts = read_raw_data(filename, n_channels)
    except Exception as e:
        print(f"Error: Cannot read raw data file: {e}")
        sys.exit(1)
    
    if verbose:
        print(f"npts: {npts}")
    
    # 6. 通道配置
    channel_configs = {
        4: [
            {'name': 'SHX', 'cmpinc': 90.0, 'cmpaz': None, 'ext': 'shx'},
            {'name': 'SHY', 'cmpinc': 90.0, 'cmpaz': None, 'ext': 'shy'},
            {'name': 'SHZ', 'cmpinc': 0.0, 'cmpaz': 0.0, 'ext': 'shz'},
            {'name': 'HYD', 'cmpinc': 12345.0, 'cmpaz': 12345.0, 'ext': 'hyd'}
        ],
        3: [
            {'name': 'SHX', 'cmpinc': 90.0, 'cmpaz': None, 'ext': 'shx'},
            {'name': 'SHY', 'cmpinc': 90.0, 'cmpaz': None, 'ext': 'shy'},
            {'name': 'SHZ', 'cmpinc': 0.0, 'cmpaz': 0.0, 'ext': 'shz'}
        ]
    }
    
    # 7. 获取一年中的第几天（使用ObsPy的julday属性）
    nzjday = starttime.julday
    
    # 8. 创建Trace并写入SAC文件
    for ch, config in enumerate(channel_configs[n_channels]):
        try:
            # 创建Trace对象（data_channels[ch]已经是float32数组，不需要再转换）
            trace = Trace(data=data_channels[ch])
            
            # 设置基本统计信息
            trace.stats.sampling_rate = sps
            trace.stats.starttime = starttime
            trace.stats.npts = npts
            trace.stats.delta = 1.0 / sps
            trace.stats.b = 0.0  # 设置Trace的b字段
            trace.stats.calib = 1.0
            
            # 设置SAC特定字段
            trace.stats.sac = AttribDict()
            trace.stats.sac.kstnm = 'OBS'
            trace.stats.sac.kcmpnm = config['name']
            trace.stats.sac.cmpinc = config['cmpinc']
            if config['cmpaz'] is not None:
                trace.stats.sac.cmpaz = config['cmpaz']
            
            # SAC时间字段（必须与starttime完全一致，以确保b字段为0）
            trace.stats.sac.nzyear = dt_dict['year']
            trace.stats.sac.nzjday = nzjday
            trace.stats.sac.nzhour = dt_dict['hour']
            trace.stats.sac.nzmin = dt_dict['min']
            trace.stats.sac.nzsec = dt_dict['sec']
            trace.stats.sac.nzmsec = nzmsec_sac
            
            # 关键修复：根据SAC时间字段重新构建starttime，确保完全一致
            # 这样可以避免ObsPy在写入时自动计算b字段
            # SAC的nzmsec是整数毫秒，需要转换为微秒
            sac_starttime = UTCDateTime(
                year=trace.stats.sac.nzyear,
                julday=trace.stats.sac.nzjday,
                hour=trace.stats.sac.nzhour,
                minute=trace.stats.sac.nzmin,
                second=trace.stats.sac.nzsec,
                microsecond=trace.stats.sac.nzmsec * 1000  # nzmsec是毫秒，转换为微秒
            )
            trace.stats.starttime = sac_starttime
            
            # 显式设置SAC的b字段为0.0
            # 必须在设置完所有字段后再设置，确保不会被覆盖
            trace.stats.sac.b = 0.0
            trace.stats.b = 0.0  # 同时设置Trace的b字段
            
            # SAC其他字段
            trace.stats.sac.leven = 1  # 等间隔数据
            trace.stats.sac.lcalda = 1  # 计算距离和方位角
            trace.stats.sac.iftype = 1  # 时间序列（ITIME）
            trace.stats.sac.iztype = 9  # 参考时间（IB）
            trace.stats.sac.idep = 6  # 未知（IDEP）
            
            # 生成输出文件名（与C版本一致：12个字符）
            output_filename = create_output_filename(filename, config['ext'])
            
            # 写入SAC文件
            trace.write(output_filename, format='SAC')
            
            if verbose:
                print(f"Created: {output_filename} (npts={npts}, channel={config['name']})")
            else:
                print(f"Created: {output_filename}")
                
        except Exception as e:
            print(f"Error: Cannot create SAC file for channel {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """命令行接口"""
    if len(sys.argv) != 4:
        print("Error: Invalid number of arguments!")
        print("Usage: python raw2sac_v1_1_obspy.py fileName sps TC")
        print("  fileName: Input raw data filename (format: AABBCCDD.YYY)")
        print("  sps: Sampling rate (Hz)")
        print("  TC: Clock parameter from A*.LOG file")
        print("\nExample:")
        print("  python raw2sac_v1_1_obspy.py 51A6D159.5DD 100.0 3145727336.0")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # 验证文件是否存在
    if not os.path.exists(filename):
        print(f"Error: File not found: {filename}")
        sys.exit(1)
    
    try:
        sps = float(sys.argv[2])
        tc = float(sys.argv[3])
    except ValueError as e:
        print(f"Error: Invalid numeric argument: {e}")
        print("  sps and TC must be numeric values")
        sys.exit(1)
    
    if sps <= 0:
        print("Error: Sampling rate (sps) must be positive")
        sys.exit(1)
    
    if tc <= 0:
        print("Error: Clock parameter (TC) must be positive")
        sys.exit(1)
    
    try:
        convert_raw_to_sac(filename, sps, tc, verbose=True)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

