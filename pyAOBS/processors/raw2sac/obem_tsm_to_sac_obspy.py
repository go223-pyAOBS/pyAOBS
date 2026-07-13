#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python implementation of OBEM TSM to SAC conversion using ObsPy

Convert OBEM TSM format data to SAC format.

Original MATLAB program:
    OBEM_RAW2SAC_TSM_main_WP12.m
    written by: JF & ZJZ
    
Edited by: Haibo Huang, 2025/01/XX. Python implementation using ObsPy.

Usage:
    python obem_tsm_to_sac_obspy.py config_file

Requirements:
    pip install obspy numpy scipy

Features:
    - Reads OBEM TSM files (512MB blocks, 10 channels)
    - Extracts 5 channels: Ex, Ey, Hx, Hy, FHz
    - Interpolates from 300Hz to 500Hz
    - Writes SAC format files using ObsPy
"""

import os
import sys
import glob
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy.interpolate import interp1d

from obspy import Trace, UTCDateTime
from obspy.core import AttribDict

# 注意：julian_day 函数已被 ObsPy 的 UTCDateTime.julday 属性替代
# 代码中直接使用 starttime.julday 获取一年中的第几天

# ============================================================================
# 3字节有符号整数解码（使用通用工具模块）
# ============================================================================

# 统一从 format_utils.py 导入格式转换工具函数
try:
    import sys
    # 添加父目录到路径以便导入通用模块
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from format_utils import three_bytes_to_int_vectorized
except ImportError:
    print("Error: Cannot import format_utils.py functions.")
    print("Please ensure format_utils.py is in processors/raw2sac/ directory.")
    sys.exit(1)


# ============================================================================
# TSM文件读取模块
# ============================================================================

def parse_tsm_filename(filename: str) -> datetime:
    """从TSM文件名解析时间
    
    MATLAB代码逻辑：
    timestr = TSMname(3:end-4);  % 去掉前2个字符和后4个字符
    ftime = ['20',timestr(1:2),'/',timestr(3:4),'/',timestr(5:6),' ',...]
    
    示例：WP220101120000.tsm -> timestr="220101120000" -> 2022/01/01 12:00:00
    
    Args:
        filename: TSM文件名（如"WP220101120000.tsm"）
    
    Returns:
        datetime对象
    """
    basename = os.path.basename(filename)
    # 去掉扩展名（.tsm或.TSM）
    name_no_ext = os.path.splitext(basename)[0]
    # 去掉前2个字符，从第3个字符开始到倒数第4个字符
    if len(name_no_ext) >= 18:
        timestr = name_no_ext[2:-4]  # 去掉前2个字符和后4个字符
    elif len(name_no_ext) >= 14:
        timestr = name_no_ext[2:]  # 只去掉前2个字符
    else:
        raise ValueError(f"Invalid TSM filename format: {filename}")
    print(f"timestr: {timestr}")
    # 解析时间字符串：YYMMDDHHMMSS (12位数字)
    if len(timestr) >= 12:
        year = 2000 + int(timestr[0:2])
        month = int(timestr[2:4])
        day = int(timestr[4:6])
        hour = int(timestr[6:8])
        minute = int(timestr[8:10])
        second = int(timestr[10:12])
    else:
        raise ValueError(f"Invalid time string in filename: {timestr}")
    
    return datetime(year, month, day, hour, minute, second)


def read_tsm_block_header(f, reference_time: datetime = None) -> Tuple[datetime, int]:
    """读取TSM块头部（16字节）
    
    Block头部结构：
    - 前8字节：时间戳（tag中无有效信息）
    - tag[0]: year
    - tag[1]: month
    - tag[2]: day
    - tag[3]: hour
    - tag[4]: minute
    - tag[5]: second
    - 后8字节：时钟计数（uint64, little-endian）
    
    注意：根据MATLAB源代码注释，tag中实际上没有有效的时间信息。
    因此直接使用参考时间（从文件名提取），而不是尝试解析tag中的时间。
    
    Args:
        f: 文件对象（二进制读取模式）
        reference_time: 参考时间（从文件名提取，必需）
    
    Returns:
        (datetime对象, 时钟计数)
    """
    # 读取前8字节（tag中无有效信息，仅用于跳过字节）
    tag = f.read(8)
    if len(tag) < 8:
        raise ValueError("Incomplete block header")
    
    # 根据MATLAB注释，tag中无有效信息，直接使用参考时间
    if reference_time is None:
        raise ValueError("reference_time is required (tag contains no valid time information)")
    
    block_time = reference_time
    
    # 读取后8字节时钟计数（uint64, little-endian）
    clock_bytes = f.read(8)
    if len(clock_bytes) < 8:
        raise ValueError("Incomplete clock bytes")
    
    little_clock = int.from_bytes(clock_bytes, byteorder='little', signed=False)
    
    return block_time, little_clock


def extract_channels_from_block_data(data: np.ndarray, 
                                     binaryformat: int = 1) -> np.ndarray:
    """从32字节数据块中提取5个通道
    
    对应MATLAB代码：
    field(ind,1) = transpose(signsymbol( data(1,:) + data(2,:)*2^8 + data(3,:)*2^8*2^8 ));
    
    使用统一的three_bytes_to_int_vectorized函数进行转换，字节顺序为'lmh'（low-mid-high）
    对应MATLAB的little-endian格式：data(1,:) + data(2,:)*2^8 + data(3,:)*2^8*2^8
    
    Args:
        data: shape为(32, nScan)的uint8数组，MATLAB fread按列填充，所以data是32行nScan列
        binaryformat: 1=TSM_NEW, 2=TSM_OLD
    
    Returns:
        field: shape为(nScan, 5)的数组，包含Ex, Ey, Hx, Hy, FHz
    """
    nScan = data.shape[1]
    field = np.zeros((nScan, 5), dtype=np.int32)
    
    if binaryformat == 1:  # TSM_NEW格式
        # MATLAB: field(ind,1) = transpose(signsymbol( data(1,:) + data(2,:)*2^8 + data(3,:)*2^8*2^8 ));
        # 注意：MATLAB索引从1开始，Python从0开始
        # data(1,:) 对应 data[0,:] (第一个字节的所有扫描点，low byte)
        # data(2,:) 对应 data[1,:] (第二个字节的所有扫描点，mid byte)
        # data(3,:) 对应 data[2,:] (第三个字节的所有扫描点，high byte)
        # OBEM TSM格式：文件中字节顺序为 (l, m, h)，使用'lmh'模式
        channel_bytes = np.column_stack([data[0, :], data[1, :], data[2, :]])  # (nScan, 3)
        field[:, 0] = three_bytes_to_int_vectorized(channel_bytes, byte_order='lmh')  # Ex: bytes 1-3
        channel_bytes = np.column_stack([data[3, :], data[4, :], data[5, :]])
        field[:, 1] = three_bytes_to_int_vectorized(channel_bytes, byte_order='lmh')  # Ey: bytes 4-6
        channel_bytes = np.column_stack([data[12, :], data[13, :], data[14, :]])
        field[:, 2] = three_bytes_to_int_vectorized(channel_bytes, byte_order='lmh')  # Hx: bytes 13-15 (ch5)
        channel_bytes = np.column_stack([data[15, :], data[16, :], data[17, :]])
        field[:, 3] = three_bytes_to_int_vectorized(channel_bytes, byte_order='lmh')  # Hy: bytes 16-18 (ch6)
        channel_bytes = np.column_stack([data[24, :], data[25, :], data[26, :]])
        field[:, 4] = three_bytes_to_int_vectorized(channel_bytes, byte_order='lmh')  # FHz: bytes 25-27 (ch9)
    
    elif binaryformat == 2:  # TSM_OLD格式
        channel_bytes = np.column_stack([data[0, :], data[1, :], data[2, :]])
        field[:, 0] = three_bytes_to_int_vectorized(channel_bytes, byte_order='lmh')  # Ex: bytes 1-3
        channel_bytes = np.column_stack([data[4, :], data[5, :], data[6, :]])
        field[:, 1] = three_bytes_to_int_vectorized(channel_bytes, byte_order='lmh')  # Ey: bytes 5-7
        channel_bytes = np.column_stack([data[16, :], data[17, :], data[18, :]])
        field[:, 2] = three_bytes_to_int_vectorized(channel_bytes, byte_order='lmh')  # Hx: bytes 17-19
        channel_bytes = np.column_stack([data[20, :], data[21, :], data[22, :]])
        field[:, 3] = three_bytes_to_int_vectorized(channel_bytes, byte_order='lmh')  # Hy: bytes 21-23
        channel_bytes = np.column_stack([data[24, :], data[25, :], data[26, :]])
        field[:, 4] = three_bytes_to_int_vectorized(channel_bytes, byte_order='lmh')  # FHz: bytes 25-27
    
    else:
        raise ValueError(f"Unknown binaryformat: {binaryformat}")
    
    return field


def read_tsm_file(filename: str, binaryformat: int = 1, 
                  nBlocks: int = 512, nScan: int = 32768) -> Tuple[np.ndarray, List[datetime]]:
    """读取OBEM TSM文件
    
    TSM文件结构：
    - 512个块（标准）
    - 每个块：16字节头 + 1MB数据（32768个扫描点）
    - 每个扫描点：32字节（10通道 × 3字节 + 2字节填充）
    
    Args:
        filename: TSM文件路径
        binaryformat: 1=TSM_NEW, 2=TSM_OLD
        nBlocks: 标准块数（512）
        nScan: 每块的扫描点数（32768）
    
    Returns:
        (field, block_times)
        field: shape为(nBlocks*nScan, 5)的数组，包含5个通道的数据
        block_times: 每个块的时间戳列表
    """
    file_size = os.path.getsize(filename)
    block_size = 16 + 1024 * 1024  # 16字节头 + 1MB数据
    expected_size = nBlocks * block_size
    
    # 检查文件大小
    if file_size < block_size:
        raise ValueError(f"TSM file too small: {file_size} bytes")
    
    # 计算实际块数
    actual_nBlocks = file_size // block_size
    if actual_nBlocks < nBlocks:
        print(f"Warning: File has only {actual_nBlocks} blocks (expected {nBlocks})")
        nBlocks = actual_nBlocks
    
    # 初始化输出数组
    field = np.zeros((nBlocks * nScan, 5), dtype=np.int32)
    block_times = []
    
    # 从文件名提取参考时间（必需，因为tag中无有效时间信息）
    try:
        reference_time = parse_tsm_filename(filename)
    except Exception as e:
        raise ValueError(f"Cannot parse time from filename: {e}. Reference time is required because tag contains no valid time information.")
    
    with open(filename, 'rb') as f:
        for iblock in range(nBlocks):
            try:
                # 读取块头部（传入参考时间作为备用）
                block_time, little_clock = read_tsm_block_header(f, reference_time=reference_time)
                block_times.append(block_time)
                
                # 读取数据区（32字节 × nScan）
                data_bytes = f.read(32 * nScan)
                if len(data_bytes) < 32 * nScan:
                    # 最后一个块可能不完整
                    print(f"Warning: Block {iblock+1} incomplete, truncating...")
                    nScan_actual = len(data_bytes) // 32
                    data_bytes = data_bytes[:32 * nScan_actual]
                else:
                    nScan_actual = nScan
                
                # 重塑为 (32, nScan) 数组
                # MATLAB的fread(fid,[32,nScan],'uchar')是按列填充的（Fortran顺序）
                # 第一个扫描点的32个字节填充到第一列 data(:, 1)
                # 第二个扫描点的32个字节填充到第二列 data(:, 2)
                # Python需要使用Fortran顺序（列优先）来匹配MATLAB的行为
                data = np.frombuffer(data_bytes, dtype=np.uint8).reshape(32, nScan_actual, order='F')
                
                # 提取5个通道
                field_block = extract_channels_from_block_data(data, binaryformat)
                
                # 存储到全局数组
                start_idx = iblock * nScan
                end_idx = start_idx + nScan_actual
                if end_idx <= len(field):
                    field[start_idx:end_idx, :] = field_block[:nScan_actual, :]
                else:
                    # 如果超出范围，截断
                    field[start_idx:, :] = field_block[:len(field)-start_idx, :]
                    break
                    
            except Exception as e:
                print(f"Error reading block {iblock+1}: {e}")
                import traceback
                traceback.print_exc()
                # 尝试跳过当前块的数据部分，继续读取下一个块
                try:
                    # 跳过剩余的数据部分（32字节 × nScan）
                    f.seek(32 * nScan, 1)  # 从当前位置向前移动
                except:
                    # 如果无法跳过，则停止读取
                    print(f"Cannot skip block {iblock+1} data, stopping...")
                    break
                continue
    
    # 移除末尾的零填充
    non_zero_mask = np.any(field != 0, axis=1)
    if np.any(non_zero_mask):
        last_valid_idx = np.where(non_zero_mask)[0][-1] + 1
        field = field[:last_valid_idx, :]
    
    return field, block_times


# ============================================================================
# 配置读取模块
# ============================================================================

def read_config_file(config_file: str) -> Dict:
    """读取配置文件
    
    配置文件格式（类似MATLAB代码中的变量设置）：
    [Paths]
    input_path = ...
    output_path = ...
    location_file = ...
    
    [Station]
    station_name = ...
    latitude = ...
    longitude = ...
    elevation = ...
    
    [Parameters]
    original_srate = 300
    target_srate = 500
    data_length = 21600  # 6小时
    et = 1.885  # 时间校准
    binaryformat = 1  # 1=TSM_NEW, 2=TSM_OLD
    
    Args:
        config_file: 配置文件路径
    
    Returns:
        配置字典
    """
    config = {}
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    current_section = None
    
    with open(config_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 跳过注释和空行
            if not line or line.startswith('#'):
                continue
            
            # 解析节名
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1].lower()
                config[current_section] = {}
                continue
            
            # 解析键值对
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # 转换数据类型
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.replace('.', '', 1).replace('-', '', 1).isdigit():
                    value = float(value) if '.' in value else int(value)
                
                if current_section:
                    config[current_section][key.lower()] = value
                else:
                    config[key.lower()] = value
    
    return config


# ============================================================================
# SAC文件写入模块（使用ObsPy）
# ============================================================================

def write_sac_file(filename: str, data: np.ndarray, 
                  dt: float, starttime: UTCDateTime,
                  station_lat: float, station_lon: float, station_elev: float,
                  station_name: str, channel_name: str, network: str,
                  verbose: bool = False) -> None:
    """写入SAC文件（使用ObsPy）
    
    对应MATLAB的newSacHeader_zjz和wtSac_zjz函数
    
    Args:
        filename: 输出SAC文件路径
        data: 数据数组（1D）
        dt: 采样间隔（秒）
        starttime: 起始时间
        station_lat, station_lon, station_elev: 台站位置
        station_name, channel_name, network: 台站信息
        verbose: 是否输出详细信息
    """
    if len(data) == 0:
        if verbose:
            print(f"Warning: Empty data, skipping {filename}")
        return
    
    # 创建Trace对象
    trace = Trace(data=data.astype(np.float32))
    
    # 设置基本统计信息
    trace.stats.sampling_rate = 1.0 / dt
    trace.stats.starttime = starttime
    trace.stats.npts = len(data)
    trace.stats.delta = dt
    trace.stats.b = 0.0
    trace.stats.calib = 1.0
    
    # 设置SAC特定字段
    trace.stats.sac = AttribDict()
    trace.stats.sac.kstnm = station_name[:8].ljust(8)  # 最多8字符
    trace.stats.sac.kcmpnm = channel_name[:8].ljust(8)  # 最多8字符
    trace.stats.sac.knetwk = network[:8].ljust(8)  # 最多8字符
    
    # SAC时间字段
    trace.stats.sac.nzyear = starttime.year
    trace.stats.sac.nzjday = starttime.julday
    trace.stats.sac.nzhour = starttime.hour
    trace.stats.sac.nzmin = starttime.minute
    trace.stats.sac.nzsec = starttime.second
    trace.stats.sac.nzmsec = int(starttime.microsecond / 1000)
    
    # SAC位置字段
    trace.stats.sac.stla = station_lat
    trace.stats.sac.stlo = station_lon
    trace.stats.sac.stel = station_elev
    
    # SAC其他字段（对应MATLAB代码）
    trace.stats.sac.delta = dt
    trace.stats.sac.b = 0.0
    trace.stats.sac.o = 0.0
    trace.stats.sac.e = (len(data) - 1) * dt
    trace.stats.sac.cmpinc = 0.0
    trace.stats.sac.cmpaz = 0.0
    trace.stats.sac.leven = 1  # 等间隔
    trace.stats.sac.iftype = 1  # 时间序列
    trace.stats.sac.iztype = 11  # 未知（对应MATLAB代码）
    
    # 写入SAC文件
    trace.write(filename, format='SAC')
    
    if verbose:
        print(f"Created: {filename} (npts={len(data)}, channel={channel_name})")


# ============================================================================
# 主函数
# ============================================================================

def convert_obem_tsm_to_sac(config_file: str, verbose: bool = True) -> None:
    """OBEM TSM转SAC主函数
    
    Args:
        config_file: 配置文件路径
        verbose: 是否输出详细信息
    """
    if verbose:
        print("=" * 60)
        print("OBEM TSM to SAC Conversion (Python)")
        print("=" * 60)
    
    # 1. 读取配置
    if verbose:
        print(f"\n[1] Reading config file: {config_file}")
    config = read_config_file(config_file)
    
    # 提取配置参数
    paths = config.get('paths', {})
    station = config.get('station', {})
    parameters = config.get('parameters', {})
    
    input_path = paths.get('input_path', '.')
    output_path = paths.get('output_path', '.')
    location_file = paths.get('location_file', '')
    
    station_name = station.get('station_name', 'OBEM')
    station_lat = float(station.get('latitude', 0.0))
    station_lon = float(station.get('longitude', 0.0))
    station_elev = float(station.get('elevation', 0.0))
    network = station.get('network', 'OBEM')
    
    original_srate = parameters.get('original_srate', 300)  # Hz
    target_srate = parameters.get('target_srate', 500)  # Hz
    data_length = parameters.get('data_length', 21600)  # 6小时
    et = parameters.get('et', 1.885)  # 时间校准
    binaryformat = parameters.get('binaryformat', 1)  # 1=TSM_NEW
    year = parameters.get('year', 2022)  # 数据年份
    
    dt0 = 1.0 / original_srate  # 原始采样间隔
    dt_new = 1.0 / target_srate  # 新采样间隔
    
    if verbose:
        print(f"    Input path: {input_path}")
        print(f"    Output path: {output_path}")
        print(f"    Station: {station_name} (lat={station_lat:.6f}°, lon={station_lon:.6f}°)")
        print(f"    Original rate: {original_srate}Hz, Target rate: {target_srate}Hz")
        print(f"    Data length: {data_length}s ({data_length/3600:.1f} hours)")
    
    # 2. 查找TSM文件
    if verbose:
        print(f"\n[2] Finding TSM files in: {input_path}")
    tsm_pattern = os.path.join(input_path, '*.TSM')
    tsm_files = glob.glob(tsm_pattern) + glob.glob(os.path.join(input_path, '*.tsm'))
    tsm_files.sort()
    
    if len(tsm_files) == 0:
        raise FileNotFoundError(f"No TSM files found in {input_path}")
    
    if verbose:
        print(f"    Found {len(tsm_files)} TSM files")
    
    # 3. 读取所有TSM文件并合并数据
    if verbose:
        print(f"\n[3] Reading TSM files...")
    
    all_fields = []
    all_block_times = []
    file_times = []
    
    for tsm_file in tsm_files:
        try:
            # 从文件名解析时间
            file_time = parse_tsm_filename(tsm_file)
            file_times.append(file_time)
            
            if verbose:
                print(f"    Reading: {os.path.basename(tsm_file)} ({file_time})")
            
            # 读取TSM文件
            field, block_times = read_tsm_file(tsm_file, binaryformat=binaryformat)
            all_fields.append(field)
            all_block_times.extend(block_times)
            
            if verbose:
                print(f"      Extracted {len(field)} samples")
                
        except Exception as e:
            print(f"    Error reading {tsm_file}: {e}")
            continue
    
    if len(all_fields) == 0:
        raise ValueError("No valid TSM data extracted")
    
    # 合并所有数据
    field = np.vstack(all_fields)
    field_length = len(field)
    
    if verbose:
        print(f"\n    Total samples: {field_length}")
    
    # 4. 构建时间轴
    if verbose:
        print(f"\n[4] Building time axis...")
    
    # 计算起始时间（相对于年初的秒数）
    if len(file_times) > 0:
        first_file_time = file_times[0]
        # 计算年初的datetime
        year_start = datetime(year, 1, 1, 0, 0, 0)
        # 计算相对于年初的秒数
        tme_st = (first_file_time - year_start).total_seconds() + et
        
        # 构建时间轴（使用原始采样率）
        tme_dt = np.linspace(tme_st, tme_st + (field_length - 1) * dt0, field_length)
        
        if verbose:
            print(f"    Start time: {first_file_time}")
            print(f"    Time range: {tme_dt[0]:.1f}s to {tme_dt[-1]:.1f}s")
    else:
        raise ValueError("No valid file times")
    
    # 5. 数据插值（从原始采样率到目标采样率）
    if verbose:
        print(f"\n[5] Interpolating data from {original_srate}Hz to {target_srate}Hz...")
    
    # 限制数据长度
    if data_length > 0 and field_length * dt0 > data_length:
        max_samples = int(data_length / dt0)
        field = field[:max_samples, :]
        tme_dt = tme_dt[:max_samples]
        field_length = len(field)
        if verbose:
            print(f"    Truncated to {field_length} samples ({data_length}s)")
    
    # 构建新的时间轴（目标采样率）
    if field_length > 0:
        tme_new = np.arange(tme_dt[0], tme_dt[-1] + dt_new, dt_new)
        npts_new = len(tme_new)
    else:
        raise ValueError("No data to interpolate")
    
    if verbose:
        print(f"    Original samples: {field_length}, New samples: {npts_new}")
    
    # 对每个通道进行插值
    field_interp = np.zeros((npts_new, 5), dtype=np.float32)
    channel_names = ['Ex', 'Ey', 'Hx', 'Hy', 'FHz']
    
    for ich in range(5):
        # 使用线性插值
        interp_func = interp1d(tme_dt, field[:, ich], kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
        field_interp[:, ich] = interp_func(tme_new)
    
    if verbose:
        print(f"    Interpolation completed")
    
    # 6. 写入SAC文件
    if verbose:
        print(f"\n[6] Writing SAC files...")
    
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 计算起始UTC时间
    start_utc = UTCDateTime(first_file_time) + et
    
    # 为每个通道写入SAC文件
    for ich, channel_name in enumerate(channel_names):
        # 构建输出文件名
        # 格式：{station_name}.{channel_name}.{year}.{julian_day}.{hour:02d}{minute:02d}{second:02d}.SAC
        # 使用UTCDateTime获取准确的儒略日
        output_filename = f"{station_name}.{channel_name}.{start_utc.year}.{start_utc.julday:03d}.{start_utc.hour:02d}{start_utc.minute:02d}{start_utc.second:02d}.SAC"
        output_filepath = os.path.join(output_path, output_filename)
        
        # 写入SAC文件
        try:
            write_sac_file(
                filename=output_filepath,
                data=field_interp[:, ich],
                dt=dt_new,
                starttime=start_utc,
                station_lat=station_lat,
                station_lon=station_lon,
                station_elev=station_elev,
                station_name=station_name,
                channel_name=channel_name,
                network=network,
                verbose=verbose
            )
        except Exception as e:
            print(f"    Error writing {channel_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if verbose:
        print(f"\n" + "=" * 60)
        print("TSM to SAC conversion completed!")
        print(f"Total samples: {field_length} -> {npts_new}")
        print(f"Output directory: {output_path}")
        print("=" * 60)


def main():
    """命令行接口"""
    if len(sys.argv) < 2:
        print("Usage: python obem_tsm_to_sac_obspy.py config_file")
        print("\nExample config file:")
        print("[Paths]")
        print("input_path = /path/to/tsm/files")
        print("output_path = /path/to/output")
        print("location_file = location.txt")
        print("\n[Station]")
        print("station_name = WP12")
        print("latitude = 20.5")
        print("longitude = 120.3")
        print("elevation = -3000")
        print("network = OBEM")
        print("\n[Parameters]")
        print("original_srate = 300")
        print("target_srate = 500")
        print("data_length = 21600")
        print("et = 1.885")
        print("binaryformat = 1")
        print("year = 2022")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    try:
        convert_obem_tsm_to_sac(config_file, verbose=True)
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

