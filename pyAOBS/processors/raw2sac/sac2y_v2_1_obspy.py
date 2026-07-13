#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python implementation of sac2y_v2.1.c using ObsPy and NumPy

Convert SAC format seismic data to SEGY format, extracting waveform segments
based on shot point information from UKOOA files.

Original C program:
    program name: sac2y.c (sac format convert to segy format)
    written by: Minghui Zhao & Xuelin Qiu
    Version: Ver.2.1
    updated by Aowei 2010/03/19, adding the drift time of OBS
    updated by Haoyu Zhang 2017/04/15 as "v2.1" version to be applicable 
                              for non-integral sampling rate
    
    Edited by: Haibo Huang, 2025/11/21. Python implementation with performance optimization.

Usage:
    python sac2y_v2_1_obspy.py sac_file ukooa_file output_segy_file config_file

Requirements:
    pip install obspy numpy pyproj
    pip install numba  # 可选，用于JIT编译加速

Performance Optimization:
    - NumPy vectorization for fast data processing
    - Batch processing for multiple shots
    - Optimized IBM float conversion
    - Optional Numba JIT compilation for hotspot functions
"""

import os
import sys
import struct
import numpy as np
from typing import Dict, List, Tuple, Optional
from obspy import read
from obspy.core import UTCDateTime

# 尝试导入 Numba（可选）
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# 统一从各工具模块导入函数
try:
    # 从 format_utils.py 导入通用工具
    from format_utils import (
        parse_time_string,
        read_location_file,
        setup_tm_projection,
        lonlat_to_utm,
        calculate_drift_rate,
        calculate_shot_time,
        extract_traces_batch
    )
    
    # 从 segy_utils.py 导入SEGY写入工具
    from segy_utils import (
        create_segy_binary_header,
        create_segy_trace_header,
        write_segy_file
    )
    
    # 从 ukooa_utils.py 导入UKOOA解析工具
    from ukooa_utils import (
        parse_ukooa_file
    )
    
    # 从 config_utils.py 导入配置读取工具
    from config_utils import (
        read_unified_config_file
    )
    
    # 导入pyproj（如果format_utils.py已导入则可用，否则需要单独导入）
    try:
        from pyproj import Transformer
    except ImportError:
        # 如果format_utils中的坐标投影函数不可用，这里也不导入
        Transformer = None
except ImportError as e:
    print(f"Error: Cannot import required modules: {e}")
    print("Please ensure all utility modules are in the same directory.")
    sys.exit(1)



# ============================================================================
# 主函数
# ============================================================================

def convert_sac_to_segy(sac_file: str, ukooa_file: str, output_file: str,
                        config_file: str, verbose: bool = True) -> None:
    """SAC转SEGY主函数
    
    Args:
        sac_file: SAC文件路径
        ukooa_file: UKOOA文件路径
        output_file: 输出SEGY文件路径
        config_file: 统一配置文件路径
        verbose: 是否输出详细信息
    """
    if verbose:
        print("=" * 60)
        print("SAC to SEGY Conversion (Python v2.1)")
        print("=" * 60)
    
    # 1. 读取统一配置文件
    if verbose:
        print(f"\n[1] Reading config file: {config_file}")
    
    config = read_unified_config_file(config_file)
    
    # 提取参数
    params_dict = config.get('parameters', {})
    drift_dict = config.get('drift', {})
    location_dict = config.get('location', {})
    
    # 构建参数字典
    params = {
        'rv': params_dict.get('rv', 8.0),
        'length': int(params_dict.get('length', 6)),
        'tcoor': int(params_dict.get('tcoor', 2)),
        'lon0': float(params_dict.get('lon0', 120.0)),
        'ex': float(params_dict.get('ex', 0.0)),
        'ey': float(params_dict.get('ey', 0.0)),
        'et': float(params_dict.get('et', 1.885))
    }
    
    # 构建漂移参数字典（可选，如果没有提供则默认无漂移）
    has_drift = False
    drift_params = {}
    
    # 检查是否有漂移配置（[Drift]部分存在且包含start_time和end_time）
    start_time_str = drift_dict.get('start_time')
    end_time_str = drift_dict.get('end_time')
    
    if start_time_str and end_time_str:
        # 有漂移配置，解析时间
        has_drift = True
        start_dt = parse_time_string(str(start_time_str))
        end_dt = parse_time_string(str(end_time_str))
        
        drift_params = {
            'styear': start_dt['year'],
            'stmonth': start_dt['month'],
            'stday': start_dt['day'],
            'sthour': start_dt['hour'],
            'stmin': start_dt['minute'],
            'stsec': start_dt['second'],
            'edyear': end_dt['year'],
            'edmonth': end_dt['month'],
            'edday': end_dt['day'],
            'edhour': end_dt['hour'],
            'edmin': end_dt['minute'],
            'edsec': end_dt['second'],
            'drift': float(drift_dict.get('drift', 0.0))
        }
    elif start_time_str or end_time_str:
        # 只提供了部分参数，给出警告
        raise ValueError("Both 'start_time' and 'end_time' must be provided in [Drift] section, or omit [Drift] section entirely to disable drift correction.")
    
    # 获取位置信息
    config_longitude = location_dict.get('longitude')
    config_latitude = location_dict.get('latitude')
    config_location_file = location_dict.get('location_file', '')
    if config_location_file == '':
        config_location_file = None
    
    if verbose:
        print(f"    rv={params['rv']}, length={params['length']}s, tcoor={params['tcoor']}s")
        print(f"    lon0={params['lon0']}°, ex={params['ex']}m, ey={params['ey']}m, et={params['et']}s")
    
    # 2. 读取SAC文件
    if verbose:
        print(f"\n[2] Reading SAC file: {sac_file}")
    sac_stream = read(sac_file)
    if len(sac_stream) == 0:
        raise ValueError("SAC file is empty")
    
    sac_trace = sac_stream[0]
    sac_stats = sac_trace.stats
    
    # 检查SAC文件
    if hasattr(sac_stats, 'sac') and sac_stats.sac.b != 0.0:
        raise ValueError(f"SAC file b field is not zero: {sac_stats.sac.b}")
    
    # 获取SAC数据和时间信息
    sac_data = sac_trace.data.astype(np.float32)  # 确保是float32
    delta = sac_stats.delta
    npts = sac_stats.npts
    
    # 创建参考时间
    sac_ref_time = sac_stats.starttime
    
    # 获取SAC头信息
    if hasattr(sac_stats, 'sac'):
        sac_sac = sac_stats.sac
        sac_year = sac_sac.nzyear
        sac_nzjday = sac_sac.nzjday
        sac_hour = sac_sac.nzhour
        sac_min = sac_sac.nzmin
        sac_sec = sac_sac.nzsec
        sac_msec = sac_sac.nzmsec
        gelev = getattr(sac_sac, 'evel', 0.0)
    else:
        # 从stats中获取
        sac_year = sac_ref_time.year
        sac_nzjday = sac_ref_time.julday
        sac_hour = sac_ref_time.hour
        sac_min = sac_ref_time.minute
        sac_sec = sac_ref_time.second
        sac_msec = int(sac_ref_time.microsecond / 1000)
        gelev = 0.0
    
    if verbose:
        print(f"    npts={npts}, delta={delta}s, year={sac_year}, julday={sac_nzjday}")
        print(f"    Reference time: {sac_ref_time}")
    
    # 计算参考时间（秒，从当天0时开始）
    t1 = sac_hour * 3600 + sac_min * 60 + sac_sec + sac_msec / 1000.0
    if verbose:
        print(f"    Reference time from midnight: {t1:.3f}s")
    
    # 3. 漂移参数处理（可选）
    if has_drift:
        if verbose:
            print(f"\n[3] Drift parameters loaded from config file")
        
        # 计算漂移率（包含详细的中间结果输出）
        drift_rate, stjday, edjday = calculate_drift_rate(drift_params)
        
        if verbose:
            # 计算timegap用于调试输出（与C代码保持一致）
            edmsec = 0
            stmsec = 0
            timegap = float((edjday - stjday + 1) * 24 * 3600 +
                            (drift_params['edhour'] - drift_params['sthour']) * 3600 +
                            (drift_params['edmin'] - drift_params['stmin']) * 60 +
                            (drift_params['edsec'] - drift_params['stsec']) +
                            (edmsec - stmsec) / 1000.0)
            
            print(f"    Start time: {drift_params['styear']}-{drift_params['stmonth']:02d}-{drift_params['stday']:02d} "
                  f"{drift_params['sthour']:02d}:{drift_params['stmin']:02d}:{drift_params['stsec']:02d}")
            print(f"    End time: {drift_params['edyear']}-{drift_params['edmonth']:02d}-{drift_params['edday']:02d} "
                  f"{drift_params['edhour']:02d}:{drift_params['edmin']:02d}:{drift_params['edsec']:02d}")
            print(f"    Julian days: stjday={stjday}, edjday={edjday}")
            print(f"    Drift (total): {drift_params['drift']:.6f} ms")
            print(f"    Time gap: {timegap:.6f} seconds ({timegap/3600:.2f} hours)")
            print(f"    Drift rate: {drift_rate:.9f} ms/s")
    else:
        # 没有漂移配置，使用默认值（无漂移）
        drift_rate = 0.0
        stjday = 0  # 默认值，不会被使用
        drift_params = {
            'sthour': 0,
            'stmin': 0,
            'stsec': 0
        }
        if verbose:
            print(f"\n[3] No drift correction (drift rate = 0.0 ms/s)")
    
    # 4. 获取台站位置
    if verbose:
        print(f"\n[4] Reading station location")
    
    # 检查SAC文件中是否有位置信息
    has_sac_location = False
    if hasattr(sac_stats, 'sac'):
        # 检查stla字段是否存在且不是未定义值(-12345)
        if hasattr(sac_stats.sac, 'stla'):
            stla_value = sac_stats.sac.stla
            # SAC使用-12345表示未定义的浮点值
            if stla_value != -12345.0 and not np.isnan(stla_value):
                has_sac_location = True
                lat1 = sac_stats.sac.stla
                lon1 = sac_stats.sac.stlo
                if verbose:
                    print(f"    Location from SAC header: lat={lat1:.6f}°, lon={lon1:.6f}°")
    
    # 如果SAC文件中没有有效的位置信息，则从配置文件读取
    if not has_sac_location:
        # 优先使用直接指定的经纬度
        if config_latitude is not None and config_longitude is not None:
            lat1 = float(config_latitude)
            lon1 = float(config_longitude)
            if verbose:
                print(f"    Location from config file: lat={lat1:.6f}°, lon={lon1:.6f}°")
        # 如果配置文件中指定了location_file，则从文件读取
        elif config_location_file:
            lon1, lat1 = read_location_file(config_location_file)
            if verbose:
                print(f"    Location from {config_location_file}: lat={lat1:.6f}°, lon={lon1:.6f}°")
        else:
            raise ValueError("SAC header has no position information, and no location found in config file")
    
    # 5. 设置UTM投影
    if verbose:
        print(f"\n[5] Setting up UTM projection (center lon={params['lon0']}°)")
    transformer = setup_tm_projection(params['lon0'])
    gx, gy = lonlat_to_utm(lon1, lat1, transformer, params['ex'], params['ey'])
    
    # 检查台站坐标是否有效
    if not np.isfinite(gx) or not np.isfinite(gy):
        raise ValueError(f"Invalid station coordinates: gx={gx}, gy={gy}. Check station location (lat={lat1}, lon={lon1}).")
    
    if verbose:
        print(f"    Station UTM coordinates: x={gx:.1f}m, y={gy:.1f}m")
    
    # 6. 创建SEGY二进制头
    if verbose:
        print(f"\n[6] Creating SEGY binary header")
    binary_header = create_segy_binary_header(delta, params['length'])
    if verbose:
        print(f"    Sample interval: {binary_header['hdt']} microseconds")
        print(f"    Samples per trace: {binary_header['hns']}")
    
    # 7. 解析UKOOA文件
    if verbose:
        print(f"\n[7] Parsing UKOOA file: {ukooa_file}")
    shots = parse_ukooa_file(ukooa_file, sac_year, sac_nzjday)
    if verbose:
        print(f"    Found {len(shots)} shots")
    
    if len(shots) == 0:
        raise ValueError("No valid shots found in UKOOA file")
    
    # 8. 处理每个炮点（批量处理优化）
    if verbose:
        print(f"\n[8] Processing shots...")
    
    # 预计算所有炮点的坐标和时间
    valid_shots = []
    shot_times_t3 = []  # 存储t3值（相对于当天0时的秒数）
    shot_coords = []  # (sx, sy, offset)
    
    prev_sample_pos = -int(params['length'] / delta + 0.5)  # 类似TT0_temp
    
    if verbose:
        print(f"    SAC reference time (t1): {t1:.3f} seconds from midnight")
        print(f"    SAC file has {npts} samples (duration: {npts * delta:.3f} seconds)")
        print(f"    Station location: lat={lat1:.6f}°, lon={lon1:.6f}°")
        print(f"    Station UTM: gx={gx:.1f}m, gy={gy:.1f}m")
        print(f"    Processing shots...")
    
    for i, shot in enumerate(shots):
        # 计算炮点坐标
        sx, sy = lonlat_to_utm(shot['longitude'], shot['latitude'],
                               transformer, params['ex'], params['ey'])
        
        # 检查坐标是否有效
        if not np.isfinite(sx) or not np.isfinite(sy):
            continue
        
        # 计算偏移距（先计算绝对值）
        offset_abs = np.sqrt((sx - gx)**2 + (sy - gy)**2)
        
        # 检查offset是否为有效值
        if not np.isfinite(offset_abs):
            continue
        
        # 偏移距符号处理（如果炮点在台站南侧，offset为负）
        # 对应C代码：if(lat2<=lat1) tr.offset=-tr.offset;
        offset = offset_abs  # 先设为正数
        if shot['latitude'] <= lat1:
            offset = -offset_abs
        
        # 计算炮点时间（含各种校正，返回相对于参考时间的秒数）
        # 如果没有漂移配置，stjday不会被使用（drift_rate=0时，漂移校正项为0）
        t3 = calculate_shot_time(
            shot, sac_ref_time, drift_params if has_drift else {}, drift_rate,
            params['rv'], offset, params['et'],
            drift_params.get('sthour', 0), drift_params.get('stmin', 0),
            drift_params.get('stsec', 0), 0,  # stmsec=0
            stjday if has_drift else 0  # 如果没有漂移，传入0（不会被使用）
        )
        
        # 计算采样位置（对应C代码的t4_temp计算）
        # C代码：t4_temp = (int)((1.0/sp.delta)*(t3-t1)-(1.0/sp.delta)*tcoor+0.5)
        # t1是SAC参考时间（从当天0时开始的秒数）
        
        # 检查delta是否有效（避免除零或无穷大）
        if delta <= 0 or not np.isfinite(delta):
            raise ValueError(f"Invalid delta value: {delta}")
        
        # 检查t3是否为有效值（避免无穷大）
        if not np.isfinite(t3):
            continue
        
        # 计算采样位置
        sample_pos_raw = ((t3 - t1) / delta) - (params['tcoor'] / delta)
        
        # 检查sample_pos_raw是否为有效值
        if not np.isfinite(sample_pos_raw):
            continue
        
        sample_pos = int(sample_pos_raw + 0.5)
        n_samples = int((params['length'] / delta) + 0.5)
        
        # 检查计算结果是否有效
        if not np.isfinite(sample_pos) or not np.isfinite(n_samples) or sample_pos < -1e9 or sample_pos > 1e9:
            continue
        
        # 检查边界
        if sample_pos < 0:
            continue
        
        if sample_pos + n_samples > npts:
            break  # SAC文件结束
        
        # 检查重叠（对应C代码的TT_temp检查）
        gap = sample_pos - prev_sample_pos - n_samples
        if gap < 0:
            continue
        
        # 保存有效炮点
        valid_shots.append(shot)
        shot_times_t3.append(t3)
        shot_coords.append((sx, sy, offset))
        prev_sample_pos = sample_pos
    
    if verbose:
        print(f"    {len(valid_shots)} valid shots after filtering")
    
    if len(valid_shots) == 0:
        raise ValueError("No valid shots after filtering")
    
    # 9. 批量提取数据
    if verbose:
        print(f"\n[9] Extracting trace data (vectorized)...")
    
    # 批量计算采样位置（向量化）
    # 检查delta是否有效
    if delta <= 0 or not np.isfinite(delta):
        raise ValueError(f"Invalid delta value: {delta}")
    
    n_samples = int((params['length'] / delta) + 0.5)
    
    # 向量化计算采样位置，过滤无效值
    shot_times_t3_arr = np.array(shot_times_t3, dtype=np.float64)
    valid_times_mask = np.isfinite(shot_times_t3_arr)
    
    # 先计算所有位置
    start_samples_raw = ((shot_times_t3_arr - t1) / delta) - (params['tcoor'] / delta)
    start_samples = np.full(len(shot_times_t3), -1, dtype=np.int32)  # 初始化为-1表示无效
    start_samples[valid_times_mask] = np.round(start_samples_raw[valid_times_mask]).astype(np.int32)
    
    # 批量提取数据
    traces_data, valid_mask = extract_traces_batch(sac_data, start_samples, n_samples)
    
    # 再次过滤（因为可能有些位置无效）
    final_valid_mask = valid_mask & ~np.isnan(traces_data[:, 0])
    
    # 额外检查坐标和offset是否有效
    coord_valid_mask = np.ones(len(shot_coords), dtype=bool)
    for i, (sx, sy, offset) in enumerate(shot_coords):
        if not np.isfinite(sx) or not np.isfinite(sy) or not np.isfinite(offset):
            coord_valid_mask[i] = False
        elif abs(offset) > 1e9:  # 偏移距不合理（超过10亿米）
            coord_valid_mask[i] = False
    
    # 组合所有过滤条件
    final_valid_mask = final_valid_mask & coord_valid_mask
    
    valid_traces = [traces_data[i] for i in range(len(traces_data)) if final_valid_mask[i]]
    valid_shots_final = [valid_shots[i] for i in range(len(valid_shots)) if final_valid_mask[i]]
    valid_coords_final = [shot_coords[i] for i in range(len(shot_coords)) if final_valid_mask[i]]
    valid_times_t3 = [shot_times_t3[i] for i in range(len(shot_times_t3)) if final_valid_mask[i]]
    
    if verbose:
        print(f"    Extracted {len(valid_traces)} traces")
    
    # 10. 批量计算坐标（如果需要）
    # 坐标已经在前面计算过了，这里只需要使用
    
    # 11. 批量创建道头
    if verbose:
        print(f"\n[10] Creating trace headers...")
    
    trace_headers = []
    for i, (shot, (sx, sy, offset), t3) in enumerate(zip(valid_shots_final, valid_coords_final, valid_times_t3)):
        # 再次检查坐标和offset是否有效（防止在批量处理后出现无效值）
        if not np.isfinite(sx) or not np.isfinite(sy) or not np.isfinite(gx) or not np.isfinite(gy):
            if verbose:
                print(f"Warning: Trace {i+1}: Invalid coordinates (sx={sx}, sy={sy}, gx={gx}, gy={gy}), skipping...")
            continue
        
        if not np.isfinite(offset):
            if verbose:
                print(f"Warning: Trace {i+1}: Invalid offset (inf/nan), skipping...")
            continue
        
        # 检查offset是否在合理范围内（避免无穷大）
        offset_abs = abs(offset)
        if offset_abs > 1e9:  # 超过10亿米（不合理）
            if verbose:
                print(f"Warning: Trace {i+1}: Offset too large ({offset_abs:.1f}m), skipping...")
            continue
        
        # t3是相对于当天0时的秒数，需要转换为时分秒
        # 对应C代码：tr.hour=(int)(t3/3600); tr.minute=(int)((t3-tr.hour*3600)/60); etc.
        hour = int(t3 / 3600)
        minute = int((t3 - hour * 3600) / 60)
        sec = int(t3 - hour * 3600 - minute * 60)
        timbas = int((t3 - hour * 3600 - minute * 60 - sec) * 1000 + 0.5)
        
        # 注意：offset应该保留符号（正数或负数），不要使用绝对值
        # 对应C代码：if(lat2<=lat1) tr.offset=-tr.offset; 然后直接写入SEGY
        trace_header = create_segy_trace_header(
            shot_info=shot,
            sx=sx, sy=sy, gx=gx, gy=gy,
            offset=offset,  # 使用有符号的offset（不是绝对值）
            gelev=gelev,
            swdep=shot['water_depth'],
            delrt=int(params['tcoor'] * (-1000)),
            ns=n_samples,
            dt=int(delta * 1e6 + 0.5),
            tracl=i + 1,
            tracr=i + 1,
            year=sac_year,
            day=shot['day'],
            hour=hour,
            minute=minute,
            sec=sec,
            timbas=timbas
        )
        trace_headers.append(trace_header)
    
    # 12. 写入SEGY文件
    if verbose:
        print(f"\n[11] Writing SEGY file: {output_file}")
    
    write_segy_file(output_file, valid_traces, trace_headers, binary_header, verbose=verbose)
    
    if verbose:
        print("\n" + "=" * 60)
        print("Conversion completed successfully!")
        print("=" * 60)


def main():
    """命令行接口"""
    if len(sys.argv) < 5:
        print("Error: Invalid number of arguments!")
        print("Usage: python sac2y_v2_1_obspy.py sac_file ukooa_file output_segy_file config_file")
        print("\nArguments:")
        print("  sac_file:        Input SAC file path")
        print("  ukooa_file:      Input UKOOA shot point file path")
        print("  output_segy_file: Output SEGY file path")
        print("  config_file:     Configuration file path (INI format)")
        print("\nExample:")
        print("  python sac2y_v2_1_obspy.py data.sac shots.ukooa output.segy sac2y_config.ini")
        sys.exit(1)
    
    sac_file = sys.argv[1]
    ukooa_file = sys.argv[2]
    output_file = sys.argv[3]
    config_file = sys.argv[4]
    
    # 验证输入文件
    if not os.path.exists(sac_file):
        print(f"Error: SAC file not found: {sac_file}")
        sys.exit(1)
    
    if not os.path.exists(ukooa_file):
        print(f"Error: UKOOA file not found: {ukooa_file}")
        sys.exit(1)
    
    if not os.path.exists(config_file):
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)
    
    try:
        convert_sac_to_segy(sac_file, ukooa_file, output_file, config_file, verbose=True)
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

