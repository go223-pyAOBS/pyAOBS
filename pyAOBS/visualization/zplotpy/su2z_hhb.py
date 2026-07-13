"""
su2z_hhb.py - SU格式转Z格式转换工具

功能说明:
--------
将Seismic Unix (SU) 格式地震数据转换为Z格式，用于ZPLOT程序进行地震相位拾取和显示。

本工具基于 su2z_hhb.c (Haibo Huang 2023) 的Python实现，支持多分量地震数据转换。

使用方法:
--------
1. 命令行使用:
   python su2z_hhb.py <输入SU文件> [选项]
   
   示例:
   # 基本转换（输出 data.z 和 file.hdr）
   python su2z_hhb.py input.su
   
   # 指定输出文件
   python su2z_hhb.py input.su --zdata output.z --hfile output.hdr
   
   # 启用详细输出和振幅测试
   python su2z_hhb.py input.su --verbose 1 --test-amplitude 5
   
   # 自定义记录号字段（如果cdp未赋值，使用fldr）
   python su2z_hhb.py input.su --key2 fldr

2. Python代码调用:
   from su2z_hhb import su2z_hhb
   
   result = su2z_hhb(
       su_file='input.su',
       zdata_file='output.z',
       hfile='output.hdr',
       verbose=1
   )

主要参数:
--------
- su_file: 输入的SU格式文件路径（必需）
- zdata_file: 输出的Z格式数据文件（默认: data.z）
- hfile: 输出的头文件（默认: file.hdr）
- key1: 第一个关键字段，用于profile number（默认: fldr）
- key2: 第二个关键字段，用于record/gather number（默认: cdp）
        注意：如果cdp字段未赋值（全部为0），所有道将归为记录号1
- key3: 第三个关键字段，用于trace number（默认: tracf）
- key4: 第四个关键字段，用于component type（默认: trid）
- npick: 拾取字数（默认: 40，最大40，与C代码MAXPICK一致）
- vred: 折合速度（km/s），如果为None则从SU头文件读取
- tstart: 起始时间（毫秒），如果为None则从SU头文件第一道的stas字段读取
- tend: 结束时间（毫秒），如果为None则从SU头文件第一道的stae字段读取，如果stae不存在则自动计算
- verbose: 详细输出级别（0=静默，1=基本信息，2=详细信息）
- test_amplitude: 输出前N道的振幅信息用于测试（默认: 0=不输出）

输出文件:
--------
1. .z文件: Z格式数据文件，包含文件头和所有道数据
2. .hdr文件: 头文件，包含详细的道头信息和拾取数组

数据类型映射:
-----------
SU格式的trid字段映射到Z格式的itype:
- trid=11 (hydrophone) -> itype=4
- trid=12 (raw vertical) -> itype=1 (vertical)
- trid=13 (raw crossline) -> itype=3 (transverse)
- trid=14 (raw inline) -> itype=2 (radial)
- trid=15 (rotated vertical) -> itype=1 (vertical)
- trid=16 (rotated transverse) -> itype=3 (transverse)
- trid=17 (rotated radial) -> itype=2 (radial)

注意事项:
--------
1. 依赖库: 需要安装 obspy 或 supython 模块来读取SU格式数据
   - 推荐使用: pip install obspy
   - 或使用项目内的 supython 模块

2. 记录号处理:
   - 如果SU文件中的cdp字段未赋值（全部为0），所有道将自动归为记录号1
   - 记录号会被规范化，从1开始连续编号（1, 2, 3...）

3. 偏移距处理:
   - 程序会自动统计偏移距范围
   - 如果包含负偏移距，会在输出中提示设置ZPLOT的xmin/xmax参数

4. 振幅检查:
   - 使用 --test-amplitude N 可以输出前N道的振幅统计信息
   - 如果振幅过小，程序会提示调整ZPLOT的显示参数

5. 坐标系统:
   - 支持UTM坐标和经纬度坐标
   - 自动处理scalco缩放因子

示例输出:
--------
转换完成后，程序会输出:
- 记录号映射关系
- 每道的基本信息（如果verbose>1）
- 偏移距统计（范围、负偏移距数量等）
- 转换结果摘要（总道数、记录数、采样参数等）

Author: Based on su2z_hhb.c by Haibo Huang, Python implementation 2024
"""

import struct
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import os

try:
    from obspy import read as obspy_read
    HAS_OBSPY = True
except ImportError:
    HAS_OBSPY = False
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../processors'))
        from supython import readsu
        HAS_SUPYTHON = True
    except ImportError:
        HAS_SUPYTHON = False


# 数据类型映射（从SU的trid到Z格式的itype）
TRID_TO_ITYPE = {
    11: 4,  # hydrophone
    12: 1,  # raw vertical -> vertical
    13: 3,  # raw crossline -> transverse
    14: 2,  # raw inline -> radial
    15: 1,  # rotated vertical -> vertical
    16: 3,  # rotated transverse -> transverse
    17: 2,  # rotated radial -> radial
}


def su2z_hhb(su_file: str, 
              zdata_file: str = 'data.z',
              hfile: str = 'file.hdr',
              key1: str = 'fldr',
              key2: str = 'cdp',  # 记录号字段（与原始C代码一致）
              key3: str = 'tracf',
              key4: str = 'trid',
              nrec: int = 0,
              npick: int = 40,  # 默认40（与C代码MAXPICK一致）
              vred: Optional[float] = None,
              tstart: Optional[int] = None,
              tend: Optional[int] = None,
              verbose: int = 0,
              test_amplitude: int = 0) -> Dict[str, any]:
    """将SU格式数据转换为Z格式
    
    Args:
        su_file: 输入的SU格式文件路径
        zdata_file: 输出的Z格式数据文件路径（.z文件）
        hfile: 输出的头文件路径（.hdr文件）
        key1: 第一个关键字段（默认'fldr'，profile number）
        key2: 第二个关键字段（默认'cdp'，record/gather number）
        key3: 第三个关键字段（默认'tracf'，trace number within record）
        key4: 第四个关键字段（默认'trid'，component name）
        nrec: 记录数（炮集数），0表示自动计算
        npick: 拾取字数（默认40，最大40，与C代码MAXPICK一致）
        vred: 折合速度（km/s），如果为None则从SU头文件读取
        tstart: 起始时间（毫秒），如果为None则从SU头文件读取
        tend: 结束时间（毫秒），如果为None则从SU头文件读取
        verbose: 详细输出级别（0=静默，1=基本信息，2=详细信息）
        test_amplitude: 输出前N道的振幅信息用于测试（0=不输出）
    
    Returns:
        包含转换信息的字典
    """
    if not os.path.exists(su_file):
        raise FileNotFoundError(f"SU文件不存在: {su_file}")
    
    # 读取SU数据
    if HAS_OBSPY:
        traces, headers = _read_su_with_obspy(su_file)
    elif HAS_SUPYTHON:
        traces, headers = _read_su_with_supython(su_file)
    else:
        raise ImportError("需要安装 obspy 或 supython 模块来读取SU格式数据")
    
    ntraces = len(traces)
    if ntraces == 0:
        raise ValueError("SU文件中没有道数据")
    
    # 获取第一道的信息来确定文件头参数
    first_trace = traces[0]
    first_header = headers[0]
    
    # 确定采样参数
    npts = len(first_trace)
    if HAS_OBSPY:
        # dt字段：从header获取（微秒），如果没有则从stats.delta计算
        if 'dt' in first_header:
            sint = int(first_header['dt'])
        else:
            # 尝试从ObsPy的stats获取
            st = obspy_read(su_file, headonly=True)
            if len(st) > 0 and hasattr(st[0].stats, 'delta'):
                sint = int(st[0].stats.delta * 1e6)  # 秒转微秒
            else:
                sint = 4000  # 默认4ms
        
        # stas和stae：从header获取，如果没有则计算
        if tstart is not None:
            stas = tstart
        elif 'stas' in first_header:
            stas = int(first_header['stas'])
        else:
            stas = 0
        
        if tend is not None:
            stae = tend
        elif 'stae' in first_header:
            stae = int(first_header['stae'])
        else:
            # 计算：stae = stas + (npts - 1) * sint / 1000
            stae = stas + (npts - 1) * sint // 1000
        
        swevel = first_header.get('swevel', 6000)  # 默认6 km/s
        vredf = swevel / 1000.0 if vred is None else vred
    else:
        # 使用supython
        dt_microseconds = int(first_header.dt[0] * 1000)  # 毫秒转微秒
        sint = dt_microseconds
        
        # tstart处理：如果命令行提供了，使用命令行参数；否则使用第一道的stas
        if tstart is not None:
            stas = tstart
        elif hasattr(first_header, 'stas') and len(first_header.stas) > 0:
            stas = int(first_header.stas[0])
        else:
            stas = 0
        
        # tend处理：如果命令行提供了，使用命令行参数；否则使用第一道的stae，如果没有则计算
        if tend is not None:
            stae = tend
        elif hasattr(first_header, 'stae') and len(first_header.stae) > 0:
            stae = int(first_header.stae[0])
        else:
            # 计算：stae = stas + (npts - 1) * sint / 1000
            stae = stas + (npts - 1) * sint // 1000
        
        swevel = first_header.swevel[0] if hasattr(first_header, 'swevel') else 6000
        vredf = swevel / 1000.0 if vred is None else vred
    
    # 确定记录数并建立记录号映射
    # 将SU的key2值（可能是任意整数）映射到Z格式的连续记录号（1, 2, 3...）
    key2_to_recno = {}  # {原始key2值: 规范化后的记录号}
    recno_counter = 1  # 从1开始
    
    # 先收集所有不同的key2值并排序
    key2_values_set = set()
    key2_all_zero_or_missing = True  # 检查是否所有key2值都是0或未赋值
    
    for hdr in headers:
        if HAS_OBSPY:
            val = hdr.get(key2, 0)
        else:
            val = getattr(hdr, key2, [0])[0] if hasattr(hdr, key2) else 0
        
        # 检查值是否有效（非0）
        if val != 0:
            key2_all_zero_or_missing = False
        
        key2_values_set.add(val)
    
    # 如果key2字段未赋值（全部为0），默认所有道归为记录号1
    if key2_all_zero_or_missing and key2 == 'cdp':
        if verbose:
            print(f"\n警告: {key2}字段未赋值（全部为0），所有道将归为记录号1")
        
        # 所有道都使用记录号1
        key2_values_set = {0}  # 统一使用0值，映射到记录号1
        key2_to_recno[0] = 1
        nrec = 1
        
        if verbose:
            print(f"所有 {len(headers)} 道数据将归为记录号1")
    else:
        # 按值排序，建立映射（确保记录号从1开始连续）
        sorted_key2_values = sorted(key2_values_set)
        for orig_val in sorted_key2_values:
            key2_to_recno[orig_val] = recno_counter
            recno_counter += 1
        
        nrec = len(key2_to_recno)
    
    if verbose:
        print(f"\n记录号映射（原始key2值 -> Z格式记录号）:")
        for orig_val, mapped_recno in sorted(key2_to_recno.items()):
            print(f"  {orig_val} -> {mapped_recno}")
    
    # 文件头结构
    ifmt = 1  # float32格式
    
    # 创建输出目录
    os.makedirs(os.path.dirname(zdata_file) if os.path.dirname(zdata_file) else '.', exist_ok=True)
    os.makedirs(os.path.dirname(hfile) if os.path.dirname(hfile) else '.', exist_ok=True)
    
    # 打开输出文件
    fpout = open(zdata_file, 'wb')
    fphdr = open(hfile, 'wb')
    
    # 初始化文件头（先写入临时值，最后更新）
    fh_ntraces = 0  # 临时值，最后更新
    fh_npts = npts
    fh_sint = sint
    fh_tstart = stas
    fh_tend = stae
    fh_nrec = nrec
    fh_npick = min(npick, 40)  # 最大40
    fh_vredf = vredf
    fh_ifmt = ifmt
    fh_xlatlong = 1.0
    fh_xelev = 1.0
    fh_xutm = 1.0
    fh_cm = 0.0
    
    # 写入临时文件头（52字节）
    header_bytes = struct.pack('<7i', fh_ntraces, fh_npts, fh_sint, fh_tstart, fh_tend, fh_nrec, fh_npick)
    header_bytes += struct.pack('<f', fh_vredf)
    header_bytes += struct.pack('<i', fh_ifmt)
    header_bytes += struct.pack('<4f', fh_xlatlong, fh_xelev, fh_xutm, fh_cm)
    fpout.write(header_bytes)
    
    # 初始化拾取数组
    picks = np.zeros(fh_npick, dtype=np.float32)
    
    # 记录跟踪
    recno = -999
    nrec_actual = 0
    current_rec_traces = 0
    last_ival1 = 0  # 保存上一个记录的ival1值
    
    # 用于统计offset范围
    offset_values = []
    
    # 遍历所有道（actual_key2已在上面确定）
    for trace_idx, (trace_data, trace_header) in enumerate(zip(traces, headers)):
        # 获取关键字段值
        if HAS_OBSPY:
            ival1 = trace_header.get(key1, trace_idx + 1)
            ival2 = trace_header.get(key2, 0)  # 如果未赋值，默认为0
            ival3 = trace_header.get(key3, trace_idx + 1)
            ival4 = trace_header.get(key4, 11)  # 默认hydrophone
        else:
            ival1 = getattr(trace_header, key1, [trace_idx + 1])[0] if hasattr(trace_header, key1) else trace_idx + 1
            ival2 = getattr(trace_header, key2, [0])[0] if hasattr(trace_header, key2) else 0
            ival3 = getattr(trace_header, key3, [trace_idx + 1])[0] if hasattr(trace_header, key3) else trace_idx + 1
            ival4 = getattr(trace_header, key4, [11])[0] if hasattr(trace_header, key4) else 11
        
        # 如果key2字段未赋值（全部为0），强制使用记录号1
        if key2_all_zero_or_missing and key2 == 'cdp':
            ival2 = 0  # 统一使用0值
        
        # 将原始key2值映射到规范化的记录号（1, 2, 3...）
        mapped_recno = key2_to_recno.get(ival2, 1)
        
        # 收集offset值用于统计
        if HAS_OBSPY:
            offset_val = float(trace_header.get('offset', 0.0))
        else:
            offset_val = float(getattr(trace_header, 'offset', [0.0])[0])
        offset_values.append(offset_val)
        
        # 检查记录号是否改变（使用映射后的记录号）
        if recno != mapped_recno:
            if nrec_actual > 0 and verbose:
                # 查找上一个记录的原始key2值
                prev_orig_key2 = None
                for orig_val, mapped_val in key2_to_recno.items():
                    if mapped_val == recno:
                        prev_orig_key2 = orig_val
                        break
                if prev_orig_key2 is not None:
                    print(f"\n\tTotal {current_rec_traces} traces for profile {last_ival1} record {recno} (原始key2={prev_orig_key2})")
                else:
                    print(f"\n\tTotal {current_rec_traces} traces for profile {last_ival1} record {recno}")
            nrec_actual += 1
            recno = mapped_recno
            last_ival1 = ival1  # 保存当前记录的ival1
            current_rec_traces = 0
            if verbose > 1:
                print("\n\tProfNo RecNo ShotNo TraceNo Offset(km) Azimuth")
        
        current_rec_traces += 1
        
        # 确定数据类型
        itype = TRID_TO_ITYPE.get(ival4, 4)  # 默认hydrophone
        
        # 构建道头结构（使用映射后的记录号）
        th_nrec = recno
        th_itsn = current_rec_traces
        th_ireci = ival3
        th_itype = itype
        th_iflag = 1  # 有效道
        th_igain = 0
        
        # 从SU头文件提取字段
        if HAS_OBSPY:
            th_offset = float(trace_header.get('offset', 0.0))  # 米
            th_azi = float(trace_header.get('cdpt', 0.0)) / 60.0  # 分
            scalco = trace_header.get('scalco', 1)
            sx = float(trace_header.get('sx', 0.0))
            sy = float(trace_header.get('sy', 0.0))
            gx = float(trace_header.get('gx', 0.0))
            gy = float(trace_header.get('gy', 0.0))
            selev_val = trace_header.get('selev', 0)
            swdep_val = trace_header.get('swdep', 0)
            gelev_val = trace_header.get('gelev', 0)
        else:
            th_offset = float(getattr(trace_header, 'offset', [0.0])[0])
            th_azi = float(getattr(trace_header, 'cdpt', [0.0])[0]) / 60.0
            scalco = getattr(trace_header, 'scalco', [1])[0]
            sx = float(getattr(trace_header, 'sx', [0.0])[0])
            sy = float(getattr(trace_header, 'sy', [0.0])[0])
            gx = float(getattr(trace_header, 'gx', [0.0])[0])
            gy = float(getattr(trace_header, 'gy', [0.0])[0])
            selev_val = getattr(trace_header, 'selev', [0])[0]
            swdep_val = getattr(trace_header, 'swdep', [0])[0]
            gelev_val = getattr(trace_header, 'gelev', [0])[0]
        
        # 计算经纬度（考虑scalco缩放因子）
        abs_scalco = abs(scalco)
        if abs_scalco < 10:
            th_slat = sy * (10 ** scalco) / 3600.0
            th_slong = sx * (10 ** scalco) / 3600.0
            th_rlat = gy * (10 ** scalco) / 3600.0
            th_rlong = gx * (10 ** scalco) / 3600.0
        else:
            th_slat = sy / abs_scalco
            th_slong = sx / abs_scalco
            th_rlat = gy / abs_scalco
            th_rlong = gx / abs_scalco
        
        th_texact = 0.0
        th_selev = int(selev_val)
        th_swdepth = int(swdep_val)
        th_relev = int(gelev_val)
        th_sxutm = sx
        th_syutm = sy
        th_sz = 0.0
        th_rxutm = gx
        th_ryutm = gy
        th_rz = 0.0
        
        # 计算记录大小（头文件用）
        record_size = 88 + fh_npick * 4
        
        # 写入头文件（Fortran未格式化格式）
        fphdr.write(struct.pack('<i', record_size))  # 记录长度标记
        # 按照C结构体布局写入22个字段（88字节）
        fphdr.write(struct.pack('<5i', th_nrec, th_itsn, th_ireci, th_itype, th_iflag))
        fphdr.write(struct.pack('<2f', th_offset, th_azi))
        fphdr.write(struct.pack('<i', th_igain))
        fphdr.write(struct.pack('<3f', th_texact, th_slat, th_slong))
        fphdr.write(struct.pack('<2i', th_selev, th_swdepth))
        fphdr.write(struct.pack('<2f', th_rlat, th_rlong))
        fphdr.write(struct.pack('<i', th_relev))
        fphdr.write(struct.pack('<6f', th_sxutm, th_syutm, th_sz, th_rxutm, th_ryutm, th_rz))
        # 写入拾取数组
        fphdr.write(picks.tobytes())
        fphdr.write(struct.pack('<i', record_size))  # 记录长度标记（结束）
        
        # 写入.z文件（道头 + 拾取占位符 + 数据）
        # 改进（阶段2）：拾取信息不写入数据文件，只写入0值占位符
        # 实际拾取信息已保存到头文件（.hdr）
        # 道头（22个浮点数）+ 拾取占位符（npick个0值）
        trace_header_array = np.zeros(22 + fh_npick, dtype=np.float32)
        trace_header_array[0] = float(th_nrec)
        trace_header_array[1] = float(th_itsn)
        trace_header_array[2] = float(th_ireci)
        trace_header_array[3] = float(th_itype)
        trace_header_array[4] = float(th_iflag)
        trace_header_array[5] = th_offset  # 米
        trace_header_array[6] = th_azi  # 分
        trace_header_array[7] = float(th_igain)
        trace_header_array[8] = th_texact
        trace_header_array[9] = th_slat
        trace_header_array[10] = th_slong
        trace_header_array[11] = float(th_selev)
        trace_header_array[12] = float(th_swdepth)
        trace_header_array[13] = th_rlat
        trace_header_array[14] = th_rlong
        trace_header_array[15] = float(th_relev)
        trace_header_array[16] = th_sxutm
        trace_header_array[17] = th_syutm
        trace_header_array[18] = th_sz
        trace_header_array[19] = th_rxutm
        trace_header_array[20] = th_ryutm
        trace_header_array[21] = th_rz
        # 拾取占位符（后npick个值，写入0值，不写入实际拾取信息）
        # 注意：实际拾取信息已保存到头文件（.hdr），数据文件中只保留占位符以保持格式兼容
        trace_header_array[22:22 + fh_npick] = 0.0  # 写入0值占位符，不写入实际拾取信息
        
        fpout.write(trace_header_array.tobytes())
        
        # 写入道数据（确保是float32格式）
        trace_data_float32 = trace_data.astype(np.float32)
        
        # 输出前几道的振幅信息用于测试
        if test_amplitude > 0 and trace_idx < test_amplitude:
            amp_max = np.max(np.abs(trace_data_float32))
            amp_min = np.min(np.abs(trace_data_float32))
            amp_mean = np.mean(trace_data_float32)
            amp_std = np.std(trace_data_float32)
            amp_rms = np.sqrt(np.mean(trace_data_float32 ** 2))
            print(f"\n道 {trace_idx + 1} 振幅统计:")
            print(f"  最大绝对值: {amp_max:.6f}")
            print(f"  最小绝对值: {amp_min:.6f}")
            print(f"  均值: {amp_mean:.6f}")
            print(f"  标准差: {amp_std:.6f}")
            print(f"  RMS: {amp_rms:.6f}")
            print(f"  采样点数: {len(trace_data_float32)}")
            print(f"  记录号: {th_nrec}, 道序号: {th_itsn}, 接收站号: {th_ireci}, 类型: {th_itype}")
            # 输出前10个采样点的值
            print(f"  前10个采样点: {trace_data_float32[:10]}")
            
            # 检查振幅是否太小，给出显示建议
            if amp_max < 0.01:
                print(f"  ⚠️  警告: 振幅较小（最大绝对值: {amp_max:.6f}）")
                print(f"     建议在ZPLOT中使用自动缩放（iscale=0）或增大amp参数（建议amp={amp_max*1000:.1f}）")
            elif amp_max < 0.1:
                print(f"  ⚠️  提示: 振幅较小（最大绝对值: {amp_max:.6f}）")
                print(f"     建议在ZPLOT中使用自动缩放（iscale=0）或增大amp参数（建议amp={amp_max*100:.1f}）")
        
        fpout.write(trace_data_float32.tobytes())
        
        fh_ntraces += 1
        
        if verbose > 1:
            # 显示映射后的记录号
            print(f"\t{ival1:6d}{th_nrec:6d}{th_ireci:7d}{th_itsn:8d}{th_offset/1000.0:11.3f}{th_azi*60.0:8.3f}")
    
    # 更新文件头中的ntraces和nrec
    fpout.seek(0)
    fpout.write(struct.pack('<i', fh_ntraces))
    if nrec == 0:
        fpout.seek(5 * 4)  # nrec字段位置
        fpout.write(struct.pack('<i', nrec_actual))
    
    fpout.close()
    fphdr.close()
    
    # 检查整体振幅范围，给出显示建议
    if verbose and fh_ntraces > 0:
        # 重新读取数据检查振幅（仅当verbose且未测试振幅时）
        if test_amplitude == 0:
            # 读取前几道检查振幅
            sample_traces = min(5, fh_ntraces)
            max_amp_overall = 0.0
            with open(zdata_file, 'rb') as f:
                f.seek(52)  # 跳过文件头
                for i in range(sample_traces):
                    # 跳过道头
                    f.seek((22 + fh_npick) * 4, 1)
                    # 读取道数据
                    trace_bytes = f.read(fh_npts * 4)
                    if len(trace_bytes) == fh_npts * 4:
                        trace_data = np.frombuffer(trace_bytes, dtype=np.float32)
                        max_amp_overall = max(max_amp_overall, np.max(np.abs(trace_data)))
            
            if max_amp_overall > 0:
                if max_amp_overall < 0.01:
                    print(f"\n⚠️  警告: 数据振幅较小（最大绝对值: {max_amp_overall:.6f}）")
                    print(f"   建议在ZPLOT中使用自动缩放（iscale=0）或增大amp参数（建议amp={max_amp_overall*1000:.1f}）")
                elif max_amp_overall < 0.1:
                    print(f"\n⚠️  提示: 数据振幅较小（最大绝对值: {max_amp_overall:.6f}）")
                    print(f"   建议在ZPLOT中使用自动缩放（iscale=0）或增大amp参数（建议amp={max_amp_overall*100:.1f}）")
    
    if verbose:
        # 查找最后一个记录的原始key2值
        last_orig_key2 = None
        for orig_val, mapped_val in key2_to_recno.items():
            if mapped_val == recno:
                last_orig_key2 = orig_val
                break
        
        if last_orig_key2 is not None:
            print(f"\n\tTotal {current_rec_traces} traces for profile {last_ival1} record {recno} (原始key2={last_orig_key2})")
        else:
            print(f"\n\tTotal {current_rec_traces} traces for profile {last_ival1} record {recno}")
        
        print(f"\n {nrec_actual} records with totally {fh_ntraces} traces")
        
        # 输出offset范围统计
        if len(offset_values) > 0:
            offset_array = np.array(offset_values)
            offset_min = np.min(offset_array)
            offset_max = np.max(offset_array)
            negative_count = np.sum(offset_array < 0)
            positive_count = np.sum(offset_array >= 0)
            
            print(f"\n偏移距统计:")
            print(f"  范围: [{offset_min/1000.0:.3f}, {offset_max/1000.0:.3f}] km")
            print(f"  负偏移距: {negative_count} 道 ({negative_count/len(offset_array)*100:.1f}%)")
            print(f"  正偏移距: {positive_count} 道 ({positive_count/len(offset_array)*100:.1f}%)")
            
            if negative_count > 0:
                print(f"  负偏移距范围: [{np.min(offset_array[offset_array < 0])/1000.0:.3f}, {np.max(offset_array[offset_array < 0])/1000.0:.3f}] km")
                print(f"  ⚠️  提示: 数据包含负偏移距，建议在ZPLOT中设置 xmin < 0 以显示所有数据")
                print(f"     建议设置: xmin={offset_min/1000.0 - 1.0:.1f}, xmax={offset_max/1000.0 + 1.0:.1f}")
            elif offset_min >= 0:
                print(f"  ⚠️  注意: 所有偏移距均为正值，如果应该有负偏移距，请检查SU文件中的offset字段")
        
        print(f"\n转换完成:")
        print(f"  数据文件: {zdata_file}")
        print(f"  头文件: {hfile}")
        print(f"  总道数: {fh_ntraces}")
        print(f"  记录数: {nrec_actual}")
        print(f"  每道采样点数: {fh_npts}")
        print(f"  采样间隔: {fh_sint/1000000.0:.6f} 秒")
        print(f"  折合速度: {fh_vredf:.3f} km/s")
        if verbose > 0 and len(key2_to_recno) > 0:
            print(f"\n注意: 记录号已规范化（从1开始连续编号）")
            if len(key2_to_recno) <= 10:  # 如果记录数不多，显示映射关系
                print(f"  记录号映射: {dict(sorted(key2_to_recno.items()))}")
    
    return {
        'zdata_file': zdata_file,
        'hfile': hfile,
        'ntraces': fh_ntraces,
        'nrec': nrec_actual,
        'npts': fh_npts,
        'sint': fh_sint,
        'vredf': fh_vredf
    }


def _read_su_with_obspy(su_file: str) -> Tuple[List[np.ndarray], List[Dict]]:
    """使用ObsPy读取SU格式数据"""
    st = obspy_read(su_file, unpack_trace_headers=True)
    
    traces = []
    headers = []
    
    for trace_idx, tr in enumerate(st):
        traces.append(tr.data)
        # 提取SU头文件信息
        hdr = {}
        su_hdr = tr.stats.su.trace_header
        
        # 提取常用字段（使用ObsPy的SU头文件属性名）
        # ObsPy使用完整的下划线命名
        field_mapping = {
            'fldr': 'original_field_record_number',
            'cdp': 'ensemble_number',
            'tracf': 'trace_number_within_the_original_field_record',
            'trid': 'trace_identification_code',
            'offset': 'distance_from_center_of_the_source_point_to_the_center_of_the_receiver_group',
            'gelev': 'receiver_group_elevation',
            'selev': 'surface_elevation_at_source',
            'swdep': 'water_depth_at_source',
            'scalco': 'scalar_to_be_applied_to_all_coordinates',
            'scalel': 'scalar_to_be_applied_to_all_elevations_and_depths',
            'sx': 'source_coordinate_x',
            'sy': 'source_coordinate_y',
            'gx': 'group_coordinate_x',
            'gy': 'group_coordinate_y',
            'swevel': 'subweathering_velocity',
            'stas': 'sweep_trace_taper_length_at_start_in_ms',
            'stae': 'sweep_trace_taper_length_at_end_in_ms',
            'cdpt': 'cdp_ensemble_number',
        }
        
        for short_name, long_name in field_mapping.items():
            if hasattr(su_hdr, long_name):
                hdr[short_name] = getattr(su_hdr, long_name)
            elif hasattr(su_hdr, short_name):
                hdr[short_name] = getattr(su_hdr, short_name)
        
        # 如果没有找到某些字段，设置默认值
        hdr.setdefault('fldr', trace_idx + 1)
        hdr.setdefault('cdp', trace_idx + 1)
        hdr.setdefault('tracf', trace_idx + 1)
        hdr.setdefault('trid', 11)  # 默认hydrophone
        hdr.setdefault('offset', 0.0)
        hdr.setdefault('cdpt', 0.0)
        hdr.setdefault('scalco', 1)
        hdr.setdefault('sx', 0.0)
        hdr.setdefault('sy', 0.0)
        hdr.setdefault('gx', 0.0)
        hdr.setdefault('gy', 0.0)
        hdr.setdefault('selev', 0)
        hdr.setdefault('gelev', 0)
        hdr.setdefault('swdep', 0)
        hdr.setdefault('swevel', 6000)
        hdr.setdefault('stas', 0)
        hdr.setdefault('stae', 0)
        
        # dt字段：从stats.delta获取（秒），转换为微秒
        if 'dt' not in hdr and hasattr(tr.stats, 'delta'):
            hdr['dt'] = int(tr.stats.delta * 1e6)  # 秒转微秒
        elif 'dt' not in hdr:
            # 如果没有delta，尝试从头文件读取
            if hasattr(su_hdr, 'sample_interval_in_ms_for_this_trace'):
                hdr['dt'] = int(getattr(su_hdr, 'sample_interval_in_ms_for_this_trace') * 1000)  # 毫秒转微秒
            else:
                hdr['dt'] = 4000  # 默认4ms
        
        headers.append(hdr)
    
    return traces, headers


def _read_su_with_supython(su_file: str) -> Tuple[List[np.ndarray], List]:
    """使用supython读取SU格式数据"""
    amp, hdr = readsu(su_file, scale=1)
    
    # 转置：supython返回的是 (ns, nx)，我们需要 (nx, ns)
    traces = [amp[:, i] for i in range(amp.shape[1])]
    
    # 创建头文件列表
    headers = []
    for i in range(len(traces)):
        headers.append(hdr)
    
    return traces, headers


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SU格式转Z格式转换工具 (su2z_hhb)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本转换
  python su2z_hhb.py input.su
  
  # 指定输出文件
  python su2z_hhb.py input.su --zdata output.z --hfile output.hdr
  
  # 自定义参数
  python su2z_hhb.py input.su --npick 10 --vred 6.0 --verbose 1
        """
    )
    
    parser.add_argument('su_file', help='输入的SU格式文件')
    parser.add_argument('--zdata', default='data.z', help='输出的Z格式数据文件（默认: data.z）')
    parser.add_argument('--hfile', default='file.hdr', help='输出的头文件（默认: file.hdr）')
    parser.add_argument('--key1', default='fldr', help='第一个关键字段（默认: fldr）')
    parser.add_argument('--key2', default='cdp', help='第二个关键字段（默认: cdp，如果cdp未赋值则所有道归为记录号1）')
    parser.add_argument('--key3', default='tracf', help='第三个关键字段（默认: tracf）')
    parser.add_argument('--key4', default='trid', help='第四个关键字段（默认: trid）')
    parser.add_argument('--nrec', type=int, default=0, help='记录数（0=自动计算）')
    parser.add_argument('--npick', type=int, default=40, help='拾取字数（默认: 40，最大40，与C代码一致）')
    parser.add_argument('--vred', type=float, default=None, help='折合速度（km/s，默认从SU头文件读取）')
    parser.add_argument('--tstart', type=int, default=None, help='起始时间（毫秒，默认从SU头文件读取）')
    parser.add_argument('--tend', type=int, default=None, help='结束时间（毫秒，默认从SU头文件读取）')
    parser.add_argument('--verbose', type=int, default=0, choices=[0, 1, 2],
                       help='详细输出级别（0=静默，1=基本信息，2=详细信息）')
    parser.add_argument('--test-amplitude', type=int, default=0, dest='test_amplitude',
                       help='输出前N道的振幅信息用于测试（默认: 0=不输出）')
    
    args = parser.parse_args()
    
    try:
        result = su2z_hhb(
            su_file=args.su_file,
            zdata_file=args.zdata,
            hfile=args.hfile,
            key1=args.key1,
            key2=args.key2,
            key3=args.key3,
            key4=args.key4,
            nrec=args.nrec,
            npick=args.npick,
            vred=args.vred,
            tstart=args.tstart,
            tend=args.tend,
            verbose=args.verbose,
            test_amplitude=args.test_amplitude
        )
        print("\n转换成功完成！")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n转换失败: {e}")
        exit(1)
