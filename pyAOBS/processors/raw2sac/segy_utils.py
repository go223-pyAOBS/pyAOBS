#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SEGY文件写入工具模块

提供SEGY格式文件的创建和写入功能：
1. EBCDIC头写入
2. 二进制头创建和写入
3. 道头创建和写入
4. 完整的SEGY文件写入（包含IEEE到IBM浮点转换）

便于后续整合到 pyAOBS 代码包中。

Author: Haibo Huang, 2025/01/XX
"""

import struct
import numpy as np
from typing import Dict, List

# 从 format_utils 导入浮点转换函数
try:
    from format_utils import ieee_to_ibm_float_vectorized
except ImportError:
    print("Error: Cannot import ieee_to_ibm_float_vectorized from format_utils")
    raise


# ============================================================================
# SEGY头写入函数
# ============================================================================

def write_ebcdic_header(f) -> None:
    """写入EBCDIC头（3200字节）
    
    SEGY标准要求3200字节的EBCDIC文本头
    这里写入空白字符（可以后续自定义）
    
    Args:
        f: 文件对象（二进制写入模式）
    """
    # 写入3200字节的空白字符（EBCDIC空格字符是0x40）
    f.write(b'\x40' * 3200)


def create_segy_binary_header(delta: float, trace_length: float,
                              n_traces: int = 0) -> Dict:
    """创建SEGY二进制头
    
    对应C代码的bhed结构
    
    Args:
        delta: 采样间隔（秒）
        trace_length: 道长度（秒）
        n_traces: 道数（稍后更新）
    
    Returns:
        二进制头字典
    """
    # 计算采样间隔（微秒）
    hdt = int(1000000 * delta + 0.5)
    
    # 计算每道的采样点数（对应C代码：bh.hns = (int)(((double)length)/sp.delta+0.5)）
    hns = int((trace_length / delta) + 0.5)
    
    binary_header = {
        'jobid': 0,
        'lino': 0,
        'reno': 0,
        'ntrpr': n_traces,  # 稍后更新
        'nart': 0,
        'hdt': hdt,  # 采样间隔（微秒）
        'dto': hdt,  # 原始采样间隔
        'hns': hns,  # 每道的采样点数
        'nso': hns,  # 原始每道的采样点数
        'format': 1,  # IBM浮点格式
        'fold': n_traces if n_traces > 0 else 1,  # CDP fold
        'tsort': 1,  # 按记录顺序
        'vscode': 1,  # 无垂直叠加
        'hsfs': 0,
        'hsfe': 0,
        'hslen': 0,
        'hstyp': 0,
        'schn': 0,
        'hstas': 0,
        'hstae': 0,
        'htatyp': 0,
        'hcorr': 1,  # 未相关
        'bgrcv': 2,  # 未恢复
        'rcvm': 1,  # 无振幅恢复
        'mfeet': 1,  # 米
        'polyt': 0,
        'vpol': 0
    }
    
    return binary_header


def write_binary_header(f, binary_header: Dict) -> None:
    """写入SEGY二进制头（400字节，big-endian）
    
    Args:
        f: 文件对象（二进制写入模式）
        binary_header: 二进制头字典
    """
    # 创建400字节缓冲区
    header_bytes = bytearray(400)
    
    # 按SEGY标准格式写入各个字段（big-endian）
    # 字段位置参考SEGY标准
    struct.pack_into('>i', header_bytes, 0, binary_header['jobid'])      # 0-3
    struct.pack_into('>i', header_bytes, 4, binary_header['lino'])       # 4-7
    struct.pack_into('>i', header_bytes, 8, binary_header['reno'])       # 8-11
    struct.pack_into('>h', header_bytes, 12, binary_header['ntrpr'])     # 12-13
    struct.pack_into('>h', header_bytes, 14, binary_header['nart'])      # 14-15
    struct.pack_into('>H', header_bytes, 16, binary_header['hdt'])       # 16-17
    struct.pack_into('>H', header_bytes, 18, binary_header['dto'])       # 18-19
    struct.pack_into('>H', header_bytes, 20, binary_header['hns'])       # 20-21
    struct.pack_into('>H', header_bytes, 22, binary_header['nso'])       # 22-23
    struct.pack_into('>h', header_bytes, 24, binary_header['format'])    # 24-25
    struct.pack_into('>h', header_bytes, 26, binary_header['fold'])      # 26-27
    struct.pack_into('>h', header_bytes, 28, binary_header['tsort'])     # 28-29
    struct.pack_into('>h', header_bytes, 30, binary_header['vscode'])    # 30-31
    # ... 继续写入其他字段（简化版，只写入关键字段）
    
    # 写入文件
    f.write(header_bytes)


def create_segy_trace_header(shot_info: Dict, sx: float, sy: float,
                            gx: float, gy: float, offset: float,
                            gelev: float, swdep: int, delrt: int,
                            ns: int, dt: int, tracl: int, tracr: int,
                            year: int, day: int, hour: int, minute: int,
                            sec: int, timbas: int) -> Dict:
    """创建SEGY道头
    
    对应C代码的segy结构
    
    Args:
        shot_info: 炮点信息字典
        sx, sy: 炮点坐标（UTM）
        gx, gy: 接收点坐标（UTM）
        offset: 偏移距（米）
        gelev: 接收点高程
        swdep: 炮点水深
        delrt: 延迟记录时间（毫秒）
        ns: 采样点数
        dt: 采样间隔（微秒）
        tracl: 道序号（线内）
        tracr: 道序号（文件内）
        year, day, hour, minute, sec, timbas: 时间信息
    
    Returns:
        道头字典
    """
    trace_header = {
        'tracl': tracl,                    # 0-3: 道序号（线内）
        'tracr': tracr,                    # 4-7: 道序号（文件内）
        'fldr': shot_info['field_record'], # 8-11: 野外记录号
        'tracf': 1,                        # 12-15: 道号（记录内）
        'ep': shot_info['shot_number'],    # 16-19: 震源点号
        'cdp': shot_info['shot_number'],   # 20-23: CDP号
        'cdpt': 1,                         # 24-27: CDP内道号
        'trid': 1,                         # 28-29: 道标识（1=地震数据）
        'nvs': 1,                          # 30-31: 垂直叠加数
        'nhs': 1,                          # 32-33: 水平叠加数
        'duse': 1,                         # 34-35: 数据用途（1=生产）
        'offset': int(np.clip(offset if np.isfinite(offset) else 0, -2e9, 2e9) + 0.5),  # 36-39: 偏移距
        'gelev': int(gelev if np.isfinite(gelev) else 0),               # 40-43: 接收点高程
        'selev': 0,                        # 44-47: 震源高程
        'sdepth': 0,                       # 48-51: 震源深度
        'gdel': 0,                         # 52-55: 接收点基准面
        'sdel': 0,                         # 56-59: 震源基准面
        'swdep': int(swdep if np.isfinite(swdep) else 0),                    # 60-63: 震源水深
        'gwdep': 0,                        # 64-67: 接收点水深
        'scalel': 0,                       # 68-69: 高程比例因子
        'scalco': -1,                      # 70-71: 坐标比例因子
        'sx': int(np.clip(sx if np.isfinite(sx) else 0, -2e9, 2e9) + 0.5),  # 72-75: 震源X坐标
        'sy': int(np.clip(sy if np.isfinite(sy) else 0, -2e9, 2e9) + 0.5),  # 76-79: 震源Y坐标
        'gx': int(np.clip(gx if np.isfinite(gx) else 0, -2e9, 2e9) + 0.5),  # 80-83: 接收点X坐标
        'gy': int(np.clip(gy if np.isfinite(gy) else 0, -2e9, 2e9) + 0.5),  # 84-87: 接收点Y坐标
        'counit': 1,                       # 88-89: 坐标单位（1=米）
        'wevel': 0,                        # 90-91: 风化层速度
        'swevel': 0,                       # 92-93: 次风化层速度
        'sut': 0,                          # 94-95: 震源井口时间
        'gut': 0,                          # 96-97: 接收点井口时间
        'sstat': 0,                        # 98-99: 震源静校正
        'gstat': 0,                        # 100-101: 接收点静校正
        'tstat': 0,                        # 102-103: 总静校正
        'laga': 0,                         # 104-105: 延迟A
        'lagb': 0,                         # 106-107: 延迟B
        'delrt': delrt,                    # 108-109: 延迟记录时间
        'muts': 0,                         # 110-111: 静音开始
        'mute': 0,                         # 112-113: 静音结束
        'ns': ns,                          # 114-115: 采样点数
        'dt': dt,                          # 116-117: 采样间隔（微秒）
        'gain': 0,                         # 118-119: 增益类型
        'igc': 0,                          # 120-121: 增益常数
        'igi': 0,                          # 122-123: 初始增益
        'corr': 1,                         # 124-125: 相关标志
        'sfs': 0,                          # 126-127: 扫描起始频率
        'sfe': 0,                          # 128-129: 扫描结束频率
        'slen': 0,                         # 130-131: 扫描长度
        'styp': 0,                         # 132-133: 扫描类型
        'stas': 0,                         # 134-135: 扫描起始时间
        'stae': 0,                         # 136-137: 扫描结束时间
        'tatyp': 0,                        # 138-139: 锥形类型
        'afilf': 0,                        # 140-141: 抗混叠频率
        'afils': 0,                        # 142-143: 抗混叠斜率
        'nofilf': 0,                       # 144-145: 陷波频率
        'nofils': 0,                       # 146-147: 陷波斜率
        'lcf': 0,                          # 148-149: 低截频率
        'hcf': 0,                          # 150-151: 高截频率
        'lcs': 0,                          # 152-153: 低截斜率
        'hcs': 0,                          # 154-155: 高截斜率
        'year': year,                      # 156-157: 年
        'day': day,                        # 158-159: 儒略日
        'hour': hour,                      # 160-161: 时
        'minute': minute,                  # 162-163: 分
        'sec': sec,                        # 164-165: 秒
        'timbas': timbas,                  # 166-167: 时间基准（毫秒）
        'trwf': 0,                         # 168-169: 道加权因子
        'grnors': 0,                       # 170-171: 检波器组号
        'grnofr': 0,                       # 172-173: 原始记录道号
        'grnlof': 0,                       # 174-175: 原始记录最后道号
        'gaps': 0,                         # 176-177: 间隙
        'otrav': 0                         # 178-179: 过行程锥形
    }
    
    return trace_header


def write_trace_header(f, trace_header: Dict) -> None:
    """写入SEGY道头（240字节，big-endian）
    
    Args:
        f: 文件对象（二进制写入模式）
        trace_header: 道头字典
    """
    # 创建240字节缓冲区
    header_bytes = bytearray(240)
    
    # 按SEGY标准格式写入各个字段（big-endian）
    struct.pack_into('>i', header_bytes, 0, trace_header['tracl'])       # 0-3
    struct.pack_into('>i', header_bytes, 4, trace_header['tracr'])       # 4-7
    struct.pack_into('>i', header_bytes, 8, trace_header['fldr'])        # 8-11
    struct.pack_into('>i', header_bytes, 12, trace_header['tracf'])      # 12-15
    struct.pack_into('>i', header_bytes, 16, trace_header['ep'])         # 16-19
    struct.pack_into('>i', header_bytes, 20, trace_header['cdp'])        # 20-23
    struct.pack_into('>i', header_bytes, 24, trace_header['cdpt'])       # 24-27
    struct.pack_into('>h', header_bytes, 28, trace_header['trid'])       # 28-29
    struct.pack_into('>h', header_bytes, 30, trace_header['nvs'])        # 30-31
    struct.pack_into('>h', header_bytes, 32, trace_header['nhs'])        # 32-33
    struct.pack_into('>h', header_bytes, 34, trace_header['duse'])       # 34-35
    struct.pack_into('>i', header_bytes, 36, trace_header['offset'])     # 36-39
    struct.pack_into('>i', header_bytes, 40, trace_header['gelev'])      # 40-43
    struct.pack_into('>i', header_bytes, 44, trace_header['selev'])      # 44-47
    struct.pack_into('>i', header_bytes, 48, trace_header['sdepth'])     # 48-51
    struct.pack_into('>i', header_bytes, 52, trace_header['gdel'])       # 52-55
    struct.pack_into('>i', header_bytes, 56, trace_header['sdel'])       # 56-59
    struct.pack_into('>i', header_bytes, 60, trace_header['swdep'])      # 60-63
    struct.pack_into('>i', header_bytes, 64, trace_header['gwdep'])      # 64-67
    struct.pack_into('>h', header_bytes, 68, trace_header['scalel'])     # 68-69
    struct.pack_into('>h', header_bytes, 70, trace_header['scalco'])     # 70-71
    struct.pack_into('>i', header_bytes, 72, trace_header['sx'])         # 72-75
    struct.pack_into('>i', header_bytes, 76, trace_header['sy'])         # 76-79
    struct.pack_into('>i', header_bytes, 80, trace_header['gx'])         # 80-83
    struct.pack_into('>i', header_bytes, 84, trace_header['gy'])         # 84-87
    struct.pack_into('>h', header_bytes, 88, trace_header['counit'])     # 88-89
    struct.pack_into('>h', header_bytes, 90, trace_header['wevel'])      # 90-91
    struct.pack_into('>h', header_bytes, 92, trace_header['swevel'])     # 92-93
    struct.pack_into('>h', header_bytes, 94, trace_header['sut'])        # 94-95
    struct.pack_into('>h', header_bytes, 96, trace_header['gut'])        # 96-97
    struct.pack_into('>h', header_bytes, 98, trace_header['sstat'])      # 98-99
    struct.pack_into('>h', header_bytes, 100, trace_header['gstat'])     # 100-101
    struct.pack_into('>h', header_bytes, 102, trace_header['tstat'])     # 102-103
    struct.pack_into('>h', header_bytes, 104, trace_header['laga'])      # 104-105
    struct.pack_into('>h', header_bytes, 106, trace_header['lagb'])      # 106-107
    struct.pack_into('>h', header_bytes, 108, trace_header['delrt'])     # 108-109
    struct.pack_into('>h', header_bytes, 110, trace_header['muts'])      # 110-111
    struct.pack_into('>h', header_bytes, 112, trace_header['mute'])      # 112-113
    struct.pack_into('>H', header_bytes, 114, trace_header['ns'])        # 114-115
    struct.pack_into('>H', header_bytes, 116, trace_header['dt'])        # 116-117
    struct.pack_into('>h', header_bytes, 118, trace_header['gain'])      # 118-119
    struct.pack_into('>h', header_bytes, 120, trace_header['igc'])       # 120-121
    struct.pack_into('>h', header_bytes, 122, trace_header['igi'])       # 122-123
    struct.pack_into('>h', header_bytes, 124, trace_header['corr'])      # 124-125
    struct.pack_into('>h', header_bytes, 126, trace_header['sfs'])       # 126-127
    struct.pack_into('>h', header_bytes, 128, trace_header['sfe'])       # 128-129
    struct.pack_into('>h', header_bytes, 130, trace_header['slen'])      # 130-131
    struct.pack_into('>h', header_bytes, 132, trace_header['styp'])      # 132-133
    struct.pack_into('>h', header_bytes, 134, trace_header['stas'])      # 134-135
    struct.pack_into('>h', header_bytes, 136, trace_header['stae'])      # 136-137
    struct.pack_into('>h', header_bytes, 138, trace_header['tatyp'])     # 138-139
    struct.pack_into('>h', header_bytes, 140, trace_header['afilf'])     # 140-141
    struct.pack_into('>h', header_bytes, 142, trace_header['afils'])     # 142-143
    struct.pack_into('>h', header_bytes, 144, trace_header['nofilf'])    # 144-145
    struct.pack_into('>h', header_bytes, 146, trace_header['nofils'])    # 146-147
    struct.pack_into('>h', header_bytes, 148, trace_header['lcf'])       # 148-149
    struct.pack_into('>h', header_bytes, 150, trace_header['hcf'])       # 150-151
    struct.pack_into('>h', header_bytes, 152, trace_header['lcs'])       # 152-153
    struct.pack_into('>h', header_bytes, 154, trace_header['hcs'])       # 154-155
    struct.pack_into('>h', header_bytes, 156, trace_header['year'])      # 156-157
    struct.pack_into('>h', header_bytes, 158, trace_header['day'])       # 158-159
    struct.pack_into('>h', header_bytes, 160, trace_header['hour'])      # 160-161
    struct.pack_into('>h', header_bytes, 162, trace_header['minute'])    # 162-163
    struct.pack_into('>h', header_bytes, 164, trace_header['sec'])       # 164-165
    struct.pack_into('>h', header_bytes, 166, trace_header['timbas'])    # 166-167
    struct.pack_into('>h', header_bytes, 168, trace_header['trwf'])      # 168-169
    struct.pack_into('>h', header_bytes, 170, trace_header['grnors'])    # 170-171
    struct.pack_into('>h', header_bytes, 172, trace_header['grnofr'])    # 172-173
    struct.pack_into('>h', header_bytes, 174, trace_header['grnlof'])    # 174-175
    struct.pack_into('>h', header_bytes, 176, trace_header['gaps'])      # 176-177
    struct.pack_into('>h', header_bytes, 178, trace_header['otrav'])     # 178-179
    # 剩余字节填充0
    struct.pack_into('>h', header_bytes, 180, 0)                         # 180-181
    # ... 继续填充到240字节
    
    # 写入文件
    f.write(header_bytes)


def write_segy_file(output_file: str, traces_data: List[np.ndarray],
                   traces_headers: List[Dict], binary_header: Dict,
                   verbose: bool = False) -> None:
    """写入SEGY文件（优化版本）
    
    Args:
        output_file: 输出SEGY文件路径
        traces_data: 道数据列表（IEEE浮点）
        traces_headers: 道头列表
        binary_header: 二进制头字典
        verbose: 是否输出详细信息
    """
    # 预分配缓冲区大小
    BUFFER_SIZE = 4 * 1024 * 1024  # 4MB
    
    with open(output_file, 'wb', buffering=BUFFER_SIZE) as f:
        # 1. 写入EBCDIC头（3200字节）
        write_ebcdic_header(f)
        
        # 2. 预留二进制头位置
        binary_header_pos = f.tell()
        f.write(b'\x00' * 400)
        
        # 3. 预转换所有数据为IBM格式（批量转换）
        if verbose:
            print("    Converting data to IBM format...")
        ibm_traces = []
        for data in traces_data:
            ibm_data = ieee_to_ibm_float_vectorized(data)
            ibm_traces.append(ibm_data)
        
        if verbose:
            print(f"    Converted {len(ibm_traces)} traces to IBM format")
        
        # 4. 写入道头和数据
        if verbose:
            print(f"    Writing {len(traces_data)} traces...")
        
        for ibm_data, header in zip(ibm_traces, traces_headers):
            # 写入道头（240字节）
            write_trace_header(f, header)
            
            # 写入数据（big-endian IBM浮点）
            # 关键理解：
            # 1. ibm_data中的每个uint32值已经是big-endian格式（在ieee_to_ibm_float_vectorized中完成字节序转换）
            # 2. 在little-endian系统上，这些值在NumPy数组中的内存表示是little-endian的
            # 3. 我们需要写入big-endian格式的字节序列
            # 4. C代码逻辑：float_to_ibm在endian==0时做了字节序转换，转换后的值在内存中是little-endian的uint32
            #    但值本身已经是big-endian格式，fwrite直接写入内存字节，得到的字节序列是big-endian的
            # 5. Python实现：ibm_data中的值已经是big-endian格式，但内存表示是little-endian
            #    我们需要先将值的字节序反转（big-endian值 -> little-endian值），然后按big-endian写入
            #    这样写入的字节序列才是正确的big-endian格式
            
            # 向量化方法：先反转字节序，然后打包写入
            # 先将big-endian值转换回little-endian值（字节序反转）
            ibm_data_le = ((ibm_data << 24) & 0xFF000000) | \
                         ((ibm_data >> 24) & 0x000000FF) | \
                         ((ibm_data & 0x0000FF00) << 8) | \
                         ((ibm_data & 0x00FF0000) >> 8)
            # 然后按big-endian打包写入（向量化）
            f.write(struct.pack('>{}I'.format(len(ibm_data_le)), *ibm_data_le))
        
        # 5. 更新二进制头
        binary_header['ntrpr'] = len(traces_data)
        binary_header['fold'] = len(traces_data)
        f.seek(binary_header_pos)
        write_binary_header(f, binary_header)
    
    print(f"SEGY file written: {output_file}")
    print(f"Number of traces: {len(traces_data)}")

