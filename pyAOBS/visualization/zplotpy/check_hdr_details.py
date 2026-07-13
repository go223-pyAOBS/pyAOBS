"""
详细检查头文件中的原始值
"""

import os
import sys
import struct
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader

def check_hdr_details():
    """详细检查头文件内容"""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    zfile = os.path.join(base_dir, 'test', 'Rotated_OBS08_4C.z')
    hfile = os.path.join(base_dir, 'test', 'Rotated_OBS08_4C.hdr')
    
    print("=" * 80)
    print("详细检查头文件内容")
    print("=" * 80)
    
    # 先读取.z文件获取npick
    loader = DataLoader()
    data = loader.load_z_format(dfile=zfile)
    header = data['header']
    npick = header.npick
    
    print(f"\n从.z文件读取的npick: {npick}")
    print(f"\n开始读取头文件: {hfile}")
    print("-" * 80)
    
    # 直接读取头文件，显示原始值
    trace_headers_raw = []
    with open(hfile, 'rb') as f:
        record_count = 0
        while True:
            # 读取记录长度标记
            rec_len_bytes = f.read(4)
            if len(rec_len_bytes) < 4:
                break
            
            rec_len = struct.unpack('<i', rec_len_bytes)[0]
            
            # 读取整数字段（6个）
            ints_bytes = f.read(24)
            if len(ints_bytes) < 24:
                break
            ishoti, itsn, ireci, itypei, iflagi, igaini = struct.unpack('<6i', ints_bytes)
            
            # 读取浮点数字段（16个）- 这里包含offsti（原始值，单位是米）
            floats_bytes = f.read(64)
            if len(floats_bytes) < 64:
                break
            floats = struct.unpack('<16f', floats_bytes)
            (offsti_raw, azi_raw, texact, slat, slong, selev, swdepth, rlat, rlong, relev,
             sxutm, syutm, sz, rxutm, ryutm, rz) = floats
            
            # 读取拾取数组
            picks_bytes = f.read(npick * 4)
            if len(picks_bytes) < npick * 4:
                break
            picks = list(struct.unpack(f'<{npick}f', picks_bytes))
            
            # 读取记录长度标记（结束）
            rec_len2_bytes = f.read(4)
            if len(rec_len2_bytes) < 4:
                break
            rec_len2 = struct.unpack('<i', rec_len2_bytes)[0]
            
            trace_headers_raw.append({
                'record': record_count + 1,
                'ishoti': ishoti,
                'itsn': itsn,
                'ireci': ireci,
                'offsti_raw': offsti_raw,  # 原始值（米）
                'offsti_km': offsti_raw / 1000.0,  # 转换为千米
                'azi_raw': azi_raw,  # 原始值（分）
                'azi_deg': azi_raw / 60.0,  # 转换为度
                'picks': picks
            })
            
            record_count += 1
            
            # 只显示前10条和最后10条
            if record_count <= 10 or record_count > len(trace_headers_raw) - 10:
                if record_count <= 10:
                    print(f"\n记录 {record_count}:")
                    print(f"  炮站号: {ishoti}, 道序号: {itsn}, 接收站号: {ireci}")
                    print(f"  炮检距(原始值，米): {offsti_raw:.3f}")
                    print(f"  炮检距(千米): {offsti_raw/1000.0:.6f}")
                    print(f"  方位角(原始值，分): {azi_raw:.3f}")
                    print(f"  方位角(度): {azi_raw/60.0:.6f}")
                    if any(p != 0.0 for p in picks[:5]):
                        print(f"  拾取数组前5个值: {picks[:5]}")
    
    print("\n" + "=" * 80)
    print(f"总共读取了 {len(trace_headers_raw)} 条记录")
    print("=" * 80)
    
    # 统计炮检距信息
    offsti_raw_list = [th['offsti_raw'] for th in trace_headers_raw]
    offsti_km_list = [th['offsti_km'] for th in trace_headers_raw]
    
    print(f"\n炮检距统计:")
    print(f"  原始值(米) - 最小值: {min(offsti_raw_list):.3f}, 最大值: {max(offsti_raw_list):.3f}")
    print(f"  转换值(千米) - 最小值: {min(offsti_km_list):.6f}, 最大值: {max(offsti_km_list):.6f}")
    print(f"  非零值数量: {sum(1 for x in offsti_raw_list if x != 0.0)}")
    
    # 显示最后10条记录
    print(f"\n最后10条记录:")
    for th in trace_headers_raw[-10:]:
        print(f"  记录 {th['record']}: 炮检距={th['offsti_km']:.6f} km, 方位角={th['azi_deg']:.6f} 度")
    
    return trace_headers_raw

if __name__ == '__main__':
    check_hdr_details()
