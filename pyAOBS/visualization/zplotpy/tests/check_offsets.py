"""
check_offsets.py - 检查.z和.hdr文件中的offsti值是否一致
"""

import struct
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader

def check_offsets(zfile, hfile=None):
    """检查.z和.hdr文件中的offsti值"""
    print(f"检查文件: {zfile}")
    if hfile:
        print(f"头文件: {hfile}")
    print("=" * 60)
    
    # 读取.z文件中的道头信息
    print("\n1. 从.z文件读取道头信息:")
    print("-" * 60)
    
    with open(zfile, 'rb') as f:
        # 读取文件头
        header_bytes = f.read(52)
        ints = struct.unpack('<7i', header_bytes[:28])
        ntraces, npts, sint, tstart, tend, nrec, npick = ints
        vredf, = struct.unpack('<f', header_bytes[28:32])
        ifmt, = struct.unpack('<i', header_bytes[32:36])
        
        print(f"  道数: {ntraces}")
        print(f"  采样点数: {npts}")
        print(f"  拾取字数: {npick}")
        print(f"  数据格式: {ifmt} (1=float32, 0=int16)")
        
        # 计算道头大小
        trace_header_size = (22 + npick) * 4
        if ifmt == 1:
            trace_data_size = npts * 4
        else:
            trace_data_size = npts * 2
        record_size = trace_header_size + trace_data_size
        
        # 读取每条道的道头
        z_offsets = []
        f.seek(52)  # 跳过文件头
        for i in range(ntraces):
            # 读取道头
            trace_header_bytes = f.read(trace_header_size)
            if len(trace_header_bytes) < trace_header_size:
                break
            
            # 解析道头（22 + npick 个浮点数）
            trace_header = struct.unpack(f'<{22 + npick}f', trace_header_bytes)
            offsti_m = trace_header[5]  # 第6个值（索引5）是offsti（米）
            z_offsets.append(offsti_m)
            
            # 跳过道数据
            f.read(trace_data_size)
        
        print(f"\n  .z文件中的offsti值（米）:")
        print(f"    前5道: {[f'{x:.2f}' for x in z_offsets[:5]]}")
        print(f"    后5道: {[f'{x:.2f}' for x in z_offsets[-5:]]}")
        print(f"    范围: [{min(z_offsets):.2f}, {max(z_offsets):.2f}] 米")
        print(f"    范围: [{min(z_offsets)/1000:.2f}, {max(z_offsets)/1000:.2f}] 千米")
    
    # 读取.hdr文件中的道头信息
    if hfile and os.path.exists(hfile):
        print("\n2. 从.hdr文件读取道头信息:")
        print("-" * 60)
        
        with open(hfile, 'rb') as f:
            h_offsets = []
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
                
                # 读取浮点数字段（16个）
                floats_bytes = f.read(64)
                if len(floats_bytes) < 64:
                    break
                floats = struct.unpack('<16f', floats_bytes)
                offsti_m = floats[0]  # 第一个浮点数是offsti（米）
                h_offsets.append(offsti_m)
                
                # 读取拾取数组
                picks_bytes = f.read(npick * 4)
                if len(picks_bytes) < npick * 4:
                    break
                
                # 读取记录长度标记（结束）
                rec_len2_bytes = f.read(4)
                if len(rec_len2_bytes) < 4:
                    break
                
                record_count += 1
            
            print(f"  道数: {record_count}")
            print(f"\n  .hdr文件中的offsti值（米）:")
            print(f"    前5道: {[f'{x:.2f}' for x in h_offsets[:5]]}")
            if len(h_offsets) > 5:
                print(f"    后5道: {[f'{x:.2f}' for x in h_offsets[-5:]]}")
            print(f"    范围: [{min(h_offsets):.2f}, {max(h_offsets):.2f}] 米")
            print(f"    范围: [{min(h_offsets)/1000:.2f}, {max(h_offsets)/1000:.2f}] 千米")
            
            # 比较
            print("\n3. 比较.z和.hdr文件中的offsti值:")
            print("-" * 60)
            
            min_len = min(len(z_offsets), len(h_offsets))
            if min_len > 0:
                matches = 0
                mismatches = []
                for i in range(min_len):
                    if abs(z_offsets[i] - h_offsets[i]) < 0.01:  # 允许0.01米的误差
                        matches += 1
                    else:
                        mismatches.append((i, z_offsets[i], h_offsets[i]))
                
                print(f"  匹配的道数: {matches}/{min_len}")
                if mismatches:
                    print(f"  不匹配的道:")
                    for idx, z_val, h_val in mismatches[:10]:  # 只显示前10个
                        print(f"    道{idx+1}: .z={z_val:.2f}米, .hdr={h_val:.2f}米, 差值={abs(z_val-h_val):.2f}米")
                else:
                    print(f"  ✓ 所有道的offsti值都匹配！")
            else:
                print(f"  警告: 道数不匹配！.z文件有{len(z_offsets)}道，.hdr文件有{len(h_offsets)}道")
    
    # 使用DataLoader检查
    print("\n4. 使用DataLoader检查:")
    print("-" * 60)
    
    loader = DataLoader()
    if hfile and os.path.exists(hfile):
        data = loader.load_z_format(dfile=zfile, hfile=hfile)
    else:
        data = loader.load_z_format(dfile=zfile)
    
    offsets = data['offsets']
    print(f"  DataLoader提取的offsets值（千米）:")
    print(f"    前5道: {[f'{x:.2f}' for x in offsets[:5]]}")
    print(f"    后5道: {[f'{x:.2f}' for x in offsets[-5:]]}")
    print(f"    范围: [{min(offsets):.2f}, {max(offsets):.2f}] 千米")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='检查.z和.hdr文件中的offsti值')
    parser.add_argument('zfile', help='.z文件路径')
    parser.add_argument('hfile', nargs='?', help='.hdr文件路径（可选）')
    
    args = parser.parse_args()
    
    check_offsets(args.zfile, args.hfile)
