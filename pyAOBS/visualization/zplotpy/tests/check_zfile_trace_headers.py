"""
检查.z文件中的道头信息，看是否包含炮检距
"""

import os
import sys
import struct
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader

def check_zfile_trace_headers():
    """检查.z文件中的道头信息"""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    zfile = os.path.join(base_dir, 'test', 'Rotated_OBS08_4C.z')
    hfile = os.path.join(base_dir, 'test', 'Rotated_OBS08_4C.hdr')
    
    print("=" * 80)
    print("检查.z文件中的道头信息")
    print("=" * 80)
    
    # 读取文件头
    loader = DataLoader()
    data = loader.load_z_format(dfile=zfile, hfile=hfile)
    header = data['header']
    
    print(f"\n文件头信息:")
    print(f"  总道数: {header.ntraces}")
    print(f"  每道采样点数: {header.npts}")
    print(f"  拾取字数: {header.npick}")
    print(f"  数据格式: {header.ifmt} ({'float32' if header.ifmt == 1 else 'int16'})")
    
    # 计算道头大小
    trace_header_size = (22 + header.npick) * 4
    if header.ifmt == 1:
        trace_data_size = header.npts * 4
    else:
        trace_data_size = header.npts * 2
    record_size = trace_header_size + trace_data_size
    
    print(f"\n道头大小: {trace_header_size} 字节 (22 + {header.npick} = {22 + header.npick} 个浮点数)")
    print(f"道数据大小: {trace_data_size} 字节")
    print(f"记录总大小: {record_size} 字节")
    
    # 读取.z文件中的道头
    print("\n" + "-" * 80)
    print("读取.z文件中的道头（前5道）:")
    print("-" * 80)
    
    with open(zfile, 'rb') as f:
        # 跳过文件头
        f.seek(52)
        
        for i in range(min(5, header.ntraces)):
            # 读取道头部分
            trace_header_bytes = f.read(trace_header_size)
            if len(trace_header_bytes) < trace_header_size:
                break
            
            # 解析道头（22 + npick 个浮点数）
            trace_header_values = struct.unpack(f'<{22 + header.npick}f', trace_header_bytes)
            
            print(f"\n道 {i+1} 的道头值（前22个字段）:")
            for j in range(22):
                print(f"  字段 {j+1:2d}: {trace_header_values[j]:15.6f}")
            
            # 跳过道数据
            f.seek(trace_data_size, 1)
    
    # 检查头文件中的UTM坐标，看是否可以计算炮检距
    print("\n" + "=" * 80)
    print("检查头文件中的UTM坐标，尝试计算炮检距")
    print("=" * 80)
    
    trace_headers = data['trace_headers']
    
    if trace_headers:
        print(f"\n前10道的UTM坐标和计算的炮检距:")
        print("-" * 80)
        
        for i, th in enumerate(trace_headers[:10]):
            # 计算炮检距：sqrt((rxutm - sxutm)^2 + (ryutm - syutm)^2)
            dx = th.rxutm - th.sxutm
            dy = th.ryutm - th.syutm
            offset_calculated = np.sqrt(dx*dx + dy*dy) / 1000.0  # 转换为千米
            
            print(f"\n道 {i+1}:")
            print(f"  炮点UTM: ({th.sxutm:.2f}, {th.syutm:.2f})")
            print(f"  接收点UTM: ({th.rxutm:.2f}, {th.ryutm:.2f})")
            print(f"  计算的炮检距: {offset_calculated:.6f} km")
            print(f"  头文件中的炮检距: {th.offsti:.6f} km")
        
        # 统计UTM坐标范围
        sxutm_list = [th.sxutm for th in trace_headers]
        syutm_list = [th.syutm for th in trace_headers]
        rxutm_list = [th.rxutm for th in trace_headers]
        ryutm_list = [th.ryutm for th in trace_headers]
        
        print(f"\nUTM坐标统计:")
        print(f"  炮点X范围: [{min(sxutm_list):.2f}, {max(sxutm_list):.2f}]")
        print(f"  炮点Y范围: [{min(syutm_list):.2f}, {max(syutm_list):.2f}]")
        print(f"  接收点X范围: [{min(rxutm_list):.2f}, {max(rxutm_list):.2f}]")
        print(f"  接收点Y范围: [{min(ryutm_list):.2f}, {max(ryutm_list):.2f}]")
        
        # 计算所有道的炮检距
        calculated_offsets = []
        for th in trace_headers:
            dx = th.rxutm - th.sxutm
            dy = th.ryutm - th.syutm
            offset = np.sqrt(dx*dx + dy*dy) / 1000.0
            calculated_offsets.append(offset)
        
        print(f"\n计算的炮检距统计:")
        print(f"  范围: [{min(calculated_offsets):.6f}, {max(calculated_offsets):.6f}] km")
        print(f"  非零值数量: {sum(1 for x in calculated_offsets if x > 0.001)}")
        
        if max(calculated_offsets) > 0.001:
            print(f"\n✓ 可以从UTM坐标计算炮检距！")
        else:
            print(f"\n✗ UTM坐标也无法计算炮检距（可能坐标都是0或相同）")

if __name__ == '__main__':
    check_zfile_trace_headers()
