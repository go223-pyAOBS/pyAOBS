"""
验证头文件的实际结构，对比C代码和Python代码的差异
"""

import os
import sys
import struct
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader

def verify_header_structure():
    """验证头文件的实际结构"""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    zfile = os.path.join(base_dir, 'test', 'Rotated_OBS08_4C.z')
    hfile = os.path.join(base_dir, 'test', 'Rotated_OBS08_4C.hdr')
    
    print("=" * 80)
    print("验证头文件结构")
    print("=" * 80)
    
    # 读取文件头获取npick
    loader = DataLoader()
    data = loader.load_z_format(dfile=zfile)
    header = data['header']
    npick = header.npick
    
    print(f"\n从.z文件读取的npick: {npick}")
    print(f"\n开始分析头文件: {hfile}")
    print("-" * 80)
    
    # 读取第一条记录，分析字节布局
    with open(hfile, 'rb') as f:
        # 读取记录长度标记
        rec_len_bytes = f.read(4)
        rec_len = struct.unpack('<i', rec_len_bytes)[0]
        print(f"\n记录长度标记: {rec_len} 字节")
        
        # 读取88字节的道头数据
        header_bytes = f.read(88)
        
        print(f"\n道头数据（88字节）:")
        print(f"  前20字节（5个int）:")
        ints1 = struct.unpack('<5i', header_bytes[0:20])
        print(f"    {ints1}")
        
        print(f"\n  字节20-27（2个float）:")
        floats1 = struct.unpack('<2f', header_bytes[20:28])
        print(f"    {floats1}")
        
        print(f"\n  字节28-31（1个int）:")
        int2 = struct.unpack('<i', header_bytes[28:32])
        print(f"    {int2}")
        
        print(f"\n  字节32-43（3个float）:")
        floats2 = struct.unpack('<3f', header_bytes[32:44])
        print(f"    {floats2}")
        
        print(f"\n  字节44-51（2个int）:")
        ints2 = struct.unpack('<2i', header_bytes[44:52])
        print(f"    {ints2}")
        
        print(f"\n  字节52-59（2个float）:")
        floats3 = struct.unpack('<2f', header_bytes[52:60])
        print(f"    {floats3}")
        
        print(f"\n  字节60-63（1个int）:")
        int3 = struct.unpack('<i', header_bytes[60:64])
        print(f"    {int3}")
        
        print(f"\n  字节64-87（6个float）:")
        floats4 = struct.unpack('<6f', header_bytes[64:88])
        print(f"    {floats4}")
        
        # 按照Python代码当前的方式读取
        print(f"\n" + "=" * 80)
        print("按照Python代码当前方式读取（6个int + 16个float）:")
        print("=" * 80)
        
        f.seek(4)  # 回到记录开始
        rec_len_bytes = f.read(4)
        ints_bytes = f.read(24)
        floats_bytes = f.read(64)
        
        ints = struct.unpack('<6i', ints_bytes)
        floats = struct.unpack('<16f', floats_bytes)
        
        print(f"\n6个整数:")
        print(f"  ishoti={ints[0]}, itsn={ints[1]}, ireci={ints[2]}")
        print(f"  itypei={ints[3]}, iflagi={ints[4]}, igaini={ints[5]}")
        
        print(f"\n16个浮点数:")
        print(f"  offsti={floats[0]:.6f}, azi={floats[1]:.6f}, texact={floats[2]:.6f}")
        print(f"  slat={floats[3]:.6f}, slong={floats[4]:.6f}, selev={floats[5]:.6f}")
        print(f"  swdepth={floats[6]:.6f}, rlat={floats[7]:.6f}, rlong={floats[8]:.6f}")
        print(f"  relev={floats[9]:.6f}, sxutm={floats[10]:.6f}, syutm={floats[11]:.6f}")
        print(f"  sz={floats[12]:.6f}, rxutm={floats[13]:.6f}, ryutm={floats[14]:.6f}")
        print(f"  rz={floats[15]:.6f}")
        
        # 对比两种方式
        print(f"\n" + "=" * 80)
        print("对比分析:")
        print("=" * 80)
        
        print(f"\n如果按照C结构体布局（混合类型）:")
        print(f"  nrec={ints1[0]}, itsn={ints1[1]}, ireci={ints1[2]}")
        print(f"  itype={ints1[3]}, iflag={ints1[4]}")
        print(f"  offset={floats1[0]:.6f}, azi={floats1[1]:.6f}")
        print(f"  igain={int2[0]}")
        print(f"  texact={floats2[0]:.6f}, slat={floats2[1]:.6f}, slong={floats2[2]:.6f}")
        print(f"  selev={ints2[0]}, swdepth={ints2[1]}")
        print(f"  rlat={floats3[0]:.6f}, rlong={floats3[1]:.6f}")
        print(f"  relev={int3[0]}")
        print(f"  sxutm={floats4[0]:.6f}, syutm={floats4[1]:.6f}, sz={floats4[2]:.6f}")
        print(f"  rxutm={floats4[3]:.6f}, ryutm={floats4[4]:.6f}, rz={floats4[5]:.6f}")
        
        print(f"\n如果按照Python代码方式（先6个int后16个float）:")
        print(f"  ishoti={ints[0]}, itsn={ints[1]}, ireci={ints[2]}")
        print(f"  itypei={ints[3]}, iflagi={ints[4]}, igaini={ints[5]}")
        print(f"  offsti={floats[0]:.6f}, azi={floats[1]:.6f}")
        print(f"  texact={floats[2]:.6f}, slat={floats[3]:.6f}, slong={floats[4]:.6f}")
        print(f"  selev={floats[5]:.6f}, swdepth={floats[6]:.6f}")
        print(f"  rlat={floats[7]:.6f}, rlong={floats[8]:.6f}")
        print(f"  relev={floats[9]:.6f}, sxutm={floats[10]:.6f}, syutm={floats[11]:.6f}")
        print(f"  sz={floats[12]:.6f}, rxutm={floats[13]:.6f}, ryutm={floats[14]:.6f}")
        print(f"  rz={floats[15]:.6f}")
        
        # 检查哪种方式更合理
        print(f"\n" + "=" * 80)
        print("合理性检查:")
        print("=" * 80)
        
        # 检查offset字段：如果按照C结构体，offset在位置20-23（浮点数）
        # 如果按照Python代码，offset在位置24-27（第一个浮点数）
        offset_c_struct = floats1[0]  # C结构体方式
        offset_python = floats[0]      # Python代码方式
        
        print(f"\n炮检距（offset）:")
        print(f"  C结构体方式（字节20-23）: {offset_c_struct:.6f} 米")
        print(f"  Python代码方式（字节24-27）: {offset_python:.6f} 米")
        
        # 检查UTM坐标
        sxutm_c = floats4[0]  # C结构体方式
        sxutm_p = floats[10]  # Python代码方式
        
        print(f"\n炮点UTM X坐标（sxutm）:")
        print(f"  C结构体方式: {sxutm_c:.2f} 米")
        print(f"  Python代码方式: {sxutm_p:.2f} 米")
        
        # 检查哪种方式的值更合理（UTM坐标通常是6-7位数）
        if abs(sxutm_c) > 1000 and abs(sxutm_p) > 1000:
            print(f"\n两种方式的UTM坐标都合理，需要进一步验证")
        elif abs(sxutm_c) > 1000:
            print(f"\n✓ C结构体方式的UTM坐标更合理")
        elif abs(sxutm_p) > 1000:
            print(f"\n✓ Python代码方式的UTM坐标更合理")
        else:
            print(f"\n两种方式的UTM坐标都不合理，可能数据有问题")

if __name__ == '__main__':
    verify_header_structure()
