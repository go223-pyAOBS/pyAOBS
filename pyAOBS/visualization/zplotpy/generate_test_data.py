"""
generate_test_data.py - 生成 Z 格式测试数据

用于生成符合 Z 格式规范的测试数据文件，用于测试数据加载模块
"""

import struct
import numpy as np
import os


def _generate_ricker_wavelet(t, t0, f0):
    """生成Ricker子波（Mexican Hat小波）
    
    Args:
        t: 时间数组
        t0: 子波中心时间
        f0: 主频（Hz）
    
    Returns:
        Ricker子波数组
    """
    # Ricker子波公式：A(t) = (1 - 2*π²*f0²*(t-t0)²) * exp(-π²*f0²*(t-t0)²)
    tau = np.pi * f0 * (t - t0)
    ricker = (1 - 2 * tau**2) * np.exp(-tau**2)
    return ricker


def generate_z_format_data(output_dir='test_data', ntraces=40, npts=1000, 
                          npick=10, ifmt=1, nrec=3):
    """生成 Z 格式测试数据
    
    Args:
        output_dir: 输出目录
        ntraces: 总道数
        npts: 每道采样点数
        npick: 拾取字数
        ifmt: 数据格式（1=float32, 0=int16）
        nrec: 记录数（炮集数）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    dfile = os.path.join(output_dir, 'test_data.z')
    hfile = os.path.join(output_dir, 'test_header.hdr')
    rfile = os.path.join(output_dir, 'test_record.rsp')
    
    # 生成文件头参数
    sint = 4000  # 4ms = 4000微秒
    tstart = 0  # 0毫秒
    tend = (npts - 1) * sint // 1000  # 毫秒
    vredf = 6.0  # 折合速度 6 km/s
    
    # 注意：不在全局设置随机种子，而是在每道生成时使用 trace_idx 相关的种子
    # 这样可以确保每道的随机数不同，但每次运行生成的数据一致
    
    # 计算每个记录的道数（尽量平均分配）
    traces_per_record = ntraces // nrec
    remainder = ntraces % nrec
    
    # 1. 生成数据文件 (dfile)
    print(f"生成数据文件: {dfile}")
    with open(dfile, 'wb') as f:
        # 写入文件头（记录1，52字节）
        header = struct.pack('<7i', ntraces, npts, sint, tstart, tend, nrec, npick)
        f.write(header)
        f.write(struct.pack('<f', vredf))  # vredf
        f.write(struct.pack('<i', ifmt))  # ifmt
        f.write(struct.pack('<4f', 0.0, 0.0, 0.0, 0.0))  # xlatlong, xelev, xutm, cm
        
        # 计算记录大小
        trace_header_size = (22 + npick) * 4
        if ifmt == 1:
            trace_data_size = npts * 4
            dtype = np.float32
        else:
            trace_data_size = npts * 2
            dtype = np.int16
        
        record_size = trace_header_size + trace_data_size
        
        # 写入每条道（记录2到记录N+1）
        trace_idx = 0
        for rec_idx in range(nrec):
            # 计算当前记录的道数
            current_record_traces = traces_per_record + (1 if rec_idx < remainder else 0)
            rec_num = rec_idx + 1  # 记录号从1开始
            
            for local_trace_idx in range(current_record_traces):
                i = trace_idx
                # 生成道头（22 + npick 个浮点数）
                trace_header = np.zeros(22 + npick, dtype=np.float32)
                # 填充一些示例值
                trace_header[0] = float(rec_num)  # 记录号（炮站号）
                trace_header[1] = float(local_trace_idx + 1)  # 道序号（在当前记录内）
                trace_header[2] = float(i + 1)  # 接收站号（全局）
                trace_header[3] = 1.0  # 数据类型（垂直）
                trace_header[4] = 1.0  # 数据标志（有效）
                # 炮检距（米）：确保与.hdr文件中的值一致
                # 使用 (i + 1) * 1000 米，即 1-40 公里，与默认offsets值一致
                trace_header[5] = float((i + 1) * 1000)  # 炮检距（米），对应1-40公里
                trace_header[6] = float(i * 10)  # 方位角（分）
                trace_header[7] = 1.0  # 增益因子
            
                # 拾取时间（后npick个值）
                # 计算理论走时（与信号生成时一致）
                offset_km = (i + 1) * 1.0
                t0 = 2.0
                v = 4.0
                theoretical_time = np.sqrt(t0**2 + (offset_km / v)**2)
                
                # 添加拾取误差（模拟手动拾取的不确定性）
                # 使用基于trace_idx的种子，确保每次运行生成的数据一致
                np.random.seed(1000 + i)  # 为每道设置不同的种子（1000+i避免与信号生成冲突）
                pick_error = np.random.normal(0, 0.03)  # 30ms的标准误差
                pick_time = theoretical_time + pick_error
                
                for j in range(npick):
                    if j == 0:  # 第1个拾取字设置为有效值（对应第一个反射）
                        trace_header[22 + j] = pick_time
                    elif j == 1:  # 第2个拾取字设置为第二个反射
                        pick_error_2 = np.random.normal(0, 0.03)
                        pick_time_2 = pick_time + 1.5 + pick_error_2
                        trace_header[22 + j] = pick_time_2 if pick_time_2 > 0 else 0.0
                    else:
                        trace_header[22 + j] = 0.0  # 其他拾取字为0
                
                # 改进（阶段2）：拾取信息不写入数据文件，只写入0值占位符
                # 实际拾取信息应保存到头文件（.hdr）
                # 为了保持格式兼容性，数据文件中写入0值占位符
                trace_header[22:22 + npick] = 0.0  # 写入0值占位符，不写入实际拾取信息
                
                # 写入道头（包含0值占位符）
                f.write(trace_header.tobytes())
                
                # 生成道数据（带有明显地震信号，不同到时）
                # 时间数组（秒）
                dt = sint / 1000000.0  # 采样间隔（秒），sint是微秒
                t = np.arange(npts) * dt
                
                # 计算炮检距（公里）
                offset_km = (i + 1) * 1.0  # 1-40公里
                
                # 模拟走时曲线：t = sqrt(t0^2 + (x/v)^2)
                # 使用双曲线走时模型
                t0 = 2.0  # 零偏移走时（秒）
                v = 4.0   # 速度（km/s）
                arrival_time = np.sqrt(t0**2 + (offset_km / v)**2)
                
                # 添加随机扰动，使到时不完全符合理论曲线（模拟拾取误差）
                # 注意：这里不设置种子，让每道的扰动不同
                # 但拾取时间会使用相同的种子，确保拾取时间与信号到时对应
                arrival_time += np.random.normal(0, 0.05)  # 5%的随机扰动
                
                # 生成Ricker子波（地震信号）
                f0 = 10.0  # 主频（Hz）
                ricker_wavelet = _generate_ricker_wavelet(t, arrival_time, f0)
                
                # 添加多个反射（模拟多层反射）
                trace_data = ricker_wavelet.copy()
                for j in range(2):  # 添加2个后续反射
                    t_reflect = arrival_time + (j + 1) * 1.5  # 后续反射时间
                    if t_reflect < t[-1]:  # 确保在时间范围内
                        amplitude = 0.5 / (j + 2)  # 振幅衰减
                        reflect_wavelet = _generate_ricker_wavelet(t, t_reflect, f0)
                        trace_data += amplitude * reflect_wavelet
                
                # 添加噪声（信噪比约为3:1）
                # 重置随机种子以确保噪声的随机性
                np.random.seed()  # 使用系统随机种子
                signal_amplitude = np.max(np.abs(trace_data))
                noise_level = signal_amplitude / 3.0
                noise = np.random.normal(0, noise_level, npts)
                trace_data += noise
                
                # 归一化到合理范围
                max_val = np.max(np.abs(trace_data))
                if max_val > 0:
                    trace_data = trace_data / max_val * 0.8  # 归一化到0.8
                
                trace_data = trace_data.astype(dtype)
                
                # 写入道数据
                f.write(trace_data.tobytes())
                
                trace_idx += 1
    
    print(f"  数据文件已生成: {ntraces} 道, {npts} 点/道")
    
    # 2. 生成头文件 (hfile) - Fortran 未格式化格式
    print(f"生成头文件: {hfile}")
    with open(hfile, 'wb') as f:
        trace_idx = 0
        for rec_idx in range(nrec):
            # 计算当前记录的道数
            current_record_traces = traces_per_record + (1 if rec_idx < remainder else 0)
            rec_num = rec_idx + 1  # 记录号从1开始
            
            for local_trace_idx in range(current_record_traces):
                i = trace_idx
                # 计算记录大小（按照C结构体的内存布局：9个int + 13个float = 22个字段，88字节）
                record_size = 88 + npick * 4  # 22个字段（88字节）+ npick个float
                
                # 写入记录长度标记
                f.write(struct.pack('<i', record_size))
                
                # 按照C结构体的内存布局写入（混合整数和浮点数）
                # 字段顺序：nrec(int), itsn(int), ireci(int), itype(int), iflag(int),
                #           offset(float), azi(float), igain(int), texact(float),
                #           slat(float), slong(float), selev(int), swdepth(int),
                #           rlat(float), rlong(float), relev(int),
                #           sxutm(float), syutm(float), sz(float),
                #           rxutm(float), ryutm(float), rz(float)
                
                # 准备字段值
                nrec = rec_num  # 记录号（炮站号）
                itsn = local_trace_idx + 1  # 道序号（在当前记录内）
                ireci = rec_num  # 接收站号
                itype = 1  # 数据类型（垂直）
                iflag = 1  # 数据标志（有效）
                offset = float((i + 1) * 1000)  # 炮检距（米），对应1-40公里
                azi = float(i * 10)  # 方位角（分）
                igain = 1  # 增益因子
                texact = 0.0
                slat = 30.0 + rec_idx * 0.1 + i * 0.01  # 不同记录有不同的位置
                slong = 120.0 + rec_idx * 0.1 + i * 0.01
                selev = 0  # 整数
                swdepth = 0  # 整数
                rlat = 30.0 + rec_idx * 0.1 + i * 0.01
                rlong = 120.0 + rec_idx * 0.1 + i * 0.01
                relev = 0  # 整数
                sxutm = float(rec_idx * 10000 + i * 1000)  # 不同记录有不同的UTM坐标
                syutm = float(rec_idx * 10000 + i * 1000)
                sz = 0.0
                rxutm = float(rec_idx * 10000 + i * 1000)
                ryutm = float(rec_idx * 10000 + i * 1000)
                rz = 0.0
                
                # 按照C结构体布局写入（88字节）
                # 前5个整数（20字节）
                f.write(struct.pack('<5i', nrec, itsn, ireci, itype, iflag))
                # 2个浮点数（8字节）
                f.write(struct.pack('<2f', offset, azi))
                # 1个整数（4字节）
                f.write(struct.pack('<i', igain))
                # 3个浮点数（12字节）
                f.write(struct.pack('<3f', texact, slat, slong))
                # 2个整数（8字节）
                f.write(struct.pack('<2i', selev, swdepth))
                # 2个浮点数（8字节）
                f.write(struct.pack('<2f', rlat, rlong))
                # 1个整数（4字节）
                f.write(struct.pack('<i', relev))
                # 6个浮点数（24字节）
                f.write(struct.pack('<6f', sxutm, syutm, sz, rxutm, ryutm, rz))
                
                # 写入拾取数组（npick个）
                # 改进（阶段2）：拾取信息不写入数据文件，只写入0值占位符
                # 实际拾取信息应保存到头文件（.hdr）
                # 为了保持格式兼容性，数据文件中写入0值占位符
                picks = np.zeros(npick, dtype=np.float32)  # 写入0值占位符，不写入实际拾取信息
                
                # 注意：实际拾取信息应该保存到头文件（.hdr），这里只写入占位符
                # 如果需要测试拾取功能，应该从头文件读取拾取信息
                
                f.write(picks.tobytes())
                
                # 写入记录长度标记（结束）
                f.write(struct.pack('<i', record_size))
                
                trace_idx += 1
    
    print(f"  头文件已生成: {ntraces} 条道头记录")
    
    # 3. 生成记录文件 (rfile) - ASCII格式
    print(f"生成记录文件: {rfile}")
    with open(rfile, 'w') as f:
        for i in range(nrec):
            ishnum = i + 1
            xmod = float(i * 10)
            ymod = float(i * 5)
            az = float(i * 90)
            title = f"Record {i+1}"
            f.write(f"{ishnum} {xmod} {ymod} {az} {title}\n")
    
    print(f"  记录文件已生成: {nrec} 条记录")
    
    print(f"\n测试数据生成完成！")
    print(f"数据文件: {dfile}")
    print(f"头文件: {hfile}")
    print(f"记录文件: {rfile}")
    
    return dfile, hfile, rfile


def generate_multicomponent_data(output_dir='test_data_multicomponent', nstations=10, npts=1000, 
                                 npick=10, ifmt=1, nrec=1, offset_range=None, include_negative=True):
    """生成多分量（4分量）Z 格式测试数据
    
    每个接收站包含4个分量：
    - itypei=1: 垂直分量 (Vertical)
    - itypei=2: 径向分量 (Radial)
    - itypei=3: 横向分量 (Transverse)
    - itypei=4: 水听器 (Hydrophone)
    
    Args:
        output_dir: 输出目录
        nstations: 接收站数量（每个站4个分量，总道数 = nstations * 4）
        npts: 每道采样点数
        npick: 拾取字数
        ifmt: 数据格式（1=float32, 0=int16）
        nrec: 记录数（炮集数），默认1
        offset_range: 偏移距范围 (min, max) 公里，例如 (-10, 10) 或 (0, 20)
                     如果为 None，则根据 include_negative 自动计算
        include_negative: 是否包含负偏移距，默认 True
                         如果为 True 且 offset_range 为 None，则生成 (-nstations/2, nstations/2) 范围
                         如果为 False 且 offset_range 为 None，则生成 (0, nstations) 范围
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 总道数 = 接收站数 * 4（每个站4个分量）
    ntraces = nstations * 4
    
    dfile = os.path.join(output_dir, 'multicomponent_data.z')
    hfile = os.path.join(output_dir, 'multicomponent_header.hdr')
    rfile = os.path.join(output_dir, 'multicomponent_record.rsp')
    
    # 生成文件头参数
    sint = 4000  # 4ms = 4000微秒
    tstart = 0  # 0毫秒
    tend = (npts - 1) * sint // 1000  # 毫秒
    vredf = 6.0  # 折合速度 6 km/s
    
    # 分量类型映射
    component_types = [1, 2, 3, 4]  # 垂直、径向、横向、水听器
    component_names = ['垂直', '径向', '横向', '水听器']
    
    # 计算偏移距范围
    if offset_range is None:
        if include_negative:
            # 包含负偏移距：从 -nstations/2 到 nstations/2
            offset_min_km = -nstations / 2.0
            offset_max_km = nstations / 2.0
        else:
            # 只有正偏移距：从 0 到 nstations
            offset_min_km = 0.0
            offset_max_km = float(nstations)
    else:
        offset_min_km, offset_max_km = offset_range
    
    # 计算每个接收站的偏移距（均匀分布）
    if nstations > 1:
        offset_step = (offset_max_km - offset_min_km) / (nstations - 1)
        station_offsets_km = [offset_min_km + i * offset_step for i in range(nstations)]
    else:
        station_offsets_km = [(offset_min_km + offset_max_km) / 2.0]
    
    # 1. 生成数据文件 (dfile)
    print(f"生成多分量数据文件: {dfile}")
    print(f"  接收站数: {nstations}, 总道数: {ntraces} (每站4个分量)")
    print(f"  偏移距范围: {offset_min_km:.1f} 到 {offset_max_km:.1f} 公里")
    
    with open(dfile, 'wb') as f:
        # 写入文件头（记录1，52字节）
        header = struct.pack('<7i', ntraces, npts, sint, tstart, tend, nrec, npick)
        f.write(header)
        f.write(struct.pack('<f', vredf))  # vredf
        f.write(struct.pack('<i', ifmt))  # ifmt
        f.write(struct.pack('<4f', 0.0, 0.0, 0.0, 0.0))  # xlatlong, xelev, xutm, cm
        
        # 计算记录大小
        trace_header_size = (22 + npick) * 4
        if ifmt == 1:
            trace_data_size = npts * 4
            dtype = np.float32
        else:
            trace_data_size = npts * 2
            dtype = np.int16
        
        # 写入每条道
        trace_idx = 0
        for station_idx in range(nstations):
            # 为每个接收站生成4个分量
            for comp_idx, itypei in enumerate(component_types):
                # 生成道头（22 + npick 个浮点数）
                trace_header = np.zeros(22 + npick, dtype=np.float32)
                
                # 填充道头信息
                trace_header[0] = float(1)  # 记录号（炮站号），默认1
                trace_header[1] = float(trace_idx + 1)  # 道序号（全局）
                trace_header[2] = float(station_idx + 1)  # 接收站号
                trace_header[3] = float(itypei)  # 数据类型（1=垂直，2=径向，3=横向，4=水听器）
                trace_header[4] = 1.0  # 数据标志（有效）
                
                # 炮检距（米）：使用计算的偏移距
                offset_km = station_offsets_km[station_idx]
                offset_m = offset_km * 1000.0  # 转换为米
                trace_header[5] = float(offset_m)
                trace_header[6] = float(station_idx * 10)  # 方位角（分）
                trace_header[7] = 1.0  # 增益因子
                
                # 拾取时间（后npick个值）
                # 使用绝对值计算走时（走时只依赖于距离，不依赖于方向）
                t0 = 2.0
                v = 4.0
                theoretical_time = np.sqrt(t0**2 + (abs(offset_km) / v)**2)
                
                # 添加拾取误差
                np.random.seed(1000 + trace_idx)
                pick_error = np.random.normal(0, 0.03)
                pick_time = theoretical_time + pick_error
                
                for j in range(npick):
                    if j == 0:
                        trace_header[22 + j] = pick_time
                    elif j == 1:
                        pick_error_2 = np.random.normal(0, 0.03)
                        pick_time_2 = pick_time + 1.5 + pick_error_2
                        trace_header[22 + j] = pick_time_2 if pick_time_2 > 0 else 0.0
                    else:
                        trace_header[22 + j] = 0.0
                
                # 改进（阶段2）：拾取信息不写入数据文件，只写入0值占位符
                # 实际拾取信息应保存到头文件（.hdr）
                # 为了保持格式兼容性，数据文件中写入0值占位符
                trace_header[22:22 + npick] = 0.0  # 写入0值占位符，不写入实际拾取信息
                
                # 写入道头（包含0值占位符）
                f.write(trace_header.tobytes())
                
                # 生成道数据（不同分量有不同的信号特征）
                dt = sint / 1000000.0  # 采样间隔（秒）
                t = np.arange(npts) * dt
                
                # 计算炮检距（公里）- 使用绝对值计算走时
                offset_km_abs = abs(offset_km)
                
                # 模拟走时曲线（使用绝对值）
                t0 = 2.0
                v = 4.0
                arrival_time = np.sqrt(t0**2 + (offset_km_abs / v)**2)
                arrival_time += np.random.normal(0, 0.05)
                
                # 根据分量类型生成不同的信号
                f0 = 10.0  # 主频（Hz）
                
                if itypei == 1:  # 垂直分量：P波信号，振幅较大
                    ricker_wavelet = _generate_ricker_wavelet(t, arrival_time, f0)
                    amplitude_factor = 1.0
                    phase_shift = 0.0
                elif itypei == 2:  # 径向分量：S波信号，振幅中等，相位略有延迟
                    ricker_wavelet = _generate_ricker_wavelet(t, arrival_time + 0.1, f0 * 0.8)
                    amplitude_factor = 0.7
                    phase_shift = 0.05
                elif itypei == 3:  # 横向分量：S波信号，振幅较小
                    ricker_wavelet = _generate_ricker_wavelet(t, arrival_time + 0.15, f0 * 0.75)
                    amplitude_factor = 0.5
                    phase_shift = 0.1
                else:  # itypei == 4: 水听器：P波信号，但频率特性不同
                    ricker_wavelet = _generate_ricker_wavelet(t, arrival_time, f0 * 1.2)
                    amplitude_factor = 0.8
                    phase_shift = -0.02
                
                # 应用振幅因子
                trace_data = ricker_wavelet * amplitude_factor
                
                # 添加多个反射
                for j in range(2):
                    t_reflect = arrival_time + (j + 1) * 1.5 + phase_shift
                    if t_reflect < t[-1]:
                        amp = (0.5 / (j + 2)) * amplitude_factor
                        reflect_wavelet = _generate_ricker_wavelet(t, t_reflect, f0)
                        trace_data += amp * reflect_wavelet
                
                # 添加噪声（信噪比约为3:1）
                np.random.seed()
                signal_amplitude = np.max(np.abs(trace_data))
                noise_level = signal_amplitude / 3.0
                noise = np.random.normal(0, noise_level, npts)
                trace_data += noise
                
                # 归一化
                max_val = np.max(np.abs(trace_data))
                if max_val > 0:
                    trace_data = trace_data / max_val * 0.8
                
                trace_data = trace_data.astype(dtype)
                
                # 写入道数据
                f.write(trace_data.tobytes())
                
                trace_idx += 1
    
    print(f"  数据文件已生成: {ntraces} 道, {npts} 点/道")
    
    # 2. 生成头文件 (hfile)
    print(f"生成多分量头文件: {hfile}")
    with open(hfile, 'wb') as f:
        trace_idx = 0
        for station_idx in range(nstations):
            for comp_idx, itypei in enumerate(component_types):
                # 计算记录大小
                record_size = 88 + npick * 4
                
                # 写入记录长度标记
                f.write(struct.pack('<i', record_size))
                
                # 准备字段值
                nrec = 1  # 记录号
                itsn = trace_idx + 1  # 道序号
                ireci = station_idx + 1  # 接收站号
                itype = itypei  # 数据类型（1-4）
                iflag = 1  # 数据标志（有效）
                offset_km = station_offsets_km[station_idx]
                offset = float(offset_km * 1000.0)  # 炮检距（米）
                azi = float(station_idx * 10)  # 方位角（分）
                igain = 1
                texact = 0.0
                slat = 30.0 + station_idx * 0.01
                slong = 120.0 + station_idx * 0.01
                selev = 0
                swdepth = 0
                rlat = 30.0 + station_idx * 0.01
                rlong = 120.0 + station_idx * 0.01
                relev = 0
                sxutm = float(station_idx * 1000)
                syutm = float(station_idx * 1000)
                sz = 0.0
                rxutm = float(station_idx * 1000)
                ryutm = float(station_idx * 1000)
                rz = 0.0
                
                # 按照C结构体布局写入
                f.write(struct.pack('<5i', nrec, itsn, ireci, itype, iflag))
                f.write(struct.pack('<2f', offset, azi))
                f.write(struct.pack('<i', igain))
                f.write(struct.pack('<3f', texact, slat, slong))
                f.write(struct.pack('<2i', selev, swdepth))
                f.write(struct.pack('<2f', rlat, rlong))
                f.write(struct.pack('<i', relev))
                f.write(struct.pack('<6f', sxutm, syutm, sz, rxutm, ryutm, rz))
                
                # 写入拾取数组
                # 改进（阶段2）：拾取信息不写入数据文件，只写入0值占位符
                # 实际拾取信息应保存到头文件（.hdr）
                # 为了保持格式兼容性，数据文件中写入0值占位符
                picks = np.zeros(npick, dtype=np.float32)  # 写入0值占位符，不写入实际拾取信息
                
                # 注意：实际拾取信息应该保存到头文件（.hdr），这里只写入占位符
                # 如果需要测试拾取功能，应该从头文件读取拾取信息
                
                f.write(picks.tobytes())
                
                # 写入记录长度标记（结束）
                f.write(struct.pack('<i', record_size))
                
                trace_idx += 1
    
    print(f"  头文件已生成: {ntraces} 条道头记录")
    
    # 3. 生成记录文件 (rfile)
    print(f"生成记录文件: {rfile}")
    with open(rfile, 'w') as f:
        for i in range(nrec):
            ishnum = i + 1
            xmod = float(i * 10)
            ymod = float(i * 5)
            az = float(i * 90)
            title = f"Multicomponent Record {i+1}"
            f.write(f"{ishnum} {xmod} {ymod} {az} {title}\n")
    
    print(f"  记录文件已生成: {nrec} 条记录")
    
    print(f"\n多分量测试数据生成完成！")
    print(f"数据文件: {dfile}")
    print(f"头文件: {hfile}")
    print(f"记录文件: {rfile}")
    print(f"\n分量说明:")
    for i, (itypei, name) in enumerate(zip(component_types, component_names)):
        print(f"  itypei={itypei}: {name}分量")
    
    return dfile, hfile, rfile


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='生成 Z 格式测试数据')
    parser.add_argument('--multicomponent', action='store_true', 
                       help='生成多分量数据（每个接收站4个分量）')
    parser.add_argument('--output-dir', default='test_data', help='输出目录')
    parser.add_argument('--ntraces', type=int, default=40, help='总道数（单分量模式）')
    parser.add_argument('--nstations', type=int, default=10, 
                       help='接收站数量（多分量模式，总道数=nstations*4）')
    parser.add_argument('--npts', type=int, default=1000, help='每道采样点数')
    parser.add_argument('--npick', type=int, default=10, help='拾取字数')
    parser.add_argument('--ifmt', type=int, default=1, choices=[0, 1], 
                       help='数据格式（1=float32, 0=int16）')
    parser.add_argument('--nrec', type=int, default=3, help='记录数（炮集数）')
    parser.add_argument('--include-negative', action='store_true', default=True,
                       help='包含负偏移距（多分量模式，默认启用）')
    parser.add_argument('--no-negative', dest='include_negative', action='store_false',
                       help='不包含负偏移距（多分量模式）')
    parser.add_argument('--offset-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                       help='偏移距范围（公里），例如 --offset-range -10 10')
    
    args = parser.parse_args()
    
    if args.multicomponent:
        # 解析偏移距范围
        offset_range = None
        if args.offset_range:
            offset_range = tuple(args.offset_range)
        
        # 生成多分量数据
        generate_multicomponent_data(
            output_dir=args.output_dir,
            nstations=args.nstations,
            npts=args.npts,
            npick=args.npick,
            ifmt=args.ifmt,
            nrec=args.nrec if args.nrec == 1 else 1,  # 多分量数据默认单记录
            offset_range=offset_range,
            include_negative=args.include_negative
        )
    else:
        # 生成单分量数据
        generate_z_format_data(
            output_dir=args.output_dir,
            ntraces=args.ntraces,
            npts=args.npts,
            npick=args.npick,
            ifmt=args.ifmt,
            nrec=args.nrec
        )
