"""
hdr_to_tx.py - 将头文件中的拾取转换为 tx.in 格式

类似 z2tx 命令的功能，将 .hdr 文件中的拾取转换为 rayinvr 可用的 tx.in 格式
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import struct
try:
    from .data_loader import TraceHeader, ZFormatHeader
except ImportError:
    from data_loader import TraceHeader, ZFormatHeader


@dataclass
class Z2TxConfig:
    """z2tx 转换配置"""
    # OBS 配置：{shot_number: config}
    obs_configs: Dict[int, Dict]  # {shot: {'xmod': float, 'tshift': float, 'xshift': float, 'picku': List[float]}}
    iamp: int = 0  # 0=走时, 1=振幅


def read_header_file(hfile: str, npick: int) -> List[TraceHeader]:
    """读取头文件
    
    Args:
        hfile: 头文件路径
        npick: 拾取字数
        
    Returns:
        TraceHeader 列表
    """
    trace_headers = []
    
    with open(hfile, 'rb') as f:
        while True:
            # 读取记录长度标记
            rec_len_bytes = f.read(4)
            if len(rec_len_bytes) < 4:
                break
            
            rec_len = struct.unpack('<i', rec_len_bytes)[0]
            
            # 读取88字节的道头数据（按照C结构体的内存布局）
            header_bytes = f.read(88)
            if len(header_bytes) < 88:
                break
            
            # 按照C结构体布局解析（22个字段，每个4字节）
            # 字段顺序：nrec(int), itsn(int), ireci(int), itype(int), iflag(int),
            #           offset(float), azi(float), igain(int), texact(float),
            #           slat(float), slong(float), selev(int), swdepth(int),
            #           rlat(float), rlong(float), relev(int),
            #           sxutm(float), syutm(float), sz(float),
            #           rxutm(float), ryutm(float), rz(float)
            
            # 解析前5个整数（字节0-19）
            nrec, itsn, ireci, itype, iflag = struct.unpack('<5i', header_bytes[0:20])
            
            # 解析2个浮点数（字节20-27）
            offset, azi = struct.unpack('<2f', header_bytes[20:28])
            
            # 解析1个整数（字节28-31）
            igain, = struct.unpack('<i', header_bytes[28:32])
            
            # 解析3个浮点数（字节32-43）
            texact, slat, slong = struct.unpack('<3f', header_bytes[32:44])
            
            # 解析2个整数（字节44-51）
            selev, swdepth = struct.unpack('<2i', header_bytes[44:52])
            
            # 解析2个浮点数（字节52-59）
            rlat, rlong = struct.unpack('<2f', header_bytes[52:60])
            
            # 解析1个整数（字节60-63）
            relev, = struct.unpack('<i', header_bytes[60:64])
            
            # 解析6个浮点数（字节64-87）
            sxutm, syutm, sz, rxutm, ryutm, rz = struct.unpack('<6f', header_bytes[64:88])
            
            # 读取拾取数组（npick个）
            picks_bytes = f.read(npick * 4)
            if len(picks_bytes) < npick * 4:
                break
            picks = list(struct.unpack(f'<{npick}f', picks_bytes))
            
            # 读取记录长度标记（结束）
            rec_len2_bytes = f.read(4)
            if len(rec_len2_bytes) < 4:
                break
            
            # 转换单位
            offsti_km = offset / 1000.0  # 米转千米
            azi_deg = azi / 60.0  # 分转度
            
            trace_header = TraceHeader(
                ishoti=nrec,  # C代码中nrec对应ishoti（炮站号/记录号）
                itsn=itsn,
                ireci=ireci,
                itypei=itype,
                iflagi=iflag,
                igaini=igain,
                offsti=offsti_km,
                azi=azi_deg,
                texact=texact,
                slat=slat,
                slong=slong,
                selev=float(selev),  # 转换为float以匹配TraceHeader定义
                swdepth=float(swdepth),  # 转换为float以匹配TraceHeader定义
                rlat=rlat,
                rlong=rlong,
                relev=float(relev),  # 转换为float以匹配TraceHeader定义
                sxutm=sxutm,
                syutm=syutm,
                sz=sz,
                rxutm=rxutm,
                ryutm=ryutm,
                rz=rz,
                picks=picks
            )
            
            trace_headers.append(trace_header)
    
    return trace_headers


def convert_hdr_to_tx(hfile: str, output_file: str, config: Z2TxConfig, npick: int) -> Tuple[bool, int]:
    """将头文件转换为 tx.in 格式
    
    Args:
        hfile: 输入头文件路径
        output_file: 输出 tx.in 文件路径
        config: 转换配置
        npick: 拾取字数
        
    Returns:
        (是否成功, 拾取总数)
    """
    try:
        # 读取头文件
        trace_headers = read_header_file(hfile, npick)
        
        if not trace_headers:
            return False, 0
        
        # 获取所有唯一的 OBS（炮站号）
        obs_shots = sorted(set(th.ishoti for th in trace_headers))
        
        # 如果没有配置，使用默认值
        default_config = {
            'xmod': 0.0,
            'tshift': 0.0,
            'xshift': 0.0,
            'picku': [0.05] * npick  # 默认不确定性 0.05
        }
        
        total_picks = 0
        
        with open(output_file, 'w') as f:
            # 遍历每个 OBS
            for shot in obs_shots:
                # 获取该 OBS 的配置
                obs_cfg = config.obs_configs.get(shot, default_config)
                xmod = obs_cfg.get('xmod', 0.0)
                tshift = obs_cfg.get('tshift', 0.0)
                xshift = obs_cfg.get('xshift', 0.0)
                picku_list = obs_cfg.get('picku', [0.05] * npick)
                
                # 确保 picku 列表长度正确
                if len(picku_list) < npick:
                    picku_list.extend([0.05] * (npick - len(picku_list)))
                elif len(picku_list) > npick:
                    picku_list = picku_list[:npick]
                
                shot_headers = [th for th in trace_headers if th.ishoti == shot]

                # 处理左侧拾取（offset < 0）
                left_written = False
                for pick_word in range(1, npick + 1):
                    for th in shot_headers:
                        if th.offsti < 0:
                            if th.picks and len(th.picks) >= pick_word and th.picks[pick_word - 1] > 0:
                                if not left_written:
                                    # 写入左侧标记
                                    f.write(f"{xmod:10.3f}{-1.0:10.3f}{0.0:10.3f}{0:10d}\n")
                                    left_written = True

                                # 计算拾取值
                                pick_time = th.picks[pick_word - 1]
                                if config.iamp != 0:
                                    # 振幅模式（使用 amscal，这里简化处理）
                                    pick = pick_time * obs_cfg.get('amscal', 1.0)
                                else:
                                    # 走时模式（使用 tshift）
                                    pick = pick_time + tshift

                                # 计算距离
                                distance = xmod + th.offsti + xshift

                                # 获取不确定性
                                uncertainty = picku_list[pick_word - 1] if pick_word <= len(picku_list) else 0.05

                                # 写入拾取行
                                f.write(f"{distance:10.3f}{pick:10.3f}{uncertainty:10.3f}{pick_word:10d}\n")
                                total_picks += 1
                
                # 处理右侧拾取（offset > 0）
                right_written = False
                for pick_word in range(1, npick + 1):
                    for th in shot_headers:
                        if th.offsti > 0:
                            if th.picks and len(th.picks) >= pick_word and th.picks[pick_word - 1] > 0:
                                if not right_written:
                                    # 写入右侧标记
                                    f.write(f"{xmod:10.3f}{1.0:10.3f}{0.0:10.3f}{0:10d}\n")
                                    right_written = True

                                # 计算拾取值
                                pick_time = th.picks[pick_word - 1]
                                if config.iamp != 0:
                                    pick = pick_time * obs_cfg.get('amscal', 1.0)
                                else:
                                    pick = pick_time + tshift

                                # 计算距离
                                distance = xmod + th.offsti + xshift

                                # 获取不确定性
                                uncertainty = picku_list[pick_word - 1] if pick_word <= len(picku_list) else 0.05

                                # 写入拾取行
                                f.write(f"{distance:10.3f}{pick:10.3f}{uncertainty:10.3f}{pick_word:10d}\n")
                                total_picks += 1
            
            # 写入结束标记
            if total_picks > 0:
                f.write(f"{0.0:10.3f}{0.0:10.3f}{0.0:10.3f}{-1:10d}\n")
        
        return True, total_picks
        
    except Exception as e:
        print(f"转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False, 0
