"""
pick_manager.py - ZPLOT 拾取管理模块

负责管理地震震相拾取数据，包括添加、删除、保存、加载拾取点
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import struct
import numpy as np


@dataclass
class Pick:
    """单个拾取点"""
    trace_idx: int      # 道索引
    pick_word: int      # 拾取字（1到npick）
    time: float         # 拾取时间（秒）
    ishoti: int         # 炮站号（用于保存）
    itsn: int           # 道序号（用于保存）


class PickManager:
    """拾取管理类
    
    管理地震震相拾取数据，支持：
    - 添加/删除拾取点
    - 保存拾取到文件（zplot.out 格式）
    - 从头文件加载已有拾取
    - 更新头文件中的拾取信息
    """
    
    def __init__(self, npick: int = 10):
        """初始化拾取管理器
        
        Args:
            npick: 拾取字数（最大拾取字编号）
        """
        self.npick = npick
        # 拾取数据结构：{trace_idx: {pick_word: time}}
        self.picks: Dict[int, Dict[int, float]] = {}
        # 道信息：{trace_idx: (ishoti, itsn)}，用于保存时识别道
        self.trace_info: Dict[int, Tuple[int, int]] = {}
    
    def set_trace_info(self, trace_idx: int, ishoti: int, itsn: int):
        """设置道的识别信息
        
        Args:
            trace_idx: 道索引
            ishoti: 炮站号
            itsn: 道序号
        """
        self.trace_info[trace_idx] = (ishoti, itsn)
    
    def set_trace_info_batch(self, trace_headers: List):
        """批量设置道信息并加载已有拾取
        
        Args:
            trace_headers: TraceHeader 列表
        """
        loaded_pick_count = 0
        for i, th in enumerate(trace_headers):
            self.set_trace_info(i, th.ishoti, th.itsn)
            # 同时加载已有的拾取
            if th.picks and len(th.picks) > 0:
                for pick_word_idx, time in enumerate(th.picks):
                    if time > 0:  # 有效拾取
                        pick_word = pick_word_idx + 1  # pick_word 从1开始
                        if self.add_pick(i, time, pick_word):
                            loaded_pick_count += 1
    
    def add_pick(self, trace_idx: int, time: float, pick_word: Optional[int] = None) -> bool:
        """添加拾取点
        
        Args:
            trace_idx: 道索引
            time: 拾取时间（秒）
            pick_word: 拾取字（1到npick），如果为None则使用默认值1
            
        Returns:
            是否成功添加
        """
        if pick_word is None:
            pick_word = 1
        
        if pick_word < 1 or pick_word > self.npick:
            return False
        
        if trace_idx not in self.picks:
            self.picks[trace_idx] = {}
        
        self.picks[trace_idx][pick_word] = time
        return True
    
    def remove_pick(self, trace_idx: int, pick_word: int) -> bool:
        """删除拾取点
        
        Args:
            trace_idx: 道索引
            pick_word: 拾取字
            
        Returns:
            是否成功删除
        """
        if trace_idx not in self.picks:
            return False
        
        if pick_word in self.picks[trace_idx]:
            del self.picks[trace_idx][pick_word]
            # 如果该道没有拾取了，删除整个条目
            if not self.picks[trace_idx]:
                del self.picks[trace_idx]
            return True
        
        return False
    
    def get_pick(self, trace_idx: int, pick_word: int) -> Optional[float]:
        """获取指定道的指定拾取字的时间
        
        Args:
            trace_idx: 道索引
            pick_word: 拾取字
            
        Returns:
            拾取时间（秒），如果不存在则返回None
        """
        if trace_idx not in self.picks:
            return None
        
        return self.picks[trace_idx].get(pick_word)
    
    def get_picks_for_trace(self, trace_idx: int) -> Dict[int, float]:
        """获取指定道的所有拾取
        
        Args:
            trace_idx: 道索引
            
        Returns:
            拾取字典 {pick_word: time}
        """
        return self.picks.get(trace_idx, {}).copy()
    
    def get_picks_by_word(self, pick_word: int) -> Dict[int, float]:
        """获取指定拾取字的所有拾取
        
        Args:
            pick_word: 拾取字（1 到 npick）
            
        Returns:
            拾取字典 {trace_idx: time}，只包含指定拾取字的拾取
        """
        result = {}
        for trace_idx, trace_picks in self.picks.items():
            if pick_word in trace_picks:
                time = trace_picks[pick_word]
                if time > 0:  # 只返回有效拾取（时间 > 0）
                    result[trace_idx] = time
        return result
    
    def get_all_picks(self) -> Dict[int, Dict[int, float]]:
        """获取所有拾取
        
        Returns:
            拾取字典 {trace_idx: {pick_word: time}}
        """
        return {k: v.copy() for k, v in self.picks.items()}
    
    def clear_picks(self, trace_idx: Optional[int] = None):
        """清空拾取
        
        Args:
            trace_idx: 如果指定，只清空该道的拾取；否则清空所有拾取
        """
        if trace_idx is not None:
            if trace_idx in self.picks:
                del self.picks[trace_idx]
        else:
            self.picks.clear()
    
    def save_picks(self, filename: str, format: str = 'zplot') -> bool:
        """保存拾取到文件
        
        Args:
            filename: 输出文件名
            format: 文件格式（'zplot' 或 'json'）
            
        Returns:
            是否成功保存
        """
        try:
            if format == 'zplot':
                return self._save_zplot_format(filename)
            elif format == 'json':
                return self._save_json_format(filename)
            else:
                return False
        except Exception as e:
            print(f"保存拾取失败: {e}")
            return False
    
    def _save_zplot_format(self, filename: str) -> bool:
        """保存为 zplot.out 格式
        
        格式：ishot itsn apick pick_time
        - ishot: 炮站号
        - itsn: 道序号
        - apick: 拾取字（0表示死道标志）
        - pick_time: 拾取时间（秒）
        """
        with open(filename, 'w') as f:
            for trace_idx, trace_picks in self.picks.items():
                if trace_idx not in self.trace_info:
                    continue
                
                ishoti, itsn = self.trace_info[trace_idx]
                
                for pick_word, time in trace_picks.items():
                    if time > 0:  # 只保存有效拾取
                        f.write(f"{ishoti:6d}{itsn:6d}{pick_word:6d}{time:12.3f}\n")
        
        return True
    
    def _save_json_format(self, filename: str) -> bool:
        """保存为 JSON 格式（可选）"""
        import json
        
        data = {
            'npick': self.npick,
            'picks': self.picks,
            'trace_info': {k: {'ishoti': v[0], 'itsn': v[1]} 
                          for k, v in self.trace_info.items()}
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True
    
    def load_picks(self, filename: str, format: str = 'zplot') -> bool:
        """从文件加载拾取
        
        Args:
            filename: 输入文件名
            format: 文件格式（'zplot' 或 'json'）
            
        Returns:
            是否成功加载
        """
        try:
            if format == 'zplot':
                return self._load_zplot_format(filename)
            elif format == 'json':
                return self._load_json_format(filename)
            else:
                return False
        except Exception as e:
            print(f"加载拾取失败: {e}")
            return False
    
    def _load_zplot_format(self, filename: str) -> bool:
        """从 zplot.out 格式加载
        
        格式：ishot itsn apick pick_time
        """
        # 需要反向查找：通过 ishoti 和 itsn 找到 trace_idx
        # 创建反向映射
        reverse_map = {}
        for trace_idx, (ishoti, itsn) in self.trace_info.items():
            key = (ishoti, itsn)
            if key not in reverse_map:
                reverse_map[key] = []
            reverse_map[key].append(trace_idx)
        
        loaded_count = 0
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    
                    ishoti = int(parts[0])
                    itsn = int(parts[1])
                    apick = int(parts[2])
                    pick_time = float(parts[3])
                    
                    # 查找对应的 trace_idx
                    key = (ishoti, itsn)
                    if key in reverse_map:
                        # 如果有多个匹配，使用第一个
                        trace_idx = reverse_map[key][0]
                        
                        if apick == 0:
                            # apick=0 表示死道标志，这里不处理
                            continue
                        elif 1 <= apick <= self.npick:
                            self.add_pick(trace_idx, pick_time, apick)
                            loaded_count += 1
                except (ValueError, IndexError):
                    continue
        
        return loaded_count > 0
    
    def _load_json_format(self, filename: str) -> bool:
        """从 JSON 格式加载（可选）"""
        import json
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        if 'npick' in data:
            self.npick = data['npick']
        
        if 'picks' in data:
            self.picks = {int(k): {int(pw): float(t) for pw, t in v.items()}
                         for k, v in data['picks'].items()}
        
        if 'trace_info' in data:
            self.trace_info = {int(k): (v['ishoti'], v['itsn'])
                              for k, v in data['trace_info'].items()}
        
        return True
    
    def update_header_picks(self, trace_headers: List) -> bool:
        """更新道头中的拾取信息
        
        Args:
            trace_headers: TraceHeader 列表（会被修改）
            
        Returns:
            是否成功更新
        """
        try:
            for i, th in enumerate(trace_headers):
                # 初始化 picks 数组
                if not th.picks:
                    th.picks = [0.0] * self.npick
                elif len(th.picks) != self.npick:
                    # 确保 picks 数组长度正确
                    if len(th.picks) < self.npick:
                        th.picks.extend([0.0] * (self.npick - len(th.picks)))
                    else:
                        th.picks = th.picks[:self.npick]
                
                # 如果该道有拾取，更新拾取；否则清除所有拾取
                if i in self.picks and self.picks[i]:
                    # 先清除所有拾取（设置为0）
                    th.picks = [0.0] * self.npick
                    
                    # 然后更新当前存在的拾取
                    for pick_word, time in self.picks[i].items():
                        if 1 <= pick_word <= self.npick and time > 0:
                            th.picks[pick_word - 1] = time
                else:
                    # 如果该道没有拾取，清除所有拾取
                    th.picks = [0.0] * self.npick
            
            return True
        except Exception as e:
            print(f"更新道头拾取失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_to_header_file(self, hfile: str, trace_headers: List) -> bool:
        """保存拾取到头文件（类似 headup 命令）
        
        改进（阶段2）：拾取信息只保存到头文件，不保存到数据文件
        - 数据文件（.z）：不包含拾取信息（或包含0值占位符）
        - 头文件（.hdr）：包含完整的拾取信息
        
        Args:
            hfile: 头文件路径
            trace_headers: TraceHeader 列表
            
        Returns:
            是否成功保存
        """
        try:
            # 先更新道头中的拾取信息
            if not self.update_header_picks(trace_headers):
                return False
            
            # 以 Fortran 未格式化格式写入头文件
            with open(hfile, 'wb') as f:
                for th in trace_headers:
                    # 计算记录大小（按照C结构体的内存布局：9个int + 13个float = 22个字段，88字节）
                    record_size = 88 + self.npick * 4  # 22个字段（88字节）+ npick个float
                    
                    # 写入记录长度标记
                    f.write(struct.pack('<i', record_size))
                    
                    # 按照C结构体的内存布局写入（混合整数和浮点数）
                    # 字段顺序：nrec(int), itsn(int), ireci(int), itype(int), iflag(int),
                    #           offset(float), azi(float), igain(int), texact(float),
                    #           slat(float), slong(float), selev(int), swdepth(int),
                    #           rlat(float), rlong(float), relev(int),
                    #           sxutm(float), syutm(float), sz(float),
                    #           rxutm(float), ryutm(float), rz(float)
                    
                    # 注意：需要转换单位（offsti 从千米转回米，azi 从度转回分）
                    offsti_m = th.offsti * 1000.0  # 千米转米
                    azi_min = th.azi * 60.0  # 度转分
                    
                    # 按照C结构体布局写入（88字节）
                    # 前5个整数（20字节）
                    f.write(struct.pack('<5i', th.ishoti, th.itsn, th.ireci, th.itypei, th.iflagi))
                    # 2个浮点数（8字节）
                    f.write(struct.pack('<2f', offsti_m, azi_min))
                    # 1个整数（4字节）
                    f.write(struct.pack('<i', th.igaini))
                    # 3个浮点数（12字节）
                    f.write(struct.pack('<3f', th.texact, th.slat, th.slong))
                    # 2个整数（8字节）- 注意：selev, swdepth, relev 在TraceHeader中是float，需要转换为int
                    f.write(struct.pack('<2i', int(th.selev), int(th.swdepth)))
                    # 2个浮点数（8字节）
                    f.write(struct.pack('<2f', th.rlat, th.rlong))
                    # 1个整数（4字节）
                    f.write(struct.pack('<i', int(th.relev)))
                    # 6个浮点数（24字节）
                    f.write(struct.pack('<6f', th.sxutm, th.syutm, th.sz, th.rxutm, th.ryutm, th.rz))
                    
                    # 写入拾取数组（npick个）
                    picks_array = th.picks if th.picks else [0.0] * self.npick
                    # 确保 picks 数组长度正确
                    if len(picks_array) < self.npick:
                        picks_array.extend([0.0] * (self.npick - len(picks_array)))
                    elif len(picks_array) > self.npick:
                        picks_array = picks_array[:self.npick]
                    
                    f.write(struct.pack(f'<{self.npick}f', *picks_array))
                    
                    # 写入记录长度标记（结束）
                    f.write(struct.pack('<i', record_size))
            
            return True
        except Exception as e:
            print(f"保存头文件失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def count_picks(self) -> int:
        """统计拾取点总数
        
        Returns:
            拾取点总数
        """
        return sum(len(picks) for picks in self.picks.values())
    
    def get_statistics(self) -> Dict[str, any]:
        """获取拾取统计信息
        
        Returns:
            统计信息字典
        """
        total_picks = sum(len(picks) for picks in self.picks.values())
        traces_with_picks = len(self.picks)
        
        pick_word_counts = {}
        for trace_picks in self.picks.values():
            for pick_word in trace_picks.keys():
                pick_word_counts[pick_word] = pick_word_counts.get(pick_word, 0) + 1
        
        return {
            'total_picks': total_picks,
            'traces_with_picks': traces_with_picks,
            'pick_word_counts': pick_word_counts,
            'npick': self.npick
        }
