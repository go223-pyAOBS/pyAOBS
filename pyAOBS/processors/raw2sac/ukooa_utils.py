#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UKOOA文件解析工具模块

提供UKOOA格式炮点文件的解析功能：
1. 单行UKOOA数据解析
2. UKOOA文件解析

便于后续整合到 pyAOBS 代码包中。

Author: Haibo Huang, 2025/01/XX
"""

import os
from typing import Dict, List, Optional

# 从 format_utils 导入字符串工具函数
try:
    from format_utils import get_subs
except ImportError:
    print("Error: Cannot import get_subs from format_utils")
    raise


# ============================================================================
# UKOOA文件解析函数
# ============================================================================

def parse_ukooa_line(line: str, sac_year: int, sac_nzjday: int) -> Optional[Dict]:
    """解析单行UKOOA数据
    
    UKOOA格式字段位置（从1开始计数）：
    - 位置1: 'S'或'N'
    - 位置5-7: 儒略日
    - 位置8-9: 小时
    - 位置10-11: 分钟
    - 位置12-13: 秒
    - 位置15-17: 毫秒
    - 位置20-24: 炮点号
    - 位置26-35: 纬度（度分秒）
    - 位置36-45: 经度（度分秒）
    - 位置65-68: 水深
    - 位置72-75: 野外记录号
    
    Args:
        line: UKOOA文件的一行
        sac_year: SAC文件年份
        sac_nzjday: SAC文件儒略日
    
    Returns:
        炮点信息字典，如果无效返回None
    """
    if len(line) < 75 or (line[0] != 'S' and line[0] != 'N'):
        return None
    
    try:
        # 提取儒略日
        day_str = get_subs(line, 5, 7)
        if not day_str:
            return None
        day = int(day_str)
        
        # 检查是否在SAC文件时间范围内
        if day < sac_nzjday:
            return None
        
        # 提取时间信息
        hour = int(get_subs(line, 8, 9) or "0")
        minute = int(get_subs(line, 10, 11) or "0")
        sec = int(get_subs(line, 12, 13) or "0")
        timbas = int(get_subs(line, 15, 17) or "0")
        
        # 提取炮点号
        shot_number_str = get_subs(line, 20, 24)
        if not shot_number_str:
            return None
        shot_number = int(shot_number_str)
        
        # 解析纬度
        latdeg = int(get_subs(line, 26, 27) or "0")
        latmin = int(get_subs(line, 28, 29) or "0")
        sec1 = int(get_subs(line, 30, 31) or "0")
        sec2 = int(get_subs(line, 33, 34) or "0")
        latsec = sec1 + sec2 / 100.0
        lat = latdeg + latmin / 60.0 + latsec / 3600.0
        if len(line) > 35 and line[35] == 'S':
            lat = -lat
        
        # 解析经度
        londeg = int(get_subs(line, 36, 38) or "0")
        lonmin = int(get_subs(line, 39, 40) or "0")
        sec3 = int(get_subs(line, 41, 42) or "0")
        sec4 = int(get_subs(line, 44, 45) or "0")
        lonsec = sec3 + sec4 / 100.0
        lon = londeg + lonmin / 60.0 + lonsec / 3600.0
        if len(line) > 45 and line[45] == 'W':
            lon = -lon
        
        # 提取水深
        swdep_str = get_subs(line, 65, 68)
        swdep = int(swdep_str) if swdep_str else 0
        
        # 提取野外记录号
        fldr_str = get_subs(line, 72, 75)
        fldr = int(fldr_str) if fldr_str else 0
        
        return {
            'day': day,
            'year': sac_year,
            'hour': hour,
            'minute': minute,
            'sec': sec,
            'msec': timbas,
            'shot_number': shot_number,
            'latitude': lat,
            'longitude': lon,
            'water_depth': swdep,
            'field_record': fldr
        }
    except (ValueError, IndexError):
        return None


def parse_ukooa_file(ukooa_file: str, sac_year: int, sac_nzjday: int) -> List[Dict]:
    """解析UKOOA格式的炮点文件
    
    Args:
        ukooa_file: UKOOA文件路径
        sac_year: SAC文件年份
        sac_nzjday: SAC文件儒略日
    
    Returns:
        炮点信息列表
    """
    shots = []
    
    if not os.path.exists(ukooa_file):
        raise FileNotFoundError(f"Cannot open the UKOOA file: {ukooa_file}")
    
    with open(ukooa_file, 'r') as f:
        for line in f:
            shot_info = parse_ukooa_line(line, sac_year, sac_nzjday)
            if shot_info is not None:
                shots.append(shot_info)
    
    return shots

