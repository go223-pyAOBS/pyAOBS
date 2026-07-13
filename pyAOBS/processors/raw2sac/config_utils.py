#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置文件读取工具模块

提供统一配置文件（INI格式）的读取功能：
- 支持多个配置节（section）
- 自动类型转换
- 时间字符串保持为字符串

便于后续整合到 pyAOBS 代码包中。

Author: Haibo Huang, 2025/01/XX
"""

import os
from typing import Dict


def read_unified_config_file(config_file: str) -> Dict:
    """读取统一配置文件（INI格式）
    
    配置文件格式（参考obem_config_example.ini）：
    [Parameters]
    rv = 8.0
    length = 6.0
    tcoor = 2.0
    lon0 = 120.0
    ex = 0.0
    ey = 0.0
    et = 1.885
    
    [Drift]
    start_time = 2017/02/27 21:18:54
    end_time = 2017/06/12 02:05:00
    drift = 123.456
    
    [Location]
    longitude = 120.3
    latitude = 20.5
    location_file = loc.in
    
    Args:
        config_file: 配置文件路径
    
    Returns:
        配置字典，包含各个节（section）的配置
        例如：{'parameters': {...}, 'drift': {...}, 'location': {...}}
    """
    config = {}
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    current_section = None
    
    with open(config_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 跳过注释和空行
            if not line or line.startswith('#'):
                continue
            
            # 解析节名
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1].lower()
                config[current_section] = {}
                continue
            
            # 解析键值对
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # 转换数据类型（保持时间字符串为字符串）
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif '/' in value or ':' in value:
                    # 包含斜杠或冒号的可能是时间字符串，保持为字符串
                    pass
                elif value.replace('.', '', 1).replace('-', '', 1).isdigit():
                    value = float(value) if '.' in value else int(value)
                
                if current_section:
                    config[current_section][key.lower()] = value
                else:
                    config[key.lower()] = value
    
    return config

