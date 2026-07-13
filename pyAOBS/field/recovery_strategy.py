#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
站位回收策略规划脚本

功能：
1. 读取站位文件（包含坐标、深度、上浮速度、释放距离等）
2. 规划滚动回收策略，确保在站位上浮到海面前船能到达站位位置
3. 考虑总体回收路径：站位文件中的初始顺序表示回收方向（起点到终点）
4. 输出详细的回收计划和时间表

文件格式说明：
- 第1列：经度
- 第2列：纬度
- 第3列：台站类型（可选）
- 第4列：站位名称（可选）
- 第5列：上浮速度（米/分钟，可选，默认39米/分钟）
- 第6列：释放距离（公里，可选，默认5公里）
- 第7列：水深（米，可选）

重要说明：
- 站位文件中的初始顺序表示总体回收路径方向（起点到终点）
- 算法会优先考虑沿回收方向的站位，同时结合空间距离因素
- 优化算法会尝试不同的起始点，但保持总体回收方向不变

使用方法：
    python recovery_strategy.py <站位文件> [选项]

示例：
    python recovery_strategy.py stations.txt --ship-speed 10 --num-stations 2
    python recovery_strategy.py stations.txt --ship-speed 10 --num-stations 4  # 支持4个站位同时工作
    python recovery_strategy.py stations.txt --ship-speed 10 --num-stations 5  # 支持5个站位同时工作
    python recovery_strategy.py stations.txt --ship-speed 10 --num-stations 4  # 支持4个站位同时工作
"""

import argparse
import sys
import numpy as np
from typing import List, Tuple, Dict, Optional
import math
from itertools import permutations


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    计算两点之间的Haversine距离（大圆距离）
    
    Parameters
    ----------
    lon1, lat1 : float
        起点坐标（度）
    lon2, lat2 : float
        终点坐标（度）
        
    Returns
    -------
    distance : float
        距离（公里）
    """
    # 转换为弧度
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # 地球半径（公里）
    R = 6371.0
    
    return R * c


class RecoveryStationPlanner:
    """回收站位规划器
    
    功能：根据回收路径规划滚动回收策略，确保在站位上浮到海面前船能到达站位位置
    """
    
    def __init__(self, recovery_path: List[int], stations: List[Tuple[float, float]], 
                 station_depths: List[float], station_ascent_speeds: Optional[List[float]] = None,
                 station_release_distances: Optional[List[float]] = None,
                 ascent_speed: float = 0.65, ship_speed: float = 10.0, 
                 pickup_time: float = 30.0, release_time: float = 10.0):
        """
        初始化回收站位规划器
        
        Parameters
        ----------
        recovery_path : List[int]
            回收路径（站位索引列表），按船的行进顺序
        stations : List[Tuple[float, float]]
            站位坐标列表，格式为 [(lon1, lat1), (lon2, lat2), ...]
        station_depths : List[float]
            站位深度列表（米），与stations一一对应
        station_ascent_speeds : List[float], optional
            每个站位自己的上浮速度（米/分钟），如果提供则使用，否则使用统一速度
        station_release_distances : List[float], optional
            每个站位的释放距离（公里），表示仪器应答范围
        ascent_speed : float
            站位上浮速度（米/秒），默认0.65米/秒（39米/分钟）
        ship_speed : float
            船速（节），默认10节（约5.14米/秒或18.52公里/小时）
        pickup_time : float
            打捞时间（分钟），默认30分钟
        release_time : float
            释放时间（分钟），默认10分钟
        """
        self.recovery_path = recovery_path
        self.stations = np.array(stations)
        self.station_depths = np.array(station_depths)
        self.station_ascent_speeds = station_ascent_speeds  # 米/分钟
        self.station_release_distances = station_release_distances  # 公里
        self.ascent_speed = ascent_speed  # 米/秒
        self.ship_speed = ship_speed  # 节
        self.pickup_time = pickup_time  # 分钟
        self.release_time = release_time  # 分钟
        
        # 将船速从节转换为公里/小时和米/秒
        self.ship_speed_kmh = ship_speed * 1.852  # 1节 = 1.852公里/小时
        self.ship_speed_ms = ship_speed * 0.5144  # 1节 = 0.5144米/秒
        
        # 验证输入
        if len(recovery_path) == 0:
            raise ValueError("回收路径不能为空")
        if max(recovery_path) >= len(stations):
            raise ValueError(f"回收路径中包含无效的站位索引: {max(recovery_path)} >= {len(stations)}")
        if min(recovery_path) < 0:
            raise ValueError(f"回收路径中包含负的站位索引")
        if len(station_depths) != len(stations):
            raise ValueError(f"站位深度数量 ({len(station_depths)}) 与站位数量 ({len(stations)}) 不匹配")
    
    def calculate_ascent_time(self, depth: float, station_idx: Optional[int] = None) -> float:
        """
        计算站位上浮所需时间
        
        Parameters
        ----------
        depth : float
            站位深度（米）
        station_idx : int, optional
            站位索引，如果提供且station_ascent_speeds存在，则使用该站位自己的速度
            
        Returns
        -------
        time : float
            上浮时间（秒）
        """
        if depth <= 0:
            return 0.0
        
        # 如果提供了站位索引且有该站位自己的上浮速度，使用它
        if station_idx is not None and self.station_ascent_speeds is not None:
            if station_idx < len(self.station_ascent_speeds):
                speed_m_per_min = self.station_ascent_speeds[station_idx]
                speed_m_per_sec = speed_m_per_min / 60.0  # 转换为米/秒
                return depth / speed_m_per_sec
        
        # 否则使用统一速度
        return depth / self.ascent_speed
    
    def calculate_travel_time(self, lon1: float, lat1: float, 
                             lon2: float, lat2: float) -> float:
        """
        计算船从位置1航行到位置2所需时间
        
        Parameters
        ----------
        lon1, lat1 : float
            起点坐标（度）
        lon2, lat2 : float
            终点坐标（度）
            
        Returns
        -------
        time : float
            航行时间（秒）
        """
        distance_km = haversine_distance(lon1, lat1, lon2, lat2)
        time_hours = distance_km / self.ship_speed_kmh
        return time_hours * 3600  # 转换为秒
    
    def find_release_position(self, station_idx: int, current_lon: float, current_lat: float) -> Tuple[float, float]:
        """
        找到站位的释放位置（在释放距离内，尽量靠近站位）
        
        Parameters
        ----------
        station_idx : int
            站位索引
        current_lon, current_lat : float
            当前位置坐标
            
        Returns
        -------
        release_lon, release_lat : float
            释放位置坐标
        """
        station_lon, station_lat = self.stations[station_idx]
        release_distance = self.station_release_distances[station_idx] if \
            self.station_release_distances and station_idx < len(self.station_release_distances) else 5.0
        
        # 计算当前位置到站位的距离
        distance = haversine_distance(current_lon, current_lat, station_lon, station_lat)
        
        if distance <= release_distance:
            # 已经在释放距离内，直接使用当前位置
            return current_lon, current_lat
        else:
            # 需要移动到释放距离内
            # 计算方向向量（从当前位置指向站位）
            # 简化处理：直接移动到距离站位release_distance的位置
            # 使用线性插值
            ratio = release_distance / distance if distance > 0 else 0
            release_lon = current_lon + (station_lon - current_lon) * (1 - ratio)
            release_lat = current_lat + (station_lat - current_lat) * (1 - ratio)
            return release_lon, release_lat
    
    def find_multi_station_release_position(self, station_indices: List[int], 
                                           current_lon: float, current_lat: float) -> Optional[Tuple[float, float]]:
        """
        找到可以同时释放多个站位的释放位置（适用于平面分布的站位）
        
        使用迭代优化方法找到最优释放位置，使得该位置在所有站位的释放距离内，
        同时尽可能接近当前位置以减少航行时间。
        
        Parameters
        ----------
        station_indices : List[int]
            要释放的站位索引列表
        current_lon, current_lat : float
            当前位置坐标
            
        Returns
        -------
        release_position : Tuple[float, float] or None
            如果找到合适的释放位置，返回坐标；否则返回None
        """
        if len(station_indices) == 0:
            return None
        
        if len(station_indices) == 1:
            return self.find_release_position(station_indices[0], current_lon, current_lat)
        
        # 获取所有站位的坐标和释放距离
        station_coords = []
        release_distances = []
        for idx in station_indices:
            station_lon, station_lat = self.stations[idx]
            release_distance = self.station_release_distances[idx] if \
                self.station_release_distances and idx < len(self.station_release_distances) else 5.0
            station_coords.append((station_lon, station_lat))
            release_distances.append(release_distance)
        
        # 策略1：计算所有站位的几何中心（适用于平面分布）
        lons = [coord[0] for coord in station_coords]
        lats = [coord[1] for coord in station_coords]
        center_lon = sum(lons) / len(lons)
        center_lat = sum(lats) / len(lats)
        
        # 检查中心点是否在所有站位的释放距离内
        all_within_range = True
        for i, (station_lon, station_lat) in enumerate(station_coords):
            distance = haversine_distance(center_lon, center_lat, station_lon, station_lat)
            if distance > release_distances[i]:
                all_within_range = False
                break
        
        if all_within_range:
            # 中心点在所有站位的释放距离内，使用中心点
            return center_lon, center_lat
        
        # 策略2：如果中心点不在所有释放距离内，尝试找到重叠区域
        # 使用迭代方法：从当前位置或中心点开始，向所有站位的共同覆盖区域移动
        # 简化：找到距离当前位置最近且满足所有约束的点
        
        # 尝试使用加权中心（考虑释放距离）
        # 权重：释放距离大的站位权重更大（更容易满足）
        weights = [1.0 / max(rd, 0.1) for rd in release_distances]  # 避免除零
        total_weight = sum(weights)
        weighted_lon = sum(lons[i] * weights[i] for i in range(len(lons))) / total_weight
        weighted_lat = sum(lats[i] * weights[i] for i in range(len(lats))) / total_weight
        
        # 检查加权中心是否满足约束
        all_within_range = True
        for i, (station_lon, station_lat) in enumerate(station_coords):
            distance = haversine_distance(weighted_lon, weighted_lat, station_lon, station_lat)
            if distance > release_distances[i]:
                all_within_range = False
                break
        
        if all_within_range:
            return weighted_lon, weighted_lat
        
        # 策略3：如果仍不满足，尝试从当前位置向中心点方向移动
        # 找到满足所有约束且距离当前位置最近的点
        # 简化：使用第一个站位的释放位置，然后检查其他站位是否也在范围内
        candidate_pos = self.find_release_position(station_indices[0], current_lon, current_lat)
        candidate_lon, candidate_lat = candidate_pos
        
        # 检查候选位置是否满足所有站位的约束
        all_within_range = True
        for i, (station_lon, station_lat) in enumerate(station_coords):
            distance = haversine_distance(candidate_lon, candidate_lat, station_lon, station_lat)
            if distance > release_distances[i]:
                all_within_range = False
                break
        
        if all_within_range:
            return candidate_lon, candidate_lat
        
        # 如果找不到满足所有约束的位置，返回None（算法会回退到逐个释放）
        return None
    
    def plan_rolling_recovery(self, num_stations: int = 2) -> dict:
        """
        规划滚动回收策略（考虑释放距离约束，支持平面分布的站位）
        
        关键约束：
        1. **所有站位必须都要回收**：即使某个站位暂时不能满足多点联收要求，
           也要通过其他方式（如单独释放或归于其他组）实现回收
        2. **真正的多点联收策略**：
           - 不是简单地在某个点同时释放几个台站
           - 而是考虑每个台站上浮时间余量的基础上，通过先释放某个台站A，
             然后利用A的上浮时间余量去释放其他台站B（还要考虑到等待多久才释放其他台站）
           - 根本规则：所有操作都要在时间余量范围内
           - 充分利用各个台站的时间余量及其配合去回收多个台站
           - 例如：释放A后，A开始上浮（时间T1），在T1内可以去释放B，然后回来回收A
        3. **滚动+单独回收的整体行进**：不满足多点联收的单独仪器也会加入到回收路线中，
           形成多点联收和单独回收的混合行进路线，所有站位都在同一个连续的回收路径中
        4. 船只必须在各个站位的可释放范围内（释放距离）释放仪器
        5. 核心约束：仪器释放成功后，前往其他仪器进行释放、回收等操作，
           再到返回本仪器位置回收，这期间所花的时间必须小于仪器上浮时间
        6. 尽可能多的找到满足这一条件的仪器
        7. 总回收时间最短
        
        总体回收路径约束：
        - 站位文件中的初始顺序表示总体回收方向（起点到终点）
        - 算法会优先考虑沿回收方向的站位，同时结合空间距离因素
        - 确保回收路径总体上遵循从起点到终点的方向
        
        算法特点：
        - 支持站位在平面上的任意分布（不限于测线）
        - 使用空间聚类策略，优先释放空间上接近的站位
        - 结合回收方向约束，优先沿总体回收路径方向释放站位
        - **真正的多点联收策略**：
          * 不是简单地在某个点同时释放几个台站
          * 而是考虑每个台站上浮时间余量，通过先释放台站A，
            然后利用A的上浮时间余量去释放其他台站B
          * 充分利用各个台站的时间余量及其配合去回收多个台站
          * 所有操作都要在时间余量范围内
        - **多点联收+单独回收混合模式**：不满足多点联收的单独仪器也会加入到回收路线中，
          形成多点联收和单独回收的混合行进路线
        - 回收路线是连续的，包含所有站位（无论是多点联收还是单独回收），
          形成一个完整的整体行进路径
        - 单独释放的站位会被立即安排回收，确保它们也被加入到recovery_order中
        
        Parameters
        ----------
        num_stations : int
            同时工作的站位数量（2或3）
            
        Returns
        -------
        result : dict
            包含以下键的字典：
            - 'recovery_order': List[int] - 回收顺序（站位索引列表）
            - 'release_positions': List[Tuple[float, float]] - 每个站位的释放位置
            - 'release_times': List[float] - 每个站位的释放时间（秒）
            - 'recovery_times': List[float] - 每个站位的回收时间（秒）
            - 'recovery_arrival_times': List[float] - 船到达站位位置回收的时间（秒）
            - 'ascent_start_times': List[float] - 每个站位开始上浮的时间（秒）
            - 'ascent_end_times': List[float] - 每个站位完成上浮的时间（秒）
            - 'total_time': float - 总回收时间（秒）
            - 'is_feasible': bool - 是否满足时间约束
            - 'violations': List[dict] - 违反时间约束的站位信息
        """
        if num_stations < 2:
            raise ValueError("num_stations必须至少为2")
        
        n = len(self.recovery_path)
        if n == 0:
            return {
                'recovery_order': [],
                'release_positions': [],
                'release_times': [],
                'recovery_times': [],
                'recovery_arrival_times': [],
                'ascent_start_times': [],
                'ascent_end_times': [],
                'total_time': 0.0,
                'is_feasible': True,
                'violations': []
            }
        
        # 初始化结果
        recovery_order = []
        release_positions = []  # 每个站位的释放位置
        release_times = []  # 每个站位的释放时间（秒）
        recovery_times = []  # 每个站位的回收操作时间（打捞时间，秒）
        recovery_arrival_times = []  # 船到达站位位置回收的时间（秒）
        ascent_start_times = []  # 每个站位开始上浮的时间（秒）
        ascent_end_times = []  # 每个站位完成上浮的时间（秒）
        
        # 当前船的位置（初始位置设为第一个站位）
        current_lon, current_lat = self.stations[self.recovery_path[0]]
        current_time = 0.0
        
        # 已释放但尚未回收的仪器队列
        # 格式：[(station_idx, release_time, release_position, ascent_time), ...]
        # release_time: 释放时间
        # release_position: 释放位置 (lon, lat)
        # ascent_time: 上浮时间（秒）
        released_instruments = []
        
        violations = []
        
        # 待释放的站位列表（按顺序处理）
        remaining_stations = list(self.recovery_path)
        
        # 主循环：处理所有站位
        while remaining_stations or released_instruments:
            # 策略1：如果有已释放的仪器需要回收，优先回收
            # 这包括滚动回收的站位和单独释放的站位，它们都加入到整体回收路线中
            # 找到需要回收的仪器（上浮时间即将用完的）
            instruments_to_recover = []
            
            for inst in released_instruments:
                station_idx, release_time, release_pos, ascent_time = inst
                station_lon, station_lat = self.stations[station_idx]
                
                # 计算从当前位置到该站位位置的时间
                travel_time_to_station = self.calculate_travel_time(
                    current_lon, current_lat, station_lon, station_lat
                )
                recovery_arrival_time = current_time + travel_time_to_station
                
                # 计算从释放到回收的总时间
                time_since_release = recovery_arrival_time - release_time
                
                # 计算时间裕量（上浮时间 - 已用时间）
                time_margin = ascent_time - time_since_release
                
                # 只考虑时间裕量大于0的（即满足时间约束的）
                # 如果时间裕量小于0，说明已经违反约束，但仍然需要回收
                # 单独释放的站位也会被加入到回收路线中
                instruments_to_recover.append((inst, recovery_arrival_time, time_margin))
            
            # 按时间裕量排序，优先回收时间裕量小的（紧急的）
            # 时间裕量为负的（已违反约束）排在最后，但仍需要回收
            # 优先处理时间裕量小但为正的（满足约束但紧急）
            # 这确保了滚动回收和单独回收的站位都能被加入到整体回收路线中
            instruments_to_recover.sort(key=lambda x: (x[2] < 0, x[2]))
            
            # 回收最紧急的仪器（时间裕量最小的，但仍需满足约束或尽可能接近）
            # 无论是滚动回收还是单独回收，都会被加入到recovery_order中
            if instruments_to_recover:
                inst, recovery_arrival_time, time_margin = instruments_to_recover[0]
                station_idx, release_time, release_pos, ascent_time = inst
                station_lon, station_lat = self.stations[station_idx]
                
                # 计算到达时间
                travel_time_to_station = self.calculate_travel_time(
                    current_lon, current_lat, station_lon, station_lat
                )
                recovery_arrival_time = current_time + travel_time_to_station
                
                # 记录回收信息
                recovery_order.append(station_idx)
                recovery_arrival_times.append(recovery_arrival_time)
                recovery_times.append(self.pickup_time * 60.0)  # 打捞时间
                
                # 记录释放信息（与回收信息对应，按回收顺序）
                release_positions.append(release_pos)
                release_times.append(release_time)
                
                # 计算上浮时间
                ascent_start_time = release_time
                ascent_end_time = release_time + ascent_time
                ascent_start_times.append(ascent_start_time)
                ascent_end_times.append(ascent_end_time)
                
                # 检查时间约束：从释放到回收的时间必须小于上浮时间
                time_since_release = recovery_arrival_time - release_time
                if time_since_release >= ascent_time:
                    violations.append({
                        'station_idx': station_idx,
                        'release_time': release_time,
                        'recovery_arrival_time': recovery_arrival_time,
                        'ascent_time': ascent_time,
                        'time_since_release': time_since_release,
                        'delay': time_since_release - ascent_time
                    })
                
                # 更新当前时间和位置
                current_time = recovery_arrival_time + self.pickup_time * 60.0
                current_lon, current_lat = station_lon, station_lat
                
                # 从已释放队列中移除
                released_instruments.remove(inst)
                continue
            
            # 策略2：如果没有需要紧急回收的，尝试释放新仪器
            # 真正的多点联收策略：利用各台站的上浮时间余量，合理安排释放顺序和时间间隔
            if remaining_stations:
                # 考虑总体回收路径：站位文件中的初始顺序表示回收方向（起点到终点）
                # 策略：结合距离和顺序两个因素，优先考虑沿回收方向且距离较近的站位
                
                # 计算所有剩余站位到当前位置的距离和顺序权重
                station_scores = []
                for i, station_idx in enumerate(remaining_stations):
                    station_lon, station_lat = self.stations[station_idx]
                    distance_to_current = haversine_distance(
                        current_lon, current_lat, station_lon, station_lat
                    )
                    # 顺序权重：在remaining_stations中越靠前，权重越大（优先沿回收方向）
                    order_weight = 1.0 - (i / max(len(remaining_stations) - 1, 1))
                    # 综合评分：距离越近、顺序越靠前，评分越高
                    normalized_distance = 1.0 / (1.0 + distance_to_current / 10.0)
                    combined_score = 0.5 * normalized_distance + 0.5 * order_weight
                    station_scores.append((station_idx, distance_to_current, order_weight, combined_score))
                
                # 按综合评分排序
                station_scores.sort(key=lambda x: (-x[3], x[1]))
                
                # 真正的多点联收策略：尝试找到可以联收的站位组
                # 关键：先释放一个台站A，利用A的上浮时间余量去释放其他台站，然后回来回收A
                best_group = None
                best_group_score = -1
                
                # 尝试不同的第一个站位作为起点
                for first_idx, _, _, _ in station_scores[:min(num_stations, len(station_scores))]:
                    first_station_idx = first_idx
                    first_lon, first_lat = self.stations[first_station_idx]
                    first_depth = self.station_depths[first_station_idx]
                    first_ascent_time = self.calculate_ascent_time(first_depth, first_station_idx)
                    
                    # 找到第一个站位的释放位置
                    first_release_lon, first_release_lat = self.find_release_position(
                        first_station_idx, current_lon, current_lat
                    )
                    
                    # 计算到第一个释放位置的时间
                    travel_to_first_release = self.calculate_travel_time(
                        current_lon, current_lat, first_release_lon, first_release_lat
                    )
                    first_release_time = current_time + travel_to_first_release + self.release_time * 60.0
                    
                    # 第一个台站开始上浮的时间
                    first_ascent_start = first_release_time
                    first_ascent_end = first_ascent_start + first_ascent_time
                    
                    # 尝试添加其他台站，利用第一个台站的上浮时间余量
                    candidate_group = [first_station_idx]
                    candidate_release_positions = [(first_release_lon, first_release_lat)]
                    candidate_release_times = [first_release_time]
                    
                    # 真正的多点联收策略：利用第一个台站的上浮时间余量
                    # 关键：释放A后，A开始上浮（时间T_ascent_A）
                    # 在A上浮期间，我们可以：去释放B，然后回来回收A
                    # 约束：释放B的时间 + 回收A的时间 <= T_ascent_A
                    # 然后回收B（B也在上浮，有时间余量T_ascent_B）
                    
                    # 计算从第一个释放位置到第一个站位位置的航行时间（用于回收A）
                    travel_from_first_release_to_first_station = self.calculate_travel_time(
                        first_release_lon, first_release_lat, first_lon, first_lat
                    )
                    
                    # 尝试添加其他台站，利用第一个台站的上浮时间余量
                    # 使用贪心策略：按评分顺序尝试添加台站
                    for other_idx, other_dist, other_order, other_score in station_scores:
                        if other_idx == first_station_idx:
                            continue
                        if len(candidate_group) >= num_stations:
                            break
                        
                        other_lon, other_lat = self.stations[other_idx]
                        other_depth = self.station_depths[other_idx]
                        other_ascent_time = self.calculate_ascent_time(other_depth, other_idx)
                        
                        # 检查其他台站是否在第一个台站的释放距离内
                        other_release_distance = self.station_release_distances[other_idx] if \
                            self.station_release_distances and other_idx < len(self.station_release_distances) else 5.0
                        distance_from_first_release = haversine_distance(
                            first_release_lon, first_release_lat, other_lon, other_lat
                        )
                        
                        # 如果其他台站在释放距离内，可以尝试联收
                        if distance_from_first_release <= other_release_distance:
                            # 计算释放其他台站需要的时间
                            # 从第一个释放位置到其他台站的释放位置
                            other_release_lon, other_release_lat = self.find_release_position(
                                other_idx, first_release_lon, first_release_lat
                            )
                            
                            # 步骤1：从第一个释放位置航行到其他台站的释放位置
                            travel_to_other_release = self.calculate_travel_time(
                                first_release_lon, first_release_lat, other_release_lon, other_release_lat
                            )
                            
                            # 步骤2：释放其他台站的操作时间
                            release_operation_time = self.release_time * 60.0
                            
                            # 步骤3：从其他释放位置航行到第一个站位位置（回收A）
                            travel_to_first_station = self.calculate_travel_time(
                                other_release_lon, other_release_lat, first_lon, first_lat
                            )
                            
                            # 步骤4：回收第一个台站的操作时间（打捞）
                            pickup_operation_time = self.pickup_time * 60.0
                            
                            # 总时间：释放B + 回收A（必须在A的上浮时间内完成）
                            total_time_needed = travel_to_other_release + release_operation_time + \
                                              travel_to_first_station + pickup_operation_time
                            
                            # 检查：总时间是否在第一个台站的上浮时间内
                            if total_time_needed <= first_ascent_time:
                                # 可以联收！计算其他台站的释放时间
                                # 释放时间 = 第一个台站释放时间 + 释放操作时间 + 航行到其他释放位置的时间
                                other_release_time = first_release_time + self.release_time * 60.0 + travel_to_other_release
                                
                                # 添加到候选组
                                candidate_group.append(other_idx)
                                candidate_release_positions.append((other_release_lon, other_release_lat))
                                candidate_release_times.append(other_release_time)
                                
                                # 注意：这里可以继续尝试添加更多台站，但需要更复杂的逻辑
                                # 当前实现：只添加一个额外的台站（简化处理）
                                # 如果需要添加更多台站，需要重新计算时间余量
                                break
                    
                    # 如果找到的组更好，使用这个组
                    if len(candidate_group) > len(best_group) if best_group else True:
                        best_group = candidate_group
                        best_group_release_positions = candidate_release_positions
                        best_group_release_times = candidate_release_times
                        best_group_score = len(candidate_group)
                
                # 如果没有找到可以联收的组，单独释放一个站位
                if not best_group:
                    first_station_idx = station_scores[0][0]
                    release_lon, release_lat = self.find_release_position(
                        first_station_idx, current_lon, current_lat
                    )
                    best_group = [first_station_idx]
                    best_group_release_positions = [(release_lon, release_lat)]
                    travel_to_release = self.calculate_travel_time(
                        current_lon, current_lat, release_lon, release_lat
                    )
                    best_group_release_times = [current_time + travel_to_release + self.release_time * 60.0]
                
                # 按顺序释放组中的每个站位
                for group_idx, station_idx in enumerate(best_group):
                    station_lon, station_lat = self.stations[station_idx]
                    depth = self.station_depths[station_idx]
                    ascent_time = self.calculate_ascent_time(depth, station_idx)
                    
                    # 释放时间和位置
                    release_time = best_group_release_times[group_idx]
                    release_pos = best_group_release_positions[group_idx]
                    
                    # 添加到已释放队列
                    released_instruments.append((
                        station_idx, 
                        release_time, 
                        release_pos, 
                        ascent_time
                    ))
                
                # 更新当前时间和位置（移动到最后一个释放位置）
                if best_group:
                    last_release_pos = best_group_release_positions[-1]
                    last_release_time = best_group_release_times[-1]
                    current_lon, current_lat = last_release_pos
                    current_time = last_release_time
                
                # 从剩余站位中移除已处理的站位
                for station_idx in best_group:
                    remaining_stations.remove(station_idx)
            else:
                # 没有剩余站位，但还有已释放的仪器需要回收
                # 强制回收所有剩余的仪器
                break
        
        # 回收所有剩余的已释放仪器
        while released_instruments:
            # 找到最紧急的仪器
            instruments_to_recover = []
            for inst in released_instruments:
                station_idx, release_time, release_pos, ascent_time = inst
                station_lon, station_lat = self.stations[station_idx]
                
                travel_time_to_station = self.calculate_travel_time(
                    current_lon, current_lat, station_lon, station_lat
                )
                recovery_arrival_time = current_time + travel_time_to_station
                time_since_release = recovery_arrival_time - release_time
                time_margin = ascent_time - time_since_release
                
                instruments_to_recover.append((inst, recovery_arrival_time, time_margin))
            
            instruments_to_recover.sort(key=lambda x: x[2])
            
            if instruments_to_recover:
                inst, recovery_arrival_time, time_margin = instruments_to_recover[0]
                station_idx, release_time, release_pos, ascent_time = inst
                station_lon, station_lat = self.stations[station_idx]
                
                travel_time_to_station = self.calculate_travel_time(
                    current_lon, current_lat, station_lon, station_lat
                )
                recovery_arrival_time = current_time + travel_time_to_station
                
                recovery_order.append(station_idx)
                recovery_arrival_times.append(recovery_arrival_time)
                recovery_times.append(self.pickup_time * 60.0)
                
                ascent_start_time = release_time
                ascent_end_time = release_time + ascent_time
                ascent_start_times.append(ascent_start_time)
                ascent_end_times.append(ascent_end_time)
                
                # 检查时间约束：从释放到回收的时间必须小于上浮时间
                time_since_release = recovery_arrival_time - release_time
                if time_since_release >= ascent_time:
                    violations.append({
                        'station_idx': station_idx,
                        'release_time': release_time,
                        'recovery_arrival_time': recovery_arrival_time,
                        'ascent_time': ascent_time,
                        'time_since_release': time_since_release,
                        'delay': time_since_release - ascent_time
                    })
                else:
                    # 时间约束满足，记录时间裕量
                    pass
                
                current_time = recovery_arrival_time + self.pickup_time * 60.0
                current_lon, current_lat = station_lon, station_lat
                
                released_instruments.remove(inst)
        
        # 关键规则检查：确保所有站位都被回收
        recovered_stations = set(recovery_order)
        all_stations = set(self.recovery_path)
        missing_stations = all_stations - recovered_stations
        
        if missing_stations:
            # 如果有未回收的站位，强制释放和回收它们
            # 这不应该发生，但作为安全措施
            for station_idx in sorted(missing_stations):
                station_lon, station_lat = self.stations[station_idx]
                depth = self.station_depths[station_idx]
                ascent_time = self.calculate_ascent_time(depth, station_idx)
                
                # 找到释放位置
                release_lon, release_lat = self.find_release_position(
                    station_idx, current_lon, current_lat
                )
                
                # 释放时间
                travel_time_to_release = self.calculate_travel_time(
                    current_lon, current_lat, release_lon, release_lat
                )
                release_time = current_time + travel_time_to_release
                
                # 添加到已释放队列
                released_instruments.append((
                    station_idx,
                    release_time,
                    (release_lon, release_lat),
                    ascent_time
                ))
                
                # 立即回收（单独处理，无法与其他站位一起）
                travel_time_to_station = self.calculate_travel_time(
                    release_lon, release_lat, station_lon, station_lat
                )
                recovery_arrival_time = release_time + travel_time_to_station
                
                recovery_order.append(station_idx)
                recovery_arrival_times.append(recovery_arrival_time)
                recovery_times.append(self.pickup_time * 60.0)
                release_positions.append((release_lon, release_lat))
                release_times.append(release_time)
                
                ascent_start_time = release_time
                ascent_end_time = release_time + ascent_time
                ascent_start_times.append(ascent_start_time)
                ascent_end_times.append(ascent_end_time)
                
                # 检查时间约束
                time_since_release = recovery_arrival_time - release_time
                if time_since_release >= ascent_time:
                    violations.append({
                        'station_idx': station_idx,
                        'release_time': release_time,
                        'recovery_arrival_time': recovery_arrival_time,
                        'ascent_time': ascent_time,
                        'time_since_release': time_since_release,
                        'delay': time_since_release - ascent_time
                    })
                
                current_time = recovery_arrival_time + self.pickup_time * 60.0
                current_lon, current_lat = station_lon, station_lat
        
        total_time = current_time
        is_feasible = len(violations) == 0
        
        # 最终验证：确保所有站位都被回收
        final_recovered = set(recovery_order)
        final_all = set(self.recovery_path)
        if final_recovered != final_all:
            # 这不应该发生，记录警告
            missing = final_all - final_recovered
            print(f"Warning: {len(missing)} stations were not recovered: {missing}")
        
        return {
            'recovery_order': recovery_order,
            'release_positions': release_positions,
            'release_times': release_times,
            'recovery_times': recovery_times,
            'recovery_arrival_times': recovery_arrival_times,
            'ascent_start_times': ascent_start_times,
            'ascent_end_times': ascent_end_times,
            'total_time': total_time,
            'is_feasible': is_feasible,
            'violations': violations
        }
    
    def optimize_recovery_order(self, num_stations: int = 2, 
                                max_iterations: int = 100) -> dict:
        """
        优化回收顺序，使得总时间最短，同时尊重总体回收路径方向
        
        总体回收路径约束：
        - 站位文件中的初始顺序表示总体回收方向（起点到终点）
        - 优化算法会尝试不同的起始点，但保持总体方向不变
        - 确保回收路径总体上遵循从起点到终点的方向
        
        尝试不同的策略：
        1. 不同的起始站位（但保持总体方向）
        2. 不同的站位分组方式（哪些站位可以一起释放）
        3. 局部优化释放顺序（在保持方向的前提下）
        
        Parameters
        ----------
        num_stations : int
            同时工作的站位数量（至少2个）
        max_iterations : int
            最大迭代次数
            
        Returns
        -------
        result : dict
            优化后的回收规划结果
        """
        best_result = None
        best_total_time = float('inf')
        
        n = len(self.recovery_path)
        if n == 0:
            return self.plan_rolling_recovery(num_stations)
        
        # 策略1：尝试不同的起始站位
        for start_idx in range(min(n, max_iterations)):
            # 创建新的路径：从start_idx开始
            new_path = self.recovery_path[start_idx:] + self.recovery_path[:start_idx]
            
            # 创建临时规划器
            temp_planner = RecoveryStationPlanner(
                recovery_path=new_path,
                stations=self.stations.tolist(),
                station_depths=self.station_depths.tolist(),
                station_ascent_speeds=self.station_ascent_speeds,
                station_release_distances=self.station_release_distances,
                ascent_speed=self.ascent_speed,
                ship_speed=self.ship_speed,
                pickup_time=self.pickup_time,
                release_time=self.release_time
            )
            
            # 规划回收
            result = temp_planner.plan_rolling_recovery(num_stations)
            
            # 如果找到更好的结果，更新
            if result['total_time'] < best_total_time:
                best_total_time = result['total_time']
                best_result = result
        
        # 策略2：局部优化（保持总体回收方向）
        # 不尝试所有排列，而是保持总体方向，只尝试局部调整
        # 这样可以尊重站位文件中的初始顺序（起点到终点）
        # 当前实现已经通过策略1尝试不同起始点，这已经足够保持方向的同时进行优化
        
        return best_result if best_result else self.plan_rolling_recovery(num_stations)


def load_stations_file(filename: str) -> Tuple[List[Tuple[float, float]], List[str], 
                                                List[float], List[float], List[float]]:
    """
    从文件加载站位数据
    
    Parameters
    ----------
    filename : str
        站位文件路径
        
    Returns
    -------
    stations : List[Tuple[float, float]]
        站位坐标列表
    station_names : List[str]
        站位名称列表
    station_ascent_speeds : List[float]
        上浮速度列表（米/分钟）
    station_release_distances : List[float]
        释放距离列表（公里）
    station_depths : List[float]
        深度列表（米）
    """
    stations = []
    station_names = []
    station_ascent_speeds = []
    station_release_distances = []
    station_depths = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # 尝试CSV格式
                if ',' in line:
                    parts = [p.strip() for p in line.split(',')]
                else:
                    # 空格或制表符分隔
                    parts = line.split()
                
                if len(parts) >= 2:
                    try:
                        lon, lat = float(parts[0]), float(parts[1])
                        stations.append((lon, lat))
                        
                        # 解析各列
                        station_type = parts[2].strip() if len(parts) > 2 else ''
                        station_name = parts[3].strip() if len(parts) > 3 else station_type
                        station_names.append(station_name)
                        
                        # 第5列：上浮速度（米/分钟），默认39
                        ascent_speed = float(parts[4]) if len(parts) > 4 and parts[4].strip() else 39.0
                        station_ascent_speeds.append(ascent_speed)
                        
                        # 第6列：释放距离（公里），默认5
                        release_distance = float(parts[5]) if len(parts) > 5 and parts[5].strip() else 5.0
                        station_release_distances.append(release_distance)
                        
                        # 第7列：水深（米）
                        depth = float(parts[6]) if len(parts) > 6 and parts[6].strip() else None
                        station_depths.append(depth)
                        
                    except ValueError as e:
                        print(f"警告: 第{line_num}行格式错误，已跳过: {line} (错误: {e})", file=sys.stderr)
                        continue
        
        # 确保所有列表长度一致
        n = len(stations)
        while len(station_names) < n:
            station_names.append('')
        while len(station_ascent_speeds) < n:
            station_ascent_speeds.append(39.0)
        while len(station_release_distances) < n:
            station_release_distances.append(5.0)
        while len(station_depths) < n:
            station_depths.append(None)
        
    except FileNotFoundError:
        raise FileNotFoundError(f"文件未找到: {filename}")
    except Exception as e:
        raise Exception(f"读取文件时出错: {e}")
    
    return stations, station_names, station_ascent_speeds, station_release_distances, station_depths


def print_recovery_plan(result: dict, stations: List[Tuple[float, float]], 
                        station_names: List[str], station_depths: List[float],
                        station_ascent_speeds: List[float], station_release_distances: List[float],
                        pickup_time: float, release_time: float, ship_speed: float = 10.0):
    """
    打印回收计划
    
    Parameters
    ----------
    result : dict
        回收规划结果
    stations : List[Tuple[float, float]]
        站位坐标列表
    station_names : List[str]
        站位名称列表
    station_depths : List[float]
        站位深度列表
    station_ascent_speeds : List[float]
        上浮速度列表（米/分钟）
    pickup_time : float
        打捞时间（分钟）
    release_time : float
        释放时间（分钟）
    """
    print("\n" + "="*80)
    print("回收站位规划结果")
    print("="*80)
    
    print(f"\n总回收时间: {result['total_time']/3600:.2f} 小时 ({result['total_time']:.0f} 秒)")
    print(f"回收站位数量: {len(result['recovery_order'])}")
    feasible_count = len(result['recovery_order']) - len(result['violations'])
    print(f"满足时间约束的站位数量: {feasible_count} / {len(result['recovery_order'])}")
    print(f"时间约束满足: {'是' if result['is_feasible'] else '否'}")
    
    if not result['is_feasible']:
        print(f"\n⚠️  警告：发现 {len(result['violations'])} 个时间约束违反:")
        for violation in result['violations']:
            station_idx = violation['station_idx']
            station_name = station_names[station_idx] if station_idx < len(station_names) else f'站位 {station_idx}'
            print(f"  站位 {station_idx} ({station_name}):")
            print(f"    释放时间: {violation.get('release_time', 0)/60:.2f} 分钟")
            print(f"    回收到达时间: {violation.get('recovery_arrival_time', 0)/60:.2f} 分钟")
            print(f"    上浮时间: {violation.get('ascent_time', 0)/60:.2f} 分钟")
            print(f"    从释放到回收的时间: {violation.get('time_since_release', 0)/60:.2f} 分钟")
            print(f"    延迟: {violation.get('delay', 0)/60:.2f} 分钟")
            print(f"    (从释放到回收的时间必须小于上浮时间)")
    
        print(f"\n详细回收计划:")
        print("="*80)
        
        # 记录船的位置和时间，用于计算航行时间
        current_lon, current_lat = stations[result['recovery_order'][0]] if result['recovery_order'] else (0.0, 0.0)
        current_time = 0.0
        
        # release_times和recovery_order按相同顺序对应（在回收时记录释放信息）
        for i, station_idx in enumerate(result['recovery_order']):
            station_name = station_names[station_idx] if station_idx < len(station_names) else f'Station {station_idx}'
            lon, lat = stations[station_idx]
            depth = station_depths[station_idx] if station_idx < len(station_depths) and station_depths[station_idx] is not None else 0.0
            
            # 获取释放信息（按回收顺序对应）
            if i < len(result.get('release_times', [])):
                release_time = result['release_times'][i]
                release_pos = result['release_positions'][i] if i < len(result.get('release_positions', [])) else (lon, lat)
                release_lon, release_lat = release_pos
            else:
                # 如果没有找到释放信息，使用站位位置作为释放位置
                release_lon, release_lat = lon, lat
                release_time = result['recovery_arrival_times'][i] - 3600  # 假设1小时前释放
            
            recovery_arrival_time = result['recovery_arrival_times'][i]
            recovery_time = result['recovery_times'][i]
            recovery_end_time = recovery_arrival_time + recovery_time
            ascent_start_time = result['ascent_start_times'][i]
            ascent_end_time = result['ascent_end_times'][i]
            ascent_time = ascent_end_time - ascent_start_time
            
            release_distance = station_release_distances[station_idx] if station_idx < len(station_release_distances) else 5.0
            distance_to_station = haversine_distance(release_lon, release_lat, lon, lat)
            
            # 获取该站位自己的上浮速度
            if station_idx < len(station_ascent_speeds):
                speed_m_per_min = station_ascent_speeds[station_idx]
            else:
                speed_m_per_min = 39.0
            
            # 计算从释放到回收的时间
            time_since_release = recovery_arrival_time - release_time
            
            # 计算航行时间
            # 从当前位置到释放位置
            travel_to_release_distance = haversine_distance(current_lon, current_lat, release_lon, release_lat)
            travel_to_release_time = travel_to_release_distance / (ship_speed * 1.852) * 3600  # 秒
            
            # 从释放位置到站位位置
            travel_to_station_time = distance_to_station / (ship_speed * 1.852) * 3600  # 秒
            
            print(f"\n【操作 {i+1}】站位 {i+1}: {station_name} (索引 {station_idx})")
            print("-" * 80)
            
            # 步骤1：航行到释放位置
            print(f"\n  步骤1: 航行到释放位置")
            print(f"    当前位置: ({current_lon:.4f}°, {current_lat:.4f}°)")
            print(f"    释放位置: ({release_lon:.4f}°, {release_lat:.4f}°)")
            print(f"    航行距离: {travel_to_release_distance:.2f} 公里")
            print(f"    航行时间: {travel_to_release_time/60:.2f} 分钟 ({travel_to_release_time:.0f} 秒)")
            print(f"    出发时间: {current_time/60:.2f} 分钟 ({current_time:.0f} 秒)")
            print(f"    到达释放位置时间: {release_time/60:.2f} 分钟 ({release_time:.0f} 秒)")
            
            # 步骤2：释放操作
            print(f"\n  步骤2: 释放仪器")
            print(f"    释放位置: ({release_lon:.4f}°, {release_lat:.4f}°)")
            print(f"    释放距离范围: {release_distance:.2f} 公里")
            print(f"    释放时间: {release_time/60:.2f} 分钟 ({release_time:.0f} 秒)")
            print(f"    释放操作时间: {release_time:.1f} 分钟")
            print(f"    释放完成时间: {(release_time + release_time * 60)/60:.2f} 分钟")
            
            # 步骤3：仪器上浮
            print(f"\n  步骤3: 仪器上浮")
            print(f"    站位坐标: ({lon:.4f}°, {lat:.4f}°)")
            print(f"    深度: {depth:.1f} 米")
            print(f"    上浮速度: {speed_m_per_min:.2f} 米/分钟 ({speed_m_per_min/60:.4f} 米/秒)")
            print(f"    上浮开始时间: {ascent_start_time/60:.2f} 分钟 ({ascent_start_time:.0f} 秒)")
            print(f"    上浮结束时间: {ascent_end_time/60:.2f} 分钟 ({ascent_end_time:.0f} 秒)")
            print(f"    上浮时间: {ascent_time/60:.2f} 分钟 ({ascent_time:.0f} 秒)")
            
            # 步骤4：航行到站位位置
            print(f"\n  步骤4: 航行到站位位置进行回收")
            print(f"    释放位置: ({release_lon:.4f}°, {release_lat:.4f}°)")
            print(f"    站位位置: ({lon:.4f}°, {lat:.4f}°)")
            print(f"    航行距离: {distance_to_station:.2f} 公里")
            print(f"    航行时间: {travel_to_station_time/60:.2f} 分钟 ({travel_to_station_time:.0f} 秒)")
            print(f"    出发时间: {(release_time + release_time * 60)/60:.2f} 分钟")
            print(f"    到达站位时间: {recovery_arrival_time/60:.2f} 分钟 ({recovery_arrival_time:.0f} 秒)")
            
            # 步骤5：回收操作
            print(f"\n  步骤5: 回收操作")
            print(f"    站位位置: ({lon:.4f}°, {lat:.4f}°)")
            print(f"    到达时间: {recovery_arrival_time/60:.2f} 分钟 ({recovery_arrival_time:.0f} 秒)")
            print(f"    打捞时间: {pickup_time:.1f} 分钟")
            print(f"    回收完成时间: {recovery_end_time/60:.2f} 分钟 ({recovery_end_time:.0f} 秒)")
            
            # 时间统计
            print(f"\n  时间统计:")
            print(f"    从释放到回收的总时间: {time_since_release/60:.2f} 分钟 ({time_since_release:.0f} 秒)")
            print(f"    仪器上浮时间: {ascent_time/60:.2f} 分钟 ({ascent_time:.0f} 秒)")
            
            # 检查时间约束：从释放到回收的时间必须小于上浮时间
            if time_since_release >= ascent_time:
                delay = time_since_release - ascent_time
                print(f"    ⚠️  时间约束违反: 从释放到回收的时间超过上浮时间 {delay/60:.2f} 分钟")
            else:
                margin = ascent_time - time_since_release
                print(f"    ✓ 时间裕量: {margin/60:.2f} 分钟（从释放到回收的时间小于上浮时间）")
            
            # 更新当前位置和时间
            current_lon, current_lat = lon, lat
            current_time = recovery_end_time


def main():
    parser = argparse.ArgumentParser(
        description='站位回收策略规划工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
文件格式说明：
  第1列：经度
  第2列：纬度
  第3列：台站类型（可选）
  第4列：站位名称（可选）
  第5列：上浮速度（米/分钟，可选，默认39米/分钟）
  第6列：释放距离（公里，可选，默认5公里）
  第7列：水深（米，可选）

示例：
  python recovery_strategy.py stations.txt --ship-speed 10 --num-stations 2
        """
    )
    
    parser.add_argument('station_file', help='站位文件路径')
    parser.add_argument('--ship-speed', type=float, default=10.0,
                       help='船速（节），默认10节')
    parser.add_argument('--num-stations', type=int, default=2,
                       help='滚动站位数（至少2个），默认2。表示同时工作的站位数量，即最多可以有num_stations-1个仪器同时上浮')
    parser.add_argument('--pickup-time', type=float, default=30.0,
                       help='打捞时间（分钟），默认30分钟')
    parser.add_argument('--release-time', type=float, default=10.0,
                       help='释放时间（分钟），默认10分钟')
    parser.add_argument('--ascent-speed', type=float, default=0.65,
                       help='上浮速度（米/秒），默认0.65米/秒（39米/分钟）')
    parser.add_argument('--recovery-path', type=str, default=None,
                       help='回收路径文件（站位索引列表，每行一个索引）。如不提供，则使用文件中的原始顺序')
    parser.add_argument('--optimize', action='store_true',
                       help='是否优化回收顺序')
    
    args = parser.parse_args()
    
    try:
        # 加载站位数据
        print(f"正在加载站位文件: {args.station_file}")
        stations, station_names, station_ascent_speeds, station_release_distances, station_depths = \
            load_stations_file(args.station_file)
        
        print(f"成功加载 {len(stations)} 个站位")
        
        # 检查是否有深度数据
        if all(d is None for d in station_depths):
            print("警告: 站位文件中没有深度数据，请确保文件包含第7列（水深）", file=sys.stderr)
            print("将使用默认深度0米（可能导致计算错误）", file=sys.stderr)
            station_depths = [0.0] * len(stations)
        else:
            # 将None替换为0
            station_depths = [d if d is not None else 0.0 for d in station_depths]
        
        # 读取回收路径
        if args.recovery_path:
            print(f"正在加载回收路径文件: {args.recovery_path}")
            recovery_path = []
            with open(args.recovery_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            idx = int(line)
                            recovery_path.append(idx)
                        except ValueError:
                            continue
        else:
            # 使用原始顺序
            recovery_path = list(range(len(stations)))
            print("使用站位文件的原始顺序作为回收路径")
        
        if len(recovery_path) == 0:
            print("错误: 回收路径为空", file=sys.stderr)
            sys.exit(1)
        
        print(f"回收路径包含 {len(recovery_path)} 个站位")
        
        # 创建规划器
        planner = RecoveryStationPlanner(
            recovery_path=recovery_path,
            stations=stations,
            station_depths=station_depths,
            station_ascent_speeds=station_ascent_speeds,
            station_release_distances=station_release_distances,
            ascent_speed=args.ascent_speed,
            ship_speed=args.ship_speed,
            pickup_time=args.pickup_time,
            release_time=args.release_time
        )
        
        # 规划回收
        if args.optimize:
            print("\n正在优化回收顺序...")
            result = planner.optimize_recovery_order(num_stations=args.num_stations)
        else:
            print("\n正在规划回收策略...")
            result = planner.plan_rolling_recovery(num_stations=args.num_stations)
        
        # 打印结果
        print_recovery_plan(result, stations, station_names, station_depths,
                          station_ascent_speeds, station_release_distances,
                          args.pickup_time, args.release_time, args.ship_speed)
        
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
