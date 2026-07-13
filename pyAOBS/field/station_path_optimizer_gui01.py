#!/usr/bin/env python
"""
站位布设路径优化器 - GUI版本
功能：对提供的一系列站位坐标点（经纬度），设计一个连接所有站位的最短布设路径
使用大圆距离（Great Circle Distance）计算地球表面两点间的最短距离

Author: Haibo Huang
Date: 2025
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Polygon
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from scipy.interpolate import griddata
import os

# 地球平均半径（公里）
EARTH_RADIUS_KM = 6371.0


class StationPathOptimizer:
    """站位布设路径优化器"""
    
    def __init__(self, stations: List[Tuple[float, float]], method='nearest_neighbor', 
                 start_idx: Optional[int] = None, end_idx: Optional[int] = None,
                 port_coord: Optional[Tuple[float, float]] = None):
        """
        初始化优化器
        
        Parameters
        ----------
        stations : List[Tuple[float, float]]
            站位坐标列表，格式为 [(lon1, lat1), (lon2, lat2), ...]
            经度（longitude）和纬度（latitude），单位为度
        method : str, optional
            优化方法，可选：
            - 'nearest_neighbor': 最近邻贪心算法（快速，近似解）
            - '2opt': 2-opt局部优化（较慢，更优解）
            - '3opt': 3-opt局部优化（更慢，最优解）
            - 'greedy': 贪心算法（从最远点开始）
        start_idx : int, optional
            起始站位索引，如果指定，路径将从该点开始
        end_idx : int, optional
            结束站位索引，如果指定，路径将在该点结束
        port_coord : Tuple[float, float], optional
            港口坐标，格式为 (经度, 纬度)，单位为度
            如果指定，优化时会考虑起点到港口的距离，选择总成本（路径距离 + 起点到港口距离）最小的解
        """
        self.stations = np.array(stations)
        self.n_stations = len(stations)
        self.method = method
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.port_coord = port_coord
        self.optimal_path = None
        self.total_distance = None
        self.port_distance = None  # 起点到港口的距离
        
        if self.n_stations < 2:
            raise ValueError("至少需要2个站位点")
        
        # 验证起点和终点索引
        if self.start_idx is not None:
            if self.start_idx < 0 or self.start_idx >= self.n_stations:
                raise ValueError(f"起始点索引 {self.start_idx} 超出范围 [0, {self.n_stations-1}]")
        if self.end_idx is not None:
            if self.end_idx < 0 or self.end_idx >= self.n_stations:
                raise ValueError(f"结束点索引 {self.end_idx} 超出范围 [0, {self.n_stations-1}]")
        if self.start_idx is not None and self.end_idx is not None and self.start_idx == self.end_idx:
            raise ValueError("起始点和结束点不能相同")
        
        # 验证经纬度范围
        lons, lats = self.stations[:, 0], self.stations[:, 1]
        if np.any(lons < -180) or np.any(lons > 180):
            raise ValueError("经度范围应在 -180 到 180 度之间")
        if np.any(lats < -90) or np.any(lats > 90):
            raise ValueError("纬度范围应在 -90 到 90 度之间")
        
        # 验证港口坐标
        if self.port_coord is not None:
            port_lon, port_lat = self.port_coord
            if port_lon < -180 or port_lon > 180:
                raise ValueError("港口经度范围应在 -180 到 180 度之间")
            if port_lat < -90 or port_lat > 90:
                raise ValueError("港口纬度范围应在 -90 到 90 度之间")
    
    @staticmethod
    def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """
        使用Haversine公式计算两点间的大圆距离
        
        Parameters
        ----------
        lon1, lat1 : float
            第一个点的经度和纬度（度）
        lon2, lat2 : float
            第二个点的经度和纬度（度）
            
        Returns
        -------
        distance : float
            两点间距离（公里）
        """
        # 转换为弧度
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        
        # Haversine公式
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = EARTH_RADIUS_KM * c
        
        return distance
    
    def calculate_distance_matrix(self) -> np.ndarray:
        """
        计算所有站位之间的大圆距离矩阵（向量化实现，更高效）
        
        Returns
        -------
        distance_matrix : np.ndarray
            距离矩阵，单位：公里
        """
        n = self.n_stations
        lons = np.radians(self.stations[:, 0])
        lats = np.radians(self.stations[:, 1])
        
        # 使用广播计算所有点对之间的距离
        dlon = lons[:, np.newaxis] - lons
        dlat = lats[:, np.newaxis] - lats
        
        # Haversine公式（向量化）
        a = np.sin(dlat/2)**2 + np.cos(lats[:, np.newaxis]) * np.cos(lats) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # clip防止数值误差
        distance_matrix = EARTH_RADIUS_KM * c
        
        return distance_matrix
    
    def calculate_port_distance(self, station_idx: int) -> float:
        """
        计算指定站位到港口的距离
        
        Parameters
        ----------
        station_idx : int
            站位索引
            
        Returns
        -------
        distance : float
            站位到港口的距离（公里），如果未指定港口则返回0
        """
        if self.port_coord is None:
            return 0.0
        
        station_lon, station_lat = self.stations[station_idx]
        port_lon, port_lat = self.port_coord
        return self.haversine_distance(station_lon, station_lat, port_lon, port_lat)
    
    def nearest_neighbor(self, start_idx: Optional[int] = None, end_idx: Optional[int] = None) -> Tuple[List[int], float]:
        """
        最近邻贪心算法（支持固定起点和终点）
        
        算法步骤：
        1. 从起点开始
        2. 每一步都选择距离当前点最近的未访问点
        3. 如果指定了终点，在访问完所有其他点后，最后访问终点
        
        Parameters
        ----------
        start_idx : int, optional
            起始站位索引，如果为None则使用self.start_idx
        end_idx : int, optional
            结束站位索引，如果为None则使用self.end_idx
            
        Returns
        -------
        path : List[int]
            路径顺序（站位索引列表）
        total_distance : float
            总距离
        """
        if start_idx is None:
            start_idx = self.start_idx if self.start_idx is not None else 0
        if end_idx is None:
            end_idx = self.end_idx
        
        distance_matrix = self.calculate_distance_matrix()
        n = self.n_stations
        
        # 如果只有一个点，直接返回
        if n == 1:
            return [start_idx], 0.0
        
        # 初始化未访问点集合
        unvisited = set(range(n))
        path = [start_idx]
        unvisited.remove(start_idx)
        
        # 如果指定了终点且与起点不同，从未访问集合中移除终点（最后访问）
        if end_idx is not None and end_idx != start_idx and end_idx in unvisited:
            unvisited.remove(end_idx)
        
        total_distance = 0.0
        current = start_idx
        
        # 使用最近邻贪心算法构建路径
        while unvisited:
            # 找到距离当前点最近的未访问点
            nearest = min(unvisited, key=lambda x: distance_matrix[current, x])
            total_distance += distance_matrix[current, nearest]
            path.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        # 如果指定了终点，最后访问终点
        if end_idx is not None:
            # 如果终点不在路径中（即终点与起点不同且未被访问），添加到末尾
            if end_idx not in path:
                total_distance += distance_matrix[current, end_idx]
                path.append(end_idx)
            # 如果终点就是起点（形成环路），需要回到起点
            elif end_idx == start_idx and len(path) > 1:
                # 从最后一个点回到起点
                total_distance += distance_matrix[current, start_idx]
                path.append(start_idx)
        
        return path, total_distance
    
    def greedy_farthest_start(self) -> Tuple[List[int], float]:
        """
        从最远点开始的贪心算法
        先找到距离最远的点对作为起点和终点，然后使用最近邻算法填充中间点
        尝试多个最远点对，选择最优解
        """
        distance_matrix = self.calculate_distance_matrix()
        n = self.n_stations
        
        if n < 2:
            return [0], 0.0
        
        # 找到距离最远的多个点对（最多尝试5对或所有可能的点对，取较小值）
        max_pairs = min(5, n * (n - 1) // 2)
        
        # 收集所有点对及其距离
        all_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                all_pairs.append((distance_matrix[i, j], i, j))
        
        # 按距离降序排序，取前max_pairs对
        all_pairs.sort(reverse=True, key=lambda x: x[0])
        farthest_pairs = all_pairs[:max_pairs]
        
        # 尝试每对最远点，选择最优解
        best_path = None
        best_distance = float('inf')
        
        for dist, start_idx, end_idx in farthest_pairs:
            # 尝试两个方向：start->end 和 end->start
            for start, end in [(start_idx, end_idx), (end_idx, start_idx)]:
                path, distance = self.nearest_neighbor(start, end)
                if distance < best_distance:
                    best_distance = distance
                    best_path = path
        
        # 如果没有找到有效路径，使用默认起点
        if best_path is None:
            best_path, best_distance = self.nearest_neighbor(0)
        
        return best_path, best_distance
    
    def two_opt(self, initial_path: Optional[List[int]] = None, max_iterations: int = 10) -> Tuple[List[int], float]:
        """
        2-opt局部优化算法（保持起点和终点固定）
        对初始路径进行2-opt交换优化，支持多次迭代以扩大搜索范围
        
        Parameters
        ----------
        initial_path : List[int], optional
            初始路径，如果为None则使用最近邻算法生成
        max_iterations : int, optional
            最大迭代次数，默认10次，用于扩大搜索范围
            
        Returns
        -------
        path : List[int]
            优化后的路径
        total_distance : float
            总距离
        """
        distance_matrix = self.calculate_distance_matrix()
        
        # 如果没有提供初始路径，使用最近邻算法生成
        if initial_path is None:
            path, _ = self.nearest_neighbor()
        else:
            path = initial_path.copy()
        
        n = len(path)
        best_distance = self._calculate_path_distance(path, distance_matrix)
        best_path = path.copy()
        
        # 多次迭代以扩大搜索范围
        for iteration in range(max_iterations):
            improved = True
            current_path = best_path.copy()
            current_distance = best_distance
            
            while improved:
                improved = False
                # 只优化中间部分，保持起点和终点不变
                # 扩大搜索范围：尝试所有可能的i和j组合
                for i in range(1, n - 2):
                    for j in range(i + 1, n - 1):  # 不包含最后一个点
                        if j - i == 1:
                            continue
                        
                        # 尝试2-opt交换（只交换中间部分）
                        new_path = current_path[:i] + current_path[i:j][::-1] + current_path[j:]
                        new_distance = self._calculate_path_distance(new_path, distance_matrix)
                        
                        if new_distance < current_distance:
                            current_path = new_path
                            current_distance = new_distance
                            improved = True
                            break  # 找到改进后重新开始搜索
                    
                    if improved:
                        break  # 找到改进后重新开始搜索
            
            # 如果本次迭代找到了更好的解，更新最佳解
            if current_distance < best_distance:
                best_path = current_path
                best_distance = current_distance
            else:
                # 如果没有改进，可以提前结束
                break
        
        return best_path, best_distance
    
    def three_opt(self, initial_path: Optional[List[int]] = None, max_iterations: int = 10) -> Tuple[List[int], float]:
        """
        3-opt局部优化算法（保持起点和终点固定）
        对初始路径进行3-opt交换优化，通过交换三条边来改进路径
        支持多次迭代以扩大搜索范围
        
        Parameters
        ----------
        initial_path : List[int], optional
            初始路径，如果为None则使用最近邻算法生成
        max_iterations : int, optional
            最大迭代次数，默认10次，用于扩大搜索范围
            
        Returns
        -------
        path : List[int]
            优化后的路径
        total_distance : float
            总距离
        """
        distance_matrix = self.calculate_distance_matrix()
        
        # 如果没有提供初始路径，使用最近邻算法生成
        if initial_path is None:
            path, _ = self.nearest_neighbor()
        else:
            path = initial_path.copy()
        
        n = len(path)
        best_distance = self._calculate_path_distance(path, distance_matrix)
        best_path = path.copy()
        
        # 多次迭代以扩大搜索范围
        for iteration in range(max_iterations):
            improved = True
            current_path = best_path.copy()
            current_distance = best_distance
            
            while improved:
                improved = False
                # 只优化中间部分，保持起点和终点不变
                # 扩大搜索范围：尝试所有可能的i、j、k组合
                for i in range(1, n - 3):
                    for j in range(i + 1, n - 2):
                        for k in range(j + 1, n - 1):  # 不包含最后一个点
                            # 尝试所有7种3-opt重组方式
                            # 原路径: current_path[0...i] -> current_path[i...j] -> current_path[j...k] -> current_path[k...n-1]
                            
                            # 方式1: 原路径（不改变）
                            # 方式2: 反转中间段 current_path[i...j]
                            new_path2 = current_path[:i] + current_path[i:j][::-1] + current_path[j:k] + current_path[k:]
                            # 方式3: 反转最后段 current_path[j...k]
                            new_path3 = current_path[:i] + current_path[i:j] + current_path[j:k][::-1] + current_path[k:]
                            # 方式4: 反转两段 current_path[i...j] 和 current_path[j...k]
                            new_path4 = current_path[:i] + current_path[i:j][::-1] + current_path[j:k][::-1] + current_path[k:]
                            # 方式5: 交换两段: current_path[i...j] 和 current_path[j...k]
                            new_path5 = current_path[:i] + current_path[j:k] + current_path[i:j] + current_path[k:]
                            # 方式6: 交换并反转: current_path[i...j] 反转后与 current_path[j...k] 交换
                            new_path6 = current_path[:i] + current_path[j:k] + current_path[i:j][::-1] + current_path[k:]
                            # 方式7: 交换并反转: current_path[i...j] 与 current_path[j...k] 反转后交换
                            new_path7 = current_path[:i] + current_path[j:k][::-1] + current_path[i:j] + current_path[k:]
                            
                            # 计算所有新路径的距离（不包括原路径）
                            candidates = [
                                (new_path2, self._calculate_path_distance(new_path2, distance_matrix)),
                                (new_path3, self._calculate_path_distance(new_path3, distance_matrix)),
                                (new_path4, self._calculate_path_distance(new_path4, distance_matrix)),
                                (new_path5, self._calculate_path_distance(new_path5, distance_matrix)),
                                (new_path6, self._calculate_path_distance(new_path6, distance_matrix)),
                                (new_path7, self._calculate_path_distance(new_path7, distance_matrix))
                            ]
                            
                            # 找到最短的路径
                            best_candidate = min(candidates, key=lambda x: x[1])
                            
                            if best_candidate[1] < current_distance:
                                current_path = best_candidate[0]
                                current_distance = best_candidate[1]
                                improved = True
                                break  # 找到改进就跳出内层循环，重新开始
                        
                        if improved:
                            break  # 找到改进就跳出中层循环
                    
                    if improved:
                        break  # 找到改进就跳出外层循环
            
            # 如果本次迭代找到了更好的解，更新最佳解
            if current_distance < best_distance:
                best_path = current_path
                best_distance = current_distance
            else:
                # 如果没有改进，可以提前结束
                break
        
        return best_path, best_distance
    
    def _calculate_path_distance(self, path: List[int], distance_matrix: np.ndarray) -> float:
        """计算路径总距离"""
        total = 0.0
        for i in range(len(path) - 1):
            total += distance_matrix[path[i], path[i + 1]]
        return total
    
    def optimize(self) -> Tuple[List[int], float]:
        """
        执行路径优化（支持固定起点和终点）
        
        Returns
        -------
        path : List[int]
            最优路径（站位索引列表）
        total_distance : float
            总距离
        """
        if self.method == 'nearest_neighbor':
            if self.start_idx is not None:
                # 如果指定了起点，直接使用
                self.optimal_path, self.total_distance = self.nearest_neighbor()
            else:
                # 尝试所有起点，选择最优的
                best_path = None
                best_distance = float('inf')
                for start_idx in range(self.n_stations):
                    # 跳过已指定的终点
                    if self.end_idx is not None and start_idx == self.end_idx:
                        continue
                    path, distance = self.nearest_neighbor(start_idx, self.end_idx)
                    if distance < best_distance:
                        best_distance = distance
                        best_path = path
                self.optimal_path = best_path
                self.total_distance = best_distance
            
        elif self.method == 'greedy':
            # 如果指定了起点和终点，使用它们
            if self.start_idx is not None and self.end_idx is not None:
                self.optimal_path, self.total_distance = self.nearest_neighbor()
            # 如果只指定了终点，尝试所有起点，选择最优的
            elif self.start_idx is None and self.end_idx is not None:
                best_path = None
                best_distance = float('inf')
                for start_idx in range(self.n_stations):
                    if start_idx == self.end_idx:
                        continue
                    path, distance = self.nearest_neighbor(start_idx, self.end_idx)
                    if distance < best_distance:
                        best_distance = distance
                        best_path = path
                self.optimal_path = best_path
                self.total_distance = best_distance
            else:
                # 如果都不指定或只指定起点，使用最远点
                self.optimal_path, self.total_distance = self.greedy_farthest_start()
            
        elif self.method == '2opt':
            # 如果只指定了终点，尝试所有起点，选择最优的
            if self.start_idx is None and self.end_idx is not None:
                best_path = None
                best_distance = float('inf')
                # 尝试多个不同的初始路径：最近邻、最远点等
                initial_paths = []
                
                # 1. 尝试所有起点作为最近邻的起点
                for start_idx in range(self.n_stations):
                    if start_idx == self.end_idx:
                        continue
                    initial_path, _ = self.nearest_neighbor(start_idx, self.end_idx)
                    initial_paths.append(initial_path)
                
                # 2. 尝试最远点对作为起点
                distance_matrix = self.calculate_distance_matrix()
                max_dist = 0
                farthest_start = 0
                for i in range(self.n_stations):
                    if i == self.end_idx:
                        continue
                    dist = distance_matrix[i, self.end_idx]
                    if dist > max_dist:
                        max_dist = dist
                        farthest_start = i
                if farthest_start != self.end_idx:
                    initial_path, _ = self.nearest_neighbor(farthest_start, self.end_idx)
                    if initial_path not in initial_paths:
                        initial_paths.append(initial_path)
                
                # 对每个初始路径进行2-opt优化
                for initial_path in initial_paths:
                    path, distance = self.two_opt(initial_path)
                    if distance < best_distance:
                        best_distance = distance
                        best_path = path
                
                self.optimal_path = best_path
                self.total_distance = best_distance
            elif self.start_idx is None:
                # 如果没有指定起点和终点，尝试多个不同的初始路径
                best_path = None
                best_distance = float('inf')
                initial_paths = []
                
                # 1. 尝试所有起点作为最近邻的起点
                for start_idx in range(self.n_stations):
                    initial_path, _ = self.nearest_neighbor(start_idx)
                    initial_paths.append(initial_path)
                
                # 2. 尝试最远点对
                farthest_path, _ = self.greedy_farthest_start()
                if farthest_path not in initial_paths:
                    initial_paths.append(farthest_path)
                
                # 对每个初始路径进行2-opt优化
                for initial_path in initial_paths:
                    path, distance = self.two_opt(initial_path)
                    if distance < best_distance:
                        best_distance = distance
                        best_path = path
                
                self.optimal_path = best_path
                self.total_distance = best_distance
            else:
                # 先用最近邻生成初始路径，再用2-opt优化
                initial_path, _ = self.nearest_neighbor()
                self.optimal_path, self.total_distance = self.two_opt(initial_path)
            
        elif self.method == '3opt':
            # 如果只指定了终点，先用2-opt找到最优起点，再用3-opt优化
            if self.start_idx is None and self.end_idx is not None:
                # 第一步：使用2-opt找到多个候选路径
                candidate_paths = []
                
                # 尝试多个不同的初始路径
                initial_paths = []
                for start_idx in range(self.n_stations):
                    if start_idx == self.end_idx:
                        continue
                    initial_path, _ = self.nearest_neighbor(start_idx, self.end_idx)
                    initial_paths.append(initial_path)
                
                # 尝试最远点对
                distance_matrix = self.calculate_distance_matrix()
                max_dist = 0
                farthest_start = 0
                for i in range(self.n_stations):
                    if i == self.end_idx:
                        continue
                    dist = distance_matrix[i, self.end_idx]
                    if dist > max_dist:
                        max_dist = dist
                        farthest_start = i
                if farthest_start != self.end_idx:
                    initial_path, _ = self.nearest_neighbor(farthest_start, self.end_idx)
                    if initial_path not in initial_paths:
                        initial_paths.append(initial_path)
                
                # 对每个初始路径进行2-opt优化，收集候选路径
                for initial_path in initial_paths:
                    path, distance = self.two_opt(initial_path)
                    candidate_paths.append((path, distance))
                
                # 选择前3个最好的2-opt结果进行3-opt优化
                candidate_paths.sort(key=lambda x: x[1])
                best_path = None
                best_distance = float('inf')
                
                for path, _ in candidate_paths[:min(3, len(candidate_paths))]:
                    opt_path, opt_distance = self.three_opt(path)
                    if opt_distance < best_distance:
                        best_distance = opt_distance
                        best_path = opt_path
                
                if best_path is not None:
                    self.optimal_path = best_path
                    self.total_distance = best_distance
                else:
                    # 如果3-opt没有找到解，使用默认方式
                    initial_path, _ = self.nearest_neighbor()
                    self.optimal_path, self.total_distance = self.three_opt(initial_path)
            elif self.start_idx is None:
                # 如果没有指定起点和终点，尝试多个不同的初始路径
                candidate_paths = []
                initial_paths = []
                
                # 1. 尝试所有起点作为最近邻的起点
                for start_idx in range(self.n_stations):
                    initial_path, _ = self.nearest_neighbor(start_idx)
                    initial_paths.append(initial_path)
                
                # 2. 尝试最远点对
                farthest_path, _ = self.greedy_farthest_start()
                if farthest_path not in initial_paths:
                    initial_paths.append(farthest_path)
                
                # 对每个初始路径进行2-opt优化，收集候选路径
                for initial_path in initial_paths:
                    path, distance = self.two_opt(initial_path)
                    candidate_paths.append((path, distance))
                
                # 选择前3个最好的2-opt结果进行3-opt优化
                candidate_paths.sort(key=lambda x: x[1])
                best_path = None
                best_distance = float('inf')
                
                for path, _ in candidate_paths[:min(3, len(candidate_paths))]:
                    opt_path, opt_distance = self.three_opt(path)
                    if opt_distance < best_distance:
                        best_distance = opt_distance
                        best_path = opt_path
                
                self.optimal_path = best_path
                self.total_distance = best_distance
            else:
                # 先用最近邻生成初始路径，再用2-opt优化，最后用3-opt优化
                initial_path, _ = self.nearest_neighbor()
                # 先用2-opt预处理
                path_2opt, _ = self.two_opt(initial_path)
                # 再用3-opt进一步优化
                self.optimal_path, self.total_distance = self.three_opt(path_2opt)
        
        # 验证路径是否以指定的终点结束
        if self.end_idx is not None and self.optimal_path is not None:
            if self.optimal_path[-1] != self.end_idx:
                # 如果路径的终点与指定的终点不一致，强制修正
                # 这可能发生在某些优化算法中
                if self.end_idx in self.optimal_path:
                    # 如果终点在路径中，将其移到末尾
                    self.optimal_path.remove(self.end_idx)
                    self.optimal_path.append(self.end_idx)
                    # 重新计算距离
                    self.total_distance = self._calculate_path_distance(self.optimal_path, self.calculate_distance_matrix())
                else:
                    # 如果终点不在路径中，直接添加到末尾
                    distance_matrix = self.calculate_distance_matrix()
                    if len(self.optimal_path) > 0:
                        self.total_distance += distance_matrix[self.optimal_path[-1], self.end_idx]
                    self.optimal_path.append(self.end_idx)
        
        # 计算港口距离（如果指定了港口）
        if self.port_coord is not None and self.optimal_path is not None:
            self.port_distance = self.calculate_port_distance(self.optimal_path[0])
        else:
            self.port_distance = None
        
        return self.optimal_path, self.total_distance


class RecoveryStationPlanner:
    """回收站位规划器
    
    功能：根据回收路径规划滚动回收策略，确保在站位上浮到海面前船能到达站位位置
    """
    
    def __init__(self, recovery_path: List[int], stations: List[Tuple[float, float]], 
                 station_depths: List[float], ascent_speed: float = 0.5, 
                 ship_speed: float = 10.0):
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
        ascent_speed : float
            站位上浮速度（米/秒），默认0.5米/秒
        ship_speed : float
            船速（节），默认10节（约5.14米/秒或18.52公里/小时）
        """
        self.recovery_path = recovery_path
        self.stations = np.array(stations)
        self.station_depths = np.array(station_depths)
        self.ascent_speed = ascent_speed  # 米/秒
        self.ship_speed = ship_speed  # 节
        
        # 将船速从节转换为公里/小时
        self.ship_speed_kmh = ship_speed * 1.852  # 1节 = 1.852公里/小时
        self.ship_speed_ms = ship_speed * 0.5144  # 1节 = 0.5144米/秒
        
        # 验证回收路径中的站位索引是否有效
        if len(recovery_path) == 0:
            raise ValueError("回收路径不能为空")
        if max(recovery_path) >= len(stations):
            raise ValueError(f"回收路径中包含无效的站位索引: {max(recovery_path)} >= {len(stations)}")
        if min(recovery_path) < 0:
            raise ValueError(f"回收路径中包含负的站位索引")
        
        # 验证站位深度数量
        if len(station_depths) != len(stations):
            raise ValueError(f"站位深度数量 ({len(station_depths)}) 与站位数量 ({len(stations)}) 不匹配")
    
    def calculate_ascent_time(self, depth: float) -> float:
        """
        计算站位上浮所需时间
        
        Parameters
        ----------
        depth : float
            站位深度（米）
            
        Returns
        -------
        time : float
            上浮时间（秒）
        """
        if depth <= 0:
            return 0.0
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
        distance_km = StationPathOptimizer.haversine_distance(lon1, lat1, lon2, lat2)
        time_hours = distance_km / self.ship_speed_kmh
        return time_hours * 3600  # 转换为秒
    
    def plan_rolling_recovery(self, num_stations: int = 2) -> dict:
        """
        规划滚动回收策略
        
        Parameters
        ----------
        num_stations : int
            同时工作的站位数量（2或3）
            
        Returns
        -------
        result : dict
            包含以下键的字典：
            - 'recovery_order': List[int] - 回收顺序（站位索引列表）
            - 'recovery_times': List[float] - 每个站位的回收时间（秒）
            - 'arrival_times': List[float] - 船到达每个站位的时间（秒）
            - 'ascent_start_times': List[float] - 每个站位开始上浮的时间（秒）
            - 'total_time': float - 总回收时间（秒）
            - 'is_feasible': bool - 是否满足时间约束
            - 'violations': List[dict] - 违反时间约束的站位信息
        """
        if num_stations not in [2, 3]:
            raise ValueError("num_stations必须是2或3")
        
        n = len(self.recovery_path)
        if n == 0:
            return {
                'recovery_order': [],
                'recovery_times': [],
                'arrival_times': [],
                'ascent_start_times': [],
                'total_time': 0.0,
                'is_feasible': True,
                'violations': []
            }
        
        # 计算每个站位上浮所需时间
        ascent_times = [self.calculate_ascent_time(depth) 
                        for depth in self.station_depths]
        
        # 初始化结果
        recovery_order = []
        recovery_times = []  # 每个站位的回收操作时间（假设固定，如300秒）
        arrival_times = []  # 船到达每个站位的时间
        ascent_start_times = []  # 每个站位开始上浮的时间
        
        # 当前船的位置（初始位置设为第一个站位）
        current_lon, current_lat = self.stations[self.recovery_path[0]]
        current_time = 0.0
        
        # 正在上浮的站位队列（最多num_stations-1个）
        # 格式：[(station_idx, ascent_start_time, ascent_end_time), ...]
        ascending_stations = []
        
        violations = []
        
        # 遍历回收路径
        for i, station_idx in enumerate(self.recovery_path):
            station_lon, station_lat = self.stations[station_idx]
            depth = self.station_depths[station_idx]
            ascent_time = ascent_times[station_idx]
            
            # 计算船从当前位置航行到该站位所需时间
            travel_time = self.calculate_travel_time(
                current_lon, current_lat, station_lon, station_lat
            )
            
            # 更新船的位置和时间
            arrival_time = current_time + travel_time
            arrival_times.append(arrival_time)
            
            # 确定该站位开始上浮的时间
            # 策略：如果队列未满，立即开始上浮；否则等待队列中有站位完成上浮
            if len(ascending_stations) < num_stations - 1:
                # 队列未满，立即开始上浮
                ascent_start_time = arrival_time
            else:
                # 队列已满，等待最早完成的站位
                # 找到最早完成上浮的站位
                earliest_end_time = min([end_time for _, _, end_time in ascending_stations])
                ascent_start_time = max(arrival_time, earliest_end_time)
            
            ascent_start_times.append(ascent_start_time)
            
            # 计算该站位完成上浮的时间
            ascent_end_time = ascent_start_time + ascent_time
            
            # 检查时间约束：船必须在站位上浮到海面前到达
            # 实际上，我们要求船在站位开始上浮前到达（更安全）
            if arrival_time > ascent_start_time:
                violations.append({
                    'station_idx': station_idx,
                    'arrival_time': arrival_time,
                    'ascent_start_time': ascent_start_time,
                    'delay': arrival_time - ascent_start_time
                })
            
            # 假设回收操作需要固定时间（如300秒）
            recovery_time = 300.0  # 秒
            recovery_times.append(recovery_time)
            
            # 该站位完成回收的时间
            recovery_end_time = ascent_end_time + recovery_time
            
            # 更新当前时间和位置
            current_time = recovery_end_time
            current_lon, current_lat = station_lon, station_lat
            
            # 更新上浮队列：移除已完成上浮的站位，添加新站位
            ascending_stations = [
                (idx, start, end) for idx, start, end in ascending_stations
                if end > ascent_start_time
            ]
            ascending_stations.append((station_idx, ascent_start_time, ascent_end_time))
            
            recovery_order.append(station_idx)
        
        total_time = current_time
        is_feasible = len(violations) == 0
        
        return {
            'recovery_order': recovery_order,
            'recovery_times': recovery_times,
            'arrival_times': arrival_times,
            'ascent_start_times': ascent_start_times,
            'total_time': total_time,
            'is_feasible': is_feasible,
            'violations': violations
        }
    
    def optimize_recovery_order(self, num_stations: int = 2, 
                                max_iterations: int = 100) -> dict:
        """
        优化回收顺序，使得总时间最短
        
        Parameters
        ----------
        num_stations : int
            同时工作的站位数量（2或3）
        max_iterations : int
            最大迭代次数
            
        Returns
        -------
        result : dict
            优化后的回收规划结果
        """
        # 使用贪心算法：尝试不同的起始站位
        best_result = None
        best_total_time = float('inf')
        
        n = len(self.recovery_path)
        if n == 0:
            return self.plan_rolling_recovery(num_stations)
        
        # 尝试不同的起始站位
        for start_idx in range(min(n, max_iterations)):
            # 创建新的路径顺序（从start_idx开始，循环）
            new_path = self.recovery_path[start_idx:] + self.recovery_path[:start_idx]
            
            # 创建临时规划器
            temp_planner = RecoveryStationPlanner(
                new_path, self.stations.tolist(), 
                self.station_depths.tolist(),
                self.ascent_speed, self.ship_speed
            )
            
            # 规划回收
            result = temp_planner.plan_rolling_recovery(num_stations)
            
            # 如果时间更短且满足约束，更新最佳结果
            if result['total_time'] < best_total_time and result['is_feasible']:
                best_total_time = result['total_time']
                best_result = result
        
        # 如果所有尝试都不满足约束，返回总时间最短的结果
        if best_result is None:
            for start_idx in range(min(n, max_iterations)):
                new_path = self.recovery_path[start_idx:] + self.recovery_path[:start_idx]
                temp_planner = RecoveryStationPlanner(
                    new_path, self.stations.tolist(), 
                    self.station_depths.tolist(),
                    self.ascent_speed, self.ship_speed
                )
                result = temp_planner.plan_rolling_recovery(num_stations)
                if result['total_time'] < best_total_time:
                    best_total_time = result['total_time']
                    best_result = result
        
        return best_result if best_result is not None else self.plan_rolling_recovery(num_stations)


class StationPathOptimizerGUI:
    """站位布设路径优化器GUI"""
    
    def __init__(self, master=None):
        """初始化GUI"""
        # 创建主窗口
        if master is None:
            self.root = tk.Tk()
        else:
            self.root = master
        
        self.root.title('站位布设路径优化器')
        self.root.geometry('1400x900+100+50')
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        
        # 数据
        self.stations = []  # List[Tuple[float, float]] 站位坐标列表（用于优化器）
        self.station_names = []  # List[str] 站位名称列表（可选，与stations一一对应）
        self.station_types = []  # List[str] 台站类型列表（第三列，与stations一一对应）
        self.station_beams = []  # List[str] 站位名称列表（可选，与stations一一对应）
        self.station_ascent_speeds = []  # List[float] 站位上浮速度列表（第5列，与stations一一对应）
        self.station_depths_from_file = []  # List[float] 站位深度列表（第6列，与stations一一对应）
        self.station_all_columns = []  # List[List] 站位所有列数据（用于导出）
        self.optimizer = None  # StationPathOptimizer实例
        self.deployment_path = None  # 投放路径（站位索引列表）
        self.deployment_distance = None  # 投放路径总距离
        self.previous_deployment_start_idx = None  # 上一次投放路径的起点（用于确保可逆性）
        self.survey_line_order = None  # 测线访问顺序（测线索引列表）
        self.survey_line_transition_distance = None  # 测线之间的航渡路径距离（直线距离）
        self.total_survey_length = None  # 测线总长度（所有测线的长度之和）
        self.survey_line_entry_points = {}  # 每条测线的进入点 {line_idx: (lon, lat)}
        self.survey_line_exit_points = {}  # 每条测线的退出点 {line_idx: (lon, lat)}
        self.transition_start_points = {}  # 航渡路径起点 {line_pair: (lon, lat)}，key为 (line1_idx, line2_idx)
        self.transition_end_points = {}  # 航渡路径终点 {line_pair: (lon, lat)}，key为 (line1_idx, line2_idx)
        self.turn_path_lengths = {}  # 转弯路径长度 {line_pair: length}，key为 (line1_idx, line2_idx)
        self.recovery_path = None  # 回收路径（站位索引列表）
        self.recovery_distance = None  # 回收路径总距离
        self.deployment_to_survey_distance = None  # 从最后一个投放站位到第一条测线的距离
        self.survey_to_recovery_distance = None  # 从最后一条测线到第一个回收站位的距离
        self.start_idx = None  # 起点索引
        self.end_idx = None  # 终点索引
        self.selected_start_idx = None  # 当前选择的起点索引
        self.selected_end_idx = None  # 当前选择的终点索引
        self.selected_first_survey_line_idx = None  # 用户选择的第一条测线索引
        self.deployment_station_types = []  # 用于投放的站位类型列表
        self.recovery_station_types = []  # 用于回收的站位类型列表
        self.deployment_type_vars = {}  # 投放类型选择变量字典 {type: BooleanVar}
        self.recovery_type_vars = {}  # 回收类型选择变量字典 {type: BooleanVar}
        
        # 绘图相关
        self.fig = None
        self.ax = None
        self.canvas = None
        self.station_artists = []  # 存储站位点的图形对象
        self.path_artist = None  # 存储路径线的图形对象
        self.path_labels = []  # 存储路径顺序标注
        self.deployment_path_artist = None  # 投放路径图形对象
        self.recovery_path_artist = None  # 回收路径图形对象
        self.deployment_labels = []  # 投放路径标注
        self.recovery_labels = []  # 回收路径标注
        self.start_artist = None  # 起点标记
        self.end_artist = None  # 终点标记
        self.survey_line_artists = []  # 存储测线的图形对象
        self.survey_line_labels = []  # 存储测线编号标注
        self.survey_line_start_artists = []  # 存储测线起点标记
        self.transition_path_artists = []  # 存储航渡路径图形对象
        self.turn_path_artists = []  # 存储转弯路径的图形对象
        self.turn_paths = []  # 存储计算的转弯路径数据
        self.bathymetry_artist = None  # 存储水深图的图形对象
        
        # 初始化路径标注列表
        self.path_labels = []
        
        # 航时计算相关数据
        self.bathymetry_filename = None  # 水深文件路径（用于按需读取）
        self.bathymetry_data = None  # 水深网格数据（仅用于非.grd文件或已加载的数据）
        self.bathymetry_lon = None  # 水深网格经度（仅用于非.grd文件）
        self.bathymetry_lat = None  # 水深网格纬度（仅用于非.grd文件）
        self.bathymetry_full_range = None  # 完整水深网格的范围 [lon_min, lon_max, lat_min, lat_max]（用于.grd和.nc文件）
        self.bathymetry_file_type = None  # 文件类型：'grd', 'nc', 'txt'等
        self.bathymetry_var_name = None  # 数据变量名（用于.nc文件）
        self.bathymetry_lon_name = None  # 经度坐标名（用于.nc文件）
        self.bathymetry_lat_name = None  # 纬度坐标名（用于.nc文件）
        self.survey_lines = []  # 作业测线坐标列表
        self.station_depths = []  # 每个站位的水深（米）
        self.default_depth = 3000.0  # 默认水深（米），当没有加载水深文件时使用
        
        # 水深图缓存（用于性能优化）
        self._bathymetry_cache = None  # 缓存处理后的水深图数据
        self._bathymetry_cache_extent = None  # 缓存的范围
        self._bathymetry_cache_hash = None  # 缓存哈希值（用于判断是否需要重新计算）
        
        # 航时计算参数
        self.cruise_speed = 10.0  # 走航速度（节，knots）
        self.working_speed = 5.0  # 作业速度（节，knots）
        self.deployment_time = 300.0  # 台站投放所需时间（秒）
        self.ascent_speed = 0.5  # 仪器上浮速度（米/秒）
        self.turn_radius = 5.0  # 转弯半径（公里）
        
        # 航时计算结果
        self.time_results = None  # 存储计算结果
        
        # 回收规划结果
        self.recovery_plan_result = None  # 存储回收规划结果
        
        # 创建界面
        self.create_widgets()
        
        # 创建菜单栏
        self.create_menu()
    
    def create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        filemenu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='文件', menu=filemenu)
        filemenu.add_command(label='加载站位', command=self.load_stations_file, accelerator='Ctrl+O')
        filemenu.add_command(label='保存路径图', command=self.save_path_figure)
        filemenu.add_command(label='导出结果', command=self.export_results)
        filemenu.add_command(label='保存排序站位', command=self.save_sorted_stations)
        filemenu.add_command(label='保存时间结果', command=self.save_time_results)
        filemenu.add_separator()
        filemenu.add_command(label='退出', command=self.root.quit)
        
        # 绑定快捷键
        self.root.bind('<Control-o>', lambda e: self.load_stations_file())
    
    def create_widgets(self):
        """创建GUI组件"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky='nswe')
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        
        # 绘图区域
        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=0, column=0, sticky='nswe', padx=(0, 10))
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        
        # 创建matplotlib图形
        self.fig = plt.Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        
        # 创建canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nswe')
        
        # 创建工具栏
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky='ew')
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        # 侧边栏
        side_frame = ttk.Frame(main_frame)
        side_frame.grid(row=0, column=1, sticky='nswe')
        side_frame.columnconfigure(0, weight=1)
        
        # File operations
        file_frame = ttk.LabelFrame(side_frame, text='文件操作')
        file_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        file_frame.columnconfigure(0, weight=1)
        ttk.Button(file_frame, text='加载站位', command=self.load_stations_file).grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        ttk.Button(file_frame, text='保存路径图', command=self.save_path_figure).grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        ttk.Button(file_frame, text='保存排序站位', command=self.save_sorted_stations).grid(row=2, column=0, sticky='ew', padx=5, pady=5)
        
        # Start and end point selection (in one row)
        selection_frame = ttk.LabelFrame(side_frame, text='起点和终点选择')
        selection_frame.grid(row=1, column=0, sticky='ew', pady=(0, 10))
        selection_frame.columnconfigure(1, weight=1)
        selection_frame.columnconfigure(3, weight=1)
        
        ttk.Label(selection_frame, text='起点:').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.start_var = tk.StringVar(value='未选择')
        self.start_combo = ttk.Combobox(selection_frame, textvariable=self.start_var, 
                                       state='readonly', width=15)
        self.start_combo.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        self.start_combo.bind('<<ComboboxSelected>>', self.on_start_selected)
        
        ttk.Label(selection_frame, text='终点:').grid(row=0, column=2, sticky='w', padx=5, pady=5)
        self.end_var = tk.StringVar(value='未选择')
        self.end_combo = ttk.Combobox(selection_frame, textvariable=self.end_var,
                                     state='readonly', width=15)
        self.end_combo.grid(row=0, column=3, sticky='ew', padx=5, pady=5)
        self.end_combo.bind('<<ComboboxSelected>>', self.on_end_selected)
        
        ttk.Button(selection_frame, text='点击地图选择', command=self.enable_click_selection).grid(
            row=1, column=0, columnspan=4, sticky='ew', padx=5, pady=5)
        
        # First survey line selection
        survey_line_selection_frame = ttk.LabelFrame(side_frame, text='第一条测线选择')
        survey_line_selection_frame.grid(row=2, column=0, sticky='ew', pady=(0, 10))
        survey_line_selection_frame.columnconfigure(0, weight=1)
        
        self.first_survey_line_var = tk.StringVar(value='未选择')
        ttk.Label(survey_line_selection_frame, text='第一条测线:').grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.first_survey_line_combo = ttk.Combobox(survey_line_selection_frame, 
                                                     textvariable=self.first_survey_line_var,
                                                     state='readonly', width=20)
        self.first_survey_line_combo.grid(row=0, column=1, sticky='ew', padx=5, pady=5)
        self.first_survey_line_combo.bind('<<ComboboxSelected>>', self.on_first_survey_line_selected)
        
        ttk.Button(survey_line_selection_frame, text='点击测线选择', 
                  command=self.enable_survey_line_click_selection).grid(
            row=1, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        # 添加提示标签
        ttk.Label(survey_line_selection_frame, 
                 text='提示：先完成投放路径优化，\n再根据终点选择第一条测线',
                 foreground='gray', font=('Arial', 8)).grid(
            row=2, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        
        # Station type selection for deployment and recovery
        type_selection_frame = ttk.LabelFrame(side_frame, text='站位类型选择')
        type_selection_frame.grid(row=3, column=0, sticky='ew', pady=(0, 10))
        type_selection_frame.columnconfigure(1, weight=1)
        
        # Deployment types (first row: label + checkboxes)
        deployment_types_label = ttk.Label(type_selection_frame, text='投放站位类型:')
        deployment_types_label.grid(row=0, column=0, sticky='w', padx=5, pady=2)
        
        self.deployment_types_frame = ttk.Frame(type_selection_frame)
        self.deployment_types_frame.grid(row=0, column=1, sticky='w', padx=5, pady=2)
        
        # Recovery types (second row: label + checkboxes)
        recovery_types_label = ttk.Label(type_selection_frame, text='回收站位类型:')
        recovery_types_label.grid(row=1, column=0, sticky='w', padx=5, pady=2)
        
        self.recovery_types_frame = ttk.Frame(type_selection_frame)
        self.recovery_types_frame.grid(row=1, column=1, sticky='w', padx=5, pady=2)
        
        # Optimization method selection (in two rows)
        method_frame = ttk.LabelFrame(side_frame, text='优化方法')
        method_frame.grid(row=4, column=0, sticky='ew', pady=(0, 10))
        method_frame.columnconfigure(0, weight=1)
        method_frame.columnconfigure(1, weight=1)
        
        self.method_var = tk.StringVar(value='nearest_neighbor')
        methods = [
            ('最近邻', 'nearest_neighbor'),
            ('2-opt', '2opt'),
            ('3-opt', '3opt'),
            ('贪心算法', 'greedy')
        ]
        
        for i, (label, value) in enumerate(methods):
            row = i // 2
            col = i % 2
            ttk.Radiobutton(method_frame, text=label, variable=self.method_var,
                          value=value).grid(row=row, column=col, sticky='w', padx=5, pady=2)
        
        # Action buttons (simplified to one row with three buttons)
        action_frame = ttk.LabelFrame(side_frame, text='操作')
        action_frame.grid(row=5, column=0, sticky='ew', pady=(0, 10))
        action_frame.columnconfigure(0, weight=1)
        action_frame.columnconfigure(1, weight=1)
        action_frame.columnconfigure(2, weight=1)
        
        ttk.Button(action_frame, text='优化路径', command=self.optimize_path).grid(
            row=0, column=0, sticky='ew', padx=2, pady=5)
        ttk.Button(action_frame, text='清除路径', command=self.clear_path).grid(
            row=0, column=1, sticky='ew', padx=2, pady=5)
        ttk.Button(action_frame, text='清除全部', command=self.clear_all).grid(
            row=0, column=2, sticky='ew', padx=2, pady=5)
        
        # Path display options (in two rows)
        plot_options_frame = ttk.LabelFrame(side_frame, text='显示选项')
        plot_options_frame.grid(row=6, column=0, sticky='ew', pady=(0, 10))
        plot_options_frame.columnconfigure(0, weight=1)
        plot_options_frame.columnconfigure(1, weight=1)
        
        self.show_deployment_path_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(plot_options_frame, text='投放路径', 
                       variable=self.show_deployment_path_var,
                       command=self.update_path_display).grid(row=0, column=0, sticky='w', padx=5, pady=2)
        
        self.show_recovery_path_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(plot_options_frame, text='回收路径', 
                       variable=self.show_recovery_path_var,
                       command=self.update_path_display).grid(row=0, column=1, sticky='w', padx=5, pady=2)
        
        self.show_deployment_labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(plot_options_frame, text='投放标签', 
                       variable=self.show_deployment_labels_var,
                       command=self.update_path_display).grid(row=1, column=0, sticky='w', padx=5, pady=2)
        
        self.show_recovery_labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(plot_options_frame, text='回收标签', 
                       variable=self.show_recovery_labels_var,
                       command=self.update_path_display).grid(row=1, column=1, sticky='w', padx=5, pady=2)
        
        self.show_survey_lines_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(plot_options_frame, text='测线', 
                       variable=self.show_survey_lines_var,
                       command=self.update_path_display).grid(row=2, column=0, sticky='w', padx=5, pady=2)
        
        self.show_transition_paths_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(plot_options_frame, text='过渡路径', 
                       variable=self.show_transition_paths_var,
                       command=self.update_path_display).grid(row=2, column=1, sticky='w', padx=5, pady=2)
        
        self.show_turn_paths_var = tk.BooleanVar(value=False)  # 默认不显示转弯路径
        ttk.Checkbutton(plot_options_frame, text='转弯路径', 
                       variable=self.show_turn_paths_var,
                       command=self.update_path_display).grid(row=3, column=0, sticky='w', padx=5, pady=2)
        
        # Time calculation parameters
        time_frame = ttk.LabelFrame(side_frame, text='航时计算参数')
        time_frame.grid(row=7, column=0, sticky='ew', pady=(0, 10))
        time_frame.columnconfigure(1, weight=1)
        
        # File loading
        ttk.Label(time_frame, text='水深网格:').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Button(time_frame, text='加载', command=self.load_bathymetry).grid(
            row=0, column=1, sticky='ew', padx=5, pady=2)
        
        ttk.Label(time_frame, text='测线:').grid(row=1, column=0, sticky='w', padx=5, pady=2)
        ttk.Button(time_frame, text='加载', command=self.load_survey_lines).grid(
            row=1, column=1, sticky='ew', padx=5, pady=2)
        
        # Speed parameters
        ttk.Label(time_frame, text='走航速度(节):').grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.cruise_speed_var = tk.StringVar(value='10.0')
        ttk.Entry(time_frame, textvariable=self.cruise_speed_var, width=10).grid(
            row=2, column=1, sticky='ew', padx=5, pady=2)
        
        ttk.Label(time_frame, text='作业速度(节):').grid(row=3, column=0, sticky='w', padx=5, pady=2)
        self.working_speed_var = tk.StringVar(value='5.0')
        ttk.Entry(time_frame, textvariable=self.working_speed_var, width=10).grid(
            row=3, column=1, sticky='ew', padx=5, pady=2)
        
        ttk.Label(time_frame, text='投放时间(秒):').grid(row=4, column=0, sticky='w', padx=5, pady=2)
        self.deployment_time_var = tk.StringVar(value='300.0')
        ttk.Entry(time_frame, textvariable=self.deployment_time_var, width=10).grid(
            row=4, column=1, sticky='ew', padx=5, pady=2)
        
        # Default depth selection
        ttk.Label(time_frame, text='Default Depth (m):').grid(row=5, column=0, sticky='w', padx=5, pady=2)
        self.default_depth_var = tk.StringVar(value='3000')
        default_depth_frame = ttk.Frame(time_frame)
        default_depth_frame.grid(row=5, column=1, sticky='ew', padx=5, pady=2)
        ttk.Radiobutton(default_depth_frame, text='2000', variable=self.default_depth_var, 
                       value='2000').pack(side='left', padx=2)
        ttk.Radiobutton(default_depth_frame, text='3000', variable=self.default_depth_var, 
                       value='3000').pack(side='left', padx=2)
        ttk.Radiobutton(default_depth_frame, text='4000', variable=self.default_depth_var, 
                       value='4000').pack(side='left', padx=2)
        
        ttk.Label(time_frame, text='上浮速度(米/秒):').grid(row=5, column=0, sticky='w', padx=5, pady=2)
        self.ascent_speed_var = tk.StringVar(value='0.5')
        ttk.Entry(time_frame, textvariable=self.ascent_speed_var, width=10).grid(
            row=5, column=1, sticky='ew', padx=5, pady=2)
        
        ttk.Label(time_frame, text='转弯半径(公里):').grid(row=6, column=0, sticky='w', padx=5, pady=2)
        self.turn_radius_var = tk.StringVar(value='5.0')
        turn_combo = ttk.Combobox(time_frame, textvariable=self.turn_radius_var, 
                                  values=['2', '3', '5', '7', '9'], width=8, state='readonly')
        turn_combo.grid(row=6, column=1, sticky='ew', padx=5, pady=2)
        
        ttk.Button(time_frame, text='计算时间', command=self.calculate_time).grid(
            row=7, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        # Recovery planning frame
        recovery_frame = ttk.LabelFrame(side_frame, text='回收站位规划')
        recovery_frame.grid(row=8, column=0, sticky='ew', pady=(0, 10))
        recovery_frame.columnconfigure(1, weight=1)
        
        ttk.Label(recovery_frame, text='回收路径:').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.recovery_path_status_var = tk.StringVar(value='未设置')
        ttk.Label(recovery_frame, textvariable=self.recovery_path_status_var).grid(
            row=0, column=1, sticky='w', padx=5, pady=2)
        
        ttk.Label(recovery_frame, text='站位深度(米):').grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.station_depths_var = tk.StringVar(value='')
        self.station_depths_entry = ttk.Entry(recovery_frame, textvariable=self.station_depths_var, width=20)
        self.station_depths_entry.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        ttk.Label(recovery_frame, text='(优先从文件第6列读取，否则从水深文件读取)').grid(
            row=2, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        
        ttk.Label(recovery_frame, text='上浮速度(米/秒):').grid(row=3, column=0, sticky='w', padx=5, pady=2)
        self.recovery_ascent_speed_var = tk.StringVar(value='0.5')
        self.recovery_ascent_speed_entry = ttk.Entry(recovery_frame, textvariable=self.recovery_ascent_speed_var, width=10)
        self.recovery_ascent_speed_entry.grid(row=3, column=1, sticky='ew', padx=5, pady=2)
        ttk.Label(recovery_frame, text='(从文件第5列读取，如无则使用默认值0.5)').grid(
            row=4, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        
        ttk.Label(recovery_frame, text='船速(节):').grid(row=5, column=0, sticky='w', padx=5, pady=2)
        self.recovery_ship_speed_var = tk.StringVar(value='10.0')
        ttk.Entry(recovery_frame, textvariable=self.recovery_ship_speed_var, width=10).grid(
            row=5, column=1, sticky='ew', padx=5, pady=2)
        
        ttk.Label(recovery_frame, text='滚动站位数:').grid(row=6, column=0, sticky='w', padx=5, pady=2)
        self.rolling_stations_var = tk.StringVar(value='2')
        rolling_combo = ttk.Combobox(recovery_frame, textvariable=self.rolling_stations_var,
                                    values=['2', '3'], width=8, state='readonly')
        rolling_combo.grid(row=6, column=1, sticky='ew', padx=5, pady=2)
        
        ttk.Button(recovery_frame, text='规划回收策略', command=self.plan_recovery).grid(
            row=7, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        # Results display
        result_frame = ttk.LabelFrame(side_frame, text='结果')
        result_frame.grid(row=9, column=0, sticky='nsew', pady=(0, 10))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        side_frame.rowconfigure(9, weight=1)
        
        # Create text display area
        self.result_text = tk.Text(result_frame, wrap=tk.WORD, width=30, height=15)
        self.result_text.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.result_text.config(yscrollcommand=scrollbar.set)
        
        # Save time results button
        save_time_button = ttk.Button(result_frame, text='保存时间结果', command=self.save_time_results)
        save_time_button.grid(row=1, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        # 点击选择模式
        self.click_selection_mode = None  # 'start', 'end', or None
        self.click_cid = None
    
    @staticmethod
    def get_station_type_color(station_type: str) -> str:
        """
        根据台站类型返回对应的颜色
        
        Parameters
        ----------
        station_type : str
            台站类型
            
        Returns
        -------
        color : str
            颜色代码
        """
        # 定义台站类型到颜色的映射
        type_color_map = {
            'A': 'blue',      # 类型A - 蓝色
            'B': 'green',     # 类型B - 绿色
            'C': 'orange',    # 类型C - 橙色
            'D': 'purple',    # 类型D - 紫色
            'E': 'brown',     # 类型E - 棕色
            'F': 'pink',      # 类型F - 粉色
            'G': 'gray',      # 类型G - 灰色
            'H': 'cyan',      # 类型H - 青色
            'I': 'red',       # 类型I - 红色
            'J': 'yellow',    # 类型J - 黄色
            'K': 'magenta',   # 类型K - 品红色
            'L': 'lime',      # 类型L - 酸橙绿
            'M': 'navy',      # 类型M - 海军蓝
            'N': 'olive',     # 类型N - 橄榄绿
            'O': 'teal',      # 类型O - 青绿色
            'P': 'coral',     # 类型P - 珊瑚色
        }
        
        # 如果类型在映射中，返回对应颜色；否则使用默认颜色
        station_type_upper = station_type.strip().upper() if station_type else ''
        if not station_type_upper:
            return 'gray'  # 未分类使用灰色
        return type_color_map.get(station_type_upper, 'blue')  # 默认蓝色
    
    def load_stations_file(self):
        """加载站位文件（支持第三列站位名称）"""
        filename = filedialog.askopenfilename(
            title='选择站位文件',
            filetypes=[
                ('文本文件', '*.txt'),
                ('CSV文件', '*.csv'),
                ('所有文件', '*.*')
            ]
        )
        
        if not filename:
            return
        
        try:
            # 加载站位数据（支持第三列站位名称）
            stations, station_names = self._load_stations_with_names(filename)
            if len(stations) < 2:
                messagebox.showerror('错误', '至少需要2个站位点')
                return
            
            self.stations = stations
            self.station_names = station_names
            # 确保station_names长度与stations一致
            while len(self.station_names) < len(self.stations):
                self.station_names.append('')
            # station_types已在_load_stations_with_names中设置，确保长度一致
            if not hasattr(self, 'station_types'):
                self.station_types = []
            while len(self.station_types) < len(self.stations):
                self.station_types.append('')
            self.optimizer = None
            self.start_idx = None
            self.end_idx = None
            self.selected_start_idx = None
            self.selected_end_idx = None
            
            # 更新起点和终点下拉菜单
            self.update_station_combos()
            
            # 更新站位类型选择界面
            self.update_station_type_selection()
            
            # 更新回收规划界面的深度和上浮速度显示
            if hasattr(self, 'station_depths_var') and hasattr(self, 'station_depths_from_file'):
                if any(d is not None for d in self.station_depths_from_file):
                    # 如果有从文件读取的深度，显示在输入框中（只读提示）
                    depths_str = ','.join([f'{d:.1f}' if d is not None else '0' 
                                          for d in self.station_depths_from_file])
                    self.station_depths_var.set(depths_str)
                    self.station_depths_entry.config(state='readonly')
                else:
                    self.station_depths_entry.config(state='normal')
            
            if hasattr(self, 'recovery_ascent_speed_var') and hasattr(self, 'station_ascent_speeds'):
                if any(s is not None for s in self.station_ascent_speeds):
                    # 如果有从文件读取的上浮速度，使用平均值
                    valid_speeds = [s for s in self.station_ascent_speeds if s is not None]
                    if valid_speeds:
                        avg_speed = np.mean(valid_speeds)
                        self.recovery_ascent_speed_var.set(f'{avg_speed:.2f}')
                        self.recovery_ascent_speed_entry.config(state='readonly')
                else:
                    if hasattr(self, 'recovery_ascent_speed_entry'):
                        self.recovery_ascent_speed_entry.config(state='normal')
            
            # 绘制站位
            self.plot_stations()
            
            self.log_result(f"成功加载 {len(stations)} 个站位点")
            self.log_result(f"文件: {filename}")
            if any(self.station_types):
                unique_types = set([t for t in self.station_types if t])
                self.log_result(f"已读取台站类型: {len(unique_types)} 种类型 - {', '.join(sorted(unique_types))}")
            
            # 处理站位深度：优先使用文件第6列，否则从水深文件读取
            self.station_depths = []
            has_depth_from_file = any(d is not None for d in self.station_depths_from_file)
            
            if has_depth_from_file:
                # 使用文件第6列的深度
                self.station_depths = [d if d is not None else 0.0 for d in self.station_depths_from_file]
                avg_depth = np.mean([d for d in self.station_depths if d > 0])
                self.log_result(f"已从站位文件读取深度，平均水深: {avg_depth:.1f} 米")
            elif self.bathymetry_data is not None or (hasattr(self, 'bathymetry_filename') and self.bathymetry_filename):
                # 从水深文件读取
                if self.bathymetry_data is not None:
                    self._interpolate_station_depths()
                elif hasattr(self, 'bathymetry_filename') and self.bathymetry_filename:
                    # 对于.grd或.nc文件，延迟读取
                    pass  # 将在需要时读取
                if self.station_depths:
                    avg_depth = np.mean(self.station_depths)
                    self.log_result(f"已从水深文件计算站位水深，平均水深: {avg_depth:.1f} 米")
            
            # 检查是否有上浮速度数据
            has_ascent_speed = any(s is not None for s in self.station_ascent_speeds)
            if has_ascent_speed:
                avg_speed = np.mean([s for s in self.station_ascent_speeds if s is not None])
                self.log_result(f"已从站位文件读取上浮速度，平均速度: {avg_speed:.2f} 米/秒")
            
        except Exception as e:
            messagebox.showerror('错误', f'加载文件失败: {str(e)}')
            import traceback
            traceback.print_exc()
    
    def _load_stations_with_names(self, filename: str) -> Tuple[List[Tuple[float, float]], List[str],List[float]]:
        """
        从文件加载站位坐标和台站类型（第三列为台站类型）
        
        支持格式：
        1. CSV文件：经度, 纬度, 台站类型 或 经度,纬度,台站类型
        2. 文本文件：每行 经度 纬度 台站类型（空格或制表符分隔）
        
        列说明：
        - 第1列：经度
        - 第2列：纬度
        - 第3列：台站类型（可选）
        - 第4列：站位名称（可选）
        - 第5列：上浮速度（米/秒，可选）
        - 第6列：站位深度（米，可选）
        
        Parameters
        ----------
        filename : str
            文件路径
            
        Returns
        -------
        stations : List[Tuple[float, float]]
            站位坐标列表，格式为 [(lon1, lat1), (lon2, lat2), ...]
        station_names : List[str]
            站位名称列表（保留用于兼容性，实际存储台站类型），格式为 ['type1', 'type2', ...]，如果某行没有类型则为空字符串
        """
        stations = []
        station_names = []
        station_types = []
        station_beams = []
        station_ascent_speeds = []
        station_depths_from_file = []
        station_all_columns = []  # 保存所有列数据
        
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
                            
                            # 保存所有列数据
                            all_cols = parts.copy()
                            station_all_columns.append(all_cols)
                            
                            # 解析各列数据
                            # 第1列：经度 (parts[0])
                            # 第2列：纬度 (parts[1])
                            # 第3列：台站类型 (parts[2])
                            # 第4列：站位名称 (parts[3])
                            # 第5列：上浮速度 (parts[4])
                            # 第6列：站位深度 (parts[5])
                            station_type = parts[2].strip() if len(parts) > 2 else ''
                            station_name = parts[3].strip() if len(parts) > 3 else ''
                            # 兼容旧格式：如果第4列是数字，可能是beams；否则是名称
                            beams = ''
                            if len(parts) > 3:
                                try:
                                    # 尝试将第4列解析为数字（beams）
                                    beams = float(parts[3])
                                except ValueError:
                                    # 第4列不是数字，是名称，beams为空
                                    beams = ''
                            ascent_speed = float(parts[4]) if len(parts) > 4 and parts[4].strip() else None  # 第5列：上浮速度
                            depth = float(parts[5]) if len(parts) > 5 and parts[5].strip() else None  # 第6列：站位深度
                            
                            station_types.append(station_type)
                            station_names.append(station_name if station_name else station_type)  # 如果有名称用名称，否则用类型
                            station_beams.append(beams)
                            station_ascent_speeds.append(ascent_speed)
                            station_depths_from_file.append(depth)
                        except ValueError as e:
                            print(f"警告: 第{line_num}行格式错误，已跳过: {line} (错误: {e})")
                            continue
            
            # 确保所有列表长度与stations一致
            while len(station_names) < len(stations):
                station_names.append('')
            while len(station_types) < len(stations):
                station_types.append('')
            while len(station_beams) < len(stations):
                station_beams.append('')
            while len(station_ascent_speeds) < len(stations):
                station_ascent_speeds.append(None)
            while len(station_depths_from_file) < len(stations):
                station_depths_from_file.append(None)
            while len(station_all_columns) < len(stations):
                station_all_columns.append([])
            
            # 保存到实例变量
            self.station_types = station_types
            self.station_beams = station_beams
            self.station_ascent_speeds = station_ascent_speeds
            self.station_depths_from_file = station_depths_from_file
            self.station_all_columns = station_all_columns
        
        except FileNotFoundError:
            raise FileNotFoundError(f"文件未找到: {filename}")
        except Exception as e:
            raise Exception(f"读取文件时出错: {e}")
        
        return stations, station_names
    
    def load_bathymetry(self):
        """加载水深网格文件"""
        filename = filedialog.askopenfilename(
            title='选择水深网格文件',
            filetypes=[
                ('GMT网格文件', '*.grd'),
                ('NetCDF文件', '*.nc'),
                ('文本文件', '*.txt'),
                ('CSV文件', '*.csv'),
                ('所有文件', '*.*')
            ]
        )
        
        if not filename:
            return
        
        try:
            ext = os.path.splitext(filename)[1].lower()
            
            if ext == '.grd':
                # GMT网格文件，可能是NetCDF格式或GMT二进制格式
                # 对于.grd文件，只保存文件路径和范围信息，不加载全部数据
                # 在绘图时再按需读取需要的区域
                success = False
                last_error = None
                
                # 方法1: 优先使用pygmt读取文件头信息（只获取范围，不加载数据）
                try:
                    import pygmt
                    # 使用pygmt.grdinfo获取网格信息（不加载数据）
                    info = pygmt.grdinfo(filename, C='n', D='n')
                    # 解析范围信息：格式通常是 "xmin xmax ymin ymax zmin zmax"
                    # 或者使用pygmt.grdinfo的-R选项直接获取范围
                    region_info = pygmt.grdinfo(filename, R=True)
                    # region_info格式：-Rlon_min/lon_max/lat_min/lat_max
                    if region_info.startswith('-R'):
                        coords = region_info[2:].split('/')
                        if len(coords) >= 4:
                            self.bathymetry_full_range = [
                                float(coords[0]),  # lon_min
                                float(coords[1]),  # lon_max
                                float(coords[2]),  # lat_min
                                float(coords[3])   # lat_max
                            ]
                            self.bathymetry_filename = filename
                            self.bathymetry_file_type = 'grd'
                            self.bathymetry_data = None  # 不加载全部数据
                            self.bathymetry_lon = None
                            self.bathymetry_lat = None
                            success = True
                except ImportError:
                    last_error = ImportError('pygmt未安装')
                except Exception as e:
                    last_error = e
                    # 如果pygmt.grdinfo失败，尝试使用xarray只读取坐标信息
                    try:
                        import xarray as xr
                        # 只打开数据集，不加载数据
                        with xr.open_dataset(filename, decode_times=False) as ds:
                            if 'lon' in ds.coords and 'lat' in ds.coords:
                                lon_min = float(ds.lon.min().values)
                                lon_max = float(ds.lon.max().values)
                                lat_min = float(ds.lat.min().values)
                                lat_max = float(ds.lat.max().values)
                                self.bathymetry_full_range = [lon_min, lon_max, lat_min, lat_max]
                                self.bathymetry_filename = filename
                                self.bathymetry_file_type = 'grd'
                                self.bathymetry_data = None
                                self.bathymetry_lon = None
                                self.bathymetry_lat = None
                                success = True
                    except Exception:
                        pass
                
                # 如果所有方法都失败，显示错误信息
                if not success:
                    error_msg = f'读取.grd文件信息失败:\n\n'
                    if last_error:
                        error_msg += f'最后错误: {str(last_error)}\n\n'
                    error_msg += f'建议:\n'
                    error_msg += f'1. 确保已安装pygmt: pip install pygmt\n'
                    error_msg += f'2. 或安装xarray: pip install xarray\n'
                    error_msg += f'3. 或安装GMT命令行工具'
                    messagebox.showerror('错误', error_msg)
                    return
            
            elif ext == '.nc':
                # 对于.nc文件，只读取范围信息，不加载全部数据
                # 在绘图时再按需读取需要的区域
                try:
                    import xarray as xr
                    # 只打开数据集，不加载数据
                    with xr.open_dataset(filename, decode_times=False) as ds:
                        # 获取坐标范围（不加载数据）
                        if 'lon' in ds.coords and 'lat' in ds.coords:
                            lon_min = float(ds.lon.min().values)
                            lon_max = float(ds.lon.max().values)
                            lat_min = float(ds.lat.min().values)
                            lat_max = float(ds.lat.max().values)
                            
                            # 查找深度变量名（不加载数据）
                            depth_var_name = None
                            depth_vars = ['depth', 'bathymetry', 'z', 'elevation', 'topo']
                            for var in depth_vars:
                                if var in ds.variables:
                                    depth_var_name = var
                                    break
                            
                            # 如果没找到，使用第一个数据变量
                            if depth_var_name is None:
                                data_vars = list(ds.data_vars.keys())
                                if data_vars:
                                    depth_var_name = data_vars[0]
                            
                            if depth_var_name:
                                self.bathymetry_full_range = [lon_min, lon_max, lat_min, lat_max]
                                self.bathymetry_filename = filename
                                self.bathymetry_file_type = 'nc'
                                self.bathymetry_data = None  # 不加载全部数据
                                self.bathymetry_lon = None
                                self.bathymetry_lat = None
                                # 保存变量名，用于后续读取
                                self.bathymetry_var_name = depth_var_name
                                self.bathymetry_lon_name = 'lon'
                                self.bathymetry_lat_name = 'lat'
                            else:
                                raise ValueError('未找到有效的数据变量')
                        else:
                            # 尝试从变量中获取坐标
                            coords = list(ds.coords.keys())
                            if len(coords) >= 2:
                                lon_coord = coords[0]
                                lat_coord = coords[1]
                                lon_min = float(ds[lon_coord].min().values)
                                lon_max = float(ds[lon_coord].max().values)
                                lat_min = float(ds[lat_coord].min().values)
                                lat_max = float(ds[lat_coord].max().values)
                                
                                # 获取第一个数据变量
                                data_vars = list(ds.data_vars.keys())
                                if data_vars:
                                    depth_var_name = data_vars[0]
                                    self.bathymetry_full_range = [lon_min, lon_max, lat_min, lat_max]
                                    self.bathymetry_filename = filename
                                    self.bathymetry_file_type = 'nc'
                                    self.bathymetry_data = None
                                    self.bathymetry_lon = None
                                    self.bathymetry_lat = None
                                    self.bathymetry_var_name = depth_var_name
                                    self.bathymetry_lon_name = lon_coord
                                    self.bathymetry_lat_name = lat_coord
                                else:
                                    raise ValueError('未找到有效的数据变量')
                            else:
                                raise ValueError('无法识别坐标变量')
                except ImportError:
                    messagebox.showerror('错误', '需要安装xarray库来读取NetCDF文件\n\n请运行: pip install xarray')
                    return
                except Exception as e:
                    messagebox.showerror('错误', f'读取NetCDF文件失败: {str(e)}')
                    return
            else:
                # 读取文本或CSV格式的水深网格
                # 格式：经度 纬度 水深（每行一个点）
                lons = []
                lats = []
                depths = []
                
                with open(filename, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        
                        if ',' in line:
                            parts = [p.strip() for p in line.split(',')]
                        else:
                            parts = line.split()
                        
                        if len(parts) >= 3:
                            try:
                                lon, lat, depth = float(parts[0]), float(parts[1]), float(parts[2])
                                lons.append(lon)
                                lats.append(lat)
                                depths.append(depth)
                            except ValueError:
                                continue
                
                if not lons:
                    messagebox.showerror('错误', '未能读取有效的水深数据')
                    return
                
                # 转换为网格（如果数据不是网格格式，需要插值）
                lons = np.array(lons)
                lats = np.array(lats)
                depths = np.array(depths)
                
                # 检查是否是规则网格
                unique_lons = np.unique(lons)
                unique_lats = np.unique(lats)
                
                if len(unique_lons) * len(unique_lats) == len(lons):
                    # 是规则网格
                    self.bathymetry_lon = unique_lons
                    self.bathymetry_lat = unique_lats
                    self.bathymetry_data = depths.reshape(len(unique_lats), len(unique_lons))
                else:
                    # 不规则数据，需要插值到规则网格
                    messagebox.showinfo('提示', '检测到不规则网格数据，将插值到规则网格')
                    # 创建规则网格
                    lon_min, lon_max = lons.min(), lons.max()
                    lat_min, lat_max = lats.min(), lats.max()
                    grid_resolution = 0.01  # 约1km分辨率
                    self.bathymetry_lon = np.arange(lon_min, lon_max + grid_resolution, grid_resolution)
                    self.bathymetry_lat = np.arange(lat_min, lat_max + grid_resolution, grid_resolution)
                    lon_grid, lat_grid = np.meshgrid(self.bathymetry_lon, self.bathymetry_lat)
                    # 插值
                    self.bathymetry_data = griddata((lons, lats), depths, 
                                                   (lon_grid, lat_grid), method='linear')
                self.bathymetry_filename = filename
                self.bathymetry_file_type = 'txt'
            
            # 清除水深图缓存（数据已更新）
            self._bathymetry_cache = None
            self._bathymetry_cache_extent = None
            self._bathymetry_cache_hash = None
            
            # 计算站位水深（对于.grd和.nc文件，延迟到需要时再读取）
            if self.stations and self.bathymetry_data is not None:
                self._interpolate_station_depths()
            
            # 显示加载成功信息
            if self.bathymetry_file_type in ['grd', 'nc']:
                # 对于.grd和.nc文件，只加载了范围信息
                if self.bathymetry_full_range:
                    lon_min, lon_max, lat_min, lat_max = self.bathymetry_full_range
                    messagebox.showinfo('成功', f'成功加载水深网格文件\n\n文件类型: {self.bathymetry_file_type.upper()}\n'
                                              f'经度范围: {lon_min:.4f}° ~ {lon_max:.4f}°\n'
                                              f'纬度范围: {lat_min:.4f}° ~ {lat_max:.4f}°\n\n'
                                              f'注意: 数据将按需读取，提高性能')
                    self.log_result(f"成功加载水深网格: {filename}")
                    self.log_result(f"文件类型: {self.bathymetry_file_type.upper()}")
                    self.log_result(f"经度范围: {lon_min:.4f}° ~ {lon_max:.4f}°")
                    self.log_result(f"纬度范围: {lat_min:.4f}° ~ {lat_max:.4f}°")
                    self.log_result(f"注意: 数据将按需读取，绘图时只加载需要的区域")
                else:
                    messagebox.showinfo('成功', f'成功加载水深网格文件\n\n文件类型: {self.bathymetry_file_type.upper()}')
                    self.log_result(f"成功加载水深网格: {filename}")
            else:
                # 对于.txt文件，已加载全部数据
                data_shape = self.bathymetry_data.shape if self.bathymetry_data is not None else (0, 0)
                messagebox.showinfo('成功', f'成功加载水深网格文件\n\n网格大小: {data_shape}')
                self.log_result(f"成功加载水深网格: {filename}")
                if self.bathymetry_data is not None:
                    self.log_result(f"水深范围: {np.nanmin(self.bathymetry_data):.1f} ~ {np.nanmax(self.bathymetry_data):.1f} 米")
                    # 如果数据很大，提示会进行降采样
                    if data_shape[0] > 200 or data_shape[1] > 200:
                        self.log_result(f"注意: 数据较大，将自动降采样以提高显示性能")
            
            # 重新绘制以显示水深图
            if self.stations:
                # 更新界面，然后绘制
                self.root.update_idletasks()
                self.plot_stations()
            
        except Exception as e:
            messagebox.showerror('错误', f'加载水深文件失败: {str(e)}')
            import traceback
            traceback.print_exc()
    
    def load_survey_lines(self):
        """加载作业测线坐标文件"""
        filename = filedialog.askopenfilename(
            title='选择作业测线文件',
            filetypes=[
                ('文本文件', '*.txt'),
                ('CSV文件', '*.csv'),
                ('所有文件', '*.*')
            ]
        )
        
        if not filename:
            return
        
        try:
            survey_lines = []
            current_line = []
            
            with open(filename, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        # 空行或注释行表示一条测线结束
                        if current_line:
                            survey_lines.append(current_line)
                            current_line = []
                        continue
                    
                    if ',' in line:
                        parts = [p.strip() for p in line.split(',')]
                    else:
                        parts = line.split()
                    
                    if len(parts) >= 2:
                        try:
                            lon, lat = float(parts[0]), float(parts[1])
                            current_line.append((lon, lat))
                        except ValueError:
                            continue
                
                # 添加最后一条测线
                if current_line:
                    survey_lines.append(current_line)
            
            if not survey_lines:
                messagebox.showerror('错误', '未能读取有效的测线数据')
                return
            
            self.survey_lines = survey_lines
            # 更新第一条测线下拉菜单
            self.update_survey_line_combo()
            messagebox.showinfo('成功', f'成功加载 {len(survey_lines)} 条作业测线')
            self.log_result(f"成功加载作业测线: {filename}")
            self.log_result(f"共 {len(survey_lines)} 条测线")
            
            # 计算总测线长度
            total_length = 0.0
            for line in survey_lines:
                for i in range(len(line) - 1):
                    lon1, lat1 = line[i]
                    lon2, lat2 = line[i + 1]
                    dist = StationPathOptimizer.haversine_distance(lon1, lat1, lon2, lat2)
                    total_length += dist
            self.log_result(f"总测线长度: {total_length:.2f} km")
            
            # 重新绘制以显示测线
            if self.stations:
                self.plot_stations()
            
        except Exception as e:
            messagebox.showerror('错误', f'加载测线文件失败: {str(e)}')
            import traceback
            traceback.print_exc()
    
    def _interpolate_station_depths(self):
        """根据水深网格插值计算每个站位的水深"""
        if not self.stations:
            return
        
        # 对于.grd和.nc文件，延迟到需要时再读取（在计算航时时读取）
        if self.bathymetry_file_type in ['grd', 'nc'] and self.bathymetry_filename:
            # 暂时不计算，在calculate_time()中按需读取
            self.station_depths = []
            return
        
        # 对于其他格式，使用已加载的数据
        if self.bathymetry_data is None or self.bathymetry_lon is None or self.bathymetry_lat is None:
            return
        
        self.station_depths = []
        
        # 创建网格坐标
        lon_grid, lat_grid = np.meshgrid(self.bathymetry_lon, self.bathymetry_lat)
        points = np.column_stack((lon_grid.flatten(), lat_grid.flatten()))
        values = self.bathymetry_data.flatten()
        
        # 移除NaN值
        valid_mask = ~np.isnan(values)
        points = points[valid_mask]
        values = values[valid_mask]
        
        for lon, lat in self.stations:
            # 使用线性插值
            try:
                depth = griddata(points, values, (lon, lat), method='linear')
                if np.isnan(depth):
                    # 如果线性插值失败，使用最近邻
                    depth = griddata(points, values, (lon, lat), method='nearest')
                default_depth = float(self.default_depth_var.get()) if hasattr(self, 'default_depth_var') else 3000.0
                self.station_depths.append(abs(depth) if not np.isnan(depth) else default_depth)  # 水深取绝对值
            except:
                # 如果插值失败，使用最近网格点
                lon_idx = np.argmin(np.abs(self.bathymetry_lon - lon))
                lat_idx = np.argmin(np.abs(self.bathymetry_lat - lat))
                default_depth = float(self.default_depth_var.get()) if hasattr(self, 'default_depth_var') else 3000.0
                if 0 <= lat_idx < self.bathymetry_data.shape[0] and 0 <= lon_idx < self.bathymetry_data.shape[1]:
                    depth = self.bathymetry_data[lat_idx, lon_idx]
                    self.station_depths.append(abs(depth) if not np.isnan(depth) else default_depth)
                else:
                    # 使用用户选择的默认水深
                    self.station_depths.append(default_depth)
    
    def update_station_combos(self):
        """更新起点和终点下拉菜单"""
        if not self.stations:
            self.start_combo['values'] = []
            self.end_combo['values'] = []
            self.start_var.set('未选择')
            self.end_var.set('未选择')
            return
        
        # 创建选项列表（如果有站位名称则显示名称，否则显示索引）
        station_options = []
        for i in range(len(self.stations)):
            lon, lat = self.stations[i]
            if i < len(self.station_names) and self.station_names[i]:
                # 有站位名称时显示：站位名 (经度, 纬度)
                station_options.append(f'{self.station_names[i]} ({lon:.4f}, {lat:.4f})')
            else:
                # 没有站位名称时显示：站位 索引 (经度, 纬度)
                station_options.append(f'站位 {i} ({lon:.4f}, {lat:.4f})')
        station_options.insert(0, '未选择')
        
        self.start_combo['values'] = station_options
        self.end_combo['values'] = station_options
        
        # 如果之前有选择，尝试恢复
        if self.selected_start_idx is not None:
            if self.selected_start_idx < len(self.stations):
                self.start_combo.current(self.selected_start_idx + 1)
                self.on_start_selected()
        
        if self.selected_end_idx is not None:
            if self.selected_end_idx < len(self.stations):
                self.end_combo.current(self.selected_end_idx + 1)
                self.on_end_selected()
    
    def update_survey_line_combo(self):
        """更新第一条测线下拉菜单"""
        if not self.survey_lines:
            self.first_survey_line_combo['values'] = []
            self.first_survey_line_var.set('未选择')
            return
        
        # 创建测线选项列表（只包含有效测线）
        line_options = []
        valid_line_indices = []  # 记录有效测线的原始索引
        for i, line in enumerate(self.survey_lines):
            if len(line) >= 2:
                start_lon, start_lat = line[0]
                end_lon, end_lat = line[-1]
                line_options.append(f'测线 {i} (起点: {start_lon:.4f}°, {start_lat:.4f}°)')
                valid_line_indices.append(i)
        line_options.insert(0, '未选择')
        
        self.first_survey_line_combo['values'] = line_options
        # 保存有效测线索引映射（用于调试）
        self.valid_survey_line_indices = valid_line_indices
        
        # 如果之前有选择，尝试恢复
        if self.selected_first_survey_line_idx is not None:
            if self.selected_first_survey_line_idx < len(self.survey_lines):
                # 找到该测线在下拉菜单中的索引
                if self.selected_first_survey_line_idx in valid_line_indices:
                    combo_index = valid_line_indices.index(self.selected_first_survey_line_idx) + 1
                    self.first_survey_line_combo.current(combo_index)
                    self.on_first_survey_line_selected()
                else:
                    self.log_result(f"[update_survey_line_combo] 警告：测线 {self.selected_first_survey_line_idx} 不在有效测线列表中")
    
    def on_first_survey_line_selected(self, event=None):
        """第一条测线选择事件"""
        selection = self.first_survey_line_combo.current()
        self.log_result(f"[on_first_survey_line_selected] 下拉菜单选择值: {selection}")
        self.log_result(f"[on_first_survey_line_selected] 调用前的 selected_first_survey_line_idx: {self.selected_first_survey_line_idx}")
        
        if selection == 0:  # "未选择"
            self.log_result(f"[on_first_survey_line_selected] 选择为'未选择'，清除选择")
            self.selected_first_survey_line_idx = None
            self.plot_stations()
        else:
            # 从下拉菜单索引映射到实际测线索引
            if hasattr(self, 'valid_survey_line_indices') and self.valid_survey_line_indices:
                # 下拉菜单索引从1开始（0是"未选择"），所以 selection-1 是有效测线列表中的索引
                if selection - 1 < len(self.valid_survey_line_indices):
                    new_selected_idx = self.valid_survey_line_indices[selection - 1]
                    self.log_result(f"[on_first_survey_line_selected] 使用有效测线索引映射: 下拉索引 {selection} -> 测线索引 {new_selected_idx}")
                else:
                    self.log_result(f"[on_first_survey_line_selected] ⚠ 错误：下拉索引 {selection} 超出范围！")
                    return
            else:
                # 如果没有有效索引列表，使用简单映射（假设所有测线都有效）
                new_selected_idx = selection - 1
                self.log_result(f"[on_first_survey_line_selected] 使用简单映射: selection={selection} - 1 = {new_selected_idx}")
            
            self.log_result(f"[on_first_survey_line_selected] 计算得到测线索引: {new_selected_idx}")
            
            # 如果之前已经有选择且不一致，记录警告
            if self.selected_first_survey_line_idx is not None and self.selected_first_survey_line_idx != new_selected_idx:
                self.log_result(f"[on_first_survey_line_selected] ⚠ 警告：值不一致！之前: {self.selected_first_survey_line_idx}, 现在: {new_selected_idx}")
            
            self.selected_first_survey_line_idx = new_selected_idx
            self.log_result(f"[on_first_survey_line_selected] 设置后的 selected_first_survey_line_idx: {self.selected_first_survey_line_idx}")
            self.log_result(f"已选择第一条测线: 测线 {self.selected_first_survey_line_idx}")
            
            # 如果已经完成第一阶段优化（投放路径），自动执行第二阶段和第三阶段优化
            if self.deployment_path is not None and len(self.deployment_path) > 0:
                self.log_result(f"[on_first_survey_line_selected] 检测到已完成第一阶段优化，调用 optimize_survey_and_recovery_path")
                self.optimize_survey_and_recovery_path()
            else:
                self.log_result(f"[on_first_survey_line_selected] 尚未完成第一阶段优化，只重新绘制")
                self.plot_stations()
    
    def enable_survey_line_click_selection(self):
        """启用点击选择第一条测线模式"""
        if not self.survey_lines:
            messagebox.showwarning('警告', '请先加载作业测线文件')
            return
        
        # 检查是否已完成第一阶段优化
        if self.deployment_path is None or len(self.deployment_path) == 0:
            messagebox.showwarning('提示', '请先完成第一阶段优化（投放路径）\n\n选择投放起点后，点击"优化路径"按钮')
            return
        
        self.click_selection_mode = 'first_survey_line'
        if self.click_cid is not None:
            self.canvas.mpl_disconnect(self.click_cid)
        self.click_cid = self.canvas.mpl_connect('button_press_event', self.on_survey_line_click)
        messagebox.showinfo('提示', '点击选择模式已启用\n\n请根据投放终点选择一条测线作为第一条测线\n\n系统将自动确定测线起点并完成后续优化')
    
    def on_survey_line_click(self, event):
        """处理测线点击事件"""
        # 调试信息：记录点击事件
        self.log_result(f"\n[调试] 鼠标点击事件触发")
        self.log_result(f"[调试] 点击模式: {self.click_selection_mode}")
        self.log_result(f"[调试] 点击坐标: ({event.xdata}, {event.ydata})")
        
        if event.inaxes != self.ax:
            self.log_result(f"[调试] 点击不在坐标轴内")
            return
        
        if self.click_selection_mode != 'first_survey_line':
            self.log_result(f"[调试] 当前不是测线选择模式")
            return
        
        if event.button != 1:  # 只响应左键点击
            self.log_result(f"[调试] 不是左键点击 (button={event.button})")
            return
        
        # 找到距离点击位置最近的测线
        click_lon, click_lat = event.xdata, event.ydata
        if click_lon is None or click_lat is None:
            self.log_result(f"[调试] 点击坐标无效")
            return
        
        self.log_result(f"[调试] 开始查找最近测线，点击位置: ({click_lon:.6f}, {click_lat:.6f})")
        self.log_result(f"[调试] 总测线数: {len(self.survey_lines)}")
        
        min_distance = float('inf')
        nearest_line_idx = None
        line_distances = []  # 记录每条测线的最小距离，用于调试
        
        for i, line in enumerate(self.survey_lines):
            if len(line) < 2:
                continue
            
            # 计算点击位置到测线线段的最短距离（点到线段距离）
            line_min_distance = float('inf')
            
            # 遍历测线的所有线段（相邻两点之间的线段）
            for j in range(len(line) - 1):
                p1 = line[j]
                p2 = line[j + 1]
                
                # 计算点到线段的最短距离
                # 使用Haversine公式计算距离
                # 先计算点到两个端点的距离
                dist_to_p1 = StationPathOptimizer.haversine_distance(
                    click_lon, click_lat, p1[0], p1[1]
                )
                dist_to_p2 = StationPathOptimizer.haversine_distance(
                    click_lon, click_lat, p2[0], p2[1]
                )
                
                # 计算线段长度
                segment_length = StationPathOptimizer.haversine_distance(
                    p1[0], p1[1], p2[0], p2[1]
                )
                
                # 如果线段长度为0，使用点到点的距离
                if segment_length < 1e-6:
                    segment_distance = min(dist_to_p1, dist_to_p2)
                else:
                    # 计算点到线段的垂直距离
                    # 使用向量投影方法（近似，因为Haversine距离不是欧几里得距离）
                    # 这里使用简化的方法：计算点到线段上最近点的距离
                    # 通过插值找到线段上最近的点
                    # 使用参数t表示在线段上的位置 (0到1之间)
                    # 计算点击点到线段上各点的距离，找到最小值
                    # 为了简化，我们采样线段上的多个点
                    n_samples = max(10, int(segment_length * 100))  # 每100米一个采样点
                    segment_distance = float('inf')
                    for k in range(n_samples + 1):
                        t = k / n_samples
                        # 线性插值（经纬度）
                        interp_lon = p1[0] + t * (p2[0] - p1[0])
                        interp_lat = p1[1] + t * (p2[1] - p1[1])
                        dist = StationPathOptimizer.haversine_distance(
                            click_lon, click_lat, interp_lon, interp_lat
                        )
                        if dist < segment_distance:
                            segment_distance = dist
                    
                    # 也考虑两个端点
                    segment_distance = min(segment_distance, dist_to_p1, dist_to_p2)
                
                if segment_distance < line_min_distance:
                    line_min_distance = segment_distance
            
            line_distances.append((i, line_min_distance))
            self.log_result(f"[调试] 测线 {i}: 最小距离 = {line_min_distance:.6f} km")
            
            if line_min_distance < min_distance:
                min_distance = line_min_distance
                nearest_line_idx = i
        
        # 按距离排序，显示所有测线的距离（用于调试）
        line_distances.sort(key=lambda x: x[1])
        self.log_result(f"[调试] 测线距离排序（从小到大）:")
        for idx, dist in line_distances[:5]:  # 只显示前5个
            self.log_result(f"[调试]   测线 {idx}: {dist:.6f} km")
        
        if nearest_line_idx is not None:
            self.log_result(f"[调试] ✓ 找到最近测线: 测线 {nearest_line_idx}, 距离: {min_distance:.6f} km")
            
            # 检查该测线是否在下拉菜单中（验证索引映射）
            if hasattr(self, 'valid_survey_line_indices'):
                if nearest_line_idx not in self.valid_survey_line_indices:
                    self.log_result(f"[调试] ⚠ 警告：测线 {nearest_line_idx} 不在有效测线列表中！")
                    self.log_result(f"[调试]   有效测线索引: {self.valid_survey_line_indices}")
                else:
                    combo_index = self.valid_survey_line_indices.index(nearest_line_idx) + 1  # +1 因为第一个是"未选择"
                    self.log_result(f"[调试] 测线 {nearest_line_idx} 在下拉菜单中的索引: {combo_index}")
            else:
                # 如果没有有效索引列表，使用简单映射（假设所有测线都有效）
                combo_index = nearest_line_idx + 1
                self.log_result(f"[调试] 使用简单映射，下拉菜单索引: {combo_index}")
            
            # 先设置变量，再更新下拉菜单（避免事件触发问题）
            self.selected_first_survey_line_idx = nearest_line_idx
            
            # 临时断开事件绑定，避免触发事件
            self.first_survey_line_combo.unbind('<<ComboboxSelected>>')
            
            # 找到测线在下拉菜单中的正确索引
            if hasattr(self, 'valid_survey_line_indices') and nearest_line_idx in self.valid_survey_line_indices:
                combo_index = self.valid_survey_line_indices.index(nearest_line_idx) + 1
            else:
                combo_index = nearest_line_idx + 1
            
            self.first_survey_line_combo.current(combo_index)
            self.first_survey_line_combo.bind('<<ComboboxSelected>>', self.on_first_survey_line_selected)
            
            self.log_result(f"[调试] 下拉菜单已更新为索引 {combo_index} (对应测线 {nearest_line_idx})")
            self.log_result(f"[调试] 当前 selected_first_survey_line_idx = {self.selected_first_survey_line_idx}")
            
            # 断开点击事件
            if self.click_cid is not None:
                self.canvas.mpl_disconnect(self.click_cid)
                self.click_cid = None
            self.click_selection_mode = None
            
            # 验证设置是否成功
            if self.selected_first_survey_line_idx != nearest_line_idx:
                self.log_result(f"[调试] ✗ 警告！selected_first_survey_line_idx 被修改了！")
                self.log_result(f"[调试]   期望值: {nearest_line_idx}, 实际值: {self.selected_first_survey_line_idx}")
                # 强制设置正确的值
                self.selected_first_survey_line_idx = nearest_line_idx
            else:
                self.log_result(f"[调试] ✓ 确认 selected_first_survey_line_idx = {self.selected_first_survey_line_idx}")
            
            # 调用选择事件处理函数（会自动触发第二阶段和第三阶段优化）
            # 注意：on_first_survey_line_selected 会从下拉菜单读取值，所以应该是一致的
            self.log_result(f"[调试] 调用 on_first_survey_line_selected()")
            self.on_first_survey_line_selected()
            
            # 再次验证（防止事件处理函数修改了值）
            if self.selected_first_survey_line_idx != nearest_line_idx:
                self.log_result(f"[调试] ✗ 错误！on_first_survey_line_selected 修改了值！")
                self.log_result(f"[调试]   期望值: {nearest_line_idx}, 实际值: {self.selected_first_survey_line_idx}")
                self.selected_first_survey_line_idx = nearest_line_idx
                messagebox.showwarning('警告', f'测线选择可能有问题！\n期望: 测线 {nearest_line_idx}\n实际: 测线 {self.selected_first_survey_line_idx}')
            
            # 如果还没有完成第一阶段优化，只显示选择完成消息
            if self.deployment_path is None or len(self.deployment_path) == 0:
                messagebox.showinfo('完成', f'已选择测线 {nearest_line_idx} 作为第一条测线\n\n请先完成第一阶段优化（投放路径）')
            else:
                self.log_result(f"[调试] 已完成第一阶段优化，将自动触发第二阶段和第三阶段优化")
        else:
            self.log_result(f"[调试] 未找到最近测线")
    
    def update_station_type_selection(self):
        """更新站位类型选择界面"""
        # 清除现有的选择框
        for widget in self.deployment_types_frame.winfo_children():
            widget.destroy()
        for widget in self.recovery_types_frame.winfo_children():
            widget.destroy()
        
        self.deployment_type_vars.clear()
        self.recovery_type_vars.clear()
        
        if not self.stations or not self.station_types:
            return
        
        # 获取所有唯一的站位类型
        unique_types = sorted(set([t for t in self.station_types if t]))
        
        if not unique_types:
            # 如果没有类型信息，显示提示
            ttk.Label(self.deployment_types_frame, text='无站位类型信息', 
                     foreground='gray').pack(side='left', padx=5, pady=2)
            ttk.Label(self.recovery_types_frame, text='无站位类型信息', 
                     foreground='gray').pack(side='left', padx=5, pady=2)
            return
        
        # 为每个类型创建复选框（横向排列）
        for i, station_type in enumerate(unique_types):
            # 投放类型选择
            var_deploy = tk.BooleanVar(value=True)  # 默认全选
            self.deployment_type_vars[station_type] = var_deploy
            check_deploy = ttk.Checkbutton(self.deployment_types_frame, 
                                         text=station_type,
                                         variable=var_deploy)
            check_deploy.pack(side='left', padx=5, pady=2)
            
            # 回收类型选择
            var_recover = tk.BooleanVar(value=True)  # 默认全选
            self.recovery_type_vars[station_type] = var_recover
            check_recover = ttk.Checkbutton(self.recovery_types_frame, 
                                          text=station_type,
                                          variable=var_recover)
            check_recover.pack(side='left', padx=5, pady=2)
    
    def get_deployment_stations(self):
        """获取用于投放的站位索引列表"""
        if not self.stations:
            return []
        
        # 如果没有站位类型信息或没有类型选择界面，返回所有站位
        if not self.station_types or not self.deployment_type_vars:
            return list(range(len(self.stations)))
        
        # 获取选中的投放类型
        selected_types = [t for t, var in self.deployment_type_vars.items() 
                         if var.get()]
        
        if not selected_types:
            # 如果没有选择任何类型，返回所有站位
            return list(range(len(self.stations)))
        
        # 返回匹配类型的站位索引
        deployment_indices = []
        for i, station_type in enumerate(self.station_types):
            if i < len(self.stations) and (not station_type or station_type in selected_types):
                deployment_indices.append(i)
        
        return deployment_indices if deployment_indices else list(range(len(self.stations)))
    
    def get_recovery_stations(self):
        """获取用于回收的站位索引列表"""
        if not self.stations:
            return []
        
        # 如果没有站位类型信息或没有类型选择界面，返回所有站位
        if not self.station_types or not self.recovery_type_vars:
            return list(range(len(self.stations)))
        
        # 获取选中的回收类型
        selected_types = [t for t, var in self.recovery_type_vars.items() 
                         if var.get()]
        
        if not selected_types:
            # 如果没有选择任何类型，返回所有站位
            return list(range(len(self.stations)))
        
        # 返回匹配类型的站位索引
        recovery_indices = []
        for i, station_type in enumerate(self.station_types):
            if i < len(self.stations) and (not station_type or station_type in selected_types):
                recovery_indices.append(i)
        
        return recovery_indices if recovery_indices else list(range(len(self.stations)))
    
    def on_start_selected(self, event=None):
        """起点选择事件"""
        selection = self.start_combo.current()
        if selection == 0:  # "未选择"
            self.selected_start_idx = None
            self.start_idx = None
        else:
            self.selected_start_idx = selection - 1
            self.start_idx = selection - 1
        
        # 如果起点和终点相同，清除终点
        if self.selected_start_idx is not None and self.selected_start_idx == self.selected_end_idx:
            self.end_combo.current(0)
            self.selected_end_idx = None
            self.end_idx = None
        
        self.plot_stations()
    
    def on_end_selected(self, event=None):
        """终点选择事件"""
        selection = self.end_combo.current()
        if selection == 0:  # "未选择"
            self.selected_end_idx = None
            self.end_idx = None
        else:
            self.selected_end_idx = selection - 1
            self.end_idx = selection - 1
        
        # 如果起点和终点相同，清除起点
        if self.selected_start_idx is not None and self.selected_start_idx == self.selected_end_idx:
            self.start_combo.current(0)
            self.selected_start_idx = None
            self.start_idx = None
        
        self.plot_stations()
    
    def enable_click_selection(self):
        """启用点击选择模式"""
        if not self.stations:
            messagebox.showwarning('警告', '请先加载站位文件')
            return
        
        # 切换选择模式
        if self.click_selection_mode is None:
            self.click_selection_mode = 'start'
            messagebox.showinfo('提示', '点击选择模式已启用\n\n请先点击一个站位作为起点，然后点击另一个站位作为终点')
            self.log_result("点击选择模式已启用")
        else:
            self.disable_click_selection()
            return
        
        # 连接鼠标点击事件
        if self.click_cid is None:
            self.click_cid = self.canvas.mpl_connect('button_press_event', self.on_station_click)
    
    def disable_click_selection(self):
        """禁用点击选择模式"""
        if self.click_cid is not None:
            self.canvas.mpl_disconnect(self.click_cid)
            self.click_cid = None
        self.click_selection_mode = None
        self.log_result("点击选择模式已禁用")
    
    def on_station_click(self, event):
        """处理站位点击事件"""
        if event.inaxes != self.ax:
            return
        
        if event.button != 1:  # 只处理左键
            return
        
        # 如果当前是测线选择模式，不处理站位点击
        if self.click_selection_mode == 'first_survey_line':
            return
        
        # 找到最近的站位
        click_lon, click_lat = event.xdata, event.ydata
        if click_lon is None or click_lat is None:
            return
        
        min_dist = float('inf')
        nearest_idx = None
        
        for i, (lon, lat) in enumerate(self.stations):
            dist = np.sqrt((lon - click_lon)**2 + (lat - click_lat)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        # 检查是否点击在站位附近（阈值可调整）
        if min_dist > 0.5:  # 如果距离太远，忽略
            return
        
        # 根据当前模式设置起点或终点
        if self.click_selection_mode == 'start':
            self.selected_start_idx = nearest_idx
            self.start_idx = nearest_idx
            self.start_combo.current(nearest_idx + 1)
            self.click_selection_mode = 'end'
            self.log_result(f"已选择起点: 站位 {nearest_idx}")
        elif self.click_selection_mode == 'end':
            if nearest_idx == self.selected_start_idx:
                messagebox.showwarning('警告', '终点不能与起点相同')
                return
            self.selected_end_idx = nearest_idx
            self.end_idx = nearest_idx
            self.end_combo.current(nearest_idx + 1)
            self.disable_click_selection()
            self.log_result(f"已选择终点: 站位 {nearest_idx}")
            messagebox.showinfo('完成', '起点和终点已选择完成\n\n可以点击"开始搜索路径"按钮进行优化')
        
        self.plot_stations()
    
    def plot_bathymetry(self):
        """使用pygmt风格绘制水深底图（根据站位和测线坐标范围）"""
        # 清除之前的水深图
        if self.bathymetry_artist is not None:
            try:
                self.bathymetry_artist.remove()
            except:
                pass
            self.bathymetry_artist = None
        
        # 检查是否有水深数据
        if self.bathymetry_filename is None and (self.bathymetry_data is None or self.bathymetry_lon is None or self.bathymetry_lat is None):
            return
        
        try:
            # ========== 第一步：先确定绘图区域 ==========
            # 收集所有需要显示的坐标点（站位和测线）
            all_lons = []
            all_lats = []
            
            # 添加站位坐标
            if self.stations:
                for lon, lat in self.stations:
                    all_lons.append(lon)
                    all_lats.append(lat)
            
            # 添加测线坐标
            if self.survey_lines:
                for line in self.survey_lines:
                    for lon, lat in line:
                        all_lons.append(lon)
                        all_lats.append(lat)
            
            # 计算绘图区域
            if not all_lons:
                # 如果没有坐标点，使用整个水深网格范围
                if self.bathymetry_file_type == 'grd' and self.bathymetry_full_range:
                    plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max = self.bathymetry_full_range
                elif self.bathymetry_lon is not None and self.bathymetry_lat is not None:
                    plot_lon_min = self.bathymetry_lon.min()
                    plot_lon_max = self.bathymetry_lon.max()
                    plot_lat_min = self.bathymetry_lat.min()
                    plot_lat_max = self.bathymetry_lat.max()
                else:
                    return  # 无法确定范围
            else:
                # 计算坐标范围，并添加一些边界扩展
                all_lons = np.array(all_lons)
                all_lats = np.array(all_lats)
                lon_range = all_lons.max() - all_lons.min()
                lat_range = all_lats.max() - all_lats.min()
                
                # 如果范围为0（只有一个点），使用固定扩展
                if lon_range == 0:
                    lon_range = 0.1  # 约11公里
                if lat_range == 0:
                    lat_range = 0.1  # 约11公里
                
                margin = 0.1  # 10%的边界扩展
                fixed_margin = 0.05  # 固定扩展（度），约5.5公里
                
                plot_lon_min = all_lons.min() - max(lon_range * margin, fixed_margin)
                plot_lon_max = all_lons.max() + max(lon_range * margin, fixed_margin)
                plot_lat_min = all_lats.min() - max(lat_range * margin, fixed_margin)
                plot_lat_max = all_lats.max() + max(lat_range * margin, fixed_margin)
                
                # 确保范围不超过水深网格的范围
                if (self.bathymetry_file_type in ['grd', 'nc'] and self.bathymetry_full_range):
                    full_lon_min, full_lon_max, full_lat_min, full_lat_max = self.bathymetry_full_range
                    plot_lon_min = max(plot_lon_min, full_lon_min)
                    plot_lon_max = min(plot_lon_max, full_lon_max)
                    plot_lat_min = max(plot_lat_min, full_lat_min)
                    plot_lat_max = min(plot_lat_max, full_lat_max)
                elif self.bathymetry_lon is not None and self.bathymetry_lat is not None:
                    plot_lon_min = max(plot_lon_min, self.bathymetry_lon.min())
                    plot_lon_max = min(plot_lon_max, self.bathymetry_lon.max())
                    plot_lat_min = max(plot_lat_min, self.bathymetry_lat.min())
                    plot_lat_max = min(plot_lat_max, self.bathymetry_lat.max())
            
            # ========== 第二步：根据文件类型按需读取数据 ==========
            # 计算缓存哈希值（用于判断是否需要重新计算）
            cache_key = (
                hash((plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max)) +
                hash(self.bathymetry_filename if self.bathymetry_filename else '')
            )
            
            # 检查缓存是否有效
            if (self._bathymetry_cache is not None and 
                self._bathymetry_cache_hash == cache_key and
                self._bathymetry_cache_extent == (plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max)):
                # 使用缓存的数据
                plot_data = self._bathymetry_cache['data']
                lon_min, lon_max, lat_min, lat_max = self._bathymetry_cache_extent
            else:
                # 需要重新读取数据
                if self.bathymetry_file_type == 'grd' and self.bathymetry_filename:
                    # 对于.grd文件，使用pygmt.grdcut只读取需要的区域
                    try:
                        import pygmt
                        # 使用pygmt.grdcut裁剪到指定区域
                        # region格式：[lon_min, lon_max, lat_min, lat_max]
                        region = f"{plot_lon_min}/{plot_lon_max}/{plot_lat_min}/{plot_lat_max}"
                        grid = pygmt.grdcut(self.bathymetry_filename, region=region)
                        # 提取数据（grid是xarray DataArray）
                        plot_lon = grid.lon.values
                        plot_lat = grid.lat.values
                        plot_data = grid.values
                        
                        lon_min = float(plot_lon.min())
                        lon_max = float(plot_lon.max())
                        lat_min = float(plot_lat.min())
                        lat_max = float(plot_lat.max())
                    except ImportError:
                        # pygmt未安装，尝试使用xarray
                        try:
                            import xarray as xr
                            with xr.open_dataset(self.bathymetry_filename, decode_times=False) as ds:
                                # 裁剪到指定区域
                                if 'lon' in ds.coords and 'lat' in ds.coords:
                                    lon_slice = slice(plot_lon_min, plot_lon_max)
                                    lat_slice = slice(plot_lat_min, plot_lat_max)
                                    ds_cropped = ds.sel(lon=lon_slice, lat=lat_slice, method='nearest')
                                    plot_lon = ds_cropped.lon.values
                                    plot_lat = ds_cropped.lat.values
                                    # 获取第一个数据变量
                                    data_vars = list(ds_cropped.data_vars.keys())
                                    if data_vars:
                                        plot_data = ds_cropped[data_vars[0]].values
                                    else:
                                        return
                                    lon_min = plot_lon.min()
                                    lon_max = plot_lon.max()
                                    lat_min = plot_lat.min()
                                    lat_max = plot_lat.max()
                        except Exception as e:
                            print(f"读取.grd文件区域失败: {str(e)}")
                            return
                    except Exception as e:
                        print(f"使用pygmt读取.grd文件区域失败: {str(e)}")
                        return
                elif self.bathymetry_file_type == 'nc' and self.bathymetry_filename:
                    # 对于.nc文件，使用xarray按需读取指定区域
                    try:
                        import xarray as xr
                        with xr.open_dataset(self.bathymetry_filename, decode_times=False) as ds:
                            # 使用保存的坐标名和数据变量名
                            lon_name = self.bathymetry_lon_name
                            lat_name = self.bathymetry_lat_name
                            var_name = self.bathymetry_var_name
                            
                            # 找到范围内的坐标值（不加载数据，只读取坐标）
                            lon_coords = ds[lon_name].values
                            lat_coords = ds[lat_name].values
                            
                            # 找到范围内的索引
                            lon_mask = (lon_coords >= plot_lon_min) & (lon_coords <= plot_lon_max)
                            lat_mask = (lat_coords >= plot_lat_min) & (lat_coords <= plot_lat_max)
                            
                            lon_indices = np.where(lon_mask)[0]
                            lat_indices = np.where(lat_mask)[0]
                            
                            if len(lon_indices) == 0 or len(lat_indices) == 0:
                                return
                            
                            # 使用isel通过索引选择（支持按需读取）
                            ds_cropped = ds.isel({lon_name: lon_indices, lat_name: lat_indices})
                            
                            # 提取坐标和数据（此时才真正读取数据）
                            plot_lon = ds_cropped[lon_name].values
                            plot_lat = ds_cropped[lat_name].values
                            plot_data = ds_cropped[var_name].values
                            
                            lon_min = float(plot_lon.min())
                            lon_max = float(plot_lon.max())
                            lat_min = float(plot_lat.min())
                            lat_max = float(plot_lat.max())
                    except Exception as e:
                        print(f"读取.nc文件区域失败: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        return
                else:
                    # 对于其他格式（如.txt），从已加载的数据中裁剪
                    if self.bathymetry_data is None or self.bathymetry_lon is None or self.bathymetry_lat is None:
                        return
                    
                    # 从水深网格中提取需要显示的范围
                    lon_indices = np.where((self.bathymetry_lon >= plot_lon_min) & 
                                          (self.bathymetry_lon <= plot_lon_max))[0]
                    lat_indices = np.where((self.bathymetry_lat >= plot_lat_min) & 
                                          (self.bathymetry_lat <= plot_lat_max))[0]
                    
                    if len(lon_indices) == 0 or len(lat_indices) == 0:
                        return
                    
                    # 提取范围内的数据
                    lon_start, lon_end = lon_indices[0], lon_indices[-1] + 1
                    lat_start, lat_end = lat_indices[0], lat_indices[-1] + 1
                    
                    plot_data = self.bathymetry_data[lat_start:lat_end, lon_start:lon_end]
                    plot_lon = self.bathymetry_lon[lon_start:lon_end]
                    plot_lat = self.bathymetry_lat[lat_start:lat_end]
                    
                    lon_min = plot_lon.min()
                    lon_max = plot_lon.max()
                    lat_min = plot_lat.min()
                    lat_max = plot_lat.max()
                
                # 检查数据大小，如果太大则降采样以提高性能
                data_shape = plot_data.shape
                max_size = 200  # 最大显示尺寸
                
                if data_shape[0] > max_size or data_shape[1] > max_size:
                    # 需要降采样
                    step_y = max(1, data_shape[0] // max_size)
                    step_x = max(1, data_shape[1] // max_size)
                    plot_data = plot_data[::step_y, ::step_x].copy()
                    plot_lon = plot_lon[::step_x]
                    plot_lat = plot_lat[::step_y]
                    
                    lon_min = plot_lon.min()
                    lon_max = plot_lon.max()
                    lat_min = plot_lat.min()
                    lat_max = plot_lat.max()
                
                # 转换为float32以提高性能
                if plot_data.dtype != np.float32:
                    plot_data = plot_data.astype(np.float32, copy=False)
                
                # 更新缓存
                self._bathymetry_cache = {'data': plot_data}
                self._bathymetry_cache_extent = (lon_min, lon_max, lat_min, lat_max)
                self._bathymetry_cache_hash = cache_key
            
            # 使用pygmt风格的色标（适合水深图，深色表示深水，浅色表示浅水）
            # 缓存colormap以避免重复创建
            if not hasattr(self, '_bathymetry_cmap'):
                try:
                    import pygmt
                    from matplotlib.colors import LinearSegmentedColormap
                    # 使用pygmt风格的水深色标（类似batlow或ocean）
                    # 深蓝色到浅蓝色的渐变，适合水深图
                    # 使用类似pygmt ocean的配色：深蓝->浅蓝->浅绿->黄色
                    colors = ['#000080', '#0000FF', '#0080FF', '#00FFFF', '#80FF80', '#FFFF00']
                    n_bins = 256
                    self._bathymetry_cmap = LinearSegmentedColormap.from_list('bathymetry', colors, N=n_bins)
                except ImportError:
                    # pygmt不可用，使用matplotlib内置的ocean colormap或viridis
                    try:
                        # 尝试使用ocean colormap（如果可用）
                        self._bathymetry_cmap = plt.cm.ocean
                    except:
                        # 使用反转的viridis（深色表示深水）
                        self._bathymetry_cmap = plt.cm.viridis_r
            
            # 使用imshow绘制水深图，优化性能参数
            # 使用较低的alpha值，确保不影响其他元素
            # 注意：不在这里调用canvas.draw()，避免阻塞，等所有绘制完成后再draw
            im = self.ax.imshow(plot_data, 
                               extent=[lon_min, lon_max, lat_min, lat_max],
                               cmap=self._bathymetry_cmap, alpha=0.4, zorder=0,
                               origin='lower', aspect='auto',
                               interpolation='nearest',  # 使用nearest而不是bilinear以提高性能
                               rasterized=True)  # 栅格化以提高性能
            
            self.bathymetry_artist = im
            
        except Exception as e:
            # 如果绘制失败，记录错误但不中断程序
            print(f"绘制水深图失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def plot_stations(self):
        """绘制站位"""
        self.ax.clear()
        # 清除之前的站位图形对象
        self.station_artists.clear()
        # 清除之前的水深图
        if self.bathymetry_artist is not None:
            try:
                self.bathymetry_artist.remove()
            except:
                pass
            self.bathymetry_artist = None
        # 清除之前的测线和转弯路径（但保留数据）
        for artist in self.survey_line_artists:
            try:
                artist.remove()
            except:
                pass
        self.survey_line_artists.clear()
        for artist in self.turn_path_artists:
            try:
                artist.remove()
            except:
                pass
        self.turn_path_artists.clear()
        
        if not self.stations:
            self.ax.text(0.5, 0.5, 'Please load station file', 
                        ha='center', va='center', transform=self.ax.transAxes,
                        fontsize=14)
            self.canvas.draw()
            return
        
        # 绘制水深底图（如果存在）
        self.plot_bathymetry()
        
        # 绘制所有站位
        lons = [s[0] for s in self.stations]
        lats = [s[1] for s in self.stations]
        
        # 根据台站类型分组绘制
        if self.station_types and any(self.station_types):
            # 按类型分组绘制
            type_groups = {}
            for i, (lon, lat) in enumerate(self.stations):
                station_type = self.station_types[i] if i < len(self.station_types) else ''
                if station_type not in type_groups:
                    type_groups[station_type] = {'lons': [], 'lats': [], 'indices': []}
                type_groups[station_type]['lons'].append(lon)
                type_groups[station_type]['lats'].append(lat)
                type_groups[station_type]['indices'].append(i)
            
            # 为每种类型绘制不同颜色的点
            for station_type, coords in type_groups.items():
                color = self.get_station_type_color(station_type)
                label = f'Type {station_type}' if station_type else 'Unclassified'
                scatter = self.ax.scatter(coords['lons'], coords['lats'], 
                                        c=color, s=100, marker='o', 
                                        label=label, zorder=3, picker=5)
                self.station_artists.append(scatter)
        else:
            # 如果没有类型信息，使用默认蓝色
            scatter = self.ax.scatter(lons, lats, c='blue', s=100, marker='o', 
                                     label='Station', zorder=3, picker=5)
            self.station_artists.append(scatter)
        
        # 标注站位编号（仅在未优化时显示原始序号）
        # 如果已有优化路径，则不显示原始序号，只显示路径顺序（在plot_path中显示）
        if self.optimizer is None or self.optimizer.optimal_path is None:
            for i, (lon, lat) in enumerate(self.stations):
                # 填充色是黄色，文字用黑色
                facecolor = 'yellow'
                text_color = 'white' if facecolor.lower() in ['black', 'k', '#000000'] else 'black'
                self.ax.annotate(f'{i}', (lon, lat), xytext=(3, 3),
                               textcoords='offset points', fontsize=8, fontweight='bold',
                               color=text_color,
                               bbox=dict(boxstyle='round,pad=0.15', facecolor=facecolor, 
                                       alpha=0.8, edgecolor='black', linewidth=0.5))
        
        # 标记起点
        if self.selected_start_idx is not None:
            start_lon, start_lat = self.stations[self.selected_start_idx]
            start_scatter = self.ax.scatter(start_lon, start_lat, c='green', s=200, 
                                          marker='s', label='Start', zorder=4,
                                          edgecolors='black', linewidths=2)
            self.start_artist = start_scatter
        
        # 标记终点
        if self.selected_end_idx is not None:
            end_lon, end_lat = self.stations[self.selected_end_idx]
            end_scatter = self.ax.scatter(end_lon, end_lat, c='red', s=200,
                                        marker='s', label='End', zorder=4,
                                        edgecolors='black', linewidths=2)
            self.end_artist = end_scatter
        
        # 绘制作业测线（如果存在）
        self.plot_survey_lines()
        
        # 绘制路径（如果存在）- 在plot_path方法中单独处理
        
        # 设置坐标轴
        self.ax.set_xlabel('Longitude (°)', fontsize=12)
        self.ax.set_ylabel('Latitude (°)', fontsize=12)
        
        # 设置标题（如果有优化路径，显示总距离）
        if self.optimizer is not None and self.optimizer.optimal_path is not None and self.optimizer.total_distance is not None:
            # 如果有三阶段路径，使用更详细的总距离（包含测线长度）
            if (self.deployment_path is not None and 
                self.total_survey_length is not None and 
                self.recovery_path is not None):
                total_distance = (self.deployment_distance + 
                                self.total_survey_length +
                                (self.deployment_to_survey_distance or 0) +
                                (self.survey_line_transition_distance or 0) +
                                (self.survey_to_recovery_distance or 0) +
                                self.recovery_distance)
                title = f'Station Distribution\nTotal Stations: {len(self.stations)} | Total Distance: {total_distance:.2f} km'
            else:
                title = f'Station Distribution\nTotal Stations: {len(self.stations)} | Total Distance: {self.optimizer.total_distance:.2f} km'
        else:
            title = f'Station Distribution\nTotal Stations: {len(self.stations)}'
        self.ax.set_title(title, fontsize=14)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='best')
        
        # 调整坐标轴比例
        lon_range = max(lons) - min(lons)
        lat_range = max(lats) - min(lats)
        if lon_range > 0 and lat_range > 0:
            avg_lat = np.mean(lats)
            aspect_ratio = lon_range / (lat_range * np.cos(np.radians(avg_lat)))
            self.ax.set_aspect(aspect_ratio, adjustable='box')
        
        # 绘制路径（如果存在）- 在设置完坐标轴后绘制
        if self.optimizer is not None and self.optimizer.optimal_path is not None:
            self.plot_path()
        
        # 如果有转弯路径数据，根据用户选择绘制转弯路径
        if self.turn_paths:
            self.plot_turn_paths()
        
        self.canvas.draw()
    
    def plot_path(self):
        """绘制优化后的路径（支持三阶段路径）"""
        # 检查是否有三阶段路径
        if (self.deployment_path is None and 
            (self.optimizer is None or self.optimizer.optimal_path is None)):
            return
        
        # 清除之前的路径
        if self.path_artist is not None:
            try:
                self.path_artist.remove()
            except:
                pass
            self.path_artist = None
        
        if self.deployment_path_artist is not None:
            try:
                self.deployment_path_artist.remove()
            except:
                pass
            self.deployment_path_artist = None
        
        if self.recovery_path_artist is not None:
            try:
                self.recovery_path_artist.remove()
            except:
                pass
            self.recovery_path_artist = None
        
        # 清除之前的航渡路径（避免重复绘制）
        for artist in self.transition_path_artists:
            try:
                artist.remove()
            except:
                pass
        self.transition_path_artists.clear()
        
        # 清除之前的路径标注
        for label in self.path_labels:
            try:
                label.remove()
            except:
                pass
        self.path_labels.clear()
        
        for label in self.deployment_labels:
            try:
                label.remove()
            except:
                pass
        self.deployment_labels.clear()
        
        for label in self.recovery_labels:
            try:
                label.remove()
            except:
                pass
        self.recovery_labels.clear()
        
        # 如果有三阶段路径，分别绘制
        if self.deployment_path is not None:
            # 绘制投放路径（蓝色）- 根据用户选择
            if self.show_deployment_path_var.get():
                deployment_coords = np.array([self.stations[i] for i in self.deployment_path])
                self.deployment_path_artist, = self.ax.plot(deployment_coords[:, 0], deployment_coords[:, 1], 
                                            'b-', linewidth=2, alpha=0.7, label='Deployment Path', zorder=1)
            
            # 在投放路径上标注顺序 - 根据用户选择
            if self.show_deployment_labels_var.get():
                for i, idx in enumerate(self.deployment_path):
                    lon, lat = self.stations[idx]
                    # 填充色是lightblue，文字用黑色
                    facecolor = 'lightblue'
                    text_color = 'white' if facecolor.lower() in ['black', 'k', '#000000'] else 'black'
                    text = self.ax.annotate(f'{i+1}', (lon, lat), xytext=(0, -15),
                                   textcoords='offset points', fontsize=6, color=text_color,
                                   fontweight='bold', ha='center',
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor=facecolor, 
                                           alpha=0.8, edgecolor='blue', linewidth=1))
                    self.deployment_labels.append(text)
            
            # 绘制回收路径（红色）- 根据用户选择
            if self.recovery_path is not None and self.show_recovery_path_var.get():
                recovery_coords = np.array([self.stations[i] for i in self.recovery_path])
                self.recovery_path_artist, = self.ax.plot(recovery_coords[:, 0], recovery_coords[:, 1], 
                                               'r-', linewidth=2, alpha=0.7, label='Recovery Path', zorder=1)
                
                # 在回收路径上标注顺序 - 根据用户选择
                if self.show_recovery_labels_var.get():
                    for i, idx in enumerate(self.recovery_path):
                        lon, lat = self.stations[idx]
                        # 填充色是lightcoral，文字用黑色
                        facecolor = 'lightcoral'
                        text_color = 'white' if facecolor.lower() in ['black', 'k', '#000000'] else 'black'
                        text = self.ax.annotate(f'{i+1}', (lon, lat), xytext=(0, 15),
                                       textcoords='offset points', fontsize=6, color=text_color,
                                       fontweight='bold', ha='center',
                                       bbox=dict(boxstyle='round,pad=0.2', facecolor=facecolor, 
                                               alpha=0.8, edgecolor='red', linewidth=1))
                        self.recovery_labels.append(text)
            
            # 绘制从投放到测线的连接线（绿色）- 根据用户选择
            if (self.show_transition_paths_var.get() and self.deployment_path and 
                self.survey_line_order and self.deployment_to_survey_distance is not None):
                last_deployment_idx = self.deployment_path[-1]
                last_deployment_lon, last_deployment_lat = self.stations[last_deployment_idx]
                first_line_idx = self.survey_line_order[0]
                
                # 使用保存的进入点（而不是固定的line[0]）
                if hasattr(self, 'survey_line_entry_points') and first_line_idx in self.survey_line_entry_points:
                    entry_point = self.survey_line_entry_points[first_line_idx]
                    self.log_result(f"[绘制] 使用保存的进入点: 测线 {first_line_idx} -> ({entry_point[0]:.6f}°, {entry_point[1]:.6f}°)")
                else:
                    # 如果没有保存的点，使用默认值（向后兼容）
                    entry_point = self.survey_lines[first_line_idx][0]
                    self.log_result(f"[绘制] 警告：未找到保存的进入点，使用默认值: ({entry_point[0]:.6f}°, {entry_point[1]:.6f}°)")
                
                artist, = self.ax.plot([last_deployment_lon, entry_point[0]], 
                            [last_deployment_lat, entry_point[1]], 
                            'g--', linewidth=2, alpha=0.7, label='To Survey', zorder=1)
                self.transition_path_artists.append(artist)
            
            # 绘制测线之间的航渡路径（绿色虚线）- 根据用户选择
            # 注意：只有当有多条测线时才绘制测线之间的航渡路径
            if (self.show_transition_paths_var.get() and self.survey_line_order and 
                len(self.survey_line_order) > 1):
                for i in range(len(self.survey_line_order) - 1):
                    line1_idx = self.survey_line_order[i]
                    line2_idx = self.survey_line_order[i + 1]
                    
                    # 使用保存的退出点和进入点
                    if (hasattr(self, 'survey_line_exit_points') and line1_idx in self.survey_line_exit_points and
                        hasattr(self, 'survey_line_entry_points') and line2_idx in self.survey_line_entry_points):
                        exit_point = self.survey_line_exit_points[line1_idx]
                        entry_point = self.survey_line_entry_points[line2_idx]
                        self.log_result(f"[绘制] 测线 {line1_idx} -> {line2_idx}: 退出点 ({exit_point[0]:.6f}°, {exit_point[1]:.6f}°) -> 进入点 ({entry_point[0]:.6f}°, {entry_point[1]:.6f}°)")
                    else:
                        # 如果没有保存的点，使用默认值（向后兼容）
                        line1 = self.survey_lines[line1_idx]
                        line2 = self.survey_lines[line2_idx]
                        if len(line1) >= 2 and len(line2) >= 2:
                            exit_point = line1[-1]
                            entry_point = line2[0]
                            self.log_result(f"[绘制] 警告：未找到保存的点，使用默认值")
                        else:
                            continue
                    
                    artist, = self.ax.plot([exit_point[0], entry_point[0]], 
                                [exit_point[1], entry_point[1]], 
                                'g--', linewidth=2, alpha=0.7, zorder=1)
                    self.transition_path_artists.append(artist)
            
            # 绘制从测线到回收的连接线（橙色虚线）- 根据用户选择
            if (self.show_transition_paths_var.get() and self.survey_line_order and 
                self.recovery_path and self.survey_to_recovery_distance is not None):
                last_survey_line_idx = self.survey_line_order[-1]
                
                # 使用保存的退出点（而不是固定的line[-1]）
                if hasattr(self, 'survey_line_exit_points') and last_survey_line_idx in self.survey_line_exit_points:
                    exit_point = self.survey_line_exit_points[last_survey_line_idx]
                    self.log_result(f"[绘制] 使用保存的退出点: 测线 {last_survey_line_idx} -> ({exit_point[0]:.6f}°, {exit_point[1]:.6f}°)")
                else:
                    # 如果没有保存的点，使用默认值（向后兼容）
                    exit_point = self.survey_lines[last_survey_line_idx][-1]
                    self.log_result(f"[绘制] 警告：未找到保存的退出点，使用默认值: ({exit_point[0]:.6f}°, {exit_point[1]:.6f}°)")
                
                first_recovery_idx = self.recovery_path[0]
                first_recovery_lon, first_recovery_lat = self.stations[first_recovery_idx]
                artist, = self.ax.plot([exit_point[0], first_recovery_lon], 
                            [exit_point[1], first_recovery_lat], 
                            'orange', linestyle='--', linewidth=2, alpha=0.7, label='To Recovery', zorder=1)
                self.transition_path_artists.append(artist)
            
            # 在图表上显示总距离信息
            if (self.deployment_distance is not None and 
                self.recovery_distance is not None and
                self.total_survey_length is not None):
                # 计算走航路径总距离
                cruise_distance = (self.deployment_to_survey_distance +
                                 self.survey_line_transition_distance +
                                 self.survey_to_recovery_distance)
                total_distance = (self.deployment_distance + 
                                self.total_survey_length +  # 添加测线长度
                                cruise_distance +
                                self.recovery_distance)
                distance_text = self.ax.text(0.02, 0.98, 
                    f'Total Distance: {total_distance:.2f} km\n'
                    f'= Deployment {self.deployment_distance:.2f} km\n'
                    f'+ Survey Lines {self.total_survey_length:.2f} km\n'
                    f'+ Recovery {self.recovery_distance:.2f} km\n'
                    f'+ Transit {cruise_distance:.2f} km',
                    transform=self.ax.transAxes,
                    fontsize=10, fontweight='bold', color='black',
                    ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            alpha=0.9, edgecolor='black', linewidth=2))
            return
        
        # 兼容旧代码：如果没有三阶段路径，使用原来的方式
        path_coords = np.array([self.stations[i] for i in self.optimizer.optimal_path])
        path_line, = self.ax.plot(path_coords[:, 0], path_coords[:, 1],
                                 'r-', linewidth=2, alpha=0.6, label='Path', zorder=1)
        self.path_artist = path_line
        
        # 在路径上标注顺序
        for i, idx in enumerate(self.optimizer.optimal_path):
            lon, lat = self.stations[idx]
            # 填充色是white，文字用黑色
            facecolor = 'white'
            text_color = 'white' if facecolor.lower() in ['black', 'k', '#000000'] else 'black'
            text = self.ax.annotate(f'{i+1}', (lon, lat), xytext=(0, -15),
                           textcoords='offset points', fontsize=9, color=text_color,
                           fontweight='bold', ha='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=facecolor, 
                                   alpha=0.8, edgecolor='red', linewidth=1.5))
            self.path_labels.append(text)
        
        # 在图表上显示总距离信息
        if self.optimizer.total_distance is not None:
            # 在左上角显示总距离
            distance_text = self.ax.text(0.02, 0.98, f'Total Distance: {self.optimizer.total_distance:.2f} km',
                                         transform=self.ax.transAxes,
                                         fontsize=12, fontweight='bold', color='red',
                                         ha='left', va='top',
                                         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                                 alpha=0.9, edgecolor='red', linewidth=2))
            self.path_labels.append(distance_text)
        
        self.canvas.draw()
    
    def plot_survey_lines(self):
        """绘制作业测线（根据优化顺序显示编号）"""
        # 清除之前的测线和标注
        for artist in self.survey_line_artists:
            try:
                artist.remove()
            except:
                pass
        self.survey_line_artists.clear()
        
        for label in self.survey_line_labels:
            try:
                label.remove()
            except:
                pass
        self.survey_line_labels.clear()
        
        for artist in self.survey_line_start_artists:
            try:
                artist.remove()
            except:
                pass
        self.survey_line_start_artists.clear()
        
        if not self.survey_lines:
            return
        
        # 根据用户选择决定是否显示测线
        if not self.show_survey_lines_var.get():
            return
        
        # 如果有优化后的测线顺序，按顺序绘制并编号
        if self.survey_line_order is not None:
            for order_idx, line_idx in enumerate(self.survey_line_order):
                line = self.survey_lines[line_idx]
                if len(line) < 2:
                    continue
                
                lons = [p[0] for p in line]
                lats = [p[1] for p in line]
                
                # 第一条测线（起始测线）使用更粗的红色虚线，其他使用蓝色虚线
                if order_idx == 0:
                    # 起始测线：红色粗虚线
                    line_artist, = self.ax.plot(lons, lats, 'r--', linewidth=2.5, 
                                                alpha=0.9, label='Start Line' if order_idx == 0 else '', zorder=2)
                else:
                    # 其他测线：蓝色虚线
                    line_artist, = self.ax.plot(lons, lats, 'b--', linewidth=1.5, 
                                                alpha=0.7, label='Survey Line' if order_idx == 1 else '', zorder=2)
                self.survey_line_artists.append(line_artist)
                
                # 只在第一个进线点标记红色
                if order_idx == 0:
                    start_lon, start_lat = line[0]
                    start_marker, = self.ax.plot(start_lon, start_lat, 'ro', 
                                                markersize=10, markeredgecolor='darkred',
                                                markeredgewidth=2, label='Start Point',
                                                zorder=3)
                    self.survey_line_start_artists.append(start_marker)
                
                # 在测线中点添加编号
                mid_idx = len(line) // 2
                mid_lon, mid_lat = line[mid_idx]
                
                # 第一条测线（起始测线）使用特殊标注
                if order_idx == 0:
                    # 起始测线：红色背景，白色文字，更大字体
                    facecolor = 'red'
                    text_color = 'white'
                    label_text = f'Start L{order_idx+1}'
                    fontsize = 11
                    edgecolor = 'darkred'
                    linewidth = 2
                else:
                    # 其他测线：蓝色背景，白色文字
                    facecolor = 'lightblue'
                    text_color = 'white'
                    label_text = f'L{order_idx+1}'
                    fontsize = 9
                    edgecolor = 'blue'
                    linewidth = 1.5
                
                text = self.ax.annotate(label_text, (mid_lon, mid_lat), 
                                       xytext=(0, 0), textcoords='offset points',
                                       fontsize=fontsize, color=text_color, fontweight='bold',
                                       ha='center', va='center',
                                       bbox=dict(boxstyle='round,pad=0.4', facecolor=facecolor, 
                                               alpha=0.9, edgecolor=edgecolor, linewidth=linewidth))
                self.survey_line_labels.append(text)
        else:
            # 如果没有优化顺序，按原始顺序绘制并编号
            # 如果用户选择了第一条测线，高亮显示
            for i, line in enumerate(self.survey_lines):
                if len(line) < 2:
                    continue
                
                lons = [p[0] for p in line]
                lats = [p[1] for p in line]
                
                # 如果这是用户选择的第一条测线，使用更粗的线条和不同颜色
                if i == self.selected_first_survey_line_idx:
                    line_artist, = self.ax.plot(lons, lats, 'r--', linewidth=2.5, 
                                                alpha=0.9, label='Selected First Line' if i == 0 else '', zorder=2)
                else:
                    # 绘制测线（使用蓝色虚线）
                    line_artist, = self.ax.plot(lons, lats, 'b--', linewidth=1.5, 
                                                alpha=0.7, label='Survey Line' if i == 0 else '', zorder=2)
                self.survey_line_artists.append(line_artist)
                
                # 在测线中点添加编号
                mid_idx = len(line) // 2
                mid_lon, mid_lat = line[mid_idx]
                
                # 如果这是用户选择的第一条测线，使用特殊标注
                if i == self.selected_first_survey_line_idx:
                    facecolor = 'red'
                    text_color = 'white'
                    label_text = f'L{i+1} (Selected)'
                    fontsize = 10
                    edgecolor = 'darkred'
                    linewidth = 2
                else:
                    facecolor = 'lightblue'
                    text_color = 'white'
                    label_text = f'L{i+1}'
                    fontsize = 9
                    edgecolor = 'blue'
                    linewidth = 1.5
                
                text = self.ax.annotate(label_text, (mid_lon, mid_lat), 
                                       xytext=(0, 0), textcoords='offset points',
                                       fontsize=fontsize, color=text_color, fontweight='bold',
                                       ha='center', va='center',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor=facecolor, 
                                               alpha=0.8, edgecolor=edgecolor, linewidth=linewidth))
                self.survey_line_labels.append(text)
    
    def find_nearest_survey_line_point(self, point_lon, point_lat, use_start=True):
        """
        找到距离给定点最近的测线起点或终点
        
        Parameters
        ----------
        point_lon : float
            点的经度
        point_lat : float
            点的纬度
        use_start : bool
            True表示找测线起点，False表示找测线终点
            
        Returns
        -------
        nearest_line_idx : int
            最近测线的索引
        nearest_point : Tuple[float, float]
            最近点的坐标 (lon, lat)
        distance : float
            距离（公里）
        """
        if not self.survey_lines:
            return None, None, float('inf')
        
        min_distance = float('inf')
        nearest_line_idx = 0
        nearest_point = None
        
        for i, line in enumerate(self.survey_lines):
            if use_start:
                target_point = line[0]  # 测线起点
            else:
                target_point = line[-1]  # 测线终点
            
            distance = StationPathOptimizer.haversine_distance(
                point_lon, point_lat, target_point[0], target_point[1]
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest_line_idx = i
                nearest_point = target_point
        
        return nearest_line_idx, nearest_point, min_distance
    
    def optimize_survey_line_order_from_selected(self, first_line_idx, start_point_lon, start_point_lat, use_start_point):
        """
        优化测线顺序，从用户选择的测线开始，找到最优的测线访问顺序
        
        Parameters
        ----------
        first_line_idx : int
            用户选择的第一条测线索引
        start_point_lon : float
            起点的经度（投放终点到第一条测线的最近端点）
        start_point_lat : float
            起点的纬度
        use_start_point : bool
            True表示从测线起点进入，False表示从测线终点进入
            
        Returns
        -------
        line_order : List[int]
            测线访问顺序（索引列表）
        total_distance : float
            总距离（公里）
        """
        if not self.survey_lines or len(self.survey_lines) == 0:
            return [], 0.0
        
        if len(self.survey_lines) == 1:
            # 只有一条测线，直接返回
            return [0], 0.0
        
        # 添加调试日志
        if hasattr(self, 'log_result'):
            self.log_result(f"[优化方法] 开始优化测线顺序")
            self.log_result(f"[优化方法] 用户选择的第一条测线索引: {first_line_idx}")
            self.log_result(f"[优化方法] 传入的测线起点坐标: ({start_point_lon:.6f}°, {start_point_lat:.6f}°)")
            self.log_result(f"[优化方法] use_start_point 参数: {use_start_point}")
            self.log_result(f"[优化方法]   含义: {'从测线起点进入' if use_start_point else '从测线终点进入（反向）'}")
            self.log_result(f"[优化方法] 总测线数: {len(self.survey_lines)}")
        
        # 使用最近邻算法优化测线顺序，从用户选择的测线开始
        n_lines = len(self.survey_lines)
        visited = [False] * n_lines
        line_order = []
        total_distance = 0.0
        
        # 第一条测线是用户选择的
        current_line_idx = first_line_idx
        visited[current_line_idx] = True
        line_order.append(current_line_idx)
        
        if hasattr(self, 'log_result'):
            self.log_result(f"[优化方法] 第一条测线已设置: 测线 {current_line_idx}")
        
        # 确定当前测线的退出点（如果从起点进入，则从终点退出；如果从终点进入，则从起点退出）
        current_line = self.survey_lines[current_line_idx]
        line_start_coord = current_line[0]
        line_end_coord = current_line[-1]
        
        if hasattr(self, 'log_result'):
            self.log_result(f"[优化方法] 测线 {current_line_idx} 端点1 (line[0]): ({line_start_coord[0]:.6f}°, {line_start_coord[1]:.6f}°)")
            self.log_result(f"[优化方法] 测线 {current_line_idx} 端点2 (line[-1]): ({line_end_coord[0]:.6f}°, {line_end_coord[1]:.6f}°)")
        
        # 保存第一条测线的进入点和退出点
        if not hasattr(self, 'survey_line_entry_points'):
            self.survey_line_entry_points = {}
        if not hasattr(self, 'survey_line_exit_points'):
            self.survey_line_exit_points = {}
        
        if use_start_point:
            # 从起点进入，从终点退出
            entry_point = (line_start_coord[0], line_start_coord[1])
            exit_point = (line_end_coord[0], line_end_coord[1])
            current_lon, current_lat = exit_point
            if hasattr(self, 'log_result'):
                self.log_result(f"[优化方法] 从起点进入，进入点: ({entry_point[0]:.6f}°, {entry_point[1]:.6f}°)")
                self.log_result(f"[优化方法] 退出点: ({exit_point[0]:.6f}°, {exit_point[1]:.6f}°)")
        else:
            # 从终点进入，从起点退出
            entry_point = (line_end_coord[0], line_end_coord[1])
            exit_point = (line_start_coord[0], line_start_coord[1])
            current_lon, current_lat = exit_point
            if hasattr(self, 'log_result'):
                self.log_result(f"[优化方法] 从终点进入（反向），进入点: ({entry_point[0]:.6f}°, {entry_point[1]:.6f}°)")
                self.log_result(f"[优化方法] 退出点: ({exit_point[0]:.6f}°, {exit_point[1]:.6f}°)")
        
        self.survey_line_entry_points[current_line_idx] = entry_point
        self.survey_line_exit_points[current_line_idx] = exit_point
        
        # 继续优化剩余测线的顺序
        for _ in range(n_lines - 1):
            min_distance = float('inf')
            next_line_idx = -1
            next_use_start_point = True  # 记录选择的端点
            
            for i in range(n_lines):
                if visited[i]:
                    continue
                
                # 计算到测线两个端点的距离，选择距离更近的端点
                line = self.survey_lines[i]
                line_start = line[0]
                line_end = line[-1]
                
                dist_to_start = StationPathOptimizer.haversine_distance(
                    current_lon, current_lat, line_start[0], line_start[1]
                )
                dist_to_end = StationPathOptimizer.haversine_distance(
                    current_lon, current_lat, line_end[0], line_end[1]
                )
                
                # 选择距离更近的端点
                if dist_to_start <= dist_to_end:
                    distance = dist_to_start
                    use_start = True
                else:
                    distance = dist_to_end
                    use_start = False
                
                if distance < min_distance:
                    min_distance = distance
                    next_line_idx = i
                    next_use_start_point = use_start
            
            if next_line_idx == -1:
                break
            
            # 添加下一条测线
            visited[next_line_idx] = True
            line_order.append(next_line_idx)
            total_distance += min_distance
            
            # 根据选择的端点确定进入点和退出点
            next_line = self.survey_lines[next_line_idx]
            if next_use_start_point:
                # 从起点进入，从终点退出
                next_entry_point = (next_line[0][0], next_line[0][1])
                next_exit_point = (next_line[-1][0], next_line[-1][1])
            else:
                # 从终点进入，从起点退出（反向）
                next_entry_point = (next_line[-1][0], next_line[-1][1])
                next_exit_point = (next_line[0][0], next_line[0][1])
            
            self.survey_line_entry_points[next_line_idx] = next_entry_point
            self.survey_line_exit_points[next_line_idx] = next_exit_point
            
            if hasattr(self, 'log_result'):
                entry_desc = "起点" if next_use_start_point else "终点（反向）"
                self.log_result(f"[优化方法] 测线 {next_line_idx}: 从{entry_desc}进入")
                self.log_result(f"[优化方法]   进入点: ({next_entry_point[0]:.6f}°, {next_entry_point[1]:.6f}°)")
                self.log_result(f"[优化方法]   退出点: ({next_exit_point[0]:.6f}°, {next_exit_point[1]:.6f}°)")
                self.log_result(f"[优化方法]   距离: {min_distance:.6f} km")
            
            # 更新当前位置为下一条测线的退出点
            current_lon, current_lat = next_exit_point
        
        return line_order, total_distance
    
    def optimize_survey_line_order(self, start_point_lon, start_point_lat):
        """
        优化测线顺序，从给定起点开始，找到最优的测线访问顺序
        
        Parameters
        ----------
        start_point_lon : float
            起点的经度
        start_point_lat : float
            起点的纬度
            
        Returns
        -------
        line_order : List[int]
            测线访问顺序（索引列表）
        total_distance : float
            总距离（公里）
        """
        if not self.survey_lines or len(self.survey_lines) == 0:
            return [], 0.0
        
        if len(self.survey_lines) == 1:
            # 只有一条测线，直接返回
            nearest_line_idx, nearest_point, distance = self.find_nearest_survey_line_point(
                start_point_lon, start_point_lat, use_start=True
            )
            return [0], distance
        
        # 使用最近邻算法优化测线顺序
        n_lines = len(self.survey_lines)
        visited = [False] * n_lines
        line_order = []
        total_distance = 0.0
        
        # 从起点找到最近的测线起点
        current_lon, current_lat = start_point_lon, start_point_lat
        
        for _ in range(n_lines):
            min_distance = float('inf')
            next_line_idx = -1
            
            for i in range(n_lines):
                if visited[i]:
                    continue
                
                # 计算到测线起点的距离
                line_start = self.survey_lines[i][0]
                distance = StationPathOptimizer.haversine_distance(
                    current_lon, current_lat, line_start[0], line_start[1]
                )
                
                if distance < min_distance:
                    min_distance = distance
                    next_line_idx = i
            
            if next_line_idx >= 0:
                visited[next_line_idx] = True
                line_order.append(next_line_idx)
                total_distance += min_distance
                # 移动到这条测线的终点
                current_lon, current_lat = self.survey_lines[next_line_idx][-1]
        
        return line_order, total_distance
    
    def calculate_turn_paths(self, turn_radius_km):
        """
        计算测线之间的转弯路径
        
        Parameters
        ----------
        turn_radius_km : float
            转弯半径（公里）
            
        Returns
        -------
        turn_paths : List[List[Tuple[float, float]]]
            转弯路径列表，每个元素是一条转弯路径的坐标列表
        """
        if not self.survey_lines or len(self.survey_lines) < 2:
            return []
        
        turn_paths = []
        
        for i in range(len(self.survey_lines) - 1):
            line1 = self.survey_lines[i]
            line2 = self.survey_lines[i + 1]
            
            if len(line1) < 2 or len(line2) < 2:
                continue
            
            # 获取第一条测线的终点和第二条测线的起点
            exit_point = line1[-1]  # 第一条测线的终点
            entry_point = line2[0]  # 第二条测线的起点
            
            # 使用完整的过渡路径：测线1退出点 → 转弯路径（圆弧）→ 航渡路径起点 → 
            # 航渡路径（直线）→ 航渡路径终点 → 转弯路径（圆弧）→ 测线2进入点
            turn_path = self._calculate_transition_path(
                exit_point, entry_point, line1, line2, turn_radius_km
            )
            turn_paths.append(turn_path)
        
        return turn_paths
    
    def _calculate_arc_path(self, start, end, turn_radius_km):
        """
        计算两点之间的圆弧路径
        
        Parameters
        ----------
        start : Tuple[float, float]
            起点坐标 (lon, lat)
        end : Tuple[float, float]
            终点坐标 (lon, lat)
        turn_radius_km : float
            转弯半径（公里）
            
        Returns
        -------
        path : List[Tuple[float, float]]
            路径坐标列表
        """
        lon1, lat1 = start
        lon2, lat2 = end
        
        # 计算两点间的大圆距离
        distance_km = StationPathOptimizer.haversine_distance(lon1, lat1, lon2, lat2)
        
        # 如果距离小于转弯半径的2倍，直接连接
        if distance_km < turn_radius_km * 2:
            return [start, end]
        
        # 计算转弯路径：使用大圆路径，路径的平滑度与转弯半径相关
        # 转弯半径越小，需要更多的点来保证路径平滑
        # 计算中间点数量（根据转弯半径和距离）
        # 确保最小点数为20，转弯半径越小，点数越多
        n_points = max(20, int(distance_km / max(turn_radius_km, 0.1) * 30))
        path = []
        
        for i in range(n_points + 1):
            t = i / n_points
            
            # 球面线性插值（Slerp）
            # 转换为弧度
            lon1_rad, lat1_rad = np.radians([lon1, lat1])
            lon2_rad, lat2_rad = np.radians([lon2, lat2])
            
            # 计算大圆路径上的点
            d = np.arccos(np.sin(lat1_rad) * np.sin(lat2_rad) + 
                         np.cos(lat1_rad) * np.cos(lat2_rad) * np.cos(lon2_rad - lon1_rad))
            
            if d == 0:
                path.append((lon1, lat1))
                continue
            
            a = np.sin((1 - t) * d) / np.sin(d)
            b = np.sin(t * d) / np.sin(d)
            
            x = a * np.cos(lat1_rad) * np.cos(lon1_rad) + b * np.cos(lat2_rad) * np.cos(lon2_rad)
            y = a * np.cos(lat1_rad) * np.sin(lon1_rad) + b * np.cos(lat2_rad) * np.sin(lon2_rad)
            z = a * np.sin(lat1_rad) + b * np.sin(lat2_rad)
            
            lat_rad = np.arcsin(z)
            lon_rad = np.arctan2(y, x)
            
            path.append((np.degrees(lon_rad), np.degrees(lat_rad)))
        
        return path
    
    @staticmethod
    def _calculate_bearing(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """
        计算从点1到点2的方位角（度）
        
        Parameters
        ----------
        lon1, lat1 : float
            起点坐标（度）
        lon2, lat2 : float
            终点坐标（度）
            
        Returns
        -------
        bearing : float
            方位角（度），0-360度，0度为正北方向
        """
        lon1_rad, lat1_rad = np.radians([lon1, lat1])
        lon2_rad, lat2_rad = np.radians([lon2, lat2])
        
        dlon = lon2_rad - lon1_rad
        
        y = np.sin(dlon) * np.cos(lat2_rad)
        x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
        
        bearing_rad = np.arctan2(y, x)
        bearing_deg = np.degrees(bearing_rad)
        
        # 转换为0-360度
        bearing_deg = (bearing_deg + 360) % 360
        
        return bearing_deg
    
    def _calculate_line_direction(self, line: List[Tuple[float, float]]) -> float:
        """
        计算测线的方向（方位角）
        
        Parameters
        ----------
        line : List[Tuple[float, float]]
            测线坐标列表
            
        Returns
        -------
        direction : float
            测线方向（度），0-360度
        """
        if len(line) < 2:
            return 0.0
        
        # 使用测线的起点和终点计算方向
        start = line[0]
        end = line[-1]
        
        return self._calculate_bearing(start[0], start[1], end[0], end[1])
    
    def _calculate_point_at_distance(self, lon: float, lat: float, bearing: float, distance_km: float) -> Tuple[float, float]:
        """
        从给定点出发，沿着指定方位角前进指定距离，计算目标点坐标
        
        Parameters
        ----------
        lon, lat : float
            起点坐标（度）
        bearing : float
            方位角（度）
        distance_km : float
            距离（公里）
            
        Returns
        -------
        target_lon, target_lat : Tuple[float, float]
            目标点坐标（度）
        """
        lon_rad = np.radians(lon)
        lat_rad = np.radians(lat)
        bearing_rad = np.radians(bearing)
        
        # 角距离（弧度）
        angular_distance = distance_km / EARTH_RADIUS_KM
        
        # 计算目标点纬度
        target_lat_rad = np.arcsin(
            np.sin(lat_rad) * np.cos(angular_distance) +
            np.cos(lat_rad) * np.sin(angular_distance) * np.cos(bearing_rad)
        )
        
        # 计算目标点经度
        target_lon_rad = lon_rad + np.arctan2(
            np.sin(bearing_rad) * np.sin(angular_distance) * np.cos(lat_rad),
            np.cos(angular_distance) - np.sin(lat_rad) * np.sin(target_lat_rad)
        )
        
        return (np.degrees(target_lon_rad), np.degrees(target_lat_rad))
    
    def _calculate_turn_arc(self, start_point: Tuple[float, float], start_bearing: float,
                            end_point: Tuple[float, float], end_bearing: float,
                            radius_km: float) -> List[Tuple[float, float]]:
        """
        计算球面上的转弯圆弧路径
        
        使用基于球面几何的方法：从起点沿起始方向前进半径距离得到切点，
        然后使用球面线性插值生成从起点到切点的圆弧路径。
        
        Parameters
        ----------
        start_point : Tuple[float, float]
            圆弧起点坐标 (lon, lat)
        start_bearing : float
            起始方向（度）
        end_point : Tuple[float, float]
            圆弧终点坐标 (lon, lat) - 应该是切点
        end_bearing : float
            结束方向（度）- 切点处的方向
        radius_km : float
            转弯半径（公里）
            
        Returns
        -------
        path : List[Tuple[float, float]]
            圆弧路径坐标列表
        """
        # 计算切点：从起点沿起始方向前进半径距离
        tangent_point = self._calculate_point_at_distance(
            start_point[0], start_point[1], start_bearing, radius_km
        )
        
        # 验证切点是否与给定的终点一致（允许小的误差）
        dist_check = StationPathOptimizer.haversine_distance(
            tangent_point[0], tangent_point[1], end_point[0], end_point[1]
        )
        
        # 如果切点与终点不一致，使用计算的切点
        if dist_check > 0.01:  # 如果距离大于10米，使用计算的切点
            end_point = tangent_point
        
        # 计算从起点到切点的距离（应该是半径）
        dist_to_tangent = StationPathOptimizer.haversine_distance(
            start_point[0], start_point[1], end_point[0], end_point[1]
        )
        
        # 如果距离太小，直接返回
        if dist_to_tangent < 0.001:
            return [start_point, end_point]
        
        # 计算从起点到切点的方位角
        bearing_to_tangent = self._calculate_bearing(
            start_point[0], start_point[1], end_point[0], end_point[1]
        )
        
        # 计算角度差（从起始方向到切点方向）
        angle_diff = (bearing_to_tangent - start_bearing + 180) % 360 - 180
        
        # 如果角度差很小，直接连接
        if abs(angle_diff) < 0.5:
            return [start_point, end_point]
        
        # 计算圆弧上的点
        # 使用球面线性插值（slerp）从起点到切点
        # 圆弧的角度范围是从起始方向到切点方向
        n_points = max(30, int(abs(angle_diff) / 1))  # 每1度一个点，最少30个点
        
        path = [start_point]
        
        # 转换为弧度
        lon1_rad, lat1_rad = np.radians([start_point[0], start_point[1]])
        lon2_rad, lat2_rad = np.radians([end_point[0], end_point[1]])
        
        # 计算大圆距离
        d = np.arccos(np.sin(lat1_rad) * np.sin(lat2_rad) + 
                     np.cos(lat1_rad) * np.cos(lat2_rad) * np.cos(lon2_rad - lon1_rad))
        
        if d == 0:
            return [start_point, end_point]
        
        # 生成圆弧上的点
        for i in range(1, n_points):
            t = i / n_points
            
            # 球面线性插值
            a = np.sin((1 - t) * d) / np.sin(d)
            b = np.sin(t * d) / np.sin(d)
            
            x = a * np.cos(lat1_rad) * np.cos(lon1_rad) + b * np.cos(lat2_rad) * np.cos(lon2_rad)
            y = a * np.cos(lat1_rad) * np.sin(lon1_rad) + b * np.cos(lat2_rad) * np.sin(lon2_rad)
            z = a * np.sin(lat1_rad) + b * np.sin(lat2_rad)
            
            lat_rad = np.arcsin(np.clip(z, -1, 1))
            lon_rad = np.arctan2(y, x)
            
            arc_point = (np.degrees(lon_rad), np.degrees(lat_rad))
            path.append(arc_point)
        
        # 确保最后一个点是切点
        path.append(end_point)
        
        return path
    
    def _calculate_arc_by_angle(self, start_point: Tuple[float, float], start_bearing: float,
                                 turn_angle_deg: float, turn_radius_km: float) -> List[Tuple[float, float]]:
        """
        计算从起点沿起始方向转过一定角度后的圆弧路径
        
        在球面上，圆弧路径是通过逐步改变方向并前进小段距离来生成的。
        转弯角度对应的弧长 = 半径 × 角度（弧度）
        
        Parameters
        ----------
        start_point : Tuple[float, float]
            起点坐标 (lon, lat)
        start_bearing : float
            起始方向（度）
        turn_angle_deg : float
            转弯角度（度），正值表示顺时针，负值表示逆时针
        turn_radius_km : float
            转弯半径（公里）
            
        Returns
        -------
        path : List[Tuple[float, float]]
            圆弧路径坐标列表
        """
        # 计算转弯角度对应的弧长
        # 弧长 = 半径 × 角度（弧度）
        turn_angle_rad = np.radians(abs(turn_angle_deg))
        arc_length_km = turn_radius_km * turn_angle_rad
        
        # 如果角度或弧长太小，直接返回起点
        if abs(turn_angle_deg) < 0.1 or arc_length_km < 0.001:
            return [start_point]
        
        # 生成圆弧路径上的点
        # 通过逐步改变方向并前进小段距离来生成圆弧
        n_points = max(30, int(abs(turn_angle_deg) / 1))  # 每1度一个点，最少30个点
        path = [start_point]
        
        current_point = start_point
        current_bearing = start_bearing
        
        # 计算每一步的角度增量
        angle_step = turn_angle_deg / n_points
        # 计算每一步的弧长增量
        arc_step_km = arc_length_km / n_points
        
        # 逐步生成圆弧上的点
        for i in range(1, n_points + 1):
            # 更新方向：逐步转向
            current_bearing = (current_bearing + angle_step) % 360
            
            # 从当前点沿当前方向前进一小段距离
            next_point = self._calculate_point_at_distance(
                current_point[0], current_point[1], current_bearing, arc_step_km
            )
            
            path.append(next_point)
            current_point = next_point
        
        return path
    
    def _calculate_transition_path(self, exit_point: Tuple[float, float], entry_point: Tuple[float, float],
                                   line1: List[Tuple[float, float]], line2: List[Tuple[float, float]],
                                   turn_radius_km: float) -> List[Tuple[float, float]]:
        """
        计算完整的过渡路径：测线1退出点 → 转弯路径（圆弧，转过一定弧长）→ 
        航渡路径（直线）→ 转弯路径（圆弧，转过一定弧长）→ 测线2进入点
        
        流程：
        1. 从测线端点开始转弯，转过一定的弧线长度
        2. 以直线开往下个测线端点
        3. 快到端点时，再次开始转弯，转过一定的弧线长度，上线
        
        Parameters
        ----------
        exit_point : Tuple[float, float]
            测线1退出点坐标 (lon, lat)
        entry_point : Tuple[float, float]
            测线2进入点坐标 (lon, lat)
        line1 : List[Tuple[float, float]]
            测线1的坐标列表
        line2 : List[Tuple[float, float]]
            测线2的坐标列表
        turn_radius_km : float
            转弯半径（公里）
            
        Returns
        -------
        path : List[Tuple[float, float]]
            完整的过渡路径坐标列表
        """
        # 计算测线1的方向（从退出点沿测线方向）
        line1_direction = self._calculate_line_direction(line1)
        
        # 计算测线2的方向（从进入点沿测线方向）
        line2_direction = self._calculate_line_direction(line2)
        
        # 计算从退出点到进入点的直接距离
        direct_distance = StationPathOptimizer.haversine_distance(
            exit_point[0], exit_point[1], entry_point[0], entry_point[1]
        )
        
        # 如果距离太短，直接连接
        if direct_distance < turn_radius_km * 2:
            return [exit_point, entry_point]
        
        # 计算航渡方向（从退出点到进入点的直接方向）
        transit_bearing = self._calculate_bearing(
            exit_point[0], exit_point[1], entry_point[0], entry_point[1]
        )
        
        # 计算第一个转弯的角度：从测线1方向转向航渡方向
        angle_diff1 = (transit_bearing - line1_direction + 180) % 360 - 180
        
        # 确定转弯角度（通常为90度或根据角度差确定）
        # 如果角度差小于90度，使用90度转弯；否则使用实际角度差
        turn_angle1 = 90.0 if abs(angle_diff1) > 45 else abs(angle_diff1)
        # 根据角度差的符号确定转弯方向
        if angle_diff1 < 0:
            turn_angle1 = -turn_angle1
        
        # 计算第一个圆弧的终点（转弯后的点）
        # 从退出点沿测线1方向转过turn_angle1角度
        arc1_end_bearing = (line1_direction + turn_angle1) % 360
        # 计算转弯弧长
        turn_angle_rad1 = np.radians(abs(turn_angle1))
        arc_length1_km = turn_radius_km * turn_angle_rad1
        transit_start = self._calculate_point_at_distance(
            exit_point[0], exit_point[1], arc1_end_bearing, arc_length1_km
        )
        
        # 计算航渡路径的方向（从transit_start到进入点的方向）
        # 先使用从transit_start到entry_point的方向作为初始航渡方向
        initial_transit_bearing = self._calculate_bearing(
            transit_start[0], transit_start[1], entry_point[0], entry_point[1]
        )
        
        # 计算第二个转弯的角度：从航渡方向转向测线2方向
        angle_diff2 = (line2_direction - initial_transit_bearing + 180) % 360 - 180
        
        # 确定第二个转弯角度（通常为90度或根据角度差确定）
        turn_angle2 = 90.0 if abs(angle_diff2) > 45 else abs(angle_diff2)
        if angle_diff2 < 0:
            turn_angle2 = -turn_angle2
        
        # 计算第二个圆弧的起点（转弯前的点）
        # 从进入点沿测线2反方向后退弧长距离，得到转弯起点
        line2_reverse_direction = (line2_direction + 180) % 360
        turn_angle_rad2 = np.radians(abs(turn_angle2))
        arc_length2_km = turn_radius_km * turn_angle_rad2
        transit_end = self._calculate_point_at_distance(
            entry_point[0], entry_point[1], line2_reverse_direction, arc_length2_km
        )
        
        # 重新计算航渡路径长度（从第一个转弯终点到第二个转弯起点）
        transit_distance = StationPathOptimizer.haversine_distance(
            transit_start[0], transit_start[1], transit_end[0], transit_end[1]
        )
        
        # 如果航渡路径太短，简化处理
        if transit_distance < turn_radius_km * 0.5:
            # 使用简化的圆弧路径
            return self._calculate_arc_path(exit_point, entry_point, turn_radius_km)
        
        # 计算航渡路径的方向（从transit_start到transit_end）
        transit_bearing_final = self._calculate_bearing(
            transit_start[0], transit_start[1], transit_end[0], transit_end[1]
        )
        
        # 构建完整路径
        full_path = []
        
        # 1. 第一个圆弧：从退出点开始转弯，转过一定的弧线长度
        arc1 = self._calculate_arc_by_angle(
            exit_point, line1_direction, turn_angle1, turn_radius_km
        )
        full_path.extend(arc1[:-1])  # 不包括最后一个点（会与下一段重复）
        
        # 2. 直线航渡路径：从第一个转弯终点到第二个转弯起点
        n_transit_points = max(10, int(transit_distance / 0.1))  # 每0.1公里一个点
        
        for i in range(1, n_transit_points + 1):  # 从1开始，避免重复第一个点
            t = i / n_transit_points
            transit_point = self._calculate_point_at_distance(
                transit_start[0], transit_start[1], transit_bearing_final, 
                transit_distance * t
            )
            full_path.append(transit_point)
        
        # 3. 第二个圆弧：快到端点时，再次开始转弯，转过一定的弧线长度，上线
        # 从transit_end沿航渡方向转向测线2方向
        arc2 = self._calculate_arc_by_angle(
            transit_end, transit_bearing_final, turn_angle2, turn_radius_km
        )
        full_path.extend(arc2[1:])  # 不包括第一个点（会与上一段重复）
        
        # 确保最后一个点是进入点
        if len(full_path) == 0 or full_path[-1] != entry_point:
            full_path.append(entry_point)
        
        return full_path
    
    def plot_turn_paths(self):
        """绘制转弯路径（根据用户选择）"""
        # 清除之前的转弯路径
        for artist in self.turn_path_artists:
            try:
                artist.remove()
            except:
                pass
        self.turn_path_artists.clear()
        
        # 检查用户是否选择显示转弯路径
        if not hasattr(self, 'show_turn_paths_var') or not self.show_turn_paths_var.get():
            return
        
        if not self.turn_paths:
            return
        
        # 绘制每条转弯路径
        for i, turn_path in enumerate(self.turn_paths):
            if len(turn_path) < 2:
                continue
            
            lons = [p[0] for p in turn_path]
            lats = [p[1] for p in turn_path]
            
            # 绘制转弯路径（使用橙色点线）
            turn_artist, = self.ax.plot(lons, lats, 'orange', linestyle=':', 
                                       linewidth=2, alpha=0.8, 
                                       label='Turn Path' if i == 0 else '', zorder=2)
            self.turn_path_artists.append(turn_artist)
    
    def optimize_path(self):
        """执行路径优化（第一阶段：只优化投放路径）"""
        if not self.stations:
            messagebox.showwarning('警告', '请先加载站位文件')
            return
        
        if len(self.stations) < 2:
            messagebox.showerror('错误', '至少需要2个站位点')
            return
        
        if self.start_idx is None:
            messagebox.showwarning('警告', '请先选择投放起点')
            return
        
        try:
            # 获取优化方法
            method = self.method_var.get()
            
            self.log_result(f"\n{'='*60}")
            self.log_result(f"开始第一阶段路径优化（投放路径）...")
            self.log_result(f"优化方法: {method}")
            self.log_result(f"起点: 站位 {self.start_idx}")
            self.root.update()  # 更新界面
            
            # ========== 第一阶段：优化投放路径 ==========
            self.log_result(f"\n【第一阶段】优化投放路径...")
            # 获取用于投放的站位索引
            deployment_indices = self.get_deployment_stations()
            if not deployment_indices:
                messagebox.showerror('错误', '没有可用的投放站位，请至少选择一个投放站位类型')
                return
            
            # 创建只包含投放站位的坐标列表
            deployment_stations = [self.stations[i] for i in deployment_indices]
            
            # 转换起点索引（如果指定了起点）
            deployment_start_idx = None
            if self.start_idx is not None and self.start_idx in deployment_indices:
                deployment_start_idx = deployment_indices.index(self.start_idx)
            
            # 转换终点索引（优先使用用户选择的终点）
            deployment_end_idx = None
            
            # 1. 优先检查用户是否明确选择了终点
            if self.selected_end_idx is not None and self.selected_end_idx in deployment_indices:
                deployment_end_idx = deployment_indices.index(self.selected_end_idx)
                self.log_result(f"使用用户选择的终点: 站位 {self.selected_end_idx}")
            elif self.end_idx is not None and self.end_idx in deployment_indices:
                deployment_end_idx = deployment_indices.index(self.end_idx)
                self.log_result(f"使用设置的终点: 站位 {self.end_idx}")
            # 2. 如果没有用户选择的终点，检查可逆性需求
            elif (self.previous_deployment_start_idx is not None and
                  self.previous_deployment_start_idx in deployment_indices):
                # 检查当前起点是否是上一次的终点
                if (hasattr(self, 'deployment_path') and self.deployment_path and 
                    self.start_idx == self.deployment_path[-1]):
                    # 当前起点是上一次的终点，将上一次的起点设为终点（确保可逆性）
                    deployment_end_idx = deployment_indices.index(self.previous_deployment_start_idx)
                    self.log_result(f"检测到可逆性需求：当前起点 {self.start_idx} 是上一次的终点，将上一次的起点 {self.previous_deployment_start_idx} 设为终点")
            
            if deployment_end_idx is not None:
                self.log_result(f"投放路径将固定终点: 站位 {deployment_indices[deployment_end_idx]}")
            else:
                self.log_result(f"投放路径不固定终点（将自动选择最优终点）")
            
            if self.deployment_type_vars:
                selected_deploy_types = [t for t, var in self.deployment_type_vars.items() if var.get()]
                if selected_deploy_types:
                    self.log_result(f"投放站位类型: {selected_deploy_types}")
            self.log_result(f"可用投放站位数: {len(deployment_indices)}")
            
            deployment_optimizer = StationPathOptimizer(
                stations=deployment_stations,
                method=method,
                start_idx=deployment_start_idx,
                end_idx=deployment_end_idx  # 如果设置了，确保路径以该点结束
            )
            deployment_path_local, self.deployment_distance = deployment_optimizer.optimize()
            
            # 将本地索引转换回全局索引
            self.deployment_path = [deployment_indices[i] for i in deployment_path_local]
            self.log_result(f"投放路径顺序: {self.deployment_path}")
            self.log_result(f"投放路径总距离: {self.deployment_distance:.2f} km")
            
            # 获取最后一个投放站位的坐标（这就是投放终点）
            last_deployment_idx = self.deployment_path[-1]
            last_deployment_lon, last_deployment_lat = self.stations[last_deployment_idx]
            
            # 保存当前的起点，用于下次计算时确保可逆性
            if self.start_idx is not None:
                self.previous_deployment_start_idx = self.start_idx
            
            # 验证计算的终点是否与用户选择的终点一致
            if deployment_end_idx is not None:
                expected_end_idx = deployment_indices[deployment_end_idx]
                if last_deployment_idx != expected_end_idx:
                    self.log_result(f"警告：计算的终点 {last_deployment_idx} 与期望的终点 {expected_end_idx} 不一致！")
                    messagebox.showwarning('警告', 
                        f'计算的终点与选择的终点不一致！\n\n'
                        f'选择的终点: 站位 {expected_end_idx}\n'
                        f'计算的终点: 站位 {last_deployment_idx}\n\n'
                        f'这可能是因为路径优化算法无法找到以选择终点结束的路径。')
                else:
                    self.log_result(f"✓ 计算的终点与选择的终点一致: 站位 {last_deployment_idx}")
            
            # 如果用户没有选择终点，自动设置并标注终点
            if self.selected_end_idx is None:
                self.selected_end_idx = last_deployment_idx
                self.end_idx = last_deployment_idx
            else:
                # 用户选择了终点，保持用户的选择，但更新end_idx为实际计算的终点
                self.end_idx = last_deployment_idx
                # 如果计算的终点与用户选择的不同，提示用户
                if last_deployment_idx != self.selected_end_idx:
                    self.log_result(f"注意：实际计算的终点 {last_deployment_idx} 与用户选择的终点 {self.selected_end_idx} 不同")
            
            # 更新终点下拉菜单
            if self.end_combo:
                # 找到终点在列表中的位置
                end_position = last_deployment_idx + 1  # +1 因为第一个是"未选择"
                combo_values = self.end_combo['values']
                if combo_values and end_position < len(combo_values):
                    self.end_combo.current(end_position)
            
            self.log_result(f"\n投放终点已自动标注: 站位 {last_deployment_idx}")
            self.log_result(f"投放终点坐标: ({last_deployment_lon:.4f}°, {last_deployment_lat:.4f}°)")
            
            # 为了兼容性，创建一个包含投放路径的optimizer对象
            self.optimizer = deployment_optimizer
            self.optimizer.optimal_path = self.deployment_path
            self.optimizer.total_distance = self.deployment_distance
            
            # 显示结果
            self.log_result(f"\n{'='*60}")
            self.log_result(f"第一阶段路径优化完成!")
            self.log_result(f"投放路径总距离: {self.deployment_distance:.2f} km")
            
            # 显示详细路径信息
            self.log_result(f"\n投放路径:")
            for i, idx in enumerate(self.deployment_path):
                lon, lat = self.stations[idx]
                station_name = self.station_types[idx] if idx < len(self.station_names) and self.station_names[idx] else f'站位 {idx}'
                self.log_result(f"  {i+1}. {station_name} (索引 {idx}): ({lon:.4f}°, {lat:.4f}°)")
            
            # 重新绘制站位和路径（会显示终点）
            self.plot_stations()
            self.plot_path()
            
            messagebox.showinfo('完成', 
                f'第一阶段路径优化完成!\n\n'
                f'投放路径总距离: {self.deployment_distance:.2f} km\n\n'
                f'投放终点已自动标注\n\n'
                f'请根据终点选择要进的第一条测线')
            
        except Exception as e:
            messagebox.showerror('错误', f'路径优化失败: {str(e)}')
            import traceback
            traceback.print_exc()
            self.log_result(f"\n错误: {str(e)}")
    
    def optimize_survey_and_recovery_path(self):
        """执行路径优化（第二阶段和第三阶段：测线路径和回收路径）"""
        if not self.survey_lines:
            messagebox.showwarning('警告', '请先加载作业测线文件')
            return
        
        if self.selected_end_idx is None or self.deployment_path is None or len(self.deployment_path) == 0:
            messagebox.showwarning('警告', '请先完成第一阶段优化（投放路径）')
            return
        
        # 检查用户是否选择了第一条测线
        if self.selected_first_survey_line_idx is None:
            messagebox.showwarning('警告', '请先选择第一条测线\n\n请使用"点击测线选择"按钮或下拉菜单选择第一条测线')
            return
        
        try:
            # 获取优化方法
            method = self.method_var.get()
            
            self.log_result(f"\n{'='*60}")
            self.log_result(f"开始第二阶段和第三阶段路径优化...")
            self.log_result(f"优化方法: {method}")
            
            # 获取最后一个投放站位的坐标（投放终点）
            last_deployment_idx = self.deployment_path[-1]
            last_deployment_lon, last_deployment_lat = self.stations[last_deployment_idx]
            
            # ========== 第二阶段：优化测线路径 ==========
            self.log_result(f"\n【第二阶段】优化测线路径...")
            
            first_line_idx = self.selected_first_survey_line_idx
            first_line = self.survey_lines[first_line_idx]
            
            if len(first_line) < 2:
                messagebox.showerror('错误', f'选择的测线 {first_line_idx} 数据无效')
                return
            
            # 根据投放终点和这条线的两个端点的距离，自动确定哪个端点是这个测线的起点
            self.log_result(f"\n[计算测线起点] 开始计算测线 {first_line_idx} 的起点...")
            self.log_result(f"[计算测线起点] 投放终点坐标: ({last_deployment_lon:.6f}°, {last_deployment_lat:.6f}°)")
            
            line_start = first_line[0]
            line_end = first_line[-1]
            self.log_result(f"[计算测线起点] 测线端点1 (line[0]): ({line_start[0]:.6f}°, {line_start[1]:.6f}°)")
            self.log_result(f"[计算测线起点] 测线端点2 (line[-1]): ({line_end[0]:.6f}°, {line_end[1]:.6f}°)")
            
            dist_to_start = StationPathOptimizer.haversine_distance(
                last_deployment_lon, last_deployment_lat, line_start[0], line_start[1]
            )
            dist_to_end = StationPathOptimizer.haversine_distance(
                last_deployment_lon, last_deployment_lat, line_end[0], line_end[1]
            )
            
            self.log_result(f"[计算测线起点] 投放终点到端点1的距离: {dist_to_start:.6f} km")
            self.log_result(f"[计算测线起点] 投放终点到端点2的距离: {dist_to_end:.6f} km")
            
            # 选择距离更近的端点作为进入点（测线起点）
            if dist_to_start <= dist_to_end:
                nearest_point = line_start
                self.deployment_to_survey_distance = dist_to_start
                use_start_point = True
                self.log_result(f"[计算测线起点] ✓ 选择端点1作为测线起点")
                self.log_result(f"[计算测线起点]   原因: 端点1距离更近 ({dist_to_start:.6f} km <= {dist_to_end:.6f} km)")
                self.log_result(f"[计算测线起点]   测线起点坐标: ({nearest_point[0]:.6f}°, {nearest_point[1]:.6f}°)")
                self.log_result(f"[计算测线起点]   use_start_point = True (从测线起点进入)")
            else:
                nearest_point = line_end
                self.deployment_to_survey_distance = dist_to_end
                use_start_point = False
                self.log_result(f"[计算测线起点] ✓ 选择端点2作为测线起点")
                self.log_result(f"[计算测线起点]   原因: 端点2距离更近 ({dist_to_end:.6f} km < {dist_to_start:.6f} km)")
                self.log_result(f"[计算测线起点]   测线起点坐标: ({nearest_point[0]:.6f}°, {nearest_point[1]:.6f}°)")
                self.log_result(f"[计算测线起点]   use_start_point = False (从测线终点进入，即反向)")
            
            self.log_result(f"\n用户选择的第一条测线: 测线 {first_line_idx}")
            self.log_result(f"从投放终点到测线 {first_line_idx} 最近端点的距离: {self.deployment_to_survey_distance:.2f} km")
            
            # 优化测线顺序（从用户选择的测线开始）
            self.log_result(f"\n[验证] 用户选择的测线索引: {first_line_idx}")
            self.log_result(f"[验证] 传递给优化方法的参数:")
            self.log_result(f"[验证]   first_line_idx = {first_line_idx}")
            self.log_result(f"[验证]   start_point_lon = {nearest_point[0]:.6f}°")
            self.log_result(f"[验证]   start_point_lat = {nearest_point[1]:.6f}°")
            self.log_result(f"[验证]   use_start_point = {use_start_point}")
            
            self.survey_line_order, self.survey_line_transition_distance = \
                self.optimize_survey_line_order_from_selected(first_line_idx, nearest_point[0], nearest_point[1], use_start_point)
            
            # 验证第一条测线是否是用户选择的测线
            if self.survey_line_order and len(self.survey_line_order) > 0:
                actual_first_line_idx = self.survey_line_order[0]
                if actual_first_line_idx == first_line_idx:
                    self.log_result(f"[验证] ✓ 第一条测线正确: 测线 {actual_first_line_idx} (用户选择的测线)")
                else:
                    self.log_result(f"[验证] ✗ 错误！第一条测线是测线 {actual_first_line_idx}，但用户选择的是测线 {first_line_idx}")
                    messagebox.showwarning('警告', f'第一条测线不匹配！\n用户选择: 测线 {first_line_idx}\n实际第一条: 测线 {actual_first_line_idx}')
            
            self.log_result(f"测线访问顺序: {self.survey_line_order}")
            self.log_result(f"测线之间过渡总距离: {self.survey_line_transition_distance:.2f} km")
            
            # 计算测线总长度
            self.total_survey_length = 0.0
            for line_idx in self.survey_line_order:
                line = self.survey_lines[line_idx]
                line_length = 0.0
                for i in range(len(line) - 1):
                    lon1, lat1 = line[i]
                    lon2, lat2 = line[i + 1]
                    dist = StationPathOptimizer.haversine_distance(lon1, lat1, lon2, lat2)
                    line_length += dist
                self.total_survey_length += line_length
            self.log_result(f"测线总长度: {self.total_survey_length:.2f} km")
            
            # 获取最后一条测线的终点坐标
            last_survey_line_idx = self.survey_line_order[-1]
            last_survey_end_lon, last_survey_end_lat = self.survey_lines[last_survey_line_idx][-1]
            
            # ========== 第三阶段：优化回收路径 ==========
            self.log_result(f"\n【第三阶段】优化回收路径...")
            # 获取用于回收的站位索引
            recovery_indices = self.get_recovery_stations()
            if not recovery_indices:
                messagebox.showerror('错误', '没有可用的回收站位，请至少选择一个回收站位类型')
                return
            
            # 找到最近的回收站位作为回收起点
            min_recovery_start_distance = float('inf')
            recovery_start_idx_global = None
            for i in recovery_indices:
                lon, lat = self.stations[i]
                distance = StationPathOptimizer.haversine_distance(
                    last_survey_end_lon, last_survey_end_lat, lon, lat
                )
                if distance < min_recovery_start_distance:
                    min_recovery_start_distance = distance
                    recovery_start_idx_global = i
            
            if recovery_start_idx_global is None:
                messagebox.showerror('错误', '无法找到回收起点')
                return
            
            self.survey_to_recovery_distance = min_recovery_start_distance
            if self.recovery_type_vars:
                selected_recovery_types = [t for t, var in self.recovery_type_vars.items() if var.get()]
                if selected_recovery_types:
                    self.log_result(f"回收站位类型: {selected_recovery_types}")
            self.log_result(f"可用回收站位数: {len(recovery_indices)}")
            self.log_result(f"从最后一条测线终点到最近回收站位的距离: {self.survey_to_recovery_distance:.2f} km")
            recovery_start_name = self.station_names[recovery_start_idx_global] if recovery_start_idx_global < len(self.station_names) and self.station_names[recovery_start_idx_global] else f'站位 {recovery_start_idx_global}'
            self.log_result(f"回收起点: {recovery_start_name} (索引 {recovery_start_idx_global})")
            
            # 创建只包含回收站位的坐标列表
            recovery_stations = [self.stations[i] for i in recovery_indices]
            
            # 检查回收站位数量
            if len(recovery_stations) < 1:
                messagebox.showerror('错误', '没有可用的回收站位')
                return
            
            # 如果只有一个回收站位，直接返回该站位
            if len(recovery_stations) == 1:
                self.recovery_path = [recovery_start_idx_global]
                self.recovery_distance = 0.0
                self.log_result(f"回收路径顺序: {self.recovery_path}")
                self.log_result(f"回收路径总距离: {self.recovery_distance:.2f} km")
                
                # 更新回收路径状态显示
                if hasattr(self, 'recovery_path_status_var'):
                    self.recovery_path_status_var.set(f'已设置 ({len(self.recovery_path)} 个站位)')
            else:
                # 转换起点索引（回收路径终点由优化算法自动选择，不固定）
                recovery_start_idx_local = recovery_indices.index(recovery_start_idx_global)
                recovery_end_idx_local = None  # 不固定终点，让优化算法选择最优终点
                
                # 优化回收路径（从最近站位开始，访问所有回收站位，自动选择最优终点）
                recovery_optimizer = StationPathOptimizer(
                    stations=recovery_stations,
                    method=method,
                    start_idx=recovery_start_idx_local,
                    end_idx=recovery_end_idx_local  # 不固定终点，由优化算法自动选择最优终点
                )
                recovery_path_local, self.recovery_distance = recovery_optimizer.optimize()
                
                # 将本地索引转换回全局索引
                self.recovery_path = [recovery_indices[i] for i in recovery_path_local]
                self.log_result(f"回收路径顺序: {self.recovery_path}")
                self.log_result(f"回收路径总距离: {self.recovery_distance:.2f} km")
                
                # 更新回收路径状态显示
                if hasattr(self, 'recovery_path_status_var'):
                    self.recovery_path_status_var.set(f'已设置 ({len(self.recovery_path)} 个站位)')
            
            # 计算总距离（包含测线长度）
            cruise_distance = (self.deployment_to_survey_distance +
                             self.survey_line_transition_distance +
                             self.survey_to_recovery_distance)
            total_distance = (self.deployment_distance + 
                            self.total_survey_length +  # 添加测线长度
                            cruise_distance +
                            self.recovery_distance)
            
            # 更新optimizer对象的总距离
            if self.optimizer is not None:
                self.optimizer.total_distance = total_distance
            
            # 显示结果
            self.log_result(f"\n{'='*60}")
            self.log_result(f"三阶段路径优化完成!")
            self.log_result(f"\n总距离: {total_distance:.2f} km")
            self.log_result(f"  = 投放路径: {self.deployment_distance:.2f} km")
            self.log_result(f"  + 测线长度: {self.total_survey_length:.2f} km")
            self.log_result(f"  + 回收路径: {self.recovery_distance:.2f} km")
            self.log_result(f"  + 走航路径: {cruise_distance:.2f} km")
            self.log_result(f"    (到测线: {self.deployment_to_survey_distance:.2f} km + ")
            self.log_result(f"     测线过渡: {self.survey_line_transition_distance:.2f} km + ")
            self.log_result(f"     到回收: {self.survey_to_recovery_distance:.2f} km)")
            
            # 显示详细路径信息
            self.log_result(f"\n测线顺序:")
            for i, line_idx in enumerate(self.survey_line_order):
                self.log_result(f"  {i+1}. 测线 {line_idx}")
            
            self.log_result(f"\n回收路径:")
            for i, idx in enumerate(self.recovery_path):
                lon, lat = self.stations[idx]
                station_name = self.station_names[idx] if idx < len(self.station_names) and self.station_names[idx] else f'站位 {idx}'
                self.log_result(f"  {i+1}. {station_name} (索引 {idx}): ({lon:.4f}°, {lat:.4f}°)")
            
            # 重新绘制站位和路径
            self.plot_stations()
            self.plot_path()
            
            messagebox.showinfo('完成', 
                f'三阶段路径优化完成!\n\n'
                f'总距离: {total_distance:.2f} km\n'
                f'= 投放路径 {self.deployment_distance:.2f} km\n'
                f'+ 测线长度 {self.total_survey_length:.2f} km\n'
                f'+ 回收路径 {self.recovery_distance:.2f} km\n'
                f'+ 走航路径 {cruise_distance:.2f} km')
            
        except Exception as e:
            messagebox.showerror('错误', f'路径优化失败: {str(e)}')
            import traceback
            traceback.print_exc()
            self.log_result(f"\n错误: {str(e)}")
    
    def clear_path(self):
        """清除路径"""
        # 清除测线的进入和退出点
        if hasattr(self, 'survey_line_entry_points'):
            self.survey_line_entry_points.clear()
        if hasattr(self, 'survey_line_exit_points'):
            self.survey_line_exit_points.clear()
        
        if self.path_artist is not None:
            try:
                self.path_artist.remove()
            except:
                pass
            self.path_artist = None
        
        if self.deployment_path_artist is not None:
            try:
                self.deployment_path_artist.remove()
            except:
                pass
            self.deployment_path_artist = None
        
        if self.recovery_path_artist is not None:
            try:
                self.recovery_path_artist.remove()
            except:
                pass
            self.recovery_path_artist = None
        
        # 清除路径标注
        for label in self.path_labels:
            try:
                label.remove()
            except:
                pass
        self.path_labels.clear()
        
        for label in self.deployment_labels:
            try:
                label.remove()
            except:
                pass
        self.deployment_labels.clear()
        
        for label in self.recovery_labels:
            try:
                label.remove()
            except:
                pass
        self.recovery_labels.clear()
        
        # 清除转弯路径（测线保留，因为它们是数据）
        for artist in self.turn_path_artists:
            try:
                artist.remove()
            except:
                pass
        self.turn_path_artists.clear()
        self.turn_paths = []
        
        # 清除航渡路径
        for artist in self.transition_path_artists:
            try:
                artist.remove()
            except:
                pass
        self.transition_path_artists.clear()
        
        # 清除测线标注和起点标记
        for label in self.survey_line_labels:
            try:
                label.remove()
            except:
                pass
        self.survey_line_labels.clear()
        
        for artist in self.survey_line_start_artists:
            try:
                artist.remove()
            except:
                pass
        self.survey_line_start_artists.clear()
        
        self.optimizer = None
        self.deployment_path = None
        self.deployment_distance = None
        self.survey_line_order = None
        self.survey_line_transition_distance = None
        self.recovery_path = None
        self.recovery_distance = None
        self.deployment_to_survey_distance = None
        self.survey_to_recovery_distance = None
        self.time_results = None
        # 清除测线的进入和退出点
        if hasattr(self, 'survey_line_entry_points'):
            self.survey_line_entry_points.clear()
        if hasattr(self, 'survey_line_exit_points'):
            self.survey_line_exit_points.clear()
        self.plot_stations()
        self.log_result("路径已清除")
    
    def update_path_display(self):
        """根据用户选择更新路径显示"""
        # 清除航渡路径（需要重新绘制）
        for artist in self.transition_path_artists:
            try:
                artist.remove()
            except:
                pass
        self.transition_path_artists.clear()
        
        # 清除测线和标注（需要重新绘制）
        for artist in self.survey_line_artists:
            try:
                artist.remove()
            except:
                pass
        self.survey_line_artists.clear()
        
        for label in self.survey_line_labels:
            try:
                label.remove()
            except:
                pass
        self.survey_line_labels.clear()
        
        for artist in self.survey_line_start_artists:
            try:
                artist.remove()
            except:
                pass
        self.survey_line_start_artists.clear()
        
        # 重新绘制路径
        if self.deployment_path is not None or (self.optimizer is not None and self.optimizer.optimal_path is not None):
            self.plot_path()
        
        # 重新绘制测线和航渡路径
        if self.survey_lines:
            self.plot_survey_lines()
        
        # 根据用户选择绘制转弯路径
        if self.turn_paths:
            self.plot_turn_paths()
        
        self.canvas.draw()
    
    def clear_all(self):
        """清除所有"""
        self.clear_path()
        self.disable_click_selection()
        # 清除起点选择
        if hasattr(self, 'start_combo'):
            combo_values = self.start_combo['values']
            if combo_values and len(combo_values) > 0:
                self.start_combo.current(0)
            else:
                self.start_combo.set('')
        # 清除终点选择
        if hasattr(self, 'end_combo'):
            combo_values = self.end_combo['values']
            if combo_values and len(combo_values) > 0:
                self.end_combo.current(0)
            else:
                self.end_combo.set('')
        self.selected_start_idx = None
        self.selected_end_idx = None
        self.start_idx = None
        self.end_idx = None
        # 清除第一条测线选择
        if hasattr(self, 'first_survey_line_combo'):
            combo_values = self.first_survey_line_combo['values']
            if combo_values and len(combo_values) > 0:
                self.first_survey_line_combo.current(0)
            else:
                self.first_survey_line_combo.set('')
        self.selected_first_survey_line_idx = None
        self.plot_stations()
        self.log_result("已清除所有选择和路径")
    
    def save_path_figure(self):
        """保存路径图"""
        if self.fig is None:
            messagebox.showwarning('警告', '没有可保存的图形')
            return
        
        filename = filedialog.asksaveasfilename(
            title='保存路径图',
            filetypes=[
                ('PNG files', '*.png'),
                ('PDF files', '*.pdf'),
                ('PostScript files', '*.ps'),
                ('EPS files', '*.eps'),
                ('JPEG files', '*.jpg'),
                ('TIFF files', '*.tif'),
                ('All files', '*.*')
            ]
        )
        
        if filename:
            try:
                import os
                base_name, ext = os.path.splitext(filename)
                supported_extensions = ['.png', '.pdf', '.ps', '.eps', '.jpg', '.jpeg', '.tif', '.tiff']
                
                if not ext:
                    filename = filename + '.png'
                elif ext.lower() not in supported_extensions:
                    filename = base_name + '.png'
                
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo('成功', f'图形已保存到: {filename}')
                self.log_result(f"图形已保存: {filename}")
            except Exception as e:
                messagebox.showerror('错误', f'保存失败: {str(e)}')
    
    def export_results(self):
        """导出结果"""
        if self.optimizer is None or self.optimizer.optimal_path is None:
            messagebox.showwarning('警告', '没有可导出的结果')
            return
        
        filename = filedialog.asksaveasfilename(
            title='导出结果',
            defaultextension='.txt',
            filetypes=[
                ('文本文件', '*.txt'),
                ('CSV文件', '*.csv'),
                ('All files', '*.*')
            ]
        )
        
        if filename:
            try:
                import os
                ext = os.path.splitext(filename)[1].lower()
                
                if ext == '.csv':
                    # 导出CSV格式
                    import csv
                    with open(filename, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        # 如果有站位名称，添加名称列
                        if any(self.station_names):
                            writer.writerow(['顺序', '站位索引', '站位名称', '经度', '纬度'])
                            for i, idx in enumerate(self.optimizer.optimal_path):
                                lon, lat = self.stations[idx]
                                station_name = self.station_names[idx] if idx < len(self.station_names) else ''
                                writer.writerow([i+1, idx, station_name, lon, lat])
                        else:
                            writer.writerow(['顺序', '站位索引', '经度', '纬度'])
                            for i, idx in enumerate(self.optimizer.optimal_path):
                                lon, lat = self.stations[idx]
                                writer.writerow([i+1, idx, lon, lat])
                else:
                    # 导出文本格式
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write("站位布设路径优化结果\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(f"优化方法: {self.optimizer.method}\n")
                        f.write(f"总距离: {self.optimizer.total_distance:.2f} km\n")
                        if self.start_idx is not None:
                            start_name = self.station_names[self.start_idx] if self.start_idx < len(self.station_names) and self.station_names[self.start_idx] else f'站位 {self.start_idx}'
                            f.write(f"起点: {start_name} (索引 {self.start_idx})\n")
                        if self.end_idx is not None:
                            end_name = self.station_names[self.end_idx] if self.end_idx < len(self.station_names) and self.station_names[self.end_idx] else f'站位 {self.end_idx}'
                            f.write(f"终点: {end_name} (索引 {self.end_idx})\n")
                        f.write("\n路径顺序:\n")
                        for i, idx in enumerate(self.optimizer.optimal_path):
                            lon, lat = self.stations[idx]
                            station_name = self.station_names[idx] if idx < len(self.station_names) and self.station_names[idx] else f'站位 {idx}'
                            f.write(f"  {i+1}. {station_name} (索引 {idx}): ({lon:.4f}°, {lat:.4f}°)\n")
                
                messagebox.showinfo('成功', f'结果已导出到: {filename}')
                self.log_result(f"结果已导出: {filename}")
            except Exception as e:
                messagebox.showerror('错误', f'导出失败: {str(e)}')
    
    def save_sorted_stations(self):
        """保存排序后的站位坐标（包含投放路径和回收路径）"""
        # 检查是否有三阶段路径（投放和回收）
        has_deployment = self.deployment_path is not None
        has_recovery = self.recovery_path is not None
        # 检查是否有旧版本的路径
        has_old_path = self.optimizer is not None and self.optimizer.optimal_path is not None
        
        if not has_deployment and not has_recovery and not has_old_path:
            messagebox.showwarning('警告', '没有可保存的排序结果\n\n请先进行路径优化')
            return
        
        filename = filedialog.asksaveasfilename(
            title='保存排序后的站位坐标',
            defaultextension='.txt',
            filetypes=[
                ('文本文件', '*.txt'),
                ('CSV文件', '*.csv'),
                ('All files', '*.*')
            ]
        )
        
        if filename:
            try:
                import os
                ext = os.path.splitext(filename)[1].lower()
                
                if ext == '.csv':
                    # 导出CSV格式（包含所有列信息）
                    import csv
                    with open(filename, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        
                        # 检查是否有所有列数据
                        has_all_columns = (hasattr(self, 'station_all_columns') and 
                                         self.station_all_columns and 
                                         len(self.station_all_columns) > 0 and
                                         len(self.station_all_columns[0]) > 2)
                        
                        if has_deployment or has_recovery:
                            # 三阶段路径：分别保存投放和回收路径
                            if has_all_columns:
                                # 确定最大列数
                                max_cols = max(len(cols) for cols in self.station_all_columns)
                                # 写入表头
                                header = ['路径类型'] + [f'列{i+1}' for i in range(max_cols)]
                                writer.writerow(header)
                                
                                # 保存投放路径
                                if has_deployment:
                                    for idx in self.deployment_path:
                                        if idx < len(self.station_all_columns):
                                            row = ['投放'] + self.station_all_columns[idx]
                                            # 确保行长度与表头一致
                                            while len(row) < len(header):
                                                row.append('')
                                            writer.writerow(row)
                                
                                # 保存回收路径
                                if has_recovery:
                                    for idx in self.recovery_path:
                                        if idx < len(self.station_all_columns):
                                            row = ['回收'] + self.station_all_columns[idx]
                                            # 确保行长度与表头一致
                                            while len(row) < len(header):
                                                row.append('')
                                            writer.writerow(row)
                            else:
                                # 如果没有所有列数据，使用基本格式
                                has_names = any(self.station_names)
                                if has_names:
                                    writer.writerow(['路径类型', '经度', '纬度', '站位名称'])
                                else:
                                    writer.writerow(['路径类型', '经度', '纬度'])
                                
                                # 保存投放路径
                                if has_deployment:
                                    for idx in self.deployment_path:
                                        lon, lat = self.stations[idx]
                                        station_name = self.station_names[idx] if idx < len(self.station_names) else ''
                                        if has_names:
                                            writer.writerow(['投放', f"{lon:.6f}", f"{lat:.6f}", station_name])
                                        else:
                                            writer.writerow(['投放', f"{lon:.6f}", f"{lat:.6f}"])
                                
                                # 保存回收路径
                                if has_recovery:
                                    for idx in self.recovery_path:
                                        lon, lat = self.stations[idx]
                                        station_name = self.station_names[idx] if idx < len(self.station_names) else ''
                                        if has_names:
                                            writer.writerow(['回收', f"{lon:.6f}", f"{lat:.6f}", station_name])
                                        else:
                                            writer.writerow(['回收', f"{lon:.6f}", f"{lat:.6f}"])
                        else:
                            # 旧版本路径
                            if has_all_columns:
                                # 确定最大列数
                                max_cols = max(len(cols) for cols in self.station_all_columns)
                                # 写入表头
                                header = [f'列{i+1}' for i in range(max_cols)]
                                writer.writerow(header)
                                
                                for idx in self.optimizer.optimal_path:
                                    if idx < len(self.station_all_columns):
                                        row = self.station_all_columns[idx].copy()
                                        # 确保行长度与表头一致
                                        while len(row) < len(header):
                                            row.append('')
                                        writer.writerow(row)
                            else:
                                # 如果没有所有列数据，使用基本格式
                                has_names = any(self.station_names)
                                if has_names:
                                    writer.writerow(['经度', '纬度', '站位名称'])
                                    for idx in self.optimizer.optimal_path:
                                        lon, lat = self.stations[idx]
                                        station_name = self.station_names[idx] if idx < len(self.station_names) else ''
                                        writer.writerow([f"{lon:.6f}", f"{lat:.6f}", station_name])
                                else:
                                    writer.writerow(['经度', '纬度'])
                                    for idx in self.optimizer.optimal_path:
                                        lon, lat = self.stations[idx]
                                        writer.writerow([f"{lon:.6f}", f"{lat:.6f}"])
                else:
                    # 导出文本格式（空格分隔，包含所有列）
                    with open(filename, 'w', encoding='utf-8') as f:
                        # 检查是否有所有列数据
                        has_all_columns = (hasattr(self, 'station_all_columns') and 
                                         self.station_all_columns and 
                                         len(self.station_all_columns) > 0)
                        
                        if has_deployment or has_recovery:
                            # 三阶段路径：分别保存投放和回收路径
                            # 保存投放路径
                            if has_deployment:
                                f.write("# 投放路径\n")
                                for idx in self.deployment_path:
                                    if has_all_columns and idx < len(self.station_all_columns):
                                        # 写入所有列数据（空格分隔）
                                        row_data = ' '.join(str(col) for col in self.station_all_columns[idx])
                                        f.write(f"{row_data}\n")
                                    else:
                                        # 基本格式
                                        lon, lat = self.stations[idx]
                                        if idx < len(self.station_names) and self.station_names[idx]:
                                            beams = self.station_beams[idx] if idx < len(self.station_beams) else ''
                                            f.write(f"{lon:.6f} {lat:.6f} {self.station_names[idx]:13s} {beams:.2f}\n")
                                        else:
                                            f.write(f"{lon:.6f} {lat:.6f}\n")
                            
                            # 保存回收路径
                            if has_recovery:
                                f.write("\n# 回收路径\n")
                                for idx in self.recovery_path:
                                    if has_all_columns and idx < len(self.station_all_columns):
                                        # 写入所有列数据（空格分隔）
                                        row_data = ' '.join(str(col) for col in self.station_all_columns[idx])
                                        f.write(f"{row_data}\n")
                                    else:
                                        # 基本格式
                                        lon, lat = self.stations[idx]
                                        if idx < len(self.station_names) and self.station_names[idx]:
                                            beams = self.station_beams[idx] if idx < len(self.station_beams) else ''
                                            f.write(f"{lon:.6f} {lat:.6f} {self.station_names[idx]} {beams:.2f}\n")
                                        else:
                                            f.write(f"{lon:.6f} {lat:.6f}\n")
                        else:
                            # 旧版本路径
                            for idx in self.optimizer.optimal_path:
                                if has_all_columns and idx < len(self.station_all_columns):
                                    # 写入所有列数据（空格分隔）
                                    row_data = ' '.join(str(col) for col in self.station_all_columns[idx])
                                    f.write(f"{row_data}\n")
                                else:
                                    # 基本格式
                                    lon, lat = self.stations[idx]
                                    if idx < len(self.station_names) and self.station_names[idx]:
                                        f.write(f"{lon:.6f} {lat:.6f} {self.station_names[idx]}\n")
                                    else:
                                        f.write(f"{lon:.6f} {lat:.6f}\n")
                
                # 统计保存的站位数量
                total_count = 0
                if has_deployment:
                    total_count += len(self.deployment_path)
                if has_recovery:
                    total_count += len(self.recovery_path)
                if not has_deployment and not has_recovery and has_old_path:
                    total_count = len(self.optimizer.optimal_path)
                
                message_text = f'排序后的站位坐标已保存到: {filename}\n\n'
                if has_deployment:
                    message_text += f'投放路径: {len(self.deployment_path)} 个站位\n'
                if has_recovery:
                    message_text += f'回收路径: {len(self.recovery_path)} 个站位\n'
                if has_deployment or has_recovery:
                    message_text += f'总计: {total_count} 个站位'
                else:
                    message_text += f'共保存 {total_count} 个站位'
                
                messagebox.showinfo('成功', message_text)
                self.log_result(f"排序后的站位坐标已保存: {filename}")
                if has_deployment:
                    self.log_result(f"投放路径: {len(self.deployment_path)} 个站位")
                if has_recovery:
                    self.log_result(f"回收路径: {len(self.recovery_path)} 个站位")
                self.log_result(f"总计: {total_count} 个站位")
            except Exception as e:
                messagebox.showerror('错误', f'保存失败: {str(e)}')
                import traceback
                traceback.print_exc()
    
    def calculate_time(self):
        """计算航时"""
        # 检查必要的数据
        if not self.stations:
            messagebox.showwarning('警告', '请先加载站位文件')
            return
        
        # 检查是否进行了三阶段优化
        if (self.deployment_path is None or self.survey_line_order is None or 
            self.recovery_path is None):
            messagebox.showwarning('警告', '请先进行路径优化（需要三阶段优化）')
            return
        
        try:
            # 获取参数
            cruise_speed = float(self.cruise_speed_var.get())  # 节
            working_speed = float(self.working_speed_var.get())  # 节
            deployment_time = float(self.deployment_time_var.get())  # 秒
            ascent_speed = float(self.ascent_speed_var.get())  # 米/秒
            turn_radius_km = float(self.turn_radius_var.get())  # 转弯半径（公里）
            
            # 转换速度单位：节 -> 米/秒 (1 knot = 0.514444 m/s)
            cruise_speed_ms = cruise_speed * 0.514444
            working_speed_ms = working_speed * 0.514444
            
            # 1. 计算投放时间（第一阶段：投放路径）
            deployment_times = []
            deployment_details = []  # 保存各段的详细信息
            path = self.deployment_path
            
            for i in range(len(path) - 1):
                idx1, idx2 = path[i], path[i + 1]
                lon1, lat1 = self.stations[idx1]
                lon2, lat2 = self.stations[idx2]
                
                # 计算站间距（公里）
                distance_km = StationPathOptimizer.haversine_distance(lon1, lat1, lon2, lat2)
                distance_m = distance_km * 1000
                
                # 走航时间（秒）
                cruise_time = distance_m / cruise_speed_ms
                
                # 总投放时间 = 走航时间 + 投放所需时间
                total_deployment_time = cruise_time + deployment_time
                deployment_times.append(total_deployment_time)
                
                # 保存详细信息
                from_name = self.station_names[idx1] if idx1 < len(self.station_names) and self.station_names[idx1] else f'站位 {idx1}'
                to_name = self.station_names[idx2] if idx2 < len(self.station_names) and self.station_names[idx2] else f'站位 {idx2}'
                deployment_details.append({
                    'segment': i + 1,
                    'from_station': idx1,
                    'to_station': idx2,
                    'from_station_name': from_name,
                    'to_station_name': to_name,
                    'distance_km': distance_km,
                    'cruise_speed_knots': cruise_speed,
                    'cruise_time': cruise_time,
                    'deployment_time': deployment_time,
                    'total_time': total_deployment_time
                })
            
            total_deployment_time = sum(deployment_times)
            
            # 1.5. 计算从最后一个投放站位到第一条测线的走航时间
            last_deployment_idx = self.deployment_path[-1]
            last_deployment_lon, last_deployment_lat = self.stations[last_deployment_idx]
            deployment_to_survey_time = self.deployment_to_survey_distance * 1000 / cruise_speed_ms  # 秒
            
            # 2. 计算走航调查时间（第二阶段：测线路径）
            # 计算每条测线的长度（按优化后的顺序）
            survey_line_lengths = []
            for line_idx in self.survey_line_order:
                line = self.survey_lines[line_idx]
                line_length = 0.0
                for i in range(len(line) - 1):
                    lon1, lat1 = line[i]
                    lon2, lat2 = line[i + 1]
                    dist = StationPathOptimizer.haversine_distance(lon1, lat1, lon2, lat2)
                    line_length += dist
                survey_line_lengths.append(line_length)
            
            total_survey_length = sum(survey_line_lengths)  # 总测线长度（公里）
            
            # 计算换线过程中的转弯路径（按优化后的顺序）
            # 换线次数 = 测线数 - 1
            n_turns = len(self.survey_line_order) - 1
            
            # 计算转弯路径坐标（使用转弯半径，单位：公里）
            # 需要按优化后的顺序计算转弯路径，使用保存的退出点和进入点
            self.turn_paths = []
            for i in range(len(self.survey_line_order) - 1):
                line1_idx = self.survey_line_order[i]
                line2_idx = self.survey_line_order[i + 1]
                
                # 使用保存的退出点和进入点
                line1 = self.survey_lines[line1_idx]
                line2 = self.survey_lines[line2_idx]
                
                if len(line1) < 2 or len(line2) < 2:
                    continue
                
                if (hasattr(self, 'survey_line_exit_points') and line1_idx in self.survey_line_exit_points and
                    hasattr(self, 'survey_line_entry_points') and line2_idx in self.survey_line_entry_points):
                    exit_point = self.survey_line_exit_points[line1_idx]
                    entry_point = self.survey_line_entry_points[line2_idx]
                    self.log_result(f"[转弯路径] 测线 {line1_idx} -> {line2_idx}: 使用保存的退出点和进入点")
                else:
                    # 如果没有保存的点，使用默认值（向后兼容）
                    exit_point = line1[-1]  # 第一条测线的终点
                    entry_point = line2[0]  # 第二条测线的起点
                    self.log_result(f"[转弯路径] 警告：未找到保存的点，使用默认值")
                
                # 使用完整的过渡路径：测线1退出点 → 转弯路径（圆弧）→ 航渡路径起点 → 
                # 航渡路径（直线）→ 航渡路径终点 → 转弯路径（圆弧）→ 测线2进入点
                turn_path = self._calculate_transition_path(
                    exit_point, entry_point, line1, line2, turn_radius_km
                )
                self.turn_paths.append(turn_path)
            
            # 计算实际转弯路径长度
            total_turn_length = 0.0
            for turn_path in self.turn_paths:
                for i in range(len(turn_path) - 1):
                    lon1, lat1 = turn_path[i]
                    lon2, lat2 = turn_path[i + 1]
                    dist = StationPathOptimizer.haversine_distance(lon1, lat1, lon2, lat2)
                    total_turn_length += dist * 1000  # 转换为米
            
            # 测线之间的过渡走航时间（按优化后的顺序）
            survey_transition_time = self.survey_line_transition_distance * 1000 / cruise_speed_ms  # 秒
            
            # 计算每条测线的作业时间
            survey_line_times = []
            for line_length in survey_line_lengths:
                line_time = (line_length * 1000) / working_speed_ms  # 秒
                survey_line_times.append(line_time)
            
            # 计算转弯时间（转弯路径长度 / 作业速度）
            turn_time = total_turn_length / working_speed_ms  # 秒
            
            # 作业时间 = (测线长度 + 转弯路径) / 作业速度
            total_survey_time = (total_survey_length * 1000 + total_turn_length) / working_speed_ms  # 秒
            
            # 从最后一条测线到回收起点的走航时间
            survey_to_recovery_time = self.survey_to_recovery_distance * 1000 / cruise_speed_ms  # 秒
            
            # 3. 计算仪器回收时间
            # 优先使用文件第6列的深度，否则从水深文件读取
            if not self.station_depths:
                # 检查是否有从文件读取的深度
                if hasattr(self, 'station_depths_from_file') and any(
                    d is not None for d in self.station_depths_from_file
                ):
                    self.station_depths = [d if d is not None else 0.0 
                                          for d in self.station_depths_from_file]
                    self.log_result("使用站位文件第6列的深度数据")
                # 如果没有水深数据，尝试从水深文件插值
                if self.bathymetry_file_type == 'grd' and self.bathymetry_filename:
                    # 对于.grd文件，按需读取站位周围的小区域
                    try:
                        import pygmt
                        self.station_depths = []
                        for lon, lat in self.stations:
                            # 读取站位周围的小区域（约0.1度，约11公里）
                            region = f"{lon - 0.05}/{lon + 0.05}/{lat - 0.05}/{lat + 0.05}"
                            try:
                                grid = pygmt.grdcut(self.bathymetry_filename, region=region)
                                # 使用最近邻方法获取站位水深
                                depth = float(grid.sel(lon=lon, lat=lat, method='nearest').values)
                                self.station_depths.append(abs(depth) if not np.isnan(depth) else float(self.default_depth_var.get()))
                            except:
                                # 如果读取失败，使用用户选择的默认水深
                                self.station_depths.append(float(self.default_depth_var.get()))
                    except ImportError:
                        messagebox.showwarning('警告', '需要pygmt库来读取.grd文件\n\n请运行: pip install pygmt')
                        return
                    except Exception as e:
                        messagebox.showwarning('警告', f'读取水深数据失败: {str(e)}')
                        return
                elif self.bathymetry_file_type == 'nc' and self.bathymetry_filename:
                    # 对于.nc文件，按需读取站位周围的小区域
                    try:
                        import xarray as xr
                        self.station_depths = []
                        lon_name = self.bathymetry_lon_name
                        lat_name = self.bathymetry_lat_name
                        var_name = self.bathymetry_var_name
                        
                        with xr.open_dataset(self.bathymetry_filename, decode_times=False) as ds:
                            for lon, lat in self.stations:
                                try:
                                    # 使用sel方法直接选择最近的坐标值（不使用slice）
                                    # 这样可以避免method参数与slice的冲突
                                    depth = float(ds[var_name].sel({lon_name: lon, lat_name: lat}, method='nearest').values)
                                    self.station_depths.append(abs(depth) if not np.isnan(depth) else float(self.default_depth_var.get()))
                                except:
                                    # 如果读取失败，使用用户选择的默认水深
                                    self.station_depths.append(float(self.default_depth_var.get()))
                    except ImportError:
                        messagebox.showwarning('警告', '需要xarray库来读取.nc文件\n\n请运行: pip install xarray')
                        return
                    except Exception as e:
                        messagebox.showwarning('警告', f'读取水深数据失败: {str(e)}')
                        import traceback
                        traceback.print_exc()
                        return
                elif self.bathymetry_data is not None:
                    self._interpolate_station_depths()
                else:
                    # 如果没有加载水深文件，使用用户选择的默认水深
                    default_depth = float(self.default_depth_var.get())
                    self.station_depths = [default_depth] * len(self.stations)
                    self.log_result(f"未加载水深文件，使用默认水深: {default_depth} 米")
                    messagebox.showinfo('提示', f'未加载水深文件\n\n使用默认水深: {default_depth} 米\n\n所有站位将使用此水深值')
            
            # 3. 计算仪器回收时间（第三阶段：回收路径）
            recovery_times = []
            recovery_details = []  # 保存各段的详细信息
            recovery_path = self.recovery_path
            
            for i, idx in enumerate(recovery_path):
                if idx < len(self.station_depths):
                    depth = self.station_depths[idx]  # 米
                else:
                    depth = 0.0
                
                # 上浮时间（秒）
                ascent_time = depth / ascent_speed if ascent_speed > 0 else 0
                
                # 获取站位名称
                station_name = self.station_names[idx] if idx < len(self.station_names) and self.station_names[idx] else f'站位 {idx}'
                
                # 回收时间 = 上浮时间 + 走航到下一个站位的时间（如果是最后一个站位，则不需要走航）
                if i == len(recovery_path) - 1:
                    # 最后一个站位（起点），只需要上浮时间
                    recovery_time = ascent_time
                    recovery_details.append({
                        'segment': i + 1,
                        'station': idx,
                        'station_name': station_name,
                        'from_station': idx,
                        'to_station': None,  # 最后一个站位，没有下一个
                        'depth': depth,
                        'ascent_time': ascent_time,
                        'cruise_time': 0.0,
                        'distance_km': 0.0,
                        'cruise_speed_knots': cruise_speed,
                        'total_time': recovery_time
                    })
                else:
                    # 需要走航到下一个站位
                    next_idx = recovery_path[i + 1]
                    next_station_name = self.station_names[next_idx] if next_idx < len(self.station_names) and self.station_names[next_idx] else f'站位 {next_idx}'
                    lon1, lat1 = self.stations[idx]
                    lon2, lat2 = self.stations[next_idx]
                    distance_km = StationPathOptimizer.haversine_distance(lon1, lat1, lon2, lat2)
                    distance_m = distance_km * 1000
                    cruise_time = distance_m / cruise_speed_ms
                    recovery_time = ascent_time + cruise_time
                    recovery_details.append({
                        'segment': i + 1,
                        'station': idx,
                        'station_name': station_name,
                        'from_station': idx,
                        'to_station': next_idx,
                        'depth': depth,
                        'ascent_time': ascent_time,
                        'cruise_time': cruise_time,
                        'distance_km': distance_km,
                        'cruise_speed_knots': cruise_speed,
                        'total_time': recovery_time
                    })
                
                recovery_times.append(recovery_time)
            
            total_recovery_time = sum(recovery_times)
            
            # 4. 计算总体用时
            total_time = (total_deployment_time + 
                         deployment_to_survey_time +
                         total_survey_time + 
                         survey_transition_time +
                         survey_to_recovery_time +
                         total_recovery_time)
            
            # 保存结果
            self.time_results = {
                'deployment_time': total_deployment_time,
                'deployment_to_survey_time': deployment_to_survey_time,
                'survey_time': total_survey_time,
                'survey_transition_time': survey_transition_time,
                'survey_to_recovery_time': survey_to_recovery_time,
                'recovery_time': total_recovery_time,
                'total_time': total_time,
                'deployment_times': deployment_times,
                'deployment_details': deployment_details,  # 各段详细信息
                'recovery_times': recovery_times,
                'recovery_details': recovery_details,  # 回收各段详细信息
                'survey_line_lengths': survey_line_lengths,
                'survey_line_times': survey_line_times,  # 每条测线的作业时间
                'survey_line_order': self.survey_line_order,  # 测线顺序
                'total_survey_length': total_survey_length,
                'n_turns': n_turns,
                'total_turn_length': total_turn_length,
                'turn_time': turn_time,  # 转弯时间
                'working_speed': working_speed,  # 作业速度
                'cruise_speed': cruise_speed,  # 保存船速，用于显示
                'deployment_time_param': deployment_time  # 保存投放时间参数
            }
            
            # 显示结果
            self.log_result("\n" + "="*60)
            self.log_result("航时计算结果（三阶段）")
            self.log_result("="*60)
            self.log_result(f"\n【第一阶段】投放时间:")
            self.log_result(f"   总投放时间: {total_deployment_time/3600:.2f} 小时 ({total_deployment_time:.0f} 秒)")
            self.log_result(f"   站位数: {len(self.deployment_path)}")
            self.log_result(f"   平均每站投放时间: {total_deployment_time/len(self.deployment_path)/60:.2f} 分钟")
            self.log_result(f"   到测线走航时间: {deployment_to_survey_time/3600:.2f} 小时 ({deployment_to_survey_time:.0f} 秒)")
            
            self.log_result(f"\n【第二阶段】走航调查时间:")
            self.log_result(f"   总测线长度: {total_survey_length:.2f} km")
            self.log_result(f"   测线数量: {len(self.survey_line_order)}")
            self.log_result(f"   测线顺序: {self.survey_line_order}")
            self.log_result(f"   换线次数: {n_turns}")
            self.log_result(f"   转弯半径: {turn_radius_km:.1f} km")
            self.log_result(f"   转弯路径总长度: {total_turn_length/1000:.2f} km")
            self.log_result(f"   测线过渡走航时间: {survey_transition_time/3600:.2f} 小时 ({survey_transition_time:.0f} 秒)")
            self.log_result(f"   总调查时间: {total_survey_time/3600:.2f} 小时 ({total_survey_time:.0f} 秒)")
            self.log_result(f"   到回收走航时间: {survey_to_recovery_time/3600:.2f} 小时 ({survey_to_recovery_time:.0f} 秒)")
            
            self.log_result(f"\n【第三阶段】仪器回收时间:")
            self.log_result(f"   总回收时间: {total_recovery_time/3600:.2f} 小时 ({total_recovery_time:.0f} 秒)")
            avg_depth = np.mean(self.station_depths) if self.station_depths else 0
            self.log_result(f"   平均水深: {avg_depth:.1f} 米")
            self.log_result(f"   站位数: {len(self.recovery_path)}")
            self.log_result(f"   平均每站回收时间: {total_recovery_time/len(self.recovery_path)/60:.2f} 分钟")
            
            self.log_result(f"\n【总体用时】:")
            self.log_result(f"   总时间: {total_time/3600:.2f} 小时 ({total_time:.0f} 秒)")
            self.log_result(f"   总时间: {total_time/86400:.2f} 天")
            self.log_result(f"   详细分解:")
            self.log_result(f"     - 投放路径: {total_deployment_time/3600:.2f} 小时")
            self.log_result(f"     - 到测线: {deployment_to_survey_time/3600:.2f} 小时")
            self.log_result(f"     - 测线作业: {total_survey_time/3600:.2f} 小时")
            self.log_result(f"     - 测线过渡: {survey_transition_time/3600:.2f} 小时")
            self.log_result(f"     - 到回收: {survey_to_recovery_time/3600:.2f} 小时")
            self.log_result(f"     - 回收路径: {total_recovery_time/3600:.2f} 小时")
            
            self.log_result("\n" + "="*60)
            
            # 绘制作业路径和转弯路径（转弯路径根据用户选择显示）
            self.plot_survey_lines()
            self.plot_turn_paths()  # 会根据show_turn_paths_var决定是否绘制
            self.canvas.draw()
            
            messagebox.showinfo('完成', 
                              f'航时计算完成!\n\n'
                              f'总时间: {total_time/3600:.2f} 小时\n'
                              f'总时间: {total_time/86400:.2f} 天')
            
        except ValueError as e:
            messagebox.showerror('错误', f'参数格式错误: {str(e)}\n\n请检查输入参数是否为有效数字')
        except Exception as e:
            messagebox.showerror('错误', f'计算失败: {str(e)}')
            import traceback
            traceback.print_exc()
            self.log_result(f"\n错误: {str(e)}")
    
    def log_result(self, message: str):
        """在结果文本框中显示消息"""
        self.result_text.insert(tk.END, message + '\n')
        self.result_text.see(tk.END)
        self.root.update_idletasks()
    
    def save_time_results(self):
        """保存航时计算结果"""
        if self.time_results is None:
            messagebox.showwarning('警告', '没有可保存的航时计算结果\n\n请先进行航时计算')
            return
        
        filename = filedialog.asksaveasfilename(
            title='保存航时计算结果',
            defaultextension='.txt',
            filetypes=[
                ('文本文件', '*.txt'),
                ('CSV文件', '*.csv'),
                ('所有文件', '*.*')
            ]
        )
        
        if not filename:
            return
        
        try:
            import os
            ext = os.path.splitext(filename)[1].lower()
            
            # 获取计算结果
            results = self.time_results
            # 使用三阶段路径（如果有）
            if self.deployment_path is not None:
                deployment_path = self.deployment_path
                recovery_path = self.recovery_path if self.recovery_path else []
            else:
                deployment_path = self.optimizer.optimal_path if self.optimizer else []
                recovery_path = []
            
            if ext == '.csv':
                # 导出CSV格式
                import csv
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['航时计算结果'])
                    writer.writerow([])
                    
                    # 总体信息
                    writer.writerow(['项目', '数值', '单位'])
                    writer.writerow(['总投放时间', f"{results['deployment_time']/3600:.2f}", '小时'])
                    writer.writerow(['总调查时间', f"{results['survey_time']/3600:.2f}", '小时'])
                    writer.writerow(['总回收时间', f"{results['recovery_time']/3600:.2f}", '小时'])
                    writer.writerow(['总时间', f"{results['total_time']/3600:.2f}", '小时'])
                    writer.writerow(['总时间', f"{results['total_time']/86400:.2f}", '天'])
                    writer.writerow([])
                    
                    # 详细参数
                    writer.writerow(['参数设置'])
                    writer.writerow(['走航速度', self.cruise_speed_var.get(), 'knots'])
                    writer.writerow(['作业速度', self.working_speed_var.get(), 'knots'])
                    writer.writerow(['投放时间', self.deployment_time_var.get(), '秒'])
                    writer.writerow(['上浮速度', self.ascent_speed_var.get(), 'm/s'])
                    writer.writerow(['转弯半径', self.turn_radius_var.get(), 'km'])
                    writer.writerow([])
                    
                    # 投放时间详情
                    writer.writerow(['投放时间详情'])
                    writer.writerow(['段', '起点站', '终点站', '距离(km)', '船速(knots)', '走航时间(秒)', '走航时间(分钟)', '投放时间(秒)', '总时间(秒)', '总时间(分钟)'])
                    if 'deployment_details' in results and results['deployment_details']:
                        for detail in results['deployment_details']:
                            from_name = detail.get('from_station_name', self.station_names[detail['from_station']] if detail['from_station'] < len(self.station_names) and self.station_names[detail['from_station']] else f'站位 {detail["from_station"]}')
                            to_name = detail.get('to_station_name', self.station_names[detail['to_station']] if detail['to_station'] < len(self.station_names) and self.station_names[detail['to_station']] else f'站位 {detail["to_station"]}')
                            writer.writerow([
                                detail['segment'],
                                from_name,
                                to_name,
                                f"{detail['distance_km']:.2f}",
                                f"{detail['cruise_speed_knots']:.2f}",
                                f"{detail['cruise_time']:.2f}",
                                f"{detail['cruise_time']/60:.2f}",
                                f"{detail['deployment_time']:.2f}",
                                f"{detail['total_time']:.2f}",
                                f"{detail['total_time']/60:.2f}"
                            ])
                    writer.writerow([])
                    
                    # 回收时间详情
                    writer.writerow(['回收时间详情'])
                    writer.writerow(['段', '站位', '下一站', '距离(km)', '船速(knots)', '走航时间(秒)', '走航时间(分钟)', '水深(米)', '上浮时间(秒)', '总时间(秒)', '总时间(分钟)'])
                    if 'recovery_details' in results and results['recovery_details']:
                        for detail in results['recovery_details']:
                            station_name = detail.get('station_name', f'站位 {detail["from_station"]}')
                            if detail['to_station'] is not None:
                                to_name = self.station_names[detail['to_station']] if detail['to_station'] < len(self.station_names) and self.station_names[detail['to_station']] else f'站位 {detail["to_station"]}'
                                writer.writerow([
                                    detail['segment'],
                                    station_name,
                                    to_name,
                                    f"{detail['distance_km']:.2f}",
                                    f"{detail['cruise_speed_knots']:.2f}",
                                    f"{detail['cruise_time']:.2f}",
                                    f"{detail['cruise_time']/60:.2f}",
                                    f"{detail['depth']:.1f}",
                                    f"{detail['ascent_time']:.2f}",
                                    f"{detail['total_time']:.2f}",
                                    f"{detail['total_time']/60:.2f}"
                                ])
                            else:
                                writer.writerow([
                                    detail['segment'],
                                    station_name,
                                    '终点',
                                    '0.00',
                                    f"{detail['cruise_speed_knots']:.2f}",
                                    '0.00',
                                    '0.00',
                                    f"{detail['depth']:.1f}",
                                    f"{detail['ascent_time']:.2f}",
                                    f"{detail['total_time']:.2f}",
                                    f"{detail['total_time']/60:.2f}"
                                ])
                    writer.writerow([])
                    
                    # 测线信息
                    writer.writerow(['测线信息'])
                    writer.writerow(['测线数量', len(self.survey_lines), '条'])
                    writer.writerow(['总测线长度', f"{results['total_survey_length']:.2f}", 'km'])
                    writer.writerow(['转弯路径长度', f"{results['total_turn_length']/1000:.2f}", 'km'])
                    writer.writerow(['换线次数', results['n_turns'], '次'])
            else:
                # 导出文本格式
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("航时计算结果\n")
                    f.write("=" * 60 + "\n\n")
                    
                    # 总体信息
                    f.write("总体用时:\n")
                    f.write(f"  总投放时间: {results['deployment_time']/3600:.2f} 小时 ({results['deployment_time']:.0f} 秒)\n")
                    f.write(f"  总调查时间: {results['survey_time']/3600:.2f} 小时 ({results['survey_time']:.0f} 秒)\n")
                    f.write(f"  总回收时间: {results['recovery_time']/3600:.2f} 小时 ({results['recovery_time']:.0f} 秒)\n")
                    f.write(f"  总时间: {results['total_time']/3600:.2f} 小时 ({results['total_time']:.0f} 秒)\n")
                    f.write(f"  总时间: {results['total_time']/86400:.2f} 天\n\n")
                    
                    # 详细参数
                    f.write("参数设置:\n")
                    f.write(f"  走航速度: {self.cruise_speed_var.get()} knots\n")
                    f.write(f"  作业速度: {self.working_speed_var.get()} knots\n")
                    f.write(f"  投放时间: {self.deployment_time_var.get()} 秒\n")
                    f.write(f"  上浮速度: {self.ascent_speed_var.get()} m/s\n")
                    f.write(f"  转弯半径: {self.turn_radius_var.get()} km\n\n")
                    
                    # 投放时间详情
                    f.write("1. 投放时间详情:\n")
                    deployment_path = self.deployment_path if self.deployment_path else []
                    f.write(f"  站位数: {len(deployment_path)}\n")
                    if len(deployment_path) > 0:
                        f.write(f"  平均每站投放时间: {results['deployment_time']/len(deployment_path)/60:.2f} 分钟\n")
                    if 'deployment_details' in results and results['deployment_details']:
                        f.write("  各段投放时间详情:\n")
                        for detail in results['deployment_details']:
                            from_name = detail.get('from_station_name', self.station_names[detail['from_station']] if detail['from_station'] < len(self.station_names) and self.station_names[detail['from_station']] else f'站位 {detail["from_station"]}')
                            to_name = detail.get('to_station_name', self.station_names[detail['to_station']] if detail['to_station'] < len(self.station_names) and self.station_names[detail['to_station']] else f'站位 {detail["to_station"]}')
                            f.write(f"    段 {detail['segment']} ({from_name} -> {to_name}):  距离: {detail['distance_km']:.2f} km  总时间: {detail['total_time']/60:.2f} 分钟 ({detail['cruise_time']:.2f} + {detail['deployment_time']:.2f} 秒)\n")
                    elif results['deployment_times']:
                        f.write("  各段投放时间:\n")
                        for i, time in enumerate(results['deployment_times']):
                            f.write(f"    段 {i+1}: {time/60:.2f} 分钟 ({time:.0f} 秒)\n")
                    f.write("\n")
                    
                    # 调查时间详情
                    f.write("2. 走航调查时间详情:\n")
                    f.write(f"  测线数量: {len(self.survey_lines)}\n")
                    f.write(f"  总测线长度: {results['total_survey_length']:.2f} km\n")
                    f.write(f"  换线次数: {results['n_turns']}\n")
                    f.write(f"  转弯半径: {self.turn_radius_var.get()} km\n")
                    f.write(f"  转弯路径总长度: {results['total_turn_length']/1000:.2f} km\n")
                    f.write(f"  作业速度: {results.get('working_speed', float(self.working_speed_var.get())):.2f} knots\n")
                    f.write(f"  走航速度: {results.get('cruise_speed', float(self.cruise_speed_var.get())):.2f} knots\n\n")
                    
                    # 详细时间分解
                    f.write("  时间分解:\n")
                    f.write(f"    到测线走航时间: {results['deployment_to_survey_time']/3600:.2f} 小时 ({results['deployment_to_survey_time']:.0f} 秒)\n")
                    
                    # 各测线作业时间
                    if results.get('survey_line_times') and results.get('survey_line_order'):
                        f.write("    各测线作业时间:\n")
                        for i, (line_idx, line_time) in enumerate(zip(results['survey_line_order'], results['survey_line_times'])):
                            line_length = results['survey_line_lengths'][i]
                            f.write(f"      测线 {i+1} (原索引 {line_idx}): 长度 {line_length:.2f} km, 时间 {line_time/3600:.2f} 小时 ({line_time:.0f} 秒)\n")
                    
                    # 转弯时间
                    if results.get('turn_time') is not None:
                        f.write(f"    转弯时间: {results['turn_time']/3600:.2f} 小时 ({results['turn_time']:.0f} 秒)\n")
                    
                    # 航渡时间（测线之间的走航时间）
                    f.write(f"    测线间航渡走航时间: {results['survey_transition_time']/3600:.2f} 小时 ({results['survey_transition_time']:.0f} 秒)\n")
                    
                    # 到回收走航时间
                    f.write(f"    到回收走航时间: {results['survey_to_recovery_time']/3600:.2f} 小时 ({results['survey_to_recovery_time']:.0f} 秒)\n")
                    
                    f.write(f"    总调查时间: {results['survey_time']/3600:.2f} 小时 ({results['survey_time']:.0f} 秒)\n")
                    f.write("\n")
                    
                    # 回收时间详情
                    f.write("3. 仪器回收时间详情:\n")
                    avg_depth = np.mean(self.station_depths) if self.station_depths else 0
                    f.write(f"  平均水深: {avg_depth:.1f} 米\n")
                    if recovery_path:
                        f.write(f"  站位数: {len(recovery_path)}\n")
                        f.write(f"  平均每站回收时间: {results['recovery_time']/len(recovery_path)/60:.2f} 分钟\n")
                    if 'recovery_details' in results and results['recovery_details']:
                        f.write("  各段回收时间详情:\n")
                        for detail in results['recovery_details']:
                            from_name = detail.get('station_name', f'站位 {detail["from_station"]}')
                            if detail['to_station'] is not None:
                                to_name = self.station_names[detail['to_station']] if detail['to_station'] < len(self.station_names) and self.station_names[detail['to_station']] else f'站位 {detail["to_station"]}'
                                f.write(f"    段 {detail['segment']} ({from_name} -> {to_name}):  距离: {detail['distance_km']:.2f} km  水深: {detail['depth']:.1f}米  总时间: {detail['total_time']/60:.2f} 分钟 ({detail['cruise_time']:.2f} + {detail['ascent_time']:.2f} 秒)\n")
                            else:
                                f.write(f"    段 {detail['segment']} ({from_name}):  水深: {detail['depth']:.1f}米  总时间: {detail['total_time']/60:.2f} 分钟 ({detail['ascent_time']:.2f} 秒)\n")
                    elif results['recovery_times'] and recovery_path:
                        f.write("  各站回收时间:\n")
                        for i, idx in enumerate(recovery_path):
                            depth = self.station_depths[idx] if idx < len(self.station_depths) else 0.0
                            recovery_time = results['recovery_times'][i]
                            lon, lat = self.stations[idx]
                            station_name = self.station_names[idx] if idx < len(self.station_names) and self.station_names[idx] else f'站位 {idx}'
                            f.write(f"    站 {i+1} ({station_name}): {recovery_time/60:.2f} 分钟 (水深: {depth:.1f} 米, 坐标: {lon:.4f}°, {lat:.4f}°)\n")
                    f.write("\n")
                    
                    # 路径信息
                    if deployment_path:
                        f.write("投放路径信息:\n")
                        f.write(f"  路径顺序: {deployment_path}\n")
                        f.write("  详细路径:\n")
                        for i, idx in enumerate(deployment_path):
                            lon, lat = self.stations[idx]
                            station_name = self.station_names[idx] if idx < len(self.station_names) and self.station_names[idx] else f'站位 {idx}'
                            f.write(f"    {i+1}. {station_name} (索引 {idx}): ({lon:.4f}°, {lat:.4f}°)\n")
                    if recovery_path:
                        f.write("\n回收路径信息:\n")
                        f.write(f"  路径顺序: {recovery_path}\n")
                        f.write("  详细路径:\n")
                        for i, idx in enumerate(recovery_path):
                            lon, lat = self.stations[idx]
                            station_name = self.station_names[idx] if idx < len(self.station_names) and self.station_names[idx] else f'站位 {idx}'
                            f.write(f"    {i+1}. {station_name} (索引 {idx}): ({lon:.4f}°, {lat:.4f}°)\n")
                    if self.deployment_distance is not None and self.recovery_distance is not None:
                        total_distance = (self.deployment_distance + 
                                        self.deployment_to_survey_distance +
                                        self.survey_line_transition_distance +
                                        self.survey_to_recovery_distance +
                                        self.recovery_distance)
                        f.write(f"\n总路径距离: {total_distance:.2f} km\n")
                        f.write(f"  投放路径: {self.deployment_distance:.2f} km\n")
                        f.write(f"  回收路径: {self.recovery_distance:.2f} km\n")
            
            messagebox.showinfo('成功', f'航时计算结果已保存到: {filename}')
            self.log_result(f"航时计算结果已保存: {filename}")
            
        except Exception as e:
            messagebox.showerror('错误', f'保存失败: {str(e)}')
            import traceback
            traceback.print_exc()
    
    def plan_recovery(self):
        """规划回收站位策略"""
        try:
            # 检查回收路径是否已设置
            if not self.recovery_path or len(self.recovery_path) == 0:
                messagebox.showwarning('警告', '请先设置回收路径！\n\n请先优化回收路径。')
                return
            
            # 检查站位数据
            if not self.stations or len(self.stations) == 0:
                messagebox.showerror('错误', '请先加载站位数据！')
                return
            
            # 获取站位深度：优先使用文件第6列，否则从水深文件读取，最后使用输入框
            station_depths = []
            has_depth_from_file = hasattr(self, 'station_depths_from_file') and any(
                d is not None for d in self.station_depths_from_file
            )
            
            if has_depth_from_file:
                # 使用文件第6列的深度
                station_depths = [d if d is not None else 0.0 for d in self.station_depths_from_file]
                self.log_result("使用站位文件第6列的深度数据")
            elif self.station_depths and len(self.station_depths) == len(self.stations):
                # 使用从水深文件读取的深度
                station_depths = self.station_depths.copy()
                self.log_result("使用从水深文件读取的深度数据")
            else:
                # 尝试从输入框读取
                depths_str = self.station_depths_var.get().strip()
                if depths_str:
                    try:
                        station_depths = [float(d.strip()) for d in depths_str.split(',')]
                        if len(station_depths) != len(self.stations):
                            raise ValueError(f"深度数量 ({len(station_depths)}) 与站位数量 ({len(self.stations)}) 不匹配")
                    except ValueError as e:
                        messagebox.showerror('错误', f'站位深度输入无效: {str(e)}\n\n请检查输入格式。')
                        return
                else:
                    messagebox.showerror('错误', 
                        '无法获取站位深度！\n\n'
                        '请确保：\n'
                        '1. 站位文件包含第6列（深度），或\n'
                        '2. 已加载水深文件，或\n'
                        '3. 在输入框中手动输入深度（逗号分隔）')
                    return
            
            # 获取上浮速度：优先使用文件第5列，否则使用输入框或默认值
            ascent_speed = None
            has_ascent_speed_from_file = hasattr(self, 'station_ascent_speeds') and any(
                s is not None for s in self.station_ascent_speeds
            )
            
            if has_ascent_speed_from_file:
                # 使用文件第5列的上浮速度（取平均值，或使用每个站位自己的速度）
                # 对于回收路径中的站位，使用各自的上浮速度
                # 这里先使用平均值作为统一速度
                valid_speeds = [s for s in self.station_ascent_speeds if s is not None]
                if valid_speeds:
                    ascent_speed = np.mean(valid_speeds)
                    self.log_result(f"使用站位文件第5列的上浮速度，平均速度: {ascent_speed:.2f} 米/秒")
            else:
                # 使用输入框的值或默认值
                try:
                    ascent_speed = float(self.recovery_ascent_speed_var.get())
                except ValueError:
                    ascent_speed = 0.5  # 默认值
                    self.log_result(f"使用默认上浮速度: {ascent_speed} 米/秒")
            
            # 获取其他参数
            try:
                ship_speed = float(self.recovery_ship_speed_var.get())
                num_stations = int(self.rolling_stations_var.get())
                
                if ascent_speed <= 0:
                    raise ValueError("上浮速度必须大于0")
                if ship_speed <= 0:
                    raise ValueError("船速必须大于0")
                
            except ValueError as e:
                messagebox.showerror('错误', f'参数值无效: {str(e)}')
                return
            
            # 对于回收路径中的站位，使用各自的深度和上浮速度（如果有）
            recovery_path_depths = []
            recovery_path_ascent_speeds = []
            
            for station_idx in self.recovery_path:
                if station_idx < len(station_depths):
                    recovery_path_depths.append(station_depths[station_idx])
                else:
                    recovery_path_depths.append(0.0)
                
                # 如果文件中有每个站位自己的上浮速度，使用它；否则使用统一速度
                if has_ascent_speed_from_file and station_idx < len(self.station_ascent_speeds):
                    if self.station_ascent_speeds[station_idx] is not None:
                        recovery_path_ascent_speeds.append(self.station_ascent_speeds[station_idx])
                    else:
                        recovery_path_ascent_speeds.append(ascent_speed)
                else:
                    recovery_path_ascent_speeds.append(ascent_speed)
            
            # 创建回收规划器（使用回收路径中站位的深度）
            # 注意：RecoveryStationPlanner需要所有站位的深度，但我们只关心回收路径中的站位
            # 创建一个临时的深度列表，只包含回收路径中的站位
            planner = RecoveryStationPlanner(
                recovery_path=self.recovery_path,
                stations=self.stations,
                station_depths=station_depths,  # 所有站位的深度
                ascent_speed=ascent_speed,  # 统一的上浮速度（如果文件中有每个站位自己的速度，后续可以扩展）
                ship_speed=ship_speed
            )
            
            # 优化回收顺序
            self.log_result("\n" + "="*60)
            self.log_result(f"回收站位规划（{num_stations}个站位滚动回收）")
            self.log_result("="*60)
            self.log_result(f"\n参数设置:")
            self.log_result(f"  上浮速度: {ascent_speed} 米/秒")
            self.log_result(f"  船速: {ship_speed} 节 ({ship_speed * 1.852:.2f} 公里/小时)")
            self.log_result(f"  滚动站位数: {num_stations}")
            self.log_result(f"  回收路径长度: {len(self.recovery_path)} 个站位")
            
            # 规划回收
            result = planner.optimize_recovery_order(num_stations=num_stations)
            
            # 显示结果
            self.log_result(f"\n规划结果:")
            self.log_result(f"  总回收时间: {result['total_time']/3600:.2f} 小时 ({result['total_time']:.0f} 秒)")
            self.log_result(f"  时间约束满足: {'是' if result['is_feasible'] else '否'}")
            
            if not result['is_feasible']:
                self.log_result(f"\n⚠️ 警告：发现 {len(result['violations'])} 个时间约束违反:")
                for violation in result['violations']:
                    station_idx = violation['station_idx']
                    station_name = self.station_names[station_idx] if station_idx < len(self.station_names) and self.station_names[station_idx] else f'站位 {station_idx}'
                    self.log_result(f"  站位 {station_idx} ({station_name}):")
                    self.log_result(f"    到达时间: {violation['arrival_time']/60:.2f} 分钟")
                    self.log_result(f"    上浮开始时间: {violation['ascent_start_time']/60:.2f} 分钟")
                    self.log_result(f"    延迟: {violation['delay']/60:.2f} 分钟")
            
            self.log_result(f"\n详细回收计划:")
            for i, station_idx in enumerate(result['recovery_order']):
                station_name = self.station_names[station_idx] if station_idx < len(self.station_names) and self.station_names[station_idx] else f'站位 {station_idx}'
                lon, lat = self.stations[station_idx]
                depth = station_depths[station_idx] if station_idx < len(station_depths) else 0.0
                arrival_time = result['arrival_times'][i]
                ascent_start_time = result['ascent_start_times'][i]
                # 使用该站位自己的上浮速度（如果有），否则使用统一速度
                station_ascent_speed = ascent_speed
                if has_ascent_speed_from_file and station_idx < len(self.station_ascent_speeds):
                    if self.station_ascent_speeds[station_idx] is not None:
                        station_ascent_speed = self.station_ascent_speeds[station_idx]
                ascent_time = depth / station_ascent_speed
                recovery_time = result['recovery_times'][i]
                
                self.log_result(f"\n  站位 {i+1}: {station_name} (索引 {station_idx})")
                self.log_result(f"    坐标: ({lon:.4f}°, {lat:.4f}°)")
                self.log_result(f"    深度: {depth:.1f} 米")
                self.log_result(f"    上浮速度: {station_ascent_speed:.2f} 米/秒")
                self.log_result(f"    船到达时间: {arrival_time/60:.2f} 分钟 ({arrival_time:.0f} 秒)")
                self.log_result(f"    上浮开始时间: {ascent_start_time/60:.2f} 分钟 ({ascent_start_time:.0f} 秒)")
                self.log_result(f"    上浮时间: {ascent_time/60:.2f} 分钟 ({ascent_time:.0f} 秒)")
                self.log_result(f"    回收操作时间: {recovery_time/60:.2f} 分钟 ({recovery_time:.0f} 秒)")
                
                # 检查时间约束
                if arrival_time > ascent_start_time:
                    delay = arrival_time - ascent_start_time
                    self.log_result(f"    ⚠️ 时间约束违反: 船延迟 {delay/60:.2f} 分钟到达")
                else:
                    margin = ascent_start_time - arrival_time
                    self.log_result(f"    ✓ 时间裕量: {margin/60:.2f} 分钟")
            
            # 更新回收路径状态显示
            self.recovery_path_status_var.set(f'已设置 ({len(self.recovery_path)} 个站位)')
            
            # 保存结果供后续使用
            self.recovery_plan_result = result
            
            messagebox.showinfo('成功', 
                f'回收规划完成！\n\n'
                f'总时间: {result["total_time"]/3600:.2f} 小时\n'
                f'时间约束: {"满足" if result["is_feasible"] else "不满足"}')
            
        except Exception as e:
            error_msg = f'回收规划失败: {str(e)}'
            messagebox.showerror('错误', error_msg)
            self.log_result(f"\n❌ {error_msg}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    root = tk.Tk()
    app = StationPathOptimizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
