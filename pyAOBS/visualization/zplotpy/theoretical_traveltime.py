"""
theoretical_traveltime.py - 理论走时计算模块

基于 RAYINVR 计算理论走时，用于与观测拾取对比
参考 vedit 和 imodel 的实现方式

Author: Based on RAYINVR by Colin A. Zelt (1994)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import logging

# 导入模型读取相关模块
try:
    from ...model_building.zeltform import ZeltVelocityModel2d
    from ..show_model import GridModelProcessor
except ImportError:
    # 备用导入路径（用于开发模式）
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from pyAOBS.model_building.zeltform import ZeltVelocityModel2d
    from pyAOBS.visualization.show_model import GridModelProcessor

# 导入 RAYINVR 包装器
try:
    from ...modeling.rayinvr.rayinvr_wrapper import RayinvrWrapper
except ImportError:
    # 备用导入路径（用于开发模式）
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from pyAOBS.modeling.rayinvr.rayinvr_wrapper import RayinvrWrapper

logger = logging.getLogger(__name__)


class ModelLoader:
    """模型加载器 - 统一处理不同格式的速度模型"""
    
    def __init__(self):
        """初始化模型加载器"""
        self.model_file = None
        self.zelt_model = None  # v.in格式模型
        self.grid_data = None   # grid格式模型数据
        self.model_type = None  # 'vin' 或 'grid'
    
    @staticmethod
    def is_vin_format(filename: str) -> bool:
        """检测文件是否为 v.in 格式
        
        Args:
            filename: 文件路径
            
        Returns:
            bool: 如果是 v.in 格式返回 True
        """
        file_path = Path(filename)
        
        # 检查文件名
        if file_path.name == 'v.in' or file_path.name.endswith('.vin'):
            return True
        
        # 通过读取文件内容判断
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                if not lines:
                    return False
                
                # v.in 格式的第一行通常以层号（1-2位数字）开头
                first_line = lines[0].strip()
                if first_line and first_line[0].isdigit():
                    # 进一步检查：尝试读取前3行，看是否符合v.in格式
                    # v.in 格式每3行为一组：层号+x坐标, 计数+值, 标志
                    if len(lines) >= 3:
                        # 检查前几行是否符合v.in格式特征
                        try:
                            layer_num = int(first_line.split()[0])
                            if 1 <= layer_num <= 100:  # 合理的层号范围
                                return True
                        except (ValueError, IndexError):
                            pass
        except Exception:
            pass
        
        return False
    
    def load_model(self, model_file: str) -> bool:
        """加载模型文件，自动检测格式
        
        Args:
            model_file: 模型文件路径
            
        Returns:
            bool: 加载成功返回 True
        """
        if not Path(model_file).exists():
            logger.error(f"模型文件不存在: {model_file}")
            return False
        
        self.model_file = Path(model_file)
        
        # 检测格式
        if self.is_vin_format(str(self.model_file)):
            # 加载 v.in 格式（RAYINVR 使用）
            try:
                self.zelt_model = ZeltVelocityModel2d(model_file=str(self.model_file))
                self.model_type = 'vin'
                logger.info(f"成功加载 v.in 格式模型: {model_file}")
                return True
            except Exception as e:
                logger.error(f"加载 v.in 模型失败: {e}")
                return False
        else:
            # 加载 grid 格式
            try:
                processor = GridModelProcessor(grid_file=str(self.model_file))
                self.grid_data = processor.velocity_grid
                self.model_type = 'grid'
                logger.info(f"成功加载 grid 格式模型: {model_file}")
                return True
            except Exception as e:
                logger.error(f"加载 grid 模型失败: {e}")
                return False
    
    def get_model_info(self) -> Dict:
        """获取模型信息
        
        Returns:
            dict: 模型信息字典，包含：
                - model_file: 模型文件路径
                - model_type: 模型类型 ('vin' 或 'grid')
                - has_model: 是否有模型
                - n_layers: 层数（v.in格式）
                - shape: 形状（grid格式）
                - bounds: 模型边界 (xmin, xmax, zmin, zmax)（如果可用）
        """
        info = {
            'model_file': str(self.model_file) if self.model_file else None,
            'model_type': self.model_type,
            'has_model': self.zelt_model is not None or self.grid_data is not None
        }
        
        if self.model_type == 'vin' and self.zelt_model:
            info['n_layers'] = len(self.zelt_model.depth_nodes) if hasattr(self.zelt_model, 'depth_nodes') else None
            # 获取模型边界
            try:
                xmin, xmax, zmin, zmax = self.zelt_model.get_model_bounds()
                info['bounds'] = {
                    'xmin': xmin,
                    'xmax': xmax,
                    'zmin': zmin,
                    'zmax': zmax
                }
            except Exception as e:
                logger.warning(f"无法获取模型边界: {e}")
                info['bounds'] = None
        elif self.model_type == 'grid' and self.grid_data is not None:
            info['shape'] = dict(self.grid_data.dims) if hasattr(self.grid_data, 'dims') else None
            # 对于 grid 格式，尝试从坐标获取边界
            try:
                if hasattr(self.grid_data, 'coords'):
                    x_coords = self.grid_data.coords.get('x', None)
                    z_coords = self.grid_data.coords.get('z', None)
                    if x_coords is not None and z_coords is not None:
                        info['bounds'] = {
                            'xmin': float(x_coords.min()),
                            'xmax': float(x_coords.max()),
                            'zmin': float(z_coords.min()),
                            'zmax': float(z_coords.max())
                        }
                    else:
                        info['bounds'] = None
                else:
                    info['bounds'] = None
            except Exception as e:
                logger.warning(f"无法获取 grid 模型边界: {e}")
                info['bounds'] = None
        else:
            info['bounds'] = None
        
        return info


class TheoreticalTravelTimeCalculator:
    """理论走时计算器 - 基于 RAYINVR"""
    
    def __init__(self, model_file_path: Optional[str] = None, 
                 working_dir: Optional[str] = None,
                 data_loader=None,
                 pick_manager=None):
        """初始化理论走时计算器
        
        Args:
            model_file_path: 模型文件路径（v.in 格式，RAYINVR 需要）
            working_dir: 工作目录，如果为 None 则使用模型文件所在目录
            data_loader: DataLoader 实例（可选，用于自动生成输入文件）
            pick_manager: PickManager 实例（可选，用于生成 tx.in）
        """
        self.model_loader = ModelLoader()
        self.model_file_path = model_file_path
        self.working_dir = working_dir
        self.data_loader = data_loader
        self.pick_manager = pick_manager
        
        # RAYINVR 相关
        self.rayinvr_wrapper = None
        self.theoretical_times = None  # 理论走时数据
        
        # 如果提供了模型文件，自动加载
        if model_file_path:
            self.load_model(model_file_path)
    
    def load_model(self, model_file_path: str) -> bool:
        """加载速度模型
        
        Args:
            model_file_path: 模型文件路径
            
        Returns:
            bool: 加载成功返回 True
        """
        success = self.model_loader.load_model(model_file_path)
        if success:
            self.model_file_path = model_file_path
            # 设置工作目录
            if self.working_dir is None:
                self.working_dir = str(Path(model_file_path).parent)
        return success
    
    def prepare_rayinvr_input(self) -> bool:
        """准备 RAYINVR 输入文件
        
        确保 v.in 文件在工作目录中（RAYINVR 需要）
        工作目录设置为模型文件所在目录
        
        Returns:
            bool: 准备成功返回 True
        """
        if self.model_loader.model_type != 'vin':
            logger.error("RAYINVR 需要 v.in 格式的模型文件")
            return False
        
        if not self.model_file_path or not Path(self.model_file_path).exists():
            logger.error("模型文件不存在")
            return False
        
        # 工作目录设置为模型文件所在目录（确保r.in和v.in在同一目录）
        model_dir = Path(self.model_file_path).parent
        self.working_dir = str(model_dir)  # 更新working_dir为模型目录
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查模型目录中是否有 v.in 文件
        vin_file_in_modeldir = model_dir / 'v.in'
        if not vin_file_in_modeldir.exists():
            # 如果模型文件不是v.in，需要复制或重命名
            model_path = Path(self.model_file_path)
            if model_path.name == 'v.in' and model_path.parent == model_dir:
                # 模型文件就是v.in且在模型目录，无需操作
                pass
            else:
                # 需要复制到模型目录
                import shutil
                try:
                    shutil.copy2(self.model_file_path, vin_file_in_modeldir)
                    logger.info(f"已将模型文件复制到模型目录: {vin_file_in_modeldir}")
                except Exception as e:
                    logger.error(f"复制模型文件失败: {e}")
                    return False
        
        logger.info(f"RAYINVR工作目录设置为: {self.working_dir}")
        return True
    
    def calculate_travel_times(self, 
                              auto_generate_inputs: bool = True,
                              shot_position: Optional[Tuple[float, float]] = None,
                              ray_params: Optional[Dict] = None,
                              use_observed_picks: bool = False,
                              pick_word: int = 1) -> bool:
        """计算理论走时
        
        改进：支持自动生成 r.in 和 tx.in 文件
        
        Args:
            auto_generate_inputs: 是否自动生成输入文件（最小化参数）
            shot_position: 炮点位置 (x, z)，如果为None则从数据中提取或使用默认值
            ray_params: 射线参数字典，可包含 ray, nray, xmin, xmax 等
            use_observed_picks: 是否使用观测拾取生成 tx.in（最小化格式）
            pick_word: 使用的拾取字（如果生成 tx.in）
            
        Returns:
            bool: 计算成功返回 True
        """
        # 准备 RAYINVR 输入
        if not self.prepare_rayinvr_input():
            return False
        
        # 自动生成输入文件（如果启用）
        if auto_generate_inputs:
            # 检查是否有 data_loader（生成 r.in 需要）
            if not self.data_loader:
                work_dir = Path(self.working_dir) if self.working_dir else Path(self.model_file_path).parent
                existing_r_in = work_dir / 'r.in'
                if not existing_r_in.exists():
                    logger.warning("auto_generate_inputs=True 但未提供 data_loader，且工作目录中不存在 r.in")
                    logger.warning("无法生成 r.in 文件，RAYINVR 可能无法运行")
                    return False
                else:
                    logger.info(f"检测到工作目录中已存在 r.in 文件: {existing_r_in}")
                    logger.info("跳过自动生成，将使用现有的 r.in 文件")
            else:
                # 总是重新生成 r.in（使用用户指定的参数），覆盖现有文件
                # 这样确保用户修改的参数（如 shot_position, nray 等）能够生效
                try:
                    r_in_path = self.generate_r_in_from_data(
                        shot_position, ray_params
                    )
                    logger.info(f"已自动生成/更新最小化 r.in 文件: {r_in_path}")
                except Exception as e:
                    logger.error(f"自动生成 r.in 文件失败: {e}")
                    return False
            
            # 可选生成符合 RAYINVR 格式的 tx.in
            if use_observed_picks:
                try:
                    tx_in_path = self.generate_tx_in_from_picks(
                        pick_word=pick_word,
                        shot_position=shot_position,
                        phase_id=1,  # 默认震相标识为 1
                        uncertainty=0.050  # 默认误差为 0.050 秒
                    )
                    if tx_in_path:
                        logger.info(f"已自动生成符合 RAYINVR 格式的 tx.in 文件: {tx_in_path}")
                except Exception as e:
                    logger.warning(f"自动生成 tx.in 文件失败（可能没有拾取数据）: {e}")
        
        # 初始化 RAYINVR 包装器
        try:
            self.rayinvr_wrapper = RayinvrWrapper(working_dir=self.working_dir)
        except Exception as e:
            logger.error(f"初始化 RAYINVR 包装器失败: {e}")
            return False
        
        # 运行 RAYINVR
        try:
            logger.info("开始运行 RAYINVR 计算理论走时...")
            success = self.rayinvr_wrapper.run_rayinvr()
            if not success:
                logger.error("RAYINVR 运行失败")
                return False
            
            logger.info("RAYINVR 计算完成")
            
            # 读取并输出理论走时（从 tx.out 文件）
            self._read_and_output_theoretical_traveltime()
            
            return True
            
        except Exception as e:
            logger.error(f"运行 RAYINVR 时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def get_theoretical_times(self, distances: Optional[np.ndarray] = None) -> Optional[Dict]:
        """获取理论走时数据
        
        参考 vedit 的实现方式，优先从射线数据中提取走时，确保数据正确
        
        Args:
            distances: 距离数组（km），如果为 None 则获取所有理论走时
            
        Returns:
            dict: 包含理论走时数据的字典，格式为：
                {
                    'distance': np.ndarray,  # 距离 (km)
                    'time': np.ndarray,      # 走时 (s)
                    'rays': list            # 射线数据（可选）
                }
        """
        if self.rayinvr_wrapper is None:
            logger.error("请先运行 calculate_travel_times()")
            return None
        
        try:
            # 方法1：尝试从射线数据中提取走时（参考 vedit 的实现）
            # vedit 中从 ray['x'][-1] 和 ray['total_time'] 构建走时曲线
            rays = self.rayinvr_wrapper.get_all_rays(max_rays=1000)
            
            if rays and len(rays) > 0:
                # 从射线数据提取走时
                ray_distances = []
                ray_times = []
                
                for ray in rays:
                    if len(ray.get('x', [])) > 0 and 'total_time' in ray:
                        # 使用射线的最后一个点作为距离，total_time 作为走时
                        distance = ray['x'][-1]
                        time = ray['total_time']
                        if distance > 0 and time > 0:  # 只添加有效数据
                            ray_distances.append(distance)
                            ray_times.append(time)
                
                if ray_distances and ray_times:
                    # 排序并去重（按距离）
                    sorted_pairs = sorted(zip(ray_distances, ray_times))
                    unique_distances = []
                    unique_times = []
                    prev_dist = None
                    for dist, t in sorted_pairs:
                        if prev_dist is None or abs(dist - prev_dist) > 0.01:  # 避免重复距离
                            unique_distances.append(dist)
                            unique_times.append(t)
                            prev_dist = dist
                    
                    if unique_distances:
                        distances_array = np.array(unique_distances)
                        times_array = np.array(unique_times)
                        
                        logger.info(f"从 {len(rays)} 条射线中提取了 {len(distances_array)} 个走时数据点")
                        
                        # 如果指定了距离，进行插值
                        if distances is not None:
                            from scipy.interpolate import interp1d
                            interp_func = interp1d(
                                distances_array, 
                                times_array, 
                                kind='linear',
                                bounds_error=False,
                                fill_value='extrapolate'
                            )
                            interpolated_times = interp_func(distances)
                            
                            return {
                                'distance': distances,
                                'time': interpolated_times,
                                'original_distance': distances_array,
                                'original_time': times_array,
                                'rays': rays
                            }
                        else:
                            return {
                                'distance': distances_array,
                                'time': times_array,
                                'rays': rays
                            }
            
            # 方法2：如果从射线数据无法获取，尝试使用 get_travel_times() 方法
            logger.info("尝试使用 get_travel_times() 方法获取走时数据")
            travel_time_data = self.rayinvr_wrapper.get_travel_times()
            
            if travel_time_data is None:
                logger.warning("未获取到理论走时数据（两种方法都失败）")
                return None
            
            # 如果指定了距离，进行插值
            if distances is not None:
                from scipy.interpolate import interp1d
                distances_array = travel_time_data['distance']
                times_array = travel_time_data['time']
                
                if len(distances_array) == 0:
                    logger.warning("get_travel_times() 返回的数据为空")
                    return None
                
                # 创建插值函数
                interp_func = interp1d(
                    distances_array, 
                    times_array, 
                    kind='linear',
                    bounds_error=False,
                    fill_value='extrapolate'
                )
                
                # 插值到指定距离
                interpolated_times = interp_func(distances)
                
                return {
                    'distance': distances,
                    'time': interpolated_times,
                    'original_distance': distances_array,
                    'original_time': times_array
                }
            else:
                # 返回所有数据
                return travel_time_data
                
        except Exception as e:
            logger.error(f"获取理论走时数据失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def get_all_rays(self, max_rays: int = 100) -> List[Dict]:
        """获取所有射线路径数据
        
        Args:
            max_rays: 最大返回射线数
            
        Returns:
            list: 射线数据列表
        """
        if self.rayinvr_wrapper is None:
            logger.error("请先运行 calculate_travel_times()")
            return []
        
        try:
            rays = self.rayinvr_wrapper.get_all_rays(max_rays=max_rays)
            return rays
        except Exception as e:
            logger.error(f"获取射线数据失败: {e}")
            return []
    
    def compare_with_observed(self, 
                             observed_picks: Dict[int, float],
                             trace_offsets: Dict[int, float]) -> Optional[Dict]:
        """与观测拾取对比
        
        Args:
            observed_picks: 观测拾取字典 {trace_idx: pick_time}
            trace_offsets: 道偏移距字典 {trace_idx: offset_km}
            
        Returns:
            dict: 对比结果，包含：
                {
                    'trace_idx': np.ndarray,
                    'distance': np.ndarray,      # 距离 (km)
                    'observed_time': np.ndarray, # 观测走时 (s)
                    'theoretical_time': np.ndarray, # 理论走时 (s)
                    'residual': np.ndarray      # 残差 (s)
                }
        """
        # 获取理论走时
        distances = np.array([trace_offsets[idx] for idx in observed_picks.keys()])
        theoretical_data = self.get_theoretical_times(distances)
        
        if theoretical_data is None:
            return None
        
        # 构建对比结果
        trace_indices = np.array(list(observed_picks.keys()))
        observed_times = np.array([observed_picks[idx] for idx in trace_indices])
        theoretical_times = theoretical_data['time']
        residuals = observed_times - theoretical_times
        
        return {
            'trace_idx': trace_indices,
            'distance': distances,
            'observed_time': observed_times,
            'theoretical_time': theoretical_times,
            'residual': residuals
        }
    
    def get_model_info(self) -> Dict:
        """获取模型信息
        
        Returns:
            dict: 模型信息
        """
        return self.model_loader.get_model_info()
    
    def get_water_layer_depth(self) -> Optional[Union[float, Tuple[np.ndarray, np.ndarray]]]:
        """从 v.in 模型的第一层提取水层深度界面
        
        注意：v.in 格式中第一层（layer_idx=0）就是水层，
        第一层的深度节点定义了水层底界（海底面）
        
        Returns:
            - 如果水层深度是常数：返回 float（深度值，km）
            - 如果水层深度随x变化：返回 (x_array, depth_array) 元组
            - 如果无法获取：返回 None
        """
        if self.model_loader.model_type != 'vin' or not self.model_loader.zelt_model:
            logger.warning("无法获取水层深度：模型不是 v.in 格式或未加载")
            return None
        
        try:
            zelt_model = self.model_loader.zelt_model
            
            # 获取第一层的几何信息（x坐标和深度值）
            x_coords, depths = zelt_model.get_layer_geometry(0)
            
            if not x_coords or not depths:
                logger.warning("第一层没有深度节点数据")
                return None
            
            x_array = np.array(x_coords)
            depth_array = np.array(depths)
            
            # 检查深度是否随x变化
            depth_range = np.max(depth_array) - np.min(depth_array)
            if depth_range < 0.01:  # 如果深度变化 < 10m，认为是常数
                constant_depth = float(np.mean(depth_array))
                logger.info(f"水层深度为常数: {constant_depth:.3f} km")
                return constant_depth
            else:
                logger.info(f"水层深度随x变化: 范围 [{np.min(depth_array):.3f}, {np.max(depth_array):.3f}] km")
                return (x_array, depth_array)
                
        except Exception as e:
            logger.error(f"获取水层深度失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def extract_water_layer_segments(self, 
                                     ray: Dict,
                                     water_depth: Union[float, Tuple[np.ndarray, np.ndarray]],
                                     z_surface: float = 0.0) -> List[Dict]:
        """从射线路径中提取水层段
        
        注意：水层定义为 z_surface（通常为0，海面）到 water_depth（海底面）之间的区域
        即：z_surface <= z < water_depth
        
        Args:
            ray: 射线数据字典，必须包含：
                - 'x': np.ndarray，x坐标数组（km）
                - 'z': np.ndarray，z坐标数组（km，向下为正）
                - 'time': np.ndarray，累积走时数组（s）
            water_depth: 水层深度（km）
                - float: 常数深度（海底面深度，相对于海面）
                - Tuple[np.ndarray, np.ndarray]: (x_array, depth_array) 变化深度
            z_surface: 海面深度（km），默认 0.0
        
        Returns:
            water_segments: 水层路径段列表，每个段包含：
                {
                    'x': np.ndarray,      # 水层中的x坐标
                    'z': np.ndarray,      # 水层中的z坐标
                    'time': np.ndarray,   # 水层中的累积走时
                    'segment_length': float,  # 段长度（km）
                    'segment_time': float     # 段走时（s）
                }
        """
        if 'x' not in ray or 'z' not in ray or 'time' not in ray:
            logger.error("射线数据缺少必要字段（需要 'x', 'z', 'time'）")
            return []
        
        x = np.asarray(ray['x'])
        z = np.asarray(ray['z'])
        time = np.asarray(ray['time'])
        
        if len(x) == 0 or len(z) == 0 or len(time) == 0:
            logger.warning("射线数据为空")
            return []
        
        if len(x) != len(z) or len(x) != len(time):
            logger.error(f"射线数据长度不一致: x={len(x)}, z={len(z)}, time={len(time)}")
            return []
        
        # 1. 将 water_depth 转换为数组形式（每个x位置对应的深度）
        if isinstance(water_depth, (int, float)):
            # 常数深度：所有位置使用相同的深度
            water_depth_array = np.full_like(x, water_depth)
        elif isinstance(water_depth, tuple) and len(water_depth) == 2:
            # 变化深度：需要插值到射线的x坐标
            x_depth, depth_values = water_depth
            x_depth = np.asarray(x_depth)
            depth_values = np.asarray(depth_values)
            
            # 使用线性插值
            from scipy.interpolate import interp1d
            try:
                interp_func = interp1d(
                    x_depth, 
                    depth_values, 
                    kind='linear',
                    bounds_error=False,
                    fill_value=(depth_values[0] if len(depth_values) > 0 else 0.0, 
                               depth_values[-1] if len(depth_values) > 0 else 0.0)
                )
                water_depth_array = interp_func(x)
            except Exception as e:
                logger.warning(f"插值水层深度失败: {e}，使用常数深度")
                water_depth_array = np.full_like(x, np.mean(depth_values) if len(depth_values) > 0 else 0.0)
        else:
            logger.error(f"无效的 water_depth 类型: {type(water_depth)}")
            return []
        
        # 2. 识别水层中的点（z_surface <= z < water_depth）
        # 注意：z向下为正，所以水层是 z >= z_surface 且 z < water_depth
        in_water = (z >= z_surface) & (z < water_depth_array)
        
        if not np.any(in_water):
            # 没有点在水层中
            return []
        
        # 3. 找到连续的水层段
        # 使用差分找到段的开始和结束位置
        segment_starts = []
        segment_ends = []
        
        # 找到从非水层进入水层的点
        for i in range(len(in_water) - 1):
            if not in_water[i] and in_water[i+1]:
                segment_starts.append(i+1)
            if in_water[i] and not in_water[i+1]:
                segment_ends.append(i+1)
        
        # 处理边界情况
        if in_water[0]:
            segment_starts.insert(0, 0)
        if in_water[-1]:
            segment_ends.append(len(in_water))
        
        # 确保段的数量一致
        if len(segment_starts) != len(segment_ends):
            logger.warning(f"水层段数量不一致: starts={len(segment_starts)}, ends={len(segment_ends)}")
            # 取较小的数量
            n_segments = min(len(segment_starts), len(segment_ends))
            segment_starts = segment_starts[:n_segments]
            segment_ends = segment_ends[:n_segments]
        
        # 4. 提取每个水层段
        water_segments = []
        
        for start, end in zip(segment_starts, segment_ends):
            if start >= end:
                continue
            
            segment_x = x[start:end]
            segment_z = z[start:end]
            segment_time = time[start:end]
            
            # 计算段长度（路径积分：sum(sqrt(dx^2 + dz^2))）
            if len(segment_x) > 1:
                dx = np.diff(segment_x)
                dz = np.diff(segment_z)
                segment_length = np.sum(np.sqrt(dx**2 + dz**2))
            else:
                segment_length = 0.0
            
            # 计算段走时（最后一个点的累积走时 - 第一个点的累积走时）
            if len(segment_time) > 1:
                segment_travel_time = segment_time[-1] - segment_time[0]
            else:
                segment_travel_time = 0.0
            
            water_segments.append({
                'x': segment_x.copy(),
                'z': segment_z.copy(),
                'time': segment_time.copy(),
                'segment_length': float(segment_length),
                'segment_time': float(segment_travel_time),
                'start_idx': int(start),
                'end_idx': int(end)
            })
        
        return water_segments
    
    def calculate_water_layer_statistics(self,
                                        rays: List[Dict],
                                        water_depth: Optional[Union[float, Tuple[np.ndarray, np.ndarray]]] = None,
                                        z_surface: float = 0.0) -> List[Dict]:
        """计算所有射线在水层中的统计信息
        
        Args:
            rays: 射线列表，每个射线是包含 'x', 'z', 'time' 的字典
            water_depth: 水层深度（km），如果为None则自动从模型获取
                - float: 常数深度
                - Tuple[np.ndarray, np.ndarray]: (x_array, depth_array) 变化深度
            z_surface: 海面深度（km），默认 0.0
        
        Returns:
            water_stats_list: 每个射线的水层统计信息列表，每个元素包含：
                {
                    'ray_idx': int,                    # 射线索引
                    'total_water_path_length': float,  # 总水层路径长度（km）
                    'total_water_time': float,         # 总水层走时（s）
                    'n_segments': int,                 # 水层段数量
                    'water_segments': list             # 水层段列表
                }
        """
        # 如果没有提供 water_depth，尝试从模型获取
        if water_depth is None:
            water_depth = self.get_water_layer_depth()
            if water_depth is None:
                logger.warning("无法获取水层深度，无法计算水层统计信息")
                return []
        
        water_stats_list = []
        
        for ray_idx, ray in enumerate(rays):
            # 提取水层段
            segments = self.extract_water_layer_segments(ray, water_depth, z_surface)
            
            # 计算总长度和总走时
            total_length = sum(seg['segment_length'] for seg in segments)
            total_time = sum(seg['segment_time'] for seg in segments)
            
            water_stats_list.append({
                'ray_idx': ray_idx,
                'total_water_path_length': total_length,
                'total_water_time': total_time,
                'n_segments': len(segments),
                'water_segments': segments
            })
        
        return water_stats_list
    
    def _get_replacement_velocity(self, 
                                   water_depth: Union[float, Tuple[np.ndarray, np.ndarray]],
                                   default_v_replacement: Optional[float] = None) -> float:
        """获取替换速度
        
        如果未指定，尝试从模型获取水层下第一层的速度
        
        Args:
            water_depth: 水层深度（km）
            default_v_replacement: 默认替换速度（km/s），如果为None则尝试从模型获取
        
        Returns:
            替换速度（km/s）
        """
        if default_v_replacement is not None:
            return default_v_replacement
        
        # 尝试从模型获取水层下第一层的速度
        if self.model_loader and self.model_loader.model_type == 'vin' and self.model_loader.zelt_model:
            try:
                zelt_model = self.model_loader.zelt_model
                
                # 获取第二层（layer_idx=1）的速度作为替换速度
                # 第二层是水层下的第一层
                if hasattr(zelt_model, 'vupper_nodes') and len(zelt_model.vupper_nodes) > 1:
                    # 获取第二层上界速度的平均值
                    v_upper_values = zelt_model.vupper_nodes[1].get_values()
                    if v_upper_values:
                        v_replacement = float(np.mean(v_upper_values))
                        logger.info(f"从模型获取替换速度（第二层上界速度）: {v_replacement:.3f} km/s")
                        return v_replacement
                
                # 如果无法获取，使用默认值
                logger.warning("无法从模型获取替换速度，使用默认值 2.0 km/s")
                return 2.0
                
            except Exception as e:
                logger.warning(f"获取替换速度失败: {e}，使用默认值 2.0 km/s")
                return 2.0
        else:
            # 使用默认值
            logger.info("使用默认替换速度: 2.0 km/s")
            return 2.0
    
    def calculate_water_layer_correction(self,
                                       rays: Optional[List[Dict]] = None,
                                       water_depth: Optional[Union[float, Tuple[np.ndarray, np.ndarray]]] = None,
                                       v_water: float = 1.5,
                                       v_replacement: Optional[float] = None,
                                       z_surface: float = 0.0,
                                       return_by_distance: bool = False) -> Union[Dict[int, float], Dict[float, float]]:
        """计算水层校正量
        
        校正公式：校正量 = 水层路径长度 * (1/v_water - 1/v_replacement)
        
        Args:
            rays: 射线列表，如果为None则从 rayinvr_wrapper 获取
            water_depth: 水层深度（km），如果为None则自动从模型获取
                - float: 常数深度
                - Tuple[np.ndarray, np.ndarray]: (x_array, depth_array) 变化深度
            v_water: 水层速度（km/s），默认 1.5
            v_replacement: 替换速度（km/s），如果为None则尝试从模型获取或使用默认值 2.0
            z_surface: 海面深度（km），默认 0.0
            return_by_distance: 如果为True，返回按距离索引的字典；否则返回按射线索引的字典
        
        Returns:
            校正量字典：
                - 如果 return_by_distance=False: {ray_idx: correction_time}
                - 如果 return_by_distance=True: {distance: correction_time}
        """
        # 1. 获取射线数据
        if rays is None:
            if self.rayinvr_wrapper is None:
                logger.error("请先运行 calculate_travel_times() 或提供 rays 参数")
                return {}
            rays = self.get_all_rays(max_rays=1000)
            if not rays:
                logger.warning("未获取到射线数据")
                return {}
        
        # 2. 获取水层深度
        if water_depth is None:
            water_depth = self.get_water_layer_depth()
            if water_depth is None:
                logger.warning("无法获取水层深度，无法计算水层校正量")
                return {}
        
        # 3. 获取替换速度
        v_replacement = self._get_replacement_velocity(water_depth, v_replacement)
        
        # 4. 计算水层统计信息
        water_stats = self.calculate_water_layer_statistics(rays, water_depth, z_surface)
        
        if not water_stats:
            logger.warning("未计算到水层统计信息")
            return {}
        
        # 5. 计算校正量
        # 校正量 = 水层路径长度 * (1/v_water - 1/v_replacement)
        correction_factor = (1.0 / v_water) - (1.0 / v_replacement)
        
        corrections_by_ray = {}
        corrections_by_distance = {}
        
        for stat in water_stats:
            ray_idx = stat['ray_idx']
            total_water_path_length = stat['total_water_path_length']
            
            # 计算校正量（秒）
            correction = total_water_path_length * correction_factor
            
            corrections_by_ray[ray_idx] = correction
            
            # 如果射线有距离信息，也按距离索引
            if return_by_distance and ray_idx < len(rays):
                ray = rays[ray_idx]
                if 'x' in ray and len(ray['x']) > 0:
                    # 使用射线的终点距离
                    distance = float(ray['x'][-1])
                    corrections_by_distance[distance] = correction
        
        logger.info(f"计算了 {len(corrections_by_ray)} 条射线的水层校正量")
        logger.info(f"水层速度: {v_water:.3f} km/s, 替换速度: {v_replacement:.3f} km/s")
        logger.info(f"校正系数: {correction_factor:.6f} s/km")
        
        if return_by_distance:
            return corrections_by_distance
        else:
            return corrections_by_ray
    
    def apply_water_layer_correction(self,
                                    original_times: Union[Dict[int, float], np.ndarray, List[float]],
                                    corrections: Union[Dict[int, float], Dict[float, float]],
                                    by_distance: bool = False) -> Union[Dict[int, float], np.ndarray]:
        """应用水层校正
        
        校正后的走时：t_corrected = t_original - correction
        
        Args:
            original_times: 原始走时
                - Dict[int, float]: 按射线索引的走时字典
                - np.ndarray 或 List[float]: 走时数组（按射线顺序）
            corrections: 校正量字典
                - Dict[int, float]: 按射线索引的校正量
                - Dict[float, float]: 按距离索引的校正量（需要 by_distance=True）
            by_distance: 如果为True，使用距离索引的校正量
        
        Returns:
            校正后的走时，格式与 original_times 相同
        """
        if isinstance(original_times, dict):
            # 字典格式：按射线索引
            corrected_times = {}
            for ray_idx, original_time in original_times.items():
                if by_distance:
                    # 需要从射线获取距离，这里暂时不支持
                    logger.warning("by_distance=True 时，original_times 应为数组格式")
                    correction = corrections.get(ray_idx, 0.0)
                else:
                    correction = corrections.get(ray_idx, 0.0)
                corrected_times[ray_idx] = original_time - correction
            return corrected_times
        
        elif isinstance(original_times, (np.ndarray, list)):
            # 数组格式：按射线顺序
            original_times_array = np.asarray(original_times)
            corrected_times = original_times_array.copy()
            
            for i in range(len(original_times_array)):
                if by_distance:
                    # 需要从射线获取距离，这里暂时不支持
                    logger.warning("by_distance=True 时，需要提供射线数据以获取距离")
                    correction = 0.0
                else:
                    correction = corrections.get(i, 0.0)
                corrected_times[i] = original_times_array[i] - correction
            
            return corrected_times
        else:
            logger.error(f"不支持的 original_times 类型: {type(original_times)}")
            return original_times
    
    # ========== 自动生成 r.in 和 tx.in 文件的方法 ==========
    
    def _extract_receiver_positions(self) -> List[float]:
        """从数据中提取接收点位置
        
        优先级：
        1. 如果道头中有模型位置（xmod），使用模型位置
        2. 否则，使用偏移距（offsti）
        3. 如果都没有，返回空列表
        
        Returns:
            接收点位置列表（X坐标，km）
        """
        if not self.data_loader:
            return []
        
        receivers = []
        trace_headers = self.data_loader.trace_headers
        
        for th in trace_headers:
            # 优先使用模型位置（如果有）
            # 注意：这里假设 xmod 在 records 中，需要检查
            if self.data_loader.records:
                # 如果有记录文件，尝试从记录中获取模型位置
                # 但记录文件可能没有对应每个道的模型位置
                # 所以这里主要使用偏移距
                pass
            
            # 使用偏移距作为接收点位置
            if th.offsti > 0:
                receivers.append(th.offsti)
            elif hasattr(th, 'rxutm') and th.rxutm != 0:
                # 如果有UTM坐标，可以使用（但需要转换为模型坐标）
                # 这里暂时使用偏移距
                receivers.append(abs(th.rxutm) / 1000.0)  # 米转千米
        
        return receivers
    
    def _extract_shot_position(self) -> Optional[Tuple[float, float]]:
        """从数据中提取炮点位置
        
        优先级：
        1. 从记录文件（.rsp）中读取（xmod, ymod）
        2. 从道头中提取（sxutm, sz）
        3. 如果都没有，返回 None
        
        Returns:
            炮点位置 (x, z) km，如果无法提取则返回 None
        """
        if not self.data_loader:
            return None
        
        # 1. 尝试从记录文件中获取
        if self.data_loader.records:
            # 使用第一个记录的位置作为炮点位置
            first_record = self.data_loader.records[0]
            if hasattr(first_record, 'xmod') and hasattr(first_record, 'ymod'):
                if first_record.xmod != 0 or first_record.ymod != 0:
                    return (first_record.xmod, 0.0)  # z坐标默认为0
        
        # 2. 尝试从道头中提取
        if self.data_loader.trace_headers:
            first_header = self.data_loader.trace_headers[0]
            if hasattr(first_header, 'sxutm') and hasattr(first_header, 'sz'):
                if first_header.sxutm != 0:
                    x = first_header.sxutm / 1000.0  # 米转千米
                    z = first_header.sz if first_header.sz != 0 else 0.0
                    return (x, z)
        
        return None
    
    def _get_offsets_for_picks(self, picks: Dict[int, float]) -> Dict[int, float]:
        """获取拾取点对应的偏移距
        
        Args:
            picks: 拾取字典 {trace_idx: pick_time}
        
        Returns:
            偏移距字典 {trace_idx: offset_km}
        """
        if not self.data_loader:
            return {}
        
        offsets = {}
        trace_headers = self.data_loader.trace_headers
        
        for trace_idx in picks.keys():
            if 0 <= trace_idx < len(trace_headers):
                th = trace_headers[trace_idx]
                offsets[trace_idx] = th.offsti
        
        return offsets
    
    def generate_r_in_from_data(self, 
                                 shot_position: Optional[Tuple[float, float]] = None,
                                 ray_params: Optional[Dict] = None) -> str:
        """
        根据加载的数据自动生成最小化的 r.in 文件
        只包含最基本参数，确保射线追踪能够进行
        
        注意：模型范围（xmin, xmax, zmin, zmax）直接从模型提取，不从数据确认
        
        Args:
            shot_position: 炮点位置 (x, z)，如果为None则从数据提取或使用默认值
            ray_params: 射线参数字典，可包含：
                - 'ray': 射线参数数组（控制震相类型），默认 [1.2]（第1层反射）
                  - 编码规则：L.1=第L层折射, L.2=第L层反射, L.3=第L层首波
                  - 可以是单个值（如 1.1）或数组（如 [1.1, 2.1]）
                - 'nray': 射线数量，默认 5
                - 'xmin', 'xmax': X轴范围（可选，覆盖从模型提取的值）
                - 'zmin', 'zmax': Z轴范围（可选，覆盖从模型提取的值）
        
        Returns:
            r.in 文件路径
        """
        if not self.data_loader:
            raise ValueError("需要提供 data_loader 以自动生成 r.in")
        
        # 1. 从模型获取边界信息（必须从模型提取）
        model_info = self.model_loader.get_model_info()
        model_bounds = model_info.get('bounds')
        
        if not model_bounds:
            raise ValueError("无法从模型获取边界信息（xmin, xmax, zmin, zmax），请确保模型已正确加载")
        
        # 2. 获取接收点位置（仅用于生成 tx.in，不用于确定模型范围）
        receivers = self._extract_receiver_positions()
        
        # 3. 获取炮点位置（默认 (0.0, 0.0)）
        if shot_position is None:
            shot_position = self._extract_shot_position() or (0.0, 0.0)
            logger.info(f"未提供炮点位置，从数据提取或使用默认值: {shot_position}")
        else:
            logger.info(f"使用用户提供的炮点位置: {shot_position}")
        
        # 4. 设置默认射线参数（直接从模型边界提取）
        if ray_params is None:
            ray_params = {}
        
        # 直接从模型边界提取默认值
        default_xmin = model_bounds['xmin']
        default_xmax = model_bounds['xmax']
        default_zmin = model_bounds['zmin']
        default_zmax = model_bounds['zmax']
        logger.info(f"从模型提取边界: x=[{default_xmin:.1f}, {default_xmax:.1f}], z=[{default_zmin:.1f}, {default_zmax:.1f}]")
        
        default_ray_params = {
            'ray': [1.2],    # 射线参数数组（默认：第1层反射）
            'nray': 10,      # 射线数量（默认：10，与 vedit 保持一致）
            'xmin': default_xmin,
            'xmax': default_xmax,
            'zmin': default_zmin,
            'zmax': default_zmax
        }
        # 合并用户参数和默认值（用户参数优先）
        final_ray_params = {**default_ray_params, **ray_params}
        logger.info(f"射线参数 - ray: {final_ray_params.get('ray')}, nray: {final_ray_params.get('nray')}")
        
        # 确保 ray 是列表格式
        if isinstance(final_ray_params.get('ray'), (int, float)):
            final_ray_params['ray'] = [final_ray_params['ray']]
        elif not isinstance(final_ray_params.get('ray'), list):
            final_ray_params['ray'] = [1.2]  # 默认值
        
        # 5. 生成最小化 r.in 文件（使用用户参数）
        r_in_path = self._write_minimal_r_in_file(
            shot_position, receivers, final_ray_params
        )
        
        return r_in_path
    
    def _write_minimal_r_in_file(self, 
                                  shot_position: Tuple[float, float],
                                  receivers: List[float],
                                  ray_params: Dict) -> str:
        """
        写入最小化的 r.in 文件
        
        注意：模型范围（xmin, xmax, zmin, zmax）直接从模型提取，不从数据确认
        
        Args:
            shot_position: 炮点位置 (x, z)
            receivers: 接收点位置列表（仅用于日志，不用于确定模型范围）
            ray_params: 射线参数字典，包含 ray, nray, xmin, xmax, zmin, zmax 等
        
        Returns:
            r.in 文件路径
        """
        # 从模型获取边界信息（必须从模型提取）
        model_info = self.model_loader.get_model_info()
        model_bounds = model_info.get('bounds')
        
        if not model_bounds:
            raise ValueError("无法从模型获取边界信息，无法生成 r.in 文件")
        
        # 确定 x 和 z 范围（优先使用用户指定，否则使用模型边界）
        # 确定 x 范围
        if 'xmin' in ray_params and 'xmax' in ray_params:
            # 用户明确指定了范围
            xmin = ray_params['xmin']
            xmax = ray_params['xmax']
            logger.info(f"使用用户指定的 X 范围: [{xmin:.1f}, {xmax:.1f}]")
            # 验证是否与模型一致
            if abs(xmin - model_bounds['xmin']) > 0.01 or abs(xmax - model_bounds['xmax']) > 0.01:
                logger.warning(f"用户指定的 X 范围与模型不一致！模型: [{model_bounds['xmin']:.1f}, {model_bounds['xmax']:.1f}], "
                             f"使用: [{xmin:.1f}, {xmax:.1f}]")
        else:
            # 直接从模型边界提取
            xmin = model_bounds['xmin']
            xmax = model_bounds['xmax']
            logger.info(f"从模型提取 X 范围: [{xmin:.1f}, {xmax:.1f}]")
        
        # 确定 z 范围
        if 'zmin' in ray_params and 'zmax' in ray_params:
            # 用户明确指定了范围
            zmin = ray_params['zmin']
            zmax = ray_params['zmax']
            logger.info(f"使用用户指定的 Z 范围: [{zmin:.1f}, {zmax:.1f}]")
            # 验证是否与模型一致
            if abs(zmin - model_bounds['zmin']) > 0.01 or abs(zmax - model_bounds['zmax']) > 0.01:
                logger.warning(f"用户指定的 Z 范围与模型不一致！模型: [{model_bounds['zmin']:.1f}, {model_bounds['zmax']:.1f}], "
                             f"使用: [{zmin:.1f}, {zmax:.1f}]")
        else:
            # 直接从模型边界提取
            zmin = model_bounds['zmin']
            zmax = model_bounds['zmax']
            logger.info(f"从模型提取 Z 范围: [{zmin:.1f}, {zmax:.1f}]")
        
        # 验证炮点位置是否在模型范围内
        if not (xmin <= shot_position[0] <= xmax):
            logger.warning(f"炮点X位置 {shot_position[0]:.2f} 不在模型X范围 [{xmin:.1f}, {xmax:.1f}] 内")
        if not (zmin <= shot_position[1] <= zmax):
            logger.warning(f"炮点Z位置 {shot_position[1]:.2f} 不在模型Z范围 [{zmin:.1f}, {zmax:.1f}] 内")
        
        # 获取其他参数
        ray_values = ray_params.get('ray', [1.2])  # 默认：第1层反射
        nray_default = ray_params.get('nray', 5)  # 默认每个ray组的射线数量
        
        # 确保 ray_values 是列表
        if isinstance(ray_values, (int, float)):
            ray_values = [ray_values]
        elif not isinstance(ray_values, list):
            ray_values = [1.2]  # 默认值
        
        # nray 应该是数组，长度与 ray 数组相同
        # 如果用户只提供了一个nray值，则所有ray组使用相同的nray值
        if isinstance(nray_default, (int, float)):
            nray_array = [nray_default] * len(ray_values)
        elif isinstance(nray_default, list):
            # 如果用户提供了数组，使用它（可能需要填充或截断）
            if len(nray_default) < len(ray_values):
                # 填充：用最后一个值填充
                nray_array = nray_default + [nray_default[-1] if nray_default else 5] * (len(ray_values) - len(nray_default))
            else:
                # 截断：只取前len(ray_values)个值
                nray_array = nray_default[:len(ray_values)]
        else:
            nray_array = [5] * len(ray_values)
        
        # 格式化 ray 参数（RAYINVR 需要数组格式）
        # 标准格式：ray= 1.2,  2.1,  3.1（逗号分隔，可以跨多行）
        # 注意：ray 参数格式是 L.X，其中 L 是层号，X 是类型（1=折射，2=反射，3=首波）
        # 格式应该是 1.2, 2.1, 2.2 等，而不是 1.20, 2.10, 2.20
        # 使用 'g' 格式会自动去掉尾随的零
        def format_ray_value(r):
            """格式化 ray 值，确保 L.X 格式正确（如 1.2 而不是 1.20）"""
            # 使用 'g' 格式去掉尾随零，但保留小数点
            s = f"{r:g}"
            # 如果格式化为整数（如 1.0 -> 1），需要添加 .0
            if '.' not in s:
                s = f"{r:.1f}"
            return s
        
        def format_array_values(values, values_per_line=12, indent_spaces=16, is_integer=False):
            """格式化数组值，支持多行显示
            
            Args:
                values: 要格式化的值列表
                values_per_line: 每行最多显示的值数量
                indent_spaces: 后续行的缩进空格数
                is_integer: 如果为True，格式化为整数；否则使用format_ray_value格式化
            """
            if len(values) == 1:
                if is_integer:
                    return f" {int(values[0])}"
                else:
                    return f" {format_ray_value(values[0])}"
            
            if is_integer:
                formatted_list = [str(int(v)) for v in values]
            else:
                formatted_list = [format_ray_value(v) for v in values]
            
            # 第一行：值之间2个空格
            first_line_values = formatted_list[:values_per_line]
            first_line = f" {',  '.join(first_line_values)}"
            
            # 后续行：缩进对齐
            if len(formatted_list) > values_per_line:
                remaining_lines = []
                for i in range(values_per_line, len(formatted_list), values_per_line):
                    line_values = formatted_list[i:i+values_per_line]
                    remaining_lines.append(f"{' ' * indent_spaces}{',  '.join(line_values)}")
                return first_line + ',\n' + ',\n'.join(remaining_lines)
            else:
                return first_line
        
        ray_str = format_array_values(ray_values, is_integer=False)
        nray_str = format_array_values(nray_array, is_integer=True)
        
        # 生成最小化内容（使用用户指定的参数）
        # 注意：根据标准 r.in 文件格式（参考 d:\python-learn\pyAOBS\pyAOBS\modeling\r.in）：
        # 1. namelist 名称使用小写：&pltpar, &axepar, &trapar, &invpar
        # 2. 使用 &end 作为结束标记（不是单独的 &）
        # 3. 参数可以用逗号分隔在同一行或多行
        # 4. 参数格式：param=value 或 param=value,（注意等号前后可以有空格）
        # 5. 多行数组时，后续行需要缩进对齐
        # 6. 重要：trapar 中必须包含 imodf=1，才能从 v.in 文件读取速度模型
        #    （参考 main.f 第 218-226 行：if(imodf.ne.1) 则从 r.in 读取，否则从 v.in 读取）
        # 7. 重要：i2pt=0 表示不进行两点射线追踪，适用于理论走时计算
        #    （i2pt>0 用于连接特定的源和接收点，需要从 tx.in 读取接收点位置）
        # 8. 重要：itxout 在 pltpar 中，将计算得到的走时-距离对写入 tx.out。
        #    震相标识：itxout=1 用数组 ibreak；itxout=2 或 3 用数组 ivray。
        #    itxout=3 时将计算走时插值到观测接收点（需 invr=1，否则退化为 itxout=2）。默认 0。
        #    RAYINVR 文档注明：itxout 可能存在 bug，设置 itx=itxout 可保证正确结果，故设 itx=1。
        # 9. 重要：ishot=2 为默认值（炮点模式）；isrch=1（标准射线搜索方式）
        r_in_content = f"""  &pltpar  iplot=0, imod=1, iray=1, itxout=1, itx=1,
  &end
  &axepar  xmin={xmin:.1f}, xmax={xmax:.1f}, zmin={zmin:.1f}, zmax={zmax:.1f},
  &end
  &trapar  imodf=1, i2pt=0, ishot=2, isrch=1, xshot={shot_position[0]:.2f}, zshot={shot_position[1]:.2f},
           ray={ray_str},
           nray={nray_str},
           vfile="v.in",
  &end
  &invpar
  &end
"""
        
        # 写入文件 - 确保在模型文件所在目录生成
        # 优先使用模型文件所在目录，而不是working_dir
        model_dir = Path(self.model_file_path).parent if self.model_file_path else Path.cwd()
        model_dir.mkdir(parents=True, exist_ok=True)
        r_in_path = model_dir / 'r.in'
        with open(r_in_path, 'w') as f:
            f.write(r_in_content)
        
        logger.info(f"r.in 文件已生成在模型目录: {r_in_path}")
        return str(r_in_path)
    
    def generate_tx_in_from_picks(self, 
                                   pick_word: int = 1,
                                   shot_position: Optional[Tuple[float, float]] = None,
                                   phase_id: int = 1,
                                   uncertainty: float = 0.050) -> Optional[str]:
        """
        根据拾取数据生成符合 RAYINVR 格式要求的 tx.in 文件
        
        Args:
            pick_word: 使用的拾取字（默认 1）
            shot_position: 炮点位置 (x, z)，如果为None则从数据提取或使用默认值
            phase_id: 震相标识（默认 1，表示第一个震相）
            uncertainty: 走时误差/不确定性（默认 0.050 秒）
        
        Returns:
            tx.in 文件路径，如果没有拾取数据则返回None
        """
        if not self.pick_manager:
            return None
        
        # 1. 获取拾取数据
        picks = self.pick_manager.get_picks_by_word(pick_word)
        if not picks:
            return None
        
        # 2. 获取偏移距
        offsets = self._get_offsets_for_picks(picks)
        
        # 3. 确定炮点位置
        if shot_position is None:
            if self.data_loader:
                shot_pos = self._extract_shot_position() or (0.0, 0.0)
            else:
                shot_pos = (0.0, 0.0)
        else:
            shot_pos = shot_position
        
        # 4. 生成符合 RAYINVR 格式的 tx.in 文件
        tx_in_path = self._write_minimal_tx_in_file(
            picks, offsets, shot_pos, phase_id, uncertainty
        )
        
        return tx_in_path
    
    def _write_minimal_tx_in_file(self, 
                                   picks: Dict[int, float],
                                   offsets: Dict[int, float],
                                   shot_position: Tuple[float, float],
                                   phase_id: int = 1,
                                   uncertainty: float = 0.050) -> str:
        """
        写入符合 RAYINVR 格式要求的 tx.in 文件
        
        RAYINVR tx.in 格式要求（参考 main.f 第 308-309 行）：
        - 格式：format(3f10.3,i10) - 每行4列，固定宽度
        - 第1列：xpf (距离，10位，3位小数)
        - 第2列：tpf (走时，10位，3位小数)
        - 第3列：upf (误差/不确定性，10位，3位小数)
        - 第4列：ipf (震相标识，10位整数)
        
        文件结构：
        1. 每个炮点组以一行开始：xshot -1.000 0.000 0
        2. 接下来是多个观测走时点，格式：xpf tpf upf ipf
        3. 文件末尾以一行结束：0.000 0.000 0.000 -1
        
        Args:
            picks: 拾取字典 {trace_idx: pick_time}
            offsets: 偏移距字典 {trace_idx: offset_km}
            shot_position: 炮点位置 (x, z)
            phase_id: 震相标识（默认 1）
            uncertainty: 走时误差/不确定性（默认 0.050 秒）
        
        Returns:
            tx.in 文件路径
        """
        # 1. 生成符合 RAYINVR 格式的内容
        tx_in_lines = []
        
        # 1.1 炮点组开始标记（格式：xshot -1.000 0.000 0）
        # 注意：使用固定宽度格式，确保符合 format(3f10.3,i10)
        tx_in_lines.append(f"{shot_position[0]:10.3f}{-1.0:10.3f}{0.0:10.3f}{0:10d}")
        
        # 1.2 观测走时点（格式：xpf tpf upf ipf）
        # 按偏移距排序，确保数据有序
        sorted_items = sorted(picks.items(), key=lambda x: offsets.get(x[0], 0.0))
        for trace_idx, pick_time in sorted_items:
            offset = offsets.get(trace_idx, 0.0)
            # 使用固定宽度格式：3个浮点数（各10位，3位小数）+ 1个整数（10位）
            tx_in_lines.append(f"{offset:10.3f}{pick_time:10.3f}{uncertainty:10.3f}{phase_id:10d}")
        
        # 1.3 文件结束标记（格式：0.000 0.000 0.000 -1）
        tx_in_lines.append(f"{0.0:10.3f}{0.0:10.3f}{0.0:10.3f}{-1:10d}")
        
        tx_in_content = "\n".join(tx_in_lines) + "\n"
        
        # 2. 写入文件
        work_dir = Path(self.working_dir) if self.working_dir else Path(self.model_file_path).parent
        work_dir.mkdir(parents=True, exist_ok=True)
        tx_in_path = work_dir / 'tx.in'
        with open(tx_in_path, 'w') as f:
            f.write(tx_in_content)
        
        return str(tx_in_path)
    
    def read_tx_in_file(self, tx_in_path: str) -> Optional[Dict]:
        """
        读取符合 RAYINVR 格式的 tx.in 文件（参考 vedit 和 RAYINVR 实现）
        
        参考：
        - RAYINVR main.f 第 308-309 行：format(3f10.3,i10)
        - RAYINVR main.f 第 910-319 行：读取逻辑
        - vedit 通过 RayinvrWrapper.get_observed_data() 获取观测数据
        
        Args:
            tx_in_path: tx.in 文件路径
        
        Returns:
            包含观测走时数据的字典，格式为：
            {
                'shots': [
                    {
                        'shot_position': float,  # 炮点 X 坐标
                        'observations': [
                            {
                                'x': float,      # 距离 (km)
                                't': float,     # 走时 (s)
                                'u': float,     # 误差 (s)
                                'phase': int    # 震相标识
                            },
                            ...
                        ]
                    },
                    ...
                ]
            }
            如果文件不存在或格式错误，返回 None
        """
        tx_in_path = Path(tx_in_path)
        if not tx_in_path.exists():
            logger.warning(f"tx.in 文件不存在: {tx_in_path}")
            return None
        
        shots = []
        current_shot = None
        
        try:
            with open(tx_in_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    # 只去掉换行符，保留前导和中间的空格（固定宽度格式要求）
                    line = line.rstrip('\n\r')
                    if not line:  # 跳过空行
                        continue
                    
                    # 检查行长度（固定宽度格式要求每行至少40字符）
                    if len(line) < 40:
                        logger.warning(f"tx.in 第 {line_num} 行长度不足（应为至少40字符，实际为{len(line)}字符）: {line}")
                        continue
                    
                    # 解析固定宽度格式：format(3f10.3,i10)
                    # 每列10位宽度
                    try:
                        # 第1列：xpf (距离，10位，3位小数)
                        xpf_str = line[0:10].strip()
                        # 第2列：tpf (走时，10位，3位小数)
                        tpf_str = line[10:20].strip()
                        # 第3列：upf (误差，10位，3位小数)
                        upf_str = line[20:30].strip()
                        # 第4列：ipf (震相标识，10位整数)
                        ipf_str = line[30:40].strip()
                        
                        # 转换为数值
                        xpf = float(xpf_str) if xpf_str else 0.0
                        tpf = float(tpf_str) if tpf_str else 0.0
                        upf = float(upf_str) if upf_str else 0.0
                        ipf = int(ipf_str) if ipf_str else 0
                        
                    except (ValueError, IndexError) as e:
                        logger.warning(f"tx.in 第 {line_num} 行格式错误: {line}, 错误: {e}")
                        continue
                    
                    # 根据 ipf 值判断记录类型
                    if ipf == -1:
                        # 文件结束标记
                        break
                    elif ipf <= 0:
                        # 新的炮点位置（ipf=0 或负数）
                        # 保存之前的炮点（如果有）
                        if current_shot is not None:
                            shots.append(current_shot)
                        
                        # 创建新的炮点组
                        current_shot = {
                            'shot_position': xpf,  # xpf 是炮点 X 坐标
                            'observations': []
                        }
                    else:
                        # 观测走时点（ipf > 0）
                        if current_shot is None:
                            # 如果没有炮点标记，创建一个默认炮点
                            logger.warning(f"tx.in 第 {line_num} 行：观测点前没有炮点标记，使用默认炮点位置 0.0")
                            current_shot = {
                                'shot_position': 0.0,
                                'observations': []
                            }
                        
                        # 添加观测点
                        current_shot['observations'].append({
                            'x': xpf,      # 距离 (km)
                            't': tpf,      # 走时 (s)
                            'u': upf,      # 误差 (s)
                            'phase': ipf   # 震相标识
                        })
            
            # 保存最后一个炮点（如果有）
            if current_shot is not None:
                shots.append(current_shot)
            
            if not shots:
                logger.warning(f"tx.in 文件中没有找到有效的观测数据")
                return None
            
            return {
                'shots': shots,
                'total_observations': sum(len(shot['observations']) for shot in shots)
            }
            
        except Exception as e:
            logger.error(f"读取 tx.in 文件时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def read_tx_out_file(self, tx_out_path: Optional[str] = None) -> Optional[Dict]:
        """读取 RAYINVR 输出的理论走时文件 tx.out
        
        tx.out 文件格式与 tx.in 相同，都是 format(3f10.3,i10)
        
        Args:
            tx_out_path: tx.out 文件路径，如果为None则从工作目录查找
            
        Returns:
            包含理论走时数据的字典，格式为：
            {
                'shots': [
                    {
                        'shot_position': float,  # 炮点 X 坐标
                        'observations': [
                            {
                                'x': float,      # 距离 (km)
                                't': float,     # 走时 (s)
                                'u': float,     # 误差 (s，理论走时通常为0)
                                'phase': int    # 震相标识
                            },
                            ...
                        ]
                    },
                    ...
                ],
                'total_observations': int
            }
            如果文件不存在或格式错误，返回 None
        """
        if tx_out_path is None:
            work_dir = Path(self.working_dir) if self.working_dir else Path(self.model_file_path).parent
            tx_out_path = work_dir / 'tx.out'
        else:
            tx_out_path = Path(tx_out_path)
        
        if not tx_out_path.exists():
            logger.warning(f"tx.out 文件不存在: {tx_out_path}")
            return None
        
        # 复用 read_tx_in_file 的逻辑（格式相同）
        return self.read_tx_in_file(str(tx_out_path))
    
    def _read_and_output_theoretical_traveltime(self):
        """读取并输出理论走时数据（从 tx.out 文件）"""
        try:
            tx_out_data = self.read_tx_out_file()
            if tx_out_data is None:
                logger.warning("无法读取理论走时数据（tx.out 文件不存在或格式错误）")
                return
            
            total_obs = tx_out_data.get('total_observations', 0)
            shots = tx_out_data.get('shots', [])
            
            if total_obs == 0:
                logger.warning("tx.out 文件中没有理论走时数据")
                return
            
            logger.info("=" * 80)
            logger.info("理论走时计算结果（从 tx.out 文件读取）")
            logger.info("=" * 80)
            logger.info(f"总观测点数: {total_obs}")
            logger.info(f"炮点数量: {len(shots)}")
            
            # 输出每个炮点的理论走时
            for shot_idx, shot in enumerate(shots, 1):
                shot_pos = shot.get('shot_position', 0.0)
                observations = shot.get('observations', [])
                
                logger.info(f"\n炮点 {shot_idx}: X = {shot_pos:.3f} km")
                logger.info(f"  理论走时点数: {len(observations)}")
                
                # 输出前10个点和后10个点（如果数据点较多）
                if len(observations) <= 20:
                    # 输出所有点
                    logger.info("  距离(km)    走时(s)    震相")
                    logger.info("  " + "-" * 35)
                    for obs in observations:
                        logger.info(f"  {obs['x']:10.3f}  {obs['t']:10.3f}  {obs['phase']:5d}")
                else:
                    # 输出前10个和后10个点
                    logger.info("  距离(km)    走时(s)    震相")
                    logger.info("  " + "-" * 35)
                    for obs in observations[:10]:
                        logger.info(f"  {obs['x']:10.3f}  {obs['t']:10.3f}  {obs['phase']:5d}")
                    logger.info(f"  ... (省略 {len(observations) - 20} 个点) ...")
                    for obs in observations[-10:]:
                        logger.info(f"  {obs['x']:10.3f}  {obs['t']:10.3f}  {obs['phase']:5d}")
                
                # 统计信息
                if observations:
                    distances = [obs['x'] for obs in observations]
                    times = [obs['t'] for obs in observations]
                    logger.info(f"\n  距离范围: [{min(distances):.3f}, {max(distances):.3f}] km")
                    logger.info(f"  走时范围: [{min(times):.3f}, {max(times):.3f}] s")
            
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"读取并输出理论走时数据时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def validate_tx_in_file(self, tx_in_path: str) -> Tuple[bool, Optional[str]]:
        """
        验证 tx.in 文件格式是否符合 RAYINVR 要求
        
        Args:
            tx_in_path: tx.in 文件路径
        
        Returns:
            (is_valid, error_message)
            - is_valid: 文件格式是否有效
            - error_message: 如果无效，返回错误信息；否则返回 None
        """
        tx_in_path = Path(tx_in_path)
        if not tx_in_path.exists():
            return False, f"文件不存在: {tx_in_path}"
        
        try:
            data = self.read_tx_in_file(tx_in_path)
            if data is None:
                return False, "文件格式错误或没有有效的观测数据"
            
            # 检查是否有观测数据
            if data['total_observations'] == 0:
                return False, "文件中没有观测走时数据"
            
            return True, None
            
        except Exception as e:
            return False, f"验证文件时出错: {e}"
