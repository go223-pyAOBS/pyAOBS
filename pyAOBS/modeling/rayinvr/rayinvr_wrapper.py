"""
简化版Python-RAYINVR接口
主要功能:
1. 运行run_rayinvr_main()完成所有rayinvr计算
2. 获取射线路径数据
"""
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
from pathlib import Path
import os
import time
import shutil

class RayinvrWrapper:
    """RAYINVR Fortran程序的Python封装类"""
    
    def __init__(self, working_dir=None):
        """初始化wrapper，加载librayinvr.so库
        
        Parameters
        ----------
        working_dir : str, optional
            工作目录路径，如果为None则使用当前目录
        """
        # 获取.so文件路径
        current_dir = Path(__file__).parent
        lib_path = current_dir / "librayinvr.so"
        # print(f"正在从以下位置加载RAYINVR库: {lib_path}")
        
        # 加载动态库
        try:
            self.lib = ctypes.CDLL(str(lib_path))
            # print("RAYINVR库加载成功")
        except Exception as e:
            print(f"加载RAYINVR库失败: {e}")
            raise
        
        # 设置函数参数类型
        self._setup_function_signatures()
        
        # 保存原始工作目录
        self.original_dir = os.getcwd()
        
        # 设置工作目录
        self.working_dir = working_dir if working_dir else self.original_dir
        # print(f"设置工作目录为: {self.working_dir}")
        
    def _setup_function_signatures(self):
        """设置Fortran函数的参数类型"""
        
        # 运行主程序 - 完成所有计算
        self.lib.run_rayinvr_main_.argtypes = []
        self.lib.run_rayinvr_main_.restype = None
        
        # 获取射线路径数据
        self.lib.get_ray_paths_.argtypes = [
            ctypes.POINTER(ctypes.c_int),    # ray_idx
            ndpointer(dtype=np.float32),     # x_array
            ndpointer(dtype=np.float32),     # z_array
            ctypes.POINTER(ctypes.c_int),    # n_points
            ctypes.POINTER(ctypes.c_int)     # max_points
        ]
        self.lib.get_ray_paths_.restype = None
        
        # 获取射线角度信息
        self.lib.get_ray_angles_.argtypes = [
            ctypes.POINTER(ctypes.c_int),    # ray_idx
            ctypes.POINTER(ctypes.c_float),  # initial_angle
            ctypes.POINTER(ctypes.c_float),  # final_angle
            ctypes.POINTER(ctypes.c_int),    # shot_num
            ctypes.POINTER(ctypes.c_int)     # ray_num
        ]
        self.lib.get_ray_angles_.restype = None
        
        # 获取走时数据
        self.lib.get_travel_times_.argtypes = [
            ndpointer(dtype=np.float32),     # x_array
            ndpointer(dtype=np.float32),     # t_array
            ctypes.POINTER(ctypes.c_int),    # n_points
            ctypes.POINTER(ctypes.c_int)     # max_points
        ]
        self.lib.get_travel_times_.restype = None
        
        # 获取射线数量
        self.lib.get_ray_count_.argtypes = [
            ctypes.POINTER(ctypes.c_int)     # n_rays
        ]
        self.lib.get_ray_count_.restype = None
        
        # 获取指定射线
        self.lib.get_stored_ray_.argtypes = [
            ctypes.POINTER(ctypes.c_int),    # ray_idx
            ndpointer(dtype=np.float32),     # x_array
            ndpointer(dtype=np.float32),     # z_array
            ndpointer(dtype=np.float32),     # t_array
            ctypes.POINTER(ctypes.c_int),    # n_points
            ctypes.POINTER(ctypes.c_int)     # max_points
        ]
        self.lib.get_stored_ray_.restype = None
        
        # 获取射线走时
        self.lib.get_ray_time_.argtypes = [
            ctypes.POINTER(ctypes.c_int),    # ray_idx
            ctypes.POINTER(ctypes.c_float)   # travel_time
        ]
        self.lib.get_ray_time_.restype = None
        
        # 获取观测走时数据
        self.lib.get_observed_data_.argtypes = [
            ctypes.POINTER(ctypes.c_int),     # nobs
            ndpointer(dtype=np.float32),      # x_array
            ndpointer(dtype=np.float32),      # t_array
            ndpointer(dtype=np.float32),      # u_array
            ndpointer(dtype=np.int32)         # phase_array
        ]
        self.lib.get_observed_data_.restype = None
    
    def _update_v_in(self):
        """确保v.in文件被正确更新"""
        try:
            v_in_path = os.path.join(self.working_dir, 'v.in')
            if not os.path.exists(v_in_path):
                print(f"错误：找不到v.in文件: {v_in_path}")
                return False
            
            # 检查文件是否被修改
            try:
                with open(v_in_path, 'r') as f:
                    content = f.read()
                # 如果文件内容为空，说明文件可能没有被正确更新
                if not content.strip():
                    print("错误：v.in文件内容为空")
                    return False
            except Exception as e:
                print(f"读取v.in文件时出错: {e}")
                return False
            
            # 确保文件权限正确
            try:
                os.chmod(v_in_path, 0o644)
                # print(f"已设置v.in文件的权限为644")
            except Exception as e:
                print(f"设置v.in文件权限时出错: {e}")
            
            return True
        except Exception as e:
            print(f"更新v.in文件时出错: {e}")
            return False

    def run_rayinvr(self):
        """运行RAYINVR主程序"""
        try:
            # 重新加载Fortran库
            current_dir = Path(__file__).parent
            lib_path = current_dir / "librayinvr.so"
            import ctypes  # 确保ctypes在正确的作用域中
            self.lib = ctypes.CDLL(str(lib_path))
            self._setup_function_signatures()
            
            # 保存当前目录
            original_dir = os.getcwd()
            
            try:
                # 设置工作目录
                if self.working_dir:
                    os.chdir(self.working_dir)
                    print(f"切换到工作目录: {self.working_dir}")
                else:
                    os.chdir(str(current_dir))
                    print(f"使用默认目录: {current_dir}")
                
                # 运行主程序
                print("开始运行RAYINVR主程序...")
                self.lib.run_rayinvr_main_()
                print("RAYINVR主程序运行完成")
                
                return True
                
            finally:
                # 恢复原始目录
                os.chdir(original_dir)
                print(f"恢复原始目录: {original_dir}")
            
        except Exception as e:
            print(f"运行RAYINVR主程序时出错: {e}")
            return False
    
    def get_ray_paths(self, ray_idx=0):
        """获取指定射线的路径
        
        Parameters
        ----------
        ray_idx : int, optional
            射线索引，默认为0
            
        Returns
        -------
        dict
            包含射线路径的字典，格式为：
            {
                ray_idx: {
                    'shot_num': int,
                    'ray_num': int,
                    'distance': float,
                    'depth': float,
                    'time': float,
                    'points': [(x1, z1), (x2, z2), ...],
                    'initial_angle': float,
                    'final_angle': float
                }
            }
        """
        try:
            # 获取射线数量
            n_rays = ctypes.c_int()
            self.lib.get_ray_count_(ctypes.byref(n_rays))
            
            if ray_idx < 0 or ray_idx >= n_rays.value:
                print(f"警告：射线索引 {ray_idx} 超出范围 [0, {n_rays.value-1}]")
                return {}
            
            # 分配数组
            max_points = 1000  # 最大点数
            x_array = np.zeros(max_points, dtype=np.float32)
            z_array = np.zeros(max_points, dtype=np.float32)
            n_points = ctypes.c_int()
            
            # 获取射线路径
            self.lib.get_ray_paths_(
                ctypes.byref(ctypes.c_int(ray_idx)),
                x_array,
                z_array,
                ctypes.byref(n_points),
                ctypes.byref(ctypes.c_int(max_points))
            )
            
            # 获取射线角度信息
            initial_angle = ctypes.c_float()
            final_angle = ctypes.c_float()
            shot_num = ctypes.c_int()
            ray_num = ctypes.c_int()
            
            try:
                self.lib.get_ray_angles_(
                    ctypes.byref(ctypes.c_int(ray_idx)),
                    ctypes.byref(initial_angle),
                    ctypes.byref(final_angle),
                    ctypes.byref(shot_num),
                    ctypes.byref(ray_num)
                )
            except Exception as e:
                print(f"警告：获取射线角度信息失败: {e}")
                initial_angle.value = 0.0
                final_angle.value = 0.0
                shot_num.value = ray_idx
                ray_num.value = 1
            
            # 获取射线走时
            travel_time = ctypes.c_float()
            try:
                self.lib.get_ray_time_(
                    ctypes.byref(ctypes.c_int(ray_idx)),
                    ctypes.byref(travel_time)
                )
            except Exception as e:
                print(f"警告：获取射线走时失败: {e}")
                travel_time.value = 0.0
            
            # 构建返回数据
            points = list(zip(x_array[:n_points.value], z_array[:n_points.value]))
            if not points:
                print(f"警告：射线 {ray_idx} 没有路径点")
                return {}
            
            return {
                ray_idx: {
                    'shot_num': shot_num.value,
                    'ray_num': ray_num.value,
                    'distance': x_array[n_points.value-1],
                    'depth': z_array[n_points.value-1],
                    'time': travel_time.value,
                    'points': points,
                    'initial_angle': initial_angle.value,
                    'final_angle': final_angle.value
                }
            }
            
        except Exception as e:
            print(f"获取射线路径时出错: {e}")
            return {}
    
    def get_travel_times(self):
        """获取走时数据
        
        Returns
        -------
        dict
            包含走时数据的字典
        """
        try:
            # 分配内存用于接收数据
            max_points = 1000  # 足够大的缓冲区
            x_array = np.zeros(max_points, dtype=np.float32)
            t_array = np.zeros(max_points, dtype=np.float32)
            n_points = ctypes.c_int(0)
            
            # 调用Fortran函数获取走时数据
            self.lib.get_travel_times_(
                x_array,
                t_array,
                ctypes.byref(n_points),
                ctypes.byref(ctypes.c_int(max_points))
            )
            
            # 返回结果
            if n_points.value == 0:
                print("警告：没有找到走时数据")
                return None
                
            return {
                'distance': x_array[:n_points.value].copy(),
                'time': t_array[:n_points.value].copy()
            }
        except Exception as e:
            print(f"获取走时数据时出错: {e}")
            return None
    
    def get_ray_count(self):
        """获取存储的射线数量"""
        n_rays = ctypes.c_int(0)
        self.lib.get_ray_count_(ctypes.byref(n_rays))
        return n_rays.value

    def get_stored_ray(self, ray_idx=1):
        """获取指定索引的存储射线
        
        Parameters
        ----------
        ray_idx : int, optional
            射线索引（从1开始），默认为1
            
        Returns
        -------
        dict
            包含射线路径和走时的字典
        """
        try:
            # print(f"\nGetting stored ray {ray_idx}...")
            
            # 检查索引有效性
            if ray_idx < 1:
                print(f"Error: Invalid ray index {ray_idx}")
                return None
            
            # 分配内存
            max_points = 2000  # 足够大的缓冲区
            x_array = np.zeros(max_points, dtype=np.float32)
            z_array = np.zeros(max_points, dtype=np.float32)
            t_array = np.zeros(max_points, dtype=np.float32)
            n_points = ctypes.c_int(0)
            travel_time = ctypes.c_float(0.0)
            
            # 获取射线路径
            try:
                self.lib.get_stored_ray_(
                    ctypes.byref(ctypes.c_int(ray_idx)),
                    x_array,
                    z_array,
                    t_array,
                    ctypes.byref(n_points),
                    ctypes.byref(ctypes.c_int(max_points))
                )
            except Exception as e:
                print(f"Error getting ray path: {str(e)}")
                return None
            
            # 检查点数
            if n_points.value <= 0 or n_points.value > max_points:
                print(f"Warning: Invalid number of points ({n_points.value}) for ray {ray_idx}")
                return None
            
            # 获取射线走时
            try:
                self.lib.get_ray_time_(
                    ctypes.byref(ctypes.c_int(ray_idx)),
                    ctypes.byref(travel_time)
                )
            except Exception as e:
                print(f"Warning: Error getting ray time: {str(e)}")
                travel_time.value = 0.0
            
            # 验证数据
            if (np.any(np.isnan(x_array[:n_points.value])) or 
                np.any(np.isnan(z_array[:n_points.value])) or 
                np.any(np.isnan(t_array[:n_points.value]))):
                print(f"Warning: NaN values found in ray {ray_idx}")
                return None
            
            # 构建返回数据
            ray_data = {
                'x': x_array[:n_points.value].copy(),
                'z': z_array[:n_points.value].copy(),
                'time': t_array[:n_points.value].copy(),
                'total_time': travel_time.value,
                'npoints': n_points.value
            }
            
            # print(f"Successfully retrieved ray {ray_idx} with {n_points.value} points")
            return ray_data
            
        except Exception as e:
            print(f"Error in get_stored_ray: {str(e)}")
            return None

    def get_all_rays(self, max_rays=100):
        """获取所有存储的射线
        
        Parameters
        ----------
        max_rays : int, optional
            最大返回射线数，默认为100
            
        Returns
        -------
        list
            包含所有射线的列表
        """
        try:
            # print("\n========== Getting All Rays ==========")
            n_rays = self.get_ray_count()
            # print(f"Total rays found: {n_rays}")
            
            if n_rays == 0:
                print("警告：没有找到存储的射线")
                return []
            
            rays = []
            for i in range(1, min(n_rays+1, max_rays+1)):
                try:
                    # print(f"Getting ray {i}...")
                    ray = self.get_stored_ray(i)
                    if ray is not None and ray['npoints'] > 0:
                        # 验证数据有效性
                        if (len(ray['x']) == ray['npoints'] and 
                            len(ray['z']) == ray['npoints'] and 
                            len(ray['time']) == ray['npoints']):
                            rays.append(ray)
                            # print(f"Successfully added ray {i} with {ray['npoints']} points")
                        else:
                            print(f"Warning: Invalid data for ray {i}")
                    else:
                        print(f"Warning: No data for ray {i}")
                except Exception as e:
                    print(f"Error getting ray {i}: {str(e)}")
                    continue
            
            # print(f"Successfully retrieved {len(rays)} rays")
            # print("========== Getting All Rays Completed ==========\n")
            return rays
            
        except Exception as e:
            print(f"Error in get_all_rays: {str(e)}")
            return []

    def get_observed_data(self):
        """获取观测走时数据
        
        Returns
        -------
        dict
            包含观测走时数据的字典，包括：
            - x: 距离数组
            - t: 走时数组
            - u: 误差数组
            - phase: 相位标识符数组
        """
        # 分配内存用于接收数据
        max_obs = 10000  # 足够大的缓冲区
        x_array = np.zeros(max_obs, dtype=np.float32)
        t_array = np.zeros(max_obs, dtype=np.float32)
        u_array = np.zeros(max_obs, dtype=np.float32)
        phase_array = np.zeros(max_obs, dtype=np.int32)
        nobs = ctypes.c_int(0)
        
        # 调用Fortran函数获取观测数据
        self.lib.get_observed_data_(
            ctypes.byref(nobs),
            x_array,
            t_array,
            u_array,
            phase_array
        )
        
        if nobs.value == 0:
            print("警告：没有找到观测数据")
            return None
            
        return {
            'x': x_array[:nobs.value].copy(),
            't': t_array[:nobs.value].copy(),
            'u': u_array[:nobs.value].copy(),
            'phase': phase_array[:nobs.value].copy()
        }
