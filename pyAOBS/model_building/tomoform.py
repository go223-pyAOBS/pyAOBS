"""
tomoform.py - Python implementation of velocity model manipulation tools

This module provides classes and functions for handling 2D velocity models,
including reading/writing model files, editing velocity structures, and 
generating mesh grids. It is designed to work with the pyAOBS visualization tools.

Author: Haibo Huang
Date: 2025
"""


import numpy as np
from typing import Optional, Tuple, List, Union
from pathlib import Path
import xarray as xr
from scipy.interpolate import interp1d, griddata
from scipy.ndimage import gaussian_filter

class SlownessMesh2D:
    """2D slowness mesh class"""
    
    def __init__(self, nx: int, nz: int, v_water: float, v_air: float):
        """Initialize mesh
        
        Args:
            nx: Number of horizontal points
            nz: Number of vertical points 
            v_water: Water velocity
            v_air: Air velocity
        """
        self.nx = nx
        self.nz = nz
        self.v_water = v_water
        self.v_air = v_air
        
        # Initialize arrays
        self.xpos = np.zeros(nx)
        self.zpos = np.zeros(nz)
        self.topo = np.zeros(nx)
        self.z = np.zeros(nz)
        self.vgrid = np.ones((nx, nz)) * v_water
        self.pgrid = 1.0 / self.vgrid
        
    @classmethod
    def from_file(cls, filename: str) -> 'SlownessMesh2D':
        """Read mesh from smesh format file（与 tomo2d ``smesh.cc`` / readme 3.1 一致）。

        行顺序：① ``nx nz v_water v_air``；② ``nx`` 个 ``xpos``；③ ``nx`` 个 ``topo``（水深/
        海底深度，km，与 C 端一致）；④ ``nz`` 个 ``zpos``（相对海底的深度节点）；随后 ``nx``
        行 ``nz`` 列 ``vgrid``。
        """
        with open(filename) as f:
            # Read header
            nx, nz, v_water, v_air = map(float, f.readline().split())
            nx, nz = int(nx), int(nz)
            
            # Create mesh object
            mesh = cls(nx, nz, v_water, v_air)
            
            # Read coordinates
            mesh.xpos = np.array(list(map(float, f.readline().split())))
            mesh.topo = np.array(list(map(float, f.readline().split())))
            mesh.zpos = np.array(list(map(float, f.readline().split())))  # 相对于地形的深度值
            
            # 正确处理广播
            mesh.z = mesh.zpos[:, np.newaxis] + mesh.topo[np.newaxis, :]
            
            # Read velocity grid
            mesh.vgrid = np.zeros((nx, nz))
            for i in range(nx):
                mesh.vgrid[i,:] = list(map(float, f.readline().split()))
                        
            mesh.pgrid = 1.0 / mesh.vgrid
            return mesh
            
    def to_file(self, filename: str):
        """Write mesh to smesh format file
        
        Args:
            filename: Output file path
        """
        with open(filename, 'w') as f:
            # Write header
            f.write(f"{self.nx} {self.nz} {self.v_water} {self.v_air}\n")
            
            # Write coordinates
            f.write(" ".join(map(str, self.xpos)) + "\n")
            f.write(" ".join(map(str, self.topo)) + "\n") 
            f.write(" ".join(map(str, self.zpos)) + "\n")
            
            # Write velocity grid
            for i in range(self.nx):
                f.write(" ".join(map(str, self.vgrid[i,:])) + "\n")
    
    def gaussian_smooth(self, Lh: float, Lv: float):
        """Apply Gaussian smoothing to velocity field below topo
        

        Args:
            Lh: Horizontal correlation length
            Lv: Vertical correlation length
        """
        # Convert correlation lengths to sigma values for gaussian_filter
        sigma_h = Lh / (2 * np.sqrt(2 * np.log(2)))
        sigma_v = Lv / (2 * np.sqrt(2 * np.log(2)))
        
        # 保存原始的水层和空气层速度
        original_vgrid = self.vgrid.copy()
        
        # 应用平滑
        self.vgrid = gaussian_filter(self.vgrid, sigma=[sigma_v, sigma_h], mode='reflect')
        self.pgrid = 1.0 / self.vgrid
    
    def add_checkerboard(self, 
                        amplitude: float,
                        ch: float,
                        cv: float):
        """Add checkerboard pattern to velocity field below topo
        
        Args:
            amplitude: Amplitude in percent
            ch: Horizontal wavelength
            cv: Vertical wavelength
        """
        x, z = np.meshgrid(self.xpos, self.z, indexing='ij')
        pattern = amplitude * 0.01 * np.sin(2*np.pi*x/ch) * np.sin(2*np.pi*z/cv)
        self.vgrid *= (1.0 + pattern)
        self.pgrid = 1.0 / self.vgrid

        
    def add_anomaly(self,
                   amplitude: float,
                   xmin: float,
                   xmax: float, 
                   zmin: float,
                   zmax: float):
        """Add rectangular velocity anomaly below topo
        
        Args:
            amplitude: Amplitude in percent
            xmin, xmax: X coordinate range
            zmin, zmax: Z coordinate range relative to topo
        """
        x, z = np.meshgrid(self.xpos, self.z, indexing='ij')
        # 只在地形以下且在指定范围内添加异常
        mask = ((x >= xmin) & (x <= xmax) & 
               (z >= zmin) & (z <= zmax))  


        self.vgrid[mask] *= (1.0 + amplitude * 0.01)
        self.pgrid = 1.0 / self.vgrid
        
    def add_gaussian(self,
                    amplitude: float,
                    x0: float,
                    z0: float,
                    Lh: float, 
                    Lv: float):
        """Add Gaussian anomaly below topo
        
        Args:
            amplitude: Amplitude in percent
            x0: Center x coordinate
            z0: Center z coordinate relative to topo
            Lh: Horizontal correlation length
            Lv: Vertical correlation length
        """
        x, z = np.meshgrid(self.xpos, self.z, indexing='ij')
        r2 = ((x - x0)/Lh)**2 + ((z - z0)/Lv)**2
        pattern = amplitude * 0.01 * np.exp(-r2)
        self.vgrid *= (1.0 + pattern)
        self.pgrid = 1.0 / self.vgrid
        

    def to_xarray(self, dx: float = None, dz: float = None) -> xr.Dataset:
        """将模型转换为 xarray 数据集。

        Args:
            dx (float, optional): x方向的采样间隔（km）。如果不指定，使用原始间隔。
            dz (float, optional): z方向的采样间隔（km）。如果不指定，使用原始间隔。

        Returns:
            xr.Dataset: 包含速度场的数据集。
        """
        # 计算原始深度间隔
        dz_orig = self.zpos[1] - self.zpos[0]
        dx_orig = self.xpos[1] - self.xpos[0]
        
        # 对每个水平位置计算深度范围
        max_depth = float('-inf')
        min_height = float('inf')
        
        for i in range(self.nx):
            # 计算该位置的深度范围
            if self.topo[i] < 0:  # 地形在海平面以上
                # 空气层从地形上方1km开始
                curr_min = self.topo[i] - 1.0
                # 最大深度是相对于地形的深度加上地形高度
                curr_max = self.zpos[-1] + self.topo[i]
            else:  # 地形在海平面以下
                # 空气层从海平面上方1km开始
                curr_min = -1.0
                # 最大深度是相对于地形的深度加上地形高度
                curr_max = self.zpos[-1] + self.topo[i]
            
            max_depth = max(max_depth, curr_max)
            min_height = min(min_height, curr_min)
        
        # 生成统一的深度节点，使用指定的采样间隔或原始间隔
        dz = dz if dz is not None else dz_orig
        dx = dx if dx is not None else dx_orig
        
        full_zpos = np.arange(min_height, max_depth + dz, dz)
        x_new = np.arange(self.xpos[0], self.xpos[-1] + dx, dx)
        
        nz_full = len(full_zpos)
        nx_new = len(x_new)
        
        # 创建新的速度网格
        full_vgrid = np.ones((nx_new, nz_full))
        
        # 对每个水平位置填充速度值
        for i, x in enumerate(x_new):
            # 找到最近的原始x坐标索引
            ix = np.searchsorted(self.xpos, x)
            if ix == len(self.xpos):
                ix = len(self.xpos) - 1
            elif ix > 0 and (x - self.xpos[ix-1]) < (self.xpos[ix] - x):
                ix = ix - 1
                
            # 获取该位置的地形值（线性插值）
            if ix == len(self.xpos) - 1:
                topo_val = self.topo[ix]
            else:
                x1, x2 = self.xpos[ix], self.xpos[ix + 1]
                t1, t2 = self.topo[ix], self.topo[ix + 1]
                topo_val = t1 + (x - x1) * (t2 - t1) / (x2 - x1)
            
            for j, z in enumerate(full_zpos):
                actual_z = z  # 现在z已经是实际深度
                if actual_z < 0:  # 海平面以上
                    full_vgrid[i,j] = self.v_air
                elif 0 <= actual_z <= topo_val:  # 水层
                    full_vgrid[i,j] = self.v_water
                else:  # 地下
                    # 计算相对于地形的深度
                    rel_z = actual_z - topo_val
                    if rel_z in self.zpos:
                        k = np.where(self.zpos == rel_z)[0][0]
                        full_vgrid[i,j] = self.vgrid[ix,k]
                    else:
                        # 线性插值
                        k = np.searchsorted(self.zpos, rel_z)
                        if k == 0:
                            full_vgrid[i,j] = self.vgrid[ix,0]
                        elif k == len(self.zpos):
                            full_vgrid[i,j] = self.vgrid[ix,-1]
                        else:
                            z1, z2 = self.zpos[k-1], self.zpos[k]
                            v1, v2 = self.vgrid[ix,k-1], self.vgrid[ix,k]
                            full_vgrid[i,j] = v1 + (v2-v1)*(rel_z-z1)/(z2-z1)
        
        # 转置以匹配xarray格式
        full_vgrid_t = full_vgrid.T
        
        return xr.Dataset(
            data_vars={
                'velocity': (('z', 'x'), full_vgrid_t),
                'slowness': (('z', 'x'), 1.0/full_vgrid_t),
                'topo': ('x', np.interp(x_new, self.xpos, self.topo)),
                'v_water': self.v_water,
                'v_air': self.v_air
            },
            coords={
                'x': x_new,
                'z': full_zpos
            },
            attrs={
                'description': 'Velocity model with air and water layers'
            }
        )


class VelocityModelGenerator:
    """Class for generating velocity models"""
    
    @staticmethod
    def uniform_gradient(nx: int,
                        nz: int,
                        xmax: float,
                        zmax: float,
                        v0: float,
                        gradient: float,
                        v_water: float = 1.5,
                        v_air: float = 0.33) -> SlownessMesh2D:
        """Generate model with uniform velocity gradient
        
        Args:
            nx: Number of horizontal points
            nz: Number of vertical points
            xmax: Maximum x coordinate
            zmax: Maximum z coordinate
            v0: Surface velocity
            gradient: Velocity gradient
            v_water: Water velocity
            v_air: Air velocity
            
        Returns:
            SlownessMesh2D object
        """
        mesh = SlownessMesh2D(nx, nz, v_water, v_air)
        
        # Generate coordinates
        mesh.xpos = np.linspace(0, xmax, nx)
        mesh.zpos = np.linspace(0, zmax, nz)
        mesh.topo = np.zeros(nx)

        # Generate velocity field
        z = np.tile(mesh.zpos, (nx, 1))
        mesh.vgrid = v0 + gradient * z
        mesh.pgrid = 1.0 / mesh.vgrid
        
        return mesh
    
    @staticmethod
    def from_interfaces(interfaces: List[Tuple[np.ndarray, np.ndarray, float]],
                       nx: int,
                       nz: int,
                       xmax: float,
                       zmax: float,
                       v_water: float = 1.5,
                       v_air: float = 0.33) -> SlownessMesh2D:
        """Generate model from interface definitions
        
        Args:
            interfaces: List of (x, z, velocity) tuples defining interfaces
            nx: Number of horizontal points
            nz: Number of vertical points
            xmax: Maximum x coordinate
            zmax: Maximum z coordinate
            v_water: Water velocity
            v_air: Air velocity
            
        Returns:
            SlownessMesh2D object
        """
        mesh = SlownessMesh2D(nx, nz, v_water, v_air)
        
        # Generate coordinates
        mesh.xpos = np.linspace(0, xmax, nx)
        mesh.zpos = np.linspace(0, zmax, nz)
        mesh.topo = np.zeros(nx)
        
        # Initialize velocity grid
        x, z = np.meshgrid(mesh.xpos, mesh.zpos, indexing='ij')
        mesh.vgrid = np.ones_like(x) * v_water
        
        # Add interfaces
        for interface_x, interface_z, velocity in interfaces:
            # Interpolate interface
            f = interp1d(interface_x, interface_z, 
                        bounds_error=False, fill_value='extrapolate')
            z_int = f(mesh.xpos)
            
            # Set velocities below interface
            for i in range(nx):
                mask = mesh.zpos >= z_int[i]
                mesh.vgrid[i,mask] = velocity
                
        mesh.pgrid = 1.0 / mesh.vgrid
        return mesh


def load_tomo2d_interface_file(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """读取 tomo2d ``Interface2d`` / ``-F`` 反射界面文本：每行 ``x z``（可含空行与 ``#`` 注释）。

    与 ``interface.cc`` 中自文件读入的约定一致：x 须严格递增，至少 2 个节点。
    """
    xs: List[float] = []
    zs: List[float] = []
    with open(filename, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            xs.append(float(parts[0]))
            zs.append(float(parts[1]))
    if len(xs) < 2:
        raise ValueError(f"反射界面文件至少需要 2 个有效节点: {filename!r}")
    x = np.asarray(xs, dtype=float)
    z = np.asarray(zs, dtype=float)
    if np.any(np.diff(x) <= 0):
        raise ValueError(f"反射界面文件的 x 坐标须严格递增: {filename!r}")
    return x, z


def plot_velocity_model(smesh_file: str,
                         output_dir: Optional[str] = None,
                         refl_file: Optional[str] = None,
                         **kwargs) -> Tuple[SlownessMesh2D, str]:
    """Create velocity model and plot it
    
    Args:
        smesh_file: Input smesh format file
        output_dir: Output directory for plots
        refl_file: 可选，tomo2d ``-F`` 反射界面文件（每行 x z），叠加绘于速度剖面上
        **kwargs: Additional arguments passed to GridModelVisualizer.plot_xarray:
            - cmap: 内置名（如 ``viridis``）或 **.cpt 文件绝对路径**（存在时由 ``load_cpt`` 解析）
            - figsize: 英寸 (宽, 高)，默认 (10, 5.0)，与 GUI 嵌入图一致；可覆盖
            - plot_interfaces: bool, whether to plot interfaces (default: True)
            - interface_color: str or list, color(s) for interface lines (default: 'black')
            - interface_linewidth: float or list, width(s) of interface lines (default: 1.0)
            - interface_linestyle: str or list, style(s) of interface lines (default: '-')
            - extra_interfaces: 可选，覆盖/追加自定义界面曲线列表（见 show_model.plot_xarray）
            
    Returns:
        Tuple of (SlownessMesh2D object, plot file path)
    """
    # Read model
    mesh = SlownessMesh2D.from_file(smesh_file)
    
    try:
        from pyAOBS.visualization.show_model import GridModelVisualizer
    except ImportError:
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from pyAOBS.visualization.show_model import GridModelVisualizer
        
    if output_dir is None:
        output_dir = Path(smesh_file).parent
    else:
        output_dir = Path(output_dir)
            
    plot_file = str(output_dir / 'velocity_model.png')
        
    # Convert to xarray and plot
    ds = mesh.to_xarray()

    extra_list: Optional[List[dict]] = kwargs.pop("extra_interfaces", None)
    if refl_file:
        rx, rz = load_tomo2d_interface_file(refl_file)
        refl_entry = {
            "x": rx,
            "z": rz,
            "label": Path(refl_file).name,
            "color": "crimson",
            "linewidth": 1.8,
            "linestyle": "--",
        }
        if extra_list is None:
            extra_list = [refl_entry]
        else:
            extra_list = list(extra_list) + [refl_entry]
    
    # 设置默认参数（figsize 与 tomo2d_gui 嵌入预览一致，导出图偏「扁」）
    plot_params = {
        "figsize": (10, 5.0),
        'plot_interfaces': True,
        'interface_color': 'black',
        'interface_linewidth': 1.0,
        'interface_linestyle': '-',
        'model': mesh,  # 传入模型实例以绘制界面
        'extra_interfaces': extra_list,
    }
    
    # 更新用户提供的参数
    plot_params.update(kwargs)
    
    viz = GridModelVisualizer(output_dir=str(output_dir))
    viz.plot_xarray(
        plot_file,
        data=ds,
        title='Velocity Model',
        colorbar_label='Velocity (km/s)',
        **plot_params
    )
    
    return mesh, plot_file 