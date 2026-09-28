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


def _pgrid_from_v(vgrid) -> np.ndarray:
    """速度 → 慢度。v=0（如百分异常场）处为 +inf，不触发 divide-by-zero 警告。"""
    v = np.asarray(vgrid, dtype=float)
    p = np.full(v.shape, np.inf, dtype=float)
    np.divide(1.0, v, out=p, where=np.abs(v) > 0)
    return p


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
        self.pgrid = _pgrid_from_v(self.vgrid)
        
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
                        
            mesh.pgrid = _pgrid_from_v(mesh.vgrid)
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
        self.pgrid = _pgrid_from_v(self.vgrid)
    
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
        self.pgrid = _pgrid_from_v(self.vgrid)

        
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
        self.pgrid = _pgrid_from_v(self.vgrid)
        
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
        self.pgrid = _pgrid_from_v(self.vgrid)
        

    def _regular_plot_axes(self, dx: float = None, dz: float = None):
        """``to_xarray`` 用的规则 (x, z) 轴，与历史 ``np.arange`` 边界一致。"""
        dz_orig = self.zpos[1] - self.zpos[0]
        dx_orig = self.xpos[1] - self.xpos[0]
        dz = dz_orig if dz is None else dz
        dx = dx_orig if dx is None else dx
        topo = np.asarray(self.topo, dtype=float)
        curr_min = np.where(topo < 0.0, topo - 1.0, -1.0)
        curr_max = float(self.zpos[-1]) + topo
        min_height = float(np.min(curr_min))
        max_depth = float(np.max(curr_max))
        full_zpos = np.arange(min_height, max_depth + dz, dz)
        # np.arange 跨 0 时常得到 -2e-16，会被当成空气
        full_zpos = np.where(np.abs(full_zpos) < 1e-10, 0.0, full_zpos)
        x_new = np.arange(self.xpos[0], self.xpos[-1] + dx, dx)
        return x_new, full_zpos

    def _vgrid_on_regular(
        self,
        vgrid,
        x_new,
        full_zpos,
        v_air: float,
        v_water: float,
    ) -> np.ndarray:
        """把节点 ``vgrid (nx, nz)`` 铺到规则绘图网格 ``(len(x_new), len(full_zpos))``。"""
        xpos = np.asarray(self.xpos, dtype=float)
        zpos = np.asarray(self.zpos, dtype=float)
        topo = np.asarray(self.topo, dtype=float)
        vg = np.asarray(vgrid, dtype=float)
        x_new = np.asarray(x_new, dtype=float)
        full_zpos = np.asarray(full_zpos, dtype=float)
        nx = int(xpos.size)
        nz = int(zpos.size)
        ix = np.searchsorted(xpos, x_new)
        ix = np.minimum(ix, nx - 1)
        left = np.maximum(ix - 1, 0)
        use_left = (ix > 0) & ((x_new - xpos[left]) < (xpos[ix] - x_new))
        ix = np.where(use_left, left, ix)
        at_end = ix >= nx - 1
        ix_r = np.minimum(ix + 1, nx - 1)
        x1, x2 = xpos[ix], xpos[ix_r]
        t1, t2 = topo[ix], topo[ix_r]
        den_x = x2 - x1
        frac = np.zeros_like(x_new, dtype=float)
        ok_x = (~at_end) & (den_x != 0.0)
        frac[ok_x] = (x_new[ok_x] - x1[ok_x]) / den_x[ok_x]
        topo_val = np.where(at_end, t1, t1 + frac * (t2 - t1))

        rel_z = full_zpos[np.newaxis, :] - topo_val[:, np.newaxis]
        k = np.searchsorted(zpos, rel_z)
        k_lo = np.clip(k - 1, 0, nz - 1)
        k_hi = np.clip(k, 0, nz - 1)
        ix2 = ix[:, np.newaxis]
        v_lo = vg[ix2, k_lo]
        v_hi = vg[ix2, k_hi]
        z_lo = zpos[k_lo]
        z_hi = zpos[k_hi]
        den_z = z_hi - z_lo
        lerp = np.array(v_lo, copy=True)
        ok_z = den_z != 0.0
        lerp[ok_z] = v_lo[ok_z] + (v_hi[ok_z] - v_lo[ok_z]) * (
            rel_z[ok_z] - z_lo[ok_z]
        ) / den_z[ok_z]
        v_sub = np.where(
            k == 0,
            vg[ix2, 0],
            np.where(k == nz, vg[ix2, nz - 1], lerp),
        )
        z_abs = full_zpos[np.newaxis, :]
        sea = -1e-8
        air = z_abs < sea
        # topo≈0：网格挂在海面，水速已在 vgrid，不要再用均匀 v_water 盖住
        has_water_col = topo_val[:, np.newaxis] > 1e-8
        water = has_water_col & (z_abs >= sea) & (
            z_abs <= topo_val[:, np.newaxis] + 1e-8
        )
        return np.where(air, float(v_air), np.where(water, float(v_water), v_sub))

    def _regular_velocity_dataset(
        self,
        x_new,
        full_zpos,
        full_vgrid,
        *,
        v_air: Optional[float] = None,
        v_water: Optional[float] = None,
    ) -> xr.Dataset:
        va = self.v_air if v_air is None else v_air
        vw = self.v_water if v_water is None else v_water
        full_vgrid_t = np.asarray(full_vgrid, dtype=float).T
        with np.errstate(divide="ignore", invalid="ignore"):
            slow = 1.0 / full_vgrid_t
        return xr.Dataset(
            data_vars={
                "velocity": (("z", "x"), full_vgrid_t),
                "slowness": (("z", "x"), slow),
                "topo": ("x", np.interp(x_new, self.xpos, self.topo)),
                "v_water": vw,
                "v_air": va,
            },
            coords={"x": x_new, "z": full_zpos},
            attrs={"description": "Velocity model with air and water layers"},
        )

    def to_xarray(self, dx: float = None, dz: float = None) -> xr.Dataset:
        """将模型转换为 xarray 数据集。

        Args:
            dx (float, optional): x方向的采样间隔（km）。如果不指定，使用原始间隔。
            dz (float, optional): z方向的采样间隔（km）。如果不指定，使用原始间隔。

        Returns:
            xr.Dataset: 包含速度场的数据集。
        """
        x_new, full_zpos = self._regular_plot_axes(dx, dz)
        full_vgrid = self._vgrid_on_regular(
            self.vgrid, x_new, full_zpos, self.v_air, self.v_water
        )
        return self._regular_velocity_dataset(x_new, full_zpos, full_vgrid)


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
        mesh.pgrid = _pgrid_from_v(mesh.vgrid)
        
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
                
        mesh.pgrid = _pgrid_from_v(mesh.vgrid)
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