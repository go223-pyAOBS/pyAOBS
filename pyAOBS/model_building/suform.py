"""
SU Format Velocity Model Module

This module provides functions to convert different velocity model formats
(SlownessMesh2D, ZeltVelocityModel2d, and grid format) to SU format.
"""

import numpy as np
from typing import Union, Optional
import xarray as xr
from pathlib import Path

from pyAOBS.processors.supython import writesu, makehdr
from pyAOBS.model_building.zeltform import ZeltVelocityModel2d
from pyAOBS.model_building.tomoform import SlownessMesh2D

def velocity_to_su(velocity_model: Union[ZeltVelocityModel2d, SlownessMesh2D, xr.Dataset, np.ndarray],
                  output_file: str,
                  dx: float = 1.0,
                  dz: float = 0.5,
                  x_start: Optional[float] = None,
                  z_start: Optional[float] = None) -> None:
    """将速度模型转换为SU格式。

    Args:
        velocity_model: 输入的速度模型，可以是以下类型之一：
            - ZeltVelocityModel2d: Zelt格式的速度模型
            - SlownessMesh2D: 慢度网格模型
            - xr.Dataset: 包含速度场的xarray数据集
            - np.ndarray: 速度场矩阵
        output_file (str): 输出SU文件的路径
        dx (float): x方向的采样间隔（km），默认1.0 km
        dz (float): z方向的采样间隔（km），默认0.5 km
        x_start (float, optional): 起始x坐标，仅在输入为numpy数组时需要
        z_start (float, optional): 起始z坐标，仅在输入为numpy数组时需要
    """
    # 将不同格式的速度模型转换为统一的numpy数组格式
    if isinstance(velocity_model, ZeltVelocityModel2d):
        # 对于Zelt模型，使用to_xarray方法并获取velocity字段
        ds = velocity_model.to_xarray(dx=dx, dz=dz)
        velocity = ds.velocity.values
        x = ds.x.values
        z = ds.z.values
    elif isinstance(velocity_model, SlownessMesh2D):
        # 对于SlownessMesh2D模型，使用to_xarray方法获取数据集
        ds = velocity_model.to_xarray(dx=dx, dz=dz)
        velocity = ds.velocity.values
        x = ds.x.values
        z = ds.z.values
    elif isinstance(velocity_model, xr.Dataset):
        # 对于xarray数据集，直接获取velocity字段
        velocity = velocity_model.velocity.values
        x = velocity_model.x.values
        z = velocity_model.z.values
    elif isinstance(velocity_model, np.ndarray):
        # 对于numpy数组，需要创建坐标轴
        if x_start is None or z_start is None:
            raise ValueError("x_start and z_start must be provided for numpy array input")
            
        # 获取原始数组的形状和坐标
        nz_orig, nx_orig = velocity_model.shape
        x_orig = np.linspace(x_start, x_start + (nx_orig-1) * dx, nx_orig)
        z_orig = np.linspace(z_start, z_start + (nz_orig-1) * dz, nz_orig)
        
        # 创建新的坐标轴
        x = np.arange(x_start, x_start + nx_orig * dx, dx)
        z = np.arange(z_start, z_start + nz_orig * dz, dz)
        
        # 创建网格点
        X_orig, Z_orig = np.meshgrid(x_orig, z_orig)
        X_new, Z_new = np.meshgrid(x, z)
        
        # 使用scipy的griddata进行插值
        from scipy.interpolate import griddata
        velocity = griddata(
            points=(X_orig.flatten(), Z_orig.flatten()),
            values=velocity_model.flatten(),
            xi=(X_new, Z_new),
            method='linear',
            fill_value=np.nan
        )
    else:
        raise TypeError("Unsupported velocity model type")

    # 将速度从km/s转换为m/s
    velocity = velocity * 1000.0

    # 确保速度数组是float32类型
    velocity = velocity.astype(np.float32)

    # 创建SU头文件
    nz, nx = velocity.shape
    dt = dz * 1000  # 转换为米
    dx = dx * 1000  # 转换为米
    hdr = makehdr(velocity, dx=dx, dt=dt, t0=z[0]*1000, f2=x[0]*1000)

    # 写入SU文件
    writesu(output_file, velocity, hdr)

def slowness_to_su(slowness_model: np.ndarray,
                   output_file: str,
                   dx: float = 1.0,
                   dz: float = 0.5,
                   x_start: float = 0.0,
                   z_start: float = 0.0) -> None:
    """将慢度模型转换为SU格式。

    Args:
        slowness_model (np.ndarray): 慢度场矩阵
        output_file (str): 输出SU文件的路径
        dx (float): x方向的采样间隔（km），默认1.0 km
        dz (float): z方向的采样间隔（km），默认0.5 km
        x_start (float): 起始x坐标，默认0.0 km
        z_start (float): 起始z坐标，默认0.0 km
    """
    # 将慢度转换为速度（km/s）
    with np.errstate(divide='ignore', invalid='ignore'):
        velocity = np.where(slowness_model > 0, 1.0 / slowness_model, 0)
    
    # 调用velocity_to_su进行转换
    velocity_to_su(velocity, output_file, dx, dz, x_start, z_start)

def mesh_to_su(mesh: SlownessMesh2D,
               output_file: str,
               dx: float = 1.0,
               dz: float = 0.5) -> None:
    """将smesh格式的速度模型转换为SU格式。

    Args:
        mesh (SlownessMesh2D): 输入的慢度网格模型
        output_file (str): 输出SU文件的路径
        dx (float): x方向的采样间隔（km），默认1.0 km
        dz (float): z方向的采样间隔（km），默认0.5 km
    """
    
    # 调用velocity_to_su进行转换
    velocity_to_su(mesh, output_file, dx=dx, dz=dz)

def grid_to_su(grid_file: str,
               output_file: str,
               dx: float = 1.0,
               dz: float = 0.5) -> None:
    """将网格格式的速度模型转换为SU格式。

    Args:
        grid_file (str): 输入的网格文件路径
        output_file (str): 输出SU文件的路径
        dx (float): x方向的采样间隔（km），默认1.0 km
        dz (float): z方向的采样间隔（km），默认0.5 km
    """
    # 读取网格文件
    ds = xr.open_dataset(grid_file)
    
    # 创建新的坐标轴
    x_new = np.arange(ds.x.values[0], ds.x.values[-1] + dx, dx)
    z_new = np.arange(ds.z.values[0], ds.z.values[-1] + dz, dz)
    
    # 使用xarray的插值功能进行重采样
    ds_resampled = ds.interp(
        x=x_new,
        z=z_new,
        method='linear',  # 使用线性插值
        kwargs={'fill_value': None}  # 对超出范围的值使用NaN
    )
    
    # 调用velocity_to_su进行转换
    velocity_to_su(ds_resampled, output_file, dx=dx, dz=dz) 