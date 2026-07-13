import os
import numpy as np
import matplotlib.pyplot as plt
import pytest
from pyAOBS.modeling.rayinvr.models import VelocityModel, PhaseType
from pyAOBS.modeling.rayinvr.ray_tracer import RayTracer, RayTracerConfig
from pyAOBS.model_building.models import Point2d
from pyAOBS.modeling.rayinvr.rayinvr_wrapper import RayinvrWrapper

pytestmark = pytest.mark.integration



def test_rayinvr_wrapper():
    # 创建接口实例
    rayinvr = RayinvrWrapper()
    # 运行主程序
    rayinvr.run_main()
    

        
    # 测试射线追踪
    result = rayinvr.trace_ray(
        x_start=0.0,
        z_start=0.0,
        angle=1.0,
        layer=1,
        iblk=1000
    )
    
    print("射线追踪结果:")
    print(f"点数: {result['npoints']}")
    print(f"状态: {result['status']}")
    print("路径点坐标:")
    for x, z, t in zip(result['x'], result['z'], result['time']):
        print(f"x={x:.3f}, z={z:.3f}, t={t:.3f}")
        
    # 测试速度查询
    v = rayinvr.get_velocity(100.0, 50.0)
    print(f"\n位置(100.0, 50.0)处的速度: {v:.3f} km/s")

if __name__ == '__main__':
    #test_ray_tracing()
    test_rayinvr_wrapper()