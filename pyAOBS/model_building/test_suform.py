# 转换Zelt模型
from pyAOBS.model_building.zeltform import ZeltVelocityModel2d
from pyAOBS.model_building.suform import velocity_to_su
from pyAOBS.model_building.suform import grid_to_su
from pyAOBS.model_building.suform import mesh_to_su
from pyAOBS.model_building.tomoform import SlownessMesh2D
import numpy as np
zelt_model = ZeltVelocityModel2d("v.in")
velocity_to_su(zelt_model, "v.output.su", dx=0.5, dz=0.5)
