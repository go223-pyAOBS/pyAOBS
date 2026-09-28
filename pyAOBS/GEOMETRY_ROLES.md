# pyAOBS 炮点 / OBS 几何角色约定

**实现模块**：[`geometry_roles.py`](geometry_roles.py)  
**RTM 详表**：[`visualization/obs_rtm_qt/madagascar_obs_rtm/diag/obs_segy_geometry.txt`](visualization/obs_rtm_qt/madagascar_obs_rtm/diag/obs_segy_geometry.txt)  
**道头 API**：[`processors/raw2sac/segy_trace_header.py`](processors/raw2sac/segy_trace_header.py)

## 一句话

本工区默认 **`--geom obs`**：

| 物理角色 | SU/SEGY 槽 | Z 字段 |
|----------|------------|--------|
| **OBS**（固定） | `sx, sy` | `sxutm, syutm` |
| **炮点**（移动） | `gx, gy` | `rxutm, ryutm` |

口诀：`s*` = Station/OBS，`g*` = Gun/炮（与 SEGY Source/Group **字面相反**）。

## 算法字段

联合反演 / `OrientationObservation`：

- `source_xyz` = **物理炮点**
- `receiver_xyz` = **物理 OBS**（`(dx,dy,dz)` 加在这一端）
- 水深采样在 **OBS**

## 模式

| `geom` | 含义 |
|--------|------|
| `obs`（默认） | 上表本工区约定 |
| `segy` | 字面 Source=s*、Group=g* |
| `auto` | 唯一 XY 更少侧 = OBS（仅未知装填回退） |

## 禁止

- 新代码再用「方差/深度启发式」当**主路径**猜炮检
- 不说明模式就写「sx=震源」
- 把 RTM 互易里的 `sou/rec`（波场角色）当成道头物理角色

## 迁移

姿态校正、位置图已改走 `geometry_roles`（默认 `obs`）。其它模块读坐标时请改为：

```python
from pyAOBS.geometry_roles import physical_shot_obs_xyz, DEFAULT_GEOM

shot_xyz, obs_xyz = physical_shot_obs_xyz(trace_header, geom=DEFAULT_GEOM, use_utm=True)
```
