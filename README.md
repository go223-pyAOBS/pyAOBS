# pyAOBS

<p align="center">
  <img src="images/logo.png" alt="pyAOBS Logo" width="420"/>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Release](https://img.shields.io/badge/release-3.0.0rc2-orange.svg)](https://github.com/go223-pyAOBS/pyAOBS/releases)

**pyAOBS**（Python Active-source Ocean Bottom Seismology）是面向**主动源海底地震学（OBS）**的 Python 工具包：从数据转换、震相拾取、走时层析、速度模型解释，到岩性识别与 LIP / KKHS02 岩石学约束，提供统一的 **Workbench** 与可独立调用的 GUI / API。

| | |
|--|--|
| **当前标签** | [`v3.0.0rc2`](https://github.com/go223-pyAOBS/pyAOBS/releases/tag/v3.0.0rc2)（完整源码已上 GitHub） |
| **变更记录** | [CHANGELOG.md](CHANGELOG.md) |
| **仓库** | https://github.com/go223-pyAOBS/pyAOBS |
| **作者** | Haibo Huang（go223@scsio.ac.cn） |

---

## 目录

1. [功能总览](#功能总览)
2. [推荐工作流](#推荐工作流)
3. [安装](#安装)
4. [快速开始](#快速开始)
5. [Workbench](#workbench)
6. [可视化 GUI](#可视化-gui)
7. [速度模型（model_building）](#速度模型-model_building)
8. [正演 / 反演（modeling）](#正演--反演-modeling)
9. [数据处理（processors）](#数据处理-processors)
10. [岩性与物性（utils）](#岩性与物性-utils)
11. [岩石学 / LIP（petrology）](#岩石学--lip-petrology)
12. [野外站位工具（field）](#野外站位工具-field)
13. [包结构](#包结构)
14. [启动命令速查](#启动命令速查)
15. [更多文档](#更多文档)
16. [许可与引用](#许可与引用)

---

## 功能总览

| 模块 | 做什么 | 主要入口 |
|------|--------|----------|
| **Workbench** | 项目化工作台：节点流水线、运行历史、批量复跑、GUI 审计 | `python -m pyAOBS.workbench.app` |
| **idata / raw2sac** | OBS 数据格式统一转换（RAW/OBEM/SAC/SEGY） | Workbench `data.gui` |
| **zplotpy** | 震相拾取、剖面浏览、叠加与去噪 | `python -m pyAOBS.visualization.zplotpy.gui` |
| **tomo2d** | 首达/反射走时正演与层析反演（Python 包装 + C++ 内核） | `python -m pyAOBS.modeling.tomo2d.gui` |
| **imodel (Qt)** | 二维速度模型交互解释、物性/岩性、重力、petrology 导出 | `python -m pyAOBS.visualization.imodel_qt` |
| **iphase** | `tx.in` 震相选择、合并、QC（兼容 txphase/txconv 思路） | `python -m pyAOBS.visualization.iphase.iphase_gui` |
| **model_building** | Zelt `v.in` 读写、插值、TOMO2D 慢度网格 | Python API |
| **rayinvr / vedit** | 射线追踪库；`v.in` Tk 编辑器 | API / `modeling/vedit/main.py` |
| **utils（岩性）** | 岩石数据库、温压校正、速度→岩性分类 | imodel 内嵌；`classify_velocity_model` |
| **petrology** | KKHS02 式熔融柱 / H–Vp / 分离结晶 ΔVp / 反演界 | `python -m pyAOBS.petrology.gui` |
| **field** | OBS 站位布设与回收路径规划 | `field/*_gui.py` |
| **SU / denoise / relocation** | SU 读写、去噪管线、OBS 定向与水深辅助 | API / CLI |

---

## 推荐工作流

典型主动源 OBS 解释链（与 Workbench 内置插件顺序一致）：

```text
data.gui (格式转换)
    → zplotpy.gui (拾取)
        → tomo2d.gui / tomo2d.shell (正演·反演)
            → imodel.gui (速度模型解释 / 岩性 / 导出)
                → iphase.gui (震相整理)
                    → petrology.lip.gui (可选：壳厚–速度 → 地幔条件)
```

---

## 安装

**Python ≥ 3.9**（开发与验收以 **3.11** 为主）。NumPy 需满足 `>=1.24,<2.1`。

### 从源码（推荐）

```bash
git clone https://github.com/go223-pyAOBS/pyAOBS.git
cd pyAOBS
git checkout v3.0.0rc2
pip install -e ".[gui-qt]"
# 岩石学热力学可选：
# pip install -e ".[gui-qt,petrology]"
# 尽量一次装齐：
# pip install -e ".[full]"
```

### 从 Git 标签（Direct URL）

新版 pip **不要**使用 `#egg=包名[extra]`（会报 `invalid-egg-fragment`）：

```bash
pip install "pyAOBS[gui-qt] @ git+https://github.com/go223-pyAOBS/pyAOBS.git@v3.0.0rc2"
```

仅核心包：

```bash
pip install "pyAOBS @ git+https://github.com/go223-pyAOBS/pyAOBS.git@v3.0.0rc2"
```

### Extras 说明

| Extra | 内容 | 用途 |
|-------|------|------|
| **`gui-qt`** | PySide6 | Workbench、imodel Qt、LIP Petrology、iphase/zplot 主 GUI |
| **`petrology`** | burnman | 热力学物性（仓库内另有 vendored 回退） |
| **`gmt`** | pygmt | GMT 类绘图 |
| **`full`** | gui-qt + petrology + pygmt + pyproj + obspy | 全可选依赖 |

### 核心依赖（自动安装）

numpy、xarray、scipy、matplotlib、pandas、scikit-learn、seaborn、openpyxl、dearpygui 等（见 `setup.py`）。

### 二进制 / 外部程序

- **TOMO2D**：C++ 内核需本地编译或配置环境变量 `PYAOBS_TOMO2D_BIN` / `TOMO2D_BIN`（见 [`pyAOBS/modeling/tomo2d/src/README_OMP_BUILD.md`](pyAOBS/modeling/tomo2d/src/README_OMP_BUILD.md)）。
- **RAYINVR**：外部 Rayinvr 生态与 `v.in` / `r.in` 衔接；仓库提供 Python 包装与 `vedit` 编辑器。

---

## 快速开始

### 打开工作台

```bash
python -m pyAOBS.workbench.app
```

无 PySide6 时可尝试 Tk 回退：

```bash
# Windows PowerShell
$env:PYAOBS_WORKBENCH_UI="tk"
python -m pyAOBS.workbench.app
```

### Zelt 速度模型（API）

```python
from pyAOBS.model_building import ZeltVelocityModel2d, EnhancedZeltModel

model = ZeltVelocityModel2d("velocity.in")
v = model.at(100.0, 1.5)  # x (km), z (km) → Vp

enhanced = EnhancedZeltModel("velocity.in")
avg = enhanced.compute_average_velocities()
```

### SU 读写

```python
from pyAOBS import readsu, writesu, plotsu
```

### 岩性分类（简化 API）

```python
from pyAOBS.utils.simple_rock_classifier import classify_velocity_model
# 详见 pyAOBS/utils/README.md
```

---

## Workbench

统一的**项目化 GUI**：管理数据集、模型、拾取、运行节点与历史。

| 插件 ID | 名称 | 作用 |
|---------|------|------|
| `data.gui` | idata | RAW / OBEM TSM / SAC↔SEGY 统一转换 |
| `zplotpy.gui` | zplot | 震相拾取 |
| `tomo2d.gui` | TOMO2D GUI | 层析图形界面 |
| `tomo2d.shell` | TOMO2D CLI | 模板或自定义命令行 |
| `imodel.gui` | imodel | 速度模型解释（**默认 Qt**） |
| `iphase.gui` | iphase | 震相分析 |
| `petrology.lip.gui` | LIP Petrology | 地幔熔融 / H–Vp |

项目目录约定：`project.yaml`、`runs/<run_id>/`、`state/`、`datasets/`、`models/`、`picks/` 等。

**详细手册：** [`pyAOBS/workbench/README.md`](pyAOBS/workbench/README.md)

---

## 可视化 GUI

### imodel（速度模型解释）— 默认 Qt

```bash
python -m pyAOBS.visualization.imodel_qt
```

能力概要：网格/Zelt 模型交互、剖面、物性与岩性、重力工具、向 petrology 导出观测/沿迹、Fig.12a / 15c 预览、Workbench 状态回写。

Tk 旧版（兼容）：

```bash
python -m pyAOBS.visualization.imodel_gui
```

更多：[`IMODEL_README.md`](pyAOBS/visualization/IMODEL_README.md)、[`IMODEL_START_GUIDE.md`](pyAOBS/visualization/IMODEL_START_GUIDE.md)

### zplotpy（震相拾取）

```bash
python -m pyAOBS.visualization.zplotpy.gui
```

Qt Fast Viewer；可选对接 `processors/denoise`；Fortran/f2py 内核在 NumPy 小版本变化后可自动重编。

### iphase（tx.in 震相）

```bash
# GUI
python -m pyAOBS.visualization.iphase.iphase_gui

# CLI 示例
python -m pyAOBS.visualization.iphase info tx.in
python -m pyAOBS.visualization.iphase select ...
python -m pyAOBS.visualization.iphase combine ...
```

库 API：`read_tx` / `write_tx`、`select_phases`、`combine_ppp_pps_pss` 等。

### 其它绘图

- `ZeltModelVisualizer` / `GridModelVisualizer`（`visualization/show_model.py`）
- 可选 PyGMT（`pip install "pyAOBS[gmt]"`）
- 重力格网说明：[`visualization/data/README.txt`](pyAOBS/visualization/data/README.txt)

---

## 速度模型（model_building）

| 符号 | 用途 |
|------|------|
| `Point2d`, `ZNode2d`, `TrapezoidCell2d` | 二维几何与节点 |
| `read_vin_model` / `write_vin_model` | Zelt `v.in` 读写 |
| **`ZeltVelocityModel2d`** | 读模型、`at(x,z)`、转 xarray、层几何 |
| **`EnhancedZeltModel`** | 层平均速度、层间与双程时等增强计算 |
| `SlownessMesh2D`, `VelocityModelGenerator` | TOMO2D 慢度网格 |
| `velocity_to_su` / `mesh_to_su` | 模型导出 SU |

```python
from pyAOBS.visualization import ZeltModelVisualizer
from pyAOBS.model_building import ZeltVelocityModel2d

model = ZeltVelocityModel2d("velocity.in")
ZeltModelVisualizer(model).plot_zeltmodel(
    output_file="velocity_model.png",
    title="Velocity Model",
    colorbar_label="Velocity (km/s)",
)
```

---

## 正演 / 反演（modeling）

### TOMO2D

```bash
python -m pyAOBS.modeling.tomo2d.gui
```

Python 包装类 `TomoAnd`：`gen_smesh`、`tt_forward`、`tt_inverse`、`edit_smesh` 等。  
编译与 OMP：[`modeling/tomo2d/src/README_OMP_BUILD.md`](pyAOBS/modeling/tomo2d/src/README_OMP_BUILD.md)。

### RAYINVR

库：`VelocityModel`、`RayTracer`、`RayTracerConfig`、`RayinvrWrapper`（无独立 `-m` GUI）。

### vedit（v.in 编辑器）

```bash
python pyAOBS/modeling/vedit/main.py
```

说明：[`modeling/vedit/readme.md`](pyAOBS/modeling/vedit/readme.md)

### hybrid

机器学习初值、快速扫描等研究脚本（`modeling/hybrid/`），无统一 GUI；适合二次开发。

---

## 数据处理（processors）

### idata / raw2sac（格式转换）

Workbench 插件 **`data.gui`**，或直接运行：

```bash
python pyAOBS/processors/raw2sac/idata.py
```

涵盖：

| 工具 | 功能 |
|------|------|
| `raw2sac_v1_1_obspy` | RAW → SAC |
| `obem_tsm_to_sac_obspy` | OBEM TSM → SAC |
| `sac2y_v2_1_obspy` | SAC → SEGY |
| `fntime_v1_1` | 走时相关辅助 |

文档：[`README_obspy.md`](pyAOBS/processors/raw2sac/README_obspy.md)、[`README_UTILS.md`](pyAOBS/processors/raw2sac/README_UTILS.md)

### SU

`SUProcessor`、`readsu` / `writesu` / `plotsu` / `makehdr` 等（根包懒导出）。

### 去噪（denoise）

`denoise_trace` / `denoise_section`（SSQ / WSST / GCV 等后端），供 zplot 等调用。

### 重定位 / 定向（relocation）

- `run_orientation_correction`：OBS 水平分量定向  
- 水深采样、偏振特征  
- vendored **Ppol**（ObsPy）：[`processors/relocation/ppol/README.md`](pyAOBS/processors/relocation/ppol/README.md)

---

## 岩性与物性（utils）

| 模块 | 功能 |
|------|------|
| `rocks.py` | 岩石属性库、`RockClassifier`（随机森林） |
| `isrock.py` | 温压校正 + 速度模型岩性识别 |
| **`simple_rock_classifier.py`** | **推荐**：`classify_velocity_model()` |
| `empirical_formulas.py` | Gardner / Brocher / Castagna、温压、蛇纹岩化等 |

数据：`utils/rocks.xlsx`、`rocks.csv`；合并库流程见 [`ROCK_DATA_MERGE_GUIDE.md`](pyAOBS/utils/ROCK_DATA_MERGE_GUIDE.md)。

维护命令示例：

```bash
python -m pyAOBS.utils.test_isrock
python -m pyAOBS.utils.rock_database_health_check
python -m pyAOBS.utils.merge_rock_database
```

完整 API：[`pyAOBS/utils/README.md`](pyAOBS/utils/README.md)

**注意：** imodel 岩性链参考态常为 **200 MPa / 25°C**；petrology 正演参考态为 **600 MPa / 400°C**，二者不要混用。

---

## 岩石学 / LIP（petrology）

对齐 Korenaga et al. (2002, JGR)（**KKHS02**）的地壳—地幔约束工作流，并保留 Modern 正演轨。

### 启动 GUI

```bash
python -m pyAOBS.petrology.gui
# 兼容：python -m petrology.gui
```

可从 imodel / Workbench 导入观测 JSON 与沿迹 CSV（`--import-obs`、`--import-transect`）。

GUI 内含：H–Vp 扫描、经典曲线、公式查阅、读者指南（F1）、分离结晶 Fig.2/Fig.5 预览等。

### 科学链（四步）

| Step | 内容 | 主要模块 |
|------|------|----------|
| 1 | 方程 (1)：\(V_p(P,F)\) | `vp_regression.py` |
| 2 | 观测 → bulk \(V_p\) 可行界 | `invert.py` |
| 3 | 分离结晶 \(\Delta V_p\)（W&L + Langmuir） | `fc/`；默认**裸算**（`fig5_dvp_calibrate=False`） |
| 4 | 主动上涌 → 壳厚–速度 (H–Vp) | `active_upwelling.py` |

- **Reproduction 轨**：贴论文图件与数值量级  
- **Modern 轨**：Katz / REEBOX 等正演（接口与 Fig.12 布局对齐，曲线不必逐点等同）

### 与 imodel 的桥接

`petrology/imodel_bridge/`、`seismic/transect.py`：网格/剖面 → 火成岩壳厚与 \(V_{\mathrm{LC}}\) 等观测量。

### 验收脚本（开发者）

`petrology/validation/reproduce_fig*.py`、`check_r3.py`、`check_fig11.py` 等。  
**完整科学说明与验收表：** [`pyAOBS/petrology/README.md`](pyAOBS/petrology/README.md)

---

## 野外站位工具（field）

海上 OBS **布设 / 回收路径**规划（Tk GUI），**未**挂入 Workbench 主链：

```bash
python pyAOBS/field/recovery_strategy_gui.py
python pyAOBS/field/station_path_optimizer_gui.py
```

库：`RecoveryStationPlanner`、`load_stations_file` 等。

---

## 包结构

```text
pyAOBS/
├── workbench/          # 项目工作台（Qt / Tk）
├── visualization/      # imodel、zplotpy、iphase、绘图
├── model_building/     # Zelt / TOMO2D 网格
├── modeling/           # tomo2d、rayinvr、hybrid、vedit
├── processors/         # SU、raw2sac、denoise、relocation
├── utils/              # 岩性、经验公式、岩石库
├── petrology/          # KKHS02 / LIP GUI 与正演
├── field/              # 站位路径规划
└── tests/              # 单元测试（一般不进 wheel）
```

仓库根目录另有 `setup.py`、`MANIFEST.in`、`CHANGELOG.md`、`images/` 等。  
大体积测试数据、Perple_X、ScienceDirect 缓存等**不进入** Git / sdist（见 `MANIFEST.in` / `.gitignore`）。

---

## 启动命令速查

| 组件 | 命令 |
|------|------|
| Workbench | `python -m pyAOBS.workbench.app` |
| imodel Qt | `python -m pyAOBS.visualization.imodel_qt` |
| imodel Tk | `python -m pyAOBS.visualization.imodel_gui` |
| zplot | `python -m pyAOBS.visualization.zplotpy.gui` |
| iphase GUI | `python -m pyAOBS.visualization.iphase.iphase_gui` |
| TOMO2D GUI | `python -m pyAOBS.modeling.tomo2d.gui` |
| LIP Petrology | `python -m pyAOBS.petrology.gui` |
| idata | `python pyAOBS/processors/raw2sac/idata.py` |
| vedit | `python pyAOBS/modeling/vedit/main.py` |

---

## 更多文档

| 文档 | 内容 |
|------|------|
| [CHANGELOG.md](CHANGELOG.md) | 版本变更 |
| [pyAOBS/workbench/README.md](pyAOBS/workbench/README.md) | Workbench 全手册 |
| [pyAOBS/petrology/README.md](pyAOBS/petrology/README.md) | KKHS02 双轨与验收 |
| [pyAOBS/utils/README.md](pyAOBS/utils/README.md) | 岩性 API |
| [pyAOBS/visualization/IMODEL_README.md](pyAOBS/visualization/IMODEL_README.md) | imodel 能力 |
| [pyAOBS/processors/raw2sac/README_obspy.md](pyAOBS/processors/raw2sac/README_obspy.md) | 数据转换 CLI |
| [pyAOBS/modeling/tomo2d/src/README_OMP_BUILD.md](pyAOBS/modeling/tomo2d/src/README_OMP_BUILD.md) | TOMO2D 编译 |

在线文档站点（若已部署）：https://go223-pyAOBS.github.io/pyAOBS

---

## 许可与引用

本项目采用 **MIT License** — 见 [LICENSE](LICENSE)。

若在研究中使用 pyAOBS，请引用本软件，并按具体模块引用相关方法论文（例如层析/拾取所用算法，以及 petrology 模块对应的 Korenaga et al., 2002 等）。

```bibtex
@software{pyAOBS,
  author = {Huang, Haibo},
  title  = {pyAOBS: Python toolkit for active-source ocean-bottom seismology},
  year   = {2026},
  url    = {https://github.com/go223-pyAOBS/pyAOBS},
  note   = {Version 3.0.0rc2}
}
```

**岩石学模块相关论文示例：**

> Korenaga, J., Kelemen, P. B., Holbrook, W. S., & Sobolev, S. V. (2002). Methods for resolving the origin of large igneous provinces from crustal seismology. *JGR Solid Earth*, 107(B9). https://doi.org/10.1029/2001JB001030

---

## 贡献

欢迎 Issue 与 Pull Request。提交前请尽量：

1. 用 `git add <具体路径>` 只纳入相关文件；  
2. 不要提交 `.venv`、大型测试道集、密钥或内嵌 token；  
3. GUI / petrology 相关改动注明如何手动冒烟验证。
