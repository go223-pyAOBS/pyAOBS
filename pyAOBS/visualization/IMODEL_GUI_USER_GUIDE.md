# imodel_gui.py 完整使用说明

## 目录

1. [概述](#概述)
2. [快速开始](#快速开始)
3. [主要功能模块](#主要功能模块)
4. [DEM曲线缓存文件](#dem曲线缓存文件)
5. [详细功能说明](#详细功能说明)
6. [常见问题](#常见问题)
7. [技术细节](#技术细节)

---

## 概述

`imodel_gui.py` 是一个基于Tkinter的交互式速度模型分析工具，提供完整的图形用户界面，用于：

- **速度模型可视化**：加载和显示2D/3D速度模型（Grid格式）
- **交互式点选择**：鼠标点击选择模型中的点，支持编号显示
- **多边形选择**：绘制多边形区域进行采样和分析
- **剖面提取**：提取垂直、水平或沿路径的速度剖面
- **物性计算**：自动计算密度、压力、温度、岩石类型、弹性模量等
- **岩石数据库可视化**：Vp-Vs散点图和Vp-Vp/Vs关系图
- **DEM曲线分析**：显示不同纵横比和孔隙度对应的理论曲线
- **重力模拟**：计算和显示重力异常

---

## 快速开始

### 启动程序

**方法1：直接运行模块**
```bash
python -m pyAOBS.visualization.imodel_gui
```

**方法2：在Python中调用**
```python
from pyAOBS.visualization.imodel_gui import main
main()
```

**方法3：使用启动脚本**
- Windows: 双击 `imodel-gui.bat`
- Linux/Mac: 运行 `imodel-gui.sh`

### 基本操作流程

1. **加载模型**
   - 点击菜单 `File > Open Model` 或使用快捷键 `Ctrl+O`
   - 选择Grid格式文件（.grd, .nc等）

2. **查看模型**
   - 模型自动在主窗口显示
   - 使用工具栏进行缩放、平移等操作

3. **选择点或区域**
   - 点击 `Point Selection` 按钮启用点选择
   - 点击 `Polygon Selection` 按钮启用多边形选择

4. **查看结果**
   - 选择点后，结果自动显示在右侧结果面板
   - 可以添加到散点图或Vp/Vs图中进行分析

---

## 主要功能模块

### 1. 速度模型可视化

#### 加载模型
- **支持格式**：NetCDF (.nc, .grd), xarray Dataset
- **坐标系统**：自动识别x、z坐标
- **速度变量**：自动检测速度变量名（velocity, vp, v等）

#### 显示选项
- **颜色映射**：可在侧边栏选择（viridis, plasma, jet等）
- **等值线**：可显示速度等值线
- **坐标范围**：自动设置或手动调整

### 2. 交互式点选择

#### 启用点选择
点击侧边栏的 `Point Selection` 按钮

#### 操作方式
- **左键点击**：在主图上添加点（带编号）
- **右键点击**：删除最近的点
- **按D键**：删除最后一个点
- **按C键**：清除所有点

#### 点编号显示
- 主图中：红色圆点 + 红色背景白色数字编号
- 散点图中：红色星形点 + 红色背景白色数字编号
- Vp-Vp/Vs图中：红色星形点 + 红色背景白色数字编号
- **编号一致性**：所有图中的编号与主图一致

#### 添加点到子图
- **添加到Vp-Vs散点图**：点击 `Add Selected Points`（Vp-Vs Plot列）
- **添加到Vp/Vs-Vp图**：点击 `Add Selected Points`（Vp/Vs-Vp Plot列）

### 3. 多边形选择

#### 启用多边形选择
点击侧边栏的 `Polygon Selection` 按钮

#### 操作方式
- **左键点击**：添加多边形顶点
- **右键点击**：完成多边形（自动闭合）

#### 多边形功能

**3.1 添加多边形平均值**
- **Vp-Vs散点图**：点击 `Add Polygon Average`
- **Vp/Vs-Vp图**：点击 `Add Polygon Average`
- 功能：计算多边形内所有点的平均值，添加为单个点

**3.2 添加多边形采样点**
- **Vp-Vs散点图**：点击 `Add Polygon Samples`
- **Vp/Vs-Vp图**：点击 `Add Polygon Samples`
- 功能：
  - 弹出对话框询问采样点数（1-100，默认10）
  - 在多边形内均匀采样指定数量的点
  - 在主图上显示采样点（橙色方块）
  - 所有点自动添加到子图

**采样算法**：
- 使用最大化最小距离算法，确保采样点均匀分布
- 如果网格点不够，使用随机采样补充
- 自动避免点聚集

### 4. 剖面提取

#### 垂直剖面
- 在侧边栏输入X坐标（km）
- 点击 `Extract Profile` 按钮
- 结果显示在结果面板

#### 水平剖面
- 在侧边栏输入Z坐标（深度，km）
- 点击 `Extract Profile` 按钮

### 5. 物性计算

#### 计算单个点的物性
1. 使用点选择模式选择一个点
2. 点击 `Calculate Properties` 按钮
3. 结果自动显示在结果面板

#### 计算参数
- **P波速度** (vp)：从模型直接获取
- **S波速度** (vs)：使用Brocher或Castagna公式计算
- **密度** (density)：使用Gardner、Brocher或Nafe-Drake公式
- **压力** (pressure)：根据深度估算（默认30 MPa/km）
- **温度** (temperature)：根据深度估算（默认30°C/km）
- **岩石类型** (rock_type)：使用机器学习分类
- **弹性模量**：体积模量K、剪切模量μ、杨氏模量E、泊松比ν

#### 速度校正
所有速度值自动校正到标准条件：
- **目标压力**：200 MPa
- **目标温度**：25°C
- 使用经验公式进行压力-温度校正

### 6. 岩石数据库可视化

#### 6.1 Vp-Vs散点图

**打开散点图**
- 点击 `Show Vp-Vs Plot` 按钮

**功能**
- 显示岩石数据库中的Vp-Vs关系
- 突出显示特定岩石类型（Basalt, Serpentinite, Gabbro, Dunite, Granite）
- 显示从模型添加的点（红色星形，带编号）
- 显示多边形采样点（橙色方块）

**添加点**
- **点选择**：点击 `Add Selected Points`
- **多边形平均值**：点击 `Add Polygon Average`
- **多边形采样**：点击 `Add Polygon Samples`

#### 6.2 Vp/Vs-Vp关系图

**打开关系图**
- 点击 `Show Vp/Vs-Vp Plot` 按钮

**显示要素**（可通过复选框控制）
- ✅ **采样岩石分布**：数据库中的岩石样本点
- ✅ **含水量(蛇纹石化%)**：参考点，显示不同含水量对应的Vp和Vp/Vs
- ✅ **纵横比曲线**：不同纵横比（0.05, 0.03, 0.02, ...）对应的Vp-Vp/Vs曲线（黑色实线）
- ✅ **孔隙度曲线**：不同孔隙度（12.51%, 8.08%, 5.69%, ...）对应的曲线（viridis渐变色虚线）

**添加点**
- **点选择**：点击 `Add Selected Points`
- **多边形平均值**：点击 `Add Polygon Average`
- **多边形采样**：点击 `Add Polygon Samples`

**保存图片**
- 点击 `Save Figure` 按钮
- 支持格式：PNG, PDF, PS, EPS, JPEG, TIFF
- 自动根据选择的文件类型添加扩展名

**坐标轴设置**
- Vp轴范围：3.0 - 8.5 km/s
- Vp/Vs轴范围：1.6 - 2.15
- Vp/Vs轴刻度间隔：0.05

### 7. 重力模拟

#### 打开重力窗口
- 点击 `Show Gravity Anomaly Plot` 按钮

#### 设置参数
- **背景密度**：默认2.67 g/cm³
- **观测高度**：默认0.0 km
- **最大网格点数**：默认1000
- **扩展距离**：默认10.0 km

#### 添加重力体
- 使用多边形选择绘制区域
- 点击 `Add Polygon as Gravity Body`
- 输入密度值（g/cm³）

#### 计算重力
- 点击 `Calculate Gravity` 按钮
- 显示重力异常图

---

## DEM曲线缓存文件

### 概述

DEM（Differential Effective Medium）曲线计算较为耗时，程序实现了缓存机制以提高性能。计算后的曲线数据会自动保存到缓存文件中，下次使用时直接加载，无需重新计算。

### 文件位置

缓存文件保存在 `pyAOBS/utils/` 目录下。

### 文件命名规则

```
dem_curves_cache_{岩石名称}_{vp}_{vs}_{density}_{临界孔隙度}.pkl
```

**示例**：
```
dem_curves_cache_dunite_8.299_4.731_3.31_0.65.pkl
```

**命名说明**：
- `dunite`：背景岩石名称
- `8.299`：背景材料的P波速度（km/s）
- `4.731`：背景材料的S波速度（km/s）
- `3.31`：背景材料的密度（g/cm³）
- `0.65`：临界孔隙度

**数值格式化**：
- 整数自动去除小数点（如 `3.31` 而不是 `3.310`）
- 保留必要的小数位，去除尾随零

### 文件格式

缓存文件使用Python的`pickle`格式保存，是一个字典结构：

```python
{
    aspect_ratio_1: (vp_array_1, vp_vs_ratio_array_1, porosity_range),
    aspect_ratio_2: (vp_array_2, vp_vs_ratio_array_2, porosity_range),
    ...
    aspect_ratio_n: (vp_array_n, vp_vs_ratio_array_n, porosity_range)
}
```

**数据结构说明**：

- **键（aspect_ratio）**：`float`类型，纵横比值（如0.05, 0.03, 0.02等）
- **值（tuple）**：包含三个numpy数组的元组
  - `vp_array`：`numpy.ndarray`，P波速度数组（km/s），长度与porosity_range相同
  - `vp_vs_ratio_array`：`numpy.ndarray`，Vp/Vs比值数组，长度与porosity_range相同
  - `porosity_range`：`numpy.ndarray`，孔隙度数组（0到0.4，500个点）

**数据特点**：
- 无效值用`NaN`表示（计算失败或超出合理范围的点）
- 所有数组长度相同（500个点）
- 孔隙度数组是单调递增的

### 如何读取缓存文件

#### 方法1：使用程序自动加载

程序会在以下情况自动加载缓存：
1. 打开Vp/Vs-Vp关系图时
2. 如果缓存文件存在且数据完整，直接使用
3. 如果缓存文件不存在或数据不完整，自动计算并保存

**用户无需手动操作**，程序会自动处理。

#### 方法2：在Python代码中手动读取

```python
import pickle
import numpy as np
from pathlib import Path

# 1. 构建缓存文件路径
utils_dir = Path('pyAOBS/utils')
cache_file = utils_dir / 'dem_curves_cache_dunite_8.299_4.731_3.31_0.65.pkl'

# 2. 读取缓存文件
with open(cache_file, 'rb') as f:
    curves_data = pickle.load(f)

# 3. 使用数据
for aspect_ratio, (vp_array, vp_vs_ratio_array, porosity_range) in curves_data.items():
    print(f"纵横比: {aspect_ratio}")
    print(f"  Vp数组长度: {len(vp_array)}")
    print(f"  Vp/Vs数组长度: {len(vp_vs_ratio_array)}")
    print(f"  孔隙度范围: {porosity_range[0]:.4f} - {porosity_range[-1]:.4f}")
    
    # 获取有效数据（去除NaN）
    valid_mask = ~np.isnan(vp_array)
    vp_valid = vp_array[valid_mask]
    vp_vs_ratio_valid = vp_vs_ratio_array[valid_mask]
    porosity_valid = porosity_range[valid_mask]
    
    print(f"  有效数据点数: {len(vp_valid)}")
```

#### 方法3：检查缓存文件是否存在

```python
from pathlib import Path

cache_file = Path('pyAOBS/utils/dem_curves_cache_dunite_8.299_4.731_3.31_0.65.pkl')

if cache_file.exists():
    print(f"缓存文件存在: {cache_file}")
    print(f"文件大小: {cache_file.stat().st_size / 1024:.2f} KB")
else:
    print("缓存文件不存在，需要重新计算")
```

### 缓存文件管理

#### 自动管理

程序会自动：
1. **检查缓存**：打开Vp/Vs-Vp图时检查缓存文件是否存在
2. **验证完整性**：检查所有纵横比是否都有数据
3. **重新计算**：如果缓存不完整，自动重新计算
4. **保存缓存**：计算完成后自动保存

#### 手动管理

**删除缓存文件**：
```python
from pathlib import Path

cache_file = Path('pyAOBS/utils/dem_curves_cache_dunite_8.299_4.731_3.31_0.65.pkl')
if cache_file.exists():
    cache_file.unlink()
    print("缓存文件已删除")
```

**查看所有缓存文件**：
```python
from pathlib import Path

utils_dir = Path('pyAOBS/utils')
cache_files = list(utils_dir.glob('dem_curves_cache_*.pkl'))

print(f"找到 {len(cache_files)} 个缓存文件:")
for f in cache_files:
    print(f"  {f.name} ({f.stat().st_size / 1024:.2f} KB)")
```

### 缓存文件参数

当前默认参数（可在代码中修改）：

```python
# 背景材料参数（干dunite）
host_vp_dunite = 8.299      # km/s
host_vs_dunite = 4.731      # km/s
host_density_dunite = 3.310 # g/cm³

# 包含物参数（水）
inclusion_k = 2.2           # 体积模量 (GPa)
inclusion_mu = 0.0          # 剪切模量 (GPa)
inclusion_density = 1.03    # 密度 (g/cm³)

# DEM参数
critical_porosity = 0.65    # 临界孔隙度

# 纵横比列表
aspect_ratios = [0.05, 0.03, 0.02, 0.013, 0.01, 0.0067, 0.005, 0.002, 0.001, 0.0001]

# 孔隙度范围
porosity_max = 0.4         # 最大孔隙度（40%）
n_points = 500              # 孔隙度点数
```

### 缓存文件大小

典型的缓存文件大小：
- **单个文件**：约 50-200 KB（取决于有效数据点数量）
- **10个纵横比**：约 500 KB - 2 MB

### 注意事项

1. **参数一致性**：缓存文件与计算参数绑定，如果修改了参数（如背景材料、临界孔隙度），需要删除旧缓存或使用新的文件名
2. **跨平台兼容**：pickle文件在不同Python版本间可能不兼容，建议在同一环境中使用
3. **数据完整性**：程序会自动验证缓存数据的完整性，不完整的数据会被重新计算
4. **性能优化**：首次计算可能需要几分钟，后续使用缓存文件几乎瞬间加载

---

## 详细功能说明

### 1. 模型设置

#### 波型选择
- **Vp模型**：模型数据是P波速度
- **Vs模型**：模型数据是S波速度
- 选择后，程序会自动计算另一种波速

#### 速度转换方法
- **Brocher**：Brocher经验公式（推荐）
- **Castagna**：Castagna经验公式

#### 密度计算方法
- **Gardner**：Gardner经验公式
- **Brocher**：Brocher经验公式
- **Nafe-Drake**：Nafe-Drake经验公式

### 2. 点选择功能

#### 点编号系统
- 主图中显示编号（红色背景白色数字）
- 添加到子图后，编号保持一致
- 编号从1开始，按选择顺序递增

#### 点的信息
每个点包含以下信息：
- 坐标（x, z）
- 速度值（vp, vs，已校正到200 MPa, 25°C）
- 物性参数（密度、压力、温度等）
- 点类型（point, polygon_average, polygon_sample）

### 3. 多边形采样功能

#### 采样策略
1. **网格采样**：在多边形边界框内创建密集网格
2. **筛选**：保留多边形内的网格点
3. **均匀分布**：使用最大化最小距离算法选择均匀分布的点
4. **随机补充**：如果网格点不够，使用随机采样补充

#### 采样点数
- 范围：1-100
- 默认：10
- 建议：根据多边形大小选择，小区域10-20点，大区域20-50点

### 4. Vp/Vs-Vp图显示控制

#### 复选框功能
- **采样岩石分布**：显示/隐藏数据库中的岩石样本点
- **含水量(蛇纹石化%)**：显示/隐藏含水量参考点
- **纵横比曲线**：显示/隐藏不同纵横比对应的曲线
- **孔隙度曲线**：显示/隐藏不同孔隙度对应的曲线

#### 实时更新
- 勾选/取消复选框后，图形立即更新
- 无需重新打开窗口

### 5. 文件保存

#### 支持的格式
- **PNG**：便携式网络图形（推荐用于演示）
- **PDF**：便携式文档格式（推荐用于论文）
- **PS**：PostScript格式
- **EPS**：封装PostScript格式
- **JPEG**：联合图像专家组格式
- **TIFF**：标记图像文件格式

#### 保存方式
1. 点击 `Save Figure` 按钮
2. 选择文件类型（下拉菜单）
3. 输入文件名（可带或不带扩展名）
4. 程序自动根据选择的文件类型添加正确的扩展名

#### 注意事项
- 如果选择PDF类型但文件名没有扩展名，会自动添加`.pdf`
- 如果选择"All files"，默认使用PNG格式
- 所有图片保存为300 DPI高分辨率

---

## 常见问题

### Q1: 如何加载自己的速度模型？

**A**: 确保模型文件是Grid格式（NetCDF），包含：
- 速度变量（变量名可以是velocity, vp, vs, v等）
- x坐标（坐标名可以是x, lon, longitude, distance等）
- z坐标（坐标名可以是z, y, depth, t, time等）

程序会自动识别这些变量和坐标。

### Q2: 点选择后没有显示编号？

**A**: 检查以下几点：
1. 确保使用的是点选择模式（不是多边形选择）
2. 编号只显示在`type='point'`的点上
3. 如果是从多边形采样添加的点，不会显示编号（显示为橙色方块）

### Q3: DEM曲线计算很慢？

**A**: 
1. 首次计算需要几分钟，这是正常的
2. 计算完成后会自动保存缓存文件
3. 下次使用时，如果缓存文件存在，会直接加载（几乎瞬间）
4. 如果仍然很慢，检查缓存文件是否在正确位置

### Q4: 缓存文件在哪里？

**A**: 缓存文件保存在 `pyAOBS/utils/` 目录下，文件名格式为：
```
dem_curves_cache_{岩石名称}_{vp}_{vs}_{density}_{临界孔隙度}.pkl
```

### Q5: 如何清除缓存文件？

**A**: 
1. **手动删除**：直接删除 `pyAOBS/utils/` 目录下的 `.pkl` 文件
2. **程序自动**：如果修改了计算参数，程序会自动检测并重新计算

### Q6: 子图中的点没有编号？

**A**: 
1. 确保使用的是点选择模式添加的点（不是多边形采样）
2. 编号只对`type='point'`的点显示
3. 如果仍然没有，尝试清除子图中的点，重新添加

### Q7: 如何修改DEM计算参数？

**A**: 在代码中修改以下参数（在`show_vp_vs_ratio_plot`函数中）：
```python
host_vp_dunite = 8.299      # 背景材料Vp
host_vs_dunite = 4.731      # 背景材料Vs
host_density_dunite = 3.310 # 背景材料密度
critical_porosity = 0.65    # 临界孔隙度
aspect_ratios = [...]       # 纵横比列表
porosity_max = 0.4         # 最大孔隙度
n_points = 500             # 孔隙度点数
```

修改后，程序会使用新的文件名保存缓存。

### Q8: 多边形采样点分布不均匀？

**A**: 
1. 确保多边形面积足够大（太小可能无法均匀分布）
2. 增加采样点数
3. 如果多边形形状复杂，可能需要更多采样点

### Q9: 保存的PDF文件打不开？

**A**: 
1. 确保选择了"PDF files"类型
2. 如果文件名没有扩展名，程序会自动添加`.pdf`
3. 如果仍然有问题，尝试手动输入完整文件名（如`output.pdf`）

### Q10: 如何批量处理多个模型？

**A**: 目前GUI版本不支持批量处理，建议：
1. 使用编程方式（`imodel.py`）编写脚本
2. 或者逐个加载模型进行处理

---

## 技术细节

### 文件结构

```
pyAOBS/
├── visualization/
│   ├── imodel_gui.py          # 主GUI程序
│   ├── imodel.py              # 核心功能（无GUI）
│   └── IMODEL_GUI_USER_GUIDE.md  # 本文档
└── utils/
    ├── empirical_formulas.py  # 经验公式（包含DEM计算）
    └── dem_curves_cache_*.pkl  # DEM曲线缓存文件
```

### 依赖库

- **Tkinter**：GUI框架
- **matplotlib**：绘图和交互
- **numpy**：数值计算
- **pandas**：数据处理
- **xarray**：Grid数据读取
- **pickle**：缓存文件序列化

### 性能优化

1. **DEM曲线缓存**：避免重复计算
2. **数据校正缓存**：速度校正结果可缓存
3. **图形更新优化**：使用增量更新而非完全重绘

### 扩展开发

如果需要扩展功能，主要修改点：

1. **添加新的显示选项**：在`_refresh_vp_vs_ratio_plot`函数中添加
2. **修改DEM参数**：在`show_vp_vs_ratio_plot`函数中修改
3. **添加新的文件格式**：在`save_vp_vs_ratio_figure`函数中添加

---

## 附录

### A. DEM曲线缓存文件完整示例

```python
import pickle
import numpy as np
from pathlib import Path

# 读取缓存文件
cache_file = Path('pyAOBS/utils/dem_curves_cache_dunite_8.299_4.731_3.31_0.65.pkl')

with open(cache_file, 'rb') as f:
    curves_data = pickle.load(f)

# 查看数据结构
print(f"缓存文件包含 {len(curves_data)} 个纵横比")
print("\n纵横比列表:")
for aspect_ratio in sorted(curves_data.keys()):
    vp_array, vp_vs_ratio_array, porosity_range = curves_data[aspect_ratio]
    valid_count = np.sum(~np.isnan(vp_array))
    print(f"  {aspect_ratio:.6f}: {valid_count}/{len(vp_array)} 有效数据点")

# 提取特定纵横比的数据
aspect_ratio = 0.01
if aspect_ratio in curves_data:
    vp_array, vp_vs_ratio_array, porosity_range = curves_data[aspect_ratio]
    
    # 获取有效数据
    valid_mask = ~np.isnan(vp_array)
    vp_valid = vp_array[valid_mask]
    vp_vs_ratio_valid = vp_vs_ratio_array[valid_mask]
    porosity_valid = porosity_range[valid_mask]
    
    print(f"\n纵横比 {aspect_ratio} 的有效数据:")
    print(f"  孔隙度范围: {porosity_valid[0]:.4f} - {porosity_valid[-1]:.4f}")
    print(f"  Vp范围: {vp_valid.min():.2f} - {vp_valid.max():.2f} km/s")
    print(f"  Vp/Vs范围: {vp_vs_ratio_valid.min():.2f} - {vp_vs_ratio_valid.max():.2f}")
```

### B. 手动创建缓存文件

如果需要手动创建或修改缓存文件：

```python
import pickle
import numpy as np
from pathlib import Path

# 创建示例数据
curves_data = {
    0.05: (
        np.array([8.0, 7.5, 7.0, np.nan, ...]),  # vp_array
        np.array([1.8, 1.85, 1.9, np.nan, ...]),  # vp_vs_ratio_array
        np.linspace(0, 0.4, 500)                  # porosity_range
    ),
    0.03: (
        np.array([8.0, 7.8, 7.5, ...]),
        np.array([1.8, 1.82, 1.85, ...]),
        np.linspace(0, 0.4, 500)
    ),
    # ... 更多纵横比
}

# 保存缓存文件
cache_file = Path('pyAOBS/utils/dem_curves_cache_dunite_8.299_4.731_3.31_0.65.pkl')
with open(cache_file, 'wb') as f:
    pickle.dump(curves_data, f)

print(f"缓存文件已保存: {cache_file}")
```

### C. 验证缓存文件完整性

```python
import pickle
import numpy as np
from pathlib import Path

def validate_cache_file(cache_file):
    """验证缓存文件的完整性"""
    try:
        with open(cache_file, 'rb') as f:
            curves_data = pickle.load(f)
        
        # 检查是否为字典
        if not isinstance(curves_data, dict):
            return False, "缓存文件不是字典格式"
        
        # 检查每个纵横比的数据
        expected_aspect_ratios = [0.05, 0.03, 0.02, 0.013, 0.01, 0.0067, 0.005, 0.002, 0.001, 0.0001]
        missing = []
        invalid = []
        
        for aspect_ratio in expected_aspect_ratios:
            if aspect_ratio not in curves_data:
                missing.append(aspect_ratio)
            else:
                vp_array, vp_vs_ratio_array, porosity_range = curves_data[aspect_ratio]
                # 检查数据格式
                if not (isinstance(vp_array, np.ndarray) and 
                       isinstance(vp_vs_ratio_array, np.ndarray) and
                       isinstance(porosity_range, np.ndarray)):
                    invalid.append(aspect_ratio)
                # 检查数组长度
                elif not (len(vp_array) == len(vp_vs_ratio_array) == len(porosity_range)):
                    invalid.append(aspect_ratio)
                # 检查是否有有效数据
                elif np.sum(~np.isnan(vp_array)) == 0:
                    invalid.append(aspect_ratio)
        
        if missing:
            return False, f"缺少纵横比: {missing}"
        if invalid:
            return False, f"无效数据: {invalid}"
        
        return True, "缓存文件完整"
    
    except Exception as e:
        return False, f"读取错误: {e}"

# 使用示例
cache_file = Path('pyAOBS/utils/dem_curves_cache_dunite_8.299_4.731_3.31_0.65.pkl')
is_valid, message = validate_cache_file(cache_file)
print(f"验证结果: {message}")
```

---

## 更新日志

### 最新版本功能
- ✅ DEM曲线缓存机制
- ✅ 点编号显示（主图和子图一致）
- ✅ 多边形采样功能（自定义采样点数）
- ✅ Vp/Vs-Vp图显示控制（复选框）
- ✅ 多种文件格式支持（PNG, PDF, PS, EPS, JPEG, TIFF）
- ✅ 进度条显示（DEM计算时）

---

## 联系与支持

如有问题或建议，请联系：
- **作者**：Haibo Huang
- **日期**：2025

---

**文档版本**：1.0  
**最后更新**：2025
