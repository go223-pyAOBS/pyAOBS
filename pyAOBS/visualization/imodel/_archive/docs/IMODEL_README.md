# imodel.py - 交互式速度模型分析工具

## 概述

`imodel.py` 是一个交互式速度模型分析工具，提供图形界面用于：
- 可视化速度模型（grid格式）
- 鼠标交互（取点、多边形选择）
- 提取一维速度剖面
- 计算物性参数（密度、压力、岩性、重力等）
- 结果可视化和导出

## 快速开始

### GUI方式（推荐，参考vedit风格）

```python
from pyAOBS.visualization.imodel_gui import main

# 启动GUI应用
main()
```

或直接运行：
```bash
python -m pyAOBS.visualization.imodel_gui
```

### 编程方式（无GUI）

```python
from pyAOBS.visualization.imodel import InteractiveModelViewer

# 1. 创建查看器
viewer = InteractiveModelViewer(grid_file='velocity.grd')

# 2. 绘制模型
viewer.plot_model(cmap='viridis')

# 3. 启用交互功能
viewer.enable_point_selection()
viewer.enable_polygon_selection()

# 4. 显示窗口
viewer.show()

# 5. 提取剖面（在交互窗口中选择点后）
profile = viewer.extract_vertical_profile(x=50.0)

# 6. 计算物性
properties = viewer.property_calculator.calculate_properties_for_profile(profile)

# 7. 计算重力
gravity = viewer.calculate_gravity(properties)

# 8. 导出结果
viewer.export_results('output')
```

## 核心功能

### 1. 速度模型可视化

- 支持grid格式（.grd, .nc）
- 可自定义颜色映射
- 支持等值线显示
- 自动识别坐标系统

### 2. 交互式选择

**点选择**:
- 左键点击：添加点
- 右键点击：删除最近的点
- 按D键：删除最后一个点
- 按C键：清除所有点

**多边形选择**:
- 左键点击：添加顶点
- 右键点击：完成多边形
- 自动绘制多边形区域

### 3. 剖面提取

- **垂直剖面**: 固定x坐标，提取深度方向的速度
- **水平剖面**: 固定深度，提取水平方向的速度
- **路径剖面**: 沿任意路径提取速度

### 4. 物性计算

自动计算以下参数：
- **P波速度** (vp): 从模型直接获取
- **S波速度** (vs): 使用Brocher公式估算
- **密度** (density): 使用Gardner公式计算
- **压力** (pressure): 根据深度估算（默认30 MPa/km）
- **温度** (temperature): 根据深度估算（默认30°C/km）
- **岩石类型** (rock_type): 使用机器学习分类
- **弹性模量**: 体积模量、剪切模量、杨氏模量、泊松比

### 5. 重力计算

- **Bouguer异常**: 基于密度差异
- **自由空气异常**: 基于高程
- 支持自定义参考密度

## 架构设计

```
imodel.py (核心功能，无GUI)
│
├── PointSelector (点选择器)
├── ProfileExtractor (剖面提取器)
├── PropertyCalculator (物性计算器)
└── GravityCalculator (重力计算器)

imodel_gui.py (GUI界面，参考vedit)
│
└── InteractiveModelViewerGUI (主窗口类)
    ├── Tkinter主窗口
    ├── Matplotlib Canvas (嵌入)
    ├── 侧边栏控件
    ├── 菜单栏
    └── 结果显示区域
```

**设计理念**：
- `imodel.py`: 提供核心功能类，不依赖GUI，可独立使用
- `imodel_gui.py`: 基于Tkinter的GUI界面，参考vedit的实现方式
- 两者分离，便于测试和扩展

## 代码复用策略

### 最大化复用现有代码

1. **密度计算**: 复用 `GridModelProcessor.velocity_to_density()`
2. **岩石分类**: 复用 `SimpleRockClassifier.classify()`
3. **S波速度**: 复用 `Rock.calculate_vs()`
4. **弹性模量**: 复用 `Rock.calculate_elastic_moduli()`

### 新增功能

1. **交互式界面**: 使用matplotlib实现
2. **点/多边形选择**: 基于matplotlib事件系统
3. **剖面提取**: 从xarray Dataset提取数据
4. **重力计算**: 实现Bouguer板模型

## 文件结构

```
visualization/
├── imodel.py                    # 主实现文件
├── example_imodel.py            # 使用示例
├── IMODEL_STRATEGY.md          # 实现策略文档
├── IMODEL_IMPLEMENTATION_SUMMARY.md  # 实现总结
└── IMODEL_README.md            # 本文档
```

## 依赖关系

```
imodel.py
  ├── matplotlib (交互式绘图)
  ├── xarray (grid数据处理)
  ├── numpy (数值计算)
  ├── pandas (数据管理)
  │
  └── pyAOBS模块
      ├── visualization.show_model
      │   ├── GridModelProcessor (密度计算)
      │   └── GridModelVisualizer (可选，用于参考)
      │
      └── utils
          ├── simple_rock_classifier (岩石分类)
          └── rocks (物性计算)
```

## 使用场景

### 场景1: 快速查看模型

```python
viewer = interactive_model_viewer('velocity.grd')
# 在图形窗口中点击查看不同位置的速度值
```

### 场景2: 提取剖面分析

```python
viewer = InteractiveModelViewer('velocity.grd')
viewer.plot_model()

# 提取垂直剖面
profile = viewer.extract_vertical_profile(x=50.0)

# 计算物性
props = viewer.property_calculator.calculate_properties_for_profile(profile)

# 可视化
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1)
axes[0].plot(props['vp'], props['depth'])
axes[1].plot(props['density'], props['depth'])
plt.show()
```

### 场景3: 批量计算

```python
# 定义多个点
points = [(10, 1), (20, 2), (30, 3), (40, 4), (50, 5)]

# 批量计算
results = []
for x, z in points:
    props = viewer.calculate_properties_at_point(x, z)
    results.append(props)

results_df = pd.DataFrame(results)
results_df.to_csv('results.csv')
```

## 扩展功能建议

### 已实现 ✅
- [x] 基本交互功能
- [x] 点选择
- [x] 多边形选择
- [x] 剖面提取
- [x] 物性计算
- [x] 重力计算

### 待实现 🔄
- [ ] 结果可视化面板
- [ ] 批量计算优化
- [ ] 3D可视化
- [ ] 多模型对比
- [ ] 自定义计算参数GUI
- [ ] Web界面

## 技术说明

### GUI实现（参考vedit）

**技术栈**：
- `Tkinter`: 主GUI框架
- `matplotlib.backends.backend_tkagg.FigureCanvasTkAgg`: 将matplotlib嵌入Tkinter
- `NavigationToolbar2Tk`: matplotlib标准工具栏
- `ttk.Frame`: 创建侧边栏和控件布局

**交互式绘图**：
- `matplotlib事件系统`: 鼠标和键盘事件处理
- `PolygonSelector`: 多边形选择工具
- `PointSelector`: 自定义点选择器
- `blitting`: 提高绘制性能

### 数据格式

支持的标准格式：
- NetCDF (.nc, .grd)
- xarray Dataset
- 坐标系统：自动识别x/z坐标

### 性能考虑

- 大型数据：使用数据降采样
- 密度计算：缓存结果避免重复计算
- 交互响应：使用blitting技术

## 常见问题

**Q: 如何加载自己的速度模型？**
A: 确保模型是grid格式（NetCDF），包含速度变量和x、z坐标。

**Q: 如何自定义计算参数？**
A: 使用`PropertyCalculator.calculate_all_properties()`的参数：
```python
props = viewer.property_calculator.calculate_all_properties(
    x, z,
    pressure_gradient=25.0,  # 自定义压力梯度
    temperature_gradient=20.0,  # 自定义地温梯度
    density_method='brocher'  # 自定义密度计算方法
)
```

**Q: 如何导出结果？**
A: 使用`viewer.export_results('output_dir')`，会导出所有剖面和物性结果。

## 相关文档

- [IMODEL_STRATEGY.md](IMODEL_STRATEGY.md) - 详细实现策略
- [IMODEL_IMPLEMENTATION_SUMMARY.md](IMODEL_IMPLEMENTATION_SUMMARY.md) - 实现总结
- [example_imodel.py](example_imodel.py) - 使用示例代码

## 作者

Haibo Huang, 2025
