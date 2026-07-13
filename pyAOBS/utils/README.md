# pyAOBS Utils 模块 - 地震波速度模型与岩石属性关联系统

## 概述

`utils` 模块是 pyAOBS 包的核心工具模块，主要功能是**建立地震波速度模型与岩石物性之间的关联**。该模块提供了完整的岩石物性数据库管理、岩石类型识别、温度压力校正等功能，用于从地震波速度模型中推断地下岩石类型。

## 🚀 快速开始（推荐）

**最简单的使用方式** - 只需提供速度数据：

```python
from pyAOBS.utils import classify_velocity_model

# 分类单个点
rock_type = classify_velocity_model(vp=6.1, vs=3.5, density=2.65)

# 分类整个速度模型
results = classify_velocity_model(
    vp=[6.1, 6.5, 7.5, 8.1],
    vs=[3.5, 3.7, 4.1, 4.5],
    depth=[0, 10, 20, 30]
)
print(results)
```

**详细使用说明请查看 [QUICK_START.md](QUICK_START.md)**

---

## 完整功能说明

## 模块结构

```
utils/
├── __init__.py          # 模块初始化，导出主要接口
├── rocks.py             # 核心模块：岩石物性数据类、数据库管理、基础分类器
├── isrock.py            # 高级识别模块：温度压力校正、速度模型识别
├── test_isrock.py       # 单元测试和演示代码
├── rocks.csv            # 岩石物性数据库（CSV格式）
└── rocks.xlsx           # 岩石物性数据库（Excel格式）
```

## 核心功能

### 1. 岩石物性数据管理 (`rocks.py`)

#### 1.1 数据类定义

**`RockProperties`** - 岩石物性基本参数类
```python
@dataclass
class RockProperties:
    vp: float                    # P波速度 (km/s)
    vs: Optional[float]          # S波速度 (km/s)
    density: Optional[float]      # 密度 (g/cm³)
    porosity: Optional[float]    # 孔隙度 (0-1)
    temperature: Optional[float]  # 温度 (°C)，默认25
    pressure: Optional[float]     # 压力 (MPa)，默认200
    fluid_saturation: Optional[float]  # 流体饱和度 (0-1)
    tectonic_setting: Optional[TectonicSetting]  # 构造背景
```

**`RockMeasurement`** - 岩石测量数据类（包含元数据）
```python
@dataclass
class RockMeasurement:
    properties: RockProperties    # 岩石物性参数
    source: str                  # 数据来源（实验室/研究者）
    method: str                  # 测量方法
    date: Optional[str]          # 测量日期
    uncertainty: Optional[Dict[str, float]]  # 测量不确定度
    quality_score: Optional[float]  # 数据质量分数 (0-1)
    notes: Optional[str]          # 备注信息
    location: Optional[str]       # 采样位置
    tectonic_description: Optional[str]  # 构造背景描述
```

**`TectonicSetting`** - 构造环境枚举类
```python
class TectonicSetting(Enum):
    OROGENIC_BELT = auto()       # 造山带
    PASSIVE_MARGIN = auto()      # 被动陆缘
    SUBDUCTION_ZONE = auto()     # 俯冲带
    RIFT = auto()                # 裂谷
    CRATON = auto()              # 克拉通
    BASIN = auto()               # 沉积盆地
    VOLCANIC_ARC = auto()        # 火山弧
    OCEANIC_CRUST = auto()       # 大洋地壳
    CONTINENTAL_CRUST = auto()   # 大陆地壳
    UNKNOWN = auto()             # 未知
```

#### 1.2 岩石类 (`Rock`)

提供岩石物性计算和转换功能：

**主要方法：**
- `vp_vs_ratio`: 计算Vp/Vs比值
- `poisson_ratio`: 计算泊松比
- `calculate_density(method)`: 使用经验公式计算密度
  - `'gardner'`: Gardner公式（适用于沉积岩）
  - `'nafe_drake'`: Nafe-Drake公式（适用于海洋沉积物）
  - `'brocher'`: Brocher公式（适用于地壳岩石）
- `calculate_vs(method)`: 使用经验公式计算S波速度
  - `'brocher'`: Brocher公式
  - `'castagna'`: Castagna公式（泥岩）
- `calculate_elastic_moduli()`: 计算弹性模量
  - 体积模量 (bulk_modulus)
  - 剪切模量 (shear_modulus)
  - 杨氏模量 (young_modulus)
  - 拉梅常数λ (lame_lambda)
- `calculate_temperature_effect()`: 计算温度对速度的影响
- `calculate_pressure_effect()`: 计算压力对速度的影响

#### 1.3 岩石数据库类 (`RockDatabase`)

管理实验室测量的岩石物性数据，支持从Excel文件加载和保存。

**主要功能：**
- `load_from_excel(file_path)`: 从Excel文件加载数据
- `save_to_excel(file_path)`: 保存数据到Excel文件
- `get_consensus_properties()`: 获取同一岩石不同测量结果的共识值
- `add_measurement()`: 添加单个测量数据
- `add_measurements_from_excel()`: 从Excel文件批量添加数据
- `filter_measurements()`: 按条件筛选测量数据
- `get_rocks_by_tectonic_setting()`: 按构造环境获取岩石数据
- `plot_measurement_comparison()`: 绘制不同来源的测量结果对比图
- `plot_tectonic_distribution()`: 绘制构造背景分布图

**Excel文件格式要求：**
- 必需列：`rock_type`, `vp`, `source`, `method`
- 可选列：`vs`, `density`, `porosity`, `temperature`, `pressure`, 
         `fluid_saturation`, `date`, `notes`, `location`, 
         `tectonic_setting`, `tectonic_description`

#### 1.4 数据质量控制 (`RockQualityControl`)

- `check_measurement_consistency()`: 检查同一岩石不同测量结果的一致性
- `calculate_quality_score()`: 计算测量数据的质量分数

#### 1.5 基础分类器 (`RockClassifier`)

基于机器学习的岩石分类器，使用随机森林算法。

**主要方法：**
- `classify_by_vp(vp)`: 基于P波速度分类
- `classify_by_vp_vs(vp, vs)`: 基于P波和S波速度分类
- `classify_with_uncertainty(vp, threshold)`: 带不确定性的分类
- `classify_by_vp_and_setting(vp, setting)`: 考虑构造环境的分类
- `plot_classification_results()`: 绘制分类结果

### 2. 高级岩石识别 (`isrock.py`)

#### 2.1 温度压力校正参数 (`CorrectionParameters`)

存储不同岩石类型的温度压力校正系数：

**温度校正系数（基于Christensen, 1979）：**
- 花岗岩-花岗闪长岩: -0.39e-4 /°C
- 辉长岩-紫苏辉长岩: -0.57e-4 /°C
- 玄武岩: -0.39e-4 /°C
- 板岩: -0.40e-4 /°C
- 镁铁质麻粒岩: -0.52e-4 /°C
- 长英质麻粒岩: -0.49e-4 /°C
- 角闪岩: -0.55e-4 /°C
- 橄榄岩: -0.56e-4 /°C
- 榴辉岩: -0.53e-4 /°C
- 蛇纹岩: -0.68e-4 /°C
- 默认值: -0.50e-4 /°C

**压力校正系数：**
- P波: 0.0002 /MPa
- S波: 0.00015 /MPa
- 密度: 0.0001 /MPa

#### 2.2 高级识别器 (`RockIdentifier`)

继承自`RockClassifier`，提供更强大的识别功能，包括温度压力校正。

**主要特性：**
1. **自动数据校正**：初始化时自动将训练数据校正到标准条件（25°C, 200MPa）
2. **温度压力校正**：识别前对输入数据进行校正
3. **迭代识别**：先使用标准系数初步识别，再使用特定岩石类型系数精确校正

**主要方法：**

**`pressure_correction(velocity, pressure, target_pressure, rock_type, is_s_wave)`**
- 将速度值校正到目标压力条件
- 支持P波和S波
- 可选择特定岩石类型的校正系数

**`temperature_correction(velocity, temperature, target_temperature, rock_type, is_s_wave)`**
- 将速度值校正到目标温度条件
- 支持P波和S波
- 根据岩石类型选择特定校正系数

**`density_correction(density, pressure, temperature, target_pressure, target_temperature)`**
- 将密度值校正到目标压力和温度条件

**`identify_rock(vp, vs, density, porosity, tectonic_setting, min_probability, max_candidates)`**
- 识别单个样本的岩石类型
- 返回候选岩石列表及其概率
- 考虑构造环境信息

**`identify_velocity_model(model_data, min_probability)`**
- 识别整个速度模型中的岩石类型
- 包括两步校正过程：
  1. 使用标准系数进行初步校正和识别
  2. 根据初步识别结果，使用特定岩石类型系数进行精确校正和最终识别
- 考虑构造环境对识别结果的影响

**`plot_identification_results(results, output_file)`**
- 绘制识别结果的可视化图表
- 包括概率分布、Vp-Vs关系、密度-孔隙度关系等

#### 2.3 便捷函数

**`identify_rock_type(vp, vs, density)`**
- 基于物性参数识别岩石类型的简化接口
- 返回(岩石类型, 置信度)元组

**`identify_rocks_from_model(model_file, database_file, min_probability, output_file)`**
- 从速度模型文件识别岩石类型的完整流程
- 包括数据加载、识别、结果可视化

## 工作流程

### 典型使用流程

1. **加载岩石数据库**
   ```python
   from pyAOBS.utils import load_rock_database
   db = load_rock_database('rocks.xlsx')
   ```

2. **创建识别器**
   ```python
   from pyAOBS.utils.isrock import RockIdentifier
   identifier = RockIdentifier('rocks.xlsx')
   identifier.train_classifier()
   ```

3. **识别单个样本**
   ```python
   result = identifier.identify_rock(
       vp=6.1,
       vs=3.5,
       density=2.65,
       porosity=0.02,
       tectonic_setting=TectonicSetting.CONTINENTAL_CRUST
   )
   ```

4. **识别速度模型**
   ```python
   model_data = {
       'vp': np.array([6.1, 6.5, 7.5, 8.1]),
       'vs': np.array([3.5, 3.7, 4.1, 4.5]),
       'density': np.array([2.65, 2.9, 3.0, 3.3]),
       'porosity': np.array([0.02, 0.02, 0.02, 0.02]),
       'pressure': np.array([100.0, 150.0, 200.0, 250.0]),
       'temperature': np.array([20.0, 25.0, 30.0, 35.0]),
       'tectonic_setting': [TectonicSetting.CONTINENTAL_CRUST, ...]
   }
   results = identifier.identify_velocity_model(model_data)
   ```

## 温度压力校正原理

### 压力校正

速度随压力的变化关系：
```
V_corrected = V_original × (1 + β × ΔP)
```
其中：
- `β`: 压力校正系数（P波: 0.0002/MPa, S波: 0.00015/MPa）
- `ΔP = target_pressure - original_pressure`

### 温度校正

速度随温度的变化关系：
```
V_corrected = V_original × (1 - α × ΔT)
```
其中：
- `α`: 温度校正系数（根据岩石类型不同，范围约-0.39e-4 到 -0.68e-4 /°C）
- `ΔT = original_temperature - target_temperature`
- 注意：温度升高，速度降低，所以使用减号

### 密度校正

密度随压力和温度的变化：
```
ρ_corrected = ρ_original × (1 + β_p × ΔP) × (1 + α_T × ΔT)
```

## 数据格式

### Excel数据库格式

| 列名 | 类型 | 必需 | 说明 |
|------|------|------|------|
| rock_type | str | 是 | 岩石类型 |
| vp | float | 是 | P波速度 (km/s) |
| vs | float | 否 | S波速度 (km/s) |
| density | float | 否 | 密度 (g/cm³) |
| porosity | float | 否 | 孔隙度 (0-1) |
| temperature | float | 否 | 温度 (°C) |
| pressure | float | 否 | 压力 (MPa) |
| source | str | 是 | 数据来源 |
| method | str | 是 | 测量方法 |
| tectonic_setting | str | 否 | 构造环境 |

## 依赖库

- `numpy`: 数值计算
- `pandas`: 数据处理
- `scikit-learn`: 机器学习分类器
- `matplotlib`: 数据可视化
- `seaborn`: 高级可视化
- `openpyxl`: Excel文件读写

## 测试

运行测试代码：
```python
python -m pyAOBS.utils.test_isrock
```

测试包括：
- 基础分类功能测试
- 高级识别功能测试
- 温度压力校正测试
- 速度模型识别测试

## 参考文献

1. Christensen, N.I. (1979). Compressional wave velocities in rocks at high temperatures and pressures, critical thermal gradients, and crustal low-velocity zones. *Journal of Geophysical Research*, 84(B12), 6849-6857.

2. Brocher, T.M. (2005). Empirical relations between elastic wavespeeds and density in the Earth's crust. *Bulletin of the Seismological Society of America*, 95(6), 2081-2092.

3. Gardner, G.H.F., Gardner, L.W., & Gregory, A.R. (1974). Formation velocity and density—the diagnostic basics for stratigraphic traps. *Geophysics*, 39(6), 770-780.

4. Nafe, J.E., & Drake, C.L. (1963). Physical properties of marine sediments. In *The Sea* (Vol. 3, pp. 794-815).

5. Castagna, J.P., Batzle, M.L., & Eastwood, R.L. (1985). Relationships between compressional-wave and shear-wave velocities in clastic silicate rocks. *Geophysics*, 50(4), 571-581.

## 作者

Haibo Huang, 2025

## 更新日志

- 2025: 初始版本，实现基础岩石识别和温度压力校正功能
