# Utils 模块代码结构详细说明

## 1. 模块架构图

```
utils/
│
├── rocks.py (核心数据模型和基础功能)
│   ├── TectonicSetting (枚举类)
│   ├── RockProperties (数据类)
│   ├── RockMeasurement (数据类)
│   ├── Rock (岩石类)
│   │   ├── 物性计算
│   │   ├── 弹性模量计算
│   │   └── 温度压力效应计算
│   ├── RockQualityControl (质量控制类)
│   ├── RockDatabase (数据库管理类)
│   │   ├── 数据加载/保存
│   │   ├── 数据筛选
│   │   ├── 共识值计算
│   │   └── 可视化
│   └── RockClassifier (基础分类器)
│       ├── 基于速度的分类
│       ├── 考虑构造环境的分类
│       └── 不确定性分类
│
├── isrock.py (高级识别和校正功能)
│   ├── CorrectionParameters (校正参数类)
│   ├── RockIdentifier (高级识别器)
│   │   ├── 继承自 RockClassifier
│   │   ├── 温度压力校正
│   │   ├── 迭代识别算法
│   │   └── 速度模型识别
│   └── 便捷函数
│       ├── identify_rock_type()
│       └── identify_rocks_from_model()
│
└── test_isrock.py (测试和演示)
    ├── TestRockClassificationBase
    └── TestRockIdentification
```

## 2. 类继承关系

```
RockClassifier (基础分类器)
    │
    └── RockIdentifier (高级识别器)
        ├── 继承所有基础分类方法
        ├── 添加温度压力校正功能
        └── 添加迭代识别算法
```

## 3. 核心类详细说明

### 3.1 RockProperties (rocks.py)

**用途**: 存储岩石的基本物性参数

**属性**:
- `vp`: P波速度 (km/s) - 必需
- `vs`: S波速度 (km/s) - 可选
- `density`: 密度 (g/cm³) - 可选
- `porosity`: 孔隙度 (0-1) - 可选
- `temperature`: 温度 (°C) - 默认25
- `pressure`: 压力 (MPa) - 默认200
- `fluid_saturation`: 流体饱和度 (0-1) - 默认0.5
- `tectonic_setting`: 构造背景 - 默认UNKNOWN

**使用场景**: 
- 作为其他类的输入参数
- 存储单次测量的物性数据

### 3.2 Rock (rocks.py)

**用途**: 岩石对象，提供物性计算和转换功能

**主要方法**:

1. **物性计算**
   - `vp_vs_ratio`: 计算Vp/Vs比值
   - `poisson_ratio`: 计算泊松比

2. **经验公式计算**
   - `calculate_density(method)`: 
     - `'gardner'`: ρ = 0.31 × (Vp in m/s)^0.25 (Gardner et al., 1974)
     - `'nafe_drake'`: 多项式公式 (Nafe & Drake, 1963)
     - `'brocher'`: 多项式公式 (Brocher, 2005，基于Nafe-Drake曲线拟合)
   
   - `calculate_vs(method)`:
     - `'brocher'`: Vs = 0.7858 - 1.2344×Vp + 0.7949×Vp² - 0.1238×Vp³ + 0.0064×Vp⁴
     - `'castagna'`: Vs = (Vp - 1.36) / 1.16 (Castagna et al., 1985)

3. **弹性模量计算**
   - `calculate_elastic_moduli()`: 计算体积模量、剪切模量、杨氏模量、拉梅常数

4. **环境效应**
   - `calculate_temperature_effect()`: 温度对速度的影响
   - `calculate_pressure_effect()`: 压力对速度的影响

### 3.3 RockDatabase (rocks.py)

**用途**: 管理实验室测量的岩石物性数据库

**核心功能**:

1. **数据管理**
   - `load_from_excel()`: 从Excel加载数据
   - `save_to_excel()`: 保存数据到Excel
   - `add_measurement()`: 添加单个测量
   - `add_measurements_from_excel()`: 批量添加
   - `remove_measurement()`: 删除测量

2. **数据查询**
   - `get_consensus_properties()`: 获取共识值（加权平均）
   - `filter_measurements()`: 按条件筛选
   - `get_rocks_by_tectonic_setting()`: 按构造环境查询
   - `get_measurement_sources()`: 获取数据来源列表

3. **数据分析**
   - `get_database_summary()`: 获取数据库摘要
   - `get_tectonic_distribution()`: 获取构造分布
   - `plot_measurement_comparison()`: 绘制测量对比图
   - `plot_tectonic_distribution()`: 绘制构造分布热力图

**数据质量控制**:
- 自动计算质量分数
- 检查数据一致性（变异系数）
- 支持按质量分数筛选

### 3.4 RockClassifier (rocks.py)

**用途**: 基于机器学习的岩石分类器

**算法**: 随机森林分类器 (RandomForestClassifier)

**特征**: vp, vs, density

**主要方法**:

1. `classify_by_vp(vp)`: 
   - 基于P波速度的最近邻分类
   - 返回最匹配的岩石类型

2. `classify_by_vp_vs(vp, vs)`:
   - 基于P波和S波速度的欧氏距离分类
   - 返回最匹配的岩石类型

3. `classify_with_uncertainty(vp, threshold)`:
   - 带不确定性的分类
   - 返回可能的岩石类型及其概率
   - 基于速度差异阈值

4. `classify_by_vp_and_setting(vp, setting)`:
   - 考虑构造环境的分类
   - 先在特定构造环境中搜索，再扩展到全部数据

### 3.5 RockIdentifier (isrock.py)

**用途**: 高级岩石识别器，包含温度压力校正功能

**继承关系**: `RockIdentifier(RockClassifier)`

**核心特性**:

1. **自动数据校正**
   - 初始化时自动将训练数据校正到标准条件（25°C, 200MPa）
   - 使用两步校正：先标准系数，再特定岩石类型系数

2. **温度压力校正方法**

   **`pressure_correction(velocity, pressure, target_pressure, rock_type, is_s_wave)`**
   ```
   校正公式: V_corrected = V_original × (1 + β × ΔP)
   其中: ΔP = target_pressure - original_pressure
        β = 0.0002 (P波) 或 0.00015 (S波)
   ```

   **`temperature_correction(velocity, temperature, target_temperature, rock_type, is_s_wave)`**
   ```
   校正公式: V_corrected = V_original × (1 - α × ΔT)
   其中: ΔT = original_temperature - target_temperature
        α = 根据岩石类型选择（范围: -0.39e-4 到 -0.68e-4 /°C）
   ```

   **`density_correction(density, pressure, temperature, target_pressure, target_temperature)`**
   ```
   校正公式: ρ_corrected = ρ_original × (1 + β_p × ΔP) × (1 + α_T × ΔT)
   ```

3. **识别方法**

   **`identify_rock(vp, vs, density, porosity, tectonic_setting, min_probability, max_candidates)`**
   - 输入: 物性参数和构造环境
   - 输出: 候选岩石列表（按概率排序）
   - 考虑构造环境对概率的调整

   **`identify_velocity_model(model_data, min_probability)`**
   - 两步迭代识别算法：
     1. 使用标准系数进行初步校正和识别
     2. 根据初步识别结果，使用特定岩石类型系数进行精确校正和最终识别
   - 考虑构造环境对识别结果的影响
   - 返回每个深度点的识别结果

## 4. 数据流

### 4.1 单样本识别流程

```
输入物性参数 (vp, vs, density, porosity)
    ↓
温度压力校正 (如果需要)
    ↓
特征标准化
    ↓
随机森林分类器
    ↓
概率计算
    ↓
构造环境调整
    ↓
输出候选岩石列表
```

### 4.2 速度模型识别流程

```
速度模型数据 (包含深度、压力、温度信息)
    ↓
第一步：标准系数校正
    ├── 压力校正
    ├── 温度校正
    └── 密度校正
    ↓
初步识别
    ↓
第二步：特定岩石类型系数校正
    ├── 根据初步识别结果选择校正系数
    ├── 精确压力校正
    ├── 精确温度校正
    └── 精确密度校正
    ↓
最终识别
    ↓
构造环境调整
    ↓
输出识别结果（每个深度点的岩石类型）
```

## 5. 关键算法

### 5.1 温度压力校正算法

**原理**: 将不同温压条件下的测量数据校正到标准条件（25°C, 200MPa），以便进行统一比较和识别。

**校正顺序**:
1. 先进行压力校正（压力对速度的影响较大）
2. 再进行温度校正（温度对速度的影响较小）

**校正系数来源**:
- 主要基于 Christensen (1979) 的实验室测量数据
- 不同岩石类型有不同的温度校正系数

### 5.2 迭代识别算法

**目的**: 提高识别精度，通过迭代校正逐步逼近真实岩石类型。

**步骤**:
1. **初步识别**: 使用标准校正系数，对所有样本进行统一校正和识别
2. **精确校正**: 根据初步识别结果，选择特定岩石类型的校正系数
3. **最终识别**: 使用精确校正后的数据进行最终识别

**优势**:
- 考虑了不同岩石类型对温压的响应差异
- 提高了识别精度
- 减少了校正误差

### 5.3 构造环境约束

**原理**: 不同构造环境下，某些岩石类型的出现概率不同。

**调整策略**:
- 大洋地壳: 增加辉长岩和玄武岩的概率（×1.3）
- 俯冲带: 增加橄榄岩和榴辉岩的概率（×1.3）
- 大陆地壳: 增加花岗岩和片麻岩的概率

## 6. 数据格式规范

### 6.1 Excel数据库格式

**必需列**:
- `rock_type`: 岩石类型（字符串）
- `vp`: P波速度 (km/s)
- `source`: 数据来源
- `method`: 测量方法

**可选列**:
- `vs`: S波速度 (km/s)
- `density`: 密度 (g/cm³)
- `porosity`: 孔隙度 (0-1)
- `temperature`: 温度 (°C)
- `pressure`: 压力 (MPa)
- `fluid_saturation`: 流体饱和度 (0-1)
- `tectonic_setting`: 构造环境（字符串，需匹配TectonicSetting枚举值）
- `date`: 测量日期 (YYYY-MM-DD)
- `notes`: 备注
- `location`: 采样位置
- `tectonic_description`: 构造背景描述

### 6.2 速度模型数据格式

**必需字段**:
- `vp`: P波速度数组 (km/s)
- `vs`: S波速度数组 (km/s)
- `density`: 密度数组 (g/cm³)
- `porosity`: 孔隙度数组 (0-1)
- `pressure`: 压力数组 (MPa)
- `temperature`: 温度数组 (°C)
- `tectonic_setting`: 构造环境数组 (TectonicSetting枚举)

**可选字段**:
- `depth`: 深度数组 (km)

## 7. 使用示例

### 7.1 基础使用

```python
from pyAOBS.utils import Rock, create_common_rock, RockProperties

# 创建岩石对象
granite = create_common_rock('granite')
print(f"Vp/Vs ratio: {granite.vp_vs_ratio}")
print(f"Poisson ratio: {granite.poisson_ratio}")

# 计算弹性模量
moduli = granite.calculate_elastic_moduli()
print(f"Bulk modulus: {moduli['bulk_modulus']} GPa")
```

### 7.2 数据库管理

```python
from pyAOBS.utils import load_rock_database

# 加载数据库
db = load_rock_database('rocks.xlsx')

# 获取共识值
granite_props = db.get_consensus_properties('GRANITE', min_quality_score=0.7)

# 筛选数据
high_quality = db.filter_measurements('GRANITE', min_quality_score=0.8)

# 可视化
db.plot_measurement_comparison('GRANITE', 'vp', 'granite_vp_comparison.png')
```

### 7.3 岩石识别

```python
from pyAOBS.utils.isrock import RockIdentifier, TectonicSetting

# 创建识别器
identifier = RockIdentifier('rocks.xlsx')
identifier.train_classifier()

# 识别单个样本
result = identifier.identify_rock(
    vp=6.1,
    vs=3.5,
    density=2.65,
    porosity=0.02,
    tectonic_setting=TectonicSetting.CONTINENTAL_CRUST
)

print(f"最可能的岩石类型: {result['candidates'][0]['rock_type']}")
print(f"概率: {result['candidates'][0]['probability']:.2f}")
```

### 7.4 速度模型识别

```python
import numpy as np
from pyAOBS.utils.isrock import RockIdentifier, TectonicSetting

# 创建速度模型数据
model_data = {
    'vp': np.array([6.1, 6.5, 7.5, 8.1]),
    'vs': np.array([3.5, 3.7, 4.1, 4.5]),
    'density': np.array([2.65, 2.9, 3.0, 3.3]),
    'porosity': np.array([0.02, 0.02, 0.02, 0.02]),
    'pressure': np.array([100.0, 150.0, 200.0, 250.0]),
    'temperature': np.array([20.0, 25.0, 30.0, 35.0]),
    'tectonic_setting': [
        TectonicSetting.CONTINENTAL_CRUST,
        TectonicSetting.CONTINENTAL_CRUST,
        TectonicSetting.OCEANIC_CRUST,
        TectonicSetting.SUBDUCTION_ZONE
    ]
}

# 识别
identifier = RockIdentifier('rocks.xlsx')
results = identifier.identify_velocity_model(model_data, min_probability=0.1)

# 可视化结果
identifier.plot_identification_results(results, 'model_results.png')
```

## 8. 扩展性

### 8.1 添加新的岩石类型

1. 在Excel数据库中添加新的岩石类型数据
2. 如需特定温度校正系数，在`CorrectionParameters`中添加
3. 重新训练分类器

### 8.2 添加新的校正公式

1. 在`CorrectionParameters`类中添加新的校正系数
2. 在`RockIdentifier`的校正方法中使用新系数

### 8.3 自定义分类器

可以替换`RockClassifier`中的随机森林分类器为其他算法：
- 支持sklearn的任意分类器
- 只需实现`fit()`和`predict_proba()`方法

## 9. 性能优化建议

1. **数据预处理**: 在初始化时进行数据校正，避免重复计算
2. **批量处理**: 使用`identify_velocity_model()`批量识别，比循环调用`identify_rock()`更高效
3. **缓存结果**: 对于重复的识别任务，可以缓存校正后的数据
4. **特征选择**: 根据实际需求选择特征，减少不必要的计算

## 10. 常见问题

### Q1: 为什么需要温度压力校正？

A: 实验室测量数据通常在不同温压条件下获得，而实际地下的温压条件也不同。为了统一比较和识别，需要将所有数据校正到标准条件。

### Q2: 校正系数如何选择？

A: 校正系数主要基于Christensen (1979)的实验室测量数据。如果知道岩石类型，使用特定系数；否则使用默认系数。

### Q3: 识别结果不准确怎么办？

A: 
1. 检查输入数据是否正确
2. 确认温压条件是否准确
3. 检查数据库是否包含相关岩石类型
4. 调整`min_probability`阈值
5. 考虑构造环境信息

### Q4: 如何提高识别精度？

A:
1. 增加数据库中的样本数量
2. 使用更准确的温压条件
3. 提供构造环境信息
4. 使用迭代识别算法（`identify_velocity_model`）

## 11. 未来改进方向

1. **深度学习分类器**: 使用神经网络替代随机森林
2. **不确定性量化**: 更详细的概率分布和置信区间
3. **多参数融合**: 整合更多物性参数（如电导率、磁性等）
4. **实时校正**: 支持在线校正和识别
5. **可视化增强**: 3D可视化、交互式图表
