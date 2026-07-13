# 快速开始指南 - 简化的岩石分类接口

## 概述

为了简化使用，我们提供了**一键式**的简化接口，您只需要提供速度数据，即可自动完成岩石分类。

## 最简单的使用方式

### 方式1: 使用便捷函数（推荐）

```python
from pyAOBS.utils import classify_velocity_model

# 分类单个点
rock_type = classify_velocity_model(vp=6.1, vs=3.5, density=2.65)
print(f"岩石类型: {rock_type}")

# 分类整个速度模型
results = classify_velocity_model(
    vp=[6.1, 6.5, 7.5, 8.1],
    vs=[3.5, 3.7, 4.1, 4.5],
    depth=[0, 10, 20, 30]  # 可选，用于估算压力和温度
)

print(results)
# 输出:
#      vp   vs  density rock_type  probability  depth
# 0  6.1  3.5     2.65   GRANITE         0.85      0
# 1  6.5  3.7     2.90    BASALT         0.78     10
# 2  7.5  4.1     3.00    GABBRO         0.82     20
# 3  8.1  4.5     3.30  PERIDOTITE        0.91     30
```

### 方式2: 使用SimpleRockClassifier类

```python
from pyAOBS.utils import SimpleRockClassifier

# 创建分类器（自动加载数据库）
classifier = SimpleRockClassifier()

# 分类单个点
rock_type = classifier.classify(vp=6.1, vs=3.5, density=2.65)
print(f"岩石类型: {rock_type}")

# 分类速度模型
results = classifier.classify_model(
    vp=[6.1, 6.5, 7.5],
    vs=[3.5, 3.7, 4.1],
    density=[2.65, 2.9, 3.0]
)
print(results)
```

### 方式3: 从文件加载并分类

```python
from pyAOBS.utils import classify_from_file

# 从CSV/Excel文件加载并分类
results = classify_from_file(
    'velocity_model.csv',
    vp_column='Vp',           # P波速度列名
    vs_column='Vs',           # S波速度列名（可选）
    depth_column='depth',     # 深度列名（可选）
    output_file='classified_model.csv'  # 保存结果
)
```

## 完整示例

### 示例1: 基本分类

```python
import numpy as np
from pyAOBS.utils import classify_velocity_model

# 创建速度模型数据
vp = np.array([6.1, 6.5, 7.5, 8.1])  # P波速度 (km/s)
vs = np.array([3.5, 3.7, 4.1, 4.5])  # S波速度 (km/s)
depth = np.array([0, 10, 20, 30])     # 深度 (km)

# 一键分类
results = classify_velocity_model(
    vp=vp,
    vs=vs,
    depth=depth
)

# 查看结果
print(results[['depth', 'vp', 'rock_type', 'probability']])
```

### 示例2: 带温度压力校正的分类

```python
from pyAOBS.utils import SimpleRockClassifier

classifier = SimpleRockClassifier(auto_correct=True)

# 提供温度和压力信息，自动进行校正
results = classifier.classify_model(
    vp=[6.1, 6.5, 7.5],
    vs=[3.5, 3.7, 4.1],
    density=[2.65, 2.9, 3.0],
    pressure=[100, 200, 300],      # 压力 (MPa)
    temperature=[20, 50, 100]      # 温度 (°C)
)

print(results)
```

### 示例3: 从文件分类

```python
from pyAOBS.utils import classify_from_file

# 假设您有一个速度模型文件 velocity_model.csv:
# depth,Vp,Vs,density
# 0,6.1,3.5,2.65
# 10,6.5,3.7,2.90
# 20,7.5,4.1,3.00

results = classify_from_file(
    'velocity_model.csv',
    vp_column='Vp',
    vs_column='Vs',
    density_column='density',
    depth_column='depth',
    output_file='classified_results.csv'
)

print(results)
```

### 示例4: 只提供P波速度（自动估算其他参数）

```python
from pyAOBS.utils import classify_velocity_model

# 只提供P波速度，自动估算Vs和密度
results = classify_velocity_model(
    vp=[6.1, 6.5, 7.5, 8.1],
    depth=[0, 10, 20, 30]
)

print(results)
# 系统会自动：
# 1. 使用Brocher公式估算Vs
# 2. 使用Gardner公式估算密度
# 3. 根据深度估算压力和温度
# 4. 进行温度压力校正
# 5. 进行岩石分类
```

## 参数说明

### classify_velocity_model() 参数

- `vp` (必需): P波速度，可以是单个值或数组
- `vs` (可选): S波速度，如果不提供会自动估算
- `density` (可选): 密度，如果不提供会自动估算
- `depth` (可选): 深度，用于估算压力和温度
- `pressure` (可选): 压力 (MPa)，如果不提供会根据深度估算
- `temperature` (可选): 温度 (°C)，如果不提供会根据深度估算
- `database_file` (可选): 数据库文件路径，默认使用rocks.xlsx

### SimpleRockClassifier 参数

- `database_file`: 数据库文件路径
- `auto_correct`: 是否自动进行温度压力校正（默认True）
- `standard_pressure`: 标准压力 (MPa)，默认200.0
- `standard_temperature`: 标准温度 (°C)，默认25.0

## 输出格式

### 单个点分类
返回字符串：岩石类型名称，如 `'GRANITE'`

### 模型分类
返回DataFrame，包含以下列：
- `vp`: P波速度
- `vs`: S波速度
- `density`: 密度
- `rock_type`: 岩石类型
- `probability`: 分类概率
- `depth`: 深度（如果提供）
- `pressure`: 压力（如果提供或估算）
- `temperature`: 温度（如果提供或估算）

## 自动功能

简化接口会自动处理以下内容：

1. **参数估算**：
   - 如果缺少Vs，使用Brocher公式估算
   - 如果缺少密度，使用Gardner公式估算
   - 如果提供深度，自动估算压力和温度

2. **温度压力校正**：
   - 如果提供压力或温度信息，自动进行校正
   - 校正到标准条件（25°C, 200MPa）

3. **数据库加载**：
   - 自动查找默认数据库文件（rocks.xlsx）
   - 自动训练分类器

## 与完整接口的对比

### 简化接口（推荐）
```python
from pyAOBS.utils import classify_velocity_model
results = classify_velocity_model(vp=[6.1, 6.5], vs=[3.5, 3.7])
```

### 完整接口（高级用户）
```python
from pyAOBS.utils.isrock import RockIdentifier
identifier = RockIdentifier('rocks.xlsx')
identifier.train_classifier()
model_data = {...}
results = identifier.identify_velocity_model(model_data)
```

**简化接口的优势**：
- ✅ 代码更简洁
- ✅ 自动处理所有细节
- ✅ 适合大多数使用场景
- ✅ 学习成本低

**完整接口的优势**：
- ✅ 更多控制选项
- ✅ 可以自定义校正参数
- ✅ 适合特殊需求

## 常见问题

**Q: 如果我只知道P波速度怎么办？**
A: 没问题！系统会自动估算S波速度和密度：
```python
rock_type = classify_velocity_model(vp=6.1)
```

**Q: 如何指定数据库文件？**
A: 使用database_file参数：
```python
results = classify_velocity_model(
    vp=[6.1, 6.5],
    database_file='my_rocks.xlsx'
)
```

**Q: 如何禁用温度压力校正？**
A: 使用SimpleRockClassifier并设置auto_correct=False：
```python
classifier = SimpleRockClassifier(auto_correct=False)
```

**Q: 如何获取分类概率？**
A: 使用return_probabilities参数：
```python
result = classifier.classify(
    vp=6.1, vs=3.5,
    return_probabilities=True
)
print(result['probability'])
```

## 下一步

- 查看 [README.md](README.md) 了解完整功能
- 查看 [CODE_STRUCTURE.md](CODE_STRUCTURE.md) 了解详细实现
- 查看测试代码 [test_isrock.py](test_isrock.py) 了解更多示例
