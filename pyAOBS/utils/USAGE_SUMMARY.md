# 使用方式总结

## 🎯 推荐使用方式（最简单）

### 一键式函数接口

```python
from pyAOBS.utils import classify_velocity_model

# 分类单个点
rock_type = classify_velocity_model(vp=6.1, vs=3.5, density=2.65)

# 分类速度模型
results = classify_velocity_model(
    vp=[6.1, 6.5, 7.5],
    vs=[3.5, 3.7, 4.1],
    depth=[0, 10, 20]  # 自动估算压力和温度
)
```

**优点**：
- ✅ 代码最简洁（只需1-3行）
- ✅ 自动处理所有细节
- ✅ 自动估算缺失参数
- ✅ 自动进行温度压力校正

## 📋 三种使用方式对比

### 方式1: 便捷函数（推荐 ⭐）

**适用场景**: 大多数用户，快速分类

```python
from pyAOBS.utils import classify_velocity_model

# 最简单
rock_type = classify_velocity_model(vp=6.1)

# 提供更多信息
results = classify_velocity_model(
    vp=[6.1, 6.5, 7.5],
    vs=[3.5, 3.7, 4.1],
    depth=[0, 10, 20]
)
```

**特点**：
- 代码最少
- 自动处理一切
- 适合90%的使用场景

### 方式2: SimpleRockClassifier类

**适用场景**: 需要更多控制选项

```python
from pyAOBS.utils import SimpleRockClassifier

classifier = SimpleRockClassifier(
    database_file='rocks.xlsx',
    auto_correct=True
)

# 分类单个点
rock_type = classifier.classify(vp=6.1, vs=3.5)

# 分类模型
results = classifier.classify_model(
    vp=[6.1, 6.5, 7.5],
    vs=[3.5, 3.7, 4.1]
)
```

**特点**：
- 可以控制数据库文件
- 可以禁用自动校正
- 可以获取概率分布

### 方式3: 完整接口（高级）

**适用场景**: 特殊需求，需要完全控制

```python
from pyAOBS.utils.isrock import RockIdentifier

identifier = RockIdentifier('rocks.xlsx')
identifier.train_classifier()
# ... 更多自定义操作
```

**特点**：
- 完全控制
- 可以自定义校正参数
- 适合特殊需求

## 🔄 从复杂到简单的迁移

### 原始代码（复杂）

```python
# 需要导入多个类
from pyAOBS.utils.isrock import RockIdentifier, TectonicSetting
from pyAOBS.utils.rocks import RockClassifier

# 需要手动初始化
identifier = RockIdentifier('rocks.xlsx')
identifier.train_classifier()

# 需要手动准备数据
model_data = {
    'vp': vp,
    'vs': vs,
    'density': density,
    'porosity': porosity,
    'pressure': pressure,
    'temperature': temperature,
    'tectonic_setting': [TectonicSetting.CONTINENTAL_CRUST] * len(vp)
}

# 需要手动调用
results = identifier.identify_velocity_model(model_data)

# 需要手动提取结果
for rock_type, data_list in results.items():
    ...
```

### 简化代码（推荐）

```python
# 只需一个导入
from pyAOBS.utils import classify_velocity_model

# 一键完成
results = classify_velocity_model(
    vp=vp,
    vs=vs,
    depth=depth  # 自动估算压力和温度
)

# 直接使用DataFrame
print(results[['depth', 'rock_type', 'probability']])
```

## 📊 功能对比表

| 功能 | 便捷函数 | SimpleRockClassifier | 完整接口 |
|------|---------|---------------------|---------|
| 代码简洁度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| 自动参数估算 | ✅ | ✅ | ❌ |
| 自动校正 | ✅ | ✅ (可选) | ✅ |
| 控制选项 | ⭐ | ⭐⭐ | ⭐⭐⭐ |
| 学习成本 | 低 | 中 | 高 |
| 适用场景 | 90%用户 | 需要控制 | 特殊需求 |

## 🚀 快速开始示例

### 示例1: 最简单的使用

```python
from pyAOBS.utils import classify_velocity_model

# 只提供P波速度，其他自动估算
rock_type = classify_velocity_model(vp=6.1)
print(f"岩石类型: {rock_type}")
```

### 示例2: 提供更多信息

```python
results = classify_velocity_model(
    vp=[6.1, 6.5, 7.5],
    vs=[3.5, 3.7, 4.1],
    density=[2.65, 2.9, 3.0],
    depth=[0, 10, 20]
)

print(results)
```

### 示例3: 从文件加载

```python
from pyAOBS.utils import classify_from_file

results = classify_from_file(
    'velocity_model.csv',
    vp_column='Vp',
    depth_column='depth',
    output_file='results.csv'
)
```

## 💡 使用建议

1. **新用户**: 直接使用 `classify_velocity_model()`
2. **大多数场景**: 使用便捷函数或 `SimpleRockClassifier`
3. **特殊需求**: 使用完整接口

## 📚 相关文档

- [QUICK_START.md](QUICK_START.md) - 快速开始指南
- [README.md](README.md) - 完整功能说明
- [CODE_STRUCTURE.md](CODE_STRUCTURE.md) - 代码结构说明
- [SIMPLIFICATION_GUIDE.md](SIMPLIFICATION_GUIDE.md) - 简化策略说明
