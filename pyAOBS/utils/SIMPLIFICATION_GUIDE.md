# 简化策略说明

## 问题分析

原始代码结构复杂的原因：
1. **多层抽象**：RockProperties → Rock → RockDatabase → RockClassifier → RockIdentifier
2. **功能分散**：数据管理、分类、校正等功能分散在不同类中
3. **使用复杂**：需要了解多个类的接口和初始化顺序

## 简化策略

### 1. 创建统一的高级接口

**新增文件**: `simple_rock_classifier.py`

**核心思想**：
- 隐藏所有复杂的内部实现
- 提供"一键式"API
- 自动处理参数估算、校正、分类等所有步骤

### 2. 使用对比

#### 原始方式（复杂）

```python
# 需要了解多个类
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
    'pressure': pressure,
    'temperature': temperature,
    'tectonic_setting': settings
}

# 需要手动调用
results = identifier.identify_velocity_model(model_data)
```

#### 简化方式（推荐）

```python
# 只需一个函数
from pyAOBS.utils import classify_velocity_model

# 一键完成所有操作
results = classify_velocity_model(
    vp=[6.1, 6.5, 7.5],
    vs=[3.5, 3.7, 4.1],
    depth=[0, 10, 20]  # 自动估算压力和温度
)
```

### 3. 自动处理的功能

简化接口自动处理以下内容：

1. **参数估算**
   - 缺少Vs → 自动使用Brocher公式估算
   - 缺少密度 → 自动使用Gardner公式估算
   - 提供深度 → 自动估算压力和温度

2. **数据校正**
   - 自动进行温度压力校正
   - 自动选择校正系数
   - 自动校正到标准条件

3. **分类器管理**
   - 自动加载数据库
   - 自动训练分类器
   - 延迟加载（只在需要时初始化）

4. **结果格式化**
   - 自动转换为DataFrame
   - 自动添加深度、压力、温度等信息

## 接口设计

### 三层接口设计

```
┌─────────────────────────────────────┐
│  便捷函数层 (最简单)                │
│  classify_velocity_model()          │
│  classify_from_file()               │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  简化类层 (中等复杂度)               │
│  SimpleRockClassifier               │
│  - 自动处理所有细节                  │
│  - 提供更多控制选项                  │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  完整功能层 (高级用户)               │
│  RockIdentifier, RockDatabase等    │
│  - 完全控制                          │
│  - 自定义参数                        │
└─────────────────────────────────────┘
```

### 推荐使用路径

1. **大多数用户** → 使用便捷函数
   ```python
   classify_velocity_model()
   ```

2. **需要更多控制** → 使用SimpleRockClassifier
   ```python
   classifier = SimpleRockClassifier(auto_correct=True)
   ```

3. **特殊需求** → 使用完整接口
   ```python
   identifier = RockIdentifier(...)
   ```

## 代码简化示例

### 示例1: 单点分类

**原始方式**:
```python
from pyAOBS.utils.isrock import RockIdentifier
identifier = RockIdentifier('rocks.xlsx')
identifier.train_classifier()
result = identifier.identify_rock(
    vp=6.1, vs=3.5, density=2.65,
    porosity=0.02,
    tectonic_setting=TectonicSetting.CONTINENTAL_CRUST
)
rock_type = result['candidates'][0]['rock_type']
```

**简化方式**:
```python
from pyAOBS.utils import classify_velocity_model
rock_type = classify_velocity_model(vp=6.1, vs=3.5, density=2.65)
```

### 示例2: 模型分类

**原始方式**:
```python
# 需要准备完整的数据结构
model_data = {
    'vp': vp,
    'vs': vs,
    'density': density,
    'porosity': porosity,
    'pressure': pressure,
    'temperature': temperature,
    'tectonic_setting': settings
}
results = identifier.identify_velocity_model(model_data)
# 需要手动提取和格式化结果
```

**简化方式**:
```python
# 只需提供速度数据
results = classify_velocity_model(
    vp=vp,
    vs=vs,
    depth=depth  # 自动估算压力和温度
)
# 直接得到格式化的DataFrame
```

## 保留的功能

简化接口**保留了所有核心功能**：
- ✅ 温度压力校正
- ✅ 迭代识别算法
- ✅ 构造环境约束
- ✅ 概率计算
- ✅ 多候选结果

只是**隐藏了实现细节**，让使用更简单。

## 迁移指南

### 从完整接口迁移到简化接口

**步骤1**: 替换导入
```python
# 旧
from pyAOBS.utils.isrock import RockIdentifier

# 新
from pyAOBS.utils import classify_velocity_model
```

**步骤2**: 简化调用
```python
# 旧
identifier = RockIdentifier('rocks.xlsx')
identifier.train_classifier()
results = identifier.identify_velocity_model(model_data)

# 新
results = classify_velocity_model(vp=vp, vs=vs, depth=depth)
```

**步骤3**: 简化结果处理
```python
# 旧：需要从复杂的结果结构中提取
for rock_type, data_list in results.items():
    for data in data_list:
        ...

# 新：直接使用DataFrame
print(results[['depth', 'rock_type', 'probability']])
```

## 性能考虑

简化接口的性能与完整接口相同，因为：
- 底层使用相同的分类器
- 延迟加载避免不必要的初始化
- 自动缓存训练好的分类器

## 总结

**简化策略的核心**：
1. ✅ **统一接口** - 一个函数完成所有操作
2. ✅ **自动处理** - 自动估算、校正、分类
3. ✅ **智能默认** - 合理的默认值和自动选择
4. ✅ **保留功能** - 不损失任何核心功能
5. ✅ **向后兼容** - 完整接口仍然可用

**使用建议**：
- 🎯 **新用户** → 直接使用简化接口
- 🎯 **大多数场景** → 使用简化接口
- 🎯 **特殊需求** → 使用完整接口
