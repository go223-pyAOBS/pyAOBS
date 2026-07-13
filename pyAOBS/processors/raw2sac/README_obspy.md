# raw2sac_v1_1_obspy.py - Python实现（使用ObsPy）

基于ObsPy库的Python实现，将OBS原始数据转换为SAC格式。

## 安装依赖

```bash
pip install obspy numpy
```

## 使用方法

```bash
python raw2sac_v1_1_obspy.py fileName sps TC
```

### 参数说明

- `fileName`: 输入原始数据文件名（格式：`AABBCCDD.YYY`，12字符十六进制格式）
- `sps`: 采样率 (Hz)
- `TC`: 时钟参数（从A*.LOG文件读取）

### 示例

```bash
python raw2sac_v1_1_obspy.py 51A6D159.5DD 100.0 3145727336.0
```

### 输出文件

**4通道文件**（宽带+水听器）：
- `XXXXXXXXX.shx` - SHX分量
- `XXXXXXXXX.shy` - SHY分量
- `XXXXXXXXX.shz` - SHZ分量
- `XXXXXXXXX.hyd` - HYD分量

**3通道文件**（短周期）：
- `XXXXXXXXX.shx` - SHX分量
- `XXXXXXXXX.shy` - SHY分量
- `XXXXXXXXX.shz` - SHZ分量

## 功能特点

- ✅ **使用ObsPy库**：自动处理SAC文件格式，无需手动构建632字节头文件
- ✅ **代码简洁**：约300行代码（C版本约560行）
- ✅ **复用代码**：复用`fntime_v1_1.py`的日期时间解析函数
- ✅ **可靠性高**：ObsPy经过大量测试，兼容性极佳
- ✅ **易于维护**：模块化设计，清晰的代码结构

## 与C版本的区别

| 特性 | C版本 | Python版本（ObsPy） |
|------|-------|-------------------|
| **依赖** | 无（标准C库） | ObsPy + NumPy |
| **代码量** | ~560行 | ~300行 |
| **SAC头文件** | 手动构建632字节 | ObsPy自动处理 |
| **维护成本** | 高 | 低 |
| **扩展性** | 有限 | 强 |

## 技术细节

### 3字节整数转换

实现与C版本完全一致的3字节整数转换（符号扩展）：

```python
def three_bytes_to_int(low_byte, mid_byte, high_byte) -> int:
    # 符号扩展高字节
    if high_byte & 0x80:  # 负数
        signed_high = high_byte | 0xFFFFFF00
    else:  # 正数
        signed_high = high_byte
    
    # 组合：low | (mid << 8) | (signed_high << 16)
    value = low_byte | (mid_byte << 8) | (signed_high << 16)
    
    # 转换为有符号32位
    if value & 0x800000:
        value = value - 0x1000000
    
    return value
```

### SAC头文件设置

使用ObsPy的`Trace`对象自动处理SAC头文件：

```python
from obspy import Trace
from obspy.core import AttribDict

trace = Trace(data=np.array(data, dtype=np.float32))
trace.stats.sac = AttribDict()
trace.stats.sac.kstnm = 'OBS'
trace.stats.sac.kcmpnm = 'SHX'
trace.stats.sac.cmpinc = 90.0
# ... 其他字段
trace.write('output.sac', format='SAC')
```

## 错误处理

- 文件名格式验证
- 文件存在性检查
- 参数有效性验证
- 详细的错误信息

## 注意事项

1. **文件名格式**：输入文件名必须符合`AABBCCDD.YYY`格式（8个十六进制字符 + 点号 + 3个十六进制字符）
2. **依赖安装**：确保已安装ObsPy和NumPy
3. **文件权限**：确保有写入当前目录的权限
4. **输出文件名**：输出文件名基于输入文件名的前9个字符（与C版本一致）

## 测试

建议使用与C版本相同的测试数据进行比较验证。

## 许可证

与原始C程序保持一致。

