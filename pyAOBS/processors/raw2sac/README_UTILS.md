# Utils工具模块完整说明文档

## 概述

本文档描述了 `processors/raw2sac/` 目录下所有工具模块（Utils）的完整功能和使用方法。这些工具模块为OBS Raw、OBEM TSM、SAC to SEGY等格式转换提供通用的工具函数，便于后续整合到 pyAOBS 代码包中。

## 模块架构

```
processors/raw2sac/
├── format_utils.py       # 统一格式转换工具模块（核心）
├── segy_utils.py         # SEGY文件写入工具模块
├── ukooa_utils.py        # UKOOA文件解析工具模块
├── config_utils.py       # 配置文件读取工具模块
└── three_bytes_utils.py  # 3字节转换工具（已整合到format_utils.py）
```

### 模块依赖关系

```
sac2y_v2_1_obspy.py
    ├── format_utils.py      (通用格式转换工具)
    ├── segy_utils.py        (SEGY工具)
    │   └── format_utils.py  (导入 ieee_to_ibm_float_vectorized)
    ├── ukooa_utils.py       (UKOOA工具)
    │   └── format_utils.py  (导入 get_subs)
    └── config_utils.py      (配置工具)

raw2sac_v1_1_obspy.py
    └── format_utils.py      (导入时间解析、3字节转换等)

obem_tsm_to_sac_obspy.py
    └── format_utils.py      (导入3字节转换)
```

---

## 1. format_utils.py - 统一格式转换工具模块

**文件路径**: `processors/raw2sac/format_utils.py`

**功能概述**: 整合了所有格式转换的通用工具函数，是核心工具模块。

### 1.1 十六进制转换工具

#### `hexto_int(hex_char: str) -> int`

将单个十六进制字符转换为整数（0-15）。

**参数**:
- `hex_char`: 单个十六进制字符（0-9, A-F, a-f）

**返回**: 整数（0-15）

**示例**:
```python
from format_utils import hexto_int
value = hexto_int('A')  # 返回 10
value = hexto_int('F')  # 返回 15
```

#### `hexto_dec(hex_str: str) -> int`

将十六进制字符串转换为十进制整数。

**参数**:
- `hex_str`: 十六进制字符串（如 "5DD", "FFF"）

**返回**: 十进制整数值

**示例**:
```python
from format_utils import hexto_dec
value = hexto_dec("5DD")  # 返回 1501
value = hexto_dec("FFF")  # 返回 4095
```

### 1.2 文件名时间解析工具

#### `parse_datetime_from_filename(filename: str) -> Dict[str, int]`

从文件名前8个十六进制字符解析日期和时间。

**参数**:
- `filename`: 文件名，格式为 `AABBCCDD.YYY` 或仅前8个十六进制字符

**返回**: 包含 `year, mon, day, hour, min, sec, date0` 的字典

**示例**:
```python
from format_utils import parse_datetime_from_filename
dt_dict = parse_datetime_from_filename("51A6D159.5DD")
# 返回: {'year': 2025, 'mon': 1, 'day': 6, 'hour': 13, 'min': 21, 'sec': 89, 'date0': 5}
```

#### `parse_milliseconds(filename: str, pcik: float) -> float`

从文件名解析毫秒值。

**参数**:
- `filename`: 文件名，格式为 `AABBCCDD.YYY`
- `pcik`: 内部时钟频率（TC / 256.0）

**返回**: 实际毫秒值（秒）

**示例**:
```python
from format_utils import parse_milliseconds
pcik = 524288.0 / 256.0  # TC / 256.0
msec = parse_milliseconds("51A6D159.5DD", pcik)
```

### 1.3 3字节有符号整数转换工具

#### `three_bytes_to_int_vectorized(data_bytes: np.ndarray, byte_order: str = 'hml') -> np.ndarray`

批量将3字节数组转换为有符号32位整数（向量化版本）。

**参数**:
- `data_bytes`: shape为 `(n, 3)` 的uint8数组，每行3个字节
- `byte_order`: 字节顺序，`'hml'`（高-中-低，OBS Raw格式）或 `'lmh'`（低-中-高，OBEM TSM格式）

**返回**: shape为 `(n,)` 的int32数组

**字节顺序说明**:
- `'hml'`: OBS Raw格式，文件中字节顺序为（高字节、中字节、低字节）
- `'lmh'`: OBEM TSM格式，文件中字节顺序为（低字节、中字节、高字节）

**示例**:
```python
import numpy as np
from format_utils import three_bytes_to_int_vectorized

# OBS Raw格式
data = np.array([[0x80, 0x00, 0x01], [0x7F, 0xFF, 0xFF]], dtype=np.uint8)
result = three_bytes_to_int_vectorized(data, byte_order='hml')

# OBEM TSM格式
data = np.array([[0x01, 0x00, 0x80], [0xFF, 0xFF, 0x7F]], dtype=np.uint8)
result = three_bytes_to_int_vectorized(data, byte_order='lmh')
```

**性能优化**: 
- 默认使用纯NumPy向量化版本（性能优秀）
- 可选启用Numba JIT编译加速：
  ```bash
  export THREE_BYTES_USE_NUMBA=1
  python your_script.py
  ```

### 1.4 字节交换工具

#### `swap_bytes_2(value: int) -> int`

2字节字节序转换（little-endian ↔ big-endian）。

**参数**:
- `value`: 2字节整数值

**返回**: 字节序转换后的值

#### `swap_bytes_4(value: int) -> int`

4字节字节序转换（little-endian ↔ big-endian）。

**参数**:
- `value`: 4字节整数值

**返回**: 字节序转换后的值

### 1.5 浮点格式转换工具

#### `ieee_to_ibm_float_vectorized(ieee_data: np.ndarray) -> np.ndarray`

向量化IEEE→IBM浮点转换（高度优化）。

**参数**:
- `ieee_data`: IEEE浮点数组（float32）

**返回**: IBM浮点数组（uint32，big-endian格式）

**说明**: 
- 对应C代码的`float_to_ibm()`函数
- 支持批量转换，性能优秀
- 注意：IBM浮点转换是固有有损的，精度损失约3位

**示例**:
```python
import numpy as np
from format_utils import ieee_to_ibm_float_vectorized

ieee_data = np.array([1.0, -2.5, 3.14159], dtype=np.float32)
ibm_data = ieee_to_ibm_float_vectorized(ieee_data)
```

### 1.6 时间字符串解析工具

#### `parse_time_string(time_str: str) -> Dict[str, int]`

解析时间字符串（格式：年/月/日 时:分:秒）。

**支持的格式**:
- `"2017/02/27 21:18:54"`
- `"2017/2/27 21:18:54"`
- `"2017-02-27 21:18:54"`（也支持-分隔符）

**参数**:
- `time_str`: 时间字符串

**返回**: 包含 `year, month, day, hour, minute, second` 的字典

**示例**:
```python
from format_utils import parse_time_string
time_dict = parse_time_string("2017/02/27 21:18:54")
# 返回: {'year': 2017, 'month': 2, 'day': 27, 'hour': 21, 'minute': 18, 'second': 54}
```

### 1.7 字符串工具函数

#### `get_subs(line: str, begin: int, end: int) -> str`

提取字符串的子串（去除空格）。

**参数**:
- `line`: 输入字符串
- `begin`: 起始位置（从1开始）
- `end`: 结束位置（从1开始，包含）

**返回**: 提取的子串（去除空格）

**注意**: begin和end是从1开始的（与C代码保持一致）

**示例**:
```python
from format_utils import get_subs
line = "S  123 45 67 89"
sub = get_subs(line, 5, 7)  # 返回 "123"
```

### 1.8 文件读取工具

#### `read_location_file(loc_file: str) -> Tuple[float, float]`

从文件读取位置信息。

**文件格式**: `lon lat`（空格分隔）

**参数**:
- `loc_file`: 位置文件路径

**返回**: `(longitude, latitude)` 元组

**示例**:
```python
from format_utils import read_location_file
lon, lat = read_location_file("loc.in")
```

### 1.9 坐标投影工具

**依赖**: 需要安装 `pyproj` 库：`pip install pyproj`

#### `setup_tm_projection(lon0: float) -> Optional[Transformer]`

设置横墨卡托投影（TM）。

**参数**:
- `lon0`: 投影中心经度

**返回**: pyproj Transformer对象，如果pyproj不可用则返回None

**说明**: 使用WGS-84椭球，与C代码保持一致

**示例**:
```python
from format_utils import setup_tm_projection
transformer = setup_tm_projection(120.0)
```

#### `lonlat_to_utm(lon: float, lat: float, transformer: Transformer, ex: float = 0.0, ey: float = 0.0) -> Tuple[float, float]`

经纬度转UTM坐标（单点转换）。

**参数**:
- `lon`: 经度（度）
- `lat`: 纬度（度）
- `transformer`: pyproj转换器（由`setup_tm_projection`创建）
- `ex`: X方向偏移（米，默认0.0）
- `ey`: Y方向偏移（米，默认0.0）

**返回**: `(x, y)` UTM坐标（米）

**说明**:
- 如果lon < 0，自动加360度
- X坐标加500000米（UTM偏移）
- 南半球Y坐标加10000000米

**示例**:
```python
from format_utils import setup_tm_projection, lonlat_to_utm
transformer = setup_tm_projection(120.0)
x, y = lonlat_to_utm(120.3, 20.5, transformer, ex=0.0, ey=0.0)
```

#### `lonlat_to_utm_batch(lons: np.ndarray, lats: np.ndarray, transformer: Transformer, ex: float = 0.0, ey: float = 0.0) -> Tuple[np.ndarray, np.ndarray]`

批量转换坐标（向量化）。

**参数**:
- `lons`: 经度数组
- `lats`: 纬度数组
- `transformer`: pyproj转换器
- `ex`: X方向偏移（米，默认0.0）
- `ey`: Y方向偏移（米，默认0.0）

**返回**: `(xs, ys)` UTM坐标数组

**示例**:
```python
import numpy as np
from format_utils import setup_tm_projection, lonlat_to_utm_batch

transformer = setup_tm_projection(120.0)
lons = np.array([120.3, 120.5, 120.7])
lats = np.array([20.5, 20.6, 20.7])
xs, ys = lonlat_to_utm_batch(lons, lats, transformer)
```

### 1.10 时间处理工具

**依赖**: 需要安装 `obspy` 库

#### `calculate_drift_rate(drift_params: Dict) -> Tuple[float, int, int]`

计算OBS漂移率。

**参数**:
- `drift_params`: 漂移参数字典，包含：
  - `styear, stmonth, stday, sthour, stmin, stsec`: 开始时间
  - `edyear, edmonth, edday, edhour, edmin, edsec`: 结束时间
  - `drift`: 漂移量（毫秒）

**返回**: `(drift_rate, stjday, edjday)` 元组
- `drift_rate`: 漂移率（ms/s）
- `stjday`: 开始记录儒略日（相对于年初）
- `edjday`: 结束记录儒略日（相对于年初）

**说明**: 使用ObsPy的`UTCDateTime.julday`属性计算一年中的第几天

**示例**:
```python
from format_utils import calculate_drift_rate

drift_params = {
    'styear': 2017, 'stmonth': 2, 'stday': 27,
    'sthour': 21, 'stmin': 18, 'stsec': 54,
    'edyear': 2017, 'edmonth': 6, 'edday': 12,
    'edhour': 2, 'edmin': 5, 'edsec': 0,
    'drift': 123.456  # 毫秒
}
drift_rate, stjday, edjday = calculate_drift_rate(drift_params)
```

#### `calculate_shot_time(shot_info: Dict, sac_ref_time, drift_params: Dict, drift_rate: float, rv: float, offset: float, et: float, sthour: int, stmin: int, stsec: int, stmsec: int, stjday: int) -> float`

计算炮点时间（含各种校正），返回相对于参考时间的秒数。

**参数**:
- `shot_info`: 炮点信息字典，包含 `hour, minute, sec, msec, day`
- `sac_ref_time`: SAC文件参考时间（需要`julday`属性）
- `drift_params`: 漂移参数字典
- `drift_rate`: 漂移率（ms/s）
- `rv`: 声速（m/s）
- `offset`: 偏移距（米）
- `et`: 时间偏移（秒）
- `sthour, stmin, stsec, stmsec`: 开始记录时间
- `stjday`: 开始记录儒略日

**返回**: 校正后的炮点时间（秒，从当天0时开始）

**说明**: 包含偏移时间校正、漂移校正和时间偏移校正

### 1.11 数据提取工具

#### `extract_traces_batch(sac_data: np.ndarray, start_samples: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]`

批量提取多个道的数据（高度优化）。

**参数**:
- `sac_data`: SAC数据数组（1D）
- `start_samples`: 起始采样点数组
- `n_samples`: 每个道的采样点数

**返回**: `(traces, valid_mask)` 元组
- `traces`: shape为 `(n_traces, n_samples)` 的数据数组
- `valid_mask`: 有效道的布尔数组

**示例**:
```python
import numpy as np
from format_utils import extract_traces_batch

sac_data = np.random.randn(100000).astype(np.float32)
start_samples = np.array([0, 1000, 2000, 3000])
n_samples = 500
traces, valid_mask = extract_traces_batch(sac_data, start_samples, n_samples)
```

---

## 2. segy_utils.py - SEGY文件写入工具模块

**文件路径**: `processors/raw2sac/segy_utils.py`

**功能概述**: 提供SEGY格式文件的创建和写入功能。

**依赖**: 需要从 `format_utils` 导入 `ieee_to_ibm_float_vectorized`

### 2.1 EBCDIC头写入

#### `write_ebcdic_header(f) -> None`

写入EBCDIC头（3200字节）。

**参数**:
- `f`: 文件对象（二进制写入模式）

**说明**: SEGY标准要求3200字节的EBCDIC文本头，这里写入空白字符（可以后续自定义）

### 2.2 二进制头创建和写入

#### `create_segy_binary_header(delta: float, trace_length: float, n_traces: int = 0) -> Dict`

创建SEGY二进制头。

**参数**:
- `delta`: 采样间隔（秒）
- `trace_length`: 道长度（秒）
- `n_traces`: 道数（稍后更新，默认0）

**返回**: 二进制头字典

**示例**:
```python
from segy_utils import create_segy_binary_header
binary_header = create_segy_binary_header(delta=0.002, trace_length=6.0, n_traces=0)
```

#### `write_binary_header(f, binary_header: Dict) -> None`

写入SEGY二进制头（400字节，big-endian）。

**参数**:
- `f`: 文件对象（二进制写入模式）
- `binary_header`: 二进制头字典

### 2.3 道头创建和写入

#### `create_segy_trace_header(shot_info: Dict, sx: float, sy: float, gx: float, gy: float, offset: float, gelev: float, swdep: int, delrt: int, ns: int, dt: int, tracl: int, tracr: int, year: int, day: int, hour: int, minute: int, sec: int, timbas: int) -> Dict`

创建SEGY道头。

**参数**:
- `shot_info`: 炮点信息字典
- `sx, sy`: 炮点坐标（UTM，米）
- `gx, gy`: 接收点坐标（UTM，米）
- `offset`: 偏移距（米）
- `gelev`: 接收点高程（米）
- `swdep`: 炮点水深（米）
- `delrt`: 延迟记录时间（毫秒）
- `ns`: 采样点数
- `dt`: 采样间隔（微秒）
- `tracl`: 道序号（线内）
- `tracr`: 道序号（文件内）
- `year, day, hour, minute, sec, timbas`: 时间信息

**返回**: 道头字典

**示例**:
```python
from segy_utils import create_segy_trace_header

shot_info = {
    'shot_number': 100,
    'field_record': 10
}
trace_header = create_segy_trace_header(
    shot_info=shot_info,
    sx=500000.0, sy=3000000.0,
    gx=500100.0, gy=3000100.0,
    offset=141.42,
    gelev=-3000.0,
    swdep=3000,
    delrt=-2000,
    ns=3000,
    dt=2000,
    tracl=1,
    tracr=1,
    year=2017,
    day=100,
    hour=12,
    minute=0,
    sec=0,
    timbas=0
)
```

#### `write_trace_header(f, trace_header: Dict) -> None`

写入SEGY道头（240字节，big-endian）。

**参数**:
- `f`: 文件对象（二进制写入模式）
- `trace_header`: 道头字典

### 2.4 完整SEGY文件写入

#### `write_segy_file(output_file: str, traces_data: List[np.ndarray], traces_headers: List[Dict], binary_header: Dict, verbose: bool = False) -> None`

写入完整SEGY文件（优化版本）。

**参数**:
- `output_file`: 输出SEGY文件路径
- `traces_data`: 道数据列表（IEEE浮点）
- `traces_headers`: 道头列表
- `binary_header`: 二进制头字典
- `verbose`: 是否输出详细信息（默认False）

**功能**:
1. 写入EBCDIC头（3200字节）
2. 预留二进制头位置
3. 批量转换所有数据为IBM格式
4. 写入道头和数据
5. 更新二进制头（包含道数）

**示例**:
```python
import numpy as np
from segy_utils import create_segy_binary_header, create_segy_trace_header, write_segy_file

# 创建二进制头
binary_header = create_segy_binary_header(delta=0.002, trace_length=6.0)

# 准备数据和道头
traces_data = [np.random.randn(3000).astype(np.float32) for _ in range(10)]
traces_headers = [...]  # 道头列表

# 写入文件
write_segy_file("output.segy", traces_data, traces_headers, binary_header, verbose=True)
```

---

## 3. ukooa_utils.py - UKOOA文件解析工具模块

**文件路径**: `processors/raw2sac/ukooa_utils.py`

**功能概述**: 提供UKOOA格式炮点文件的解析功能。

**依赖**: 需要从 `format_utils` 导入 `get_subs`

### 3.1 单行UKOOA数据解析

#### `parse_ukooa_line(line: str, sac_year: int, sac_nzjday: int) -> Optional[Dict]`

解析单行UKOOA数据。

**UKOOA格式字段位置（从1开始计数）**:
- 位置1: 'S'或'N'
- 位置5-7: 儒略日
- 位置8-9: 小时
- 位置10-11: 分钟
- 位置12-13: 秒
- 位置15-17: 毫秒
- 位置20-24: 炮点号
- 位置26-35: 纬度（度分秒）
- 位置36-45: 经度（度分秒）
- 位置65-68: 水深
- 位置72-75: 野外记录号

**参数**:
- `line`: UKOOA文件的一行
- `sac_year`: SAC文件年份
- `sac_nzjday`: SAC文件儒略日

**返回**: 炮点信息字典，如果无效返回None

**返回的字典包含**:
- `day`: 儒略日
- `year`: 年份
- `hour, minute, sec, msec`: 时间信息
- `shot_number`: 炮点号
- `latitude, longitude`: 经纬度（度）
- `water_depth`: 水深（米）
- `field_record`: 野外记录号

### 3.2 UKOOA文件解析

#### `parse_ukooa_file(ukooa_file: str, sac_year: int, sac_nzjday: int) -> List[Dict]`

解析UKOOA格式的炮点文件。

**参数**:
- `ukooa_file`: UKOOA文件路径
- `sac_year`: SAC文件年份
- `sac_nzjday`: SAC文件儒略日

**返回**: 炮点信息列表（过滤掉无效炮点和不在时间范围内的炮点）

**示例**:
```python
from ukooa_utils import parse_ukooa_file

shots = parse_ukooa_file("shots.ukooa", sac_year=2017, sac_nzjday=100)
for shot in shots:
    print(f"Shot {shot['shot_number']}: lat={shot['latitude']:.6f}, lon={shot['longitude']:.6f}")
```

---

## 4. config_utils.py - 配置文件读取工具模块

**文件路径**: `processors/raw2sac/config_utils.py`

**功能概述**: 提供统一配置文件（INI格式）的读取功能。

### 4.1 配置文件读取

#### `read_unified_config_file(config_file: str) -> Dict`

读取统一配置文件（INI格式）。

**配置文件格式**:
```ini
[Parameters]
rv = 8.0
length = 6.0
tcoor = 2.0
lon0 = 120.0
ex = 0.0
ey = 0.0
et = 1.885

[Drift]
start_time = 2017/02/27 21:18:54
end_time = 2017/06/12 02:05:00
drift = 123.456

[Location]
longitude = 120.3
latitude = 20.5
location_file = loc.in
```

**参数**:
- `config_file`: 配置文件路径

**返回**: 配置字典，包含各个节（section）的配置

**特性**:
- 支持多个配置节（section）
- 自动类型转换（整数、浮点数、布尔值）
- 时间字符串保持为字符串（包含'/'或':'的字符串）
- 忽略注释行（以#开头）
- 节名和键名自动转换为小写

**示例**:
```python
from config_utils import read_unified_config_file

config = read_unified_config_file("config.ini")
params = config['parameters']  # 访问 [Parameters] 节
drift = config['drift']        # 访问 [Drift] 节
location = config['location']  # 访问 [Location] 节

rv = params['rv']              # 获取 rv 参数
start_time = drift['start_time']  # 获取开始时间字符串
```

---

## 完整使用示例

### 示例1：从OBS Raw文件名解析时间

```python
from format_utils import parse_datetime_from_filename, parse_milliseconds

filename = "51A6D159.5DD"
pcik = 524288.0 / 256.0

# 解析日期时间
dt_dict = parse_datetime_from_filename(filename)
print(f"Year: {dt_dict['year']}, Month: {dt_dict['mon']}, Day: {dt_dict['day']}")

# 解析毫秒
msec = parse_milliseconds(filename, pcik)
print(f"Milliseconds: {msec:.3f} seconds")
```

### 示例2：转换3字节数据

```python
import numpy as np
from format_utils import three_bytes_to_int_vectorized

# OBS Raw格式（hml顺序）
data_hml = np.array([[0x80, 0x00, 0x01], [0x7F, 0xFF, 0xFF]], dtype=np.uint8)
result = three_bytes_to_int_vectorized(data_hml, byte_order='hml')
print(f"Result: {result}")

# OBEM TSM格式（lmh顺序）
data_lmh = np.array([[0x01, 0x00, 0x80], [0xFF, 0xFF, 0x7F]], dtype=np.uint8)
result = three_bytes_to_int_vectorized(data_lmh, byte_order='lmh')
print(f"Result: {result}")
```

### 示例3：坐标投影

```python
from format_utils import setup_tm_projection, lonlat_to_utm, lonlat_to_utm_batch
import numpy as np

# 设置投影
transformer = setup_tm_projection(lon0=120.0)

# 单点转换
x, y = lonlat_to_utm(120.3, 20.5, transformer)
print(f"UTM coordinates: x={x:.1f}m, y={y:.1f}m")

# 批量转换
lons = np.array([120.3, 120.5, 120.7])
lats = np.array([20.5, 20.6, 20.7])
xs, ys = lonlat_to_utm_batch(lons, lats, transformer)
print(f"UTM coordinates: xs={xs}, ys={ys}")
```

### 示例4：解析UKOOA文件

```python
from ukooa_utils import parse_ukooa_file

shots = parse_ukooa_file("shots.ukooa", sac_year=2017, sac_nzjday=100)
print(f"Found {len(shots)} shots")

for shot in shots[:5]:  # 打印前5个炮点
    print(f"Shot {shot['shot_number']}: "
          f"time={shot['hour']:02d}:{shot['minute']:02d}:{shot['sec']:02d}, "
          f"lat={shot['latitude']:.6f}°, lon={shot['longitude']:.6f}°")
```

### 示例5：写入SEGY文件

```python
import numpy as np
from segy_utils import create_segy_binary_header, create_segy_trace_header, write_segy_file

# 创建二进制头
binary_header = create_segy_binary_header(delta=0.002, trace_length=6.0)

# 准备数据
traces_data = []
traces_headers = []

for i in range(10):
    # 创建道数据
    trace_data = np.random.randn(3000).astype(np.float32)
    traces_data.append(trace_data)
    
    # 创建道头
    shot_info = {'shot_number': i+1, 'field_record': 1}
    trace_header = create_segy_trace_header(
        shot_info=shot_info,
        sx=500000.0, sy=3000000.0,
        gx=500100.0, gy=3000100.0,
        offset=141.42,
        gelev=-3000.0,
        swdep=3000,
        delrt=-2000,
        ns=3000,
        dt=2000,
        tracl=i+1,
        tracr=i+1,
        year=2017,
        day=100,
        hour=12,
        minute=0,
        sec=0,
        timbas=0
    )
    traces_headers.append(trace_header)

# 写入文件
write_segy_file("output.segy", traces_data, traces_headers, binary_header, verbose=True)
```

---

## 依赖关系

### 必需依赖

- **NumPy**: 所有数值计算和数组操作
- **ObsPy**: 时间处理函数（`calculate_drift_rate`, `calculate_shot_time`）
- **pyproj**: 坐标投影函数（`setup_tm_projection`, `lonlat_to_utm`, `lonlat_to_utm_batch`）

### 可选依赖

- **Numba**: 3字节转换性能优化（通过环境变量启用）

### 安装依赖

```bash
# 必需依赖
pip install numpy obspy pyproj

# 可选依赖（性能优化）
pip install numba
```

---

## 性能优化说明

### 1. 向量化操作

所有工具函数都使用NumPy向量化操作，性能优秀：
- 3字节转换：批量处理，避免循环
- 坐标投影：批量转换，减少函数调用开销
- 数据提取：批量提取，预分配数组

### 2. Numba JIT编译（可选）

对于3字节转换，可以启用Numba JIT编译加速：

```bash
export THREE_BYTES_USE_NUMBA=1
python your_script.py
```

**注意**: Numba版本在某些情况下可能反而更慢，因此默认禁用。仅在明确需要时才启用。

### 3. 文件I/O优化

SEGY文件写入使用大缓冲区（4MB）提高I/O性能：
```python
BUFFER_SIZE = 4 * 1024 * 1024  # 4MB
with open(output_file, 'wb', buffering=BUFFER_SIZE) as f:
    ...
```

---

## 注意事项

1. **Julian Day**: 使用ObsPy的`UTCDateTime.julday`属性替代自定义函数，该属性表示一年中的第几天（相对于年初）。

2. **字节序**: 
   - SEGY格式使用big-endian字节序
   - 3字节数据转换时需要注意字节顺序（`hml` vs `lmh`）

3. **IBM浮点转换**: 
   - IBM浮点转换是固有有损的，精度损失约3位
   - 使用RMS相对误差（1%容忍度）进行验证

4. **坐标投影**: 
   - 使用WGS-84椭球，与C代码保持一致
   - 经度小于0时自动加360度
   - 南半球Y坐标自动加10000000米

5. **错误处理**: 
   - 所有函数都包含输入验证
   - 无效输入返回NaN或None，而不是抛出异常（在某些情况下）

---

## 模块使用统计

### 使用 format_utils.py 的脚本

- `raw2sac_v1_1_obspy.py`: 文件名时间解析、3字节转换
- `obem_tsm_to_sac_obspy.py`: 3字节转换
- `sac2y_v2_1_obspy.py`: 时间字符串解析、坐标投影、字节交换、IEEE到IBM转换、时间处理、数据提取

### 使用 segy_utils.py 的脚本

- `sac2y_v2_1_obspy.py`: SEGY文件写入

### 使用 ukooa_utils.py 的脚本

- `sac2y_v2_1_obspy.py`: UKOOA文件解析

### 使用 config_utils.py 的脚本

- `sac2y_v2_1_obspy.py`: 配置文件读取

---

## 版本历史

- **2025/01/XX**: 初始版本，整合所有工具函数到统一模块
- 从独立的工具脚本（`fntime_v1_1.py`, `three_bytes_utils.py`等）提取通用函数
- 简化`sac2y_v2_1_obspy.py`，移除重复代码
- 创建模块化的工具包结构

---

## 作者信息

- **Author**: Haibo Huang
- **Date**: 2025/12/01
- **Purpose**: 统一格式转换工具，便于整合到 pyAOBS 代码包

---

## 参考文档

- SAC格式文档：SAC (Seismic Analysis Code) 格式规范
- SEGY格式文档：SEG Y Rev 1标准
- UKOOA格式文档：UKOOA炮点文件格式规范
- ObsPy文档：https://docs.obspy.org/
- pyproj文档：https://pyproj4.github.io/pyproj/

---

## 许可证

本文档和代码遵循项目许可证（请参考项目根目录的LICENSE文件）。

