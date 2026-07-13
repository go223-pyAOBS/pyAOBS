# SAC2Y v2.1 Python实现策略

## 一、概述

本文档描述如何将 `sac2y_v2.1.c` 转换为Python实现，利用现有的代码库和Python生态系统。

### 1.1 功能目标

将SAC格式的地震数据转换为SEGY格式，核心功能：
- 读取单个连续记录的SAC文件
- 解析UKOOA格式的炮点文件
- 根据炮点时间和位置信息从SAC文件中截取数据段
- 进行时间校正（偏移、漂移、延迟）
- 坐标投影（经纬度→UTM）
- 格式转换（SAC→SEGY，IEEE浮点→IBM浮点）
- 字节序转换（little-endian→big-endian）

### 1.2 复用代码库

- **`raw2sac_v1_1_obspy.py`**:
  - SAC文件读取经验
  - 时间解析函数（`julian_day`, `parse_datetime_from_filename`等）
  - 数值转换经验

- **`su_processor.py`**:
  - SEGY/SU文件处理经验
  - 道头字段映射
  - 数据网格化经验

## 二、技术栈选择

### 2.1 核心库

1. **ObsPy** (`obspy`):
   - 读取SAC文件：`obspy.read()` 或 `obspy.io.sac.SACTrace()`
   - 写入SEGY文件：`obspy.Stream.write(..., format='SEGY')`
   - 时间处理：`UTCDateTime`

2. **pyproj** (`pyproj`):
   - UTM/TM投影：`pyproj.Transformer`
   - WGS-84椭球参数支持
   - 比C代码中的自定义投影函数更可靠

3. **NumPy** (`numpy`):
   - 数组操作和数值计算
   - 高效的数据截取和处理

4. **struct** (标准库):
   - 字节序转换
   - 二进制数据处理

### 2.2 可选库

- **segyio**: 如果ObsPy的SEGY写入不满足需求，可以使用segyio进行底层控制

## 三、模块设计

### 3.1 文件结构

```
sac2y_v2_1_obspy.py
├── 参数和配置类
│   ├── read_parameter_file()
│   └── read_drift_file()
├── UKOOA解析模块
│   ├── parse_ukooa_file()
│   └── parse_ukooa_line()
├── 坐标投影模块
│   ├── setup_utm_projection()
│   ├── lonlat_to_utm()
│   └── 使用pyproj或自定义TM投影
├── 时间处理模块
│   ├── calculate_reference_time()
│   ├── calculate_drift_rate()
│   ├── calculate_shot_time()
│   └── 复用fntime_v1_1.py的julian_day()
├── 数据截取模块
│   ├── extract_trace_from_sac()
│   └── calculate_sample_position()
├── 格式转换模块
│   ├── ieee_to_ibm_float()
│   ├── swap_bytes_2()
│   ├── swap_bytes_4()
│   └── create_segy_binary_header()
├── SEGY写入模块
│   ├── write_ebcdic_header()
│   ├── write_binary_header()
│   ├── write_trace_header()
│   └── write_trace_data()
└── 主函数
    └── convert_sac_to_segy()
```

### 3.2 核心函数设计

#### 3.2.1 UKOOA文件解析

```python
def parse_ukooa_file(ukooa_file: str) -> List[Dict]:
    """解析UKOOA格式的炮点文件
    
    Args:
        ukooa_file: UKOOA文件路径
    
    Returns:
        List[Dict]: 每个炮点的信息字典
        [
            {
                'shot_number': int,      # 炮点号
                'year': int,             # 年
                'day': int,              # 儒略日
                'hour': int,             # 时
                'minute': int,           # 分
                'sec': int,              # 秒
                'msec': int,             # 毫秒
                'latitude': float,       # 纬度（度）
                'longitude': float,      # 经度（度）
                'water_depth': int,      # 水深（米）
                'field_record': int      # 野外记录号
            },
            ...
        ]
    """
```

#### 3.2.2 坐标投影

```python
from pyproj import Transformer

def setup_utm_transformer(lon0: float) -> Transformer:
    """设置UTM/TM投影转换器
    
    Args:
        lon0: 投影中心经度
    
    Returns:
        Transformer: pyproj转换器
    """
    # 使用pyproj创建自定义TM投影
    # 参数对应C代码中的WGS-84椭球和TM投影
    
def lonlat_to_utm(lon: float, lat: float, lon0: float, 
                  ex: float = 0.0, ey: float = 0.0) -> Tuple[float, float]:
    """经纬度转UTM坐标（带偏移）
    
    Args:
        lon: 经度（度）
        lat: 纬度（度）
        lon0: UTM投影中心经度
        ex: X方向偏移（米）
        ey: Y方向偏移（米）
    
    Returns:
        Tuple[float, float]: (x, y) UTM坐标（米）
    """
```

#### 3.2.3 时间校正

```python
def calculate_shot_time(shot_info: Dict, sac_ref_time: UTCDateTime,
                       drift_rate: float, rv: float, offset: float,
                       et: float, tcoor: float) -> UTCDateTime:
    """计算炮点时间（含各种校正）
    
    Args:
        shot_info: 炮点信息字典
        sac_ref_time: SAC文件参考时间
        drift_rate: 漂移率（ms/s）
        rv: 声速（m/s）
        offset: 偏移距（米）
        et: 时间偏移（秒）
        tcoor: 延迟校正（秒）
    
    Returns:
        UTCDateTime: 校正后的炮点时间
    """
    # 1. 基础时间
    # 2. 偏移时间（根据距离和声速）
    # 3. 漂移校正
    # 4. 时间偏移
    # 5. 延迟校正
```

#### 3.2.4 数据截取

```python
def extract_trace_from_sac(sac_stream: Stream, shot_time: UTCDateTime,
                           trace_length: float, tcoor: float,
                           sac_ref_time: UTCDateTime) -> Optional[np.ndarray]:
    """从SAC文件中截取指定时间段的数据
    
    Args:
        sac_stream: ObsPy Stream对象（包含SAC数据）
        shot_time: 炮点时间（已校正）
        trace_length: 道长度（秒）
        tcoor: 延迟校正（秒）
        sac_ref_time: SAC文件参考时间
    
    Returns:
        Optional[np.ndarray]: 截取的数据数组，如果时间超出范围返回None
    """
    # 1. 计算采样点位置
    # 2. 检查边界
    # 3. 提取数据
    # 4. 确保长度正确
```

#### 3.2.5 IBM浮点转换

```python
def ieee_to_ibm_float(ieee_data: np.ndarray) -> np.ndarray:
    """IEEE浮点转IBM浮点格式
    
    Args:
        ieee_data: IEEE浮点数组（float32）
    
    Returns:
        np.ndarray: IBM浮点数组（uint32，big-endian）
    
    算法（对应C代码float_to_ibm函数）：
    1. IEEE: 符号位(1) + 指数位(8) + 尾数位(23)
    2. IBM: 符号位(1) + 指数位(7) + 尾数位(24，十六进制)
    3. 转换指数基数
    4. 调整尾数对齐
    """
```

#### 3.2.6 字节序转换

```python
def swap_bytes_2(value: int) -> int:
    """2字节字节序转换（little-endian ↔ big-endian）
    
    对应C代码：up_side_down2()
    """
    return ((value >> 8) & 0x00FF) | ((value << 8) & 0xFF00)

def swap_bytes_4(value: int) -> int:
    """4字节字节序转换（little-endian ↔ big-endian）
    
    对应C代码：up_side_down4()
    """
    return ((value << 24) & 0xFF000000) | \
           ((value << 8) & 0x00FF0000) | \
           ((value >> 8) & 0x0000FF00) | \
           ((value >> 24) & 0x000000FF)
```

## 四、实现细节

### 4.1 SAC文件读取

使用ObsPy读取SAC文件：

```python
from obspy import read
from obspy.io.sac import SACTrace

# 方法1：使用ObsPy的read函数
sac_stream = read(sac_file)
sac_trace = sac_stream[0]  # 通常只有一个trace

# 方法2：直接使用SACTrace（更底层控制）
sac_trace = SACTrace.read(sac_file)

# 获取头文件信息
sac_stats = sac_trace.stats
sac_ref_time = sac_stats.starttime
delta = sac_stats.delta  # 采样间隔
npts = sac_stats.npts    # 数据点数

# 获取数据
sac_data = sac_trace.data  # numpy数组，float32
```

### 4.2 UKOOA文件解析

```python
def parse_ukooa_line(line: str) -> Optional[Dict]:
    """解析单行UKOOA数据
    
    UKOOA格式字段位置（从1开始计数）：
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
    """
    if len(line) < 75 or (line[0] != 'S' and line[0] != 'N'):
        return None
    
    # 提取各个字段
    shot_info = {}
    shot_info['day'] = int(line[4:7].strip())
    shot_info['hour'] = int(line[7:9].strip())
    shot_info['minute'] = int(line[9:11].strip())
    shot_info['sec'] = int(line[11:13].strip())
    shot_info['msec'] = int(line[14:17].strip())
    shot_info['shot_number'] = int(line[19:24].strip())
    
    # 解析纬度
    lat_deg = int(line[25:27].strip())
    lat_min = int(line[27:29].strip())
    lat_sec1 = int(line[29:31].strip())
    lat_sec2 = int(line[32:34].strip())
    lat_sec = lat_sec1 + lat_sec2 / 100.0
    shot_info['latitude'] = lat_deg + lat_min/60.0 + lat_sec/3600.0
    if line[34] == 'S':
        shot_info['latitude'] = -shot_info['latitude']
    
    # 解析经度（类似）
    # ...
    
    return shot_info
```

### 4.3 UTM投影

使用pyproj库实现：

```python
from pyproj import Transformer, CRS

def setup_tm_projection(lon0: float) -> Transformer:
    """设置横墨卡托投影（TM）
    
    对应C代码中的vtm()和tm()函数
    """
    # 定义WGS-84椭球
    # 定义TM投影（以lon0为中心经线）
    # 创建转换器
    
def lonlat_to_utm(lon: float, lat: float, lon0: float,
                  ex: float = 0.0, ey: float = 0.0) -> Tuple[float, float]:
    """经纬度转UTM坐标
    
    对应C代码中的utm()函数
    """
    # 1. 设置TM投影
    # 2. 转换坐标
    # 3. 添加UTM偏移（X+500000，南半球Y+10000000）
    # 4. 添加ex, ey偏移
```

### 4.4 数据截取

```python
def extract_trace_from_sac(sac_data: np.ndarray, sac_ref_time: UTCDateTime,
                           shot_time: UTCDateTime, trace_length: float,
                           delta: float, tcoor: float) -> Optional[Tuple[int, int]]:
    """计算数据截取位置
    
    Args:
        sac_data: SAC数据数组
        sac_ref_time: SAC参考时间
        shot_time: 炮点时间（已校正）
        trace_length: 道长度（秒）
        delta: 采样间隔（秒）
        tcoor: 延迟校正（秒）
    
    Returns:
        Optional[Tuple[int, int]]: (起始采样点, 采样点数)，如果无效返回None
    
    对应C代码逻辑：
    t4_temp = (int)((1.0/sp.delta)*(t3-t1)-(1.0/sp.delta)*tcoor+0.5);
    samn_temp = (int)(((double)length)/sp.delta+0.5);
    """
    # 计算时间差（秒）
    time_diff = (shot_time - sac_ref_time) - tcoor
    
    # 计算起始采样点位置
    start_sample = int((time_diff / delta) + 0.5)
    
    # 计算需要的采样点数
    n_samples = int((trace_length / delta) + 0.5)
    
    # 检查边界
    if start_sample < 0:
        return None  # 数据还未开始
    
    if start_sample + n_samples > len(sac_data):
        return None  # 超出数据范围
    
    return (start_sample, n_samples)
```

### 4.5 IBM浮点转换

```python
def ieee_to_ibm_float(ieee_data: np.ndarray) -> np.ndarray:
    """IEEE浮点转IBM浮点格式（向量化版本）
    
    对应C代码float_to_ibm()函数
    
    Args:
        ieee_data: IEEE浮点数组（float32）
    
    Returns:
        np.ndarray: IBM浮点数组（uint32，big-endian格式）
    """
    # 将IEEE浮点转换为整数表示
    ieee_int = ieee_data.view(np.uint32)
    
    # 提取符号位、指数和尾数
    sign = ieee_int & 0x80000000
    exponent = (ieee_int & 0x7F800000) >> 23
    mantissa = (ieee_int & 0x007FFFFF) | 0x00800000  # 添加隐含的1
    
    # 转换指数：IEEE偏移127 → IBM偏移64（十六进制）
    # IEEE指数范围：-126到127
    # IBM指数范围：-64到63（十六进制）
    ibm_exponent = (exponent - 127 + 64) >> 2
    ibm_mantissa_shift = (exponent - 127 + 64) & 0x3
    
    # 调整尾数（IBM使用十六进制尾数）
    if ibm_mantissa_shift > 0:
        mantissa = mantissa >> ibm_mantissa_shift
    
    # 组合IBM浮点数
    ibm_float = sign | (ibm_exponent << 24) | (mantissa & 0x00FFFFFF)
    
    # 转换为big-endian
    ibm_float_be = swap_bytes_4_vectorized(ibm_float)
    
    return ibm_float_be.view(np.uint32)
```

### 4.6 SEGY文件写入

有两种方案：

#### 方案A：使用ObsPy（推荐，简单）

```python
from obspy import Stream, Trace
from obspy.io.segy import SEGYTraceHeader, SEGYBinaryFileHeader

def write_segy_with_obspy(output_file: str, traces_data: List[np.ndarray],
                          trace_headers: List[Dict], binary_header: Dict):
    """使用ObsPy写入SEGY文件"""
    stream = Stream()
    
    for i, (data, header) in enumerate(zip(traces_data, trace_headers)):
        # 创建Trace
        trace = Trace(data=data.astype(np.float32))
        
        # 设置基本统计信息
        trace.stats.delta = header['delta']
        trace.stats.npts = len(data)
        
        # 设置SEGY头
        trace.stats.segy = {}
        trace.stats.segy.trace_header = SEGYTraceHeader()
        # 填充各个字段...
        
        stream.append(trace)
    
    # 写入文件
    stream.write(output_file, format='SEGY', byteorder='>')  # big-endian
```

**问题**：ObsPy可能不完全支持IBM浮点格式和所有字节序细节。

#### 方案B：使用segyio（推荐，更底层控制）

```python
import segyio

def write_segy_with_segyio(output_file: str, traces_data: List[np.ndarray],
                           trace_headers: List[Dict], binary_header: Dict):
    """使用segyio写入SEGY文件（更底层控制）"""
    with segyio.open(output_file, 'w', ignore_geometry=True) as segyfile:
        # 设置二进制头
        segyfile.bin = {
            segyio.BinField.Samples: binary_header['hns'],
            segyio.BinField.Interval: binary_header['hdt'],
            segyio.BinField.Format: 1,  # IBM浮点
            # ...
        }
        
        # 写入道头和数据
        for i, (data, header) in enumerate(zip(traces_data, trace_headers)):
            # 设置道头
            segyfile.header[i] = {
                segyio.TraceField.TRACE_SEQUENCE_LINE: header['tracl'],
                segyio.TraceField.offset: header['offset'],
                # ...
            }
            
            # 写入数据（需要转换为IBM格式）
            ibm_data = ieee_to_ibm_float(data)
            segyfile.trace[i] = ibm_data
```

**问题**：segyio可能也不直接支持IBM格式，需要手动转换。

#### 方案C：手动写入（最可靠，完全控制）

```python
def write_segy_manual(output_file: str, traces_data: List[np.ndarray],
                     trace_headers: List[Dict], binary_header: Dict):
    """手动写入SEGY文件（完全控制格式）"""
    with open(output_file, 'wb') as f:
        # 1. 写入EBCDIC头（3200字节）
        write_ebcdic_header(f)
        
        # 2. 写入二进制头占位符（400字节）
        binary_header_pos = f.tell()
        f.write(b'\x00' * 400)
        
        # 3. 写入道数据
        for i, (data, header) in enumerate(zip(traces_data, trace_headers)):
            # 转换数据格式
            ibm_data = ieee_to_ibm_float(data)
            
            # 写入道头（240字节，big-endian）
            write_trace_header(f, header)
            
            # 写入数据（big-endian IBM浮点）
            for value in ibm_data:
                f.write(struct.pack('>I', value))  # big-endian uint32
        
        # 4. 回到二进制头位置，写入实际值
        f.seek(binary_header_pos)
        write_binary_header(f, binary_header, len(traces_data))
```

**推荐**：使用方案C（手动写入），因为：
- 完全控制格式
- 确保IBM浮点和字节序正确
- 与C代码输出一致

## 五、关键实现细节

### 5.1 时间处理

复用`fntime_v1_1.py`中的函数：

```python
# 从fntime_v1_1.py导入
from fntime_v1_1 import julian_day

# 计算漂移率
def calculate_drift_rate(drift_file: str, sac_year: int) -> float:
    """计算OBS漂移率"""
    # 读取drift.in文件
    # 计算时间间隔
    # 计算漂移率：drift / timegap (ms/s)
```

### 5.2 字节序处理

SEGY标准要求big-endian，Python代码需要：

```python
import struct

# 写入2字节（big-endian）
def write_int16_be(f, value: int):
    f.write(struct.pack('>h', value))  # '>'表示big-endian

# 写入4字节（big-endian）
def write_int32_be(f, value: int):
    f.write(struct.pack('>i', value))

# 或者使用numpy
def swap_bytes_vectorized(data: np.ndarray, dtype):
    """向量化字节序转换"""
    if dtype == np.int16:
        return data.astype('>i2')  # big-endian int16
    elif dtype == np.int32:
        return data.astype('>i4')  # big-endian int32
```

### 5.3 IBM浮点转换（向量化）

```python
def ieee_to_ibm_float_vectorized(ieee_data: np.ndarray) -> np.ndarray:
    """向量化IEEE→IBM浮点转换"""
    # 将IEEE float32转换为uint32进行位操作
    ieee_uint32 = ieee_data.view(np.uint32)
    
    # 提取各个部分
    sign = ieee_uint32 & 0x80000000
    exponent = ((ieee_uint32 & 0x7F800000) >> 23).astype(np.int32) - 127
    mantissa = (ieee_uint32 & 0x007FFFFF) | 0x00800000
    
    # 转换指数
    ibm_exponent = (exponent + 64) >> 2
    shift = (exponent + 64) & 0x3
    
    # 调整尾数
    mantissa_shifted = mantissa >> shift[:, np.newaxis]  # 需要broadcasting
    
    # 组合IBM浮点数
    ibm_float = sign | ((ibm_exponent.astype(np.uint32) << 24)) | (mantissa_shifted & 0x00FFFFFF)
    
    # 转换为big-endian
    ibm_float_be = ibm_float.byteswap()  # 或使用struct.pack
    
    return ibm_float_be
```

### 5.4 数据截取的性能优化

```python
def extract_traces_batch(sac_data: np.ndarray, shot_positions: List[int],
                        n_samples: int) -> np.ndarray:
    """批量截取多个道的数据（向量化）"""
    # 预先分配输出数组
    n_traces = len(shot_positions)
    output = np.zeros((n_traces, n_samples), dtype=np.float32)
    
    # 批量提取
    for i, pos in enumerate(shot_positions):
        if pos >= 0 and pos + n_samples <= len(sac_data):
            output[i] = sac_data[pos:pos+n_samples]
        else:
            output[i] = np.nan
    
    return output
```

## 六、主函数流程

```python
def convert_sac_to_segy(sac_file: str, ukooa_file: str, output_file: str,
                        par_file: str = "par.in", drift_file: str = "drift.in",
                        loc_file: Optional[str] = None) -> None:
    """SAC转SEGY主函数
    
    Args:
        sac_file: SAC文件路径
        ukooa_file: UKOOA文件路径
        output_file: 输出SEGY文件路径
        par_file: 参数文件路径
        drift_file: 漂移文件路径
        loc_file: 位置文件路径（可选）
    """
    # 1. 读取参数文件
    params = read_parameter_file(par_file)
    
    # 2. 读取SAC文件
    sac_stream = read(sac_file)
    sac_trace = sac_stream[0]
    sac_data = sac_trace.data
    sac_stats = sac_trace.stats
    
    # 3. 检查SAC文件
    if sac_stats.sac.b != 0.0:
        raise ValueError("SAC file b field must be 0.0")
    
    # 4. 计算参考时间
    sac_ref_time = sac_stats.starttime
    
    # 5. 读取漂移文件并计算漂移率
    drift_rate = calculate_drift_rate(drift_file, sac_stats.sac.nzyear)
    
    # 6. 获取台站位置
    if sac_stats.sac.stla == -12345:
        if loc_file is None:
            raise ValueError("SAC header has no position, need loc.in file")
        lat1, lon1 = read_location_file(loc_file)
    else:
        lat1 = sac_stats.sac.stla
        lon1 = sac_stats.sac.stlo
    
    # 7. 设置UTM投影
    transformer = setup_tm_projection(params['lon0'])
    gx, gy = lonlat_to_utm(lon1, lat1, params['lon0'])
    
    # 8. 设置SEGY二进制头
    binary_header = create_segy_binary_header(
        delta=sac_stats.delta,
        trace_length=params['length'],
        n_traces=0  # 稍后更新
    )
    
    # 9. 解析UKOOA文件
    shots = parse_ukooa_file(ukooa_file)
    
    # 10. 处理每个炮点
    traces_data = []
    traces_headers = []
    
    prev_sample_pos = -params['length'] / sac_stats.delta  # 类似TT0_temp
    
    for shot in shots:
        # 计算炮点时间（含各种校正）
        shot_time = calculate_shot_time(
            shot, sac_ref_time, drift_rate,
            params['rv'], params['ex'], params['ey'],
            params['et'], params['tcoor']
        )
        
        # 计算炮点位置
        sx, sy = lonlat_to_utm(shot['longitude'], shot['latitude'],
                               params['lon0'], params['ex'], params['ey'])
        
        # 计算偏移距
        offset = calculate_offset(gx, gy, sx, sy, lat1, shot['latitude'])
        
        # 计算数据截取位置
        sample_pos, n_samples = extract_trace_from_sac(
            sac_data, sac_ref_time, shot_time,
            params['length'], sac_stats.delta, params['tcoor']
        )
        
        if sample_pos is None:
            continue  # 跳过无效的炮点
        
        # 检查重叠
        gap = sample_pos - prev_sample_pos - n_samples
        if gap < 0:
            continue  # 与上一道重叠，跳过
        
        # 截取数据
        trace_data = sac_data[sample_pos:sample_pos+n_samples]
        
        # 创建SEGY道头
        trace_header = create_segy_trace_header(
            tracl=len(traces_data) + 1,
            tracr=len(traces_data) + 1,
            fldr=shot['field_record'],
            ep=shot['shot_number'],
            cdp=shot['shot_number'],
            offset=offset,
            sx=sx, sy=sy, gx=gx, gy=gy,
            year=sac_stats.sac.nzyear,
            day=shot['day'],
            hour=shot['hour'],
            minute=shot['minute'],
            sec=shot['sec'],
            timbas=shot['msec'],
            swdep=shot['water_depth'],
            gelev=sac_stats.sac.evel,
            delrt=params['tcoor'] * (-1000),
            ns=n_samples,
            dt=int(sac_stats.delta * 1e6 + 0.5)
        )
        
        traces_data.append(trace_data)
        traces_headers.append(trace_header)
        prev_sample_pos = sample_pos
    
    # 11. 写入SEGY文件
    write_segy_file(output_file, traces_data, traces_headers,
                   binary_header, len(traces_data))
```

## 七、与现有代码的集成

### 7.1 复用fntime_v1_1.py

```python
# 导入时间处理函数
from fntime_v1_1 import julian_day

# 可以在需要的地方使用
jday = julian_day(year, month, day)
```

### 7.2 复用su_processor.py

虽然`su_processor.py`主要用于读取SEGY文件，但可以参考：
- 道头字段映射
- 数据结构设计
- 错误处理模式

### 7.3 复用raw2sac_v1_1_obspy.py

- SAC文件读取模式
- 时间解析经验
- 数值转换经验

## 八、注意事项

### 8.1 字节序

- **SAC文件**：通常是little-endian（Intel格式）
- **SEGY文件**：标准要求big-endian（网络字节序）
- **Python默认**：取决于系统，需要使用struct明确指定

### 8.2 浮点格式

- **SAC文件**：IEEE 754浮点（float32）
- **SEGY标准**：IBM浮点格式（旧标准）或IEEE浮点（新标准）
- **C代码使用**：IBM浮点格式（format=1）

### 8.3 坐标投影

- **WGS-84椭球**：标准参数
- **UTM投影**：使用pyproj更可靠
- **TM投影**：如果需要自定义中心经线，可以使用pyproj创建自定义投影

### 8.4 时间处理

- **时区**：确保所有时间都是UTC
- **精度**：毫秒级别的精度处理
- **跨天处理**：正确处理跨天的情况

### 8.5 性能优化

详见"十三、性能优化策略"章节。

## 十三、性能优化策略（关键）

性能优化是Python实现接近C代码速度的关键。以下策略基于`raw2sac_v1_1_obspy.py`的成功经验。

### 13.1 核心优化原则

1. **避免Python循环**：所有循环都转换为NumPy向量化操作
2. **一次性内存分配**：预分配所有数组，避免动态扩展
3. **批量处理**：尽可能批量处理多个数据点/道
4. **最小化数据拷贝**：使用视图（view）而不是拷贝
5. **I/O优化**：一次性读取，批量写入
6. **可选的JIT编译**：对热点函数使用Numba（需测试效果）

### 13.2 数据读取优化

#### 13.2.1 SAC文件读取

```python
# 优化：一次性读取所有数据到内存
from obspy import read

sac_stream = read(sac_file, headonly=False)
sac_trace = sac_stream[0]
sac_data = sac_trace.data  # 已经是numpy数组，float32

# 避免重复读取头文件
sac_stats = sac_trace.stats
delta = sac_stats.delta  # 采样间隔
npts = sac_stats.npts    # 数据点数
```

**性能要点**：
- ObsPy读取SAC文件已经高度优化
- `sac_data`是连续的numpy数组，内存布局友好
- 避免重复读取文件

#### 13.2.2 UKOOA文件解析优化

```python
def parse_ukooa_file_vectorized(ukooa_file: str) -> np.ndarray:
    """向量化解析UKOOA文件（优化版本）
    
    使用NumPy向量化操作一次性解析所有行，避免Python循环。
    
    Returns:
        np.ndarray: 结构化数组，包含所有炮点信息
    """
    # 一次性读取所有行
    with open(ukooa_file, 'r') as f:
        lines = f.readlines()
    
    # 筛选有效行（以'S'或'N'开头）
    valid_lines = [line for line in lines if len(line) >= 75 and line[0] in 'SN']
    n_shots = len(valid_lines)
    
    # 预分配结构化数组
    dtype = np.dtype([
        ('day', 'i2'),
        ('hour', 'i2'),
        ('minute', 'i2'),
        ('sec', 'i2'),
        ('msec', 'i2'),
        ('shot_number', 'i4'),
        ('latitude', 'f8'),
        ('longitude', 'f8'),
        ('water_depth', 'i4'),
        ('field_record', 'i4'),
    ])
    shots = np.zeros(n_shots, dtype=dtype)
    
    # 向量化解析（如果需要，可以使用Numba加速）
    for i, line in enumerate(valid_lines):
        shots[i] = parse_ukooa_line_optimized(line)
    
    return shots

@jit(nopython=True, cache=True)
def parse_ukooa_line_optimized_numba(line_bytes: bytes) -> Tuple:
    """Numba加速的单行解析（如果行数很多）"""
    # 使用字节串操作，避免字符串操作开销
    # ...
```

**优化策略**：
- 一次性读取所有行到内存
- 预分配结构化数组
- 对于大量炮点，考虑使用Numba JIT编译解析函数

### 13.3 数据截取优化（关键瓶颈）

这是最关键的优化点，因为需要为每个炮点截取数据。

#### 13.3.1 批量计算采样位置

```python
def calculate_sample_positions_vectorized(
    shot_times: np.ndarray,           # 所有炮点时间
    sac_ref_time: UTCDateTime,
    trace_length: float,
    delta: float,
    tcoor: float
) -> Tuple[np.ndarray, np.ndarray]:
    """向量化计算所有炮点的采样位置
    
    Args:
        shot_times: 炮点时间数组（UTCDateTime数组或时间戳数组）
        sac_ref_time: SAC参考时间
        trace_length: 道长度（秒）
        delta: 采样间隔（秒）
        tcoor: 延迟校正（秒）
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (start_samples, n_samples_array)
        start_samples: 起始采样点位置数组
        n_samples_array: 每个道的采样点数数组
    """
    # 转换为相对时间（秒），使用numpy向量化
    if isinstance(shot_times[0], UTCDateTime):
        # 转换为时间戳数组
        shot_timestamps = np.array([t.timestamp for t in shot_times])
        ref_timestamp = sac_ref_time.timestamp
    else:
        shot_timestamps = shot_times
        ref_timestamp = sac_ref_time.timestamp if isinstance(sac_ref_time, UTCDateTime) else sac_ref_time
    
    # 向量化计算时间差
    time_diffs = shot_timestamps - ref_timestamp - tcoor
    
    # 向量化计算采样位置
    start_samples = np.round(time_diffs / delta).astype(np.int32)
    n_samples_array = np.full(len(shot_times), int(np.round(trace_length / delta)), dtype=np.int32)
    
    return start_samples, n_samples_array

def extract_traces_batch(sac_data: np.ndarray,
                        start_samples: np.ndarray,
                        n_samples: int) -> np.ndarray:
    """批量提取多个道的数据（高度优化）
    
    Args:
        sac_data: SAC数据数组（1D）
        start_samples: 起始采样点数组
        n_samples: 每个道的采样点数（固定）
    
    Returns:
        np.ndarray: shape (n_traces, n_samples) 的数据数组
    """
    n_traces = len(start_samples)
    
    # 预分配输出数组
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    
    # 过滤有效位置
    valid_mask = (start_samples >= 0) & (start_samples + n_samples <= len(sac_data))
    valid_starts = start_samples[valid_mask]
    valid_indices = np.where(valid_mask)[0]
    
    # 批量提取（使用高级索引）
    for i, start in enumerate(valid_starts):
        idx = valid_indices[i]
        traces[idx] = sac_data[start:start+n_samples]
    
    # 标记无效位置
    traces[~valid_mask] = np.nan
    
    return traces, valid_mask
```

**进一步优化（使用NumPy高级索引）**：

```python
@jit(nopython=True, cache=True, parallel=True)
def extract_traces_batch_numba(sac_data: np.ndarray,
                               start_samples: np.ndarray,
                               n_samples: int,
                               traces: np.ndarray):
    """Numba加速的批量提取（并行化）
    
    这个函数直接修改预分配的traces数组，避免返回值开销。
    """
    n_traces = len(start_samples)
    data_len = len(sac_data)
    
    for i in prange(n_traces):  # Numba并行循环
        start = start_samples[i]
        if start >= 0 and start + n_samples <= data_len:
            for j in range(n_samples):
                traces[i, j] = sac_data[start + j]
        else:
            for j in range(n_samples):
                traces[i, j] = np.nan
```

### 13.4 IBM浮点转换优化（关键）

这是另一个性能瓶颈，需要向量化实现。

#### 13.4.1 纯NumPy向量化版本（推荐）

```python
def ieee_to_ibm_float_vectorized(ieee_data: np.ndarray) -> np.ndarray:
    """向量化IEEE→IBM浮点转换（高度优化）
    
    性能要点：
    1. 完全向量化，无Python循环
    2. 使用NumPy位操作
    3. 最小化中间数组创建
    
    Args:
        ieee_data: IEEE浮点数组（float32）
    
    Returns:
        np.ndarray: IBM浮点数组（uint32，big-endian）
    """
    # 转换为uint32视图（不拷贝数据）
    ieee_uint32 = ieee_data.view(np.uint32)
    
    # 提取符号位（向量化）
    sign = ieee_uint32 & 0x80000000
    
    # 提取指数（向量化）
    exponent = ((ieee_uint32 & 0x7F800000) >> 23).astype(np.int32) - 127
    
    # 提取并处理尾数（向量化）
    mantissa = (ieee_uint32 & 0x007FFFFF) | 0x00800000
    
    # 计算IBM指数和尾数移位（向量化）
    ibm_exponent = ((exponent + 64) >> 2).astype(np.uint32)
    shift = (exponent + 64) & 0x3
    
    # 根据shift值调整尾数（需要条件操作）
    # 方法1：使用np.where（慢一点但向量化）
    mantissa_shifted = np.where(shift == 0, mantissa,
                   np.where(shift == 1, mantissa >> 1,
                   np.where(shift == 2, mantissa >> 2, mantissa >> 3)))
    
    # 方法2：分块处理（更快，但代码更复杂）
    # 可以分别处理shift=0,1,2,3的情况
    
    # 组合IBM浮点数（向量化）
    ibm_float = sign | (ibm_exponent << 24) | (mantissa_shifted & 0x00FFFFFF)
    
    # 转换为big-endian（向量化）
    ibm_float_be = ibm_float.byteswap()
    
    return ibm_float_be
```

#### 13.4.2 Numba加速版本（可选）

```python
@jit(nopython=True, cache=True, parallel=True)
def ieee_to_ibm_float_numba(ieee_data: np.ndarray, output: np.ndarray):
    """Numba JIT编译的IBM浮点转换（并行化）
    
    直接修改输出数组，避免返回值开销。
    """
    n = len(ieee_data)
    
    for i in prange(n):  # 并行循环
        # 将float32转换为uint32进行位操作
        ieee_bits = np.float32(ieee_data[i]).view(np.uint32)[0]
        
        if ieee_bits == 0:
            output[i] = 0
        else:
            # 提取符号、指数、尾数
            sign = ieee_bits & 0x80000000
            exponent = ((ieee_bits & 0x7F800000) >> 23) - 127
            mantissa = (ieee_bits & 0x007FFFFF) | 0x00800000
            
            # 转换指数
            ibm_exp = (exponent + 64) >> 2
            shift = (exponent + 64) & 0x3
            
            # 调整尾数
            if shift == 0:
                mantissa_shifted = mantissa
            elif shift == 1:
                mantissa_shifted = mantissa >> 1
            elif shift == 2:
                mantissa_shifted = mantissa >> 2
            else:
                mantissa_shifted = mantissa >> 3
            
            # 组合IBM浮点数
            ibm_float = sign | (ibm_exp << 24) | (mantissa_shifted & 0x00FFFFFF)
            
            # Big-endian转换
            output[i] = ((ibm_float << 24) & 0xFF000000) | \
                       ((ibm_float << 8) & 0x00FF0000) | \
                       ((ibm_float >> 8) & 0x0000FF00) | \
                       ((ibm_float >> 24) & 0x000000FF)
```

### 13.5 坐标投影优化

使用pyproj库，它已经高度优化（底层是C代码）。

```python
from pyproj import Transformer

# 一次性创建转换器（只创建一次，重复使用）
def setup_utm_transformer_optimized(lon0: float) -> Transformer:
    """创建优化的UTM转换器"""
    # 创建转换器（底层已经优化）
    # 对于大量坐标转换，使用批量转换API
    return Transformer.from_crs(
        "EPSG:4326",  # WGS84
        f"+proj=tmerc +lon_0={lon0} +datum=WGS84 +units=m +no_defs",
        always_xy=True
    )

def lonlat_to_utm_batch(lons: np.ndarray, lats: np.ndarray,
                       transformer: Transformer,
                       ex: float = 0.0, ey: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """批量转换坐标（向量化）
    
    使用pyproj的批量转换API，比循环调用快很多。
    """
    # 批量转换（向量化）
    xs, ys = transformer.transform(lons, lats)
    
    # 添加UTM偏移和用户偏移
    xs = xs + 500000.0 + ex
    
    # 南半球处理（向量化）
    south_mask = lats < 0
    ys[south_mask] = ys[south_mask] + 10000000.0
    
    ys = ys + ey
    
    return xs, ys
```

### 13.6 时间计算优化

```python
def calculate_shot_times_vectorized(
    shots: np.ndarray,
    sac_ref_time: UTCDateTime,
    drift_rate: float,
    rv: float,
    offsets: np.ndarray,
    et: float,
    tcoor: float
) -> np.ndarray:
    """向量化计算所有炮点时间
    
    所有计算都使用NumPy向量化，避免Python循环。
    """
    # 基础时间（向量化）
    base_times = np.array([
        UTCDateTime(year=sac_ref_time.year, julday=shot['day'],
                   hour=shot['hour'], minute=shot['minute'],
                   second=shot['sec'], microsecond=shot['msec']*1000).timestamp
        for shot in shots
    ])
    
    # 偏移时间（向量化）
    offset_times = np.where(rv > 0.001, offsets / (rv * 1000.0), 0.0)
    
    # 漂移校正（向量化）
    # 需要计算每个炮点相对于参考时间的漂移
    drift_corrections = calculate_drift_corrections_vectorized(shots, drift_rate)
    
    # 总时间校正（向量化）
    corrected_times = base_times + offset_times + drift_corrections + et + tcoor
    
    return corrected_times

def calculate_drift_corrections_vectorized(shots: np.ndarray,
                                          drift_rate: float) -> np.ndarray:
    """向量化计算漂移校正"""
    # 计算每个炮点相对于参考时间的漂移
    # 使用向量化操作
    # ...
```

### 13.7 SEGY文件写入优化

#### 13.7.1 批量写入策略

```python
def write_segy_optimized(output_file: str,
                        traces_data: List[np.ndarray],
                        traces_headers: List[Dict],
                        binary_header: Dict):
    """优化的SEGY写入（批量I/O）
    
    优化策略：
    1. 预转换所有数据格式
    2. 批量写入缓冲区
    3. 最小化系统调用
    """
    # 预分配缓冲区大小（例如4MB）
    BUFFER_SIZE = 4 * 1024 * 1024
    
    with open(output_file, 'wb', buffering=BUFFER_SIZE) as f:
        # 1. 写入EBCDIC头（一次性）
        write_ebcdic_header(f)
        
        # 2. 预留二进制头位置
        binary_header_pos = f.tell()
        f.write(b'\x00' * 400)
        
        # 3. 预转换所有数据（批量转换，避免循环中的转换开销）
        print("Converting data to IBM format...")
        ibm_traces = []
        for data in traces_data:
            ibm_data = ieee_to_ibm_float_vectorized(data)
            ibm_traces.append(ibm_data)
        
        # 4. 批量写入道头和数据
        print("Writing traces...")
        for i, (ibm_data, header) in enumerate(zip(ibm_traces, traces_headers)):
            # 写入道头（240字节）
            write_trace_header_optimized(f, header)
            
            # 写入数据（批量写入，使用tobytes()避免循环）
            f.write(ibm_data.tobytes('C'))  # C顺序，直接内存布局
        
        # 5. 更新二进制头
        f.seek(binary_header_pos)
        write_binary_header_optimized(f, binary_header, len(traces_data))
```

#### 13.7.2 道头写入优化

```python
import struct

def write_trace_header_optimized(f, header: Dict):
    """优化的道头写入（使用struct.pack一次性写入）"""
    # 使用struct一次性打包所有字段（比逐个写入快）
    # 创建240字节的缓冲区
    header_bytes = bytearray(240)
    
    # 批量写入字段（使用struct.pack）
    # 字段位置参考SEGY标准
    struct.pack_into('>i', header_bytes, 0, header['tracl'])   # 位置0-3
    struct.pack_into('>i', header_bytes, 4, header['tracr'])   # 位置4-7
    struct.pack_into('>i', header_bytes, 8, header['fldr'])    # 位置8-11
    # ... 继续写入所有字段
    
    # 一次性写入
    f.write(header_bytes)
```

或者使用NumPy结构化数组：

```python
def write_trace_header_numpy(f, header: Dict):
    """使用NumPy结构化数组写入道头"""
    # 定义SEGY道头的dtype（对应240字节）
    segy_header_dtype = np.dtype([
        ('tracl', '>i4'),   # big-endian int32
        ('tracr', '>i4'),
        ('fldr', '>i4'),
        # ... 定义所有字段
    ])
    
    # 创建结构化数组
    header_array = np.zeros(1, dtype=segy_header_dtype)
    header_array[0]['tracl'] = header['tracl']
    header_array[0]['tracr'] = header['tracr']
    # ... 填充所有字段
    
    # 一次性写入（240字节）
    f.write(header_array.tobytes())
```

### 13.8 内存优化

```python
def process_sac_to_segy_optimized(sac_file: str, ukooa_file: str,
                                  output_file: str, **params):
    """内存优化的主处理函数"""
    
    # 1. 读取SAC文件（一次性）
    sac_stream = read(sac_file)
    sac_data = sac_stream[0].data.copy()  # 拷贝，避免视图问题
    
    # 2. 解析UKOOA（一次性）
    shots = parse_ukooa_file_vectorized(ukooa_file)
    
    # 3. 预计算所有炮点位置和时间（向量化）
    shot_times = calculate_shot_times_vectorized(shots, ...)
    start_samples, n_samples = calculate_sample_positions_vectorized(shot_times, ...)
    
    # 4. 批量提取所有道（一次性）
    traces_data, valid_mask = extract_traces_batch(sac_data, start_samples, n_samples)
    
    # 5. 过滤有效道
    valid_traces = traces_data[valid_mask]
    valid_shots = shots[valid_mask]
    
    # 6. 批量计算坐标（向量化）
    lons = valid_shots['longitude']
    lats = valid_shots['latitude']
    transformer = setup_utm_transformer_optimized(params['lon0'])
    sxs, sys = lonlat_to_utm_batch(lons, lats, transformer, ...)
    
    # 7. 批量创建道头（向量化创建）
    trace_headers = create_trace_headers_batch(valid_shots, sxs, sys, ...)
    
    # 8. 批量转换数据格式（向量化）
    print("Converting data formats...")
    ibm_traces = []
    for trace in valid_traces:
        ibm_trace = ieee_to_ibm_float_vectorized(trace)
        ibm_traces.append(ibm_trace)
    
    # 9. 批量写入（优化I/O）
    write_segy_optimized(output_file, ibm_traces, trace_headers, ...)
```

### 13.9 性能测试和基准

```python
import time

def benchmark_conversion():
    """性能基准测试"""
    start = time.time()
    
    # 测试各个步骤的性能
    t1 = time.time()
    sac_data = read_sac_file(...)
    print(f"SAC读取: {time.time() - t1:.3f}s")
    
    t1 = time.time()
    shots = parse_ukooa_file(...)
    print(f"UKOOA解析: {time.time() - t1:.3f}s")
    
    t1 = time.time()
    traces = extract_traces_batch(...)
    print(f"数据截取: {time.time() - t1:.3f}s")
    
    t1 = time.time()
    ibm_traces = [ieee_to_ibm_float_vectorized(t) for t in traces]
    print(f"IBM转换: {time.time() - t1:.3f}s")
    
    t1 = time.time()
    write_segy_file(...)
    print(f"SEGY写入: {time.time() - t1:.3f}s")
    
    print(f"总时间: {time.time() - start:.3f}s")
```

### 13.10 性能优化检查清单

- [ ] 所有循环都向量化了吗？
- [ ] 数组都预分配了吗？
- [ ] 数据转换是否批量进行？
- [ ] I/O是否使用缓冲区？
- [ ] 是否避免了不必要的拷贝？
- [ ] 热点函数是否使用Numba（可选）？
- [ ] 坐标转换是否批量进行？
- [ ] 道头创建是否批量进行？

### 13.11 预期性能目标

基于`raw2sac_v1_1_obspy.py`的经验：
- **纯NumPy向量化版本**：应该能达到C代码的2-5倍时间（可接受）
- **使用Numba加速**：可能接近C代码速度（1-2倍时间）
- **关键瓶颈**：
  1. 数据截取（批量处理最关键）
  2. IBM浮点转换（向量化最关键）
  3. SEGY写入（批量I/O最关键）

### 13.12 实际优化建议

1. **优先实现纯NumPy版本**：先保证正确性，再优化性能
2. **性能分析**：使用`cProfile`找出瓶颈
   ```python
   import cProfile
   cProfile.run('convert_sac_to_segy(...)', 'profile.stats')
   ```
3. **逐步优化**：每次优化一个模块，测试效果
4. **Numba谨慎使用**：先测试纯NumPy版本，再考虑Numba
5. **内存vs速度权衡**：批量处理可能增加内存使用，需要权衡

### 13.13 性能优化示例：数据截取对比

#### 慢速版本（Python循环）
```python
def extract_traces_slow(sac_data, start_samples, n_samples):
    traces = []
    for start in start_samples:
        traces.append(sac_data[start:start+n_samples])
    return np.array(traces)
# 时间复杂度：O(n_traces * n_samples)，Python循环开销大
```

#### 快速版本（向量化）
```python
def extract_traces_fast(sac_data, start_samples, n_samples):
    n_traces = len(start_samples)
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    for i, start in enumerate(start_samples):
        if start >= 0 and start + n_samples <= len(sac_data):
            traces[i] = sac_data[start:start+n_samples]
    return traces
# 时间复杂度：O(n_traces * n_samples)，但NumPy循环比Python快10-100倍
```

#### 最快版本（批量+预过滤）
```python
def extract_traces_optimized(sac_data, start_samples, n_samples):
    # 预过滤有效位置（向量化）
    valid_mask = (start_samples >= 0) & (start_samples + n_samples <= len(sac_data))
    valid_starts = start_samples[valid_mask]
    
    n_traces = len(start_samples)
    traces = np.zeros((n_traces, n_samples), dtype=np.float32)
    
    # 批量提取有效位置（NumPy循环，比Python快很多）
    valid_indices = np.where(valid_mask)[0]
    for i, start in zip(valid_indices, valid_starts):
        traces[i] = sac_data[start:start+n_samples]
    
    traces[~valid_mask] = np.nan
    return traces, valid_mask
# 进一步优化：减少无效位置的拷贝操作
```

### 13.14 性能监控代码

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(description: str):
    """性能计时上下文管理器"""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{description}: {elapsed:.3f}s")

# 使用示例
with timer("数据截取"):
    traces = extract_traces_batch(sac_data, start_samples, n_samples)

with timer("IBM浮点转换"):
    ibm_traces = [ieee_to_ibm_float_vectorized(t) for t in traces]
```

### 13.15 关键性能指标（KPI）

| 操作 | 目标时间 | 瓶颈 |
|------|---------|------|
| SAC文件读取 | < 0.1s | I/O速度 |
| UKOOA解析 | < 0.1s | 文件大小 |
| 数据截取（1000道） | < 1s | **关键瓶颈** |
| IBM浮点转换（1000道） | < 2s | **关键瓶颈** |
| 坐标投影（1000点） | < 0.1s | pyproj已优化 |
| SEGY写入（1000道） | < 1s | I/O速度 |

**总目标**：对于1000道的处理，总时间应该 < 5秒（C代码约为1-2秒）

## 九、实现优先级

### Phase 1: 核心功能（必须）- 基础实现
1. SAC文件读取
2. UKOOA文件解析
3. 基本数据截取（循环版本，保证正确性）
4. SEGY文件写入（简化版）

### Phase 2: 时间校正（重要）
1. 偏移时间计算
2. 漂移校正
3. 延迟校正

### Phase 3: 坐标投影（重要）
1. UTM投影实现
2. 偏移距计算

### Phase 4: 格式转换（必须）
1. IBM浮点转换（基础版本）
2. 字节序转换

### Phase 5: 性能优化（关键）- 向量化实现
1. **数据截取向量化**（最关键）
   - 批量计算采样位置
   - 批量提取数据
   - 使用NumPy高级索引

2. **IBM浮点转换向量化**（关键瓶颈）
   - 纯NumPy向量化版本
   - 可选Numba加速版本

3. **坐标投影批量处理**
   - 使用pyproj批量API
   - 向量化偏移计算

4. **时间计算向量化**
   - 批量计算所有炮点时间
   - 向量化校正计算

5. **SEGY写入优化**
   - 批量数据转换
   - 批量I/O写入
   - 缓冲区优化

### Phase 6: 高级优化（可选）
1. Numba JIT编译加速
2. 并行处理（多线程/多进程）
3. 内存映射大文件
4. 性能分析和优化

### Phase 7: 完善和测试
1. 错误处理
2. 日志输出
3. 参数验证
4. 单元测试和集成测试

## 十、测试策略

1. **单元测试**：
   - UKOOA解析
   - 时间计算
   - 坐标转换
   - IBM浮点转换

2. **集成测试**：
   - 使用真实SAC和UKOOA文件
   - 对比C代码输出

3. **格式验证**：
   - 使用Seismic Unix读取输出文件
   - 验证道头字段
   - 验证数据正确性

## 十一、依赖库

```python
# requirements.txt
obspy>=1.4.0
numpy>=1.20.0
pyproj>=3.0.0
# 性能优化库（推荐）
numba>=0.56.0  # JIT编译加速（可选但推荐）
# 可选
segyio>=1.9.0  # 如果使用segyio进行底层控制
```

### 11.1 性能优化库说明

- **numba**: 用于JIT编译热点函数，可以显著加速数值计算密集的部分
- **numpy**: 使用最新版本（>=1.20.0）以获得更好的性能
- **pyproj**: 坐标转换库，底层是C代码，已经高度优化

## 十二、代码结构建议

```
processors/raw2sac/
├── sac2y_v2_1_obspy.py      # 主程序
├── sac2y_utils.py            # 工具函数（投影、转换等）
├── sac2y_ukooa.py            # UKOOA解析
└── sac2y_segy.py             # SEGY写入模块
```

或者所有功能在一个文件中（更简单）：
```
processors/raw2sac/
└── sac2y_v2_1_obspy.py      # 完整实现
```

---

## 附录：性能优化要点快速参考

### 最关键的3个优化点

1. **数据截取向量化**（最关键）
   - 批量计算所有炮点的采样位置
   - 预分配输出数组
   - 批量提取数据，避免Python循环

2. **IBM浮点转换向量化**（关键瓶颈）
   - 完全向量化，零Python循环
   - 使用NumPy位操作和视图
   - 批量转换所有道的数据

3. **批量处理策略**
   - 所有操作尽可能批量进行
   - 预计算所有炮点信息
   - 批量I/O写入

### 性能优化口诀

- **"向量化一切"**：Python循环 → NumPy向量化
- **"预分配所有"**：预分配数组，避免动态扩展
- **"批量处理"**：一次处理多个数据点
- **"最小化拷贝"**：使用视图和原地操作
- **"缓冲区I/O"**：使用大缓冲区读写文件

### 性能提升预期

| 优化阶段 | 预期时间（1000道） | 提升倍数 |
|---------|------------------|---------|
| 未优化（Python循环） | ~20-50秒 | 基准 |
| NumPy向量化 | ~5-10秒 | 2-5x |
| 批量优化 | ~2-5秒 | 4-10x |
| Numba加速（可选） | ~1-2秒 | 10-20x |

**目标**：向量化版本应达到C代码的2-5倍时间，这是可接受的性能损失。

---

**重要提示**：性能优化是迭代过程。先实现正确的功能，再使用profiler找出瓶颈，逐个优化。不要过早优化！

## 十三、性能优化要点总结（快速参考）

### 关键优化策略（按优先级排序）

#### 1. **数据截取优化**（最关键瓶颈）
- ✅ 使用NumPy批量提取，避免Python循环
- ✅ 预分配输出数组
- ✅ 向量化计算采样位置
- ✅ 预过滤有效位置，减少无效拷贝

#### 2. **IBM浮点转换优化**（关键瓶颈）
- ✅ 完全向量化，零Python循环
- ✅ 使用NumPy位操作和视图（view）
- ✅ 可选Numba JIT编译加速

#### 3. **批量处理策略**
- ✅ 所有炮点信息一次性解析
- ✅ 所有采样位置批量计算
- ✅ 所有坐标批量转换（pyproj批量API）
- ✅ 所有数据批量转换格式

#### 4. **I/O优化**
- ✅ 一次性读取SAC文件
- ✅ 使用大缓冲区写入SEGY文件
- ✅ 批量写入，最小化系统调用

#### 5. **内存优化**
- ✅ 预分配所有数组
- ✅ 使用视图（view）而非拷贝
- ✅ 及时释放不需要的数据

### 性能优化检查清单

在实现代码时，请确保：

- [ ] **向量化**：所有循环都使用NumPy向量化
- [ ] **预分配**：所有数组都预分配大小
- [ ] **批量处理**：相关操作批量进行
- [ ] **避免拷贝**：使用视图和原地操作
- [ ] **I/O优化**：使用缓冲区，批量读写
- [ ] **热点识别**：使用profiler找出瓶颈
- [ ] **逐步优化**：先正确后优化，一次优化一个模块

### 预期性能对比

| 操作 | 未优化（Python循环） | 优化后（向量化） | C代码 |
|------|---------------------|----------------|-------|
| 数据截取（1000道） | ~10-30s | ~0.5-1s | ~0.2s |
| IBM转换（1000道） | ~5-15s | ~1-2s | ~0.3s |
| 坐标投影（1000点） | ~1-2s | ~0.05s | ~0.02s |
| **总时间（1000道）** | **~20-50s** | **~2-5s** | **~1-2s** |

**目标**：优化后的Python版本应该达到C代码的2-5倍时间，这是可接受的性能损失（相对于开发效率和维护性）。

### 快速性能提升技巧

1. **识别热点**：使用`cProfile`找出耗时最多的函数
2. **向量化优先**：将Python循环改为NumPy向量化操作
3. **批量操作**：一次处理多个数据点而不是循环
4. **预分配数组**：避免动态扩展数组的开销
5. **使用合适的数据类型**：float32而不是float64（如果精度足够）
6. **避免不必要的转换**：直接使用NumPy数组，避免list转换

### 性能优化示例代码片段

```python
# ❌ 慢速版本（Python循环）
traces = []
for start in start_samples:
    traces.append(sac_data[start:start+n_samples])

# ✅ 快速版本（向量化+预分配）
traces = np.zeros((len(start_samples), n_samples), dtype=np.float32)
for i, start in enumerate(start_samples):
    traces[i] = sac_data[start:start+n_samples]

# ✅ 最快版本（批量+过滤）
valid_mask = (start_samples >= 0) & (start_samples + n_samples <= len(sac_data))
traces = np.zeros((len(start_samples), n_samples), dtype=np.float32)
valid_indices = np.where(valid_mask)[0]
for i, start in zip(valid_indices, start_samples[valid_mask]):
    traces[i] = sac_data[start:start+n_samples]
```

---

**记住**：性能优化是一个迭代过程。先实现正确的功能，然后使用profiler找出瓶颈，逐个优化。不要过早优化！
