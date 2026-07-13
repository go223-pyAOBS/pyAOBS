"""
data_loader.py - ZPLOT 数据加载模块

负责读取 Z 格式数据文件、头文件和记录文件

改进说明（Z 格式文件结构优化）：
- 阶段1（已完成）：优先从头文件读取拾取信息，保持向后兼容
  - 如果提供了头文件，从头文件读取拾取信息（优先级高）
  - 如果没有头文件，从数据文件读取拾取信息（向后兼容）
  - TraceHeader 中添加了 picks_from_header 字段标记拾取信息来源
  
- 阶段2（待实现）：写入格式改进
  - 如果将来实现保存数据文件功能，应确保：
    1. 保存数据文件时，不写入拾取信息（或写入 0 值占位）
    2. 拾取信息只保存到头文件（`.hdr`）
    3. 参考 pick_manager.py 中的 save_to_header_file 方法
"""

import struct
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import os
import logging
from collections import OrderedDict


logger = logging.getLogger(__name__)


class LazyTraceCollection:
    """按需读取道数据的轻量容器。

    特性：
    - 仅在 `__getitem__` 被访问时从磁盘读取指定道
    - 内置 LRU 缓存，避免重复读取热点道
    - 对外保持类列表接口：支持 `len()` 与索引访问
    """

    def __init__(
        self,
        file_path: str,
        ntraces: int,
        npts: int,
        trace_dtype: np.dtype,
        data_offset: int,
        record_size: int,
        trace_header_size: int,
        cache_size: int = 2048,
    ):
        self.file_path = file_path
        self.ntraces = int(ntraces)
        self.npts = int(npts)
        self.trace_dtype = trace_dtype
        self.data_offset = int(data_offset)
        self.record_size = int(record_size)
        self.trace_header_size = int(trace_header_size)
        self.cache_size = int(max(16, cache_size))
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

    def __len__(self) -> int:
        return self.ntraces

    def _resolve_index(self, idx: int) -> int:
        if idx < 0:
            idx = self.ntraces + idx
        if idx < 0 or idx >= self.ntraces:
            raise IndexError(f"trace index out of range: {idx}")
        return idx

    def _read_one_trace(self, idx: int) -> np.ndarray:
        byte_offset = self.data_offset + idx * self.record_size + self.trace_header_size
        with open(self.file_path, "rb") as f:
            f.seek(byte_offset)
            trace = np.fromfile(f, dtype=self.trace_dtype, count=self.npts)
        if trace.shape[0] < self.npts:
            raise ValueError(f"道数据不完整：索引 {idx}, 期望 {self.npts} 点，读取 {trace.shape[0]} 点")
        return trace

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.ntraces)
            return [self[i] for i in range(start, stop, step)]
        if isinstance(idx, (list, tuple, np.ndarray)):
            return [self[int(i)] for i in idx]

        ridx = self._resolve_index(int(idx))
        cached = self._cache.get(ridx)
        if cached is not None:
            # LRU: 访问后移动到末尾
            self._cache.move_to_end(ridx)
            return cached

        trace = self._read_one_trace(ridx)
        self._cache[ridx] = trace
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return trace

    def clear_cache(self):
        self._cache.clear()


@dataclass
class ZFormatHeader:
    """Z 格式文件头"""
    ntraces: int      # 总道数
    npts: int         # 每道采样点数
    sint: int         # 采样间隔（微秒）
    tstart: int       # 起始时间（毫秒）
    tend: int         # 结束时间（毫秒）
    nrec: int         # 记录数（炮集数）
    npick: int        # 拾取字数
    vredf: float      # 折合速度（km/s）
    ifmt: int         # 数据格式（1=float32, 0=int16）
    xlatlong: float   # 经纬度相关
    xelev: float      # 高程
    xutm: float       # UTM坐标
    cm: float         # 其他参数
    
    @property
    def sampling_interval(self) -> float:
        """采样间隔（秒）"""
        return self.sint / 1_000_000.0
    
    @property
    def start_time(self) -> float:
        """起始时间（秒）"""
        return self.tstart / 1_000.0
    
    @property
    def end_time(self) -> float:
        """结束时间（秒）
        
        如果文件头中的tend字段无效（为0），则根据采样点数和采样间隔自动计算
        """
        if self.tend > 0:
            # 使用文件头中的tend值
            return self.tend / 1_000.0
        else:
            # 如果tend为0或无效，根据采样点数和采样间隔计算
            # end_time = start_time + (npts - 1) * sampling_interval
            return self.start_time + (self.npts - 1) * self.sampling_interval
    
    @property
    def times(self) -> np.ndarray:
        """时间数组（秒）"""
        return np.arange(self.npts) * self.sampling_interval + self.start_time


@dataclass
class TraceHeader:
    """道头信息"""
    ishoti: int           # 炮站号
    itsn: int             # 道序号
    ireci: int            # 接收站号
    itypei: int           # 数据类型（1=垂直，2=径向，3=横向，4=水听器）
    iflagi: int           # 数据标志（1=有效，其他=死道）
    igaini: int           # 增益因子
    offsti: float         # 炮检距（千米）
    azi: float            # 方位角（度）
    texact: float         # 精确时间
    slat: float           # 炮点纬度
    slong: float          # 炮点经度
    selev: float          # 炮点高程（米）
    swdepth: float        # 炮点水深
    rlat: float           # 接收点纬度
    rlong: float          # 接收点经度
    relev: float          # 接收点高程（米）
    sxutm: float          # 炮点UTM X坐标
    syutm: float          # 炮点UTM Y坐标
    sz: float             # 炮点深度
    rxutm: float          # 接收点UTM X坐标
    ryutm: float          # 接收点UTM Y坐标
    rz: float             # 接收点深度
    picks: List[float]    # 拾取时间数组（npick个值）
    picks_from_header: bool = True  # 拾取信息是否来自头文件（True=头文件，False=数据文件）


@dataclass
class RecordInfo:
    """记录（炮集）信息"""
    ishnum: int      # 记录号
    xmod: float      # X模型位置
    ymod: float      # Y模型位置
    az: float        # 方位角
    title: str       # 标题


class DataLoader:
    """数据加载类 - 读取 Z 格式数据"""
    
    def __init__(self):
        self.header: Optional[ZFormatHeader] = None
        self.trace_headers: List[TraceHeader] = []
        self.traces: Any = []
        self.trace_matrix: Optional[np.ndarray] = None
        self.records: List[RecordInfo] = []
        self._last_layout_name: str = "unknown"
        self._last_trace_data_offset: int = 52
        self._header_parse_mode_counts: Dict[str, int] = {"float": 0, "mixed": 0}
        self._lazy_trace_loading: bool = os.environ.get("ZPLOTPY_LAZY_TRACES", "1").strip() not in ("0", "false", "False")
        # 默认静默加载诊断输出；如需打开可设置 ZPLOTPY_VERBOSE_LOAD=1
        self._verbose_load_logs: bool = os.environ.get("ZPLOTPY_VERBOSE_LOAD", "0").strip() in ("1", "true", "True")
        
    def load_z_format(self, dfile: str, hfile: Optional[str] = None, 
                     rfile: Optional[str] = None) -> Dict[str, Any]:
        """加载 Z 格式数据
        
        改进：优先从头文件读取拾取信息，保持向后兼容
        - 如果提供了头文件，从头文件读取拾取信息（优先级高）
        - 如果没有头文件，从数据文件读取拾取信息（向后兼容）
        
        Args:
            dfile: 数据文件路径
            hfile: 头文件路径（可选）
            rfile: 记录文件路径（可选）
            
        Returns:
            包含以下键的字典：
            - 'header': ZFormatHeader 对象
            - 'traces': 道数据列表（numpy数组）
            - 'trace_headers': 道头信息列表
            - 'records': 记录信息列表
            - 'offsets': 炮检距数组
            - 'times': 时间数组
        """
        # 读取文件头
        self.header = self._read_file_header(dfile)
        
        # 读取道数据
        self.traces = self._read_traces(dfile, self.header)
        
        # 优先从头文件读取道头信息（包括拾取信息）
        if hfile and os.path.exists(hfile):
            # 从头文件读取，拾取信息标记为来自头文件
            self.trace_headers = self._read_header_file(hfile, self.header)
        else:
            # 如果没有头文件，从.z文件读取道头信息（向后兼容）
            # 拾取信息标记为来自数据文件
            try:
                self.trace_headers = self._read_trace_headers_from_zfile(dfile, self.header)
                # 验证读取的道头数量是否正确
                if len(self.trace_headers) != len(self.traces):
                    # 如果数量不匹配，使用默认道头
                    self.trace_headers = self._create_default_headers(self.header)
            except Exception as e:
                # 如果读取失败，使用默认道头
                import traceback
                traceback.print_exc()
                self.trace_headers = self._create_default_headers(self.header)
        
        # 读取记录文件（如果提供）
        if rfile and os.path.exists(rfile):
            self.records = self._read_record_file(rfile)
        
        # 提取炮检距数组
        # 策略：确保offsets数组的长度等于traces的长度，并且范围覆盖所有道
        # 如果头文件中的道数少于数据文件，使用头文件中的偏移，其余使用默认值（索引+1）
        # 如果头文件中的道数等于数据文件，但offsets范围不合理，也使用默认值
        
        offsets_list = []
        
        # 首先，从头文件中提取offsets值（如果有）
        for i in range(len(self.trace_headers)):
            if i < len(self.traces):
                offsets_list.append(self.trace_headers[i].offsti)
        
        # 对于没有道头的道，使用索引+1作为默认偏移
        # 这样确保offsets范围是1-40（或1-ntraces）
        for i in range(len(self.trace_headers), len(self.traces)):
            offsets_list.append(float(i + 1))
        
        offsets = np.array(offsets_list)
        
        # 检查offsets是否有效
        # 如果头文件中的offsets都是0或接近0，且UTM坐标也无法计算，才使用默认值
        if len(self.trace_headers) > 0 and len(self.trace_headers) == len(self.traces):
            # 头文件道数等于数据文件道数
            header_offsets = np.array([th.offsti for th in self.trace_headers])
            max_offset = np.max(header_offsets)
            
            # 只有当所有offsets都接近0时，才使用默认值
            # 如果至少有一个非零的offset值，就使用头文件中的值
            if max_offset < 0.001:  # 所有offsets都接近0
                offsets = np.array([float(i + 1) for i in range(len(self.traces))])

        self._log_load_summary(dfile=dfile, hfile=hfile)
        self._log_trace_amplitude_samples()
        
        return {
            'header': self.header,
            'traces': self.traces,
            'trace_matrix': self.trace_matrix,
            'trace_headers': self.trace_headers,
            'records': self.records,
            'offsets': offsets,
            'times': self.header.times
        }
    
    def _read_file_header(self, dfile: str) -> ZFormatHeader:
        """读取文件头（记录1，52字节）"""
        with open(dfile, 'rb') as f:
            # 读取52字节文件头
            header_bytes = f.read(52)
            
            if len(header_bytes) < 52:
                raise ValueError(f"文件头不完整：只读取了{len(header_bytes)}字节")
            
            # 解析文件头
            # 前7个整数
            ints = struct.unpack('<7i', header_bytes[:28])
            ntraces, npts, sint, tstart, tend, nrec, npick = ints
            
            # 第8个是浮点数（vredf）
            vredf, = struct.unpack('<f', header_bytes[28:32])
            
            # 第9个是整数（ifmt）
            ifmt, = struct.unpack('<i', header_bytes[32:36])
            
            # 最后4个浮点数
            floats = struct.unpack('<4f', header_bytes[36:52])
            xlatlong, xelev, xutm, cm = floats
            
            return ZFormatHeader(
                ntraces=ntraces,
                npts=npts,
                sint=sint,
                tstart=tstart,
                tend=tend,
                nrec=nrec,
                npick=npick,
                vredf=vredf,
                ifmt=ifmt,
                xlatlong=xlatlong,
                xelev=xelev,
                xutm=xutm,
                cm=cm
            )

    def _compute_trace_record_size(self, header: ZFormatHeader) -> int:
        """计算单条道记录字节数（道头 + 道数据）。"""
        trace_header_size = (22 + header.npick) * 4
        if header.ifmt == 1:
            trace_data_size = header.npts * 4
        else:
            trace_data_size = header.npts * 2
        return trace_header_size + trace_data_size

    def _infer_trace_data_offset(self, dfile: str, header: ZFormatHeader, record_size: int) -> int:
        """根据文件大小推断首道数据偏移，兼容两种历史 .z 布局。

        Fortran 直接访问文件在不同编译/运行环境下常见两种布局：
        1) 紧凑布局: [52字节文件头][ntraces条道记录]，首道偏移 52
        2) 固定记录布局: [1条完整record(前52字节是文件头+其余填充)][ntraces条道记录]
           即首道偏移 record_size
        """
        file_size = os.path.getsize(dfile)
        expected_compact = 52 + header.ntraces * record_size
        expected_fixed = (header.ntraces + 1) * record_size

        if file_size == expected_compact:
            self._last_layout_name = "compact"
            self._last_trace_data_offset = 52
            return 52
        if file_size == expected_fixed:
            self._last_layout_name = "fixed_record"
            self._last_trace_data_offset = record_size
            return record_size

        # 容忍少量尾部差异（例如附加字节），按更接近的一种布局处理
        if abs(file_size - expected_fixed) < abs(file_size - expected_compact):
            self._last_layout_name = "fixed_record_approx"
            self._last_trace_data_offset = record_size
            return record_size
        self._last_layout_name = "compact_approx"
        self._last_trace_data_offset = 52
        return 52
    
    def _read_traces(self, dfile: str, header: ZFormatHeader) -> List[np.ndarray]:
        """读取所有道数据"""
        # 计算记录大小
        if header.ifmt == 1:  # float32
            trace_data_size = header.npts * 4
            trace_dtype = np.float32
        else:  # int16
            trace_data_size = header.npts * 2
            trace_dtype = np.int16

        trace_header_size = (22 + header.npick) * 4
        record_size = trace_header_size + trace_data_size
        data_offset = self._infer_trace_data_offset(dfile, header, record_size)

        # 进阶优化：大文件优先启用按需读取，避免一次性加载全部道数据
        file_size = os.path.getsize(dfile)
        should_use_lazy = self._lazy_trace_loading and (
            header.ntraces >= 1500 or file_size >= 64 * 1024 * 1024
        )
        if should_use_lazy:
            self.trace_matrix = None
            return LazyTraceCollection(
                file_path=dfile,
                ntraces=header.ntraces,
                npts=header.npts,
                trace_dtype=trace_dtype,
                data_offset=data_offset,
                record_size=record_size,
                trace_header_size=trace_header_size,
                cache_size=2048,
            )

        # 性能优化：使用结构化 dtype 一次性批量读取，避免逐道 Python 循环
        # 记录结构 = [道头字节块 | 道数据数组]
        record_dtype = np.dtype(
            [
                ("trace_header", f"V{trace_header_size}"),
                ("trace_data", trace_dtype, (header.npts,)),
            ]
        )
        records = np.fromfile(
            dfile,
            dtype=record_dtype,
            count=header.ntraces,
            offset=data_offset,
        )

        if records.shape[0] < header.ntraces:
            raise ValueError(f"道数据不完整：期望 {header.ntraces} 道，实际读取 {records.shape[0]} 道")

        trace_matrix = records["trace_data"]
        self.trace_matrix = trace_matrix
        # 保持对外接口兼容：返回 List[np.ndarray]
        # 关键优化：返回 view 而非逐道 copy，减少大文件加载时的大量内存拷贝。
        return [trace_matrix[i] for i in range(trace_matrix.shape[0])]
    
    def _read_header_file(self, hfile: str, header: ZFormatHeader) -> List[TraceHeader]:
        """读取头文件（Fortran 未格式化格式）
        
        注意：头文件按照C结构体的内存布局写入，字段顺序是混合的：
        - 5个int: nrec, itsn, ireci, itype, iflag
        - 2个float: offset, azi
        - 1个int: igain
        - 1个float: texact
        - 2个float: slat, slong
        - 2个int: selev, swdepth
        - 2个float: rlat, rlong
        - 1个int: relev
        - 6个float: sxutm, syutm, sz, rxutm, ryutm, rz
        总共：9个int + 13个float = 22个字段，88字节
        """
        trace_headers = []
        
        with open(hfile, 'rb') as f:
            # 尝试读取所有记录，不限制为 header.ntraces
            # 因为头文件可能包含不同数量的记录
            record_count = 0
            while True:
                # 读取记录长度标记（4字节）
                rec_len_bytes = f.read(4)
                if len(rec_len_bytes) < 4:
                    break
                
                rec_len = struct.unpack('<i', rec_len_bytes)[0]
                
                # 读取88字节的道头数据（按照C结构体的内存布局）
                header_bytes = f.read(88)
                if len(header_bytes) < 88:
                    break
                
                # 按照C结构体布局解析（22个字段，每个4字节）
                # 字段顺序：nrec(int), itsn(int), ireci(int), itype(int), iflag(int),
                #           offset(float), azi(float), igain(int), texact(float),
                #           slat(float), slong(float), selev(int), swdepth(int),
                #           rlat(float), rlong(float), relev(int),
                #           sxutm(float), syutm(float), sz(float),
                #           rxutm(float), ryutm(float), rz(float)
                
                # 解析前5个整数（字节0-19）
                nrec, itsn, ireci, itype, iflag = struct.unpack('<5i', header_bytes[0:20])
                
                # 解析2个浮点数（字节20-27）
                offset, azi = struct.unpack('<2f', header_bytes[20:28])
                
                # 解析1个整数（字节28-31）
                igain, = struct.unpack('<i', header_bytes[28:32])
                
                # 解析3个浮点数（字节32-43）
                texact, slat, slong = struct.unpack('<3f', header_bytes[32:44])
                
                # 解析2个整数（字节44-51）
                selev, swdepth = struct.unpack('<2i', header_bytes[44:52])
                
                # 解析2个浮点数（字节52-59）
                rlat, rlong = struct.unpack('<2f', header_bytes[52:60])
                
                # 解析1个整数（字节60-63）
                relev, = struct.unpack('<i', header_bytes[60:64])
                
                # 解析6个浮点数（字节64-87）
                sxutm, syutm, sz, rxutm, ryutm, rz = struct.unpack('<6f', header_bytes[64:88])
                
                # 读取拾取数组（npick个）
                picks_bytes = f.read(header.npick * 4)
                if len(picks_bytes) < header.npick * 4:
                    break
                picks = list(struct.unpack(f'<{header.npick}f', picks_bytes))
                
                # 读取记录长度标记（应该与rec_len相同）
                rec_len2_bytes = f.read(4)
                if len(rec_len2_bytes) < 4:
                    break
                rec_len2 = struct.unpack('<i', rec_len2_bytes)[0]
                
                # 转换单位
                offsti_km = offset / 1000.0  # 米转千米
                azi_deg = azi / 60.0  # 分转度
                
                # 如果offsti为0或接近0，尝试从UTM坐标计算炮检距
                if abs(offsti_km) < 0.001 and (abs(sxutm) > 0.001 or abs(rxutm) > 0.001):
                    # 计算炮检距：sqrt((rxutm - sxutm)^2 + (ryutm - syutm)^2)
                    dx = rxutm - sxutm
                    dy = ryutm - syutm
                    offsti_km = np.sqrt(dx*dx + dy*dy) / 1000.0  # 转换为千米
                
                trace_header = TraceHeader(
                    ishoti=nrec,  # C代码中nrec对应ishoti（炮站号/记录号）
                    itsn=itsn,
                    ireci=ireci,
                    itypei=itype,
                    iflagi=iflag,
                    igaini=igain,
                    offsti=offsti_km,
                    azi=azi_deg,
                    texact=texact,
                    slat=slat,
                    slong=slong,
                    selev=float(selev),  # 转换为float以匹配TraceHeader定义
                    swdepth=float(swdepth),  # 转换为float以匹配TraceHeader定义
                    rlat=rlat,
                    rlong=rlong,
                    relev=float(relev),  # 转换为float以匹配TraceHeader定义
                    sxutm=sxutm,
                    syutm=syutm,
                    sz=sz,
                    rxutm=rxutm,
                    ryutm=ryutm,
                    rz=rz,
                    picks=picks,
                    picks_from_header=True  # 从头文件读取，标记为来自头文件
                )
                
                trace_headers.append(trace_header)
                record_count += 1
        
        return trace_headers
    
    def _read_trace_headers_from_zfile(self, dfile: str, header: ZFormatHeader) -> List[TraceHeader]:
        """从.z文件读取道头信息（当头文件不存在时，向后兼容）
        
        改进：虽然从数据文件读取拾取信息，但标记为来自数据文件
        这样可以区分拾取信息的来源，优先使用头文件中的拾取信息
        
        .z文件中的道头格式：22 + npick 个浮点数
        - trace_header[0] = 记录号 (ishoti)
        - trace_header[1] = 道序号 (itsn)
        - trace_header[2] = 接收站号 (ireci)
        - trace_header[3] = 数据类型 (itypei)
        - trace_header[4] = 数据标志 (iflagi)
        - trace_header[5] = 炮检距（米）
        - trace_header[6] = 方位角（分）
        - trace_header[7] = 增益因子 (igaini)
        - trace_header[8-21] = 其他字段
        - trace_header[22:] = 拾取时间数组（向后兼容，但优先级低于头文件）
        """
        trace_headers = []
        self._header_parse_mode_counts = {"float": 0, "mixed": 0}
        
        # 计算记录大小
        trace_header_size = (22 + header.npick) * 4
        if header.ifmt == 1:  # float32
            trace_data_size = header.npts * 4
        else:  # int16
            trace_data_size = header.npts * 2
        
        record_size = trace_header_size + trace_data_size
        data_offset = self._infer_trace_data_offset(dfile, header, record_size)
        
        with open(dfile, 'rb') as f:
            # 跳过文件头（自动识别紧凑布局/固定记录布局）
            f.seek(data_offset)
            
            # 读取每条道的道头
            for i in range(header.ntraces):
                # 读取道头部分
                trace_header_bytes = f.read(trace_header_size)
                if len(trace_header_bytes) < trace_header_size:
                    break

                trace_header = self._parse_trace_header_from_zfile_bytes(
                    trace_header_bytes=trace_header_bytes,
                    ntraces=header.ntraces,
                    npick=header.npick,
                )
                trace_headers.append(trace_header)
                
                # 跳过道数据
                f.seek(trace_data_size, 1)
        
        return trace_headers

    def _parse_trace_header_from_zfile_bytes(
        self,
        trace_header_bytes: bytes,
        ntraces: int,
        npick: int,
    ) -> TraceHeader:
        """解析 .z 道头字节，自动兼容两种编码：
        1) 全 float32 编码（历史实现常见）
        2) Fortran 混合 int/float 编码（与 readf.f 对应）
        """
        if len(trace_header_bytes) < 88:
            raise ValueError("道头字节长度不足 88")

        # 方案A：全 float32 编码
        fvals = struct.unpack(f'<{22 + npick}f', trace_header_bytes)
        f_candidate = {
            "ishoti": int(fvals[0]),
            "itsn": int(fvals[1]),
            "ireci": int(fvals[2]),
            "itypei": int(fvals[3]),
            "iflagi": int(fvals[4]),
            "offset_m": fvals[5],
            "azi_min": fvals[6],
            "igaini": int(fvals[7]),
            "texact": fvals[8],
            "slat": fvals[9],
            "slong": fvals[10],
            "selev": float(int(fvals[11])),
            "swdepth": float(int(fvals[12])),
            "rlat": fvals[13],
            "rlong": fvals[14],
            "relev": float(int(fvals[15])),
            "sxutm": fvals[16],
            "syutm": fvals[17],
            "sz": fvals[18],
            "rxutm": fvals[19],
            "ryutm": fvals[20],
            "rz": fvals[21],
            "picks": list(fvals[22:22 + npick]),
        }

        # 方案B：Fortran 混合 int/float 编码（与 .hdr 一致）
        m_ishoti, m_itsn, m_ireci, m_itypei, m_iflagi = struct.unpack('<5i', trace_header_bytes[0:20])
        m_offset_m, m_azi_min = struct.unpack('<2f', trace_header_bytes[20:28])
        m_igaini, = struct.unpack('<i', trace_header_bytes[28:32])
        m_texact, m_slat, m_slong = struct.unpack('<3f', trace_header_bytes[32:44])
        m_selev_i, m_swdepth_i = struct.unpack('<2i', trace_header_bytes[44:52])
        m_rlat, m_rlong = struct.unpack('<2f', trace_header_bytes[52:60])
        m_relev_i, = struct.unpack('<i', trace_header_bytes[60:64])
        m_sxutm, m_syutm, m_sz, m_rxutm, m_ryutm, m_rz = struct.unpack('<6f', trace_header_bytes[64:88])
        if npick > 0:
            m_picks = list(struct.unpack(f'<{npick}f', trace_header_bytes[88:88 + npick * 4]))
        else:
            m_picks = []
        m_candidate = {
            "ishoti": m_ishoti,
            "itsn": m_itsn,
            "ireci": m_ireci,
            "itypei": m_itypei,
            "iflagi": m_iflagi,
            "offset_m": m_offset_m,
            "azi_min": m_azi_min,
            "igaini": m_igaini,
            "texact": m_texact,
            "slat": m_slat,
            "slong": m_slong,
            "selev": float(m_selev_i),
            "swdepth": float(m_swdepth_i),
            "rlat": m_rlat,
            "rlong": m_rlong,
            "relev": float(m_relev_i),
            "sxutm": m_sxutm,
            "syutm": m_syutm,
            "sz": m_sz,
            "rxutm": m_rxutm,
            "ryutm": m_ryutm,
            "rz": m_rz,
            "picks": m_picks,
        }

        def _score_layout(candidate: Dict[str, Any]) -> int:
            score = 0
            max_idx = max(10 * max(1, ntraces), 1000)

            # itsn/ireci 在真实数据中一般是正整数，误解码时常出现 0 或极端值
            if 1 <= candidate["itsn"] <= max_idx:
                score += 4
            if 1 <= candidate["ireci"] <= max_idx:
                score += 3
            if 1 <= candidate["ishoti"] <= max_idx:
                score += 2

            if -16 <= candidate["itypei"] <= 16:
                score += 2
            if -2048 <= candidate["iflagi"] <= 2048:
                score += 1
            if -2_000_000.0 <= float(candidate["offset_m"]) <= 2_000_000.0:
                score += 1
            if -21600.0 <= float(candidate["azi_min"]) <= 21600.0:
                score += 1

            # 误判 mixed->float 时前几个字段经常全部变成 0，这里明显降权
            if candidate["ishoti"] == 0 and candidate["itsn"] == 0 and candidate["ireci"] == 0:
                score -= 6
            return score

        score_float = _score_layout(f_candidate)
        score_mixed = _score_layout(m_candidate)
        if score_float >= score_mixed:
            candidate = f_candidate
            selected_mode = "float"
        else:
            candidate = m_candidate
            selected_mode = "mixed"
        self._header_parse_mode_counts[selected_mode] = self._header_parse_mode_counts.get(selected_mode, 0) + 1

        ishoti = int(candidate["ishoti"])
        itsn = int(candidate["itsn"])
        ireci = int(candidate["ireci"])
        itypei = int(candidate["itypei"])
        iflagi = int(candidate["iflagi"])
        offset_m = float(candidate["offset_m"])
        azi_min = float(candidate["azi_min"])
        igaini = int(candidate["igaini"])
        texact = float(candidate["texact"])
        slat = float(candidate["slat"])
        slong = float(candidate["slong"])
        selev = float(candidate["selev"])
        swdepth = float(candidate["swdepth"])
        rlat = float(candidate["rlat"])
        rlong = float(candidate["rlong"])
        relev = float(candidate["relev"])
        sxutm = float(candidate["sxutm"])
        syutm = float(candidate["syutm"])
        sz = float(candidate["sz"])
        rxutm = float(candidate["rxutm"])
        ryutm = float(candidate["ryutm"])
        rz = float(candidate["rz"])
        picks = list(candidate["picks"])

        offsti_km = offset_m / 1000.0
        azi_deg = azi_min / 60.0

        return TraceHeader(
            ishoti=ishoti,
            itsn=itsn,
            ireci=ireci,
            itypei=itypei,
            iflagi=iflagi,
            igaini=igaini,
            offsti=offsti_km,
            azi=azi_deg,
            texact=texact,
            slat=slat,
            slong=slong,
            selev=selev,
            swdepth=swdepth,
            rlat=rlat,
            rlong=rlong,
            relev=relev,
            sxutm=sxutm,
            syutm=syutm,
            sz=sz,
            rxutm=rxutm,
            ryutm=ryutm,
            rz=rz,
            picks=picks,
            picks_from_header=False,
        )

    def _log_load_summary(self, dfile: str, hfile: Optional[str]) -> None:
        """输出一次性加载诊断日志，便于排查大文件读取问题。"""
        if self.header is None:
            return

        header_source = "hdr" if (hfile and os.path.exists(hfile)) else "zfile"
        samples = []
        for th in self.trace_headers[:3]:
            samples.append(
                f"(ishoti={th.ishoti},itsn={th.itsn},ireci={th.ireci},"
                f"iflag={th.iflagi},itype={th.itypei},off_km={th.offsti:.3f})"
            )
        sample_text = "; ".join(samples) if samples else "no-trace-header"

        msg = (
            "[ZLOAD] "
            f"file={os.path.basename(dfile)} "
            f"layout={self._last_layout_name} "
            f"offset={self._last_trace_data_offset} "
            f"lazy_traces={isinstance(self.traces, LazyTraceCollection)} "
            f"header_source={header_source} "
            f"parse_modes={self._header_parse_mode_counts} "
            f"ntr={self.header.ntraces} npts={self.header.npts} npick={self.header.npick} ifmt={self.header.ifmt} "
            f"samples={sample_text}"
        )
        if self._verbose_load_logs:
            # print 保证在未配置 logging 时也能看到
            print(msg)
        logger.info(msg)

    def _log_trace_amplitude_samples(self, max_traces: int = 3, max_points: int = 8) -> None:
        """输出前几道振幅测试值，便于快速确认波形是否有效。"""
        if not self.traces:
            msg = "[ZAMP] no-trace-data"
            if self._verbose_load_logs:
                print(msg)
            logger.info(msg)
            return

        lines = []
        nshow = min(max_traces, len(self.traces))
        for i in range(nshow):
            tr = self.traces[i]
            if tr is None or len(tr) == 0:
                lines.append(f"trace#{i+1}: empty")
                continue
            tr_min = float(np.min(tr))
            tr_max = float(np.max(tr))
            tr_mean_abs = float(np.mean(np.abs(tr)))
            head_vals = ", ".join(f"{float(v):.6g}" for v in tr[:max_points])
            lines.append(
                f"trace#{i+1}: min={tr_min:.6g}, max={tr_max:.6g}, "
                f"mean_abs={tr_mean_abs:.6g}, head=[{head_vals}]"
            )

        msg = "[ZAMP] " + " | ".join(lines)
        if self._verbose_load_logs:
            print(msg)
        logger.info(msg)
    
    def _create_default_headers(self, header: ZFormatHeader) -> List[TraceHeader]:
        """创建默认道头（当头文件不存在且无法从.z文件读取时使用）"""
        trace_headers = []
        
        for i in range(header.ntraces):
            trace_header = TraceHeader(
                ishoti=1,
                itsn=i + 1,
                ireci=i + 1,
                itypei=1,  # 默认垂直分量
                iflagi=1,  # 默认有效
                igaini=1,
                offsti=float(i + 1),  # 默认炮检距
                azi=0.0,
                texact=0.0,
                slat=0.0,
                slong=0.0,
                selev=0.0,
                swdepth=0.0,
                rlat=0.0,
                rlong=0.0,
                relev=0.0,
                sxutm=0.0,
                syutm=0.0,
                sz=0.0,
                rxutm=0.0,
                ryutm=0.0,
                rz=0.0,
                picks=[0.0] * header.npick,
                picks_from_header=False  # 默认道头，拾取信息为空
            )
            trace_headers.append(trace_header)
        
        return trace_headers
    
    def save_z_format(self, dfile: str, hfile: Optional[str] = None,
                     write_picks_to_data: bool = False) -> bool:
        """保存 Z 格式数据
        
        改进：根据阶段2要求，拾取信息不写入数据文件（或写入0值占位）
        - 数据文件：只写入道头（22个字段）+ 0值占位（npick个）+ 道数据
        - 头文件：写入完整的道头信息（包括拾取信息）
        
        Args:
            dfile: 数据文件路径
            hfile: 头文件路径（可选，如果提供则保存拾取信息）
            write_picks_to_data: 是否在数据文件中写入拾取信息（默认False，写入0值占位）
            
        Returns:
            是否成功保存
        """
        if not self.header or not self.traces or not self.trace_headers:
            raise ValueError("没有加载的数据可保存")
        
        try:
            # 保存数据文件（不包含拾取信息，或写入0值占位）
            with open(dfile, 'wb') as f:
                # 1. 写入文件头（52字节）
                header_bytes = struct.pack(
                    '<7i',  # 前7个整数
                    self.header.ntraces,
                    self.header.npts,
                    self.header.sint,
                    self.header.tstart,
                    self.header.tend,
                    self.header.nrec,
                    self.header.npick
                )
                f.write(header_bytes)
                
                # 第8个是浮点数（vredf）
                f.write(struct.pack('<f', self.header.vredf))
                
                # 第9个是整数（ifmt）
                f.write(struct.pack('<i', self.header.ifmt))
                
                # 最后4个浮点数
                f.write(struct.pack('<4f', 
                    self.header.xlatlong,
                    self.header.xelev,
                    self.header.xutm,
                    self.header.cm
                ))
                
                # 2. 写入每条道
                trace_header_size = 22 * 4  # 只写入22个字段（不包含拾取）
                picks_placeholder_size = self.header.npick * 4  # 拾取占位符大小
                
                if self.header.ifmt == 1:  # float32
                    trace_data_size = self.header.npts * 4
                    dtype = np.float32
                else:  # int16
                    trace_data_size = self.header.npts * 2
                    dtype = np.int16
                
                # 容错保存：
                # - 懒加载模式下，个别道可能因源文件截断而读取失败
                # - 为避免整次保存失败，失败道用 0 值补齐并继续写出
                nwrite = min(int(self.header.ntraces), len(self.trace_headers), len(self.traces))
                bad_trace_count = 0
                bad_trace_indices: List[int] = []
                for i in range(nwrite):
                    th = self.trace_headers[i]
                    try:
                        trace = self.traces[i]
                    except Exception as trace_err:
                        bad_trace_count += 1
                        bad_trace_indices.append(int(i))
                        logger.warning(
                            "保存Z文件时读取道失败，使用0值占位继续保存: idx=%s, err=%s",
                            i, trace_err
                        )
                        trace = np.zeros(int(self.header.npts), dtype=dtype)
                    else:
                        trace = np.asarray(trace)
                        if trace.shape[0] < int(self.header.npts):
                            bad_trace_count += 1
                            bad_trace_indices.append(int(i))
                            logger.warning(
                                "保存Z文件时道长度不足，自动补零: idx=%s, got=%s, expected=%s",
                                i, trace.shape[0], self.header.npts
                            )
                            padded = np.zeros(int(self.header.npts), dtype=trace.dtype if trace.size > 0 else dtype)
                            if trace.size > 0:
                                padded[:trace.shape[0]] = trace
                            trace = padded
                        elif trace.shape[0] > int(self.header.npts):
                            trace = trace[:int(self.header.npts)]
                    # 2.1 写入道头（22个字段，不包含拾取）
                    trace_header_array = np.zeros(22, dtype=np.float32)
                    trace_header_array[0] = float(th.ishoti)
                    trace_header_array[1] = float(th.itsn)
                    trace_header_array[2] = float(th.ireci)
                    trace_header_array[3] = float(th.itypei)
                    trace_header_array[4] = float(th.iflagi)
                    trace_header_array[5] = th.offsti * 1000.0  # 千米转米
                    trace_header_array[6] = th.azi * 60.0  # 度转分
                    trace_header_array[7] = float(th.igaini)
                    trace_header_array[8] = th.texact
                    trace_header_array[9] = th.slat
                    trace_header_array[10] = th.slong
                    trace_header_array[11] = float(th.selev)
                    trace_header_array[12] = float(th.swdepth)
                    trace_header_array[13] = th.rlat
                    trace_header_array[14] = th.rlong
                    trace_header_array[15] = float(th.relev)
                    trace_header_array[16] = th.sxutm
                    trace_header_array[17] = th.syutm
                    trace_header_array[18] = th.sz
                    trace_header_array[19] = th.rxutm
                    trace_header_array[20] = th.ryutm
                    trace_header_array[21] = th.rz
                    
                    f.write(trace_header_array.tobytes())
                    
                    # 2.2 写入拾取占位符（0值，不写入实际拾取信息）
                    if write_picks_to_data:
                        # 向后兼容模式：写入实际拾取信息
                        picks_array = th.picks if th.picks else [0.0] * self.header.npick
                        if len(picks_array) < self.header.npick:
                            picks_array.extend([0.0] * (self.header.npick - len(picks_array)))
                        elif len(picks_array) > self.header.npick:
                            picks_array = picks_array[:self.header.npick]
                        f.write(struct.pack(f'<{self.header.npick}f', *picks_array))
                    else:
                        # 新格式：写入0值占位符（拾取信息不存储在数据文件中）
                        picks_placeholder = [0.0] * self.header.npick
                        f.write(struct.pack(f'<{self.header.npick}f', *picks_placeholder))
                    
                    # 2.3 写入道数据
                    trace_data = trace.astype(dtype, copy=False)
                    f.write(trace_data.tobytes())

                if bad_trace_count > 0:
                    preview = ",".join(str(x) for x in bad_trace_indices[:10])
                    more = "" if len(bad_trace_indices) <= 10 else f"...(+{len(bad_trace_indices) - 10})"
                    logger.warning(
                        "保存Z文件完成（含容错补零）: bad_traces=%s, indices=%s%s",
                        bad_trace_count, preview, more
                    )
            
            # 3. 如果提供了头文件路径，保存拾取信息到头文件
            if hfile:
                # 使用 pick_manager 保存拾取信息到头文件
                # 注意：这里需要导入 pick_manager，或者直接调用保存方法
                # 为了保持模块独立性，这里只保存数据文件
                # 头文件的保存应该通过 pick_manager.save_to_header_file 完成
                pass
            
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ValueError(f"保存数据文件失败: {e}")
    
    def _read_record_file(self, rfile: str) -> List[RecordInfo]:
        """读取记录文件（ASCII格式）"""
        records = []
        
        with open(rfile, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        ishnum = int(parts[0])
                        xmod = float(parts[1])
                        ymod = float(parts[2])
                        az = float(parts[3])
                        title = ' '.join(parts[4:]) if len(parts) > 4 else ''
                        
                        records.append(RecordInfo(
                            ishnum=ishnum,
                            xmod=xmod,
                            ymod=ymod,
                            az=az,
                            title=title
                        ))
                    except (ValueError, IndexError):
                        continue
        
        return records
    
    def get_trace(self, trace_idx: int) -> Optional[np.ndarray]:
        """获取指定道的数据"""
        if 0 <= trace_idx < len(self.traces):
            return self.traces[trace_idx]
        return None
    
    def get_trace_header(self, trace_idx: int) -> Optional[TraceHeader]:
        """获取指定道的头信息"""
        if 0 <= trace_idx < len(self.trace_headers):
            return self.trace_headers[trace_idx]
        return None
    
    def get_traces_for_record(self, record_idx: int) -> List[int]:
        """获取指定记录（炮集）的所有道索引
        
        Args:
            record_idx: 记录索引（从0开始）
            
        Returns:
            道索引列表
        """
        if not self.records or record_idx >= len(self.records):
            return []
        
        record = self.records[record_idx]
        trace_indices = []
        
        for i, th in enumerate(self.trace_headers):
            if th.ishoti == record.ishnum:
                trace_indices.append(i)
        
        return trace_indices
