# -*- coding: utf-8 -*-
"""SEGY / SU 数据集：道头缓存、按需读样本、写回道头。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import struct

import numpy as np

from .raw2sac_paths import import_segy_trace_header

_sth = import_segy_trace_header()
SEGY_TRACE_FIELDS = _sth.SEGY_TRACE_FIELDS
SEGY_TRACE_HEADER_BYTES = _sth.SEGY_TRACE_HEADER_BYTES
scale_segy_factor = _sth.scale_segy_factor
offset_m_from_xy = _sth.offset_m_from_xy
_pack_th = _sth.pack_trace_header
_unpack_th = _sth.unpack_trace_header

try:
    from format_utils import ibm_to_ieee_float32  # type: ignore
except ImportError:
    from pyAOBS.processors.raw2sac.format_utils import ibm_to_ieee_float32

FIELD_NAMES: List[str] = list(SEGY_TRACE_FIELDS.keys())


def pack_trace_header(trace_header, endian="big", base=None, *, verify=False):
    return _pack_th(trace_header, endian=endian, base=base, verify=verify)


def unpack_trace_header(buf, endian="little", fields=None):
    return _unpack_th(buf, endian=endian, fields=fields)

# 几何相关列置顶
GEOM_PRIORITY = [
    "tracl", "tracr", "fldr", "tracf", "ep", "cdp", "trid",
    "offset", "scalel", "scalco", "sx", "sy", "gx", "gy",
    "gelev", "selev", "sdepth", "swdep", "gwdep", "counit",
    "ns", "dt",
]


def ordered_field_names() -> List[str]:
    seen = set()
    out: List[str] = []
    for name in GEOM_PRIORITY:
        if name in SEGY_TRACE_FIELDS and name not in seen:
            out.append(name)
            seen.add(name)
    for name in FIELD_NAMES:
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


COLUMN_NAMES: List[str] = ordered_field_names()


def _is_su_path(path: Path) -> bool:
    return path.suffix.lower() in {".su", ".rsf"} or path.name.lower().endswith(".su")


def _probe_endian_and_format(path: Path) -> Tuple[str, bool, int]:
    """
    返回 (endian, is_su, data_offset)。
    data_offset: SU=0；SEGY=3600。
    """
    size = path.stat().st_size
    is_su = _is_su_path(path)
    candidates: List[Tuple[str, bool, int]] = []
    if is_su:
        candidates = [("little", True, 0), ("big", True, 0)]
    else:
        # SEGY 常规 big；也尝试 little / 无 reel 头的 SU 伪装
        candidates = [
            ("big", False, 3600),
            ("little", False, 3600),
            ("little", True, 0),
            ("big", True, 0),
        ]

    best: Optional[Tuple[str, bool, int, int]] = None  # endian, is_su, off, score
    with path.open("rb") as f:
        blob = f.read(min(size, 3600 + SEGY_TRACE_HEADER_BYTES + 8))
    for endian, su, off in candidates:
        if off + SEGY_TRACE_HEADER_BYTES > len(blob) and off > 0:
            continue
        if off > 0 and size < off + SEGY_TRACE_HEADER_BYTES:
            continue
        try:
            th = unpack_trace_header(blob[off : off + SEGY_TRACE_HEADER_BYTES], endian=endian)
        except Exception:
            continue
        ns = int(th.get("ns", 0) or 0)
        dt = int(th.get("dt", 0) or 0)
        if ns <= 0 or ns > 2_000_000:
            continue
        tr_bytes = SEGY_TRACE_HEADER_BYTES + ns * 4
        if tr_bytes <= 0:
            continue
        rem = size - off
        if rem < tr_bytes:
            continue
        ntr = rem // tr_bytes
        if ntr <= 0:
            continue
        # 余数越小越好；ns/dt 合理加分
        score = 0
        if rem % tr_bytes == 0:
            score += 100
        if 1 <= dt <= 1_000_000:
            score += 10
        if 16 <= ns <= 500_000:
            score += 10
        if su == is_su:
            score += 5
        if best is None or score > best[3]:
            best = (endian, su, off, score)
    if best is None:
        # 回退
        if is_su:
            return "little", True, 0
        return "big", False, 3600
    return best[0], best[1], best[2]


def _binary_header_format_code(bhed: bytes) -> int:
    """SEGY 二进制头 format（字节 24–25，big-endian）。1=IBM，5=IEEE。"""
    if not bhed or len(bhed) < 26:
        return 0
    return int(struct.unpack(">h", bhed[24:26])[0])


@dataclass
class SegyDataset:
    """内存中仅缓存道头；样本按需从文件读取。"""

    path: Optional[Path] = None
    endian: str = "little"
    is_su: bool = True
    data_offset: int = 0
    sample_format: int = 0  # SEGY bhed format；SU 视为 IEEE(5)
    binary_header: bytes = b""
    ebcdic_header: bytes = b""
    headers: List[Dict[str, int]] = field(default_factory=list)
    dirty: set = field(default_factory=set)
    _trace_byte_offsets: List[int] = field(default_factory=list)

    @property
    def ntraces(self) -> int:
        return len(self.headers)

    @property
    def dirty_count(self) -> int:
        return len(self.dirty)

    @property
    def is_open(self) -> bool:
        return self.path is not None and self.ntraces > 0

    def clear(self) -> None:
        self.path = None
        self.headers = []
        self.dirty = set()
        self._trace_byte_offsets = []
        self.binary_header = b""
        self.ebcdic_header = b""
        self.sample_format = 0

    def open(self, path: str | Path, *, endian: Optional[str] = None) -> None:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(str(p))
        det_endian, is_su, data_offset = _probe_endian_and_format(p)
        if endian in ("little", "big"):
            det_endian = endian
        headers: List[Dict[str, int]] = []
        offsets: List[int] = []
        ebcdic = b""
        bhed = b""
        # 整文件读入再解析：比逐道 seek 快（中等文件）；超大文件仍走流式
        size = p.stat().st_size
        use_blob = size <= 512 * 1024 * 1024  # 512 MiB
        if use_blob:
            blob = p.read_bytes()
            if not is_su and data_offset >= 3600:
                ebcdic = blob[:3200]
                bhed = blob[3200:3600]
                pos = 3600
            else:
                pos = 0
            n = len(blob)
            while pos + SEGY_TRACE_HEADER_BYTES <= n:
                hdr = blob[pos : pos + SEGY_TRACE_HEADER_BYTES]
                th = unpack_trace_header(hdr, endian=det_endian)
                ns = int(th.get("ns", 0) or 0)
                if ns <= 0:
                    raise RuntimeError(
                        f"{p}: bad ns={ns} at byte {pos} (try other endian)"
                    )
                tr_bytes = SEGY_TRACE_HEADER_BYTES + ns * 4
                if pos + tr_bytes > n:
                    break
                offsets.append(pos)
                headers.append(th)
                pos += tr_bytes
        else:
            with p.open("rb") as f:
                if not is_su and data_offset >= 3600:
                    ebcdic = f.read(3200)
                    bhed = f.read(400)
                    pos = 3600
                else:
                    pos = 0
                    f.seek(0)
                while True:
                    hdr = f.read(SEGY_TRACE_HEADER_BYTES)
                    if not hdr or len(hdr) < SEGY_TRACE_HEADER_BYTES:
                        break
                    th = unpack_trace_header(hdr, endian=det_endian)
                    ns = int(th.get("ns", 0) or 0)
                    if ns <= 0:
                        raise RuntimeError(
                            f"{p}: bad ns={ns} at byte {pos} (try other endian)"
                        )
                    offsets.append(pos)
                    headers.append(th)
                    sample_bytes = ns * 4
                    f.seek(sample_bytes, 1)
                    pos += SEGY_TRACE_HEADER_BYTES + sample_bytes
        if not headers:
            raise RuntimeError(f"{p}: no traces found")
        self.path = p
        self.endian = det_endian
        self.is_su = is_su
        self.data_offset = data_offset if not is_su else 0
        self.ebcdic_header = ebcdic
        self.binary_header = bhed
        if is_su:
            self.sample_format = 5  # IEEE float（SU 惯例）
        else:
            fmt = _binary_header_format_code(bhed)
            self.sample_format = fmt if fmt else 1  # sac2y 默认 IBM
        self.headers = headers
        self._trace_byte_offsets = offsets
        self.dirty = set()

    def get_header(self, row: int) -> Dict[str, int]:
        return self.headers[row]

    def set_header_value(self, row: int, field: str, value: int) -> None:
        if field not in SEGY_TRACE_FIELDS:
            raise KeyError(field)
        self.headers[row][field] = int(value)
        self.dirty.add(row)

    def set_header_row(self, row: int, values: Dict[str, int], *, fields: Optional[Sequence[str]] = None) -> None:
        names = fields or values.keys()
        for name in names:
            if name not in SEGY_TRACE_FIELDS:
                continue
            if name in values:
                self.headers[row][name] = int(values[name])
        self.dirty.add(row)

    def physical_xy(self, row: int) -> Dict[str, float]:
        """返回已 scalco 的物理坐标（米）。"""
        th = self.headers[row]
        sc = int(th.get("scalco", 0) or 0)
        return {
            "sx": scale_segy_factor(th.get("sx", 0), sc),
            "sy": scale_segy_factor(th.get("sy", 0), sc),
            "gx": scale_segy_factor(th.get("gx", 0), sc),
            "gy": scale_segy_factor(th.get("gy", 0), sc),
        }

    def recompute_offset(self, rows: Optional[Sequence[int]] = None, *, keep_sign: bool = True) -> int:
        """按物理 XY 重算 offset（不受 scalco）。返回修改道数。"""
        targets = list(rows) if rows is not None else list(range(self.ntraces))
        n = 0
        for i in targets:
            xy = self.physical_xy(i)
            dist = offset_m_from_xy(xy["sx"], xy["sy"], xy["gx"], xy["gy"])
            old = int(self.headers[i].get("offset", 0) or 0)
            if keep_sign and old < 0:
                new_v = -int(round(abs(dist)))
            else:
                new_v = int(round(dist))
            if new_v != old:
                self.headers[i]["offset"] = new_v
                self.dirty.add(i)
                n += 1
        return n

    def read_samples(self, row: int) -> np.ndarray:
        if self.path is None:
            raise RuntimeError("no file open")
        th = self.headers[row]
        ns = int(th.get("ns", 0) or 0)
        off = self._trace_byte_offsets[row] + SEGY_TRACE_HEADER_BYTES
        with self.path.open("rb") as f:
            f.seek(off)
            buf = f.read(ns * 4)
        if len(buf) < ns * 4:
            raise RuntimeError(f"short read on trace {row}")
        return self._decode_samples(buf, ns)

    def read_samples_many(self, rows: Sequence[int]) -> List[np.ndarray]:
        """一次打开文件批量读样本（道集预览用）。"""
        if self.path is None:
            raise RuntimeError("no file open")
        out: List[np.ndarray] = []
        with self.path.open("rb") as f:
            for row in rows:
                i = int(row)
                th = self.headers[i]
                ns = int(th.get("ns", 0) or 0)
                off = self._trace_byte_offsets[i] + SEGY_TRACE_HEADER_BYTES
                f.seek(off)
                buf = f.read(ns * 4)
                if len(buf) < ns * 4:
                    raise RuntimeError(f"short read on trace {i}")
                out.append(self._decode_samples(buf, ns))
        return out

    def _decode_samples(self, buf: bytes, ns: int) -> np.ndarray:
        if (not self.is_su) and int(self.sample_format) == 1:
            return ibm_to_ieee_float32(buf, src_endian=self.endian)
        dtype = np.dtype(">f4" if self.endian == "big" else "<f4")
        return np.frombuffer(buf, dtype=dtype, count=ns).astype(np.float32, copy=True)

    def export_su(
        self,
        path: str | Path,
        *,
        endian: str = "little",
    ) -> Path:
        """导出为 SU：无卷头；道头+IEEE float 样本；默认 little-endian（PC 惯例）。

        几何道头原样写入（约定：炮=sx/sy，OBS=gx/gy）。
        若源为 SEGY IBM(float format=1)，会先解码为 IEEE 再写出。
        """
        if self.path is None or self.ntraces == 0:
            raise RuntimeError("no file open")
        if endian not in ("little", "big"):
            raise ValueError("endian must be little or big")
        dest = Path(path).expanduser().resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".idata_tmp")
        samp_dtype = np.dtype("<f4" if endian == "little" else ">f4")
        with tmp.open("wb") as fout:
            for i, th in enumerate(self.headers):
                samples = np.asarray(self.read_samples(i), dtype=np.float32)
                ns = int(samples.size)
                th_out = dict(th)
                th_out["ns"] = ns
                fout.write(pack_trace_header(th_out, endian=endian))
                fout.write(samples.astype(samp_dtype, copy=False).tobytes())
        if dest.exists():
            dest.unlink()
        tmp.replace(dest)
        return dest

    def export_segy(
        self,
        path: str | Path,
        *,
        endian: str = "big",
        sample_format: int = 5,
    ) -> Path:
        """导出为 SEGY：3200+400 卷头 + 道；默认 big-endian、IEEE float（format=5）。

        从 SU 或其它 SEGY 转出时，样本经 ``read_samples`` 得到 IEEE，再按目标
        endian 打包。不写 IBM(format=1)，避免二次量化损失。
        """
        if self.path is None or self.ntraces == 0:
            raise RuntimeError("no file open")
        if endian not in ("little", "big"):
            raise ValueError("endian must be little or big")
        if int(sample_format) not in (1, 5):
            # 本路径只保证 IEEE 写出；format 代码写入卷头供阅读器识别
            sample_format = 5
        dest = Path(path).expanduser().resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".idata_tmp")

        th0 = self.headers[0] if self.headers else {}
        ns0 = int(th0.get("ns", 0) or 0)
        dt0 = int(th0.get("dt", 0) or 0)

        if self.ebcdic_header and len(self.ebcdic_header) >= 3200:
            ebcdic = self.ebcdic_header[:3200]
        else:
            ebcdic = b"\x40" * 3200

        bhed = bytearray(400)
        if self.binary_header and len(self.binary_header) >= 400 and not self.is_su:
            bhed[:] = self.binary_header[:400]
        # 覆盖关键字段（1-based 字节 → 0-based offset）
        ntr_write = min(int(self.ntraces), 32767)
        struct.pack_into(">h", bhed, 12, ntr_write)  # ntrpr
        if dt0 > 0:
            struct.pack_into(">H", bhed, 16, dt0 & 0xFFFF)  # hdt
            struct.pack_into(">H", bhed, 18, dt0 & 0xFFFF)  # dto
        if ns0 > 0:
            struct.pack_into(">H", bhed, 20, ns0 & 0xFFFF)  # hns
            struct.pack_into(">H", bhed, 22, ns0 & 0xFFFF)  # nso
        # 强制 IEEE：本导出路径样本为 IEEE float
        struct.pack_into(">h", bhed, 24, 5)  # format
        struct.pack_into(">h", bhed, 54, 1)  # mfeet=1 米

        samp_dtype = np.dtype(">f4" if endian == "big" else "<f4")
        with tmp.open("wb") as fout:
            fout.write(ebcdic)
            fout.write(bytes(bhed))
            for i, th in enumerate(self.headers):
                samples = np.asarray(self.read_samples(i), dtype=np.float32)
                ns = int(samples.size)
                th_out = dict(th)
                th_out["ns"] = ns
                fout.write(pack_trace_header(th_out, endian=endian))
                fout.write(samples.astype(samp_dtype, copy=False).tobytes())
        if dest.exists():
            dest.unlink()
        tmp.replace(dest)
        return dest

    def save(self, path: Optional[str | Path] = None) -> Path:
        """写回道头。同路径且仅道头脏时原地补丁（快）；否则整文件重写。

        另存为若目标后缀与当前容器不一致（SU↔SEGY），走真正格式转换：
        SEGY→SU 用 ``export_su``；SU→SEGY 用 ``export_segy``（IEEE format=5）。
        同格式另存为仍按源容器拷贝（SEGY 可保留原 IBM 样本字节）。
        """
        if self.path is None:
            raise RuntimeError("no file open")
        dest = Path(path).expanduser().resolve() if path is not None else self.path
        src = self.path
        same = dest.resolve() == src.resolve()

        # 跨格式另存为：按目标后缀转换（不能只改扩展名）
        if not same:
            dest_is_su = _is_su_path(dest)
            if dest_is_su and not self.is_su:
                out = self.export_su(dest, endian="little")
                self.open(out, endian="little")
                return out
            if (not dest_is_su) and self.is_su:
                # 无后缀时按 SEGY 处理；.segy/.sgy 明确走 SEGY
                out = self.export_segy(dest, endian="big", sample_format=5)
                self.open(out, endian="big")
                return out

        # 快路径：原地只改脏道头，不重写样本、不重新 open
        if same and self.dirty:
            with dest.open("r+b") as f:
                for i in sorted(self.dirty):
                    hdr_off = self._trace_byte_offsets[i]
                    f.seek(hdr_off)
                    orig_hdr = f.read(SEGY_TRACE_HEADER_BYTES)
                    packed = pack_trace_header(
                        self.headers[i], endian=self.endian, base=orig_hdr
                    )
                    f.seek(hdr_off)
                    f.write(packed)
            self.dirty.clear()
            return dest

        if same and not self.dirty:
            return dest

        # 另存为 / 无脏标记时的整文件写
        tmp = dest.with_suffix(dest.suffix + ".idata_tmp")
        with src.open("rb") as fin, tmp.open("wb") as fout:
            if not self.is_su:
                if self.ebcdic_header and self.binary_header:
                    fout.write(self.ebcdic_header[:3200].ljust(3200, b"\x40")[:3200])
                    fout.write(self.binary_header[:400].ljust(400, b"\x00")[:400])
                else:
                    fout.write(fin.read(3600))
            for i, th in enumerate(self.headers):
                ns = int(th.get("ns", 0) or 0)
                hdr_off = self._trace_byte_offsets[i]
                sample_off = hdr_off + SEGY_TRACE_HEADER_BYTES
                fin.seek(hdr_off)
                orig_hdr = fin.read(SEGY_TRACE_HEADER_BYTES)
                fout.write(
                    pack_trace_header(th, endian=self.endian, base=orig_hdr)
                )
                # 顺序读样本，避免二次 seek
                if fin.tell() != sample_off:
                    fin.seek(sample_off)
                fout.write(fin.read(ns * 4))
        if same:
            bak = dest.with_suffix(dest.suffix + ".bak")
            if bak.exists():
                bak.unlink()
            dest.replace(bak)
            tmp.replace(dest)
            try:
                bak.unlink()
            except Exception:
                pass
            self.dirty.clear()
            return dest

        if dest.exists():
            dest.unlink()
        tmp.replace(dest)
        self.path = dest
        self.open(dest, endian=self.endian)
        return dest

    def swap_source_group_slots(self, rows: Optional[Sequence[int]] = None) -> int:
        """交换 s*/g* 几何槽（literal SEGY ↔ 本工区 obs）。"""
        pairs = [
            ("sx", "gx"),
            ("sy", "gy"),
            ("selev", "gelev"),
            ("swdep", "gwdep"),
            ("sut", "gut"),
            ("sstat", "gstat"),
        ]
        targets = list(rows) if rows is not None else list(range(self.ntraces))
        for i in targets:
            th = self.headers[i]
            for a, b in pairs:
                th[a], th[b] = int(th.get(b, 0) or 0), int(th.get(a, 0) or 0)
            self.dirty.add(i)
        return len(targets)


def convert_segy_to_su(
    src: str | Path,
    dest: str | Path,
    *,
    endian: str = "little",
    src_endian: Optional[str] = None,
) -> Path:
    """SEGY（或已打开的同类文件）→ SU 文件。"""
    ds = SegyDataset()
    ds.open(src, endian=src_endian)
    return ds.export_su(dest, endian=endian)
