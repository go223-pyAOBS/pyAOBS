# -*- coding: utf-8 -*-
"""
SEGY / SU 道头字段偏移（与本目录 segy.h 一致）
================================================
来源:
  - Barry et al., Geophysics 40, 344–352 (1975) SEGY 推荐格式
  - Colorado School of Mines CWP/SU ``segy.h``（本仓库 ``processors/raw2sac/segy.h``）

说明:
  - 磁带/文件上的道识别头固定 **240 字节**；下列 offset 为字节偏移。
  - 类型同 segy.h：``int``→i32，``short``→i16，``unsigned short``→u16。
  - ``trid``：标准 1=地震道 …；9–N 为 optional use。
  - segy.h 注释里 CWP 对 trid≥9 的傅里叶等含义是处理标志，与 OBS 用工区
    约定标分量（水听/Z/R/T）不是同一套语义（见 modeling/madagascar_obs_rtm/trid_components.txt）。
  - segy.h 在 otrav 之后的 d1/f1/… 是 SU **内存扩展**，不占 240 字节磁盘道头。

坐标 / 高程（segy.h；物理长度 = 原值经 scalco/scalel 缩放，本工区为米）
  - ``scalco`` → sx,sy,gx,gy；``scalel`` → gelev…gwdep。
  - 符号只表示运算，不是“负的缩放倍数”：
      scale > 0 → 物理量 = 整型值 × scale
      scale < 0 → 物理量 = 整型值 ÷ |scale|   （负号=除，因子用绝对值）
      scale = 0 或 ±1 → 不缩放
  - SEGY 字面：sx/sy=震源 XY，gx/gy=检波点 XY；
    gelev=检波点高程、selev=震源高程（海平面以上为正）；
    sdepth=震源深度（恒正）；swdep/gwdep=炮/检处水深。

本工区 OBS 装填约定（相对 SEGY 字面炮/检对调，详见 obs_segy_geometry.txt）
  - sx,sy = OBS UTM；gx,gy = 炮点 UTM
  - selev = OBS 高程（海平面以下为负；深度 ≡ -selev，或用恒正的 sdepth）
  - gelev = 炮点高程（海平面以下为负）
  - offset = 炮检距(m)，不受 scalco；有非零道头则用，否则由 sx/sy/gx/gy 计算

供 ``segy_utils.py``、``madagascar_obs_rtm/su_to_shots.py`` 等共用。
"""
from __future__ import print_function
import struct

SEGY_TRACE_HEADER_BYTES = 240

# name -> (byte_offset, kind)  — 与 segy.h 成员顺序/布局一致（至 otrav）
SEGY_TRACE_FIELDS = {
    "tracl": (0, "i32"),
    "tracr": (4, "i32"),
    "fldr": (8, "i32"),
    "tracf": (12, "i32"),
    "ep": (16, "i32"),
    "cdp": (20, "i32"),
    "cdpt": (24, "i32"),
    "trid": (28, "i16"),
    "nvs": (30, "i16"),
    "nhs": (32, "i16"),
    "duse": (34, "i16"),
    "offset": (36, "i32"),
    "gelev": (40, "i32"),
    "selev": (44, "i32"),
    "sdepth": (48, "i32"),
    "gdel": (52, "i32"),
    "sdel": (56, "i32"),
    "swdep": (60, "i32"),
    "gwdep": (64, "i32"),
    "scalel": (68, "i16"),
    "scalco": (70, "i16"),
    "sx": (72, "i32"),
    "sy": (76, "i32"),
    "gx": (80, "i32"),
    "gy": (84, "i32"),
    # 以下与 SEG-Y Rev1 / segy.h 一致：gy 占 84..87，counit 从 88 起
    # （旧表误把 counit 放在 86，pack 时会覆盖 gy 高 16 位）
    "counit": (88, "i16"),
    "wevel": (90, "i16"),
    "swevel": (92, "i16"),
    "sut": (94, "i16"),
    "gut": (96, "i16"),
    "sstat": (98, "i16"),
    "gstat": (100, "i16"),
    "tstat": (102, "i16"),
    "laga": (104, "i16"),
    "lagb": (106, "i16"),
    "delrt": (108, "i16"),
    "muts": (110, "i16"),
    "mute": (112, "i16"),
    "ns": (114, "u16"),
    "dt": (116, "u16"),  # microseconds
    "gain": (118, "i16"),
    "igc": (120, "i16"),
    "igi": (122, "i16"),
    "corr": (124, "i16"),
    "sfs": (126, "i16"),
    "sfe": (128, "i16"),
    "slen": (130, "i16"),
    "styp": (132, "i16"),
    "stas": (134, "i16"),
    "stae": (136, "i16"),
    "tatyp": (138, "i16"),
    "afilf": (140, "i16"),
    "afils": (142, "i16"),
    "nofilf": (144, "i16"),
    "nofils": (146, "i16"),
    "lcf": (148, "i16"),
    "hcf": (150, "i16"),
    "lcs": (152, "i16"),
    "hcs": (154, "i16"),
    "year": (156, "i16"),
    "day": (158, "i16"),
    "hour": (160, "i16"),
    "minute": (162, "i16"),
    "sec": (164, "i16"),
    "timbas": (166, "i16"),
    "trwf": (168, "i16"),
    "grnors": (170, "i16"),
    "grnofr": (172, "i16"),
    "grnlof": (174, "i16"),
    "gaps": (176, "i16"),
    "otrav": (178, "i16"),
}

SEGY_TRACE_OFFSET = {k: v[0] for k, v in SEGY_TRACE_FIELDS.items()}

# Madagascar sfsuread/sfsegyread tfile：按键顺序存 int32
MADAGASCAR_TFILE_KEY_ORDER = [
    "tracl", "tracr", "fldr", "tracf", "ep", "cdp", "cdpt", "trid",
    "nvs", "nhs", "duse", "offset", "gelev", "selev", "sdepth", "gdel",
    "sdel", "swdep", "gwdep", "scalel", "scalco", "sx", "sy", "gx", "gy",
    "counit", "wevel", "swevel", "sut", "gut", "sstat", "gstat", "tstat",
    "laga", "lagb", "delrt", "muts", "mute", "ns", "dt",
]

_STRUCT_CODE = {"i32": "i", "i16": "h", "u16": "H"}
_KIND_SIZE = {"i32": 4, "i16": 2, "u16": 2}

# segy.h 成员顺序/类型（磁盘 240B 至 otrav；其后 d1/f1… 为 SU 内存扩展，不进表）
_SEGY_H_LAYOUT = (
    ("tracl", "i32"), ("tracr", "i32"), ("fldr", "i32"), ("tracf", "i32"),
    ("ep", "i32"), ("cdp", "i32"), ("cdpt", "i32"),
    ("trid", "i16"), ("nvs", "i16"), ("nhs", "i16"), ("duse", "i16"),
    ("offset", "i32"), ("gelev", "i32"), ("selev", "i32"), ("sdepth", "i32"),
    ("gdel", "i32"), ("sdel", "i32"), ("swdep", "i32"), ("gwdep", "i32"),
    ("scalel", "i16"), ("scalco", "i16"),
    ("sx", "i32"), ("sy", "i32"), ("gx", "i32"), ("gy", "i32"),
    ("counit", "i16"), ("wevel", "i16"), ("swevel", "i16"),
    ("sut", "i16"), ("gut", "i16"), ("sstat", "i16"), ("gstat", "i16"),
    ("tstat", "i16"), ("laga", "i16"), ("lagb", "i16"), ("delrt", "i16"),
    ("muts", "i16"), ("mute", "i16"), ("ns", "u16"), ("dt", "u16"),
    ("gain", "i16"), ("igc", "i16"), ("igi", "i16"), ("corr", "i16"),
    ("sfs", "i16"), ("sfe", "i16"), ("slen", "i16"), ("styp", "i16"),
    ("stas", "i16"), ("stae", "i16"), ("tatyp", "i16"),
    ("afilf", "i16"), ("afils", "i16"), ("nofilf", "i16"), ("nofils", "i16"),
    ("lcf", "i16"), ("hcf", "i16"), ("lcs", "i16"), ("hcs", "i16"),
    ("year", "i16"), ("day", "i16"), ("hour", "i16"), ("minute", "i16"),
    ("sec", "i16"), ("timbas", "i16"), ("trwf", "i16"),
    ("grnors", "i16"), ("grnofr", "i16"), ("grnlof", "i16"),
    ("gaps", "i16"), ("otrav", "i16"),
)


def validate_segy_trace_fields(fields=None):
    """
    校验道头偏移表：与 segy.h 布局一致、无重叠、pack/unpack 往返不丢值。
    返回 (ok: bool, errors: list[str])。
    """
    table = fields if fields is not None else SEGY_TRACE_FIELDS
    errors = []

    # 1) 对照 segy.h 累计布局
    off = 0
    expected = {}
    for name, kind in _SEGY_H_LAYOUT:
        expected[name] = (off, kind)
        off += _KIND_SIZE[kind]
    if off != 180:
        errors.append("segy.h layout size to otrav end=%d (expect 180)" % off)

    for name, (eoff, ekind) in expected.items():
        if name not in table:
            errors.append("missing field %s" % name)
            continue
        goff, gkind = table[name]
        if goff != eoff or gkind != ekind:
            errors.append(
                "%s: got (%d,%s) expect (%d,%s)" % (name, goff, gkind, eoff, ekind)
            )
    for name in table:
        if name not in expected:
            errors.append("unexpected extra field %s" % name)

    # 2) 重叠 / 越界
    items = sorted(table.items(), key=lambda kv: kv[1][0])
    prev_end = 0
    prev = None
    for name, (o, kind) in items:
        if kind not in _KIND_SIZE:
            errors.append("%s: bad kind %s" % (name, kind))
            continue
        sz = _KIND_SIZE[kind]
        if o < 0 or o + sz > SEGY_TRACE_HEADER_BYTES:
            errors.append("%s: out of 240-byte header (%d+%d)" % (name, o, sz))
        if o < prev_end:
            errors.append(
                "overlap: %s ends@%d vs %s@%d" % (prev, prev_end, name, o)
            )
        prev_end = o + sz
        prev = name

    # 3) 往返：每个字段写入可区分值，确认不被邻域覆盖
    # （延迟调用 pack/unpack，定义在后面时由 assert_segy_trace_fields_ok 再测）
    return (len(errors) == 0), errors


def _roundtrip_all_fields():
    """每个字段单独 round-trip，检测覆盖类 bug（如旧 counit@86 盖 gy）。"""
    errors = []
    for endian in ("little", "big"):
        for name, (_off, kind) in SEGY_TRACE_FIELDS.items():
            if kind == "u16":
                val = 12345
            elif kind == "i16":
                val = -1234 if name != "counit" else 1
            else:
                val = 2500000 + (hash(name) % 1000)
            th = {name: val, "ns": 10, "dt": 4000, "counit": 1}
            # 保证被测字段不被上面默认覆盖
            th[name] = val
            buf = pack_trace_header(th, endian=endian, verify=True)
            out = unpack_trace_header(buf, endian=endian)
            if int(out.get(name, 0)) != int(val):
                errors.append(
                    "roundtrip %s endian=%s: wrote %s got %s"
                    % (name, endian, val, out.get(name))
                )
        # 几何捆绑：sx/sy/gx/gy + counit 同时写入
        th = {
            "sx": 111111, "sy": 222222, "gx": 333333, "gy": 444444,
            "counit": 1, "scalco": -1, "ns": 100, "dt": 4000,
        }
        buf = pack_trace_header(th, endian=endian, verify=True)
        out = unpack_trace_header(buf, endian=endian)
        for k, v in th.items():
            if int(out.get(k, 0)) != int(v):
                errors.append(
                    "geom bundle endian=%s %s: wrote %s got %s"
                    % (endian, k, v, out.get(k))
                )
    return errors


def assert_segy_trace_fields_ok():
    """导入时 / 测试时调用：偏移或往返失败则抛 AssertionError。"""
    ok, errors = validate_segy_trace_fields()
    errors = list(errors) + _roundtrip_all_fields()
    if errors:
        raise AssertionError(
            "SEGY_TRACE_FIELDS validation failed:\n  - "
            + "\n  - ".join(errors)
        )


def unpack_trace_header(buf, endian="little", fields=None):
    """解析 240 字节道头 → dict[str, int]。"""
    if len(buf) < SEGY_TRACE_HEADER_BYTES:
        raise ValueError("trace header shorter than 240 bytes")
    prefix = ">" if endian == "big" else "<"
    want = fields or SEGY_TRACE_FIELDS.keys()
    out = {}
    for name in want:
        if name not in SEGY_TRACE_FIELDS:
            continue
        off, kind = SEGY_TRACE_FIELDS[name]
        out[name] = struct.unpack_from(prefix + _STRUCT_CODE[kind], buf, off)[0]
    return out


def _clip_field_val(kind, val):
    val = int(val)
    if kind == "i16":
        return max(-32768, min(32767, val))
    if kind == "u16":
        return max(0, min(65535, val))
    if kind == "i32":
        return max(-2147483648, min(2147483647, val))
    return val


# 按偏移排序的字段表（避免每次 pack 排序）
_FIELDS_BY_OFFSET = tuple(
    sorted(SEGY_TRACE_FIELDS.items(), key=lambda kv: kv[1][0])
)
_GEOM_LAST = ("scalco", "sx", "sy", "gx", "gy")


def pack_trace_header(trace_header, endian="big", base=None, *, verify=False):
    """
    dict → 240 字节道头。
    缺省字段填 0；endian 默认 big（SEGY 磁带惯例），SU-on-PC 读盘多用 little。
    按字节偏移写入；``sx/sy/gx/gy`` 最后再写一次，防止邻域字段误覆盖。
    base: 可选原始 240 字节，在其上补丁写入（保留未映射区）。
    verify: 为 True 时读回校验坐标（慢，仅调试用）。
    """
    if SEGY_TRACE_FIELDS.get("counit", (None,))[0] != 88:
        raise RuntimeError(
            "SEGY_TRACE_FIELDS counit offset is %s (expect 88); "
            "stale segy_trace_header module — please restart idata"
            % (SEGY_TRACE_FIELDS.get("counit"),)
        )
    prefix = ">" if endian == "big" else "<"
    if base is not None and len(base) >= SEGY_TRACE_HEADER_BYTES:
        buf = bytearray(base[:SEGY_TRACE_HEADER_BYTES])
    else:
        buf = bytearray(SEGY_TRACE_HEADER_BYTES)
    for name, (off, kind) in _FIELDS_BY_OFFSET:
        val = _clip_field_val(kind, trace_header.get(name, 0) or 0)
        struct.pack_into(prefix + _STRUCT_CODE[kind], buf, off, val)
    for name in _GEOM_LAST:
        off, kind = SEGY_TRACE_FIELDS[name]
        val = _clip_field_val(kind, trace_header.get(name, 0) or 0)
        struct.pack_into(prefix + _STRUCT_CODE[kind], buf, off, val)
    out = bytes(buf)
    if verify:
        check = unpack_trace_header(
            out, endian=endian, fields=("sx", "sy", "gx", "gy")
        )
        for name in ("sx", "sy", "gx", "gy"):
            want = _clip_field_val("i32", trace_header.get(name, 0) or 0)
            got = int(check.get(name, 0))
            if got != want:
                raise RuntimeError(
                    "pack_trace_header corrupted %s: wrote %s read back %s "
                    "(counit_off=%s endian=%s)"
                    % (name, want, got, SEGY_TRACE_FIELDS.get("counit"), endian)
                )
    return out


def scale_segy_factor(val, scale):
    """
    应用 scalco / scalel（segy.h）。

    道头里的正负号只指示运算方式，缩放因子本身始终取 |scale|：
      scale > 0 : val * scale
      scale < 0 : val / |scale|
      0 或 ±1   : 原样返回
    例：scalco=-100 → 米 = 整型坐标 / 100（不是 ×(-100)）。
    """
    s = int(scale)
    if s == 0 or abs(s) == 1:
        return float(val)
    if s > 0:
        return float(val) * float(s)
    return float(val) / float(-s)


def elev_to_depth_m(elev_m, depth_pos_m=None):
    """
    高程(海平面以上为正) → 深度(向下为正，米)。
    - 若提供恒正的 depth_pos_m（如 sdepth）且 >0，优先用它；
    - 否则 depth = -elev（海平面以下 elev 为负 → 深度为正）。
    """
    if depth_pos_m is not None and float(depth_pos_m) > 0:
        return float(depth_pos_m)
    return -float(elev_m)


def obs_xy_from_header(th):
    """本工区 OBS：检波点 XY = sx,sy（已 scalco，米）。"""
    return float(th["sx"]), float(th["sy"])


def shot_xy_from_header(th):
    """本工区 OBS：炮点 XY = gx,gy（已 scalco，米）。"""
    return float(th["gx"]), float(th["gy"])


def obs_depth_m_from_header(th):
    """本工区 OBS：检波点深度(m) ← selev（负高程）或 sdepth（正）。"""
    return elev_to_depth_m(th.get("selev", 0.0), th.get("sdepth"))


def shot_depth_m_from_header(th):
    """本工区 OBS：炮点深度(m) ← gelev（负高程）。"""
    return elev_to_depth_m(th.get("gelev", 0.0), None)


def offset_m_from_xy(sx_m, sy_m, gx_m, gy_m):
    """由平面坐标计算炮检水平距 (m)。本工区：OBS=(sx,sy)，炮=(gx,gy)。"""
    import math
    return math.hypot(float(sx_m) - float(gx_m), float(sy_m) - float(gy_m))


def resolve_offset_m(th, tol_m=1.0, warn=None, min_header_m=1.0):
    """
    确定 offset（米）——与 zplot 逻辑对齐：

      visualization/zplotpy/data_loader.py（读 .hdr 后）:
        if abs(offset_km) < 0.001 and UTM 有效:
            offset = hypot(rxutm-sxutm, ryutm-syutm)   # 即 |offset|<1 m 则用坐标算

      visualization/zplotpy/su2z_hhb.py / zplot/su/su2z_hhb.c:
        先写入道头 offset（米）；坐标 sx→sxutm、gx→rxutm（再由界面按变化量判谁是接收点）

    本函数规则:
      - |道头 offset| >= min_header_m（默认 1 m，同 zplot 的 0.001 km）→ 用道头值
        （保留符号，与 zplot 正负偏移一致；距离取 abs 时可自行 abs）
      - 否则 → hypot(sx-gx, sy-gy)（坐标须已 scalco，米）
      - 若采用道头且与坐标计算差 > tol_m → warn

    注意：segy.h 中 offset 不受 scalco；sx..gy 须已是米。
    返回 (offset_m, source)  source 为 'header' | 'xy'
    """
    sx = float(th["sx"])
    sy = float(th["sy"])
    gx = float(th["gx"])
    gy = float(th["gy"])
    computed = offset_m_from_xy(sx, sy, gx, gy)

    raw = th.get("offset", None)
    try:
        hdr = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        hdr = 0.0

    # 与 zplot data_loader：|offset_km|<0.001 → 重算  等价于 |offset_m|<1
    if abs(hdr) >= float(min_header_m):
        used = hdr  # 保留符号（负偏移距）
        src = "header"
        if warn is not None and computed > 0 and abs(abs(used) - computed) > float(tol_m):
            warn(
                "offset header=%.3f m vs xy=%.3f m (d=%.3f m)"
                % (used, computed, abs(abs(used) - computed))
            )
        return used, src

    # 坐标无效时仍退回头值
    if computed <= 0.0 and abs(hdr) > 0.0:
        return hdr, "header"
    return computed, "xy"


# 导入时自检：偏移重叠 / 与 segy.h 不一致 / pack 覆盖 会直接失败
assert_segy_trace_fields_ok()
