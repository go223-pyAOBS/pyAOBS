# -*- coding: utf-8 -*-
"""
SU → 按炮 RSF + 炮/OBS 坐标（接 RTM 流程）
============================================

默认（推荐）: 调用 Madagascar ``sfsuread`` 读 SU，再用道头分炮。
回退: ``--native`` 纯 Python 解析 240 字节道头（无 Madagascar 时）。

输出:
  shots/shot_000.rsf ...   n1=时间, n2=该炮 OBS 道
  shots_xz.txt             炮 x z (km)
  obs_xz.txt               OBS x z (km)  —— 取自首炮各道检波点
  su_summary.txt           道头统计
  _su_work/                sfsuread 中间文件（data.rsf / hdr.rsf）

位置: pyAOBS/visualization/obs_rtm_qt/scripts/（工区根无本脚本）
请在工区目录下执行（写出 shots/ 等到 cwd）。

用法:
  # 先看 trid 有哪些分量
  python path/to/scripts/su_to_shots.py --su line.su --list-trid

  # 声学 RTM：水听（推荐）
  python path/to/scripts/su_to_shots.py --native --su Rotated_TWI119_4C.su --component hydro

  # 或按 trid 数字
  python path/to/scripts/su_to_shots.py --native --su line.su --trid 11

  # 旋转后三分量分别导出
  python path/to/scripts/su_to_shots.py --native --su line.su --all-trid

TWI119 / 本工区 trid 约定（Rotated_*_4C.su）:
  11  水听 hydro          ← 声学 RTM 首选
  12  垂直（未旋转）
  13  水平1（未旋转）
  14  水平2（未旋转）
  15  垂直（旋转后） z
  16  与径向配对的水平分量（旋转后；记录名「垂向」, 按 transverse 用）
  17  径向（旋转后） radial

几何（详见工区 diag/obs_segy_geometry.txt）:
  --geom obs     炮=gx/gy/gelev；OBS=sx/sy/selev（本工区装填）
  --geom segy    SEGY 字面炮检
  --geom offset  sx/gx 不可用时：只用道头 offset + --obs-x 重建二维 x
                 shot_x = obs_x + sign*(offset_m/1000)  [km]
"""
from __future__ import print_function
import argparse
import glob
import os
import shutil
import subprocess
import sys

try:
    import numpy as np
except ImportError:
    np = None

# 道头偏移：与 processors/raw2sac/segy.h（SEGY 标准）共用 segy_trace_header.py
# 优先从 raw2sac 目录直接 import，避免触发 processors/__init__.py 重依赖
def _import_segy_trace_header():
    """定位 processors/raw2sac（兼容 modeling/ 与 obs_rtm_qt/ 下脚本路径）。"""
    _here = os.path.dirname(os.path.abspath(__file__))
    _raw2sac = None
    cur = _here
    for _ in range(8):
        cand = os.path.join(cur, "processors", "raw2sac")
        if os.path.isfile(os.path.join(cand, "segy_trace_header.py")):
            _raw2sac = cand
            break
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    if _raw2sac is None:
        raise ImportError(
            "找不到 processors/raw2sac/segy_trace_header.py（从 %s 向上搜）" % _here
        )
    if _raw2sac not in sys.path:
        sys.path.insert(0, _raw2sac)
    from segy_trace_header import (  # type: ignore
        SEGY_TRACE_HEADER_BYTES,
        MADAGASCAR_TFILE_KEY_ORDER,
        unpack_trace_header,
        scale_segy_factor,
        elev_to_depth_m,
        obs_xy_from_header,
        shot_xy_from_header,
        obs_depth_m_from_header,
        shot_depth_m_from_header,
        resolve_offset_m,
        offset_m_from_xy,
    )
    return dict(
        SEGY_TRACE_HEADER_BYTES=SEGY_TRACE_HEADER_BYTES,
        MADAGASCAR_TFILE_KEY_ORDER=MADAGASCAR_TFILE_KEY_ORDER,
        unpack_trace_header=unpack_trace_header,
        scale_segy_factor=scale_segy_factor,
        elev_to_depth_m=elev_to_depth_m,
        obs_xy_from_header=obs_xy_from_header,
        shot_xy_from_header=shot_xy_from_header,
        obs_depth_m_from_header=obs_depth_m_from_header,
        shot_depth_m_from_header=shot_depth_m_from_header,
        resolve_offset_m=resolve_offset_m,
        offset_m_from_xy=offset_m_from_xy,
    )


_segy = _import_segy_trace_header()
SEGY_TRACE_HEADER_BYTES = _segy["SEGY_TRACE_HEADER_BYTES"]
MADAGASCAR_TFILE_KEY_ORDER = _segy["MADAGASCAR_TFILE_KEY_ORDER"]
unpack_trace_header = _segy["unpack_trace_header"]
scale_segy_factor = _segy["scale_segy_factor"]
elev_to_depth_m = _segy["elev_to_depth_m"]
obs_xy_from_header = _segy["obs_xy_from_header"]
shot_xy_from_header = _segy["shot_xy_from_header"]
obs_depth_m_from_header = _segy["obs_depth_m_from_header"]
shot_depth_m_from_header = _segy["shot_depth_m_from_header"]
resolve_offset_m = _segy["resolve_offset_m"]
offset_m_from_xy = _segy["offset_m_from_xy"]

# 本工区 trid → (短名, 说明)
TRID_INFO = {
    11: ("hydro", "水听"),
    12: ("z_raw", "垂直(未旋转)"),
    13: ("h1_raw", "水平1(未旋转)"),
    14: ("h2_raw", "水平2(未旋转)"),
    15: ("z", "垂直(旋转后)"),
    16: ("trans", "横向/配对水平(旋转后; 记录称垂向)"),
    17: ("radial", "径向(旋转后)"),
}

# --component 别名 → trid
COMPONENT_ALIAS = {
    "hydro": 11, "p": 11, "water": 11, "h": 11, "水听": 11,
    "z_raw": 12, "vert_raw": 12, "垂直未旋": 12,
    "h1": 13, "h1_raw": 13,
    "h2": 14, "h2_raw": 14,
    "z": 15, "vert": 15, "vertical": 15, "垂直": 15,
    "trans": 16, "t": 16, "transverse": 16, "横向": 16, "垂向": 16,
    "radial": 17, "r": 17, "径向": 17,
}

# ---------------------------------------------------------------------------
# 可选: m8r
# ---------------------------------------------------------------------------
def _try_import_m8r():
    try:
        import m8r
        return m8r
    except ImportError:
        try:
            import rsf.api as m8r
            return m8r
        except ImportError:
            return None


# ---------------------------------------------------------------------------
# Madagascar 调用
# ---------------------------------------------------------------------------
def find_exe(*names):
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def endian_to_sf(endian):
    """SU on PC/Linux: little → endian=n；big → endian=y。"""
    if endian in ("little", "n", "0", False):
        return "n"
    return "y"


def parse_rsf_header(path):
    meta = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta


def resolve_rsf_binary(rsf_path, meta=None):
    """把 in= 解析成绝对路径（Madagascar 常写成相对 cwd 的 basename@）。"""
    meta = meta or parse_rsf_header(rsf_path)
    in_path = meta.get("in", "")
    if not in_path:
        raise RuntimeError("%s: missing in=" % rsf_path)
    if os.path.isabs(in_path) and os.path.isfile(in_path):
        return in_path
    # 1) 相对 rsf 头文件所在目录
    cand = os.path.join(os.path.dirname(os.path.abspath(rsf_path)), os.path.basename(in_path))
    if os.path.isfile(cand):
        return cand
    # 2) 相对 cwd
    cand2 = os.path.abspath(in_path)
    if os.path.isfile(cand2):
        return cand2
    # 3) 约定 rsf@ 与头文件同名
    cand3 = os.path.abspath(rsf_path) + "@"
    if os.path.isfile(cand3):
        return cand3
    raise RuntimeError(
        "%s: cannot find binary in=%s (tried %s)" % (rsf_path, in_path, cand)
    )


def fix_rsf_in_abspath(rsf_path):
    """重写头文件 in= 为绝对路径，避免后续 sf* 找不到数据体。已正确则跳过写盘。"""
    meta = parse_rsf_header(rsf_path)
    bin_path = resolve_rsf_binary(rsf_path, meta)
    cur = str(meta.get("in", "") or "").strip().strip("\"'")
    try:
        if cur and os.path.normpath(os.path.abspath(cur)) == os.path.normpath(
            os.path.abspath(bin_path)
        ):
            return bin_path
    except Exception:
        pass
    lines = []
    with open(rsf_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("in="):
                lines.append("in=%s\n" % bin_path)
            else:
                lines.append(line)
    with open(rsf_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return bin_path


def run_suread(su_path, data_rsf, hdr_rsf, endian="little"):
    """sfsuread < su tfile=hdr endian= > data  （suread ≡ segyread su=y）"""
    suread = find_exe("sfsuread", "sfsegyread")
    if not suread:
        raise RuntimeError(
            "未找到 sfsuread/sfsegyread。请 source $RSFROOT/share/madagascar/etc/env.sh\n"
            "或改用: python .../obs_rtm_qt/scripts/su_to_shots.py --native --su ..."
        )
    su_path = os.path.abspath(su_path)
    data_rsf = os.path.abspath(data_rsf)
    hdr_rsf = os.path.abspath(hdr_rsf)
    os.makedirs(os.path.dirname(hdr_rsf) or ".", exist_ok=True)

    end = endian_to_sf(endian)
    su_flag = "su=y" if "segyread" in os.path.basename(suread) else ""
    # datapath= 让二进制落在 hdr 同目录，减少 in= 错位
    datapath = os.path.dirname(hdr_rsf) + os.sep
    sh = (
        '%s tfile="%s" endian=%s datapath="%s" %s < "%s" > "%s"'
        % (suread, hdr_rsf, end, datapath, su_flag, su_path, data_rsf)
    )
    print("RUN:", sh)
    subprocess.check_call(sh, shell=True)
    fix_rsf_in_abspath(hdr_rsf)
    fix_rsf_in_abspath(data_rsf)


def sf_header_key(hdr_rsf, key, out_rsf):
    """
    从 sfsuread 的 tfile 抽出单个键 → 1D float RSF。
    正确做法（不是 sfheaderwindow）:
      < tfile.rsf sfdd type=float | sfheadermath output=trid > out.rsf
    """
    dd = find_exe("sfdd")
    hm = find_exe("sfheadermath")
    if not dd or not hm:
        raise RuntimeError("需要 sfdd 与 sfheadermath（Madagascar bin）")
    fix_rsf_in_abspath(hdr_rsf)
    out_rsf = os.path.abspath(out_rsf)
    sh = (
        'sfdd < "%s" type=float | sfheadermath output="%s" > "%s"'
        % (os.path.abspath(hdr_rsf), key, out_rsf)
    )
    subprocess.check_call(sh, shell=True)
    fix_rsf_in_abspath(out_rsf)


def read_rsf_float1d(path):
    """读 1D（或可摊平的）float RSF。"""
    meta = parse_rsf_header(path)
    n1 = int(meta.get("n1", 1))
    n2 = int(meta.get("n2", 1))
    n3 = int(meta.get("n3", 1))
    ntot = n1 * n2 * n3
    in_path = resolve_rsf_binary(path, meta)
    data = np.fromfile(in_path, dtype=np.float32, count=ntot)
    if data.size != ntot:
        raise RuntimeError("short read %s: got %d want %d" % (path, data.size, ntot))
    return np.ravel(data), meta


def read_rsf_float2d_traces(path):
    """读 data.rsf：n1=时间, n2=道 → (n1, n2) float32。"""
    fix_rsf_in_abspath(path)
    m8r = _try_import_m8r()
    if m8r is not None:
        try:
            inp = m8r.Input(path)
            n1 = int(inp.int("n1"))
            n2 = int(inp.int("n2"))
            d1 = float(inp.float("d1"))
            buf = np.zeros(n1 * n2, dtype=np.float32)
            inp.read(buf)
            return buf.reshape((n2, n1)).T.copy(), d1
        except Exception:
            pass

    meta = parse_rsf_header(path)
    n1, n2 = int(meta["n1"]), int(meta["n2"])
    d1 = float(meta.get("d1", "0.001"))
    in_path = resolve_rsf_binary(path, meta)
    raw = np.fromfile(in_path, dtype=np.float32, count=n1 * n2)
    return raw.reshape((n2, n1)).T.copy(), d1


def load_keys_from_hdr(hdr_rsf, keys, workdir):
    """从 sfsuread 的 tfile 抽多个键。"""
    out = {}
    for key in keys:
        tmp = os.path.join(workdir, "_key_%s.rsf" % key)
        sf_header_key(hdr_rsf, key, tmp)
        arr, _ = read_rsf_float1d(tmp)
        out[key] = arr
    return out


def read_tfile_keys_numpy(hdr_rsf, key_names):
    """直接读 tfile 整型数组；键序见 MADAGASCAR_TFILE_KEY_ORDER（同 sfsegyread）。"""
    order = MADAGASCAR_TFILE_KEY_ORDER
    fix_rsf_in_abspath(hdr_rsf)
    meta = parse_rsf_header(hdr_rsf)
    n1, n2 = int(meta["n1"]), int(meta["n2"])
    raw = np.fromfile(resolve_rsf_binary(hdr_rsf, meta), dtype=np.int32, count=n1 * n2)
    if raw.size != n1 * n2:
        raise RuntimeError("tfile short read")
    arr = raw.reshape((n2, n1)).T  # (nkeys, ntraces)
    out = {}
    for kn in key_names:
        if kn not in order:
            continue
        ik = order.index(kn)
        if ik >= n1:
            continue
        out[kn] = arr[ik, :].astype(np.float64)
    return out


def scale_coord(val, scalco):
    return scale_segy_factor(val, scalco)


def scale_elev(val, scalel):
    return scale_segy_factor(val, scalel)


def m_to_km(x_m, xy_unit):
    """xy_unit=m：输入已是米；=km：输入已是千米。"""
    if xy_unit == "km":
        return float(x_m)
    if xy_unit == "m":
        return float(x_m) * 0.001
    raise ValueError("xy_unit must be m or km")


def _trace_z_km(th, args, zobs_m, zshot_m):
    """应用 --zobs-mode / --zshot 覆盖后返回 (zshot_km, zobs_km)。"""
    if args.zobs_mode == "const":
        zobs_m = float(args.zobs_const) * 1000.0
    elif args.zobs_mode == "sdepth":
        zobs_m = abs(float(th.get("sdepth", 0.0)))
    elif args.zobs_mode == "selev":
        zobs_m = elev_to_depth_m(th.get("selev", 0.0), None)
    elif args.zobs_mode == "gelev":
        if args.geom == "segy":
            zobs_m = elev_to_depth_m(th.get("gelev", 0.0), None)
    elif args.zobs_mode == "gwdep":
        z = abs(float(th.get("gwdep", 0.0)))
        if z > 0:
            zobs_m = z

    if args.zshot is not None and float(args.zshot) >= 0:
        zshot_km = float(args.zshot)
    else:
        zshot_km = m_to_km(zshot_m, "m")
    return zshot_km, m_to_km(zobs_m, "m")


def trace_shot_obs_xz_km(th, args):
    """
    得到 (shot_x, shot_z, obs_x, obs_z)，单位 km。

    --geom obs:    炮=gx/gy/gelev；OBS=sx/sy/selev/sdepth
    --geom segy:   SEGY 字面
    --geom offset: 不信任 sx/gx；用道头 offset + --obs-x
                   shot_x = obs_x + offset_sign * (offset_m/1000)
    """
    if args.geom == "offset":
        # 只用道头 offset（米，可带符号）；xy 不可靠时勿走 resolve 的 xy 回退
        try:
            off_m = float(th.get("offset", 0.0) or 0.0)
        except (TypeError, ValueError):
            off_m = 0.0
        if abs(off_m) < float(getattr(args, "offset_min_header", 1.0)):
            raise RuntimeError(
                "--geom offset 需要可靠道头 offset，当前 |offset|=%g m" % off_m
            )
        obs_x_km = float(args.obs_x)
        sign = float(args.offset_sign)
        shot_x_km = obs_x_km + sign * (off_m * 0.001)
        # z 仍可从头或 --zshot / --zobs-mode const
        zobs_m = obs_depth_m_from_header(th)
        zshot_m = shot_depth_m_from_header(th)
        zshot_km, zobs_km = _trace_z_km(th, args, zobs_m, zshot_m)
        return shot_x_km, zshot_km, obs_x_km, zobs_km

    if args.geom == "obs":
        ox_m, oy_m = obs_xy_from_header(th)
        sx_m, sy_m = shot_xy_from_header(th)
        zobs_m = obs_depth_m_from_header(th)
        zshot_m = shot_depth_m_from_header(th)
    else:
        sx_m, sy_m = float(th["sx"]), float(th["sy"])
        ox_m, oy_m = float(th["gx"]), float(th["gy"])
        zshot_m = elev_to_depth_m(th.get("selev", 0.0), th.get("sdepth"))
        zobs_m = elev_to_depth_m(th.get("gelev", 0.0), th.get("gwdep"))

    if args.line_axis == "x":
        shot_x_m, obs_x_m = sx_m, ox_m
    else:
        shot_x_m, obs_x_m = sy_m, oy_m

    zshot_km, zobs_km = _trace_z_km(th, args, zobs_m, zshot_m)
    return (
        m_to_km(shot_x_m, args.xy_unit),
        zshot_km,
        m_to_km(obs_x_m, args.xy_unit),
        zobs_km,
    )


def write_rsf_2d(path, data, d1, o1=0.0, label1="Time", unit1="s",
                 label2="OBS", unit2=""):
    if np is None:
        raise RuntimeError("需要 numpy")
    data = np.asarray(data, dtype=np.float32)
    n1, n2 = data.shape
    path = os.path.abspath(path)
    bin_path = path + "@"
    with open(bin_path, "wb") as b:
        for j in range(n2):
            b.write(np.ascontiguousarray(data[:, j]).tobytes())
    with open(path, "w", encoding="utf-8") as h:
        # 必须写绝对路径：in=shot_000.rsf@ 时，scons 在工区根跑 sfwindow
        # 会到 cwd 找二进制，而不是 shots/ 目录 → No such file
        h.write("in=%s\n" % bin_path)
        h.write("n1=%d\nd1=%g\no1=%g\n" % (n1, d1, o1))
        h.write("label1=%s\nunit1=%s\n" % (label1, unit1))
        h.write("n2=%d\nd2=1\no2=0\n" % n2)
        h.write("label2=%s\nunit2=%s\n" % (label2, unit2))
        h.write("data_format=native_float\nesize=4\n")


def fix_shot_rsf_headers(paths, log=None):
    """只把给定 shot_*.rsf 的 in= 改成绝对路径（作业炮子集）。"""
    paths = [p for p in (paths or []) if p and os.path.isfile(p)]
    if not paths:
        return 0
    if log:
        log("检查作业炮 RSF 头 in=（%d 个）…" % len(paths))
    n = 0
    for i, path in enumerate(paths, 1):
        name = os.path.basename(path)
        try:
            meta = parse_rsf_header(path)
            bin_path = resolve_rsf_binary(path, meta)
            cur = str(meta.get("in", "") or "").strip().strip("\"'")
            already = False
            try:
                already = bool(cur) and os.path.normpath(
                    os.path.abspath(cur)
                ) == os.path.normpath(os.path.abspath(bin_path))
            except Exception:
                already = False
            if not already:
                fix_rsf_in_abspath(path)
                n += 1
        except Exception as exc:
            if log:
                log("fix in= skip %s: %s" % (name, exc))
        if log and len(paths) > 20 and (i == 1 or i == len(paths) or i % 50 == 0):
            log("  in= 检查进度 %d / %d" % (i, len(paths)))
    if log:
        if n:
            log("已改写 %d / %d 个作业炮的 in= 为绝对路径" % (n, len(paths)))
        else:
            log("作业炮 in= 均已是绝对路径，无需改写")
    return n


def fix_shot_rsf_headers_in_dir(shot_dir, log=None, shot_indices=None):
    """
    批量修 in=。
    ``shot_indices`` 给定时只处理这些炮；否则扫整个目录（一般勿用）。
    """
    if not shot_dir or not os.path.isdir(shot_dir):
        return 0
    if shot_indices is not None:
        paths = []
        for i in shot_indices:
            p = os.path.join(shot_dir, "shot_%03d.rsf" % int(i))
            if os.path.isfile(p):
                paths.append(p)
        return fix_shot_rsf_headers(paths, log=log)
    names = [
        name
        for name in sorted(os.listdir(shot_dir))
        if name.startswith("shot_")
        and name.endswith(".rsf")
        and not name.endswith(".rsf@")
    ]
    paths = [os.path.join(shot_dir, name) for name in names]
    return fix_shot_rsf_headers(paths, log=log)


# ---------------------------------------------------------------------------
# 原生 SU 读头（--native）：偏移与 segy.h / segy_trace_header.py 一致
# ---------------------------------------------------------------------------
def read_su_traces_native(path, endian="little"):
    import struct
    with open(path, "rb") as f:
        while True:
            hdr = f.read(SEGY_TRACE_HEADER_BYTES)
            if not hdr or len(hdr) < SEGY_TRACE_HEADER_BYTES:
                break
            th = unpack_trace_header(hdr, endian=endian)
            ns = int(th["ns"])
            if ns <= 0 or ns > 200000:
                raise RuntimeError("%s: bad ns=%d (try --endian)" % (path, ns))
            dt_us = int(th["dt"])
            raw = f.read(ns * 4)
            if len(raw) < ns * 4:
                break
            fmt = (">" if endian == "big" else "<") + "%df" % ns
            samps = np.asarray(struct.unpack(fmt, raw), dtype=np.float32)
            scalco = int(th["scalco"])
            scalel = int(th["scalel"])
            tr = dict(
                fldr=int(th["fldr"]),
                ep=int(th["ep"]),
                tracf=int(th["tracf"]),
                tracl=int(th["tracl"]),
                trid=int(th["trid"]),
                # 下列已乘/除 scalco、scalel，单位米（本工区）
                sx=scale_coord(th["sx"], scalco),
                sy=scale_coord(th["sy"], scalco),
                gx=scale_coord(th["gx"], scalco),
                gy=scale_coord(th["gy"], scalco),
                gelev=scale_elev(th["gelev"], scalel),
                selev=scale_elev(th["selev"], scalel),
                sdepth=scale_elev(th["sdepth"], scalel),
                gdel=scale_elev(th.get("gdel", 0), scalel),
                sdel=scale_elev(th.get("sdel", 0), scalel),
                swdep=scale_elev(th.get("swdep", 0), scalel),
                gwdep=scale_elev(th.get("gwdep", 0), scalel),
                # offset 不受 scalco；保留道头原值，稍后 resolve
                offset=float(th.get("offset", 0) or 0),
                scalco=scalco,
                scalel=scalel,
                ns=ns,
                dt=dt_us * 1e-6 if dt_us > 0 else None,
                data=samps,
            )
            # offset_m 在 emit_outputs 里统一 resolve（便于 tol/告警）
            yield tr


def trid_histogram(traces):
    """traces: iterable of dict with 'trid'."""
    hist = {}
    for t in traces:
        tid = int(t.get("trid", -999))
        hist[tid] = hist.get(tid, 0) + 1
    return hist


def print_trid_hist(hist):
    print("trid  count   name     说明")
    for tid in sorted(hist.keys()):
        name, desc = TRID_INFO.get(int(tid), ("?", "?"))
        print("%4d  %-8d %-8s %s" % (tid, hist[tid], name, desc))
    if len(hist) > 1:
        print("筛选: --component hydro|z|radial|trans  或  --trid N  或  --all-trid")
        print("声学 RTM 推荐: --component hydro  (trid=11 水听)")


def resolve_component_to_trid(component, trid_str):
    """--component 与 --trid 合并；component 优先补全 trid。"""
    if component:
        key = component.strip().lower()
        # 允许直接写中文别名（不 lower 中文）
        if component.strip() in COMPONENT_ALIAS:
            return str(COMPONENT_ALIAS[component.strip()])
        if key in COMPONENT_ALIAS:
            return str(COMPONENT_ALIAS[key])
        raise RuntimeError(
            "未知 --component=%s。可选: %s"
            % (component, ", ".join(sorted(set(COMPONENT_ALIAS.keys()))))
        )
    return trid_str


def parse_trid_list(s):
    """'1,2,3' 或单值 → list[int]；空 → None。"""
    if s is None or s == "":
        return None
    out = []
    for part in str(s).replace(" ", "").split(","):
        if part:
            out.append(int(part))
    return out or None


def filter_by_trid(traces, trid_keep):
    if not trid_keep:
        return list(traces)
    keep = set(int(x) for x in trid_keep)
    return [t for t in traces if int(t.get("trid", -999)) in keep]


def emit_outputs(order_keys, groups, args, get_trace_fields, outdir=None,
                 shots_txt=None, obs_txt=None, summary_txt=None):
    """
    groups[key] = list of trace dict
    """
    outdir = outdir or args.outdir
    shots_txt = shots_txt or getattr(args, "shots_xz", None) or "shots_xz.txt"
    obs_txt = obs_txt or getattr(args, "obs_xz", None) or "obs_xz.txt"
    summary_txt = summary_txt or getattr(args, "summary", None) or "su_summary.txt"
    for p in (shots_txt, obs_txt, summary_txt):
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)
    shots_xyz = []
    obs_ref = None
    dt_use = args.dt
    summary = []
    trid_note = getattr(args, "_trid_active", None)
    offset_rows = []  # (ishot, iobs, offset_m, offset_src, offset_xy_m)
    n_hdr_off = 0
    n_xy_off = 0
    n_off_mismatch = 0

    def _off_warn(msg):
        nonlocal n_off_mismatch
        n_off_mismatch += 1
        if n_off_mismatch <= 8:
            print("WARN offset: %s" % msg, file=sys.stderr)

    for ishot, k in enumerate(order_keys):
        trs = [get_trace_fields(t) for t in groups[k]]
        # 排序：offset 模式按 offset；obs 按 sx；segy 按 gx
        if args.geom == "offset":
            trs.sort(key=lambda t: (t["tracf"], float(t.get("offset", 0) or 0), t["tracl"]))
        elif args.geom == "obs":
            trs.sort(key=lambda t: (t["tracf"], t["sx"], t["sy"], t["tracl"]))
        else:
            trs.sort(key=lambda t: (t["tracf"], t["gx"], t["gy"], t["tracl"]))
        if not trs:
            summary.append("WARN shot %s: empty after trid filter" % k)
            continue
        ns = trs[0]["ns"]
        if not dt_use:
            dt_use = trs[0]["dt"] or 0.001
        nobs = len(trs)
        gather = np.zeros((ns, nobs), dtype=np.float32)
        for j, t in enumerate(trs):
            gather[:, j] = t["data"]

        shot_x, shot_z, _, _ = trace_shot_obs_xz_km(trs[0], args)
        shots_xyz.append((shot_x, shot_z))

        obs_pts = []
        for j, t in enumerate(trs):
            _, _, ox, oz = trace_shot_obs_xz_km(t, args)
            obs_pts.append((ox, oz))
            if args.geom == "offset":
                om = float(t.get("offset", 0.0) or 0.0)
                osrc = "header"
                xy = float("nan")  # xy 不可靠，不对比
                n_hdr_off += 1
            else:
                om, osrc = resolve_offset_m(
                    t,
                    tol_m=args.offset_tol,
                    warn=_off_warn,
                    min_header_m=args.offset_min_header,
                )
                if osrc == "header":
                    n_hdr_off += 1
                else:
                    n_xy_off += 1
                xy = offset_m_from_xy(t["sx"], t["sy"], t["gx"], t["gy"])
            t["offset_m"], t["offset_src"] = om, osrc
            offset_rows.append((ishot, j, om, osrc, xy))

        if obs_ref is None:
            obs_ref = obs_pts
        elif len(obs_pts) != len(obs_ref):
            summary.append("WARN shot %s: nobs=%d != %d" % (k, len(obs_pts), len(obs_ref)))

        out = os.path.join(outdir, "shot_%03d.rsf" % ishot)
        write_rsf_2d(out, gather, d1=dt_use)
        try:
            fix_rsf_in_abspath(out)
        except Exception:
            pass
        tid0 = trs[0].get("trid", "?")
        off0 = trs[0].get("offset_m", float("nan"))
        summary.append(
            "shot_%03d key=%s nobs=%d shot_x=%.3f km shot_z=%.4f km "
            "offset0=%.1f m dt=%g trid=%s geom=%s"
            % (ishot, k, nobs, shot_x, shot_z, off0, dt_use, tid0, args.geom)
        )
        print(summary[-1])

    with open(shots_txt, "w", encoding="utf-8") as f:
        if args.geom == "offset":
            f.write(
                "# x_km z_km  shots from offset: "
                "x=obs_x%+g*(offset_m/1000); obs_x=%g\n"
                % (float(args.offset_sign), float(args.obs_x))
            )
        elif args.geom == "obs":
            f.write("# x_km z_km  shots from gx,gy + gelev (OBS header convention)\n")
        else:
            f.write("# x_km z_km  shots from sx,sy + selev/sdepth (SEGY literal)\n")
        for x, z in shots_xyz:
            f.write("%g %g\n" % (x, z))
    if obs_ref:
        with open(obs_txt, "w", encoding="utf-8") as f:
            if args.geom == "offset":
                f.write("# x_km z_km  OBS = --obs-x / z from header or --zobs-*\n")
            elif args.geom == "obs":
                f.write("# x_km z_km  OBS from sx,sy + selev/sdepth\n")
            else:
                f.write("# x_km z_km  receivers from gx,gy + gelev\n")
            for x, z in obs_ref:
                f.write("%g %g\n" % (x, z))
    # offset 表：与 shots_xz 同目录（分层后在 prep/geom/）
    off_txt = getattr(args, "offsets_txt", None) or ""
    if not off_txt:
        off_txt = os.path.join(os.path.dirname(shots_txt) or ".", "offsets.txt")
    od = os.path.dirname(off_txt)
    if od:
        os.makedirs(od, exist_ok=True)
    with open(off_txt, "w", encoding="utf-8") as f:
        f.write("# ishot iobs offset_m source offset_xy_m\n")
        f.write("# 规则同 zplot data_loader: |hdr|>=1m → header(可带符号); 否则 xy\n")
        f.write("# source=header|xy; offset_xy_m=hypot(sx-gx,sy-gy)\n")
        for row in offset_rows:
            f.write("%d %d %.6g %s %.6g\n" % row)

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(
            "backend=%s endian=%s group=%s geom=%s xy_unit=%s nshot=%d dt=%g trid=%s\n"
            % (
                "native" if args.native else "sfsuread",
                args.endian, args.group, args.geom, args.xy_unit,
                len(shots_xyz), dt_use, trid_note,
            )
        )
        f.write(
            "offset: header=%d xy_computed=%d mismatch_warn=%d tol_m=%g\n"
            % (n_hdr_off, n_xy_off, n_off_mismatch, args.offset_tol)
        )
        f.write("\n".join(summary) + "\n")

    print("wrote %d gathers -> %s/" % (len(shots_xyz), outdir))
    print("wrote %s, %s, %s, %s" % (shots_txt, obs_txt, summary_txt, off_txt))
    print(
        "offset: 道头=%d  坐标计算=%d  差异告警=%d (tol=%.1f m)"
        % (n_hdr_off, n_xy_off, n_off_mismatch, args.offset_tol)
    )


def regroup(traces, args):
    """按 group 键分炮；可选 trid 过滤。返回 order_keys, groups, hist。"""
    hist = trid_histogram(traces)
    trid_keep = parse_trid_list(args.trid)
    if trid_keep:
        traces = filter_by_trid(traces, trid_keep)
        if not traces:
            raise RuntimeError(
                "trid 过滤后无道。现有 trid: %s" % sorted(hist.keys())
            )
    groups = {}
    order_keys = []
    for tr in traces:
        if args.group == "file":
            k = tr.get("_file", "file")
        else:
            k = tr[args.group]
        if k not in groups:
            groups[k] = []
            order_keys.append(k)
        groups[k].append(tr)
    return order_keys, groups, hist


def collect_traces_suread(files, args):
    workdir = args.workdir
    os.makedirs(workdir, exist_ok=True)
    traces = []

    for fp in files:
        tag = os.path.splitext(os.path.basename(fp))[0]
        data_rsf = os.path.join(workdir, tag + "_data.rsf")
        hdr_rsf = os.path.join(workdir, tag + "_hdr.rsf")
        run_suread(fp, data_rsf, hdr_rsf, endian=args.endian)

        key_names = [
            "fldr", "ep", "tracf", "tracl", "trid",
            "sx", "sy", "gx", "gy",
            "scalco", "scalel",
            "gelev", "selev", "sdepth", "gdel", "sdel", "swdep", "gwdep",
            "offset", "dt", "ns",
        ]
        data, d1 = read_rsf_float2d_traces(data_rsf)
        n1, n2 = data.shape
        if args.dt:
            d1 = args.dt

        # tfile 整型键索引更稳；headermath 在 n1=nkeys 时容易抽成长度 1
        keys = read_tfile_keys_numpy(hdr_rsf, key_names)
        if "trid" not in keys or len(keys.get("trid", [])) != n2:
            try:
                keys = load_keys_from_hdr(hdr_rsf, key_names, workdir)
            except Exception as e:
                raise RuntimeError(
                    "道头读取失败 (%s)。请: sfheaderattr < %s 或 --native"
                    % (e, hdr_rsf)
                )

        for kn, arr in list(keys.items()):
            arr = np.ravel(arr)
            if arr.size == 1 and n2 > 1:
                keys[kn] = np.repeat(arr, n2)
            elif arr.size != n2:
                raise RuntimeError(
                    "key %s length %d != ntraces %d" % (kn, arr.size, n2)
                )
            else:
                keys[kn] = arr

        if "trid" not in keys:
            raise RuntimeError(
                "道头无 trid。请: sfheaderattr < %s 或 --native" % hdr_rsf
            )

        for i in range(n2):
            scalco = float(keys["scalco"][i]) if "scalco" in keys else 1.0
            scalel = float(keys["scalel"][i]) if "scalel" in keys else 1.0
            dt_hdr = float(keys["dt"][i]) if "dt" in keys else 0.0
            if dt_hdr > 1.0:
                dt_s = dt_hdr * 1e-6
            else:
                dt_s = d1

            def _k(name, default=0.0):
                return float(keys[name][i]) if name in keys else default

            traces.append(dict(
                fldr=int(_k("fldr")),
                ep=int(_k("ep")),
                tracf=int(_k("tracf", i)),
                tracl=int(_k("tracl", i)),
                trid=int(_k("trid")),
                sx=scale_coord(_k("sx"), scalco),
                sy=scale_coord(_k("sy"), scalco),
                gx=scale_coord(_k("gx"), scalco),
                gy=scale_coord(_k("gy"), scalco),
                gelev=scale_elev(_k("gelev"), scalel),
                selev=scale_elev(_k("selev"), scalel),
                sdepth=scale_elev(_k("sdepth"), scalel),
                gdel=scale_elev(_k("gdel"), scalel),
                sdel=scale_elev(_k("sdel"), scalel),
                swdep=scale_elev(_k("swdep"), scalel),
                gwdep=scale_elev(_k("gwdep"), scalel),
                offset=_k("offset"),  # 道头原值(m)，不经 scalco
                scalco=scalco,
                scalel=scalel,
                ns=n1,
                dt=dt_s,
                data=data[:, i].copy(),
                _file=tag,
            ))
    return traces


def collect_traces_native(files, args, trid_keep=None):
    """
    读 SU。
    trid_keep is None：保留全部分量波形（很占内存，仅 --all-trid 需要）。
    trid_keep 为 list/set（可为空）：只保留匹配 trid 的波形；空集=只统计 hist、不存波形。
    直方图始终统计全部道的 trid。
    """
    filter_on = trid_keep is not None
    keep = set(int(x) for x in trid_keep) if filter_on else None
    traces = []
    hist = {}
    n_skip = 0
    for fp in files:
        tag = os.path.splitext(os.path.basename(fp))[0]
        for tr in read_su_traces_native(fp, endian=args.endian):
            tid = int(tr.get("trid", -999))
            hist[tid] = hist.get(tid, 0) + 1
            if filter_on and tid not in keep:
                n_skip += 1
                continue
            tr["_file"] = tag
            traces.append(tr)
    if filter_on:
        print(
            "native: kept %d traces (trid in %s), skipped %d"
            % (len(traces), sorted(keep), n_skip)
        )
    return traces, hist


def process_traces(traces, args, hist=None):
    if hist is None:
        hist = trid_histogram(traces)
    print_trid_hist(hist)

    if args.list_trid:
        trid_out = getattr(args, "trid_list", None) or "trid_list.txt"
        td = os.path.dirname(trid_out)
        if td:
            os.makedirs(td, exist_ok=True)
        with open(trid_out, "w", encoding="utf-8") as f:
            for tid in sorted(hist.keys()):
                f.write("%d %d\n" % (tid, hist[tid]))
        print("wrote %s" % trid_out)
        return

    # 多分量且未指定：拒绝混在一起，避免 RTM 把 4C 当一道
    trid_keep = parse_trid_list(args.trid)
    if args.all_trid:
        for tid in sorted(hist.keys()):
            args.trid = str(tid)
            args._trid_active = tid
            order_keys, groups, _ = regroup(traces, args)
            if args.max_shot > 0:
                order_keys = order_keys[: args.max_shot]
            outdir = "%s_trid%d" % (args.outdir.rstrip("/\\"), tid)
            geom_dir = os.path.dirname(
                getattr(args, "shots_xz", None) or "shots_xz.txt"
            ) or "."
            emit_outputs(
                order_keys, groups, args, lambda t: t,
                outdir=outdir,
                shots_txt=os.path.join(geom_dir, "shots_xz_trid%d.txt" % tid),
                obs_txt=os.path.join(geom_dir, "obs_xz_trid%d.txt" % tid),
                summary_txt=os.path.join(
                    os.path.dirname(getattr(args, "summary", None) or ".") or ".",
                    "su_summary_trid%d.txt" % tid,
                ),
            )
        print("声学 RTM 请用: shots_trid11/ （水听）或 --component hydro")
        return

    if trid_keep is None and len(hist) > 1:
        raise RuntimeError(
            "检测到多个 trid=%s。请指定分量，例如:\n"
            "  python .../obs_rtm_qt/scripts/su_to_shots.py --native --su ... --component hydro\n"
            "  python .../obs_rtm_qt/scripts/su_to_shots.py --native --su ... --trid 11\n"
            "  python .../obs_rtm_qt/scripts/su_to_shots.py --native --su ... --all-trid"
            % (sorted(hist.keys()),)
        )

    args._trid_active = trid_keep if trid_keep else sorted(hist.keys())
    order_keys, groups, _ = regroup(traces, args)
    if args.max_shot > 0:
        order_keys = order_keys[: args.max_shot]
    emit_outputs(order_keys, groups, args, lambda t: t)


def main():
    ap = argparse.ArgumentParser(
        description="SU → per-shot RSF (default: sfsuread; filter by trid)")
    ap.add_argument("--su", default="", help="单个 .su")
    ap.add_argument("--su-dir", default="", help="目录内多个 .su")
    ap.add_argument("--endian", choices=["little", "big"], default="little",
                    help="传给 sfsuread：little→endian=n，big→endian=y")
    ap.add_argument("--group", choices=["fldr", "ep", "gx", "sx", "file"], default="fldr",
                    help="分炮键；OBS 数据用 fldr（或 gx=炮点），勿用 sx（sx 是 OBS）")
    ap.add_argument("--trid", default="",
                    help="只保留该分量，如 11（水听）或 15,17")
    ap.add_argument("--component", default="",
                    help="分量别名: hydro|z|radial|trans|z_raw|h1|h2（见文件头 TRID_INFO）")
    ap.add_argument("--list-trid", action="store_true",
                    help="只统计 trid 分布后退出")
    ap.add_argument("--all-trid", action="store_true",
                    help="每个 trid 分别导出到 shots_tridN/")
    ap.add_argument("--geom", choices=["obs", "segy", "offset"], default="obs",
                    help="obs/segy 用坐标；offset=仅用道头 offset+--obs-x（sx/gx 坏时）")
    ap.add_argument("--obs-x", type=float, default=0.0,
                    help="--geom offset：OBS 在测线上的 x (km)，须与速度模型同坐标系")
    ap.add_argument("--offset-sign", type=float, default=1.0,
                    help="--geom offset：shot_x=obs_x+sign*(offset_m/1000)；取 +1 或 -1")
    ap.add_argument("--xy-unit", choices=["m", "km"], default="m",
                    help="道头经 scalco 后的平面坐标单位（本工区为 m；offset 模式不用）")
    ap.add_argument("--line-axis", choices=["x", "y"], default="x")
    ap.add_argument("--zshot", type=float, default=-1.0,
                    help="炮深 km；默认 -1 表示从道头 gelev(obs)/selev(segy) 读")
    ap.add_argument("--zobs-mode",
                    choices=["auto", "selev", "sdepth", "gelev", "const", "gwdep"],
                    default="auto",
                    help="OBS 深度：auto 按 --geom；本工区常用 selev/sdepth")
    ap.add_argument("--zobs-const", type=float, default=2.5,
                    help="--zobs-mode const 时的 OBS 深度 (km)")
    ap.add_argument("--offset-tol", type=float, default=1.0,
                    help="道头 |offset| 与 xy 计算差超过该值(m)则告警（offset 模式跳过）")
    ap.add_argument("--offset-min-header", type=float, default=1.0,
                    help="|道头 offset|>=该值(m)才采信头（默认1m，同 zplot 0.001km）")
    ap.add_argument("--outdir", default="inputs/shots")
    ap.add_argument(
        "--shots-xz",
        dest="shots_xz",
        default="prep/geom/shots_xz.txt",
    )
    ap.add_argument(
        "--obs-xz",
        dest="obs_xz",
        default="prep/geom/obs_xz.txt",
    )
    ap.add_argument(
        "--summary",
        default="diag/su_summary.txt",
        help="道头摘要（diag/）",
    )
    ap.add_argument(
        "--offsets",
        dest="offsets_txt",
        default="prep/geom/offsets.txt",
    )
    ap.add_argument(
        "--trid-list",
        default="diag/trid_list.txt",
        help="--list-trid 输出路径",
    )
    ap.add_argument(
        "--workdir",
        default="cache/_su_work",
        help="sfsuread 中间目录",
    )
    ap.add_argument("--dt", type=float, default=0.0, help="覆盖采样率 (s)")
    ap.add_argument("--max-shot", type=int, default=-1)
    ap.add_argument("--native", action="store_true",
                    help="不调用 sfsuread，用纯 Python 读 SU 道头")
    args = ap.parse_args()
    if abs(abs(float(args.offset_sign)) - 1.0) > 1e-12:
        ap.error("--offset-sign 只能是 +1 或 -1")
    if args.geom == "offset":
        print(
            "geom=offset: shot_x = %.6g + (%g)*(offset_m/1000) km；忽略 sx/gx"
            % (float(args.obs_x), float(args.offset_sign))
        )

    if args.su:
        files = [args.su]
    elif args.su_dir:
        files = sorted(glob.glob(os.path.join(args.su_dir, "*.su")))
        if not files:
            ap.error("目录内无 .su: %s" % args.su_dir)
        if args.group != "file":
            print("note: --su-dir 时建议 --group file", file=sys.stderr)
    else:
        ap.error("需要 --su 或 --su-dir")

    if np is None:
        sys.exit("需要 numpy: pip install numpy")

    try:
        args.trid = resolve_component_to_trid(args.component, args.trid)
    except RuntimeError as e:
        sys.exit(str(e))
    if args.component and args.trid:
        print("component %s -> trid=%s" % (args.component, args.trid))

    trid_keep_early = parse_trid_list(args.trid)
    # list-trid：只统计 trid，不保留波形；单分量导出：尽早丢弃其它分量；
    # all-trid：必须保留全波形（None）
    if args.list_trid:
        early_filter = []  # 空集：hist only
    elif args.all_trid:
        early_filter = None
    elif trid_keep_early:
        early_filter = trid_keep_early
    else:
        early_filter = None

    hist = None
    if args.native:
        print("backend: native Python SU reader")
        traces, hist = collect_traces_native(files, args, trid_keep=early_filter)
    else:
        print("backend: sfsuread")
        try:
            traces = collect_traces_suread(files, args)
        except Exception as e:
            print("sfsuread 路径失败: %s" % e, file=sys.stderr)
            print("自动回退 --native ...", file=sys.stderr)
            traces, hist = collect_traces_native(files, args, trid_keep=early_filter)

    process_traces(traces, args, hist=hist)
    if not args.list_trid:
        print("检查: GUI「工区几何」落点 / offset 符号（明细在 diag/）")
        print("几何说明: diag/obs_segy_geometry.txt  |  scalco/scalel 见 segy.h")
        print("若不对: --geom obs|segy|offset / --obs-x / --offset-sign / --zobs-mode")


if __name__ == "__main__":
    main()
