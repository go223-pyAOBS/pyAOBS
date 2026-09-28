# -*- coding: utf-8 -*-
"""
预处理：zplotpy.DataProcessor（带通 / 速度 mute / 显示增益）+ 多边形 mute。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..project import PreprocessParams
from .polygon_mute import apply_polygon_mute, list_to_points

_PROCESSOR = None
_BP_BACKEND: Optional[str] = None  # "src" | "scipy" | "none"
_BP_WARNED = False


def _get_processor():
    global _PROCESSOR
    if _PROCESSOR is None:
        try:
            from pyAOBS.visualization.zplotpy.core.data_processor import DataProcessor
        except ImportError:
            from visualization.zplotpy.core.data_processor import DataProcessor  # type: ignore
        _PROCESSOR = DataProcessor(enable_cache=False)
    return _PROCESSOR


def bandpass_backend() -> str:
    """最近一次带通实际使用的后端（供状态栏/日志）。"""
    return str(_BP_BACKEND or "unknown")


def _bandpass_scipy(
    trace: np.ndarray, freqlo: float, freqhi: float, fs: float, npoles: int
) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    nyq = 0.5 * float(fs)
    lo = max(1e-6, min(float(freqlo), nyq * 0.99))
    hi = max(lo * 1.01, min(float(freqhi), nyq * 0.99))
    if lo >= hi:
        return np.asarray(trace, dtype=np.float32)
    b, a = butter(max(1, int(npoles) // 2), [lo / nyq, hi / nyq], btype="band")
    y = filtfilt(b, a, np.asarray(trace, dtype=np.float64))
    return y.astype(np.float32, copy=False)


def _apply_bandpass_trace(
    trace: np.ndarray,
    freqlo: float,
    freqhi: float,
    npoles: int,
    izerop: int,
    fs: float,
) -> np.ndarray:
    """优先 zplot Fortran bndpas；失败则 scipy butter+filtfilt。"""
    global _BP_BACKEND, _BP_WARNED
    if freqlo <= 0 or freqhi <= 0 or freqlo >= freqhi:
        _BP_BACKEND = "none"
        return np.asarray(trace, dtype=np.float32)
    try:
        proc = _get_processor()
        out = proc.apply_bandpass_filter(
            np.asarray(trace, dtype=np.float32),
            float(freqlo),
            float(freqhi),
            int(npoles),
            int(izerop),
            float(fs),
        )
        _BP_BACKEND = "src"
        return np.asarray(out, dtype=np.float32)
    except Exception as exc:
        try:
            out = _bandpass_scipy(trace, freqlo, freqhi, fs, npoles)
            _BP_BACKEND = "scipy"
            if not _BP_WARNED:
                _BP_WARNED = True
                print(
                    "[obs_rtm] 带通：src 内核不可用，已回退 scipy（%s）" % exc
                )
            return out
        except Exception:
            _BP_BACKEND = "none"
            raise


# 最近一次速度 mute 统计（供 GUI 日志/状态栏）
_LAST_VEL_MUTE_STATS: Optional[dict] = None


def last_vel_mute_stats() -> Optional[dict]:
    return _LAST_VEL_MUTE_STATS


def _velocity_mute_tline(offset_km: float, tmute: float, vmute: float) -> float:
    """左右分支 mute 线：t = tmute + |x|/vmute（原始时间，无折合）。"""
    x = abs(float(offset_km))
    tm = float(tmute)
    vm = float(vmute)
    if vm <= 1e-12:
        return tm
    return tm + x / vm


def _mute_edge_weights(
    tj: np.ndarray, t_line: float, tp: float, *, cut_deep: bool
) -> np.ndarray:
    """
    Madagascar mutter 式边缘权重：过渡带宽 tp（秒）。
    切深：t < t_line 保留，t_line→t_line+tp 余弦降到 0，其后置零。
    切浅：t > t_line+tp 保留，t_line→t_line+tp 余弦升到 1，其前置零。
    tp<=0：硬切（无过渡）。
    """
    w = np.ones(tj.shape, dtype=np.float64)
    tp = float(tp)
    if tp <= 1e-12:
        if cut_deep:
            w[tj > t_line] = 0.0
        else:
            w[tj < t_line] = 0.0
        return w
    t1 = float(t_line)
    t2 = t1 + tp
    if cut_deep:
        w[tj >= t2] = 0.0
        mid = (tj > t1) & (tj < t2)
        if np.any(mid):
            x = (tj[mid] - t1) / tp
            w[mid] = 0.5 * (1.0 + np.cos(np.pi * x))
    else:
        w[tj <= t1] = 0.0
        mid = (tj > t1) & (tj < t2)
        if np.any(mid):
            x = (tj[mid] - t1) / tp
            w[mid] = 0.5 * (1.0 - np.cos(np.pi * x))
        w[tj >= t2] = 1.0
    return w


def apply_velocity_mute(
    gather: np.ndarray,
    times: np.ndarray,
    offsets_km: np.ndarray,
    *,
    tmute: float,
    vmute: float,
    invert: bool = False,
    tp: float = 0.15,
) -> np.ndarray:
    """
    速度 mute（作用在原始道集绝对时间上，对齐 Madagascar mutter abs=y）。

    线：t_mute = tmute + |x|/vmute。
    默认 invert=False（inner）：切深；invert=True：切浅。
    tp：边缘余弦过渡（秒），默认 0.15 与 mutter tp 一致；0=硬切。
    显示折合用全局 display_vred，不参与本公式。
    """
    global _LAST_VEL_MUTE_STATS
    data = np.asarray(gather, dtype=np.float32).copy()
    times = np.asarray(times, dtype=np.float64)
    off = np.asarray(offsets_km, dtype=np.float64)
    nt, ntr = data.shape
    if nt < 2 or times.size < 2:
        _LAST_VEL_MUTE_STATS = {"ntr": ntr, "fully_muted": 0, "past_end": 0}
        return data
    tmax = float(times[min(nt, times.size) - 1])
    t0 = float(times[0])
    xmax = float(np.max(np.abs(off))) if off.size else 0.0
    vm_use = float(vmute)
    tm_use = float(tmute)
    tp_use = max(0.0, float(tp))
    cut_deep = not bool(invert)

    fully = 0
    line_beyond_t = 0  # t_mute ≥ T：默认模式下该道不切任何样点
    example = None
    for j in range(ntr):
        x = float(off[j]) if j < off.size else 0.0
        t_line = _velocity_mute_tline(x, tm_use, vm_use)
        if example is None and abs(x) > 1.0:
            example = (x, t_line)
        ns = int(min(nt, times.size))
        tj = times[:ns]
        if cut_deep:
            if t_line >= tmax - 1e-9:
                line_beyond_t += 1
            elif t_line + tp_use <= t0:
                data[:ns, j] = 0.0
            else:
                w = _mute_edge_weights(tj, t_line, tp_use, cut_deep=True)
                data[:ns, j] *= w.astype(np.float32)
        else:
            if t_line <= t0 + 1e-9 and tp_use <= 1e-12:
                pass
            elif t_line >= tmax - 1e-9:
                data[:ns, j] = 0.0
                line_beyond_t += 1
            else:
                w = _mute_edge_weights(tj, t_line, tp_use, cut_deep=False)
                data[:ns, j] *= w.astype(np.float32)
        if float(np.max(np.abs(data[:ns, j]))) < 1e-30:
            fully += 1
    x_at_tmax = -1.0
    if vm_use > 1e-12:
        x_at_tmax = max(0.0, (tmax - tm_use) * vm_use)
    raw_e = float(np.sum(np.asarray(gather, dtype=np.float64) ** 2)) + 1e-30
    keep_e = float(np.sum(np.asarray(data, dtype=np.float64) ** 2))
    raw_peak = float(np.max(np.abs(gather))) + 1e-30
    keep_peak = float(np.max(np.abs(data)))
    alive = max(0, int(ntr) - int(fully))
    _LAST_VEL_MUTE_STATS = {
        "ntr": int(ntr),
        "fully_muted": int(fully),
        "past_end": int(line_beyond_t),
        "alive_traces": int(alive),
        "alive_ratio": float(alive / max(1, ntr)),
        "energy_ratio": float(keep_e / raw_e),
        "keep_ratio": float(keep_peak / raw_peak),
        "tmax": float(tmax),
        "tmute": float(tm_use),
        "vmute": float(vm_use),
        "vmute_in": float(vmute),
        "tp": float(tp_use),
        "invert": bool(invert),
        "cut": "deep" if cut_deep else "shallow",
        "xmax_km": float(xmax),
        "x_line_at_T_km": float(x_at_tmax),
        "vmute_auto_adjusted": False,
        "example_offset": float(example[0]) if example else 0.0,
        "example_ttmute": float(example[1]) if example else 0.0,
    }
    return data


def apply_mute_only(
    gather: np.ndarray,
    times: np.ndarray,
    offsets_km: Optional[np.ndarray],
    params: PreprocessParams,
    *,
    x_coords: Optional[Sequence[float]] = None,
    skip_poly: bool = False,
    x_reduce_origin: float = 0.0,
) -> np.ndarray:
    """仅速度 mute + 多边形 mute（选择道集阶段）。

    ``skip_poly=True``：只做速度 mute，多边形留给画布叠层（避免 bake 后再 mute 把波形抹掉）。
    ``x_reduce_origin``：折合原点（OBS x），须与画布 ``t-|x-xobs|/vred`` 一致。
    """
    global _LAST_VEL_MUTE_STATS
    data = np.asarray(gather, dtype=np.float32).copy()
    nt, ntr = data.shape
    if nt < 2:
        return data

    if params.use_mute and offsets_km is not None:
        data = apply_velocity_mute(
            data,
            times,
            np.asarray(offsets_km, dtype=float),
            tmute=float(params.tmute),
            vmute=float(params.vmute),
            invert=bool(getattr(params, "vel_mute_invert", False)),
            tp=float(getattr(params, "mute_tp", 0.15) or 0.0),
        )
    else:
        _LAST_VEL_MUTE_STATS = None

    if (
        not skip_poly
        and params.use_poly_mute
        and params.poly_points
        and len(params.poly_points) >= 3
    ):
        if x_coords is not None:
            xc = x_coords
        elif params.poly_x_mode == "offset" and offsets_km is not None:
            # UI「offset」= 拼图 model x；offsets_km 若是相对 OBS，须加回原点
            # （勿直接当 model x，否则 OBS≠0 时多边形整体错位）
            xc = np.asarray(offsets_km, dtype=float) + float(x_reduce_origin)
        else:
            xc = np.arange(ntr, dtype=float)
        data = apply_polygon_mute(
            data,
            times,
            xc,
            list_to_points(params.poly_points),
            enabled=True,
            invert=bool(params.poly_invert),
            # 与画布折合显示一致，否则多边形与波形错位
            display_vred=float(getattr(params, "display_vred", 0.0) or 0.0),
            x_reduce_origin=float(x_reduce_origin),
            tp=float(getattr(params, "mute_tp", 0.15) or 0.0),
        )
    return data


def apply_bandpass_only(
    gather: np.ndarray,
    times: np.ndarray,
    params: PreprocessParams,
    *,
    npoles: int = 4,
    izerop: int = 0,
) -> np.ndarray:
    """仅带通（预处理阶段；输入应为已 mute 波形）。"""
    data = np.asarray(gather, dtype=np.float32).copy()
    nt, ntr = data.shape
    if nt < 2 or not params.use_bandpass:
        return data
    dt = float(times[1] - times[0]) if len(times) > 1 else 0.004
    fs = 1.0 / dt if dt > 0 else 250.0
    for j in range(ntr):
        data[:, j] = _apply_bandpass_trace(
            data[:, j],
            float(params.freqlo),
            float(params.freqhi),
            int(npoles),
            int(izerop),
            float(fs),
        )
    return data


def apply_bandpass_mute(
    gather: np.ndarray,
    times: np.ndarray,
    offsets_km: Optional[np.ndarray],
    params: PreprocessParams,
    *,
    npoles: int = 4,
    izerop: int = 0,
    x_coords: Optional[Sequence[float]] = None,
    do_mute: bool = True,
    do_bandpass: bool = True,
    x_reduce_origin: float = 0.0,
) -> np.ndarray:
    """
    gather: (nt, ntr)
    times: (nt,) s
    offsets_km: (ntr,) 速度 mute 用 (km)
    x_coords: 多边形 mute 的 x；默认按 poly_x_mode 取 offset 或 trace 序号
    """
    data = np.asarray(gather, dtype=np.float32).copy()
    if do_mute:
        data = apply_mute_only(
            data,
            times,
            offsets_km,
            params,
            x_coords=x_coords,
            x_reduce_origin=float(x_reduce_origin),
        )
    if do_bandpass:
        data = apply_bandpass_only(
            data, times, params, npoles=npoles, izerop=izerop
        )
    return data


def _vg2_vectorized(data: np.ndarray, dt: float, tvg: float, pvg: float) -> np.ndarray:
    """近似 Fortran vg2：滑窗 sum(|x|^pvg) 作除数（整道集向量化）。"""
    work = np.asarray(data, dtype=np.float64)
    if work.ndim != 2 or work.shape[0] < 2 or dt <= 0:
        return work.astype(np.float32, copy=False)
    nw = int(round(float(tvg) / float(dt))) + 1
    if nw % 2 == 0:
        nw += 1
    nw = max(3, nw)
    try:
        from scipy.ndimage import uniform_filter1d
    except ImportError:
        return work.astype(np.float32, copy=False)
    powers = np.abs(work) ** float(pvg)
    # uniform_filter1d 是均值；乘 nw 得近似窗口和
    denom = uniform_filter1d(powers, size=nw, axis=0, mode="nearest") * float(nw)
    denom = np.maximum(denom, 1e-20)
    return (work / denom).astype(np.float32)


def apply_display_gain(
    gather: np.ndarray,
    times: np.ndarray,
    offsets_km: Optional[np.ndarray],
    params: PreprocessParams,
) -> np.ndarray:
    """
    对齐 zplotpy 增益语义；预览用向量化实现（避免逐道 DataProcessor 调用）。
    """
    if not getattr(params, "use_gain", True):
        return np.asarray(gather, dtype=np.float32)
    data = np.asarray(gather, dtype=np.float32)
    nt, ntr = data.shape
    if nt < 1 or ntr < 1:
        return data.copy()
    times = np.asarray(times, dtype=np.float64)
    if offsets_km is None:
        offs = np.zeros(ntr, dtype=np.float64)
    else:
        offs = np.asarray(offsets_km, dtype=np.float64)
        if offs.size < ntr:
            offs = np.pad(offs, (0, ntr - offs.size), mode="edge")

    iscale = int(params.iscale)
    amp = float(params.amp)
    rcor = float(params.rcor)
    clip = float(params.clip)
    r_nonneg = max(0.0, rcor)

    work = data.astype(np.float32, copy=True)
    if iscale == 2:
        dt = float(times[1] - times[0]) if times.size > 1 else 0.004
        work = _vg2_vectorized(work, dt, float(params.tvg), float(params.pvg))

    if iscale in (0, 2):
        ampmax = np.max(np.abs(work), axis=0)
        ampmax = np.maximum(ampmax, 1e-20)
        out = work * np.float32(amp) / ampmax.astype(np.float32)[None, :]
    else:
        # iscale=1：scalef = sf * |off*10|^rcor；sf<=0 时用「有能量的参考道」估
        # 切浅时常把最远偏（道序第 0 道）整道抹掉；若仍用第 0 道 → sf=0 → 整幅全黑
        off_for_pow = np.maximum(np.abs(offs) * 10.0, 1e-6)
        sf = float(params.sf)
        if sf <= 0.0:
            ampmax = np.max(np.abs(work), axis=0)
            # 选峰值最大的道作参考（mute 后仍存活）
            j_ref = int(np.argmax(ampmax)) if ntr else 0
            ampmax0 = float(ampmax[j_ref]) if ntr else 0.0
            if ampmax0 < 1e-20:
                # 再尝试任一存活道
                live = np.where(ampmax >= 1e-20)[0]
                if live.size:
                    j_ref = int(live[0])
                    ampmax0 = float(ampmax[j_ref])
            denom = ampmax0 * (float(off_for_pow[j_ref]) ** r_nonneg)
            sf = amp / denom if denom > 1e-20 else 0.0
        scalef = (sf * (off_for_pow ** r_nonneg)).astype(np.float32)
        out = work * scalef[None, :]

    if clip > 0.0:
        out = np.clip(out, -clip, clip)
    return np.asarray(out, dtype=np.float32)


def time_axis(nt: int, dt: float, o1: float = 0.0) -> np.ndarray:
    return o1 + np.arange(nt, dtype=np.float64) * float(dt)
