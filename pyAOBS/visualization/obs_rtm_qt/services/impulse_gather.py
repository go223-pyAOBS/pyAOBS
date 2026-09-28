# -*- coding: utf-8 -*-
"""脉冲道集：与正传同频的带限 Ricker，供诊断性单脉冲 RTM。"""

from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import numpy as np

from ..project import ObsRtmProject
from .geometry import load_xz_txt
from .rsf_io import write_gather
from .rtm_job import rtm_run_dir
from .velocity import write_xz_rsf


def impulse_run_dir(project: ObsRtmProject) -> str:
    """``rtm_work/impulse/``：与正式作业隔离。"""
    d = os.path.join(rtm_run_dir(project), "impulse")
    os.makedirs(d, exist_ok=True)
    return d


def impulse_fm_hz(project: ObsRtmProject) -> float:
    """与正传 ``wav`` 一致的峰值频率：0.5*(fmin+fmax)。"""
    r = project.rtm
    fm = 0.5 * (float(r.fmin) + float(r.fmax))
    if fm <= 0:
        fm = max(float(r.fmax), 6.0)
    return float(fm)


def ricker_impulse_trace(
    nt: int,
    dt: float,
    fm: float,
    it_peak: int,
    amp_peak: float,
) -> np.ndarray:
    """
    零相位 Ricker，峰值落在 ``it_peak``，峰值振幅 = ``amp_peak``。

    形状与 Madagascar ``spike | ricker1 frequency=fm``（零相位谱）一致的常用时域式：
    ``(1-2π²f²t²) exp(-π²f²t²)``，与正传 ``wav`` 同 ``fm``。
    """
    nt = max(int(nt), 2)
    dt = float(dt)
    if dt <= 1e-12:
        raise ValueError("dt 无效")
    fm = max(float(fm), 1e-6)
    it_peak = int(max(0, min(int(it_peak), nt - 1)))
    t = (np.arange(nt, dtype=np.float64) - float(it_peak)) * dt
    pf = np.pi * fm
    w = (1.0 - 2.0 * (pf * t) ** 2) * np.exp(-(pf * t) ** 2)
    # 理论峰值 1；数值上再归一一次，保证 amp 精确落在 it_peak
    peak = float(w[it_peak])
    if abs(peak) < 1e-30:
        peak = float(np.max(np.abs(w))) or 1.0
    return np.asarray(w * (float(amp_peak) / peak), dtype=np.float32)


def impulse_nt_jsnap(
    project: ObsRtmProject, it: int
) -> Tuple[int, int, int]:
    """
    脉冲作业用的 (nt, jsnap, nsnap)。

    - nt 截到拾取时刻 + Ricker 右瓣/数值垫（不必跑满整道记录长）
    - jsnap：脉冲诊断默认更密（约 32–40 帧），便于波场动画；
      若 RTM 面板 jsnap 更小则更密；过大则仍压到 ≤48 帧以控体积
    """
    r = project.rtm
    dt = max(float(r.dt), 1e-6)
    nt_full = max(int(r.nt), 2)
    it_u = int(max(0, min(int(it), nt_full - 1)))
    fm = impulse_fm_hz(project)
    # 右瓣 ~1.5/fm + 数值垫；至少 0.8s
    half = max(int(1.5 / max(fm, 1.0) / dt), 40)
    pad = max(int(2.0 / max(fm, 1.0) / dt), int(0.8 / dt), half + 20, 50)
    nt = int(min(nt_full, it_u + pad + 1))
    nt = int(max(nt, min(nt_full, 128)))
    # 目标约 40 张；UI jsnap 更小 → 更密动画
    jsnap_ui = max(int(getattr(r, "jsnap", 80) or 80), 1)
    jsnap_lo = max(nt // 40, 1)
    jsnap = min(jsnap_lo, jsnap_ui)
    nsnap = max(nt // jsnap, 1)
    if nsnap > 48:
        jsnap = max(nt // 40, 1)
        nsnap = max(nt // jsnap, 1)
    if nsnap < 8:
        jsnap = max(nt // 16, 1)
        nsnap = max(nt // jsnap, 1)
    return nt, jsnap, nsnap


def sample_impulse_from_raw_shot(
    project: ObsRtmProject,
    *,
    ishot: int,
    iobs: int,
    t_true: float,
    shot_path_hint: Optional[str] = None,
) -> dict:
    """
    从原始/未增益道集取样（优先 ``shots/``，避免拼图显示链 amp）。

    返回 dict: path, amp, it, t_true, d1, o1, nt_shot, source_note
    """
    from .rsf_io import read_gather

    ishot = int(ishot)
    iobs = int(iobs)
    candidates = []
    raw = project.path(project.shots_dir, "shot_%03d.rsf" % ishot)
    if os.path.isfile(raw):
        candidates.append(("shots/", raw))
    hint = (shot_path_hint or "").strip()
    if hint and os.path.isfile(hint):
        note = os.path.basename(os.path.dirname(hint)) or "hint"
        candidates.append((note + "/", hint))
    for sub, label in (
        (project.shots_proc_dir, "shots_proc/"),
        (project.shots_mute_dir, "shots_mute/"),
    ):
        p = project.path(sub, "shot_%03d.rsf" % ishot)
        if os.path.isfile(p):
            candidates.append((label, p))
    # 去重保序
    seen = set()
    uniq = []
    for note, p in candidates:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        uniq.append((note, p))
    if not uniq:
        raise FileNotFoundError("找不到 shot_%03d.rsf（shots/ 等）" % ishot)

    path = uniq[0][1]
    source_note = uniq[0][0]
    data, meta = read_gather(path)
    nt_shot = int(data.shape[0])
    n2 = int(data.shape[1]) if data.ndim == 2 else 1
    if not (0 <= iobs < n2):
        raise ValueError(
            "OBS 列 iobs=%d 越界（shot n2=%d，拼图仅第 0 道）" % (iobs, n2)
        )
    d1 = float(meta.get("d1", 0.004) or 0.004)
    if d1 <= 1e-12:
        d1 = float(project.rtm.dt) if float(project.rtm.dt) > 1e-12 else 0.004
    o1 = float(meta.get("o1", 0.0) or 0.0)
    it = int(round((float(t_true) - o1) / d1))
    it = int(max(0, min(it, nt_shot - 1)))
    amp = float(data[it, iobs])
    t_snap = float(o1 + it * d1)
    return {
        "path": path,
        "amp": amp,
        "it": it,
        "t_true": t_snap,
        "d1": d1,
        "o1": o1,
        "nt_shot": nt_shot,
        "source_note": source_note,
    }


def write_impulse_obs_gather(
    project: ObsRtmProject,
    *,
    iobs: int,
    ishot: int,
    it: int,
    amp: float,
    nt: Optional[int] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[str, str, int, int, float]:
    """
    写入脉冲 OBS 道集与单炮 rec_shots。

    返回 (obs_gath_path, rec_shots_path, nt, it_used, amp_used)。
    道集形状 (nt, 1)：仅检波=拾取炮；在 it 处置与正传同 ``fm`` 的
    带限 Ricker（峰值=拾取样点 amp），不再用单点 δ。
    """
    r = project.rtm
    shot_pts = load_xz_txt(project.path(project.shots_xz))
    obs_pts = load_xz_txt(project.path(project.obs_xz))
    if not shot_pts:
        raise RuntimeError("缺少 shots_xz.txt")
    if not obs_pts:
        raise RuntimeError("缺少 obs_xz.txt")
    iobs = int(iobs)
    ishot = int(ishot)
    if not (0 <= iobs < len(obs_pts)):
        raise ValueError("OBS 下标越界: %d (n_obs=%d)" % (iobs, len(obs_pts)))
    if not (0 <= ishot < len(shot_pts)):
        raise ValueError("炮号越界: %d (n_shots=%d)" % (ishot, len(shot_pts)))

    nt_full = max(int(r.nt), 2)
    if nt is None:
        nt, _js, _ns = impulse_nt_jsnap(project, it)
    else:
        nt = max(int(nt), 2)
    nt = int(min(nt, nt_full))
    d1 = float(r.dt)
    o1 = 0.0
    it_u = int(max(0, min(int(it), nt - 1)))
    amp_u = float(amp)
    if abs(amp_u) < 1e-30:
        raise ValueError(
            "脉冲振幅≈0，拒绝写入；请在道集上重新点选有效样点"
        )

    fm = impulse_fm_hz(project)
    half = max(int(1.5 / max(fm, 1.0) / max(d1, 1e-12)), 1)
    run = impulse_run_dir(project)
    gather = np.zeros((nt, 1), dtype=np.float32)
    gather[:, 0] = ricker_impulse_trace(nt, d1, fm, it_u, amp_u)
    gpath = os.path.join(run, "obs_gath_%03d.rsf" % iobs)
    write_gather(gpath, gather, d1=d1, o1=o1, label2="ImpulseShot")

    sx, sz = float(shot_pts[ishot][0]), float(shot_pts[ishot][1])
    rec_path = os.path.join(run, "rec_shots.rsf")
    write_xz_rsf(rec_path, [(sx, sz)], label2="SHOT_REC")

    if log:
        t0 = o1 + it_u * d1
        xo, zo = float(obs_pts[iobs][0]), float(obs_pts[iobs][1])
        clip_note = ""
        if it_u < half:
            clip_note = "（拾取偏早，Ricker 左瓣被 t=0 截断）"
        log(
            "脉冲道集: OBS=%d @ (%.3f,%.3f)  shot=%d @ (%.3f,%.3f)  "
            "t=%.4fs (it=%d/%d) amp=%.4g  Ricker fm=%.3gHz%s  → %s"
            % (
                iobs,
                xo,
                zo,
                ishot,
                sx,
                sz,
                t0,
                it_u,
                nt,
                amp_u,
                fm,
                clip_note,
                os.path.basename(gpath),
            )
        )
    return gpath, rec_path, nt, it_u, amp_u
