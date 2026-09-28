# -*- coding: utf-8 -*-
"""
道集处理：
  预处理页：仅预览（不写盘）
  「应用选道」：bp(raw)→mute→gain → shots_proc/；mute(raw)→shots_mute/
  （先带通再 mute，避免 mute 后滤波产生平行假震相）
  写出后清空目录中非本次选道文件；禁止对已定稿波形再滤/再增益。
"""

from __future__ import annotations

import os
import shutil
from typing import Callable, List, Optional, Set

from ..project import ObsRtmProject, list_shot_rsf
from .montage import _obs_x_ref_km, montage_model_x_km
from .preprocess import (
    apply_bandpass_only,
    apply_display_gain,
    apply_mute_only,
    time_axis,
)
from .rsf_io import load_offsets_table, read_gather, write_gather


def _poly_x_coords_for_shot(project: ObsRtmProject, ishot: int, ntr: int):
    """拼图多边形横轴为 model_x；单炮各道共用该炮的 model_x。"""
    import numpy as np

    if ntr <= 0:
        return None
    if ishot < 0:
        return np.arange(int(ntr), dtype=float)
    x_model = float(montage_model_x_km(project, ishot))
    return np.full(int(ntr), x_model, dtype=float)


def _mute_shot_gather(project: ObsRtmProject, data, times, off_km, ishot: int, prep):
    """速度 mute 用相对 offset；多边形 mute 用 model_x + OBS 折合原点。"""
    ntr = int(getattr(data, "shape", (0, 0))[1]) if data is not None else 0
    return apply_mute_only(
        data,
        times,
        off_km,
        prep,
        x_coords=_poly_x_coords_for_shot(project, ishot, ntr),
        x_reduce_origin=float(_obs_x_ref_km(project)),
    )


def _shot_index_from_name(path: str) -> int:
    base = os.path.basename(path)
    try:
        return int(base.replace("shot_", "").replace(".rsf", ""))
    except ValueError:
        return -1


def purge_extra_shot_rsf(
    dir_path: str,
    keep_basenames: Set[str],
    *,
    log: Optional[Callable[[str], None]] = None,
) -> int:
    """
    删除目录中不在 keep 内的 shot_*.rsf / shot_*.rsf@。
    使 shots_proc、shots_mute 仅保留本次选道结果。
    """
    if not dir_path or not os.path.isdir(dir_path):
        return 0
    keep = {os.path.basename(k) for k in keep_basenames}
    n_del = 0
    for name in list(os.listdir(dir_path)):
        if not name.startswith("shot_"):
            continue
        if name.endswith(".rsf@"):
            stem = name[:-5] + ".rsf"  # shot_001.rsf@ → shot_001.rsf
            if stem in keep:
                continue
        elif name.endswith(".rsf"):
            if name in keep:
                continue
        else:
            continue
        path = os.path.join(dir_path, name)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
                n_del += 1
        except OSError as exc:
            if log:
                log("删除旧文件失败 %s: %s" % (name, exc))
    if log and n_del:
        log(
            "已清理 %s/ 中非本次选道文件 %d 个"
            % (os.path.basename(dir_path.rstrip("/\\")) or dir_path, n_del)
        )
    return n_del


def offsets_km_for_shot(project: ObsRtmProject, ishot: int, ntr: int):
    """单炮各道相对 OBS 的偏移 km（速度 mute / 增益用，非拼图横轴）。"""
    import numpy as np

    from .montage import montage_rel_offset_km

    table = load_offsets_table(project.path(project.offsets_txt))
    row = table.get(ishot, [])
    if not row:
        off0 = float(montage_rel_offset_km(project, ishot))
        return np.full(max(1, int(ntr)), off0, dtype=float)
    # 多道：相对第 0 道几何偏移，叠加 offsets 表差值
    arr = np.asarray(row[:ntr], dtype=float) * 0.001  # m → km
    if arr.size < ntr:
        arr = np.pad(arr, (0, ntr - arr.size))
    off0 = float(montage_rel_offset_km(project, ishot))
    arr = arr - float(arr[0]) + off0
    return arr


def list_mute_rsf(project: ObsRtmProject) -> List[str]:
    d = project.path(getattr(project, "shots_mute_dir", None) or "shots_mute")
    if not os.path.isdir(d):
        return []
    names = sorted(
        n
        for n in os.listdir(d)
        if n.startswith("shot_") and n.endswith(".rsf") and not n.endswith(".rsf@")
    )
    return [os.path.join(d, n) for n in names]


def mute_path_for_shot(project: ObsRtmProject, src_or_name: str) -> str:
    name = os.path.basename(src_or_name)
    return project.path(getattr(project, "shots_mute_dir", None) or "shots_mute", name)


def process_one_mute(
    project: ObsRtmProject,
    src_rsf: str,
    dst_rsf: str,
) -> str:
    """原始道集 → 仅 mute → dst。"""
    data, meta = read_gather(src_rsf)
    nt, ntr = data.shape
    d1, o1 = float(meta["d1"]), float(meta["o1"])
    times = time_axis(nt, d1, o1)
    ishot = _shot_index_from_name(src_rsf)
    off_km = offsets_km_for_shot(project, ishot, ntr)
    out = _mute_shot_gather(
        project, data, times, off_km, ishot, project.preprocess
    )
    write_gather(dst_rsf, out, d1=d1, o1=o1)
    return dst_rsf


def process_one_filter(
    project: ObsRtmProject,
    src_rsf: str,
    dst_rsf: str,
) -> str:
    """已 mute 的原始结果 → 带通 + 增益一次 → dst（不再 mute）。"""
    data, meta = read_gather(src_rsf)
    nt, ntr = data.shape
    d1, o1 = float(meta["d1"]), float(meta["o1"])
    times = time_axis(nt, d1, o1)
    prep = project.preprocess
    out = apply_bandpass_only(data, times, prep)
    ishot = _shot_index_from_name(src_rsf)
    off_km = offsets_km_for_shot(project, ishot, ntr)
    out = apply_display_gain(out, times, off_km, prep)
    write_gather(dst_rsf, out, d1=d1, o1=o1)
    return dst_rsf


def process_one_rtm_final(
    project: ObsRtmProject,
    src_rsf: str,
    dst_proc: str,
    *,
    apply_prep: bool,
    dst_mute: Optional[str] = None,
) -> str:
    """
    定稿写出（对齐 Madagascar：先带通再 mutter，避免 mute 后滤波假震相）：
      shots_mute/ ← mute(raw)
      shots_proc/ ← gain( mute( bp(raw) ) )   若 apply_prep
                 ← mute(raw)                 否则
    滤波/增益各至多一次，绝不对已定稿波形重复。
    """
    data, meta = read_gather(src_rsf)
    nt, ntr = data.shape
    d1, o1 = float(meta["d1"]), float(meta["o1"])
    times = time_axis(nt, d1, o1)
    ishot = _shot_index_from_name(src_rsf)
    off_km = offsets_km_for_shot(project, ishot, ntr)
    prep = project.preprocess
    # 备查：原始上的 mute（无滤波振铃）
    muted_raw = _mute_shot_gather(project, data, times, off_km, ishot, prep)
    if dst_mute:
        write_gather(dst_mute, muted_raw, d1=d1, o1=o1)
    if apply_prep:
        tmp = apply_bandpass_only(data, times, prep)
        muted = _mute_shot_gather(project, tmp, times, off_km, ishot, prep)
        out = apply_display_gain(muted, times, off_km, prep)
    else:
        out = muted_raw
    write_gather(dst_proc, out, d1=d1, o1=o1)
    return dst_proc


def process_mute_shots(
    project: ObsRtmProject,
    *,
    shot_paths: Optional[List[str]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> int:
    """兼容：仅 mute → shots_mute/；写完后清理非本次选道文件。"""
    project.ensure_workdir()
    srcs = shot_paths or list_shot_rsf(project)
    if not srcs:
        raise RuntimeError("没有 shot_*.rsf，请先导入 SU")
    out_dir = project.path(getattr(project, "shots_mute_dir", None) or "shots_mute")
    os.makedirs(out_dir, exist_ok=True)
    keep: Set[str] = set()
    n_ok = 0
    for i, src in enumerate(srcs):
        name = os.path.basename(src)
        dst = os.path.join(out_dir, name)
        try:
            process_one_mute(project, src, dst)
            keep.add(name)
            n_ok += 1
            if log and (i < 5 or i == len(srcs) - 1 or (i + 1) % 50 == 0):
                log("mute %s -> %s (%d/%d)" % (name, dst, i + 1, len(srcs)))
        except Exception as exc:
            if log:
                log("FAIL mute %s: %s" % (name, exc))
    purge_extra_shot_rsf(out_dir, keep, log=log)
    return n_ok


def process_rtm_shots(
    project: ObsRtmProject,
    *,
    shot_paths: Optional[List[str]] = None,
    apply_prep: bool = True,
    log: Optional[Callable[[str], None]] = None,
) -> int:
    """
    「应用选道」定稿：bp(raw)→mute→gain → shots_proc/；mute(raw)→shots_mute/。
    两目录均只保留本次选道文件。
    """
    project.ensure_workdir()
    srcs = shot_paths or list_shot_rsf(project)
    if not srcs:
        raise RuntimeError("没有 shot_*.rsf，请先导入 SU")
    mute_dir = project.path(getattr(project, "shots_mute_dir", None) or "shots_mute")
    out_dir = project.path(project.shots_proc_dir)
    os.makedirs(mute_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    p = project.preprocess
    note = os.path.join(out_dir, "README.txt")
    with open(note, "w", encoding="utf-8") as f:
        f.write(
            "stage=apply_select_final\n"
            "pipeline=%s\n"
            "bandpass=%s %g-%g Hz\n"
            "gain=%s\n"
            "mute_tp=%g\n"
            "order=bp_then_mute_then_gain\n"
            "no_double_filter=1\n"
            "purge_extra=1\n"
            % (
                "raw->bp->mute->gain" if apply_prep else "raw->mute",
                p.use_bandpass,
                p.freqlo,
                p.freqhi,
                p.use_gain,
                float(getattr(p, "mute_tp", 0.15) or 0.0),
            )
        )
    keep: Set[str] = set()
    n_ok = 0
    for i, src in enumerate(srcs):
        name = os.path.basename(src)
        dst_m = os.path.join(mute_dir, name)
        dst_p = os.path.join(out_dir, name)
        try:
            process_one_rtm_final(
                project,
                src,
                dst_p,
                apply_prep=bool(apply_prep),
                dst_mute=dst_m,
            )
            keep.add(name)
            n_ok += 1
            if log and (i < 5 or i == len(srcs) - 1 or (i + 1) % 50 == 0):
                log(
                    "final %s -> %s (%s) (%d/%d)"
                    % (
                        name,
                        dst_p,
                        "bp→mute→gain" if apply_prep else "mute",
                        i + 1,
                        len(srcs),
                    )
                )
        except Exception as exc:
            if log:
                log("FAIL final %s: %s" % (name, exc))
    purge_extra_shot_rsf(mute_dir, keep, log=log)
    purge_extra_shot_rsf(out_dir, keep, log=log)
    return n_ok


def process_filter_shots(
    project: ObsRtmProject,
    *,
    shot_paths: Optional[List[str]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> int:
    """兼容旧调用：从 shots_mute/ 带通+增益 → shots_proc/（不 auto-mute）。"""
    project.ensure_workdir()
    raw_srcs = shot_paths or list_shot_rsf(project)
    if not raw_srcs:
        raise RuntimeError("没有 shot_*.rsf")
    mute_dir = project.path(getattr(project, "shots_mute_dir", None) or "shots_mute")
    out_dir = project.path(project.shots_proc_dir)
    os.makedirs(out_dir, exist_ok=True)
    n_ok = 0
    for i, src in enumerate(raw_srcs):
        name = os.path.basename(src)
        muted = os.path.join(mute_dir, name)
        if not os.path.isfile(muted):
            if log:
                log("SKIP %s: 无 shots_mute/" % name)
            continue
        dst = os.path.join(out_dir, name)
        try:
            process_one_filter(project, muted, dst)
            n_ok += 1
            if log and (i < 5 or i == len(raw_srcs) - 1 or (i + 1) % 50 == 0):
                log("bp+gain %s -> %s (%d/%d)" % (name, dst, i + 1, len(raw_srcs)))
        except Exception as exc:
            if log:
                log("FAIL bp+gain %s: %s" % (name, exc))
    return n_ok


def process_shots(
    project: ObsRtmProject,
    *,
    shot_paths: Optional[List[str]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> int:
    """兼容：mute + 带通增益一次写入 shots_proc/。"""
    apply_prep = bool(getattr(project.preprocess, "apply_preprocess", True))
    return process_rtm_shots(
        project, shot_paths=shot_paths, apply_prep=apply_prep, log=log
    )


def ensure_raw_backup(project: ObsRtmProject, log: Optional[Callable[[str], None]] = None) -> None:
    from .workdir_layout import INPUTS_RAW

    raw = project.path(INPUTS_RAW)
    marker = os.path.join(raw, ".shots_backup_ok")
    if os.path.isfile(marker):
        return
    src = project.path(project.shots_dir)
    if not os.path.isdir(src):
        return
    os.makedirs(raw, exist_ok=True)
    n = 0
    for name in os.listdir(src):
        if name.endswith(".rsf") or name.endswith(".rsf@"):
            dst = os.path.join(raw, name)
            if not os.path.isfile(dst):
                shutil.copy2(os.path.join(src, name), dst)
                n += 1
    try:
        with open(marker, "w", encoding="utf-8") as f:
            f.write("ok\n")
    except OSError:
        pass
    if log and n:
        log("已备份原始 shots → %s/（%d 文件）" % (INPUTS_RAW, n))
