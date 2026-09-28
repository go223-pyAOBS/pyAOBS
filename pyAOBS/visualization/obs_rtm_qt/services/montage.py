# -*- coding: utf-8 -*-
"""
按模型测线坐标拼多炮超级道集。

横轴 = model distance = OBS_x + 各炮相对 OBS 的 offset
（优先 shots_xz 的 shot_x；否则 obs_x + sign×offsets.txt）。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..project import ObsRtmProject, list_shot_rsf
from .geometry import load_xz_txt
from .rsf_io import load_offsets_table, parse_rsf_header, resolve_rsf_binary


def _read_first_trace(path: str) -> Tuple[np.ndarray, float, float]:
    """只读第 0 道（OBS montage），避免整道集拷贝。"""
    meta = parse_rsf_header(path)
    n1 = int(meta["n1"])
    d1 = float(meta.get("d1", "0.004"))
    o1 = float(meta.get("o1", "0"))
    bin_path = resolve_rsf_binary(path, meta)
    # native float，道优先：前 n1 个样本即第 0 道
    tr = np.fromfile(bin_path, dtype=np.float32, count=n1)
    if tr.size != n1:
        raise RuntimeError("short read %s: got %d want %d" % (path, tr.size, n1))
    return tr, d1, o1


def _shot_index_from_path(path: str) -> int:
    base = os.path.basename(path)
    try:
        return int(base.replace("shot_", "").replace(".rsf", ""))
    except ValueError:
        return -1


def _obs_x_ref_km(project: ObsRtmProject) -> float:
    obs = load_xz_txt(project.path(project.obs_xz))
    if obs:
        return float(obs[0][0])
    return float(getattr(project.geometry, "obs_x_km", 0.0) or 0.0)


def _shot_x_by_index(project: ObsRtmProject) -> Dict[int, float]:
    """shots_xz.txt 第 i 行 ↔ shot_i.rsf 的炮点 x（km）。"""
    pts = load_xz_txt(project.path(project.shots_xz))
    return {i: float(pts[i][0]) for i in range(len(pts))}


def montage_rel_offset_km(project: ObsRtmProject, ishot: int) -> float:
    """相对 OBS 的偏移 km（速度 mute / 折合 / 增益 rcor 用）。"""
    return float(montage_model_x_km(project, ishot) - _obs_x_ref_km(project))


def montage_model_x_km(project: ObsRtmProject, ishot: int) -> float:
    """
    拼图横轴：模型测线坐标 km。
    model_x = OBS_x + 相对 offset。
    优先 shots_xz 的 shot_x；否则 obs_x + sign×(offset_m/1000)。
    """
    shot_x = _shot_x_by_index(project)
    if int(ishot) in shot_x:
        return float(shot_x[int(ishot)])
    obs = _obs_x_ref_km(project)
    off_table = load_offsets_table(project.path(project.offsets_txt))
    row = off_table.get(int(ishot)) or []
    om = float(row[0]) if row else 0.0  # m
    geom = str(getattr(project.geometry, "geom", "offset") or "offset")
    sign = float(getattr(project.geometry, "offset_sign", 1.0) or 1.0)
    if geom == "offset":
        return obs + sign * (om * 0.001)
    return obs + om * 0.001


def montage_offset_km(project: ObsRtmProject, ishot: int) -> float:
    """兼容旧名：现为模型测线坐标（同 montage_model_x_km）。"""
    return montage_model_x_km(project, ishot)


def montage_offset_span_km(
    project: ObsRtmProject, *, stride: int = 1
) -> Optional[Tuple[float, float]]:
    """全炮模型测线坐标 [xmin, xmax]（km）。"""
    stride = max(1, int(stride))
    shot_x = _shot_x_by_index(project)
    if shot_x:
        xs = np.sort(np.asarray(list(shot_x.values()), dtype=float))
    else:
        paths = list_shot_rsf(project)
        if not paths:
            return None
        xs = np.sort(
            np.asarray(
                [
                    float(montage_model_x_km(project, _shot_index_from_path(p)))
                    for p in paths
                ],
                dtype=float,
            )
        )
    if xs.size == 0:
        return None
    xs = xs[::stride]
    if xs.size == 0:
        return None
    x0, x1 = float(xs[0]), float(xs[-1])
    if xs.size >= 2:
        dx = float(np.median(np.diff(xs)))
        if np.isfinite(dx) and abs(dx) > 1e-12:
            x0 -= 0.5 * abs(dx)
            x1 += 0.5 * abs(dx)
    if x1 < x0:
        x0, x1 = x1, x0
    return x0, x1


def typical_montage_dx_km(project: ObsRtmProject, *, stride: int = 1) -> float:
    """
    全炮典型道间距 (km)。

    手选稀疏子集仍用绝对 offset 排布时，邻道空隙可达数公里；
    wiggle/density 定宽应取该典型间距，而不是空隙本身。
    """
    stride = max(1, int(stride))
    shot_x = _shot_x_by_index(project)
    if len(shot_x) >= 2:
        xs = np.sort(np.asarray(list(shot_x.values()), dtype=float))
    else:
        paths = list_shot_rsf(project)
        if len(paths) < 2:
            return 0.0
        xs = np.sort(
            np.asarray(
                [
                    float(montage_model_x_km(project, _shot_index_from_path(p)))
                    for p in paths
                ],
                dtype=float,
            )
        )
    if xs.size < 2:
        return 0.0
    xs = xs[::stride]
    if xs.size < 2:
        return 0.0
    d = np.diff(xs)
    d = d[np.isfinite(d) & (d > 1e-12)]
    if d.size == 0:
        return 0.0
    return float(np.median(d))


def list_shot_rsf_in_dir(dir_path: str) -> List[str]:
    """列出目录下 shot_*.rsf（不含 .rsf@）。"""
    if not dir_path or not os.path.isdir(dir_path):
        return []
    names = sorted(
        n
        for n in os.listdir(dir_path)
        if n.startswith("shot_") and n.endswith(".rsf") and not n.endswith(".rsf@")
    )
    return [os.path.join(dir_path, n) for n in names]


def build_offset_montage(
    project: ObsRtmProject,
    *,
    stride: int = 1,
    max_traces: int = 2000,
    force_include: Optional[str] = None,
    shot_indices: Optional[List[int]] = None,
    shot_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float, List[str]]:
    """
    返回:
      gather (nt, ntr), offsets_km (ntr,), d1, o1, shot_paths_used

    ``force_include``: 保证该 shot 路径出现在 montage 中（即使被 stride 抽掉）。
    ``shot_indices``: 若给定，只拼这些炮（显示用，通常来自手选浏览）；不再 stride 抽稀。
    ``shot_dir``: 若给定，从该目录读 shot_*.rsf（如 shots_proc/）。
    横轴为模型测线坐标 model_x = OBS_x + 相对 offset，并按该坐标排序。
    """
    if shot_dir:
        paths = list_shot_rsf_in_dir(shot_dir)
        if not paths:
            raise RuntimeError("目录中没有 shot_*.rsf: %s" % shot_dir)
    else:
        paths = list_shot_rsf(project)
        if not paths:
            raise RuntimeError("没有 shot_*.rsf")
    stride = max(1, int(stride))
    max_traces = max(1, int(max_traces))
    if shot_indices is not None:
        want = {int(i) for i in shot_indices}
        paths = [p for p in paths if _shot_index_from_path(p) in want]
        if not paths:
            raise RuntimeError("指定炮在目标目录下均未找到")
        sel = list(paths[:max_traces])
    else:
        sel = list(paths[::stride][:max_traces])
        if force_include:
            # 允许传入 shots/ 路径，映射到 shot_dir 同名文件
            fi_name = os.path.basename(force_include)
            fi = force_include
            if shot_dir:
                cand = os.path.join(shot_dir, fi_name)
                if os.path.isfile(cand):
                    fi = cand
            fi_norm = os.path.normpath(os.path.abspath(fi))
            have = {os.path.normpath(os.path.abspath(p)) for p in sel}
            if fi_norm not in have and os.path.isfile(fi):
                sel.append(fi)
                if len(sel) > max_traces:
                    sel = [
                        p
                        for p in sel
                        if os.path.normpath(os.path.abspath(p)) == fi_norm
                    ] + [
                        p
                        for p in sel
                        if os.path.normpath(os.path.abspath(p)) != fi_norm
                    ][: max_traces - 1]

    # 按模型测线坐标排序后再读波形（全炮 / 手选子集同一规则）
    meta_rows = []
    for p in sel:
        ishot = _shot_index_from_path(p)
        x_model = montage_model_x_km(project, ishot)
        meta_rows.append((p, x_model))
    meta_rows.sort(key=lambda x: x[1])

    cols: List[np.ndarray] = []
    xs_model: List[float] = []
    used: List[str] = []
    d1 = 0.004
    o1 = 0.0
    nt = None
    for p, x_model in meta_rows:
        tr, d1, o1 = _read_first_trace(p)
        if nt is None:
            nt = int(tr.size)
        if tr.size != nt:
            # 对齐到公共 nt
            if tr.size > nt:
                tr = tr[:nt]
            else:
                tr = np.pad(tr, (0, nt - tr.size))
        cols.append(tr)
        xs_model.append(float(x_model))
        used.append(p)
    if not cols:
        raise RuntimeError("montage 为空")
    gather = np.column_stack(cols).astype(np.float32, copy=False)
    offsets = np.asarray(xs_model, dtype=float)
    return gather, offsets, d1, o1, used


def used_path_index_map(used: List[str]) -> Dict[str, int]:
    """basename / abspath → 道下标；换炮时 O(1) 查找，避免反复 abspath。"""
    m: Dict[str, int] = {}
    for i, p in enumerate(used):
        m[os.path.basename(p)] = i
        try:
            m[os.path.normpath(os.path.abspath(p))] = i
        except Exception:
            pass
    return m


def highlight_index_for_path(
    used: List[str],
    path: Optional[str],
    *,
    index_map: Optional[Dict[str, int]] = None,
) -> Optional[int]:
    """在 montage 返回的 used 列表中定位选中炮下标。"""
    if not path or not used:
        return None
    m = index_map if index_map is not None else used_path_index_map(used)
    try:
        target = os.path.normpath(os.path.abspath(path))
    except Exception:
        target = path
    if target in m:
        return int(m[target])
    base = os.path.basename(path)
    if base in m:
        return int(m[base])
    return None
