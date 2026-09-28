"""Tk-independent FileResult + compute helpers (extracted from iphase_gui)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import numpy as np

from ...phase_filter import select_phases
from ...phase_combine import combine_ppp_pps_pss_from_datasets, compute_ppp_pps_diff_pairs
from ...qc_metrics import stats_ppp_pps_diff_by_offset
from ...io_tx import read_tx
from ...models import PhaseDataset
from ...theoretical_ppp_pps import fit_ppp_time_curve_local_linear

PHASE_PPP = 5
PHASE_PPS = 14
PHASE_PSS = 24
DEFAULT_PSP_PHASE = 40


@dataclass
class FileResult:
    path: Path
    ds: object
    ds_ppp: object
    ds_pps: object
    ds_pss: object
    diff_pairs: list[tuple[float, float, float]]
    fit_stats: dict
    tx2_out: object | None
    # 合并 tx.in 按台站拆分后：台站模型距离与站号（可选）
    obs_x: float | None = None
    obs_id: str | None = None


def phase_true_offset_time(ds, phase_id: int) -> tuple[np.ndarray, np.ndarray]:
    offs, ts = [], []
    for s in ds.shots:
        for p in s.picks:
            if p.phase_id == phase_id:
                offs.append(p.x - s.xshot)
                ts.append(p.t)
    return np.asarray(offs, dtype=float), np.asarray(ts, dtype=float)


def phase_model_distance_time(ds, phase_id: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ts = [], []
    for s in ds.shots:
        for p in s.picks:
            if p.phase_id == phase_id:
                xs.append(p.x)
                ts.append(p.t)
    return np.asarray(xs, dtype=float), np.asarray(ts, dtype=float)


def phase_model_trueoff_time(ds, phase_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, offs, ts = [], [], []
    for s in ds.shots:
        for p in s.picks:
            if p.phase_id == phase_id:
                xs.append(p.x)
                offs.append(p.x - s.xshot)
                ts.append(p.t)
    return np.asarray(xs, dtype=float), np.asarray(offs, dtype=float), np.asarray(ts, dtype=float)


_phase_true_offset_time = phase_true_offset_time
_phase_model_distance_time = phase_model_distance_time
_phase_model_trueoff_time = phase_model_trueoff_time


def obs_tag_from_path(path: Path) -> str:
    """从文件名中提取 OBS 号数字，如 tx_OBS22.in / tx-obs_22.in -> 22。"""
    m = re.search(r"OBS\D*(\d+)", path.stem, flags=re.IGNORECASE)
    if m:
        return str(int(m.group(1)))
    return path.stem


def obs_tag_from_result(res: FileResult) -> str:
    """优先拆分后的站号，否则回退文件名 OBS 号。"""
    if res.obs_id is not None and str(res.obs_id).strip() not in ("", "—"):
        return str(res.obs_id).strip()
    return obs_tag_from_path(res.path)


def result_display_name(res: FileResult) -> str:
    """图例/状态栏用短名。"""
    tag = obs_tag_from_result(res)
    if tag.isdigit():
        return f"OBS{tag}"
    if res.obs_x is not None:
        return f"x={float(res.obs_x):.3f}"
    return res.path.stem


def result_cache_key(res: FileResult) -> str:
    """反演/理论缓存键：同源合并文件按台站区分。"""
    try:
        base = str(res.path.resolve())
    except Exception:
        base = str(res.path)
    if res.obs_x is not None:
        return f"{base}@{round(float(res.obs_x), 3):.3f}"
    return base


def equi_tx_path_for_result(res: FileResult) -> Path:
    """2Dequi 输出路径：多站时按站号/x 区分，避免互相覆盖。"""
    parent = res.path.parent
    tag = obs_tag_from_result(res)
    if res.obs_x is not None:
        if tag.isdigit():
            return parent / f"tx_OBS{tag}_2Dequiv.in"
        return parent / f"tx_x{round(float(res.obs_x), 3):.3f}_2Dequiv.in"
    return parent / "tx_2Dequiv.in"


def obs_model_distance_from_tx(res: FileResult) -> float | None:
    """
    OBS 模型距离：
    - 拆分结果优先用 ``obs_x``；
    - 否则：tshot==-1 的炮头 xshot 众数。
    """
    if res.obs_x is not None and np.isfinite(res.obs_x):
        return float(res.obs_x)
    xs: list[float] = [
        float(s.xshot) for s in res.ds.shots if abs(float(s.tshot) + 1.0) < 1e-6
    ]
    if not xs:
        xs = [float(s.xshot) for s in res.ds.shots]
    if not xs:
        return None

    xr = np.round(np.asarray(xs, dtype=float), 3)
    vals, counts = np.unique(xr, return_counts=True)
    if vals.size == 0:
        return None
    return float(vals[int(np.argmax(counts))])


_obs_tag_from_path = obs_tag_from_path
_obs_tag_from_result = obs_tag_from_result
_obs_model_distance_from_tx = obs_model_distance_from_tx


def split_phase_dataset_by_obs_x(
    ds: PhaseDataset,
    *,
    x_decimals: int = 3,
) -> list[tuple[float, PhaseDataset]]:
    """按炮头 ``xshot``（台站模型距离）拆成多个子集；左右支同 x 合并。

    实现委托 ``rayinvr.tx_obs_catalog.split_tx_dataset_by_obs_x``（与 vedit / tomo2d 一致）。
    """
    from pyAOBS.modeling.rayinvr.tx_obs_catalog import split_tx_dataset_by_obs_x

    from ...io_tx import phase_dataset_to_tx, tx_dataset_to_phase

    x_tol = 10.0 ** (-int(x_decimals))
    parts = split_tx_dataset_by_obs_x(phase_dataset_to_tx(ds), x_tol=x_tol)
    return [(float(xobs), tx_dataset_to_phase(sub)) for xobs, sub in parts]


def _nearest_station_id(
    x: float,
    stations: Sequence[tuple[int, float, float]] | None,
    *,
    tol: float = 0.001,
) -> int | None:
    from pyAOBS.modeling.rayinvr.tx_obs_catalog import nearest_station_id

    return nearest_station_id(x, stations or (), tol=tol)


def compute_file_result_from_dataset(
    path: Path,
    ds: PhaseDataset,
    *,
    psp_phase_id: int = DEFAULT_PSP_PHASE,
    window_points: int = 11,
    strict_only: bool = False,
    obs_x: float | None = None,
    obs_id: str | None = None,
) -> FileResult:
    """对已准备好的 ``PhaseDataset`` 计算 FileResult（可带台站元数据）。"""
    ds_ppp, _ = select_phases(ds, [PHASE_PPP])
    ds_pps, _ = select_phases(ds, [PHASE_PPS])
    ds_pss, _ = select_phases(ds, [PHASE_PSS])

    if strict_only:
        diff_pairs = compute_ppp_pps_diff_pairs(
            ds_ppp, ds_pps, ip1=PHASE_PPP, ip2=PHASE_PPS, tol=0.05
        )
    else:
        ppp_x, ppp_off, ppp_t = phase_model_trueoff_time(ds_ppp, PHASE_PPP)
        pps_x, pps_off, pps_t = phase_model_trueoff_time(ds_pps, PHASE_PPS)
        diff_pairs: list[tuple[float, float, float]] = []
        for sgn in (-1.0, 1.0):
            m_ppp = (ppp_off * sgn) > 0.0
            m_pps = (pps_off * sgn) > 0.0
            if np.count_nonzero(m_ppp) < 3 or np.count_nonzero(m_pps) < 1:
                continue
            ppp_fit = fit_ppp_time_curve_local_linear(
                ppp_x[m_ppp],
                ppp_t[m_ppp],
                pps_x[m_pps],
                window_points=window_points,
                split_by_sign=False,
            )
            for j in range(int(np.sum(m_pps))):
                t_ppp_f = float(ppp_fit[j])
                if not np.isfinite(t_ppp_f):
                    continue
                md = float(pps_x[m_pps][j])
                to = float(pps_off[m_pps][j])
                dt = float(pps_t[m_pps][j]) - t_ppp_f
                diff_pairs.append((md, to, dt))
        diff_pairs.sort(key=lambda t: t[0])
    fit_stats = stats_ppp_pps_diff_by_offset(diff_pairs, degree=2)

    tx2_out = None
    try:
        res = combine_ppp_pps_pss_from_datasets(
            ds_ppp,
            ds_pps,
            ds_pss,
            ip1=PHASE_PPP,
            ip2=PHASE_PPS,
            ip3=PHASE_PSS,
            ip4=104,
            ip5=int(psp_phase_id),
            tol=0.05,
        )
        tx2_out = res.tx2_out
    except Exception:
        tx2_out = None

    return FileResult(
        path=path,
        ds=ds,
        ds_ppp=ds_ppp,
        ds_pps=ds_pps,
        ds_pss=ds_pss,
        diff_pairs=diff_pairs,
        fit_stats=fit_stats,
        tx2_out=tx2_out,
        obs_x=float(obs_x) if obs_x is not None else None,
        obs_id=str(obs_id) if obs_id is not None else None,
    )


def compute_file_result(
    path: Path,
    *,
    psp_phase_id: int = DEFAULT_PSP_PHASE,
    window_points: int = 11,
    strict_only: bool = False,
    allowed_phase_ids: set[int] | None = None,
) -> FileResult:
    """读单个 tx.in 为**一个** FileResult（不按站拆；兼容旧调用）。"""
    ds_raw = read_tx(str(path))
    if allowed_phase_ids is not None:
        if allowed_phase_ids:
            ds, _ = select_phases(ds_raw, sorted(allowed_phase_ids))
        else:
            ds = PhaseDataset()
    else:
        ds = ds_raw
    parts = split_phase_dataset_by_obs_x(ds)
    obs_x = parts[0][0] if len(parts) == 1 else None
    tag = obs_tag_from_path(path)
    obs_id = tag if tag.isdigit() and len(parts) == 1 else None
    return compute_file_result_from_dataset(
        path,
        ds,
        psp_phase_id=psp_phase_id,
        window_points=window_points,
        strict_only=strict_only,
        obs_x=obs_x,
        obs_id=obs_id,
    )


def compute_file_results(
    path: Path,
    *,
    psp_phase_id: int = DEFAULT_PSP_PHASE,
    window_points: int = 11,
    strict_only: bool = False,
    allowed_phase_ids: set[int] | None = None,
    stations: Sequence[tuple[int, float, float]] | None = None,
    split_multi_obs: bool = True,
) -> list[FileResult]:
    """
    加载 tx.in；若含多个台站 xshot，则拆成多个 FileResult（主分析单位 = 一站）。

    ``path`` 仍指向源文件（便于找同目录 r.in / tx.out）；``obs_x`` / ``obs_id`` 标识台站。
    """
    ds_raw = read_tx(str(path))
    if allowed_phase_ids is not None:
        if allowed_phase_ids:
            ds, _ = select_phases(ds_raw, sorted(allowed_phase_ids))
        else:
            ds = PhaseDataset()
    else:
        ds = ds_raw

    if not split_multi_obs:
        return [
            compute_file_result_from_dataset(
                path,
                ds,
                psp_phase_id=psp_phase_id,
                window_points=window_points,
                strict_only=strict_only,
            )
        ]

    parts = split_phase_dataset_by_obs_x(ds)
    if not parts:
        return [
            compute_file_result_from_dataset(
                path,
                ds,
                psp_phase_id=psp_phase_id,
                window_points=window_points,
                strict_only=strict_only,
            )
        ]

    path_tag = obs_tag_from_path(path)
    out: list[FileResult] = []
    for xobs, ds_part in parts:
        oid: str | None = None
        nid = _nearest_station_id(xobs, stations, tol=0.001)
        if nid is not None:
            oid = str(int(nid))
        elif len(parts) == 1 and path_tag.isdigit():
            oid = path_tag
        out.append(
            compute_file_result_from_dataset(
                path,
                ds_part,
                psp_phase_id=psp_phase_id,
                window_points=window_points,
                strict_only=strict_only,
                obs_x=float(xobs),
                obs_id=oid,
            )
        )
    return out


_compute_file_result = compute_file_result
_compute_file_results = compute_file_results
