"""
Joint OBS attitude correction using 3C waveforms + travel-time constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import math
import numpy as np

from .polarization_features import extract_polarization_features


DepthSampler = Callable[[float, float], Optional[float]]
ProgressCallback = Callable[[int, int, str], None]


@dataclass
class OrientationObservation:
    trace_idx: int
    pick_word: int
    t0: float
    dt: float
    z: np.ndarray
    r: np.ndarray
    t: np.ndarray
    source_xyz: np.ndarray
    receiver_xyz: np.ndarray
    offset_km: float = 0.0
    source_xy_geo: Optional[np.ndarray] = None
    source_xy_utm: Optional[np.ndarray] = None


@dataclass
class OrientationCorrectionInput:
    observations: List[OrientationObservation]
    initial_azimuth_deg: float = 0.0
    initial_tilt_deg: float = 0.0
    initial_position_correction: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    depth_sampler: Optional[DepthSampler] = None
    max_iterations: int = 4
    w_tt: float = 1.0
    w_pol: float = 1.0
    w_sym: float = 0.8
    progress_callback: Optional[ProgressCallback] = None


@dataclass
class OrientationCorrectionResult:
    success: bool
    azimuth_deg: float
    tilt_deg: float
    position_correction: Tuple[float, float, float]
    objective: float
    iterations: int
    source_depth_history: List[float] = field(default_factory=list)
    iteration_history: List[Dict[str, float]] = field(default_factory=list)
    details: Dict[str, float] = field(default_factory=dict)
    message: str = ""


def _rotate_components(r: np.ndarray, t: np.ndarray, z: np.ndarray, az_deg: float, tilt_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    az = np.deg2rad(float(az_deg))
    tilt = np.deg2rad(float(tilt_deg))
    # Horizontal rotation (R/T plane).
    rr = np.cos(az) * r + np.sin(az) * t
    tt = -np.sin(az) * r + np.cos(az) * t
    # Tilt around local T axis (mix R and Z).
    r2 = np.cos(tilt) * rr + np.sin(tilt) * z
    z2 = -np.sin(tilt) * rr + np.cos(tilt) * z
    return r2, tt, z2


def _travel_misfit(
    obs: List[OrientationObservation],
    depth_override: Optional[float],
    position_corr: Optional[np.ndarray] = None,
    time_shift_sec: float = 0.0,
) -> float:
    # User-defined direct-water-wave formula:
    # t_pred = sqrt(offset_km^2 + water_depth_km^2) / 1.5
    # No water depth -> cannot run correction.
    if depth_override is None or not np.isfinite(depth_override) or depth_override <= 0.0:
        return 1e9
    water_v = 1.5  # km/s
    misfits: List[float] = []
    pos = np.asarray(position_corr if position_corr is not None else np.zeros(3, dtype=float), dtype=float)
    if pos.size != 3:
        pos = np.zeros(3, dtype=float)

    def _len_to_km(v: float) -> float:
        d = abs(float(v))
        # UTM/meters -> km; already-km stays unchanged.
        return d / 1000.0 if d > 50.0 else d

    for o in obs:
        off_km = abs(float(o.offset_km))
        try:
            src_xy = np.asarray(o.source_xyz[:2], dtype=float)
            rec_xy = np.asarray(o.receiver_xyz[:2], dtype=float) + np.asarray(pos[:2], dtype=float)
            if np.all(np.isfinite(src_xy)) and np.all(np.isfinite(rec_xy)):
                geom = float(np.linalg.norm(rec_xy - src_xy))
                if np.isfinite(geom) and geom > 1e-9:
                    off_km = _len_to_km(geom)
        except Exception:
            pass
        if off_km <= 1e-9:
            continue
        # dz 也参与几何项：在水深上做小幅修正，使位置参数可被走时残差驱动。
        depth_eff_km = max(1e-6, float(depth_override) + _len_to_km(float(pos[2])))
        slant_km = float(np.sqrt(off_km * off_km + depth_eff_km * depth_eff_km))
        t_pred = slant_km / water_v + float(time_shift_sec)
        misfits.append((t_pred - float(o.t0)) / max(float(o.dt), 1e-5))
    if not misfits:
        return 1e9
    arr = np.asarray(misfits, dtype=float)
    return float(np.mean(arr * arr))


def _waveform_misfit(obs: List[OrientationObservation], azimuth_deg: float, tilt_deg: float) -> Tuple[float, float, float]:
    sym_scores: List[float] = []
    energy_scores: List[float] = []
    pol_scores: List[float] = []

    def _mirror_pair_loss(arr: np.ndarray, odd: bool) -> float:
        x = np.asarray(arr, dtype=float)
        n = int(x.size)
        if n < 6:
            return 1.0
        m = n // 2
        left = x[:m]
        right = x[-m:][::-1]
        if left.size == 0 or right.size == 0:
            return 1.0
        if odd:
            diff = left + right
        else:
            diff = left - right
        den = float(np.mean(x * x)) + 1e-12
        return float(np.mean(diff * diff) / den)

    for o in obs:
        r2, t2, z2 = _rotate_components(o.r, o.t, o.z, azimuth_deg, tilt_deg)
        e_r = float(np.mean(r2 * r2))
        e_t = float(np.mean(t2 * t2))
        e_z = float(np.mean(z2 * z2))
        # Symmetry constraints:
        # - Z tends to be even-symmetric (same polarity on mirrored sides)
        # - R tends to be odd-symmetric (opposite polarity on mirrored sides)
        z_even = _mirror_pair_loss(z2, odd=False)
        r_odd = _mirror_pair_loss(r2, odd=True)
        sym = 0.5 * (z_even + r_odd)
        # Energy constraints:
        # - T should be smallest
        # - R should dominate over Z
        t_ratio = e_t / max(e_r + e_z, 1e-12)
        r_dominance_penalty = max(0.0, e_z - e_r) / max(e_r + e_z, 1e-12)
        energy = t_ratio + r_dominance_penalty
        feat = extract_polarization_features(z2, r2, t2)
        # principal_vector order is [R, T, Z] in polarization_features.
        radial_align = abs(float(np.asarray(feat.principal_vector, dtype=float)[0]))
        # Prefer principal direction aligned with radial axis + strong polarization quality.
        pol_loss = (
            (1.0 - radial_align)
            + 0.5 * (1.0 - float(feat.rectilinearity))
            + 0.5 * (1.0 - float(feat.dominant_energy_ratio))
        )
        sym_scores.append(sym)
        energy_scores.append(float(max(0.0, energy)))
        pol_scores.append(float(np.clip(pol_loss, 0.0, 2.0)))
    if not sym_scores:
        return 1e9, 1e9, 1e9
    return (
        float(np.mean(pol_scores)),
        float(np.mean(sym_scores)),
        float(np.mean(energy_scores)),
    )


def _polarization_quality(obs: List[OrientationObservation], azimuth_deg: float, tilt_deg: float) -> Dict[str, float]:
    rect_vals: List[float] = []
    dom_vals: List[float] = []
    lin_vals: List[float] = []
    for o in obs:
        r2, t2, z2 = _rotate_components(o.r, o.t, o.z, azimuth_deg, tilt_deg)
        feat = extract_polarization_features(z2, r2, t2)
        rect_vals.append(float(np.clip(feat.rectilinearity, 0.0, 1.0)))
        dom_vals.append(float(np.clip(feat.dominant_energy_ratio, 0.0, 1.0)))
        lin_vals.append(float(np.clip(feat.linearity, 0.0, 1.0)))
    if not rect_vals:
        return {
            "rectilinearity_mean": float("nan"),
            "dominant_energy_ratio_mean": float("nan"),
            "linearity_mean": float("nan"),
        }
    return {
        "rectilinearity_mean": float(np.mean(np.asarray(rect_vals, dtype=float))),
        "dominant_energy_ratio_mean": float(np.mean(np.asarray(dom_vals, dtype=float))),
        "linearity_mean": float(np.mean(np.asarray(lin_vals, dtype=float))),
    }


def _objective(
    obs: List[OrientationObservation],
    azimuth_deg: float,
    tilt_deg: float,
    position_corr: np.ndarray,
    time_shift_sec: float,
    depth_override: Optional[float],
    w_tt: float,
    w_pol: float,
    w_sym: float,
    scales: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    max_shift_sec = 1.0
    j_tt = _travel_misfit(
        obs,
        depth_override=depth_override,
        position_corr=position_corr,
        time_shift_sec=float(time_shift_sec),
    )
    j_pol, j_sym_shape, j_energy = _waveform_misfit(obs, azimuth_deg=azimuth_deg, tilt_deg=tilt_deg)
    j_sym = float(j_sym_shape + j_energy)
    # Keep position correction and global time-shift stable.
    j_pos = float(np.mean(np.asarray(position_corr, dtype=float) ** 2))
    j_shift = float((float(time_shift_sec) / max(1e-6, max_shift_sec)) ** 2)
    s_tt = float(max(1e-9, abs(float(scales.get("J_tt", 1.0))))) if scales else 1.0
    s_pol = float(max(1e-9, abs(float(scales.get("J_pol", 1.0))))) if scales else 1.0
    s_sym = float(max(1e-9, abs(float(scales.get("J_sym", 1.0))))) if scales else 1.0
    jn_tt = j_tt / s_tt
    jn_pol = j_pol / s_pol
    jn_sym = j_sym / s_sym
    total = float(w_tt * jn_tt + w_pol * jn_pol + w_sym * jn_sym + 1e-4 * j_pos + 0.2 * j_shift)
    return total, {
        "J_tt": j_tt,
        "J_pol": j_pol,
        "J_sym": j_sym,
        "J_sym_shape": float(j_sym_shape),
        "J_energy": float(j_energy),
        "J_pos": j_pos,
        "J_shift": j_shift,
        "J_tt_n": jn_tt,
        "J_pol_n": jn_pol,
        "J_sym_n": jn_sym,
        "scale_tt": s_tt,
        "scale_pol": s_pol,
        "scale_sym": s_sym,
        "time_shift_sec": float(time_shift_sec),
    }


def run_orientation_correction(inp: OrientationCorrectionInput) -> OrientationCorrectionResult:
    obs = list(inp.observations or [])
    if len(obs) == 0:
        return OrientationCorrectionResult(
            success=False,
            azimuth_deg=float(inp.initial_azimuth_deg),
            tilt_deg=float(inp.initial_tilt_deg),
            position_correction=tuple(float(v) for v in inp.initial_position_correction),
            objective=float("inf"),
            iterations=0,
            message="缺少观测数据，无法执行姿态校正",
        )

    az = float(inp.initial_azimuth_deg)
    tilt = float(inp.initial_tilt_deg)
    pos = np.asarray(inp.initial_position_correction, dtype=float).copy()
    if pos.size != 3:
        pos = np.zeros(3, dtype=float)
    src_depth_history: List[float] = []
    iter_history: List[Dict[str, float]] = []

    # Reasonable initial search scales (coordinate unit follows input coordinates).
    scale_ref = float(np.median([np.linalg.norm(o.source_xyz - o.receiver_xyz) for o in obs])) if obs else 1000.0
    pos_step_xy = max(10.0, 0.03 * scale_ref)
    pos_step_z = max(2.0, 0.01 * scale_ref)
    az_step = 8.0
    tilt_step = 5.0

    best_obj = float("inf")
    best_parts: Dict[str, float] = {}
    depth_override: Optional[float] = None
    time_shift = 0.0
    iters = max(1, int(inp.max_iterations))
    scales: Optional[Dict[str, float]] = None

    for _it in range(iters):
        if inp.progress_callback is not None:
            try:
                inp.progress_callback(int(_it), int(iters), "正在更新水深并搜索最优参数...")
            except Exception:
                pass
        # Depth can be updated each outer iteration.
        if inp.depth_sampler is not None:
            sampled = None
            # Prefer explicit source XY (utm/geo) if present.
            xy_vals = [o.source_xy_utm for o in obs if o.source_xy_utm is not None]
            if not xy_vals:
                xy_vals = [o.source_xy_geo for o in obs if o.source_xy_geo is not None]
            if xy_vals:
                arr = np.asarray(xy_vals, dtype=float)
                sampled = inp.depth_sampler(float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))
            if sampled is None or not np.isfinite(float(sampled)):
                sampled = inp.depth_sampler(
                    float(np.median([o.source_xyz[0] for o in obs])),
                    float(np.median([o.source_xyz[1] for o in obs])),
                )
            if sampled is not None and np.isfinite(float(sampled)):
                depth_override = float(sampled)
                src_depth_history.append(depth_override)
        if scales is None:
            _, base_parts = _objective(
                obs=obs,
                azimuth_deg=float(az),
                tilt_deg=float(tilt),
                position_corr=np.asarray(pos, dtype=float),
                time_shift_sec=float(time_shift),
                depth_override=depth_override,
                w_tt=float(inp.w_tt),
                w_pol=float(inp.w_pol),
                w_sym=float(inp.w_sym),
                scales=None,
            )
            scales = {
                "J_tt": float(max(1e-9, abs(float(base_parts.get("J_tt", 1.0))))),
                "J_pol": float(max(1e-9, abs(float(base_parts.get("J_pol", 1.0))))),
                "J_sym": float(max(1e-9, abs(float(base_parts.get("J_sym", 1.0))))),
            }

        tshift_step = max(0.05, 0.30 * (0.55 ** _it))
        candidates: List[Tuple[float, float, np.ndarray, float]] = []
        for da in (-az_step, -0.5 * az_step, 0.0, 0.5 * az_step, az_step):
            for dt in (-tilt_step, -0.5 * tilt_step, 0.0, 0.5 * tilt_step, tilt_step):
                for dx in (-pos_step_xy, 0.0, pos_step_xy):
                    for dy in (-pos_step_xy, 0.0, pos_step_xy):
                        for dz in (-pos_step_z, 0.0, pos_step_z):
                            for ds in (-tshift_step, 0.0, tshift_step):
                                candidates.append(
                                    (
                                        az + da,
                                        tilt + dt,
                                        pos + np.array([dx, dy, dz], dtype=float),
                                        float(np.clip(time_shift + ds, -1.0, 1.0)),
                                    )
                                )

        local_best = None
        local_best_obj = float("inf")
        local_best_parts: Dict[str, float] = {}
        for caz, ctilt, cpos, cshift in candidates:
            obj, parts = _objective(
                obs=obs,
                azimuth_deg=float(caz),
                tilt_deg=float(ctilt),
                position_corr=np.asarray(cpos, dtype=float),
                time_shift_sec=float(cshift),
                depth_override=depth_override,
                w_tt=float(inp.w_tt),
                w_pol=float(inp.w_pol),
                w_sym=float(inp.w_sym),
                scales=scales,
            )
            if obj < local_best_obj:
                local_best_obj = obj
                local_best = (float(caz), float(ctilt), np.asarray(cpos, dtype=float), float(cshift))
                local_best_parts = parts

        if local_best is None:
            break
        az, tilt, pos, time_shift = local_best
        best_obj = local_best_obj
        best_parts = dict(local_best_parts)
        pol_q = _polarization_quality(obs=obs, azimuth_deg=float(az), tilt_deg=float(tilt))
        iter_history.append(
            {
                "iter": float(len(iter_history) + 1),
                "objective": float(best_obj),
                "J_tt": float(best_parts.get("J_tt", np.nan)),
                "J_pol": float(best_parts.get("J_pol", np.nan)),
                "J_sym": float(best_parts.get("J_sym", np.nan)),
                "J_sym_shape": float(best_parts.get("J_sym_shape", np.nan)),
                "J_energy": float(best_parts.get("J_energy", np.nan)),
                "J_tt_n": float(best_parts.get("J_tt_n", np.nan)),
                "J_pol_n": float(best_parts.get("J_pol_n", np.nan)),
                "J_sym_n": float(best_parts.get("J_sym_n", np.nan)),
                "azimuth_deg": float(az),
                "tilt_deg": float(tilt),
                "dx": float(pos[0]),
                "dy": float(pos[1]),
                "dz": float(pos[2]),
                "time_shift_sec": float(time_shift),
                "rectilinearity_mean": float(pol_q.get("rectilinearity_mean", np.nan)),
                "dominant_energy_ratio_mean": float(pol_q.get("dominant_energy_ratio_mean", np.nan)),
                "linearity_mean": float(pol_q.get("linearity_mean", np.nan)),
            }
        )
        if inp.progress_callback is not None:
            try:
                inp.progress_callback(int(_it + 1), int(iters), "当前轮迭代完成")
            except Exception:
                pass

        # Shrink steps to refine.
        az_step *= 0.55
        tilt_step *= 0.55
        pos_step_xy *= 0.55
        pos_step_z *= 0.55

    # Normalize azimuth.
    az = (float(az) + 180.0) % 360.0 - 180.0
    ok = np.isfinite(best_obj) and best_obj < 1e8
    msg = "姿态校正完成" if ok else "姿态校正未收敛"
    return OrientationCorrectionResult(
        success=bool(ok),
        azimuth_deg=float(az),
        tilt_deg=float(tilt),
        position_correction=(float(pos[0]), float(pos[1]), float(pos[2])),
        objective=float(best_obj),
        iterations=int(iters),
        source_depth_history=src_depth_history,
        iteration_history=iter_history,
        details=best_parts,
        message=msg,
    )

