"""
Joint OBS attitude correction using 3C waveforms + travel-time constraints.

波形极化对齐 ppol（Scholz et al.）：
  多炮方位一致性（主约束）：
    每炮 ORI_i = (BAZ_th_i(pos) − BAZ_PCA_i) mod 360
    先消 180° 模糊使 ORI 聚团，再最小化圆离散度；
    候选 az 贴合该圆均值。位置改正通过 BAZ_th 改变各 ORI_i。
  倾角（可选 correct_tilt）：INC_obs vs INC_th = atan(offset/depth)
  横切能量：几何 R/T 上 T≈0（质量项，并入 w_pol）

波形对称（非 ppol，默认关闭 w_sym=0）：
  可选 Z 偶 / R 奇镜像先验。

震相约定（按 apick / pick_word）：
  - pick_word == 1：直达水波 → 走时/位置 + 姿态；倾角可用 atan(x/h)
  - pick_word != 1：折射/反射等次生相 → 默认仅参与姿态（极化/ORI）
  - 同时存在时：走时与倾角理论角仅用直达；方位极化可用全部 V 段
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import math
import numpy as np

from .polarization_features import extract_polarization_features


DepthSampler = Callable[[float, float], Optional[float]]
ProgressCallback = Callable[[int, int, str], None]

# apick=1 约定为直达水波；其它字为折射/反射等次生相
DIRECT_WATER_PICK_WORD = 1


def is_direct_water_phase(pick_word: int) -> bool:
    """是否直达水波震相（apick=1）。"""
    return int(pick_word) == int(DIRECT_WATER_PICK_WORD)


def split_observations_by_phase(
    obs: List["OrientationObservation"],
) -> Tuple[List["OrientationObservation"], List["OrientationObservation"]]:
    """拆成 (直达水波, 次生相)。"""
    direct: List[OrientationObservation] = []
    secondary: List[OrientationObservation] = []
    for o in obs or []:
        if is_direct_water_phase(int(getattr(o, "pick_word", 1))):
            direct.append(o)
        else:
            secondary.append(o)
    return direct, secondary


@dataclass
class PhaseCorrectionPolicy:
    """按观测震相组成解析后的校正策略。"""

    n_direct: int
    n_secondary: int
    w_tt: float
    correct_tilt: bool
    allow_position: bool
    allow_time_shift: bool
    mode_key: str
    message: str


def _secondary_apick_summary(secondary: List["OrientationObservation"]) -> str:
    """次生相按 apick 计数，如 apick2:20, apick3:14。"""
    counts: Dict[int, int] = {}
    for o in secondary:
        pw = int(getattr(o, "pick_word", 0) or 0)
        counts[pw] = int(counts.get(pw, 0)) + 1
    if not counts:
        return ""
    parts = [f"apick{k}:{counts[k]}" for k in sorted(counts.keys())]
    return ", ".join(parts)


def resolve_phase_policy(
    obs: List["OrientationObservation"],
    *,
    w_tt: float,
    correct_tilt: bool,
) -> PhaseCorrectionPolicy:
    """按 apick 组成决定走时/位置/倾角是否启用。"""
    direct, secondary = split_observations_by_phase(obs)
    n_d, n_s = len(direct), len(secondary)
    sec_txt = _secondary_apick_summary(secondary)
    if n_d > 0 and n_s > 0:
        mode = "mixed"
        msg = (
            f"混合震相：直达 {n_d} 段(apick=1) + 次生 {n_s} 段"
            f"{f'（{sec_txt}）' if sec_txt else ''}；"
            f"走时/位置/倾角理论角仅用直达，方位极化用全部。"
            f"若未故意选次生相，请切换 apick 查看/清除其它字下的 V 段"
        )
        return PhaseCorrectionPolicy(
            n_direct=n_d,
            n_secondary=n_s,
            w_tt=float(max(0.0, w_tt)),
            correct_tilt=bool(correct_tilt),
            allow_position=True,
            allow_time_shift=True,
            mode_key=mode,
            message=msg,
        )
    if n_d > 0:
        mode = "direct"
        msg = f"直达水波模式：{n_d} 段(apick=1)；走时/位置 + 姿态（倾角可选）"
        return PhaseCorrectionPolicy(
            n_direct=n_d,
            n_secondary=0,
            w_tt=float(max(0.0, w_tt)),
            correct_tilt=bool(correct_tilt),
            allow_position=True,
            allow_time_shift=True,
            mode_key=mode,
            message=msg,
        )
    # 仅次生相：强制仅姿态
    mode = "secondary_attitude"
    msg = (
        f"次生相模式：{n_s} 段"
        f"{f'（{sec_txt}）' if sec_txt else ''}；仅姿态（极化/ORI），"
        f"不启用走时/位置/倾角几何（请用 apick=1 标直达）"
    )
    return PhaseCorrectionPolicy(
        n_direct=0,
        n_secondary=n_s,
        w_tt=0.0,
        correct_tilt=False,
        allow_position=False,
        allow_time_shift=False,
        mode_key=mode,
        message=msg,
    )


def rotate_components(
    r: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    az_deg: float,
    tilt_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将仪器三分量 (R,T,Z) 旋到校正后的 (R',T',Z')。

    约定：
      1) 水平面绕 Z 转方位 az（只动 R/T）
            R1 =  cos(az)*R + sin(az)*T
            T' = -sin(az)*R + cos(az)*T
      2) 绕 T' 转倾角 tilt（R1–Z 平面）
            R' =  cos(tilt)*R1 + sin(tilt)*Z
            Z' = -sin(tilt)*R1 + cos(tilt)*Z

    **tilt=0 ⇒ Z'=Z（样点恒等）**；仅 R/T 随方位变化。
    """
    r = np.asarray(r, dtype=float)
    t = np.asarray(t, dtype=float)
    z = np.asarray(z, dtype=float)
    az = np.deg2rad(float(az_deg))
    tilt = np.deg2rad(float(tilt_deg))
    c_az, s_az = float(np.cos(az)), float(np.sin(az))
    r1 = c_az * r + s_az * t
    t2 = -s_az * r + c_az * t
    c_t, s_t = float(np.cos(tilt)), float(np.sin(tilt))
    r2 = c_t * r1 + s_t * z
    z2 = -s_t * r1 + c_t * z
    return r2, t2, z2


# 兼容旧内部名
_rotate_components = rotate_components


def _wrap_deg(a: float) -> float:
    return float(a) % 360.0


def _angle_diff_deg(a: float, b: float) -> float:
    """有符号角差，结果 ∈ (-180, 180]."""
    return (_wrap_deg(float(a) - float(b) + 180.0) % 360.0) - 180.0


def _angle_diff_abs_deg(a: float, b: float, *, amb180: bool = False) -> float:
    """最小绝对角差；amb180=True 时允许 PCA 水平偏振 180° 模糊。"""
    d = abs(_angle_diff_deg(a, b))
    if amb180:
        d = min(d, abs(180.0 - d))
    return float(d)


def _circular_mean_deg(angles_deg: List[float], weights: Optional[List[float]] = None) -> float:
    """加权圆均值（度）。"""
    if not angles_deg:
        return 0.0
    ang = np.deg2rad(np.asarray(angles_deg, dtype=float))
    if weights is None:
        w = np.ones_like(ang)
    else:
        w = np.asarray(weights, dtype=float)
        if w.size != ang.size:
            w = np.ones_like(ang)
        w = np.maximum(w, 0.0)
        if float(np.sum(w)) <= 1e-12:
            w = np.ones_like(ang)
    s = float(np.sum(w * np.sin(ang)))
    c = float(np.sum(w * np.cos(ang)))
    return _wrap_deg(float(np.degrees(np.arctan2(s, c))))


def _align_ori_to_ref_amb180(ori_deg: float, ref_deg: float) -> float:
    """在 ori 与 ori+180 中选更接近 ref 的一支（水平 PCA 180° 模糊）。"""
    o0 = _wrap_deg(ori_deg)
    o1 = _wrap_deg(ori_deg + 180.0)
    if abs(_angle_diff_deg(o0, ref_deg)) <= abs(_angle_diff_deg(o1, ref_deg)):
        return o0
    return o1


def _cluster_ori_amb180(
    oris_deg: List[float],
    weights: Optional[List[float]] = None,
    *,
    n_iter: int = 4,
) -> Tuple[List[float], float]:
    """多炮 ORI 消 180° 后聚团，返回 (aligned_oris, circ_mean)。"""
    if not oris_deg:
        return [], 0.0
    aligned = [_wrap_deg(o) for o in oris_deg]
    mean = _circular_mean_deg(aligned, weights)
    for _ in range(max(1, int(n_iter))):
        aligned = [_align_ori_to_ref_amb180(o, mean) for o in oris_deg]
        mean = _circular_mean_deg(aligned, weights)
    return aligned, mean


def _geom_baz_deg(
    receiver_xy: np.ndarray,
    source_xy: np.ndarray,
    position_corr_xy: Optional[np.ndarray] = None,
) -> Optional[float]:
    """理论反方位角（ppol：台站→事件），含位置改正。

    坐标约定：x≈East、y≈North（UTM/局部直角）。位置改正加在接收点（OBS）上。
    """
    rec = np.asarray(receiver_xy[:2], dtype=float).copy()
    src = np.asarray(source_xy[:2], dtype=float)
    if position_corr_xy is not None:
        dxy = np.asarray(position_corr_xy[:2], dtype=float)
        if dxy.size >= 2 and np.all(np.isfinite(dxy)):
            rec = rec + dxy
    if not (np.all(np.isfinite(rec)) and np.all(np.isfinite(src))):
        return None
    d = src - rec
    if float(np.linalg.norm(d)) < 1e-9:
        return None
    # 地理方位：自北向东顺时针 ≡ atan2(East, North)
    return _wrap_deg(float(np.degrees(np.arctan2(d[0], d[1]))))


def _pca_baz_2d(data_1: np.ndarray, data_2: np.ndarray) -> Tuple[float, float, float]:
    """水平面 2D PCA（对齐 ppol）：返回 (BAZ_obs, POL_HOR, SNR_HOR)。"""
    x = np.asarray(data_1, dtype=float).reshape(-1)
    y = np.asarray(data_2, dtype=float).reshape(-1)
    n = int(min(x.size, y.size))
    if n < 3:
        return 0.0, 0.0, 0.0
    x = x[:n] - float(np.mean(x[:n]))
    y = y[:n] - float(np.mean(y[:n]))
    cov = np.cov(np.vstack([x, y]))
    if not np.all(np.isfinite(cov)):
        return 0.0, 0.0, 0.0
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(np.abs(vals))[::-1]
    vals = np.abs(np.asarray(vals[order], dtype=float))
    vecs = np.asarray(vecs[:, order], dtype=float)
    v1 = vecs[:, 0]
    baz = _wrap_deg(float(np.degrees(np.arctan2(v1[1].real, v1[0].real))))
    l1 = float(max(vals[0], 1e-12))
    l2 = float(max(vals[1], 1e-12)) if vals.size > 1 else 1e-12
    pol_hor = float(np.clip(1.0 - l2 / l1, 0.0, 1.0))  # Jurkevics 1988
    snr_hor = float((l1 - l2) / l2)  # De Meersman et al. 2006
    return baz, pol_hor, snr_hor


def _rotate_ne_rt(n: np.ndarray, e: np.ndarray, baz_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """N/E → R/T（与 ObsPy rotate_ne_rt 一致；baz 为台站→事件方位）。"""
    ba = np.deg2rad(float(baz_deg))
    n = np.asarray(n, dtype=float)
    e = np.asarray(e, dtype=float)
    r = -e * np.sin(ba) - n * np.cos(ba)
    t = -e * np.cos(ba) + n * np.sin(ba)
    return r, t


def _len_to_km(v: float) -> float:
    d = abs(float(v))
    return d / 1000.0 if d > 50.0 else d


def _geom_offset_depth_km(
    o: "OrientationObservation",
    position_corr: np.ndarray,
    depth_override: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """直达水波几何：水平偏移与有效水深（km），含位置改正。"""
    pos = np.asarray(position_corr, dtype=float)
    if pos.size != 3:
        pos = np.zeros(3, dtype=float)
    off_km = abs(float(o.offset_km))
    try:
        src_xy = np.asarray(o.source_xyz[:2], dtype=float)
        rec_xy = np.asarray(o.receiver_xyz[:2], dtype=float) + pos[:2]
        if np.all(np.isfinite(src_xy)) and np.all(np.isfinite(rec_xy)):
            geom = float(np.linalg.norm(rec_xy - src_xy))
            if np.isfinite(geom) and geom > 1e-9:
                off_km = _len_to_km(geom)
    except Exception:
        pass
    if off_km <= 1e-9:
        return None, None
    if depth_override is None or not np.isfinite(float(depth_override)) or float(depth_override) <= 0.0:
        return float(off_km), None
    depth_eff_km = max(1e-6, float(depth_override) + _len_to_km(float(pos[2])))
    return float(off_km), float(depth_eff_km)


def _geom_inc_deg(offset_km: float, depth_km: float) -> float:
    """几何入射角（自垂直量起，度）：直达水波直线射线 atan(offset/depth)。"""
    return float(np.degrees(np.arctan2(max(0.0, float(offset_km)), max(1e-6, float(depth_km)))))


def _pca_inc_rz(data_z: np.ndarray, data_r: np.ndarray) -> Tuple[float, float]:
    """R–Z 平面 2D PCA（对齐 ppol INC）：返回 (INC_obs_deg, POL_RZ)。

    协方差顺序 (Z, R)；INC = atan(v_R / v_Z)，自垂直量起；符号用于 180° 消歧。
    """
    z = np.asarray(data_z, dtype=float).reshape(-1)
    r = np.asarray(data_r, dtype=float).reshape(-1)
    n = int(min(z.size, r.size))
    if n < 3:
        return 0.0, 0.0
    z = z[:n] - float(np.mean(z[:n]))
    r = r[:n] - float(np.mean(r[:n]))
    cov = np.cov(np.vstack([z, r]))
    if not np.all(np.isfinite(cov)):
        return 0.0, 0.0
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(np.abs(vals))[::-1]
    vals = np.abs(np.asarray(vals[order], dtype=float))
    vecs = np.asarray(vecs[:, order], dtype=float)
    v1 = vecs[:, 0]
    vz, vr = float(v1[0].real), float(v1[1].real)
    if abs(vz) < 1e-15 and abs(vr) < 1e-15:
        return 0.0, 0.0
    inc = float(np.degrees(np.arctan2(vr, vz)))
    l1 = float(max(vals[0], 1e-12))
    l2 = float(max(vals[1], 1e-12)) if vals.size > 1 else 1e-12
    pol_rz = float(np.clip(1.0 - l2 / l1, 0.0, 1.0))
    return inc, pol_rz


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
class PpolTraceResult:
    """单道 ppol 中间量（用于结果分布图）。"""

    trace_idx: int
    offset_km: float
    baz_obs_deg: float
    baz_th_deg: float
    ori_raw_deg: float
    ori_aligned_deg: float
    pol_hor: float
    snr_hor: float
    residual_to_az_deg: float
    residual_to_mean_deg: float
    t_energy_ratio: float
    r_energy_ratio: float = float("nan")
    z_energy_ratio: float = float("nan")
    valid: bool = True


@dataclass
class PpolTraceSummary:
    """多道 ppol 分布与统计。"""

    rows: List[PpolTraceResult]
    azimuth_deg: float
    ori_circ_mean_deg: float
    ori_circ_std_deg: float
    residual_mae_deg: float
    residual_rms_deg: float
    pol_hor_mean: float
    n_valid: int
    n_total: int
    message: str = ""
    # 呈现用：μ 对齐到解 az 后的一支，及 |az−μ|（最短 / 含180°模糊）
    ori_mean_aligned_to_az_deg: float = float("nan")
    az_minus_ori_shortest_deg: float = float("nan")
    az_minus_ori_amb180_deg: float = float("nan")


def compute_ppol_trace_results(
    observations: List[OrientationObservation],
    *,
    azimuth_deg: float,
    tilt_deg: float = 0.0,
    position_correction: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> PpolTraceSummary:
    """按当前解计算每道 ppol：BAZ_PCA / BAZ_th / ORI / POL_HOR / 残差。"""
    pos = np.asarray(position_correction, dtype=float)
    if pos.size != 3:
        pos = np.zeros(3, dtype=float)
    az = float(azimuth_deg)
    tilt = float(tilt_deg)

    raw_rows: List[Dict[str, float]] = []
    oris: List[float] = []
    ori_w: List[float] = []

    for o in observations or []:
        baz_obs, pol_hor, snr_hor = _pca_baz_2d(o.r, o.t)
        baz_th = _geom_baz_deg(o.receiver_xyz, o.source_xyz, position_corr_xy=pos[:2])
        n_c, e_c, z_c = _rotate_components(o.r, o.t, o.z, az, tilt)
        if baz_th is None:
            e_r = float(np.mean(n_c * n_c))
            e_t = float(np.mean(e_c * e_c))
            e_z = float(np.mean(z_c * z_c))
            e_tot = max(e_r + e_t + e_z, 1e-12)
            raw_rows.append(
                {
                    "trace_idx": float(o.trace_idx),
                    "offset_km": float(o.offset_km),
                    "baz_obs": float(baz_obs),
                    "baz_th": float("nan"),
                    "ori_raw": float("nan"),
                    "pol_hor": float(pol_hor),
                    "snr_hor": float(snr_hor),
                    "t_ratio": float(e_t / e_tot),
                    "r_ratio": float(e_r / e_tot),
                    "z_ratio": float(e_z / e_tot),
                    "valid": 0.0,
                }
            )
            continue

        ori_raw = _wrap_deg(float(baz_th) - float(baz_obs))
        r_g, t_g = _rotate_ne_rt(n_c, e_c, float(baz_th))
        e_r = float(np.mean(r_g * r_g))
        e_t = float(np.mean(t_g * t_g))
        e_z = float(np.mean(z_c * z_c))
        e_tot = max(e_r + e_t + e_z, 1e-12)
        w_i = float(max(0.05, pol_hor))
        oris.append(ori_raw)
        ori_w.append(w_i)
        raw_rows.append(
            {
                "trace_idx": float(o.trace_idx),
                "offset_km": float(o.offset_km),
                "baz_obs": float(baz_obs),
                "baz_th": float(baz_th),
                "ori_raw": float(ori_raw),
                "pol_hor": float(pol_hor),
                "snr_hor": float(snr_hor),
                "t_ratio": float(e_t / e_tot),
                "r_ratio": float(e_r / e_tot),
                "z_ratio": float(e_z / e_tot),
                "valid": 1.0,
            }
        )

    if oris:
        aligned_mu, ori_mean = _cluster_ori_amb180(oris, ori_w)
        aligned_az = [_align_ori_to_ref_amb180(o, az) for o in oris]
        ori_mean_az = _align_ori_to_ref_amb180(ori_mean, az)
        w_arr = np.asarray(ori_w, dtype=float)
        w_arr = w_arr / max(float(np.sum(w_arr)), 1e-12)
        d_scatter = np.asarray([_angle_diff_deg(a, ori_mean) for a in aligned_mu], dtype=float)
        ori_std = float(np.sqrt(np.sum(w_arr * d_scatter * d_scatter)))
        d_short = float(abs(_angle_diff_deg(az, ori_mean)))
        d_amb = float(abs(_angle_diff_deg(az, ori_mean_az)))
    else:
        aligned_mu, aligned_az = [], []
        ori_mean = ori_mean_az = ori_std = float("nan")
        d_short = d_amb = float("nan")

    rows: List[PpolTraceResult] = []
    residuals: List[float] = []
    pol_list: List[float] = []
    ai = 0
    for rr in raw_rows:
        valid = bool(rr["valid"] > 0.5)
        if valid and ai < len(aligned_az) and ai < len(aligned_mu):
            # 图/残差相对解 az：每道 ORI 先消 180° 对齐到 az，避免 μ≈az+180 时残差虚高
            ori_al = float(aligned_az[ai])
            res_az = float(_angle_diff_deg(ori_al, az))
            res_mu = float(_angle_diff_deg(float(aligned_mu[ai]), ori_mean))
            residuals.append(res_az)
            pol_list.append(float(rr["pol_hor"]))
            ai += 1
        else:
            ori_al = float("nan")
            res_az = float("nan")
            res_mu = float("nan")
        rows.append(
            PpolTraceResult(
                trace_idx=int(rr["trace_idx"]),
                offset_km=float(rr["offset_km"]),
                baz_obs_deg=float(rr["baz_obs"]),
                baz_th_deg=float(rr["baz_th"]),
                ori_raw_deg=float(rr["ori_raw"]),
                ori_aligned_deg=ori_al,
                pol_hor=float(rr["pol_hor"]),
                snr_hor=float(rr["snr_hor"]),
                residual_to_az_deg=res_az,
                residual_to_mean_deg=res_mu,
                t_energy_ratio=float(rr["t_ratio"]),
                r_energy_ratio=float(rr.get("r_ratio", float("nan"))),
                z_energy_ratio=float(rr.get("z_ratio", float("nan"))),
                valid=valid,
            )
        )

    n_valid = int(sum(1 for r in rows if r.valid))
    if residuals:
        res_arr = np.asarray(residuals, dtype=float)
        mae = float(np.mean(np.abs(res_arr)))
        rms = float(np.sqrt(np.mean(res_arr * res_arr)))
        pol_mean = float(np.mean(np.asarray(pol_list, dtype=float))) if pol_list else float("nan")
        msg = (
            f"有效道 {n_valid}/{len(rows)} | "
            f"解方位 az={az:.2f}°（旋分量用此值＝「校正后」） | "
            f"数据估方位 ORI圆均值 μ={ori_mean:.2f}°"
            f"（对齐到az后 μ′={ori_mean_az:.2f}°） | "
            f"|az−μ|最短={d_short:.2f}°，含180°模糊={d_amb:.2f}° | "
            f"圆标准差 σ={ori_std:.2f}°（多炮离散） | "
            f"相对az残差 MAE={mae:.2f}° RMS={rms:.2f}° | "
            f"POL_HOR均值={pol_mean:.3f}"
            f"。说明：「原方位」仅为校正前初值假定（默认0°）；"
            f"μ 与 az 应接近（PCA 可差约180°，请看含模糊偏差）。"
        )
    else:
        mae = rms = pol_mean = float("nan")
        msg = f"有效道 0/{len(rows)}：无法计算 BAZ_th/ORI（检查几何坐标）"

    return PpolTraceSummary(
        rows=rows,
        azimuth_deg=float(az),
        ori_circ_mean_deg=float(ori_mean),
        ori_circ_std_deg=float(ori_std),
        residual_mae_deg=float(mae),
        residual_rms_deg=float(rms),
        pol_hor_mean=float(pol_mean),
        n_valid=n_valid,
        n_total=int(len(rows)),
        message=msg,
        ori_mean_aligned_to_az_deg=float(ori_mean_az),
        az_minus_ori_shortest_deg=float(d_short),
        az_minus_ori_amb180_deg=float(d_amb),
    )


@dataclass
class OrientationCorrectionInput:
    observations: List[OrientationObservation]
    initial_azimuth_deg: float = 0.0
    initial_tilt_deg: float = 0.0
    initial_position_correction: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # 用户预设全局走时 shift（正=观测加走时/变晚，负=减走时/变早）；
    # 作用在观测侧：t_obs' = t0 + shift；最优值 ≈ 残差(预测-观测)
    initial_time_shift_sec: float = 0.0
    depth_sampler: Optional[DepthSampler] = None
    max_iterations: int = 4
    w_tt: float = 0.15
    w_pol: float = 1.0
    # 非 ppol 窗对称；对齐 ppol 时默认关闭
    w_sym: float = 0.0
    # 默认不校正倾角：固定为 initial_tilt_deg（通常为 0）
    correct_tilt: bool = False
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


def _travel_misfit(
    obs: List[OrientationObservation],
    depth_override: Optional[float],
    position_corr: Optional[np.ndarray] = None,
    time_shift_sec: float = 0.0,
) -> float:
    # 直达水波公式（仅 apick=1）：
    # t_pred = sqrt(offset_km^2 + water_depth_km^2) / 1.5
    # 次生相不进入走时项。无水深 -> 无法约束走时。
    if depth_override is None or not np.isfinite(depth_override) or depth_override <= 0.0:
        return 1e9
    water_v = 1.5  # km/s
    misfits: List[float] = []
    pos = np.asarray(position_corr if position_corr is not None else np.zeros(3, dtype=float), dtype=float)
    if pos.size != 3:
        pos = np.zeros(3, dtype=float)

    for o in obs:
        if not is_direct_water_phase(int(getattr(o, "pick_word", DIRECT_WATER_PICK_WORD))):
            continue
        off_km, depth_eff_km = _geom_offset_depth_km(o, pos, depth_override)
        if off_km is None or depth_eff_km is None:
            continue
        slant_km = float(np.sqrt(off_km * off_km + depth_eff_km * depth_eff_km))
        # 走时 shift 作用在观测侧：t_obs' = t0 + time_shift
        # 残差定义 residual = t_pred - t0；最优 time_shift ≈ residual
        t_pred = slant_km / water_v
        misfits.append((t_pred - (float(o.t0) + float(time_shift_sec))) / max(float(o.dt), 1e-5))
    if not misfits:
        return 1e9
    arr = np.asarray(misfits, dtype=float)
    return float(np.mean(arr * arr))


def _waveform_misfit(
    obs: List[OrientationObservation],
    azimuth_deg: float,
    tilt_deg: float,
    position_corr: Optional[np.ndarray] = None,
    depth_override: Optional[float] = None,
    correct_tilt: bool = False,
) -> Tuple[float, float, float, float, Dict[str, float]]:
    """波形残差（对齐 ppol；多炮 ORI 一致性为主）。

    返回 (J_pol_az, J_sym_shape, J_t_energy, J_inc, extras)。

    方位（ppol 多炮）：
      ORI_i = (BAZ_th_i(pos) − BAZ_obs_i)；消 180° 后求圆均值 μ；
      J = 各炮相对 μ 的圆离散（多炮一致）+ |az−μ|（候选方位贴合）
      + 弱质量项 (1−POL_HOR)。位置通过 BAZ_th 进入各 ORI_i。

    倾角 / 横切 / 对称：同前；对称默认权重 0。
    """
    sym_scores: List[float] = []
    energy_scores: List[float] = []
    inc_scores: List[float] = []
    oris: List[float] = []
    ori_w: List[float] = []
    pos = np.asarray(position_corr if position_corr is not None else np.zeros(3), dtype=float)
    if pos.size != 3:
        pos = np.zeros(3, dtype=float)

    def _mirror_pair_loss(arr: np.ndarray, odd: bool) -> float:
        """以窗中心（拾取）为镜：偶对称 left≈right；奇对称 left≈-right。"""
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
        baz_obs, pol_hor, _snr_hor = _pca_baz_2d(o.r, o.t)
        baz_th = _geom_baz_deg(o.receiver_xyz, o.source_xyz, position_corr_xy=pos)
        w_i = float(max(0.05, pol_hor))  # 直线度作多炮加权

        if baz_th is None:
            if correct_tilt and is_direct_water_phase(int(getattr(o, "pick_word", DIRECT_WATER_PICK_WORD))):
                inc_scores.append(1.0)
            r2, t2, z2 = _rotate_components(o.r, o.t, o.z, azimuth_deg, tilt_deg)
            e_r = float(np.mean(r2 * r2))
            e_t = float(np.mean(t2 * t2))
            e_z = float(np.mean(z2 * z2))
            energy_scores.append(float(e_t / max(e_r + e_t + e_z, 1e-12)))
            sym_scores.append(0.5 * (_mirror_pair_loss(z2, odd=False) + _mirror_pair_loss(r2, odd=True)))
            continue

        oris.append(_wrap_deg(float(baz_th) - float(baz_obs)))
        ori_w.append(w_i)

        n_c, e_c, z_c = _rotate_components(o.r, o.t, o.z, azimuth_deg, tilt_deg)
        r_g, t_g = _rotate_ne_rt(n_c, e_c, float(baz_th))
        e_r = float(np.mean(r_g * r_g))
        e_t = float(np.mean(t_g * t_g))
        e_z = float(np.mean(z_c * z_c))
        t_ratio = e_t / max(e_r + e_t + e_z, 1e-12)
        energy_scores.append(float(max(0.0, t_ratio)))
        sym_scores.append(0.5 * (_mirror_pair_loss(z_c, odd=False) + _mirror_pair_loss(r_g, odd=True)))

        if correct_tilt:
            # 倾角几何 INC_th=atan(x/h) 仅对直达水波成立；次生相不进 tilt 项
            if not is_direct_water_phase(int(getattr(o, "pick_word", DIRECT_WATER_PICK_WORD))):
                continue
            inc_obs, pol_rz = _pca_inc_rz(z_c, r_g)
            if float(inc_obs) < 0.0:
                r_flip, _t_flip = _rotate_ne_rt(n_c, e_c, float(baz_th) + 180.0)
                inc_obs, pol_rz = _pca_inc_rz(z_c, r_flip)
            inc_obs = abs(float(inc_obs))
            off_km, depth_km = _geom_offset_depth_km(o, pos, depth_override)
            if off_km is not None and depth_km is not None:
                inc_th = _geom_inc_deg(off_km, depth_km)
                d_inc = abs(float(inc_obs) - float(inc_th))
                inc_loss = (d_inc / 45.0) ** 2 + 0.25 * (1.0 - float(pol_rz))
            else:
                inc_loss = 0.25 * (1.0 - float(pol_rz))
            inc_scores.append(float(np.clip(inc_loss, 0.0, 2.0)))

    extras: Dict[str, float] = {
        "n_ori": float(len(oris)),
        "ori_circ_mean_deg": float("nan"),
        "ori_circ_std_deg": float("nan"),
    }
    if not oris and not energy_scores:
        return 1e9, 1e9, 1e9, 1e9, extras

    # --- 多炮 ORI 一致性（ppol 主项）---
    if len(oris) >= 1:
        aligned, ori_mean = _cluster_ori_amb180(oris, ori_w)
        # 亦对齐到候选 az，保证搜索 az 与聚团一致
        aligned_az = [_align_ori_to_ref_amb180(o, float(azimuth_deg)) for o in oris]
        w_arr = np.asarray(ori_w, dtype=float)
        w_arr = w_arr / max(float(np.sum(w_arr)), 1e-12)
        # 多炮离散：各 ORI 相对圆均值
        d_scatter = np.asarray(
            [_angle_diff_deg(a, ori_mean) for a in aligned], dtype=float
        )
        j_multi = float(np.sum(w_arr * (d_scatter / 90.0) ** 2))
        # 候选方位贴合圆均值
        j_az = float((_angle_diff_deg(float(azimuth_deg), ori_mean) / 90.0) ** 2)
        # 质量：直线度不足则惩罚（已体现在权重，再加弱项）
        j_q = float(np.sum(w_arr * (1.0 - np.minimum(np.asarray(ori_w, dtype=float), 1.0))))
        # 与 az 对齐后的离散（位置错时 ORI 随炮方位散开，此项大）
        d_to_az = np.asarray(
            [_angle_diff_deg(a, float(azimuth_deg)) for a in aligned_az], dtype=float
        )
        j_fit = float(np.sum(w_arr * (d_to_az / 90.0) ** 2))
        j_pol_az = float(np.clip(0.6 * j_multi + 0.3 * j_fit + 0.1 * j_az + 0.15 * j_q, 0.0, 3.0))
        extras["ori_circ_mean_deg"] = float(ori_mean)
        extras["ori_circ_std_deg"] = float(np.sqrt(np.sum(w_arr * d_scatter * d_scatter)))
    else:
        j_pol_az = 1.0

    j_inc = float(np.mean(inc_scores)) if inc_scores else 0.0
    j_sym = float(np.mean(sym_scores)) if sym_scores else 0.0
    j_e = float(np.mean(energy_scores)) if energy_scores else 0.0
    return j_pol_az, j_sym, j_e, float(j_inc), extras


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
    prior_time_shift_sec: float = 0.0,
    correct_tilt: bool = False,
) -> Tuple[float, Dict[str, float]]:
    max_shift_sec = 1.0
    j_tt = _travel_misfit(
        obs,
        depth_override=depth_override,
        position_corr=position_corr,
        time_shift_sec=float(time_shift_sec),
    )
    j_pol, j_sym_shape, j_energy, j_inc, wf_extras = _waveform_misfit(
        obs,
        azimuth_deg=azimuth_deg,
        tilt_deg=tilt_deg,
        position_corr=position_corr,
        depth_override=depth_override,
        correct_tilt=bool(correct_tilt),
    )
    # ppol 侧：多炮方位一致 +（可选）入射角 + 横切能量 → w_pol
    j_pol_tot = float(j_pol + float(j_energy) + (j_inc if correct_tilt else 0.0))
    # 非 ppol：窗对称（默认 w_sym=0）
    j_sym = float(j_sym_shape)
    # Keep position correction and global time-shift stable (相对用户预设 prior 惩罚漂移).
    j_pos = float(np.mean(np.asarray(position_corr, dtype=float) ** 2))
    d_shift = float(time_shift_sec) - float(prior_time_shift_sec)
    j_shift = float((d_shift / max(1e-6, max_shift_sec)) ** 2)
    s_tt = float(max(1e-9, abs(float(scales.get("J_tt", 1.0))))) if scales else 1.0
    s_pol = float(max(1e-9, abs(float(scales.get("J_pol", 1.0))))) if scales else 1.0
    s_sym = float(max(1e-9, abs(float(scales.get("J_sym", 1.0))))) if scales else 1.0
    jn_tt = j_tt / s_tt
    jn_pol = j_pol_tot / s_pol
    jn_sym = j_sym / s_sym
    total = float(w_tt * jn_tt + w_pol * jn_pol + w_sym * jn_sym + 1e-4 * j_pos + 0.2 * j_shift)
    out = {
        "J_tt": j_tt,
        "J_pol": j_pol_tot,
        "J_pol_az": float(j_pol),
        "J_inc": float(j_inc),
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
        "prior_time_shift_sec": float(prior_time_shift_sec),
        "n_ori": float(wf_extras.get("n_ori", 0.0)),
        "ori_circ_mean_deg": float(wf_extras.get("ori_circ_mean_deg", float("nan"))),
        "ori_circ_std_deg": float(wf_extras.get("ori_circ_std_deg", float("nan"))),
    }
    return total, out


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

    policy = resolve_phase_policy(
        obs, w_tt=float(inp.w_tt), correct_tilt=bool(inp.correct_tilt)
    )
    w_tt_eff = float(policy.w_tt)
    correct_tilt_eff = bool(policy.correct_tilt)

    az = float(inp.initial_azimuth_deg)
    tilt = float(inp.initial_tilt_deg)
    pos = np.asarray(inp.initial_position_correction, dtype=float).copy()
    if pos.size != 3:
        pos = np.zeros(3, dtype=float)
    src_depth_history: List[float] = []
    iter_history: List[Dict[str, float]] = []

    # Reasonable initial search scales (coordinate unit follows input coordinates).
    scale_ref = float(np.median([np.linalg.norm(o.source_xyz - o.receiver_xyz) for o in obs])) if obs else 1000.0
    if bool(policy.allow_position):
        pos_step_xy = max(10.0, 0.03 * scale_ref)
        pos_step_z = max(2.0, 0.01 * scale_ref)
    else:
        pos_step_xy = 0.0
        pos_step_z = 0.0
    az_step = 8.0
    tilt_step = 5.0

    best_obj = float("inf")
    best_parts: Dict[str, float] = {}
    depth_override: Optional[float] = None
    prior_tt = float(inp.initial_time_shift_sec)
    time_shift = float(prior_tt)
    iters = max(1, int(inp.max_iterations))
    scales: Optional[Dict[str, float]] = None

    for _it in range(iters):
        if inp.progress_callback is not None:
            try:
                inp.progress_callback(
                    int(_it),
                    int(iters),
                    f"{policy.message}；正在更新水深并搜索最优参数...",
                )
            except Exception:
                pass
        # Depth can be updated each outer iteration.
        # 水深应在 OBS（receiver）处采样，不是炮点（source）。
        # 仅次生相时不必采样水深（走时/倾角几何均不用）。
        if inp.depth_sampler is not None and (policy.n_direct > 0):
            sampled = None
            rec_xy = np.asarray([o.receiver_xyz[:2] for o in obs], dtype=float)
            if rec_xy.size >= 2 and np.any(np.isfinite(rec_xy)):
                sampled = inp.depth_sampler(
                    float(np.nanmedian(rec_xy[:, 0])),
                    float(np.nanmedian(rec_xy[:, 1])),
                )
            if sampled is None or not np.isfinite(float(sampled)):
                # 兼容旧字段：仅作回退
                xy_vals = [o.source_xy_utm for o in obs if o.source_xy_utm is not None]
                if not xy_vals:
                    xy_vals = [o.source_xy_geo for o in obs if o.source_xy_geo is not None]
                if xy_vals:
                    arr = np.asarray(xy_vals, dtype=float)
                    sampled = inp.depth_sampler(
                        float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
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
                w_tt=float(w_tt_eff),
                w_pol=float(inp.w_pol),
                w_sym=float(inp.w_sym),
                scales=None,
                prior_time_shift_sec=float(prior_tt),
                correct_tilt=bool(correct_tilt_eff),
            )
            scales = {
                "J_tt": float(max(1e-9, abs(float(base_parts.get("J_tt", 1.0))))),
                "J_pol": float(max(1e-9, abs(float(base_parts.get("J_pol", 1.0))))),
                "J_sym": float(max(1e-9, abs(float(base_parts.get("J_sym", 1.0))))),
            }

        tshift_step = max(0.05, 0.30 * (0.55 ** _it)) if bool(policy.allow_time_shift) else 0.0
        # 默认不校正倾角：tilt 固定 0（勿沿用旧解大倾角，否则 Z 被 R 混叠）
        if bool(correct_tilt_eff):
            tilt_deltas = (-tilt_step, -0.5 * tilt_step, 0.0, 0.5 * tilt_step, tilt_step)
        else:
            tilt_deltas = (0.0,)
            tilt = 0.0
        # 走时在用户预设 prior 附近微调（±1 s）；次生相模式冻结为 prior
        tt_lo = float(prior_tt) - 1.0
        tt_hi = float(prior_tt) + 1.0
        if bool(policy.allow_time_shift):
            shift_deltas = (-tshift_step, 0.0, tshift_step)
        else:
            shift_deltas = (0.0,)
            time_shift = float(prior_tt)
        if bool(policy.allow_position):
            dx_deltas = (-pos_step_xy, 0.0, pos_step_xy)
            dy_deltas = (-pos_step_xy, 0.0, pos_step_xy)
            dz_deltas = (-pos_step_z, 0.0, pos_step_z)
        else:
            dx_deltas = (0.0,)
            dy_deltas = (0.0,)
            dz_deltas = (0.0,)
        candidates: List[Tuple[float, float, np.ndarray, float]] = []
        for da in (-az_step, -0.5 * az_step, 0.0, 0.5 * az_step, az_step):
            for dt in tilt_deltas:
                for dx in dx_deltas:
                    for dy in dy_deltas:
                        for dz in dz_deltas:
                            for ds in shift_deltas:
                                candidates.append(
                                    (
                                        az + da,
                                        tilt + dt,
                                        pos + np.array([dx, dy, dz], dtype=float),
                                        float(np.clip(time_shift + ds, tt_lo, tt_hi)),
                                    )
                                )

        local_best = None
        local_best_obj = float("inf")
        local_best_parts: Dict[str, float] = {}
        for caz, ctilt, cpos, cshift in candidates:
            # 倾角搜索硬限制，防止候选把 Z 旋坏
            ctilt_use = float(ctilt)
            if bool(correct_tilt_eff):
                ctilt_use = float(np.clip(ctilt_use, -15.0, 15.0))
            else:
                ctilt_use = 0.0
            obj, parts = _objective(
                obs=obs,
                azimuth_deg=float(caz),
                tilt_deg=float(ctilt_use),
                position_corr=np.asarray(cpos, dtype=float),
                time_shift_sec=float(cshift),
                depth_override=depth_override,
                w_tt=float(w_tt_eff),
                w_pol=float(inp.w_pol),
                w_sym=float(inp.w_sym),
                scales=scales,
                prior_time_shift_sec=float(prior_tt),
                correct_tilt=bool(correct_tilt_eff),
            )
            if obj < local_best_obj:
                local_best_obj = obj
                local_best = (float(caz), float(ctilt_use), np.asarray(cpos, dtype=float), float(cshift))
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
                "J_pol_az": float(best_parts.get("J_pol_az", np.nan)),
                "J_inc": float(best_parts.get("J_inc", np.nan)),
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
                "n_ori": float(best_parts.get("n_ori", np.nan)),
                "ori_circ_mean_deg": float(best_parts.get("ori_circ_mean_deg", np.nan)),
                "ori_circ_std_deg": float(best_parts.get("ori_circ_std_deg", np.nan)),
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
    if not bool(correct_tilt_eff):
        tilt = 0.0
    else:
        # 传感器倾角应为小量；限制在 ±15°，避免被入射角量级带飞而毁掉 Z
        tilt = float(np.clip(float(tilt), -15.0, 15.0))
    if not bool(policy.allow_position):
        pos = np.asarray(inp.initial_position_correction, dtype=float).copy()
        if pos.size != 3:
            pos = np.zeros(3, dtype=float)
    if not bool(policy.allow_time_shift):
        time_shift = float(prior_tt)
    # 走时三量：预置 prior、校正增量 corr、最终 final = prior + corr
    tt_prior = float(prior_tt)
    tt_final = float(time_shift)
    tt_corr = float(tt_final - tt_prior)
    best_parts = dict(best_parts or {})
    best_parts["prior_time_shift_sec"] = tt_prior
    best_parts["tt_corr_sec"] = tt_corr
    best_parts["time_shift_sec"] = tt_final
    best_parts["initial_azimuth_deg"] = float(inp.initial_azimuth_deg)
    best_parts["phase_mode"] = float({"direct": 1.0, "mixed": 2.0, "secondary_attitude": 3.0}.get(policy.mode_key, 0.0))
    best_parts["n_direct"] = float(policy.n_direct)
    best_parts["n_secondary"] = float(policy.n_secondary)
    best_parts["w_tt_eff"] = float(w_tt_eff)
    best_parts["correct_tilt_eff"] = 1.0 if correct_tilt_eff else 0.0
    ok = np.isfinite(best_obj) and best_obj < 1e8
    if ok:
        tilt_note = "倾角已校正" if bool(correct_tilt_eff) else "倾角未校正（固定初值）"
        msg = f"姿态校正完成（{policy.message}；{tilt_note}）"
    else:
        msg = f"姿态校正未收敛（{policy.message}）"
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

