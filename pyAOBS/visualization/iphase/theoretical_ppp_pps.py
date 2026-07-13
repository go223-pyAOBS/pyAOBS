"""
理论 PPS-PPP 计算与沉积层参数反演（简化 1D 近似）

本模块中的 1D 公式（单层沉积层 + 水层）只刻画
「接收点附近上行段的 P→S 差异」，假定 PPP/PPS
在深部（壳/幔广角折射段）的路径基本相同。

因此：
1. 对于真正的广角 PPP/PPS（折射 + 基底转换），
   这里的 1D PPS-PPP 只是**近似/示意用模型**，
   主要用于看随偏移距的大致趋势和做简单反演初值；
2. 更严格、物理一致的建模应使用 RAYINVR /
   `TheoreticalTravelTimeCalculator` 或 `RayTracer`
   直接计算 PPP、PPS 两类震相的理论走时，再取差值。

参考：RAYINVR、TheoreticalTravelTimeCalculator、ray_tracer
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np
from .theory2d_service import build_theory2d_bundle_from_txout, make_delta_query


# -----------------------------------------------------------------------------
# 1D 近似：同一道、同一路径，仅最后 C->R 段波型不同
# -----------------------------------------------------------------------------

def pps_minus_ppp_from_conversion_receiver_segment(
    h_cr: float,
    vp: float,
    vs: float,
    dx_cr: np.ndarray | Sequence[float],
) -> np.ndarray:
    """
    用“转换点 C -> 接收点 R”段计算理论 PPS-PPP（趋势验证/简化近似）。

    假设（针对 OBS 同一道的 PPP/PPS）：
    - PPP 与 PPS 在深部路径相同；
    - 仅在转换点 C 到接收点 R 的最后一段，PPP 为 P、PPS 为 S；
    - C->R 段几何长度 L_cr = sqrt(h_cr^2 + dx_cr^2)。

    则：
        Delta_t = PPS - PPP = L_cr * (1/Vs - 1/Vp)

    Parameters
    ----------
    h_cr : float
        转换点到接收点的垂向距离 (km)
    vp, vs : float
        C->R 段对应介质的 P/S 波速度 (km/s)
    dx_cr : array-like
        转换点到接收点的水平偏移距 (km)，可正可负；结果对称于 0

    Returns
    -------
    np.ndarray
        理论 PPS-PPP (s)
    """
    if h_cr <= 0 or vp <= 0 or vs <= 0:
        raise ValueError("h_cr, vp, vs must be positive")
    if vs >= vp:
        raise ValueError("vs should be smaller than vp for typical sediment")

    dx = np.asarray(dx_cr, dtype=float)
    l_cr = np.sqrt(h_cr * h_cr + dx * dx)
    return l_cr * (1.0 / vs - 1.0 / vp)


# -----------------------------------------------------------------------------
# 1D 近似：由 PPP 拟合斜率估计射线参数 p，再计算 C->R 段 PPS-PPP
# -----------------------------------------------------------------------------


def _linear_fit_1d_safe(xw: np.ndarray, tw: np.ndarray) -> tuple[float, float] | None:
    """
    一元线性拟合 t ≈ a1*x + a0，避免 x 几乎常数时 np.polyfit 的 RankWarning 与病态。
    若 x 无展宽，退化为水平线 a1=0, a0=mean(t)。
    """
    if xw.size < 2 or tw.size < 2:
        return None
    mu_x = float(np.mean(xw))
    xc = xw.astype(float, copy=False) - mu_x
    var = float(np.dot(xc, xc))
    scale = max(abs(mu_x), 1e-6)
    eps = max(1e-24, (1e-14 * scale) ** 2 * float(xw.size))
    mu_t = float(np.mean(tw))
    if var < eps:
        return 0.0, mu_t
    tc = tw.astype(float, copy=False) - mu_t
    a1 = float(np.dot(xc, tc) / var)
    a0 = float(mu_t - a1 * mu_x)
    return a1, a0


def _sort_xy_by_abscissa(x: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    去掉非有限值后按横坐标（模型距离或偏移距）升序排列，保证两支拆分后子序列仍单调，
    避免拾取顺序混乱导致局部窗口错配或连线折返。
    """
    xv = np.asarray(x, dtype=float)
    tv = np.asarray(t, dtype=float)
    vf = np.isfinite(xv) & np.isfinite(tv)
    if xv.size == 0 or not np.any(vf):
        return xv[:0], tv[:0]
    xv = xv[vf]
    tv = tv[vf]
    order = np.argsort(xv, kind="mergesort")
    return xv[order], tv[order]


def _local_linear_eval(
    xb: np.ndarray,
    tb: np.ndarray,
    qb: np.ndarray,
    *,
    window_points: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    """
    局部滑动线性拟合：返回 query 点处的 (t_fit, dt/dx)。
    - 仅在 xb 有效区间内计算，区间外返回 NaN；
    - 每个 query 点使用最近的 window_points 个样本做线性拟合。
    """
    t_out = np.full_like(qb, np.nan, dtype=float)
    p_out = np.full_like(qb, np.nan, dtype=float)
    if xb.size < 2:
        return t_out, p_out

    order = np.argsort(xb)
    xs = xb[order]
    ts = tb[order]
    xmin, xmax = xs[0], xs[-1]
    in_range = (qb >= xmin) & (qb <= xmax)
    if not np.any(in_range):
        return t_out, p_out

    k = int(max(3, window_points))
    if k % 2 == 0:
        k += 1
    k = min(k, xs.size)

    q_valid = qb[in_range]
    for i, q in enumerate(q_valid):
        # 选最近 k 个点
        idx = np.argsort(np.abs(xs - q))[:k]
        xw = xs[idx]
        tw = ts[idx]
        # 小样本退化时至少保证 2 点
        if xw.size < 2:
            continue
        fit = _linear_fit_1d_safe(xw, tw)
        if fit is None:
            continue
        a1, a0 = fit
        t_out[np.where(in_range)[0][i]] = a1 * q + a0
        p_out[np.where(in_range)[0][i]] = a1
    return t_out, p_out


def fit_ppp_time_curve_local_linear(
    ppp_offsets: np.ndarray | Sequence[float],
    ppp_times: np.ndarray | Sequence[float],
    query_offsets: np.ndarray | Sequence[float],
    *,
    window_points: int = 11,
    split_by_sign: bool = True,
) -> np.ndarray:
    """用 LocalLinear 拟合 PPP 的 t(x) 曲线，在 query_offsets 上求 t_fit。"""
    x = np.asarray(ppp_offsets, dtype=float)
    t = np.asarray(ppp_times, dtype=float)
    q = np.asarray(query_offsets, dtype=float)
    t_fit = np.full_like(q, np.nan, dtype=float)
    x, t = _sort_xy_by_abscissa(x, t)
    if x.size < 3:
        return t_fit

    def _eval(xb: np.ndarray, tb: np.ndarray, qb: np.ndarray) -> np.ndarray:
        tf, _ = _local_linear_eval(xb, tb, qb, window_points=window_points)
        return tf

    if split_by_sign:
        pos = x > 0
        neg = x < 0
        qpos = q > 0
        qneg = q < 0
        if np.any(pos) and np.any(qpos):
            t_fit[qpos] = _eval(x[pos], t[pos], q[qpos])
        if np.any(neg) and np.any(qneg):
            t_fit[qneg] = _eval(x[neg], t[neg], q[qneg])
        qzero = q == 0
        if np.any(qzero):
            t_fit[qzero] = _eval(x, t, q[qzero])
    else:
        t_fit = _eval(x, t, q)
    return t_fit


def _smooth_1d_same(y: np.ndarray, half_win: int) -> np.ndarray:
    """对一维数组做滑动平均，保持长度不变（边界用半窗）。"""
    n = y.size
    if half_win <= 0 or n < 3:
        return y
    w = 2 * half_win + 1
    kernel = np.ones(w, dtype=float) / w
    out = np.convolve(y, kernel, mode="same")
    return out


def _estimate_p_from_fitted_curve_local_linear(
    offsets: np.ndarray | Sequence[float],
    times: np.ndarray | Sequence[float],
    query_offsets: np.ndarray | Sequence[float],
    *,
    window_points: int = 11,
    split_by_sign: bool = True,
    dense_points: int = 401,
    edge_guard_frac: float = 0.05,
    smooth_dense_half_win: int = 5,
) -> np.ndarray:
    """
    用“拟合后曲线求导”的方式估计射线参数 p=dt/dx。

    流程：
      1) 对原始 t(x) 做 LocalLinear 拟合，得到致密拟合曲线 t_fit(x)；
      2) 对致密曲线做轻量平滑，避免邻域切换导致的斜率跳变（毛刺）；
      3) 在致密网格上对平滑后的曲线求导得到 p_fit(x)；
      4) 将 p_fit(x) 插值回 query_offsets。
    """
    x = np.asarray(offsets, dtype=float)
    t = np.asarray(times, dtype=float)
    q = np.asarray(query_offsets, dtype=float)
    p_est = np.full_like(q, np.nan, dtype=float)
    x, t = _sort_xy_by_abscissa(x, t)
    if x.size < 3 or t.size < 3 or q.size == 0:
        return p_est

    def _eval_branch(xb: np.ndarray, tb: np.ndarray, qb: np.ndarray) -> np.ndarray:
        out = np.full_like(qb, np.nan, dtype=float)
        if xb.size < 3 or qb.size == 0:
            return out
        xmin = float(np.min(xb))
        xmax = float(np.max(xb))
        in_range = (qb >= xmin) & (qb <= xmax)
        if not np.any(in_range):
            return out

        n_dense = int(max(101, min(1001, dense_points)))
        xg = np.linspace(xmin, xmax, n_dense)
        tg = fit_ppp_time_curve_local_linear(
            xb,
            tb,
            xg,
            window_points=window_points,
            split_by_sign=False,
        )
        vg = np.isfinite(tg)
        if np.count_nonzero(vg) < 3:
            return out

        xgv = xg[vg]
        tgv = tg[vg]
        # 致密曲线由逐点局部拟合拼接，邻域切换处斜率会跳变；先平滑再求导使 p 光滑
        if smooth_dense_half_win > 0 and tgv.size > 2 * smooth_dense_half_win:
            tgv = _smooth_1d_same(tgv, smooth_dense_half_win)
        pgv = np.gradient(tgv, xgv)
        # 边界一侧导数对噪声更敏感：在拟合曲线两端留保护带，避免边界奇异斜率
        span = float(xgv[-1] - xgv[0]) if xgv.size >= 2 else 0.0
        guard = max(2.0 * float(xgv[1] - xgv[0]), edge_guard_frac * span) if xgv.size >= 2 else 0.0
        q_in = qb[in_range]
        q_good = (q_in >= (xgv[0] + guard)) & (q_in <= (xgv[-1] - guard))
        if np.any(q_good):
            out_idx = np.where(in_range)[0][q_good]
            out[out_idx] = np.interp(q_in[q_good], xgv, pgv, left=np.nan, right=np.nan)
        return out

    if split_by_sign:
        pos = x > 0
        neg = x < 0
        qpos = q > 0
        qneg = q < 0
        if np.any(pos) and np.any(qpos):
            p_est[qpos] = _eval_branch(x[pos], t[pos], q[qpos])
        if np.any(neg) and np.any(qneg):
            p_est[qneg] = _eval_branch(x[neg], t[neg], q[qneg])
        qzero = q == 0
        if np.any(qzero):
            p_est[qzero] = _eval_branch(x, t, q[qzero])
    else:
        p_est = _eval_branch(x, t, q)

    return p_est


def pps_minus_ppp_from_ppp_slope(
    ppp_offsets: np.ndarray | Sequence[float],
    ppp_times: np.ndarray | Sequence[float],
    query_offsets: np.ndarray | Sequence[float],
    *,
    h_cr: float,
    vp: float,
    vs: float,
    window_points: int = 11,
    split_by_sign: bool = True,
    smooth_dense_half_win: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    根据 PPP 走时曲线斜率估计射线参数 p=dt/dx，并计算理论 PPS-PPP。

    近似步骤：
    1) 用 PPP 的 t(x) 局部滑动线性拟合求导，得到 p(x)=dt/dx；
    2) 由 PPP 计算入射角：
         sin(theta_p)=|p|*Vp；
    3) 采用“PPP 与 PPS 在沉积层内共享同一路径”假设：
         L = h_cr / cos(theta_p)
         Delta_t = L*(1/Vs - 1/Vp).

    Parameters
    ----------
    ppp_offsets, ppp_times : array-like
        PPP 拾取的 true_offset 与到时
    query_offsets : array-like
        需要计算理论 PPS-PPP 的偏移距（通常用观测 PPS-PPP 的 offset）
    h_cr, vp, vs : float
        C->R 段厚度与 P/S 速度
    window_points : int
        局部线性窗口点数（建议奇数，默认 11）
    split_by_sign : bool
        是否正/负偏移距分开拟合
    smooth_dense_half_win : int
        求导前对致密拟合曲线的平滑半窗（0 表示不平滑），默认 5 以减弱 PPS-PPP 毛刺

    Returns
    -------
    (dt_theory, p_est) : tuple[np.ndarray, np.ndarray]
        dt_theory 为理论 PPS-PPP，p_est 为估计射线参数（s/km）
    """
    if h_cr <= 0 or vp <= 0 or vs <= 0:
        raise ValueError("h_cr, vp, vs must be positive")
    if vs >= vp:
        raise ValueError("vs should be smaller than vp for typical sediments")

    x = np.asarray(ppp_offsets, dtype=float)
    t = np.asarray(ppp_times, dtype=float)
    q = np.asarray(query_offsets, dtype=float)

    if x.size < 3 or t.size < 3:
        return np.full_like(q, np.nan, dtype=float), np.full_like(q, np.nan, dtype=float)

    p_est = _estimate_p_from_fitted_curve_local_linear(
        x,
        t,
        q,
        window_points=window_points,
        split_by_sign=split_by_sign,
        smooth_dense_half_win=smooth_dense_half_win,
    )

    # 用 |p| 计算 PPP 入射角；超物理范围直接标记为 NaN（不做临界裁剪）
    p_abs = np.abs(p_est)
    p_max = (1.0 / vp) * 0.98
    valid_p = np.isfinite(p_est) & (p_abs < p_max)
    p_use = np.full_like(p_abs, np.nan, dtype=float)
    p_use[valid_p] = p_abs[valid_p]

    cp = np.sqrt(1.0 - (p_use * vp) ** 2)
    # 共享同一路径长度 L，仅速度不同
    l_path = h_cr / cp
    dt = l_path * (1.0 / vs - 1.0 / vp)
    # 对非物理或异常值进行屏蔽
    dt[~valid_p] = np.nan
    dt[dt < 0] = np.nan
    return dt, p_est


# -----------------------------------------------------------------------------
# 局部反演：固定 Vp，反演 h(xc) 与 Vp/Vs(xc)
# -----------------------------------------------------------------------------


@dataclass
class LocalInversionResult:
    """局部反演结果（含窗口剖面 + 逐点结果）。"""

    # 窗口剖面结果
    x_conv: np.ndarray
    h_conv: np.ndarray
    vpratio_conv: np.ndarray
    npts_conv: np.ndarray
    rms_conv: np.ndarray
    # 全量点（与输入同长度）
    pred_dt: np.ndarray
    corrected_dt: np.ndarray
    # 逐点反演结果（与输入同长度，invalid 为 NaN）
    x_conv_pick: np.ndarray
    h_pick: np.ndarray
    vpratio_pick: np.ndarray
    valid_mask: np.ndarray


@dataclass
class MFieldInversionResult:
    """m(x,z)=1/Vs-1/Vp 网格反演结果。"""

    x_edges: np.ndarray
    z_edges: np.ndarray
    m_grid: np.ndarray
    vpvs_grid: np.ndarray
    coverage_grid: np.ndarray
    pathlen_grid: np.ndarray
    dt_pred: np.ndarray
    residual: np.ndarray
    m_at_conv: np.ndarray
    vpvs_at_conv: np.ndarray
    valid_mask: np.ndarray
    success: bool
    message: str


def invert_m_field_from_paths(
    x_rec: np.ndarray | Sequence[float],
    z_rec: np.ndarray | Sequence[float],
    x_conv: np.ndarray | Sequence[float],
    z_conv: np.ndarray | Sequence[float],
    dt_obs: np.ndarray | Sequence[float],
    *,
    vp_fixed: float,
    nx: int = 80,
    nz: int = 40,
    smooth_lambda_x: float = 0.2,
    smooth_lambda_z: float = 0.2,
    damp: float = 1e-4,
    samples_per_ray: int = 60,
) -> MFieldInversionResult:
    """
    用路径积分关系反演 m(x,z)=1/Vs-1/Vp，并换算点值 Vp/Vs。

    每条射线方程：
      dt_i = ∫_ray m(x,z) ds  ≈ sum_j K_ij * m_j
    """
    xr = np.asarray(x_rec, dtype=float)
    zr = np.asarray(z_rec, dtype=float)
    xc = np.asarray(x_conv, dtype=float)
    zc = np.asarray(z_conv, dtype=float)
    dt = np.asarray(dt_obs, dtype=float)

    n_all = xr.size
    valid = (
        np.isfinite(xr)
        & np.isfinite(zr)
        & np.isfinite(xc)
        & np.isfinite(zc)
        & np.isfinite(dt)
        & (zc > zr + 1e-5)
    )
    if np.sum(valid) < 3:
        nan = np.full(n_all, np.nan, dtype=float)
        return MFieldInversionResult(
            x_edges=np.asarray([], dtype=float),
            z_edges=np.asarray([], dtype=float),
            m_grid=np.asarray([[]], dtype=float),
            vpvs_grid=np.asarray([[]], dtype=float),
            coverage_grid=np.asarray([[]], dtype=float),
            pathlen_grid=np.asarray([[]], dtype=float),
            dt_pred=nan.copy(),
            residual=nan.copy(),
            m_at_conv=nan.copy(),
            vpvs_at_conv=nan.copy(),
            valid_mask=valid,
            success=False,
            message="not enough valid rays",
        )

    xr_v, zr_v = xr[valid], zr[valid]
    xc_v, zc_v = xc[valid], zc[valid]
    dt_v = dt[valid]
    n_ray = xr_v.size

    xmin = float(min(np.min(xr_v), np.min(xc_v)))
    xmax = float(max(np.max(xr_v), np.max(xc_v)))
    zmin = float(min(np.min(zr_v), np.min(zc_v)))
    zmax = float(max(np.max(zr_v), np.max(zc_v)))
    if xmax <= xmin:
        xmax = xmin + 1.0
    if zmax <= zmin:
        zmax = zmin + 0.2
    x_edges = np.linspace(xmin, xmax, max(8, int(nx)) + 1)
    z_edges = np.linspace(max(0.0, zmin), zmax, max(6, int(nz)) + 1)
    nxu = x_edges.size - 1
    nzu = z_edges.size - 1
    n_model = nxu * nzu

    # 构造射线路径矩阵 K
    K = np.zeros((n_ray, n_model), dtype=float)
    hit_count = np.zeros(n_model, dtype=float)
    path_len = np.zeros(n_model, dtype=float)
    ns = max(20, int(samples_per_ray))
    tau = np.linspace(0.0, 1.0, ns + 1)
    for i in range(n_ray):
        xline = xr_v[i] + (xc_v[i] - xr_v[i]) * tau
        zline = zr_v[i] + (zc_v[i] - zr_v[i]) * tau
        dx = np.diff(xline)
        dz = np.diff(zline)
        ds = np.sqrt(dx * dx + dz * dz)
        xm = 0.5 * (xline[:-1] + xline[1:])
        zm = 0.5 * (zline[:-1] + zline[1:])
        ix = np.searchsorted(x_edges, xm, side="right") - 1
        iz = np.searchsorted(z_edges, zm, side="right") - 1
        ok = (ix >= 0) & (ix < nxu) & (iz >= 0) & (iz < nzu) & np.isfinite(ds)
        if not np.any(ok):
            continue
        flat = iz[ok] * nxu + ix[ok]
        np.add.at(K[i], flat, ds[ok])
        np.add.at(hit_count, flat, 1.0)
        np.add.at(path_len, flat, ds[ok])

    # 正则项：x/z 一阶差分平滑 + 小阻尼
    dx_rows = []
    for iz in range(nzu):
        base = iz * nxu
        for ix in range(nxu - 1):
            row = np.zeros(n_model, dtype=float)
            row[base + ix] = -1.0
            row[base + ix + 1] = 1.0
            dx_rows.append(row)
    dz_rows = []
    for iz in range(nzu - 1):
        base0 = iz * nxu
        base1 = (iz + 1) * nxu
        for ix in range(nxu):
            row = np.zeros(n_model, dtype=float)
            row[base0 + ix] = -1.0
            row[base1 + ix] = 1.0
            dz_rows.append(row)
    Dx = np.asarray(dx_rows, dtype=float) if dx_rows else np.zeros((0, n_model), dtype=float)
    Dz = np.asarray(dz_rows, dtype=float) if dz_rows else np.zeros((0, n_model), dtype=float)
    I = np.eye(n_model, dtype=float)

    blocks = [K]
    rhs = [dt_v]
    if Dx.shape[0] > 0 and smooth_lambda_x > 0:
        blocks.append(np.sqrt(smooth_lambda_x) * Dx)
        rhs.append(np.zeros(Dx.shape[0], dtype=float))
    if Dz.shape[0] > 0 and smooth_lambda_z > 0:
        blocks.append(np.sqrt(smooth_lambda_z) * Dz)
        rhs.append(np.zeros(Dz.shape[0], dtype=float))
    if damp > 0:
        blocks.append(np.sqrt(damp) * I)
        rhs.append(np.zeros(n_model, dtype=float))

    A = np.vstack(blocks)
    b = np.concatenate(rhs)
    m_vec, *_ = np.linalg.lstsq(A, b, rcond=None)
    m_grid = m_vec.reshape(nzu, nxu)
    coverage_grid = hit_count.reshape(nzu, nxu)
    pathlen_grid = path_len.reshape(nzu, nxu)

    dt_pred_v = K @ m_vec
    resid_v = dt_v - dt_pred_v

    x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
    z_cent = 0.5 * (z_edges[:-1] + z_edges[1:])

    def _sample_bilinear(xq: np.ndarray, zq: np.ndarray, grid: np.ndarray) -> np.ndarray:
        out = np.full_like(xq, np.nan, dtype=float)
        if x_cent.size < 2 or z_cent.size < 2:
            return out
        dxv = x_cent[1] - x_cent[0]
        dzv = z_cent[1] - z_cent[0]
        fx = (xq - x_cent[0]) / dxv
        fz = (zq - z_cent[0]) / dzv
        ix0 = np.floor(fx).astype(int)
        iz0 = np.floor(fz).astype(int)
        tx = fx - ix0
        tz = fz - iz0
        ok = (ix0 >= 0) & (ix0 < x_cent.size - 1) & (iz0 >= 0) & (iz0 < z_cent.size - 1)
        if not np.any(ok):
            return out
        i0 = ix0[ok]
        j0 = iz0[ok]
        tx0 = tx[ok]
        tz0 = tz[ok]
        g00 = grid[j0, i0]
        g10 = grid[j0, i0 + 1]
        g01 = grid[j0 + 1, i0]
        g11 = grid[j0 + 1, i0 + 1]
        out[ok] = (
            (1 - tx0) * (1 - tz0) * g00
            + tx0 * (1 - tz0) * g10
            + (1 - tx0) * tz0 * g01
            + tx0 * tz0 * g11
        )
        return out

    m_at_v = _sample_bilinear(xc_v, zc_v, m_grid)
    vpvs_grid = 1.0 + vp_fixed * m_grid
    vpvs_at_v = 1.0 + vp_fixed * m_at_v

    dt_pred = np.full(n_all, np.nan, dtype=float)
    residual = np.full(n_all, np.nan, dtype=float)
    m_at = np.full(n_all, np.nan, dtype=float)
    vpvs_at = np.full(n_all, np.nan, dtype=float)
    dt_pred[valid] = dt_pred_v
    residual[valid] = resid_v
    m_at[valid] = m_at_v
    vpvs_at[valid] = vpvs_at_v

    return MFieldInversionResult(
        x_edges=x_edges,
        z_edges=z_edges,
        m_grid=m_grid,
        vpvs_grid=vpvs_grid,
        coverage_grid=coverage_grid,
        pathlen_grid=pathlen_grid,
        dt_pred=dt_pred,
        residual=residual,
        m_at_conv=m_at,
        vpvs_at_conv=vpvs_at,
        valid_mask=valid,
        success=True,
        message="ok",
    )


def invert_local_h_vpratio(
    model_x: np.ndarray | Sequence[float],
    true_offset: np.ndarray | Sequence[float],
    obs_dt: np.ndarray | Sequence[float],
    p_est: np.ndarray | Sequence[float],
    *,
    vp_fixed: float,
    h0: float = 1.5,
    vpratio0: float = 2.0,
    window_km: float = 1.5,
    min_points: int = 4,
    h_bounds: tuple[float, float] = (0.05, 10.0),
    ratio_bounds: tuple[float, float] = (1.05, 4.0),
    n_ratio_grid: int = 80,
    n_iter: int = 3,
    smooth_lambda_h: float = 0.2,
    smooth_lambda_r: float = 0.2,
) -> LocalInversionResult:
    """
    固定 Vp，进行“逐点 + 位置迭代”的局部反演，并给出校正前后曲线。

    模型：
      Delta_t = h * (Vp/Vs - 1) / (Vp * cos(theta_p))
      sin(theta_p) = |p|*Vp

    迭代框架（考虑“一个转换点影响多个拾取点”的耦合）：
      1) 给定 h_i, 由几何关系算每个点 x_c,i
      2) 在 x_c 上建立参数节点（中心点），每个拾取点由多个节点加权影响
      3) 通过全局目标函数联合更新 h 节点与 Vp/Vs 节点：
           min ||t_obs - t_pred||^2 + lambda_h*||Dh||^2 + lambda_r*||Dr||^2
         其中 t_pred 由节点加权映射得到
      4) 用更新后的 h_i 重算 x_c,i，循环 n_iter 次
    """
    x = np.asarray(model_x, dtype=float)
    off = np.asarray(true_offset, dtype=float)
    t = np.asarray(obs_dt, dtype=float)
    p = np.asarray(p_est, dtype=float)

    p_abs = np.abs(p)
    valid = np.isfinite(x) & np.isfinite(off) & np.isfinite(t) & np.isfinite(p) & (p_abs * vp_fixed < 0.98)
    if not np.any(valid):
        n = len(x)
        nan = np.full(n, np.nan, dtype=float)
        return LocalInversionResult(
            x_conv=np.asarray([], dtype=float),
            h_conv=np.asarray([], dtype=float),
            vpratio_conv=np.asarray([], dtype=float),
            npts_conv=np.asarray([], dtype=int),
            rms_conv=np.asarray([], dtype=float),
            pred_dt=nan.copy(),
            corrected_dt=nan.copy(),
            x_conv_pick=nan.copy(),
            h_pick=nan.copy(),
            vpratio_pick=nan.copy(),
            valid_mask=np.zeros(n, dtype=bool),
        )

    x_v = x[valid]
    off_v = off[valid]
    t_v = t[valid]
    p_v = p_abs[valid]
    # 对本项目 tx.in 定义：true_offset = pick_x - shot_header_x
    # 其中 shot_header_x 为接收点(OBS)位置，因此可由 pick_x - true_offset 还原接收点坐标
    x_rec_v = x_v - off_v

    cp_v = np.sqrt(1.0 - (p_v * vp_fixed) ** 2)
    sin_v = p_v * vp_fixed
    tan_v = sin_v / np.maximum(cp_v, 1e-8)
    sign_v = np.sign(off_v)
    sign_v[sign_v == 0] = 1.0
    if window_km <= 0:
        window_km = 1.5
    r0 = max(ratio_bounds[0], min(ratio_bounds[1], vpratio0))

    # 逐点初值
    h_i = np.full_like(t_v, h0, dtype=float)
    r_i = np.full_like(t_v, r0, dtype=float)

    c_arr = np.asarray([], dtype=float)
    h_arr = np.asarray([], dtype=float)
    r_arr = np.asarray([], dtype=float)
    n_arr = np.asarray([], dtype=int)
    rms_arr = np.asarray([], dtype=float)

    half = window_km / 2.0
    sigma = max(half / 2.0, 0.15)
    step = max(window_km * 0.5, 0.2)
    n_iter_use = max(1, int(n_iter))

    def _build_reg_matrix(m: int) -> np.ndarray:
        if m <= 1:
            return np.zeros((0, m), dtype=float)
        d = np.zeros((m - 1, m), dtype=float)
        for ii in range(m - 1):
            d[ii, ii] = -1.0
            d[ii, ii + 1] = 1.0
        return d

    for _ in range(n_iter_use):
        # 用当前 h_i 更新每个拾取点的转换点位置（以接收点为基准）
        # off>0 表示拾取点在接收点右侧，对应转换点也位于接收点右侧
        x_conv_v = x_rec_v + sign_v * (h_i * tan_v)
        xmin, xmax = float(np.nanmin(x_conv_v)), float(np.nanmax(x_conv_v))
        centers = np.arange(xmin, xmax + step, step)
        if centers.size == 0:
            centers = np.array([np.nanmedian(x_conv_v)])
        m_nodes = centers.size

        # 影响权重矩阵 W: 每个拾取点受多个转换点节点影响
        dx = x_conv_v[:, None] - centers[None, :]
        w = np.exp(-0.5 * (dx / sigma) ** 2)
        w[np.abs(dx) > half] = 0.0
        sw = np.sum(w, axis=1, keepdims=True)
        no_support = sw[:, 0] <= 1e-12
        if np.any(no_support):
            idx_near = np.argmin(np.abs(dx[no_support]), axis=1)
            w[no_support] = 0.0
            w[no_support, idx_near] = 1.0
            sw = np.sum(w, axis=1, keepdims=True)
        w = w / np.maximum(sw, 1e-12)

        # 节点初值
        h_node = np.full(m_nodes, float(np.nanmedian(h_i)), dtype=float)
        r_node = np.full(m_nodes, float(np.nanmedian(r_i)), dtype=float)

        dmat = _build_reg_matrix(m_nodes)
        dh = np.sqrt(max(0.0, smooth_lambda_h)) * dmat if dmat.size else None
        dr = np.sqrt(max(0.0, smooth_lambda_r)) * dmat if dmat.size else None

        # 内层交替最小二乘（全局目标函数）
        for _inner in range(3):
            # update h_node: t ≈ diag(b) * W * h_node
            r_eff = w @ r_node
            r_eff = np.clip(r_eff, ratio_bounds[0], ratio_bounds[1])
            b = (r_eff - 1.0) / (vp_fixed * cp_v)
            ah = b[:, None] * w
            if dh is not None:
                ah2 = np.vstack([ah, dh])
                yh2 = np.concatenate([t_v, np.zeros(dh.shape[0], dtype=float)])
            else:
                ah2 = ah
                yh2 = t_v
            h_node = np.linalg.lstsq(ah2, yh2, rcond=None)[0]
            h_node = np.clip(h_node, h_bounds[0], h_bounds[1])

            # update r_node: t + c ≈ diag(c) * W * r_node
            h_eff = w @ h_node
            ccoef = h_eff / (vp_fixed * cp_v)
            ar = ccoef[:, None] * w
            yr = t_v + ccoef
            if dr is not None:
                ar2 = np.vstack([ar, dr])
                yr2 = np.concatenate([yr, np.zeros(dr.shape[0], dtype=float)])
            else:
                ar2 = ar
                yr2 = yr
            r_node = np.linalg.lstsq(ar2, yr2, rcond=None)[0]
            r_node = np.clip(r_node, ratio_bounds[0], ratio_bounds[1])

        # 节点 -> 逐点
        h_i = w @ h_node
        r_i = w @ r_node
        h_i = np.clip(h_i, h_bounds[0], h_bounds[1])
        r_i = np.clip(r_i, ratio_bounds[0], ratio_bounds[1])

        c_arr = centers.astype(float)
        h_arr = h_node.astype(float)
        r_arr = r_node.astype(float)
        n_arr = np.sum(w > 0, axis=0).astype(int)
        pred_tmp = h_i * (r_i - 1.0) / (vp_fixed * cp_v)
        resid_tmp = t_v - pred_tmp
        rms_arr = np.full(m_nodes, np.nan, dtype=float)
        for j in range(m_nodes):
            mj = w[:, j] > 0
            if np.any(mj):
                rms_arr[j] = float(np.sqrt(np.mean(resid_tmp[mj] ** 2)))

    # 最终一次用更新后的 h_i 重算逐点 x_c 与理论值
    x_conv_v = x_rec_v + sign_v * (h_i * tan_v)
    pred_v = h_i * (r_i - 1.0) / (vp_fixed * cp_v)

    pred = np.full_like(t, np.nan, dtype=float)
    pred[valid] = pred_v
    med_pred = float(np.nanmedian(pred_v)) if np.any(np.isfinite(pred_v)) else 0.0
    corrected = np.full_like(t, np.nan, dtype=float)
    corrected[valid] = t_v - (pred_v - med_pred)

    x_pick = np.full_like(t, np.nan, dtype=float)
    h_pick = np.full_like(t, np.nan, dtype=float)
    r_pick = np.full_like(t, np.nan, dtype=float)
    x_pick[valid] = x_conv_v
    h_pick[valid] = h_i
    r_pick[valid] = r_i

    return LocalInversionResult(
        x_conv=c_arr,
        h_conv=h_arr,
        vpratio_conv=r_arr,
        npts_conv=n_arr,
        rms_conv=rms_arr,
        pred_dt=pred,
        corrected_dt=corrected,
        x_conv_pick=x_pick,
        h_pick=h_pick,
        vpratio_pick=r_pick,
        valid_mask=valid,
    )


def pss_minus_psp_from_pss_slope_with_profile(
    pss_model_x: np.ndarray | Sequence[float],
    pss_true_off: np.ndarray | Sequence[float],
    pss_times: np.ndarray | Sequence[float],
    *,
    vp: float,
    h_profile_x: np.ndarray | Sequence[float] | None = None,
    h_profile: np.ndarray | Sequence[float] | None = None,
    vpratio_profile: np.ndarray | Sequence[float] | None = None,
    h_default: float = 1.5,
    vpratio_default: float = 2.0,
    window_points: int = 11,
    split_by_sign: bool = True,
    n_iter: int = 2,
    smooth_dense_half_win: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    基于 PSS 的局部斜率估计 p，并结合已反演的 h(x)/VpVs(x) 计算理论 PSS-PSP。

    返回：
      dt_pss_psp, x_conv, h_used, vpratio_used, p_est

    说明：
      时差采用几何段长公式
        L = sqrt(h^2 + (x_conv-x_rec)^2)
        dt = L*(1/Vs - 1/Vp)
      因此当 PSS-PSP 与 PPS-PPP 使用同一转换点与同一参数时，两者理论时差应一致。
    """
    xm = np.asarray(pss_model_x, dtype=float)
    xo = np.asarray(pss_true_off, dtype=float)
    tt = np.asarray(pss_times, dtype=float)
    n = xm.size
    out_nan = np.full(n, np.nan, dtype=float)
    if vp <= 0 or n < 3:
        return out_nan, out_nan, out_nan, out_nan, out_nan

    # 估计 p=dt/dx（x 用 true_offset）
    x = xo
    q = xo
    p_est = np.full_like(q, np.nan, dtype=float)

    p_est = _estimate_p_from_fitted_curve_local_linear(
        x,
        tt,
        q,
        window_points=window_points,
        split_by_sign=split_by_sign,
        smooth_dense_half_win=smooth_dense_half_win,
    )

    # profile 插值函数（若无 profile，则用默认值）
    use_profile = (
        h_profile_x is not None
        and h_profile is not None
        and vpratio_profile is not None
        and len(np.asarray(h_profile_x)) >= 2
    )
    if use_profile:
        px = np.asarray(h_profile_x, dtype=float)
        ph = np.asarray(h_profile, dtype=float)
        pr = np.asarray(vpratio_profile, dtype=float)
        v = np.isfinite(px) & np.isfinite(ph) & np.isfinite(pr)
        px, ph, pr = px[v], ph[v], pr[v]
        if px.size < 2:
            use_profile = False
        else:
            so = np.argsort(px)
            px, ph, pr = px[so], ph[so], pr[so]

    def _h_of_x(xq: np.ndarray) -> np.ndarray:
        if use_profile:
            return np.interp(xq, px, ph, left=ph[0], right=ph[-1])
        return np.full_like(xq, h_default, dtype=float)

    def _r_of_x(xq: np.ndarray) -> np.ndarray:
        if use_profile:
            return np.interp(xq, px, pr, left=pr[0], right=pr[-1])
        return np.full_like(xq, vpratio_default, dtype=float)

    x_rec = xm - xo
    sign = np.sign(xo)
    sign[sign == 0] = 1.0

    h_use = _h_of_x(x_rec)
    r_use = _r_of_x(x_rec)
    x_conv = np.full_like(x_rec, np.nan, dtype=float)
    dt = np.full_like(x_rec, np.nan, dtype=float)

    n_iter_use = max(1, int(n_iter))
    p_abs = np.abs(p_est)
    valid = np.isfinite(p_est) & np.isfinite(x_rec) & np.isfinite(h_use) & np.isfinite(r_use) & (r_use > 1.0)
    if not np.any(valid):
        return dt, x_conv, h_use, r_use, p_est

    for _ in range(n_iter_use):
        vs = vp / np.maximum(r_use, 1.001)
        s = p_abs * vs
        cp = np.sqrt(np.maximum(1.0 - s * s, 0.0))
        good = valid & (s < 0.98) & (cp > 1e-6)
        if not np.any(good):
            break
        tan = s / np.maximum(cp, 1e-8)
        x_conv[good] = x_rec[good] + sign[good] * h_use[good] * tan[good]
        h_use[good] = _h_of_x(x_conv[good])
        r_use[good] = _r_of_x(x_conv[good])

    vs = vp / np.maximum(r_use, 1.001)
    good = valid & np.isfinite(x_conv) & np.isfinite(h_use) & (vs > 0)
    if np.any(good):
        l_seg = np.sqrt(h_use[good] * h_use[good] + (x_conv[good] - x_rec[good]) ** 2)
        dt[good] = l_seg * (1.0 / vs[good] - 1.0 / vp)
    dt[dt < 0] = np.nan
    return dt, x_conv, h_use, r_use, p_est


# -----------------------------------------------------------------------------
# 1D 单层沉积层解析公式（射线参数 + Snell 定律）
# -----------------------------------------------------------------------------

def _offset_to_ray_param_sediment_only(
    offset: float, h: float, vp: float
) -> float:
    """
    单层水平沉积层：从偏移距反推 P 波射线参数。
    几何：炮/检在界面顶，反射面深度 h，offset = x_r - x_s。
    x/2 = h * tan(i_p)  =>  i_p = arctan(|offset|/(2h))
    p = sin(i_p) / Vp  (km⁻¹·s)
    """
    if h <= 0 or vp <= 0:
        return 0.0
    x = abs(offset)
    i_p = np.arctan(x / (2.0 * h))
    return np.sin(i_p) / vp


def _offset_to_ray_param_water_sediment(
    offset: float,
    h_water: float,
    v_water: float,
    h_sed: float,
    vp_sed: float,
) -> float:
    """
    水层 + 沉积层：从偏移距反推射线参数 p。
    总偏移 x = 2*h_water*tan(i_w) + 2*h_sed*tan(i_p)，Snell: sin(i_w)/Vw = sin(i_p)/Vp = p。
    用牛顿迭代求解 p 使得计算得到的 offset 与给定值一致。
    """
    x_target = abs(offset)
    if x_target < 1e-6:
        return 0.0
    p_min, p_max = 0.0, min(1.0 / v_water, 1.0 / vp_sed) * 0.999
    p = 0.5 * (p_min + p_max)
    for _ in range(20):
        sw, sp = np.sqrt(1 - (p * v_water) ** 2), np.sqrt(1 - (p * vp_sed) ** 2)
        if sw < 1e-10 or sp < 1e-10:
            break
        x = 2 * h_water * (p * v_water) / sw + 2 * h_sed * (p * vp_sed) / sp
        # d(x)/d(p)
        dx = 2 * h_water * v_water / (sw**3) + 2 * h_sed * vp_sed / (sp**3)
        p = p - (x - x_target) / (dx + 1e-12)
        p = np.clip(p, p_min, p_max)
        if abs(x - x_target) < 1e-6:
            break
    return p


def pps_minus_ppp_1d_layer(
    h: float,
    vp: float,
    vs: float,
    offsets: np.ndarray | Sequence[float],
    *,
    h_water: float = 0.0,
    v_water: float = 1.5,
) -> np.ndarray:
    """
    单层水平沉积层（可选水层）的**近似**理论 PPS-PPP (s)。

    模型假设：
    - 水层 (h_water, v_water) + 沉积层 (h, vp, vs)，接收点在海底；
    - PPP 与 PPS 在深部路径（广角折射段）相同，只在接收点下方
      的上行沉积层段一部分，P 与 S 波型不同；
    - 水层走时在两者中相同，在 PPS-PPP 中相互抵消；
    - 差值近似为：h/(Vs*cos(i_s)) - h/(Vp*cos(i_p))。

    对真正的广角 PPP/PPS（壳/幔折射 + 基底转换），
    该模型仅用于**趋势分析与反演初值**，更精确的
    理论曲线应使用 RAYINVR / `pps_minus_ppp_from_rayinvr`。

    Parameters
    ----------
    h : float
        沉积层厚度 (km)
    vp, vs : float
        沉积层 P/S 波速度 (km/s)
    offsets : array-like
        真实偏移距 (km)，有符号
    h_water : float
        水层厚度 (km)，0 表示无水层
    v_water : float
        水层 P 波速度 (km/s)

    Returns
    -------
    np.ndarray
        各偏移距对应的理论 PPS-PPP (s)，形状与 offsets 一致
    """
    offsets = np.asarray(offsets, dtype=float)
    out = np.empty_like(offsets)
    for i, off in enumerate(offsets):
        if h_water > 0 and v_water > 0:
            p = _offset_to_ray_param_water_sediment(
                off, h_water, v_water, h, vp
            )
        else:
            p = _offset_to_ray_param_sediment_only(off, h, vp)
        cp = np.sqrt(max(0, 1 - (p * vp) ** 2))
        if cp < 1e-10:
            out[i] = np.nan
            continue
        # 共享同一路径长度 L=h/cos(theta_p)，仅速度不同
        l_path = h / cp
        dt = l_path * (1.0 / vs - 1.0 / vp)
        out[i] = dt
    return out


# -----------------------------------------------------------------------------
# 沉积层参数反演
# -----------------------------------------------------------------------------


@dataclass
class FitResult:
    """沉积层参数反演结果"""

    h: float
    vp: float
    vs: float
    residuals: np.ndarray
    rms: float
    n_iter: int
    success: bool
    message: str


def fit_sediment_params(
    pairs: Sequence[tuple[float, float, float]],
    h0: float,
    vp0: float,
    vs0: float,
    *,
    h_water: float = 0.0,
    v_water: float = 1.5,
    bounds: Optional[dict[str, tuple[float, float]]] = None,
) -> FitResult:
    """
    调整沉积层参数 (h, Vp, Vs)，使理论 PPS-PPP 在观测偏移距处与拾取值尽量一致。

    Parameters
    ----------
    pairs : list of (model_dist, true_offset, t_diff)
        compute_ppp_pps_diff_pairs 的返回值
    h0, vp0, vs0 : float
        沉积层初值 (km, km/s, km/s)
    h_water, v_water : float
        水层参数 (km, km/s)，固定
    bounds : dict, optional
        参数上下界，如 {"h": (0.1, 5), "vp": (2, 6), "vs": (1, 4)}

    Returns
    -------
    FitResult
        最优参数及残差统计
    """
    offsets = np.array([p[1] for p in pairs])
    t_obs = np.array([p[2] for p in pairs])

    def model(x: np.ndarray) -> np.ndarray:
        h, vp, vs = x[0], x[1], x[2]
        return pps_minus_ppp_1d_layer(
            h, vp, vs, offsets, h_water=h_water, v_water=v_water
        )

    def misfit(x: np.ndarray) -> float:
        t_pred = model(x)
        valid = np.isfinite(t_pred)
        if not np.any(valid):
            return 1e10
        return float(np.mean((t_obs[valid] - t_pred[valid]) ** 2))

    x0 = np.array([h0, vp0, vs0], dtype=float)
    _bounds = bounds or {}
    b = [
        _bounds.get("h", (0.05, 10)),
        _bounds.get("vp", (1.5, 8)),
        _bounds.get("vs", (0.5, 5)),
    ]

    try:
        from scipy.optimize import minimize

        res = minimize(
            misfit,
            x0,
            method="L-BFGS-B",
            bounds=b,
            options={"maxiter": 200},
        )
        x_opt = res.x
        success = res.success
        msg = res.message if isinstance(res.message, str) else str(res.message)
        n_iter = getattr(res, "nit", 0)
    except ImportError:
        x_opt = x0
        success = False
        msg = "scipy not available"
        n_iter = 0

    t_pred = model(x_opt)
    residuals = t_obs - t_pred
    valid = np.isfinite(residuals)
    rms = float(np.sqrt(np.mean(residuals[valid] ** 2))) if np.any(valid) else np.nan

    return FitResult(
        h=float(x_opt[0]),
        vp=float(x_opt[1]),
        vs=float(x_opt[2]),
        residuals=residuals,
        rms=rms,
        n_iter=n_iter,
        success=success,
        message=msg,
    )


# -----------------------------------------------------------------------------
# RAYINVR 接口（2D 模型）
# -----------------------------------------------------------------------------


def pps_minus_ppp_from_rayinvr(
    working_dir: str,
    *,
    phase_ppp: int = 5,
    phase_pps: int = 14,
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """
    从 RAYINVR 输出的 tx.out 中，按相位分别读取 PPP、PPS 走时，构造 PPS-PPP 的插值函数。

    前置条件：已运行 RAYINVR，tx.out 中存在 phase_ppp、phase_pps 对应的理论走时。
    实际解析逻辑建议复用 TheoreticalTravelTimeCalculator 的 tx.out 读取。

    Parameters
    ----------
    working_dir : str
        RAYINVR 工作目录（含 tx.out）
    phase_ppp, phase_pps : int
        PPP、PPS 的相位号（与 tx.in 中 ipf 一致）

    Returns
    -------
    callable or None
        f(x_query) -> 理论 PPS-PPP 数组，x_query 为接收点 x（与 tx.out 约定一致）；若无法读取则返回 None
    """
    tx_out = __import__("pathlib").Path(working_dir) / "tx.out"
    if not tx_out.exists():
        return None
    try:
        bundle = build_theory2d_bundle_from_txout(
            tx_out,
            phase_ppp=phase_ppp,
            phase_pps=phase_pps,
            input_hash="",
        )
    except Exception:
        return None
    if bundle.global_x.size < 2:
        return None

    query = make_delta_query(bundle, allow_extrapolation=False)

    def eval_at_x(x_query: np.ndarray) -> np.ndarray:
        """x_query: model distance（覆盖区外返回 NaN，不外推）。"""
        return np.asarray(query(np.asarray(x_query, dtype=float), None), dtype=float)

    return eval_at_x
