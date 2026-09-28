# -*- coding: utf-8 -*-
"""工区几何：读 xz 文本 + 网格落点检查（纯 Python，避免子进程/刷屏导致崩溃）。"""

from __future__ import annotations

import math
import os
from typing import Callable, Dict, List, Optional, Tuple

from ..project import GridParams, ObsRtmProject
from .rsf_io import load_offsets_table


def load_xz_txt(path: str) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    if not path or not os.path.isfile(path):
        return pts
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a = line.replace(",", " ").split()
            if len(a) >= 2:
                pts.append((float(a[0]), float(a[1])))
    return pts


def save_xz_txt(path: str, pts: List[Tuple[float, float]]) -> None:
    """写出 x z（km），两列空白分隔。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# x_km  z_km\n")
        for x, z in pts:
            f.write("%.6f  %.6f\n" % (float(x), float(z)))


def current_obs_x_km(project: ObsRtmProject) -> float:
    """优先 obs_xz.txt 第一台；否则几何参数 obs_x_km。"""
    obs = load_xz_txt(project.path(project.obs_xz))
    if obs:
        return float(obs[0][0])
    return float(getattr(project.geometry, "obs_x_km", 0.0) or 0.0)


def apply_obs_x_shift(
    project: ObsRtmProject,
    new_obs_x_km: float,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[float, float, float]:
    """
    offset 模式：按 offsets.txt 以新 OBS x 重建 shots_xz（与 su_to_shots 一致）：
      shot_x = obs_x + offset_sign * (offset_m/1000)
    无 offsets 时退化为整体平移。同步 obs_xz / geometry.obs_x_km。
    返回 (old_obs_x, new_obs_x, delta_obs)。
    """
    from .polygon_mute import invalidate_offset_cache

    project.ensure_workdir()
    shots_path = project.path(project.shots_xz)
    obs_path = project.path(project.obs_xz)
    shots = load_xz_txt(shots_path)
    obs = load_xz_txt(obs_path)
    if not shots and not obs:
        raise FileNotFoundError("缺少 shots_xz.txt / obs_xz.txt，请先导入 SU")
    old = float(obs[0][0]) if obs else float(
        getattr(project.geometry, "obs_x_km", 0.0) or 0.0
    )
    new = float(new_obs_x_km)
    delta = new - old
    geom = project.geometry
    sign = float(getattr(geom, "offset_sign", 1.0) or 1.0)
    zobs = float(obs[0][1]) if obs else float(
        getattr(geom, "zobs_const_km", 0.0) or 0.0
    )
    zshot_default = float(getattr(geom, "zshot_km", 0.01) or 0.01)

    off_table = load_offsets_table(project.path(project.offsets_txt))
    rebuilt = 0
    if off_table and str(getattr(geom, "geom", "") or "") == "offset":
        # 按炮号重建；保留原 shots 行序中的 z，缺省用 zshot
        z_by_i = {i: float(z) for i, (_x, z) in enumerate(shots)}
        max_i = max(max(off_table.keys()), max(z_by_i.keys()) if z_by_i else -1)
        new_shots: List[Tuple[float, float]] = []
        for i in range(max_i + 1):
            row = off_table.get(i) or []
            om = float(row[0]) if row else 0.0  # m
            sx = new + sign * (om * 0.001)
            sz = z_by_i.get(i, zshot_default)
            new_shots.append((sx, sz))
            if row:
                rebuilt += 1
        if new_shots:
            shots = new_shots
            save_xz_txt(shots_path, shots)
    elif abs(delta) > 1e-12 and shots:
        shots = [(float(x) + delta, float(z)) for x, z in shots]
        save_xz_txt(shots_path, shots)
    elif abs(delta) < 1e-12 and not off_table:
        project.geometry.obs_x_km = new
        if log:
            log("OBS x 未变：%.6f km（无需更新）" % new)
        return old, new, 0.0

    save_xz_txt(obs_path, [(new, zobs)])
    project.geometry.obs_x_km = new
    invalidate_offset_cache(project.workdir)
    if log:
        if rebuilt:
            log(
                "已按 offsets.txt 重建几何：OBS x %.6f → %.6f km（sign=%+g）；"
                "更新 %d 炮 shots_xz"
                % (old, new, sign, rebuilt)
            )
        else:
            log(
                "已平移几何：OBS x %.6f → %.6f km（Δ=%+.6f）；shots=%d"
                % (old, new, delta, len(shots))
            )
    return old, new, delta


def to_index(coord: float, origin: float, delta: float) -> int:
    return int(round((coord - origin) / delta))


def check_landing(
    project: ObsRtmProject,
    *,
    write_detail: bool = True,
    max_list_out: int = 40,
) -> Tuple[str, int, int]:
    """
    检查炮/OBS 是否落在网格内。
    返回 (摘要文本, 炮 OUT 数, OBS OUT 数)。
    明细写入工区 diag/geom_check.txt，避免向 GUI 日志刷数千行。
    """
    g = project.grid
    shots_path = project.path(project.shots_xz)
    obs_path = project.path(project.obs_xz)
    if not os.path.isfile(shots_path):
        raise FileNotFoundError("缺少 %s，请先导入 SU" % shots_path)

    shots = load_xz_txt(shots_path)
    obs = load_xz_txt(obs_path) if os.path.isfile(obs_path) else []

    def _check(pts: List[Tuple[float, float]], kind: str):
        out_idx = []
        lines = []
        for i, (x, z) in enumerate(pts):
            ix = to_index(x, g.ox, g.dx)
            iz = to_index(z, g.oz, g.dz)
            ok = 0 <= ix < g.nx and 0 <= iz < g.nz
            tag = "OK" if ok else "OUT"
            lines.append(
                "%s[%03d] x=%8.3f z=%7.3f -> ix=%5d iz=%5d %s"
                % (kind, i, x, z, ix, iz, tag)
            )
            if not ok:
                out_idx.append(i)
        return lines, out_idx

    shot_lines, shot_out = _check(shots, "shot")
    obs_lines, obs_out = _check(obs, "obs")

    zmax = g.oz + (g.nz - 1) * g.dz
    xmax = g.ox + (g.nx - 1) * g.dx
    vmax, dt = 8.0, 0.0015
    cfl = dt * vmax * math.sqrt(1.0 / (g.dx ** 2) + 1.0 / (g.dz ** 2))

    diag_dir = project.path("diag")
    os.makedirs(diag_dir, exist_ok=True)
    detail_path = os.path.join(diag_dir, "geom_check.txt")
    if write_detail:
        with open(detail_path, "w", encoding="utf-8") as f:
            f.write("=== Grid ===\n")
            f.write(
                "z: n1=%d o1=%g d1=%g -> zmax=%g km\n"
                % (g.nz, g.oz, g.dz, zmax)
            )
            f.write(
                "x: n2=%d o2=%g d2=%g -> xmax=%g km\n"
                % (g.nx, g.ox, g.dx, xmax)
            )
            f.write("\n=== Shots ===\n")
            f.write("\n".join(shot_lines) + "\n")
            f.write("\n=== OBS ===\n")
            f.write("\n".join(obs_lines) + "\n")
            f.write("\n=== CFL (vmax=8, dt=1.5ms) ===\n")
            f.write("CFL=%.4f (want < ~0.8)\n" % cfl)

    summary = [
        "=== 落点检查 ===",
        "网格: ox=%.3f dx=%.4f nx=%d (xmax=%.3f) | oz=%.3f dz=%.4f nz=%d (zmax=%.3f)"
        % (g.ox, g.dx, g.nx, xmax, g.oz, g.dz, g.nz, zmax),
        "炮: %d 点, OUT=%d | OBS: %d 点, OUT=%d"
        % (len(shots), len(shot_out), len(obs), len(obs_out)),
        "CFL≈%.4f (vmax=8 km/s, dt=1.5 ms)" % cfl,
    ]
    if shot_out:
        show = shot_out[:max_list_out]
        summary.append(
            "炮 OUT 索引: %s%s"
            % (show, " ..." if len(shot_out) > max_list_out else "")
        )
    if obs_out:
        show = obs_out[:max_list_out]
        summary.append(
            "OBS OUT 索引: %s%s"
            % (show, " ..." if len(obs_out) > max_list_out else "")
        )
    if write_detail:
        summary.append("明细: %s" % detail_path)
    if not shot_out and not obs_out:
        summary.append("全部落点在网格内 (OK)")
    return "\n".join(summary), len(shot_out), len(obs_out)


def run_obs_geometry_check(
    project: ObsRtmProject,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> int:
    """兼容旧接口：进程内检查，返回 OUT 总数（0=成功）。"""
    text, n_shot_out, n_obs_out = check_landing(project)
    if log:
        log(text)
    return int(n_shot_out + n_obs_out)


def suggest_grid_from_shots(
    shots: List[Tuple[float, float]],
    *,
    pad_km: float = 10.0,
    dx: float = 0.5,
    dz: float = 0.25,
    zmax: float = 40.0,
    obs: Optional[List[Tuple[float, float]]] = None,
) -> GridParams:
    """由炮点（及可选 OBS）范围粗估 ox/nx；深度用 zmax。"""
    pts = list(shots or [])
    if obs:
        pts.extend(obs)
    if not pts:
        return GridParams()
    xs = [p[0] for p in pts]
    zs = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    z_need = max(float(zmax), max(zs) + 1.0)
    ox = xmin - pad_km
    x1 = xmax + pad_km
    dx = max(float(dx), 1e-4)
    dz = max(float(dz), 1e-4)
    nx = max(int(round((x1 - ox) / dx)) + 1, 2)
    nz = max(int(round(z_need / dz)) + 1, 2)
    return GridParams(oz=0.0, dz=dz, nz=nz, ox=ox, dx=dx, nx=nx)


def _percentile_abs(values: List[float], p: float) -> float:
    """对已按绝对值排序的列表取百分位；空列表返回 nan。"""
    if not values:
        return float("nan")
    if len(values) == 1:
        return float(values[0])
    p = max(0.0, min(100.0, float(p)))
    k = (len(values) - 1) * (p / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return float(values[int(k)])
    return float(values[f] * (c - k) + values[c] * (k - f))


def _abs_residual_stats(
    residuals: List[float], *, bad_km: float = 0.5
) -> Tuple[float, float, float]:
    """返回 (median|r|, p95|r|, fraction |r|>bad_km)。"""
    if not residuals:
        return float("nan"), float("nan"), float("nan")
    abs_r = sorted(abs(float(r)) for r in residuals)
    n = len(abs_r)
    med = _percentile_abs(abs_r, 50.0)
    p95 = _percentile_abs(abs_r, 95.0)
    frac = sum(1 for a in abs_r if a > bad_km) / float(n)
    return med, p95, frac


def _tag_good_bad(med: float, *, good_km: float = 0.1, bad_km: float = 0.5) -> str:
    if not math.isfinite(med):
        return ""
    if med < good_km:
        return "  ← 好"
    if med > bad_km:
        return "  ← 差"
    return ""


def check_offset_sign_consistency(
    project: ObsRtmProject,
    *,
    max_report: int = 12,
) -> Tuple[str, dict]:
    """
    核对 shots_xz 与 offsets.txt / offset_sign 是否一致。

    关系：shot_x - obs_x ≈ offset_sign * (offset_m / 1000)

    同时评估翻转符号 (-offset_sign)，返回 (给用户看的摘要文本, 统计 dict)。
    明细写入工区 diag/offset_sign_check.txt。
    """
    shots_path = project.path(project.shots_xz)
    obs_path = project.path(project.obs_xz)
    offsets_path = project.path(project.offsets_txt)

    missing = []
    if not os.path.isfile(shots_path):
        missing.append(os.path.basename(shots_path) or "shots_xz.txt")
    if not os.path.isfile(offsets_path):
        missing.append(os.path.basename(offsets_path) or "offsets.txt")
    if missing:
        raise FileNotFoundError(
            "缺少 %s，请先导入 SU（生成 shots_xz / offsets）" % "、".join(missing)
        )

    shots = load_xz_txt(shots_path)
    if not shots:
        raise ValueError("shots_xz 为空：%s" % shots_path)

    obs = load_xz_txt(obs_path) if os.path.isfile(obs_path) else []
    off_table = load_offsets_table(offsets_path)
    if not off_table:
        raise ValueError("offsets.txt 无有效行：%s" % offsets_path)

    geom = project.geometry
    sign = float(getattr(geom, "offset_sign", 1.0) or 1.0)
    obs_x_ref = float(obs[0][0]) if obs else float(getattr(geom, "obs_x_km", 0.0) or 0.0)

    rows: List[Tuple[int, float, float, float, float]] = []
    # (ishot, dx, off_km_signed, residual, residual_flip)
    for ishot, (shot_x, _z) in enumerate(shots):
        row = off_table.get(int(ishot))
        if not row:
            continue
        vals = [float(v) for v in row]
        off_m = float(vals[0]) if len(vals) == 1 else float(sum(vals) / len(vals))
        dx = float(shot_x) - obs_x_ref
        pred = sign * (off_m * 0.001)
        pred_flip = (-sign) * (off_m * 0.001)
        rows.append((int(ishot), dx, pred, dx - pred, dx - pred_flip))

    if not rows:
        raise ValueError(
            "shots_xz 与 offsets.txt 无共同炮号（n_shot=%d, n_off=%d）"
            % (len(shots), len(off_table))
        )

    res_cur = [r[3] for r in rows]
    res_flip = [r[4] for r in rows]
    med, p95, frac_bad = _abs_residual_stats(res_cur, bad_km=0.5)
    med_f, p95_f, frac_bad_f = _abs_residual_stats(res_flip, bad_km=0.5)
    n = len(rows)

    # 结论：当前好 / 建议翻转 / 不一致
    if med < 0.1 or (med <= med_f and med < 0.25):
        status = "ok"
        advice = "当前 offset 符号与 shots_xz 一致（OK）。"
    elif med_f < 0.5 * max(med, 1e-12) and med_f < 0.2:
        status = "recommend_flip"
        advice = (
            "建议：把 offset 符号改为 %g 后重新导入 SU（或重写 shots_xz）。"
            % (-sign)
        )
    else:
        status = "warn"
        advice = (
            "警告：shots_xz 与 offsets×offset_sign 不一致；"
            "请检查几何模式（offset/obs/segy）或重新导入 SU。"
        )

    diag_dir = project.path("diag")
    os.makedirs(diag_dir, exist_ok=True)
    detail_path = os.path.join(diag_dir, "offset_sign_check.txt")
    with open(detail_path, "w", encoding="utf-8") as f:
        f.write("# offset_sign consistency check\n")
        f.write(
            "# obs_x_ref=%.6f km  offset_sign=%g  n=%d  status=%s\n"
            % (obs_x_ref, sign, n, status)
        )
        f.write(
            "# ishot  dx_km  off_km_signed  residual  residual_flip\n"
        )
        for ishot, dx, off_signed, r, rf in rows:
            f.write(
                "%6d  %12.6f  %12.6f  %12.6f  %12.6f\n"
                % (ishot, dx, off_signed, r, rf)
            )

    sign_flip = -sign
    summary_lines = [
        "offset 符号检查（n=%d）" % n,
        "obs_x_ref=%.3f km，当前 sign=%+g"
        % (obs_x_ref, sign),
        "当前 sign=%+g：|残差| 中位=%.3f km，P95=%.3f km，>|0.5km|=%.1f%%%s"
        % (sign, med, p95, 100.0 * frac_bad, _tag_good_bad(med)),
        "翻转 sign=%+g：|残差| 中位=%.3f km，P95=%.3f km，>|0.5km|=%.1f%%%s"
        % (sign_flip, med_f, p95_f, 100.0 * frac_bad_f, _tag_good_bad(med_f)),
        advice,
        "明细：%s" % detail_path,
    ]

    # 摘要中附若干最大 |残差| 样例
    max_report = max(0, int(max_report))
    if max_report and status != "ok":
        worst = sorted(rows, key=lambda t: abs(t[3]), reverse=True)[:max_report]
        summary_lines.append("最大 |残差| 样例（ishot, dx, res, res_flip）：")
        for ishot, dx, _off, r, rf in worst:
            summary_lines.append(
                "  %d: dx=%.3f res=%.3f res_flip=%.3f" % (ishot, dx, r, rf)
            )

    stats: Dict[str, object] = {
        "n": n,
        "offset_sign": sign,
        "offset_sign_flip": sign_flip,
        "obs_x_ref": obs_x_ref,
        "median_abs_res": med,
        "p95_abs_res": p95,
        "frac_abs_res_gt_0p5": frac_bad,
        "median_abs_res_flip": med_f,
        "p95_abs_res_flip": p95_f,
        "frac_abs_res_gt_0p5_flip": frac_bad_f,
        "status": status,
        "detail_path": detail_path,
        "ok": status == "ok",
        "recommend_flip": status == "recommend_flip",
    }
    return "\n".join(summary_lines), stats
