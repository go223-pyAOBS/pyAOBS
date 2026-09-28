# -*- coding: utf-8 -*-
"""
速度模型：读层析 RSF / 插值到工区网格、bath 填海水、写出 vel.rsf / bath1d / ss / rr。

bath 填海水与成像 vel/bath1d 写出（GUI 权威实现）。
"""

from __future__ import annotations

import os
import struct
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from ..project import GridParams, ObsRtmProject
from .geometry import load_xz_txt
from .rsf_io import parse_rsf_header, resolve_rsf_binary


def read_vel_rsf(path: str) -> Tuple[np.ndarray, Dict[str, float]]:
    """读 2D 速度 RSF → (n1=z, n2=x), meta(o1,d1,n1,o2,d2,n2)。"""
    meta = parse_rsf_header(path)
    n1 = int(meta["n1"])
    n2 = int(meta.get("n2", 1))
    o1 = float(meta.get("o1", 0))
    d1 = float(meta.get("d1", 1))
    o2 = float(meta.get("o2", 0))
    d2 = float(meta.get("d2", 1))
    bin_path = resolve_rsf_binary(path, meta)
    raw = np.fromfile(bin_path, dtype=np.float32, count=n1 * n2)
    if raw.size != n1 * n2:
        raise RuntimeError("short read %s: %d/%d" % (path, raw.size, n1 * n2))
    # 与 shot gather 相同：文件按 (n2, n1) 行主序块 → reshape (n2,n1).T
    data = raw.reshape((n2, n1)).T.copy()
    return data, {
        "o1": o1, "d1": d1, "n1": float(n1),
        "o2": o2, "d2": d2, "n2": float(n2),
    }


def write_vel_rsf(
    path: str,
    vel: np.ndarray,
    grid: GridParams,
    *,
    label1: str = "Depth",
    unit1: str = "km",
    label2: str = "Distance",
    unit2: str = "km",
) -> None:
    vel = np.asarray(vel, dtype=np.float32)
    nz, nx = vel.shape
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    bin_path = path + "@"
    with open(bin_path, "wb") as b:
        for ix in range(nx):
            b.write(np.ascontiguousarray(vel[:, ix]).tobytes())
    with open(path, "w", encoding="utf-8") as h:
        h.write("in=%s\n" % os.path.abspath(bin_path))
        h.write("n1=%d\nd1=%g\no1=%g\n" % (nz, grid.dz, grid.oz))
        h.write("label1=%s\nunit1=%s\n" % (label1, unit1))
        h.write("n2=%d\n" % nx)
        h.write("d2=%g\n" % grid.dx)
        h.write("o2=%g\n" % grid.ox)
        h.write("label2=%s\nunit2=%s\n" % (label2, unit2))
        h.write("data_format=native_float\nesize=4\n")


def write_bath1d_rsf(path: str, bath: np.ndarray, ox: float, dx: float) -> None:
    bath = np.asarray(bath, dtype=np.float32).ravel()
    bin_path = path + "@"
    with open(bin_path, "wb") as b:
        b.write(bath.tobytes())
    with open(path, "w", encoding="utf-8") as h:
        h.write("in=%s\n" % os.path.abspath(bin_path))
        h.write("n1=%d\nd1=%g\no1=%g\n" % (bath.size, dx, ox))
        h.write("label1=Distance\nunit1=km\n")
        h.write("data_format=native_float\nesize=4\n")


def write_xz_rsf(path: str, pts, label2: str = "OBS") -> None:
    bin_path = path + "@"
    with open(bin_path, "wb") as b:
        for x, z in pts:
            b.write(struct.pack("ff", float(x), float(z)))
    with open(path, "w", encoding="utf-8") as h:
        h.write("in=%s\n" % os.path.abspath(bin_path))
        h.write("n1=2\nd1=1\no1=0\nlabel1=xz\n")
        h.write("n2=%d\n" % len(pts))
        h.write("d2=1\n")
        h.write("o2=0\n")
        h.write("label2=%s\n" % label2)
        h.write("data_format=native_float\nesize=4\n")


def load_bath_txt(path: str) -> Tuple[np.ndarray, np.ndarray]:
    xs, zs = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a = line.replace(",", " ").split()
            xs.append(float(a[0]))
            zs.append(float(a[1]))
    if not xs:
        raise RuntimeError("bath 文件无数据: %s" % path)
    o = np.argsort(xs)
    return np.asarray(xs, float)[o], np.asarray(zs, float)[o]


def resample_to_grid(
    vel: np.ndarray,
    meta: Dict[str, float],
    grid: GridParams,
) -> np.ndarray:
    """把任意 (z,x) 速度插到工区网格（优先 scipy，否则双线性）。"""
    o1, d1 = float(meta["o1"]), float(meta["d1"])
    o2, d2 = float(meta["o2"]), float(meta["d2"])
    z = grid.oz + np.arange(grid.nz) * grid.dz
    x = grid.ox + np.arange(grid.nx) * grid.dx
    iz = (z - o1) / d1
    ix = (x - o2) / d2
    try:
        from scipy.ndimage import map_coordinates

        zz, xx = np.meshgrid(iz, ix, indexing="ij")
        coords = np.vstack([zz.ravel(), xx.ravel()])
        out = map_coordinates(vel.astype(np.float64), coords, order=1, mode="nearest")
        return out.reshape((grid.nz, grid.nx)).astype(np.float32)
    except ImportError:
        # 最近邻回退
        nz_s, nx_s = vel.shape
        izi = np.clip(np.rint(iz).astype(int), 0, nz_s - 1)
        ixi = np.clip(np.rint(ix).astype(int), 0, nx_s - 1)
        return vel[np.ix_(izi, ixi)].astype(np.float32)


def suggest_seafloor_iface_idx(zelt_model) -> Optional[int]:
    """
    默认海底 = 第 2 个界面（0-based 索引 1：Interface 1=海面，Interface 2=海底）。
    层数不足时回退到仅有的上一层。
    """
    if zelt_model is None:
        return None
    try:
        n = len(zelt_model.depth_nodes)
    except Exception:
        return None
    if n <= 0:
        return None
    # 常规 Zelt：第 1 层节点=海面，第 2 层=地形海底
    if n >= 2:
        return 1
    return 0


def bath_from_zelt_iface(
    grid: GridParams, zelt_model, iface_idx: int
) -> Optional[np.ndarray]:
    """将 Zelt 第 iface_idx 层插值到工区 x 网格 → bath_1d；失败返回 None。"""
    if zelt_model is None or iface_idx is None:
        return None
    try:
        from pyAOBS.visualization.imodel.gui.zelt_iface_qt import (
            interpolate_interface_depths_on_grid,
        )

        x = grid.ox + np.arange(grid.nx, dtype=float) * grid.dx
        bath = interpolate_interface_depths_on_grid(zelt_model, int(iface_idx), x)
        bath = np.asarray(bath, dtype=np.float32).ravel()
        if bath.size < grid.nx:
            bath = np.pad(bath, (0, grid.nx - bath.size), mode="edge")
        bath = bath[: grid.nx]
        if not np.any(np.isfinite(bath)):
            return None
        # 全零/近零多半是失败插值，勿当地形
        if float(np.nanmax(np.abs(bath))) < 1e-4:
            return None
        return np.nan_to_num(bath, nan=float(np.nanmedian(bath))).astype(np.float32)
    except Exception:
        return None


def resolve_zelt_for_bath(project: ObsRtmProject):
    """解析可用于地形 bath 的 Zelt 模型（优先保留的原始 v.in）。"""
    from .model_import import load_zelt_model_optional

    vp = project.velocity
    cands = []
    for key in ("zelt_vin_path", "tomo_path"):
        p = str(getattr(vp, key, "") or "").strip()
        if p:
            cands.append(p if os.path.isabs(p) else project.path(p))
    tv = str(getattr(project, "tomo_vel", "") or "").strip()
    if tv:
        cands.append(tv if os.path.isabs(tv) else project.path(tv))
    seen = set()
    for p in cands:
        npth = os.path.normpath(os.path.abspath(p)) if p else ""
        if not npth or npth in seen:
            continue
        seen.add(npth)
        zelt = load_zelt_model_optional(npth)
        if zelt is not None:
            return zelt, npth
    return None, ""


def bath_on_grid(
    grid: GridParams,
    *,
    bath_path: str = "",
    flat_z_km: float = 0.0,
    obs_xz_path: str = "",
    zelt_model=None,
    seafloor_iface_idx: Optional[int] = None,
    prefer_zelt: bool = True,
) -> np.ndarray:
    """
    返回 bath_1d (nx,)。

    默认优先 **v.in 海底界面 S（地形）**；其次 bath_x.txt → OBS z → 平海底。
    ``seafloor_iface_idx is None``（UI Auto）时默认第 2 个界面。
    """
    x = grid.ox + np.arange(grid.nx, dtype=float) * grid.dx

    if prefer_zelt and zelt_model is not None:
        idx = seafloor_iface_idx
        if idx is None:
            idx = suggest_seafloor_iface_idx(zelt_model)
        if idx is not None:
            bath_z = bath_from_zelt_iface(grid, zelt_model, int(idx))
            if bath_z is not None:
                return bath_z

    if bath_path and os.path.isfile(bath_path):
        bx, bz = load_bath_txt(bath_path)
        return np.interp(x, bx, bz, left=float(bz[0]), right=float(bz[-1])).astype(np.float32)
    if obs_xz_path and os.path.isfile(obs_xz_path):
        pts = load_xz_txt(obs_xz_path)
        if pts:
            bx = np.asarray([p[0] for p in pts], float)
            bz = np.asarray([p[1] for p in pts], float)
            o = np.argsort(bx)
            return np.interp(x, bx[o], bz[o], left=float(bz[o[0]]), right=float(bz[o[-1]])).astype(
                np.float32
            )
    return np.full(grid.nx, float(flat_z_km), dtype=np.float32)


def resolve_bath_1d(
    project: ObsRtmProject,
    grid: Optional[GridParams] = None,
    *,
    log: Optional[Callable[[str], None]] = None,
    prefer_zelt: Optional[bool] = None,
) -> Tuple[np.ndarray, str]:
    """
    按工程参数解析 bath_1d，返回 (bath, source_tag)。
    source_tag: zelt_S# | bath_file | obs_xz | flat

    - 用户模型：默认用 v.in·S 地形；
    - 内置一维：不用 Zelt 地形（仅 bath 文件 / OBS z / 平海底），避免切回一维仍带用户海底。
    """
    g = grid if grid is not None else project.grid
    vp = project.velocity
    src = str(getattr(vp, "vel_source", "file") or "file").strip().lower()
    use_zelt = (
        bool(prefer_zelt)
        if prefer_zelt is not None
        else (src != "builtin_1d")
    )
    bath_path = str(getattr(vp, "bath_path", "") or "").strip()
    if bath_path and not os.path.isabs(bath_path):
        bath_path = project.path(bath_path)
    obs_path = project.path(project.obs_xz) if project.workdir else ""

    zelt, zpath = (None, "")
    s_idx = None
    used_idx = None
    if use_zelt:
        zelt, zpath = resolve_zelt_for_bath(project)
        s_idx = getattr(vp, "iface_seafloor", None)
        if s_idx is not None:
            try:
                s_idx = int(s_idx)
            except (TypeError, ValueError):
                s_idx = None
        used_idx = s_idx
        if zelt is not None and used_idx is None:
            used_idx = suggest_seafloor_iface_idx(zelt)

    bath = bath_on_grid(
        g,
        bath_path=bath_path,
        flat_z_km=float(getattr(vp, "flat_bath_km", 0.0) or 0.0),
        obs_xz_path=obs_path,
        zelt_model=zelt if use_zelt else None,
        seafloor_iface_idx=s_idx if use_zelt else None,
        prefer_zelt=use_zelt,
    )

    # 判定实际来源（与 bath_on_grid 优先级一致）
    tag = "flat"
    if use_zelt and zelt is not None and used_idx is not None:
        trial = bath_from_zelt_iface(g, zelt, int(used_idx))
        if trial is not None and np.allclose(bath, trial, rtol=0, atol=1e-4):
            tag = "zelt_S%d" % (int(used_idx) + 1)
    if tag == "flat" and bath_path and os.path.isfile(bath_path):
        tag = "bath_file"
    elif tag == "flat" and obs_path and os.path.isfile(obs_path):
        pts = load_xz_txt(obs_path)
        if pts:
            tag = "obs_xz"

    if log:
        extra = ""
        if tag.startswith("zelt_S") and zpath:
            extra = " ← %s" % os.path.basename(zpath)
        elif src == "builtin_1d" and not use_zelt:
            extra = " (一维不用 v.in 地形)"
        log(
            "bath[%s]: min=%.3f max=%.3f km%s"
            % (tag, float(bath.min()), float(bath.max()), extra)
        )
    return bath.astype(np.float32), tag


def fill_water(vel: np.ndarray, grid: GridParams, bath_1d: np.ndarray, vwater: float) -> np.ndarray:
    """z < bath(x) → vwater。"""
    out = np.asarray(vel, dtype=np.float32).copy()
    z = grid.oz + np.arange(grid.nz) * grid.dz
    bath_1d = np.asarray(bath_1d, dtype=float)
    for ix in range(min(grid.nx, bath_1d.size, out.shape[1])):
        out[z < bath_1d[ix], ix] = float(vwater)
    return out


def smooth_vel(vel: np.ndarray, rect1: int = 5, rect2: int = 5) -> np.ndarray:
    """简单盒式光滑（近似 sfsmooth；无 Madagascar 时用）。"""
    try:
        from scipy.ndimage import uniform_filter
    except ImportError:
        return vel
    r1 = max(1, int(rect1))
    r2 = max(1, int(rect2))
    return uniform_filter(vel.astype(np.float64), size=(r1, r2), mode="nearest").astype(np.float32)


# 内置一维：海底以下深度 (km) → 速度 (km/s)
_V1D_LAYERED_KNOTS = (
    (0.0, 2.0),
    (0.5, 2.5),
    (2.0, 4.0),
    (5.0, 6.0),
    (10.0, 7.0),
    (20.0, 7.8),
    (40.0, 8.2),
)


def v1d_profile_vs_depth(
    z_below: np.ndarray,
    *,
    preset: str = "linear_crust",
    v0: float = 2.0,
    grad: float = 0.5,
    vmax: float = 8.0,
) -> np.ndarray:
    """海底以下深度 → 岩体速度（不含海水）。"""
    zb = np.asarray(z_below, dtype=float)
    zb = np.maximum(zb, 0.0)
    preset = (preset or "linear_crust").strip().lower()
    if preset == "layered_crust":
        zk = np.asarray([k[0] for k in _V1D_LAYERED_KNOTS], float)
        vk = np.asarray([k[1] for k in _V1D_LAYERED_KNOTS], float)
        return np.interp(zb, zk, vk, left=float(vk[0]), right=float(vk[-1])).astype(
            np.float32
        )
    # linear_crust：v = min(v0 + grad * z_below, vmax)
    return np.minimum(
        float(v0) + float(grad) * zb, float(vmax)
    ).astype(np.float32)


def apply_v1d_isovalue_interface(
    vel: np.ndarray,
    bath_1d: np.ndarray,
    *,
    oz: float,
    dz: float,
    v_iso: float,
    dv: float,
    vwater: float = 1.50,
    fill_water: bool = True,
    taper_samples: int = 2,
) -> Tuple[np.ndarray, int]:
    """
    在速度体上把等值速度 ``v_iso`` 做成有限跳变界面（仅海底以下）。

    对每列：在岩体段找首次穿越 ``v_iso`` 的深度，其下侧加 ``dv``
   （短过渡，减轻数值色散）。水柱保持 ``vwater``。

    返回 (vel_out, n_cols_applied)。若等值线落在水柱内或剖面达不到该速度则该列跳过。
    """
    out = np.asarray(vel, dtype=np.float32).copy()
    nz, nx = out.shape
    z = float(oz) + np.arange(nz, dtype=float) * float(dz)
    bath = np.asarray(bath_1d, dtype=float)
    if bath.size < nx:
        bath = np.pad(bath, (0, nx - bath.size), mode="edge")
    bath = bath[:nx]
    v_iso = float(v_iso)
    dv = float(dv)
    tap = max(int(taper_samples), 0)
    n_ok = 0
    for ix in range(nx):
        b = float(bath[ix])
        col = out[:, ix]
        # 仅在海底以下的岩体段找穿越
        iz0 = int(np.searchsorted(z, b, side="left"))
        iz0 = max(0, min(iz0, nz - 1))
        rock = col[iz0:]
        if rock.size < 2:
            continue
        # 找首次从下方达到/穿越 v_iso（假设整体随深度增大）
        crossed = None
        for j in range(rock.size - 1):
            a, c = float(rock[j]), float(rock[j + 1])
            if (a - v_iso) * (c - v_iso) <= 0.0 or (
                a < v_iso <= c or c < v_iso <= a
            ):
                # 线性插值到等值深度（列内下标）
                if abs(c - a) < 1e-12:
                    frac = 0.0
                else:
                    frac = (v_iso - a) / (c - a)
                frac = float(np.clip(frac, 0.0, 1.0))
                crossed = iz0 + j + frac
                break
        if crossed is None:
            # 整段都高于/低于：取最接近点（仅当区间覆盖）
            rmin, rmax = float(np.min(rock)), float(np.max(rock))
            if not (rmin <= v_iso <= rmax):
                continue
            j = int(np.argmin(np.abs(rock - v_iso)))
            crossed = float(iz0 + j)
        iz_c = float(crossed)
        # 界面以下加 dv，过渡 tap 个样点
        for iz in range(iz0, nz):
            depth_off = float(iz) - iz_c
            if depth_off < -1e-6:
                continue
            if tap > 0 and depth_off < float(tap):
                w = 0.5 - 0.5 * np.cos(np.pi * depth_off / float(tap))
            else:
                w = 1.0
            col[iz] = np.float32(float(col[iz]) + dv * w)
        if fill_water:
            col[z < b] = float(vwater)
        out[:, ix] = col
        n_ok += 1
    return out, n_ok


def build_builtin_1d_velocity(
    grid: GridParams,
    bath_1d: np.ndarray,
    *,
    vwater: float = 1.50,
    fill_water: bool = True,
    ref: str = "subbottom",
    preset: str = "linear_crust",
    v0: float = 2.0,
    grad: float = 0.5,
    vmax: float = 8.0,
    iface_enable: bool = False,
    iface_v: float = 6.0,
    iface_dv: float = 0.8,
    iface_log: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    """
    内置一维速度 → (nz, nx)。

    起伏水深（不同 OBS 水深）:
      - bath_1d[x] 为海底深度；z < bath → 海水 vwater（若 fill_water）
      - ref=subbottom（推荐）: 岩体速度按「海底以下深度」z-bath(x)，
        沉积层随海底起伏，适合水深变化大的 OBS
      - ref=absolute: 岩体速度按绝对深度 z（填水后浅海与深海同 z 同速）
      - iface_enable: 将等值速度 iface_v 做成有限跳变（诊断用）
    """
    nz, nx = int(grid.nz), int(grid.nx)
    z = grid.oz + np.arange(nz, dtype=float) * grid.dz
    bath = np.asarray(bath_1d, dtype=float)
    if bath.size < nx:
        bath = np.pad(bath, (0, nx - bath.size), mode="edge")
    bath = bath[:nx]

    vel = np.empty((nz, nx), dtype=np.float32)
    ref = (ref or "subbottom").strip().lower()
    for ix in range(nx):
        b = float(bath[ix])
        if ref == "absolute":
            zb = z.copy()  # 绝对深度；水柱稍后覆盖
        else:
            zb = z - b  # 海底以下；水柱为负，profile 会 clamp 到 0
        col = v1d_profile_vs_depth(
            zb, preset=preset, v0=v0, grad=grad, vmax=vmax
        )
        if fill_water:
            col = col.copy()
            col[z < b] = float(vwater)
        vel[:, ix] = col

    if iface_enable and abs(float(iface_dv)) > 1e-12:
        vel, n_ok = apply_v1d_isovalue_interface(
            vel,
            bath,
            oz=float(grid.oz),
            dz=float(grid.dz),
            v_iso=float(iface_v),
            dv=float(iface_dv),
            vwater=float(vwater),
            fill_water=bool(fill_water),
        )
        if iface_log:
            if n_ok <= 0:
                iface_log(
                    "等值线→界面: 未找到 v=%.3g km/s（检查剖面是否覆盖该速度）"
                    % float(iface_v)
                )
            else:
                iface_log(
                    "等值线→界面: v=%.3g km/s  ΔV=%+.3g  作用于 %d/%d 列"
                    % (float(iface_v), float(iface_dv), n_ok, nx)
                )
    return vel


def build_velocity_model(
    project: ObsRtmProject,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (vel, bath_1d)，并写入:
      vel.rsf, vels.rsf(可选), bath1d.rsf, ss.rsf, rr.rsf
    """
    def _log(m: str) -> None:
        if log:
            log(m)

    project.ensure_workdir()
    grid = project.grid
    vp = project.velocity

    obs_path = project.path(project.obs_xz)
    # 锁定本次生成的速度来源（RTM 自动生成时勿与 GUI 中途切换搞混）
    src = str(getattr(vp, "vel_source", "file") or "file").strip().lower()
    use_zelt_bath = src != "builtin_1d"
    _log(
        "成像速度来源=%s  地形策略=%s"
        % (
            "内置一维" if src == "builtin_1d" else "用户模型",
            "v.in·S / bath文件 / OBS / 平海底"
            if use_zelt_bath
            else "bath文件 / OBS / 平海底（不用 v.in 地形）",
        )
    )

    if use_zelt_bath:
        # 用户 v.in：默认用 Interfaces·S（地形海底）作 bath；保留原始 vin 路径
        tomo0 = str(vp.tomo_path or project.tomo_vel or "").strip()
        if tomo0 and not os.path.isabs(tomo0) and project.workdir:
            tomo0 = project.path(tomo0)
        from .model_import import load_zelt_model_optional

        z0 = load_zelt_model_optional(tomo0) if tomo0 else None
        if z0 is not None:
            vp.zelt_vin_path = tomo0
            if getattr(vp, "iface_seafloor", None) is None:
                sug = suggest_seafloor_iface_idx(z0)
                if sug is not None:
                    vp.iface_seafloor = int(sug)
                    _log(
                        "Interfaces·S 未选 → 默认 Interface %d（第 2 界面）作 bath"
                        % (int(sug) + 1)
                    )

    # 显式 prefer_zelt：一维绝不吃用户模型海底
    bath, bath_tag = resolve_bath_1d(
        project, grid, log=_log, prefer_zelt=use_zelt_bath
    )
    if bath_tag == "flat" and use_zelt_bath:
        _log("bath 回退：无可用 v.in 海底 / bath 文件 / OBS → 平海底")
    elif src == "builtin_1d":
        _log("一维 bath[%s] 已排除 v.in 地形" % bath_tag)

    if src == "builtin_1d":
        ref = str(getattr(vp, "v1d_ref", "subbottom") or "subbottom")
        preset = str(getattr(vp, "v1d_preset", "linear_crust") or "linear_crust")
        iface_on = bool(getattr(vp, "v1d_iface_enable", False))
        _log(
            "内置一维速度: preset=%s ref=%s v0=%.3g grad=%.3g vmax=%.3g"
            "%s"
            % (
                preset,
                ref,
                float(getattr(vp, "v1d_v0", 2.0)),
                float(getattr(vp, "v1d_grad", 0.5)),
                float(getattr(vp, "v1d_vmax", 8.0)),
                (
                    "  界面(v=%.3g ΔV=%+.3g)"
                    % (
                        float(getattr(vp, "v1d_iface_v", 6.0)),
                        float(getattr(vp, "v1d_iface_dv", 0.8)),
                    )
                    if iface_on
                    else ""
                ),
            )
        )
        vel = build_builtin_1d_velocity(
            grid,
            bath,
            vwater=float(vp.vwater),
            fill_water=bool(vp.fill_water),
            ref=ref,
            preset=preset,
            v0=float(getattr(vp, "v1d_v0", 2.0)),
            grad=float(getattr(vp, "v1d_grad", 0.5)),
            vmax=float(getattr(vp, "v1d_vmax", 8.0)),
            iface_enable=iface_on,
            iface_v=float(getattr(vp, "v1d_iface_v", 6.0)),
            iface_dv=float(getattr(vp, "v1d_iface_dv", 0.8)),
            iface_log=_log,
        )
        # 1D 路径已按列处理海水；若关闭 fill_water 则上面未填水
    else:
        tomo_path = project.tomo_vel or vp.tomo_path
        if not tomo_path or not os.path.isfile(tomo_path):
            raise FileNotFoundError(
                "请指定速度模型，或将「速度来源」改为内置一维"
            )

        from .model_import import convert_model_to_rsf, load_any_velocity

        if not tomo_path.lower().endswith(".rsf"):
            from .workdir_layout import TOMO_VEL

            out_tomo = project.path(TOMO_VEL)
            _log("转换 %s → %s" % (tomo_path, out_tomo))
            convert_model_to_rsf(
                tomo_path,
                out_tomo,
                vin_dx_km=float(vp.vin_dx_km),
                vin_dz_km=float(vp.vin_dz_km),
                grid=None,
                resample_to_project_grid=False,
            )
            project.tomo_vel = out_tomo
            vp.tomo_path = out_tomo
            tomo_path = out_tomo
        _log("读速度: %s" % tomo_path)
        vel0, meta, kind = load_any_velocity(
            tomo_path,
            vin_dx_km=float(vp.vin_dx_km),
            vin_dz_km=float(vp.vin_dz_km),
        )
        _log("格式=%s  n1=%d n2=%d" % (kind, int(meta["n1"]), int(meta["n2"])))
        same = (
            int(meta["n1"]) == grid.nz
            and int(meta["n2"]) == grid.nx
            and abs(float(meta["o1"]) - grid.oz) < 1e-9
            and abs(float(meta["d1"]) - grid.dz) < 1e-9
            and abs(float(meta["o2"]) - grid.ox) < 1e-9
            and abs(float(meta["d2"]) - grid.dx) < 1e-9
        )
        if same:
            vel = vel0.astype(np.float32)
            _log("网格与工区一致，跳过重采样")
        else:
            _log(
                "重采样到 ox=%g nx=%d oz=%g nz=%d"
                % (grid.ox, grid.nx, grid.oz, grid.nz)
            )
            vel = resample_to_grid(vel0, meta, grid)

        if vp.fill_water:
            vel = fill_water(vel, grid, bath, float(vp.vwater))
            _log("海水填充 vwater=%g km/s" % vp.vwater)

    if vp.smooth_rect > 0:
        vel = smooth_vel(vel, vp.smooth_rect, vp.smooth_rect)
        _log("光滑 rect=%d" % vp.smooth_rect)

    from .workdir_layout import BATH1D, RR_RSF, SS_RSF, path_vel_write

    out_vel = path_vel_write(project)
    vp.out_vel = os.path.relpath(out_vel, project.workdir).replace("\\", "/")
    if hasattr(project, "rtm") and project.rtm is not None:
        project.rtm.vel_rsf = vp.out_vel
    write_vel_rsf(out_vel, vel, grid)
    _log("wrote %s" % out_vel)

    write_bath1d_rsf(project.path(BATH1D), bath, grid.ox, grid.dx)
    _log("wrote %s" % BATH1D)

    shots = load_xz_txt(project.path(project.shots_xz))
    obs = load_xz_txt(obs_path)
    if shots:
        write_xz_rsf(project.path(SS_RSF), shots, label2="Shot")
        _log("wrote %s n2=%d" % (SS_RSF, len(shots)))
    if obs:
        write_xz_rsf(project.path(RR_RSF), obs, label2="OBS")
        _log("wrote %s n2=%d" % (RR_RSF, len(obs)))

    return vel, bath
