#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""台侧多次进核对照：壳内高速体，不是均匀偏置。

真模型：背景与 recv_peg_fwd 相同（水 1.5，壳 4.0+0.35/km，莫霍下 8.0），
另在壳内放一块高速体。初值=无异常背景。炮在海面、台在海底。
``tt_inverse -Y -F -w -u``。对照几何：0/1、4/5、0/1/4/5。
台在 38–62 km（夹住高速体），炮离开网格边 8 km。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "recv_peg_fwd"))
from make_recv_peg_case import (  # noqa: E402
    DX,
    DZ,
    H,
    H_MOHO,
    OBS_X,
    SHOT_Z,
    V_AIR,
    V_CRUST0,
    V_GRAD,
    V_MANTLE,
    V_WATER,
    XMAX,
    XMIN,
    ZMAX,
    _fmt_r_line,
    _fmt_s_line,
    _xs,
    _zs,
    write_moho,
    write_seafloor,
)

# 壳内高速体：台阵下方、紧挨海底之下。
# 背景 4.0+0.35*(z-2) 在莫霍附近已是 5.4–6.1；块放浅部，块内约 4.64–5.20，不和壳底速度带重叠。
AX0, AX1 = 42.0, 58.0
AZ0, AZ1 = 2.4, 4.0
DV_ANOM = 0.50
# 5 台夹住高速体。不再用 30/70：其 0/1 ±30 会落到网格边 x=0/100，
# 图方法星形不完整，边台斜穿还会把「框内快」和「框下快」拧成一次更新。
OBS_XS = (38.0, 44.0, 50.0, 56.0, 62.0)
SHOT_PAD = 8.0  # km；炮离开网格边（4×DX）


def _offset_span(abs_max: int) -> tuple[float, ...]:
    return tuple(float(i) for i in range(-abs_max, -9, 2)) + tuple(
        float(i) for i in range(10, abs_max + 1, 2)
    )


# 0/1 停在 ±30（背景 Pg 约 27 km 折莫霍）；4/5 用到 ±40 照框下。
OFFSETS_01 = _offset_span(30)
OFFSETS_45 = _offset_span(40)
OFFSETS = OFFSETS_45


def offsets_for(code: int) -> tuple[float, ...]:
    if code in (0, 1):
        return OFFSETS_01
    if code in (4, 5):
        return OFFSETS_45
    raise ValueError(f"unsupported raycode {code}")


def shot_in_mesh(x: float) -> bool:
    return XMIN + SHOT_PAD - 1e-9 <= x <= XMAX - SHOT_PAD + 1e-9


def iter_shots(ox: float, code: int):
    for dx in offsets_for(code):
        sx = ox + dx
        if shot_in_mesh(sx):
            yield sx


LH, LV = 6.0, 0.6  # Lv≈3×DZ；0.3 太贴格子，1.5+ 又会糊到框下
WSV = 80.0
TT_ERR = 0.05
TV = 20.0


def in_anomaly(x: float, z: float) -> bool:
    return AX0 - 1e-9 <= x <= AX1 + 1e-9 and AZ0 - 1e-9 <= z <= AZ1 + 1e-9


def v_bg(z: float) -> float:
    if z > H_MOHO + 1e-9:
        return V_MANTLE
    if z > H + 1e-9:
        return V_CRUST0 + V_GRAD * (z - H)
    return V_WATER


def v_at(x: float, z: float, *, anomaly: bool) -> float:
    v = v_bg(z)
    if anomaly and in_anomaly(x, z):
        return v + DV_ANOM
    return v


def write_smesh(path: Path, *, anomaly: bool) -> None:
    xs, zs = _xs(), _zs()
    nx, nz = len(xs), len(zs)
    lines = [
        f"{nx} {nz} {V_WATER:.4f} {V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    for x in xs:
        lines.append(" ".join(f"{v_at(x, z, anomaly=anomaly):.4f}" for z in zs))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_true(path: Path) -> None:
    write_smesh(path, anomaly=True)


def set_pick_uncert(path: Path | str, u: float = TT_ERR) -> None:
    """tt_forward 把拾取误差写死成 0.01；反演前改成至少 TT_ERR。"""
    path = Path(path)
    out: list[str] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        parts = ln.split()
        if parts and parts[0] == "r" and len(parts) >= 6:
            parts[5] = f"{u:g}"
            out.append(" ".join(parts))
        else:
            out.append(ln)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_start(path: Path) -> None:
    write_smesh(path, anomaly=False)


def write_vcorr(path: Path, *, lh: float = LH, lv: float = LV) -> None:
    # :g，避免 0.35 被 .1f 收成 0.3
    path.write_text(
        "2 2\n"
        f"{XMIN:.0f} {XMAX:.0f}\n"
        "0.0 0.0\n"
        f"0.0 {ZMAX:g}\n"
        f"{lh:g} {lh:g}\n"
        f"{lh:g} {lh:g}\n"
        f"{lv:g} {lv:g}\n"
        f"{lv:g} {lv:g}\n",
        encoding="utf-8",
    )


def write_geom(path: Path, codes: tuple[int, ...]) -> None:
    lines = [str(len(OBS_XS))]
    for ox in OBS_XS:
        recs = [
            _fmt_r_line(sx, SHOT_Z, kind, 0.0, TT_ERR)
            for kind in codes
            for sx in iter_shots(ox, kind)
        ]
        lines.append(_fmt_s_line(ox, H, len(recs)))
        lines.extend(recs)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_smesh(path: Path) -> tuple[list[float], list[float], list[list[float]]]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    nx, nz, _vw, _va = lines[0].split()
    xs = [float(x) for x in lines[1].split()]
    zs = [float(z) for z in lines[3].split()]
    vel: list[list[float]] = []
    for i in range(int(nx)):
        vel.append([float(v) for v in lines[4 + i].split()])
        if len(vel[-1]) != int(nz):
            raise ValueError(f"{path}: column {i} has {len(vel[-1])} z values, expected {nz}")
    if len(xs) != int(nx) or len(zs) != int(nz):
        raise ValueError(f"{path}: nx/nz mismatch")
    return xs, zs, vel


def node_stats(
    xs: list[float],
    zs: list[float],
    vel: list[list[float]],
    *,
    x_lo: float,
    x_hi: float,
    layer: str,
) -> tuple[float, float, float, int]:
    vals: list[float] = []
    for i, x in enumerate(xs):
        if x < x_lo - 1e-9 or x > x_hi + 1e-9:
            continue
        for k, z in enumerate(zs):
            if layer == "water":
                ok = z < H - 1e-9
            elif layer == "crust":
                ok = H + 1e-9 < z <= H_MOHO + 1e-9
            elif layer == "mantle":
                ok = z > H_MOHO + 1e-9
            else:
                raise ValueError(layer)
            if ok:
                vals.append(vel[i][k])
    if not vals:
        raise ValueError("no nodes in requested window")
    mean = sum(vals) / len(vals)
    return mean, min(vals), max(vals), len(vals)


def box_stats(
    xs: list[float],
    zs: list[float],
    vel: list[list[float]],
    *,
    x_lo: float,
    x_hi: float,
    z_lo: float,
    z_hi: float,
) -> tuple[float, float, float, int]:
    vals: list[float] = []
    for i, x in enumerate(xs):
        if x < x_lo - 1e-9 or x > x_hi + 1e-9:
            continue
        for k, z in enumerate(zs):
            if z < z_lo - 1e-9 or z > z_hi + 1e-9:
                continue
            vals.append(vel[i][k])
    if not vals:
        raise ValueError("no nodes in box")
    return sum(vals) / len(vals), min(vals), max(vals), len(vals)


def crust_field_compare(
    xs: list[float],
    zs: list[float],
    va: list[list[float]],
    vb: list[list[float]],
    *,
    x_lo: float,
    x_hi: float,
) -> tuple[float, float, int]:
    da: list[float] = []
    db: list[float] = []
    for i, x in enumerate(xs):
        if x < x_lo - 1e-9 or x > x_hi + 1e-9:
            continue
        for k, z in enumerate(zs):
            if not (H + 1e-9 < z <= H_MOHO + 1e-9):
                continue
            da.append(va[i][k])
            db.append(vb[i][k])
    n = len(da)
    if n < 2:
        raise ValueError("too few crust nodes")
    rms = math.sqrt(sum((a - b) ** 2 for a, b in zip(da, db)) / n)
    ma = sum(da) / n
    mb = sum(db) / n
    cov = sum((a - ma) * (b - mb) for a, b in zip(da, db))
    sa = math.sqrt(sum((a - ma) ** 2 for a in da))
    sb = math.sqrt(sum((b - mb) ** 2 for b in db))
    corr = cov / (sa * sb) if sa > 0.0 and sb > 0.0 else 0.0
    return rms, corr, n


def illum_x_range() -> tuple[float, float]:
    xs = [
        sx
        for ox in OBS_XS
        for code in (0, 4)
        for sx in iter_shots(ox, code)
    ]
    return min(xs) - DX, max(xs) + DX


def main() -> None:
    write_true(HERE / "true.smesh")
    write_start(HERE / "start.smesh")
    write_seafloor(HERE / "seafloor.refl")
    write_moho(HERE / "moho.refl")
    write_geom(HERE / "geom_inv_01.dat", (0, 1))
    write_geom(HERE / "geom_inv_45.dat", (4, 5))
    write_geom(HERE / "geom_inv_0145.dat", (0, 1, 4, 5))
    write_vcorr(HERE / "vcorr.dat")
    xs, zs, vt = parse_smesh(HERE / "true.smesh")
    _, _, vs = parse_smesh(HERE / "start.smesh")
    ta, *_ = box_stats(xs, zs, vt, x_lo=AX0, x_hi=AX1, z_lo=AZ0, z_hi=AZ1)
    sa, *_ = box_stats(xs, zs, vs, x_lo=AX0, x_hi=AX1, z_lo=AZ0, z_hi=AZ1)
    print(f"wrote {HERE}")
    print(
        f"  true.smesh   背景壳 {V_CRUST0}+{V_GRAD}*(z-H)  "
        f"高速体 x=[{AX0:g},{AX1:g}] z=[{AZ0:g},{AZ1:g}] +{DV_ANOM:g}"
    )
    print(f"  start.smesh  无异常背景  框内均值 真 {ta:.3f} / 初 {sa:.3f}")
    print("  seafloor.refl  (-Y)  moho.refl (-F)")
    n01 = sum(1 for ox in OBS_XS for _ in iter_shots(ox, 0))
    n45 = sum(1 for ox in OBS_XS for _ in iter_shots(ox, 4))
    print(
        f"  geom_inv_01.dat / _45.dat / _0145.dat  "
        f"OBS {OBS_XS}  0/1 ±{max(OFFSETS_01):g}（{n01} 炮/相位）  "
        f"4/5 ±{max(OFFSETS_45):g}（{n45} 炮/相位，贴边丢掉）  "
        f"炮∈[{XMIN + SHOT_PAD:g},{XMAX - SHOT_PAD:g}]"
    )
    print(f"  vcorr.dat     Lh={LH:g} Lv={LV:g}  （-CV，配 -SV{WSV:g} -TV{TV:g}）")
    print(f"  拾取误差 TT_ERR={TT_ERR:g}（正演后对 syn_inv_*.dat 跑 set_pick_uncert）")


if __name__ == "__main__":
    main()
