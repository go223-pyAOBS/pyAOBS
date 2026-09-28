#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vs 段工区：6 PSP 转折 + 12 PSP-Moho。冻真 Vp；莫霍可动。不改 cmp_0178。

真模型 = TRUE_1D + BEL_LVZ + 起伏莫霍。
初值面下 = START_1D（整体偏快、弱梯度、无 LVZ），跳变在光滑初值莫霍；盖层贴真 Vs。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "inv_2d"))
sys.path.insert(0, str(HERE.parents[2]))
import inv_grid as g  # noqa: E402
import make_fwd_all as fwd  # noqa: E402
import make_pmp as pmp  # noqa: E402
import make_ppp_psp_inv_case as m2  # noqa: E402
from make_lvz import write_smesh  # noqa: E402

SRC = HERE / "cmp_0178"
DEST = HERE / "inv_612"
DX_MIN = 10.0
CODES = (6, 12)
XMAX = 150.0
DX = 2.0
# 面下 LVZ 在 x=58。5 台、20 km 间距，模型 0–150 km。
OBS_XS = (30.0, 50.0, 70.0, 90.0, 110.0)
SHOT_XS = tuple(float(x) for x in range(2, 147, 8))
VCORR_LH, VCORR_LV = 4.0, 1.2
# 基底/转换面最深 4 km。cmp_0178 的 0.03 斜率在 150 km 工区右侧会到 ~7 km。
Z_CONV_MAX = 4.0
z_conv_uncapped = g.z_conv


def z_conv(x: float) -> float:
    return min(Z_CONV_MAX, z_conv_uncapped(x))


def write_vcorr(path: Path, lh: float, lv: float) -> None:
    path.write_text(
        f"2 2\n0 {XMAX:.0f}\n0.0 0.0\n0.0 16.0\n"
        f"{lh:.1f} {lh:.1f}\n{lh:.1f} {lh:.1f}\n"
        f"{lv:.1f} {lv:.1f}\n{lv:.1f} {lv:.1f}\n",
        encoding="utf-8",
    )


def _old_1d_col(x: float, zs: list[float]) -> list[float]:
    zc = z_conv_uncapped(x)
    h_lid = max(zc - g.H, 0.2)
    col: list[float] = []
    for z in zs:
        if z <= g.H + 1e-9:
            col.append(g.V_WATER)
        elif z < zc - 1e-9:
            col.append(fwd.OLD_LID0 + (fwd.OLD_LID_VP - fwd.OLD_LID0) * (z - g.H) / h_lid)
        else:
            col.append(fwd.OLD_CRUST0 + fwd.OLD_CRUST_GRAD * (z - zc))
    return col


def extend_smesh(path: Path) -> int:
    xs, zs, vel = m2.parse_smesh(path)
    extra: list[float] = []
    x = xs[-1] + DX
    while x <= XMAX + 1e-9:
        extra.append(round(x, 4))
        x += DX
    if not extra:
        return 0
    for xv in extra:
        xs.append(xv)
        vel.append(_old_1d_col(xv, zs))
    write_smesh(path, xs, zs, vel)
    return len(extra)


def extend_refl(path: Path, zfun) -> int:
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        a = ln.split()
        if len(a) >= 2:
            rows.append((float(a[0]), float(a[1])))
    extra = 0
    x = rows[-1][0] + DX
    while x <= XMAX + 1e-9:
        rows.append((round(x, 4), float(zfun(x))))
        extra += 1
        x += DX
    if extra:
        path.write_text("".join(f"{a:.4f} {b:.4f}\n" for a, b in rows), encoding="utf-8")
    return extra


def write_refl(path: Path, xs: list[float], zfun) -> None:
    path.write_text("".join(f"{x:.4f} {float(zfun(x)):.4f}\n" for x in xs), encoding="utf-8")


def _xs_of(path: Path) -> list[float]:
    return [float(ln.split()[0]) for ln in path.read_text(encoding="utf-8").splitlines() if ln.split()]


def main() -> int:
    if not (SRC / "true_vp.smesh").is_file():
        raise SystemExit(f"missing {SRC} (run make_lvz.py / make_pmp.py first)")
    DEST.mkdir(parents=True, exist_ok=True)
    for name in (
        "true_vp.smesh",
        "seafloor.refl",
        "conv.refl",
        "vcorr.dat",
        "start_vp.smesh",
    ):
        shutil.copyfile(SRC / name, DEST / name)
    shutil.copyfile(SRC / "moho_true.refl", DEST / "moho_true.refl")
    shutil.copyfile(SRC / "moho.refl", DEST / "moho.refl")
    n_x = extend_smesh(DEST / "true_vp.smesh")
    extend_smesh(DEST / "start_vp.smesh")
    extend_refl(DEST / "seafloor.refl", lambda _x: g.H)
    xs = _xs_of(DEST / "seafloor.refl")
    write_refl(DEST / "conv.refl", xs, z_conv)
    extend_refl(DEST / "moho_true.refl", pmp.z_moho_true)
    extend_refl(DEST / "moho.refl", pmp.z_moho_start)
    write_vcorr(DEST / "vcorr.dat", VCORR_LH, VCORR_LV)
    zc_vals = [z_conv(x) for x in xs]
    print(
        f"  网格扩到 0–{XMAX:.0f} km  +{n_x} 列  OBS={OBS_XS}  nshot={len(SHOT_XS)}"
    )
    print(
        f"  转换面封顶 {Z_CONV_MAX:.1f} km  "
        f"zc(50)={z_conv(50):.2f}  zc(150)={z_conv(150):.2f}  max={max(zc_vals):.2f}"
    )
    g.z_conv = z_conv
    fwd.reshape_crust_mantle(
        DEST / "true_vp.smesh",
        DEST / "moho_true.refl",
        z_conv=z_conv,
        z_conv_old=z_conv_uncapped,
    )
    fwd.reshape_crust_mantle(
        DEST / "start_vp.smesh",
        DEST / "moho.refl",
        keep_anom=False,
        bg=fwd.START_1D,
        z_conv=z_conv,
    )
    n_lid = g.graft_lid_vp(DEST / "true_vs.smesh", DEST / "start_vs.smesh")
    s1 = fwd.START_1D
    print(
        f"  start_vs 盖层 <- 真 Vs  n={n_lid}  面下 = START_1D "
        f"{s1.crust0:g}→{s1.crust_moho:g} / mantle {s1.mantle0:g}+{s1.mantle_grad:g} "
        f"（无 LVZ，整体偏快，跳变在初值莫霍）"
    )
    print(f"  vcorr Lh={VCORR_LH:g} Lv={VCORR_LV:g}  OBS={OBS_XS}")

    recs_by_obs: dict[float, list[str]] = {}
    nrec = 0
    for ox in OBS_XS:
        recs = []
        for code in CODES:
            for x in SHOT_XS:
                if abs(x - ox) <= DX_MIN:
                    continue
                recs.append(f"r  {x:8.3f}     0.010 {code:4d}     0.000     0.050")
        recs_by_obs[ox] = recs
        nrec += len(recs)
    lines = [str(len(OBS_XS))]
    for ox in OBS_XS:
        recs = recs_by_obs[ox]
        lines.append(f"s  {ox:8.3f}     2.000 {len(recs):4d}")
        lines.extend(recs)
    (DEST / "geom_612.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"wrote {DEST}  nsrc={len(OBS_XS)} nrec_total={nrec}  "
        f"|Δx|>{DX_MIN:.0f}  codes={CODES}"
    )
    print(f"  start Moho (x=50) {pmp.z_moho_start(50):.2f} km  true {pmp.z_moho_true(50):.2f} km")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
