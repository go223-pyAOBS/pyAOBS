#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""正演工区：全部震相 0/1/6–15。

cmp_0178 的真模型里莫霍只是 -F 运动学反射面，面下按 7.20+0.21(z−zc) 连续爬到 ~8.5。
fwd_all 把盖层底 Vp 改成 3.50，面下 Vs(zc+)=3.50（Vp=6.06），壳底 7.40，莫霍下 8.00+0.10(z−zm)。
反演初值用 START_1D（整体偏快、弱梯度、无 LVZ），不要写成与真背景同一套端点。
不改 cmp_0178。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "inv_2d"))
sys.path.insert(0, str(HERE.parents[2]))
import inv_grid as g  # noqa: E402
import make_ppp_psp_inv_case as m2  # noqa: E402

SRC = HERE / "cmp_0178"
DEST = HERE / "fwd_all"
OBS_XS = (30.0, 50.0)
# 网格 0–124 km。OBS 50：Δx 约 −48…+60 km。
SHOT_XS = tuple(float(x) for x in range(2, 107, 8)) + (110.0,)
CODES = (0, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
OLD_CRUST0 = 7.20
OLD_LID0 = 1.80
OLD_LID_VP = 4.00
LID_VP_IFACE = 3.50
CRUST0 = g.KAPPA * LID_VP_IFACE  # 面下 Vs(zc+)=面上 Vp
CRUST_MOHO = 7.40
MANTLE0 = 8.00
MANTLE_GRAD = 0.10
OLD_CRUST_GRAD = g.KAPPA * 0.12


class Vel1D(NamedTuple):
    lid0: float
    lid_iface: float
    crust0: float
    crust_moho: float
    mantle0: float
    mantle_grad: float


TRUE_1D = Vel1D(OLD_LID0, LID_VP_IFACE, CRUST0, CRUST_MOHO, MANTLE0, MANTLE_GRAD)
# 初值 1D：整体偏快 ~0.3–0.6 km/s Vp，梯度更弱，无 LVZ。
# 用来拉开大尺度差；真模型仍是 TRUE_1D + 低速体。
START_1D = Vel1D(2.40, 3.80, 6.55, 7.70, 8.30, 0.05)
COPY = (
    "true_vp.smesh",
    "true_vs.smesh",
    "seafloor.refl",
    "conv.refl",
    "moho_true.refl",
)


def _load_xz(path: Path) -> tuple[list[float], list[float]]:
    xs, zs = [], []
    for raw in path.read_text(encoding="utf-8").splitlines():
        a = raw.split()
        if len(a) >= 2:
            xs.append(float(a[0]))
            zs.append(float(a[1]))
    return xs, zs


def _interp(x: float, xs: list[float], zs: list[float]) -> float:
    if x <= xs[0]:
        return zs[0]
    if x >= xs[-1]:
        return zs[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            dx = xs[i + 1] - xs[i]
            t = 0.0 if abs(dx) < 1e-12 else (x - xs[i]) / dx
            return zs[i] + t * (zs[i + 1] - zs[i])
    return zs[-1]


def reshape_crust_mantle(
    vp_p: Path,
    moho_p: Path,
    vs_p: Path | None = None,
    *,
    keep_anom: bool = True,
    bg: Vel1D | None = None,
    z_conv=None,
    z_conv_old=None,
) -> None:
    """按 bg 重铺盖层/地壳/地幔 1D。默认真模型 TRUE_1D，初值 START_1D。

    keep_anom=True：在旧背景上改梯度，保留低速异常（真模型）。
    keep_anom=False：写成 bg 的 1D（初值，无异常）。
    z_conv / z_conv_old：新/旧转换面。界面变浅时用旧面剥异常，再用新面铺背景。
    """
    if bg is None:
        bg = TRUE_1D if keep_anom else START_1D
    zc_new_fn = z_conv or g.z_conv
    zc_old_fn = z_conv_old or zc_new_fn
    mx, mz = _load_xz(moho_p)
    xs, zs, vel = m2.parse_smesh(vp_p)
    n_lid = n_crust = n_mantle = 0
    lines = [
        f"{len(xs)} {len(zs)} {g.V_WATER:.4f} {g.V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    for i, x in enumerate(xs):
        zc = zc_new_fn(x)
        zc_old = zc_old_fn(x)
        zm = _interp(x, mx, mz)
        h = max(zm - zc, 0.4)
        h_lid = max(zc - g.H, 0.2)
        h_lid_old = max(zc_old - g.H, 0.2)
        row = []
        for k, z in enumerate(zs):
            v = vel[i][k]
            if z <= g.H + 1e-9:
                row.append(f"{v:.4f}")
                continue
            if z < zc - 1e-9:
                new_lid = bg.lid0 + (bg.lid_iface - bg.lid0) * (z - g.H) / h_lid
                if keep_anom and z < zc_old - 1e-9:
                    old_lid = OLD_LID0 + (OLD_LID_VP - OLD_LID0) * (z - g.H) / h_lid_old
                    v = v - old_lid + new_lid
                else:
                    v = new_lid
                n_lid += 1
                row.append(f"{max(v, 1.55):.4f}")
                continue
            if z < zm - 1e-9:
                new_bg = bg.crust0 + (bg.crust_moho - bg.crust0) * (z - zc) / h
                if keep_anom and z >= zc_old - 1e-9:
                    old_bg = OLD_CRUST0 + OLD_CRUST_GRAD * (z - zc_old)
                    v = v - old_bg + new_bg
                else:
                    v = new_bg
                n_crust += 1
            else:
                v = bg.mantle0 + bg.mantle_grad * (z - zm)
                n_mantle += 1
            row.append(f"{max(v, 4.80):.4f}")
        lines.append(" ".join(row))
    vp_p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    vs_out = vs_p or vp_p.with_name(vp_p.name.replace("vp.smesh", "vs.smesh"))
    g.write_vs_from_vp(vp_p, vs_out, g.KAPPA)
    _, _, newv = m2.parse_smesh(vp_p)
    tag = "true" if keep_anom else "start"
    print(
        f"  {vp_p.name} [{tag}] lid {bg.lid0:.2f}→{bg.lid_iface:.2f} n={n_lid}  "
        f"crust {bg.crust0:.2f}→{bg.crust_moho:.2f} (Vs0={bg.crust0 / g.KAPPA:.2f})  n={n_crust}  "
        f"mantle {bg.mantle0:.2f}+{bg.mantle_grad:.2f}(z-zm)  n={n_mantle}  vs→{vs_out.name}"
    )
    for x0 in (30.0, 50.0):
        ix = min(range(len(xs)), key=lambda j: abs(xs[j] - x0))
        zc = zc_new_fn(xs[ix])
        zm = _interp(xs[ix], mx, mz)
        iz_lid = min(range(len(zs)), key=lambda j: abs(zs[j] - (zc - 0.2)))
        izc = min(range(len(zs)), key=lambda j: abs(zs[j] - (zc + 0.2)))
        izm0 = min(range(len(zs)), key=lambda j: abs(zs[j] - (zm - 0.2)))
        izm1 = min(range(len(zs)), key=lambda j: abs(zs[j] - (zm + 0.2)))
        print(
            f"  x={xs[ix]:.0f}  面上Vp={newv[ix][iz_lid]:.3f}  "
            f"面下Vp/Vs={newv[ix][izc]:.3f}/{newv[ix][izc] / g.KAPPA:.3f}  "
            f"moho-={newv[ix][izm0]:.3f}  moho+={newv[ix][izm1]:.3f}"
        )


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in COPY:
        sp = SRC / name
        if not sp.is_file():
            raise SystemExit(f"missing {sp} (run make_pmp.py / cmp_0178 first)")
        shutil.copyfile(sp, DEST / name)
    reshape_crust_mantle(DEST / "true_vp.smesh", DEST / "moho_true.refl")
    recs = []
    for code in CODES:
        for x in SHOT_XS:
            recs.append(f"r  {x:8.3f}     0.010 {code:4d}     0.000     0.050")
    lines = [str(len(OBS_XS))]
    for ox in OBS_XS:
        lines.append(f"s  {ox:8.3f}     2.000 {len(recs):4d}")
        lines.extend(recs)
    (DEST / "geom_all_ph.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {DEST}  nsrc={len(OBS_XS)} nrec/src={len(recs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
