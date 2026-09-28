#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PPP+PSP：盖层底 Vp=4.0；面下顶 7.2，使折合 Vs=7.2/1.73≈4.16 > 4.0。

传统两步：PPP 反 Vp → 面下 Vp/1.73 → -DQ 钉盖层，混合网格当 0 反。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ps_inv"))
import make_ps_inv_case as base  # noqa: E402

V_WATER = base.V_WATER
H = base.H
Z_CONV = base.Z_CONV
XMIN, XMAX, DX = base.XMIN, base.XMAX, base.DX
ZMAX, DZ = base.ZMAX, base.DZ
KAPPA_TRUE = base.KAPPA_TRUE
KAPPA_START = base.KAPPA_START
# 盖层底 4.00。面下顶 7.20，折合 Vs=7.20/1.73≈4.16 > 4.00，初至才能进面下。
LID_VP = 4.00
CRUST_VP = 7.20
VP_TRUE = dict(
    sed0=1.80,
    sed_grad=(LID_VP - 1.80) / (Z_CONV - H),
    crust0=CRUST_VP,
    crust_grad=KAPPA_TRUE * 0.12,
)
VP_START = dict(
    sed0=2.10,
    sed_grad=(3.85 - 2.10) / (Z_CONV - H),
    crust0=CRUST_VP - 0.18,
    crust_grad=0.10,
)
OBS_XS = base.OBS_XS
OBS_Z = base.OBS_Z
SHOT_Z = base.SHOT_Z
SHOT_DX = base.SHOT_DX
SHOT_XS = base.SHOT_XS
LH, LV = base.LH, base.LV
WSV, WTV = base.WSV, base.WTV
PICK_U = base.PICK_U
CODES_PPP = base.CODES_PPP
CODES = (6,)
CODES_JOINT = (0, 6)
# 传统折合第二步：PSP 走时当折射 0 算（原始 tomo2d 没有 raytype 6）。
CODES_PSP0 = (0,)

write_vp = base.write_vp
write_vs = base.write_vs
write_seafloor = base.write_seafloor
write_conv = base.write_conv
write_vcorr = base.write_vcorr
write_geom = base.write_geom
write_vs_from_smesh = base.write_vs_from_smesh
parse_smesh = base.parse_smesh
node_stats = base.node_stats
illum_x_range = base.illum_x_range
vs_at = base.vs_at

# 传统两步：不用 -B，-DQ 面上 1000、面下 30。
DAMP_LID = 1000.0
DAMP_BELOW = 30.0


def write_lid_damp(
    path: Path,
    *,
    w_lid: float = DAMP_LID,
    w_below: float = DAMP_BELOW,
    w_water: float | None = None,
    z_conv: float = Z_CONV,
) -> None:
    """写 ``-DQ`` 空间阻尼：转换面以上 ``w_lid``，面上及以下 ``w_below``。"""
    if w_water is None:
        w_water = w_lid
    xs = base.fwd._xs()
    zs = base.fwd._zs()
    lines = [
        f"{len(xs)} {len(zs)}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    for _x in xs:
        row = []
        for z in zs:
            if z <= H + 1e-9:
                row.append(f"{w_water:.6g}")
            elif z < z_conv - 1e-9:
                row.append(f"{w_lid:.6g}")
            else:
                row.append(f"{w_below:.6g}")
        lines.append(" ".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_damp(path: Path) -> tuple[list[float], list[float], list[list[float]]]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    nx, nz = (int(x) for x in lines[0].split()[:2])
    xs = [float(x) for x in lines[1].split()]
    zs = [float(z) for z in lines[3].split()]
    w: list[list[float]] = []
    for i in range(nx):
        w.append([float(v) for v in lines[4 + i].split()])
        if len(w[-1]) != nz:
            raise ValueError(f"{path}: column {i} has {len(w[-1])} z values, expected {nz}")
    if len(xs) != nx or len(zs) != nz:
        raise ValueError(f"{path}: nx/nz mismatch")
    return xs, zs, w


def write_mixed_smesh(
    cover_path: Path,
    below_path: Path,
    out_path: Path,
    *,
    z_conv: float = Z_CONV,
    below_kappa: float = 1.0,
) -> None:
    """折合网格：盖层取 cover（Vp），面下取 below/κ（Vs）。κ=1.73 即原始 -Cb。"""
    xs, zs, cover = parse_smesh(cover_path)
    xs2, zs2, below = parse_smesh(below_path)
    if xs2 != xs or zs2 != zs:
        raise ValueError("mixed grid mismatch")
    if below_kappa <= 0.0:
        raise ValueError("below_kappa must be positive")
    lines = [
        f"{len(xs)} {len(zs)} {V_WATER:.4f} {base.fwd.V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    for i, col in enumerate(cover):
        row = []
        for k, z in enumerate(zs):
            if z < z_conv - 1e-9:
                row.append(f"{col[k]:.4f}")
            else:
                row.append(f"{(below[i][k] / below_kappa):.4f}")
        lines.append(" ".join(row))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_vs_from_vp_below(
    vp_path: Path,
    out_path: Path,
    kappa: float,
    *,
    lid_vs_path: Path | None = None,
    z_conv: float = Z_CONV,
) -> None:
    """盖层下 ``Vs=Vp/κ``；盖层保持 ``lid_vs``（缺省也按 κ 转）；水保持水速。"""
    xs, zs, vp = parse_smesh(vp_path)
    lid = None
    if lid_vs_path is not None:
        xs2, zs2, lid = parse_smesh(lid_vs_path)
        if xs2 != xs or zs2 != zs:
            raise ValueError("lid Vs grid mismatch")
    lines = [
        f"{len(xs)} {len(zs)} {V_WATER:.4f} {base.fwd.V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    for i, col in enumerate(vp):
        vs = []
        for k, (z, v) in enumerate(zip(zs, col)):
            if z <= H + 1e-9:
                vs.append(f"{V_WATER:.4f}")
            elif z < z_conv - 1e-9 and lid is not None:
                vs.append(f"{lid[i][k]:.4f}")
            else:
                vs.append(f"{(v / kappa):.4f}")
        lines.append(" ".join(vs))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def relabel_syn_raytype(src: Path, dst: Path, *, frm: int, to: int) -> int:
    """把 syn/geom 风格文件里 r 行的 raytype ``frm`` 改成 ``to``。"""
    n = 0
    out: list[str] = []
    for ln in src.read_text(encoding="utf-8").splitlines():
        parts = ln.split()
        if parts and parts[0] == "r" and int(float(parts[3])) == frm:
            parts[3] = str(to)
            n += 1
            out.append(" ".join(parts))
        else:
            out.append(ln)
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    return n


def iface_vp(*, start: bool = False) -> tuple[float, float, float]:
    """(盖层底解析 Vp, 结点 z=4.8 Vp, 面下顶 z=5.0 Vp)。"""
    kw = VP_START if start else VP_TRUE
    lid_face = kw["sed0"] + kw["sed_grad"] * (Z_CONV - H)
    lid_node = kw["sed0"] + kw["sed_grad"] * (4.8 - H)
    below0 = kw["crust0"]
    return lid_face, lid_node, below0


def main() -> None:
    here = Path(__file__).resolve().parent
    write_vp(here / "true_vp.smesh", **VP_TRUE)
    write_vp(here / "start_vp.smesh", **VP_START)
    write_vp(here / "vp.smesh", **VP_START)
    write_vs(here / "true_vs.smesh", KAPPA_TRUE, **VP_TRUE)
    write_vs(here / "start_vs.smesh", KAPPA_START, **VP_START)
    write_mixed_smesh(
        here / "true_vp.smesh", here / "true_vp.smesh", here / "true_mixed.smesh",
        below_kappa=KAPPA_TRUE,
    )
    write_mixed_smesh(
        here / "start_vp.smesh", here / "start_vp.smesh", here / "start_mixed.smesh",
        below_kappa=KAPPA_TRUE,
    )
    write_seafloor(here / "seafloor.refl")
    write_conv(here / "conv.refl")
    write_geom(here / "geom_ppp.dat", codes=CODES_PPP)
    write_geom(here / "geom_inv.dat", codes=CODES)
    write_geom(here / "geom_psp0.dat", codes=CODES_PSP0)
    write_geom(here / "geom_joint.dat", codes=CODES_JOINT)
    write_vcorr(here / "vcorr.dat")
    write_lid_damp(here / "damp_lid.dat")
    t_face, t_node, t_below = iface_vp()
    s_face, s_node, s_below = iface_vp(start=True)
    print(f"wrote {here}")
    print(f"  geom_joint.dat  codes {list(CODES_JOINT)}  （同一次 PPP+PSP）")
    print(f"  geom_psp0.dat   codes {list(CODES_PSP0)}  （折合第二步：PSP 当 0）")
    print(
        f"  true Vp  盖层底 {t_face:.2f}（结点 {t_node:.2f}）  面下顶 {t_below:.2f}  "
        f"跳 {t_below - t_face:.2f}"
    )
    print(
        f"  start Vp 盖层底 {s_face:.2f}（结点 {s_node:.2f}）  面下顶 {s_below:.2f}"
    )
    print(f"  start_vs.smesh  start_vp / {KAPPA_START:g}  （双场对照）")
    print(
        f"  折合 mixed  盖层底 Vp={t_face:.2f}  面下 Vs={t_below / KAPPA_TRUE:.2f}  "
        f"（> {t_face:.2f}，初至可下潜）"
    )
    print(f"  vcorr.dat     Lh={LH:g} Lv={LV:g}")
    print(f"  damp_lid.dat  盖层 w={DAMP_LID:.3g}  面下 w={DAMP_BELOW:g}")


if __name__ == "__main__":
    main()
