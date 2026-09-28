#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""崎岖转换面：二维积分转折枝当观测，传统两步 tt_inverse（图论初至）。

真模型 = rugged fast（Vs0≈4.16 > 盖层底 Vp），与 PPP+PSP_inv2 同一套「面下够快才能下潜」。
观测：二维折线射击 PPP(0) 与 PSP(6，写入 syn 时标成 0)。
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "thin2km"))
sys.path.insert(0, str(HERE.parent.parent / "thin2km_incline"))
sys.path.insert(0, str(HERE.parents[2]))  # PPP+PSP_inv2
sys.path.insert(0, str(HERE.parents[3] / "ps_inv"))

import run_thin2km_rugged as rug  # noqa: E402  patches z_conv
import run_thin2km_incline as inc  # noqa: E402
import make_ppp_psp_inv_case as m2  # noqa: E402

H, KAPPA = inc.H, inc.KAPPA
SED0, SED_GRAD = inc.SED0, inc.SED_GRAD
OBS_XS, SHOT_XS, SHOT_Z = inc.OBS_XS, inc.SHOT_XS, inc.SHOT_Z
V_WATER, V_AIR = inc.V_WATER, inc.V_AIR
MODEL_NAME = "fast"
CRUST_TRUE = 7.20
CRUST_START = 7.02
SED0_START = 2.20
SED_GRAD_START = 0.40
PICK_U = 0.05


def zc(x: float) -> float:
    return rug.z_conv(x)


def vp_true(x: float, z: float) -> float:
    zi = zc(x)
    if z <= H + 1e-12:
        return V_WATER
    if z < zi:
        return SED0 + SED_GRAD * (z - H)
    return CRUST_TRUE + KAPPA * 0.12 * (z - zi)


def vp_start(x: float, z: float) -> float:
    zi = zc(x)
    if z <= H + 1e-12:
        return V_WATER
    if z < zi:
        return SED0_START + SED_GRAD_START * (z - H)
    return CRUST_START + 0.10 * (z - zi)


def write_mixed(cover: Path, below: Path, out: Path, kappa: float) -> None:
    xs, zs, cov = m2.parse_smesh(cover)
    xs2, zs2, bel = m2.parse_smesh(below)
    if xs2 != xs or zs2 != zs:
        raise SystemExit("mixed grid mismatch")
    lines = [
        f"{len(xs)} {len(zs)} {V_WATER:.4f} {V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    for i, x in enumerate(xs):
        zi = zc(x)
        row = []
        for k, z in enumerate(zs):
            if z < zi - 1e-9:
                row.append(f"{cov[i][k]:.4f}")
            else:
                row.append(f"{(bel[i][k] / kappa):.4f}")
        lines.append(" ".join(row))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_damp(path: Path) -> None:
    xs, zs = inc._xs(), inc._zs()
    lines = [
        f"{len(xs)} {len(zs)}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    for x in xs:
        zi = zc(x)
        row = []
        for z in zs:
            if z <= H + 1e-9:
                row.append(f"{m2.DAMP_LID:.6g}")
            elif z < zi - 1e-9:
                row.append(f"{m2.DAMP_LID:.6g}")
            else:
                row.append(f"{m2.DAMP_BELOW:.6g}")
        lines.append(" ".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_geom(path: Path, codes: tuple[int, ...]) -> None:
    inc.t.write_geom(path, codes)


def write_syn(path: Path, rays: list, *, out_code: int) -> int:
    by_obs: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for r in rays:
        by_obs[round(r.obs_x, 3)].append((r.obs_x + r.offset, r.t))
    lines = [str(sum(1 for ox in OBS_XS if by_obs[round(ox, 3)]))]
    n = 0
    for ox in OBS_XS:
        picks = by_obs[round(ox, 3)]
        if not picks:
            continue
        picks.sort()
        lines.append(f"s {ox:g} {H:g} {len(picks)}")
        for sx, tt in picks:
            lines.append(f"r {sx:g} {SHOT_Z:g} {out_code} {tt:.5f} {PICK_U:g}")
            n += 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n


def shoot_codes(md: inc.Model, codes: tuple[int, ...]):
    xs, zs = inc.conv_polyline()
    spec = inc.r2.Ray2dSpec.from_xy(md.crust_vp, md.vs0, xs, zs)
    flags = {c: (bs, lu) for c, _n, bs, lu in inc.PHASES}
    out = []
    for code in codes:
        n_ok = 0
        bs, lu = flags[code]
        print(f"  2d shoot code {code} …", flush=True)
        for ox in OBS_XS:
            for sx in SHOT_XS:
                got = inc.r2.shoot_pair(spec, sx, ox, code)
                if got is None:
                    continue
                n_ok += 1
                out.append(
                    inc.t.ShotRay(
                        code, got["xs"], got["zs"], abs(got["px0"]),
                        got["t"], got["zmax"], sx - ox, lu, bs, ox,
                    )
                )
        print(f"    hit {n_ok}/{len(OBS_XS) * len(SHOT_XS)}", flush=True)
    return out


def latest(glob_pat: str) -> Path:
    cands = sorted(HERE.glob(glob_pat))
    if not cands:
        raise SystemExit(f"no {glob_pat}")
    return max(cands, key=lambda p: (int(p.name.split(".")[-2]), int(p.name.split(".")[-1])))


def main() -> int:
    inc.write_smesh(HERE / "true_vp.smesh", vp_true)
    inc.write_smesh(HERE / "start_vp.smesh", vp_start)
    write_mixed(HERE / "true_vp.smesh", HERE / "true_vp.smesh", HERE / "true_mixed.smesh", KAPPA)
    write_mixed(HERE / "start_vp.smesh", HERE / "start_vp.smesh", HERE / "start_mixed.smesh", KAPPA)
    inc.t.write_iface(HERE / "seafloor.refl", H)
    inc.write_conv(HERE / "conv.refl")
    write_damp(HERE / "damp_lid.dat")
    m2.write_vcorr(HERE / "vcorr.dat")
    write_geom(HERE / "geom_ppp.dat", (0,))
    write_geom(HERE / "geom_psp0.dat", (0,))
    md = inc.Model(MODEL_NAME, CRUST_TRUE)
    print(f"true Vs0={md.vs0:.2f}  lidVp(50)={inc.lid_vp_at(50):.2f}  zc(30)={zc(30):.2f} zc(70)={zc(70):.2f}")
    ppp = shoot_codes(md, (0,))
    psp = shoot_codes(md, (6,))
    n0 = write_syn(HERE / "syn_ppp.dat", ppp, out_code=0)
    n6 = write_syn(HERE / "syn_inv.dat", psp, out_code=0)
    print(f"wrote syn_ppp.dat n={n0}  syn_inv.dat n={n6} (PSP as type 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
