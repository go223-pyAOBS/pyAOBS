#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""二维非均匀真模型：崎岖转换面 + 盖层/面下各一块低速区。初值仍是 1D 梯度（无异常）。"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inv_2d"))
from make_fair_grids import prepare, write_geom_code, write_geom_codes  # noqa: E402
import inv_grid as g  # noqa: E402
import make_ppp_psp_inv_case as m2  # noqa: E402

# 盖层低速（壳）：台 40 下方。面下低速（幔）：x≈58、z≈8。
LID_LVZ = dict(x0=40.0, z0=3.10, rx=5.0, rz=0.45, amp=-0.40)
BEL_LVZ = dict(x0=58.0, z0=8.00, rx=7.0, rz=1.80, amp=-0.70)


def _gauss(x: float, z: float, spec: dict) -> float:
    return spec["amp"] * math.exp(
        -((x - spec["x0"]) / spec["rx"]) ** 2 - ((z - spec["z0"]) / spec["rz"]) ** 2
    )


def write_smesh(path: Path, xs, zs, vel) -> None:
    lines = [
        f"{len(xs)} {len(zs)} {g.V_WATER:.4f} {g.V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    for col in vel:
        lines.append(" ".join(f"{v:.4f}" for v in col))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_lvz(vp_path: Path) -> dict:
    xs, zs, vel = m2.parse_smesh(vp_path)
    n_lid = n_bel = 0
    min_below_vs = 1e9
    max_lid_vp = 0.0
    peak_lid = peak_bel = 0.0
    for i, x in enumerate(xs):
        zi = g.z_conv(x)
        for k, z in enumerate(zs):
            if z <= g.H + 1e-9:
                continue
            if z < zi - 1e-9:
                dv = _gauss(x, z, LID_LVZ)
                if abs(dv) > 1e-4:
                    vel[i][k] = max(1.60, vel[i][k] + dv)
                    n_lid += 1
                    peak_lid = min(peak_lid, dv)
                max_lid_vp = max(max_lid_vp, vel[i][k])
            else:
                dv = _gauss(x, z, BEL_LVZ)
                if abs(dv) > 1e-4:
                    vel[i][k] = max(5.40, vel[i][k] + dv)
                    n_bel += 1
                    peak_bel = min(peak_bel, dv)
                min_below_vs = min(min_below_vs, vel[i][k] / g.KAPPA)
    write_smesh(vp_path, xs, zs, vel)
    return {
        "n_lid": n_lid,
        "n_bel": n_bel,
        "peak_lid": peak_lid,
        "peak_bel": peak_bel,
        "max_lid_vp": max_lid_vp,
        "min_below_vs": min_below_vs,
    }


def seed_child(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    names = (
        "true_vp.smesh",
        "true_vs.smesh",
        "true_mixed.smesh",
        "start_vp.smesh",
        "seafloor.refl",
        "conv.refl",
        "damp_lid.dat",
        "vcorr.dat",
        "geom_ppp.dat",
        "geom_psp6.dat",
        "geom_pps7.dat",
        "geom_pss8.dat",
        "geom_07.dat",
        "geom_078.dat",
        "geom_all.dat",
        "rec_vp.smesh",
        "ppp_vp.smesh",
    )
    for name in names:
        sp = HERE / name
        if sp.is_file():
            shutil.copyfile(sp, dest / name)


def main() -> int:
    prepare(HERE)
    write_geom_codes(HERE / "geom_ppp.dat", HERE / "geom_all.dat", (0, 6, 7, 8))
    write_geom_codes(HERE / "geom_ppp.dat", HERE / "geom_078.dat", (0, 7, 8))
    write_geom_codes(HERE / "geom_ppp.dat", HERE / "geom_07.dat", (0, 7))
    write_geom_code(HERE / "geom_ppp.dat", HERE / "geom_psp6.dat", 6)
    write_geom_code(HERE / "geom_ppp.dat", HERE / "geom_pps7.dat", 7)
    write_geom_code(HERE / "geom_ppp.dat", HERE / "geom_pss8.dat", 8)
    info = apply_lvz(HERE / "true_vp.smesh")
    g.write_vs_from_vp(HERE / "true_vp.smesh", HERE / "true_vs.smesh", g.KAPPA)
    g.write_psp_speed(HERE / "true_vp.smesh", HERE / "true_vs.smesh", HERE / "true_mixed.smesh")
    print(
        f"lvz2d: lid n={info['n_lid']} peak {info['peak_lid']:+.2f} km/s  "
        f"below n={info['n_bel']} peak {info['peak_bel']:+.2f} km/s"
    )
    print(
        f"  max lid Vp={info['max_lid_vp']:.2f}  min below Vs={info['min_below_vs']:.2f}  "
        f"(PSP 下潜要求 Vs_below > lid Vp)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
