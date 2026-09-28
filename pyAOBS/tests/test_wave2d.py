# -*- coding: utf-8 -*-
"""wave2d：均匀介质到时 + 水柱 Vs 强制为 0。不改 tomo2d。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyAOBS.modeling.wave2d.elastic2d import propagate, suggest_dt
from pyAOBS.modeling.wave2d.grid import RegularModel, resample_dual
from pyAOBS.modeling.wave2d.io_smesh import parse_smesh
from pyAOBS.modeling.wave2d.ricker import ricker_delay


ROOT = Path(__file__).resolve().parent.parent
WORK = (
    ROOT
    / "modeling"
    / "tomo2d"
    / "example_water"
    / "PPP+PSP_inv2"
    / "psp_p_shoot"
    / "thin2km_rugged"
    / "lvz2d"
    / "inv_612"
)


def test_water_vs_forced_zero() -> None:
    m = resample_dual(
        WORK / "true_vp.smesh",
        WORK / "true_vs.smesh",
        WORK / "seafloor.refl",
        dx=0.5,
        dz=0.2,
        xmax=20.0,
        zmax=6.0,
    )
    assert m.water.any()
    assert float(m.vs[m.water].max()) == 0.0
    assert float(m.vp[m.water].max()) < 1.51
    xs, zs, vs = parse_smesh(WORK / "true_vs.smesh")
    k = min(i for i, z in enumerate(zs) if z < 1.9)
    assert vs[0][k] > 1.0


def test_homog_p_arrival() -> None:
    dx = 0.20
    x = np.arange(0.0, 12.0 + 0.5 * dx, dx)
    z = np.arange(0.0, 8.0 + 0.5 * dx, dx)
    nz, nx = z.size, x.size
    vp = np.full((nz, nx), 3.0)
    vs = np.full((nz, nx), 1.5)
    rho = np.full((nz, nx), 2000.0)
    water = np.zeros((nz, nx), dtype=bool)
    model = RegularModel(x=x, z=z, vp=vp, vs=vs, rho=rho, water=water, dx=dx, dz=dx)
    dt = min(suggest_dt(vp, dx, dx), 0.004)
    f0 = 2.0
    for absorb in ("cerjan", "pml"):
        g = propagate(
            model,
            src_x=4.0,
            src_z=3.0,
            rec_x=np.array([8.0]),
            rec_z=3.0,
            tmax=4.5,
            f0=f0,
            dt=dt,
            src_kind="expl",
            nb=10,
            absorb=absorb,
        )
        dist = 4.0
        t_pred = dist / 3.0 + ricker_delay(f0)
        tr = np.abs(g.data[0])
        thr = 0.25 * float(tr.max())
        i0 = int(np.argmax(g.t >= g.delay * 0.5))
        hit = np.where(tr[i0:] >= thr)[0]
        assert len(hit), absorb
        t_peak = float(g.t[i0 + int(hit[0])])
        assert abs(t_peak - t_pred) < 0.45, (absorb, t_peak, t_pred)


def test_pml_builds() -> None:
    from pyAOBS.modeling.wave2d.pml import build_cpml

    st = build_cpml(40, 60, nb=8, dt=0.001, dx=100.0, dz=100.0, vmax=3000.0, f0=5.0)
    assert st.k_x[0] > 1.0
    assert st.k_x[30] == 1.0
    assert st.a_x[0] != 0.0
    assert abs(st.a_x[30]) < 1e-15
    assert st.k_z[-1] > 1.0
    assert st.k_z[5] == 1.0
