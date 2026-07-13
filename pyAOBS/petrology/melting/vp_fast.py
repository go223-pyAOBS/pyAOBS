"""Fast Vp from pooled melt oxides (no BurnMan) for scan grids."""

from __future__ import annotations

from typing import Mapping

from petrology.vp_regression import predict_vp_km_s


def vp_from_melt_proxy(
    oxides_wt: Mapping[str, float],
    *,
    pbar_gpa: float,
    fbar: float,
    vp_bias_km_s: float = 0.0,
) -> float:
    """
    Lightweight bulk Vp for REEBOX scan grids.

    Uses eq.(1) at (P̄, F̄) plus a small MgO/SiO2 correction from pooled melt
    (typical LIP melt: lower MgO → slightly lower Vp). Full norm Vp is
    ``compute_norm_vp=True`` or ``--norm-vp`` / ``--refine-norm``.
    """
    vp_base = predict_vp_km_s(pbar_gpa, fbar)
    sio2 = float(oxides_wt.get("SiO2", 50.0))
    mgo = float(oxides_wt.get("MgO", 8.0))
    # Empirical tie to Fig.3 regression space (~±0.15 km/s over MORB–OIB spread).
    chem = -0.004 * (sio2 - 50.0) - 0.006 * (mgo - 8.0)
    return float(vp_base + chem + vp_bias_km_s)
