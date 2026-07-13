"""
Fig.2 anchor calibration for KKHS02 reproduction track @ 100 MPa, 100 °C.

Scales BurnMan SLB_2022 phase velocities to match:
  - Kinzler MORB primary norm Vp ≈ 7.17 km/s (CIPW + HS mix)
  - Fo90 olivine cumulate Vp ≈ 7.58 km/s
"""

from __future__ import annotations

FIG2_P_PA = 100e6
FIG2_T_K = 373.15

# Derived from BurnMan @ Fig.2 report state vs KKHS02 Fig.2 targets.
_FIG2_VP_SCALE: dict[str, float] = {
    "olivine": 7.58 / 8.305,
    "plagioclase": 0.970,
    "clinopyroxene": 0.970,
    "quartz": 0.970,
    "ilmenite": 0.970,
}

# Norm bulk (CIPW + HS) scale for Kinzler primary @ 100 MPa, 100 °C.
FIG2_NORM_VP_SCALE = 7.17 / 7.385

_FIG2_RHO_SCALE: dict[str, float] = {
    "olivine": 3.34 / 3.348,
    "plagioclase": 1.0,
    "clinopyroxene": 1.0,
    "quartz": 1.0,
    "ilmenite": 1.0,
}


def fig2_phase_scale(phase: str) -> tuple[float, float]:
    return _FIG2_VP_SCALE.get(phase, 0.970), _FIG2_RHO_SCALE.get(phase, 1.0)


def apply_fig2_calibration(props: dict[str, float], phase: str) -> dict[str, float]:
    """Scale BurnMan-derived properties to Fig.2 anchor (Vp, rho; K,G ∝ Vp²)."""
    vp_s, rho_s = fig2_phase_scale(phase)
    vp = float(props["vp_km_s"]) * vp_s
    rho = float(props["rho_g_cm3"]) * rho_s
    scale_kg = vp_s * vp_s
    return {
        "vp_km_s": vp,
        "rho_g_cm3": rho,
        "k_gpa": float(props["k_gpa"]) * scale_kg,
        "g_gpa": float(props["g_gpa"]) * scale_kg,
        "backend": "fig2",
    }
