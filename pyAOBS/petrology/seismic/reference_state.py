"""P–T correction to KKHS02 reference state (600 MPa, 400 °C) — Fig.7 derivatives."""

from __future__ import annotations

# KKHS02 Fig.7 / §5: ∂Vp/∂P, ∂Vp/∂T @ reference 600 MPa, 400 °C
DV_DP_KM_S_PER_MPA = 0.2e-3
DV_DT_KM_S_PER_C = -0.4e-3

KKHS02_REFERENCE_P_MPA = 600.0
KKHS02_REFERENCE_T_C = 400.0

from petrology.constants import KM_PER_GPA

# §4 depth–pressure: 30 km/GPa → ~33.33 MPa/km lithostatic
MPA_PER_KM = 1000.0 / KM_PER_GPA

# Conductive geotherm for in-situ T (Fig.15a)
DEFAULT_GEOTHERM_C_PER_KM = 20.0
DEFAULT_SURFACE_T_C = 0.0


def lithostatic_pressure_mpa(depth_km: float) -> float:
    """Lithostatic pressure (MPa) from depth below surface (km)."""
    return max(float(depth_km), 0.0) * MPA_PER_KM


def conductive_geotherm_c(
    depth_km: float,
    *,
    gradient_c_per_km: float = DEFAULT_GEOTHERM_C_PER_KM,
    surface_t_c: float = DEFAULT_SURFACE_T_C,
) -> float:
    """Linear conductive geotherm T(z) = T_surface + (dT/dz)·z."""
    return float(surface_t_c) + float(gradient_c_per_km) * max(float(depth_km), 0.0)


def correct_vp_to_reference_km_s(
    vp_insitu_km_s: float,
    *,
    p_local_mpa: float,
    t_local_c: float,
    p_ref_mpa: float = KKHS02_REFERENCE_P_MPA,
    t_ref_c: float = KKHS02_REFERENCE_T_C,
    dv_dp: float = DV_DP_KM_S_PER_MPA,
    dv_dt: float = DV_DT_KM_S_PER_C,
) -> float:
    """
    Correct in-situ Vp to reference state (KKHS02 eq. from Fig.7).

    V_ref = V_obs + (∂V/∂P)(P_ref − P_loc) + (∂V/∂T)(T_ref − T_loc)
    """
    vp = float(vp_insitu_km_s)
    dp = float(p_ref_mpa) - float(p_local_mpa)
    dt = float(t_ref_c) - float(t_local_c)
    return vp + float(dv_dp) * dp + float(dv_dt) * dt


def correct_vp_depth_to_reference_km_s(
    vp_insitu_km_s: float,
    depth_km: float,
    *,
    gradient_c_per_km: float = DEFAULT_GEOTHERM_C_PER_KM,
    surface_t_c: float = DEFAULT_SURFACE_T_C,
) -> float:
    """Convenience: lithostatic P(z) + conductive T(z) → V @ 600 MPa, 400 °C."""
    p = lithostatic_pressure_mpa(depth_km)
    t = conductive_geotherm_c(depth_km, gradient_c_per_km=gradient_c_per_km, surface_t_c=surface_t_c)
    return correct_vp_to_reference_km_s(vp_insitu_km_s, p_local_mpa=p, t_local_c=t)
