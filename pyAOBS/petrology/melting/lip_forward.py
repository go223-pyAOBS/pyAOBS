"""
LIP forward closure: melting column → bulk Vp → V_LC bounding check.

Connects Modern melting core to ``invert.py`` without W&L FC or REEBOX GUI.
"""

from __future__ import annotations

from dataclasses import dataclass

from petrology.fractionation import delta_vp_km_s
from petrology.invert import bulk_vp_bounds, feasible_pf_region

from .column import MeltingColumnResult, forward_melting_column


@dataclass(frozen=True)
class LipForwardResult:
    """Melting forward model compared with observed lower-crust Vp."""

    column: MeltingColumnResult
    v_lc_obs_km_s: float
    v_bulk_lower_km_s: float
    v_bulk_upper_km_s: float
    delta_vp_fc_km_s: float
    vp_bulk_used_km_s: float
    v_lc_theory_km_s: float
    bulk_in_bounds: bool
    vlc_match_km_s: float
    n_feasible_pf: int


def forward_lip_column(
    *,
    v_lc_obs_km_s: float,
    tp_c: float,
    b_km: float = 0.0,
    chi: float = 1.0,
    pyroxenite_frac: float = 0.0,
    f_lower: float = 0.5,
    f_solid: float = 0.75,
    p_fc_mpa: float = 400.0,
    use_norm_vp: bool = True,
    vp_bias_km_s: float = -0.10,
    vp_tolerance_km_s: float = 0.05,
    **column_kw,
) -> LipForwardResult:
    """
    Run melting column and test whether predicted bulk Vp lies in V_LC bounds.

    V_bulk prediction: norm-based Vp from pooled melt if ``use_norm_vp`` (default),
    else eq.(1) at (P̄, F̄).
    """
    col = forward_melting_column(
        tp_c=tp_c,
        b_km=b_km,
        chi=chi,
        pyroxenite_frac=pyroxenite_frac,
        compute_norm_vp=use_norm_vp,
        vp_bias_km_s=vp_bias_km_s,
        **column_kw,
    )
    vp_bulk = col.vp_bulk_norm_km_s if (use_norm_vp and col.vp_bulk_norm_km_s is not None) else col.vp_bulk_eq1_km_s

    d_vp = delta_vp_km_s(
        f_solid,
        bulk_vp_km_s=vp_bulk,
        p_fc_mpa=p_fc_mpa,
        engine="auto",
        melt_oxides_wt=col.pooled_melt_wt,
    )
    bounds = bulk_vp_bounds(v_lc_obs_km_s, f_lower=f_lower, delta_vp_fc_km_s=d_vp)
    in_bounds = bounds.v_bulk_lower_km_s <= vp_bulk <= bounds.v_bulk_upper_km_s
    v_lc_theory = float(vp_bulk + d_vp)

    # Optional: eq.(1) feasible region at observed bulk (diagnostic).
    region = feasible_pf_region(bounds, vp_tolerance_km_s=vp_tolerance_km_s)

    return LipForwardResult(
        column=col,
        v_lc_obs_km_s=float(v_lc_obs_km_s),
        v_bulk_lower_km_s=bounds.v_bulk_lower_km_s,
        v_bulk_upper_km_s=bounds.v_bulk_upper_km_s,
        delta_vp_fc_km_s=d_vp,
        vp_bulk_used_km_s=float(vp_bulk),
        v_lc_theory_km_s=v_lc_theory,
        bulk_in_bounds=bool(in_bounds),
        vlc_match_km_s=float(v_lc_theory - v_lc_obs_km_s),
        n_feasible_pf=int(region["n_feasible"]),
    )
