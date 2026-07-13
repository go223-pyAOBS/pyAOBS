"""
pyAOBS.petrology — Mantle melting, fractional crystallization, and bulk-crust
Vp workflows following Korenaga et al. (2002, JGR) and a modern forward-model track.

See README.md for:
  - Paper mapping (KKHS02 four steps)
  - Dual tracks: reproduction vs modern
"""

__all__: list[str] = []

# Planned: track = "reproduction" | "modern"
DEFAULT_TRACK = "reproduction"

from petrology.config import DEFAULT_CONFIG, PetrologyConfig
from petrology.active_upwelling import (
    ActiveUpwellingResult,
    p0_from_tp,
    solve_active_upwelling,
    solidus_tk83_c,
    sweep_hvp,
)
from petrology.fractionation import (
    delta_vp_km_s,
    theoretical_lower_crust_vp_km_s,
    theoretical_upper_crust_vp_km_s,
)
from petrology.invert import (
    bulk_vp_bounds,
    bulk_vp_bounds_from_fractionation,
    feasible_pf_region,
    invert_observation_rows,
    invert_single_observation,
)
from petrology.norm_velocity import norm_velocity_from_bulk_wt, norm_velocity_from_record
from petrology.vp_regression import load_eq1, predict_vp_km_s
from petrology.fc import (
    CrystallizationPathMode,
    CrystallizationState,
    load_kinzler1997_morb_primary,
    simulate_crystallization_path,
)

__all__ = [
    "DEFAULT_TRACK",
    "DEFAULT_CONFIG",
    "PetrologyConfig",
    "solidus_tk83_c",
    "p0_from_tp",
    "solve_active_upwelling",
    "sweep_hvp",
    "ActiveUpwellingResult",
    "delta_vp_km_s",
    "theoretical_lower_crust_vp_km_s",
    "theoretical_upper_crust_vp_km_s",
    "bulk_vp_bounds",
    "bulk_vp_bounds_from_fractionation",
    "feasible_pf_region",
    "invert_single_observation",
    "invert_observation_rows",
    "norm_velocity_from_bulk_wt",
    "norm_velocity_from_record",
    "load_eq1",
    "predict_vp_km_s",
    "CrystallizationPathMode",
    "CrystallizationState",
    "load_kinzler1997_morb_primary",
    "simulate_crystallization_path",
]
