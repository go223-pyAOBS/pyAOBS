"""MVP and W&L FC ΔVp for KKHS02 Fig.5 / Step-2 bounding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

DeltaVpEngine = Literal["param", "wl1990", "auto"]


@dataclass(frozen=True)
class FractionationParams:
    """Compact parameterization tuned to KKHS02 Fig.5 first-order behavior."""

    delta_vp_max: float = 0.45  # km/s, low-F upper envelope
    delta_vp_exp: float = 0.8  # controls decay with solid fraction
    pressure_sensitivity_per_gpa: float = 0.015  # weak pressure dependence
    composition_sensitivity: float = 0.08  # weak dependence on bulk Vp anomaly
    upper_crust_drop: float = 0.75  # km/s max residual melt Vp depression


DEFAULT_FRACTIONATION = FractionationParams()


def delta_vp_param_km_s(
    f_solid: float,
    *,
    bulk_vp_km_s: float,
    p_fc_mpa: float,
    params: FractionationParams = DEFAULT_FRACTIONATION,
) -> float:
    """Legacy MVP parameterization (fast surrogate)."""
    f = float(np.clip(f_solid, 0.02, 0.98))
    p_gpa = float(p_fc_mpa) / 1000.0
    base = params.delta_vp_max * ((1.0 - f) ** params.delta_vp_exp)
    p_fac = 1.0 + params.pressure_sensitivity_per_gpa * (p_gpa - 0.4)
    c_fac = 1.0 - params.composition_sensitivity * (bulk_vp_km_s - 7.2)
    delta = base * p_fac * c_fac
    return float(np.clip(delta, 0.0, 0.6))


def delta_vp_km_s(
    f_solid: float,
    *,
    bulk_vp_km_s: float,
    p_fc_mpa: float,
    params: FractionationParams = DEFAULT_FRACTIONATION,
    engine: DeltaVpEngine = "auto",
    melt_oxides_wt: Mapping[str, float] | None = None,
    **wl_kw,
) -> float:
    """
    ΔVp = V_LC,theory − V_bulk (km/s).

    ``engine``:
      - ``param``: legacy Fig.5 MVP formula
      - ``wl1990``: W&L FC + Langmuir (1992) P rules via ``fc.delta_vp``
      - ``auto``: ``wl1990`` when ``melt_oxides_wt`` is given, else ``param``
    """
    eng = engine
    if eng == "auto":
        eng = "wl1990" if melt_oxides_wt is not None else "param"

    if eng == "wl1990":
        if melt_oxides_wt is None:
            raise ValueError("melt_oxides_wt required for delta_vp engine 'wl1990'")
        from petrology.fc.delta_vp import delta_vp_wl_fc

        return delta_vp_wl_fc(
            melt_oxides_wt=melt_oxides_wt,
            f_solid=f_solid,
            p_fc_mpa=p_fc_mpa,
            bulk_vp_km_s=bulk_vp_km_s,
            **wl_kw,
        )

    return delta_vp_param_km_s(
        f_solid,
        bulk_vp_km_s=bulk_vp_km_s,
        p_fc_mpa=p_fc_mpa,
        params=params,
    )


def theoretical_lower_crust_vp_km_s(
    bulk_vp_km_s: float,
    f_solid: float,
    *,
    p_fc_mpa: float,
    params: FractionationParams = DEFAULT_FRACTIONATION,
    engine: DeltaVpEngine = "auto",
    melt_oxides_wt: Mapping[str, float] | None = None,
    **wl_kw,
) -> float:
    return float(
        bulk_vp_km_s
        + delta_vp_km_s(
            f_solid,
            bulk_vp_km_s=bulk_vp_km_s,
            p_fc_mpa=p_fc_mpa,
            params=params,
            engine=engine,
            melt_oxides_wt=melt_oxides_wt,
            **wl_kw,
        )
    )


def theoretical_upper_crust_vp_km_s(
    bulk_vp_km_s: float,
    f_solid: float,
    *,
    params: FractionationParams = DEFAULT_FRACTIONATION,
) -> float:
    """Residual melt norm Vp proxy (Fig.5c–d MVP)."""
    f = float(np.clip(f_solid, 0.02, 0.98))
    drop = params.upper_crust_drop * (f**0.9)
    return float(np.clip(bulk_vp_km_s - drop, 5.8, 7.4))
