"""
ΔVp = V_LC,cumulate − V_bulk from W&L (1990) + Langmuir (1992) FC.

KKHS02 Step 3: perfect fractional crystallization at P_fc → cumulate norm Vp
minus starting melt bulk norm Vp (both at the Fig.5 reference state by default).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping

import numpy as np

from petrology.config import DEFAULT_CONFIG, PetrologyConfig
from petrology.minerals import Backend
from petrology.norm_velocity import norm_velocity_from_bulk_wt

from .wl1990 import CrystallizationPathMode, normalize_melt_oxides, simulate_crystallization_path

# KKHS02 Fig.5 / eq.(1) reporting reference (600 MPa, 400°C).
FIG5_P_EVAL_MPA = 600.0
FIG5_T_EVAL_C = 400.0

# Production / R3 defaults: physical W&L+Langmuir (no Fig.5 empirical ΔVp lift).
# Set fig5_dvp_calibrate=True only when matching digitized Fig.5 F-loci.
R3_DELTA_VP_WL_KW: dict[str, str | bool] = {
    "kd_engine": "langmuir",
    "mineral_backend": "sb1994_fig2ol",
    "fig5_dvp_calibrate": False,
}

_OXIDE_KEYS = ("SiO2", "TiO2", "Al2O3", "Cr2O3", "FeO", "MgO", "CaO", "Na2O", "K2O")


def _melt_cache_key(melt: Mapping[str, float]) -> tuple[float, ...]:
    norm = normalize_melt_oxides(melt)
    return tuple(round(float(norm[k]), 3) for k in _OXIDE_KEYS)


@lru_cache(maxsize=512)
def _delta_vp_wl_fc_cached(
    melt_key: tuple[float, ...],
    f_solid: float,
    p_fc_mpa: float,
    bulk_vp_km_s: float | None,
    p_eval_mpa: float,
    t_eval_c: float,
    path: CrystallizationPathMode,
    cipw_backend: str,
    mineral_backend: str,
) -> float:
    melt = {k: melt_key[i] for i, k in enumerate(_OXIDE_KEYS)}
    return _delta_vp_wl_fc_uncached(
        melt_oxides_wt=melt,
        f_solid=f_solid,
        p_fc_mpa=p_fc_mpa,
        bulk_vp_km_s=bulk_vp_km_s,
        p_eval_mpa=p_eval_mpa,
        t_eval_c=t_eval_c,
        path=path,
        cipw_backend=cipw_backend,
        mineral_backend=mineral_backend,  # type: ignore[arg-type]
    )


def _delta_vp_wl_fc_uncached(
    *,
    melt_oxides_wt: Mapping[str, float],
    f_solid: float,
    p_fc_mpa: float = 400.0,
    bulk_vp_km_s: float | None = None,
    p_eval_mpa: float = FIG5_P_EVAL_MPA,
    t_eval_c: float = FIG5_T_EVAL_C,
    path: CrystallizationPathMode = "fc_100",
    cipw_backend: str = "auto",
    mineral_backend: Backend | str = "auto",
    config: PetrologyConfig | None = None,
    sub_steps_per_interval: int = 25,
    kd_engine: str = "heuristic",
    clamp_negative: bool = True,
    fig2_ab_calibrate: bool = False,
    fig2_ab_anchor_vp: bool = False,
    fig5_dvp_calibrate: bool = False,
) -> float:
    f = float(np.clip(f_solid, 0.0, 0.98))
    if f <= 1e-6:
        return 0.0

    cfg = config or DEFAULT_CONFIG
    melt = normalize_melt_oxides(melt_oxides_wt)
    p_eval_pa = float(p_eval_mpa) * 1e6
    t_eval_k = float(t_eval_c) + 273.15

    if bulk_vp_km_s is None:
        bulk_vp_km_s = float(
            norm_velocity_from_bulk_wt(
                melt,
                p_pa=p_eval_pa,
                t_k=t_eval_k,
                cipw_backend=cipw_backend,
                mineral_backend=mineral_backend,  # type: ignore[arg-type]
                config=cfg,
            )["vp_km_s"]
        )

    states = simulate_crystallization_path(
        primary_melt_oxides_wt=melt,
        f_grid=np.array([0.0, f], dtype=float),
        path=path,
        p_fc_mpa=float(p_fc_mpa),
        p_eval_mpa=float(p_eval_mpa),
        t_eval_c=float(t_eval_c),
        cipw_backend=cipw_backend,
        mineral_backend=mineral_backend,  # type: ignore[arg-type]
        config=cfg,
        sub_steps_per_interval=int(sub_steps_per_interval),
        kd_engine=kd_engine,  # type: ignore[arg-type]
        fig2_ab_calibrate=fig2_ab_calibrate,
        fig2_ab_anchor_vp=fig2_ab_anchor_vp,
    )
    vlc = float(states[-1].vp_cumulate_km_s)
    delta = vlc - float(bulk_vp_km_s)
    if fig5_dvp_calibrate:
        from .fig5_dvp_cal import apply_fig5_dvp_calibration

        delta = apply_fig5_dvp_calibration(delta, f, float(p_fc_mpa), enabled=True)
    if clamp_negative:
        return float(max(0.0, delta))
    return float(delta)


def delta_vp_wl_fc(
    *,
    melt_oxides_wt: Mapping[str, float],
    f_solid: float,
    p_fc_mpa: float = 400.0,
    bulk_vp_km_s: float | None = None,
    p_eval_mpa: float = FIG5_P_EVAL_MPA,
    t_eval_c: float = FIG5_T_EVAL_C,
    path: CrystallizationPathMode = "fc_100",
    cipw_backend: str = "auto",
    mineral_backend: Backend | str = "auto",
    config: PetrologyConfig | None = None,
    sub_steps_per_interval: int = 25,
    use_cache: bool = True,
    kd_engine: str = "heuristic",
    clamp_negative: bool = True,
    fig2_ab_calibrate: bool = False,
    fig2_ab_anchor_vp: bool = False,
    fig5_dvp_calibrate: bool = False,
) -> float:
    """W&L FC ΔVp = V_LC,cumulate − V_bulk (km/s).

    Set ``clamp_negative=False`` for R3 diagnostics (paper target ~+0.15 km/s @ F=0.7–0.8).
    Set ``fig5_dvp_calibrate=True`` to add Fig.5 F-locus bias correction
    (off by default; production prefers raw W&L+Langmuir ΔVp).
    """
    if (
        not use_cache
        or not clamp_negative
        or kd_engine != "heuristic"
        or fig2_ab_calibrate
        or fig2_ab_anchor_vp
        or fig5_dvp_calibrate
    ):
        return _delta_vp_wl_fc_uncached(
            melt_oxides_wt=melt_oxides_wt,
            f_solid=f_solid,
            p_fc_mpa=p_fc_mpa,
            bulk_vp_km_s=bulk_vp_km_s,
            p_eval_mpa=p_eval_mpa,
            t_eval_c=t_eval_c,
            path=path,
            cipw_backend=cipw_backend,
            mineral_backend=mineral_backend,
            config=config,
            sub_steps_per_interval=sub_steps_per_interval,
            kd_engine=kd_engine,
            clamp_negative=clamp_negative,
            fig2_ab_calibrate=fig2_ab_calibrate,
            fig2_ab_anchor_vp=fig2_ab_anchor_vp,
            fig5_dvp_calibrate=fig5_dvp_calibrate,
        )

    key = _melt_cache_key(melt_oxides_wt)
    vp_key = None if bulk_vp_km_s is None else round(float(bulk_vp_km_s), 4)
    mb = mineral_backend if isinstance(mineral_backend, str) else str(mineral_backend)
    return _delta_vp_wl_fc_cached(
        key,
        round(float(f_solid), 4),
        round(float(p_fc_mpa), 1),
        vp_key,
        round(float(p_eval_mpa), 1),
        round(float(t_eval_c), 1),
        path,
        str(cipw_backend),
        mb,
    )


def delta_vp_catalog_record(
    rec: dict,
    *,
    f_solid: float,
    p_fc_mpa: float = 400.0,
    bulk_vp_km_s: float | None = None,
    **kw,
) -> float:
    """ΔVp for one melt-catalog row (uses ``oxides_from_record`` keys)."""
    from petrology.data.load_catalog import oxides_from_record

    ox = {k: v for k, v in oxides_from_record(rec).items() if k in _OXIDE_KEYS}
    return delta_vp_wl_fc(
        melt_oxides_wt=ox,
        f_solid=f_solid,
        p_fc_mpa=p_fc_mpa,
        bulk_vp_km_s=bulk_vp_km_s,
        **kw,
    )
