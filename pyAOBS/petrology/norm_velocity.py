"""Norm-based bulk Vp from melt major-element oxides (KKHS02 §2.1 path A)."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .cipw import cipw_norm_mass_fractions
from .config import DEFAULT_CONFIG, PetrologyConfig
from .minerals import REF_P_PA, REF_T_K, Backend, phase_properties
from .mixing import hashin_shtrikman_vp

_PHASES = ("olivine", "plagioclase", "clinopyroxene", "quartz", "ilmenite")


def norm_velocity_from_bulk_wt(
    oxides_wt: Mapping[str, float],
    *,
    p_pa: float = REF_P_PA,
    t_k: float = REF_T_K,
    cipw_backend: str | None = None,
    mineral_backend: Backend | None = None,
    config: PetrologyConfig | None = None,
) -> dict[str, float]:
    """
    CIPW norm → mineral K,G,ρ @ (P,T) → HS-average Vp (km/s).

    Default backends follow ``PetrologyConfig`` (reproduction: auto → pyrolite + burnman).
    """
    cfg = config or DEFAULT_CONFIG
    cipw_backend = cipw_backend or cfg.cipw_backend
    mineral_backend = mineral_backend or cfg.mineral_backend

    norm = cipw_norm_mass_fractions(
        oxides_wt,
        backend=cipw_backend,
        prefer_pip=cfg.prefer_pip_vendored,
    )
    fo = norm.get("Fo", 0.90)
    an = norm.get("An", 0.50)
    cpx_mg = norm.get("CpxMg", fo)
    ol_mf = float(norm.get("olivine", 0.0))
    use_ol_threshold = mineral_backend == "sb1994_fig2ol"

    mass_frac = []
    rhos = []
    ks = []
    gs = []
    phase_names = []
    phase_backends: list[str] = []

    for ph in _PHASES:
        mf = norm.get(ph, 0.0)
        if mf <= 1e-8:
            continue
        props = phase_properties(
            ph,
            fo=fo,
            an=an,
            cpx_mg=cpx_mg,
            p_pa=p_pa,
            t_k=t_k,
            backend=mineral_backend,
            prefer_pip=cfg.prefer_pip_vendored,
        )
        if use_ol_threshold:
            from .sb1994 import calibrate_ol_if_dominant

            props = calibrate_ol_if_dominant(props, ph, ol_mf)
        rho = props["rho_g_cm3"]
        mass_frac.append(mf)
        rhos.append(rho)
        ks.append(props["k_gpa"])
        gs.append(props["g_gpa"])
        phase_names.append(ph)
        phase_backends.append(props.get("backend", mineral_backend))

    mf_arr = np.array(mass_frac)
    rho_arr = np.array(rhos)
    vol = mf_arr / rho_arr
    vol /= vol.sum()

    mix = hashin_shtrikman_vp(vol, np.array(ks), np.array(gs), rho_arr)
    mix["Fo"] = fo
    mix["An"] = an
    mix["CpxMg"] = cpx_mg
    mix["cipw_backend"] = cipw_backend
    if all(b == phase_backends[0] for b in phase_backends):
        mix["mineral_backend"] = phase_backends[0]
    else:
        mix["mineral_backend"] = "mixed"
    mix["norm_mass_fractions"] = {ph: float(norm.get(ph, 0.0)) for ph in _PHASES}
    mix["phase_volume_fractions"] = {ph: float(v) for ph, v in zip(phase_names, vol)}
    if mineral_backend == "fig2":
        from .fig2_elastic import FIG2_NORM_VP_SCALE

        mix["vp_km_s"] = float(mix["vp_km_s"]) * FIG2_NORM_VP_SCALE
    return mix


def norm_velocity_from_record(rec: dict, **kwargs) -> dict[str, float]:
    from .data.load_catalog import oxides_from_record

    return norm_velocity_from_bulk_wt(oxides_from_record(rec), **kwargs)
