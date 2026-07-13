"""Hashin–Shtrikman Vp and density for crystallizing mineral assemblages (KKHS02 §2.1)."""

from __future__ import annotations

from typing import Literal, Mapping

import numpy as np

from petrology.minerals import Backend, phase_properties
from petrology.mixing import hashin_shtrikman_vp

FractionBasis = Literal["mass", "volume"]


def assemblage_vp_rho(
    *,
    ol_frac: float,
    pl_frac: float,
    cpx_frac: float,
    fo_pct: float,
    an_pct: float,
    di_pct: float,
    p_pa: float,
    t_k: float,
    mineral_backend: Backend = "auto",
    fraction_basis: FractionBasis = "mass",
    apply_mixed_hs_scale: bool = True,
) -> tuple[float, float]:
    """
    Return (Vp km/s, rho g/cm3) for Ol+Pl+Cpx assemblage at (P,T).

    ``fraction_basis='mass'`` (KKHS02): mode fractions are **mass** fractions;
    convert to volume with mineral densities before Hashin–Shtrikman mixing.
    ``'volume'``: legacy behaviour (fractions treated as volume fractions).
    """
    fo = float(np.clip(fo_pct / 100.0, 0.05, 0.99))
    an = float(np.clip(an_pct / 100.0, 0.02, 0.99))
    cpx_mg = float(np.clip(di_pct / 100.0, 0.05, 0.99))

    mass_total = float(ol_frac) + float(pl_frac) + float(cpx_frac)
    ol_mass_share = float(ol_frac) / mass_total if mass_total > 0.0 else 0.0
    use_ol_threshold = mineral_backend == "sb1994_fig2ol"

    entries: list[tuple[float, float, float, float]] = []
    for name, mass_frac, comp_kw in (
        ("olivine", ol_frac, {"fo": fo, "an": an, "cpx_mg": cpx_mg}),
        ("plagioclase", pl_frac, {"fo": fo, "an": an, "cpx_mg": cpx_mg}),
        ("clinopyroxene", cpx_frac, {"fo": fo, "an": an, "cpx_mg": cpx_mg}),
    ):
        if mass_frac <= 0.0:
            continue
        props = phase_properties(
            name,
            fo=comp_kw["fo"],
            an=comp_kw["an"],
            cpx_mg=comp_kw["cpx_mg"],
            p_pa=p_pa,
            t_k=t_k,
            backend=mineral_backend,
        )
        if use_ol_threshold:
            from petrology.sb1994 import calibrate_ol_if_dominant

            props = calibrate_ol_if_dominant(props, name, ol_mass_share)
        entries.append((mass_frac, props["k_gpa"], props["g_gpa"], props["rho_g_cm3"]))

    if not entries:
        raise ValueError("Empty mineral assemblage")

    mf = np.array([e[0] for e in entries], dtype=float)
    mf /= mf.sum()
    k = np.array([e[1] for e in entries], dtype=float)
    g = np.array([e[2] for e in entries], dtype=float)
    rho = np.array([e[3] for e in entries], dtype=float)

    if fraction_basis == "mass":
        vol = mf / rho
        vf = vol / vol.sum()
    else:
        vf = mf

    mix = hashin_shtrikman_vp(vf, k, g, rho)
    vp_km_s = float(mix["vp_km_s"])
    if apply_mixed_hs_scale and use_ol_threshold and ol_mass_share <= 0.80:
        from petrology.sb1994 import (
            sb1994_fig2ol_mixed_hs_vp_scale,
            sb1994_fig2ol_mixed_hs_weight,
        )

        w = sb1994_fig2ol_mixed_hs_weight(ol_mass_share)
        if w > 0.0:
            base = sb1994_fig2ol_mixed_hs_vp_scale()
            vp_km_s *= 1.0 + (base - 1.0) * w
    return vp_km_s, float(mix["rho_g_cm3"])


def liquid_density_bw1970(oxides_wt: Mapping[str, float], *, t_c: float = 100.0) -> float:
    """Bottinga & Weill (1970) style melt density proxy (g/cm3)."""
    mg = float(oxides_wt.get("MgO", 0.0))
    fe = float(oxides_wt.get("FeO", 0.0))
    ca = float(oxides_wt.get("CaO", 0.0))
    alk = float(oxides_wt.get("Na2O", 0.0) + oxides_wt.get("K2O", 0.0))
    si = float(oxides_wt.get("SiO2", 0.0))
    rho = 2.52 + 0.0050 * mg + 0.0040 * fe + 0.0012 * ca - 0.0015 * alk - 0.0006 * (si - 50.0)
    rho -= 0.00025 * (t_c - 100.0)
    return float(np.clip(rho, 2.45, 3.05))
