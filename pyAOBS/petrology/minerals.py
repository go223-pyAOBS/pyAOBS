"""
Single-mineral elastic properties for norm-based Vp (KKHS02 reproduction).

Backends:
  - ``sb1994_fig2ol``: S&B Table I; Ol Fig.2 anchor when Ol mass fraction > 80% in HS mix
  - ``sb1994``: Sobolev & Babeyko (1994) Table I + linear P–T (``data/sb1994/``)
  - ``burnman``: SLB_2022 solutions @ reference (600 MPa, 400 °C)
  - ``empirical``: S&B-style linearized Vp (fallback / fast MVP)
  - ``auto``: try burnman, else empirical
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .vendored import burnman_available, ensure_vendored

Backend = Literal["auto", "empirical", "burnman", "fig2", "sb1994", "sb1994_fig2ol"]

REF_P_PA = 600e6
REF_T_K = 673.15

_EMP_VP = {
    "olivine": {"base": 8.20, "fo_slope": 0.35},
    "plagioclase": {"base": 6.85, "an_slope": 0.55},
    "clinopyroxene": {"base": 7.05, "mg_slope": 0.40},
    "quartz": {"base": 6.05},
    "ilmenite": {"base": 7.35},
}

_EMP_RHO = {
    "olivine": 3.32,
    "plagioclase": 2.72,
    "clinopyroxene": 3.15,
    "quartz": 2.65,
    "ilmenite": 4.72,
}

_EMP_KG = {
    "olivine": (130.0, 80.0),
    "plagioclase": (76.0, 40.0),
    "clinopyroxene": (110.0, 65.0),
    "quartz": (37.0, 44.0),
    "ilmenite": (170.0, 95.0),
}


def _make_burnman_phase(phase: str, fo: float, an: float, cpx_mg: float):
    """Build SLB_2022 solution; import submodule only (avoids full optional stack)."""
    ensure_vendored("burnman")
    from burnman.minerals import SLB_2022  # noqa: WPS433

    fo = float(np.clip(fo, 0.01, 0.99))
    an = float(np.clip(an, 0.0, 0.99))
    cpx_mg = float(np.clip(cpx_mg, 0.01, 0.99))

    if phase == "olivine":
        sol = SLB_2022.ol()
        sol.set_composition([fo, 1.0 - fo])
    elif phase == "plagioclase":
        sol = SLB_2022.plg()
        sol.set_composition([an, 1.0 - an])
    elif phase == "clinopyroxene":
        sol = SLB_2022.cpx()
        # diopside–hedenbergite endmembers (MORB-like); cats/jd = 0
        sol.set_composition([cpx_mg, 1.0 - cpx_mg, 0.0, 0.0, 0.0])
    elif phase == "quartz":
        sol = SLB_2022.qtz()
    elif phase == "ilmenite":
        sol = SLB_2022.il()
        # SLB_2022.il is a 3-endmember solution; use Fe-ilmenite-dominant proxy.
        sol.set_composition([0.0, 1.0, 0.0])
    else:
        raise KeyError(phase)
    return sol


def _phase_props_burnman(
    phase: str,
    fo: float,
    an: float,
    cpx_mg: float,
    p_pa: float,
    t_k: float,
    *,
    prefer_pip: bool = True,
) -> tuple[float, float, float, float]:
    if not burnman_available(prefer_pip=prefer_pip):
        raise ImportError("burnman not available (vendored or pip)")
    sol = _make_burnman_phase(phase, fo, an, cpx_mg)
    # BurnMan evaluate() expects list-like variables and array-like P/T.
    # For single state queries, set_state + aliased properties is more robust.
    sol.set_state(float(p_pa), float(t_k))
    vp = float(sol.v_p) / 1000.0
    rho = float(sol.rho) / 1000.0
    k = float(sol.isentropic_bulk_modulus_reuss) / 1e9
    g = float(sol.shear_modulus) / 1e9
    # Guard against non-physical states (e.g., quartz near unstable modifier regime).
    if not np.isfinite(vp) or not np.isfinite(rho) or not np.isfinite(k) or not np.isfinite(g):
        raise ValueError(f"non-finite BurnMan properties for {phase}")
    if vp <= 0 or rho <= 0 or k <= 0 or g <= 0:
        raise ValueError(f"non-physical BurnMan properties for {phase}")
    if vp > 20.0:  # km/s, broad hard bound for crustal silicates
        raise ValueError(f"implausible BurnMan vp for {phase}: {vp:.2f} km/s")
    return vp, rho, k, g


def phase_properties(
    phase: str,
    *,
    fo: float = 0.90,
    an: float = 0.50,
    cpx_mg: float = 0.85,
    p_pa: float = REF_P_PA,
    t_k: float = REF_T_K,
    backend: Backend = "auto",
    prefer_pip: bool = True,
) -> dict[str, float]:
    """Return vp_km_s, rho_g_cm3, k_gpa, g_gpa for one normative phase."""
    if backend == "sb1994_fig2ol":
        try:
            from .sb1994 import phase_properties_sb1994_fig2ol

            return phase_properties_sb1994_fig2ol(
                phase, fo=fo, an=an, cpx_mg=cpx_mg, p_pa=p_pa, t_k=t_k,
            )
        except KeyError:
            if phase == "ilmenite":
                backend = "empirical"
            else:
                raise

    if backend == "sb1994":
        try:
            from .sb1994 import phase_properties_sb1994

            return phase_properties_sb1994(
                phase, fo=fo, an=an, cpx_mg=cpx_mg, p_pa=p_pa, t_k=t_k,
            )
        except KeyError:
            if phase == "ilmenite":
                backend = "empirical"
            else:
                raise

    use_burnman = backend in ("auto", "burnman", "fig2")
    if use_burnman:
        try:
            vp, rho, k, g = _phase_props_burnman(
                phase, fo, an, cpx_mg, p_pa, t_k, prefer_pip=prefer_pip
            )
            props = {
                "vp_km_s": vp,
                "rho_g_cm3": rho,
                "k_gpa": k,
                "g_gpa": g,
                "backend": "burnman",
            }
            if backend == "fig2":
                from .fig2_elastic import apply_fig2_calibration

                props = apply_fig2_calibration(props, phase)
            return props
        except Exception:
            # BurnMan quartz can be unstable at low-T report states (e.g., 100 MPa, 100 C).
            # Keep BurnMan for other phases but fall back to empirical quartz only.
            if backend == "burnman" and phase == "quartz":
                backend = "auto"
            if backend == "fig2":
                emp = phase_properties(
                    phase,
                    fo=fo,
                    an=an,
                    cpx_mg=cpx_mg,
                    p_pa=p_pa,
                    t_k=t_k,
                    backend="empirical",
                    prefer_pip=prefer_pip,
                )
                from .fig2_elastic import apply_fig2_calibration

                return apply_fig2_calibration(emp, phase)
            if backend in ("burnman",):
                raise

    emp = _EMP_VP[phase]
    if phase == "olivine":
        vp = emp["base"] + emp["fo_slope"] * (fo - 0.5)
    elif phase == "plagioclase":
        vp = emp["base"] + emp["an_slope"] * (an - 0.5)
    elif phase == "clinopyroxene":
        vp = emp["base"] + emp["mg_slope"] * (cpx_mg - 0.5)
    else:
        vp = emp["base"]
    rho = _EMP_RHO[phase]
    k, g = _EMP_KG[phase]
    return {
        "vp_km_s": float(vp),
        "rho_g_cm3": float(rho),
        "k_gpa": float(k),
        "g_gpa": float(g),
        "backend": "empirical",
    }
