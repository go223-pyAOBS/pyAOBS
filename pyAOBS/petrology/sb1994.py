"""
Sobolev & Babeyko (1994) single-mineral elastic properties (Table I).

Data source: ``petrology/data/sb1994/SB1994.xlsx`` (user compilation) →
``endmembers.json``.  Reference state: 25 °C, 0.1 MPa (S&B Table I).

P–T extrapolation follows the linear forward form used in S&B §3 / §5 for
in-situ moduli and density:

  X(P,T) = X₀ + (∂X/∂P)·ΔP + (∂X/∂T)·ΔT

with ΔP in GPa, ΔT in K relative to room conditions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np

from .mixing import vp_from_kgrho

P0_PA = 1.0e5
T0_K = 298.15

_DATA_DIR = Path(__file__).resolve().parent / "data" / "sb1994"
_ENDMEMBERS_JSON = _DATA_DIR / "endmembers.json"

# Aliases for norm phases → endmember keys in the compilation.
_PHASE_BINARY: dict[str, tuple[str, str]] = {
    "olivine": ("Fo", "Fa"),
    "plagioclase": ("Ab", "An"),
    "clinopyroxene": ("Di", "Cfs"),
}
_PHASE_FIXED: dict[str, str] = {
    "quartz": "α-Qz",
}


@dataclass(frozen=True)
class SbEndmember:
    key: str
    rho: float
    alpha: float  # α·10⁵ K⁻¹ (Table I)
    beta: float  # β·10³ GPa⁻¹
    k: float  # GPa @ room
    k_p: float  # ∂K/∂P (GPa/GPa)
    k_t_neg: float  # −∂K/∂T (MPa/K)
    g: float
    g_p: float
    g_t_neg: float  # −∂G/∂T (MPa/K)


def _fix_k(key: str, k: float) -> float:
    """Correct known OCR/typo values (Fa, Hc, Odi) in some Table I reproductions."""
    if k > 500.0 and key in {"Fa", "Hc", "Odi"}:
        return k / 10.0
    return k


@lru_cache(maxsize=1)
def load_endmembers() -> dict[str, SbEndmember]:
    """Load compiled endmembers; falls back to bundled JSON."""
    path = _ENDMEMBERS_JSON
    if not path.is_file():
        raise FileNotFoundError(f"S&B 1994 endmember table not found: {path}")
    raw: Mapping[str, Mapping[str, float | str]] = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, SbEndmember] = {}
    for key, rec in raw.items():
        k = float(rec["k"])
        out[key] = SbEndmember(
            key=key,
            rho=float(rec["rho"]),
            alpha=float(rec["alpha"]),
            beta=float(rec["beta"]),
            k=_fix_k(key, k),
            k_p=float(rec["k_p"]),
            k_t_neg=float(rec["k_t_neg"]),
            g=float(rec["g"]),
            g_p=float(rec["g_p"]),
            g_t_neg=float(rec["g_t_neg"]),
        )
    return out


def _delta_p_gpa(p_pa: float) -> float:
    return float(p_pa) / 1.0e9 - P0_PA / 1.0e9


def _delta_t_k(t_k: float) -> float:
    return float(t_k) - T0_K


def endmember_at_pt(key: str, *, p_pa: float, t_k: float) -> dict[str, float]:
    """Return rho, k_gpa, g_gpa for one Table-I endmember at (P,T)."""
    em = load_endmembers()[key]
    dP = _delta_p_gpa(p_pa)
    dT = _delta_t_k(t_k)

    rho = em.rho + em.rho * em.beta * 1.0e-3 * dP - em.rho * em.alpha * 1.0e-5 * dT
    k = em.k + em.k_p * dP - em.k_t_neg * 1.0e-3 * dT
    g = em.g + em.g_p * dP - em.g_t_neg * 1.0e-3 * dT

    if not (np.isfinite(rho) and np.isfinite(k) and np.isfinite(g)):
        raise ValueError(f"non-finite S&B properties for {key}")
    if rho <= 0 or k <= 0 or g <= 0:
        raise ValueError(f"non-physical S&B properties for {key}: rho={rho}, K={k}, G={g}")

    vp = vp_from_kgrho(k, g, rho)
    return {
        "vp_km_s": vp,
        "rho_g_cm3": float(rho),
        "k_gpa": float(k),
        "g_gpa": float(g),
        "backend": "sb1994",
    }


def _mix_binary(
    key_a: str,
    key_b: str,
    frac_a: float,
    *,
    p_pa: float,
    t_k: float,
) -> dict[str, float]:
    x = float(np.clip(frac_a, 0.0, 1.0))
    pa = endmember_at_pt(key_a, p_pa=p_pa, t_k=t_k)
    pb = endmember_at_pt(key_b, p_pa=p_pa, t_k=t_k)
    w_b = 1.0 - x
    rho = x * pa["rho_g_cm3"] + w_b * pb["rho_g_cm3"]
    k = x * pa["k_gpa"] + w_b * pb["k_gpa"]
    g = x * pa["g_gpa"] + w_b * pb["g_gpa"]
    vp = vp_from_kgrho(k, g, rho)
    return {
        "vp_km_s": vp,
        "rho_g_cm3": rho,
        "k_gpa": k,
        "g_gpa": g,
        "backend": "sb1994",
    }


def phase_properties_sb1994(
    phase: str,
    *,
    fo: float = 0.90,
    an: float = 0.50,
    cpx_mg: float = 0.85,
    p_pa: float = P0_PA,
    t_k: float = T0_K,
) -> dict[str, float]:
    """Norm-phase elastic properties via S&B (1994) Table I + linear P–T."""
    if phase in _PHASE_BINARY:
        key_a, key_b = _PHASE_BINARY[phase]
        if phase == "olivine":
            frac_a = fo
        elif phase == "plagioclase":
            frac_a = an
        else:
            frac_a = cpx_mg
        return _mix_binary(key_a, key_b, frac_a, p_pa=p_pa, t_k=t_k)

    if phase in _PHASE_FIXED:
        return endmember_at_pt(_PHASE_FIXED[phase], p_pa=p_pa, t_k=t_k)

    raise KeyError(f"S&B 1994 backend has no entry for phase {phase!r}")


FIG2_OL_VP_TARGET_KM_S = 7.58
FIG2_OL_RHO_TARGET_G_CM3 = 3.34
OL_FIG2OL_MASS_THRESHOLD = 0.80
# Ol share (mass) below which full mixed-assemblage HS Vp scale applies (Fig.2a / Fig.5).
OL_FIG2OL_MIXED_HS_OL_FULL = 0.42
# Ol share above which mixed HS scale is zero (early Ol-rich cumulates use Ol anchor only).
OL_FIG2OL_MIXED_HS_OL_ZERO = 0.68


def sb1994_fig2ol_mixed_hs_weight(ol_mass_fraction: float) -> float:
    """Blend 0→1 for mixed HS Vp scale vs olivine mass fraction in the assemblage."""
    ol = float(ol_mass_fraction)
    if ol >= OL_FIG2OL_MIXED_HS_OL_ZERO:
        return 0.0
    if ol <= OL_FIG2OL_MIXED_HS_OL_FULL:
        return 1.0
    return (OL_FIG2OL_MIXED_HS_OL_ZERO - ol) / (
        OL_FIG2OL_MIXED_HS_OL_ZERO - OL_FIG2OL_MIXED_HS_OL_FULL
    )


@lru_cache(maxsize=1)
def sb1994_fig2ol_mixed_hs_vp_scale() -> float:
    """
    Scale Hashin–Shtrikman Vp for Ol-poor sb1994_fig2ol cumulates.

    KKHS02 Fig.2(a) cumulative solid Vp exceeds S&B Table I + HS at the
    Fig.2 report state (100 MPa, 100 °C) by ~5% for fc_100, F ≈ 0.5–0.8.
    Mean anchor/HS ratio closes ΔVp ≈ +0.15 km/s @ F = 0.7–0.8 (Fig.5).
    """
    import numpy as np

    from petrology.fc.assemblage import assemblage_vp_rho
    from petrology.fc.fig2_ab import (
        cumulative_phases_pct,
        incremental_compositions,
        paper_cum_vp_km_s,
    )
    from petrology.fig2_elastic import FIG2_P_PA, FIG2_T_K

    ratios: list[float] = []
    for f in np.linspace(0.5, 0.8, 7):
        ol_p, cpx_p, pl_p = cumulative_phases_pct("fc_100", float(f))
        fo, an, di = incremental_compositions("fc_100", float(f))
        vp_hs, _ = assemblage_vp_rho(
            ol_frac=ol_p / 100.0,
            pl_frac=pl_p / 100.0,
            cpx_frac=cpx_p / 100.0,
            fo_pct=fo,
            an_pct=an,
            di_pct=di,
            p_pa=FIG2_P_PA,
            t_k=FIG2_T_K,
            mineral_backend="sb1994_fig2ol",
            apply_mixed_hs_scale=False,
        )
        if vp_hs > 1e-6:
            ratios.append(paper_cum_vp_km_s("fc_100", float(f)) / vp_hs)
    if not ratios:
        return 1.0537
    return float(np.mean(ratios))


@lru_cache(maxsize=1)
def sb1994_ol_fig2_anchor_scales() -> tuple[float, float]:
    """Ol-only Vp/rho scales @ KKHS02 Fig.2 report state (Fo90 → 7.58 km/s)."""
    from .fig2_elastic import FIG2_P_PA, FIG2_T_K

    raw = _mix_binary("Fo", "Fa", 0.90, p_pa=FIG2_P_PA, t_k=FIG2_T_K)
    vp_scale = FIG2_OL_VP_TARGET_KM_S / raw["vp_km_s"]
    rho_scale = FIG2_OL_RHO_TARGET_G_CM3 / raw["rho_g_cm3"]
    return float(vp_scale), float(rho_scale)


def apply_sb1994_fig2ol_calibration(
    props: dict[str, float],
    phase: str,
) -> dict[str, float]:
    """Scale S&B olivine to Fig.2 Fo90 anchor; Pl/Cpx/Qz unchanged."""
    if phase != "olivine":
        return {**props, "backend": "sb1994_fig2ol"}
    vp_s, rho_s = sb1994_ol_fig2_anchor_scales()
    vp = float(props["vp_km_s"]) * vp_s
    rho = float(props["rho_g_cm3"]) * rho_s
    scale_kg = vp_s * vp_s
    return {
        "vp_km_s": vp,
        "rho_g_cm3": rho,
        "k_gpa": float(props["k_gpa"]) * scale_kg,
        "g_gpa": float(props["g_gpa"]) * scale_kg,
        "backend": "sb1994_fig2ol",
    }


def calibrate_ol_if_dominant(
    props: dict[str, float],
    phase: str,
    ol_mass_fraction: float,
    *,
    threshold: float = OL_FIG2OL_MASS_THRESHOLD,
) -> dict[str, float]:
    """Apply Fig.2 Fo90 Ol anchor only when olivine dominates the assemblage (mass %)."""
    if phase != "olivine" or float(ol_mass_fraction) <= threshold:
        return props
    return apply_sb1994_fig2ol_calibration(props, phase)


def phase_properties_sb1994_fig2ol(
    phase: str,
    *,
    fo: float = 0.90,
    an: float = 0.50,
    cpx_mg: float = 0.85,
    p_pa: float = P0_PA,
    t_k: float = T0_K,
) -> dict[str, float]:
    """Raw S&B Table I; Ol Fig.2 scaling is applied in ``assemblage_vp_rho`` when Ol > 80%."""
    props = phase_properties_sb1994(
        phase, fo=fo, an=an, cpx_mg=cpx_mg, p_pa=p_pa, t_k=t_k,
    )
    return {**props, "backend": "sb1994_fig2ol"}
