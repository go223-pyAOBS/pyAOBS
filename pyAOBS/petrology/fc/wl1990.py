"""
Weaver & Langmuir (1990) fractional crystallization (KKHS02 reproduction track).

Architecture
------------
**Paper logic** (``wl_partition.py`` + ``wl_kd.py``):
  - Langmuir (1992) T+P Kd from ``BASALT+langmuir.FOR`` KDCALC (KDMODE=4)
  - STOICH pressure-dependent Cpx threshold (P>4 kbar → 1.1)
  - Polybaric P schedule (MODES(6), DRIVER PHI loop)
  - Perfect FC mass balance on oxide wt%

**Modern backends** (unchanged from KKHS02 chain):
  - Pyrolite CIPW → norm-based melt Vp
  - BurnMan SLB_2022 mineral K,G,ρ → Hashin–Shtrikman mix
  - Optional ``fig2`` scalar calibration for Fig.2 anchor reproduction

**Numerical guards** (not in W&L, stability only):
  - Oxide feasibility weights in ``_constrained_modes``
  - Sub-stepping on ``f_grid`` intervals
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import numpy as np

from petrology.config import DEFAULT_CONFIG, PetrologyConfig
from petrology.minerals import Backend
from petrology.norm_velocity import norm_velocity_from_bulk_wt

from .assemblage import assemblage_vp_rho, liquid_density_bw1970
from .fig2_norm import apply_fig2_incremental_vp_calibration, apply_fig2_residual_norm_calibration
from .fig2_ab import (
    cumulative_compositions_weighted,
    cumulative_phases_pct,
    incremental_compositions,
    incremental_modes_from_cumulative,
    paper_cum_rho_g_cm3,
    paper_cum_vp_km_s,
    paper_inc_rho_g_cm3,
    paper_inc_vp_km_s,
    paper_residual_rho_g_cm3,
    paper_residual_vp_km_s,
)
from .wl_components import normalize_melt_oxides, oxides_wt_to_csj
from .wl_kd import DEFAULT_T_K, mpa_to_kbar
from .wl_state import equilibrium_state
from .wl_polybaric import polybaric_pressure_levels_mpa, polybaric_pressure_mpa
from .wl_partition import (
    clinopyroxene_di_pct,
    melt_an_fraction,
    olivine_fo_pct,
    plagioclase_an_pct,
)
from .wl_high_al_morb import HighAlFcRegime, is_high_al_morb
from .wl_saturation import incremental_modes

CrystallizationPathMode = Literal["fc_100", "eq_100", "polybaric_fc"]
KdEngine = Literal["heuristic", "langmuir", "basalt1990"]
KdMode1990 = Literal["1990", "langmuir"]

_OXIDE_KEYS = ("SiO2", "TiO2", "Al2O3", "Cr2O3", "FeO", "MgO", "CaO", "Na2O", "K2O")


def _residual_norm_vp_km_s(
    melt_wt: Mapping[str, float],
    *,
    f_solid: float,
    path: CrystallizationPathMode,
    p_eval_pa: float,
    t_eval_k: float,
    mineral_backend: Backend,
    cipw_backend: str,
    cfg: PetrologyConfig,
) -> float:
    """CIPW norm Vp for residual liquid, with optional Fig.2 residual calibration."""
    vp = float(
        norm_velocity_from_bulk_wt(
            melt_wt,
            p_pa=p_eval_pa,
            t_k=t_eval_k,
            cipw_backend=cipw_backend,
            mineral_backend=mineral_backend,
            config=cfg,
        )["vp_km_s"]
    )
    return apply_fig2_residual_norm_calibration(
        vp,
        f_solid=f_solid,
        path=path,
        p_pa=p_eval_pa,
        t_k=t_eval_k,
        mineral_backend=mineral_backend,
    )


def _incremental_vp_km_s(
    vp_raw: float,
    *,
    f_solid: float,
    path: CrystallizationPathMode,
    p_eval_pa: float,
    t_eval_k: float,
    mineral_backend: Backend,
) -> float:
    """Incremental solid Vp with optional Fig.2 calibration (cumulate Vp unchanged)."""
    return apply_fig2_incremental_vp_calibration(
        vp_raw,
        f_solid=f_solid,
        path=path,
        p_pa=p_eval_pa,
        t_k=t_eval_k,
        mineral_backend=mineral_backend,
    )
_SYSTEM_G = 100.0

PRIMARY_MELT_JSON = (
    Path(__file__).resolve().parents[1] / "data" / "mantle_melts" / "kinzler1997_morb_primary.json"
)


@dataclass(frozen=True)
class CrystallizationState:
    path: CrystallizationPathMode
    f_solid: float
    p_fc_mpa: float
    melt_oxides_wt: dict[str, float]
    vp_cumulate_km_s: float
    vp_incremental_km_s: float
    vp_residual_norm_km_s: float
    vp_eq_solid_km_s: float | None
    rho_cumulate_g_cm3: float
    rho_incremental_g_cm3: float
    rho_residual_liquid_g_cm3: float
    inc_ol: float
    inc_pl: float
    inc_cpx: float
    cum_ol: float
    cum_pl: float
    cum_cpx: float
    fo_pct: float
    an_pct: float
    di_pct: float
    cum_fo_pct: float
    cum_an_pct: float
    cum_di_pct: float


def load_kinzler1997_morb_primary(path: Path | str | None = None) -> dict:
    path = Path(path or PRIMARY_MELT_JSON)
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "id": data.get("id"),
        "P_melt_GPa": float(data["P_melt_GPa"]),
        "F_melt": float(data["F_melt"]),
        "oxides_wt_percent": normalize_melt_oxides(data["oxides_wt_percent"]),
    }


def _grams_from_wt(wt: Mapping[str, float]) -> dict[str, float]:
    return {k: float(wt[k]) * _SYSTEM_G / 100.0 for k in _OXIDE_KEYS}


def _wt_from_grams(grams: Mapping[str, float]) -> dict[str, float]:
    s = sum(max(0.0, float(grams[k])) for k in _OXIDE_KEYS)
    if s <= 0:
        raise ValueError("Empty composition")
    return {k: 100.0 * max(0.0, float(grams[k])) / s for k in _OXIDE_KEYS}


def _mineral_oxide_wt(*, fo_pct: float, an_pct: float, di_pct: float, phase: str) -> dict[str, float]:
    fo = float(np.clip(fo_pct / 100.0, 0.05, 0.99))
    an = float(np.clip(an_pct / 100.0, 0.02, 0.99))
    di = float(np.clip(di_pct / 100.0, 0.05, 0.99))
    if phase == "olivine":
        return {
            "SiO2": 40.0,
            "MgO": 57.0 * fo + 6.0 * (1.0 - fo),
            "FeO": 57.0 * (1.0 - fo) + 1.0 * fo,
        }
    if phase == "plagioclase":
        return {
            "SiO2": 52.0,
            "Al2O3": 30.0,
            "CaO": 14.0 * an,
            "Na2O": 10.0 * (1.0 - an),
            "K2O": 0.0,
        }
    if phase == "clinopyroxene":
        return {
            "SiO2": 51.0,
            "CaO": 22.0,
            "MgO": 16.0 * di,
            "FeO": 16.0 * (1.0 - di),
            "Al2O3": 3.5,
        }
    raise KeyError(phase)


def _increment_solid_wt(
    ol: float,
    pl: float,
    cpx: float,
    *,
    fo_pct: float,
    an_pct: float,
    di_pct: float,
) -> dict[str, float]:
    solid = {k: 0.0 for k in _OXIDE_KEYS}
    for phase, frac in (("olivine", ol), ("plagioclase", pl), ("clinopyroxene", cpx)):
        if frac <= 0.0:
            continue
        ox = _mineral_oxide_wt(fo_pct=fo_pct, an_pct=an_pct, di_pct=di_pct, phase=phase)
        norm = sum(ox.values())
        for k, v in ox.items():
            solid[k] += frac * 100.0 * v / norm
    return solid


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _oxide_feasibility_weights(
    melt: Mapping[str, float],
    *,
    fo_pct: float,
    an_pct: float,
    di_pct: float,
) -> tuple[float, float, float]:
    """Modern guard: zero phases when melt lacks required oxides."""

    def _weight(phase: str, key_oxides: tuple[str, ...]) -> float:
        ox = _mineral_oxide_wt(fo_pct=fo_pct, an_pct=an_pct, di_pct=di_pct, phase=phase)
        ratios: list[float] = []
        for k in key_oxides:
            need = float(ox.get(k, 0.0))
            if need < 0.5:
                continue
            avail = float(melt.get(k, 0.0))
            ratios.append(avail / need)
        if not ratios:
            return 1.0
        return float(np.clip(min(ratios) / 0.40, 0.0, 1.0) ** 0.5)

    w_ol = _weight("olivine", ("MgO", "SiO2"))
    w_pl = _weight("plagioclase", ("Al2O3", "CaO", "SiO2"))
    w_cpx = _weight("clinopyroxene", ("CaO", "MgO", "SiO2"))

    if float(melt.get("Al2O3", 0.0)) < 2.0:
        w_pl *= _sigmoid((float(melt.get("Al2O3", 0.0)) - 0.5) / 0.35)
    mg = float(melt.get("MgO", 0.0))
    if mg < 1.0:
        w_ol = 0.0
    elif mg < 4.0:
        w_ol *= _sigmoid((mg - 1.5) / 1.0)

    return w_ol, w_pl, w_cpx


def _constrained_modes(
    melt: Mapping[str, float],
    *,
    p_mpa: float,
    f_solid: float,
    path: CrystallizationPathMode,
    t_k: float = DEFAULT_T_K,
    polybaric_p_high_mpa: float = 800.0,
    polybaric_p_low_mpa: float = 100.0,
    polybaric_dp_mpa: float = 50.0,
    high_al_regime: HighAlFcRegime | None = None,
) -> tuple[float, float, float]:
    """W&L + Langmuir saturation modes + oxide feasibility guards."""
    ol, pl, cpx = incremental_modes(
        melt,
        p_mpa=p_mpa,
        f_solid=f_solid,
        path=path,
        t_k=t_k,
        polybaric_p_high_mpa=polybaric_p_high_mpa,
        polybaric_p_low_mpa=polybaric_p_low_mpa,
        polybaric_dp_mpa=polybaric_dp_mpa,
        high_al_regime=high_al_regime,
    )
    if high_al_regime is not None and high_al_regime.enabled:
        return ol, pl, cpx
    if is_high_al_morb(melt):
        return ol, pl, cpx
    fo = olivine_fo_pct(melt, t_k=t_k, p_mpa=p_mpa)
    an = plagioclase_an_pct(melt, t_k=t_k, p_mpa=p_mpa)
    di = clinopyroxene_di_pct(melt, t_k=t_k, p_mpa=p_mpa)

    w_ol, w_pl, w_cpx = _oxide_feasibility_weights(melt, fo_pct=fo, an_pct=an, di_pct=di)
    x = np.array([ol * w_ol, pl * w_pl, cpx * w_cpx], dtype=float)
    s = float(np.sum(x))
    if s <= 1e-12:
        return 1.0, 0.0, 0.0
    x /= s
    return float(x[0]), float(x[1]), float(x[2])


def _update_melt_fc(melt: Mapping[str, float], f_old: float, d_f: float, solid_wt: Mapping[str, float]) -> dict[str, float]:
    f_new = f_old + d_f
    denom = 1.0 - f_new
    if denom <= 1e-9:
        return normalize_melt_oxides(melt)

    max_df = d_f
    for k in _OXIDE_KEYS:
        c_s = float(solid_wt.get(k, 0.0))
        if c_s <= 0.0:
            continue
        avail = (1.0 - f_old) * float(melt[k])
        if avail <= 1e-8:
            if c_s < 0.10:
                continue
            max_df = 0.0
            break
        max_df = min(max_df, avail / c_s)

    if max_df <= 0.0:
        return normalize_melt_oxides(melt)

    f_new = f_old + max_df
    denom = 1.0 - f_new
    out: dict[str, float] = {}
    for k in _OXIDE_KEYS:
        numer = (1.0 - f_old) * float(melt[k]) - max_df * float(solid_wt.get(k, 0.0))
        out[k] = max(0.0, numer / denom)
    s = sum(out.values())
    if s <= 1e-9:
        return normalize_melt_oxides(melt)
    if abs(s - 100.0) > 0.5:
        return {k: 100.0 * out[k] / s for k in _OXIDE_KEYS}
    return out


def _crystallization_pressure(
    path: CrystallizationPathMode,
    f_solid: float,
    *,
    p_fc_fixed_mpa: float | None = None,
    polybaric_p_high_mpa: float = 800.0,
    polybaric_p_low_mpa: float = 100.0,
    polybaric_dp_mpa: float = 50.0,
) -> float:
    if path == "polybaric_fc":
        return polybaric_pressure_mpa(
            f_solid,
            p_high_mpa=polybaric_p_high_mpa,
            p_low_mpa=polybaric_p_low_mpa,
            dp_mpa=polybaric_dp_mpa,
        )
    if p_fc_fixed_mpa is not None:
        return float(p_fc_fixed_mpa)
    return 100.0


def _eq_liquid_from_solid(initial_g: Mapping[str, float], solid_g: Mapping[str, float], f_solid: float) -> dict[str, float]:
    if f_solid <= 0.0:
        return {k: float(initial_g[k]) for k in _OXIDE_KEYS}
    liq = {k: max(1e-9, float(initial_g[k]) - float(solid_g[k])) for k in _OXIDE_KEYS}
    target = _SYSTEM_G * (1.0 - f_solid)
    s = sum(liq.values())
    if s <= 0.0:
        return {k: float(initial_g[k]) for k in _OXIDE_KEYS}
    scale = target / s
    return {k: liq[k] * scale for k in _OXIDE_KEYS}


# Re-export for diagnostics / backward compatibility.
_melt_an_fraction = melt_an_fraction
_olivine_fo_pct = olivine_fo_pct
_plagioclase_an_pct = plagioclase_an_pct
_cpx_di_pct = clinopyroxene_di_pct


def _modes_from_state_fa(fa: np.ndarray) -> tuple[float, float, float]:
    """Incremental ol/pl/cpx from STATE FA (plag, ol, cpx)."""
    fa = np.asarray(fa, dtype=float).ravel()
    pl, ol, cpx = float(fa[0]), float(fa[1]), float(fa[2])
    s = pl + ol + cpx
    if s <= 1e-12:
        return 1.0, 0.0, 0.0
    return ol / s, pl / s, cpx / s


def _state_fa_needs_saturation_fallback(fa: np.ndarray, melt: Mapping[str, float]) -> bool:
    """
    Detect unphysical Langmuir STATE plag-monolith for olivine-bearing melts.

    High-MgO mantle melts (e.g. Walter 1998 @ 7 GPa) often return FA ≈ (1,0,0)
    from STATE while saturation + oxide guards yield Ol → Pl → Cpx FC paths.
    """
    fa = np.asarray(fa, dtype=float).ravel()
    if fa.size < 3:
        return True
    pl, ol, cpx = float(fa[0]), float(fa[1]), float(fa[2])
    s = pl + ol + cpx
    if s <= 1e-12:
        return True
    pl, ol, cpx = pl / s, ol / s, cpx / s
    mg = float(melt.get("MgO", 0.0))
    if pl >= 0.90 and ol < 0.05 and mg >= 8.0:
        return True
    if pl >= 0.85 and (ol + cpx) < 0.10 and mg >= 5.0:
        return True
    return False


def _state_modes_from_melt(
    melt_wt: Mapping[str, float],
    *,
    t_k: float,
    p_mpa: float,
    path: CrystallizationPathMode,
    f_solid: float,
    polybaric_p_high_mpa: float,
    polybaric_p_low_mpa: float,
    polybaric_dp_mpa: float,
    high_al_regime: HighAlFcRegime | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Incremental ol/pl/cpx + Fo/An/Di from Langmuir STATE at melt composition."""
    if high_al_regime is not None and high_al_regime.enabled:
        ol, pl, cpx = _constrained_modes(
            melt_wt,
            p_mpa=p_mpa,
            f_solid=f_solid,
            path=path,
            t_k=t_k,
            polybaric_p_high_mpa=polybaric_p_high_mpa,
            polybaric_p_low_mpa=polybaric_p_low_mpa,
            polybaric_dp_mpa=polybaric_dp_mpa,
            high_al_regime=high_al_regime,
        )
        fo = olivine_fo_pct(melt_wt, t_k=t_k, p_mpa=p_mpa)
        an = plagioclase_an_pct(melt_wt, t_k=t_k, p_mpa=p_mpa)
        di = clinopyroxene_di_pct(melt_wt, t_k=t_k, p_mpa=p_mpa)
        return ol, pl, cpx, fo, an, di

    csj = oxides_wt_to_csj(melt_wt)
    p_kbar = mpa_to_kbar(p_mpa)
    res = equilibrium_state(csj, t_k=t_k, p_kbar=p_kbar)
    if res.nl > 0 and not _state_fa_needs_saturation_fallback(res.fa, melt_wt):
        ol, pl, cpx = _modes_from_state_fa(res.fa)
        return ol, pl, cpx, res.fo_pct, res.an_pct, res.di_pct
    ol, pl, cpx = _constrained_modes(
        melt_wt,
        p_mpa=p_mpa,
        f_solid=f_solid,
        path=path,
        t_k=t_k,
        polybaric_p_high_mpa=polybaric_p_high_mpa,
        polybaric_p_low_mpa=polybaric_p_low_mpa,
        polybaric_dp_mpa=polybaric_dp_mpa,
    )
    fo = olivine_fo_pct(melt_wt, t_k=t_k, p_mpa=p_mpa)
    an = plagioclase_an_pct(melt_wt, t_k=t_k, p_mpa=p_mpa)
    di = clinopyroxene_di_pct(melt_wt, t_k=t_k, p_mpa=p_mpa)
    return ol, pl, cpx, fo, an, di


def _simulate_equilibrium_path_state(
    *,
    primary_melt_oxides_wt: Mapping[str, float],
    f_grid: np.ndarray,
    p_eval_mpa: float,
    t_eval_c: float,
    cipw_backend: str,
    mineral_backend: Backend,
    cfg: PetrologyConfig,
    p_fc_mpa: float | None,
    t_fc_k: float,
    sub_steps_per_interval: int,
) -> list[CrystallizationState]:
    """Batch equilibrium crystallization: STATE modes + lever-rule melt + HS Vp."""
    p_eval_pa = float(p_eval_mpa) * 1e6
    t_eval_k = float(t_eval_c) + 273.15
    initial_wt = normalize_melt_oxides(primary_melt_oxides_wt)
    initial_g = _grams_from_wt(initial_wt)
    melt_wt = dict(initial_wt)
    solid_total_g = {k: 0.0 for k in _OXIDE_KEYS}

    f_curr = 0.0
    cum_mass = {"ol": 0.0, "pl": 0.0, "cpx": 0.0}
    cum_fo_num = cum_fo_den = cum_an_num = cum_an_den = cum_di_num = cum_di_den = 0.0
    init_fo = olivine_fo_pct(initial_wt, t_k=t_fc_k, p_mpa=p_fc_mpa or 100.0)
    init_an = plagioclase_an_pct(initial_wt, t_k=t_fc_k, p_mpa=p_fc_mpa or 100.0)
    init_di = clinopyroxene_di_pct(initial_wt, t_k=t_fc_k, p_mpa=p_fc_mpa or 100.0)

    out: list[CrystallizationState] = []
    last_inc = (1.0, 0.0, 0.0, init_fo, init_an, init_di)
    high_al_regime = (
        HighAlFcRegime(
            enabled=True,
            high_al_morb=True,
            primary_al2o3=float(initial_wt.get("Al2O3", 0.0)),
        )
        if is_high_al_morb(initial_wt)
        else None
    )

    for f_target in f_grid:
        f_target = float(f_target)
        p_fc = _crystallization_pressure(
            "eq_100",
            f_target,
            p_fc_fixed_mpa=p_fc_mpa,
        )

        if f_target <= f_curr + 1e-12:
            fo, an, di = init_fo, init_an, init_di
            vp_inc, rho_inc = assemblage_vp_rho(
                ol_frac=1.0, pl_frac=0.0, cpx_frac=0.0, fo_pct=fo, an_pct=an, di_pct=di,
                p_pa=p_eval_pa, t_k=t_eval_k, mineral_backend=mineral_backend,
            )
            vp_inc_r = _incremental_vp_km_s(
                float(vp_inc),
                f_solid=0.0,
                path="eq_100",
                p_eval_pa=p_eval_pa,
                t_eval_k=t_eval_k,
                mineral_backend=mineral_backend,
            )
            res_vp = _residual_norm_vp_km_s(
                initial_wt,
                f_solid=0.0,
                path="eq_100",
                p_eval_pa=p_eval_pa,
                t_eval_k=t_eval_k,
                mineral_backend=mineral_backend,
                cipw_backend=cipw_backend,
                cfg=cfg,
            )
            out.append(
                CrystallizationState(
                    path="eq_100", f_solid=0.0, p_fc_mpa=p_fc,
                    melt_oxides_wt=dict(initial_wt),
                    vp_cumulate_km_s=vp_inc, vp_incremental_km_s=vp_inc_r,
                    vp_residual_norm_km_s=res_vp,
                    vp_eq_solid_km_s=vp_inc,
                    rho_cumulate_g_cm3=rho_inc, rho_incremental_g_cm3=rho_inc,
                    rho_residual_liquid_g_cm3=float(liquid_density_bw1970(initial_wt, t_c=t_eval_c)),
                    inc_ol=1.0, inc_pl=0.0, inc_cpx=0.0, cum_ol=1.0, cum_pl=0.0, cum_cpx=0.0,
                    fo_pct=fo, an_pct=an, di_pct=di,
                    cum_fo_pct=fo, cum_an_pct=an, cum_di_pct=di,
                )
            )
            continue

        d_f_step = (f_target - f_curr) / sub_steps_per_interval

        for istep in range(sub_steps_per_interval):
            f_next = f_curr + d_f_step * (istep + 1)
            p_here = _crystallization_pressure("eq_100", f_next, p_fc_fixed_mpa=p_fc_mpa)
            ol, pl, cpx, fo, an, di = _state_modes_from_melt(
                melt_wt,
                t_k=t_fc_k,
                p_mpa=p_here,
                path="eq_100",
                f_solid=f_next,
                polybaric_p_high_mpa=800.0,
                polybaric_p_low_mpa=100.0,
                polybaric_dp_mpa=50.0,
                high_al_regime=high_al_regime,
            )
            solid_wt = _increment_solid_wt(ol, pl, cpx, fo_pct=fo, an_pct=an, di_pct=di)
            last_inc = (ol, pl, cpx, fo, an, di)
            solid_total_g = {
                k: solid_total_g[k] + d_f_step * _SYSTEM_G * solid_wt[k] / 100.0
                for k in _OXIDE_KEYS
            }
            melt_wt = _wt_from_grams(_eq_liquid_from_solid(initial_g, solid_total_g, f_next))
            cum_mass["ol"] += d_f_step * ol
            cum_mass["pl"] += d_f_step * pl
            cum_mass["cpx"] += d_f_step * cpx
            if ol > 0.0:
                cum_fo_num += d_f_step * ol * fo
                cum_fo_den += d_f_step * ol
            if pl > 0.0:
                cum_an_num += d_f_step * pl * an
                cum_an_den += d_f_step * pl
            if cpx > 0.0:
                cum_di_num += d_f_step * cpx * di
                cum_di_den += d_f_step * cpx

        f_curr = f_target
        ol, pl, cpx, fo_inc, an_inc, di_inc = last_inc
        cum_fo = cum_fo_num / cum_fo_den if cum_fo_den > 0.0 else init_fo
        cum_an = cum_an_num / cum_an_den if cum_an_den > 0.0 else init_an
        cum_di = cum_di_num / cum_di_den if cum_di_den > 0.0 else init_di

        vp_inc, rho_inc = assemblage_vp_rho(
            ol_frac=ol, pl_frac=pl, cpx_frac=cpx,
            fo_pct=fo_inc, an_pct=an_inc, di_pct=di_inc,
            p_pa=p_eval_pa, t_k=t_eval_k, mineral_backend=mineral_backend,
        )
        vp_inc = _incremental_vp_km_s(
            float(vp_inc),
            f_solid=f_target,
            path="eq_100",
            p_eval_pa=p_eval_pa,
            t_eval_k=t_eval_k,
            mineral_backend=mineral_backend,
        )
        cum_total = max(sum(cum_mass.values()), 1e-9)
        vp_cum, rho_cum = assemblage_vp_rho(
            ol_frac=cum_mass["ol"] / cum_total,
            pl_frac=cum_mass["pl"] / cum_total,
            cpx_frac=cum_mass["cpx"] / cum_total,
            fo_pct=cum_fo, an_pct=cum_an, di_pct=cum_di,
            p_pa=p_eval_pa, t_k=t_eval_k, mineral_backend=mineral_backend,
        )
        res_vp = _residual_norm_vp_km_s(
            melt_wt,
            f_solid=f_target,
            path="eq_100",
            p_eval_pa=p_eval_pa,
            t_eval_k=t_eval_k,
            mineral_backend=mineral_backend,
            cipw_backend=cipw_backend,
            cfg=cfg,
        )

        vp_eq_solid = None
        if f_target > 0.0:
            avg_wt = _wt_from_grams({k: solid_total_g[k] / f_target for k in _OXIDE_KEYS})
            o2, p2, c2, fo2, an2, di2 = _state_modes_from_melt(
                avg_wt,
                t_k=t_fc_k,
                p_mpa=p_fc,
                path="eq_100",
                f_solid=f_target,
                polybaric_p_high_mpa=800.0,
                polybaric_p_low_mpa=100.0,
                polybaric_dp_mpa=50.0,
            )
            vp_eq_solid, _ = assemblage_vp_rho(
                ol_frac=o2, pl_frac=p2, cpx_frac=c2,
                fo_pct=fo2, an_pct=an2, di_pct=di2,
                p_pa=p_eval_pa, t_k=t_eval_k, mineral_backend=mineral_backend,
            )

        out.append(
            CrystallizationState(
                path="eq_100", f_solid=f_target, p_fc_mpa=p_fc,
                melt_oxides_wt={k: float(melt_wt[k]) for k in _OXIDE_KEYS},
                vp_cumulate_km_s=float(vp_cum), vp_incremental_km_s=float(vp_inc),
                vp_residual_norm_km_s=res_vp,
                vp_eq_solid_km_s=float(vp_eq_solid) if vp_eq_solid is not None else None,
                rho_cumulate_g_cm3=float(rho_cum), rho_incremental_g_cm3=float(rho_inc),
                rho_residual_liquid_g_cm3=float(liquid_density_bw1970(melt_wt, t_c=t_eval_c)),
                inc_ol=float(ol), inc_pl=float(pl), inc_cpx=float(cpx),
                cum_ol=float(cum_mass["ol"] / cum_total),
                cum_pl=float(cum_mass["pl"] / cum_total),
                cum_cpx=float(cum_mass["cpx"] / cum_total),
                fo_pct=float(fo_inc), an_pct=float(an_inc), di_pct=float(di_inc),
                cum_fo_pct=float(cum_fo), cum_an_pct=float(cum_an), cum_di_pct=float(cum_di),
            )
        )

    return out


def _simulate_fractional_path_state(
    *,
    primary_melt_oxides_wt: Mapping[str, float],
    f_grid: np.ndarray,
    path: CrystallizationPathMode,
    p_eval_mpa: float,
    t_eval_c: float,
    cipw_backend: str,
    mineral_backend: Backend,
    cfg: PetrologyConfig,
    p_fc_mpa: float | None,
    t_fc_k: float,
    polybaric_p_high_mpa: float,
    polybaric_p_low_mpa: float,
    polybaric_dp_mpa: float,
    sub_steps_per_interval: int,
) -> list[CrystallizationState]:
    """Fractional crystallization: STATE incremental modes + FC mass balance + HS Vp."""
    p_eval_pa = float(p_eval_mpa) * 1e6
    t_eval_k = float(t_eval_c) + 273.15
    initial_wt = normalize_melt_oxides(primary_melt_oxides_wt)
    melt_wt = dict(initial_wt)

    init_fo = olivine_fo_pct(initial_wt, t_k=t_fc_k, p_mpa=p_fc_mpa or 100.0)
    init_an = plagioclase_an_pct(initial_wt, t_k=t_fc_k, p_mpa=p_fc_mpa or 100.0)
    init_di = clinopyroxene_di_pct(initial_wt, t_k=t_fc_k, p_mpa=p_fc_mpa or 100.0)

    f_curr = 0.0
    cum_mass = {"ol": 0.0, "pl": 0.0, "cpx": 0.0}
    cum_fo_num = cum_fo_den = cum_an_num = cum_an_den = cum_di_num = cum_di_den = 0.0
    out: list[CrystallizationState] = []
    last_inc = (1.0, 0.0, 0.0, init_fo, init_an, init_di)
    high_al_regime = (
        HighAlFcRegime(
            enabled=True,
            high_al_morb=True,
            primary_al2o3=float(initial_wt.get("Al2O3", 0.0)),
        )
        if is_high_al_morb(initial_wt)
        else None
    )

    for f_target in f_grid:
        f_target = float(f_target)
        p_fc = _crystallization_pressure(
            path,
            f_target,
            p_fc_fixed_mpa=p_fc_mpa,
            polybaric_p_high_mpa=polybaric_p_high_mpa,
            polybaric_p_low_mpa=polybaric_p_low_mpa,
            polybaric_dp_mpa=polybaric_dp_mpa,
        )

        if f_target <= f_curr + 1e-12:
            fo, an, di = init_fo, init_an, init_di
            vp_inc, rho_inc = assemblage_vp_rho(
                ol_frac=1.0, pl_frac=0.0, cpx_frac=0.0, fo_pct=fo, an_pct=an, di_pct=di,
                p_pa=p_eval_pa, t_k=t_eval_k, mineral_backend=mineral_backend,
            )
            vp_inc_r = _incremental_vp_km_s(
                float(vp_inc),
                f_solid=0.0,
                path=path,
                p_eval_pa=p_eval_pa,
                t_eval_k=t_eval_k,
                mineral_backend=mineral_backend,
            )
            res_vp = _residual_norm_vp_km_s(
                initial_wt,
                f_solid=0.0,
                path=path,
                p_eval_pa=p_eval_pa,
                t_eval_k=t_eval_k,
                mineral_backend=mineral_backend,
                cipw_backend=cipw_backend,
                cfg=cfg,
            )
            out.append(
                CrystallizationState(
                    path=path, f_solid=0.0, p_fc_mpa=p_fc,
                    melt_oxides_wt=dict(initial_wt),
                    vp_cumulate_km_s=vp_inc, vp_incremental_km_s=vp_inc_r,
                    vp_residual_norm_km_s=res_vp,
                    vp_eq_solid_km_s=None,
                    rho_cumulate_g_cm3=rho_inc, rho_incremental_g_cm3=rho_inc,
                    rho_residual_liquid_g_cm3=float(liquid_density_bw1970(initial_wt, t_c=t_eval_c)),
                    inc_ol=1.0, inc_pl=0.0, inc_cpx=0.0, cum_ol=1.0, cum_pl=0.0, cum_cpx=0.0,
                    fo_pct=fo, an_pct=an, di_pct=di,
                    cum_fo_pct=fo, cum_an_pct=an, cum_di_pct=di,
                )
            )
            continue

        d_f_step = (f_target - f_curr) / sub_steps_per_interval

        for istep in range(sub_steps_per_interval):
            f_next = f_curr + d_f_step * (istep + 1)
            p_here = _crystallization_pressure(
                path,
                f_next,
                p_fc_fixed_mpa=p_fc_mpa,
                polybaric_p_high_mpa=polybaric_p_high_mpa,
                polybaric_p_low_mpa=polybaric_p_low_mpa,
                polybaric_dp_mpa=polybaric_dp_mpa,
            )
            ol, pl, cpx, fo, an, di = _state_modes_from_melt(
                melt_wt,
                t_k=t_fc_k,
                p_mpa=p_here,
                path=path,
                f_solid=f_next,
                polybaric_p_high_mpa=polybaric_p_high_mpa,
                polybaric_p_low_mpa=polybaric_p_low_mpa,
                polybaric_dp_mpa=polybaric_dp_mpa,
                high_al_regime=high_al_regime,
            )
            solid_wt = _increment_solid_wt(ol, pl, cpx, fo_pct=fo, an_pct=an, di_pct=di)
            last_inc = (ol, pl, cpx, fo, an, di)
            melt_wt = _update_melt_fc(melt_wt, f_curr + d_f_step * istep, d_f_step, solid_wt)
            cum_mass["ol"] += d_f_step * ol
            cum_mass["pl"] += d_f_step * pl
            cum_mass["cpx"] += d_f_step * cpx
            if ol > 0.0:
                cum_fo_num += d_f_step * ol * fo
                cum_fo_den += d_f_step * ol
            if pl > 0.0:
                cum_an_num += d_f_step * pl * an
                cum_an_den += d_f_step * pl
            if cpx > 0.0:
                cum_di_num += d_f_step * cpx * di
                cum_di_den += d_f_step * cpx

        f_curr = f_target
        ol, pl, cpx, fo_inc, an_inc, di_inc = last_inc
        cum_fo = cum_fo_num / cum_fo_den if cum_fo_den > 0.0 else init_fo
        cum_an = cum_an_num / cum_an_den if cum_an_den > 0.0 else init_an
        cum_di = cum_di_num / cum_di_den if cum_di_den > 0.0 else init_di

        vp_inc, rho_inc = assemblage_vp_rho(
            ol_frac=ol, pl_frac=pl, cpx_frac=cpx,
            fo_pct=fo_inc, an_pct=an_inc, di_pct=di_inc,
            p_pa=p_eval_pa, t_k=t_eval_k, mineral_backend=mineral_backend,
        )
        vp_inc = _incremental_vp_km_s(
            float(vp_inc),
            f_solid=f_target,
            path=path,
            p_eval_pa=p_eval_pa,
            t_eval_k=t_eval_k,
            mineral_backend=mineral_backend,
        )
        cum_total = max(sum(cum_mass.values()), 1e-9)
        vp_cum, rho_cum = assemblage_vp_rho(
            ol_frac=cum_mass["ol"] / cum_total,
            pl_frac=cum_mass["pl"] / cum_total,
            cpx_frac=cum_mass["cpx"] / cum_total,
            fo_pct=cum_fo, an_pct=cum_an, di_pct=cum_di,
            p_pa=p_eval_pa, t_k=t_eval_k, mineral_backend=mineral_backend,
        )
        res_vp = _residual_norm_vp_km_s(
            melt_wt,
            f_solid=f_target,
            path=path,
            p_eval_pa=p_eval_pa,
            t_eval_k=t_eval_k,
            mineral_backend=mineral_backend,
            cipw_backend=cipw_backend,
            cfg=cfg,
        )
        out.append(
            CrystallizationState(
                path=path, f_solid=f_target, p_fc_mpa=p_fc,
                melt_oxides_wt={k: float(melt_wt[k]) for k in _OXIDE_KEYS},
                vp_cumulate_km_s=float(vp_cum), vp_incremental_km_s=float(vp_inc),
                vp_residual_norm_km_s=res_vp,
                vp_eq_solid_km_s=None,
                rho_cumulate_g_cm3=float(rho_cum), rho_incremental_g_cm3=float(rho_inc),
                rho_residual_liquid_g_cm3=float(liquid_density_bw1970(melt_wt, t_c=t_eval_c)),
                inc_ol=float(ol), inc_pl=float(pl), inc_cpx=float(cpx),
                cum_ol=float(cum_mass["ol"] / cum_total),
                cum_pl=float(cum_mass["pl"] / cum_total),
                cum_cpx=float(cum_mass["cpx"] / cum_total),
                fo_pct=float(fo_inc), an_pct=float(an_inc), di_pct=float(di_inc),
                cum_fo_pct=float(cum_fo), cum_an_pct=float(cum_an), cum_di_pct=float(cum_di),
            )
        )

    return out


def _simulate_crystallization_path_state(
    *,
    primary_melt_oxides_wt: Mapping[str, float],
    f_grid: np.ndarray,
    path: CrystallizationPathMode,
    p_eval_mpa: float,
    t_eval_c: float,
    cipw_backend: str,
    mineral_backend: Backend,
    cfg: PetrologyConfig,
    p_fc_mpa: float | None,
    t_fc_k: float,
    polybaric_p_high_mpa: float,
    polybaric_p_low_mpa: float,
    polybaric_dp_mpa: float,
    state_fc_substeps: int,
    sub_steps_per_interval: int = 50,
) -> list[CrystallizationState]:
    if path == "eq_100":
        return _simulate_equilibrium_path_state(
            primary_melt_oxides_wt=primary_melt_oxides_wt,
            f_grid=f_grid,
            p_eval_mpa=p_eval_mpa,
            t_eval_c=t_eval_c,
            cipw_backend=cipw_backend,
            mineral_backend=mineral_backend,
            cfg=cfg,
            p_fc_mpa=p_fc_mpa,
            t_fc_k=t_fc_k,
            sub_steps_per_interval=sub_steps_per_interval,
        )

    return _simulate_fractional_path_state(
        primary_melt_oxides_wt=primary_melt_oxides_wt,
        f_grid=f_grid,
        path=path,
        p_eval_mpa=p_eval_mpa,
        t_eval_c=t_eval_c,
        cipw_backend=cipw_backend,
        mineral_backend=mineral_backend,
        cfg=cfg,
        p_fc_mpa=p_fc_mpa,
        t_fc_k=t_fc_k,
        polybaric_p_high_mpa=polybaric_p_high_mpa,
        polybaric_p_low_mpa=polybaric_p_low_mpa,
        polybaric_dp_mpa=polybaric_dp_mpa,
        sub_steps_per_interval=sub_steps_per_interval,
    )


def _simulate_polybaric_heuristic(
    *,
    primary_melt_oxides_wt: Mapping[str, float],
    f_grid: np.ndarray,
    p_eval_mpa: float,
    t_eval_c: float,
    cipw_backend: str,
    mineral_backend: Backend,
    cfg: PetrologyConfig,
    sub_steps_per_interval: int,
    t_fc_k: float,
    polybaric_p_high_mpa: float,
    polybaric_p_low_mpa: float,
    polybaric_dp_mpa: float,
) -> list[CrystallizationState]:
    """Fortran polybaric DRIVER with oxide-heuristic FC inside each P segment."""
    p_eval_pa = float(p_eval_mpa) * 1e6
    t_eval_k = float(t_eval_c) + 273.15
    initial_wt = normalize_melt_oxides(primary_melt_oxides_wt)
    melt_wt = dict(initial_wt)

    init_fo = olivine_fo_pct(initial_wt, t_k=t_fc_k, p_mpa=polybaric_p_high_mpa)
    init_an = plagioclase_an_pct(initial_wt, t_k=t_fc_k, p_mpa=polybaric_p_high_mpa)
    init_di = clinopyroxene_di_pct(initial_wt, t_k=t_fc_k, p_mpa=polybaric_p_high_mpa)

    levels = polybaric_pressure_levels_mpa(
        p_high_mpa=polybaric_p_high_mpa,
        p_low_mpa=polybaric_p_low_mpa,
        dp_mpa=polybaric_dp_mpa,
    )
    f_max = float(f_grid[-1])
    n_steps = max(len(levels) * sub_steps_per_interval, 1)
    d_f = f_max / n_steps

    f_curr = 0.0
    cum_mass = {"ol": 0.0, "pl": 0.0, "cpx": 0.0}
    cum_fo_num = cum_fo_den = cum_an_num = cum_an_den = cum_di_num = cum_di_den = 0.0
    last_inc = (1.0, 0.0, 0.0, init_fo, init_an, init_di)
    out: list[CrystallizationState] = []
    fg_i = 0

    def _append_state(f_target: float, p_fc: float, melt: Mapping[str, float]) -> None:
        nonlocal cum_fo_num, cum_fo_den, cum_an_num, cum_an_den, cum_di_num, cum_di_den
        ol, pl, cpx, fo_inc, an_inc, di_inc = last_inc
        cum_fo = cum_fo_num / cum_fo_den if cum_fo_den > 0.0 else init_fo
        cum_an = cum_an_num / cum_an_den if cum_an_den > 0.0 else init_an
        cum_di = cum_di_num / cum_di_den if cum_di_den > 0.0 else init_di
        cum_total = max(sum(cum_mass.values()), 1e-9)

        vp_inc, rho_inc = assemblage_vp_rho(
            ol_frac=ol, pl_frac=pl, cpx_frac=cpx,
            fo_pct=fo_inc, an_pct=an_inc, di_pct=di_inc,
            p_pa=p_eval_pa, t_k=t_eval_k, mineral_backend=mineral_backend,
        )
        vp_inc = _incremental_vp_km_s(
            float(vp_inc),
            f_solid=f_target,
            path="polybaric_fc",
            p_eval_pa=p_eval_pa,
            t_eval_k=t_eval_k,
            mineral_backend=mineral_backend,
        )
        vp_cum, rho_cum = assemblage_vp_rho(
            ol_frac=cum_mass["ol"] / cum_total,
            pl_frac=cum_mass["pl"] / cum_total,
            cpx_frac=cum_mass["cpx"] / cum_total,
            fo_pct=cum_fo, an_pct=cum_an, di_pct=cum_di,
            p_pa=p_eval_pa, t_k=t_eval_k, mineral_backend=mineral_backend,
        )
        res_vp = _residual_norm_vp_km_s(
            melt,
            f_solid=f_target,
            path="polybaric_fc",
            p_eval_pa=p_eval_pa,
            t_eval_k=t_eval_k,
            mineral_backend=mineral_backend,
            cipw_backend=cipw_backend,
            cfg=cfg,
        )
        out.append(
            CrystallizationState(
                path="polybaric_fc",
                f_solid=f_target,
                p_fc_mpa=float(p_fc),
                melt_oxides_wt={k: float(melt[k]) for k in _OXIDE_KEYS},
                vp_cumulate_km_s=float(vp_cum),
                vp_incremental_km_s=float(vp_inc),
                vp_residual_norm_km_s=res_vp,
                vp_eq_solid_km_s=None,
                rho_cumulate_g_cm3=float(rho_cum),
                rho_incremental_g_cm3=float(rho_inc),
                rho_residual_liquid_g_cm3=float(liquid_density_bw1970(melt, t_c=t_eval_c)),
                inc_ol=float(ol), inc_pl=float(pl), inc_cpx=float(cpx),
                cum_ol=float(cum_mass["ol"] / cum_total),
                cum_pl=float(cum_mass["pl"] / cum_total),
                cum_cpx=float(cum_mass["cpx"] / cum_total),
                fo_pct=float(fo_inc), an_pct=float(an_inc), di_pct=float(di_inc),
                cum_fo_pct=float(cum_fo), cum_an_pct=float(cum_an), cum_di_pct=float(cum_di),
            )
        )

    while fg_i < len(f_grid) and float(f_grid[fg_i]) <= 1e-12:
        fo, an, di = init_fo, init_an, init_di
        ol, pl, cpx = 1.0, 0.0, 0.0
        last_inc = (ol, pl, cpx, fo, an, di)
        vp_inc, rho_inc = assemblage_vp_rho(
            ol_frac=ol, pl_frac=pl, cpx_frac=cpx, fo_pct=fo, an_pct=an, di_pct=di,
            p_pa=p_eval_pa, t_k=t_eval_k, mineral_backend=mineral_backend,
        )
        vp_inc_r = _incremental_vp_km_s(
            float(vp_inc),
            f_solid=0.0,
            path="polybaric_fc",
            p_eval_pa=p_eval_pa,
            t_eval_k=t_eval_k,
            mineral_backend=mineral_backend,
        )
        res_vp = _residual_norm_vp_km_s(
            initial_wt,
            f_solid=0.0,
            path="polybaric_fc",
            p_eval_pa=p_eval_pa,
            t_eval_k=t_eval_k,
            mineral_backend=mineral_backend,
            cipw_backend=cipw_backend,
            cfg=cfg,
        )
        out.append(
            CrystallizationState(
                path="polybaric_fc",
                f_solid=0.0,
                p_fc_mpa=float(levels[0]),
                melt_oxides_wt=dict(initial_wt),
                vp_cumulate_km_s=float(vp_inc),
                vp_incremental_km_s=float(vp_inc_r),
                vp_residual_norm_km_s=res_vp,
                vp_eq_solid_km_s=None,
                rho_cumulate_g_cm3=float(rho_inc),
                rho_incremental_g_cm3=float(rho_inc),
                rho_residual_liquid_g_cm3=float(liquid_density_bw1970(initial_wt, t_c=t_eval_c)),
                inc_ol=ol, inc_pl=pl, inc_cpx=cpx,
                cum_ol=ol, cum_pl=pl, cum_cpx=cpx,
                fo_pct=fo, an_pct=an, di_pct=di,
                cum_fo_pct=fo, cum_an_pct=an, cum_di_pct=di,
            )
        )
        fg_i += 1

    for p_mpa in levels:
        for _ in range(sub_steps_per_interval):
            if f_curr >= f_max - 1e-9:
                break
            f_next = min(f_curr + d_f, f_max)
            step = f_next - f_curr
            ol, pl, cpx = _constrained_modes(
                melt_wt,
                p_mpa=p_mpa,
                f_solid=f_next,
                path="fc_100",
                t_k=t_fc_k,
            )
            fo = olivine_fo_pct(melt_wt, t_k=t_fc_k, p_mpa=p_mpa)
            an = plagioclase_an_pct(melt_wt, t_k=t_fc_k, p_mpa=p_mpa)
            di = clinopyroxene_di_pct(melt_wt, t_k=t_fc_k, p_mpa=p_mpa)
            solid_wt = _increment_solid_wt(ol, pl, cpx, fo_pct=fo, an_pct=an, di_pct=di)
            last_inc = (ol, pl, cpx, fo, an, di)
            melt_wt = _update_melt_fc(melt_wt, f_curr, step, solid_wt)

            cum_mass["ol"] += step * ol
            cum_mass["pl"] += step * pl
            cum_mass["cpx"] += step * cpx
            if ol > 0.0:
                cum_fo_num += step * ol * fo
                cum_fo_den += step * ol
            if pl > 0.0:
                cum_an_num += step * pl * an
                cum_an_den += step * pl
            if cpx > 0.0:
                cum_di_num += step * cpx * di
                cum_di_den += step * cpx

            f_curr = f_next
            while fg_i < len(f_grid) and f_curr >= float(f_grid[fg_i]) - 1e-9:
                _append_state(float(f_grid[fg_i]), p_mpa, melt_wt)
                fg_i += 1

    while fg_i < len(f_grid):
        _append_state(float(f_grid[fg_i]), levels[-1], melt_wt)
        fg_i += 1

    return out


def _simulate_fig2_ab(
    *,
    primary_melt_oxides_wt: Mapping[str, float],
    f_grid: np.ndarray,
    path: CrystallizationPathMode,
    p_eval_mpa: float,
    t_eval_c: float,
    cipw_backend: str,
    mineral_backend: Backend,
    cfg: PetrologyConfig,
    sub_steps_per_interval: int,
    p_fc_mpa: float | None,
    t_fc_k: float,
    polybaric_p_high_mpa: float,
    polybaric_p_low_mpa: float,
    polybaric_dp_mpa: float,
    fig2_ab_anchor_vp: bool = False,
) -> list[CrystallizationState]:
    """KKHS02 Fig.2 (a,b): paper phase schedule + mass balance.

    ``fig2_ab_anchor_vp=False`` (default): Vp/rho from HS + CIPW norm (Korenaga §2.1).
    ``fig2_ab_anchor_vp=True``: interpolate sparse Fig.2(a,b) anchor curves (layout only).
    """
    p_eval_pa = float(p_eval_mpa) * 1e6
    t_eval_k = float(t_eval_c) + 273.15
    initial_wt = normalize_melt_oxides(primary_melt_oxides_wt)
    initial_g = _grams_from_wt(initial_wt)
    melt_wt = dict(initial_wt)
    solid_total_g = {k: 0.0 for k in _OXIDE_KEYS}

    f_curr = 0.0
    cum_mass = {"ol": 0.0, "pl": 0.0, "cpx": 0.0}
    cum_fo_num = cum_fo_den = cum_an_num = cum_an_den = cum_di_num = cum_di_den = 0.0
    out: list[CrystallizationState] = []
    p_key = path
    f_prev_out = 0.0

    def _vp_rho_bundle(
        *,
        ol_i: float,
        pl_i: float,
        cpx_i: float,
        fo_i: float,
        an_i: float,
        di_i: float,
        ol_c: float,
        pl_c: float,
        cpx_c: float,
        fo_c: float,
        an_c: float,
        di_c: float,
        melt_wt: Mapping[str, float],
        f_solid: float,
    ) -> tuple[float, float, float, float, float, float | None]:
        if fig2_ab_anchor_vp:
            vp_inc = paper_inc_vp_km_s(p_key, f_solid)
            rho_inc = paper_inc_rho_g_cm3(p_key, f_solid)
            vp_cum = paper_cum_vp_km_s(p_key, f_solid)
            rho_cum = paper_cum_rho_g_cm3(p_key, f_solid)
            res_vp = (
                paper_residual_vp_km_s(p_key, f_solid)
                if path != "eq_100"
                else _residual_norm_vp_km_s(
                    melt_wt,
                    f_solid=f_solid,
                    path=path,
                    p_eval_pa=p_eval_pa,
                    t_eval_k=t_eval_k,
                    mineral_backend=mineral_backend,
                    cipw_backend=cipw_backend,
                    cfg=cfg,
                )
            )
            res_rho = (
                paper_residual_rho_g_cm3(p_key, f_solid)
                if path != "eq_100"
                else float(liquid_density_bw1970(melt_wt, t_c=t_eval_c))
            )
            vp_eq = vp_cum if path == "eq_100" and f_solid > 0.0 else None
            return vp_cum, vp_inc, res_vp, rho_cum, rho_inc, res_rho, vp_eq

        vp_inc, rho_inc = assemblage_vp_rho(
            ol_frac=ol_i, pl_frac=pl_i, cpx_frac=cpx_i,
            fo_pct=fo_i, an_pct=an_i, di_pct=di_i,
            p_pa=p_eval_pa, t_k=t_eval_k, mineral_backend=mineral_backend,
        )
        vp_cum, rho_cum = assemblage_vp_rho(
            ol_frac=ol_c, pl_frac=pl_c, cpx_frac=cpx_c,
            fo_pct=fo_c, an_pct=an_c, di_pct=di_c,
            p_pa=p_eval_pa, t_k=t_eval_k, mineral_backend=mineral_backend,
        )
        vp_inc = _incremental_vp_km_s(
            vp_inc,
            f_solid=f_solid,
            path=p_key,
            p_eval_pa=p_eval_pa,
            t_eval_k=t_eval_k,
            mineral_backend=mineral_backend,
        )
        res_vp = _residual_norm_vp_km_s(
            melt_wt,
            f_solid=f_solid,
            path=p_key,
            p_eval_pa=p_eval_pa,
            t_eval_k=t_eval_k,
            mineral_backend=mineral_backend,
            cipw_backend=cipw_backend,
            cfg=cfg,
        )
        res_rho = float(liquid_density_bw1970(melt_wt, t_c=t_eval_c))
        vp_eq = vp_cum if path == "eq_100" and f_solid > 0.0 else None
        return vp_cum, vp_inc, res_vp, rho_cum, rho_inc, res_rho, vp_eq

    for f_target in f_grid:
        f_target = float(f_target)
        p_fc = _crystallization_pressure(
            path,
            f_target,
            p_fc_fixed_mpa=p_fc_mpa,
            polybaric_p_high_mpa=polybaric_p_high_mpa,
            polybaric_p_low_mpa=polybaric_p_low_mpa,
            polybaric_dp_mpa=polybaric_dp_mpa,
        )

        if f_target <= f_curr + 1e-12:
            fo, an, di = incremental_compositions(p_key, 0.0)
            ol, pl, cpx = 1.0, 0.0, 0.0
            vp_cum, vp_inc, res_vp, rho_cum, rho_inc, res_rho, vp_eq = _vp_rho_bundle(
                ol_i=ol, pl_i=pl, cpx_i=cpx, fo_i=fo, an_i=an, di_i=di,
                ol_c=ol, pl_c=pl, cpx_c=cpx, fo_c=fo, an_c=an, di_c=di,
                melt_wt=initial_wt, f_solid=0.0,
            )
            out.append(
                CrystallizationState(
                    path=path, f_solid=0.0, p_fc_mpa=p_fc,
                    melt_oxides_wt=dict(initial_wt),
                    vp_cumulate_km_s=vp_cum, vp_incremental_km_s=vp_inc,
                    vp_residual_norm_km_s=res_vp,
                    vp_eq_solid_km_s=vp_eq,
                    rho_cumulate_g_cm3=rho_cum, rho_incremental_g_cm3=rho_inc,
                    rho_residual_liquid_g_cm3=res_rho,
                    inc_ol=ol, inc_pl=pl, inc_cpx=cpx, cum_ol=ol, cum_pl=pl, cum_cpx=cpx,
                    fo_pct=fo, an_pct=an, di_pct=di,
                    cum_fo_pct=fo, cum_an_pct=an, cum_di_pct=di,
                )
            )
            continue

        d_f_step = (f_target - f_curr) / sub_steps_per_interval

        for istep in range(sub_steps_per_interval):
            f_prev_step = f_curr + d_f_step * istep
            f_next = f_curr + d_f_step * (istep + 1)
            ol, pl, cpx = incremental_modes_from_cumulative(p_key, f_next, f_prev_step)
            fo, an, di = incremental_compositions(p_key, f_next)
            solid_wt = _increment_solid_wt(ol, pl, cpx, fo_pct=fo, an_pct=an, di_pct=di)
            d_f = f_next - f_prev_step

            if path == "eq_100":
                solid_total_g = {
                    k: solid_total_g[k] + d_f * _SYSTEM_G * solid_wt[k] / 100.0
                    for k in _OXIDE_KEYS
                }
                melt_wt = _wt_from_grams(_eq_liquid_from_solid(initial_g, solid_total_g, f_next))
            else:
                melt_wt = _update_melt_fc(melt_wt, f_prev_step, d_f, solid_wt)

            cum_mass["ol"] += d_f * ol
            cum_mass["pl"] += d_f * pl
            cum_mass["cpx"] += d_f * cpx
            if ol > 0.0:
                cum_fo_num += d_f * ol * fo
                cum_fo_den += d_f * ol
            if pl > 0.0:
                cum_an_num += d_f * pl * an
                cum_an_den += d_f * pl
            if cpx > 0.0:
                cum_di_num += d_f * cpx * di
                cum_di_den += d_f * cpx

        f_curr = f_target

        ol_pct, cpx_pct, pl_pct = cumulative_phases_pct(p_key, f_target)
        cum_ol = ol_pct / 100.0
        cum_pl = pl_pct / 100.0
        cum_cpx = cpx_pct / 100.0
        ol, pl, cpx = incremental_modes_from_cumulative(p_key, f_target, f_prev_out)
        fo_inc, an_inc, di_inc = incremental_compositions(p_key, f_target)
        cum_fo, cum_an, cum_di = cumulative_compositions_weighted(
            p_key, f_target,
            cum_ol=cum_ol, cum_pl=cum_pl, cum_cpx=cum_cpx,
            fo_num=cum_fo_num, fo_den=cum_fo_den,
            an_num=cum_an_num, an_den=cum_an_den,
            di_num=cum_di_num, di_den=cum_di_den,
        )

        vp_cum, vp_inc, res_vp, rho_cum, rho_inc, res_rho, vp_eq = _vp_rho_bundle(
            ol_i=ol, pl_i=pl, cpx_i=cpx,
            fo_i=fo_inc, an_i=an_inc, di_i=di_inc,
            ol_c=cum_ol, pl_c=cum_pl, cpx_c=cum_cpx,
            fo_c=cum_fo, an_c=cum_an, di_c=cum_di,
            melt_wt=melt_wt, f_solid=f_target,
        )

        out.append(
            CrystallizationState(
                path=path, f_solid=f_target, p_fc_mpa=p_fc,
                melt_oxides_wt={k: float(melt_wt[k]) for k in _OXIDE_KEYS},
                vp_cumulate_km_s=float(vp_cum), vp_incremental_km_s=float(vp_inc),
                vp_residual_norm_km_s=float(res_vp),
                vp_eq_solid_km_s=float(vp_eq) if vp_eq is not None else None,
                rho_cumulate_g_cm3=float(rho_cum), rho_incremental_g_cm3=float(rho_inc),
                rho_residual_liquid_g_cm3=float(res_rho),
                inc_ol=float(ol), inc_pl=float(pl), inc_cpx=float(cpx),
                cum_ol=float(cum_ol), cum_pl=float(cum_pl), cum_cpx=float(cum_cpx),
                fo_pct=float(fo_inc), an_pct=float(an_inc), di_pct=float(di_inc),
                cum_fo_pct=float(cum_fo), cum_an_pct=float(cum_an), cum_di_pct=float(cum_di),
            )
        )
        f_prev_out = f_target

    return out


def _simulate_crystallization_path_basalt1990(
    *,
    primary_melt_oxides_wt: Mapping[str, float],
    f_grid: np.ndarray,
    path: CrystallizationPathMode,
    p_eval_mpa: float,
    t_eval_c: float,
    cipw_backend: str,
    mineral_backend: Backend,
    cfg: PetrologyConfig,
    p_fc_mpa: float | None,
    t_fc_k: float,
    polybaric_p_high_mpa: float,
    polybaric_p_low_mpa: float,
    polybaric_dp_mpa: float,
    kd_mode_1990: KdMode1990 = "langmuir",
) -> list[CrystallizationState]:
    """BASALT.FOR (1990) STATE engine on f_grid — ``kd_mode_1990`` selects Kd formula."""
    from .basalt1990_fc import basalt1990_fc_path

    initial_wt = normalize_melt_oxides(primary_melt_oxides_wt)
    p_eval_pa = float(p_eval_mpa) * 1e6
    t_eval_k = float(t_eval_c) + 273.15
    init_fo = olivine_fo_pct(initial_wt, t_k=t_fc_k, p_mpa=p_fc_mpa or 100.0)
    init_an = plagioclase_an_pct(initial_wt, t_k=t_fc_k, p_mpa=p_fc_mpa or 100.0)
    init_di = clinopyroxene_di_pct(initial_wt, t_k=t_fc_k, p_mpa=p_fc_mpa or 100.0)

    steps = basalt1990_fc_path(
        initial_wt,
        f_grid,
        path=path,
        kd_mode=kd_mode_1990,
        t_fc_k=t_fc_k,
        p_fc_mpa=p_fc_mpa,
        polybaric_p_high_mpa=polybaric_p_high_mpa,
        polybaric_p_low_mpa=polybaric_p_low_mpa,
        polybaric_dp_mpa=polybaric_dp_mpa,
    )

    out: list[CrystallizationState] = []
    cum_mass = {"ol": 0.0, "pl": 0.0, "cpx": 0.0}
    cum_fo_num = cum_fo_den = cum_an_num = cum_an_den = cum_di_num = cum_di_den = 0.0
    f_prev = 0.0

    for step in steps:
        f_target = float(step.f_solid)
        p_fc = float(step.p_kbar) * 100.0
        melt_wt = normalize_melt_oxides(step.melt_oxides_wt)
        ol, pl, cpx = step.ol_frac, step.pl_frac, step.cpx_frac
        if float(ol + pl + cpx) <= 1e-12:
            ol, pl, cpx = _constrained_modes(
                melt_wt,
                p_mpa=p_fc,
                f_solid=f_target,
                path=path,
                t_k=step.t_k,
                polybaric_p_high_mpa=polybaric_p_high_mpa,
                polybaric_p_low_mpa=polybaric_p_low_mpa,
                polybaric_dp_mpa=polybaric_dp_mpa,
            )
        fo, an, di = step.fo_pct, step.an_pct, step.di_pct

        if f_target <= 1e-12:
            vp_inc, rho_inc = assemblage_vp_rho(
                ol_frac=1.0, pl_frac=0.0, cpx_frac=0.0, fo_pct=fo, an_pct=an, di_pct=di,
                p_pa=p_eval_pa, t_k=t_eval_k, mineral_backend=mineral_backend,
            )
            vp_inc_r = _incremental_vp_km_s(
                float(vp_inc),
                f_solid=0.0,
                path=path,
                p_eval_pa=p_eval_pa,
                t_eval_k=t_eval_k,
                mineral_backend=mineral_backend,
            )
            res_vp = _residual_norm_vp_km_s(
                initial_wt,
                f_solid=0.0,
                path=path,
                p_eval_pa=p_eval_pa,
                t_eval_k=t_eval_k,
                mineral_backend=mineral_backend,
                cipw_backend=cipw_backend,
                cfg=cfg,
            )
            out.append(
                CrystallizationState(
                    path=path, f_solid=0.0, p_fc_mpa=p_fc,
                    melt_oxides_wt=dict(initial_wt),
                    vp_cumulate_km_s=vp_inc, vp_incremental_km_s=vp_inc_r,
                    vp_residual_norm_km_s=res_vp,
                    vp_eq_solid_km_s=None,
                    rho_cumulate_g_cm3=rho_inc, rho_incremental_g_cm3=rho_inc,
                    rho_residual_liquid_g_cm3=float(liquid_density_bw1970(initial_wt, t_c=t_eval_c)),
                    inc_ol=1.0, inc_pl=0.0, inc_cpx=0.0, cum_ol=1.0, cum_pl=0.0, cum_cpx=0.0,
                    fo_pct=fo, an_pct=an, di_pct=di,
                    cum_fo_pct=fo, cum_an_pct=an, cum_di_pct=di,
                )
            )
            continue

        d_f = f_target - f_prev
        cum_mass["ol"] += d_f * ol
        cum_mass["pl"] += d_f * pl
        cum_mass["cpx"] += d_f * cpx
        if ol > 0.0:
            cum_fo_num += d_f * ol * fo
            cum_fo_den += d_f * ol
        if pl > 0.0:
            cum_an_num += d_f * pl * an
            cum_an_den += d_f * pl
        if cpx > 0.0:
            cum_di_num += d_f * cpx * di
            cum_di_den += d_f * cpx
        f_prev = f_target

        cum_total = max(sum(cum_mass.values()), 1e-9)
        cum_fo = cum_fo_num / cum_fo_den if cum_fo_den > 0.0 else fo
        cum_an = cum_an_num / cum_an_den if cum_an_den > 0.0 else an
        cum_di = cum_di_num / cum_di_den if cum_di_den > 0.0 else di

        vp_inc, rho_inc = assemblage_vp_rho(
            ol_frac=ol, pl_frac=pl, cpx_frac=cpx,
            fo_pct=fo, an_pct=an, di_pct=di,
            p_pa=p_eval_pa, t_k=t_eval_k, mineral_backend=mineral_backend,
        )
        vp_inc = _incremental_vp_km_s(
            float(vp_inc),
            f_solid=f_target,
            path=path,
            p_eval_pa=p_eval_pa,
            t_eval_k=t_eval_k,
            mineral_backend=mineral_backend,
        )
        vp_cum, rho_cum = assemblage_vp_rho(
            ol_frac=cum_mass["ol"] / cum_total,
            pl_frac=cum_mass["pl"] / cum_total,
            cpx_frac=cum_mass["cpx"] / cum_total,
            fo_pct=cum_fo, an_pct=cum_an, di_pct=cum_di,
            p_pa=p_eval_pa, t_k=t_eval_k, mineral_backend=mineral_backend,
        )
        res_vp = _residual_norm_vp_km_s(
            melt_wt,
            f_solid=f_target,
            path=path,
            p_eval_pa=p_eval_pa,
            t_eval_k=t_eval_k,
            mineral_backend=mineral_backend,
            cipw_backend=cipw_backend,
            cfg=cfg,
        )
        out.append(
            CrystallizationState(
                path=path, f_solid=f_target, p_fc_mpa=p_fc,
                melt_oxides_wt={k: float(melt_wt[k]) for k in _OXIDE_KEYS},
                vp_cumulate_km_s=float(vp_cum), vp_incremental_km_s=float(vp_inc),
                vp_residual_norm_km_s=res_vp,
                vp_eq_solid_km_s=None,
                rho_cumulate_g_cm3=float(rho_cum), rho_incremental_g_cm3=float(rho_inc),
                rho_residual_liquid_g_cm3=float(liquid_density_bw1970(melt_wt, t_c=t_eval_c)),
                inc_ol=float(ol), inc_pl=float(pl), inc_cpx=float(cpx),
                cum_ol=float(cum_mass["ol"] / cum_total),
                cum_pl=float(cum_mass["pl"] / cum_total),
                cum_cpx=float(cum_mass["cpx"] / cum_total),
                fo_pct=float(fo), an_pct=float(an), di_pct=float(di),
                cum_fo_pct=float(cum_fo), cum_an_pct=float(cum_an), cum_di_pct=float(cum_di),
            )
        )

    return out


def simulate_crystallization_path(
    *,
    primary_melt_oxides_wt: Mapping[str, float],
    f_grid: np.ndarray,
    path: CrystallizationPathMode,
    p_eval_mpa: float = 100.0,
    t_eval_c: float = 100.0,
    cipw_backend: str | None = None,
    mineral_backend: Backend | None = None,
    config: PetrologyConfig | None = None,
    sub_steps_per_interval: int = 50,
    p_fc_mpa: float | None = None,
    t_fc_k: float = DEFAULT_T_K,
    polybaric_p_high_mpa: float = 800.0,
    polybaric_p_low_mpa: float = 100.0,
    polybaric_dp_mpa: float = 50.0,
    use_state_engine: bool = False,
    state_fc_substeps: int = 40,
    fig2_ab_calibrate: bool = False,
    fig2_ab_anchor_vp: bool = False,
    kd_engine: KdEngine = "heuristic",
    kd_mode_1990: KdMode1990 = "langmuir",
) -> list[CrystallizationState]:
    """W&L crystallization on Pyrolite + BurnMan (KKHS02 §2.1 chain).

    ``p_fc_mpa`` fixes crystallization pressure (100/400/800 MPa isobaric FC).
    When omitted, ``fc_100`` uses 100 MPa and ``polybaric_fc`` depressurizes 800→100 MPa.

    ``fig2_ab_calibrate=True``: drive phase modes from KKHS02 Fig.2c–h anchors + FC mass balance.
    ``fig2_ab_anchor_vp=True``: also interpolate sparse Fig.2(a,b) Vp/ρ anchors (layout only).
    Default (calibrate on, anchor off): paper phases + HS/CIPW physics (Korenaga §2.1).

    ``kd_engine``:
      - ``heuristic`` — partition + mass balance (default)
      - ``langmuir`` — BASALT+langmuir STATE (same as ``use_state_engine=True``)
      - ``basalt1990`` — original BASALT.FOR STATE; ``kd_mode_1990`` is ``1990`` or ``langmuir``

    ``use_state_engine=True`` is deprecated alias for ``kd_engine='langmuir'``.
    """
    cfg = config or DEFAULT_CONFIG
    cipw_backend = cipw_backend or cfg.cipw_backend
    mineral_backend = mineral_backend or cfg.mineral_backend

    if use_state_engine and kd_engine == "heuristic":
        kd_engine = "langmuir"

    f_grid = np.asarray(f_grid, dtype=float)
    if f_grid.ndim != 1 or len(f_grid) < 2:
        raise ValueError("f_grid must be a 1-D array with >= 2 points")
    if np.any(np.diff(f_grid) <= 0):
        raise ValueError("f_grid must be strictly increasing")

    if fig2_ab_calibrate:
        return _simulate_fig2_ab(
            primary_melt_oxides_wt=primary_melt_oxides_wt,
            f_grid=f_grid,
            path=path,
            p_eval_mpa=p_eval_mpa,
            t_eval_c=t_eval_c,
            cipw_backend=cipw_backend,
            mineral_backend=mineral_backend,
            cfg=cfg,
            sub_steps_per_interval=sub_steps_per_interval,
            p_fc_mpa=p_fc_mpa,
            t_fc_k=t_fc_k,
            polybaric_p_high_mpa=polybaric_p_high_mpa,
            polybaric_p_low_mpa=polybaric_p_low_mpa,
            polybaric_dp_mpa=polybaric_dp_mpa,
            fig2_ab_anchor_vp=fig2_ab_anchor_vp,
        )

    if kd_engine == "basalt1990" and path != "eq_100":
        return _simulate_crystallization_path_basalt1990(
            primary_melt_oxides_wt=primary_melt_oxides_wt,
            f_grid=f_grid,
            path=path,
            p_eval_mpa=p_eval_mpa,
            t_eval_c=t_eval_c,
            cipw_backend=cipw_backend,
            mineral_backend=mineral_backend,
            cfg=cfg,
            p_fc_mpa=p_fc_mpa,
            t_fc_k=t_fc_k,
            polybaric_p_high_mpa=polybaric_p_high_mpa,
            polybaric_p_low_mpa=polybaric_p_low_mpa,
            polybaric_dp_mpa=polybaric_dp_mpa,
            kd_mode_1990=kd_mode_1990,
        )

    if kd_engine == "langmuir":
        return _simulate_crystallization_path_state(
            primary_melt_oxides_wt=primary_melt_oxides_wt,
            f_grid=f_grid,
            path=path,
            p_eval_mpa=p_eval_mpa,
            t_eval_c=t_eval_c,
            cipw_backend=cipw_backend,
            mineral_backend=mineral_backend,
            cfg=cfg,
            p_fc_mpa=p_fc_mpa,
            t_fc_k=t_fc_k,
            polybaric_p_high_mpa=polybaric_p_high_mpa,
            polybaric_p_low_mpa=polybaric_p_low_mpa,
            polybaric_dp_mpa=polybaric_dp_mpa,
            state_fc_substeps=state_fc_substeps,
            sub_steps_per_interval=sub_steps_per_interval,
        )

    if path == "polybaric_fc":
        return _simulate_polybaric_heuristic(
            primary_melt_oxides_wt=primary_melt_oxides_wt,
            f_grid=f_grid,
            p_eval_mpa=p_eval_mpa,
            t_eval_c=t_eval_c,
            cipw_backend=cipw_backend,
            mineral_backend=mineral_backend,
            cfg=cfg,
            sub_steps_per_interval=sub_steps_per_interval,
            t_fc_k=t_fc_k,
            polybaric_p_high_mpa=polybaric_p_high_mpa,
            polybaric_p_low_mpa=polybaric_p_low_mpa,
            polybaric_dp_mpa=polybaric_dp_mpa,
        )

    p_eval_pa = float(p_eval_mpa) * 1e6
    t_eval_k = float(t_eval_c) + 273.15

    initial_wt = normalize_melt_oxides(primary_melt_oxides_wt)
    initial_g = _grams_from_wt(initial_wt)
    melt_wt = dict(initial_wt)
    solid_total_g = {k: 0.0 for k in _OXIDE_KEYS}

    f_curr = 0.0
    cum_mass = {"ol": 0.0, "pl": 0.0, "cpx": 0.0}
    cum_fo_num = cum_fo_den = cum_an_num = cum_an_den = cum_di_num = cum_di_den = 0.0
    init_fo = olivine_fo_pct(initial_wt, t_k=t_fc_k, p_mpa=p_fc_mpa or 100.0)
    init_an = plagioclase_an_pct(initial_wt, t_k=t_fc_k, p_mpa=p_fc_mpa or 100.0)
    init_di = clinopyroxene_di_pct(initial_wt, t_k=t_fc_k, p_mpa=p_fc_mpa or 100.0)

    out: list[CrystallizationState] = []
    last_inc = (1.0, 0.0, 0.0, init_fo, init_an, init_di)
    high_al_regime = (
        HighAlFcRegime(
            enabled=True,
            high_al_morb=True,
            primary_al2o3=float(initial_wt.get("Al2O3", 0.0)),
        )
        if is_high_al_morb(initial_wt)
        else None
    )

    for f_target in f_grid:
        f_target = float(f_target)
        p_fc = _crystallization_pressure(
            path,
            f_target,
            p_fc_fixed_mpa=p_fc_mpa,
            polybaric_p_high_mpa=polybaric_p_high_mpa,
            polybaric_p_low_mpa=polybaric_p_low_mpa,
            polybaric_dp_mpa=polybaric_dp_mpa,
        )

        if f_target <= f_curr + 1e-12:
            fo, an, di = init_fo, init_an, init_di
            ol, pl, cpx = 1.0, 0.0, 0.0
            vp_inc, rho_inc = assemblage_vp_rho(
                ol_frac=ol, pl_frac=pl, cpx_frac=cpx, fo_pct=fo, an_pct=an, di_pct=di,
                p_pa=p_eval_pa, t_k=t_eval_k, mineral_backend=mineral_backend,
            )
            vp_inc_r = _incremental_vp_km_s(
                float(vp_inc),
                f_solid=0.0,
                path=path,
                p_eval_pa=p_eval_pa,
                t_eval_k=t_eval_k,
                mineral_backend=mineral_backend,
            )
            res_vp = _residual_norm_vp_km_s(
                initial_wt,
                f_solid=0.0,
                path=path,
                p_eval_pa=p_eval_pa,
                t_eval_k=t_eval_k,
                mineral_backend=mineral_backend,
                cipw_backend=cipw_backend,
                cfg=cfg,
            )
            out.append(
                CrystallizationState(
                    path=path, f_solid=0.0, p_fc_mpa=p_fc,
                    melt_oxides_wt=dict(initial_wt),
                    vp_cumulate_km_s=vp_inc, vp_incremental_km_s=vp_inc_r,
                    vp_residual_norm_km_s=res_vp,
                    vp_eq_solid_km_s=vp_inc if path == "eq_100" else None,
                    rho_cumulate_g_cm3=rho_inc, rho_incremental_g_cm3=rho_inc,
                    rho_residual_liquid_g_cm3=float(liquid_density_bw1970(initial_wt, t_c=t_eval_c)),
                    inc_ol=ol, inc_pl=pl, inc_cpx=cpx, cum_ol=ol, cum_pl=pl, cum_cpx=cpx,
                    fo_pct=fo, an_pct=an, di_pct=di,
                    cum_fo_pct=fo, cum_an_pct=an, cum_di_pct=di,
                )
            )
            continue

        d_f_step = (f_target - f_curr) / sub_steps_per_interval

        for istep in range(sub_steps_per_interval):
            f_next = f_curr + d_f_step * (istep + 1)
            p_here = _crystallization_pressure(
                path,
                f_next,
                p_fc_fixed_mpa=p_fc_mpa,
                polybaric_p_high_mpa=polybaric_p_high_mpa,
                polybaric_p_low_mpa=polybaric_p_low_mpa,
                polybaric_dp_mpa=polybaric_dp_mpa,
            )
            ol, pl, cpx = _constrained_modes(
                melt_wt,
                p_mpa=p_here,
                f_solid=f_next,
                path=path,
                t_k=t_fc_k,
                polybaric_p_high_mpa=polybaric_p_high_mpa,
                polybaric_p_low_mpa=polybaric_p_low_mpa,
                polybaric_dp_mpa=polybaric_dp_mpa,
                high_al_regime=high_al_regime,
            )
            fo = olivine_fo_pct(melt_wt, t_k=t_fc_k, p_mpa=p_here)
            an = plagioclase_an_pct(melt_wt, t_k=t_fc_k, p_mpa=p_here)
            di = clinopyroxene_di_pct(melt_wt, t_k=t_fc_k, p_mpa=p_here)
            solid_wt = _increment_solid_wt(ol, pl, cpx, fo_pct=fo, an_pct=an, di_pct=di)
            last_inc = (ol, pl, cpx, fo, an, di)

            if path == "eq_100":
                solid_total_g = {
                    k: solid_total_g[k] + d_f_step * _SYSTEM_G * solid_wt[k] / 100.0
                    for k in _OXIDE_KEYS
                }
                melt_wt = _wt_from_grams(_eq_liquid_from_solid(initial_g, solid_total_g, f_next))
            else:
                melt_wt = _update_melt_fc(melt_wt, f_curr + d_f_step * istep, d_f_step, solid_wt)

            cum_mass["ol"] += d_f_step * ol
            cum_mass["pl"] += d_f_step * pl
            cum_mass["cpx"] += d_f_step * cpx
            if ol > 0.0:
                cum_fo_num += d_f_step * ol * fo
                cum_fo_den += d_f_step * ol
            if pl > 0.0:
                cum_an_num += d_f_step * pl * an
                cum_an_den += d_f_step * pl
            if cpx > 0.0:
                cum_di_num += d_f_step * cpx * di
                cum_di_den += d_f_step * cpx

        f_curr = f_target
        ol, pl, cpx, fo_inc, an_inc, di_inc = last_inc
        cum_fo = cum_fo_num / cum_fo_den if cum_fo_den > 0.0 else init_fo
        cum_an = cum_an_num / cum_an_den if cum_an_den > 0.0 else init_an
        cum_di = cum_di_num / cum_di_den if cum_di_den > 0.0 else init_di

        vp_inc, rho_inc = assemblage_vp_rho(
            ol_frac=ol, pl_frac=pl, cpx_frac=cpx,
            fo_pct=fo_inc, an_pct=an_inc, di_pct=di_inc,
            p_pa=p_eval_pa, t_k=t_eval_k, mineral_backend=mineral_backend,
        )
        vp_inc = _incremental_vp_km_s(
            float(vp_inc),
            f_solid=f_target,
            path=path,
            p_eval_pa=p_eval_pa,
            t_eval_k=t_eval_k,
            mineral_backend=mineral_backend,
        )
        cum_total = max(sum(cum_mass.values()), 1e-9)
        vp_cum, rho_cum = assemblage_vp_rho(
            ol_frac=cum_mass["ol"] / cum_total,
            pl_frac=cum_mass["pl"] / cum_total,
            cpx_frac=cum_mass["cpx"] / cum_total,
            fo_pct=cum_fo, an_pct=cum_an, di_pct=cum_di,
            p_pa=p_eval_pa, t_k=t_eval_k, mineral_backend=mineral_backend,
        )
        res_vp = _residual_norm_vp_km_s(
            melt_wt,
            f_solid=f_target,
            path=path,
            p_eval_pa=p_eval_pa,
            t_eval_k=t_eval_k,
            mineral_backend=mineral_backend,
            cipw_backend=cipw_backend,
            cfg=cfg,
        )

        vp_eq_solid = None
        if path == "eq_100" and f_target > 0.0:
            avg_wt = _wt_from_grams({k: solid_total_g[k] / f_target for k in _OXIDE_KEYS})
            o2, p2, c2 = _constrained_modes(
                avg_wt,
                p_mpa=p_fc,
                f_solid=f_target,
                path=path,
                t_k=t_fc_k,
                polybaric_p_high_mpa=polybaric_p_high_mpa,
                polybaric_p_low_mpa=polybaric_p_low_mpa,
                polybaric_dp_mpa=polybaric_dp_mpa,
            )
            vp_eq_solid, _ = assemblage_vp_rho(
                ol_frac=o2, pl_frac=p2, cpx_frac=c2,
                fo_pct=olivine_fo_pct(avg_wt, t_k=t_fc_k, p_mpa=p_fc),
                an_pct=plagioclase_an_pct(avg_wt, t_k=t_fc_k, p_mpa=p_fc),
                di_pct=clinopyroxene_di_pct(avg_wt, t_k=t_fc_k, p_mpa=p_fc),
                p_pa=p_eval_pa, t_k=t_eval_k, mineral_backend=mineral_backend,
            )

        out.append(
            CrystallizationState(
                path=path, f_solid=f_target, p_fc_mpa=p_fc,
                melt_oxides_wt={k: float(melt_wt[k]) for k in _OXIDE_KEYS},
                vp_cumulate_km_s=float(vp_cum), vp_incremental_km_s=float(vp_inc),
                vp_residual_norm_km_s=res_vp,
                vp_eq_solid_km_s=float(vp_eq_solid) if vp_eq_solid is not None else None,
                rho_cumulate_g_cm3=float(rho_cum), rho_incremental_g_cm3=float(rho_inc),
                rho_residual_liquid_g_cm3=float(liquid_density_bw1970(melt_wt, t_c=t_eval_c)),
                inc_ol=float(ol), inc_pl=float(pl), inc_cpx=float(cpx),
                cum_ol=float(cum_mass["ol"] / cum_total),
                cum_pl=float(cum_mass["pl"] / cum_total),
                cum_cpx=float(cum_mass["cpx"] / cum_total),
                fo_pct=float(fo_inc), an_pct=float(an_inc), di_pct=float(di_inc),
                cum_fo_pct=float(cum_fo), cum_an_pct=float(cum_an), cum_di_pct=float(cum_di),
            )
        )

    return out
