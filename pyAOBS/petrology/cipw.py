"""
CIPW normative minerals for mafic bulk compositions (KKHS02 path A).

Uses ``pyrolite.mineral.normative.CIPW_norm`` (vendored under ``petrology/pyrolite``
or pip) when available; otherwise a compact dry-basalt fallback.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .vendored import ensure_vendored, pyrolite_available

_MW = {
    "SiO2": 60.084,
    "TiO2": 79.866,
    "Al2O3": 101.961,
    "FeO": 71.844,
    "Fe2O3": 159.688,
    "MgO": 40.304,
    "CaO": 56.079,
    "Na2O": 61.979,
    "K2O": 94.196,
    "P2O5": 141.944,
    "Cr2O3": 151.990,
    "MnO": 70.937,
    "H2O": 18.015,
}

_NORM_MW = {
    "olivine": 140.693,
    "plagioclase": 270.0,
    "clinopyroxene": 216.55,
    "quartz": 60.084,
    "ilmenite": 151.710,
}

_OXIDE_KEYS = (
    "SiO2",
    "TiO2",
    "Al2O3",
    "Fe2O3",
    "FeO",
    "MnO",
    "MgO",
    "CaO",
    "Na2O",
    "K2O",
    "P2O5",
    "Cr2O3",
    "H2O",
)

# pyrolite CIPW_norm returns **full mineral names** (wt% normative mass per row)
_PYROLITE_OLIVINE = ("olivine", "forsterite", "fayalite")
_PYROLITE_PLAG = ("anorthite", "albite", "orthoclase", "nepheline", "leucite")
_PYROLITE_CPX = (
    "diopside",
    "hypersthene",
    "enstatite",
    "ferrosilite",
    "clinoenstatite",
    "clinoferrosilite",
    "wollastonite",
)


def _normalize_oxides(oxides_wt: Mapping[str, float]) -> dict[str, float]:
    vals = {k: float(oxides_wt.get(k, 0.0) or 0.0) for k in _OXIDE_KEYS}
    s = sum(vals.values())
    if s <= 0:
        raise ValueError("oxide sum must be positive")
    return {k: v / s * 100.0 for k, v in vals.items()}


def _moles(ox: dict[str, float]) -> dict[str, float]:
    return {k: ox[k] / _MW[k] for k in ox}


def _cipw_fallback(oxides_wt: Mapping[str, float]) -> dict[str, float]:
    """Return mass fractions (0–1) of olivine, plagioclase, clinopyroxene, quartz, ilmenite."""
    ox = _normalize_oxides(oxides_wt)
    n = _moles(ox)

    si = n["SiO2"]
    ti = n["TiO2"]
    al = n["Al2O3"]
    fe = n["FeO"]
    mg = n["MgO"]
    ca = n["CaO"]
    na = n["Na2O"]
    k = n["K2O"]
    p = n["P2O5"]

    ap = min(p / 2.0, ca / 3.0) if p > 0 else 0.0
    ca -= 3.0 * ap
    p -= 2.0 * ap

    il = min(ti, fe)
    ti -= il
    fe -= il

    or_ = min(k, al, si / 3.0) if k > 0 else 0.0
    k -= or_
    al -= or_
    si -= 3.0 * or_

    ab = min(na, al, si / 3.0) if na > 0 else 0.0
    na -= ab
    al -= ab
    si -= 3.0 * ab

    an = min(ca, al / 2.0, si / 2.0) if ca > 0 and al > 0 else 0.0
    ca -= an
    al -= 2.0 * an
    si -= 2.0 * an

    di = min(ca, mg, si / 2.0) if ca > 0 and mg > 0 else 0.0
    ca -= di
    mg -= di
    si -= 2.0 * di

    hy_mg = min(mg, si) if mg > 0 else 0.0
    mg -= hy_mg
    si -= hy_mg
    hy_fe = min(fe, si) if fe > 0 else 0.0
    fe -= hy_fe
    si -= hy_fe

    ol_mg = min(mg, si / 2.0) if mg > 0 else 0.0
    mg -= ol_mg
    si -= 2.0 * ol_mg
    ol_fe = min(fe, si / 2.0) if fe > 0 else 0.0
    fe -= ol_fe
    si -= 2.0 * ol_fe

    qz = max(si, 0.0)

    mass = {
        "olivine": (ol_mg + ol_fe) * _NORM_MW["olivine"],
        "plagioclase": (ab + an + or_) * _NORM_MW["plagioclase"],
        "clinopyroxene": (di + hy_mg + hy_fe) * _NORM_MW["clinopyroxene"],
        "quartz": qz * _NORM_MW["quartz"],
        "ilmenite": il * _NORM_MW["ilmenite"],
    }
    total = sum(mass.values())
    if total <= 0:
        raise ValueError("CIPW produced zero normative mass")
    frac = {k: v / total for k, v in mass.items() if v > 0}

    ol_fe_t = ol_fe + hy_fe * 0.5
    ol_mg_t = ol_mg + hy_mg * 0.5 + di * 0.5
    fo = ol_mg_t / max(ol_mg_t + ol_fe_t, 1e-12)
    an_frac = an / max(ab + an + or_, 1e-12)
    cpx_mg = (di + hy_mg) / max(di + hy_mg + hy_fe, 1e-12)

    frac["Fo"] = float(np.clip(fo, 0.05, 0.99))
    frac["An"] = float(np.clip(an_frac, 0.0, 0.99))
    frac["CpxMg"] = float(np.clip(cpx_mg, 0.05, 0.99))
    return frac


def _sum_cols(row: Mapping[str, float], names: tuple[str, ...]) -> float:
    return float(sum(float(row.get(n, 0.0) or 0.0) for n in names))


def _cipw_pyrolite(oxides_wt: Mapping[str, float], *, prefer_pip: bool = True) -> dict[str, float]:
    import pandas as pd
    from pyrolite.mineral.normative import CIPW_norm

    row = {k: float(oxides_wt.get(k, 0.0) or 0.0) for k in _OXIDE_KEYS}
    df = pd.DataFrame([row])
    norm = CIPW_norm(df, adjust_all_Fe=True)
    cols = norm.iloc[0].to_dict()

    # Prefer grouped olivine / diopside / hypersthene columns when present
    ol = _sum_cols(cols, ("olivine",)) or _sum_cols(cols, ("forsterite", "fayalite"))
    pl = _sum_cols(cols, _PYROLITE_PLAG)
    cpx = _sum_cols(cols, _PYROLITE_CPX)
    qz = _sum_cols(cols, ("quartz",))
    il = _sum_cols(cols, ("ilmenite",))

    total = ol + pl + cpx + qz + il
    if total <= 0:
        raise ValueError("pyrolite CIPW produced zero normative mass")

    fo_num = _sum_cols(cols, ("forsterite",))
    fo_den = fo_num + _sum_cols(cols, ("fayalite",))
    # Some pyrolite outputs only grouped olivine without Fo/Fa split.
    # Fall back to melt Mg-Fe proxy regardless of olivine abundance keying.
    if fo_den <= 0:
        fo_num = float(oxides_wt.get("MgO", 0.0) or 0.0)
        fo_den = fo_num + float(oxides_wt.get("FeO", 0.0) or 0.0)

    an_m = _sum_cols(cols, ("anorthite",))
    pl_m = _sum_cols(cols, _PYROLITE_PLAG)

    mg_cpx = _sum_cols(cols, ("clinoenstatite", "enstatite", "diopside"))
    fe_cpx = _sum_cols(cols, ("clinoferrosilite", "ferrosilite"))
    if (mg_cpx + fe_cpx) <= 0:
        # Keep cpx Mg# consistent with olivine proxy when pyrolite does not
        # report cpx Fe/Mg split in this output schema.
        mg_cpx = fo_num
        fe_cpx = max(fo_den - fo_num, 0.0)

    out = {
        "olivine": ol / total,
        "plagioclase": pl / total,
        "clinopyroxene": cpx / total,
        "quartz": qz / total,
        "ilmenite": il / total,
        "Fo": float(np.clip(fo_num / max(fo_den, 1e-12), 0.05, 0.99)),
        "An": float(np.clip(an_m / max(pl_m, 1e-12), 0.0, 0.99)),
        "CpxMg": float(np.clip(mg_cpx / max(mg_cpx + fe_cpx, 1e-12), 0.05, 0.99)),
    }
    return {k: v for k, v in out.items() if v > 1e-6 or k in ("Fo", "An", "CpxMg")}


def cipw_norm_mass_fractions(
    oxides_wt: Mapping[str, float],
    backend: str = "auto",
    *,
    prefer_pip: bool = True,
) -> dict[str, float]:
    """
    Normative mineral mass fractions and Fo / An / CpxMg metadata.

    Phase keys: olivine, plagioclase, clinopyroxene, quartz, ilmenite (subset may be absent).
    """
    if backend == "fallback":
        return _cipw_fallback(oxides_wt)

    if backend in ("auto", "pyrolite"):
        if pyrolite_available(prefer_pip=prefer_pip):
            ensure_vendored("pyrolite", prefer_pip=prefer_pip)
            try:
                return _cipw_pyrolite(oxides_wt, prefer_pip=prefer_pip)
            except Exception:
                if backend == "pyrolite":
                    raise
        elif backend == "pyrolite":
            raise ImportError("pyrolite not available (vendored or pip)")

    return _cipw_fallback(oxides_wt)
