"""
pMELTS KLB-1 melt major elements from pyMelt ``klb1_pmelts_grid.csv``.

Modern-track chemistry: interpolate liquid wt% oxides at fixed (P, F) using
``liq_mass`` as melt fraction (Ghiorso et al., 2002 grid shipped with pyMelt).

Reproduction track continues to use ``kinzler1997_batch.batch_melt_oxides``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Mapping

import numpy as np

ChemistryBackend = Literal["kinzler1997", "pmelts_klb1"]

_OXIDE_KEYS = (
    "SiO2",
    "TiO2",
    "Al2O3",
    "Cr2O3",
    "Fe2O3",
    "FeO",
    "MgO",
    "CaO",
    "Na2O",
    "K2O",
    "P2O5",
    "H2O",
)

# pyMELTS liquid wt% columns → standard oxide keys
_PMELTS_LIQ_MAP: dict[str, str] = {
    "SiO2": "liq_SiO2_wtpt",
    "TiO2": "liq_TiO2_wtpt",
    "Al2O3": "liq_Al2O3_wtpt",
    "Fe2O3": "liq_Fe2O3_wtpt",
    "FeO": "liq_FeO_wtpt",
    "MgO": "liq_MgO_wtpt",
    "CaO": "liq_CaO_wtpt",
    "Na2O": "liq_Na2O_wtpt",
}


def pmelts_klb1_grid_path() -> Path:
    """Locate pyMelt KLB-1 pMELTS grid (package install or vendored copy)."""
    try:
        import pyMelt

        bundled = Path(pyMelt.__file__).resolve().parent / "phaseDiagrams" / "build" / "klb1_pmelts_grid.csv"
        if bundled.is_file():
            return bundled
    except ImportError:
        pass
    local = Path(__file__).resolve().parents[1] / "data" / "pmelts" / "klb1_pmelts_grid.csv"
    if local.is_file():
        return local
    raise FileNotFoundError(
        "klb1_pmelts_grid.csv not found (install pyMelt or place under petrology/data/pmelts/)"
    )


def _normalize_oxides(ox: Mapping[str, float]) -> dict[str, float]:
    vals = {k: float(max(0.0, ox.get(k, 0.0))) for k in _OXIDE_KEYS}
    s = sum(vals.values())
    if s <= 0.0:
        raise ValueError("empty melt oxide sum from pMELTS lookup")
    return {k: 100.0 * v / s for k, v in vals.items()}


@lru_cache(maxsize=1)
def _load_pmelts_slices() -> tuple[np.ndarray, dict[float, dict[str, np.ndarray]]]:
    import pandas as pd

    path = pmelts_klb1_grid_path()
    df = pd.read_csv(path)
    if "liq_mass" not in df.columns:
        raise ValueError(f"{path} missing liq_mass column")

    melt = df[df["liq_mass"] > 1e-6].copy()
    melt = melt[melt["liq_SiO2_wtpt"].notna()]
    pressures = np.sort(melt["pressure"].unique())

    slices: dict[float, dict[str, np.ndarray]] = {}
    for p in pressures:
        sub = melt[np.isclose(melt["pressure"], p)].sort_values("liq_mass")
        f_arr = sub["liq_mass"].to_numpy(dtype=float)
        if len(f_arr) < 2:
            continue
        ox: dict[str, np.ndarray] = {}
        for key, col in _PMELTS_LIQ_MAP.items():
            if col not in sub.columns:
                continue
            ox[key] = sub[col].to_numpy(dtype=float)
        slices[float(p)] = {"f": f_arr, **ox}

    if len(slices) < 2:
        raise ValueError(f"insufficient pMELTS grid slices in {path}")
    return pressures, slices


def _interp_oxides_at_p(
    slices: dict[float, dict[str, np.ndarray]],
    p_gpa: float,
    f: float,
) -> dict[str, float]:
    data = slices[p_gpa]
    f_grid = data["f"]
    f_use = float(np.clip(f, float(f_grid[0]), float(f_grid[-1])))
    out: dict[str, float] = {}
    for key in _PMELTS_LIQ_MAP:
        arr = data.get(key)
        if arr is None:
            out[key] = 0.0
            continue
        out[key] = float(np.interp(f_use, f_grid, arr))
    out["Cr2O3"] = 0.0
    out["K2O"] = 0.0
    out["P2O5"] = 0.0
    out["H2O"] = 0.0
    return _normalize_oxides(out)


def pmelts_melt_oxides(
    p_gpa: float,
    f: float,
    *,
    f_min: float = 0.02,
) -> dict[str, float]:
    """
    Instantaneous KLB-1 melt composition (wt%) at isobaric melt fraction *F*.

    Bilinear interpolation in (P, F) on the pyMelt pMELTS grid.
    """
    p = float(p_gpa)
    f_use = float(max(f, f_min))
    pressures, slices = _load_pmelts_slices()
    p_keys = np.array(sorted(slices.keys()), dtype=float)

    if p <= p_keys[0]:
        return _interp_oxides_at_p(slices, float(p_keys[0]), f_use)
    if p >= p_keys[-1]:
        return _interp_oxides_at_p(slices, float(p_keys[-1]), f_use)

    hi = int(np.searchsorted(p_keys, p, side="right"))
    lo = hi - 1
    p_lo, p_hi = float(p_keys[lo]), float(p_keys[hi])
    if abs(p_hi - p_lo) < 1e-9:
        return _interp_oxides_at_p(slices, p_lo, f_use)

    w = (p - p_lo) / (p_hi - p_lo)
    ox_lo = _interp_oxides_at_p(slices, p_lo, f_use)
    ox_hi = _interp_oxides_at_p(slices, p_hi, f_use)
    blended = {k: (1.0 - w) * ox_lo[k] + w * ox_hi[k] for k in _OXIDE_KEYS}
    return _normalize_oxides(blended)


def melt_oxides(
    p_gpa: float,
    f: float,
    *,
    chemistry_backend: ChemistryBackend = "kinzler1997",
    source: Mapping[str, float] | None = None,
    f_min: float = 0.02,
) -> dict[str, float]:
    """Dispatch melt major elements by chemistry backend."""
    backend = str(chemistry_backend)
    if backend == "pmelts_klb1":
        return pmelts_melt_oxides(p_gpa, f, f_min=f_min)
    if backend == "kinzler1997":
        from .kinzler1997_batch import batch_melt_oxides

        return batch_melt_oxides(p_gpa, max(float(f), f_min), source=source)
    raise ValueError(f"unknown chemistry_backend: {chemistry_backend!r}")


def default_chemistry_for_lithology(
    *,
    family: str,
    lithology_backend: str = "native",
) -> ChemistryBackend:
    """Modern pymelt peridotite → pMELTS KLB-1; else Kinzler (1997)."""
    if lithology_backend == "pymelt" and family == "peridotite":
        return "pmelts_klb1"
    return "kinzler1997"
