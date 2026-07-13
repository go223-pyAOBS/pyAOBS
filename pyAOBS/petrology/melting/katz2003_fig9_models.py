"""
Dry melting models for Katz et al. (2003) Fig. 9 comparison.

Default plot: isobaric F(T) at 1 GPa (blue) and 3 GPa (green) for models with
solid Table 2 / pyMelt / grid implementations:

  - Katz et al. (2003) — ``katz2003.py`` Table 2
  - McKenzie & Bickle (1988) — pyMelt ``mckenzie.lherzolite``
  - pMELTS (Ghiorso et al., 2002) — CSV grid subset

Langmuir (1992) and Iwamori (1995) are deferred (no pyMelt lithology; paper
recast pending literature check).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

_FIG9_DATA = Path(__file__).resolve().parents[1] / "data" / "katz2003_fig9"

# Fig. 9: blue = 1 GPa, green = 3 GPa (parameterizations); pMELTS uses pink.
FIG9_PRESSURE_COLORS: dict[float, str] = {1.0: "#2471a3", 3.0: "#27ae60"}
FIG9_PMELTS_COLOR = "#e056fd"

# Default Fig. 9 — Katz (2003) paper caption (Langmuir / Iwamori deferred).
FIG9_PAPER_MODEL_ORDER: tuple[str, ...] = (
    "mckenzie1988",
    "pmelts2002",
)

# Extended overlay: Katz Table 2 native + paper models.
FIG9_EXTENDED_MODEL_ORDER: tuple[str, ...] = (
    "katz2003",
    "mckenzie1988",
    "pmelts2002",
)

FIG9_MODEL_ORDER: tuple[str, ...] = FIG9_PAPER_MODEL_ORDER

FIG9_MODEL_PLOT_STYLE: dict[str, dict[str, object]] = {
    "katz2003": {
        "label": "Katz (2003)",
        "linestyle": "-",
        "linewidth": 2.6,
        "pressure_colors": True,
    },
    "mckenzie1988": {
        "label": "McKenzie & Bickle (1988)",
        "linestyle": (0, (9, 5)),
        "linewidth": 2.6,
        "pressure_colors": True,
    },
    "pmelts2002": {
        "label": "pMELTS (Ghiorso et al., 2002)",
        "linestyle": (0, (9, 5)),
        "linewidth": 2.6,
        "pressure_colors": False,
        "color": FIG9_PMELTS_COLOR,
    },
}


def fig9_curve_style(model_key: str, p_gpa: float) -> dict[str, object]:
    """Matplotlib plot kwargs for one Fig. 9 curve (model × pressure)."""
    sty = FIG9_MODEL_PLOT_STYLE.get(model_key, {})
    ls = sty.get("linestyle", "-")
    lw = float(sty.get("linewidth", 2.0))
    if sty.get("pressure_colors", True):
        col = FIG9_PRESSURE_COLORS.get(float(p_gpa), "#333333")
    else:
        col = sty.get("color", FIG9_PMELTS_COLOR)
    return {"color": col, "ls": ls, "lw": lw}


def fig9_model_label(model_key: str, p_gpa: float) -> str:
    sty = FIG9_MODEL_PLOT_STYLE.get(model_key, {})
    base = str(sty.get("label", model_key))
    return f"{base} ({p_gpa:g} GPa)"


class DryMeltModel(Protocol):
    key: str
    label: str
    verified: str

    def melt_fraction(self, p_gpa: float, t_c: float) -> float: ...

    def melt_fraction_profile(
        self, p_gpa: float, t_c: np.ndarray | list[float]
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class _CallableDryModel:
    key: str
    label: str
    verified: str
    _f: Callable[[float, float], float]

    def melt_fraction(self, p_gpa: float, t_c: float) -> float:
        return float(np.clip(self._f(float(p_gpa), float(t_c)), 0.0, 1.0))

    def melt_fraction_profile(
        self, p_gpa: float, t_c: np.ndarray | list[float]
    ) -> np.ndarray:
        return np.array([self.melt_fraction(p_gpa, float(ti)) for ti in t_c], dtype=float)


@lru_cache(maxsize=1)
def _pmelts_tables() -> dict[float, tuple[np.ndarray, np.ndarray]]:
    import pandas as pd

    csv_path = _FIG9_DATA / "pmelts_klb1_p1_p3.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing pMELTS subset: {csv_path}")
    df = pd.read_csv(csv_path)
    out: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for p in (1.0, 3.0):
        sub = df[np.isclose(df["pressure"], p)].sort_values("temperature")
        t = sub["temperature"].to_numpy(dtype=float)
        f = sub["f"].to_numpy(dtype=float)
        if len(t) < 2:
            raise ValueError(f"pMELTS grid has fewer than 2 rows at P={p} GPa")
        out[p] = (t, f)
    return out


def pmelts2002_melt_fraction(p_gpa: float, t_c: float) -> float:
    """pMELTS KLB-1 ``liq_mass`` vs T at fixed P (Ghiorso et al., 2002 grid)."""
    tables = _pmelts_tables()
    p = float(p_gpa)
    key = min(tables.keys(), key=lambda pk: abs(pk - p))
    if abs(key - p) > 0.05:
        raise ValueError(f"pMELTS table only has P=1 and 3 GPa; got P={p}")
    t_grid, f_grid = tables[key]
    t = float(t_c)
    if t <= float(t_grid[0]):
        return 0.0
    if t >= float(t_grid[-1]):
        return 1.0
    return float(np.interp(t, t_grid, f_grid))


def build_fig9_dry_melt_models(*, cpx_mass: float | None = None) -> dict[str, DryMeltModel]:
    """Return Fig. 9 dry melting models (Katz, MB88, pMELTS)."""
    from petrology.melting.katz2003 import KATZ_CPX_MASS, katz2003_melt_fraction_dry
    from petrology.melting.pymelt_lithology_adapter import instantiate_pymelt_lithology

    mc = float(KATZ_CPX_MASS if cpx_mass is None else cpx_mass)
    mb88 = instantiate_pymelt_lithology("mckenzie_lherzolite")

    def _katz(p: float, t: float) -> float:
        return katz2003_melt_fraction_dry(p, t, cpx_mass=mc)

    def _mb88(p: float, t: float) -> float:
        return float(mb88.F(float(p), float(t)))

    return {
        "katz2003": _CallableDryModel(
            key="katz2003",
            label="Katz et al. (2003)",
            verified="yes",
            _f=_katz,
        ),
        "mckenzie1988": _CallableDryModel(
            key="mckenzie1988",
            label="McKenzie & Bickle (1988)",
            verified="yes",
            _f=_mb88,
        ),
        "pmelts2002": _CallableDryModel(
            key="pmelts2002",
            label="pMELTS (Ghiorso et al., 2002)",
            verified="partial",
            _f=pmelts2002_melt_fraction,
        ),
    }
