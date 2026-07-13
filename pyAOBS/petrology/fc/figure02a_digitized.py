"""
KKHS02 Figure 2(a) digitized P-wave velocity curves.

Source: ``petrology/data/figure02a数值化.txt`` (GetData Graph Digitizer).

Curves (solid fraction *x*, Vp km/s *y*):
  cum.sol.(1kb), inc.sol.(1kb), frac.res.(1kb), eq.sol.(1kb), eq.res.(1kb),
  cum.sol.(8-1kb), inc.sol.(8-1kb), frac.res.(8-1kb)
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

PathKey = Literal["fc_100", "polybaric_fc", "eq_100"]

FIGURE02A_TXT = Path(__file__).resolve().parents[1] / "data" / "figure02a数值化.txt"

CurveKey = Literal[
    "cum_sol_1kb",
    "inc_sol_1kb",
    "frac_res_1kb",
    "eq_sol_1kb",
    "eq_res_1kb",
    "cum_sol_8_1kb",
    "inc_sol_8_1kb",
    "frac_res_8_1kb",
]

_FILE_TO_KEY: dict[str, CurveKey] = {
    "cum.sol.(1kb)": "cum_sol_1kb",
    "inc.sol.(1kb)": "inc_sol_1kb",
    "frac.res.(1kb)": "frac_res_1kb",
    "eq.sol.(1kb)": "eq_sol_1kb",
    "eq.res.(1kb)": "eq_res_1kb",
    "cum.sol.(8-1kb)": "cum_sol_8_1kb",
    "inc.sol.(8-1kb)": "inc_sol_8_1kb",
    "frac.res.(8-1kb)": "frac_res_8_1kb",
}

_PATH_CUM: dict[PathKey, CurveKey] = {
    "fc_100": "cum_sol_1kb",
    "polybaric_fc": "cum_sol_8_1kb",
    "eq_100": "eq_sol_1kb",
}
_PATH_INC: dict[PathKey, CurveKey] = {
    "fc_100": "inc_sol_1kb",
    "polybaric_fc": "inc_sol_8_1kb",
    "eq_100": "inc_sol_1kb",
}
_PATH_FRAC_RES: dict[PathKey, CurveKey] = {
    "fc_100": "frac_res_1kb",
    "polybaric_fc": "frac_res_8_1kb",
}
_PATH_EQ_RES: dict[PathKey, CurveKey] = {
    "eq_100": "eq_res_1kb",
}

_NUMERIC_LINE = re.compile(r"^\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s")


def _dedupe_increasing_x(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sort by *F*; at duplicate *F* keep last digitized point."""
    pts = sorted(((float(x), float(y)) for x, y in points), key=lambda t: t[0])
    out: list[tuple[float, float]] = []
    for x, y in pts:
        if out and abs(out[-1][0] - x) <= 1e-9:
            out[-1] = (x, y)
        else:
            out.append((x, y))
    return out


def _prepend_f0(points: list[tuple[float, float]], *, y_at_zero: float | None = None) -> list[tuple[float, float]]:
    """Extrapolate to F=0 when digitization starts above zero."""
    if not points:
        return points
    if points[0][0] <= 1e-9:
        return points
    y0 = float(y_at_zero) if y_at_zero is not None else float(points[0][1])
    return [(0.0, y0)] + points


def load_figure02a_curves(path: Path | str | None = None) -> dict[CurveKey, list[tuple[float, float]]]:
    """Parse GetData export → monotonic *F* knots per curve."""
    path = Path(path or FIGURE02A_TXT)
    raw: dict[str, list[tuple[float, float]]] = {}
    section: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("Generated") or line.startswith("from file"):
            continue
        if not _NUMERIC_LINE.match(line):
            section = line
            raw.setdefault(section, [])
            continue
        if section is None:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        raw[section].append((float(parts[0]), float(parts[1])))

    out: dict[CurveKey, list[tuple[float, float]]] = {}
    for file_name, key in _FILE_TO_KEY.items():
        pts = _dedupe_increasing_x(raw.get(file_name, []))
        if key.startswith("cum_") or key.startswith("eq_sol"):
            pts = _prepend_f0(pts)
        out[key] = pts
    return out


@lru_cache(maxsize=1)
def _curves_cached() -> dict[CurveKey, list[tuple[float, float]]]:
    return load_figure02a_curves()


def _interp_knots(points: list[tuple[float, float]], f_solid: float) -> float:
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    return float(np.interp(float(f_solid), x, y, left=float(y[0]), right=float(y[-1])))


def digitized_cum_vp_km_s(path: PathKey, f_solid: float) -> float:
    return _interp_knots(_curves_cached()[_PATH_CUM[path]], f_solid)


def digitized_inc_vp_km_s(path: PathKey, f_solid: float) -> float:
    return _interp_knots(_curves_cached()[_PATH_INC[path]], f_solid)


def digitized_frac_residual_vp_km_s(path: PathKey, f_solid: float) -> float:
    key = _PATH_FRAC_RES[path]
    return _interp_knots(_curves_cached()[key], f_solid)


def digitized_eq_residual_vp_km_s(f_solid: float) -> float:
    return _interp_knots(_curves_cached()["eq_res_1kb"], f_solid)


def digitized_residual_vp_km_s(path: PathKey, f_solid: float) -> float:
    if path == "eq_100":
        return digitized_eq_residual_vp_km_s(f_solid)
    return digitized_frac_residual_vp_km_s(path, f_solid)
