"""Load merged melt catalog (hand-entered + Kinzler 1997 grid)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

_OXIDES = ("SiO2", "TiO2", "Al2O3", "Cr2O3", "FeO", "MgO", "CaO", "Na2O", "K2O", "P2O5", "H2O")

_DATA = Path(__file__).resolve().parent / "mantle_melts"


def _parse_bool(val: str) -> bool:
    return str(val).strip().upper() in ("TRUE", "1", "YES", "T")


def load_melt_catalog(
    catalog_path: Path | str | None = None,
    grid_path: Path | str | None = None,
    *,
    include_grid: bool = True,
) -> list[dict]:
    """Return list of melt records as dicts with typed P_melt_GPa, F_melt."""
    catalog_path = Path(catalog_path or _DATA / "catalog.csv")
    rows: list[dict] = []
    for path in [catalog_path]:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                rows.append(_coerce_record(rec))
    if include_grid:
        grid_path = Path(grid_path or _DATA / "kinzler1997_grid.csv")
        if grid_path.is_file():
            with grid_path.open(newline="", encoding="utf-8") as fh:
                for rec in csv.DictReader(fh):
                    rows.append(_coerce_record(rec))
    return rows


def _coerce_record(rec: dict) -> dict:
    out = dict(rec)
    out["P_melt_GPa"] = float(rec["P_melt_GPa"])
    out["F_melt"] = float(rec["F_melt"])
    out["include_in_regression"] = _parse_bool(rec.get("include_in_regression", "true"))
    for ox in _OXIDES:
        if ox in rec and rec[ox] not in (None, ""):
            out[ox] = float(str(rec[ox]).strip())
    return out


def oxides_from_record(rec: dict) -> dict[str, float]:
    return {ox: float(rec.get(ox, 0.0) or 0.0) for ox in _OXIDES if ox != "H2O"}
