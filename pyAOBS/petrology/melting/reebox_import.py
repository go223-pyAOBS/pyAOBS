"""
Import REEBOX PRO CSV exports for benchmark against Python core.

Expected columns (flexible aliases):
  Tp, chi/lambda, pyroxenite_frac/Phi, H_km, SiO2..K2O (pooled melt wt%).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

_OXIDE_ALIASES = {
    "sio2": "SiO2",
    "tio2": "TiO2",
    "al2o3": "Al2O3",
    "cr2o3": "Cr2O3",
    "feo": "FeO",
    "mgo": "MgO",
    "cao": "CaO",
    "na2o": "Na2O",
    "k2o": "K2O",
    "p2o5": "P2O5",
}


@dataclass(frozen=True)
class ReeboxBenchmarkRow:
    tp_c: float | None
    chi: float | None
    pyroxenite_frac: float | None
    h_km: float | None
    melt_wt: dict[str, float]
    meta: dict[str, str]


def _norm_key(k: str) -> str:
    return k.strip().lower().replace(" ", "_").replace("%", "")


def load_reebox_csv(path: Path | str) -> list[ReeboxBenchmarkRow]:
    path = Path(path)
    rows: list[ReeboxBenchmarkRow] = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return rows
        keys = {_norm_key(k): k for k in reader.fieldnames}
        for raw in reader:
            meta = dict(raw)
            tp = chi = phi = h = None
            if "tp" in keys or "tp_c" in keys or "potential_temperature" in keys:
                for alias in ("tp", "tp_c", "potential_temperature", "t_p"):
                    if alias in keys:
                        try:
                            tp = float(raw[keys[alias]])
                        except (TypeError, ValueError):
                            pass
                        break
            for alias, attr in (
                ("chi", "chi"),
                ("lambda", "chi"),
                ("λ", "chi"),
                ("pyroxenite_frac", "phi"),
                ("phi", "phi"),
                ("pyroxenite", "phi"),
                ("h_km", "h"),
                ("h", "h"),
                ("crustal_thickness", "h"),
            ):
                if alias in keys:
                    try:
                        val = float(raw[keys[alias]])
                        if attr == "chi":
                            chi = val
                        elif attr == "phi":
                            phi = val
                        else:
                            h = val
                    except (TypeError, ValueError):
                        pass
            melt: dict[str, float] = {}
            for nk, orig in keys.items():
                if nk in _OXIDE_ALIASES:
                    try:
                        melt[_OXIDE_ALIASES[nk]] = float(raw[orig])
                    except (TypeError, ValueError):
                        pass
            rows.append(
                ReeboxBenchmarkRow(
                    tp_c=tp,
                    chi=chi,
                    pyroxenite_frac=phi,
                    h_km=h,
                    melt_wt=melt,
                    meta=meta,
                )
            )
    return rows
