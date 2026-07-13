"""
CDAT composition library for BASALT.FOR / BASALT+langmuir MAIN.

Fortran CDAT format
-------------------
Optional header ``ncase ncomp`` (e.g. ``3 7``), then one row per case::

    CAA  NAAL  MGO   FEO   CAWO  TIO2  SIO2

Rows are **W&L explicit components** (sum of first six normalized to 1 on load).
SIO2 is implicit in STATE/CIMPL; the 7th column is usually 0.

Build / refresh on-disk files::

    py -3.11 petrology/validation/build_cdat_library.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .basalt1990.common import CNAMEJ, NCOMPT
from .wl_components import csj_to_oxides_wt, normalize_melt_oxides, oxides_wt_to_csj

FIG2_PAPER_SAMPLE = "kinzler1997_morb_primary"

CDAT_DIR = Path(__file__).resolve().parents[1] / "data" / "cdat"
CATALOG_JSON = CDAT_DIR / "catalog.json"

_OXIDE_KEYS = (
    "SiO2",
    "TiO2",
    "Al2O3",
    "Cr2O3",
    "FeO",
    "MgO",
    "CaO",
    "Na2O",
    "K2O",
)


@dataclass(frozen=True)
class CdatSample:
    """One catalogued melt / bulk composition."""

    id: str
    name: str
    description: str
    oxides_wt_percent: dict[str, float]
    source: str = ""
    notes: str = ""
    tags: tuple[str, ...] = ()

    def csj(self) -> np.ndarray:
        return oxides_wt_to_csj(self.oxides_wt_percent)

    def oxides_normalized(self) -> dict[str, float]:
        return normalize_melt_oxides(self.oxides_wt_percent)


# Built-in samples (oxide wt%). Values rounded for readability; loader renormalizes.
_BUILTIN: dict[str, CdatSample] = {}


def _reg(sample: CdatSample) -> CdatSample:
    _BUILTIN[sample.id] = sample
    return sample


reg = _reg

kinzler1997_morb_primary = _reg(
    CdatSample(
        id="kinzler1997_morb_primary",
        name="Kinzler (1997) MORB primary",
        description="KKHS02 Fig.2 starting liquid; polybaric aggregated melt F≈0.09 @ 1.5 GPa",
        source="petrology/data/mantle_melts/kinzler1997_morb_primary.json",
        oxides_wt_percent={
            "SiO2": 48.2,
            "TiO2": 0.94,
            "Al2O3": 16.4,
            "Cr2O3": 0.12,
            "FeO": 7.96,
            "MgO": 12.5,
            "CaO": 11.4,
            "K2O": 0.07,
            "Na2O": 2.27,
        },
        tags=("morb", "fig2", "kkhs02"),
    )
)

magnesian_tholeiite = _reg(
    CdatSample(
        id="magnesian_tholeiite",
        name="Magnesian tholeiite",
        description="High-Mg demo; crystallizes under Langmuir Kd @ 1273–1473 K",
        source="run_basalt1990.py / test_basalt1990_fortran.py",
        oxides_wt_percent={
            "SiO2": 45.0,
            "TiO2": 0.5,
            "Al2O3": 10.0,
            "FeO": 8.0,
            "MgO": 25.0,
            "CaO": 10.0,
            "Na2O": 1.0,
            "K2O": 0.1,
            "Cr2O3": 0.0,
        },
        tags=("demo", "crystallizes", "high_mg"),
    )
)

picrite_high_mg = _reg(
    CdatSample(
        id="picrite_high_mg",
        name="Picrite (high MgO)",
        description="Very magnesian; olivine-saturated liquid proxy",
        source="synthetic",
        oxides_wt_percent={
            "SiO2": 43.0,
            "TiO2": 0.4,
            "Al2O3": 8.5,
            "FeO": 7.5,
            "MgO": 32.0,
            "CaO": 7.5,
            "Na2O": 0.8,
            "K2O": 0.05,
            "Cr2O3": 0.25,
        },
        tags=("picrite", "high_mg", "crystallizes"),
    )
)

ferro_basalt = _reg(
    CdatSample(
        id="ferro_basalt",
        name="Ferro-basalt",
        description="Evolved low-Mg, high-Fe basalt end-member",
        source="synthetic",
        oxides_wt_percent={
            "SiO2": 52.0,
            "TiO2": 2.5,
            "Al2O3": 14.0,
            "FeO": 14.0,
            "MgO": 4.5,
            "CaO": 9.0,
            "Na2O": 3.0,
            "K2O": 0.5,
            "Cr2O3": 0.05,
        },
        tags=("evolved", "low_mg"),
    )
)

alkali_basalt = _reg(
    CdatSample(
        id="alkali_basalt",
        name="Alkali basalt",
        description="Higher Na2O + K2O; stronger plagioclase control",
        source="synthetic",
        oxides_wt_percent={
            "SiO2": 47.0,
            "TiO2": 2.0,
            "Al2O3": 15.5,
            "FeO": 10.0,
            "MgO": 8.0,
            "CaO": 10.0,
            "Na2O": 4.5,
            "K2O": 1.5,
            "Cr2O3": 0.1,
        },
        tags=("alkali", "morb"),
    )
)

mid_ocean_ridge_glass = _reg(
    CdatSample(
        id="mid_ocean_ridge_glass",
        name="MORB glass (typical)",
        description="Representative N-MORB glass composition (literature average)",
        source="literature composite",
        oxides_wt_percent={
            "SiO2": 49.5,
            "TiO2": 1.2,
            "Al2O3": 15.8,
            "FeO": 10.2,
            "MgO": 7.8,
            "CaO": 11.5,
            "Na2O": 2.6,
            "K2O": 0.15,
            "Cr2O3": 0.1,
        },
        tags=("morb", "glass"),
    )
)


def list_samples() -> list[CdatSample]:
    """All built-in catalog samples."""
    return list(_BUILTIN.values())


def get_sample(sample_id: str) -> CdatSample:
    if sample_id not in _BUILTIN:
        known = ", ".join(sorted(_BUILTIN))
        raise KeyError(f"unknown CDAT sample {sample_id!r}; known: {known}")
    return _BUILTIN[sample_id]


def get_csj(sample_id: str) -> np.ndarray:
    return get_sample(sample_id).csj()


def load_kinzler_from_json(path: Path | None = None) -> CdatSample:
    """Load Kinzler primary melt JSON; falls back to built-in if missing."""
    path = path or (CDAT_DIR.parent / "mantle_melts" / "kinzler1997_morb_primary.json")
    if not path.is_file():
        return kinzler1997_morb_primary
    data = json.loads(path.read_text(encoding="utf-8"))
    wt = {k: float(v) for k, v in data["oxides_wt_percent"].items()}
    return CdatSample(
        id=data.get("id", "kinzler1997_morb_primary"),
        name="Kinzler (1997) MORB primary",
        description=data.get("description", ""),
        source=str(path),
        oxides_wt_percent=wt,
        tags=("morb", "fig2"),
    )


def load_cdat_file(path: Path | str) -> list[np.ndarray]:
    """
    Load CDAT file → list of CSJ rows (7 floats per case).

    Skips ``#`` comments; normalizes explicit six components to sum 1.
    """
    path = Path(path)
    rows: list[np.ndarray] = []
    text = path.read_text(encoding="utf-8").strip().splitlines()
    start = 0
    if text:
        parts = text[0].split()
        if len(parts) == 2 and all(_is_number(p) for p in parts):
            start = 1
    for line in text[start:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        vals = [float(x) for x in line.split()]
        if len(vals) < 6:
            continue
        row = np.zeros(NCOMPT, dtype=float)
        row[: min(len(vals), NCOMPT)] = vals[:NCOMPT]
        s = float(np.sum(row[:6]))
        if s > 1e-12:
            row[:6] /= s
        rows.append(row)
    return rows


def save_cdat_file(
    path: Path | str,
    rows: list[np.ndarray],
    *,
    comments: list[str] | None = None,
) -> None:
    """Write CDAT with Fortran-style header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [f"{len(rows)} {NCOMPT}"]
    if comments:
        for c in comments:
            lines.append(f"# {c}")
    for r in rows:
        lines.append(" ".join(f"{v:.6f}" for v in r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sample_cdat(sample: CdatSample, path: Path | None = None) -> Path:
    """Write one sample to ``data/cdat/<id>.cdat``."""
    path = path or (CDAT_DIR / f"{sample.id}.cdat")
    csj = sample.csj()
    wt = sample.oxides_normalized()
    oxide_s = ", ".join(f"{k} {wt[k]:.2f}" for k in _OXIDE_KEYS if wt.get(k, 0) > 0.01)
    comments = [
        sample.name,
        sample.description,
        f"source: {sample.source}" if sample.source else "",
        f"oxides wt%: {oxide_s}",
        "columns: " + " ".join(CNAMEJ[:NCOMPT]),
    ]
    comments = [c for c in comments if c]
    save_cdat_file(path, [csj], comments=comments)
    return path


def write_all_samples_cdat(path: Path | None = None) -> Path:
    """Write ``all_samples.cdat`` with every catalog entry."""
    path = path or (CDAT_DIR / "all_samples.cdat")
    samples = list_samples()
    rows = [s.csj() for s in samples]
    comments = [f"case {i + 1}: {s.id} — {s.name}" for i, s in enumerate(samples)]
    save_cdat_file(path, rows, comments=comments)
    return path


def write_catalog_json(path: Path | None = None) -> Path:
    """Write ``catalog.json`` metadata (oxides + CSJ for each sample)."""
    path = path or CATALOG_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for s in list_samples():
        csj = s.csj()
        entries.append(
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "source": s.source,
                "notes": s.notes,
                "tags": list(s.tags),
                "oxides_wt_percent": s.oxides_normalized(),
                "csj": {CNAMEJ[j]: float(csj[j]) for j in range(6)},
                "cdat_file": f"{s.id}.cdat",
            }
        )
    path.write_text(json.dumps({"samples": entries}, indent=2) + "\n", encoding="utf-8")
    return path


def build_library(force: bool = False) -> dict[str, Path]:
    """
    Generate all CDAT files under ``data/cdat/``.

    Returns mapping sample_id → file path.
    """
    CDAT_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for s in list_samples():
        p = CDAT_DIR / f"{s.id}.cdat"
        if force or not p.is_file():
            write_sample_cdat(s, p)
        out[s.id] = p
    all_p = CDAT_DIR / "all_samples.cdat"
    if force or not all_p.is_file():
        write_all_samples_cdat(all_p)
    out["all_samples"] = all_p
    write_catalog_json()
    out["catalog"] = CATALOG_JSON
    return out


def csj_to_report(csj: np.ndarray) -> dict[str, float | dict[str, float]]:
    """CSJ row → normalized oxides + component dict (for debugging)."""
    csj = np.asarray(csj, dtype=float).ravel()
    wt = csj_to_oxides_wt(csj[:6], sio2=float(csj[6]) if csj.size > 6 else 0.0)
    return {
        "csj": {CNAMEJ[j]: float(csj[j]) for j in range(min(6, csj.size))},
        "oxides_wt_percent": wt,
    }


def resolve_cdat_path(name: str) -> Path:
    """
    Resolve ``sample_id``, ``sample_id.cdat``, or path under ``data/cdat/``.
    """
    p = Path(name)
    if p.is_file():
        return p
    cand = CDAT_DIR / name
    if cand.is_file():
        return cand
    cand2 = CDAT_DIR / f"{name}.cdat"
    if cand2.is_file():
        return cand2
    if name in _BUILTIN:
        build_library()
        return CDAT_DIR / f"{name}.cdat"
    raise FileNotFoundError(f"CDAT not found: {name!r} (looked in {CDAT_DIR})")


def fig2_primary_melt(
    *,
    sample_id: str = FIG2_PAPER_SAMPLE,
    data_path: Path | str | None = None,
) -> dict:
    """
    Melt package for Fig.2 reproduce scripts.

    Returns ``id``, ``name``, ``description``, ``oxides_wt_percent``, optional
    ``P_melt_GPa`` / ``F_melt`` (Kinzler JSON only), and ``tags``.
    """
    if data_path is not None:
        path = Path(data_path)
        if not path.is_file():
            path = resolve_cdat_path(str(data_path))
        rows = load_cdat_file(path)
        if not rows:
            raise ValueError(f"no composition rows in {path}")
        wt = normalize_melt_oxides(csj_to_oxides_wt(rows[0][:6]))
        stem = path.stem
        return {
            "id": stem,
            "name": stem,
            "description": f"CDAT file {path.name}",
            "P_melt_GPa": float("nan"),
            "F_melt": float("nan"),
            "oxides_wt_percent": wt,
            "tags": (),
            "source": str(path),
        }

    if sample_id == FIG2_PAPER_SAMPLE:
        json_path = CDAT_DIR.parent / "mantle_melts" / "kinzler1997_morb_primary.json"
        sample = load_kinzler_from_json(json_path if json_path.is_file() else None)
        out = {
            "id": FIG2_PAPER_SAMPLE,
            "name": sample.name,
            "description": sample.description,
            "P_melt_GPa": float("nan"),
            "F_melt": float("nan"),
            "oxides_wt_percent": sample.oxides_normalized(),
            "tags": sample.tags,
            "source": sample.source,
        }
        if json_path.is_file():
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            if "P_melt_GPa" in meta:
                out["P_melt_GPa"] = float(meta["P_melt_GPa"])
            if "F_melt" in meta:
                out["F_melt"] = float(meta["F_melt"])
        return out

    sample = get_sample(sample_id)
    return {
        "id": sample.id,
        "name": sample.name,
        "description": sample.description,
        "P_melt_GPa": float("nan"),
        "F_melt": float("nan"),
        "oxides_wt_percent": sample.oxides_normalized(),
        "tags": sample.tags,
        "source": sample.source,
    }


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
