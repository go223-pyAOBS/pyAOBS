"""
Adapt pyMelt lithology objects for REEBOX-core isentropic melting.

Uses pyMelt ``TSolidus`` / ``TLiquidus`` / ``F(P,T)`` while keeping REEBOX
entropy coupling in ``isentropic.py``. Peridotite melt chemistry defaults to
``pmelts_klb1`` (pyMelt KLB-1 grid); pyroxenite / eclogite remain Kinzler batch
or fixed end-members.

Registry keys (``list_pymelt_lithology_keys()``):
  katz_lherzolite, pertermann_g2, matthews_klb1, matthews_kg1, matthews_eclogite,
  mckenzie_lherzolite, shorttle_kg1,
  ball_depleted_mantle, ball_mixed_mantle, ball_primitive_mantle
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np

from .kinzler1997_batch import HZ_DEP1_WT
from .melt_chemistry import ChemistryBackend, default_chemistry_for_lithology, melt_oxides
from .lithology import DELTA_S_PERIDOTITE, DELTA_S_PYROXENITE, G2_BULK_WT, Lithology, RHO_KG_M3
from .pymelt_bridge import _import_pymelt, _instantiate_lithology, ensure_pymelt_scipy_compat

# module path, constructor name(s)
PYMELT_LITHOLOGY_REGISTRY: dict[str, tuple[str, str]] = {
    "katz_lherzolite": ("katz", "lherzolite"),
    "pertermann_g2": ("pertermann", "g2"),
    "matthews_klb1": ("matthews", "klb1"),
    "matthews_kg1": ("matthews", "kg1"),
    "matthews_eclogite": ("matthews", "eclogite"),
    "mckenzie_lherzolite": ("mckenzie", "lherzolite"),
    "shorttle_kg1": ("shorttle", "kg1"),
    "ball_depleted_mantle": ("ball", "depleted_mantle"),
    "ball_mixed_mantle": ("ball", "mixed_mantle"),
    "ball_primitive_mantle": ("ball", "primitive_mantle"),
}

# Keys known to fail TSolidus at reference P (skip in auto-discovery).
_SKIP_KEYS: frozenset[str] = frozenset({"shorttle_harzburgite"})


@dataclass(frozen=True)
class LithologyMeta:
    family: str  # peridotite | pyroxenite | eclogite | hydrous
    description: str


LITHOLOGY_META: dict[str, LithologyMeta] = {
    "katz_lherzolite": LithologyMeta("peridotite", "Katz (2003) lherzolite"),
    "mckenzie_lherzolite": LithologyMeta("peridotite", "McKenzie (1989) lherzolite"),
    "matthews_klb1": LithologyMeta("peridotite", "Matthews et al. KLB-1 peridotite"),
    "ball_depleted_mantle": LithologyMeta("peridotite", "Ball et al. depleted mantle (ΔS=407)"),
    "ball_mixed_mantle": LithologyMeta("peridotite", "Ball et al. mixed mantle (ΔS=407)"),
    "ball_primitive_mantle": LithologyMeta("peridotite", "Ball et al. primitive mantle (ΔS=407)"),
    "pertermann_g2": LithologyMeta("pyroxenite", "Pertermann & Hirschmann G2"),
    "matthews_kg1": LithologyMeta("pyroxenite", "Matthews et al. KG-1 garnet pyroxenite"),
    "shorttle_kg1": LithologyMeta("pyroxenite", "Shorttle et al. KG-1"),
    "matthews_eclogite": LithologyMeta("eclogite", "Matthews et al. eclogite"),
}


@dataclass(frozen=True)
class LithologyPreset:
    peridotite_key: str
    pyroxenite_key: str
    peridotite_h2o_wt: float = 0.0
    pyroxenite_h2o_wt: float = 0.0
    description: str = ""


LITHOLOGY_PRESETS: dict[str, LithologyPreset] = {
    "lip_default": LithologyPreset(
        "katz_lherzolite",
        "pertermann_g2",
        description="Katz peridotite + G2 pyroxenite (REEBOX default pair)",
    ),
    "lip_kg1": LithologyPreset(
        "katz_lherzolite",
        "matthews_kg1",
        description="Katz peridotite + Matthews KG-1 enriched end-member",
    ),
    "lip_klb1_hydrous": LithologyPreset(
        "matthews_klb1",
        "matthews_kg1",
        peridotite_h2o_wt=0.1,
        description="Hydrous KLB-1 + KG-1 (0.1 wt% H2O on peridotite)",
    ),
    "lip_eclogite": LithologyPreset(
        "mckenzie_lherzolite",
        "matthews_eclogite",
        description="McKenzie peridotite + Matthews eclogite",
    ),
    "lip_ball": LithologyPreset(
        "ball_primitive_mantle",
        "pertermann_g2",
        description="Ball primitive mantle + G2 pyroxenite",
    ),
    "greenland_kg1": LithologyPreset(
        "katz_lherzolite",
        "matthews_kg1",
        description="Greenland-like scan: Katz + KG-1 enriched component",
    ),
}

# Batch-melt source proxies (wt% oxides) by family.
_SOURCE_WT: dict[str, dict[str, float]] = {
    "katz_lherzolite": dict(HZ_DEP1_WT),
    "mckenzie_lherzolite": dict(HZ_DEP1_WT),
    "matthews_klb1": dict(HZ_DEP1_WT),
    "ball_depleted_mantle": dict(HZ_DEP1_WT),
    "ball_mixed_mantle": dict(HZ_DEP1_WT),
    "ball_primitive_mantle": dict(HZ_DEP1_WT),
    "pertermann_g2": dict(G2_BULK_WT),
    "matthews_kg1": dict(G2_BULK_WT),
    "shorttle_kg1": dict(G2_BULK_WT),
    "matthews_eclogite": {
        "SiO2": 45.0,
        "TiO2": 0.8,
        "Al2O3": 12.0,
        "Cr2O3": 0.0,
        "FeO": 10.0,
        "MgO": 8.0,
        "CaO": 18.0,
        "Na2O": 2.5,
        "K2O": 0.2,
        "P2O5": 0.05,
        "H2O": 0.0,
    },
}

_DEFAULT_DELTA_S: dict[str, float] = {
    "katz_lherzolite": DELTA_S_PERIDOTITE,
    "mckenzie_lherzolite": 250.0,
    "matthews_klb1": DELTA_S_PERIDOTITE,
    "ball_depleted_mantle": 407.0,
    "ball_mixed_mantle": 407.0,
    "ball_primitive_mantle": 407.0,
    "pertermann_g2": DELTA_S_PYROXENITE,
    "matthews_kg1": DELTA_S_PYROXENITE,
    "shorttle_kg1": 380.0,
    "matthews_eclogite": DELTA_S_PYROXENITE,
}


def list_pymelt_lithology_keys() -> list[str]:
    return sorted(PYMELT_LITHOLOGY_REGISTRY.keys())


def list_lithology_presets() -> list[str]:
    return sorted(LITHOLOGY_PRESETS.keys())


def peridotite_lithology_keys() -> list[str]:
    return [
        k
        for k in list_pymelt_lithology_keys()
        if LITHOLOGY_META.get(k, LithologyMeta("peridotite", "")).family == "peridotite"
    ]


def enriched_lithology_keys() -> list[str]:
    return [
        k
        for k in list_pymelt_lithology_keys()
        if LITHOLOGY_META.get(k, LithologyMeta("pyroxenite", "")).family in ("pyroxenite", "eclogite")
    ]


def lithology_catalog(*, p_gpa: float = 2.0) -> list[dict[str, Any]]:
    """Human-readable catalog for CLI / validation."""
    rows: list[dict[str, Any]] = []
    for key in list_pymelt_lithology_keys():
        meta = LITHOLOGY_META.get(key)
        row: dict[str, Any] = {
            "key": key,
            "family": meta.family if meta else "unknown",
            "description": meta.description if meta else "",
        }
        try:
            d = lithology_diagnostics(key, p_gpa=p_gpa)
            row.update(d)
            row["valid"] = np.isfinite(d["tsol_c"])
        except Exception as exc:
            row["valid"] = False
            row["error"] = str(exc)
        rows.append(row)
    return rows


def print_lithology_catalog(*, p_gpa: float = 2.0) -> None:
    print("=" * 72)
    print("pyMelt lithology registry")
    print("=" * 72)
    for row in lithology_catalog(p_gpa=p_gpa):
        if row.get("valid"):
            print(
                f"  {row['key']:24s} [{row['family']:10s}] "
                f"Tsol={row['tsol_c']:7.1f}C  Tliq={row['tliq_c']:7.1f}C  "
                f"ΔS={row['delta_s']:.0f}"
            )
        else:
            err = row.get("error", "invalid solidus")
            print(f"  {row['key']:24s} SKIP ({err})")
    print("\nPresets:")
    for name in list_lithology_presets():
        p = LITHOLOGY_PRESETS[name]
        print(
            f"  {name:18s}  {p.peridotite_key} + {p.pyroxenite_key}"
            f"{f' (H2O={p.peridotite_h2o_wt:g})' if p.peridotite_h2o_wt else ''}"
            f"  — {p.description}"
        )


def resolve_lithology_col_kwargs(
    *,
    lithology_backend: str = "native",
    lithology_preset: str | None = None,
    peridotite_lith: str = "katz_lherzolite",
    pyroxenite_lith: str = "pertermann_g2",
    peridotite_h2o_wt: float = 0.0,
    pyroxenite_h2o_wt: float = 0.0,
    peridotite_chemistry: ChemistryBackend | None = None,
) -> dict[str, Any]:
    """Kwargs for ``forward_melting_column`` / ``scan_hvp_lip`` lithology options."""
    chem_kw = {}
    if peridotite_chemistry is not None:
        chem_kw["peridotite_chemistry"] = peridotite_chemistry
    if lithology_preset:
        if lithology_preset not in LITHOLOGY_PRESETS:
            raise KeyError(
                f"unknown lithology preset {lithology_preset!r}; "
                f"choose from {list_lithology_presets()}"
            )
        p = LITHOLOGY_PRESETS[lithology_preset]
        return {
            "lithology_backend": "pymelt",
            "peridotite_lith": p.peridotite_key,
            "pyroxenite_lith": p.pyroxenite_key,
            "peridotite_h2o_wt": p.peridotite_h2o_wt,
            "pyroxenite_h2o_wt": p.pyroxenite_h2o_wt,
            **chem_kw,
        }
    if lithology_backend not in ("native", "pymelt"):
        raise ValueError(f"unknown lithology backend: {lithology_backend}")
    out = {
        "lithology_backend": lithology_backend,
        "peridotite_lith": peridotite_lith,
        "pyroxenite_lith": pyroxenite_lith,
        "peridotite_h2o_wt": float(peridotite_h2o_wt),
        "pyroxenite_h2o_wt": float(pyroxenite_h2o_wt),
    }
    out.update(chem_kw)
    return out


def add_lithology_cli(parser: argparse.ArgumentParser) -> None:
    """Register lithology arguments on an ``ArgumentParser``."""
    preset_help = ", ".join(list_lithology_presets())
    g = parser.add_argument_group("Lithology (reebox engine)")
    g.add_argument(
        "--lithology-backend",
        choices=("native", "pymelt"),
        default="native",
        help="native=Katz+G2 analytic; pymelt=pyMelt registry curves",
    )
    g.add_argument(
        "--lithology-preset",
        type=str,
        default=None,
        metavar="NAME",
        help=f"Named pair (implies --lithology-backend pymelt): {preset_help}",
    )
    g.add_argument("--peridotite-lith", default="katz_lherzolite")
    g.add_argument("--pyroxenite-lith", default="pertermann_g2")
    g.add_argument("--peridotite-h2o", type=float, default=0.0, metavar="WT")
    g.add_argument("--pyroxenite-h2o", type=float, default=0.0, metavar="WT")
    g.add_argument(
        "--list-lithologies",
        action="store_true",
        help="Print lithology catalog and exit",
    )


def lithology_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return resolve_lithology_col_kwargs(
        lithology_backend=args.lithology_backend,
        lithology_preset=args.lithology_preset,
        peridotite_lith=args.peridotite_lith,
        pyroxenite_lith=args.pyroxenite_lith,
        peridotite_h2o_wt=args.peridotite_h2o,
        pyroxenite_h2o_wt=args.pyroxenite_h2o,
    )


def t_from_f_pymelt(pm_lith: Any, p_gpa: float, f: float, *, n_iter: int = 64) -> float:
    """Invert ``F(P,T)=f`` via bisection between solidus and liquidus."""
    f = float(np.clip(f, 0.0, 0.999))
    if f <= 0.0:
        return float(pm_lith.TSolidus(float(p_gpa)))
    ts = float(pm_lith.TSolidus(float(p_gpa)))
    tl = float(pm_lith.TLiquidus(float(p_gpa)))
    if f >= 0.999:
        return tl
    lo, hi = ts, tl
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        fm = float(pm_lith.F(float(p_gpa), mid))
        if fm < f:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def _wrap_hydrous(base: Any, h2o_wt: float) -> Any:
    if h2o_wt <= 1e-9:
        return base
    ensure_pymelt_scipy_compat()
    import pyMelt as pm

    return pm.hydrousLithology(base, float(h2o_wt))


def is_lithology_valid(key: str, *, p_gpa: float = 2.0) -> bool:
    """True if TSolidus is finite at reference pressure."""
    if key in _SKIP_KEYS:
        return False
    try:
        pm = instantiate_pymelt_lithology(key)
        ts = float(pm.TSolidus(float(p_gpa)))
        return np.isfinite(ts)
    except Exception:
        return False


def instantiate_pymelt_lithology(key: str, *, h2o_wt: float = 0.0) -> Any:
    """Return raw pyMelt lithology instance."""
    if key not in PYMELT_LITHOLOGY_REGISTRY:
        raise KeyError(f"unknown pymelt lithology {key!r}; choose from {list_pymelt_lithology_keys()}")
    mod_name, ctor_name = PYMELT_LITHOLOGY_REGISTRY[key]
    _, _, lithologies, _ = _import_pymelt()
    mod = getattr(lithologies, mod_name)
    base = _instantiate_lithology(mod, ctor_name)
    return _wrap_hydrous(base, h2o_wt)


def _solidus_liquidus_callables(pm_lith: Any) -> tuple[Any, Any]:
    def solidus(p: float) -> float:
        return float(pm_lith.TSolidus(float(p)))

    def liquidus(p: float) -> float:
        return float(pm_lith.TLiquidus(float(p)))

    return solidus, liquidus


def _rho_kg_m3(pm_lith: Any) -> float:
    rhos = getattr(pm_lith, "rhos", None)
    if rhos is None:
        return RHO_KG_M3
    val = float(rhos)
    # pyMelt rhos in g/cm³
    return val * 1000.0 if val < 500.0 else val


def pymelt_lithology(
    key: str,
    *,
    u0: float = 1.0,
    name: str | None = None,
    h2o_wt: float = 0.0,
    chemistry_backend: ChemistryBackend | None = None,
) -> Lithology:
    """Build REEBOX ``Lithology`` backed by pyMelt melting curves."""
    pm = instantiate_pymelt_lithology(key, h2o_wt=h2o_wt)
    solidus_c, liquidus_c = _solidus_liquidus_callables(pm)
    delta_s = float(getattr(pm, "DeltaS", _DEFAULT_DELTA_S.get(key, DELTA_S_PERIDOTITE)))
    meta = LITHOLOGY_META.get(key, LithologyMeta("peridotite", key))
    chem = chemistry_backend or default_chemistry_for_lithology(
        family=meta.family, lithology_backend="pymelt"
    )
    label = name or key
    if h2o_wt > 1e-9:
        label = f"{label}_h2o{h2o_wt:g}"
    return Lithology(
        name=label,
        u0=float(u0),
        delta_s_fusion=delta_s,
        source_wt=dict(_SOURCE_WT.get(key, HZ_DEP1_WT)),
        solidus_c=solidus_c,
        liquidus_c=liquidus_c,
        melt_beta=1.2,
        rho_kg_m3=_rho_kg_m3(pm),
        chemistry_backend=chem,
        _pymelt=pm,
    )


def heterogeneous_source_pymelt(
    *,
    pyroxenite_frac: float,
    peridotite_key: str = "katz_lherzolite",
    pyroxenite_key: str = "pertermann_g2",
    peridotite_h2o_wt: float = 0.0,
    pyroxenite_h2o_wt: float = 0.0,
    peridotite_chemistry: ChemistryBackend | None = None,
) -> list[Lithology]:
    """Peridotite + enriched lithology pair from pyMelt registry."""
    phi = float(np.clip(pyroxenite_frac, 0.0, 0.99))
    if phi <= 0.0:
        return [
            pymelt_lithology(
                peridotite_key,
                u0=1.0,
                h2o_wt=peridotite_h2o_wt,
                chemistry_backend=peridotite_chemistry,
            ),
        ]
    if phi >= 0.99:
        return [
            pymelt_lithology(
                peridotite_key,
                u0=1.0,
                h2o_wt=peridotite_h2o_wt,
                chemistry_backend=peridotite_chemistry,
            ),
        ]
    return [
        pymelt_lithology(
            peridotite_key,
            u0=1.0 - phi,
            h2o_wt=peridotite_h2o_wt,
            chemistry_backend=peridotite_chemistry,
        ),
        pymelt_lithology(pyroxenite_key, u0=phi, h2o_wt=pyroxenite_h2o_wt, chemistry_backend="kinzler1997"),
    ]


def lithology_diagnostics(key: str, *, p_gpa: float = 2.0, h2o_wt: float = 0.0) -> dict[str, float]:
    """Solidus/liquidus spot check for validation scripts."""
    pm = instantiate_pymelt_lithology(key, h2o_wt=h2o_wt)
    t_mid = 0.5 * (float(pm.TSolidus(p_gpa)) + float(pm.TLiquidus(p_gpa)))
    return {
        "key": key,
        "p_gpa": float(p_gpa),
        "h2o_wt": float(h2o_wt),
        "tsol_c": float(pm.TSolidus(p_gpa)),
        "tliq_c": float(pm.TLiquidus(p_gpa)),
        "f_mid": float(pm.F(p_gpa, t_mid)),
        "delta_s": float(getattr(pm, "DeltaS", float("nan"))),
    }
