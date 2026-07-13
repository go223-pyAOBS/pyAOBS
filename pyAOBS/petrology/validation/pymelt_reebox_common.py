"""
Shared pyMelt vs REEBOX-core benchmark helpers.

Tier A (shared_geometry): same inputs → should match (P0, Pf when aligned).
Tier B (shared_algorithm): same F(P) + passive χ=1 → H should match pyMelt spreadingCentre.
Tier C (different_physics): isentropic vs PM2001 F — informational only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from petrology.melting.isentropic import IsentropicPath, IsentropicStep
from petrology.melting.reebox_geometry import (
    km_to_lithosphere_pressure_gpa,
    p0_gpa_at_solidus,
    triangular_crust_thickness_km,
)

Tier = Literal["shared_geometry", "shared_algorithm", "different_physics"]


@dataclass(frozen=True)
class ParamSpec:
    key: str
    label: str
    tier: Tier
    unit: str
    tol: float | None  # None → no pass/fail
    note: str


PARAM_SPECS: tuple[ParamSpec, ...] = (
    ParamSpec("P0", "P0 起始熔融压力", "shared_geometry", "GPa", 0.015, "对齐 Katz 固相线 ∩ 绝热线"),
    ParamSpec("Pf", "Pf 最浅熔融压力", "shared_geometry", "GPa", 0.025, "同 P0/Pf 边界"),
    ParamSpec(
        "H_common",
        "H 共性 (χ=1, pyMelt F)",
        "shared_algorithm",
        "km",
        1.5,
        "本代码三角积分 + pyMelt F(P) vs pyMelt spreadingCentre",
    ),
    ParamSpec(
        "H_passive_reebox_f",
        "H 被动 (χ=1, REEBOX F)",
        "shared_algorithm",
        "km",
        None,
        "同积分器 + 等熵 F；与 H_common 差反映 F 差异",
    ),
    ParamSpec("H", "H 生产默认", "different_physics", "km", None, "REEBOX 等熵 F + 可选 χ 权重"),
    ParamSpec("F_rmse", "F(P) RMSE", "different_physics", "", None, "等熵 vs PM2001；物理不同时偏大属正常"),
    ParamSpec("F_path", "路径平均 F̄", "different_physics", "", None, "PM2001 vs 等熵"),
    ParamSpec("F_pf", "F@Pf", "different_physics", "", None, "柱底熔融分数"),
)


def path_from_p_f(
    p_gpa: np.ndarray,
    f_total: np.ndarray,
    *,
    tp_c: float,
    p0_gpa: float,
    pf_gpa: float,
) -> IsentropicPath:
    """Minimal IsentropicPath for triangular H integration from (P, F) arrays."""
    steps = [
        IsentropicStep(
            p_gpa=float(p),
            t_c=float(tp_c),
            f_by_name={"bulk": float(f)},
            df_dp_by_name={"bulk": 0.0},
            dtdp_c_per_gpa=0.0,
        )
        for p, f in zip(p_gpa, f_total)
    ]
    return IsentropicPath(
        tp_c=float(tp_c),
        p0_gpa=float(p0_gpa),
        pf_gpa=float(pf_gpa),
        steps=steps,
        u0_by_name={"bulk": 1.0},
    )


def _as_float_array(series) -> np.ndarray:
    if hasattr(series, "iloc"):
        return np.asarray(series, dtype=float)
    return np.asarray(series, dtype=float)


def pymelt_spreading_h_km(geosettings, column, *, b_km: float) -> float:
    from petrology.melting.pymelt_bridge import _km_to_lithosphere_pressure_gpa, _pymelt_tc_km

    p_lith = _km_to_lithosphere_pressure_gpa(b_km)
    sc = geosettings.spreadingCentre(column, P_lithosphere=float(p_lith))
    return float(_pymelt_tc_km(getattr(sc, "tc", None)))


def f_rmse_on_grid(p_a: np.ndarray, f_a: np.ndarray, p_b: np.ndarray, f_b: np.ndarray, n: int = 200) -> float:
    p_lo = max(float(np.min(p_a)), float(np.min(p_b)))
    p_hi = min(float(np.max(p_a)), float(np.max(p_b)))
    if p_hi <= p_lo:
        return float("nan")
    grid = np.linspace(p_hi, p_lo, int(n))
    order_a = np.argsort(p_a)
    order_b = np.argsort(p_b)
    fa = np.interp(grid, p_a[order_a], f_a[order_a])
    fb = np.interp(grid, p_b[order_b], f_b[order_b])
    return float(np.sqrt(np.mean((fa - fb) ** 2)))


def check_param(spec: ParamSpec, ree_val: float, pm_val: float | None = None) -> dict[str, Any]:
    delta = float(ree_val - pm_val) if pm_val is not None else float("nan")
    rel = abs(delta) / max(abs(pm_val), 1e-9) if pm_val is not None and pm_val == pm_val else float("nan")
    passed: bool | None = None
    if spec.tol is not None and pm_val is not None and ree_val == ree_val and pm_val == pm_val:
        passed = abs(delta) <= spec.tol
    return {
        "key": spec.key,
        "label": spec.label,
        "tier": spec.tier,
        "unit": spec.unit,
        "ree": float(ree_val),
        "pm": float(pm_val) if pm_val is not None else None,
        "delta": delta,
        "rel_delta": rel,
        "tol": spec.tol,
        "pass": passed,
        "note": spec.note,
    }


def run_benchmark_case(
    *,
    tp_c: float,
    chi: float,
    phi: float,
    b_km: float,
    label: str = "",
    geometry: str = "reebox",
    n_isentropic_steps: int = 48,
) -> dict[str, Any]:
    from petrology.melting.heterogeneous import forward_heterogeneous_column
    from petrology.melting.lithology import heterogeneous_source
    from petrology.melting.pymelt_bridge import _import_pymelt, build_mantle, forward_pymelt_column

    liths = heterogeneous_source(pyroxenite_frac=phi)
    p0_ref = p0_gpa_at_solidus(tp_c, liths)

    ree = forward_heterogeneous_column(
        tp_c=tp_c,
        b_km=b_km,
        chi=chi,
        pyroxenite_frac=phi,
        n_isentropic_steps=n_isentropic_steps,
        geometry=geometry,
    )
    pm = forward_pymelt_column(
        tp_c=tp_c,
        pyroxenite_frac=phi,
        chi=chi if geometry == "reebox" else None,
        b_km=b_km,
        align_geometry=(geometry == "reebox"),
    )

    _, geosettings, _, _ = _import_pymelt()
    p_pm = _as_float_array(pm.column.P)
    f_pm = _as_float_array(pm.column.F)
    path = ree.isentropic_path
    p_ree = np.array([s.p_gpa for s in path.steps], dtype=float)
    f_ree = np.array([path.f_bulk_at(s) for s in path.steps], dtype=float)

    h_pm_sc = pymelt_spreading_h_km(geosettings, pm.column, b_km=b_km)
    pm_path = path_from_p_f(p_pm, f_pm, tp_c=tp_c, p0_gpa=p0_ref, pf_gpa=float(pm.pf_gpa))
    h_common_ree_int = triangular_crust_thickness_km(pm_path, b_km=b_km, chi=1.0)
    h_passive_ree_f = triangular_crust_thickness_km(path, b_km=b_km, chi=1.0)

    rmse = f_rmse_on_grid(p_ree, f_ree, p_pm, f_pm)
    ree_f_path = float(path.fbar())
    ree_f_pf = float(path.f_total)

    checks: list[dict[str, Any]] = []
    for spec in PARAM_SPECS:
        if spec.key == "P0":
            checks.append(check_param(spec, ree.p0_gpa, pm.p0_gpa))
        elif spec.key == "Pf":
            checks.append(check_param(spec, ree.pf_gpa, pm.pf_gpa))
        elif spec.key == "H_common":
            checks.append(check_param(spec, h_common_ree_int, h_pm_sc))
        elif spec.key == "H_passive_reebox_f":
            c = check_param(spec, h_passive_ree_f, h_pm_sc)
            c["note"] = f"{spec.note} (Δ vs pyMelt SC={h_passive_ree_f - h_pm_sc:+.1f} km)"
            checks.append(c)
        elif spec.key == "H":
            c = check_param(spec, ree.h_km, pm.h_km)
            checks.append(c)
        elif spec.key == "F_rmse":
            c = check_param(spec, rmse, 0.0)
            c["pm"] = None
            c["delta"] = float("nan")
            c["pass"] = None
            checks.append(c)
        elif spec.key == "F_path":
            checks.append(check_param(spec, ree_f_path, float(pm.fbar)))
        elif spec.key == "F_pf":
            checks.append(check_param(spec, ree_f_pf, float(f_pm[-1]) if len(f_pm) else 0.0))

    n_pass = sum(1 for c in checks if c["pass"] is True)
    n_fail = sum(1 for c in checks if c["pass"] is False)
    n_na = sum(1 for c in checks if c["pass"] is None)

    tag = label or f"Tp{tp_c:.0f}_chi{chi:g}_phi{phi:.2f}"
    return {
        "label": tag,
        "geometry": geometry,
        "tp_c": tp_c,
        "chi": chi,
        "phi": phi,
        "b_km": b_km,
        "p0_ref_gpa": p0_ref,
        "checks": checks,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_na": n_na,
        "p_ree": p_ree,
        "f_ree": f_ree,
        "p_pm": p_pm,
        "f_pm": f_pm,
        "h_common_ree_int": h_common_ree_int,
        "h_pm_spreading": h_pm_sc,
        "h_passive_ree_f": h_passive_ree_f,
        "ree_h_km": ree.h_km,
        "pm_h_km": pm.h_km,
        "pm_version": pm.pymelt_version,
        "pm_crust_method": pm.crust_method,
    }
