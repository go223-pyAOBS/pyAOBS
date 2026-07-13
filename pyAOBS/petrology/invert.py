"""Step-2 bounding inversion MVP for KKHS02 workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .fractionation import delta_vp_km_s
from .vp_regression import load_eq1, predict_vp_km_s


@dataclass(frozen=True)
class BulkVpBounds:
    """Bulk Vp interval inferred from lower-crust velocity."""

    v_bulk_lower_km_s: float
    v_bulk_upper_km_s: float
    delta_vp_fc_km_s: float
    f_lower: float
    notes: str = "V_bulk in [V_LC - f_lower*ΔVp, V_LC]"


def bulk_vp_bounds(
    v_lc_obs_km_s: float,
    *,
    f_lower: float,
    delta_vp_fc_km_s: float,
) -> BulkVpBounds:
    """
    Step-2 bound using FC-derived ΔVp and lower-crust fraction.

    `f_lower` scales the FC offset from cumulate-only to whole-crust bulk.
    """
    f_l = float(np.clip(f_lower, 0.0, 1.0))
    d = max(float(delta_vp_fc_km_s), 0.0)
    lower = float(v_lc_obs_km_s - f_l * d)
    upper = float(v_lc_obs_km_s)
    if lower > upper:
        lower, upper = upper, lower
    return BulkVpBounds(
        v_bulk_lower_km_s=lower,
        v_bulk_upper_km_s=upper,
        delta_vp_fc_km_s=d,
        f_lower=f_l,
    )


def bulk_vp_bounds_from_fractionation(
    v_lc_obs_km_s: float,
    *,
    f_lower: float = 0.5,
    f_solid: float = 0.75,
    p_fc_mpa: float = 400.0,
    bulk_vp_guess_km_s: float = 7.2,
    delta_vp_engine: str = "param",
    melt_oxides_wt: dict | None = None,
    delta_vp_wl_kw: dict | None = None,
) -> BulkVpBounds:
    """Estimate ΔVp via MVP parameterization or W&L FC when melt composition is given."""
    wl_kw = dict(delta_vp_wl_kw or {})
    d = delta_vp_km_s(
        f_solid,
        bulk_vp_km_s=bulk_vp_guess_km_s,
        p_fc_mpa=p_fc_mpa,
        engine=delta_vp_engine,  # type: ignore[arg-type]
        melt_oxides_wt=melt_oxides_wt,
        **wl_kw,
    )
    return bulk_vp_bounds(
        v_lc_obs_km_s,
        f_lower=f_lower,
        delta_vp_fc_km_s=d,
    )


def feasible_pf_region(
    bounds: BulkVpBounds,
    *,
    p_range_gpa: tuple[float, float] = (0.8, 3.2),
    f_range: tuple[float, float] = (0.02, 0.22),
    p_step: float = 0.02,
    f_step: float = 0.002,
    vp_bias_km_s: float = 0.0,
    vp_tolerance_km_s: float = 0.0,
    eq1: dict | None = None,
) -> dict:
    """
    Solve feasible (P, F) region where eq.(1) prediction falls in bulk bounds.
    """
    eq1 = eq1 or load_eq1()
    p_grid = np.arange(p_range_gpa[0], p_range_gpa[1] + p_step * 0.5, p_step)
    f_grid = np.arange(f_range[0], f_range[1] + f_step * 0.5, f_step)

    feas = []
    all_pts = []
    for p in p_grid:
        for f in f_grid:
            vp_raw = predict_vp_km_s(float(p), float(f), eq1=eq1)
            vp = vp_raw + float(vp_bias_km_s)
            all_pts.append((float(p), float(f), float(vp)))
            if (bounds.v_bulk_lower_km_s - vp_tolerance_km_s) <= vp <= (
                bounds.v_bulk_upper_km_s + vp_tolerance_km_s
            ):
                feas.append((float(p), float(f), float(vp)))

    arr = np.array(feas, dtype=float) if feas else np.empty((0, 3), dtype=float)
    all_arr = np.array(all_pts, dtype=float) if all_pts else np.empty((0, 3), dtype=float)
    out = {
        "n_feasible": int(arr.shape[0]),
        "points": arr,
        "vp_model_range_km_s": (
            float(all_arr[:, 2].min()) if all_arr.size else float("nan"),
            float(all_arr[:, 2].max()) if all_arr.size else float("nan"),
        ),
    }
    if arr.shape[0] > 0:
        out.update(
            {
                "p_range_gpa": (float(arr[:, 0].min()), float(arr[:, 0].max())),
                "f_range": (float(arr[:, 1].min()), float(arr[:, 1].max())),
                "vp_range_km_s": (float(arr[:, 2].min()), float(arr[:, 2].max())),
                "p_mean_gpa": float(arr[:, 0].mean()),
                "f_mean": float(arr[:, 1].mean()),
            }
        )
    elif all_arr.shape[0] > 0:
        center = 0.5 * (bounds.v_bulk_lower_km_s + bounds.v_bulk_upper_km_s)
        idx = int(np.argmin(np.abs(all_arr[:, 2] - center)))
        out.update(
            {
                "nearest_point": (
                    float(all_arr[idx, 0]),
                    float(all_arr[idx, 1]),
                    float(all_arr[idx, 2]),
                ),
                "nearest_distance_km_s": float(abs(all_arr[idx, 2] - center)),
            }
        )
    return out


def invert_single_observation(
    v_lc_obs_km_s: float,
    *,
    f_lower: float = 0.5,
    f_solid: float = 0.75,
    p_fc_mpa: float = 400.0,
    bulk_vp_guess_km_s: float = 7.2,
    delta_vp_engine: str = "param",
    melt_oxides_wt: dict | None = None,
    delta_vp_wl_kw: dict | None = None,
    vp_bias_km_s: float = -0.10,
    vp_tolerance_km_s: float = 0.05,
    eq1: dict | None = None,
) -> dict:
    """Convenience wrapper for one observation row."""
    bounds = bulk_vp_bounds_from_fractionation(
        v_lc_obs_km_s,
        f_lower=f_lower,
        f_solid=f_solid,
        p_fc_mpa=p_fc_mpa,
        bulk_vp_guess_km_s=bulk_vp_guess_km_s,
        delta_vp_engine=delta_vp_engine,
        melt_oxides_wt=melt_oxides_wt,
        delta_vp_wl_kw=delta_vp_wl_kw,
    )
    region = feasible_pf_region(
        bounds,
        vp_bias_km_s=vp_bias_km_s,
        vp_tolerance_km_s=vp_tolerance_km_s,
        eq1=eq1,
    )
    return {
        "bounds": bounds,
        "region": region,
    }


def invert_observation_rows(
    rows: list[dict],
    *,
    v_lc_key: str = "v_lc_obs_km_s",
    f_lower_key: str = "f_lower",
    f_solid: float = 0.75,
    p_fc_mpa: float = 400.0,
    bulk_vp_guess_km_s: float = 7.2,
    delta_vp_engine: str = "param",
    melt_oxides_wt: dict | None = None,
    delta_vp_wl_kw: dict | None = None,
    vp_bias_km_s: float = -0.10,
    vp_tolerance_km_s: float = 0.05,
) -> list[dict]:
    """Batch inversion for multiple profile points / stations."""
    eq1 = load_eq1()
    out = []
    for i, row in enumerate(rows):
        v_lc = float(row[v_lc_key])
        f_lower = float(row.get(f_lower_key, 0.5))
        res = invert_single_observation(
            v_lc,
            f_lower=f_lower,
            f_solid=f_solid,
            p_fc_mpa=p_fc_mpa,
            bulk_vp_guess_km_s=bulk_vp_guess_km_s,
            delta_vp_engine=delta_vp_engine,
            melt_oxides_wt=melt_oxides_wt,
            delta_vp_wl_kw=delta_vp_wl_kw,
            vp_bias_km_s=vp_bias_km_s,
            vp_tolerance_km_s=vp_tolerance_km_s,
            eq1=eq1,
        )
        bounds = res["bounds"]
        region = res["region"]
        rec = {
            "row_index": i,
            "id": row.get("id", f"obs_{i:03d}"),
            "v_lc_obs_km_s": v_lc,
            "f_lower": f_lower,
            "delta_vp_fc_km_s": bounds.delta_vp_fc_km_s,
            "v_bulk_lower_km_s": bounds.v_bulk_lower_km_s,
            "v_bulk_upper_km_s": bounds.v_bulk_upper_km_s,
            "n_feasible": region["n_feasible"],
            "p_mean_gpa": region.get("p_mean_gpa"),
            "f_mean": region.get("f_mean"),
            "p_min_gpa": region.get("p_range_gpa", (None, None))[0] if region["n_feasible"] else None,
            "p_max_gpa": region.get("p_range_gpa", (None, None))[1] if region["n_feasible"] else None,
            "f_min": region.get("f_range", (None, None))[0] if region["n_feasible"] else None,
            "f_max": region.get("f_range", (None, None))[1] if region["n_feasible"] else None,
            "nearest_distance_km_s": region.get("nearest_distance_km_s"),
            "nearest_p_gpa": region.get("nearest_point", (None, None, None))[0]
            if "nearest_point" in region
            else None,
            "nearest_f": region.get("nearest_point", (None, None, None))[1]
            if "nearest_point" in region
            else None,
            "nearest_vp_km_s": region.get("nearest_point", (None, None, None))[2]
            if "nearest_point" in region
            else None,
        }
        out.append(rec)
    return out

