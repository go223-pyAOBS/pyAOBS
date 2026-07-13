"""KKHS02 equation (1): norm-based V_bulk vs mean melting pressure and degree.

Inputs
------
P_bar, F_bar : average melt pressure (GPa) and melt fraction from §4 column model.

Output
------
V_bulk (km/s) at the paper reference state: 600 MPa, 400 °C, pore-free norm velocity.

Form (product tanh windows, §2.2)::

    V_bulk = a0
           + W_L(P,F) * (b0 + b1*P + b2*F + b3*P² + b4*P*F + b5*F²)
           + W_H(P,F) * (c0 + c1*P + c2*F + c3*P² + c4*P*F + c5*F²)

Coefficients and window parameters are stored in ``data/korenaga2002_eq1.json``.

Note
----
Production ``b1 = -0.55``. The KKHS02 printed table lists ``b1 = +0.55`` (sign error);
see ``equation_1.print_errata`` and :func:`eq1_with_printed_coefficients`.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

_EQ1_PATH = Path(__file__).resolve().parent / "data" / "korenaga2002_eq1.json"


def load_eq1(path: Path | str | None = None) -> dict:
    path = Path(path or _EQ1_PATH)
    return json.loads(path.read_text(encoding="utf-8"))


def eq1_with_printed_coefficients(eq1: Mapping | None = None) -> dict:
    """Return a deep copy of eq.(1) using the *printed* (erroneous) coefficient table.

    Production predictions use :func:`load_eq1` (corrected ``b1 = -0.55``).
    This helper is for errata / comparison plots only.
    """
    eq1 = copy.deepcopy(eq1 or load_eq1())
    errata = eq1["equation_1"].get("print_errata") or {}
    printed = errata.get("printed_coefficients_for_comparison")
    if not printed:
        raise KeyError("equation_1.print_errata.printed_coefficients_for_comparison missing")
    eq1["equation_1"]["coefficients"] = {
        k: float(v) for k, v in printed.items() if k.startswith(("a", "b", "c"))
    }
    return eq1


def _coef(eq1: Mapping, prefix: str) -> dict[str, float]:
    c = eq1["equation_1"]["coefficients"]
    return {f"{prefix}{i}": float(c[f"{prefix}{i}"]) for i in range(6)}


def branch_poly(p_gpa: float, f_melt: float, coef: Mapping[str, float], prefix: str) -> float:
    """Quadratic branch: k0 + k1*P + k2*F + k3*P² + k4*P*F + k5*F²."""
    p = float(p_gpa)
    f = float(f_melt)
    return (
        coef[f"{prefix}0"]
        + coef[f"{prefix}1"] * p
        + coef[f"{prefix}2"] * f
        + coef[f"{prefix}3"] * p**2
        + coef[f"{prefix}4"] * p * f
        + coef[f"{prefix}5"] * f**2
    )


def window_w_l(p_gpa: float, f_melt: float, eq1: Mapping | None = None) -> float:
    """W_L: low-P, low-F branch weight (eq. 1, §2.2)."""
    wl, _ = window_weights(p_gpa, f_melt, eq1=eq1)
    return wl


def window_w_h(p_gpa: float, f_melt: float, eq1: Mapping | None = None) -> float:
    """W_H: high-P, high-F branch weight (eq. 1, §2.2)."""
    _, wh = window_weights(p_gpa, f_melt, eq1=eq1)
    return wh


def window_weights(
    p_gpa: float,
    f_melt: float,
    *,
    eq1: Mapping | None = None,
    w: Mapping | None = None,
) -> tuple[float, float]:
    """
    Product hyperbolic-tangent windows (KKHS02 §2.2).

    W_L = ¼ [1 − tanh(α(P − P_t))] [1 − tanh(β(F − F_t))]
    W_H = ¼ [1 + tanh(α(P − P_t))] [1 + tanh(β(F − F_t))]
    """
    eq1 = eq1 or load_eq1()
    w = w or eq1["equation_1"]["window_functions"]
    alpha = float(w["alpha"])
    beta = float(w["beta"])
    pt = float(w["P_t_GPa"])
    ft = float(w["F_t"])
    tp = alpha * (float(p_gpa) - pt)
    tf = beta * (float(f_melt) - ft)
    wl = 0.25 * (1.0 - math.tanh(tp)) * (1.0 - math.tanh(tf))
    wh = 0.25 * (1.0 + math.tanh(tp)) * (1.0 + math.tanh(tf))
    return wl, wh


def in_eq1_applicable_range(p_gpa: float, f_melt: float, eq1: Mapping | None = None) -> bool:
    """True when (P, F) lie inside the paper eq.(1) fit domain."""
    eq1 = eq1 or load_eq1()
    r = eq1["equation_1"]["applicable_range"]
    p_lo, p_hi = r["P_GPa"]
    f_lo, f_hi = r["F"]
    p = float(p_gpa)
    f = float(f_melt)
    return p_lo <= p <= p_hi and f_lo <= f <= f_hi


def predict_v_bulk_km_s(p_gpa: float, f_melt: float, eq1: Mapping | None = None) -> float:
    """
    Predict norm-based bulk crustal Vp (km/s) — KKHS02 eq.(1).

    Alias semantics: ``V_bulk`` at 600 MPa, 400 °C reference state.
    """
    eq1 = eq1 or load_eq1()
    c = eq1["equation_1"]["coefficients"]
    wl, wh = window_weights(p_gpa, f_melt, eq1=eq1)
    poly_b = branch_poly(p_gpa, f_melt, c, "b")
    poly_c = branch_poly(p_gpa, f_melt, c, "c")
    return float(c["a0"]) + wl * poly_b + wh * poly_c


def predict_v_bulk_grid(
    p_gpa: np.ndarray,
    f_melt: np.ndarray,
    *,
    eq1: Mapping | None = None,
    bias: float = 0.0,
) -> np.ndarray:
    """Vectorized eq.(1) on matching ``p_gpa`` / ``f_melt`` arrays (e.g. meshgrid)."""
    eq1 = eq1 or load_eq1()
    c = eq1["equation_1"]["coefficients"]
    w = eq1["equation_1"]["window_functions"]
    p = np.asarray(p_gpa, dtype=float)
    f = np.asarray(f_melt, dtype=float)
    alpha = float(w["alpha"])
    beta = float(w["beta"])
    pt = float(w["P_t_GPa"])
    ft = float(w["F_t"])
    tp = alpha * (p - pt)
    tf = beta * (f - ft)
    wl = 0.25 * (1.0 - np.tanh(tp)) * (1.0 - np.tanh(tf))
    wh = 0.25 * (1.0 + np.tanh(tp)) * (1.0 + np.tanh(tf))
    poly_b = (
        float(c["b0"])
        + float(c["b1"]) * p
        + float(c["b2"]) * f
        + float(c["b3"]) * p**2
        + float(c["b4"]) * p * f
        + float(c["b5"]) * f**2
    )
    poly_c = (
        float(c["c0"])
        + float(c["c1"]) * p
        + float(c["c2"]) * f
        + float(c["c3"]) * p**2
        + float(c["c4"]) * p * f
        + float(c["c5"]) * f**2
    )
    return float(c["a0"]) + wl * poly_b + wh * poly_c + float(bias)

def predict_vp_km_s(p_gpa: float, f_melt: float, eq1: Mapping | None = None) -> float:
    """Backward-compatible alias for :func:`predict_v_bulk_km_s`."""
    return predict_v_bulk_km_s(p_gpa, f_melt, eq1=eq1)


def predict_v_bulk_fig11_km_s(
    p_gpa: float,
    f_melt: float,
    *,
    eq1: Mapping | None = None,
) -> float:
    """
    Holbrook (2000) Ch3 eq.(3.1) linear surrogate — **not** used for KKHS02 Fig.11.

    Paper Fig.11 d/h uses windowed eq.(1) with §4 ``P_bar`` and ``F_bar`` only.
    Kept for cross-checks against catalog linear fits.
    """
    eq1 = eq1 or load_eq1()
    c = eq1["fig11_vbulk_linear"]["coefficients"]
    p = float(p_gpa)
    f = float(f_melt)
    return (
        float(c["c0"])
        + float(c["c1"]) * p
        + float(c["c2"]) * f
        + float(c["c3"]) * p * f
    )


# Legacy private names
_branch_poly = branch_poly
_window_weights = window_weights


def fit_vp_linear_pf(
    p_gpa: list[float] | np.ndarray,
    f_melt: list[float] | np.ndarray,
    vp_km_s: list[float] | np.ndarray,
) -> np.ndarray:
    """Least-squares Vp ≈ c0 + c1*P + c2*F + c3*P*F (MVP surrogate for Fig.3)."""
    p = np.asarray(p_gpa, dtype=float)
    f = np.asarray(f_melt, dtype=float)
    v = np.asarray(vp_km_s, dtype=float)
    x = np.column_stack([np.ones_like(p), p, f, p * f])
    coef, _, _, _ = np.linalg.lstsq(x, v, rcond=None)
    return coef


def predict_vp_linear_pf(p_gpa: float, f_melt: float, coef: np.ndarray) -> float:
    return float(coef @ np.array([1.0, p_gpa, f_melt, p_gpa * f_melt]))


def misfit_km_s(vp_obs: float, p_gpa: float, f_melt: float, eq1: Mapping | None = None) -> float:
    return vp_obs - predict_v_bulk_km_s(p_gpa, f_melt, eq1=eq1)


def catalog_eq1_bias(
    *,
    cipw_backend: str = "auto",
    mineral_backend: str = "auto",
    eq1: Mapping | None = None,
) -> dict:
    """Mean additive bias so paper eq.(1) matches catalog norm Vp (Fig.3 regression set)."""
    from petrology.data.load_catalog import load_melt_catalog
    from petrology.norm_velocity import norm_velocity_from_record

    eq1 = eq1 or load_eq1()
    rows = [r for r in load_melt_catalog() if r.get("include_in_regression")]
    resid = []
    for rec in rows:
        nv = norm_velocity_from_record(
            rec,
            cipw_backend=cipw_backend,
            mineral_backend=mineral_backend,
        )
        vp_obs = float(nv["vp_km_s"])
        vp_pred = predict_v_bulk_km_s(rec["P_melt_GPa"], rec["F_melt"], eq1=eq1)
        resid.append(vp_obs - vp_pred)

    resid = np.asarray(resid, dtype=float)
    bias = float(np.mean(resid))
    vp_pred_cal = np.array(
        [
            predict_v_bulk_km_s(r["P_melt_GPa"], r["F_melt"], eq1=eq1) + bias
            for r in rows
        ]
    )
    vp_obs = np.array(
        [
            float(
                norm_velocity_from_record(
                    r,
                    cipw_backend=cipw_backend,
                    mineral_backend=mineral_backend,
                )["vp_km_s"]
            )
            for r in rows
        ]
    )
    rms_raw = float(np.sqrt(np.mean(resid**2)))
    rms_cal = float(np.sqrt(np.mean((vp_obs - vp_pred_cal) ** 2)))
    morb_pred = predict_v_bulk_km_s(
        eq1["validation"]["morb_P_GPa"],
        eq1["validation"]["morb_F"],
        eq1=eq1,
    )
    return {
        "n_points": len(rows),
        "eq1_bias_km_s": bias,
        "rms_raw_km_s": rms_raw,
        "rms_calibrated_km_s": rms_cal,
        "mean_residual_raw_km_s": bias,
        "mean_residual_calibrated_km_s": float(np.mean(vp_obs - vp_pred_cal)),
        "paper_eq1_morb_km_s": float(morb_pred),
        "paper_eq1_morb_calibrated_km_s": float(morb_pred + bias),
        "expected_morb_km_s": float(eq1["validation"]["expected_Vp_km_s"]),
        "cipw_backend": cipw_backend,
        "mineral_backend": mineral_backend,
    }


def predict_vp_km_s_calibrated(
    p_gpa: float,
    f_melt: float,
    *,
    eq1: Mapping | None = None,
    eq1_bias_km_s: float | None = None,
    cipw_backend: str = "auto",
    mineral_backend: str = "auto",
) -> float:
    """Paper eq.(1) plus catalog mean bias (Fig.3 calibration)."""
    if eq1_bias_km_s is None:
        eq1_bias_km_s = catalog_eq1_bias(
            cipw_backend=cipw_backend,
            mineral_backend=mineral_backend,
            eq1=eq1,
        )["eq1_bias_km_s"]
    return predict_v_bulk_km_s(p_gpa, f_melt, eq1=eq1) + float(eq1_bias_km_s)
