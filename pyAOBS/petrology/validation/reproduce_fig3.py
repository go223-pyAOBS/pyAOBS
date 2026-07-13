"""Reproduce KKHS02 Fig.3: norm Vp vs equation (1) prediction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.data.load_catalog import load_melt_catalog
from petrology.norm_velocity import norm_velocity_from_record
from petrology.vp_regression import (
    catalog_eq1_bias,
    fit_vp_linear_pf,
    load_eq1,
    predict_vp_km_s,
    predict_vp_linear_pf,
)


def run(
    *,
    save_figure: Path | None = None,
    show: bool = False,
    use_paper_eq1: bool = True,
    cipw_backend: str = "auto",
    mineral_backend: str = "auto",
    calibrated_plot: bool = False,
) -> dict:
    eq1 = load_eq1()
    rows = load_melt_catalog()
    reg = [r for r in rows if r.get("include_in_regression")]

    vp_obs = []
    p_list = []
    f_list = []
    labels = []
    styles = []

    for rec in reg:
        nv = norm_velocity_from_record(
            rec,
            cipw_backend=cipw_backend,
            mineral_backend=mineral_backend,
        )
        vp_obs.append(nv["vp_km_s"])
        p_list.append(rec["P_melt_GPa"])
        f_list.append(rec["F_melt"])
        labels.append(rec["id"])
        styles.append(rec.get("melt_style", ""))

    vp_obs = np.array(vp_obs)
    fit_coef = fit_vp_linear_pf(p_list, f_list, vp_obs)
    vp_pred = np.array(
        [
            predict_vp_km_s(p, f, eq1=eq1) if use_paper_eq1 else predict_vp_linear_pf(p, f, fit_coef)
            for p, f in zip(p_list, f_list)
        ]
    )
    resid = vp_obs - vp_pred

    paper_morb = predict_vp_km_s(
        eq1["validation"]["morb_P_GPa"],
        eq1["validation"]["morb_F"],
        eq1=eq1,
    )

    summary = {
        "n_points": len(reg),
        "vp_obs_range": (float(vp_obs.min()), float(vp_obs.max())),
        "vp_pred_range": (float(vp_pred.min()), float(vp_pred.max())),
        "rms_km_s": float(np.sqrt(np.mean(resid**2))),
        "mean_residual_km_s": float(np.mean(resid)),
        "predictor": "paper_eq1" if use_paper_eq1 else "catalog_linear_pf",
        "cipw_backend": cipw_backend,
        "mineral_backend": mineral_backend,
        "paper_eq1_morb_km_s": paper_morb,
        "norm_morb_km_s": float(
            norm_velocity_from_record(
                next(r for r in reg if r["id"] == "k97_grid_P1_F0.1"),
                cipw_backend=cipw_backend,
                mineral_backend=mineral_backend,
            )["vp_km_s"]
        ),
    }

    if use_paper_eq1:
        cal = catalog_eq1_bias(
            cipw_backend=cipw_backend,
            mineral_backend=mineral_backend,
            eq1=eq1,
        )
        summary["eq1_catalog_bias_km_s"] = cal["eq1_bias_km_s"]
        summary["rms_calibrated_km_s"] = cal["rms_calibrated_km_s"]
        summary["paper_eq1_morb_calibrated_km_s"] = cal["paper_eq1_morb_calibrated_km_s"]
        summary["expected_morb_km_s"] = cal["expected_morb_km_s"]

    print(f"Catalog points (regression): {summary['n_points']}")
    print(f"Norm Vp range: {summary['vp_obs_range'][0]:.3f} – {summary['vp_obs_range'][1]:.3f} km/s")
    print(f"Predictor: {summary['predictor']}")
    print(f"Pred Vp range: {summary['vp_pred_range'][0]:.3f} – {summary['vp_pred_range'][1]:.3f} km/s")
    print(f"RMS misfit: {summary['rms_km_s']:.3f} km/s")
    print(f"Norm Vp @ P=1, F=0.1 (grid): {summary['norm_morb_km_s']:.3f} km/s")
    print(f"Paper eq.(1) @ MORB (1 GPa, F=0.1): {summary['paper_eq1_morb_km_s']:.3f} km/s (expect ~7.1)")
    if use_paper_eq1:
        bias = summary["eq1_catalog_bias_km_s"]
        print(
            f"Catalog mean bias (obs - pred): {bias:+.3f} km/s  "
            f"-> calibrated RMS {summary['rms_calibrated_km_s']:.3f} km/s"
        )
        print(
            f"Calibrated eq.(1) @ MORB: {summary['paper_eq1_morb_calibrated_km_s']:.3f} km/s "
            f"(expect {summary['expected_morb_km_s']:.1f})"
        )
        print("Use predict_vp_km_s_calibrated() or add eq1_bias_km_s to forward eq.(1) paths.")
        print("Use --linear-surrogate for catalog-only linear P-F fit (lower RMS vs this MVP norm Vp).")
    else:
        summary["linear_coef"] = fit_coef.tolist()
        print(f"Linear surrogate coef: {fit_coef.tolist()}")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable ({exc}) — skipping figure")
        return summary

    try:
        fig, ax = plt.subplots(figsize=(6, 5))
    except Exception as exc:
        print(f"matplotlib init failed ({exc}) — skipping figure")
        return summary

    calc = np.array([s == "calculated" for s in styles])
    x_pred = vp_pred.copy()
    if use_paper_eq1 and calibrated_plot and "eq1_catalog_bias_km_s" in summary:
        x_pred = x_pred + float(summary["eq1_catalog_bias_km_s"])
        xlabel = "Vp predicted + catalog bias (km/s)"
    else:
        xlabel = "Vp predicted (km/s)"
    ax.scatter(
        x_pred[~calc],
        vp_obs[~calc],
        c="C0",
        label="experimental / aggregated",
        alpha=0.85,
        edgecolors="k",
        linewidths=0.3,
    )
    ax.scatter(
        x_pred[calc],
        vp_obs[calc],
        c="C1",
        marker="s",
        s=18,
        label="Kinzler1997 grid",
        alpha=0.7,
    )
    lim = (
        min(vp_obs.min(), x_pred.min()) - 0.05,
        max(vp_obs.max(), x_pred.max()) + 0.05,
    )
    ax.plot(lim, lim, "k--", lw=0.8, label="1:1")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Vp observed — norm @ 600 MPa, 400°C (km/s)")
    title = "KKHS02 Fig.3 reproduction (MVP)"
    if not use_paper_eq1:
        title += " [catalog linear]"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()

    if save_figure:
        save_figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_figure, dpi=150)
        print(f"Saved {save_figure}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce KKHS02 Fig.3 scatter (MVP)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures" / "reproduce_fig3_mvp.png",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--cipw-backend",
        choices=("auto", "pyrolite", "fallback"),
        default="auto",
    )
    parser.add_argument(
        "--mineral-backend",
        choices=("auto", "burnman", "empirical"),
        default="auto",
    )
    parser.add_argument(
        "--linear-surrogate",
        action="store_true",
        help="Use catalog linear P–F fit instead of paper eq.(1)",
    )
    parser.add_argument(
        "--calibrated-plot",
        action="store_true",
        help="Overlay calibrated eq.(1) predictions on scatter (paper eq1 only)",
    )
    args = parser.parse_args()
    run(
        save_figure=args.output,
        show=args.show,
        use_paper_eq1=not args.linear_surrogate,
        cipw_backend=args.cipw_backend,
        mineral_backend=args.mineral_backend,
        calibrated_plot=args.calibrated_plot,
    )


if __name__ == "__main__":
    main()
