"""
Fig.15a→c: seismic transect → P–T correction → (H, V_LC) on Fig.12a.

Pipeline (KKHS02 §5 SIGMA transect 2):
  1. In-situ lower-crust Vp samples + whole-crust H along profile
  2. P–T correct each sample to 600 MPa, 400 °C (Fig.7 derivatives)
  3. Harmonic mean in lower crust; 20 km window @ 10 km step; 100 MC
  4. Plot open circles (H, V_LC) on digitized Fig.12a — **no ΔVp shift**
  5. Thick crust only (H > 15 km): read V_bulk,model ≤ V_LC; ΔVp band for interpretation

Usage::

    py -3.11 petrology/validation/reproduce_fig15_transect.py
    py -3.11 petrology/validation/reproduce_fig15_transect.py --write-demo-csv
    py -3.11 petrology/validation/reproduce_fig15_transect.py --show
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.seismic.reference_state import (
    conductive_geotherm_c,
    correct_vp_to_reference_km_s,
    lithostatic_pressure_mpa,
)
from petrology.seismic.transect import (
    TransectWindow,
    aggregate_transect_windows,
    load_transect_csv,
    prepare_samples,
)
from petrology.validation.fig12a_background import DEFAULT_HVP_DIGITIZED

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_TRANSECT_CSV = DATA_DIR / "greenland_transect2_demo.csv"


def write_demo_transect_csv(path: Path, *, rng_seed: int = 42) -> None:
    """Synthetic SIGMA-like profile: thick landward crust, thin seaward."""
    rng = np.random.default_rng(rng_seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []

    for dist in np.arange(0.0, 105.0, 5.0):
        if dist < 65.0:
            h = 30.0 - 0.015 * dist + rng.normal(0.0, 0.4)
            v_ref_target = 7.02 - 0.0008 * dist + rng.normal(0.0, 0.008)
        else:
            h = max(8.0, 20.0 - 0.22 * (dist - 65.0) + rng.normal(0.0, 0.5))
            v_ref_target = 6.88 + rng.normal(0.0, 0.025)

        h = float(np.clip(h, 7.0, 34.0))
        f_lower = 0.7
        z_top = (1.0 - f_lower) * h
        depths = np.linspace(z_top + 0.4, h - 0.3, 8)

        for z in depths:
            v_ref = v_ref_target + rng.normal(0.0, 0.012)
            p = lithostatic_pressure_mpa(z)
            t = conductive_geotherm_c(z)
            v_insitu = v_ref - 0.0002 * (600.0 - p) + 0.0004 * (400.0 - t)
            rows.append(
                {
                    "distance_km": float(dist),
                    "depth_km": float(z),
                    "vp_insitu_km_s": float(v_insitu),
                    "h_whole_km": h,
                }
            )

    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["distance_km", "depth_km", "vp_insitu_km_s", "h_whole_km"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote demo transect ({len(rows)} samples) → {path}")


def export_windows_csv(path: Path, windows: list[TransectWindow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from petrology.seismic.transect import export_transect_windows_csv

    export_transect_windows_csv(path, windows)
    print(f"Exported {len(windows)} windows → {path}")


def plot_fig15_composite(
    windows: list[TransectWindow],
    *,
    fig12a_file: Path,
    save_figure: Path | None,
    show: bool,
    delta_vp_max_km_s: float,
    h_min_km: float,
    window_half_km: float,
    distance_step_km: float,
    n_mc: int,
    transect_label: str = "SIGMA transect 2",
) -> None:
    from petrology.hvp.fig15_composite import create_fig15_composite_figure
    import matplotlib.pyplot as plt

    fig, _axes = create_fig15_composite_figure(
        windows,
        fig12a_file=fig12a_file,
        transect_label=transect_label,
        window_half_width_km=window_half_km,
        distance_step_km=distance_step_km,
        n_mc=n_mc,
        delta_vp_max_km_s=delta_vp_max_km_s,
        h_min_km=h_min_km,
    )

    if save_figure:
        save_figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_figure, dpi=160)
        print(f"Saved {save_figure}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def run(
    *,
    transect_csv: Path,
    fig12a_file: Path = DEFAULT_HVP_DIGITIZED,
    save_figure: Path | None = None,
    export_windows: Path | None = None,
    show: bool = False,
    window_half_km: float = 10.0,
    distance_step_km: float = 10.0,
    h_min_km: float = 15.0,
    f_lower: float = 0.7,
    delta_vp_max_km_s: float = 0.15,
    n_mc: int = 100,
    vp_pick_sigma_km_s: float = 0.0,
) -> dict:
    rows = load_transect_csv(transect_csv)
    samples = prepare_samples(rows, f_lower=f_lower)
    windows = aggregate_transect_windows(
        samples,
        window_half_width_km=window_half_km,
        distance_step_km=distance_step_km,
        h_min_km=h_min_km,
        f_lower=f_lower,
        delta_vp_max_km_s=delta_vp_max_km_s,
        n_mc=n_mc,
        vp_pick_sigma_km_s=vp_pick_sigma_km_s,
    )

    n_thick = sum(1 for w in windows if w.thick_crust)
    print(f"Transect samples: {len(samples)} → {len(windows)} windows ({n_thick} thick-crust)")
    for w in windows:
        flag = "thick" if w.thick_crust else "thin "
        print(
            f"  [{flag}] dist={w.distance_km:5.0f} km  H={w.h_whole_km:5.1f}  "
            f"V_LC={w.v_lc_km_s:.3f}±{w.v_lc_sigma_km_s:.3f}  "
            f"bulk∈[{w.v_bulk_lower_km_s:.3f},{w.v_bulk_upper_km_s:.3f}]"
        )

    if export_windows:
        export_windows_csv(export_windows, windows)

    plot_fig15_composite(
        windows,
        fig12a_file=fig12a_file,
        save_figure=save_figure,
        show=show,
        delta_vp_max_km_s=delta_vp_max_km_s,
        h_min_km=h_min_km,
        window_half_km=window_half_km,
        distance_step_km=distance_step_km,
        n_mc=n_mc,
    )

    return {
        "n_samples": len(samples),
        "n_windows": len(windows),
        "n_thick": n_thick,
        "transect_csv": str(transect_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fig.15 transect → Fig.12a overlay")
    parser.add_argument("--transect-csv", type=Path, default=DEFAULT_TRANSECT_CSV)
    parser.add_argument("--fig12a-data", type=Path, default=DEFAULT_HVP_DIGITIZED)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=FIG_DIR / "fig15_abc_transect.png",
    )
    parser.add_argument(
        "--export-windows",
        type=Path,
        default=DATA_DIR / "greenland_transect2_windows.csv",
    )
    parser.add_argument("--write-demo-csv", action="store_true")
    parser.add_argument("--window-half-km", type=float, default=10.0)
    parser.add_argument("--distance-step-km", type=float, default=10.0)
    parser.add_argument("--h-min-km", type=float, default=15.0)
    parser.add_argument("--f-lower", type=float, default=0.7)
    parser.add_argument("--delta-vp-max", type=float, default=0.15)
    parser.add_argument("--n-mc", type=int, default=100)
    parser.add_argument("--vp-pick-sigma", type=float, default=0.0)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.write_demo_csv or not args.transect_csv.is_file():
        write_demo_transect_csv(args.transect_csv)

    run(
        transect_csv=args.transect_csv,
        fig12a_file=args.fig12a_data,
        save_figure=args.output,
        export_windows=args.export_windows,
        show=args.show,
        window_half_km=args.window_half_km,
        distance_step_km=args.distance_step_km,
        h_min_km=args.h_min_km,
        f_lower=args.f_lower,
        delta_vp_max_km_s=args.delta_vp_max,
        n_mc=args.n_mc,
        vp_pick_sigma_km_s=args.vp_pick_sigma,
    )


if __name__ == "__main__":
    main()
