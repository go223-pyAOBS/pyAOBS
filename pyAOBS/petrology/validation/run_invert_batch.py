"""Batch Step-2 inversion for multiple observations from CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.invert import invert_observation_rows


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run(
    *,
    input_csv: Path,
    output_csv: Path | None,
    output_fig: Path | None,
    v_lc_key: str,
    f_lower_key: str,
    label_field: str,
    f_solid: float,
    p_fc_mpa: float,
    vp_bias_km_s: float,
    vp_tolerance_km_s: float,
    show: bool,
) -> list[dict]:
    rows = _read_rows(input_csv)
    result = invert_observation_rows(
        rows,
        v_lc_key=v_lc_key,
        f_lower_key=f_lower_key,
        f_solid=f_solid,
        p_fc_mpa=p_fc_mpa,
        vp_bias_km_s=vp_bias_km_s,
        vp_tolerance_km_s=vp_tolerance_km_s,
    )
    n_total = len(result)
    n_ok = sum(int(r["n_feasible"] > 0) for r in result)
    print(f"Rows: {n_total}; feasible: {n_ok}; empty: {n_total - n_ok}")
    if n_ok > 0:
        p_means = [float(r["p_mean_gpa"]) for r in result if r["p_mean_gpa"] is not None]
        f_means = [float(r["f_mean"]) for r in result if r["f_mean"] is not None]
        print(f"Mean feasible P: {sum(p_means)/len(p_means):.3f} GPa")
        print(f"Mean feasible F: {sum(f_means)/len(f_means):.4f}")

    if output_csv:
        _write_rows(output_csv, result)
        print(f"Wrote {output_csv}")

    if output_fig:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"matplotlib unavailable ({exc}) — skipping figure")
            return result

        ok = [r for r in result if r["n_feasible"] > 0]
        fig, ax = plt.subplots(figsize=(6, 5))
        if len(ok) > 0:
            x = [float(r["v_lc_obs_km_s"]) for r in ok]
            y = [float(r["p_mean_gpa"]) for r in ok]
            c = [float(r["f_mean"]) for r in ok]
            sc = ax.scatter(
                x,
                y,
                c=c,
                cmap="viridis",
                s=70,
                edgecolors="k",
                linewidths=0.5,
                label="feasible",
                zorder=3,
            )
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label("Mean feasible F")
            for r in ok:
                label = str(r.get(label_field, r["id"]))
                ax.annotate(
                    label,
                    (float(r["v_lc_obs_km_s"]), float(r["p_mean_gpa"])),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=8,
                )

        bad = [r for r in result if r["n_feasible"] <= 0]
        if len(bad) > 0:
            xb = [float(r["v_lc_obs_km_s"]) for r in bad]
            yb = [
                float(r["nearest_p_gpa"])
                if r.get("nearest_p_gpa") is not None
                else np.nan
                for r in bad
            ]
            ax.scatter(
                xb,
                yb,
                marker="x",
                c="0.5",
                s=60,
                linewidths=1.2,
                label="non-feasible",
                zorder=2,
            )
            for r, xbi, ybi in zip(bad, xb, yb):
                label = str(r.get(label_field, r["id"]))
                ax.annotate(
                    label,
                    (xbi, ybi),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=8,
                    color="0.35",
                )

        ax.set_xlabel("Observed V_LC (km/s)")
        ax.set_ylabel("Mean feasible P (GPa)")
        ax.set_title("Batch inversion summary (MVP)")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        output_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_fig, dpi=150)
        print(f"Saved {output_fig}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch run Step-2 invert MVP")
    parser.add_argument("input_csv", type=Path, help="CSV with columns: id,v_lc_obs_km_s,f_lower")
    parser.add_argument("--v-lc-key", default="v_lc_obs_km_s")
    parser.add_argument("--f-lower-key", default="f_lower")
    parser.add_argument(
        "--label-field",
        default="id",
        help="CSV field used for point labels on summary figure",
    )
    parser.add_argument("--f-solid", type=float, default=0.75)
    parser.add_argument("--p-fc-mpa", type=float, default=400.0)
    parser.add_argument("--vp-bias", type=float, default=-0.10)
    parser.add_argument("--vp-tol", type=float, default=0.05)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures" / "invert_batch_results.csv",
    )
    parser.add_argument(
        "--output-fig",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures" / "invert_batch_summary.png",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    run(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        output_fig=args.output_fig,
        v_lc_key=args.v_lc_key,
        f_lower_key=args.f_lower_key,
        label_field=args.label_field,
        f_solid=args.f_solid,
        p_fc_mpa=args.p_fc_mpa,
        vp_bias_km_s=args.vp_bias,
        vp_tolerance_km_s=args.vp_tol,
        show=args.show,
    )


if __name__ == "__main__":
    main()

