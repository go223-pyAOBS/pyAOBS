"""
Overlay catalog FC simulation on digitized KKHS02 Fig.5 (a–d).

Paper layout: **x = V_LC or V_UC, y = ΔVp** @ 600 MPa, 400°C.
Primary validation: panels a/b (P_fc = 100 / 400 MPa, VLC).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.data.load_catalog import load_melt_catalog, oxides_from_record
from petrology.fc.delta_vp import R3_DELTA_VP_WL_KW
from petrology.fc.figure05_digitized import (
    PanelKey,
    f_loci_curve,
    f_loci_curve_raw,
    get_panel,
    load_figure05_digitized,
    panel_p_fc_mpa,
    panel_phase,
    residual_to_f_loci,
    sigma_delta_vp_at_f,
)
from petrology.fractionation import delta_vp_km_s, theoretical_upper_crust_vp_km_s
from petrology.norm_velocity import norm_velocity_from_record

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

PANEL_MAP: dict[str, PanelKey] = {
    "a": "a_vlc_100",
    "b": "b_vlc_400",
    "c": "c_vuc_100",
    "d": "d_vuc_400",
}


def _simulate_paths(
    *,
    p_fc_mpa: float,
    phase: str,
    f_grid: np.ndarray,
    mineral_backend: str,
    kd_engine: str,
    fig5_dvp_calibrate: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = [r for r in load_melt_catalog() if r.get("include_in_regression")]
    d_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    ids: list[str] = []
    for rec in rows:
        bulk = float(
            norm_velocity_from_record(rec, mineral_backend=mineral_backend)["vp_km_s"]
        )
        ox = {k: v for k, v in oxides_from_record(rec).items() if k != "P2O5"}
        if phase == "vlc":
            d = np.array(
                [
                    delta_vp_km_s(
                        f,
                        bulk_vp_km_s=bulk,
                        p_fc_mpa=p_fc_mpa,
                        engine="wl1990",
                        melt_oxides_wt=ox,
                        kd_engine=kd_engine,
                        mineral_backend=mineral_backend,
                        fig5_dvp_calibrate=fig5_dvp_calibrate,
                    )
                    for f in f_grid
                ]
            )
            y = bulk + d
        else:
            d = np.array(
                [
                    delta_vp_km_s(
                        f,
                        bulk_vp_km_s=bulk,
                        p_fc_mpa=p_fc_mpa,
                        engine="wl1990",
                        melt_oxides_wt=ox,
                        kd_engine=kd_engine,
                        mineral_backend=mineral_backend,
                        fig5_dvp_calibrate=fig5_dvp_calibrate,
                    )
                    for f in f_grid
                ]
            )
            vuc = np.array([theoretical_upper_crust_vp_km_s(bulk, f) for f in f_grid])
            y = vuc
        d_all.append(d)
        y_all.append(y)
        ids.append(str(rec["id"]))
    return np.asarray(d_all), np.asarray(y_all), ids


def _marker_slice(f_grid: np.ndarray, f_target: float) -> int:
    return int(np.argmin(np.abs(f_grid - f_target)))


def run(
    *,
    panels: tuple[str, ...] = ("a", "b", "c", "d"),
    f_grid: np.ndarray | None = None,
    mineral_backend: str = "sb1994_fig2ol",
    kd_engine: str = "langmuir",
    fig5_dvp_calibrate: bool = True,
    save_figure: Path | None = None,
    show: bool = False,
) -> dict:
    data = load_figure05_digitized()
    f_grid = np.asarray(f_grid if f_grid is not None else np.linspace(0.05, 0.95, 37), dtype=float)
    provisional = bool(data["meta"].get("provisional", True))

    summary: dict = {"provisional_digitized": provisional, "panels": {}}

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable ({exc})")
        return summary

    active = [p for p in panels if p in PANEL_MAP]
    n = len(active)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.8 * nrows), squeeze=False)

    for idx, letter in enumerate(active):
        pkey = PANEL_MAP[letter]
        ax = axes[idx // ncols][idx % ncols]
        panel = get_panel(pkey, data)
        p_fc = panel_p_fc_mpa(pkey, data)
        phase = panel_phase(pkey, data)
        ax_lim = panel["axes"]
        x_label = "V_LC (km/s)" if phase == "vlc" else "V_UC (km/s)"

        d_paths, y_paths, _ids = _simulate_paths(
            p_fc_mpa=p_fc,
            phase=phase,
            f_grid=f_grid,
            mineral_backend=mineral_backend,
            kd_engine=kd_engine,
            fig5_dvp_calibrate=fig5_dvp_calibrate,
        )

        # Paper axes: x = V_y, y = ΔVp
        for d, y in zip(d_paths, y_paths):
            ax.plot(y, d, color="0.55", lw=0.6, alpha=0.45, zorder=1)

        f_loci_colors = {0.5: "C0", 0.6: "C1", 0.7: "C2", 0.8: "C3"}
        for f_t, color in f_loci_colors.items():
            raw = f_loci_curve_raw(pkey, f_t, data=data)
            if raw:
                ax.scatter(
                    [p[0] for p in raw],
                    [p[1] for p in raw],
                    s=12,
                    color=color,
                    alpha=0.35,
                    zorder=2,
                )
            curve = f_loci_curve(pkey, f_t, data=data)
            if curve:
                ax.plot(
                    [p[0] for p in curve],
                    [p[1] for p in curve],
                    color=color,
                    lw=2.0,
                    zorder=5,
                    label=f"paper F={f_t:g} (fit)" if idx == 0 else None,
                )

        for f_t, mfc, ms in ((0.5, "none", 55), (0.8, "C3", 45)):
            j = _marker_slice(f_grid, f_t)
            ax.scatter(
                y_paths[:, j],
                d_paths[:, j],
                s=ms,
                facecolors=mfc,
                edgecolors="0.15",
                linewidths=0.7,
                zorder=3,
                label=f"sim F={f_t:g}" if idx == 0 else None,
            )

        # Residual vs digitized F-loci @ F=0.5–0.8
        stats: dict = {}
        for f_t in (0.5, 0.6, 0.7, 0.8):
            j = _marker_slice(f_grid, f_t)
            dd = []
            for d, y in zip(d_paths[:, j], y_paths[:, j]):
                d_dvp, _ = residual_to_f_loci(d, y, pkey, f_t, data=data)
                if not np.isnan(d_dvp):
                    dd.append(d_dvp)
            if dd:
                stats[f"f{f_t}_dvp_bias"] = float(np.mean(dd))
                stats[f"f{f_t}_dvp_rms"] = float(np.sqrt(np.mean(np.square(dd))))
                if f_t in (0.5, 0.8):
                    stats[f"f{f_t}_sigma_paper"] = sigma_delta_vp_at_f(pkey, f_t, data=data)

        summary["panels"][pkey] = stats
        title = panel.get("label", pkey)
        if provisional:
            title += " [provisional envelope]"
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel("ΔVp = V − V_bulk (km/s)")
        ax.set_xlim(ax_lim["x_lim"])
        ax.set_ylim(ax_lim["y_lim"])
        ax.grid(True, ls=":", lw=0.4, alpha=0.5)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=7)

        print(f"{pkey} @ P_fc={p_fc:.0f} MPa ({phase})")
        for k, v in stats.items():
            print(f"  {k}: {v:+.4f}" if "bias" in k or "rms" in k else f"  {k}: {v:.4f}")

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(
        f"Fig.5 compare — wl1990 ({mineral_backend}, {kd_engine}, "
        f"fig5_cal={'on' if fig5_dvp_calibrate else 'off'})",
        fontsize=11,
    )
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
    parser = argparse.ArgumentParser(description="Compare FC simulation vs digitized Fig.5")
    parser.add_argument(
        "--panels",
        type=str,
        default="a,b,c,d",
        help="Comma-separated panel letters (default a,b,c,d; prioritize a,b)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=FIG_DIR / "compare_figure05_digitized.png",
    )
    parser.add_argument("--mineral-backend", default="sb1994_fig2ol")
    parser.add_argument("--kd-engine", default="langmuir")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    letters = tuple(x.strip() for x in args.panels.split(",") if x.strip())
    run(
        panels=letters,
        mineral_backend=args.mineral_backend,
        kd_engine=args.kd_engine,
        save_figure=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
