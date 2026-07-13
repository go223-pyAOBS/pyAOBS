"""
Reproduce KKHS02 Fig.5 layout (paper caption).

Panels (report state 600 MPa / 400 °C)::

  (a) V_LC vs ΔVp = V_LC − V_bulk, P_fc = 100 MPa
  (b) same, P_fc = 400 MPa
  (c) V_UC vs ΔVp = V_bulk − V_UC, P_fc = 100 MPa
      (V_UC = norm Vp of residual liquid as solid rock)
  (d) same, P_fc = 400 MPa

Thin lines = FC paths for catalog melts; open / filled markers = F_xl = 0.5 / 0.8.
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
from petrology.fc.delta_vp import FIG5_P_EVAL_MPA, FIG5_T_EVAL_C, R3_DELTA_VP_WL_KW
from petrology.fc.wl1990 import simulate_crystallization_path
from petrology.norm_velocity import norm_velocity_from_record

# Paper Fig.5 axis windows (digitized / caption)
_AX_VLC = {"x_lim": (6.8, 7.8), "y_lim": (0.0, 0.40), "xlabel": r"$V_{\mathrm{LC}}$ (km/s)"}
_AX_VUC = {"x_lim": (6.3, 7.3), "y_lim": (0.0, 0.45), "xlabel": r"$V_{\mathrm{UC}}$ (km/s)"}

_PANELS: tuple[tuple[str, float, str], ...] = (
    ("a", 100.0, "vlc"),
    ("b", 400.0, "vlc"),
    ("c", 100.0, "vuc"),
    ("d", 400.0, "vuc"),
)


def _marker_index(f_grid: np.ndarray, f_target: float) -> int:
    return int(np.argmin(np.abs(f_grid - f_target)))


def run(
    *,
    f_grid: np.ndarray | None = None,
    mineral_backend: str | None = None,
    kd_engine: str | None = None,
    p_eval_mpa: float = FIG5_P_EVAL_MPA,
    t_eval_c: float = FIG5_T_EVAL_C,
    save_figure: Path | None = None,
    show: bool = False,
) -> dict:
    rows = [r for r in load_melt_catalog() if r.get("include_in_regression")]
    f_grid = np.asarray(
        f_grid if f_grid is not None else np.linspace(0.05, 0.95, 19),
        dtype=float,
    )
    mb = str(mineral_backend or R3_DELTA_VP_WL_KW["mineral_backend"])
    kd = str(kd_engine or R3_DELTA_VP_WL_KW["kd_engine"])

    bulks: list[float] = []
    melts: list[dict[str, float]] = []
    for rec in rows:
        bulks.append(
            float(
                norm_velocity_from_record(
                    rec,
                    p_pa=float(p_eval_mpa) * 1e6,
                    t_k=float(t_eval_c) + 273.15,
                    mineral_backend=mb,  # type: ignore[arg-type]
                )["vp_km_s"]
            )
        )
        melts.append({k: v for k, v in oxides_from_record(rec).items() if k != "P2O5"})

    # Cache paths per P_fc: list of (V, ΔVp) arrays
    cache: dict[float, list[tuple[np.ndarray, np.ndarray]]] = {}
    for p_fc in (100.0, 400.0):
        paths = []
        for bulk, ox in zip(bulks, melts):
            # VLC path; VUC reuses same FC states via phase switch in _path_xy
            # Call once for vlc and once for vuc would double cost — compute both from one sim.
            states = simulate_crystallization_path(
                primary_melt_oxides_wt=ox,
                f_grid=f_grid,
                path="fc_100",
                p_fc_mpa=p_fc,
                p_eval_mpa=float(p_eval_mpa),
                t_eval_c=float(t_eval_c),
                mineral_backend=mb,  # type: ignore[arg-type]
                kd_engine=kd,  # type: ignore[arg-type]
            )
            by_f = {float(s.f_solid): s for s in states}
            v_lc = np.empty(len(f_grid))
            d_lc = np.empty(len(f_grid))
            v_uc = np.empty(len(f_grid))
            d_uc = np.empty(len(f_grid))
            for i, f in enumerate(f_grid):
                s = by_f.get(float(f)) or by_f[min(by_f.keys(), key=lambda k: abs(k - float(f)))]
                vlc = float(s.vp_cumulate_km_s)
                vuc = float(s.vp_residual_norm_km_s)
                v_lc[i] = vlc
                d_lc[i] = max(0.0, vlc - bulk)
                v_uc[i] = vuc
                d_uc[i] = max(0.0, bulk - vuc)
            paths.append((v_lc, d_lc, v_uc, d_uc))
        cache[p_fc] = paths  # type: ignore[assignment]

    i05 = _marker_index(f_grid, 0.5)
    i08 = _marker_index(f_grid, 0.8)

    # Summary @ F=0.7–0.8 for VLC @ 400 MPa (Step-2 band)
    d_band: list[float] = []
    for v_lc, d_lc, _v_uc, _d_uc in cache[400.0]:  # type: ignore[misc]
        for j, f in enumerate(f_grid):
            if 0.7 <= f <= 0.8:
                d_band.append(float(d_lc[j]))
    summary = {
        "n_points": len(rows),
        "mineral_backend": mb,
        "kd_engine": kd,
        "p_eval_mpa": float(p_eval_mpa),
        "t_eval_c": float(t_eval_c),
        "f_grid": [float(x) for x in f_grid],
        "delta_vp_f07_08_mean_km_s": float(np.mean(d_band)) if d_band else float("nan"),
        "delta_vp_f07_08_std_km_s": float(np.std(d_band)) if d_band else float("nan"),
        "layout": "paper_fig5_abcd",
    }
    print(f"Catalog melts: {summary['n_points']}  ({mb}, {kd})")
    print(f"Report state: {p_eval_mpa:.0f} MPa / {t_eval_c:.0f} C")
    print(
        "ΔVp_LC @ F=0.7–0.8 (P_fc=400): "
        f"{summary['delta_vp_f07_08_mean_km_s']:.3f} ± {summary['delta_vp_f07_08_std_km_s']:.3f} km/s"
    )

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable ({exc}) — skipping figure")
        return summary

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.0), sharex=False, sharey=False)
    letter_axes = {
        "a": axes[0, 0],
        "b": axes[0, 1],
        "c": axes[1, 0],
        "d": axes[1, 1],
    }

    for letter, p_fc, phase in _PANELS:
        ax = letter_axes[letter]
        ax_meta = _AX_VLC if phase == "vlc" else _AX_VUC
        paths = cache[p_fc]
        for trip in paths:
            v_lc, d_lc, v_uc, d_uc = trip  # type: ignore[misc]
            if phase == "vlc":
                vx, dvp = v_lc, d_lc
            else:
                vx, dvp = v_uc, d_uc
            ax.plot(vx, dvp, color="0.55", lw=0.55, alpha=0.45, zorder=1)

        # F=0.5 open, F=0.8 filled + quadratic loci (paper caption)
        x05, y05, x08, y08 = [], [], [], []
        for trip in paths:
            v_lc, d_lc, v_uc, d_uc = trip  # type: ignore[misc]
            vx, dvp = (v_lc, d_lc) if phase == "vlc" else (v_uc, d_uc)
            x05.append(vx[i05])
            y05.append(dvp[i05])
            x08.append(vx[i08])
            y08.append(dvp[i08])
        x05_a = np.asarray(x05, dtype=float)
        y05_a = np.asarray(y05, dtype=float)
        x08_a = np.asarray(x08, dtype=float)
        y08_a = np.asarray(y08, dtype=float)

        def _quad_locus(x: np.ndarray, y: np.ndarray, color: str, label: str | None) -> float:
            ok = np.isfinite(x) & np.isfinite(y)
            if int(ok.sum()) < 4:
                return float("nan")
            xx, yy = x[ok], y[ok]
            coef = np.polyfit(xx, yy, 2)
            xs = np.linspace(float(xx.min()), float(xx.max()), 80)
            ys = np.polyval(coef, xs)
            resid = yy - np.polyval(coef, xx)
            sigma = float(np.std(resid, ddof=1)) if len(resid) > 2 else float("nan")
            ax.plot(xs, ys, color=color, lw=1.6, zorder=2, label=label)
            if np.isfinite(sigma):
                ax.fill_between(xs, ys - sigma, ys + sigma, color=color, alpha=0.12, zorder=1)
            return sigma

        sig05 = _quad_locus(x05_a, y05_a, "C0", r"quad $F=0.5$" if letter == "a" else None)
        sig08 = _quad_locus(x08_a, y08_a, "C3", r"quad $F=0.8$" if letter == "a" else None)
        print(f"  ({letter}) 1σ residual @ F=0.5: {sig05:.3f}  @ F=0.8: {sig08:.3f} km/s")

        ax.scatter(
            x05_a,
            y05_a,
            s=36,
            facecolors="none",
            edgecolors="0.1",
            linewidths=0.8,
            zorder=3,
            label=r"$F_{\mathrm{xl}}=0.5$",
        )
        ax.scatter(
            x08_a,
            y08_a,
            s=28,
            c="0.15",
            edgecolors="0.1",
            linewidths=0.4,
            zorder=3,
            label=r"$F_{\mathrm{xl}}=0.8$",
        )

        ylabel = (
            r"$\Delta V_p = V_{\mathrm{LC}} - V_{\mathrm{bulk}}$ (km/s)"
            if phase == "vlc"
            else r"$\Delta V_p = V_{\mathrm{bulk}} - V_{\mathrm{UC}}$ (km/s)"
        )
        ax.set_xlabel(ax_meta["xlabel"])
        ax.set_ylabel(ylabel)
        ax.set_xlim(*ax_meta["x_lim"])
        ax.set_ylim(*ax_meta["y_lim"])
        ax.set_title(f"({letter})  $P_{{\\mathrm{{fc}}}}={int(p_fc)}$ MPa")
        ax.grid(True, ls=":", lw=0.4, alpha=0.5)
        if letter == "a":
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "KKHS02 Fig.5 — fractional crystallization effects on $V_P$\n"
        f"(velocities @ {int(p_eval_mpa)} MPa, {int(t_eval_c)} °C; "
        f"{mb}, {kd}; raw ΔVp)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

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
    parser = argparse.ArgumentParser(description="Reproduce KKHS02 Fig.5 (paper layout a–d)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures" / "reproduce_fig5_wl1990.png",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--mineral-backend",
        choices=("auto", "burnman", "empirical", "fig2", "sb1994", "sb1994_fig2ol"),
        default=None,
    )
    parser.add_argument(
        "--kd-engine",
        choices=("heuristic", "langmuir"),
        default=None,
    )
    parser.add_argument("--quick", action="store_true", help="Coarse F grid for smoke test")
    args = parser.parse_args()
    f_grid = np.linspace(0.1, 0.9, 9) if args.quick else None
    run(
        f_grid=f_grid,
        save_figure=args.output,
        show=args.show,
        mineral_backend=args.mineral_backend,
        kd_engine=args.kd_engine,
    )


if __name__ == "__main__":
    main()
