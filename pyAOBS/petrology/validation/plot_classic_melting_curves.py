"""
Classic mantle melting theory curves (no H, no Vp).

Reference curves:
  - Katz (2003) dry lherzolite solidus / liquidus (native analytic + optional pyMelt)
  - REEBOX G2 pyroxenite solidus offset (Pertermann & Hirschmann 2003 style)
  - McKenzie & Bickle (1988) adiabat: T = Tp + 20 °C/GPa  (T_p = T_0 - 20 P_0)
  - Brown & Lesher (2016) REEBOX PRO incrementally isentropic F(P) decompression paths

Examples:
  python petrology/validation/plot_classic_melting_curves.py
  python petrology/validation/plot_classic_melting_curves.py --show
  python petrology/validation/plot_classic_melting_curves.py -o petrology/figures/classic_melting
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

FIG_DIR = Path(__file__).resolve().parents[1] / "figures" / "classic_melting"

# Tab order / labels for GUI preview (must match build_* exports).
CLASSIC_CURVE_PANELS: tuple[tuple[str, str], ...] = (
    ("katz_fig1", "00 Katz (2003) Fig.1"),
    ("katz_fig2", "00b Katz (2003) Fig.2"),
    ("katz_fig3", "00c Katz (2003) Fig.3"),
    ("katz_fig4", "00d Katz (2003) Fig.4"),
    ("katz_fig5", "00e Katz (2003) Fig.5"),
    ("katz_fig6", "00f Katz (2003) Fig.6"),
    ("tp", "01 T–P 固相线 / 绝热线"),
    ("fdp", "02 等熵减压 F(P)"),
    ("ft", "03 等压 F(T)"),
    ("solidus", "04 岩性固相线"),
    ("dfdp", "05 dF/dP 产率"),
)


def _try_pymelt_solidus(key: str, p_gpa, *, h2o_wt: float = 0.0):
    try:
        from petrology.melting.pymelt_lithology_adapter import pymelt_lithology

        lith = pymelt_lithology(key, h2o_wt=h2o_wt)
        ts = [lith.solidus_gpa(float(p)) for p in p_gpa]
        tl = [lith.liquidus_gpa(float(p)) for p in p_gpa]
        return ts, tl, True
    except Exception:
        return None, None, False


def build_tp_diagram_figure():
    """T–P: solidus, liquidus, adiabats (classic petrology textbook panel)."""
    import matplotlib.pyplot as plt
    import numpy as np

    from petrology.active_upwelling import adiabat_mb88_c
    from petrology.melting.lithology import (
        g2_pyroxenite_liquidus,
        g2_pyroxenite_solidus,
        katz2003_peridotite_liquidus,
        katz2003_peridotite_solidus,
    )

    p = np.linspace(0.0, 5.0, 200)
    ts_katz = np.array([katz2003_peridotite_solidus(x) for x in p])
    tl_katz = np.array([katz2003_peridotite_liquidus(x) for x in p])
    ts_g2 = np.array([g2_pyroxenite_solidus(x) for x in p])
    tl_g2 = np.array([g2_pyroxenite_liquidus(x) for x in p])

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=100)
    ax.fill_between(p, ts_katz, tl_katz, color="#fdebd0", alpha=0.45, label="Katz lherzolite melting interval")
    ax.plot(p, ts_katz, "k-", lw=2, label="Katz solidus (native)")
    ax.plot(p, tl_katz, "k--", lw=1.2, label="Katz liquidus (native)")
    ax.plot(p, ts_g2, color="#c0392b", lw=1.8, label="G2 pyroxenite solidus (native)")
    ax.plot(p, tl_g2, color="#c0392b", ls="--", lw=1.0, label="G2 liquidus (native)")

    pm_ts, pm_tl, ok = _try_pymelt_solidus("katz_lherzolite", p)
    if ok:
        ax.plot(p, pm_ts, color="#2980b9", ls=":", lw=1.5, label="Katz solidus (pyMelt)")
        ax.plot(p, pm_tl, color="#2980b9", ls="-.", lw=1.0, alpha=0.8, label="Katz liquidus (pyMelt)")
    pm2_ts, pm2_tl, ok2 = _try_pymelt_solidus("pertermann_g2", p)
    if ok2:
        ax.plot(p, pm2_ts, color="#e67e22", ls=":", lw=1.5, label="G2 solidus (pyMelt)")
        ax.plot(p, pm2_tl, color="#e67e22", ls="-.", lw=1.0, alpha=0.8, label="G2 liquidus (pyMelt)")

    tp_vals = (1200.0, 1300.0, 1400.0, 1500.0)
    cmap = plt.cm.plasma(np.linspace(0.15, 0.85, len(tp_vals)))
    for tp, col in zip(tp_vals, cmap):
        t_ad = adiabat_mb88_c(tp, p)
        ax.plot(p, t_ad, color=col, lw=1.1, alpha=0.85)
        j = int(np.argmin(np.abs(t_ad - ts_katz)))
        ax.text(float(p[j]), float(t_ad[j]), f"Tp={tp:.0f}", fontsize=8, color=col)

    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Classic T–P diagram\n(solidus / liquidus / adiabat T = Tp + 20 P)")
    ax.set_xlim(0, 5)
    ax.set_ylim(900, 2000)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_tp_diagram(out_dir: Path, *, show: bool) -> None:
    import matplotlib.pyplot as plt

    fig = build_tp_diagram_figure()
    path = out_dir / "01_tp_solidus_adiabat.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"Wrote {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def build_f_vs_p_decompression_figure(*, tp_c: float):
    """F(P) along isentropic decompression (REEBOX-core paths)."""
    import matplotlib.pyplot as plt

    from petrology.melting.isentropic import integrate_isentropic_path
    from petrology.melting.lithology import dry_peridotite_lithology, heterogeneous_source
    from petrology.melting.reebox_geometry import p0_gpa_at_solidus

    pf = 0.05
    n_steps = 100

    def _path_f(lithologies, label: str, style: str):
        p0 = p0_gpa_at_solidus(tp_c, lithologies)
        path = integrate_isentropic_path(
            lithologies, tp_c=tp_c, p0_gpa=p0, pf_gpa=pf, n_steps=n_steps
        )
        ps = [st.p_gpa for st in path.steps]
        fs = [path.f_bulk_at(st) for st in path.steps]
        comps = {
            name: [st.f_by_name.get(name, 0.0) for st in path.steps]
            for name in [L.name for L in lithologies]
        }
        return ps, fs, comps, label, style, p0

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), dpi=100)

    per = [dry_peridotite_lithology()]
    het_native = heterogeneous_source(pyroxenite_frac=0.10, backend="native")

    series = [
        _path_f(per, "Dry Katz lherzolite", "k-"),
        _path_f(het_native, "10% G2 heterogeneous (native)", "#c0392b"),
    ]

    try:
        het_pm = heterogeneous_source(
            pyroxenite_frac=0.10,
            backend="pymelt",
            peridotite_key="katz_lherzolite",
            pyroxenite_key="pertermann_g2",
        )
        series.append(_path_f(het_pm, "10% G2 (pyMelt Katz+G2)", "#2980b9"))
    except Exception:
        pass

    ax = axes[0]
    for ps, fs, _, lab, sty, p0 in series:
        ax.plot(ps, fs, sty, lw=1.8, label=f"{lab}  P0={p0:.2f} GPa")
    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("Bulk melt fraction F")
    ax.set_title(f"Isentropic decompression F(P)  Tp={tp_c:.0f} °C")
    ax.invert_xaxis()
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.25)

    ax2 = axes[1]
    ps_het, _, comps, _, sty, _ = series[1]
    for name, fc in comps.items():
        ax2.plot(ps_het, fc, sty, lw=1.4, label=name)
    ax2.set_xlabel("Pressure (GPa)")
    ax2.set_ylabel("End-member melt fraction F")
    ax2.set_title("Heterogeneous 10% G2 (native) — end-member F(P)")
    ax2.invert_xaxis()
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=7, loc="lower left")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    return fig


def plot_f_vs_p_decompression(out_dir: Path, *, show: bool, tp_c: float) -> None:
    import matplotlib.pyplot as plt

    fig = build_f_vs_p_decompression_figure(tp_c=tp_c)
    path = out_dir / "02_f_vs_p_decompression.png"
    fig.savefig(path)
    print(f"Wrote {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def build_f_vs_t_isobar_figure(*, p_gpa: float = 2.0):
    """F(T) at constant P — batch melting interval (Katz vs G2)."""
    import matplotlib.pyplot as plt
    import numpy as np

    from petrology.melting.lithology import dry_peridotite_lithology, g2_pyroxenite_lithology

    per = dry_peridotite_lithology()
    g2 = g2_pyroxenite_lithology(u0=1.0)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=100)
    for lith, color, lab in (
        (per, "k", "Katz lherzolite"),
        (g2, "#c0392b", "G2 pyroxenite"),
    ):
        ts, tl = lith.solidus_gpa(p_gpa), lith.liquidus_gpa(p_gpa)
        t = np.linspace(ts - 20, tl + 20, 200)
        f = [lith.f_from_t(p_gpa, float(ti)) for ti in t]
        ax.plot(t, f, color=color, lw=2, label=lab)
        ax.axvline(ts, color=color, ls=":", lw=0.9, alpha=0.6)
        ax.axvline(tl, color=color, ls="--", lw=0.9, alpha=0.6)

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Melt fraction F")
    ax.set_title(f"Batch melting F(T) at P = {p_gpa:.1f} GPa")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_f_vs_t_isobar(out_dir: Path, *, show: bool) -> None:
    import matplotlib.pyplot as plt

    fig = build_f_vs_t_isobar_figure()
    path = out_dir / "03_f_vs_t_isobar.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"Wrote {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def build_lithology_solidus_catalog_figure():
    """Compare solidus T(P) for registered pyMelt lithologies."""
    import matplotlib.pyplot as plt
    import numpy as np

    from petrology.melting.lithology import (
        g2_pyroxenite_solidus,
        katz2003_peridotite_solidus,
    )
    from petrology.melting.pymelt_lithology_adapter import list_pymelt_lithology_keys

    p = np.linspace(0.5, 4.5, 100)
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=100)

    ax.plot(p, [katz2003_peridotite_solidus(x) for x in p], "k-", lw=2, label="native Katz")
    ax.plot(p, [g2_pyroxenite_solidus(x) for x in p], color="#c0392b", lw=1.8, label="native G2 (−150°C)")

    keys = list_pymelt_lithology_keys()
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(keys), 1)))
    n_ok = 0
    for key, col in zip(keys, cmap):
        ts, _, ok = _try_pymelt_solidus(key, p)
        if not ok:
            continue
        ax.plot(p, ts, color=col, lw=1.1, alpha=0.85, label=key)
        n_ok += 1

    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("Solidus T (°C)")
    ax.set_title(f"Lithology solidus comparison (pyMelt n={n_ok})")
    ax.legend(loc="best", fontsize=6, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_lithology_solidus_catalog(out_dir: Path, *, show: bool) -> None:
    import matplotlib.pyplot as plt

    fig = build_lithology_solidus_catalog_figure()
    path = out_dir / "04_lithology_solidus_catalog.png"
    fig.savefig(path)
    print(f"Wrote {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def build_dfdp_adiabat_figure(*, tp_c: float):
    """dF/dP vs P along dry adiabat (melting productivity)."""
    import matplotlib.pyplot as plt
    import numpy as np

    from petrology.melting.lithology import dry_peridotite_lithology

    lith = dry_peridotite_lithology()
    p = np.linspace(0.5, 4.0, 80)
    dfdp_num = []
    for pg in p:
        t_ad = tp_c - 30.0 * pg
        if t_ad <= lith.solidus_gpa(pg):
            dfdp_num.append(np.nan)
            continue
        dp = 0.02
        p2 = max(pg - dp, 0.05)
        t2 = tp_c - 30.0 * p2
        f1 = lith.f_from_t(pg, t_ad)
        f2 = lith.f_from_t(p2, t2) if t2 > lith.solidus_gpa(p2) else 0.0
        dfdp_num.append((f2 - f1) / (p2 - pg))

    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=100)
    ax.plot(p, dfdp_num, "k-", lw=1.8, label=f"Katz lherzolite  Tp={tp_c:.0f}°C adiabat")
    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("dF/dP (1/GPa)")
    ax.set_title("Melting productivity dF/dP along adiabat (native Katz)")
    ax.invert_xaxis()
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_dfdp_isobar(out_dir: Path, *, show: bool, tp_c: float) -> None:
    import matplotlib.pyplot as plt

    fig = build_dfdp_adiabat_figure(tp_c=tp_c)
    path = out_dir / "05_dfdp_adiabat.png"
    fig.savefig(path)
    print(f"Wrote {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def build_classic_curve_figure(key: str, *, tp_c: float = 1350.0):
    """Build one classic panel by key (see CLASSIC_CURVE_PANELS)."""
    from petrology.validation.katz2003_workflow import panel_by_key

    builders = {
        "tp": lambda: build_tp_diagram_figure(),
        "fdp": lambda: build_f_vs_p_decompression_figure(tp_c=tp_c),
        "ft": lambda: build_f_vs_t_isobar_figure(),
        "solidus": lambda: build_lithology_solidus_catalog_figure(),
        "dfdp": lambda: build_dfdp_adiabat_figure(tp_c=tp_c),
    }
    if key.startswith("katz_fig"):
        return panel_by_key(key).builder()
    if key not in builders:
        raise KeyError(f"Unknown classic curve key: {key!r}")
    return builders[key]()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot classic melting theory curves (no H/Vp)")
    parser.add_argument("-o", "--out-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--tp", type=float, default=1350.0, help="Tp for decompression / dF/dP panels")
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=(
            "tp",
            "fdp",
            "ft",
            "solidus",
            "dfdp",
            "katz_fig1",
            "katz_fig2",
            "katz_fig3",
            "katz_fig4",
            "katz_fig5",
            "katz_fig6",
            "all",
        ),
        default=("all",),
    )
    args = parser.parse_args()
    sections = set(args.only)
    if "all" in sections:
        sections = {
            "katz_fig1",
            "katz_fig2",
            "katz_fig3",
            "katz_fig4",
            "katz_fig5",
            "katz_fig6",
            "tp",
            "fdp",
            "ft",
            "solidus",
            "dfdp",
        }

    print("=" * 60)
    print("Classic melting curves (no H, no Vp)")
    print(f"  output: {args.out_dir}")
    print("=" * 60)

    if "katz_fig1" in sections:
        from petrology.melting.katz2003 import build_katz2003_fig1_figure

        import matplotlib.pyplot as plt

        fig = build_katz2003_fig1_figure()
        path = args.out_dir / "00_katz2003_fig1.png"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        print(f"Wrote {path}")
        if args.show:
            plt.show()
        else:
            plt.close(fig)

    if "tp" in sections:
        plot_tp_diagram(args.out_dir, show=args.show)
    if "fdp" in sections:
        plot_f_vs_p_decompression(args.out_dir, show=args.show, tp_c=args.tp)
    if "ft" in sections:
        plot_f_vs_t_isobar(args.out_dir, show=args.show)
    if "solidus" in sections:
        plot_lithology_solidus_catalog(args.out_dir, show=args.show)
    if "dfdp" in sections:
        plot_dfdp_isobar(args.out_dir, show=args.show, tp_c=args.tp)

    print("\nDone.")


if __name__ == "__main__":
    main()
