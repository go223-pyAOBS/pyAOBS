"""
Compare Reproduction vs Modern melting tracks (F, melt chemistry, Vp/H).

Cases:
  - ``repro_linear``       — KKHS02 Step-4 linear F(P) on **native** KKHS02 geometry
  - ``repro_reebox``       — REEBOX isentropic + Katz Table 2 native F(P,T), Kinzler chemistry
  - ``modern_reebox``      — REEBOX isentropic + pyMelt Katz F(P,T), pMELTS chemistry
  - ``linear_on_reebox``   — same linear F(P) law, evaluated on **REEBOX** P0–Pf (shape only)

F(P) and P–T plots:
  (a) F(P) native geometry
  (b) P–T decompression path (isentropic / dry adiabat)
  (c) F(P) on shared REEBOX geometry
  (d) Isobaric F(T) at fixed P (Katz native vs pyMelt Katz, matched Mcpx=17 wt%)

Usage::

  py -3.11 petrology/validation/compare_tracks_melting.py --plot
  py -3.11 petrology/validation/compare_tracks_melting.py --ref-geom modern_reebox
  py -3.11 petrology/validation/compare_tracks_melting.py --ref-geom repro_reebox
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.active_upwelling import adiabat_mb88_c, DFDP_DEFAULT_PER_GPA

from petrology.melting.katz2003 import KATZ_CPX_MASS

_OX_SHOW = ("SiO2", "MgO", "FeO", "CaO", "Al2O3", "TiO2", "Na2O")


@dataclass
class ColumnSnapshot:
    name: str
    track: str
    tp_c: float
    chi: float
    b_km: float
    p0_gpa: float
    pf_gpa: float
    pbar_gpa: float
    fbar: float
    f_max: float
    h_km: float
    chemistry: str
    melting_f: str
    pooled: dict[str, float]
    vp_eq1_km_s: float | None = None
    vp_norm_km_s: float | None = None
    p_path: np.ndarray | None = None
    f_path: np.ndarray | None = None
    t_path: np.ndarray | None = None
    geometry_note: str = ""


def _adiabat_t_c(tp_c: float, p_gpa: np.ndarray | float) -> np.ndarray:
    p = np.asarray(p_gpa, dtype=float)
    return adiabat_mb88_c(tp_c, p)


def _attach_adiabat_t(snap: ColumnSnapshot) -> ColumnSnapshot:
    if snap.p_path is None or snap.f_path is None:
        return snap
    t = _adiabat_t_c(snap.tp_c, snap.p_path)
    return ColumnSnapshot(**{**snap.__dict__, "t_path": t})


def _linear_f_path(
    p0_gpa: float,
    pf_gpa: float,
    *,
    dfdp_per_gpa: float = DFDP_DEFAULT_PER_GPA,
    f_cap: float | None = None,
    n: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """KKHS02 linear active melting: F(P) = dF/dP * (P0 - P)."""
    p = np.linspace(float(pf_gpa), float(p0_gpa), int(n))
    f = np.clip(dfdp_per_gpa * (float(p0_gpa) - p), 0.0, None)
    if f_cap is not None:
        f = np.clip(f, 0.0, float(f_cap))
    return p, f


def _resample_f(p_grid: np.ndarray, p_path: np.ndarray, f_path: np.ndarray) -> np.ndarray:
    """Interpolate F(P) onto ``p_grid`` (path pressure need not be uniform)."""
    order = np.argsort(p_path)
    p_s = p_path[order]
    f_s = f_path[order]
    uniq_p, idx = np.unique(p_s, return_index=True)
    return np.interp(p_grid, uniq_p, f_s[idx], left=f_s[idx[0]], right=f_s[idx[-1]])


def _run_repro_linear(tp_c: float, chi: float, b_km: float) -> ColumnSnapshot:
    from petrology.melting.column import forward_melting_column

    r = forward_melting_column(
        tp_c=tp_c,
        b_km=b_km,
        chi=chi,
        melting_engine="kinzler_linear",
        lithology_backend="native",
        compute_norm_vp=True,
    )
    p, f = _linear_f_path(r.p0_gpa, r.pf_gpa, f_cap=r.f_max)
    t = _adiabat_t_c(tp_c, p)
    return ColumnSnapshot(
        name="repro_linear",
        track="reproduction",
        tp_c=tp_c,
        chi=chi,
        b_km=b_km,
        p0_gpa=r.p0_gpa,
        pf_gpa=r.pf_gpa,
        pbar_gpa=r.pbar_gpa,
        fbar=r.fbar,
        f_max=r.f_max,
        h_km=r.h_km,
        chemistry="kinzler1997",
        melting_f="linear F(P), dF/dP=0.12/GPa",
        pooled=dict(r.pooled_melt_wt),
        vp_eq1_km_s=float(r.vp_bulk_eq1_km_s),
        vp_norm_km_s=r.vp_bulk_norm_km_s,
        p_path=p,
        f_path=f,
        t_path=t,
        geometry_note="native KKHS02 Step-4 (TK83 solidus + linear F)",
    )


def _linear_on_reebox_geom(
    ref: ColumnSnapshot,
    *,
    dfdp_per_gpa: float = DFDP_DEFAULT_PER_GPA,
) -> ColumnSnapshot:
    """Linear F(P) evaluated on REEBOX P0–Pf (F-law only, not full column)."""
    p, f = _linear_f_path(ref.p0_gpa, ref.pf_gpa, dfdp_per_gpa=dfdp_per_gpa, f_cap=None)
    t = _adiabat_t_c(ref.tp_c, p)
    f_shallow = float(dfdp_per_gpa * (ref.p0_gpa - ref.pf_gpa))
    return ColumnSnapshot(
        name="linear_on_reebox",
        track="reproduction",
        tp_c=ref.tp_c,
        chi=ref.chi,
        b_km=ref.b_km,
        p0_gpa=ref.p0_gpa,
        pf_gpa=ref.pf_gpa,
        pbar_gpa=ref.pbar_gpa,
        fbar=float(np.mean(f)),
        f_max=f_shallow,
        h_km=ref.h_km,
        chemistry="kinzler1997",
        melting_f=f"linear F(P) on REEBOX geom (ref={ref.name})",
        pooled={},
        p_path=p,
        f_path=f,
        t_path=t,
        geometry_note=f"shared REEBOX P0={ref.p0_gpa:.2f}, Pf={ref.pf_gpa:.2f} GPa",
    )


def _run_reebox(
    tp_c: float,
    chi: float,
    b_km: float,
    *,
    lithology_backend: str,
    label: str,
    track: str,
    chemistry: str,
    melting_f: str,
) -> ColumnSnapshot:
    from petrology.melting.heterogeneous import forward_heterogeneous_column

    r = forward_heterogeneous_column(
        tp_c=tp_c,
        b_km=b_km,
        chi=chi,
        pyroxenite_frac=0.0,
        lithology_backend=lithology_backend,
        peridotite_lith="katz_lherzolite",
        compute_norm_vp=True,
    )
    steps = r.isentropic_path.steps
    p_path = np.array([s.p_gpa for s in steps], dtype=float)
    f_path = np.array([r.isentropic_path.f_bulk_at(s) for s in steps], dtype=float)
    t_path = np.array([s.t_c for s in steps], dtype=float)
    return ColumnSnapshot(
        name=label,
        track=track,
        tp_c=tp_c,
        chi=chi,
        b_km=b_km,
        p0_gpa=r.p0_gpa,
        pf_gpa=r.pf_gpa,
        pbar_gpa=r.pbar_gpa,
        fbar=r.fbar,
        f_max=r.f_max,
        h_km=r.h_km,
        chemistry=chemistry,
        melting_f=melting_f,
        pooled=dict(r.pooled_melt_wt),
        vp_eq1_km_s=float(r.vp_bulk_eq1_km_s),
        vp_norm_km_s=r.vp_bulk_norm_km_s,
        p_path=p_path,
        f_path=f_path,
        t_path=t_path,
        geometry_note="REEBOX isentropic column geometry",
    )


def _print_f_on_shared_geom(
    ref: ColumnSnapshot,
    linear_snap: ColumnSnapshot,
    isentropic_snaps: list[ColumnSnapshot],
) -> None:
    p_stations = [ref.p0_gpa]
    for p in (3.0, 2.5, 2.0, 1.5, 1.0, 0.5):
        if ref.pf_gpa + 0.02 <= p <= ref.p0_gpa - 0.02:
            p_stations.append(p)
    p_stations.append(ref.pf_gpa)
    p_stations = sorted(set(round(float(x), 3) for x in p_stations), reverse=True)

    p_grid = np.array(p_stations, dtype=float)
    print(f"\n## F(P) on shared REEBOX geometry (P0={ref.p0_gpa:.2f}, Pf={ref.pf_gpa:.2f} GPa)\n")
    cols = ["linear_on_reebox"] + [s.name for s in isentropic_snaps]
    hdr = f"{'P(GPa)':>8}  " + "  ".join(f"{c:>18}" for c in cols)
    print(hdr)
    print("-" * len(hdr))

    f_lin = _resample_f(p_grid, linear_snap.p_path, linear_snap.f_path)
    f_map = {"linear_on_reebox": f_lin}
    for s in isentropic_snaps:
        f_map[s.name] = _resample_f(p_grid, s.p_path, s.f_path)

    for i, p in enumerate(p_grid):
        row = f"{p:8.2f}  " + "  ".join(f"{f_map[c][i]:18.3f}" for c in cols)
        print(row)


def _print_vp_table(snaps: list[ColumnSnapshot]) -> None:
    print("\n## Vp and H (eq.1 + norm Vp from pooled melt)\n")
    print(f"{'case':<18} {'H_km':>6} {'Pbar':>6} {'Fbar':>6} {'Vp_eq1':>8} {'Vp_norm':>8}")
    print("-" * 58)
    for s in snaps:
        if s.vp_eq1_km_s is None:
            continue
        vn = f"{s.vp_norm_km_s:8.3f}" if s.vp_norm_km_s is not None else "     n/a"
        print(
            f"{s.name:<18} {s.h_km:6.1f} {s.pbar_gpa:6.2f} {s.fbar:6.3f} "
            f"{s.vp_eq1_km_s:8.3f} {vn}"
        )


def _print_inversion_sensitivity(repro: ColumnSnapshot, modern: ColumnSnapshot) -> None:
    from petrology.vp_regression import predict_vp_km_s

    print("\n## Inversion sensitivity (eq.1 Vp at Pbar, Fbar)\n")
    for label, s in (("repro_reebox", repro), ("modern_reebox", modern)):
        vp = predict_vp_km_s(s.pbar_gpa, s.fbar)
        print(f"  {label}: Pbar={s.pbar_gpa:.2f} GPa Fbar={s.fbar:.3f} -> Vp_eq1={vp:.3f} km/s")
    dvp = predict_vp_km_s(modern.pbar_gpa, modern.fbar) - predict_vp_km_s(
        repro.pbar_gpa, repro.fbar
    )
    print(f"  dVp_eq1 (modern - repro) = {dvp:+.3f} km/s")
    if repro.vp_norm_km_s is not None and modern.vp_norm_km_s is not None:
        dvn = modern.vp_norm_km_s - repro.vp_norm_km_s
        print(f"  dVp_norm (modern - repro) = {dvn:+.3f} km/s")
    print("  (Full Tp/chi inversion requires forward scan — see compare_tracks_inversion_sensitivity.py)")


def _print_column_table(snaps: list[ColumnSnapshot]) -> None:
    print("\n## 熔融柱对比（相同 Tp, chi, b）\n")
    print(
        f"{'case':<18} {'track':<12} {'P0':>6} {'Pf':>6} {'Pbar':>6} "
        f"{'Fbar':>6} {'Fmax':>6} {'H_km':>6}  geometry / F model"
    )
    print("-" * 105)
    for s in snaps:
        if not s.pooled:
            continue
        print(
            f"{s.name:<18} {s.track:<12} {s.p0_gpa:6.2f} {s.pf_gpa:6.2f} {s.pbar_gpa:6.2f} "
            f"{s.fbar:6.3f} {s.f_max:6.3f} {s.h_km:6.1f}  {s.geometry_note or s.melting_f}"
        )

    print("\n## 柱内 pooled 熔体主量 (wt%)\n")
    hdr = f"{'case':<18} " + "  ".join(f"{ox:>6}" for ox in _OX_SHOW)
    print(hdr)
    print("-" * len(hdr))
    for s in snaps:
        if not s.pooled:
            continue
        row = "  ".join(f"{s.pooled.get(k, 0):6.2f}" for k in _OX_SHOW)
        print(f"{s.name:<18} {row}")


def _isobaric_f_t_curves(
    pressures_gpa: list[float],
    *,
    t_min_c: float = 1100.0,
    t_max_c: float = 1800.0,
    n: int = 250,
    cpx_mass: float = KATZ_CPX_MASS,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Isobaric F(T) at fixed P: Katz Table 2 native vs pyMelt katz_lherzolite (same Mcpx)."""
    from petrology.melting.katz2003 import katz2003_melt_fraction_dry
    from petrology.melting.pymelt_lithology_adapter import instantiate_pymelt_lithology

    t = np.linspace(float(t_min_c), float(t_max_c), int(n))
    pm = instantiate_pymelt_lithology("katz_lherzolite")
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    mc = float(cpx_mass)
    for p in pressures_gpa:
        pg = float(p)
        f_k = np.array(
            [katz2003_melt_fraction_dry(pg, float(ti), cpx_mass=mc) for ti in t],
            dtype=float,
        )
        f_p = np.array([float(pm.F(pg, float(ti))) for ti in t], dtype=float)
        out[f"katz_native_P{pg:g}"] = (t, f_k)
        out[f"pymelt_katz_P{pg:g}"] = (t, f_p)
    return out


def _plot_style() -> dict[str, tuple[str, str, float]]:
    return {
        "repro_linear": ("#2471a3", "-", 2.2),
        "linear_on_reebox": ("#2471a3", ":", 2.0),
        "repro_reebox": ("#27ae60", "--", 2.0),
        "modern_reebox": ("#c0392b", "-", 2.0),
    }


def _plot_comparison(
    native_snaps: list[ColumnSnapshot],
    shared_snaps: list[ColumnSnapshot],
    ref: ColumnSnapshot,
    isobaric: dict[str, tuple[np.ndarray, np.ndarray]],
    out: Path,
    *,
    f_max_plot: float = 1.5,
    cpx_mass: float = KATZ_CPX_MASS,
) -> None:
    import matplotlib.pyplot as plt

    style = _plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), dpi=120)

    ax = axes[0, 0]
    for s in native_snaps:
        if s.p_path is None:
            continue
        col, ls, lw = style.get(s.name, ("#333", "-", 1.8))
        ax.plot(s.p_path, s.f_path, ls, color=col, lw=lw, label=s.name)
    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("F")
    ax.set_title("(a) F(P) — native geometry")
    ax.invert_xaxis()
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, min(f_max_plot, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else f_max_plot))

    ax = axes[0, 1]
    pt_snaps = list(native_snaps) + [s for s in shared_snaps if s.name == "linear_on_reebox"]
    for s in pt_snaps:
        if s.t_path is None or s.p_path is None:
            continue
        col, ls, lw = style.get(s.name, ("#333", "-", 1.8))
        lab = "linear on REEBOX" if s.name == "linear_on_reebox" else s.name
        ax.plot(s.p_path, s.t_path, ls, color=col, lw=lw, label=lab)
    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(
        "(b) P–T decompression path\n"
        "(isentropic latent-heat buffered vs dry adiabat T=Tp−30P)"
    )
    ax.invert_xaxis()
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for s in shared_snaps:
        if s.p_path is None:
            continue
        col, ls, lw = style.get(s.name, ("#333", "-", 1.8))
        lab = "linear on REEBOX" if s.name == "linear_on_reebox" else s.name
        ax.plot(s.p_path, s.f_path, ls, color=col, lw=lw, label=lab)
    ax.axvline(ref.p0_gpa, color="#888", ls=":", lw=1, alpha=0.7)
    ax.axvline(ref.pf_gpa, color="#888", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("F")
    ax.set_title(f"(c) F(P) — shared REEBOX geom ({ref.name})")
    ax.invert_xaxis()
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, f_max_plot)

    ax = axes[1, 1]
    p_keys = sorted(
        {k.split("_P", 1)[1] for k in isobaric if k.startswith("katz_native_")},
        key=float,
    )
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(p_keys), 1)))
    mc_pct = 100.0 * float(cpx_mass)
    for pg_s, col in zip(p_keys, cmap):
        pg = float(pg_s)
        t_k, f_k = isobaric[f"katz_native_P{pg:g}"]
        t_p, f_p = isobaric[f"pymelt_katz_P{pg:g}"]
        ax.plot(t_k, f_k, "-", color=col, lw=2.0, label=f"P={pg:g} GPa Katz native")
        ax.plot(t_p, f_p, "--", color=col, lw=1.5, label=f"P={pg:g} GPa pyMelt Katz")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("F")
    ax.set_title(
        f"(d) Isobaric F(T) at fixed P\n"
        f"Katz Table 2 native vs pyMelt katz_lherzolite (Mcpx={mc_pct:g} wt%)"
    )
    ax.legend(fontsize=6, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, min(1.05, f_max_plot))

    s0 = native_snaps[0]
    fig.suptitle(
        f"Melting tracks — Tp={s0.tp_c:.0f} C, chi={s0.chi:g}, b={s0.b_km:g} km",
        fontsize=10,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"\nWrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare reproduction vs modern melting tracks")
    parser.add_argument("--tp", type=float, default=1450.0)
    parser.add_argument("--chi", type=float, default=8.0)
    parser.add_argument("--b", type=float, default=0.0)
    parser.add_argument(
        "--ref-geom",
        choices=("repro_reebox", "modern_reebox"),
        default="modern_reebox",
        help="REEBOX P0/Pf reference for shared F(P) panel (default: modern_reebox)",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--isobaric-p",
        default="1,2,3",
        help="Pressures (GPa) for isobaric F(T) panel, comma-separated",
    )
    parser.add_argument("--t-min", type=float, default=1100.0)
    parser.add_argument("--t-max", type=float, default=1800.0)
    parser.add_argument("--f-max", type=float, default=1.5, help="Y-axis cap for path F plots")
    parser.add_argument(
        "--cpx-mass",
        type=float,
        default=KATZ_CPX_MASS,
        help="Clinopyroxene mass fraction for isobaric Katz F(T) panel (default: KATZ_CPX_MASS)",
    )
    parser.add_argument(
        "-o",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures" / "compare_tracks_melting.png",
    )
    args = parser.parse_args()

    tp, chi, b = float(args.tp), float(args.chi), float(args.b)

    print(f"# Melting track comparison (Tp={tp:g} C, chi={chi:g}, b={b:g} km)")
    print("# Scope: F(P), melt chemistry, Vp_eq1, Vp_norm\n")

    repro_linear = _run_repro_linear(tp, chi, b)
    repro_reebox = _run_reebox(
        tp,
        chi,
        b,
        lithology_backend="native",
        label="repro_reebox",
        track="reproduction",
        chemistry="kinzler1997",
        melting_f="isentropic Katz Table 2 F(P,T)",
    )
    modern_reebox = _run_reebox(
        tp,
        chi,
        b,
        lithology_backend="pymelt",
        label="modern_reebox",
        track="modern",
        chemistry="pmelts_klb1",
        melting_f="isentropic pyMelt Katz F(P,T)",
    )

    ref_name = str(args.ref_geom)
    ref_snap = repro_reebox if ref_name == "repro_reebox" else modern_reebox
    linear_on_reebox = _linear_on_reebox_geom(ref_snap)

    column_snaps = [repro_linear, repro_reebox, modern_reebox]
    _print_column_table(column_snaps)
    _print_vp_table(column_snaps)
    _print_inversion_sensitivity(repro_reebox, modern_reebox)

    print(f"\n# Shared F(P) reference geometry: {ref_name} (P0={ref_snap.p0_gpa:.2f}, Pf={ref_snap.pf_gpa:.2f} GPa)")

    isentropic = [repro_reebox, modern_reebox]
    _print_f_on_shared_geom(ref_snap, linear_on_reebox, isentropic)

    print("\n## F(P) shape notes\n")
    print("  - (a) repro_linear uses KKHS02 native P0/Pf — not comparable in absolute P to REEBOX.")
    print("  - (b) P–T: isentropic paths are near-isothermal (latent heat); linear uses dry adiabat.")
    print("  - (c) linear F(P) vs isentropic curves on the selected REEBOX P0–Pf interval.")
    shallow_f_repro = float(repro_reebox.f_path[-1])
    shallow_f_mod = float(modern_reebox.f_path[-1])
    print(
        f"  - Shallow end (Pf={ref_snap.pf_gpa:.2f} GPa): linear F={linear_on_reebox.f_max:.2f}, "
        f"repro_reebox F={shallow_f_repro:.2f}, modern_reebox F={shallow_f_mod:.2f}"
    )

    print("\n## Modern vs Repro-linear (native columns)\n")
    print(f"  dFbar = {modern_reebox.fbar - repro_linear.fbar:+.3f}")
    print(f"  dP0   = {modern_reebox.p0_gpa - repro_linear.p0_gpa:+.2f} GPa")

    if args.plot:
        iso_p = [float(x.strip()) for x in str(args.isobaric_p).split(",") if x.strip()]
        isobaric = _isobaric_f_t_curves(
            iso_p,
            t_min_c=float(args.t_min),
            t_max_c=float(args.t_max),
            cpx_mass=float(args.cpx_mass),
        )
        _plot_comparison(
            native_snaps=[repro_linear, repro_reebox, modern_reebox],
            shared_snaps=[linear_on_reebox, repro_reebox, modern_reebox],
            ref=ref_snap,
            isobaric=isobaric,
            out=args.o,
            f_max_plot=float(args.f_max),
            cpx_mass=float(args.cpx_mass),
        )


if __name__ == "__main__":
    main()
