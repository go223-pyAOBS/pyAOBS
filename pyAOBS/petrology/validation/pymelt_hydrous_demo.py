"""
复现 pyMelt 官方含水熔融教程 (Simon Matthews, 2022).

文档: https://pymelt.readthedocs.io/en/latest/HydrousMelting.html

图件:
  1. solidus.png           — 高 H2O 固相线
  2. dFdP.png              — dF/dP vs P (Tp=1300°C)
  3. dTdP.png              — dT/dP vs P + 固体绝热线
  4. dTdP_partial.png      — (dT/dP)|_F vs P
  5. dTdF_partial.png      — (dT/dF)|_P vs P
  6. saturated_batch.png   — 10 wt% H2O 绝热减压 (prevent_freezing 开/关)

运行 (WSL obs, pip install pyMelt):
  python petrology/validation/pymelt_hydrous_demo.py
  python petrology/validation/pymelt_hydrous_demo.py --no-show -o output/pymelt_hydrous
  python petrology/validation/pymelt_hydrous_demo.py --only solidus dFdP
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from petrology.melting.pymelt_bridge import ensure_pymelt_scipy_compat

# 教程默认
H2O_SOLIDUS = np.array([0.1, 0.2, 0.3, 1.0])
H2O_COLUMNS = np.array([0.05, 0.1, 0.15])
P_SOLIDUS = np.linspace(0, 8, 100)
DP_STEP = -0.01


@dataclass
class DecompressionCase:
    """Tp=1300°C 干/湿绝热柱 + 偏导数 (教程 § melting expressions)."""

    lz: object
    dry_mantle: object
    dry: object
    h2o_wt: np.ndarray
    wet_cols: list
    dry_dtdf: np.ndarray
    dry_dtdp: np.ndarray
    wet_dtdf: list[np.ndarray]
    wet_dtdp: list[np.ndarray]


def _instantiate(obj):
    return obj() if isinstance(obj, type) else obj()


def _import_pymelt():
    ensure_pymelt_scipy_compat()
    import pyMelt as m

    return m


def _save(fig, out: Path | None, show: bool) -> None:
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        print(f"Wrote {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _uniform_dp(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    if len(p) < 3:
        return float(abs(p[-1] - p[0])) if len(p) >= 2 else 1.0
    return float(abs(p[2] - p[1]))


def _forward_diff_y_vs_p(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """教程 finite difference: -(y[i]-y[i+1]) / (P[2]-P[1])."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    dp = _uniform_dp(p)
    dy = -(y[:-1] - y[1:]) / dp
    pmid = p[:-1]
    return dy, pmid


def _klb1(m):
    return _instantiate(m.lithologies.matthews.klb1)


def compute_decompression_case(*, tp_c: float = 1300.0, h2o_wt: np.ndarray | None = None) -> DecompressionCase:
    m = _import_pymelt()
    if h2o_wt is None:
        h2o_wt = H2O_COLUMNS

    lz = _klb1(m)
    dry_mantle = m.mantle([lz], [1.0], ["lz"])
    dry = dry_mantle.adiabaticMelt(tp_c, dP=DP_STEP)

    dry_dtdf = np.array([lz.dTdF(float(dry.P[j]), float(dry.T[j])) for j in range(len(dry.T))])
    dry_dtdp = np.array([lz.dTdP(float(dry.P[j]), float(dry.T[j])) for j in range(len(dry.T))])

    wet_cols: list = []
    wet_dtdf: list[np.ndarray] = []
    wet_dtdp: list[np.ndarray] = []

    for wt in h2o_wt:
        hlz = m.hydrousLithology(lz, float(wt))
        wet_mantle = m.mantle([hlz], [1.0], ["lz"])
        col = wet_mantle.adiabaticMelt(tp_c, dP=DP_STEP)
        wet_cols.append(col)
        wet_dtdf.append(
            np.array([hlz.dTdF(float(col.P[j]), float(col.T[j])) for j in range(len(col.T))])
        )
        wet_dtdp.append(
            np.array([hlz.dTdP(float(col.P[j]), float(col.T[j])) for j in range(len(col.T))])
        )

    return DecompressionCase(
        lz=lz,
        dry_mantle=dry_mantle,
        dry=dry,
        h2o_wt=np.asarray(h2o_wt, dtype=float),
        wet_cols=wet_cols,
        dry_dtdf=dry_dtdf,
        dry_dtdp=dry_dtdp,
        wet_dtdf=wet_dtdf,
        wet_dtdp=wet_dtdp,
    )


def plot_solidus(*, out: Path | None, show: bool) -> None:
    """§ What happens to the Solidus at high H2O contents?"""
    m = _import_pymelt()
    lz = _klb1(m)
    tsol = np.zeros((len(H2O_SOLIDUS) + 1, len(P_SOLIDUS)))
    tsol[0, :] = lz.TSolidus(P_SOLIDUS)

    for i, wt in enumerate(H2O_SOLIDUS):
        hlz = m.hydrousLithology(lz, float(wt))
        for j, pj in enumerate(P_SOLIDUS):
            tsol[i + 1, j] = hlz.TSolidus(float(pj))

    fig, ax = plt.subplots(dpi=100)
    for i in range(tsol.shape[0]):
        ax.plot(tsol[i, :], P_SOLIDUS)
    ax.invert_yaxis()
    ax.set_ylabel("Pressure (GPa)")
    ax.set_xlabel("Temperature (°C)")
    ax.set_title("Hydrous solidus (KLB1; left = fluid-saturated branch)")
    fig.tight_layout()
    _save(fig, out, show)


def plot_dfdp(case: DecompressionCase, *, tp_c: float, out: Path | None, show: bool) -> None:
    """§ dF/dP during integration."""
    fig, ax = plt.subplots(dpi=100)
    dfdp, pm = _forward_diff_y_vs_p(np.asarray(case.dry.F), np.asarray(case.dry.P))
    ax.plot(dfdp, pm, label="Dry")

    for i, wt in enumerate(case.h2o_wt):
        dfdp_w, pm_w = _forward_diff_y_vs_p(np.asarray(case.wet_cols[i].F), np.asarray(case.wet_cols[i].P))
        ax.plot(dfdp_w, pm_w, label=f"{wt:.2f} wt% H$_2$O")

    ax.legend()
    ax.set_ylabel("Pressure (GPa)")
    ax.set_xlabel(r"$dF/dP$ (GPa$^{-1}$)")
    ax.set_title(f"$T_p = {tp_c:.0f}^\\circ$C")
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, out, show)


def plot_dtdp(case: DecompressionCase, *, tp_c: float, out: Path | None, show: bool) -> None:
    """§ dT/dP along geotherm + solid adiabat reference."""
    fig, ax = plt.subplots(dpi=100)
    dtdp, pm = _forward_diff_y_vs_p(np.asarray(case.dry.T), np.asarray(case.dry.P))
    ax.plot(dtdp, pm, label="Dry")

    pp = np.linspace(0, 7, 100)
    solid_adiabat = case.dry_mantle.adiabat(pp, tp_c)
    ad_dtdp, ad_p = _forward_diff_y_vs_p(np.asarray(solid_adiabat), pp)
    ax.plot(ad_dtdp, ad_p, c="k", label="Solid Adiabat")

    for i, wt in enumerate(case.h2o_wt):
        dtdp_w, pm_w = _forward_diff_y_vs_p(np.asarray(case.wet_cols[i].T), np.asarray(case.wet_cols[i].P))
        ax.plot(dtdp_w, pm_w, label=f"{wt:.2f} wt% H$_2$O")

    ax.legend()
    ax.set_ylabel("Pressure (GPa)")
    ax.set_xlabel(r"$dT/dP$ (K GPa$^{-1}$)")
    ax.set_title(f"$T_p = {tp_c:.0f}^\\circ$C")
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, out, show)


def plot_dtdp_partial(case: DecompressionCase, *, tp_c: float, out: Path | None, show: bool) -> None:
    """§ (dT/dP)|_F from lithology.dTdP (numerical in pyMelt)."""
    fig, ax = plt.subplots(dpi=100)
    ax.plot(case.dry_dtdp[1:], np.asarray(case.dry.P)[1:], label="Dry")
    for i, wt in enumerate(case.h2o_wt):
        ax.plot(case.wet_dtdp[i][1:], np.asarray(case.wet_cols[i].P)[1:], label=f"{wt:.2f} wt% H$_2$O")
    ax.invert_yaxis()
    ax.set_ylabel("Pressure (GPa)")
    ax.set_xlabel(r"$(dT/dP)|_F$ (K GPa$^{-1}$)")
    ax.set_title(f"$T_p = {tp_c:.0f}^\\circ$C")
    ax.legend()
    fig.tight_layout()
    _save(fig, out, show)


def plot_dtdf_partial(case: DecompressionCase, *, tp_c: float, out: Path | None, show: bool) -> None:
    """§ (dT/dF)|_P from lithology.dTdF."""
    fig, ax = plt.subplots(dpi=100)
    ax.plot(case.dry_dtdf, np.asarray(case.dry.P), label="Dry")
    for i, wt in enumerate(case.h2o_wt):
        ax.plot(case.wet_dtdf[i], np.asarray(case.wet_cols[i].P), label=f"{wt:.2f} wt% H$_2$O")
    ax.invert_yaxis()
    ax.set_ylabel("Pressure (GPa)")
    ax.set_xlabel(r"$(dT/dF)|_P$ (K)")
    ax.set_title(f"$T_p = {tp_c:.0f}^\\circ$C")
    ax.legend()
    fig.tight_layout()
    _save(fig, out, show)


def plot_saturated_batch(
    *,
    tp_c: float = 1200.0,
    pstart_gpa: float = 8.0,
    h2o_wt: float = 10.0,
    out: Path | None,
    show: bool,
) -> None:
    """
    § Adiabatically decompress H2O-saturated peridotite.

    Batch melting thought experiment (文档 Disclaimer).
    """
    m = _import_pymelt()
    lz = _klb1(m)
    hlz = m.hydrousLithology(lz, h2o_wt)
    mantle = m.mantle([hlz], [1.0])

    # 1 wt% solidus for overlay (教程 tsol[-1])
    p_line = P_SOLIDUS
    tsol_sat = np.array([hlz.TSolidus(float(pj)) for pj in p_line])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        col_no_freeze = mantle.adiabaticMelt(tp_c, Pstart=pstart_gpa, prevent_freezing=True)
        col_freeze = mantle.adiabaticMelt(tp_c, Pstart=pstart_gpa, prevent_freezing=False)

    fig, axes = plt.subplots(1, 2, sharey="row", dpi=100)
    ax_t, ax_f = axes

    ax_t.plot(col_no_freeze.T, col_no_freeze.P, label="Geotherm\nno freezing")
    ax_t.plot(col_freeze.T, col_freeze.P, label="Geotherm\nfreezing allowed")
    ax_t.plot(tsol_sat, p_line, label="Solidus")
    ax_f.plot(col_no_freeze.F, col_no_freeze.P)
    ax_f.plot(col_freeze.F, col_freeze.P)

    ax_t.invert_yaxis()
    ax_t.set_ylabel("Pressure (GPa)")
    ax_t.set_xlabel("Temperature (°C)")
    ax_f.set_xlabel("Melt Fraction")
    ax_t.legend(fontsize=8)
    ax_t.set_title(f"{h2o_wt:g} wt% H$_2$O, $T_p={tp_c:.0f}^\\circ$C, $P_{{start}}={pstart_gpa:g}$ GPa")
    fig.tight_layout()
    _save(fig, out, show)


ALL_SECTIONS = (
    "solidus",
    "dFdP",
    "dTdP",
    "dTdP_partial",
    "dTdF_partial",
    "saturated_batch",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce pyMelt HydrousMelting tutorial figures",
    )
    parser.add_argument("--tp", type=float, default=1300.0, help="Tp for § dF/dP, dT/dP plots")
    parser.add_argument("--tp-saturated", type=float, default=1200.0, help="Tp for § saturated batch")
    parser.add_argument("--pstart", type=float, default=8.0, help="Pstart (GPa) for saturated case")
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=ALL_SECTIONS,
        default=None,
        help="Run selected sections only (default: all)",
    )
    args = parser.parse_args()
    show = not args.no_show
    sections = list(args.only) if args.only else list(ALL_SECTIONS)

    def out(name: str) -> Path | None:
        return args.output_dir / f"{name}.png" if args.output_dir else None

    case: DecompressionCase | None = None
    need_case = any(s in sections for s in ("dFdP", "dTdP", "dTdP_partial", "dTdF_partial"))
    if need_case:
        print(f"Computing decompression columns at Tp={args.tp:.0f}°C …")
        case = compute_decompression_case(tp_c=args.tp)

    if "solidus" in sections:
        plot_solidus(out=out("solidus"), show=show and len(sections) == 1)

    assert case is not None or not need_case
    if case is not None:
        if "dFdP" in sections:
            plot_dfdp(case, tp_c=args.tp, out=out("dFdP"), show=show and len(sections) == 1)
        if "dTdP" in sections:
            plot_dtdp(case, tp_c=args.tp, out=out("dTdP"), show=show and len(sections) == 1)
        if "dTdP_partial" in sections:
            plot_dtdp_partial(case, tp_c=args.tp, out=out("dTdP_partial"), show=show and len(sections) == 1)
        if "dTdF_partial" in sections:
            plot_dtdf_partial(case, tp_c=args.tp, out=out("dTdF_partial"), show=show and len(sections) == 1)

    if "saturated_batch" in sections:
        print(f"Computing saturated batch case Tp={args.tp_saturated:.0f}°C …")
        plot_saturated_batch(
            tp_c=args.tp_saturated,
            pstart_gpa=args.pstart,
            out=out("saturated_batch"),
            show=show and len(sections) == 1,
        )


if __name__ == "__main__":
    main()
