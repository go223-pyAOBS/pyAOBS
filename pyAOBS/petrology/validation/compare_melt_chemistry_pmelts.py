"""
Compare Kinzler (1997) vs pMELTS KLB-1 melt chemistry and norm Vp @ (P, F).

  py -3.11 petrology/validation/compare_melt_chemistry_pmelts.py
  py -3.11 petrology/validation/compare_melt_chemistry_pmelts.py --p 1 2 3 --f 0.05 0.1 0.15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Kinzler vs pMELTS melt chemistry")
    parser.add_argument("--p", type=float, nargs="+", default=[1.0, 2.0, 3.0], metavar="GPa")
    parser.add_argument("--f", type=float, nargs="+", default=[0.05, 0.10, 0.15], metavar="F")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    from petrology.melting.melt_chemistry import melt_oxides, pmelts_klb1_grid_path
    from petrology.norm_velocity import norm_velocity_from_bulk_wt
    from petrology.vp_regression import predict_vp_km_s

    print(f"pMELTS grid: {pmelts_klb1_grid_path()}")
    print(f"{'P':>5} {'F':>6}  {'Vp_K':>7} {'Vp_P':>7} {'dVp':>7}  SiO2_K SiO2_P  MgO_K MgO_P")
    print("-" * 72)

    for p in args.p:
        for f in args.f:
            ox_k = melt_oxides(p, f, chemistry_backend="kinzler1997")
            ox_p = melt_oxides(p, f, chemistry_backend="pmelts_klb1")
            vk = norm_velocity_from_bulk_wt(ox_k)["vp_km_s"]
            vp = norm_velocity_from_bulk_wt(ox_p)["vp_km_s"]
            eq1 = predict_vp_km_s(p, f)
            print(
                f"{p:5.1f} {f:6.3f}  {vk:7.3f} {vp:7.3f} {vp - vk:+7.3f}  "
                f"{ox_k['SiO2']:5.1f} {ox_p['SiO2']:5.1f}  "
                f"{ox_k['MgO']:4.1f} {ox_p['MgO']:4.1f}  (eq1={eq1:.3f})"
            )

    # REEBOX column spot check (pymelt lithology uses pmelts chemistry)
    from petrology.melting.heterogeneous import forward_heterogeneous_column

    het = forward_heterogeneous_column(
        tp_c=1450.0,
        b_km=0.0,
        chi=8.0,
        pyroxenite_frac=0.0,
        lithology_backend="pymelt",
        peridotite_lith="katz_lherzolite",
        compute_norm_vp=True,
    )
    print()
    print(
        f"REEBOX pymelt column @ Tp=1450 chi=8: "
        f"Fbar={het.fbar:.3f} Vp_norm={het.vp_bulk_norm_km_s:.3f} eq1={het.vp_bulk_eq1_km_s:.3f}"
    )
    print(f"  pooled SiO2={het.pooled_melt_wt.get('SiO2', 0):.1f} wt%")

    if args.show:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        fs = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
        p = 1.0
        vk = [norm_velocity_from_bulk_wt(melt_oxides(p, f, chemistry_backend="kinzler1997"))["vp_km_s"] for f in fs]
        vp = [norm_velocity_from_bulk_wt(melt_oxides(p, f, chemistry_backend="pmelts_klb1"))["vp_km_s"] for f in fs]
        ve = [predict_vp_km_s(p, f) for f in fs]
        ax.plot(fs, vk, "o-", label="Kinzler → norm Vp")
        ax.plot(fs, vp, "s-", label="pMELTS → norm Vp")
        ax.plot(fs, ve, "k--", label="eq.(1)")
        ax.set_xlabel("F")
        ax.set_ylabel("Vp (km/s) @ 600 MPa, 400°C")
        ax.set_title(f"Melt chemistry backends @ P={p:g} GPa")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = Path(__file__).resolve().parents[1] / "figures" / "compare_melt_chemistry_p1.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"Wrote {out}")
        plt.show()


if __name__ == "__main__":
    main()
