"""
Validate pyMelt vs REEBOX-core on key parameters (table only).

For full tiered checks, CSV export, and plots see:
  python petrology/validation/pymelt_reebox_benchmark.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DEFAULT_CASES = [
    (1350.0, 4.0, 0.10, 20.0),
    (1300.0, 8.0, 0.10, 0.0),
    (1380.0, 16.0, 0.15, 0.0),
    (1350.0, 4.0, 0.00, 0.0),
    (1350.0, 1.0, 0.10, 0.0),
]


def validate_case(
    *,
    tp_c: float,
    chi: float,
    phi: float,
    b_km: float,
    geometry: str,
    label: str,
) -> dict:
    from petrology.melting.heterogeneous import forward_heterogeneous_column
    from petrology.melting.pymelt_bridge import forward_pymelt_column

    ree = forward_heterogeneous_column(
        tp_c=tp_c,
        b_km=b_km,
        chi=chi,
        pyroxenite_frac=phi,
        n_isentropic_steps=48,
        geometry=geometry,
    )
    pm = forward_pymelt_column(
        tp_c=tp_c,
        pyroxenite_frac=phi,
        chi=chi,
        b_km=b_km,
        align_geometry=True,
    )

    path = ree.isentropic_path
    ree_f_path = float(path.fbar())
    ree_f_pf = float(path.f_total)

    import numpy as np

    f_pm = np.asarray(pm.column.F)
    pm_f_path = float(pm.fbar)
    pm_f_pf = float(f_pm[-1]) if len(f_pm) else 0.0

    return {
        "label": label,
        "geometry": geometry,
        "tp_c": tp_c,
        "chi": chi,
        "phi": phi,
        "b_km": b_km,
        "ree_P0": ree.p0_gpa,
        "pm_P0": pm.p0_gpa,
        "dP0": ree.p0_gpa - pm.p0_gpa,
        "ree_Pf": ree.pf_gpa,
        "pm_Pf": pm.pf_gpa,
        "dPf": ree.pf_gpa - pm.pf_gpa,
        "ree_H": ree.h_km,
        "pm_H": pm.h_km,
        "dH": ree.h_km - pm.h_km,
        "ree_F_path": ree_f_path,
        "pm_F_path": pm_f_path,
        "dF_path": ree_f_path - pm_f_path,
        "ree_F_pf": ree_f_pf,
        "pm_F_pf": pm_f_pf,
        "dF_pf": ree_f_pf - pm_f_pf,
        "ree_Fmax": ree.f_max,
        "pm_Fmax": pm.f_max,
        "pm_crust": pm.crust_method,
        "pm_version": pm.pymelt_version,
    }


def _print_row(r: dict) -> None:
    print(f"\n--- {r['label']}  [{r['geometry']}] ---")
    print(f"  Tp={r['tp_c']:.0f}°C  χ={r['chi']:.1f}  Φ={r['phi']:.2f}  b={r['b_km']:.0f} km")
    print(f"  P0 (GPa):  REEBOX {r['ree_P0']:.3f}  pyMelt {r['pm_P0']:.3f}  Δ {r['dP0']:+.3f}")
    print(f"  Pf (GPa):  REEBOX {r['ree_Pf']:.3f}  pyMelt {r['pm_Pf']:.3f}  Δ {r['dPf']:+.3f}")
    print(f"  H (km):    REEBOX {r['ree_H']:.1f}  pyMelt {r['pm_H']:.1f}  Δ {r['dH']:+.1f}")
    print(f"  F_path:    REEBOX {r['ree_F_path']:.3f}  pyMelt {r['pm_F_path']:.3f}  Δ {r['dF_path']:+.3f}")
    print(f"  F@Pf:      REEBOX {r['ree_F_pf']:.3f}  pyMelt {r['pm_F_pf']:.3f}  Δ {r['dF_pf']:+.3f}")
    print(f"  Fmax:      REEBOX {r['ree_Fmax']:.3f}  pyMelt {r['pm_Fmax']:.3f}")
    print(f"  pyMelt H method: {r['pm_crust']} (v{r['pm_version']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="pyMelt key-parameter validation vs REEBOX-core")
    parser.add_argument(
        "--geometry",
        choices=("reebox", "kkhs02"),
        default="reebox",
        help="REEBOX-core geometry (default: self-consistent reebox)",
    )
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for i, (tp, chi, phi, b) in enumerate(DEFAULT_CASES):
        rows.append(
            validate_case(
                tp_c=tp,
                chi=chi,
                phi=phi,
                b_km=b,
                geometry=args.geometry,
                label=f"case_{i + 1}",
            )
        )

    print("=" * 64)
    print("pyMelt 关键参数验证 vs REEBOX-core")
    print(f"  REEBOX geometry = {args.geometry}")
    print("  关键量: P0, Pf, H, F_path, F@Pf, Fmax")
    print("=" * 64)
    for r in rows:
        _print_row(r)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {args.output}")


if __name__ == "__main__":
    main()
