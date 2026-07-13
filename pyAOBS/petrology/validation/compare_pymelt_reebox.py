"""
Compare pyMelt vs pyAOBS REEBOX-core forward models.

When ``chi`` is passed, pyMelt uses the same REEBOX-core P0/Pf as pyAOBS
(``build_reebox_column``). H on the pyMelt side uses ``spreadingCentre`` integration;
REEBOX-core uses triangular H with optional χ active weight.

Fair F comparison uses **path mean** and **F@Pf** (same pressure bounds).
KKHS02 kinematic F̄ is shown separately as reference only.

Prerequisites (user installs):
  pip install pyMelt

Example:
  python petrology/validation/compare_pymelt_reebox.py
  python petrology/validation/compare_pymelt_reebox.py --tp 1350 --chi 4 --phi 0.10
  python petrology/validation/compare_pymelt_reebox.py --plot-dir output/f_compare
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_OXIDE_KEYS = ("SiO2", "MgO", "CaO", "TiO2", "Al2O3", "FeO", "Na2O")


def _fmt_melt(m: dict[str, float] | None) -> str:
    if not m:
        return "(majors n/a)"
    return " ".join(f"{k}={m.get(k, float('nan')):.1f}" for k in _OXIDE_KEYS if k in m)


def _ree_f_metrics(ree) -> dict[str, float]:
    path = ree.isentropic_path
    f_pf = float(path.f_total) if path.steps else 0.0
    return {
        "F_path": float(path.fbar()),
        "F_pf": f_pf,
        "F_kkhs02": float(ree.fbar),
        "Fmax_rmc": float(ree.f_max),
    }


def _pm_f_metrics(pm) -> dict[str, float]:
    import numpy as np

    f_arr = np.asarray(pm.column.F)
    return {
        "F_path": float(pm.fbar),
        "F_pf": float(f_arr[-1]) if len(f_arr) else 0.0,
        "Fmax": float(pm.f_max),
    }


def _plot_f_vs_p(*, label: str, ree, pm, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    path = ree.isentropic_path
    p_ree = [st.p_gpa for st in path.steps]
    f_ree = [float(path.f_bulk_at(st)) for st in path.steps]

    col = pm.column
    p_pm = np.asarray(col.P)
    f_pm = np.asarray(col.F)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.plot(p_ree, f_ree, "o-", ms=3, lw=1.5, label="REEBOX-core (isentropic)")
    ax.plot(p_pm, f_pm, "s-", ms=3, lw=1.5, label="pyMelt (PM2001)")
    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("Bulk melt fraction F")
    ax.set_title(label)
    ax.invert_xaxis()
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def compare_one(
    *,
    tp_c: float,
    chi: float,
    pyroxenite_frac: float,
    b_km: float = 0.0,
    label: str = "",
    plot_dir: Path | None = None,
) -> dict[str, Any]:
    from petrology.melting.heterogeneous import forward_heterogeneous_column
    from petrology.melting.pymelt_bridge import forward_pymelt_column

    ree = forward_heterogeneous_column(
        tp_c=float(tp_c),
        b_km=float(b_km),
        chi=float(chi),
        pyroxenite_frac=float(pyroxenite_frac),
        n_isentropic_steps=48,
        compute_norm_vp=False,
    )
    pm = forward_pymelt_column(
        tp_c=float(tp_c),
        pyroxenite_frac=float(pyroxenite_frac),
        chi=float(chi),
        b_km=float(b_km),
    )

    rf = _ree_f_metrics(ree)
    pf = _pm_f_metrics(pm)
    tag = label or f"Tp{tp_c:.0f}_chi{chi:g}_phi{pyroxenite_frac:.2f}"

    if plot_dir is not None:
        _plot_f_vs_p(label=tag, ree=ree, pm=pm, out_path=plot_dir / f"{tag.replace(' ', '_')}.png")

    return {
        "label": tag,
        "tp_c": tp_c,
        "chi": chi,
        "phi": pyroxenite_frac,
        "b_km": b_km,
        "ree_H_km": ree.h_km,
        "pm_H_km": pm.h_km,
        "dH_km": ree.h_km - pm.h_km,
        "ree_P0": ree.p0_gpa,
        "pm_P0": pm.p0_gpa,
        "ree_Pf": ree.pf_gpa,
        "pm_Pf": pm.pf_gpa,
        "ree_F_path": rf["F_path"],
        "pm_F_path": pf["F_path"],
        "dF_path": rf["F_path"] - pf["F_path"],
        "ree_F_pf": rf["F_pf"],
        "pm_F_pf": pf["F_pf"],
        "dF_pf": rf["F_pf"] - pf["F_pf"],
        "ree_F_kkhs02": rf["F_kkhs02"],
        "ree_Fmax": rf["Fmax_rmc"],
        "pm_Fmax": pf["Fmax"],
        "ree_melt": _fmt_melt(ree.pooled_melt_wt),
        "pm_melt": _fmt_melt(pm.pooled_melt_wt),
        "pm_crust_method": pm.crust_method,
        "pm_version": pm.pymelt_version,
    }


def _print_row(r: dict) -> None:
    print(f"\n--- {r['label']} ---")
    print(f"  Tp={r['tp_c']:.0f}°C  χ={r['chi']:.1f}  Φ={r['phi']:.2f}  b={r['b_km']:.0f} km")
    print(f"  H_km:        REEBOX {r['ree_H_km']:6.1f}  |  pyMelt {r['pm_H_km']:6.1f}  (Δ {r['dH_km']:+.1f})")
    print(f"  P0/Pf:       {r['ree_P0']:.2f}/{r['ree_Pf']:.2f} GPa  |  {r['pm_P0']:.2f}/{r['pm_Pf']:.2f} GPa")
    print("  F 公平对比 (同 P0/Pf):")
    print(f"    路径平均:  REEBOX {r['ree_F_path']:.3f}  |  pyMelt {r['pm_F_path']:.3f}  (Δ {r['dF_path']:+.3f})")
    print(f"    Pf 处 F:   REEBOX {r['ree_F_pf']:.3f}  |  pyMelt {r['pm_F_pf']:.3f}  (Δ {r['dF_pf']:+.3f})")
    print(f"    Fmax:      REEBOX {r['ree_Fmax']:.3f}  |  pyMelt {r['pm_Fmax']:.3f}")
    print(f"  F 参考:      KKHS02 F̄={r['ree_F_kkhs02']:.3f} (运动学代理，非路径平均)")
    print(f"  Melt wt%:    REEBOX {r['ree_melt']}")
    print(f"               pyMelt {r['pm_melt']}")
    print(f"  pyMelt crust: {r['pm_crust_method']} (v{r['pm_version']})")


def default_cases() -> list[tuple[float, float, float, float]]:
    return [
        (1350.0, 4.0, 0.10, 20.0),
        (1300.0, 8.0, 0.10, 0.0),
        (1380.0, 16.0, 0.15, 0.0),
        (1350.0, 4.0, 0.00, 0.0),
    ]


def load_cases_csv(path: Path) -> list[tuple[float, float, float, float]]:
    out: list[tuple[float, float, float, float]] = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for raw in csv.DictReader(fh):
            out.append(
                (
                    float(raw["tp_c"]),
                    float(raw.get("chi", raw.get("lambda", 1.0))),
                    float(raw.get("phi", raw.get("pyroxenite_frac", 0.0))),
                    float(raw.get("b_km", 0.0)),
                )
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pyMelt vs REEBOX-core")
    parser.add_argument("--tp", type=float, default=None)
    parser.add_argument("--chi", type=float, default=4.0)
    parser.add_argument("--phi", type=float, default=0.10, dest="pyroxenite_frac")
    parser.add_argument("--b-km", type=float, default=20.0)
    parser.add_argument("--cases", type=Path, default=None, help="CSV: tp_c,chi,phi,b_km")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Write comparison CSV")
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Save F(P) comparison PNGs to this directory",
    )
    args = parser.parse_args()

    if args.cases:
        cases = load_cases_csv(args.cases)
    elif args.tp is not None:
        cases = [(args.tp, args.chi, args.pyroxenite_frac, args.b_km)]
    else:
        cases = default_cases()

    rows: list[dict] = []
    for i, (tp, chi, phi, b) in enumerate(cases):
        try:
            rows.append(
                compare_one(
                    tp_c=tp,
                    chi=chi,
                    pyroxenite_frac=phi,
                    b_km=b,
                    label=f"case_{i+1}",
                    plot_dir=args.plot_dir,
                )
            )
        except ImportError as exc:
            print(f"pyMelt not available: {exc}")
            print("Install: pip install pyMelt")
            sys.exit(1)
        except Exception as exc:
            print(f"Case Tp={tp} chi={chi} phi={phi} FAILED: {exc}")
            raise

    print("=" * 60)
    print("pyMelt vs REEBOX-core  (H/P 几何对齐；F 路径级对比)")
    print("=" * 60)
    for r in rows:
        _print_row(r)

    if args.plot_dir:
        print(f"\nF(P) 图已保存至 {args.plot_dir}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0].keys())
        with args.output.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV: {args.output}")


if __name__ == "__main__":
    main()
