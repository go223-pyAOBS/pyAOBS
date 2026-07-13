"""
pyMelt vs REEBOX-core benchmark: 检测值表 + 对比图 + 共性问题交叉检验.

关键物理量分三类:
  A shared_geometry   — 应对齐 (P0, Pf)
  B shared_algorithm  — 同 F(P)、χ=1 时 H 应一致 (三角积分 vs spreadingCentre)
  C different_physics — 等熵 vs PM2001，仅作信息对比

共性问题: 用 pyMelt 的 F(P) 喂给本代码 ``triangular_crust_thickness_km`` (χ=1)，
          应与 pyMelt ``spreadingCentre`` 给出相同 H（检验积分实现，而非熔融物理).

  python petrology/validation/pymelt_reebox_benchmark.py
  python petrology/validation/pymelt_reebox_benchmark.py --out-dir petrology/figures/pymelt_benchmark
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.validation.pymelt_reebox_common import PARAM_SPECS, run_benchmark_case

DEFAULT_CASES = [
    (1350.0, 4.0, 0.10, 20.0),
    (1300.0, 8.0, 0.10, 0.0),
    (1380.0, 16.0, 0.15, 0.0),
    (1350.0, 4.0, 0.00, 0.0),
    (1350.0, 1.0, 0.10, 0.0),
]

TIER_ZH = {
    "shared_geometry": "A 几何共用",
    "shared_algorithm": "B 算法共用",
    "different_physics": "C 物理不同",
}


def _print_case(result: dict) -> None:
    print(f"\n{'=' * 70}")
    print(
        f"{result['label']}  [{result['geometry']}]  "
        f"Tp={result['tp_c']:.0f}°C χ={result['chi']:.1f} Φ={result['phi']:.2f} b={result['b_km']:.0f}km"
    )
    print(f"pyMelt v{result['pm_version']}  crust={result['pm_crust_method']}")
    print(f"{'参数':<22} {'层级':<12} {'REEBOX':>10} {'pyMelt':>10} {'Δ':>10} {'判定':>6}")
    print("-" * 70)
    for c in result["checks"]:
        pm_s = f"{c['pm']:10.4f}" if c["pm"] is not None else f"{'—':>10}"
        delta_s = f"{c['delta']:+10.4f}" if c["pm"] is not None else f"{'—':>10}"
        if c["pass"] is True:
            verdict = "PASS"
        elif c["pass"] is False:
            verdict = "FAIL"
        else:
            verdict = "INFO"
        print(f"{c['label']:<22} {TIER_ZH[c['tier']]:<12} {c['ree']:10.4f} {pm_s} {delta_s} {verdict:>6}")
    print(f"小结: PASS={result['n_pass']} FAIL={result['n_fail']} INFO={result['n_na']}")


def _export_checks_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case", "geometry", "tp_c", "chi", "phi", "b_km",
        "param", "tier", "unit", "ree", "pm", "delta", "tol", "pass", "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results:
            for c in r["checks"]:
                w.writerow(
                    {
                        "case": r["label"],
                        "geometry": r["geometry"],
                        "tp_c": r["tp_c"],
                        "chi": r["chi"],
                        "phi": r["phi"],
                        "b_km": r["b_km"],
                        "param": c["key"],
                        "tier": c["tier"],
                        "unit": c["unit"],
                        "ree": c["ree"],
                        "pm": c["pm"],
                        "delta": c["delta"],
                        "tol": c["tol"],
                        "pass": c["pass"],
                        "note": c["note"],
                    }
                )


def _plot_case(result: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 8), dpi=120)
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(result["p_ree"], result["f_ree"], "o-", ms=3, lw=1.4, label="REEBOX isentropic F(P)")
    ax0.plot(result["p_pm"], result["f_pm"], "s-", ms=3, lw=1.4, label="pyMelt PM2001 F(P)")
    ax0.set_xlabel("P (GPa)")
    ax0.set_ylabel("Bulk F")
    ax0.invert_xaxis()
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)
    ax0.set_title("(a) Melt fraction vs P")

    ax1 = fig.add_subplot(gs[0, 1])
    tier_ab = [c for c in result["checks"] if c["tier"] in ("shared_geometry", "shared_algorithm")]
    labels = [c["key"] for c in tier_ab]
    deltas = [c["delta"] for c in tier_ab]
    colors = [
        "#2ca02c" if c["pass"] is True else "#d62728" if c["pass"] is False else "#7f7f7f"
        for c in tier_ab
    ]
    ax1.barh(labels, deltas, color=colors, alpha=0.85)
    ax1.axvline(0, color="k", lw=0.8)
    ax1.set_xlabel("REEBOX - pyMelt")
    ax1.set_title("(b) Tier A/B deltas (green=PASS)")

    ax2 = fig.add_subplot(gs[1, 0])
    h_labels = ["REEBOX H", "pyMelt SC", "integ+pmF", "integ+reeF"]
    h_vals = [
        result["ree_h_km"],
        result["h_pm_spreading"],
        result["h_common_ree_int"],
        result["h_passive_ree_f"],
    ]
    bars = ax2.bar(h_labels, h_vals, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"], alpha=0.85)
    ax2.set_ylabel("H (km)")
    ax2.set_title("(c) H: orange vs green should match (chi=1, same F)")
    for b, v in zip(bars, h_vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    hc = next(c for c in result["checks"] if c["key"] == "H_common")
    fr = next(c for c in result["checks"] if c["key"] == "F_rmse")
    lines = [
        f"Case: {result['label']}  geometry={result['geometry']}",
        f"Tp={result['tp_c']:.0f}C  chi={result['chi']:.1f}  phi={result['phi']:.2f}",
        "",
        "Tier A: P0/Pf aligned",
        "Tier B: H_common = our integrator on pyMelt F",
        "Tier C: F/H production (isentropic vs PM2001)",
        "",
        f"H_common: delta={hc['delta']:+.2f} km  pass={hc['pass']}",
        f"F RMSE: {fr['ree']:.4f}",
    ]
    ax3.text(0.02, 0.98, "\n".join(lines), transform=ax3.transAxes, va="top", fontsize=9, family="monospace")

    fig.suptitle(f"pyMelt vs REEBOX — {result['label']}", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_summary_grid(results: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
    cases = [r["label"] for r in results]

    for ax, key, title in zip(
        axes,
        ["P0", "Pf", "H_common"],
        ["dP0 (GPa)", "dPf (GPa)", "dH_common (km)"],
    ):
        deltas = []
        colors = []
        for r in results:
            c = next(x for x in r["checks"] if x["key"] == key)
            deltas.append(c["delta"])
            if c["pass"] is True:
                colors.append("#2ca02c")
            elif c["pass"] is False:
                colors.append("#d62728")
            else:
                colors.append("#7f7f7f")
        ax.bar(cases, deltas, color=colors, alpha=0.85)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        tol = next(s.tol for s in PARAM_SPECS if s.key == key)
        if tol is not None:
            ax.axhline(tol, color="r", ls="--", lw=0.7, alpha=0.5)
            ax.axhline(-tol, color="r", ls="--", lw=0.7, alpha=0.5)

    fig.suptitle("Multi-case shared deltas", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="pyMelt vs REEBOX benchmark with plots")
    parser.add_argument("--geometry", choices=("reebox", "kkhs02"), default="reebox")
    parser.add_argument("--out-dir", type=Path, default=Path("petrology/figures/pymelt_benchmark"))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--show-pymelt-warnings",
        action="store_true",
        help="Show pyMelt geosettings divide-by-zero warnings (harmless at F≈0)",
    )
    args = parser.parse_args()

    if not args.show_pymelt_warnings:
        warnings.filterwarnings(
            "ignore",
            message="divide by zero encountered in divide",
            category=RuntimeWarning,
            module=r"pyMelt\.geosettings",
        )

    results: list[dict] = []
    for i, (tp, chi, phi, b) in enumerate(DEFAULT_CASES):
        try:
            results.append(
                run_benchmark_case(
                    tp_c=tp,
                    chi=chi,
                    phi=phi,
                    b_km=b,
                    label=f"case_{i + 1}",
                    geometry=args.geometry,
                )
            )
        except ImportError as exc:
            print(f"pyMelt not installed: {exc}\n  pip install pyMelt")
            sys.exit(1)

    print("=" * 70)
    print("pyMelt vs REEBOX-core benchmark")
    print(f"  geometry={args.geometry}")
    print("  A: P0/Pf  |  B: H_common (same F, chi=1)  |  C: F/H production")
    print("=" * 70)
    for r in results:
        _print_case(r)

    out = args.out_dir
    _export_checks_csv(results, out / "benchmark_checks.csv")
    print(f"\nCSV: {out / 'benchmark_checks.csv'}")

    if not args.no_plots:
        for r in results:
            _plot_case(r, out / f"{r['label']}_benchmark.png")
        _plot_summary_grid(results, out / "benchmark_summary.png")
        print(f"Plots: {out}/")


if __name__ == "__main__":
    main()
