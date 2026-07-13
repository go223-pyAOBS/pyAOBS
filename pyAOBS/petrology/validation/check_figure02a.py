"""
Compare **computed** Fig.2(a) Vp against user digitization (figure02a数值化.txt).

This script does NOT feed digitized points into the model.  It runs the physics
chain and reports gaps vs the paper curves.

Modes
-----
heuristic      — full wl1990 mass balance + HS mixing (``fig2_ab_calibrate=False``)
langmuir       — BASALT+langmuir STATE phases + HS/CIPW Vp (``kd_engine=langmuir``, no Fig.2 anchors)
phases_physics — paper Fig.2c-h phases + HS/CIPW Vp (``fig2_ab_calibrate=True``, default Korenaga)
anchor         — sparse anchor Vp interpolation (``fig2_ab_anchor_vp=True``)
paper_hs       — paper phases only, standalone HS (diagnostic)

Usage::

    py -3.11 petrology/validation/check_figure02a.py
    py -3.11 petrology/validation/check_figure02a.py --kd-engine langmuir --mineral-backend sb1994
    py -3.11 petrology/validation/check_figure02a.py --kd-engine langmuir --mineral-backend fig2,sb1994
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.assemblage import assemblage_vp_rho
from petrology.fc.cdat_library import fig2_primary_melt
from petrology.fc.fig2_ab import (
    cumulative_phases_pct,
    incremental_compositions,
    incremental_modes_from_cumulative,
)
from petrology.fc.figure02a_digitized import (
    CurveKey,
    digitized_cum_vp_km_s,
    digitized_inc_vp_km_s,
    digitized_residual_vp_km_s,
    load_figure02a_curves,
)
from petrology.fc.wl1990 import simulate_crystallization_path
from petrology.norm_velocity import norm_velocity_from_bulk_wt

P_EVAL_MPA = 100.0
T_EVAL_C = 100.0
P_EVAL_PA = P_EVAL_MPA * 1e6
T_EVAL_K = T_EVAL_C + 273.15

# Map digitized curve → (wl path, quantity)
_CURVE_SPEC: dict[CurveKey, tuple[str, str]] = {
    "cum_sol_1kb": ("fc_100", "vp_cumulate_km_s"),
    "cum_sol_8_1kb": ("polybaric_fc", "vp_cumulate_km_s"),
    "eq_sol_1kb": ("eq_100", "vp_eq_solid_km_s"),
    "inc_sol_1kb": ("fc_100", "vp_incremental_km_s"),
    "inc_sol_8_1kb": ("polybaric_fc", "vp_incremental_km_s"),
    "frac_res_1kb": ("fc_100", "vp_residual_norm_km_s"),
    "frac_res_8_1kb": ("polybaric_fc", "vp_residual_norm_km_s"),
    "eq_res_1kb": ("eq_100", "vp_residual_norm_km_s"),
}


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _ref_y(curve_key: CurveKey, f: float) -> float:
    if curve_key == "eq_res_1kb":
        return digitized_residual_vp_km_s("eq_100", f)
    if curve_key.startswith("frac_res"):
        path = "polybaric_fc" if "8-1kb" in curve_key else "fc_100"
        return digitized_residual_vp_km_s(path, f)
    if curve_key.startswith("cum_") or curve_key == "eq_sol_1kb":
        path = {"cum_sol_1kb": "fc_100", "cum_sol_8_1kb": "polybaric_fc", "eq_sol_1kb": "eq_100"}[
            curve_key
        ]
        return digitized_cum_vp_km_s(path, f)  # type: ignore[arg-type]
    path = "fc_100" if curve_key == "inc_sol_1kb" else "polybaric_fc"
    return digitized_inc_vp_km_s(path, f)


def _paper_hs_vp(path: str, f: float, *, mineral_backend: str) -> tuple[float, float, float]:
    """Vp from paper phase schedule + HS mixing (no anchor lookup)."""
    f = float(f)
    ol_p, cpx_p, pl_p = cumulative_phases_pct(path, f)  # type: ignore[arg-type]
    ol_c, pl_c, cpx_c = ol_p / 100.0, pl_p / 100.0, cpx_p / 100.0
    fo, an, di = incremental_compositions(path, f)  # type: ignore[arg-type]
    f0 = max(0.0, f - 0.02)
    ol_i, pl_i, cpx_i = incremental_modes_from_cumulative(path, f, f0)  # type: ignore[arg-type]

    vp_cum, _ = assemblage_vp_rho(
        ol_frac=ol_c, pl_frac=pl_c, cpx_frac=cpx_c,
        fo_pct=fo, an_pct=an, di_pct=di,
        p_pa=P_EVAL_PA, t_k=T_EVAL_K, mineral_backend=mineral_backend,
    )
    vp_inc, _ = assemblage_vp_rho(
        ol_frac=ol_i, pl_frac=pl_i, cpx_frac=cpx_i,
        fo_pct=fo, an_pct=an, di_pct=di,
        p_pa=P_EVAL_PA, t_k=T_EVAL_K, mineral_backend=mineral_backend,
    )
    if path == "eq_100":
        vp_eq = vp_cum
    else:
        vp_eq = float("nan")
    return vp_cum, vp_inc, vp_eq


def _simulate(
    path: str,
    f_grid: np.ndarray,
    *,
    kd_engine: str,
    fig2_ab_calibrate: bool,
    fig2_ab_anchor_vp: bool = False,
    mineral_backend: str,
) -> dict[float, object]:
    melt = fig2_primary_melt()["oxides_wt_percent"]
    st = simulate_crystallization_path(
        primary_melt_oxides_wt=melt,
        f_grid=f_grid,
        path=path,  # type: ignore[arg-type]
        mineral_backend=mineral_backend,
        cipw_backend="fallback",
        fig2_ab_calibrate=fig2_ab_calibrate,
        fig2_ab_anchor_vp=fig2_ab_anchor_vp,
        kd_engine=kd_engine,  # type: ignore[arg-type]
    )
    return {float(s.f_solid): s for s in st}


def _compare_curve(
    curve_key: CurveKey,
    *,
    kd_engine: str,
    mineral_backends: tuple[str, ...],
    modes: tuple[str, ...],
) -> None:
    dig = load_figure02a_curves()[curve_key]
    f_pts = np.array([p[0] for p in dig], dtype=float)
    ref = np.array([p[1] for p in dig], dtype=float)
    path, attr = _CURVE_SPEC[curve_key]

    print(f"\n--- {curve_key}  (n={len(f_pts)} digitized points) ---")
    print(f"{'backend':8s} {'mode':10s}  {'RMS':>7s}  {'max|d|':>7s}  {'bias':>7s}")
    for mb in mineral_backends:
        by_mode = _curve_modes_for_backend(
            f_pts, path, attr,
            kd_engine=kd_engine, mineral_backend=mb, modes=modes,
        )
        for mode, got in by_mode.items():
            mask = np.isfinite(got) & np.isfinite(ref)
            if not np.any(mask):
                print(f"  {mb:8s} {mode:10s}  (no finite values)")
                continue
            d = got[mask] - ref[mask]
            print(
                f"  {mb:8s} {mode:10s}  {_rms(got[mask], ref[mask]):7.4f}  "
                f"{float(np.max(np.abs(d))):7.4f}  {float(np.mean(d)):+7.4f}"
            )

        key_mode = "langmuir" if "langmuir" in by_mode else "heuristic"
        if key_mode in by_mode:
            d = by_mode[key_mode] - ref
            i = int(np.argmax(np.abs(d)))
            print(
                f"  worst {mb}/{key_mode} @ F={f_pts[i]:.3f}: ref={ref[i]:.3f}  "
                f"code={by_mode[key_mode][i]:.3f}  d={d[i]:+.3f}"
            )


def _curve_modes_for_backend(
    f_pts: np.ndarray,
    path: str,
    attr: str,
    *,
    kd_engine: str,
    mineral_backend: str,
    modes: tuple[str, ...],
) -> dict[str, np.ndarray]:
    f_unique = np.unique(np.round(f_pts, 4))
    f_sim = np.sort(np.unique(np.concatenate([f_unique, [0.0]])))
    by_mode: dict[str, np.ndarray] = {}

    if (
        "heuristic" in modes
        or "langmuir" in modes
        or "phases_physics" in modes
        or "anchor" in modes
        or kd_engine != "heuristic"
    ):
        for label, cal, anchor, engine in [
            ("heuristic", False, False, "heuristic"),
            ("langmuir", False, False, "langmuir"),
            ("phases_physics", True, False, "heuristic"),
            ("anchor", True, True, "heuristic"),
        ]:
            if label not in modes:
                continue
            use_engine = kd_engine if label == "heuristic" and kd_engine != "heuristic" else engine
            states = _simulate(
                path, f_sim,
                kd_engine=use_engine,
                fig2_ab_calibrate=cal,
                fig2_ab_anchor_vp=anchor,
                mineral_backend=mineral_backend,
            )
            vals = []
            for f in f_pts:
                s = states.get(float(f)) or states.get(float(np.round(f, 4)))
                if s is None:
                    fk = min(states.keys(), key=lambda k: abs(k - f))
                    s = states[fk]
                v = getattr(s, attr)
                if v is None and attr == "vp_eq_solid_km_s":
                    v = s.vp_cumulate_km_s
                vals.append(float(v))
            by_mode[label] = np.array(vals)

    if "paper_hs" in modes:
        vals = []
        for f in f_pts:
            vp_cum, vp_inc, vp_eq = _paper_hs_vp(path, f, mineral_backend=mineral_backend)
            if attr == "vp_cumulate_km_s":
                vals.append(vp_cum)
            elif attr == "vp_incremental_km_s":
                vals.append(vp_inc)
            elif attr == "vp_eq_solid_km_s":
                vals.append(vp_eq)
            else:
                vals.append(float("nan"))
        by_mode["paper_hs"] = np.array(vals)

    return by_mode


def _phase_gap_fc(*, kd_engine: str = "heuristic") -> None:
    """Phase budget vs paper Fig.2c at selected F."""
    from petrology.fc.fig2_ab import cumulative_phases_pct as paper_phases

    f_check = np.array([0.08, 0.22, 0.40, 0.55, 0.70, 0.80])
    states = _simulate(
        "fc_100", f_check,
        kd_engine=kd_engine,
        fig2_ab_calibrate=False,
        mineral_backend="fig2",
    )
    label = "langmuir STATE" if kd_engine == "langmuir" else f"{kd_engine} heuristic"
    print(f"\n=== Phase path gap (fc_100 {label} vs paper Fig.2c) ===")
    print(f"{'F':>5s}  {'Ol%':>5s} {'dOl':>6s}  {'Pl%':>5s} {'dPl':>6s}  {'Cpx%':>5s} {'dCpx':>6s}")
    for f in f_check:
        s = states[float(f)]
        ol_p, cpx_p, pl_p = paper_phases("fc_100", float(f))
        ol = 100 * s.cum_ol
        pl = 100 * s.cum_pl
        cpx = 100 * s.cum_cpx
        print(
            f"{f:5.2f}  {ol:5.1f} {ol-ol_p:+6.1f}  {pl:5.1f} {pl-pl_p:+6.1f}  "
            f"{cpx:5.1f} {cpx-cpx_p:+6.1f}"
        )


def _primary_melt_vp(*, mineral_backends: tuple[str, ...]) -> None:
    melt = fig2_primary_melt()["oxides_wt_percent"]
    print(f"\n=== Primary melt norm Vp @100 MPa, 100°C ===")
    for mb in mineral_backends:
        bulk = norm_velocity_from_bulk_wt(
            melt, p_pa=P_EVAL_PA, t_k=T_EVAL_K, cipw_backend="fallback", mineral_backend=mb,
        )
        print(f"  {mb:8s}  code={bulk['vp_km_s']:.3f} km/s   backend={bulk.get('mineral_backend')}")
    print("  paper horizontal line ≈ 7.17 km/s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fig.2(a): computed Vp vs digitized reference")
    parser.add_argument("--kd-engine", default="heuristic", choices=("heuristic", "langmuir", "basalt1990"))
    parser.add_argument("--mineral-backend", default="fig2")
    parser.add_argument(
        "--modes",
        default="heuristic,phases_physics,anchor",
        help="comma-separated: heuristic, langmuir, phases_physics, paper_hs, anchor",
    )
    args = parser.parse_args()
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    mineral_backends = tuple(m.strip() for m in args.mineral_backend.split(",") if m.strip())

    dig = load_figure02a_curves()
    print("=== Reference: petrology/data/figure02a数值化.txt ===")
    for key, pts in dig.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        print(f"  {key:18s}  n={len(pts):2d}  F=[{min(xs):.3f},{max(xs):.3f}]  Vp=[{min(ys):.3f},{max(ys):.3f}]")

    _primary_melt_vp(mineral_backends=mineral_backends)
    _phase_gap_fc(kd_engine=args.kd_engine)

    print("\n=== Computed vs digitized (panel a) ===")
    print(f"kd_engine={args.kd_engine}  mineral_backends={mineral_backends}  modes={modes}")

    for curve_key in dig:
        _compare_curve(
            curve_key,
            kd_engine=args.kd_engine,
            mineral_backends=mineral_backends,
            modes=modes,
        )

    print("\n=== Interpretation ===")
    print("  - langmuir + sb1994: Korenaga §2.1 chain (STATE phases + S&B Table I + HS)")
    print("  - large langmuir RMS on cum/inc: crystallization path (phases) dominates over mineral backend")
    print("  - sb1994 vs fig2 bias shift: ~uniform offset if phases match; fig2 is anchor-calibrated BurnMan")
    print("  - paper_hs: paper Fig.2c-h phases only (isolates phase schedule from FC path)")
    print("  - anchor mode: sparse knot interpolation only (~0.05 RMS on cum vs full digitization)")


if __name__ == "__main__":
    main()
