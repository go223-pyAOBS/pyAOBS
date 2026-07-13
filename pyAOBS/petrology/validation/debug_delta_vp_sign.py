"""
P0 diagnostic: unclamped ΔVp = V_LC,cumulate − V_bulk (Fig.5 / R3).

``delta_vp_wl_fc`` clamps negative values to 0; this script exposes raw deltas,
phase fractions at F_xl, and catalog statistics.

Usage::

    py -3.11 petrology/validation/debug_delta_vp_sign.py
    py -3.11 petrology/validation/debug_delta_vp_sign.py --f 0.75 --backend sb1994_fig2ol
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
from petrology.fc.cdat_library import fig2_primary_melt
from petrology.fc.delta_vp import FIG5_P_EVAL_MPA, FIG5_T_EVAL_C, delta_vp_wl_fc
from petrology.fc.wl1990 import load_kinzler1997_morb_primary, simulate_crystallization_path
from petrology.norm_velocity import norm_velocity_from_bulk_wt, norm_velocity_from_record


def _raw_delta_vp(
    melt_oxides_wt: dict[str, float],
    *,
    f_solid: float,
    p_fc_mpa: float,
    bulk_vp_km_s: float | None = None,
    p_eval_mpa: float,
    t_eval_c: float,
    mineral_backend: str,
    cipw_backend: str,
) -> dict[str, float]:
    """V_LC − V_bulk without max(0, …) clamp."""
    p_eval_pa = float(p_eval_mpa) * 1e6
    t_eval_k = float(t_eval_c) + 273.15
    melt = {k: float(v) for k, v in melt_oxides_wt.items()}

    if bulk_vp_km_s is None:
        bulk_vp_km_s = float(
            norm_velocity_from_bulk_wt(
                melt,
                p_pa=p_eval_pa,
                t_k=t_eval_k,
                cipw_backend=cipw_backend,
                mineral_backend=mineral_backend,  # type: ignore[arg-type]
            )["vp_km_s"]
        )

    f = float(np.clip(f_solid, 0.0, 0.98))
    if f <= 1e-6:
        return {
            "bulk_vp_km_s": float(bulk_vp_km_s),
            "vlc_km_s": float(bulk_vp_km_s),
            "delta_raw_km_s": 0.0,
            "delta_clamped_km_s": 0.0,
            "cum_ol": 1.0,
            "cum_pl": 0.0,
            "cum_cpx": 0.0,
            "inc_ol": 1.0,
            "inc_pl": 0.0,
            "inc_cpx": 0.0,
        }

    states = simulate_crystallization_path(
        primary_melt_oxides_wt=melt,
        f_grid=np.array([0.0, f], dtype=float),
        path="fc_100",
        p_fc_mpa=float(p_fc_mpa),
        p_eval_mpa=float(p_eval_mpa),
        t_eval_c=float(t_eval_c),
        kd_engine="langmuir",
        fig2_ab_calibrate=False,
        cipw_backend=cipw_backend,
        mineral_backend=mineral_backend,  # type: ignore[arg-type]
    )
    s = states[-1]
    vlc = float(s.vp_cumulate_km_s)
    delta_raw = vlc - float(bulk_vp_km_s)
    return {
        "bulk_vp_km_s": float(bulk_vp_km_s),
        "vlc_km_s": vlc,
        "delta_raw_km_s": delta_raw,
        "delta_clamped_km_s": float(max(0.0, delta_raw)),
        "cum_ol": float(s.cum_ol),
        "cum_pl": float(s.cum_pl),
        "cum_cpx": float(s.cum_cpx),
        "inc_ol": float(s.inc_ol),
        "inc_pl": float(s.inc_pl),
        "inc_cpx": float(s.inc_cpx),
        "residual_vp_km_s": float(s.vp_residual_norm_km_s),
    }


def _print_anchor(label: str, melt: dict[str, float], **kw) -> None:
    r = _raw_delta_vp(melt, **kw)
    d_cl = delta_vp_wl_fc(
        melt_oxides_wt=melt,
        f_solid=kw["f_solid"],
        p_fc_mpa=kw["p_fc_mpa"],
        bulk_vp_km_s=r["bulk_vp_km_s"],
        p_eval_mpa=kw["p_eval_mpa"],
        t_eval_c=kw["t_eval_c"],
        mineral_backend=kw["mineral_backend"],
        cipw_backend=kw["cipw_backend"],
        use_cache=False,
    )
    print(f"\n=== {label} ===")
    print(
        f"  bulk={r['bulk_vp_km_s']:.3f}  V_LC={r['vlc_km_s']:.3f}  "
        f"ΔVp_raw={r['delta_raw_km_s']:+.3f}  ΔVp_clamped={r['delta_clamped_km_s']:.3f}  "
        f"(delta_vp_wl_fc={d_cl:.3f})"
    )
    print(
        f"  cum % Ol/Pl/Cpx = {100*r['cum_ol']:.0f}/{100*r['cum_pl']:.0f}/{100*r['cum_cpx']:.0f}  "
        f"inc % = {100*r['inc_ol']:.0f}/{100*r['inc_pl']:.0f}/{100*r['inc_cpx']:.0f}"
    )
    print(f"  residual norm Vp = {r['residual_vp_km_s']:.3f} km/s")


def _catalog_summary(
    *,
    f_solid: float,
    p_fc_mpa: float,
    p_eval_mpa: float,
    t_eval_c: float,
    mineral_backend: str,
    cipw_backend: str,
    top_n: int,
) -> None:
    rows = [r for r in load_melt_catalog() if r.get("include_in_regression")]
    raw: list[float] = []
    clamped: list[float] = []
    neg = pos = zero = 0
    details: list[tuple[str, float, float, float, float, float, float]] = []

    for rec in rows:
        ox = {k: v for k, v in oxides_from_record(rec).items() if k != "P2O5"}
        bulk = float(norm_velocity_from_record(rec, p_pa=p_eval_mpa * 1e6, t_k=t_eval_c + 273.15,
                                               cipw_backend=cipw_backend,
                                               mineral_backend=mineral_backend)["vp_km_s"])
        r = _raw_delta_vp(
            ox,
            f_solid=f_solid,
            p_fc_mpa=p_fc_mpa,
            bulk_vp_km_s=bulk,
            p_eval_mpa=p_eval_mpa,
            t_eval_c=t_eval_c,
            mineral_backend=mineral_backend,
            cipw_backend=cipw_backend,
        )
        dr = r["delta_raw_km_s"]
        raw.append(dr)
        clamped.append(r["delta_clamped_km_s"])
        if dr < -0.005:
            neg += 1
        elif dr > 0.005:
            pos += 1
        else:
            zero += 1
        sid = str(rec.get("id") or rec.get("source_id") or "?")[:40]
        details.append((sid, bulk, r["vlc_km_s"], dr, r["cum_ol"], r["cum_pl"], r["cum_cpx"]))

    raw_a = np.asarray(raw)
    clamp_a = np.asarray(clamped)
    print(f"\n=== Catalog (n={len(rows)}) @ F={f_solid:.2f}, P_fc={p_fc_mpa:.0f} MPa, "
          f"report {p_eval_mpa:.0f} MPa / {t_eval_c:.0f}°C, backend={mineral_backend} ===")
    print(f"  ΔVp_raw:     mean={raw_a.mean():+.3f}  std={raw_a.std():.3f}  "
          f"min={raw_a.min():+.3f}  max={raw_a.max():+.3f}")
    print(f"  ΔVp_clamped: mean={clamp_a.mean():+.3f}  std={clamp_a.std():.3f}")
    print(f"  sign: negative={neg}  ~zero={zero}  positive={pos}  "
          f"(paper target @ F=0.7–0.8: mean ~+0.15 km/s)")
    print(f"\n  Most negative ΔVp_raw (need V_LC >> bulk):")
    for row in sorted(details, key=lambda x: x[3])[:top_n]:
        sid, bulk, vlc, dr, ol, pl, cpx = row
        print(
            f"    {sid:40s} bulk={bulk:.3f} vlc={vlc:.3f} Δ={dr:+.3f}  "
            f"cum {100*ol:.0f}/{100*pl:.0f}/{100*cpx:.0f}"
        )
    print(f"\n  Most positive ΔVp_raw:")
    for row in sorted(details, key=lambda x: x[3], reverse=True)[:top_n]:
        sid, bulk, vlc, dr, ol, pl, cpx = row
        print(
            f"    {sid:40s} bulk={bulk:.3f} vlc={vlc:.3f} Δ={dr:+.3f}  "
            f"cum {100*ol:.0f}/{100*pl:.0f}/{100*cpx:.0f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Unclamped ΔVp diagnostic (R3 / Fig.5)")
    p.add_argument("--f", type=float, default=0.75, help="Solid fraction F_xl")
    p.add_argument("--p-fc", type=float, default=400.0, help="Crystallization pressure (MPa)")
    p.add_argument("--p-eval", type=float, default=FIG5_P_EVAL_MPA, help="Report P (MPa)")
    p.add_argument("--t-eval", type=float, default=FIG5_T_EVAL_C, help="Report T (°C)")
    p.add_argument("--backend", default="auto", help="mineral backend (auto|fig2|sb1994|sb1994_fig2ol|burnman)")
    p.add_argument("--cipw", default="auto", help="CIPW backend")
    p.add_argument("--top", type=int, default=5, help="Extreme catalog rows to print")
    args = p.parse_args()

    base_kw = dict(
        f_solid=float(args.f),
        p_eval_mpa=float(args.p_eval),
        t_eval_c=float(args.t_eval),
        mineral_backend=str(args.backend),
        cipw_backend=str(args.cipw),
    )
    anchor_kw = {**base_kw, "p_fc_mpa": float(args.p_fc)}

    kinzler = load_kinzler1997_morb_primary()["oxides_wt_percent"]
    fig2 = fig2_primary_melt()["oxides_wt_percent"]

    _print_anchor("Kinzler 1997 MORB primary (Fig.2 / validation default)", kinzler, **anchor_kw)
    _print_anchor("Fig.2 CDAT primary melt", fig2, **anchor_kw)

    for p_fc in (100.0, 400.0, 800.0):
        _catalog_summary(p_fc_mpa=p_fc, top_n=int(args.top), **base_kw)

    # Fig.2 report state cross-check @ 100 MPa, 100°C
    kw_fig2 = {**anchor_kw, "p_eval_mpa": 100.0, "t_eval_c": 100.0}
    print("\n=== Fig.2 report state (100 MPa, 100°C) — Kinzler primary ===")
    _print_anchor("Kinzler @ Fig.2 P/T", kinzler, **kw_fig2)


if __name__ == "__main__":
    main()
