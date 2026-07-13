"""
Smoke-test vendored / pip pyrolite + burnman for the KKHS02 norm-Vp chain.

Run from repo root (WSL example):

    cd pyAOBS
    python -m petrology.validation.validate_backends

Use ``--vendored-only`` to force in-tree copies under ``petrology/pyrolite`` and
``petrology/burnman``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.config import PetrologyConfig
from petrology.data.load_catalog import load_melt_catalog
from petrology.norm_velocity import norm_velocity_from_bulk_wt, norm_velocity_from_record
from petrology.vendored import burnman_available, pyrolite_available, vendored_path
from petrology.vp_regression import load_eq1, predict_vp_km_s


def _print_backend_status(prefer_pip: bool) -> None:
    print("=== Backend availability ===")
    print(f"  pyrolite vendored : {vendored_path('pyrolite') / 'pyrolite'}")
    print(f"  burnman vendored  : {vendored_path('burnman') / 'burnman'}")
    print(f"  pyrolite import   : {pyrolite_available(prefer_pip=prefer_pip)}")
    print(f"  burnman import    : {burnman_available(prefer_pip=prefer_pip)}")


def _compare_morb_anchor(cfg: PetrologyConfig) -> None:
    rows = load_melt_catalog()
    rec = next((r for r in rows if r["id"] == "k97_grid_P1_F0.1"), rows[0])
    print(f"\n=== Sample melt: {rec['id']} (P={rec['P_melt_GPa']} GPa, F={rec['F_melt']}) ===")

    combos = [
        ("fallback + empirical", {"cipw_backend": "fallback", "mineral_backend": "empirical"}),
        ("pyrolite + empirical", {"cipw_backend": "pyrolite", "mineral_backend": "empirical"}),
        ("pyrolite + burnman", {"cipw_backend": "pyrolite", "mineral_backend": "burnman"}),
        ("auto + auto", {"cipw_backend": "auto", "mineral_backend": "auto"}),
    ]
    for label, kw in combos:
        try:
            nv = norm_velocity_from_record(rec, config=cfg, **kw)
            print(
                f"  {label:24s}  Vp={nv['vp_km_s']:.3f} km/s  "
                f"(cipw={nv.get('cipw_backend')}, mineral={nv.get('mineral_backend')})  "
                f"Fo={nv['Fo']:.2f} An={nv['An']:.2f}"
            )
        except Exception as exc:
            print(f"  {label:24s}  ERROR: {exc}")

    eq1 = load_eq1()
    vp_eq = predict_vp_km_s(rec["P_melt_GPa"], rec["F_melt"], eq1=eq1)
    print(f"\n  paper eq.(1) @ (P,F)     : {vp_eq:.3f} km/s")


def _burnman_single_minerals(cfg: PetrologyConfig) -> None:
    from petrology.minerals import REF_P_PA, REF_T_K, phase_properties

    print("\n=== BurnMan SLB_2022 @ 600 MPa, 400 °C ===")
    for ph in ("olivine", "plagioclase", "clinopyroxene", "quartz", "ilmenite"):
        try:
            p = phase_properties(
                ph,
                fo=0.90,
                an=0.50,
                cpx_mg=0.85,
                p_pa=REF_P_PA,
                t_k=REF_T_K,
                backend="burnman",
                prefer_pip=cfg.prefer_pip_vendored,
            )
            print(
                f"  {ph:16s}  Vp={p['vp_km_s']:.3f}  rho={p['rho_g_cm3']:.3f}  "
                f"K={p['k_gpa']:.1f}  G={p['g_gpa']:.1f}"
            )
        except Exception as exc:
            print(f"  {ph:16s}  ERROR: {exc}")
            # Keep diagnostics readable by showing the fallback value as well.
            try:
                p = phase_properties(
                    ph,
                    fo=0.90,
                    an=0.50,
                    cpx_mg=0.85,
                    p_pa=REF_P_PA,
                    t_k=REF_T_K,
                    backend="empirical",
                    prefer_pip=cfg.prefer_pip_vendored,
                )
                print(
                    f"  {'(fallback empirical)':16s}  Vp={p['vp_km_s']:.3f}  "
                    f"rho={p['rho_g_cm3']:.3f}  K={p['k_gpa']:.1f}  G={p['g_gpa']:.1f}"
                )
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate pyrolite + burnman backends")
    parser.add_argument(
        "--vendored-only",
        action="store_true",
        help="Force petrology/pyrolite and petrology/burnman (ignore pip copies)",
    )
    parser.add_argument("--json", type=Path, help="Write summary JSON")
    args = parser.parse_args()

    cfg = PetrologyConfig.reproduction()
    cfg = PetrologyConfig(
        track=cfg.track,
        cipw_backend=cfg.cipw_backend,
        mineral_backend=cfg.mineral_backend,
        reference_p_pa=cfg.reference_p_pa,
        reference_t_k=cfg.reference_t_k,
        prefer_pip_vendored=not args.vendored_only,
    )

    _print_backend_status(prefer_pip=cfg.prefer_pip_vendored)
    _burnman_single_minerals(cfg)
    _compare_morb_anchor(cfg)

    if args.json:
        rec = next(r for r in load_melt_catalog() if r["id"] == "k97_grid_P1_F0.1")
        summary = {
            "prefer_pip": cfg.prefer_pip_vendored,
            "pyrolite_ok": pyrolite_available(prefer_pip=cfg.prefer_pip_vendored),
            "burnman_ok": burnman_available(prefer_pip=cfg.prefer_pip_vendored),
            "auto_norm_vp": norm_velocity_from_record(rec, config=cfg),
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
