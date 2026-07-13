"""Quick comparison: sb1994 vs fig2 @ KKHS02 Fig.2 report state (100 MPa, 100 C)."""

from __future__ import annotations

from petrology.fc.wl1990 import load_kinzler1997_morb_primary
from petrology.minerals import phase_properties
from petrology.norm_velocity import norm_velocity_from_bulk_wt

P_EVAL = 100e6
T_EVAL = 373.15


def main() -> None:
    print("=== single phases @ 100 MPa, 100 C ===")
    specs = [
        ("olivine", {"fo": 0.90}),
        ("plagioclase", {"an": 0.50}),
        ("clinopyroxene", {"cpx_mg": 0.85}),
        ("quartz", {}),
    ]
    for phase, kw in specs:
        for mb in ("sb1994", "fig2", "burnman"):
            try:
                p = phase_properties(phase, p_pa=P_EVAL, t_k=T_EVAL, backend=mb, **kw)
                print(
                    f"{phase:16s} {mb:8s}  Vp={p['vp_km_s']:.3f}  "
                    f"rho={p['rho_g_cm3']:.3f}  K={p['k_gpa']:.1f}  G={p['g_gpa']:.1f}"
                )
            except Exception as exc:
                print(f"{phase:16s} {mb:8s}  ERR: {exc}")

    melt = load_kinzler1997_morb_primary()["oxides_wt_percent"]
    print("\n=== Kinzler primary norm bulk (HS) ===")
    for mb in ("sb1994", "fig2"):
        r = norm_velocity_from_bulk_wt(
            melt, p_pa=P_EVAL, t_k=T_EVAL, mineral_backend=mb, cipw_backend="fallback",
        )
        print(f"{mb:8s}  Vp={r['vp_km_s']:.3f}  rho={r['rho_g_cm3']:.3f}  backend={r.get('mineral_backend')}")


if __name__ == "__main__":
    main()
