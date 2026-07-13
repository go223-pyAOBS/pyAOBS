"""Smoke test: Sallarès (2005) Galápagos H–Vp reproduction."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.melting.sallares2005 import solve_sallares_melting
from petrology.reference.galapagos_sallares_observations import GALAPAGOS_SALLARES_OBSERVATIONS


def main() -> int:
    r_noc = solve_sallares_melting(tp_c=1300, upwelling_x=1.0)
    assert 6.0 <= r_noc.h_km <= 8.5, r_noc.h_km
    assert 7.0 <= r_noc.vp_bulk_km_s <= 7.45, r_noc.vp_bulk_km_s

    r_thick = solve_sallares_melting(tp_c=1300, upwelling_x=8.0)
    assert r_thick.h_km >= 17.0, r_thick.h_km
    assert r_thick.vp_bulk_km_s > r_noc.vp_bulk_km_s, (
        r_thick.vp_bulk_km_s,
        r_noc.vp_bulk_km_s,
    )

    obs_thick = [p for p in GALAPAGOS_SALLARES_OBSERVATIONS if p.thick_crest]
    assert obs_thick, "missing thick-crust observations"
    for p in obs_thick:
        assert p.v_lc_km_s < r_thick.vp_bulk_km_s - 0.15, (
            p.profile_id,
            p.v_lc_km_s,
            r_thick.vp_bulk_km_s,
        )

    print("PASS Sallarès Galápagos: NOC H≈7 km, thick-crust model V > observed V_LC")
    print(f"  NOC: H={r_noc.h_km:.2f} km V={r_noc.vp_bulk_km_s:.3f} km/s")
    print(f"  Tp1300 X=8: H={r_thick.h_km:.1f} km V={r_thick.vp_bulk_km_s:.3f} km/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
