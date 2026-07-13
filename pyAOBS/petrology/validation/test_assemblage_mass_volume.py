"""Test KKHS02-style mass→volume assemblage mixing."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from petrology.fc.assemblage import assemblage_vp_rho


def test_mass_basis_lowers_pl_rich_vp():
    """Pl-rich assemblage: mass→volume should yield lower Vp than volume-fraction shortcut."""
    vp_mass, _ = assemblage_vp_rho(
        ol_frac=0.12, pl_frac=0.68, cpx_frac=0.20,
        fo_pct=58, an_pct=54, di_pct=58,
        p_pa=100e6, t_k=373.15, mineral_backend="fig2", fraction_basis="mass",
    )
    vp_vol, _ = assemblage_vp_rho(
        ol_frac=0.12, pl_frac=0.68, cpx_frac=0.20,
        fo_pct=58, an_pct=54, di_pct=58,
        p_pa=100e6, t_k=373.15, mineral_backend="fig2", fraction_basis="volume",
    )
    assert vp_mass < vp_vol


if __name__ == "__main__":
    test_mass_basis_lowers_pl_rich_vp()
    print("ok")
