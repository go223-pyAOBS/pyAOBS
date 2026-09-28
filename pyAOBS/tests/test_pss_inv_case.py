# -*- coding: utf-8 -*-
"""PSS 反 Vs 工区生成：geom 只有 raytype 8。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PSS = ROOT / "modeling" / "tomo2d" / "example_water" / "pss_inv"


def _load():
    spec = importlib.util.spec_from_file_location("make_pss_inv_case", PSS / "make_pss_inv_case.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_pss_inv_case"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_write_pss_inv_case(tmp_path: Path) -> None:
    m = _load()
    m.write_vp(tmp_path / "true_vp.smesh", **m.VP_TRUE)
    m.write_vp(tmp_path / "start_vp.smesh", **m.VP_START)
    m.write_vs(tmp_path / "true_vs.smesh", m.KAPPA_TRUE, **m.VP_TRUE)
    m.write_vs(tmp_path / "start_vs.smesh", m.KAPPA_START, **m.VP_TRUE)
    m.write_seafloor(tmp_path / "seafloor.refl")
    m.write_conv(tmp_path / "conv.refl")
    m.write_geom(tmp_path / "geom_ppp.dat", codes=m.CODES_PPP)
    m.write_geom(tmp_path / "geom_inv.dat", codes=m.CODES)
    m.write_vcorr(tmp_path / "vcorr.dat")
    assert m.CODES == (8,)
    geom = (tmp_path / "geom_inv.dat").read_text(encoding="utf-8")
    kinds = [ln.split()[3] for ln in geom.splitlines() if ln.startswith("r")]
    assert set(kinds) == {"8"}
    assert kinds
    ppp = (tmp_path / "geom_ppp.dat").read_text(encoding="utf-8")
    ppp_kinds = [ln.split()[3] for ln in ppp.splitlines() if ln.startswith("r")]
    assert set(ppp_kinds) == {"0"}
    xs, zs, vtrue = m.parse_smesh(tmp_path / "true_vs.smesh")
    _, _, vstart = m.parse_smesh(tmp_path / "start_vs.smesh")
    _, _, vp_t = m.parse_smesh(tmp_path / "true_vp.smesh")
    k = zs.index(min(z for z in zs if z > m.Z_CONV + 1e-9))
    assert abs(vtrue[0][0] - m.V_WATER) < 1e-4
    assert abs(vtrue[0][k] - vp_t[0][k] / m.KAPPA_TRUE) < 1e-4
    assert abs(vstart[0][k] - vp_t[0][k] / m.KAPPA_START) < 1e-4
