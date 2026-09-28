# -*- coding: utf-8 -*-
"""PSP 面下 Vs 反演工区生成。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PS = ROOT / "modeling" / "tomo2d" / "example_water" / "ps_inv"


def _load():
    spec = importlib.util.spec_from_file_location("make_ps_inv_case", PS / "make_ps_inv_case.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_ps_inv_case"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_write_ps_inv_case(tmp_path: Path) -> None:
    m = _load()
    m.write_vp(tmp_path / "true_vp.smesh", **m.VP_TRUE)
    m.write_vp(tmp_path / "start_vp.smesh", **m.VP_START)
    m.write_vs(tmp_path / "true_vs.smesh", m.KAPPA_TRUE, **m.VP_TRUE)
    m.write_vs(tmp_path / "start_vs.smesh", m.KAPPA_START, **m.VP_START)
    m.write_seafloor(tmp_path / "seafloor.refl")
    m.write_conv(tmp_path / "conv.refl")
    m.write_geom(tmp_path / "geom_ppp.dat", codes=m.CODES_PPP)
    m.write_geom(tmp_path / "geom_inv.dat", codes=m.CODES)
    m.write_vcorr(tmp_path / "vcorr.dat")
    m.write_vs_from_smesh(tmp_path / "true_vp.smesh", tmp_path / "from_vp.smesh", m.KAPPA_TRUE)
    xs, zs, vtrue = m.parse_smesh(tmp_path / "true_vs.smesh")
    _, _, vstart = m.parse_smesh(tmp_path / "start_vs.smesh")
    _, _, vp_t = m.parse_smesh(tmp_path / "true_vp.smesh")
    _, _, vp_s = m.parse_smesh(tmp_path / "start_vp.smesh")
    k = zs.index(min(z for z in zs if z > m.Z_CONV + 1e-9))
    j = zs.index(min(z for z in zs if z > m.H + 1e-9))
    assert abs(vtrue[0][0] - m.V_WATER) < 1e-4
    assert abs(vstart[0][0] - m.V_WATER) < 1e-4
    assert abs(vtrue[0][k] - vp_t[0][k] / m.KAPPA_TRUE) < 1e-4
    assert abs(vstart[0][k] - vp_s[0][k] / m.KAPPA_START) < 1e-4
    assert vp_s[0][j] != vp_t[0][j]
    assert vp_t[0][k] > vp_t[0][j]
    assert vp_s[0][k] > vp_s[0][j]
    lid_t = [v for z, v in zip(zs, vp_t[0]) if m.H + 1e-9 < z < m.Z_CONV - 1e-9]
    lid_s = [v for z, v in zip(zs, vp_s[0]) if m.H + 1e-9 < z < m.Z_CONV - 1e-9]
    crust_t = [v for z, v in zip(zs, vp_t[0]) if z > m.Z_CONV + 1e-9]
    crust_s = [v for z, v in zip(zs, vp_s[0]) if z > m.Z_CONV + 1e-9]
    assert max(lid_t) > min(lid_t) and max(lid_s) > min(lid_s)
    assert max(crust_t) > min(crust_t) and max(crust_s) > min(crust_s)
    geom = (tmp_path / "geom_inv.dat").read_text(encoding="utf-8")
    kinds = [ln.split()[3] for ln in geom.splitlines() if ln.startswith("r")]
    assert set(kinds) == {"6"}
    ppp = (tmp_path / "geom_ppp.dat").read_text(encoding="utf-8")
    ppp_kinds = [ln.split()[3] for ln in ppp.splitlines() if ln.startswith("r")]
    assert set(ppp_kinds) == {"0"}
    _, _, from_vp = m.parse_smesh(tmp_path / "from_vp.smesh")
    assert abs(from_vp[0][k] - vp_t[0][k] / m.KAPPA_TRUE) < 1e-4
