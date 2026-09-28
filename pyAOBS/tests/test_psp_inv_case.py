# -*- coding: utf-8 -*-
"""PSP 冻真 Vp 反 Vs：geom 只有 raytype 6。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PSP = ROOT / "modeling" / "tomo2d" / "example_water" / "psp_inv"


def _load():
    spec = importlib.util.spec_from_file_location("make_psp_inv_case", PSP / "make_psp_inv_case.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_psp_inv_case"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_write_psp_inv_case(tmp_path: Path) -> None:
    m = _load()
    m.write_vp(tmp_path / "true_vp.smesh", **m.VP_TRUE)
    m.write_vs(tmp_path / "true_vs.smesh", m.KAPPA_TRUE, **m.VP_TRUE)
    m.write_vs(tmp_path / "start_vs.smesh", m.KAPPA_START, **m.VP_TRUE)
    m.write_geom(tmp_path / "geom_inv.dat", codes=m.CODES)
    assert m.CODES == (6,)
    geom = (tmp_path / "geom_inv.dat").read_text(encoding="utf-8")
    kinds = [ln.split()[3] for ln in geom.splitlines() if ln.startswith("r")]
    assert set(kinds) == {"6"}
    xs, zs, vtrue = m.parse_smesh(tmp_path / "true_vs.smesh")
    _, _, vstart = m.parse_smesh(tmp_path / "start_vs.smesh")
    _, _, vp_t = m.parse_smesh(tmp_path / "true_vp.smesh")
    k = zs.index(min(z for z in zs if z > m.Z_CONV + 1e-9))
    j = zs.index(min(z for z in zs if z > m.H + 1e-9))
    assert abs(vtrue[0][k] - vp_t[0][k] / m.KAPPA_TRUE) < 1e-4
    assert abs(vstart[0][k] - vp_t[0][k] / m.KAPPA_START) < 1e-4
    assert abs(vstart[0][j] - vp_t[0][j] / m.KAPPA_START) < 1e-4
