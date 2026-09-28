# -*- coding: utf-8 -*-
"""PPP+PPS 同一次联合：geom_joint 为 0+7。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASE = ROOT / "modeling" / "tomo2d" / "example_water" / "PPP+PPS_inv"


def _load():
    spec = importlib.util.spec_from_file_location(
        "make_ppp_pps_inv_case", CASE / "make_ppp_pps_inv_case.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_ppp_pps_inv_case"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_write_ppp_pps_inv_case(tmp_path: Path) -> None:
    m = _load()
    m.write_vp(tmp_path / "true_vp.smesh", **m.VP_TRUE)
    m.write_vp(tmp_path / "start_vp.smesh", **m.VP_START)
    m.write_vs(tmp_path / "true_vs.smesh", m.KAPPA_TRUE, **m.VP_TRUE)
    m.write_geom(tmp_path / "geom_ppp.dat", codes=m.CODES_PPP)
    m.write_geom(tmp_path / "geom_inv.dat", codes=m.CODES)
    m.write_geom(tmp_path / "geom_joint.dat", codes=m.CODES_JOINT)
    assert m.CODES == (7,)
    assert m.CODES_PPP == (0,)
    assert m.CODES_JOINT == (0, 7)
    geom = (tmp_path / "geom_inv.dat").read_text(encoding="utf-8")
    kinds = [ln.split()[3] for ln in geom.splitlines() if ln.startswith("r")]
    assert set(kinds) == {"7"}
    ppp = (tmp_path / "geom_ppp.dat").read_text(encoding="utf-8")
    ppp_kinds = [ln.split()[3] for ln in ppp.splitlines() if ln.startswith("r")]
    assert set(ppp_kinds) == {"0"}
    joint = (tmp_path / "geom_joint.dat").read_text(encoding="utf-8")
    joint_kinds = [ln.split()[3] for ln in joint.splitlines() if ln.startswith("r")]
    assert set(joint_kinds) == {"0", "7"}
    assert joint_kinds.count("0") == joint_kinds.count("7")
