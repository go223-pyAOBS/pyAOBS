# -*- coding: utf-8 -*-
"""台侧一阶 peg-leg 工区生成。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_case():
    path = (
        Path(__file__).resolve().parent.parent
        / "modeling"
        / "tomo2d"
        / "example_water"
        / "recv_peg_fwd"
        / "make_recv_peg_case.py"
    )
    spec = importlib.util.spec_from_file_location("make_recv_peg_case", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_recv_peg_case"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_case = _load_case()


def test_recv_peg_twt() -> None:
    assert _case.t_water_twt() == pytest.approx(2.0 * _case.H / _case.V_WATER)


def test_write_recv_peg_case(tmp_path: Path) -> None:
    _case.write_smesh(tmp_path / "crust.smesh")
    _case.write_seafloor(tmp_path / "seafloor.refl")
    _case.write_moho(tmp_path / "moho.refl")
    _case.write_geom(tmp_path / "geom_peg.dat")
    sm = (tmp_path / "crust.smesh").read_text(encoding="utf-8").splitlines()
    zs = [float(z) for z in sm[3].split()]
    vcol = [float(v) for v in sm[4].split()]
    water_v = [v for z, v in zip(zs, vcol) if z <= _case.H + 1e-9]
    crust_v = [v for z, v in zip(zs, vcol) if _case.H + 1e-9 < z <= _case.H_MOHO + 1e-9]
    mantle_v = [v for z, v in zip(zs, vcol) if z > _case.H_MOHO + 1e-9]
    assert water_v and all(v == pytest.approx(_case.V_WATER, abs=1e-4) for v in water_v)
    assert crust_v and min(crust_v) >= _case.V_CRUST0 - 1e-4
    assert max(crust_v) < _case.V_MANTLE - 0.5
    assert mantle_v and all(v == pytest.approx(_case.V_MANTLE, abs=1e-4) for v in mantle_v)
    moho = (tmp_path / "moho.refl").read_text(encoding="utf-8").splitlines()
    assert all(abs(float(ln.split()[1]) - _case.H_MOHO) < 1e-6 for ln in moho if ln.strip())
    geom = (tmp_path / "geom_peg.dat").read_text(encoding="utf-8").splitlines()
    r = [ln for ln in geom if ln.startswith("r")]
    codes = [int(ln[21:26]) for ln in r]
    assert codes.count(0) == len(_case.OFFSETS)
    assert codes.count(4) == len(_case.OFFSETS)
    assert codes.count(1) == len(_case.OFFSETS)
    assert codes.count(5) == len(_case.OFFSETS)


def test_analytic_pmp_vertical() -> None:
    ta = _case.t_analytic(1, 0.05)
    assert ta is not None
    assert ta == pytest.approx(_case.t_pmp_vertical(), abs=0.02)


def test_analytic_peg_later_than_primary() -> None:
    t0 = _case.t_analytic(0, 15.0)
    t4 = _case.t_analytic(4, 15.0)
    t1 = _case.t_analytic(1, 15.0)
    t5 = _case.t_analytic(5, 15.0)
    assert t0 is not None and t4 is not None
    assert t1 is not None and t5 is not None
    assert t4 > t0
    assert t5 > t1
    twt = _case.t_water_twt()
    assert t4 - t0 == pytest.approx(twt, abs=0.25)
    assert t5 - t1 == pytest.approx(twt, abs=0.25)
    assert _case.t_analytic(0, 30.0) is not None
    assert _case.t_analytic(4, 30.0) is not None
