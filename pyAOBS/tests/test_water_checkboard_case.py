# -*- coding: utf-8 -*-
"""water_checkboard：水柱 10×1 km 棋盘格 + 5 台。"""

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
        / "water_checkboard"
        / "make_water_checkboard_case.py"
    )
    spec = importlib.util.spec_from_file_location("make_water_checkboard_case", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_water_checkboard_case"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_case = _load_case()


def test_write_water_checkboard_case(tmp_path: Path) -> None:
    _case.write_true(tmp_path / "true.smesh")
    _case.write_start(tmp_path / "start.smesh")
    _case.write_seafloor(tmp_path / "seafloor.refl")
    _case.write_geom(tmp_path / "geom_inv.dat")
    _case.write_geom(tmp_path / "geom_c2.dat", codes=(2,))
    _case.write_vcorr(tmp_path / "vcorr.dat")
    xs, zs, vt = _case.parse_smesh(tmp_path / "true.smesh")
    _, _, vs = _case.parse_smesh(tmp_path / "start.smesh")
    x_lo, x_hi = _case.illum_x_range()
    tw, tlo, thi, _ = _case.node_stats(xs, zs, vt, x_lo=x_lo, x_hi=x_hi, water=True)
    sw, slo, shi, _ = _case.node_stats(xs, zs, vs, x_lo=x_lo, x_hi=x_hi, water=True)
    ts, *_ = _case.node_stats(xs, zs, vt, x_lo=x_lo, x_hi=x_hi, water=False)
    ss, *_ = _case.node_stats(xs, zs, vs, x_lo=x_lo, x_hi=x_hi, water=False)
    assert tlo == pytest.approx(_case.V_WATER - _case.DV)
    assert thi == pytest.approx(_case.V_WATER + _case.DV)
    assert tw == pytest.approx(_case.V_WATER, abs=0.02)
    assert sw == pytest.approx(_case.V_START, abs=1e-4)
    assert slo == pytest.approx(shi)
    assert ts == pytest.approx(_case.V_SEDIMENT, abs=1e-4)
    assert ss == pytest.approx(_case.V_SEDIMENT, abs=1e-4)
    assert _case.v_at(35.0, 0.5, checker=True) == pytest.approx(_case.V_WATER + _case.DV)
    assert _case.v_at(35.0, 1.5, checker=True) == pytest.approx(_case.V_WATER - _case.DV)
    assert _case.v_at(45.0, 0.5, checker=True) == pytest.approx(_case.V_WATER - _case.DV)
    assert _case.checker_sign(35.0, 0.5) == -_case.checker_sign(35.0, 1.5)
    pol, ratio, corr, n = _case.checker_recovery(
        xs, zs, vs, vt, x_lo=x_lo, x_hi=x_hi
    )
    assert n > 100
    assert pol == pytest.approx(0.0)
    assert ratio == pytest.approx(0.0)
    assert abs(corr) < 0.3
    geom = (tmp_path / "geom_inv.dat").read_text(encoding="utf-8").splitlines()
    s = [ln for ln in geom if ln.startswith("s")]
    r = [ln for ln in geom if ln.startswith("r")]
    assert len(s) == 5
    assert len(r) == len(_case.SHOT_XS) * 2 * len(_case.OBS_XS)
    r2 = [
        ln
        for ln in (tmp_path / "geom_c2.dat").read_text(encoding="utf-8").splitlines()
        if ln.startswith("r")
    ]
    assert len(r2) == len(_case.SHOT_XS) * len(_case.OBS_XS)
    assert _case.CX == pytest.approx(10.0)
    assert _case.CZ == pytest.approx(1.0)
    assert _case.LH < _case.CX
    assert _case.LV < _case.CZ
    vc = (tmp_path / "vcorr.dat").read_text(encoding="utf-8").splitlines()
    assert float(vc[4].split()[0]) == pytest.approx(_case.LH)
    assert float(vc[6].split()[0]) == pytest.approx(_case.LV)


def test_plot_inversion_models(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    checker = (
        Path(__file__).resolve().parent.parent
        / "modeling"
        / "tomo2d"
        / "example_water"
        / "water_checkboard"
        / "check_water_checkboard.py"
    )
    spec = importlib.util.spec_from_file_location("check_water_checkboard_plot", checker)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_water_checkboard_plot"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    true_p = tmp_path / "true.smesh"
    start_p = tmp_path / "start.smesh"
    rec_p = tmp_path / "rec.smesh"
    _case.write_true(true_p)
    _case.write_start(start_p)
    _case.write_start(rec_p)
    out = tmp_path / "check_inv_models.png"
    geom = tmp_path / "geom_inv.dat"
    _case.write_geom(geom)
    mod.plot_inversion(rec_p, true_p, start_p, out, show=False, geom_path=geom)
    assert out.is_file() and out.stat().st_size > 2000
    cmp = tmp_path / "check_inv_compare.png"
    mod.plot_compare(rec_p, rec_p, true_p, start_p, cmp, show=False, geom_path=geom)
    assert cmp.is_file() and cmp.stat().st_size > 2000
