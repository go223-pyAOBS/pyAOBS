# -*- coding: utf-8 -*-
"""water_inv2：water_fwd 真模型 + 4 台、200 m 炮。"""

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
        / "water_inv2"
        / "make_water_inv2_case.py"
    )
    spec = importlib.util.spec_from_file_location("make_water_inv2_case", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_water_inv2_case"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_case = _load_case()


def test_write_water_inv2_case(tmp_path: Path) -> None:
    _case.write_true(tmp_path / "true.smesh")
    _case.write_start(tmp_path / "start.smesh")
    _case.write_seafloor(tmp_path / "seafloor.refl")
    _case.write_geom(tmp_path / "geom_inv.dat")
    _case.write_vcorr(tmp_path / "vcorr.dat")
    xs, zs, vt = _case.parse_smesh(tmp_path / "true.smesh")
    _, _, vs = _case.parse_smesh(tmp_path / "start.smesh")
    x_lo, x_hi = _case.illum_x_range()
    tw, tlo, thi, _ = _case.node_stats(xs, zs, vt, x_lo=x_lo, x_hi=x_hi, water=True)
    sw, slo, shi, _ = _case.node_stats(xs, zs, vs, x_lo=x_lo, x_hi=x_hi, water=True)
    ts, *_ = _case.node_stats(xs, zs, vt, x_lo=x_lo, x_hi=x_hi, water=False)
    ss, *_ = _case.node_stats(xs, zs, vs, x_lo=x_lo, x_hi=x_hi, water=False)
    assert tw == pytest.approx(_case.V_TRUE_BG, abs=0.03)
    assert sw == pytest.approx(_case.V_START, abs=1e-4)
    assert slo == pytest.approx(shi)
    assert thi - tlo > 0.05
    assert ts == pytest.approx(_case.V_SEDIMENT, abs=1e-4)
    assert ss == pytest.approx(_case.V_SEDIMENT, abs=1e-4)
    rms, corr, n = _case.water_field_compare(
        xs, zs, vs, vt, x_lo=x_lo, x_hi=x_hi
    )
    assert n > 100
    assert rms > 0.03
    assert abs(corr) < 0.3
    geom = (tmp_path / "geom_inv.dat").read_text(encoding="utf-8").splitlines()
    s = [ln for ln in geom if ln.startswith("s")]
    r = [ln for ln in geom if ln.startswith("r")]
    assert len(s) == len(_case.OBS_XS)
    assert len(r) == len(_case.SHOT_XS) * 2 * len(_case.OBS_XS)
    assert _case.SHOT_DX == pytest.approx(0.2)
    vc = (tmp_path / "vcorr.dat").read_text(encoding="utf-8").splitlines()
    assert float(vc[4].split()[0]) == pytest.approx(_case.LH)
    assert float(vc[6].split()[0]) == pytest.approx(_case.LV)
    assert len(_case.OBS_XS) == 4
    assert _case.OBS_XS[0] == pytest.approx(30.0)
    assert _case.OBS_XS[-1] == pytest.approx(70.0)
    assert _case.LH == pytest.approx(8.0)
    assert _case.LV == pytest.approx(1.0)
    assert _case.WSV == pytest.approx(200.0)
    _case.write_geom(tmp_path / "geom_c2.dat", codes=(2,))
    r2 = [
        ln
        for ln in (tmp_path / "geom_c2.dat").read_text(encoding="utf-8").splitlines()
        if ln.startswith("r")
    ]
    assert len(r2) == len(_case.SHOT_XS) * len(_case.OBS_XS)
    assert int(r2[0][21:26]) == 2
    assert int(r2[-1][21:26]) == 2


def test_true_matches_water_fwd() -> None:
    fwd = (
        Path(__file__).resolve().parent.parent
        / "modeling"
        / "tomo2d"
        / "example_water"
        / "water_fwd"
        / "water.smesh"
    )
    if not fwd.is_file():
        pytest.skip("water_fwd/water.smesh missing")
    text_fwd = fwd.read_text(encoding="utf-8")
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "true.smesh"
        _case.write_true(p)
        assert p.read_text(encoding="utf-8") == text_fwd


def test_plot_inversion_models(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    checker = (
        Path(__file__).resolve().parent.parent
        / "modeling"
        / "tomo2d"
        / "example_water"
        / "water_inv2"
        / "check_water_inv2.py"
    )
    spec = importlib.util.spec_from_file_location("check_water_inv2_plot", checker)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_water_inv2_plot"] = mod
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
