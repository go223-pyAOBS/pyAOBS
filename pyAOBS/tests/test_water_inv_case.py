# -*- coding: utf-8 -*-
"""水核反演最小工区：生成文件与统计。"""

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
        / "water_inv"
        / "make_water_inv_case.py"
    )
    spec = importlib.util.spec_from_file_location("make_water_inv_case", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_water_inv_case"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_case = _load_case()


def test_write_water_inv_case(tmp_path: Path) -> None:
    _case.write_smesh(tmp_path / "true.smesh", _case.V_TRUE)
    _case.write_smesh(tmp_path / "start.smesh", _case.V_START)
    _case.write_seafloor(tmp_path / "seafloor.refl")
    _case.write_geom(tmp_path / "geom_inv.dat")
    _case.write_vcorr(tmp_path / "vcorr.dat")
    xs, zs, vt = _case.parse_smesh(tmp_path / "true.smesh")
    _, _, vs = _case.parse_smesh(tmp_path / "start.smesh")
    x_lo, x_hi = _case.illum_x_range()
    tw, *_ = _case.node_stats(xs, zs, vt, x_lo=x_lo, x_hi=x_hi, water=True)
    sw, *_ = _case.node_stats(xs, zs, vs, x_lo=x_lo, x_hi=x_hi, water=True)
    ts, *_ = _case.node_stats(xs, zs, vt, x_lo=x_lo, x_hi=x_hi, water=False)
    ss, *_ = _case.node_stats(xs, zs, vs, x_lo=x_lo, x_hi=x_hi, water=False)
    assert tw == pytest.approx(_case.V_TRUE, abs=1e-4)
    assert sw == pytest.approx(_case.V_START, abs=1e-4)
    assert ts == pytest.approx(_case.V_SEDIMENT, abs=1e-4)
    assert ss == pytest.approx(_case.V_SEDIMENT, abs=1e-4)
    geom = (tmp_path / "geom_inv.dat").read_text(encoding="utf-8").splitlines()
    s = [ln for ln in geom if ln.startswith("s")]
    r = [ln for ln in geom if ln.startswith("r")]
    assert len(s) == len(_case.OBS_XS)
    assert len(r) == len(_case.SHOT_XS) * 2 * len(_case.OBS_XS)
    assert int(r[0][21:26]) == 2
    assert int(r[-1][21:26]) == 3
    assert _case.SHOT_DX == pytest.approx(0.2)
    assert _case.SHOT_XS == tuple(_case.OBS_X + dx for dx in _case.OFFSETS)
    assert _case.SHOT_XS[1] - _case.SHOT_XS[0] == pytest.approx(_case.SHOT_DX)
    seaf = (tmp_path / "seafloor.refl").read_text(encoding="utf-8").splitlines()
    assert float(seaf[0].split()[1]) == pytest.approx(_case.H)
    vc = (tmp_path / "vcorr.dat").read_text(encoding="utf-8").splitlines()
    assert vc[0].split() == ["2", "2"]
    assert float(vc[4].split()[0]) == pytest.approx(_case.LH)
    assert float(vc[6].split()[0]) == pytest.approx(_case.LV)


def test_analytic_zero_offset() -> None:
    assert (_case.t_mult(0.0) - _case.t_direct(0.0)) == pytest.approx(
        2.0 * _case.H / _case.V_TRUE
    )


def test_plot_inversion_models(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    checker = (
        Path(__file__).resolve().parent.parent
        / "modeling"
        / "tomo2d"
        / "example_water"
        / "water_inv"
        / "check_water_inv.py"
    )
    spec = importlib.util.spec_from_file_location("check_water_inv_plot", checker)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_water_inv_plot"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    true_p = tmp_path / "true.smesh"
    start_p = tmp_path / "start.smesh"
    rec_p = tmp_path / "rec.smesh"
    _case.write_smesh(true_p, _case.V_TRUE)
    _case.write_smesh(start_p, _case.V_START)
    _case.write_smesh(rec_p, 1.46)
    out = tmp_path / "check_inv_models.png"
    geom = tmp_path / "geom_inv.dat"
    _case.write_geom(geom)
    mod.plot_inversion(
        rec_p, true_p, start_p, out, show=False, geom_path=geom
    )
    assert out.is_file() and out.stat().st_size > 2000


def test_parse_picks_and_ttimes_plot(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    checker = (
        Path(__file__).resolve().parent.parent
        / "modeling"
        / "tomo2d"
        / "example_water"
        / "water_inv"
        / "check_water_inv.py"
    )
    spec = importlib.util.spec_from_file_location("check_water_inv_tt", checker)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_water_inv_tt"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    obs = [(2, 52.0, 0.01, 1.95, 50.0), (3, 52.0, 0.01, 4.36, 50.0)]
    start = [(2, 52.0, 0.01, 1.82, 50.0), (3, 52.0, 0.01, 4.07, 50.0)]
    rec = [(2, 52.0, 0.01, 1.94, 50.0), (3, 52.0, 0.01, 4.35, 50.0)]
    out = tmp_path / "check_inv_ttimes.png"
    mod.plot_ttimes_fit(obs, start, rec, out, show=False)
    assert out.is_file() and out.stat().st_size > 2000
    text = "1\ns 50.0 2.0 1\nr 52.0 0.01 2 1.95 0.01\n"
    picks = mod.parse_picks(text)
    assert picks == [(2, 52.0, 0.01, 1.95, 50.0)]


def test_plot_inv_rays(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    checker = (
        Path(__file__).resolve().parent.parent
        / "modeling"
        / "tomo2d"
        / "example_water"
        / "water_inv"
        / "check_water_inv.py"
    )
    spec = importlib.util.spec_from_file_location("check_water_inv_rays", checker)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_water_inv_rays"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    rec_p = tmp_path / "rec.smesh"
    geom = tmp_path / "geom_inv.dat"
    seaf = tmp_path / "seafloor.refl"
    _case.write_smesh(rec_p, 1.46)
    _case.write_geom(geom)
    _case.write_seafloor(seaf)
    rays = [([50.0, 52.0], [2.0, 0.01]), ([50.0, 52.0, 50.0, 52.0], [2.0, 0.0, 2.0, 0.01])]
    picks = [(2, 52.0, 0.01, 1.9, 50.0), (3, 52.0, 0.01, 4.3, 50.0)]
    out = tmp_path / "check_inv_rays.png"
    mod.plot_inv_rays(rec_p, seaf, rays, picks, geom, out, show=False)
    assert out.is_file() and out.stat().st_size > 2000
