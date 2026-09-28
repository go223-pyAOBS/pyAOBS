# -*- coding: utf-8 -*-
"""壳核反演最小工区：生成文件与统计。"""

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
        / "crust_inv"
        / "make_crust_inv_case.py"
    )
    spec = importlib.util.spec_from_file_location("make_crust_inv_case", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_crust_inv_case"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_case = _load_case()


def test_write_crust_inv_case(tmp_path: Path) -> None:
    _case.write_smesh(tmp_path / "true.smesh", _case.V_WATER, _case.V_SED_TRUE)
    _case.write_smesh(tmp_path / "start.smesh", _case.V_WATER, _case.V_SED_START)
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
    assert tw == pytest.approx(_case.V_WATER, abs=1e-4)
    assert sw == pytest.approx(_case.V_WATER, abs=1e-4)
    assert ts == pytest.approx(_case.V_SED_TRUE, abs=1e-4)
    assert ss == pytest.approx(_case.V_SED_START, abs=1e-4)
    geom = (tmp_path / "geom_inv.dat").read_text(encoding="utf-8").splitlines()
    r = [ln for ln in geom if ln.startswith("r")]
    assert len(r) == len(_case.OFFSETS)
    assert int(r[0][21:26]) == 0
    seaf = (tmp_path / "seafloor.refl").read_text(encoding="utf-8").splitlines()
    assert float(seaf[0].split()[1]) == pytest.approx(_case.H)
    tws, *_ = _case.node_stats(xs, zs, vt, x_lo=x_lo, x_hi=x_hi, water=True, strict=True)
    assert tws == pytest.approx(_case.V_WATER, abs=1e-4)


def _load_checker(name: str):
    path = (
        Path(__file__).resolve().parent.parent
        / "modeling"
        / "tomo2d"
        / "example_water"
        / "crust_inv"
        / "check_crust_inv.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_plot_inversion_models(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    mod = _load_checker("check_crust_inv_plot")
    true_p = tmp_path / "true.smesh"
    start_p = tmp_path / "start.smesh"
    rec_p = tmp_path / "rec.smesh"
    _case.write_smesh(true_p, _case.V_WATER, _case.V_SED_TRUE)
    _case.write_smesh(start_p, _case.V_WATER, _case.V_SED_START)
    _case.write_smesh(rec_p, _case.V_WATER, 1.81)
    out = tmp_path / "check_inv_models.png"
    geom = tmp_path / "geom_inv.dat"
    _case.write_geom(geom)
    mod.plot_inversion(rec_p, true_p, start_p, out, show=False, geom_path=geom)
    assert out.is_file() and out.stat().st_size > 2000


def test_parse_picks_and_ttimes_plot(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    mod = _load_checker("check_crust_inv_tt")
    obs = [(0, 52.0, 2.0, 1.11, 50.0)]
    start = [(0, 52.0, 2.0, 1.00, 50.0)]
    rec = [(0, 52.0, 2.0, 1.10, 50.0)]
    out = tmp_path / "check_inv_ttimes.png"
    mod.plot_ttimes_fit(obs, start, rec, out, show=False)
    assert out.is_file() and out.stat().st_size > 2000
    text = "1\ns 50.0 2.0 1\nr 52.0 2.0 0 1.11 0.01\n"
    picks = mod.parse_picks(text)
    assert picks == [(0, 52.0, 2.0, 1.11, 50.0)]


def test_plot_inv_rays(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    mod = _load_checker("check_crust_inv_rays")
    rec_p = tmp_path / "rec.smesh"
    geom = tmp_path / "geom_inv.dat"
    seaf = tmp_path / "seafloor.refl"
    _case.write_smesh(rec_p, _case.V_WATER, 1.81)
    _case.write_geom(geom)
    _case.write_seafloor(seaf)
    rays = [([50.0, 52.0], [2.0, 2.0])]
    picks = [(0, 52.0, 2.0, 1.1, 50.0)]
    out = tmp_path / "check_inv_rays.png"
    mod.plot_inv_rays(rec_p, seaf, rays, picks, geom, out, show=False)
    assert out.is_file() and out.stat().st_size > 2000
