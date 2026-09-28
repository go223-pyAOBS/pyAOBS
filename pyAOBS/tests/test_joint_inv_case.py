# -*- coding: utf-8 -*-
"""水+壳联合反演工区：生成文件与出图。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load(name: str, rel: tuple[str, ...]):
    path = Path(__file__).resolve().parent.parent.joinpath(*rel)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_case = _load(
    "make_joint_inv_case",
    ("modeling", "tomo2d", "example_water", "joint_inv", "make_joint_inv_case.py"),
)


def test_write_joint_inv_case(tmp_path: Path) -> None:
    _case.write_smesh(tmp_path / "true.smesh", _case.V_WATER_TRUE, _case.V_SED_TRUE)
    _case.write_smesh(tmp_path / "start.smesh", _case.V_WATER_START, _case.V_SED_START)
    _case.write_seafloor(tmp_path / "seafloor.refl")
    _case.write_basement(tmp_path / "basement.refl")
    _case.write_geom(tmp_path / "geom_inv.dat")
    _case.write_vcorr(tmp_path / "vcorr.dat")
    xs, zs, vt = _case.parse_smesh(tmp_path / "true.smesh")
    _, _, vs = _case.parse_smesh(tmp_path / "start.smesh")
    x_lo, x_hi = _case.illum_x_range()
    tw, *_ = _case.node_stats(xs, zs, vt, x_lo=x_lo, x_hi=x_hi, water=True, strict=True)
    sw, *_ = _case.node_stats(xs, zs, vs, x_lo=x_lo, x_hi=x_hi, water=True, strict=True)
    ts, *_ = _case.node_stats(xs, zs, vt, x_lo=x_lo, x_hi=x_hi, water=False)
    ss, *_ = _case.node_stats(xs, zs, vs, x_lo=x_lo, x_hi=x_hi, water=False)
    assert tw == pytest.approx(_case.V_WATER_TRUE, abs=1e-4)
    assert sw == pytest.approx(_case.V_WATER_START, abs=1e-4)
    assert ts == pytest.approx(_case.V_SED_TRUE, abs=1e-4)
    assert ss == pytest.approx(_case.V_SED_START, abs=1e-4)
    geom = (tmp_path / "geom_inv.dat").read_text(encoding="utf-8").splitlines()
    r = [ln for ln in geom if ln.startswith("r")]
    assert len(r) == len(_case.OFFSETS) * 4
    codes = [int(ln[21:26]) for ln in r]
    assert codes.count(0) == len(_case.OFFSETS)
    assert codes.count(1) == len(_case.OFFSETS)
    assert codes.count(2) == len(_case.OFFSETS)
    assert codes.count(3) == len(_case.OFFSETS)
    seaf = (tmp_path / "seafloor.refl").read_text(encoding="utf-8").splitlines()
    base = (tmp_path / "basement.refl").read_text(encoding="utf-8").splitlines()
    assert float(seaf[0].split()[1]) == pytest.approx(_case.H)
    assert float(base[0].split()[1]) == pytest.approx(_case.H_REFL)


def test_plot_joint_inversion_models(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    mod = _load(
        "check_joint_inv_plot",
        ("modeling", "tomo2d", "example_water", "joint_inv", "check_joint_inv.py"),
    )
    true_p = tmp_path / "true.smesh"
    start_p = tmp_path / "start.smesh"
    rec_p = tmp_path / "rec.smesh"
    _case.write_smesh(true_p, _case.V_WATER_TRUE, _case.V_SED_TRUE)
    _case.write_smesh(start_p, _case.V_WATER_START, _case.V_SED_START)
    _case.write_smesh(rec_p, 1.47, 1.82)
    geom = tmp_path / "geom_inv.dat"
    _case.write_geom(geom)
    out = tmp_path / "check_inv_models.png"
    mod.plot_inversion(rec_p, true_p, start_p, out, show=False, geom_path=geom)
    assert out.is_file() and out.stat().st_size > 2000


def test_plot_joint_ttimes(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    mod = _load(
        "check_joint_inv_tt",
        ("modeling", "tomo2d", "example_water", "joint_inv", "check_joint_inv.py"),
    )
    obs = [(0, 52.0, 2.0, 1.11, 50.0), (2, 52.0, 0.01, 1.95, 50.0)]
    start = [(0, 52.0, 2.0, 1.00, 50.0), (2, 52.0, 0.01, 1.82, 50.0)]
    rec = [(0, 52.0, 2.0, 1.10, 50.0), (2, 52.0, 0.01, 1.94, 50.0)]
    out = tmp_path / "check_inv_ttimes.png"
    mod.plot_ttimes_fit(obs, start, rec, out, show=False)
    assert out.is_file() and out.stat().st_size > 2000
