# -*- coding: utf-8 -*-
"""水波正演最小工区：生成文件 + 解析解。"""

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
        / "water_fwd"
        / "make_water_fwd_case.py"
    )
    spec = importlib.util.spec_from_file_location("make_water_fwd_case", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_water_fwd_case"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_case = _load_case()


def test_analytic_zero_offset() -> None:
    h_leg = _case.H - _case.SHOT_Z
    assert _case.t_direct(0.0) == pytest.approx(h_leg / _case.V_WATER)
    assert _case.t_mult(0.0) == pytest.approx((3.0 * _case.H - _case.SHOT_Z) / _case.V_WATER)
    assert (_case.t_mult(0.0) - _case.t_direct(0.0)) == pytest.approx(
        2.0 * _case.H / _case.V_WATER
    )


def test_write_water_fwd_case(tmp_path: Path) -> None:
    _case.write_smesh(tmp_path / "water.smesh")
    _case.write_seafloor(tmp_path / "seafloor.refl")
    _case.write_geom(tmp_path / "geom_water.dat")
    _case.write_analytic(tmp_path / "analytic.txt")
    sm = (tmp_path / "water.smesh").read_text(encoding="utf-8").splitlines()
    nx, nz, vw, _va = sm[0].split()
    assert int(nx) > 1 and int(nz) > 1
    assert float(vw) == pytest.approx(_case.V_WATER)
    assert all(abs(float(t)) < 1e-9 for t in sm[2].split())
    # 第 0 列：水柱叠加随机扰动；海底以下仍为浅沉积
    vcol = [float(v) for v in sm[4].split()]
    xs = [float(x) for x in sm[1].split()]
    zs = [float(z) for z in sm[3].split()]
    assert any(abs(z - _case.H) < 1e-9 for z in zs)
    x0 = xs[0]
    water_v = [v for z, v in zip(zs, vcol) if z <= _case.H + 1e-9]
    sed_v = [v for z, v in zip(zs, vcol) if z > _case.H + 1e-9]
    assert water_v and sed_v
    assert min(water_v) >= _case.V_WATER_MIN - 1e-6
    assert max(water_v) <= _case.V_WATER_MAX + 1e-6
    assert max(water_v) - min(water_v) > 0.01
    assert all(v == pytest.approx(_case.V_SEDIMENT, abs=1e-4) for v in sed_v)
    assert _case.v_at(x0, 0.0) == pytest.approx(_case.V_WATER, abs=1e-4)
    assert _case.v_at(x0, _case.H - 0.1) == pytest.approx(_case.V_WATER, abs=1e-4)
    assert _case.v_at(x0, _case.H + 0.2) == pytest.approx(_case.V_SEDIMENT)
    geom = (tmp_path / "geom_water.dat").read_text(encoding="utf-8").splitlines()
    assert geom[0] == "1"
    r = [ln for ln in geom if ln.startswith("r")]
    assert len(r) == len(_case.OFFSETS) * 2
    assert int(r[0][21:26]) == 2
    assert int(r[-1][21:26]) == 3
    seaf = (tmp_path / "seafloor.refl").read_text(encoding="utf-8").splitlines()
    assert not any(ln.startswith("#") for ln in seaf)
    assert float(seaf[0].split()[1]) == pytest.approx(_case.H)


def test_water_noise_reproducible_and_zero_amp(tmp_path: Path) -> None:
    a = tmp_path / "a.smesh"
    b = tmp_path / "b.smesh"
    c = tmp_path / "c.smesh"
    _case.write_smesh(a, seed=7)
    _case.write_smesh(b, seed=7)
    _case.write_smesh(c, noise_amp=0.0)
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")
    sm = c.read_text(encoding="utf-8").splitlines()
    xs = [float(x) for x in sm[1].split()]
    zs = [float(z) for z in sm[3].split()]
    vcol = [float(v) for v in sm[4].split()]
    x0 = xs[0]
    for z, v in zip(zs, vcol):
        assert v == pytest.approx(_case.v_at(x0, z), abs=1e-4)


def test_compare_syn_to_analytic_accepts_perfect() -> None:
    lines = ["1", "s 50.0 2.0 2"]
    dx = 0.0
    lines.append(f"r 50.0 0.01 2 {_case.t_direct(dx):.6f} 0.01")
    lines.append(f"r 50.0 0.01 3 {_case.t_mult(dx):.6f} 0.01")
    recs = _case.compare_syn_to_analytic("\n".join(lines) + "\n", atol=1e-6)
    assert len(recs) == 2
    assert recs[0][0] == 2 and recs[1][0] == 3


def test_parse_ray_file(tmp_path: Path) -> None:
    checker = (
        Path(__file__).resolve().parent.parent
        / "modeling"
        / "tomo2d"
        / "example_water"
        / "water_fwd"
        / "check_analytic.py"
    )
    spec = importlib.util.spec_from_file_location("check_analytic", checker)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_analytic"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    ray = tmp_path / "rays.dat"
    ray.write_text(
        "> ray 1\n50.0 2.0\n50.0 0.0\n> ray 2\n52.0 2.0\n52.0 0.0\n50.0 2.0\n",
        encoding="utf-8",
    )
    segs = mod.parse_ray_file(ray)
    assert len(segs) == 2
    assert segs[0] == ([50.0, 50.0], [2.0, 0.0])
    assert len(segs[1][0]) == 3


def test_plot_rays_smesh_background(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    checker = (
        Path(__file__).resolve().parent.parent
        / "modeling"
        / "tomo2d"
        / "example_water"
        / "water_fwd"
        / "check_analytic.py"
    )
    spec = importlib.util.spec_from_file_location("check_analytic_plot", checker)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_analytic_plot"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    case_dir = checker.parent
    rays = [([50.0, 50.0], [2.0, 0.0]), ([52.0, 52.0, 50.0], [2.0, 0.0, 2.0])]
    recs = [(2, 0.0, 1.3, 1.3, 0.0), (3, 2.0, 4.0, 4.0, 0.0)]
    out = tmp_path / "check_rays.png"
    mod.plot_rays(
        rays,
        recs,
        out,
        show=False,
        smesh_path=case_dir / "water.smesh",
        refl_path=case_dir / "seafloor.refl",
    )
    assert out.is_file() and out.stat().st_size > 2000
