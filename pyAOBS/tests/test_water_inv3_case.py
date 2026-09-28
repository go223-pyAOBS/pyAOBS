# -*- coding: utf-8 -*-
"""water_inv3：0–200 km、水深 3 km、30×1 km 高速异常；10 台、分震相窗口。"""

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
        / "water_inv3"
        / "make_water_inv3_case.py"
    )
    spec = importlib.util.spec_from_file_location("make_water_inv3_case", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_water_inv3_case"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_case = _load_case()


def _r_rows(path: Path) -> list[tuple[float, float, int]]:
    rows: list[tuple[float, float, int]] = []
    src_x = None
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        if ln.startswith("s"):
            src_x = float(ln[1:11])
        elif ln.startswith("r"):
            x = float(ln[1:11])
            code = int(ln[21:26])
            assert src_x is not None
            rows.append((src_x, x, code))
    return rows


def test_write_water_inv3_case(tmp_path: Path) -> None:
    _case.write_true(tmp_path / "true.smesh")
    _case.write_start(tmp_path / "start.smesh")
    _case.write_seafloor(tmp_path / "seafloor.refl")
    _case.write_geom(tmp_path / "geom_inv.dat")
    _case.write_geom(tmp_path / "geom_c2.dat", codes=(2,))
    _case.write_vcorr(tmp_path / "vcorr.dat")
    xs, zs, vt = _case.parse_smesh(tmp_path / "true.smesh")
    _, _, vs = _case.parse_smesh(tmp_path / "start.smesh")
    assert xs[0] == pytest.approx(0.0)
    assert xs[-1] == pytest.approx(200.0)
    assert zs[-1] == pytest.approx(_case.ZMAX)
    x_lo, x_hi = _case.illum_x_range()
    tw, tlo, thi, _ = _case.node_stats(xs, zs, vt, x_lo=x_lo, x_hi=x_hi, water=True)
    sw, slo, shi, _ = _case.node_stats(xs, zs, vs, x_lo=x_lo, x_hi=x_hi, water=True)
    ts, *_ = _case.node_stats(xs, zs, vt, x_lo=x_lo, x_hi=x_hi, water=False)
    ss, *_ = _case.node_stats(xs, zs, vs, x_lo=x_lo, x_hi=x_hi, water=False)
    assert tlo == pytest.approx(_case.V_WATER)
    assert thi == pytest.approx(_case.V_ANOM)
    assert tw > _case.V_WATER
    assert sw == pytest.approx(_case.V_START, abs=1e-4)
    assert slo == pytest.approx(shi)
    assert ts == pytest.approx(_case.V_SEDIMENT, abs=1e-4)
    assert ss == pytest.approx(_case.V_SEDIMENT, abs=1e-4)
    a_m, a_lo, a_hi, n_a = _case.box_stats(
        xs, zs, vt, x_lo=_case.AX0, x_hi=_case.AX1, z_lo=_case.AZ0, z_hi=_case.AZ1
    )
    assert n_a > 20
    assert a_m == pytest.approx(_case.V_ANOM, abs=1e-4)
    assert a_lo == pytest.approx(a_hi)
    outside = [
        vt[i][k]
        for i, x in enumerate(xs)
        for k, z in enumerate(zs)
        if z <= _case.H + 1e-9 and not _case.in_anomaly(x, z)
    ]
    assert outside
    assert min(outside) == pytest.approx(_case.V_WATER)
    assert max(outside) == pytest.approx(_case.V_WATER)
    seaf = (tmp_path / "seafloor.refl").read_text(encoding="utf-8").splitlines()
    assert float(seaf[0].split()[1]) == pytest.approx(_case.H)
    assert _case.H == pytest.approx(3.0)
    assert _case.AX1 - _case.AX0 == pytest.approx(30.0)
    assert _case.AZ1 - _case.AZ0 == pytest.approx(1.0)
    assert len(_case.OBS_XS) == 10
    assert _case.OBS_XS[0] == pytest.approx(55.0)
    assert _case.OBS_XS[-1] == pytest.approx(145.0)
    assert _case.OBS_XS[1] - _case.OBS_XS[0] == pytest.approx(10.0)
    assert _case.SHOT_DX == pytest.approx(0.2)
    assert _case.RANGE_2 == pytest.approx(20.0)
    assert _case.RANGE_3 == pytest.approx(40.0)
    rows = _r_rows(tmp_path / "geom_inv.dat")
    c2 = [r for r in rows if r[2] == 2]
    c3 = [r for r in rows if r[2] == 3]
    assert c2 and c3
    assert max(abs(sx - rx) for sx, rx, _c in c2) == pytest.approx(_case.RANGE_2)
    assert max(abs(sx - rx) for sx, rx, _c in c3) == pytest.approx(_case.RANGE_3)
    assert all(abs(sx - rx) <= _case.RANGE_2 + 1e-9 for sx, rx, _c in c2)
    assert all(abs(sx - rx) <= _case.RANGE_3 + 1e-9 for sx, rx, _c in c3)
    s = [
        ln
        for ln in (tmp_path / "geom_inv.dat").read_text(encoding="utf-8").splitlines()
        if ln.startswith("s")
    ]
    assert len(s) == 10
    r2 = _r_rows(tmp_path / "geom_c2.dat")
    assert r2 and all(c == 2 for _s, _x, c in r2)
    assert max(abs(sx - rx) for sx, rx, _c in r2) == pytest.approx(_case.RANGE_2)
    vc = (tmp_path / "vcorr.dat").read_text(encoding="utf-8").splitlines()
    assert vc[0].split() == ["2", "2"]
    assert vc[1].split() == ["0", "200"]
    assert float(vc[4].split()[0]) == pytest.approx(_case.LH)
    assert float(vc[6].split()[0]) == pytest.approx(_case.LV)
    assert _case.LH == pytest.approx(8.0)
    assert _case.LV == pytest.approx(1.0)
    assert _case.WSV == pytest.approx(200.0)


def test_plot_inversion_models(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    checker = (
        Path(__file__).resolve().parent.parent
        / "modeling"
        / "tomo2d"
        / "example_water"
        / "water_inv3"
        / "check_water_inv3.py"
    )
    spec = importlib.util.spec_from_file_location("check_water_inv3_plot", checker)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_water_inv3_plot"] = mod
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
