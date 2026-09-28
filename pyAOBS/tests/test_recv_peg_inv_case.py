# -*- coding: utf-8 -*-
"""壳内高速体 + 0/1 vs 4/5 对照工区。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_HERE = (
    Path(__file__).resolve().parent.parent
    / "modeling"
    / "tomo2d"
    / "example_water"
    / "recv_peg_inv"
)
_case = _load("make_recv_peg_inv_case", _HERE / "make_recv_peg_inv_case.py")


def test_write_recv_peg_inv_case(tmp_path: Path) -> None:
    _case.write_true(tmp_path / "true.smesh")
    _case.write_start(tmp_path / "start.smesh")
    _case.write_seafloor(tmp_path / "seafloor.refl")
    _case.write_moho(tmp_path / "moho.refl")
    _case.write_geom(tmp_path / "geom_inv_01.dat", (0, 1))
    _case.write_geom(tmp_path / "geom_inv_45.dat", (4, 5))
    _case.write_geom(tmp_path / "geom_inv_0145.dat", (0, 1, 4, 5))
    _case.write_vcorr(tmp_path / "vcorr.dat")
    xs, zs, vt = _case.parse_smesh(tmp_path / "true.smesh")
    _, _, vs = _case.parse_smesh(tmp_path / "start.smesh")
    x_lo, x_hi = _case.illum_x_range()
    tw, *_ = _case.node_stats(xs, zs, vt, x_lo=x_lo, x_hi=x_hi, layer="water")
    sw, *_ = _case.node_stats(xs, zs, vs, x_lo=x_lo, x_hi=x_hi, layer="water")
    tm, *_ = _case.node_stats(xs, zs, vt, x_lo=x_lo, x_hi=x_hi, layer="mantle")
    assert tw == pytest.approx(_case.V_WATER, abs=1e-4)
    assert sw == pytest.approx(_case.V_WATER, abs=1e-4)
    assert tm == pytest.approx(_case.V_MANTLE, abs=1e-4)
    ta, *_ = _case.box_stats(
        xs, zs, vt, x_lo=_case.AX0, x_hi=_case.AX1, z_lo=_case.AZ0, z_hi=_case.AZ1
    )
    sa, *_ = _case.box_stats(
        xs, zs, vs, x_lo=_case.AX0, x_hi=_case.AX1, z_lo=_case.AZ0, z_hi=_case.AZ1
    )
    assert ta == pytest.approx(sa + _case.DV_ANOM, abs=1e-3)
    mid_x = 0.5 * (_case.AX0 + _case.AX1)
    mid_z = 0.5 * (_case.AZ0 + _case.AZ1)
    assert _case.in_anomaly(mid_x, mid_z)
    assert not _case.in_anomaly(mid_x, 5.5)
    assert _case.OBS_XS == (38.0, 44.0, 50.0, 56.0, 62.0)
    shots01 = [sx for ox in _case.OBS_XS for sx in _case.iter_shots(ox, 0)]
    shots45 = [sx for ox in _case.OBS_XS for sx in _case.iter_shots(ox, 4)]
    n01, n45 = len(shots01), len(shots45)
    pad_lo = _case.XMIN + _case.SHOT_PAD
    pad_hi = _case.XMAX - _case.SHOT_PAD
    assert shots01 and min(shots01) >= pad_lo - 1e-9
    assert max(shots01) <= pad_hi + 1e-9
    assert shots45 and min(shots45) >= pad_lo - 1e-9
    assert max(shots45) <= pad_hi + 1e-9
    # 每台 0/1 ±30 都应留在边距内，不能再出现 x=0/100
    assert all(len(list(_case.iter_shots(ox, 0))) == 22 for ox in _case.OBS_XS)
    assert 0.0 not in shots01 and 100.0 not in shots01
    for name, expect in (
        ("geom_inv_01.dat", {0: n01, 1: n01}),
        ("geom_inv_45.dat", {4: n45, 5: n45}),
        ("geom_inv_0145.dat", {0: n01, 1: n01, 4: n45, 5: n45}),
    ):
        r = [
            ln
            for ln in (tmp_path / name).read_text(encoding="utf-8").splitlines()
            if ln.startswith("r")
        ]
        codes = [int(ln[21:26]) for ln in r]
        for kind, n in expect.items():
            assert codes.count(kind) == n
        uncert = [float(ln.split()[-1]) for ln in r]
        assert uncert and min(uncert) >= _case.TT_ERR - 1e-9
        for ln in r:
            parts = ln.split()
            code = int(parts[3])
            x = float(parts[1])
            # s-line x is not on r; check |shot-obs| via offsets_for
            assert any(
                abs(x - (ox + dx)) < 1e-6
                for ox in _case.OBS_XS
                for dx in _case.offsets_for(code)
            )
    syn = tmp_path / "syn.dat"
    syn.write_text("1\ns 40 2 1\nr 10 0.01 0 7.0 0.01\n", encoding="utf-8")
    _case.set_pick_uncert(syn)
    assert syn.read_text(encoding="utf-8").split()[-1] == f"{_case.TT_ERR:g}"
    assert _case.TT_ERR >= 0.05
    assert _case.TV > 0
    assert max(abs(dx) for dx in _case.OFFSETS_01) == 30.0
    assert max(abs(dx) for dx in _case.OFFSETS_45) == 40.0
    assert _case.LV == pytest.approx(0.6)
    vcorr_txt = (tmp_path / "vcorr.dat").read_text(encoding="utf-8")
    assert "0.6" in vcorr_txt
    assert "0.3\n" not in vcorr_txt


def test_plot_recv_peg_inv_models(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    chk = _load("check_recv_peg_inv_plot", _HERE / "check_recv_peg_inv.py")
    true_p = tmp_path / "true.smesh"
    start_p = tmp_path / "start.smesh"
    rec_p = tmp_path / "rec.smesh"
    _case.write_true(true_p)
    _case.write_start(start_p)
    _case.write_smesh(rec_p, anomaly=True)
    geom = tmp_path / "geom_inv_01.dat"
    _case.write_geom(geom, (0, 1))
    out = tmp_path / "check_inv_models.png"
    chk.plot_inversion(rec_p, true_p, start_p, out, show=False, geom_path=geom)
    assert out.is_file() and out.stat().st_size > 2000
    cmp = tmp_path / "check_inv_compare.png"
    chk.plot_compare(
        rec_p,
        start_p,
        true_p,
        start_p,
        cmp,
        show=False,
        geom_path=geom,
        labels=("真异常", "初值"),
    )
    assert cmp.is_file() and cmp.stat().st_size > 2000


def test_plot_ttimes(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    chk = _load("check_recv_peg_inv_tt", _HERE / "check_recv_peg_inv.py")
    obs = [(0, 60.0, 0.01, 4.0, 50.0), (4, 60.0, 0.01, 6.6, 50.0)]
    start = [(0, 60.0, 0.01, 3.9, 50.0), (4, 60.0, 0.01, 6.5, 50.0)]
    rec = [(0, 60.0, 0.01, 4.0, 50.0), (4, 60.0, 0.01, 6.55, 50.0)]
    out = tmp_path / "check_inv_ttimes.png"
    chk.plot_ttimes_fit(obs, start, rec, out, show=False)
    assert out.is_file() and out.stat().st_size > 2000
