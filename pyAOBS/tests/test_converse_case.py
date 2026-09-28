# -*- coding: utf-8 -*-
"""折合 PSP 示例工区：生成文件、参考走时、作图。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FWD_DIR = ROOT / "modeling" / "tomo2d" / "example_water" / "converse_fwd"
INV_DIR = ROOT / "modeling" / "tomo2d" / "example_water" / "converse_inv"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_fwd = _load("make_converse_fwd_case", FWD_DIR / "make_converse_fwd_case.py")
_inv = _load("make_converse_inv_case", INV_DIR / "make_converse_inv_case.py")


def test_analytic_zero_offset() -> None:
    t0 = _fwd.t_head_wave(0.0)
    t6 = _fwd.t_psp_ref(0.0)
    assert t0 == pytest.approx((_fwd.H - _fwd.SHOT_Z) / _fwd.V_WATER, abs=1e-6)
    assert t6 == pytest.approx(
        (_fwd.H - _fwd.SHOT_Z) / _fwd.V_WATER + 2.0 * _fwd.t_lid_vertical(),
        abs=2e-3,
    )
    assert t6 > t0 + 1.5


def test_head_wave_faster_than_water_direct() -> None:
    dx = 20.0
    t_dir = (dx**2 + (_fwd.H - _fwd.SHOT_Z) ** 2) ** 0.5 / _fwd.V_WATER
    assert _fwd.t_head_wave(dx) < t_dir - 1.0
    assert _fwd.t_psp_ref(0.0) > _fwd.t_head_wave(0.0) + 1.5


def test_write_converse_fwd_case(tmp_path: Path) -> None:
    _fwd.write_smesh(tmp_path / "converse.smesh")
    _fwd.write_seafloor(tmp_path / "seafloor.refl")
    _fwd.write_conv(tmp_path / "conv.refl")
    _fwd.write_geom(tmp_path / "geom_conv.dat")
    _fwd.write_analytic(tmp_path / "analytic.txt")
    sm = (tmp_path / "converse.smesh").read_text(encoding="utf-8").splitlines()
    xs = [float(x) for x in sm[1].split()]
    zs = [float(z) for z in sm[3].split()]
    vcol = [float(v) for v in sm[4].split()]
    water = [v for z, v in zip(zs, vcol) if z <= _fwd.H + 1e-9]
    lid = [v for z, v in zip(zs, vcol) if _fwd.H + 1e-9 < z < _fwd.Z_CONV - 1e-9]
    slow = [v for z, v in zip(zs, vcol) if z > _fwd.Z_CONV + 1e-9]
    assert water and lid and slow
    assert all(v == pytest.approx(_fwd.V_WATER, abs=1e-4) for v in water)
    assert max(lid) > min(lid) + 0.8
    assert min(lid) == pytest.approx(_fwd.VP_SED0, abs=0.2)
    assert max(lid) == pytest.approx(_fwd.vp_sed(_fwd.Z_CONV), abs=0.25)
    assert max(slow) > min(slow) + 0.2
    assert min(slow) == pytest.approx(_fwd.VS0, abs=0.2)
    geom = (tmp_path / "geom_conv.dat").read_text(encoding="utf-8").splitlines()
    r = [ln for ln in geom if ln.startswith("r")]
    assert len(r) == len(_fwd.OFFSETS) * 2
    assert int(r[0][21:26]) == 0
    assert int(r[-1][21:26]) == 6
    conv = (tmp_path / "conv.refl").read_text(encoding="utf-8").splitlines()
    assert float(conv[0].split()[1]) == pytest.approx(_fwd.Z_CONV)
    _ = xs


def test_write_converse_inv_case(tmp_path: Path) -> None:
    _inv.write_smesh(tmp_path / "true.smesh", _inv.VS_TRUE)
    _inv.write_smesh(tmp_path / "start.smesh", _inv.VS_START)
    _inv.write_vp(tmp_path / "true_vp.smesh", _inv.VS_TRUE)
    _inv.write_vs(tmp_path / "true_vs.smesh", _inv.VS_TRUE)
    _inv.write_vs(tmp_path / "start_vs.smesh", _inv.VS_START)
    _inv.write_seafloor(tmp_path / "seafloor.refl")
    _inv.write_conv(tmp_path / "conv.refl")
    _inv.write_geom(tmp_path / "geom_inv.dat")
    _inv.write_vcorr(tmp_path / "vcorr.dat")
    xs, zs, vt = _inv.parse_smesh(tmp_path / "true.smesh")
    _, _, vs = _inv.parse_smesh(tmp_path / "start.smesh")
    x_lo, x_hi = _inv.illum_x_range()
    tw, *_ = _inv.node_stats(
        xs, zs, vt, x_lo=x_lo, x_hi=x_hi, z_lo=0.0, z_hi=_inv.H, z_hi_inclusive=True
    )
    lid, *_ = _inv.node_stats(
        xs, zs, vt, x_lo=x_lo, x_hi=x_hi, z_lo=_inv.H + 1e-6, z_hi=_inv.Z_CONV
    )
    st_s, *_ = _inv.node_stats(
        xs,
        zs,
        vs,
        x_lo=x_lo,
        x_hi=x_hi,
        z_lo=_inv.Z_CONV,
        z_hi=_inv.Z_CONV + 2.5,
        z_hi_inclusive=True,
    )
    t_s, *_ = _inv.node_stats(
        xs,
        zs,
        vt,
        x_lo=x_lo,
        x_hi=x_hi,
        z_lo=_inv.Z_CONV,
        z_hi=_inv.Z_CONV + 2.5,
        z_hi_inclusive=True,
    )
    assert tw == pytest.approx(_inv.V_WATER, abs=1e-4)
    assert lid == pytest.approx(_inv.expected_lid_mean(), abs=0.15)
    assert t_s < st_s - 0.2
    _, _, vp = _inv.parse_smesh(tmp_path / "true_vp.smesh")
    _, _, vts = _inv.parse_smesh(tmp_path / "true_vs.smesh")
    _, _, vss = _inv.parse_smesh(tmp_path / "start_vs.smesh")
    lid_p, *_ = _inv.node_stats(
        xs, zs, vp, x_lo=x_lo, x_hi=x_hi, z_lo=_inv.H + 1e-6, z_hi=_inv.Z_CONV
    )
    lid_vs_t, *_ = _inv.node_stats(
        xs, zs, vts, x_lo=x_lo, x_hi=x_hi, z_lo=_inv.H + 1e-6, z_hi=_inv.Z_CONV
    )
    lid_vs_s, *_ = _inv.node_stats(
        xs, zs, vss, x_lo=x_lo, x_hi=x_hi, z_lo=_inv.H + 1e-6, z_hi=_inv.Z_CONV
    )
    t_vs, *_ = _inv.node_stats(
        xs, zs, vts, x_lo=x_lo, x_hi=x_hi, z_lo=_inv.Z_CONV, z_hi=_inv.Z_CONV + 2.5,
        z_hi_inclusive=True,
    )
    s_vs, *_ = _inv.node_stats(
        xs, zs, vss, x_lo=x_lo, x_hi=x_hi, z_lo=_inv.Z_CONV, z_hi=_inv.Z_CONV + 2.5,
        z_hi_inclusive=True,
    )
    assert lid_p == pytest.approx(_inv.expected_lid_mean(), abs=0.15)
    assert lid_vs_t == pytest.approx(lid_vs_s, abs=1e-4)
    assert t_vs < s_vs - 0.2
    geom = (tmp_path / "geom_inv.dat").read_text(encoding="utf-8").splitlines()
    s = [ln for ln in geom if ln.startswith("s")]
    r = [ln for ln in geom if ln.startswith("r")]
    assert _inv.OBS_Z == pytest.approx(_inv.H)
    assert _inv.SHOT_Z == pytest.approx(_fwd.SHOT_Z)
    assert all(float(ln.split()[2]) == pytest.approx(_inv.H) for ln in s)
    assert all(float(ln.split()[2]) == pytest.approx(_inv.SHOT_Z) for ln in r)
    assert len(r) == len(_inv.OBS_XS) * len(_inv.SHOT_XS)
    assert int(r[0][21:26]) == 6


def test_compare_syn_to_analytic_accepts_perfect() -> None:
    lines = ["1", "s 50.0 2.0 2"]
    lines.append(f"r 50.0 0.01 0 {_fwd.t_head_wave(0.0):.6f} 0.01")
    lines.append(f"r 58.0 0.01 6 {_fwd.t_psp_ref(8.0):.6f} 0.01")
    recs = _fwd.compare_syn_to_analytic("\n".join(lines) + "\n", atol=1e-4)
    assert [r[0] for r in recs] == [0, 6]


def _load_fwd_checker(name: str):
    return _load(name, FWD_DIR / "check_converse_fwd.py")


def _load_inv_checker(name: str):
    return _load(name, INV_DIR / "check_converse_inv.py")


def test_split_psp_phase_segments() -> None:
    mod = _load_fwd_checker("check_converse_fwd_split")
    segs = mod.split_psp_phase_segments(
        [50.0, 50.0, 58.0, 58.0],
        [2.0, 5.0, 6.5, 0.01],
        5.0,
    )
    kinds = [s[2] for s in segs]
    assert kinds == [False, True, False]
    assert max(segs[1][1]) > 5.0
    assert max(segs[0][1]) <= 5.0 + 1e-9
    assert min(segs[2][1]) < 5.0


def test_plot_fwd_ttimes(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    mod = _load_fwd_checker("check_converse_fwd_tt")
    recs = [
        (0, 0.0, 1.33, 1.33, 0.0),
        (6, 0.0, 2.33, 2.33, 0.0),
        (0, 8.0, 2.70, 2.65, 0.05),
        (6, 8.0, 4.80, 4.75, 0.05),
    ]
    out = tmp_path / "check_ttimes.png"
    mod.plot_ttimes(recs, out, show=False)
    assert out.is_file() and out.stat().st_size > 2000


def test_plot_inv_ttimes(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    mod = _load_inv_checker("check_converse_inv_tt")
    obs = [(6, 58.0, 0.01, 5.10, 50.0)]
    start = [(6, 58.0, 0.01, 4.70, 50.0)]
    rec = [(6, 58.0, 0.01, 5.05, 50.0)]
    out = tmp_path / "check_inv_ttimes.png"
    mod.plot_ttimes_fit(obs, start, rec, out, show=False)
    assert out.is_file() and out.stat().st_size > 2000
