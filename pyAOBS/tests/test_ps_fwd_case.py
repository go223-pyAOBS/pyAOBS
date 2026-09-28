# -*- coding: utf-8 -*-
"""PPP/PPS/PSS 正演工区生成。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PS = ROOT / "modeling" / "tomo2d" / "example_water" / "ps_fwd"


def _load():
    spec = importlib.util.spec_from_file_location("make_ps_fwd_case", PS / "make_ps_fwd_case.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_ps_fwd_case"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_write_ps_fwd_case(tmp_path: Path) -> None:
    m = _load()
    m.write_smesh(tmp_path / "vp.smesh")
    m.write_seafloor(tmp_path / "seafloor.refl")
    m.write_conv(tmp_path / "conv.refl")
    m.write_geom(tmp_path / "geom_ps.dat")
    sm = (tmp_path / "vp.smesh").read_text(encoding="utf-8").splitlines()
    zs = [float(z) for z in sm[3].split()]
    vcol = [float(v) for v in sm[4].split()]
    water = [v for z, v in zip(zs, vcol) if z <= m.H + 1e-9]
    lid = [v for z, v in zip(zs, vcol) if m.H + 1e-9 < z < m.Z_CONV - 1e-9]
    crust = [v for z, v in zip(zs, vcol) if z > m.Z_CONV + 1e-9]
    assert water and lid and crust
    assert all(abs(v - m.V_WATER) < 1e-4 for v in water)
    assert max(lid) > min(lid)
    assert max(crust) > min(crust)
    assert min(crust) > 5.5
    assert lid[-1] < crust[0]
    geom = (tmp_path / "geom_ps.dat").read_text(encoding="utf-8")
    assert "    7" in geom and "    8" in geom


def test_zero_offset_lid_has_pink_s() -> None:
    """零偏炮–台 x 相同，旧判据会把台侧盖层 S 全部画成 P。"""
    sys.path.insert(0, str(PS))
    sys.path.insert(0, str(PS.parent / "converse_fwd"))
    import check_ps_fwd as chk

    # 下行 P（水+盖层）再上行 S，不进入转换面以下。
    xs = [50.0] * 11
    zs = [0.01, 1.0, 2.0, 3.0, 4.0, 4.9, 4.0, 3.0, 2.5, 2.2, 2.0]
    segs = chk.split_psx_phase_segments(xs, zs, 7)
    assert segs
    assert any(is_s for _x, _z, is_s in segs)
    assert any(not is_s for _x, _z, is_s in segs)
    water_s = [
        z
        for sx, sz, is_s in segs
        if is_s
        for z in sz
        if z < chk.H - 1e-3
    ]
    assert not water_s


def test_pss_interface_hug_is_pink_s() -> None:
    """PSS 面下单跳贴在转换面上时，水平段必须是 S，不能画成炮侧 P。"""
    sys.path.insert(0, str(PS))
    sys.path.insert(0, str(PS.parent / "converse_fwd"))
    import check_ps_fwd as chk

    xs = [70.0, 70.0, 70.0, 60.0, 50.0, 50.0]
    zs = [0.2, 2.0, 5.0, 5.0, 5.0, 2.0]
    segs = chk.split_psx_phase_segments(xs, zs, 8)
    hug = [
        (sx, sz, is_s)
        for sx, sz, is_s in segs
        if is_s and max(sx) - min(sx) > 1.0
    ]
    assert hug
    assert any(abs(z - chk.Z_CONV) < 0.05 for _sx, sz, _s in hug for z in sz)
    water_s = [
        z
        for sx, sz, is_s in segs
        if is_s
        for z in sz
        if z < chk.H - 1e-3
    ]
    assert not water_s


def test_conversion_points_mark_shot_and_obs_side() -> None:
    """炮侧 C 在第一次过面，台侧 C 在最后一次过面。"""
    sys.path.insert(0, str(PS))
    sys.path.insert(0, str(PS.parent / "converse_fwd"))
    import check_ps_fwd as chk

    xs = [70.0, 70.0, 70.0, 60.0, 50.0, 50.0, 50.0]
    zs = [0.2, 2.0, 5.0, 6.2, 5.0, 3.0, 2.0]
    pts = chk.find_psx_conversion_points(xs, zs)
    assert pts is not None
    c_shot, c_obs = pts
    assert abs(c_shot[0] - 70.0) < 1e-6
    assert abs(c_obs[0] - 50.0) < 1e-6
    assert abs(c_shot[1] - chk.Z_CONV) < 1e-6
    assert abs(c_obs[1] - chk.Z_CONV) < 1e-6


def test_real_ps_rays_have_conversion_points() -> None:
    """7/8 只要仍过转换面即可；位置由最短路径决定，不要求铅垂或 Snell。"""
    sys.path.insert(0, str(PS))
    sys.path.insert(0, str(PS.parent / "converse_fwd"))
    import check_ps_fwd as chk

    ray_path = PS / "rays_ps.dat"
    syn_path = PS / "syn_ps.dat"
    if not ray_path.is_file() or not syn_path.is_file():
        return
    rays = chk.parse_ray_file(ray_path)
    recs = chk.parse_syn(syn_path.read_text(encoding="utf-8"))
    n = 0
    for (code, dx, _t), (xs, zs) in zip(recs, rays):
        if code not in (7, 8) or dx < 8.0:
            continue
        pts = chk.find_psx_conversion_points(xs, zs)
        assert pts is not None
        n += 1
    assert n >= 4
