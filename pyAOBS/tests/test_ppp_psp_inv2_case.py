# -*- coding: utf-8 -*-
"""PPP+PSP_inv2：盖层底 4.0、面下顶 7.2；折合 Vs>盖层底 Vp。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASE = ROOT / "modeling" / "tomo2d" / "example_water" / "PPP+PSP_inv2"


def _load():
    spec = importlib.util.spec_from_file_location(
        "make_ppp_psp_inv2_case", CASE / "make_ppp_psp_inv_case.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_ppp_psp_inv2_case"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_write_ppp_psp_inv2_soft_iface(tmp_path: Path) -> None:
    m = _load()
    face, node, below = m.iface_vp()
    assert abs(face - 4.00) < 1e-9
    assert abs(below - 7.20) < 1e-9
    assert below / m.KAPPA_TRUE > face
    assert (below - 0.18) / m.KAPPA_TRUE > face
    assert node < face
    m.write_vp(tmp_path / "true_vp.smesh", **m.VP_TRUE)
    m.write_geom(tmp_path / "geom_joint.dat", codes=m.CODES_JOINT)
    m.write_geom(tmp_path / "geom_psp0.dat", codes=m.CODES_PSP0)
    xs, zs, vp = m.parse_smesh(tmp_path / "true_vp.smesh")
    k_lid = max(i for i, z in enumerate(zs) if z < m.Z_CONV - 1e-9)
    k_b = min(i for i, z in enumerate(zs) if z >= m.Z_CONV - 1e-9)
    assert abs(vp[0][k_lid] - node) < 1e-3
    assert abs(vp[0][k_b] - below) < 1e-3
    joint = (tmp_path / "geom_joint.dat").read_text(encoding="utf-8")
    kinds = [ln.split()[3] for ln in joint.splitlines() if ln.startswith("r")]
    assert set(kinds) == {"0", "6"}
    psp0 = (tmp_path / "geom_psp0.dat").read_text(encoding="utf-8")
    kinds0 = [ln.split()[3] for ln in psp0.splitlines() if ln.startswith("r")]
    assert set(kinds0) == {"0"}


def test_write_lid_damp_and_mixed(tmp_path: Path) -> None:
    m = _load()
    m.write_vp(tmp_path / "rec_vp.smesh", **m.VP_TRUE)
    m.write_vp(tmp_path / "start_vp.smesh", **m.VP_START)
    m.write_mixed_smesh(
        tmp_path / "rec_vp.smesh",
        tmp_path / "rec_vp.smesh",
        tmp_path / "mixed.smesh",
        below_kappa=m.KAPPA_TRUE,
    )
    xs, zs, mix = m.parse_smesh(tmp_path / "mixed.smesh")
    _, _, rec = m.parse_smesh(tmp_path / "rec_vp.smesh")
    k_lid = max(i for i, z in enumerate(zs) if z < m.Z_CONV - 1e-9 and z > m.H)
    k_b = min(i for i, z in enumerate(zs) if z >= m.Z_CONV - 1e-9)
    assert abs(mix[0][k_lid] - rec[0][k_lid]) < 1e-6
    assert abs(mix[0][k_b] - rec[0][k_b] / m.KAPPA_TRUE) < 1e-4
    assert mix[0][k_b] > 4.00
    syn = "5\ns 30 2 2\nr 20 0.01 6 3.1 0.05\nr 22 0.01 6 3.2 0.05\n"
    (tmp_path / "syn6.dat").write_text(syn, encoding="utf-8")
    n = m.relabel_syn_raytype(tmp_path / "syn6.dat", tmp_path / "syn0.dat", frm=6, to=0)
    assert n == 2
    kinds = [ln.split()[3] for ln in (tmp_path / "syn0.dat").read_text(encoding="utf-8").splitlines() if ln.startswith("r")]
    assert kinds == ["0", "0"]

    m.write_lid_damp(tmp_path / "damp_lid.dat")
    _xs, zs_d, w = m.parse_damp(tmp_path / "damp_lid.dat")
    assert max(w[0][i] for i, z in enumerate(zs_d) if m.H < z < m.Z_CONV - 1e-9) == m.DAMP_LID
    assert min(w[0][i] for i, z in enumerate(zs_d) if z >= m.Z_CONV - 1e-9) == m.DAMP_BELOW
    assert m.DAMP_LID == 1000.0
    assert m.DAMP_BELOW == 30.0


def test_inv2_run_script_tightens_s_only() -> None:
    sh = (CASE / "run_wsl.sh").read_text(encoding="utf-8")
    assert "TOMO2D_INV_PSX_ZMAX" not in sh
    assert "TOMO2D_INV_LSQR_MAXITER=8000" in sh
    assert "-Ss200" in sh
    assert "-SV50" in sh
    src = (ROOT / "modeling" / "tomo2d" / "src" / "inverse.cc").read_text(
        encoding="utf-8"
    )
    assert "skip undamped probe" in src
    assert "vs_psx + depth" in src
    assert "inv_lsqr_itermax" in src
    assert "min_below" not in src
    assert "setPsxMoho" in src
    assert "TOMO2D_INV_PSX_ZMAX" not in src
    assert "inv_psx_zmax_below" not in src


def test_inv2_twostep_script() -> None:
    sh = (CASE / "run_wsl_twostep.sh").read_text(encoding="utf-8")
    assert "write_mixed_smesh" in sh
    assert "below_kappa=m.KAPPA_TRUE" in sh
    assert "-k2.0" not in sh
    assert "-Oout_ppp" in sh
    assert "-Oout_psp" in sh
    invs = sh.split('"$BIN/tt_inverse"')
    ppp_inv = invs[1]
    psp_inv = invs[-1]
    assert "-B" not in ppp_inv.split("-Oout_ppp")[0]
    assert "-Yseafloor.refl" in ppp_inv.split("-Oout_ppp")[0]
    assert "-Bconv.refl" not in psp_inv.split("-Oout_psp")[0]
    assert "-DQdamp_lid.dat" in psp_inv
    assert "-DV1" in psp_inv
    assert "geom_psp0.dat" in sh
    assert "-Xconv.refl" not in psp_inv.split("-Oout_psp")[0]
    assert "syn_inv.dat" in sh
    assert "geom_psp0.dat" in sh
    assert "-TV" not in psp_inv.split("-Oout_psp")[0]


def test_inverse_wires_pps_multiples() -> None:
    src = (ROOT / "modeling" / "tomo2d" / "src" / "inverse.cc").read_text(
        encoding="utf-8"
    )
    assert "solve_pps_ss" in src
    assert "solve_pps_peg" in src
    assert "recv peg not wired in inversion yet" not in src
    assert "psx_code==10" in src
    assert "psx_code==14" in src
    sh = (
        ROOT
        / "modeling"
        / "tomo2d"
        / "example_water"
        / "PPP+PSP_inv2"
        / "psp_p_shoot"
        / "thin2km_rugged"
        / "lvz2d"
        / "run_inv_pps_lid.sh"
    ).read_text(encoding="utf-8")
    assert "TOMO2D_INV_FREEZE_BELOW=1" in sh
    assert "geom_7.dat" in sh
    assert "geom_10.dat" in sh
    assert "geom_710.dat" in sh
    assert "FREEZE_LID" in sh


def test_lsqr_precond_median_and_miniter() -> None:
    src = (ROOT / "modeling" / "tomo2d" / "src" / "lsqr.cc").read_text(
        encoding="utf-8"
    )
    hdr = (ROOT / "modeling" / "tomo2d" / "src" / "lsqr.h").read_text(
        encoding="utf-8"
    )
    inv = (ROOT / "modeling" / "tomo2d" / "src" / "inverse.cc").read_text(
        encoding="utf-8"
    )
    assert "allow_precond=true" in hdr
    assert "median_positive" in src
    assert "empty D=0" in src or "空列 D=0" in src
    assert "iter>=miniter_use && test2 < ATOL" in src
    assert "istop=" in src
    assert "dv || dd" in inv
    assert "TOMO2D_INV_LSQR_PRECOND_MINITER" in src


def test_sens_weight_t_r_per_block() -> None:
    inv = (ROOT / "modeling" / "tomo2d" / "src" / "inverse.cc").read_text(
        encoding="utf-8"
    )
    hdr = (ROOT / "modeling" / "tomo2d" / "src" / "inverse.h").read_text(
        encoding="utf-8"
    )
    assert "TOMO2D_INV_SENS_WEIGHT" in inv
    assert "TOMO2D_INV_SENS_KAPPA" in inv
    assert "TOMO2D_INV_SENS_EPS" in inv
    assert "calc_sens_weights" in inv
    assert "calc_sens_weights" in hdr
    assert "sensCouple" in inv
    assert "2.0 / s" in inv
    assert "fill_vel_averaging(Rv_h, Rv_v, mvscale, true)" in inv
    assert "fill_vel_damping(Tv, true)" in inv
    assert "per-block Vp / Vs-lid / Vs-below / Moho" in inv
    assert "Td(i)[i] = fac * (sens_weight ? sensDepW(i) : 1.0)" in inv


def test_linesearch_armijo() -> None:
    inv = (ROOT / "modeling" / "tomo2d" / "src" / "inverse.cc").read_text(
        encoding="utf-8"
    )
    hdr = (ROOT / "modeling" / "tomo2d" / "src" / "inverse.h").read_text(
        encoding="utf-8"
    )
    assert "TOMO2D_INV_LINESEARCH" in inv
    assert "TOMO2D_INV_LS_C" in inv
    assert "TOMO2D_INV_LS_RHO" in inv
    assert "TOMO2D_INV_LS_AMIN" in inv
    assert "void applyDmodel" in hdr
    assert "void restoreModel" in hdr
    assert "ls_phi0 + ls_c*ls_alpha*ls_slope" in inv
    assert "ls_alpha *= ls_rho" in inv
    assert "line_search && !do_lm && single_iset && !strategy_skip_update" in inv
    assert "Armijo on true chi2" in inv
    assert "writeCurrentModels" in inv
    assert "linesearch stalled" in inv


def test_lvz2d_start_1d_biased_not_just_lvz() -> None:
    """反演初值 1D 必须整体偏离真背景，不能只缺一块低速。"""
    lvz = (
        ROOT
        / "modeling"
        / "tomo2d"
        / "example_water"
        / "PPP+PSP_inv2"
        / "psp_p_shoot"
        / "thin2km_rugged"
        / "lvz2d"
    )
    spec = importlib.util.spec_from_file_location("make_fwd_all_start1d", lvz / "make_fwd_all.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    t, s = mod.TRUE_1D, mod.START_1D
    assert s.lid0 >= t.lid0 + 0.3
    assert s.lid_iface >= t.lid_iface + 0.2
    assert s.crust0 >= t.crust0 + 0.3
    assert s.crust_moho >= t.crust_moho + 0.2
    assert s.mantle0 >= t.mantle0 + 0.2
    assert s.mantle_grad < t.mantle_grad
    m612 = (lvz / "make_inv_612.py").read_text(encoding="utf-8")
    assert "bg=fwd.START_1D" in m612
    assert 'DEST / "moho.refl"' in m612
    mpps = (lvz / "make_inv_pps_lid.py").read_text(encoding="utf-8")
    assert "bg=fwd.START_1D" in mpps


def test_lm_trust_region() -> None:
    inv = (ROOT / "modeling" / "tomo2d" / "src" / "inverse.cc").read_text(
        encoding="utf-8"
    )
    hdr = (ROOT / "modeling" / "tomo2d" / "src" / "inverse.h").read_text(
        encoding="utf-8"
    )
    assert "TOMO2D_INV_LM" in inv
    assert "TOMO2D_INV_LM_UP" in inv
    assert "TOMO2D_INV_LM_RHO_ACCEPT" in inv
    assert "bool use_lm" in hdr
    assert "weight_d_v *= lm_lambda" in inv
    assert "lm re-LSQR" in inv
    assert "trust-region: scale T and re-LSQR" in inv
    assert "lm ON: linesearch ignored" in inv

