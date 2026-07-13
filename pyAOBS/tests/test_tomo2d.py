from unittest.mock import patch

import pytest

from pyAOBS.modeling.tomo2d.tomand import TomoAnd


pytestmark = pytest.mark.unit


def test_gen_smesh_builds_expected_args_uniform():
    tomo = TomoAnd(bin_path="dummy_bin")
    with patch.object(tomo, "_run_cmd", return_value="ok") as mocked:
        result = tomo.gen_smesh(
            vel_opt="uniform",
            v0=1.5,
            gradient=0.1,
            grid_opt="uniform",
            nx=101,
            nz=51,
            xmax=100.0,
            zmax=30.0,
            v_air=0.33,
        )

    assert result == "ok"
    mocked.assert_called_once_with(
        "gen_smesh",
        args=["-A1.5", "-B0.1", "-N101/51", "-D100.0/30.0", "-R0.33"],
    )


def test_gen_vcorr_builds_expected_args_variable_grid():
    tomo = TomoAnd(bin_path="dummy_bin")
    with patch.object(tomo, "_run_cmd", return_value="ok") as mocked:
        tomo.gen_vcorr(
            vel_opt="uniform",
            abnormal_h=2.0,
            abnormal_v=1.0,
            normal_h=5.0,
            normal_v=3.0,
            grid_opt="variable",
            x_file="x.dat",
            z_file="z.dat",
            topo_file="topo.dat",
        )

    mocked.assert_called_once_with(
        "gen_vcorr",
        args=["-A2.0/1.0/5.0/3.0", "-Xx.dat", "-Zz.dat", "-Ttopo.dat"],
    )


def test_gen_damp_raises_when_grid_params_incomplete():
    tomo = TomoAnd(bin_path="dummy_bin")
    with pytest.raises(ValueError, match="缺少必需参数"):
        tomo.gen_damp(
            vel_opt="uniform",
            abnormal_damp=20,
            normal_damp=10,
            grid_opt="uniform",
            nx=101,
            xmax=100.0,
            zmax=30.0,
        )


def test_tt_forward_raises_when_numerical_options_partial():
    tomo = TomoAnd(bin_path="dummy_bin")
    with pytest.raises(ValueError, match="tt_forward\\(numerical options\\)"):
        tomo.tt_forward(smesh="model.smesh", xorder=3, zorder=3)


def test_stat_smesh_list_r_requires_average_file():
    tomo = TomoAnd(bin_path="dummy_bin")
    with pytest.raises(ValueError, match="stat_smesh\\(cmd_type='r'\\)"):
        tomo.stat_smesh(mode="list", list_file="files.lst", cmd_type="r")


def test_edit_smesh_invalid_cmd_type_raises():
    tomo = TomoAnd(bin_path="dummy_bin")
    with pytest.raises(ValueError, match="不支持的 cmd_type"):
        tomo.edit_smesh(smesh_file="in.smesh", cmd_type="unknown")


def test_tt_inverse_builds_smoothing_and_damping_args():
    tomo = TomoAnd(bin_path="dummy_bin")
    with patch.object(tomo, "_run_cmd", return_value="ok") as mocked:
        tomo.tt_inverse(
            mesh="model.smesh",
            data="obs.dat",
            xorder=3,
            zorder=3,
            clen=5.0,
            nintp=9,
            bend_cg_tol=1e-5,
            bend_br_tol=1e-5,
            smooth_opts={"vel": 0.2, "dep": 0.3},
            damp_opts={"vel": 10.0, "dep": 5.0},
            verbose=True,
            verbose_level=2,
        )

    mocked.assert_called_once_with(
        "tt_inverse",
        args=[
            "-Mmodel.smesh",
            "-Gobs.dat",
            "-N3/3/5.0/9/1e-05/1e-05",
            "-SV0.2",
            "-SD0.3",
            "-DV10.0",
            "-DD5.0",
            "-V2",
        ],
    )


def test_run_cmd_check_only_returns_help_text():
    tomo = TomoAnd(bin_path="dummy_bin")
    help_text = tomo._run_cmd("gen_smesh", check_only=True)
    assert isinstance(help_text, str)
    assert "gen_smesh" in help_text
