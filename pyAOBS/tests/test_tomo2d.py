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


def test_gen_smesh_hang_sea_surface_appends_s_and_g():
    tomo = TomoAnd(bin_path="dummy_bin")
    args = tomo._build_gen_smesh_program_args(
        {
            "vel_opt": "zelt",
            "grid_opt": "zelt",
            "v_in": "v.in",
            "ilayer": 2,
            "dx": 0.5,
            "z_file": "z.dat",
            "hang_sea_surface": True,
            "seafloor_out": "sf.dat",
        }
    )
    assert args is not None
    assert "-S" in args
    assert "-Gsf.dat" in args
    assert "-Cv.in/2" in args


def test_gen_smesh_hang_sea_surface_off_by_default():
    tomo = TomoAnd(bin_path="dummy_bin")
    args = tomo._build_gen_smesh_program_args(
        {
            "vel_opt": "zelt",
            "grid_opt": "zelt",
            "v_in": "v.in",
            "ilayer": 2,
            "dx": 0.5,
            "z_file": "z.dat",
        }
    )
    assert args is not None
    assert "-S" not in args
    assert not any(str(a).startswith("-G") for a in args)


def test_gen_smesh_seafloor_out_requires_hang():
    tomo = TomoAnd(bin_path="dummy_bin")
    with pytest.raises(ValueError, match="-S"):
        tomo._build_gen_smesh_program_args(
            {
                "vel_opt": "zelt",
                "grid_opt": "zelt",
                "v_in": "v.in",
                "ilayer": 2,
                "dx": 0.5,
                "z_file": "z.dat",
                "seafloor_out": "sf.dat",
            }
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


def test_gen_vcorr_simple_2x2_writes_correlation_length_file(tmp_path):
    from pyAOBS.modeling.tomo2d.simple_vcorr import format_simple_vcorr

    expected = (
        "2 2\n"
        "0 410\n"
        "0.0 0.0\n"
        "0.0 45.0\n"
        "2.0 4.0\n"
        "2.0 4.0\n"
        "1.0 4.0\n"
        "1.0 4.0\n"
    )
    assert format_simple_vcorr(
        Lht=2.0, Lhb=4.0, Lvt=1.0, Lvb=4.0, xmax=410, zmax=45
    ) == expected

    tomo = TomoAnd(bin_path="dummy_bin")
    tomo.proc_cwd = str(tmp_path)
    out = tmp_path / "vcorr.dat"
    with patch.object(tomo, "_run_cmd") as mocked:
        text = tomo.gen_vcorr(
            mode="simple_2x2",
            Lht=2.0,
            Lhb=4.0,
            Lvt=1.0,
            Lvb=4.0,
            xmax=410,
            zmax=45,
            out_file="vcorr.dat",
        )
    mocked.assert_not_called()
    assert text == expected
    assert out.read_text(encoding="utf-8") == expected
    assert tomo.resolve_cmdline_gen_vcorr(
        mode="simple_2x2",
        Lht=2.0,
        Lhb=4.0,
        Lvt=1.0,
        Lvb=4.0,
        xmax=410,
        zmax=45,
        out_file="vcorr.dat",
    ) is None


def test_gen_dcorr_builds_uniform_and_from_vcorr_args():
    tomo = TomoAnd(bin_path="dummy_bin")
    with patch.object(tomo, "_run_cmd", return_value="ok") as mocked:
        tomo.gen_dcorr(mode="uniform", lh=4.0, xmin=0.0, xmax=410.0, out_file="dcorr.dat")
    mocked.assert_called_once_with("gen_dcorr", args=["-A4.0", "-D0.0/410.0"])

    with patch.object(tomo, "_run_cmd", return_value="ok") as mocked:
        with patch.object(tomo, "_verify_gen_dcorr_inputs"):
            tomo.gen_dcorr(
                mode="from_vcorr",
                vcorr_file="corr_v.dat",
                refl_file="refl.dat",
                out_file="dcorr.dat",
            )
    mocked.assert_called_once_with(
        "gen_dcorr", args=["-Vcorr_v.dat", "-Rrefl.dat"]
    )

    with patch.object(tomo, "_run_cmd", return_value="ok") as mocked:
        with patch.object(tomo, "_verify_gen_dcorr_inputs"):
            tomo.gen_dcorr(
                mode="zelt",
                abnormal_d=2.0,
                normal_d=8.0,
                v_in="v.in",
                ilayer=1,
                dx=0.5,
                top_layer=2,
                bot_layer=5,
            )
    mocked.assert_called_once_with(
        "gen_dcorr",
        args=["-A2.0/8.0", "-Cv.in/1", "-E0.5", "-F2/5"],
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


def test_tt_inverse_bare_s_uses_mesh_topo():
    tomo = TomoAnd(bin_path="dummy_bin")
    with patch.object(tomo, "_run_cmd", return_value="ok") as mocked:
        tomo.tt_inverse(
            mesh="model.smesh",
            data="obs.dat",
            apply_filter=True,
        )
    mocked.assert_called_once()
    args = mocked.call_args.kwargs["args"]
    assert "-s" in args
    assert not any(a.startswith("-s") and a != "-s" for a in args)


def test_tt_inverse_s_with_bound_file():
    tomo = TomoAnd(bin_path="dummy_bin")
    prog = tomo._build_tt_inverse_program_args(
        "model.smesh", "obs.dat", {"filter_bound_file": "bound.dat"}
    )
    assert prog is not None
    assert "-sbound.dat" in prog


def test_tt_inverse_freeze_refl_appends_u():
    tomo = TomoAnd(bin_path="dummy_bin")
    prog = tomo._build_tt_inverse_program_args(
        "model.smesh", "obs.dat", {"refl_file": "seafloor.dat", "freeze_refl": True}
    )
    assert prog is not None
    assert "-Fseafloor.dat" in prog
    assert "-u" in prog
    i_f = prog.index("-Fseafloor.dat")
    assert prog[i_f + 1] == "-u"


def test_tt_forward_vsmesh_appends_u():
    tomo = TomoAnd(bin_path="dummy_bin")
    prog = tomo._build_tt_forward_program_args(
        "model.smesh",
        "geom.dat",
        {"vsmesh": "vs.smesh"},
    )
    assert prog is not None
    assert "-Uvs.smesh" in prog


def test_tt_inverse_vsmesh_appends_u():
    tomo = TomoAnd(bin_path="dummy_bin")
    prog = tomo._build_tt_inverse_program_args(
        "model.smesh",
        "obs.dat",
        {"vsmesh": "vs.smesh"},
    )
    assert prog is not None
    assert "-Uvs.smesh" in prog


def test_tt_forward_conv_file_appends_x():
    tomo = TomoAnd(bin_path="dummy_bin")
    prog = tomo._build_tt_forward_program_args(
        "model.smesh",
        "geom.dat",
        {"conv_file": "conv.dat"},
    )
    assert prog is not None
    assert "-Xconv.dat" in prog
    assert "-Bconv.dat" not in prog


def test_tt_inverse_conv_file_appends_b():
    tomo = TomoAnd(bin_path="dummy_bin")
    prog = tomo._build_tt_inverse_program_args(
        "model.smesh",
        "obs.dat",
        {"conv_file": "conv.dat"},
    )
    assert prog is not None
    assert "-Bconv.dat" in prog
    assert "-Yconv.dat" not in prog


def test_tt_forward_seafloor_file_appends_b():
    tomo = TomoAnd(bin_path="dummy_bin")
    prog = tomo._build_tt_forward_program_args(
        "model.smesh",
        "geom.dat",
        {"refl_file": "moho.dat", "seafloor_file": "seafloor.dat"},
    )
    assert prog is not None
    assert "-Fmoho.dat" in prog
    assert "-Bseafloor.dat" in prog


def test_tt_inverse_seafloor_file_appends_y_flag():
    tomo = TomoAnd(bin_path="dummy_bin")
    prog = tomo._build_tt_inverse_program_args(
        "model.smesh",
        "obs.dat",
        {"seafloor_file": "seafloor.dat", "invert_water_only": True},
    )
    assert prog is not None
    assert "-Yseafloor.dat" in prog
    assert "-y" in prog


def test_tt_inverse_invert_water_only_requires_seafloor_or_f():
    tomo = TomoAnd(bin_path="dummy_bin")
    with pytest.raises(ValueError, match="-y"):
        tomo._build_tt_inverse_program_args(
            "model.smesh", "obs.dat", {"invert_water_only": True}
        )


def test_tt_inverse_invert_water_only_with_f_only():
    tomo = TomoAnd(bin_path="dummy_bin")
    prog = tomo._build_tt_inverse_program_args(
        "model.smesh",
        "obs.dat",
        {"refl_file": "seafloor.dat", "invert_water_only": True},
    )
    assert prog is not None
    assert "-Fseafloor.dat" in prog
    assert "-y" in prog
    assert not any(a.startswith("-Y") for a in prog)


def test_tt_inverse_invert_crust_only_appends_w():
    tomo = TomoAnd(bin_path="dummy_bin")
    prog = tomo._build_tt_inverse_program_args(
        "model.smesh",
        "obs.dat",
        {"seafloor_file": "seafloor.dat", "invert_crust_only": True},
    )
    assert prog is not None
    assert "-Yseafloor.dat" in prog
    assert "-w" in prog
    assert "-y" not in prog


def test_tt_inverse_invert_crust_only_requires_seafloor_or_f():
    tomo = TomoAnd(bin_path="dummy_bin")
    with pytest.raises(ValueError, match="-w"):
        tomo._build_tt_inverse_program_args(
            "model.smesh", "obs.dat", {"invert_crust_only": True}
        )


def test_tt_inverse_invert_crust_only_with_f_only():
    tomo = TomoAnd(bin_path="dummy_bin")
    prog = tomo._build_tt_inverse_program_args(
        "model.smesh",
        "obs.dat",
        {"refl_file": "seafloor.dat", "invert_crust_only": True},
    )
    assert prog is not None
    assert "-Fseafloor.dat" in prog
    assert "-w" in prog
    assert not any(a.startswith("-Y") for a in prog)


def test_tt_inverse_water_and_crust_only_mutually_exclusive():
    tomo = TomoAnd(bin_path="dummy_bin")
    with pytest.raises(ValueError, match="mutually exclusive"):
        tomo._build_tt_inverse_program_args(
            "model.smesh",
            "obs.dat",
            {
                "seafloor_file": "seafloor.dat",
                "invert_water_only": True,
                "invert_crust_only": True,
            },
        )


def test_tt_inverse_freeze_refl_requires_f():
    tomo = TomoAnd(bin_path="dummy_bin")
    with pytest.raises(ValueError, match="-u"):
        tomo._build_tt_inverse_program_args(
            "model.smesh", "obs.dat", {"freeze_refl": True}
        )


def test_run_cmd_check_only_returns_help_text():
    tomo = TomoAnd(bin_path="dummy_bin")
    help_text = tomo._run_cmd("gen_smesh", check_only=True)
    assert isinstance(help_text, str)
    assert "gen_smesh" in help_text


def test_run_cmd_passes_run_env(monkeypatch, tmp_path):
    import subprocess

    tomo = TomoAnd(bin_path=str(tmp_path))
    tomo.capture_subprocess_output = True
    tomo.run_env = {"TOMO2D_INV_OMP": "1", "OMP_NUM_THREADS": "2"}
    # 可执行占位，避免真实查找失败：直接 mock resolve
    monkeypatch.setattr(tomo, "_resolve_executable", lambda name: str(tmp_path / "tt_inverse"))
    captured = {}

    def fake_run(cmd, **kw):
        captured["env"] = kw.get("env")
        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    tomo._run_cmd("tt_inverse", args=["-Mmesh"])
    assert captured["env"] is not None
    assert captured["env"]["TOMO2D_INV_OMP"] == "1"
    assert captured["env"]["OMP_NUM_THREADS"] == "2"


def test_run_cmd_streaming_lines(monkeypatch, tmp_path):
    import sys

    tomo = TomoAnd(bin_path=str(tmp_path))
    tomo.capture_subprocess_output = True
    lines: list[tuple[str, str]] = []
    tomo.stream_output_line = lambda s, t: lines.append((s, t))
    monkeypatch.setattr(tomo, "_resolve_executable", lambda name: sys.executable)
    # 用当前解释器打印两行，走 Popen 流式路径
    code = "import sys; print('hello-out'); print('hello-err', file=sys.stderr)"
    result = tomo._run_cmd("python", args=["-c", code])
    assert result.returncode == 0
    assert any(s == "stdout" and "hello-out" in t for s, t in lines)
    assert any(s == "stderr" and "hello-err" in t for s, t in lines)
    assert "hello-out" in (result.stdout or "")
