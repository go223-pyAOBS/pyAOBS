"""tomo2d GUI services 无 UI 单测。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyAOBS.modeling.tomo2d.gui.services.audits import (
    assert_existing_file,
    audit_path_under_work_dir,
)
from pyAOBS.modeling.tomo2d.gui.services.collectors import (
    collect_gen_dcorr_kwargs,
    collect_gen_smesh_kwargs,
    collect_gen_vcorr_kwargs,
    collect_tt_forward_args,
    collect_tt_inverse_args,
    format_tt_inverse_flag_summary,
    to_number,
)
from pyAOBS.modeling.tomo2d.gui.services.file_log import append_gui_file_log
from pyAOBS.modeling.tomo2d.gui.services.paths import (
    resolve_existing_file,
    resolve_work_dir,
    split_log_path_list,
    to_workdir_relative,
    validate_work_dir,
)
from pyAOBS.modeling.tomo2d.gui.services.preview import (
    format_resolved_cmdline,
    preview_append_resolved_cmdline,
)
from pyAOBS.modeling.tomo2d.gui.services.profile_io import (
    apply_tt_inverse_kwargs_updates,
    clear_tt_inverse_form_keys,
    json_to_entry_str,
    read_profile_json,
    sync_fwd_smesh_from_inv_if_missing,
)
from pyAOBS.modeling.tomo2d.gui.services.smesh_plot_core import (
    builtin_scale_p_cpt_path,
    builtin_smesh_cmap_path,
    colorbar_label_for_cmap,
    looks_like_grid_name,
    looks_like_interface,
    looks_like_model_name,
    looks_like_smesh,
    looks_like_smesh_name,
    looks_like_vin_name,
    load_model_plot_data,
    normalize_velocity_plot_dataset,
    normalize_dropped_path,
    resolve_inv_start_smesh,
    guess_smesh_initial_path,
    resolve_optional_refl_path,
    resolve_plot_smesh_cmap,
)
from pyAOBS.modeling.tomo2d.gui.services.workflow import (
    GEN_SMESH_OUT_REQUIRED,
    prepare_gen_smesh,
    prepare_tx_convert,
    preview_gen_smesh,
    preview_tx_convert,
    resolve_tx_convert_paths,
)
from pyAOBS.modeling.tomo2d.gui.services.workbench_state import (
    load_workbench_profile,
    save_workbench_profile,
)
from pyAOBS.modeling.tomo2d.gui.state.form_state import FormState
from pyAOBS.modeling.tomo2d.tomand import TomoAnd


pytestmark = pytest.mark.unit


def test_to_number_and_json_entry() -> None:
    assert to_number("") is None
    assert to_number("12") == 12
    assert to_number("1.5") == 1.5
    assert json_to_entry_str(None) == ""
    assert json_to_entry_str(True) == "1"
    assert json_to_entry_str(2.0) == "2"


def test_paths_relative_and_validate(tmp_path: Path) -> None:
    work = tmp_path / "proj"
    work.mkdir()
    nested = work / "in" / "a.dat"
    nested.parent.mkdir()
    nested.write_text("x", encoding="utf-8")

    r = to_workdir_relative(str(nested), work, warn_outside=False)
    assert r.value == "in/a.dat"
    assert validate_work_dir(str(work)) == work.resolve()
    with pytest.raises(ValueError):
        validate_work_dir(str(tmp_path / "missing"))


def test_split_log_path_list_real_newlines_and_jammed_drops() -> None:
    a = r"D:\proj\runs\r1\outputs\tt_inverse.log"
    b = r"D:\proj\runs\r2\outputs\tt_inverse.log"
    assert split_log_path_list(f"{a}\n{b}") == [a, b]
    jammed_url = f"file:///{a.replace(chr(92), '/')}file:///{b.replace(chr(92), '/')}"
    parts = split_log_path_list(jammed_url)
    assert len(parts) == 2
    assert parts[0].replace("\\", "/").endswith("r1/outputs/tt_inverse.log")
    assert parts[1].replace("\\", "/").endswith("r2/outputs/tt_inverse.log")
    jammed_drive = a + b
    assert split_log_path_list(jammed_drive) == [a, b]
    spaced = r"D:\my project\tt_inverse.log"
    assert split_log_path_list(spaced) == [spaced]


def test_resolve_existing_file_mnt_and_log_all(tmp_path: Path) -> None:
    import sys

    folder = tmp_path / "out.vpfd41.st1_65.SV80"
    folder.mkdir()
    log = folder / "log.all.vpfd41.w0.1"
    log.write_text("# dummy\n", encoding="utf-8")
    missing = folder / "log.all"
    hit = resolve_existing_file(str(missing), tmp_path)
    assert hit.resolve() == log.resolve()

    if not sys.platform.startswith("linux"):
        drive = tmp_path.drive[:1] or "D"
        rest = str(tmp_path.resolve()).replace("\\", "/")
        if len(rest) >= 2 and rest[1] == ":":
            rest = rest[2:].lstrip("/")
        mnt = f"/mnt/{drive.lower()}/{rest}/out.vpfd41.st1_65.SV80/log.all"
        hit2 = resolve_existing_file(mnt, tmp_path)
        assert hit2.resolve() == log.resolve()


def test_collect_gen_smesh_and_tt_forward() -> None:
    st = FormState(
        {
            "gen.vel_opt": "uniform",
            "gen.grid_opt": "uniform",
            "gen.v0": "1.5",
            "gen.gradient": "0.1",
            "gen.nx": "101",
            "gen.nz": "51",
            "gen.xmax": "100",
            "gen.zmax": "30",
            "gen.smesh_out": "out.smesh",
            "fwd.smesh": "m.smesh",
            "fwd.geom": "g.dat",
        }
    )
    kw = collect_gen_smesh_kwargs(st)
    assert kw["vel_opt"] == "uniform"
    assert kw["v0"] == 1.5
    assert kw["out_file"] == "out.smesh"
    smesh, geom, fkw = collect_tt_forward_args(st)
    assert smesh == "m.smesh" and geom == "g.dat"
    assert isinstance(fkw, dict)


def test_collect_gen_smesh_hang_sea_surface() -> None:
    st = FormState(
        {
            "gen.vel_opt": "zelt",
            "gen.grid_opt": "zelt",
            "gen.v_in": "v.in",
            "gen.ilayer": "2",
            "gen.dx": "0.5",
            "gen.z_file": "z.dat",
            "gen.hang_sea_surface": True,
            "gen.seafloor_out": "sf.dat",
            "gen.smesh_out": "out.smesh",
        }
    )
    kw = collect_gen_smesh_kwargs(st)
    assert kw["hang_sea_surface"] is True
    assert kw["seafloor_out"] == "sf.dat"
    assert kw["grid_opt"] == "zelt"


def test_collect_gen_smesh_hang_sea_off_by_default() -> None:
    st = FormState(
        {
            "gen.vel_opt": "zelt",
            "gen.v_in": "v.in",
            "gen.ilayer": "2",
            "gen.dx": "0.5",
            "gen.z_file": "z.dat",
            "gen.smesh_out": "out.smesh",
        }
    )
    kw = collect_gen_smesh_kwargs(st)
    assert "hang_sea_surface" not in kw
    assert "seafloor_out" not in kw


def test_collect_gen_smesh_seafloor_out_requires_hang() -> None:
    st = FormState(
        {
            "gen.vel_opt": "zelt",
            "gen.v_in": "v.in",
            "gen.ilayer": "2",
            "gen.dx": "0.5",
            "gen.z_file": "z.dat",
            "gen.seafloor_out": "sf.dat",
        }
    )
    with pytest.raises(ValueError, match="-S"):
        collect_gen_smesh_kwargs(st)


def test_audits_and_assert_file(tmp_path: Path) -> None:
    f = tmp_path / "a.dat"
    f.write_text("1", encoding="utf-8")
    assert_existing_file(str(f.name), "a", tmp_path)
    with pytest.raises(ValueError, match="不存在"):
        assert_existing_file("nope.dat", "a", tmp_path)
    line = audit_path_under_work_dir(tmp_path, "a", "a.dat")
    assert "a.dat" in line and ("存在" in line or "✓" in line)


def test_preview_cmdline_helpers() -> None:
    assert format_resolved_cmdline(["exe", "a b"]) 
    text = preview_append_resolved_cmdline("body", lambda: ["tomo", "-h"])
    assert "解析后命令行" in text and "tomo" in text


def test_profile_clear_and_sync_fwd(tmp_path: Path) -> None:
    st = FormState({"inv.mesh": "m.smesh", "fwd.smesh": ""})
    clear_tt_inverse_form_keys(st)
    assert st.get_str("inv.mesh") == ""
    mesh = tmp_path / "from_inv.smesh"
    mesh.write_text("x", encoding="utf-8")
    st2 = FormState(
        {
            "work_dir": str(tmp_path),
            "inv.mesh": "from_inv.smesh",
            "fwd.smesh": "",
        }
    )
    sync_fwd_smesh_from_inv_if_missing(st2)
    assert st2.get_str("fwd.smesh") == "from_inv.smesh"


def test_read_profile_json(tmp_path: Path) -> None:
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"work_dir": str(tmp_path), "gen.v0": "1.5"}), encoding="utf-8")
    data = read_profile_json(str(p))
    assert data["gen.v0"] == "1.5"


def test_file_log_and_workbench_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    append_gui_file_log(tmp_path, "hello", enabled=True)
    log = tmp_path / "tomo2d_gui.log"
    assert log.is_file() and "hello" in log.read_text(encoding="utf-8")
    append_gui_file_log(tmp_path, "skip", enabled=False)
    assert "skip" not in log.read_text(encoding="utf-8")

    state_file = tmp_path / "gui_state.json"
    monkeypatch.setenv("PYAOBS_GUI_STATE_FILE", str(state_file))
    save_workbench_profile({"work_dir": "C:/proj"})
    assert load_workbench_profile()["work_dir"] == "C:/proj"


def test_smesh_cmap_and_refl(tmp_path: Path) -> None:
    builtin = builtin_scale_p_cpt_path()
    vs_p = builtin_smesh_cmap_path("vs")
    vpvs_p = builtin_smesh_cmap_path("vpvs")
    water_p = builtin_smesh_cmap_path("water")
    assert builtin.is_file() and vs_p.is_file() and vpvs_p.is_file() and water_p.is_file()
    st = FormState({"gui.plot_smesh_cmap": "", "fwd.refl_file": ""})
    assert resolve_plot_smesh_cmap(st, tmp_path) == str(builtin)
    st.set("gui.plot_smesh_cmap", "vp")
    assert resolve_plot_smesh_cmap(st, tmp_path) == str(builtin)
    st.set("gui.plot_smesh_cmap", "vs")
    assert resolve_plot_smesh_cmap(st, tmp_path) == str(vs_p)
    st.set("gui.plot_smesh_cmap", "vpvs")
    assert resolve_plot_smesh_cmap(st, tmp_path) == str(vpvs_p)
    st.set("gui.plot_smesh_cmap", "water")
    assert resolve_plot_smesh_cmap(st, tmp_path) == str(water_p)
    st.set("gui.plot_smesh_cmap", "scale_water.cpt")
    assert resolve_plot_smesh_cmap(st, tmp_path) == str(water_p)
    st.set("gui.plot_smesh_cmap", "inputs/scale_s.cpt")
    assert resolve_plot_smesh_cmap(st, tmp_path) == str(vs_p)
    missing = []
    st.set("gui.plot_smesh_cmap", "no.cpt")
    assert (
        resolve_plot_smesh_cmap(st, tmp_path, on_missing_cpt=missing.append)
        == str(builtin)
    )
    assert missing
    st.set("gui.plot_smesh_cmap", "seismic")
    assert resolve_plot_smesh_cmap(st, tmp_path) == "seismic"
    assert resolve_optional_refl_path(st, tmp_path) is None


def test_builtin_vpvs_cpt_haiti_vpvs1() -> None:
    text = builtin_smesh_cmap_path("vpvs").read_text(encoding="utf-8")
    first = next(
        ln.split()
        for ln in text.splitlines()
        if ln.strip() and not ln.startswith("#") and ln[0] not in "BFN"
    )
    assert first[:4] == ["1.65", "0", "0", "127"]
    last = [
        ln.split()
        for ln in text.splitlines()
        if ln.strip() and not ln.startswith("#") and ln[0] not in "BFN"
    ][-1]
    assert last[-4:] == ["2", "127", "0", "0"]


def test_builtin_water_cpt_range() -> None:
    from pyAOBS.visualization.gmt_cpt import parse_gmt_cpt_for_matplotlib

    p = builtin_smesh_cmap_path("water")
    _cmap, lo, hi = parse_gmt_cpt_for_matplotlib(str(p))
    assert lo == pytest.approx(1.35)
    assert hi == pytest.approx(1.65)
    first = next(
        ln.split()
        for ln in p.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#") and ln[0] not in "BFN"
    )
    last = [
        ln.split()
        for ln in p.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#") and ln[0] not in "BFN"
    ][-1]
    # 浅层/低速=暖色，深层/高速=冷色
    assert float(first[1]) > float(first[3])
    assert float(last[-1]) > float(last[-3])
    rgba_hi = _cmap(1.0)
    rgba_over = _cmap(1.5)
    # 沉积 >1.65 用 F 棕，不要裁成深蓝
    assert rgba_over[0] > rgba_over[2]
    assert rgba_over[0] > rgba_hi[0]


def test_mask_air_layer_for_plot() -> None:
    import numpy as np

    from pyAOBS.modeling.tomo2d.gui.services.smesh_plot_core import (
        mask_air_layer_for_plot,
    )

    x = np.array([0.0, 1.0])
    z = np.array([-0.2, 0.0, 1.0])
    data = np.array([[0.33, 0.33], [1.46, 1.46], [1.54, 1.54]])
    out = mask_air_layer_for_plot(data, x, z, mesh=None)
    assert np.isnan(out[0, 0]) and np.isnan(out[0, 1])
    assert out[1, 0] == pytest.approx(1.46)
    assert out[2, 1] == pytest.approx(1.54)


def test_water_smesh_sea_surface_not_masked_as_air() -> None:
    """挂海面的水网格：z≈0 是水速，不能被 arange 的 -2e-16 涂成空气白。"""
    import numpy as np
    from pathlib import Path

    from pyAOBS.modeling.tomo2d.gui.services.smesh_plot_core import (
        load_smesh_plot_data,
        mask_air_layer_for_plot,
        normalize_velocity_plot_dataset,
    )

    smesh = (
        Path(__file__).resolve().parent.parent
        / "modeling"
        / "tomo2d"
        / "example_water"
        / "water_fwd"
        / "water.smesh"
    )
    if not smesh.is_file():
        pytest.skip("water.smesh 不在工区")
    mesh, ds, _extra = load_smesh_plot_data(smesh, None, with_xarray=True)
    ds = normalize_velocity_plot_dataset(ds)
    data = np.asarray(ds["velocity"].values, dtype=float)
    x = np.asarray(ds["x"].values, dtype=float)
    z = np.asarray(ds["z"].values, dtype=float)
    i0 = int(np.argmin(np.abs(z)))
    assert abs(float(z[i0])) < 1e-9
    assert float(data[i0, data.shape[1] // 2]) > 1.2
    out = mask_air_layer_for_plot(data, x, z, mesh)
    assert np.isfinite(out[i0]).all()
    assert float(np.nanmin(z[np.isfinite(out).any(axis=1)])) == pytest.approx(0.0)


def test_air_axis_zlim_leaves_blank_above_sea() -> None:
    from pyAOBS.modeling.tomo2d.gui.services.smesh_plot_core import air_axis_zlim

    lo, hi = air_axis_zlim(0.0, 4.0)
    assert lo == pytest.approx(4.0)
    assert hi < 0.0


def test_builtin_vs_cpt_white_at_1p5() -> None:
    text = builtin_smesh_cmap_path("vs").read_text(encoding="utf-8")
    first = next(
        ln.split()
        for ln in text.splitlines()
        if ln.strip() and not ln.startswith("#")
    )
    assert first[:4] == ["1.5", "255", "255", "255"]


def test_vpvs_cmap_limits_and_label() -> None:
    from pyAOBS.modeling.tomo2d.gui.plots.inv_monitor_model import _mpl_cmap
    from pyAOBS.visualization.gmt_cpt import parse_gmt_cpt_for_matplotlib

    path = builtin_smesh_cmap_path("vpvs")
    _, zmin, zmax = parse_gmt_cpt_for_matplotlib(str(path))
    assert abs(zmin - 1.65) < 1e-9
    assert abs(zmax - 2.0) < 1e-9
    cmap, lv = _mpl_cmap(str(path))
    assert cmap is not None and lv is not None
    assert abs(lv[0] - 1.65) < 1e-9
    assert abs(lv[1] - 2.0) < 1e-9
    assert colorbar_label_for_cmap(str(path)) == "Vp/Vs"
    assert colorbar_label_for_cmap("vp") == "km/s"


def test_mpl_nav_skips_colorbar_axes() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure

    from pyAOBS.utils.mpl_plot_nav import PyqtgraphStyleNav, is_colorbar_axes

    fig = Figure()
    ax = fig.add_subplot(1, 2, 1)
    cax = fig.add_subplot(1, 2, 2)
    sm = ScalarMappable(norm=Normalize(1.5, 8.7), cmap="gray")
    sm.set_array([])
    fig.colorbar(sm, cax=cax)
    assert is_colorbar_axes(cax)
    assert not is_colorbar_axes(ax)

    class _Canvas:
        figure = fig

        def mpl_connect(self, *_a, **_k):
            return 0

        def draw_idle(self) -> None:
            pass

    nav = PyqtgraphStyleNav(_Canvas(), auto_home_on_draw=False)
    ax.set_xlim(0, 10)
    ax.set_ylim(5, 0)
    cax.set_ylim(1.5, 8.7)
    views = nav.capture_views()
    assert len(views) == 1
    assert views[0][0] == (0.0, 10.0)
    cax.set_ylim(1.65, 2.0)
    nav._saved_views = views
    nav._user_view_active = True
    assert nav.restore_saved_views()
    lo, hi = cax.get_ylim()
    assert abs(lo - 1.65) < 1e-6
    assert abs(hi - 2.0) < 1e-6


def test_colorbar_limits_reset_after_cmap_switch() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    from pyAOBS.modeling.tomo2d.gui.plots.inv_monitor_model import (
        _apply_colorbar_limits,
    )

    fig = Figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    cax = fig.add_axes([0.85, 0.15, 0.04, 0.7])
    sm = ScalarMappable(norm=Normalize(1.5, 8.7), cmap="gray")
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    _apply_colorbar_limits(cb, cax, 1.55, 2.2, cb_label="km/s")
    lo, hi = cax.get_ylim()
    assert abs(lo - 1.55) < 1e-6
    assert abs(hi - 2.2) < 1e-6
    ticks = list(cb.get_ticks())
    assert abs(ticks[0] - 1.55) < 1e-6
    assert abs(ticks[-1] - 2.2) < 1e-6


def test_decimal_log_tick_labels() -> None:
    from pyAOBS.modeling.tomo2d.tt_inverse_log_analysis import (
        build_figure_multi_param_influence,
        decimal_log_tick_values,
        format_decimal_log_tick,
    )

    assert format_decimal_log_tick(600) == "600"
    assert format_decimal_log_tick(6e2) == "600"
    assert format_decimal_log_tick(150) == "150"
    assert format_decimal_log_tick(0.001) == "0.001"
    assert format_decimal_log_tick(1) == "1"
    ticks = decimal_log_tick_values(140.0, 160.0, [150.0])
    assert any(abs(float(t) - 150.0) < 1e-9 for t in ticks)
    ticks = decimal_log_tick_values(100.0, 200.0, [150.0])
    assert any(abs(float(t) - 150.0) < 1e-9 for t in ticks)

    def row(w_sv: float):
        r = [0.0] * 26
        r[0] = 5
        r[1] = 1
        r[13] = w_sv
        r[14] = 1.0
        r[15] = 0.01
        r[16] = 0.01
        r[20] = 2.0
        r[23] = 1.0
        return [r]

    fig = build_figure_multi_param_influence({"sv150": row(150.0)})
    fig.canvas.draw()
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels() if t.get_text()]
    assert "150" in labels
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_decimal_log_axis_keeps_150_label() -> None:
    import os

    import numpy as np
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from pyAOBS.modeling.tomo2d.gui.plots.inv_analysis_pg import DecimalLogAxis

    _app = QApplication.instance() or QApplication([])
    ax = DecimalLogAxis(orientation="bottom")
    ax.logMode = True
    ax.set_linear_data([150.0])
    ticks = ax.logTickValues(np.log10(140.0), np.log10(160.0), 400.0, [])
    vals = [float(v) for _sp, vs in ticks for v in vs]
    assert any(abs(10.0 ** v - 150.0) < 1e-6 for v in vals)
    assert "150" in ax.logTickStrings(vals, 1.0, 1.0)


def test_looks_like_smesh(tmp_path: Path) -> None:
    a = tmp_path / "model.smesh"
    b = tmp_path / "out.smesh.5.1"
    c = tmp_path / "scale_p.cpt"
    a.write_text("x", encoding="utf-8")
    b.write_text("x", encoding="utf-8")
    c.write_text("x", encoding="utf-8")
    assert looks_like_smesh(a)
    assert looks_like_smesh(b)
    assert not looks_like_smesh(c)
    assert not looks_like_smesh(tmp_path / "missing.smesh")
    assert looks_like_smesh_name(tmp_path / "missing.smesh")
    assert looks_like_smesh_name("out.smesh.5.1")
    assert not looks_like_smesh_name("scale_p.cpt")


def test_looks_like_interface(tmp_path: Path) -> None:
    a = tmp_path / "out.refl.2.1"
    b = tmp_path / "refl_init.dat"
    c = tmp_path / "bathy.dat"
    d = tmp_path / "model.smesh"
    e = tmp_path / "ttimes.dat"
    for p in (a, b, c, d, e):
        p.write_text("0 0\n1 1\n", encoding="utf-8")
    assert looks_like_interface(a)
    assert looks_like_interface(b)
    assert looks_like_interface(c)
    assert not looks_like_interface(d)
    assert not looks_like_interface(e)


def test_looks_like_vin_and_grid() -> None:
    assert looks_like_vin_name("v.in")
    assert looks_like_vin_name("model.vin")
    assert looks_like_vin_name("vpvs.in")
    assert not looks_like_vin_name("tx.in")
    assert not looks_like_vin_name("r.in")
    assert looks_like_grid_name("vpvs.grd")
    assert looks_like_grid_name("vs.nc")
    assert looks_like_model_name("out.smesh.1.1")
    assert looks_like_model_name("v.in")
    assert looks_like_model_name("a.grd")
    assert not looks_like_model_name("scale_p.cpt")


def test_load_model_plot_data_grd_and_vin(tmp_path: Path) -> None:
    import numpy as np
    import xarray as xr

    x = np.linspace(0.0, 10.0, 6)
    z = np.linspace(0.0, 4.0, 5)
    vel = 1.7 + 0.05 * z[:, None] + 0.0 * x[None, :]
    ds0 = xr.Dataset(
        data_vars={"vpvs": (("z", "x"), vel)},
        coords={"x": x, "z": z},
    )
    grd = tmp_path / "vpvs.nc"
    ds0.to_netcdf(grd)
    mesh, ds, extra = load_model_plot_data(grd)
    assert mesh is None
    assert extra is None
    assert ds["velocity"].shape == (5, 6)
    assert float(ds["velocity"].min()) < 2.0

    norm = normalize_velocity_plot_dataset(ds0)
    assert "velocity" in norm.data_vars

    geo = xr.Dataset(
        data_vars={"z": (("lat", "lon"), np.ones((3, 4)))},
        coords={"lon": [10.0, 11.0, 12.0, 13.0], "lat": [18.0, 19.0, 20.0]},
    )
    geo_p = tmp_path / "gravity.nc"
    geo.to_netcdf(geo_p)
    with pytest.raises(ValueError, match="经纬度"):
        load_model_plot_data(geo_p)


def test_normalize_dropped_path_wsl_and_windows() -> None:
    import sys

    p = normalize_dropped_path(r"C:\data\out.smesh.1.1")
    if sys.platform.startswith("linux"):
        assert str(p).replace("\\", "/") == "/mnt/c/data/out.smesh.1.1"
    else:
        assert p.name == "out.smesh.1.1"

    mnt = normalize_dropped_path("/mnt/d/work/model.smesh")
    if sys.platform.startswith("linux"):
        assert str(mnt) == "/mnt/d/work/model.smesh"
    else:
        assert str(mnt).replace("\\", "/").lower().startswith("d:/work/model.smesh".lower())

    wsl_url = "file://wsl.localhost/Ubuntu/home/user/model.smesh"
    u = normalize_dropped_path(wsl_url)
    if sys.platform.startswith("linux"):
        assert str(u) == "/home/user/model.smesh"
    else:
        s = str(u).replace("/", "\\").lower()
        assert "wsl.localhost" in s and s.endswith("home\\user\\model.smesh")


def test_parse_clipboard_path_text() -> None:
    from pyAOBS.modeling.tomo2d.gui.services.smesh_plot_core import (
        parse_clipboard_path_text,
    )

    rows = parse_clipboard_path_text(r'"C:\data\out.smesh.1.1"')
    assert len(rows) == 1
    assert rows[0].name == "out.smesh.1.1"
    rows = parse_clipboard_path_text("a.smesh\nb.refl.1.1\n")
    assert [p.name for p in rows] == ["a.smesh", "b.refl.1.1"]
    assert parse_clipboard_path_text("   \n") == []


def test_pick_smesh_paths_keeps_order_and_dedupes() -> None:
    from pyAOBS.modeling.tomo2d.gui.services.smesh_plot_core import pick_smesh_paths

    rows = pick_smesh_paths(
        [
            Path("a.smesh.1.0"),
            Path("notes.txt"),
            Path("b.smesh"),
            Path("a.smesh.1.0"),
        ]
    )
    assert [p.name for p in rows] == ["a.smesh.1.0", "b.smesh"]


def test_resolve_inv_start_smesh_prefers_inv_mesh(tmp_path: Path) -> None:
    inv = tmp_path / "start.smesh"
    fwd = tmp_path / "fwd.smesh"
    inv.write_text("i", encoding="utf-8")
    fwd.write_text("f", encoding="utf-8")
    st = FormState({"inv.mesh": "start.smesh", "fwd.smesh": "fwd.smesh"})
    assert resolve_inv_start_smesh(st, tmp_path) == inv
    st.set("inv.mesh", "missing.smesh")
    assert resolve_inv_start_smesh(st, tmp_path) == fwd
    st.set("fwd.smesh", "")
    assert resolve_inv_start_smesh(st, tmp_path) is None


def test_guess_smesh_initial_path_uses_current_tab(tmp_path: Path) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.smesh_plot_core import (
        lookup_plot_sources_for_tab,
    )

    fwd = tmp_path / "model.smesh"
    gen = tmp_path / "new.smesh"
    inv = tmp_path / "start.smesh"
    fwd.write_text("f", encoding="utf-8")
    gen.write_text("g", encoding="utf-8")
    inv.write_text("i", encoding="utf-8")
    st = FormState(
        {
            "fwd.smesh": "model.smesh",
            "gen.smesh_out": "new.smesh",
            "inv.mesh": "start.smesh",
        }
    )
    assert guess_smesh_initial_path(st, tmp_path) is None
    assert guess_smesh_initial_path(st, tmp_path, tab_id="tt_forward") == fwd
    assert guess_smesh_initial_path(st, tmp_path, tab_id="gen_smesh") == gen
    assert guess_smesh_initial_path(st, tmp_path, tab_id="tt_inverse") == inv
    st.set("gen.smesh_out", "missing.smesh")
    hit = lookup_plot_sources_for_tab(st, tmp_path, "gen_smesh")
    assert hit.smesh_path is None
    assert hit.smesh_missing is not None
    assert "smesh 输出文件" in hit.missing_smesh_message()
    damp = lookup_plot_sources_for_tab(st, tmp_path, "gen_damp")
    assert damp.smesh_key is None
    assert "没有 smesh" in damp.missing_smesh_message()
    # 不跨页签用 fwd.smesh
    assert guess_smesh_initial_path(st, tmp_path, tab_id="gen_smesh") is None
    from pyAOBS.modeling.tomo2d.gui.services.smesh_plot_core import CMD_TAB_PLOT_SOURCES

    assert [s.tab_id for s in CMD_TAB_PLOT_SOURCES] == [
        "gen_smesh",
        "tt_forward",
        "gen_damp",
        "gen_vcorr",
        "gen_dcorr",
        "tt_inverse",
        "stat_smesh",
        "edit_smesh",
        "pipeline",
        "tx_convert",
        "checkerboard",
        "monte_carlo",
        "wave2d",
    ]


def test_lookup_plot_refl_stays_on_current_tab(tmp_path: Path) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.smesh_plot_core import (
        lookup_plot_sources_for_tab,
    )

    mesh = tmp_path / "inv.smesh"
    inv_r = tmp_path / "inv.refl"
    fwd_r = tmp_path / "fwd.refl"
    mesh.write_text("m", encoding="utf-8")
    inv_r.write_text("0 0\n1 1\n", encoding="utf-8")
    fwd_r.write_text("0 0\n1 1\n", encoding="utf-8")
    st = FormState(
        {
            "inv.mesh": "inv.smesh",
            "inv.refl_file": "inv.refl",
            "fwd.smesh": "inv.smesh",
            "fwd.refl_file": "fwd.refl",
        }
    )
    inv_hit = lookup_plot_sources_for_tab(st, tmp_path, "tt_inverse")
    assert inv_hit.smesh_path == mesh
    assert inv_hit.refl_path == inv_r
    fwd_hit = lookup_plot_sources_for_tab(st, tmp_path, "tt_forward")
    assert fwd_hit.refl_path == fwd_r
    st.set("inv.refl_file", "gone.refl")
    miss = lookup_plot_sources_for_tab(st, tmp_path, "tt_inverse")
    assert miss.refl_path is None
    msg = miss.missing_refl_message()
    assert msg is not None and "refl_file" in msg
    # 空字段：不提示、也不借用正演页界面
    st.set("inv.refl_file", "")
    empty = lookup_plot_sources_for_tab(st, tmp_path, "tt_inverse")
    assert empty.refl_path is None
    assert empty.missing_refl_message() is None


def test_workflow_preview_and_prepare_gen(tmp_path: Path) -> None:
    st = FormState(
        {
            "gen.vel_opt": "uniform",
            "gen.grid_opt": "uniform",
            "gen.v0": "1.5",
            "gen.gradient": "0.1",
            "gen.nx": "10",
            "gen.nz": "5",
            "gen.xmax": "1",
            "gen.zmax": "1",
        }
    )
    tomo = TomoAnd(bin_path="dummy_bin")
    text = preview_gen_smesh(st, tmp_path, tomo)
    assert "gen_smesh" in text
    assert ("未传" in text) or ("out_file" in text.lower())
    with pytest.raises(ValueError, match="输出|smesh"):
        prepare_gen_smesh(st, tmp_path, tomo)
    st.set("gen.smesh_out", "o.smesh")
    prep = prepare_gen_smesh(st, tmp_path, tomo)
    assert prep.title == "gen_smesh" and callable(prep.job)


def test_tx_convert_paths(tmp_path: Path) -> None:
    st = FormState(
        {
            "tx.station_lis": "station.lis",
            "tx.tx_in": "tx.in",
            "tx.data_out": "out/tt.dat",
            "tx.geom_out": "out/g.dat",
        }
    )
    s, t, d, g = resolve_tx_convert_paths(st, tmp_path)
    assert s.name == "station.lis"
    assert d.parent.name == "out"
    assert resolve_work_dir(str(tmp_path)) == tmp_path.resolve()


def test_tx_convert_obs_filter_preview_and_prepare(tmp_path: Path) -> None:
    (tmp_path / "station.lis").write_text("1 1.0 0.0\n2 2.0 0.0\n", encoding="utf-8")
    (tmp_path / "tx.in").write_text(
        "1.0 0 0 0\n1.1 0.5 0.05 1\n2.0 0 0 0\n2.1 0.7 0.05 1\n0 0 0 -1\n",
        encoding="utf-8",
    )
    st = FormState(
        {
            "tx.station_lis": "station.lis",
            "tx.tx_in": "tx.in",
            "tx.data_out": "ttimes.dat",
            "tx.geom_out": "geom.dat",
            "tx.obs_ids": "2",
        }
    )
    text = preview_tx_convert(st, tmp_path)
    assert "include_obs" in text
    assert "选用 OBS: 2" in text
    prep = prepare_tx_convert(st, tmp_path)
    stats = prep.job()
    assert stats.nshot == 1
    st.set("tx.obs_ids", "none")
    with pytest.raises(ValueError, match="未选择"):
        prepare_tx_convert(st, tmp_path)


def test_tx_convert_water_phases_opt_in(tmp_path: Path) -> None:
    (tmp_path / "station.lis").write_text("1 1.0 0.0\n", encoding="utf-8")
    (tmp_path / "tx.in").write_text(
        "1.0 0 0 0\n0.5 1.2 0.05 5\n0 0 0 -1\n",
        encoding="utf-8",
    )
    st = FormState(
        {
            "tx.station_lis": "station.lis",
            "tx.tx_in": "tx.in",
            "tx.data_out": "ttimes.dat",
            "tx.geom_out": "geom.dat",
            "tx.refr_phases": "",
            "tx.water_phases": "5",
        }
    )
    text = preview_tx_convert(st, tmp_path)
    assert "water_phases" in text
    assert "直达水波震相→2" in text
    prep = prepare_tx_convert(st, tmp_path)
    prep.job()
    r_lines = [
        ln
        for ln in (tmp_path / "ttimes.dat").read_text(encoding="utf-8").splitlines()
        if ln.startswith("r")
    ]
    assert len(r_lines) == 1
    assert int(r_lines[0][21:26]) == 2


def test_tx_convert_recv_peg_phases_opt_in(tmp_path: Path) -> None:
    (tmp_path / "station.lis").write_text("1 1.0 0.0\n", encoding="utf-8")
    (tmp_path / "tx.in").write_text(
        "1.0 0 0 0\n0.5 1.2 0.05 7\n0.6 2.2 0.05 8\n0 0 0 -1\n",
        encoding="utf-8",
    )
    st = FormState(
        {
            "tx.station_lis": "station.lis",
            "tx.tx_in": "tx.in",
            "tx.data_out": "ttimes.dat",
            "tx.geom_out": "geom.dat",
            "tx.refr_phases": "",
            "tx.refr_mult_phases": "7",
            "tx.refl_mult_phases": "8",
        }
    )
    text = preview_tx_convert(st, tmp_path)
    assert "refr_mult_phases" in text
    assert "refl_mult_phases" in text
    assert "折射台侧多次震相→4" in text
    assert "反射台侧多次震相→5" in text
    prep = prepare_tx_convert(st, tmp_path)
    prep.job()
    codes = [
        int(ln[21:26])
        for ln in (tmp_path / "ttimes.dat").read_text(encoding="utf-8").splitlines()
        if ln.startswith("r")
    ]
    assert codes == [4, 5]


def test_collect_tt_inverse_freeze_refl() -> None:
    st = FormState(
        {
            "inv.mesh": "m.smesh",
            "inv.data": "d.dat",
            "inv.refl_file": "seafloor.dat",
            "inv.freeze_refl": True,
        }
    )
    _, _, kw = collect_tt_inverse_args(st)
    assert kw["freeze_refl"] is True
    assert kw["refl_file"] == "seafloor.dat"


def test_collect_tt_inverse_freeze_requires_f() -> None:
    st = FormState(
        {
            "inv.mesh": "m.smesh",
            "inv.data": "d.dat",
            "inv.freeze_refl": True,
        }
    )
    with pytest.raises(ValueError, match="-u"):
        collect_tt_inverse_args(st)


def test_collect_tt_inverse_freeze_off_by_default() -> None:
    st = FormState({"inv.mesh": "m.smesh", "inv.data": "d.dat", "inv.refl_file": "moho.dat"})
    _, _, kw = collect_tt_inverse_args(st)
    assert "freeze_refl" not in kw


def test_collect_tt_inverse_seafloor_flags_off_by_default() -> None:
    """0/1 默认：不传 -Y/-y/-w，核与原来相同。"""
    st = FormState({"inv.mesh": "m.smesh", "inv.data": "d.dat"})
    _, _, kw = collect_tt_inverse_args(st)
    assert "seafloor_file" not in kw
    assert "invert_water_only" not in kw
    assert "invert_crust_only" not in kw


def test_collect_tt_forward_seafloor_file() -> None:
    st = FormState(
        {
            "fwd.smesh": "m.smesh",
            "fwd.geom": "g.dat",
            "fwd.refl_file": "moho.dat",
            "fwd.seafloor_file": "sf.dat",
        }
    )
    _, _, kw = collect_tt_forward_args(st)
    assert kw["refl_file"] == "moho.dat"
    assert kw["seafloor_file"] == "sf.dat"


def test_collect_tt_forward_seafloor_off_by_default() -> None:
    st = FormState({"fwd.smesh": "m.smesh", "fwd.geom": "g.dat"})
    _, _, kw = collect_tt_forward_args(st)
    assert "seafloor_file" not in kw
    assert "refl_file" not in kw
    assert "conv_file" not in kw


def test_collect_tt_forward_conv_file() -> None:
    st = FormState(
        {
            "fwd.smesh": "m.smesh",
            "fwd.geom": "g.dat",
            "fwd.conv_file": "conv.dat",
        }
    )
    _, _, kw = collect_tt_forward_args(st)
    assert kw["conv_file"] == "conv.dat"


def test_collect_tt_forward_vsmesh() -> None:
    st = FormState(
        {
            "fwd.smesh": "m.smesh",
            "fwd.geom": "g.dat",
            "fwd.vsmesh": "vs.smesh",
        }
    )
    _, _, kw = collect_tt_forward_args(st)
    assert kw["vsmesh"] == "vs.smesh"


def test_collect_tt_inverse_conv_file() -> None:
    st = FormState(
        {
            "inv.mesh": "m.smesh",
            "inv.data": "obs.dat",
            "inv.conv_file": "conv.dat",
        }
    )
    _, _, kw = collect_tt_inverse_args(st)
    assert kw["conv_file"] == "conv.dat"


def test_collect_tt_inverse_vsmesh() -> None:
    st = FormState(
        {
            "inv.mesh": "m.smesh",
            "inv.data": "obs.dat",
            "inv.vsmesh": "vs.smesh",
        }
    )
    _, _, kw = collect_tt_inverse_args(st)
    assert kw["vsmesh"] == "vs.smesh"


def test_collect_tt_inverse_seafloor_file() -> None:
    st = FormState(
        {
            "inv.mesh": "m.smesh",
            "inv.data": "d.dat",
            "inv.seafloor_file": "sf.dat",
        }
    )
    _, _, kw = collect_tt_inverse_args(st)
    assert kw["seafloor_file"] == "sf.dat"
    assert "invert_water_only" not in kw
    assert "invert_crust_only" not in kw


def test_collect_tt_inverse_invert_water_only() -> None:
    st = FormState(
        {
            "inv.mesh": "m.smesh",
            "inv.data": "d.dat",
            "inv.seafloor_file": "sf.dat",
            "inv.invert_water_only": True,
        }
    )
    _, _, kw = collect_tt_inverse_args(st)
    assert kw["invert_water_only"] is True
    assert kw["seafloor_file"] == "sf.dat"


def test_collect_tt_inverse_invert_crust_only() -> None:
    st = FormState(
        {
            "inv.mesh": "m.smesh",
            "inv.data": "d.dat",
            "inv.seafloor_file": "sf.dat",
            "inv.invert_crust_only": True,
        }
    )
    _, _, kw = collect_tt_inverse_args(st)
    assert kw["invert_crust_only"] is True
    assert "invert_water_only" not in kw


def test_collect_tt_inverse_water_crust_mutex() -> None:
    st = FormState(
        {
            "inv.mesh": "m.smesh",
            "inv.data": "d.dat",
            "inv.seafloor_file": "sf.dat",
            "inv.invert_water_only": True,
            "inv.invert_crust_only": True,
        }
    )
    with pytest.raises(ValueError, match="互斥"):
        collect_tt_inverse_args(st)


def test_collect_tt_inverse_invert_water_requires_iface() -> None:
    st = FormState(
        {
            "inv.mesh": "m.smesh",
            "inv.data": "d.dat",
            "inv.invert_water_only": True,
        }
    )
    with pytest.raises(ValueError, match="-Y"):
        collect_tt_inverse_args(st)


def test_collect_tt_inverse_invert_crust_requires_iface() -> None:
    st = FormState(
        {
            "inv.mesh": "m.smesh",
            "inv.data": "d.dat",
            "inv.invert_crust_only": True,
        }
    )
    with pytest.raises(ValueError, match="-Y"):
        collect_tt_inverse_args(st)


def test_collect_tt_inverse_invert_water_with_f_only() -> None:
    st = FormState(
        {
            "inv.mesh": "m.smesh",
            "inv.data": "d.dat",
            "inv.refl_file": "moho.dat",
            "inv.invert_water_only": True,
        }
    )
    _, _, kw = collect_tt_inverse_args(st)
    assert kw["invert_water_only"] is True
    assert kw["refl_file"] == "moho.dat"
    assert "seafloor_file" not in kw


def test_apply_tt_inverse_kwargs_replay_seafloor(tmp_path: Path) -> None:
    u = apply_tt_inverse_kwargs_updates(
        {"seafloor_file": "inputs/sf.dat", "invert_crust_only": True},
        tmp_path,
    )
    assert u["inv.invert_crust_only"] is True
    assert "invert_water_only" not in u
    assert str(u["inv.seafloor_file"]).replace("\\", "/").endswith("inputs/sf.dat")


def test_clear_tt_inverse_form_keys_clears_seafloor() -> None:
    st = FormState(
        {
            "inv.mesh": "m.smesh",
            "inv.seafloor_file": "sf.dat",
            "inv.invert_water_only": True,
            "inv.use_repro_bundle": True,
        }
    )
    clear_tt_inverse_form_keys(st)
    assert st.get_str("inv.seafloor_file") == ""
    assert st.get_bool("inv.invert_water_only") is False
    assert st.get_bool("inv.use_repro_bundle") is True


def test_gen_smesh_out_required_message() -> None:
    assert "smesh" in GEN_SMESH_OUT_REQUIRED.lower() or "输出" in GEN_SMESH_OUT_REQUIRED


def test_workbench_last_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.workbench_state import (
        load_last_project_path,
        save_workbench_profile,
    )

    state_file = tmp_path / "gui_state.json"
    monkeypatch.setenv("PYAOBS_GUI_STATE_FILE", str(state_file))
    save_workbench_profile({"gen.v0": "1"}, last_project=str(tmp_path / "meta" / "tomo2d_project.json"))
    assert "tomo2d_project.json" in load_last_project_path()


def test_collect_run_env_bools_and_threads() -> None:
    from pyAOBS.modeling.tomo2d.gui.services.run_env import (
        collect_run_env,
        format_run_env_preview,
        merge_subprocess_env,
    )

    st = FormState(
        {
            "env.omp_num_threads": "4",
            "env.inv_omp": True,
            "env.fwd_omp": False,
            "env.inv_legacy_baseline": False,
            "env.inv_reuse_forward": True,
            "env.inv_reuse_thresh": "1e-3",
            "env.inv_coarse2fine": False,
            "env.inv_lsqr_precond": False,
            "env.inv_diag": False,
        }
    )
    env = collect_run_env(st)
    assert env["OMP_NUM_THREADS"] == "4"
    assert env["TOMO2D_INV_OMP"] == "1"
    assert env["TOMO2D_FWD_OMP"] == "0"
    assert env["TOMO2D_INV_REUSE_FORWARD"] == "1"
    assert env["TOMO2D_INV_REUSE_THRESH"] == "1e-3"
    assert env["TOMO2D_INV_LSQR_PRECOND"] == "0"
    assert "TOMO2D_INV_LSQR_PRECOND_MAX" not in env
    assert env["TOMO2D_INV_SENS_WEIGHT"] == "0"
    assert env["TOMO2D_INV_LINESEARCH"] == "0"
    assert env["TOMO2D_INV_LM"] == "0"
    assert env["TOMO2D_GRAPH_FS_ENUM"] == "1"
    assert env["TOMO2D_INV_STATUS_JSONL"] == "outputs/status.jsonl"
    assert "TOMO2D_INV_C2F_SMOOTH_START" not in env
    preview = format_run_env_preview(env)
    assert "TOMO2D_INV_OMP=1" in preview
    from pyAOBS.modeling.tomo2d.gui.services.run_env import format_parallel_status_line

    status = format_parallel_status_line(env)
    assert "OMP_NUM_THREADS=4" in status
    assert "tt_inverse并行=开" in status
    assert "tt_forward并行=关" in status
    assert "前向复用" in status
    assert "图论FS" in status
    assert "列预条件" not in status
    merged = merge_subprocess_env(env)
    assert merged is not None
    assert merged["OMP_NUM_THREADS"] == "4"


def test_collect_run_env_empty_threads_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.run_env import collect_run_env

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    st = FormState({"env.omp_num_threads": "", "env.inv_omp": False})
    env = collect_run_env(st)
    assert "OMP_NUM_THREADS" not in env
    assert env["TOMO2D_INV_OMP"] == "0"


def test_collect_run_env_lsqr_precond_off() -> None:
    from pyAOBS.modeling.tomo2d.gui.services.run_env import collect_run_env

    st = FormState({"env.inv_lsqr_precond": False, "env.inv_lsqr_precond_max": "30"})
    env = collect_run_env(st)
    assert env["TOMO2D_INV_LSQR_PRECOND"] == "0"
    assert env["TOMO2D_INV_LSQR_PRECOND_MAX"] == "30"


def test_collect_run_env_graph_fs_enum_off() -> None:
    from pyAOBS.modeling.tomo2d.gui.services.run_env import collect_run_env

    st = FormState({"env.graph_fs_enum": False})
    env = collect_run_env(st)
    assert env["TOMO2D_GRAPH_FS_ENUM"] == "0"


def test_format_strategy_flags_only_on() -> None:
    from pyAOBS.modeling.tomo2d.gui.services.run_env import (
        format_strategy_flags,
        strategy_env_from_form,
    )

    env = strategy_env_from_form(
        {
            "env.inv_reuse_forward": True,
            "env.inv_reuse_thresh": "1e-3",
            "env.inv_coarse2fine": False,
            "env.inv_lsqr_precond": True,
            "env.inv_sens_weight": True,
            "env.inv_linesearch": True,
            "env.inv_lm": True,
            "env.inv_legacy_baseline": False,
        }
    )
    s = format_strategy_flags(env)
    assert "前向复用" in s and "1e-3" in s
    assert "列预条件 maxD=10" in s
    assert "灵敏度加权 κ=10" in s
    assert "线搜索" in s
    assert "LM" in s
    assert "=开" not in s
    assert "C2F" not in s
    assert "Legacy" not in s
    legacy = format_strategy_flags(
        strategy_env_from_form({"env.inv_legacy_baseline": True, "env.inv_reuse_forward": True})
    )
    assert legacy == "Legacy"


def test_param_tooltip_parallel_env() -> None:
    from pyAOBS.modeling.tomo2d.param_hints import get_param_tooltip, soft_wrap_tooltip

    tip = get_param_tooltip("env.inv_omp")
    assert "TOMO2D_INV_OMP" in tip
    assert "作用" in tip
    assert "\n" in tip
    assert get_param_tooltip("env.inv_lsqr_precond").find("TOMO2D_INV_LSQR_PRECOND") >= 0
    assert "TOMO2D_INV_SENS_WEIGHT" in get_param_tooltip("env.inv_sens_weight")
    assert "TOMO2D_INV_LINESEARCH" in get_param_tooltip("env.inv_linesearch")
    assert "TOMO2D_INV_LM" in get_param_tooltip("env.inv_lm")
    assert "TOMO2D_GRAPH_FS_ENUM" in get_param_tooltip("env.graph_fs_enum")
    assert get_param_tooltip("no.such.key") == ""
    assert "短句" in soft_wrap_tooltip("短句")
    assert "预设" in get_param_tooltip("tab.parallel_env") or "常显" in get_param_tooltip(
        "tab.parallel_env"
    )
    assert "-B" in get_param_tooltip("fwd.seafloor_file")
    assert "-Y" in get_param_tooltip("inv.seafloor_file")
    assert "贴面" in get_param_tooltip("fwd.do_full_refl")
    assert "穿" in get_param_tooltip("inv.do_full_refl")
    assert "-U" in get_param_tooltip("fwd.vsmesh")
    assert "-U" in get_param_tooltip("inv.vsmesh")
    assert "-y" in get_param_tooltip("inv.invert_water_only")
    assert "-w" in get_param_tooltip("inv.invert_crust_only")


def test_parallel_env_presets_dict() -> None:
    from pyAOBS.modeling.tomo2d.gui.panels.parallel_env_panel import _PRESETS

    assert "快速（OMP+复用）" in _PRESETS
    fast = _PRESETS["快速（OMP+复用）"]
    assert fast["env.inv_reuse_forward"] is True
    assert fast["env.inv_legacy_baseline"] is False
    assert fast["env.inv_lsqr_precond"] is False
    legacy = _PRESETS["对拍（Legacy）"]
    assert legacy["env.inv_legacy_baseline"] is True
    assert legacy["env.inv_lsqr_precond"] is False


def test_ttimes_noise_and_checkerboard_smesh(tmp_path: Path) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.smesh_ops import (
        apply_checkerboard_to_file,
        apply_random_init_to_file,
        checkerboard_velocity_fields,
        stack_mean_std,
        write_velocity_grid_as_smesh,
    )
    from pyAOBS.modeling.tomo2d.gui.services.ttimes_noise import add_traveltime_noise
    from pyAOBS.model_building.tomoform import SlownessMesh2D

    # 最小 smesh
    m = SlownessMesh2D(4, 3, 1.5, 0.33)
    m.xpos = __import__("numpy").linspace(0, 3, 4)
    m.topo = __import__("numpy").zeros(4)
    m.zpos = __import__("numpy").linspace(0, 2, 3)
    m.vgrid[:] = 4.0
    m.pgrid = 1.0 / m.vgrid
    src = tmp_path / "bg.smesh"
    m.to_file(str(src))
    _mesh, v_bg, v_cb, dv = checkerboard_velocity_fields(
        src, amp_percent=5, h_len=10, v_len=5
    )
    assert v_bg.shape == m.vgrid.shape
    assert (v_cb != v_bg).any()
    assert abs(float((v_cb - v_bg - dv).max())) < 1e-12
    true = tmp_path / "cb.smesh"
    apply_checkerboard_to_file(src, true, amp_percent=5, h_len=10, v_len=5)
    r1 = tmp_path / "r1.smesh"
    r2 = tmp_path / "r2.smesh"
    apply_random_init_to_file(src, r1, amp_percent=2, seed=1)
    apply_random_init_to_file(src, r2, amp_percent=2, seed=2)
    _, mean_v, std_v = stack_mean_std([r1, r2])
    assert mean_v.shape == m.vgrid.shape
    assert (std_v >= 0).all()
    out = tmp_path / "mean.smesh"
    write_velocity_grid_as_smesh(src, mean_v, out)
    assert out.is_file()

    data = tmp_path / "tt.dat"
    data.write_text(
        "1\n"
        "s     0.000     0.000    2\n"
        "r     1.000     0.010    0     1.000     0.050\n"
        "r     2.000     0.010    0     2.000     0.050\n",
        encoding="utf-8",
    )
    noisy = tmp_path / "tt_n.dat"
    n = add_traveltime_noise(data, noisy, sigma=0.01, seed=42)
    assert n == 2
    assert noisy.is_file()


def test_random_init_skips_air_and_water_nodes(tmp_path: Path) -> None:
    import numpy as np

    from pyAOBS.modeling.tomo2d.gui.services.smesh_ops import (
        air_water_node_mask,
        random_init_velocity_fields,
    )
    from pyAOBS.model_building.tomoform import SlownessMesh2D

    m = SlownessMesh2D(3, 5, 1.5, 0.33)
    m.xpos = np.linspace(0.0, 2.0, 3)
    m.topo = np.full(3, 2.0)
    m.zpos = np.array([-2.5, -1.0, 0.0, 1.0, 2.0])
    m.vgrid[:] = 4.0
    m.vgrid[:, 0] = m.v_air
    m.vgrid[:, 1] = m.v_water
    m.pgrid = 1.0 / m.vgrid
    src = tmp_path / "aw.smesh"
    m.to_file(str(src))

    frozen = air_water_node_mask(m)
    assert frozen[:, 0].all() and frozen[:, 1].all()
    assert not frozen[:, 2:].any()

    _mesh, v_bg, v_pert, dv = random_init_velocity_fields(
        src, amp_percent=8, seed=7, smooth_sigma=0.0
    )
    assert np.allclose(v_pert[:, :2], v_bg[:, :2])
    assert np.allclose(dv[:, :2], 0.0)
    assert (np.abs(dv[:, 2:]) > 0).any()

    m2 = SlownessMesh2D(3, 3, 1.5, 0.33)
    m2.xpos = np.linspace(0.0, 2.0, 3)
    m2.topo = np.zeros(3)
    m2.zpos = np.linspace(0.0, 2.0, 3)
    m2.vgrid[:] = 4.0
    m2.vgrid[:, 0] = 1.5
    m2.pgrid = 1.0 / m2.vgrid
    src2 = tmp_path / "wcol.smesh"
    m2.to_file(str(src2))
    _m, bg2, pert2, dv2 = random_init_velocity_fields(
        src2, amp_percent=8, seed=3, smooth_sigma=0.0
    )
    assert np.allclose(pert2[:, 0], bg2[:, 0])
    assert (np.abs(dv2[:, 1:]) > 0).any()


def test_layered_1d_init_monotonic_and_skips_water(tmp_path: Path) -> None:
    import numpy as np

    from pyAOBS.modeling.tomo2d.gui.services.mc_init_models import (
        Layer1dBounds,
        layer1d_bounds_from_state,
        layered_1d_init_velocity_fields,
        parse_lo_hi,
        resolve_mc_init_mode,
        sample_layered_1d_profile,
    )
    from pyAOBS.modeling.tomo2d.gui.state.form_state import FormState
    from pyAOBS.model_building.tomoform import SlownessMesh2D

    assert parse_lo_hi("2.5 0.2", (0.0, 1.0)) == (0.2, 2.5)
    assert parse_lo_hi("3", (1.0, 2.0)) == (3.0, 3.0)
    assert parse_lo_hi("0.2 ~ 2.5", (0.0, 1.0)) == (0.2, 2.5)
    assert parse_lo_hi("6~11", (0.0, 1.0)) == (6.0, 11.0)
    from PySide6.QtWidgets import QApplication

    from pyAOBS.modeling.tomo2d.gui.widgets.form_rows import MinMaxRow

    _app = QApplication.instance() or QApplication([])
    mm = MinMaxRow("t")
    mm.set_text("0.2 2.5")
    assert mm.lo.text() == "0.2" and mm.hi.text() == "2.5"
    assert mm.text() == "0.2 2.5"
    mm.set_text("6 ~ 11")
    assert mm.text() == "6 11"
    mm.set_text("0.3")
    assert mm.lo.text() == "0.3" and mm.hi.text() == "0.3"
    st = FormState({"mc.random_init": True})
    assert resolve_mc_init_mode(st) == "layers1d"
    st.set("mc.init_mode", "smesh")
    assert resolve_mc_init_mode(st) == "layers1d"
    st.set("mc.init_mode", "分段随机 1D")
    assert resolve_mc_init_mode(st) == "layers1d"
    st.set("mc.init_mode", "扰动已有 smesh")
    assert resolve_mc_init_mode(st) == "layers1d"
    st.set("mc.init_mode", "v.in")
    assert resolve_mc_init_mode(st) == "vinlayers"
    st.set("mc.init_mode", "扰动 v.in 分层")
    assert resolve_mc_init_mode(st) == "vinlayers"
    st.set("mc.init_mode", "不随机")
    assert resolve_mc_init_mode(st) == "layers1d"

    rng = np.random.default_rng(11)
    s1d = sample_layered_1d_profile(
        rng, bounds=Layer1dBounds(), v_water=1.5, z_max=20.0
    )
    z1d, v1d = s1d.z_bsf, s1d.v
    assert np.all(np.diff(z1d) > 0)
    assert np.all(np.diff(v1d) >= -1e-12)
    assert float(v1d[0]) >= 1.5
    assert abs(s1d.uc_bot_bsf - (s1d.h_sed + s1d.h_uc)) < 1e-12
    assert abs(s1d.moho_bsf - (s1d.h_sed + s1d.h_uc + s1d.h_lc)) < 1e-12
    assert abs(s1d.h_crust - (s1d.h_uc + s1d.h_lc)) < 1e-12
    assert abs(s1d.h_mantle - (20.0 - s1d.h_sed - s1d.h_crust)) < 1e-6
    assert abs(float(z1d[-1]) - 20.0) < 1e-6
    assert np.all(np.diff(v1d) >= -1e-12)  # 正梯度（单调不减）
    st_old = FormState({"mc.crust_h": "15 35"})
    b_old = layer1d_bounds_from_state(st_old)
    assert b_old.uc_h == (6.0, 14.0)
    assert b_old.lc_h == (9.0, 21.0)
    st_new = FormState({"mc.uc_h": "6 11", "mc.crust_h": "15 35"})
    assert layer1d_bounds_from_state(st_new).uc_h == (6.0, 11.0)
    bdef = Layer1dBounds()
    uc_vs: list[float] = []
    lc_vs: list[float] = []
    for seed in range(20):
        s = sample_layered_1d_profile(
            np.random.default_rng(seed), bounds=bdef, v_water=1.5, z_max=20.0
        )
        v = s.v
        z = s.z_bsf
        # 沉积顶底 + 上地壳底（顶与基底共用）+ 下地壳底（顶与 Conrad 共用）+ 地幔底（顶与莫霍共用）
        assert len(v) == 5
        assert abs(float(z[1]) - s.h_sed) < 1e-6
        assert abs(float(z[2]) - s.uc_bot_bsf) < 1e-6
        assert abs(float(z[3]) - s.moho_bsf) < 1e-6
        assert int(np.isclose(z, s.uc_bot_bsf, atol=1e-5).sum()) == 1
        assert int(np.isclose(z, s.moho_bsf, atol=1e-5).sum()) == 1
        for z_ifc in (s.h_sed, s.uc_bot_bsf, s.moho_bsf):
            v_up = float(np.interp(z_ifc - 1e-3, z, v))
            v_dn = float(np.interp(z_ifc + 1e-3, z, v))
            assert abs(v_up - v_dn) < 0.05
        uc_vs.append(float(v[2]))
        lc_vs.append(float(v[3]))
    assert max(uc_vs) - min(uc_vs) > 0.15
    assert max(lc_vs) - min(lc_vs) > 0.05

    m = SlownessMesh2D(4, 6, 1.5, 0.33)
    m.xpos = np.linspace(0.0, 6.0, 4)
    m.topo = np.array([1.0, 1.2, 0.8, 1.1])
    m.zpos = np.array([-1.5, -0.5, 0.0, 2.0, 8.0, 16.0])
    m.vgrid[:] = 6.0
    m.vgrid[:, 0] = m.v_air
    m.vgrid[:, 1] = m.v_water
    m.pgrid = 1.0 / m.vgrid
    src = tmp_path / "base.smesh"
    m.to_file(str(src))

    _mesh, v_bg, v_new, dv, sample = layered_1d_init_velocity_fields(
        src, seed=3, bounds=Layer1dBounds()
    )
    _z, _v = sample.z_bsf, sample.v
    assert abs(sample.moho_bsf - (sample.h_sed + sample.h_uc + sample.h_lc)) < 1e-12
    assert np.allclose(v_new[:, :2], v_bg[:, :2])
    assert np.allclose(dv[:, :2], 0.0)
    sub = v_new[:, 2:]
    assert np.all(np.diff(sub, axis=1) >= -1e-9)
    assert np.allclose(sub[0], sub[1])
    assert np.allclose(sub[0], sub[-1])
    _a, _b, v2, _d2, _p2 = layered_1d_init_velocity_fields(src, seed=4)
    assert not np.allclose(v_new[:, 2:], v2[:, 2:])

    from pyAOBS.modeling.tomo2d.gui.services.mc_init_models import (
        apply_layered_1d_init_to_file,
        collect_layered_1d_profiles,
        draw_layered_1d_ensemble,
        layered_1d_preview_overlays,
        moho_interface_xz,
        moho_overlays_from_samples,
    )

    profiles, z_max, vw = collect_layered_1d_profiles(src, n=5, seed0=3)
    assert len(profiles) == 5
    assert vw == 1.5
    assert z_max > 0
    z0, v0 = profiles[0].z_bsf, profiles[0].v
    assert np.allclose(z0, _z) and np.allclose(v0, _v)
    x_m, z_m = moho_interface_xz(_mesh, sample.moho_bsf)
    assert np.allclose(z_m, np.asarray(_mesh.topo, dtype=float) + sample.moho_bsf)
    extra = moho_overlays_from_samples(_mesh, profiles)
    assert extra and extra[0]["color"].startswith("#")
    both = layered_1d_preview_overlays(_mesh, profiles)
    assert len(both) == len(extra)
    assert {d["linestyle"] for d in both} == {"--"}
    dst = tmp_path / "init.smesh"
    refl = tmp_path / "moho.refl"
    _p, rp = apply_layered_1d_init_to_file(
        src, dst, seed=3, bounds=Layer1dBounds(), refl_dst=refl
    )
    assert rp is not None and rp.is_file()
    first = refl.read_text(encoding="utf-8").lstrip().splitlines()[0]
    assert not first.startswith("#")
    loaded = np.loadtxt(refl)
    assert loaded.shape[0] >= 2
    assert np.allclose(loaded[:, 1], z_m)
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure

    ax = Figure().add_subplot(111)
    draw_layered_1d_ensemble(ax, profiles, z_max=z_max, title="t", highlight=0)
    # 5 条 Vp + 5 条各次 Moho + Vp 均值 + Moho 均值
    assert len(ax.lines) == 12
    labels = {ln.get_label() for ln in ax.lines}
    assert "第 1 次实现" in labels
    assert "第 1 次 Conrad" not in labels
    hi = next(ln for ln in ax.lines if ln.get_label() == "第 1 次实现")
    assert hi.get_linewidth() >= 2.0


def _write_tiny_vin(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                " 1    0.00  10.00",
                " 0    0.00   0.00",
                "         0      0",
                " 1    0.00  10.00",
                " 0    1.50   1.50",
                "         0      0",
                " 1    0.00  10.00",
                " 0    1.50   1.50",
                "         0      0",
                " 1    0.00  10.00",
                " 0    2.00   2.00",
                "         0      0",
                " 1    0.00  10.00",
                " 0    4.00   4.00",
                "         0      0",
                " 1    0.00  10.00",
                " 0    6.50   6.50",
                "         0      0",
                " 1    0.00  10.00",
                " 0   12.00  12.00",
                "         0      0",
                " 1    0.00  10.00",
                " 0    8.00   8.00",
                "         0      0",
                " 1    0.00  10.00",
                " 0    8.20   8.20",
                "         0      0",
                " 1    0.00  10.00",
                " 0   20.00  20.00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_vin_layers_named_ifaces_and_selective_fill(tmp_path: Path) -> None:
    import numpy as np

    from pyAOBS.model_building.tomoform import SlownessMesh2D
    from pyAOBS.modeling.tomo2d.gui.services.mc_init_models import (
        Layer1dBounds,
        apply_mc_init_to_file,
        mc_init_velocity_fields,
        resolve_mc_init_mode,
    )
    from pyAOBS.modeling.tomo2d.gui.services.mc_vin_layers import (
        UNIT_LC,
        UNIT_MANTLE,
        UNIT_SED,
        UNIT_UC,
        VinPerturbSpec,
        available_units,
        collect_vin_1d_profiles,
        default_vin_marks,
        draw_vin_1d_ensemble,
        format_vin_units,
        load_zelt,
        paint_vin_on_smesh,
        parse_vin_units,
        sample_unit_velocities,
        unit_iface_span,
        vin_1d_datum,
        vin_n_ifaces,
        vin_named_iface_overlays,
        vin_pick_overlays,
        vin_spec_from_state,
    )
    from pyAOBS.modeling.tomo2d.gui.state.form_state import FormState
    from pyAOBS.modeling.vedit.core.geo_ifaces import GeoIfaceMarks

    vin = tmp_path / "v.in"
    _write_tiny_vin(vin)
    n = vin_n_ifaces(vin)
    assert n == 4
    marks = default_vin_marks(n)
    assert marks.seafloor == 1
    assert marks.basement is None
    assert marks.conrad is None
    assert marks.moho == 2
    assert available_units(marks) == [UNIT_UC, UNIT_MANTLE]
    assert unit_iface_span(marks, n, UNIT_UC) == (1, 2)
    assert unit_iface_span(marks, n, UNIT_MANTLE) == (2, 3)
    assert UNIT_SED not in available_units(marks)
    assert UNIT_LC not in available_units(marks)

    with_basement = GeoIfaceMarks(seafloor=1, basement=2, conrad=None, moho=2).clamped(n)
    assert available_units(with_basement) == [UNIT_SED, UNIT_UC, UNIT_MANTLE]
    with_conrad = GeoIfaceMarks(seafloor=1, basement=None, conrad=2, moho=3).clamped(n)
    assert UNIT_LC in available_units(with_conrad)
    assert unit_iface_span(with_conrad, n, UNIT_UC) == (1, 2)
    assert unit_iface_span(with_conrad, n, UNIT_LC) == (2, 3)

    assert parse_vin_units("", [UNIT_UC, UNIT_MANTLE]) == [UNIT_UC, UNIT_MANTLE]
    assert parse_vin_units("none", [UNIT_UC, UNIT_MANTLE]) == []
    assert parse_vin_units("uc", [UNIT_UC, UNIT_MANTLE]) == [UNIT_UC]
    assert format_vin_units([]) == "none"
    assert resolve_mc_init_mode(FormState({"mc.init_mode": "vin"})) == "vinlayers"

    st = FormState({"mc.vin_units": "uc"})
    spec = vin_spec_from_state(st, vin)
    assert spec.units == (UNIT_UC,)
    zelt = load_zelt(vin)
    labs = [d["label"] for d in vin_named_iface_overlays(zelt, spec.marks)]
    assert "海底" in labs
    assert "莫霍" in labs
    assert "Conrad" not in labs
    assert "基底" not in labs

    m = SlownessMesh2D(3, 5, 1.5, 0.33)
    m.xpos = np.linspace(0.0, 10.0, 3)
    m.topo = np.array([2.0, 2.0, 2.0])
    m.zpos = np.array([-1.0, 0.0, 4.0, 10.0, 16.0])
    m.vgrid[:] = 5.0
    m.vgrid[:, 0] = m.v_water
    m.pgrid = 1.0 / m.vgrid
    src = tmp_path / "base.smesh"
    m.to_file(str(src))

    painted = paint_vin_on_smesh(m, zelt)
    spec_uc = VinPerturbSpec(marks=marks, units=(UNIT_UC,), n_ifaces=n)
    bounds = Layer1dBounds(uc_v=(5.8, 6.2), uc_h=(7.0, 7.0))
    _mesh, v_bg, v_new, dv = mc_init_velocity_fields(
        src,
        mode="vinlayers",
        seed=7,
        vin_path=vin,
        vin_spec=spec_uc,
        bounds=bounds,
    )
    assert np.allclose(v_new[:, 0], v_bg[:, 0])
    crust = v_new[:, 2]
    mantle = v_new[:, 4]
    assert not np.allclose(crust, painted[:, 2])
    assert np.allclose(mantle, painted[:, 4])
    assert 7.7 < float(np.mean(mantle)) < 8.4
    _a, _b, v2, _d2 = mc_init_velocity_fields(
        src,
        mode="vinlayers",
        seed=8,
        vin_path=vin,
        vin_spec=spec_uc,
        bounds=bounds,
    )
    assert not np.allclose(v_new[:, 2], v2[:, 2])

    dst = tmp_path / "init.smesh"
    refl = tmp_path / "moho.refl"
    _p, rp = apply_mc_init_to_file(
        src,
        dst,
        mode="vinlayers",
        seed=3,
        vin_path=vin,
        vin_spec=spec_uc,
        bounds=bounds,
        refl_dst=refl,
    )
    assert rp is not None and rp.is_file()
    first = refl.read_text(encoding="utf-8").lstrip().splitlines()[0]
    assert not first.startswith("#")
    loaded = np.loadtxt(refl)
    assert loaded.shape[0] >= 2
    assert np.allclose(loaded[:, 1], 9.0)

    extra = vin_pick_overlays(zelt, marks, [UNIT_UC])
    texts = [d.get("text") for d in extra]
    assert "海底" in texts
    assert "莫霍" in texts
    assert "Conrad" not in texts
    assert any(d.get("z_lo") is not None for d in extra)

    assert vin_1d_datum(marks, n, [UNIT_UC, UNIT_MANTLE]) == ("seafloor", 1)
    assert vin_1d_datum(marks, n, [UNIT_MANTLE]) == ("moho", 2)
    bm_marks = GeoIfaceMarks(seafloor=1, basement=2, conrad=None, moho=3).clamped(n)
    assert vin_1d_datum(bm_marks, n, [UNIT_UC]) == ("basement", 2)
    cn_marks = GeoIfaceMarks(seafloor=1, basement=None, conrad=2, moho=3).clamped(n)
    assert vin_1d_datum(cn_marks, n, [UNIT_LC]) == ("conrad", 2)

    spec_mantle = VinPerturbSpec(marks=marks, units=(UNIT_MANTLE,), n_ifaces=n)
    prof_m, z_m, role_m = collect_vin_1d_profiles(
        vin, spec=spec_mantle, n=3, seed0=7, bounds=bounds, v_water=1.5
    )
    assert role_m == "moho"
    assert len(prof_m) == 3
    assert abs(float(prof_m[0].z_rel[0])) < 1e-9
    assert prof_m[0].iface_rel.get("moho", 1.0) <= 1e-6

    spec_all = VinPerturbSpec(marks=marks, units=(UNIT_UC, UNIT_MANTLE), n_ifaces=n)
    prof_s, _z_s, role_s = collect_vin_1d_profiles(
        vin, spec=spec_all, n=3, seed0=7, bounds=bounds, v_water=1.5
    )
    assert role_s == "seafloor"
    assert abs(float(prof_s[0].z_rel[0])) < 1e-9
    assert prof_s[0].iface_rel["moho"] == bounds.uc_h[0]
    sv = sample_unit_velocities(
        np.random.default_rng(7),
        bounds,
        [UNIT_UC, UNIT_MANTLE],
        v_water=1.5,
        marks=marks,
        n_ifaces=n,
    )
    assert sv[UNIT_UC][1] == sv[UNIT_MANTLE][0]
    for s1 in prof_s:
        assert np.all(np.diff(s1.z_rel) > 1e-9)
        zm = s1.iface_rel["moho"]
        v_up = float(np.interp(zm - 1e-3, s1.z_rel, s1.v))
        v_dn = float(np.interp(zm + 1e-3, s1.z_rel, s1.v))
        assert abs(v_up - v_dn) < 0.05

    spec_bm = VinPerturbSpec(marks=bm_marks, units=(UNIT_UC,), n_ifaces=n)
    prof_b, _zb, role_b = collect_vin_1d_profiles(
        vin, spec=spec_bm, n=2, seed0=7, bounds=bounds, v_water=1.5
    )
    assert role_b == "basement"
    assert abs(float(prof_b[0].z_rel[0])) < 1e-9
    assert "conrad" not in prof_b[0].iface_rel

    spec_cn = VinPerturbSpec(marks=cn_marks, units=(UNIT_LC,), n_ifaces=n)
    prof_c, _zc, role_c = collect_vin_1d_profiles(
        vin,
        spec=spec_cn,
        n=2,
        seed0=7,
        bounds=Layer1dBounds(lc_h=(5.0, 5.0), lc_v=(6.6, 7.2)),
        v_water=1.5,
    )
    assert role_c == "conrad"
    assert abs(float(prof_c[0].z_rel[0])) < 1e-9
    assert prof_c[0].iface_rel["moho"] == 5.0
    assert "basement" not in prof_c[0].iface_rel

    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.figure import Figure

    ax = Figure().add_subplot(111)
    draw_vin_1d_ensemble(ax, prof_s, z_max=_z_s, title="t", highlight=0)
    labels = {ln.get_label() for ln in ax.lines}
    assert "第 1 次实现" in labels
    assert "第 1 次 Conrad" not in labels
    assert "第 1 次 莫霍" in labels
    ax2 = Figure().add_subplot(111)
    draw_vin_1d_ensemble(ax2, prof_m, z_max=z_m, title="t", highlight=0)
    labels_m = {ln.get_label() for ln in ax2.lines}
    assert "第 1 次 Conrad" not in labels_m
    assert ax2.get_ylabel() == "莫霍以下深度 (km)"


def test_load_zelt_caches_and_clones(tmp_path: Path, caplog) -> None:
    import logging

    from pyAOBS.modeling.tomo2d.gui.services.mc_vin_layers import load_zelt, vin_n_ifaces

    vin = tmp_path / "v.in"
    _write_tiny_vin(vin)
    caplog.set_level(logging.INFO, logger="pyAOBS.model_building.read")
    a = load_zelt(vin, clone=False)
    n0 = sum(1 for r in caplog.records if "Reading velocity model" in r.message)
    assert n0 == 1
    assert vin_n_ifaces(vin) == 4
    b = load_zelt(vin, clone=True)
    c = load_zelt(vin, clone=False)
    n1 = sum(1 for r in caplog.records if "Reading velocity model" in r.message)
    assert n1 == 1
    assert c is a
    assert b is not a
    b.depth_nodes[1].val[0] = 99.0
    assert a.depth_nodes[1].val[0] != 99.0
    vin.write_text(vin.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    load_zelt(vin, clone=False)
    n2 = sum(1 for r in caplog.records if "Reading velocity model" in r.message)
    assert n2 == 2


def test_subsample_refl_keeps_first_stride_and_last() -> None:
    from pyAOBS.modeling.tomo2d.gui.services.refl_stride import (
        parse_refl_stride,
        subsample_refl_lines,
    )

    assert parse_refl_stride("") == 1
    assert parse_refl_stride("1") == 1
    assert parse_refl_stride("2") == 2
    lines = [f"{i} 30\n" for i in range(5)]
    out = subsample_refl_lines(lines, 2)
    assert [ln.strip() for ln in out] == ["0 30", "2 30", "4 30"]
    assert subsample_refl_lines(lines, 1) == lines


def test_collect_tt_inverse_refl_stride(tmp_path: Path) -> None:
    src = tmp_path / "refl.dat"
    src.write_text("".join(f"{i} 30\n" for i in range(5)), encoding="utf-8")
    st = FormState(
        {
            "work_dir": str(tmp_path),
            "inv.mesh": "m.smesh",
            "inv.data": "d.dat",
            "inv.refl_file": "refl.dat",
            "inv.refl_stride": "2",
        }
    )
    _, _, kw = collect_tt_inverse_args(st)
    assert kw["refl_file"] == "refl.dat"
    assert kw["_refl_stride"] == 2
    assert not (tmp_path / "refl_s2.dat").exists()
    assert src.read_text(encoding="utf-8").startswith("0 30")

    st.set("inv.refl_stride", "1")
    _, _, kw1 = collect_tt_inverse_args(st)
    assert kw1["refl_file"] == "refl.dat"
    assert "_refl_stride" not in kw1

    st.set("inv.refl_stride", "")
    _, _, kw_empty = collect_tt_inverse_args(st)
    assert kw_empty["refl_file"] == "refl.dat"
    assert "_refl_stride" not in kw_empty

    st.set("inv.refl_stride", "abc")
    with pytest.raises(ValueError, match="正整数"):
        collect_tt_inverse_args(st)

    st.set("inv.refl_stride", "2")
    st.set("inv.refl_file", "missing.dat")
    with pytest.raises(ValueError, match="不存在"):
        collect_tt_inverse_args(st)

    st.set("inv.refl_file", "refl.dat")
    _, _, kw2 = collect_tt_inverse_args(st)
    from pyAOBS.modeling.tomo2d.gui.services.refl_stride import materialize_refl_for_cwd

    dest = materialize_refl_for_cwd(kw2, tmp_path, tmp_path)
    assert dest is not None
    assert dest == tmp_path / ".tomo2d_tmp" / "refl_stride.dat"
    kept = [ln.strip() for ln in dest.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert kept == ["0 30", "2 30", "4 30"]
    assert kw2["refl_file"] == ".tomo2d_tmp/refl_stride.dat"
    assert "_refl_stride" not in kw2
    assert not (tmp_path / "refl_s2.dat").exists()


def test_checkerboard_prefers_companion_refl_over_inv_start(tmp_path: Path) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.qc_workflows import (
        _checkerboard_fwd_inv_kwargs,
        _resolve_checkerboard_refl,
        preview_checkerboard,
    )

    models = tmp_path / "runs" / "ttinv_x" / "outputs" / "models"
    models.mkdir(parents=True)
    smesh = models / "out.smesh.8.1"
    companion = models / "out.refl.8.1"
    start = tmp_path / "outputs" / "vpfd42.refl"
    start.parent.mkdir(parents=True)
    smesh.write_text("mesh", encoding="utf-8")
    companion.write_text("0 10\n1 10\n2 10\n", encoding="utf-8")
    start.write_text("0 30\n0.5 30\n", encoding="utf-8")
    st = FormState(
        {
            "work_dir": str(tmp_path),
            "cb.bg_smesh": "runs/ttinv_x/outputs/models/out.smesh.8.1",
            "cb.geom": "geom.dat",
            "inv.refl_file": "outputs/vpfd42.refl",
            "inv.refl_stride": "2",
        }
    )
    rel, src, companion_hit = _resolve_checkerboard_refl(
        st, tmp_path, st.get_str("cb.bg_smesh")
    )
    assert companion_hit
    assert src == "背景同轮界面"
    assert rel.replace("\\", "/").endswith("runs/ttinv_x/outputs/models/out.refl.8.1")
    assert st.get_str("cb.refl_file").replace("\\", "/").endswith(
        "runs/ttinv_x/outputs/models/out.refl.8.1"
    )

    fwd_kw, inv_kw, notes = _checkerboard_fwd_inv_kwargs(st, tmp_path)
    assert "_refl_stride" not in inv_kw
    assert Path(str(inv_kw["refl_file"])).name == "out.refl.8.1"
    assert Path(str(fwd_kw["refl_file"])).name == "out.refl.8.1"
    assert any("同轮" in n for n in notes)

    text = preview_checkerboard(st, tmp_path)
    assert "out.refl.8.1" in text
    assert "背景同轮界面" in text
    assert "抽稀步长: 1" in text
    assert "vpfd42.refl" not in text.split("棋盘格分辨率测试")[1].split("步骤:")[0]
    from pyAOBS.modeling.tomo2d.gui.services.qc_workflows import (
        resolve_checkerboard_inputs,
    )

    plan = resolve_checkerboard_inputs(st, tmp_path)
    assert plan.refl == st.get_str("cb.refl_file")
    assert plan.staged_refl == "inputs/out.refl.8.1"
    assert plan.staged_refl in text
    assert inv_kw["refl_file"] == plan.staged_refl
    assert fwd_kw["refl_file"] == plan.staged_refl


def test_checkerboard_no_companion_does_not_use_inv_refl(tmp_path: Path) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.qc_workflows import (
        _checkerboard_fwd_inv_kwargs,
        _resolve_checkerboard_refl,
    )

    bg = tmp_path / "bg.smesh"
    start = tmp_path / "outputs" / "vpfd42.refl"
    chosen = tmp_path / "outputs" / "user.refl"
    start.parent.mkdir(parents=True)
    bg.write_text("mesh", encoding="utf-8")
    start.write_text("0 30\n", encoding="utf-8")
    chosen.write_text("0 10\n1 10\n", encoding="utf-8")
    st = FormState(
        {
            "work_dir": str(tmp_path),
            "cb.bg_smesh": "bg.smesh",
            "inv.refl_file": "outputs/vpfd42.refl",
            "inv.mesh": "bg.smesh",
            "inv.data": "d.dat",
        }
    )
    rel, src, companion_hit = _resolve_checkerboard_refl(st, tmp_path, "bg.smesh")
    assert not companion_hit
    assert rel is None
    assert src == "cb.refl_file"

    fwd_kw, inv_kw, notes = _checkerboard_fwd_inv_kwargs(st, tmp_path)
    assert "refl_file" not in inv_kw
    assert "refl_file" not in fwd_kw
    assert any("自选" in n or "cb.refl_file" in n for n in notes)

    st.set("cb.refl_file", "outputs/user.refl")
    rel2, src2, hit2 = _resolve_checkerboard_refl(st, tmp_path, "bg.smesh")
    assert not hit2
    assert src2 == "cb.refl_file"
    assert rel2.replace("\\", "/").endswith("outputs/user.refl")
    _fwd, inv2, _n = _checkerboard_fwd_inv_kwargs(st, tmp_path)
    assert Path(str(inv2["refl_file"])).name == "user.refl"


def test_fill_cb_refl_from_companion(tmp_path: Path) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.qc_workflows import (
        fill_cb_refl_from_companion,
    )

    models = tmp_path / "runs" / "ttinv_x" / "outputs" / "models"
    models.mkdir(parents=True)
    (models / "out.smesh.8.1").write_text("m", encoding="utf-8")
    (models / "out.refl.8.1").write_text("0 1\n", encoding="utf-8")
    st = FormState(
        {
            "work_dir": str(tmp_path),
            "cb.bg_smesh": "runs/ttinv_x/outputs/models/out.smesh.8.1",
            "cb.refl_file": "",
        }
    )
    filled = fill_cb_refl_from_companion(st, tmp_path)
    assert filled is not None
    assert filled.replace("\\", "/").endswith("out.refl.8.1")
    assert st.get_str("cb.refl_file") == filled

    st2 = FormState({"work_dir": str(tmp_path), "cb.bg_smesh": "missing.smesh"})
    assert fill_cb_refl_from_companion(st2, tmp_path) is None


def test_preview_monte_carlo_lists_tt_inverse_like_checkerboard(tmp_path: Path) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.qc_workflows import preview_monte_carlo

    st = FormState(
        {
            "work_dir": str(tmp_path),
            "mc.base_mesh": "base.smesh",
            "mc.data": "tt.dat",
            "mc.n_runs": "5",
            "mc.seed": "7",
            "mc.init_mode": "smesh",
            "mc.sed_h": "0.2 2.5",
            "mc.uc_h": "6 11",
            "mc.lc_h": "10 25",
            "mc.mantle_v": "7.6 8.2",
            "inv.niter": "3",
            "inv.refl_file": "outputs/old.refl",
        }
    )
    text = preview_monte_carlo(st, tmp_path)
    assert "tomo.tt_inverse" in text
    assert "reals/000/init.smesh" in text
    assert "reals/000/data.dat" in text
    assert "reals/000/moho.refl" in text
    assert "reals/000/tt_inverse.log" in text
    assert "outputs/mean_velocity.smesh" in text
    assert "N=5" in text
    assert "seed=7" in text
    assert "pred χ² < 1.8" in text
    assert "outputs/dws.dat" in text
    assert "上地壳厚度" in text
    assert "Conrad" not in text
    call = text.split("tomo.tt_inverse", 1)[1]
    assert "reals/000/moho.refl" in call
    assert "old.refl" not in call
    assert "mesh='reals/000/init.smesh'" in call

    class _FakeTomo:
        def resolve_cmdline_tt_inverse(self, **kw):
            return [
                "tt_inverse",
                "-M",
                kw["mesh"],
                "-G",
                kw["data"],
                "-F",
                kw["refl_file"],
                "-L",
                kw["log_file"],
            ]

    with_cmd = preview_monte_carlo(st, tmp_path, _FakeTomo())
    assert "# 解析后命令行（tt_inverse  第 1 次）:" in with_cmd
    assert "reals/000/init.smesh" in with_cmd.split("# 解析后命令行")[1]

    st_smesh = FormState(
        {
            "work_dir": str(tmp_path),
            "mc.base_mesh": "base.smesh",
            "mc.data": "tt.dat",
            "mc.init_mode": "smesh",
            "inv.refl_file": "outputs/iface.refl",
        }
    )
    text_s = preview_monte_carlo(st_smesh, tmp_path)
    assert "tomo.tt_inverse" in text_s
    assert "分段随机 1D" in text_s
    assert "相关扰动" not in text_s
    call_s = text_s.split("tomo.tt_inverse", 1)[1]
    assert "reals/000/moho.refl" in call_s
    assert "iface.refl" not in call_s

    vin = tmp_path / "v.in"
    _write_tiny_vin(vin)
    st_vin = FormState(
        {
            "work_dir": str(tmp_path),
            "mc.base_mesh": "base.smesh",
            "mc.data": "tt.dat",
            "mc.init_mode": "v.in",
            "mc.v_in": "v.in",
            "mc.vin_units": "uc",
            "inv.refl_file": "outputs/old.refl",
        }
    )
    text_v = preview_monte_carlo(st_vin, tmp_path)
    assert "起始方式: v.in" in text_v
    assert "每次 -F 用选定的莫霍界面" in text_v
    assert "reals/000/moho.refl" in text_v
    assert "Conrad" not in text_v
    call_v = text_v.split("tomo.tt_inverse", 1)[1]
    assert "reals/000/moho.refl" in call_v
    assert "old.refl" not in call_v


def _tt_inverse_log_line(*, iter_n: int, pred_chi: float, chi_total: float = 2.0) -> str:
    vals = [0.0] * 26
    vals[0] = float(iter_n)
    vals[1] = 1.0
    vals[4] = float(chi_total)
    vals[20] = float(pred_chi)
    return " ".join(f"{v:g}" for v in vals)


def test_mc_chi_filter_and_default(tmp_path: Path) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.qc_workflows import (
        MC_CHI_MAX_DEFAULT,
        find_realization_tt_inverse_log,
        mc_chi_max_from_state,
        realization_pred_chi,
    )

    assert mc_chi_max_from_state(FormState({})) == MC_CHI_MAX_DEFAULT
    assert mc_chi_max_from_state(FormState({"mc.chi_max": ""})) == 1.8
    assert mc_chi_max_from_state(FormState({"mc.chi_max": "2.5"})) == 2.5
    assert mc_chi_max_from_state(FormState({"mc.chi_max": "abc"})) == 1.8

    real = tmp_path / "reals" / "003"
    logs = real / "logs"
    logs.mkdir(parents=True)
    (logs / "tt_inverse.log").write_text(
        "# header\n"
        + _tt_inverse_log_line(iter_n=1, pred_chi=2.2)
        + "\n"
        + _tt_inverse_log_line(iter_n=2, pred_chi=1.4)
        + "\n",
        encoding="utf-8",
    )
    assert find_realization_tt_inverse_log(real) == logs / "tt_inverse.log"
    assert realization_pred_chi(real) == 1.4
    assert realization_pred_chi(real) < mc_chi_max_from_state(FormState({}))

    empty = tmp_path / "reals" / "004"
    empty.mkdir(parents=True)
    assert realization_pred_chi(empty) is None


def test_format_mc_result_notes_chi_stats() -> None:
    from pyAOBS.modeling.tomo2d.gui.services.qc_workflows import (
        format_mc_result_notes,
        summarize_chi_values,
    )

    summ = summarize_chi_values([1.1, 0.9, 1.6])
    assert summ["n"] == 3
    assert abs(summ["mean"] - 1.2) < 1e-9
    assert summ["min"] == 0.9
    assert summ["max"] == 1.6
    mean_t, std_t, hint = format_mc_result_notes(
        {
            "n_runs": 5,
            "n_kept": 3,
            "chi_max": 1.8,
            "realization_chi": [
                {"pred_chi": 1.1, "kept": True},
                {"pred_chi": 0.9, "kept": True},
                {"pred_chi": 1.6, "kept": True},
                {"pred_chi": 2.2, "kept": False},
            ],
        }
    )
    assert "用 3/5 个模型平均" in mean_t
    assert "pred χ² < 1.8" in mean_t
    assert "均值 1.20" in mean_t
    assert "最小 0.90" in mean_t
    assert "最大 1.60" in mean_t
    assert "中位" in mean_t
    assert "3 个模型" in std_t
    assert "3/5 个模型" in hint

    unfiltered, _std, _h = format_mc_result_notes(
        {
            "n_runs": 4,
            "n_kept": 4,
            "realization_chi": [
                {"pred_chi": 1.36, "kept": True},
                {"pred_chi": 1.83, "kept": True},
                {"pred_chi": 2.04, "kept": True},
                {"pred_chi": 3.28, "kept": True},
            ],
        }
    )
    assert "最大 1.36" in unfiltered
    assert "3.28" not in unfiltered
    assert "用 1/4 个模型" in unfiltered or "用 1 个模型" in unfiltered


def test_collect_gen_dcorr_kwargs_uniform() -> None:
    st = FormState(
        {
            "dcorr.mode": "uniform",
            "dcorr.lh": "4",
            "dcorr.xmin": "0",
            "dcorr.xmax": "410",
            "dcorr.out_file": "outputs/dcorr.dat",
        }
    )
    kw = collect_gen_dcorr_kwargs(st)
    assert kw["mode"] == "uniform"
    assert kw["lh"] == 4
    assert kw["xmin"] == 0
    assert kw["xmax"] == 410
    assert kw["out_file"] == "outputs/dcorr.dat"


def test_collect_gen_vcorr_kwargs_simple_2x2() -> None:
    st = FormState(
        {
            "vcorr.mode": "simple_2x2",
            "vcorr.Lht": "2.0",
            "vcorr.Lhb": "4.0",
            "vcorr.Lvt": "1.0",
            "vcorr.Lvb": "4.0",
            "vcorr.xmin": "0",
            "vcorr.xmax": "410",
            "vcorr.zmin": "0",
            "vcorr.zmax": "45",
            "vcorr.out_file": "outputs/vcorr.dat",
        }
    )
    kw = collect_gen_vcorr_kwargs(st)
    assert kw["mode"] == "simple_2x2"
    assert kw["Lht"] == 2.0
    assert kw["Lhb"] == 4.0
    assert kw["Lvt"] == 1.0
    assert kw["Lvb"] == 4.0
    assert kw["xmin"] == 0
    assert kw["xmax"] == 410
    assert kw["zmin"] == 0
    assert kw["zmax"] == 45
    assert kw["out_file"] == "outputs/vcorr.dat"
    assert "vel_opt" not in kw


def test_collect_gen_vcorr_kwargs_program_without_mode_key() -> None:
    st = FormState(
        {
            "vcorr.vel_opt": "uniform",
            "vcorr.grid_opt": "uniform",
            "vcorr.abnormal_h": "2",
            "vcorr.abnormal_v": "1",
            "vcorr.normal_h": "4",
            "vcorr.normal_v": "4",
            "vcorr.nx": "2",
            "vcorr.nz": "2",
            "vcorr.xmax": "410",
            "vcorr.zmax": "45",
        }
    )
    kw = collect_gen_vcorr_kwargs(st)
    assert kw["mode"] == "program"
    assert kw["vel_opt"] == "uniform"
    assert kw["abnormal_h"] == 2


def test_collect_tt_inverse_damp_mutex_auto_ignores_fixed_leftover() -> None:
    st = FormState(
        {
            "inv.mesh": "m.smesh",
            "inv.data": "d.dat",
            "inv.auto_damp_max_dv": "20",
            "inv.damp_vel": "20",
            "inv.damp_kind": "auto",
        }
    )
    _, _, kw = collect_tt_inverse_args(st)
    assert kw["auto_damp_max_dv"] == 20
    assert "damp_opts" not in kw


def test_collect_tt_inverse_damp_mutex_fixed_ignores_auto_leftover() -> None:
    st = FormState(
        {
            "inv.mesh": "m.smesh",
            "inv.data": "d.dat",
            "inv.auto_damp_max_dv": "20",
            "inv.damp_vel": "20",
            "inv.damp_kind": "fixed",
        }
    )
    _, _, kw = collect_tt_inverse_args(st)
    assert kw["damp_opts"]["vel"] == 20
    assert "auto_damp_max_dv" not in kw


def test_collect_tt_inverse_damp_conflict_raises() -> None:
    st = FormState(
        {
            "inv.mesh": "m.smesh",
            "inv.data": "d.dat",
            "inv.auto_damp_max_dv": "20",
            "inv.damp_vel": "20",
            "inv.damp_kind": "conflict",
        }
    )
    with pytest.raises(ValueError, match="互斥"):
        collect_tt_inverse_args(st)


def test_format_tt_inverse_flag_summary() -> None:
    auto = format_tt_inverse_flag_summary({"auto_damp_max_dv": 20})
    assert "自动阻尼 -T" in auto
    assert "-TV 20%" in auto
    assert "滤波" not in auto
    fixed = format_tt_inverse_flag_summary({"damp_opts": {"vel": 20, "dep": 20}})
    assert "固定阻尼 -D" in fixed
    assert "-DV 20" in fixed
    with_s = format_tt_inverse_flag_summary({"filter_bound_file": "bound.dat"})
    assert "滤波 -s" in with_s
    assert "bound.dat" in with_s
    assert "=开" not in with_s
    mesh_s = format_tt_inverse_flag_summary({"apply_filter": True})
    assert "滤波 -s" in mesh_s
    assert "mesh" in mesh_s
    yw = format_tt_inverse_flag_summary(
        {"seafloor_file": "sf.dat", "invert_crust_only": True}
    )
    assert "只反壳 -w" in yw
    assert "海底 -Y" in yw
    assert "只反水" not in yw
    water = format_tt_inverse_flag_summary(
        {"seafloor_file": "sf.dat", "invert_water_only": True}
    )
    assert "只反水 -y" in water
    assert "海底 -Y" in water


def test_collect_tt_inverse_apply_filter_mesh_topo() -> None:
    st = FormState(
        {
            "inv.mesh": "m.smesh",
            "inv.data": "d.dat",
            "inv.apply_filter": True,
        }
    )
    _, _, kw = collect_tt_inverse_args(st)
    assert kw.get("apply_filter") is True
    assert "filter_bound_file" not in kw


def test_collect_tt_inverse_apply_filter_off_ignores_leftover_file() -> None:
    st = FormState(
        {
            "inv.mesh": "m.smesh",
            "inv.data": "d.dat",
            "inv.apply_filter": False,
            "inv.filter_bound_file": "bound.dat",
        }
    )
    _, _, kw = collect_tt_inverse_args(st)
    assert "apply_filter" not in kw
    assert "filter_bound_file" not in kw


def test_collect_tt_inverse_old_profile_file_implies_filter() -> None:
    st = FormState(
        {
            "inv.mesh": "m.smesh",
            "inv.data": "d.dat",
            "inv.filter_bound_file": "bound.dat",
        }
    )
    assert st.get_bool("inv.apply_filter") is True
    _, _, kw = collect_tt_inverse_args(st)
    assert kw.get("apply_filter") is True
    assert kw.get("filter_bound_file") == "bound.dat"


def test_parse_tt_inverse_log_header_new_and_old(tmp_path: Path) -> None:
    from pyAOBS.modeling.tomo2d.tt_inverse_log_analysis import (
        format_tt_inverse_log_header_summary,
        format_tt_inverse_run_params,
        parse_tt_inverse_log_header,
    )

    p_new = tmp_path / "new.log"
    p_new.write_text(
        "# damping: MODE=auto  using=-T/-TV/-TD  fixed_-D=OFF"
        "  -TV_percent=20 -TV_frac=0.2\n"
        "# filter_-s: OFF  (ON=2D filter after each iter; OFF=no -s)\n"
        "# lsqr_precond: ON maxD=50  (OFF=Legacy or TOMO2D_INV_LSQR_PRECOND=0; maxD<=0=unbounded)\n"
        "# accel: reuse=ON thresh=0.001 c2f=OFF legacy=OFF\n"
        "1 1 0 0.1\n",
        encoding="utf-8",
    )
    info = parse_tt_inverse_log_header(p_new)
    assert info["damping_mode"] == "auto"
    assert info["filter_2d"] is False
    assert info["lsqr_precond"] is True
    assert info["lsqr_precond_maxd"] == 50
    assert info["reuse_forward"] is True
    assert info["coarse2fine"] is False
    summ = format_tt_inverse_log_header_summary(info)
    assert "自动阻尼 -T" in summ
    assert "滤波" not in summ
    assert "列预条件 maxD=50" in summ
    assert "前向复用" in summ
    assert "C2F" not in summ
    assert "=开" not in summ

    p_fixed = tmp_path / "fixed.log"
    p_fixed.write_text(
        "# damping: MODE=fixed  using=-D/-DV/-DD  auto_-T=OFF  -DV=20 -DD=20\n"
        "# filter_-s: ON  (ON=2D filter after each iter; OFF=no -s)\n",
        encoding="utf-8",
    )
    info_f = parse_tt_inverse_log_header(p_fixed)
    assert info_f["damping_mode"] == "fixed"
    assert info_f["filter_2d"] is True
    assert "固定阻尼 -D" in format_tt_inverse_log_header_summary(info_f)
    assert "滤波 -s" in format_tt_inverse_log_header_summary(info_f)

    p_params = tmp_path / "params.log"
    p_params.write_text(
        "# strategy jumping=1 robust=0 crit_chi=0\n"
        "# smooth_vel -SV on=1 wmin=200 wmax=200 dw=1 log10(-XV)=0\n"
        "# smooth_dep -SD on=1 wmin=10 wmax=10 dw=1 log10(-XD)=0\n"
        "# damping: MODE=auto  using=-T/-TV/-TD  fixed_-D=OFF"
        "  -TV_percent=20 -TV_frac=0.2  -TD_percent=20 -TD_frac=0.2\n"
        "# filter_-s: ON\n"
        "# lsqr_precond: ON maxD=50\n"
        "# accel: reuse=ON thresh=0.001 c2f=OFF legacy=OFF\n"
        "1 1 0 0.1\n",
        encoding="utf-8",
    )
    info_p = parse_tt_inverse_log_header(p_params)
    assert info_p["jumping"] is True
    assert info_p["sv_on"] is True
    assert info_p["sv_wmin"] == 200
    assert info_p["tv_percent"] == 20
    runp = format_tt_inverse_run_params(info_p)
    assert "-SV200" in runp
    assert "-SD10" in runp
    assert "-TV20%" in runp
    assert "跳跃" in runp
    assert "滤波 -s" in runp
    assert "自动阻尼 -T" not in runp or "-TV20%" in runp

    p_old = tmp_path / "old.log"
    p_old.write_text(
        "# damp_vel 1 0.2\n"
        "# smooth_vel 1 0.2 0.2 0.3 0\n",
        encoding="utf-8",
    )
    info_o = parse_tt_inverse_log_header(p_old)
    assert info_o["damping_mode"] == "auto"
    assert info_o["filter_2d"] is False


def test_inv_analysis_drop_writes_one_log_per_line(tmp_path: Path) -> None:
    import os

    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QUrl
    from PySide6.QtCore import QMimeData
    from PySide6.QtWidgets import QApplication

    from pyAOBS.modeling.tomo2d.gui.dialogs.inv_analysis_dialog import InvAnalysisDialog

    _app = QApplication.instance() or QApplication([])
    a = tmp_path / "runs" / "r1" / "outputs" / "tt_inverse.log"
    b = tmp_path / "runs" / "r2" / "outputs" / "tt_inverse.log"
    a.parent.mkdir(parents=True)
    b.parent.mkdir(parents=True)
    a.write_text("x", encoding="utf-8")
    b.write_text("y", encoding="utf-8")
    dlg = InvAnalysisDialog(FormState({"work_dir": str(tmp_path)}))
    md = QMimeData()
    md.setUrls([QUrl.fromLocalFile(str(a)), QUrl.fromLocalFile(str(b))])
    dlg.multi_edit.insertFromMimeData(md)
    lines = [ln for ln in dlg.multi_edit.toPlainText().splitlines() if ln.strip()]
    assert len(lines) == 2
    assert "r1" in lines[0] and "r2" in lines[1]
    dlg.close()


def test_pareto_figure_meta_and_hit_index() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from pyAOBS.modeling.tomo2d.gui.plots.pareto_log_menu import pareto_hit_index
    from pyAOBS.modeling.tomo2d.tt_inverse_log_analysis import (
        build_figure_multi_pareto_and_score,
    )

    def row(pred: float, r: float) -> list[list[float]]:
        v = [0.0] * 26
        v[0] = 1.0
        v[20] = pred
        v[23] = r
        return [v]

    series = {"lo": row(1.0, 0.01), "hi": row(3.0, 0.5)}
    fig = build_figure_multi_pareto_and_score(series, rough_weight=0.001)
    meta = fig._pyaobs_pareto
    assert meta["names"] == ["lo", "hi"]
    assert meta["bar_series_index"][0] == 0
    fig.canvas.draw()

    class _Evt:
        def __init__(self, ax, xdata, ydata, x, y) -> None:
            self.inaxes = ax
            self.xdata = xdata
            self.ydata = ydata
            self.x = x
            self.y = y

    ax0, ax1 = fig.axes[:2]
    offs = meta["scatter"].get_offsets()
    px, py = ax0.transData.transform(offs[1])
    hit = pareto_hit_index(
        fig, _Evt(ax0, float(offs[1, 0]), float(offs[1, 1]), float(px), float(py))
    )
    assert hit == 1
    patch = meta["bars"].patches[0]
    cx = patch.get_x() + patch.get_width() * 0.5
    cy = patch.get_y() + patch.get_height() * 0.5
    bx, by = ax1.transData.transform((cx, cy))
    hit_bar = pareto_hit_index(fig, _Evt(ax1, cx, cy, float(bx), float(by)))
    assert hit_bar == 0
    assert pareto_hit_index(fig, _Evt(ax1, 0.0, 0.0, float(bx), float(by))) == 0
    assert pareto_hit_index(fig, _Evt(None, 0.0, 0.0, 1.0, 1.0)) is None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_overlay_and_param_influence_pick_meta() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from pyAOBS.modeling.tomo2d.gui.plots.pareto_log_menu import pareto_hit_index
    from pyAOBS.modeling.tomo2d.tt_inverse_log_analysis import (
        build_figure_multi_overlay,
        build_figure_multi_param_influence,
        build_figure_multi_summary_table,
    )

    def row(pred: float, r: float, niter: int = 3) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(1, niter + 1):
            v = [0.0] * 26
            v[0] = float(i)
            v[3] = 0.2 / i
            v[13] = 80.0
            v[14] = 10.0
            v[15] = 20.0
            v[16] = 20.0
            v[20] = pred / i
            v[23] = r
            out.append(v)
        return out

    series = {"lo": row(1.0, 0.01), "hi": row(3.0, 0.5)}
    fig_o = build_figure_multi_overlay(series)
    meta_o = fig_o._pyaobs_pareto
    assert meta_o["names"] == ["lo", "hi"]
    assert len(fig_o.axes) == 3
    assert not fig_o.axes[0].get_legend()
    fig_o.canvas.draw()

    class _Evt:
        def __init__(self, ax, x, y) -> None:
            self.inaxes = ax
            self.x = x
            self.y = y
            self.xdata = None
            self.ydata = None

    ln = meta_o["lines"][0]
    xd, yd = ln.get_data()
    px, py = fig_o.axes[0].transData.transform((xd[-1], yd[-1]))
    assert pareto_hit_index(fig_o, _Evt(fig_o.axes[0], float(px), float(py))) == 0
    import matplotlib.pyplot as plt

    plt.close(fig_o)

    fig_p = build_figure_multi_param_influence(series)
    meta_p = fig_p._pyaobs_pareto
    assert meta_p["names"] == ["lo", "hi"]
    assert len(meta_p["scatters"]) == 8
    fig_p.canvas.draw()
    sc = meta_p["scatters"][0]
    offs = sc.get_offsets()
    ax = sc.axes
    qx, qy = ax.transData.transform(offs[1])
    assert pareto_hit_index(fig_p, _Evt(ax, float(qx), float(qy))) == 1
    assert ax.get_xscale() == "log"
    assert ax.xaxis.get_major_formatter()(600, 0) == "600"
    plt.close(fig_p)

    fig_t = build_figure_multi_summary_table(series, rough_weight=0.001)
    assert fig_t._pyaobs_summary_order == ["lo", "hi"]
    plt.close(fig_t)
    fig_t2 = build_figure_multi_summary_table(
        {"a": row(5.0, 1.0), "b": row(0.5, 0.01)}, rough_weight=0.001
    )
    assert fig_t2._pyaobs_summary_order == ["b", "a"]
    plt.close(fig_t2)


def test_cross_subplot_highlight_scatter_and_bars() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np

    from pyAOBS.modeling.tomo2d.gui.plots.pareto_log_menu import apply_series_highlight
    from pyAOBS.modeling.tomo2d.tt_inverse_log_analysis import (
        build_figure_multi_param_influence,
        build_figure_multi_pareto_and_score,
    )

    def row(pred: float, r: float) -> list[list[float]]:
        v = [0.0] * 26
        v[0] = 1.0
        v[13] = 80.0
        v[14] = 10.0
        v[15] = 20.0
        v[16] = 20.0
        v[20] = pred
        v[23] = r
        return [v]

    series = {"lo": row(1.0, 0.01), "hi": row(3.0, 0.5)}
    fig = build_figure_multi_pareto_and_score(series, rough_weight=0.001)
    fig.canvas.draw()
    apply_series_highlight(fig, 1)
    meta = fig._pyaobs_pareto
    assert meta["highlight_index"] == 1
    sz = np.asarray(meta["scatter"].get_sizes(), dtype=float)
    assert sz[1] > sz[0]
    mapping = list(meta["bar_series_index"])
    patches = meta["bars"].patches
    sel = patches[mapping.index(1)]
    other = patches[mapping.index(0)]
    assert sel.get_linewidth() > other.get_linewidth()
    apply_series_highlight(fig, None)
    assert meta["highlight_index"] is None
    import matplotlib.pyplot as plt

    plt.close(fig)

    fig_p = build_figure_multi_param_influence(series)
    fig_p.canvas.draw()
    apply_series_highlight(fig_p, 0)
    for sc in fig_p._pyaobs_pareto["scatters"]:
        psz = np.asarray(sc.get_sizes(), dtype=float)
        assert psz[0] > psz[1]
    apply_series_highlight(fig_p, None)
    plt.close(fig_p)


def test_highlight_group_syncs_by_name_across_figures() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np

    from pyAOBS.modeling.tomo2d.gui.plots.pareto_log_menu import SeriesHighlightGroup
    from pyAOBS.modeling.tomo2d.tt_inverse_log_analysis import (
        build_figure_multi_overlay,
        build_figure_multi_param_influence,
        build_figure_multi_pareto_and_score,
        build_figure_multi_summary_table,
    )

    def row(pred: float, r: float, niter: int = 3) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(1, niter + 1):
            v = [0.0] * 26
            v[0] = float(i)
            v[13] = 80.0
            v[14] = 10.0
            v[15] = 20.0
            v[16] = 20.0
            v[20] = pred
            v[23] = r
            out.append(v)
        return out

    series = {"lo": row(1.0, 0.01), "hi": row(3.0, 0.5)}
    figs = [
        build_figure_multi_pareto_and_score(series, rough_weight=0.001),
        build_figure_multi_param_influence(series),
        build_figure_multi_overlay(series),
        build_figure_multi_summary_table(series, rough_weight=0.001),
    ]
    for fig in figs:
        fig.canvas.draw()

    class _Win:
        def __init__(self, fig) -> None:
            self._fig = fig

    group = SeriesHighlightGroup()
    wins = [_Win(fig) for fig in figs]
    for win in wins:
        group.register(win)
    group.select_name("hi")
    assert group.selected_name == "hi"
    assert figs[0]._pyaobs_pareto["highlight_index"] == 1
    assert figs[1]._pyaobs_pareto["highlight_index"] == 1
    assert figs[2]._pyaobs_pareto["highlight_index"] == 1
    assert figs[3]._pyaobs_pareto["names"] == ["lo", "hi"]
    assert figs[3]._pyaobs_pareto["highlight_index"] == 1
    lw0, lw1 = [], []
    meta_o = figs[2]._pyaobs_pareto
    for ln, si in zip(meta_o["lines"], meta_o["line_series_index"]):
        (lw0 if si == 0 else lw1).append(ln.get_linewidth())
    assert min(lw1) > max(lw0)
    gold = np.asarray(figs[3]._pyaobs_pareto["table"][(2, 0)].get_facecolor(), dtype=float)
    pale = np.asarray(figs[3]._pyaobs_pareto["table"][(1, 0)].get_facecolor(), dtype=float)
    assert gold[0] > pale[0] or gold[1] > pale[1]
    group.select_name(None)
    assert figs[0]._pyaobs_pareto["highlight_index"] is None
    group.select_names(["lo", "hi"])
    assert group.selected_names == ["lo", "hi"]
    assert figs[0]._pyaobs_pareto["highlight_indices"] == (0, 1)
    sz = np.asarray(figs[0]._pyaobs_pareto["scatter"].get_sizes(), dtype=float)
    assert sz[0] > 1.0 and sz[1] > 1.0
    group.select_name("lo")
    group.select_range_to(figs[0], 1)
    assert group.selected_names == ["lo", "hi"]
    group.select_name("lo")
    group.toggle_from_figure(figs[0], 1)
    assert group.selected_names == ["lo", "hi"]
    group.toggle_from_figure(figs[0], 0)
    assert group.selected_names == ["hi"]
    import matplotlib.pyplot as plt

    for fig in figs:
        plt.close(fig)


def test_pareto_box_indices_and_params_table() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from pyAOBS.modeling.tomo2d.gui.plots.pareto_log_menu import (
        format_logs_params_table,
        pareto_box_indices,
    )
    from pyAOBS.modeling.tomo2d.tt_inverse_log_analysis import (
        build_figure_multi_overlay,
        build_figure_multi_pareto_and_score,
    )

    def row(pred: float, r: float, niter: int = 4) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(1, niter + 1):
            v = [0.0] * 26
            v[0] = float(i)
            v[13] = 80.0
            v[14] = 10.0
            v[15] = 20.0
            v[16] = 20.0
            v[20] = pred
            v[23] = r
            out.append(v)
        return out

    series = {"lo": row(1.0, 0.01), "hi": row(3.0, 0.5)}
    fig = build_figure_multi_pareto_and_score(series, rough_weight=0.001)
    fig.canvas.draw()
    sc = fig._pyaobs_pareto["scatter"]
    pts = sc.axes.transData.transform(sc.get_offsets())
    x0, y0 = float(pts[:, 0].min()) - 8.0, float(pts[:, 1].min()) - 8.0
    x1, y1 = float(pts[:, 0].max()) + 8.0, float(pts[:, 1].max()) + 8.0
    assert pareto_box_indices(fig, x0, y0, x1, y1) == [0, 1]
    px, py = float(pts[1, 0]), float(pts[1, 1])
    assert pareto_box_indices(fig, px - 4, py - 4, px + 4, py + 4) == [1]
    assert pareto_box_indices(fig, px, py, px + 1, py + 1) == []
    import matplotlib.pyplot as plt

    plt.close(fig)

    fig_o = build_figure_multi_overlay(series)
    fig_o.canvas.draw()
    ln = fig_o._pyaobs_pareto["lines"][0]
    xd, yd = ln.get_data()
    pts_l = ln.axes.transData.transform(
        __import__("numpy").column_stack([xd, yd])
    )
    bx0, by0 = float(pts_l[:, 0].min()) - 6, float(pts_l[:, 1].min()) - 6
    bx1, by1 = float(pts_l[:, 0].max()) + 6, float(pts_l[:, 1].max()) + 6
    assert 0 in pareto_box_indices(fig_o, bx0, by0, bx1, by1)
    plt.close(fig_o)

    text = format_logs_params_table(
        [("lo", None, series["lo"]), ("hi", None, series["hi"])], 0.001
    )
    assert "标签" in text and "score" in text
    assert "lo" in text and "hi" in text


def test_summary_table_ranked_and_overlay_data() -> None:
    from pyAOBS.modeling.tomo2d.tt_inverse_log_analysis import (
        overlay_curve_series,
        pareto_score_data,
        summary_table_ranked,
    )

    def row(pred: float, r: float, niter: int = 3) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(1, niter + 1):
            v = [0.0] * 26
            v[0] = float(i)
            v[20] = pred
            v[23] = r
            out.append(v)
        return out

    series = {"lo": row(1.0, 0.01), "hi": row(3.0, 0.5)}
    ranked = summary_table_ranked(series, 0.001)
    assert [nm for _s, nm, _m in ranked] == ["lo", "hi"]
    ranked2 = summary_table_ranked({"a": row(5.0, 1.0), "b": row(0.5, 0.01)}, 0.001)
    assert [nm for _s, nm, _m in ranked2] == ["b", "a"]
    ov = overlay_curve_series(series)
    assert [c["name"] for c in ov] == ["lo", "hi"]
    pd = pareto_score_data(series, 0.001)
    assert pd["names"] == ["lo", "hi"]
    assert pd["bar_order"][0] == 0


def test_analysis_pg_windows_highlight_and_box() -> None:
    import os

    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QRectF
    from PySide6.QtWidgets import QApplication

    from pyAOBS.modeling.tomo2d.gui.plots.inv_analysis_pg import (
        show_overlay_window,
        show_pareto_window,
        show_summary_table_window,
    )
    from pyAOBS.modeling.tomo2d.gui.plots.pareto_log_menu import SeriesHighlightGroup

    _app = QApplication.instance() or QApplication([])

    def row(pred: float, r: float, niter: int = 3) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(1, niter + 1):
            v = [0.0] * 26
            v[0] = float(i)
            v[6] = 0.2 / i
            v[9] = 0.3 / i
            v[20] = pred
            v[23] = r
            out.append(v)
        return out

    series = {"lo": row(1.0, 0.01), "hi": row(3.0, 0.5)}
    ov = show_overlay_window(series, title="t")
    pa = show_pareto_window(series, rough_weight=0.001, title="p")
    tb = show_summary_table_window(series, rough_weight=0.001, title="tab")
    assert ov.names == ["lo", "hi"]
    assert pa.names == ["lo", "hi"]
    assert tb.names == ["lo", "hi"]
    assert tb.table.columnWidth(0) >= 200
    assert tb.table.columnWidth(0) <= 560
    group = SeriesHighlightGroup()
    for w in (ov, pa, tb):
        group.register(w)
    group.select_name("hi")
    assert group.selected_name == "hi"
    rec = next(r for r in ov._pickers if r["kind"] == "line" and r["series"] == 1)
    rec0 = next(r for r in ov._pickers if r["kind"] == "line" and r["series"] == 0)
    assert rec["item"].opts["pen"].widthF() > rec0["item"].opts["pen"].widthF()
    sc = next(r for r in pa._pickers if r["kind"] == "scatter")
    vb = sc["vb"]
    xs, ys = sc["x"], sc["y"]
    pad = 0.2
    box = QRectF(
        float(min(xs)) - pad,
        float(min(ys)) - pad,
        float(max(xs) - min(xs)) + 2 * pad,
        float(max(ys) - min(ys)) + 2 * pad,
    )
    assert pa.box_indices(vb, box) == [0, 1]
    ov.close()
    pa.close()
    tb.close()


def test_analysis_pg_scatter_hit_uses_spot_scene_pos() -> None:
    import os

    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtWidgets import QApplication

    from pyAOBS.modeling.tomo2d.gui.plots.inv_analysis_pg import (
        show_overlay_window,
        show_param_influence_window,
        show_pareto_window,
    )

    _app = QApplication.instance() or QApplication([])

    def row(pred: float, r: float, niter: int = 3) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(1, niter + 1):
            v = [0.0] * 26
            v[0] = float(i)
            v[6] = (0.2 if pred < 2 else 0.8) / i
            v[9] = (0.3 if pred < 2 else 0.9) / i
            v[13] = 0.01 if pred < 2 else 0.1
            v[14] = 0.02 if pred < 2 else 0.2
            v[15] = 0.03 if pred < 2 else 0.3
            v[16] = 0.04 if pred < 2 else 0.4
            v[20] = pred
            v[23] = r
            out.append(v)
        return out

    series = {"lo": row(1.0, 0.01), "hi": row(3.0, 0.5)}
    pa = show_pareto_window(series, rough_weight=0.001, title="p")
    ov = show_overlay_window(series, title="t")
    infl = show_param_influence_window(series, title="i")
    pa.show()
    ov.show()
    infl.show()
    _app.processEvents()

    class _Ev:
        def __init__(self, sp: QPointF) -> None:
            self._sp = sp

        def scenePos(self):
            return self._sp

        def button(self):
            return Qt.MouseButton.LeftButton

        def modifiers(self):
            return Qt.KeyboardModifier.NoModifier

        def accept(self) -> None:
            self.accepted = True

        def ignore(self) -> None:
            self.accepted = False

        def isAccepted(self) -> bool:
            return bool(getattr(self, "accepted", False))

    rec = next(r for r in pa._pickers if r["kind"] == "scatter")
    item = rec["item"]
    pts = item.points()
    assert len(pts) >= 1
    sp0 = item.mapToScene(pts[0].pos())
    assert pa.hit_index(rec["vb"], _Ev(sp0)) == 0
    picked: list = []
    pa.user_pick = lambda kind, **kw: picked.append((kind, kw.get("index")))
    ev = _Ev(sp0)
    item.mouseClickEvent(ev)
    assert picked and picked[-1] == ("click", 0)

    line = next(r for r in ov._pickers if r["kind"] == "line" and r["series"] == 1)
    sc = line["item"].scatter
    lpts = sc.points()
    assert len(lpts) >= 1
    lsp = sc.mapToScene(lpts[0].pos())
    assert ov.hit_index(line["vb"], _Ev(lsp)) == 1

    irec = next(r for r in infl._pickers if r["kind"] == "scatter")
    ipts = irec["item"].points()
    assert len(ipts) >= 1
    isp = irec["item"].mapToScene(ipts[0].pos())
    assert infl.hit_index(irec["vb"], _Ev(isp)) == 0

    ov.close()
    pa.close()
    infl.close()


def test_score_bar_shift_range_follows_visual_order() -> None:
    import os

    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from pyAOBS.modeling.tomo2d.gui.plots.inv_analysis_pg import show_pareto_window
    from pyAOBS.modeling.tomo2d.gui.plots.pareto_log_menu import SeriesHighlightGroup

    _app = QApplication.instance() or QApplication([])

    def row(pred: float, r: float) -> list[list[float]]:
        v = [0.0] * 26
        v[0] = 1.0
        v[20] = pred
        v[23] = r
        return [v]

    # names 顺序 a,b,c；得分顺序 b < c < a，条形从上到下是 b,c,a
    series = {
        "a": row(3.0, 0.5),
        "b": row(0.5, 0.01),
        "c": row(1.5, 0.1),
    }
    pa = show_pareto_window(series, rough_weight=0.001, title="p")
    bar = next(r for r in pa._pickers if r["kind"] == "bar")
    vis = [pa.names[int(j)] for j in bar["index_map"]]
    assert vis == ["b", "c", "a"]
    group = SeriesHighlightGroup()
    group.register(pa)
    group.select_name("b")
    group.select_range_to(pa, pa.names.index("a"), visual_order=vis)
    assert group.selected_names == ["b", "c", "a"]
    group.select_name("b")
    group.select_range_to(pa, pa.names.index("a"))
    assert "c" not in group.selected_names
    pa.close()


def test_mpl_nav_claim_accepts_event_or_noarg() -> None:
    from pyAOBS.utils.mpl_plot_nav import PyqtgraphStyleNav

    assert PyqtgraphStyleNav._claim(None, object()) is False
    assert PyqtgraphStyleNav._claim(lambda: True, object()) is True
    assert PyqtgraphStyleNav._claim(lambda e: e == "hit", "hit") is True
    assert PyqtgraphStyleNav._claim(lambda e: e == "hit", "miss") is False


def test_mpl_nav_log_zoom_and_pan_stay_positive() -> None:
    from pyAOBS.utils.mpl_plot_nav import PyqtgraphStyleNav

    lo, hi = PyqtgraphStyleNav._zoom_span(0.1, 1.0, 0.15, 10.0, log_scale=False)
    assert lo < 0
    nlo, nhi = PyqtgraphStyleNav._zoom_span(0.1, 1.0, 0.15, 10.0, log_scale=True)
    assert nlo > 0 and nhi > nlo
    plo, phi = PyqtgraphStyleNav._pan_span(0.1, 10.0, 0.8, log_scale=True)
    assert plo > 0 and phi > plo
    qlo, qhi = PyqtgraphStyleNav._pan_span(0.1, 10.0, 0.8, log_scale=False)
    assert qlo < 0


def test_file_dialog_options_linux_skips_native(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QFileDialog

    from pyAOBS.modeling.tomo2d.gui import dialog_utils as du

    monkeypatch.setattr(du.sys, "platform", "linux")
    opts = du.file_dialog_options()
    assert bool(opts & QFileDialog.Option.DontUseNativeDialog)
    monkeypatch.setattr(du.sys, "platform", "win32")
    opts_win = du.file_dialog_options()
    assert not bool(opts_win & QFileDialog.Option.DontUseNativeDialog)





