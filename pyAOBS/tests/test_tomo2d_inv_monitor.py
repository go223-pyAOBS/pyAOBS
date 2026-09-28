"""反演监视服务（日志定位 / 快照 / 模型目录）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyAOBS.modeling.tomo2d.gui.services.inv_monitor import (
    build_model_catalog,
    build_monitor_spec_from_paths,
    collect_monitor_snapshot,
    format_monitor_status,
    progress_fraction,
)
from pyAOBS.modeling.tomo2d.gui.services.smesh_ops import (
    find_latest_inverse_smesh,
    list_inverse_smesh_files,
)

pytestmark = pytest.mark.unit


def test_build_monitor_spec_log_candidates(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "outputs" / "logs").mkdir(parents=True)
    log = run / "outputs" / "logs" / "tt_inverse.log"
    log.write_text(
        "# hdr\n"
        "1 0 0 0.1 1.5 10 0.1 1.0 0 0 0 0 0 1 1 1 1 1 1 0.1 1.2 0.01 0.01 0.1 0.1 0.1\n",
        encoding="utf-8",
    )
    spec = build_monitor_spec_from_paths(
        cwd=run,
        log_file="outputs/tt_inverse.log",
        out_root="outputs/out",
        niter=5,
        run_dir=run,
    )
    assert spec.resolve_log() == log
    assert spec.niter == 5


def test_parse_status_jsonl_and_snapshot(tmp_path: Path) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.inv_monitor import parse_status_jsonl

    run = tmp_path / "run"
    (run / "outputs").mkdir(parents=True)
    st = run / "outputs" / "status.jsonl"
    st.write_text(
        '{"iter":1,"iset":0,"rms":0.2,"chi2":2.0,"pred_chi":1.8,"rough_v":0.3}\n'
        '{"iter":2,"iset":1,"rms":0.1,"chi2":1.1,"pred_chi":1.0,"rough_v":0.2}\n',
        encoding="utf-8",
    )
    rows = parse_status_jsonl(st)
    assert len(rows) == 2
    assert rows[-1]["iter"] == 2
    spec = build_monitor_spec_from_paths(
        cwd=run,
        log_file="outputs/missing.log",
        out_root="outputs/out",
        status_jsonl="outputs/status.jsonl",
    )
    snap = collect_monitor_snapshot(spec)
    assert snap.status_n == 2
    assert snap.last_iter == 2
    assert snap.last_chi == pytest.approx(1.1)
    assert snap.n_rows == 2  # 无 -L 时用 jsonl 画曲线


def test_collect_snapshot_and_latest_smesh(tmp_path: Path) -> None:
    run = tmp_path / "run"
    od = run / "outputs"
    od.mkdir(parents=True)
    (od / "tt_inverse.log").write_text(
        "1 0 0 0.2 2.0 10 0.2 2.0 0 0 0 0 0 1 1 1 1 1 1 0.1 1.8 0.01 0.01 0.2 0.2 0.1\n"
        "2 1 0 0.15 1.2 10 0.15 1.2 0 0 0 0 0 1 1 1 1 1 1 0.1 1.1 0.01 0.01 0.15 0.15 0.1\n",
        encoding="utf-8",
    )
    (od / "models").mkdir()
    (od / "models" / "out.smesh.1.0").write_text("a", encoding="utf-8")
    (od / "models" / "out.smesh.2.1").write_text("b", encoding="utf-8")
    spec = build_monitor_spec_from_paths(
        cwd=run,
        log_file="outputs/tt_inverse.log",
        out_root="outputs/out",
        niter=5,
    )
    snap = collect_monitor_snapshot(spec)
    assert snap.n_rows == 2
    assert snap.ray_stamp == ""
    assert isinstance(snap.tres_stamp, str)
    snap_r = collect_monitor_snapshot(spec, include_rays=True)
    assert snap_r.n_rows == 2
    assert snap_r.smesh_path is not None
    assert snap.last_iter == 2
    assert snap.last_iset == 1
    assert snap.last_chi == pytest.approx(1.2)
    assert snap.smesh_path is not None
    assert snap.smesh_path.name == "out.smesh.2.1"
    st = format_monitor_status(snap, niter=5)
    assert "iter 2/5" in st
    assert find_latest_inverse_smesh(od / "out").name == "out.smesh.2.1"
    cur, mx = progress_fraction(snap, niter=5)
    assert (cur, mx) == (2, 5)

    listed = list_inverse_smesh_files(od / "out")
    assert [p.name for p, _, _ in listed] == ["out.smesh.1.0", "out.smesh.2.1"]
    cat = build_model_catalog(spec)
    assert len(cat) == 2
    assert cat[0].iter == 1 and cat[0].chi2 == pytest.approx(2.0)
    assert cat[0].w_sv == pytest.approx(1.0)
    assert cat[-1].iter == 2 and cat[-1].rms == pytest.approx(0.15)

    from pyAOBS.modeling.tomo2d.gui.services.smesh_ops import companion_inverse_refl

    (od / "models" / "out.refl.2.1").write_text("x z\n", encoding="utf-8")
    hit = companion_inverse_refl(od / "models" / "out.smesh.2.1")
    assert hit is not None and hit.name == "out.refl.2.1"
    assert companion_inverse_refl(od / "models" / "out.smesh.1.0") is None


def test_collect_run_inversion_params_from_log_and_manifest(tmp_path: Path) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.inv_monitor import (
        collect_run_inversion_params,
    )

    run = tmp_path / "run"
    (run / "outputs" / "logs").mkdir(parents=True)
    (run / "outputs" / "logs" / "tt_inverse.log").write_text(
        "# strategy jumping=1 robust=0 crit_chi=0\n"
        "# smooth_vel -SV on=1 wmin=200 wmax=200 dw=1 log10(-XV)=0\n"
        "# smooth_dep -SD on=1 wmin=10 wmax=10 dw=1 log10(-XD)=0\n"
        "# damping: MODE=auto  using=-T/-TV/-TD  fixed_-D=OFF"
        "  -TV_percent=20 -TV_frac=0.2  -TD_percent=20 -TD_frac=0.2\n"
        "# filter_-s: ON\n"
        "1 1 0 0.1\n",
        encoding="utf-8",
    )
    (run / "manifest.json").write_text(
        '{"python_replay":{"kwargs":{"niter":8,"target_chi2":1.0,'
        '"smooth_opts":{"corr_v_fn":"inputs/vcorr_vpfd41.dat"}}}}',
        encoding="utf-8",
    )
    spec = build_monitor_spec_from_paths(
        cwd=run,
        log_file="outputs/tt_inverse.log",
        out_root="outputs/out",
        run_dir=run,
    )
    txt = collect_run_inversion_params(spec)
    assert "-SV200" in txt
    assert "-TV20%" in txt
    assert "-I8" in txt
    assert "-J1" in txt
    assert "vcorr_vpfd41.dat" in txt


def test_pg_velocity_lut_uint8() -> None:
    pytest.importorskip("pyqtgraph")
    from pyAOBS.visualization.pg_velocity import colormap_from_spec, lut_uint8

    cmap, lv = colormap_from_spec("jet")
    assert lv is None
    lut = lut_uint8(cmap)
    assert lut.shape[1] == 3
    assert lut.dtype == np.uint8
    assert lut.max() > 200
    assert lut.min() < 40


def test_monitor_model_imshow_is_colored() -> None:
    """监视窗速度场走 matplotlib imshow，两端映射色不同。"""
    import os

    pytest.importorskip("PySide6")
    pytest.importorskip("matplotlib")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from pyAOBS.modeling.tomo2d.gui.plots.inv_monitor_model import MonitorModelWidget

    _app = QApplication.instance() or QApplication([])

    class _V:
        def __init__(self, v):
            self.values = v

    z = np.linspace(0.0, 8.0, 16)
    x = np.linspace(0.0, 20.0, 24)
    vel = np.linspace(1.5, 8.0, 16)[:, None] + np.zeros((16, 24))
    ds = {"velocity": _V(vel), "x": _V(x), "z": _V(z)}
    w = MonitorModelWidget()
    xmin, xmax = w.set_velocity(ds, None, None, "jet", "test")
    assert xmin == 0.0 and xmax == 20.0
    images = w.ax_mesh.get_images()
    assert images
    im = images[0]
    arr = np.asarray(im.get_array(), dtype=float)
    c0 = np.asarray(im.cmap(im.norm(arr[0, 0])))[:3]
    c1 = np.asarray(im.cmap(im.norm(arr[-1, -1])))[:3]
    assert not np.allclose(c0, c1)
    w.close()


def test_monitor_compare_stack_has_three_velocity_panels() -> None:
    """对比模式自上而下：ΔV、B、A 三块速度图；切回单模型后恢复拟合布局。"""
    import os

    pytest.importorskip("PySide6")
    pytest.importorskip("matplotlib")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from pyAOBS.modeling.tomo2d.gui.plots.inv_monitor_model import MonitorModelWidget

    _app = QApplication.instance() or QApplication([])

    class _V:
        def __init__(self, v):
            self.values = v

    z = np.linspace(0.0, 8.0, 16)
    x = np.linspace(0.0, 20.0, 24)
    vel_a = np.full((16, 24), 4.0)
    vel_b = np.full((16, 24), 4.5)
    ds_a = {"velocity": _V(vel_a), "x": _V(x), "z": _V(z)}
    ds_b = {"velocity": _V(vel_b), "x": _V(x), "z": _V(z)}
    ds_d = {"velocity": _V(vel_b - vel_a), "x": _V(x), "z": _V(z)}
    w = MonitorModelWidget()
    w.set_velocity_stack(
        [
            {
                "ds": ds_d,
                "cmap_spec": "seismic_r",
                "title": "ΔV",
                "vlim": (-1.0, 1.0),
                "cb_label": "ΔV (km/s)",
            },
            {"ds": ds_b, "cmap_spec": "jet", "title": "B"},
            {"ds": ds_a, "cmap_spec": "jet", "title": "A"},
        ]
    )
    assert w._layout_mode == "compare"
    assert w.ax_diff.get_images() and w.ax_b.get_images() and w.ax_a.get_images()
    assert "ΔV" in (w.ax_diff.get_title() or "")
    assert w.ax_b.get_title().startswith("B")
    assert w.ax_a.get_title().startswith("A")
    w.set_velocity(ds_b, None, None, "jet", "test")
    assert w._layout_mode == "fit"
    assert w.ax_fit_refr is not None
    assert w.ax_mesh.get_images()
    w.close()


def test_smesh_plot_imshow_is_colored() -> None:
    """绘制 smesh 与监视窗同走 matplotlib imshow，两端映射色不同。"""
    import os

    pytest.importorskip("matplotlib")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pyAOBS.modeling.tomo2d.gui.dialogs.smesh_plot import draw_smesh_velocity_figure

    class _V:
        def __init__(self, v):
            self.values = v

    z = np.linspace(0.0, 8.0, 16)
    x = np.linspace(0.0, 20.0, 24)
    vel = np.linspace(1.5, 8.0, 16)[:, None] + np.zeros((16, 24))
    ds = {"velocity": _V(vel), "x": _V(x), "z": _V(z)}
    fig = draw_smesh_velocity_figure(ds, None, None, "jet", "test")
    ax = fig.axes[0]
    images = ax.get_images()
    assert images
    im = images[0]
    arr = np.asarray(im.get_array(), dtype=float)
    c0 = np.asarray(im.cmap(im.norm(arr[0, 0])))[:3]
    c1 = np.asarray(im.cmap(im.norm(arr[-1, -1])))[:3]
    assert not np.allclose(c0, c1)
    fig.clear()


def test_smesh_plot_overlays_sampled_rays() -> None:
    pytest.importorskip("matplotlib")
    from pyAOBS.modeling.tomo2d.gui.dialogs.smesh_plot import draw_smesh_velocity_figure

    class _V:
        def __init__(self, v):
            self.values = v

    z = np.linspace(0.0, 8.0, 8)
    x = np.linspace(0.0, 20.0, 12)
    vel = np.linspace(1.5, 8.0, 8)[:, None] + np.zeros((8, 12))
    ds = {"velocity": _V(vel), "x": _V(x), "z": _V(z)}
    groups = [
        (1, [([0.0, 5.0, 10.0], [0.5, 2.0, 0.5])]),
        (2, [([2.0, 8.0], [0.2, 1.5])]),
    ]
    fig = draw_smesh_velocity_figure(
        ds, None, None, "jet", "test", ray_groups=groups
    )
    ax = fig.axes[0]
    n_before = len(ax.lines)
    fig2 = draw_smesh_velocity_figure(ds, None, None, "jet", "test")
    assert len(ax.lines) >= 2
    assert len(ax.lines) > len(fig2.axes[0].lines)
    fig.clear()
    fig2.clear()
    assert n_before >= 2


def test_smesh_plot_overlays_gmt_contours() -> None:
    """A 线有标注，C 线无标注。"""
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    from pyAOBS.modeling.tomo2d.gui.plots.velocity_contours import (
        DEFAULT_VP_CONTOURS,
        overlay_velocity_contours,
        parse_gmt_contour_text,
    )

    by_v = {s.value: s.annotate for s in DEFAULT_VP_CONTOURS}
    assert by_v[2.5] is False
    assert by_v[5.1] is False
    assert by_v[6.0] is True
    assert by_v[6.8] is True
    assert by_v[7.0] is True

    z = np.linspace(0.0, 8.0, 16)
    x = np.linspace(0.0, 20.0, 24)
    vel = np.linspace(1.5, 8.0, 16)[:, None] + np.zeros((16, 24))
    specs = parse_gmt_contour_text("4.0 A\n5.0 C\n6.0 A\n")
    fig = Figure()
    ax = fig.add_subplot(111)
    cs_c, cs_a = overlay_velocity_contours(ax, vel, x, z, specs)
    assert cs_c is not None and list(cs_c.levels) == [5.0]
    assert cs_a is not None and list(cs_a.levels) == [4.0, 6.0]
    labels = [t.get_text() for t in ax.texts]
    assert any("4" in t for t in labels)
    assert any("6" in t for t in labels)
    assert not any(t.strip() in {"5", "5.0"} for t in labels)
    fig.clear()


def test_velocity_as_nz_nx_square_grid_not_transposed() -> None:
    """nx==nz 时速度已是 (z, x)，不能再转置，否则等值线会竖过来。"""
    from pyAOBS.modeling.tomo2d.gui.plots.velocity_contours import (
        overlay_velocity_contours,
        parse_gmt_contour_text,
        velocity_as_nz_nx,
    )

    n = 8
    x = np.linspace(0.0, 10.0, n)
    z = np.linspace(0.0, 4.0, n)
    vel = np.linspace(1.40, 1.54, n)[:, None] + np.zeros((n, n))
    arr, _xx, _zz = velocity_as_nz_nx(vel, x, z)
    assert arr.shape == (n, n)
    np.testing.assert_allclose(arr[:, 0], vel[:, 0])
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    fig = Figure()
    ax = fig.add_subplot(111)
    _c, cs_a = overlay_velocity_contours(
        ax, vel, x, z, parse_gmt_contour_text("1.46 A\n1.50 A\n")
    )
    assert cs_a is not None
    for segs in cs_a.allsegs:
        for seg in segs:
            if len(seg) >= 4:
                assert float(np.ptp(seg[:, 1])) < 0.15
    fig.clear()


def test_auto_contour_specs_for_sigma_range() -> None:
    from pyAOBS.modeling.tomo2d.gui.plots.velocity_contours import auto_contour_specs

    specs = auto_contour_specs(0.0, 0.25)
    vals = [s.value for s in specs]
    assert vals
    assert all(0.0 < v < 0.25 for v in vals)
    assert any(s.annotate for s in specs)
    assert auto_contour_specs(1.0, 1.0) == []
    assert auto_contour_specs(0.0, -1.0) == []


def test_contour_specs_for_state() -> None:
    from pyAOBS.modeling.tomo2d.gui.plots.velocity_contours import (
        DEFAULT_VP_CONTOURS,
        DEFAULT_VPVS_CONTOURS,
        DEFAULT_VS_CONTOURS,
        DEFAULT_WATER_CONTOURS,
        contour_specs_for_state,
        contours_enabled,
        contours_for_cmap_id,
        set_contours_enabled,
    )
    from pyAOBS.modeling.tomo2d.gui.state.form_state import FormState

    assert contours_for_cmap_id("vp") is DEFAULT_VP_CONTOURS
    assert contours_for_cmap_id("vs") is DEFAULT_VS_CONTOURS
    assert contours_for_cmap_id("vpvs") is DEFAULT_VPVS_CONTOURS
    assert contours_for_cmap_id("water") is DEFAULT_WATER_CONTOURS
    water_vals = [s.value for s in DEFAULT_WATER_CONTOURS]
    assert water_vals[0] == pytest.approx(1.35)
    assert water_vals[-1] == pytest.approx(1.65)
    assert all(
        abs(water_vals[i + 1] - water_vals[i] - 0.01) < 1e-9
        for i in range(len(water_vals) - 1)
    )
    assert {s.value for s in DEFAULT_VS_CONTOURS} >= {1.5, 2.5, 5.0}
    assert {s.value for s in DEFAULT_VPVS_CONTOURS} == {
        1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0
    }

    st = FormState()
    assert contours_enabled(st) is True
    assert [s.value for s in contour_specs_for_state(st)] == [
        s.value for s in DEFAULT_VP_CONTOURS
    ]
    st.set("gui.plot_smesh_cmap", "vs")
    assert [s.value for s in contour_specs_for_state(st)] == [
        s.value for s in DEFAULT_VS_CONTOURS
    ]
    st.set("gui.plot_smesh_cmap", "vpvs")
    assert [s.value for s in contour_specs_for_state(st)] == [
        s.value for s in DEFAULT_VPVS_CONTOURS
    ]
    st.set("gui.plot_smesh_cmap", "water")
    assert [s.value for s in contour_specs_for_state(st)] == [
        s.value for s in DEFAULT_WATER_CONTOURS
    ]
    set_contours_enabled(st, False)
    assert contours_enabled(st) is False
    assert contour_specs_for_state(st) == []
    set_contours_enabled(st, True)
    assert contours_enabled(st) is True


def test_dws_mask_and_resolve(tmp_path: Path) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.dws_plot import (
        dws_mask_enabled,
        load_dws_xyz,
        mask_velocity_with_dws,
        resolve_plot_dws_for_smesh,
        set_dws_mask_enabled,
    )
    from pyAOBS.modeling.tomo2d.gui.state.form_state import FormState

    st = FormState()
    assert dws_mask_enabled(st) is True
    set_dws_mask_enabled(st, False)
    assert dws_mask_enabled(st) is False
    set_dws_mask_enabled(st, True)

    models = tmp_path / "outputs" / "models"
    dws_dir = tmp_path / "outputs" / "dws"
    models.mkdir(parents=True)
    dws_dir.mkdir(parents=True)
    smesh = models / "out.smesh.1.1"
    smesh.write_text("x", encoding="utf-8")
    dws = dws_dir / "dws.dat"
    dws.write_text("0 1 8\n1 1 0\n2 1 3\n", encoding="utf-8")
    hit = resolve_plot_dws_for_smesh(smesh, FormState(), tmp_path)
    assert hit is not None and hit.name == "dws.dat"
    xyz = load_dws_xyz(hit)
    assert xyz is not None and xyz.shape == (3, 3)

    from pyAOBS.modeling.tomo2d.gui.services.dws_plot import (
        dws_xyz_for_explicit,
        dws_xyz_for_plot,
        get_explicit_dws_path,
        set_explicit_dws_path,
    )

    st2 = FormState()
    assert get_explicit_dws_path(st2, tmp_path) is None
    assert dws_xyz_for_explicit(st2, None) is None
    auto = dws_xyz_for_plot(st2, tmp_path, smesh)
    assert auto is not None
    from pyAOBS.modeling.tomo2d.gui.services.dws_plot import (
        dws_watch_stamp,
        load_plot_dws,
    )

    isolated = tmp_path / "isolated"
    run = isolated / "runs" / "ttinv_x"
    run_dws = run / "outputs" / "dws"
    run_dws.mkdir(parents=True)
    elsewhere = isolated / "inputs" / "mesh.smesh"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text("m", encoding="utf-8")
    (run_dws / "cover.dat").write_text("0 1 1\n1 1 0\n2 1 2\n", encoding="utf-8")
    # 初始网格在 inputs/，DWS 在运行包 — 须靠 run_dir 找到
    assert resolve_plot_dws_for_smesh(elsewhere, FormState(), isolated) is None
    hit_run = resolve_plot_dws_for_smesh(
        elsewhere, FormState(), isolated, run_dir=run
    )
    assert hit_run is not None and hit_run.name == "cover.dat"
    later = isolated / "runs" / "ttinv_y" / "outputs" / "dws"
    later.mkdir(parents=True)
    stamp1 = dws_watch_stamp(elsewhere, isolated, run_dir=isolated / "runs" / "ttinv_y")
    (later / "dws.dat").write_text("0 1 1\n1 1 0\n2 1 2\n", encoding="utf-8")
    stamp2 = dws_watch_stamp(elsewhere, isolated, run_dir=isolated / "runs" / "ttinv_y")
    assert stamp2 != stamp1
    xyz_off, p_off, _ = load_plot_dws(
        st2, tmp_path, smesh, enabled=False
    )
    assert xyz_off is None and p_off is None
    xyz_on, p_on, note_on = load_plot_dws(
        st2, tmp_path, smesh, enabled=True
    )
    assert xyz_on is not None and p_on is not None and note_on == ""
    set_explicit_dws_path(st2, dws, tmp_path)
    exp = get_explicit_dws_path(st2, tmp_path)
    assert exp is not None and exp.name == "dws.dat"
    assert dws_xyz_for_explicit(st2, exp) is not None
    set_dws_mask_enabled(st2, False)
    assert dws_xyz_for_explicit(st2, exp) is None
    assert dws_xyz_for_plot(st2, tmp_path, smesh) is None

    # 蒙特卡洛：smesh 在 reals/iii/out/models，DWS 在 reals/iii/dws
    mc_real = tmp_path / "runs" / "montecarlo_x" / "reals" / "000"
    mc_smesh = mc_real / "out" / "models" / "out.smesh.2.1"
    mc_smesh.parent.mkdir(parents=True)
    mc_smesh.write_text("m", encoding="utf-8")
    (mc_real / "dws").mkdir(parents=True)
    (mc_real / "dws" / "dws.dat").write_text(
        "0 1 4\n1 1 0\n2 1 5\n", encoding="utf-8"
    )
    hit_mc = resolve_plot_dws_for_smesh(mc_smesh, FormState(), tmp_path)
    assert hit_mc is not None and hit_mc.parent.name == "dws"
    assert hit_mc.parent.parent.name == "000"

    class _Mesh:
        xpos = np.array([0.0, 1.0, 2.0])
        topo = np.array([0.2, 0.2, 0.2])

    x = np.array([0.0, 1.0, 2.0])
    z = np.array([0.0, 1.0])
    vel = np.ones((2, 3), dtype=float) * 4.0
    masked = mask_velocity_with_dws(vel, x, z, xyz, mesh=_Mesh())
    assert np.isfinite(masked[0, 0])  # 水层 z=0 < topo
    assert np.isfinite(masked[1, 0])  # DWS=8
    assert not np.isfinite(masked[1, 1])  # DWS=0

    class _MeshNodes:
        xpos = np.array([0.0, 1.0, 2.0])
        zpos = np.array([0.0, 1.0])
        topo = np.array([0.2, 0.2, 0.2])

    # printMaskGrid：i 外 k 内，z = zpos+topo
    node_xyz = np.array(
        [
            [0.0, 0.2, 8.0],
            [0.0, 1.2, 8.0],
            [1.0, 0.2, 0.0],
            [1.0, 1.2, 0.0],
            [2.0, 0.2, 3.0],
            [2.0, 1.2, 3.0],
        ]
    )
    masked2 = mask_velocity_with_dws(vel, x, z, node_xyz, mesh=_MeshNodes())
    assert np.isfinite(masked2[0, 0])
    assert not np.isfinite(masked2[1, 1])

    from pyAOBS.modeling.tomo2d.gui.services.dws_plot import alpha_from_dws

    a = alpha_from_dws(
        np.array([[np.inf, 0.0, 1.0, 100.0, 1e4]], dtype=float)
    )
    assert a[0, 0] == 1.0
    assert a[0, 1] == 0.0
    assert 0.0 < a[0, 2] < a[0, 3] <= a[0, 4] <= 1.0

    from pyAOBS.modeling.tomo2d.gui.services.dws_plot import looks_like_dws_name

    assert looks_like_dws_name("dws.dat")
    assert looks_like_dws_name("outputs/dws/dws.dat")
    assert looks_like_dws_name("grav_dws.dat")
    assert not looks_like_dws_name("out.smesh.1.1")

    other = tmp_path / "coverage.xyz"
    other.write_text("# hdr\n0,1,8\n1,1,0\n2,1,3\n", encoding="utf-8")
    xyz2 = load_dws_xyz(other)
    assert xyz2 is not None and xyz2.shape == (3, 3)


def test_dws_resolve_same_dir_beats_form_and_per_run(tmp_path: Path) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.dws_plot import (
        describe_dws_for_smeshes,
        format_dws_match_report,
        resolve_plot_dws_for_smesh,
    )
    from pyAOBS.modeling.tomo2d.gui.state.form_state import FormState

    xyz = "0 1 1\n1 1 0\n2 1 2\n"
    picked = tmp_path / "user_models"
    picked.mkdir()
    smesh = picked / "mine.smesh"
    smesh.write_text("x", encoding="utf-8")
    local = picked / "dws.dat"
    local.write_text(xyz, encoding="utf-8")
    form_dws = tmp_path / "outputs" / "dws" / "dws.dat"
    form_dws.parent.mkdir(parents=True)
    form_dws.write_text("0 1 9\n1 1 9\n2 1 9\n", encoding="utf-8")
    st = FormState()
    st.set("inv.dws_file", str(form_dws))
    hit = resolve_plot_dws_for_smesh(smesh, st, tmp_path)
    assert hit is not None and hit.resolve() == local.resolve()

    def _run_smesh(run: Path) -> Path:
        models = run / "outputs" / "models"
        ddir = run / "outputs" / "dws"
        models.mkdir(parents=True)
        ddir.mkdir(parents=True)
        s = models / "out.smesh.1.1"
        s.write_text("x", encoding="utf-8")
        (ddir / "dws.dat").write_text(xyz, encoding="utf-8")
        return s

    sa = _run_smesh(tmp_path / "runs" / "packA")
    sb = _run_smesh(tmp_path / "runs" / "packB")
    st2 = FormState()
    st2.set("inv.dws_file", str(tmp_path / "runs" / "packA" / "outputs" / "dws" / "dws.dat"))
    hb = resolve_plot_dws_for_smesh(sb, st2, tmp_path)
    assert hb is not None
    assert hb.resolve() == (tmp_path / "runs" / "packB" / "outputs" / "dws" / "dws.dat").resolve()

    pairs = describe_dws_for_smeshes([sa, sb, smesh], st2, tmp_path)
    assert [p[1] is not None for p in pairs] == [True, True, True]
    assert pairs[0][1].resolve() != pairs[1][1].resolve()
    assert pairs[2][1].resolve() == local.resolve()
    text = format_dws_match_report(pairs, work=tmp_path)
    assert "3 个不同文件" in text
    assert "未找到" not in text


def test_union_dws_xyz_for_smeshes(tmp_path: Path) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.dws_plot import union_dws_xyz_for_smeshes
    from pyAOBS.modeling.tomo2d.gui.state.form_state import FormState

    def _smesh_and_dws(run: Path, vals: list[float]) -> Path:
        models = run / "outputs" / "models"
        ddir = run / "outputs" / "dws"
        models.mkdir(parents=True)
        ddir.mkdir(parents=True)
        smesh = models / "out.smesh.1.0"
        smesh.write_text("2 2 1.5 0.34\n0.0 1.0\n0.0 0.0\n0.0 1.0\n4 4.1\n4.2 4.3\n")
        coords = ((0, 0), (1, 0), (0, 1), (1, 1))
        (ddir / "dws.dat").write_text(
            "".join(f"{x} {z} {v}\n" for (x, z), v in zip(coords, vals)),
            encoding="utf-8",
        )
        return smesh

    a = _smesh_and_dws(tmp_path / "runs" / "r1", [8.0, 0.0, 8.0, 0.0])
    b = _smesh_and_dws(tmp_path / "runs" / "r2", [0.0, 6.0, 0.0, 6.0])
    st = FormState({"work_dir": str(tmp_path)})
    xyz, n = union_dws_xyz_for_smeshes(st, tmp_path, [a, b], enabled=True)
    assert n == 2 and xyz is not None
    np.testing.assert_allclose(xyz[:, 2], [8.0, 6.0, 8.0, 6.0])
    xyz_off, n_off = union_dws_xyz_for_smeshes(st, tmp_path, [a, b], enabled=False)
    assert xyz_off is None and n_off == 0


def test_intersect_dws_xyz_for_smeshes(tmp_path: Path) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.dws_plot import intersect_dws_xyz_for_smeshes
    from pyAOBS.modeling.tomo2d.gui.state.form_state import FormState

    def _smesh_and_dws(run: Path, vals: list[float]) -> Path:
        models = run / "outputs" / "models"
        ddir = run / "outputs" / "dws"
        models.mkdir(parents=True)
        ddir.mkdir(parents=True)
        smesh = models / "out.smesh.1.0"
        smesh.write_text("2 2 1.5 0.34\n0.0 1.0\n0.0 0.0\n0.0 1.0\n4 4.1\n4.2 4.3\n")
        coords = ((0, 0), (1, 0), (0, 1), (1, 1))
        (ddir / "dws.dat").write_text(
            "".join(f"{x} {z} {v}\n" for (x, z), v in zip(coords, vals)),
            encoding="utf-8",
        )
        return smesh

    a = _smesh_and_dws(tmp_path / "runs" / "r1", [8.0, 8.0, 8.0, 8.0])
    b = _smesh_and_dws(tmp_path / "runs" / "r2", [0.0, 0.0, 2.0, 2.0])
    st = FormState({"work_dir": str(tmp_path)})
    xyz, n = intersect_dws_xyz_for_smeshes(st, tmp_path, [a, b], enabled=True)
    assert n == 2 and xyz is not None
    np.testing.assert_allclose(xyz[:, 2], [0.0, 0.0, 2.0, 2.0])
    disjoint_a = _smesh_and_dws(tmp_path / "runs" / "r3", [8.0, 0.0, 8.0, 0.0])
    disjoint_b = _smesh_and_dws(tmp_path / "runs" / "r4", [0.0, 6.0, 0.0, 6.0])
    xyz2, n2 = intersect_dws_xyz_for_smeshes(
        st, tmp_path, [disjoint_a, disjoint_b], enabled=True
    )
    assert n2 == 2 and xyz2 is not None
    np.testing.assert_allclose(xyz2[:, 2], 0.0)


def test_mean_dws_xyz_matches_covered_members(tmp_path: Path) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.dws_plot import mean_dws_xyz_for_smeshes
    from pyAOBS.modeling.tomo2d.gui.state.form_state import FormState

    def _smesh_and_dws(run: Path, vals: list[float]) -> Path:
        models = run / "outputs" / "models"
        ddir = run / "outputs" / "dws"
        models.mkdir(parents=True)
        ddir.mkdir(parents=True)
        smesh = models / "out.smesh.1.0"
        smesh.write_text("2 2 1.5 0.34\n0.0 1.0\n0.0 0.0\n0.0 1.0\n4 4.1\n4.2 4.3\n")
        coords = ((0, 0), (1, 0), (0, 1), (1, 1))
        (ddir / "dws.dat").write_text(
            "".join(f"{x} {z} {v}\n" for (x, z), v in zip(coords, vals)),
            encoding="utf-8",
        )
        return smesh

    a = _smesh_and_dws(tmp_path / "runs" / "r1", [8.0, 8.0, 8.0, 8.0])
    b = _smesh_and_dws(tmp_path / "runs" / "r2", [0.0, 0.0, 2.0, 2.0])
    st = FormState({"work_dir": str(tmp_path)})
    xyz, n = mean_dws_xyz_for_smeshes(st, tmp_path, [a, b], enabled=True)
    assert n == 2 and xyz is not None
    np.testing.assert_allclose(xyz[:, 2], [8.0, 8.0, 5.0, 5.0])
    from pyAOBS.modeling.tomo2d.gui.services.dws_plot import (
        dws_xyz_for_plot,
        mean_dws_xyz_from_arrays,
    )

    loaded = [
        dws_xyz_for_plot(st, tmp_path, p, enabled=True) for p in (a, b)
    ]
    xyz2, n2 = mean_dws_xyz_from_arrays(loaded)
    assert n2 == 2 and xyz2 is not None
    np.testing.assert_allclose(xyz2[:, 2], xyz[:, 2])


def test_save_image_filter_covers_jpg_pdf_ps() -> None:
    from pyAOBS.modeling.tomo2d.gui.plots.export_figure import SAVE_IMAGE_FILTER
    from pyAOBS.utils.qt_file_dialog import ensure_save_suffix

    text = SAVE_IMAGE_FILTER.lower()
    for ext in (".png", ".jpg", ".pdf", ".ps", ".eps", ".svg", ".tif"):
        assert ext in text
    assert ensure_save_suffix("out", "JPEG (*.jpg *.jpeg)", default_suffix=".png").endswith(".jpg")
    assert ensure_save_suffix("out", "PDF (*.pdf)", default_suffix=".png").endswith(".pdf")
    assert ensure_save_suffix("out", "PostScript (*.ps)", default_suffix=".png").endswith(".ps")


def test_pdf_from_qimage_is_valid_pdf(tmp_path: Path) -> None:
    import os

    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QColor, QImage, QPainter
    from PySide6.QtWidgets import QApplication

    from pyAOBS.modeling.tomo2d.gui.plots.export_figure import _save_pdf_from_qimage

    _app = QApplication.instance() or QApplication([])
    img = QImage(48, 32, QImage.Format.Format_RGB32)
    img.fill(QColor(255, 255, 255))
    p = QPainter(img)
    p.setPen(QColor(0, 0, 0))
    p.drawText(4, 20, "12.3")
    p.end()
    out = tmp_path / "ticks.pdf"
    _save_pdf_from_qimage(img, str(out), dpi=96.0)
    data = out.read_bytes()
    assert data.startswith(b"%PDF")
    assert len(data) > 200


def _ps_dsc_bbox(path: Path) -> tuple[float, float, float, float]:
    for line in path.read_text(encoding="latin-1").splitlines():
        if line.startswith("%%BoundingBox:") and "(atend)" not in line:
            x0, y0, x1, y1 = (float(p) for p in line.split()[1:5])
            return x0, y0, x1, y1
    raise AssertionError(f"no BoundingBox in {path}")


def test_fit_figsize_to_a4_wide_uses_landscape() -> None:
    from pyAOBS.modeling.tomo2d.gui.plots.export_figure import _fit_figsize_to_a4

    w, h, orient = _fit_figsize_to_a4(9.6, 5.2)
    assert orient == "landscape"
    assert w <= 11.69
    assert h <= 8.27
    w2, h2, orient2 = _fit_figsize_to_a4(20.0, 4.0)
    assert orient2 == "landscape"
    assert w2 < 20.0
    assert w2 <= 11.69 - 0.8
    w3, h3, orient3 = _fit_figsize_to_a4(5.0, 10.0)
    assert orient3 == "portrait"
    assert abs(w3 - 5.0) < 1e-6
    assert abs(h3 - 10.0) < 1e-6


def test_savefig_ps_wide_stays_on_a4(tmp_path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from pyAOBS.modeling.tomo2d.gui.plots.export_figure import _savefig_ps

    fig = Figure(figsize=(16.0, 4.0))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1])
    out = tmp_path / "wide.ps"
    _savefig_ps(fig, str(out), "ps", tight=False)
    text = out.read_text(encoding="latin-1")
    assert text.startswith("%!PS")
    assert "%%DocumentPaperSizes: a4" in text[:800]
    assert "%%Orientation: landscape" in text[:800]
    x0, y0, x1, y1 = _ps_dsc_bbox(out)
    assert min(x0, y0) >= -20
    assert max(x1, y1) <= 860
    # 保存后不改现场图尺寸
    ow, oh = fig.get_size_inches()
    assert ow == pytest.approx(16.0)
    assert oh == pytest.approx(4.0)


def test_savefig_ps_grid_alpha_no_transparency_warning(tmp_path: Path) -> None:
    import warnings

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from pyAOBS.modeling.tomo2d.gui.plots.export_figure import _savefig_ps

    fig = Figure(figsize=(6.0, 4.0))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1])
    ax.grid(True, alpha=0.3)
    out = tmp_path / "grid.ps"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _savefig_ps(fig, str(out), "ps", tight=False)
    assert out.is_file()
    assert not any("transparency" in str(w.message).lower() for w in caught)
    gl = ax.xaxis.get_gridlines()
    if gl:
        assert gl[0].get_alpha() == pytest.approx(0.3)


def test_save_page_from_qimage_wide_ps_fits_a4(tmp_path: Path) -> None:
    import os

    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QColor, QImage
    from PySide6.QtWidgets import QApplication

    from pyAOBS.modeling.tomo2d.gui.plots.export_figure import _save_page_from_qimage

    _app = QApplication.instance() or QApplication([])
    img = QImage(2000, 400, QImage.Format.Format_RGB32)
    img.fill(QColor(255, 255, 255))
    out = tmp_path / "grab.ps"
    _save_page_from_qimage(img, str(out), "ps")
    text = out.read_text(encoding="latin-1")
    assert text.startswith("%!PS")
    assert "%%DocumentPaperSizes: a4" in text[:800]
    x0, y0, x1, y1 = _ps_dsc_bbox(out)
    assert min(x0, y0) >= -20
    assert max(x1, y1) <= 860


def test_model_picker_refresh_replots_when_last_row_index_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """换运行包后末行下标常相同，selectRow 不再发信号，必须强制重绘。"""
    import os

    pytest.importorskip("PySide6")
    pytest.importorskip("matplotlib")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from pyAOBS.modeling.tomo2d.gui.dialogs.model_picker_dialog import ModelPickerDialog
    from pyAOBS.modeling.tomo2d.gui.services.inv_monitor import (
        InvMonitorSpec,
        ModelCandidate,
    )
    from pyAOBS.modeling.tomo2d.gui.state.form_state import FormState

    _app = QApplication.instance() or QApplication([])
    packs: list[list[ModelCandidate]] = []
    for name in ("run_a", "run_b"):
        d = tmp_path / name
        d.mkdir()
        rows = []
        for it in (1, 2):
            p = d / f"smesh.{it}.0"
            p.write_text("x", encoding="utf-8")
            rows.append(ModelCandidate(path=p, iter=it, iset=0))
        packs.append(rows)
    idx = {"n": 0}

    monkeypatch.setattr(
        "pyAOBS.modeling.tomo2d.gui.dialogs.model_picker_dialog.build_model_catalog",
        lambda _spec: packs[idx["n"]],
    )
    monkeypatch.setattr(
        "pyAOBS.modeling.tomo2d.gui.dialogs.model_picker_dialog.collect_run_inversion_params",
        lambda _spec: "",
    )

    dlg = ModelPickerDialog(FormState({"work_dir": str(tmp_path)}))
    previews: list[Path] = []

    def fake_preview(path: Path) -> None:
        previews.append(Path(path))
        dlg._path = Path(path)

    dlg._preview = fake_preview  # type: ignore[method-assign]
    dlg._spec = InvMonitorSpec(out_root=tmp_path / "run_a", run_dir=tmp_path / "run_a")
    dlg.refresh()
    assert previews[-1].parent.name == "run_a"
    assert dlg.table.currentRow() == 1

    idx["n"] = 1
    dlg._spec = InvMonitorSpec(out_root=tmp_path / "run_b", run_dir=tmp_path / "run_b")
    dlg.refresh()
    assert previews[-1].parent.name == "run_b"
    assert dlg._path is not None and dlg._path.parent.name == "run_b"

    packs.append([])
    idx["n"] = 2
    dlg.refresh()
    assert dlg._path is None
    dlg.close()


def test_model_picker_select_user_run_outside_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """浏览自选：工区 runs/ 为空时仍能绑定外部运行包，刷新不丢。"""
    import os

    pytest.importorskip("PySide6")
    pytest.importorskip("matplotlib")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from pyAOBS.modeling.tomo2d.gui.dialogs.model_picker_dialog import ModelPickerDialog
    from pyAOBS.modeling.tomo2d.gui.services.inv_monitor import ModelCandidate
    from pyAOBS.modeling.tomo2d.gui.state.form_state import FormState

    _app = QApplication.instance() or QApplication([])
    work = tmp_path / "work"
    (work / "runs").mkdir(parents=True)
    other = tmp_path / "archive" / "ttinv_old"
    models = other / "outputs" / "models"
    models.mkdir(parents=True)
    (other / "manifest.json").write_text("{}", encoding="utf-8")
    smesh = models / "out.smesh.1.0"
    smesh.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "pyAOBS.modeling.tomo2d.gui.dialogs.model_picker_dialog.build_model_catalog",
        lambda spec: [
            ModelCandidate(path=smesh, iter=1, iset=0, run_name=spec.run_dir.name)
        ]
        if spec.run_dir is not None
        else [],
    )
    monkeypatch.setattr(
        "pyAOBS.modeling.tomo2d.gui.dialogs.model_picker_dialog.collect_run_inversion_params",
        lambda _spec: "",
    )

    dlg = ModelPickerDialog(FormState({"work_dir": str(work)}))
    dlg._preview = lambda path: None  # type: ignore[method-assign]
    dlg.reload_runs()
    assert dlg.combo_run.count() == 0
    assert dlg.select_user_run_dir(models) is True
    assert dlg.combo_run.count() == 1
    assert Path(str(dlg.combo_run.currentData())).resolve() == other.resolve()
    assert "自选" in dlg.combo_run.currentText()
    dlg.reload_runs()
    assert dlg.combo_run.count() == 1
    assert Path(str(dlg.combo_run.currentData())).resolve() == other.resolve()
    dlg.close()

