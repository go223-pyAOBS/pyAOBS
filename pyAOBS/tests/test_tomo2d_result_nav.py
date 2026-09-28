"""运行包服务：runs 列表 / manifest 摘要 / 速度差。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pyAOBS.modeling.tomo2d.gui.services.result_nav import (
    compute_smesh_velocity_diff,
    default_adjacent_pair,
    diff_vlim_half_range,
    diff_vlim_is_auto,
    format_diff_vlim_caption,
    format_run_summary_text,
    infer_run_dir_from_smesh,
    find_smesh_for_inverse_log,
    list_tt_inverse_run_dirs,
    load_run_summary,
    order_smesh_pair,
    resample_vgrid_field_to_xarray,
    resolve_diff_colorbar_limits,
    resolve_user_run_dir,
    set_diff_vlim_auto,
    set_diff_vlim_half_range,
)

pytestmark = pytest.mark.unit


def _write_tiny_smesh(path: Path, v0: float = 4.0) -> None:
    """最小合法 smesh：nx=2 nz=2。"""
    lines = [
        "2 2 1.5 0.34",
        "0.0 1.0",
        "0.0 0.0",
        "0.0 1.0",
        f"{v0} {v0 + 0.1}",
        f"{v0 + 0.2} {v0 + 0.3}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_list_runs_and_summary(tmp_path: Path) -> None:
    work = tmp_path / "work"
    r1 = work / "runs" / "ttinv_old"
    r2 = work / "runs" / "ttinv_new"
    for r in (r1, r2):
        (r / "outputs" / "models").mkdir(parents=True)
        (r / "manifest.json").write_text(
            '{"schema_version":2,"post_run":{"status":"finished","exit_code":0,'
            '"output_files":[{"path":"outputs/models/out.smesh.1.0","kind":"model","bytes":1},'
            '{"path":"outputs/logs/tt_inverse.log","kind":"log","bytes":2}]},'
            '"python_replay":{"kwargs":{"out_root":"outputs/out"}}}',
            encoding="utf-8",
        )
    # 新目录更新 mtime
    (r2 / "outputs" / "touch").write_text("x", encoding="utf-8")

    runs = list_tt_inverse_run_dirs(work)
    assert len(runs) == 2
    assert runs[0].name == "ttinv_new"

    s = load_run_summary(r2)
    assert s.status == "finished"
    assert s.exit_code == 0
    assert s.n_models == 1
    assert s.kind_counts.get("log") == 1
    assert "status=finished" in format_run_summary_text(s)


def test_infer_run_dir_from_smesh(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "pack1"
    smesh = run / "outputs" / "models" / "out.smesh.2.0"
    smesh.parent.mkdir(parents=True)
    smesh.write_text("x", encoding="utf-8")
    (run / "manifest.json").write_text("{}", encoding="utf-8")
    assert infer_run_dir_from_smesh(smesh) == run.resolve()


def test_resolve_user_run_dir_walks_up_and_outside_runs(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "pack1"
    models = run / "outputs" / "models"
    models.mkdir(parents=True)
    smesh = models / "out.smesh.2.0"
    smesh.write_text("x", encoding="utf-8")
    (run / "manifest.json").write_text("{}", encoding="utf-8")
    assert resolve_user_run_dir(run) == run.resolve()
    assert resolve_user_run_dir(models) == run.resolve()
    assert resolve_user_run_dir(smesh) == run.resolve()

    other = tmp_path / "archive" / "ttinv_old"
    omodels = other / "outputs" / "models"
    omodels.mkdir(parents=True)
    (other / "manifest.json").write_text("{}", encoding="utf-8")
    (omodels / "out.smesh.1.0").write_text("x", encoding="utf-8")
    assert resolve_user_run_dir(omodels) == other.resolve()
    assert resolve_user_run_dir(tmp_path / "no_such") is None


def test_find_smesh_for_inverse_log(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "pack1"
    models = run / "outputs" / "models"
    models.mkdir(parents=True)
    (run / "manifest.json").write_text(
        '{"python_replay":{"kwargs":{"out_root":"outputs/out"}}}',
        encoding="utf-8",
    )
    (models / "out.smesh.1.0").write_text("x", encoding="utf-8")
    latest = models / "out.smesh.2.0"
    latest.write_text("y", encoding="utf-8")
    log = run / "outputs" / "tt_inverse.log"
    log.write_text("1 0 0 0.1\n", encoding="utf-8")
    assert find_smesh_for_inverse_log(log) == latest.resolve()


def test_find_smesh_for_loose_inverse_log(tmp_path: Path) -> None:
    log = tmp_path / "tt_inverse.log"
    log.write_text("1 0 0 0.1\n", encoding="utf-8")
    (tmp_path / "out.smesh.1.0").write_text("x", encoding="utf-8")
    latest = tmp_path / "out.smesh.3.1"
    latest.write_text("y", encoding="utf-8")
    assert find_smesh_for_inverse_log(log) == latest.resolve()


def test_find_smesh_for_legacy_named_run_dir(tmp_path: Path) -> None:
    """一次反演一个文件夹：日志旁任意前缀的 ``*.smesh.<iter>.<iset>``。"""
    run = (
        tmp_path
        / "fmodel"
        / "out.vpfd41.st1_65.w0.1.SV80.SD10.DV20.DD20.DQ-1"
    )
    run.mkdir(parents=True)
    log = run / "log.all.vpfd41.w0.1.Lh1.0-4.0"
    log.write_text("1 0 0 0.1\n", encoding="utf-8")
    (run / "mesh.smesh.1.0").write_text("x", encoding="utf-8")
    latest = run / "mesh.smesh.4.0"
    latest.write_text("y", encoding="utf-8")
    (run / "mesh.refl.4.0").write_text("0 0\n", encoding="utf-8")
    assert find_smesh_for_inverse_log(log) == latest.resolve()


def test_find_smesh_when_out_root_is_the_run_directory(tmp_path: Path) -> None:
    """``-O`` 等于目录路径时，smesh 写在父目录、文件名前缀为目录名。"""
    fmodel = tmp_path / "fmodel"
    run = fmodel / "out.vpfd41.st1_65.SV80"
    run.mkdir(parents=True)
    log = run / "log.all.vpfd41"
    log.write_text("1 0 0 0.1\n", encoding="utf-8")
    latest = fmodel / f"{run.name}.smesh.3.1"
    latest.write_text("y", encoding="utf-8")
    (fmodel / f"{run.name}.smesh.1.0").write_text("x", encoding="utf-8")
    assert find_smesh_for_inverse_log(log) == latest.resolve()


def test_monitor_spec_for_run_points_at_run_outputs(tmp_path: Path) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.result_nav import monitor_spec_for_run
    from pyAOBS.modeling.tomo2d.gui.services.inv_monitor import build_model_catalog

    work = tmp_path / "work"
    rd = work / "runs" / "ttinv_20260101_120000"
    models = rd / "outputs" / "models"
    models.mkdir(parents=True)
    (rd / "manifest.json").write_text(
        '{"python_replay":{"kwargs":{"out_root":"outputs/out"}}}',
        encoding="utf-8",
    )
    (models / "out.smesh.1.0").write_text("2 2 1.5 0.34\n", encoding="utf-8")
    (models / "out.smesh.2.0").write_text("2 2 1.5 0.34\n", encoding="utf-8")
    spec = monitor_spec_for_run(rd)
    assert spec.run_dir == rd.resolve()
    assert spec.out_root == (rd / "outputs" / "out").resolve()
    cat = build_model_catalog(spec)
    assert [c.iter for c in cat] == [1, 2]


def test_adjacent_pair_and_velocity_diff(tmp_path: Path) -> None:
    od = tmp_path / "outputs"
    models = od / "models"
    models.mkdir(parents=True)
    a = models / "out.smesh.1.0"
    b = models / "out.smesh.2.0"
    _write_tiny_smesh(a, 4.0)
    _write_tiny_smesh(b, 4.5)
    pair = default_adjacent_pair(od / "out")
    assert pair is not None
    assert pair[0].name == "out.smesh.1.0"
    assert pair[1].name == "out.smesh.2.0"

    diff = compute_smesh_velocity_diff(a, b, mode="abs")
    assert diff.dv.shape == (2, 2)
    assert np.allclose(diff.dv, 0.5)
    assert diff.mean == pytest.approx(0.5)

    pct = compute_smesh_velocity_diff(a, b, mode="percent")
    assert pct.dv[0, 0] == pytest.approx(100.0 * 0.5 / 4.0)

    early = tmp_path / "out.smesh.1.1"
    late = tmp_path / "out.smesh.4.2"
    assert order_smesh_pair(late, early) == (early, late)
    assert order_smesh_pair(early, late) == (early, late)


def test_diff_colorbar_limits_fixed_vs_auto() -> None:
    from pyAOBS.modeling.tomo2d.gui.state.form_state import FormState

    lo, hi = resolve_diff_colorbar_limits(0.02, auto=True, half_range=0.5)
    assert lo == pytest.approx(-0.02)
    assert hi == pytest.approx(0.02)
    lo, hi = resolve_diff_colorbar_limits(0.02, auto=False, half_range=0.5)
    assert lo == pytest.approx(-0.5)
    assert hi == pytest.approx(0.5)
    note = format_diff_vlim_caption(1.2, -0.5, 0.5, auto=False)
    assert "饱和" in note
    st = FormState()
    assert diff_vlim_is_auto(st) is False
    assert diff_vlim_half_range(st, "abs") == pytest.approx(0.5)
    assert diff_vlim_half_range(st, "percent") == pytest.approx(5.0)
    set_diff_vlim_auto(st, True)
    set_diff_vlim_half_range(st, "abs", 0.25)
    assert diff_vlim_is_auto(st) is True
    assert diff_vlim_half_range(st, "abs") == pytest.approx(0.25)


def test_sigma_colorbar_limits_auto_vs_fixed() -> None:
    from pyAOBS.modeling.tomo2d.gui.services.result_nav import (
        format_sigma_vlim_caption,
        resolve_sigma_colorbar_limits,
        sigma_data_max,
        sigma_robust_hi,
    )

    std_v = np.array([[0.0, 0.4], [0.1, 0.2]], dtype=float)
    assert sigma_data_max(std_v) == pytest.approx(0.4)
    a_lo, a_hi = resolve_sigma_colorbar_limits(0.4, auto=True, half_range=0.5)
    assert a_lo == pytest.approx(0.0)
    assert a_hi == pytest.approx(0.4)
    r_lo, r_hi = resolve_sigma_colorbar_limits(
        0.4, auto=True, half_range=0.5, robust_hi=0.15
    )
    assert r_lo == pytest.approx(0.0)
    assert r_hi == pytest.approx(0.15)
    f_lo, f_hi = resolve_sigma_colorbar_limits(0.4, auto=False, half_range=0.5)
    assert f_lo == pytest.approx(0.0)
    assert f_hi == pytest.approx(0.5)
    note = format_sigma_vlim_caption(0.5, 0.8, auto=False)
    assert "饱和" in note
    auto_note = format_sigma_vlim_caption(0.15, 0.4, auto=True)
    assert "P96" in auto_note
    assert "尾部饱和" in auto_note

    blob = np.full((30, 30), 0.08)
    blob[0, 0] = 3.0
    hi = sigma_robust_hi(blob)
    assert 0.08 <= hi < 1.0


def test_sigma_cmap_uses_develf_cpt() -> None:
    from pyAOBS.modeling.tomo2d.gui.services.result_nav import (
        resolve_sigma_cmap_and_limits,
    )
    from pyAOBS.modeling.tomo2d.gui.services.smesh_plot_core import builtin_sigma_cpt_path
    from pyAOBS.visualization.gmt_cpt import parse_gmt_cpt_for_matplotlib

    cpt = builtin_sigma_cpt_path()
    assert cpt.is_file()
    _, z0, z1 = parse_gmt_cpt_for_matplotlib(str(cpt))
    assert z0 == pytest.approx(0.0)
    assert z1 == pytest.approx(0.40)
    spec, lo, hi, gamma = resolve_sigma_cmap_and_limits(
        np.array([[0.0, 0.2], [0.08, 1.0]]), auto=True, half_range=0.5
    )
    assert Path(spec) == cpt
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.40)
    assert gamma is None
    _spec2, _lo2, hi2, g2 = resolve_sigma_cmap_and_limits(
        np.array([[0.01]]), auto=False, half_range=0.05
    )
    assert hi2 == pytest.approx(0.40)
    assert g2 is None


def test_resample_diff_onto_plot_grid_not_native_vgrid(tmp_path: Path) -> None:
    """绘图网格含空气层，形状 ≠ 节点 vgrid；差值须重采样而不能直接对齐。"""
    from pyAOBS.model_building.tomoform import SlownessMesh2D

    a = tmp_path / "out.smesh.1.0"
    b = tmp_path / "out.smesh.2.0"
    _write_tiny_smesh(a, 4.0)
    _write_tiny_smesh(b, 4.5)
    diff = compute_smesh_velocity_diff(a, b, mode="abs")
    mesh = SlownessMesh2D.from_file(str(b))
    vel_ds = mesh.to_xarray()
    assert tuple(vel_ds["velocity"].shape) != tuple(diff.dv.shape)
    plot_ds = resample_vgrid_field_to_xarray(mesh, diff.dv)
    assert tuple(plot_ds["velocity"].shape) == tuple(vel_ds["velocity"].shape)
    np.testing.assert_allclose(mesh.vgrid[0, 0], 4.5)
    z = np.asarray(plot_ds["z"].values)
    air = plot_ds["velocity"].values[z < 0]
    assert air.size == 0 or np.allclose(air, 0.0)


def test_compare_tray_two_then_rotate(tmp_path: Path) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.model_compare import CompareTray

    a = tmp_path / "a.smesh"
    b = tmp_path / "b.smesh"
    c = tmp_path / "c.smesh"
    a.write_text("a", encoding="utf-8")
    b.write_text("b", encoding="utf-8")
    c.write_text("c", encoding="utf-8")
    t = CompareTray()
    msg, pa, pb = t.add(a)
    assert pb is None and t.path_a == a.resolve()
    assert "A" in msg
    msg, pa, pb = t.add(a)
    assert pb is None and "不同" in msg
    msg, pa, pb = t.add(b)
    assert pa == a.resolve() and pb == b.resolve()
    msg, pa, pb = t.add(b)
    assert pa is None and pb is None
    msg, pa, pb = t.add(c)
    assert pa == b.resolve() and pb == c.resolve()
    assert [p.name for p in t.ensemble] == ["a.smesh", "b.smesh", "c.smesh"]
    t.clear()
    assert t.path_a is None and t.path_b is None
    assert t.ensemble == []


def test_multipath_wraps_like_analysis_log_list() -> None:
    import os

    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QPlainTextEdit

    from pyAOBS.modeling.tomo2d.gui.widgets.form_rows import MultiPathRow

    _app = QApplication.instance() or QApplication([])
    row = MultiPathRow("集合:")
    assert row.edit.lineWrapMode() == QPlainTextEdit.LineWrapMode.WidgetWidth
    row.edit.setPlainText("a.smesh\nb.smesh\nc.smesh")
    assert row.paths() == ["a.smesh", "b.smesh", "c.smesh"]
    row.close()


def test_compare_tray_extend_ensemble_does_not_rotate(tmp_path: Path) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.model_compare import CompareTray

    a = tmp_path / "a.smesh"
    b = tmp_path / "b.smesh"
    c = tmp_path / "c.smesh"
    a.write_text("a", encoding="utf-8")
    b.write_text("b", encoding="utf-8")
    c.write_text("c", encoding="utf-8")
    t = CompareTray()
    assert t.extend_ensemble([a, b, a]) == 2
    assert t.path_a == a.resolve() and t.path_b == b.resolve()
    assert t.extend_ensemble([b]) == 0
    assert t.extend_ensemble([c]) == 1
    assert t.path_a == a.resolve() and t.path_b == b.resolve()
    assert [p.name for p in t.ensemble] == ["a.smesh", "b.smesh", "c.smesh"]


def test_apply_smesh_diff_velocity_panels_honor_contours(tmp_path: Path) -> None:
    """勾选等值线时 A/B 必须 draw_contours=True；None 规格表示用公用表，不能当假。"""
    pytest.importorskip("xarray")
    from pyAOBS.modeling.tomo2d.gui.plots.velocity_contours import set_contours_enabled
    from pyAOBS.modeling.tomo2d.gui.services.model_compare import apply_smesh_diff
    from pyAOBS.modeling.tomo2d.gui.state.form_state import FormState

    a = tmp_path / "a.smesh"
    b = tmp_path / "b.smesh"
    _write_tiny_smesh(a, 4.0)
    _write_tiny_smesh(b, 5.0)
    st = FormState({"work_dir": str(tmp_path)})

    class _Stub:
        def set_save_dir(self, *_a, **_k) -> None:
            pass

        def setWindowTitle(self, *_a, **_k) -> None:
            pass

        def set_velocity_stack(self, panels, **_k) -> None:
            self.panels = panels

    w = _Stub()
    set_contours_enabled(st, True)
    apply_smesh_diff(w, st, a, b)
    assert w.panels[0]["draw_contours"] is False
    assert w.panels[1]["draw_contours"] is True
    assert w.panels[2]["draw_contours"] is True

    set_contours_enabled(st, False)
    apply_smesh_diff(w, st, a, b)
    assert w.panels[1]["draw_contours"] is False
    assert w.panels[2]["draw_contours"] is False


def test_compare_tray_set_paths(tmp_path: Path) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.model_compare import CompareTray

    a = tmp_path / "a.smesh"
    b = tmp_path / "b.smesh"
    a.write_text("a", encoding="utf-8")
    b.write_text("b", encoding="utf-8")
    t = CompareTray()
    t.set_paths(a, b)
    assert t.path_a == a.resolve() and t.path_b == b.resolve()
    t.set_paths(a, tmp_path / "missing.smesh")
    assert t.path_a == a.resolve() and t.path_b is None
    t.set_paths(None, None)
    assert t.path_a is None and t.path_b is None


def test_stack_reflector_mean_std_and_write(tmp_path: Path) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.smesh_ops import (
        stack_reflector_mean_std,
        write_interface_xz,
    )

    a = tmp_path / "a.refl"
    b = tmp_path / "b.refl"
    a.write_text("0 8\n10 9\n", encoding="utf-8")
    b.write_text("0 10\n10 11\n", encoding="utf-8")
    x, mz, sz = stack_reflector_mean_std([a, b])
    assert x[0] == 0 and x[-1] == 10
    np.testing.assert_allclose(mz[0], 9.0)
    np.testing.assert_allclose(sz[0], 1.0)
    out = write_interface_xz(x, mz, tmp_path / "mean.refl", header="must not appear")
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "9.000000" in text
    assert "#" not in text
    dup = write_interface_xz(
        [0.0, 0.0, 10.0], [8.0, 9.0, 10.0], tmp_path / "dup.refl"
    )
    dxy = np.loadtxt(dup)
    assert dxy.shape[0] == 2
    np.testing.assert_allclose(dxy[0, 0], 0.0)
    np.testing.assert_allclose(dxy[0, 1], 8.5)


def test_stack_mean_std_skips_uncovered_nodes(tmp_path: Path) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.smesh_ops import stack_mean_std

    a = tmp_path / "a.smesh"
    b = tmp_path / "b.smesh"
    _write_tiny_smesh(a, 4.0)
    _write_tiny_smesh(b, 6.0)
    xyz_a = np.array(
        [[0.0, 0.0, 8.0], [1.0, 0.0, 8.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    xyz_b = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 6.0], [1.0, 1.0, 6.0]]
    )
    _, mean_v, std_v = stack_mean_std([a, b], dws_xyz_list=[xyz_a, xyz_b])
    np.testing.assert_allclose(mean_v[0, 0], 4.0)
    np.testing.assert_allclose(mean_v[0, 1], 4.1)
    np.testing.assert_allclose(mean_v[1, 0], 6.2)
    np.testing.assert_allclose(mean_v[1, 1], 6.3)
    np.testing.assert_allclose(std_v, 0.0)
    _, mean_all, _ = stack_mean_std([a, b])
    np.testing.assert_allclose(mean_all[0, 0], 5.0)


def test_apply_smesh_ensemble_stats_two_models(tmp_path: Path) -> None:
    pytest.importorskip("xarray")
    from pyAOBS.modeling.tomo2d.gui.services.model_compare import (
        apply_smesh_ensemble_stats,
        write_ensemble_stat_files,
    )
    from pyAOBS.modeling.tomo2d.gui.state.form_state import FormState

    a = tmp_path / "out.smesh.1.0"
    b = tmp_path / "out.smesh.2.0"
    _write_tiny_smesh(a, 4.0)
    _write_tiny_smesh(b, 6.0)
    (tmp_path / "out.refl.1.0").write_text("0.0 8.0\n1.0 8.2\n", encoding="utf-8")
    (tmp_path / "out.refl.2.0").write_text("0.0 9.0\n1.0 9.2\n", encoding="utf-8")
    st = FormState({"work_dir": str(tmp_path)})

    class _Stub:
        def set_save_dir(self, *_a, **_k) -> None:
            pass

        def setWindowTitle(self, *_a, **_k) -> None:
            pass

        def set_velocity_stack(self, panels, **k) -> None:
            self.panels = panels
            self.dws_xyz = k.get("dws_xyz")

        def set_model_source(self, *_a, **_k) -> None:
            pass

    w = _Stub()
    stat = apply_smesh_ensemble_stats(w, st, [a, b])
    assert stat.n == 2
    assert stat.n_dws == 0
    assert getattr(w, "dws_xyz", None) is None
    assert len(w.panels) == 2
    assert "均值" in w.panels[0]["title"]
    assert w.panels[1].get("vlim") is not None
    assert Path(w.panels[1]["cmap_spec"]).name == "develf.cpt"
    assert w.panels[1].get("norm_gamma") is None
    assert w.panels[1].get("extra")
    assert w.panels[0].get("extra")
    assert w.panels[1]["draw_contours"] is True
    std_specs = w.panels[1]["contour_specs"]
    assert std_specs
    assert all(0.0 < s.value < w.panels[1]["vlim"][1] + 1e-9 for s in std_specs)
    assert w.panels[0].get("vlim") is None
    from pyAOBS.modeling.tomo2d.gui.services.model_compare import paint_smesh_ensemble_stats

    paint_smesh_ensemble_stats(
        w, st, stat, auto_vlim=False, half_range=0.05, reset_home=False
    )
    assert w.panels[0].get("vlim") is None
    lo, hi = w.panels[1]["vlim"]
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.40)
    paint_smesh_ensemble_stats(w, st, stat, auto_vlim=True, half_range=0.05)
    assert w.panels[1]["vlim"][0] == pytest.approx(0.0)
    assert w.panels[1]["vlim"][1] == pytest.approx(0.40)
    written = write_ensemble_stat_files(stat, tmp_path)
    assert written.mean_smesh is not None and written.mean_smesh.is_file()
    assert written.std_smesh is not None and written.std_smesh.is_file()
    assert stat.n_refl == 2
    assert written.mean_refl is not None and written.mean_refl.is_file()

    from pyAOBS.modeling.tomo2d.gui.services.model_compare import (
        save_ensemble_reflector,
        save_ensemble_velocity,
    )

    mean_p = tmp_path / "custom_mean.smesh"
    assert save_ensemble_velocity(stat, mean_p, kind="mean").is_file()
    std_p = tmp_path / "custom_std.smesh"
    assert save_ensemble_velocity(stat, std_p, kind="std").is_file()
    r_mean = save_ensemble_reflector(stat, tmp_path / "r_mean.refl", which="mean")
    r_lo = save_ensemble_reflector(stat, tmp_path / "r_m.refl", which="minus")
    r_hi = save_ensemble_reflector(stat, tmp_path / "r_p.refl", which="plus")
    assert r_mean.is_file() and r_lo.is_file() and r_hi.is_file()
    txt = r_lo.read_text(encoding="utf-8")
    assert "mean−σ" in txt or "mean-" in txt


def test_sigma_cmap_zero_is_white() -> None:
    pytest.importorskip("matplotlib")
    from pyAOBS.modeling.tomo2d.gui.plots.inv_monitor_model import _mpl_cmap

    cmap, _lv = _mpl_cmap("YlGnBu_0white")
    r, g, b, _a = cmap(0.0)
    assert r > 0.99 and g > 0.99 and b > 0.99
    r1, g1, b1, _a1 = cmap(1.0)
    assert r1 + g1 + b1 < 2.5


def _legacy_vgrid_on_regular(mesh, vgrid, x_new, full_zpos, v_air, v_water):
    """对照用：原先 to_xarray 的双重 Python 循环。"""
    nx_new = len(x_new)
    nz_full = len(full_zpos)
    full_vgrid = np.ones((nx_new, nz_full))
    xpos = np.asarray(mesh.xpos, dtype=float)
    zpos = np.asarray(mesh.zpos, dtype=float)
    topo = np.asarray(mesh.topo, dtype=float)
    vg = np.asarray(vgrid, dtype=float)
    for i, x in enumerate(x_new):
        ix = int(np.searchsorted(xpos, x))
        if ix == len(xpos):
            ix = len(xpos) - 1
        elif ix > 0 and (x - xpos[ix - 1]) < (xpos[ix] - x):
            ix = ix - 1
        if ix == len(xpos) - 1:
            topo_val = topo[ix]
        else:
            x1, x2 = xpos[ix], xpos[ix + 1]
            t1, t2 = topo[ix], topo[ix + 1]
            topo_val = t1 + (x - x1) * (t2 - t1) / (x2 - x1)
        for j, z in enumerate(full_zpos):
            if z < -1e-8:
                full_vgrid[i, j] = v_air
            elif topo_val > 1e-8 and -1e-8 <= z <= topo_val + 1e-8:
                full_vgrid[i, j] = v_water
            else:
                rel_z = z - topo_val
                k = int(np.searchsorted(zpos, rel_z))
                if k == 0:
                    full_vgrid[i, j] = vg[ix, 0]
                elif k == len(zpos):
                    full_vgrid[i, j] = vg[ix, -1]
                else:
                    z1, z2 = zpos[k - 1], zpos[k]
                    v1, v2 = vg[ix, k - 1], vg[ix, k]
                    full_vgrid[i, j] = v1 + (v2 - v1) * (rel_z - z1) / (z2 - z1)
    return full_vgrid


def test_to_xarray_vectorized_matches_nested_loop() -> None:
    pytest.importorskip("xarray")
    from pyAOBS.model_building.tomoform import SlownessMesh2D

    mesh = SlownessMesh2D(5, 4, 1.5, 0.33)
    mesh.xpos = np.linspace(0.0, 8.0, 5)
    mesh.zpos = np.linspace(0.0, 6.0, 4)
    mesh.topo = np.array([1.0, 1.2, 0.8, 1.1, 1.0])
    mesh.vgrid = 4.0 + np.arange(5)[:, None] * 0.1 + mesh.zpos[None, :] * 0.25
    mesh.pgrid = 1.0 / mesh.vgrid
    ds = mesh.to_xarray()
    x_new, z_new = mesh._regular_plot_axes()
    slow = _legacy_vgrid_on_regular(
        mesh, mesh.vgrid, x_new, z_new, mesh.v_air, mesh.v_water
    )
    np.testing.assert_allclose(ds["velocity"].values, slow.T, rtol=1e-12, atol=1e-12)
    z = ds.z.values
    assert np.allclose(ds["velocity"].values[z < -1e-8, :], mesh.v_air)


def test_resample_fields_share_plot_axes(tmp_path: Path) -> None:
    pytest.importorskip("xarray")
    from pyAOBS.model_building.tomoform import SlownessMesh2D
    from pyAOBS.modeling.tomo2d.gui.services.result_nav import (
        resample_vgrid_fields_to_xarray,
    )

    a = tmp_path / "out.smesh.1.0"
    _write_tiny_smesh(a, 4.0)
    mesh = SlownessMesh2D.from_file(str(a))
    mean = mesh.vgrid
    std = np.full_like(mean, 0.2)
    ds_m, ds_s = resample_vgrid_fields_to_xarray(mesh, [mean, std])
    np.testing.assert_allclose(ds_m.x.values, ds_s.x.values)
    np.testing.assert_allclose(ds_m.z.values, ds_s.z.values)
    assert float(ds_m.v_air) == 0.0 and float(ds_s.v_water) == 0.0


def test_monte_carlo_result_lists_and_paints(tmp_path: Path) -> None:
    pytest.importorskip("xarray")
    from pyAOBS.modeling.tomo2d.gui.services.qc_workflows import (
        list_monte_carlo_runs,
        load_mc_interface_stats,
        mc_interface_mean_std_overlays,
        paint_monte_carlo_result,
        resolve_monte_carlo_run_dir,
    )
    from pyAOBS.modeling.tomo2d.gui.services.smesh_ops import (
        load_interface_mean_std,
        write_interface_mean_std,
        write_interface_xz,
    )
    from pyAOBS.modeling.tomo2d.gui.state.form_state import FormState

    work = tmp_path / "work"
    run = work / "runs" / "montecarlo_test"
    out = run / "outputs"
    out.mkdir(parents=True)
    _write_tiny_smesh(out / "mean_velocity.smesh", 5.0)
    _write_tiny_smesh(out / "std_velocity.smesh", 0.2)
    write_interface_xz([0.0, 1.0], [8.0, 8.2], out / "mean_moho.refl")
    write_interface_mean_std(
        [0.0, 1.0], [8.0, 8.2], [0.4, 0.5], out / "moho_mean_std.txt"
    )
    assert list_monte_carlo_runs(work) == [run]
    assert resolve_monte_carlo_run_dir(work) == run
    packed = load_interface_mean_std(out / "moho_mean_std.txt")
    assert packed is not None
    xs, zm, zs = packed
    np.testing.assert_allclose(zm[0], 8.0)
    np.testing.assert_allclose(zs[0], 0.4)
    iface = load_mc_interface_stats(run)
    assert iface is not None
    extra = mc_interface_mean_std_overlays(*iface, name="界面")
    assert extra[0].get("z_lo") is not None
    assert extra[0].get("z_hi") is not None

    class _Stub:
        def set_save_dir(self, *_a, **_k) -> None:
            pass

        def set_model_source(self, *_a, **_k) -> None:
            pass

        def set_interaction_hint(self, *_a, **_k) -> None:
            pass

        def set_velocity_stack(self, panels, **k) -> None:
            self.panels = panels
            self.dws_xyz = k.get("dws_xyz")

    w = _Stub()
    st = FormState({"work_dir": str(work)})
    (out / "dws.dat").write_text(
        "0.0 0.0 4\n1.0 0.0 5\n0.0 1.0 0\n1.0 1.0 3\n",
        encoding="utf-8",
    )
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "n_runs": 5,
                "n_kept": 3,
                "chi_max": 1.8,
                "realization_chi": [
                    {"pred_chi": 1.1, "kept": True},
                    {"pred_chi": 1.6, "kept": True},
                    {"pred_chi": 0.9, "kept": True},
                    {"pred_chi": 2.4, "kept": False},
                    {"pred_chi": 3.0, "kept": False},
                ],
            }
        ),
        encoding="utf-8",
    )
    hit = paint_monte_carlo_result(w, st, work, run_dir=run)
    assert hit == run
    assert len(w.panels) == 2
    assert "均值" in w.panels[0]["title"]
    assert "3/5 个模型" in w.panels[0]["title"]
    assert "pred χ² < 1.8" in w.panels[0]["title"]
    assert "均值 1.20" in w.panels[0]["title"]
    assert "最小 0.90" in w.panels[0]["title"]
    assert "最大 1.60" in w.panels[0]["title"]
    assert "误差" in w.panels[1]["title"]
    assert "3 个模型" in w.panels[1]["title"]
    assert Path(w.panels[1]["cmap_spec"]).name == "develf.cpt"
    assert w.panels[1].get("vlim") == pytest.approx((0.0, 0.40))
    assert w.panels[1].get("norm_gamma") is None
    assert w.panels[0].get("extra")
    assert w.panels[0]["extra"][0].get("z_lo") is not None
    assert w.dws_xyz is not None
    assert w.dws_xyz.shape[0] >= 3
    assert getattr(w, "_mc_stat", None) is not None


def _tt_log_line(pred_chi: float) -> str:
    vals = [0.0] * 26
    vals[0] = 2.0
    vals[1] = 1.0
    vals[4] = 2.0
    vals[20] = float(pred_chi)
    return " ".join(f"{v:g}" for v in vals)


def test_stack_filtered_mc_drops_high_chi(tmp_path: Path) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.qc_workflows import (
        stack_filtered_mc_ensemble,
    )
    from pyAOBS.modeling.tomo2d.gui.state.form_state import FormState

    work = tmp_path / "work"
    run = work / "runs" / "montecarlo_chi"
    out = run / "outputs"
    out.mkdir(parents=True)
    _write_tiny_smesh(out / "mean_velocity.smesh", 5.0)
    _write_tiny_smesh(out / "std_velocity.smesh", 0.2)
    for i, (v0, chi) in enumerate(((4.0, 1.2), (6.0, 2.5), (5.0, 1.5))):
        real = run / "reals" / f"{i:03d}"
        (real / "out").mkdir(parents=True)
        (real / "logs").mkdir(parents=True)
        _write_tiny_smesh(real / "out" / "out.smesh.1.1", v0)
        (real / "logs" / "tt_inverse.log").write_text(
            _tt_log_line(chi) + "\n", encoding="utf-8"
        )
    st = FormState({"work_dir": str(work), "mc.chi_max": "1.8"})
    got = stack_filtered_mc_ensemble(st, work, run, {}, chi_max=1.8)
    assert got is not None
    assert len(got["kept"]) == 2
    assert len(got["records"]) == 3
    np.testing.assert_allclose(got["mean_v"][0, 0], 4.5)


def test_format_mc_notes_filters_even_if_kept_true() -> None:
    from pyAOBS.modeling.tomo2d.gui.services.qc_workflows import format_mc_result_notes

    title, _std, _hint = format_mc_result_notes(
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
    assert "pred χ² < 1.8" in title
    assert "最大 1.36" in title
    assert "3.28" not in title


