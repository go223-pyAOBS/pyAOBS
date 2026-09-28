"""走时残差 .tres 解析（监视拟合图）。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pyAOBS.modeling.tomo2d.gui.services.tres_sample import (
    has_inverse_tres_files,
    list_inverse_outlier_files,
    list_inverse_tres_files,
    load_outliers_for_monitor,
    load_tres_for_monitor,
    normalize_tres_raytypes,
    parse_outlier_file,
    parse_tomo2d_tres_file,
)

pytestmark = pytest.mark.unit


def test_parse_tomo2d_tres_file(tmp_path: Path) -> None:
    p = tmp_path / "out.tres.1.3"
    p.write_text(
        "10.0 -0.12\n"
        "20.5 0.04\n"
        "bad\n"
        "30 0.01 extra\n",
        encoding="utf-8",
    )
    xs, rs, codes = parse_tomo2d_tres_file(p)
    assert xs == pytest.approx([10.0, 20.5, 30.0])
    assert rs == pytest.approx([-0.12, 0.04, 0.01])
    assert codes == [-1, -1, -1]


def test_parse_tomo2d_tres_file_with_raytype(tmp_path: Path) -> None:
    p = tmp_path / "out.tres.1.1"
    p.write_text(
        "# rcv_x residual raytype\n"
        "10.0 -0.12 0\n"
        "20.5 0.04 1\n"
        "30 0.01 0\n",
        encoding="utf-8",
    )
    xs, rs, codes = parse_tomo2d_tres_file(p)
    assert xs == pytest.approx([10.0, 20.5, 30.0])
    assert rs == pytest.approx([-0.12, 0.04, 0.01])
    assert codes == [0, 1, 0]


def test_normalize_tres_raytypes_tomo2d_and_one_based() -> None:
    assert normalize_tres_raytypes([0, 1, 0]) == [0, 1, 0]
    assert normalize_tres_raytypes([1, 2, 1]) == [0, 1, 0]
    assert normalize_tres_raytypes([-1, -1]) == [-1, -1]


def test_list_and_load_tres_prefer_iter(tmp_path: Path) -> None:
    od = tmp_path / "outputs"
    res = od / "residuals"
    res.mkdir(parents=True)
    (res / "out.tres.1.1").write_text("1 0.2\n2 -0.1\n", encoding="utf-8")
    (res / "out.tres.2.1").write_text("1 0.05\n", encoding="utf-8")
    (res / "out.tres.2.2").write_text("3 -0.05\n", encoding="utf-8")
    (od / "out.tres.2.9").write_text("8 0.01\n", encoding="utf-8")

    entries = list_inverse_tres_files(od / "out")
    assert {(it, isrc) for _, it, isrc in entries} == {(1, 1), (2, 1), (2, 2), (2, 9)}
    assert has_inverse_tres_files(od / "out") is True
    assert has_inverse_tres_files(od / "missing") is False

    groups, note = load_tres_for_monitor(od / "out", iter_prefer=2)
    assert "iter=2" in note and "3 炮" in note
    assert "按 OBS 着色" in note
    assert {isrc for isrc, _xs, _rs, _c in groups} == {1, 2, 9}

    groups1, note1 = load_tres_for_monitor(od / "out", iter_prefer=1)
    assert "iter=1" in note1
    assert len(groups1) == 1

    _, note_fb = load_tres_for_monitor(od / "out", iter_prefer=99)
    assert "iter=2" in note_fb
    assert "模型 iter=99" in note_fb


def test_load_tres_attaches_raytype_from_data(tmp_path: Path) -> None:
    run = tmp_path / "run"
    inp = run / "inputs"
    out = run / "outputs"
    inp.mkdir(parents=True)
    out.mkdir()
    (inp / "data.dat").write_text(
        "1\n"
        "s 10.0 0.0 3\n"
        "r 1.0 0.0 0 1.0 0.05\n"
        "r 2.0 0.0 1 2.0 0.05\n"
        "r 3.0 0.0 0 3.0 0.05\n",
        encoding="utf-8",
    )
    (out / "out.tres.1.1").write_text("1.0 -0.10\n2.0 0.20\n3.0 0.00\n", encoding="utf-8")
    groups, note = load_tres_for_monitor(out / "out", iter_prefer=1, run_dir=run)
    assert len(groups) == 1
    _isrc, xs, rs, codes = groups[0]
    assert xs == pytest.approx([1.0, 2.0, 3.0])
    assert rs == pytest.approx([-0.10, 0.20, 0.00])
    assert codes == [0, 1, 0]
    assert "上=折射" in note and "补列" in note


def test_load_tres_prefers_file_raytype_over_data(tmp_path: Path) -> None:
    run = tmp_path / "run"
    inp = run / "inputs"
    out = run / "outputs"
    inp.mkdir(parents=True)
    out.mkdir()
    (inp / "data.dat").write_text(
        "1\n"
        "s 10.0 0.0 3\n"
        "r 1.0 0.0 1 1.0 0.05\n"
        "r 2.0 0.0 1 2.0 0.05\n"
        "r 3.0 0.0 1 3.0 0.05\n",
        encoding="utf-8",
    )
    (out / "out.tres.1.1").write_text(
        "# rcv_x residual raytype\n"
        "1.0 -0.10 0\n"
        "2.0 0.20 1\n"
        "3.0 0.00 0\n",
        encoding="utf-8",
    )
    groups, note = load_tres_for_monitor(out / "out", iter_prefer=1, run_dir=run)
    assert groups[0][3] == [0, 1, 0]
    assert "第三列" in note and "补列" not in note


def test_parse_and_load_outlier_files(tmp_path: Path) -> None:
    od = tmp_path / "outputs"
    od.mkdir()
    body = (
        "# isrc ircv src_x rcv_x raytype tres_s lin_res\n"
        "1 3 10.0 1.5 0 -0.12 4.2\n"
        "2 1 20.0 2.5 1 0.08 -5.1\n"
    )
    (od / "out.outliers.2.1").write_text(body, encoding="utf-8")
    (od / "out.outliers.3.1").write_text(
        "3 2 30.0 3.5 0 0.01 6.0\n", encoding="utf-8"
    )
    (od / "out.outliers.final").write_text(
        "9 9 99.0 9.9 0 0.0 9.0\n", encoding="utf-8"
    )
    rows = parse_outlier_file(od / "out.outliers.2.1")
    assert len(rows) == 2
    assert rows[0][:5] == (1, 3, 10.0, 1.5, 0)
    listed = list_inverse_outlier_files(od / "out")
    assert [(it, iset) for _, it, iset in listed] == [(2, 1), (3, 1)]
    got, note = load_outliers_for_monitor(od / "out", iter_prefer=2, iset_prefer=1)
    assert len(got) == 2
    assert "iter=2" in note and "iset=1" in note
    latest, n2 = load_outliers_for_monitor(od / "out")
    assert len(latest) == 1 and latest[0][0] == 3
    assert "iter=3" in n2

    only = tmp_path / "only_final"
    only.mkdir()
    (only / "out.outliers.final").write_text(
        "1 1 0.0 1.0 0 0.2 4.5\n", encoding="utf-8"
    )
    fin, n3 = load_outliers_for_monitor(only / "out")
    assert len(fin) == 1
    assert ".outliers.final" in n3


def test_list_tres_inside_legacy_named_dir(tmp_path: Path) -> None:
    run = tmp_path / "fmodel" / "out.vpfd41.SV80"
    run.mkdir(parents=True)
    (run / "mesh.tres.1.1").write_text("1.0 0.10\n", encoding="utf-8")
    (run / "mesh.tres.2.1").write_text("1.0 0.02\n", encoding="utf-8")
    (run / "mesh.tres.2.3").write_text("2.0 -0.01\n", encoding="utf-8")
    entries = list_inverse_tres_files(run)
    assert {(it, isrc) for _, it, isrc in entries} == {(1, 1), (2, 1), (2, 3)}
    groups, note = load_tres_for_monitor(run, iter_prefer=2)
    assert "iter=2" in note and "2 炮" in note
    assert {isrc for isrc, *_ in groups} == {1, 3}

    from pyAOBS.modeling.tomo2d.gui.services.tres_sample import (
        out_root_from_inverse_smesh,
        tres_out_root_candidates,
    )

    smesh = run / "mesh.smesh.2.0"
    smesh.write_text("x", encoding="utf-8")
    assert out_root_from_inverse_smesh(smesh) == run / "mesh"
    cands = tres_out_root_candidates(smesh=smesh, log=run / "log.all")
    assert run in cands
    groups2, _ = load_tres_for_monitor(run / "mesh", iter_prefer=2)
    assert {isrc for isrc, *_ in groups2} == {1, 3}
