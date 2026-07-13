"""tt_inverse 可复现运行包（inputs/outputs/manifest.json）。"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from pyAOBS.modeling.tomo2d.tt_inverse_bundle import (
    build_tt_inverse_bundle,
    finalize_tt_inverse_manifest,
    make_tt_inverse_run_dir_name,
    write_tt_inverse_manifest,
)

pytestmark = pytest.mark.unit


def test_run_dir_name_preserves_data_stem_suffix_like_ph8_13():
    """超长 data 主干被截断时应保留末尾，避免 ph8_13 → ph8_1。"""
    n = make_tt_inverse_run_dir_name(
        "mesh.smesh",
        "ttimes_st36_38_ph8_13.dat",
        run_id="abc123",
        stamp="20260409T170729",
    )
    assert "ph8_13" in n


def test_bundle_copies_inputs_and_rewrites_output_paths(tmp_path):
    wd = tmp_path / "work"
    wd.mkdir()
    (wd / "m.smesh").write_text("mesh", encoding="utf-8")
    (wd / "d.dat").write_text("data", encoding="utf-8")
    kwargs = {"log_file": "iter.log", "out_root": "myout.root"}
    br = build_tt_inverse_bundle(wd, "m.smesh", "d.dat", kwargs)

    assert br.run_dir.is_dir()
    assert (br.run_dir / "inputs" / "mesh.smesh").read_text(encoding="utf-8") == "mesh"
    assert (br.run_dir / "inputs" / "data.dat").read_text(encoding="utf-8") == "data"
    assert br.kwargs["log_file"] == "outputs/iter.log"
    assert br.kwargs["out_root"] == "outputs/myout.root"
    assert br.kwargs["dws_file"] == "outputs/dws.dat"
    assert br.mesh.startswith("inputs/")
    assert br.data.startswith("inputs/")
    assert len(br.inputs_manifest) == 2
    assert all("sha256" in row for row in br.inputs_manifest)


def test_manifest_roundtrip_and_finalize(tmp_path):
    wd = tmp_path / "w"
    wd.mkdir()
    (wd / "a.smesh").write_text("x", encoding="utf-8")
    (wd / "b.dat").write_text("y", encoding="utf-8")
    br = build_tt_inverse_bundle(wd, "a.smesh", "b.dat", {})
    argv = [
        "/fake/tt_inverse",
        f"-M{br.mesh}",
        f"-G{br.data}",
    ]
    write_tt_inverse_manifest(
        manifest_path=br.manifest_path,
        work_dir=wd,
        run_dir=br.run_dir,
        executable_resolved=argv[0],
        argv=argv,
        mesh=br.mesh,
        data=br.data,
        kwargs=br.kwargs,
        inputs_rows=br.inputs_manifest,
        status="running",
    )
    data = json.loads(br.manifest_path.read_text(encoding="utf-8"))
    assert data["kind"] == "pyaobs.tomo2d.tt_inverse"
    assert data["inputs"][0]["role"] == "mesh"

    (br.run_dir / "outputs").mkdir(exist_ok=True)
    (br.run_dir / "outputs" / "x.txt").write_text("out", encoding="utf-8")
    finalize_tt_inverse_manifest(
        br.manifest_path, result=SimpleNamespace(returncode=0), err=None
    )
    data2 = json.loads(br.manifest_path.read_text(encoding="utf-8"))
    assert data2["post_run"]["status"] == "finished"
    assert data2["post_run"]["exit_code"] == 0
    assert any(o["path"].startswith("outputs/") for o in data2["post_run"]["output_files"])
    assert br.kwargs["dws_file"] == "outputs/dws.dat"


def test_bundle_gravity_default_grav_dws(tmp_path):
    wd = tmp_path / "work"
    wd.mkdir()
    (wd / "m.smesh").write_text("m", encoding="utf-8")
    (wd / "d.dat").write_text("d", encoding="utf-8")
    (wd / "g.dat").write_text("g", encoding="utf-8")
    kwargs = {
        "gravity_opts": {
            "grav_file": "g.dat",
            "grid_spec": "0/1/0/1/0.1/0.1",
            "refrange": "0/1",
            "continent": ("c.dat", 1),
        }
    }
    (wd / "c.dat").write_text("c", encoding="utf-8")
    br = build_tt_inverse_bundle(wd, "m.smesh", "d.dat", kwargs)
    assert br.kwargs["gravity_opts"]["grav_dws"] == "outputs/grav_dws.dat"
