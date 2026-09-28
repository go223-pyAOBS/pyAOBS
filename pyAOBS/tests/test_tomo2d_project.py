"""tomo2d 工区工程单测（无 UI）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyAOBS.modeling.tomo2d.gui.project import Tomo2dProject
from pyAOBS.modeling.tomo2d.gui.services.workdir_layout import (
    LAYOUT_DIRS,
    PROJECT_JSON,
    infer_workdir_from_json,
    project_json_path,
    resolve_project_json,
)
from pyAOBS.modeling.tomo2d.gui.state.form_state import FormState


pytestmark = pytest.mark.unit


def test_create_new_layout(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    proj = Tomo2dProject.create_new(str(root), name="demo")
    assert proj.name == "demo"
    assert proj.is_open
    for rel in LAYOUT_DIRS:
        assert (root / rel).is_dir()
    out = proj.save()
    assert Path(out) == Path(project_json_path(str(root)))
    assert Path(out).is_file()


def test_load_infers_workdir(tmp_path: Path) -> None:
    root = tmp_path / "w"
    root.mkdir()
    proj = Tomo2dProject.create_new(str(root), name="w")
    proj.profile = {"gen.v0": "2.0"}
    proj.bin_path = "/opt/tomo2d"
    json_path = proj.save()

    loaded = Tomo2dProject.load(json_path)
    assert Path(loaded.workdir).resolve() == root.resolve()
    assert loaded.profile["gen.v0"] == "2.0"
    assert loaded.bin_path == "/opt/tomo2d"

    loaded2 = Tomo2dProject.load(str(root))
    assert Path(loaded2.workdir).resolve() == root.resolve()


def test_resolve_and_infer_helpers(tmp_path: Path) -> None:
    root = tmp_path / "a"
    meta = root / "meta"
    meta.mkdir(parents=True)
    jp = meta / "tomo2d_project.json"
    jp.write_text("{}", encoding="utf-8")
    assert resolve_project_json(str(root)).endswith(PROJECT_JSON.replace("\\", "/")) or resolve_project_json(
        str(root)
    ).endswith("tomo2d_project.json")
    assert Path(infer_workdir_from_json(str(jp))).resolve() == root.resolve()


def test_form_state_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    proj = Tomo2dProject.create_new(str(root))
    st = FormState(
        {
            "work_dir": str(tmp_path),
            "bin_path": "bin",
            "gen.v0": "1.5",
            "gen.smesh_out": "outputs/m.smesh",
        }
    )
    proj.capture_from_form_state(st)
    assert Path(proj.workdir).resolve() == tmp_path.resolve()
    assert proj.profile["gen.v0"] == "1.5"
    st2 = FormState()
    proj.workdir = str(root.resolve())
    proj.apply_to_form_state(st2)
    assert st2.get_str("work_dir") == str(root.resolve())
    assert st2.get_str("gen.v0") == "1.5"
