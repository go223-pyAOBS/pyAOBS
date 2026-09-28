"""vedit 工区工程单测（无 GUI）。"""

from __future__ import annotations

import json
from pathlib import Path


def test_vedit_project_create_save_load(tmp_path: Path) -> None:
    from pyAOBS.modeling.vedit.gui.project import VeditProject
    from pyAOBS.modeling.vedit.gui.services.workdir_layout import (
        INPUTS_DIR,
        OUTPUTS_DIR,
        project_json_path,
    )

    work = tmp_path / "demo_proj"
    proj = VeditProject.create_new(str(work), name="demo")
    assert (work / INPUTS_DIR).is_dir()
    assert (work / OUTPUTS_DIR).is_dir()
    proj.workflow.vin_path = "inputs/v.in"
    proj.editor.dx_sm = 0.02
    path = proj.save()
    assert Path(path) == Path(project_json_path(str(work)))
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    assert raw["name"] == "demo"
    assert raw["workflow"]["vin_path"] == "inputs/v.in"
    assert "dirty" not in raw

    loaded = VeditProject.load(path)
    assert loaded.name == "demo"
    assert loaded.workflow.vin_path == "inputs/v.in"
    assert loaded.editor.dx_sm == 0.02
    assert loaded.abs_or_join("inputs/v.in").endswith(str(Path("inputs") / "v.in"))
