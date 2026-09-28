"""imodel 工区工程单测（无 GUI；避开 imodel 包对 pygmt 的硬依赖）。"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path


GUI_DIR = Path(__file__).resolve().parents[1] / "visualization" / "imodel" / "gui"


def _ensure_pkg(name: str, path: Path | None = None) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        if path is not None:
            mod.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = mod
    return mod


def _load_imodel_project_modules():
    """按文件加载 project / workdir_layout，不执行 imodel/__init__.py。"""
    imodel_root = GUI_DIR.parent
    _ensure_pkg("pyAOBS")
    _ensure_pkg("pyAOBS.visualization")
    _ensure_pkg("pyAOBS.visualization.imodel", imodel_root)
    gui_pkg = _ensure_pkg("pyAOBS.visualization.imodel.gui", GUI_DIR)
    svc_pkg = _ensure_pkg(
        "pyAOBS.visualization.imodel.gui.services", GUI_DIR / "services"
    )

    layout_path = GUI_DIR / "services" / "workdir_layout.py"
    spec_l = importlib.util.spec_from_file_location(
        "pyAOBS.visualization.imodel.gui.services.workdir_layout",
        layout_path,
        submodule_search_locations=[str(GUI_DIR / "services")],
    )
    assert spec_l and spec_l.loader
    layout = importlib.util.module_from_spec(spec_l)
    sys.modules[spec_l.name] = layout
    spec_l.loader.exec_module(layout)
    setattr(svc_pkg, "workdir_layout", layout)

    proj_path = GUI_DIR / "project.py"
    spec_p = importlib.util.spec_from_file_location(
        "pyAOBS.visualization.imodel.gui.project",
        proj_path,
        submodule_search_locations=[str(GUI_DIR)],
    )
    assert spec_p and spec_p.loader
    project = importlib.util.module_from_spec(spec_p)
    sys.modules[spec_p.name] = project
    spec_p.loader.exec_module(project)
    setattr(gui_pkg, "project", project)
    return project, layout


def test_imodel_project_create_save_load(tmp_path: Path) -> None:
    project_mod, layout = _load_imodel_project_modules()
    ImodelProject = project_mod.ImodelProject

    work = tmp_path / "demo_imodel"
    proj = ImodelProject.create_new(str(work), name="demo")
    assert (work / layout.INPUTS_DIR).is_dir()
    assert (work / layout.OUTPUTS_DIR).is_dir()
    proj.workflow.vp_model = "inputs/v.in"
    proj.workflow.vs_model = "inputs/vs.nc"
    proj.analysis.show_interfaces = False
    path = proj.save()
    assert Path(path) == Path(layout.project_json_path(str(work)))
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    assert raw["name"] == "demo"
    assert raw["workflow"]["vp_model"] == "inputs/v.in"
    assert "dirty" not in raw

    loaded = ImodelProject.load(path)
    assert loaded.name == "demo"
    assert loaded.workflow.vp_model == "inputs/v.in"
    assert loaded.workflow.vs_model == "inputs/vs.nc"
    assert loaded.analysis.show_interfaces is False
    assert loaded.abs_or_join("inputs/v.in").endswith(str(Path("inputs") / "v.in"))


def test_imodel_gui_section_roundtrip(tmp_path: Path) -> None:
    project_mod, _layout = _load_imodel_project_modules()
    ImodelProject = project_mod.ImodelProject

    work = tmp_path / "sec"
    proj = ImodelProject.create_new(str(work), name="sec")
    proj.workflow.vp_model = "inputs/model.vin"
    proj.analysis.basement_selection = "Interface 2"
    sec = proj.to_imodel_gui_section()
    assert sec["model_file"].endswith("model.vin")
    assert "imodel_project" in sec
    again = ImodelProject.from_imodel_gui_section(sec, workdir=str(work), name="sec2")
    assert again.workflow.vp_model.endswith("model.vin")
    assert again.analysis.basement_selection == "Interface 2"


def test_workbench_imodel_form_passes_project(tmp_path: Path) -> None:
    from pyAOBS.workbench.shell.logic.gui_form_commands import apply_gui_quick_form_to_command

    project_mod, _layout = _load_imodel_project_modules()
    ImodelProject = project_mod.ImodelProject

    work = tmp_path / "wb_imodel"
    proj = ImodelProject.create_new(str(work), name="wb")
    proj.workflow.vp_model = "inputs/v.in"
    jp = proj.save()

    nid, args, msg, env_extra, inputs = apply_gui_quick_form_to_command(
        "imodel.gui",
        node_id="",
        gui_form={
            "imodel_node": "OBS_A",
            "imodel_workdir": str(work),
            "imodel_model": "inputs/v.in",
            "imodel_aux": "",
        },
        project_root=tmp_path,
    )
    assert nid == "OBS_A"
    assert str(Path(jp)) in args.replace('"', "")
    assert "PYAOBS_IMODEL_PROJECT=" in env_extra
    assert "inputs/v.in" in inputs
    assert "工区" in msg
