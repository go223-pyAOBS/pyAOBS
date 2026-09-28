from pathlib import Path

from pyAOBS.workbench.core.project_layout import default_node_id, resolve_node_id
from pyAOBS.workbench.shell.logic.gui_form_commands import apply_gui_quick_form_to_command
from pyAOBS.visualization.zplotpy.project import ZplotProject
from pyAOBS.visualization.iphase.gui.project import IphaseProject


def test_resolve_node_id_uses_user_workspace_name() -> None:
    assert resolve_node_id("南海A", work_dir="tools/zplotpy") == "南海A"
    assert resolve_node_id("", work_dir="tools/LineA") == "LineA"
    assert resolve_node_id("OBS_node", "tomo2d_shell", work_dir="tools/LineA") == "LineA"
    assert resolve_node_id("", work_dir="") == "workspace"
    assert default_node_id("zplotpy.gui") == "workspace"
    assert default_node_id("zplotpy.gui", work_dir="D:/survey/南海A") == "南海A"


def test_zplotpy_form_passes_workdir(tmp_path: Path) -> None:
    work = tmp_path / "zplot_job"
    proj = ZplotProject.create_new(str(work), name="z")
    jp = proj.save()
    nid, args, msg, env_extra, _inputs = apply_gui_quick_form_to_command(
        "zplotpy.gui",
        node_id="",
        gui_form={"zplot_node": "OBS_A", "zplot_workdir": str(work)},
        project_root=tmp_path,
    )
    assert nid == "OBS_A"
    assert str(Path(jp)) in args.replace('"', "")
    assert "PYAOBS_ZPLOTPY_PROJECT=" in env_extra
    assert "工区" in msg


def test_zplotpy_form_empty_workdir() -> None:
    nid, args, msg, env_extra, inputs = apply_gui_quick_form_to_command(
        "zplotpy.gui",
        node_id="n1",
        gui_form={},
        project_root=None,
    )
    assert nid == "n1"
    assert args == ""
    assert env_extra == ""
    assert inputs == []
    assert "新建/打开" in msg


def test_iphase_form_passes_workdir(tmp_path: Path) -> None:
    work = tmp_path / "iphase_job"
    proj = IphaseProject.create_new(str(work), name="p")
    jp = proj.save()
    nid, args, msg, env_extra, _inputs = apply_gui_quick_form_to_command(
        "iphase.gui",
        node_id="",
        gui_form={"iphase_node": "OBS_B", "iphase_workdir": str(work)},
        project_root=tmp_path,
    )
    assert nid == "OBS_B"
    assert str(Path(jp)) in args.replace('"', "")
    assert "PYAOBS_IPHASE_PROJECT=" in env_extra
    assert "工区" in msg


def test_vedit_form_passes_workdir(tmp_path: Path) -> None:
    work = tmp_path / "vedit_job"
    work.mkdir()
    meta = work / "meta"
    meta.mkdir()
    jp = meta / "vedit_project.json"
    jp.write_text("{}", encoding="utf-8")
    nid, args, msg, env_extra, _inputs = apply_gui_quick_form_to_command(
        "vedit.gui",
        node_id="",
        gui_form={"vedit_node": "vedit_gui", "vedit_workdir": str(work)},
        project_root=tmp_path,
    )
    assert nid == "vedit_gui"
    assert str(jp) in args.replace('"', "")
    assert "PYAOBS_VEDIT_PROJECT=" in env_extra
    assert "工区" in msg


def test_vedit_form_passes_vin(tmp_path: Path) -> None:
    vin = tmp_path / "v.in"
    vin.write_text("1\n", encoding="utf-8")
    nid, args, msg, env_extra, inputs = apply_gui_quick_form_to_command(
        "vedit.gui",
        node_id="",
        gui_form={"vedit_model": str(vin)},
        project_root=tmp_path,
    )
    assert nid == "workspace"
    assert str(vin) in args.replace('"', "")
    assert env_extra == ""
    assert str(vin) in inputs
    assert "v.in" in msg


def test_zplot_iphase_resolve_open_path(tmp_path: Path) -> None:
    zwork = tmp_path / "zw"
    zproj = ZplotProject.create_new(str(zwork), name="z")
    zjp = Path(zproj.save())
    assert Path(ZplotProject.resolve_open_path(str(zwork)) or "") == zjp
    assert ZplotProject.resolve_open_path(str(tmp_path / "missing")) is None

    iwork = tmp_path / "iw"
    iproj = IphaseProject.create_new(str(iwork), name="i")
    ijp = Path(iproj.save())
    assert Path(IphaseProject.resolve_open_path(str(iwork)) or "") == ijp


def test_form_fills_workdir_from_workspace_registry(tmp_path: Path) -> None:
    work = tmp_path / "tools" / "zplotpy"
    proj = ZplotProject.create_new(str(work), name="z")
    jp = Path(proj.save())
    nid, args, msg, env_extra, _inputs = apply_gui_quick_form_to_command(
        "zplotpy.gui",
        node_id="",
        gui_form={},
        project_root=tmp_path,
        workspaces={"zplotpy.gui": "tools/zplotpy"},
    )
    assert nid == "zplotpy"
    assert str(jp) in args.replace('"', "")
    assert "PYAOBS_ZPLOTPY_PROJECT=" in env_extra
    assert "工区" in msg


def test_tomo2d_and_idata_form_pass_workdir(tmp_path: Path) -> None:
    tomo = tmp_path / "tools" / "tomo2d"
    tomo.mkdir(parents=True)
    (tomo / "meta").mkdir()
    tjson = tomo / "meta" / "tomo2d_project.json"
    tjson.write_text("{}", encoding="utf-8")
    nid, args, msg, env_extra, _inputs = apply_gui_quick_form_to_command(
        "tomo2d.gui",
        node_id="",
        gui_form={"tomo_workdir": str(tomo)},
        project_root=tmp_path,
    )
    assert nid == "tomo2d"
    assert str(tjson) in args.replace('"', "")
    assert "PYAOBS_TOMO2D_PROJECT=" in env_extra
    assert "工区" in msg

    idata = tmp_path / "tools" / "idata"
    idata.mkdir(parents=True)
    (idata / "meta").mkdir()
    ijson = idata / "meta" / "idata_project.json"
    ijson.write_text("{}", encoding="utf-8")
    nid, args, msg, env_extra, _inputs = apply_gui_quick_form_to_command(
        "data.gui",
        node_id="",
        gui_form={"data_workdir": str(idata)},
        project_root=tmp_path,
    )
    assert nid == "idata"
    assert str(ijson) in args.replace('"', "")
    assert "PYAOBS_IDATA_PROJECT=" in env_extra


def test_form_node_id_uses_user_workdir_name(tmp_path: Path) -> None:
    work = tmp_path / "tools" / "LineA"
    work.mkdir(parents=True)
    (work / "meta").mkdir()
    jp = work / "meta" / "zplotpy_project.json"
    jp.write_text("{}", encoding="utf-8")
    nid, args, _msg, env_extra, _inputs = apply_gui_quick_form_to_command(
        "zplotpy.gui",
        node_id="",
        gui_form={"zplot_workdir": str(work)},
        project_root=tmp_path,
    )
    assert nid == "LineA"
    assert str(jp) in args.replace('"', "")
    assert "PYAOBS_ZPLOTPY_PROJECT=" in env_extra


def test_form_keeps_user_workspace_name(tmp_path: Path) -> None:
    work = tmp_path / "tools" / "zplotpy"
    work.mkdir(parents=True)
    nid, _args, _msg, _env, _inputs = apply_gui_quick_form_to_command(
        "zplotpy.gui",
        node_id="zplotpy",
        gui_form={"zplot_node": "南海A", "zplot_workdir": str(work)},
        project_root=tmp_path,
    )
    assert nid == "南海A"
