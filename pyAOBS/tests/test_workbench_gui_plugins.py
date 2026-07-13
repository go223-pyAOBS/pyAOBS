from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workbench.plugins.gui_plugins import (
    DataGuiPlugin,
    IphaseGuiPlugin,
    ImodelGuiPlugin,
    ObemTsmToSacPlugin,
    Raw2SacPlugin,
    Sac2yPlugin,
    Tomo2dGuiPlugin,
    ZplotpyGuiPlugin,
)


def test_zplotpy_plugin_build_spec(tmp_path: Path) -> None:
    plugin = ZplotpyGuiPlugin()
    payload = {
        "node_id": "zplot_run",
        "executable": "python",
        "args": "--itype 0",
        "cwd": "",
        "env_text": "",
        "inputs_text": "",
    }
    spec = plugin.build_spec(tmp_path, payload)
    assert spec.node_id == "zplot_run"
    assert spec.command[:3] == ["python", "-m", "pyAOBS.workbench.gui_audit_launchers.zplotpy_gui"]
    assert "--itype" in spec.command


def test_imodel_plugin_default_node(tmp_path: Path) -> None:
    plugin = ImodelGuiPlugin()
    payload = {
        "node_id": "",
        "executable": "python",
        "args": "",
        "cwd": "",
        "env_text": "",
        "inputs_text": "",
    }
    spec = plugin.build_spec(tmp_path, payload)
    assert spec.node_id == "imodel_gui"
    assert spec.command[:3] == ["python", "-m", "pyAOBS.workbench.gui_audit_launchers.imodel_gui"]


def test_iphase_plugin_with_relative_cwd(tmp_path: Path) -> None:
    plugin = IphaseGuiPlugin()
    (tmp_path / "work").mkdir()
    payload = {
        "node_id": "",
        "executable": "python",
        "args": "",
        "cwd": "work",
        "env_text": "",
        "inputs_text": "",
    }
    spec = plugin.build_spec(tmp_path, payload)
    assert spec.node_id == "iphase_gui"
    assert spec.cwd == (tmp_path / "work").resolve()
    assert spec.command[:3] == ["python", "-m", "pyAOBS.workbench.gui_audit_launchers.iphase_gui"]


def test_tomo2d_gui_plugin_default_node(tmp_path: Path) -> None:
    plugin = Tomo2dGuiPlugin()
    payload = {
        "node_id": "",
        "executable": "python",
        "args": "",
        "cwd": "",
        "env_text": "",
        "inputs_text": "",
    }
    spec = plugin.build_spec(tmp_path, payload)
    assert spec.node_id == "tomo2d_gui"
    assert spec.command[:3] == ["python", "-m", "pyAOBS.modeling.tomo2d.gui"]


def test_obem_tsm_plugin_default_node(tmp_path: Path) -> None:
    plugin = ObemTsmToSacPlugin()
    payload = {
        "node_id": "",
        "executable": "python",
        "args": "cfg.ini",
        "cwd": "",
        "env_text": "",
        "inputs_text": "",
    }
    spec = plugin.build_spec(tmp_path, payload)
    assert spec.node_id == "obem_tsm_to_sac"
    assert spec.command[:3] == [
        "python",
        "-m",
        "pyAOBS.processors.raw2sac.obem_tsm_to_sac_obspy",
    ]


def test_sac2y_plugin_default_node(tmp_path: Path) -> None:
    plugin = Sac2yPlugin()
    payload = {
        "node_id": "",
        "executable": "python",
        "args": "",
        "cwd": "",
        "env_text": "",
        "inputs_text": "",
    }
    spec = plugin.build_spec(tmp_path, payload)
    assert spec.node_id == "sac2y_v2_1"
    assert spec.command[:3] == [
        "python",
        "-m",
        "pyAOBS.processors.raw2sac.sac2y_v2_1_obspy",
    ]


def test_raw2sac_plugin_default_node(tmp_path: Path) -> None:
    plugin = Raw2SacPlugin()
    payload = {
        "node_id": "",
        "executable": "python",
        "args": "",
        "cwd": "",
        "env_text": "",
        "inputs_text": "",
    }
    spec = plugin.build_spec(tmp_path, payload)
    assert spec.node_id == "raw2sac_v1_1"
    assert spec.command[:3] == [
        "python",
        "-m",
        "pyAOBS.processors.raw2sac.raw2sac_v1_1_obspy",
    ]


def test_data_gui_plugin_default_node(tmp_path: Path) -> None:
    plugin = DataGuiPlugin()
    payload = {
        "node_id": "",
        "executable": "python",
        "args": "",
        "cwd": "",
        "env_text": "",
        "inputs_text": "",
    }
    spec = plugin.build_spec(tmp_path, payload)
    assert spec.node_id == "data_gui"
    assert spec.command[0] == "python"
    assert spec.command[1].endswith("processors\\raw2sac\\idata.py") or spec.command[1].endswith(
        "processors/raw2sac/idata.py"
    )
    assert spec.command[2:] == []

