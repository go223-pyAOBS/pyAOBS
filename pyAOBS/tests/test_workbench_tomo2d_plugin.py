from pathlib import Path

import pytest

from pyAOBS.workbench.plugins.base import PluginValidationError
from pyAOBS.workbench.plugins.tomo2d_plugin import Tomo2DShellPlugin


def test_tomo2d_plugin_build_spec_basic(tmp_path: Path) -> None:
    plugin = Tomo2DShellPlugin()
    payload = {
        "node_id": "inv1",
        "executable": "tt_inverse",
        "args": "-Minputs/mesh.dat -I5 -V-1",
        "cwd": "work",
        "env_text": "TOMO2D_INV_OMP=1\nOMP_NUM_THREADS=4\n",
        "inputs_text": "work/inputs/mesh.dat\n",
    }
    (tmp_path / "work").mkdir()

    spec = plugin.build_spec(tmp_path, payload)
    assert spec.node_id == "inv1"
    assert spec.command[0] == "tt_inverse"
    assert "-V-1" in spec.command
    assert spec.cwd == (tmp_path / "work").resolve()
    assert spec.env["TOMO2D_INV_OMP"] == "1"
    assert len(spec.inputs) == 1


def test_tomo2d_plugin_reject_invalid_env(tmp_path: Path) -> None:
    plugin = Tomo2DShellPlugin()
    payload = {
        "node_id": "inv1",
        "executable": "tt_inverse",
        "args": "",
        "cwd": "",
        "env_text": "BADENV",
        "inputs_text": "",
    }
    with pytest.raises(PluginValidationError):
        plugin.build_spec(tmp_path, payload)

