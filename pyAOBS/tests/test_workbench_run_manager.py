from pathlib import Path
import sys
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workbench.core.project_manager import ProjectManager
from workbench.core.run_manager import RUN_MANIFEST_FILE, RunManager


def test_run_command_writes_manifest_and_logs(tmp_path: Path) -> None:
    pm = ProjectManager()
    project = pm.create_project(tmp_path / "run_project", name="RunProject")
    rm = RunManager()

    run = rm.run_command(
        project=project,
        node_id="demo_node",
        command=[sys.executable, "-c", "print('hello workbench')"],
        params={"demo": True},
    )

    manifest_path = run.run_dir / RUN_MANIFEST_FILE
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["return_code"] == 0
    assert manifest["node_id"] == "demo_node"

    stdout_path = run.logs_dir / "stdout.log"
    stderr_path = run.logs_dir / "stderr.log"
    assert stdout_path.exists()
    assert stderr_path.exists()
    assert "hello workbench" in stdout_path.read_text(encoding="utf-8")


def test_run_command_reuses_node_workspace_and_overwrites(tmp_path: Path) -> None:
    pm = ProjectManager()
    project = pm.create_project(tmp_path / "run_project", name="RunProject")
    rm = RunManager()

    run1 = rm.run_command(
        project=project,
        node_id="OBS_node",
        command=[sys.executable, "-c", "print('first run')"],
    )
    assert run1.run_id == "OBS_node"
    assert run1.run_dir.name == "OBS_node"
    out1 = (run1.logs_dir / "stdout.log").read_text(encoding="utf-8")
    assert "first run" in out1

    run2 = rm.run_command(
        project=project,
        node_id="OBS_node",
        command=[sys.executable, "-c", "print('second run')"],
    )
    assert run2.run_id == "OBS_node"
    assert run2.run_dir == run1.run_dir
    out2 = (run2.logs_dir / "stdout.log").read_text(encoding="utf-8")
    assert "second run" in out2
    assert "first run" not in out2


def test_run_command_gui_rerun_preserves_inputs_dir(tmp_path: Path) -> None:
    pm = ProjectManager()
    project = pm.create_project(tmp_path / "run_project", name="RunProject")
    rm = RunManager()

    run1 = rm.run_command(
        project=project,
        node_id="OBS_node",
        command=[sys.executable, "-c", "print('first gui run')"],
        params={"plugin": "imodel.gui"},
    )
    preserved = run1.inputs_dir / "v.in"
    preserved.write_text("demo", encoding="utf-8")
    assert preserved.exists()

    run2 = rm.run_command(
        project=project,
        node_id="OBS_node",
        command=[sys.executable, "-c", "print('second gui run')"],
        params={"plugin": "imodel.gui"},
    )
    assert run2.run_dir == run1.run_dir
    assert (run2.inputs_dir / "v.in").exists()

