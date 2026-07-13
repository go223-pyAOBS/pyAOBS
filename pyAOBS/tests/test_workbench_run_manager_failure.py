from pathlib import Path
import json
import sys

from pyAOBS.workbench.core.project_manager import ProjectManager
from pyAOBS.workbench.core.run_manager import RUN_MANIFEST_FILE, RunManager


def test_run_command_failure_persists_manifest(tmp_path: Path) -> None:
    pm = ProjectManager()
    project = pm.create_project(tmp_path / "run_project_fail", name="RunFail")
    rm = RunManager()

    run = rm.run_command(
        project=project,
        node_id="fail_node",
        command=[sys.executable, "-c", "import sys; sys.exit(3)"],
    )

    manifest_path = run.run_dir / RUN_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["return_code"] == 3

