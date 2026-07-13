import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workbench.core.project_manager import ProjectManager
from workbench.core.run_manager import RunManager


class RunManagerAuditEnvTest(unittest.TestCase):
    def test_run_command_injects_audit_env_and_manifest_entry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "wb_project"
            pm = ProjectManager()
            project = pm.create_project(root, name="audit-test")
            rm = RunManager()

            cmd = [
                sys.executable,
                "-c",
                (
                    "import os, json, pathlib; "
                    "p=pathlib.Path(os.environ['PYAOBS_AUDIT_LOG']); "
                    "p.parent.mkdir(parents=True, exist_ok=True); "
                    "p.write_text(json.dumps({'ok':1,'run_id':os.environ.get('PYAOBS_RUN_ID','')})+'\\n', encoding='utf-8'); "
                    "s=pathlib.Path(os.environ['PYAOBS_GUI_STATE_FILE']); "
                    "s.parent.mkdir(parents=True, exist_ok=True); "
                    "s.write_text(json.dumps({'ok':1,'run_id':os.environ.get('PYAOBS_RUN_ID','')})+'\\n', encoding='utf-8'); "
                    "print(os.environ.get('PYAOBS_RUN_ID',''))"
                ),
            ]
            run = rm.run_command(project=project, node_id="audit_case", command=cmd)
            manifest = json.loads(run.manifest_file.read_text(encoding="utf-8"))

            self.assertIn("audit", manifest)
            audit_rel = str(manifest["audit"].get("session_log", ""))
            self.assertTrue(audit_rel.endswith(f"{run.run_id}.jsonl"))
            self.assertIn("gui_state", manifest)
            gui_state_rel = str(manifest["gui_state"].get("state_file", ""))
            self.assertTrue(gui_state_rel.endswith(f"{run.run_id}.json"))

            stdout_text = (run.logs_dir / "stdout.log").read_text(encoding="utf-8")
            self.assertIn(run.run_id, stdout_text)

            audit_path = project.root / audit_rel
            self.assertTrue(audit_path.exists())
            audit_text = audit_path.read_text(encoding="utf-8")
            self.assertIn(run.run_id, audit_text)
            gui_state_path = project.root / gui_state_rel
            self.assertTrue(gui_state_path.exists())

    def test_gui_node_uses_run_sandbox_and_env_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "wb_project"
            pm = ProjectManager()
            project = pm.create_project(root, name="gui-sandbox-test")
            rm = RunManager()

            cmd = [
                sys.executable,
                "-c",
                (
                    "import os, pathlib; "
                    "out=pathlib.Path(os.environ['PYAOBS_RUN_OUTPUTS_DIR'])/'marker.txt'; "
                    "out.write_text(os.getcwd(), encoding='utf-8'); "
                    "print(os.environ.get('PYAOBS_RUN_INPUTS_DIR','')); "
                    "print(os.environ.get('PYAOBS_RUN_OUTPUTS_DIR','')); "
                ),
            ]
            run = rm.run_command(
                project=project,
                node_id="gui_case",
                command=cmd,
                params={"plugin": "data.gui"},
                cwd=project.root,
            )
            manifest = json.loads(run.manifest_file.read_text(encoding="utf-8"))

            self.assertEqual(manifest.get("status"), "success")
            self.assertIn("sandbox", manifest)
            self.assertEqual(manifest.get("cwd"), str(run.outputs_dir))

            marker = run.outputs_dir / "marker.txt"
            if not marker.exists():
                marker = run.outputs_dir / "idata" / "marker.txt"
            self.assertTrue(marker.exists())
            expected_outputs_dir = (run.outputs_dir / "idata").resolve()
            self.assertEqual(Path(marker.read_text(encoding="utf-8")).resolve(), expected_outputs_dir)
            expected_inputs_dir = (run.inputs_dir / "idata").resolve()
            self.assertTrue(expected_inputs_dir.exists())
            stdout_lines = [
                ln.strip()
                for ln in (run.logs_dir / "stdout.log").read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
            self.assertTrue(stdout_lines)
            self.assertEqual(Path(stdout_lines[0]).resolve(), expected_inputs_dir)


if __name__ == "__main__":
    unittest.main()
