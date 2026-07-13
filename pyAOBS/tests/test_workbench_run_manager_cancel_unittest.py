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
from workbench.shell.logic.helpers import parse_batch_recent_values


class RunManagerCancelTest(unittest.TestCase):
    def test_cancel_event_marks_run_cancelled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "wb_project"
            pm = ProjectManager()
            project = pm.create_project(root, name="cancel-test")
            rm = RunManager()

            import threading

            cancel_event = threading.Event()
            cancel_event.set()
            run = rm.run_command(
                project=project,
                node_id="cancel_case",
                command=[sys.executable, "-c", "import time; time.sleep(3)"],
                cancel_event=cancel_event,
            )
            manifest = json.loads(run.manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(manifest.get("status"), "cancelled")
            self.assertEqual(manifest.get("error"), "cancelled by user")
            self.assertIn("pid", manifest)


class BatchQueueRecentParseTest(unittest.TestCase):
    def test_parse_batch_recent_values_valid(self) -> None:
        rec = parse_batch_recent_values(
            ("src_001", "success", "run_123")
        )
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertEqual(rec["source_run_id"], "src_001")
        self.assertEqual(rec["status"], "success")
        self.assertEqual(rec["new_run_id"], "run_123")

    def test_parse_batch_recent_values_invalid(self) -> None:
        self.assertIsNone(parse_batch_recent_values(None))
        self.assertIsNone(parse_batch_recent_values(("only_src",)))


if __name__ == "__main__":
    unittest.main()
