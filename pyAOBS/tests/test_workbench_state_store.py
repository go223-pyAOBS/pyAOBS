from pathlib import Path

from pyAOBS.workbench.core.project_manager import ProjectManager
from pyAOBS.workbench.core.state_store import StateStore


def test_save_and_load_ui_state(tmp_path: Path) -> None:
    pm = ProjectManager()
    project = pm.create_project(tmp_path / "state_project", name="StateProject")
    store = StateStore()

    state = {
        "active_tab": "tomo2d",
        "recent_files": ["datasets/raw/a.sgy", "picks/line01.txin"],
        "window": {"width": 1280, "height": 800},
    }
    ref = store.save_ui_state(project, state)
    assert ref.path.exists()

    restored = store.load_ui_state(project)
    assert restored == state

