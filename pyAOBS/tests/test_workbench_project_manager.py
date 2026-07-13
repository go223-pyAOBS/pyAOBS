from pathlib import Path

import pytest

from pyAOBS.workbench.core.project_manager import (
    DEFAULT_LAYOUT_DIRS,
    PROJECT_META_FILE,
    ProjectError,
    ProjectManager,
)


def test_create_and_open_project(tmp_path: Path) -> None:
    pm = ProjectManager()
    root = tmp_path / "demo_project"

    ctx = pm.create_project(root, name="Demo")
    assert ctx.root == root.resolve()
    assert ctx.metadata["name"] == "Demo"
    assert (root / PROJECT_META_FILE).exists()

    for rel in DEFAULT_LAYOUT_DIRS:
        assert (root / rel).exists(), rel

    reopened = pm.open_project(root)
    assert reopened.metadata["name"] == "Demo"


def test_validate_missing_layout_fails(tmp_path: Path) -> None:
    pm = ProjectManager()
    root = tmp_path / "broken_project"
    pm.create_project(root, name="Broken")

    # break layout
    (root / "runs").rmdir()

    with pytest.raises(ProjectError):
        pm.validate_project(root)

