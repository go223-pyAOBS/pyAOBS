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
    assert "workspaces" in reopened.metadata
    assert reopened.workspaces["zplotpy.gui"] == "tools/zplotpy"
    assert (root / "tools" / "zplotpy").is_dir()
    assert (root / "data" / "raw").is_dir()


def test_open_legacy_layout(tmp_path: Path) -> None:
    from pyAOBS.workbench.core.project_layout import LEGACY_LAYOUT_DIRS

    pm = ProjectManager()
    root = tmp_path / "legacy"
    root.mkdir()
    for rel in LEGACY_LAYOUT_DIRS:
        (root / rel).mkdir(parents=True, exist_ok=True)
    (root / PROJECT_META_FILE).write_text(
        '{"schema_version": 1, "name": "Old", "tool": "pyAOBS-workbench"}\n',
        encoding="utf-8",
    )
    ctx = pm.open_project(root)
    assert ctx.metadata["name"] == "Old"


def test_register_workspace(tmp_path: Path) -> None:
    pm = ProjectManager()
    ctx = pm.create_project(tmp_path / "ws", name="WS")
    ctx = pm.set_workspace(ctx, "zplotpy.gui", ctx.root / "tools" / "zplotpy")
    assert ctx.workspaces["zplotpy.gui"] == "tools/zplotpy"


def test_validate_missing_layout_fails(tmp_path: Path) -> None:
    pm = ProjectManager()
    root = tmp_path / "broken_project"
    pm.create_project(root, name="Broken")

    # break layout
    (root / "_wb" / "runs").rmdir()

    with pytest.raises(ProjectError):
        pm.validate_project(root)


def test_scan_run_history_reads_wb_runs(tmp_path: Path) -> None:
    from pyAOBS.workbench.core.run_manager import RunManager
    from pyAOBS.workbench.shell.logic.run_history import scan_run_history
    import sys

    pm = ProjectManager()
    ctx = pm.create_project(tmp_path / "hist", name="H")
    rm = RunManager()
    run = rm.run_command(
        ctx, "zplotpy", command=[sys.executable, "-c", "print(1)"]
    )
    records, _st, nodes, _tr = scan_run_history(ctx.root)
    assert any(r["run_id"] == "zplotpy" for r in records)
    assert "zplotpy" in nodes
    assert run.run_dir.parent.parent.name == "_wb"

