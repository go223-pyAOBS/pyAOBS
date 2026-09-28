"""vedit ProfileEditor：选中、微调、插删、V-Plot、撤销。"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "modeling" / "vedit" / "examples"


@pytest.fixture
def editor():
    from pyAOBS.modeling.vedit.core import Model
    from pyAOBS.modeling.vedit.gui.editor_controller import ProfileEditor

    model = Model.load(str(EXAMPLES / "v1.in"))
    ed = ProfileEditor()
    ed.set_model(model)
    return ed


def _node(model, ilayer: int, ipart: int, inode: int):
    from model import NodeIndex

    return model.get_node(NodeIndex(ilayer, ipart, inode))


def test_pick_and_nudge(editor) -> None:
    editor.pick_node(0, 0, 1)
    assert editor.selected == {(0, 0): [1]}
    y0 = _node(editor.model, 0, 0, 1)[1]
    assert editor.nudge("down", large=True)
    y1 = _node(editor.model, 0, 0, 1)[1]
    assert y1 > y0
    assert editor.dirty
    assert editor.undo()
    y2 = _node(editor.model, 0, 0, 1)[1]
    assert abs(y2 - y0) < 1e-9


def test_insert_depth_node(editor) -> None:
    editor.pick_node(0, 0, 1)
    tpl = editor.model.get_tpl((0, 0))
    n0 = len(tpl)
    x = (tpl.x[1] + tpl.x[2]) / 2
    y = (tpl.y[1] + tpl.y[2]) / 2
    assert editor.insert_node_at(x, y)
    assert len(editor.model.get_tpl((0, 0))) == n0 + 1
    assert (0, 0) in editor.selected


def test_drag_checkpoint_discard_when_no_move(editor) -> None:
    editor.pick_node(0, 0, 1)
    editor.begin_drag(10.0, 1.0)
    assert editor.history.can_undo()
    editor.end_drag()  # 未移动 → 丢弃检查点
    assert not editor.history.can_undo()


def test_vplot_toggle_and_velocity_edit(editor) -> None:
    editor.pick_node(0, 0, 1)
    assert editor.toggle_velocity_mode() == "enter"
    assert editor.velocity_edit_mode
    assert len(editor.model[0].v_top.x) >= 3
    editor.pick_velocity_node(0, True, 1)
    x0 = editor.model[0].v_top.x[1]
    editor.begin_v_drag(x0)
    assert editor.drag_v_to(x0 + 0.15)
    editor.end_v_drag()
    assert abs(editor.model[0].v_top.x[1] - (x0 + 0.15)) < 1e-6
    assert editor.set_velocity_value(5.55)
    assert abs(editor.model[0].v_top.y[1] - 5.55) < 1e-9
    assert editor.toggle_velocity_mode() == "exit"
    assert not editor.velocity_edit_mode


def test_vplot_without_selection_opens_velocity_hint(editor) -> None:
    editor.clear_selection()
    assert editor.toggle_velocity_mode() == "need_velocity_window"


def test_ray_job_missing_rin(tmp_path: Path) -> None:
    from pyAOBS.modeling.vedit.gui.workers.ray_worker import run_rayinvr_job

    (tmp_path / "v.in").write_text("x\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="r.in"):
        run_rayinvr_job(str(tmp_path))
