"""vedit core：v.in 往返、节点约束、.edit 路径、历史栈。"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "modeling" / "vedit" / "examples"


@pytest.fixture
def v1_path() -> Path:
    p = EXAMPLES / "v1.in"
    assert p.exists(), f"missing example: {p}"
    return p


def test_model_load_dump_roundtrip(v1_path: Path, tmp_path: Path) -> None:
    from pyAOBS.modeling.vedit.core import Model

    model = Model.load(str(v1_path))
    assert model.nlayer >= 2
    out = tmp_path / "roundtrip.in"
    model.dump(str(out))
    model2 = Model.load(str(out))
    assert model.dumps() == model2.dumps()


def test_model_wide_format_roundtrip(v1_path: Path, tmp_path: Path) -> None:
    from pyAOBS.modeling.vedit.core import Model

    model = Model.load(str(v1_path))
    out = tmp_path / "wide.in"
    model.dump(str(out), format_wide=True)
    text = out.read_text(encoding="utf-8")
    assert text.strip()
    model2 = Model.load(str(out))
    # 宽格式写入后再读，层数应一致
    assert model2.nlayer == model.nlayer


def test_move_leading_node_forbidden(v1_path: Path) -> None:
    from pyAOBS.modeling.vedit.core import Model, NodeIndex

    model = Model.load(str(v1_path))
    idx = NodeIndex(0, 0, 0)
    with pytest.raises(ValueError, match="LEADING"):
        model.move_node(idx, 0.1, 0.0)


def test_delete_leading_node_forbidden(v1_path: Path) -> None:
    from pyAOBS.modeling.vedit.core import Model, NodeIndex

    model = Model.load(str(v1_path))
    with pytest.raises(ValueError, match="LEADING"):
        model.delete_node(NodeIndex(0, 0, 0))


def test_insert_node_between(v1_path: Path) -> None:
    from pyAOBS.modeling.vedit.core import Model, NodeIndex

    model = Model.load(str(v1_path))
    tpl = model.get_tpl((0, 0))
    assert tpl is not None and len(tpl) >= 3
    n_before = len(tpl)
    mid_x = (tpl.x[1] + tpl.x[2]) / 2
    mid_y = (tpl.y[1] + tpl.y[2]) / 2
    model.insert_node(NodeIndex(0, 0, 1), (mid_x, mid_y, 0))
    assert len(model.get_tpl((0, 0))) == n_before + 1


def test_edit_path_helpers(tmp_path: Path) -> None:
    from pyAOBS.modeling.vedit.core import edit_path_for, ensure_edit_copy

    vin = tmp_path / "v.in"
    vin.write_text("placeholder\n", encoding="utf-8")
    assert edit_path_for(vin) == Path(str(vin) + ".edit")
    edit = ensure_edit_copy(vin)
    assert edit.exists()
    assert edit.read_text(encoding="utf-8") == "placeholder\n"
    # 第二次不覆盖
    edit.write_text("kept\n", encoding="utf-8")
    ensure_edit_copy(vin)
    assert edit.read_text(encoding="utf-8") == "kept\n"


def test_model_edit_history_undo_redo(v1_path: Path) -> None:
    from pyAOBS.modeling.vedit.core import Model, ModelEditHistory, NodeIndex

    model = Model.load(str(v1_path))
    hist = ModelEditHistory()
    before = model.dumps()
    hist.checkpoint(model)
    # 垂直移动中间节点（允许）
    model.move_node(NodeIndex(0, 0, 1), 0.0, 0.05)
    after = model.dumps()
    assert after != before
    restored = hist.undo(model)
    assert restored is not None
    prev_model, _pois = restored
    assert prev_model.dumps() == before
    redone = hist.redo(prev_model)
    assert redone is not None
    nxt_model, _ = redone
    assert nxt_model.dumps() == after


def test_model_edit_history_pois_undo(v1_path: Path) -> None:
    from pyAOBS.modeling.vedit.core import Model, ModelEditHistory
    from pyAOBS.modeling.vedit.core.pois import PoisModel

    model = Model.load(str(v1_path))
    n = max(1, len(model) - 1)
    pois = PoisModel.from_arrays([0.25] * n)
    hist = ModelEditHistory()
    hist.checkpoint(model, pois)
    pois.set_block_nu(1, 1, 0.3)
    assert pois.effective_nu(1, 1, n_layers=n) == 0.3
    snap = hist.undo(model, pois)
    assert snap is not None
    _m, prev_pois = snap
    assert prev_pois is not None
    assert abs(prev_pois.effective_nu(1, 1, n_layers=n) - 0.25) < 1e-9


def test_pin_obs_seafloor(v1_path: Path) -> None:
    from pyAOBS.modeling.vedit.core import Model
    from pyAOBS.modeling.vedit.core.obs_seafloor import (
        check_obs_seafloor,
        pin_obs_seafloor,
    )

    model = Model.load(str(v1_path))
    assert len(model) > 1
    # 取界面1中点 x，故意给错误 zshot
    xs = list(model[1].depth.x)
    xmid = 0.5 * (float(xs[0]) + float(xs[-1]))
    shots = [(xmid, 99.0)]
    assert check_obs_seafloor(model, shots)
    msgs = pin_obs_seafloor(model, shots)
    assert msgs
    assert not check_obs_seafloor(model, shots)



def test_pack_ray_groups_delete_keeps_rbnd_cbnd() -> None:
    from pyAOBS.modeling.vedit.core.rin_ray_groups import (
        RayGroupSpec,
        assert_packed_consistent,
        pack_ray_groups,
        parse_ray_groups,
        resize_group_array,
        validate_ray_groups,
    )

    groups = [
        RayGroupSpec(1, 2.2, 2, 2, reflections=[2, -1], conversions=[], ivray=1),
        RayGroupSpec(2, 3.1, 3, 1, reflections=[], conversions=[1, 2], ivray=2),
        RayGroupSpec(3, 4.2, 4, 2, reflections=[4], conversions=[], ivray=3),
    ]
    # L.2 主反射已在 reflections[0]
    packed = pack_ray_groups(groups)
    assert assert_packed_consistent(packed) == []
    again = parse_ray_groups(
        packed["ray"],
        nrbnd=packed["nrbnd"],
        rbnd=packed["rbnd"],
        ncbnd=packed["ncbnd"],
        cbnd=packed["cbnd"],
        ivray=packed["ivray"],
    )
    assert validate_ray_groups(again) == []
    # 删中间组
    del groups[1]
    for i, g in enumerate(groups):
        g.index = i + 1
    packed2 = pack_ray_groups(groups)
    assert assert_packed_consistent(packed2) == []
    again2 = parse_ray_groups(
        packed2["ray"],
        nrbnd=packed2["nrbnd"],
        rbnd=packed2["rbnd"],
        ncbnd=packed2["ncbnd"],
        cbnd=packed2["cbnd"],
        ivray=packed2["ivray"],
    )
    assert len(again2) == 2
    assert again2[0].ivray == 1
    assert again2[1].ivray == 3
    assert again2[1].conversions == []
    # 平行数组按组删除
    space = [0.1, 0.2, 0.3]
    assert resize_group_array(
        space, n_old=3, n_new=2, delete_index=1, default=0.0
    ) == [0.1, 0.3]

