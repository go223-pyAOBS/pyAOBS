"""modeling.rayinvr.vin_io 统一 v.in 解析测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyAOBS.modeling.rayinvr.vin_io import (
    edit_model_to_vin_dict,
    is_vin_file,
    load_edit_model,
    load_zelt_model,
    read_vin_dict,
    vin_dict_to_edit_model,
    write_vin_dict,
)
from pyAOBS.modeling.vedit.model import Model

SAMPLE = Path(__file__).resolve().parents[1] / "modeling" / "vedit" / "examples" / "v1.in"
if not SAMPLE.is_file():
    SAMPLE = Path(__file__).resolve().parents[1] / "modeling" / "rayinvr" / "v.in"


def test_is_vin_file() -> None:
    assert SAMPLE.is_file()
    assert is_vin_file(SAMPLE)
    assert not is_vin_file(SAMPLE.parent / "no_such_file.in")


def test_load_edit_matches_legacy_loads() -> None:
    legacy = Model.loads(SAMPLE.read_text(encoding="utf-8", errors="replace"))
    unified = load_edit_model(SAMPLE)
    assert len(unified) == len(legacy)
    for i in range(len(legacy) - 1):
        assert len(unified[i].depth) == len(legacy[i].depth)
        assert np.allclose(unified[i].depth.x, legacy[i].depth.x)
        assert np.allclose(unified[i].depth.y, legacy[i].depth.y)
        assert np.allclose(unified[i].v_top.y, legacy[i].v_top.y)
        assert np.allclose(unified[i].v_bot.y, legacy[i].v_bot.y)
    assert np.allclose(unified[-1].depth.x, legacy[-1].depth.x)
    assert np.allclose(unified[-1].depth.y, legacy[-1].depth.y)


def test_model_load_uses_vin_io() -> None:
    m = Model.load(str(SAMPLE))
    assert len(m) >= 2
    assert m.nlayer == len(m)


def test_dict_roundtrip(tmp_path: Path) -> None:
    d = read_vin_dict(SAMPLE)
    out = tmp_path / "v_round.in"
    write_vin_dict(out, d)
    d2 = read_vin_dict(out)
    assert len(d2["layer_boundary_x"]) == len(d["layer_boundary_x"])
    m = vin_dict_to_edit_model(d2)
    d3 = edit_model_to_vin_dict(m)
    assert len(d3["layer_boundary_x"]) == len(d["layer_boundary_x"])


def test_load_zelt_model() -> None:
    z = load_zelt_model(SAMPLE)
    xmin, xmax, zmin, zmax = z.get_model_bounds()
    assert xmax > xmin
    assert zmax >= zmin
