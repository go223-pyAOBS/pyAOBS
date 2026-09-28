"""走时预览：同一 OBS 左右支合并。"""
from __future__ import annotations

import pytest

from pyAOBS.modeling.tomo2d.gui.services.tt_plot_data import (
    TtPick,
    group_obs_by_shot_x,
    picks_to_arrays,
)

pytestmark = pytest.mark.unit


def test_group_obs_merges_left_right_branches() -> None:
    picks = [
        TtPick(shot_x=10.0, rcv_x=2.0, t=1.0, u=0.02, code=1, shot_idx=0),
        TtPick(shot_x=10.0, rcv_x=4.0, t=1.1, u=0.02, code=1, shot_idx=0),
        TtPick(shot_x=10.0, rcv_x=16.0, t=1.2, u=0.02, code=1, shot_idx=1),
        TtPick(shot_x=10.0, rcv_x=18.0, t=1.3, u=0.02, code=1, shot_idx=1),
        TtPick(shot_x=25.0, rcv_x=20.0, t=0.8, u=0.02, code=1, shot_idx=2),
    ]
    groups = group_obs_by_shot_x(picks_to_arrays(picks))
    assert [g["xobs"] for g in groups] == [10.0, 25.0]
    g0 = groups[0]
    assert g0["shot_idxs"] == [0, 1]
    assert g0["n"] == 4
    assert g0["n_left"] == 2
    assert g0["n_right"] == 2
    assert groups[1]["n"] == 1
    assert groups[1]["shot_idxs"] == [2]
    assert g0["obs_id"] is None
    assert groups[1]["obs_id"] is None


def test_group_obs_matches_station_lis_id() -> None:
    picks = [
        TtPick(shot_x=10.0, rcv_x=2.0, t=1.0, u=0.02, code=1, shot_idx=0),
        TtPick(shot_x=10.0, rcv_x=16.0, t=1.2, u=0.02, code=1, shot_idx=1),
        TtPick(shot_x=25.0004, rcv_x=20.0, t=0.8, u=0.02, code=1, shot_idx=2),
        TtPick(shot_x=99.0, rcv_x=90.0, t=0.5, u=0.02, code=1, shot_idx=3),
    ]
    stations = [(12, 10.0, 0.5), (30, 25.0, 1.0)]
    groups = group_obs_by_shot_x(picks_to_arrays(picks), stations=stations)
    by_x = {round(g["xobs"], 3): g for g in groups}
    assert by_x[10.0]["obs_id"] == 12
    assert by_x[25.0]["obs_id"] == 30
    assert by_x[99.0]["obs_id"] is None


def test_build_obs_catalog_label_and_sort() -> None:
    from pyAOBS.modeling.tomo2d.gui.services.tt_plot_data import build_obs_catalog

    picks = [
        TtPick(shot_x=10.0, rcv_x=2.0, t=1.0, u=0.02, code=1, shot_idx=0),
        TtPick(shot_x=10.0, rcv_x=16.0, t=1.2, u=0.02, code=1, shot_idx=1),
        TtPick(shot_x=25.0, rcv_x=20.0, t=0.8, u=0.02, code=1, shot_idx=2),
    ]
    stations = [(30, 25.0, 1.0), (12, 10.0, 0.5)]
    cat = build_obs_catalog(picks_to_arrays(picks), stations)
    assert [c["obs_id"] for c in cat] == [12, 30]
    assert cat[0]["label"].startswith("12  x=10.000 km")
    assert "L1/R1" in cat[0]["label"]
    assert cat[1]["label"].startswith("30  x=25.000 km")


def test_build_phase_catalog_tx_and_ttimes() -> None:
    from pyAOBS.modeling.tomo2d.gui.services.tt_plot_data import build_phase_catalog

    picks = [
        TtPick(shot_x=10.0, rcv_x=2.0, t=1.0, u=0.02, code=1, shot_idx=0),
        TtPick(shot_x=10.0, rcv_x=16.0, t=1.2, u=0.02, code=1, shot_idx=1),
        TtPick(shot_x=25.0, rcv_x=20.0, t=0.8, u=0.02, code=11, shot_idx=2),
    ]
    cat = build_phase_catalog(picks_to_arrays(picks), kind="tx")
    assert [c["code"] for c in cat] == [1, 11]
    assert cat[0]["n"] == 2 and cat[0]["label"].startswith("1  (n=2)")
    assert cat[1]["n"] == 1 and "11" in cat[1]["label"]

    tt = [
        TtPick(shot_x=10.0, rcv_x=2.0, t=1.0, u=0.02, code=0, shot_idx=0),
        TtPick(shot_x=10.0, rcv_x=16.0, t=1.2, u=0.02, code=1, shot_idx=0),
    ]
    tcat = build_phase_catalog(picks_to_arrays(tt), kind="ttimes")
    assert [c["code"] for c in tcat] == [0, 1]
    assert "折射" in tcat[0]["label"]
    assert "反射" in tcat[1]["label"]
