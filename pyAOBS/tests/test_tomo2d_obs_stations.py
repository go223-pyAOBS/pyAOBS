"""station.lis OBS 号与模型距离匹配。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pyAOBS.modeling.tomo2d.gui.services.obs_stations import (
    list_tomo2d_sources,
    load_obs_context,
    match_isrc_to_obs,
    nearest_obs_id,
    parse_station_lis,
)
from pyAOBS.modeling.tomo2d.gui.state.form_state import FormState

pytestmark = pytest.mark.unit


def test_parse_station_and_match_by_x(tmp_path: Path) -> None:
    p = tmp_path / "station.lis"
    p.write_text(
        "01 408.680 2.38\n"
        "30 282.729 1.02\n"
        "38 234.665 3.57\n",
        encoding="utf-8",
    )
    rows = parse_station_lis(p)
    assert rows[0] == (1, pytest.approx(408.680), pytest.approx(2.38))
    assert nearest_obs_id(282.729, rows) == 30
    assert nearest_obs_id(234.67, rows, tol=0.05) == 38
    assert nearest_obs_id(100.0, rows) is None


def test_list_sources_and_isrc_map(tmp_path: Path) -> None:
    data = tmp_path / "ttimes.dat"
    data.write_text(
        "2\n"
        "s   282.729     1.030    1\n"
        "r    92.765     0.010    0     1.000     0.060\n"
        "s   234.665     3.580    0\n",
        encoding="utf-8",
    )
    src = list_tomo2d_sources(data)
    assert [isrc for isrc, _, _ in src] == [1, 2]
    assert src[0][1] == pytest.approx(282.729)
    stations = [(30, 282.729, 1.02), (38, 234.665, 3.57)]
    mp = match_isrc_to_obs(src, stations)
    assert mp == {1: 30, 2: 38}


def test_load_obs_context_from_form(tmp_path: Path) -> None:
    (tmp_path / "inputs").mkdir()
    stn = tmp_path / "inputs" / "station.lis"
    stn.write_text("30 282.729 1.02\n38 234.665 3.57\n", encoding="utf-8")
    dat = tmp_path / "ttimes.dat"
    dat.write_text(
        "1\n"
        "s   282.729     1.030    0\n",
        encoding="utf-8",
    )
    state = FormState(
        {
            "tx.station_lis": "inputs/station.lis",
            "inv.data": "ttimes.dat",
        }
    )
    ctx = load_obs_context(state, tmp_path)
    assert ctx.obs_id(1) == 30
    assert ctx.label(1) == "OBS 30"
    assert ctx.label(9) == "OBS 9"
    assert len(ctx.stations) == 2
