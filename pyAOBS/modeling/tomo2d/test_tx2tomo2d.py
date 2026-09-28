# -*- coding: utf-8 -*-
"""tx2tomo2d 单元测试：单文件与多文件合并。"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyAOBS.modeling.tomo2d.tx2tomo2d import (
    convert_tx_in_to_tomo2d,
    list_obs_in_tx,
    parse_obs_id_spec,
    parse_tx_in_list,
)


def _write(p: Path, text: str) -> None:
    p.write_text(text, encoding="utf-8")


def test_parse_tx_in_list_separators() -> None:
    assert parse_tx_in_list("a.in") == ["a.in"]
    assert parse_tx_in_list("a.in\nb.in") == ["a.in", "b.in"]
    assert parse_tx_in_list("a.in;b.in") == ["a.in", "b.in"]
    assert parse_tx_in_list(["a.in", "b.in"]) == ["a.in", "b.in"]


def test_merge_two_tx_in(tmp_path: Path) -> None:
    station = tmp_path / "station.lis"
    tx1 = tmp_path / "tx1.in"
    tx2 = tmp_path / "tx2.in"
    data = tmp_path / "ttimes.dat"
    geom = tmp_path / "geom.dat"

    # 两炮：x=1.0 / x=2.0
    _write(station, "1 1.000 0.000\n2 2.000 0.000\n")
    # 文件1：仅炮1的折射拾取
    _write(
        tx1,
        "    1.000    -1.000     0.000         0\n"
        "    1.100     0.500     0.050         1\n"
        "    1.200     0.600     0.050         1\n"
        "   -1.000    -1.000    -1.000        -1\n",
    )
    # 文件2：仅炮2的折射拾取
    _write(
        tx2,
        "    2.000    -1.000     0.000         0\n"
        "    2.100     0.700     0.050         1\n"
        "   -1.000    -1.000    -1.000        -1\n",
    )

    stats = convert_tx_in_to_tomo2d(
        station,
        [tx1, tx2],
        data,
        geom,
        refr_phases={1},
        refl_phases=set(),
    )
    assert stats.n_tx_files == 2
    assert stats.nshot == 2
    assert stats.ntime == 3

    dlines = data.read_text(encoding="utf-8").splitlines()
    assert dlines[0] == "2"
    # 两个 s 头 + 3 条 r
    assert sum(1 for ln in dlines if ln.startswith("s")) == 2
    assert sum(1 for ln in dlines if ln.startswith("r")) == 3

    glines = geom.read_text(encoding="utf-8").splitlines()
    assert glines[0] == "2"
    # geom 中 r 行 t/u 为 0
    r_geom = [ln for ln in glines if ln.startswith("r")]
    assert all(ln.endswith("     0.000     0.000") for ln in r_geom)


def test_parse_obs_id_spec() -> None:
    assert parse_obs_id_spec(None) is None
    assert parse_obs_id_spec("") is None
    assert parse_obs_id_spec("all") is None
    assert parse_obs_id_spec("none") == set()
    assert parse_obs_id_spec("12, 30") == {12, 30}
    assert parse_obs_id_spec([7, 8]) == {7, 8}


def test_list_obs_and_include_obs_filter(tmp_path: Path) -> None:
    station = tmp_path / "station.lis"
    tx1 = tmp_path / "tx1.in"
    tx2 = tmp_path / "tx2.in"
    data = tmp_path / "ttimes.dat"
    geom = tmp_path / "geom.dat"
    _write(station, "12 1.000 0.500\n30 2.000 0.800\n")
    _write(
        tx1,
        "    1.000    -1.000     0.000         0\n"
        "    1.100     0.500     0.050         1\n"
        "    1.200     0.600     0.050         1\n"
        "   -1.000    -1.000    -1.000        -1\n",
    )
    _write(
        tx2,
        "    2.000    -1.000     0.000         0\n"
        "    2.100     0.700     0.050         1\n"
        "    9.000    -1.000     0.000         0\n"
        "    9.100     0.100     0.050         1\n"
        "   -1.000    -1.000    -1.000        -1\n",
    )
    rows, unmatched = list_obs_in_tx(station, [tx1, tx2])
    assert unmatched == 1
    assert [r.obs_id for r in rows] == [12, 30]
    assert rows[0].n_picks == 2 and rows[0].n_blocks == 1
    assert rows[1].n_picks == 1 and rows[1].n_blocks == 1

    stats = convert_tx_in_to_tomo2d(
        station,
        [tx1, tx2],
        data,
        geom,
        refr_phases={1},
        refl_phases=set(),
        include_obs={12},
    )
    assert stats.nshot == 1
    assert stats.ntime == 2
    dlines = data.read_text(encoding="utf-8").splitlines()
    assert dlines[0] == "1"
    assert sum(1 for ln in dlines if ln.startswith("s")) == 1
    assert sum(1 for ln in dlines if ln.startswith("r")) == 2


def test_include_obs_empty_raises(tmp_path: Path) -> None:
    station = tmp_path / "station.lis"
    tx = tmp_path / "tx.in"
    _write(station, "1 1.000 0.000\n")
    _write(
        tx,
        "    1.000    -1.000     0.000         0\n"
        "    1.100     0.500     0.050         1\n"
        "   -1.000    -1.000    -1.000        -1\n",
    )
    with pytest.raises(ValueError, match="未选择"):
        convert_tx_in_to_tomo2d(
            station,
            tx,
            tmp_path / "d.dat",
            tmp_path / "g.dat",
            refr_phases={1},
            refl_phases=set(),
            include_obs=[],
        )


def test_single_tx_in_still_works(tmp_path: Path) -> None:
    station = tmp_path / "station.lis"
    tx = tmp_path / "tx.in"
    data = tmp_path / "ttimes.dat"
    geom = tmp_path / "geom.dat"
    _write(station, "1 1.000 0.000\n")
    _write(
        tx,
        "    1.000    -1.000     0.000         0\n"
        "    1.100     0.500     0.050         1\n"
        "   -1.000    -1.000    -1.000        -1\n",
    )
    stats = convert_tx_in_to_tomo2d(
        station, tx, data, geom, refr_phases={1}, refl_phases=set()
    )
    assert stats.n_tx_files == 1
    assert stats.nshot == 1
    assert stats.ntime == 1
