"""modeling.rayinvr.tx_io 统一解析测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyAOBS.modeling.rayinvr.tx_io import (
    format_tx_line,
    parse_tx_line,
    read_tx_file,
    validate_tx_file,
    write_tx_file,
    write_tx_from_picks,
)

SAMPLE = Path(__file__).resolve().parents[1] / "modeling" / "rayinvr" / "tx.in"


def test_parse_fixed_width_and_free() -> None:
    fixed = format_tx_line(436.91, -1.0, 0.0, 0)
    assert len(fixed) == 40
    assert parse_tx_line(fixed) == (436.91, -1.0, 0.0, 0)
    free = "10.5  1.2  0.05  3"
    assert parse_tx_line(free) == (10.5, 1.2, 0.05, 3)


def test_read_sample_tx_in() -> None:
    assert SAMPLE.is_file(), f"missing sample: {SAMPLE}"
    ds = read_tx_file(SAMPLE)
    assert ds.n_shots >= 1
    assert ds.n_picks > 0
    assert abs(ds.shots[0].xshot - 436.910) < 1e-3
    zplot = ds.to_zplot_dict()
    assert zplot["total_observations"] == ds.n_picks
    assert zplot["shots"][0]["shot_position"] == ds.shots[0].xshot
    iphase = ds.to_iphase_by_shot()
    assert iphase[0]["shot_x"] == ds.shots[0].xshot
    assert len(iphase[0]["obs"]) == len(ds.shots[0].picks)
    flat = ds.to_flat_arrays()
    assert flat is not None
    assert flat["x"].shape == flat["t"].shape == flat["phase"].shape
    assert flat["phase"].dtype == np.int32


def test_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "tx.in"
    write_tx_from_picks(
        out,
        shot_x=100.0,
        picks=[(101.0, 1.5, 0.05, 1), (102.0, 1.6, 0.05, 1)],
    )
    ok, err = validate_tx_file(out)
    assert ok and err is None
    ds = read_tx_file(out)
    assert ds.n_shots == 1
    assert ds.n_picks == 2
    out2 = tmp_path / "tx2.in"
    write_tx_file(ds, out2)
    ds2 = read_tx_file(out2)
    assert ds2.n_picks == 2
    assert abs(ds2.shots[0].picks[0].t - 1.5) < 1e-9


def test_iphase_read_tx_adapter() -> None:
    from pyAOBS.visualization.iphase.io_tx import read_tx, write_tx

    ds = read_tx(SAMPLE)
    assert ds.n_shots >= 1
    assert ds.n_picks > 0


def test_theory2d_parse_alias() -> None:
    from pyAOBS.visualization.iphase.theory2d_service import _parse_tx_file_by_shot

    shots = _parse_tx_file_by_shot(SAMPLE)
    assert shots and "shot_x" in shots[0] and "obs" in shots[0]
    assert len(shots[0]["obs"]) > 0
