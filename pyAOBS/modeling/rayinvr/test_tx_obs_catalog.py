# -*- coding: utf-8 -*-
"""``tx_obs_catalog`` 单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyAOBS.modeling.rayinvr.tx_io import read_tx_file
from pyAOBS.modeling.rayinvr.tx_obs_catalog import (
    count_lr_picks,
    format_obs_catalog_label,
    group_tx_dataset_by_obs_x,
    obs_groups_from_tx_path,
)
from pyAOBS.modeling.vedit.core.obs_catalog import build_obs_catalog


def test_count_lr_picks():
    n, nl, nr = count_lr_picks(10.0, [8.0, 9.0, 11.0, 12.0], tol=0.001)
    assert n == 4
    assert nl == 2
    assert nr == 2


def test_format_obs_catalog_label_lr():
    s = format_obs_catalog_label(obs_id=3, xobs=30.818, n=12, n_left=5, n_right=7)
    assert s == "3  x=30.818 km  (n=12)  L5/R7"


def test_haiti_tx_merges_61_obs():
    p = Path(__file__).resolve().parents[1] / "vedit" / "haiti" / "cache" / "rayinvr" / "tx.in"
    if not p.is_file():
        pytest.skip("haiti tx.in not present")
    ds = read_tx_file(p)
    assert len(ds.shots) == 122
    groups = group_tx_dataset_by_obs_x(ds)
    assert len(groups) == 61
    assert all(g.n_branches >= 1 for g in groups)
    assert sum(g.n_picks for g in groups) == ds.n_picks


def test_vedit_catalog_labels_match_iphase_style():
    p = Path(__file__).resolve().parents[1] / "vedit" / "haiti" / "cache" / "rayinvr" / "tx.in"
    if not p.is_file():
        pytest.skip("haiti tx.in not present")
    cat = build_obs_catalog(tx_in=p)
    assert len(cat) == 61
    sample = next(e for e in cat if e.n_left > 0 and e.n_right > 0)
    lbl = sample.label()
    assert f"L{sample.n_left}/R{sample.n_right}" in lbl
    assert "km" in lbl
    assert "(tx.in)" not in lbl


def test_split_tx_dataset_by_obs_x():
    p = Path(__file__).resolve().parents[1] / "vedit" / "haiti" / "cache" / "rayinvr" / "tx.in"
    if not p.is_file():
        pytest.skip("haiti tx.in not present")
    from pyAOBS.modeling.rayinvr.tx_io import read_tx_file
    from pyAOBS.modeling.rayinvr.tx_obs_catalog import split_tx_dataset_by_obs_x

    ds = read_tx_file(p)
    parts = split_tx_dataset_by_obs_x(ds)
    assert len(parts) == 61
    assert sum(sub.n_picks for _x, sub in parts) == ds.n_picks


def test_iphase_split_phase_dataset_by_obs_x():
    p = Path(__file__).resolve().parents[1] / "vedit" / "haiti" / "cache" / "rayinvr" / "tx.in"
    if not p.is_file():
        pytest.skip("haiti tx.in not present")
    from pyAOBS.visualization.iphase.gui.services.file_result import (
        split_phase_dataset_by_obs_x,
    )
    from pyAOBS.visualization.iphase.io_tx import read_tx

    ds = read_tx(p)
    parts = split_phase_dataset_by_obs_x(ds)
    assert len(parts) == 61
    assert sum(part.n_picks for _x, part in parts) == ds.n_picks
