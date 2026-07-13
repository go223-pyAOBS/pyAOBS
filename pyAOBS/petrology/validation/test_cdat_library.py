"""CDAT sample library tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.cdat_library import (
    CDAT_DIR,
    build_library,
    fig2_primary_melt,
    get_csj,
    get_sample,
    list_samples,
    load_cdat_file,
    resolve_cdat_path,
    write_all_samples_cdat,
)


def test_builtin_samples_normalize():
    for s in list_samples():
        csj = s.csj()
        assert abs(float(np.sum(csj[:6])) - 1.0) < 1e-6


def test_build_library_files():
    paths = build_library(force=True)
    assert paths["catalog"].is_file()
    assert (CDAT_DIR / "kinzler1997_morb_primary.cdat").is_file()
    assert (CDAT_DIR / "all_samples.cdat").is_file()
    cat = json.loads(paths["catalog"].read_text(encoding="utf-8"))
    assert len(cat["samples"]) == len(list_samples())


def test_load_all_samples_roundtrip():
    build_library(force=True)
    rows = load_cdat_file(CDAT_DIR / "all_samples.cdat")
    assert len(rows) == len(list_samples())
    ref = get_csj("magnesian_tholeiite")
    mag_rows = [r for r in rows if np.allclose(r, ref, atol=1e-5)]
    assert len(mag_rows) == 1


def test_resolve_cdat_path_by_id():
    build_library(force=True)
    p = resolve_cdat_path("picrite_high_mg")
    assert p.name == "picrite_high_mg.cdat"
    rows = load_cdat_file(p)
    assert len(rows) == 1


def test_kinzler_matches_json():
    s = get_sample("kinzler1997_morb_primary")
    p = fig2_primary_melt(sample_id="kinzler1997_morb_primary")
    assert p["id"] == "kinzler1997_morb_primary"
    json_path = CDAT_DIR.parent / "mantle_melts" / "kinzler1997_morb_primary.json"
    if json_path.is_file():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        from petrology.fc.wl_components import normalize_melt_oxides

        ref = normalize_melt_oxides(data["oxides_wt_percent"])["SiO2"]
        assert abs(p["oxides_wt_percent"]["SiO2"] - ref) < 1e-6
    else:
        assert abs(p["oxides_wt_percent"]["SiO2"] - s.oxides_normalized()["SiO2"]) < 0.5


def test_fig2_primary_melt_other_sample():
    p = fig2_primary_melt(sample_id="magnesian_tholeiite")
    assert p["id"] == "magnesian_tholeiite"
    assert p["oxides_wt_percent"]["MgO"] > 20.0


if __name__ == "__main__":
    test_builtin_samples_normalize()
    test_build_library_files()
    test_load_all_samples_roundtrip()
    test_resolve_cdat_path_by_id()
    test_kinzler_matches_json()
    test_fig2_primary_melt_other_sample()
    print("ok")
