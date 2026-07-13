"""Fig.3 eq.(1) catalog bias sanity checks."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.vp_regression import catalog_eq1_bias, predict_vp_km_s_calibrated


def test_catalog_eq1_bias_reduces_mean_residual():
    cal = catalog_eq1_bias()
    assert cal["n_points"] >= 30
    assert cal["eq1_bias_km_s"] > 0.15
    assert abs(cal["mean_residual_calibrated_km_s"]) < 1e-9
    assert cal["rms_calibrated_km_s"] <= cal["rms_raw_km_s"] + 1e-9


def test_predict_vp_km_s_calibrated_matches_bias():
    cal = catalog_eq1_bias()
    bias = cal["eq1_bias_km_s"]
    raw = predict_vp_km_s_calibrated(1.0, 0.1, eq1_bias_km_s=0.0)
    adj = predict_vp_km_s_calibrated(1.0, 0.1, eq1_bias_km_s=bias)
    assert abs(adj - raw - bias) < 1e-9


if __name__ == "__main__":
    test_catalog_eq1_bias_reduces_mean_residual()
    test_predict_vp_km_s_calibrated_matches_bias()
    print("ok")
