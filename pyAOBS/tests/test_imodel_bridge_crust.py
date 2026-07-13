"""Tests for imodel ↔ petrology crust metrics bridge."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from petrology.imodel_bridge import (
    CrustObservation,
    load_observation_json,
    metrics_from_depth_samples,
    save_observation_json,
    thickness_from_interfaces,
)
from petrology.imodel_bridge.crust_geometry import f_lower_from_lc_top
from petrology.seismic.transect import harmonic_mean_km_s

Z_B = 5.0
Z_LC = 12.0
Z_M = 35.0
H = Z_M - Z_B
F_LOWER = f_lower_from_lc_top(
    z_basement_km=Z_B, z_moho_km=Z_M, z_lc_top_km=Z_LC
)


def test_harmonic_mean_basic():
    v = np.array([7.0, 7.2, 6.9])
    assert abs(harmonic_mean_km_s(v) - 3.0 / np.sum(1.0 / v)) < 1e-9


def test_f_lower_from_geometry():
    assert abs(F_LOWER - (Z_M - Z_LC) / H) < 1e-9


def test_metrics_from_depth_samples():
    z_top = (1.0 - F_LOWER) * H
    depths = np.linspace(z_top + 0.5, H - 0.5, 6)
    vp = np.full_like(depths, 7.0)
    obs = metrics_from_depth_samples(
        depths,
        vp,
        h_whole_km=H,
        f_lower=F_LOWER,
        pt_correct=False,
    )
    assert abs(obs.v_lc_km_s - 7.0) < 1e-6
    assert obs.n_samples == 6
    assert obs.h_whole_km == H


def test_observation_json_roundtrip():
    obs = CrustObservation(h_whole_km=H, v_lc_km_s=7.01, f_lower=F_LOWER, source="test")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "obs.json"
        save_observation_json(obs, path)
        loaded = load_observation_json(path)
    assert loaded.h_whole_km == obs.h_whole_km
    assert loaded.v_lc_km_s == obs.v_lc_km_s
    assert loaded.source == "test"


def test_thickness_from_interfaces():
    h = thickness_from_interfaces(
        10.0,
        igneous_top_z_fn=lambda x: Z_B,
        moho_z_fn=lambda x: Z_M,
    )
    assert h == H


def test_crust_from_interfaces_at_x():
    from petrology.imodel_bridge.imodel_adapter import crust_from_interfaces_at_x

    def moho(_x):
        return Z_M

    def basement(_x):
        return Z_B

    def lc_top(_x):
        return Z_LC

    def vp(_x, _z):
        return 7.0

    obs = crust_from_interfaces_at_x(
        10.0,
        moho_z_fn=moho,
        igneous_top_z_fn=basement,
        lower_crust_top_z_fn=lc_top,
        grid_vp_fn=vp,
        pt_correct=False,
    )
    assert abs(obs.h_whole_km - H) < 1e-6
    assert abs(obs.v_lc_km_s - 7.0) < 1e-6
    assert abs(obs.f_lower - F_LOWER) < 1e-6
    assert obs.source == "imodel_interfaces"


def test_crust_from_interfaces_basement_moho_not_seafloor():
    """H = M − B; seafloor must not affect igneous column."""
    from petrology.imodel_bridge.imodel_adapter import crust_from_interfaces_at_x

    z_seafloor = 2.0

    def moho(_x):
        return Z_M

    def basement(_x):
        return Z_B

    def lc_top(_x):
        return Z_LC

    def vp(_x, _z):
        return 7.2

    obs = crust_from_interfaces_at_x(
        10.0,
        moho_z_fn=moho,
        igneous_top_z_fn=basement,
        lower_crust_top_z_fn=lc_top,
        grid_vp_fn=vp,
        pt_correct=False,
    )
    assert abs(obs.h_whole_km - H) < 1e-6
    assert abs(obs.v_lc_km_s - 7.2) < 1e-6
    assert obs.n_samples is not None and obs.n_samples >= 2


def test_profile_average_relative_to_basement():
    from petrology.imodel_bridge.imodel_adapter import crust_from_vertical_profile

    z_top = Z_B + (1.0 - F_LOWER) * H
    depths = np.linspace(z_top + 0.2, Z_B + H - 0.2, 8)
    vp = np.full_like(depths, 6.95)
    obs = crust_from_vertical_profile(
        depths,
        vp,
        h_whole_km=H,
        f_lower=F_LOWER,
        z_surface_km=Z_B,
        pt_correct=False,
        source="imodel_profile",
    )
    assert abs(obs.v_lc_km_s - 6.95) < 1e-6
    assert obs.n_samples >= 2


def test_polygon_top_depth_at_x_not_global_min():
    from petrology.imodel_bridge.crust_geometry import (
        f_lower_from_polygon_at_x,
        polygon_top_depth_at_x,
    )

    # Trapezoid: top edge at x=10 is z=12, but global min vertex is at x=5,z=10
    poly = [(0.0, 20.0), (5.0, 10.0), (10.0, 12.0), (20.0, 20.0), (10.0, 25.0)]
    assert abs(polygon_top_depth_at_x(poly, 10.0) - 12.0) < 1e-6
    assert abs(polygon_top_depth_at_x(poly, 0.0) - 20.0) < 1e-6

    z_b, z_m = 5.0, 35.0
    _h, h_lc, f_lo = f_lower_from_polygon_at_x(
        poly, 10.0, z_basement_km=z_b, z_moho_km=z_m
    )
    assert abs(_h - 30.0) < 1e-6
    assert abs(h_lc - abs(z_m - 12.0)) < 1e-6
    assert abs(f_lo - h_lc / _h) < 1e-6


def test_polygon_effective_x_range_intersection():
    from petrology.imodel_bridge.crust_geometry import polygon_effective_x_range

    poly = [(0.0, 10.0), (50.0, 10.0), (50.0, 30.0), (0.0, 30.0)]
    lo, hi = polygon_effective_x_range(poly, x_min_km=10.0, x_max_km=40.0)
    assert abs(lo - 10.0) < 1e-9
    assert abs(hi - 40.0) < 1e-9

    lo2, hi2 = polygon_effective_x_range(poly)
    assert abs(lo2 - 0.0) < 1e-9
    assert abs(hi2 - 50.0) < 1e-9


def test_mc_sigma_sparse_depth_resampling():
    from petrology.seismic.transect import (
        TransectSample,
        mc_harmonic_sigma_lc_samples,
        mc_harmonic_sigma_stations,
    )

    flat_lc = []
    for x in (0.0, 10.0, 20.0):
        for _ in range(12):
            flat_lc.append(
                TransectSample(
                    x, 18.0, 7.0, 30.0, 7.00, True, f_lower=0.7
                )
            )
    sigma_old = mc_harmonic_sigma_stations(
        np.array([7.0, 7.0, 7.0]), n_mc=200, rng=np.random.default_rng(0)
    )
    sigma_flat = mc_harmonic_sigma_lc_samples(
        flat_lc, n_mc=200, vp_pick_sigma_km_s=0.0, rng=np.random.default_rng(0)
    )
    assert sigma_old < 1e-6
    assert sigma_flat < 1e-6

    spread_lc = []
    for x in (0.0, 10.0, 20.0):
        for z, vp in ((18.0, 6.5), (20.0, 6.8), (22.0, 7.1)):
            spread_lc.append(
                TransectSample(x, z, vp, 30.0, vp, True, f_lower=0.7)
            )
    sigma_spread = mc_harmonic_sigma_lc_samples(
        spread_lc,
        n_mc=500,
        mc_depth_draws_per_station=1,
        vp_pick_sigma_km_s=0.0,
        rng=np.random.default_rng(42),
    )
    assert 0.05 < sigma_spread < 0.35

    sigma_two_stations = mc_harmonic_sigma_lc_samples(
        [
            TransectSample(0.0, 18.0, 7.0, 30.0, 6.95, True, 0.7),
            TransectSample(0.0, 20.0, 7.0, 30.0, 7.05, True, 0.7),
            TransectSample(10.0, 18.0, 7.0, 30.0, 7.10, True, 0.7),
            TransectSample(10.0, 20.0, 7.0, 30.0, 7.12, True, 0.7),
        ],
        n_mc=200,
        vp_pick_sigma_km_s=0.0,
        rng=np.random.default_rng(1),
    )
    assert sigma_two_stations > 0.01


def test_mc_sigma_uses_distance_stations_not_depth_pool():
    from petrology.seismic.transect import (
        TransectSample,
        aggregate_transect_windows,
        lc_station_harmonic_means,
        mc_harmonic_sigma_stations,
    )

    lc = [
        TransectSample(0.0, 20.0, 7.0, 30.0, 7.00, True, f_lower=0.7),
        TransectSample(0.0, 22.0, 7.0, 30.0, 7.00, True, f_lower=0.7),
        TransectSample(10.0, 20.0, 7.0, 30.0, 7.10, True, f_lower=0.7),
        TransectSample(10.0, 22.0, 7.0, 30.0, 7.10, True, f_lower=0.7),
    ]
    stations = lc_station_harmonic_means(lc)
    assert stations.size == 2
    sigma_st = mc_harmonic_sigma_stations(stations, n_mc=200, rng=np.random.default_rng(0))
    # depth pool (4 nearly duplicate points) would give much smaller sigma
    v_all = np.array([s.vp_ref_km_s for s in lc], dtype=float)
    rng = np.random.default_rng(0)
    mc = np.empty(200)
    for i in range(200):
        idx = rng.integers(0, 4, size=4)
        mc[i] = float(4 / np.sum(1.0 / v_all[idx]))
    sigma_depth = float(np.std(mc, ddof=1))
    assert sigma_st >= sigma_depth
    assert abs(sigma_st - 0.036) < 0.02

    samples = lc + [
        TransectSample(0.0, 5.0, 7.5, 30.0, 7.5, False, f_lower=0.7),
        TransectSample(10.0, 5.0, 7.5, 30.0, 7.5, False, f_lower=0.7),
    ]
    windows = aggregate_transect_windows(
        samples,
        window_half_width_km=10.0,
        distance_step_km=10.0,
        distance_range_km=(5.0, 5.0),
    )
    assert len(windows) == 1
    assert windows[0].n_stations == 2
    assert windows[0].v_lc_sigma_km_s > 0.01


def test_parse_digitized_fig12a():
    from petrology.hvp.fig12a_background import DEFAULT_HVP_DIGITIZED, parse_hvp_digitized

    series = parse_hvp_digitized(DEFAULT_HVP_DIGITIZED)
    assert len(series) >= 10
    kinds = {s.kind for s in series}
    assert "track" in kinds and "tp" in kinds
