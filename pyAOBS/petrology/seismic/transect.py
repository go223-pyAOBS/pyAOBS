"""Along-transect lower-crust Vp aggregation (Fig.15a → 15c)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from petrology.seismic.reference_state import correct_vp_depth_to_reference_km_s

DEFAULT_VP_PICK_SIGMA_KM_S = 0.0
DEFAULT_MC_DEPTH_DRAWS_PER_STATION = 1


@dataclass(frozen=True)
class TransectSample:
    distance_km: float
    depth_km: float
    vp_insitu_km_s: float
    h_whole_km: float
    vp_ref_km_s: float
    in_lower_crust: bool
    f_lower: float | None = None


@dataclass(frozen=True)
class TransectWindow:
    """One horizontal window: (H, V_LC) for Fig.15c on Fig.12a."""

    distance_km: float
    h_whole_km: float
    h_lower_km: float
    v_lc_km_s: float
    v_lc_sigma_km_s: float
    n_samples: int
    n_stations: int
    thick_crust: bool
    v_bulk_upper_km_s: float
    v_bulk_lower_km_s: float
    f_lower: float | None = None


def harmonic_mean_km_s(velocities_km_s: np.ndarray) -> float:
    v = np.asarray(velocities_km_s, dtype=float)
    v = v[np.isfinite(v) & (v > 0.0)]
    if v.size == 0:
        return float("nan")
    return float(v.size / np.sum(1.0 / v))


def lc_station_harmonic_means(lc_samples: list[TransectSample]) -> np.ndarray:
    """
    Per-profile-distance harmonic mean V_LC (Fig.15: average at each model distance).

    Multiple depth samples at the same ``distance_km`` are merged before window MC.
    """
    by_dist: dict[float, list[float]] = {}
    for s in lc_samples:
        by_dist.setdefault(float(s.distance_km), []).append(float(s.vp_ref_km_s))
    if not by_dist:
        return np.array([], dtype=float)
    return np.array(
        [
            harmonic_mean_km_s(np.asarray(vs, dtype=float))
            for _d, vs in sorted(by_dist.items())
        ],
        dtype=float,
    )


def mc_harmonic_sigma_lc_samples(
    lc_samples: list[TransectSample],
    *,
    n_mc: int = 100,
    mc_depth_draws_per_station: int = DEFAULT_MC_DEPTH_DRAWS_PER_STATION,
    vp_pick_sigma_km_s: float = DEFAULT_VP_PICK_SIGMA_KM_S,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Fig.15 MC uncertainty: sparse depth resampling within lower crust (primary).

    Smooth tomography grids often have many nearly identical depth samples at one x;
    bootstrapping the full depth pool barely changes σ. The main spread comes from
    which depths within the LC column contribute (e.g. Vp 6.5 at LC top vs 7.1 near
    Moho → harmonic mean ~6.8, ensemble range ~O(0.1–0.3) km/s). Each ensemble:

    1. Per distance: draw ``mc_depth_draws_per_station`` random depth Vp(s) with
       replacement from that station's LC sample pool.
    2. Station V_LC = harmonic mean of drawn depths.
    3. Window V_LC = harmonic mean of station values (bootstrap stations if ≥2).

    Optional ``vp_pick_sigma_km_s`` adds Gaussian pick noise (default 0).
    """
    by_dist: dict[float, np.ndarray] = {}
    for s in lc_samples:
        d = float(s.distance_km)
        arr = by_dist.setdefault(d, [])
        arr.append(float(s.vp_ref_km_s))
    by_dist = {d: np.asarray(vs, dtype=float) for d, vs in by_dist.items()}
    if not by_dist:
        return 0.0

    rng = rng or np.random.default_rng()
    sig = max(0.0, float(vp_pick_sigma_km_s))
    k_draw = max(1, int(mc_depth_draws_per_station))
    mc_vals = np.empty(max(int(n_mc), 2), dtype=float)

    for i in range(int(n_mc)):
        station_v: list[float] = []
        for d in sorted(by_dist.keys()):
            vs = by_dist[d]
            n = int(vs.size)
            if n < 1:
                continue
            idx = rng.integers(0, n, size=min(k_draw, n))
            v_draw = vs[idx].astype(float, copy=True)
            if sig > 0.0:
                v_draw = v_draw + rng.normal(0.0, sig, size=v_draw.size)
                v_draw = np.maximum(v_draw, 0.1)
            station_v.append(harmonic_mean_km_s(v_draw))
        if not station_v:
            mc_vals[i] = np.nan
            continue
        sm = np.asarray(station_v, dtype=float)
        if sm.size >= 2:
            sidx = rng.integers(0, sm.size, size=sm.size)
            mc_vals[i] = harmonic_mean_km_s(sm[sidx])
        else:
            mc_vals[i] = harmonic_mean_km_s(sm)

    mc_vals = mc_vals[np.isfinite(mc_vals)]
    if mc_vals.size < 2:
        return 0.0
    return float(np.std(mc_vals, ddof=1))


def mc_harmonic_sigma_stations(
    station_v_lc: np.ndarray,
    *,
    n_mc: int = 100,
    rng: np.random.Generator | None = None,
) -> float:
    """Legacy station-only bootstrap (kept for tests)."""
    v_st = np.asarray(station_v_lc, dtype=float)
    v_st = v_st[np.isfinite(v_st) & (v_st > 0.0)]
    n_st = int(v_st.size)
    if n_st < 2 or int(n_mc) < 2:
        return 0.0
    rng = rng or np.random.default_rng()
    mc_vals = np.empty(int(n_mc), dtype=float)
    for i in range(int(n_mc)):
        idx = rng.integers(0, n_st, size=n_st)
        mc_vals[i] = harmonic_mean_km_s(v_st[idx])
    return float(np.std(mc_vals, ddof=1))


def lower_crust_depth_mask(
    depth_km: np.ndarray,
    h_whole_km: np.ndarray,
    *,
    f_lower: float,
) -> np.ndarray:
    """
    Lower crust = bottom ``f_lower`` fraction of the igneous column.

    Depth z measured from surface; column spans [0, H].
    """
    f = float(np.clip(f_lower, 0.0, 1.0))
    z_top = (1.0 - f) * h_whole_km
    return (depth_km >= z_top) & (depth_km <= h_whole_km)


def load_transect_csv(path: Path | str) -> list[dict[str, float]]:
    """
    Load depth samples: distance_km, depth_km, vp_insitu_km_s, h_whole_km.

    Header required; extra columns ignored.
    """
    path = Path(path)
    rows: list[dict[str, float]] = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for rec in reader:
            if not rec.get("distance_km"):
                continue
            rows.append(
                {
                    "distance_km": float(rec["distance_km"]),
                    "depth_km": float(rec["depth_km"]),
                    "vp_insitu_km_s": float(rec["vp_insitu_km_s"]),
                    "h_whole_km": float(rec["h_whole_km"]),
                }
            )
    return rows


def prepare_samples(
    rows: list[dict[str, float]],
    *,
    f_lower: float = 0.7,
    gradient_c_per_km: float = 20.0,
    surface_t_c: float = 0.0,
) -> list[TransectSample]:
    out: list[TransectSample] = []
    for r in rows:
        depth = float(r["depth_km"])
        h = float(r["h_whole_km"])
        vp_ref = correct_vp_depth_to_reference_km_s(
            float(r["vp_insitu_km_s"]),
            depth,
            gradient_c_per_km=gradient_c_per_km,
            surface_t_c=surface_t_c,
        )
        if "in_lower_crust" in r:
            in_lc = bool(r["in_lower_crust"])
        elif "f_lower" in r:
            f_row = float(r["f_lower"])
            z_top = (1.0 - f_row) * h
            in_lc = z_top <= depth <= h
        else:
            z_top = (1.0 - f_lower) * h
            in_lc = z_top <= depth <= h
        f_row: float | None = float(r["f_lower"]) if "f_lower" in r else float(f_lower)
        out.append(
            TransectSample(
                distance_km=float(r["distance_km"]),
                depth_km=depth,
                vp_insitu_km_s=float(r["vp_insitu_km_s"]),
                h_whole_km=h,
                vp_ref_km_s=vp_ref,
                in_lower_crust=bool(in_lc),
                f_lower=f_row,
            )
        )
    return out


def aggregate_transect_windows(
    samples: list[TransectSample],
    *,
    window_half_width_km: float = 10.0,
    distance_step_km: float = 10.0,
    distance_range_km: tuple[float, float] | None = None,
    h_min_km: float = 15.0,
    f_lower: float = 0.7,
    delta_vp_max_km_s: float = 0.15,
    n_mc: int = 100,
    mc_depth_draws_per_station: int = DEFAULT_MC_DEPTH_DRAWS_PER_STATION,
    vp_pick_sigma_km_s: float = DEFAULT_VP_PICK_SIGMA_KM_S,
    rng: np.random.Generator | None = None,
) -> list[TransectWindow]:
    """
    Fig.15a/b style: 20 km window (2×half_width), step ``distance_step_km``, MC ensembles.

    Pipeline per window:
      1. Harmonic mean V_LC at each profile distance (merge depth samples).
      2. Window V_LC = harmonic mean of station values.
      3. MC σ = sparse depth resampling per station (``mc_depth_draws_per_station``)
         + optional pick noise (``vp_pick_sigma_km_s``) + station bootstrap when ≥2.

    Step-2 bounds (interpretation): V_bulk ∈ [V_LC − ΔVp_max, V_LC]; not applied as plot shift.
    """
    if not samples:
        return []

    rng = rng or np.random.default_rng(42)
    dists = np.array([s.distance_km for s in samples], dtype=float)
    d_lo = float(dists.min()) if distance_range_km is None else float(distance_range_km[0])
    d_hi = float(dists.max()) if distance_range_km is None else float(distance_range_km[1])
    centers = np.arange(d_lo, d_hi + 0.5 * distance_step_km, distance_step_km)

    windows: list[TransectWindow] = []
    for d0 in centers:
        in_win = [s for s in samples if abs(s.distance_km - d0) <= window_half_width_km]
        lc = [s for s in in_win if s.in_lower_crust]
        v_stations = lc_station_harmonic_means(lc)
        if v_stations.size < 2:
            continue

        h_mean = float(np.mean([s.h_whole_km for s in in_win]))
        f_vals = np.array(
            [float(s.f_lower) for s in in_win if s.f_lower is not None],
            dtype=float,
        )
        f_mean = float(np.mean(f_vals)) if f_vals.size else float(f_lower)
        h_lower_mean = float(
            np.mean([float(s.f_lower) * s.h_whole_km for s in in_win if s.f_lower is not None])
        ) if f_vals.size else float(f_mean * h_mean)

        v_lc = harmonic_mean_km_s(v_stations)
        v_sigma = mc_harmonic_sigma_lc_samples(
            lc,
            n_mc=n_mc,
            mc_depth_draws_per_station=mc_depth_draws_per_station,
            vp_pick_sigma_km_s=vp_pick_sigma_km_s,
            rng=rng,
        )

        thick = h_mean > float(h_min_km)
        v_upper = v_lc
        v_lower = v_lc - float(delta_vp_max_km_s) if thick else float("nan")

        windows.append(
            TransectWindow(
                distance_km=float(d0),
                h_whole_km=h_mean,
                h_lower_km=h_lower_mean,
                v_lc_km_s=v_lc,
                v_lc_sigma_km_s=v_sigma,
                n_samples=len(lc),
                n_stations=int(v_stations.size),
                thick_crust=thick,
                v_bulk_upper_km_s=v_upper,
                v_bulk_lower_km_s=v_lower,
                f_lower=f_mean,
            )
        )
    return windows


def export_transect_windows_csv(path: Path | str, windows: list[TransectWindow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "distance_km",
        "h_whole_km",
        "h_lower_km",
        "f_lower",
        "v_lc_km_s",
        "v_lc_sigma_km_s",
        "n_samples",
        "n_stations",
        "thick_crust",
        "v_bulk_upper_km_s",
        "v_bulk_lower_km_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for win in windows:
            w.writerow({k: getattr(win, k) for k in fields})


def load_transect_windows_csv(path: Path | str) -> list[TransectWindow]:
    path = Path(path)
    out: list[TransectWindow] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for rec in csv.DictReader(fh):
            if not rec.get("h_whole_km"):
                continue
            thick_raw = str(rec.get("thick_crust", "True")).strip().lower()
            thick = thick_raw in ("1", "true", "yes", "y")
            h_whole = float(rec["h_whole_km"])
            f_lo = rec.get("f_lower")
            f_lower = float(f_lo) if f_lo not in (None, "") else None
            h_lower_raw = rec.get("h_lower_km")
            if h_lower_raw not in (None, ""):
                h_lower = float(h_lower_raw)
            elif f_lower is not None:
                h_lower = float(f_lower) * h_whole
            else:
                h_lower = 0.7 * h_whole
            out.append(
                TransectWindow(
                    distance_km=float(rec["distance_km"]),
                    h_whole_km=h_whole,
                    h_lower_km=h_lower,
                    v_lc_km_s=float(rec["v_lc_km_s"]),
                    v_lc_sigma_km_s=float(rec.get("v_lc_sigma_km_s") or 0.0),
                    n_samples=int(float(rec.get("n_samples") or 0)),
                    n_stations=int(float(rec.get("n_stations") or rec.get("n_samples") or 0)),
                    thick_crust=thick,
                    v_bulk_upper_km_s=float(rec.get("v_bulk_upper_km_s") or rec["v_lc_km_s"]),
                    v_bulk_lower_km_s=float(rec.get("v_bulk_lower_km_s") or "nan"),
                    f_lower=f_lower,
                )
            )
    return out
