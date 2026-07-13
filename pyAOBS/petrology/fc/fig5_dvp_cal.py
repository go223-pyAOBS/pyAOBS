"""
Fig.5 digitized F-locus ΔVp calibration (KKHS02 Step-3).

Optional empirical correction to ``delta_vp_wl_fc`` so catalog FC tracks match
digitized Fig.5a/b F-loci @ 600 MPa / 400 °C.

Production / R3 defaults leave this **off** (raw W&L+Langmuir); enable only for
pixel-level Fig.5 acceptance (``check_fig5_digitized``).

Model (per P_fc = 100 / 400 MPa)::

    ΔVp_corr = c0 + c1 * F_solid

Coefficients are fit from mean(simulation − paper) across the melt catalog,
then negated (simulation is systematically low).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

FIG5_DVP_CAL_JSON = Path(__file__).resolve().parents[1] / "data" / "figure05_dvp_calibration.json"

_PANEL_BY_PFC: dict[float, str] = {100.0: "a_vlc_100", 400.0: "b_vlc_400"}


def load_fig5_dvp_calibration(path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path or FIG5_DVP_CAL_JSON)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _cal_cached() -> dict[str, Any]:
    return load_fig5_dvp_calibration()


def clear_fig5_dvp_calibration_cache() -> None:
    _cal_cached.cache_clear()


def _panel_key(p_fc_mpa: float) -> str | None:
    p = float(p_fc_mpa)
    if abs(p - 100.0) < 25.0:
        return _PANEL_BY_PFC[100.0]
    if abs(p - 400.0) < 150.0:
        return _PANEL_BY_PFC[400.0]
    return None


def correction_km_s(
    f_solid: float,
    p_fc_mpa: float,
    cal: dict[str, Any] | None = None,
) -> float:
    """Additive ΔVp correction (km/s) from Fig.5 digitized F-loci fit."""
    cal = cal if cal is not None else _cal_cached()
    tracks = cal.get("tracks") or {}
    if not tracks:
        return 0.0

    f = float(np.clip(f_solid, 0.05, 0.95))
    p = float(p_fc_mpa)

    def _eval_track(track: dict[str, Any]) -> float:
        coeff = track.get("coeff")
        if not coeff:
            return 0.0
        return float(np.polyval(np.asarray(coeff, dtype=float), f))

    if "100" in tracks and "400" in tracks:
        c100 = _eval_track(tracks["100"])
        c400 = _eval_track(tracks["400"])
        if p <= 100.0:
            return c100
        if p >= 400.0:
            return c400
        t = (p - 100.0) / 300.0
        return float((1.0 - t) * c100 + t * c400)

    key = _panel_key(p)
    if key and key in tracks:
        return _eval_track(tracks[key])
    return 0.0


def apply_fig5_dvp_calibration(
    delta_vp_km_s: float,
    f_solid: float,
    p_fc_mpa: float,
    *,
    enabled: bool = True,
    cal: dict[str, Any] | None = None,
) -> float:
    if not enabled:
        return float(delta_vp_km_s)
    return float(delta_vp_km_s) + correction_km_s(f_solid, p_fc_mpa, cal=cal)


def compute_calibration(
    *,
    mineral_backend: str = "sb1994_fig2ol",
    kd_engine: str = "langmuir",
    f_values: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8),
) -> dict[str, Any]:
    """
    Derive linear ΔVp correction vs F from catalog sim − digitized F-loci.

    Returns JSON-serializable calibration dict.
    """
    from petrology.data.load_catalog import load_melt_catalog, oxides_from_record
    from petrology.fc.figure05_digitized import (
        load_figure05_digitized,
        residual_to_f_loci,
    )
    from petrology.fractionation import delta_vp_km_s
    from petrology.norm_velocity import norm_velocity_from_record

    data = load_figure05_digitized()
    rows = [r for r in load_melt_catalog() if r.get("include_in_regression")]

    tracks: dict[str, Any] = {}
    for p_fc, pkey in ((100.0, "a_vlc_100"), (400.0, "b_vlc_400")):
        biases: list[float] = []
        f_used: list[float] = []
        for f_t in f_values:
            dd = []
            for rec in rows:
                bulk = float(
                    norm_velocity_from_record(rec, mineral_backend=mineral_backend)["vp_km_s"]
                )
                ox = {k: v for k, v in oxides_from_record(rec).items() if k != "P2O5"}
                d_sim = delta_vp_km_s(
                    f_t,
                    bulk_vp_km_s=bulk,
                    p_fc_mpa=p_fc,
                    engine="wl1990",
                    melt_oxides_wt=ox,
                    kd_engine=kd_engine,
                    mineral_backend=mineral_backend,
                    fig5_dvp_calibrate=False,
                )
                y_sim = bulk + d_sim
                bias, _ = residual_to_f_loci(d_sim, y_sim, pkey, f_t, data=data)  # type: ignore[arg-type]
                if not np.isnan(bias):
                    dd.append(bias)
            if dd:
                f_used.append(float(f_t))
                biases.append(float(np.mean(dd)))

        if len(f_used) < 2:
            continue

        f_arr = np.asarray(f_used, dtype=float)
        # correction = -bias (simulation low → add positive)
        corr_arr = -np.asarray(biases, dtype=float)
        coeff = np.polyfit(f_arr, corr_arr, 1).tolist()

        tracks[f"{int(p_fc)}"] = {
            "panel": pkey,
            "p_fc_MPa": p_fc,
            "coeff": [float(c) for c in coeff],
            "f_nodes": f_used,
            "bias_sim_minus_paper": biases,
            "correction_at_nodes": [float(c) for c in corr_arr],
        }

    return {
        "meta": {
            "source": "Fig.5 digitized F-loci (100Mpa.txt, 400Mpa.txt)",
            "model": "delta_vp += c0 + c1 * F_solid",
            "p_fc_tracks_MPa": [100, 400],
            "mineral_backend": mineral_backend,
            "kd_engine": kd_engine,
            "catalog_n": len(rows),
        },
        "tracks": tracks,
    }


def save_fig5_dvp_calibration(data: dict[str, Any], path: Path | str | None = None) -> Path:
    path = Path(path or FIG5_DVP_CAL_JSON)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    clear_fig5_dvp_calibration_cache()
    return path
