"""
KKHS02 Figure 5 digitized VLC / VUC vs ΔVp envelopes.

Paper axes: **x = V_LC (or V_UC), y = ΔVp** @ 600 MPa, 400°C.

Primary use: calibrate / validate W&L FC ΔVp (Step-3 → Step-2) against paper Fig.5a/b.

Data: ``petrology/data/figure05_digitized.json`` (updated from GetData via
``validation/import_figure05_digitized.py``).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np

PanelKey = Literal["a_vlc_100", "b_vlc_400", "c_vuc_100", "d_vuc_400"]

FIGURE05_JSON = Path(__file__).resolve().parents[1] / "data" / "figure05_digitized.json"
FIGURE05_TXT = Path(__file__).resolve().parents[1] / "data" / "figure05_digitized.txt"

_NUMERIC_LINE = re.compile(r"^\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s")
_FXL_HEADER = re.compile(r"^Fxl\s*=\s*([\d.]+)\s*$", re.IGNORECASE)

DEFAULT_AXES_VLC = {
    "x": "v_lc_km_s",
    "y": "delta_vp_km_s",
    "x_lim": [6.8, 7.8],
    "y_lim": [0.05, 0.35],
}
DEFAULT_AXES_VUC = {
    "x": "v_uc_km_s",
    "y": "delta_vp_km_s",
    "x_lim": [6.3, 7.3],
    "y_lim": [0.05, 0.35],
}


def load_figure05_digitized(path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path or FIGURE05_JSON)
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _data_cached() -> dict[str, Any]:
    return load_figure05_digitized()


def list_panels(data: dict[str, Any] | None = None) -> list[PanelKey]:
    data = data or _data_cached()
    return list(data["panels"].keys())  # type: ignore[return-value]


def get_panel(key: PanelKey, data: dict[str, Any] | None = None) -> dict[str, Any]:
    data = data or _data_cached()
    return data["panels"][key]


def panel_p_fc_mpa(key: PanelKey, data: dict[str, Any] | None = None) -> float:
    return float(get_panel(key, data)["p_fc_MPa"])


def panel_phase(key: PanelKey, data: dict[str, Any] | None = None) -> str:
    return str(get_panel(key, data)["phase"])


def _as_vy_dvp_pair(a: float, b: float) -> tuple[float, float]:
    """Normalize stored pair to (V_y km/s, ΔVp km/s) regardless of legacy order."""
    if a > 3.0 and b < 2.0:
        return float(a), float(b)
    if b > 3.0 and a < 2.0:
        return float(b), float(a)
    return float(a), float(b)


def median_envelope(key: PanelKey, data: dict[str, Any] | None = None) -> list[tuple[float, float]]:
    """Median F-locus envelope as (V_y, ΔVp) pairs."""
    pts = get_panel(key, data).get("median_envelope") or []
    return [_as_vy_dvp_pair(p[0], p[1]) for p in pts]


def sigma_delta_vp_at_f(key: PanelKey, f_solid: float, data: dict[str, Any] | None = None) -> float:
    """Paper 1σ in ΔVp at F=0.5 or 0.8 (linear interp between tabulated keys)."""
    sig = get_panel(key, data).get("sigma_delta_vp_km_s") or {}
    if not sig:
        return float("nan")
    fs = sorted(float(k) for k in sig.keys())
    f = float(f_solid)
    if f <= fs[0]:
        return float(sig[str(fs[0]) if str(fs[0]) in sig else fs[0]])
    if f >= fs[-1]:
        k = str(fs[-1]) if str(fs[-1]) in sig else f"{fs[-1]:g}"
        return float(sig.get(k, sig[str(fs[-1])]))
    for i in range(len(fs) - 1):
        if fs[i] <= f <= fs[i + 1]:
            t = (f - fs[i]) / (fs[i + 1] - fs[i])
            k0 = str(fs[i]) if str(fs[i]) in sig else f"{fs[i]:g}"
            k1 = str(fs[i + 1]) if str(fs[i + 1]) in sig else f"{fs[i + 1]:g}"
            return float((1 - t) * sig[k0] + t * sig[k1])
    return float("nan")


def interp_median_dvp(key: PanelKey, v_y_km_s: float, data: dict[str, Any] | None = None) -> float:
    """Interpolate median envelope: given V_LC or V_UC, return expected ΔVp."""
    env = median_envelope(key, data)
    if len(env) < 2:
        return float("nan")
    vlcs = np.array([p[0] for p in env], dtype=float)
    dvps = np.array([p[1] for p in env], dtype=float)
    order = np.argsort(vlcs)
    return float(
        np.interp(float(v_y_km_s), vlcs[order], dvps[order], left=float(dvps[order][0]), right=float(dvps[order][-1]))
    )


def interp_median_vy(key: PanelKey, delta_vp_km_s: float, data: dict[str, Any] | None = None) -> float:
    """Interpolate median envelope: given ΔVp, return expected V_LC or V_UC."""
    env = median_envelope(key, data)
    if len(env) < 2:
        return float("nan")
    vlcs = np.array([p[0] for p in env], dtype=float)
    dvps = np.array([p[1] for p in env], dtype=float)
    order = np.argsort(dvps)
    return float(
        np.interp(
            float(delta_vp_km_s),
            dvps[order],
            vlcs[order],
            left=float(vlcs[order][0]),
            right=float(vlcs[order][-1]),
        )
    )


def f_loci_curve_raw(
    key: PanelKey,
    f_solid: float,
    data: dict[str, Any] | None = None,
) -> list[tuple[float, float]]:
    """Raw digitized F-locus as (V_y, ΔVp) in click order."""
    loc = get_panel(key, data).get("f_loci") or {}
    for k, block in loc.items():
        if abs(float(k) - float(f_solid)) < 0.05:
            raw = block.get("curve_raw") or block.get("points") or []
            return [_as_vy_dvp_pair(p[0], p[1]) for p in raw]
    return []


def f_loci_curve(
    key: PanelKey,
    f_solid: float,
    data: dict[str, Any] | None = None,
    *,
    use_fit: bool = True,
) -> list[tuple[float, float]]:
    """F-locus for plotting / interpolation: fitted (V_y, ΔVp) if available."""
    loc = get_panel(key, data).get("f_loci") or {}
    for k, block in loc.items():
        if abs(float(k) - float(f_solid)) < 0.05:
            if use_fit and block.get("curve"):
                return [_as_vy_dvp_pair(p[0], p[1]) for p in block["curve"]]
            raw = block.get("curve_raw") or block.get("points") or []
            return [_as_vy_dvp_pair(p[0], p[1]) for p in raw]
    return []


def f_loci_points(key: PanelKey, f_solid: float, data: dict[str, Any] | None = None) -> list[tuple[float, float]]:
    loc = get_panel(key, data).get("f_loci") or {}
    for k, block in loc.items():
        if abs(float(k) - float(f_solid)) < 0.05:
            pts = block.get("curve_raw") or block.get("points") or []
            return [_as_vy_dvp_pair(p[0], p[1]) for p in pts]
    return []


def residual_to_median(
    delta_vp_km_s: float,
    v_y_km_s: float,
    key: PanelKey,
    data: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """Returns (d_dvp, d_vy) = simulation − digitized median."""
    dvp_med = interp_median_dvp(key, v_y_km_s, data=data)
    if np.isnan(dvp_med):
        return float("nan"), float("nan")
    return float(delta_vp_km_s - dvp_med), float(v_y_km_s - interp_median_vy(key, delta_vp_km_s, data=data))


def residual_to_f_loci(
    delta_vp_km_s: float,
    v_y_km_s: float,
    key: PanelKey,
    f_solid: float,
    data: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """Simulation − fitted F-locus at the same V_y (primary: ΔVp residual)."""
    vlcs, dvps = _curve_for_interp(f_loci_curve(key, f_solid, data=data))
    if vlcs.size < 2:
        return float("nan"), float("nan")
    dvp_exp = float(np.interp(float(v_y_km_s), vlcs, dvps, left=np.nan, right=np.nan))
    if np.isnan(dvp_exp):
        return float("nan"), float("nan")
    return float(delta_vp_km_s - dvp_exp), float("nan")


def _curve_for_interp(points: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Sort by V_y and dedupe for ΔVp(V_y) interpolation."""
    if len(points) < 2:
        return np.array([]), np.array([])
    deduped: list[tuple[float, float]] = []
    for vlc, dvp in sorted(points, key=lambda t: t[0]):
        if deduped and abs(deduped[-1][0] - vlc) <= 1e-9:
            deduped[-1] = (float(vlc), float(dvp))
        else:
            deduped.append((float(vlc), float(dvp)))
    return (
        np.array([p[0] for p in deduped], dtype=float),
        np.array([p[1] for p in deduped], dtype=float),
    )


def fit_f_locus_curve(
    points: list[tuple[float, float]],
    *,
    degree: int = 2,
    outlier_sigma: float = 2.5,
    n_grid: int = 80,
    max_iterations: int = 3,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], dict[str, Any]]:
    """
    Robust polynomial fit ΔVp(V_LC).

    Returns (fitted_curve, inliers, meta) as (V_y, ΔVp) pairs.
    """
    if len(points) < 3:
        return list(points), list(points), {"degree": 0, "n_inliers": len(points), "n_total": len(points)}

    vlc = np.array([p[0] for p in points], dtype=float)
    dvp = np.array([p[1] for p in points], dtype=float)
    deg = min(degree, len(points) - 1)
    mask = np.ones(len(vlc), dtype=bool)

    for _ in range(max_iterations):
        v, d = vlc[mask], dvp[mask]
        if len(v) < deg + 1:
            break
        coeff = np.polyfit(v, d, deg)
        pred = np.polyval(coeff, vlc)
        resid = dvp - pred
        r_in = resid[mask]
        med = float(np.median(r_in))
        mad = float(np.median(np.abs(r_in - med)))
        scale = 1.4826 * mad if mad > 1e-12 else float(np.std(r_in) or 1e-6)
        new_mask = np.abs(resid) <= outlier_sigma * scale
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    v_in, d_in = vlc[mask], dvp[mask]
    coeff = np.polyfit(v_in, d_in, min(deg, len(v_in) - 1))
    v_lo, v_hi = float(v_in.min()), float(v_in.max())
    v_grid = np.linspace(v_lo, v_hi, n_grid)
    d_fit = np.polyval(coeff, v_grid)

    fitted = [(float(v), float(d)) for v, d in zip(v_grid, d_fit)]
    inliers = [(float(v), float(d)) for v, d in zip(v_in, d_in)]
    meta: dict[str, Any] = {
        "degree": int(min(deg, len(v_in) - 1)),
        "n_inliers": int(mask.sum()),
        "n_total": len(points),
        "coeff": [float(c) for c in coeff],
    }
    return fitted, inliers, meta


def parse_getdata_fxl_txt(path: Path | str) -> dict[float, list[tuple[float, float]]]:
    """
    Parse native GetData export with ``Fxl=0.5`` section headers.

    File columns are (V_y, ΔVp); stored as (V_y, ΔVp) in click order.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    curves: dict[float, list[tuple[float, float]]] = {}
    f_cur: float | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = _FXL_HEADER.match(line)
        if m:
            f_cur = float(m.group(1))
            curves.setdefault(f_cur, [])
            continue
        if f_cur is None or line.startswith("Generated by"):
            continue
        if not _NUMERIC_LINE.match(line):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        vlc, dvp = float(parts[0]), float(parts[1])
        curves[f_cur].append((vlc, dvp))
    return curves


def build_median_envelope_from_f_loci(
    fitted_curves: dict[float, list[tuple[float, float]]],
    *,
    n_grid: int = 48,
) -> list[list[float]]:
    """Median ΔVp across fitted F-loci on a common V_y grid."""
    if not fitted_curves:
        return []
    by_f: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    vy_lo, vy_hi = float("inf"), float("-inf")
    for f, pts in fitted_curves.items():
        vlcs, dvps = _curve_for_interp(pts)
        if vlcs.size < 2:
            continue
        by_f[f] = (vlcs, dvps)
        vy_lo = min(vy_lo, float(vlcs[0]))
        vy_hi = max(vy_hi, float(vlcs[-1]))
    if not by_f:
        return []
    vy_grid = np.linspace(vy_lo, vy_hi, n_grid)
    stack = []
    for vlcs, dvps in by_f.values():
        stack.append(np.interp(vy_grid, vlcs, dvps, left=np.nan, right=np.nan))
    dvp_med = np.nanmedian(np.vstack(stack), axis=0)
    ok = ~np.isnan(dvp_med)
    return [[float(vy_grid[i]), float(dvp_med[i])] for i in range(len(vy_grid)) if ok[i]]


def merge_fxl_file_into_panel(
    panel: dict[str, Any],
    curves: dict[float, list[tuple[float, float]]],
    *,
    rebuild_median: bool = False,
    fit_degree: int = 2,
) -> None:
    """Write raw + fitted F-locus curves into panel JSON."""
    loc = panel.setdefault("f_loci", {})
    fitted_map: dict[float, list[tuple[float, float]]] = {}
    for f, pts in sorted(curves.items()):
        fk = f"{float(f):g}"
        block = loc.setdefault(fk, {"symbol": "open" if f <= 0.55 else "solid"})
        block["curve_raw"] = [[v, d] for v, d in pts]
        fitted, inliers, meta = fit_f_locus_curve(pts, degree=fit_degree)
        block["curve"] = [[v, d] for v, d in fitted]
        block["fit"] = meta
        fitted_map[f] = fitted
    if rebuild_median:
        med = build_median_envelope_from_f_loci(fitted_map)
        if med:
            panel["median_envelope"] = med


def merge_getdata_fxl_files(
    files: dict[PanelKey, Path | str],
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge ``100Mpa.txt`` / ``400Mpa.txt`` style exports into Fig.5 JSON."""
    data = json.loads(json.dumps(data or _data_cached()))
    panels = data["panels"]
    for pkey, path in files.items():
        if pkey not in panels:
            continue
        curves = parse_getdata_fxl_txt(path)
        if curves:
            phase = panels[pkey].get("phase", "vlc")
            panels[pkey]["axes"] = dict(DEFAULT_AXES_VUC if phase == "vuc" else DEFAULT_AXES_VLC)
            merge_fxl_file_into_panel(panels[pkey], curves)
    data["meta"]["provisional"] = False
    data["meta"]["notes"] = (
        "Panels a/b: F-loci from GetData (100Mpa.txt, 400Mpa.txt); "
        "curve=poly2 robust fit, curve_raw=raw clicks."
    )
    return data


def parse_figure05_txt(path: Path | str | None = None) -> dict[str, list[tuple[float, float]]]:
    """Parse legacy section-header export (see figure05_digitized.template.txt)."""
    path = Path(path or FIGURE05_TXT)
    if not path.is_file():
        return {}
    raw: dict[str, list[tuple[float, float]]] = {}
    section: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not _NUMERIC_LINE.match(line):
            section = line
            raw.setdefault(section, [])
            continue
        if section is None:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        raw[section].append(_as_vy_dvp_pair(float(parts[0]), float(parts[1])))
    return raw


def merge_txt_into_json(
    txt_sections: dict[str, list[tuple[float, float]]],
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge parsed txt sections into panel JSON structure."""
    data = json.loads(json.dumps(data or _data_cached()))
    panels = data["panels"]

    for section, pts in txt_sections.items():
        if not pts:
            continue
        if section.endswith("_median"):
            pkey = section[: -len("_median")]
            if pkey in panels:
                panels[pkey]["median_envelope"] = [[v, d] for v, d in pts]
        elif section.endswith("_f0.5"):
            pkey = section[: -len("_f0.5")]
            if pkey in panels:
                panels[pkey].setdefault("f_loci", {}).setdefault("0.5", {})
                panels[pkey]["f_loci"]["0.5"]["curve_raw"] = [[v, d] for v, d in pts]
        elif section.endswith("_f0.8"):
            pkey = section[: -len("_f0.8")]
            if pkey in panels:
                panels[pkey].setdefault("f_loci", {}).setdefault("0.8", {})
                panels[pkey]["f_loci"]["0.8"]["curve_raw"] = [[v, d] for v, d in pts]
        elif "_path_" in section:
            pkey, _, _rest = section.partition("_path_")
            if pkey in panels:
                panels[pkey].setdefault("sample_paths", []).append([[v, d] for v, d in pts])

    data["meta"]["provisional"] = False
    return data


def save_figure05_digitized(data: dict[str, Any], path: Path | str | None = None) -> Path:
    path = Path(path or FIGURE05_JSON)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _data_cached.cache_clear()
    return path
