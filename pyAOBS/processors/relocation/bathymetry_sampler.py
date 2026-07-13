"""
Bathymetry/depth sampling helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional
import numpy as np


@dataclass
class BathymetrySampler:
    sample: Callable[[float, float], Optional[float]]
    source: str = "unknown"

    def __call__(self, x: float, y: float) -> Optional[float]:
        try:
            v = self.sample(float(x), float(y))
        except Exception:
            return None
        if v is None or not np.isfinite(float(v)):
            return None
        return float(v)


def _build_grid_sampler(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> BathymetrySampler:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if x.size < 2 or y.size < 2 or z.ndim != 2:
        return BathymetrySampler(sample=lambda _x, _y: None, source="grid-invalid")

    # Ensure monotonically increasing coordinates for interpolation.
    if x[0] > x[-1]:
        x = x[::-1]
        z = z[:, ::-1]
    if y[0] > y[-1]:
        y = y[::-1]
        z = z[::-1, :]

    def _interp(xq: float, yq: float) -> Optional[float]:
        if not (np.isfinite(xq) and np.isfinite(yq)):
            return None
        if xq < x[0] or xq > x[-1] or yq < y[0] or yq > y[-1]:
            return None
        ix = int(np.searchsorted(x, xq, side="right")) - 1
        iy = int(np.searchsorted(y, yq, side="right")) - 1
        ix = int(np.clip(ix, 0, x.size - 2))
        iy = int(np.clip(iy, 0, y.size - 2))
        x0, x1 = float(x[ix]), float(x[ix + 1])
        y0, y1 = float(y[iy]), float(y[iy + 1])
        if abs(x1 - x0) < 1e-12 or abs(y1 - y0) < 1e-12:
            return None
        tx = float((xq - x0) / (x1 - x0))
        ty = float((yq - y0) / (y1 - y0))
        z00 = float(z[iy, ix])
        z10 = float(z[iy, ix + 1])
        z01 = float(z[iy + 1, ix])
        z11 = float(z[iy + 1, ix + 1])
        if not np.isfinite([z00, z10, z01, z11]).all():
            return None
        return (
            (1.0 - tx) * (1.0 - ty) * z00
            + tx * (1.0 - ty) * z10
            + (1.0 - tx) * ty * z01
            + tx * ty * z11
        )

    return BathymetrySampler(sample=_interp, source="grid")


def _build_points_sampler(x: np.ndarray, y: np.ndarray, z: np.ndarray, max_dist: float | None = None) -> BathymetrySampler:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    z = np.asarray(z, dtype=float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[valid]
    y = y[valid]
    z = z[valid]
    if x.size == 0:
        return BathymetrySampler(sample=lambda _x, _y: None, source="points-empty")

    xy = np.column_stack([x, y])
    if max_dist is None:
        # Default to unlimited nearest-neighbor sampling to avoid false "no depth" failures.
        max_dist = float("inf")

    def _nearest(xq: float, yq: float) -> Optional[float]:
        if not (np.isfinite(xq) and np.isfinite(yq)):
            return None
        d2 = (xy[:, 0] - xq) ** 2 + (xy[:, 1] - yq) ** 2
        if d2.size == 0:
            return None
        i = int(np.argmin(d2))
        if np.isfinite(float(max_dist)) and float(np.sqrt(d2[i])) > float(max_dist):
            return None
        return float(z[i])

    return BathymetrySampler(sample=_nearest, source="points")


def build_bathymetry_sampler(terrain_meta: Optional[Dict[str, object]]) -> Optional[BathymetrySampler]:
    """Build a depth sampler from location-map terrain metadata."""
    if not terrain_meta:
        return None
    mode = str(terrain_meta.get("mode", "")).lower()
    try:
        if mode == "grid":
            return _build_grid_sampler(
                np.asarray(terrain_meta.get("x", []), dtype=float),
                np.asarray(terrain_meta.get("y", []), dtype=float),
                np.asarray(terrain_meta.get("z", []), dtype=float),
            )
        if mode == "points":
            return _build_points_sampler(
                np.asarray(terrain_meta.get("x", []), dtype=float),
                np.asarray(terrain_meta.get("y", []), dtype=float),
                np.asarray(terrain_meta.get("z", []), dtype=float),
            )
    except Exception:
        return None
    return None

