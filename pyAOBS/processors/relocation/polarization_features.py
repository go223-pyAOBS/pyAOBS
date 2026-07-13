"""
Polarization feature extraction for 3C waveform windows.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PolarizationFeatures:
    principal_vector: np.ndarray
    eigenvalues: np.ndarray
    linearity: float
    planarity: float
    rectilinearity: float
    dominant_energy_ratio: float


def _safe_cov(x: np.ndarray) -> np.ndarray:
    if x.shape[1] < 2:
        return np.eye(x.shape[0], dtype=float) * 1e-12
    c = np.cov(x)
    if not np.all(np.isfinite(c)):
        return np.eye(x.shape[0], dtype=float) * 1e-12
    return np.asarray(c, dtype=float)


def extract_polarization_features(z: np.ndarray, r: np.ndarray, t: np.ndarray) -> PolarizationFeatures:
    """Extract PCA-like polarization indicators for Z/R/T window."""
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    t = np.asarray(t, dtype=float)
    n = int(min(z.size, r.size, t.size))
    if n < 3:
        return PolarizationFeatures(
            principal_vector=np.array([0.0, 0.0, 1.0], dtype=float),
            eigenvalues=np.array([0.0, 0.0, 0.0], dtype=float),
            linearity=0.0,
            planarity=0.0,
            rectilinearity=0.0,
            dominant_energy_ratio=0.0,
        )

    z = z[:n] - float(np.mean(z[:n]))
    r = r[:n] - float(np.mean(r[:n]))
    t = t[:n] - float(np.mean(t[:n]))
    x = np.vstack([r, t, z])  # local RTZ order
    cov = _safe_cov(x)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.asarray(vals[order], dtype=float)
    vecs = np.asarray(vecs[:, order], dtype=float)
    v1 = vecs[:, 0] if vecs.shape[1] > 0 else np.array([1.0, 0.0, 0.0], dtype=float)

    l1 = float(max(vals[0], 0.0))
    l2 = float(max(vals[1], 0.0)) if vals.size > 1 else 0.0
    l3 = float(max(vals[2], 0.0)) if vals.size > 2 else 0.0
    eps = 1e-12
    trace = max(l1 + l2 + l3, eps)
    linearity = (l1 - l2) / max(l1, eps)
    planarity = (l2 - l3) / max(l1, eps)
    rectilinearity = 1.0 - (l2 + l3) / max(2.0 * l1, eps)
    dominant_energy_ratio = l1 / trace

    return PolarizationFeatures(
        principal_vector=np.asarray(v1, dtype=float),
        eigenvalues=np.array([l1, l2, l3], dtype=float),
        linearity=float(np.clip(linearity, 0.0, 1.0)),
        planarity=float(np.clip(planarity, 0.0, 1.0)),
        rectilinearity=float(np.clip(rectilinearity, 0.0, 1.0)),
        dominant_energy_ratio=float(np.clip(dominant_energy_ratio, 0.0, 1.0)),
    )

