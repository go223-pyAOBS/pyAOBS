from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class GCVResult:
    weighted_tf: np.ndarray  # complex
    weights: np.ndarray  # float in [0,1]
    lambdas: np.ndarray  # per-frame lambda


def _soft_shrink_mag(mag: np.ndarray, lam: float) -> np.ndarray:
    return np.maximum(mag - lam, 0.0)


def _gcv_score(y: np.ndarray, yhat: np.ndarray, df: int) -> float:
    n = y.size
    denom = (n - df) ** 2
    if denom <= 0:
        return np.inf
    rss = float(np.sum((y - yhat) ** 2))
    return rss / denom


def gcv_shrink_per_frame(
    tf: np.ndarray,
    *,
    n_candidates: int = 24,
    min_keep: int = 1,
    strength: float = 1.0,
    floor_quantile: float = 0.2,
    floor_scale: float = 0.2,
) -> GCVResult:
    """Apply a simple GCV-selected soft-threshold per time frame (column-wise).

    This is a pragmatic P1 implementation:
    - For each frame t, treat |tf[:,t]| as observations.
    - Choose lambda by minimizing a GCV-like score over candidate lambdas.
    - Produce weights w in [0,1] so that tf_weighted = w * tf.

    Notes:
    - This is not a strict per-(f,t) GCV; it is adaptive per frame, which is
      much faster and still time-adaptive.
    """
    Z = np.asarray(tf)
    if Z.ndim != 2:
        raise ValueError("tf must be 2D (n_freq, n_frames)")

    n_freq, n_frames = Z.shape
    mag = np.abs(Z)
    eps = 1e-12

    weights = np.zeros((n_freq, n_frames), dtype=np.float64)
    lambdas = np.zeros((n_frames,), dtype=np.float64)

    # Candidate lambdas: per frame, between low and high quantiles of magnitude
    # Include lower quantiles so lambda>0 is often competitive.
    qs = np.linspace(0.20, 0.95, num=n_candidates)

    for t in range(n_frames):
        y = mag[:, t]
        if not np.any(y > 0):
            lambdas[t] = 0.0
            weights[:, t] = 0.0
            continue

        cand = np.quantile(y, qs)
        cand = np.unique(cand)
        if cand.size == 0:
            cand = np.array([0.0], dtype=np.float64)
        # Always allow lambda=0 (no shrink) as a candidate.
        if cand[0] != 0.0:
            cand = np.concatenate(([0.0], cand))

        # Fallback: no thresholding (keep signal)
        best_score = np.inf
        best_lam = 0.0
        best_yhat = y.copy()

        for lam in cand:
            lam = float(lam)
            yhat = _soft_shrink_mag(y, lam)
            df = int(np.sum(y > lam))
            if df < min_keep:
                continue
            score = _gcv_score(y, yhat, df=df)
            if score < best_score:
                best_score = score
                best_lam = lam
                best_yhat = yhat

        if not np.isfinite(strength) or strength <= 0:
            raise ValueError(f"`strength` 必须为正数，实际为 {strength!r}")

        # Make `strength` a robust knob by enforcing a data-driven floor on lambda.
        # This avoids the common case where GCV chooses lambda≈0 (no shrink).
        q = float(np.clip(floor_quantile, 0.0, 1.0))
        y_pos = y[y > 0]
        if y_pos.size == 0:
            lam_floor = 0.0
        else:
            lam_floor = float(np.quantile(y_pos, q)) * float(max(0.0, floor_scale))
        lam_base = max(float(best_lam), lam_floor)
        # Keep strength effective but less aggressive than linear scaling.
        lam_eff = lam_base * float(np.sqrt(strength))
        # Cap lambda to avoid over-suppressing coherent weak signals.
        if y_pos.size > 0:
            lam_cap = float(np.quantile(y_pos, 0.98)) * 0.9
            lam_eff = min(lam_eff, lam_cap)

        lambdas[t] = lam_eff
        yhat_eff = _soft_shrink_mag(y, lam_eff)
        w = yhat_eff / (y + eps)
        w = np.clip(w, 0.0, 1.0)
        weights[:, t] = w

    weighted_tf = Z * weights
    return GCVResult(weighted_tf=weighted_tf, weights=weights, lambdas=lambdas)

