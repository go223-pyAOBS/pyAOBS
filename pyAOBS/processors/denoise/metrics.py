"""Quantitative A/B metrics for denoise evaluation (input A vs output B)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class DenoiseABMetrics:
    """Per-trace metrics; A = pre-denoise, B = post-denoise (same length segment)."""

    snr_db: float
    """10*log10( mean(A^2) / MSE(A,B) ); large when residual is small vs input power."""

    rmse: float
    """sqrt(mean((A-B)^2))."""

    cc: float
    """Pearson correlation between A and B in [-1, 1]."""

    mae: float
    """mean(|A-B|)."""


def ab_trace_metrics(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-30) -> DenoiseABMetrics:
    """Compute SNR (dB), RMSE, Pearson CC, MAE for aligned samples of A and B."""
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    n = int(min(aa.size, bb.size))
    if n < 2:
        nan = float("nan")
        return DenoiseABMetrics(snr_db=nan, rmse=nan, cc=nan, mae=nan)
    aa = aa[:n]
    bb = bb[:n]
    diff = aa - bb
    mse = float(np.mean(diff * diff))
    rmse = float(np.sqrt(mse + eps))
    mae = float(np.mean(np.abs(diff)))
    pa = float(np.mean(aa * aa) + eps)
    ratio = pa / (mse + eps)
    if (not np.isfinite(ratio)) or ratio <= 0.0:
        snr_db = float("nan")
    else:
        snr_db = float(10.0 * np.log10(ratio))
        if np.isfinite(snr_db) and snr_db > 120.0:
            snr_db = 120.0
    ac = aa - np.mean(aa)
    bc = bb - np.mean(bb)
    da = float(np.sqrt(np.sum(ac * ac)) + eps)
    db = float(np.sqrt(np.sum(bc * bc)) + eps)
    cc = float(np.sum(ac * bc) / (da * db))
    if not np.isfinite(cc):
        cc = float("nan")
    else:
        cc = float(np.clip(cc, -1.0, 1.0))
    return DenoiseABMetrics(snr_db=snr_db, rmse=rmse, cc=cc, mae=mae)


def summarize_metrics(rows: Sequence[DenoiseABMetrics]) -> Tuple[float, float, float, float, float, float, float, float]:
    """Return (snr_mean, snr_median, rmse_mean, rmse_median, cc_mean, cc_median, mae_mean, mae_median); NaNs ignored."""
    if not rows:
        nan = float("nan")
        return (nan,) * 8

    def _stat(vals: np.ndarray) -> Tuple[float, float]:
        v = vals[np.isfinite(vals)]
        if v.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(v)), float(np.median(v))

    snr = np.asarray([r.snr_db for r in rows], dtype=np.float64)
    rmse = np.asarray([r.rmse for r in rows], dtype=np.float64)
    cc = np.asarray([r.cc for r in rows], dtype=np.float64)
    mae = np.asarray([r.mae for r in rows], dtype=np.float64)
    sm, sd = _stat(snr)
    rmm, rmd = _stat(rmse)
    cm, cd = _stat(cc)
    mm, md = _stat(mae)
    return sm, sd, rmm, rmd, cm, cd, mm, md
