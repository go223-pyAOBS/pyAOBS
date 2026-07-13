from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# 高斯 g(t)=exp(-½(t/σ)²) 的半高全宽 FWHM = 2*sqrt(2*ln(2))*σ
_GAUSS_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


def sigma_from_wavelet_length_fwhm_sec(wavelet_length_sec: float) -> float:
    """由高斯子波长度（FWHM，秒）换算为内部 σ（秒）。"""
    fwhm = float(wavelet_length_sec)
    if (not np.isfinite(fwhm)) or fwhm <= 0.0:
        raise ValueError(f"子波长度（FWHM）须为有限正数秒，实际为 {wavelet_length_sec!r}")
    return fwhm / float(_GAUSS_FWHM_FACTOR)


def wavelet_length_fwhm_sec_from_pick_kwargs(kwargs: object) -> float:
    """解析 kwargs：显式 ``pick_wavelet_length_sec``（FWHM 秒），否则回退 ``pick_half_width_sec``（旧版：σ）。"""
    if isinstance(kwargs, dict):
        raw = kwargs
    else:
        raw = {}
    wl = raw.get("pick_wavelet_length_sec", None)
    if wl is not None and np.isfinite(float(wl)) and float(wl) > 0.0:
        return float(wl)
    sig_legacy = raw.get("pick_half_width_sec", None)
    if sig_legacy is not None and np.isfinite(float(sig_legacy)) and float(sig_legacy) > 0.0:
        return float(sig_legacy) * float(_GAUSS_FWHM_FACTOR)
    return float(0.08) * float(_GAUSS_FWHM_FACTOR)


def pick_time_soft_weights(
    n_samples: int,
    dt: float,
    pick_times: Sequence[float],
    *,
    times_axis: Optional[np.ndarray] = None,
    t0_fallback: float = 0.0,
    wavelet_length_fwhm_sec: float = 0.19,
    floor: float = 0.12,
) -> np.ndarray:
    """Soft multiplicative weights along time (1 near picks, approaches `floor` away).

    ``wavelet_length_fwhm_sec``：时间门高斯包络的 **半高全宽 T**（秒），
    与经典「子波有效长度」同量级；内部用 σ = T / (2√(2ln2))。

    Broadcast to shape (n_freq, n_samples) as ``w[None, :]`` in the TF domain.
    """
    if n_samples <= 0:
        return np.asarray([], dtype=np.float64)
    fl = float(np.clip(floor, 0.0, 0.9999))
    dt_p = float(dt)
    if (not np.isfinite(dt_p)) or dt_p <= 0.0:
        dt_p = 1.0
    fwhm_in = float(wavelet_length_fwhm_sec)
    if (not np.isfinite(fwhm_in)) or fwhm_in <= 0.0:
        fwhm_in = max(float(dt_p) * 2.0, 0.02)
    sig = float(fwhm_in) / float(_GAUSS_FWHM_FACTOR)
    sig = max(sig, float(dt_p) * 0.25)

    t_axis: np.ndarray
    if times_axis is not None:
        ta = np.asarray(times_axis, dtype=np.float64).reshape(-1)
        if ta.size >= n_samples:
            t_axis = np.asarray(ta[:n_samples], dtype=np.float64)
        else:
            t_axis = float(t0_fallback) + np.arange(n_samples, dtype=np.float64) * dt_p
    else:
        t_axis = float(t0_fallback) + np.arange(n_samples, dtype=np.float64) * dt_p

    picks: list[float] = []
    for tc in pick_times:
        try:
            v = float(tc)
        except Exception:
            continue
        if np.isfinite(v):
            picks.append(v)
    if not picks:
        return np.ones(n_samples, dtype=np.float64)

    bell = np.zeros(n_samples, dtype=np.float64)
    inv_sig2 = 1.0 / (sig * sig)
    for tc in picks:
        d = t_axis - float(tc)
        g = np.exp(-0.5 * (d * d) * inv_sig2)
        bell = np.maximum(bell, g)
    return fl + (1.0 - fl) * bell
