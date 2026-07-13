from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _hann(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / (n - 1))


def _frame_signal(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    n = x.shape[0]
    if n <= n_fft:
        pad = n_fft - n
        x_pad = np.pad(x, (0, pad), mode="constant")
        return x_pad[None, :]
    n_frames = 1 + (n - n_fft) // hop
    idx = (np.arange(n_frames)[:, None] * hop) + np.arange(n_fft)[None, :]
    return x[idx]


def stft(
    x: np.ndarray,
    *,
    n_fft: int,
    hop: int,
    window: np.ndarray,
) -> np.ndarray:
    frames = _frame_signal(x, n_fft=n_fft, hop=hop).astype(np.float64, copy=False)
    if window.shape[0] != n_fft:
        raise ValueError("window length must equal n_fft")
    frames = frames * window[None, :]
    return np.fft.rfft(frames, n=n_fft, axis=1).T  # (n_freq, n_frames)


def istft(
    X: np.ndarray,
    *,
    n_fft: int,
    hop: int,
    window: np.ndarray,
    length: int,
) -> np.ndarray:
    # Overlap-add with window normalization
    n_freq, n_frames = X.shape
    if n_freq != (n_fft // 2 + 1):
        raise ValueError("STFT freq bins mismatch with n_fft")

    frames = np.fft.irfft(X.T, n=n_fft, axis=1)  # (n_frames, n_fft)
    frames = frames * window[None, :]

    y = np.zeros((n_frames * hop + n_fft,), dtype=np.float64)
    wsum = np.zeros_like(y)

    for i in range(n_frames):
        start = i * hop
        y[start : start + n_fft] += frames[i]
        wsum[start : start + n_fft] += window**2

    # Avoid edge blow-ups where window energy is tiny (partial overlap at boundaries).
    # In the steady region (COLA), wsum is near-constant; at edges it decays to ~0.
    wmax = float(np.max(wsum)) if wsum.size else 0.0
    nz = wsum > max(1e-12, 1e-3 * wmax)
    y[nz] /= wsum[nz]
    y[~nz] = 0.0
    y = y[:length]
    return y


def _inst_freq_from_phase(X: np.ndarray, dt_frame: float) -> np.ndarray:
    """Estimate instantaneous frequency (Hz) per TF point from STFT phase evolution.

    Uses finite difference on unwrapped phase along time frames.
    Returns array (n_freq, n_frames) with NaNs where undefined.
    """
    phase = np.unwrap(np.angle(X), axis=1)
    dphi = np.diff(phase, axis=1)  # (n_freq, n_frames-1)
    omega = dphi / dt_frame  # rad/s
    f_inst = omega / (2.0 * np.pi)  # Hz
    out = np.full_like(X.real, np.nan, dtype=np.float64)
    out[:, 1:] = f_inst
    return out


@dataclass(frozen=True)
class WSSTResult:
    tf: np.ndarray  # synchrosqueezed TF (n_freq, n_frames)
    freq: np.ndarray  # Hz (n_freq,)
    stft: np.ndarray  # original STFT (n_freq, n_frames)
    reassigned_bins: np.ndarray  # int32 (n_freq, n_frames) target bin or -1
    n_fft: int
    hop: int
    window: np.ndarray
    length: int


def wsst_like_sst(
    x: np.ndarray,
    dt: float,
    *,
    f_s: float,
    f_e: float,
    n_fft: Optional[int] = None,
    hop: Optional[int] = None,
) -> WSSTResult:
    """A self-contained WSST-like transform using STFT synchrosqueezing.

    Notes:
    - This is a pragmatic, self-implemented backend for P1 (no external deps).
    - It provides a sharpened TF representation (synchrosqueezed STFT).
    - Inverse is performed via weighted original STFT + iSTFT (see reconstruct()).
    """
    x = np.asarray(x, dtype=np.float64)
    fs = 1.0 / float(dt)

    if n_fft is None:
        # heuristic: ~0.25 s window, clamp to power-of-two-ish
        target = max(256, min(4096, int(fs * 0.25)))
        n_fft = int(2 ** np.round(np.log2(target)))
    if hop is None:
        hop = max(1, n_fft // 4)

    window = _hann(n_fft)
    X = stft(x, n_fft=n_fft, hop=hop, window=window)  # (n_freq, n_frames)

    n_freq = X.shape[0]
    freq = np.fft.rfftfreq(n_fft, d=dt)
    dt_frame = hop * dt

    f_inst = _inst_freq_from_phase(X, dt_frame=dt_frame)  # Hz
    df = freq[1] - freq[0] if n_freq > 1 else fs / n_fft

    # reassignment bin index
    k_target = np.full((n_freq, X.shape[1]), -1, dtype=np.int32)
    valid = np.isfinite(f_inst)
    k = np.zeros_like(k_target, dtype=np.int32)
    k[valid] = np.rint(f_inst[valid] / df).astype(np.int32)
    valid &= (k >= 0) & (k < n_freq)

    # restrict to desired band
    band = (freq >= f_s) & (freq <= f_e)
    valid &= band[:, None]

    k_target[valid] = k[valid]

    # synchrosqueeze: accumulate coefficients into reassigned bins
    S = np.zeros_like(X)
    # vectorized accumulation: loop over frames to keep memory modest
    for t in range(X.shape[1]):
        kt = k_target[:, t]
        m = kt >= 0
        if not np.any(m):
            continue
        np.add.at(S[:, t], kt[m], X[m, t])

    return WSSTResult(
        tf=S,
        freq=freq,
        stft=X,
        reassigned_bins=k_target,
        n_fft=n_fft,
        hop=hop,
        window=window,
        length=x.shape[0],
    )


def reconstruct_from_weights(
    wsst: WSSTResult,
    tf_weights: np.ndarray,
    *,
    f_s: Optional[float] = None,
    f_e: Optional[float] = None,
) -> np.ndarray:
    """Reconstruct time-domain signal by mapping TF weights back to original STFT.

    `tf_weights` is defined on the synchrosqueezed TF grid (same shape as wsst.tf).
    We apply, for each original STFT coefficient X[k,t], the weight at its reassigned
    bin (k_target,t) if defined; otherwise use 0 (aggressive) or 1 (conservative).
    Here we choose 0 to match the idea that non-reassigned/invalid bins are likely noise.
    """
    W = np.asarray(tf_weights, dtype=np.float64)
    if W.shape != wsst.tf.shape:
        raise ValueError("tf_weights shape must match wsst.tf")

    X = wsst.stft
    k_target = wsst.reassigned_bins

    # For bins without valid reassignment, keep a small residual instead of hard-zero,
    # which helps preserve weak but coherent signal energy.
    unassigned_keep = 0.15
    Xw = X * float(unassigned_keep)
    for t in range(X.shape[1]):
        kt = k_target[:, t]
        m = kt >= 0
        if not np.any(m):
            Xw[:, t] = X[:, t]
            continue
        # weight each original bin by weight at its reassigned location
        Xw[m, t] = X[m, t] * W[kt[m], t]

    # Strict band limiting (optional): zero out bins outside [f_s, f_e]
    if f_s is not None or f_e is not None:
        f_s_val = float(f_s) if f_s is not None else -np.inf
        f_e_val = float(f_e) if f_e is not None else np.inf
        band = (wsst.freq >= f_s_val) & (wsst.freq <= f_e_val)
        Xw[~band, :] = 0.0

    y = istft(Xw, n_fft=wsst.n_fft, hop=wsst.hop, window=wsst.window, length=wsst.length)
    return y

