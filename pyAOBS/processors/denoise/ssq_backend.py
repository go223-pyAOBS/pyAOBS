from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

try:
    # ssqueezepy is optional; if unavailable we fall back to builtin backend.
    from ssqueezepy import ssq_cwt, issq_cwt  # type: ignore

    HAVE_SSQ = True
except Exception:  # pragma: no cover - optional dependency
    ssq_cwt = None  # type: ignore
    issq_cwt = None  # type: ignore
    HAVE_SSQ = False


@dataclass(frozen=True)
class SSQResult:
    tf: np.ndarray        # synchrosqueezed TF (Tx), shape (n_freq, n_samples)
    freq: np.ndarray      # frequencies (Hz), shape (n_freq,)
    meta: Dict[str, Any]  # extra info for inverse (wavelet, etc.)


def ssq_forward(
    x: np.ndarray,
    dt: float,
    *,
    f_s: float,
    f_e: float,
    wavelet: str = "gmw",
) -> SSQResult:
    """Forward WSST backend via ssqueezepy.ssq_cwt.

    Returns synchrosqueezed CWT (Tx) and associated frequencies (Hz),
    already裁剪到 [f_s, f_e] 频带。
    """
    if not HAVE_SSQ:
        raise RuntimeError("ssqueezepy 未安装，无法使用 ssq_cwt 后端")

    x = np.asarray(x, dtype=np.float64)
    fs = 1.0 / float(dt)

    # Tx: (n_freq, n_samples), ssq_freqs: (n_freq,)
    Tx, Wx, ssq_freqs, scales = ssq_cwt(x, wavelet=wavelet, fs=fs)  # type: ignore

    freq = np.asarray(ssq_freqs, dtype=np.float64)
    band = (freq >= f_s) & (freq <= f_e)
    if not np.any(band):
        # 若频带选择过窄，退回全频带
        band = np.ones_like(freq, dtype=bool)

    Tx_band = Tx[band, :]
    freq_band = freq[band]

    meta: Dict[str, Any] = {
        "wavelet": wavelet,
        # issq_cwt 只需要 Tx + wavelet，其他留作将来扩展
    }
    return SSQResult(tf=Tx_band, freq=freq_band, meta=meta)


def ssq_inverse(
    tf_weighted: np.ndarray,
    meta: Dict[str, Any],
) -> np.ndarray:
    """Inverse WSST via ssqueezepy.issq_cwt."""
    if not HAVE_SSQ:
        raise RuntimeError("ssqueezepy 未安装，无法使用 issq_cwt 后端")

    wavelet = meta.get("wavelet", "gmw")
    y = issq_cwt(tf_weighted, wavelet=wavelet)  # type: ignore
    return np.asarray(y, dtype=np.float64)

