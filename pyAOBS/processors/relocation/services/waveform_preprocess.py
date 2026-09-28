# -*- coding: utf-8 -*-
"""姿态校正截窗预处理：rmean / rtrend / 可选带通（不含增益）。"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .models import AttitudeUiParams


def apply_rmean(trace: np.ndarray) -> np.ndarray:
    x = np.asarray(trace, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x
    return x - float(np.mean(x))


def apply_rtrend(trace: np.ndarray) -> np.ndarray:
    x = np.asarray(trace, dtype=np.float64).reshape(-1)
    n = int(x.size)
    if n < 2:
        return x
    t = np.arange(n, dtype=np.float64)
    t0 = t - t.mean()
    denom = float(np.dot(t0, t0))
    if denom <= 1e-30:
        return x - float(np.mean(x))
    slope = float(np.dot(t0, x - float(np.mean(x)))) / denom
    intercept = float(np.mean(x))
    return x - (slope * t0 + intercept)


def apply_bandpass(
    trace: np.ndarray,
    *,
    sampling_rate: float,
    freqlo: float,
    freqhi: float,
    npoles: int = 8,
    zerophase: bool = True,
) -> np.ndarray:
    """带通；优先复用 zplot DataProcessor，失败则 SciPy butter。"""
    x = np.asarray(trace, dtype=np.float64).reshape(-1)
    if x.size < 8 or sampling_rate <= 0:
        return x
    flo = float(freqlo)
    fhi = float(freqhi)
    if not (0.0 < flo < fhi):
        return x
    nyq = 0.5 * float(sampling_rate)
    if fhi >= nyq * 0.99:
        fhi = nyq * 0.99
    if flo >= fhi:
        return x
    try:
        from pyAOBS.visualization.zplotpy.core.data_processor import DataProcessor

        dp = DataProcessor(enable_cache=False)
        return np.asarray(
            dp.apply_bandpass_filter(
                x,
                flo,
                fhi,
                int(npoles),
                1 if zerophase else 0,
                float(sampling_rate),
            ),
            dtype=np.float64,
        ).reshape(-1)
    except Exception:
        pass
    try:
        from scipy.signal import butter, filtfilt, lfilter

        wn = [flo / nyq, fhi / nyq]
        b, a = butter(max(1, int(npoles) // 2), wn, btype="band")
        if zerophase and x.size > 3 * max(len(a), len(b)):
            return np.asarray(filtfilt(b, a, x), dtype=np.float64)
        return np.asarray(lfilter(b, a, x), dtype=np.float64)
    except Exception:
        return x


def preprocess_window(
    trace: np.ndarray,
    ui: AttitudeUiParams,
    *,
    sampling_rate: float,
) -> np.ndarray:
    """截窗后处理顺序：rmean → rtrend →（可选）带通。不加增益。"""
    y = np.asarray(trace, dtype=np.float64).reshape(-1)
    if bool(ui.use_rmean):
        y = apply_rmean(y)
    if bool(ui.use_rtrend):
        y = apply_rtrend(y)
    if bool(ui.use_bandpass):
        y = apply_bandpass(
            y,
            sampling_rate=float(sampling_rate),
            freqlo=float(ui.freqlo),
            freqhi=float(ui.freqhi),
            npoles=int(ui.npoles),
            zerophase=bool(ui.izerop),
        )
    return y


def preprocess_zrt(
    z: np.ndarray,
    r: np.ndarray,
    t: np.ndarray,
    ui: AttitudeUiParams,
    *,
    sampling_rate: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        preprocess_window(z, ui, sampling_rate=sampling_rate),
        preprocess_window(r, ui, sampling_rate=sampling_rate),
        preprocess_window(t, ui, sampling_rate=sampling_rate),
    )


def preprocess_summary(ui: Optional[AttitudeUiParams]) -> str:
    ui = ui or AttitudeUiParams()
    parts = []
    parts.append("rmean" if ui.use_rmean else "no-rmean")
    parts.append("rtrend" if ui.use_rtrend else "no-rtrend")
    if ui.use_bandpass:
        parts.append(f"bp={ui.freqlo:g}-{ui.freqhi:g}Hz")
    else:
        parts.append("no-bp")
    parts.append("no-gain")
    return ", ".join(parts)
