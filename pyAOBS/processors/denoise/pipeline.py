from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .types import DenoiseDebug, DenoiseResult
from .wsst_backend import reconstruct_from_weights, wsst_like_sst
from .gcv_threshold import gcv_shrink_per_frame
from .morphology_mask import build_mask_from_tf
from . import ssq_backend
from .pick_guidance import pick_time_soft_weights, wavelet_length_fwhm_sec_from_pick_kwargs


def _pick_time_gate_vector(*, n_samples: int, dt: float, kwargs: Dict[str, Any]) -> Optional[np.ndarray]:
    """Returns shape (n_samples,) soft weights, or None when pick guidance is inactive."""
    if not bool(kwargs.get("pick_guidance_enable", False)):
        return None
    pts = kwargs.get("pick_times", None)
    if not pts:
        return None
    fwhm = float(wavelet_length_fwhm_sec_from_pick_kwargs(kwargs))
    fl = float(kwargs.get("pick_guidance_floor", 0.12))
    ta = kwargs.get("pick_times_axis", None)
    t0_fb = float(kwargs.get("pick_t0_fallback", 0.0))
    w = pick_time_soft_weights(
        int(n_samples),
        float(dt),
        list(pts),
        times_axis=ta,
        t0_fallback=t0_fb,
        wavelet_length_fwhm_sec=fwhm,
        floor=fl,
    )
    if w.size != int(n_samples):
        return None
    return w


def _gate_vector_to_tf_cols(
    w1: np.ndarray,
    *,
    n_cols: int,
) -> np.ndarray:
    """Map sample-axis gate (len=n_s) to TF columns (may be STFT frames != n_s)."""
    if w1.ndim != 1:
        raise ValueError("gate must be 1D")
    n_s = int(w1.size)
    out_n = int(max(1, n_cols))
    if out_n == n_s:
        return np.asarray(w1, dtype=np.float64)
    xi = np.linspace(0.0, float(max(1, n_s - 1)), num=out_n, dtype=np.float64)
    return np.interp(xi, np.arange(n_s, dtype=np.float64), np.asarray(w1, dtype=np.float64))


def _validate_common(
    dt: float,
    f_s: float,
    f_e: float,
    bwconn: int,
) -> None:
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"`dt` 必须为正数秒，实际为 {dt!r}")

    if not np.isfinite(f_s) or not np.isfinite(f_e):
        raise ValueError(f"`f_s`/`f_e` 必须为有限实数，实际为 f_s={f_s!r}, f_e={f_e!r}")

    if f_s < 0 or f_e <= 0:
        raise ValueError(f"`f_s`/`f_e` 必须满足 f_s>=0 且 f_e>0，实际为 f_s={f_s!r}, f_e={f_e!r}")

    if f_s >= f_e:
        raise ValueError(f"`f_s` 必须小于 `f_e`，实际为 f_s={f_s!r}, f_e={f_e!r}")

    if bwconn not in (4, 8):
        raise ValueError(f"`bwconn` 只支持 4 或 8 邻域，实际为 {bwconn!r}")


def denoise_trace(
    trace: Union[np.ndarray, list],
    dt: float,
    *,
    f_s: float,
    f_e: float,
    bwconn: int = 8,
    strength: float = 1.0,
    return_debug: bool = False,
    **kwargs: Any,
) -> Union[np.ndarray, DenoiseResult]:
    """Single-trace denoise entrypoint.

    P0 behavior:
    - Validate parameters
    - Return the input trace unchanged (pass-through)
    - Optionally wrap in DenoiseResult for forward compatibility

    P1+ will replace pass-through with WSST+GCV(+morphology) pipeline.
    """
    _validate_common(dt=dt, f_s=f_s, f_e=f_e, bwconn=bwconn)

    x = np.asarray(trace)
    if x.ndim != 1:
        raise ValueError(f"`trace` 必须是一维数组 (1D)，实际维度为 {x.ndim}")
    if x.size == 0:
        raise ValueError("`trace` 不能为空数组")
    if not np.all(np.isfinite(x)):
        raise ValueError("`trace` 包含 NaN/Inf，去噪前请先处理无效值")

    if not np.isfinite(strength) or strength <= 0:
        raise ValueError(f"`strength` 必须为正数，实际为 {strength!r}")

    morph_enable = bool(kwargs.get("morph_enable", True))
    morph_min_area = int(max(1, kwargs.get("morph_min_area", 24)))
    morph_quantile = float(kwargs.get("morph_quantile", 0.70))
    morph_floor_ratio = float(kwargs.get("morph_floor_ratio", 0.03))
    morph_expand = int(max(0, kwargs.get("morph_expand", 1)))
    morph_keep_strong_q = float(kwargs.get("morph_keep_strong_quantile", 0.95))
    morph_conn = int(kwargs.get("morph_bwconn", bwconn))

    # P1: 优先使用 ssqueezepy.ssq_cwt 后端；若不可用则回退到自实现 WSST-like。
    use_ssq = getattr(ssq_backend, "HAVE_SSQ", False)
    n_s = int(x.size)
    w_gate = _pick_time_gate_vector(n_samples=n_s, dt=float(dt), kwargs=kwargs)

    if use_ssq:
        ssq_res = ssq_backend.ssq_forward(x, dt, f_s=f_s, f_e=f_e)
        gcv = gcv_shrink_per_frame(ssq_res.tf, strength=float(strength))
        tf_weighted = ssq_res.tf * gcv.weights
        if w_gate is not None:
            tf_weighted = tf_weighted * w_gate[np.newaxis, :]
        if morph_enable:
            mask = build_mask_from_tf(
                tf_weighted,
                ssq_res.freq,
                bwconn=morph_conn,
                amp_quantile=morph_quantile,
                floor_ratio=morph_floor_ratio,
                min_area=morph_min_area,
                expand_steps=morph_expand,
                keep_strong_quantile=morph_keep_strong_q,
            )
            tf_final = tf_weighted * mask
            morph_keep_ratio = float(np.mean(mask))
            stage = "P1-ssq_cwt+gcv+morph"
        else:
            tf_final = tf_weighted
            morph_keep_ratio = 1.0
            stage = "P1-ssq_cwt+gcv"
        y = ssq_backend.ssq_inverse(tf_final, ssq_res.meta)
        tf_org = ssq_res.tf
        tf_gcv = tf_weighted
        freq = ssq_res.freq
    else:
        wsst = wsst_like_sst(x, dt, f_s=f_s, f_e=f_e)
        gcv = gcv_shrink_per_frame(wsst.tf, strength=float(strength))
        n_fr_tf = int(wsst.tf.shape[1])
        if w_gate is not None:
            w_tf = _gate_vector_to_tf_cols(w_gate, n_cols=n_fr_tf)
        else:
            w_tf = None
        if morph_enable:
            tf_weighted = gcv.weighted_tf
            if w_tf is not None:
                tf_weighted = tf_weighted * w_tf[np.newaxis, :]
            mask = build_mask_from_tf(
                tf_weighted,
                wsst.freq,
                bwconn=morph_conn,
                amp_quantile=morph_quantile,
                floor_ratio=morph_floor_ratio,
                min_area=morph_min_area,
                expand_steps=morph_expand,
                keep_strong_quantile=morph_keep_strong_q,
            )
            if w_tf is None:
                final_weights = gcv.weights * mask
            else:
                final_weights = gcv.weights * w_tf[np.newaxis, :] * mask
            tf_final = tf_weighted * mask
            morph_keep_ratio = float(np.mean(mask))
            stage = "P1-wsst_like_sst+gcv+morph-fallback"
        else:
            if w_tf is None:
                final_weights = gcv.weights
                tf_final = gcv.weighted_tf
            else:
                final_weights = gcv.weights * w_tf[np.newaxis, :]
                tf_final = gcv.weighted_tf * w_tf[np.newaxis, :]
            morph_keep_ratio = 1.0
            stage = "P1-wsst_like_sst+gcv-fallback"
        y = reconstruct_from_weights(wsst, final_weights, f_s=f_s, f_e=f_e)
        tf_org = wsst.tf
        if w_tf is None:
            tf_gcv = gcv.weighted_tf
        else:
            tf_gcv = gcv.weighted_tf * w_tf[np.newaxis, :]
        freq = wsst.freq

    if not (return_debug or kwargs.get("return_result", False)):
        return y

    meta: Dict[str, Any] = {
        "stage": stage,
        "dt": float(dt),
        "f_s": float(f_s),
        "f_e": float(f_e),
        "bwconn": int(bwconn),
        "strength": float(strength),
        "morph_enable": int(bool(morph_enable)),
        "morph_keep_ratio": float(morph_keep_ratio),
        "morph_min_area": int(morph_min_area),
        "morph_quantile": float(morph_quantile),
    }

    debug: Optional[DenoiseDebug] = None
    if return_debug:
        debug = DenoiseDebug(
            org_tf=tf_org,
            gcv_tf=tf_gcv,
            final_tf=tf_final,
            freq=freq,
        )

    return DenoiseResult(data=y, debug=debug, meta=meta)


def denoise_section(
    traces: Union[np.ndarray, list],
    dt: float,
    *,
    f_s: float,
    f_e: float,
    bwconn: int = 8,
    strength: float = 1.0,
    workers: int = 1,
    return_debug: bool = False,
    **kwargs: Any,
) -> Union[np.ndarray, DenoiseResult]:
    """Multi-trace denoise entrypoint (gather/section).

    P0 behavior: validate + pass-through.
    """
    _validate_common(dt=dt, f_s=f_s, f_e=f_e, bwconn=bwconn)
    if workers < 1:
        raise ValueError(f"`workers` 必须 >= 1，实际为 {workers!r}")
    if return_debug:
        raise ValueError("P0 阶段暂不支持 `denoise_section(return_debug=True)`（会产生大量中间数据）")

    X = np.asarray(traces)
    if X.ndim != 2:
        raise ValueError(f"`traces` 必须是二维数组 (n_traces, n_samples)，实际维度为 {X.ndim}")
    if X.size == 0 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("`traces` 不能为空")
    if not np.all(np.isfinite(X)):
        raise ValueError("`traces` 包含 NaN/Inf，去噪前请先处理无效值")

    if not np.isfinite(strength) or strength <= 0:
        raise ValueError(f"`strength` 必须为正数，实际为 {strength!r}")

    Y = X.astype(np.float64, copy=True)
    n_tr = int(Y.shape[0])

    # Optional selected indices support (backward-compatible via kwargs).
    indices_raw = kwargs.get("indices", None)
    if indices_raw is None:
        target_idx = np.arange(n_tr, dtype=int)
    else:
        idx_arr = np.asarray(indices_raw, dtype=int).reshape(-1)
        target_idx = np.asarray([int(i) for i in idx_arr.tolist() if 0 <= int(i) < n_tr], dtype=int)
        if target_idx.size > 0:
            target_idx = np.unique(target_idx)
    if target_idx.size == 0:
        if not kwargs.get("return_result", False):
            return Y
        meta: Dict[str, Any] = {
            "stage": "P2-section-empty",
            "dt": float(dt),
            "f_s": float(f_s),
            "f_e": float(f_e),
            "bwconn": int(bwconn),
            "strength": float(strength),
            "workers": int(workers),
            "applied_traces": 0,
            "target_traces": 0,
        }
        return DenoiseResult(data=Y, debug=None, meta=meta)

    run_workers = int(max(1, min(int(workers), int(target_idx.size))))
    progress_cb = kwargs.get("progress_callback", None)
    morph_kwargs = {
        "morph_enable": kwargs.get("morph_enable", True),
        "morph_min_area": kwargs.get("morph_min_area", 24),
        "morph_quantile": kwargs.get("morph_quantile", 0.70),
        "morph_floor_ratio": kwargs.get("morph_floor_ratio", 0.03),
        "morph_expand": kwargs.get("morph_expand", 1),
        "morph_keep_strong_quantile": kwargs.get("morph_keep_strong_quantile", 0.95),
        "morph_bwconn": kwargs.get("morph_bwconn", bwconn),
    }

    pick_rows: Optional[List[List[float]]]
    pr_raw = kwargs.get("pick_times_per_row", None)
    if pr_raw is None:
        pick_rows = None
    elif not isinstance(pr_raw, (list, tuple)) or len(pr_raw) != int(n_tr):
        pick_rows = None
    else:
        pick_rows = [list(row) if row is not None else [] for row in pr_raw]  # type: ignore[arg-type]

    pick_static = {
        "pick_guidance_enable": bool(kwargs.get("pick_guidance_enable", False)),
        "pick_times_axis": kwargs.get("pick_times_axis", None),
        "pick_guidance_floor": float(kwargs.get("pick_guidance_floor", 0.12)),
        "pick_t0_fallback": float(kwargs.get("pick_t0_fallback", 0.0)),
        # 显式 FWHM（秒）；与旧字段 pick_half_width_sec（曾为 σ）二择一由 wavelet_length_fwhm_sec_from_pick_kwargs 解析
        "pick_wavelet_length_sec": kwargs.get("pick_wavelet_length_sec", None),
        "pick_half_width_sec": kwargs.get("pick_half_width_sec", None),
    }

    def _extras_for_row(i_row: int) -> Dict[str, Any]:
        tc: Dict[str, Any] = {**morph_kwargs, **pick_static}
        if pick_rows is not None:
            ir = int(i_row)
            tc["pick_times"] = list(pick_rows[ir]) if 0 <= ir < len(pick_rows) else []
        else:
            tc["pick_times"] = list(kwargs.get("pick_times", []) or [])
        return tc

    if progress_cb is not None and not callable(progress_cb):
        progress_cb = None
    done_count = 0

    def _emit_progress(done: int, total: int) -> None:
        cb = progress_cb
        if cb is None:
            return
        try:
            cb(int(done), int(total))
        except Exception:
            pass

    def _denoise_one(i_trace: int) -> Tuple[int, np.ndarray, bool]:
        x = np.asarray(Y[int(i_trace), :], dtype=np.float64)
        try:
            y = denoise_trace(
                x,
                dt=dt,
                f_s=f_s,
                f_e=f_e,
                bwconn=bwconn,
                strength=float(strength),
                return_debug=False,
                **_extras_for_row(int(i_trace)),
            )
            y = np.asarray(y, dtype=np.float64).reshape(-1)
            if y.size != x.size:
                y = np.resize(y, x.shape)
            return int(i_trace), y, True
        except Exception:
            # Fail-safe: keep original trace.
            return int(i_trace), x, False

    applied = 0
    if run_workers <= 1 or target_idx.size <= 1:
        for i_tr in target_idx.tolist():
            i_out, y_out, ok = _denoise_one(int(i_tr))
            Y[i_out, :] = y_out
            if ok:
                applied += 1
            done_count += 1
            _emit_progress(done_count, int(target_idx.size))
    else:
        chunk_size = int(np.ceil(target_idx.size / max(1, run_workers * 4)))
        chunk_size = int(min(8, max(2, chunk_size)))
        chunks: List[List[int]] = [
            [int(v) for v in target_idx[k : k + chunk_size].tolist()]
            for k in range(0, int(target_idx.size), chunk_size)
        ]

        def _run_chunk(chunk: List[int]) -> List[Tuple[int, np.ndarray, bool]]:
            return [_denoise_one(int(i_tr)) for i_tr in chunk]

        with ThreadPoolExecutor(max_workers=run_workers) as pool:
            futs = [pool.submit(_run_chunk, ch) for ch in chunks]
            for fut in as_completed(futs):
                for i_out, y_out, ok in fut.result():
                    Y[i_out, :] = y_out
                    if ok:
                        applied += 1
                    done_count += 1
                    _emit_progress(done_count, int(target_idx.size))

    if not kwargs.get("return_result", False):
        return Y

    meta: Dict[str, Any] = {
        "stage": "P2-section-trace-denoise",
        "dt": float(dt),
        "f_s": float(f_s),
        "f_e": float(f_e),
        "bwconn": int(bwconn),
        "strength": float(strength),
        "workers": int(workers),
        "applied_traces": int(applied),
        "target_traces": int(target_idx.size),
    }
    return DenoiseResult(data=Y, debug=None, meta=meta)

