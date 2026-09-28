# -*- coding: utf-8 -*-
"""Denoise A/B compare plots mixed into QtFastViewer."""

from __future__ import annotations

import math
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as exc:  # pragma: no cover
    raise RuntimeError("未安装 PySide6") from exc

try:
    from pyAOBS.processors.denoise import denoise_trace, denoise_section
    from pyAOBS.processors.denoise.metrics import ab_trace_metrics, summarize_metrics
    from pyAOBS.processors.denoise.ssq_backend import HAVE_SSQ as DENOISE_HAVE_SSQ
except ImportError:
    denoise_trace = None  # type: ignore
    denoise_section = None  # type: ignore
    ab_trace_metrics = None  # type: ignore
    summarize_metrics = None  # type: ignore
    DENOISE_HAVE_SSQ = False


def _tf_abs_amp(tf: object) -> np.ndarray:
    """|TF| amplitude as float64; avoids ComplexWarning if backend returns complex."""
    return np.abs(np.asarray(tf)).astype(np.float64, copy=False)


class DenoiseCompareMixin:
    """去噪前后对比图 / 全道谱剖面。"""

    def _denoise_compare_cached_trace_ids(self) -> List[int]:
        """返回当前去噪缓存中成对存在的全局道号（已排序）。"""
        if not getattr(self, "_denoise_frozen_ready", False):
            return []
        return sorted(
            int(k)
            for k in self._denoise_frozen_by_trace.keys()
            if int(k) in self._denoise_frozen_original_by_trace
        )



    def _open_denoise_compare_plot(self) -> None:
        """弹出 Matplotlib 窗口：同一道的去噪前后时域、频谱及 TF 形态对比。"""
        if self.loaded is None:
            QtWidgets.QMessageBox.warning(self, "Denoise compare", "Load data first.")
            return
        if (not getattr(self, "_denoise_frozen_ready", False)) or len(getattr(self, "_denoise_frozen_by_trace", {}) or {}) == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Denoise compare",
                "No denoise cache yet. Enable denoise and click Start to compute, then try again.",
            )
            return
        cand = self._denoise_compare_cached_trace_ids()
        if not cand:
            QtWidgets.QMessageBox.warning(
                self,
                "Denoise compare",
                "Cache has no pre-denoise waveforms. Click Start to recompute, then try again.",
            )
            return
        items = [str(i) for i in cand]
        default_row = max(0, len(items) // 2)
        pick, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Denoise A/B figure",
            "Trace index (global):",
            items,
            default_row,
            False,
        )
        if not ok:
            return
        try:
            gidx = int(str(pick).strip())
        except Exception:
            return
        if gidx not in cand:
            return
        self._figure_denoise_ab_compare(int(gidx))



    def _figure_denoise_ab_compare(self, gidx: int) -> None:
        """绘制单独图窗：指定全局道在去噪前后的时域曲线、振幅谱（dB）及 TF（需临时重算）。 """
        try:
            import matplotlib.colors as mcolors
            from matplotlib.figure import Figure
            try:
                from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            except Exception:
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        except Exception as exc_import:
            QtWidgets.QMessageBox.warning(
                self,
                "Denoise compare",
                f"Matplotlib is not available ({type(exc_import).__name__}). Install matplotlib to plot.",
            )
            self._debug_log("DENOISE_COMPARE", f"matplotlib missing: {type(exc_import).__name__}")
            return

        x0_raw = np.asarray(self._denoise_frozen_original_by_trace.get(int(gidx)), dtype=np.float64)
        y0_raw = np.asarray(self._denoise_frozen_by_trace.get(int(gidx)), dtype=np.float64)
        if x0_raw.size == 0 or y0_raw.size == 0:
            QtWidgets.QMessageBox.warning(self, "Denoise compare", "Invalid cache for the selected trace.")
            return
        n = int(min(int(x0_raw.size), int(y0_raw.size)))
        times = np.asarray(self.loaded.get("times", []), dtype=float)
        if times.size < max(8, int(n)):
            QtWidgets.QMessageBox.warning(self, "Denoise compare", "Time axis is too short for this comparison.")
            return
        n = min(n, int(times.size))
        times = np.asarray(times[:n], dtype=float)
        x0 = np.asarray(x0_raw[:n], dtype=np.float64)
        y0 = np.asarray(y0_raw[:n], dtype=np.float64)
        if times.size >= 2:
            dt_raw = float(abs(times[1] - times[0]))
            dt_ok = dt_raw if (np.isfinite(dt_raw) and dt_raw > 0.0) else None
        else:
            dt_ok = None
        if dt_ok is None:
            QtWidgets.QMessageBox.warning(self, "Denoise compare", "Invalid sample interval dt; cannot plot.")
            return

        self._sync_denoise_params_from_ui()
        f_s = float(self._denoise_params.get("f_s", 3.0))
        f_e = float(self._denoise_params.get("f_e", 20.0))
        bwconn = int(self._denoise_params.get("bwconn", 8))
        strength = float(self._denoise_params.get("strength", 3.0))
        morph_enable = bool(self._denoise_params.get("morph_enable", True))
        morph_quantile = float(self._denoise_params.get("morph_quantile", 0.70))
        morph_min_area = int(self._denoise_params.get("morph_min_area", 24))
        morph_expand = int(self._denoise_params.get("morph_expand", 1))
        morph_floor_ratio = float(self._denoise_params.get("morph_floor_ratio", 0.03))
        morph_keep_strong_q = float(self._denoise_params.get("morph_keep_strong_q", 0.95))
        pg = bool(self._denoise_params.get("pick_guidance", False))
        cand_cmp = self._denoise_compare_cached_trace_ids()
        pt = self._denoise_pick_template_times_for_globals([int(x) for x in cand_cmp]) if (pg and cand_cmp) else []
        ta = np.asarray(times[:n], dtype=np.float64)
        t0fb = float(ta[0]) if ta.size > 0 else 0.0
        phw = float(self._denoise_params.get("pick_wavelet_length_sec", 0.19))
        pfl = float(self._denoise_params.get("pick_guidance_floor", 0.12))

        dbg_ok = False
        org_tf_amp: Optional[np.ndarray] = None
        fin_tf_amp: Optional[np.ndarray] = None
        freq_hz: Optional[np.ndarray] = None
        dbg_stage = ""
        try:
            dr = denoise_trace(
                np.asarray(x0, dtype=np.float64),
                dt=dt_ok,
                f_s=f_s,
                f_e=f_e,
                bwconn=bwconn,
                strength=strength,
                morph_enable=morph_enable,
                morph_quantile=morph_quantile,
                morph_min_area=morph_min_area,
                morph_expand=morph_expand,
                morph_floor_ratio=morph_floor_ratio,
                morph_keep_strong_quantile=morph_keep_strong_q,
                morph_bwconn=bwconn,
                return_debug=True,
                pick_guidance_enable=bool(pg),
                pick_times=pt,
                pick_times_axis=ta,
                pick_t0_fallback=t0fb,
                pick_wavelet_length_sec=phw,
                pick_guidance_floor=pfl,
            )
            if hasattr(dr, "debug") and dr.debug is not None:
                dbg = dr.debug
                org_tf_amp = _tf_abs_amp(dbg.org_tf)
                fin_tf_amp = _tf_abs_amp(dbg.final_tf)
                freq_hz = np.asarray(dbg.freq, dtype=np.float64)
                meta = getattr(dr, "meta", {}) or {}
                dbg_stage = str(meta.get("stage", "")).strip()
                dbg_ok = org_tf_amp.size > 0 and fin_tf_amp.size > 0 and freq_hz.size > 1
                if dbg_ok and (org_tf_amp.shape != fin_tf_amp.shape):
                    dbg_ok = False
        except Exception as exc_tf:
            self._debug_log("DENOISE_COMPARE", f"tf-debug fail: {type(exc_tf).__name__}: {exc_tf}")
            dbg_ok = False

        def _spectral_db(sig: np.ndarray, dt_use: float) -> Tuple[np.ndarray, np.ndarray]:
            sig_loc = np.asarray(sig, dtype=np.float64).reshape(-1)
            nn = int(sig_loc.size)
            if nn < 8:
                return np.asarray([0.0], dtype=np.float64), np.asarray([-200.0, -200.0], dtype=np.float64)
            xc = sig_loc - float(np.mean(sig_loc))
            win = np.hanning(nn)
            spec = np.fft.rfft(xc * win)
            fq = np.fft.rfftfreq(nn, d=float(dt_use))
            mag = np.abs(spec)
            ref = float(np.max(mag)) if mag.size > 0 else 0.0
            floor = float(max(1e-20 * max(ref, 1.0), 1e-30))
            db = 20.0 * np.log10(np.maximum(mag, floor))
            return fq, db

        fq_a, db_a = _spectral_db(x0, dt_ok)
        fq_b, db_b = _spectral_db(y0, dt_ok)

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Denoise compare — trace {int(gidx)}")
        dlg.setMinimumSize(940, 720)
        vbox = QtWidgets.QVBoxLayout(dlg)

        figure = Figure(figsize=(11, 9), tight_layout=False)
        canvas = FigureCanvas(figure)

        span = float(max(times[-1] - times[0], dt_ok))

        irec_lab = ""
        try:
            irec_lab = str(int(self.spin_irec.value()))
        except Exception:
            irec_lab = ""

        suptitles = []
        suptitles.append(
            f"record={irec_lab} · trace={int(gidx)} · dt={dt_ok:g} s · f_band=({f_s:g},{f_e:g}) Hz",
        )
        suptitles.append(
            "Blue/orange: time-domain A/B; light green: filled (B−A) scaled by std; "
            + (
                "TF: " + dbg_stage + " (trace B on main view may differ slightly after coherence blend). "
                if dbg_ok
                else "TF: not available (time and spectrum plots only). "
            ),
        )
        try:
            _m1 = ab_trace_metrics(x0, y0)
            suptitles.append(
                f"Metrics (A=input, B=output): SNR={_m1.snr_db:.2f} dB  RMSE={_m1.rmse:.4g}  "
                f"CC={_m1.cc:.4f}  MAE={_m1.mae:.4g}  |  SNR=10·log10(mean(A²)/MSE(A,B))"
            )
            self._debug_log(
                "DENOISE_METRICS",
                f"trace={int(gidx)} SNR_dB={_m1.snr_db:.4f} RMSE={_m1.rmse:.6e} CC={_m1.cc:.6f} MAE={_m1.mae:.6e}",
            )
        except Exception as exc_m:
            suptitles.append(f"Metrics: unavailable ({type(exc_m).__name__})")
            self._debug_log("DENOISE_METRICS", f"trace={int(gidx)} fail:{type(exc_m).__name__}")

        figure.suptitle("\n".join(suptitles), fontsize=10)

        if dbg_ok:
            gs = figure.add_gridspec(3, 2, height_ratios=[2.4, 1.05, 2.05], width_ratios=[1.0, 1.0])
            gs.update(left=0.075, right=0.975, top=0.90, bottom=0.065, hspace=0.40, wspace=0.20)
            ax_t = figure.add_subplot(gs[0, :])
            ax_sp = figure.add_subplot(gs[1, :])
            ax_tf0 = figure.add_subplot(gs[2, 0])
            ax_tf1 = figure.add_subplot(gs[2, 1])
        else:
            gs = figure.add_gridspec(2, 1, height_ratios=[2.4, 1.05])
            gs.update(left=0.075, right=0.975, top=0.90, bottom=0.065, hspace=0.36)
            ax_t = figure.add_subplot(gs[0, :])
            ax_sp = figure.add_subplot(gs[1, :])
            ax_tf0 = ax_tf1 = None  # type: ignore[misc]

        ax_t.plot(times, x0, color="#2166ac", lw=0.95, alpha=0.92, label="A pre-denoise (display pipeline)")
        ax_t.plot(times, y0, color="#ef8a62", lw=0.95, alpha=0.92, label="B final (cache, incl. coherence blend)")

        eps_d = np.maximum(np.std(x0), np.std(y0))
        diff_v = np.asarray(y0 - x0, dtype=np.float64)
        gx = np.clip(diff_v / max(8.0 * float(eps_d if eps_d > 1e-30 else 1.0), 1e-20), -1.0, 1.0)
        ax_t.fill_between(times, 0.0, gx, alpha=0.22, color="#4daf4a", label="Δ(B−A) fill (scaled to ±1)")
        ax_t.set_xlabel(f"Time (s) [{span:g} s]")
        ax_t.set_ylabel("Amplitude")
        ax_t.legend(loc="upper right", fontsize=9)
        ax_t.grid(True, alpha=0.25)

        ax_sp.plot(fq_a, db_a, color="#2166ac", lw=1.05, label="A Hanning FFT spectrum (relative dB)")
        ax_sp.plot(fq_b, db_b, color="#ef8a62", lw=1.05, label="B Hanning FFT spectrum (relative dB)")
        ax_sp.set_xlabel("Frequency (Hz)")
        ax_sp.set_ylabel("relative dB")
        f_hi = float(min(125.0, 0.5 / float(dt_ok)))
        ax_sp.set_xlim(0.0, max(f_hi, float(f_e) * 3.0, 50.0))
        ax_sp.grid(True, which="major", alpha=0.27)
        ax_sp.legend(loc="upper right", fontsize=9)

        if (
            dbg_ok
            and org_tf_amp is not None
            and fin_tf_amp is not None
            and freq_hz is not None
            and ax_tf0 is not None
            and ax_tf1 is not None
        ):
            m_tf = int(org_tf_amp.shape[1])
            t1_tf = float(times[0]) + float(dt_ok) * float(max(0, m_tf - 1))
            ext = [float(times[0]), float(min(t1_tf, float(times[-1]))), float(freq_hz[0]), float(freq_hz[-1])]
            vmin_db = float(
                max(
                    -140.0,
                    20.0 * np.log10(float(np.maximum(np.percentile(org_tf_amp, 1.5), 1e-40))),
                ),
            )
            vmax_raw = float(
                (np.percentile(org_tf_amp, 99.0) + np.percentile(fin_tf_amp, 99.0)) / 3.5 + 1e-20,
            )
            vmax_db = float(20.0 * np.log10(max(vmax_raw, 1e-30)))
            if not (np.isfinite(vmin_db) and np.isfinite(vmax_db) and vmax_db > vmin_db + 1e-3):
                vmin_db, vmax_db = -80.0, 0.0
            try:
                nrm = mcolors.Normalize(vmin=vmin_db, vmax=vmax_db)
            except Exception:
                nrm = None

            _ = ax_tf0.imshow(
                20.0 * np.log10(org_tf_amp + 1e-30),
                aspect="auto",
                origin="lower",
                cmap="inferno",
                interpolation="nearest",
                extent=list(ext),
                norm=nrm,
            )
            im1 = ax_tf1.imshow(
                20.0 * np.log10(fin_tf_amp + 1e-30),
                aspect="auto",
                origin="lower",
                cmap="inferno",
                interpolation="nearest",
                extent=list(ext),
                norm=nrm,
            )
            ax_tf0.set_title("|TF| before denoise (transform domain)")
            ax_tf1.set_title("|TF| after GCV+morph (single denoise_trace)")
            for ax_tf in (ax_tf0, ax_tf1):
                ax_tf.set_xlabel("Time (s)")
                ax_tf.set_ylabel("Freq (Hz)")
            try:
                cbar = figure.colorbar(im1, ax=[ax_tf0, ax_tf1], shrink=0.78, pad=0.02)
                cbar.ax.set_ylabel("dB")
                cbar.ax.tick_params(labelsize=8)
            except Exception:
                pass

        btn_row = QtWidgets.QHBoxLayout()
        btn_export = QtWidgets.QPushButton("Export PNG...")
        btn_close = QtWidgets.QPushButton("Close")
        btn_row.addStretch(1)
        btn_row.addWidget(btn_export)
        btn_row.addWidget(btn_close)
        btn_close.clicked.connect(dlg.close)

        path_default = Path.cwd() / f"zplotpy_denoise_compare_trace{int(gidx)}_irec{irec_lab or 'NA'}.png"

        def _export_png():
            tgt, sf = self._get_save_file_name(
                "Export comparison PNG",
                str(path_default),
                "PNG (*.png);;All files (*)",
                default_suffix=".png",
                parent=dlg,
            )
            if not tgt:
                return
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"This figure includes Axes that are not compatible with tight_layout.*",
                        category=UserWarning,
                    )
                    figure.savefig(str(tgt), dpi=145, bbox_inches="tight")
                self._set_status_text(f"Exported comparison: {tgt}", hold_ms=2600)
                self._debug_log("DENOISE_COMPARE", f"export_png path={tgt}")
            except Exception as exc_sv:
                QtWidgets.QMessageBox.warning(dlg, "Export failed", str(exc_sv))

        btn_export.clicked.connect(_export_png)

        vbox.addWidget(canvas)
        vbox.addLayout(btn_row)

        dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.resize(980, 820)
        dlg.show()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"This figure includes Axes that are not compatible with tight_layout.*",
                    category=UserWarning,
                )
                canvas.draw_idle()
        except Exception:
            pass
        self._floating_dialogs.append(dlg)

        try:
            self._debug_log(
                "DENOISE_COMPARE",
                f"gidx={int(gidx)} n={int(n)} dbg_tf={int(dbg_ok)} stage={dbg_stage}",
            )
        except Exception:
            pass



    def _open_denoise_compare_all_tf_spectrum(self) -> None:
        """全缓存道：振幅谱剖面 (freq × trace) + 各道 |TF| 沿时间轴横向拼接（ribbon，dB）。"""
        if self.loaded is None:
            QtWidgets.QMessageBox.warning(self, "Denoise compare", "Load data first.")
            return
        if (not getattr(self, "_denoise_frozen_ready", False)) or len(getattr(self, "_denoise_frozen_by_trace", {}) or {}) == 0:
            QtWidgets.QMessageBox.warning(
                self,
                "Denoise compare",
                "No denoise cache yet. Enable denoise and click Start to compute, then try again.",
            )
            return
        cand = self._denoise_compare_cached_trace_ids()
        if not cand:
            QtWidgets.QMessageBox.warning(
                self,
                "Denoise compare",
                "Cache has no pre-denoise waveforms. Click Start to recompute, then try again.",
            )
            return
        n_tr = int(len(cand))
        if n_tr > 1200:
            ans = QtWidgets.QMessageBox.question(
                self,
                "Denoise compare",
                f"{n_tr} traces in cache. |TF| ribbon runs denoise_trace per trace and may take a long time. Continue?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if ans != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        try:
            import matplotlib.colors as mcolors
            from matplotlib.figure import Figure
            try:
                from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            except Exception:
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        except Exception as exc_import:
            QtWidgets.QMessageBox.warning(
                self,
                "Denoise compare",
                f"Matplotlib is not available ({type(exc_import).__name__}). Install matplotlib to plot.",
            )
            return

        times = np.asarray(self.loaded.get("times", []), dtype=float)
        if times.size < 8:
            QtWidgets.QMessageBox.warning(self, "Denoise compare", "Time axis is too short.")
            return
        dt_raw = float(abs(times[1] - times[0]))
        if (not np.isfinite(dt_raw)) or dt_raw <= 0.0:
            QtWidgets.QMessageBox.warning(self, "Denoise compare", "Invalid sample interval dt.")
            return
        dt_ok = float(dt_raw)

        lens = [
            int(
                min(
                    int(np.asarray(self._denoise_frozen_original_by_trace[int(g)], dtype=np.float64).size),
                    int(np.asarray(self._denoise_frozen_by_trace[int(g)], dtype=np.float64).size),
                ),
            )
            for g in cand
        ]
        n_min = int(min(lens + [int(times.size)]))
        if n_min < 8:
            QtWidgets.QMessageBox.warning(self, "Denoise compare", "Not enough samples per trace.")
            return

        self._sync_denoise_params_from_ui()
        f_s = float(self._denoise_params.get("f_s", 3.0))
        f_e = float(self._denoise_params.get("f_e", 20.0))
        bwconn = int(self._denoise_params.get("bwconn", 8))
        strength = float(self._denoise_params.get("strength", 3.0))
        morph_enable = bool(self._denoise_params.get("morph_enable", True))
        morph_quantile = float(self._denoise_params.get("morph_quantile", 0.70))
        morph_min_area = int(self._denoise_params.get("morph_min_area", 24))
        morph_expand = int(self._denoise_params.get("morph_expand", 1))
        morph_floor_ratio = float(self._denoise_params.get("morph_floor_ratio", 0.03))
        morph_keep_strong_q = float(self._denoise_params.get("morph_keep_strong_q", 0.95))
        worker_n = int(max(1, int(self._denoise_params.get("workers", 1))))
        pg_cmp = bool(self._denoise_params.get("pick_guidance", False))
        phw_cmp = float(self._denoise_params.get("pick_wavelet_length_sec", 0.19))
        pfl_cmp = float(self._denoise_params.get("pick_guidance_floor", 0.12))
        times_cmp = np.asarray(times[:n_min], dtype=np.float64)
        t0_cmp = float(times_cmp[0]) if times_cmp.size > 0 else 0.0
        cand_cmp_ids = [int(x) for x in cand]
        pg_template_cmp = (
            self._denoise_pick_template_times_for_globals(cand_cmp_ids) if pg_cmp else []
        )

        mat_a = np.zeros((n_tr, n_min), dtype=np.float64)
        mat_b = np.zeros((n_tr, n_min), dtype=np.float64)
        for j, gidx in enumerate(cand):
            xa = np.asarray(self._denoise_frozen_original_by_trace[int(gidx)], dtype=np.float64)[:n_min]
            xb = np.asarray(self._denoise_frozen_by_trace[int(gidx)], dtype=np.float64)[:n_min]
            mat_a[j, :] = xa
            mat_b[j, :] = xb

        win1 = np.hanning(n_min)
        xa_dm = mat_a - np.mean(mat_a, axis=1, keepdims=True)
        xb_dm = mat_b - np.mean(mat_b, axis=1, keepdims=True)
        spec_a = np.abs(np.fft.rfft(xa_dm * win1, axis=1))
        spec_b = np.abs(np.fft.rfft(xb_dm * win1, axis=1))
        fq = np.fft.rfftfreq(n_min, d=dt_ok)
        colmax_a = np.maximum(np.max(spec_a, axis=1, keepdims=True), 1e-30)
        colmax_b = np.maximum(np.max(spec_b, axis=1, keepdims=True), 1e-30)
        db_a_g = 20.0 * np.log10(np.maximum(spec_a / colmax_a, 1e-30))
        db_b_g = 20.0 * np.log10(np.maximum(spec_b / colmax_b, 1e-30))
        img_spec_a = np.asarray(db_a_g.T, dtype=np.float64)
        img_spec_b = np.asarray(db_b_g.T, dtype=np.float64)

        per_ab_metrics = [ab_trace_metrics(mat_a[j, :], mat_b[j, :]) for j in range(n_tr)]
        sm, sd, rmm, rmd, cm, cd, mm, md = summarize_metrics(per_ab_metrics)
        lbl_denoise_ab_metrics = QtWidgets.QLabel(
            "A/B metrics (cached traces): SNR dB = 10·log10(mean(A²)/MSE(A,B)); "
            "values are median | mean.\n"
            f"SNR: {sd:.2f} | {sm:.2f} dB   RMSE: {rmd:.4g} | {rmm:.4g}   "
            f"CC: {cd:.4f} | {cm:.4f}   MAE: {md:.4g} | {mm:.4g}"
        )
        lbl_denoise_ab_metrics.setStyleSheet(
            "font-family: Consolas, 'Courier New', monospace; font-size:11px; color:#333;",
        )
        lbl_denoise_ab_metrics.setWordWrap(True)
        try:
            self._debug_log(
                "DENOISE_METRICS",
                f"all n={n_tr} SNR_med={sd:.4f} SNR_mean={sm:.4f} RMSE_med={rmd:.6e} CC_med={cd:.6f}",
            )
        except Exception:
            pass

        prog = QtWidgets.QProgressDialog(
            "|TF| ribbon: denoise_trace (return_debug) per trace…",
            "Cancel",
            0,
            n_tr,
            self,
        )
        prog.setWindowTitle("Denoise compare — all traces")
        prog.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)
        try:
            prog.show()
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass

        def _tf_job(gidx: int) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
            xv = np.asarray(self._denoise_frozen_original_by_trace[int(gidx)], dtype=np.float64)[:n_min]
            try:
                pt_j = list(pg_template_cmp) if pg_cmp else []
                dr = denoise_trace(
                    xv,
                    dt=dt_ok,
                    f_s=f_s,
                    f_e=f_e,
                    bwconn=bwconn,
                    strength=strength,
                    morph_enable=morph_enable,
                    morph_quantile=morph_quantile,
                    morph_min_area=morph_min_area,
                    morph_expand=morph_expand,
                    morph_floor_ratio=morph_floor_ratio,
                    morph_keep_strong_quantile=morph_keep_strong_q,
                    morph_bwconn=bwconn,
                    return_debug=True,
                    pick_guidance_enable=bool(pg_cmp),
                    pick_times=pt_j,
                    pick_times_axis=times_cmp,
                    pick_t0_fallback=t0_cmp,
                    pick_wavelet_length_sec=phw_cmp,
                    pick_guidance_floor=pfl_cmp,
                )
                dbg = getattr(dr, "debug", None)
                if dbg is None:
                    return int(gidx), None, None, None
                oa = _tf_abs_amp(dbg.org_tf)
                fa = _tf_abs_amp(dbg.final_tf)
                fr = np.asarray(dbg.freq, dtype=np.float64)
                if oa.shape != fa.shape or fr.size < 2:
                    return int(gidx), None, None, None
                return int(gidx), oa, fa, fr
            except Exception:
                return int(gidx), None, None, None

        tf_pairs: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        freq_hz: Optional[np.ndarray] = None
        done_tf = 0
        cancel_tf = False
        if worker_n <= 1 or n_tr <= 2:
            for gidx in cand:
                if prog.wasCanceled():
                    cancel_tf = True
                    break
                gi, oa, fa, fr = _tf_job(int(gidx))
                if fr is not None and freq_hz is None:
                    freq_hz = fr
                if oa is not None and fa is not None and fr is not None:
                    tf_pairs[int(gi)] = (oa, fa, fr)
                done_tf += 1
                prog.setValue(done_tf)
                try:
                    QtWidgets.QApplication.processEvents()
                except Exception:
                    pass
        else:
            max_w = int(min(worker_n, n_tr, 32))
            with ThreadPoolExecutor(max_workers=max_w) as pool:
                fut_to_g = {pool.submit(_tf_job, int(g)): int(g) for g in cand}
                for fut in as_completed(fut_to_g):
                    if prog.wasCanceled():
                        cancel_tf = True
                        break
                    gi, oa, fa, fr = fut.result()
                    if fr is not None and freq_hz is None:
                        freq_hz = fr
                    if oa is not None and fa is not None and fr is not None:
                        tf_pairs[int(gi)] = (oa, fa, fr)
                    done_tf += 1
                    prog.setValue(done_tf)
                    try:
                        QtWidgets.QApplication.processEvents()
                    except Exception:
                        pass
        try:
            prog.close()
        except Exception:
            pass

        stack_o: List[np.ndarray] = []
        stack_f: List[np.ndarray] = []
        ref_shape: Optional[Tuple[int, ...]] = None
        for gidx in cand:
            pr = tf_pairs.get(int(gidx))
            if not pr:
                continue
            oa, fa, _fr = pr
            if ref_shape is None:
                ref_shape = tuple(oa.shape)
            if tuple(oa.shape) == ref_shape and tuple(fa.shape) == ref_shape:
                stack_o.append(oa)
                stack_f.append(fa)

        ribbon_org: Optional[np.ndarray] = None
        ribbon_fin: Optional[np.ndarray] = None
        n_t_one = 0
        if stack_o and stack_f and len(stack_o) == len(stack_f):
            try:
                ribbon_org = np.concatenate(stack_o, axis=1)
                ribbon_fin = np.concatenate(stack_f, axis=1)
                n_t_one = int(stack_o[0].shape[1])
            except Exception:
                ribbon_org = ribbon_fin = None
                n_t_one = 0
        tf_ok = bool(
            ribbon_org is not None
            and ribbon_fin is not None
            and ribbon_org.shape == ribbon_fin.shape
            and freq_hz is not None
            and int(freq_hz.size) > 1
            and int(ribbon_org.shape[1]) > 1,
        )

        t0 = float(times[0])
        if tf_ok and ribbon_org is not None:
            t1_tf = float(t0 + dt_ok * float(max(0, int(ribbon_org.shape[1]) - 1)))
        else:
            t1_tf = float(times[int(min(n_min - 1, int(times.size) - 1))])
        ext_tf: Optional[List[float]] = None
        if tf_ok and ribbon_org is not None and freq_hz is not None:
            ext_tf = [t0, float(t1_tf), float(freq_hz[0]), float(freq_hz[-1])]

        irec_lab = ""
        try:
            irec_lab = str(int(self.spin_irec.value()))
        except Exception:
            irec_lab = ""

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Denoise — all traces (n={n_tr})")
        vbox = QtWidgets.QVBoxLayout(dlg)
        figure = Figure(figsize=(12, 9), tight_layout=False)
        canvas = FigureCanvas(figure)

        if tf_ok and ribbon_org is not None and ribbon_fin is not None and ext_tf is not None:
            gs = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.15], width_ratios=[1.0, 1.0])
            gs.update(left=0.07, right=0.98, top=0.91, bottom=0.07, hspace=0.33, wspace=0.22)
            ax_sa = figure.add_subplot(gs[0, 0])
            ax_sb = figure.add_subplot(gs[0, 1])
            ax_t0 = figure.add_subplot(gs[1, 0])
            ax_t1 = figure.add_subplot(gs[1, 1])
        else:
            gs = figure.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
            gs.update(left=0.07, right=0.98, top=0.90, bottom=0.10, wspace=0.22)
            ax_sa = figure.add_subplot(gs[0, 0])
            ax_sb = figure.add_subplot(gs[0, 1])
            ax_t0 = ax_t1 = None  # type: ignore[misc]

        x_edges = np.arange(n_tr + 1, dtype=np.float64) - 0.5
        fq0 = float(fq[0])
        fq1 = float(fq[-1])
        extent_sp = [float(x_edges[0]), float(x_edges[-1]), fq0, fq1]

        ax_sa.imshow(
            img_spec_a,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
            extent=extent_sp,
        )
        ax_sa.set_title("Spectrum gather A (per-trace peak = 0 dB)")
        ax_sa.set_xlabel("Trace column (sorted cache order)")
        ax_sa.set_ylabel("Frequency (Hz)")
        tick_idx = np.linspace(0, n_tr - 1, num=min(9, n_tr), dtype=int)
        ax_sa.set_xticks([float(i) for i in tick_idx])
        ax_sa.set_xticklabels([str(int(cand[int(i)])) for i in tick_idx], rotation=35, fontsize=7)

        ax_sb.imshow(
            img_spec_b,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
            extent=extent_sp,
        )
        ax_sb.set_title("Spectrum gather B (per-trace peak = 0 dB)")
        ax_sb.set_xlabel("Trace column (sorted cache order)")
        ax_sb.set_ylabel("Frequency (Hz)")
        ax_sb.set_xticks([float(i) for i in tick_idx])
        ax_sb.set_xticklabels([str(int(cand[int(i)])) for i in tick_idx], rotation=35, fontsize=7)
        f_hi = float(min(125.0, 0.5 / float(dt_ok)))
        for axs in (ax_sa, ax_sb):
            axs.set_ylim(0.0, max(f_hi, float(f_e) * 3.0, 50.0))

        if tf_ok and ribbon_org is not None and ribbon_fin is not None and ext_tf is not None and ax_t0 is not None and ax_t1 is not None:
            mo = ribbon_org.astype(np.float64, copy=False)
            mf = ribbon_fin.astype(np.float64, copy=False)
            lo = 20.0 * np.log10(mo + 1e-30)
            lf = 20.0 * np.log10(mf + 1e-30)
            vmin_db = float(max(-140.0, float(np.percentile(lo, 2.0))))
            vmax_db = float(min(0.0, float(np.percentile(lf, 99.5))))
            if not (np.isfinite(vmin_db) and np.isfinite(vmax_db) and vmax_db > vmin_db + 1e-3):
                vmin_db, vmax_db = -80.0, 0.0
            try:
                nrm = mcolors.Normalize(vmin=vmin_db, vmax=vmax_db)
            except Exception:
                nrm = None
            _ = ax_t0.imshow(
                lo,
                aspect="auto",
                origin="lower",
                cmap="inferno",
                interpolation="nearest",
                extent=list(ext_tf),
                norm=nrm,
            )
            im1 = ax_t1.imshow(
                lf,
                aspect="auto",
                origin="lower",
                cmap="inferno",
                interpolation="nearest",
                extent=list(ext_tf),
                norm=nrm,
            )
            n_blk = int(len(stack_o))
            if n_t_one > 0 and n_blk > 1:
                for k in range(1, n_blk):
                    xv = float(t0 + float(k) * float(n_t_one) * float(dt_ok))
                    ax_t0.axvline(xv, color="w", lw=0.45, alpha=0.55)
                    ax_t1.axvline(xv, color="w", lw=0.45, alpha=0.55)
            ax_t0.set_title(f"|TF| before — horizontal concat ({n_blk} traces)")
            ax_t1.set_title(f"|TF| after GCV+morph — horizontal concat ({n_blk} traces)")
            for ax_tf in (ax_t0, ax_t1):
                ax_tf.set_xlabel("Time (s), traces L→R in cache order")
                ax_tf.set_ylabel("Freq (Hz)")
            try:
                cbar = figure.colorbar(im1, ax=[ax_t0, ax_t1], shrink=0.82, pad=0.02)
                cbar.ax.set_ylabel("dB")
            except Exception:
                pass

        sub = (
            f"record={irec_lab} · traces={n_tr} · n={n_min} · dt={dt_ok:g}s · f=({f_s:g},{f_e:g})Hz · "
            f"TF ribbon traces={len(stack_o)} · "
            f"SNR_med={sd:.1f}dB RMSE_med={rmd:.3g} CC_med={cd:.3f}"
        )
        if cancel_tf:
            sub += " (TF ribbon may be partial: canceled)"
        elif int(len(stack_o)) < n_tr:
            sub += f" (TF skipped/failed on {n_tr - int(len(stack_o))} traces)"
        figure.suptitle(sub, fontsize=10)

        btn_row = QtWidgets.QHBoxLayout()
        btn_export = QtWidgets.QPushButton("Export PNG...")
        btn_close = QtWidgets.QPushButton("Close")
        btn_row.addStretch(1)
        btn_row.addWidget(btn_export)
        btn_row.addWidget(btn_close)
        btn_close.clicked.connect(dlg.close)
        path_default = Path.cwd() / f"zplotpy_denoise_all_tf_spec_n{n_tr}_irec{irec_lab or 'NA'}.png"

        def _export_png():
            tgt, sf = self._get_save_file_name(
                "Export PNG",
                str(path_default),
                "PNG (*.png);;All files (*)",
                default_suffix=".png",
                parent=dlg,
            )
            if not tgt:
                return
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"This figure includes Axes that are not compatible with tight_layout.*",
                        category=UserWarning,
                    )
                    figure.savefig(str(tgt), dpi=150, bbox_inches="tight")
                self._set_status_text(f"Exported: {tgt}", hold_ms=2600)
            except Exception as exc_sv:
                QtWidgets.QMessageBox.warning(dlg, "Export failed", str(exc_sv))

        btn_export.clicked.connect(_export_png)
        vbox.addWidget(canvas)
        vbox.addWidget(lbl_denoise_ab_metrics)
        vbox.addLayout(btn_row)
        dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.resize(1100, 880)
        dlg.show()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"This figure includes Axes that are not compatible with tight_layout.*",
                    category=UserWarning,
                )
                canvas.draw_idle()
        except Exception:
            pass
        self._floating_dialogs.append(dlg)
        self._debug_log(
            "DENOISE_COMPARE",
            f"all_tf_ribbon n_tr={n_tr} n_min={n_min} tf_used={len(stack_o)} cancel={int(cancel_tf)}",
        )


