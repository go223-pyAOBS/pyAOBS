"""重力工具箱（Qt）：多边形 Talwani、剖面、简化整块模型 FFT —— 与 Tk gravity_ui 语义接近。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_FULL_MODEL_CACHE_VERSION = 6

from matplotlib.figure import Figure

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT

from matplotlib.path import Path as MplPath

from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QMainWindow,
    QCheckBox,
    QFileDialog,
)

from pyAOBS.utils import (
    calculate_density,
    calculate_vp_from_vs_brocher,
    calculate_vs,
)
from pyAOBS.visualization.imodel import GravityCalculator, PropertyCalculator

from pyAOBS.visualization.imodel_gui.talwani_optional import (
    TALWANI_AVAILABLE,
    convert_density_units,
    talwani2d_gravity_multibody,
)
from pyAOBS.visualization.gravity_obs_grid import (
    DEFAULT_GRAVITY_OBS_FILENAME,
    default_observed_gravity_path,
    interpolate_observed_gravity_mgal,
    profile_lon_lat_for_x_obs_km,
    visualization_package_data_dir,
)
from .gravity_plan_map_qt import GravityPlanMapWindow
from .scroll_safe_widgets import ComboBoxNoScrollUnlessFocused
from .ui_chrome import apply_tool_window_chrome
from .styles import hint_label


def _density_method_for_library(density_method: str) -> str:
    dm = str(density_method or "gardner").strip().lower()
    if PropertyCalculator.uses_tomo2d_sediment_density(dm):
        return "gardner"
    return dm


def _uses_tomo2d_sediment_density(method: str) -> bool:
    return PropertyCalculator.uses_tomo2d_sediment_density(str(method).strip().lower())


def _tomo2d_sediment_density_from_vp(vp: float) -> float:
    return float(PropertyCalculator.tomo2d_sediment_density_from_vp(float(vp)))


class GravityToolboxQt(QMainWindow):
    """独立窗口：管理 gravity_bodies + 曲线显示。"""

    def __init__(
        self,
        parent: QWidget,
        *,
        get_ctx: Callable[[], dict[str, Any]],
        log_fn: Callable[[str], None],
        on_obs_settings_changed: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Gravity Simulation")
        self.resize(780, 700)
        self._get_ctx = get_ctx
        self._log_fn = log_fn
        self._on_obs_settings_changed = on_obs_settings_changed

        self._grav_calc = GravityCalculator()
        self._full_model_gravity_cache: Optional[Dict[str, Any]] = None
        self._bodies: List[Tuple[np.ndarray, np.ndarray, float]] = []
        self._plan_map_win: Optional[GravityPlanMapWindow] = None
        self._overlay_default_applied = False
        self._fig = Figure(figsize=(8, 5))
        self._ax = self._fig.add_subplot(111)

        cw = QWidget()
        self.setCentralWidget(cw)
        main = QVBoxLayout(cw)

        opts = QGroupBox("Parameters")
        form = QFormLayout(opts)
        self._bg = QDoubleSpinBox()
        self._bg.setRange(0.5, 5.0)
        self._bg.setDecimals(3)
        self._bg.setValue(2.67)
        self._obs = QDoubleSpinBox()
        self._obs.setRange(-200.0, 200.0)
        self._obs.setDecimals(2)
        self._obs.setValue(0.0)
        self._max_n = QSpinBox()
        self._max_n.setRange(10, 500)
        self._max_n.setValue(100)
        self._extend = QDoubleSpinBox()
        self._extend.setRange(0.0, 200.0)
        self._extend.setValue(40.0)
        self._method = ComboBoxNoScrollUnlessFocused()
        self._method.addItems(["tomo2d_fft", "talwani_grid"])
        self._strict = QCheckBox("Strict jgrav mode")
        form.addRow("Background density (g/cm³)", self._bg)
        form.addRow("Observation level (km, +down)", self._obs)
        form.addRow("Max grid size", self._max_n)
        form.addRow("Extension distance (km)", self._extend)
        form.addRow("Gravity method (full model)", self._method)
        form.addRow("", self._strict)
        main.addWidget(opts)

        obs_g = QGroupBox("Observed gravity (world .grd)")
        obs_form = QFormLayout(obs_g)
        dw = QWidget()
        dhl = QHBoxLayout(dw)
        dhl.setContentsMargins(0, 0, 0, 0)
        self._grav_data_dir_edit = QLineEdit()
        btn_br = QPushButton("Browse…")
        btn_br.clicked.connect(self._browse_gravity_data_dir)
        dhl.addWidget(self._grav_data_dir_edit)
        dhl.addWidget(btn_br)
        obs_form.addRow("Data directory", dw)
        self._grav_fn_edit = QLineEdit(DEFAULT_GRAVITY_OBS_FILENAME)
        obs_form.addRow("Grid filename", self._grav_fn_edit)
        self._overlay_world_chk = QCheckBox(
            "Compare: overlay observed gravity (world .grd) on theoretical profile"
        )
        self._overlay_world_chk.setToolTip(
            "勾选后，在 Full Model / Bodies 剖面图上用橙色方点线叠加 world .grd 观测异常；"
            "需放置 gravity_world.grd 并填写 Profile Lon/Lat（或模型 x 轴含经纬度）。"
        )
        obs_form.addRow("", self._overlay_world_chk)
        self._overlay_hint = hint_label(
            "理论曲线：蓝线 (Full Model / Talwani bodies)；观测：橙线 (world .grd 插值)。"
        )
        obs_form.addRow("", self._overlay_hint)
        self._grav_ll_edit = QLineEdit()
        self._grav_ll_edit.setPlaceholderText("lon₀ lat₀ lon₁ lat₁ 或 lonₘᵢₙ lonₘₐₓ latₘᵢₙ latₘₐₓ")
        obs_form.addRow("Profile Lon/Lat", self._grav_ll_edit)
        main.addWidget(obs_g)

        for w in (
            self._grav_data_dir_edit,
            self._grav_fn_edit,
            self._grav_ll_edit,
        ):
            w.editingFinished.connect(self._emit_obs_settings)
        self._overlay_world_chk.stateChanged.connect(lambda *_: self._emit_obs_settings())

        row = QHBoxLayout()
        for text, slot in (
            ("Show / refresh plot", self._plot_empty_message),
            ("Add Polygon", self._add_polygon_body),
            ("Calculate Profile", self._calculate_profile),
            ("Full Model", self._full_model),
            ("Compare Methods", self._compare_full_model_methods),
            ("平面图 lon×lat", self._open_plan_map),
            ("Clear Bodies", self._clear_bodies),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)  # type: ignore[arg-type]
            row.addWidget(b)
        main.addLayout(row)

        note = QLabel(
            "Full Model 与 Tk gravity_ui 对齐（延伸区 taper / strict / 下采样等）；"
            "可将全球格网重力放入 Data directory 下的 Grid filename（默认 visualization/data/"
            + DEFAULT_GRAVITY_OBS_FILENAME
            + "）；「平面图 lon×lat」在世界格网上绘制剖面轨迹与观测站位。"
        )
        note.setWordWrap(True)

        canvas = FigureCanvasQTAgg(self._fig)
        toolbar = NavigationToolbar2QT(canvas, cw)
        main.addWidget(toolbar)
        main.addWidget(canvas, stretch=1)
        main.addWidget(note)
        self._canvas = canvas
        self._plot_empty_message()
        self.refresh_obs_widgets_from_ctx()
        apply_tool_window_chrome(self)

    def _maybe_enable_overlay_default(self) -> None:
        """首次打开且 world .grd 存在时默认勾选观测叠加，便于理论/观测对比。"""
        if getattr(self, "_overlay_default_applied", False):
            return
        self._overlay_default_applied = True
        if self._overlay_world_chk.isChecked():
            return
        if self.gravity_world_grd_filepath().is_file():
            self._overlay_world_chk.setChecked(True)
            self._emit_obs_settings()
            self._log_fn(
                "\nGravity toolbox: auto-enabled observed overlay (world .grd found). "
                "Fill Profile Lon/Lat if overlay fails."
            )

    def gravity_world_grd_filepath(self) -> Path:
        """当前 UI 所指世界重力格网路径。"""
        dd = self._grav_data_dir_edit.text().strip() or str(visualization_package_data_dir())
        fn = self._grav_fn_edit.text().strip() or DEFAULT_GRAVITY_OBS_FILENAME
        return default_observed_gravity_path(directory=dd, filename=fn)

    def _open_plan_map(self) -> None:
        c = self._ctx()
        if c.get("grid_data") is None or c.get("profile_extractor") is None:
            QMessageBox.warning(self, "Warning", "请先加载模型。")
            return
        fp = self.gravity_world_grd_filepath()
        if not fp.is_file():
            QMessageBox.warning(self, "Warning", f"未找到重力平面图数据文件:\n{fp}")
            return
        if self._plan_map_win is None:
            self._plan_map_win = GravityPlanMapWindow(
                self,
                get_ctx=lambda: self._ctx(),
                get_gravity_filepath=self.gravity_world_grd_filepath,
                get_profile_endpoint_line=lambda: self._grav_ll_edit.text(),
                log_fn=self._log_fn,
            )
        self._plan_map_win.redraw_plan_map()
        self._plan_map_win.show()
        self._plan_map_win.raise_()

    def _emit_obs_settings(self) -> None:
        if self._on_obs_settings_changed is None:
            return
        ddir = self._grav_data_dir_edit.text().strip()
        if not ddir:
            ddir = str(visualization_package_data_dir())
        self._on_obs_settings_changed(
            {
                "gravity_obs_data_dir": ddir,
                "gravity_obs_filename": self._grav_fn_edit.text().strip()
                or DEFAULT_GRAVITY_OBS_FILENAME,
                "gravity_obs_overlay": bool(self._overlay_world_chk.isChecked()),
                "gravity_profile_lon_lat_csv": self._grav_ll_edit.text().strip(),
            }
        )

    def _browse_gravity_data_dir(self) -> None:
        start = self._grav_data_dir_edit.text().strip() or str(visualization_package_data_dir())
        path = QFileDialog.getExistingDirectory(self, "Select gravity grid directory", start)
        if path:
            self._grav_data_dir_edit.setText(path)
            self._emit_obs_settings()

    def refresh_obs_widgets_from_ctx(self) -> None:
        c = self._ctx()
        dd = str(c.get("gravity_obs_data_dir") or "").strip()
        self._grav_data_dir_edit.setText(dd if dd else str(visualization_package_data_dir()))
        self._grav_fn_edit.setText(
            str(c.get("gravity_obs_filename") or DEFAULT_GRAVITY_OBS_FILENAME).strip()
        )
        self._overlay_world_chk.setChecked(bool(c.get("gravity_obs_overlay", False)))
        self._grav_ll_edit.setText(str(c.get("gravity_profile_lon_lat_csv") or ""))

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self.refresh_obs_widgets_from_ctx()
        self._maybe_enable_overlay_default()

    @property
    def bodies(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        return self._bodies

    def _plot_empty_message(self) -> None:
        self._ax.clear()
        n = len(self._bodies)
        self._ax.text(
            0.5,
            0.5,
            f"{n} gravity body/bodies.\nCalculate profile after adding polygons or run Full Model.",
            ha="center",
            va="center",
            transform=self._ax.transAxes,
            fontsize=11,
        )
        self._ax.set_xlabel("Distance (km)")
        self._ax.set_ylabel("Gravity anomaly (mGal)")
        self._ax.set_title("Gravity")
        self._ax.grid(True, alpha=0.3)
        self._canvas.draw()

    def _ctx(self) -> dict[str, Any]:
        return self._get_ctx()

    @staticmethod
    def _build_full_model_cache_key(ctx: dict[str, Any]) -> str:
        """与 imodel_gui/gravity_ui._build_full_model_cache_key 一致。"""
        file_key = str(ctx.get("current_model_file") or "").strip()
        if file_key:
            try:
                p = Path(file_key)
                st = p.stat()
                return f"file:{file_key}:{int(st.st_mtime_ns)}:{int(st.st_size)}"
            except Exception:
                return f"file:{file_key}"
        ds = ctx.get("grid_data")
        if ds is None:
            return "none"
        dims_key = "|".join(f"{k}:{int(v)}" for k, v in sorted(ds.sizes.items()))
        var_key = ",".join(sorted(list(ds.data_vars)))
        return f"inmem:{id(ds)}:{dims_key}:{var_key}"

    @staticmethod
    def _build_full_model_param_signature(
        *,
        background_density: float,
        obs_level: float,
        max_grid_size: int,
        extension_dist: float,
        density_method: str,
        gravity_method: str,
        strict_jgrav: bool,
    ) -> str:
        payload = {
            "background_density": float(background_density),
            "obs_level": float(obs_level),
            "max_grid_size": int(max_grid_size),
            "extension_dist": float(extension_dist),
            "density_method": str(density_method or ""),
            "gravity_method": str(gravity_method or ""),
            "strict_jgrav": bool(strict_jgrav),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    def clear_full_model_cache(self) -> None:
        """加载新模型时由主窗口调用，与 Tk 中重置 _full_model_gravity_cache 一致。"""
        self._full_model_gravity_cache = None

    def _sample_world_observed_overlay(
        self, x_obs_km: np.ndarray
    ) -> tuple[Optional[np.ndarray], str]:
        """从 world .grd 在观测 x 位置上插值；返回 (数组或 None, 说明/失败原因)。"""
        if not self._overlay_world_chk.isChecked():
            return None, "overlay disabled"
        dd = self._grav_data_dir_edit.text().strip() or str(visualization_package_data_dir())
        fn = self._grav_fn_edit.text().strip() or DEFAULT_GRAVITY_OBS_FILENAME
        fp = default_observed_gravity_path(directory=dd, filename=fn)
        c = self._ctx()
        gd = c.get("grid_data")
        pe = c.get("profile_extractor")
        if gd is None or pe is None:
            return None, "no velocity grid / profile extractor"
        x_coord = pe.x_coord
        try:
            x_prof_km = np.asarray(gd.coords[x_coord].values, dtype=float)
        except KeyError:
            x_prof_km = np.asarray(gd[x_coord].values, dtype=float)

        ep = self._grav_ll_edit.text().strip()

        lon, lat, geo_note = profile_lon_lat_for_x_obs_km(
            x_obs_km=np.asarray(x_obs_km, dtype=float),
            x_grid_km=x_prof_km,
            grid_data=gd,
            x_coord=x_coord,
            endpoint_line=ep or None,
        )
        if lon is None or lat is None:
            self._log_fn(f"\nObserved gravity overlay skipped — {geo_note}")
            return None, geo_note
        try:
            vals, dinfo = interpolate_observed_gravity_mgal(fp, lon, lat)
        except ImportError as e:
            self._log_fn(f"\nObserved gravity overlay skipped ({e}).")
            return None, str(e)
        except OSError as e:
            self._log_fn(f"\nObserved gravity overlay skipped ({e}). Expected file: {fp}")
            return None, str(e)
        except Exception as e:
            self._log_fn(f"\nObserved gravity overlay failed: {e}")
            return None, str(e)

        self._log_fn(
            f"\nObserved gravity overlay: source={fp} — {geo_note}; {dinfo} "
            f"range [{float(np.nanmin(vals)):.2f},{float(np.nanmax(vals)):.2f}]"
        )
        return vals.astype(float), geo_note

    def _append_observed_world_line(
        self,
        x_obs_km: np.ndarray,
        *,
        precomputed: Optional[np.ndarray] = None,
        precomputed_note: str = "",
    ) -> bool:
        """在当前 axes 上绘制世界格网观测曲线；成功返回 True。"""
        xk = np.asarray(x_obs_km, dtype=float)
        if precomputed is not None:
            obs = precomputed
            note = precomputed_note
        else:
            obs, note = self._sample_world_observed_overlay(xk)
        if obs is None:
            if self._overlay_world_chk.isChecked() and note and note != "overlay disabled":
                self._ax.text(
                    0.02,
                    0.02,
                    f"Observed overlay skipped:\n{note}",
                    transform=self._ax.transAxes,
                    fontsize=8,
                    color="darkred",
                    va="bottom",
                    ha="left",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
                )
            return False
        self._ax.plot(
            xk,
            obs,
            color="darkorange",
            marker="s",
            linestyle="-",
            ms=5,
            lw=1.2,
            alpha=0.92,
            label="observed (world .grd)",
            zorder=5,
        )
        return True

    def _render_full_model_gravity_result(
        self,
        *,
        x_obs_km: np.ndarray,
        gravity_anomaly: np.ndarray,
        obs_level: float,
        gravity_method: str,
        strict_jgrav: bool,
        from_cache: bool = False,
        anomaly_note: str = "",
        model_shape: tuple[int, int] | None = None,
    ) -> None:
        """绘制 Full Model 曲线并写日志（与 gravity_ui._render_full_model_gravity_result 对齐）。"""
        x_obs_km = np.asarray(x_obs_km, dtype=float)
        gravity_anomaly = np.asarray(gravity_anomaly, dtype=float)
        legend = f"full model ({gravity_method})"
        if anomaly_note:
            legend += f", {anomaly_note}"
        cache_suffix = " [cached]" if from_cache else ""
        obs, obs_note = self._sample_world_observed_overlay(x_obs_km)

        self._ax.clear()
        self._ax.plot(
            x_obs_km,
            gravity_anomaly,
            "b-",
            lw=2,
            label=f"theoretical {legend}",
            zorder=4,
        )
        self._append_observed_world_line(
            x_obs_km, precomputed=obs, precomputed_note=obs_note
        )
        self._ax.set_title(
            f"Full model gravity (z_obs={obs_level:.2f} km, strict={'ON' if strict_jgrav else 'OFF'}){cache_suffix}"
        )
        self._ax.set_xlabel("Distance (km)")
        self._ax.set_ylabel("Gravity anomaly (mGal)")
        self._ax.grid(True, alpha=0.3)
        self._ax.axhline(0, color="k", ls="--", alpha=0.5)
        self._ax.legend()
        self._canvas.draw()

        self._log_fn("\n" + "=" * 50)
        if from_cache:
            self._log_fn("Reusing cached gravity anomaly from full model (model unchanged).")
        else:
            self._log_fn("Gravity anomaly from full model calculated:")
        if model_shape is not None:
            self._log_fn(f"  Model grid: {model_shape[0]} x {model_shape[1]}")
        self._log_fn(f"  Gravity method: {gravity_method}")
        self._log_fn(f"  Number of observation points: {len(x_obs_km)}")
        self._log_fn(
            f"  Gravity range: {float(np.nanmin(gravity_anomaly)):.2f} to {float(np.nanmax(gravity_anomaly)):.2f} mGal"
        )
        self._log_fn(f"  Mean gravity: {float(np.nanmean(gravity_anomaly)):.2f} mGal")
        self._log_fn(f"  Std gravity: {float(np.nanstd(gravity_anomaly)):.2f} mGal")
        if anomaly_note:
            self._log_fn(f"  Anomaly display mode: {anomaly_note}")

    @staticmethod
    def _shared_reference_alignment(
        x_obs_km: np.ndarray,
        curve_a: np.ndarray,
        curve_b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """与 gravity_ui._shared_reference_alignment 一致：共享边缘线性基线去除。"""
        x = np.asarray(x_obs_km, dtype=float)
        a = np.asarray(curve_a, dtype=float).copy()
        b = np.asarray(curve_b, dtype=float).copy()
        n = int(min(len(x), len(a), len(b)))
        if n < 6:
            return a, b, "no-alignment (insufficient points)"

        edge_n = max(3, min(12, n // 8))
        t = np.linspace(0.0, 1.0, num=n, dtype=float)

        def _align_one(y: np.ndarray) -> np.ndarray:
            y2 = np.asarray(y[:n], dtype=float).copy()
            left_mean = float(np.nanmean(y2[:edge_n]))
            right_mean = float(np.nanmean(y2[-edge_n:]))
            baseline = left_mean + (right_mean - left_mean) * t
            return y2 - baseline

        a_out = _align_one(a)
        b_out = _align_one(b)
        note = f"shared-edge linear baseline removed (edge_n={edge_n})"
        return a_out, b_out, note

    def _render_full_model_gravity_compare_result(
        self,
        *,
        x_obs_km: np.ndarray,
        gravity_tomo2d: np.ndarray,
        gravity_talwani: np.ndarray,
        gravity_tomo2d_aligned: np.ndarray,
        gravity_talwani_aligned: np.ndarray,
        obs_level: float,
        strict_jgrav: bool,
        alignment_note: str = "",
    ) -> None:
        x_obs_km = np.asarray(x_obs_km, dtype=float)
        g_tomo = np.asarray(gravity_tomo2d, dtype=float)
        g_tal = np.asarray(gravity_talwani, dtype=float)
        g_tomo_a = np.asarray(gravity_tomo2d_aligned, dtype=float)
        g_tal_a = np.asarray(gravity_talwani_aligned, dtype=float)
        diff = g_tomo_a - g_tal_a
        obs, obs_note = self._sample_world_observed_overlay(x_obs_km)

        self._ax.clear()
        self._ax.plot(x_obs_km, g_tomo_a, "b-", lw=2, label="tomo2d_fft (aligned)")
        self._ax.plot(
            x_obs_km, g_tal_a, "r-", lw=2, alpha=0.85, label="talwani_grid (aligned)"
        )
        self._ax.plot(
            x_obs_km, diff, "k--", lw=1.3, alpha=0.9, label="difference (aligned)"
        )
        self._append_observed_world_line(
            x_obs_km, precomputed=obs, precomputed_note=obs_note
        )
        self._ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.8)
        self._ax.set_xlabel("Distance (km)", fontsize=12)
        self._ax.set_ylabel("Gravity Anomaly (mGal)", fontsize=12)
        title_suffix = f" | {alignment_note}" if alignment_note else ""
        self._ax.set_title(
            f"Full-Model Gravity Method Comparison (z_obs={obs_level:.2f} km){title_suffix}",
            fontsize=12,
            fontweight="bold",
        )
        self._ax.grid(True, alpha=0.3)
        self._ax.legend(fontsize=9)
        self._canvas.draw()

        valid = np.isfinite(g_tomo) & np.isfinite(g_tal)
        raw_diff = g_tomo - g_tal
        rmse_raw = float(np.sqrt(np.mean((raw_diff[valid]) ** 2))) if np.any(valid) else float("nan")
        max_abs_raw = float(np.nanmax(np.abs(raw_diff[valid]))) if np.any(valid) else float("nan")
        corr_raw = (
            float(np.corrcoef(g_tomo[valid], g_tal[valid])[0, 1]) if np.sum(valid) > 2 else float("nan")
        )

        valid_a = np.isfinite(g_tomo_a) & np.isfinite(g_tal_a)
        rmse = float(np.sqrt(np.mean((diff[valid_a]) ** 2))) if np.any(valid_a) else float("nan")
        max_abs = float(np.nanmax(np.abs(diff[valid_a]))) if np.any(valid_a) else float("nan")
        corr = (
            float(np.corrcoef(g_tomo_a[valid_a], g_tal_a[valid_a])[0, 1])
            if np.sum(valid_a) > 2
            else float("nan")
        )
        self._log_fn("\n" + "=" * 50)
        self._log_fn("Compared full-model gravity methods:")
        self._log_fn(f"  Strict jgrav mode: {'ON' if strict_jgrav else 'OFF'}")
        self._log_fn(f"  Observation points: {len(x_obs_km)}")
        self._log_fn(f"  Raw tomo2d range: {np.nanmin(g_tomo):.2f} to {np.nanmax(g_tomo):.2f} mGal")
        self._log_fn(f"  Raw talwani range: {np.nanmin(g_tal):.2f} to {np.nanmax(g_tal):.2f} mGal")
        self._log_fn(f"  Raw difference RMSE: {rmse_raw:.2f} mGal")
        self._log_fn(f"  Raw difference max|.|: {max_abs_raw:.2f} mGal")
        self._log_fn(f"  Raw curve correlation: {corr_raw:.4f}")
        if alignment_note:
            self._log_fn(f"  Alignment: {alignment_note}")
        self._log_fn(f"  Aligned difference RMSE: {rmse:.2f} mGal")
        self._log_fn(f"  Aligned difference max|.|: {max_abs:.2f} mGal")
        self._log_fn(f"  Aligned curve correlation: {corr:.4f}")

    def _run_synthetic_gravity_benchmark(self, strict_jgrav: bool) -> None:
        """与 gravity_ui._run_synthetic_gravity_benchmark 一致。"""
        try:
            x_coords = np.linspace(0.0, 200.0, 81)
            z_coords = np.linspace(0.0, 60.0, 61)
            dens = np.zeros((len(z_coords), len(x_coords)), dtype=float)

            rect_mask = (
                (x_coords[np.newaxis, :] >= 70.0)
                & (x_coords[np.newaxis, :] <= 130.0)
                & (z_coords[:, np.newaxis] >= 10.0)
                & (z_coords[:, np.newaxis] <= 35.0)
            )
            dens[rect_mask] = 320.0
            sig = np.abs(dens) > 1e-9

            x_obs = np.linspace(0.5, 199.5, 100)

            g_tomo = self._grav_calc.calculate_tomo2d_fft_anomaly(
                x_coords_km=x_coords,
                z_coords_km=z_coords,
                density_contrast_kg_m3=dens,
                significant_mask=sig,
                x_obs_km=x_obs,
                obs_level_km=0.0,
                ref_x_range_km=(float(x_obs[0]), float(x_obs[-1])),
            )
            g_tal = self._grav_calc.calculate_talwani_grid_anomaly(
                x_coords_km=x_coords,
                z_coords_km=z_coords,
                density_contrast_kg_m3=dens,
                significant_mask=sig,
                x_obs_km=x_obs,
                obs_level_km=0.0,
            )

            valid = np.isfinite(g_tomo) & np.isfinite(g_tal)
            diff_raw = g_tomo - g_tal
            rmse_raw = float(np.sqrt(np.mean((diff_raw[valid]) ** 2))) if np.any(valid) else float("nan")
            corr_raw = (
                float(np.corrcoef(g_tomo[valid], g_tal[valid])[0, 1]) if np.sum(valid) > 2 else float("nan")
            )

            g_tomo_a, g_tal_a, note = self._shared_reference_alignment(x_obs, g_tomo, g_tal)
            valid_a = np.isfinite(g_tomo_a) & np.isfinite(g_tal_a)
            diff_a = g_tomo_a - g_tal_a
            rmse_a = float(np.sqrt(np.mean((diff_a[valid_a]) ** 2))) if np.any(valid_a) else float("nan")
            corr_a = (
                float(np.corrcoef(g_tomo_a[valid_a], g_tal_a[valid_a])[0, 1])
                if np.sum(valid_a) > 2
                else float("nan")
            )

            self._log_fn("  Synthetic benchmark (single rectangle, +320 kg/m³):")
            self._log_fn(f"    Strict jgrav mode: {'ON' if strict_jgrav else 'OFF'}")
            self._log_fn(f"    Raw RMSE: {rmse_raw:.2f} mGal, Raw corr: {corr_raw:.4f}")
            self._log_fn(f"    Aligned RMSE: {rmse_a:.2f} mGal, Aligned corr: {corr_a:.4f}")
            self._log_fn(f"    Alignment used: {note}")
        except Exception as exc:
            self._log_fn(f"  Synthetic benchmark failed: {exc}")

    def _compare_full_model_methods(self) -> None:
        """依次计算 tomo2d_fft 与 talwani_grid 并绘制对比曲线（对齐 Tk Compare Methods）。"""
        if not TALWANI_AVAILABLE:
            QMessageBox.critical(self, "Error", "Talwani gravity module not available.")
            return
        c = self._ctx()
        if c.get("grid_data") is None:
            QMessageBox.warning(self, "Warning", "No model loaded. Please open a model first.")
            return

        strict_jgrav = bool(self._strict.isChecked())
        restore_idx = self._method.currentIndex()
        cache_tomo: Optional[Dict[str, Any]] = None
        cache_tal: Optional[Dict[str, Any]] = None

        try:
            self._log_fn("\n" + "=" * 50)
            self._log_fn("Starting full-model method comparison...")
            self._log_fn("  Step 1/2: tomo2d_fft")
            i_fft = self._method.findText("tomo2d_fft")
            self._method.setCurrentIndex(i_fft if i_fft >= 0 else 0)
            self._full_model()
            if isinstance(self._full_model_gravity_cache, dict):
                cache_tomo = dict(self._full_model_gravity_cache)

            self._log_fn("  Step 2/2: talwani_grid")
            i_tal = self._method.findText("talwani_grid")
            self._method.setCurrentIndex(i_tal if i_tal >= 0 else 1)
            self._full_model()
            if isinstance(self._full_model_gravity_cache, dict):
                cache_tal = dict(self._full_model_gravity_cache)
        finally:
            self._method.setCurrentIndex(restore_idx)

        if not isinstance(cache_tomo, dict) or not isinstance(cache_tal, dict):
            QMessageBox.critical(self, "Error", "Failed to collect results for method comparison.")
            return

        x_tomo = np.asarray(cache_tomo.get("x_obs_km", []), dtype=float)
        g_tomo = np.asarray(cache_tomo.get("gravity_anomaly", []), dtype=float)
        x_tal = np.asarray(cache_tal.get("x_obs_km", []), dtype=float)
        g_tal = np.asarray(cache_tal.get("gravity_anomaly", []), dtype=float)
        obs_level = float(cache_tomo.get("obs_level", cache_tal.get("obs_level", 0.0)))

        if x_tomo.size < 2 or g_tomo.size != x_tomo.size or x_tal.size < 2 or g_tal.size != x_tal.size:
            QMessageBox.critical(self, "Error", "Comparison data is invalid or incomplete.")
            return

        if x_tal.size != x_tomo.size or not np.allclose(x_tal, x_tomo, atol=1e-8, rtol=1e-6):
            g_tal = np.interp(x_tomo, x_tal, g_tal, left=np.nan, right=np.nan)

        g_tomo_aligned, g_tal_aligned, alignment_note = self._shared_reference_alignment(
            x_tomo, g_tomo, g_tal
        )

        self._render_full_model_gravity_compare_result(
            x_obs_km=x_tomo,
            gravity_tomo2d=g_tomo,
            gravity_talwani=g_tal,
            gravity_tomo2d_aligned=g_tomo_aligned,
            gravity_talwani_aligned=g_tal_aligned,
            obs_level=obs_level,
            strict_jgrav=strict_jgrav,
            alignment_note=alignment_note,
        )
        self._run_synthetic_gravity_benchmark(strict_jgrav=strict_jgrav)

    def _add_polygon_body(self) -> None:
        if not TALWANI_AVAILABLE:
            QMessageBox.critical(self, "Error", "Talwani module not available.")
            return
        c = self._ctx()
        polys = c.get("selected_polygons") or []
        if not polys:
            QMessageBox.warning(self, "Warning", "No polygon. Enable polygon selection on main window first.")
            return
        grid_data = c.get("grid_data")
        pe = c.get("profile_extractor")
        pc = c.get("property_calculator")
        if grid_data is None or pe is None or pc is None:
            QMessageBox.warning(self, "Warning", "Load model first.")
            return
        polygon = polys[-1]
        if len(polygon) < 3:
            QMessageBox.warning(self, "Warning", "Polygon needs ≥3 vertices.")
            return

        velocity_method = str(c.get("velocity_method") or "brocher").strip().lower()
        density_method = str(c.get("density_method") or "gardner").strip().lower()
        model_type = str(c.get("model_type") or "vp").strip().lower()

        try:
            x_coord = pe.x_coord
            z_coord = pe.z_coord
            velocity_var = pc.velocity_var
            poly_x = [p[0] for p in polygon]
            poly_z = [p[1] for p in polygon]
            x_min, x_max = min(poly_x), max(poly_x)
            z_min, z_max = min(poly_z), max(poly_z)
            poly_path = MplPath(polygon)
            x_coords = grid_data[x_coord].values
            z_coords = grid_data[z_coord].values
            velocities: list[float] = []
            for x in x_coords:
                if x_min <= x <= x_max:
                    for z in z_coords:
                        if z_min <= z <= z_max:
                            if poly_path.contains_point((float(x), float(z))):
                                try:
                                    vel = float(
                                        grid_data[velocity_var].sel(
                                            {x_coord: x, z_coord: z}, method="nearest"
                                        ).values
                                    )
                                    velocities.append(vel)
                                except Exception:
                                    pass
            if not velocities:
                QMessageBox.warning(self, "Warning", "Could not sample velocities inside polygon.")
                return
            avg_velocity = float(np.mean(velocities))
            if model_type == "vp":
                vp = avg_velocity
                vs = calculate_vs(vp, method=velocity_method)
            else:
                vs = avg_velocity
                vp = float(calculate_vp_from_vs_brocher(vs))

            if _uses_tomo2d_sediment_density(density_method):
                avg_density = _tomo2d_sediment_density_from_vp(vp)
            else:
                avg_density = calculate_density(vp, method=_density_method_for_library(density_method))

            bg_density = float(self._bg.value())
            density_contrast = avg_density - bg_density
            density_contrast_kg_m3 = convert_density_units(density_contrast, "g/cm3", "kg/m3")
            poly_x_m = np.array(poly_x) * 1000.0
            poly_z_m = np.array(poly_z) * 1000.0
            self._bodies.append((poly_x_m, poly_z_m, float(density_contrast_kg_m3)))
            self._log_fn(
                f"\nAdded gravity body (#{len(self._bodies)}): Vp≈{vp:.2f} Vs≈{vs:.2f} "
                f"rho≈{avg_density:.3f} g/cm³, contrast≈{density_contrast:.3f} g/cm³"
            )
            self._plot_empty_message()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _calculate_profile(self) -> None:
        if not TALWANI_AVAILABLE:
            QMessageBox.critical(self, "Error", "Talwani module not available.")
            return
        if not self._bodies:
            QMessageBox.warning(self, "Warning", "No gravity bodies.")
            return
        c = self._ctx()
        grid_data = c.get("grid_data")
        pe = c.get("profile_extractor")
        if grid_data is None or pe is None:
            return
        try:
            obs_level = float(self._obs.value())
            obs_level_m = obs_level * 1000.0
            x_coord = pe.x_coord
            x_coords = grid_data[x_coord].values
            x_min, x_max = x_coords.min(), x_coords.max()
            n_obs_profile = 200
            x_obs_edges = np.linspace(x_min, x_max, n_obs_profile + 1)
            x_obs_km = 0.5 * (x_obs_edges[:-1] + x_obs_edges[1:])
            x_obs_m = x_obs_km * 1000.0

            grav = np.zeros_like(x_obs_m)
            bodies_list = [(x, z, rho) for x, z, rho in self._bodies]
            for i, x0_m in enumerate(x_obs_m):
                try:
                    grav[i] = talwani2d_gravity_multibody(
                        bodies_list, float(x0_m), obs_level_m, check_vertex=False
                    )
                except Exception:
                    grav[i] = np.nan

            self._ax.clear()
            self._ax.plot(
                x_obs_km,
                grav,
                "b-",
                lw=2,
                label="theoretical Talwani (bodies)",
                zorder=4,
            )
            self._append_observed_world_line(np.asarray(x_obs_km, dtype=float))
            self._ax.set_xlabel("Distance (km)")
            self._ax.set_ylabel("Gravity anomaly (mGal)")
            self._ax.set_title(f"Bodies profile (obs z={obs_level:.2f} km)")
            self._ax.axhline(0, color="k", ls="--", alpha=0.5)
            self._ax.grid(True, alpha=0.3)
            self._ax.legend()
            self._canvas.draw()
            self._log_fn(
                f"\nGravity profile: bodies={len(self._bodies)}, stations={len(x_obs_km)}, "
                f"range [{np.nanmin(grav):.2f},{np.nanmax(grav):.2f}] mGal"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _full_model(self) -> None:
        c = self._ctx()
        grid_data = c.get("grid_data")
        pe = c.get("profile_extractor")
        pc = c.get("property_calculator")
        density_method = str(c.get("density_method") or "gardner").strip().lower()
        model_type = str(c.get("model_type") or "vp").strip().lower()
        seafloor_fn = c.get("seafloor_depths_fn")
        basement_fn = c.get("basement_depths_fn")

        if grid_data is None or pe is None or pc is None:
            QMessageBox.warning(self, "Warning", "Load model first.")
            return
        gm = self._method.currentText().strip().lower()
        if gm == "talwani_grid" and not TALWANI_AVAILABLE:
            QMessageBox.critical(self, "Error", "Talwani not available.")
            return

        strict_jgrav = bool(self._strict.isChecked())

        try:
            bg_density = float(self._bg.value())
            obs_level = float(self._obs.value())
            max_gs = int(self._max_n.value())
            if max_gs < 10:
                max_gs = 10
                self._max_n.setValue(10)
            elif max_gs > 500:
                max_gs = 500
                self._max_n.setValue(500)

            extension_dist = float(self._extend.value())
            if extension_dist < 0.0:
                extension_dist = 0.0
                self._extend.setValue(0.0)
            elif extension_dist > 200.0:
                extension_dist = 200.0
                self._extend.setValue(200.0)

            param_signature = self._build_full_model_param_signature(
                background_density=bg_density,
                obs_level=obs_level,
                max_grid_size=max_gs,
                extension_dist=extension_dist,
                density_method=density_method,
                gravity_method=gm,
                strict_jgrav=strict_jgrav,
            )
            model_key = self._build_full_model_cache_key(c)
            cache = self._full_model_gravity_cache if isinstance(self._full_model_gravity_cache, dict) else None
            if (
                cache is not None
                and cache.get("model_key") == model_key
                and cache.get("param_signature") == param_signature
                and int(cache.get("cache_version", 0)) == _FULL_MODEL_CACHE_VERSION
            ):
                x_obs_km_cached = np.asarray(cache.get("x_obs_km", []), dtype=float)
                gravity_cached = np.asarray(cache.get("gravity_anomaly", []), dtype=float)
                obs_level_cached = float(cache.get("obs_level", 0.0))
                if x_obs_km_cached.size > 1 and gravity_cached.size == x_obs_km_cached.size:
                    self._render_full_model_gravity_result(
                        x_obs_km=x_obs_km_cached,
                        gravity_anomaly=gravity_cached,
                        obs_level=obs_level_cached,
                        gravity_method=str(cache.get("gravity_method", gm) or gm),
                        strict_jgrav=bool(cache.get("strict_jgrav", strict_jgrav)),
                        from_cache=True,
                        anomaly_note=str(cache.get("anomaly_note", "") or ""),
                        model_shape=cache.get("model_shape"),
                    )
                    return

            x_coord = pe.x_coord
            z_coord = pe.z_coord
            vel_var = pc.velocity_var
            x_coords = np.asarray(grid_data[x_coord].values, dtype=float)
            z_coords = np.asarray(grid_data[z_coord].values, dtype=float)
            velocity_data = np.asarray(grid_data[vel_var].values, dtype=float)

            nz, nx = velocity_data.shape
            if nz < 2 or nx < 2:
                QMessageBox.warning(self, "Warning", "Model too small.")
                return

            if str(model_type).strip().lower() == "vs":
                vs_grid = velocity_data.astype(float)
                velocity_data = np.vectorize(lambda v: float(calculate_vp_from_vs_brocher(float(v))))(vs_grid)
            else:
                velocity_data = velocity_data.astype(float)

            orig_x_min0 = float(x_coords[0])
            orig_x_max0 = float(x_coords[-1])
            orig_z_max0 = float(z_coords[-1])
            nx_orig = len(x_coords)
            nz_orig = len(z_coords)
            n_extend_x = 0

            self._log_fn("\n" + "=" * 50)
            self._log_fn("Calculating gravity anomaly from full model...")
            self._log_fn(f"  Original model size: {velocity_data.shape}")
            self._log_fn(f"  Background density: {bg_density:.3f} g/cm³")
            self._log_fn(f"  Observation level: {obs_level:.2f} km")
            self._log_fn(f"  Density method: {density_method}")
            self._log_fn(f"  Gravity method: {gm}")
            self._log_fn(f"  Strict jgrav mode: {'ON' if strict_jgrav else 'OFF'}")
            self._log_fn(f"  Max grid size: {max_gs}")
            self._log_fn(f"  Extension distance: {extension_dist:.1f} km")

            if extension_dist > 0.0:
                self._log_fn(f"  Extending model by {extension_dist:.1f} km on each side...")
                if len(x_coords) > 1:
                    dx_original = float(x_coords[1] - x_coords[0])
                else:
                    dx_original = 1.0
                if len(z_coords) > 1:
                    dz_original = float(z_coords[1] - z_coords[0])
                else:
                    dz_original = 0.5

                n_extend_x = int(np.ceil(extension_dist / dx_original))
                n_extend_z = int(np.ceil(extension_dist / dz_original))

                x_min = float(x_coords[0])
                x_max = float(x_coords[-1])
                z_max = float(z_coords[-1])

                x_left_extend = np.linspace(x_min - extension_dist, x_min - dx_original, n_extend_x)
                x_right_extend = np.linspace(x_max + dx_original, x_max + extension_dist, n_extend_x)
                z_bottom_extend = np.linspace(z_max + dz_original, z_max + extension_dist, n_extend_z)

                x_coords = np.concatenate([x_left_extend, x_coords, x_right_extend])
                z_coords = np.concatenate([z_coords, z_bottom_extend])

                nz_ext = len(z_coords)
                nx_ext = len(x_coords)
                velocity_extended = np.zeros((nz_ext, nx_ext), dtype=float)

                velocity_extended[:nz_orig, n_extend_x : n_extend_x + nx_orig] = velocity_data

                if nx_orig > 0:
                    left_column = velocity_data[:, 0]
                    for j in range(n_extend_x):
                        velocity_extended[:nz_orig, j] = left_column
                        if n_extend_z > 0:
                            velocity_extended[nz_orig:, j] = (
                                left_column[-1] if len(left_column) > 0 else velocity_data[-1, 0]
                            )
                    right_column = velocity_data[:, -1]
                    right_start = n_extend_x + nx_orig
                    for j in range(n_extend_x):
                        velocity_extended[:nz_orig, right_start + j] = right_column
                        if n_extend_z > 0:
                            velocity_extended[nz_orig:, right_start + j] = (
                                right_column[-1] if len(right_column) > 0 else velocity_data[-1, -1]
                            )

                if n_extend_z > 0 and nx_orig > 0:
                    bottom_row = velocity_data[-1, :]
                    for i in range(n_extend_z):
                        velocity_extended[nz_orig + i, n_extend_x : n_extend_x + nx_orig] = bottom_row

                velocity_data = velocity_extended
                self._log_fn(f"  Extended model size: {velocity_data.shape}")

                original_x_min = float(x_coords[n_extend_x])
                original_x_max = float(x_coords[n_extend_x + nx_orig - 1])
            else:
                original_x_min = orig_x_min0
                original_x_max = orig_x_max0

            original_z_max = orig_z_max0

            self._log_fn("  Downsampling to coarse grid...")
            x_downsample = max(1, len(x_coords) // max_gs)
            z_downsample = max(1, len(z_coords) // max_gs)

            if x_downsample > 1 or z_downsample > 1:
                nz_coarse = (len(z_coords) + z_downsample - 1) // z_downsample
                nx_coarse = (len(x_coords) + x_downsample - 1) // x_downsample

                z_coarse = np.zeros(nz_coarse)
                x_coarse = np.zeros(nx_coarse)
                velocity_coarse = np.zeros((nz_coarse, nx_coarse))

                for i in range(nz_coarse):
                    start_idx = i * z_downsample
                    end_idx = min(start_idx + z_downsample, len(z_coords))
                    z_coarse[i] = float(np.mean(z_coords[start_idx:end_idx]))

                for j in range(nx_coarse):
                    start_idx = j * x_downsample
                    end_idx = min(start_idx + x_downsample, len(x_coords))
                    x_coarse[j] = float(np.mean(x_coords[start_idx:end_idx]))

                for i in range(nz_coarse):
                    z_start = i * z_downsample
                    z_end = min(z_start + z_downsample, len(z_coords))
                    for j in range(nx_coarse):
                        x_start = j * x_downsample
                        x_end = min(x_start + x_downsample, len(x_coords))
                        velocity_coarse[i, j] = float(
                            np.nanmean(velocity_data[z_start:z_end, x_start:x_end])
                        )

                x_coords = x_coarse
                z_coords = z_coarse
                velocity_data = velocity_coarse
                self._log_fn(f"  Coarse grid size: {velocity_data.shape}")
                self._log_fn(f"  Downsample factors: z={z_downsample}, x={x_downsample}")
            else:
                self._log_fn("  Model already small enough, no downsampling needed")

            self._log_fn("  Converting velocity to density...")
            valid_mask = ~(np.isnan(velocity_data) | (velocity_data <= 0.0))
            vp_data = np.where(valid_mask, velocity_data, np.nan)
            if str(model_type).strip().lower() == "vs":
                self._log_fn(
                    "  Note: full-model gravity always treats primary grid as Vp; "
                    "Wave Type selector is ignored here (Vs was converted to Vp)."
                )

            density_method_library = _density_method_for_library(density_method)
            density_data = np.where(
                valid_mask,
                calculate_density(vp_data, method=density_method_library),
                bg_density,
            )

            seawater_density = float(getattr(pc, "seawater_density_g_cm3", 1.03))
            if seafloor_fn is not None:
                try:
                    seafloor_depths = np.asarray(
                        seafloor_fn(np.asarray(x_coords, dtype=float)), dtype=float
                    )
                except Exception:
                    seafloor_depths = np.zeros(len(x_coords), dtype=float)
                water_mask = np.zeros_like(valid_mask, dtype=bool)
                for i in range(len(z_coords)):
                    z_val = float(z_coords[i])
                    for j in range(len(x_coords)):
                        if not valid_mask[i, j]:
                            continue
                        seafloor_z = float(seafloor_depths[j]) if j < len(seafloor_depths) else 0.0
                        vp_val = float(vp_data[i, j])
                        if pc.is_water_zone(vp=vp_val, z=z_val, seafloor_depth=seafloor_z):
                            water_mask[i, j] = True
                density_data[water_mask] = seawater_density
                n_water = int(np.sum(water_mask))
                n_valid = int(np.sum(valid_mask))
                wr = (n_water / n_valid) if n_valid > 0 else 0.0
                self._log_fn(
                    f"  Water cells: {n_water}/{n_valid} ({wr:.1%}), "
                    f"density fixed at {seawater_density:.3f} g/cm³"
                )
                if _uses_tomo2d_sediment_density(density_method) and basement_fn is not None:
                    try:
                        basement_depths = np.asarray(
                            basement_fn(np.asarray(x_coords, dtype=float)), dtype=float
                        )
                    except Exception:
                        basement_depths = np.full(len(x_coords), np.nan, dtype=float)
                    sed_mask = np.zeros_like(valid_mask, dtype=bool)
                    z_vals = np.asarray(z_coords, dtype=float)
                    for j in range(len(x_coords)):
                        seafloor_z = float(seafloor_depths[j]) if j < len(seafloor_depths) else 0.0
                        basement_z = float(basement_depths[j]) if j < len(basement_depths) else float("nan")
                        if not np.isfinite(basement_z) or basement_z <= seafloor_z:
                            continue
                        sed_mask[:, j] = (z_vals >= seafloor_z) & (z_vals < basement_z)
                    sed_mask = sed_mask & valid_mask & (~water_mask)
                    if np.any(sed_mask):
                        vp_sed = np.asarray(vp_data, dtype=float)[sed_mask]
                        sed_density = 1.0 + 1.18 * np.power(np.maximum(vp_sed - 1.5, 0.0), 0.22)
                        sed_density = np.minimum(sed_density, 2.6)
                        density_data[sed_mask] = sed_density
                        self._log_fn(
                            f"  Sediment cells (tomo2d mode): {int(np.sum(sed_mask))}, "
                            "rho = 1 + 1.18*(Vp-1.5)^0.22, capped at 2.6 g/cm³"
                        )

            effective_bg_density = float(bg_density)
            density_contrast = density_data - effective_bg_density
            valid_density = density_data[valid_mask]
            if (not strict_jgrav) and valid_density.size > 0:
                pos_ratio0 = float(np.mean(density_contrast[valid_mask] > 0.0))
                neg_ratio0 = float(np.mean(density_contrast[valid_mask] < 0.0))
                if min(pos_ratio0, neg_ratio0) < 0.02:
                    auto_bg = float(np.nanmedian(valid_density))
                    if np.isfinite(auto_bg):
                        effective_bg_density = auto_bg
                        density_contrast = density_data - effective_bg_density
                        self._log_fn(
                            "  Density contrast is strongly one-sided; "
                            f"using median density as effective background for anomaly centering: {effective_bg_density:.3f} g/cm³ "
                            f"(user background={bg_density:.3f} g/cm³)"
                        )

            density_contrast_kg_m3 = density_contrast * 1000.0

            x_orig_min = original_x_min
            x_orig_max = original_x_max
            z_orig_for_taper = original_z_max

            if extension_dist > 0.0:
                nz_c = len(z_coords)
                nx_c = len(x_coords)
                extension_mask = np.zeros((nz_c, nx_c), dtype=bool)
                extension_weight = np.ones((nz_c, nx_c), dtype=float)

                for i in range(nz_c):
                    for j in range(nx_c):
                        x_val = float(x_coords[j])
                        z_val = float(z_coords[i])
                        is_ext = (
                            x_val < x_orig_min or x_val > x_orig_max or z_val > z_orig_for_taper
                        )
                        if is_ext:
                            extension_mask[i, j] = True
                            wx = 1.0
                            wz = 1.0
                            if x_val < x_orig_min:
                                wx = max(
                                    0.0,
                                    min(
                                        1.0,
                                        (x_val - (x_orig_min - extension_dist))
                                        / max(extension_dist, 1e-6),
                                    ),
                                )
                            elif x_val > x_orig_max:
                                wx = max(
                                    0.0,
                                    min(
                                        1.0,
                                        ((x_orig_max + extension_dist) - x_val)
                                        / max(extension_dist, 1e-6),
                                    ),
                                )
                            if z_val > z_orig_for_taper:
                                wz = max(
                                    0.0,
                                    min(
                                        1.0,
                                        ((z_orig_for_taper + extension_dist) - z_val)
                                        / max(extension_dist, 1e-6),
                                    ),
                                )
                            extension_weight[i, j] = max(0.0, min(wx, wz))

                if not strict_jgrav:
                    density_contrast_kg_m3 = density_contrast_kg_m3 * extension_weight

                density_threshold = 10.0
                significant_mask = (np.abs(density_contrast_kg_m3) > density_threshold) & valid_mask
                n_extension = int(np.sum(extension_mask))
                self._log_fn(
                    f"  Extension cells: {n_extension} (density contrast tapered to suppress edge-induced trend)"
                )
            else:
                density_threshold = 10.0
                significant_mask = (np.abs(density_contrast_kg_m3) > density_threshold) & valid_mask

            n_significant = int(np.sum(significant_mask))
            self._log_fn(f"  Significant density cells: {n_significant} / {density_contrast_kg_m3.size}")

            if n_significant == 0:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "No significant density contrast found. Try adjusting background density.",
                )
                return

            if extension_dist > 0.0:
                x_min_obs, x_max_obs = original_x_min, original_x_max
            else:
                x_min_obs = float(np.min(x_coords))
                x_max_obs = float(np.max(x_coords))

            n_obs = min(50, len(x_coords))
            x_obs_edges = np.linspace(x_min_obs, x_max_obs, n_obs + 1)
            x_obs_km = 0.5 * (x_obs_edges[:-1] + x_obs_edges[1:])

            self._log_fn(f"  Number of observation points: {n_obs}")

            if gm == "tomo2d_fft":
                anomaly = self._grav_calc.calculate_tomo2d_fft_anomaly(
                    x_coords_km=np.asarray(x_coords, dtype=float),
                    z_coords_km=np.asarray(z_coords, dtype=float),
                    density_contrast_kg_m3=np.asarray(density_contrast_kg_m3, dtype=float),
                    significant_mask=np.asarray(significant_mask, dtype=bool),
                    x_obs_km=np.asarray(x_obs_km, dtype=float),
                    obs_level_km=float(obs_level),
                    ref_x_range_km=(float(x_min_obs), float(x_max_obs)),
                )
            else:
                anomaly = self._grav_calc.calculate_talwani_grid_anomaly(
                    x_coords_km=np.asarray(x_coords, dtype=float),
                    z_coords_km=np.asarray(z_coords, dtype=float),
                    density_contrast_kg_m3=np.asarray(density_contrast_kg_m3, dtype=float),
                    significant_mask=np.asarray(significant_mask, dtype=bool),
                    x_obs_km=np.asarray(x_obs_km, dtype=float),
                    obs_level_km=float(obs_level),
                )

            anomaly_note = ""
            gmin = float(np.nanmin(anomaly))
            gmax = float(np.nanmax(anomaly))
            if (not strict_jgrav) and np.isfinite(gmin) and np.isfinite(gmax) and (
                gmin >= 0.0 or gmax <= 0.0
            ):
                n_edge = max(2, min(8, len(anomaly) // 6))
                left_mean = float(np.nanmean(anomaly[:n_edge]))
                right_mean = float(np.nanmean(anomaly[-n_edge:]))
                if len(anomaly) > 1:
                    tlin = np.linspace(0.0, 1.0, num=len(anomaly))
                    baseline = left_mean + (right_mean - left_mean) * tlin
                    anomaly = anomaly - baseline
                    anomaly_note = "detrended"
                    self._log_fn(
                        "  Raw anomaly is one-signed; removed linear edge baseline for display "
                        f"(left={left_mean:.2f}, right={right_mean:.2f} mGal)."
                    )

            self._full_model_gravity_cache = {
                "cache_version": _FULL_MODEL_CACHE_VERSION,
                "model_key": model_key,
                "param_signature": param_signature,
                "x_obs_km": np.asarray(x_obs_km, dtype=float),
                "gravity_anomaly": np.asarray(anomaly, dtype=float),
                "obs_level": float(obs_level),
                "gravity_method": gm,
                "strict_jgrav": bool(strict_jgrav),
                "model_shape": (int(velocity_data.shape[0]), int(velocity_data.shape[1])),
                "anomaly_note": anomaly_note,
            }
            self._render_full_model_gravity_result(
                x_obs_km=np.asarray(x_obs_km, dtype=float),
                gravity_anomaly=np.asarray(anomaly, dtype=float),
                obs_level=float(obs_level),
                gravity_method=gm,
                strict_jgrav=strict_jgrav,
                from_cache=False,
                anomaly_note=anomaly_note,
                model_shape=(int(velocity_data.shape[0]), int(velocity_data.shape[1])),
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            QMessageBox.critical(self, "Error", str(e))

    def _clear_bodies(self) -> None:
        self._bodies.clear()
        self._plot_empty_message()
        self._log_fn("\nCleared gravity bodies.")
