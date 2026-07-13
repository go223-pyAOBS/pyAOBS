"""岩石数据库 Vp-Vs / Vp/Vs-Vp 窗口（Qt）；支持投点与多边形采样（对齐 Tk RockScatterMixin MVP）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import matplotlib

_candidate_backends = ("qtagg", "QtAgg", "Qt5Agg")
for _b in _candidate_backends:
    try:
        matplotlib.use(_b)
        break
    except ValueError:
        continue

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QMainWindow,
)

from pyAOBS.utils import calculate_vp_from_vs_brocher, calculate_vs, correct_velocity

from .rock_polygon_samples import uniform_polygon_interior_samples_xy
from .vp_vs_ratio_dem import (
    build_water_reference,
    draw_aspect_ratio_curves,
    draw_porosity_curves,
    draw_rock_database_points_ratio,
    draw_water_content_points,
    ensure_dem_curves,
    setup_vp_vs_ratio_axes,
    try_load_dem_curves_default_cache,
)
from .ui_chrome import apply_tool_window_chrome

ScatterMode = Literal["vp_vs", "ratio"]
RockScatterCtxGetter = Callable[[], Dict[str, Any]]


def scatter_model_points_holder(parent: QWidget, mode: ScatterMode) -> List[dict]:
    """与 Tk 主 viewer 上一组 model_points / vp_vs_ratio_model_points 类似：不因关闭子窗口而丢失。"""
    attr = "_rock_share_vp_vs" if mode == "vp_vs" else "_rock_share_ratio"
    cur = getattr(parent, attr, None)
    if isinstance(cur, list):
        return cur
    lst: List[dict] = []
    setattr(parent, attr, lst)
    return lst


def _is_valid_vp_vs_pair(vp: float, vs: float) -> bool:
    try:
        vp_f = float(vp)
        vs_f = float(vs)
    except (TypeError, ValueError):
        return False
    if (not np.isfinite(vp_f)) or (not np.isfinite(vs_f)) or vs_f <= 0.0 or vp_f <= 0.0:
        return False
    ratio = vp_f / vs_f
    return bool(np.isfinite(ratio) and (1.0 <= ratio <= 3.5))


def _get_point_vp_vs(ctx: Dict[str, Any], x: float, z: float) -> Tuple[float, float]:
    pc = ctx["property_calculator"]
    vm = str(ctx.get("velocity_method") or "brocher").strip()
    assert pc is not None
    if getattr(pc, "vs_grid_data", None) is not None:
        vp, vs, _ = pc.get_vp_vs_at_point(float(x), float(z), velocity_method=vm)
        return float(vp), float(vs)
    v0 = float(ctx["sample_primary_velocity"](float(x), float(z)))
    if ctx.get("model_type") == "vs":
        vs = v0
        vp = float(calculate_vp_from_vs_brocher(vs))
        return vp, vs
    vp = v0
    vs = float(calculate_vs(vp, method=vm))
    return vp, vs


def _scatter_db_vp_vs(ax, db_data: pd.DataFrame) -> Tuple[float, float, float, float]:
    """绘制数据库；返回坐标范围 (vs_min, vs_max, vp_min, vp_max) —— 对齐 Tk show_vp_vs_scatter。"""
    ax.clear()
    vs_min, vs_max = 2.0, 5.0
    vp_min, vp_max = 4.0, 8.5
    vp_vs_ratios = [1.7, 1.9, 2.1]
    vs_line = np.linspace(vs_min, vs_max, 100)
    for ratio in vp_vs_ratios:
        vp_line = ratio * vs_line
        mask = (vp_line >= vp_min) & (vp_line <= vp_max)
        ax.plot(
            vs_line[mask],
            vp_line[mask],
            linestyle="--",
            color="gray",
            alpha=0.5,
            linewidth=1.5,
            zorder=1,
            label=f"Vp/Vs = {ratio:.1f}",
        )

    highlighted_rocks_lower = {
        "basalt": "blue",
        "serpentinite": "green",
        "gabbro": "orange",
        "dunite": "purple",
        "granite": "brown",
    }
    highlighted_rocks_display = {
        "basalt": "Basalt",
        "serpentinite": "Serpentinite",
        "gabbro": "Gabbro",
        "dunite": "Dunite",
        "granite": "Granite",
    }

    if "rock_type" in db_data.columns:
        other_rocks_mask = pd.Series([False] * len(db_data), index=db_data.index)
        for rock_type in db_data["rock_type"].unique():
            rtl = str(rock_type).lower().strip()
            if rtl not in highlighted_rocks_lower:
                other_rocks_mask |= db_data["rock_type"] == rock_type
        if other_rocks_mask.any():
            ax.scatter(
                db_data.loc[other_rocks_mask, "vs"],
                db_data.loc[other_rocks_mask, "vp"],
                c="lightgray",
                alpha=0.4,
                s=30,
                edgecolors="gray",
                linewidths=0.3,
                zorder=5,
                label="_nolegend_",
            )
        rock_type_lower_map: Dict[str, Any] = {}
        for rock_type in db_data["rock_type"].unique():
            rock_type_lower_map[str(rock_type).lower().strip()] = rock_type

        for rock_key_lower, color in highlighted_rocks_lower.items():
            matched = rock_type_lower_map.get(rock_key_lower)
            if matched is None:
                continue
            mask = db_data["rock_type"] == matched
            if mask.any():
                disp = highlighted_rocks_display.get(rock_key_lower, str(matched))
                ax.scatter(
                    db_data.loc[mask, "vs"],
                    db_data.loc[mask, "vp"],
                    c=color,
                    label=disp,
                    alpha=0.7,
                    s=60,
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=5,
                )
    else:
        ax.scatter(
            db_data["vs"],
            db_data["vp"],
            alpha=0.6,
            s=50,
            edgecolors="black",
            linewidths=0.5,
            label="Database",
            zorder=5,
        )

    ax.set_xlim(vs_min, vs_max)
    ax.set_ylim(vp_min, vp_max)
    ax.set_xlabel("S-wave Velocity (km/s)", fontsize=12)
    ax.set_ylabel("P-wave Velocity (km/s)", fontsize=12)
    ax.set_title("Vp-Vs Scatter Plot - Rock Database", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_aspect("auto")
    return vs_min, vs_max, vp_min, vp_max


def _overlay_vp_vs_model(ax, model_points: List[dict], lims: Tuple[float, float, float, float]) -> None:
    vs_min, vs_max, vp_min, vp_max = lims
    if not model_points:
        return
    pts = [p for p in model_points if _is_valid_vp_vs_pair(p.get("vp"), p.get("vs"))]
    if not pts:
        return
    mvp = [float(p["vp"]) for p in pts]
    mvs = [float(p["vs"]) for p in pts]
    ax.scatter(
        mvs,
        mvp,
        c="red",
        marker="*",
        s=200,
        edgecolors="darkred",
        linewidths=1.5,
        label="Model Points",
        zorder=10,
    )
    for i, point in enumerate(pts):
        if point.get("type") != "point":
            continue
        pn = int(point.get("point_number", i + 1))
        ax.annotate(
            f"{pn}",
            (float(point["vs"]), float(point["vp"])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="white",
            bbox=dict(boxstyle="round", facecolor="red", alpha=0.7),
            ha="left",
            va="bottom",
            zorder=11,
        )
    ax.set_xlim(vs_min, vs_max)
    ax.set_ylim(vp_min, vp_max)


def _overlay_ratio_model(ax, model_points: List[dict]) -> None:
    if not model_points:
        return
    pts = [p for p in model_points if _is_valid_vp_vs_pair(p.get("vp"), p.get("vs"))]
    if not pts:
        return
    mvp = [float(p["vp"]) for p in pts]
    mratio = [float(p["vp"]) / float(p["vs"]) for p in pts]
    ax.scatter(
        mvp,
        mratio,
        c="red",
        marker="*",
        s=200,
        edgecolors="darkred",
        linewidths=1.5,
        label="Model Points",
        zorder=10,
    )
    for i, point in enumerate(pts):
        if point.get("type") != "point":
            continue
        pn = int(point.get("point_number", i + 1))
        vp = float(point["vp"])
        r = vp / float(point["vs"])
        ax.annotate(
            f"{pn}",
            (vp, r),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color="white",
            bbox=dict(boxstyle="round", facecolor="red", alpha=0.7),
            ha="left",
            va="bottom",
            zorder=11,
        )


class RockScatterToolWindow(QMainWindow):
    """Vp-Vs 或 Vp/Vs-Vp：数据库 + 可叠加模型点（温压校正到 200 MPa / 25°C）。"""

    def __init__(
        self,
        parent: QWidget,
        mode: ScatterMode,
        get_ctx: RockScatterCtxGetter,
        log_fn: Callable[[str], None],
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._get_ctx = get_ctx
        self._log_fn = log_fn
        self._model_points = scatter_model_points_holder(parent, mode)
        self._db_data: Optional[pd.DataFrame] = None
        self._vp_vs_lims: Tuple[float, float, float, float] = (2.0, 5.0, 4.0, 8.5)
        self._water_ref: Optional[Dict[str, Any]] = None
        self._dem_curves: Optional[Dict[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None
        self._dem_meta: Optional[Dict[str, Any]] = None
        self._chk_rocks: Optional[QCheckBox] = None
        self._chk_water: Optional[QCheckBox] = None
        self._chk_aspect: Optional[QCheckBox] = None
        self._chk_porosity: Optional[QCheckBox] = None

        self.setWindowTitle(
            "Vp-Vs Scatter Plot - Rock Database" if mode == "vp_vs" else "Vp/Vs vs Vp Plot - Rock Database"
        )
        self.resize(900, 700)

        self._fig = Figure(figsize=(8, 6))
        self._ax = self._fig.add_subplot(111)

        central = QWidget()
        self.setCentralWidget(central)
        lay = QVBoxLayout(central)

        top_row = QHBoxLayout()
        btn_save = QPushButton("Save Figure")
        btn_save.clicked.connect(self._save_fig)
        top_row.addWidget(btn_save)

        if mode == "ratio":
            opt_g = QGroupBox("显示选项")
            hl = QHBoxLayout(opt_g)
            self._chk_rocks = QCheckBox("采样岩石分布")
            self._chk_water = QCheckBox("含水量(蛇纹石化%)")
            self._chk_aspect = QCheckBox("纵横比曲线")
            self._chk_porosity = QCheckBox("孔隙度曲线")
            for c in (
                self._chk_rocks,
                self._chk_water,
                self._chk_aspect,
                self._chk_porosity,
            ):
                c.setChecked(True)
                c.stateChanged.connect(self._ratio_options_changed)
                hl.addWidget(c)
            top_row.addStretch(1)
            top_row.addWidget(opt_g)
        lay.addLayout(top_row)

        canvas = FigureCanvasQTAgg(self._fig)
        lay.addWidget(canvas, stretch=1)
        tb = NavigationToolbar2QT(canvas, central)
        lay.addWidget(tb)
        self._canvas = canvas

        ok, msg = self._reload_database()
        if not ok:
            self._ax.text(0.5, 0.5, msg, ha="center", va="center", transform=self._ax.transAxes)
        elif mode == "ratio":
            self._ensure_ratio_dem_assets()
            self._refresh_plot()
            self._log_ratio_plot_opened_summary()
        else:
            self._refresh_plot()
            self._log_vp_vs_plot_opened_summary()
        apply_tool_window_chrome(self)

    def _ratio_options_changed(self, *_args: Any) -> None:
        self._refresh_plot()

    def _log_vp_vs_plot_opened_summary(self) -> None:
        if self._db_data is None:
            return
        db_data = self._db_data
        self._log_fn("\nVp-Vs scatter plot displayed:")
        self._log_fn(f"  Database points: {len(db_data)}")
        self._log_fn(f"  Model points: {len(self._model_points)}")
        self._log_fn(f"  Data status: All velocities corrected to 200 MPa, 25°C")
        self._log_fn(f"  Coordinate range: Vs [2.0-5.0 km/s], Vp [4.0-8.5 km/s]")
        if "rock_type" in db_data.columns:
            unique_rocks = db_data["rock_type"].unique()
            self._log_fn(f"  Total rock types: {len(unique_rocks)}")
            hl = {"basalt", "serpentinite", "gabbro", "dunite", "granite"}
            highlighted_count = sum(1 for r in unique_rocks if str(r).lower().strip() in hl)
            self._log_fn(f"  Highlighted rock types: {highlighted_count}")

    def _log_ratio_plot_opened_summary(self) -> None:
        assert self._chk_rocks is not None
        if self._db_data is None:
            return
        self._log_fn("\nVp/Vs vs Vp plot displayed:")
        self._log_fn(f"  Database points: {len(self._db_data)}")
        self._log_fn(f"  Model points: {len(self._model_points)}")
        wr = self._water_ref
        if wr and isinstance(wr, dict) and wr.get("vp"):
            self._log_fn(f"  Water content reference points: {len(wr['vp'])}")
        if self._dem_curves:
            self._log_fn(f"  DEM curves calculated: {len(self._dem_curves)} aspect ratios")
        self._log_fn(
            f"  Display options: Rocks={self._chk_rocks.isChecked()}, "
            f"Water={self._chk_water.isChecked() if self._chk_water else False}, "
            f"AspectRatio={self._chk_aspect.isChecked() if self._chk_aspect else False}, "
            f"Porosity={self._chk_porosity.isChecked() if self._chk_porosity else False}"
        )

    def reload_from_classifier_and_redraw(self) -> tuple[bool, str]:
        ok, msg = self._reload_database()
        if not ok:
            return False, msg
        if self._mode == "ratio":
            self._ensure_ratio_dem_assets()
            self._refresh_plot()
            self._log_ratio_plot_opened_summary()
        else:
            self._refresh_plot()
            self._log_vp_vs_plot_opened_summary()
        return True, ""

    def _ensure_ratio_dem_assets(self) -> None:
        """含水参考与 DEM（缓存命中则很快）；首次计算可能较慢。"""
        assert self._db_data is not None and self._mode == "ratio"
        df = self._db_data
        if "vp_vs_ratio" not in df.columns:
            df.loc[:, "vp_vs_ratio"] = df["vp"] / df["vs"]
        try:
            self._water_ref = build_water_reference()
        except Exception as e:
            self._water_ref = None
            QMessageBox.warning(self, "Water reference", str(e))

        cached = try_load_dem_curves_default_cache()
        if cached is not None:
            self._dem_curves, self._dem_meta = cached
        else:
            dlg = QProgressDialog("正在计算DEM曲线…", None, 0, 100, self)
            dlg.setWindowTitle("计算DEM曲线")
            dlg.setWindowModality(Qt.WindowModal)
            dlg.setMinimumDuration(0)
            dlg.setValue(0)
            dlg.show()

            try:

                def _cb(cur: int, total: int, msg: str) -> None:
                    if total <= 0:
                        return
                    dlg.setMaximum(100)
                    dlg.setValue(int(100 * cur / total))
                    dlg.setLabelText(msg)
                    QApplication.processEvents()

                self._dem_curves, self._dem_meta = ensure_dem_curves(
                    log_fn=self._log_fn,
                    progress_callback=_cb,
                )
            except Exception as e:
                QMessageBox.critical(self, "DEM", str(e))
                self._dem_curves = None
                self._dem_meta = None
                if self._chk_aspect:
                    self._chk_aspect.setEnabled(False)
                    self._chk_porosity.setEnabled(False)
            finally:
                dlg.setValue(100)
                dlg.close()

        dem_n = len(self._dem_curves or {})

        wr = self._water_ref if isinstance(self._water_ref, dict) else None
        wl = len(wr["vp"]) if wr is not None and wr.get("vp") is not None else 0
        self._log_fn(
            f"Vp/Vs vs Vp: water ref points={wl}, DEM aspect curves={dem_n}; "
            f"rocks layer default ON (toggle in window)."
        )

    def _reload_database(self) -> tuple[bool, str]:
        ctx = self._get_ctx()
        pc = ctx.get("property_calculator")
        if pc is None or pc.rock_classifier is None:
            return False, "Rock classifier not available."
        try:
            classifier = pc.rock_classifier.classifier
            if classifier is None:
                return False, "Classifier not loaded."
            db_data = classifier.rock_database_standard
            if "vp" not in db_data.columns or "vs" not in db_data.columns:
                return False, "Database missing vp or vs columns."
        except Exception as e:
            return False, str(e)
        self._db_data = db_data.copy()
        if self._mode == "ratio":
            self._db_data["vp_vs_ratio"] = self._db_data["vp"] / self._db_data["vs"]
        return True, ""

    def _refresh_ratio_plot(self) -> None:
        assert self._db_data is not None
        df = self._db_data
        assert self._chk_rocks is not None

        self._fig.subplots_adjust(left=0.1, right=0.7, top=0.95, bottom=0.1)
        self._ax.clear()

        if self._chk_rocks.isChecked():
            draw_rock_database_points_ratio(self._ax, df)

        _overlay_ratio_model(self._ax, self._model_points)

        if (
            self._chk_water is not None
            and self._chk_water.isChecked()
            and self._water_ref
        ):
            draw_water_content_points(self._ax, self._water_ref)

        if (
            self._chk_aspect is not None
            and self._chk_aspect.isChecked()
            and self._dem_curves
            and self._dem_meta
        ):
            draw_aspect_ratio_curves(self._ax, self._dem_curves, self._dem_meta)

        if (
            self._chk_porosity is not None
            and self._chk_porosity.isChecked()
            and self._dem_curves
            and self._dem_meta
        ):
            draw_porosity_curves(self._ax, self._dem_curves, self._dem_meta)

        hv = (
            float(self._dem_meta.get("host_vp_dunite", 8.299))
            if self._dem_meta
            else 8.299
        )
        hv_s = (
            float(self._dem_meta.get("host_vs_dunite", 4.731))
            if self._dem_meta
            else 4.731
        )
        setup_vp_vs_ratio_axes(self._ax, host_vp_dunite=hv, host_vs_dunite=hv_s)

    def _refresh_plot(self) -> None:
        assert self._db_data is not None
        if self._mode == "vp_vs":
            self._fig.subplots_adjust(left=0.1, right=0.7, top=0.95, bottom=0.1)
            self._vp_vs_lims = _scatter_db_vp_vs(self._ax, self._db_data)
            _overlay_vp_vs_model(self._ax, self._model_points, self._vp_vs_lims)
        else:
            self._refresh_ratio_plot()
        self._canvas.draw()

    def _save_fig(self) -> None:
        from .file_dialogs_qt import normalize_save_path_for_filter

        path, selected = QFileDialog.getSaveFileName(
            self, "Save Figure", str(Path.cwd()), "PNG (*.png);;PDF (*.pdf);;All (*.*)"
        )
        if not path:
            return
        path = normalize_save_path_for_filter(path, selected)
        try:
            self._fig.savefig(path, dpi=300, bbox_inches="tight")
            QMessageBox.information(self, "Saved", path)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def add_selected_points(self) -> None:
        ctx = self._get_ctx()
        gd = ctx.get("grid_data")
        pc = ctx.get("property_calculator")
        if gd is None or pc is None:
            QMessageBox.warning(self, "Warning", "Please load model first.")
            return
        pts = ctx["selected_points"]()
        if not pts:
            QMessageBox.warning(
                self,
                "Warning",
                "No points selected. Enable point pick on main window first.",
            )
            return
        comp_pt = ctx["compute_pt"]
        added = 0
        for idx, (x, z) in enumerate(pts):
            try:
                vo, wo = _get_point_vp_vs(ctx, float(x), float(z))
                if not _is_valid_vp_vs_pair(vo, wo):
                    self._log_fn(
                        f"  Warning: Skipped point ({float(x):.2f}, {float(z):.2f}) due to invalid Vp/Vs"
                    )
                    continue
                pr, tc = comp_pt(float(x), float(z), vo)
                vp_c = float(
                    correct_velocity(
                        vo,
                        pressure=pr,
                        temperature=tc,
                        target_pressure=200.0,
                        target_temperature=25.0,
                        is_s_wave=False,
                    )
                )
                vs_c = float(
                    correct_velocity(
                        wo,
                        pressure=pr,
                        temperature=tc,
                        target_pressure=200.0,
                        target_temperature=25.0,
                        is_s_wave=True,
                    )
                )
                self._model_points.append(
                    {
                        "vp": vp_c,
                        "vs": vs_c,
                        "x": float(x),
                        "z": float(z),
                        "type": "point",
                        "point_number": idx + 1,
                    }
                )
                added += 1
            except Exception as e:
                self._log_fn(f"  Failed point ({x},{z}): {e}")
        self._refresh_plot()
        target = "scatter plot" if self._mode == "vp_vs" else "Vp/Vs vs Vp plot"
        self._log_fn(f"  Added {added} points to {target}")
        if added > 0:
            suf = "scatter plot" if self._mode == "vp_vs" else "plot"
            self._log_fn(f"    All points corrected to 200 MPa, 25°C before adding to {suf}")

    def _polygon_grid_samples(
        self, ctx: Dict[str, Any], polygon: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        gd = ctx["grid_data"]
        pe = ctx["profile_extractor"]
        xc = gd[pe.x_coord].values
        zc = gd[pe.z_coord].values
        px = [p[0] for p in polygon]
        pz = [p[1] for p in polygon]
        x_min, x_max = min(px), max(px)
        z_min, z_max = min(pz), max(pz)
        poly_path = MplPath(polygon)
        out: List[Tuple[float, float]] = []
        for xx in xc:
            if x_min <= float(xx) <= x_max:
                for zz in zc:
                    fz = float(zz)
                    if z_min <= fz <= z_max and poly_path.contains_point((float(xx), fz)):
                        out.append((float(xx), fz))
        return out

    def add_polygon_average(self) -> None:
        ctx = self._get_ctx()
        if ctx.get("grid_data") is None or ctx.get("property_calculator") is None:
            QMessageBox.warning(self, "Warning", "Please load model first.")
            return
        polys = ctx["selected_polygons"]()
        if not polys:
            QMessageBox.warning(
                self,
                "Warning",
                "No polygons selected. Please select polygon first.",
            )
            return
        polygon = polys[-1]
        if len(polygon) < 3:
            QMessageBox.warning(self, "Warning", "Polygon must have at least 3 vertices.")
            return
        cells = self._polygon_grid_samples(ctx, [(float(v[0]), float(v[1])) for v in polygon])
        vp_vals: List[float] = []
        vs_vals: List[float] = []
        for x, z in cells:
            try:
                vp, vs = _get_point_vp_vs(ctx, x, z)
                if _is_valid_vp_vs_pair(vp, vs):
                    vp_vals.append(vp)
                    vs_vals.append(vs)
            except Exception:
                pass
        if not vp_vals:
            QMessageBox.warning(self, "Warning", "No valid points found in polygon.")
            return
        vo = float(np.mean(vp_vals))
        wo = float(np.mean(vs_vals))
        comp_pt = ctx["compute_pt"]
        px_f = float(np.mean([float(p[0]) for p in polygon]))
        pz_f = float(np.mean([float(p[1]) for p in polygon]))
        pr, tc = comp_pt(px_f, pz_f, vo)
        vp_c = float(
            correct_velocity(
                vo, pressure=pr, temperature=tc, target_pressure=200.0, target_temperature=25.0, is_s_wave=False
            )
        )
        vs_c = float(
            correct_velocity(
                wo, pressure=pr, temperature=tc, target_pressure=200.0, target_temperature=25.0, is_s_wave=True
            )
        )
        self._model_points.append(
            {
                "vp": vp_c,
                "vs": vs_c,
                "x": px_f,
                "z": pz_f,
                "type": "polygon_average",
                "n_points": len(vp_vals),
            }
        )
        self._refresh_plot()
        plot_name = "scatter plot" if self._mode == "vp_vs" else "Vp/Vs vs Vp plot"
        self._log_fn(f"  Added polygon average point (n={len(vp_vals)}) to {plot_name}")
        self._log_fn(
            f"    Original: Vp={vo:.2f}, Vs={wo:.2f} km/s (P={pr:.1f} MPa, T={tc:.1f}°C)"
        )
        self._log_fn(
            f"    Corrected: Vp={vp_c:.2f}, Vs={vs_c:.2f} km/s (200 MPa, 25°C)"
        )

    def add_polygon_samples(self) -> None:
        ctx = self._get_ctx()
        if ctx.get("grid_data") is None or ctx.get("property_calculator") is None:
            QMessageBox.warning(self, "Warning", "Please load model first.")
            return
        polys = ctx["selected_polygons"]()
        if not polys:
            QMessageBox.warning(
                self,
                "Warning",
                "No polygons selected. Please select polygon first.",
            )
            return
        polygon = polys[-1]
        if len(polygon) < 3:
            QMessageBox.warning(self, "Warning", "Polygon must have at least 3 vertices.")
            return
        n_samples, ok = QInputDialog.getInt(
            self, "Sample Count", "Enter number of samples:", 10, 1, 100, 1
        )
        if not ok:
            return
        poly_xy = [(float(v[0]), float(v[1])) for v in polygon]
        rng = np.random.default_rng()
        chosen = uniform_polygon_interior_samples_xy(poly_xy, int(n_samples), rng=rng)
        if not chosen:
            QMessageBox.warning(self, "Warning", "No valid samples found in polygon.")
            return

        draw_main = ctx.get("add_polygon_sample_markers_on_main")
        comp_pt = ctx["compute_pt"]
        added = 0
        invalid_count = 0
        sample_x: List[float] = []
        sample_z: List[float] = []
        for x, z in chosen:
            try:
                vo, wo = _get_point_vp_vs(ctx, float(x), float(z))
                if not _is_valid_vp_vs_pair(vo, wo):
                    invalid_count += 1
                    continue
                pr, tc = comp_pt(float(x), float(z), vo)
                vp_c = float(
                    correct_velocity(
                        vo,
                        pressure=pr,
                        temperature=tc,
                        target_pressure=200.0,
                        target_temperature=25.0,
                        is_s_wave=False,
                    )
                )
                vs_c = float(
                    correct_velocity(
                        wo,
                        pressure=pr,
                        temperature=tc,
                        target_pressure=200.0,
                        target_temperature=25.0,
                        is_s_wave=True,
                    )
                )
                self._model_points.append(
                    {
                        "vp": vp_c,
                        "vs": vs_c,
                        "x": float(x),
                        "z": float(z),
                        "type": "polygon_sample",
                    }
                )
                sample_x.append(float(x))
                sample_z.append(float(z))
                added += 1
            except Exception:
                pass
        if sample_x and callable(draw_main):
            draw_main(sample_x, sample_z)

        self._refresh_plot()
        plot_name = "scatter plot" if self._mode == "vp_vs" else "Vp/Vs vs Vp plot"
        self._log_fn(f"  Added {added} polygon samples to {plot_name}")
        if invalid_count > 0:
            self._log_fn(f"  Warning: skipped {invalid_count} invalid Vp/Vs samples")
        if sample_x:
            self._log_fn(f"  Sample points displayed on main plot (orange squares)")
        if added > 0:
            suf = "scatter plot" if self._mode == "vp_vs" else "plot"
            self._log_fn(f"    All samples corrected to 200 MPa, 25°C before adding to {suf}")

    def clear_model_points(self) -> None:
        self._model_points.clear()
        self._refresh_plot()
        plot_name = "scatter plot" if self._mode == "vp_vs" else "Vp/Vs vs Vp plot"
        self._log_fn(f"  Cleared all model points from {plot_name}")


def get_or_create_rock_scatter_window(
    parent: QWidget,
    mode: ScatterMode,
    get_ctx: RockScatterCtxGetter,
    log_fn: Callable[[str], None],
    *,
    refresh_if_visible: bool = True,
    _hold: Optional[dict[str, Any]] = None,
) -> RockScatterToolWindow:
    key = "vp_vs" if mode == "vp_vs" else "ratio"
    h = _hold if _hold is not None else getattr(parent, "_rock_scatter_windows", None)
    if h is None:
        h = {}
        setattr(parent, "_rock_scatter_windows", h)
    w = h.get(key)
    if isinstance(w, RockScatterToolWindow):
        try:
            if w.isVisible():
                if refresh_if_visible:
                    ok, msg = w.reload_from_classifier_and_redraw()
                    if not ok:
                        QMessageBox.warning(parent, "Warning", msg)
                w.raise_()
                w.activateWindow()
                return w
        except Exception:
            pass
        h.pop(key, None)
    win = RockScatterToolWindow(parent, mode, get_ctx, log_fn)
    h[key] = win
    win.show()
    return win


def open_vp_vs_scatter(
    parent: QWidget,
    get_ctx: RockScatterCtxGetter,
    log_fn: Callable[[str], None],
    *,
    _hold: Optional[dict[str, Any]] = None,
) -> None:
    get_or_create_rock_scatter_window(
        parent, "vp_vs", get_ctx, log_fn, refresh_if_visible=True, _hold=_hold
    )


def open_vp_vs_ratio_plot(
    parent: QWidget,
    get_ctx: RockScatterCtxGetter,
    log_fn: Callable[[str], None],
    *,
    _hold: Optional[dict[str, Any]] = None,
) -> None:
    get_or_create_rock_scatter_window(
        parent, "ratio", get_ctx, log_fn, refresh_if_visible=True, _hold=_hold
    )


