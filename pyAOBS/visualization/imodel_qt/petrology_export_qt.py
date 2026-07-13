"""LIP Petrology export dialog and helpers for imodel Qt."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from matplotlib.path import Path as MplPath
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from petrology.imodel_bridge.export_contract import CrustObservation, save_observation_json
from petrology.imodel_bridge.imodel_adapter import (
    crust_from_interfaces_at_x,
    crust_from_polygon_selection,
    crust_from_vertical_profile,
    format_crust_observation_summary,
)

from .styles import apply_dialog_style, hint_label, primary_button, status_panel

OnObservationFn = Callable[[CrustObservation, Optional[Path], float, float], None]


def _surface_z_at(seafloor_fn: Callable[[float], float] | None, x_km: float) -> float:
    if seafloor_fn is None:
        return 0.0
    z = float(seafloor_fn(float(x_km)))
    return z if np.isfinite(z) else 0.0


def _require_igneous_column(
    ctx: dict[str, Any],
    x_km: float,
    *,
    igneous_top_fn: Callable[[float], float],
    moho_fn: Callable[[float], float],
) -> tuple[float, float, float]:
    if not ctx.get("has_basement_interface"):
        raise RuntimeError("请先在 Profile 中选择 B（沉积基底 / 火成柱顶）")
    if not ctx.get("has_moho_interface"):
        raise RuntimeError("请先在 Profile 中选择 M（Moho / 火成柱底）")
    z_top = float(igneous_top_fn(float(x_km)))
    z_bot = float(moho_fn(float(x_km)))
    if not np.isfinite(z_top) or not np.isfinite(z_bot):
        raise RuntimeError(f"x={x_km:g} km 处 B/M 深度无效")
    h = abs(z_bot - z_top)
    if h <= 0.0:
        raise RuntimeError(f"x={x_km:g} km 处 H≤0（B={z_top:g}, M={z_bot:g}）")
    return z_top, z_bot, h


def _require_crust_interfaces(
    ctx: dict[str, Any],
    x_km: float,
    *,
    moho_fn: Callable[[float], float],
    igneous_top_fn: Callable[[float], float],
    seafloor_fn: Callable[[float], float] | None,
) -> tuple[float, float, float]:
    del seafloor_fn  # H 不再从海底起算
    return _require_igneous_column(
        ctx, x_km, igneous_top_fn=igneous_top_fn, moho_fn=moho_fn
    )


class PetrologyExportDialog(QDialog):
    """Export single-point (H, V_LC) for LIP Petrology — Fig.12a anchor."""

    def __init__(
        self,
        parent: QWidget | None,
        *,
        get_ctx: Callable[[], dict[str, Any]],
        log_fn: Callable[[str], None],
        on_observation: OnObservationFn | None = None,
    ) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle("单点导出到 LIP Petrology")
        self.resize(520, 480)
        self._get_ctx = get_ctx
        self._log = log_fn
        self._on_observation = on_observation
        self._result: CrustObservation | None = None
        self._saved_path: Path | None = None

        root = QVBoxLayout(self)
        root.addWidget(
            hint_label(
                "单点观测 (H, V_LC, f_lower) → 保存 JSON → LIP Petrology 反演锚点。\n"
                "H = |M − B|；f_lower 由下地壳顶界/多边形自动计算。\n"
                "沿迹多窗滑窗请用侧栏 Transect…；预览可叠加已导出的沿迹窗。"
            )
        )

        method_grp = QGroupBox("计算方法")
        method_grp.setObjectName("ImodelBridgeGroup")
        mv = QVBoxLayout(method_grp)
        self._rb_polygon = QRadioButton("多边形选段（需先在主图完成 Polygon Selection）")
        self._rb_interfaces = QRadioButton("界面自动 H + x 处竖线采样（推荐）")
        self._rb_profile = QRadioButton("X 范围平均剖面 → 下地壳谐和平均")
        self._rb_interfaces.setChecked(True)
        mv.addWidget(self._rb_polygon)
        mv.addWidget(self._rb_interfaces)
        mv.addWidget(self._rb_profile)
        for rb in (self._rb_polygon, self._rb_interfaces, self._rb_profile):
            rb.toggled.connect(lambda *_a: self._sync_lc_top_enabled())
        root.addWidget(method_grp)

        params = QFormLayout()
        self._x_spin = QDoubleSpinBox()
        self._x_spin.setRange(-1e6, 1e6)
        self._x_spin.setDecimals(2)
        self._x_spin.setSuffix(" km")
        self._x_spin.setValue(0.0)
        params.addRow("剖面 x:", self._x_spin)

        self._lc_top = QComboBox()
        self._lc_top.setToolTip(
            "下地壳顶界 v.in 界面；f_lower = (M − 顶界) / (M − B) 由代码自动计算"
        )
        params.addRow("下地壳顶:", self._lc_top)
        root.addLayout(params)

        self._launch_chk = QCheckBox("保存后启动 LIP Petrology GUI 并导入")
        self._launch_chk.setChecked(True)
        root.addWidget(self._launch_chk)

        self._summary = status_panel("(尚未计算)")
        root.addWidget(self._summary)

        btn_row = QHBoxLayout()
        btn_calc = primary_button("计算")
        btn_calc.clicked.connect(self._on_compute)
        btn_preview = QPushButton("H–Vp图预览")
        btn_preview.setToolTip("单点投 Fig.12a；(c) 面板；若已有沿迹导出则一并叠加")
        btn_preview.clicked.connect(self._on_preview)
        btn_save = QPushButton("保存 JSON…")
        btn_save.clicked.connect(self._on_save)
        btn_row.addWidget(btn_calc)
        btn_row.addWidget(btn_preview)
        btn_row.addWidget(btn_save)
        btn_row.addStretch()
        root.addLayout(btn_row)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        apply_dialog_style(self)
        self._sync_lc_top_enabled()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh_lc_top_combo()

    def _refresh_lc_top_combo(self) -> None:
        current = self._lc_top.currentData()
        self._lc_top.clear()
        opts = self._ctx().get("lc_top_interface_options") or []
        if not opts:
            self._lc_top.addItem("(无 v.in 界面 — 请用手画多边形)", None)
        for label, fn in opts:
            self._lc_top.addItem(str(label), fn)
        idx = self._lc_top.findData(current)
        if idx >= 0:
            self._lc_top.setCurrentIndex(idx)
        elif self._lc_top.count() > 0 and opts:
            self._lc_top.setCurrentIndex(0)

    def _require_lc_top_fn(self) -> Callable[[float], float]:
        fn = self._selected_lc_top_fn()
        if fn is None:
            raise RuntimeError("请选择下地壳顶界 v.in 界面，或改用手画多边形")
        return fn

    def _selected_lc_top_fn(self) -> Callable[[float], float] | None:
        if self._rb_polygon.isChecked():
            return None
        data = self._lc_top.currentData()
        return data if callable(data) else None

    def _sync_lc_top_enabled(self) -> None:
        polygon = self._rb_polygon.isChecked()
        self._lc_top.setEnabled(not polygon)

    @property
    def observation(self) -> CrustObservation | None:
        return self._result

    @property
    def saved_path(self) -> Path | None:
        return self._saved_path

    def _ctx(self) -> dict[str, Any]:
        return self._get_ctx()

    def _grid_vp_fn(self, ctx: dict[str, Any]) -> Callable[[float, float], float]:
        sample = ctx.get("sample_primary_velocity")
        if sample is None:
            raise RuntimeError("无速度采样函数")
        return lambda x, z: float(sample(float(x), float(z)))

    def _pt_fns(self, ctx: dict[str, Any], x_km: float):
        compute_pt = ctx.get("compute_pt")
        if compute_pt is None:
            return None, None

        def p_fn(_x: float, z: float) -> float:
            p, _t = compute_pt(float(x_km), z, 6.0)
            return float(p)

        def t_fn(_x: float, z: float) -> float:
            _p, t = compute_pt(float(x_km), z, 6.0)
            return float(t)

        return p_fn, t_fn

    def _on_compute(self) -> None:
        try:
            obs = self._compute_observation()
        except Exception as exc:
            QMessageBox.critical(self, "计算失败", str(exc))
            return
        self._result = obs
        self._summary.setText(format_crust_observation_summary(obs))
        self._log(f"Petrology export:\n{format_crust_observation_summary(obs)}")
        self._notify_observation(None)

    def _notify_observation(self, saved: Path | None) -> None:
        if self._result is None or self._on_observation is None:
            return
        self._on_observation(
            self._result,
            saved,
            float(self._result.f_lower),
            float(self._x_spin.value()),
        )

    def _resolve_transect_windows(self) -> list:
        raw = self._ctx().get("transect_windows")
        if callable(raw):
            return list(raw() or [])
        return list(raw or [])

    def _on_preview(self) -> None:
        if self._result is None:
            self._on_compute()
        if self._result is None:
            return
        from .hvp_overlay_preview_qt import show_hvp_overlay_preview

        ctx = self._ctx()
        model = str(ctx.get("current_model_file") or "").strip()
        label = Path(model).stem if model else "profile"
        windows = self._resolve_transect_windows()
        show_hvp_overlay_preview(
            self,
            observation=self._result,
            windows=windows or None,
            transect_label=label,
        )

    def _compute_observation(self) -> CrustObservation:
        ctx = self._ctx()
        if ctx.get("grid_data") is None:
            raise RuntimeError("请先加载 Vp 模型")
        grid_vp_fn = self._grid_vp_fn(ctx)
        x_km = float(self._x_spin.value())
        p_fn, t_fn = self._pt_fns(ctx, x_km)

        moho_fn = ctx.get("moho_depth_at_x")
        top_fn = ctx.get("igneous_top_depth_at_x") or ctx.get("basement_depth_at_x")
        if moho_fn is None or top_fn is None:
            raise RuntimeError("请设置 B（沉积基底）与 M（Moho）")
        z_top, z_bot, h = _require_crust_interfaces(
            ctx,
            x_km,
            moho_fn=moho_fn,
            igneous_top_fn=top_fn,
            seafloor_fn=None,
        )

        if self._rb_polygon.isChecked():
            polys = ctx.get("selected_polygons") or []
            if callable(polys):
                polys = polys()
            if not polys:
                raise RuntimeError("请先用 Polygon Selection 选择下地壳区域")
            polygon = polys[-1]
            if len(polygon) < 3:
                raise RuntimeError("多边形顶点不足")
            poly_xy = [(float(p[0]), float(p[1])) for p in polygon]
            from petrology.imodel_bridge.crust_geometry import polygon_effective_x_range

            prof_x0 = ctx.get("x_min_km")
            prof_x1 = ctx.get("x_max_km")
            prof_ok = (
                prof_x0 is not None
                and prof_x1 is not None
                and float(prof_x1) > float(prof_x0)
            )
            eff_lo, eff_hi = polygon_effective_x_range(
                poly_xy,
                x_min_km=float(prof_x0) if prof_ok else None,
                x_max_km=float(prof_x1) if prof_ok else None,
            )
            x_km = float(np.clip(x_km, eff_lo, eff_hi))
            p_fn, t_fn = self._pt_fns(ctx, x_km)
            z_top, z_bot, h = _require_crust_interfaces(
                ctx,
                x_km,
                moho_fn=moho_fn,
                igneous_top_fn=top_fn,
                seafloor_fn=None,
            )
            return crust_from_polygon_selection(
                poly_xy,
                grid_vp_fn=grid_vp_fn,
                h_whole_km=h,
                z_basement_km=z_top,
                z_moho_km=z_bot,
                x_km=x_km,
                x_min_km=float(prof_x0) if prof_ok else None,
                x_max_km=float(prof_x1) if prof_ok else None,
                p_local_mpa_fn=p_fn,
                t_local_c_fn=t_fn,
            )

        lc_top_fn = self._require_lc_top_fn()

        if self._rb_interfaces.isChecked():
            return crust_from_interfaces_at_x(
                x_km,
                moho_z_fn=moho_fn,
                igneous_top_z_fn=top_fn,
                grid_vp_fn=grid_vp_fn,
                lower_crust_top_z_fn=lc_top_fn,
                p_local_mpa_fn=p_fn,
                t_local_c_fn=t_fn,
            )

        from petrology.imodel_bridge.crust_geometry import (
            effective_f_lower,
            lower_crust_relative_bounds,
        )

        lc_top_z = float(lc_top_fn(x_km))
        z_rel_lo, _z_rel_hi = lower_crust_relative_bounds(
            h, z_basement_km=z_top, lower_crust_top_z_km=lc_top_z
        )
        f_used = effective_f_lower(h, z_rel_lo)
        prof = ctx.get("build_absolute_profile") or ctx.get("build_averaged_profile")
        if prof is None:
            raise RuntimeError("剖面提取未配置")
        df = prof()
        if df is None or df.empty:
            raise RuntimeError("剖面为空；请设置 Profile Extraction 的 X 范围并提取剖面")
        depths = df["depth"].values.astype(float)
        vps = df["vp"].values.astype(float)
        return crust_from_vertical_profile(
            depths,
            vps,
            h_whole_km=h,
            f_lower=f_used,
            z_surface_km=z_top,
            x_km=x_km,
            p_local_mpa_fn=p_fn,
            t_local_c_fn=t_fn,
        )

    def _default_save_path(self) -> Path:
        ctx = self._ctx()
        model = str(ctx.get("current_model_file") or "").strip()
        if model:
            return Path(model).with_suffix(".petrology_obs.json")
        return Path.cwd() / "imodel_petrology_obs.json"

    def _on_save(self) -> None:
        if self._result is None:
            try:
                self._on_compute()
            except Exception:
                return
        if self._result is None:
            return
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存 CrustObservation JSON",
            str(self._default_save_path()),
            "JSON (*.json)",
        )
        if not path:
            return
        save_observation_json(self._result, path)
        self._saved_path = Path(path)
        self._log(f"已保存 Petrology 观测 → {path}")
        self._notify_observation(self._saved_path)
        if self._launch_chk.isChecked():
            launch_lip_petrology_with_obs(path)

    def accept_and_export(self) -> CrustObservation | None:
        self._on_compute()
        if self._result is None:
            return None
        path = self._default_save_path()
        save_observation_json(self._result, path)
        self._saved_path = path
        self._log(f"已自动保存 → {path}")
        self._notify_observation(path)
        if self._launch_chk.isChecked():
            launch_lip_petrology_with_obs(str(path))
        return self._result


def polygon_grid_samples(
    ctx: dict[str, Any],
    polygon: list[tuple[float, float]],
    *,
    max_cells: int = 800,
) -> list[tuple[float, float, float]]:
    """Sample (x, z, vp) inside polygon on velocity grid (X = 多边形 ∩ 剖面范围)."""
    grid = ctx.get("grid_data")
    pe = ctx.get("profile_extractor")
    sample = ctx.get("sample_primary_velocity")
    if grid is None or pe is None or sample is None:
        return []
    from petrology.imodel_bridge.crust_geometry import polygon_effective_x_range

    prof_x0 = ctx.get("x_min_km")
    prof_x1 = ctx.get("x_max_km")
    prof_ok = (
        prof_x0 is not None
        and prof_x1 is not None
        and float(prof_x1) > float(prof_x0)
    )
    try:
        x_lo, x_hi = polygon_effective_x_range(
            polygon,
            x_min_km=float(prof_x0) if prof_ok else None,
            x_max_km=float(prof_x1) if prof_ok else None,
        )
    except ValueError:
        return []
    x_coord = pe.x_coord
    z_coord = pe.z_coord
    xs = np.asarray(grid.coords[x_coord].values, dtype=float)
    zs = np.asarray(grid.coords[z_coord].values, dtype=float)
    poly = MplPath(np.asarray(polygon, dtype=float))
    pz = [p[1] for p in polygon]
    z_lo, z_hi = min(pz), max(pz)
    xi = xs[(xs >= x_lo) & (xs <= x_hi)]
    zi = zs[(zs >= z_lo) & (zs <= z_hi)]
    if xi.size == 0 or zi.size == 0:
        return []
    step = max(1, int(np.ceil((xi.size * zi.size) / max_cells)))
    out: list[tuple[float, float, float]] = []
    stride = max(1, int(np.sqrt(step)))
    for x in xi[::stride]:
        for z in zi[::stride]:
            if poly.contains_point((float(x), float(z))):
                try:
                    vp = float(sample(float(x), float(z)))
                    if np.isfinite(vp) and vp > 0:
                        out.append((float(x), float(z), vp))
                except Exception:
                    pass
    return out


def launch_lip_petrology_with_obs(json_path: str) -> None:
    """Start LIP Petrology GUI subprocess with observation import flag."""
    cmd = [sys.executable, "-m", "petrology.gui", "--import-obs", str(json_path)]
    try:
        subprocess.Popen(cmd, cwd=str(Path.cwd()))
    except Exception as exc:
        QMessageBox.warning(
            None,
            "启动 LIP Petrology",
            f"无法启动 GUI:\n{exc}\n\n请手动导入: {json_path}",
        )
