"""密度 / 温度 / 压力 二维剖面（对齐 Tk PropertyPanelMixin.plot_property_profile）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal, Optional

import numpy as np

from matplotlib.figure import Figure

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT

from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pyAOBS.utils import (
    calculate_density,
    calculate_pressure_from_depth,
    calculate_temperature_from_depth,
)
from pyAOBS.visualization.imodel import ProfileExtractor, PropertyCalculator

from .zelt_iface_qt import plot_zelt_interfaces_on_axes
from .ui_chrome import apply_tool_window_chrome


def _density_method_for_library(density_method: str) -> str:
    dm = str(density_method or "gardner").strip().lower()
    if PropertyCalculator.uses_tomo2d_sediment_density(dm):
        return "gardner"
    return dm


def build_property_grid(
    grid_data,
    extractor: ProfileExtractor,
    prop_calc: PropertyCalculator | None,
    prop: Literal["density", "temperature", "pressure"],
    *,
    density_method: str,
    seafloor_depths: np.ndarray | None = None,
    basement_depths: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    z_coord = extractor.z_coord
    x_coord = extractor.x_coord
    z_vals = np.asarray(grid_data[z_coord].values, dtype=float)
    x_vals = np.asarray(grid_data[x_coord].values, dtype=float)
    vel_var = extractor.velocity_var
    velocity_data = np.asarray(grid_data[vel_var].values, dtype=float)
    nz, nx = velocity_data.shape

    water_density = (
        prop_calc.seawater_density_g_cm3 if prop_calc is not None else 1.03
    )
    water_temp = prop_calc.seawater_temperature_c if prop_calc is not None else 4.0
    seafloor_temp = prop_calc.seafloor_temperature_c if prop_calc is not None else water_temp
    water_pressure_grad = (
        prop_calc.seawater_pressure_gradient_mpa_per_km if prop_calc is not None else 10.1
    )
    geothermal_gradient = prop_calc.geothermal_gradient_c_per_km if prop_calc is not None else 30.0

    if seafloor_depths is None:
        seafloor_depths = np.zeros(nx, dtype=float)
    else:
        seafloor_depths = np.asarray(seafloor_depths, dtype=float).reshape(-1)[:nx]

    if basement_depths is None:
        basement_depths = np.full(nx, np.nan, dtype=float)
    else:
        basement_depths = np.asarray(basement_depths, dtype=float).reshape(-1)[:nx]

    def _is_water(vp_val: float, z_val: float, seafloor_val: float) -> bool:
        if prop_calc is not None:
            return prop_calc.is_water_zone(vp=vp_val, z=z_val, seafloor_depth=float(seafloor_val))
        return z_val < seafloor_val and abs(vp_val - 1.5) <= 0.25

    property_grid = np.zeros_like(velocity_data, dtype=float)

    if prop == "density":
        lib = _density_method_for_library(density_method)
        with np.errstate(invalid="ignore"):
            property_grid = calculate_density(velocity_data, method=lib)
        property_grid = np.asarray(property_grid, dtype=float)

        water_mask = np.zeros((nz, nx), dtype=bool)
        for i in range(nz):
            z = float(z_vals[i])
            for j in range(nx):
                vp = float(velocity_data[i, j])
                seafloor_z = float(seafloor_depths[j]) if j < len(seafloor_depths) else 0.0
                if _is_water(vp, z, seafloor_z):
                    property_grid[i, j] = water_density
                    water_mask[i, j] = True

        if PropertyCalculator.uses_tomo2d_sediment_density(density_method):
            sed_mask = np.zeros((nz, nx), dtype=bool)
            for j in range(nx):
                seafloor_z = float(seafloor_depths[j]) if j < len(seafloor_depths) else 0.0
                basement_z = float(basement_depths[j]) if j < len(basement_depths) else np.nan
                if not np.isfinite(basement_z) or basement_z <= seafloor_z:
                    continue
                z_col = np.asarray(z_vals, dtype=float)
                sed_col = (z_col >= seafloor_z) & (z_col < basement_z)
                sed_mask[:, j] = sed_col
            sed_mask = sed_mask & (~water_mask)
            if np.any(sed_mask):
                property_grid[sed_mask] = 1.0 + 1.18 * np.power(
                    np.maximum(np.asarray(velocity_data, dtype=float)[sed_mask] - 1.5, 0.0),
                    0.22,
                )
                property_grid[sed_mask] = np.minimum(property_grid[sed_mask], 2.6)

        title = "Density Model (g/cm³)"
        cmap = "viridis"

    elif prop == "temperature":
        for i in range(nz):
            z = float(z_vals[i])
            for j in range(nx):
                vp = float(velocity_data[i, j])
                seafloor_z = float(seafloor_depths[j]) if j < len(seafloor_depths) else 0.0
                if _is_water(vp, z, seafloor_z):
                    property_grid[i, j] = water_temp
                else:
                    property_grid[i, j] = calculate_temperature_from_depth(
                        z,
                        temperature_gradient=geothermal_gradient,
                        surface_temperature=seafloor_temp,
                        seafloor_depth=seafloor_z,
                    )
        title = "Temperature Model (°C, from seafloor)"
        cmap = "hot"

    elif prop == "pressure":
        pressure_gradient = 30.0
        for i in range(nz):
            z = float(z_vals[i])
            for j in range(nx):
                vp = float(velocity_data[i, j])
                seafloor_z = float(seafloor_depths[j]) if j < len(seafloor_depths) else 0.0
                if _is_water(vp, z, seafloor_z):
                    property_grid[i, j] = max(0.0, z) * water_pressure_grad
                else:
                    hydro_at_seafloor = max(0.0, float(seafloor_z)) * water_pressure_grad
                    rock_increment = calculate_pressure_from_depth(
                        z,
                        pressure_gradient=pressure_gradient,
                        seafloor_depth=seafloor_z,
                    )
                    property_grid[i, j] = hydro_at_seafloor + float(rock_increment)
        title = "Pressure Model (MPa, from seafloor)"
        cmap = "plasma"
    else:
        raise ValueError(prop)

    vmin = float(np.nanmin(property_grid))
    vmax = float(np.nanmax(property_grid))
    return x_vals, z_vals, property_grid, title, cmap


def open_property_grid_window(
    parent: QWidget,
    grid_data,
    extractor: ProfileExtractor,
    prop_calc: PropertyCalculator | None,
    prop: str,
    density_method: str,
    *,
    seafloor_depths_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    basement_depths_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    zelt_model: Any = None,
    basement_interface_idx: Optional[int] = None,
    seafloor_interface_idx: Optional[int] = None,
    moho_interface_idx: Optional[int] = None,
) -> None:
    prop = str(prop).lower().strip()
    if prop not in {"density", "temperature", "pressure"}:
        QMessageBox.critical(parent, "Error", f"Unknown property: {prop}")
        return

    nx = int(grid_data.sizes.get(extractor.x_coord, 0))
    x_vals = np.asarray(grid_data.coords[extractor.x_coord].values, dtype=float)
    if seafloor_depths_fn is not None:
        sf = np.asarray(seafloor_depths_fn(x_vals), dtype=float).reshape(-1)
        if sf.size < nx:
            sf = np.pad(sf, (0, nx - sf.size), constant_values=0.0)
        sf = sf[:nx]
    else:
        sf = np.zeros(nx, dtype=float)
    if basement_depths_fn is not None:
        bf = np.asarray(basement_depths_fn(x_vals), dtype=float).reshape(-1)
        if bf.size < nx:
            bf = np.pad(bf, (0, nx - bf.size), constant_values=np.nan)
        bf = bf[:nx]
    else:
        bf = np.full(nx, np.nan)

    hook_note = ""
    if seafloor_depths_fn is not None or basement_depths_fn is not None:
        hook_note = "\n(seafloor/basement from v.in selections + velocity auto-detect)"

    try:
        x_vals, z_vals, pgrid, title, cmap = build_property_grid(
            grid_data,
            extractor,
            prop_calc,
            prop,  # type: ignore[arg-type]
            density_method=density_method,
            seafloor_depths=sf,
            basement_depths=bf,
        )
    except Exception as e:
        QMessageBox.critical(parent, "Error", str(e))
        return

    fig = Figure(figsize=(10, 3.8))
    ax = fig.add_subplot(111)
    vmin = float(np.nanmin(pgrid))
    vmax = float(np.nanmax(pgrid))
    im = ax.pcolormesh(
        x_vals, z_vals, pgrid, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(title.split("(")[0].strip())
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_title(title + hook_note)
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.3)

    interface_artists: list = []

    def _clear_interfaces() -> None:
        for artist in interface_artists:
            try:
                artist.remove()
            except Exception:
                pass
        interface_artists.clear()

    def _draw_interfaces() -> None:
        _clear_interfaces()
        if zelt_model is None:
            return
        interface_artists.extend(
            plot_zelt_interfaces_on_axes(
                ax,
                zelt_model,
                basement_interface_idx=basement_interface_idx,
                seafloor_interface_idx=seafloor_interface_idx,
                moho_interface_idx=moho_interface_idx,
            )
        )

    win = QMainWindow(parent)
    win.setWindowTitle(title)
    win.resize(1000, 500)
    cw = QWidget()
    win.setCentralWidget(cw)
    lay = QVBoxLayout(cw)
    row = QHBoxLayout()
    btn = QPushButton("Save Figure")
    row.addWidget(btn)
    chk_iface: QCheckBox | None = None
    if zelt_model is not None:
        chk_iface = QCheckBox("Show Interfaces (v.in layers)")
        chk_iface.setChecked(True)
        row.addWidget(chk_iface)

        def _on_iface_toggled(_state: int) -> None:
            if chk_iface is not None and chk_iface.isChecked():
                _draw_interfaces()
            else:
                _clear_interfaces()
            canvas.draw()

        chk_iface.stateChanged.connect(_on_iface_toggled)
        _draw_interfaces()
    row.addStretch(1)
    lay.addLayout(row)
    canvas = FigureCanvasQTAgg(fig)
    lay.addWidget(NavigationToolbar2QT(canvas, cw))
    lay.addWidget(canvas)

    def _save() -> None:
        from .file_dialogs_qt import normalize_save_path_for_filter

        path, selected = QFileDialog.getSaveFileName(
            win, "Save Figure", str(Path.cwd()), "PNG (*.png);;PDF (*.pdf);;All (*.*)"
        )
        if path:
            path = normalize_save_path_for_filter(path, selected)
            try:
                fig.savefig(path, dpi=300, bbox_inches="tight")
            except Exception as ex:
                QMessageBox.critical(win, "Error", str(ex))

    btn.clicked.connect(_save)
    apply_tool_window_chrome(win)
    win.show()
