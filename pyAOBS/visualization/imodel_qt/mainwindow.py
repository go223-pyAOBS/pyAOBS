"""imodel Qt 主窗口（剖面、物性、日志）。"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib

_candidate_backends = ("qtagg", "QtAgg", "Qt5Agg")
for _b in _candidate_backends:
    try:
        matplotlib.use(_b)
        break
    except ValueError:
        continue


from matplotlib.figure import Figure

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QShowEvent
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pyAOBS.utils import (
    calculate_temperature_from_depth,
    calculate_vp_from_vs_brocher,
    calculate_vs,
)
from pyAOBS.visualization.imodel import ProfileExtractor, PropertyCalculator
from pyAOBS.visualization.gravity_obs_grid import (
    DEFAULT_GRAVITY_OBS_FILENAME,
    visualization_package_data_dir,
)

from .canvas import redraw_velocity_axes
from .model_load import load_velocity_grid
from .profile_ops import averaged_vertical_profile
from .property_format import json_compatible, property_result_row_dict
from .property_verbose import (
    iter_manual_calculate_lines,
    iter_point_property_lines,
    load_zone_label_map,
)
from .mpl_selection import MplWorkbenchSelection
from .rock_scatter_qt import (
    RockScatterToolWindow,
    scatter_model_points_holder,
    get_or_create_rock_scatter_window,
)
from .property_grid_qt import open_property_grid_window
from .gravity_toolbox_qt import GravityToolboxQt
from .workbench_state_qt import (
    gui_state_file_from_env,
    load_imodel_section,
    resolve_restorable_input_path,
    run_inputs_dir_from_env,
    save_imodel_section,
)
from .zelt_iface_qt import (
    compute_seafloor_depth_map,
    interpolate_interface_depth_at_x,
    interpolate_interface_depths_on_grid,
)
from .file_dialogs_qt import normalize_save_path_for_filter
from .interface_files_qt import interfaces_xy_equal, parse_interface_file
from .petrology_export_qt import PetrologyExportDialog
from .hvp_overlay_preview_qt import show_hvp_overlay_preview
from .transect_export_qt import TransectExportDialog
from .scroll_safe_widgets import ComboBoxNoScrollUnlessFocused
from .styles import (
    apply_imodel_chrome,
    apply_imodel_font,
    compact_status_panel,
    polish_status_bar_label,
    primary_button,
    section_title,
    set_path_badge,
    show_modeless_dialog,
    status_bar_message_label,
)

from petrology.imodel_bridge.export_contract import CrustObservation, load_observation_json
from petrology.seismic.transect import TransectWindow, load_transect_windows_csv


USER_GUIDE_TEXT = """
Interactive Velocity Model Viewer - User Guide

1. File Operations:
   - Load Vp / Load Vs: Load velocity models (top toolbar)
   - Save Figure: Save current displayed figure
   - Export Results: Export profiles and point calculation results (CSV/JSON)

2. Interactive Tools:
   - Point Selection: Left-click to add, Right-click to remove, D to delete last, C to clear all
   - Polygon Selection: Left-click to add vertices, Right-click to complete polygon

3. Profile Extraction:
   - Enter X coordinate to extract vertical profile
   - Enter X range and sampling interval to extract 1D vertical profile

4. Property Calculation:
   - Enter coordinates to calculate all property parameters at that point

5. Shortcuts:
   - Ctrl+O: Open model
"""

ABOUT_TEXT = """
Interactive Velocity Model Viewer

Version: 1.0
Author: Haibo Huang

Features:
- Visualize velocity models
- Interactive point/polygon selection
- Extract 1D profiles
- Calculate property parameters (density, pressure, rock type, gravity, etc.)

Qt viewer note:
- Session JSON (PYAOBS_GUI_STATE_FILE), v.in interfaces, seafloor/basement depth for
  properties & property grids are integrated; rock scatter includes Vp/Vs–Vp DEM
 overlays (Tk parity) + gravity Compare Methods + basement-aware averaged profiles.
"""

PARAMS_PANEL_MIN_WIDTH = 440


class VerticalProfileDialog(QDialog):
    """显示平均垂直剖面 Vp–深度，可导出 CSV。"""

    def __init__(
        self,
        parent: QWidget,
        profile: pd.DataFrame,
        *,
        depth_ylabel: str = "Depth (km)",
        plot_title: str = "Averaged vertical profile",
    ) -> None:
        super().__init__(parent)
        self.setModal(False)
        self.setWindowTitle("垂直剖面（平均）")
        self.resize(560, 480)
        self._profile = profile

        lay = QVBoxLayout(self)

        fig = Figure(figsize=(5, 4))
        canvas = FigureCanvasQTAgg(fig)
        tb = NavigationToolbar2QT(canvas, self)
        lay.addWidget(tb)
        lay.addWidget(canvas)

        ax = fig.add_subplot(111)
        ax.plot(profile["vp"].values, profile["depth"].values, "b-", lw=1.2)
        ax.invert_yaxis()
        ax.set_xlabel("Vp (km/s)")
        ax.set_ylabel(depth_ylabel)
        ax.set_title(plot_title)
        ax.grid(True, alpha=0.3)

        bbox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close
        )
        bbox.button(QDialogButtonBox.StandardButton.Save).clicked.connect(self._save_csv)  # type: ignore[union-attr]
        bbox.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.reject)  # type: ignore[union-attr]
        lay.addWidget(bbox)

    def _save_csv(self) -> None:
        path, _fl = QFileDialog.getSaveFileName(
            self, "导出 CSV", str(Path.cwd()), "CSV (*.csv);;All (*.*)"
        )
        if not path:
            return
        try:
            if not str(path).lower().endswith(".csv"):
                path = path + ".csv"
            self._profile.to_csv(path, index=False)
            QMessageBox.information(self, "完成", f"已保存:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "失败", str(e))


# Calculation log: non-detailed mode keeps point summary fields only.
_LOG_COMPACT_DETAIL_PREFIXES = (
    "  P-wave velocity:",
    "  S-wave velocity:",
    "  Vp/Vs ratio:",
    "  Density (",
    "  Density:",
    "  Pressure:",
    "  Temperature:",
    "  Zone:",
    "  Warning: Vp/Vs",
)


class ImodelQtMainWindow(QMainWindow):
    """主界面：速度图 + 侧边工具 + 日志。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Interactive Velocity Model Viewer (Qt)")
        self.resize(1360, 900)

        self._gui_state_file = gui_state_file_from_env()
        self._run_inputs_dir = run_inputs_dir_from_env()
        self._current_model_file = ""
        self._restoring_state = False
        self._zelt_model = None
        self._basement_interface_idx: Optional[int] = None
        self._seafloor_interface_idx: Optional[int] = None
        self._moho_interface_idx: Optional[int] = None
        self._seafloor_depth_map: Optional[np.ndarray] = None
        self._loaded_interfaces: list[dict[str, Any]] = []
        self._basement_interface_data: Optional[dict[str, Any]] = None
        self._moho_interface_data: Optional[dict[str, Any]] = None
        self._basement_interface_file = ""

        self._profiles: dict[str, pd.DataFrame] = {}
        self._profile_counter = 0
        self._zone_labels = load_zone_label_map()

        self._cbar = None
        self._grid_data = None
        self._vs_grid_data = None
        self._property_result_rows: list[dict[str, Any]] = []
        self._profile_extractor: ProfileExtractor | None = None
        self._prop_calc: PropertyCalculator | None = None
        self._classifier_diag_logged = False

        self._grav_obs_data_dir = str(visualization_package_data_dir())
        self._grav_obs_filename = DEFAULT_GRAVITY_OBS_FILENAME
        self._grav_obs_overlay_enabled = False
        self._grav_profile_lon_lat_csv = ""

        self._last_petrology_obs: CrustObservation | None = None
        self._petrology_obs_json_path = ""
        self._petrology_f_lower: float | None = None
        self._petrology_export_x_km = 0.0
        self._petrology_transect_windows_path = ""
        self._last_transect_windows: list[TransectWindow] = []
        self._log_all_lines: list[str] = []

        outer = QWidget()
        self.setCentralWidget(outer)
        out_lay = QVBoxLayout(outer)
        out_lay.setContentsMargins(6, 6, 6, 6)

        top_h = QSplitter(Qt.Orientation.Horizontal)
        self._top_h = top_h
        plot_panel = QWidget()
        plot_panel.setObjectName("ImodelPlotPanel")
        pl = QVBoxLayout(plot_panel)
        pl.setContentsMargins(6, 6, 6, 6)
        self._fig = Figure(figsize=(10, 3.8))
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax = self._fig.add_subplot(111)
        tb_row = QWidget()
        tb_lay = QHBoxLayout(tb_row)
        tb_lay.setContentsMargins(0, 0, 0, 0)
        tb_lay.addWidget(NavigationToolbar2QT(self._canvas, tb_row))
        pl.addWidget(tb_row)
        pl.addWidget(self._canvas, stretch=1)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side = QWidget()
        side.setObjectName("ImodelSidePanel")
        self._side = side
        side_lay = QVBoxLayout(side)
        side_lay.setContentsMargins(3, 3, 3, 3)
        side_lay.setSpacing(2)

        def _side_btn(text: str, slot, tip: str = "") -> QPushButton:
            b = QPushButton(text)
            if tip:
                b.setToolTip(tip)
            b.clicked.connect(slot)  # type: ignore[arg-type]
            return b

        tools_bar = QWidget()
        tools_bar.setObjectName("ImodelToolsBar")
        tools_bar.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        tl = QHBoxLayout(tools_bar)
        tl.setContentsMargins(4, 4, 4, 4)
        tl.setSpacing(4)
        tools_lbl = QLabel("Tools")
        tools_lbl.setObjectName("ImodelToolsCaption")
        tl.addWidget(tools_lbl)
        for txt, slot, tip in (
            ("Point", self._enable_point_selection, "Point Selection"),
            ("Poly", self._enable_polygon_selection, "Polygon Selection"),
            ("Clear", self._clear_selections, "Clear all selections"),
        ):
            tl.addWidget(_side_btn(txt, slot, tip), stretch=1)
        side_lay.addWidget(tools_bar)

        grp_prof = QGroupBox("Profile")
        pf = QVBoxLayout(grp_prof)
        pf.setContentsMargins(3, 2, 3, 3)
        pf.setSpacing(2)
        xrow = QHBoxLayout()
        xrow.setSpacing(2)
        xrow.addWidget(QLabel("X:"))
        self._x_min_edit = QLineEdit()
        self._x_max_edit = QLineEdit()
        self._x_min_edit.setMaximumWidth(52)
        self._x_max_edit.setMaximumWidth(52)
        xrow.addWidget(self._x_min_edit)
        xrow.addWidget(QLabel("–"))
        xrow.addWidget(self._x_max_edit)
        xrow.addWidget(QLabel("dx:"))
        self._dx_combo = ComboBoxNoScrollUnlessFocused()
        self._dx_combo.addItems(["0.5", "1.0"])
        self._dx_combo.setMaximumWidth(48)
        xrow.addWidget(self._dx_combo)
        self._interfaces_chk = QCheckBox("v.in layers")
        self._interfaces_chk.setEnabled(False)
        self._interfaces_chk.toggled.connect(self._on_interfaces_toggled)
        xrow.addWidget(self._interfaces_chk)
        xrow.addStretch(1)
        pf.addLayout(xrow)
        iface_row = QHBoxLayout()
        iface_row.setSpacing(3)
        self._basement_combo = ComboBoxNoScrollUnlessFocused()
        self._seafloor_combo = ComboBoxNoScrollUnlessFocused()
        self._moho_combo = ComboBoxNoScrollUnlessFocused()
        self._basement_combo.setToolTip("B — Basement interface（沉积基底）")
        self._seafloor_combo.setToolTip("S — Seafloor interface（海底面）")
        self._moho_combo.setToolTip("M — Moho interface（地壳底 / 火成柱底界）")
        self._basement_combo.currentIndexChanged.connect(self._on_basement_combo_changed)
        self._seafloor_combo.currentIndexChanged.connect(self._on_seafloor_combo_changed)
        self._moho_combo.currentIndexChanged.connect(self._on_moho_combo_changed)
        for label, combo in (("B:", self._basement_combo), ("S:", self._seafloor_combo), ("M:", self._moho_combo)):
            iface_row.addWidget(QLabel(label))
            iface_row.addWidget(combo, stretch=1)
        pf.addLayout(iface_row)
        iface_file_row = QHBoxLayout()
        iface_file_row.setSpacing(3)
        self._interface_file_label = QLabel("None")
        set_path_badge(self._interface_file_label, active=False, text="None")
        btn_iface_load = _side_btn("Load", self._load_interface_file, "Load interface file")
        btn_iface_save = _side_btn("Save", self._save_interface_file, "Save interface file")
        btn_iface_clear = _side_btn("Clr", self._clear_loaded_interfaces, "Clear loaded interfaces")
        iface_file_row.addWidget(self._interface_file_label, stretch=1)
        iface_file_row.addWidget(btn_iface_load)
        iface_file_row.addWidget(btn_iface_save)
        iface_file_row.addWidget(btn_iface_clear)
        btn_x = _side_btn("Profile", self._on_vertical_profile, "Extract vertical profile")
        iface_file_row.addWidget(btn_x)
        pf.addLayout(iface_file_row)
        side_lay.addWidget(grp_prof)

        grp_pet = QGroupBox("Petrology")
        grp_pet.setObjectName("ImodelBridgeGroup")
        pet_g = QGridLayout(grp_pet)
        pet_g.setContentsMargins(4, 2, 4, 4)
        pet_g.setHorizontalSpacing(3)
        pet_g.setVerticalSpacing(3)
        btn_pet = primary_button("Export…")
        btn_pet.setToolTip("单点 (H, V_LC) → JSON → LIP 反演锚点（一个 x）")
        btn_pet.clicked.connect(self._open_petrology_export)
        btn_pet_quick = primary_button("Quick")
        btn_pet_quick.setToolTip("界面自动 H，保存并启动 LIP GUI")
        btn_pet_quick.clicked.connect(self._quick_send_to_petrology)
        btn_tran = primary_button("Transect…")
        btn_tran.setToolTip("沿迹滑窗 Fig.15 (a–c)：多距离窗 + MC + windows CSV")
        btn_tran.clicked.connect(self._open_transect_export)
        btn_hvp = _side_btn("H–Vp图", self._preview_hvp_overlay, "Fig.12a 底图 + 单点/沿迹投点")
        pet_g.addWidget(btn_pet, 0, 0)
        pet_g.addWidget(btn_pet_quick, 0, 1)
        pet_g.addWidget(btn_tran, 0, 2)
        pet_g.addWidget(btn_hvp, 1, 0, 1, 3)
        self._bridge_status = compact_status_panel("观测: (未导出)\n沿迹: (未导出)")
        pet_g.addWidget(self._bridge_status, 2, 0, 1, 3)
        side_lay.addWidget(grp_pet)

        grp_ms = QGroupBox("Model")
        ms = QVBoxLayout(grp_ms)
        ms.setContentsMargins(3, 2, 3, 3)
        ms.setSpacing(2)
        self._wave_display = ComboBoxNoScrollUnlessFocused()
        self._wave_display.addItems(["P-wave (Vp)", "S-wave (Vs)"])
        self._wave_display.setCurrentIndex(0)
        self._model_type = "vp"
        self._wave_display.currentIndexChanged.connect(self._on_wave_type_changed)
        self._vel_method = ComboBoxNoScrollUnlessFocused()
        self._vel_method.addItems(["brocher", "castagna"])
        self._density_method_combo = ComboBoxNoScrollUnlessFocused()
        self._density_method_combo.addItems(
            ["gardner", "brocher", "nafe_drake", "tomo2d_sediment"]
        )
        self._contours_chk = QCheckBox("Contours")
        self._contours_chk.toggled.connect(self._toggle_contours)
        wave_row = QHBoxLayout()
        wave_row.setSpacing(3)
        wave_row.addWidget(QLabel("Wave"))
        wave_row.addWidget(self._wave_display, stretch=1)
        wave_row.addWidget(QLabel("Vs"))
        wave_row.addWidget(self._vel_method, stretch=1)
        ms.addLayout(wave_row)
        dens_row = QHBoxLayout()
        dens_row.setSpacing(3)
        dens_row.addWidget(QLabel("ρ"))
        dens_row.addWidget(self._density_method_combo, stretch=1)
        dens_row.addWidget(self._contours_chk)
        ms.addLayout(dens_row)
        side_lay.addWidget(grp_ms)

        grp_pr = QGroupBox("Point props")
        pr = QHBoxLayout(grp_pr)
        pr.setContentsMargins(4, 2, 4, 4)
        pr.setSpacing(3)
        pr.addWidget(QLabel("X"))
        self._px = QLineEdit()
        self._px.setMaximumWidth(56)
        pr.addWidget(self._px)
        pr.addWidget(QLabel("Z"))
        self._pz = QLineEdit()
        self._pz.setMaximumWidth(56)
        pr.addWidget(self._pz)
        btn_calc = _side_btn("Calc", self._on_calculate_properties, "Calculate properties at (X, Z)")
        pr.addWidget(btn_calc)
        pr.addStretch(1)
        side_lay.addWidget(grp_pr)

        grp_rock = QGroupBox("Rocks")
        rg = QGridLayout(grp_rock)
        rg.setContentsMargins(3, 2, 3, 3)
        rg.setHorizontalSpacing(2)
        rg.setVerticalSpacing(2)
        b_show_vp = _side_btn("Vp-Vs图", lambda: get_or_create_rock_scatter_window(
            self, "vp_vs", self._rock_scatter_ctx, self._append_log, refresh_if_visible=True
        ), "Show Vp-Vs scatter")
        b_show_ra = _side_btn("Vp/Vs-H图", lambda: get_or_create_rock_scatter_window(
            self, "ratio", self._rock_scatter_ctx, self._append_log, refresh_if_visible=True
        ), "Show Vp/Vs vs Vp plot")
        b_cl_vp = _side_btn("Clr", self._rock_clear_vp_vs, "Clear Vp-Vs model points")
        b_cl_ra = _side_btn("Clr", self._rock_clear_ratio, "Clear Vp/Vs-H model points")
        rg.addWidget(b_show_vp, 0, 0)
        rg.addWidget(b_cl_vp, 0, 1)
        rg.addWidget(b_show_ra, 0, 2)
        rg.addWidget(b_cl_ra, 0, 3)
        b_ap_vp = _side_btn("Add Points→Vp-Vs图", self._rock_add_vp_vs_points, "Add selected points to Vp-Vs plot")
        b_ap_ra = _side_btn("Add Points→Vp/Vs-H图", self._rock_add_ratio_points, "Add selected points to Vp/Vs-H plot")
        rg.addWidget(b_ap_vp, 1, 0, 1, 2)
        rg.addWidget(b_ap_ra, 1, 2, 1, 2)
        b_po_vp = _side_btn("Poly Avg→Vp-Vs图", self._rock_add_vp_vs_poly_avg, "Polygon average → Vp-Vs plot")
        b_po_ra = _side_btn("Poly Avg→Vp/Vs-H图", self._rock_add_ratio_poly_avg, "Polygon average → Vp/Vs-H plot")
        rg.addWidget(b_po_vp, 2, 0, 1, 2)
        rg.addWidget(b_po_ra, 2, 2, 1, 2)
        b_ps_vp = _side_btn("Poly Samp→Vp-Vs图", self._rock_add_vp_vs_poly_samples, "Polygon samples → Vp-Vs plot")
        b_ps_ra = _side_btn("Poly Samp→Vp/Vs-H图", self._rock_add_ratio_poly_samples, "Polygon samples → Vp/Vs-H plot")
        rg.addWidget(b_ps_vp, 3, 0, 1, 2)
        rg.addWidget(b_ps_ra, 3, 2, 1, 2)
        rg.setColumnStretch(0, 1)
        rg.setColumnStretch(2, 1)
        side_lay.addWidget(grp_rock)

        right_scroll.setWidget(side)
        self._right_scroll = right_scroll

        self._params_toggle = QToolButton()
        self._params_toggle.setObjectName("ImodelParamsToggle")
        self._params_toggle.setArrowType(Qt.ArrowType.LeftArrow)
        self._params_toggle.setFixedWidth(22)
        self._params_toggle.setAutoRaise(False)
        self._params_toggle.setToolTip("Hide parameters panel")
        self._params_toggle.clicked.connect(self._on_params_toggle_clicked)

        params_container = QWidget()
        params_row = QHBoxLayout(params_container)
        params_row.setContentsMargins(0, 0, 0, 0)
        params_row.setSpacing(0)
        params_row.addWidget(self._params_toggle)
        params_row.addWidget(right_scroll, stretch=1)
        self._params_container = params_container

        self._params_visible = True
        self._saved_params_split: int | None = None

        log_panel = QWidget()
        log_panel.setObjectName("ImodelLogPanel")
        log_lay = QVBoxLayout(log_panel)
        log_lay.setContentsMargins(8, 8, 8, 8)
        log_lay.addWidget(section_title("Calculation Results"))
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(120)
        self._log.setAcceptRichText(False)
        self._log_show_detailed = False
        log_lay.addWidget(self._log, stretch=1)
        log_btn_row = QHBoxLayout()
        clr_btn = QPushButton("Clear log")
        clr_btn.clicked.connect(self._clear_log)
        log_btn_row.addWidget(clr_btn)
        self._log_detailed_chk = QCheckBox("Detailed")
        self._log_detailed_chk.setChecked(False)
        self._log_detailed_chk.setToolTip("Off: point summary; On: full property details.")
        self._log_detailed_chk.toggled.connect(self._on_log_detail_toggled)
        log_btn_row.addWidget(self._log_detailed_chk)
        log_btn_row.addStretch(1)
        log_lay.addLayout(log_btn_row)

        left_split = QSplitter(Qt.Orientation.Vertical)
        self._left_split = left_split
        left_split.addWidget(plot_panel)
        left_split.addWidget(log_panel)
        left_split.setStretchFactor(0, 5)
        left_split.setStretchFactor(1, 1)
        left_split.setSizes([720, 160])

        top_h.addWidget(left_split)
        top_h.addWidget(params_container)
        top_h.setStretchFactor(0, 3)
        top_h.setStretchFactor(1, 0)

        out_lay.addWidget(top_h)

        self._grav_win: GravityToolboxQt | None = None
        self._mpl_sel = MplWorkbenchSelection(
            self._ax, self._canvas, redraw_plot=self._redraw_main_plot_clearing_overlays
        )

        self._setup_menus()
        self._build_toolbar()
        self._build_status_bar()
        apply_imodel_chrome(self)
        QTimer.singleShot(0, self._restore_workbench_state)
        QTimer.singleShot(0, lambda: self._apply_top_splitter_sizes(expand=True))

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, lambda: self._apply_top_splitter_sizes(expand=True))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._params_visible:
            QTimer.singleShot(0, lambda: self._apply_top_splitter_sizes(expand=False))

    def _apply_top_splitter_sizes(self, *, expand: bool = True) -> None:
        if self._top_h.width() <= 0:
            return
        toggle_w = self._params_toggle.width()
        total = self._top_h.width()
        if not self._params_visible:
            self._top_h.setSizes([max(1, total - toggle_w), toggle_w])
            return

        min_params_w = self._params_panel_ideal_width()
        if self._saved_params_split is not None:
            min_params_w = max(min_params_w, self._saved_params_split - toggle_w)
        self._side.setMinimumWidth(min_params_w)
        self._right_scroll.setMinimumWidth(min_params_w)
        min_params_total = min_params_w + toggle_w

        sizes = self._top_h.sizes()
        current_params = sizes[1] if len(sizes) >= 2 else 0
        if expand or current_params < min_params_total:
            plot_w = max(320, total - min_params_total)
            self._top_h.setSizes([plot_w, min_params_total])

    def _on_params_toggle_clicked(self) -> None:
        toggle_w = self._params_toggle.width()
        total = sum(self._top_h.sizes()) or self._top_h.width()
        if self._params_visible:
            sizes = self._top_h.sizes()
            if len(sizes) >= 2 and sizes[1] > toggle_w:
                self._saved_params_split = sizes[1]
            self._params_visible = False
            self._right_scroll.hide()
            self._params_toggle.setArrowType(Qt.ArrowType.RightArrow)
            self._params_toggle.setToolTip("Show parameters panel")
            self._top_h.setSizes([max(1, total - toggle_w), toggle_w])
            return

        self._params_visible = True
        self._right_scroll.show()
        self._params_toggle.setArrowType(Qt.ArrowType.LeftArrow)
        self._params_toggle.setToolTip("Hide parameters panel")
        self._apply_top_splitter_sizes(expand=True)

    def _params_panel_ideal_width(self) -> int:
        """侧栏内容完整显示所需宽度（紧凑双列布局）。"""
        self._side.adjustSize()
        hint = max(self._side.sizeHint().width(), self._side.minimumSizeHint().width())
        return max(int(hint) + 16, PARAMS_PANEL_MIN_WIDTH)

    def _build_toolbar(self) -> None:
        tb = QToolBar("主工具栏")
        tb.setMovable(False)
        self.addToolBar(tb)
        title = QLabel("Interactive Velocity Model (imodel)")
        title.setObjectName("ImodelAppTitle")
        tb.addWidget(title)
        for text, slot in (
            ("Load Vp", self._open_vp_model),
            ("Load Vs", self._open_vs_model),
            ("Save Figure", self._save_figure),
            ("Export Results", self._export_results),
        ):
            act = QAction(text, self)
            act.triggered.connect(slot)
            tb.addAction(act)
        tb.addSeparator()
        act_grav = QAction("Gravity Toolbox", self)
        act_grav.triggered.connect(self._open_gravity_toolbox)
        tb.addAction(act_grav)
        tb.addSeparator()
        for lbl, prop in (
            ("Density Profile", "density"),
            ("Temperature Profile", "temperature"),
            ("Pressure Profile", "pressure"),
        ):
            act = QAction(lbl, self)
            act.triggered.connect(lambda checked=False, p=prop: self._open_prop_grid(p))
            tb.addAction(act)

    def _build_status_bar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._status_msg = status_bar_message_label("就绪", "info")
        sb.addWidget(self._status_msg, stretch=1)

    def _set_status(self, text: str, level: str = "info") -> None:
        self._status_msg.setText(text)
        polish_status_bar_label(self._status_msg, level)

    def _flash_status(self, text: str, level: str = "success", ms: int = 5000) -> None:
        self._set_status(text, level)
        QTimer.singleShot(ms, lambda: self._set_status("就绪", "info"))

    def _setup_menus(self) -> None:
        """对齐 imodel Tk `viewer.create_menu()` 的条目与快捷键。"""
        mb = self.menuBar()

        file_menu = mb.addMenu("File")
        act_vp = QAction("Load Vp Model", self)
        act_vp.setShortcut("Ctrl+O")
        act_vp.triggered.connect(self._open_vp_model)
        file_menu.addAction(act_vp)
        act_vs = QAction("Load Vs Model", self)
        act_vs.triggered.connect(self._open_vs_model)
        file_menu.addAction(act_vs)
        file_menu.addSeparator()
        act_save = QAction("Save Figure", self)
        act_save.triggered.connect(self._save_figure)
        file_menu.addAction(act_save)
        act_exp = QAction("Export Results", self)
        act_exp.triggered.connect(self._export_results)
        file_menu.addAction(act_exp)
        file_menu.addSeparator()
        act_exit = QAction("Exit", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        tools_menu = mb.addMenu("Tools")
        act_ps = QAction("Point Selection", self)
        act_ps.triggered.connect(self._enable_point_selection)
        tools_menu.addAction(act_ps)
        act_poly = QAction("Polygon Selection", self)
        act_poly.triggered.connect(self._enable_polygon_selection)
        tools_menu.addAction(act_poly)
        tools_menu.addSeparator()
        act_clr = QAction("Clear Selections", self)
        act_clr.triggered.connect(self._clear_selections)
        tools_menu.addAction(act_clr)
        tools_menu.addSeparator()
        act_pet = QAction("Export to LIP Petrology…", self)
        act_pet.setShortcut("Ctrl+Shift+P")
        act_pet.triggered.connect(self._open_petrology_export)
        tools_menu.addAction(act_pet)
        act_pet_prev = QAction("Preview H–Vp overlay (Fig.12a)", self)
        act_pet_prev.setShortcut("Ctrl+Shift+2")
        act_pet_prev.triggered.connect(self._preview_hvp_overlay)
        tools_menu.addAction(act_pet_prev)

        help_menu = mb.addMenu("Help")
        act_guide = QAction("User Guide", self)
        act_guide.triggered.connect(self._show_user_guide)
        help_menu.addAction(act_guide)
        act_about = QAction("About", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _on_wave_type_changed(self) -> None:
        txt = self._wave_display.currentText()
        if "P-wave" in txt or "Vp" in txt:
            self._model_type = "vp"
            self._append_log("Model type: P-wave (Vp) primary")
        else:
            self._model_type = "vs"
            self._append_log("Model type: S-wave (Vs) primary")

    def _toggle_contours(self) -> None:
        self._redraw_main_plot_clearing_overlays()

    def _redraw_main_plot_clearing_overlays(self) -> None:
        ds = self._grid_data
        self._mpl_sel._disconnect_point_selector()
        self._mpl_sel._disconnect_polygon_selector_gui()
        self._mpl_sel.forget_polygon_sample_artists()
        self._remove_colorbar()
        if ds is None:
            self._ax.clear()
            self._canvas.draw()
            return

        cmap_name = "viridis"
        vmin = vmax = None
        collections = getattr(self._ax, "collections", None)
        old_im = collections[0] if collections else None
        if old_im is not None:
            cmap = getattr(old_im, "cmap", None)
            if cmap is not None:
                cmap_name = cmap.name
            if hasattr(old_im, "get_clim"):
                try:
                    vmin, vmax = old_im.get_clim()
                except Exception:
                    vmin = vmax = None

        vel_name = (
            self._profile_extractor.velocity_var
            if self._profile_extractor
            else "velocity"
        )
        try:
            vdata = ds[vel_name].values
            if vmin is None:
                vmin = float(np.nanmin(vdata))
            if vmax is None:
                vmax = float(np.nanmax(vdata))
        except Exception:
            vmin = vmax = None

        im, _vn, vmi, vma = redraw_velocity_axes(
            self._ax,
            ds,
            cmap=str(cmap_name),
            vmin=vmin,
            vmax=vmax,
        )
        if self._contours_chk.isChecked():
            try:
                xf = ds.coords[self._profile_extractor.x_coord].values  # type: ignore[union-attr]
                zf = ds.coords[self._profile_extractor.z_coord].values  # type: ignore[union-attr]
                v2 = ds[vel_name].values
                lvmin = float(vmin) if vmin is not None else float(np.nanmin(v2))
                lvmax = float(vmax) if vmax is not None else float(np.nanmax(v2))
                levels = np.linspace(lvmin, lvmax, 10)
                cs = self._ax.contour(
                    xf, zf, v2, levels=levels, colors="white", linewidths=0.8, alpha=0.6
                )
                self._ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
            except Exception:
                pass

        try:
            self._cbar = self._fig.colorbar(im, ax=self._ax)
            self._cbar.set_label("Velocity (km/s)")
        except Exception:
            self._cbar = None
        self._maybe_plot_interfaces()
        self._canvas.draw()

    def _maybe_plot_interfaces(self) -> None:
        if not self._interfaces_chk.isChecked():
            return
        if self._grid_data is None:
            return
        if self._zelt_model is not None:
            self._plot_zelt_interfaces()
        self._plot_loaded_file_interfaces()
        self._plot_legacy_basement_interface()
        self._apply_interface_legend()

    def _apply_interface_legend(self) -> None:
        try:
            handles, labels = self._ax.get_legend_handles_labels()
            pairs = [
                (h, lbl)
                for h, lbl in zip(handles, labels)
                if lbl and not str(lbl).startswith("_")
            ]
            if not pairs:
                return
            hs = [p[0] for p in pairs]
            ls = [p[1] for p in pairs]
            if len(ls) > 6:
                hs = [hs[0], hs[-1]]
                ls = [ls[0], ls[-1]]
            self._ax.legend(hs, ls, loc="upper right", fontsize=8, framealpha=0.8)
        except Exception:
            pass

    def _plot_loaded_file_interfaces(self) -> None:
        for iface in self._loaded_interfaces:
            xv = np.asarray(iface["x"], dtype=float)
            zv = np.asarray(iface["z"], dtype=float)
            if xv.size == 0:
                continue
            nm = str(iface.get("name", "loaded"))
            self._ax.plot(
                xv,
                zv,
                "g-",
                linewidth=2.0,
                alpha=0.8,
                label=f"Loaded: {nm}",
            )

    def _plot_legacy_basement_interface(self) -> None:
        if self._basement_interface_data is None:
            return
        for iface in self._loaded_interfaces:
            if interfaces_xy_equal(iface, self._basement_interface_data):
                return
        try:
            xc = np.asarray(self._basement_interface_data["x"], dtype=float)
            zc = np.asarray(self._basement_interface_data["z"], dtype=float)
            if xc.size == 0:
                return
            self._ax.plot(
                xc,
                zc,
                "g-",
                linewidth=2.0,
                alpha=0.8,
                label="Loaded Interface",
            )
        except Exception:
            pass

    def _update_interface_file_label(self) -> None:
        n = len(self._loaded_interfaces)
        if n == 0:
            set_path_badge(self._interface_file_label, active=False, text="None")
        elif n == 1:
            p = self._basement_interface_file or ""
            set_path_badge(
                self._interface_file_label,
                active=True,
                text=Path(p).name if p else "1 interface",
            )
        else:
            set_path_badge(self._interface_file_label, active=True, text=f"{n} interfaces")

    def _load_interface_file(self) -> None:
        if self._grid_data is None:
            QMessageBox.warning(self, "Warning", "Please load velocity grid first.")
            return
        path, _fl = QFileDialog.getOpenFileName(
            self,
            "Load Interface File",
            str(Path.cwd()),
            "Text (*.txt *.dat);;CSV (*.csv);;All (*.*)",
        )
        if not path:
            return
        try:
            interfaces_in_file = parse_interface_file(path)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to load interface file:\n{e}")
            return
        for iface in interfaces_in_file:
            iface["file"] = path
            self._loaded_interfaces.append(iface)
        if interfaces_in_file:
            first = interfaces_in_file[0]
            self._basement_interface_data = {"x": first["x"], "z": first["z"]}
            self._basement_interface_file = path
        self._append_log(f"Interface file loaded: {path}")
        self._append_log(f"  Number of interfaces: {len(interfaces_in_file)}")
        for i, iface in enumerate(interfaces_in_file):
            self._append_log(
                f"    Interface {i + 1}: {iface['name']} ({len(iface['x'])} points)"
            )
        self._update_interface_file_label()
        self._invalidate_seafloor_cache()
        self._redraw_main_plot_clearing_overlays()

    def _clear_loaded_interfaces(self) -> None:
        n = len(self._loaded_interfaces)
        self._loaded_interfaces.clear()
        self._basement_interface_data = None
        self._basement_interface_file = ""
        self._update_interface_file_label()
        self._invalidate_seafloor_cache()
        self._append_log(f"Cleared {n} loaded interface(s)")
        self._redraw_main_plot_clearing_overlays()

    def _save_interface_file(self) -> None:
        if self._zelt_model is not None:
            self._save_vin_interfaces_qt()
        else:
            self._save_grid_interfaces_qt()

    def _save_vin_interfaces_qt(self) -> None:
        zm = self._zelt_model
        if zm is None:
            QMessageBox.warning(self, "Warning", "No v.in model loaded.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Save Interfaces")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Select interfaces to save:"))
        checks: list[QCheckBox] = []
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        ilay = QVBoxLayout(inner)
        n_nodes = len(zm.depth_nodes)
        for i in range(n_nodes):
            if i == n_nodes - 1:
                text = f"Bottom Boundary (Interface {i + 1})"
            elif self._basement_interface_idx is not None and i == self._basement_interface_idx:
                text = f"B — Basement Interface {i + 1} (Selected)"
            elif self._seafloor_interface_idx is not None and i == self._seafloor_interface_idx:
                text = f"S — Seafloor Interface {i + 1} (Selected)"
            elif self._moho_interface_idx is not None and i == self._moho_interface_idx:
                text = f"M — Moho Interface {i + 1} (Selected)"
            else:
                text = f"Layer {i + 1} Interface"
            cb = QCheckBox(text)
            cb.setChecked(True)
            checks.append(cb)
            ilay.addWidget(cb)
        scroll.setWidget(inner)
        lay.addWidget(scroll)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        selected = [i for i, cb in enumerate(checks) if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select at least one interface.")
            return
        path, _fl = QFileDialog.getSaveFileName(
            self,
            "Save Interfaces",
            str(Path.cwd()),
            "Text (*.txt);;All (*.*)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("# Interface file saved from v.in model\n")
                f.write(f"# Number of interfaces: {len(selected)}\n")
                f.write("# Format: x(km) z(km)\n")
                f.write("# Each interface is separated by a blank line\n\n")
                for idx in selected:
                    x_coords, z_coords = zm.get_layer_geometry(idx)
                    valid_pairs = [
                        (float(x), float(z))
                        for x, z in zip(x_coords, z_coords)
                        if not (np.isnan(x) or np.isnan(z))
                    ]
                    if not valid_pairs:
                        continue
                    if idx == len(zm.depth_nodes) - 1:
                        f.write("# Bottom Boundary\n")
                    else:
                        f.write(f"# Interface {idx + 1}\n")
                    for x, z in valid_pairs:
                        f.write(f"{x:.6f}\t{z:.6f}\n")
                    f.write("\n")
            self._append_log(f"Interfaces saved: {path} ({len(selected)} interfaces)")
            self._flash_status(Path(path).name, "success")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to save interfaces:\n{e}")

    def _save_grid_interfaces_qt(self) -> None:
        if self._grid_data is None or self._profile_extractor is None:
            QMessageBox.warning(self, "Warning", "Please load model first.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Save Interface from Grid Model")
        vbox = QVBoxLayout(dlg)
        vbox.addWidget(QLabel("Select interface source:"))
        rb_loaded = QRadioButton("Save currently loaded interface")
        rb_contour = QRadioButton("Extract from velocity contour")
        rb_loaded.setChecked(True)
        grp = QButtonGroup(dlg)
        grp.addButton(rb_loaded)
        grp.addButton(rb_contour)
        vbox.addWidget(rb_loaded)
        vbox.addWidget(rb_contour)
        row_ce = QHBoxLayout()
        row_ce.addWidget(QLabel("Velocity value (km/s):"))
        ce = QLineEdit("3.0")
        row_ce.addWidget(ce)
        vbox.addLayout(row_ce)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        vbox.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        path, _fl = QFileDialog.getSaveFileName(
            self,
            "Save Interface",
            str(Path.cwd()),
            "Text (*.txt);;All (*.*)",
        )
        if not path:
            return
        try:
            if rb_loaded.isChecked():
                if self._basement_interface_data is None:
                    QMessageBox.warning(self, "Warning", "No interface loaded.")
                    return
                x_coords = np.asarray(self._basement_interface_data["x"], dtype=float)
                z_coords = np.asarray(self._basement_interface_data["z"], dtype=float)
            else:
                contour_value = float(ce.text().strip())
                z_coord = self._profile_extractor.z_coord
                x_coord = self._profile_extractor.x_coord
                z_vals = np.asarray(self._grid_data[z_coord].values, dtype=float)
                x_vals = np.asarray(self._grid_data[x_coord].values, dtype=float)
                vel_name = self._profile_extractor.velocity_var
                velocity_data = np.asarray(self._grid_data[vel_name].values, dtype=float)
                xs: list[float] = []
                zs: list[float] = []
                for j, x in enumerate(x_vals):
                    vp_col = velocity_data[:, j]
                    diff = np.abs(vp_col - contour_value)
                    min_idx = int(np.argmin(diff))
                    if diff[min_idx] < 0.1:
                        if min_idx > 0 and min_idx < len(vp_col) - 1:
                            v1, v2 = vp_col[min_idx], vp_col[min_idx + 1]
                            z1, z2 = z_vals[min_idx], z_vals[min_idx + 1]
                            if v2 != v1:
                                z_interp = z1 + (z2 - z1) * (contour_value - v1) / (v2 - v1)
                            else:
                                z_interp = z1
                            xs.append(float(x))
                            zs.append(float(z_interp))
                        else:
                            xs.append(float(x))
                            zs.append(float(z_vals[min_idx]))
                if not xs:
                    QMessageBox.warning(
                        self,
                        "Warning",
                        f"No contour found for velocity {contour_value:g} km/s",
                    )
                    return
                order = np.argsort(np.asarray(xs))
                x_coords = np.asarray(xs, dtype=float)[order]
                z_coords = np.asarray(zs, dtype=float)[order]

            with open(path, "w", encoding="utf-8") as f:
                f.write("# Interface file\n# Format: x(km) z(km)\n")
                if rb_contour.isChecked():
                    f.write(f"# Extracted from velocity contour: {ce.text().strip()} km/s\n")
                f.write(f"# Number of points: {len(x_coords)}\n\n")
                for x, z in zip(x_coords, z_coords):
                    f.write(f"{float(x):.6f}\t{float(z):.6f}\n")
            self._append_log(f"Interface saved: {path}")
            self._flash_status(Path(path).name, "success")
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid velocity value.")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to save interface:\n{e}")

    def _plot_zelt_interfaces(self) -> None:
        zm = self._zelt_model
        if zm is None:
            return
        try:
            n = len(zm.depth_nodes)
            for i in range(n):
                x_coords, z_coords = zm.get_layer_geometry(i)
                vm = ~(np.isnan(x_coords) | np.isnan(z_coords))
                if not np.any(vm):
                    continue
                xv = np.asarray(x_coords)[vm]
                zv = np.asarray(z_coords)[vm]
                is_basement = (
                    self._basement_interface_idx is not None and i == self._basement_interface_idx
                )
                is_seafloor = (
                    self._seafloor_interface_idx is not None and i == self._seafloor_interface_idx
                )
                is_moho = self._moho_interface_idx is not None and i == self._moho_interface_idx
                if i == n - 1:
                    self._ax.plot(
                        xv,
                        zv,
                        "k--",
                        lw=2.0,
                        alpha=0.8,
                        label="Bottom boundary",
                    )
                elif is_moho:
                    self._ax.plot(
                        xv,
                        zv,
                        "g-",
                        lw=2.5,
                        alpha=0.9,
                        label=f"M — Moho Interface {i + 1}",
                    )
                elif is_basement:
                    self._ax.plot(
                        xv,
                        zv,
                        "r-",
                        lw=2.5,
                        alpha=0.9,
                        label=f"B — Basement Interface {i + 1}",
                    )
                elif is_seafloor:
                    self._ax.plot(
                        xv,
                        zv,
                        "b-",
                        lw=2.5,
                        alpha=0.9,
                        label=f"S — Seafloor Interface {i + 1}",
                    )
                else:
                    lbl = f"Layer {i + 1} interface" if i == 0 else ""
                    self._ax.plot(xv, zv, "k-", lw=1.5, alpha=0.7, label=lbl)
        except Exception as e:
            self._append_log(f"Warning: interface plot failed: {e}")

    def _on_interfaces_toggled(self, checked: bool) -> None:
        if checked and self._grid_data is None:
            self._interfaces_chk.setChecked(False)
            QMessageBox.information(
                self,
                "Interfaces",
                "Please load a velocity model first.",
            )
            return
        self._redraw_main_plot_clearing_overlays()
        self._persist_state_maybe()

    def _iface_idx_from_combo(self, text: str) -> Optional[int]:
        t = str(text or "").strip()
        if t.startswith("Interface "):
            try:
                return int(t.replace("Interface ", "")) - 1
            except ValueError:
                return None
        return None

    def _refresh_interface_controls(self) -> None:
        self._basement_combo.blockSignals(True)
        self._seafloor_combo.blockSignals(True)
        self._moho_combo.blockSignals(True)
        try:
            self._basement_combo.clear()
            self._seafloor_combo.clear()
            self._moho_combo.clear()
            dn = getattr(self._zelt_model, "depth_nodes", None) if self._zelt_model else None
            if dn is not None and len(dn) > 0:
                self._basement_combo.addItem("None")
                self._moho_combo.addItem("None")
                for i in range(len(dn) - 1):
                    self._basement_combo.addItem(f"Interface {i + 1}")
                    self._moho_combo.addItem(f"Interface {i + 1}")
                self._seafloor_combo.addItem("Auto")
                for i in range(len(dn) - 1):
                    self._seafloor_combo.addItem(f"Interface {i + 1}")
                self._basement_interface_idx = None
                self._seafloor_interface_idx = None
                self._moho_interface_idx = None
                self._basement_combo.setCurrentIndex(0)
                self._seafloor_combo.setCurrentIndex(0)
                self._moho_combo.setCurrentIndex(0)
                if not self._restoring_state:
                    n_iface = len(dn) - 1
                    self._append_log(
                        f"Interface selectors B/S/M updated: {n_iface} v.in interfaces"
                    )
            else:
                self._basement_combo.addItem("None")
                self._seafloor_combo.addItem("Auto")
                self._moho_combo.addItem("None")
                self._basement_interface_idx = None
                self._seafloor_interface_idx = None
                self._moho_interface_idx = None
        finally:
            self._basement_combo.blockSignals(False)
            self._seafloor_combo.blockSignals(False)
            self._moho_combo.blockSignals(False)
        self._invalidate_seafloor_cache()
        if self._grid_data is not None:
            self._interfaces_chk.setEnabled(True)
        else:
            self._interfaces_chk.setEnabled(False)
            self._interfaces_chk.setChecked(False)

    def _on_basement_combo_changed(self, *_args: Any) -> None:
        t = self._basement_combo.currentText()
        self._basement_interface_idx = None if t == "None" else self._iface_idx_from_combo(t)
        self._invalidate_seafloor_cache()
        if not self._restoring_state:
            if self._basement_interface_idx is None:
                self._append_log("B (Basement): None — profiles start from model top (z=0)")
            else:
                self._append_log(
                    f"B (Basement) set to: {t} (index {self._basement_interface_idx})"
                )
        self._redraw_main_plot_clearing_overlays()
        self._persist_state_maybe()

    def _on_seafloor_combo_changed(self, *_args: Any) -> None:
        t = self._seafloor_combo.currentText()
        self._seafloor_interface_idx = None if t == "Auto" else self._iface_idx_from_combo(t)
        self._invalidate_seafloor_cache()
        if not self._restoring_state:
            if self._seafloor_interface_idx is None:
                self._append_log("S (Seafloor): Auto — detect from velocity (vp≈1.5 km/s)")
            else:
                self._append_log(
                    f"S (Seafloor) set to: {t} (index {self._seafloor_interface_idx})"
                )
        self._redraw_main_plot_clearing_overlays()
        self._persist_state_maybe()

    def _on_moho_combo_changed(self, *_args: Any) -> None:
        t = self._moho_combo.currentText()
        self._moho_interface_idx = None if t == "None" else self._iface_idx_from_combo(t)
        self._moho_interface_data = None
        if not self._restoring_state:
            if self._moho_interface_idx is None:
                self._append_log("M (Moho): None — Petrology H 需选择 Moho 界面")
            else:
                self._append_log(
                    f"M (Moho) set to: {t} (index {self._moho_interface_idx})"
                )
        self._redraw_main_plot_clearing_overlays()
        self._persist_state_maybe()

    def _apply_gravity_obs_settings_from_toolbox(self, d: dict[str, Any]) -> None:
        dd = str(d.get("gravity_obs_data_dir", "")).strip()
        self._grav_obs_data_dir = dd if dd else str(visualization_package_data_dir())
        fn = str(d.get("gravity_obs_filename", "")).strip()
        self._grav_obs_filename = fn if fn else DEFAULT_GRAVITY_OBS_FILENAME
        self._grav_obs_overlay_enabled = bool(d.get("gravity_obs_overlay", False))
        self._grav_profile_lon_lat_csv = str(d.get("gravity_profile_lon_lat_csv", "") or "").strip()
        self._persist_state_maybe()
        self._invalidate_full_model_gravity_cache()

    def _invalidate_full_model_gravity_cache(self) -> None:
        """丢弃重力工具箱 Full Model 缓存（海底/基底、界面文件、Vs 网格等变化时）。"""
        if self._grav_win is not None:
            self._grav_win.clear_full_model_cache()

    def _invalidate_seafloor_cache(self) -> None:
        self._seafloor_depth_map = None
        self._invalidate_full_model_gravity_cache()

    def _ensure_seafloor_depth_map(self) -> None:
        if self._seafloor_depth_map is not None:
            return
        if self._grid_data is None or self._profile_extractor is None:
            return
        self._seafloor_depth_map = compute_seafloor_depth_map(
            self._grid_data,
            self._profile_extractor,
            zelt_model=self._zelt_model,
            seafloor_interface_idx=self._seafloor_interface_idx,
        )

    def _get_seafloor_depth(self, x: float) -> float:
        if self._zelt_model is not None and self._seafloor_interface_idx is not None:
            return interpolate_interface_depth_at_x(
                self._zelt_model, int(self._seafloor_interface_idx), float(x)
            )
        self._ensure_seafloor_depth_map()
        if self._seafloor_depth_map is None or self._grid_data is None or self._profile_extractor is None:
            return 0.0
        x_vals = np.asarray(self._grid_data.coords[self._profile_extractor.x_coord].values, dtype=float)
        if len(self._seafloor_depth_map) != len(x_vals):
            self._invalidate_seafloor_cache()
            self._ensure_seafloor_depth_map()
        if self._seafloor_depth_map is None:
            return 0.0
        idx = int(np.argmin(np.abs(x_vals - float(x))))
        return float(self._seafloor_depth_map[idx])

    def _interp_interface_data_at_x(self, data: dict[str, Any], x: float) -> float:
        try:
            x_coords = np.asarray(data["x"], dtype=float)
            z_coords = np.asarray(data["z"], dtype=float)
            if np.any(x_coords):
                vm = np.argsort(x_coords)
                xv = x_coords[vm]
                zv = z_coords[vm]
                if len(xv) > 1:
                    return float(np.interp(float(x), xv, zv))
                return float(zv[0])
        except Exception:
            pass
        return 0.0

    def _moho_depth_at_x(self, x: float) -> float:
        """Moho depth at x — igneous crust bottom (Petrology H, transect)."""
        if self._zelt_model is not None and self._moho_interface_idx is not None:
            return interpolate_interface_depth_at_x(
                self._zelt_model, int(self._moho_interface_idx), float(x)
            )
        if self._moho_interface_data is not None:
            return self._interp_interface_data_at_x(self._moho_interface_data, float(x))
        return 0.0

    def _basement_depth_at_x(self, x: float) -> float:
        """沉积基底深度 — 剖面 depth 校正（B，非 Moho）。"""
        if self._zelt_model is not None and self._basement_interface_idx is not None:
            return interpolate_interface_depth_at_x(
                self._zelt_model, int(self._basement_interface_idx), float(x)
            )
        if self._basement_interface_data is not None:
            return self._interp_interface_data_at_x(self._basement_interface_data, float(x))
        return 0.0

    def _compute_pt_for_velocity_correction(
        self, x: float, z: float, vp_for_zone: float
    ) -> tuple[float, float]:
        """与 Tk PropertiesUIMixin 一致：校正到标准温压用的 (MPa, °C)。"""
        pc = self._prop_calc
        if pc is None:
            pressure = max(0.0, float(z)) * 30.0
            temperature = 25.0 + max(0.0, float(z)) * 30.0
            return pressure, temperature
        seafloor_z = float(self._get_seafloor_depth(float(x)))
        is_water = pc.is_water_zone(
            vp=float(vp_for_zone),
            z=float(z),
            seafloor_depth=seafloor_z,
        )
        pressure = pc.compute_total_pressure(
            z=float(z),
            seafloor_depth=seafloor_z,
            is_water=is_water,
            rock_pressure_gradient=30.0,
        )
        if is_water:
            temperature = float(pc.seawater_temperature_c)
        else:
            temperature = float(
                calculate_temperature_from_depth(
                    float(z),
                    temperature_gradient=float(pc.geothermal_gradient_c_per_km),
                    surface_temperature=float(pc.seafloor_temperature_c),
                    seafloor_depth=seafloor_z,
                )
            )
        return pressure, temperature

    def _petrology_status_line(self) -> str:
        if self._last_petrology_obs is None:
            return "观测: (未导出)"
        obs = self._last_petrology_obs
        tail = f" [{Path(self._petrology_obs_json_path).name}]" if self._petrology_obs_json_path else ""
        return (
            f"观测: H={obs.h_whole_km:.1f} km V_LC={obs.v_lc_km_s:.3f} ({obs.source}){tail}"
        )

    def _transect_status_line(self) -> str:
        if not self._last_transect_windows:
            return "沿迹: (未导出)"
        thick = sum(1 for w in self._last_transect_windows if w.thick_crust)
        tail = (
            f" [{Path(self._petrology_transect_windows_path).name}]"
            if self._petrology_transect_windows_path
            else ""
        )
        return (
            f"沿迹: {len(self._last_transect_windows)} 窗 "
            f"(厚 {thick}, 薄 {len(self._last_transect_windows) - thick}){tail}"
        )

    def _update_bridge_status_label(self) -> None:
        self._bridge_status.setText(
            f"{self._petrology_status_line()}\n{self._transect_status_line()}"
        )

    def _update_transect_status_label(self) -> None:
        self._update_bridge_status_label()

    def _update_petrology_status_label(self) -> None:
        self._update_bridge_status_label()

    def _on_transect_windows_saved(
        self,
        windows: list[TransectWindow],
        csv_path: Path,
        fig_path: Path | None,
    ) -> None:
        self._last_transect_windows = list(windows)
        self._petrology_transect_windows_path = str(csv_path.resolve())
        self._update_transect_status_label()
        self._append_log(f"沿迹滑窗已保存: {csv_path}" + (f"  Fig.15c → {fig_path}" if fig_path else ""))
        self._persist_state_maybe()

    def _open_transect_export(self) -> None:
        if self._grid_data is None:
            QMessageBox.warning(self, "沿迹导出", "请先加载 Vp 模型。")
            return
        dlg = TransectExportDialog(
            self,
            get_ctx=self._petrology_export_ctx,
            log_fn=self._append_log,
            on_windows_saved=self._on_transect_windows_saved,
        )
        show_modeless_dialog(dlg)

    def _lc_top_interface_options(self) -> list[tuple[str, Any]]:
        """v.in 层界面（除 B/M）供下地壳顶界选择。"""
        if self._zelt_model is None:
            return []
        opts: list[tuple[str, Any]] = []
        n_layers = len(self._zelt_model.depth_nodes) - 1
        for i in range(n_layers):
            if i == self._basement_interface_idx or i == self._moho_interface_idx:
                continue
            label = f"Layer {i + 1}"
            layer_idx = int(i)

            def _depth_fn(x: float, idx: int = layer_idx) -> float:
                return interpolate_interface_depth_at_x(
                    self._zelt_model, idx, float(x)
                )

            opts.append((label, _depth_fn))
        return opts

    def _resolve_preview_observation(self) -> CrustObservation | None:
        if self._last_petrology_obs is not None:
            return self._last_petrology_obs
        if self._grid_data is None:
            return None
        try:
            from petrology.imodel_bridge.imodel_adapter import crust_from_interfaces_at_x

            ctx = self._petrology_export_ctx()
            x_km = float(ctx.get("default_x_km") or 0.0)
            moho = ctx.get("moho_depth_at_x")
            top = ctx.get("igneous_top_depth_at_x") or ctx.get("basement_depth_at_x")
            if moho is None or top is None:
                return None
            opts = ctx.get("lc_top_interface_options") or []
            if not opts:
                return None
            lc_fn = opts[0][1]
            return crust_from_interfaces_at_x(
                x_km,
                moho_z_fn=moho,
                igneous_top_z_fn=top,
                grid_vp_fn=lambda x, z: float(ctx["sample_primary_velocity"](x, z)),
                lower_crust_top_z_fn=lc_fn,
                p_local_mpa_fn=None,
                t_local_c_fn=None,
                pt_correct=True,
            )
        except Exception:
            return None

    def _resolve_preview_transect_windows(self) -> list:
        if self._last_transect_windows:
            return list(self._last_transect_windows)
        if not self._petrology_transect_windows_path:
            return []
        try:
            self._last_transect_windows = load_transect_windows_csv(
                self._petrology_transect_windows_path
            )
            return list(self._last_transect_windows)
        except Exception:
            return []

    def _preview_hvp_overlay(self) -> None:
        obs = self._resolve_preview_observation()
        windows = self._resolve_preview_transect_windows()
        if obs is None and not windows:
            QMessageBox.warning(
                self,
                "H–Vp 投图",
                "请先导出单点观测（Export / Quick）或沿迹滑窗（Transect…）。",
            )
            return
        show_hvp_overlay_preview(self, observation=obs, windows=windows or None)

    def _preview_fig15c_transect(self) -> None:
        self._preview_hvp_overlay()

    def _preview_fig12a_petrology(self) -> None:
        self._preview_hvp_overlay()

    def _on_petrology_observation_saved(
        self,
        obs: CrustObservation,
        saved: Path | None,
        f_lower: float,
        x_km: float,
    ) -> None:
        self._last_petrology_obs = obs
        self._petrology_f_lower = float(f_lower)
        self._petrology_export_x_km = float(x_km)
        if saved is not None:
            self._petrology_obs_json_path = str(saved.resolve())
        self._update_petrology_status_label()
        self._persist_state_maybe()

    def _restore_petrology_from_section(self, sec: dict[str, Any]) -> None:
        raw_obs = sec.get("petrology_observation")
        if isinstance(raw_obs, dict):
            try:
                self._last_petrology_obs = CrustObservation.from_dict(raw_obs)
            except Exception:
                self._last_petrology_obs = None
        json_path = str(sec.get("petrology_obs_json") or "").strip()
        if json_path:
            resolved = resolve_restorable_input_path(
                json_path,
                state_file=self._gui_state_file,
                run_inputs_dir=self._run_inputs_dir,
            )
            if resolved and Path(resolved).exists():
                try:
                    self._last_petrology_obs = load_observation_json(resolved)
                    self._petrology_obs_json_path = resolved
                except Exception:
                    self._petrology_obs_json_path = json_path
            else:
                self._petrology_obs_json_path = json_path
        if "petrology_f_lower" in sec:
            try:
                self._petrology_f_lower = float(sec["petrology_f_lower"])
            except (TypeError, ValueError):
                pass
        if "petrology_export_x_km" in sec:
            try:
                self._petrology_export_x_km = float(sec["petrology_export_x_km"])
            except (TypeError, ValueError):
                pass
        tran_path = str(sec.get("petrology_transect_windows") or "").strip()
        if tran_path:
            resolved = resolve_restorable_input_path(
                tran_path,
                state_file=self._gui_state_file,
                run_inputs_dir=self._run_inputs_dir,
            )
            if resolved and Path(resolved).exists():
                try:
                    self._last_transect_windows = load_transect_windows_csv(resolved)
                    self._petrology_transect_windows_path = resolved
                except Exception:
                    self._petrology_transect_windows_path = tran_path
            else:
                self._petrology_transect_windows_path = tran_path
        self._update_petrology_status_label()
        self._update_transect_status_label()
        if self._last_petrology_obs is not None:
            from petrology.imodel_bridge.imodel_adapter import format_crust_observation_summary

            self._append_log(
                "Restored Petrology observation:\n" + format_crust_observation_summary(self._last_petrology_obs)
            )

    def _petrology_export_ctx(self) -> dict[str, Any]:
        has_basement = (
            (self._zelt_model is not None and self._basement_interface_idx is not None)
            or self._basement_interface_data is not None
        )
        has_moho = (
            (self._zelt_model is not None and self._moho_interface_idx is not None)
            or self._moho_interface_data is not None
        )

        def _build_profile(*, relative_to_basement: bool):
            if self._grid_data is None:
                return None
            x0 = float(self._x_min_edit.text().strip())
            x1 = float(self._x_max_edit.text().strip())
            dx = float(self._dx_combo.currentText())
            bd_fn = self._basement_depth_at_x if relative_to_basement and has_basement else None
            return averaged_vertical_profile(
                self._grid_data,
                x0,
                x1,
                dx,
                basement_depth_fn=bd_fn,
            )

        try:
            x_default = 0.5 * (
                float(self._x_min_edit.text().strip()) + float(self._x_max_edit.text().strip())
            )
            x_min_km = float(self._x_min_edit.text().strip())
            x_max_km = float(self._x_max_edit.text().strip())
        except ValueError:
            x_default = 0.0
            x_min_km = 0.0
            x_max_km = 0.0

        return {
            "grid_data": self._grid_data,
            "profile_extractor": self._profile_extractor,
            "property_calculator": self._prop_calc,
            "sample_primary_velocity": self._sample_primary_velocity_at_point,
            "compute_pt": self._compute_pt_for_velocity_correction,
            "selected_polygons": lambda: list(self._mpl_sel.selected_polygons),
            "basement_depth_at_x": self._basement_depth_at_x,
            "igneous_top_depth_at_x": self._basement_depth_at_x,
            "moho_depth_at_x": self._moho_depth_at_x,
            "seafloor_depth_at_x": self._get_seafloor_depth,
            "has_basement_interface": has_basement,
            "has_moho_interface": has_moho,
            "lc_top_interface_options": self._lc_top_interface_options(),
            "build_averaged_profile": lambda: _build_profile(relative_to_basement=True),
            "build_absolute_profile": lambda: _build_profile(relative_to_basement=False),
            "current_model_file": self._current_model_file,
            "default_x_km": x_default,
            "x_min_km": x_min_km,
            "x_max_km": x_max_km,
            "dx_km": float(self._dx_combo.currentText()),
            "petrology_f_lower": self._petrology_f_lower,
            "transect_windows": lambda: list(self._last_transect_windows),
        }

    def _open_petrology_export(self) -> None:
        if self._grid_data is None:
            QMessageBox.warning(self, "Petrology", "请先加载 Vp 模型。")
            return
        dlg = PetrologyExportDialog(
            self,
            get_ctx=self._petrology_export_ctx,
            log_fn=self._append_log,
            on_observation=self._on_petrology_observation_saved,
        )
        try:
            x0 = self._petrology_export_x_km or self._petrology_export_ctx()["default_x_km"]
            dlg._x_spin.setValue(float(x0))
        except Exception:
            pass
        show_modeless_dialog(dlg)

    def _quick_send_to_petrology(self) -> None:
        if self._grid_data is None:
            QMessageBox.warning(self, "Petrology", "请先加载 Vp 模型。")
            return
        dlg = PetrologyExportDialog(
            self,
            get_ctx=self._petrology_export_ctx,
            log_fn=self._append_log,
            on_observation=self._on_petrology_observation_saved,
        )
        try:
            dlg._x_spin.setValue(
                float(self._petrology_export_x_km or self._petrology_export_ctx()["default_x_km"])
            )
        except Exception:
            pass
        dlg._rb_interfaces.setChecked(True)
        obs = dlg.accept_and_export()
        if obs is not None:
            QMessageBox.information(
                self,
                "Petrology",
                f"已导出并尝试启动 LIP GUI:\nH={obs.h_whole_km:.2f} km\nV_LC={obs.v_lc_km_s:.4f} km/s",
            )

    def _rock_scatter_ctx(self) -> dict[str, Any]:
        return {
            "grid_data": self._grid_data,
            "profile_extractor": self._profile_extractor,
            "property_calculator": self._prop_calc,
            "velocity_method": self._vel_method.currentText().strip(),
            "model_type": self._model_type,
            "selected_points": self._mpl_sel.selected_points_xy,
            "selected_polygons": lambda: list(self._mpl_sel.selected_polygons),
            "compute_pt": self._compute_pt_for_velocity_correction,
            "sample_primary_velocity": self._sample_primary_velocity_at_point,
            "add_polygon_sample_markers_on_main": self._mpl_sel.add_polygon_sample_markers,
        }

    def _rock_add_vp_vs_points(self) -> None:
        get_or_create_rock_scatter_window(
            self, "vp_vs", self._rock_scatter_ctx, self._append_log, refresh_if_visible=False
        ).add_selected_points()

    def _rock_add_ratio_points(self) -> None:
        get_or_create_rock_scatter_window(
            self, "ratio", self._rock_scatter_ctx, self._append_log, refresh_if_visible=False
        ).add_selected_points()

    def _rock_add_vp_vs_poly_avg(self) -> None:
        get_or_create_rock_scatter_window(
            self, "vp_vs", self._rock_scatter_ctx, self._append_log, refresh_if_visible=False
        ).add_polygon_average()

    def _rock_add_ratio_poly_avg(self) -> None:
        get_or_create_rock_scatter_window(
            self, "ratio", self._rock_scatter_ctx, self._append_log, refresh_if_visible=False
        ).add_polygon_average()

    def _rock_add_vp_vs_poly_samples(self) -> None:
        get_or_create_rock_scatter_window(
            self, "vp_vs", self._rock_scatter_ctx, self._append_log, refresh_if_visible=False
        ).add_polygon_samples()

    def _rock_add_ratio_poly_samples(self) -> None:
        get_or_create_rock_scatter_window(
            self, "ratio", self._rock_scatter_ctx, self._append_log, refresh_if_visible=False
        ).add_polygon_samples()

    def _rock_clear_vp_vs(self) -> None:
        scatter_model_points_holder(self, "vp_vs").clear()
        h = getattr(self, "_rock_scatter_windows", None) or {}
        w = h.get("vp_vs")
        if isinstance(w, RockScatterToolWindow):
            try:
                w._refresh_plot()
            except Exception:
                pass
        self._append_log("  Cleared all model points from scatter plot")

    def _rock_clear_ratio(self) -> None:
        scatter_model_points_holder(self, "ratio").clear()
        h = getattr(self, "_rock_scatter_windows", None) or {}
        w = h.get("ratio")
        if isinstance(w, RockScatterToolWindow):
            try:
                w._refresh_plot()
            except Exception:
                pass
        self._append_log("  Cleared all model points from Vp/Vs vs Vp plot")

    def _surface_seafloor_array(self, x_vals: np.ndarray) -> np.ndarray:
        assert self._grid_data is not None and self._profile_extractor is not None
        return compute_seafloor_depth_map(
            self._grid_data,
            self._profile_extractor,
            zelt_model=self._zelt_model,
            seafloor_interface_idx=self._seafloor_interface_idx,
        )

    def _surface_basement_array(self, x_vals: np.ndarray) -> np.ndarray:
        xv = np.asarray(x_vals, dtype=float)
        if self._zelt_model is not None and self._basement_interface_idx is not None:
            return interpolate_interface_depths_on_grid(
                self._zelt_model, int(self._basement_interface_idx), xv
            )
        if self._basement_interface_data is not None:
            try:
                xc = np.asarray(self._basement_interface_data["x"], dtype=float)
                zc = np.asarray(self._basement_interface_data["z"], dtype=float)
                if xc.size == 0:
                    return np.full(xv.size, np.nan)
                order = np.argsort(xc)
                return np.interp(xv, xc[order], zc[order])
            except Exception:
                return np.full(xv.size, np.nan)
        return np.full(xv.size, np.nan)

    def _workbench_interface_files(self) -> list[str]:
        """写入会话 JSON 的界面文件路径（去重、稳定顺序）。"""
        out: list[str] = []
        seen: set[str] = set()
        for iface in self._loaded_interfaces:
            fp = str(iface.get("file") or "").strip()
            if not fp:
                continue
            try:
                key = str(Path(fp).resolve())
            except Exception:
                key = fp
            if key not in seen:
                seen.add(key)
                out.append(key)
        return out

    def _restore_interface_files_from_section(self, sec: dict[str, Any]) -> None:
        raw = sec.get("interface_files")
        if not isinstance(raw, list) or not raw:
            return
        self._loaded_interfaces.clear()
        self._basement_interface_data = None
        self._basement_interface_file = ""
        loaded_ct = 0
        for rawp in raw:
            p = resolve_restorable_input_path(
                str(rawp).strip(),
                state_file=self._gui_state_file,
                run_inputs_dir=self._run_inputs_dir,
            )
            if not p or not Path(p).exists():
                continue
            try:
                interfaces_in_file = parse_interface_file(p)
            except Exception as exc:
                self._append_log(f"Restore interface file skipped ({p}): {exc}")
                continue
            for iface in interfaces_in_file:
                iface["file"] = p
                self._loaded_interfaces.append(iface)
            loaded_ct += len(interfaces_in_file)
        if self._loaded_interfaces:
            first = self._loaded_interfaces[0]
            self._basement_interface_data = {"x": first["x"], "z": first["z"]}
            self._basement_interface_file = str(
                self._loaded_interfaces[0].get("file") or ""
            )
            n_paths = len([str(x).strip() for x in raw if str(x).strip()])
            self._append_log(
                f"Restored {n_paths} interface file path(s), {loaded_ct} curve(s) total"
            )
        self._update_interface_file_label()
        self._invalidate_seafloor_cache()
        self._redraw_main_plot_clearing_overlays()

    def _persist_state_maybe(self) -> None:
        if self._restoring_state:
            return
        self._save_workbench_state_safe()

    def _save_workbench_state_safe(self) -> None:
        if self._gui_state_file is None:
            return
        try:
            save_imodel_section(
                self._gui_state_file,
                model_file=self._current_model_file,
                show_interfaces=self._interfaces_chk.isChecked(),
                basement_selection=self._basement_combo.currentText(),
                seafloor_selection=self._seafloor_combo.currentText(),
                moho_selection=self._moho_combo.currentText(),
                interface_files=self._workbench_interface_files(),
                gravity_obs_data_dir=self._grav_obs_data_dir,
                gravity_obs_filename=self._grav_obs_filename,
                gravity_obs_overlay=self._grav_obs_overlay_enabled,
                gravity_profile_lon_lat_csv=self._grav_profile_lon_lat_csv,
                petrology_observation=(
                    self._last_petrology_obs.to_dict() if self._last_petrology_obs else None
                ),
                petrology_obs_json=self._petrology_obs_json_path or None,
                petrology_f_lower=self._petrology_f_lower,
                petrology_export_x_km=self._petrology_export_x_km,
                petrology_transect_windows=self._petrology_transect_windows_path or None,
            )
        except Exception:
            pass

    def _restore_workbench_state(self) -> None:
        if self._gui_state_file is None:
            return
        sec = load_imodel_section(self._gui_state_file)
        if not sec:
            return
        raw_path = str(sec.get("model_file", "")).strip()
        path = resolve_restorable_input_path(
            raw_path,
            state_file=self._gui_state_file,
            run_inputs_dir=self._run_inputs_dir,
        )
        if not path or not Path(path).exists():
            return
        self._restoring_state = True
        try:
            if not self._load_vp_from_path(path, log=False, persist=False):
                return
            si = sec.get("show_interfaces")
            if isinstance(si, bool):
                self._interfaces_chk.setChecked(si)
            bs = str(sec.get("basement_selection", "")).strip()
            if bs and self._basement_combo.findText(bs) >= 0:
                self._basement_combo.setCurrentText(bs)
                self._on_basement_combo_changed()
            sf_sel = str(sec.get("seafloor_selection", "")).strip()
            if sf_sel and self._seafloor_combo.findText(sf_sel) >= 0:
                self._seafloor_combo.setCurrentText(sf_sel)
                self._on_seafloor_combo_changed()
            ms = str(sec.get("moho_selection", "")).strip()
            if ms and self._moho_combo.findText(ms) >= 0:
                self._moho_combo.setCurrentText(ms)
                self._on_moho_combo_changed()
            self._restore_interface_files_from_section(sec)
            self._restore_petrology_from_section(sec)
            gdir = str(sec.get("gravity_obs_data_dir", "")).strip()
            if gdir:
                self._grav_obs_data_dir = gdir
            gfn = str(sec.get("gravity_obs_filename", "")).strip()
            if gfn:
                self._grav_obs_filename = gfn
            if "gravity_obs_overlay" in sec:
                self._grav_obs_overlay_enabled = bool(sec["gravity_obs_overlay"])
            gll = sec.get("gravity_profile_lon_lat_csv")
            if isinstance(gll, str) and gll.strip():
                self._grav_profile_lon_lat_csv = gll.strip()
            self._append_log(
                f"Restored session from {self._gui_state_file} (model: {Path(path).name})"
            )
        finally:
            self._restoring_state = False
        self._save_workbench_state_safe()

    def _load_vp_from_path(self, path: str, *, log: bool = True, persist: bool = True) -> bool:
        try:
            ds, zelt = load_velocity_grid(path)
        except Exception as e:
            traceback.print_exc()
            if log:
                QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")
            else:
                self._append_log(f"Restore failed: {e}")
            return False
        self._grid_data = ds
        self._zelt_model = zelt
        self._current_model_file = str(Path(path).resolve())
        self._invalidate_full_model_gravity_cache()
        self._classifier_diag_logged = False
        self._loaded_interfaces.clear()
        self._basement_interface_data = None
        self._basement_interface_file = ""
        self._update_interface_file_label()
        self._update_calculators()
        self._set_default_extent_fields()
        self._refresh_interface_controls()
        self._invalidate_seafloor_cache()
        self._redraw_main_plot_clearing_overlays()
        self._flash_status(Path(path).name, "success")
        if log:
            self._append_log(f"Vp model loaded: {path}")
            if self._prop_calc is not None:
                vv_main = getattr(self._prop_calc, "velocity_var", None)
                if vv_main:
                    self._append_log(f"Velocity variable detected: {vv_main}")
                vvv = getattr(self._prop_calc, "vs_velocity_var", None)
                if vvv:
                    self._append_log(f"Vs velocity variable detected: {vvv}")
        if log and self._zelt_model is not None:
            dn = len(getattr(self._zelt_model, "depth_nodes", []) or [])
            self._append_log(f"v.in: {dn} depth nodes (interface selection enabled)")
        self._log_vp_vs_coverage_summary()
        self._log_classifier_diagnostics()
        if persist:
            self._persist_state_maybe()
        return True

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_workbench_state_safe()
        super().closeEvent(event)

    def _enable_point_selection(self) -> None:
        if self._grid_data is None:
            QMessageBox.warning(self, "Warning", "Please load model first.")
            return

        def on_pick(x: float, z: float, index: int) -> None:
            if self._prop_calc is None:
                return
            dens = self._density_method_combo.currentText().strip()
            try:
                props = self._prop_calc.calculate_all_properties(
                    x,
                    z,
                    density_method=dens,
                    seafloor_depth=self._get_seafloor_depth(x),
                    basement_depth=self._basement_depth_at_x(x),
                )
            except Exception as exc:
                self._append_log(f"Point property error: {exc}")
                return
            props = dict(props or {})
            row = property_result_row_dict(x, z, props, source="point_selection")
            self._property_result_rows.append(row)

            lbl = f"Point {index + 1}: x={x:.2f} km, z={z:.2f} km"
            for line in iter_point_property_lines(
                x,
                z,
                dens,
                props,
                zone_label_map=self._zone_labels,
                point_label=lbl,
            ):
                self._append_log(line)

        self._mpl_sel.enable_point_selection(on_pick, self._append_log)

    def _enable_polygon_selection(self) -> None:
        if self._grid_data is None:
            QMessageBox.warning(self, "Warning", "Please load model first.")
            return
        self._mpl_sel.enable_polygon_selection(self._append_log)

    def _clear_selections(self) -> None:
        self._mpl_sel.clear_everything(self._append_log)

    def _open_gravity_toolbox(self) -> None:
        if self._grav_win is None:
            self._grav_win = GravityToolboxQt(
                self,
                get_ctx=lambda: self._gravity_ctx(),
                log_fn=self._append_log,
                on_obs_settings_changed=self._apply_gravity_obs_settings_from_toolbox,
            )
        else:
            self._grav_win.refresh_obs_widgets_from_ctx()
        self._grav_win.show()
        self._grav_win.raise_()

    def _gravity_ctx(self) -> dict[str, Any]:
        return {
            "grid_data": self._grid_data,
            "profile_extractor": self._profile_extractor,
            "property_calculator": self._prop_calc,
            "selected_polygons": self._mpl_sel.selected_polygons,
            "model_type": self._model_type,
            "velocity_method": self._vel_method.currentText().strip(),
            "density_method": self._density_method_combo.currentText().strip(),
            "seafloor_depths_fn": self._surface_seafloor_array,
            "basement_depths_fn": self._surface_basement_array,
            "current_model_file": self._current_model_file,
            "gravity_obs_data_dir": self._grav_obs_data_dir,
            "gravity_obs_filename": self._grav_obs_filename,
            "gravity_obs_overlay": self._grav_obs_overlay_enabled,
            "gravity_profile_lon_lat_csv": self._grav_profile_lon_lat_csv,
        }

    def _open_prop_grid(self, prop: str) -> None:
        if self._grid_data is None or self._profile_extractor is None:
            QMessageBox.warning(self, "Warning", "Load model first.")
            return
        open_property_grid_window(
            self,
            self._grid_data,
            self._profile_extractor,
            self._prop_calc,
            prop,
            self._density_method_combo.currentText().strip(),
            seafloor_depths_fn=self._surface_seafloor_array,
            basement_depths_fn=self._surface_basement_array,
            zelt_model=self._zelt_model,
            basement_interface_idx=self._basement_interface_idx,
            seafloor_interface_idx=self._seafloor_interface_idx,
            moho_interface_idx=self._moho_interface_idx,
        )

    def _show_user_guide(self) -> None:
        QMessageBox.information(self, "User Guide", USER_GUIDE_TEXT.strip())

    def _show_about(self) -> None:
        QMessageBox.information(self, "About", ABOUT_TEXT.strip())

    def _save_figure(self) -> None:
        path, selected = QFileDialog.getSaveFileName(
            self,
            "Save Figure",
            str(Path.cwd()),
            "PNG (*.png);;PDF (*.pdf);;PostScript (*.ps);;EPS (*.eps);;"
            "JPEG (*.jpg);;TIFF (*.tif);;All (*.*)",
        )
        if not path:
            return
        path = normalize_save_path_for_filter(path, selected)
        try:
            self._fig.savefig(path, dpi=300, bbox_inches="tight")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to save figure:\n{e}")
            return
        self._flash_status(Path(path).name, "success")
        self._append_log(f"Figure saved to: {path}")

    def _export_results(self) -> None:
        if not self._property_result_rows and not self._profiles:
            QMessageBox.information(
                self,
                "No Data",
                "No profiles or point results to export yet.",
            )
            return
        directory = QFileDialog.getExistingDirectory(self, "Select Export Directory", str(Path.cwd()))
        if not directory:
            return
        output_path = Path(directory)
        output_path.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        if self._profiles:
            for pname, pdf in sorted(self._profiles.items()):
                target = output_path / f"{pname}.csv"
                pdf.to_csv(target, index=False)
                written.append(target.name)
        if self._property_result_rows:
            csv_p = output_path / "property_point_results.csv"
            pd.DataFrame(self._property_result_rows).to_csv(csv_p, index=False)
            json_p = output_path / "property_point_results.json"
            with json_p.open("w", encoding="utf-8") as f:
                json.dump(json_compatible(self._property_result_rows), f, ensure_ascii=False, indent=2)
            written.extend([csv_p.name, json_p.name])
        msg = "\nResults exported to:\n" + str(output_path) + "\n\nFiles:\n  " + "\n  ".join(
            written
        )
        self._append_log(f"\nResults exported to: {directory}")
        QMessageBox.information(self, "Success", msg)

    def _line_visible_in_log(self, line: str) -> bool:
        if self._log_show_detailed:
            return True
        stripped = line.strip()
        if not stripped:
            return False
        if stripped.startswith("="):
            return False
        if line.startswith("  "):
            return any(line.startswith(prefix) for prefix in _LOG_COMPACT_DETAIL_PREFIXES)
        if stripped.startswith("Point ") or stripped.startswith("Property Results"):
            return True
        return True

    def _refresh_log_view(self) -> None:
        self._log.clear()
        for line in self._log_all_lines:
            if self._line_visible_in_log(line):
                self._log.append(line)

    def _clear_log(self) -> None:
        self._log_all_lines.clear()
        self._log.clear()

    def _append_log(self, text: str) -> None:
        raw = str(text or "")
        if not raw:
            return
        lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        for line in lines:
            if line.strip() == "":
                continue
            self._log_all_lines.append(line)
        self._refresh_log_view()

    def _on_log_detail_toggled(self, checked: bool) -> None:
        self._log_show_detailed = bool(checked)
        self._refresh_log_view()

    def _remove_colorbar(self) -> None:
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None

    def _update_calculators(self) -> None:
        ds = self._grid_data
        vs = self._vs_grid_data
        self._profile_extractor = ProfileExtractor(ds) if ds is not None else None
        self._prop_calc = (
            PropertyCalculator(ds, vs_grid_data=vs)
            if ds is not None
            else None
        )

    @staticmethod
    def _is_valid_vp_vs_pair(vp: float, vs: float) -> bool:
        try:
            vp_f = float(vp)
            vs_f = float(vs)
        except (TypeError, ValueError):
            return False
        if (
            (not np.isfinite(vp_f))
            or (not np.isfinite(vs_f))
            or vs_f <= 0.0
            or vp_f <= 0.0
        ):
            return False
        ratio = vp_f / vs_f
        return bool(np.isfinite(ratio) and (1.0 <= ratio <= 3.5))

    def _sample_primary_velocity_at_point(self, x: float, z: float) -> float:
        assert self._grid_data is not None and self._profile_extractor is not None
        assert self._prop_calc is not None
        x_coord = self._profile_extractor.x_coord
        z_coord = self._profile_extractor.z_coord
        velocity_var = self._prop_calc.velocity_var
        return PropertyCalculator._sample_dataset_value(
            self._grid_data,
            velocity_var,
            x_coord,
            z_coord,
            float(x),
            float(z),
        )

    def _get_point_vp_vs_for_diag(self, x: float, z: float) -> tuple[float, float, str]:
        velocity_method = self._vel_method.currentText().strip()
        pc = self._prop_calc
        assert pc is not None
        has_vs_model = getattr(pc, "vs_grid_data", None) is not None
        if has_vs_model:
            vp, vs, source = pc.get_vp_vs_at_point(
                x, z, velocity_method=velocity_method
            )
            return float(vp), float(vs), str(source or "")
        original_velocity = self._sample_primary_velocity_at_point(x, z)
        if self._model_type == "vp":
            vp = original_velocity
            vs = float(calculate_vs(vp, method=velocity_method))
            return vp, vs, ""
        vs = original_velocity
        vp = float(calculate_vp_from_vs_brocher(vs))
        return vp, vs, ""

    def _log_vp_vs_coverage_summary(self, max_samples: int = 3000) -> None:
        """对齐 Tk `RockScatterMixin._log_vp_vs_coverage_summary`。"""
        if (
            self._grid_data is None
            or self._prop_calc is None
            or self._profile_extractor is None
        ):
            return
        if max_samples <= 0:
            return
        x_coord = self._profile_extractor.x_coord
        z_coord = self._profile_extractor.z_coord
        x_vals = np.asarray(self._grid_data[x_coord].values, dtype=float)
        z_vals = np.asarray(self._grid_data[z_coord].values, dtype=float)
        if x_vals.size == 0 or z_vals.size == 0:
            return

        total_cells = int(x_vals.size * z_vals.size)
        target_nx = max(1, min(x_vals.size, int(np.sqrt(max_samples))))
        target_nz = max(1, min(z_vals.size, int(max_samples // target_nx)))
        x_idx = np.unique(np.linspace(0, x_vals.size - 1, num=target_nx, dtype=int))
        z_idx = np.unique(np.linspace(0, z_vals.size - 1, num=target_nz, dtype=int))

        sampled = 0
        valid = 0
        invalid = 0
        non_finite = 0
        source_model = 0
        source_empirical = 0
        has_vs_model = getattr(self._prop_calc, "vs_grid_data", None) is not None

        for iz in z_idx:
            for ix in x_idx:
                x = float(x_vals[ix])
                z = float(z_vals[iz])
                sampled += 1
                vp, vs, src = self._get_point_vp_vs_for_diag(x, z)
                if has_vs_model:
                    if src == "model":
                        source_model += 1
                    else:
                        source_empirical += 1
                else:
                    source_empirical += 1

                if (not np.isfinite(vp)) or (not np.isfinite(vs)):
                    non_finite += 1
                    continue
                if self._is_valid_vp_vs_pair(vp, vs):
                    valid += 1
                else:
                    invalid += 1

        if sampled <= 0:
            return
        valid_pct = 100.0 * valid / sampled
        invalid_pct = 100.0 * invalid / sampled
        non_finite_pct = 100.0 * non_finite / sampled
        self._append_log(
            f"Vp/Vs coverage diagnostics (sampled {sampled}/{total_cells} cells):"
        )
        self._append_log(
            f"  valid pairs: {valid} ({valid_pct:.1f}%), "
            f"invalid ratio/range: {invalid} ({invalid_pct:.1f}%), "
            f"non-finite: {non_finite} ({non_finite_pct:.1f}%)"
        )
        if has_vs_model:
            model_pct = 100.0 * source_model / sampled
            empirical_pct = 100.0 * source_empirical / sampled
            self._append_log(
                f"  Vs source mix: model={source_model} ({model_pct:.1f}%), "
                f"empirical={source_empirical} ({empirical_pct:.1f}%)"
            )

    def _log_classifier_diagnostics(self) -> None:
        """对齐 Tk `PropertiesUIMixin._log_classifier_diagnostics`。"""
        if self._classifier_diag_logged:
            return
        calc = self._prop_calc
        diag = getattr(calc, "classifier_diagnostics", None) if calc is not None else None
        if not isinstance(diag, dict):
            return

        if not diag.get("enabled"):
            self._append_log("Rock DB diagnostics: classifier unavailable")
            self._classifier_diag_logged = True
            return

        db_path = str(diag.get("database_path", "") or "").strip()
        rows = diag.get("row_count")
        rock_types = diag.get("rock_type_count")
        coverage = diag.get("coverage") if isinstance(diag.get("coverage"), dict) else {}
        error = str(diag.get("error", "") or "").strip()
        top_sources = diag.get("source_top5") if isinstance(diag.get("source_top5"), list) else []

        self._append_log("Rock DB diagnostics:")
        if db_path:
            self._append_log(f"  Database: {db_path}")
        if rows is not None:
            self._append_log(f"  Rows: {rows}")
        if rock_types is not None:
            self._append_log(f"  Rock types: {rock_types}")
        self._append_log(
            "  Coverage: "
            f"sio2={float(coverage.get('sio2_wt', 0.0)):.1%}, "
            f"facies={float(coverage.get('rock_facies', 0.0)):.1%}, "
            f"felsic_or_mafic={float(coverage.get('felsic_or_mafic', 0.0)):.1%}"
        )
        if top_sources:
            formatted = ", ".join(
                f"{str(item.get('source', 'unknown'))}({int(item.get('count', 0))})"
                for item in top_sources[:5]
                if isinstance(item, dict)
            )
            if formatted:
                self._append_log(f"  Top sources: {formatted}")
        if error:
            self._append_log(f"  Diagnostics note: {error}")
        self._classifier_diag_logged = True

    def _set_default_extent_fields(self) -> None:
        if self._grid_data is None or self._profile_extractor is None:
            self._x_min_edit.clear()
            self._x_max_edit.clear()
            self._px.clear()
            self._pz.clear()
            return
        x_vals = self._grid_data.coords[self._profile_extractor.x_coord].values
        z_vals = self._grid_data.coords[self._profile_extractor.z_coord].values
        self._x_min_edit.setText(f"{float(min(x_vals)):.4g}")
        self._x_max_edit.setText(f"{float(max(x_vals)):.4g}")
        xm = 0.5 * (float(min(x_vals)) + float(max(x_vals)))
        zm = 0.5 * (float(min(z_vals)) + float(max(z_vals)))
        self._px.setText(f"{xm:.4g}")
        self._pz.setText(f"{zm:.4g}")

    def _open_vp_model(self) -> None:
        path, _fl = QFileDialog.getOpenFileName(
            self,
            "Load Vp Model",
            str(Path.cwd()),
            "Grid (*.grd *.nc);;NetCDF (*.nc);;Zelt v.in (v.in *.vin);;All (*.*)",
        )
        if not path:
            return
        self._load_vp_from_path(path, log=True, persist=True)

    def _open_vs_model(self) -> None:
        path, _fl = QFileDialog.getOpenFileName(
            self,
            "Load Vs Model",
            str(Path.cwd()),
            "Grid (*.grd *.nc);;NetCDF (*.nc);;All (*.*)",
        )
        if not path:
            return
        try:
            ds, zelt = load_velocity_grid(path)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to load Vs model:\n{e}")
            return
        if zelt is not None:
            QMessageBox.warning(
                self,
                "Invalid Vs Model",
                "Vs model must be a grid file (.grd / .nc), not v.in.",
            )
            return
        self._vs_grid_data = ds
        self._update_calculators()
        self._invalidate_full_model_gravity_cache()
        self._append_log(f"Vs model loaded: {path}")
        if self._prop_calc is not None:
            vvv = getattr(self._prop_calc, "vs_velocity_var", None)
            if vvv:
                self._append_log(f"Vs velocity variable detected: {vvv}")
        self._log_vp_vs_coverage_summary()

    def _on_vertical_profile(self) -> None:
        if self._grid_data is None:
            QMessageBox.warning(self, "Warning", "Please load model first.")
            return
        try:
            x0 = float(self._x_min_edit.text().strip())
            x1 = float(self._x_max_edit.text().strip())
            dx = float(self._dx_combo.currentText())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid X range or sampling interval.")
            return
        try:
            df = averaged_vertical_profile(
                self._grid_data,
                x0,
                x1,
                dx,
                basement_depth_fn=self._basement_depth_at_x,
            )
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Profile error", str(e))
            return
        self._profile_counter += 1
        pname = f"profile_{self._profile_counter}"
        self._profiles[pname] = df.copy()

        has_basement = (
            (self._zelt_model is not None and self._basement_interface_idx is not None)
            or self._basement_interface_data is not None
        )
        dy = "Depth beneath basement (km)" if has_basement else "Depth (km)"
        ptitle = (
            "Averaged vertical profile (relative to basement)"
            if has_basement
            else "Averaged vertical profile"
        )
        self._append_log(
            f"Vertical profile [{x0:g}, {x1:g}] km, sampling={dx} km → {pname} ({len(df)} rows)"
            + ("; depths relative to basement" if has_basement else "")
        )
        show_modeless_dialog(
            VerticalProfileDialog(self, df, depth_ylabel=dy, plot_title=ptitle)
        )

    def _on_calculate_properties(self) -> None:
        if self._prop_calc is None:
            QMessageBox.warning(self, "Warning", "Please load model first.")
            return
        try:
            x = float(self._px.text().strip())
            z = float(self._pz.text().strip())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid X or Z.")
            return

        dens = self._density_method_combo.currentText().strip()
        try:
            props = self._prop_calc.calculate_all_properties(
                x,
                z,
                density_method=dens,
                seafloor_depth=self._get_seafloor_depth(x),
                basement_depth=self._basement_depth_at_x(x),
            )
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", str(e))
            return

        props = dict(props or {})
        row = property_result_row_dict(x, z, props, source="manual_calculate")
        self._property_result_rows.append(row)

        props["_x_disp"] = x
        props["_z_disp"] = z
        for line in iter_manual_calculate_lines(x, z, props, zone_label_map=self._zone_labels):
            self._append_log(line)


def run_application(argv: Optional[list[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    qt_app = QApplication(argv)
    apply_imodel_font(qt_app)
    win = ImodelQtMainWindow()
    win.show()
    return qt_app.exec()
