# -*- coding: utf-8 -*-
"""Theme / themed dialogs mixed into QtFastViewer."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as exc:  # pragma: no cover
    raise RuntimeError("未安装 PySide6") from exc

try:
    import pyqtgraph as pg
except Exception:
    pg = None  # type: ignore


class ThemeMixin:
    """应用主题、样式表与主题感知文件对话框。"""

    _THEME_PRESETS: Dict[str, Dict[str, str]] = {
        "default": {
            "name": "默认主题",
            "window_bg": "#f2f4f8",
            "surface_bg": "#ffffff",
            "panel_bg": "#ffffff",
            "text": "#1f2937",
            "label_text": "#000000",
            "border": "#c7ccd5",
            "accent": "#2563eb",
            "accent_text": "#ffffff",
            "disabled_bg": "#edf0f4",
            "disabled_text": "#9aa1ab",
            "plot_bg": "#ffffff",
            "plot_axis": "#1f2937",
            "plot_grid_alpha": "0.12",
            "status_text": "#1f2937",
            "hint_text": "#5f6773",
            "wave_pen": "#0a0a0a",
            "shade_pen": "#2f3338",
            "pick_edge": "#505050",
            "pick_active_edge": "#ffffff",
        },
        "light": {
            "name": "浅蓝",
            "window_bg": "#eaf2fb",
            "surface_bg": "#ffffff",
            "panel_bg": "#f6faff",
            "text": "#1e293b",
            "label_text": "#1e293b",
            "border": "#b7c8de",
            "accent": "#2b6cb0",
            "accent_text": "#ffffff",
            "disabled_bg": "#e7eef7",
            "disabled_text": "#92a1b2",
            "plot_bg": "#ffffff",
            "plot_axis": "#1e293b",
            "plot_grid_alpha": "0.12",
            "status_text": "#1e293b",
            "hint_text": "#617489",
            "wave_pen": "#0f172a",
            "shade_pen": "#334155",
            "pick_edge": "#475569",
            "pick_active_edge": "#ffffff",
        },
        "dark": {
            "name": "深青",
            "window_bg": "#1f252c",
            "surface_bg": "#2a313a",
            "panel_bg": "#2f3944",
            "text": "#e5edf5",
            "label_text": "#e5edf5",
            "border": "#4b5b6b",
            "accent": "#2ea6a6",
            "accent_text": "#062b2b",
            "disabled_bg": "#3a444f",
            "disabled_text": "#8695a5",
            "plot_bg": "#1c232b",
            "plot_axis": "#dbe5ef",
            "plot_grid_alpha": "0.18",
            "status_text": "#e5edf5",
            "hint_text": "#b7c6d6",
            "wave_pen": "#e6edf5",
            "shade_pen": "#9ab2c8",
            "pick_edge": "#d5deea",
            "pick_active_edge": "#ffffff",
        },
        "solarized_light": {
            "name": "Solarized 浅色",
            "window_bg": "#fdf6e3",
            "surface_bg": "#fffdf6",
            "panel_bg": "#f8f1dd",
            "text": "#586e75",
            "label_text": "#586e75",
            "border": "#d6c7a1",
            "accent": "#268bd2",
            "accent_text": "#ffffff",
            "disabled_bg": "#efe6cf",
            "disabled_text": "#9ea99a",
            "plot_bg": "#fffdf6",
            "plot_axis": "#586e75",
            "plot_grid_alpha": "0.12",
            "status_text": "#586e75",
            "hint_text": "#6a7f86",
            "wave_pen": "#35484f",
            "shade_pen": "#5d6f74",
            "pick_edge": "#5f7378",
            "pick_active_edge": "#ffffff",
        },
        "solarized_dark": {
            "name": "Solarized 深色",
            "window_bg": "#002b36",
            "surface_bg": "#073642",
            "panel_bg": "#0a3f4d",
            "text": "#93a1a1",
            "label_text": "#93a1a1",
            "border": "#365864",
            "accent": "#b58900",
            "accent_text": "#1f1a00",
            "disabled_bg": "#274853",
            "disabled_text": "#6d8487",
            "plot_bg": "#032b36",
            "plot_axis": "#93a1a1",
            "plot_grid_alpha": "0.2",
            "status_text": "#93a1a1",
            "hint_text": "#8ca7a7",
            "wave_pen": "#d2dddd",
            "shade_pen": "#8fb0b1",
            "pick_edge": "#c7d1d1",
            "pick_active_edge": "#f8fcfc",
        },
        "nord": {
            "name": "Nord",
            "window_bg": "#2e3440",
            "surface_bg": "#3b4252",
            "panel_bg": "#434c5e",
            "text": "#eceff4",
            "label_text": "#eceff4",
            "border": "#5b657a",
            "accent": "#88c0d0",
            "accent_text": "#1f252e",
            "disabled_bg": "#4c566a",
            "disabled_text": "#9da7ba",
            "plot_bg": "#2b313d",
            "plot_axis": "#e5e9f0",
            "plot_grid_alpha": "0.18",
            "status_text": "#eceff4",
            "hint_text": "#c2cad7",
            "wave_pen": "#eceff4",
            "shade_pen": "#a7b2c4",
            "pick_edge": "#d7deea",
            "pick_active_edge": "#ffffff",
        },
        "graphite": {
            "name": "石墨灰",
            "window_bg": "#26282d",
            "surface_bg": "#30343b",
            "panel_bg": "#383c45",
            "text": "#eceff3",
            "label_text": "#eceff3",
            "border": "#596170",
            "accent": "#7aa2f7",
            "accent_text": "#111827",
            "disabled_bg": "#434955",
            "disabled_text": "#96a0b2",
            "plot_bg": "#262b33",
            "plot_axis": "#eceff3",
            "plot_grid_alpha": "0.18",
            "status_text": "#eceff3",
            "hint_text": "#bcc4d3",
            "wave_pen": "#edf1f7",
            "shade_pen": "#aeb7c9",
            "pick_edge": "#d8deea",
            "pick_active_edge": "#ffffff",
        },
        "forest": {
            "name": "森林绿",
            "window_bg": "#eef6f1",
            "surface_bg": "#ffffff",
            "panel_bg": "#f3fbf6",
            "text": "#1f3d2d",
            "label_text": "#1f3d2d",
            "border": "#b4d0c0",
            "accent": "#2f855a",
            "accent_text": "#ffffff",
            "disabled_bg": "#e6f1ea",
            "disabled_text": "#8aa294",
            "plot_bg": "#ffffff",
            "plot_axis": "#1f3d2d",
            "plot_grid_alpha": "0.12",
            "status_text": "#1f3d2d",
            "hint_text": "#557565",
            "wave_pen": "#173826",
            "shade_pen": "#395f4c",
            "pick_edge": "#365646",
            "pick_active_edge": "#ffffff",
        },
    }


    def _apply_optional_theme(self, theme_mode: str) -> None:
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        mode = str(theme_mode).strip().lower()
        theme = self._THEME_PRESETS.get(mode)
        if not theme:
            self._set_status_text(f"未知主题模式：{theme_mode}")
            return
        try:
            effective_theme = dict(theme)
            if mode != "default":
                dark_modes = {"dark", "solarized_dark", "nord", "graphite"}
                text_color = "#ffffff" if mode in dark_modes else "#000000"
                # 非默认主题按明暗设置高对比不透明文字：浅色黑字，深色白字
                effective_theme["text"] = text_color
                effective_theme["label_text"] = text_color
                effective_theme["status_text"] = text_color
                effective_theme["hint_text"] = text_color
                effective_theme["disabled_text"] = text_color
                effective_theme["accent_text"] = text_color
                effective_theme["plot_axis"] = text_color
            self._active_theme_mode = mode
            self._active_theme = effective_theme
            if mode == "default":
                # 默认主题回归原生 Qt 外观，不套自定义样式表
                app.setStyleSheet("")
                self._apply_plot_theme(effective_theme, native_default=True)
            else:
                app.setStyleSheet(self._build_theme_stylesheet(effective_theme))
                self._apply_plot_theme(effective_theme, native_default=False)
            self._sync_params_panel_content_height()
            self._enforce_params_panel_fixed_height()
            self._fit_params_strip_to_content()
            self._set_status_text(f"主题已应用：{effective_theme['name']}")
        except Exception as exc:
            self._set_status_text(f"主题应用失败：{exc}")


    def _theme_color(self, key: str, default: str) -> str:
        return str(self._active_theme.get(key, default))


    def _file_dialog_options(self) -> QtWidgets.QFileDialog.Option:
        """非默认主题使用 Qt 文件对话框，确保文件/目录名受高对比主题控制。"""
        options = QtWidgets.QFileDialog.Option(0)
        if str(getattr(self, "_active_theme_mode", "default")).strip().lower() != "default":
            options |= QtWidgets.QFileDialog.Option.DontUseNativeDialog
        return options


    def _get_save_file_name(
        self,
        caption: str,
        directory: str = "",
        filter: str = "",
        *,
        default_suffix: str = "",
        preferred_suffix: str = "",
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> Tuple[str, str]:
        """保存对话框：用户未输入扩展名时按过滤器自动补全。"""
        from pyAOBS.utils.qt_file_dialog import get_save_file_name

        return get_save_file_name(
            parent if parent is not None else self,
            caption,
            directory,
            filter,
            options=self._file_dialog_options(),
            default_suffix=default_suffix,
            preferred_suffix=preferred_suffix,
        )


    def _show_themed_info(self, title: str, text: str) -> None:
        """统一信息弹窗：非默认主题时强制高对比文本。"""
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Information)
        box.setWindowTitle(str(title))
        box.setText(str(text))
        box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        if str(getattr(self, "_active_theme_mode", "default")).strip().lower() != "default":
            txt = self._theme_color("text", "#000000")
            lbl = self._theme_color("label_text", txt)
            bg = self._theme_color("surface_bg", "#ffffff")
            panel = self._theme_color("panel_bg", bg)
            border = self._theme_color("border", "#888888")
            accent = self._theme_color("accent", "#3b82f6")
            box.setStyleSheet(
                "QMessageBox, QMessageBox QWidget { color: %s; background-color: %s; }"
                "QMessageBox QLabel { color: %s; background-color: transparent; }"
                "QMessageBox QPushButton { color: %s; background-color: %s; border: 1px solid %s; padding: 4px 10px; min-width: 70px; }"
                "QMessageBox QPushButton:hover { border-color: %s; }"
                % (txt, bg, lbl, txt, panel, border, accent)
            )
        box.exec()


    def _apply_file_action_text_contrast(self, text_color: str, native_default: bool = False) -> None:
        """提升文件相关操作按钮文字对比度。"""
        file_buttons = [
            self.btn_open,
            self.btn_open_hdr,
            self.btn_open_rec,
            self.btn_save_z,
            self.btn_save_picks,
            self.btn_save_hdr,
            self.btn_write_txin,
            self.btn_export_fig,
        ]
        for btn in file_buttons:
            try:
                if native_default:
                    btn.setStyleSheet("")
                else:
                    btn.setStyleSheet(
                        "QPushButton { color: %s; font-weight: 600; } "
                        "QPushButton:disabled { color: %s; font-weight: 600; }" % (text_color, text_color)
                    )
            except Exception:
                pass


    def _apply_plot_theme(self, theme: Dict[str, str], native_default: bool = False) -> None:
        """同步图窗、状态栏和辅助提示的主题颜色。"""
        try:
            self.plot.setBackground(theme.get("plot_bg", theme.get("surface_bg", "#ffffff")))
            plot_item = self.plot.getPlotItem()
            axis_color = theme.get("plot_axis", theme.get("text", "#1f2937"))
            axis_pen = pg.mkPen(axis_color, width=1)
            for axis_name in ("left", "bottom"):
                axis = plot_item.getAxis(axis_name)
                axis.setPen(axis_pen)
                axis.setTextPen(axis_pen)
            grid_alpha = float(theme.get("plot_grid_alpha", 0.12))
            self.plot.showGrid(x=True, y=True, alpha=max(0.0, min(1.0, grid_alpha)))
            wave_pen = pg.mkPen(theme.get("wave_pen", "#0a0a0a"), width=1)
            for item in self._curve_items:
                item.setPen(wave_pen)
            if self._shade_item is not None:
                self._shade_item.setPen(pg.mkPen(theme.get("shade_pen", "#30343b"), width=1))
            if self._stack_item is not None:
                self._stack_item.setPen(pg.mkPen(theme.get("stack_pen", "#1478dc"), width=1.5))
            if self._static_preview_item is not None:
                self._static_preview_item.setPen(
                    pg.mkPen(
                        theme.get("static_preview_pen", "#c828a0"),
                        width=1.8,
                        style=QtCore.Qt.PenStyle.DashLine,
                    )
                )
            if self._theoretical_item is not None:
                self._theoretical_item.setPen(pg.mkPen(theme.get("theory_pen", "#f07814"), width=2))
            if self._txin_item is not None:
                self._txin_item.setPen(pg.mkPen(theme.get("txin_pen", "#7c3aed"), width=1))
            if self._txin_map_preview_item is not None:
                self._txin_map_preview_item.setPen(pg.mkPen(theme.get("txin_preview_pen", "#f97316"), width=1.5))
            if self._water_corr_item is not None:
                self._water_corr_item.setPen(
                    pg.mkPen(theme.get("water_pen", "#1ea0d2"), width=2, style=QtCore.Qt.PenStyle.DashLine)
                )
            if self._pick_item is not None:
                self._pick_item.setPen(pg.mkPen(theme.get("pick_pen", "#dc1e1e"), width=1))
                self._pick_item.setBrush(pg.mkBrush(theme.get("pick_brush", "#ff7878")))
        except Exception:
            pass

        if native_default:
            try:
                self.lbl_status.setStyleSheet("")
            except Exception:
                pass
            try:
                self.lbl_gain_hint.setStyleSheet("color:#666; font-size:11px;")
            except Exception:
                pass
            self._apply_panel_frame_contrast(native_default=True)
            self._apply_param_panel_text_contrast("#000000", native_default=True)
            self._apply_file_action_text_contrast("#000000", native_default=True)
            return

        try:
            self.lbl_status.setStyleSheet(
                "padding: 2px 6px; border-top: 1px solid {0}; background-color: {1}; color: {2};".format(
                    theme.get("border", "#c7ccd5"),
                    theme.get("surface_bg", "#ffffff"),
                    theme.get("status_text", theme.get("text", "#1f2937")),
                )
            )
        except Exception:
            pass

        try:
            self.lbl_gain_hint.setStyleSheet(
                "color:{0}; font-size:11px;".format(
                    theme.get("hint_text", theme.get("text", "#666666"))
                )
            )
        except Exception:
            pass
        self._apply_panel_frame_contrast(native_default=False)
        self._apply_param_panel_text_contrast(theme.get("label_text", theme.get("text", "#000000")), native_default=False)
        self._apply_file_action_text_contrast(theme.get("text", "#000000"), native_default=False)


    def _build_theme_stylesheet(self, theme: Dict[str, str]) -> str:
        """构建主题样式：仅切换颜色，不修改尺寸/间距。"""
        return f"""
QWidget {{
    background-color: {theme['window_bg']};
    color: {theme['text']};
}}
QMainWindow, QScrollArea, QMenu, QMenuBar, QStatusBar {{
    background-color: {theme['window_bg']};
    color: {theme['text']};
}}
QLabel {{
    color: {theme['label_text']};
    background: transparent;
}}
QLabel:disabled {{
    color: {theme['label_text']};
}}
QGroupBox {{
    color: {theme['label_text']};
    border-color: {theme.get('panel_border', theme['accent'])};
    background-color: {theme['panel_bg']};
}}
QGroupBox::title {{
    color: {theme['label_text']};
    background-color: {theme.get('panel_title_bg', theme['window_bg'])};
    font-weight: 600;
    font-size: 16px;
}}
QGroupBox:disabled, QGroupBox::title:disabled {{
    color: {theme['label_text']};
}}
QPushButton, QToolButton {{
    border-color: {theme['border']};
    background-color: {theme['surface_bg']};
    color: {theme['text']};
}}
QPushButton:hover, QToolButton:hover {{
    border-color: {theme['accent']};
}}
QPushButton:pressed, QToolButton:pressed {{
    background-color: {theme['panel_bg']};
}}
QPushButton:disabled, QToolButton:disabled {{
    background-color: {theme['disabled_bg']};
    color: {theme['disabled_text']};
    border-color: {theme['border']};
}}
QComboBox, QSpinBox, QDoubleSpinBox {{
    border-color: {theme['border']};
    background-color: {theme['surface_bg']};
    color: {theme['text']};
    selection-background-color: {theme['accent']};
    selection-color: {theme['accent_text']};
}}
QComboBox QAbstractItemView {{
    border: 1px solid {theme['border']};
    background-color: {theme['surface_bg']};
    color: {theme['text']};
    selection-background-color: {theme['accent']};
    selection-color: {theme['accent_text']};
}}
QCheckBox, QRadioButton {{
    color: {theme['label_text']};
}}
QCheckBox:disabled, QRadioButton:disabled {{
    color: {theme['label_text']};
}}
QFileDialog, QFileDialog QWidget {{
    color: {theme['text']};
}}
QFileDialog QTreeView, QFileDialog QListView, QFileDialog QTableView {{
    color: {theme['text']};
    background-color: {theme['surface_bg']};
    selection-color: {theme['accent_text']};
    selection-background-color: {theme['accent']};
}}
QFileDialog QLineEdit, QFileDialog QComboBox {{
    color: {theme['text']};
    background-color: {theme['surface_bg']};
    border-color: {theme['border']};
}}
QFileDialog QHeaderView::section {{
    color: {theme['label_text']};
    background-color: {theme['panel_bg']};
}}
QScrollBar::handle:vertical {{
    background: {theme['border']};
}}
QFrame {{
    color: {theme['border']};
}}
"""

