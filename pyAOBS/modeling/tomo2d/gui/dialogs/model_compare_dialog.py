"""独立模型对比：多模型集合统计 + A/B 差值 B−A。"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, QEvent, Signal
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..dialog_utils import (
    file_dialog_options,
    set_wrapping_caption,
    show_modeless_dialog,
    show_modeless_message,
    show_modeless_text,
    style_wrapping_caption,
)
from ..plots.inv_monitor_model import MonitorModelWidget
from ..services.dws_plot import dws_mask_enabled
from ..services.file_filters import REFL_SAVE_FILTERS, SMESH_OPEN_FILTERS, SMESH_SAVE_FILTERS
from ..services.model_compare import (
    EnsembleStatResult,
    MESH_FORM_FIELDS,
    REFL_FORM_FIELDS,
    apply_smesh_diff,
    apply_smesh_ensemble_stats,
    compare_status,
    compare_tray,
    clear_compare_tray,
    paint_smesh_ensemble_stats,
    save_ensemble_reflector,
    save_ensemble_velocity,
    write_ensemble_stat_files,
    write_path_to_form,
)
from ..services.paths import resolve_work_dir
from ..services.result_nav import (
    diff_vlim_half_range,
    diff_vlim_is_auto,
    format_diff_vlim_caption,
    format_sigma_vlim_caption,
    resolve_diff_colorbar_limits,
    set_diff_vlim_auto,
    set_diff_vlim_half_range,
)
from ..services.smesh_plot_core import (
    builtin_sigma_cpt_path,
    pick_smesh_paths,
    running_in_wsl,
)
from ..services.ui_prefs import restore_window_layout, save_window_layout
from ..state.form_state import FormState
from ..widgets.form_rows import MultiPathRow, PathRow

_LAYOUT_KEY = "model_compare"

_singleton: "ModelCompareDialog | None" = None


class ModelCompareDialog(QDialog):
    """非模态：集合统计 + A/B 差值。与速度图右键「添加到对比模型」共用托盘。"""

    applied = Signal(str, str)  # field_key, relative_path

    def __init__(self, state: FormState, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("模型对比")
        self.resize(980, 900)
        self.setAcceptDrops(True)
        self._plotting = False
        self._last_stats: EnsembleStatResult | None = None
        self._view = "diff"

        root = QVBoxLayout(self)
        tip = QLabel(
            "集合可放多个 smesh：统计均值 Vp 与误差 σ（两套也行）。"
            "差值仍是指定 A/B 的 B−A（上 ΔV，中 B，下 A）。网格须一致。"
            "速度图上 **右键**「写入表单」把当前模型写回参数，"
            "或「添加到对比模型」记入集合；凑齐 A/B 仍自动画差值。"
            "「打开所在目录」打开当前 smesh 所在文件夹。"
            "集合列表与反演分析「多日志」相同：长路径折行、退格删除；"
            "点「统计均值…」或「绘制差值」时才读取。"
            "统计后在均值图 / 误差图上右键保存速度或反射面，并可加入对比（右键拖动仍是缩放）。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#475569;")
        root.addWidget(tip)

        try:
            work0 = resolve_work_dir(state.get_str("work_dir"))
        except Exception:
            work0 = Path.cwd()

        def _work() -> Path:
            try:
                return resolve_work_dir(self.state.get_str("work_dir"))
            except Exception:
                return work0

        self.row_ens = MultiPathRow(
            "集合:",
            work_dir_getter=_work,
            name_filter=SMESH_OPEN_FILTERS,
            browse_caption="选择一个或多个 smesh",
            placeholder=(
                "每行一个 smesh。长路径会折行显示，文件之间仍是独立一行；"
                "可添加多个 / 拖放 / 粘贴。统计用全部，差值用下方 A/B。"
            ),
        )
        self.row_ens.edit.setMinimumHeight(72)
        self.row_ens.setMinimumHeight(96)

        self.row_a = PathRow(
            "A 基准:",
            mode="open_file",
            work_dir_getter=_work,
            keep_absolute=True,
            name_filter=SMESH_OPEN_FILTERS,
        )
        self.row_b = PathRow(
            "B 对比:",
            mode="open_file",
            work_dir_getter=_work,
            keep_absolute=True,
            name_filter=SMESH_OPEN_FILTERS,
        )
        self.row_a.edit.setPlaceholderText("基准模型（拖放 / 浏览 / 粘贴）")
        self.row_b.edit.setPlaceholderText("对比模型（图 = B − A）")

        opt = QHBoxLayout()
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("ΔV (km/s)", "abs")
        self.combo_mode.addItem("ΔV (%)", "percent")
        self.lbl_vlim = QLabel("色标 ±km/s")
        self.spin_vlim = QDoubleSpinBox()
        self.spin_vlim.setKeyboardTracking(False)
        self.ck_vlim_auto = QCheckBox("随数据")
        self.ck_vlim_auto.setToolTip(
            "勾选：色标拉到本图数据幅度。"
            "差值用 ±|max|；统计误差 σ 用 0–最大。"
            "不勾选：固定 ±km/s（差值对称，误差用正半幅作上界）。"
            "改完立即重绘，不必再点统计/绘制差值。均值 Vp 仍用速度色标。"
        )
        self.ck_contours = QCheckBox("叠加等值线")
        from ..plots.velocity_contours import contours_enabled

        self.ck_contours.setChecked(contours_enabled(self.state))
        self.ck_contours.setToolTip(
            "勾选：中图 B、下图 A 按当前色标叠对应等值线（vp / vs / vpvs）。上图 ΔV 不叠。"
        )
        self.ck_dws = QCheckBox("DWS 遮罩")
        self.ck_dws.setChecked(dws_mask_enabled(self.state))
        from ..widgets.smesh_cmap_combo import SmeshCmapCombo

        self.cmap_combo = SmeshCmapCombo(
            self.state, on_changed=lambda *_a: self._redraw_current()
        )
        self.ck_dws.setToolTip(
            "勾选：差值用 A∩B 交集（两侧都有覆盖才显示，权利用 min）；"
            "统计用各模型 DWS 的平均（各点只平均 DWS>0 的成员）。"
            "无覆盖留白；有覆盖按 log(DWS) 透明。"
            "每个 smesh 就近找：同目录，或该次 GUI 运行包 outputs/dws/。"
        )
        self.btn_dws_list = QPushButton("DWS 列表…")
        self.btn_dws_list.setToolTip(
            "列出当前集合（不足则 A/B）每个 smesh 实际用到的 DWS，便于核对。"
        )
        self.btn_plot = QPushButton("绘制差值")
        self.btn_stats = QPushButton("统计均值…")
        self.btn_stats.setToolTip(
            "集合内全部模型：上均值 Vp、下误差 σ。"
            "各网格点只平均该处 DWS>0 的成员（无覆盖不计入）。"
            "画完后在图上右键保存速度 / 反射面。"
        )
        self.btn_clear = QPushButton("清除")
        self.btn_paste = QPushButton("粘贴")
        self.btn_paste.setToolTip("剪贴板里的 .smesh 写入集合；两个则同时填 A、B。")
        opt.addWidget(self.combo_mode)
        opt.addWidget(self.lbl_vlim)
        opt.addWidget(self.spin_vlim)
        opt.addWidget(self.ck_vlim_auto)
        opt.addWidget(self.ck_contours)
        opt.addWidget(self.cmap_combo)
        opt.addWidget(self.ck_dws)
        opt.addWidget(self.btn_dws_list)
        opt.addWidget(self.btn_paste)
        opt.addWidget(self.btn_plot)
        opt.addWidget(self.btn_stats)
        opt.addWidget(self.btn_clear)
        opt.addStretch(1)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        opt.addWidget(btn_close)

        self.lbl_mesh = QLabel(compare_status())
        style_wrapping_caption(self.lbl_mesh)

        self._model_pg = MonitorModelWidget(self, compare_bar=False)
        self._model_pg.show_empty("请指定两个不同的 smesh")
        self._model_pg.set_model_source(None, self.state)
        self._model_pg._allow_clear_compare = True
        self._model_pg.set_stats_context_handler(self._on_stats_context_menu)

        below = QWidget()
        below_lay = QVBoxLayout(below)
        below_lay.setContentsMargins(0, 0, 0, 0)
        below_lay.setSpacing(6)
        below_lay.addWidget(self.row_a)
        below_lay.addWidget(self.row_b)
        below_lay.addLayout(opt)
        below_lay.addWidget(self.lbl_mesh)
        below_lay.addWidget(self._model_pg, stretch=1)
        below.setMinimumHeight(280)

        self._split = QSplitter(Qt.Orientation.Vertical)
        self._split.setChildrenCollapsible(False)
        self._split.setHandleWidth(8)
        self._split.addWidget(self.row_ens)
        self._split.addWidget(below)
        self._split.setStretchFactor(0, 1)
        self._split.setStretchFactor(1, 5)
        self._split.setSizes([160, 700])
        root.addWidget(self._split, stretch=1)

        self.row_a.edit.editingFinished.connect(self._on_paths_edited)
        self.row_b.edit.editingFinished.connect(self._on_paths_edited)
        self.row_ens.pathsCommitted.connect(self._on_ensemble_edited)
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.spin_vlim.valueChanged.connect(self._on_vlim_changed)
        self.ck_vlim_auto.toggled.connect(self._on_vlim_auto_toggled)
        self.ck_contours.toggled.connect(self._on_style_toggled)
        self.ck_dws.toggled.connect(self._on_style_toggled)
        self.btn_plot.clicked.connect(self._plot)
        self.btn_stats.clicked.connect(self._plot_stats)
        self.btn_clear.clicked.connect(self._clear)
        self.btn_dws_list.clicked.connect(self._show_dws_list)
        self.btn_paste.clicked.connect(self._paste_from_clipboard)
        paste_sc = QShortcut(QKeySequence.StandardKey.Paste, self)
        paste_sc.setContext(Qt.ShortcutContext.WindowShortcut)
        paste_sc.activated.connect(self._paste_from_clipboard)
        self._load_vlim_widgets()
        self.sync_from_tray(plot=False)

    def _mode(self) -> str:
        return str(self.combo_mode.currentData() or "abs")

    def _load_vlim_widgets(self) -> None:
        auto = diff_vlim_is_auto(self.state)
        self.spin_vlim.blockSignals(True)
        self.ck_vlim_auto.blockSignals(True)
        if self._view == "stats":
            self.combo_mode.setEnabled(False)
            self.lbl_vlim.setText("色标 ±km/s")
            self.spin_vlim.setRange(0.001, 10.0)
            self.spin_vlim.setSingleStep(0.05)
            self.spin_vlim.setDecimals(3)
            half = diff_vlim_half_range(self.state, mode="abs")
            self.spin_vlim.setValue(float(half))
            self.spin_vlim.setToolTip(
                "误差 σ 色标上界（km/s），与差值 ± 共用。"
                "勾选「随数据」则用本图 σ 最大。均值 Vp 不受此项影响。"
            )
        else:
            self.combo_mode.setEnabled(True)
            mode = self._mode()
            half = diff_vlim_half_range(self.state, mode=mode)
            if mode == "percent":
                self.lbl_vlim.setText("色标 ±%")
                self.spin_vlim.setRange(0.01, 100.0)
                self.spin_vlim.setSingleStep(0.5)
                self.spin_vlim.setDecimals(2)
            else:
                self.lbl_vlim.setText("色标 ±km/s")
                self.spin_vlim.setRange(0.001, 10.0)
                self.spin_vlim.setSingleStep(0.05)
                self.spin_vlim.setDecimals(3)
            self.spin_vlim.setValue(float(half))
            self.spin_vlim.setToolTip("差值 ΔV 的对称色标半幅；与统计误差 σ 上界共用 ±km/s。")
        self.ck_vlim_auto.setChecked(bool(auto))
        self.spin_vlim.setEnabled(not auto)
        self.ck_vlim_auto.blockSignals(False)
        self.spin_vlim.blockSignals(False)

    def _redraw_current(self) -> None:
        if self._view == "stats":
            self._repaint_stats(reset_home=False)
        else:
            self._plot_if_ready()

    def _repaint_stats(self, *, reset_home: bool = False) -> None:
        if self._last_stats is None or self._last_stats.n < 2:
            return
        if self._plotting:
            return
        self._plotting = True
        try:
            paint_smesh_ensemble_stats(
                self._model_pg,
                self.state,
                self._last_stats,
                auto_vlim=self.ck_vlim_auto.isChecked(),
                half_range=float(self.spin_vlim.value()),
                reset_home=reset_home,
            )
            self._set_stats_caption(self._last_stats)
        except Exception as e:
            self._model_pg.show_empty(f"统计失败: {e}")
            set_wrapping_caption(self.lbl_mesh, f"统计失败: {e}")
            show_modeless_message("统计失败", str(e), icon=QMessageBox.Icon.Warning)
        finally:
            self._plotting = False

    def _set_stats_caption(self, stat: EnsembleStatResult) -> None:
        bits = [f"n={stat.n}", f"反射面 {stat.n_refl}/{stat.n}"]
        if stat.refl_mean is not None and stat.refl_std is not None:
            import numpy as np

            bits.append(f"面 σ 均值={float(np.mean(stat.refl_std)):.4g} km")
        if self.ck_dws.isChecked():
            if stat.n_dws:
                bits.append(f"平均 DWS ×{stat.n_dws}")
            else:
                bits.append("未找到 DWS")
        if stat.sigma_vlim is not None:
            from ..services.result_nav import sigma_data_max

            bits.append(
                format_sigma_vlim_caption(
                    stat.sigma_vlim[1],
                    sigma_data_max(stat.std_v),
                    auto=self.ck_vlim_auto.isChecked(),
                    scale="cpt" if builtin_sigma_cpt_path().is_file() else "robust",
                )
            )
        set_wrapping_caption(self.lbl_mesh, " · ".join(bits))

    def _path_of(self, row: PathRow) -> Path | None:
        s = row.edit.text().strip().strip('"')
        if not s:
            return None
        p = Path(s).expanduser()
        try:
            p = p.resolve()
        except OSError:
            pass
        return p if p.is_file() else None

    def _set_row(self, row: PathRow, path: Path | None) -> None:
        row.edit.blockSignals(True)
        row.edit.setText("" if path is None else str(path).replace("\\", "/"))
        row.edit.blockSignals(False)

    def _resolve_listed(self, text: str) -> Path | None:
        s = text.strip().strip('"')
        if not s:
            return None
        p = Path(s).expanduser()
        if not p.is_absolute():
            try:
                p = resolve_work_dir(self.state.get_str("work_dir")) / p
            except Exception:
                pass
        try:
            p = Path(os.path.normpath(os.path.abspath(os.fspath(p))))
        except OSError:
            pass
        return p if p.is_file() else None

    def _ensemble_paths(self) -> list[Path]:
        out: list[Path] = []
        for s in self.row_ens.paths():
            p = self._resolve_listed(s)
            if p is not None:
                out.append(p)
        return out

    def _dws_list_paths(self) -> list[Path]:
        paths = self._ensemble_paths()
        if paths:
            return paths
        return [p for p in (self._path_of(self.row_a), self._path_of(self.row_b)) if p is not None]

    def _show_dws_list(self) -> None:
        paths = self._dws_list_paths()
        if not paths:
            show_modeless_message(
                "DWS 列表",
                "请先在集合中放入 smesh，或指定 A/B。",
                icon=QMessageBox.Icon.Warning,
            )
            return
        from ..services.dws_plot import describe_dws_for_smeshes, format_dws_match_report

        try:
            work = resolve_work_dir(self.state.get_str("work_dir"))
        except Exception:
            work = Path.cwd()
        pairs = describe_dws_for_smeshes(paths, self.state, work)
        show_modeless_text(
            "DWS 列表",
            format_dws_match_report(pairs, work=work),
            summary="每个 smesh 实际匹配到的 DWS（同目录优先，其次该次运行 outputs/dws/）。",
            activate=True,
            width=720,
            height=520,
            monospace=True,
        )

    def sync_from_tray(self, *, plot: bool = False, dws_xyz=None) -> None:
        t = compare_tray()
        self.row_ens.edit.blockSignals(True)
        ens = t.ensemble or [
            p for p in (t.path_a, t.path_b) if p is not None
        ]
        self.row_ens.set_paths([str(p) for p in ens], remember=False)
        self.row_ens.edit.blockSignals(False)
        self._set_row(self.row_a, t.path_a)
        self._set_row(self.row_b, t.path_b)
        set_wrapping_caption(self.lbl_mesh, compare_status())
        if plot and t.path_a is not None and t.path_b is not None:
            self._plot(dws_xyz=dws_xyz)

    def _push_tray_from_rows(self) -> None:
        t = compare_tray()
        t.set_ensemble(self._ensemble_paths())
        t.set_paths(self._path_of(self.row_a), self._path_of(self.row_b))
        set_wrapping_caption(self.lbl_mesh, compare_status())

    def _on_ensemble_edited(self) -> None:
        compare_tray().set_ensemble(self._ensemble_paths())
        set_wrapping_caption(self.lbl_mesh, compare_status())

    def _on_paths_edited(self) -> None:
        self._push_tray_from_rows()
        pa, pb = self._path_of(self.row_a), self._path_of(self.row_b)
        if pa is not None and pb is not None and pa != pb:
            self._plot()

    def _on_mode_changed(self, *_a) -> None:
        self._load_vlim_widgets()
        self._redraw_current()

    def _on_vlim_changed(self, value: float) -> None:
        if self._view == "stats":
            set_diff_vlim_half_range(self.state, "abs", float(value))
        else:
            set_diff_vlim_half_range(self.state, self._mode(), float(value))
        if not self.ck_vlim_auto.isChecked():
            self._redraw_current()

    def _on_vlim_auto_toggled(self, on: bool) -> None:
        set_diff_vlim_auto(self.state, bool(on))
        self.spin_vlim.setEnabled(not bool(on))
        self._redraw_current()

    def _on_style_toggled(self, *_a) -> None:
        from ..plots.velocity_contours import set_contours_enabled
        from ..services.dws_plot import set_dws_mask_enabled

        set_contours_enabled(self.state, self.ck_contours.isChecked())
        set_dws_mask_enabled(self.state, self.ck_dws.isChecked())
        if self._view == "stats":
            self._repaint_stats(reset_home=False)
        else:
            self._plot_if_ready(reset_home=False)

    def _plot_if_ready(self, *, reset_home: bool = True) -> None:
        pa, pb = self._path_of(self.row_a), self._path_of(self.row_b)
        if pa is not None and pb is not None and pa != pb:
            self._plot(reset_home=reset_home)

    def _plot(self, *, dws_xyz=None, reset_home: bool = True) -> None:
        _ = dws_xyz
        if self._plotting:
            return
        pa, pb = self._path_of(self.row_a), self._path_of(self.row_b)
        if pa is None or pb is None:
            show_modeless_message(
                "模型对比",
                "请先指定两个 smesh（A 与 B）。",
                icon=QMessageBox.Icon.Warning,
            )
            return
        try:
            if pa.resolve() == pb.resolve():
                show_modeless_message(
                    "模型对比", "A 与 B 是同一个文件。", icon=QMessageBox.Icon.Warning
                )
                return
        except OSError:
            pass
        self._push_tray_from_rows()
        self._view = "diff"
        self._load_vlim_widgets()
        self._plotting = True
        try:
            auto = self.ck_vlim_auto.isChecked()
            half = float(self.spin_vlim.value())
            diff = apply_smesh_diff(
                self._model_pg,
                self.state,
                pa,
                pb,
                mode=self._mode(),
                dws_xyz=None,
                auto_vlim=auto,
                half_range=half,
                reset_home=reset_home,
            )
            unit = "ΔV (%)" if self._mode() == "percent" else "ΔV (km/s)"
            v_lo, v_hi = resolve_diff_colorbar_limits(
                diff.vmax, auto=auto, half_range=half
            )
            extras = [
                f"mean={diff.mean:.4g}",
                f"std={diff.std:.4g}",
                f"|max|={diff.vmax:.4g}",
                format_diff_vlim_caption(diff.vmax, v_lo, v_hi, auto=auto),
                unit,
            ]
            set_wrapping_caption(
                self.lbl_mesh,
                f"上 ΔV · 中 B · 下 A（自选 B − A）  ·  {pb.name} − {pa.name}",
                " · ".join(extras),
                f"A: {pa}",
                f"B: {pb}",
            )
            self._model_pg.set_interaction_hint("")
        except Exception as e:
            self._model_pg.show_empty(f"差值失败: {e}")
            set_wrapping_caption(self.lbl_mesh, f"差值失败: {e}")
            show_modeless_message("差值失败", str(e), icon=QMessageBox.Icon.Warning)
        finally:
            self._plotting = False

    def _plot_stats(self) -> None:
        if self._plotting:
            return
        self._push_tray_from_rows()
        paths = self._ensemble_paths()
        if len(paths) < 2:
            pa, pb = self._path_of(self.row_a), self._path_of(self.row_b)
            paths = [p for p in (pa, pb) if p is not None]
        if len(paths) < 2:
            show_modeless_message(
                "模型对比",
                "统计至少需要集合里 2 个不同的 smesh（两个模型也可以）。",
                icon=QMessageBox.Icon.Warning,
            )
            return
        self._view = "stats"
        self._plotting = True
        try:
            stat = apply_smesh_ensemble_stats(
                self._model_pg,
                self.state,
                paths,
                auto_vlim=self.ck_vlim_auto.isChecked(),
            )
            self._last_stats = stat
            self._load_vlim_widgets()
            self._set_stats_caption(stat)
            self._model_pg.set_interaction_hint("右键保存、写入表单或加入对比（拖动仍缩放）")
        except Exception as e:
            self._last_stats = None
            self._model_pg.show_empty(f"统计失败: {e}")
            set_wrapping_caption(self.lbl_mesh, f"统计失败: {e}")
            show_modeless_message("统计失败", str(e), icon=QMessageBox.Icon.Warning)
        finally:
            self._plotting = False

    def _on_stats_context_menu(self, panel: str, menu: QMenu) -> None:
        if self._view != "stats" or self._last_stats is None:
            return
        has_refl = (
            self._last_stats.refl_x is not None
            and self._last_stats.refl_mean is not None
        )
        has_refl_std = has_refl and self._last_stats.refl_std is not None
        if panel == "mean":
            act_vel = menu.addAction("保存均值速度…")
            act_vel.triggered.connect(lambda: self._save_stats_velocity("mean"))
        else:
            act_vel = menu.addAction("保存速度误差 σ…")
            act_vel.triggered.connect(lambda: self._save_stats_velocity("std"))
        if has_refl:
            act_rm = menu.addAction("保存平均反射面…")
            act_rm.triggered.connect(lambda: self._save_stats_reflector("mean"))
        else:
            act_rm = menu.addAction("保存平均反射面…")
            act_rm.setEnabled(False)
        if has_refl_std:
            act_lo = menu.addAction("保存反射面 mean−σ…")
            act_hi = menu.addAction("保存反射面 mean+σ…")
            act_lo.triggered.connect(lambda: self._save_stats_reflector("minus"))
            act_hi.triggered.connect(lambda: self._save_stats_reflector("plus"))
        else:
            act_lo = menu.addAction("保存反射面 mean−σ…")
            act_hi = menu.addAction("保存反射面 mean+σ…")
            act_lo.setEnabled(False)
            act_hi.setEnabled(False)
        if panel == "mean":
            menu.addSeparator()
            sub_m = menu.addMenu("写入表单（均值速度）")
            for key, label in MESH_FORM_FIELDS:
                act = sub_m.addAction(label)
                act.triggered.connect(
                    lambda _c=False, k=key: self._apply_stats_to_form("mean", k)
                )
            if has_refl:
                sub_r = menu.addMenu("写入表单（平均反射面）")
                for key, label in REFL_FORM_FIELDS:
                    act = sub_r.addAction(label)
                    act.triggered.connect(
                        lambda _c=False, k=key: self._apply_stats_to_form("refl", k)
                    )

    def _outputs_dir(self) -> Path:
        try:
            work = resolve_work_dir(self.state.get_str("work_dir"))
        except Exception:
            work = Path.cwd()
        out = work / "outputs"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _pick_save_path(self, caption: str, default_name: str, filters: str) -> Path | None:
        start = str(self._outputs_dir() / default_name)
        path, _flt = QFileDialog.getSaveFileName(
            self,
            caption,
            start,
            filters,
            options=file_dialog_options(),
        )
        if not str(path).strip():
            return None
        return Path(path)

    def _save_stats_velocity(self, kind: str) -> None:
        if self._last_stats is None:
            return
        if kind == "mean":
            dest = self._pick_save_path(
                "保存均值速度", "ensemble_mean.smesh", SMESH_SAVE_FILTERS
            )
        else:
            dest = self._pick_save_path(
                "保存速度误差 σ", "ensemble_std.smesh", SMESH_SAVE_FILTERS
            )
        if dest is None:
            return
        try:
            saved = save_ensemble_velocity(self._last_stats, dest, kind=kind)
        except Exception as e:
            show_modeless_message("保存失败", str(e), icon=QMessageBox.Icon.Warning)
            return
        show_modeless_message("已保存", str(saved), icon=QMessageBox.Icon.Information)

    def _save_stats_reflector(self, which: str) -> None:
        if self._last_stats is None:
            return
        names = {
            "mean": ("保存平均反射面", "ensemble_mean.refl"),
            "minus": ("保存反射面 mean−σ", "ensemble_mean_m1sigma.refl"),
            "plus": ("保存反射面 mean+σ", "ensemble_mean_p1sigma.refl"),
        }
        caption, default = names.get(which, ("保存反射面", "ensemble.refl"))
        dest = self._pick_save_path(caption, default, REFL_SAVE_FILTERS)
        if dest is None:
            return
        try:
            saved = save_ensemble_reflector(self._last_stats, dest, which=which)
        except Exception as e:
            show_modeless_message("保存失败", str(e), icon=QMessageBox.Icon.Warning)
            return
        show_modeless_message("已保存", str(saved), icon=QMessageBox.Icon.Information)

    def _apply_stats_to_form(self, kind: str, field_key: str) -> None:
        if self._last_stats is None or self._last_stats.n < 2:
            return
        try:
            work = resolve_work_dir(self.state.get_str("work_dir"))
            stat = write_ensemble_stat_files(self._last_stats, work)
            self._last_stats = stat
        except Exception as e:
            show_modeless_message("写出失败", str(e), icon=QMessageBox.Icon.Warning)
            return
        if kind == "mean":
            path = stat.mean_smesh
        else:
            path = stat.mean_refl
        if path is None:
            show_modeless_message(
                "模型对比",
                "没有可写回的文件（反射面需集合里至少 2 条配套 *.refl.*）。",
                icon=QMessageBox.Icon.Warning,
            )
            return
        try:
            rel = write_path_to_form(self.state, field_key, path)
        except Exception as e:
            show_modeless_message("写出失败", str(e), icon=QMessageBox.Icon.Warning)
            return
        self.applied.emit(field_key, rel)
        show_modeless_message(
            "已写回表单",
            f"{field_key} ← {rel}",
            icon=QMessageBox.Icon.Information,
        )

    def _clear(self) -> None:
        clear_compare_tray()
        self._last_stats = None
        self.row_ens.edit.blockSignals(True)
        self.row_ens.set_paths([], remember=False)
        self.row_ens.edit.blockSignals(False)
        self._set_row(self.row_a, None)
        self._set_row(self.row_b, None)
        self._model_pg.show_empty("请指定集合或 A/B 两个 smesh")
        self._model_pg.set_interaction_hint("")
        set_wrapping_caption(self.lbl_mesh, compare_status())

    def _clipboard_smesh(self) -> list[Path]:
        from .smesh_plot import clipboard_plot_paths

        return pick_smesh_paths(clipboard_plot_paths())

    def _paste_from_clipboard(self) -> None:
        from PySide6.QtWidgets import QApplication, QLineEdit, QPlainTextEdit

        w = QApplication.focusWidget()
        if isinstance(w, (QLineEdit, QPlainTextEdit)):
            w.paste()
            return
        paths = self._clipboard_smesh()
        if not paths:
            extra = ""
            if running_in_wsl():
                extra = "\nWSLg 不能从 Windows 资源管理器拖入，请用复制+粘贴。"
            show_modeless_message(
                "粘贴",
                "剪贴板里没有 .smesh 文件。" + extra,
                icon=QMessageBox.Icon.Information,
            )
            return
        self._apply_dropped(paths)

    def _apply_dropped(self, paths: list[Path]) -> None:
        t = compare_tray()
        t.set_ensemble(list(t.ensemble) + paths)
        if len(paths) >= 2:
            t.set_paths(paths[0], paths[1])
        elif t.path_a is None:
            t.set_paths(paths[0], t.path_b)
        elif t.path_b is None:
            t.set_paths(t.path_a, paths[0])
        else:
            t._append_ensemble(paths[0])
        self.sync_from_tray(plot=False)
        if len(paths) >= 3:
            self._plot_stats()
            return
        pa, pb = t.path_a, t.path_b
        if pa is not None and pb is not None and pa != pb:
            self._plot()
        else:
            self._on_paths_edited()

    def _mime_has_files(self, event) -> bool:
        md = event.mimeData() if event is not None else None
        if md is None:
            return False
        try:
            return md.hasUrls() or md.hasText() or md.hasFormat("text/uri-list")
        except Exception:
            return False

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if self._mime_has_files(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # noqa: N802
        self.dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        from .smesh_plot import drop_local_paths

        paths = pick_smesh_paths(drop_local_paths(event))
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        self._apply_dropped(paths)

    def closeEvent(self, event) -> None:  # noqa: N802
        global _singleton
        try:
            save_window_layout(
                self, _LAYOUT_KEY, splitters={"ens_plot": self._split}
            )
        except Exception:
            pass
        if _singleton is self:
            _singleton = None
        try:
            import matplotlib.pyplot as plt

            plt.close(self._model_pg.fig)
        except Exception:
            pass
        super().closeEvent(event)


def open_model_compare_dialog(
    state: FormState,
    parent=None,
) -> ModelCompareDialog:
    """单例非模态对比窗。"""
    global _singleton
    dlg = None
    if _singleton is not None:
        try:
            _ = _singleton.isVisible()
            dlg = _singleton
        except RuntimeError:
            _singleton = None
    if dlg is None:
        dlg = ModelCompareDialog(state, parent)
        _singleton = dlg
        restore_window_layout(
            dlg, _LAYOUT_KEY, splitters={"ens_plot": dlg._split}
        )
        show_modeless_dialog(dlg, activate=True)
    else:
        dlg.state = state
        show_modeless_dialog(dlg, activate=True)
    return dlg
