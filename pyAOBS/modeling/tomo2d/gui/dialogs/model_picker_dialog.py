"""反演模型挑选助手：列表 + 预览；写回表单 / 打开目录走预览图右键。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pyAOBS.utils.qt_combo import connect_combo_deferred

from ..dialog_utils import (
    set_wrapping_caption,
    show_modeless_dialog,
    show_modeless_message,
    style_wrapping_caption,
    file_dialog_options,
)
from ..plots.inv_monitor_model import MonitorModelWidget
from ..services.inv_monitor import (
    InvMonitorSpec,
    ModelCandidate,
    build_model_catalog,
    collect_run_inversion_params,
)
from ..services.paths import resolve_work_dir
from ..services.result_nav import (
    infer_run_dir_from_smesh,
    list_tt_inverse_run_dirs,
    monitor_spec_for_run,
    resolve_user_run_dir,
)
from ..services.smesh_plot_core import (
    load_smesh_plot_data,
    resolve_plot_refl_for_smesh,
    resolve_plot_smesh_cmap,
)
from ..services.ui_prefs import restore_window_layout, save_window_layout
from ..state.form_state import FormState

_LAYOUT_KEY = "model_picker"

_singleton: "ModelPickerDialog | None" = None


def _fmt(v: float | None) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.6g}"
    except (TypeError, ValueError):
        return "—"


class ModelPickerDialog(QDialog):
    """按迭代列出 smesh 与过程指标；点选预览；写回/打开目录走图上右键。"""

    def __init__(self, state: FormState, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("模型挑选助手")
        self.resize(1120, 740)

        self._spec: InvMonitorSpec | None = None
        self._rows: list[ModelCandidate] = []
        self._path: Path | None = None
        self._skip_table_sel = False
        self._extra_runs: list[Path] = []

        root = QVBoxLayout(self)
        tip = QLabel(
            "先选一次反演运行包，再从该次各轮 smesh 里选用。"
            "下拉为当前工区 runs/；「浏览…」可自选其它目录（工区外亦可）。"
            "右侧预览：上两行折射/反射残差、下为速度场。"
            "预览图右键：「写入表单」「添加到对比模型」「打开所在目录」。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#475569;")
        root.addWidget(tip)

        self.lbl_status = QLabel("未绑定目标")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("font-weight:600;")
        root.addWidget(self.lbl_status)
        self.lbl_params = QLabel("反演参数：选择运行包后显示 -SV / -TV / -J 等")
        style_wrapping_caption(self.lbl_params)
        root.addWidget(self.lbl_params)

        run_row = QHBoxLayout()
        run_row.addWidget(QLabel("运行包"))
        self.combo_run = QComboBox()
        self.combo_run.setMinimumWidth(320)
        self.combo_run.setToolTip(
            "当前工区 runs/ 下的 tt_inverse（及棋盘格/蒙特卡洛）运行包。"
            "工区外的包请点「浏览…」。"
        )
        self.btn_browse_run = QPushButton("浏览…")
        self.btn_browse_run.setToolTip(
            "自选运行包目录：包根、outputs/、models/ 均可；不必在当前工区 runs/ 下。"
        )
        self.btn_refresh = QPushButton("刷新")
        self.btn_bind = QPushButton("表单路径")
        self.btn_bind.setToolTip("未建运行包时：按表单 inv.out_root 绑定（相对工区）")
        run_row.addWidget(self.combo_run, stretch=1)
        run_row.addWidget(self.btn_browse_run)
        run_row.addWidget(self.btn_refresh)
        run_row.addWidget(self.btn_bind)
        root.addLayout(run_row)

        bar = QHBoxLayout()
        self.btn_plot = QPushButton("独立图窗…")
        self.ck_contours = QCheckBox("叠加等值线")
        from ..plots.velocity_contours import contours_enabled

        self.ck_contours.setChecked(contours_enabled(self.state))
        self.ck_contours.setToolTip(
            "勾选：预览按当前色标叠对应等值线（vp / vs / vpvs）。不勾选则只画色块与界面。"
        )
        self.ck_contours.toggled.connect(self._on_contours_toggled)
        self.ck_dws = QCheckBox("DWS 遮罩")
        from ..services.dws_plot import dws_mask_enabled
        from ..widgets.smesh_cmap_combo import SmeshCmapCombo

        self.cmap_combo = SmeshCmapCombo(
            self.state, on_changed=lambda *_a: self._refresh_preview(reset_home=False)
        )
        self.ck_dws.setChecked(dws_mask_enabled(self.state))
        self.ck_dws.setToolTip(
            "勾选：自动找该次运行 outputs/dws/（或与 smesh 同目录；-K 在反演结束才写出）。"
            "无覆盖留白；有覆盖按 log(DWS) 透明（越大越实）。不勾选则整幅实色。"
        )
        self.ck_dws.toggled.connect(self._on_dws_toggled)
        bar.addWidget(self.btn_plot)
        bar.addWidget(self.ck_contours)
        bar.addWidget(self.cmap_combo)
        bar.addWidget(self.ck_dws)
        bar.addStretch(1)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        bar.addWidget(btn_close)
        root.addLayout(bar)

        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(
            [
                "运行包",
                "iter",
                "iset",
                "-SV",
                "-SD",
                "χ²",
                "RMS",
                "pred χ²",
                "粗糙度",
                "文件",
            ]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)

        self._model_pg = MonitorModelWidget(self)
        self._model_pg.show_empty("选择一行以预览")

        self._split = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.lbl_table = QLabel("该次运行的各轮 smesh（点击行预览）")
        self.lbl_table.setWordWrap(True)
        ll.addWidget(self.lbl_table)
        ll.addWidget(self.table, stretch=1)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.lbl_mesh = QLabel("走时拟合（上） / 速度场（下）")
        style_wrapping_caption(self.lbl_mesh)
        rl.addWidget(self.lbl_mesh)
        rl.addWidget(self._model_pg, stretch=1)
        self._split.addWidget(left)
        self._split.addWidget(right)
        self._split.setStretchFactor(0, 3)
        self._split.setStretchFactor(1, 2)
        root.addWidget(self._split, stretch=1)

        self.btn_refresh.clicked.connect(self.reload_runs)
        self.btn_browse_run.clicked.connect(self.browse_run_dir)
        self.btn_bind.clicked.connect(self.bind_from_form)
        connect_combo_deferred(
            self.combo_run, self._on_run_changed, signal="currentIndexChanged"
        )
        self.btn_plot.clicked.connect(self.plot_selected_window)
        self.table.itemSelectionChanged.connect(self._on_selection)

        restore_window_layout(
            self, _LAYOUT_KEY, splitters={"main": self._split}
        )

    def attach_spec(self, spec: InvMonitorSpec) -> None:
        self._spec = spec
        if spec.run_dir is not None:
            self.reload_runs(prefer=spec.run_dir)
            return
        self.refresh()

    def reload_runs(self, prefer: Path | str | None = None) -> None:
        """列出 runs/ + 用户自选包；默认最近一次（★ gui.last_tt_inverse_run）。"""
        work = resolve_work_dir(self.state.get_str("work_dir"))
        prev = prefer or self.combo_run.currentData()
        pref_p = Path(str(prefer)).resolve() if prefer else None
        if pref_p is not None and pref_p.is_dir():
            self._remember_extra_run(pref_p)
        self.combo_run.blockSignals(True)
        self.combo_run.clear()
        last = self.state.get_str("gui.last_tt_inverse_run")
        for p in self._run_dirs_for_combo(work):
            self.combo_run.addItem(
                self._run_combo_label(p, last=last, work=work),
                str(p.resolve()),
            )
        if self.combo_run.count() == 0:
            self.combo_run.blockSignals(False)
            self._spec = None
            self.lbl_status.setText(
                f"工区 {work} 下尚无 runs/ 运行包。"
                "可点「浏览…」自选其它目录，或「表单路径」绑定 inv.out_root。"
            )
            self.table.setRowCount(0)
            self._rows = []
            self.lbl_params.setText("反演参数：无运行包")
            self.lbl_params.setToolTip("")
            self._path = None
            self._model_pg.show_empty("工区下尚无运行包")
            return
        idx = 0
        want = None
        if prev:
            want = str(Path(str(prev)).resolve())
        elif last:
            try:
                want = str(Path(last).resolve())
            except OSError:
                want = last
        if want:
            for i in range(self.combo_run.count()):
                if self.combo_run.itemData(i) == want:
                    idx = i
                    break
        self.combo_run.setCurrentIndex(idx)
        self.combo_run.blockSignals(False)
        self._on_run_changed()

    def _run_dirs_for_combo(self, work: Path) -> list[Path]:
        listed = list_tt_inverse_run_dirs(work)
        seen = {p.resolve() for p in listed}
        extras: list[Path] = []
        for e in self._extra_runs:
            try:
                er = e.resolve()
            except OSError:
                continue
            if er.is_dir() and er not in seen:
                extras.append(er)
                seen.add(er)
        return extras + listed

    @staticmethod
    def _run_combo_label(p: Path, *, last: str, work: Path) -> str:
        name = p.name
        try:
            if last and Path(last).resolve() == p.resolve():
                name = f"★ {name}"
        except OSError:
            pass
        try:
            p.resolve().relative_to((work / "runs").resolve())
            return name
        except ValueError:
            return f"{name}  ·  自选"

    def _remember_extra_run(self, rd: Path) -> None:
        try:
            r = rd.resolve()
        except OSError:
            r = Path(rd)
        if not r.is_dir():
            return
        self._extra_runs = [p for p in self._extra_runs if p != r]
        self._extra_runs.insert(0, r)

    def _run_browse_start(self) -> str:
        data = self.combo_run.currentData()
        if data:
            p = Path(str(data))
            if p.is_dir():
                return str(p)
        try:
            work = resolve_work_dir(self.state.get_str("work_dir"))
            runs = work / "runs"
            return str(runs if runs.is_dir() else work)
        except Exception:
            return ""

    def browse_run_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "选择运行包目录",
            self._run_browse_start(),
            file_dialog_options(QFileDialog.Option.ShowDirsOnly),
        )
        if path:
            self.select_user_run_dir(path)

    def select_user_run_dir(self, path: Path | str) -> bool:
        """绑定用户指定的运行包（目录或包内文件）。供浏览与测试调用。"""
        rd = resolve_user_run_dir(path)
        if rd is None:
            show_modeless_message(
                "模型挑选",
                "未识别为运行包。请选含 manifest.json 或 outputs/models 的目录"
                "（点到包根、outputs/、models/ 均可）。",
                icon=QMessageBox.Icon.Warning,
            )
            return False
        self._remember_extra_run(rd)
        self.reload_runs(prefer=rd)
        return True

    def _on_run_changed(self, *_args) -> None:
        data = self.combo_run.currentData()
        if not data:
            return
        niter = self.state.get_str("inv.niter") or None
        self._spec = monitor_spec_for_run(Path(str(data)), niter=niter)
        self.refresh()

    def bind_from_form(self) -> None:
        from ..services.inv_monitor import build_monitor_spec_from_paths

        work = resolve_work_dir(self.state.get_str("work_dir"))
        log = self.state.get_str("inv.log_file") or "tt_inverse.log"
        out = self.state.get_str("inv.out_root") or "out"
        niter = self.state.get_str("inv.niter") or None
        st = self.state.get_str("env.inv_status_jsonl_path") or "outputs/status.jsonl"
        log_rel = (
            log
            if "/" in log.replace("\\", "/") or Path(log).is_absolute()
            else f"outputs/{Path(log).name}"
        )
        out_rel = (
            out
            if "/" in out.replace("\\", "/") or Path(out).is_absolute()
            else f"outputs/{Path(out).name}"
        )
        spec = build_monitor_spec_from_paths(
            cwd=work,
            log_file=log_rel,
            out_root=out_rel,
            niter=niter,
            status_jsonl=st,
        )
        spec.log_candidates.extend(
            [
                work / log,
                work / "outputs" / Path(log).name,
                work / "outputs" / "logs" / Path(log).name,
            ]
        )
        self.combo_run.blockSignals(True)
        self.combo_run.setCurrentIndex(-1)
        self.combo_run.blockSignals(False)
        self._spec = spec
        self.refresh()

    def refresh(self) -> None:
        if self._spec is None:
            self.lbl_status.setText(
                "未绑定运行包 — 请从下拉选择 runs/，「浏览…」自选，或点「表单路径」"
            )
            self.table.setRowCount(0)
            self._rows = []
            self.lbl_params.setText("反演参数：未绑定运行包")
            self.lbl_params.setToolTip("")
            self._path = None
            self._model_pg.show_empty("未绑定运行包")
            return
        self._skip_table_sel = True
        try:
            self._rows = build_model_catalog(self._spec)
            self._fill_table()
            out = self._spec.out_root
            run = self._spec.run_dir
            self.lbl_status.setText(
                f"共 {len(self._rows)} 个模型"
                + (f" · 运行包 {run.name}" if run is not None else "")
                + (f" · out_root={out}" if out is not None else "")
                + " · 点击行预览，右键写入表单 / 打开所在目录"
            )
            self.lbl_table.setText("该次运行的各轮 smesh（点击行预览）")
            params = collect_run_inversion_params(self._spec)
            if params:
                set_wrapping_caption(self.lbl_params, f"反演参数：{params}")
            else:
                set_wrapping_caption(
                    self.lbl_params,
                    "反演参数：未读到 -L 头 / manifest（该次可能未写日志）",
                )
            self._select_last_catalog_rows()
        finally:
            self._skip_table_sel = False
        self._preview_catalog_selection()

    def _selected_candidates(self) -> list[ModelCandidate]:
        rows = sorted(
            {idx.row() for idx in self.table.selectionModel().selectedRows()}
        )
        return [self._rows[r] for r in rows if 0 <= r < len(self._rows)]

    def _selected_candidate(self) -> ModelCandidate | None:
        cands = self._selected_candidates()
        return cands[0] if cands else None

    def _select_last_catalog_rows(self) -> None:
        """换包后选中末轮。"""
        n = len(self._rows)
        if n <= 0:
            self.table.clearSelection()
            return
        self.table.selectRow(n - 1)

    def _preview_catalog_selection(self) -> None:
        """换运行包后必须显式重绘：末行下标常与上一包相同，selectRow 不会再发选中信号。"""
        if not self._rows:
            self._path = None
            self._model_pg.show_empty("该运行包尚无写出的 smesh")
            set_wrapping_caption(self.lbl_mesh, "尚无模型可预览")
            return
        self._on_selection()

    def _on_selection(self) -> None:
        if self._skip_table_sel:
            return
        cands = self._selected_candidates()
        if not cands:
            self._path = None
            self._model_pg.show_empty("未选中模型")
            set_wrapping_caption(self.lbl_mesh, "未选中模型")
            return
        self._preview(cands[0].path)

    def _run_name_of(self, cand: ModelCandidate) -> str:
        if cand.run_name:
            return cand.run_name
        rd = infer_run_dir_from_smesh(cand.path)
        if rd is not None:
            return rd.name
        if self._spec is not None and self._spec.run_dir is not None:
            return self._spec.run_dir.name
        return "—"

    def _fill_table(self) -> None:
        self.table.setRowCount(len(self._rows))
        for i, c in enumerate(self._rows):
            run = self._run_name_of(c)
            vals = [
                run,
                str(c.iter),
                str(c.iset),
                _fmt(c.w_sv),
                _fmt(c.w_sd),
                _fmt(c.chi2),
                _fmt(c.rms),
                _fmt(c.pred_chi),
                _fmt(c.rough_v),
                c.path.name,
            ]
            tip_bits = [
                str(c.path),
                f"运行包={run}",
                f"-SV={_fmt(c.w_sv)}",
                f"-SD={_fmt(c.w_sd)}",
            ]
            if c.w_dv is not None:
                tip_bits.append(f"wdv={_fmt(c.w_dv)}")
            if c.w_dd is not None:
                tip_bits.append(f"wdd={_fmt(c.w_dd)}")
            tip = "\n".join(tip_bits)
            for j, text in enumerate(vals):
                item = QTableWidgetItem(text)
                if 1 <= j <= 8:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                item.setData(Qt.ItemDataRole.UserRole, i)
                item.setToolTip(tip)
                self.table.setItem(i, j, item)

    def _on_contours_toggled(self, on: bool) -> None:
        from ..plots.velocity_contours import set_contours_enabled

        set_contours_enabled(self.state, bool(on))
        self._refresh_preview(reset_home=False)

    def _on_dws_toggled(self, on: bool) -> None:
        from ..services.dws_plot import set_dws_mask_enabled

        set_dws_mask_enabled(self.state, bool(on))
        self._refresh_preview(reset_home=False)

    def _refresh_preview(self, *, reset_home: bool = True) -> None:
        if self._path is not None:
            self._preview(self._path, reset_home=reset_home)

    def _preview(self, path: Path, *, reset_home: bool = True) -> None:
        try:
            work = resolve_work_dir(self.state.get_str("work_dir"))
            self._model_pg.set_save_dir(work)
            label = self._fill_model_widget(
                self._model_pg, path, work, reset_home=reset_home
            )
            self._path = path
            set_wrapping_caption(self.lbl_mesh, *label.split("\n", 1))
        except Exception as e:
            self._model_pg.show_empty(f"绘制失败: {e}")
            set_wrapping_caption(self.lbl_mesh, f"绘制失败: {e}")

    def _fill_model_widget(
        self,
        widget: MonitorModelWidget,
        path: Path,
        work: Path,
        *,
        reset_home: bool = True,
    ) -> str:
        from ..plots.velocity_contours import contour_specs_for_state
        from ..services.dws_plot import dws_xyz_for_plot
        from ..services.obs_stations import load_obs_context

        refl = resolve_plot_refl_for_smesh(path, self.state, work)
        mesh, ds, extra = load_smesh_plot_data(path, refl)
        cmap = resolve_plot_smesh_cmap(self.state, work)
        dws_xyz = dws_xyz_for_plot(
            self.state,
            work,
            path,
            out_root=self._spec.out_root if self._spec else None,
            run_dir=self._spec.run_dir if self._spec else None,
            enabled=self.ck_dws.isChecked(),
        )
        title = path.name + ("  ·  DWS" if dws_xyz is not None else "")
        xmin, xmax = widget.set_velocity(
            ds,
            mesh,
            extra,
            cmap,
            title,
            reset_home=reset_home,
            contour_specs=contour_specs_for_state(self.state),
            dws_xyz=dws_xyz,
        )
        widget.set_model_source(path, self.state)
        ctx = load_obs_context(self.state, work)
        n_obs = widget.add_stations(
            ctx.stations,
            x_range=(xmin, xmax),
            label_ids=set(ctx.isrc_to_obs.values()) if ctx.isrc_to_obs else None,
        )
        tres_note = self._apply_tres_fit(widget, path, ctx)
        extras: list[str] = []
        if refl:
            extras.append(f"refl {Path(refl).name}")
        if n_obs:
            obs_note = f"OBS×{n_obs}"
            if ctx.station_path is not None:
                obs_note = f"{obs_note} {ctx.station_path.name}"
            extras.append(obs_note)
        if tres_note:
            extras.append(tres_note)
        line2 = " · ".join(extras)
        return f"{path.name}\n{line2}" if line2 else path.name

    def _apply_tres_fit(self, widget: MonitorModelWidget, path: Path, ctx) -> str:
        from ..services.smesh_ops import parse_inverse_smesh_name
        from ..services.tres_sample import load_outliers_for_monitor, load_tres_for_monitor

        if self._spec is None or self._spec.out_root is None:
            widget.set_residuals(None, ctx, note="走时拟合（残差 vs 接收点 X）")
            return ""
        key = parse_inverse_smesh_name(path)
        iter_prefer = key[0] if key else None
        iset_prefer = key[1] if key else None
        try:
            groups, note = load_tres_for_monitor(
                self._spec.out_root,
                iter_prefer=iter_prefer,
                run_dir=self._spec.run_dir,
            )
        except Exception as e:
            widget.set_residuals(None, ctx, note=f"残差读取失败: {e}")
            return str(e)
        outliers: list = []
        try:
            outliers, onote = load_outliers_for_monitor(
                self._spec.out_root,
                iter_prefer=iter_prefer,
                iset_prefer=iset_prefer,
            )
            if onote:
                note = f"{note} · {onote}" if note else onote
        except Exception:
            outliers = []
        if not groups:
            widget.set_residuals(
                None, ctx, note=note or "本轮残差写出后显示（需 out_level≥1）"
            )
            return note
        widget.set_residuals(
            groups, ctx, note=note, stations=ctx.stations, outliers=outliers or None
        )
        return note

    def plot_selected_window(self) -> None:
        c = self._selected_candidate()
        if c is None:
            show_modeless_message(
                "模型挑选", "请先选中一行", icon=QMessageBox.Icon.Warning
            )
            return
        try:
            work = resolve_work_dir(self.state.get_str("work_dir"))
            win = MonitorModelWidget()
            win.resize(960, 720)
            win.set_save_dir(work)
            win.setWindowTitle(f"选用预览 · {c.path.name}")
            self._fill_model_widget(win, c.path, work)
            show_modeless_dialog(win, activate=True)
        except Exception as e:
            show_modeless_message(
                "绘制失败", str(e), icon=QMessageBox.Icon.Warning
            )

    def closeEvent(self, event) -> None:  # noqa: N802
        global _singleton
        try:
            save_window_layout(
                self, _LAYOUT_KEY, splitters={"main": self._split}
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


def open_model_picker_dialog(
    state: FormState,
    parent=None,
    *,
    spec: InvMonitorSpec | None = None,
) -> ModelPickerDialog:
    """单例非模态挑选窗。"""
    global _singleton
    dlg = None
    if _singleton is not None:
        try:
            _ = _singleton.isVisible()
            dlg = _singleton
        except RuntimeError:
            _singleton = None

    if dlg is None:
        dlg = ModelPickerDialog(state, parent)
        _singleton = dlg
        show_modeless_dialog(dlg, activate=True)
    else:
        dlg.state = state
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    if spec is not None:
        dlg.attach_spec(spec)
    elif dlg._spec is None:
        dlg.reload_runs()
    else:
        dlg.reload_runs(prefer=dlg._spec.run_dir)
    return dlg
