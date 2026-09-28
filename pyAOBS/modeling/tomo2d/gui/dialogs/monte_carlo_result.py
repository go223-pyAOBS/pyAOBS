"""蒙特卡洛结果：非模态两幅图（均值 Vp / 误差 σ + 界面均值 ±σ）。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..dialog_utils import (
    add_combo_path_item,
    file_dialog_options,
    show_modeless_dialog,
    show_modeless_message,
    wire_path_combo_tooltips,
)
from ..plots.inv_monitor_model import MonitorModelWidget
from ..state.form_state import FormState


class MonteCarloResultWindow(QWidget):
    """独立绘图窗：均值 Vp、误差 σ，叠界面均值 ±σ；可换运行包。"""

    def __init__(
        self,
        state: FormState,
        *,
        pull: Callable[[], None] | None = None,
        run_dir: Path | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self._pull = pull
        self._run_dir = Path(run_dir) if run_dir is not None else None
        self._filling = False
        self.setWindowTitle("蒙特卡洛结果")
        self.resize(980, 980)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        pack = QHBoxLayout()
        pack.addWidget(QLabel("结果包"))
        self.combo_run = QComboBox()
        self.combo_run.setMinimumWidth(360)
        self.combo_run.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.combo_run.setMinimumContentsLength(28)
        pack.addWidget(self.combo_run, stretch=1)
        btn_browse = QPushButton("浏览…")
        btn_browse.setToolTip("自选蒙特卡洛运行包（须含 outputs/mean_velocity.smesh）")
        btn_browse.clicked.connect(self._browse_run)
        btn_reload = QPushButton("刷新列表")
        btn_reload.setToolTip("重扫工区 runs/montecarlo_*，并重画当前包")
        btn_reload.clicked.connect(self._reload_list_and_plot)
        pack.addWidget(btn_browse)
        pack.addWidget(btn_reload)
        root.addLayout(pack)

        bar = QHBoxLayout()
        self.ck_contours = QCheckBox("叠加等值线")
        from ..plots.velocity_contours import contours_enabled
        from ..widgets.smesh_cmap_combo import SmeshCmapCombo

        self.ck_contours.setChecked(contours_enabled(self.state))
        self.cmap_combo = SmeshCmapCombo(
            self.state, on_changed=lambda *_a: self._paint(reset_home=False)
        )
        self.ck_dws = QCheckBox("DWS 遮罩")
        from ..services.dws_plot import dws_mask_enabled

        self.ck_dws.setChecked(dws_mask_enabled(self.state))
        self.ck_dws.setToolTip(
            "勾选：按均值 smesh 就近找 DWS。无覆盖留白；有覆盖按 log(DWS) 透明。"
        )
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        bar.addWidget(self.ck_contours)
        bar.addWidget(self.cmap_combo)
        bar.addWidget(self.ck_dws)
        bar.addStretch(1)
        bar.addWidget(btn_close)
        root.addLayout(bar)

        self.plot = MonitorModelWidget(self, compare_bar=False)
        self.plot.set_stats_context_handler(self._on_stats_context_menu)
        self.plot.show_empty_stack("等待读取蒙特卡洛结果…")
        self.plot.set_interaction_hint(
            "上：均值 Vp · 下：误差 σ · 红线/色带：界面均值 ±σ · 右键保存 / 写入表单"
        )
        root.addWidget(self.plot, stretch=1)

        wire_path_combo_tooltips(
            self.combo_run,
            hint="工区 runs/ 下的完整蒙特卡洛包；悬停看完整路径。",
        )
        self.combo_run.currentIndexChanged.connect(self._on_run_chosen)
        self.ck_contours.toggled.connect(self._on_contours_toggled)
        self.ck_dws.toggled.connect(self._on_dws_toggled)

    def _work(self):
        from ..services.paths import resolve_work_dir

        if callable(self._pull):
            self._pull()
        return resolve_work_dir(self.state.get_str("work_dir"))

    def _fill_combo(self, work: Path, select: Path | None) -> None:
        from ..services.qc_workflows import (
            list_monte_carlo_runs,
            monte_carlo_run_label,
        )

        runs = list_monte_carlo_runs(work)
        if select is not None:
            sel = Path(select)
            try:
                sel = sel.resolve()
            except OSError:
                pass
            if not any(self._same_run(p, sel) for p in runs):
                runs = [sel] + runs
        self._filling = True
        try:
            self.combo_run.blockSignals(True)
            self.combo_run.clear()
            for p in runs:
                add_combo_path_item(self.combo_run, monte_carlo_run_label(p), str(p))
            idx = 0
            if select is not None:
                for i in range(self.combo_run.count()):
                    if self._same_run(self.combo_run.itemData(i), select):
                        idx = i
                        break
            if self.combo_run.count():
                self.combo_run.setCurrentIndex(idx)
        finally:
            self.combo_run.blockSignals(False)
            self._filling = False

    @staticmethod
    def _same_run(a, b) -> bool:
        if a is None or b is None:
            return False
        try:
            return Path(a).resolve() == Path(b).resolve()
        except OSError:
            return Path(str(a)) == Path(str(b))

    def _on_run_chosen(self, _idx: int = 0) -> None:
        if self._filling:
            return
        data = self.combo_run.currentData()
        if not data:
            return
        self._run_dir = Path(str(data))
        self._paint(reset_home=True)

    def _on_contours_toggled(self, on: bool) -> None:
        from ..plots.velocity_contours import set_contours_enabled

        set_contours_enabled(self.state, bool(on))
        self._paint(reset_home=False)

    def _on_dws_toggled(self, on: bool) -> None:
        from ..services.dws_plot import set_dws_mask_enabled

        set_dws_mask_enabled(self.state, bool(on))
        self._paint(reset_home=False)

    def _reload_list_and_plot(self) -> None:
        try:
            work = self._work()
        except Exception as e:
            show_modeless_message("蒙特卡洛结果图", str(e))
            return
        current = self._run_dir
        self._fill_combo(work, current)
        if self.combo_run.count() == 0:
            show_modeless_message(
                "蒙特卡洛结果图", "没有完整的蒙特卡洛结果。请先「运行蒙特卡洛分析」。"
            )
            return
        data = self.combo_run.currentData()
        self._run_dir = Path(str(data)) if data else None
        self._paint(reset_home=True)

    def _browse_run(self) -> None:
        from ..services.qc_workflows import resolve_monte_carlo_run_dir

        try:
            work = self._work()
        except Exception as e:
            show_modeless_message("蒙特卡洛结果图", str(e))
            return
        start = str(work / "runs") if (work / "runs").is_dir() else str(work)
        picked = QFileDialog.getExistingDirectory(
            self, "选择蒙特卡洛结果包", start, file_dialog_options()
        )
        if not picked:
            return
        p = Path(picked)
        last_err = ""
        for cand in (p, p.parent, p.parent.parent):
            try:
                hit = resolve_monte_carlo_run_dir(work, cand)
            except FileNotFoundError as e:
                last_err = str(e)
                continue
            self._run_dir = hit
            self._fill_combo(work, hit)
            self._paint(reset_home=True)
            return
        show_modeless_message(
            "蒙特卡洛结果图",
            last_err
            or (
                "所选目录不是完整蒙特卡洛包（需要 outputs/mean_velocity.smesh "
                f"与 std_velocity.smesh）。\n{p}"
            ),
        )

    def refresh(self, *, reset_home: bool = True) -> None:
        try:
            work = self._work()
        except Exception as e:
            show_modeless_message("蒙特卡洛结果图", str(e))
            return
        self._fill_combo(work, self._run_dir)
        if self.combo_run.count() == 0:
            show_modeless_message(
                "蒙特卡洛结果图", "没有完整的蒙特卡洛结果。请先「运行蒙特卡洛分析」。"
            )
            return
        data = self.combo_run.currentData()
        if data:
            self._run_dir = Path(str(data))
        self._paint(reset_home=reset_home)

    def _last_stat(self):
        return getattr(self.plot, "_mc_stat", None)

    def _on_stats_context_menu(self, panel: str, menu: QMenu) -> None:
        from ..services.model_compare import MESH_FORM_FIELDS, REFL_FORM_FIELDS

        stat = self._last_stat()
        if stat is None:
            return
        has_refl = stat.refl_x is not None and stat.refl_mean is not None
        has_refl_std = has_refl and stat.refl_std is not None
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
        from ..services.paths import resolve_work_dir

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
        from ..services.file_filters import SMESH_SAVE_FILTERS
        from ..services.model_compare import save_ensemble_velocity

        if self._last_stat() is None:
            return
        if kind == "mean":
            dest = self._pick_save_path(
                "保存均值速度", "mc_mean_velocity.smesh", SMESH_SAVE_FILTERS
            )
        else:
            dest = self._pick_save_path(
                "保存速度误差 σ", "mc_std_velocity.smesh", SMESH_SAVE_FILTERS
            )
        if dest is None:
            return
        try:
            saved = save_ensemble_velocity(self._last_stat(), dest, kind=kind)
        except Exception as e:
            show_modeless_message("保存失败", str(e), icon=QMessageBox.Icon.Warning)
            return
        show_modeless_message("已保存", str(saved), icon=QMessageBox.Icon.Information)

    def _save_stats_reflector(self, which: str) -> None:
        from ..services.file_filters import REFL_SAVE_FILTERS
        from ..services.model_compare import save_ensemble_reflector

        if self._last_stat() is None:
            return
        names = {
            "mean": ("保存平均反射面", "mc_mean_moho.refl"),
            "minus": ("保存反射面 mean−σ", "mc_mean_moho_m1sigma.refl"),
            "plus": ("保存反射面 mean+σ", "mc_mean_moho_p1sigma.refl"),
        }
        caption, default = names.get(which, ("保存反射面", "mc_moho.refl"))
        dest = self._pick_save_path(caption, default, REFL_SAVE_FILTERS)
        if dest is None:
            return
        try:
            saved = save_ensemble_reflector(self._last_stat(), dest, which=which)
        except Exception as e:
            show_modeless_message("保存失败", str(e), icon=QMessageBox.Icon.Warning)
            return
        show_modeless_message("已保存", str(saved), icon=QMessageBox.Icon.Information)

    def _apply_stats_to_form(self, kind: str, field_key: str) -> None:
        from ..services.model_compare import write_ensemble_stat_files, write_path_to_form
        from ..services.paths import resolve_work_dir

        stat = self._last_stat()
        if stat is None:
            return
        try:
            work = resolve_work_dir(self.state.get_str("work_dir"))
            stat = write_ensemble_stat_files(stat, work)
            setattr(self.plot, "_mc_stat", stat)
        except Exception as e:
            show_modeless_message("写出失败", str(e), icon=QMessageBox.Icon.Warning)
            return
        path = stat.mean_smesh if kind == "mean" else stat.mean_refl
        if path is None:
            show_modeless_message(
                "蒙特卡洛结果",
                "没有可写回的文件（反射面需至少 2 条配套界面）。",
                icon=QMessageBox.Icon.Warning,
            )
            return
        try:
            rel = write_path_to_form(self.state, field_key, path)
        except Exception as e:
            show_modeless_message("写出失败", str(e), icon=QMessageBox.Icon.Warning)
            return
        show_modeless_message(
            "已写回表单",
            f"{field_key} ← {rel}",
            icon=QMessageBox.Icon.Information,
        )

    def _paint(self, *, reset_home: bool = True) -> None:
        from ..services.paths import resolve_work_dir
        from ..services.qc_workflows import paint_monte_carlo_result

        if callable(self._pull):
            self._pull()
        try:
            work = resolve_work_dir(self.state.get_str("work_dir"))
            run = paint_monte_carlo_result(
                self.plot,
                self.state,
                work,
                run_dir=self._run_dir,
                reset_home=reset_home,
            )
            self._run_dir = run
            self.setWindowTitle(f"蒙特卡洛结果 · {run.name}")
        except Exception as e:
            show_modeless_message("蒙特卡洛结果图", str(e))


def open_monte_carlo_result(
    state: FormState,
    *,
    pull: Callable[[], None] | None = None,
    run_dir: Path | str | None = None,
    existing: MonteCarloResultWindow | None = None,
) -> MonteCarloResultWindow | None:
    """打开或刷新非模态结果窗。没有完整结果时提示并返回 None。"""
    from ..services.paths import resolve_work_dir
    from ..services.qc_workflows import resolve_monte_carlo_run_dir

    if callable(pull):
        pull()
    try:
        work = resolve_work_dir(state.get_str("work_dir"))
        resolved = resolve_monte_carlo_run_dir(work, run_dir)
    except Exception as e:
        show_modeless_message("蒙特卡洛结果图", str(e))
        return existing

    win = existing
    try:
        if win is not None and win.isVisible():
            win._run_dir = resolved
            win.refresh(reset_home=True)
            win.raise_()
            win.activateWindow()
            return win
    except RuntimeError:
        win = None

    win = MonteCarloResultWindow(state, pull=pull, run_dir=resolved)
    try:
        win.refresh(reset_home=True)
    except Exception as e:
        show_modeless_message("蒙特卡洛结果图", str(e))
        win.deleteLater()
        return None
    show_modeless_dialog(win, activate=True)
    return win
