"""棋盘格测试结果：非模态三幅图（真异常 / 恢复异常 / 残差）。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
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


class CheckerboardResultWindow(QWidget):
    """独立绘图窗：真异常%、恢复异常%、残差%；可换测试包、勾选 DWS。"""

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
        self.setWindowTitle("棋盘格结果")
        self.resize(960, 980)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        pack = QHBoxLayout()
        pack.addWidget(QLabel("测试包"))
        self.combo_run = QComboBox()
        self.combo_run.setMinimumWidth(360)
        self.combo_run.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.combo_run.setMinimumContentsLength(28)
        pack.addWidget(self.combo_run, stretch=1)
        btn_browse = QPushButton("浏览…")
        btn_browse.setToolTip("自选棋盘格运行包目录（须含 outputs/true|recovered_anomaly_pct.smesh）")
        btn_browse.clicked.connect(self._browse_run)
        btn_reload = QPushButton("刷新列表")
        btn_reload.setToolTip("重扫工区 runs/checkerboard_*，并重画当前包")
        btn_reload.clicked.connect(self._reload_list_and_plot)
        pack.addWidget(btn_browse)
        pack.addWidget(btn_reload)
        root.addLayout(pack)

        bar = QHBoxLayout()
        self.ck_dws = QCheckBox("DWS 遮罩")
        from ..services.dws_plot import dws_mask_enabled

        self.ck_dws.setChecked(dws_mask_enabled(self.state))
        self.ck_dws.setToolTip(
            "勾选：按所选运行包 outputs/dws（或 recovered.smesh 就近）遮罩。"
            "无覆盖留白；有覆盖按 log(DWS) 透明。三幅图共用。"
        )
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        bar.addWidget(self.ck_dws)
        bar.addStretch(1)
        bar.addWidget(btn_close)
        root.addLayout(bar)

        self.plot = MonitorModelWidget(self, compare_bar=False)
        self.plot.show_empty_stack("等待读取棋盘格结果…")
        self.plot.set_interaction_hint(
            "上：真异常% · 中：恢复异常% · 下：残差% · 右键写入表单"
        )
        root.addWidget(self.plot, stretch=1)

        wire_path_combo_tooltips(
            self.combo_run,
            hint="工区 runs/ 下的完整棋盘格包；悬停看完整路径。",
        )
        self.combo_run.currentIndexChanged.connect(self._on_run_chosen)
        self.ck_dws.toggled.connect(self._on_dws_toggled)

    def _work(self):
        from ..services.paths import resolve_work_dir

        if callable(self._pull):
            self._pull()
        return resolve_work_dir(self.state.get_str("work_dir"))

    def _fill_combo(self, work: Path, select: Path | None) -> None:
        from ..services.qc_workflows import (
            checkerboard_run_label,
            list_checkerboard_runs,
        )

        runs = list_checkerboard_runs(work)
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
                add_combo_path_item(self.combo_run, checkerboard_run_label(p), str(p))
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

    def _on_dws_toggled(self, on: bool) -> None:
        from ..services.dws_plot import set_dws_mask_enabled

        set_dws_mask_enabled(self.state, bool(on))
        self._paint(reset_home=False)

    def _reload_list_and_plot(self) -> None:
        try:
            work = self._work()
        except Exception as e:
            show_modeless_message("棋盘结果图", str(e))
            return
        current = self._run_dir
        self._fill_combo(work, current)
        if self.combo_run.count() == 0:
            show_modeless_message(
                "棋盘结果图", "没有完整的棋盘格结果。请先「运行棋盘格测试」。"
            )
            return
        data = self.combo_run.currentData()
        self._run_dir = Path(str(data)) if data else None
        self._paint(reset_home=True)

    def _browse_run(self) -> None:
        from ..services.qc_workflows import resolve_checkerboard_run_dir

        try:
            work = self._work()
        except Exception as e:
            show_modeless_message("棋盘结果图", str(e))
            return
        start = str(work / "runs") if (work / "runs").is_dir() else str(work)
        picked = QFileDialog.getExistingDirectory(
            self, "选择棋盘格测试包", start, file_dialog_options()
        )
        if not picked:
            return
        p = Path(picked)
        last_err = ""
        for cand in (p, p.parent, p.parent.parent):
            try:
                hit = resolve_checkerboard_run_dir(work, cand)
            except FileNotFoundError as e:
                last_err = str(e)
                continue
            self._run_dir = hit
            self._fill_combo(work, hit)
            self._paint(reset_home=True)
            return
        show_modeless_message(
            "棋盘结果图",
            last_err
            or (
                "所选目录不是完整棋盘格包（需要 outputs/true_anomaly_pct.smesh "
                f"与 recovered_anomaly_pct.smesh）。\n{p}"
            ),
        )

    def refresh(self, *, reset_home: bool = True) -> None:
        try:
            work = self._work()
        except Exception as e:
            show_modeless_message("棋盘结果图", str(e))
            return
        self._fill_combo(work, self._run_dir)
        if self.combo_run.count() == 0:
            show_modeless_message(
                "棋盘结果图", "没有完整的棋盘格结果。请先「运行棋盘格测试」。"
            )
            return
        data = self.combo_run.currentData()
        if data:
            self._run_dir = Path(str(data))
        self._paint(reset_home=reset_home)

    def _paint(self, *, reset_home: bool = True) -> None:
        from ..services.paths import resolve_work_dir
        from ..services.qc_workflows import paint_checkerboard_result

        try:
            work = resolve_work_dir(self.state.get_str("work_dir"))
            run = paint_checkerboard_result(
                self.plot,
                self.state,
                work,
                run_dir=self._run_dir,
                reset_home=reset_home,
            )
            self._run_dir = run
            self.setWindowTitle(f"棋盘格结果 · {run.name}")
        except Exception as e:
            show_modeless_message("棋盘结果图", str(e))


def open_checkerboard_result(
    state: FormState,
    *,
    pull: Callable[[], None] | None = None,
    run_dir: Path | str | None = None,
    existing: CheckerboardResultWindow | None = None,
) -> CheckerboardResultWindow | None:
    """打开或刷新非模态结果窗。没有完整结果时提示并返回 None。"""
    from ..services.paths import resolve_work_dir
    from ..services.qc_workflows import resolve_checkerboard_run_dir

    if callable(pull):
        pull()
    try:
        work = resolve_work_dir(state.get_str("work_dir"))
        resolved = resolve_checkerboard_run_dir(work, run_dir)
    except Exception as e:
        show_modeless_message("棋盘结果图", str(e))
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

    win = CheckerboardResultWindow(state, pull=pull, run_dir=resolved)
    try:
        win.refresh(reset_home=True)
    except Exception as e:
        show_modeless_message("棋盘结果图", str(e))
        win.deleteLater()
        return None
    show_modeless_dialog(win, activate=True)
    return win
