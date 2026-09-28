"""反演结果分析（tt_inverse -L 日志）非模态工具窗。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent, QTextCursor
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..dialog_utils import file_dialog_options, show_modeless_dialog, show_modeless_message
from ..services.paths import (
    resolve_existing_file,
    resolve_work_dir,
    split_log_path_list,
    to_workdir_relative,
)
from ..state.form_state import FormState


class _MultiLogEdit(QPlainTextEdit):
    """拖入/粘贴多个日志时写入真正的换行（界面折行 ≠ 文档换行）。"""

    def __init__(self, dialog: "InvAnalysisDialog") -> None:
        super().__init__(dialog)
        self._dialog = dialog
        self.setAcceptDrops(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setPlaceholderText(
            "每行一个日志路径。可从资源管理器一次拖入多个 .log；"
            "长路径会折行显示，文件之间仍是独立一行。"
        )

    def canInsertFromMimeData(self, source) -> bool:  # noqa: N802
        if source is None:
            return False
        if source.hasUrls() or source.hasFormat("text/uri-list"):
            return True
        return super().canInsertFromMimeData(source)

    def insertFromMimeData(self, source) -> None:  # noqa: N802
        if self._dialog._ingest_log_mime(source):
            return
        super().insertFromMimeData(source)


class InvAnalysisDialog(QDialog):
    def __init__(self, state: FormState, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("反演结果分析（tt_inverse -L 日志）")
        self.resize(720, 540)
        self.setAcceptDrops(True)

        root = QVBoxLayout(self)
        self.nb = QTabWidget()
        root.addWidget(self.nb)

        # --- 单日志 ---
        tab1 = QWidget()
        t1 = QVBoxLayout(tab1)
        t1.addWidget(
            QLabel("tt_inverse.log 路径（可相对 work_dir；运行包内多为 outputs/tt_inverse.log）：")
        )
        row1 = QHBoxLayout()
        self.single_edit = QLineEdit()
        btn_b1 = QPushButton("浏览…")
        btn_b1.clicked.connect(self._browse_single)
        row1.addWidget(self.single_edit, stretch=1)
        row1.addWidget(btn_b1)
        t1.addLayout(row1)
        btn_run1 = QPushButton("绘制 RMS / χ² 随迭代")
        btn_run1.clicked.connect(self._run_single)
        t1.addWidget(btn_run1)
        t1.addStretch(1)
        self.nb.addTab(tab1, "单日志")

        # --- 多日志 ---
        tab2 = QWidget()
        t2 = QVBoxLayout(tab2)
        t2.addWidget(
            QLabel(
                "每个日志单独一行（相对 work_dir）。可拖入 / 粘贴多个文件；"
                "「添加文件…」也可一次多选。"
            )
        )
        self.multi_edit = _MultiLogEdit(self)
        t2.addWidget(self.multi_edit, stretch=1)
        btn_add = QPushButton("添加文件…")
        btn_add.clicked.connect(self._add_multi)
        t2.addWidget(btn_add)
        row_w = QHBoxLayout()
        row_w.addWidget(QLabel("粗糙度权重 w (score=pred_chi*(1+w*R)):"))
        self.w_edit = QLineEdit("0.001")
        self.w_edit.setMaximumWidth(100)
        row_w.addWidget(self.w_edit)
        row_w.addStretch(1)
        t2.addLayout(row_w)
        btn_run2 = QPushButton("生成多日志分析（叠画+Pareto+参数影响+汇总表）")
        btn_run2.clicked.connect(self._run_multi)
        t2.addWidget(btn_run2)
        self.nb.addTab(tab2, "多日志")
        tab2.setAcceptDrops(True)
        tab2.installEventFilter(self)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        close_row.addWidget(btn_close)
        root.addLayout(close_row)

    def _work(self) -> Path:
        return resolve_work_dir(self.state.get_str("work_dir"))

    def _resolve(self, path_str: str) -> Path:
        return resolve_existing_file(path_str, self._work())

    def _drop_mime_has_files(self, md) -> bool:
        if md is None:
            return False
        try:
            return md.hasUrls() or md.hasText() or md.hasFormat("text/uri-list")
        except Exception:
            return False

    def _ingest_log_mime(self, md) -> bool:
        from .smesh_plot import drop_local_paths

        class _Ev:
            def mimeData(self):
                return md

        paths = [str(p) for p in drop_local_paths(_Ev()) if p]
        if not paths and md is not None and md.hasText():
            paths = split_log_path_list(md.text())
        logs = [p for p in paths if self._looks_like_log(p)]
        if not logs:
            logs = paths
        if not logs:
            return False
        self._append_multi_paths(logs)
        self.nb.setCurrentIndex(1)
        return True

    @staticmethod
    def _looks_like_log(path: str) -> bool:
        name = Path(str(path)).name.lower()
        return name.endswith((".log", ".txt")) or "tt_inverse" in name

    def _append_multi_paths(self, paths: list[str]) -> None:
        existing = split_log_path_list(self.multi_edit.toPlainText())
        seen = {p.replace("\\", "/").casefold() for p in existing}
        work = self._work()
        for path in paths:
            raw = str(path or "").strip()
            if not raw:
                continue
            try:
                from .smesh_plot_core import normalize_dropped_path

                abs_p = normalize_dropped_path(raw)
                if not abs_p.is_absolute():
                    abs_p = work / abs_p
                rel = to_workdir_relative(
                    str(abs_p.resolve()) if abs_p.exists() else str(abs_p),
                    work,
                    warn_outside=False,
                ).value
            except Exception:
                rel = raw.replace("\\", "/")
            key = rel.replace("\\", "/").casefold()
            if key in seen:
                continue
            seen.add(key)
            existing.append(rel)
        self.multi_edit.setPlainText("\n".join(existing))
        self.multi_edit.moveCursor(QTextCursor.MoveOperation.End)

    def eventFilter(self, watched, event):  # noqa: N802
        if event is None:
            return super().eventFilter(watched, event)
        t = event.type()
        if t in (
            QEvent.Type.DragEnter,
            QEvent.Type.DragMove,
            QEvent.Type.Drop,
        ):
            if t == QEvent.Type.Drop:
                self.dropEvent(event)
            elif t == QEvent.Type.DragMove:
                self.dragMoveEvent(event)
            else:
                self.dragEnterEvent(event)
            return True
        return super().eventFilter(watched, event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if self._drop_mime_has_files(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # noqa: N802
        self.dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        if self._ingest_log_mime(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def _browse_single(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 tt_inverse 日志文件",
            str(self._work()),
            "log (*.log);;文本 (*.txt);;所有文件 (*)",
            options=file_dialog_options(),
        )
        if path:
            rel = to_workdir_relative(path, self._work(), warn_outside=False).value
            self.single_edit.setText(rel)

    def _add_multi(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择多个 tt_inverse 日志",
            str(self._work()),
            "log (*.log);;文本 (*.txt);;所有文件 (*)",
            options=file_dialog_options(),
        )
        if paths:
            self._append_multi_paths(paths)

    def _tila(self):
        try:
            from ... import tt_inverse_log_analysis as tila
        except ImportError:
            from pyAOBS.modeling.tomo2d import tt_inverse_log_analysis as tila
        return tila

    def _run_single(self) -> None:
        try:
            tila = self._tila()
            p = self._resolve(self.single_edit.text())
            rows = tila.parse_tt_inverse_log(p)
            from ..plots.inv_analysis_pg import show_single_log_window

            show_single_log_window(rows, title=str(p.name), save_dir=str(p.parent))
        except Exception as e:
            show_modeless_message("反演分析", str(e), icon=QMessageBox.Icon.Warning)

    def _run_multi(self) -> None:
        try:
            wgt = float(self.w_edit.text().strip() or "0.001")
        except ValueError:
            show_modeless_message(
                "反演分析", "粗糙度权重 w 须为数字", icon=QMessageBox.Icon.Warning
            )
            return
        lines = split_log_path_list(self.multi_edit.toPlainText())
        if len(lines) < 2:
            show_modeless_message(
                "反演分析",
                "多日志分析至少需要 2 个日志路径（每个文件单独一行；"
                "可从资源管理器拖入多个 .log）。",
                icon=QMessageBox.Icon.Warning,
            )
            return
        tila = self._tila()
        series: dict[str, list] = {}
        path_by_name: dict[str, Path] = {}
        for i, ln in enumerate(lines):
            try:
                p = self._resolve(ln)
                rows = tila.parse_tt_inverse_log(p)
                if not rows:
                    raise ValueError(f"无有效数据行: {p}")
                key = p.name
                if key in series:
                    key = f"{key} [{i+1}]"
                series[key] = rows
                path_by_name[key] = p
            except Exception as e:
                show_modeless_message("反演分析", str(e), icon=QMessageBox.Icon.Warning)
                return
        try:
            from ..plots.inv_analysis_pg import (
                show_overlay_window,
                show_param_influence_window,
                show_pareto_window,
                show_summary_table_window,
            )
            from ..plots.pareto_log_menu import SeriesHighlightGroup, wire_pareto_log_menu

            group = SeriesHighlightGroup()
            menu_kw = dict(
                state=self.state,
                path_by_name=path_by_name,
                rows_by_name=series,
                rough_weight=wgt,
                group=group,
            )
            overlay = show_overlay_window(series, title="多日志：折射/反射 RMS 叠画")
            wire_pareto_log_menu(overlay, **menu_kw)
            pareto = show_pareto_window(
                series, rough_weight=wgt, title="多日志：Pareto 与综合得分"
            )
            wire_pareto_log_menu(pareto, **menu_kw)
            infl = show_param_influence_window(
                series, title="多日志：反演参数与指标（平滑/阻尼）"
            )
            wire_pareto_log_menu(infl, **menu_kw)
            table = show_summary_table_window(
                series,
                rough_weight=wgt,
                title="多日志：末步参数与指标汇总表",
            )
            wire_pareto_log_menu(table, **menu_kw)
        except Exception as e:
            show_modeless_message("反演分析", str(e), icon=QMessageBox.Icon.Critical)


def open_inv_analysis_dialog(state: FormState, parent=None) -> InvAnalysisDialog:
    dlg = InvAnalysisDialog(state, parent)
    show_modeless_dialog(dlg, activate=True)
    return dlg
