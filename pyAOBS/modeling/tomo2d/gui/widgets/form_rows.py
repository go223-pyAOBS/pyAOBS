"""表单行控件 + 与 FormState 的双向绑定。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..dialog_utils import file_dialog_options
from ..services.ui_prefs import (
    filter_recent_for_mode,
    list_recent_paths,
    push_recent_path,
)
from ..state.form_state import FormState


class FormBinder:
    """把 QLineEdit / QPlainTextEdit / QComboBox / QCheckBox / MinMaxRow 绑到 FormState 键。"""

    def __init__(self, state: FormState) -> None:
        self.state = state
        self._lines: dict[str, QLineEdit] = {}
        self._plains: dict[str, QPlainTextEdit] = {}
        self._combos: dict[str, QComboBox] = {}
        self._checks: dict[str, QCheckBox] = {}
        self._minmaxs: dict[str, MinMaxRow] = {}
        self._file_keys: list[str] = []

    def file_keys(self) -> list[str]:
        return list(self._file_keys)

    def bind_line(self, key: str, edit: QLineEdit, *, is_file: bool = False) -> None:
        self._lines[key] = edit
        if is_file:
            self._file_keys.append(key)
        if self.state.has(key):
            edit.setText(self.state.get_str(key))
        else:
            self.state.set(key, edit.text())
        edit.textChanged.connect(lambda t, k=key: self.state.set(k, t))

    def bind_plain(self, key: str, edit: QPlainTextEdit, *, is_file: bool = False) -> None:
        self._plains[key] = edit
        if is_file:
            self._file_keys.append(key)
        if self.state.has(key):
            edit.setPlainText(self.state.get_str(key))
        else:
            self.state.set(key, edit.toPlainText())
        edit.textChanged.connect(
            lambda k=key, e=edit: self.state.set(k, e.toPlainText())
        )

    def bind_combo(self, key: str, combo: QComboBox) -> None:
        self._combos[key] = combo
        if self.state.has(key):
            idx = combo.findText(self.state.get_str(key))
            if idx >= 0:
                combo.setCurrentIndex(idx)
        else:
            self.state.set(key, combo.currentText())
        combo.currentTextChanged.connect(lambda t, k=key: self.state.set(k, t))

    def bind_check(self, key: str, box: QCheckBox) -> None:
        self._checks[key] = box
        if self.state.has(key):
            box.setChecked(self.state.get_bool(key))
        else:
            self.state.set(key, box.isChecked())
        box.toggled.connect(lambda v, k=key: self.state.set(k, bool(v)))

    def bind_minmax(self, key: str, row: MinMaxRow) -> None:
        self._minmaxs[key] = row
        if self.state.has(key):
            row.set_text(self.state.get_str(key))
        else:
            self.state.set(key, row.text())
        row.changed.connect(lambda t, k=key: self.state.set(k, t))

    def pull_from_widgets(self) -> None:
        for k, e in self._lines.items():
            self.state.set(k, e.text())
        for k, e in self._plains.items():
            self.state.set(k, e.toPlainText())
        for k, c in self._combos.items():
            self.state.set(k, c.currentText())
        for k, b in self._checks.items():
            self.state.set(k, b.isChecked())
        for k, r in self._minmaxs.items():
            self.state.set(k, r.text())

    def push_to_widgets(self) -> None:
        for k, e in self._lines.items():
            e.blockSignals(True)
            e.setText(self.state.get_str(k))
            e.blockSignals(False)
        for k, e in self._plains.items():
            e.blockSignals(True)
            e.setPlainText(self.state.get_str(k))
            e.blockSignals(False)
        for k, c in self._combos.items():
            c.blockSignals(True)
            idx = c.findText(self.state.get_str(k))
            if idx >= 0:
                c.setCurrentIndex(idx)
            c.blockSignals(False)
        for k, b in self._checks.items():
            b.blockSignals(True)
            b.setChecked(self.state.get_bool(k))
            b.blockSignals(False)
        for k, r in self._minmaxs.items():
            r.set_text(self.state.get_str(k))


def labeled_line(parent: QWidget, label: str) -> tuple[QLabel, QLineEdit, QHBoxLayout]:
    row = QHBoxLayout()
    lb = QLabel(label, parent)
    edit = QLineEdit(parent)
    row.addWidget(lb)
    row.addWidget(edit, stretch=1)
    return lb, edit, row


class MinMaxRow(QWidget):
    """标签 + ``[min] ~ [max]``。状态仍写成 ``\"min max\"``，兼容旧工程。"""

    changed = Signal(str)

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.label = QLabel(label, self)
        self.lo = QLineEdit(self)
        self.hi = QLineEdit(self)
        self.lo.setPlaceholderText("min")
        self.hi.setPlaceholderText("max")
        self.lo.setFixedWidth(72)
        self.hi.setFixedWidth(72)
        tilde = QLabel("~", self)
        tilde.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.label)
        lay.addStretch(1)
        lay.addWidget(self.lo)
        lay.addWidget(tilde)
        lay.addWidget(self.hi)
        self.lo.textChanged.connect(self._emit)
        self.hi.textChanged.connect(self._emit)

    def _emit(self, *_a) -> None:
        self.changed.emit(self.text())

    def text(self) -> str:
        a = self.lo.text().strip()
        b = self.hi.text().strip()
        if a and b:
            return f"{a} {b}"
        return a or b

    def set_text(self, raw: str) -> None:
        s = (raw or "").strip().replace("～", "~").replace("—", " ").replace(",", " ")
        if "~" in s:
            parts = [p.strip() for p in s.split("~", 1)]
        else:
            parts = s.split()
        self.lo.blockSignals(True)
        self.hi.blockSignals(True)
        if not parts:
            self.lo.clear()
            self.hi.clear()
        elif len(parts) == 1:
            self.lo.setText(parts[0])
            self.hi.setText(parts[0])
        else:
            self.lo.setText(parts[0])
            self.hi.setText(parts[1])
        self.lo.blockSignals(False)
        self.hi.blockSignals(False)


def labeled_combo(
    parent: QWidget, label: str, values: list[str]
) -> tuple[QLabel, QComboBox, QHBoxLayout]:
    row = QHBoxLayout()
    lb = QLabel(label, parent)
    combo = QComboBox(parent)
    combo.addItems(values)
    row.addWidget(lb)
    row.addWidget(combo, stretch=1)
    return lb, combo, row


class PathRow(QWidget):
    """标签 + 路径编辑 + 浏览 + 最近路径；支持拖放文件/目录。"""

    def __init__(
        self,
        label: str,
        *,
        mode: str = "open_file",
        parent: QWidget | None = None,
        work_dir_getter: Callable[[], Path] | None = None,
        keep_absolute: bool = False,
        name_filter: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._work_dir_getter = work_dir_getter
        self._keep_absolute = keep_absolute
        self._name_filter = name_filter or "所有文件 (*)"
        self.setAcceptDrops(True)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(label)
        self.edit = QLineEdit()
        self.edit.setAcceptDrops(False)  # 由外层 PathRow 统一处理
        self.edit.setPlaceholderText("可拖放路径到此处…")
        self.edit.setToolTip(
            "可拖放文件/目录；右侧 ▾ 为最近路径；浏览后会记入最近列表。"
        )
        self.btn_recent = QToolButton()
        self.btn_recent.setText("▾")
        self.btn_recent.setToolTip("最近路径")
        self.btn_recent.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.btn_recent.setMenu(QMenu(self))
        self.btn_recent.menu().aboutToShow.connect(self._rebuild_recent_menu)
        self.btn = QPushButton("浏览…")
        self.btn.clicked.connect(self._browse)
        lay.addWidget(self.label)
        lay.addWidget(self.edit, stretch=1)
        lay.addWidget(self.btn_recent)
        lay.addWidget(self.btn)

    def _work(self) -> Path | None:
        if self._work_dir_getter is None:
            return None
        try:
            from ..services.paths import resolve_work_dir

            return resolve_work_dir(str(self._work_dir_getter()))
        except Exception:
            try:
                return Path(self._work_dir_getter()).expanduser()
            except Exception:
                return None

    def _start_dir(self) -> str:
        cur = self.edit.text().strip()
        if cur:
            p = Path(cur).expanduser()
            work = self._work()
            if not p.is_absolute() and work is not None:
                p = work / p
            if p.is_file():
                return str(p.parent)
            if p.is_dir():
                return str(p)
        work = self._work()
        if work is not None:
            return str(work)
        return ""

    def _normalize_path_text(self, path: str) -> str:
        raw = str(path or "").strip().strip('"')
        if not raw:
            return ""
        p = Path(raw).expanduser()
        if self._keep_absolute:
            try:
                if p.exists():
                    return str(p.resolve()).replace("\\", "/")
            except OSError:
                pass
            if p.is_absolute():
                return str(p).replace("\\", "/")
            # 相对路径：相对进程 cwd 转绝对，避免子进程 cwd=work_dir 时找不到
            try:
                return str((Path.cwd() / p).resolve()).replace("\\", "/")
            except OSError:
                return str((Path.cwd() / p)).replace("\\", "/")
        work = self._work()
        if work is not None:
            try:
                from ..services.paths import to_workdir_relative

                if p.is_absolute() or (work / p).exists():
                    abs_p = p if p.is_absolute() else (work / p)
                    return to_workdir_relative(str(abs_p.resolve()), work).value
            except Exception:
                pass
        try:
            if p.exists():
                return str(p.resolve()) if p.is_absolute() else raw.replace("\\", "/")
        except OSError:
            pass
        return raw.replace("\\", "/")

    def set_path(self, path: str, *, remember: bool = True) -> None:
        text = self._normalize_path_text(path)
        if not text:
            return
        self.edit.setText(text)
        if remember:
            # 最近列表存绝对路径，便于跨工区复用
            try:
                abs_p = Path(path).expanduser()
                work = self._work()
                if work is not None and not abs_p.is_absolute():
                    abs_p = work / abs_p
                if abs_p.exists():
                    push_recent_path(str(abs_p.resolve()))
                else:
                    push_recent_path(text)
            except Exception:
                push_recent_path(text)

    def _browse(self) -> None:
        start = self._start_dir()
        if self._mode == "dir":
            path = QFileDialog.getExistingDirectory(
                self,
                "选择目录",
                start,
                file_dialog_options(QFileDialog.Option.ShowDirsOnly),
            )
        elif self._mode == "save_file":
            path, _ = QFileDialog.getSaveFileName(
                self,
                "保存为",
                start,
                self._name_filter,
                options=file_dialog_options(),
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "选择文件",
                start,
                self._name_filter,
                options=file_dialog_options(),
            )
        if path:
            self.set_path(path, remember=True)

    def _rebuild_recent_menu(self) -> None:
        menu = self.btn_recent.menu()
        menu.clear()
        paths = filter_recent_for_mode(
            list_recent_paths(existing_only=False), mode=self._mode
        )
        if not paths:
            act = QAction("（无最近路径）", menu)
            act.setEnabled(False)
            menu.addAction(act)
            return
        for p in paths:
            act = QAction(p, menu)
            act.triggered.connect(lambda _c=False, path=p: self.set_path(path))
            menu.addAction(act)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if self._extract_drop_path(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        path = self._extract_drop_path(event)
        if path:
            self.set_path(path, remember=True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def _extract_drop_path(self, event) -> str | None:
        md = event.mimeData()
        if md is None:
            return None
        path: str | None = None
        if md.hasUrls():
            for url in md.urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    break
        elif md.hasText():
            t = md.text().strip().strip('"')
            if t:
                path = t.splitlines()[0].strip()
        if not path:
            return None
        p = Path(path)
        if self._mode == "dir":
            if p.is_file():
                return str(p.parent)
            return str(p)
        # 文件模式：若拖入目录则仍填入（用户可能手改文件名）
        return str(p)


class _WrappingPathEdit(QPlainTextEdit):
    """长路径折行显示，文件之间仍是文档里的独立一行（对齐反演分析多日志列表）。"""

    def __init__(self, row: "MultiPathRow") -> None:
        super().__init__(row)
        self._row = row
        self.setAcceptDrops(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)

    def canInsertFromMimeData(self, source) -> bool:  # noqa: N802
        if source is not None and (source.hasUrls() or source.hasFormat("text/uri-list")):
            return True
        return super().canInsertFromMimeData(source)

    def insertFromMimeData(self, source) -> None:  # noqa: N802
        if self._row._ingest_mime(source):
            return
        super().insertFromMimeData(source)


class MultiPathRow(QWidget):
    """多文件路径：多行文本（每行一个路径）+「添加多个…」；支持拖放多个文件。"""

    pathsCommitted = Signal()

    def __init__(
        self,
        label: str,
        *,
        parent: QWidget | None = None,
        work_dir_getter: Callable[[], Path] | None = None,
        name_filter: str | None = None,
        browse_caption: str | None = None,
        placeholder: str = (
            "每行一个路径。长路径会折行显示，文件之间仍是独立一行；"
            "可「添加多个…」或拖放 / 粘贴。"
        ),
    ) -> None:
        super().__init__(parent)
        self._work_dir_getter = work_dir_getter
        self._name_filter = name_filter or (
            "tx.in (*.in);;文本 (*.txt *.dat);;所有文件 (*)"
        )
        self._browse_caption = browse_caption or "选择一个或多个 tx.in"
        self.setAcceptDrops(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        top = QHBoxLayout()
        self.label = QLabel(label)
        top.addWidget(self.label)
        top.addStretch(1)
        self.btn = QPushButton("添加多个…")
        self.btn.clicked.connect(self._browse_many)
        top.addWidget(self.btn)
        root.addLayout(top)

        self.edit = _WrappingPathEdit(self)
        self.edit.setPlaceholderText(placeholder)
        self.edit.setMaximumBlockCount(0)
        self.edit.setMinimumHeight(96)
        root.addWidget(self.edit, stretch=1)

    def _work(self) -> Path | None:
        if self._work_dir_getter is None:
            return None
        try:
            from ..services.paths import resolve_work_dir

            return resolve_work_dir(str(self._work_dir_getter()))
        except Exception:
            try:
                return Path(self._work_dir_getter()).expanduser()
            except Exception:
                return None

    def _start_dir(self) -> str:
        cur = self.edit.toPlainText().strip().splitlines()
        if cur:
            p = Path(cur[-1].strip().strip('"')).expanduser()
            work = self._work()
            if not p.is_absolute() and work is not None:
                p = work / p
            if p.is_file():
                return str(p.parent)
            if p.is_dir():
                return str(p)
        work = self._work()
        if work is not None:
            return str(work)
        return ""

    def _normalize_one(self, path: str) -> str:
        raw = str(path or "").strip().strip('"').strip("'")
        if not raw:
            return ""
        p = Path(raw).expanduser()
        work = self._work()
        if work is not None:
            try:
                from ..services.paths import to_workdir_relative

                if p.is_absolute() or (work / p).exists():
                    abs_p = p if p.is_absolute() else (work / p)
                    return to_workdir_relative(str(abs_p.resolve()), work).value
            except Exception:
                pass
        try:
            if p.exists():
                return str(p.resolve()) if p.is_absolute() else raw.replace("\\", "/")
        except OSError:
            pass
        return raw.replace("\\", "/")

    def paths(self) -> list[str]:
        out: list[str] = []
        for line in self.edit.toPlainText().splitlines():
            s = line.strip().strip('"').strip("'")
            if s:
                out.append(s)
        return out

    def _ingest_mime(self, md) -> bool:
        if md is None:
            return False
        paths: list[str] = []
        try:
            if md.hasUrls():
                for url in md.urls():
                    if url.isLocalFile():
                        paths.append(url.toLocalFile())
            elif md.hasFormat("text/uri-list") and md.hasText():
                paths.extend(
                    ln.strip() for ln in md.text().splitlines() if ln.strip()
                )
        except Exception:
            paths = []
        if not paths:
            return False
        self.append_paths(paths, remember=True)
        return True

    def set_paths(self, paths: list[str], *, remember: bool = True) -> None:
        norms = [self._normalize_one(p) for p in paths]
        norms = [p for p in norms if p]
        self.edit.setPlainText("\n".join(norms))
        if remember:
            work = self._work()
            for p in norms:
                try:
                    abs_p = Path(p).expanduser()
                    if work is not None and not abs_p.is_absolute():
                        abs_p = work / abs_p
                    if abs_p.exists():
                        push_recent_path(str(abs_p.resolve()))
                    else:
                        push_recent_path(p)
                except Exception:
                    push_recent_path(p)

    def append_paths(self, paths: list[str], *, remember: bool = True) -> None:
        existing = self.paths()
        work = self._work()
        # 仅有「默认占位且文件不存在」时，浏览应替换而非追加，避免留下无目录的裸文件名
        if len(existing) == 1:
            only = existing[0]
            try:
                cand = Path(only).expanduser()
                if not cand.is_absolute() and work is not None:
                    cand = work / cand
                if not cand.is_file():
                    existing = []
            except Exception:
                existing = []
        seen = {x.replace("\\", "/") for x in existing}
        merged = list(existing)
        for p in paths:
            n = self._normalize_one(p)
            if not n:
                continue
            key = n.replace("\\", "/")
            if key in seen:
                continue
            seen.add(key)
            merged.append(n)
        self.set_paths(merged, remember=remember)
        self.pathsCommitted.emit()

    def _browse_many(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            self._browse_caption,
            self._start_dir(),
            self._name_filter,
            options=file_dialog_options(),
        )
        if paths:
            self.append_paths(paths, remember=True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if self._extract_drop_paths(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        paths = self._extract_drop_paths(event)
        if paths:
            self.append_paths(paths, remember=True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def _extract_drop_paths(self, event) -> list[str]:
        md = event.mimeData()
        if md is None:
            return []
        out: list[str] = []
        if md.hasUrls():
            for url in md.urls():
                if url.isLocalFile():
                    out.append(url.toLocalFile())
        elif md.hasText():
            for line in md.text().splitlines():
                t = line.strip().strip('"')
                if t:
                    out.append(t)
        return out
