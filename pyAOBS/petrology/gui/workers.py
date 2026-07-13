"""Background task helpers for LIP GUI."""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot

ProgressReport = Callable[[int, int, str], None]


class BackgroundWorker(QObject):
    """Runs ``task_fn(report)`` or ``task_fn()`` on a ``QThread``."""

    finished = Signal(object)
    error = Signal(str)
    progress = Signal(int, int, str)

    def __init__(
        self,
        task_fn: Callable[..., Any],
        *,
        with_progress: bool = True,
    ) -> None:
        super().__init__()
        self._task_fn = task_fn
        self._with_progress = with_progress

    @Slot()
    def run(self) -> None:
        try:
            if self._with_progress:

                def report(done: int, total: int, message: str = "") -> None:
                    self.progress.emit(int(done), int(total), message or "")

                result = self._task_fn(report)
            else:
                result = self._task_fn()
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class _MainThreadDispatcher(QObject):
    """Relay worker signals onto the GUI thread via @Slot handlers."""

    def __init__(
        self,
        on_done: Callable[[Any], None],
        on_error: Callable[[str], None],
        thread: QThread,
        on_progress: ProgressReport | None = None,
    ) -> None:
        super().__init__()
        self._on_done = on_done
        self._on_error = on_error
        self._thread = thread
        self._on_progress = on_progress

    @Slot(object)
    def _handle_finished(self, result: object) -> None:
        try:
            self._on_done(result)
        finally:
            self._thread.quit()

    @Slot(str)
    def _handle_error(self, message: str) -> None:
        try:
            self._on_error(message)
        finally:
            self._thread.quit()

    @Slot(int, int, str)
    def _handle_progress(self, done: int, total: int, message: str) -> None:
        if self._on_progress is not None:
            self._on_progress(done, total, message)


def run_in_thread(
    parent: QObject,
    task_fn: Callable[..., Any],
    on_done: Callable[[Any], None],
    on_error: Callable[[str], None],
    *,
    on_progress: ProgressReport | None = None,
    with_progress: bool = True,
) -> tuple[QThread, BackgroundWorker]:
    """Start ``task_fn`` on a worker thread; callbacks always run on ``parent``'s thread."""
    thread = QThread(parent)
    worker = BackgroundWorker(task_fn, with_progress=with_progress)
    dispatcher = _MainThreadDispatcher(on_done, on_error, thread, on_progress)
    dispatcher.setParent(parent)

    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    queued = Qt.ConnectionType.QueuedConnection
    worker.finished.connect(dispatcher._handle_finished, queued)
    worker.error.connect(dispatcher._handle_error, queued)
    if on_progress is not None:
        worker.progress.connect(dispatcher._handle_progress, queued)

    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(dispatcher.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread, worker
