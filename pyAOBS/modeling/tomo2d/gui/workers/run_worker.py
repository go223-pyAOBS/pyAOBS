"""在 QThread 中执行 TomoAnd 命令。"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, QThread, Slot, Signal


class CommandRunWorker(QObject):
    finished = Signal(bool, str, object)  # ok, log_text, error_or_None
    output_line = Signal(str, str)  # stream ("stdout"|"stderr"), line

    def __init__(self, fn: Callable[[], Any]) -> None:
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            result = self._fn()
            log = "" if result is None else str(result)
            self.finished.emit(True, log, None)
        except Exception as exc:
            self.finished.emit(False, "", exc)


class _MainThreadRelay(QObject):
    """挂在主线程 parent 上，把 worker 信号排队到 GUI 线程再回调。"""

    def __init__(
        self,
        parent: QObject,
        *,
        on_finished: Callable[[bool, str, object], None],
        on_output_line: Callable[[str, str], None] | None,
        thread: QThread,
    ) -> None:
        super().__init__(parent)
        self._on_finished = on_finished
        self._on_output_line = on_output_line
        self._thread = thread

    @Slot(bool, str, object)
    def on_finished(self, ok: bool, log: str, err: object) -> None:
        try:
            self._on_finished(bool(ok), str(log or ""), err)
        finally:
            self._thread.quit()

    @Slot(str, str)
    def on_output_line(self, stream: str, line: str) -> None:
        if self._on_output_line is not None:
            self._on_output_line(stream, line)


def start_command_worker(
    parent: QObject,
    fn: Callable[[], Any],
    on_finished: Callable[[bool, str, object], None],
    *,
    on_output_line: Callable[[str, str], None] | None = None,
    autostart: bool = True,
) -> tuple[QThread, CommandRunWorker]:
    """
    启动后台命令线程。

    ``autostart=False`` 时由调用方在挂好 ``worker`` 引用后再 ``thread.start()``，
    以便 ``job`` 内能把 ``stream_output_line`` 接到 ``worker.output_line``。

    finished / output_line 一律经主线程 ``QObject`` 槽转发，避免在 worker 线程
    里改 UI、``killTimer`` 或 ``thread.wait()`` 自身。
    """
    thread = QThread(parent)
    worker = CommandRunWorker(fn)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    relay = _MainThreadRelay(
        parent,
        on_finished=on_finished,
        on_output_line=on_output_line,
        thread=thread,
    )
    # worker(线程B) → relay(主线程)：AutoConnection 自动变为 QueuedConnection
    worker.finished.connect(relay.on_finished)
    if on_output_line is not None:
        worker.output_line.connect(relay.on_output_line)

    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(relay.deleteLater)
    thread.finished.connect(thread.deleteLater)

    if autostart:
        thread.start()
    return thread, worker
