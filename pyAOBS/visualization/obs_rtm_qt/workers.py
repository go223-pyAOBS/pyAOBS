# -*- coding: utf-8 -*-
"""后台任务：结果/日志经主线程 Bridge 投递，避免跨线程改 QWidget。"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot


class FuncWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    log = Signal(str)

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    @Slot()
    def run(self) -> None:
        try:
            def _log(msg: str) -> None:
                self.log.emit(str(msg))

            kwargs = dict(self._kwargs)
            try:
                sig = inspect.signature(self._fn)
                if "log" in sig.parameters:
                    kwargs["log"] = _log
            except (TypeError, ValueError):
                pass
            result = self._fn(*self._args, **kwargs)
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit("%s: %s" % (type(exc).__name__, exc))


class _GuiBridge(QObject):
    """住在 GUI 线程；用 @Slot 接收 QueuedConnection，再调用户回调。"""

    def __init__(
        self,
        parent: QObject,
        *,
        on_finished: Optional[Callable[[Any], None]],
        on_failed: Optional[Callable[[str], None]],
        on_log: Optional[Callable[[str], None]],
        thread: QThread,
    ) -> None:
        super().__init__(parent)
        self._on_finished = on_finished
        self._on_failed = on_failed
        self._on_log = on_log
        self._thread = thread

    @Slot(object)
    def on_finished(self, result: object) -> None:
        try:
            if self._on_finished is not None:
                self._on_finished(result)
        finally:
            self._quit_thread()

    @Slot(str)
    def on_failed(self, msg: str) -> None:
        try:
            if self._on_failed is not None:
                self._on_failed(msg)
        finally:
            self._quit_thread()

    @Slot(str)
    def on_log(self, msg: str) -> None:
        if self._on_log is not None:
            self._on_log(msg)

    def _quit_thread(self) -> None:
        th = self._thread
        if th is not None and th.isRunning():
            th.quit()


def start_worker(
    parent: QObject,
    fn: Callable[..., Any],
    *args: Any,
    on_finished: Optional[Callable[[Any], None]] = None,
    on_failed: Optional[Callable[[str], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
    **kwargs: Any,
) -> QThread:
    thread = QThread(parent)
    worker = FuncWorker(fn, *args, **kwargs)
    worker.moveToThread(thread)

    # Bridge 父对象在 GUI 线程，槽一定在主线程执行
    bridge = _GuiBridge(
        parent,
        on_finished=on_finished,
        on_failed=on_failed,
        on_log=on_log,
        thread=thread,
    )

    queued = Qt.ConnectionType.QueuedConnection
    thread.started.connect(worker.run)
    worker.log.connect(bridge.on_log, queued)
    worker.finished.connect(bridge.on_finished, queued)
    worker.failed.connect(bridge.on_failed, queued)
    thread.finished.connect(thread.deleteLater)
    thread.finished.connect(bridge.deleteLater)

    thread.start()

    if not hasattr(parent, "_obs_rtm_threads"):
        setattr(parent, "_obs_rtm_threads", [])
    getattr(parent, "_obs_rtm_threads").append(thread)
    if not hasattr(parent, "_obs_rtm_workers"):
        setattr(parent, "_obs_rtm_workers", [])
    getattr(parent, "_obs_rtm_workers").append(worker)
    if not hasattr(parent, "_obs_rtm_bridges"):
        setattr(parent, "_obs_rtm_bridges", [])
    getattr(parent, "_obs_rtm_bridges").append(bridge)
    return thread
