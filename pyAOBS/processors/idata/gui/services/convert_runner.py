# -*- coding: utf-8 -*-
"""子进程运行 raw2sac 转换脚本，经 Qt 信号回传日志。"""

from __future__ import annotations

import subprocess
from typing import List, Optional

from PySide6.QtCore import QObject, QThread, Signal

from .env_context import IdataEnvContext


class _ConvertWorker(QThread):
    finished_ok = Signal(int, str)  # returncode, output

    def __init__(self, command: List[str], cwd: str, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._command = command
        self._cwd = cwd

    def run(self) -> None:
        try:
            proc = subprocess.run(
                self._command,
                cwd=self._cwd,
                capture_output=True,
                text=True,
            )
            output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
            self.finished_ok.emit(int(proc.returncode), output.strip())
        except Exception as exc:
            self.finished_ok.emit(-1, f"Execution failed: {exc}")


class ConvertRunner(QObject):
    """编排转换脚本执行。"""

    log = Signal(str)
    status = Signal(str)
    job_finished = Signal(int, str, list)  # returncode, output, command

    def __init__(self, env: IdataEnvContext, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.env = env
        self._active_jobs = 0
        self._workers: list[_ConvertWorker] = []

    @property
    def active_jobs(self) -> int:
        return self._active_jobs

    def run_script(self, script_name: str, args: List[str]) -> None:
        script_path = self.env.raw2sac_dir / script_name
        if not script_path.exists():
            self.log.emit(f"[转换] 找不到脚本: {script_path}")
            self.status.emit("Script not found.")
            self.job_finished.emit(-1, f"Cannot find script: {script_path}", [])
            return
        command = [self.env.python_exe, str(script_path)] + list(args)
        self._active_jobs += 1
        self.status.emit(f"Running ({self._active_jobs} active): {' '.join(command)}")
        self.log.emit(f"[转换] 启动 {script_name}  （并发任务={self._active_jobs}）")
        self.log.emit(f"$ {' '.join(command)}")
        self.env.audit("run_script_started", script=script_name, command=command)

        worker = _ConvertWorker(command, self.env.default_run_cwd(), self)
        self._workers.append(worker)

        def _done(code: int, output: str, cmd: list = command, w: _ConvertWorker = worker) -> None:
            self._active_jobs = max(0, self._active_jobs - 1)
            if w in self._workers:
                self._workers.remove(w)
            self.env.audit(
                "run_script_finished",
                return_code=code,
                output_preview=(output or "")[:8000],
            )
            if output:
                self.log.emit(output)
            tag = "成功" if code == 0 else "失败"
            self.log.emit(
                f"[转换] {tag}  return_code={code}  "
                f"剩余并发={self._active_jobs}  脚本={cmd[1] if len(cmd) > 1 else '?'}"
            )
            if code == 0:
                self.status.emit(f"Done (active jobs: {self._active_jobs}).")
            else:
                self.status.emit(f"Failed (active jobs: {self._active_jobs}).")
            self.job_finished.emit(code, output, cmd)
            w.deleteLater()

        worker.finished_ok.connect(_done)
        worker.start()
