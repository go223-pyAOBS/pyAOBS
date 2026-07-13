"""
Run manager for pyAOBS Workbench.

Phase-2 scope:
- create run directories under project runs/
- persist run manifest with status transitions
- execute commands with stdout/stderr capture
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Sequence

from .project_manager import ProjectContext, ProjectError


RUN_MANIFEST_FILE = "manifest.json"
GUI_AUDIT_REL_DIR = Path("state") / "gui_sessions"
GUI_STATE_REL_DIR = Path("state") / "gui_states"


@dataclass(frozen=True)
class RunContext:
    """Filesystem context for one run."""

    run_id: str
    run_dir: Path
    inputs_dir: Path
    outputs_dir: Path
    logs_dir: Path
    manifest_file: Path


class RunManager:
    """Create/update run records and execute commands in a reproducible layout."""

    def __init__(self) -> None:
        self._active_nodes_lock = threading.Lock()
        self._active_nodes: set[str] = set()

    def start_run(
        self,
        project: ProjectContext,
        node_id: str,
        *,
        params: dict[str, Any] | None = None,
        inputs: Sequence[str | Path] | None = None,
        env: dict[str, str] | None = None,
    ) -> RunContext:
        run_id = self._new_run_id(node_id)
        run_dir = project.resolve(Path("runs") / run_id)
        inputs_dir = run_dir / "inputs"
        outputs_dir = run_dir / "outputs"
        logs_dir = run_dir / "logs"
        for p in (run_dir, inputs_dir, outputs_dir, logs_dir):
            p.mkdir(parents=True, exist_ok=True)

        manifest_file = run_dir / RUN_MANIFEST_FILE
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "node_id": node_id,
            "status": "running",
            "created_at": _utc_now_iso(),
            "started_at": _utc_now_iso(),
            "finished_at": None,
            "elapsed_s": None,
            "return_code": None,
            "params": params or {},
            "env_overrides": env or {},
            "command": None,
            "cwd": None,
            "logs": {},
            "outputs": [],
            "input_files": self._snapshot_input_metadata(project.root, inputs or []),
            "error": None,
            "platform": {
                "python": sys.version,
                "os_name": os.name,
            },
        }
        self._write_manifest(manifest_file, manifest)
        return RunContext(
            run_id=run_id,
            run_dir=run_dir,
            inputs_dir=inputs_dir,
            outputs_dir=outputs_dir,
            logs_dir=logs_dir,
            manifest_file=manifest_file,
        )

    def complete_run(
        self,
        run: RunContext,
        *,
        status: str,
        return_code: int | None,
        outputs: Sequence[str | Path] | None = None,
        error: str | None = None,
    ) -> None:
        manifest = self._read_manifest(run.manifest_file)
        finished_at = _utc_now_iso()
        manifest["status"] = status
        manifest["return_code"] = return_code
        manifest["finished_at"] = finished_at
        manifest["outputs"] = [str(Path(p)) for p in (outputs or [])]
        manifest["error"] = error
        if manifest.get("started_at"):
            manifest["elapsed_s"] = _elapsed_seconds(
                manifest["started_at"], finished_at
            )
        self._write_manifest(run.manifest_file, manifest)

    def run_command(
        self,
        project: ProjectContext,
        node_id: str,
        command: Sequence[str],
        *,
        params: dict[str, Any] | None = None,
        inputs: Sequence[str | Path] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        timeout_s: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> RunContext:
        if not command:
            raise ProjectError("run_command requires a non-empty command sequence.")
        normalized_node_id = self._new_run_id(node_id)
        self._acquire_node_slot(normalized_node_id)
        try:
            plugin_id = str((params or {}).get("plugin", "")).strip()
            is_gui_node = plugin_id.endswith(".gui")
            run = self.start_run(project, node_id, params=params, inputs=inputs, env=env)
            stdout_path = run.logs_dir / "stdout.log"
            stderr_path = run.logs_dir / "stderr.log"
            audit_rel = GUI_AUDIT_REL_DIR / f"{run.run_id}.jsonl"
            audit_abs = project.resolve(audit_rel)
            audit_abs.parent.mkdir(parents=True, exist_ok=True)
            gui_state_rel = GUI_STATE_REL_DIR / f"{run.run_id}.json"
            gui_state_abs = project.resolve(gui_state_rel)
            gui_state_abs.parent.mkdir(parents=True, exist_ok=True)

            merged_env = dict(os.environ)
            if env:
                merged_env.update(env)
            merged_env["PYAOBS_RUN_ID"] = run.run_id
            merged_env["PYAOBS_PROJECT_ROOT"] = str(project.root)
            merged_env["PYAOBS_RUN_DIR"] = str(run.run_dir)
            merged_env["PYAOBS_AUDIT_LOG"] = str(audit_abs)
            merged_env["PYAOBS_GUI_STATE_FILE"] = str(gui_state_abs)
            gui_inputs_dir = run.inputs_dir
            gui_outputs_dir = run.outputs_dir
            if is_gui_node:
                gui_inputs_dir = run.inputs_dir / self._gui_inputs_subdir(plugin_id)
                gui_inputs_dir.mkdir(parents=True, exist_ok=True)
                gui_outputs_dir = run.outputs_dir / self._gui_outputs_subdir(plugin_id)
                gui_outputs_dir.mkdir(parents=True, exist_ok=True)
            merged_env["PYAOBS_RUN_INPUTS_DIR"] = str(gui_inputs_dir)
            merged_env["PYAOBS_RUN_OUTPUTS_DIR"] = str(gui_outputs_dir)
            if is_gui_node:
                # GUI nodes always run inside this run sandbox.
                run_cwd = gui_outputs_dir
            else:
                run_cwd = Path(cwd).expanduser().resolve() if cwd else project.root
            run.outputs_dir.mkdir(parents=True, exist_ok=True)

            manifest = self._read_manifest(run.manifest_file)
            manifest["started_at"] = _utc_now_iso()
            manifest["command"] = [str(x) for x in command]
            manifest["cwd"] = str(run_cwd)
            if is_gui_node:
                manifest["sandbox"] = {
                    "mode": "gui_run_sandbox",
                    "run_dir": str(run.run_dir),
                    "inputs_dir": str(gui_inputs_dir),
                    "outputs_dir": str(gui_outputs_dir),
                    "requested_cwd_ignored": str(cwd or ""),
                }
            manifest["logs"] = {
                "stdout": str(stdout_path.relative_to(run.run_dir)),
                "stderr": str(stderr_path.relative_to(run.run_dir)),
            }
            manifest["audit"] = {
                "session_log": str(audit_rel),
            }
            manifest["gui_state"] = {
                "state_file": str(gui_state_rel),
            }
            self._write_manifest(run.manifest_file, manifest)

            cancelled = False
            with stdout_path.open("w", encoding="utf-8") as so, stderr_path.open(
                "w", encoding="utf-8"
            ) as se:
                try:
                    proc = subprocess.Popen(
                        list(command),
                        cwd=str(run_cwd),
                        env=merged_env,
                        stdout=so,
                        stderr=se,
                        text=True,
                    )
                    manifest = self._read_manifest(run.manifest_file)
                    manifest["pid"] = proc.pid
                    self._write_manifest(run.manifest_file, manifest)

                    started_monotonic = time.monotonic()
                    while True:
                        rc = proc.poll()
                        if rc is not None:
                            break
                        if timeout_s is not None and (time.monotonic() - started_monotonic) > timeout_s:
                            proc.terminate()
                            try:
                                proc.wait(timeout=3)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait()
                            self.complete_run(
                                run,
                                status="failed",
                                return_code=None,
                                error=f"timeout after {timeout_s}s",
                            )
                            return run
                        if cancel_event is not None and cancel_event.is_set():
                            cancelled = True
                            proc.terminate()
                            try:
                                proc.wait(timeout=3)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait()
                            break
                        time.sleep(0.2)
                    completed_returncode = proc.returncode
                except subprocess.TimeoutExpired as exc:
                    self.complete_run(
                        run,
                        status="failed",
                        return_code=None,
                        error=f"timeout after {timeout_s}s: {exc}",
                    )
                    return run
                except Exception as exc:  # defensive: persist failure context
                    self.complete_run(
                        run,
                        status="failed",
                        return_code=None,
                        error=f"run_command exception: {exc}",
                    )
                    return run

            if cancelled:
                self.complete_run(
                    run,
                    status="cancelled",
                    return_code=completed_returncode,
                    outputs=_discover_outputs(run.outputs_dir),
                    error="cancelled by user",
                )
                return run

            status = "success" if completed_returncode == 0 else "failed"
            self.complete_run(
                run,
                status=status,
                return_code=completed_returncode,
                outputs=_discover_outputs(run.outputs_dir),
                error=None if status == "success" else "non-zero return code",
            )
            return run
        finally:
            self._release_node_slot(normalized_node_id)

    @staticmethod
    def _new_run_id(node_id: str) -> str:
        safe_node = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in node_id)
        return safe_node or "OBS_node"

    @staticmethod
    def _gui_inputs_subdir(plugin_id: str) -> str:
        base = plugin_id.strip().lower().split(".", 1)[0]
        if base == "data":
            return "idata"
        return base or "gui"

    @staticmethod
    def _gui_outputs_subdir(plugin_id: str) -> str:
        base = plugin_id.strip().lower().split(".", 1)[0]
        if base == "data":
            return "idata"
        return base or "gui"

    def _acquire_node_slot(self, run_id: str) -> None:
        with self._active_nodes_lock:
            if run_id in self._active_nodes:
                raise ProjectError(
                    f"节点工作目录 '{run_id}' 正在运行，无法并发覆盖。请等待完成或修改 node_id。"
                )
            self._active_nodes.add(run_id)

    def _release_node_slot(self, run_id: str) -> None:
        with self._active_nodes_lock:
            self._active_nodes.discard(run_id)

    @staticmethod
    def _snapshot_input_metadata(
        project_root: Path, inputs: Sequence[str | Path]
    ) -> list[dict[str, Any]]:
        info: list[dict[str, Any]] = []
        for raw in inputs:
            p = Path(raw)
            abs_p = p if p.is_absolute() else (project_root / p)
            abs_p = abs_p.resolve()
            item: dict[str, Any] = {
                "path": str(abs_p),
                "exists": abs_p.exists(),
            }
            if abs_p.exists() and abs_p.is_file():
                item["sha256"] = _sha256_file(abs_p)
                item["size_bytes"] = abs_p.stat().st_size
            info.append(item)
        return info

    @staticmethod
    def _read_manifest(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProjectError(f"Invalid run manifest at '{path}': {exc}") from exc

    @staticmethod
    def _write_manifest(path: Path, data: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _elapsed_seconds(started_at: str, finished_at: str) -> float | None:
    try:
        st = datetime.strptime(started_at, "%Y-%m-%dT%H:%M:%SZ")
        ft = datetime.strptime(finished_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None
    return max(0.0, (ft - st).total_seconds())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _discover_outputs(outputs_dir: Path) -> list[str]:
    if not outputs_dir.exists():
        return []
    files = [p for p in outputs_dir.rglob("*") if p.is_file()]
    files.sort(key=lambda p: str(p))
    return [str(p.relative_to(outputs_dir.parent)) for p in files]

