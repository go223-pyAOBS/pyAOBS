# -*- coding: utf-8 -*-
"""Workbench 环境变量、audit、GUI state、路径备份（自原 Tk idata 抽出）。"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Optional

from .raw2sac_paths import RAW2SAC_DIR, ensure_raw2sac_on_path


class IdataEnvContext:
    """共享路径 / audit / 状态持久化。"""

    def __init__(self, raw2sac_dir: Optional[Path] = None) -> None:
        self.raw2sac_dir = Path(raw2sac_dir) if raw2sac_dir is not None else RAW2SAC_DIR
        ensure_raw2sac_on_path()
        self.python_exe = os.environ.get("PYTHON", "") or __import__("sys").executable or "python"
        self.audit_log_path = os.environ.get("PYAOBS_AUDIT_LOG", "").strip()
        self.run_id = os.environ.get("PYAOBS_RUN_ID", "").strip()
        self.project_root = os.environ.get("PYAOBS_PROJECT_ROOT", "").strip()
        self.run_dir = (
            Path(os.environ["PYAOBS_RUN_DIR"].strip())
            if os.environ.get("PYAOBS_RUN_DIR", "").strip()
            else None
        )
        self.inputs_dir = (
            Path(os.environ["PYAOBS_RUN_INPUTS_DIR"].strip())
            if os.environ.get("PYAOBS_RUN_INPUTS_DIR", "").strip()
            else None
        )
        self.outputs_dir = (
            Path(os.environ["PYAOBS_RUN_OUTPUTS_DIR"].strip())
            if os.environ.get("PYAOBS_RUN_OUTPUTS_DIR", "").strip()
            else None
        )
        self.gui_state_file = (
            Path(os.environ["PYAOBS_GUI_STATE_FILE"].strip())
            if os.environ.get("PYAOBS_GUI_STATE_FILE", "").strip()
            else None
        )
        self._field_last_values: dict[str, str] = {}
        # Workbench 环境备份；工区绑定后覆盖 inputs/outputs
        self._env_inputs_dir = self.inputs_dir
        self._env_outputs_dir = self.outputs_dir
        self._env_project_root = self.project_root
        self._bound_workdir: Optional[Path] = None

    def bind_workdir(self, workdir: Optional[str | Path]) -> None:
        """绑定 idata 工区目录，使打开/保存/转换 cwd 落在 inputs/outputs。"""
        if workdir is None or not str(workdir).strip():
            self._bound_workdir = None
            self.inputs_dir = self._env_inputs_dir
            self.outputs_dir = self._env_outputs_dir
            self.project_root = self._env_project_root
            return
        root = Path(workdir).expanduser().resolve()
        self._bound_workdir = root
        self.project_root = str(root)
        self.inputs_dir = root / "inputs"
        self.outputs_dir = root / "outputs"
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def audit(self, event: str, **payload: object) -> None:
        if not self.audit_log_path:
            return
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            "run_id": self.run_id,
            "payload": payload,
        }
        try:
            ap = Path(self.audit_log_path)
            ap.parent.mkdir(parents=True, exist_ok=True)
            with ap.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def default_open_initial_dir(self) -> str:
        if self._bound_workdir is not None:
            return str(self.inputs_dir or self._bound_workdir)
        if self.project_root:
            return self.project_root
        if self.run_dir is not None:
            return str(self.run_dir)
        return str(self.raw2sac_dir)

    def default_save_initial_dir(self) -> str:
        if self.outputs_dir is not None:
            self.outputs_dir.mkdir(parents=True, exist_ok=True)
            return str(self.outputs_dir)
        return str(self.raw2sac_dir)

    def default_run_cwd(self) -> str:
        if self.outputs_dir is not None:
            self.outputs_dir.mkdir(parents=True, exist_ok=True)
            return str(self.outputs_dir)
        return str(self.raw2sac_dir)

    def backup_input_file(self, selected: str) -> str:
        src = None
        for cand in self.candidate_existing_paths(selected):
            if cand.exists() and cand.is_file():
                src = cand
                break
        if src is None:
            src = Path(selected).expanduser()
        if self.inputs_dir is None or not src.exists() or not src.is_file():
            return selected
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        dest = self.inputs_dir / src.name
        try:
            if src.resolve() == dest.resolve():
                return str(dest)
        except Exception:
            pass
        if dest.exists():
            stem = src.stem
            suffix = src.suffix
            idx = 1
            while True:
                candidate = self.inputs_dir / f"{stem}_{idx}{suffix}"
                if not candidate.exists():
                    dest = candidate
                    break
                idx += 1
        try:
            shutil.copy2(src, dest)
            return str(dest)
        except Exception as exc:
            self.audit("input_backup_failed", source=str(src), error=str(exc))
            return selected

    @staticmethod
    def candidate_path_keys(path_text: str) -> list[str]:
        raw = str(path_text or "").strip()
        if not raw:
            return []
        keys = [raw]
        if raw.startswith("/mnt/") and len(raw) > 6 and raw[5].isalpha() and raw[6] == "/":
            drive = raw[5].upper()
            rest = raw[7:]
            keys.append(str(Path(f"{drive}:/{rest}")))
        return list(dict.fromkeys(keys))

    @classmethod
    def candidate_existing_paths(cls, path_text: str) -> list[Path]:
        candidates: list[Path] = []
        for key in cls.candidate_path_keys(path_text):
            try:
                candidates.append(Path(key).expanduser())
            except Exception:
                continue
        return candidates

    def normalize_restored_path(self, path_text: str) -> str:
        raw = str(path_text or "").strip()
        if not raw:
            return ""
        for cand in self.candidate_existing_paths(raw):
            if cand.exists():
                return str(cand)
        return raw

    def rewrite_save_target(self, selected: str) -> str:
        if self.outputs_dir is None:
            return selected
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        chosen = Path(selected).expanduser()
        target = self.outputs_dir / chosen.name
        try:
            chosen_abs = chosen.resolve()
            outputs_abs = self.outputs_dir.resolve()
            try:
                rel = chosen_abs.relative_to(outputs_abs)
                target = outputs_abs / rel
            except Exception:
                target = outputs_abs / chosen_abs.name
        except Exception:
            target = self.outputs_dir / chosen.name
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return str(target)

    def load_gui_state(self) -> dict[str, Any]:
        if self.gui_state_file is None or not self.gui_state_file.exists():
            return {}
        try:
            raw = json.loads(self.gui_state_file.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    def save_gui_state(self, idata_state: dict[str, Any]) -> None:
        if self.gui_state_file is None:
            return
        try:
            existing = self.load_gui_state()
            existing["idata"] = idata_state
            self.gui_state_file.parent.mkdir(parents=True, exist_ok=True)
            self.gui_state_file.write_text(
                json.dumps(existing, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    def note_field_change(
        self,
        field_name: str,
        value: str,
        *,
        on_changed: Optional[Callable[[], None]] = None,
    ) -> None:
        last = self._field_last_values.get(field_name)
        if last == value:
            return
        self._field_last_values[field_name] = value
        self.audit("field_changed", field=field_name, value=value)
        if on_changed is not None:
            on_changed()
