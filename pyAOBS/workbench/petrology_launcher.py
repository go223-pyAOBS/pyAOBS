"""Launch LIP Petrology GUI with observation from Workbench / imodel session."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_gui_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def petrology_obs_path_from_state(state_file: Path | str | None) -> Path | None:
    """Resolve ``petrology_obs_json`` from Workbench session (``imodel_gui`` section)."""
    if not state_file:
        return None
    state_file = Path(state_file)
    sec = load_gui_state(state_file).get("imodel_gui", {})
    if not isinstance(sec, dict):
        return None
    json_path = str(sec.get("petrology_obs_json") or "").strip()
    if json_path and Path(json_path).is_file():
        return Path(json_path)
    return None


def petrology_transect_path_from_state(state_file: Path | str | None) -> Path | None:
    """Resolve ``petrology_transect_windows`` CSV from Workbench session (``imodel_gui`` section)."""
    if not state_file:
        return None
    sec = load_gui_state(Path(state_file)).get("imodel_gui", {})
    if not isinstance(sec, dict):
        return None
    csv_path = str(sec.get("petrology_transect_windows") or "").strip()
    if csv_path and Path(csv_path).is_file():
        return Path(csv_path)
    return None


def petrology_observation_dict_from_state(state_file: Path | str | None) -> dict[str, Any] | None:
    if not state_file:
        return None
    sec = load_gui_state(Path(state_file)).get("imodel_gui", {})
    if not isinstance(sec, dict):
        return None
    raw = sec.get("petrology_observation")
    return raw if isinstance(raw, dict) else None


def write_observation_json_from_state(state_file: Path | str, out_path: Path | str) -> Path | None:
    """Materialize inline ``petrology_observation`` dict to JSON if path missing."""
    p = petrology_obs_path_from_state(state_file)
    if p is not None:
        return p
    obs = petrology_observation_dict_from_state(state_file)
    if obs is None:
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_path


def launch_lip_petrology(
    *,
    import_obs: Path | str | None = None,
    import_transect: Path | str | None = None,
    cwd: Path | str | None = None,
) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "petrology.gui"]
    if import_obs:
        cmd.extend(["--import-obs", str(import_obs)])
    if import_transect:
        cmd.extend(["--import-transect", str(import_transect)])
    return subprocess.Popen(cmd, cwd=str(cwd) if cwd else None)


def launch_lip_from_workbench_state(
    state_file: Path | str,
    *,
    fallback_json: Path | str | None = None,
    import_transect: Path | str | None = None,
    cwd: Path | str | None = None,
) -> subprocess.Popen | None:
    obs_path = petrology_obs_path_from_state(state_file)
    if obs_path is None and fallback_json:
        fb = Path(fallback_json)
        if fb.is_file():
            obs_path = fb
    if obs_path is None:
        obs_path = write_observation_json_from_state(
            state_file,
            Path(state_file).with_suffix(".petrology_obs.json"),
        )
    tran_path = import_transect
    if tran_path is None:
        tran_path = petrology_transect_path_from_state(state_file)
    if obs_path is None and tran_path is None:
        return launch_lip_petrology(cwd=cwd)
    return launch_lip_petrology(
        import_obs=obs_path,
        import_transect=tran_path,
        cwd=cwd,
    )
