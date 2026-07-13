"""Qt imodel 会话状态读写（与 Tk WorkbenchStateMixin 共用同一 JSON 结构）。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


def gui_state_file_from_env() -> Optional[Path]:
    raw = __import__("os").environ.get("PYAOBS_GUI_STATE_FILE", "").strip()
    return Path(raw).expanduser() if raw else None


def run_inputs_dir_from_env() -> Optional[Path]:
    raw = __import__("os").environ.get("PYAOBS_RUN_INPUTS_DIR", "").strip()
    return Path(raw).expanduser() if raw else None


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


def candidate_existing_paths(path_text: str) -> list[Path]:
    candidates: list[Path] = []
    for key in candidate_path_keys(path_text):
        try:
            candidates.append(Path(key).expanduser())
        except Exception:
            continue
    return candidates


def load_imodel_section(state_file: Path) -> Dict[str, Any]:
    if not state_file.exists():
        return {}
    try:
        raw = json.loads(state_file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        obj = raw.get("imodel_gui", {})
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def save_imodel_section(
    state_file: Path,
    *,
    model_file: str,
    show_interfaces: bool,
    basement_selection: str,
    seafloor_selection: str,
    moho_selection: str = "",
    interface_files: Optional[List[str]] = None,
    gravity_obs_data_dir: Optional[str] = None,
    gravity_obs_filename: Optional[str] = None,
    gravity_obs_overlay: Optional[bool] = None,
    gravity_profile_lon_lat_csv: Optional[str] = None,
    petrology_observation: Optional[Dict[str, Any]] = None,
    petrology_obs_json: Optional[str] = None,
    petrology_f_lower: Optional[float] = None,
    petrology_export_x_km: Optional[float] = None,
    petrology_transect_windows: Optional[str] = None,
) -> None:
    raw: Dict[str, Any] = {}
    if state_file.exists():
        try:
            loaded = json.loads(state_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw = loaded
        except Exception:
            raw = {}
    prev = raw.get("imodel_gui", {})
    prev = prev if isinstance(prev, dict) else {}
    prev_model = str(prev.get("model_file", "") or "").strip()
    mf = str(model_file or "").strip() or prev_model
    if interface_files is None:
        pif = prev.get("interface_files")
        if_list: List[str] = (
            [str(x).strip() for x in pif if str(x).strip()] if isinstance(pif, list) else []
        )
    else:
        if_list = [str(x).strip() for x in interface_files if str(x).strip()]

    merged: Dict[str, Any] = dict(prev)
    merged.update(
        {
            "model_file": mf,
            "show_interfaces": bool(show_interfaces),
            "basement_selection": str(basement_selection or "").strip(),
            "seafloor_selection": str(seafloor_selection or "").strip(),
            "moho_selection": str(moho_selection or "").strip(),
            "interface_files": if_list,
        }
    )

    if gravity_obs_data_dir is not None:
        merged["gravity_obs_data_dir"] = str(gravity_obs_data_dir).strip()
    elif "gravity_obs_data_dir" not in merged:
        merged["gravity_obs_data_dir"] = ""

    if gravity_obs_filename is not None:
        merged["gravity_obs_filename"] = str(gravity_obs_filename).strip()
    elif "gravity_obs_filename" not in merged:
        merged["gravity_obs_filename"] = ""

    if gravity_profile_lon_lat_csv is not None:
        merged["gravity_profile_lon_lat_csv"] = str(gravity_profile_lon_lat_csv).strip()
    elif "gravity_profile_lon_lat_csv" not in merged:
        merged["gravity_profile_lon_lat_csv"] = ""

    if gravity_obs_overlay is not None:
        merged["gravity_obs_overlay"] = bool(gravity_obs_overlay)
    elif "gravity_obs_overlay" not in merged:
        merged["gravity_obs_overlay"] = False

    if petrology_observation is not None:
        merged["petrology_observation"] = petrology_observation
    if petrology_obs_json is not None:
        merged["petrology_obs_json"] = str(petrology_obs_json).strip()
    if petrology_f_lower is not None:
        merged["petrology_f_lower"] = float(petrology_f_lower)
    if petrology_export_x_km is not None:
        merged["petrology_export_x_km"] = float(petrology_export_x_km)
    if petrology_transect_windows is not None:
        merged["petrology_transect_windows"] = str(petrology_transect_windows).strip()

    raw["imodel_gui"] = merged
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_restorable_input_path(
    path_text: str,
    *,
    state_file: Optional[Path],
    run_inputs_dir: Optional[Path],
) -> str:
    raw = str(path_text or "").strip()
    if not raw:
        return ""
    for cand in candidate_existing_paths(raw):
        if cand.exists():
            return str(cand)
    if state_file is None or not state_file.exists():
        return raw
    try:
        obj = json.loads(state_file.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return raw
        backup_map = obj.get("input_backups", {})
        if not isinstance(backup_map, dict):
            return raw
        src_text = ""
        for key in candidate_path_keys(raw):
            mapped = str(backup_map.get(key, "")).strip()
            if mapped:
                src_text = mapped
                break
        if not src_text:
            target_name = Path(raw).name.lower()
            for k, v in backup_map.items():
                try:
                    if Path(str(k)).name.lower() == target_name and str(v).strip():
                        src_text = str(v).strip()
                        break
                except Exception:
                    continue
        if not src_text:
            return raw
        src: Optional[Path] = None
        for cand in candidate_existing_paths(src_text):
            if cand.exists():
                src = cand
                break
        if src is None or run_inputs_dir is None:
            return str(src) if src is not None else raw
        run_inputs_dir.mkdir(parents=True, exist_ok=True)
        dest = run_inputs_dir / src.name
        if not dest.exists():
            shutil.copy2(src, dest)
        backup_map[str(dest)] = str(src)
        obj["input_backups"] = backup_map
        state_file.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return str(dest)
    except Exception:
        return raw
