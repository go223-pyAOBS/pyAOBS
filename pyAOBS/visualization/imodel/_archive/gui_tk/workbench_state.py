"""会话状态 / workbench 路径解析与恢复（轻量依赖，不拉 Matplotlib/pyAOBS）。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

try:
    from ..imodel_qt.interface_files_qt import parse_interface_file
except ImportError:  # pragma: no cover
    parse_interface_file = None  # type: ignore[assignment, misc]


class WorkbenchStateMixin:
    def _on_close(self):
        self._save_workbench_state()
        self.root.destroy()

    def _restore_workbench_state(self) -> None:
        state = self._load_workbench_state()
        if not state:
            return
        restored: list[str] = []
        self._restoring_state = True
        try:
            model_file = self._resolve_restorable_input_path(str(state.get("model_file", "")).strip())
            if model_file:
                if not Path(model_file).exists():
                    self.log_result(f"恢复状态失败：模型文件不存在 {model_file}")
                    return
                loaded = self._load_model_from_file(model_file, persist_state=False)
                if not loaded:
                    return
                restored.append(f"模型: {Path(model_file).name}")
            elif self.grid_data is None:
                return

            show_interfaces = state.get("show_interfaces")
            if isinstance(show_interfaces, bool):
                self.show_interfaces_var.set(show_interfaces)
                restored.append(f"界面显示: {'开' if show_interfaces else '关'}")
            basement = str(state.get("basement_selection", "")).strip()
            if basement and hasattr(self, "basement_interface_combo"):
                options = [str(x) for x in self.basement_interface_combo.cget("values")]
                if basement in options:
                    self.basement_interface_var.set(basement)
                    self._on_basement_interface_selected()
                    restored.append(f"基底面: {basement}")
            seafloor = str(state.get("seafloor_selection", "")).strip()
            if seafloor and hasattr(self, "seafloor_interface_combo"):
                options = [str(x) for x in self.seafloor_interface_combo.cget("values")]
                if seafloor in options:
                    self.seafloor_interface_var.set(seafloor)
                    self._on_seafloor_interface_selected()
                    restored.append(f"海底面: {seafloor}")

            if parse_interface_file is not None:
                raw_ifiles = state.get("interface_files")
                if isinstance(raw_ifiles, list) and raw_ifiles:
                    n_path = 0
                    self.loaded_interfaces = []
                    self.basement_interface_data = None
                    self.basement_interface_file = None
                    if hasattr(self, "interface_file_label"):
                        self.interface_file_label.config(text="None", foreground="gray")
                    for rawp in raw_ifiles:
                        rp = self._resolve_restorable_input_path(str(rawp).strip())
                        if not rp or not Path(rp).exists():
                            continue
                        n_path += 1
                        try:
                            interfaces_in_file = parse_interface_file(rp)
                        except Exception as exc:
                            self.log_result(f"Restore interface file skipped ({rp}): {exc}")
                            continue
                        for iface in interfaces_in_file:
                            iface["file"] = rp
                            self.loaded_interfaces.append(iface)
                    if self.loaded_interfaces:
                        first = self.loaded_interfaces[0]
                        self.basement_interface_data = {"x": first["x"], "z": first["z"]}
                        self.basement_interface_file = str(self.loaded_interfaces[0].get("file") or "")
                        num_interfaces = len(self.loaded_interfaces)
                        if hasattr(self, "interface_file_label"):
                            if num_interfaces == 1 and self.basement_interface_file:
                                self.interface_file_label.config(
                                    text=Path(self.basement_interface_file).name,
                                    foreground="black",
                                )
                            else:
                                self.interface_file_label.config(
                                    text=f"{num_interfaces} interfaces",
                                    foreground="black",
                                )
                        restored.append(f"界面文件: {n_path} 个路径, 曲线 {num_interfaces} 条")
                        if getattr(self, "grid_data", None) is not None and hasattr(self, "plot_model"):
                            self.plot_model()
        finally:
            self._restoring_state = False
        self._save_workbench_state()
        if restored:
            state_path = str(self._gui_state_file) if self._gui_state_file is not None else "(未配置)"
            self.log_result("已恢复上次会话状态: " + " | ".join(restored) + f" | 状态文件: {state_path}")

    def _load_workbench_state(self) -> dict:
        if self._gui_state_file is None or not self._gui_state_file.exists():
            return {}
        try:
            raw = json.loads(self._gui_state_file.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return {}
            obj = raw.get("imodel_gui", {})
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _save_workbench_state(self) -> None:
        if self._gui_state_file is None:
            return
        try:
            raw: dict = {}
            if self._gui_state_file.exists():
                loaded = json.loads(self._gui_state_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    raw = loaded
            prev_state = raw.get("imodel_gui", {}) if isinstance(raw.get("imodel_gui"), dict) else {}
            prev_model_file = str(prev_state.get("model_file", "")).strip()
            model_file = str(self.current_model_file or "").strip() or prev_model_file
            try:
                ifiles = self._workbench_interface_files()
            except Exception:
                ifiles = []
            raw["imodel_gui"] = {
                "model_file": model_file,
                "show_interfaces": bool(self.show_interfaces_var.get()),
                "basement_selection": str(self.basement_interface_var.get()).strip(),
                "seafloor_selection": str(self.seafloor_interface_var.get()).strip(),
                "interface_files": ifiles,
            }
            self._gui_state_file.parent.mkdir(parents=True, exist_ok=True)
            self._gui_state_file.write_text(
                json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    def _resolve_restorable_input_path(self, path_text: str) -> str:
        raw = str(path_text or "").strip()
        if not raw:
            return ""
        for candidate in self._candidate_existing_paths(raw):
            if candidate.exists():
                return str(candidate)
        if self._gui_state_file is None or not self._gui_state_file.exists():
            return raw
        try:
            obj = json.loads(self._gui_state_file.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                return raw
            backup_map = obj.get("input_backups", {})
            if not isinstance(backup_map, dict):
                return raw
            src_text = ""
            for key in self._candidate_path_keys(raw):
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
            src = None
            for cand in self._candidate_existing_paths(src_text):
                if cand.exists():
                    src = cand
                    break
            if src is None or self._run_inputs_dir is None:
                return str(src) if src is not None else raw
            self._run_inputs_dir.mkdir(parents=True, exist_ok=True)
            dest = self._run_inputs_dir / src.name
            if not dest.exists():
                shutil.copy2(src, dest)
            backup_map[str(dest)] = str(src)
            obj["input_backups"] = backup_map
            self._gui_state_file.write_text(
                json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return str(dest)
        except Exception:
            return raw

    @staticmethod
    def _candidate_path_keys(path_text: str) -> list[str]:
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
    def _candidate_existing_paths(cls, path_text: str) -> list[Path]:
        candidates: list[Path] = []
        for key in cls._candidate_path_keys(path_text):
            try:
                candidates.append(Path(key).expanduser())
            except Exception:
                continue
        return candidates
