from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable


def write_audit(event: str, **payload: object) -> None:
    audit_path = os.environ.get("PYAOBS_AUDIT_LOG", "").strip()
    if not audit_path:
        return
    rec = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        "run_id": os.environ.get("PYAOBS_RUN_ID", "").strip(),
        "payload": payload,
    }
    try:
        p = Path(audit_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


class AuditRuntimeHooks:
    """Install lightweight runtime hooks to audit common GUI actions."""

    def __init__(self) -> None:
        self._installed = False
        self._restore_actions: list[Callable[[], None]] = []
        self._project_root = self._env_path("PYAOBS_PROJECT_ROOT")
        self._run_dir = self._env_path("PYAOBS_RUN_DIR")
        self._inputs_dir = self._env_path("PYAOBS_RUN_INPUTS_DIR")
        self._outputs_dir = self._env_path("PYAOBS_RUN_OUTPUTS_DIR")
        self._gui_state_file = self._env_path("PYAOBS_GUI_STATE_FILE")
        self._copied_inputs: dict[str, str] = {}
        self._gui_state_cache: dict[str, Any] = {}
        self._runtime_state: dict[str, Any] = {}
        self._window_count = 0
        self._notebook_count = 0
        self._state_last_write_ts = 0.0
        self._state_last_merge_ts = 0.0
        self._state_dirty = False
        self._load_gui_state()
        self._runtime_state = self._runtime_state_obj()

    def install(self) -> None:
        if self._installed:
            return
        self._installed = True
        self._hook_tk_state_persistence()
        self._hook_tk_filedialog()
        self._hook_tk_messagebox()
        self._hook_subprocess()
        write_audit("runtime_hooks_installed")

    def restore(self) -> None:
        for action in reversed(self._restore_actions):
            try:
                action()
            except Exception:
                pass
        self._restore_actions.clear()
        self._save_gui_state(force=True)
        if self._installed:
            write_audit("runtime_hooks_restored")
        self._installed = False

    def _hook_tk_state_persistence(self) -> None:
        try:
            import tkinter as tk  # type: ignore
            from tkinter import ttk  # type: ignore
        except Exception:
            return

        orig_var_init = tk.Variable.__init__
        orig_var_set = tk.Variable.set
        orig_tk_init = tk.Tk.__init__
        orig_toplevel_init = tk.Toplevel.__init__
        orig_nb_init = ttk.Notebook.__init__

        def var_init_wrapped(var_self: Any, *args: Any, **kwargs: Any) -> None:
            orig_var_init(var_self, *args, **kwargs)
            name = str(getattr(var_self, "_name", "") or "").strip()
            if not name:
                return
            variables = self._runtime_variables()
            if name in variables:
                try:
                    orig_var_set(var_self, variables[name])
                except Exception:
                    pass
            else:
                try:
                    variables[name] = str(var_self.get())
                    self._save_gui_state()
                except Exception:
                    pass

        def var_set_wrapped(var_self: Any, value: Any) -> Any:
            result = orig_var_set(var_self, value)
            name = str(getattr(var_self, "_name", "") or "").strip()
            if name:
                try:
                    variables = self._runtime_variables()
                    new_val = str(var_self.get())
                    if variables.get(name) != new_val:
                        variables[name] = new_val
                        self._save_gui_state()
                except Exception:
                    pass
            return result

        def _bind_window_persist(win: Any, key: str) -> None:
            saved = str(self._runtime_windows().get(key, "") or "").strip()
            if saved:
                try:
                    win.geometry(saved)
                except Exception:
                    pass

            def on_configure(_event: Any = None) -> None:
                try:
                    geom = str(win.geometry())
                    windows = self._runtime_windows()
                    if windows.get(key) != geom:
                        windows[key] = geom
                        self._save_gui_state()
                except Exception:
                    pass

            try:
                win.bind("<Configure>", on_configure, add="+")
            except Exception:
                pass

        def tk_init_wrapped(root_self: Any, *args: Any, **kwargs: Any) -> None:
            orig_tk_init(root_self, *args, **kwargs)
            key = f"tk_{self._window_count}"
            self._window_count += 1
            _bind_window_persist(root_self, key)

        def toplevel_init_wrapped(top_self: Any, *args: Any, **kwargs: Any) -> None:
            orig_toplevel_init(top_self, *args, **kwargs)
            key = f"toplevel_{self._window_count}"
            self._window_count += 1
            _bind_window_persist(top_self, key)

        def nb_init_wrapped(nb_self: Any, *args: Any, **kwargs: Any) -> None:
            orig_nb_init(nb_self, *args, **kwargs)
            idx = self._notebook_count
            self._notebook_count += 1
            key = str(idx)
            saved_idx = self._runtime_notebooks().get(key, None)
            if saved_idx is not None:
                try:
                    nb_self.after_idle(lambda i=int(saved_idx): nb_self.select(i))
                except Exception:
                    pass

            def on_tab_changed(_event: Any = None) -> None:
                try:
                    cur = int(nb_self.index(nb_self.select()))
                    self._runtime_notebooks()[key] = cur
                    self._save_gui_state()
                except Exception:
                    pass

            try:
                nb_self.bind("<<NotebookTabChanged>>", on_tab_changed, add="+")
            except Exception:
                pass

        tk.Variable.__init__ = var_init_wrapped  # type: ignore[assignment]
        tk.Variable.set = var_set_wrapped  # type: ignore[assignment]
        tk.Tk.__init__ = tk_init_wrapped  # type: ignore[assignment]
        tk.Toplevel.__init__ = toplevel_init_wrapped  # type: ignore[assignment]
        ttk.Notebook.__init__ = nb_init_wrapped  # type: ignore[assignment]
        self._restore_actions.append(lambda: setattr(tk.Variable, "__init__", orig_var_init))
        self._restore_actions.append(lambda: setattr(tk.Variable, "set", orig_var_set))
        self._restore_actions.append(lambda: setattr(tk.Tk, "__init__", orig_tk_init))
        self._restore_actions.append(lambda: setattr(tk.Toplevel, "__init__", orig_toplevel_init))
        self._restore_actions.append(lambda: setattr(ttk.Notebook, "__init__", orig_nb_init))

    def _hook_tk_filedialog(self) -> None:
        try:
            from tkinter import filedialog  # type: ignore
        except Exception:
            return
        for name in ("askopenfilename", "askopenfilenames", "asksaveasfilename", "askdirectory"):
            orig = getattr(filedialog, name, None)
            if not callable(orig):
                continue

            def make_wrapper(func_name: str, func: Callable[..., Any]) -> Callable[..., Any]:
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    kwargs = dict(kwargs)
                    kwargs = self._inject_filedialog_defaults(func_name, kwargs)
                    result = func(*args, **kwargs)
                    result = self._rewrite_filedialog_result(func_name, result)
                    write_audit(
                        "tk_filedialog",
                        api=func_name,
                        title=str(kwargs.get("title", "")),
                        result=str(result),
                    )
                    return result

                return wrapped

            setattr(filedialog, name, make_wrapper(name, orig))
            self._restore_actions.append(lambda n=name, f=orig: setattr(filedialog, n, f))

    def _hook_tk_messagebox(self) -> None:
        try:
            from tkinter import messagebox  # type: ignore
        except Exception:
            return
        for name in (
            "showinfo",
            "showwarning",
            "showerror",
            "askyesno",
            "askokcancel",
            "askyesnocancel",
            "askretrycancel",
        ):
            orig = getattr(messagebox, name, None)
            if not callable(orig):
                continue

            def make_wrapper(func_name: str, func: Callable[..., Any]) -> Callable[..., Any]:
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    result = func(*args, **kwargs)
                    title = ""
                    if args:
                        title = str(args[0])
                    elif "title" in kwargs:
                        title = str(kwargs.get("title", ""))
                    write_audit(
                        "tk_messagebox",
                        api=func_name,
                        title=title,
                        result=str(result),
                    )
                    return result

                return wrapped

            setattr(messagebox, name, make_wrapper(name, orig))
            self._restore_actions.append(lambda n=name, f=orig: setattr(messagebox, n, f))

    def _hook_subprocess(self) -> None:
        orig_run = subprocess.run
        orig_popen = subprocess.Popen

        def run_wrapped(*args: Any, **kwargs: Any) -> Any:
            cmd = kwargs.get("args", args[0] if args else "")
            kwargs = dict(kwargs)
            if "cwd" not in kwargs and self._outputs_dir:
                kwargs["cwd"] = str(self._outputs_dir)
            write_audit("subprocess_run_start", command=str(cmd))
            result = orig_run(*args, **kwargs)
            write_audit("subprocess_run_finish", command=str(cmd), returncode=getattr(result, "returncode", None))
            return result

        def popen_wrapped(*args: Any, **kwargs: Any) -> Any:
            cmd = kwargs.get("args", args[0] if args else "")
            kwargs = dict(kwargs)
            if "cwd" not in kwargs and self._outputs_dir:
                kwargs["cwd"] = str(self._outputs_dir)
            write_audit("subprocess_popen_start", command=str(cmd))
            proc = orig_popen(*args, **kwargs)
            write_audit("subprocess_popen_pid", command=str(cmd), pid=getattr(proc, "pid", None))
            return proc

        subprocess.run = run_wrapped  # type: ignore[assignment]
        subprocess.Popen = popen_wrapped  # type: ignore[assignment]
        self._restore_actions.append(lambda: setattr(subprocess, "run", orig_run))
        self._restore_actions.append(lambda: setattr(subprocess, "Popen", orig_popen))

    def _inject_filedialog_defaults(self, api_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        if kwargs.get("initialdir"):
            return kwargs
        fd_state = self._filedialog_state()
        if api_name == "asksaveasfilename" and self._outputs_dir is not None:
            last_save_dir = str(fd_state.get("last_save_dir", "") or "").strip()
            if last_save_dir and Path(last_save_dir).exists():
                kwargs["initialdir"] = last_save_dir
                return kwargs
            self._outputs_dir.mkdir(parents=True, exist_ok=True)
            kwargs["initialdir"] = str(self._outputs_dir)
            return kwargs
        if api_name in ("askopenfilename", "askopenfilenames", "askdirectory"):
            last_open_dir = str(fd_state.get("last_open_dir", "") or "").strip()
            if last_open_dir and Path(last_open_dir).exists():
                kwargs["initialdir"] = last_open_dir
                return kwargs
            if self._project_root is not None and self._project_root.exists():
                kwargs["initialdir"] = str(self._project_root)
            elif self._run_dir is not None:
                kwargs["initialdir"] = str(self._run_dir)
        return kwargs

    def _rewrite_filedialog_result(self, api_name: str, result: Any) -> Any:
        if api_name == "askopenfilename":
            rewritten = self._backup_input_file_and_return(result)
            self._update_filedialog_state(api_name, rewritten)
            return rewritten
        if api_name == "askopenfilenames":
            if not result:
                return result
            copied: list[str] = []
            for raw in list(result):
                copied_path = self._backup_input_file_and_return(raw)
                if copied_path:
                    copied.append(copied_path)
            self._update_filedialog_state(api_name, copied)
            return tuple(copied)
        if api_name == "asksaveasfilename":
            rewritten = self._rewrite_save_target_to_outputs(result)
            self._update_filedialog_state(api_name, rewritten)
            return rewritten
        if api_name == "askdirectory":
            self._update_filedialog_state(api_name, result)
        return result

    def _backup_input_file_and_return(self, selected: Any) -> str:
        src_text = str(selected or "").strip()
        if not src_text:
            return ""
        src = None
        for cand in self._candidate_existing_paths(src_text):
            if cand.exists() and cand.is_file():
                src = cand
                break
        if src is None:
            src = Path(src_text).expanduser()
        if not src.exists() or not src.is_file() or self._inputs_dir is None:
            return src_text
        key = str(src.resolve()).lower()
        cached = self._copied_inputs.get(key, "")
        if cached and Path(cached).exists():
            return cached
        self._inputs_dir.mkdir(parents=True, exist_ok=True)
        dest = self._inputs_dir / src.name
        try:
            if src.resolve() == dest.resolve():
                self._copied_inputs[key] = str(dest)
                return str(dest)
        except Exception:
            pass
        backup_map = self._input_backup_map()
        if dest.exists():
            existing_src = str(backup_map.get(str(dest), "")).strip()
            if existing_src and str(src.resolve()).lower() == existing_src.lower():
                try:
                    shutil.copy2(src, dest)
                except Exception:
                    pass
                self._copied_inputs[key] = str(dest)
                return str(dest)
            # 若同名文件已在 GUI inputs 目标位，按“覆盖并复用”策略处理，避免 _1/_2 累积。
            try:
                shutil.copy2(src, dest)
                self._copied_inputs[key] = str(dest)
                backup_map[str(dest)] = str(src)
                self._save_gui_state()
                return str(dest)
            except Exception:
                pass
            stem = src.stem
            suffix = src.suffix
            idx = 1
            while True:
                candidate = self._inputs_dir / f"{stem}_{idx}{suffix}"
                if not candidate.exists():
                    dest = candidate
                    break
                existing_src = str(backup_map.get(str(candidate), "")).strip()
                if existing_src and str(src.resolve()).lower() == existing_src.lower():
                    self._copied_inputs[key] = str(candidate)
                    return str(candidate)
                idx += 1
        try:
            shutil.copy2(src, dest)
            self._copied_inputs[key] = str(dest)
            backup_map[str(dest)] = str(src)
            self._save_gui_state()
            write_audit(
                "input_file_backed_up",
                source=str(src),
                backup=str(dest),
            )
            return str(dest)
        except Exception as exc:
            write_audit(
                "input_file_backup_failed",
                source=str(src),
                error=str(exc),
            )
            return src_text

    def _rewrite_save_target_to_outputs(self, selected: Any) -> str:
        raw = str(selected or "").strip()
        if not raw:
            return ""
        if self._outputs_dir is None:
            return raw
        self._outputs_dir.mkdir(parents=True, exist_ok=True)
        chosen = Path(raw).expanduser()
        target = self._outputs_dir / chosen.name
        try:
            chosen_abs = chosen.resolve()
            outputs_abs = self._outputs_dir.resolve()
            try:
                rel = chosen_abs.relative_to(outputs_abs)
                target = outputs_abs / rel
            except Exception:
                target = outputs_abs / chosen_abs.name
        except Exception:
            target = self._outputs_dir / chosen.name
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        write_audit("save_target_rewritten", selected=raw, rewritten=str(target))
        return str(target)

    @staticmethod
    def _env_path(env_key: str) -> Path | None:
        raw = os.environ.get(env_key, "").strip()
        if not raw:
            return None
        try:
            for key in AuditRuntimeHooks._candidate_path_keys(raw):
                p = Path(key).expanduser()
                if p.exists():
                    return p.resolve()
            return Path(raw).expanduser().resolve()
        except Exception:
            return None

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
        out: list[Path] = []
        for key in cls._candidate_path_keys(path_text):
            try:
                out.append(Path(key).expanduser())
            except Exception:
                continue
        return out

    def _load_gui_state(self) -> None:
        if self._gui_state_file is None:
            self._gui_state_cache = {}
            return
        try:
            if self._gui_state_file.exists():
                self._gui_state_cache = json.loads(self._gui_state_file.read_text(encoding="utf-8"))
            else:
                self._gui_state_cache = {}
        except Exception:
            self._gui_state_cache = {}

    def _save_gui_state_impl(self, *, force: bool) -> None:
        if self._gui_state_file is None:
            return
        if not force and not self._state_dirty:
            return
        now = time.monotonic()
        # GUI 交互期尽量降低磁盘写入频率，避免主线程卡顿
        if not force and (now - self._state_last_write_ts) < 1.5:
            return
        try:
            # 只做低频 merge，减少每次保存都读盘导致的卡顿
            do_merge_from_disk = force or (now - self._state_last_merge_ts) > 8.0
            existing: dict[str, Any] = {}
            if do_merge_from_disk and self._gui_state_file.exists():
                try:
                    loaded = json.loads(self._gui_state_file.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        existing = loaded
                except Exception:
                    existing = {}

            merged: dict[str, Any] = dict(existing) if existing else dict(self._gui_state_cache)
            for k in ("runtime", "filedialog", "input_backups"):
                if k in self._gui_state_cache:
                    merged[k] = self._gui_state_cache.get(k)
            self._gui_state_cache = dict(merged)
            self._gui_state_file.parent.mkdir(parents=True, exist_ok=True)
            self._gui_state_file.write_text(
                json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            self._state_last_write_ts = now
            if do_merge_from_disk:
                self._state_last_merge_ts = now
            self._state_dirty = False
        except Exception:
            pass

    def _filedialog_state(self) -> dict[str, Any]:
        obj = self._gui_state_cache.get("filedialog", {})
        if not isinstance(obj, dict):
            obj = {}
            self._gui_state_cache["filedialog"] = obj
        return obj

    def _update_filedialog_state(self, api_name: str, result: Any) -> None:
        if not result:
            return
        fd_state = self._filedialog_state()
        path_text = ""
        if api_name == "askopenfilenames":
            first = result[0] if isinstance(result, (list, tuple)) and result else ""
            path_text = str(first or "").strip()
        else:
            path_text = str(result).strip()
        if not path_text:
            return
        p = Path(path_text).expanduser()
        base_dir = p if p.is_dir() else p.parent
        if not str(base_dir).strip():
            return
        if api_name in ("askopenfilename", "askopenfilenames", "askdirectory"):
            fd_state["last_open_dir"] = str(base_dir)
        if api_name == "asksaveasfilename":
            fd_state["last_save_dir"] = str(base_dir)
        fd_state["last_used_dir"] = str(base_dir)
        self._save_gui_state()

    def _save_gui_state(self, force: bool = False) -> None:
        self._state_dirty = True
        self._save_gui_state_impl(force=force)

    def _runtime_state_obj(self) -> dict[str, Any]:
        obj = self._gui_state_cache.get("runtime", {})
        if not isinstance(obj, dict):
            obj = {}
            self._gui_state_cache["runtime"] = obj
        return obj

    def _runtime_variables(self) -> dict[str, Any]:
        obj = self._runtime_state.get("variables", {})
        if not isinstance(obj, dict):
            obj = {}
            self._runtime_state["variables"] = obj
        return obj

    def _runtime_notebooks(self) -> dict[str, Any]:
        obj = self._runtime_state.get("notebooks", {})
        if not isinstance(obj, dict):
            obj = {}
            self._runtime_state["notebooks"] = obj
        return obj

    def _runtime_windows(self) -> dict[str, Any]:
        obj = self._runtime_state.get("windows", {})
        if not isinstance(obj, dict):
            obj = {}
            self._runtime_state["windows"] = obj
        return obj

    def _input_backup_map(self) -> dict[str, Any]:
        obj = self._gui_state_cache.get("input_backups", {})
        if not isinstance(obj, dict):
            obj = {}
            self._gui_state_cache["input_backups"] = obj
        return obj

