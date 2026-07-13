"""Unified UI launcher for raw2sac data conversion tools."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk


class IdataApp:
    """Simple Tk UI that wraps three raw2sac converters."""

    def __init__(self, root: tk.Tk | None = None) -> None:
        self.root = root or tk.Tk()
        self.root.title("idata - Data Conversion Workbench")
        self.root.geometry("980x680")
        self._configure_fonts()

        self.raw2sac_dir = Path(__file__).resolve().parent
        self.python_exe = sys.executable or "python"
        self._active_jobs = 0
        self._audit_log_path = os.environ.get("PYAOBS_AUDIT_LOG", "").strip()
        self._run_id = os.environ.get("PYAOBS_RUN_ID", "").strip()
        self._project_root = os.environ.get("PYAOBS_PROJECT_ROOT", "").strip()
        self._run_dir = Path(os.environ.get("PYAOBS_RUN_DIR", "").strip()) if os.environ.get("PYAOBS_RUN_DIR", "").strip() else None
        self._inputs_dir = Path(os.environ.get("PYAOBS_RUN_INPUTS_DIR", "").strip()) if os.environ.get("PYAOBS_RUN_INPUTS_DIR", "").strip() else None
        self._outputs_dir = Path(os.environ.get("PYAOBS_RUN_OUTPUTS_DIR", "").strip()) if os.environ.get("PYAOBS_RUN_OUTPUTS_DIR", "").strip() else None
        self._gui_state_file = (
            Path(os.environ.get("PYAOBS_GUI_STATE_FILE", "").strip())
            if os.environ.get("PYAOBS_GUI_STATE_FILE", "").strip()
            else None
        )
        self._field_last_values: dict[str, str] = {}

        self.status_var = tk.StringVar(value="Ready.")
        self._build_ui()
        self._restore_state()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._audit(
            "idata_started",
            run_id=self._run_id,
            project_root=self._project_root,
        )

    def run(self) -> None:
        self.root.mainloop()

    def _configure_fonts(self) -> None:
        base_size = 12
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            try:
                f = tkfont.nametofont(name)
                f.configure(size=base_size)
            except Exception:
                pass
        style = ttk.Style(self.root)
        style.configure(".", font=("Segoe UI", base_size))
        style.configure("Treeview.Heading", font=("Segoe UI", base_size, "bold"))

    def _audit(self, event: str, **payload: object) -> None:
        if not self._audit_log_path:
            return
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            "run_id": self._run_id,
            "payload": payload,
        }
        try:
            ap = Path(self._audit_log_path)
            ap.parent.mkdir(parents=True, exist_ok=True)
            with ap.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _on_close(self) -> None:
        self._save_state()
        self._audit("idata_closed", active_jobs=self._active_jobs)
        self.root.destroy()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(
            top,
            text="idata: OBEM/RAW/SAC conversion tools in one place",
        ).pack(side=tk.LEFT)
        ttk.Button(top, text="Clear Log", command=self._clear_log).pack(side=tk.RIGHT)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self.tab_obem = ttk.Frame(self.notebook, padding=8)
        self.tab_sac2y = ttk.Frame(self.notebook, padding=8)
        self.tab_raw2sac = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.tab_obem, text="OBEM TSM -> SAC")
        self.notebook.add(self.tab_sac2y, text="SAC -> SEGY")
        self.notebook.add(self.tab_raw2sac, text="RAW -> SAC")

        self._build_obem_tab()
        self._build_sac2y_tab()
        self._build_raw2sac_tab()

        ttk.Label(self.root, text="Execution Log").pack(anchor=tk.W, padx=8)
        self.log_text = tk.Text(self.root, height=16, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=False, padx=8, pady=(0, 8))
        self.log_text.insert("1.0", "idata started.\n")
        self.log_text.configure(state=tk.DISABLED)

        status_frame = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        status_frame.pack(fill=tk.X)
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor=tk.W)

    def _build_obem_tab(self) -> None:
        ttk.Label(
            self.tab_obem,
            text="Run obem_tsm_to_sac_obspy.py with one config file:",
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))
        ttk.Label(self.tab_obem, text="config_file").grid(row=1, column=0, sticky=tk.W)
        self.obem_config_var = tk.StringVar(value="")
        ttk.Entry(self.tab_obem, textvariable=self.obem_config_var).grid(
            row=1, column=1, sticky=tk.EW, padx=(6, 6)
        )
        ttk.Button(
            self.tab_obem,
            text="Browse...",
            command=lambda: self._browse_file(
                self.obem_config_var,
                title="Select OBEM config file",
                filetypes=[("Config/INI", "*.ini *.cfg *.txt"), ("All Files", "*.*")],
            ),
        ).grid(row=1, column=2, sticky=tk.W)
        ttk.Button(
            self.tab_obem,
            text="Run Conversion",
            command=self._run_obem,
        ).grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.tab_obem.columnconfigure(1, weight=1)
        self._bind_var_audit("obem.config_file", self.obem_config_var)

    def _build_sac2y_tab(self) -> None:
        ttk.Label(
            self.tab_sac2y,
            text="Run sac2y_v2_1_obspy.py with 4 arguments:",
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))

        self.sac2y_sac_var = tk.StringVar(value="")
        self.sac2y_ukooa_var = tk.StringVar(value="")
        self.sac2y_output_var = tk.StringVar(value="")
        self.sac2y_config_var = tk.StringVar(value="")

        self._add_path_row(
            self.tab_sac2y, 1, "sac_file", self.sac2y_sac_var, "Select SAC file"
        )
        self._add_path_row(
            self.tab_sac2y, 2, "ukooa_file", self.sac2y_ukooa_var, "Select UKOOA file"
        )
        self._add_path_row(
            self.tab_sac2y,
            3,
            "output_segy_file",
            self.sac2y_output_var,
            "Select output SEGY file",
            save_dialog=True,
        )
        self._add_path_row(
            self.tab_sac2y,
            4,
            "config_file",
            self.sac2y_config_var,
            "Select config file",
            filetypes=[("Config/INI", "*.ini *.cfg *.txt"), ("All Files", "*.*")],
        )
        ttk.Button(
            self.tab_sac2y,
            text="Run Conversion",
            command=self._run_sac2y,
        ).grid(row=5, column=0, sticky=tk.W, pady=(10, 0))
        self.tab_sac2y.columnconfigure(1, weight=1)
        self._bind_var_audit("sac2y.sac_file", self.sac2y_sac_var)
        self._bind_var_audit("sac2y.ukooa_file", self.sac2y_ukooa_var)
        self._bind_var_audit("sac2y.output_segy_file", self.sac2y_output_var)
        self._bind_var_audit("sac2y.config_file", self.sac2y_config_var)

    def _build_raw2sac_tab(self) -> None:
        ttk.Label(
            self.tab_raw2sac,
            text="Run raw2sac_v1_1_obspy.py with fileName, sps, TC:",
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))
        self.raw_file_var = tk.StringVar(value="")
        self.raw_sps_var = tk.StringVar(value="1000")
        self.raw_tc_var = tk.StringVar(value="256")

        self._add_path_row(
            self.tab_raw2sac, 1, "fileName", self.raw_file_var, "Select raw data file"
        )
        ttk.Label(self.tab_raw2sac, text="sps").grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(self.tab_raw2sac, textvariable=self.raw_sps_var, width=16).grid(
            row=2, column=1, sticky=tk.W, padx=(6, 6)
        )
        ttk.Label(self.tab_raw2sac, text="TC").grid(row=3, column=0, sticky=tk.W)
        ttk.Entry(self.tab_raw2sac, textvariable=self.raw_tc_var, width=16).grid(
            row=3, column=1, sticky=tk.W, padx=(6, 6)
        )
        ttk.Button(
            self.tab_raw2sac,
            text="Run Conversion",
            command=self._run_raw2sac,
        ).grid(row=4, column=0, sticky=tk.W, pady=(10, 0))
        self.tab_raw2sac.columnconfigure(1, weight=1)
        self._bind_var_audit("raw2sac.file_name", self.raw_file_var)
        self._bind_var_audit("raw2sac.sps", self.raw_sps_var)
        self._bind_var_audit("raw2sac.tc", self.raw_tc_var)

    def _add_path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        var: tk.StringVar,
        dialog_title: str,
        *,
        filetypes: list[tuple[str, str]] | None = None,
        save_dialog: bool = False,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky=tk.EW, padx=(6, 6))
        if save_dialog:
            ttk.Button(
                parent,
                text="Browse...",
                command=lambda: self._browse_save(var, dialog_title),
            ).grid(row=row, column=2, sticky=tk.W)
            return
        ttk.Button(
            parent,
            text="Browse...",
            command=lambda: self._browse_file(var, title=dialog_title, filetypes=filetypes),
        ).grid(row=row, column=2, sticky=tk.W)

    def _browse_file(
        self,
        var: tk.StringVar,
        *,
        title: str,
        filetypes: list[tuple[str, str]] | None = None,
    ) -> None:
        selected = filedialog.askopenfilename(
            title=title,
            initialdir=self._default_open_initial_dir(),
            filetypes=filetypes or [("All Files", "*.*")],
        )
        if selected:
            backed_up = self._backup_input_file(selected)
            var.set(backed_up)
            self._audit("browse_file", title=title, selected=selected, backup=backed_up)

    def _browse_save(self, var: tk.StringVar, title: str) -> None:
        selected = filedialog.asksaveasfilename(
            title=title,
            initialdir=self._default_save_initial_dir(),
            defaultextension=".segy",
            filetypes=[("SEGY files", "*.segy *.sgy"), ("All Files", "*.*")],
        )
        if selected:
            target = self._rewrite_save_target(selected)
            var.set(target)
            self._audit("browse_save", title=title, selected=selected, rewritten=target)

    def _run_obem(self) -> None:
        config_file = self.obem_config_var.get().strip()
        if not config_file:
            messagebox.showwarning("Missing input", "config_file is required.")
            return
        self._audit("run_obem_clicked", config_file=config_file)
        self._run_script("obem_tsm_to_sac_obspy.py", [config_file])

    def _run_sac2y(self) -> None:
        args = [
            self.sac2y_sac_var.get().strip(),
            self.sac2y_ukooa_var.get().strip(),
            self.sac2y_output_var.get().strip(),
            self.sac2y_config_var.get().strip(),
        ]
        if not all(args):
            messagebox.showwarning(
                "Missing input",
                "sac_file / ukooa_file / output_segy_file / config_file are required.",
            )
            return
        self._audit("run_sac2y_clicked", args=args)
        self._run_script("sac2y_v2_1_obspy.py", args)

    def _run_raw2sac(self) -> None:
        file_name = self.raw_file_var.get().strip()
        sps = self.raw_sps_var.get().strip()
        tc = self.raw_tc_var.get().strip()
        if not file_name or not sps or not tc:
            messagebox.showwarning("Missing input", "fileName / sps / TC are required.")
            return
        self._audit("run_raw2sac_clicked", file_name=file_name, sps=sps, tc=tc)
        self._run_script("raw2sac_v1_1_obspy.py", [file_name, sps, tc])

    def _run_script(self, script_name: str, args: list[str]) -> None:
        script_path = self.raw2sac_dir / script_name
        if not script_path.exists():
            messagebox.showerror("Script not found", f"Cannot find script: {script_path}")
            return
        command = [self.python_exe, str(script_path)] + args
        self._active_jobs += 1
        self.status_var.set(f"Running ({self._active_jobs} active): {' '.join(command)}")
        self._append_log(f"$ {' '.join(command)}")
        self._audit("run_script_started", script=script_name, command=command)

        worker = threading.Thread(
            target=self._run_worker,
            args=(command,),
            daemon=True,
        )
        worker.start()

    def _run_worker(self, command: list[str]) -> None:
        try:
            proc = subprocess.run(
                command,
                cwd=self._default_run_cwd(),
                capture_output=True,
                text=True,
            )
            output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
            self.root.after(
                0,
                lambda: self._on_job_done(proc.returncode, output.strip()),
            )
        except Exception as exc:
            self.root.after(0, lambda: self._on_job_done(-1, f"Execution failed: {exc}"))

    def _on_job_done(self, return_code: int, output: str) -> None:
        self._active_jobs = max(0, self._active_jobs - 1)
        stamp = datetime.now().strftime("%H:%M:%S")
        self._append_log(f"[{stamp}] return_code={return_code}")
        if output:
            self._append_log(output)
        self._audit("run_script_finished", return_code=return_code, output_preview=output[:8000])
        if return_code == 0:
            self.status_var.set(f"Done (active jobs: {self._active_jobs}).")
        else:
            self.status_var.set(f"Failed (active jobs: {self._active_jobs}).")

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.configure(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)
        self._audit("log_cleared")

    def _on_tab_changed(self, _event: tk.Event | None = None) -> None:
        try:
            tab_id = self.notebook.select()
            title = str(self.notebook.tab(tab_id, "text"))
        except Exception:
            title = ""
        self._audit("tab_changed", tab=title)
        self._save_state()

    def _bind_var_audit(self, field_name: str, var: tk.StringVar) -> None:
        def on_change(*_args: object) -> None:
            value = var.get()
            last = self._field_last_values.get(field_name)
            if last == value:
                return
            self._field_last_values[field_name] = value
            self._audit("field_changed", field=field_name, value=value)
            self._save_state()

        var.trace_add("write", on_change)
        self._field_last_values[field_name] = var.get()

    def _restore_state(self) -> None:
        if self._gui_state_file is None:
            return
        try:
            if not self._gui_state_file.exists():
                return
            raw = json.loads(self._gui_state_file.read_text(encoding="utf-8"))
        except Exception:
            return
        idata_state = raw.get("idata", {}) if isinstance(raw, dict) else {}
        if not isinstance(idata_state, dict):
            return
        fields = idata_state.get("fields", {})
        if isinstance(fields, dict):
            self.obem_config_var.set(
                self._normalize_restored_path(str(fields.get("obem_config", self.obem_config_var.get())))
            )
            self.sac2y_sac_var.set(
                self._normalize_restored_path(str(fields.get("sac2y_sac", self.sac2y_sac_var.get())))
            )
            self.sac2y_ukooa_var.set(
                self._normalize_restored_path(str(fields.get("sac2y_ukooa", self.sac2y_ukooa_var.get())))
            )
            self.sac2y_output_var.set(
                self._normalize_restored_path(str(fields.get("sac2y_output", self.sac2y_output_var.get())))
            )
            self.sac2y_config_var.set(
                self._normalize_restored_path(str(fields.get("sac2y_config", self.sac2y_config_var.get())))
            )
            self.raw_file_var.set(
                self._normalize_restored_path(str(fields.get("raw_file", self.raw_file_var.get())))
            )
            self.raw_sps_var.set(
                self._normalize_restored_path(str(fields.get("raw_sps", self.raw_sps_var.get())))
            )
            self.raw_tc_var.set(
                self._normalize_restored_path(str(fields.get("raw_tc", self.raw_tc_var.get())))
            )
        try:
            tab_index = int(idata_state.get("tab_index", 0))
        except (TypeError, ValueError):
            tab_index = 0
        try:
            self.notebook.select(max(0, min(tab_index, len(self.notebook.tabs()) - 1)))
        except Exception:
            pass
        geometry = str(idata_state.get("window_geometry", "") or "").strip()
        if geometry:
            try:
                self.root.geometry(geometry)
            except Exception:
                pass

    def _save_state(self) -> None:
        if self._gui_state_file is None:
            return
        try:
            existing: dict[str, object] = {}
            if self._gui_state_file.exists():
                loaded = json.loads(self._gui_state_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            tab_index = 0
            try:
                tab_index = int(self.notebook.index(self.notebook.select()))
            except Exception:
                tab_index = 0
            existing["idata"] = {
                "window_geometry": self.root.geometry(),
                "tab_index": tab_index,
                "fields": {
                    "obem_config": self.obem_config_var.get(),
                    "sac2y_sac": self.sac2y_sac_var.get(),
                    "sac2y_ukooa": self.sac2y_ukooa_var.get(),
                    "sac2y_output": self.sac2y_output_var.get(),
                    "sac2y_config": self.sac2y_config_var.get(),
                    "raw_file": self.raw_file_var.get(),
                    "raw_sps": self.raw_sps_var.get(),
                    "raw_tc": self.raw_tc_var.get(),
                },
            }
            self._gui_state_file.parent.mkdir(parents=True, exist_ok=True)
            self._gui_state_file.write_text(
                json.dumps(existing, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    def _default_open_initial_dir(self) -> str:
        if self._project_root:
            return self._project_root
        if self._run_dir is not None:
            return str(self._run_dir)
        return str(self.raw2sac_dir)

    def _default_save_initial_dir(self) -> str:
        if self._outputs_dir is not None:
            self._outputs_dir.mkdir(parents=True, exist_ok=True)
            return str(self._outputs_dir)
        return str(self.raw2sac_dir)

    def _default_run_cwd(self) -> str:
        if self._outputs_dir is not None:
            self._outputs_dir.mkdir(parents=True, exist_ok=True)
            return str(self._outputs_dir)
        return str(self.raw2sac_dir)

    def _backup_input_file(self, selected: str) -> str:
        src = None
        for cand in self._candidate_existing_paths(selected):
            if cand.exists() and cand.is_file():
                src = cand
                break
        if src is None:
            src = Path(selected).expanduser()
        if self._inputs_dir is None or not src.exists() or not src.is_file():
            return selected
        self._inputs_dir.mkdir(parents=True, exist_ok=True)
        dest = self._inputs_dir / src.name
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
                candidate = self._inputs_dir / f"{stem}_{idx}{suffix}"
                if not candidate.exists():
                    dest = candidate
                    break
                idx += 1
        try:
            shutil.copy2(src, dest)
            return str(dest)
        except Exception as exc:
            self._audit("input_backup_failed", source=str(src), error=str(exc))
            return selected

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

    def _normalize_restored_path(self, path_text: str) -> str:
        raw = str(path_text or "").strip()
        if not raw:
            return ""
        for cand in self._candidate_existing_paths(raw):
            if cand.exists():
                return str(cand)
        return raw

    def _rewrite_save_target(self, selected: str) -> str:
        if self._outputs_dir is None:
            return selected
        self._outputs_dir.mkdir(parents=True, exist_ok=True)
        chosen = Path(selected).expanduser()
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
        return str(target)


def main() -> None:
    app = IdataApp()
    app.run()


if __name__ == "__main__":
    main()
