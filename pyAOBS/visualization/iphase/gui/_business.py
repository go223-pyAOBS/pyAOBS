"""IPhase business methods ported from iphase_gui (Tk-independent).

Uses `_Var` shims so former `.get()`/`.set()` call sites stay intact.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFileDialog, QMessageBox

from ..phase_filter import select_phases
from ..phase_combine import combine_ppp_pps_pss_from_datasets, compute_ppp_pps_diff_pairs
from ..qc_metrics import stats_ppp_pps_diff_by_offset
from ..io_tx import read_tx, write_tx
from ..models import PhaseDataset, Shot
from ..plot_tx import plot_offset_time
from ..theoretical_ppp_pps import (
    LocalInversionResult,
    fit_ppp_time_curve_local_linear,
    invert_local_h_vpratio,
    invert_m_field_from_paths,
    pss_minus_psp_from_pss_slope_with_profile,
    pps_minus_ppp_from_ppp_slope,
)
from ..equi2d import build_equi_dataset, build_equi_dataset_picks_plus_equiv_psp
from ..theory2d_service import (
    Theory2DBundle,
    build_theory2d_bundle_from_txout,
    collect_phase_points_from_txout,
    collect_theory2d_delta_by_branch,
    hash_file,
    make_delta_query,
    merge_phase_points_sides,
    parse_pois_from_rin,
    parse_rin_input_files,
    run_rayinvr_forward,
    set_rin_tfile,
    split_phase_points_by_offset_side,
    update_rin_shots_from_receivers_and_depth,
    write_pois_full_to_rin,
    write_pois_to_rin,
)
from ..rin_phase_groups import parse_rin_phase_arrays_from_file

from .dialog_utils import show_modeless_dialog, show_modeless_message
from .dialogs.export_tx_select import (
    ExportTxSelectDialog,
    build_preview_picks_from_results,
    _stations_from_depth_arrays,
)
from .dialogs.plot_tool_window import PlotToolWindow
from .plotting.main_figure import MainFigurePlotMixin
from .services.file_result import (
    DEFAULT_PSP_PHASE,
    PHASE_PPP,
    PHASE_PPS,
    PHASE_PSS,
    FileResult,
    compute_file_result as _compute_file_result,
    compute_file_results as _compute_file_results,
    equi_tx_path_for_result as _equi_tx_path_for_result,
    obs_model_distance_from_tx as _obs_model_distance_from_tx,
    obs_tag_from_path as _obs_tag_from_path,
    obs_tag_from_result as _obs_tag_from_result,
    phase_model_distance_time as _phase_model_distance_time,
    phase_model_trueoff_time as _phase_model_trueoff_time,
    phase_true_offset_time as _phase_true_offset_time,
    result_cache_key as _result_cache_key,
    result_display_name as _result_display_name,
)
from .services.workdir_layout import OUTPUTS_DIR


class _Var:
    """Minimal stand-in for tk.Variable (.get / .set)."""

    __slots__ = ("_value",)

    def __init__(self, value=None) -> None:
        self._value = value

    def get(self):
        return self._value

    def set(self, value) -> None:
        self._value = value


class _StatusVar(_Var):
    """Status string that also updates a QLabel when bound."""

    __slots__ = ("_value", "_label")

    def __init__(self, value: str = "", label=None) -> None:
        self._value = value
        self._label = label

    def bind_label(self, label) -> None:
        self._label = label
        if label is not None:
            label.setText(str(self._value))

    def set(self, value) -> None:
        self._value = value
        if self._label is not None:
            try:
                self._label.setText(str(value))
            except Exception:
                pass


class _MsgBox:
    @staticmethod
    def showinfo(title, message):
        show_modeless_message(str(title), str(message), icon=QMessageBox.Icon.Information)

    @staticmethod
    def showwarning(title, message):
        show_modeless_message(str(title), str(message), icon=QMessageBox.Icon.Warning)

    @staticmethod
    def showerror(title, message):
        show_modeless_message(str(title), str(message), icon=QMessageBox.Icon.Critical)


messagebox = _MsgBox


class _FileDialog:
    @staticmethod
    def askopenfilename(title="", filetypes=None, **_kw):
        filters = _qt_filters(filetypes)
        path, _ = QFileDialog.getOpenFileName(None, title or "打开", "", filters)
        return path or ""

    @staticmethod
    def askopenfilenames(title="", filetypes=None, **_kw):
        filters = _qt_filters(filetypes)
        paths, _ = QFileDialog.getOpenFileNames(None, title or "打开", "", filters)
        return paths or []

    @staticmethod
    def asksaveasfilename(title="", filetypes=None, defaultextension="", typevariable=None, **_kw):
        filters = _qt_filters(filetypes)
        path, selected = QFileDialog.getSaveFileName(None, title or "保存", "", filters)
        if path and typevariable is not None and selected:
            typevariable.set(selected.split("(", 1)[0].strip() or typevariable.get())
        return path or ""


filedialog = _FileDialog


def _qt_filters(filetypes) -> str:
    if not filetypes:
        return "All files (*.*)"
    parts = []
    for label, pattern in filetypes:
        pat = str(pattern).replace(";", " ").replace(",", " ")
        globs = []
        for tok in pat.split():
            tok = tok.strip()
            if not tok:
                continue
            if "*" not in tok and "." not in tok:
                tok = f"*.{tok}"
            elif not tok.startswith("*"):
                tok = f"*{tok}" if tok.startswith(".") else f"*.{tok}"
            globs.append(tok)
        if not globs:
            globs = ["*.*"]
        parts.append(f"{label} ({' '.join(globs)})")
    return ";;".join(parts)


class IPhaseBusinessMixin(MainFigurePlotMixin):
    """Business logic formerly on IPhaseGui (minus Tk chrome)."""


    def open_single(self) -> None:
        p = filedialog.askopenfilename(
            title="选择 tx.in",
            filetypes=[("tx files", "*.in"), ("all files", "*.*")],
        )
        if not p:
            return
        self.files = [Path(p)]
        self._load_and_draw()
        self._save_workbench_state()

    def open_multiple(self) -> None:
        ps = filedialog.askopenfilenames(
            title="选择多个 tx.in",
            filetypes=[("tx files", "*.in"), ("all files", "*.*")],
        )
        if not ps:
            return
        self.files = [Path(x) for x in ps]
        self._load_and_draw()
        self._save_workbench_state()

    def open_rin_phase_groups_editor(self) -> None:
        """
        从 iphase 打开独立 r.in 相位组编辑器。

        采用独立进程，避免在同一 Tk 进程中创建第二个主窗口导致不稳定。
        编辑器打开期间轮询 r.in 修改时间：保存后即自动重载，无需先关闭编辑器。
        """
        if self._rin_editor_proc is not None and self._rin_editor_proc.poll() is None:
            self.status_var.set("r.in 相位组编辑器已在运行；保存后会自动重载")
            return
        rin_path = self._guess_rin_path_for_current_context()
        if rin_path is None:
            chosen = filedialog.askopenfilename(
                title="选择 r.in",
                filetypes=[("RAYINVR input", "r.in"), ("All files", "*.*")],
            )
            if not chosen:
                return
            rin_path = Path(chosen)
        try:
            # 用 -m 启动，避免“脚本直接运行”导致包导入失败
            repo_root = str(Path(__file__).resolve().parents[4])
            env = os.environ.copy()
            pp = env.get("PYTHONPATH", "").strip()
            env["PYTHONPATH"] = repo_root if not pp else f"{repo_root}{os.pathsep}{pp}"
            self._rin_editor_proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "pyAOBS.visualization.iphase.rin_phase_groups_gui",
                    str(rin_path),
                ],
                env=env,
            )
            self._rin_editor_path = rin_path
            try:
                self._rin_editor_last_mtime_ns = rin_path.stat().st_mtime_ns
            except OSError:
                self._rin_editor_last_mtime_ns = None
            self.status_var.set(f"已打开 r.in 相位组编辑器：{rin_path.name}（保存后自动重载）")
            self._schedule_rin_editor_poll()
        except Exception as e:
            messagebox.showerror("打开失败", f"启动 r.in 相位组编辑器失败：\n{e}")
            self._rin_editor_proc = None
            self._rin_editor_path = None
            self._rin_editor_last_mtime_ns = None
            return

    def _schedule_rin_editor_poll(self) -> None:
        if self._rin_editor_poll_job is None:
            self._rin_editor_poll_job = self.after(900, self._poll_rin_editor)

    def _poll_rin_editor(self) -> None:
        self._rin_editor_poll_job = None
        proc = self._rin_editor_proc
        rin_path = self._rin_editor_path
        if proc is None or rin_path is None:
            return

        changed = False
        try:
            now_mtime = rin_path.stat().st_mtime_ns
        except OSError:
            now_mtime = None
        if now_mtime is not None and now_mtime != self._rin_editor_last_mtime_ns:
            changed = True
            self._rin_editor_last_mtime_ns = now_mtime

        if changed and self.files:
            self._reload_if_loaded()
            self.status_var.set("检测到 r.in 已保存：iphase 已自动重载")

        if proc.poll() is None:
            self._schedule_rin_editor_poll()
            return

        # 编辑器退出：做一次兜底刷新，避免最后一次保存错过轮询窗口
        if self.files:
            self._reload_if_loaded()
        self.status_var.set("r.in 相位组编辑器已关闭")
        self._rin_editor_proc = None
        self._rin_editor_path = None
        self._rin_editor_last_mtime_ns = None

    def redraw(self) -> None:
        if not self.files:
            return
        self._draw_results()

    def _on_model_param_changed(self) -> None:
        """关键模型参数变化后，失效旧反演状态。"""
        self.pss_inversion_ready = False
        self.pss_profile_by_file = {}
        self.redraw()

    def load_seafloor_depth(self) -> None:
        """
        加载海底深度文本文件：
        - 第1列: 模型距离 x (km)
        - 第2列: 海底深度 z (km, 向下为正)
        """
        p = filedialog.askopenfilename(
            title="选择海底深度文件",
            filetypes=[
                ("Depth text", "*.txt *.dat *.asc *.csv"),
                ("TXT", "*.txt"),
                ("DAT", "*.dat"),
                ("ASC", "*.asc"),
                ("CSV", "*.csv"),
                ("All files", "*.*"),
            ],
        )
        if not p:
            return
        self._load_seafloor_depth_from_file(Path(p), show_error=True)

    def load_shot_depth_table(self) -> None:
        """
        加载 xshot、zshot 两列文件；若已打开 tx.in 则自动用该表更新 r.in。
        """
        p = filedialog.askopenfilename(
            title="选择炮点深度表（两列 xshot zshot，或三列 station.lis）",
            filetypes=[
                ("Text/CSV/LIS", "*.txt *.dat *.asc *.csv *.lis"),
                ("TXT", "*.txt"),
                ("CSV", "*.csv"),
                ("LIS", "*.lis"),
                ("All files", "*.*"),
            ],
        )
        if not p:
            return
        self._load_shot_depth_from_file(Path(p), show_error=True)

    def _load_seafloor_depth_from_file(self, path: Path, *, show_error: bool) -> bool:
        try:
            arr = np.loadtxt(path, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] < 2:
                raise ValueError("文件至少需要两列：x depth")
            x = arr[:, 0].astype(float)
            z = arr[:, 1].astype(float)
            v = np.isfinite(x) & np.isfinite(z)
            x = x[v]
            z = z[v]
            if x.size < 2:
                raise ValueError("有效点数不足，至少需要2行")
            so = np.argsort(x)
            x = x[so]
            z = z[so]
            self.seafloor_path = path
            self.seafloor_x = x
            self.seafloor_z = z
            self.status_var.set(f"已加载海底深度: {self.seafloor_path.name} ({x.size} points)")
            self._save_workbench_state()
            return True
        except Exception as e:
            if show_error:
                messagebox.showerror("加载失败", f"海底深度文件格式错误:\n{e}")
            return False

    def _load_shot_depth_from_file(self, path: Path, *, show_error: bool) -> bool:
        try:
            x, z, labels = self._parse_xz_depth_table(path, return_ids=True)
            if x.size == 0:
                raise ValueError("有效点数不足")
            self.shot_depth_path = path
            self.shot_depth_x = x
            self.shot_depth_z = z
            self.shot_depth_ids = labels
            self.status_var.set(f"已加载炮点深度表: {self.shot_depth_path.name} ({x.size} 行)")
            self._try_auto_sync_rin_shots()
            self._save_workbench_state()
            return True
        except Exception as e:
            if show_error:
                messagebox.showerror("加载失败", f"炮点深度表格式错误:\n{e}")
            return False

    @staticmethod
    def _parse_xz_depth_table(
        path: Path,
        *,
        return_ids: bool = False,
    ):
        """
        解析深度表为 (x, z)，可选站号。

        支持：
        - 两列：``xshot zshot``（iphase 默认，如 ``offset_depth_OBS.txt``）
        - 三列及以上：按 ``station.lis`` 约定取第 2、3 列（``id  x  z``）
        """
        arr = np.loadtxt(path, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        ncols = int(arr.shape[1]) if arr.size else 0
        if ncols < 2:
            raise ValueError(
                "文件至少需要两列：xshot zshot；"
                "或三列 station.lis：站号 x z"
            )
        if ncols >= 3:
            ids_raw = arr[:, 0]
            x = arr[:, 1].astype(float)
            z = arr[:, 2].astype(float)
        else:
            ids_raw = None
            x = arr[:, 0].astype(float)
            z = arr[:, 1].astype(float)
        v = np.isfinite(x) & np.isfinite(z)
        x = x[v]
        z = z[v]
        if not return_ids:
            return x, z
        labels: list[str] | None = None
        if ids_raw is not None:
            ids_f = np.asarray(ids_raw, dtype=float)[v]
            labels = []
            for val in ids_f:
                if np.isfinite(val) and abs(val - round(val)) < 1e-6:
                    labels.append(str(int(round(val))))
                elif np.isfinite(val):
                    labels.append(f"{val:g}")
                else:
                    labels.append("")
        return x, z, labels

    def _try_auto_sync_rin_shots(self) -> None:
        """若已打开 tx.in 且已加载炮点深度表，则自动更新 r.in；结果与提示仅显示在状态栏。"""
        if not self.results:
            return
        if self.shot_depth_x is None or self.shot_depth_z is None or self.shot_depth_x.size == 0:
            self.status_var.set("提示：未加载炮点深度表，无法自动更新 r.in")
            return
        work_dir = self.results[0].path.parent
        r_in = work_dir / "r.in"
        receiver_x: list[float] = []
        seen: set[float] = set()
        for r in self.results:
            xobs = _obs_model_distance_from_tx(r)
            if xobs is None:
                continue
            xr = round(float(xobs), 3)
            if xr in seen:
                continue
            seen.add(xr)
            receiver_x.append(float(xobs))
        if not receiver_x:
            receiver_x = [float(s.xshot) for s in self.results[0].ds.shots]
        ok, msg = update_rin_shots_from_receivers_and_depth(
            receiver_x,
            self.shot_depth_x,
            self.shot_depth_z,
            r_in,
        )
        if ok:
            self.status_var.set(msg)
        else:
            self.status_var.set(f"r.in 未更新: {msg}")

    def _refresh_pois_from_rin(self) -> None:
        """优先 pois_branches.json；否则保留工区 analysis 已有左右支；再否则从 r.in 填双侧同值。"""
        if self._load_pois_branches_file():
            return
        left = str(self.pois_left_var.get()).strip()
        right = str(self.pois_right_var.get()).strip()
        r_in = self._resolve_active_rin_path()
        if r_in is not None:
            first, rest = parse_pois_from_rin(r_in)
            if first is not None:
                self.pois_first_val = first
            if left or right:
                # 工区已恢复左右支，只补 first，不覆盖
                return
            if rest or first is not None:
                self.pois_left_var.set(rest)
                self.pois_right_var.set(rest)
                left_edit = getattr(self, "_pois_left_edit", None)
                right_edit = getattr(self, "_pois_right_edit", None)
                if left_edit is not None:
                    left_edit.blockSignals(True)
                    left_edit.setText(rest)
                    left_edit.blockSignals(False)
                if right_edit is not None:
                    right_edit.blockSignals(True)
                    right_edit.setText(rest)
                    right_edit.blockSignals(False)

    def _ensure_tx_equi_for_results(self) -> None:
        """后台写 tx_2Dequiv.in：未勾选等效 PSP 时为合成 PPP/PPS+PSS；勾选时为原始 PPP/PPS/PSS/PSP + 等效 PSP。"""
        try:
            ratio = float(self.pps_pss_ratio.get())
        except Exception:
            ratio = 1.0
        if ratio <= 0:
            ratio = 1.0
        try:
            psp_ph = int(self.psp_phase_id.get())
        except Exception:
            psp_ph = DEFAULT_PSP_PHASE
        use_picks_equiv = bool(self.equi_write_equiv_psp.get())
        phase_psp_out = psp_ph if use_picks_equiv else None
        for r in self.results:
            try:
                if use_picks_equiv:
                    out_ds = build_equi_dataset_picks_plus_equiv_psp(
                        r.ds,
                        r.diff_pairs,
                        phase_ppp=PHASE_PPP,
                        phase_pps=PHASE_PPS,
                        phase_pss=PHASE_PSS,
                        phase_psp=psp_ph,
                    )
                else:
                    out_ds = build_equi_dataset(
                        r.diff_pairs,
                        r.ds_pss,
                        pps_pss_ratio=ratio,
                        phase_ppp=PHASE_PPP,
                        phase_pps=PHASE_PPS,
                        phase_pss=PHASE_PSS,
                        phase_psp=None,
                    )
                if out_ds.n_picks > 0:
                    out_path = _equi_tx_path_for_result(r)
                    write_tx(out_ds, out_path)
            except Exception:
                pass

    def _file_result_for_plot(self, r: FileResult) -> FileResult:
        """
        2Dequi 模式：左上图/PPP-PPS 差值/理论对比使用 2Dequi tx；
        FileResult.path 仍保留为原始 tx.in；保留 obs_x/obs_id。
        """
        if self.theory_mode.get() != "2Dequi":
            return r
        equi = _equi_tx_path_for_result(r)
        if not equi.exists():
            # 兼容旧文件名
            legacy = r.path.parent / "tx_2Dequiv.in"
            equi = legacy if legacy.exists() else equi
        if not equi.exists():
            return r
        try:
            allowed_phase_ids = self._enabled_ivrays_for_tx_path(r.path) if bool(self.use_rin_enabled_filter.get()) else None
            from .services.file_result import (
                compute_file_result_from_dataset,
                split_phase_dataset_by_obs_x,
            )

            ds_equi = read_tx(str(equi))
            if allowed_phase_ids is not None:
                if allowed_phase_ids:
                    ds_equi, _ = select_phases(ds_equi, sorted(allowed_phase_ids))
                else:
                    ds_equi = PhaseDataset()
            if r.obs_x is not None:
                parts = split_phase_dataset_by_obs_x(ds_equi)
                match = None
                for xobs, part in parts:
                    if abs(float(xobs) - float(r.obs_x)) <= 0.001:
                        match = part
                        break
                if match is not None:
                    ds_equi = match
            r2 = compute_file_result_from_dataset(
                equi,
                ds_equi,
                psp_phase_id=int(self.psp_phase_id.get()),
                window_points=self._window_points_value(),
                strict_only=bool(self.strict_diff_pair.get()),
                obs_x=r.obs_x,
                obs_id=r.obs_id,
            )
            return FileResult(
                path=r.path,
                ds=r2.ds,
                ds_ppp=r2.ds_ppp,
                ds_pps=r2.ds_pps,
                ds_pss=r2.ds_pss,
                diff_pairs=r.diff_pairs
                if bool(self.equi_write_equiv_psp.get())
                else r2.diff_pairs,
                fit_stats=r2.fit_stats,
                tx2_out=r2.tx2_out,
                obs_x=r.obs_x,
                obs_id=r.obs_id,
            )
        except Exception:
            return r

    def _enabled_ivrays_for_tx_path(self, tx_path: Path) -> set[int] | None:
        """
        从 tx 同目录 r.in 读取保留组：
        - enabled 语义按 nray>0 判断（写回禁用组时 nray=0）
        - 返回 ivray 集合；若 r.in 不存在或解析失败，返回 None（不启用过滤）
        """
        r_in = tx_path.parent / "r.in"
        if not r_in.exists():
            return None
        try:
            arr = parse_rin_phase_arrays_from_file(r_in, parse_path_arrays=False)
            n = min(len(arr.ivray), len(arr.nray))
            return {int(arr.ivray[i]) for i in range(n) if int(arr.nray[i]) > 0}
        except Exception:
            return None

    def _on_pps_pss_ratio_changed(self) -> None:
        """PPS/PSS 比值变化时重新生成 tx_2Dequiv.in，若为 2Dequi 模式则运行 2D 正演并重绘。"""
        self._ensure_tx_equi_for_results()
        if self.theory_mode.get() == "2Dequi":
            self._on_theory_mode_changed()
        else:
            self.redraw()

    def _on_equi_equiv_psp_changed(self) -> None:
        """是否写入等效 PSP 变更：重写 tx_2Dequiv.in；2Dequi 下重跑正演。"""
        self._ensure_tx_equi_for_results()
        if self.theory_mode.get() == "2Dequi":
            self._on_theory_mode_changed()
        else:
            self.redraw()

    def _run_2dequi_forward_all(self) -> None:
        """对当前已加载结果逐个目录运行 2Dequi 正演（r.in→tx_2Dequiv.in），并更新分支理论缓存。"""
        if not self.results:
            return
        self._ensure_tx_equi_for_results()
        for r in self.results:
            wd = r.path.parent
            r_in = wd / "r.in"
            tx_equi = wd / "tx_2Dequiv.in"
            if r_in.exists() and tx_equi.exists():
                set_rin_tfile(r_in, "tx_2Dequiv.in")
                fr = run_rayinvr_forward(
                    wd,
                    force_run=True,
                    tx_in_override=tx_equi,
                    sync_override_to_tfile=True,
                )
                if fr.success:
                    tx_out = wd / "tx.out"
                    try:
                        left_b, right_b = collect_theory2d_delta_by_branch(
                            tx_out, phase_ppp=PHASE_PPP, phase_pps=PHASE_PPS
                        )
                        self._theory2d_left_branch = left_b
                        self._theory2d_right_branch = right_b
                    except Exception:
                        pass

    def _on_theory_mode_changed(self) -> None:
        """时差模式切换：2Dequi 时更新 r.in tfile、自动生成 tx_2Dequiv.in、运行 2D 正演。"""
        self._clear_theory2d_branch_cache()
        mode = self.theory_mode.get()
        if mode == "2Dequi" and self.results:
            self._run_2dequi_forward_all()
        self.redraw()

    def _flush_param_edits(self) -> None:
        """提交参数条中尚未触发 editingFinished 的行编辑（点按钮前调用）。"""
        flushers = getattr(self, "_param_line_flushers", None)
        if not flushers:
            return
        for fn in list(flushers):
            try:
                fn()
            except Exception:
                pass

    def _t2d_side_cache_map(self) -> dict:
        if not hasattr(self, "_t2d_side_cache") or self._t2d_side_cache is None:
            self._t2d_side_cache = {}
        return self._t2d_side_cache

    @staticmethod
    def _t2d_wd_key(path: Path) -> str:
        try:
            return str(Path(path).resolve())
        except Exception:
            return str(path)

    def _clear_theory2d_branch_cache(self, work_dir: Path | None = None) -> None:
        self._theory2d_left_branch = None
        self._theory2d_right_branch = None
        cache = self._t2d_side_cache_map()
        if work_dir is None:
            cache.clear()
            return
        cache.pop(self._t2d_wd_key(work_dir), None)

    def _ensure_t2d_side_entry(self, work_dir: Path) -> dict:
        key = self._t2d_wd_key(work_dir)
        cache = self._t2d_side_cache_map()
        ent = cache.get(key)
        if ent is None:
            ent = {
                "left_delta": None,
                "right_delta": None,
                "left_phases": None,
                "right_phases": None,
            }
            cache[key] = ent
        return ent

    def _update_t2d_side_from_txout(
        self,
        work_dir: Path,
        *,
        side: str,
        phase_ids: tuple[int, ...] | None = None,
    ) -> None:
        """从当前 tx.out 更新左/右/双侧缓存。side: left|right|both。"""
        tx_out = Path(work_dir) / "tx.out"
        ent = self._ensure_t2d_side_entry(work_dir)
        pids = phase_ids or (PHASE_PPP, PHASE_PPS, int(self.psp_phase_id.get()))
        left_b, right_b = (np.asarray([]), np.asarray([])), (np.asarray([]), np.asarray([]))
        left_ph: dict = {}
        right_ph: dict = {}
        if tx_out.exists():
            try:
                left_b, right_b = collect_theory2d_delta_by_branch(
                    tx_out, phase_ppp=PHASE_PPP, phase_pps=PHASE_PPS
                )
            except Exception:
                pass
            try:
                pts = collect_phase_points_from_txout(tx_out, phase_ids=tuple(int(p) for p in pids))
                left_ph, right_ph = split_phase_points_by_offset_side(pts)
            except Exception:
                left_ph, right_ph = {}, {}
        side_l = str(side).lower()
        if side_l in ("left", "both"):
            ent["left_delta"] = left_b
            ent["left_phases"] = left_ph
            self._theory2d_left_branch = left_b
        if side_l in ("right", "both"):
            ent["right_delta"] = right_b
            ent["right_phases"] = right_ph
            self._theory2d_right_branch = right_b

    def _merged_theory_phase_pts(
        self,
        r: FileResult,
        *,
        phase_ids: tuple[int, ...],
    ) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """绘图用：合并左右支缓存；若尚无缓存则从 tx.out 初始化双侧。"""
        wd = r.path.parent
        ent = self._ensure_t2d_side_entry(wd)
        if ent.get("left_phases") is None and ent.get("right_phases") is None:
            self._update_t2d_side_from_txout(wd, side="both", phase_ids=phase_ids)
            ent = self._ensure_t2d_side_entry(wd)
        return merge_phase_points_sides(ent.get("left_phases"), ent.get("right_phases"))

    def _invalidate_theory2d_disk_cache(self, work_dir: Path | None = None) -> None:
        """删除 theory2d pickle 缓存，避免正演后仍读旧 bundle。"""
        try:
            cache_dir = self._theory2d_cache_dir()
        except Exception:
            return
        if work_dir is not None:
            try:
                key = self._theory2d_cache_key(
                    Path(work_dir), phase_ppp=PHASE_PPP, phase_pps=PHASE_PPS
                )
                pkl = cache_dir / f"bundle_{key}.pkl"
                if pkl.exists():
                    pkl.unlink()
            except Exception:
                pass
            return
        try:
            for p in cache_dir.glob("bundle_*.pkl"):
                try:
                    p.unlink()
                except Exception:
                    pass
        except Exception:
            pass

    def _pois_branches_json_path(self) -> Path | None:
        r_in = self._resolve_active_rin_path()
        if r_in is not None:
            return r_in.parent / "pois_branches.json"
        if self.results:
            return self.results[0].path.parent / "pois_branches.json"
        if self.files:
            return Path(self.files[0]).parent / "pois_branches.json"
        workdir = getattr(getattr(self, "project", None), "workdir", None)
        if workdir:
            return Path(workdir) / "pois_branches.json"
        return None

    def save_pois_branches(self) -> None:
        """保存左/右支 pois 到 pois_branches.json，并写入工区 analysis。"""
        self._flush_param_edits()
        left_edit = getattr(self, "_pois_left_edit", None)
        right_edit = getattr(self, "_pois_right_edit", None)
        left = (left_edit.text().strip() if left_edit is not None else "") or str(
            self.pois_left_var.get()
        ).strip()
        right = (right_edit.text().strip() if right_edit is not None else "") or str(
            self.pois_right_var.get()
        ).strip()
        self.pois_left_var.set(left)
        self.pois_right_var.set(right)
        first = self.pois_first_val if self.pois_first_val is not None else 0.5
        payload = {
            "first": float(first),
            "left": left,
            "right": right,
            "full_left": f"{float(first):.3f},{left}" if left else f"{float(first):.3f}",
            "full_right": f"{float(first):.3f},{right}" if right else f"{float(first):.3f}",
        }
        out = self._pois_branches_json_path()
        if out is None:
            messagebox.showwarning("保存 pois", "无法确定保存目录（请先加载 tx.in 或指定 r.in）。")
            return
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("保存 pois 失败", f"{out}\n{exc}")
            return
        try:
            proj = getattr(self, "project", None)
            if proj is not None and getattr(proj, "analysis", None) is not None:
                proj.analysis.pois_left = left
                proj.analysis.pois_right = right
                if getattr(proj, "workdir", None):
                    proj.dirty = True
        except Exception:
            pass
        self.status_var.set(f"已保存左/右支 pois → {out}")
        messagebox.showinfo("保存 pois", f"已保存左支与右支 pois：\n{out}")

    def _load_pois_branches_file(self) -> bool:
        """若存在 pois_branches.json 则载入左右支到界面。返回是否载入成功。"""
        path = self._pois_branches_json_path()
        if path is None or not path.is_file():
            return False
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not isinstance(raw, dict):
            return False
        try:
            first = raw.get("first", None)
            if first is not None:
                self.pois_first_val = float(first)
        except Exception:
            pass
        left = str(raw.get("left", "") or "").strip()
        right = str(raw.get("right", "") or "").strip()
        if not left and not right:
            return False
        if left:
            self.pois_left_var.set(left)
        if right:
            self.pois_right_var.set(right)
        for edit, val in (
            (getattr(self, "_pois_left_edit", None), left),
            (getattr(self, "_pois_right_edit", None), right),
        ):
            if edit is not None and val:
                edit.blockSignals(True)
                edit.setText(val)
                edit.blockSignals(False)
        return True

    def _guess_rin_path_for_current_context(self) -> Path | None:
        """优先：输入页指定 / 编辑器路径 / 结果目录 / 工区目录下的 r.in。"""
        return self._resolve_active_rin_path()

    def _resolve_active_rin_path(self) -> Path | None:
        """解析当前应读写的 r.in（必须已存在）。

        优先走时文件同目录（与 2D 正演 cwd 一致），再才是输入页/工区指定路径。
        """
        candidates: list[Path] = []
        if self.results:
            candidates.append(self.results[0].path.parent / "r.in")
        if self.files:
            candidates.append(Path(self.files[0]).parent / "r.in")
        try:
            panel = getattr(self, "panel_input", None)
            if panel is not None:
                raw = str(panel.rin_path() or "").strip()
                if raw:
                    candidates.append(Path(raw))
        except Exception:
            pass
        if getattr(self, "_rin_editor_path", None):
            candidates.append(Path(self._rin_editor_path))
        workdir = getattr(getattr(self, "project", None), "workdir", None)
        if workdir:
            candidates.append(Path(workdir) / "r.in")
        seen: set[str] = set()
        for p in candidates:
            try:
                key = str(p.resolve()) if p.exists() else str(p)
            except Exception:
                key = str(p)
            if key in seen:
                continue
            seen.add(key)
            if p.is_file():
                return p
        return None

    def _read_pois_rest_from_ui(self, *, prefer: str = "left") -> tuple[str, str]:
        """直接读界面框文字（优先于 var），返回 (rest, which)。"""
        left_edit = getattr(self, "_pois_left_edit", None)
        right_edit = getattr(self, "_pois_right_edit", None)
        left = ""
        right = ""
        if left_edit is not None:
            try:
                left = left_edit.text().strip()
            except Exception:
                left = ""
        if right_edit is not None:
            try:
                right = right_edit.text().strip()
            except Exception:
                right = ""
        if not left:
            left = str(self.pois_left_var.get()).strip()
        if not right:
            right = str(self.pois_right_var.get()).strip()
        # 同步回 var，避免后续逻辑读到旧值
        if left:
            self.pois_left_var.set(left)
        if right:
            self.pois_right_var.set(right)
        if prefer == "right":
            rest = right or left
            which = "右支" if right else "左支"
        else:
            rest = left or right
            which = "左支" if left else "右支"
        return rest, which

    def _sync_pois_ui_to_rin(self, *, prefer: str = "left") -> tuple[bool, str]:
        """把界面 pois（左/右支）写回当前解析到的 r.in。"""
        self._flush_param_edits()
        rest, which = self._read_pois_rest_from_ui(prefer=prefer)
        if not rest:
            return False, "pois 为空，跳过写回"
        first = self.pois_first_val if self.pois_first_val is not None else 0.5
        full = f"{first:.3f},{rest}"
        r_in = self._resolve_active_rin_path()
        if r_in is None:
            # 尝试落在 tx 同目录（即使尚无文件也不擅自创建完整 r.in）
            hint = ""
            if self.results:
                hint = str(self.results[0].path.parent / "r.in")
            return False, f"未找到 r.in（请在输入页指定，或放在走时文件同目录）{('：' + hint) if hint else ''}"
        ok, msg = write_pois_full_to_rin(r_in, full)
        if not ok:
            return False, f"pois{which}: {msg}"
        # 写回后再读一次，确认磁盘内容
        first2, rest2 = parse_pois_from_rin(r_in)
        if first2 is None:
            return False, f"pois{which}: 写后校验失败（仍无 pois=）：{r_in}"
        return True, f"已写回 pois{which} → {r_in}（pois={first2:.3f},{rest2}）"

    def _write_pois_left_and_run(self) -> None:
        """左支：写回左支 pois → 正演 → 仅更新 OBS 左侧理论缓存。"""
        self._flush_param_edits()
        if not self.results:
            self.status_var.set("pois左支: 请先加载 tx.in")
            return
        # 先确保右侧缓存已有（避免首次改左支时右侧空白）
        wd = self.results[0].path.parent
        ent = self._ensure_t2d_side_entry(wd)
        if ent.get("right_phases") is None and (wd / "tx.out").exists():
            self._update_t2d_side_from_txout(wd, side="both")
        ok, msg = self._sync_pois_ui_to_rin(prefer="left")
        if not ok and "为空" in msg:
            return
        if not ok:
            self.status_var.set(msg)
            messagebox.showwarning("pois 写回失败", msg)
            return
        self._invalidate_theory2d_disk_cache(wd)
        fr = run_rayinvr_forward(
            wd,
            force_run=True,
            tx_in_override=self.results[0].path,
            sync_override_to_tfile=True,
        )
        if not fr.success:
            self.status_var.set(f"左支正演失败: {fr.message}")
            return
        self._update_t2d_side_from_txout(wd, side="left")
        self.redraw()
        self.status_var.set(f"{msg}；仅刷新 OBS 左侧理论点（右支缓存保留）")

    def _write_pois_right_and_run(self) -> None:
        """右支：写回右支 pois → 正演 → 仅更新 OBS 右侧理论缓存。"""
        self._flush_param_edits()
        if not self.results:
            self.status_var.set("pois右支: 请先加载 tx.in")
            return
        wd = self.results[0].path.parent
        ent = self._ensure_t2d_side_entry(wd)
        if ent.get("left_phases") is None and (wd / "tx.out").exists():
            self._update_t2d_side_from_txout(wd, side="both")
        ok, msg = self._sync_pois_ui_to_rin(prefer="right")
        if not ok and "为空" in msg:
            return
        if not ok:
            self.status_var.set(msg)
            messagebox.showwarning("pois 写回失败", msg)
            return
        self._invalidate_theory2d_disk_cache(wd)
        fr = run_rayinvr_forward(
            wd,
            force_run=True,
            tx_in_override=self.results[0].path,
            sync_override_to_tfile=True,
        )
        if not fr.success:
            self.status_var.set(f"右支正演失败: {fr.message}")
            return
        self._update_t2d_side_from_txout(wd, side="right")
        self.redraw()
        self.status_var.set(f"{msg}；仅刷新 OBS 右侧理论点（左支缓存保留）")

    @staticmethod
    def _hash_array(a: np.ndarray) -> str:
        arr = np.asarray(a, dtype=np.float64)
        h = hashlib.sha1()
        h.update(str(arr.shape).encode("utf-8"))
        h.update(arr.tobytes())
        return h.hexdigest()

    @staticmethod
    def _inv_cache_dir() -> Path:
        d = Path(tempfile.gettempdir()) / "pyaobs_iphase_cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _theory2d_cache_dir() -> Path:
        d = IPhaseBusinessMixin._inv_cache_dir() / "theory2d"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _hash_file_safe(path: Path) -> str:
        try:
            return hash_file(path)
        except Exception:
            return "error"

    def _theory2d_cache_key(self, work_dir: Path, *, phase_ppp: int, phase_pps: int) -> str:
        spec = parse_rin_input_files(work_dir)
        v = self._hash_file_safe(spec.v_file)
        r = self._hash_file_safe(spec.r_file)
        tx = self._hash_file_safe(spec.t_file)
        src = "|".join(
            [
                "v1_theory2d",
                f"ppp={int(phase_ppp)}",
                f"pps={int(phase_pps)}",
                f"vname={spec.v_file.name}",
                f"tname={spec.t_file.name}",
                f"v={v}",
                f"r={r}",
                f"tx={tx}",
            ]
        )
        return hashlib.sha1(src.encode("utf-8")).hexdigest()

    def _load_or_build_theory2d_bundle(
        self,
        r: FileResult,
        *,
        phase_ppp: int = PHASE_PPP,
        phase_pps: int = PHASE_PPS,
        auto_run: bool = True,
        force_forward: bool = False,
    ) -> tuple[Theory2DBundle | None, bool, str]:
        """
        返回 (bundle, cache_hit, message)。
        message 为可读状态：ok / 缺输入 / 正演失败 / 解析失败等。
        """
        wd = r.path.parent
        key = self._theory2d_cache_key(wd, phase_ppp=phase_ppp, phase_pps=phase_pps)
        pkl = self._theory2d_cache_dir() / f"bundle_{key}.pkl"
        tx_out = wd / "tx.out"
        use_cache = not bool(self.force_recompute.get())
        if use_cache and pkl.exists():
            try:
                with open(pkl, "rb") as f:
                    obj = pickle.load(f)
                if isinstance(obj, Theory2DBundle):
                    if tx_out.exists():
                        try:
                            left_b, right_b = collect_theory2d_delta_by_branch(
                                tx_out, phase_ppp=phase_ppp, phase_pps=phase_pps
                            )
                            self._theory2d_left_branch = left_b
                            self._theory2d_right_branch = right_b
                        except Exception:
                            pass
                    return obj, True, "cache_hit"
            except Exception:
                pass

        # 需要解析新bundle时，如果 tx.out 不可用或要求强制重算，则自动正演
        need_forward = bool(force_forward) or bool(self.force_recompute.get()) or (not tx_out.exists())
        if auto_run and need_forward:
            fr = run_rayinvr_forward(
                wd,
                force_run=bool(force_forward) or bool(self.force_recompute.get()),
                tx_in_override=r.path,
                sync_override_to_tfile=True,
            )
            if not fr.success:
                return None, False, f"{fr.code}:{fr.message}"
            tx_out = Path(fr.tx_out_path) if fr.tx_out_path else tx_out
        elif not tx_out.exists():
            return None, False, "txout_missing:未找到 tx.out"

        try:
            bundle = build_theory2d_bundle_from_txout(
                tx_out,
                phase_ppp=phase_ppp,
                phase_pps=phase_pps,
                input_hash=key,
            )
        except Exception as e:
            return None, False, f"parse_failed:{e}"

        if bundle.global_x.size < 2:
            try:
                pts = collect_phase_points_from_txout(tx_out, phase_ids=(phase_ppp, phase_pps))
                n1 = int(pts.get(int(phase_ppp), (np.asarray([]), np.asarray([]), np.asarray([])))[0].size)
                n2 = int(pts.get(int(phase_pps), (np.asarray([]), np.asarray([]), np.asarray([])))[0].size)
            except Exception:
                n1, n2 = -1, -1
            return None, False, (
                f"coverage_empty:PPP/PPS 理论点不足(ppp={int(phase_ppp)}, pps={int(phase_pps)}, "
                f"n_ppp={n1}, n_pps={n2}, diag={bundle.input_hash})"
            )

        try:
            with open(pkl, "wb") as f:
                pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
        # 首次正演后填充分支，保证导出 theory2d 时左右两支都有数据
        try:
            left_b, right_b = collect_theory2d_delta_by_branch(
                tx_out, phase_ppp=phase_ppp, phase_pps=phase_pps
            )
            self._theory2d_left_branch = left_b
            self._theory2d_right_branch = right_b
        except Exception:
            pass
        return bundle, False, "ok"

    def run_theory2d_forward_all(self) -> None:
        """对已加载文件执行 2D 正演，并刷新左右支理论缓存。"""
        self._flush_param_edits()
        if not self.results:
            if self.files:
                self._load_and_draw()
            else:
                messagebox.showwarning("提示", "请先加载 tx.in 文件。")
                return

        sync_note = ""
        if self.theory_mode.get() in ("2D", "2Dequi"):
            ok_sync, sync_msg = self._sync_pois_ui_to_rin(prefer="left")
            if ok_sync:
                sync_note = sync_msg
            else:
                sync_note = sync_msg
                if "为空" not in sync_msg:
                    messagebox.showwarning("pois 写回失败", sync_msg)

        ok_msgs: list[str] = []
        err_msgs: list[str] = []
        # 合并多站 tx.in 会拆成多个 FileResult，但 RAYINVR 仍按目录/同文件只跑一次
        seen_jobs: set[tuple[str, str]] = set()
        for r in self.results:
            wd = r.path.parent
            try:
                job_key = (str(wd.resolve()), str(r.path.resolve()))
            except Exception:
                job_key = (str(wd), str(r.path))
            if job_key in seen_jobs:
                continue
            seen_jobs.add(job_key)
            self._invalidate_theory2d_disk_cache(wd)
            try:
                spec = parse_rin_input_files(wd)
                tflag = "r.in" if spec.tfile_from_rin else "fallback"
                vflag = "r.in" if spec.vfile_from_rin else "fallback"
                inp = f"tfile={spec.t_file.name}({tflag}), vfile={spec.v_file.name}({vflag})"
            except Exception as e:
                err_msgs.append(f"{r.path.name}: 解析r.in失败:{e}")
                continue

            fr = run_rayinvr_forward(
                wd,
                force_run=True,
                tx_in_override=r.path,
                sync_override_to_tfile=True,
            )
            if not fr.success:
                err_msgs.append(f"{r.path.name}: {fr.code}:{fr.message} [{inp}]")
                continue
            tx_out = wd / "tx.out"
            if not tx_out.exists():
                err_msgs.append(f"{r.path.name}: tx.out未生成 [{inp}]")
                continue
            try:
                phase_cnt: dict[int, int] = {}
                with open(tx_out, "r", encoding="utf-8", errors="ignore") as _fo:
                    for _ln in _fo:
                        _raw = _ln.rstrip("\n\r")
                        _ipf = None
                        if len(_raw) >= 40:
                            try:
                                _ipf = int(_raw[30:40])
                            except Exception:
                                pass
                        if _ipf is None:
                            _pts = _raw.split()
                            if len(_pts) >= 4:
                                try:
                                    _ipf = int(_pts[3])
                                except Exception:
                                    pass
                        if _ipf is not None and _ipf > 0:
                            phase_cnt[_ipf] = phase_cnt.get(_ipf, 0) + 1
                counts_str = ", ".join(f"phase{k}={v}" for k, v in sorted(phase_cnt.items()))
            except Exception as e:
                err_msgs.append(f"{r.path.name}: tx.out读取失败:{e} [{inp}]")
                continue
            ok_msgs.append(f"{r.path.name}: {counts_str} [{inp}]")
            self._update_t2d_side_from_txout(wd, side="both")

        info_lines = [f"2D正演完成: 成功{len(ok_msgs)} 失败{len(err_msgs)}"]
        if sync_note:
            info_lines.insert(1, sync_note)
        info_lines.extend(ok_msgs[:5])
        info_lines.extend(err_msgs[:5])
        self.theory2d_notice = ""
        self.redraw()
        try:
            self.canvas.draw()
        except Exception:
            pass
        self.status_var.set("；".join(info_lines[:4]))
        # 延后弹窗，避开与 matplotlib 重绘同一帧的 Wayland 尺寸协商
        summary = info_lines[0]
        detail = "\n".join(info_lines)
        icon = (
            QMessageBox.Icon.Warning if err_msgs else QMessageBox.Icon.Information
        )
        show_modeless_message(
            "2D正演结果",
            summary,
            detailed=detail,
            icon=icon,
            defer_ms=50,
        )

    def _cached_invert_local(
        self,
        model_x: np.ndarray,
        true_off: np.ndarray,
        t_obs: np.ndarray,
        p_est: np.ndarray,
        *,
        vp: float,
        h0: float,
        ratio0: float,
    ) -> tuple[object, bool]:
        window_km = 0.4
        # 提高对深度(h)敏感性：
        # - 放松 h 平滑（更易响应局部误差）
        # - 收紧 Vp/Vs 平滑（减少 r 与 h 的互相替代）
        smooth_h = 0.001
        smooth_r = 0.002
        ratio_lo = max(1.05, float(ratio0) - 0.35)
        ratio_hi = min(4.0, float(ratio0) + 0.35)
        if ratio_hi <= ratio_lo + 1e-6:
            ratio_lo, ratio_hi = 1.05, 4.0
        key_src = "|".join(
            [
                "v2_local",
                f"vp={vp:.6f}",
                f"h0={h0:.6f}",
                f"r0={ratio0:.6f}",
                f"win={self._window_points_value()}",
                f"wkm={window_km:.3f}",
                f"slh={smooth_h:.4f}",
                f"slr={smooth_r:.4f}",
                self._hash_array(model_x),
                self._hash_array(true_off),
                self._hash_array(t_obs),
                self._hash_array(p_est),
            ]
        )
        key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()
        pkl = self._inv_cache_dir() / f"local_{key}.pkl"
        use_cache = not bool(self.force_recompute.get())
        if use_cache and pkl.exists():
            try:
                with open(pkl, "rb") as f:
                    return pickle.load(f), True
            except Exception:
                pass
        res = invert_local_h_vpratio(
            model_x,
            true_off,
            t_obs,
            p_est,
            vp_fixed=vp,
            h0=h0,
            vpratio0=ratio0,
            window_km=window_km,
            min_points=3,
            ratio_bounds=(ratio_lo, ratio_hi),
            smooth_lambda_h=smooth_h,
            smooth_lambda_r=smooth_r,
        )
        try:
            with open(pkl, "wb") as f:
                pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
        return res, False

    def _cached_invert_local_split(
        self,
        model_x: np.ndarray,
        true_off: np.ndarray,
        t_obs: np.ndarray,
        p_est: np.ndarray,
        *,
        vp: float,
        h0: float,
        ratio0: float,
    ) -> tuple[LocalInversionResult, int, int]:
        """
        左右支（offset<0 / >0）分别反演，再合并结果。
        返回: (合并结果, cache_hits, cache_total)
        """
        n = len(model_x)
        pred = np.full(n, np.nan, dtype=float)
        corr = np.full(n, np.nan, dtype=float)
        xpk = np.full(n, np.nan, dtype=float)
        hpk = np.full(n, np.nan, dtype=float)
        rpk = np.full(n, np.nan, dtype=float)
        vmask = np.zeros(n, dtype=bool)

        xc_list: list[np.ndarray] = []
        hh_list: list[np.ndarray] = []
        rr_list: list[np.ndarray] = []
        nn_list: list[np.ndarray] = []
        rm_list: list[np.ndarray] = []

        cache_hits = 0
        cache_total = 0

        for branch in ("neg", "pos"):
            if branch == "neg":
                m = true_off < 0
            else:
                m = true_off > 0
            if np.sum(m) < 4:
                continue
            cache_total += 1
            res, hit = self._cached_invert_local(
                model_x[m],
                true_off[m],
                t_obs[m],
                p_est[m],
                vp=vp,
                h0=h0,
                ratio0=ratio0,
            )
            if hit:
                cache_hits += 1
            pred[m] = res.pred_dt
            corr[m] = res.corrected_dt
            xpk[m] = res.x_conv_pick
            hpk[m] = res.h_pick
            rpk[m] = res.vpratio_pick
            vmask[m] = res.valid_mask
            if res.x_conv.size > 0:
                xc_list.append(np.asarray(res.x_conv, dtype=float))
                hh_list.append(np.asarray(res.h_conv, dtype=float))
                rr_list.append(np.asarray(res.vpratio_conv, dtype=float))
                nn_list.append(np.asarray(res.npts_conv, dtype=int))
                rm_list.append(np.asarray(res.rms_conv, dtype=float))

        # 0 偏移点单独走一次（可选）
        m0 = true_off == 0
        if np.any(m0) and np.sum(m0) >= 3:
            cache_total += 1
            res0, hit0 = self._cached_invert_local(
                model_x[m0],
                true_off[m0],
                t_obs[m0],
                p_est[m0],
                vp=vp,
                h0=h0,
                ratio0=ratio0,
            )
            if hit0:
                cache_hits += 1
            pred[m0] = res0.pred_dt
            corr[m0] = res0.corrected_dt
            xpk[m0] = res0.x_conv_pick
            hpk[m0] = res0.h_pick
            rpk[m0] = res0.vpratio_pick
            vmask[m0] = res0.valid_mask
            if res0.x_conv.size > 0:
                xc_list.append(np.asarray(res0.x_conv, dtype=float))
                hh_list.append(np.asarray(res0.h_conv, dtype=float))
                rr_list.append(np.asarray(res0.vpratio_conv, dtype=float))
                nn_list.append(np.asarray(res0.npts_conv, dtype=int))
                rm_list.append(np.asarray(res0.rms_conv, dtype=float))

        if xc_list:
            xc = np.concatenate(xc_list)
            hh = np.concatenate(hh_list)
            rr = np.concatenate(rr_list)
            nn = np.concatenate(nn_list)
            rms = np.concatenate(rm_list)
            so = np.argsort(xc)
            xc, hh, rr, nn, rms = xc[so], hh[so], rr[so], nn[so], rms[so]
        else:
            xc = np.asarray([], dtype=float)
            hh = np.asarray([], dtype=float)
            rr = np.asarray([], dtype=float)
            nn = np.asarray([], dtype=int)
            rms = np.asarray([], dtype=float)

        merged = LocalInversionResult(
            x_conv=xc,
            h_conv=hh,
            vpratio_conv=rr,
            npts_conv=nn,
            rms_conv=rms,
            pred_dt=pred,
            corrected_dt=corr,
            x_conv_pick=xpk,
            h_pick=hpk,
            vpratio_pick=rpk,
            valid_mask=vmask,
        )
        return merged, cache_hits, cache_total

    def _build_diff_pairs_with_controlled_interp(
        self,
        r: FileResult,
        *,
        min_src_points: int = 6,
        max_gap: float = 2.5,
        min_cover_ratio: float = 0.6,
    ) -> list[tuple[float, float, float]]:
        """
        构建用于反演的 PPS-PPP 差值：
        - 优先使用严格配对 diff_pairs；
        - 对未严格配到的情况，在 PSS 偏移距上对 PPP/PPS 受控插值后补充差值。
        """
        strict = list(r.diff_pairs) if r.diff_pairs else []
        ppp_x, ppp_off, ppp_t = _phase_model_trueoff_time(r.ds_ppp, PHASE_PPP)
        pps_x, pps_off, pps_t = _phase_model_trueoff_time(r.ds_pps, PHASE_PPS)
        pss_x, pss_off, _ = _phase_model_trueoff_time(r.ds_pss, PHASE_PSS)
        if ppp_off.size < min_src_points or pps_off.size < min_src_points or pss_off.size < 3:
            return strict

        interp_pairs: list[tuple[float, float, float]] = []
        for sign in (-1, 1):
            if sign < 0:
                mppp = ppp_off < 0
                mpps = pps_off < 0
                mtgt = pss_off < 0
            else:
                mppp = ppp_off > 0
                mpps = pps_off > 0
                mtgt = pss_off > 0
            if np.sum(mppp) < min_src_points or np.sum(mpps) < min_src_points or np.sum(mtgt) < 3:
                continue

            src1 = ppp_off[mppp]
            src2 = pps_off[mpps]
            tgt_off = pss_off[mtgt]
            tgt_x = pss_x[mtgt]
            ov_min = max(float(np.min(src1)), float(np.min(src2)))
            ov_max = min(float(np.max(src1)), float(np.max(src2)))
            inside = (tgt_off >= ov_min) & (tgt_off <= ov_max)
            if np.sum(inside) < 2:
                continue
            tgt_off_i = tgt_off[inside]
            tgt_x_i = tgt_x[inside]
            cover_ratio = float(np.sum(inside)) / float(np.sum(mtgt))
            if cover_ratio < min_cover_ratio:
                continue

            # 距源点太远的目标点丢弃（避免跨大间隙插值）
            d1 = np.min(np.abs(tgt_off_i[:, None] - src1[None, :]), axis=1)
            d2 = np.min(np.abs(tgt_off_i[:, None] - src2[None, :]), axis=1)
            s1 = np.sort(src1)
            s2 = np.sort(src2)
            if s1.size >= 2:
                g1_med = float(np.median(np.diff(s1)))
            else:
                g1_med = max_gap
            if s2.size >= 2:
                g2_med = float(np.median(np.diff(s2)))
            else:
                g2_med = max_gap
            gap1_lim = max(max_gap, 2.5 * g1_med)
            gap2_lim = max(max_gap, 2.5 * g2_med)
            near = (d1 <= gap1_lim) & (d2 <= gap2_lim)
            if np.sum(near) < 2:
                continue
            tgt_off_i = tgt_off_i[near]
            tgt_x_i = tgt_x_i[near]

            ppp_fit = fit_ppp_time_curve_local_linear(
                src1,
                ppp_t[mppp],
                tgt_off_i,
                window_points=self._window_points_value(),
                split_by_sign=False,
            )
            pps_fit = fit_ppp_time_curve_local_linear(
                src2,
                pps_t[mpps],
                tgt_off_i,
                window_points=self._window_points_value(),
                split_by_sign=False,
            )
            v = np.isfinite(ppp_fit) & np.isfinite(pps_fit)
            if not np.any(v):
                continue
            dt = pps_fit[v] - ppp_fit[v]
            for xx, oo, dd in zip(tgt_x_i[v], tgt_off_i[v], dt):
                if np.isfinite(dd):
                    interp_pairs.append((float(xx), float(oo), float(dd)))

        # 合并 strict + interp，strict 优先
        out_map: dict[tuple[float, float], tuple[float, float, float]] = {}
        for md, to, dt in strict:
            if np.isfinite(md) and np.isfinite(to) and np.isfinite(dt):
                out_map[(round(float(md), 3), round(float(to), 3))] = (float(md), float(to), float(dt))
        for md, to, dt in interp_pairs:
            k = (round(float(md), 3), round(float(to), 3))
            if k not in out_map:
                out_map[k] = (float(md), float(to), float(dt))

        out = list(out_map.values())
        if out:
            out.sort(key=lambda t: t[0])
        # 统一将反演使用的 t_obs 转为“拟合后的 PPS-PPP 曲线值”：
        # - 按左右偏移距分支分别做 LocalLinear 拟合；
        # - 拟合失败处回退原始值，避免有效点丢失。
        arr_out = np.asarray(out, dtype=float)
        if arr_out.size == 0:
            return out
        md_out = arr_out[:, 0]
        to_out = arr_out[:, 1]
        dt_out = arr_out[:, 2]
        dt_fit = np.full_like(dt_out, np.nan, dtype=float)

        for sgn in (-1.0, 1.0):
            m = (to_out * sgn) > 0.0
            if np.count_nonzero(m) < 3:
                continue
            yf = fit_ppp_time_curve_local_linear(
                md_out[m],
                dt_out[m],
                md_out[m],
                window_points=self._window_points_value(),
                split_by_sign=False,
            )
            dt_fit[m] = yf

        # 0 偏移点（若存在）用全体再拟合一次
        mz = to_out == 0.0
        if np.any(mz) and np.count_nonzero(np.isfinite(md_out) & np.isfinite(dt_out)) >= 3:
            yz = fit_ppp_time_curve_local_linear(
                md_out,
                dt_out,
                md_out[mz],
                window_points=self._window_points_value(),
                split_by_sign=False,
            )
            dt_fit[mz] = yz

        dt_use = np.where(np.isfinite(dt_fit), dt_fit, dt_out)
        out_fit: list[tuple[float, float, float]] = []
        for xx, oo, dd in zip(md_out, to_out, dt_use):
            if np.isfinite(xx) and np.isfinite(oo) and np.isfinite(dd):
                out_fit.append((float(xx), float(oo), float(dd)))
        return out_fit

    def _cached_invert_mfield(
        self,
        xrec: np.ndarray,
        zrec: np.ndarray,
        xconv: np.ndarray,
        zconv: np.ndarray,
        dt: np.ndarray,
        *,
        vp: float,
    ) -> tuple[object | None, bool]:
        key_src = "|".join(
            [
                "v2_mfield",
                f"vp={vp:.6f}",
                self._hash_array(xrec),
                self._hash_array(zrec),
                self._hash_array(xconv),
                self._hash_array(zconv),
                self._hash_array(dt),
            ]
        )
        key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()
        pkl = self._inv_cache_dir() / f"mfield_{key}.pkl"
        use_cache = not bool(self.force_recompute.get())
        if use_cache and pkl.exists():
            try:
                with open(pkl, "rb") as f:
                    return pickle.load(f), True
            except Exception:
                pass
        try:
            res = invert_m_field_from_paths(
                xrec,
                zrec,
                xconv,
                zconv,
                dt,
                vp_fixed=vp,
                nx=90,
                nz=48,
                smooth_lambda_x=0.35,
                smooth_lambda_z=0.35,
                damp=1e-4,
                samples_per_ray=60,
            )
        except Exception:
            return None, False
        try:
            with open(pkl, "wb") as f:
                pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
        return res, False

    def save_figure(self) -> None:
        """导出当前图像到文件（jpg/pdf/ps/tif/png）。"""
        if self.fig is None:
            messagebox.showwarning("提示", "当前没有可保存的图像。")
            return

        type_var = _Var("PNG")
        p = filedialog.asksaveasfilename(
            title="保存图像",
            defaultextension="",
            filetypes=[
                ("JPEG", "*.jpg;*.jpeg"),
                ("PDF", "*.pdf"),
                ("PostScript", "*.ps"),
                ("TIFF", "*.tif;*.tiff"),
                ("PNG", "*.png"),
                ("All files", "*.*"),
            ],
            typevariable=type_var,
        )
        if not p:
            return

        try:
            out = Path(p)
            if out.suffix == "":
                ext_map = {
                    "JPEG": ".jpg",
                    "PDF": ".pdf",
                    "PostScript": ".ps",
                    "TIFF": ".tif",
                    "PNG": ".png",
                }
                out = out.with_suffix(ext_map.get(type_var.get(), ".png"))

            fmt = out.suffix.lower().lstrip(".")
            if fmt == "jpg":
                fmt = "jpeg"
            if fmt in ("ps", "eps"):
                # PostScript 不支持透明度：临时将 alpha 设为 1，保存后恢复
                changed = self._set_figure_alpha(1.0)
                self.fig.savefig(out, dpi=300, bbox_inches="tight", format=fmt)
                self._restore_figure_alpha(changed)
            else:
                # 某些环境缺少中文字体，忽略 glyph warning（图中文字已尽量改为英文）
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Glyph .* missing from current font")
                    self.fig.savefig(out, dpi=300, bbox_inches="tight", format=fmt)
            self.status_var.set(f"图像已保存: {out}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def _set_figure_alpha(self, alpha_value: float) -> list[tuple[object, float | None]]:
        """将图中 artist 的 alpha 临时设为固定值，返回可恢复列表。"""
        changed: list[tuple[object, float | None]] = []
        for artist in self.fig.findobj():
            if not hasattr(artist, "get_alpha") or not hasattr(artist, "set_alpha"):
                continue
            try:
                old = artist.get_alpha()
            except Exception:
                continue
            if old is None:
                continue
            try:
                artist.set_alpha(alpha_value)
                changed.append((artist, old))
            except Exception:
                pass
        return changed

    @staticmethod
    def _restore_figure_alpha(changed: list[tuple[object, float | None]]) -> None:
        """恢复 _set_figure_alpha 改动。"""
        for artist, old in changed:
            try:
                artist.set_alpha(old)
            except Exception:
                pass

    def show_export_tx_select(self) -> None:
        """走时图：打开 tomo2d 风格的 OBS/震相勾选预览，并可导出筛选 tx.in。"""
        if not self.results:
            messagebox.showwarning("提示", "请先加载 tx.in 文件。")
            return
        stations = _stations_from_depth_arrays(
            getattr(self, "shot_depth_x", None),
            getattr(self, "shot_depth_z", None),
            getattr(self, "shot_depth_ids", None),
        )
        picks = build_preview_picks_from_results(self.results, stations=stations or None)
        if not picks:
            messagebox.showwarning("提示", "当前结果中没有可预览的拾取点。")
            return
        existing = getattr(self, "_export_tx_select_dlg", None)
        if existing is not None:
            try:
                if existing.isVisible():
                    existing.set_picks(picks)
                    existing.raise_()
                    existing.activateWindow()
                    return
            except RuntimeError:
                self._export_tx_select_dlg = None
        dlg = ExportTxSelectDialog(None, picks=picks)
        dlg.export_requested.connect(self._export_tx_by_obs_phases)
        self._export_tx_select_dlg = dlg
        show_modeless_dialog(dlg, activate=True)

    def _collect_export_tx_choices(self) -> tuple[list[tuple[str, str]], list[int]]:
        """返回 ([(obs_key, label), ...], sorted phase ids)（按台站 x 合并）。"""
        stations = _stations_from_depth_arrays(
            getattr(self, "shot_depth_x", None),
            getattr(self, "shot_depth_z", None),
            getattr(self, "shot_depth_ids", None),
        )
        picks = build_preview_picks_from_results(self.results, stations=stations or None)
        from .dialogs.export_tx_select import build_obs_records

        obs_items = [(str(r["key"]), str(r["label"])) for r in build_obs_records(picks)]
        phase_set = {int(p.phase_id) for p in picks}
        return obs_items, sorted(phase_set)

    def _export_tx_by_obs_phases(self, obs_keys: list, phase_ids: list) -> None:
        """按所选台站 xobs + 震相写筛选后的 tx.in（每个台站一个文件）。"""
        if not obs_keys:
            messagebox.showwarning("导出", "请至少选择一个 OBS。")
            return
        if not phase_ids:
            messagebox.showwarning("导出", "请至少选择一个震相号。")
            return
        phase_ids_i = [int(p) for p in phase_ids]
        # obs_key 现为台站模型距离字符串，如 "30.818"
        try:
            xobs_set = {round(float(k), 3) for k in obs_keys}
        except Exception:
            messagebox.showwarning("导出", "OBS 选择无效。")
            return

        out_dir: Path | None = None
        project = getattr(self, "project", None)
        workdir = getattr(project, "workdir", None) if project is not None else None
        if workdir:
            out_dir = Path(workdir) / OUTPUTS_DIR
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                out_dir = None
        if out_dir is None:
            start = str(self.results[0].path.parent) if self.results else ""
            chosen = QFileDialog.getExistingDirectory(None, "选择导出目录", start)
            if not chosen:
                return
            out_dir = Path(chosen)

        stations = _stations_from_depth_arrays(
            getattr(self, "shot_depth_x", None),
            getattr(self, "shot_depth_z", None),
            getattr(self, "shot_depth_ids", None),
        )

        ok_msgs: list[str] = []
        err_msgs: list[str] = []
        for xobs in sorted(xobs_set):
            out_ds = PhaseDataset()
            for r in self.results:
                for shot in r.ds.shots:
                    if round(float(shot.xshot), 3) != xobs:
                        continue
                    kept = [p for p in shot.picks if int(p.phase_id) in phase_ids_i]
                    if not kept:
                        continue
                    new_shot = Shot(xshot=shot.xshot, tshot=shot.tshot, ushot=shot.ushot)
                    new_shot.picks = list(kept)
                    out_ds.shots.append(new_shot)
            if out_ds.n_picks == 0:
                err_msgs.append(f"x={xobs:.3f}: 所选震相无拾取点")
                continue
            oid = None
            if stations:
                from .dialogs.export_tx_select import _nearest_obs_id

                oid = _nearest_obs_id(xobs, stations, tol=0.001)
            if oid is not None:
                out_name = f"tx_OBS{int(oid)}_sel.in"
            else:
                out_name = f"tx_x{xobs:.3f}_sel.in"
            out_path = out_dir / out_name
            try:
                write_tx(out_ds, out_path)
                ok_msgs.append(
                    f"x={xobs:.3f} -> {out_path.name} "
                    f"(nshot={out_ds.n_shots}, npick={out_ds.n_picks})"
                )
            except Exception as e:
                err_msgs.append(f"x={xobs:.3f}: {e}")

        msg = ""
        if ok_msgs:
            msg += f"导出目录: {out_dir}\n震相: {phase_ids_i}\n\n成功:\n" + "\n".join(ok_msgs)
        if err_msgs:
            if msg:
                msg += "\n\n"
            msg += "失败:\n" + "\n".join(err_msgs)
        if err_msgs:
            messagebox.showwarning("筛选导出结果", msg or "无结果")
        else:
            messagebox.showinfo("筛选导出结果", msg or "无结果")
            self.status_var.set(f"已导出 {len(ok_msgs)} 个筛选 tx.in → {out_dir}")

    def export_psp_files(self) -> None:
        """
        导出每个 OBS 的 PSP 到新文件:
        - picked: 用实测 PPS-PPP 差值（model distance 插值）计算
        - theory2d: 仅用 tx.out 理论 PSS-PSP（不足则该点不导出）
        - theory_pss: 用 PSS 斜率 + 反演 h/VpVs 剖面计算理论 PSS-PSP
        - theory2Dequi: 基于 tx_2Dequiv.in 正演 tx.out；仅用 PSS-PSP（不足则该点不导出）
        文件名: tx_OBS??_PSP_<导出模式>.in（theory2Dequi 固定为 tx_theory2Dequi.in）
        """
        if not self.results:
            messagebox.showwarning("提示", "请先加载 tx.in 文件。")
            return

        mode = self.psp_export_mode.get().strip().lower()
        ok_msgs: list[str] = []
        err_msgs: list[str] = []

        for r in self.results:
            if mode == "theory2dequi":
                out_path = r.path.parent / "tx_theory2Dequi.in"
            else:
                obs_tag_num = _obs_tag_from_result(r)
                if obs_tag_num.isdigit():
                    out_name = f"tx_OBS{obs_tag_num}_PSP_{mode}.in"
                elif r.obs_x is not None:
                    out_name = f"tx_x{round(float(r.obs_x), 3):.3f}_PSP_{mode}.in"
                else:
                    out_name = f"tx_{r.path.stem}_PSP_{mode}.in"
                out_path = r.path.with_name(out_name)

            try:
                out_ds = self._build_psp_dataset(r, mode=mode)
                if out_ds.n_picks == 0:
                    err_msgs.append(f"{_result_display_name(r)}: 没有可导出的 PSP 点")
                    continue
                write_tx(out_ds, out_path)
                ok_msgs.append(f"{_result_display_name(r)} -> {out_path.name} ({out_ds.n_picks} picks)")
            except Exception as e:
                err_msgs.append(f"{_result_display_name(r)}: {e}")

        msg = ""
        if ok_msgs:
            msg += "导出成功:\n" + "\n".join(ok_msgs)
        if err_msgs:
            if msg:
                msg += "\n\n"
            msg += "导出失败:\n" + "\n".join(err_msgs)

        if err_msgs:
            messagebox.showwarning("PSP导出结果", msg)
        else:
            messagebox.showinfo("PSP导出结果", msg or "无输出")
        self.status_var.set(f"PSP导出模式={mode}；成功{len(ok_msgs)}，失败{len(err_msgs)}")

    def show_local_inversion_result(self) -> None:
        """
        第一版局部反演：
        1) 由 PPP 局部斜率估计 p；
        2) 近似计算转换点位置 x_c；
        3) 固定 Vp，滑窗反演 h 与 Vp/Vs；
        4) 绘制校正前后 PPS-PPP 并输出 x_c-h-ratio 表。
        """
        if not self.results:
            messagebox.showwarning("提示", "请先加载 tx.in 文件。")
            return

        try:
            vp = float(self.vp_cr.get())
            vs = float(self.vs_cr.get())
            h0 = float(self.h_cr.get())
        except Exception:
            messagebox.showwarning("参数错误", "请检查 h_cr/Vp/Vs 输入。")
            return
        if vp <= 0 or vs <= 0 or h0 <= 0:
            messagebox.showwarning("参数错误", "h_cr/Vp/Vs 必须为正数。")
            return

        ratio0 = max(1.05, vp / max(vs, 1e-6))
        multi_mode = len(self.results) > 1

        win = PlotToolWindow(title="局部反演校正结果", parent=None)
        fig2 = Figure(figsize=(12.0, 8.0), dpi=100)
        ax_a = fig2.add_subplot(2, 2, 1)
        ax_b = fig2.add_subplot(2, 2, 2)
        ax_c = fig2.add_subplot(2, 2, 3)
        ax_d = fig2.add_subplot(2, 2, 4)

        ok_files = 0
        export_msgs: list[str] = []
        x2d_all: list[float] = []
        h2d_all: list[float] = []
        r2d_all: list[float] = []
        xres_all: list[float] = []
        zres_all: list[float] = []
        eres_all: list[float] = []
        xconv_pps_all: list[float] = []
        zconv_pps_all: list[float] = []
        xconv_pss_all: list[float] = []
        zconv_pss_all: list[float] = []
        xrec_all: list[float] = []
        rec_marks: list[tuple[float, str]] = []
        xrec_path_all: list[float] = []
        zrec_path_all: list[float] = []
        xconv_path_all: list[float] = []
        zconv_path_all: list[float] = []
        h_path_all: list[float] = []
        dt_path_all: list[float] = []
        pick_tables: list[tuple[Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        cache_hits_local = 0
        cache_total_local = 0
        cache_hit_mfield = False
        near_compare_msgs: list[str] = []
        # 线型/颜色与主窗口保持一致
        c_pps_inv = "#FF7F00"
        c_pps_diag = "#32CD32"
        c_psspsp_inv = "#FF00FF"
        inv_fit_stats: list[tuple[str, str, dict[str, float]]] = []

        for i, r in enumerate(self.results):
            diff_use = self._build_diff_pairs_with_controlled_interp(r)
            if not diff_use:
                continue
            arr = np.asarray(diff_use, dtype=float)
            model_x = arr[:, 0]
            true_off = arr[:, 1]
            t_obs = arr[:, 2]

            ppp_offsets, ppp_times = _phase_true_offset_time(r.ds_ppp, PHASE_PPP)
            if ppp_offsets.size < 4:
                continue

            try:
                _, p_est = pps_minus_ppp_from_ppp_slope(
                    ppp_offsets,
                    ppp_times,
                    true_off,
                    h_cr=h0,
                    vp=vp,
                    vs=vs,
                    window_points=self._window_points_value(),
                    split_by_sign=True,
                    smooth_dense_half_win=self._smooth_dense_half_win_value(),
                )
            except Exception:
                continue

            inv, hit_local, total_local = self._cached_invert_local_split(
                model_x,
                true_off,
                t_obs,
                p_est,
                vp=vp,
                h0=h0,
                ratio0=ratio0,
            )
            cache_hits_local += int(hit_local)
            cache_total_local += int(total_local)
            # 接收点位置（x_rec = model_x - true_offset），用于二维图标注
            rec_x = model_x - true_off
            vv_rec = np.isfinite(rec_x)
            if np.any(vv_rec):
                xrec_all.extend(rec_x[vv_rec].tolist())
            x_obs = _obs_model_distance_from_tx(r)
            if x_obs is not None:
                rec_marks.append((float(x_obs), _obs_tag_from_result(r)))
            if (
                self.seafloor_x is not None
                and self.seafloor_z is not None
                and self.seafloor_x.size >= 2
                and self.seafloor_z.size >= 2
            ):
                z_rec = np.interp(
                    rec_x,
                    self.seafloor_x,
                    self.seafloor_z,
                    left=float(self.seafloor_z[0]),
                    right=float(self.seafloor_z[-1]),
                )
            else:
                z_rec = np.zeros_like(rec_x)

            x_plot = model_x
            c = f"C{i % 10}"
            lab = _result_display_name(r)

            v_obs = np.isfinite(t_obs)
            v_inv = np.isfinite(inv.pred_dt)
            if np.any(v_obs):
                ax_a.scatter(x_plot[v_obs], t_obs[v_obs], s=14, alpha=0.45, c=c, label=f"{lab} obs(PPS-PPP)")
            if np.any(v_inv):
                so = np.argsort(x_plot[v_inv])
                ax_a.plot(
                    x_plot[v_inv][so],
                    inv.pred_dt[v_inv][so],
                    "-",
                    color=c_pps_inv,
                    linewidth=1.6,
                    alpha=0.95,
                    label=f"{lab} inv(PPS-PPP)",
                )
                err_inv = (t_obs[v_inv] - inv.pred_dt[v_inv]).astype(float)
                ae_inv = np.abs(err_inv)
                inv_fit_stats.append(
                    (
                        _obs_tag_from_result(r),
                        c,
                        {
                            "rms": float(np.sqrt(np.mean(err_inv ** 2))),
                            "abs_median": float(np.median(ae_inv)),
                            "abs_max": float(np.max(ae_inv)),
                        },
                    )
                )
            # 对比：用 PPS 斜率估计 p 得到的 PPS-PPP
            pps_off_a, pps_t_a = _phase_true_offset_time(r.ds_pps, PHASE_PPS)
            if pps_off_a.size >= 4:
                try:
                    dt_pps_diag, _ = pps_minus_ppp_from_ppp_slope(
                        pps_off_a,
                        pps_t_a,
                        true_off,
                        h_cr=float(self.h_cr.get()),
                        vp=float(self.vp_cr.get()),
                        vs=float(self.vs_cr.get()),
                        window_points=self._window_points_value(),
                        split_by_sign=True,
                        smooth_dense_half_win=self._smooth_dense_half_win_value(),
                    )
                    vv_diag = np.isfinite(x_plot) & np.isfinite(dt_pps_diag)
                    if np.any(vv_diag):
                        so_d = np.argsort(x_plot[vv_diag])
                        ax_a.plot(
                            x_plot[vv_diag][so_d],
                            dt_pps_diag[vv_diag][so_d],
                            "--",
                            color=c_pps_diag,
                            linewidth=1.3,
                            alpha=0.9,
                            label=f"{lab} inv(PPS-PPP,p from PPS)",
                        )
                except Exception:
                    pass
            # 加入 PSS-PSP 时差（用当前反演剖面）
            xm_pss_a, off_pss_a, t_pss_a = _phase_model_trueoff_time(r.ds_pss, PHASE_PSS)
            if xm_pss_a.size >= 4:
                try:
                    dt_pss_a, xconv_pss_a, _, _, _ = pss_minus_psp_from_pss_slope_with_profile(
                        xm_pss_a,
                        off_pss_a,
                        t_pss_a,
                        vp=float(self.vp_cr.get()),
                        h_profile_x=inv.x_conv if inv.x_conv.size > 1 else None,
                        h_profile=inv.h_conv if inv.h_conv.size > 1 else None,
                        vpratio_profile=inv.vpratio_conv if inv.vpratio_conv.size > 1 else None,
                        h_default=float(self.h_cr.get()),
                        vpratio_default=float(self.vp_cr.get()) / max(float(self.vs_cr.get()), 1e-6),
                        window_points=self._window_points_value(),
                        split_by_sign=True,
                        n_iter=2,
                        smooth_dense_half_win=self._smooth_dense_half_win_value(),
                    )
                    vv_pss = np.isfinite(xm_pss_a) & np.isfinite(dt_pss_a)
                    if np.any(vv_pss):
                        so2 = np.argsort(xm_pss_a[vv_pss])
                        ax_a.plot(
                            xm_pss_a[vv_pss][so2],
                            dt_pss_a[vv_pss][so2],
                            "--",
                            color=c_psspsp_inv,
                            linewidth=1.4,
                            alpha=0.95,
                            label=f"{lab} inv(PSS-PSP)",
                        )
                    # 近邻点对比：PSS-PSP 转换点 vs PPS-PPP 转换点
                    v_pps = np.isfinite(inv.x_conv_pick) & np.isfinite(inv.pred_dt)
                    v_pss = np.isfinite(xconv_pss_a) & np.isfinite(dt_pss_a)
                    if np.any(v_pps) and np.any(v_pss):
                        x_pps = np.asarray(inv.x_conv_pick[v_pps], dtype=float)
                        dt_pps = np.asarray(inv.pred_dt[v_pps], dtype=float)
                        md_pps = np.asarray(model_x[v_pps], dtype=float)
                        x_pss = np.asarray(xconv_pss_a[v_pss], dtype=float)
                        dt_pss = np.asarray(dt_pss_a[v_pss], dtype=float)
                        md_pss = np.asarray(xm_pss_a[v_pss], dtype=float)

                        rows = []
                        for j in range(x_pss.size):
                            k = int(np.argmin(np.abs(x_pps - x_pss[j])))
                            dx = float(abs(x_pps[k] - x_pss[j]))
                            rows.append(
                                (
                                    dx,
                                    float(x_pss[j]),
                                    float(md_pss[j]),
                                    float(dt_pss[j]),
                                    float(x_pps[k]),
                                    float(md_pps[k]),
                                    float(dt_pps[k]),
                                    float(dt_pss[j] - dt_pps[k]),
                                )
                            )
                        rows.sort(key=lambda t: t[0])
                        # 取最接近的若干点（仅弹窗摘要，不再自动写 CSV）
                        topn = min(12, len(rows))
                        rows_top = rows[:topn]
                        if rows_top:
                            head = rows_top[:3]
                            txt = "; ".join(
                                [
                                    f"dx={a:.3f}km, md_pss={mds:.3f}, md_ppp={mdp:.3f}, dt_pss={dpss:.4f}, dt_pps={dppp:.4f}, d={dd:.4f}"
                                    for a, _xpss, mds, dpss, _xpps, mdp, dppp, dd in head
                                ]
                            )
                            near_compare_msgs.append(f"{r.path.name}: {txt}")
                except Exception:
                    pass

            # 记录该文件的反演剖面，供主图 PSS->PSP 使用
            self.pss_profile_by_file[_result_cache_key(r)] = inv
            if inv.x_conv.size > 0 and np.any(np.isfinite(inv.x_conv)):
                ax_b.plot(inv.x_conv, inv.h_conv, "-", color=c, linewidth=1.6, alpha=0.95, label=f"{lab} h")
                ax_b.plot(inv.x_conv, inv.vpratio_conv, "--", color=c, linewidth=1.2, alpha=0.9, label=f"{lab} Vp/Vs")

                # 用逐点结果构建二维剖面散点集（1D 反演不再自动输出 CSV）
                vv2 = np.isfinite(inv.x_conv_pick) & np.isfinite(inv.h_pick) & np.isfinite(inv.vpratio_pick)
                if np.any(vv2):
                    x2d_all.extend(inv.x_conv_pick[vv2].tolist())
                    h2d_all.extend(inv.h_pick[vv2].tolist())
                    r2d_all.extend(inv.vpratio_pick[vv2].tolist())
                # 为 m(x,z) 反演收集射线路径
                vv_path = (
                    np.isfinite(rec_x)
                    & np.isfinite(z_rec)
                    & np.isfinite(inv.x_conv_pick)
                    & np.isfinite(inv.h_pick)
                    & np.isfinite(t_obs)
                )
                if np.any(vv_path):
                    n_prev = len(dt_path_all)
                    xrec_path_all.extend(rec_x[vv_path].tolist())
                    zrec_path_all.extend(z_rec[vv_path].tolist())
                    xconv_path_all.extend(inv.x_conv_pick[vv_path].tolist())
                    if (
                        self.seafloor_x is not None
                        and self.seafloor_z is not None
                        and self.seafloor_x.size >= 2
                        and self.seafloor_z.size >= 2
                    ):
                        ztop_c = np.interp(
                            inv.x_conv_pick[vv_path],
                            self.seafloor_x,
                            self.seafloor_z,
                            left=float(self.seafloor_z[0]),
                            right=float(self.seafloor_z[-1]),
                        )
                    else:
                        ztop_c = np.zeros(np.sum(vv_path), dtype=float)
                    zconv_c = ztop_c + inv.h_pick[vv_path]
                    zconv_path_all.extend(zconv_c.tolist())
                    h_path_all.extend(inv.h_pick[vv_path].tolist())
                    dt_path_all.extend(t_obs[vv_path].tolist())
                    # 统一维护 PPS-PPP 转换点集合（供第3/4图一致显示）
                    xconv_pps_all.extend(inv.x_conv_pick[vv_path].tolist())
                    zconv_pps_all.extend(zconv_c.tolist())
                    gid = np.full(model_x.shape, -1, dtype=int)
                    gid[vv_path] = np.arange(n_prev, n_prev + int(np.sum(vv_path)))
                else:
                    gid = np.full(model_x.shape, -1, dtype=int)
                pick_tables.append(
                    (
                        r.path,
                        model_x.copy(),
                        rec_x.copy(),
                        true_off.copy(),
                        t_obs.copy(),
                        inv.pred_dt.copy(),
                        inv.x_conv_pick.copy(),
                        inv.h_pick.copy(),
                        gid,
                    )
                )
                # 残差散点：e = obs - pred，位置取逐点反演的底界位置
                e_pick = t_obs - inv.pred_dt
                vv_e = np.isfinite(inv.x_conv_pick) & np.isfinite(inv.h_pick) & np.isfinite(e_pick)
                if np.any(vv_e):
                    xr = inv.x_conv_pick[vv_e]
                    if (
                        self.seafloor_x is not None
                        and self.seafloor_z is not None
                        and self.seafloor_x.size >= 2
                        and self.seafloor_z.size >= 2
                    ):
                        ztop_r = np.interp(
                            xr,
                            self.seafloor_x,
                            self.seafloor_z,
                            left=float(self.seafloor_z[0]),
                            right=float(self.seafloor_z[-1]),
                        )
                    else:
                        ztop_r = np.zeros_like(xr)
                    zr = ztop_r + inv.h_pick[vv_e]
                    xres_all.extend(xr.tolist())
                    zres_all.extend(zr.tolist())
                    eres_all.extend(e_pick[vv_e].tolist())
                # PSS->PSP 转换点（用于在第3/4图高亮标注）
                xm_pss_i, off_pss_i, t_pss_i = _phase_model_trueoff_time(r.ds_pss, PHASE_PSS)
                if xm_pss_i.size >= 4:
                    try:
                        _, xconv_pss_i, h_pss_i, _, _ = pss_minus_psp_from_pss_slope_with_profile(
                            xm_pss_i,
                            off_pss_i,
                            t_pss_i,
                            vp=float(self.vp_cr.get()),
                            h_profile_x=inv.x_conv,
                            h_profile=inv.h_conv,
                            vpratio_profile=inv.vpratio_conv,
                            h_default=float(self.h_cr.get()),
                            vpratio_default=float(self.vp_cr.get()) / max(float(self.vs_cr.get()), 1e-6),
                            window_points=self._window_points_value(),
                            split_by_sign=True,
                            n_iter=2,
                            smooth_dense_half_win=self._smooth_dense_half_win_value(),
                        )
                        vv_pss = np.isfinite(xconv_pss_i) & np.isfinite(h_pss_i)
                        if np.any(vv_pss):
                            if (
                                self.seafloor_x is not None
                                and self.seafloor_z is not None
                                and self.seafloor_x.size >= 2
                                and self.seafloor_z.size >= 2
                            ):
                                ztop_pss = np.interp(
                                    xconv_pss_i[vv_pss],
                                    self.seafloor_x,
                                    self.seafloor_z,
                                    left=float(self.seafloor_z[0]),
                                    right=float(self.seafloor_z[-1]),
                                )
                            else:
                                ztop_pss = np.zeros(np.sum(vv_pss), dtype=float)
                            zconv_pss = ztop_pss + h_pss_i[vv_pss]
                            xconv_pss_all.extend(xconv_pss_i[vv_pss].tolist())
                            zconv_pss_all.extend(zconv_pss.tolist())
                    except Exception:
                        pass
                ok_files += 1

        if ok_files == 0:
            win.close()
            messagebox.showwarning("反演失败", "没有可反演的数据（请检查 PPP/PPS 拾取覆盖）。")
            return
        self.pss_inversion_ready = True
        # 反演完成后立即刷新主窗口，使图6按反演剖面重算 PSP_new
        self._draw_results()

        mfield = None
        if dt_path_all:
            mfield, cache_hit_mfield = self._cached_invert_mfield(
                np.asarray(xrec_path_all, dtype=float),
                np.asarray(zrec_path_all, dtype=float),
                np.asarray(xconv_path_all, dtype=float),
                np.asarray(zconv_path_all, dtype=float),
                np.asarray(dt_path_all, dtype=float),
                vp=vp,
            )

        ax_a.set_title("Observed vs inverted PPS-PPP and PSS-PSP")
        ax_a.set_xlabel("Model distance (km)")
        ax_a.set_ylabel("PPS-PPP (s)")
        ax_a.grid(True, alpha=0.5)
        ax_a.legend(fontsize=8, loc="lower left")
        # 第1图拟合误差标注（inv PPS-PPP vs observed）
        if inv_fit_stats:
            x0, y0 = 0.985, 0.97
            dy = 0.078
            for i, (obs_tag, color, st) in enumerate(inv_fit_stats[:6]):
                txt = (
                    f"{obs_tag}: "
                    f"RMS={st['rms']:.4f}, "
                    f"|e|med={st['abs_median']:.4f}, "
                    f"|e|max={st['abs_max']:.4f}"
                )
                ax_a.text(
                    x0,
                    y0 - i * dy,
                    txt,
                    transform=ax_a.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=color, alpha=0.88),
                )

        ax_b.set_title("Inverted h and Vp/Vs vs x_conv")
        ax_b.set_xlabel("x_conv (km)")
        ax_b.set_ylabel("h (km) / VpVs")
        ax_b.grid(True, alpha=0.5)
        ax_b.legend(fontsize=8, loc="lower left")

        # 二维剖面（合并）：厚度作为底界，0~h(x) 内用 Vp/Vs 着色
        point_available = (
            mfield is not None
            and mfield.success
            and mfield.x_edges.size > 1
            and len(xconv_path_all) > 2
        )
        use_point = (self.section_field_mode.get().strip().lower() == "point") and point_available
        if use_point:
            x2d = np.asarray(xconv_path_all, dtype=float)
            h2d = np.asarray(h_path_all, dtype=float)
            r2d = np.asarray(mfield.vpvs_at_conv, dtype=float)
        elif x2d_all and h2d_all and r2d_all:
            x2d = np.asarray(x2d_all, dtype=float)
            h2d = np.asarray(h2d_all, dtype=float)
            r2d = np.asarray(r2d_all, dtype=float)
        else:
            x2d = np.asarray([], dtype=float)
            h2d = np.asarray([], dtype=float)
            r2d = np.asarray([], dtype=float)

        vsec = np.isfinite(x2d) & np.isfinite(h2d) & np.isfinite(r2d)
        if np.any(vsec):
            x2d = x2d[vsec]
            h2d = h2d[vsec]
            r2d = r2d[vsec]
            gx, gz, gr, ztop, zbot = self._build_layer_section_grid(
                x2d,
                h2d,
                r2d,
                nx=220,
                nz=130,
                top_x=self.seafloor_x,
                top_z=self.seafloor_z,
                thickness_mode="nearest",
            )
            in_layer = (gz >= ztop[None, :]) & (gz <= zbot[None, :])
            x_allowed = self._receiver_constrained_xmask(
                gx[0, :],
                np.asarray(xrec_path_all, dtype=float),
                np.asarray(xconv_path_all, dtype=float),
            )
            in_layer_constrained = in_layer & x_allowed[None, :]
            gr = np.asarray(gr, dtype=float)
            gr[~in_layer_constrained] = np.nan
            cov_on_sec = None
            if point_available:
                xe = np.asarray(mfield.x_edges, dtype=float)
                ze = np.asarray(mfield.z_edges, dtype=float)
                covg = np.asarray(mfield.coverage_grid, dtype=float)
                if xe.size >= 2 and ze.size >= 2 and covg.size > 0:
                    ix = np.searchsorted(xe, gx, side="right") - 1
                    iz = np.searchsorted(ze, gz, side="right") - 1
                    ix = np.clip(ix, 0, covg.shape[1] - 1)
                    iz = np.clip(iz, 0, covg.shape[0] - 1)
                    cov_on_sec = covg[iz, ix]

            pcm = ax_c.pcolormesh(gx, gz, gr, shading="auto", cmap="plasma")
            if cov_on_sec is not None:
                uncon = np.where((cov_on_sec <= 0) & in_layer_constrained, 1.0, np.nan)
                ax_c.pcolormesh(gx, gz, uncon, shading="auto", cmap="Greys", vmin=0, vmax=1, alpha=0.24)
            # 海底线：若已加载海底，保留全段；否则仅在受约束区显示
            if self.seafloor_x is not None and self.seafloor_z is not None and self.seafloor_x.size >= 2:
                ax_c.plot(gx[0, :], ztop, color="cyan", linewidth=1.2, label="seafloor")
            else:
                ztop_plot = np.asarray(ztop, dtype=float).copy()
                ztop_plot[~x_allowed] = np.nan
                ax_c.plot(gx[0, :], ztop_plot, color="cyan", linewidth=1.2, label="seafloor")
            zbot_plot = np.asarray(zbot, dtype=float).copy()
            zbot_plot[~x_allowed] = np.nan
            ax_c.plot(gx[0, :], zbot_plot, color="k", linewidth=1.6, label="layer bottom")
            ztop_pick = np.interp(x2d, gx[0, :], ztop)
            ax_c.scatter(x2d, ztop_pick + h2d, s=7, c="k", alpha=0.22)
            # PPS-PPP 转换点（统一集合，左右图一致）
            if xconv_pps_all and zconv_pps_all:
                xpp = np.asarray(xconv_pps_all, dtype=float)
                zpp = np.asarray(zconv_pps_all, dtype=float)
                vpp = np.isfinite(xpp) & np.isfinite(zpp)
                if np.any(vpp):
                    ax_c.scatter(
                        xpp[vpp],
                        zpp[vpp],
                        marker="D",
                        s=24,
                        c="#00FFFF",
                        edgecolors="black",
                        linewidths=0.6,
                        alpha=0.75,
                        label="PPS-PPP conv",
                        zorder=7,
                    )
            if xconv_pss_all and zconv_pss_all:
                xp = np.asarray(xconv_pss_all, dtype=float)
                zp = np.asarray(zconv_pss_all, dtype=float)
                v_pss = np.isfinite(xp) & np.isfinite(zp)
                if np.any(v_pss):
                    ax_c.scatter(
                        xp[v_pss],
                        zp[v_pss],
                        marker="*",
                        s=78,
                        c="#FFFF00",
                        edgecolors="black",
                        linewidths=0.6,
                        alpha=0.95,
                        label="PSS->PSP conv",
                        zorder=7,
                    )
            if xrec_all:
                xrec = np.asarray(xrec_all, dtype=float)
                xrec = xrec[np.isfinite(xrec)]
                if xrec.size > 0:
                    xru = np.unique(np.round(xrec, 3))
                    zrec = np.interp(xru, gx[0, :], ztop)
                    ax_c.scatter(
                        xru,
                        zrec,
                        marker="v",
                        s=42,
                        c="lime",
                        edgecolors="black",
                        linewidths=0.5,
                        alpha=0.95,
                        label="receiver",
                        zorder=6,
                    )
                    # 接收点编号（OBS号）
                    shown_rec = set()
                    for x_m, tag in rec_marks:
                        key = round(float(x_m), 3)
                        if key in shown_rec:
                            continue
                        shown_rec.add(key)
                        y_m = float(np.interp(x_m, gx[0, :], ztop))
                        txt = re.sub(r"\D+", "", str(tag)) or str(tag)
                        ax_c.text(
                            x_m,
                            y_m - 0.03,
                            txt,
                            color="lime",
                            fontsize=8.5,
                            ha="center",
                            va="top",
                            zorder=7,
                            bbox=dict(boxstyle="round,pad=0.1", facecolor="black", edgecolor="none", alpha=0.45),
                        )
            if use_point:
                ax_c.set_title("2D layer section (Vp/Vs point-est fill)")
            elif point_available:
                ax_c.set_title("2D layer section (Vp/Vs eff fill; point available)")
            else:
                ax_c.set_title("2D layer section (Vp/Vs eff fill)")
            ax_c.set_xlabel("x_conv (km)")
            ax_c.set_ylabel("Depth (km)")
            # 动态深度显示范围：
            # 若有海底，顶部仅保留“最浅水深以上 0.5 km”；
            # 若无海底，顶部保持 0 km。
            if self.seafloor_x is not None and self.seafloor_z is not None and self.seafloor_z.size >= 1:
                z_top_view = float(np.nanmin(ztop)) - 0.5
            else:
                z_top_view = 0.0
            ax_c.set_ylim(float(np.nanmax(gz)), z_top_view)
            ax_c.grid(True, alpha=0.28)
            ax_c.legend(
                fontsize=8,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=4,
                framealpha=0.9,
            )
            fig2.colorbar(pcm, ax=ax_c, shrink=0.85, pad=0.02, label="Vp/Vs")

            # 右下：残差热图 + 统计
            if use_point:
                xr = np.asarray(xconv_path_all, dtype=float)
                zr = np.asarray(zconv_path_all, dtype=float)
                er = np.asarray(mfield.residual, dtype=float)
            elif xres_all and zres_all and eres_all:
                xr = np.asarray(xres_all, dtype=float)
                zr = np.asarray(zres_all, dtype=float)
                er = np.asarray(eres_all, dtype=float)
            else:
                xr = np.asarray([], dtype=float)
                zr = np.asarray([], dtype=float)
                er = np.asarray([], dtype=float)

            if xr.size > 0:
                vr = np.isfinite(xr) & np.isfinite(zr) & np.isfinite(er)
                xr, zr, er = xr[vr], zr[vr], er[vr]
                if xr.size >= 3:
                    dx = gx[..., None] - xr[None, None, :]
                    dz = gz[..., None] - zr[None, None, :]
                    d2 = dx * dx + dz * dz
                    w = 1.0 / np.maximum(d2, 1e-12)
                    exact = d2 < 1e-12
                    ge = np.sum(w * er[None, None, :], axis=2) / np.maximum(np.sum(w, axis=2), 1e-12)
                    if np.any(exact):
                        ie, je, ke = np.where(exact)
                        ge[ie, je] = er[ke]
                    ge[~in_layer_constrained] = np.nan

                    q = float(np.nanpercentile(np.abs(er), 95)) if np.any(np.isfinite(er)) else 1.0
                    if not np.isfinite(q) or q <= 0:
                        q = max(1e-3, float(np.nanmax(np.abs(er))) if np.any(np.isfinite(er)) else 1.0)
                    im = ax_d.pcolormesh(gx, gz, ge, shading="auto", cmap="RdBu_r", vmin=-q, vmax=q)
                    if cov_on_sec is not None:
                        uncon2 = np.where((cov_on_sec <= 0) & in_layer_constrained, 1.0, np.nan)
                        ax_d.pcolormesh(gx, gz, uncon2, shading="auto", cmap="Greys", vmin=0, vmax=1, alpha=0.24)
                    if self.seafloor_x is not None and self.seafloor_z is not None and self.seafloor_x.size >= 2:
                        ax_d.plot(gx[0, :], ztop, color="cyan", linewidth=1.0)
                    else:
                        ztop_plot2 = np.asarray(ztop, dtype=float).copy()
                        ztop_plot2[~x_allowed] = np.nan
                        ax_d.plot(gx[0, :], ztop_plot2, color="cyan", linewidth=1.0)
                    zbot_plot2 = np.asarray(zbot, dtype=float).copy()
                    zbot_plot2[~x_allowed] = np.nan
                    ax_d.plot(gx[0, :], zbot_plot2, color="k", linewidth=1.4)
                    ax_d.scatter(xr, zr, s=6, c="k", alpha=0.2)
                    # PPS-PPP 转换点（与左图同一集合）
                    if xconv_pps_all and zconv_pps_all:
                        xpp2 = np.asarray(xconv_pps_all, dtype=float)
                        zpp2 = np.asarray(zconv_pps_all, dtype=float)
                        vpp2 = np.isfinite(xpp2) & np.isfinite(zpp2)
                        if np.any(vpp2):
                            ax_d.scatter(
                                xpp2[vpp2],
                                zpp2[vpp2],
                                marker="D",
                                s=22,
                                c="#00FFFF",
                                edgecolors="black",
                                linewidths=0.55,
                                alpha=0.75,
                                zorder=7,
                            )
                    if xconv_pss_all and zconv_pss_all:
                        xp2 = np.asarray(xconv_pss_all, dtype=float)
                        zp2 = np.asarray(zconv_pss_all, dtype=float)
                        v_pss2 = np.isfinite(xp2) & np.isfinite(zp2)
                        if np.any(v_pss2):
                            ax_d.scatter(
                                xp2[v_pss2],
                                zp2[v_pss2],
                                marker="*",
                                s=72,
                                c="#FFFF00",
                                edgecolors="black",
                                linewidths=0.55,
                                alpha=0.95,
                                zorder=7,
                            )
                    if xrec_all:
                        xrec2 = np.asarray(xrec_all, dtype=float)
                        xrec2 = xrec2[np.isfinite(xrec2)]
                        if xrec2.size > 0:
                            xru2 = np.unique(np.round(xrec2, 3))
                            zrec2 = np.interp(xru2, gx[0, :], ztop)
                            ax_d.scatter(
                                xru2,
                                zrec2,
                                marker="v",
                                s=36,
                                c="lime",
                                edgecolors="black",
                                linewidths=0.45,
                                alpha=0.95,
                                zorder=6,
                            )
                            # 接收点编号（OBS号）
                            shown_rec2 = set()
                            for x_m, tag in rec_marks:
                                key2 = round(float(x_m), 3)
                                if key2 in shown_rec2:
                                    continue
                                shown_rec2.add(key2)
                                y_m2 = float(np.interp(x_m, gx[0, :], ztop))
                                txt2 = re.sub(r"\D+", "", str(tag)) or str(tag)
                                ax_d.text(
                                    x_m,
                                    y_m2 - 0.03,
                                    txt2,
                                    color="lime",
                                    fontsize=8.0,
                                    ha="center",
                                    va="top",
                                    zorder=7,
                                    bbox=dict(boxstyle="round,pad=0.1", facecolor="black", edgecolor="none", alpha=0.45),
                                )
                    ax_d.set_title("Residual heatmap (obs - pred)")
                    ax_d.set_xlabel("x_conv (km)")
                    ax_d.set_ylabel("Depth (km)")
                    ax_d.set_ylim(float(np.nanmax(gz)), z_top_view)
                    ax_d.grid(True, alpha=0.25)
                    fig2.colorbar(im, ax=ax_d, shrink=0.85, pad=0.02, label="Residual (s)")

                    rms = float(np.sqrt(np.mean(er**2)))
                    med = float(np.median(np.abs(er)))
                    emax = float(np.max(np.abs(er)))
                    mean = float(np.mean(er))
                    ax_d.text(
                        0.02,
                        0.98,
                        f"N={er.size}\nRMS={rms:.4f}s\n|e|med={med:.4f}s\n|e|max={emax:.4f}s\nmean={mean:.4f}s",
                        transform=ax_d.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8.5,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.9),
                    )
                    if cov_on_sec is not None:
                        c_in = cov_on_sec[in_layer_constrained]
                        frac_un = float(np.mean(c_in <= 0.0)) if c_in.size else np.nan
                        ax_d.text(
                            0.98,
                            0.98,
                            f"unconstrained={frac_un*100:.1f}%",
                            transform=ax_d.transAxes,
                            ha="right",
                            va="top",
                            fontsize=8.2,
                            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="gray", alpha=0.85),
                        )
                else:
                    ax_d.set_title("Residual heatmap")
                    ax_d.text(0.5, 0.5, "Residual points not enough", ha="center", va="center", transform=ax_d.transAxes)
            else:
                ax_d.set_title("Residual heatmap")
                ax_d.text(0.5, 0.5, "No valid residual points", ha="center", va="center", transform=ax_d.transAxes)
        else:
            ax_c.set_title("2D layer section")
            ax_c.text(0.5, 0.5, "No valid per-pick inversion points", ha="center", va="center", transform=ax_c.transAxes)
            ax_d.set_title("Residual heatmap")
            ax_d.text(0.5, 0.5, "No valid residual points", ha="center", va="center", transform=ax_d.transAxes)

        fig2.tight_layout()
        self._finish_tool_figure(win, fig2)

        topo_msg = self.seafloor_path.name if self.seafloor_path is not None else "未加载(默认z=0)"
        mode_msg = self.section_field_mode.get().strip().lower()
        if mode_msg == "point" and not point_available:
            mode_msg = "point(requested)->eff(fallback)"
        cache_msg = f"缓存 local={cache_hits_local}/{cache_total_local}, mfield={'hit' if cache_hit_mfield else 'miss'}"
        init_msg = f"初值: h0={h0:.3f} km, Vp={vp:.3f} km/s, Vp/Vs0={ratio0:.3f}; 海底={topo_msg}; 场模式={mode_msg}; {cache_msg}"
        self.status_var.set(f"局部反演完成：{ok_files} 个文件；{init_msg}")
        extra = ""
        if near_compare_msgs:
            extra = "\n\n近邻转换点对比(节选):\n" + "\n".join(near_compare_msgs[:5])
        out_txt = "\n".join(export_msgs) if export_msgs else "无（1D 反演不自动输出 CSV）"
        messagebox.showinfo("局部反演完成", init_msg + "\n\n已输出：\n" + out_txt + extra)

    @staticmethod
    def _idw_grid_2d_profile(
        x: np.ndarray,
        z: np.ndarray,
        v: np.ndarray,
        *,
        nx: int = 180,
        nz: int = 120,
        power: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        将离散 (x,z,v) 用 IDW 插值到规则二维网格。
        返回网格坐标 (X, Z) 与网格值 Vg。
        """
        xv = np.asarray(x, dtype=float)
        zv = np.asarray(z, dtype=float)
        vv = np.asarray(v, dtype=float)
        m = np.isfinite(xv) & np.isfinite(zv) & np.isfinite(vv)
        xv = xv[m]
        zv = zv[m]
        vv = vv[m]
        if xv.size == 0:
            gx = np.zeros((2, 2), dtype=float)
            gz = np.zeros((2, 2), dtype=float)
            gv = np.full((2, 2), np.nan, dtype=float)
            return gx, gz, gv

        xmin, xmax = float(np.min(xv)), float(np.max(xv))
        zmin, zmax = float(np.min(zv)), float(np.max(zv))
        if xmin == xmax:
            xmin -= 0.5
            xmax += 0.5
        if zmin == zmax:
            zmin = max(0.0, zmin - 0.2)
            zmax += 0.2
        gx1 = np.linspace(xmin, xmax, max(30, int(nx)))
        gz1 = np.linspace(zmin, zmax, max(20, int(nz)))
        gx, gz = np.meshgrid(gx1, gz1)

        dx = gx[..., None] - xv[None, None, :]
        dz = gz[..., None] - zv[None, None, :]
        d2 = dx * dx + dz * dz

        exact = d2 < 1e-12
        w = 1.0 / np.maximum(d2, 1e-12) ** (power / 2.0)
        wsum = np.sum(w, axis=2)
        gv = np.sum(w * vv[None, None, :], axis=2) / np.maximum(wsum, 1e-12)

        if np.any(exact):
            ie, je, ke = np.where(exact)
            gv[ie, je] = vv[ke]
        return gx, gz, gv

    @staticmethod
    def _build_layer_section_grid(
        x: np.ndarray,
        h: np.ndarray,
        r: np.ndarray,
        *,
        nx: int = 220,
        nz: int = 130,
        top_x: np.ndarray | None = None,
        top_z: np.ndarray | None = None,
        thickness_mode: str = "nearest",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        构造层状二维剖面网格：
        - 顶界 z_top(x)：若提供海底曲线则使用该曲线，否则为 0；
        - 底界 z_bot(x)=z_top(x)+h(x)；
        - z_top~z_bot 内填充 Vp/Vs，界外为 NaN。
        """
        xv = np.asarray(x, dtype=float)
        hv = np.asarray(h, dtype=float)
        rv = np.asarray(r, dtype=float)
        m = np.isfinite(xv) & np.isfinite(hv) & np.isfinite(rv)
        xv, hv, rv = xv[m], hv[m], rv[m]
        if xv.size == 0:
            gx = np.zeros((2, 2), dtype=float)
            gz = np.zeros((2, 2), dtype=float)
            gv = np.full((2, 2), np.nan, dtype=float)
            zline = np.zeros(2, dtype=float)
            return gx, gz, gv, zline, zline

        # 合并重复 x（取均值）
        xr = np.round(xv, 3)
        ux = np.unique(xr)
        x1, h1, r1 = [], [], []
        for u in ux:
            mm = xr == u
            x1.append(float(np.mean(xv[mm])))
            h1.append(float(np.mean(hv[mm])))
            r1.append(float(np.mean(rv[mm])))
        x1 = np.asarray(x1, dtype=float)
        h1 = np.asarray(h1, dtype=float)
        r1 = np.asarray(r1, dtype=float)
        so = np.argsort(x1)
        x1, h1, r1 = x1[so], h1[so], r1[so]

        if x1.size == 1:
            x1 = np.array([x1[0] - 0.5, x1[0] + 0.5], dtype=float)
            h1 = np.array([h1[0], h1[0]], dtype=float)
            r1 = np.array([r1[0], r1[0]], dtype=float)

        xg = np.linspace(float(x1[0]), float(x1[-1]), max(60, int(nx)))
        if str(thickness_mode).lower() == "nearest":
            # 厚度边界不做线性平滑，使用最近邻映射，避免“虚假连续层底”
            idx_nn = np.argmin(np.abs(xg[:, None] - x1[None, :]), axis=1)
            hline = h1[idx_nn]
            rline = r1[idx_nn]
        else:
            hline = np.interp(xg, x1, h1)
            rline = np.interp(xg, x1, r1)
        if (
            top_x is not None
            and top_z is not None
            and np.asarray(top_x).size >= 2
            and np.asarray(top_z).size >= 2
        ):
            tx = np.asarray(top_x, dtype=float)
            tz = np.asarray(top_z, dtype=float)
            tv = np.isfinite(tx) & np.isfinite(tz)
            tx = tx[tv]
            tz = tz[tv]
            if tx.size >= 2:
                so_t = np.argsort(tx)
                tx = tx[so_t]
                tz = tz[so_t]
                z_top = np.interp(xg, tx, tz, left=tz[0], right=tz[-1])
            else:
                z_top = np.zeros_like(xg)
        else:
            z_top = np.zeros_like(xg)
        z_bot = z_top + hline

        zmin = float(np.nanmin(z_top))
        zmax = float(np.nanmax(z_bot))
        if zmax <= zmin:
            zmax = zmin + 0.2
        pad = 0.05 * (zmax - zmin)
        zg = np.linspace(max(0.0, zmin - pad), zmax + pad, max(30, int(nz)))
        gx, gz = np.meshgrid(xg, zg)

        gv = np.full_like(gx, np.nan, dtype=float)
        inside = (gz >= z_top[None, :]) & (gz <= z_bot[None, :])
        rmesh = np.broadcast_to(rline[None, :], gv.shape)
        gv[inside] = rmesh[inside]
        return gx, gz, gv, z_top, z_bot

    @staticmethod
    def _receiver_constrained_xmask(
        x_grid: np.ndarray,
        x_rec: np.ndarray,
        x_conv: np.ndarray,
        *,
        rec_round: int = 3,
        extend_km: float = 0.6,
        merge_gap_km: float = 0.4,
    ) -> np.ndarray:
        """
        仅允许在“每个接收点的最远两个转换点区间”内显示插值结果。
        为避免显示过于破碎，区间两端做小幅外延，并合并小间隙。
        返回 x_grid 上的布尔掩膜（并集区间）。
        """
        xg = np.asarray(x_grid, dtype=float)
        xr = np.asarray(x_rec, dtype=float)
        xc = np.asarray(x_conv, dtype=float)
        ok = np.isfinite(xg)
        out = np.zeros_like(xg, dtype=bool)
        v = np.isfinite(xr) & np.isfinite(xc)
        if not np.any(v):
            out[ok] = True
            return out
        xr = xr[v]
        xc = xc[v]

        intervals: list[tuple[float, float]] = []
        grp = np.round(xr, rec_round)
        for g in np.unique(grp):
            m = grp == g
            if np.sum(m) < 1:
                continue
            cc = xc[m]
            cmin = float(np.min(cc))
            cmax = float(np.max(cc))
            if cmax < cmin:
                cmin, cmax = cmax, cmin
            if cmax == cmin:
                cmin -= max(0.2, 0.5 * extend_km)
                cmax += max(0.2, 0.5 * extend_km)
            cmin -= max(0.0, extend_km)
            cmax += max(0.0, extend_km)
            intervals.append((cmin, cmax))

        if intervals:
            intervals.sort(key=lambda t: t[0])
            merged: list[tuple[float, float]] = [intervals[0]]
            for a, b in intervals[1:]:
                la, lb = merged[-1]
                if a <= lb + max(0.0, merge_gap_km):
                    merged[-1] = (la, max(lb, b))
                else:
                    merged.append((a, b))
            for a, b in merged:
                out |= (xg >= a) & (xg <= b)
        if not np.any(out):
            out[ok] = True
        return out

    def _compute_pss_psp_pairs_observed(self, r: FileResult) -> list[tuple[float, float, float]]:
        """观测 PSS-PSP 差值（与主图策略一致：严格配对或拟合取值）。"""
        psp_id_curr = int(self.psp_phase_id.get())
        pairs: list[tuple[float, float, float]] = []
        try:
            ds_psp_in, _ = select_phases(r.ds, [psp_id_curr])
            if bool(self.strict_diff_pair.get()):
                return compute_ppp_pps_diff_pairs(
                    ds_psp_in, r.ds_pss, ip1=psp_id_curr, ip2=PHASE_PSS, tol=0.05
                )
            psp_x, psp_off, psp_t = _phase_model_trueoff_time(ds_psp_in, psp_id_curr)
            pss_x, pss_off, pss_t = _phase_model_trueoff_time(r.ds_pss, PHASE_PSS)
            for sgn in (-1.0, 1.0):
                m_psp = (psp_off * sgn) > 0.0
                m_pss = (pss_off * sgn) > 0.0
                if np.count_nonzero(m_psp) < 3 or np.count_nonzero(m_pss) < 1:
                    continue
                psp_fit = fit_ppp_time_curve_local_linear(
                    psp_x[m_psp], psp_t[m_psp], pss_x[m_pss],
                    window_points=self._window_points_value(), split_by_sign=False,
                )
                for j in range(int(np.sum(m_pss))):
                    t_psp_f = float(psp_fit[j])
                    if not np.isfinite(t_psp_f):
                        continue
                    md = float(pss_x[m_pss][j])
                    to = float(pss_off[m_pss][j])
                    dt = float(pss_t[m_pss][j]) - t_psp_f
                    pairs.append((md, to, dt))
            pairs.sort(key=lambda t: t[0])
        except Exception:
            return []
        return pairs

    def show_1d_diagnostics(self) -> None:
        """单独弹窗：1D 下观测/理论 PPS-PPP 与 PSS-PSP 对比（|p|、L、Δt）。

        公式与口径说明见同目录 ``TIME_DIFF_FORMULAS.md``。注意：主理论中 PPS-PPP 用 PPP
        斜率估 L，PSS-PSP 用剖面迭代几何 L；本弹窗中绿色虚线「统一 L」仅为与橙色线
        同形对比，PSS 侧 p 来自 PSS 斜率，与主函数 ``pss_minus_psp_*`` 的 L 可能不同。
        """
        if not self.results:
            messagebox.showwarning("提示", "请先加载 tx.in 文件。")
            return

        win = PlotToolWindow(title="1D 诊断对比", parent=None)
        fig = Figure(figsize=(13.5, 8.5), dpi=100)
        ax_tt = fig.add_subplot(2, 2, 1)
        ax_p = fig.add_subplot(2, 2, 2)
        ax_p_ang = ax_p.twinx()
        ax_p_ang.set_ylabel("θ (°): solid=from vertical; dotted=from horiz. (≈RAYINVR f.angle)")
        ax_p_ang.grid(False)
        ax_l = fig.add_subplot(2, 2, 3)
        ax_dt = fig.add_subplot(2, 2, 4)

        summary_lines: list[str] = []
        vp = float(self.vp_cr.get())
        vs = float(self.vs_cr.get())
        vs_safe = max(vs, 1e-6)
        h = float(self.h_cr.get())
        c_obs_pp = "#1E90FF"   # Obs PPS-PPP
        c_obs_pps = "#FF8C00"  # PPS-based diagnostic
        c_the_pp = "#FF7F00"   # Theo PPS-PPP
        c_obs_ps = "#FF1493"   # Obs PSS-PSP
        c_the_ps = "#32CD32"   # Theo PSS-PSP
        vred_diag = 7.0

        for i, r in enumerate(self.results):
            label = _result_display_name(r)
            if not r.diff_pairs:
                continue
            psp_id_curr = int(self.psp_phase_id.get())
            ds_psp_in, _ = select_phases(r.ds, [psp_id_curr])
            fac_const = (1.0 / max(vs, 1e-6) - 1.0 / vp)
            arr = np.asarray(r.diff_pairs, dtype=float)
            md = arr[:, 0]
            off = arr[:, 1]
            obs_ppsppp = arr[:, 2]
            ax_dt.scatter(md, obs_ppsppp, s=14, alpha=0.35, c=c_obs_pp, marker="o")
            # 图1：PPP/PPS/PSS/PSP 的真实与折合走时及其拟合
            for ds_phase, pid, name, col, mk in (
                (r.ds_ppp, PHASE_PPP, "PPP", "#1F77B4", "o"),
                (r.ds_pps, PHASE_PPS, "PPS", "#FF7F0E", "^"),
                (r.ds_pss, PHASE_PSS, "PSS", "#2CA02C", "D"),
                (ds_psp_in, psp_id_curr, "PSP", "#7B68EE", "s"),
            ):
                _xm, _off, _tt = _phase_model_trueoff_time(ds_phase, pid)
                vv = np.isfinite(_xm) & np.isfinite(_off) & np.isfinite(_tt)
                if not np.any(vv):
                    continue
                xv = _xm[vv]
                ov = _off[vv]
                tv = _tt[vv]
                tr = tv - np.abs(ov) / vred_diag
                ax_tt.scatter(xv, tv, s=12, alpha=0.25, c=col, marker=mk)
                ax_tt.scatter(xv, tr, s=12, alpha=0.25, c=col, marker="x")
                for sgn in (-1.0, 1.0):
                    ms = (ov * sgn) > 0.0
                    if np.count_nonzero(ms) < 4:
                        continue
                    xs = xv[ms]
                    ts = tv[ms]
                    rs = tr[ms]
                    so = np.argsort(xs)
                    xs = xs[so]
                    ts = ts[so]
                    rs = rs[so]
                    t_fit = fit_ppp_time_curve_local_linear(
                        xs,
                        ts,
                        xs,
                        window_points=self._window_points_value(),
                        split_by_sign=False,
                    )
                    r_fit = fit_ppp_time_curve_local_linear(
                        xs,
                        rs,
                        xs,
                        window_points=self._window_points_value(),
                        split_by_sign=False,
                    )
                    vt = np.isfinite(t_fit)
                    vr = np.isfinite(r_fit)
                    if np.any(vt):
                        ax_tt.plot(xs[vt], t_fit[vt], "-", color=col, linewidth=1.2, alpha=0.8)
                    if np.any(vr):
                        ax_tt.plot(xs[vr], r_fit[vr], "--", color=col, linewidth=1.2, alpha=0.8)

            ppp_off, ppp_t = _phase_true_offset_time(r.ds_ppp, PHASE_PPP)
            th_ppsppp, p_ppp = pps_minus_ppp_from_ppp_slope(
                ppp_off, ppp_t, off,
                h_cr=h, vp=vp, vs=vs,
                window_points=self._window_points_value(),
                split_by_sign=True,
                smooth_dense_half_win=self._smooth_dense_half_win_value(),
            )
            v_p = np.isfinite(md) & np.isfinite(p_ppp)
            if np.any(v_p):
                ax_p.scatter(md[v_p], np.abs(p_ppp[v_p]), s=14, alpha=0.45, c=c_obs_pp, marker="o")
                # PPP 支：sin θ_v = |p|·Vp
                pa = np.abs(p_ppp[v_p])
                th_p = np.degrees(np.arcsin(np.clip(pa * vp, 0.0, 1.0)))
                # θ_h = 90°−θ_v（相对水平）；与 RAYINVR r1.out emergent f.angle 对照时 V 须取 f.angle 同一端介质
                th_p_h = 90.0 - th_p
                ax_p_ang.scatter(md[v_p], th_p, s=10, alpha=0.3, c=c_obs_pp, marker="o")
                so_th = np.argsort(md[v_p])
                ax_p_ang.plot(md[v_p][so_th], th_p[so_th], "-", color=c_obs_pp, linewidth=1.0, alpha=0.55)
                ax_p_ang.plot(md[v_p][so_th], th_p_h[so_th], ":", color=c_obs_pp, linewidth=1.0, alpha=0.45)
            p_abs = np.abs(p_ppp)
            cp = np.sqrt(np.maximum(1.0 - (p_abs * vp) ** 2, 0.0))
            l_pp = np.full_like(p_abs, np.nan, dtype=float)
            vg = np.isfinite(cp) & (cp > 1e-6)
            l_pp[vg] = h / cp[vg]
            # 先给旧口径占位（共享 Lppp），后续若有 Lpps 则作为对照线绘制
            th_ppsppp_old = np.full_like(l_pp, np.nan, dtype=float)
            v_upp = np.isfinite(l_pp) & np.isfinite(fac_const) & (fac_const > 0)
            th_ppsppp_old[v_upp] = l_pp[v_upp] / max(vs, 1e-6) - l_pp[v_upp] / max(vp, 1e-6)
            v_lpp = np.isfinite(md) & np.isfinite(l_pp)
            if np.any(v_lpp):
                ax_l.scatter(md[v_lpp], l_pp[v_lpp], s=14, alpha=0.45, c=c_obs_pp, marker="o")

            # PPS 支补充：图2/图3加入 p、角度、L（PPS/PSS 统一按 Vs 口径）
            pps_off, pps_t = _phase_true_offset_time(r.ds_pps, PHASE_PPS)
            p_pps = np.full_like(off, np.nan, dtype=float)
            if pps_off.size >= 4:
                try:
                    _dt_ppsppp_diag, p_pps = pps_minus_ppp_from_ppp_slope(
                        pps_off,
                        pps_t,
                        off,
                        h_cr=h,
                        vp=vp,
                        vs=vs,
                        window_points=self._window_points_value(),
                        split_by_sign=True,
                        smooth_dense_half_win=self._smooth_dense_half_win_value(),
                    )
                except Exception:
                    p_pps = np.full_like(off, np.nan, dtype=float)
            v_pps = np.isfinite(md) & np.isfinite(p_pps)
            l_pps_full = np.full_like(p_pps, np.nan, dtype=float)
            if np.any(v_pps):
                ppps_abs = np.abs(p_pps[v_pps])
                ax_p.scatter(md[v_pps], ppps_abs, s=16, alpha=0.45, c=c_obs_pps, marker="^")
                th_pps = np.degrees(np.arcsin(np.clip(ppps_abs * vs_safe, 0.0, 1.0)))
                th_pps_h = 90.0 - th_pps
                so_pps = np.argsort(md[v_pps])
                ax_p_ang.plot(md[v_pps][so_pps], th_pps[so_pps], "-.", color=c_obs_pps, linewidth=1.0, alpha=0.6)
                ax_p_ang.plot(md[v_pps][so_pps], th_pps_h[so_pps], ":", color=c_obs_pps, linewidth=1.0, alpha=0.45)
                cp_pps = np.sqrt(np.maximum(1.0 - (ppps_abs * vs_safe) ** 2, 0.0))
                l_pps = np.full_like(ppps_abs, np.nan, dtype=float)
                vl_pps = np.isfinite(cp_pps) & (cp_pps > 1e-6) & ((ppps_abs * vs_safe) < 0.95)
                l_pps[vl_pps] = h / cp_pps[vl_pps]
                l_pps_full[v_pps] = l_pps
                l_cap = max(10.0 * float(h), 30.0)
                l_pps[(~np.isfinite(l_pps)) | (l_pps <= 0.0) | (l_pps > l_cap)] = np.nan
                v_lpps = np.isfinite(l_pps)
                if np.any(v_lpps):
                    ax_l.scatter(md[v_pps][v_lpps], l_pps[v_lpps], s=16, alpha=0.45, c=c_obs_pps, marker="^")
            # 图4新口径（主线）：Δt = Lpps/Vs - Lppp/Vp
            th_ppsppp_new = np.full_like(md, np.nan, dtype=float)
            v_new = np.isfinite(l_pps_full) & np.isfinite(l_pp)
            if np.any(v_new):
                th_ppsppp_new[v_new] = l_pps_full[v_new] / max(vs, 1e-6) - l_pp[v_new] / max(vp, 1e-6)
            v_pp_new = np.isfinite(md) & np.isfinite(th_ppsppp_new)
            if np.any(v_pp_new):
                so_new = np.argsort(md[v_pp_new])
                ax_dt.plot(
                    md[v_pp_new][so_new],
                    th_ppsppp_new[v_pp_new][so_new],
                    "-",
                    color=c_the_pp,
                    linewidth=1.9,
                    alpha=0.98,
                )
            # 图4旧口径（对照虚线）：Δt = Lppp/Vs - Lppp/Vp
            v_pp_old = np.isfinite(md) & np.isfinite(th_ppsppp_old)
            if np.any(v_pp_old):
                so_old = np.argsort(md[v_pp_old])
                ax_dt.plot(
                    md[v_pp_old][so_old],
                    th_ppsppp_old[v_pp_old][so_old],
                    "--",
                    color=c_the_pp,
                    linewidth=1.2,
                    alpha=0.55,
                )
            # 图4新增口径（点划线）：Δt = Lpps/Vs - Lpps/Vp
            th_ppsppp_lpps = np.full_like(md, np.nan, dtype=float)
            v_lpps_only = np.isfinite(l_pps_full)
            if np.any(v_lpps_only):
                th_ppsppp_lpps[v_lpps_only] = (
                    l_pps_full[v_lpps_only] / max(vs, 1e-6) - l_pps_full[v_lpps_only] / max(vp, 1e-6)
                )
            v_pp_lpps = np.isfinite(md) & np.isfinite(th_ppsppp_lpps)
            if np.any(v_pp_lpps):
                so_lpps = np.argsort(md[v_pp_lpps])
                ax_dt.plot(
                    md[v_pp_lpps][so_lpps],
                    th_ppsppp_lpps[v_pp_lpps][so_lpps],
                    "-.",
                    color=c_the_pp,
                    linewidth=1.1,
                    alpha=0.5,
                )

            pss_psp_pairs = self._compute_pss_psp_pairs_observed(r)
            obs_psspsp = np.asarray([], dtype=float)
            th_psspsp_main = np.asarray([], dtype=float)
            xm_pss = np.asarray([], dtype=float)
            if pss_psp_pairs:
                arr2 = np.asarray(pss_psp_pairs, dtype=float)
                md2 = arr2[:, 0]
                obs_psspsp = arr2[:, 2]
                ax_dt.scatter(md2, obs_psspsp, s=18, alpha=0.38, c=c_obs_ps, marker="D")

                # 在 PSS 采样点上，计算 PSS 与 PSP 两支的 p/L，并在图2/图3/图4展示
                xm_pss, off_pss, t_pss = _phase_model_trueoff_time(r.ds_pss, PHASE_PSS)
                xm_psp, off_psp, t_psp = _phase_model_trueoff_time(ds_psp_in, psp_id_curr)

                p_pss = np.full_like(off_pss, np.nan, dtype=float)
                p_psp = np.full_like(off_pss, np.nan, dtype=float)
                if off_pss.size >= 4:
                    _dtmp, p_pss = pps_minus_ppp_from_ppp_slope(
                        off_pss,
                        t_pss,
                        off_pss,
                        h_cr=h,
                        vp=vp,
                        vs=vs,
                        window_points=self._window_points_value(),
                        split_by_sign=True,
                        smooth_dense_half_win=self._smooth_dense_half_win_value(),
                    )
                if off_psp.size >= 4 and off_pss.size >= 1:
                    try:
                        _dtmp2, p_psp = pps_minus_ppp_from_ppp_slope(
                            off_psp,
                            t_psp,
                            off_pss,
                            h_cr=h,
                            vp=vp,
                            vs=vs,
                            window_points=self._window_points_value(),
                            split_by_sign=True,
                            smooth_dense_half_win=self._smooth_dense_half_win_value(),
                        )
                    except Exception:
                        p_psp = np.full_like(off_pss, np.nan, dtype=float)

                # 图2：PSS 与 PSP 的 p/角度
                if xm_pss.size == p_pss.size:
                    v_pss_p = np.isfinite(xm_pss) & np.isfinite(p_pss)
                    if np.any(v_pss_p):
                        ax_p.scatter(xm_pss[v_pss_p], np.abs(p_pss[v_pss_p]), s=22, alpha=0.45, c=c_obs_ps, marker="D")
                        pas = np.abs(p_pss[v_pss_p])
                        th_s = np.degrees(np.arcsin(np.clip(pas * vs_safe, 0.0, 1.0)))
                        th_s_h = 90.0 - th_s
                        ax_p_ang.scatter(xm_pss[v_pss_p], th_s, s=12, alpha=0.3, c=c_obs_ps, marker="D")
                        so_ts = np.argsort(xm_pss[v_pss_p])
                        ax_p_ang.plot(xm_pss[v_pss_p][so_ts], th_s[so_ts], "--", color=c_obs_ps, linewidth=1.0, alpha=0.55)
                        ax_p_ang.plot(xm_pss[v_pss_p][so_ts], th_s_h[so_ts], ":", color=c_obs_ps, linewidth=1.0, alpha=0.4)
                if xm_pss.size == p_psp.size:
                    v_psp_p = np.isfinite(xm_pss) & np.isfinite(p_psp)
                    if np.any(v_psp_p):
                        c_obs_psp = "#7B68EE"
                        ax_p.scatter(xm_pss[v_psp_p], np.abs(p_psp[v_psp_p]), s=18, alpha=0.45, c=c_obs_psp, marker="s")
                        psp_abs = np.abs(p_psp[v_psp_p])
                        th_psp = np.degrees(np.arcsin(np.clip(psp_abs * vp, 0.0, 1.0)))
                        th_psp_h = 90.0 - th_psp
                        so_tsp = np.argsort(xm_pss[v_psp_p])
                        ax_p_ang.plot(
                            xm_pss[v_psp_p][so_tsp],
                            th_psp[so_tsp],
                            "-.",
                            color=c_obs_psp,
                            linewidth=1.0,
                            alpha=0.6,
                        )
                        ax_p_ang.plot(
                            xm_pss[v_psp_p][so_tsp],
                            th_psp_h[so_tsp],
                            ":",
                            color=c_obs_psp,
                            linewidth=1.0,
                            alpha=0.45,
                        )

                # 图3：L（PSS 用 Vs，PSP 用 Vp）
                p_abs_ps = np.abs(p_pss)
                p_abs_psp = np.abs(p_psp)
                crit_guard = 0.95
                cp_ps = np.sqrt(np.maximum(1.0 - (p_abs_ps * vs_safe) ** 2, 0.0))
                cp_psp = np.sqrt(np.maximum(1.0 - (p_abs_psp * vp) ** 2, 0.0))
                l_ps = np.full_like(p_abs_ps, np.nan, dtype=float)
                l_psp = np.full_like(p_abs_psp, np.nan, dtype=float)
                vcp_ps = np.isfinite(cp_ps) & (cp_ps > 1e-6) & np.isfinite(p_abs_ps) & ((p_abs_ps * vs_safe) < crit_guard)
                vcp_psp = np.isfinite(cp_psp) & (cp_psp > 1e-6) & np.isfinite(p_abs_psp) & ((p_abs_psp * vp) < crit_guard)
                l_ps[vcp_ps] = h / cp_ps[vcp_ps]
                l_psp[vcp_psp] = h / cp_psp[vcp_psp]
                l_cap = max(10.0 * float(h), 30.0)
                l_ps[(~np.isfinite(l_ps)) | (l_ps <= 0.0) | (l_ps > l_cap)] = np.nan
                l_psp[(~np.isfinite(l_psp)) | (l_psp <= 0.0) | (l_psp > l_cap)] = np.nan
                if xm_pss.size == l_ps.size:
                    vl = np.isfinite(xm_pss) & np.isfinite(l_ps)
                    if np.any(vl):
                        ax_l.scatter(xm_pss[vl], l_ps[vl], s=22, alpha=0.45, c=c_obs_ps, marker="D")
                if xm_pss.size == l_psp.size:
                    vlp = np.isfinite(xm_pss) & np.isfinite(l_psp)
                    if np.any(vlp):
                        ax_l.scatter(xm_pss[vlp], l_psp[vlp], s=18, alpha=0.45, c="#7B68EE", marker="s")

                # 图4：PSS-PSP 三条理论线
                th_psspsp_main = np.full_like(xm_pss, np.nan, dtype=float)
                th_psspsp_old_psp = np.full_like(xm_pss, np.nan, dtype=float)
                th_psspsp_old_pss = np.full_like(xm_pss, np.nan, dtype=float)
                v_m = np.isfinite(l_ps) & np.isfinite(l_psp)
                if np.any(v_m):
                    th_psspsp_main[v_m] = l_ps[v_m] / max(vs, 1e-6) - l_psp[v_m] / max(vp, 1e-6)
                v_psp = np.isfinite(l_psp)
                if np.any(v_psp):
                    th_psspsp_old_psp[v_psp] = l_psp[v_psp] / max(vs, 1e-6) - l_psp[v_psp] / max(vp, 1e-6)
                v_pss = np.isfinite(l_ps)
                if np.any(v_pss):
                    th_psspsp_old_pss[v_pss] = l_ps[v_pss] / max(vs, 1e-6) - l_ps[v_pss] / max(vp, 1e-6)
                v1 = np.isfinite(xm_pss) & np.isfinite(th_psspsp_main)
                if np.any(v1):
                    so1 = np.argsort(xm_pss[v1])
                    ax_dt.plot(
                        xm_pss[v1][so1],
                        th_psspsp_main[v1][so1],
                        "-",
                        color=c_the_ps,
                        linewidth=1.7,
                        alpha=0.95,
                    )
                v2 = np.isfinite(xm_pss) & np.isfinite(th_psspsp_old_psp)
                if np.any(v2):
                    so2 = np.argsort(xm_pss[v2])
                    ax_dt.plot(
                        xm_pss[v2][so2],
                        th_psspsp_old_psp[v2][so2],
                        "--",
                        color=c_the_ps,
                        linewidth=1.2,
                        alpha=0.58,
                    )
                v3 = np.isfinite(xm_pss) & np.isfinite(th_psspsp_old_pss)
                if np.any(v3):
                    so3 = np.argsort(xm_pss[v3])
                    ax_dt.plot(
                        xm_pss[v3][so3],
                        th_psspsp_old_pss[v3][so3],
                        "-.",
                        color=c_the_ps,
                        linewidth=1.1,
                        alpha=0.5,
                    )

            med_obs_pp = float(np.nanmedian(obs_ppsppp)) if obs_ppsppp.size else float("nan")
            med_th_pp = float(np.nanmedian(th_ppsppp_new)) if np.any(np.isfinite(th_ppsppp_new)) else float("nan")
            med_obs_ps = float(np.nanmedian(obs_psspsp)) if obs_psspsp.size else float("nan")
            med_th_ps = float(np.nanmedian(th_psspsp_main)) if np.any(np.isfinite(th_psspsp_main)) else float("nan")
            summary_lines.append(
                f"{label}: obs(PPS-PPP)={med_obs_pp:.4f}, theo(PPS-PPP)={med_th_pp:.4f}; "
                f"obs(PSS-PSP)={med_obs_ps:.4f}, theo(PSS-PSP)={med_th_ps:.4f}"
            )
            # 共同网格统计：只在共同 x 区间对比理论曲线
            try:
                vpp = np.isfinite(md) & np.isfinite(th_ppsppp_new)
                vps = np.isfinite(xm_pss) & np.isfinite(th_psspsp_main)
                if np.count_nonzero(vpp) >= 2 and np.count_nonzero(vps) >= 2:
                    x_pp = md[vpp]
                    y_pp = th_ppsppp_new[vpp]
                    x_ps = xm_pss[vps]
                    y_ps = th_psspsp_main[vps]
                    so_pp = np.argsort(x_pp)
                    so_ps = np.argsort(x_ps)
                    x_pp, y_pp = x_pp[so_pp], y_pp[so_pp]
                    x_ps, y_ps = x_ps[so_ps], y_ps[so_ps]
                    xmin = max(float(np.min(x_pp)), float(np.min(x_ps)))
                    xmax = min(float(np.max(x_pp)), float(np.max(x_ps)))
                    if xmax > xmin:
                        ngrid = int(max(50, min(300, min(x_pp.size, x_ps.size) * 2)))
                        xg = np.linspace(xmin, xmax, ngrid)
                        ypp_g = np.interp(xg, x_pp, y_pp)
                        yps_g = np.interp(xg, x_ps, y_ps)
                        med_pp_g = float(np.nanmedian(ypp_g))
                        med_ps_g = float(np.nanmedian(yps_g))
                        summary_lines.append(
                            f"{label} [common-grid]: theo(PPS-PPP)={med_pp_g:.4f}, theo(PSS-PSP)={med_ps_g:.4f}"
                        )
            except Exception:
                pass

        ax_tt.set_title("PPP/PPS/PSS Travel Times (real & reduced) + fits")
        ax_tt.set_xlabel("Model distance (km)")
        ax_tt.set_ylabel("Time (s)")
        ax_tt.grid(True, alpha=0.5)
        ax_tt.legend(
            handles=[
                Line2D([0], [0], marker="o", linestyle="-", color="#1F77B4", label="PPP real+fit", markersize=4),
                Line2D([0], [0], marker="x", linestyle="--", color="#1F77B4", label="PPP reduced+fit", markersize=4),
                Line2D([0], [0], marker="^", linestyle="-", color="#FF7F0E", label="PPS real+fit", markersize=4),
                Line2D([0], [0], marker="x", linestyle="--", color="#FF7F0E", label="PPS reduced+fit", markersize=4),
                Line2D([0], [0], marker="D", linestyle="-", color="#2CA02C", label="PSS real+fit", markersize=4),
                Line2D([0], [0], marker="x", linestyle="--", color="#2CA02C", label="PSS reduced+fit", markersize=4),
                Line2D([0], [0], marker="s", linestyle="-", color="#7B68EE", label="PSP real+fit", markersize=4),
                Line2D([0], [0], marker="x", linestyle="--", color="#7B68EE", label="PSP reduced+fit", markersize=4),
            ],
            fontsize=7,
            loc="best",
            ncol=2,
        )

        ax_dt.set_title("Observed/Theoretical Dt (PPS-PPP & PSS-PSP)")
        ax_dt.set_xlabel("Model distance (km)")
        ax_dt.set_ylabel("Dt (s)")
        ax_dt.grid(True, alpha=0.5)
        ax_dt.legend(
            handles=[
                Line2D([0], [0], marker="o", linestyle="None", color=c_obs_pp, label="Obs PPS-PPP", markersize=5),
                Line2D([0], [0], linestyle="-", color=c_the_pp, label="PPS-PPP main"),
                Line2D([0], [0], linestyle="--", color=c_the_pp, label="PPS-PPP alt1"),
                Line2D([0], [0], linestyle="-.", color=c_the_pp, label="PPS-PPP alt2"),
                Line2D([0], [0], marker="D", linestyle="None", color=c_obs_ps, label="Obs PSS-PSP", markersize=5),
                Line2D([0], [0], linestyle="-", color=c_the_ps, label="PSS-PSP main"),
                Line2D([0], [0], linestyle="--", color=c_the_ps, label="PSS-PSP alt1"),
                Line2D([0], [0], linestyle="-.", color=c_the_ps, label="PSS-PSP alt2"),
            ],
            fontsize=8,
            loc="best",
            ncol=2,
        )
        ax_p.set_title("Abs(p) & θ: sin(θ_v)=|p|V; θ_h=90°−θ_v ≈ RAYINVR emergent angle")
        ax_p.set_xlabel("Model distance (km)")
        ax_p.set_ylabel("|p| (s/km)")
        ax_p.grid(True, alpha=0.5)
        ax_p.legend(
            handles=[
                Line2D([0], [0], marker="o", linestyle="None", color=c_obs_pp, label="p from PPP", markersize=5),
                Line2D([0], [0], marker="^", linestyle="None", color=c_obs_pps, label="p from PPS", markersize=5),
                Line2D([0], [0], marker="D", linestyle="None", color=c_obs_ps, label="p from PSS", markersize=5),
                Line2D([0], [0], marker="s", linestyle="None", color="#7B68EE", label="p from PSP", markersize=5),
                Line2D([0], [0], marker="o", linestyle="-", color=c_obs_pp, label="θ_v (PPP, arcsin|p|Vp)", markersize=4),
                Line2D([0], [0], marker="^", linestyle="-.", color=c_obs_pps, label="θ_v (PPS, arcsin|p|Vs)", markersize=4),
                Line2D([0], [0], linestyle=":", color=c_obs_pp, label="θ_P h=90°−θ_P v (≈f.angle)"),
                Line2D([0], [0], marker="D", linestyle="--", color=c_obs_ps, label="θ_v (PSS, arcsin|p|Vs)", markersize=4),
                Line2D([0], [0], marker="s", linestyle="-.", color="#7B68EE", label="θ_v (PSP, arcsin|p|Vp)", markersize=4),
                Line2D([0], [0], linestyle=":", color=c_obs_ps, label="θ_h=90°−θ_v"),
            ],
            fontsize=8,
            loc="best",
        )
        ax_l.set_title("Path Factor L Comparison")
        ax_l.set_xlabel("Model distance (km)")
        ax_l.set_ylabel("L (km)")
        ax_l.grid(True, alpha=0.5)
        ax_l.legend(
            handles=[
                Line2D([0], [0], marker="o", linestyle="None", color=c_obs_pp, label="L from PPP model", markersize=5),
                Line2D([0], [0], marker="^", linestyle="None", color=c_obs_pps, label="L from PPS model", markersize=5),
                Line2D([0], [0], marker="D", linestyle="None", color=c_obs_ps, label="L from PSS model", markersize=5),
                Line2D([0], [0], marker="s", linestyle="None", color="#7B68EE", label="L from PSP model", markersize=5),
            ],
            fontsize=8,
            loc="best",
        )
        if summary_lines:
            ax_dt.text(
                0.01,
                0.03,
                "\n".join(summary_lines[:3]),
                transform=ax_dt.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
            )

        self._finish_tool_figure(win, fig)

    def _load_and_draw(self) -> None:
        self.pss_inversion_ready = False
        self.pss_profile_by_file = {}
        self._clear_theory2d_branch_cache()
        self.results = []
        errors: list[str] = []
        psp_id = int(self.psp_phase_id.get())
        if psp_id <= 0:
            messagebox.showwarning("参数错误", "PSP相位号必须为正整数。")
            return
        stations = _stations_from_depth_arrays(
            getattr(self, "shot_depth_x", None),
            getattr(self, "shot_depth_z", None),
            getattr(self, "shot_depth_ids", None),
        )
        n_units = 0
        for p in self.files:
            try:
                allowed_phase_ids = (
                    self._enabled_ivrays_for_tx_path(p)
                    if bool(self.use_rin_enabled_filter.get())
                    else None
                )
                units = _compute_file_results(
                    p,
                    psp_phase_id=psp_id,
                    window_points=self._window_points_value(),
                    strict_only=bool(self.strict_diff_pair.get()),
                    allowed_phase_ids=allowed_phase_ids,
                    stations=stations or None,
                    split_multi_obs=True,
                )
                self.results.extend(units)
                n_units += len(units)
            except Exception as e:
                errors.append(f"{p.name}: {e}")
        if errors:
            messagebox.showwarning("部分文件加载失败", "\n".join(errors))
        if self.results:
            self.status_var.set(
                f"已加载 {len(self.files)} 个文件 → {n_units} 个 OBS 分析单元"
            )
        self._draw_results()
        self._try_auto_sync_rin_shots()
        self._refresh_pois_from_rin()
        self._ensure_tx_equi_for_results()
        self._save_workbench_state()

    def _load_workbench_state(self) -> dict:
        if self._gui_state_file is None or not self._gui_state_file.exists():
            return {}
        try:
            raw = json.loads(self._gui_state_file.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return {}
            obj = raw.get("iphase_gui", {})
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
            raw["iphase_gui"] = {
                "files": [str(p) for p in self.files],
                "theory_mode": self.theory_mode.get(),
                "psp_export_mode": self.psp_export_mode.get(),
                "share_y": bool(self.share_y_var.get()),
                "window_points": int(self.window_points.get()),
                "smooth_half_win": self.smooth_dense_half_win.get(),
                "psp_phase_id": int(self.psp_phase_id.get()),
                "picked_policy": self.picked_policy.get(),
                "strict_diff_pair": bool(self.strict_diff_pair.get()),
                "force_recompute": bool(self.force_recompute.get()),
                "theory2d_auto_fallback": bool(self.theory2d_auto_fallback.get()),
                "use_rin_enabled_filter": bool(self.use_rin_enabled_filter.get()),
                "h_cr": float(self.h_cr.get()),
                "vp_cr": float(self.vp_cr.get()),
                "vs_cr": float(self.vs_cr.get()),
                "obs_mark_y": float(self.obs_mark_y.get()),
                "pois_left": self.pois_left_var.get(),
                "pois_right": self.pois_right_var.get(),
                "pps_pss_ratio": float(self.pps_pss_ratio.get()),
                "equi_write_equiv_psp": bool(self.equi_write_equiv_psp.get()),
                "section_field_mode": self.section_field_mode.get(),
                "seafloor_path": str(self.seafloor_path) if self.seafloor_path else "",
                "shot_depth_path": str(self.shot_depth_path) if self.shot_depth_path else "",
            }
            self._gui_state_file.parent.mkdir(parents=True, exist_ok=True)
            self._gui_state_file.write_text(
                json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    def _restore_workbench_state(self) -> None:
        state = self._load_workbench_state()
        if not state:
            return
        restored: list[str] = []
        try:
            self.theory_mode.set(str(state.get("theory_mode", self.theory_mode.get())))
            self.psp_export_mode.set(str(state.get("psp_export_mode", self.psp_export_mode.get())))
            self.share_y_var.set(bool(state.get("share_y", self.share_y_var.get())))
            self.window_points.set(int(state.get("window_points", self.window_points.get())))
            self.smooth_dense_half_win.set(str(state.get("smooth_half_win", self.smooth_dense_half_win.get())))
            self.psp_phase_id.set(int(state.get("psp_phase_id", self.psp_phase_id.get())))
            self.picked_policy.set(str(state.get("picked_policy", self.picked_policy.get())))
            self.strict_diff_pair.set(bool(state.get("strict_diff_pair", self.strict_diff_pair.get())))
            self.force_recompute.set(bool(state.get("force_recompute", self.force_recompute.get())))
            self.theory2d_auto_fallback.set(
                bool(state.get("theory2d_auto_fallback", self.theory2d_auto_fallback.get()))
            )
            self.use_rin_enabled_filter.set(
                bool(state.get("use_rin_enabled_filter", self.use_rin_enabled_filter.get()))
            )
            self.h_cr.set(float(state.get("h_cr", self.h_cr.get())))
            self.vp_cr.set(float(state.get("vp_cr", self.vp_cr.get())))
            self.vs_cr.set(float(state.get("vs_cr", self.vs_cr.get())))
            self.obs_mark_y.set(float(state.get("obs_mark_y", self.obs_mark_y.get())))
            self.pois_left_var.set(str(state.get("pois_left", self.pois_left_var.get())))
            self.pois_right_var.set(str(state.get("pois_right", self.pois_right_var.get())))
            self.pps_pss_ratio.set(float(state.get("pps_pss_ratio", self.pps_pss_ratio.get())))
            self.equi_write_equiv_psp.set(
                bool(state.get("equi_write_equiv_psp", self.equi_write_equiv_psp.get()))
            )
            self.section_field_mode.set(str(state.get("section_field_mode", self.section_field_mode.get())))
            restored.append("参数")
        except Exception:
            pass

        seafloor_path = str(state.get("seafloor_path", "")).strip()
        seafloor_path = self._resolve_restorable_input_path(seafloor_path)
        if seafloor_path and Path(seafloor_path).exists():
            if self._load_seafloor_depth_from_file(Path(seafloor_path), show_error=False):
                restored.append(f"海底深度: {Path(seafloor_path).name}")
        shot_depth_path = str(state.get("shot_depth_path", "")).strip()
        shot_depth_path = self._resolve_restorable_input_path(shot_depth_path)
        if shot_depth_path and Path(shot_depth_path).exists():
            if self._load_shot_depth_from_file(Path(shot_depth_path), show_error=False):
                restored.append(f"炮点深度: {Path(shot_depth_path).name}")

        files_raw = state.get("files", [])
        files: list[Path] = []
        if isinstance(files_raw, list):
            for it in files_raw:
                p = Path(self._resolve_restorable_input_path(str(it)))
                if p.exists():
                    files.append(p)
        if files:
            self.files = files
            self._load_and_draw()
            restored.append(f"走时文件: {len(files)} 个")
        if restored:
            state_path = str(self._gui_state_file) if self._gui_state_file is not None else "(未配置)"
            self.status_var.set("已恢复上次会话状态（" + "，".join(restored) + f"），状态文件: {state_path}")

    def _on_close(self) -> None:
        self._save_workbench_state()

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

    def _calc_theory(self, r: FileResult, model_x: np.ndarray, true_off: np.ndarray) -> np.ndarray:
        mode = self.theory_mode.get()
        if mode in ("2D", "2Dequi"):
            return self._calc_theory_2d(r, model_x, true_off)
        return self._calc_theory_1d(r, true_off)

    def _calc_theory_2d(self, r: FileResult, model_x: np.ndarray, true_off: np.ndarray) -> np.ndarray:
        """2D/2Dequi：优先用左右支分侧缓存拼理论 PPS−PPP；否则读 tx.out 并初始化双侧缓存。"""
        xq = np.asarray(model_x, dtype=float)
        oq = np.asarray(true_off, dtype=float)
        wd = r.path.parent
        ent = self._ensure_t2d_side_entry(wd)
        left_b = ent.get("left_delta")
        right_b = ent.get("right_delta")
        if left_b is None and right_b is None:
            # 首次：从现有 tx.out 填双侧
            if (wd / "tx.out").exists():
                self._update_t2d_side_from_txout(wd, side="both")
                ent = self._ensure_t2d_side_entry(wd)
                left_b = ent.get("left_delta")
                right_b = ent.get("right_delta")

        if left_b is not None or right_b is not None:
            left_x, left_dt = left_b if left_b is not None else (np.asarray([]), np.asarray([]))
            right_x, right_dt = right_b if right_b is not None else (np.asarray([]), np.asarray([]))
            left_x = np.asarray(left_x, dtype=float)
            left_dt = np.asarray(left_dt, dtype=float)
            right_x = np.asarray(right_x, dtype=float)
            right_dt = np.asarray(right_dt, dtype=float)
            dt_out = np.full_like(xq, np.nan, dtype=float)
            m_left = oq < 0
            m_right = oq >= 0
            if left_x.size >= 2 and np.any(m_left):
                dt_out[m_left] = np.interp(
                    xq[m_left], left_x, left_dt, left=left_dt[0], right=left_dt[-1]
                )
            if right_x.size >= 2 and np.any(m_right):
                dt_out[m_right] = np.interp(
                    xq[m_right], right_x, right_dt, left=right_dt[0], right=right_dt[-1]
                )
            if np.any(np.isfinite(dt_out)):
                self.theory2d_notice = ""
                return dt_out

        # 回退：整份 tx.out 拟合（并尝试填缓存）
        tx_out_path = wd / "tx.out"
        if not tx_out_path.exists():
            if self.theory2d_auto_fallback.get():
                self.theory2d_notice = "2D回退1D: tx.out不存在"
                return self._calc_theory_1d(r, true_off)
            self.theory2d_notice = "2D失败: tx.out不存在"
            return np.full_like(model_x, np.nan, dtype=float)
        try:
            phase_pts = collect_phase_points_from_txout(
                tx_out_path, phase_ids=(PHASE_PPP, PHASE_PPS)
            )
        except Exception:
            if self.theory2d_auto_fallback.get():
                self.theory2d_notice = "2D回退1D: tx.out解析失败"
                return self._calc_theory_1d(r, true_off)
            return np.full_like(model_x, np.nan, dtype=float)

        ppp_data = phase_pts.get(int(PHASE_PPP))
        pps_data = phase_pts.get(int(PHASE_PPS))
        if ppp_data is None or pps_data is None or ppp_data[0].size < 3 or pps_data[0].size < 3:
            if self.theory2d_auto_fallback.get():
                self.theory2d_notice = "2D回退1D: PPP/PPS理论点不足"
                return self._calc_theory_1d(r, true_off)
            return np.full_like(model_x, np.nan, dtype=float)

        dt_out = np.full_like(xq, np.nan, dtype=float)
        self.theory2d_notice = ""
        wp = self._window_points_value()

        for sgn in (-1.0, 1.0):
            mq = (oq * sgn) > 0.0
            if not np.any(mq):
                continue
            m_ppp = (
                (ppp_data[0] * sgn - ppp_data[2] * sgn) > 0.0
                if ppp_data[2].size == ppp_data[0].size
                else np.ones(ppp_data[0].size, dtype=bool)
            )
            m_pps = (
                (pps_data[0] * sgn - pps_data[2] * sgn) > 0.0
                if pps_data[2].size == pps_data[0].size
                else np.ones(pps_data[0].size, dtype=bool)
            )
            if np.count_nonzero(m_ppp) < 3 or np.count_nonzero(m_pps) < 3:
                continue
            t_ppp_fit = fit_ppp_time_curve_local_linear(
                ppp_data[0][m_ppp], ppp_data[1][m_ppp], xq[mq],
                window_points=wp, split_by_sign=False,
            )
            t_pps_fit = fit_ppp_time_curve_local_linear(
                pps_data[0][m_pps], pps_data[1][m_pps], xq[mq],
                window_points=wp, split_by_sign=False,
            )
            v = np.isfinite(t_ppp_fit) & np.isfinite(t_pps_fit)
            dt_out[mq] = np.where(v, t_pps_fit - t_ppp_fit, np.nan)
        return dt_out

    def _calc_theory_2d_pss_psp(self, r: FileResult, model_x: np.ndarray, true_off: np.ndarray) -> np.ndarray:
        """
        2D/2Dequi：从 tx.out 中 PSS 与 PSP 两支理论走时，在 query 点拟合取值，
        返回理论 PSS-PSP 时差 Δt = t_PSS - t_PSP（与拾取定义一致）。
        若某相位在 tx.out 中缺失或点数不足，对应点为 NaN。
        """
        xq = np.asarray(model_x, dtype=float)
        oq = np.asarray(true_off, dtype=float)
        dt_out = np.full_like(xq, np.nan, dtype=float)
        psp_id = int(self.psp_phase_id.get())
        tx_out_path = r.path.parent / "tx.out"
        if not tx_out_path.exists():
            return dt_out
        try:
            phase_pts = collect_phase_points_from_txout(
                tx_out_path, phase_ids=(PHASE_PSS, psp_id)
            )
        except Exception:
            return dt_out
        pss_data = phase_pts.get(int(PHASE_PSS))
        psp_data = phase_pts.get(int(psp_id))
        if (
            pss_data is None
            or psp_data is None
            or pss_data[0].size < 3
            or psp_data[0].size < 3
        ):
            return dt_out
        wp = self._window_points_value()
        for sgn in (-1.0, 1.0):
            mq = (oq * sgn) > 0.0
            if not np.any(mq):
                continue
            m_pss = (
                (pss_data[0] * sgn - pss_data[2] * sgn) > 0.0
                if pss_data[2].size == pss_data[0].size
                else np.ones(pss_data[0].size, dtype=bool)
            )
            m_psp = (
                (psp_data[0] * sgn - psp_data[2] * sgn) > 0.0
                if psp_data[2].size == psp_data[0].size
                else np.ones(psp_data[0].size, dtype=bool)
            )
            if np.count_nonzero(m_pss) < 3 or np.count_nonzero(m_psp) < 3:
                continue
            t_pss_fit = fit_ppp_time_curve_local_linear(
                pss_data[0][m_pss],
                pss_data[1][m_pss],
                xq[mq],
                window_points=wp,
                split_by_sign=False,
            )
            t_psp_fit = fit_ppp_time_curve_local_linear(
                psp_data[0][m_psp],
                psp_data[1][m_psp],
                xq[mq],
                window_points=wp,
                split_by_sign=False,
            )
            v = np.isfinite(t_pss_fit) & np.isfinite(t_psp_fit)
            dt_out[mq] = np.where(v, t_pss_fit - t_psp_fit, np.nan)
        return dt_out

    def _delta_fn_pss_psp_from_txout_only(self, r: FileResult):
        """
        导出/插值用：仅在 tx.out 可插出 PSS、PSP 两支时差时返回 Δt(PSS-PSP)；
        否则 NaN（不回退 PPS-PPP，与主图右下「PSP from Δt theory(PSS)」一致）。
        """

        def _inner(x: float, off: float | None = None) -> float:
            if off is None or not np.isfinite(off):
                return float("nan")
            dt = self._calc_theory_2d_pss_psp(
                r,
                np.asarray([float(x)], dtype=float),
                np.asarray([float(off)], dtype=float),
            )
            if dt.size and np.isfinite(dt[0]):
                return float(dt[0])
            return float("nan")

        return _inner

    def _calc_theory_1d(self, r: FileResult, true_off: np.ndarray) -> np.ndarray:
        # 默认 1D：PPP slope -> p(x)
        ppp_offsets, ppp_times = _phase_true_offset_time(r.ds_ppp, PHASE_PPP)
        if ppp_offsets.size < 4:
            return np.full_like(true_off, np.nan, dtype=float)
        try:
            dt, _ = pps_minus_ppp_from_ppp_slope(
                ppp_offsets,
                ppp_times,
                true_off,
                h_cr=float(self.h_cr.get()),
                vp=float(self.vp_cr.get()),
                vs=float(self.vs_cr.get()),
                window_points=self._window_points_value(),
                split_by_sign=True,
                smooth_dense_half_win=self._smooth_dense_half_win_value(),
            )
            return dt
        except Exception:
            return np.full_like(true_off, np.nan, dtype=float)

    def _window_points_value(self) -> int:
        try:
            w = int(self.window_points.get())
        except Exception:
            return 11
        if w < 3:
            w = 3
        # 局部线性拟合偏好奇数窗口
        if w % 2 == 0:
            w += 1
        return w

    def _smooth_dense_half_win_value(self) -> int:
        """求导前致密曲线平滑半窗，0 表示不平滑。"""
        try:
            v = int(self.smooth_dense_half_win.get())
        except Exception:
            return 5
        return max(0, min(20, v))

    def _build_psp_dataset(self, r: FileResult, *, mode: str) -> PhaseDataset:
        """
        基于 PSS 拾取构造 PSP 数据集：
          PSP = PSS - Delta_t
        其中 Delta_t：theory2d / theory2Dequi 仅为 tx.out 理论 PSS-PSP（不可用则跳过该点）；
        picked 为实测 PPS-PPP 拟合；theory_pss 为 1D 剖面公式。
        """
        mode = mode.lower()
        if mode == "theory_pss":
            return self._build_psp_dataset_from_pss_theory(r)

        delta_fn = self._build_delta_function(r, mode=mode)
        out = PhaseDataset()

        for s in r.ds_pss.shots:
            new_shot = Shot(xshot=s.xshot, tshot=s.tshot, ushot=s.ushot)
            for p in s.picks:
                if p.phase_id != PHASE_PSS:
                    continue
                dt = float(delta_fn(p.x, p.x - s.xshot))
                if not np.isfinite(dt):
                    continue
                t_psp = float(p.t - dt)
                new_shot.add_pick(
                    x=float(p.x),
                    t=t_psp,
                    u=float(p.u),
                    phase_id=int(self.psp_phase_id.get()),
                )
            if new_shot.picks:
                out.shots.append(new_shot)
        return out

    def _build_psp_dataset_from_pss_theory(self, r: FileResult) -> PhaseDataset:
        """
        theory_pss 模式：
        - 若已执行反演校正：使用反演得到的 h(x), VpVs(x)；
        - 否则：使用用户输入初值 h_cr 与 Vp/Vs 计算。
        之后用 PSS 斜率估计 p，计算理论 PSS-PSP，并得到 PSP。
        """
        inv_local = None
        if self.pss_inversion_ready:
            inv_local = self.pss_profile_by_file.get(_result_cache_key(r))

        # 按 PSS picks 顺序生成理论 Delta_t
        pss_picks: list[tuple[int, object]] = []
        xm, off, tt = [], [], []
        for si, s in enumerate(r.ds_pss.shots):
            for p in s.picks:
                if p.phase_id == PHASE_PSS:
                    pss_picks.append((si, p))
                    xm.append(float(p.x))
                    off.append(float(p.x - s.xshot))
                    tt.append(float(p.t))
        if len(pss_picks) < 3:
            raise RuntimeError("PSS 拾取不足，无法计算 theory_pss")

        xm_a = np.asarray(xm, dtype=float)
        off_a = np.asarray(off, dtype=float)
        tt_a = np.asarray(tt, dtype=float)
        if inv_local is not None:
            hx = inv_local.x_conv
            hh = inv_local.h_conv
            hr = inv_local.vpratio_conv
        else:
            hx = None
            hh = None
            hr = None
        dt_pss, _, _, _, _ = pss_minus_psp_from_pss_slope_with_profile(
            xm_a,
            off_a,
            tt_a,
            vp=float(self.vp_cr.get()),
            h_profile_x=hx,
            h_profile=hh,
            vpratio_profile=hr,
            h_default=float(self.h_cr.get()),
            vpratio_default=float(self.vp_cr.get()) / max(float(self.vs_cr.get()), 1e-6),
            window_points=self._window_points_value(),
            split_by_sign=True,
            n_iter=2,
            smooth_dense_half_win=self._smooth_dense_half_win_value(),
        )

        out = PhaseDataset()
        # 先复制 shot header
        for s in r.ds_pss.shots:
            out.shots.append(Shot(xshot=s.xshot, tshot=s.tshot, ushot=s.ushot))

        for (si, p), dt in zip(pss_picks, dt_pss):
            if not np.isfinite(dt):
                continue
            t_psp = float(p.t - dt)
            out.shots[si].add_pick(
                x=float(p.x),
                t=t_psp,
                u=float(p.u),
                phase_id=int(self.psp_phase_id.get()),
            )
        # 清理空 shot
        out.shots = [s for s in out.shots if s.picks]
        return out

    def _reload_if_loaded(self) -> None:
        """当关键参数（如 PSP 相位号）变更时，若已加载文件则重算并重绘。"""
        if self.files:
            self._load_and_draw()

    def _build_delta_function(self, r: FileResult, *, mode: str):
        """
        构造 Delta_t(model_distance) 函数。
        theory2d / theory2dequi：仅用 tx.out 上 PSS-PSP，失败为 NaN（不回退 PPS-PPP）。
        """
        mode = mode.lower()
        if mode == "theory2dequi":
            # 生成 tx_2Dequiv.in → r.in 指向其 → 正演 → 用 tx.out 中理论 PPS-PPP 差（同主窗口 2Dequi）
            self._ensure_tx_equi_for_results()
            wd = r.path.parent
            tx_equi = wd / "tx_2Dequiv.in"
            if not tx_equi.exists():
                raise RuntimeError("缺少 tx_2Dequiv.in，请检查数据与 PPS/PSS 比值")
            r_in = wd / "r.in"
            if r_in.exists():
                set_rin_tfile(r_in, "tx_2Dequiv.in")
            fr = run_rayinvr_forward(
                wd,
                force_run=True,
                tx_in_override=tx_equi,
                sync_override_to_tfile=True,
            )
            if not fr.success:
                raise RuntimeError(f"2Dequi 正演失败: {fr.message}")
            tx_out = wd / "tx.out"
            (left_x, left_dt), (right_x, right_dt) = collect_theory2d_delta_by_branch(
                tx_out, phase_ppp=PHASE_PPP, phase_pps=PHASE_PPS
            )
            left_ok = left_x.size >= 2
            right_ok = right_x.size >= 2
            if left_ok or right_ok:

                def _delta_t2dequi(x: float, off: float | None = None) -> float:
                    if not np.isfinite(x):
                        return float("nan")
                    if off is not None and np.isfinite(off):
                        if off < 0 and left_ok:
                            return float(np.interp(x, left_x, left_dt, left=left_dt[0], right=left_dt[-1]))
                        if off >= 0 and right_ok:
                            return float(
                                np.interp(x, right_x, right_dt, left=right_dt[0], right=right_dt[-1])
                            )
                    return float("nan")

                self.theory2d_notice = ""
                return self._delta_fn_pss_psp_from_txout_only(r)

            try:
                bundle = build_theory2d_bundle_from_txout(
                    tx_out,
                    phase_ppp=PHASE_PPP,
                    phase_pps=PHASE_PPS,
                    input_hash="theory2dequi_export",
                )
                if bundle.global_x.size < 2:
                    raise RuntimeError("tx.out 中 PPP/PPS 理论点不足")
                query = make_delta_query(bundle, allow_extrapolation=False)

                def _delta_q(x: float, off: float | None = None) -> float:
                    shot_x = None
                    if off is not None and np.isfinite(off):
                        shot_x = float(x - off)
                    arr = np.asarray(
                        query(
                            np.asarray([x], dtype=float),
                            None if shot_x is None else np.asarray([shot_x], dtype=float),
                        ),
                        dtype=float,
                    )
                    return float(arr[0]) if arr.size else float("nan")

                self.theory2d_notice = ""
                return self._delta_fn_pss_psp_from_txout_only(r)
            except Exception as ex:
                if self.theory2d_auto_fallback.get():
                    self.theory2d_notice = f"theory2Dequi回退picked: {ex}"
                    mode = "picked"
                else:
                    raise RuntimeError(f"theory2Dequi: {ex}") from ex

        if mode == "theory2d":
            # 有任一支理论结果即用分支合并导出，保证左右两侧 PSP 都写入（缺支用另一支回退）
            left_b = self._theory2d_left_branch
            right_b = self._theory2d_right_branch
            if left_b is not None or right_b is not None:
                left_x, left_dt = left_b if left_b is not None else (np.asarray([]), np.asarray([]))
                right_x, right_dt = right_b if right_b is not None else (np.asarray([]), np.asarray([]))
                left_ok = left_x.size >= 2
                right_ok = right_x.size >= 2

                def _delta_branch(x: float, off: float | None = None) -> float:
                    if not np.isfinite(x):
                        return float("nan")
                    # 分支存 model_x，用 off 判断左右支，用 x 插值
                    if off is not None and np.isfinite(off):
                        if off < 0 and left_ok:
                            return float(np.interp(x, left_x, left_dt, left=left_dt[0], right=left_dt[-1]))
                        if off >= 0 and right_ok:
                            return float(np.interp(x, right_x, right_dt, left=right_dt[0], right=right_dt[-1]))
                    return float("nan")

                self.theory2d_notice = ""
                return self._delta_fn_pss_psp_from_txout_only(r)

            # 回退：单次 bundle（未分左右支时）
            bundle, _hit, msg = self._load_or_build_theory2d_bundle(
                r, phase_ppp=PHASE_PPP, phase_pps=PHASE_PPS, auto_run=True
            )
            if bundle is not None:
                query = make_delta_query(bundle, allow_extrapolation=False)

                def _delta(x: float, off: float | None = None) -> float:
                    shot_x = None
                    if off is not None and np.isfinite(off):
                        shot_x = float(x - off)
                    arr = np.asarray(
                        query(
                            np.asarray([x], dtype=float),
                            None if shot_x is None else np.asarray([shot_x], dtype=float),
                        ),
                        dtype=float,
                    )
                    return float(arr[0]) if arr.size else float("nan")

                self.theory2d_notice = ""
                return self._delta_fn_pss_psp_from_txout_only(r)

            if self.theory2d_auto_fallback.get():
                self.theory2d_notice = f"theory2d回退picked: {msg}"
                mode = "picked"
            else:
                raise RuntimeError(f"未找到可用的 2D 理论差值: {msg}")

        if mode != "picked":
            raise ValueError(f"未知导出模式: {mode}")

        # picked 模式（严格按策略）：
        # 1) 对拾取 PPS-PPP 差值（r.diff_pairs）做 model-distance 拟合
        # 2) 在 PSS 的 model distance 处取拟合值
        # 3) 校正 PSS -> PSP
        if not r.diff_pairs:
            raise RuntimeError("无实测 PPS-PPP 差值对")
        # 严格配对点（PPP/PPS 同道）：用于“同点优先”校正
        strict_pairs = list(r.diff_pairs) if r.diff_pairs else []
        strict_map = {}
        for md0, to0, dt0 in strict_pairs:
            if np.isfinite(md0) and np.isfinite(to0) and np.isfinite(dt0):
                strict_map[(round(float(md0), 4), round(float(to0), 4))] = float(dt0)
        arr = np.asarray(r.diff_pairs, dtype=float)  # (model_dist, true_offset, t_diff)
        md = arr[:, 0]
        to = arr[:, 1]
        td = arr[:, 2]
        v = np.isfinite(md) & np.isfinite(td) & np.isfinite(to)
        md = md[v]
        to = to[v]
        td = td[v]
        if md.size < 1:
            raise RuntimeError("实测 PPS-PPP 差值点不足，无法插值")

        def _fit_one_branch(xb: np.ndarray, yb: np.ndarray, xq: float, *, edge_extend_km: float = 8.0) -> float:
            if xb.size == 0:
                return float("nan")
            xmn = float(np.min(xb))
            xmx = float(np.max(xb))
            in_cov = (xmn <= xq <= xmx)
            if (not in_cov) and edge_extend_km <= 0:
                return float("nan")
            if xb.size < 3:
                # 点太少时用常数/最近值，避免丢点
                if (not in_cov) and abs(float(xq) - float(np.clip(xq, xmn, xmx))) > edge_extend_km:
                    return float("nan")
                i0 = int(np.argmin(np.abs(xb - xq)))
                return float(yb[i0])
            # 超出范围时直接钳制到边界，尽量保证有值
            xqc = float(np.clip(xq, xmn, xmx))
            if not in_cov:
                if abs(float(xq) - xqc) > edge_extend_km:
                    return float("nan")
                q = np.asarray([xqc], dtype=float)
            else:
                q = np.asarray([xq], dtype=float)
            yf = fit_ppp_time_curve_local_linear(
                xb,
                yb,
                q,
                window_points=self._window_points_value(),
                split_by_sign=False,
            )
            if np.isfinite(yf[0]):
                return float(yf[0])
            # 回退：在边界钳制位置再算一次，仍失败则取最近值
            yedge = fit_ppp_time_curve_local_linear(
                xb, yb, np.asarray([xqc], dtype=float),
                window_points=self._window_points_value(), split_by_sign=False
            )
            if np.isfinite(yedge[0]):
                return float(yedge[0])
            i0 = int(np.argmin(np.abs(xb - xq)))
            return float(yb[i0])

        x_all = np.asarray(md, dtype=float)
        y_all = np.asarray(td, dtype=float)
        xmn_all = float(np.min(x_all))
        xmx_all = float(np.max(x_all))

        policy_raw = self.picked_policy.get().strip().lower()
        policy_map = {
            "插值": "fit_coverage_only",
            "严格": "strict_only",
            "宽松": "relaxed",
            # 兼容旧英文值
            "fit_coverage_only": "fit_coverage_only",
            "strict_only": "strict_only",
            "relaxed": "relaxed",
        }
        policy = policy_map.get(policy_raw, "fit_coverage_only")

        def _strict_lookup(xq: float, offq: float | None) -> float:
            if offq is None:
                return float("nan")
            k = (round(float(xq), 4), round(float(offq), 4))
            if k in strict_map:
                return float(strict_map[k])
            # 容差匹配（避免浮点微差导致漏匹配）
            best = None
            for (mx, mo), dtv in strict_map.items():
                dx = abs(float(mx) - float(xq))
                do = abs(float(mo) - float(offq))
                if dx <= 0.02 and do <= 0.02:
                    score = dx + do
                    if best is None or score < best[0]:
                        best = (score, dtv)
            return float(best[1]) if best is not None else float("nan")

        def _delta(xq: float, offq: float | None = None) -> float:
            # 若 PPP/PPS/PSS 同点可严格配对，优先使用严格观测时差
            sdt = _strict_lookup(xq, offq)
            if np.isfinite(sdt):
                return sdt
            if policy == "strict_only":
                return float("nan")
            if policy == "fit_coverage_only":
                if not (xmn_all <= xq <= xmx_all):
                    return float("nan")
                return float(_fit_one_branch(x_all, y_all, xq, edge_extend_km=0.0))
            # 按 model-distance 拟合曲线直接取值（与主窗口右上图一致）
            return float(_fit_one_branch(x_all, y_all, xq, edge_extend_km=8.0))

        return _delta


    def after(self, ms: int, callback):
        """Tk-compatible delayed call via QTimer; returns a truthy job token."""
        token = object()
        QTimer.singleShot(int(ms), callback)
        return token

    def _finish_tool_figure(self, win: PlotToolWindow, fig: Figure) -> None:
        win.set_figure(fig)
        win.show_modeless()
