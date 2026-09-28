# -*- coding: utf-8 -*-
"""File load / save / export / record nav mixed into QtFastViewer."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as exc:  # pragma: no cover
    raise RuntimeError("未安装 PySide6") from exc

try:
    import pyqtgraph.exporters as pg_exporters
except Exception:
    pg_exporters = None  # type: ignore

try:
    from ...core.pick_manager import PickManager
except ImportError:  # pragma: no cover
    from pyAOBS.visualization.zplotpy.core.pick_manager import PickManager


class FileIoMixin:
    """文件选择、加载、参数存取、导出与炮号导航。"""

    def _init_debug_log_file(self) -> None:
        """初始化调试日志文件。"""
        if not bool(getattr(self, "_debug_log_enabled", False)):
            return
        try:
            self._debug_log_path = Path.cwd() / "zplotpy_denoise_debug.log"
            self._debug_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._debug_log_path.open("a", encoding="utf-8") as f:
                f.write("\n")
                f.write(f"===== QtFastViewer session {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        except Exception:
            try:
                self._debug_log_path = Path(tempfile.gettempdir()) / "zplotpy_denoise_debug.log"
                self._debug_log_path.parent.mkdir(parents=True, exist_ok=True)
                with self._debug_log_path.open("a", encoding="utf-8") as f:
                    f.write("\n")
                    f.write(f"===== QtFastViewer session {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            except Exception:
                self._debug_log_enabled = False


    def _debug_log(self, tag: str, message: str) -> None:
        """写入调试日志（持久化）。"""
        if not bool(getattr(self, "_debug_log_enabled", False)):
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{str(tag)}] {str(message)}"
        if line == self._debug_last_line:
            return
        self._debug_last_line = line
        try:
            with self._debug_log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


    def _set_status_text(self, text: str, hold_ms: int = 0, force: bool = False) -> None:
        """设置状态栏文本；可选短时保留，避免被高频渲染状态覆盖。"""
        now_ms = int(time.monotonic() * 1000.0)
        if (not force) and now_ms < int(self._status_hold_until_ms):
            self._debug_log("STATUS_SKIP", str(text))
            return
        self.lbl_status.setText(str(text))
        self._debug_log("STATUS", str(text))
        if int(hold_ms) > 0:
            self._status_hold_until_ms = now_ms + int(hold_ms)
        else:
            self._status_hold_until_ms = 0


    def _choose_dfile(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 Z 数据文件", "", "Z files (*.z);;All files (*)", options=self._file_dialog_options()
        )
        if not path:
            return
        self._dfile = path
        self._update_file_open_status_label()
        self._load_data()


    def _choose_hfile(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择头文件", "", "Header files (*.hdr *.h);;All files (*)", options=self._file_dialog_options()
        )
        if path:
            self._hfile = path
            self._update_file_open_status_label()
            if self._dfile:
                self._load_data()


    def _choose_rfile(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择记录文件", "", "Record files (*.r *.txt);;All files (*)", options=self._file_dialog_options()
        )
        if path:
            self._rfile = path
            self._update_file_open_status_label()
            if self._dfile:
                self._load_data()


    def _save_parameters(self) -> None:
        out, _ = self._get_save_file_name(
            "保存参数配置",
            "",
            "JSON files (*.json);;All files (*)",
            default_suffix=".json",
        )
        if not out:
            return
        payload = {
            "version": 1,
            "dfile": self._dfile,
            "hfile": self._hfile,
            "rfile": self._rfile,
            "parameters": self._collect_ui_parameters(),
        }
        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self.lbl_status.setText(f"参数已保存：{out}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "保存失败", f"参数保存失败：{exc}")


    def _load_parameters(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "加载参数配置", "", "JSON files (*.json);;All files (*)", options=self._file_dialog_options()
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            conf = payload.get("parameters", payload)
            if not isinstance(conf, dict):
                raise ValueError("参数配置格式不正确")
            self._apply_ui_parameters(conf)
            if self.loaded is not None:
                self.request_render(immediate=True)
            self.lbl_status.setText(f"参数已加载：{path}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "加载失败", f"参数加载失败：{exc}")


    def _export_figure(self) -> None:
        out, selected = self._get_save_file_name(
            "导出图像",
            "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;PDF (*.pdf);;PostScript (*.ps)",
            default_suffix=".png",
        )
        if not out:
            return
        try:
            suffix = Path(out).suffix.lower()
            # 多扩展过滤器下 Path.suffix 对 .jpeg 等已够用；若仍无则按过滤器兜底
            if not suffix:
                from pyAOBS.utils.qt_file_dialog import ensure_save_suffix

                out = ensure_save_suffix(out, selected, default_suffix=".png")
                suffix = Path(out).suffix.lower()

            # 使用 PlotItem 导出，避免 OpenGL/抓屏导致空白图
            exporter = pg_exporters.ImageExporter(self.plot.getPlotItem())
            if suffix in (".png", ".jpg", ".jpeg", ".bmp"):
                exporter.export(out)
            elif suffix in (".pdf", ".ps"):
                # 先离屏导出到临时 png，再写入 pdf/ps
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                    tmp_png = tf.name
                exporter.export(tmp_png)
                import matplotlib.pyplot as plt
                import matplotlib.image as mpimg
                arr = mpimg.imread(tmp_png)
                h, w = int(arr.shape[0]), int(arr.shape[1])

                fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100)
                ax = fig.add_axes([0, 0, 1, 1])
                ax.imshow(arr)
                ax.axis("off")
                fig.savefig(out, format=suffix.lstrip("."), dpi=300, bbox_inches="tight", pad_inches=0)
                plt.close(fig)
                try:
                    os.remove(tmp_png)
                except Exception:
                    pass
            else:
                raise RuntimeError(f"不支持的导出格式: {suffix}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "导出失败", f"图像导出失败：{exc}")
            return
        self.lbl_status.setText(f"图像已导出：{out}")


    def _available_records(self) -> List[int]:
        if self.loaded is None:
            return []
        headers = self.loaded.get("trace_headers", [])
        recs = sorted({int(getattr(h, "ishoti", 0) or 0) for h in headers if int(getattr(h, "ishoti", 0) or 0) > 0})
        return recs


    def _prev_record(self) -> None:
        recs = self._available_records()
        if not recs:
            return
        cur = int(self.spin_irec.value())
        if cur <= 0:
            self.spin_irec.setValue(recs[-1])
            return
        smaller = [r for r in recs if r < cur]
        self.spin_irec.setValue(smaller[-1] if smaller else recs[-1])


    def _next_record(self) -> None:
        recs = self._available_records()
        if not recs:
            return
        cur = int(self.spin_irec.value())
        if cur <= 0:
            self.spin_irec.setValue(recs[0])
            return
        bigger = [r for r in recs if r > cur]
        self.spin_irec.setValue(bigger[0] if bigger else recs[0])


    def _show_trace_info(self) -> None:
        if self.loaded is None:
            self.lbl_status.setText("提示：请先加载数据")
            return
        offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
        if offsets.size == 0:
            self.lbl_status.setText("提示：没有可用道数据")
            return
        idx = self._extract_indices()
        if idx.size == 0:
            self.lbl_status.setText("提示：当前过滤条件下无道")
            return
        xref = self.mouse_x
        yref = self.mouse_y
        if xref is None or yref is None:
            vr = self.plot.getViewBox().viewRange()
            xref = float((vr[0][0] + vr[0][1]) * 0.5)
            yref = float((vr[1][0] + vr[1][1]) * 0.5)
        idx_offsets = offsets[idx]
        nearest_i = int(np.argmin(np.abs(idx_offsets - float(xref))))
        trace_idx = int(idx[nearest_i])
        headers = self.loaded.get("trace_headers", [])
        th = headers[trace_idx] if (trace_idx < len(headers)) else None

        shot = int(getattr(th, "ishoti", 0) or 0) if th is not None else 0
        rec = int(getattr(th, "ireci", 0) or 0) if th is not None else 0
        itsn = int(getattr(th, "itsn", trace_idx) or trace_idx) if th is not None else trace_idx
        dead = bool(int(getattr(th, "iflagi", 1) or 1) != 1) if th is not None else False
        azi = float(getattr(th, "azi", 0.0) or 0.0) if th is not None else 0.0
        off = float(offsets[trace_idx])
        cursor_time_display = float(yref)
        cursor_time_true = cursor_time_display - self._compute_display_tshift(trace_idx, off)

        pick_lines: List[str] = []
        if th is not None and getattr(th, "picks", None):
            for pi, pv in enumerate(getattr(th, "picks", []), start=1):
                if float(pv) > 0:
                    pick_lines.append(f"拾取字{pi}: {float(pv):.4f} s")
        if not pick_lines and self.pick_manager is not None:
            for pi in range(1, int(self.spin_apick.maximum()) + 1):
                pt = self.pick_manager.get_pick(trace_idx, pi)
                if pt is not None and float(pt) > 0:
                    pick_lines.append(f"拾取字{pi}: {float(pt):.4f} s")
        picks_text = "\n".join(pick_lines) if pick_lines else "无"

        msg = (
            f"道索引: {trace_idx}\n"
            f"炮站号(ishoti): {shot if shot > 0 else 'N/A'}\n"
            f"接收站号(ireci): {rec if rec > 0 else 'N/A'}\n"
            f"道序号(itsn): {itsn}\n"
            f"死道标志: {'是' if dead else '否'}\n"
            f"炮检距(offset): {off:.4f} km\n"
            f"方位角(azi): {azi:.2f}°\n"
            f"光标时间(显示): {cursor_time_display:.4f} s\n"
            f"光标时间(真实): {cursor_time_true:.4f} s\n\n"
            f"拾取时间:\n{picks_text}"
        )
        self._show_themed_info("道信息", msg)
        self.lbl_status.setText(
            f"道 {trace_idx}: offset={off:.3f}km, 炮站={shot or 'N/A'}, 显示={cursor_time_display:.3f}s, 真实={cursor_time_true:.3f}s"
        )


    def _update_file_open_status_label(self) -> None:
        """更新右下角文件打开状态与道头来源（持久显示）。"""
        if not hasattr(self, "lbl_pick_link_mode"):
            return
        parts: List[str] = []
        if self._dfile and Path(self._dfile).exists():
            parts.append(f"{Path(self._dfile).name} 打开")
        elif self._dfile:
            parts.append(".z关闭")

        if self._hfile and Path(self._hfile).exists():
            parts.append(f"{Path(self._hfile).name} 打开")
        else:
            parts.append(".hdr关闭")

        # 若尚未选择任何文件，右侧留空
        if (not self._dfile) and (not self._hfile):
            self.lbl_pick_link_mode.setText("")
        else:
            self.lbl_pick_link_mode.setText(" / ".join(parts))


    def _load_data(self) -> None:
        if not self._dfile:
            return
        t0 = time.perf_counter()
        try:
            self.loaded = self.loader.load_z_format(
                self._dfile,
                hfile=self._hfile if self._hfile and Path(self._hfile).exists() else None,
                rfile=self._rfile if self._rfile and Path(self._rfile).exists() else None,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "加载失败", str(exc))
            return
        self._clear_denoise_cache()

        header = self.loaded.get("header")
        ntr = int(getattr(header, "ntraces", 0) or 0)
        npts = int(getattr(header, "npts", 0) or 0)
        # irec 过滤实际按 trace_headers.ishoti（炮号）匹配，不一定是 1..nrec 连续编号。
        # 若上限仅设为 nrec，可能导致“下一炮”目标值被 QSpinBox 截断到无效炮号，进而无波形。
        headers = self.loaded.get("trace_headers", [])
        recs = [
            int(getattr(h, "ishoti", 0) or 0)
            for h in headers
            if int(getattr(h, "ishoti", 0) or 0) > 0
        ]
        max_shot_id = max(recs) if recs else 0
        nrec_header = int(getattr(header, "nrec", 0) or 0)
        self.spin_irec.setMaximum(max(0, nrec_header, max_shot_id))
        self.btn_reload.setEnabled(True)
        self.btn_save_params.setEnabled(True)
        self.btn_save_z.setEnabled(True)
        self.btn_data_info.setEnabled(True)
        self.btn_location_map.setEnabled(True)
        self.btn_export_fig.setEnabled(True)
        self.btn_prev_rec.setEnabled(True)
        self.btn_next_rec.setEnabled(True)
        self.btn_theory.setEnabled(True)
        self.btn_clear_theory.setEnabled(True)
        self.btn_water_corr.setEnabled(True)
        self.btn_clear_water.setEnabled(True)
        self.btn_water_curve.setEnabled(True)
        self.btn_load_txin.setEnabled(True)
        self.btn_clear_txin.setEnabled(True)
        self.btn_preview_map_txin.setEnabled(True)
        self.btn_map_txin.setEnabled(True)
        self.chk_map_txin_apick_only.setEnabled(True)
        self.chk_map_txin_view_only.setEnabled(True)
        self.spin_map_txin_tol.setEnabled(True)
        self.btn_save_picks.setEnabled(True)
        self.btn_save_hdr.setEnabled(True)
        self.btn_write_txin.setEnabled(True)
        self.btn_clear_picks.setEnabled(True)
        self.btn_auto_pick.setEnabled(True)
        self.btn_interp_pick.setEnabled(True)
        self.btn_align_pick.setEnabled(True)
        self.btn_align_adaptive.setEnabled(True)
        self.btn_eval_stack.setEnabled(True)
        self.btn_waveop_stack.setEnabled(True)
        # 姿态按钮仅在 RelocationViewer（host 模式）显示；独立 zplotpy 保持隐藏
        if bool(getattr(self, "_relocation_host_mode", False)):
            self.btn_waveop_att.setEnabled(True)
        self.btn_waveop_clear.setEnabled(True)
        self.btn_waveop_save.setEnabled(True)
        self.btn_waveop_load.setEnabled(True)
        self.btn_static_corr.setEnabled(True)
        self.btn_clear_static.setEnabled(True)
        self.btn_clear_align.setEnabled(True)
        self.plot.clear()
        self._curve_items.clear()
        self._shade_item = None
        self._density_item = None
        self._density_hl_item = None
        self._pick_item = None
        self._stack_item = None
        self._static_preview_item = None
        self._theoretical_item = None
        self._txin_item = None
        self._txin_map_preview_item = None
        self._water_corr_item = None
        self._wave_select_items = {}
        self._wave_select_marker_item = None
        self._waveop_stack_item = None
        self._mute_polygon_item = None
        self._mute_vertex_item = None
        self._did_initial_view_fit = False
        self._alignment_offsets = {}
        self.static_corrector.clear_corrections()
        self.static_correction_enabled = False
        self.static_preview_mode = False
        self.last_stacking_result = None
        self.theoretical_traveltime_calculator = None
        self.theoretical_times_data = None
        self.show_theoretical_times = False
        self.txin_overlay_data = None
        self.show_txin_overlay = False
        self.txin_map_preview_data = None
        self.water_layer_corrections = {}
        self.water_layer_corrected_times = None
        self.show_water_layer_correction = False
        self._removed_traces = set()
        self.waveform_selections = []
        self.waveop_stack_result = None
        self._mute_edit_mode = False
        self._mute_enabled = False
        self._mute_invert = False
        self._mute_polygon_points = []
        self._mute_drag_vertex_idx = None
        self._mute_selected_vertex_idx = None
        self._mute_drag_active = False
        self._mute_drag_last_status_ms = 0
        self._sync_plot_pan_lock_state()
        self._update_mute_status_button()
        self._orientation_preview_enabled = False
        self._orientation_preview_solution = {}
        self._orientation_preview_cache = None
        if hasattr(self, "_set_orientation_main_preview"):
            try:
                self._set_orientation_main_preview(False, keep_cache=False)
            except Exception:
                pass
        self._refresh_waveop_selection_list()
        npick = int(getattr(header, "npick", 10) or 10)
        self.pick_manager = PickManager(npick=npick)
        self._pick_undo_stack.clear()
        self._pick_redo_stack.clear()
        self._update_undo_button_state()
        self.spin_apick.setMaximum(max(1, npick))
        self.spin_apick.setValue(min(self.spin_apick.value(), max(1, npick)))
        offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
        times = np.asarray(self.loaded.get("times", []), dtype=float)
        if offsets.size > 0:
            self.spin_xmin.setValue(float(np.min(offsets)))
            self.spin_xmax.setValue(float(np.max(offsets)))
        if times.size > 0:
            self.spin_tmin.setValue(float(np.min(times)))
            self.spin_tmax.setValue(float(np.max(times)))
        self._sync_window_controls_from_view()
        trace_headers = self.loaded.get("trace_headers", [])
        if self.pick_manager is not None and trace_headers:
            self.pick_manager.set_trace_info_batch(trace_headers)
        self.request_render(immediate=True)
        dt = (time.perf_counter() - t0) * 1000.0
        self._update_file_open_status_label()
        self.lbl_status.setText(f"加载完成: ntr={ntr}, npts={npts}, {dt:.1f} ms")

