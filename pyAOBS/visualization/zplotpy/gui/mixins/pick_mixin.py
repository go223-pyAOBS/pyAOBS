# -*- coding: utf-8 -*-
"""Pick / TXIN / alignment logic mixed into QtFastViewer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as exc:  # pragma: no cover
    raise RuntimeError("未安装 PySide6") from exc

try:
    import pyqtgraph as pg
except Exception:
    pg = None  # type: ignore

try:
    from ...core.hdr_to_tx import Z2TxConfig, convert_hdr_to_tx
except ImportError:  # pragma: no cover
    from pyAOBS.visualization.zplotpy.core.hdr_to_tx import Z2TxConfig, convert_hdr_to_tx


class PickMixin:
    """人工/自动拾取、撤销、TXIN 叠加与对齐。"""

    def _toggle_pick_mode(self) -> None:
        self.chk_pick_mode.setChecked(not self.chk_pick_mode.isChecked())
        self.lbl_status.setText("拾取模式已开启" if self.chk_pick_mode.isChecked() else "拾取模式已关闭")


    def _on_apick_changed(self, _value: int) -> None:
        self._refresh_waveop_selection_list()
        apick = int(self.spin_apick.value())
        self._set_status_text(f"当前拾取字: {apick}，V段: {self._waveop_apick_segment_count(apick)}", hold_ms=1200)


    def _set_apick_from_shortcut(self, pick_word: int) -> None:
        if self.spin_apick.maximum() <= 0:
            return
        val = int(max(1, min(self.spin_apick.maximum(), int(pick_word))))
        self.spin_apick.setValue(val)
        self.lbl_status.setText(f"当前拾取字: {val}，V段: {self._waveop_apick_segment_count(val)}")


    def _shift_apick(self, delta: int) -> None:
        cur = int(self.spin_apick.value())
        nxt = cur + int(delta)
        nxt = max(1, min(int(self.spin_apick.maximum()), nxt))
        if nxt != cur:
            self.spin_apick.setValue(nxt)
        apick = int(self.spin_apick.value())
        self.lbl_status.setText(f"当前拾取字: {apick}，V段: {self._waveop_apick_segment_count(apick)}")


    def _snapshot_pick_state(self) -> Dict[int, Dict[int, float]]:
        if self.pick_manager is None:
            return {}
        snap = self.pick_manager.get_all_picks()
        return {
            int(trace_idx): {int(word): float(tpk) for word, tpk in by_word.items()}
            for trace_idx, by_word in snap.items()
        }


    def _push_pick_undo(self, reason: str) -> None:
        if self.pick_manager is None:
            return
        self._pick_undo_stack.append((str(reason), self._snapshot_pick_state()))
        if len(self._pick_undo_stack) > int(self._pick_undo_limit):
            self._pick_undo_stack = self._pick_undo_stack[-int(self._pick_undo_limit):]
        self._pick_redo_stack.clear()
        self._update_undo_button_state()


    def _restore_pick_state(self, snapshot: Dict[int, Dict[int, float]]) -> None:
        if self.pick_manager is None:
            return
        self.pick_manager.clear_picks()
        for trace_idx, by_word in snapshot.items():
            for pick_word, tpk in by_word.items():
                self.pick_manager.add_pick(int(trace_idx), float(tpk), int(pick_word))


    def _update_undo_button_state(self) -> None:
        has_undo = self.pick_manager is not None and len(self._pick_undo_stack) > 0
        has_redo = self.pick_manager is not None and len(self._pick_redo_stack) > 0
        self.btn_undo_pick.setEnabled(bool(has_undo))
        self.btn_redo_pick.setEnabled(bool(has_redo))


    def _undo_last_pick_edit(self) -> None:
        if self.pick_manager is None:
            return
        if not self._pick_undo_stack:
            self.lbl_status.setText("撤销失败：没有可撤销的拾取操作")
            self._update_undo_button_state()
            return
        reason, snapshot = self._pick_undo_stack.pop()
        self._pick_redo_stack.append((reason, self._snapshot_pick_state()))
        if len(self._pick_redo_stack) > int(self._pick_undo_limit):
            self._pick_redo_stack = self._pick_redo_stack[-int(self._pick_undo_limit):]
        self._restore_pick_state(snapshot)
        self._update_undo_button_state()
        self.request_render(delay_ms=10)
        self.lbl_status.setText(f"已撤销：{reason}")


    def _redo_last_pick_edit(self) -> None:
        if self.pick_manager is None:
            return
        if not self._pick_redo_stack:
            self.lbl_status.setText("重做失败：没有可重做的拾取操作")
            self._update_undo_button_state()
            return
        reason, snapshot = self._pick_redo_stack.pop()
        self._pick_undo_stack.append((reason, self._snapshot_pick_state()))
        if len(self._pick_undo_stack) > int(self._pick_undo_limit):
            self._pick_undo_stack = self._pick_undo_stack[-int(self._pick_undo_limit):]
        self._restore_pick_state(snapshot)
        self._update_undo_button_state()
        self.request_render(delay_ms=10)
        self.lbl_status.setText(f"已重做：{reason}")


    def _ensure_txin_item(self) -> None:
        if self._txin_item is None:
            self._txin_item = pg.ScatterPlotItem(
                size=8.0,
                pen=pg.mkPen(self._theme_color("txin_pen", "#7c3aed"), width=1),
                pxMode=True,
            )
            self._txin_item.setZValue(36)
            self.plot.addItem(self._txin_item)


    def _clear_txin_item(self) -> None:
        if self._txin_item is not None:
            self._txin_item.setData([], [])


    def _ensure_txin_map_preview_item(self) -> None:
        if self._txin_map_preview_item is None:
            self._txin_map_preview_item = pg.ScatterPlotItem(
                size=10.0,
                pen=pg.mkPen(self._theme_color("txin_preview_pen", "#f97316"), width=1.5),
                brush=pg.mkBrush(0, 0, 0, 0),
                symbol="x",
                pxMode=True,
            )
            self._txin_map_preview_item.setZValue(37)
            self.plot.addItem(self._txin_map_preview_item)


    def _clear_txin_map_preview_item(self) -> None:
        if self._txin_map_preview_item is not None:
            self._txin_map_preview_item.setData([], [])


    def _ensure_pick_item(self) -> None:
        if self._pick_item is None:
            self._pick_item = pg.ScatterPlotItem(
                size=float(self.spin_pick_size.value()),
                pen=pg.mkPen(self._theme_color("pick_pen", "#dc1e1e"), width=1),
                brush=pg.mkBrush(self._theme_color("pick_brush", "#ff7878")),
                pxMode=True,
            )
            self._pick_item.setZValue(50)
            self.plot.addItem(self._pick_item)


    def _render_picks(self, offsets_all: np.ndarray, allowed_trace_indices: Optional[np.ndarray] = None) -> int:
        if self.pick_manager is None:
            return 0
        picks_all = self._build_pick_snapshot(use_orientation_corrected=False)
        if not picks_all:
            if self._pick_item is not None:
                self._pick_item.setData([], [])
            return 0

        apick = int(self.spin_apick.value())
        x_range, y_range = self.plot.getViewBox().viewRange()
        xmin, xmax = min(x_range), max(x_range)
        ymin, ymax = min(y_range), max(y_range)
        allowed_set = None
        if allowed_trace_indices is not None and len(allowed_trace_indices) > 0:
            allowed_set = set(int(i) for i in np.asarray(allowed_trace_indices, dtype=int))
        headers = self.loaded.get("trace_headers", []) if self.loaded is not None else []
        group_union: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
        if headers:
            for trace_idx, by_word in picks_all.items():
                if trace_idx < 0 or trace_idx >= len(headers):
                    continue
                th = headers[trace_idx]
                key = (int(getattr(th, "ishoti", 0) or 0), int(getattr(th, "ireci", 0) or 0))
                if key[0] <= 0 or key[1] <= 0:
                    continue
                arr = group_union.setdefault(key, [])
                for pick_word, t_raw in by_word.items():
                    if float(t_raw) > 0.0:
                        arr.append((int(pick_word), float(t_raw)))
        spots = []
        trace_iter: List[int]
        if allowed_set is not None:
            trace_iter = sorted(allowed_set)
        else:
            trace_iter = sorted(int(k) for k in picks_all.keys())
        for trace_idx in trace_iter:
            if allowed_set is not None and int(trace_idx) not in allowed_set:
                continue
            if trace_idx < 0 or trace_idx >= len(offsets_all):
                continue
            x = float(offsets_all[trace_idx])
            if x < xmin or x > xmax:
                continue
            tshift = self._compute_display_tshift(int(trace_idx), x)
            own = picks_all.get(int(trace_idx), {})
            merged_vals: List[Tuple[int, float]] = []
            for pick_word, t_raw in own.items():
                merged_vals.append((int(pick_word), float(t_raw)))
            if headers and trace_idx < len(headers):
                th = headers[trace_idx]
                key = (int(getattr(th, "ishoti", 0) or 0), int(getattr(th, "ireci", 0) or 0))
                for pick_word, t_raw in group_union.get(key, []):
                    merged_vals.append((int(pick_word), float(t_raw)))
            seen = set()
            for pick_word, t_raw in merged_vals:
                sig = (int(pick_word), round(float(t_raw), 6))
                if sig in seen:
                    continue
                seen.add(sig)
                t = float(t_raw) + tshift
                if t < ymin or t > ymax:
                    continue
                # 不同拾取字用不同颜色；当前活动字视觉上更突出
                color = pg.intColor(int(pick_word), hues=48, values=1, alpha=220)
                is_active = int(pick_word) == apick
                spots.append(
                    {
                        "pos": (x, t),
                        "data": {"trace_idx": int(trace_idx), "pick_word": int(pick_word)},
                        "brush": pg.mkBrush(color),
                        "pen": pg.mkPen(
                            self._theme_color("pick_active_edge", "#ffffff") if is_active else self._theme_color("pick_edge", "#505050"),
                            width=1.2 if is_active else 0.8,
                        ),
                        "size": 12.0 if is_active else 9.0,
                    }
                )

        self._ensure_pick_item()
        self._pick_item.setSize(float(self.spin_pick_size.value()))
        if spots:
            self._pick_item.setData(spots=spots)
        else:
            self._pick_item.setData([], [])
        return len(spots)


    def _get_shared_pick(self, trace_idx: int, pick_word: int) -> Optional[float]:
        if self.pick_manager is None:
            return None
        group = self._trace_group_indices(int(trace_idx))
        # Prefer current trace pick, fallback to any component in same group.
        val = self.pick_manager.get_pick(int(trace_idx), int(pick_word))
        if val is not None:
            return float(val)
        for gi in group:
            v = self.pick_manager.get_pick(int(gi), int(pick_word))
            if v is not None:
                return float(v)
        return None


    def _set_shared_pick(self, trace_idx: int, pick_word: int, t_pick: float) -> bool:
        if self.pick_manager is None:
            return False
        group = self._trace_group_indices(int(trace_idx))
        # Keep only one canonical pick per (shot,receiver,pick_word), editable from any component.
        target = int(trace_idx)
        for gi in group:
            if self.pick_manager.get_pick(int(gi), int(pick_word)) is not None:
                target = int(gi)
                break
        for gi in group:
            if int(gi) != target:
                self.pick_manager.remove_pick(int(gi), int(pick_word))
        return bool(self.pick_manager.add_pick(int(target), float(t_pick), int(pick_word)))


    def _remove_shared_pick(self, trace_idx: int, pick_word: int) -> bool:
        if self.pick_manager is None:
            return False
        group = self._trace_group_indices(int(trace_idx))
        removed = False
        for gi in group:
            removed = bool(self.pick_manager.remove_pick(int(gi), int(pick_word))) or removed
        return removed


    def _sync_picks_into_trace_headers(self, use_orientation_corrected: bool = False) -> bool:
        """将 pick_manager 的拾取写回当前 trace_headers.picks。"""
        if self.loaded is None or self.pick_manager is None:
            return False
        trace_headers = self.loaded.get("trace_headers", [])
        header = self.loaded.get("header")
        if not trace_headers or header is None:
            return False
        npick = int(getattr(header, "npick", 0) or 0)
        if npick <= 0:
            return False
        all_picks = self._build_pick_snapshot(use_orientation_corrected=use_orientation_corrected)
        for i, th in enumerate(trace_headers):
            picks_arr = [0.0] * npick
            by_word = all_picks.get(int(i), {})
            for pick_word, tpk in by_word.items():
                pw = int(pick_word)
                if 1 <= pw <= npick and float(tpk) > 0.0:
                    picks_arr[pw - 1] = float(tpk)
            th.picks = picks_arr
        return True


    def _save_z_with_picks(self) -> None:
        if self.loaded is None or self._dfile is None:
            self.lbl_status.setText("保存.z失败：请先加载 .z 数据")
            return
        use_corr = self._ask_pick_save_mode("保存 .z")
        if not self._sync_picks_into_trace_headers(use_orientation_corrected=bool(use_corr)):
            self.lbl_status.setText("保存.z失败：当前数据无可写入的道头/拾取信息")
            return
        msg = QtWidgets.QMessageBox(self)
        msg.setIcon(QtWidgets.QMessageBox.Icon.Question)
        msg.setWindowTitle("保存 .z")
        msg.setText("请选择保存方式：")
        btn_overwrite = msg.addButton("覆盖当前 .z（默认）", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        btn_save_as = msg.addButton("另存为新 .z（写入处理后波形）", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        btn_cancel = msg.addButton("取消", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(btn_overwrite)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked == btn_cancel or clicked is None:
            return

        out = str(self._dfile)
        use_processed_traces = False
        if clicked == btn_save_as:
            default_name = Path(self._dfile).with_name(f"{Path(self._dfile).stem}_picked.z")
            out_path, _ = self._get_save_file_name(
                "另存为 .z（含拾取）",
                str(default_name),
                "Z files (*.z)",
                default_suffix=".z",
            )
            if not out_path:
                return
            out = out_path
            use_processed_traces = True

        original_loader_traces = self.loader.traces
        original_loader_header = self.loader.header
        original_loader_trace_headers = self.loader.trace_headers
        try:
            if use_processed_traces:
                traces = self.loaded.get("traces", [])
                times = np.asarray(self.loaded.get("times", []), dtype=float)
                offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
                if len(traces) == 0 or times.size < 2 or offsets.size == 0:
                    self.lbl_status.setText("另存为.z失败：处理后波形不可用")
                    return
                proc_params = ZPlotParameters()
                proc_params.amp = float(self.spin_amp.value())
                proc_params.iscale = int(self.combo_iscale.currentIndex())
                proc_params.rcor = float(self.spin_rcor.value())
                proc_params.sf = float(self.spin_sf.value())
                proc_params.tvg = float(self.spin_tvg.value())
                proc_params.pvg = float(self.spin_pvg.value())
                proc_params.clip = float(self.spin_clip.value())
                proc_params.ibndps = 1 if self.chk_filter.isChecked() else 0
                proc_params.freqlo = float(self.spin_freqlo.value())
                proc_params.freqhi = float(self.spin_freqhi.value())
                proc_params.npoles = int(self.spin_npoles.value())
                proc_params.izerop = 1 if self.chk_zerop.isChecked() else 0
                proc_params.rmean = 1 if self.chk_rmean.isChecked() else 0
                proc_params.rtrend = 1 if self.chk_rtrend.isChecked() else 0
                proc_params.iout = 0 if proc_params.rmean else 2
                proc_params.vred = float(self.spin_vred.value())
                proc_params.gain_on = 1 if bool(self.chk_gain.isChecked()) else 0
                trace_headers = self.loaded.get("trace_headers", [])
                gains = np.ones(len(traces), dtype=float)
                for i, th in enumerate(trace_headers):
                    if i < gains.size:
                        gains[i] = float(max(1, int(getattr(th, "igaini", 1) or 1)))
                sr = 1.0 / float(times[1] - times[0])
                processed_traces = self.processor.process_traces(
                    traces=[np.asarray(t) for t in traces],
                    times=times,
                    offsets=offsets,
                    params=proc_params,
                    gains=gains,
                    sampling_rate=sr,
                    realtime_interaction=False,
                )
                self.loader.traces = processed_traces

            # 统一使用当前 loaded 的头信息与道头（含已同步 picks）
            self.loader.header = self.loaded.get("header")
            self.loader.trace_headers = self.loaded.get("trace_headers", [])
            # 覆盖写回时经临时文件替换，避免懒加载源被 wb 截断后读到 0 点
            ok = self.loader.save_z_format(out, hfile=None, write_picks_to_data=True)
        except Exception as exc:
            self.lbl_status.setText(f"保存.z失败：{exc}")
            return
        finally:
            self.loader.traces = original_loader_traces
            self.loader.header = original_loader_header
            self.loader.trace_headers = original_loader_trace_headers
            # 与 loader 共享懒加载对象时，保存后缓存已清空，保持 loaded 引用一致
            if self.loaded is not None and self.loader.traces is not None:
                self.loaded["traces"] = self.loader.traces

        mode_text = "校正后走时" if use_corr else "原始走时"
        if clicked == btn_overwrite:
            self.lbl_status.setText(f".z 已覆盖保存（{mode_text}）：{out}" if ok else "保存.z失败")
        else:
            self.lbl_status.setText(f".z 另存成功（处理后波形+{mode_text}）：{out}" if ok else "保存.z失败")


    def _save_picks(self) -> None:
        if self.pick_manager is None:
            return
        use_corr = self._ask_pick_save_mode("保存拾取")
        out, _ = self._get_save_file_name(
            "保存拾取",
            "",
            "zplot.out (*.out *.txt);;All files (*)",
            default_suffix=".out",
            preferred_suffix=".out",
        )
        if not out:
            return
        backup = self._snapshot_pick_state()
        try:
            if use_corr:
                self.pick_manager.clear_picks()
                corr = self._build_pick_snapshot(use_orientation_corrected=True)
                for trace_idx, by_word in corr.items():
                    for pick_word, tpk in by_word.items():
                        self.pick_manager.add_pick(int(trace_idx), float(tpk), int(pick_word))
            ok = self.pick_manager.save_picks(out, format="zplot")
        finally:
            self._restore_pick_state(backup)
        mode_text = "校正后走时" if use_corr else "原始走时"
        self.lbl_status.setText(f"拾取已保存（{mode_text}）" if ok else "拾取保存失败")


    def _save_picks_to_hdr(self) -> bool:
        if self.pick_manager is None or self.loaded is None:
            return False
        use_corr = self._ask_pick_save_mode("写入 HDR")
        self._sync_picks_into_trace_headers(use_orientation_corrected=bool(use_corr))
        trace_headers = self.loaded.get("trace_headers", [])
        if not trace_headers:
            self.lbl_status.setText("写入HDR失败：无道头信息")
            return False
        hfile = self._hfile
        if not hfile:
            hfile, _ = self._get_save_file_name(
                "选择HDR输出文件",
                "",
                "Header files (*.hdr)",
                default_suffix=".hdr",
            )
            if not hfile:
                return False
            self._hfile = hfile
        ok = self.pick_manager.save_to_header_file(hfile, trace_headers)
        mode_text = "校正后走时" if use_corr else "原始走时"
        self.lbl_status.setText(f"拾取已写入HDR（{mode_text}）" if ok else "写入HDR失败")
        return bool(ok)


    def _write_txin(self) -> None:
        """写入 tx.in：建议先「写入HDR」；若 HDR 未就绪则先同步写入再转换。

        入口在工具栏「写入HDR」旁，不在「走时模板」面板（该面板只管读/映射 tx.in）。
        """
        if self.loaded is None:
            self.lbl_status.setText("写入tx.in失败：请先加载数据")
            return
        hdr_ready = bool(self._hfile) and Path(self._hfile).exists()
        if not hdr_ready:
            reply = QtWidgets.QMessageBox.question(
                self,
                "写入 tx.in",
                "尚未写入 HDR。将先执行「写入HDR」，再生成 tx.in。是否继续？",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.Yes,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            if not self._save_picks_to_hdr():
                return
        else:
            # HDR 已存在：仍把当前拾取同步进 HDR，避免界面拾取与磁盘 HDR 不一致
            if not self._save_picks_to_hdr():
                return
        if not self._hfile:
            self.lbl_status.setText("写入tx.in失败：未设置HDR文件")
            return

        header = self.loaded.get("header")
        npick = int(getattr(header, "npick", 0) or 0)
        if npick <= 0:
            self.lbl_status.setText("写入tx.in失败：无有效npick")
            return

        out, _ = self._get_save_file_name(
            "写入 tx.in",
            "",
            "tx.in (*.in)",
            default_suffix=".in",
        )
        if not out:
            return

        trace_headers = self.loaded.get("trace_headers", [])
        shots = sorted(
            {
                int(getattr(th, "ishoti", 0) or 0)
                for th in trace_headers
                if int(getattr(th, "ishoti", 0) or 0) > 0
            }
        )
        if not shots:
            shots = sorted(set(int(getattr(th, "ishoti", 0) or 0) for th in trace_headers))
        cfg = self._prompt_tx_config(shots=shots, npick=npick)
        if cfg is None:
            self.lbl_status.setText("tx.in写入已取消")
            return
        ok, total = convert_hdr_to_tx(self._hfile, out, cfg, npick)
        self.lbl_status.setText(f"tx.in已写入：{total} picks → {out}" if ok else "tx.in写入失败")

    # 兼容旧名
    def _export_txin(self) -> None:
        self._write_txin()


    def _load_txin_overlay(self) -> None:
        if self.loaded is None:
            QtWidgets.QMessageBox.information(self, "提示", "请先加载 .z 数据，再读取 tx.in。")
            self.lbl_status.setText("读取 tx.in 失败：尚未加载数据")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 tx.in 文件", "", "tx.in files (*.in *.tx);;All files (*)", options=self._file_dialog_options()
        )
        if not path:
            self.lbl_status.setText("读取 tx.in 已取消")
            return
        try:
            data = self._parse_txin_overlay_file(path)
            offsets = np.asarray(data.get("offsets", []), dtype=float)
            times = np.asarray(data.get("times", []), dtype=float)
            pick_words = np.asarray(data.get("pick_words", []), dtype=int)
            if offsets.size == 0 or times.size == 0:
                self.lbl_status.setText("tx.in 导入失败：无有效走时点")
                return
            self.txin_overlay_data = data
            self.show_txin_overlay = True
            self.txin_map_preview_data = None
            self._clear_txin_map_preview_item()
            self.request_render(immediate=True)
            n_words = int(np.unique(pick_words).size) if pick_words.size > 0 else 0
            self.lbl_status.setText(f"tx.in 已叠加：{offsets.size} 点，{n_words} 个拾取字（分色）")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "tx.in 导入失败", str(exc))
            self.lbl_status.setText(f"tx.in 导入失败: {exc}")


    def _clear_txin_overlay(self) -> None:
        self.txin_overlay_data = None
        self.show_txin_overlay = False
        self.txin_map_preview_data = None
        self._clear_txin_item()
        self._clear_txin_map_preview_item()
        self.request_render(delay_ms=10)
        self.lbl_status.setText("已清除 tx.in 走时叠加")


    def _build_txin_map_candidates(self) -> Tuple[Dict[Tuple[int, int], Tuple[float, float]], Dict[str, object]]:
        if self.loaded is None or self.txin_overlay_data is None:
            return {}, {"error": "请先加载数据并读取 tx.in"}
        target_idx = self._extract_indices()
        if target_idx.size == 0:
            return {}, {"error": "当前剖面无可用道"}
        offsets_all = np.asarray(self.loaded.get("offsets", []), dtype=float)
        if offsets_all.size == 0:
            return {}, {"error": "缺少偏移距信息"}
        tx_off = np.asarray(self.txin_overlay_data.get("offsets", []), dtype=float)
        tx_t = np.asarray(self.txin_overlay_data.get("times", []), dtype=float)
        tx_pw = np.asarray(self.txin_overlay_data.get("pick_words", []), dtype=int)
        if tx_off.size == 0 or tx_t.size != tx_off.size or tx_pw.size != tx_off.size:
            return {}, {"error": "tx.in 叠加数据无效"}

        target_offsets = offsets_all[target_idx]
        valid_pw_max = int(self.spin_apick.maximum())
        apick_only = bool(self.chk_map_txin_apick_only.isChecked())
        apick = int(self.spin_apick.value())
        view_only = bool(self.chk_map_txin_view_only.isChecked())
        tol_ratio = max(0.10, float(self.spin_map_txin_tol.value()) / 100.0)
        x_view = self.plot.getViewBox().viewRange()[0]
        y_view = self.plot.getViewBox().viewRange()[1]
        xmin_v, xmax_v = float(min(x_view)), float(max(x_view))
        ymin_v, ymax_v = float(min(y_view)), float(max(y_view))

        uniq_off = np.unique(np.sort(target_offsets))
        if uniq_off.size >= 2:
            spacing = float(np.median(np.diff(uniq_off)))
            tol = max(0.05, tol_ratio * abs(spacing))
        else:
            tol = float("inf")

        candidate: Dict[Tuple[int, int], Tuple[float, float]] = {}
        ignored_far = 0
        ignored_bad_pw = 0
        ignored_out_view = 0
        for i in range(int(tx_off.size)):
            pw = int(tx_pw[i])
            if pw < 1 or pw > valid_pw_max:
                ignored_bad_pw += 1
                continue
            if apick_only and pw != apick:
                continue
            x = float(tx_off[i])
            t = float(tx_t[i])
            if not (math.isfinite(x) and math.isfinite(t)):
                continue
            if view_only:
                t_disp = t + self._compute_reduction_tshift(-1, x)
                if not (xmin_v <= x <= xmax_v and ymin_v <= t_disp <= ymax_v):
                    ignored_out_view += 1
                    continue
            nearest_pos = int(np.argmin(np.abs(target_offsets - x)))
            trace_idx = int(target_idx[nearest_pos])
            dist = float(abs(target_offsets[nearest_pos] - x))
            if dist > tol:
                ignored_far += 1
                continue
            key = (trace_idx, pw)
            prev = candidate.get(key)
            if prev is None or dist < prev[0]:
                candidate[key] = (dist, t)

        return candidate, {
            "ignored_far": ignored_far,
            "ignored_bad_pw": ignored_bad_pw,
            "ignored_out_view": ignored_out_view,
            "apick_only": apick_only,
            "apick": apick,
            "view_only": view_only,
            "tol_percent": float(self.spin_map_txin_tol.value()),
        }


    def _preview_txin_mapping(self) -> None:
        if self.loaded is None or self.pick_manager is None:
            self.lbl_status.setText("请先加载数据")
            return
        if self.txin_overlay_data is None:
            self.lbl_status.setText("请先读取 tx.in")
            return
        candidate, meta = self._build_txin_map_candidates()
        err = meta.get("error")
        if err:
            self.lbl_status.setText(f"预览失败：{err}")
            return
        if not candidate:
            self.txin_map_preview_data = None
            self._clear_txin_map_preview_item()
            self.request_render(delay_ms=10)
            self.lbl_status.setText("映射预览：无可新增点")
            return

        keep_existing = 0
        preview_points: List[Tuple[int, int, float]] = []
        for (trace_idx, pw), (_, tval) in candidate.items():
            exist = self.pick_manager.get_pick(trace_idx, pw)
            if exist is not None and float(exist) > 0.0:
                keep_existing += 1
                continue
            preview_points.append((int(trace_idx), int(pw), float(tval)))

        if not preview_points:
            self.txin_map_preview_data = None
            self._clear_txin_map_preview_item()
            self.request_render(delay_ms=10)
            self.lbl_status.setText(f"映射预览：新增 0，已有拾取将保留 {keep_existing}")
            return

        arr = np.asarray(preview_points, dtype=float)
        self.txin_map_preview_data = {
            "trace_indices": arr[:, 0].astype(int),
            "pick_words": arr[:, 1].astype(int),
            "times": arr[:, 2].astype(float),
        }
        self.request_render(immediate=True)
        mode_text = f"仅apick={int(meta['apick'])}" if bool(meta["apick_only"]) else "全部拾取字"
        view_text = "仅视窗" if bool(meta["view_only"]) else "全窗口"
        self.lbl_status.setText(
            f"映射预览：将新增 {int(arr.shape[0])}，已有保留 {keep_existing}，"
            f"超距忽略 {int(meta['ignored_far'])}，视窗外忽略 {int(meta['ignored_out_view'])}，"
            f"拾取字越界忽略 {int(meta['ignored_bad_pw'])}（{mode_text}, {view_text}, 容差={float(meta['tol_percent']):.1f}%）"
        )


    def _map_txin_to_picks(self) -> None:
        """
        将 tx.in 叠加点映射为当前剖面的拾取：
        - 以当前过滤结果（irec/itype/nskip/x窗/移除道）作为目标剖面
        - 以偏移距最近道进行匹配
        - 若该道该拾取字已有拾取，则保留不覆盖
        """
        if self.loaded is None or self.pick_manager is None:
            self.lbl_status.setText("请先加载数据")
            return
        if self.txin_overlay_data is None:
            self.lbl_status.setText("请先读取 tx.in")
            return

        times_all = np.asarray(self.loaded.get("times", []), dtype=float)
        if times_all.size == 0:
            self.lbl_status.setText("映射失败：缺少时间轴")
            return
        candidate, meta = self._build_txin_map_candidates()
        err = meta.get("error")
        if err:
            self.lbl_status.setText(f"映射失败：{err}")
            return

        if not candidate:
            self.lbl_status.setText("映射完成：无可映射点（可能偏移距不匹配）")
            return

        added = 0
        kept_existing = 0
        tmin = float(np.min(times_all))
        tmax = float(np.max(times_all))
        to_add: List[Tuple[int, int, float]] = []
        for (trace_idx, pw), (_, tval) in candidate.items():
            exist = self.pick_manager.get_pick(trace_idx, pw)
            if exist is not None and float(exist) > 0.0:
                kept_existing += 1
                continue
            to_add.append((int(trace_idx), int(pw), float(np.clip(tval, tmin, tmax))))

        if to_add:
            self._push_pick_undo("tx.in 映射拾取")
            for trace_idx, pw, tval in to_add:
                self.pick_manager.add_pick(trace_idx, tval, pw)
            added = len(to_add)

        if added > 0:
            self.request_render(delay_ms=10)
        self.txin_map_preview_data = None
        self._clear_txin_map_preview_item()
        mode_text = f"仅apick={int(meta['apick'])}" if bool(meta["apick_only"]) else "全部拾取字"
        view_text = "仅视窗" if bool(meta["view_only"]) else "全窗口"
        self.lbl_status.setText(
            f"tx.in 映射完成：新增 {added}，保留已有 {kept_existing}，"
            f"超距忽略 {int(meta['ignored_far'])}，视窗外忽略 {int(meta['ignored_out_view'])}，拾取字越界忽略 {int(meta['ignored_bad_pw'])}"
            f"（{mode_text}, {view_text}, 容差={float(meta['tol_percent']):.1f}%）"
        )


    def _parse_txin_overlay_file(self, txin_path: str) -> Dict[str, np.ndarray]:
        """
        解析 tx.in 并转换为可叠加曲线：
        - 输入通常为「模型距离 x + 真实走时 t」
        - 通过分段标记行（第4列=0）中的 xmod，换算偏移距：offset = x - xmod
        - 显示时再统一做折合时间修正（在渲染阶段）
        """
        points_all: List[Tuple[float, float, int]] = []
        current_xmod: Optional[float] = None

        with open(txin_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#") or line.startswith("!"):
                    continue
                fields = line.split()
                if len(fields) < 4:
                    continue
                try:
                    col1 = float(fields[0])
                    col2 = float(fields[1])
                    _col3 = float(fields[2])
                    ipick = int(round(float(fields[3])))
                except Exception:
                    continue

                # 结束标记
                if ipick == -1:
                    break
                # 分段标记：xmod, (+/-1), 0, 0
                if ipick == 0:
                    current_xmod = float(col1)
                    continue
                # 常规拾取行
                if ipick > 0:
                    if not (math.isfinite(col1) and math.isfinite(col2)):
                        continue
                    if current_xmod is None:
                        # 无分段标记时退化：假定 x 已是偏移距
                        off = float(col1)
                    else:
                        off = float(col1 - current_xmod)
                    points_all.append((off, float(col2), int(ipick)))

        if not points_all:
            raise ValueError("未读取到可用拾取点（请检查 tx.in 文件内容）")

        arr = np.asarray(points_all, dtype=float)
        order = np.argsort(arr[:, 0])
        arr = arr[order]
        return {
            "offsets": arr[:, 0].astype(float),
            "times": arr[:, 1].astype(float),
            "pick_words": arr[:, 2].astype(int),
        }


    def _prompt_tx_config(self, shots: List[int], npick: int) -> Optional[Z2TxConfig]:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("tx.in 转换配置")
        dialog.resize(760, 420)
        layout = QtWidgets.QVBoxLayout(dialog)

        table = QtWidgets.QTableWidget(len(shots), 4, dialog)
        table.setHorizontalHeaderLabels(["OBS", "xmod", "tshift", "xshift"])
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setStretchLastSection(True)
        # 若已加载 .rec/.rsp，按 ishnum 预填 xmod
        rec_xmod: Dict[int, float] = {}
        try:
            for rec in (self.loaded.get("records") or []) if self.loaded else []:
                ish = int(getattr(rec, "ishnum", 0) or 0)
                if ish > 0:
                    rec_xmod[ish] = float(getattr(rec, "xmod", 0.0) or 0.0)
        except Exception:
            rec_xmod = {}
        for row, shot in enumerate(shots):
            obs_item = QtWidgets.QTableWidgetItem(str(int(shot)))
            obs_item.setFlags(obs_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, obs_item)
            x0 = rec_xmod.get(int(shot), 0.0)
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{float(x0):.6g}"))
            table.setItem(row, 2, QtWidgets.QTableWidgetItem("0.0"))
            table.setItem(row, 3, QtWidgets.QTableWidgetItem("0.0"))
        layout.addWidget(table)

        form = QtWidgets.QFormLayout()
        edit_picku = QtWidgets.QLineEdit("0.05", dialog)
        edit_picku.setPlaceholderText("单值或逗号分隔列表，如 0.05 或 0.03,0.04,0.05")
        combo_mode = QtWidgets.QComboBox(dialog)
        combo_mode.addItems(["走时(iamp=0)", "振幅(iamp=1)"])
        form.addRow("picku", edit_picku)
        form.addRow("模式", combo_mode)
        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return None

        picku_text = edit_picku.text().strip()
        try:
            if "," in picku_text:
                picku_list = [float(x.strip()) for x in picku_text.split(",") if x.strip()]
                if not picku_list:
                    raise ValueError("picku 为空")
            else:
                picku_val = float(picku_text)
                picku_list = [picku_val] * max(1, int(npick))
        except Exception:
            QtWidgets.QMessageBox.warning(dialog, "参数错误", "picku 格式错误，请输入单值或逗号分隔浮点数。")
            return None

        obs_configs: Dict[int, Dict] = {}
        for row, shot in enumerate(shots):
            try:
                xmod_item = table.item(row, 1)
                tshift_item = table.item(row, 2)
                xshift_item = table.item(row, 3)
                xmod = float(xmod_item.text() if xmod_item is not None else 0.0)
                tshift = float(tshift_item.text() if tshift_item is not None else 0.0)
                xshift = float(xshift_item.text() if xshift_item is not None else 0.0)
            except Exception:
                QtWidgets.QMessageBox.warning(dialog, "参数错误", f"OBS {shot} 的数值格式错误。")
                return None
            obs_configs[int(shot)] = {
                "xmod": xmod,
                "tshift": tshift,
                "xshift": xshift,
                "picku": list(picku_list),
            }

        iamp = 1 if combo_mode.currentIndex() == 1 else 0
        return Z2TxConfig(obs_configs=obs_configs, iamp=iamp)


    def _clear_picks(self) -> None:
        if self.pick_manager is None:
            return
        pick_word = int(self.spin_apick.value()) if hasattr(self, "spin_apick") else 1
        by_word = self.pick_manager.get_picks_by_word(pick_word)
        if not by_word:
            self.lbl_status.setText(f"当前拾取字(apick={pick_word})无可清空拾取")
            return
        self._push_pick_undo(f"清空拾取字{pick_word}")
        removed = 0
        for trace_idx in list(by_word.keys()):
            if self.pick_manager.remove_pick(int(trace_idx), pick_word):
                removed += 1
        self.request_render(immediate=True)
        self.lbl_status.setText(f"已清空当前拾取字(apick={pick_word})：{removed} 道")


    def _apply_shift_hover_pick(self) -> None:
        """拾取模式下，按住 Shift 时在鼠标经过道上连续拾取（每道一次）。"""
        if not self._shift_pressed or not self._shift_hover_pick_active:
            return
        if self.loaded is None or self.pick_manager is None:
            return
        if not self.chk_pick_mode.isChecked():
            return
        if self.mouse_x is None or self.mouse_y is None:
            return
        if self._last_render_trace_indices.size == 0 or self._last_render_offsets.size == 0:
            return
        x = float(self.mouse_x)
        y = float(self.mouse_y)
        nearest_i = int(np.argmin(np.abs(self._last_render_offsets - x)))
        trace_idx = int(self._last_render_trace_indices[nearest_i])
        if trace_idx in self._shift_hover_picked_traces:
            return
        x_trace = float(self._last_render_offsets[nearest_i])
        y_pick = float(y) - self._compute_display_tshift(trace_idx, x_trace)
        apick = int(self.spin_apick.value())
        old_pick = self._get_shared_pick(trace_idx, apick)
        self._shift_hover_picked_traces.add(trace_idx)
        if old_pick is not None and abs(float(old_pick) - y_pick) < 1e-9:
            return
        if not self._shift_hover_pick_undo_pushed:
            self._push_pick_undo("Shift悬停拾取")
            self._shift_hover_pick_undo_pushed = True
        self._set_shared_pick(trace_idx, apick, y_pick)
        self._shift_hover_pick_updated_count += 1
        self.request_render(delay_ms=10)


    def _ask_pick_save_mode(self, title: str) -> bool:
        """独立 zplotpy：无姿态校正，始终按原始走时保存。"""
        return False


    def _build_pick_snapshot(
        self,
        use_orientation_corrected: bool = False,
    ) -> Dict[int, Dict[int, float]]:
        """原始拾取快照（姿态校正已迁出，忽略 use_orientation_corrected）。"""
        if self.pick_manager is None:
            return {}
        base_raw = self.pick_manager.get_all_picks()
        return {
            int(trace_idx): {int(word): float(tpk) for word, tpk in by_word.items()}
            for trace_idx, by_word in base_raw.items()
        }


    def _run_pick_alignment(self) -> None:
        if self.pick_manager is None:
            return
        # 与旧版交互一致：再次按 A 直接清除当前对齐
        if self._alignment_offsets:
            self._alignment_offsets = {}
            self.request_render(delay_ms=10)
            self.lbl_status.setText("已清除波形临时对齐（A 再次按下）")
            return
        pick_word = int(self.spin_apick.value())
        picks = self.pick_manager.get_picks_by_word(pick_word)
        if len(picks) < 2:
            self.lbl_status.setText("波形临时对齐失败：当前拾取字至少需要2个拾取点")
            return
        offsets_all = np.asarray(self.loaded.get("offsets", []), dtype=float) if self.loaded is not None else np.array([])
        disp_times: Dict[int, float] = {}
        for trace_idx, tpk in picks.items():
            ig = int(trace_idx)
            if ig < 0 or ig >= offsets_all.size:
                continue
            base_tshift = self._compute_display_tshift(ig, float(offsets_all[ig]))
            disp_times[ig] = float(tpk) + base_tshift
        if len(disp_times) < 2:
            self.lbl_status.setText("波形临时对齐失败：有效显示拾取不足")
            return
        ref_time = float(np.median(np.asarray(list(disp_times.values()), dtype=float)))
        self._alignment_offsets = {
            int(trace_idx): (ref_time - disp_t)
            for trace_idx, disp_t in disp_times.items()
        }
        self.request_render(delay_ms=10)
        self.lbl_status.setText(f"波形临时对齐完成：{len(self._alignment_offsets)} 道（按当前显示时间对齐）")


    def _run_adaptive_alignment(self) -> None:
        if self.loaded is None or self.pick_manager is None:
            return
        traces = self.loaded.get("traces", [])
        times = np.asarray(self.loaded.get("times", []), dtype=float)
        offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
        if len(traces) == 0 or times.size < 2 or offsets.size == 0:
            return
        idx = self._extract_indices()
        if idx.size == 0:
            self.lbl_status.setText("自适应拾取更新失败：当前过滤后无道")
            return
        xcoords = offsets[idx]
        idx_vis = idx[self._visible_mask(xcoords)]
        if idx_vis.size < 2:
            self.lbl_status.setText("自适应拾取更新失败：视窗内道数不足")
            return

        pick_word = int(self.spin_apick.value())
        by_word = self.pick_manager.get_picks_by_word(pick_word)
        # 对齐使用“折合后的显示时间域”：
        # 1) 先用当前处理参数得到用于相关的波形（含滤波）
        # 2) 按 (rvred-rvredf) 把每道重采样到统一显示时间网格
        raw_selected_traces = [np.asarray(traces[int(i)], dtype=float) for i in idx_vis]
        proc_params = self._build_processing_params()
        processed_traces = self.processor.process_traces(
            raw_selected_traces,
            times,
            offsets[idx_vis],
            proc_params,
            realtime_interaction=False,
        )
        selected_traces: List[np.ndarray] = []
        reduction_shifts: List[float] = []
        for li, gidx in enumerate(idx_vis):
            red_shift = float(self._compute_reduction_tshift(int(gidx), float(offsets[int(gidx)])))
            reduction_shifts.append(red_shift)
            tr = np.asarray(processed_traces[li], dtype=float)
            # 显示域 td = t + tshift => A(td)=A_true(td - tshift)
            # 这里将每道映射到统一 times 网格，避免 vred 下“看着对齐、算法却在真时域错配”。
            tr_disp = np.interp(
                times - red_shift,
                times,
                tr,
                left=0.0,
                right=0.0,
            )
            selected_traces.append(tr_disp)
        initial_picks: List[int] = []
        t0 = float(times[0])
        dt = float(times[1] - times[0])
        for li, gidx in enumerate(idx_vis):
            pt = by_word.get(int(gidx))
            if pt is None:
                initial_picks.append(-1)
            else:
                pt_display = float(pt) + float(reduction_shifts[li])
                initial_picks.append(int(round((pt_display - t0) / dt)))
        corr_before: Optional[float] = None
        picks_before_display: Dict[int, float] = {}
        for li, gidx in enumerate(idx_vis):
            pt = by_word.get(int(gidx))
            if pt is None:
                continue
            ptf = float(pt)
            if ptf <= 0.0:
                continue
            picks_before_display[int(li)] = ptf + float(reduction_shifts[li])
        if len(picks_before_display) >= 2:
            try:
                corr_info_before = self.adaptive_stacker.calculate_correlation(
                    traces=selected_traces,
                    times=times,
                    picks=picks_before_display,
                )
                corr_before = float(corr_info_before.get("mean_correlation", 0.0))
            except Exception:
                corr_before = None

        try:
            result = self.adaptive_stacker.align_traces(
                traces=selected_traces,
                times=times,
                initial_picks=initial_picks,
            )
        except Exception as exc:
            self.last_stacking_result = None
            self.lbl_status.setText(f"自适应拾取更新失败: {exc}")
            return
        shifts = result.get("time_shifts", [])
        if not shifts:
            self.last_stacking_result = None
            self.lbl_status.setText("自适应拾取更新失败：未返回有效偏移")
            return
        n_apply = 0
        original_picks: Dict[int, float] = {}
        updated_picks: Dict[int, float] = {}
        applied_shifts: List[float] = []
        applied_shifts_by_trace: Dict[int, float] = {}
        errors_by_trace: Dict[int, float] = {}
        corr_after: Optional[float] = None
        result_errors = list(result.get("errors", []))
        for li, gidx in enumerate(idx_vis):
            if li < len(shifts) and initial_picks[li] >= 0:
                orig = float(by_word.get(int(gidx), 0.0))
                if orig > 0:
                    if n_apply == 0:
                        self._push_pick_undo("自适应拾取更新")
                    original_picks[int(gidx)] = orig
                    # shifts 定义在“折合显示时间域”，对同一道回写到真时拾取时可直接加到原拾取。
                    new_pick = orig + float(shifts[li])
                    new_pick = float(np.clip(new_pick, t0, float(times[-1])))
                    # F 键仅更新拾取时间，不改变波形时间
                    self.pick_manager.add_pick(int(gidx), new_pick, pick_word)
                    updated_picks[int(gidx)] = new_pick
                    applied_shifts.append(float(shifts[li]))
                    applied_shifts_by_trace[int(gidx)] = float(shifts[li])
                    if li < len(result_errors):
                        errors_by_trace[int(gidx)] = float(result_errors[li])
                n_apply += 1
        picks_after_display: Dict[int, float] = {}
        for li, gidx in enumerate(idx_vis):
            orig_pt = by_word.get(int(gidx))
            if orig_pt is None:
                continue
            updated_pt = float(updated_picks.get(int(gidx), float(orig_pt)))
            if updated_pt <= 0.0:
                continue
            picks_after_display[int(li)] = updated_pt + float(reduction_shifts[li])
        if len(picks_after_display) >= 2:
            try:
                corr_info_after = self.adaptive_stacker.calculate_correlation(
                    traces=selected_traces,
                    times=times,
                    picks=picks_after_display,
                )
                corr_after = float(corr_info_after.get("mean_correlation", 0.0))
            except Exception:
                corr_after = None
        # 显式保持为空，避免 F 对波形时间产生任何影响
        self._alignment_offsets = {}
        self.last_stacking_result = {
            "original_picks": original_picks,
            "updated_picks": updated_picks,
            "time_shifts": applied_shifts,
            "errors": result_errors,
            "time_shifts_by_trace": applied_shifts_by_trace,
            "errors_by_trace": errors_by_trace,
            "quality_metric": float(result.get("quality_metric", 0.0) or 0.0),
            "mean_corr_before": corr_before,
            "mean_corr_after": corr_after,
            "offsets": offsets,
            "traces": selected_traces,
            "times": times,
        }
        self.request_render(delay_ms=10)
        msg = f"自适应拾取更新完成：更新 {n_apply} 道拾取（波形时间不变）"
        if corr_before is not None and corr_after is not None:
            delta_corr = corr_after - corr_before
            msg += f" | 一致性 mean corr: {corr_before:.3f}->{corr_after:.3f} (Δ={delta_corr:+.3f})"
        self._set_status_text(msg, hold_ms=5000)


    def _clear_alignment(self) -> None:
        self._alignment_offsets = {}
        self.request_render(delay_ms=10)
        self.lbl_status.setText("已清除波形临时对齐偏移")


    def _run_auto_pick(self) -> None:
        if self.loaded is None or self.pick_manager is None:
            return
        traces = self.loaded.get("traces", [])
        offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
        times = np.asarray(self.loaded.get("times", []), dtype=float)
        if len(traces) == 0 or times.size == 0:
            return
        idx = self._extract_indices()
        if idx.size == 0:
            return
        pick_word = int(self.spin_apick.value())
        selected_traces = [np.asarray(traces[int(i)]) for i in idx]
        selected_offsets = offsets[idx] if offsets.size else np.zeros(idx.size, dtype=float)
        results = self.auto_picker.pick_traces(
            traces=selected_traces,
            times=times,
            offsets=selected_offsets,
        )
        added = 0
        undo_pushed = False
        for local_i, res in enumerate(results):
            if not res:
                continue
            t = float(res.get("pick_time", 0.0))
            if t > 0:
                gidx = int(idx[local_i])
                old_pick = self.pick_manager.get_pick(gidx, pick_word)
                if old_pick is not None and abs(float(old_pick) - t) < 1e-9:
                    continue
                if not undo_pushed:
                    self._push_pick_undo("自动拾取")
                    undo_pushed = True
                if self.pick_manager.add_pick(gidx, t, pick_word):
                    added += 1
        self.request_render(delay_ms=10)
        self.lbl_status.setText(f"自动拾取完成：新增 {added} 个拾取")


    def _run_interp_pick(self) -> None:
        if self.loaded is None or self.pick_manager is None:
            return
        self.params.tcrcor = float(self.spin_tcrcor.value())
        self.params.tlag = float(self.spin_tlag.value())
        self.params.hilbratio = float(self.spin_hilbratio.value())
        force_pick = False
        traces = self.loaded.get("traces", [])
        offsets = np.asarray(self.loaded.get("offsets", []), dtype=float)
        times = np.asarray(self.loaded.get("times", []), dtype=float)
        if len(traces) == 0 or times.size == 0 or offsets.size == 0:
            return
        if times.size < 2:
            self.lbl_status.setText("插值相关拾取失败：时间采样点不足")
            return
        # 仅在当前过滤集合（含当前分量）内执行，避免跨分量混拾取
        idx = self._extract_indices()
        if idx.size < 3:
            self.lbl_status.setText("插值相关拾取失败：当前分量/过滤后道数不足")
            return
        idx_local_to_global = np.asarray(idx, dtype=int)
        allowed_set = set(int(i) for i in idx_local_to_global)
        global_to_local = {int(g): li for li, g in enumerate(idx_local_to_global.tolist())}
        pick_word = int(self.spin_apick.value())
        pw_picks: Dict[int, float] = self.pick_manager.get_picks_by_word(pick_word)
        pw_picks = {int(k): float(v) for k, v in pw_picks.items() if int(k) in allowed_set}
        if len(pw_picks) < 2:
            self.lbl_status.setText("插值相关拾取需要当前拾取字至少2个种子点")
            return
        # 与显示链一致：在处理后道上执行插值相关（避免“显示与拾取不一致”）
        # 性能优化：只处理当前过滤后道集合，避免全体道预处理。
        raw_traces = [np.asarray(traces[int(i)]) for i in idx_local_to_global]
        offsets_subset = offsets[idx_local_to_global]
        trace_headers = self.loaded.get("trace_headers", [])
        gains_subset = np.ones(len(raw_traces), dtype=float)
        if trace_headers:
            for li, gidx in enumerate(idx_local_to_global):
                gi = int(gidx)
                if 0 <= gi < len(trace_headers):
                    gains_subset[li] = float(max(1, int(getattr(trace_headers[gi], "igaini", 1) or 1)))
        if times.size > 1:
            sr = 1.0 / float(times[1] - times[0])
        else:
            sr = None
        try:
            proc_params = self._build_processing_params()
            traces_for_pick = self.processor.process_traces(
                traces=raw_traces,
                times=times,
                offsets=offsets_subset,
                params=proc_params,
                gains=gains_subset,
                sampling_rate=sr,
                realtime_interaction=False,
            )
        except Exception as exc:
            self.lbl_status.setText(f"插值相关拾取失败：处理后波形生成失败: {exc}")
            return

        sorted_seed = sorted(pw_picks.items(), key=lambda kv: kv[0])
        seed_pairs = []
        allowed_idx = [int(i) for i in idx_local_to_global]
        for i in range(len(sorted_seed) - 1):
            i1, t1 = sorted_seed[i]
            i2, t2 = sorted_seed[i + 1]
            if i1 == i2:
                continue
            lo_i, hi_i = min(int(i1), int(i2)), max(int(i1), int(i2))
            has_middle = any(lo_i < ai < hi_i for ai in allowed_idx)
            if has_middle:
                li1 = global_to_local.get(int(i1))
                li2 = global_to_local.get(int(i2))
                if li1 is None or li2 is None:
                    continue
                seed_pairs.append((int(li1), float(t1), int(li2), float(t2)))
        if not seed_pairs:
            self.lbl_status.setText("插值相关拾取失败：没有可插值的种子区间（需两种子之间有道）")
            return

        corr_win = max(8, int(round(float(self.params.tcrcor) / max(1e-9, float(times[1] - times[0])))))
        lag_win = max(2, int(round(float(self.params.tlag) / max(1e-9, float(times[1] - times[0])))))

        added = 0
        pair_used = 0
        undo_pushed = False
        for pick1_local_idx, pick1_time, pick2_local_idx, pick2_time in seed_pairs:
            picks = self.interp_picker.interpolation_correlation_picking(
                traces=traces_for_pick,
                times=times,
                offsets=offsets_subset,
                pick1_idx=pick1_local_idx,
                pick2_idx=pick2_local_idx,
                pick1_time=pick1_time,
                pick2_time=pick2_time,
                correlation_window=corr_win,
                search_range=lag_win,
                hilbert_ratio=float(self.params.hilbratio),
                force_pick=force_pick,
            )
            pair_used += 1
            for local_idx, tpk in picks.items():
                if int(local_idx) < 0 or int(local_idx) >= idx_local_to_global.size:
                    continue
                gidx = int(idx_local_to_global[int(local_idx)])
                if int(gidx) not in allowed_set:
                    continue
                old_pick = self.pick_manager.get_pick(int(gidx), pick_word)
                if old_pick is not None and abs(float(old_pick) - float(tpk)) < 1e-9:
                    continue
                if not undo_pushed:
                    self._push_pick_undo("插值相关拾取")
                    undo_pushed = True
                if self.pick_manager.add_pick(int(gidx), float(tpk), pick_word):
                    added += 1
        self.request_render(delay_ms=10)
        mode_text = "强制模式" if force_pick else "严格模式"
        self.lbl_status.setText(
            f"插值相关完成（{mode_text}）：区间 {pair_used} 段，新增 {added} 个拾取"
        )

