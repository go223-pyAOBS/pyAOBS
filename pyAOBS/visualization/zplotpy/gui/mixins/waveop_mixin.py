# -*- coding: utf-8 -*-
"""V-selection / waveop stack logic mixed into QtFastViewer."""

from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as exc:  # pragma: no cover
    raise RuntimeError("未安装 PySide6") from exc

try:
    import pyqtgraph as pg
except Exception:
    pg = None  # type: ignore


class WaveopMixin:
    """V 选波、列表停靠、叠加与 waveop 存取。"""

    def _waveop_apick_segment_count(self, apick: Optional[int] = None) -> int:
        if apick is None:
            apick = int(self.spin_apick.value())
        cnt = 0
        for sel in self.waveform_selections:
            if int(sel.get("pick_word", apick)) == int(apick):
                cnt += 1
        return cnt


    def _waveop_apick_status_suffix(self, apick: Optional[int] = None) -> str:
        if apick is None:
            apick = int(self.spin_apick.value())
        return f"（apick={int(apick)}，V段={self._waveop_apick_segment_count(int(apick))}）"


    def _save_waveop_state(self) -> None:
        if self.loaded is None:
            self._show_themed_info("保存V段", "请先加载数据后再保存V段。")
            return
        out, _ = self._get_save_file_name(
            "保存V段与校正基准",
            "",
            "WaveOp JSON (*.waveop.json *.json);;All files (*)",
            default_suffix=".waveop.json",
            preferred_suffix=".waveop.json",
        )
        if not out:
            return
        wave_corr = [
            {
                "trace_idx": int(k[0]),
                "pick_word": int(k[1]),
                "t_true": float(v),
            }
            for k, v in sorted(self._waveop_corrected_ttrue.items(), key=lambda kv: (int(kv[0][1]), int(kv[0][0])))
        ]
        payload = {
            "version": 1,
            "kind": "waveop_state",
            "dfile": self._dfile,
            "hfile": self._hfile,
            "rfile": self._rfile,
            "waveform_selections": [dict(s) for s in self.waveform_selections],
            "waveop_corrected_ttrue": wave_corr,
        }
        try:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self.lbl_status.setText(f"V段已保存：{out} {self._waveop_apick_status_suffix()}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "保存失败", f"V段保存失败：{exc}")


    def _load_waveop_state(self) -> None:
        if self.loaded is None:
            self._show_themed_info("加载V段", "请先加载数据后再加载V段。")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "加载V段与校正基准",
            "",
            "WaveOp JSON (*.waveop.json *.json);;All files (*)",
            options=self._file_dialog_options(),
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and str(payload.get("kind", "")).strip() not in ("", "waveop_state"):
                raise ValueError("文件类型不是V段状态文件")
            restored: List[Dict[str, float]] = []
            val = payload.get("waveform_selections", [])
            if isinstance(val, list):
                for item in val:
                    if not isinstance(item, dict):
                        continue
                    try:
                        restored.append(
                            {
                                "trace_idx": float(item.get("trace_idx", -1)),
                                "offset": float(item.get("offset", 0.0)),
                                "t_display": float(item.get("t_display", 0.0)),
                                "t_true": float(item.get("t_true", 0.0)),
                                "pick_word": float(item.get("pick_word", 1) or 1),
                            }
                        )
                    except Exception:
                        continue
            restored_corr: Dict[Tuple[int, int], float] = {}
            valc = payload.get("waveop_corrected_ttrue", [])
            if isinstance(valc, list):
                for item in valc:
                    if not isinstance(item, dict):
                        continue
                    try:
                        key = (int(item.get("trace_idx", -1)), int(item.get("pick_word", 1)))
                        tval = float(item.get("t_true", np.nan))
                        if key[0] >= 0 and np.isfinite(tval):
                            restored_corr[key] = tval
                    except Exception:
                        continue
            if (self.waveform_selections or self._waveop_corrected_ttrue) and (restored or restored_corr):
                ans = QtWidgets.QMessageBox.question(
                    self,
                    "加载V段",
                    "将覆盖当前V段与叠加校正基准，是否继续？",
                    QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No,
                )
                if ans != QtWidgets.QMessageBox.StandardButton.Yes:
                    return
            self.waveform_selections = restored
            self._waveop_corrected_ttrue = restored_corr
            self._refresh_waveop_selection_list()
            self.request_render(delay_ms=10)
            self.lbl_status.setText(f"V段已加载：{path} {self._waveop_apick_status_suffix()}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "加载失败", f"V段加载失败：{exc}")


    def _waveop_apick_display_color(self, pick_word: int):
        """与拾取点一致：按 pick_word 分色。"""
        return pg.intColor(int(pick_word), hues=48, values=1, alpha=220)


    def _ensure_wave_select_item_for(self, pick_word: int) -> pg.PlotDataItem:
        pw = int(pick_word)
        item = self._wave_select_items.get(pw)
        if item is None:
            color = self._waveop_apick_display_color(pw)
            item = pg.PlotDataItem(pen=pg.mkPen(color, width=1.8))
            item.setZValue(30)
            self.plot.addItem(item)
            self._wave_select_items[pw] = item
        return item


    def _clear_wave_select_item(self) -> None:
        for item in self._wave_select_items.values():
            try:
                item.setData([], [])
            except Exception:
                pass


    def _ensure_wave_select_marker_item(self) -> None:
        if self._wave_select_marker_item is None:
            self._wave_select_marker_item = pg.ScatterPlotItem(
                size=9.0,
                pen=pg.mkPen("#ffffff", width=1.0),
                brush=pg.mkBrush("#ef4444"),
                pxMode=True,
            )
            self._wave_select_marker_item.setZValue(31)
            self.plot.addItem(self._wave_select_marker_item)


    def _clear_wave_select_marker_item(self) -> None:
        if self._wave_select_marker_item is not None:
            self._wave_select_marker_item.setData([], [])


    def _ensure_waveop_stack_item(self) -> None:
        if self._waveop_stack_item is None:
            self._waveop_stack_item = pg.PlotDataItem(
                pen=pg.mkPen("#dc2626", width=2.0),
            )
            self._waveop_stack_item.setZValue(29)
            self.plot.addItem(self._waveop_stack_item)


    def _clear_waveop_stack_item(self) -> None:
        if self._waveop_stack_item is not None:
            self._waveop_stack_item.setData([], [])


    def _add_waveform_selection_at_cursor(self) -> None:
        """V 键：以鼠标最近道为中心，记录选波窗口（前0.3s、后0.7s）。"""
        if self.loaded is None:
            self.lbl_status.setText("V选波失败：请先加载数据")
            return
        if self.mouse_x is None or self.mouse_y is None:
            self.lbl_status.setText("V选波失败：请先将鼠标移动到剖面区域")
            return
        if self._last_render_trace_indices.size == 0 or self._last_render_offsets.size == 0:
            self.lbl_status.setText("V选波失败：当前无可选道")
            return
        x_ref = float(self.mouse_x)
        y_ref = float(self.mouse_y)
        nearest_i = int(np.argmin(np.abs(self._last_render_offsets - x_ref)))
        trace_idx = int(self._last_render_trace_indices[nearest_i])
        x_trace = float(self._last_render_offsets[nearest_i])
        tshift = float(self._compute_display_tshift(trace_idx, x_trace))
        # 优先使用“当前拾取字”在该道的拾取点作为 V 段中心；
        # 若该道该拾取字无拾取，再退回鼠标位置。
        y_center = y_ref
        center_from_pick = False
        if self.pick_manager is not None:
            apick = int(self.spin_apick.value())
            picked_t = self._get_shared_pick(trace_idx, apick)
            if picked_t is not None and float(picked_t) > 0.0:
                y_center = float(picked_t) + tshift
                center_from_pick = True
        t_true = float(y_center - tshift)
        new_sel = {
            "trace_idx": float(trace_idx),
            "offset": x_trace,
            "t_display": y_center,
            "t_true": t_true,
            "pick_word": float(int(self.spin_apick.value())),
        }
        # 每道最多保留一个 V 段：同一道再次按 V 时更新该道现有记录
        replaced = False
        key_new = (int(trace_idx), int(self.spin_apick.value()))
        for i, old in enumerate(self.waveform_selections):
            if (
                int(old.get("trace_idx", -1)) == int(trace_idx)
                and int(old.get("pick_word", int(self.spin_apick.value()))) == int(self.spin_apick.value())
            ):
                self.waveform_selections[i] = new_sel
                replaced = True
                break
        if not replaced:
            self.waveform_selections.append(new_sel)
        self._waveop_corrected_ttrue[key_new] = float(t_true)
        self._refresh_waveop_selection_list()
        self.request_render(delay_ms=10)
        src = "拾取点" if center_from_pick else "鼠标"
        action = "更新" if replaced else "添加"
        self.lbl_status.setText(
            f"V选波已{action}：道 {trace_idx}，中心={y_center:.3f}s({src})，窗口=[{y_center - 0.3:.3f}, {y_center + 0.7:.3f}]s {self._waveop_apick_status_suffix()}"
        )


    def _current_apick_waveform_selections(self) -> List[Dict[str, float]]:
        apick = int(self.spin_apick.value())
        out: List[Dict[str, float]] = []
        for sel in self.waveform_selections:
            pw = int(sel.get("pick_word", apick))
            if pw == apick:
                out.append(sel)
        return out


    def _remove_last_waveform_selection(self) -> None:
        if not self.waveform_selections:
            self.lbl_status.setText("Shift+V：当前没有可删除的 V 段")
            return
        apick = int(self.spin_apick.value())
        remove_idx = -1
        for i in range(len(self.waveform_selections) - 1, -1, -1):
            if int(self.waveform_selections[i].get("pick_word", apick)) == apick:
                remove_idx = i
                break
        if remove_idx < 0:
            self.lbl_status.setText("Shift+V：当前拾取字下没有可删除的 V 段")
            return
        last = self.waveform_selections.pop(remove_idx)
        key = (int(last.get("trace_idx", -1)), int(last.get("pick_word", int(self.spin_apick.value()))))
        if key in self._waveop_corrected_ttrue:
            del self._waveop_corrected_ttrue[key]
        self.waveop_stack_result = None
        self._refresh_waveop_selection_list()
        self.request_render(delay_ms=10)
        self.lbl_status.setText(
            f"已删除最近 V 段：道 {int(last.get('trace_idx', -1))}，中心={float(last.get('t_display', 0.0)):.3f}s {self._waveop_apick_status_suffix()}"
        )


    def _clear_waveform_selections(self) -> None:
        """清除当前 apick 下的全部 V 段（其它字的段保留）。"""
        apick = int(self.spin_apick.value())
        kept: List[Dict[str, float]] = []
        for sel in list(self.waveform_selections or []):
            if int(sel.get("pick_word", apick)) != apick:
                kept.append(sel)
        n_cleared = len(self.waveform_selections) - len(kept)
        self.waveform_selections = kept
        for key in list(self._waveop_corrected_ttrue.keys()):
            try:
                if int(key[1]) == apick:
                    del self._waveop_corrected_ttrue[key]
            except Exception:
                continue
        self.waveop_stack_result = None
        self._refresh_waveop_selection_list()
        self.request_render(delay_ms=10)
        self.lbl_status.setText(
            f"已清除当前 apick={apick} 的 {n_cleared} 个 V 段（其它字保留） {self._waveop_apick_status_suffix()}"
        )


    def relocate_waveop_list(self, host: Optional[QtWidgets.QWidget]) -> None:
        """把 V 段列表挂到外部容器；``host=None`` 时还原到本地底部条。"""
        lst = getattr(self, "list_waveop_segments", None)
        if lst is None:
            return
        dock = getattr(self, "_waveop_list_local_dock", None)

        def _detach(w: QtWidgets.QWidget) -> None:
            parent = w.parentWidget()
            if parent is None:
                return
            lay = parent.layout()
            if lay is not None:
                lay.removeWidget(w)
            w.setParent(None)

        if host is None:
            self._waveop_list_external_host = None
            if dock is not None:
                if lst.parentWidget() is not dock:
                    _detach(lst)
                    dlay = dock.layout()
                    if dlay is not None:
                        dlay.addWidget(lst, stretch=1)
                    else:
                        lst.setParent(dock)
                dock.setVisible(True)
                self._restore_waveop_dock_in_splitter()
            self._refresh_waveop_selection_list()
            return

        self._waveop_list_external_host = host
        if dock is not None:
            dock.setVisible(False)
            self._collapse_waveop_dock_in_splitter()
        if lst.parentWidget() is not host:
            _detach(lst)
            hlay = host.layout()
            if hlay is None:
                hlay = QtWidgets.QVBoxLayout(host)
                hlay.setContentsMargins(4, 4, 4, 4)
                hlay.setSpacing(2)
            hlay.addWidget(lst)
        try:
            lst.setMaximumHeight(16777215)
        except Exception:
            pass
        self._refresh_waveop_selection_list()


    def _collapse_waveop_dock_in_splitter(self) -> None:
        split = getattr(self, "_body_splitter", None)
        dock = getattr(self, "_waveop_list_local_dock", None)
        if split is None or dock is None:
            return
        try:
            sizes = list(split.sizes())
            if len(sizes) >= 3 and int(sizes[2]) > 0:
                sizes[1] = max(1, int(sizes[1]) + int(sizes[2]))
                sizes[2] = 0
                split.setSizes(sizes)
        except Exception:
            pass


    def _restore_waveop_dock_in_splitter(self) -> None:
        split = getattr(self, "_body_splitter", None)
        dock = getattr(self, "_waveop_list_local_dock", None)
        if split is None or dock is None:
            return
        try:
            sizes = list(split.sizes())
            if len(sizes) >= 3 and int(sizes[2]) < 36:
                take = min(64, max(0, int(sizes[1]) - 80))
                sizes[1] = max(1, int(sizes[1]) - take)
                sizes[2] = max(48, take)
                split.setSizes(sizes)
        except Exception:
            pass


    def on_waveop_list_changed(self, cb: Callable[[int], None]) -> None:
        """注册 V 段数量变化回调（用于外部页签标题等）。"""
        if cb is None:
            return
        cbs = getattr(self, "_waveop_list_changed_cbs", None)
        if cbs is None:
            self._waveop_list_changed_cbs = [cb]
            return
        if cb not in cbs:
            cbs.append(cb)


    def _waveop_selections_by_apick(self) -> Dict[int, List[Dict[str, float]]]:
        """全部 V 段按 pick_word 分组（缺省字按 1）。"""
        groups: Dict[int, List[Dict[str, float]]] = {}
        for sel in list(getattr(self, "waveform_selections", []) or []):
            try:
                pw = int(sel.get("pick_word", 1) or 1)
            except Exception:
                pw = 1
            groups.setdefault(pw, []).append(sel)
        return groups


    def _refresh_waveop_selection_list(self) -> None:
        widget = getattr(self, "list_waveop_segments", None)
        if widget is None:
            return
        apick = int(self.spin_apick.value())
        count = 0
        try:
            widget.clear()
            groups = self._waveop_selections_by_apick()
            total = sum(len(v) for v in groups.values())
            count = total
            if total <= 0:
                widget.addItem("无 V 段")
            else:
                # 汇总：避免其它 apick 的 V 段“隐形”却进了姿态校正
                summary_parts = [f"apick{k}:{len(groups[k])}" for k in sorted(groups.keys())]
                phase_hint = []
                n1 = len(groups.get(1, []))
                n_sec = total - n1
                if n1:
                    phase_hint.append(f"直达{n1}")
                if n_sec:
                    phase_hint.append(f"次生{n_sec}")
                widget.addItem(
                    f"全部 {total} 段（{' / '.join(summary_parts)}）"
                    + (f" [{' + '.join(phase_hint)}]" if phase_hint else "")
                )
                def _add_colored(text: str, pick_word: int, bold: bool = False) -> None:
                    item = QtWidgets.QListWidgetItem(text)
                    color = self._waveop_apick_display_color(int(pick_word))
                    item.setForeground(QtGui.QBrush(QtGui.QColor(color.red(), color.green(), color.blue())))
                    if bold:
                        font = item.font()
                        font.setBold(True)
                        item.setFont(font)
                    widget.addItem(item)

                _add_colored(f"— 当前 apick={apick} —", apick, bold=True)
                current = list(groups.get(apick, []))
                if not current:
                    widget.addItem("(当前字无 V 段；其它字仍参与姿态校正；「清除V」只清当前字)")
                else:
                    for i, sel in enumerate(current, start=1):
                        trace_idx = int(sel.get("trace_idx", -1))
                        t_disp = float(sel.get("t_display", 0.0))
                        t0 = t_disp - 0.3
                        t1 = t_disp + 0.7
                        _add_colored(
                            f"{i:02d} 道{trace_idx} {t_disp:.3f}s [{t0:.3f},{t1:.3f}]",
                            apick,
                        )
                for pw in sorted(groups.keys()):
                    if int(pw) == int(apick):
                        continue
                    _add_colored(
                        f"— apick={pw}（{'直达' if pw == 1 else '次生'}）共 {len(groups[pw])} 段 —",
                        pw,
                        bold=True,
                    )
                    for i, sel in enumerate(groups[pw], start=1):
                        trace_idx = int(sel.get("trace_idx", -1))
                        t_disp = float(sel.get("t_display", 0.0))
                        _add_colored(f"  {i:02d} 道{trace_idx} {t_disp:.3f}s", pw)
        except Exception:
            pass
        for cb in list(getattr(self, "_waveop_list_changed_cbs", []) or []):
            try:
                cb(int(count))
            except Exception:
                pass


    def _run_waveop_stack_from_selections(self) -> None:
        """波形操作：按 V 段执行 F 同款自适应拾取更新，再叠加。"""
        if self.loaded is None:
            self.lbl_status.setText("波形叠加失败：请先加载数据")
            return
        selections = self._current_apick_waveform_selections()
        if len(selections) < 2:
            self.lbl_status.setText("波形叠加失败：当前拾取字下请先用 V 至少标注2个波形段")
            return
        traces = self.loaded.get("traces", [])
        times = np.asarray(self.loaded.get("times", []), dtype=float)
        offsets_all = np.asarray(self.loaded.get("offsets", []), dtype=float)
        if len(traces) == 0 or times.size < 4 or offsets_all.size == 0:
            self.lbl_status.setText("波形叠加失败：当前数据无效")
            return
        dt = float(times[1] - times[0]) if times.size > 1 else 0.001
        t0 = float(times[0])
        tau = np.arange(-0.3, 0.7 + 0.5 * dt, dt, dtype=np.float64)
        if tau.size < 8:
            self.lbl_status.setText("波形叠加失败：时间采样不足")
            return

        trace_indices: List[int] = []
        sel_refs: List[Dict[str, float]] = []
        for sel in selections:
            ig = int(sel.get("trace_idx", -1))
            if ig < 0 or ig >= len(traces) or ig >= offsets_all.size:
                continue
            trace_indices.append(ig)
            sel_refs.append(sel)
        if len(trace_indices) < 2:
            self.lbl_status.setText("波形叠加失败：有效 V 段不足")
            return

        proc_params = self._build_processing_params()
        raw_selected_traces = [np.asarray(traces[ig], dtype=float) for ig in trace_indices]
        selected_offsets = np.asarray([float(offsets_all[ig]) for ig in trace_indices], dtype=float)
        try:
            processed_traces = self.processor.process_traces(
                raw_selected_traces,
                times,
                selected_offsets,
                proc_params,
                realtime_interaction=False,
            )
        except Exception:
            processed_traces = raw_selected_traces

        selected_traces: List[np.ndarray] = []
        reduction_shifts: List[float] = []
        for li, ig in enumerate(trace_indices):
            red_shift = float(self._compute_reduction_tshift(int(ig), float(offsets_all[int(ig)])))
            reduction_shifts.append(red_shift)
            tr = np.asarray(processed_traces[li], dtype=float)
            tr_disp = np.interp(
                times - red_shift,
                times,
                tr,
                left=0.0,
                right=0.0,
            )
            selected_traces.append(tr_disp)

        # V 段中心作为“初始拾取”，在 F 同款自适应更新中迭代
        initial_picks: List[int] = []
        for li, sel in enumerate(sel_refs):
            t_true = float(sel.get("t_true", 0.0))
            t_display_reduced = t_true + float(reduction_shifts[li])
            ip = int(round((t_display_reduced - t0) / dt))
            initial_picks.append(ip if 0 <= ip < times.size else -1)

        try:
            result = self.adaptive_stacker.align_traces(
                traces=selected_traces,
                times=times,
                initial_picks=initial_picks,
            )
        except Exception as exc:
            self.lbl_status.setText(f"V段自适应更新失败: {exc}")
            return
        shifts = list(result.get("time_shifts", []))
        if not shifts:
            self.lbl_status.setText("V段自适应更新失败：未返回有效偏移")
            return

        updated_count = 0
        segments: List[np.ndarray] = []
        centers: List[float] = []
        for li, ig in enumerate(trace_indices):
            if li >= len(shifts) or initial_picks[li] < 0:
                continue
            old_true = float(sel_refs[li].get("t_true", 0.0))
            new_true = old_true + float(shifts[li])
            new_true = float(np.clip(new_true, t0, float(times[-1])))
            # V 段属于模板基准点：仅更新 V 基准，不写入拾取点
            sel_refs[li]["t_true"] = new_true
            sel_refs[li]["t_display"] = new_true + float(self._compute_display_tshift(int(ig), float(offsets_all[int(ig)])))
            key = (int(ig), int(sel_refs[li].get("pick_word", int(self.spin_apick.value()))))
            self._waveop_corrected_ttrue[key] = float(new_true)
            updated_count += 1

            # 按“更新后的拾取中心”截取窗口进行叠加
            t_center_display_reduced = new_true + float(reduction_shifts[li])
            seg = np.interp(
                t_center_display_reduced + tau,
                times,
                np.asarray(selected_traces[li], dtype=float),
                left=0.0,
                right=0.0,
            )
            amp = float(np.percentile(np.abs(seg), 98)) if seg.size > 0 else 0.0
            if amp > 1e-12:
                seg = seg / amp
            segments.append(np.asarray(seg, dtype=np.float64))
            centers.append(float(sel_refs[li]["t_display"]))

        if updated_count < 2 or len(segments) < 2:
            self.lbl_status.setText("波形叠加失败：可更新/可叠加的 V 段不足")
            return

        stack = np.mean(np.asarray(segments, dtype=np.float64), axis=0)
        self.waveop_stack_result = {
            "tau": tau.astype(np.float64),
            "stack": np.asarray(stack, dtype=np.float64),
            "centers": np.asarray(centers, dtype=np.float64),
        }
        self._refresh_waveop_selection_list()
        self.request_render(delay_ms=10)
        self.lbl_status.setText(
            f"V段流程完成：更新 {updated_count} 条V基准并完成叠加（处理后波形，不写入拾取）"
        )


    def _on_waveop_att_clicked(self) -> None:
        """独立 zplotpy 无姿态入口；RelocationViewer 由 ZplotAttitudeMixin 覆盖。"""
        return

