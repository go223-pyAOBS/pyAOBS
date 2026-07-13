"""
zplot_workflow_controller.py - interaction workflow controller for ZPlotGUI.

Phase 2.5 gradually migrates interaction workflows out of zplot_gui.py.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from typing import Dict, List, Optional, Tuple
import os

import numpy as np

try:
    from .adaptive_stack import AdaptiveStacker
    from .auto_picker import AutoPicker
    from .auto_pick_dialog import AutoPickDialog
    from .data_processor import DataProcessor
except ImportError:
    from adaptive_stack import AdaptiveStacker
    from auto_picker import AutoPicker
    from auto_pick_dialog import AutoPickDialog
    from data_processor import DataProcessor


class ZPlotWorkflowController:
    """Controller for high-level interaction workflows."""

    def _request_plot_refresh(self, gui, *, immediate: bool = False, delay_ms: int = 20):
        """统一请求重绘：优先走调度器，兼容旧入口。"""
        if hasattr(gui, "request_plot_refresh") and callable(getattr(gui, "request_plot_refresh")):
            gui.request_plot_refresh(immediate=immediate, delay_ms=delay_ms)
            return
        gui.update_plot()

    def _get_current_view_ranges(self, gui):
        """获取当前可见窗口范围（x/y），始终返回升序区间。"""
        x_range = None
        y_range = None

        if getattr(gui, "preserve_zoom", False) and gui.user_xlim and gui.user_ylim:
            x_range = (min(gui.user_xlim[0], gui.user_xlim[1]), max(gui.user_xlim[0], gui.user_xlim[1]))
            y_range = (min(gui.user_ylim[0], gui.user_ylim[1]), max(gui.user_ylim[0], gui.user_ylim[1]))
            return x_range, y_range

        if gui.params is not None and gui.params.xmin is not None and gui.params.xmax is not None:
            x_range = (min(gui.params.xmin, gui.params.xmax), max(gui.params.xmin, gui.params.xmax))
        if gui.params is not None and gui.params.tmin is not None and gui.params.tmax is not None:
            y_range = (min(gui.params.tmin, gui.params.tmax), max(gui.params.tmin, gui.params.tmax))
        return x_range, y_range

    def _get_window_filtered_trace_indices(self, gui, trace_headers=None) -> List[int]:
        """按当前记录/分量等过滤后，再按当前窗口筛选可见道索引。"""
        traces = gui.loaded_data["traces"]
        offsets = gui.loaded_data["offsets"]
        times = gui.loaded_data["times"]
        if trace_headers is None:
            trace_headers = gui.loaded_data.get("trace_headers", [])
        records = gui.loaded_data.get("records", [])

        temp_params = type(gui.params)(**gui.params.__dict__)
        # 过滤层不再按窗口筛，窗口筛选在这里统一处理
        temp_params.xmin = None
        temp_params.xmax = None
        temp_params.tmin = None
        temp_params.tmax = None

        filtered_data = gui.plot_manager._filter_data(
            traces, offsets, times, temp_params, trace_headers
        )
        trace_indices = filtered_data.get("indices", list(range(len(filtered_data["traces"]))))
        if not trace_indices:
            return []

        x_range, y_range = self._get_current_view_ranges(gui)
        if x_range is None and y_range is None:
            return trace_indices

        x_coordinates = gui.plot_manager._calculate_x_coordinates(
            offsets, trace_indices, trace_headers, records, gui.params
        )
        if x_coordinates is None or len(x_coordinates) != len(trace_indices):
            return trace_indices

        visible_indices: List[int] = []
        data_tmin = float(np.min(times)) if len(times) > 0 else 0.0
        data_tmax = float(np.max(times)) if len(times) > 0 else 0.0
        alignment_offsets = gui.alignment_offsets if gui.alignment_active else {}
        txadj = float(getattr(gui.params, "txadj", 0.0) or 0.0)
        vred = float(getattr(gui.params, "vred", 0.0) or 0.0)

        for i, trace_idx in enumerate(trace_indices):
            x = float(x_coordinates[i])
            if x_range is not None and (x < x_range[0] or x > x_range[1]):
                continue

            if y_range is not None:
                shift = 0.0
                if vred > 0 and trace_idx < len(offsets):
                    shift -= abs(float(offsets[trace_idx])) / vred
                if trace_idx in alignment_offsets:
                    shift += float(alignment_offsets[trace_idx])
                shift += txadj
                t0 = data_tmin + shift
                t1 = data_tmax + shift
                if t1 < y_range[0] or t0 > y_range[1]:
                    continue

            visible_indices.append(trace_idx)

        return visible_indices

    def _refresh_after_pick_updates(self, gui):
        """拾取结果更新后的刷新策略：优先轻量刷新。"""
        try:
            if (
                hasattr(gui, "_refresh_pick_overlay_only")
                and callable(gui._refresh_pick_overlay_only)
                and not getattr(gui, "alignment_active", False)
            ):
                gui._refresh_pick_overlay_only()
            else:
                self._request_plot_refresh(gui, delay_ms=20)
        except Exception:
            self._request_plot_refresh(gui, delay_ms=20)

    def perform_adaptive_stacking(self, gui):
        if not gui.data_loaded or not gui.pick_manager or not gui.params:
            gui.update_status("提示：请先加载数据")
            return

        traces = gui.loaded_data["traces"]
        offsets = gui.loaded_data["offsets"]
        times = gui.loaded_data["times"]

        if not traces or len(traces) == 0:
            gui.update_status("错误: 没有可用的道数据")
            return

        self._ensure_adaptive_stacker(gui)
        pick_word = gui.params.apick if gui.params else 1

        filtered = self._prepare_filtered_stacking_data(
            gui, traces, offsets, times, pick_word=pick_word
        )
        if filtered is None:
            return
        (
            _filtered_raw_traces,
            filtered_indices_for_stacking,
            filtered_to_original_stacking,
        ) = filtered

        visible_trace_indices = self._get_window_filtered_trace_indices(gui)
        visible_count = len(visible_trace_indices)
        picked_in_view_count = 0
        for idx in visible_trace_indices:
            pick_time = gui.pick_manager.get_pick(idx, pick_word) if gui.pick_manager else None
            if pick_time is not None and pick_time > 0:
                picked_in_view_count += 1
        used_count = len(filtered_indices_for_stacking)
        gui.update_status(
            f"自适应叠加准备：窗口可见道={visible_count}，已拾取道={picked_in_view_count}，参与叠加={used_count}"
        )

        # 只对过滤后的道执行数据处理，避免全量处理大数据
        sampling_rate, filtered_traces_for_stacking = self._prepare_processed_traces(
            gui,
            traces,
            offsets,
            times,
            trace_indices=filtered_indices_for_stacking,
        )

        original_picks, initial_picks = self._collect_stacking_pick_inputs(
            gui, filtered_indices_for_stacking, pick_word, times
        )

        if all(p < 0 for p in initial_picks):
            gui.update_status(f"提示：当前拾取字 {pick_word} 在当前选择的数据类型中没有拾取点，无法进行自适应叠加")
            return

        gui.update_status("正在进行自适应叠加对齐...")
        gui.root.update()

        try:
            result = gui.adaptive_stacker.align_traces(
                filtered_traces_for_stacking,
                times,
                initial_picks=initial_picks if any(p >= 0 for p in initial_picks) else None,
                sampling_rate=sampling_rate,
            )

            updated_picks, updated_count, adaptive_shifts = self._apply_stacking_time_shifts(
                gui, result, filtered_to_original_stacking, pick_word
            )
            self._apply_adaptive_alignment_state(gui, adaptive_shifts)

            gui.last_stacking_result = {
                "result": result,
                "original_picks": original_picks,
                "updated_picks": updated_picks,
                "traces": filtered_traces_for_stacking,
                "times": times,
                "offsets": offsets,
                "filtered_indices": filtered_indices_for_stacking,
            }

            evaluation = self._evaluate_stacking_result(
                gui,
                original_picks,
                updated_picks,
                filtered_to_original_stacking,
                result,
                filtered_traces_for_stacking,
                times,
            )
            # 相关性全对全分析复杂度 O(n^2)，道数大时会明显拖慢 F 操作；
            # 默认仅在中小规模道数执行，可用环境变量强制开启/关闭。
            corr_mode = os.environ.get("ZPLOTPY_STACK_CORR_MODE", "auto").strip().lower()
            do_corr = True
            if corr_mode == "off":
                do_corr = False
            elif corr_mode == "on":
                do_corr = True
            else:
                do_corr = len(filtered_traces_for_stacking) <= 120

            if do_corr:
                self._run_stacking_correlation_analysis(
                    gui,
                    filtered_traces_for_stacking,
                    times,
                    pick_word,
                    filtered_traces_for_stacking,
                    filtered_indices_for_stacking,
                    filtered_to_original_stacking,
                    original_picks,
                    updated_picks,
                )

            gui.show_stacking_result(result, updated_count, evaluation)
            gui.update_status(
                f"自适应叠加完成：窗口可见道={visible_count}，已拾取道={picked_in_view_count}，参与叠加={used_count}，更新拾取={updated_count}"
            )
            self._request_plot_refresh(gui, delay_ms=20)

        except Exception as e:
            gui.update_status(f"自适应叠加失败: {str(e)}")
            import traceback

            traceback.print_exc()

    def _ensure_adaptive_stacker(self, gui):
        if gui.adaptive_stacker:
            return
        gui.adaptive_stacker = AdaptiveStacker(
            nsi=gui.params.nsi,
            pjgl=gui.params.pjgl,
            stkwb=gui.params.stkwb,
            stkwl=gui.params.stkwl,
            dtcw=gui.params.dtcw,
            ratio=gui.params.hilbratio,
        )

    def _prepare_processed_traces(self, gui, traces, offsets, times, trace_indices=None):
        sampling_rate = 1.0 / (times[1] - times[0]) if len(times) > 1 else 1000.0
        data_processor = gui.plot_manager.data_processor if gui.plot_manager else DataProcessor()
        if trace_indices is not None:
            selected_traces = [traces[i] for i in trace_indices]
            selected_offsets = offsets[trace_indices] if len(offsets) > 0 else offsets
            processed_traces = data_processor.process_traces(
                selected_traces, times, selected_offsets, gui.params, sampling_rate
            )
        else:
            processed_traces = data_processor.process_traces(
                traces, times, offsets, gui.params, sampling_rate
            )
        return sampling_rate, processed_traces

    def _prepare_filtered_stacking_data(self, gui, traces, offsets, times, pick_word: int):
        trace_headers = gui.loaded_data.get("trace_headers", None)
        if gui.plot_manager and trace_headers:
            filtered_indices = self._get_window_filtered_trace_indices(gui, trace_headers=trace_headers)
        else:
            filtered_indices = list(range(len(traces)))

        if len(filtered_indices) == 0:
            gui.update_status("提示：当前窗口内没有可用道数据，无法进行自适应叠加")
            return None

        # 仅保留“已拾取道”
        picked_indices = []
        for idx in filtered_indices:
            pick_time = gui.pick_manager.get_pick(idx, pick_word) if gui.pick_manager else None
            if pick_time is not None and pick_time > 0:
                picked_indices.append(idx)

        if len(picked_indices) == 0:
            gui.update_status(f"提示：当前窗口内拾取字 {pick_word} 没有拾取点，无法进行自适应叠加")
            return None

        filtered_traces = [traces[i] for i in picked_indices]
        filtered_to_original = {
            filtered_idx: original_idx
            for filtered_idx, original_idx in enumerate(picked_indices)
        }
        return filtered_traces, picked_indices, filtered_to_original

    def _collect_stacking_pick_inputs(self, gui, filtered_indices, pick_word, times):
        original_picks = {}
        initial_picks = []
        for filtered_idx, original_idx in enumerate(filtered_indices):
            pick_time = gui.pick_manager.get_pick(original_idx, pick_word)
            if pick_time is not None and pick_time > 0:
                original_picks[filtered_idx] = pick_time
                if len(times) > 1:
                    dt = times[1] - times[0]
                    initial_picks.append(int((pick_time - times[0]) / dt))
                else:
                    initial_picks.append(0)
            else:
                initial_picks.append(-1)
        return original_picks, initial_picks

    def _apply_stacking_time_shifts(self, gui, result, filtered_to_original, pick_word):
        updated_picks = {}
        updated_count = 0
        adaptive_shifts = {}

        for filtered_idx, shift in enumerate(result["time_shifts"]):
            original_idx = filtered_to_original.get(filtered_idx, filtered_idx)
            if original_idx not in gui.pick_manager.picks:
                continue
            picks = gui.pick_manager.picks[original_idx]
            if pick_word not in picks:
                continue
            updated_picks[original_idx] = picks[pick_word] + shift

        for filtered_idx, shift in enumerate(result["time_shifts"]):
            original_idx = filtered_to_original.get(filtered_idx, filtered_idx)
            if original_idx not in gui.pick_manager.picks:
                continue
            picks = gui.pick_manager.picks[original_idx]
            if pick_word not in picks:
                continue
            picks[pick_word] += shift
            adaptive_shifts[original_idx] = shift
            updated_count += 1

        return updated_picks, updated_count, adaptive_shifts

    def _apply_adaptive_alignment_state(self, gui, adaptive_shifts):
        if gui.alignment_active:
            gui.alignment_active = False
            gui.aligned_trace_indices.clear()
        for trace_idx, shift in adaptive_shifts.items():
            gui.alignment_offsets[trace_idx] = shift

    def _evaluate_stacking_result(
        self,
        gui,
        original_picks,
        updated_picks,
        filtered_to_original,
        result,
        filtered_traces,
        times,
    ):
        return gui.stacking_evaluator.evaluate(
            original_picks=original_picks,
            updated_picks={
                filtered_idx: updated_picks.get(original_idx, 0.0)
                for filtered_idx, original_idx in filtered_to_original.items()
                if original_idx in updated_picks
            },
            time_shifts=result["time_shifts"],
            errors=result["errors"],
            quality_metric=result["quality_metric"],
            traces=filtered_traces,
            times=times,
        )

    def _run_stacking_correlation_analysis(
        self,
        gui,
        processed_traces,
        times,
        pick_word,
        filtered_traces,
        filtered_indices,
        filtered_to_original,
        original_picks,
        updated_picks,
    ):
        filtered_times = times
        filtered_original_picks = original_picks
        original_to_filtered = {
            original_idx: filtered_idx
            for filtered_idx, original_idx in enumerate(filtered_indices)
        }
        filtered_updated_picks = {}
        for original_idx, pick_time in updated_picks.items():
            if original_idx in original_to_filtered:
                filtered_updated_picks[original_to_filtered[original_idx]] = pick_time

        if len(filtered_traces) == 0:
            print("\n⚠ 警告：数据类型过滤后没有剩余的道数据")
            print("  相关性分析将使用所有数据（向后兼容）\n")
            filtered_traces = processed_traces
            filtered_indices = list(range(len(processed_traces)))
            filtered_times = times
            filtered_to_original = {i: i for i in range(len(processed_traces))}
            filtered_original_picks = {}
            filtered_updated_picks = {}
            for i in range(len(processed_traces)):
                pick_time = gui.pick_manager.get_pick(i, pick_word)
                if pick_time is not None and pick_time > 0:
                    filtered_original_picks[i] = pick_time
            filtered_updated_picks = updated_picks

            if len(filtered_original_picks) == 0 and len(filtered_updated_picks) == 0:
                print("\n⚠ 警告：过滤后的数据中没有找到拾取点")
                print(f"  过滤后的道数: {len(filtered_traces)}")
                print(f"  原始拾取点数: {len(original_picks)}")
                print(f"  更新后拾取点数: {len(updated_picks)}")
                print(f"  过滤后的索引数量: {len(filtered_indices)}")
                if len(filtered_indices) > 0:
                    if len(filtered_indices) > 10:
                        print(f"  过滤后的索引范围: {filtered_indices[:10]}...")
                    else:
                        print(f"  过滤后的索引: {filtered_indices}")
                if len(original_picks) > 0:
                    original_keys = list(original_picks.keys())
                    if len(original_keys) > 10:
                        print(f"  原始拾取点索引示例: {original_keys[:10]}...")
                    else:
                        print(f"  原始拾取点索引: {original_keys}")
                    intersection = set(filtered_indices) & set(original_keys)
                    print(f"  过滤后索引与原始拾取点索引的交集: {len(intersection)} 个")
                if len(updated_picks) > 0:
                    updated_keys = list(updated_picks.keys())
                    if len(updated_keys) > 10:
                        print(f"  更新后拾取点索引示例: {updated_keys[:10]}...")
                    else:
                        print(f"  更新后拾取点索引: {updated_keys}")
                    intersection = set(filtered_indices) & set(updated_keys)
                    print(f"  过滤后索引与更新后拾取点索引的交集: {len(intersection)} 个")
                print()
            elif len(filtered_original_picks) > 0 or len(filtered_updated_picks) > 0:
                print(
                    f"\n✓ 找到过滤后的拾取点: 原始={len(filtered_original_picks)}, 更新后={len(filtered_updated_picks)}"
                )

            if len(filtered_original_picks) == 0:
                print("\n⚠ 警告：过滤后的原始拾取点字典为空")
                print(f"  过滤后的道数: {len(filtered_traces)}")
                print(f"  原始拾取点数: {len(original_picks)}")
                print(f"  过滤后的索引数量: {len(filtered_indices)}")
                if len(original_picks) > 0 and len(filtered_indices) > 0:
                    original_keys = list(original_picks.keys())
                    intersection = set(filtered_indices) & set(original_keys)
                    print(f"  索引交集: {len(intersection)} 个")
                    if len(intersection) > 0:
                        print(f"  交集示例: {list(intersection)[:5]}")
                print()

            if len(filtered_traces) > 0:
                trace_len = len(filtered_traces[0])
                times_len = len(filtered_times)
                if trace_len != times_len:
                    print(
                        f"\n⚠ 警告：过滤后的道数据长度 ({trace_len}) 与时间数组长度 ({times_len}) 不匹配"
                    )
                    print("  这可能导致相关性分析失败")
                    print()

        correlation_before = gui.adaptive_stacker.calculate_correlation(
            traces=filtered_traces,
            times=filtered_times,
            picks=filtered_original_picks,
            window_half_length=None,
        )
        correlation_after = gui.adaptive_stacker.calculate_correlation(
            traces=filtered_traces,
            times=filtered_times,
            picks=filtered_updated_picks,
            window_half_length=None,
        )

        if "valid_trace_indices" in correlation_before:
            correlation_before["valid_trace_indices"] = [
                filtered_to_original.get(idx, idx)
                for idx in correlation_before["valid_trace_indices"]
            ]
        if "valid_trace_indices" in correlation_after:
            correlation_after["valid_trace_indices"] = [
                filtered_to_original.get(idx, idx)
                for idx in correlation_after["valid_trace_indices"]
            ]

        gui._print_correlation_report(correlation_before, correlation_after, original_picks, updated_picks)

    def auto_pick(self, gui):
        if not gui.data_loaded or not gui.pick_manager:
            messagebox.showwarning("警告", "请先加载数据文件")
            return

        dialog_result = self._show_auto_pick_dialog(gui)
        if dialog_result is None:
            return
        action, params = dialog_result

        traces, times, offsets, trace_headers = self._prepare_auto_pick_context(
            gui
        )
        gui.auto_picker = self._build_auto_picker(gui, params, offsets)

        trace_indices = self.get_trace_indices_for_auto_pick(
            gui,
            params["range_type"],
            params.get("trace_start"),
            params.get("trace_end"),
            trace_headers=trace_headers,
        )
        if len(trace_indices) == 0:
            messagebox.showwarning("警告", "没有可拾取的道")
            return

        # 仅处理当前需要自动拾取的道，避免全量处理
        selected_traces = [traces[i] for i in trace_indices]
        selected_offsets = offsets[trace_indices] if len(offsets) > 0 else offsets
        sampling_rate = 1.0 / (times[1] - times[0]) if len(times) > 1 else 1000.0
        data_processor = gui.plot_manager.data_processor if gui.plot_manager else DataProcessor()
        processed_selected_traces = data_processor.process_traces(
            selected_traces, times, selected_offsets, gui.params, sampling_rate
        )

        if action == "preview":
            self.preview_auto_pick(
                gui,
                processed_selected_traces,
                times,
                trace_indices,
                selected_offsets,
                params,
            )
        else:
            self.apply_auto_pick(
                gui,
                processed_selected_traces,
                times,
                trace_indices,
                selected_offsets,
                params,
            )

    def _show_auto_pick_dialog(self, gui):
        current_pick_word = gui.params.apick if gui.params else 1
        default_params = {
            "window_length": 0.1,
            "min_energy_ratio": 1.5,
            "search_start": None,
            "search_end": None,
            "pick_word": current_pick_word,
        }
        dialog = AutoPickDialog(gui.root, default_params)
        result = dialog.show()
        if result is None:
            return None
        return result["action"], result["params"]

    def _prepare_auto_pick_context(self, gui):
        traces = gui.loaded_data["traces"]
        times = gui.loaded_data["times"]
        offsets = gui.loaded_data["offsets"]
        trace_headers = gui.loaded_data.get("trace_headers", [])
        return traces, times, offsets, trace_headers

    def _build_auto_picker(self, gui, params, offsets):
        search_start = params.get("search_start")
        search_end = params.get("search_end")
        vred = gui.params.vred if gui.params else 0.0
        avg_offset = np.mean(np.abs(offsets)) if len(offsets) > 0 else None
        return AutoPicker(
            window_length=params["window_length"],
            min_energy_ratio=params["min_energy_ratio"],
            search_start=search_start if search_start is not None else None,
            search_end=search_end if search_end is not None else None,
            vred=vred,
            offset=avg_offset,
        )

    def get_trace_indices_for_auto_pick(
        self,
        gui,
        range_type,
        start_trace=None,
        end_trace=None,
        trace_headers=None,
    ):
        if trace_headers is None:
            trace_headers = gui.loaded_data.get("trace_headers", [])

        if range_type == "all":
            return list(range(len(gui.loaded_data["traces"])))
        if range_type == "visible":
            return self._get_window_filtered_trace_indices(gui, trace_headers=trace_headers)

        if start_trace is None or end_trace is None:
            return []
        return list(range(start_trace - 1, min(end_trace, len(gui.loaded_data["traces"]))))

    def preview_auto_pick(self, gui, traces, times, trace_indices, selected_offsets, params):
        gui.update_status("正在预览自动拾取结果...")
        gui.root.update()

        results = gui.auto_picker.pick_traces(
            traces,
            times,
            offsets=selected_offsets,
            progress_callback=lambda current, total: gui.root.update(),
        )

        success_count = sum(1 for r in results if r is not None)
        total_count = len(results)
        preview_msg = (
            f"预览完成：共 {total_count} 道，成功拾取 {success_count} 道\n"
            f"是否应用这些拾取结果？"
        )

        if messagebox.askyesno("预览结果", preview_msg):
            self.apply_auto_pick_results(gui, traces, times, trace_indices, results, params)
        else:
            gui.update_status("预览已取消")

    def apply_auto_pick(self, gui, traces, times, trace_indices, selected_offsets, params):
        gui.update_status("正在执行自动拾取...")
        gui.root.update()

        results = gui.auto_picker.pick_traces(
            traces,
            times,
            offsets=selected_offsets,
            progress_callback=lambda current, total: gui.root.update(),
        )

        self.apply_auto_pick_results(gui, traces, times, trace_indices, results, params)

    def apply_auto_pick_results(
        self, gui, traces, times, trace_indices, results, params
    ):
        pick_word = params["pick_word"]
        success_count = 0

        for i, result in enumerate(results):
            if result is None:
                continue
            trace_idx = trace_indices[i]
            pick_time = result["pick_time"]
            gui.pick_manager.add_pick(trace_idx, pick_time, pick_word)
            success_count += 1

        gui.num_picks = gui.pick_manager.count_picks()
        gui.update_status(f"自动拾取完成：成功拾取 {success_count} 道")
        self._refresh_after_pick_updates(gui)

    def on_interpolation_correlation_picking(self, gui):
        if not gui.data_loaded or not gui.pick_manager or not gui.params:
            gui.update_status("提示：请先加载数据并进入拾取模式")
            return

        pick_word, existing_picks = self._collect_existing_picks_for_interpolation(gui)
        visible_set = set(self._get_window_filtered_trace_indices(gui))
        if visible_set:
            existing_picks = [p for p in existing_picks if p[0] in visible_set]
        existing_picks.sort(key=lambda x: x[0])
        if len(existing_picks) < 2:
            gui.update_status(
                f"提示：当前拾取字 {pick_word} 的拾取点少于2个（当前有{len(existing_picks)}个），无法进行插值-相关拾取"
            )
            return

        self.perform_interpolation_correlation_picking_batch(gui, existing_picks, pick_word, visible_indices=visible_set)

    def perform_interpolation_correlation_picking_batch(self, gui, existing_picks, pick_word, visible_indices=None):
        if len(existing_picks) < 2:
            return

        (
            processed_traces,
            offsets,
            times,
            local_to_original,
            original_to_local,
        ) = self._prepare_interpolation_context(
            gui,
            existing_picks,
            visible_indices=visible_indices,
        )
        visible_count = len(visible_indices) if visible_indices is not None else 0
        picked_in_view_count = len(existing_picks)
        used_count = len(local_to_original)
        gui.update_status(
            f"插值-相关准备：窗口可见道={visible_count}，已拾取道={picked_in_view_count}，参与插值道={used_count}"
        )
        correlation_window, search_range, correlation_window_samples, search_range_samples, hilbert_ratio = (
            self._build_interpolation_parameters(gui, times)
        )

        gui.update_status(
            f"正在进行插值-相关自动拾取（找到{len(existing_picks)}个拾取点，将在相邻拾取点之间拾取）..."
        )
        gui.root.update()

        try:
            all_picks = {}
            total_pairs = 0
            total_traces_between = 0

            for i in range(len(existing_picks) - 1):
                first_pick_original = existing_picks[i]
                second_pick_original = existing_picks[i + 1]
                trace_idx1_original, _ = first_pick_original
                trace_idx2_original, _ = second_pick_original
                if trace_idx1_original not in original_to_local or trace_idx2_original not in original_to_local:
                    continue
                first_pick_local = (
                    original_to_local[trace_idx1_original],
                    first_pick_original[1],
                )
                second_pick_local = (
                    original_to_local[trace_idx2_original],
                    second_pick_original[1],
                )

                pair_result = self._run_interpolation_for_pick_pair(
                    gui=gui,
                    processed_traces=processed_traces,
                    times=times,
                    offsets=offsets,
                    first_pick=first_pick_local,
                    second_pick=second_pick_local,
                    correlation_window_samples=correlation_window_samples,
                    search_range_samples=search_range_samples,
                    hilbert_ratio=hilbert_ratio,
                    local_to_original=local_to_original,
                )
                if pair_result is None:
                    continue
                picks, traces_between = pair_result
                total_pairs += 1
                total_traces_between += traces_between
                for trace_idx, pick_time in picks.items():
                    if trace_idx not in all_picks:
                        all_picks[trace_idx] = pick_time

            success_count = 0
            for trace_idx, pick_time in all_picks.items():
                if gui.pick_manager.add_pick(trace_idx, pick_time, pick_word):
                    success_count += 1

            gui.num_picks = gui.pick_manager.get_statistics()["total_picks"]
            gui.update_status(
                f"插值-相关完成：窗口可见道={visible_count}，已拾取道={picked_in_view_count}，参与插值道={used_count}；"
                f"在{total_pairs}对相邻拾取点之间，成功拾取 {success_count}/{total_traces_between} 道 "
                f"(拾取字{pick_word}, 窗口{correlation_window:.3f}s, 搜索范围{search_range:.3f}s)"
            )
            self._refresh_after_pick_updates(gui)

        except Exception as e:
            import traceback

            error_msg = f"插值-相关拾取失败: {str(e)}\n{traceback.format_exc()}"
            gui.update_status(error_msg)
            messagebox.showerror("错误", error_msg)

    def _collect_existing_picks_for_interpolation(self, gui):
        pick_word = gui.params.apick if gui.params else 1
        existing_picks = []
        for trace_idx, pick_dict in gui.pick_manager.picks.items():
            if pick_word not in pick_dict:
                continue
            pick_time = pick_dict[pick_word]
            if pick_time and pick_time > 0:
                existing_picks.append((trace_idx, pick_time))
        return pick_word, existing_picks

    def _prepare_interpolation_context(self, gui, existing_picks, visible_indices=None):
        traces = gui.loaded_data["traces"]
        offsets = gui.loaded_data["offsets"]
        times = gui.loaded_data["times"]

        # 仅处理“相邻拾取点之间”的最小道集合，避免全量处理
        needed_indices_set = set()
        if existing_picks and len(existing_picks) >= 2:
            for i in range(len(existing_picks) - 1):
                idx1 = int(existing_picks[i][0])
                idx2 = int(existing_picks[i + 1][0])
                lo = min(idx1, idx2)
                hi = max(idx1, idx2)
                for idx in range(lo, hi + 1):
                    if visible_indices is not None and idx not in visible_indices:
                        continue
                    needed_indices_set.add(idx)
        else:
            needed_indices_set = set(range(len(traces)))

        needed_indices = sorted(idx for idx in needed_indices_set if 0 <= idx < len(traces))
        if len(needed_indices) == 0:
            needed_indices = list(range(len(traces)))

        selected_traces = [traces[i] for i in needed_indices]
        selected_offsets = offsets[needed_indices] if len(offsets) > 0 else offsets

        sampling_rate = 1.0 / (times[1] - times[0]) if len(times) > 1 else 1000.0
        data_processor = gui.plot_manager.data_processor if gui.plot_manager else DataProcessor()
        processed_traces = data_processor.process_traces(
            selected_traces, times, selected_offsets, gui.params, sampling_rate
        )

        local_to_original = needed_indices
        original_to_local = {orig_idx: local_idx for local_idx, orig_idx in enumerate(local_to_original)}
        return processed_traces, selected_offsets, times, local_to_original, original_to_local

    def _build_interpolation_parameters(self, gui, times):
        correlation_window = 0.1
        search_range = 0.05
        hilbert_ratio = getattr(gui.params, "hilbratio", 1.0)
        dt = times[1] - times[0] if len(times) > 1 else 0.001
        correlation_window_samples = int(round(correlation_window / dt))
        search_range_samples = int(round(search_range / dt))
        if correlation_window_samples <= 0:
            correlation_window_samples = int(round(0.1 / dt))
        if search_range_samples <= 0:
            search_range_samples = int(round(0.05 / dt))
        return (
            correlation_window,
            search_range,
            correlation_window_samples,
            search_range_samples,
            hilbert_ratio,
        )

    def _run_interpolation_for_pick_pair(
        self,
        gui,
        processed_traces,
        times,
        offsets,
        first_pick,
        second_pick,
        correlation_window_samples,
        search_range_samples,
        hilbert_ratio,
        local_to_original,
    ):
        trace_idx1, pick_time1 = first_pick
        trace_idx2, pick_time2 = second_pick

        min_idx = min(trace_idx1, trace_idx2)
        max_idx = max(trace_idx1, trace_idx2)
        if max_idx - min_idx <= 1:
            return None

        picks = gui.interpolation_correlation_picker.interpolation_correlation_picking(
            traces=processed_traces,
            times=times,
            offsets=offsets,
            pick1_idx=trace_idx1,
            pick2_idx=trace_idx2,
            pick1_time=pick_time1,
            pick2_time=pick_time2,
            correlation_window=correlation_window_samples,
            search_range=search_range_samples,
            hilbert_ratio=hilbert_ratio,
            apply_filter=gui.params.ibndps == 1,
            filter_params={
                "freqlo": gui.params.freqlo,
                "freqhi": gui.params.freqhi,
                "npoles": gui.params.npoles,
                "izerop": gui.params.izerop,
            }
            if gui.params.ibndps == 1
            else None,
            apply_hilbert=getattr(gui.params, "ihilbt", 0) != 0,
            force_pick=False,
        )
        # 将局部道索引映射回原始道索引
        mapped_picks = {}
        for local_idx, pick_time in picks.items():
            if 0 <= local_idx < len(local_to_original):
                mapped_picks[local_to_original[local_idx]] = pick_time
        return mapped_picks, (max_idx - min_idx - 1)

    def calculate_static_correction_dialog(self, gui):
        if not gui.data_loaded or not gui.pick_manager:
            messagebox.showwarning("提示", "请先加载数据")
            return

        dialog = tk.Toplevel(gui.root)
        dialog.title("计算静校正")
        dialog.geometry("480x280")
        dialog.transient(gui.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        default_sigma = 3.0
        default_smooth_factor = 0.1

        input_frame = ttk.Frame(dialog)
        input_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        sigma_frame = ttk.Frame(input_frame)
        sigma_frame.pack(fill=tk.X, pady=5)
        ttk.Label(sigma_frame, text="高斯核标准差 (sigma, 单位: km):").pack(side=tk.LEFT, padx=5)
        sigma_var = tk.DoubleVar(value=default_sigma)
        sigma_entry = ttk.Entry(sigma_frame, textvariable=sigma_var, width=15)
        sigma_entry.pack(side=tk.LEFT, padx=5)

        smooth_frame = ttk.Frame(input_frame)
        smooth_frame.pack(fill=tk.X, pady=10)

        smooth_label_frame = ttk.Frame(smooth_frame)
        smooth_label_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(smooth_label_frame, text="平滑因子 (0-1):").pack(side=tk.LEFT, padx=5)
        smooth_value_label = ttk.Label(
            smooth_label_frame,
            text=f"{default_smooth_factor:.3f}",
            font=("Arial", 10, "bold"),
            foreground="blue",
        )
        smooth_value_label.pack(side=tk.LEFT, padx=5)

        smooth_var = tk.DoubleVar(value=default_smooth_factor)
        smooth_scale = ttk.Scale(
            smooth_frame,
            from_=0.0,
            to=1.0,
            variable=smooth_var,
            orient=tk.HORIZONTAL,
            length=350,
            command=lambda v: smooth_value_label.config(text=f"{float(v):.3f}"),
        )
        smooth_scale.pack(fill=tk.X, padx=10, pady=5)

        scale_labels_frame = ttk.Frame(smooth_frame)
        scale_labels_frame.pack(fill=tk.X, padx=10)
        ttk.Label(scale_labels_frame, text="0.0", font=("Arial", 8)).pack(side=tk.LEFT)
        ttk.Label(scale_labels_frame, text="0.5", font=("Arial", 8)).pack(side=tk.LEFT, expand=True)
        ttk.Label(scale_labels_frame, text="1.0", font=("Arial", 8)).pack(side=tk.RIGHT)

        info_text = (
            "静校正功能：从所选走时点中提取短波长时间变化\n\n"
            "参数说明：\n"
            "• sigma: 高斯核标准差（建议：2-5 km）\n"
            "• 平滑因子: 控制曲线平滑度（0-1，建议：0.05-0.2）"
        )
        info_label = ttk.Label(dialog, text=info_text, justify=tk.LEFT, font=("Arial", 9))
        info_label.pack(pady=5, padx=10)

        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        def on_ok():
            try:
                sigma = sigma_var.get()
                smoothness = smooth_var.get()
                if sigma <= 0:
                    messagebox.showerror("错误", "sigma必须大于0")
                    return
                if smoothness < 0 or smoothness > 1:
                    messagebox.showerror("错误", "平滑因子必须在0-1之间")
                    return
                dialog.destroy()
                self.calculate_static_correction(gui, sigma=sigma, smoothness=smoothness)
            except ValueError:
                messagebox.showerror("错误", "请输入有效的数值")

        ttk.Button(button_frame, text="确定", command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="取消", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

        sigma_entry.bind("<Return>", lambda e: on_ok())
        sigma_entry.focus()

    def calculate_static_correction(self, gui, sigma=3.0, smoothness=0.1):
        if not gui.data_loaded or not gui.pick_manager or not gui.params:
            gui.update_status("提示：请先加载数据")
            return

        static_context = self._prepare_static_correction_context(gui)
        if static_context is None:
            return
        pick_word, picks, trace_indices, x_coordinates, display_times = static_context

        try:
            static_corrections = gui.static_corrector.extract_short_wavelength_gaussian(
                picks=picks,
                trace_indices=trace_indices,
                x_coords=x_coordinates,
                pick_word=pick_word,
                sigma=sigma,
                min_picks=3,
                display_times=display_times,
                smoothness=smoothness,
            )
            if not static_corrections:
                gui.update_status("提示：无法计算静校正量（拾取数量不足或计算失败）")
                return

            min_corr, max_corr, mean_corr = self._apply_static_corrections_to_gui(
                gui, static_corrections, smoothness
            )
            gui.update_status(
                f"静校正计算完成：{len(static_corrections)} 个道，"
                f"范围 [{min_corr:.4f}, {max_corr:.4f}] 秒，"
                f"平均 {mean_corr:.4f} 秒 (sigma={sigma:.2f}km, smoothness={smoothness:.2f})"
            )

            result = messagebox.askyesno(
                "静校正计算完成",
                f"静校正计算完成：{len(static_corrections)} 个道\n"
                f"范围 [{min_corr:.4f}, {max_corr:.4f}] 秒\n"
                f"平均 {mean_corr:.4f} 秒\n"
                f"sigma={sigma:.2f}km, smoothness={smoothness:.2f}\n\n"
                f"是否应用静校正？\n\n"
                f"• 是：应用静校正到波形和震相（设置静校正开关=1）\n"
                f"• 否：仅显示校正后的震相曲线预览（虚线，设置静校正开关=0）",
            )

            if gui.params:
                if result:
                    gui.params.imute = 1
                    gui.update_status("已应用静校正（静校正开关=1）")
                else:
                    gui.params.imute = 0
                    gui.update_status("静校正预览模式（静校正开关=0），虚线显示校正后的震相曲线")
                if hasattr(gui, "top_toolbar") and gui.top_toolbar:
                    gui.top_toolbar.update_widgets_from_params()

            self._request_plot_refresh(gui, immediate=True)

        except Exception as e:
            import traceback

            error_msg = f"静校正计算失败: {str(e)}\n{traceback.format_exc()}"
            gui.update_status(error_msg)
            messagebox.showerror("错误", error_msg)

    def _prepare_static_correction_context(self, gui):
        pick_word = gui.params.apick
        picks = gui.pick_manager.get_all_picks()
        if not picks:
            gui.update_status("提示：请先进行拾取")
            return None

        has_picks = False
        for trace_idx, pick_dict in picks.items():
            if pick_word in pick_dict and pick_dict[pick_word] > 0:
                has_picks = True
                break
        if not has_picks:
            gui.update_status(f"提示：拾取字 {pick_word} 没有拾取数据")
            return None

        traces = gui.loaded_data["traces"]
        offsets = gui.loaded_data["offsets"]
        trace_headers = gui.loaded_data.get("trace_headers", [])
        records = gui.loaded_data.get("records", [])
        filtered_data = gui.plot_manager._filter_data(
            traces, offsets, gui.loaded_data["times"], gui.params, trace_headers
        )
        trace_indices = filtered_data.get("indices", list(range(len(filtered_data["traces"]))))
        if not trace_indices:
            gui.update_status("提示：没有可用的道数据")
            return None

        x_coordinates = gui.plot_manager._calculate_x_coordinates(
            offsets, trace_indices, trace_headers, records, gui.params
        )
        display_times = self._calculate_display_times_for_static(
            gui, picks, pick_word, trace_indices, offsets
        )
        return pick_word, picks, trace_indices, x_coordinates, display_times

    def _calculate_display_times_for_static(self, gui, picks, pick_word, trace_indices, offsets):
        display_times = {}
        alignment_offsets = gui.alignment_offsets if gui.alignment_active else None
        for trace_idx in trace_indices:
            if trace_idx not in picks or pick_word not in picks[trace_idx]:
                continue
            pick_time = picks[trace_idx][pick_word]
            if pick_time <= 0:
                continue
            display_time = pick_time
            if trace_idx < len(offsets):
                original_offset = offsets[trace_idx]
                if gui.params.vred > 0:
                    display_time = pick_time - abs(original_offset) / gui.params.vred
            if alignment_offsets and trace_idx in alignment_offsets:
                display_time = display_time + alignment_offsets[trace_idx]
            if gui.params.txadj != 0:
                display_time = display_time + gui.params.txadj
            display_times[trace_idx] = display_time
        return display_times

    def _apply_static_corrections_to_gui(self, gui, static_corrections, smoothness):
        gui.static_corrector.set_corrections(static_corrections)
        gui.static_correction_enabled = True

        if gui.plot_manager:
            gui.plot_manager.static_correction_smoothness = smoothness
            gui.static_correction_smoothness_var.set(smoothness)
            gui.static_correction_smoothness_label.config(text=f"{smoothness:.3f}")

        correction_values = list(static_corrections.values())
        min_corr = min(correction_values)
        max_corr = max(correction_values)
        mean_corr = sum(correction_values) / len(correction_values)
        return min_corr, max_corr, mean_corr

    def on_static_correction_smoothness_changed(self, gui, value, immediate=True):
        try:
            smoothness = float(value)
            gui.static_correction_smoothness_label.config(text=f"{smoothness:.3f}")

            if not gui.plot_manager:
                return
            gui.plot_manager.static_correction_smoothness = smoothness

            if not (
                gui.plot_manager.static_correction_preview_mode
                and gui.plot_manager.static_corrections
            ):
                return
            if not (gui.data_loaded and gui.pick_manager):
                return

            picks = gui.pick_manager.get_all_picks()
            if not picks:
                return

            offsets = gui.loaded_data.get("offsets", np.array([]))
            trace_indices = gui.plot_manager.current_filtered_indices
            x_coordinates = gui.plot_manager.current_x_coordinates
            alignment_offsets = gui.alignment_offsets if gui.alignment_active else None

            if hasattr(gui.plot_manager, "static_correction_curve_lines"):
                for line in list(gui.plot_manager.static_correction_curve_lines):
                    try:
                        if line in gui.plot_manager.ax.lines:
                            line.remove()
                    except Exception:
                        pass
                gui.plot_manager.static_correction_curve_lines = []

            lines_to_remove = []
            for line in list(gui.plot_manager.ax.lines):
                try:
                    label = line.get_label() or ""
                    if line.get_linestyle() == "--" and "静校正后震相曲线" in label:
                        lines_to_remove.append(line)
                except Exception:
                    pass

            for line in lines_to_remove:
                try:
                    if line in gui.plot_manager.ax.lines:
                        line.remove()
                except Exception:
                    pass

            gui.plot_manager.plot_picks(
                picks, offsets, gui.params, trace_indices, alignment_offsets, x_coordinates
            )

            if gui.canvas:
                gui.canvas.draw()
                gui.root.after_idle(lambda: gui.canvas.flush_events())

        except Exception as e:
            import traceback

            print(f"滑块更新错误: {e}")
            print(traceback.format_exc())

    def clear_static_correction(self, gui):
        gui.static_corrector.clear_corrections()
        gui.static_correction_enabled = False
        if hasattr(gui, "static_correction_slider_frame"):
            gui.static_correction_slider_frame.grid_remove()
        gui.update_status("已清除静校正")
        self._request_plot_refresh(gui, immediate=True)
