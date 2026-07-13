"""
zplot_gui_services.py - GUI orchestration services for ZPlotGUI.

This module extracts heavy orchestration logic from zplot_gui.py:
- workbench state persistence/restore
- plot update orchestration
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from tkinter import messagebox


class WorkbenchStateService:
    """Persist and restore GUI state for ZPlotGUI."""

    def load_state(self, gui) -> dict:
        if gui._gui_state_file is None or not gui._gui_state_file.exists():
            return {}
        try:
            raw = json.loads(gui._gui_state_file.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return {}
            obj = raw.get("zplotpy_gui", {})
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def save_state(self, gui) -> None:
        if gui._gui_state_file is None:
            return
        try:
            raw: dict = {}
            if gui._gui_state_file.exists():
                loaded = json.loads(gui._gui_state_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    raw = loaded
            params_dict = {}
            if hasattr(gui, "top_toolbar") and gui.top_toolbar and hasattr(gui.top_toolbar, "params"):
                try:
                    params_dict = gui.top_toolbar.params.to_dict()
                except Exception:
                    params_dict = {}
            panel_order = []
            if hasattr(gui, "top_toolbar") and gui.top_toolbar and hasattr(gui.top_toolbar, "get_panel_order"):
                try:
                    panel_order = gui.top_toolbar.get_panel_order()
                except Exception:
                    panel_order = []
            raw["zplotpy_gui"] = {
                "dfile_path": gui.dfile_path or "",
                "hfile_path": gui.hfile_path or "",
                "rfile_path": gui.rfile_path or "",
                "current_record": int(gui.current_record or 1),
                "params": params_dict,
                "toolbar_panel_order": panel_order,
            }
            gui._gui_state_file.parent.mkdir(parents=True, exist_ok=True)
            gui._gui_state_file.write_text(
                json.dumps(raw, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    def restore_state(self, gui) -> None:
        state = self.load_state(gui)
        if not state:
            return
        restored: list[str] = []
        dfile = str(state.get("dfile_path", "")).strip()
        hfile = str(state.get("hfile_path", "")).strip()
        rfile = str(state.get("rfile_path", "")).strip()
        dfile = self.resolve_restorable_input_path(gui, dfile)
        hfile = self.resolve_restorable_input_path(gui, hfile)
        rfile = self.resolve_restorable_input_path(gui, rfile)
        if not dfile or not Path(dfile).exists():
            return
        ok = gui.load_data_files(
            dfile_path=dfile,
            hfile_path=hfile if hfile and Path(hfile).exists() else None,
            rfile_path=rfile if rfile and Path(rfile).exists() else None,
            async_mode=False,
        )
        if not ok:
            return
        restored.append(f"数据: {Path(dfile).name}")
        params_dict = state.get("params", {})
        if isinstance(params_dict, dict) and hasattr(gui, "top_toolbar") and gui.top_toolbar:
            try:
                gui.top_toolbar.params.from_dict(params_dict)
                gui.params = gui.top_toolbar.params
                gui.top_toolbar.update_widgets_from_params()
                restored.append("参数")
            except Exception:
                pass
        panel_order = state.get("toolbar_panel_order", [])
        if (
            isinstance(panel_order, list)
            and hasattr(gui, "top_toolbar")
            and gui.top_toolbar
            and hasattr(gui.top_toolbar, "set_panel_order")
        ):
            try:
                gui.top_toolbar.set_panel_order(panel_order)
                restored.append("工具栏布局")
            except Exception:
                pass
        try:
            record = int(state.get("current_record", 1) or 1)
        except (TypeError, ValueError):
            record = 1
        if record > 1:
            gui.switch_to_record(record)
            restored.append(f"记录号: {record}")
        else:
            gui.update_plot()
        if restored:
            state_path = str(gui._gui_state_file) if gui._gui_state_file is not None else "(未配置)"
            gui.update_status("已恢复上次会话状态（" + "，".join(restored) + f"），状态文件: {state_path}")

    def resolve_restorable_input_path(self, gui, path_text: str) -> str:
        raw = str(path_text or "").strip()
        if not raw:
            return ""
        for candidate in self.candidate_existing_paths(raw):
            if candidate.exists():
                return str(candidate)
        if gui._gui_state_file is None or not gui._gui_state_file.exists():
            return raw
        try:
            obj = json.loads(gui._gui_state_file.read_text(encoding="utf-8"))
            if not isinstance(obj, dict):
                return raw
            backup_map = obj.get("input_backups", {})
            if not isinstance(backup_map, dict):
                return raw
            src_text = ""
            for key in self.candidate_path_keys(raw):
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
            for cand in self.candidate_existing_paths(src_text):
                if cand.exists():
                    src = cand
                    break
            if src is None or gui._run_inputs_dir is None:
                return str(src) if src is not None else raw
            gui._run_inputs_dir.mkdir(parents=True, exist_ok=True)
            dest = gui._run_inputs_dir / src.name
            if not dest.exists():
                shutil.copy2(src, dest)
            backup_map[str(dest)] = str(src)
            obj["input_backups"] = backup_map
            gui._gui_state_file.write_text(
                json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return str(dest)
        except Exception:
            return raw

    @staticmethod
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

    @classmethod
    def candidate_existing_paths(cls, path_text: str) -> list[Path]:
        candidates: list[Path] = []
        for key in cls.candidate_path_keys(path_text):
            try:
                candidates.append(Path(key).expanduser())
            except Exception:
                continue
        return candidates


class PlotRefreshScheduler:
    """Debounced scheduler that serializes heavy plot refresh calls."""

    def __init__(self, default_delay_ms: int = 90):
        self.default_delay_ms = max(0, int(default_delay_ms))

    def request(self, gui, delay_ms: int | None = None, immediate: bool = False) -> None:
        if not hasattr(gui, "root") or gui.root is None:
            return
        try:
            if not gui.root.winfo_exists():
                return
        except Exception:
            return

        if not hasattr(gui, "_plot_refresh_after_id"):
            gui._plot_refresh_after_id = None
        if not hasattr(gui, "_plot_refresh_in_progress"):
            gui._plot_refresh_in_progress = False
        if not hasattr(gui, "_plot_refresh_pending"):
            gui._plot_refresh_pending = False

        if immediate:
            self._cancel_pending(gui)
            self._run(gui)
            return

        wait_ms = self.default_delay_ms if delay_ms is None else max(0, int(delay_ms))
        self._cancel_pending(gui)
        gui._plot_refresh_after_id = gui.root.after(wait_ms, lambda: self._run(gui))

    def _cancel_pending(self, gui) -> None:
        after_id = getattr(gui, "_plot_refresh_after_id", None)
        if after_id is None:
            return
        try:
            gui.root.after_cancel(after_id)
        except Exception:
            pass
        gui._plot_refresh_after_id = None

    def _run(self, gui) -> None:
        gui._plot_refresh_after_id = None
        if getattr(gui, "_plot_refresh_in_progress", False):
            gui._plot_refresh_pending = True
            return

        gui._plot_refresh_in_progress = True
        try:
            gui._plot_update_service.update_plot(gui)
        finally:
            gui._plot_refresh_in_progress = False
            if getattr(gui, "_plot_refresh_pending", False):
                gui._plot_refresh_pending = False
                self.request(gui, delay_ms=25, immediate=False)


class PlotUpdateService:
    """Orchestrate heavy plot refresh logic for ZPlotGUI."""

    def update_plot(self, gui) -> None:
        # 检查数据是否已加载
        if not gui.data_loaded or not gui.loaded_data:
            gui.update_status("无法绘图：数据未加载")
            return

        # 检查绘图管理器是否已初始化
        if not gui.plot_manager:
            gui.update_status("无法绘图：绘图管理器未初始化")
            return

        # 检查参数对象是否已设置
        if gui.params is None:
            # 尝试从工具栏获取参数对象
            if hasattr(gui, "top_toolbar") and hasattr(gui.top_toolbar, "params"):
                gui.params = gui.top_toolbar.params
            if gui.params is None:
                gui.update_status("无法绘图：参数对象未设置")
                return

        try:
            # 若用户正在自定义视窗（平移/缩放），优先使用当前视窗参数
            if gui.preserve_zoom and gui.user_xlim and gui.user_ylim and gui.params is not None:
                gui.params.xmin = min(gui.user_xlim[0], gui.user_xlim[1])
                gui.params.xmax = max(gui.user_xlim[0], gui.user_xlim[1])
                gui.params.tmin = min(gui.user_ylim[0], gui.user_ylim[1])
                gui.params.tmax = max(gui.user_ylim[0], gui.user_ylim[1])

            try:
                zview_msg = (
                    "[ZVIEW] "
                    f"preserve_zoom={bool(getattr(gui, 'preserve_zoom', False))} "
                    f"user_xlim={getattr(gui, 'user_xlim', None)} "
                    f"user_ylim={getattr(gui, 'user_ylim', None)} "
                    f"params_window="
                    f"[{float(gui.params.xmin):.6g}, {float(gui.params.xmax):.6g}]x"
                    f"[{float(gui.params.tmin):.6g}, {float(gui.params.tmax):.6g}]"
                )
                print(zview_msg)
            except Exception:
                pass

            # 获取数据
            traces = gui.loaded_data["traces"]
            offsets = gui.loaded_data["offsets"]
            times = gui.loaded_data["times"]
            trace_headers = gui.loaded_data.get("trace_headers", [])

            if not traces or len(traces) == 0:
                gui.update_status("无法绘图：无道数据")
                return

            # 更新状态
            gui.update_status("正在绘制...")

            # 使用绘图管理器绘制（会自动清空之前的图形）
            # 传递对齐偏移量（如果启用对齐）
            alignment_offsets = gui.alignment_offsets if gui.alignment_active else None
            aligned_trace_indices = gui.aligned_trace_indices if gui.alignment_active else None
            # 传递对齐重叠百分比
            if gui.alignment_active:
                gui.plot_manager.alignment_overlap = gui.alignment_overlap
            # 获取记录信息（用于X轴类型计算）
            records = gui.loaded_data.get("records", [])

            # 应用静校正量到绘图管理器
            # ✅ 根据 imute 参数控制是否应用静校正
            # imute != 0 时启用静校正（兼容原始zplot的静校正参数）
            if gui.static_corrector.has_corrections():
                gui.plot_manager.static_corrections = gui.static_corrector.static_corrections.copy()
                # 如果 imute == 0，使用预览模式（只显示校正后的震相曲线，不应用到波形）
                gui.plot_manager.static_correction_preview_mode = gui.params.imute == 0
                # ✅ 更新平滑度参数
                gui.plot_manager.static_correction_smoothness = gui.static_correction_smoothness_var.get()
                # ✅ 显示平滑参数滑块（预览模式时）
                if gui.plot_manager.static_correction_preview_mode:
                    gui.static_correction_slider_frame.grid()
                else:
                    gui.static_correction_slider_frame.grid_remove()
            else:
                gui.plot_manager.static_corrections = {}
                gui.plot_manager.static_correction_preview_mode = False
                # ✅ 隐藏平滑参数滑块
                gui.static_correction_slider_frame.grid_remove()

            gui._internal_axes_update = True
            try:
                gui.plot_manager.plot_seismic_section(
                    traces=traces,
                    offsets=offsets,
                    times=times,
                    params=gui.params,
                    trace_headers=trace_headers,
                    preserve_zoom=gui.preserve_zoom,
                    user_xlim=gui.user_xlim,
                    user_ylim=gui.user_ylim,
                    alignment_offsets=alignment_offsets,
                    aligned_trace_indices=aligned_trace_indices,
                    records=records,  # ✅ 传递记录信息用于X轴类型计算
                    realtime_interaction=bool(getattr(gui, "_viewport_interacting", False)),
                )
            finally:
                gui._internal_axes_update = False

            realtime_interaction = bool(getattr(gui, "_viewport_interacting", False))

            # 实时交互时优先保证波形主渲染流畅，叠加层延后到交互结束后再绘制
            draw_overlays = not realtime_interaction

            # 绘制拾取点（如果有）
            if draw_overlays and gui.pick_manager:
                picks = gui.pick_manager.get_all_picks()
                if picks:
                    # 绘制拾取点（优先复用主绘图缓存，避免重复过滤计算）
                    alignment_offsets = gui.alignment_offsets if gui.alignment_active else None
                    cached_trace_indices = getattr(gui.plot_manager, "current_filtered_indices", None)
                    cached_x_coordinates = getattr(gui.plot_manager, "current_x_coordinates", None)
                    trace_indices = cached_trace_indices
                    x_coordinates = cached_x_coordinates

                    # 缓存不可用时才回退到一次显式计算
                    if trace_indices is None:
                        filtered_data = gui.plot_manager._filter_data(
                            traces, offsets, times, gui.params, trace_headers
                        )
                        trace_indices = filtered_data.get("indices", None)
                    if x_coordinates is None and trace_indices:
                        records = gui.loaded_data.get("records", [])
                        x_coordinates = gui.plot_manager._calculate_x_coordinates(
                            offsets, trace_indices, trace_headers, records, gui.params
                        )

                    gui.plot_manager.plot_picks(
                        picks, offsets, gui.params, trace_indices, alignment_offsets, x_coordinates
                    )

                    # ✅ 绘制理论走时（如果有）
                    if gui.show_theoretical_times and gui.theoretical_times_data:
                        # 如果启用了水层校正，显示校正后的走时
                        if gui.show_water_layer_correction and gui.water_layer_corrected_times:
                            # 绘制校正后的走时
                            gui.plot_manager.plot_theoretical_traveltime(
                                distances=gui.water_layer_corrected_times["distances"],
                                times=gui.water_layer_corrected_times["times"],
                                params=gui.params,
                                color="blue",
                                linestyle="-.",
                                linewidth=2.0,
                                alpha=0.8,
                                label="水层校正后走时",
                            )
                            # 绘制原始理论走时（作为对比）
                            gui.plot_manager.plot_theoretical_traveltime(
                                distances=gui.theoretical_times_data["distances"],
                                times=gui.theoretical_times_data["times"],
                                params=gui.params,
                                color="green",
                                linestyle="--",
                                linewidth=1.5,
                                alpha=0.6,
                                label="理论走时（原始）",
                            )
                        else:
                            # 只绘制原始理论走时
                            gui.plot_manager.plot_theoretical_traveltime(
                                distances=gui.theoretical_times_data["distances"],
                                times=gui.theoretical_times_data["times"],
                                params=gui.params,
                                color="green",
                                linestyle="--",
                                linewidth=2.0,
                                alpha=0.8,
                                label="理论走时",
                            )

            # 如果用户设置了缩放，恢复坐标轴范围（在绘制完成后）
            if gui.preserve_zoom and gui.user_xlim and gui.user_ylim:
                gui._internal_axes_update = True
                try:
                    gui.ax.set_xlim(gui.user_xlim)
                    # Y轴范围：根据itrev参数决定是否反转
                    if isinstance(gui.user_ylim, tuple) and len(gui.user_ylim) == 2:
                        y_min, y_max = gui.user_ylim
                        if gui.params.itrev == 1:
                            # itrev=1: 时间反转（时间向上，正常显示）
                            gui.ax.set_ylim(y_min, y_max)
                        else:
                            # itrev=0: 默认（时间向下，地震图标准显示）
                            # 确保Y轴反转（ylim[0] > ylim[1]）
                            if y_max < y_min:
                                y_max, y_min = y_min, y_max
                            gui.ax.set_ylim(y_max, y_min)
                    else:
                        gui.ax.set_ylim(gui.user_ylim)
                finally:
                    gui._internal_axes_update = False
            else:
                # 如果没有用户缩放，确保使用params中的xmin/xmax（如果已设置）
                if gui.params.xmin is not None and gui.params.xmax is not None:
                    # 确保xmin/xmax有效且合理
                    if gui.params.xmin < gui.params.xmax:
                        # 使用params中的值，不添加边距（保持精确范围）
                        gui._internal_axes_update = True
                        try:
                            gui.ax.set_xlim(gui.params.xmin, gui.params.xmax)
                        finally:
                            gui._internal_axes_update = False

            # 刷新画布：实时交互阶段避免额外事件刷新，减少主线程阻塞
            gui.canvas.draw_idle()
            if not realtime_interaction:
                gui.root.after_idle(lambda: gui.canvas.flush_events())
                gui.root.update_idletasks()

            # 状态栏显示过滤统计，便于用户判断哪些条件在剔除道
            filter_stats = getattr(gui.plot_manager, "last_filter_stats", {}) or {}
            drop_reasons = filter_stats.get("drop_reasons", {}) if isinstance(filter_stats, dict) else {}
            total_candidates = int(filter_stats.get("total_candidates", 0) or 0) if isinstance(filter_stats, dict) else 0
            passed = len(getattr(gui.plot_manager, "current_filtered_indices", []) or [])

            nskip_drop = int(filter_stats.get("nskip_drop", 0) or 0) if isinstance(filter_stats, dict) else 0
            render_stride = int(filter_stats.get("render_stride", 1) or 1) if isinstance(filter_stats, dict) else 1
            render_used = int(filter_stats.get("render_used", passed) or passed) if isinstance(filter_stats, dict) else passed
            render_visible_candidates = int(
                filter_stats.get("render_visible_candidates", passed) or passed
            ) if isinstance(filter_stats, dict) else passed

            gui.update_status(
                "绘图已更新 | "
                f"通过={passed}/{total_candidates} | "
                f"可见={render_visible_candidates} 渲染={render_used} (stride={render_stride}) | "
                f"drop(itype={int(drop_reasons.get('itype', 0) or 0)}, "
                f"xmin={int(drop_reasons.get('xmin', 0) or 0)}, "
                f"xmax={int(drop_reasons.get('xmax', 0) or 0)}, "
                f"imute={int(drop_reasons.get('imute', 0) or 0)}, "
                f"nskip≈{nskip_drop})"
            )

        except Exception as e:
            error_msg = f"绘图更新失败: {str(e)}"
            gui.update_status(error_msg)
            import traceback

            traceback.print_exc()
            # 检查窗口是否还存在再显示错误
            try:
                if hasattr(gui, "root") and gui.root.winfo_exists():
                    messagebox.showerror("绘图错误", f"{error_msg}\n\n详细信息请查看控制台输出")
            except Exception:
                pass  # 窗口已被销毁，跳过消息框
        finally:
            # 保留 finally 结构，便于后续扩展收尾逻辑
            pass
