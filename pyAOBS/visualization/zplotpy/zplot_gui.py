"""
zplot_gui.py - ZPLOT 主窗口类

基于 Tkinter 和 Matplotlib 实现的交互式地震震相拾取工具
采用顶部工具栏模式，主绘图区域全宽显示

注意（入口约定）:
- 本文件仅作历史实现与兼容保留。
- 当前默认 GUI 入口统一为 qt_fast_viewer.py（通过 gui.py/main.py/run_zplot.py 转发）。
- 新功能与行为修复应优先在 qt_fast_viewer.py 实现。

Author: Based on ZPLOT by Colin A. Zelt (1994), Python implementation 2024
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import matplotlib
_backend_pref = os.environ.get('ZPLOTPY_RENDER_BACKEND', 'tkagg').strip().lower()
_use_tkcairo = (_backend_pref == 'tkcairo')
if _use_tkcairo:
    try:
        matplotlib.use('TkCairo', force=True)
    except Exception:
        _use_tkcairo = False
        matplotlib.use('TkAgg', force=True)
else:
    matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
if _use_tkcairo:
    from matplotlib.backends.backend_tkcairo import FigureCanvasTkCairo as FigureCanvasTkAgg
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
else:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import font_manager
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import platform
import json
import logging
import threading

# 兼容相对导入和绝对导入
try:
    from .top_toolbar import TopToolbar, get_chinese_font
    from .data_loader import DataLoader
    from .plot_manager import PlotManager
    from .pick_manager import PickManager
    from .hdr_to_tx import Z2TxConfig, convert_hdr_to_tx
    from .adaptive_stack import AdaptiveStacker
    from .stacking_evaluator import StackingEvaluator
    from .interpolation_correlation_picker import InterpolationCorrelationPicker
    from .static_correction import StaticCorrector
    from .theoretical_traveltime import TheoreticalTravelTimeCalculator
    from .zplot_gui_services import WorkbenchStateService, PlotUpdateService, PlotRefreshScheduler
    from .zplot_workflow_controller import ZPlotWorkflowController
except ImportError:
    from top_toolbar import TopToolbar, get_chinese_font
    from data_loader import DataLoader
    from plot_manager import PlotManager
    from pick_manager import PickManager
    from hdr_to_tx import Z2TxConfig, convert_hdr_to_tx
    from adaptive_stack import AdaptiveStacker
    from stacking_evaluator import StackingEvaluator
    from interpolation_correlation_picker import InterpolationCorrelationPicker
    from static_correction import StaticCorrector
    from theoretical_traveltime import TheoreticalTravelTimeCalculator
    from zplot_gui_services import WorkbenchStateService, PlotUpdateService, PlotRefreshScheduler
    from zplot_workflow_controller import ZPlotWorkflowController


class ZPlotGUI:
    """ZPLOT 主窗口类 - 交互式地震震相拾取工具"""
    
    def __init__(self, master=None, initial_params=None):
        """初始化主窗口
        
        Args:
            master: Tkinter 根窗口，如果为 None 则创建新窗口
            initial_params: 初始参数字典，例如 {'itype': 1, 'irec': 1}
        """
        # 创建主窗口
        if master is None:
            self.root = tk.Tk()
        else:
            self.root = master
        
        self.root.title('ZPLOT - Seismic Phase Picking Tool')
        self.root.geometry('1400x900')
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self._gui_state_file = (
            Path(os.environ.get("PYAOBS_GUI_STATE_FILE", "").strip())
            if os.environ.get("PYAOBS_GUI_STATE_FILE", "").strip()
            else None
        )
        self._run_inputs_dir = (
            Path(os.environ.get("PYAOBS_RUN_INPUTS_DIR", "").strip())
            if os.environ.get("PYAOBS_RUN_INPUTS_DIR", "").strip()
            else None
        )
        self._state_service = WorkbenchStateService()
        self._plot_update_service = PlotUpdateService()
        self._plot_refresh_scheduler = PlotRefreshScheduler(default_delay_ms=90)
        self._plot_refresh_after_id = None
        self._plot_refresh_in_progress = False
        self._plot_refresh_pending = False
        self._workflow_controller = ZPlotWorkflowController()
        
        # 配置中文字体（必须在创建任何组件之前设置）
        self.setup_chinese_fonts()
        
        # 数据管理
        self.data_loader = DataLoader()
        self.data_loaded = False
        self.current_record = 1
        self.num_traces = 0
        self.num_picks = 0
        
        # 数据文件路径
        self.dfile_path = None
        self.hfile_path = None
        self.rfile_path = None
        
        # 加载的数据
        self.loaded_data = None
        
        # 绘图相关
        self.fig = None
        self.ax = None
        self.canvas = None
        self.plot_manager = None  # 绘图管理器（在创建绘图区域后初始化）
        
        # ✅ 静校正平滑参数更新防抖定时器
        self._smoothness_update_timer = None
        
        # 拾取管理
        self.pick_manager = None  # 拾取管理器（在数据加载后初始化）
        self.pick_mode = False  # 拾取模式开关
        self.pick_event_connector = None  # 拾取事件连接器
        self.mouse_motion_connector = None  # 鼠标移动事件连接器
        
        # 拾取相关状态
        self.dragging_pick = False  # 是否正在拖拽拾取点
        self.drag_pick_info = None  # 拖拽的拾取点信息 (trace_idx, pick_word)
        
        # 批量删除拾取点状态
        self.delete_range_state = 0  # 0=未开始, 1=已记录第一个点, 2=已记录第二个点
        self.delete_range_x1 = None  # 第一个删除范围的 x 坐标
        self.delete_range_x2 = None  # 第二个删除范围的 x 坐标
        self.last_key_class = None  # 上一次按键类型（用于检测连续按键）
        
        # 对齐功能状态
        self.alignment_offsets: Dict[int, float] = {}  # {trace_idx: offset_time} 对齐偏移量
        self.alignment_active = False  # 是否启用了对齐
        self.aligned_trace_indices: List[int] = []  # 对齐时只显示的道索引列表
        self.alignment_overlap = 0.2  # 对齐模式下各道重叠百分比（0.2 = 20%重叠）
        
        # 鼠标位置跟踪
        self.mouse_x = None
        self.mouse_y = None
        self._last_hover_trace_idx = None
        self._last_hover_time_bin = None
        
        # 用户设置的坐标轴范围（用于保存缩放状态）
        self.user_xlim = None  # (x_min, x_max) 或 None
        self.user_ylim = None  # (y_min, y_max) 或 None
        self.preserve_zoom = False  # 是否保留用户设置的缩放
        self.zoom_event_connector_x = None  # X轴缩放事件连接器
        self.zoom_event_connector_y = None  # Y轴缩放事件连接器
        self._updating_params_from_zoom = False  # 防止递归更新的标志
        self.mouse_press_connector = None
        self.mouse_release_connector = None
        self.mouse_scroll_connector = None
        self._is_drag_panning = False
        self._drag_start_data = None  # (x0, y0, xlim0, ylim0)
        
        # 保存初始参数（必须在 create_widgets 之前设置）
        self.initial_params = initial_params
        
        # 保存数据加载时的初始参数快照（用于重绘按钮恢复）
        self.initial_params_snapshot = None
        # 保存初始的范围参数（xmin, xmax, tmin, tmax）
        self.initial_range_params = None
        
        # 创建界面组件
        self.create_menu_bar()
        self.create_widgets()
        self.create_status_bar()
        # 最后一层兜底：对已创建控件做一次字体强绑
        self._force_apply_ui_font()
        
        # 事件绑定
        self.setup_event_handlers()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # 参数对象（在工具栏创建后设置）
        self.params = None
        
        # 自适应叠加器（在参数设置后初始化）
        self.adaptive_stacker = None
        self.stacking_evaluator = StackingEvaluator()  # 自适应叠加评价器
        self.last_stacking_result = None  # 保存最后一次叠加结果用于评价
        
        # 自动拾取器
        self.auto_picker = None
        
        # 插值-相关拾取器
        self.interpolation_correlation_picker = InterpolationCorrelationPicker()
        
        # 静校正器
        self.static_corrector = StaticCorrector()
        self.static_correction_enabled = False  # 静校正是否启用
        
        # ✅ 理论走时计算器
        self.theoretical_traveltime_calculator = None  # TheoreticalTravelTimeCalculator
        self.theoretical_times_data = None  # 理论走时数据
        self.show_theoretical_times = False  # 是否显示理论走时

        # 大文件加载/分级渲染状态
        self._loading_in_background = False
        self._load_thread = None
        self._progressive_render_token = 0
        self._viewport_refill_after_id = None
        self._suspend_viewport_refill = False
        self._pending_viewport_refill = False
        self._viewport_interacting = False
        self._viewport_interaction_after_id = None
        self._internal_axes_update = False
        
        # ✅ 水层校正相关
        self.water_layer_corrections = {}  # {ray_idx: correction_time} 或 {distance: correction_time}
        self.water_layer_corrected_times = None  # 校正后的走时数据
        self.show_water_layer_correction = False  # 是否显示水层校正
        self.root.after(120, self._restore_workbench_state)
    
    def setup_chinese_fonts(self):
        """配置ttk样式以支持中文显示（GUI界面使用中文，但剖面区域使用英文）"""
        import tkinter.font as tkfont
        import os
        import sys
        
        default_ui_size = 16
        chinese_font = get_chinese_font(default_ui_size)
        font_name = chinese_font[0] if isinstance(chinese_font, tuple) else chinese_font
        actual_ui_size = chinese_font[1] if isinstance(chinese_font, tuple) and len(chinese_font) > 1 else default_ui_size
        is_windows = (os.name == 'nt') or sys.platform.startswith('win')
        self._ui_font_name = font_name
        self._ui_font_size = actual_ui_size
        self._ui_font_tuple = (font_name, self._ui_font_size)
        self._ui_font_obj = None
        
        # 不配置 matplotlib 的中文字体，剖面区域使用英文字体
        # 剖面区域的字体配置在 plot_manager.py 中处理
        
        # 方法1: 设置 tkinter 默认字体（影响所有组件）
        # 必须在创建任何组件之前设置
        try:
            default_font = tkfont.nametofont("TkDefaultFont")
            # 先获取当前字体信息
            old_font = default_font.actual()
            
            # 尝试设置字体
            try:
                default_font.config(family=font_name, size=self._ui_font_size)
            except Exception as e1:
                # 如果直接设置失败，尝试创建 Font 对象
                try:
                    new_font_obj = tkfont.Font(family=font_name, size=self._ui_font_size)
                    default_font.config(font=new_font_obj)
                except Exception as e2:
                    # 最后尝试：使用字体名称的一部分
                    try:
                        # 尝试只使用字体名称的主要部分
                        font_parts = font_name.split()
                        if font_parts:
                            simple_name = font_parts[0]  # 例如 "DejaVu" 而不是 "DejaVu Sans"
                            default_font.config(family=simple_name, size=self._ui_font_size)
                    except:
                        pass
            
            # 验证是否设置成功
            new_font = default_font.actual()
            actual_family = new_font.get('family', '').lower()
            if actual_family != font_name.lower() and font_name.lower() not in actual_family:
                # 如果实际字体是 'fixed'，尝试强制设置为 'sans'
                if actual_family == 'fixed':
                    try:
                        default_font.config(family='sans', size=self._ui_font_size)
                    except:
                        pass
        except Exception:
            pass

        # 如果仍然是 fixed，执行一次跨平台强制兜底
        try:
            current_family = tkfont.nametofont("TkDefaultFont").actual().get('family', '').lower()
            if current_family == 'fixed':
                fallback_candidates = [
                    # Windows 常见
                    'Microsoft YaHei UI', 'Microsoft YaHei', '微软雅黑 UI', '微软雅黑',
                    'SimHei', '黑体', 'SimSun', '宋体', 'KaiTi', '楷体',
                    # Linux 常见
                    'Noto Sans CJK SC', 'Noto Sans CJK', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
                    'Source Han Sans CN', 'Source Han Sans SC', 'Droid Sans Fallback',
                    # macOS 常见
                    'PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'STSong',
                    # 通用兜底
                    'Arial Unicode MS', 'DejaVu Sans'
                ]
                available = set(tkfont.families())
                chosen = None
                for cand in fallback_candidates:
                    if cand in available:
                        chosen = cand
                        break
                if chosen is None:
                    # 再做一次关键字匹配，尽量找到 CJK 字体
                    preferred_keywords = [
                        'cjk', 'noto', 'wenquanyi', 'source han', 'yahei', '微软雅黑',
                        'simhei', '黑体', 'simsun', '宋体', 'pingfang', 'hiragino',
                        'stheiti', 'stsong'
                    ]
                    for fam in available:
                        fl = fam.lower()
                        if any(k in fl for k in preferred_keywords):
                            chosen = fam
                            break
                if chosen is None:
                    # 最后兜底，至少不要继续停留 fixed
                    chosen = 'DejaVu Sans'

                for name in ("TkDefaultFont", "TkTextFont", "TkFixedFont", "TkMenuFont", "TkHeadingFont"):
                    try:
                            tkfont.nametofont(name).config(family=chosen, size=self._ui_font_size)
                    except Exception:
                        pass
                try:
                    self.root.option_add('*Font', (chosen, self._ui_font_size))
                    self.root.option_add('*Menu.font', (chosen, self._ui_font_size))
                    self.root.option_add('*Listbox.font', (chosen, self._ui_font_size))
                    self.root.option_add('*TCombobox*Listbox.font', (chosen, self._ui_font_size))
                except Exception:
                    pass
                font_name = chosen
                self._ui_font_name = chosen
                self._ui_font_tuple = (chosen, self._ui_font_size)
        except Exception:
            pass

        # 创建统一字体对象，后续用于控件级强绑
        try:
            self._ui_font_obj = tkfont.Font(family=self._ui_font_name, size=self._ui_font_size)
        except Exception:
            self._ui_font_obj = None
        
        try:
            text_font = tkfont.nametofont("TkTextFont")
            text_font.config(family=font_name, size=self._ui_font_size)
        except Exception:
            pass
        
        try:
            fixed_font = tkfont.nametofont("TkFixedFont")
            fixed_font.config(family=font_name, size=self._ui_font_size)
        except Exception:
            pass
        
        # 方法2: 使用 option_add 设置全局默认字体选项
        try:
            font_tuple = (font_name, self._ui_font_size)
            self.root.option_add('*Font', font_tuple)
            self.root.option_add('*TkDefaultFont', font_tuple)
            self.root.option_add('*TkTextFont', font_tuple)
            self.root.option_add('*TkFixedFont', font_tuple)
            # 关键：这些控件常常不会继承到中文字体，容易出现“方框字”
            self.root.option_add('*Menu.font', font_tuple)
            self.root.option_add('*Listbox.font', font_tuple)
            self.root.option_add('*TCombobox*Listbox.font', font_tuple)
            self.root.option_add('*Text.font', font_tuple)
            self.root.option_add('*Label.font', font_tuple)
        except Exception:
            pass
        
        # 方法3: 为 ttk 组件设置样式
        style = ttk.Style()
        try:
            # 使用更稳妥的主题，减少部分平台对字体样式忽略的问题
            try:
                style.theme_use('clam')
            except Exception:
                pass
            style.configure('TLabel', font=chinese_font)
            style.configure('TButton', font=chinese_font)
            style.configure('TEntry', font=chinese_font)
            style.configure('TCombobox', font=chinese_font)
            style.configure('TSpinbox', font=chinese_font)
            style.configure('TCheckbutton', font=chinese_font)
            style.configure('TRadiobutton', font=chinese_font)
            style.configure('TMenubutton', font=chinese_font)
            style.configure('TNotebook.Tab', font=chinese_font)
            style.configure('TLabelFrame', font=chinese_font)
            style.configure('TLabelframe.Label', font=chinese_font)
            
            # 尝试设置所有默认样式
            style.configure('.', font=chinese_font)
        except Exception:
            pass

        # 同步调节控件高度/内边距，避免字号放大后文本被裁切
        self._apply_ui_density_metrics()
        
        # 验证字体设置并尝试备用方案（静默处理）
        try:
            current_font = tkfont.nametofont("TkDefaultFont")
            font_info = current_font.actual()
            actual_family = font_info.get('family', 'unknown')
            actual_lower = actual_family.lower()
            
            # 如果字体是 'fixed'，尝试设置为备用字体
            if actual_lower == 'fixed':
                try:
                    # 尝试使用 'sans' 或 'serif'
                    for fallback in ['sans', 'serif', 'DejaVu Sans', 'Liberation Sans']:
                        try:
                            current_font.config(family=fallback, size=self._ui_font_size)
                            new_info = current_font.actual()
                            if new_info.get('family', '').lower() != 'fixed':
                                break
                        except:
                            continue
                except Exception:
                    pass
        except Exception:
            pass

        # 打印一次 CJK 候选字体，帮助定位环境问题
        try:
            fams = list(tkfont.families())
            keys = ['cjk', 'noto', 'wenquanyi', 'source han', 'yahei', 'simhei', 'simsun', 'pingfang', 'heiti', 'song']
            candidates = [f for f in fams if any(k in f.lower() for k in keys)]
            self._cjk_font_candidates = candidates
            # 如果仍是 fixed 或选到了明显非 CJK 字体，优先切换到候选里的宋体类字体
            try:
                tk_default_now = tkfont.nametofont("TkDefaultFont").actual().get('family', '').lower()
                current_selected = str(getattr(self, '_ui_font_name', font_name)).lower()
                looks_non_cjk = ('dejavu' in current_selected) or (current_selected in ('fixed', 'sans'))
                if candidates and (tk_default_now == 'fixed' or looks_non_cjk):
                    preferred = None
                    for cand in candidates:
                        cl = cand.lower()
                        if 'song' in cl or '宋' in cand:
                            preferred = cand
                            break
                    if preferred is None:
                        preferred = candidates[0]
                    for name in ("TkDefaultFont", "TkTextFont", "TkFixedFont", "TkMenuFont", "TkHeadingFont"):
                        try:
                            tkfont.nametofont(name).config(family=preferred, size=self._ui_font_size)
                        except Exception:
                            pass
                    try:
                        self.root.option_add('*Font', (preferred, self._ui_font_size))
                        self.root.option_add('*Menu.font', (preferred, self._ui_font_size))
                        self.root.option_add('*Listbox.font', (preferred, self._ui_font_size))
                        self.root.option_add('*TCombobox*Listbox.font', (preferred, self._ui_font_size))
                    except Exception:
                        pass
                    self._ui_font_name = preferred
                    self._ui_font_tuple = (preferred, self._ui_font_size)
                    try:
                        self._ui_font_obj = tkfont.Font(family=preferred, size=self._ui_font_size)
                    except Exception:
                        pass
            except Exception:
                pass
            preview = candidates[:20]
            print(f"[ZFONT_CANDIDATES] count={len(candidates)} sample={preview}")
        except Exception:
            pass

        # 打印一次字体诊断信息，便于定位“方框字”问题
        try:
            tk_default = tkfont.nametofont("TkDefaultFont").actual().get('family', 'unknown')
            tk_text = tkfont.nametofont("TkTextFont").actual().get('family', 'unknown')
            tk_menu = tkfont.nametofont("TkMenuFont").actual().get('family', 'unknown')
            print(f"[ZFONT] selected='{getattr(self, '_ui_font_name', font_name)}' tk_default='{tk_default}' tk_text='{tk_text}' tk_menu='{tk_menu}' win={is_windows}")
        except Exception:
            pass

    def _apply_font_recursively(self, widget):
        """递归给控件强制设置字体（最终兜底）。"""
        if widget is None:
            return
        # 优先使用 Font 对象，不可用则退回 tuple
        font_value = self._ui_font_obj if self._ui_font_obj is not None else self._ui_font_tuple

        try:
            wclass = widget.winfo_class()
        except Exception:
            wclass = ''

        # 常见控件显式设字体
        if wclass in (
            'Label', 'Button', 'Entry', 'Text', 'Listbox', 'Menu', 'Menubutton',
            'Checkbutton', 'Radiobutton', 'Message', 'Spinbox', 'TCombobox',
            'TLabel', 'TButton', 'TEntry', 'TCheckbutton', 'TRadiobutton',
            'TMenubutton', 'TSpinbox'
        ):
            try:
                widget.configure(font=font_value)
            except Exception:
                pass

        # Combobox 下拉列表（Listbox）单独强绑
        if wclass == 'TCombobox':
            try:
                popdown = widget.tk.call("ttk::combobox::PopdownWindow", str(widget))
                listbox_path = f"{popdown}.f.l"
                widget.tk.call(listbox_path, "configure", "-font", f"{self._ui_font_name} {self._ui_font_size}")
            except Exception:
                pass

        try:
            for child in widget.winfo_children():
                self._apply_font_recursively(child)
        except Exception:
            pass

    def _force_apply_ui_font(self):
        """在界面创建后执行一次控件级字体强绑。"""
        try:
            if not hasattr(self, 'root') or self.root is None:
                return
            # 若启动期仍是 fixed，尝试再切到已探测到的 CJK 候选
            try:
                tk_default_now = tkfont.nametofont("TkDefaultFont").actual().get('family', '').lower()
                candidates = getattr(self, '_cjk_font_candidates', []) or []
                if tk_default_now == 'fixed' and candidates:
                    preferred = None
                    for cand in candidates:
                        cl = cand.lower()
                        if 'song' in cl or '宋' in cand:
                            preferred = cand
                            break
                    if preferred is None:
                        preferred = candidates[0]
                    for name in ("TkDefaultFont", "TkTextFont", "TkFixedFont", "TkMenuFont", "TkHeadingFont"):
                        try:
                            tkfont.nametofont(name).config(family=preferred, size=self._ui_font_size)
                        except Exception:
                            pass
                    self._ui_font_name = preferred
                    self._ui_font_tuple = (preferred, self._ui_font_size)
                    try:
                        self._ui_font_obj = tkfont.Font(family=preferred, size=self._ui_font_size)
                    except Exception:
                        pass
            except Exception:
                pass
            self._apply_font_recursively(self.root)
            # 菜单栏本身也尝试强绑一次
            try:
                menu_name = self.root.cget('menu')
                if menu_name:
                    menu_widget = self.root.nametowidget(menu_name)
                    menu_widget.configure(font=self._ui_font_obj if self._ui_font_obj else self._ui_font_tuple)
                    self._apply_font_recursively(menu_widget)
            except Exception:
                pass
        except Exception:
            pass

    def _apply_ui_density_metrics(self):
        """根据当前字号同步调整控件高度/内边距，避免大字号被截断。"""
        try:
            size = int(getattr(self, '_ui_font_size', 16))
        except Exception:
            size = 16

        # 基准字号 16；随字号放大时同步提高内边距
        delta = max(0, size - 16)
        pad_y = 2 + int(delta * 0.45)
        pad_x = 4 + int(delta * 0.35)
        entry_pad_y = 1 + int(delta * 0.35)
        tab_pad_y = 3 + int(delta * 0.55)
        tab_pad_x = 8 + int(delta * 0.45)
        frame_label_pad_y = 1 + int(delta * 0.35)
        frame_label_pad_x = 4 + int(delta * 0.30)

        try:
            style = ttk.Style()
            style.configure('TButton', padding=(pad_x, pad_y))
            style.configure('TCheckbutton', padding=(pad_x, pad_y))
            style.configure('TRadiobutton', padding=(pad_x, pad_y))
            style.configure('TMenubutton', padding=(pad_x, pad_y))
            style.configure('TEntry', padding=(pad_x, entry_pad_y))
            style.configure('TSpinbox', padding=(pad_x, entry_pad_y))
            style.configure('TCombobox', padding=(pad_x, entry_pad_y))
            style.configure('TNotebook.Tab', padding=(tab_pad_x, tab_pad_y))
            style.configure('TLabelframe.Label', padding=(frame_label_pad_x, frame_label_pad_y))

            # Treeview 常见行高截断问题（即使当前界面未显式使用，也做安全兜底）
            style.configure('Treeview', rowheight=max(20, int(size * 1.9)))
        except Exception:
            pass

        # 适度调整 Tk 全局缩放：仅在大字号时稍微拉开控件几何
        try:
            if size >= 17:
                tk_scale = 1.0 + min(0.22, (size - 16) * 0.04)
            else:
                tk_scale = 1.0
            self.root.tk.call('tk', 'scaling', tk_scale)
        except Exception:
            pass

    def set_ui_font_size(self, new_size: int):
        """运行时调整 UI 字号（菜单、工具栏、控件统一生效）。"""
        import tkinter.font as tkfont

        try:
            target_size = int(new_size)
        except Exception:
            return
        target_size = max(8, min(28, target_size))

        if not getattr(self, '_ui_font_name', None):
            chosen = get_chinese_font(target_size)
            self._ui_font_name = chosen[0] if isinstance(chosen, tuple) else str(chosen)

        if target_size == getattr(self, '_ui_font_size', target_size):
            return

        self._ui_font_size = target_size
        self._ui_font_tuple = (self._ui_font_name, self._ui_font_size)
        try:
            self._ui_font_obj = tkfont.Font(family=self._ui_font_name, size=self._ui_font_size)
        except Exception:
            self._ui_font_obj = None

        for name in ("TkDefaultFont", "TkTextFont", "TkFixedFont", "TkMenuFont", "TkHeadingFont"):
            try:
                tkfont.nametofont(name).config(family=self._ui_font_name, size=self._ui_font_size)
            except Exception:
                pass

        try:
            self.root.option_add('*Font', self._ui_font_tuple)
            self.root.option_add('*TkDefaultFont', self._ui_font_tuple)
            self.root.option_add('*TkTextFont', self._ui_font_tuple)
            self.root.option_add('*TkFixedFont', self._ui_font_tuple)
            self.root.option_add('*Menu.font', self._ui_font_tuple)
            self.root.option_add('*Listbox.font', self._ui_font_tuple)
            self.root.option_add('*TCombobox*Listbox.font', self._ui_font_tuple)
            self.root.option_add('*Text.font', self._ui_font_tuple)
            self.root.option_add('*Label.font', self._ui_font_tuple)
        except Exception:
            pass

        try:
            style = ttk.Style()
            style.configure('TLabel', font=self._ui_font_tuple)
            style.configure('TButton', font=self._ui_font_tuple)
            style.configure('TEntry', font=self._ui_font_tuple)
            style.configure('TCombobox', font=self._ui_font_tuple)
            style.configure('TSpinbox', font=self._ui_font_tuple)
            style.configure('TCheckbutton', font=self._ui_font_tuple)
            style.configure('TRadiobutton', font=self._ui_font_tuple)
            style.configure('TMenubutton', font=self._ui_font_tuple)
            style.configure('TNotebook.Tab', font=self._ui_font_tuple)
            style.configure('TLabelFrame', font=self._ui_font_tuple)
            style.configure('TLabelframe.Label', font=self._ui_font_tuple)
            style.configure('.', font=self._ui_font_tuple)
        except Exception:
            pass

        try:
            if hasattr(self, 'top_toolbar') and self.top_toolbar:
                self.top_toolbar.refresh_ui_fonts()
        except Exception:
            pass

        self._apply_ui_density_metrics()
        self._force_apply_ui_font()
        self.update_status(f'界面字体已调整为 {self._ui_font_size} 号')

    def increase_ui_font(self, step: int = 1):
        """增大 UI 字号。"""
        self.set_ui_font_size(getattr(self, '_ui_font_size', 16) + step)

    def decrease_ui_font(self, step: int = 1):
        """减小 UI 字号。"""
        self.set_ui_font_size(getattr(self, '_ui_font_size', 16) - step)

    def reset_ui_font(self):
        """恢复 UI 默认字号（可叠加环境变量缩放基准）。"""
        base_font = get_chinese_font(16)
        base_size = base_font[1] if isinstance(base_font, tuple) and len(base_font) > 1 else 16
        self.set_ui_font_size(base_size)

    def _on_increase_font_shortcut(self, event=None):
        """快捷键：放大字体。"""
        self.increase_ui_font()
        return 'break'

    def _on_decrease_font_shortcut(self, event=None):
        """快捷键：缩小字体。"""
        self.decrease_ui_font()
        return 'break'

    def _on_reset_font_shortcut(self, event=None):
        """快捷键：重置字体。"""
        self.reset_ui_font()
        return 'break'
    
    def _setup_matplotlib_chinese_font(self, font_name: str):
        """配置 matplotlib 的字体（已废弃，剖面区域使用英文）
        
        注意：此方法已不再使用，剖面区域的字体配置在 plot_manager.py 中处理
        确保剖面区域使用英文字体
        
        Args:
            font_name: 字体名称（未使用）
        """
        # 不设置中文字体，确保剖面区域使用默认英文字体
        # 剖面区域的字体配置在 plot_manager.py 中处理
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        # 使用默认的英文字体，不设置中文字体
        pass
        
    def create_menu_bar(self):
        """创建菜单栏"""
        menu_font = getattr(self, '_ui_font_tuple', get_chinese_font(10))
        menubar = tk.Menu(self.root, font=menu_font)
        self.root.config(menu=menubar)
        
        # File 菜单
        file_menu = tk.Menu(menubar, tearoff=0, font=menu_font)
        menubar.add_cascade(label='File', menu=file_menu)
        file_menu.add_command(label='Open Data File...', command=self.open_data_file, accelerator='Ctrl+O')
        file_menu.add_command(label='Open Header File...', command=self.open_header_file)
        file_menu.add_command(label='Open Record File...', command=self.open_record_file)
        file_menu.add_separator()
        file_menu.add_command(label='Reload Data', command=self.reload_data, accelerator='Ctrl+R')
        file_menu.add_command(label='Data Info...', command=self.show_data_info)
        file_menu.add_separator()
        file_menu.add_command(label='Save Parameters...', command=self.save_parameters)
        file_menu.add_command(label='Load Parameters...', command=self.load_parameters)
        file_menu.add_separator()
        file_menu.add_command(label='Save Picks (zplot.out)...', command=self.save_picks, accelerator='Ctrl+S')
        file_menu.add_command(label='Save Picks to Header (.hdr)...', command=self.save_picks_to_header)
        file_menu.add_command(label='Convert Header to tx.in...', command=self.convert_hdr_to_tx)
        file_menu.add_separator()
        file_menu.add_command(label='Export Figure...', command=self.export_figure)
        file_menu.add_separator()
        file_menu.add_command(label='Exit', command=self.quit, accelerator='Ctrl+Q')
        
        # Edit 菜单
        edit_menu = tk.Menu(menubar, tearoff=0, font=menu_font)
        menubar.add_cascade(label='Edit', menu=edit_menu)
        edit_menu.add_command(label='Edit Parameters...', command=self.edit_parameters, accelerator='E')
        edit_menu.add_command(label='Clear Picks', command=self.clear_picks)
        
        # View 菜单
        view_menu = tk.Menu(menubar, tearoff=0, font=menu_font)
        menubar.add_cascade(label='View', menu=view_menu)
        view_menu.add_command(label='Zoom In', command=self.zoom_in)
        view_menu.add_command(label='Zoom Out', command=self.zoom_out)
        view_menu.add_command(label='Reset View', command=self.reset_view)
        view_menu.add_separator()
        view_menu.add_command(label='Increase UI Font', command=self.increase_ui_font, accelerator='Ctrl++')
        view_menu.add_command(label='Decrease UI Font', command=self.decrease_ui_font, accelerator='Ctrl+-')
        view_menu.add_command(label='Reset UI Font', command=self.reset_ui_font, accelerator='Ctrl+0')
        view_menu.add_separator()
        view_menu.add_command(label='Previous Record', command=self.prev_record, accelerator='Left')
        view_menu.add_command(label='Next Record', command=self.next_record, accelerator='Right')
        view_menu.add_separator()
        view_menu.add_command(label='Toggle Toolbar', command=self.toggle_toolbar)
        
        # Tools 菜单
        tools_menu = tk.Menu(menubar, tearoff=0, font=menu_font)
        menubar.add_cascade(label='Tools', menu=tools_menu)
        tools_menu.add_command(label='Pick Mode', command=self.enter_pick_mode, accelerator='P')
        tools_menu.add_command(label='Filter Data', command=self.apply_filter)
        tools_menu.add_command(label='Apply Gain', command=self.apply_gain)
        tools_menu.add_separator()
        tools_menu.add_command(label='Calculate Static Correction...', command=self.calculate_static_correction_dialog)
        tools_menu.add_command(label='Clear Static Correction', command=self.clear_static_correction, accelerator='Ctrl+Shift+C')
        tools_menu.add_separator()
        tools_menu.add_command(label='Calculate Theoretical Travel Time...', command=self.calculate_theoretical_traveltime_dialog)
        tools_menu.add_command(label='Clear Theoretical Travel Time', command=self.clear_theoretical_traveltime)
        tools_menu.add_separator()
        tools_menu.add_command(label='Calculate Water Layer Correction...', command=self.calculate_water_layer_correction_dialog)
        tools_menu.add_command(label='Clear Water Layer Correction', command=self.clear_water_layer_correction)
        
        # Help 菜单
        help_menu = tk.Menu(menubar, tearoff=0, font=menu_font)
        menubar.add_cascade(label='Help', menu=help_menu)
        help_menu.add_command(label='Help', command=self.show_help, accelerator='H')
        help_menu.add_command(label='Keyboard Shortcuts', command=self.show_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label='About', command=self.show_about)
        
    def create_widgets(self):
        """创建界面组件 - 采用顶部工具栏模式"""
        # 主容器
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        main_container.rowconfigure(1, weight=1)  # 绘图区域可扩展
        main_container.columnconfigure(0, weight=1)
        self.main_container = main_container
        
        # 顶部工具栏（可折叠）
        self.top_toolbar = TopToolbar(main_container, self)
        self.top_toolbar.container.grid(row=0, column=0, sticky='ew', pady=(0, 2))
        
        # 获取参数对象引用（确保已初始化）
        # TopToolbar 在 __init__ 中会创建 params，所以这里应该能获取到
        try:
            if hasattr(self.top_toolbar, 'params'):
                self.params = self.top_toolbar.params
            else:
                raise AttributeError("top_toolbar 没有 params 属性")
        except Exception as e:
            # 如果工具栏参数未初始化，创建一个默认参数对象
            from parameters import ZPlotParameters
            self.params = ZPlotParameters()
        
        # 确保参数对象已设置（使用 is None 检查）
        if self.params is None:
            from parameters import ZPlotParameters
            self.params = ZPlotParameters()
        
        # 应用初始参数（如果提供了）
        if self.initial_params and self.params:
            for param_name, param_value in self.initial_params.items():
                if hasattr(self.params, param_name):
                    setattr(self.params, param_name, param_value)
                    # 更新状态栏显示
                    if param_name == 'itype':
                        itype_names = {
                            0: '全部', 1: '垂直', 2: '径向', 3: '横向', 4: '水听器',
                            -1: '垂直+径向', -2: '径向+横向', -3: '径向+水听器', -4: '垂直+水听器'
                        }
                        name = itype_names.get(param_value, f'itype={param_value}')
                        self.update_status(f'已设置数据类型过滤: {name}')
            
            # 更新工具栏控件显示（如果工具栏已创建）
            if hasattr(self, 'top_toolbar') and self.top_toolbar:
                self.top_toolbar.update_widgets_from_params()
        
        # 初始化自适应叠加器（使用参数中的设置）
        if self.params:
            self.adaptive_stacker = AdaptiveStacker(
                nsi=self.params.nsi,
                pjgl=self.params.pjgl,
                stkwb=self.params.stkwb,
                stkwl=self.params.stkwl,
                dtcw=self.params.dtcw,
                ratio=self.params.hilbratio
            )
        
        # 主绘图区域（全宽）
        self.create_plot_area(main_container)
        
    def create_plot_area(self, parent):
        """创建主绘图区域（全宽）"""
        # 绘图框架
        plot_frame = ttk.Frame(parent)
        plot_frame.grid(row=1, column=0, sticky='nsew', pady=(0, 2))
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        
        # Matplotlib 图形（全宽）
        self.fig = Figure(figsize=(14, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        
        # 初始化空图
        self.ax.set_xlabel('Offset (km)', fontsize=10)
        self.ax.set_ylabel('Time (s)', fontsize=10)
        self.ax.set_title('ZPLOT - Seismic Phase Picking', fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        # Canvas（全宽填充）
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        
        # 连接鼠标移动事件（用于获取鼠标位置）
        if self.mouse_motion_connector is None:
            self.mouse_motion_connector = self.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)
        if self.mouse_press_connector is None:
            self.mouse_press_connector = self.canvas.mpl_connect('button_press_event', self.on_plot_mouse_press)
        if self.mouse_release_connector is None:
            self.mouse_release_connector = self.canvas.mpl_connect('button_release_event', self.on_plot_mouse_release)
        if self.mouse_scroll_connector is None:
            self.mouse_scroll_connector = self.canvas.mpl_connect('scroll_event', self.on_plot_scroll)
        
        # 监听坐标轴范围变化事件（用于matplotlib工具栏缩放时更新参数）
        if self.ax and self.zoom_event_connector_x is None:
            self.zoom_event_connector_x = self.ax.callbacks.connect('xlim_changed', self.on_axes_lims_changed)
            self.zoom_event_connector_y = self.ax.callbacks.connect('ylim_changed', self.on_axes_lims_changed)
        
        # Matplotlib 工具栏（放在底部）
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky='ew')
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        # 静校正平滑参数滑块（放在工具栏下方）
        self.static_correction_slider_frame = ttk.Frame(plot_frame)
        self.static_correction_slider_frame.grid(row=2, column=0, sticky='ew', padx=10, pady=5)
        self.static_correction_slider_frame.grid_remove()  # 默认隐藏
        
        # 滑块标签和值显示
        slider_label_frame = ttk.Frame(self.static_correction_slider_frame)
        slider_label_frame.pack(fill=tk.X, pady=(0, 3))
        ttk.Label(slider_label_frame, text='静校正平滑参数:', font=('Arial', 13)).pack(side=tk.LEFT, padx=5)
        self.static_correction_smoothness_label = ttk.Label(
            slider_label_frame, text='0.100', font=('Arial', 9, 'bold'), 
            foreground='blue', width=8)
        self.static_correction_smoothness_label.pack(side=tk.LEFT, padx=5)
        
        # 滑块
        self.static_correction_smoothness_var = tk.DoubleVar(value=0.1)
        self.static_correction_smoothness_scale = ttk.Scale(
            self.static_correction_slider_frame, from_=0.0, to=1.0,
            variable=self.static_correction_smoothness_var, orient=tk.HORIZONTAL,
            command=lambda v: self.on_static_correction_smoothness_changed(float(v)))
        self.static_correction_smoothness_scale.pack(fill=tk.X, padx=10, pady=2)
        
        # 滑块刻度标签
        slider_scale_labels_frame = ttk.Frame(self.static_correction_slider_frame)
        slider_scale_labels_frame.pack(fill=tk.X, padx=10)
        ttk.Label(slider_scale_labels_frame, text='0.0 (通过所有点)', font=('Arial', 7)).pack(side=tk.LEFT)
        ttk.Label(slider_scale_labels_frame, text='0.5', font=('Arial', 7)).pack(side=tk.LEFT, expand=True)
        ttk.Label(slider_scale_labels_frame, text='1.0 (最平滑)', font=('Arial', 7)).pack(side=tk.RIGHT)
        
        # 初始化绘图管理器
        self.plot_manager = PlotManager(self.ax)
        
    def create_status_bar(self):
        """创建状态栏"""
        parent = getattr(self, 'main_container', self.root)
        status_frame = ttk.Frame(parent)
        if parent is self.root:
            status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        else:
            # 固定在主容器底部，避免顶部参数面板展开时把状态栏“挤没”
            status_frame.grid(row=2, column=0, sticky='ew')
        
        # 状态信息
        self.status_label = ttk.Label(status_frame, 
                                     text='就绪 | 记录: - | 道数: - | 拾取: -',
                                     relief=tk.SUNKEN, anchor=tk.W, padding=5)
        self.status_label.pack(fill=tk.X, padx=2, pady=2)
        
    def setup_event_handlers(self):
        """设置事件处理"""
        # 键盘快捷键
        self.root.bind('<Control-o>', lambda e: self.open_data_file())
        self.root.bind('<Control-r>', lambda e: self.reload_data())
        self.root.bind('<Control-s>', lambda e: self.save_picks())  # Ctrl+S: 保存为zplot.out格式
        self.root.bind('<Control-equal>', self._on_increase_font_shortcut)  # Ctrl+=
        self.root.bind('<Control-plus>', self._on_increase_font_shortcut)  # Ctrl++
        self.root.bind('<Control-KP_Add>', self._on_increase_font_shortcut)  # 小键盘+
        self.root.bind('<Control-minus>', self._on_decrease_font_shortcut)  # Ctrl+-
        self.root.bind('<Control-KP_Subtract>', self._on_decrease_font_shortcut)  # 小键盘-
        self.root.bind('<Control-0>', self._on_reset_font_shortcut)  # Ctrl+0
        self.root.bind('<KeyPress-s>', lambda e: self.save_picks_to_header())  # s键: 保存到.hdr文件
        self.root.bind('<Control-q>', lambda e: self.quit())
        self.root.bind('<KeyPress-q>', lambda e: self.quit())  # q键：退出程序
        self.root.bind('<KeyPress-p>', lambda e: self.enter_pick_mode())
        self.root.bind('<KeyPress-e>', lambda e: self.edit_parameters())
        self.root.bind('<Control-Shift-C>', lambda e: self.clear_static_correction())  # Ctrl+Shift+C: 清除静校正
        self.root.bind('<KeyPress-h>', lambda e: self.show_help())
        
        # X轴类型切换（1-5键，仅在非拾取模式或拾取模式下但不在拾取操作时）
        # 注意：1-3键在拾取模式下用于拾取操作，所以需要检查拾取模式
        self.root.bind('<KeyPress-4>', lambda e: self.change_xaxis_type(-4))  # 修正方位角
        self.root.bind('<KeyPress-5>', lambda e: self.change_xaxis_type(-5))  # 道号
        
        # 拾取模式快捷键
        # 键盘数字键对应鼠标按键（用于拾取）
        # 注意：1-3键在拾取模式下用于拾取，但在非拾取模式下可以用于改变X轴类型
        self.root.bind('<KeyPress-1>', lambda e: self.on_key_1())
        self.root.bind('<KeyPress-2>', lambda e: self.on_key_2())
        self.root.bind('<KeyPress-3>', lambda e: self.on_key_3())
        
        # 拾取字切换快捷键
        self.root.bind('<KeyPress-comma>', lambda e: self.switch_pick_word(-1))  # 上一个拾取字
        self.root.bind('<KeyPress-period>', lambda e: self.switch_pick_word(1))  # 下一个拾取字
        self.root.bind('<KeyPress-bracketleft>', lambda e: self.switch_pick_word(-1))  # [
        self.root.bind('<KeyPress-bracketright>', lambda e: self.switch_pick_word(1))  # ]
        
        # 删除快捷键
        self.root.bind('<KeyPress-Delete>', lambda e: self.delete_current_pick())
        self.root.bind('<KeyPress-BackSpace>', lambda e: self.delete_current_pick())
        
        # 其他快捷键
        self.root.bind('<KeyPress-d>', lambda e: self.on_delete_range_key())
        self.root.bind('<KeyPress-z>', lambda e: self.on_zoom_in_key())
        self.root.bind('<KeyPress-o>', lambda e: self.on_zoom_out_key())
        self.root.bind('<KeyPress-a>', lambda e: self.align_picks())  # a键：对齐拾取
        self.root.bind('<KeyPress-f>', lambda e: self.perform_adaptive_stacking())  # f键：自适应叠加
        self.root.bind('<KeyPress-F>', lambda e: self.show_stacking_evaluation())  # Shift+F键：显示评价可视化
        self.root.bind('<KeyPress-c>', lambda e: self.on_interpolation_correlation_picking())  # c键：插值-相关拾取
        
        # 显示窗口移动快捷键
        self.root.bind('<KeyPress-u>', lambda e: self.move_window_up())  # u键：向上移动显示窗口
        self.root.bind('<KeyPress-v>', lambda e: self.move_window_down())  # v键：向下移动显示窗口
        self.root.bind('<KeyPress-r>', lambda e: self.move_window_right())  # r键：向右移动显示窗口
        self.root.bind('<KeyPress-l>', lambda e: self.move_window_left())  # l键：向左移动显示窗口
        
        # X轴符号切换
        self.root.bind('<KeyPress-minus>', lambda e: self.toggle_xaxis_sign())  # -键：改变X轴符号
        
        # X键：移除/重新绘制最近的trace
        self.root.bind('<KeyPress-x>', lambda e: self.toggle_trace_visibility())  # x键：切换trace显示状态
        self.root.bind('<KeyPress-X>', lambda e: self.toggle_trace_visibility())  # Shift+X键：同样功能
        
        # i键：显示鼠标所在道的关键信息
        self.root.bind('<KeyPress-i>', lambda e: self.show_trace_info())  # i键：显示道信息
        self.root.bind('<KeyPress-I>', lambda e: self.show_trace_info())  # Shift+I键：同样功能
        
        # 显示范围扩展（<和>键在键盘上是Shift+,和Shift+.）
        # 尝试多种绑定方式以确保兼容性
        self.root.bind('<Shift-comma>', lambda e: self.expand_left())  # <键：向左扩展显示范围
        self.root.bind('<Shift-period>', lambda e: self.expand_right())  # >键：向右扩展显示范围
        self.root.bind('<KeyPress-comma>', lambda e: self.expand_left() if e.state & 0x1 else None)  # Shift+, (状态检查)
        self.root.bind('<KeyPress-period>', lambda e: self.expand_right() if e.state & 0x1 else None)  # Shift+. (状态检查)
        # 也尝试直接绑定字符（在某些键盘布局下）
        try:
            self.root.bind('<', lambda e: self.expand_left())
            self.root.bind('>', lambda e: self.expand_right())
        except:
            pass
        
        # 记录导航快捷键
        self.root.bind('<Left>', lambda e: self.prev_record())
        self.root.bind('<Right>', lambda e: self.next_record())
        
        # 鼠标事件（用于拾取）- 延迟连接，因为 canvas 可能还未创建
        self.pick_mode = False
        self.pick_event_connector = None
        
    def update_status(self, message: str = None):
        """更新状态栏"""
        # 检查窗口是否还存在
        try:
            if not hasattr(self, 'status_label') or not self.status_label.winfo_exists():
                return
        except:
            return
        
        if message:
            try:
                self.status_label.config(text=message)
            except:
                pass  # 窗口可能已被销毁
        else:
            if self.data_loaded and self.loaded_data:
                header = self.loaded_data['header']
                status_text = (f'就绪 | 记录: {self.current_record}/{header.nrec} | '
                             f'道数: {self.num_traces} | 拾取: {self.num_picks}')
            else:
                status_text = f'就绪 | 记录: - | 道数: - | 拾取: -'
            self.status_label.config(text=status_text)
    
    def get_current_traces(self):
        """获取当前记录的所有道索引"""
        if not self.data_loaded or not self.loaded_data:
            return []
        
        return self.data_loader.get_traces_for_record(self.current_record - 1)
    
    def get_trace_data(self, trace_idx: int):
        """获取指定道的数据"""
        if not self.data_loaded or not self.loaded_data:
            return None
        return self.data_loader.get_trace(trace_idx)
    
    def get_trace_header(self, trace_idx: int):
        """获取指定道的头信息"""
        if not self.data_loaded or not self.loaded_data:
            return None
        return self.data_loader.get_trace_header(trace_idx)

    def _apply_static_correction_state_to_plot_manager(self):
        """将静校正状态同步到绘图管理器"""
        if self.static_corrector.has_corrections():
            self.plot_manager.static_corrections = self.static_corrector.static_corrections.copy()
            # imute == 0 时仅预览，不把静校正应用到波形
            self.plot_manager.static_correction_preview_mode = (self.params.imute == 0)
            self.plot_manager.static_correction_smoothness = self.static_correction_smoothness_var.get()
            if self.plot_manager.static_correction_preview_mode:
                self.static_correction_slider_frame.grid()
            else:
                self.static_correction_slider_frame.grid_remove()
        else:
            self.plot_manager.static_corrections = {}
            self.plot_manager.static_correction_preview_mode = False
            self.static_correction_slider_frame.grid_remove()

    def _plot_picks_and_theoretical_curves(self, traces, offsets, times, trace_headers):
        """绘制拾取点及理论走时曲线"""
        if not self.pick_manager:
            return
        picks = self.pick_manager.get_all_picks()
        if not picks:
            return

        filtered_data = self.plot_manager._filter_data(
            traces, offsets, times, self.params, trace_headers
        )
        trace_indices = filtered_data.get('indices', None)

        records = self.loaded_data.get('records', [])
        x_coordinates = None
        if trace_indices:
            x_coordinates = self.plot_manager._calculate_x_coordinates(
                offsets, trace_indices, trace_headers, records, self.params
            )

        alignment_offsets = self.alignment_offsets if self.alignment_active else None
        self.plot_manager.plot_picks(
            picks, offsets, self.params, trace_indices, alignment_offsets, x_coordinates
        )

        if self.show_theoretical_times and self.theoretical_times_data:
            if self.show_water_layer_correction and self.water_layer_corrected_times:
                self.plot_manager.plot_theoretical_traveltime(
                    distances=self.water_layer_corrected_times['distances'],
                    times=self.water_layer_corrected_times['times'],
                    params=self.params,
                    color='blue',
                    linestyle='-.',
                    linewidth=2.0,
                    alpha=0.8,
                    label='水层校正后走时'
                )
                self.plot_manager.plot_theoretical_traveltime(
                    distances=self.theoretical_times_data['distances'],
                    times=self.theoretical_times_data['times'],
                    params=self.params,
                    color='green',
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.6,
                    label='理论走时（原始）'
                )
            else:
                self.plot_manager.plot_theoretical_traveltime(
                    distances=self.theoretical_times_data['distances'],
                    times=self.theoretical_times_data['times'],
                    params=self.params,
                    color='green',
                    linestyle='--',
                    linewidth=2.0,
                    alpha=0.8,
                    label='理论走时'
                )

    def _refresh_pick_overlay_only(self):
        """仅重绘拾取层，避免每次拾取都触发整幅剖面重算。"""
        if not self.data_loaded or not self.loaded_data or not self.pick_manager:
            return
        if not self.plot_manager or not self.params:
            return

        picks = self.pick_manager.get_all_picks()
        # 清理旧拾取点（若 plot_manager 已提供缓存列表）
        if hasattr(self.plot_manager, 'pick_artists'):
            for artist in list(self.plot_manager.pick_artists):
                try:
                    artist.remove()
                except Exception:
                    pass
            self.plot_manager.pick_artists = []

        if picks:
            offsets = self.loaded_data['offsets']
            trace_indices = getattr(self.plot_manager, 'current_filtered_indices', None)
            x_coordinates = getattr(self.plot_manager, 'current_x_coordinates', None)
            alignment_offsets = self.alignment_offsets if self.alignment_active else None
            self.plot_manager.plot_picks(
                picks,
                offsets,
                self.params,
                trace_indices,
                alignment_offsets,
                x_coordinates
            )
        if self.canvas:
            self.canvas.draw_idle()

    def _restore_axis_limits_after_plot(self):
        """绘制后恢复视图范围（保持与原逻辑一致）"""
        if self.preserve_zoom and self.user_xlim and self.user_ylim:
            self.ax.set_xlim(self.user_xlim)
            if isinstance(self.user_ylim, tuple) and len(self.user_ylim) == 2:
                y_min, y_max = self.user_ylim
                if self.params.itrev == 1:
                    self.ax.set_ylim(y_min, y_max)
                else:
                    if y_max < y_min:
                        y_max, y_min = y_min, y_max
                    self.ax.set_ylim(y_max, y_min)
            else:
                self.ax.set_ylim(self.user_ylim)
        else:
            if self.params.xmin is not None and self.params.xmax is not None:
                if self.params.xmin < self.params.xmax:
                    self.ax.set_xlim(self.params.xmin, self.params.xmax)
    
    def update_plot(self):
        """更新绘图（根据当前数据和参数）"""
        self._plot_update_service.update_plot(self)

    def request_plot_refresh(self, delay_ms: int | None = None, immediate: bool = False):
        """请求一次绘图刷新（防抖调度，适合高频 GUI 交互）。"""
        self._plot_refresh_scheduler.request(self, delay_ms=delay_ms, immediate=immediate)
    
    def _delayed_update_plot(self):
        """延迟更新绘图（确保对话框已关闭）"""
        try:
            # 检查窗口是否还存在
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.request_plot_refresh(delay_ms=0)
        except:
            # 窗口已被销毁，跳过更新
            pass
    
    def _try_update_plot_after_load(self):
        """尝试在数据加载后更新绘图（延迟调用）"""
        if self.params and self.data_loaded and self.plot_manager:
            self.request_plot_refresh(delay_ms=0)

    def _ensure_params_initialized(self):
        """确保参数对象已初始化"""
        if self.params is not None:
            return
        if hasattr(self, 'top_toolbar') and hasattr(self.top_toolbar, 'params'):
            self.params = self.top_toolbar.params
        if self.params is None:
            from parameters import ZPlotParameters
            self.params = ZPlotParameters()

    def _calculate_initial_x_range(self, offsets: np.ndarray) -> tuple[float, float]:
        """计算加载数据后的初始X轴范围

        范围策略可通过环境变量 `ZPLOTPY_XRANGE_MODE` 控制：
        - `center50`: 固定为 [-50, 50]（兼容旧行为）
        - 其他/默认: 根据数据自动计算并加入少量边距
        """
        # 默认采用 center50：初始加载统一显示 ±50 km
        mode = os.environ.get('ZPLOTPY_XRANGE_MODE', 'center50').strip().lower()
        if mode == 'center50':
            return -50.0, 50.0

        if offsets is None or len(offsets) == 0:
            return -50.0, 50.0

        valid = np.asarray(offsets, dtype=np.float64)
        valid = valid[np.isfinite(valid)]
        if valid.size == 0:
            return -50.0, 50.0

        x_min = float(np.min(valid))
        x_max = float(np.max(valid))
        x_range = x_max - x_min

        if x_range <= 1e-6:
            padding = max(1.0, abs(x_max) * 0.1)
            return x_min - padding, x_max + padding

        padding = max(0.5, x_range * 0.05)
        return x_min - padding, x_max + padding

    def _apply_loaded_data_defaults(self, header, offsets: np.ndarray):
        """应用数据加载后的默认参数并保存初始快照"""
        self._ensure_params_initialized()
        if self.params is None:
            return

        self.params.tmin = header.start_time
        self.params.tmax = header.end_time
        self.params.irec = 1

        x_min, x_max = self._calculate_initial_x_range(offsets)
        self.params.xmin = x_min
        self.params.xmax = x_max

        # 保存初始参数快照（用于重绘按钮恢复）
        self.initial_params_snapshot = self.params.to_dict()
        # 保存初始的范围参数
        self.initial_range_params = {
            'xmin': self.params.xmin,
            'xmax': self.params.xmax,
            'tmin': self.params.tmin,
            'tmax': self.params.tmax
        }

    def _initialize_pick_manager_from_loaded_data(self, npick: int):
        """根据当前 loaded_data 初始化拾取管理器并统计拾取"""
        self.pick_manager = PickManager(npick=npick)
        self.num_picks = 0
        trace_headers = self.loaded_data.get('trace_headers', [])
        if trace_headers:
            self.pick_manager.set_trace_info_batch(trace_headers)
            stats = self.pick_manager.get_statistics()
            self.num_picks = stats['total_picks']
        return trace_headers

    def _reload_loaded_data(self, status_message: str = '正在重新加载数据...',
                            reinit_pick_manager: bool = False,
                            update_current_record: bool = False):
        """使用当前 d/h/r 文件路径重新加载数据

        Returns:
            (success, header_or_none, error_message)
        """
        if not self.dfile_path:
            return False, None, '请先选择数据文件'

        try:
            self.update_status(status_message)
            self.loaded_data = self.data_loader.load_z_format(
                dfile=self.dfile_path,
                hfile=self.hfile_path,
                rfile=self.rfile_path
            )
            header = self.loaded_data['header']
            self.data_loaded = True
            self.num_traces = header.ntraces

            if reinit_pick_manager:
                self._initialize_pick_manager_from_loaded_data(header.npick)

            if update_current_record:
                records = self.loaded_data.get('records', [])
                if records:
                    self.current_record = records[0].ishnum

            return True, header, ''
        except Exception as e:
            error_msg = f'重新加载数据失败: {str(e)}'
            self.update_status(error_msg)
            return False, None, error_msg
    
    # ========== 文件操作 ==========
    
    def _apply_loaded_data_to_state(self, loaded_data):
        """将已加载数据应用到 GUI 状态（主线程调用）"""
        self.loaded_data = loaded_data
        self.data_loaded = True
        header = self.loaded_data['header']
        self.num_traces = header.ntraces
        self.current_record = 1

        offsets = self.loaded_data.get('offsets', np.array([]))
        self._apply_loaded_data_defaults(header, offsets)
        self._initialize_pick_manager_from_loaded_data(header.npick)

        status_msg = f'数据加载成功: {self.num_traces} 道, {header.npts} 点/道'
        if self.num_picks > 0:
            status_msg += f', 拾取: {self.num_picks}'
        self.update_status(status_msg)

        if hasattr(self, 'top_toolbar') and self.top_toolbar:
            self.top_toolbar.update_widgets_from_params()
        return header

    def _schedule_progressive_initial_render(self, header):
        """大文件首次渲染：先快速预览，再恢复全分辨率"""
        if self.params is None:
            self.root.after(100, lambda: self._delayed_update_plot())
            return

        workload = int(header.ntraces) * int(header.npts)
        large_threshold = int(os.environ.get('ZPLOTPY_LARGE_WORKLOAD', '3000000'))
        if workload < large_threshold:
            self.root.after(100, lambda: self._delayed_update_plot())
            return

        original_ndecim = int(getattr(self.params, 'ndecim', 1) or 1)
        preview_ndecim = max(original_ndecim, max(2, header.npts // 2000))
        self._progressive_render_token += 1
        token = self._progressive_render_token

        self.update_status(
            f'大文件快速预览中... (ndecim={preview_ndecim})，随后自动切换全分辨率'
        )
        self.params.ndecim = preview_ndecim
        self.root.after(50, lambda: self._delayed_update_plot())

        def _restore_full_resolution():
            if token != self._progressive_render_token:
                return
            if self.params is None:
                return
            self.params.ndecim = original_ndecim
            self.update_status('正在切换全分辨率显示...')
            self._delayed_update_plot()

        self.root.after(250, _restore_full_resolution)

    def _background_load_worker(self, dfile_path, hfile_path, rfile_path):
        """后台线程：仅做数据读取，不触碰 UI"""
        try:
            loaded_data = self.data_loader.load_z_format(
                dfile=dfile_path,
                hfile=hfile_path,
                rfile=rfile_path
            )

            def _on_success():
                self._loading_in_background = False
                self._load_thread = None
                header = self._apply_loaded_data_to_state(loaded_data)
                self._schedule_progressive_initial_render(header)
                self._save_workbench_state()

            self.root.after(0, _on_success)
        except Exception as e:
            error_msg = f'数据加载失败: {str(e)}'

            def _on_error():
                self._loading_in_background = False
                self._load_thread = None
                self.data_loaded = False
                self.loaded_data = None
                self.update_status(error_msg)
                import traceback
                traceback.print_exc()

            self.root.after(0, _on_error)

    def load_data_files(self, dfile_path, hfile_path=None, rfile_path=None, async_mode=True):
        """直接加载数据文件（不显示文件对话框）
        
        Args:
            dfile_path: 数据文件路径
            hfile_path: 头文件路径（可选）
            rfile_path: 记录文件路径（可选）
            async_mode: 是否异步加载（默认True）
        """
        if not dfile_path or not Path(dfile_path).exists():
            self.update_status(f'错误: 数据文件不存在: {dfile_path}')
            return False
        
        self.dfile_path = dfile_path
        if hfile_path:
            self.hfile_path = hfile_path
        if rfile_path:
            self.rfile_path = rfile_path

        if async_mode:
            if self._loading_in_background:
                self.update_status('提示：已有加载任务在进行中，请稍候')
                return False
            self.update_status(f'正在后台加载数据文件: {Path(dfile_path).name}...')
            self._loading_in_background = True
            self._load_thread = threading.Thread(
                target=self._background_load_worker,
                args=(dfile_path, self.hfile_path, self.rfile_path),
                daemon=True
            )
            self._load_thread.start()
            return True

        self.update_status(f'正在加载数据文件: {Path(dfile_path).name}...')
        try:
            loaded_data = self.data_loader.load_z_format(
                dfile=dfile_path,
                hfile=self.hfile_path,
                rfile=self.rfile_path
            )
            header = self._apply_loaded_data_to_state(loaded_data)
            self._schedule_progressive_initial_render(header)
            self._save_workbench_state()
            return True
        except Exception as e:
            self.data_loaded = False
            self.loaded_data = None
            error_msg = f'数据加载失败: {str(e)}'
            self.update_status(error_msg)
            import traceback
            traceback.print_exc()
            return False
    
    def open_data_file(self):
        """打开数据文件"""
        filename = filedialog.askopenfilename(
            title='选择数据文件',
            filetypes=[
                ('Z格式文件', '*.z'),
                ('SEGY文件', '*.sgy *.segy'),
                ('SU文件', '*.su'),
                ('所有文件', '*.*')
            ]
        )
        if filename:
            loaded = self.load_data_files(
                filename, self.hfile_path, self.rfile_path, async_mode=True
            )
            if loaded and self.loaded_data:
                header = self.loaded_data['header']
                try:
                    if self.root.winfo_exists():
                        messagebox.showinfo(
                            '成功',
                            f'数据文件加载成功！\n\n'
                            f'总道数: {header.ntraces}\n'
                            f'每道采样点数: {header.npts}\n'
                            f'采样间隔: {header.sampling_interval:.6f} 秒\n'
                            f'时间范围: {header.start_time:.3f} - {header.end_time:.3f} 秒\n'
                            f'记录数: {header.nrec}\n'
                            f'拾取字数: {header.npick}'
                        )
                except Exception:
                    pass
            elif not loaded:
                try:
                    if hasattr(self, 'root') and self.root.winfo_exists():
                        messagebox.showerror('错误', '数据加载失败，请查看状态栏或控制台日志')
                except Exception:
                    pass
    
    def open_header_file(self):
        """打开头文件"""
        filename = filedialog.askopenfilename(
            title='选择头文件',
            filetypes=[('头文件', '*.hdr'), ('所有文件', '*.*')]
        )
        if filename:
            self.hfile_path = filename
            self.update_status(f'头文件已选择: {Path(filename).name}')
            
            # 如果数据文件已加载，重新加载以包含头文件信息
            if self.dfile_path:
                success, header, error_msg = self._reload_loaded_data(
                    status_message='正在重新加载数据（包含头文件）...',
                    reinit_pick_manager=True,
                    update_current_record=False
                )
                if success:
                    trace_headers = self.loaded_data['trace_headers']

                    # 检查道数是否匹配
                    traces_count = len(self.loaded_data['traces'])
                    headers_count = len(trace_headers)

                    status_msg = f'头文件加载成功: {headers_count} 条道头记录'
                    if headers_count != traces_count:
                        status_msg += f' (数据文件: {traces_count} 道)'
                    if self.num_picks > 0:
                        status_msg += f', 拾取: {self.num_picks}'
                    self.update_status(status_msg)

                    info_msg = f'头文件加载成功！\n\n'
                    info_msg += f'数据文件: {traces_count} 道\n'
                    info_msg += f'头文件: {headers_count} 条道头记录\n'
                    if headers_count != traces_count:
                        info_msg += f'\n⚠ 警告: 道数不匹配！'
                    if self.num_picks > 0:
                        info_msg += f'\n已加载 {self.num_picks} 个拾取点'
                    messagebox.showinfo('成功', info_msg)

                    if self.data_loaded:
                        self.root.after(100, lambda: self._delayed_update_plot())
                else:
                    messagebox.showerror('错误', error_msg)
            else:
                messagebox.showinfo('信息', 
                    f'头文件已选择: {filename}\n\n'
                    f'提示：请先加载数据文件，然后重新加载以应用头文件信息。')
    
    def open_record_file(self):
        """打开记录文件"""
        filename = filedialog.askopenfilename(
            title='选择记录文件',
            filetypes=[('记录文件', '*.rec *.rsp'), ('文本文件', '*.txt'), ('所有文件', '*.*')]
        )
        if filename:
            self.rfile_path = filename
            self.update_status(f'记录文件已选择: {Path(filename).name}')
            
            # 如果数据文件已加载，重新加载以包含记录文件信息
            if self.dfile_path:
                success, _, error_msg = self._reload_loaded_data(
                    status_message='正在重新加载数据（包含记录文件）...',
                    reinit_pick_manager=False,
                    update_current_record=True
                )
                if success:
                    records = self.loaded_data['records']
                    if records:
                        self.update_status(f'记录文件加载成功: {len(records)} 条记录')
                        messagebox.showinfo('成功', 
                            f'记录文件加载成功！\n\n'
                            f'已加载 {len(records)} 条记录（炮集）\n'
                            f'当前记录: {self.current_record}')
                    else:
                        self.update_status('记录文件为空或格式不正确')
                        messagebox.showwarning('警告', '记录文件为空或格式不正确')
                else:
                    messagebox.showerror('错误', error_msg)
            else:
                messagebox.showinfo('信息', 
                    f'记录文件已选择: {filename}\n\n'
                    f'提示：请先加载数据文件，然后重新加载以应用记录文件信息。')
    
    def save_picks(self):
        """保存拾取结果（zplot.out 格式）"""
        if not self.data_loaded or not self.pick_manager:
            messagebox.showwarning('警告', '请先加载数据文件')
            return
        
        filename = filedialog.asksaveasfilename(
            title='保存拾取结果（zplot.out格式）',
            defaultextension='.out',
            filetypes=[('输出文件', '*.out'), ('文本文件', '*.txt'), ('所有文件', '*.*')]
        )
        if filename:
            try:
                if self.pick_manager.save_picks(filename, format='zplot'):
                    self.update_status(f'拾取结果已保存: {Path(filename).name}')
                    stats = self.pick_manager.get_statistics()
                    messagebox.showinfo('成功', 
                        f'拾取结果已保存到: {filename}\n\n'
                        f'总拾取数: {stats["total_picks"]}\n'
                        f'有拾取的道数: {stats["traces_with_picks"]}')
                else:
                    messagebox.showerror('错误', '保存拾取失败')
            except Exception as e:
                messagebox.showerror('错误', f'保存拾取失败: {str(e)}')
    
    def save_picks_to_header(self):
        """保存拾取到头文件（.hdr，类似 headup 命令）"""
        if not self.data_loaded or not self.pick_manager or not self.hfile_path:
            messagebox.showwarning('警告', '请先加载数据文件和头文件')
            return
        
        # 确认是否覆盖现有头文件
        if not messagebox.askyesno('确认', 
            f'确定要将拾取保存到头文件吗？\n\n'
            f'文件: {self.hfile_path}\n\n'
            f'这将覆盖现有的头文件。'):
            return
        
        try:
            trace_headers = self.loaded_data.get('trace_headers', [])
            if not trace_headers:
                messagebox.showerror('错误', '没有道头信息')
                return
            
            # 保存前先更新道头中的拾取信息
            # 注意：update_header_picks 会直接修改 trace_headers 中的 picks 数组
            if not self.pick_manager.update_header_picks(trace_headers):
                messagebox.showerror('错误', '更新道头拾取信息失败')
                return
            
            # 验证更新后的拾取数量
            updated_pick_count = 0
            for th in trace_headers:
                if th.picks:
                    updated_pick_count += sum(1 for p in th.picks if p > 0)
            
            # 保存到头文件
            if self.pick_manager.save_to_header_file(self.hfile_path, trace_headers):
                # 确保 loaded_data 中的 trace_headers 已更新（实际上已经更新，因为传入的是同一个对象引用）
                # 但为了确保同步，我们显式更新一下
                self.loaded_data['trace_headers'] = trace_headers
                
                stats = self.pick_manager.get_statistics()
                self.update_status(f'拾取已保存到头文件: {Path(self.hfile_path).name}')
                
                # 验证保存后的状态
                verification_msg = f'拾取已保存到头文件: {self.hfile_path}\n\n'
                verification_msg += f'PickManager 中的拾取数: {stats["total_picks"]}\n'
                verification_msg += f'道头中的拾取数: {updated_pick_count}\n'
                verification_msg += f'有拾取的道数: {stats["traces_with_picks"]}\n\n'
                
                if stats["total_picks"] != updated_pick_count:
                    verification_msg += f'⚠ 警告: 拾取数量不匹配！\n'
                    verification_msg += f'这可能表示某些拾取点未正确更新到道头。\n\n'
                
                verification_msg += f'拾取点已更新到内存中的道头信息。\n'
                verification_msg += f'下次打开时会自动加载这些拾取。'
                
                messagebox.showinfo('成功', verification_msg)
            else:
                messagebox.showerror('错误', '保存头文件失败')
        except Exception as e:
            messagebox.showerror('错误', f'保存头文件失败: {str(e)}')
            import traceback
            traceback.print_exc()
    
    def save_parameters(self):
        """保存参数配置到文件"""
        if not hasattr(self, 'top_toolbar') or not self.top_toolbar:
            messagebox.showwarning('警告', '工具栏未初始化')
            return
        
        filename = filedialog.asksaveasfilename(
            title='保存参数配置',
            defaultextension='.json',
            filetypes=[('JSON文件', '*.json'), ('所有文件', '*.*')]
        )
        
        if filename:
            try:
                # 获取当前参数
                params_dict = self.top_toolbar.params.to_dict()
                
                # 添加版本信息和元数据
                save_data = {
                    'version': '1.0',
                    'description': 'ZPLOT参数配置',
                    'parameters': params_dict
                }
                
                # 保存到JSON文件
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=2, ensure_ascii=False)
                
                self.update_status(f'参数配置已保存: {Path(filename).name}')
                messagebox.showinfo('成功', f'参数配置已保存到:\n{filename}')
            except Exception as e:
                messagebox.showerror('错误', f'保存参数配置失败: {str(e)}')
                import traceback
                traceback.print_exc()
    
    def load_parameters(self):
        """从文件加载参数配置"""
        if not hasattr(self, 'top_toolbar') or not self.top_toolbar:
            messagebox.showwarning('警告', '工具栏未初始化')
            return
        
        filename = filedialog.askopenfilename(
            title='加载参数配置',
            defaultextension='.json',
            filetypes=[('JSON文件', '*.json'), ('所有文件', '*.*')]
        )
        
        if filename:
            try:
                # 读取JSON文件
                with open(filename, 'r', encoding='utf-8') as f:
                    save_data = json.load(f)
                
                # 检查版本
                if 'parameters' not in save_data:
                    messagebox.showerror('错误', '无效的参数配置文件格式')
                    return
                
                # 加载参数
                params_dict = save_data['parameters']
                self.top_toolbar.params.from_dict(params_dict)
                
                # 更新主窗口的参数对象
                self.params = self.top_toolbar.params
                
                # 更新工具栏控件显示
                self.top_toolbar.update_widgets_from_params()
                
                # 应用参数并重绘
                self.request_plot_refresh(immediate=True)
                
                self.update_status(f'参数配置已加载: {Path(filename).name}')
                messagebox.showinfo('成功', 
                    f'参数配置已加载:\n{filename}\n\n'
                    f'参数已应用到界面并重绘图形。')
            except json.JSONDecodeError as e:
                messagebox.showerror('错误', f'JSON文件格式错误: {str(e)}')
            except Exception as e:
                messagebox.showerror('错误', f'加载参数配置失败: {str(e)}')
                import traceback
                traceback.print_exc()
    
    def convert_hdr_to_tx(self):
        """将头文件转换为 tx.in 格式（类似 z2tx 命令）"""
        if not self.data_loaded or not self.hfile_path:
            messagebox.showwarning('警告', '请先加载头文件')
            return
        
        # 创建配置对话框（简化版）
        from tkinter.simpledialog import askfloat
        
        # 获取所有唯一的 OBS
        trace_headers = self.loaded_data.get('trace_headers', [])
        if not trace_headers:
            messagebox.showerror('错误', '没有道头信息')
            return
        
        obs_shots = sorted(set(th.ishoti for th in trace_headers))
        
        # 创建配置对话框
        config_dialog = tk.Toplevel(self.root)
        config_dialog.title('tx.in 转换配置')
        config_dialog.geometry('500x400')
        
        configs = {}
        
        # 为每个 OBS 创建配置输入
        row = 0
        for shot in obs_shots:
            ttk.Label(config_dialog, text=f'OBS {shot}:').grid(row=row, column=0, padx=5, pady=2, sticky='w')
            
            # xmod
            ttk.Label(config_dialog, text='xmod:').grid(row=row, column=1, padx=2)
            xmod_var = tk.DoubleVar(value=0.0)
            ttk.Entry(config_dialog, textvariable=xmod_var, width=10).grid(row=row, column=2, padx=2)
            
            # tshift
            ttk.Label(config_dialog, text='tshift:').grid(row=row, column=3, padx=2)
            tshift_var = tk.DoubleVar(value=0.0)
            ttk.Entry(config_dialog, textvariable=tshift_var, width=10).grid(row=row, column=4, padx=2)
            
            # xshift
            ttk.Label(config_dialog, text='xshift:').grid(row=row, column=5, padx=2)
            xshift_var = tk.DoubleVar(value=0.0)
            ttk.Entry(config_dialog, textvariable=xshift_var, width=10).grid(row=row, column=6, padx=2)
            
            configs[shot] = {
                'xmod': xmod_var,
                'tshift': tshift_var,
                'xshift': xshift_var,
                'picku': [0.05] * self.loaded_data['header'].npick  # 默认不确定性
            }
            
            row += 1
        
        # 不确定性输入（所有拾取字共用）
        ttk.Label(config_dialog, text='拾取不确定性 (picku):').grid(row=row, column=0, padx=5, pady=5, sticky='w')
        picku_var = tk.StringVar(value='0.05')
        ttk.Entry(config_dialog, textvariable=picku_var, width=20).grid(row=row, column=1, columnspan=3, padx=5)
        ttk.Label(config_dialog, text='(可输入单个值或逗号分隔的列表)').grid(row=row, column=4, columnspan=3, padx=2, sticky='w')
        row += 1
        
        def do_convert():
            try:
                # 解析 picku
                picku_str = picku_var.get().strip()
                if ',' in picku_str:
                    picku_list = [float(x.strip()) for x in picku_str.split(',')]
                else:
                    picku_val = float(picku_str)
                    picku_list = [picku_val] * self.loaded_data['header'].npick
                
                # 构建配置
                obs_configs = {}
                for shot, vars_dict in configs.items():
                    obs_configs[shot] = {
                        'xmod': vars_dict['xmod'].get(),
                        'tshift': vars_dict['tshift'].get(),
                        'xshift': vars_dict['xshift'].get(),
                        'picku': picku_list
                    }
                
                config = Z2TxConfig(obs_configs=obs_configs, iamp=0)
                
                # 选择输出文件
                output_file = filedialog.asksaveasfilename(
                    title='保存 tx.in 文件',
                    defaultextension='.in',
                    filetypes=[('tx.in文件', '*.in'), ('所有文件', '*.*')]
                )
                
                if output_file:
                    success, total_picks = convert_hdr_to_tx(
                        self.hfile_path, 
                        output_file, 
                        config, 
                        self.loaded_data['header'].npick
                    )
                    
                    if success:
                        messagebox.showinfo('成功', 
                            f'tx.in 文件已生成: {output_file}\n\n'
                            f'总拾取数: {total_picks}')
                        config_dialog.destroy()
                    else:
                        messagebox.showerror('错误', '转换失败')
            except Exception as e:
                messagebox.showerror('错误', f'转换失败: {str(e)}')
                import traceback
                traceback.print_exc()
        
        ttk.Button(config_dialog, text='转换', command=do_convert).grid(row=row, column=0, columnspan=7, pady=10)
        ttk.Button(config_dialog, text='取消', command=config_dialog.destroy).grid(row=row+1, column=0, columnspan=7)
    
    def reload_data(self):
        """重新加载数据"""
        if not self.dfile_path:
            messagebox.showwarning('警告', '请先选择数据文件')
            return
        
        success, _, error_msg = self._reload_loaded_data(
            status_message='正在重新加载数据...',
            reinit_pick_manager=True,
            update_current_record=True
        )
        if success:
            self.update_status('数据重新加载成功')
            messagebox.showinfo('成功', '数据重新加载成功！')
        else:
            messagebox.showerror('错误', error_msg)
    
    def show_data_info(self):
        """显示数据信息"""
        if not self.data_loaded or not self.loaded_data:
            messagebox.showwarning('警告', '请先加载数据文件')
            return
        
        header = self.loaded_data['header']
        trace_headers = self.loaded_data['trace_headers']
        records = self.loaded_data['records']
        
        info_text = f"""数据文件信息：

文件路径：
  数据文件: {self.dfile_path or '未加载'}
  头文件: {self.hfile_path or '未加载'}
  记录文件: {self.rfile_path or '未加载'}

文件头信息：
  总道数: {header.ntraces}
  每道采样点数: {header.npts}
  采样间隔: {header.sampling_interval:.6f} 秒
  时间范围: {header.start_time:.3f} - {header.end_time:.3f} 秒
  记录数: {header.nrec}
  拾取字数: {header.npick}
  折合速度: {header.vredf:.3f} km/s
  数据格式: {'float32' if header.ifmt == 1 else 'int16'}

道头信息：
  已加载道头数: {len(trace_headers)}
  有效道数: {sum(1 for th in trace_headers if th.iflagi == 1)}

记录信息：
  已加载记录数: {len(records)}
  当前记录: {self.current_record}
"""
        
        messagebox.showinfo('数据信息', info_text)
    
    def export_figure(self):
        """导出图形"""
        if self.fig is None:
            messagebox.showwarning('警告', '没有可导出的图形')
            return
        
        filename = filedialog.asksaveasfilename(
            title='导出图形',
            defaultextension='.png',
            filetypes=[
                ('PNG文件', '*.png'),
                ('PDF文件', '*.pdf'),
                ('EPS文件', '*.eps'),
                ('所有文件', '*.*')
            ]
        )
        if filename:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            self.update_status(f'图形已导出: {Path(filename).name}')
            messagebox.showinfo('成功', f'图形已保存到: {filename}')
    
    # ========== 编辑操作 ==========
    
    def edit_parameters(self):
        """编辑参数"""
        try:
            from parameter_edit_dialog import ParameterEditDialog
        except ImportError:
            from .parameter_edit_dialog import ParameterEditDialog
        
        # 确保参数对象已初始化
        if self.params is None:
            if hasattr(self, 'top_toolbar') and self.top_toolbar and hasattr(self.top_toolbar, 'params'):
                self.params = self.top_toolbar.params
            else:
                from parameters import ZPlotParameters
                self.params = ZPlotParameters()
        
        def on_params_updated():
            """参数更新后的回调"""
            # 确保工具栏的参数对象与主窗口的参数对象同步
            if hasattr(self, 'top_toolbar') and self.top_toolbar:
                # 同步参数对象
                self.top_toolbar.params = self.params
                self.top_toolbar.update_widgets_from_params()
            # 强制重新绘制（清除用户缩放状态，使用新参数）
            self.preserve_zoom = False
            self.user_xlim = None
            self.user_ylim = None
            self.request_plot_refresh(immediate=True)
        
        dialog = ParameterEditDialog(self.root, self.params, callback=on_params_updated)
    
    def clear_picks(self):
        """清除拾取"""
        if not self.pick_manager:
            messagebox.showwarning('警告', '拾取管理器未初始化')
            return
        
        if messagebox.askyesno('确认', '确定要清除所有拾取吗？\n\n这将删除所有已拾取的走时点。'):
            # 清除所有拾取点
            self.pick_manager.clear_picks()
            # 更新图形显示
            self.request_plot_refresh(immediate=True)
            # 更新状态栏
            stats = self.pick_manager.get_statistics()
            self.update_status(f'已清除所有拾取点（共 {stats["total_picks"]} 个）')
            messagebox.showinfo('成功', '所有拾取已清除')
    
    # ========== 视图操作 ==========
    
    def zoom_in(self):
        """放大"""
        # TODO: 实现缩放功能
        pass
    
    def zoom_out(self):
        """缩小"""
        # TODO: 实现缩放功能
        pass
    
    # ========== 记录导航 ==========
    
    def prev_record(self):
        """切换到上一个记录"""
        if not self.data_loaded or not self.loaded_data:
            self.update_status('请先加载数据文件')
            return
        
        header = self.loaded_data['header']
        if header.nrec <= 1:
            self.update_status('只有一个记录，无法切换')
            return
        
        # 计算上一个记录号（循环）
        new_record = self.current_record - 1
        if new_record < 1:
            new_record = header.nrec
        
        self.switch_to_record(new_record)
    
    def next_record(self):
        """切换到下一个记录"""
        if not self.data_loaded or not self.loaded_data:
            self.update_status('请先加载数据文件')
            return
        
        header = self.loaded_data['header']
        if header.nrec <= 1:
            self.update_status('只有一个记录，无法切换')
            return
        
        # 计算下一个记录号（循环）
        new_record = self.current_record + 1
        if new_record > header.nrec:
            new_record = 1
        
        self.switch_to_record(new_record)
    
    def switch_to_record(self, record_num: int):
        """切换到指定记录
        
        Args:
            record_num: 记录号（从1开始）
        """
        if not self.data_loaded or not self.loaded_data:
            return
        
        header = self.loaded_data['header']
        if record_num < 1 or record_num > header.nrec:
            self.update_status(f'无效的记录号: {record_num}（范围: 1-{header.nrec}）')
            return
        
        # 检查记录号是否存在
        trace_headers = self.loaded_data.get('trace_headers', [])
        record_exists = False
        if trace_headers:
            for th in trace_headers:
                if th.ishoti == record_num:
                    record_exists = True
                    break
        
        if not record_exists:
            self.update_status(f'记录号 {record_num} 不存在（范围: 1-{header.nrec}）')
            messagebox.showwarning('警告', 
                f'记录号 {record_num} 不存在！\n\n'
                f'可用记录号范围: 1-{header.nrec}\n\n'
                f'请检查记录号是否正确。')
            return
        
        # 更新当前记录号
        self.current_record = record_num
        
        # 更新参数中的记录号
        if hasattr(self, 'top_toolbar') and self.top_toolbar:
            self.top_toolbar.params.irec = record_num
            # 更新工具栏中的记录号显示
            if 'irec' in self.top_toolbar.param_widgets:
                widget, var, menu_id, param_type, widget_type = self.top_toolbar.param_widgets['irec']
                if widget_type == 'spin':
                    var.set(record_num)
        
        # 更新主窗口的参数对象
        if hasattr(self, 'params'):
            self.params.irec = record_num
        
        # 清除用户设置的缩放
        self.preserve_zoom = False
        self.user_xlim = None
        self.user_ylim = None
        
        # 重新绘制图形
        self.request_plot_refresh(immediate=True)
        
        # 更新状态栏
        self.update_status(f'已切换到记录 {record_num}/{header.nrec}')
        self._save_workbench_state()

    def _load_workbench_state(self) -> dict:
        return self._state_service.load_state(self)

    def _save_workbench_state(self) -> None:
        self._state_service.save_state(self)

    def _restore_workbench_state(self) -> None:
        self._state_service.restore_state(self)

    def _on_close(self) -> None:
        self._save_workbench_state()
        self.root.destroy()

    def _resolve_restorable_input_path(self, path_text: str) -> str:
        return self._state_service.resolve_restorable_input_path(self, path_text)

    @staticmethod
    def _candidate_path_keys(path_text: str) -> list[str]:
        return WorkbenchStateService.candidate_path_keys(path_text)

    @classmethod
    def _candidate_existing_paths(cls, path_text: str) -> list[Path]:
        return WorkbenchStateService.candidate_existing_paths(path_text)
    
    def reset_view(self):
        """重置视图"""
        if self.ax:
            # 清除用户设置的缩放
            self.preserve_zoom = False
            self.user_xlim = None
            self.user_ylim = None
            # 重新绘制以使用默认范围
            self.request_plot_refresh(immediate=True)
            self.update_status('视图已重置')
    
    def toggle_toolbar(self):
        """切换工具栏显示"""
        if self.top_toolbar.expanded:
            self.top_toolbar.toggle()
    
    # ========== 工具操作 ==========
    
    def enter_pick_mode(self):
        """进入/退出拾取模式"""
        if not self.data_loaded:
            messagebox.showwarning('警告', '请先加载数据文件')
            return
        
        self.pick_mode = not self.pick_mode
        if self.pick_mode:
            self.update_status('拾取模式：右键添加拾取，左键删除拾取（按 P 退出）')
            # 连接鼠标点击事件（如果还未连接）
            if self.canvas and self.pick_event_connector is None:
                self.pick_event_connector = self.canvas.mpl_connect('button_press_event', self.on_pick_click)
                # 连接鼠标移动事件（用于拖拽拾取点，后续实现）
                # self.motion_connector = self.canvas.mpl_connect('motion_notify_event', self.on_pick_motion)
        else:
            self.update_status('已退出拾取模式')
            # 断开鼠标点击事件
            if self.canvas and self.pick_event_connector is not None:
                try:
                    self.canvas.mpl_disconnect(self.pick_event_connector)
                except:
                    pass
                self.pick_event_connector = None
    
    def find_nearest_trace(self, x_click: Optional[float] = None) -> Optional[int]:
        """查找距离鼠标最近的trace
        
        Args:
            x_click: X坐标（如果为None，使用当前鼠标位置）
            
        Returns:
            trace_idx 如果找到，否则 None
        """
        if not self.data_loaded or not self.plot_manager:
            return None
        
        # 使用鼠标位置或提供的坐标
        if x_click is None:
            if self.mouse_x is None:
                return None
            x_click = self.mouse_x
        
        # 获取数据
        traces = self.loaded_data['traces']
        offsets = self.loaded_data['offsets']
        times = self.loaded_data['times']
        trace_headers = self.loaded_data.get('trace_headers', [])
        
        # 优先复用当前绘图缓存，避免每次按键都重复全量过滤
        filtered_indices = getattr(self.plot_manager, 'current_filtered_indices', None)
        cached_x = getattr(self.plot_manager, 'current_x_coordinates', None)
        if filtered_indices is not None and cached_x is not None and len(filtered_indices) > 0:
            filtered_offsets = np.asarray(cached_x)
        else:
            filtered_data = self.plot_manager._filter_data(
                traces, offsets, times, self.params, trace_headers
            )
            filtered_offsets = filtered_data['offsets']
            filtered_indices = filtered_data['indices']
        
        if len(filtered_offsets) == 0:
            return None
        
        # 找到最近的trace
        filtered_idx = np.argmin(np.abs(filtered_offsets - x_click))
        
        if filtered_idx < len(filtered_indices):
            return filtered_indices[filtered_idx]
        else:
            # 回退到原始方法
            return np.argmin(np.abs(offsets - x_click)) if len(offsets) > 0 else None
    
    def toggle_trace_visibility(self):
        """切换最近trace的显示状态（X键功能）
        
        移除或重新绘制距离光标最近的trace（以及相关的picks）。
        当trace被移除时，在数据窗口顶部显示一个小点作为提醒。
        header word 5 (iflagi) 会被设置为当前值的负值。
        """
        if not self.data_loaded or not self.plot_manager:
            self.update_status('无法切换trace：数据未加载')
            return
        
        # 找到最近的trace
        trace_idx = self.find_nearest_trace()
        if trace_idx is None:
            self.update_status('无法找到最近的trace')
            return
        
        # 获取trace header
        trace_headers = self.loaded_data.get('trace_headers', [])
        if trace_idx >= len(trace_headers):
            self.update_status(f'Trace {trace_idx} 没有对应的header信息')
            return
        
        th = trace_headers[trace_idx]
        offsets = self.loaded_data['offsets']
        
        if trace_idx >= len(offsets):
            return
        
        offset = offsets[trace_idx]
        
        # 切换trace状态
        if trace_idx in self.plot_manager.removed_traces:
            # 重新绘制：恢复trace
            del self.plot_manager.removed_traces[trace_idx]
            # 恢复 iflagi 为正值
            th.iflagi = abs(th.iflagi)
            self.update_status(f'Trace {trace_idx} (offset={offset:.2f}km) 已恢复显示')
        else:
            # 移除：隐藏trace
            self.plot_manager.removed_traces[trace_idx] = offset
            # 设置 iflagi 为负值
            th.iflagi = -abs(th.iflagi)
            self.update_status(f'Trace {trace_idx} (offset={offset:.2f}km) 已移除')
        
        # 更新绘图（包括顶部标记）
        self.request_plot_refresh(delay_ms=20)
    
    def find_nearest_pick(self, x_click: float, y_click: float, 
                         threshold_x: Optional[float] = None,
                         threshold_y: Optional[float] = None) -> Optional[Tuple[int, int]]:
        """查找最近的拾取点
        
        Args:
            x_click: 鼠标点击的 X 坐标（数据坐标）
            y_click: 鼠标点击的 Y 坐标（数据坐标）
            threshold_x: X 方向阈值（如果为None，使用自动计算）
            threshold_y: Y 方向阈值（如果为None，使用自动计算）
            
        Returns:
            (trace_idx, pick_word) 如果找到，否则 None
        """
        if not self.pick_manager or not self.data_loaded:
            return None
        
        # 获取数据（使用与绘图时相同的过滤逻辑）
        traces = self.loaded_data['traces']
        offsets = self.loaded_data['offsets']
        times = self.loaded_data['times']
        trace_headers = self.loaded_data.get('trace_headers', [])
        
        # 优先复用当前绘图缓存，避免重复全量过滤
        filtered_indices = getattr(self.plot_manager, 'current_filtered_indices', None)
        cached_x = getattr(self.plot_manager, 'current_x_coordinates', None)
        if filtered_indices is not None and cached_x is not None and len(filtered_indices) > 0:
            filtered_offsets = np.asarray(cached_x)
        else:
            filtered_data = self.plot_manager._filter_data(
                traces, offsets, times, self.params, trace_headers
            )
            filtered_offsets = filtered_data['offsets']
            filtered_indices = filtered_data['indices']  # 原始索引映射
        
        # 在过滤后的 offsets 中找到最近的道
        if len(filtered_offsets) == 0:
            return None
        
        filtered_idx = np.argmin(np.abs(filtered_offsets - x_click))
        
        # 获取对应的原始 trace_idx
        if filtered_idx < len(filtered_indices):
            trace_idx = filtered_indices[filtered_idx]
        else:
            # 如果索引映射失败，回退到原始方法
            trace_idx = np.argmin(np.abs(offsets - x_click))
        
        # 检查 trace_idx 是否有效
        if trace_idx >= len(offsets):
            return None
        
        offset = offsets[trace_idx]
        
        # 计算阈值（基于 spick 参数和坐标轴范围）
        if threshold_x is None or threshold_y is None:
            # 使用 spick 参数和坐标轴范围来估算阈值
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            x_range = abs(xlim[1] - xlim[0])
            y_range = abs(ylim[1] - ylim[0])
            
            # spick 是拾取符号大小（单位：毫米），转换为数据坐标
            spick = self.params.spick if self.params else 0.5
            # 假设 spick 对应约 5% 的坐标轴范围
            if threshold_x is None:
                threshold_x = x_range * 0.05 * spick
            if threshold_y is None:
                threshold_y = y_range * 0.05 * spick
        
        # 遍历所有拾取点
        min_dist = float('inf')
        nearest_pick = None
        
        for tidx, pick_dict in self.pick_manager.picks.items():
            if tidx >= len(offsets):
                continue
            
            trace_offset = offsets[tidx]
            
            for pword, time in pick_dict.items():
                if time <= 0:  # 无效拾取
                    continue
                
                # 应用折合时间转换（用于显示坐标）
                if self.params and self.params.vred > 0:
                    plot_time = time - abs(trace_offset) / self.params.vred
                else:
                    plot_time = time
                
                # 计算距离
                dx = abs(x_click - trace_offset)
                dy = abs(y_click - plot_time)
                
                # 检查是否在阈值范围内
                if dx < threshold_x and dy < threshold_y:
                    dist = dx * dx + dy * dy  # 欧氏距离的平方
                    if dist < min_dist:
                        min_dist = dist
                        nearest_pick = (tidx, pword)
        
        return nearest_pick
    
    def on_pick_click(self, event):
        """处理拾取点击事件
        
        修改后的逻辑：
        - 左键（button=1）：添加/更新拾取点
        - 右键（button=3）：删除拾取点（如果在拾取点上）
        - 中键（button=2）：显示拾取走时信息菜单（如果在拾取点上）
        """
        if not self.pick_mode or not self.pick_manager or not self.data_loaded:
            return
        
        if event.inaxes != self.ax:
            return
        
        # 获取点击位置
        x_click = event.xdata
        y_click = event.ydata
        
        if x_click is None or y_click is None:
            return
        
        # 获取数据（使用与绘图时相同的过滤逻辑）
        traces = self.loaded_data['traces']
        offsets = self.loaded_data['offsets']
        times = self.loaded_data['times']
        trace_headers = self.loaded_data.get('trace_headers', [])
        
        # 优先复用当前绘图缓存，避免每次点击重复全量过滤导致卡顿
        filtered_indices = getattr(self.plot_manager, 'current_filtered_indices', None)
        cached_x = getattr(self.plot_manager, 'current_x_coordinates', None)
        if filtered_indices is None or cached_x is None or len(filtered_indices) == 0:
            filtered_data = self.plot_manager._filter_data(
                traces, offsets, times, self.params, trace_headers
            )
            filtered_offsets = filtered_data['offsets']
            filtered_indices = filtered_data['indices']
        else:
            filtered_offsets = np.asarray(cached_x)
        
        # 在过滤后的 offsets 中找到最近的道
        if len(filtered_offsets) == 0:
            self.update_status('错误: 没有可用的道数据')
            return
        
        filtered_idx = np.argmin(np.abs(filtered_offsets - x_click))
        
        # 获取对应的原始 trace_idx
        if filtered_idx < len(filtered_indices):
            trace_idx = filtered_indices[filtered_idx]
        else:
            # 如果索引映射失败，回退到原始方法
            trace_idx = np.argmin(np.abs(offsets - x_click))
        
        offset = offsets[trace_idx] if trace_idx < len(offsets) else filtered_offsets[filtered_idx]
        
        # 检查 trace_idx 是否有效
        if trace_idx >= len(self.loaded_data['traces']):
            self.update_status(f'错误: 道索引 {trace_idx} 超出范围 (0-{len(self.loaded_data["traces"])-1})')
            return
        
        # 获取活动拾取字
        pick_word = self.params.apick if self.params else 1
        
        # 根据按键类型执行不同操作
        if event.button == 1:  # 左键：添加/更新拾取点
            # 添加新拾取点
            # 应用折合时间（如果需要）
            if self.params and self.params.vred > 0:
                # 如果使用了折合时间，需要反向计算原始时间
                original_time = y_click + abs(offset) / self.params.vred
            else:
                original_time = y_click
            
            # 检查时间是否在有效范围内
            if original_time < times[0] or original_time > times[-1]:
                self.update_status(f'时间超出范围: {original_time:.3f} 不在 [{times[0]:.3f}, {times[-1]:.3f}]')
                return
            
            # 添加/更新拾取
            if self.pick_manager.add_pick(trace_idx, original_time, pick_word):
                self.num_picks = self.pick_manager.get_statistics()['total_picks']
                self.update_status(f'拾取已添加: 道{trace_idx+1}, 拾取字{pick_word}, 时间{original_time:.3f}s')
                self._refresh_pick_overlay_only()
            else:
                messagebox.showerror('错误', f'拾取失败：拾取字 {pick_word} 超出范围 (1-{self.pick_manager.npick})')
        
        elif event.button == 3:  # 右键：删除当前拾取字对应的拾取点（在该道上）
            # 不需要检查是否点击在拾取点上，只要鼠标在该道上，就删除当前拾取字对应的走时点
            # 获取当前拾取字
            pick_word = self.params.apick if self.params else 1
            
            # 检查该道上是否有当前拾取字的拾取点
            pick_time = self.pick_manager.get_pick(trace_idx, pick_word)
            if pick_time and pick_time > 0:
                # 删除拾取点
                if self.pick_manager.remove_pick(trace_idx, pick_word):
                    self.num_picks = self.pick_manager.get_statistics()['total_picks']
                    self.update_status(f'拾取已删除: 道{trace_idx+1}, 拾取字{pick_word}')
                    self._refresh_pick_overlay_only()
                else:
                    self.update_status('删除拾取失败')
            else:
                # 该道上没有当前拾取字的拾取点
                self.update_status(f'道{trace_idx+1}上没有拾取字{pick_word}的拾取点')
        
        # 中键功能已取消，后续再决定赋予其什么功能
        # elif event.button == 2:
        #     ...
    
    def show_pick_context_menu(self, event, pick_info: Tuple[int, int]):
        """显示拾取点信息菜单（中键触发）
        
        Args:
            event: 鼠标事件（matplotlib MouseEvent）
            pick_info: (trace_idx, pick_word) 元组
        """
        trace_idx, pick_word = pick_info
        
        # 创建右键菜单
        context_menu = tk.Menu(self.root, tearoff=0, font=get_chinese_font(10))
        
        # 获取拾取时间
        pick_time = self.pick_manager.get_pick(trace_idx, pick_word)
        if pick_time:
            context_menu.add_command(
                label=f'拾取信息: 道{trace_idx+1}, 拾取字{pick_word}, 时间{pick_time:.3f}s',
                state='disabled'
            )
            context_menu.add_separator()
        
        # 菜单项
        context_menu.add_command(
            label='删除拾取',
            command=lambda: self.delete_pick_from_menu(trace_idx, pick_word)
        )
        
        context_menu.add_command(
            label='编辑时间',
            command=lambda: self.edit_pick_time(trace_idx, pick_word)
        )
        
        context_menu.add_separator()
        
        context_menu.add_command(
            label=f'切换到此拾取字 ({pick_word})',
            command=lambda: self.switch_to_pick_word(pick_word)
        )
        
        # 显示菜单（在鼠标位置）
        # matplotlib MouseEvent 使用 x, y（像素坐标），需要转换为屏幕坐标
        try:
            # 获取 canvas 在屏幕上的位置
            canvas_widget = self.canvas.get_tk_widget()
            canvas_x = canvas_widget.winfo_rootx()
            canvas_y = canvas_widget.winfo_rooty()
            
            # event.x 和 event.y 是相对于 canvas 的像素坐标
            # 转换为屏幕坐标
            screen_x = canvas_x + event.x
            screen_y = canvas_y + event.y
            
            context_menu.tk_popup(screen_x, screen_y)
        except Exception as e:
            # 如果转换失败，使用 canvas 中心位置
            try:
                canvas_widget = self.canvas.get_tk_widget()
                canvas_x = canvas_widget.winfo_rootx() + canvas_widget.winfo_width() // 2
                canvas_y = canvas_widget.winfo_rooty() + canvas_widget.winfo_height() // 2
                context_menu.tk_popup(canvas_x, canvas_y)
            except:
                # 最后的备选方案：使用根窗口中心
                context_menu.tk_popup(self.root.winfo_pointerx(), self.root.winfo_pointery())
        finally:
            context_menu.grab_release()
    
    def delete_pick_from_menu(self, trace_idx: int, pick_word: int):
        """从菜单删除拾取点"""
        if self.pick_manager.remove_pick(trace_idx, pick_word):
            self.num_picks = self.pick_manager.get_statistics()['total_picks']
            self.update_status(f'拾取已删除: 道{trace_idx+1}, 拾取字{pick_word}')
            self._refresh_pick_overlay_only()
    
    def edit_pick_time(self, trace_idx: int, pick_word: int):
        """编辑拾取时间"""
        current_time = self.pick_manager.get_pick(trace_idx, pick_word)
        if current_time is None:
            return
        
        from tkinter.simpledialog import askfloat
        
        new_time = askfloat(
            '编辑拾取时间',
            f'道 {trace_idx+1}, 拾取字 {pick_word}\n当前时间: {current_time:.3f} s\n\n请输入新时间:',
            initialvalue=current_time
        )
        
        if new_time is not None and new_time > 0:
            if self.pick_manager.add_pick(trace_idx, new_time, pick_word):
                self.num_picks = self.pick_manager.get_statistics()['total_picks']
                self.update_status(f'拾取时间已更新: 道{trace_idx+1}, 拾取字{pick_word}, 时间{new_time:.3f}s')
                self._refresh_pick_overlay_only()
    
    def switch_to_pick_word(self, pick_word: int):
        """切换到指定的拾取字"""
        if not self.params:
            return
        
        if 1 <= pick_word <= self.pick_manager.npick:
            self.params.apick = pick_word
            self.update_status(f'已切换到拾取字: {pick_word}')
            self._refresh_pick_overlay_only()
    
    def on_mouse_motion(self, event):
        """鼠标移动事件处理（用于记录鼠标位置）"""
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            self.mouse_x = event.xdata
            self.mouse_y = event.ydata
            # 非拾取模式下，实时显示当前鼠标所在道信息（与拾取模式定位信息一致）
            if self.data_loaded and not self.pick_mode:
                trace_info = self._get_trace_info_at_cursor()
                if trace_info is not None:
                    trace_idx = int(trace_info['trace_idx'])
                    cursor_time = float(trace_info['cursor_time'])
                    # 降低状态栏刷新频率，避免鼠标移动时过度抖动
                    time_bin = round(cursor_time, 2)
                    if self._last_hover_trace_idx != trace_idx or self._last_hover_time_bin != time_bin:
                        self.update_status(self._format_brief_trace_info(trace_info))
                        self._last_hover_trace_idx = trace_idx
                        self._last_hover_time_bin = time_bin

        # 左键拖动平移（非拾取模式）
        if self._is_drag_panning and self._drag_start_data and event.inaxes == self.ax:
            if event.xdata is None or event.ydata is None:
                return
            x0, y0, xlim0, ylim0 = self._drag_start_data
            dx = event.xdata - x0
            dy = event.ydata - y0
            try:
                self.ax.set_xlim(xlim0[0] - dx, xlim0[1] - dx)
                self.ax.set_ylim(ylim0[0] - dy, ylim0[1] - dy)
                # 立即同步窗口参数，确保过滤逻辑跟随视窗（可越过初始±50km）
                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()
                self.user_xlim = xlim
                self.user_ylim = ylim
                self.preserve_zoom = True
                if self.params is not None:
                    self.params.xmin = min(xlim[0], xlim[1])
                    self.params.xmax = max(xlim[0], xlim[1])
                    self.params.tmin = min(ylim[0], ylim[1])
                    self.params.tmax = max(ylim[0], ylim[1])
                self.canvas.draw_idle()
            except Exception:
                pass

    def on_plot_mouse_press(self, event):
        """画布鼠标按下：启用左键拖动平移（非拾取模式）。"""
        if event.inaxes != self.ax:
            return
        if self.pick_mode:
            return
        # 非拾取模式下，复用拾取模式的“点击定位”能力：
        # 右键/中键直接显示当前点击位置对应的道信息。
        if event.button in (2, 3):
            if event.xdata is None or event.ydata is None:
                return
            self.mouse_x = event.xdata
            self.mouse_y = event.ydata
            self.show_trace_info()
            return
        if event.button != 1:
            return
        # 若 matplotlib 工具栏处于 pan/zoom 模式，避免冲突
        try:
            if self.toolbar and getattr(self.toolbar, 'mode', ''):
                return
        except Exception:
            pass
        if event.xdata is None or event.ydata is None:
            return
        self._is_drag_panning = True
        self._drag_start_data = (event.xdata, event.ydata, self.ax.get_xlim(), self.ax.get_ylim())

    def on_plot_mouse_release(self, event):
        """画布鼠标释放：结束拖动平移。"""
        if not self._is_drag_panning:
            return
        self._is_drag_panning = False
        self._drag_start_data = None
        # 拖动结束后，立即切到非交互态并触发一次高质量补绘（含阴影完整填充）
        self._viewport_interacting = False
        self._schedule_viewport_refill()

    def on_plot_scroll(self, event):
        """滚轮缩放：以鼠标位置为中心缩放视窗。"""
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self.pick_mode:
            return

        # matplotlib: event.button 为 'up'/'down'
        zoom_in = (getattr(event, 'button', None) == 'up') or (getattr(event, 'step', 0) > 0)
        factor = 0.85 if zoom_in else 1.15

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_center = event.xdata
        y_center = event.ydata

        x_range = (xlim[1] - xlim[0]) * factor
        y_range = (ylim[1] - ylim[0]) * factor

        new_x_min = x_center - x_range / 2
        new_x_max = x_center + x_range / 2
        new_y_min = y_center - y_range / 2
        new_y_max = y_center + y_range / 2

        try:
            self.ax.set_xlim(new_x_min, new_x_max)
            self.ax.set_ylim(new_y_min, new_y_max)
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            self.user_xlim = xlim
            self.user_ylim = ylim
            self.preserve_zoom = True
            if self.params is not None:
                self.params.xmin = min(xlim[0], xlim[1])
                self.params.xmax = max(xlim[0], xlim[1])
                self.params.tmin = min(ylim[0], ylim[1])
                self.params.tmax = max(ylim[0], ylim[1])
            self.canvas.draw_idle()
        except Exception:
            pass
    
    def on_delete_range_key(self):
        """处理 'd' 键：批量删除拾取点
        
        根据原始 zplot 代码：
        第一次按 d：记录鼠标位置的 x 坐标（offset）作为起始范围
        第二次按 d：记录鼠标位置的 x 坐标作为结束范围，然后删除范围内的拾取点
        如果上一次按键不是 'd' 或状态为3，则重置状态
        """
        if not self.pick_mode or not self.pick_manager or not self.data_loaded:
            self.update_status('提示：请先进入拾取模式并加载数据')
            return
        
        if self.mouse_x is None:
            self.update_status('提示：请将鼠标移动到绘图区域')
            return
        
        # 检查是否是连续的 'd' 键（根据原始代码逻辑）
        if self.last_key_class != 'd' or self.delete_range_state == 2:
            self.delete_range_state = 0
        
        if self.delete_range_state == 0:
            # 第一次按 'd'：记录起始位置
            self.delete_range_x1 = self.mouse_x
            self.delete_range_state = 1
            self.update_status(f'批量删除：已记录起始位置 x={self.delete_range_x1:.2f}，请再次按 d 键记录结束位置')
        elif self.delete_range_state == 1:
            # 第二次按 'd'：记录结束位置并执行删除
            self.delete_range_x2 = self.mouse_x
            
            # 确保 x1 < x2（处理反向范围）
            x1 = min(self.delete_range_x1, self.delete_range_x2)
            x2 = max(self.delete_range_x1, self.delete_range_x2)
            
            # 获取活动拾取字
            pick_word = self.params.apick if self.params else 1
            
            # 获取数据
            traces = self.loaded_data['traces']
            offsets = self.loaded_data['offsets']
            times = self.loaded_data['times']
            trace_headers = self.loaded_data.get('trace_headers', [])
            
            # 应用与绘图时相同的数据类型过滤，只删除当前显示的数据类型的拾取点
            # 使用 plot_manager 的过滤逻辑
            filtered_data = self.plot_manager._filter_data(
                traces, offsets, times, self.params, trace_headers
            )
            filtered_indices = filtered_data.get('indices', list(range(len(traces))))
            filtered_offsets = filtered_data['offsets']
            
            deleted_count = 0
            
            # 只在过滤后的trace中删除拾取点（只作用于用户选择的数据类型）
            for i, trace_idx in enumerate(filtered_indices):
                if trace_idx >= len(offsets):
                    continue
                
                offset = offsets[trace_idx]
                # 检查是否在范围内（支持正向和反向范围）
                if (x1 <= offset <= x2) or (x2 <= offset <= x1):
                    # 删除该道的活动拾取字（只删除当前数据类型过滤后的trace）
                    if self.pick_manager.remove_pick(trace_idx, pick_word):
                        deleted_count += 1
            
            # 重置状态（设置为2，下次按d时会重置为0）
            self.delete_range_state = 2
            self.delete_range_x1 = None
            self.delete_range_x2 = None
            
            # 更新显示
            self.num_picks = self.pick_manager.get_statistics()['total_picks']
            self.update_status(f'批量删除完成：删除了 {deleted_count} 个拾取点（范围: {x1:.2f} - {x2:.2f}）')
            self.request_plot_refresh(delay_ms=20)
        
        self.last_key_class = 'd'
    
    def _get_trace_info_at_cursor(self):
        """获取当前鼠标所在道信息（支持过滤后索引映射）。"""
        if not self.data_loaded or self.loaded_data is None:
            return None
        if self.mouse_x is None or self.mouse_y is None:
            return None

        offsets = self.loaded_data.get('offsets', [])
        if len(offsets) == 0:
            return None

        traces = self.loaded_data.get('traces', [])
        times = self.loaded_data.get('times', [])
        trace_headers = self.loaded_data.get('trace_headers', [])
        mouse_offset = float(self.mouse_x)

        # 优先复用当前绘图缓存，避免鼠标移动触发重复全量过滤
        filtered_indices = getattr(self.plot_manager, 'current_filtered_indices', None) if self.plot_manager else None
        cached_x = getattr(self.plot_manager, 'current_x_coordinates', None) if self.plot_manager else None
        if filtered_indices is None or cached_x is None or len(filtered_indices) == 0:
            if self.plot_manager and trace_headers:
                filtered_data = self.plot_manager._filter_data(
                    traces, offsets, times, self.params, trace_headers
                )
                filtered_indices = filtered_data.get('indices', list(range(len(offsets))))
                filtered_offsets = filtered_data.get('offsets', offsets)
            else:
                filtered_indices = list(range(len(offsets)))
                filtered_offsets = offsets
        else:
            filtered_offsets = np.asarray(cached_x)

        if len(filtered_offsets) == 0:
            return None

        closest_filtered_idx = int(np.argmin(np.abs(np.asarray(filtered_offsets) - mouse_offset)))
        if closest_filtered_idx < len(filtered_indices):
            trace_idx = int(filtered_indices[closest_filtered_idx])
        else:
            trace_idx = int(closest_filtered_idx if closest_filtered_idx < len(offsets) else 0)

        trace_header = None
        if trace_headers and 0 <= trace_idx < len(trace_headers):
            trace_header = trace_headers[trace_idx]

        offset = float(offsets[trace_idx]) if 0 <= trace_idx < len(offsets) else 0.0
        cursor_time = float(self.mouse_y)
        return {
            'trace_idx': trace_idx,
            'trace_header': trace_header,
            'offset': offset,
            'cursor_time': cursor_time,
            'filtered_indices': filtered_indices
        }

    def _format_brief_trace_info(self, trace_info: Dict) -> str:
        """格式化状态栏简要道信息。"""
        trace_idx = int(trace_info['trace_idx'])
        trace_header = trace_info.get('trace_header', None)
        offset = float(trace_info['offset'])
        cursor_time = float(trace_info['cursor_time'])
        brief_info = f"道 {trace_idx}: 炮检距={offset:.3f}km"
        if trace_header and hasattr(trace_header, 'ishoti'):
            brief_info += f", 炮站={trace_header.ishoti}"
        if trace_header and hasattr(trace_header, 'ireci'):
            brief_info += f", 接收站={trace_header.ireci}"
        brief_info += f", 时间={cursor_time:.3f}s"
        return brief_info

    def show_trace_info(self):
        """显示鼠标所在道的关键信息（i键功能）"""
        if not self.data_loaded:
            self.update_status('提示：请先加载数据')
            return
        if self.mouse_x is None or self.mouse_y is None:
            self.update_status('提示：请将鼠标移动到绘图区域')
            return

        trace_info = self._get_trace_info_at_cursor()
        if trace_info is None:
            self.update_status('提示：没有可用的道数据')
            return

        trace_idx = int(trace_info['trace_idx'])
        trace_header = trace_info['trace_header']
        offset = float(trace_info['offset'])
        cursor_time = float(trace_info['cursor_time'])
        filtered_indices = trace_info.get('filtered_indices', [])

        offsets = self.loaded_data.get('offsets', [])
        
        # 构建信息字符串
        info_lines = []
        info_lines.append("=" * 80)
        info_lines.append(f"道信息 (Trace {trace_idx})")
        info_lines.append("=" * 80)
        
        # Shot station/number - 炮站号
        if trace_header:
            shot_station = trace_header.ishoti if hasattr(trace_header, 'ishoti') else 'N/A'
            info_lines.append(f"炮站号 (Shot station/number): {shot_station}")
        else:
            info_lines.append(f"炮站号 (Shot station/number): N/A")
        
        # Dead trace flag - 死道标志
        if trace_header:
            dead_flag = "是" if (hasattr(trace_header, 'iflagi') and trace_header.iflagi != 1) else "否"
            info_lines.append(f"死道标志 (Dead trace flag): {dead_flag}")
        else:
            info_lines.append(f"死道标志 (Dead trace flag): N/A")
        
        # Trace sequential number - 道序号
        if trace_header:
            trace_seq = trace_header.itsn if hasattr(trace_header, 'itsn') else trace_idx
            info_lines.append(f"道序号 (Trace sequential number): {trace_seq}")
        else:
            info_lines.append(f"道序号 (Trace sequential number): {trace_idx}")
        
        # Offset - 炮检距
        offset = offsets[trace_idx] if trace_idx < len(offsets) else 0.0
        info_lines.append(f"炮检距 (Offset): {offset:.4f} km")
        
        # 显示过滤后的索引（如果应用了数据类型过滤）
        if self.plot_manager and len(filtered_indices) < len(offsets):
            filtered_idx_in_display = filtered_indices.index(trace_idx) if trace_idx in filtered_indices else -1
            if filtered_idx_in_display >= 0:
                info_lines.append(f"显示索引 (Display index, filtered): {filtered_idx_in_display}")
        
        # Receiver station/number - 接收站号
        if trace_header:
            receiver_station = trace_header.ireci if hasattr(trace_header, 'ireci') else 'N/A'
            info_lines.append(f"接收站号 (Receiver station/number): {receiver_station}")
        else:
            info_lines.append(f"接收站号 (Receiver station/number): N/A")
        
        # Model position - 模型位置
        records = self.loaded_data.get('records', [])
        model_pos = "N/A"
        if records and trace_header:
            # 尝试根据 shot station 找到对应的记录
            shot_station = trace_header.ishoti if hasattr(trace_header, 'ishoti') else None
            if shot_station is not None:
                for record in records:
                    if record.ishnum == shot_station:
                        model_pos = f"({record.xmod:.2f}, {record.ymod:.2f})"
                        break
        
        info_lines.append(f"模型位置 (Model position): {model_pos}")
        
        # Azimuth - 方位角
        if trace_header:
            azimuth = trace_header.azi if hasattr(trace_header, 'azi') else 'N/A'
            if isinstance(azimuth, (int, float)):
                info_lines.append(f"方位角 (Azimuth): {azimuth:.2f}°")
            else:
                info_lines.append(f"方位角 (Azimuth): {azimuth}")
        else:
            info_lines.append(f"方位角 (Azimuth): N/A")
        
        # Pick times - 拾取时间
        if trace_header and hasattr(trace_header, 'picks') and trace_header.picks:
            picks_str = ", ".join([f"{p:.4f}" for p in trace_header.picks if p > 0])
            if picks_str:
                info_lines.append(f"拾取时间 (Pick times): {picks_str} s")
            else:
                info_lines.append(f"拾取时间 (Pick times): 无")
        else:
            # 从 pick_manager 获取拾取时间
            if self.pick_manager:
                pick_times = []
                for pick_word in range(1, 10):  # 检查所有拾取字
                    pick_time = self.pick_manager.get_pick(trace_idx, pick_word)
                    if pick_time is not None and pick_time > 0:
                        pick_times.append(f"拾取字{pick_word}={pick_time:.4f}")
                if pick_times:
                    info_lines.append(f"拾取时间 (Pick times): {', '.join(pick_times)} s")
                else:
                    info_lines.append(f"拾取时间 (Pick times): 无")
            else:
                info_lines.append(f"拾取时间 (Pick times): 无")
        
        # Cursor time - 光标时间
        info_lines.append(f"光标时间 (Cursor time): {cursor_time:.4f} s")
        
        info_lines.append("=" * 80)
        
        # 打印信息
        info_text = "\n".join(info_lines)
        print("\n" + info_text + "\n")
        
        # 同时在状态栏显示简要信息（中文）
        self.update_status(self._format_brief_trace_info(trace_info))
    
    def on_zoom_in_key(self):
        """处理 'z' 键：以鼠标位置为中心放大
        
        根据原始 zplot 代码：缩小到原来的 50%（0.25 + 0.25 = 0.5）
        但用户要求放大30%，所以缩小到70%（相当于放大43%）
        """
        if not self.ax or self.mouse_x is None or self.mouse_y is None:
            self.update_status('提示：请将鼠标移动到绘图区域')
            return
        
        # 获取当前坐标轴范围
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        x_range = xlim[1] - xlim[0]
        y_range = abs(ylim[1] - ylim[0])  # Y轴可能反转
        
        # 根据原始代码：缩小到原来的 50%（0.25 + 0.25）
        # 但用户要求放大30%，所以使用 0.7 作为缩放因子
        zoom_factor = 0.7  # 缩小到70%，相当于放大约43%
        new_x_range = x_range * zoom_factor
        new_y_range = y_range * zoom_factor
        
        # 计算新的边界（以鼠标位置为中心）
        x_center = self.mouse_x
        y_center = self.mouse_y
        
        new_x_min = x_center - new_x_range / 2
        new_x_max = x_center + new_x_range / 2
        new_y_min = y_center - new_y_range / 2
        new_y_max = y_center + new_y_range / 2
        
        # 设置新的坐标轴范围
        self.ax.set_xlim(new_x_min, new_x_max)
        # Y轴反转（时间向下）
        if ylim[0] > ylim[1]:  # 反转的Y轴
            self.ax.set_ylim(new_y_max, new_y_min)
        else:
            self.ax.set_ylim(new_y_min, new_y_max)
        
        # 保存用户设置的缩放范围
        self.user_xlim = (new_x_min, new_x_max)
        self.user_ylim = (new_y_max, new_y_min) if ylim[0] > ylim[1] else (new_y_min, new_y_max)
        self.preserve_zoom = True
        
        # 更新参数对象中的坐标范围
        if self.params:
            self.params.xmin = new_x_min
            self.params.xmax = new_x_max
            # Y轴范围：确保tmin < tmax（即使Y轴反转）
            if ylim[0] > ylim[1]:  # 反转的Y轴
                self.params.tmin = new_y_min
                self.params.tmax = new_y_max
            else:
                self.params.tmin = new_y_min
                self.params.tmax = new_y_max
        
        # 更新工具栏控件显示
        if hasattr(self, 'top_toolbar') and self.top_toolbar:
            self.top_toolbar.update_widgets_from_params()
        
        # 更新显示
        self.canvas.draw()
        self.update_status(f'已放大：以 ({x_center:.2f}, {y_center:.2f}) 为中心')
        self.last_key_class = 'z'
    
    def on_zoom_out_key(self):
        """处理 'o' 键：以鼠标位置为中心缩小
        
        缩小倍数：30%（放大到原来的130%），与 'z' 键相反
        """
        if not self.ax or self.mouse_x is None or self.mouse_y is None:
            self.update_status('提示：请将鼠标移动到绘图区域')
            return
        
        # 获取当前坐标轴范围
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        x_range = xlim[1] - xlim[0]
        y_range = abs(ylim[1] - ylim[0])  # Y轴可能反转
        
        # 计算新的范围（以鼠标位置为中心，放大到130%）
        # 与 'z' 键相反：如果 'z' 是 0.7，那么 'o' 应该是 1/0.7 ≈ 1.43
        # 但用户要求30%，所以使用 1.3
        zoom_factor = 1.3  # 放大到130%
        new_x_range = x_range * zoom_factor
        new_y_range = y_range * zoom_factor
        
        # 计算新的边界（以鼠标位置为中心）
        x_center = self.mouse_x
        y_center = self.mouse_y
        
        new_x_min = x_center - new_x_range / 2
        new_x_max = x_center + new_x_range / 2
        new_y_min = y_center - new_y_range / 2
        new_y_max = y_center + new_y_range / 2
        
        # 设置新的坐标轴范围
        self.ax.set_xlim(new_x_min, new_x_max)
        # Y轴反转（时间向下）
        if ylim[0] > ylim[1]:  # 反转的Y轴
            self.ax.set_ylim(new_y_max, new_y_min)
        else:
            self.ax.set_ylim(new_y_min, new_y_max)
        
        # 保存用户设置的缩放范围
        self.user_xlim = (new_x_min, new_x_max)
        self.user_ylim = (new_y_max, new_y_min) if ylim[0] > ylim[1] else (new_y_min, new_y_max)
        self.preserve_zoom = True
        
        # 更新参数对象中的坐标范围（设置标志防止事件触发）
        self._updating_params_from_zoom = True
        try:
            if self.params:
                self.params.xmin = new_x_min
                self.params.xmax = new_x_max
                # Y轴范围：确保tmin < tmax（即使Y轴反转）
                if ylim[0] > ylim[1]:  # 反转的Y轴
                    self.params.tmin = new_y_min
                    self.params.tmax = new_y_max
                else:
                    self.params.tmin = new_y_min
                    self.params.tmax = new_y_max
            
            # 更新工具栏控件显示
            if hasattr(self, 'top_toolbar') and self.top_toolbar:
                self.top_toolbar.update_widgets_from_params()
        finally:
            self._updating_params_from_zoom = False
        
        # 更新显示
        self.canvas.draw()
        self.update_status(f'已缩小：以 ({x_center:.2f}, {y_center:.2f}) 为中心')
        self.last_key_class = 'o'
    
    def on_axes_lims_changed(self, ax):
        """处理坐标轴范围变化事件（matplotlib工具栏缩放时触发）
        
        Args:
            ax: 触发事件的坐标轴对象
        """
        # 防止递归更新
        if self._updating_params_from_zoom:
            return
        if self._internal_axes_update:
            return
        
        if not self.ax or ax != self.ax:
            return

        # 标记“视窗交互中”：用于阴影模式下采用快速填充策略
        self._viewport_interacting = True
        if self._viewport_interaction_after_id is not None:
            try:
                self.root.after_cancel(self._viewport_interaction_after_id)
            except Exception:
                pass
            self._viewport_interaction_after_id = None
        
        # 获取当前坐标轴范围
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # 设置标志，防止递归
        self._updating_params_from_zoom = True
        
        try:
            # 更新参数对象中的坐标范围
            if self.params:
                self.params.xmin = xlim[0]
                self.params.xmax = xlim[1]
                # Y轴范围：确保tmin < tmax（即使Y轴反转）
                if ylim[0] > ylim[1]:  # 反转的Y轴
                    self.params.tmin = ylim[1]
                    self.params.tmax = ylim[0]
                else:
                    self.params.tmin = ylim[0]
                    self.params.tmax = ylim[1]
            
            # 更新用户缩放状态
            self.user_xlim = xlim
            self.user_ylim = ylim
            self.preserve_zoom = True
            
            # 更新工具栏控件显示
            if hasattr(self, 'top_toolbar') and self.top_toolbar:
                self.top_toolbar.update_widgets_from_params()
        finally:
            # 重置标志
            self._updating_params_from_zoom = False

        # 视窗变化后，阴影模式下触发一次防抖重绘，补齐当前可见区域填充
        self._schedule_viewport_refill()

        # 一段时间无新的视窗变化后，退出交互态并再触发一次高质量补绘
        def _end_viewport_interaction():
            self._viewport_interaction_after_id = None
            self._viewport_interacting = False
            # 无论是否阴影模式，都在交互结束后再补绘一次，
            # 以确保窗口内波形（wiggle/variable-area）完整更新。
            if self.data_loaded and self.params:
                self._schedule_viewport_refill()

        self._viewport_interaction_after_id = self.root.after(180, _end_viewport_interaction)

    def _schedule_viewport_refill(self):
        """在用户平移/缩放结束后，按新视窗防抖重绘。

        说明：
        - 之前只在阴影模式触发，导致 wiggle 模式平移后不会补画新窗口道。
        - 现在扩展到所有显示模式，保证平移/缩放后都能显示新窗口内波形。
        """
        if self._suspend_viewport_refill:
            # 若当前正处于一次补绘中，记录“有新视窗变化”，
            # 结束后立即再补一次，确保阴影填充跟上最终窗口。
            self._pending_viewport_refill = True
            return
        if not self.data_loaded or not self.params:
            return

        if self._viewport_refill_after_id is not None:
            try:
                self.root.after_cancel(self._viewport_refill_after_id)
            except Exception:
                pass
            self._viewport_refill_after_id = None

        def _do_refill():
            self._viewport_refill_after_id = None
            if not self.data_loaded or not self.params:
                return
            self._suspend_viewport_refill = True
            self._pending_viewport_refill = False
            try:
                self._delayed_update_plot()
            finally:
                def _release_refill_lock():
                    self._suspend_viewport_refill = False
                    if self._pending_viewport_refill:
                        # 若补绘期间又发生视窗变化，立刻再调度一次
                        self._schedule_viewport_refill()

                self.root.after(120, _release_refill_lock)

        self._viewport_refill_after_id = self.root.after(80, _do_refill)
    
    def on_keyboard_pick(self, keyval: int):
        """处理键盘数字键拾取（模拟鼠标按键）
        
        Args:
            keyval: 1=左键, 2=中键, 3=右键
        """
        if not self.pick_mode or not self.data_loaded:
            return
        
        # 获取鼠标当前位置（在绘图区域中心）
        if self.ax:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2
            
            # 创建模拟事件
            class MockEvent:
                def __init__(self, xdata, ydata, button):
                    self.xdata = xdata
                    self.ydata = ydata
                    self.button = button
                    self.inaxes = self.ax if hasattr(self, 'ax') else None
            
            mock_event = MockEvent(x_center, y_center, keyval)
            mock_event.ax = self.ax
            mock_event.inaxes = self.ax
            
            self.on_pick_click(mock_event)
    
    def switch_pick_word(self, direction: int):
        """切换拾取字
        
        Args:
            direction: -1=上一个, 1=下一个
        """
        if not self.params or not self.top_toolbar:
            return
        
        current_apick = self.params.apick
        npick = self.pick_manager.npick if self.pick_manager else 10
        
        # 切换拾取字
        new_apick = current_apick + direction
        
        # 循环
        if new_apick < 1:
            new_apick = npick
        elif new_apick > npick:
            new_apick = 1
        
        # 更新参数
        self.params.apick = new_apick
        
        # 更新工具栏显示
        if hasattr(self.top_toolbar, 'update_pick_word_display'):
            self.top_toolbar.update_pick_word_display()
        
        self.update_status(f'当前拾取字: {new_apick}')
        # 优先仅刷新拾取层，避免切换拾取字触发整幅波形重绘
        try:
            if self.data_loaded and self.pick_manager:
                self._refresh_pick_overlay_only()
            else:
                self.request_plot_refresh(delay_ms=20)
        except Exception:
            # 回退到全量重绘，保证稳健性
            self.request_plot_refresh(delay_ms=20)
    
    def delete_current_pick(self):
        """删除当前鼠标位置下的拾取点"""
        if not self.pick_mode or not self.data_loaded:
            return
        
        # 获取鼠标当前位置（需要从 matplotlib 获取）
        # 由于无法直接获取鼠标位置，这个功能需要通过鼠标点击实现
        # 这里提供一个占位实现
        self.update_status('提示：请使用左键点击拾取点来删除')
    
    def align_picks(self):
        """对齐拾取功能
        
        按 a 键时，以鼠标位置时间为基准，将当前拾取字代表的走时对齐到基准时间（显示为一条直线），
        相应地，各个道根据拾取字走时的挪动时间，进行上下平移挪动。
        对齐功能只用于检查振幅是否一致，不改变原来的拾取走时和各道振幅时间。
        再次按 a 键，取消对齐，恢复原状。
        """
        # 如果已经对齐，则取消对齐
        if self.alignment_active:
            self.alignment_offsets.clear()
            self.aligned_trace_indices.clear()
            self.alignment_active = False
            # 清除对齐映射
            if self.plot_manager:
                self.plot_manager.aligned_trace_to_position.clear()
            self.update_status('对齐已取消，恢复原状')
            self.request_plot_refresh(delay_ms=20)
            return
        
        if not self.data_loaded or not self.pick_manager or not self.params:
            self.update_status('提示：请先加载数据并进入拾取模式')
            return
        
        if self.mouse_x is None or self.mouse_y is None:
            self.update_status('提示：请将鼠标移动到绘图区域')
            return
        
        # 获取数据
        traces = self.loaded_data['traces']
        offsets = self.loaded_data['offsets']
        times = self.loaded_data['times']
        trace_headers = self.loaded_data.get('trace_headers', [])
        
        # 应用与绘图时相同的过滤逻辑，获取实际显示的 offsets
        filtered_data = self.plot_manager._filter_data(
            traces, offsets, times, self.params, trace_headers
        )
        filtered_offsets = filtered_data['offsets']
        filtered_indices = filtered_data['indices']  # 原始索引映射
        
        # 在过滤后的 offsets 中找到最近的道
        if len(filtered_offsets) == 0:
            self.update_status('错误: 没有可用的道数据')
            return
        
        filtered_idx = np.argmin(np.abs(filtered_offsets - self.mouse_x))
        
        # 获取对应的原始 trace_idx（基准道）
        if filtered_idx < len(filtered_indices):
            reference_trace_idx = filtered_indices[filtered_idx]
        else:
            # 如果索引映射失败，回退到原始方法
            reference_trace_idx = np.argmin(np.abs(offsets - self.mouse_x))
        
        # 获取当前拾取字
        pick_word = self.params.apick if self.params else 1
        
        # 基准时间：鼠标位置的时间
        # 注意：这里使用显示时间（已经考虑了折合时间），因为对齐是在显示层面进行的
        reference_offset = offsets[reference_trace_idx] if reference_trace_idx < len(offsets) else filtered_offsets[filtered_idx]
        
        # 基准时间：鼠标位置的显示时间（如果使用了折合时间，已经是折合后的时间）
        # 对齐时，我们需要将所有道的拾取走时对齐到这个显示时间
        if self.params.vred > 0:
            # 如果使用了折合时间，鼠标位置的时间是折合后的时间
            # 我们需要将其转换为原始时间，然后计算偏移
            reference_display_time = self.mouse_y  # 折合后的显示时间
            # 转换为原始时间（用于计算偏移）
            reference_time = reference_display_time + abs(reference_offset) / self.params.vred
        else:
            reference_display_time = self.mouse_y
            reference_time = self.mouse_y
        
        # 清空之前的对齐偏移量
        self.alignment_offsets.clear()
        self.aligned_trace_indices.clear()
        
        # 仅遍历当前显示道，减少不必要计算
        aligned_count = 0
        for trace_idx in filtered_indices:
            # 获取该道的当前拾取字走时（原始时间）
            pick_time = self.pick_manager.get_pick(trace_idx, pick_word)
            
            if pick_time is None or pick_time <= 0:
                # 该道没有当前拾取字的拾取点，不进行对齐，也不显示
                continue
            
            # 计算该道拾取走时的显示时间
            offset = offsets[trace_idx] if trace_idx < len(offsets) else 0.0
            if self.params.vred > 0:
                pick_display_time = pick_time - abs(offset) / self.params.vred
            else:
                pick_display_time = pick_time
            
            # 计算该道需要移动的显示时间量（使得拾取走时对齐到基准显示时间）
            # 偏移量 = 基准显示时间 - 当前拾取走时的显示时间
            offset_time = reference_display_time - pick_display_time
            
            # 存储对齐偏移量（这是显示时间的偏移量）
            self.alignment_offsets[trace_idx] = offset_time
            self.aligned_trace_indices.append(trace_idx)  # 记录有拾取点的道
            aligned_count += 1
        
        # 标记对齐已激活
        self.alignment_active = True
        
        # 更新状态
        self.update_status(f'对齐已启用: {aligned_count} 道已对齐到基准时间 {reference_display_time:.3f}s（道{reference_trace_idx+1}），再次按A取消对齐')
        
        # 保存当前的缩放状态（如果用户已经放大）
        # 这样在重绘后可以恢复缩放
        if self.ax:
            current_xlim = self.ax.get_xlim()
            current_ylim = self.ax.get_ylim()
            # 只有在用户已经设置了缩放时才保存
            if not self.preserve_zoom:
                # 检查是否是默认范围（如果不是，说明用户已经缩放）
                # 这里我们总是保存当前范围，以便对齐后恢复
                self.user_xlim = current_xlim
                self.user_ylim = current_ylim
                self.preserve_zoom = True
        
        # 重绘图形以显示对齐效果
        self.request_plot_refresh(delay_ms=20)
    
    def _print_correlation_report(self, correlation_before: Dict, correlation_after: Dict,
                                  original_picks: Dict[int, float], updated_picks: Dict[int, float]):
        """输出相关性报告（包含对齐前后对比）
        
        Args:
            correlation_before: 对齐前的相关性计算结果
            correlation_after: 对齐后的相关性计算结果
            original_picks: 原始拾取时间字典
            updated_picks: 更新后的拾取时间字典
        """
        # 对齐前的统计
        mean_before = correlation_before.get('mean_correlation', 0.0)
        min_before = correlation_before.get('min_correlation', 0.0)
        max_before = correlation_before.get('max_correlation', 0.0)
        std_before = correlation_before.get('std_correlation', 0.0)
        
        # 对齐后的统计
        mean_after = correlation_after.get('mean_correlation', 0.0)
        min_after = correlation_after.get('min_correlation', 0.0)
        max_after = correlation_after.get('max_correlation', 0.0)
        std_after = correlation_after.get('std_correlation', 0.0)
        
        # 改进量
        improvement = mean_after - mean_before
        improvement_pct = (improvement / mean_before * 100) if mean_before > 0 else 0.0
        
        trace_correlations_before = correlation_before.get('trace_correlations', [])
        trace_correlations_after = correlation_after.get('trace_correlations', [])
        valid_trace_indices = correlation_after.get('valid_trace_indices', [])
        
        print("\n" + "="*80)
        print("自适应叠加后波形相关性分析")
        print("="*80)
        # 检查是否应用了数据类型过滤
        itype = self.params.itype if self.params else 0
        if itype != 0:
            itype_names = {
                1: '垂直', 2: '径向', 3: '横向', 4: '水听器',
                -1: '垂直+径向', -2: '径向+横向', 
                -3: '径向+水听器', -4: '垂直+水听器'
            }
            itype_name = itype_names.get(itype, f'itype={itype}')
            print(f"⚠ 注意：相关性分析仅适用于当前选择的数据类型: {itype_name}")
        print(f"窗口长度: {self.adaptive_stacker.stkwl:.3f}秒 (半长度: {self.adaptive_stacker.stkwl/2:.3f}秒)")
        print(f"有效道数: {len(valid_trace_indices)}")
        print("-"*80)
        print("对齐前后对比:")
        print(f"  对齐前 - 平均相关系数: {mean_before:.4f} (范围: [{min_before:.4f}, {max_before:.4f}], 标准差: {std_before:.4f})")
        print(f"  对齐后 - 平均相关系数: {mean_after:.4f} (范围: [{min_after:.4f}, {max_after:.4f}], 标准差: {std_after:.4f})")
        print(f"  改进量: {improvement:+.4f} ({improvement_pct:+.2f}%)")
        
        if improvement > 0:
            print(f"  ✓ 相关性提升，对齐效果良好！")
        elif improvement < -0.01:
            print(f"  ⚠ 相关性下降，可能需要检查对齐参数或数据质量")
        else:
            print(f"  → 相关性基本不变")
        
        print("-"*80)
        
        if len(trace_correlations_after) > 0 and len(valid_trace_indices) == len(trace_correlations_after):
            print("各道对齐后的平均相关系数:")
            for i, trace_idx in enumerate(valid_trace_indices):
                pick_time_before = original_picks.get(trace_idx, 0.0)
                pick_time_after = updated_picks.get(trace_idx, 0.0)
                trace_corr_before = trace_correlations_before[i] if i < len(trace_correlations_before) else 0.0
                trace_corr_after = trace_correlations_after[i] if i < len(trace_correlations_after) else 0.0
                trace_improvement = trace_corr_after - trace_corr_before
                
                print(f"  道 {trace_idx:3d}: "
                      f"对齐前={trace_corr_before:.4f}, "
                      f"对齐后={trace_corr_after:.4f} "
                      f"({trace_improvement:+.4f}) | "
                      f"走时: {pick_time_before:.3f}s -> {pick_time_after:.3f}s")
        
        print("="*80)
        print()
        
        # 同时在状态栏显示简要信息
        status_msg = (
            f'自适应叠加完成 | '
            f'相关性: {mean_before:.3f} -> {mean_after:.3f} '
            f'({improvement:+.3f}, {improvement_pct:+.1f}%)'
        )
        self.update_status(status_msg)
    
    def show_stacking_result(self, result: Dict, updated_count: int, 
                            evaluation: Optional = None):
        """显示叠加结果
        
        Args:
            result: 自适应叠加结果字典
            updated_count: 更新的拾取点数量
            evaluation: 评价结果（可选）
        """
        quality = result.get('quality_metric', 0.0)
        errors = result.get('errors', [])
        
        if errors:
            avg_error = np.mean(errors)
            max_error = np.max(errors)
            min_error = np.min(errors)
            
            status_msg = (
                f'自适应叠加完成 | '
                f'质量指标: {quality:.6f} | '
                f'平均误差: {avg_error*1000:.2f}ms | '
                f'最大误差: {max_error*1000:.2f}ms | '
                f'最小误差: {min_error*1000:.2f}ms | '
                f'更新拾取: {updated_count}个'
            )
        else:
            status_msg = (
                f'自适应叠加完成 | '
                f'质量指标: {quality:.6f} | '
                f'更新拾取: {updated_count}个'
            )
        
        self.update_status(status_msg)
        
        # 评价报告已移除（用户要求）
        # 如果需要查看详细评价，请使用 Shift+F 键查看可视化图表
    
    def show_stacking_evaluation(self):
        """显示自适应叠加评价可视化
        
        创建新窗口显示评价结果的可视化图表
        """
        if not self.last_stacking_result:
            self.update_status('提示：请先执行自适应叠加')
            return
        
        result_data = self.last_stacking_result
        evaluation = self.stacking_evaluator.evaluate(
            original_picks=result_data['original_picks'],
            updated_picks=result_data['updated_picks'],
            time_shifts=result_data['result']['time_shifts'],
            errors=result_data['result']['errors'],
            quality_metric=result_data['result']['quality_metric'],
            traces=result_data['traces'],
            times=result_data['times']
        )
        
        # 创建对比图
        fig1 = self.stacking_evaluator.create_comparison_plot(evaluation)
        
        # 创建偏移可视化图
        fig2 = self.stacking_evaluator.create_shift_visualization(
            evaluation, result_data['offsets']
        )
        
        # 显示图形
        plt.show()
        
        self.update_status('评价可视化已显示')
    
    def on_key_1(self):
        """处理 1 键：拾取模式下用于拾取，非拾取模式下改变X轴类型"""
        if self.pick_mode:
            self.on_keyboard_pick(1)
        else:
            self.change_xaxis_type(-1)  # 炮检距
    
    def on_key_2(self):
        """处理 2 键：拾取模式下用于拾取，非拾取模式下改变X轴类型"""
        if self.pick_mode:
            self.on_keyboard_pick(2)
        else:
            self.change_xaxis_type(-2)  # 模型位置
    
    def on_key_3(self):
        """处理 3 键：拾取模式下用于拾取，非拾取模式下改变X轴类型"""
        if self.pick_mode:
            self.on_keyboard_pick(3)
        else:
            self.change_xaxis_type(-3)  # 方位角
    
    def change_xaxis_type(self, ixaxis: int):
        """改变X轴类型
        
        Args:
            ixaxis: X轴类型 (-1=炮检距, -2=模型位置, -3=方位角, -4=修正方位角, -5=道号)
        """
        if not self.params:
            return
        
        self.params.ixaxis = ixaxis
        
        # 更新工具栏显示（如果工具栏存在）
        if hasattr(self, 'top_toolbar') and self.top_toolbar:
            # 更新工具栏中的X轴类型显示
            xaxis_map = {
                -1: '炮检距',
                -2: '模型位置',
                -3: '方位角',
                -4: '修正方位角',
                -5: '道号'
            }
            xaxis_name = xaxis_map.get(ixaxis, '炮检距')
            if hasattr(self.top_toolbar, 'xaxis_var'):
                self.top_toolbar.xaxis_var.set(xaxis_name)
        
        # 重绘图形
        self.request_plot_refresh(delay_ms=20)
        
        # 更新状态
        xaxis_names = {
            -1: '炮检距',
            -2: '模型位置',
            -3: '方位角',
            -4: '修正方位角',
            -5: '道号'
        }
        self.update_status(f'X轴类型已切换为: {xaxis_names.get(ixaxis, "未知")}')
    
    def move_window_up(self):
        """向上移动显示窗口（u键）"""
        if not self.ax or not self.params:
            return
        
        # 获取当前Y轴范围
        ylim = self.ax.get_ylim()
        
        # 移动距离 = tlinc（时间增量）
        move_distance = self.params.tlinc
        
        # Y轴是反转的（时间向下），所以 ylim[0] > ylim[1]
        # ylim[0] 是最大值（顶部），ylim[1] 是最小值（底部）
        # 向上移动：时间值减小，所以 ylim 的两个值都减小
        # 但保持反转状态：ylim[0] > ylim[1]
        new_y_max = ylim[0] - move_distance  # 顶部时间减小
        new_y_min = ylim[1] - move_distance  # 底部时间减小
        
        # 保持Y轴反转（时间向下）
        self.ax.set_ylim(new_y_max, new_y_min)
        
        # 保存用户设置的缩放范围（保持反转格式）
        self.user_ylim = (new_y_max, new_y_min)
        self.preserve_zoom = True
        
        # 更新显示
        self.canvas.draw()
        self.update_status(f'显示窗口已向上移动 {move_distance:.3f}s')
    
    def move_window_down(self):
        """向下移动显示窗口（v键）"""
        if not self.ax or not self.params:
            return
        
        # 获取当前Y轴范围
        ylim = self.ax.get_ylim()
        
        # 移动距离 = tlinc（时间增量）
        move_distance = self.params.tlinc
        
        # Y轴是反转的（时间向下），所以 ylim[0] > ylim[1]
        # ylim[0] 是最大值（顶部），ylim[1] 是最小值（底部）
        # 向下移动：时间值增大，所以 ylim 的两个值都增大
        # 但保持反转状态：ylim[0] > ylim[1]
        new_y_max = ylim[0] + move_distance  # 顶部时间增大
        new_y_min = ylim[1] + move_distance  # 底部时间增大
        
        # 保持Y轴反转（时间向下）
        self.ax.set_ylim(new_y_max, new_y_min)
        
        # 保存用户设置的缩放范围（保持反转格式）
        self.user_ylim = (new_y_max, new_y_min)
        self.preserve_zoom = True
        
        # 更新显示
        self.canvas.draw()
        self.update_status(f'显示窗口已向下移动 {move_distance:.3f}s')
    
    def move_window_right(self):
        """向右移动显示窗口（r键）"""
        if not self.ax:
            self.update_status('提示：请先加载数据')
            return
        
        # 获取当前X轴范围
        xlim = self.ax.get_xlim()
        x_range = xlim[1] - xlim[0]
        
        if x_range <= 0:
            self.update_status('错误：X轴范围无效')
            return
        
        # 移动距离 = 当前范围的10%
        move_distance = x_range * 0.1
        
        # 向右移动：X轴值增大
        new_x_min = xlim[0] + move_distance
        new_x_max = xlim[1] + move_distance
        
        self.ax.set_xlim(new_x_min, new_x_max)
        
        # 保存用户设置的缩放范围
        self.user_xlim = (new_x_min, new_x_max)
        self.preserve_zoom = True
        
        # 更新显示
        self.canvas.draw()
        self.update_status(f'显示窗口已向右移动: [{new_x_min:.2f}, {new_x_max:.2f}]')
    
    def move_window_left(self):
        """向左移动显示窗口（l键）"""
        if not self.ax:
            self.update_status('提示：请先加载数据')
            return
        
        # 获取当前X轴范围
        xlim = self.ax.get_xlim()
        x_range = xlim[1] - xlim[0]
        
        if x_range <= 0:
            self.update_status('错误：X轴范围无效')
            return
        
        # 移动距离 = 当前范围的10%
        move_distance = x_range * 0.1
        
        # 向左移动：X轴值减小
        new_x_min = xlim[0] - move_distance
        new_x_max = xlim[1] - move_distance
        
        self.ax.set_xlim(new_x_min, new_x_max)
        
        # 保存用户设置的缩放范围
        self.user_xlim = (new_x_min, new_x_max)
        self.preserve_zoom = True
        
        # 更新显示
        self.canvas.draw()
        self.update_status(f'显示窗口已向左移动: [{new_x_min:.2f}, {new_x_max:.2f}]')
    
    def toggle_xaxis_sign(self):
        """改变X轴符号（-键）"""
        if not self.params:
            return
        
        # 改变 ixaxis 的符号
        self.params.ixaxis = -self.params.ixaxis
        
        # 重绘图形
        self.request_plot_refresh(delay_ms=20)
        
        # 更新状态
        self.update_status(f'X轴符号已切换，当前类型: {self.params.ixaxis}')
    
    def expand_left(self):
        """向左扩展显示范围（<键）"""
        if not self.ax:
            self.update_status('提示：请先加载数据')
            return
        
        # 获取当前X轴范围
        xlim = self.ax.get_xlim()
        x_range = xlim[1] - xlim[0]
        
        if x_range <= 0:
            self.update_status('错误：X轴范围无效')
            return
        
        # 扩展10%
        expand_factor = 0.1
        new_x_min = xlim[0] - x_range * expand_factor
        new_x_max = xlim[1]
        
        self.ax.set_xlim(new_x_min, new_x_max)
        
        # 保存用户设置的缩放范围
        self.user_xlim = (new_x_min, new_x_max)
        self.preserve_zoom = True
        
        # 更新显示
        self.canvas.draw()
        self.update_status(f'显示范围已向左扩展: [{new_x_min:.2f}, {new_x_max:.2f}]')
    
    def expand_right(self):
        """向右扩展显示范围（>键）"""
        if not self.ax:
            self.update_status('提示：请先加载数据')
            return
        
        # 获取当前X轴范围
        xlim = self.ax.get_xlim()
        x_range = xlim[1] - xlim[0]
        
        if x_range <= 0:
            self.update_status('错误：X轴范围无效')
            return
        
        # 扩展10%
        expand_factor = 0.1
        new_x_min = xlim[0]
        new_x_max = xlim[1] + x_range * expand_factor
        
        self.ax.set_xlim(new_x_min, new_x_max)
        
        # 保存用户设置的缩放范围
        self.user_xlim = (new_x_min, new_x_max)
        self.preserve_zoom = True
        
        # 更新显示
        self.canvas.draw()
        self.update_status(f'显示范围已向右扩展: [{new_x_min:.2f}, {new_x_max:.2f}]')
    
    def apply_filter(self):
        """应用滤波"""
        messagebox.showinfo('信息', '滤波功能（待实现）')
    
    def apply_gain(self):
        """应用增益"""
        messagebox.showinfo('信息', '增益功能（待实现）')
    
    # ========== 事件处理 ==========
    
    def on_mouse_click(self, event):
        """鼠标点击事件"""
        if not self.pick_mode:
            return
        
        if event.inaxes != self.ax:
            return
        
        # TODO: 实现拾取逻辑
        x_data = event.xdata
        y_data = event.ydata
        
        # 临时显示拾取点
        self.ax.plot(x_data, y_data, 'ro', markersize=8)
        self.canvas.draw()
        
        self.num_picks += 1
        self.update_status()
    
    # ========== 帮助 ==========
    
    def show_help(self):
        """显示帮助"""
        help_text = """
ZPLOT - 地震震相拾取工具

═══════════════════════════════════════════════════════
基本操作
═══════════════════════════════════════════════════════
P 键          - 进入/退出拾取模式
E 键          - 编辑参数
H 键          - 显示帮助（本窗口）
Ctrl+O        - 打开数据文件
Ctrl+R        - 重新加载数据
Ctrl+S        - 保存拾取结果（zplot.out格式）
S 键          - 保存拾取到头文件（.hdr格式）
Q 键 / Ctrl+Q - 退出程序

═══════════════════════════════════════════════════════
记录导航
═══════════════════════════════════════════════════════
← (左箭头)    - 切换到上一个记录
→ (右箭头)    - 切换到下一个记录

═══════════════════════════════════════════════════════
拾取操作
═══════════════════════════════════════════════════════
左键点击      - 添加/更新拾取点
右键点击      - 删除当前拾取字对应的拾取点（在该道上）
1/2/3 键      - 模拟鼠标按键（1=左键, 2=中键, 3=右键，仅在拾取模式下）
, / . 或 [/]  - 切换拾取字（上一个/下一个）
Delete/Backspace - 删除当前拾取点

═══════════════════════════════════════════════════════
自动拾取功能
═══════════════════════════════════════════════════════
C 键          - 插值-相关自动拾取
                * 在相邻拾取点之间自动拾取中间道的走时
                * 需要至少2个拾取点才能执行
                * 使用互相关方法进行自动拾取

A 键          - 对齐拾取
                * 以鼠标位置时间为基准，对齐当前拾取字的走时

F 键          - 自适应叠加对齐
                * 使用自适应叠加算法对齐拾取点
                * 更新拾取时间

Shift+F       - 显示自适应叠加评价可视化
                * 显示对齐前后的相关性分析

═══════════════════════════════════════════════════════
批量操作
═══════════════════════════════════════════════════════
D 键          - 批量删除拾取点
                * 第一次按 D：记录鼠标位置的 x 坐标作为起始范围
                * 第二次按 D：记录鼠标位置的 x 坐标作为结束范围
                * 删除范围内的拾取点（只作用于当前选择的数据类型）

X 键          - 移除/恢复trace
                * 移除或重新绘制距离光标最近的trace（以及相关的picks）
                * 第一次按 X：移除trace，在顶部显示红色圆圈标记
                * 再次按 X：恢复trace，清除标记
                * header word 5 (iflagi) 会被设置为负值（移除时）

I 键          - 显示鼠标所在道的关键信息
                * 显示炮站号、死道标志、道序号、炮检距、接收站号
                * 显示模型位置、方位角、拾取时间、光标时间等信息
                * 信息会打印到控制台，同时在状态栏显示简要信息

═══════════════════════════════════════════════════════
视图操作
═══════════════════════════════════════════════════════
Z 键          - 以鼠标位置为中心放大（30%）
O 键          - 以鼠标位置为中心缩小（30%）
U 键          - 向上移动显示窗口
V 键          - 向下移动显示窗口
R 键          - 向右移动显示窗口（10%）
L 键          - 向左移动显示窗口（10%）
< 键          - 向左扩展显示范围
> 键          - 向右扩展显示范围

═══════════════════════════════════════════════════════
X轴类型切换（非拾取模式下）
═══════════════════════════════════════════════════════
1 键          - 炮检距
2 键          - 模型位置
3 键          - 方位角
4 键          - 修正方位角
5 键          - 道号
- 键          - 改变X轴符号

═══════════════════════════════════════════════════════
提示
═══════════════════════════════════════════════════════
- 更多功能请参考菜单栏
- 参数设置请使用工具栏或按 E 键编辑参数
- 拾取操作需要在拾取模式下进行（按 P 键进入）
        """
        messagebox.showinfo('帮助', help_text)
    
    def show_shortcuts(self):
        """显示快捷键"""
        shortcuts = """
═══════════════════════════════════════════════════════
ZPLOT 键盘快捷键完整列表
═══════════════════════════════════════════════════════

【文件操作】
  Ctrl+O      - 打开数据文件
  Ctrl+R      - 重新加载数据
  Ctrl+S      - 保存拾取结果（zplot.out格式）
  S           - 保存拾取到头文件（.hdr格式）
  Q / Ctrl+Q  - 退出程序

【基本操作】
  P           - 进入/退出拾取模式
  E           - 编辑参数
  H           - 显示帮助
  Ctrl++      - 放大界面字体
  Ctrl+-      - 缩小界面字体
  Ctrl+0      - 重置界面字体

【记录导航】
  ← (左箭头)  - 切换到上一个记录
  → (右箭头)  - 切换到下一个记录

【拾取操作】
  鼠标左键    - 添加/更新拾取点
  鼠标右键    - 删除当前拾取字对应的拾取点
  1/2/3       - 模拟鼠标按键（1=左键, 2=中键, 3=右键，仅在拾取模式下）
  , / .       - 切换拾取字（上一个/下一个）
  [ / ]       - 切换拾取字（上一个/下一个）
  Delete      - 删除当前拾取点
  Backspace   - 删除当前拾取点

【自动拾取功能】
  C           - 插值-相关自动拾取
                * 在相邻拾取点之间自动拾取中间道的走时
                * 需要至少2个拾取点才能执行
  A           - 对齐拾取（以鼠标位置时间为基准）
  F           - 自适应叠加对齐（更新拾取时间）
  Shift+F     - 显示自适应叠加评价可视化

【批量操作】
  D           - 批量删除拾取点
                * 第一次按 D：记录起始范围
                * 第二次按 D：记录结束范围并删除
                * 只作用于当前选择的数据类型
  X           - 移除/恢复trace
                * 移除或重新绘制距离光标最近的trace
                * 第一次按 X：移除trace，显示红色圆圈标记
                * 再次按 X：恢复trace，清除标记

【视图操作】
  Z           - 以鼠标位置为中心放大（30%）
  O           - 以鼠标位置为中心缩小（30%）
  U           - 向上移动显示窗口
  V           - 向下移动显示窗口
  R           - 向右移动显示窗口（10%）
  L           - 向左移动显示窗口（10%）
  <           - 向左扩展显示范围
  >           - 向右扩展显示范围

【X轴类型切换】（非拾取模式下）
  1           - 炮检距（拾取模式下：模拟鼠标左键）
  2           - 模型位置（拾取模式下：模拟鼠标中键）
  3           - 方位角（拾取模式下：模拟鼠标右键）
  4           - 修正方位角
  5           - 道号
  -           - 改变X轴符号

═══════════════════════════════════════════════════════
提示：按 H 键查看详细帮助信息
        """
        messagebox.showinfo('快捷键', shortcuts)
    
    def show_about(self):
        """显示关于"""
        about_text = """
ZPLOT - Python 实现

基于原始 ZPLOT (Colin A. Zelt, 1994)
Python 实现版本 0.1.0

功能：
- 交互式地震震相拾取
- 数据可视化
- 参数控制

开发中...
        """
        messagebox.showinfo('关于', about_text)
    
    def quit(self):
        """退出程序"""
        if messagebox.askyesno('确认', '确定要退出吗？'):
            self.root.quit()
            self.root.destroy()

    # ===== Workflow Controller Delegation =====

    def perform_adaptive_stacking(self):
        return self._workflow_controller.perform_adaptive_stacking(self)

    def auto_pick(self):
        return self._workflow_controller.auto_pick(self)

    def _get_trace_indices_for_auto_pick(self, range_type: str, start_trace: int = None,
                                         end_trace: int = None, trace_headers: List = None):
        return self._workflow_controller.get_trace_indices_for_auto_pick(
            self, range_type, start_trace, end_trace, trace_headers=trace_headers
        )

    def _preview_auto_pick(self, traces: List[np.ndarray], times: np.ndarray, trace_indices: List[int], params: Dict):
        return self._workflow_controller.preview_auto_pick(self, traces, times, trace_indices, params)

    def _apply_auto_pick(self, traces: List[np.ndarray], times: np.ndarray, trace_indices: List[int], params: Dict):
        return self._workflow_controller.apply_auto_pick(self, traces, times, trace_indices, params)

    def _apply_auto_pick_results(self, traces: List[np.ndarray], times: np.ndarray,
                                trace_indices: List[int], results: List[Optional[Dict]], params: Dict):
        return self._workflow_controller.apply_auto_pick_results(self, traces, times, trace_indices, results, params)

    def on_interpolation_correlation_picking(self):
        return self._workflow_controller.on_interpolation_correlation_picking(self)

    def _perform_interpolation_correlation_picking_batch(self, existing_picks: List[Tuple[int, float]], pick_word: int):
        return self._workflow_controller.perform_interpolation_correlation_picking_batch(self, existing_picks, pick_word)

    def calculate_static_correction_dialog(self):
        return self._workflow_controller.calculate_static_correction_dialog(self)

    def calculate_static_correction(self, sigma: float = 3.0, smoothness: float = 0.1):
        return self._workflow_controller.calculate_static_correction(self, sigma=sigma, smoothness=smoothness)

    def on_static_correction_smoothness_changed(self, value, immediate=True):
        return self._workflow_controller.on_static_correction_smoothness_changed(self, value, immediate=immediate)

    def clear_static_correction(self):
        return self._workflow_controller.clear_static_correction(self)
    
    def calculate_theoretical_traveltime_dialog(self):
        """改进的理论走时计算对话框（支持参数修改）"""
        if not self.data_loaded:
            messagebox.showwarning('提示', '请先加载数据')
            return
        
        # 创建对话框
        dialog = tk.Toplevel(self.root)
        dialog.title('计算理论走时')
        dialog.geometry('550x500')
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        
        # 1. 速度模型文件选择
        model_frame = ttk.LabelFrame(dialog, text='速度模型文件')
        model_frame.pack(fill=tk.X, pady=5, padx=10)
        model_file_var = tk.StringVar()
        model_entry = ttk.Entry(model_frame, textvariable=model_file_var, width=50)
        model_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        def browse_model_file():
            filename = filedialog.askopenfilename(
                title='选择速度模型文件',
                filetypes=[
                    ('v.in files', 'v.in'),
                    ('v.in files', '*.vin'),
                    ('All files', '*.*')
                ]
            )
            if filename:
                model_file_var.set(filename)
                # 自动从模型提取X和Z范围
                _extract_model_bounds_to_vars(filename)
        
        # 如果模型文件路径已存在（例如通过其他方式设置），自动提取范围
        def on_model_file_changed(*args):
            model_file = model_file_var.get().strip()
            if model_file and Path(model_file).exists():
                _extract_model_bounds_to_vars(model_file)
        
        # 绑定模型文件路径变化事件
        model_file_var.trace_add('write', on_model_file_changed)
        
        def _extract_model_bounds_to_vars(model_file_path):
            """从模型文件提取X和Z范围并填充到变量"""
            try:
                if not model_file_path or not Path(model_file_path).exists():
                    messagebox.showerror('错误', '模型文件不存在')
                    return
                
                # 使用与文件顶部相同的导入方式
                try:
                    from .theoretical_traveltime import ModelLoader
                except ImportError:
                    from theoretical_traveltime import ModelLoader
                
                # ModelLoader 需要先创建实例，然后调用 load_model 方法
                model_loader = ModelLoader()
                if not model_loader.load_model(model_file_path):
                    messagebox.showerror('错误', '模型加载失败')
                    return
                
                model_info = model_loader.get_model_info()
                model_bounds = model_info.get('bounds')
                
                if model_bounds:
                    xmin_var.set(model_bounds['xmin'])
                    xmax_var.set(model_bounds['xmax'])
                    zmin_var.set(model_bounds['zmin'])
                    zmax_var.set(model_bounds['zmax'])
                    # 显示成功消息
                    self.update_status(f"从模型提取范围: x=[{model_bounds['xmin']:.1f}, {model_bounds['xmax']:.1f}], "
                                     f"z=[{model_bounds['zmin']:.1f}, {model_bounds['zmax']:.1f}]")
                else:
                    messagebox.showwarning('警告', '无法从模型获取边界信息')
            except Exception as e:
                import traceback
                error_msg = f'提取模型范围失败: {e}\n{traceback.format_exc()}'
                messagebox.showerror('错误', error_msg)
        
        ttk.Button(model_frame, text='浏览...', command=browse_model_file).pack(side=tk.LEFT, padx=5)
        
        # 2. 炮点位置输入（可选）
        shot_frame = ttk.LabelFrame(dialog, text='炮点位置（可选）')
        shot_frame.pack(fill=tk.X, padx=10, pady=5)
        xshot_var = tk.DoubleVar(value=0.0)
        zshot_var = tk.DoubleVar(value=0.0)
        ttk.Label(shot_frame, text='X (km):').grid(row=0, column=0, padx=5)
        ttk.Entry(shot_frame, textvariable=xshot_var, width=10).grid(row=0, column=1, padx=5)
        ttk.Label(shot_frame, text='Z (km):').grid(row=0, column=2, padx=5)
        ttk.Entry(shot_frame, textvariable=zshot_var, width=10).grid(row=0, column=3, padx=5)
        ttk.Button(shot_frame, text='从数据提取',
                   command=lambda: self._extract_shot_position_to_vars(xshot_var, zshot_var)
                   ).grid(row=0, column=4, padx=5)
        
        # 3. 射线参数设置（可展开的高级选项）
        ray_frame = ttk.LabelFrame(dialog, text='射线参数（高级选项）')
        ray_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # ray 参数（控制震相类型）
        ray_var = tk.StringVar(value='1.2')  # 默认：第1层反射
        ttk.Label(ray_frame, text='ray= (射线参数，控制震相类型):').grid(row=0, column=0, padx=5, sticky='w')
        ray_entry = ttk.Entry(ray_frame, textvariable=ray_var, width=15)
        ray_entry.grid(row=0, column=1, padx=5)
        help_text = (
            '编码规则：L.1=第L层折射, L.2=第L层反射, L.3=第L层首波\n'
            '示例：1.1 (第1层折射), 1.2 (第1层反射), 1.1,2.1 (多组)'
        )
        ttk.Label(ray_frame, text=help_text,
                  font=('Arial', 8), foreground='gray', justify=tk.LEFT).grid(row=0, column=2, columnspan=3, padx=5, sticky='w')
        
        # nray 参数（射线数量）
        nray_var = tk.IntVar(value=5)
        ttk.Label(ray_frame, text='nray= (射线数量):').grid(row=1, column=0, padx=5, sticky='w')
        ttk.Entry(ray_frame, textvariable=nray_var, width=10).grid(row=1, column=1, padx=5)
        
        # X轴范围（从模型自动提取）- 更紧凑的布局
        xmin_var = tk.DoubleVar()
        xmax_var = tk.DoubleVar()
        ttk.Label(ray_frame, text='X范围 (km):').grid(row=2, column=0, padx=5, sticky='w')
        ttk.Entry(ray_frame, textvariable=xmin_var, width=6).grid(row=2, column=1, padx=1, sticky='w')
        ttk.Label(ray_frame, text='~').grid(row=2, column=2, padx=1)
        ttk.Entry(ray_frame, textvariable=xmax_var, width=6).grid(row=2, column=3, padx=1, sticky='w')
        ttk.Button(ray_frame, text='从模型提取',
                   command=lambda: _extract_model_bounds_to_vars(model_file_var.get())
                   ).grid(row=2, column=4, padx=3, sticky='w')
        
        # Z轴范围（从模型自动提取）- 更紧凑的布局
        zmin_var = tk.DoubleVar()
        zmax_var = tk.DoubleVar()
        ttk.Label(ray_frame, text='Z范围 (km):').grid(row=3, column=0, padx=5, sticky='w')
        ttk.Entry(ray_frame, textvariable=zmin_var, width=6).grid(row=3, column=1, padx=1, sticky='w')
        ttk.Label(ray_frame, text='~').grid(row=3, column=2, padx=1)
        ttk.Entry(ray_frame, textvariable=zmax_var, width=6).grid(row=3, column=3, padx=1, sticky='w')
        ttk.Button(ray_frame, text='从模型提取',
                   command=lambda: _extract_model_bounds_to_vars(model_file_var.get())
                   ).grid(row=3, column=4, padx=3, sticky='w')
        
        # 配置列权重，防止列过度扩展（只让最后一列扩展）
        ray_frame.columnconfigure(0, weight=0, minsize=100)  # 标签列，不扩展
        ray_frame.columnconfigure(1, weight=0, minsize=50)  # 输入框列，不扩展
        ray_frame.columnconfigure(2, weight=0, minsize=15)  # 分隔符列，不扩展
        ray_frame.columnconfigure(3, weight=0, minsize=50)  # 输入框列，不扩展
        ray_frame.columnconfigure(4, weight=0, minsize=90)  # 按钮列，不扩展
        ray_frame.columnconfigure(5, weight=1)   # 空白列，允许扩展以填充空间
        
        # 4. 是否使用观测拾取
        use_picks_var = tk.BooleanVar(value=False)
        if self.pick_manager and self.pick_manager.get_all_picks():
            ttk.Checkbutton(dialog, text='使用观测拾取生成 tx.in',
                           variable=use_picks_var).pack(pady=5)
        
        # 5. 按钮
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def on_ok():
            model_file = model_file_var.get().strip()
            if not model_file:
                messagebox.showerror('错误', '请选择速度模型文件')
                return
            
            if not Path(model_file).exists():
                messagebox.showerror('错误', f'模型文件不存在: {model_file}')
                return
            
            # 收集参数
            shot_position = (xshot_var.get(), zshot_var.get())
            
            # 解析 ray 参数（可能是单个值或多个值，逗号分隔）
            ray_str = ray_var.get().strip()
            try:
                if ',' in ray_str:
                    ray_values = [float(x.strip()) for x in ray_str.split(',')]
                else:
                    ray_values = [float(ray_str)]
            except ValueError:
                messagebox.showerror('错误', f'ray 参数格式错误: {ray_str}\n应为数字，如: 1.1 或 1.1,2.1')
                return
            
            ray_params = {
                'ray': ray_values,  # 数组形式
                'nray': nray_var.get()
            }
            # 如果用户输入了X和Z范围，使用用户输入的值
            # 注意：即使值为0.0，如果用户明确输入了，也应该使用（因为模型边界可能确实是0）
            # 但如果值都是0.0且未修改，则让后端自动从模型提取
            xmin_val = xmin_var.get()
            xmax_val = xmax_var.get()
            zmin_val = zmin_var.get()
            zmax_val = zmax_var.get()
            
            # 检查是否有非零值，或者值已被修改（通过检查是否与初始值不同）
            if (xmin_val != 0.0 or xmax_val != 0.0) or (xmin_val == 0.0 and xmax_val == 0.0 and model_file_var.get()):
                # 如果模型文件已加载，即使值为0也尝试使用（可能是有效的模型边界）
                # 但更安全的做法是：如果用户没有明确输入，让后端自动从模型提取
                # 这里我们检查：如果值都是0.0，且模型文件存在，则让后端自动提取
                if xmin_val == 0.0 and xmax_val == 0.0:
                    # 值未设置，让后端自动从模型提取
                    pass
                else:
                    ray_params['xmin'] = xmin_val
                    ray_params['xmax'] = xmax_val
            
            if (zmin_val != 0.0 or zmax_val != 0.0) or (zmin_val == 0.0 and zmax_val == 0.0 and model_file_var.get()):
                if zmin_val == 0.0 and zmax_val == 0.0:
                    # 值未设置，让后端自动从模型提取
                    pass
                else:
                    ray_params['zmin'] = zmin_val
                    ray_params['zmax'] = zmax_val
            
            dialog.destroy()
            self.calculate_theoretical_traveltime(
                model_file,
                shot_position=shot_position,
                ray_params=ray_params,
                use_observed_picks=use_picks_var.get()
            )
        
        ttk.Button(button_frame, text='确定', command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text='取消', command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # 绑定回车键
        model_entry.bind('<Return>', lambda e: on_ok())
        model_entry.focus()
    
    def _extract_shot_position_to_vars(self, xshot_var, zshot_var):
        """从数据中提取炮点位置并填充到变量"""
        if not self.data_loader:
            return
        
        # 尝试从记录文件提取
        if hasattr(self, 'rfile_path') and self.rfile_path and os.path.exists(self.rfile_path):
            records = self.data_loader.records
            if records and len(records) > 0:
                current_record_idx = getattr(self, 'current_record', 1) - 1
                if 0 <= current_record_idx < len(records):
                    current_record = records[current_record_idx]
                    if hasattr(current_record, 'xmod') and current_record.xmod != 0:
                        xshot_var.set(current_record.xmod)
                        zshot_var.set(0.0)  # 默认深度为0
                        return
        
        # 尝试从道头提取
        if self.data_loader.trace_headers:
            first_header = self.data_loader.trace_headers[0]
            if hasattr(first_header, 'sxutm') and first_header.sxutm != 0:
                xshot_var.set(first_header.sxutm / 1000.0)  # 米转千米
            if hasattr(first_header, 'sz') and first_header.sz != 0:
                zshot_var.set(first_header.sz)
                return
        
        # 如果无法提取，保持默认值 (0.0, 0.0)
    
    def _calculate_x_range_to_vars(self, xmin_var, xmax_var):
        """从偏移距计算X轴范围并填充到变量"""
        if not self.data_loader or not self.data_loader.trace_headers:
            return
        
        offsets = []
        for header in self.data_loader.trace_headers:
            if hasattr(header, 'offsti') and header.offsti is not None and header.offsti > 0:
                offsets.append(header.offsti)
        
        if offsets:
            xmin_var.set(min(offsets) - 10.0)
            xmax_var.set(max(offsets) + 10.0)
        else:
            xmin_var.set(0.0)
            xmax_var.set(100.0)
    
    def calculate_theoretical_traveltime(self, model_file: str,
                                        shot_position: Optional[Tuple[float, float]] = None,
                                        ray_params: Optional[Dict] = None,
                                        use_observed_picks: bool = False):
        """计算理论走时（改进版，支持参数修改和自动生成输入文件）
        
        Args:
            model_file: 速度模型文件路径（v.in格式）
            shot_position: 炮点位置 (x, z)，如果为None则从数据提取或使用默认值
            ray_params: 射线参数字典，可包含 ray, nray, xmin, xmax 等
            use_observed_picks: 是否使用观测拾取生成 tx.in
        """
        try:
            self.update_status('正在加载速度模型...')
            
            # 初始化理论走时计算器（传递 data_loader 和 pick_manager）
            self.theoretical_traveltime_calculator = TheoreticalTravelTimeCalculator(
                model_file_path=model_file,
                data_loader=self.data_loader,
                pick_manager=self.pick_manager
            )
            
            # 检查模型是否加载成功
            model_info = self.theoretical_traveltime_calculator.get_model_info()
            if not model_info['has_model']:
                messagebox.showerror('错误', '模型加载失败')
                return
            
            self.update_status(f'模型加载成功（格式: {model_info["model_type"]}）')
            
            # 检查是否为 v.in 格式（RAYINVR 需要）
            if model_info['model_type'] != 'vin':
                messagebox.showerror('错误', 'RAYINVR 需要 v.in 格式的模型文件')
                return
            
            # 计算理论走时（自动生成输入文件）
            self.update_status('正在运行 RAYINVR 计算理论走时...')
            success = self.theoretical_traveltime_calculator.calculate_travel_times(
                auto_generate_inputs=True,
                shot_position=shot_position,
                ray_params=ray_params,
                use_observed_picks=use_observed_picks,
                pick_word=self.params.apick
            )
            
            if not success:
                messagebox.showerror('错误', '理论走时计算失败，请检查模型文件和 RAYINVR 配置')
                return
            
            # 获取理论走时数据
            self.update_status('正在获取理论走时数据...')
            
            # 获取当前显示的道的偏移距
            if not self.data_loaded or not self.plot_manager:
                messagebox.showerror('错误', '数据未加载')
                return
            
            offsets = self.loaded_data.get('offsets', np.array([]))
            trace_indices = self.plot_manager.current_filtered_indices
            
            if len(offsets) == 0 or len(trace_indices) == 0:
                messagebox.showerror('错误', '无法获取道的偏移距信息')
                return
            
            # 计算X轴坐标（用于理论走时查询）
            trace_headers = self.loaded_data.get('trace_headers', [])
            records = self.loaded_data.get('records', [])
            x_coordinates = self.plot_manager._calculate_x_coordinates(
                offsets, trace_indices, trace_headers, records, self.params
            )
            
            # 获取理论走时（基于X坐标）
            theoretical_data = self.theoretical_traveltime_calculator.get_theoretical_times(x_coordinates)
            
            if theoretical_data is None:
                messagebox.showerror('错误', '无法获取理论走时数据')
                return
            
            # 保存理论走时数据
            self.theoretical_times_data = {
                'distances': theoretical_data['distance'],
                'times': theoretical_data['time'],
                'trace_indices': trace_indices
            }
            self.show_theoretical_times = True
            
            # 更新绘图
            self.request_plot_refresh(immediate=True)
            
            # 显示成功消息
            n_points = len(theoretical_data['distance'])
            self.update_status(f'理论走时计算完成：{n_points} 个数据点')
            messagebox.showinfo('成功', f'理论走时计算完成\n共 {n_points} 个数据点')
            
        except Exception as e:
            import traceback
            error_msg = f'理论走时计算失败: {str(e)}\n{traceback.format_exc()}'
            self.update_status(error_msg)
            messagebox.showerror('错误', error_msg)
    
    def clear_theoretical_traveltime(self):
        """清除理论走时"""
        # ✅ 清除绘制的理论走时曲线
        if self.plot_manager and hasattr(self.plot_manager, 'theoretical_traveltime_lines'):
            for line in list(self.plot_manager.theoretical_traveltime_lines):
                try:
                    if line in self.plot_manager.ax.lines:
                        line.remove()
                except:
                    pass
            self.plot_manager.theoretical_traveltime_lines = []
        
        self.theoretical_traveltime_calculator = None
        self.theoretical_times_data = None
        self.show_theoretical_times = False
        self.update_status('已清除理论走时')
        self.request_plot_refresh(immediate=True)
    
    def calculate_water_layer_correction_dialog(self):
        """水层校正计算对话框"""
        # 检查是否已计算理论走时
        if self.theoretical_traveltime_calculator is None:
            messagebox.showwarning('提示', '请先计算理论走时（Tools -> Calculate Theoretical Travel Time）')
            return
        
        # 检查是否为 v.in 格式模型
        model_info = self.theoretical_traveltime_calculator.get_model_info()
        if model_info.get('model_type') != 'vin':
            messagebox.showerror('错误', '水层校正需要 v.in 格式的速度模型')
            return
        
        # 创建对话框
        dialog = tk.Toplevel(self.root)
        dialog.title('计算水层校正')
        dialog.geometry('550x400')
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)
        
        # 1. 水层深度（自动从模型获取）
        depth_frame = ttk.LabelFrame(dialog, text='水层深度（从模型第一层自动获取）')
        depth_frame.pack(fill=tk.X, pady=5, padx=10)
        
        water_depth = self.theoretical_traveltime_calculator.get_water_layer_depth()
        depth_info_label = ttk.Label(depth_frame, text='', font=('Arial', 13))
        depth_info_label.pack(pady=5, padx=5)
        
        if isinstance(water_depth, float):
            depth_info_label.config(text=f'水层深度（常数）: {water_depth:.3f} km')
        elif isinstance(water_depth, tuple):
            x_array, depth_array = water_depth
            depth_info_label.config(
                text=f'水层深度（变化）: [{np.min(depth_array):.3f}, {np.max(depth_array):.3f}] km'
            )
        else:
            depth_info_label.config(text='⚠ 无法获取水层深度', foreground='red')
        
        # 2. 速度参数
        velocity_frame = ttk.LabelFrame(dialog, text='速度参数')
        velocity_frame.pack(fill=tk.X, padx=10, pady=5)
        
        v_water_var = tk.DoubleVar(value=1.5)
        ttk.Label(velocity_frame, text='水层速度 (km/s):').grid(row=0, column=0, padx=5, sticky='w')
        ttk.Entry(velocity_frame, textvariable=v_water_var, width=10).grid(row=0, column=1, padx=5)
        
        v_replacement_var = tk.DoubleVar()
        ttk.Label(velocity_frame, text='替换速度 (km/s，可选):').grid(row=1, column=0, padx=5, sticky='w')
        ttk.Entry(velocity_frame, textvariable=v_replacement_var, width=10).grid(row=1, column=1, padx=5)
        ttk.Label(velocity_frame, text='（留空则从模型自动获取）', 
                 font=('Arial', 8), foreground='gray').grid(row=1, column=2, padx=5, sticky='w')
        
        # 3. 说明文字
        info_frame = ttk.LabelFrame(dialog, text='说明')
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        info_text = (
            '水层校正将走时从水层速度校正到替换速度。\n'
            '校正公式：校正量 = 水层路径长度 × (1/v_water - 1/v_replacement)\n'
            '校正后的走时：t_corrected = t_original - correction\n\n'
            '注意：\n'
            '1. 需要先计算理论走时（Tools -> Calculate Theoretical Travel Time）\n'
            '2. 水层深度自动从 v.in 模型的第一层获取\n'
            '3. 替换速度默认从模型第二层获取，或使用默认值 2.0 km/s'
        )
        ttk.Label(info_frame, text=info_text, font=('Arial', 8), 
                 justify=tk.LEFT, wraplength=500).pack(pady=5, padx=5)
        
        # 4. 按钮
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def on_ok():
            v_water = v_water_var.get()
            v_replacement = v_replacement_var.get() if v_replacement_var.get() > 0 else None
            
            dialog.destroy()
            self.calculate_water_layer_correction(
                v_water=v_water,
                v_replacement=v_replacement
            )
        
        ttk.Button(button_frame, text='确定', command=on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text='取消', command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def calculate_water_layer_correction(self,
                                        v_water: float = 1.5,
                                        v_replacement: Optional[float] = None):
        """计算水层校正
        
        Args:
            v_water: 水层速度（km/s）
            v_replacement: 替换速度（km/s），如果为None则自动获取
        """
        try:
            if self.theoretical_traveltime_calculator is None:
                messagebox.showerror('错误', '请先计算理论走时')
                return
            
            self.update_status('正在计算水层校正...')
            
            # 获取射线数据
            rays = self.theoretical_traveltime_calculator.get_all_rays(max_rays=1000)
            if not rays:
                messagebox.showerror('错误', '未获取到射线数据，请先计算理论走时')
                return
            
            # 计算水层校正量
            corrections = self.theoretical_traveltime_calculator.calculate_water_layer_correction(
                rays=rays,
                water_depth=None,  # 自动从模型获取
                v_water=v_water,
                v_replacement=v_replacement,
                return_by_distance=False  # 按射线索引返回
            )
            
            if not corrections:
                messagebox.showwarning('警告', '未计算到水层校正量（可能射线未经过水层）')
                return
            
            # 保存校正量
            self.water_layer_corrections = corrections
            self.show_water_layer_correction = True
            
            # 如果有理论走时数据，应用校正
            if self.theoretical_times_data is not None:
                # 将校正量按距离映射（如果可能）
                # 这里简化处理：如果有理论走时数据，尝试应用校正
                self.update_status('正在应用水层校正...')
                
                # 获取理论走时数据
                distances = self.theoretical_times_data['distances']
                times = self.theoretical_times_data['times']
                
                # 创建按距离索引的校正量字典（简化：使用最近的射线校正量）
                # 注意：这里需要更精确的映射，暂时使用第一个校正量作为示例
                if corrections:
                    # 简化：对所有距离使用平均校正量
                    avg_correction = np.mean(list(corrections.values()))
                    corrected_times = times - avg_correction
                    
                    self.water_layer_corrected_times = {
                        'distances': distances,
                        'times': corrected_times,
                        'original_times': times,
                        'corrections': np.full_like(times, avg_correction)
                    }
                else:
                    self.water_layer_corrected_times = None
            else:
                self.water_layer_corrected_times = None
            
            # 更新绘图
            self.request_plot_refresh(immediate=True)
            
            # 显示成功消息
            n_rays = len(corrections)
            avg_correction = np.mean(list(corrections.values()))
            self.update_status(f'水层校正计算完成：{n_rays} 条射线，平均校正量 {avg_correction:.6f} s')
            messagebox.showinfo('成功', 
                              f'水层校正计算完成\n'
                              f'共 {n_rays} 条射线\n'
                              f'平均校正量: {avg_correction:.6f} s\n'
                              f'水层速度: {v_water:.3f} km/s\n'
                              f'替换速度: {v_replacement if v_replacement else "自动获取"} km/s')
            
        except Exception as e:
            import traceback
            error_msg = f'水层校正计算失败: {str(e)}\n{traceback.format_exc()}'
            self.update_status(error_msg)
            messagebox.showerror('错误', error_msg)
    
    def clear_water_layer_correction(self):
        """清除水层校正"""
        self.water_layer_corrections = {}
        self.water_layer_corrected_times = None
        self.show_water_layer_correction = False
        self.update_status('已清除水层校正')
        self.request_plot_refresh(immediate=True)
    
    def run(self):
        """运行主循环"""
        self.update_status()
        self.root.mainloop()


def main():
    """主函数（兼容入口，统一转发到 Qt Fast Viewer）。"""
    try:
        from .qt_fast_viewer import main as qt_main
    except Exception:
        from qt_fast_viewer import main as qt_main
    return qt_main()


if __name__ == '__main__':
    raise SystemExit(main())
