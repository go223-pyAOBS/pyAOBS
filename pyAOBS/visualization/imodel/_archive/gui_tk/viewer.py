"""
Tk 主窗口：布局与生命周期；业务逻辑见同包内各 *Mixin 模块。
"""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from pyAOBS.visualization.show_model import GridModelProcessor
    from pyAOBS.model_building.zeltform import ZeltVelocityModel2d
    from pyAOBS.visualization.imodel import (
        ProfileExtractor,
        PropertyCalculator,
        GravityCalculator,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from pyAOBS.visualization.show_model import GridModelProcessor
    from pyAOBS.model_building.zeltform import ZeltVelocityModel2d
    from pyAOBS.visualization.imodel import (
        ProfileExtractor,
        PropertyCalculator,
        GravityCalculator,
    )

from .workbench_state import WorkbenchStateMixin
from .model_surface import ModelSurfaceMixin
from .interaction import InteractionMixin
from .profiles import ProfileMixin
from .properties_ui import PropertiesUIMixin
from .property_panel import PropertyPanelMixin
from .rock_scatter import RockScatterMixin
from .gravity_ui import GravityUIMixin


class InteractiveModelViewerGUI(
    GravityUIMixin,
    RockScatterMixin,
    PropertyPanelMixin,
    PropertiesUIMixin,
    ProfileMixin,
    InteractionMixin,
    ModelSurfaceMixin,
    WorkbenchStateMixin,
):
    """基于Tkinter的交互式速度模型查看器 - 主窗口类"""
    
    def __init__(self, master=None, grid_file: Optional[str] = None, grid_data: Optional[xr.Dataset] = None):
        """初始化交互式模型查看器GUI
        
        Args:
            master: Tkinter根窗口，如果为None则创建新窗口
            grid_file: 速度模型grid文件路径
            grid_data: 或直接提供xarray Dataset
        """
        # 创建主窗口
        if master is None:
            self.root = tk.Tk()
        else:
            self.root = master
        
        self.root.title('Interactive Velocity Model Viewer')
        self.root.geometry('1400x900+100+50')  # 增加窗口宽度，为下方结果区留空间
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
        self.current_model_file = str(Path(grid_file).resolve()) if grid_file else ""
        self.vs_model_file = ""
        self.vs_grid_data = None
        self._restoring_state = False
        
        # 加载数据
        if grid_data is not None:
            self.grid_data = grid_data
            self.zelt_model = None  # grid_data模式下没有zelt_model
            # 自动检测速度变量名
            self._detect_velocity_var()
        elif grid_file is not None:
            # 检测是否为v.in格式
            file_path = Path(grid_file)
            is_vin_format = (
                file_path.name == 'v.in' or 
                file_path.name.endswith('.vin') or
                InteractiveModelViewerGUI._is_vin_format_static(grid_file)
            )
            
            if is_vin_format:
                # 加载v.in格式
                self.zelt_model = ZeltVelocityModel2d(model_file=grid_file)
                self.grid_data = self.zelt_model.to_xarray(dx=2.0, dz=0.5)
                self.processor = GridModelProcessor()
                self.processor.velocity_grid = self.grid_data
            else:
                # 加载grid格式
                self.zelt_model = None
                self.processor = GridModelProcessor(grid_file=grid_file)
                self.grid_data = self.processor.velocity_grid
            # 自动检测速度变量名（v.in和grid格式通用）
            self._detect_velocity_var()
        else:
            self.grid_data = None
            self.zelt_model = None
            self.velocity_var = 'velocity'
        
        # 初始化组件
        if self.grid_data is not None:
            self.profile_extractor = ProfileExtractor(self.grid_data)
            self.property_calculator = PropertyCalculator(self.grid_data, vs_grid_data=self.vs_grid_data)
            self.gravity_calculator = GravityCalculator()
        else:
            self.profile_extractor = None
            self.property_calculator = None
            self.gravity_calculator = GravityCalculator()
        
        # v.in 模型对象（用于访问界面信息）
        self.zelt_model = None
        
        # 界面显示控制
        self.show_interfaces_var = tk.BooleanVar(value=True)  # 是否显示界面（用于复选框）
        self.basement_interface_idx = None  # 沉积基底界面索引（v.in文件）
        self.basement_interface_data = None  # 沉积基底界面数据（网格文件：x, z坐标）- 单个界面（向后兼容）
        self.loaded_interfaces = []  # 加载的界面列表（网格文件）：[{'x': array, 'z': array, 'name': str, 'file': str}, ...]
        self.basement_interface_file = None  # 界面文件路径（网格文件）- 单个界面（向后兼容）
        
        # 海底面相关
        self.seafloor_interface_idx = None  # 海底面界面索引（v.in文件）
        self.seafloor_interface_data = None  # 海底面界面数据（网格文件：x, z坐标）
        self.seafloor_depth_map = None  # 海底面深度映射（x -> z），用于插值
        
        # 交互工具
        self.point_selector = None
        self.polygon_selector = None
        
        # 存储结果
        self.selected_points = []
        self.selected_polygons = []  # 存储多边形顶点
        self.polygon_patches = []  # 存储Polygon对象引用，便于清除
        self.profiles = {}
        self.property_results = {}
        self.property_result_rows = []  # 采样点计算结果明细（用于导出）
        self._classifier_diag_logged = False
        self.zone_label_map = self._load_zone_label_map()
        self.fixed_zone_var = tk.StringVar(value='-')
        self.fixed_rock_classification_var = tk.StringVar(value='-')
        self.fixed_metamorphic_grade_var = tk.StringVar(value='-')
        self.fixed_geological_meaning_var = tk.StringVar(value='-')
        
        # Vp-Vs散点图窗口
        self.scatter_window = None
        self.scatter_fig = None
        self.scatter_ax = None
        self.scatter_canvas = None
        self.model_points = []  # 存储从模型中添加的点
        
        # Vp/Vs-Vp关系图窗口
        self.vp_vs_ratio_window = None
        self.vp_vs_ratio_fig = None
        self.vp_vs_ratio_ax = None
        self.vp_vs_ratio_canvas = None
        
        # 剖面图窗口
        self.profile_window = None
        self.profile_fig = None
        self.profile_ax = None
        self.profile_canvas = None
        self.profile_all_profiles = []  # 存储所有单独的剖面
        self.profile_one_d_model = None  # 存储平均后的一维模型
        self.profile_x_samples = []  # 存储X采样点
        self.profile_title = ''  # 存储标题
        # 1D模型范围数据（从1d文件夹加载）
        self.profile_1d_ranges = {}  # {文件名: {'vp': array, 'depth': array}}
        self.profile_1d_selected = {}  # {文件名: BooleanVar} 选中的范围
        # 控制要素显示的复选框变量
        self.vp_vs_ratio_show_rocks = tk.BooleanVar(value=True)  # 采样岩石分布
        self.vp_vs_ratio_show_water = tk.BooleanVar(value=True)  # 含水量(蛇纹石化百分比)
        self.vp_vs_ratio_show_aspect_ratio = tk.BooleanVar(value=True)  # 不同纵横比对应曲线
        self.vp_vs_ratio_show_porosity = tk.BooleanVar(value=True)  # 不同孔隙度对应曲线
        # 存储绘制数据，以便重新绘制
        self.vp_vs_ratio_db_data = None
        self.vp_vs_ratio_model_points_data = None
        self.vp_vs_ratio_water_data = None
        self.vp_vs_ratio_curves_data = None
        self.vp_vs_ratio_dem_params = None
        self.vp_vs_ratio_model_points = []  # 存储添加到Vp/Vs-Vp图的模型点
        
        # 重力异常图窗口
        self.gravity_window = None
        self.gravity_fig = None
        self.gravity_ax = None
        self.gravity_canvas = None
        self.gravity_bodies = []  # 存储重力计算的多边形体 [(x, z, rho), ...]
        self._full_model_gravity_cache = None  # {'model_key': str, 'x_obs_km': np.ndarray, 'gravity_anomaly': np.ndarray, ...}
        
        # 多边形采样点在主图上的显示
        self.polygon_sample_artists = []  # 存储采样点的图形对象引用
        
        # 等值线对象存储
        self.contour_lines = None  # 主模型图的等值线对象
        
        # 创建界面
        self.create_widgets()
        self._log_classifier_diagnostics()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # 启动时先尝试恢复会话状态，避免用默认空值覆盖已有状态文件。
        self._restore_workbench_state()
        # 若仍有已加载数据则确保主图可见（恢复失败时兜底显示）。
        if self.grid_data is not None:
            self.plot_model()
    

    def _detect_velocity_var(self):
        """自动检测速度变量名"""
        if self.grid_data is None:
            self.velocity_var = 'velocity'
            return
        
        data_vars = list(self.grid_data.data_vars)
        coords = list(self.grid_data.coords)
        
        # 优先查找常见的速度变量名
        velocity_candidates = ['velocity', 'v', 'vp', 'v_p', 'vel', 'speed']
        self.velocity_var = None
        
        for var in velocity_candidates:
            if var in data_vars:
                self.velocity_var = var
                break
        
        # 如果没找到，检查是否有z变量（z通常表示速度）
        if self.velocity_var is None:
            if 'z' in data_vars:
                self.velocity_var = 'z'
            elif len(data_vars) > 0:
                # 使用第一个数据变量
                self.velocity_var = data_vars[0]
            else:
                self.velocity_var = 'velocity'
    

    def create_widgets(self):
        """创建GUI组件"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=0, column=0, sticky='nswe')
        main_frame.rowconfigure(0, weight=2)  # 上方区域（绘图+侧边栏）稍微加高
        main_frame.rowconfigure(1, weight=3)  # 下方区域（结果显示）稍微减小，让上方更高一些
        main_frame.columnconfigure(0, weight=3)  # 绘图区域权重为3（保持宽度）
        main_frame.columnconfigure(1, weight=1)  # 侧边栏权重为1
        
        # 绘图区域（高度变窄，宽度保持）
        plot_frame = ttk.Frame(main_frame)
        plot_frame.grid(row=0, column=0, sticky='nswe', padx=(0, 10))
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        
        # 创建matplotlib图形（适当增加高度，保持宽度）
        self.fig = plt.Figure(figsize=(10, 3.5))
        self.ax_main = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        
        # 创建canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nswe')
        
        # 创建工具栏
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky='ew')
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        # 侧边栏（只包含控件，不包含结果显示）
        side_frame = ttk.Frame(main_frame)
        # 让侧边栏跨越上下两行，这样在结果窗口变窄后，右侧有更多垂直空间可用于扩展控件
        side_frame.grid(row=0, column=1, rowspan=2, sticky='nswe')
        side_frame.columnconfigure(0, weight=1)
        
        # File operations
        file_frame = ttk.LabelFrame(side_frame, text='File Operations')
        file_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        file_frame.columnconfigure(0, weight=1)
        file_frame.columnconfigure(1, weight=1)
        file_frame.columnconfigure(2, weight=1)
        ttk.Button(file_frame, text='Load Vp Model', command=self.open_vp_model).grid(row=0, column=0, sticky='ew', padx=2, pady=5)
        ttk.Button(file_frame, text='Save Figure', command=self.save_figure).grid(row=0, column=1, sticky='ew', padx=2, pady=5)
        ttk.Button(file_frame, text='Export Results', command=self.export_results).grid(row=0, column=2, sticky='ew', padx=2, pady=5)
        ttk.Button(file_frame, text='Load Vs Model', command=self.open_vs_model).grid(
            row=1, column=0, columnspan=3, sticky='ew', padx=2, pady=(0, 5)
        )
        ttk.Button(file_frame, text='Export Point Results', command=self.export_point_results).grid(
            row=2, column=0, columnspan=3, sticky='ew', padx=2, pady=(0, 5)
        )
        
        # Interactive tools
        tool_frame = ttk.LabelFrame(side_frame, text='Interactive Tools')
        tool_frame.grid(row=1, column=0, sticky='ew', pady=(0, 10))
        tool_frame.columnconfigure(0, weight=1)
        tool_frame.columnconfigure(1, weight=1)
        tool_frame.columnconfigure(2, weight=1)
        ttk.Button(tool_frame, text='Point Selection', command=self.enable_point_selection).grid(row=0, column=0, sticky='ew', padx=2, pady=2)
        ttk.Button(tool_frame, text='Polygon Selection', command=self.enable_polygon_selection).grid(row=0, column=1, sticky='ew', padx=2, pady=2)
        ttk.Button(tool_frame, text='Clear', command=self.clear_selections).grid(row=0, column=2, sticky='ew', padx=2, pady=2)
        
        # Profile extraction
        profile_frame = ttk.LabelFrame(side_frame, text='Profile Extraction')
        profile_frame.grid(row=2, column=0, sticky='ew', pady=(0, 10))
        
        # X坐标范围
        ttk.Label(profile_frame, text='X range (km):').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.x_min_entry = ttk.Entry(profile_frame, width=8)
        self.x_min_entry.grid(row=0, column=1, padx=2, pady=2)
        ttk.Label(profile_frame, text='to').grid(row=0, column=2, padx=2, pady=2)
        self.x_max_entry = ttk.Entry(profile_frame, width=8)
        self.x_max_entry.grid(row=0, column=3, padx=2, pady=2)
        
        # 采样间隔选择
        ttk.Label(profile_frame, text='Sampling (km):').grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.sampling_interval_var = tk.StringVar(value='1.0')
        sampling_combo = ttk.Combobox(profile_frame, textvariable=self.sampling_interval_var,
                                     values=['0.5', '1.0'], state='readonly', width=8)
        sampling_combo.grid(row=1, column=1, columnspan=3, sticky='w', padx=2, pady=2)
        
        # Vertical Profile按钮
        ttk.Button(profile_frame, text='Extract Vertical Profile', 
                  command=self.extract_vertical_profile).grid(row=2, column=0, columnspan=4, 
                                                              sticky='ew', padx=5, pady=5)
        
        # Show Interfaces复选框（移到Profile Extraction面板）
        ttk.Checkbutton(profile_frame, text='Show Interfaces', 
                       variable=self.show_interfaces_var,
                       command=self._toggle_interfaces).grid(row=3, column=0, columnspan=4, 
                                                             sticky='w', padx=5, pady=2)
        ttk.Label(profile_frame, text='(For v.in format only)', 
                 font=('TkDefaultFont', 7), foreground='gray').grid(row=4, column=0, columnspan=4, 
                                                                     sticky='w', padx=5, pady=(0, 2))
        
        # 沉积基底界面选择（v.in文件）
        ttk.Label(profile_frame, text='Basement Interface:', 
                 font=('TkDefaultFont', 9)).grid(row=5, column=0, sticky='w', padx=5, pady=2)
        self.basement_interface_var = tk.StringVar(value='None')
        self.basement_interface_combo = ttk.Combobox(profile_frame, 
                                                     textvariable=self.basement_interface_var,
                                                     state='readonly', width=15)
        self.basement_interface_combo.grid(row=5, column=1, columnspan=3, sticky='w', padx=2, pady=2)
        self.basement_interface_combo.bind('<<ComboboxSelected>>', self._on_basement_interface_selected)
        
        # 界面文件加载（网格文件）
        ttk.Label(profile_frame, text='Interface File:', 
                 font=('TkDefaultFont', 9)).grid(row=6, column=0, sticky='w', padx=5, pady=2)
        interface_file_frame = ttk.Frame(profile_frame)
        interface_file_frame.grid(row=6, column=1, columnspan=3, sticky='ew', padx=2, pady=2)
        self.interface_file_label = ttk.Label(interface_file_frame, text='None', 
                                               foreground='gray', width=12)
        self.interface_file_label.pack(side=tk.LEFT, padx=2)
        ttk.Button(interface_file_frame, text='Load...', 
                  command=self._load_interface_file, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(interface_file_frame, text='Save...', 
                  command=self._save_interface_file, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(interface_file_frame, text='Clear', 
                  command=self._clear_loaded_interfaces, width=8).pack(side=tk.LEFT, padx=2)
        
        # 海底面选择（v.in文件）
        ttk.Label(profile_frame, text='Seafloor Interface:', 
                 font=('TkDefaultFont', 9)).grid(row=7, column=0, sticky='w', padx=5, pady=2)
        self.seafloor_interface_var = tk.StringVar(value='Auto')
        self.seafloor_interface_combo = ttk.Combobox(profile_frame, 
                                                     textvariable=self.seafloor_interface_var,
                                                     state='readonly', width=15)
        self.seafloor_interface_combo.grid(row=7, column=1, columnspan=3, sticky='w', padx=2, pady=2)
        self.seafloor_interface_combo.bind('<<ComboboxSelected>>', self._on_seafloor_interface_selected)
        
        # 更新界面选择选项
        self._update_basement_interface_options()
        self._update_seafloor_interface_options()
        
        # Model settings
        model_setting_frame = ttk.LabelFrame(side_frame, text='Model Settings')
        model_setting_frame.grid(row=3, column=0, sticky='ew', pady=(0, 10))
        
        # 模型类型选择（更清晰的标签和选项）
        ttk.Label(model_setting_frame, text='Wave Type:', font=('TkDefaultFont', 9, 'bold')).grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.model_type_var = tk.StringVar(value='vp')
        # 使用内部变量存储实际值，下拉框显示友好文本
        self.model_type_display_var = tk.StringVar(value='P-wave (Vp)')
        model_type_combo = ttk.Combobox(model_setting_frame, textvariable=self.model_type_display_var, 
                                       values=['P-wave (Vp)', 'S-wave (Vs)'], state='readonly', width=15)
        model_type_combo.grid(row=0, column=1, padx=5, pady=2)
        # 添加说明文字
        ttk.Label(model_setting_frame, text='(Other wave type will be calculated)', 
                 font=('TkDefaultFont', 7), foreground='gray').grid(row=1, column=0, columnspan=2, sticky='w', padx=5, pady=(0, 2))
        
        def on_model_type_change(e):
            selected = self.model_type_display_var.get()
            if 'P-wave' in selected or 'Vp' in selected:
                self.model_type_var.set('vp')
                self.log_result("Model type set to: P-wave (Vp model) - S-wave will be calculated from Vp")
            elif 'S-wave' in selected or 'Vs' in selected:
                self.model_type_var.set('vs')
                self.log_result("Model type set to: S-wave (Vs model) - P-wave will be calculated from Vs")
        
        model_type_combo.bind('<<ComboboxSelected>>', on_model_type_change)
        
        # 速度转换方法选择
        ttk.Label(model_setting_frame, text='Velocity Method:', font=('TkDefaultFont', 9, 'bold')).grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.velocity_method_var = tk.StringVar(value='brocher')
        velocity_combo = ttk.Combobox(model_setting_frame, textvariable=self.velocity_method_var,
                                     values=['brocher', 'castagna'], state='readonly', width=15)
        velocity_combo.grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(model_setting_frame, text='(For Vp↔Vs conversion)', 
                 font=('TkDefaultFont', 7), foreground='gray').grid(row=3, column=0, columnspan=2, sticky='w', padx=5, pady=(0, 2))
        velocity_combo.bind('<<ComboboxSelected>>', lambda e: self.log_result(f"Velocity conversion method: {self.velocity_method_var.get()}"))
        
        # 密度计算方法选择
        ttk.Label(model_setting_frame, text='Density Method:').grid(row=4, column=0, sticky='w', padx=5, pady=2)
        self.density_method_var = tk.StringVar(value='gardner')
        density_combo = ttk.Combobox(model_setting_frame, textvariable=self.density_method_var,
                                    values=['gardner', 'brocher', 'nafe_drake', 'tomo2d_sediment'], state='readonly', width=15)
        density_combo.grid(row=4, column=1, padx=5, pady=2)
        density_combo.bind('<<ComboboxSelected>>', lambda e: self.log_result(f"Density method: {self.density_method_var.get()}"))
        
        # 等值线显示选项
        self.show_contours_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(model_setting_frame, text='Show Contours', 
                       variable=self.show_contours_var,
                       command=self._toggle_contours).grid(row=5, column=0, columnspan=2, 
                                                          sticky='w', padx=5, pady=2)
        
        # Property calculation
        property_frame = ttk.LabelFrame(side_frame, text='Property Calculation')
        property_frame.grid(row=5, column=0, sticky='ew', pady=(0, 10))
        property_frame.columnconfigure(1, weight=1)
        property_frame.columnconfigure(3, weight=1)
        
        ttk.Label(property_frame, text='X (km):').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.prop_x_entry = ttk.Entry(property_frame, width=8)
        self.prop_x_entry.grid(row=0, column=1, padx=2, pady=2, sticky='ew')
        
        ttk.Label(property_frame, text='Z (km):').grid(row=0, column=2, sticky='w', padx=5, pady=2)
        self.prop_z_entry = ttk.Entry(property_frame, width=8)
        self.prop_z_entry.grid(row=0, column=3, padx=2, pady=2, sticky='ew')
        
        ttk.Button(property_frame, text='Calculate', command=self.calculate_properties).grid(row=0, column=4, sticky='ew', padx=5, pady=2)
        
        # Rock database visualization
        rock_viz_frame = ttk.LabelFrame(side_frame, text='Rock Database')
        rock_viz_frame.grid(row=6, column=0, sticky='ew', pady=(0, 10))
        
        # 第一行：两个散点图按钮（并排）
        ttk.Button(rock_viz_frame, text='Show Vp-Vs Scatter Plot', command=self.show_vp_vs_scatter).grid(row=0, column=0, sticky='ew', padx=5, pady=2)
        ttk.Button(rock_viz_frame, text='Show Vp/Vs-Vp Plot', command=self.show_vp_vs_ratio_plot).grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        
        # Vp-Vs散点图控制按钮
        ttk.Label(rock_viz_frame, text='Vp-Vs Plot:', font=('TkDefaultFont', 8)).grid(row=1, column=0, sticky='w', padx=5, pady=(5, 2))
        ttk.Button(rock_viz_frame, text='Add Selected Points', command=self.add_points_to_scatter).grid(row=2, column=0, sticky='ew', padx=5, pady=2)
        ttk.Button(rock_viz_frame, text='Add Polygon Average', command=self.add_polygon_to_scatter).grid(row=3, column=0, sticky='ew', padx=5, pady=2)
        ttk.Button(rock_viz_frame, text='Add Polygon Samples', command=self.add_polygon_samples_to_scatter).grid(row=4, column=0, sticky='ew', padx=5, pady=2)
        ttk.Button(rock_viz_frame, text='Clear Model Points', command=self.clear_scatter_model_points).grid(row=5, column=0, sticky='ew', padx=5, pady=2)
        
        # Vp/Vs-Vp图控制按钮
        ttk.Label(rock_viz_frame, text='Vp/Vs-Vp Plot:', font=('TkDefaultFont', 8)).grid(row=1, column=1, sticky='w', padx=5, pady=(5, 2))
        ttk.Button(rock_viz_frame, text='Add Selected Points', command=self.add_points_to_vp_vs_ratio).grid(row=2, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(rock_viz_frame, text='Add Polygon Average', command=self.add_polygon_to_vp_vs_ratio).grid(row=3, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(rock_viz_frame, text='Add Polygon Samples', command=self.add_polygon_samples_to_vp_vs_ratio).grid(row=4, column=1, sticky='ew', padx=5, pady=2)
        ttk.Button(rock_viz_frame, text='Clear Model Points', command=self.clear_vp_vs_ratio_model_points).grid(row=5, column=1, sticky='ew', padx=5, pady=2)
        
        # 配置列权重，使两列等宽
        rock_viz_frame.columnconfigure(0, weight=1)
        rock_viz_frame.columnconfigure(1, weight=1)
        
        # Gravity simulation
        gravity_frame = ttk.LabelFrame(side_frame, text='Gravity Simulation')
        gravity_frame.grid(row=7, column=0, sticky='ew', pady=(0, 10))
        
        ttk.Button(gravity_frame, text='Show Gravity Anomaly Plot', 
                  command=self.show_gravity_plot).grid(row=0, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
        
        ttk.Label(gravity_frame, text='Background Density:', font=('TkDefaultFont', 8)).grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.background_density_var = tk.StringVar(value='2.67')
        bg_density_entry = ttk.Entry(gravity_frame, textvariable=self.background_density_var, width=10)
        bg_density_entry.grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(gravity_frame, text='(g/cm³)', font=('TkDefaultFont', 7), foreground='gray').grid(row=1, column=2, sticky='w', padx=2)
        
        ttk.Label(gravity_frame, text='Observation Level:', font=('TkDefaultFont', 8)).grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.obs_level_var = tk.StringVar(value='0.0')
        obs_level_entry = ttk.Entry(gravity_frame, textvariable=self.obs_level_var, width=10)
        obs_level_entry.grid(row=2, column=1, padx=5, pady=2)
        ttk.Label(
            gravity_frame,
            text='(km, +down; sea level=0)',
            font=('TkDefaultFont', 7),
            foreground='gray',
        ).grid(row=2, column=2, sticky='w', padx=2)
        
        ttk.Label(gravity_frame, text='Max Grid Size:', font=('TkDefaultFont', 8)).grid(row=3, column=0, sticky='w', padx=5, pady=2)
        self.max_grid_size_var = tk.StringVar(value='100')
        max_grid_entry = ttk.Entry(gravity_frame, textvariable=self.max_grid_size_var, width=10)
        max_grid_entry.grid(row=3, column=1, padx=5, pady=2)
        ttk.Label(gravity_frame, text='(points)', font=('TkDefaultFont', 7), foreground='gray').grid(row=3, column=2, sticky='w', padx=2)
        
        ttk.Label(gravity_frame, text='Extension Distance:', font=('TkDefaultFont', 8)).grid(row=4, column=0, sticky='w', padx=5, pady=2)
        self.extension_dist_var = tk.StringVar(value='40.0')
        ext_dist_entry = ttk.Entry(gravity_frame, textvariable=self.extension_dist_var, width=10)
        ext_dist_entry.grid(row=4, column=1, padx=5, pady=2)
        ttk.Label(gravity_frame, text='(km)', font=('TkDefaultFont', 7), foreground='gray').grid(row=4, column=2, sticky='w', padx=2)

        ttk.Label(gravity_frame, text='Gravity Method:', font=('TkDefaultFont', 8)).grid(row=5, column=0, sticky='w', padx=5, pady=2)
        self.gravity_method_var = tk.StringVar(value='tomo2d_fft')
        gravity_method_combo = ttk.Combobox(
            gravity_frame,
            textvariable=self.gravity_method_var,
            values=['tomo2d_fft', 'talwani_grid'],
            state='readonly',
            width=12,
        )
        gravity_method_combo.grid(row=5, column=1, padx=5, pady=2, sticky='ew')
        gravity_method_combo.bind(
            '<<ComboboxSelected>>',
            lambda e: self.log_result(f"Gravity method: {self.gravity_method_var.get()}"),
        )
        ttk.Label(
            gravity_frame,
            text='(full model)',
            font=('TkDefaultFont', 7),
            foreground='gray',
        ).grid(row=5, column=2, sticky='w', padx=2)

        self.strict_jgrav_var = tk.BooleanVar(value=False)
        strict_chk = ttk.Checkbutton(
            gravity_frame,
            text='Strict jgrav mode',
            variable=self.strict_jgrav_var,
            command=lambda: self.log_result(
                f"Strict jgrav mode: {'ON' if self.strict_jgrav_var.get() else 'OFF'}"
            ),
        )
        strict_chk.grid(row=6, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        
        # 配置列权重，使两列等宽
        gravity_frame.columnconfigure(0, weight=1)
        gravity_frame.columnconfigure(1, weight=1)
        
        # 第一行：Full model 和 Add polygon
        ttk.Button(gravity_frame, text='Full Model', 
                  command=self.calculate_gravity_from_full_model).grid(row=7, column=0, sticky='ew', padx=2, pady=2)
        ttk.Button(gravity_frame, text='Add Polygon', 
                  command=self.add_polygon_as_gravity_body).grid(row=7, column=1, sticky='ew', padx=2, pady=2)
        
        # 第二行：Compare Methods
        ttk.Button(
            gravity_frame,
            text='Compare Methods',
            command=self.compare_full_model_gravity_methods,
        ).grid(row=8, column=0, columnspan=2, sticky='ew', padx=2, pady=2)

        # 第三行：Calculate Profile 和 Clear Bodies
        ttk.Button(gravity_frame, text='Calculate Profile', 
                  command=self.calculate_gravity_profile).grid(row=9, column=0, sticky='ew', padx=2, pady=2)
        ttk.Button(gravity_frame, text='Clear Bodies', 
                  command=self.clear_gravity_bodies).grid(row=9, column=1, sticky='ew', padx=2, pady=2)

        # Property profiles (Density / Temperature / Pressure)
        property_profile_frame = ttk.LabelFrame(side_frame, text='Property Profiles')
        property_profile_frame.grid(row=8, column=0, sticky='ew', pady=(0, 10))
        property_profile_frame.columnconfigure(0, weight=1)
        property_profile_frame.columnconfigure(1, weight=1)
        property_profile_frame.columnconfigure(2, weight=1)
        
        ttk.Button(
            property_profile_frame,
            text='Density Profile',
            command=lambda: self.plot_property_profile('density'),
        ).grid(row=0, column=0, sticky='ew', padx=4, pady=2)
        ttk.Button(
            property_profile_frame,
            text='Temperature Profile',
            command=lambda: self.plot_property_profile('temperature'),
        ).grid(row=0, column=1, sticky='ew', padx=4, pady=2)
        ttk.Button(
            property_profile_frame,
            text='Pressure Profile',
            command=lambda: self.plot_property_profile('pressure'),
        ).grid(row=0, column=2, sticky='ew', padx=4, pady=2)
        
        # Results display area (moved to bottom of main frame, extended)
        result_frame = ttk.LabelFrame(main_frame, text='Calculation Results')
        # 仅占用左侧（与主模型同列），让右侧面板有更多垂直空间
        result_frame.grid(row=1, column=0, columnspan=1, sticky='nswe', padx=10, pady=(10, 0))
        result_frame.rowconfigure(0, weight=1)
        result_frame.rowconfigure(1, weight=0)
        result_frame.columnconfigure(0, weight=1)
        
        # 创建文本显示区域（使用ScrollText或Text，增加高度）
        from tkinter.scrolledtext import ScrolledText
        self.result_text = ScrolledText(result_frame, width=80, height=12, font=('Consolas', 11))
        self.result_text.grid(row=0, column=0, sticky='nswe', padx=5, pady=5)

        # 固定地质信息区（不随日志滚动）
        fixed_info_frame = ttk.LabelFrame(result_frame, text='Current Geological Info')
        fixed_info_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=(0, 5))
        fixed_info_frame.columnconfigure(1, weight=1)
        ttk.Label(fixed_info_frame, text='Zone:').grid(row=0, column=0, sticky='nw', padx=5, pady=2)
        ttk.Label(fixed_info_frame, textvariable=self.fixed_zone_var, wraplength=900).grid(
            row=0, column=1, sticky='nw', padx=5, pady=2
        )
        ttk.Label(fixed_info_frame, text='Rock classification:').grid(row=1, column=0, sticky='nw', padx=5, pady=2)
        ttk.Label(fixed_info_frame, textvariable=self.fixed_rock_classification_var, wraplength=900).grid(
            row=1, column=1, sticky='nw', padx=5, pady=2
        )
        ttk.Label(fixed_info_frame, text='Metamorphic grade:').grid(row=2, column=0, sticky='nw', padx=5, pady=2)
        ttk.Label(fixed_info_frame, textvariable=self.fixed_metamorphic_grade_var, wraplength=900).grid(
            row=2, column=1, sticky='nw', padx=5, pady=2
        )
        ttk.Label(fixed_info_frame, text='Geological meaning:').grid(row=3, column=0, sticky='nw', padx=5, pady=2)
        ttk.Label(fixed_info_frame, textvariable=self.fixed_geological_meaning_var, wraplength=900).grid(
            row=3, column=1, sticky='nw', padx=5, pady=2
        )
        
        # 菜单栏
        self.create_menu()
        
        # 绑定键盘事件
        self.root.bind('<KeyPress>', self.on_key_press)
    

    def create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        filemenu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='File', menu=filemenu)
        filemenu.add_command(label='Load Vp Model', command=self.open_vp_model, accelerator='Ctrl+O')
        filemenu.add_command(label='Load Vs Model', command=self.open_vs_model)
        filemenu.add_separator()
        filemenu.add_command(label='Save Figure', command=self.save_figure)
        filemenu.add_command(label='Export Point Results', command=self.export_point_results)
        filemenu.add_command(label='Export Results', command=self.export_results)
        filemenu.add_separator()
        filemenu.add_command(label='Exit', command=self._on_close)
        
        # Tools menu
        toolmenu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='Tools', menu=toolmenu)
        toolmenu.add_command(label='Point Selection', command=self.enable_point_selection)
        toolmenu.add_command(label='Polygon Selection', command=self.enable_polygon_selection)
        toolmenu.add_separator()
        toolmenu.add_command(label='Clear Selections', command=self.clear_selections)
        
        # Help menu
        helpmenu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='Help', menu=helpmenu)
        helpmenu.add_command(label='User Guide', command=self.show_help)
        helpmenu.add_command(label='About', command=self.show_about)

    def show_about(self):
        """Show about information"""
        about_text = """
Interactive Velocity Model Viewer

Version: 1.0
Author: Haibo Huang

Features:
- Visualize velocity models
- Interactive point/polygon selection
- Extract 1D profiles
- Calculate property parameters (density, pressure, rock type, gravity, etc.)
        """
        messagebox.showinfo('About', about_text)
    
    def run(self):
        """运行GUI主循环"""
        self.root.mainloop()
        # Fallback: if loop exits via quit(), persist latest state.
        self._save_workbench_state()


def main():
    """主函数"""
    # 根据操作系统设置DPI感知（Windows）
    if platform.system() == 'Windows':
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except ImportError:
            pass
    
    root = tk.Tk()
    app = InteractiveModelViewerGUI(master=root)
    app.run()


if __name__ == '__main__':
    main()
