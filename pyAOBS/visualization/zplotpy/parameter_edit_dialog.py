# -*- coding: utf-8 -*-
"""
parameter_edit_dialog.py - 参数编辑对话框

用于编辑不在工具栏中的高级参数
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Callable
try:
    from .parameters import ZPlotParameters
    from .top_toolbar import get_chinese_font
except ImportError:
    from parameters import ZPlotParameters
    from top_toolbar import get_chinese_font


class ParameterEditDialog:
    """参数编辑对话框"""
    
    def __init__(self, parent, params: ZPlotParameters, callback: Optional[Callable] = None):
        """初始化参数编辑对话框
        
        Args:
            parent: 父窗口
            params: 参数对象
            callback: 参数更新后的回调函数
        """
        self.parent = parent
        self.params = params
        self.callback = callback
        
        # 统一字体设置（字号12，统一所有文字）
        self.font_normal = get_chinese_font(12)
        self.font_bold = get_chinese_font(12, 'bold')
        
        # 设置全局ttk样式，统一所有组件的字体（字号12）
        style = ttk.Style()
        # 为对话框创建独立的样式，避免影响其他窗口
        style.configure('Dialog.TLabel', font=self.font_normal)
        style.configure('Dialog.TRadiobutton', font=self.font_normal)
        style.configure('Dialog.TButton', font=self.font_normal)
        style.configure('Dialog.TEntry', font=self.font_normal)
        # 同时设置默认样式
        style.configure('TLabel', font=self.font_normal)
        style.configure('TRadiobutton', font=self.font_normal)
        style.configure('TButton', font=self.font_normal)
        style.configure('TEntry', font=self.font_normal)
        
        # 创建对话框窗口
        self.dialog = tk.Toplevel(parent)
        self.dialog.title('参数编辑')
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # 设置窗口大小
        self.dialog.geometry('800x800')
        
        # 创建界面
        self.create_widgets()
        
        # 居中显示
        self.center_window()
        
        # 绑定关闭事件
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_cancel)
    
    def center_window(self):
        """居中显示窗口"""
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')
    
    def _create_label(self, parent, text, **kwargs):
        """创建标签并应用统一字体"""
        label = ttk.Label(parent, text=text, **kwargs)
        label.configure(font=self.font_normal)
        return label
    
    def _create_radiobutton(self, parent, text, **kwargs):
        """创建单选按钮并应用统一字体"""
        rb = ttk.Radiobutton(parent, text=text, **kwargs)
        rb.configure(font=self.font_normal)
        return rb
    
    def create_widgets(self):
        """创建界面组件"""
        # 创建Notebook（标签页）
        notebook = ttk.Notebook(self.dialog)
        notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # 自适应叠加参数标签页
        adaptive_frame = ttk.Frame(notebook)
        notebook.add(adaptive_frame, text='自适应叠加参数')
        self.create_adaptive_tab(adaptive_frame)
        
        # Hilbert变换参数标签页
        hilbert_frame = ttk.Frame(notebook)
        notebook.add(hilbert_frame, text='Hilbert变换参数')
        self.create_hilbert_tab(hilbert_frame)
        
        # 颜色设置标签页
        color_frame = ttk.Frame(notebook)
        notebook.add(color_frame, text='颜色设置')
        self.create_color_tab(color_frame)
        
        # 基本控制参数标签页 (1-5)
        basic_frame = ttk.Frame(notebook)
        notebook.add(basic_frame, text='基本控制 (1-5)')
        self.create_basic_tab(basic_frame)
        
        # 显示位置参数标签页 (8-12)
        display_frame = ttk.Frame(notebook)
        notebook.add(display_frame, text='显示位置 (8-12)')
        self.create_display_position_tab(display_frame)
        
        # 能量和静音参数标签页 (16-24)
        energy_frame = ttk.Frame(notebook)
        notebook.add(energy_frame, text='能量/静音 (16-24)')
        self.create_energy_tab(energy_frame)
        
        # 轴和布局参数标签页 (25-35)
        axis_frame = ttk.Frame(notebook)
        notebook.add(axis_frame, text='轴/布局 (25-35)')
        self.create_axis_tab(axis_frame)
        
        # 按钮框架
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(button_frame, text='确定', command=self.on_ok, width=15).pack(side='right', padx=5)
        ttk.Button(button_frame, text='取消', command=self.on_cancel, width=15).pack(side='right', padx=5)
        ttk.Button(button_frame, text='重置为默认值', command=self.on_reset, width=15).pack(side='left', padx=5)
    
    def create_adaptive_tab(self, parent):
        """创建自适应叠加参数标签页"""
        # 使用滚动框架
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        row = 0
        
        # 自适应叠加开关
        ttk.Label(scrollable_frame, text='自适应叠加开关:').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.iadaptive_var = tk.IntVar(value=self.params.iadaptive)
        ttk.Radiobutton(scrollable_frame, text='关闭 (0)', variable=self.iadaptive_var, value=0).grid(row=row, column=1, sticky='w', padx=5)
        ttk.Radiobutton(scrollable_frame, text='启用 (1)', variable=self.iadaptive_var, value=1).grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 叠加迭代次数
        ttk.Label(scrollable_frame, text='叠加迭代次数 (nsi):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.nsi_entry = ttk.Entry(scrollable_frame, width=15)
        self.nsi_entry.insert(0, str(self.params.nsi))
        self.nsi_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 10)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # Lp范数指数
        ttk.Label(scrollable_frame, text='Lp范数指数 (pjgl):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.pjgl_var = tk.IntVar(value=self.params.pjgl)
        ttk.Radiobutton(scrollable_frame, text='L1 (1)', variable=self.pjgl_var, value=1).grid(row=row, column=1, sticky='w', padx=5)
        ttk.Radiobutton(scrollable_frame, text='L2 (2)', variable=self.pjgl_var, value=2).grid(row=row, column=2, sticky='w', padx=5)
        ttk.Radiobutton(scrollable_frame, text='L3 (3)', variable=self.pjgl_var, value=3).grid(row=row, column=3, sticky='w', padx=5)
        row += 1
        
        # 叠加窗口起始时间
        ttk.Label(scrollable_frame, text='叠加窗口起始时间 (stkwb, 秒):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.stkwb_entry = ttk.Entry(scrollable_frame, width=15)
        self.stkwb_entry.insert(0, str(self.params.stkwb))
        self.stkwb_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 0.04)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 叠加窗口长度
        ttk.Label(scrollable_frame, text='叠加窗口长度 (stkwl, 秒):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.stkwl_entry = ttk.Entry(scrollable_frame, width=15)
        self.stkwl_entry.insert(0, str(self.params.stkwl))
        self.stkwl_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 0.5)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 差分时间搜索范围
        ttk.Label(scrollable_frame, text='差分时间搜索范围 (dtcw, 秒):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.dtcw_entry = ttk.Entry(scrollable_frame, width=15)
        self.dtcw_entry.insert(0, str(self.params.dtcw))
        self.dtcw_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 0.1)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # Hilbert变换比例因子 (36)
        ttk.Label(scrollable_frame, text='36. hilbratio (Hilbert变换比例因子):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.hilbratio_entry = ttk.Entry(scrollable_frame, width=15)
        self.hilbratio_entry.insert(0, str(self.params.hilbratio))
        self.hilbratio_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 3.0)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 37. nsi - 叠加迭代次数 (已在上面)
        # 38. pjgl - Lp范数指数 (已在上面)
        # 39. stkwb - 叠加窗口起始时间 (已在上面)
        
        # 说明文本（使用与所有其他文字相同的字体）
        info_text = """说明：
- 自适应叠加用于提高信号质量，通过多次迭代对齐和叠加
- Lp范数指数：L1对异常值更鲁棒，L2是标准最小二乘，L3对强信号更敏感
- 叠加窗口：定义用于对齐的时间窗口
- 差分时间搜索范围：允许的最大时间偏移
- Hilbert变换比例因子：相位一致性加权因子"""
        info_label = ttk.Label(scrollable_frame, text=info_text, justify='left', font=self.font_normal)
        info_label.grid(row=row, column=0, columnspan=3, sticky='w', padx=5, pady=10)
    
    def create_hilbert_tab(self, parent):
        """创建Hilbert变换参数标签页"""
        row = 0
        
        # Hilbert变换模式
        ttk.Label(parent, text='Hilbert变换模式 (ihilbt):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.ihilbt_var = tk.IntVar(value=self.params.ihilbt)
        ttk.Radiobutton(parent, text='无 (0)', variable=self.ihilbt_var, value=0).grid(row=row, column=1, sticky='w', padx=5)
        ttk.Radiobutton(parent, text='输出相位 (1)', variable=self.ihilbt_var, value=1).grid(row=row, column=2, sticky='w', padx=5)
        ttk.Radiobutton(parent, text='输出包络 (2)', variable=self.ihilbt_var, value=2).grid(row=row, column=3, sticky='w', padx=5)
        row += 1
        
        # 说明文本（使用与所有其他文字相同的字体）
        info_text = """说明：
- 无 (0): 不进行Hilbert变换
- 输出相位 (1): 输出信号的瞬时相位
- 输出包络 (2): 输出信号的包络（振幅）"""
        info_label = ttk.Label(parent, text=info_text, justify='left', font=self.font_normal)
        info_label.grid(row=row, column=0, columnspan=4, sticky='w', padx=5, pady=10)
    
    def create_color_tab(self, parent):
        """创建颜色设置标签页"""
        # 使用滚动框架
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        row = 0
        
        # 拾取颜色数组 (pickc) - 40个值
        pickc_label = ttk.Label(scrollable_frame, text='拾取颜色数组 (pickc, 40个值):')
        pickc_label.configure(font=self.font_bold)
        pickc_label.grid(row=row, column=0, columnspan=5, sticky='w', padx=5, pady=5)
        row += 1
        
        # 创建文本区域用于编辑数组
        ttk.Label(scrollable_frame, text='输入40个整数，用逗号或空格分隔:').grid(row=row, column=0, columnspan=5, sticky='w', padx=5, pady=2)
        row += 1
        
        self.pickc_text = tk.Text(scrollable_frame, width=60, height=3)
        pickc_str = ', '.join(map(str, self.params.pickc))
        self.pickc_text.insert('1.0', pickc_str)
        self.pickc_text.grid(row=row, column=0, columnspan=5, sticky='ew', padx=5, pady=5)
        row += 1
        
        ttk.Button(scrollable_frame, text='重置为默认值', 
                  command=self.reset_pickc).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        row += 1
        
        ttk.Separator(scrollable_frame, orient='horizontal').grid(row=row, column=0, columnspan=5, sticky='ew', padx=5, pady=10)
        row += 1
        
        # 颜色设置数组 (colour) - 5个值
        colour_label = ttk.Label(scrollable_frame, text='颜色设置数组 (colour, 5个值):')
        colour_label.configure(font=self.font_bold)
        colour_label.grid(row=row, column=0, columnspan=5, sticky='w', padx=5, pady=5)
        row += 1
        
        ttk.Label(scrollable_frame, text='输入5个整数，用逗号或空格分隔:').grid(row=row, column=0, columnspan=5, sticky='w', padx=5, pady=2)
        row += 1
        
        self.colour_text = tk.Text(scrollable_frame, width=60, height=2)
        colour_str = ', '.join(map(str, self.params.colour))
        self.colour_text.insert('1.0', colour_str)
        self.colour_text.grid(row=row, column=0, columnspan=5, sticky='ew', padx=5, pady=5)
        row += 1
        
        ttk.Button(scrollable_frame, text='重置为默认值', 
                  command=self.reset_colour).grid(row=row, column=0, sticky='w', padx=5, pady=2)
        row += 1
        
        # 配置列权重
        scrollable_frame.columnconfigure(0, weight=1)
    
    def reset_pickc(self):
        """重置pickc为默认值"""
        default_pickc = list(range(2, 42))
        self.pickc_text.delete('1.0', tk.END)
        self.pickc_text.insert('1.0', ', '.join(map(str, default_pickc)))
    
    def reset_colour(self):
        """重置colour为默认值"""
        default_colour = [4, 3, 6, 4, 5]
        self.colour_text.delete('1.0', tk.END)
        self.colour_text.insert('1.0', ', '.join(map(str, default_colour)))
    
    def parse_array(self, text: str, expected_length: int, default_value: list) -> list:
        """解析数组文本
        
        Args:
            text: 输入的文本
            expected_length: 期望的长度
            default_value: 默认值
            
        Returns:
            解析后的数组
        """
        try:
            # 替换逗号为空格，然后分割
            text = text.replace(',', ' ').replace(';', ' ')
            values = [int(x.strip()) for x in text.split() if x.strip()]
            
            if len(values) == expected_length:
                return values
            elif len(values) < expected_length:
                # 如果值不够，用默认值填充
                return values + default_value[len(values):]
            else:
                # 如果值太多，截断
                return values[:expected_length]
        except ValueError:
            messagebox.showerror('错误', f'无法解析数组，请确保输入的是整数')
            return default_value
    
    def on_ok(self):
        """确定按钮"""
        try:
            # 更新基本控制参数 (1-5)
            if hasattr(self, 'icomp_entry'):
                self.params.icomp = int(self.icomp_entry.get())
            if hasattr(self, 'iscreen_var'):
                self.params.iscreen = self.iscreen_var.get()
            if hasattr(self, 'iout_var'):
                self.params.iout = self.iout_var.get()
            if hasattr(self, 'idot_var'):
                self.params.idot = self.idot_var.get()
            if hasattr(self, 'itemp_var'):
                self.params.itemp = self.itemp_var.get()
            
            # 更新显示位置参数 (8-14)
            if hasattr(self, 'xpick_entry'):
                self.params.xpick = float(self.xpick_entry.get())
            if hasattr(self, 'annht_entry'):
                self.params.annht = float(self.annht_entry.get())
            if hasattr(self, 'space_entry'):
                self.params.space = float(self.space_entry.get())
            if hasattr(self, 'ndeca_entry'):
                self.params.ndeca = int(self.ndeca_entry.get())
            if hasattr(self, 'title_entry'):
                self.params.title = self.title_entry.get()
            if hasattr(self, 'overlap_entry'):
                overlap_val = float(self.overlap_entry.get())
                self.params.overlap = max(0.0, min(90.0, overlap_val))  # 限制在0-90
            if hasattr(self, 'dens_entry'):
                self.params.dens = float(self.dens_entry.get())
            
            # 更新能量和静音参数 (16-24)
            if hasattr(self, 'tcrcor_entry'):
                self.params.tcrcor = float(self.tcrcor_entry.get())
            if hasattr(self, 'tlag_entry'):
                self.params.tlag = float(self.tlag_entry.get())
            if hasattr(self, 'tempwin_entry'):
                self.params.tempwin = float(self.tempwin_entry.get())
            if hasattr(self, 'twinener_entry'):
                self.params.twinener = float(self.twinener_entry.get())
            if hasattr(self, 'minenratio_entry'):
                self.params.minenratio = float(self.minenratio_entry.get())
            if hasattr(self, 'vmute_entry'):
                self.params.vmute = float(self.vmute_entry.get())
            if hasattr(self, 'tmute_entry'):
                self.params.tmute = float(self.tmute_entry.get())
            if hasattr(self, 'nout_entry'):
                self.params.nout = int(self.nout_entry.get())
            if hasattr(self, 'nbout_entry'):
                self.params.nbout = int(self.nbout_entry.get())
            
            # 更新轴和布局参数 (25-35, 40)
            if hasattr(self, 'albht_entry'):
                self.params.albht = float(self.albht_entry.get())
            if hasattr(self, 'amenht_entry'):
                self.params.amenht = float(self.amenht_entry.get())
            if hasattr(self, 'orig_entry'):
                self.params.orig = float(self.orig_entry.get())
            if hasattr(self, 'itrev_var'):
                self.params.itrev = self.itrev_var.get()
            if hasattr(self, 'fmax_entry'):
                self.params.fmax = float(self.fmax_entry.get())
            if hasattr(self, 'tvg_entry'):
                self.params.tvg = float(self.tvg_entry.get())
            if hasattr(self, 'pvg_entry'):
                self.params.pvg = float(self.pvg_entry.get())
            if hasattr(self, 'iann_var'):
                self.params.iann = self.iann_var.get()
            if hasattr(self, 'thick_entry'):
                self.params.thick = float(self.thick_entry.get())
            if hasattr(self, 'ndiga_entry'):
                self.params.ndiga = int(self.ndiga_entry.get())
            if hasattr(self, 'shadedc_entry'):
                self.params.shadedc = float(self.shadedc_entry.get())
            if hasattr(self, 'txadj_entry'):
                self.params.txadj = float(self.txadj_entry.get())
            
            # 更新自适应叠加参数 (36-39)
            self.params.iadaptive = self.iadaptive_var.get()
            self.params.nsi = int(self.nsi_entry.get())
            self.params.pjgl = self.pjgl_var.get()
            self.params.stkwb = float(self.stkwb_entry.get())
            self.params.stkwl = float(self.stkwl_entry.get())
            self.params.dtcw = float(self.dtcw_entry.get())
            self.params.hilbratio = float(self.hilbratio_entry.get())
            
            # 更新Hilbert变换参数
            self.params.ihilbt = self.ihilbt_var.get()
            
            # 更新颜色设置 (6-7)
            pickc_text = self.pickc_text.get('1.0', tk.END).strip()
            default_pickc = list(range(2, 42))
            self.params.pickc = self.parse_array(pickc_text, 40, default_pickc)
            
            colour_text = self.colour_text.get('1.0', tk.END).strip()
            default_colour = [4, 3, 6, 4, 5]
            self.params.colour = self.parse_array(colour_text, 5, default_colour)
            
            # 调用回调函数
            if self.callback:
                self.callback()
            
            self.dialog.destroy()
            messagebox.showinfo('成功', '参数已更新')
            
        except ValueError as e:
            messagebox.showerror('错误', f'参数值无效: {str(e)}')
    
    def on_cancel(self):
        """取消按钮"""
        self.dialog.destroy()
    
    def on_reset(self):
        """重置为默认值"""
        if messagebox.askyesno('确认', '确定要重置所有参数为默认值吗？'):
            # 重置自适应叠加参数
            self.iadaptive_var.set(0)
            self.nsi_entry.delete(0, tk.END)
            self.nsi_entry.insert(0, '10')
            self.pjgl_var.set(3)
            self.stkwb_entry.delete(0, tk.END)
            self.stkwb_entry.insert(0, '0.04')
            self.stkwl_entry.delete(0, tk.END)
            self.stkwl_entry.insert(0, '0.5')
            self.dtcw_entry.delete(0, tk.END)
            self.dtcw_entry.insert(0, '0.1')
            self.hilbratio_entry.delete(0, tk.END)
            self.hilbratio_entry.insert(0, '3.0')
            
            # 重置Hilbert变换参数
            self.ihilbt_var.set(0)
            
            # 重置颜色设置
            self.reset_pickc()
            self.reset_colour()
    
    def create_basic_tab(self, parent):
        """创建基本控制参数标签页 (1-5)"""
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        row = 0
        
        # 1. icomp - 组件选择
        ttk.Label(scrollable_frame, text='1. icomp (组件选择):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.icomp_entry = ttk.Entry(scrollable_frame, width=15)
        self.icomp_entry.insert(0, str(self.params.icomp))
        self.icomp_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 999=全部)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 2. iscreen - 屏幕输出
        ttk.Label(scrollable_frame, text='2. iscreen (屏幕输出):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.iscreen_var = tk.IntVar(value=self.params.iscreen)
        ttk.Radiobutton(scrollable_frame, text='否 (0)', variable=self.iscreen_var, value=0).grid(row=row, column=1, sticky='w', padx=5)
        ttk.Radiobutton(scrollable_frame, text='是 (1)', variable=self.iscreen_var, value=1).grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 3. iout - 输出控制
        ttk.Label(scrollable_frame, text='3. iout (输出控制):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.iout_var = tk.IntVar(value=self.params.iout)
        ttk.Radiobutton(scrollable_frame, text='无 (0)', variable=self.iout_var, value=0).grid(row=row, column=1, sticky='w', padx=5)
        ttk.Radiobutton(scrollable_frame, text='输出到文件 (1)', variable=self.iout_var, value=1).grid(row=row, column=2, sticky='w', padx=5)
        ttk.Radiobutton(scrollable_frame, text='不去直流 (2)', variable=self.iout_var, value=2).grid(row=row, column=3, sticky='w', padx=5)
        row += 1
        
        # 4. idot - 点显示
        ttk.Label(scrollable_frame, text='4. idot (点显示):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.idot_var = tk.IntVar(value=self.params.idot)
        ttk.Radiobutton(scrollable_frame, text='否 (0)', variable=self.idot_var, value=0).grid(row=row, column=1, sticky='w', padx=5)
        ttk.Radiobutton(scrollable_frame, text='是 (1)', variable=self.idot_var, value=1).grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 5. itemp - 温度/时间相关
        ttk.Label(scrollable_frame, text='5. itemp (温度/时间相关):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.itemp_var = tk.IntVar(value=self.params.itemp)
        ttk.Radiobutton(scrollable_frame, text='否 (0)', variable=self.itemp_var, value=0).grid(row=row, column=1, sticky='w', padx=5)
        ttk.Radiobutton(scrollable_frame, text='是 (1)', variable=self.itemp_var, value=1).grid(row=row, column=2, sticky='w', padx=5)
        row += 1
    
    def create_display_position_tab(self, parent):
        """创建显示位置参数标签页 (8-12)"""
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        row = 0
        
        # 8. xpick - 拾取X偏移
        ttk.Label(scrollable_frame, text='8. xpick (拾取X偏移, 毫米):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.xpick_entry = ttk.Entry(scrollable_frame, width=15)
        self.xpick_entry.insert(0, str(self.params.xpick))
        self.xpick_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 0.0)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 9. annht - 注释高度
        ttk.Label(scrollable_frame, text='9. annht (注释高度, 毫米):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.annht_entry = ttk.Entry(scrollable_frame, width=15)
        self.annht_entry.insert(0, str(self.params.annht))
        self.annht_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 0.0)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 10. space - 间距
        ttk.Label(scrollable_frame, text='10. space (间距, 毫米):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.space_entry = ttk.Entry(scrollable_frame, width=15)
        self.space_entry.insert(0, str(self.params.space))
        self.space_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 0.0)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 11. ndeca - 小数位数
        ttk.Label(scrollable_frame, text='11. ndeca (小数位数):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.ndeca_entry = ttk.Entry(scrollable_frame, width=15)
        self.ndeca_entry.insert(0, str(self.params.ndeca))
        self.ndeca_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 1)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 12. title - 标题
        ttk.Label(scrollable_frame, text='12. title (标题):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.title_entry = ttk.Entry(scrollable_frame, width=40)
        self.title_entry.insert(0, str(self.params.title))
        self.title_entry.grid(row=row, column=1, columnspan=2, sticky='ew', padx=5, pady=5)
        row += 1
        
        # 13. overlap - 重叠
        ttk.Label(scrollable_frame, text='13. overlap (重叠百分比):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.overlap_entry = ttk.Entry(scrollable_frame, width=15)
        self.overlap_entry.insert(0, str(self.params.overlap))
        self.overlap_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 10.0, 范围: 0-90)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 14. dens - 密度
        ttk.Label(scrollable_frame, text='14. dens (密度):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.dens_entry = ttk.Entry(scrollable_frame, width=15)
        self.dens_entry.insert(0, str(self.params.dens))
        self.dens_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 0.0=自动)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        scrollable_frame.columnconfigure(1, weight=1)
    
    def create_energy_tab(self, parent):
        """创建能量和静音参数标签页 (16-24)"""
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        row = 0
        
        # 16. tcrcor - 互相关窗口长度
        ttk.Label(scrollable_frame, text='16. tcrcor (互相关窗口长度, 秒):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.tcrcor_entry = ttk.Entry(scrollable_frame, width=15)
        self.tcrcor_entry.insert(0, str(self.params.tcrcor))
        self.tcrcor_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 0.60)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 17. tlag - 时间延迟
        ttk.Label(scrollable_frame, text='17. tlag (时间延迟搜索范围, 秒):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.tlag_entry = ttk.Entry(scrollable_frame, width=15)
        self.tlag_entry.insert(0, str(self.params.tlag))
        self.tlag_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 0.10)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 18. tempwin - 临时窗口
        ttk.Label(scrollable_frame, text='18. tempwin (临时窗口, 秒):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.tempwin_entry = ttk.Entry(scrollable_frame, width=15)
        self.tempwin_entry.insert(0, str(self.params.tempwin))
        self.tempwin_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 1.0)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 19. twinener - 时间窗口能量
        ttk.Label(scrollable_frame, text='19. twinener (时间窗口能量, 秒):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.twinener_entry = ttk.Entry(scrollable_frame, width=15)
        self.twinener_entry.insert(0, str(self.params.twinener))
        self.twinener_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 0.1)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 20. minenratio - 最小能量比
        ttk.Label(scrollable_frame, text='20. minenratio (最小能量比):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.minenratio_entry = ttk.Entry(scrollable_frame, width=15)
        self.minenratio_entry.insert(0, str(self.params.minenratio))
        self.minenratio_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 10.0)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 21. vmute - 静音速度
        ttk.Label(scrollable_frame, text='21. vmute (静音速度, km/s):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.vmute_entry = ttk.Entry(scrollable_frame, width=15)
        self.vmute_entry.insert(0, str(self.params.vmute))
        self.vmute_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 0.0)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 22. tmute - 静音时间
        ttk.Label(scrollable_frame, text='22. tmute (静音时间, 秒):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.tmute_entry = ttk.Entry(scrollable_frame, width=15)
        self.tmute_entry.insert(0, str(self.params.tmute))
        self.tmute_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 0.0)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 23. nout - 输出点数
        ttk.Label(scrollable_frame, text='23. nout (输出点数):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.nout_entry = ttk.Entry(scrollable_frame, width=15)
        self.nout_entry.insert(0, str(self.params.nout))
        self.nout_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 0=全部)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 24. nbout - 输出前点数
        ttk.Label(scrollable_frame, text='24. nbout (输出前点数):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.nbout_entry = ttk.Entry(scrollable_frame, width=15)
        self.nbout_entry.insert(0, str(self.params.nbout))
        self.nbout_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 0)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        scrollable_frame.columnconfigure(1, weight=1)
    
    def create_axis_tab(self, parent):
        """创建轴和布局参数标签页 (25-35)"""
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        row = 0
        
        # 25. albht - 轴标签高度
        ttk.Label(scrollable_frame, text='25. albht (轴标签高度, 毫米):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.albht_entry = ttk.Entry(scrollable_frame, width=15)
        self.albht_entry.insert(0, str(self.params.albht))
        self.albht_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 2.5)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 26. amenht - 菜单高度
        ttk.Label(scrollable_frame, text='26. amenht (菜单高度, 毫米):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.amenht_entry = ttk.Entry(scrollable_frame, width=15)
        self.amenht_entry.insert(0, str(self.params.amenht))
        self.amenht_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: -1.0=自动)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 27. orig - 原点位置
        ttk.Label(scrollable_frame, text='27. orig (原点位置, 毫米):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.orig_entry = ttk.Entry(scrollable_frame, width=15)
        self.orig_entry.insert(0, str(self.params.orig))
        self.orig_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 12.5)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 28. itrev - 时间反转
        ttk.Label(scrollable_frame, text='28. itrev (时间反转):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.itrev_var = tk.IntVar(value=self.params.itrev)
        ttk.Radiobutton(scrollable_frame, text='否 (0)', variable=self.itrev_var, value=0).grid(row=row, column=1, sticky='w', padx=5)
        ttk.Radiobutton(scrollable_frame, text='是 (1)', variable=self.itrev_var, value=1).grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 29. fmax - 最大频率
        ttk.Label(scrollable_frame, text='29. fmax (最大频率, Hz):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.fmax_entry = ttk.Entry(scrollable_frame, width=15)
        self.fmax_entry.insert(0, str(self.params.fmax))
        self.fmax_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: -1.0=自动)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 30. tvg - 时间变化增益
        ttk.Label(scrollable_frame, text='30. tvg (时间变化增益):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.tvg_entry = ttk.Entry(scrollable_frame, width=15)
        self.tvg_entry.insert(0, str(self.params.tvg))
        self.tvg_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 1.0)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 31. pvg - 功率变化增益
        ttk.Label(scrollable_frame, text='31. pvg (功率变化增益):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.pvg_entry = ttk.Entry(scrollable_frame, width=15)
        self.pvg_entry.insert(0, str(self.params.pvg))
        self.pvg_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 1.0)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 32. iann - 注释索引
        ttk.Label(scrollable_frame, text='32. iann (注释索引):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.iann_var = tk.IntVar(value=self.params.iann)
        ttk.Radiobutton(scrollable_frame, text='无 (0)', variable=self.iann_var, value=0).grid(row=row, column=1, sticky='w', padx=5)
        ttk.Radiobutton(scrollable_frame, text='显示 (>0)', variable=self.iann_var, value=1).grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 33. thick - 厚度
        ttk.Label(scrollable_frame, text='33. thick (厚度):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.thick_entry = ttk.Entry(scrollable_frame, width=15)
        self.thick_entry.insert(0, str(self.params.thick))
        self.thick_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: -1.0=自动)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 34. ndiga - 数字精度
        ttk.Label(scrollable_frame, text='34. ndiga (数字精度):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.ndiga_entry = ttk.Entry(scrollable_frame, width=15)
        self.ndiga_entry.insert(0, str(self.params.ndiga))
        self.ndiga_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: -1=自动)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 35. shadedc - 阴影颜色
        ttk.Label(scrollable_frame, text='35. shadedc (阴影颜色):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.shadedc_entry = ttk.Entry(scrollable_frame, width=15)
        self.shadedc_entry.insert(0, str(self.params.shadedc))
        self.shadedc_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 0.0)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        # 40. txadj - 时间X调整
        ttk.Label(scrollable_frame, text='40. txadj (时间X调整, 秒):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.txadj_entry = ttk.Entry(scrollable_frame, width=15)
        self.txadj_entry.insert(0, str(self.params.txadj))
        self.txadj_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        ttk.Label(scrollable_frame, text='(默认: 0.0)').grid(row=row, column=2, sticky='w', padx=5)
        row += 1
        
        scrollable_frame.columnconfigure(1, weight=1)
