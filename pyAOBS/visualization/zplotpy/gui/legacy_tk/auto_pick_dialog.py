"""
auto_pick_dialog.py - 自动拾取参数设置对话框
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable, Dict, Any


class AutoPickDialog:
    """自动拾取参数设置对话框"""
    
    def __init__(self, parent, default_params: Optional[Dict[str, Any]] = None):
        """初始化对话框
        
        Args:
            parent: 父窗口
            default_params: 默认参数字典
        """
        self.parent = parent
        self.result = None
        
        # 默认参数
        self.default_params = default_params or {}
        
        # 创建对话框窗口
        self.dialog = tk.Toplevel(parent)
        self.dialog.title('自动拾取设置')
        self.dialog.geometry('400x350')
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # 居中显示
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (350 // 2)
        self.dialog.geometry(f'400x350+{x}+{y}')
        
        # 创建界面
        self.create_widgets()
        
        # 绑定回车键
        self.dialog.bind('<Return>', lambda e: self.on_ok())
    
    def create_widgets(self):
        """创建界面组件"""
        # 主框架
        main_frame = ttk.Frame(self.dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 参数设置区域
        params_frame = ttk.LabelFrame(main_frame, text='拾取参数', padding=10)
        params_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        row = 0
        
        # 窗口长度
        ttk.Label(params_frame, text='窗口长度 (秒):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.window_length_var = tk.StringVar(value=str(self.default_params.get('window_length', 0.1)))
        window_length_entry = ttk.Entry(params_frame, textvariable=self.window_length_var, width=15)
        window_length_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        row += 1
        
        # 最小能量比
        ttk.Label(params_frame, text='最小能量比:').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.min_energy_ratio_var = tk.StringVar(value=str(self.default_params.get('min_energy_ratio', 1.5)))
        min_energy_ratio_entry = ttk.Entry(params_frame, textvariable=self.min_energy_ratio_var, width=15)
        min_energy_ratio_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        row += 1
        
        # 搜索时间范围
        ttk.Label(params_frame, text='搜索时间范围:').grid(row=row, column=0, columnspan=2, sticky='w', padx=5, pady=(10, 5))
        row += 1
        
        ttk.Label(params_frame, text='起始时间 (秒):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.search_start_var = tk.StringVar(value=str(self.default_params.get('search_start', '')))
        search_start_entry = ttk.Entry(params_frame, textvariable=self.search_start_var, width=15)
        search_start_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        row += 1
        
        ttk.Label(params_frame, text='结束时间 (秒):').grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.search_end_var = tk.StringVar(value=str(self.default_params.get('search_end', '')))
        search_end_entry = ttk.Entry(params_frame, textvariable=self.search_end_var, width=15)
        search_end_entry.grid(row=row, column=1, sticky='w', padx=5, pady=5)
        row += 1
        
        # 拾取范围
        range_frame = ttk.LabelFrame(main_frame, text='拾取范围', padding=10)
        range_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.range_all_var = tk.BooleanVar(value=True)
        self.range_visible_var = tk.BooleanVar(value=False)
        self.range_custom_var = tk.BooleanVar(value=False)
        
        ttk.Radiobutton(range_frame, text='所有道', variable=self.range_all_var, 
                       value=True, command=self.on_range_change).pack(anchor='w', pady=2)
        ttk.Radiobutton(range_frame, text='当前显示的道', variable=self.range_visible_var,
                       value=True, command=self.on_range_change).pack(anchor='w', pady=2)
        ttk.Radiobutton(range_frame, text='指定道范围', variable=self.range_custom_var,
                       value=True, command=self.on_range_change).pack(anchor='w', pady=2)
        
        # 道范围输入
        custom_frame = ttk.Frame(range_frame)
        custom_frame.pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Label(custom_frame, text='从道:').pack(side=tk.LEFT, padx=5)
        self.trace_start_var = tk.StringVar(value='1')
        self.trace_start_entry = ttk.Entry(custom_frame, textvariable=self.trace_start_var, width=10, state='disabled')
        self.trace_start_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(custom_frame, text='到道:').pack(side=tk.LEFT, padx=5)
        self.trace_end_var = tk.StringVar(value='100')
        self.trace_end_entry = ttk.Entry(custom_frame, textvariable=self.trace_end_var, width=10, state='disabled')
        self.trace_end_entry.pack(side=tk.LEFT, padx=5)
        
        # 拾取字
        ttk.Label(main_frame, text='拾取字:').pack(anchor='w', padx=5, pady=5)
        self.pick_word_var = tk.StringVar(value=str(self.default_params.get('pick_word', 1)))
        pick_word_entry = ttk.Entry(main_frame, textvariable=self.pick_word_var, width=15)
        pick_word_entry.pack(anchor='w', padx=20, pady=(0, 10))
        
        # 按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text='预览', command=self.on_preview, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text='应用', command=self.on_ok, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text='取消', command=self.on_cancel, width=10).pack(side=tk.LEFT, padx=5)
        
        # 更新自定义道范围的启用状态
        self.on_range_change()
    
    def on_range_change(self):
        """拾取范围改变时的回调"""
        # 根据选择的选项启用/禁用道范围输入
        # 存储道范围输入控件的引用
        if not hasattr(self, 'trace_start_entry'):
            # 查找道范围输入控件
            for widget in self.dialog.winfo_children():
                if isinstance(widget, ttk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.LabelFrame) and child.cget('text') == '拾取范围':
                            for grandchild in child.winfo_children():
                                if isinstance(grandchild, ttk.Frame):
                                    for entry in grandchild.winfo_children():
                                        if isinstance(entry, ttk.Entry):
                                            if not hasattr(self, 'trace_start_entry'):
                                                self.trace_start_entry = entry
                                            else:
                                                self.trace_end_entry = entry
        
        # 启用/禁用道范围输入
        if hasattr(self, 'trace_start_entry') and hasattr(self, 'trace_end_entry'):
            if self.range_custom_var.get():
                self.trace_start_entry.config(state='normal')
                self.trace_end_entry.config(state='normal')
            else:
                self.trace_start_entry.config(state='disabled')
                self.trace_end_entry.config(state='disabled')
    
    def on_preview(self):
        """预览按钮回调"""
        # 验证参数
        params = self.get_params()
        if params is None:
            return
        
        # 返回预览模式
        self.result = {'action': 'preview', 'params': params}
        self.dialog.destroy()
    
    def on_ok(self):
        """确定按钮回调"""
        # 验证参数
        params = self.get_params()
        if params is None:
            return
        
        # 返回应用模式
        self.result = {'action': 'apply', 'params': params}
        self.dialog.destroy()
    
    def on_cancel(self):
        """取消按钮回调"""
        self.result = None
        self.dialog.destroy()
    
    def get_params(self) -> Optional[Dict[str, Any]]:
        """获取参数并验证
        
        Returns:
            参数字典，如果验证失败返回None
        """
        try:
            window_length = float(self.window_length_var.get())
            if window_length <= 0:
                raise ValueError("窗口长度必须大于0")
            
            min_energy_ratio = float(self.min_energy_ratio_var.get())
            if min_energy_ratio <= 0:
                raise ValueError("最小能量比必须大于0")
            
            search_start = None
            search_start_str = self.search_start_var.get().strip()
            if search_start_str:
                search_start = float(search_start_str)
            
            search_end = None
            search_end_str = self.search_end_var.get().strip()
            if search_end_str:
                search_end = float(search_end_str)
            
            if search_start is not None and search_end is not None:
                if search_start >= search_end:
                    raise ValueError("起始时间必须小于结束时间")
            
            # 拾取范围
            if self.range_all_var.get():
                range_type = 'all'
                trace_start = None
                trace_end = None
            elif self.range_visible_var.get():
                range_type = 'visible'
                trace_start = None
                trace_end = None
            else:
                range_type = 'custom'
                trace_start = int(self.trace_start_var.get())
                trace_end = int(self.trace_end_var.get())
                if trace_start < 1 or trace_end < trace_start:
                    raise ValueError("道范围无效")
            
            pick_word = int(self.pick_word_var.get())
            if pick_word < 1:
                raise ValueError("拾取字必须大于0")
            
            return {
                'window_length': window_length,
                'min_energy_ratio': min_energy_ratio,
                'search_start': search_start,
                'search_end': search_end,
                'range_type': range_type,
                'trace_start': trace_start,
                'trace_end': trace_end,
                'pick_word': pick_word
            }
        except ValueError as e:
            import tkinter.messagebox as messagebox
            messagebox.showerror('参数错误', f'参数验证失败：{str(e)}')
            return None
    
    def show(self) -> Optional[Dict[str, Any]]:
        """显示对话框并返回结果
        
        Returns:
            结果字典，包含action和params，如果取消则返回None
        """
        self.dialog.wait_window()
        return self.result
