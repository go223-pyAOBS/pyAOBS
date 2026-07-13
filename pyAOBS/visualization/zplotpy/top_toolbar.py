"""
top_toolbar.py - 顶部工具栏类

实现可折叠的顶部工具栏，包含文件、显示、滤波、拾取等参数控制标签页
包含所有32个菜单参数的控制
"""

import tkinter as tk
from tkinter import ttk, font as tkfont
from typing import Optional, Callable, Dict, Any
import platform
import os
import sys

try:
    from .parameters import ZPlotParameters
except ImportError:
    from parameters import ZPlotParameters


# 缓存字体检测结果，避免重复检测
_chinese_font_cache = None


def _scaled_ui_size(size: int) -> int:
    """根据环境变量对 UI 字号做统一缩放。"""
    scale_raw = os.environ.get("PYAOBS_UI_FONT_SCALE", "").strip()
    if not scale_raw:
        return size
    try:
        scale = float(scale_raw)
    except ValueError:
        return size
    # 防止传入异常值导致界面不可用
    scale = min(max(scale, 0.6), 3.0)
    return max(8, int(round(size * scale)))

def get_chinese_font(size=13, weight='normal'):
    """获取支持中文的字体
    
    Args:
        size: 字体大小
        weight: 字体粗细 ('normal', 'bold')
    
    Returns:
        字体元组（用于 ttk 样式）或 Font 对象（用于 tk 组件）
    """
    global _chinese_font_cache
    size = _scaled_ui_size(size)
    forced_font = os.environ.get("PYAOBS_UI_FONT_FAMILY", "").strip()
    
    # 如果环境变量强制指定字体，优先使用
    if forced_font:
        if weight == 'bold':
            return (forced_font, size, 'bold')
        return (forced_font, size)

    # 如果已经缓存，直接返回
    if _chinese_font_cache is not None:
        font_name = _chinese_font_cache
        if weight == 'bold':
            return (font_name, size, 'bold')
        else:
            return (font_name, size)
    
    # Windows 判定使用多重条件，避免某些环境 platform.system() 异常
    system = platform.system()
    is_windows = (os.name == 'nt') or sys.platform.startswith('win') or system == 'Windows'
    
    # 根据系统选择候选字体列表
    if is_windows:
        # Windows 下优先使用常见中文字体（含中英文名称）
        chinese_fonts = [
            'Microsoft YaHei UI',
            'Microsoft YaHei',
            '微软雅黑 UI',
            '微软雅黑',
            'SimHei',
            '黑体',
            'SimSun',
            '宋体',
            'KaiTi',
            '楷体',
            'NSimSun',
            '新宋体',
            'FangSong',
            '仿宋',
        ]
    elif system == 'Darwin':  # macOS
        chinese_fonts = ['PingFang SC', 'STHeiti', 'STSong', 'Hiragino Sans GB']
    else:  # Linux
        chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 
                        'Source Han Sans CN', 'Droid Sans Fallback']
    
    font_name = chinese_fonts[0]  # 默认字体
    found_font = None
    
    # 验证字体是否可用
    try:
        import tkinter.font as tkfont
        all_fonts = list(tkfont.families())
        available_fonts_lower = [f.lower() for f in all_fonts]  # 转换为小写以便比较
        
        # 尝试匹配字体（不区分大小写）
        for font in chinese_fonts:
            font_lower = font.lower()
            for avail_font in all_fonts:
                if avail_font.lower() == font_lower:
                    font_name = avail_font  # 使用原始大小写
                    found_font = font_name
                    break
            if found_font:
                break
        
        # 如果没找到，尝试模糊匹配
        if not found_font:
            if is_windows:
                keywords = ['yahei', '微软雅黑', 'simhei', '黑体', 'simsun', '宋体', 'kaiti', '楷体', 'fangsong', '仿宋']
            elif system == 'Darwin':
                keywords = ['pingfang', 'heiti', 'stsong', 'hiragino', 'noto']
            else:
                keywords = ['wenquanyi', 'noto', 'source han', 'droid', 'pingfang']
            for font in all_fonts:
                font_lower = font.lower()
                if any(str(keyword).lower() in font_lower for keyword in keywords):
                    font_name = font
                    found_font = font
                    break
        
        # 如果还是没找到，尝试使用支持 Unicode 的通用字体
        if not found_font:
            # Windows 下优先保证中文字体，不在此阶段回退到 DejaVu；
            # Linux/macOS 再尝试 Unicode 通用字体。
            if not is_windows:
                unicode_fonts = ['Noto Sans CJK SC', 'Noto Sans', 'DejaVu Sans', 'Liberation Sans', 'Sans']
                for ufont in unicode_fonts:
                    for avail_font in all_fonts:
                        if ufont.lower() in avail_font.lower():
                            font_name = avail_font
                            found_font = font_name
                            break
                    if found_font:
                        break
        
        if not found_font:
            # 最后兜底：Windows 强制使用常见中文字体链；其他系统用通用 sans。
            if is_windows:
                if 'Microsoft YaHei UI' in all_fonts:
                    font_name = 'Microsoft YaHei UI'
                elif 'Microsoft YaHei' in all_fonts:
                    font_name = 'Microsoft YaHei'
                elif 'SimHei' in all_fonts:
                    font_name = 'SimHei'
                elif 'SimSun' in all_fonts:
                    font_name = 'SimSun'
                else:
                    font_name = 'Microsoft YaHei'
            else:
                try:
                    if 'sans' in available_fonts_lower:
                        idx = available_fonts_lower.index('sans')
                        font_name = all_fonts[idx]
                except:
                    pass
    
    except Exception as e:
        # 静默处理错误
        pass
    
    # 缓存结果
    _chinese_font_cache = font_name
    
    if weight == 'bold':
        return (font_name, size, 'bold')
    else:
        return (font_name, size)

def get_chinese_font_object(size=13, weight='normal'):
    """获取支持中文的 Font 对象
    
    Args:
        size: 字体大小
        weight: 字体粗细 ('normal', 'bold')
    
    Returns:
        tkinter Font 对象
    """
    # 复用统一字体探测逻辑，避免不同入口选出不同字体
    chosen = get_chinese_font(size=size, weight=weight)
    if isinstance(chosen, tuple):
        font_name = chosen[0]
        font_size = chosen[1] if len(chosen) > 1 else size
    else:
        font_name = chosen
        font_size = size

    font_obj = tkfont.Font(family=font_name, size=font_size)
    if weight == 'bold':
        font_obj.config(weight='bold')
    
    return font_obj


class TopToolbar:
    """顶部工具栏类 - 可折叠的参数控制工具栏"""
    DEFAULT_PANEL_ORDER = [
        '显示控制',
        '数据处理',
        '滤波设置',
        '显示参数',
        '拾取控制',
        '其他参数',
        '时间参数',
        '特殊功能',
    ]
    
    def __init__(self, parent, main_gui):
        """初始化顶部工具栏
        
        Args:
            parent: 父容器
            main_gui: 主窗口对象（用于回调）
        """
        self.parent = parent
        self.main_gui = main_gui
        self.expanded = True
        
        # 参数管理
        self.params = ZPlotParameters()
        
        # 存储所有控件的引用
        self.param_widgets: Dict[str, Any] = {}
        self._panel_labelframes = []
        self._panel_frames: Dict[str, ttk.LabelFrame] = {}
        self._panel_order = list(self.DEFAULT_PANEL_ORDER)
        self._panel_drag_source: Optional[str] = None
        self._panel_drag_started = False
        self._panel_grid_parent = None
        self.header_title_label = None
        self.shade_quality_label = None
        self._hover_help_param: Optional[str] = None
        self._shift_pressed = False
        self._param_help_texts: Dict[str, str] = {
            'iscale': '缩放模式：0自动，1固定，2变增益；控制波形振幅的显示方式。',
            'amp': '振幅缩放系数：放大或缩小波形振幅。',
            'rcor': '距离校正指数：按炮检距对振幅做补偿，常用于远距能量平衡。',
            'sf': '附加缩放因子：配合振幅显示做细调。',
            'irec': '记录号（炮集号）：选择当前要显示/处理的记录。',
            'itx': '理论走时显示开关：0关闭，1显示。',
            'imute': '静校正/静音模式开关：控制是否应用静校正相关处理。',
            'iwater': '水层校正开关：控制是否启用水层走时校正显示。',
            'ibndps': '带通滤波模式：0不滤波，1启用带通，-2用于频谱相关显示。',
            'izerop': '零相位滤波开关：1为零相位（不改相位），0为单向滤波。',
            'freqlo': '带通低截止频率（Hz）。',
            'freqhi': '带通高截止频率（Hz）。',
            'tlinc': '时间标注增量（s）：控制时间刻度标注间隔。',
            'clip': '裁剪阈值：限制振幅极值，抑制异常大振幅。',
            'vred': '折合速度（km/s）：用于折合时间显示 t - |x|/vred。',
            'ishade': '阴影填充：0不填充，正值填正峰，负值填负峰。',
            'spick': '拾取符号大小：控制拾取点标记尺寸。',
            'ixaxis': 'X轴类型：炮检距、模型位置、方位角、修正方位角或道号。',
            'itype': '数据类型过滤：选择垂直/径向/横向/水听器及组合显示。',
            'apick': '活动拾取字：当前进行拾取与编辑的拾取编号（1-40）。',
            'nskip': '抽道参数：按间隔跳过部分道以提高显示速度。',
            'xmm': 'X轴物理宽度参数（mm），影响版式与比例。',
            'xmin': '当前显示窗口 X 最小值。',
            'xmax': '当前显示窗口 X 最大值。',
            'ndecim': '时间采样抽取间隔：增大可提升速度，减小可保留细节。',
            'tmm': '时间轴物理高度参数（mm），影响版式与比例。',
            'tmin': '当前显示窗口时间最小值（s）。',
            'tmax': '当前显示窗口时间最大值（s）。',
        }
        self._param_help_texts_detailed: Dict[str, str] = {
            'iscale': '缩放模式影响振幅归一策略：自动适合快速浏览，固定便于道间对比，变增益适合弱信号增强。',
            'irec': '记录号用于切换炮集；拾取、滤波显示和自动处理都以当前记录为入口。',
            'ibndps': '带通滤波建议与freqlo/freqhi联动调整；实时交互中可能临时降级以保证帧率，停止交互后补绘。',
            'vred': '折合时间采用 t - |offset|/vred；较大vred折合弱，较小vred折合强，常用于凸显某类走时斜率。',
            'ishade': '阴影填充仅改变显示样式（正峰或负峰），不改变原始数据与拾取结果。',
            'itype': '数据类型可筛选分量（垂直/径向/横向/水听器及组合）；会影响显示和后续工作流可见道集合。',
            'apick': '活动拾取字决定当前新增/编辑的拾取类别；自动拾取与插值相关也按该拾取字写入结果。',
            'nskip': '抽道可明显提升绘图速度；值越大显示道越稀疏，建议先小值浏览再精细处理。',
            'xmin': '与xmax共同定义当前X窗口；平移缩放会实时更新该范围。',
            'xmax': '与xmin共同定义当前X窗口；建议先粗定位后再局部放大。',
            'tmin': '与tmax共同定义当前时间窗口；折合时间或静校正下建议关注窗口与数据是否重叠。',
            'tmax': '与tmin共同定义当前时间窗口；窗口过窄会减少可见波形与可处理道数。',
        }
        
        # 配置ttk样式以支持中文显示
        self.setup_chinese_fonts()
        
        # 创建工具栏容器
        self.container = ttk.Frame(parent)
        
        # 创建工具栏
        self.create_toolbar()
        self._install_hover_detail_keybinds()

    def _get_root_widget(self):
        """获取根窗口。"""
        root = self.parent
        while root and not isinstance(root, tk.Tk) and not isinstance(root, tk.Toplevel):
            try:
                root = root.master
            except Exception:
                break
        return root

    def _install_hover_detail_keybinds(self):
        """安装 Shift 键监听：悬停时切换简短/详细说明。"""
        root = self._get_root_widget()
        if root is None:
            return

        def _on_shift_press(_event=None):
            self._shift_pressed = True
            self._refresh_hover_help_status()

        def _on_shift_release(_event=None):
            self._shift_pressed = False
            self._refresh_hover_help_status()

        try:
            root.bind('<KeyPress-Shift_L>', _on_shift_press, add='+')
            root.bind('<KeyPress-Shift_R>', _on_shift_press, add='+')
            root.bind('<KeyRelease-Shift_L>', _on_shift_release, add='+')
            root.bind('<KeyRelease-Shift_R>', _on_shift_release, add='+')
        except Exception:
            pass
    
    def _apply_combobox_popup_font(self, combo: ttk.Combobox, font_name: str, font_size: int):
        """强制设置 Combobox 下拉列表字体（修复下拉项中文方框）。"""
        try:
            popdown = combo.tk.call("ttk::combobox::PopdownWindow", str(combo))
            listbox_path = f"{popdown}.f.l"
            combo.tk.call(listbox_path, "configure", "-font", f"{font_name} {font_size}")
        except Exception:
            # 不同平台 Tk 内部结构可能不同，失败则静默回退
            pass

    def _get_main_ui_size(self, default: int = 16) -> int:
        """获取主界面当前字号（用于运行时同步缩放）。"""
        try:
            size = int(getattr(self.main_gui, '_ui_font_size', default))
            return max(8, min(28, size))
        except Exception:
            return default

    def _get_ui_font(self, base_size: int = 16, weight: str = 'normal', title_boost: int = 0):
        """获取与主界面同步的字体（优先复用主窗口已确认可用字体）。"""
        current_size = self._get_main_ui_size(base_size)
        size = current_size + title_boost

        # 优先使用主窗口已选中的字体，避免再次探测导致“方框字”回归
        preferred_family = str(getattr(self.main_gui, '_ui_font_name', '') or '').strip()
        if preferred_family and preferred_family.lower() != 'fixed':
            if weight == 'bold':
                return (preferred_family, size, 'bold')
            return (preferred_family, size)

        # 兜底：从主窗口候选中挑一个 CJK 字体
        candidates = getattr(self.main_gui, '_cjk_font_candidates', []) or []
        for cand in candidates:
            cl = str(cand).lower()
            if any(k in cl for k in ('yahei', '微软雅黑', 'simhei', '黑体', 'simsun', '宋体', 'noto', 'cjk', 'wenquanyi', 'pingfang', 'heiti', 'song')):
                if weight == 'bold':
                    return (cand, size, 'bold')
                return (cand, size)

        return get_chinese_font(size, weight)

    def _create_panel_labelframe(self, parent, title: str):
        """创建顶部居中标题的参数面板。"""
        frame = ttk.LabelFrame(parent, text=title, labelanchor='n')
        self._panel_labelframes.append(frame)
        self._configure_labelframe_font(frame)
        return frame
    
    def setup_chinese_fonts(self):
        """配置ttk样式以支持中文显示"""
        style = ttk.Style()
        # 与主界面字号联动，面板标题额外 +1
        chinese_font = self._get_ui_font(16)
        panel_title_font = self._get_ui_font(16, title_boost=1)
        font_name = chinese_font[0] if isinstance(chinese_font, tuple) else chinese_font
        font_size = chinese_font[1] if isinstance(chinese_font, tuple) and len(chinese_font) > 1 else 12
        
        # 获取根窗口（通过 parent）
        root = self.parent
        while root and not isinstance(root, tk.Tk) and not isinstance(root, tk.Toplevel):
            try:
                root = root.master
            except:
                break
        
        if root:
            # 方法1: 设置 tkinter 默认字体
            try:
                default_font = tkfont.nametofont("TkDefaultFont")
                default_font.config(family=font_name, size=font_size)
            except:
                pass
            
            # 方法2: 使用 option_add 设置全局默认字体
            try:
                font_tuple = (font_name, font_size)
                root.option_add('*Font', font_tuple)
                root.option_add('*TkDefaultFont', font_tuple)
                # 关键：Combobox 下拉列表与 Menu 常见“方框字”问题
                root.option_add('*Menu.font', font_tuple)
                root.option_add('*Listbox.font', font_tuple)
                root.option_add('*TCombobox*Listbox.font', font_tuple)
            except:
                pass
        
        # 方法3: 为 ttk 组件设置样式
        try:
            style.configure('TLabel', font=chinese_font)
            style.configure('TButton', font=chinese_font)
            style.configure('TEntry', font=chinese_font)
            style.configure('TCombobox', font=chinese_font)
            style.configure('TSpinbox', font=chinese_font)
            style.configure('TCheckbutton', font=chinese_font)
            style.configure('TRadiobutton', font=chinese_font)
            style.configure('TMenubutton', font=chinese_font)
            style.configure('TNotebook.Tab', font=chinese_font)
            
            # LabelFrame 标题字体设置（关键：必须设置 TLabelframe.Label）
            style.configure('TLabelFrame', font=chinese_font)
            style.configure('TLabelframe.Label', font=panel_title_font)
            
            # 尝试设置所有默认样式
            style.configure('.', font=chinese_font)
        except Exception:
            pass
        
    def create_toolbar(self):
        """创建工具栏"""
        # 标题栏（始终可见）
        header = ttk.Frame(self.container, relief=tk.RAISED, borderwidth=1)
        header.pack(fill=tk.X, padx=2, pady=2)
        
        # 折叠/展开按钮
        self.toggle_btn = ttk.Button(header, text='▼', width=3,
                                    command=self.toggle)
        self.toggle_btn.pack(side=tk.LEFT, padx=2)
        
        # 标题
        self.header_title_label = ttk.Label(
            header,
            text='参数控制',
            font=self._get_ui_font(16, 'bold', title_boost=1)
        )
        self.header_title_label.pack(side=tk.LEFT, padx=5)
        
        # 应用所有参数按钮（始终可见）
        ttk.Button(header, text='应用所有参数', width=15,
                  command=self.apply_all_params).pack(side=tk.RIGHT, padx=5)

        # 阴影质量档位（快速/平衡/高质量）
        self.shade_quality_label = ttk.Label(
            header,
            text='阴影质量:',
            font=self._get_ui_font(16)
        )
        self.shade_quality_label.pack(side=tk.RIGHT, padx=(8, 2))
        self.shade_quality_var = tk.StringVar(value='平衡')
        self.shade_quality_combo = ttk.Combobox(
            header,
            textvariable=self.shade_quality_var,
            values=['快速', '平衡', '高质量'],
            state='readonly',
            width=8
        )
        self.shade_quality_combo.pack(side=tk.RIGHT, padx=(2, 8))
        self._apply_font_to_widget(self.shade_quality_combo)
        try:
            f = self._get_ui_font(16)
            fn = f[0] if isinstance(f, tuple) else f
            fs = f[1] if isinstance(f, tuple) and len(f) > 1 else self._get_main_ui_size(16)
            self._apply_combobox_popup_font(self.shade_quality_combo, fn, fs)
        except Exception:
            pass
        self.shade_quality_combo.bind('<<ComboboxSelected>>', self.on_shade_quality_changed)
        
        # 内容区域（可折叠）- 直接显示32个参数的网格布局
        self.content = ttk.Frame(self.container)
        self.content.pack(fill=tk.X, padx=2, pady=2)
        
        # 创建32个参数的网格布局（8列x4行）
        self.create_parameter_grid()
        
    def toggle(self):
        """切换展开/折叠状态"""
        if self.expanded:
            self.content.pack_forget()
            self.toggle_btn.config(text='▶')
            self.expanded = False
        else:
            self.content.pack(fill=tk.X, padx=2, pady=2)
            self.toggle_btn.config(text='▼')
            self.expanded = True
            
    def create_parameter_grid(self):
        """创建32个参数的网格布局（按功能分组）"""
        # 使用滚动框架支持大量参数
        canvas = tk.Canvas(self.content, height=220)
        scrollbar = ttk.Scrollbar(self.content, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 记录当前网格容器，支持后续拖拽换位时重排
        self._panel_grid_parent = scrollable_frame
        self._panel_frames = {}
        self._panel_order = self._normalize_panel_order(self._panel_order)

        # 按功能分组创建参数控件（8列布局）
        # 第1列：显示控制 (1-4)
        col0_frame = self._create_panel_labelframe(scrollable_frame, '显示控制')
        self._panel_frames['显示控制'] = col0_frame
        self._create_param_row(col0_frame, 0, 1, 'iscale', '缩放模式', 
                               'int', ['自动(0)', '固定(1)', '变增益(2)'], None, None)
        self._create_param_row(col0_frame, 1, 2, 'amp', '振幅', 'float', None, None, None)
        self._create_param_row(col0_frame, 2, 3, 'rcor', '距离校正', 'float', None, None, None)
        self._create_param_row(col0_frame, 3, 4, 'sf', '缩放因子', 'float', None, None, None)
        
        # 第2列：数据处理 (5-8)
        col1_frame = self._create_panel_labelframe(scrollable_frame, '数据处理')
        self._panel_frames['数据处理'] = col1_frame
        self._create_param_row(col1_frame, 0, 5, 'irec', '记录号', 'int', None, 1, 1000)
        self._create_param_row(col1_frame, 1, 6, 'itx', '理论走时', 
                              'int', ['不显示(0)', '显示(1)'], None, None)
        # imute: 静校正开关 (0=关闭, !=0=启用短波长静校正)
        # 注意：需要先通过菜单 Tools -> Calculate Static Correction 计算静校正量
        self._create_param_row(col1_frame, 2, 7, 'imute', '静校正开关', 'int', 
                              ['关闭(0)', '启用(1)'], None, None)
        self._create_param_row(col1_frame, 3, 8, 'iwater', '水层校正', 
                              'int', ['无(0)', '启用(1)'], None, None)
        
        # 第3列：滤波设置 (9-12)
        col2_frame = self._create_panel_labelframe(scrollable_frame, '滤波设置')
        self._panel_frames['滤波设置'] = col2_frame
        self._create_param_row(col2_frame, 0, 9, 'ibndps', '带通滤波', 
                              'int', ['无(0)', '启用(1)', 'FFT显示(-2)'], None, None)
        self._create_param_row(col2_frame, 1, 10, 'izerop', '零相位滤波', 
                              'int', ['否(0)', '是(1)'], None, None)
        self._create_param_row(col2_frame, 2, 11, 'freqlo', '低截止频率(Hz)', 'float', None, None, None)
        self._create_param_row(col2_frame, 3, 12, 'freqhi', '高截止频率(Hz)', 'float', None, None, None)
        
        # 第4列：显示参数 (13-16)
        col3_frame = self._create_panel_labelframe(scrollable_frame, '显示参数')
        self._panel_frames['显示参数'] = col3_frame
        self._create_param_row(col3_frame, 0, 13, 'tlinc', '时间增量(s)', 'float', None, None, None)
        self._create_param_row(col3_frame, 1, 14, 'clip', '裁剪值', 'float', None, None, None)
        self._create_param_row(col3_frame, 2, 15, 'vred', '折合速度(km/s)', 'float', None, None, None)
        self._create_param_row(col3_frame, 3, 16, 'ishade', '阴影填充', 
                              'int', ['无(0)', '正峰值(>0)', '负峰值(<0)'], None, None)
        
        # 第5列：拾取控制 (17-20)
        col4_frame = self._create_panel_labelframe(scrollable_frame, '拾取控制')
        self._panel_frames['拾取控制'] = col4_frame
        self._create_param_row(col4_frame, 0, 17, 'spick', '拾取符号大小', 'float', None, None, None)
        self._create_param_row(col4_frame, 1, 18, 'ixaxis', 'X轴类型', 
                              'int', ['炮检距(-1)', '模型位置(-2)', '方位角(-3)', '修正方位角(-4)', '道号(-5)'], None, None)
        self._create_param_row(col4_frame, 2, 19, 'itype', '数据类型', 
                              'int', [
                                  '全部(0)', 
                                  '垂直(1)', 
                                  '径向(2)', 
                                  '横向(3)', 
                                  '水听器(4)',
                                  '垂直+径向(-1)',
                                  '径向+横向(-2)',
                                  '径向+水听器(-3)',
                                  '垂直+水听器(-4)'
                              ], None, None)
        self._create_param_row(col4_frame, 3, 20, 'apick', '活动拾取字', 'int', None, 1, 40)
        
        # 第6列：其他参数 (21-24)
        col5_frame = self._create_panel_labelframe(scrollable_frame, '其他参数')
        self._panel_frames['其他参数'] = col5_frame
        self._create_param_row(col5_frame, 0, 21, 'nskip', '跳过的道数', 'int', None, 0, 1000)
        self._create_param_row(col5_frame, 1, 22, 'xmm', 'X轴范围(mm)', 'float', None, None, None)
        self._create_param_row(col5_frame, 2, 23, 'xmin', 'X轴最小值', 'float', None, None, None)
        self._create_param_row(col5_frame, 3, 24, 'xmax', 'X轴最大值', 'float', None, None, None)
        
        # 第7列：时间参数 (25-28)
        col6_frame = self._create_panel_labelframe(scrollable_frame, '时间参数')
        self._panel_frames['时间参数'] = col6_frame
        self._create_param_row(col6_frame, 0, 25, 'ndecim', '数据抽取间隔', 'int', None, 1, 100)
        self._create_param_row(col6_frame, 1, 26, 'tmm', '时间轴范围(mm)', 'float', None, None, None)
        self._create_param_row(col6_frame, 2, 27, 'tmin', '时间最小值(s)', 'float', None, None, None)
        self._create_param_row(col6_frame, 3, 28, 'tmax', '时间最大值(s)', 'float', None, None, None)
        
        # 第8列：特殊功能 (29-32) - 这些是按钮
        col7_frame = self._create_panel_labelframe(scrollable_frame, '特殊功能')
        self._panel_frames['特殊功能'] = col7_frame
        # 特殊功能面板单独加宽，避免按钮文本拥挤
        col7_frame.columnconfigure(0, weight=1, minsize=72)
        ttk.Button(col7_frame, text='退出 (Q)', width=14,
                  command=self.main_gui.quit).grid(row=0, column=0, padx=2, pady=2, sticky='ew')
        ttk.Button(col7_frame, text='原始/前一个图', width=14,
                  command=self.prev_plot).grid(row=1, column=0, padx=2, pady=2, sticky='ew')
        ttk.Button(col7_frame, text='重绘', width=14,
                  command=self.replot).grid(row=2, column=0, padx=2, pady=2, sticky='ew')
        ttk.Button(col7_frame, text='自动拾取', width=14,
                  command=self.auto_pick).grid(row=3, column=0, padx=2, pady=2, sticky='ew')
        
        self._apply_panel_order_layout()

        # 绑定拖拽换位事件（按下并释放到目标面板即可交换）
        for frame in self._panel_frames.values():
            self._bind_panel_drag_events(frame)

        # 全局监听释放事件，避免“按下在面板内、释放在其他组件上”时丢失交换
        root = self._get_root_widget()
        if root is not None:
            try:
                root.bind_all('<ButtonRelease-1>', self._on_panel_drag_release, add='+')
            except Exception:
                pass

    def _normalize_panel_order(self, panel_order):
        """规范化面板顺序，保证与默认面板集合一致。"""
        if not isinstance(panel_order, (list, tuple)):
            return list(self.DEFAULT_PANEL_ORDER)
        valid = [str(item) for item in panel_order if str(item) in self.DEFAULT_PANEL_ORDER]
        for name in self.DEFAULT_PANEL_ORDER:
            if name not in valid:
                valid.append(name)
        return valid

    def _apply_panel_order_layout(self):
        """按当前顺序重新布局顶部参数面板。"""
        if not self._panel_grid_parent or not self._panel_frames:
            return
        self._panel_order = self._normalize_panel_order(self._panel_order)
        for col, panel_name in enumerate(self._panel_order):
            frame = self._panel_frames.get(panel_name)
            if frame is None:
                continue
            frame.grid(row=0, column=col, sticky='nsew', padx=1, pady=2)
            minsize = 72 if panel_name == '特殊功能' else 50
            self._panel_grid_parent.columnconfigure(col, weight=1, minsize=minsize)

    def _get_panel_name_by_widget(self, widget) -> Optional[str]:
        """从事件组件向上回溯，定位所属面板名。"""
        current = widget
        while current is not None:
            for panel_name, panel_frame in self._panel_frames.items():
                if current == panel_frame:
                    return panel_name
            try:
                current = current.master
            except Exception:
                break
        return None

    def _get_panel_name_by_screen_pos(self, x_root: int, y_root: int) -> Optional[str]:
        """根据屏幕坐标定位面板（优先包围盒命中，次选最近列中心）。"""
        # 1) 优先使用面板实际包围盒命中
        for panel_name in self._panel_order:
            frame = self._panel_frames.get(panel_name)
            if frame is None:
                continue
            try:
                fx = frame.winfo_rootx()
                fy = frame.winfo_rooty()
                fw = frame.winfo_width()
                fh = frame.winfo_height()
            except Exception:
                continue
            if fw <= 0 or fh <= 0:
                continue
            if fx <= x_root <= fx + fw and fy <= y_root <= fy + fh:
                return panel_name

        # 2) 没有命中时，按 X 坐标吸附到最近面板中心（要求 Y 在工具栏行附近）
        centers = []
        y_min = None
        y_max = None
        for panel_name in self._panel_order:
            frame = self._panel_frames.get(panel_name)
            if frame is None:
                continue
            try:
                fx = frame.winfo_rootx()
                fy = frame.winfo_rooty()
                fw = frame.winfo_width()
                fh = frame.winfo_height()
            except Exception:
                continue
            if fw <= 0 or fh <= 0:
                continue
            centers.append((panel_name, fx + fw / 2.0))
            y_min = fy if y_min is None else min(y_min, fy)
            y_max = fy + fh if y_max is None else max(y_max, fy + fh)

        if not centers:
            return None
        if y_min is not None and y_max is not None and not (y_min - 12 <= y_root <= y_max + 12):
            return None

        centers.sort(key=lambda item: abs(item[1] - x_root))
        return centers[0][0]

    def _bind_panel_drag_events(self, frame):
        """为面板及其标签绑定拖拽换位事件。"""
        try:
            frame.configure(cursor='fleur')
        except Exception:
            pass

        def _bind(widget):
            try:
                widget.bind('<ButtonPress-1>', self._on_panel_drag_start, add='+')
            except Exception:
                return

        def _bind_recursive(widget):
            _bind(widget)
            try:
                for child in widget.winfo_children():
                    _bind_recursive(child)
            except Exception:
                pass

        _bind_recursive(frame)

    def _on_panel_drag_start(self, event):
        """记录拖拽起始面板。"""
        panel_name = self._get_panel_name_by_widget(event.widget)
        if not panel_name:
            panel_name = self._get_panel_name_by_screen_pos(event.x_root, event.y_root)
        if not panel_name:
            return
        self._panel_drag_source = panel_name
        self._panel_drag_started = True
        if hasattr(self.main_gui, 'update_status'):
            self.main_gui.update_status(f'正在拖动面板：{panel_name}（释放到目标面板可交换位置）')

    def _on_panel_drag_release(self, event):
        """在鼠标释放时执行面板换位。"""
        if not self._panel_drag_started:
            return
        self._panel_drag_started = False
        source = self._panel_drag_source
        self._panel_drag_source = None
        if not source or source not in self._panel_order:
            return
        root = self._get_root_widget()
        if root is None:
            return
        target = self._get_panel_name_by_screen_pos(event.x_root, event.y_root)
        if not target:
            target_widget = root.winfo_containing(event.x_root, event.y_root)
            if target_widget is not None:
                target = self._get_panel_name_by_widget(target_widget)
        if not target or target == source or target not in self._panel_order:
            return
        src_idx = self._panel_order.index(source)
        dst_idx = self._panel_order.index(target)
        self._panel_order[src_idx], self._panel_order[dst_idx] = (
            self._panel_order[dst_idx],
            self._panel_order[src_idx],
        )
        self._apply_panel_order_layout()
        if hasattr(self.main_gui, '_save_workbench_state'):
            try:
                self.main_gui._save_workbench_state()
            except Exception:
                pass
        if hasattr(self.main_gui, 'update_status'):
            self.main_gui.update_status(f'已交换面板位置：{source} <-> {target}')

    def get_panel_order(self):
        """返回当前面板顺序。"""
        return list(self._normalize_panel_order(self._panel_order))

    def set_panel_order(self, panel_order):
        """设置面板顺序并立即重排。"""
        self._panel_order = self._normalize_panel_order(panel_order)
        self._apply_panel_order_layout()
        
    def create_display_tab(self, parent):
        """创建显示控制标签页"""
        row = 0
        
        # 缩放模式
        ttk.Label(parent, text='缩放模式:').grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.scale_mode_var = tk.StringVar(value='自动')
        scale_combo = ttk.Combobox(parent, textvariable=self.scale_mode_var, 
                                  values=['自动', '固定', '变增益'], 
                                  state='readonly', width=15)
        scale_combo.grid(row=row, column=1, padx=5, pady=2)
        row += 1
        
        # 振幅
        ttk.Label(parent, text='振幅:').grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.amp_entry = ttk.Entry(parent, width=15)
        self.amp_entry.insert(0, '1.0')
        self.amp_entry.grid(row=row, column=1, padx=5, pady=2)
        row += 1
        
        # 折合速度
        ttk.Label(parent, text='折合速度 (km/s):').grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.vred_entry = ttk.Entry(parent, width=15)
        self.vred_entry.insert(0, '6.0')
        self.vred_entry.grid(row=row, column=1, padx=5, pady=2)
        row += 1
        
        # X轴类型
        ttk.Label(parent, text='X轴类型:').grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.xaxis_var = tk.StringVar(value='炮检距')
        xaxis_combo = ttk.Combobox(parent, textvariable=self.xaxis_var,
                                  values=['炮检距', '模型位置', '方位角', '道号'],
                                  state='readonly', width=15)
        xaxis_combo.grid(row=row, column=1, padx=5, pady=2)
        row += 1
        
        # X轴范围
        ttk.Label(parent, text='X轴范围:').grid(row=row, column=0, sticky='w', padx=5, pady=2)
        range_frame = ttk.Frame(parent)
        range_frame.grid(row=row, column=1, padx=5, pady=2)
        self.xmin_entry = ttk.Entry(range_frame, width=8)
        self.xmin_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(range_frame, text='到').pack(side=tk.LEFT, padx=2)
        self.xmax_entry = ttk.Entry(range_frame, width=8)
        self.xmax_entry.pack(side=tk.LEFT, padx=2)
        row += 1
        
        # 时间范围
        ttk.Label(parent, text='时间范围 (s):').grid(row=row, column=0, sticky='w', padx=5, pady=2)
        time_frame = ttk.Frame(parent)
        time_frame.grid(row=row, column=1, padx=5, pady=2)
        self.tmin_entry = ttk.Entry(time_frame, width=8)
        self.tmin_entry.insert(0, '0.0')
        self.tmin_entry.pack(side=tk.LEFT, padx=2)
        ttk.Label(time_frame, text='到').pack(side=tk.LEFT, padx=2)
        self.tmax_entry = ttk.Entry(time_frame, width=8)
        self.tmax_entry.insert(0, '20.0')
        self.tmax_entry.pack(side=tk.LEFT, padx=2)
        row += 1
        
        # 应用按钮
        ttk.Button(parent, text='应用', width=15,
                  command=self.apply_display_params).grid(row=row, column=1, padx=5, pady=10)
        
    def create_filter_tab(self, parent):
        """创建滤波设置标签页"""
        row = 0
        
        # 带通滤波开关
        self.filter_on_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text='启用带通滤波', 
                       variable=self.filter_on_var,
                       command=self.toggle_filter).grid(row=row, column=0, columnspan=2, 
                                                        sticky='w', padx=5, pady=2)
        row += 1
        
        # 低截止频率
        ttk.Label(parent, text='低截止频率 (Hz):').grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.freqlo_entry = ttk.Entry(parent, width=15)
        self.freqlo_entry.insert(0, '3.0')
        self.freqlo_entry.grid(row=row, column=1, padx=5, pady=2)
        row += 1
        
        # 高截止频率
        ttk.Label(parent, text='高截止频率 (Hz):').grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.freqhi_entry = ttk.Entry(parent, width=15)
        self.freqhi_entry.insert(0, '15.0')
        self.freqhi_entry.grid(row=row, column=1, padx=5, pady=2)
        row += 1
        
        # 零相位滤波
        self.zerophase_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text='零相位滤波', 
                       variable=self.zerophase_var).grid(row=row, column=0, columnspan=2, 
                                                         sticky='w', padx=5, pady=2)
        row += 1
        
        # 滤波器阶数
        ttk.Label(parent, text='滤波器阶数:').grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.npoles_entry = ttk.Entry(parent, width=15)
        self.npoles_entry.insert(0, '8')
        self.npoles_entry.grid(row=row, column=1, padx=5, pady=2)
        row += 1
        
        # 应用按钮
        ttk.Button(parent, text='应用滤波', width=15,
                  command=self.apply_filter_params).grid(row=row, column=1, padx=5, pady=10)
        
    def create_pick_tab(self, parent):
        """创建拾取控制标签页"""
        row = 0
        
        # 活动拾取字
        ttk.Label(parent, text='活动拾取字:').grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.pick_word_var = tk.IntVar(value=self.params.apick)
        pick_word_spin = ttk.Spinbox(parent, from_=1, to=40, textvariable=self.pick_word_var, 
                                    width=15, command=self.update_pick_word)
        pick_word_spin.grid(row=row, column=1, padx=5, pady=2)
        self.param_widgets['apick'] = (pick_word_spin, self.pick_word_var, 20, 'int', 'spin')
        row += 1
        
        # 拾取符号大小
        ttk.Label(parent, text='拾取符号大小:').grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.pick_size_entry = ttk.Entry(parent, width=15)
        self.pick_size_entry.insert(0, '8')
        self.pick_size_entry.grid(row=row, column=1, padx=5, pady=2)
        row += 1
        
        # 拾取颜色
        ttk.Label(parent, text='拾取颜色:').grid(row=row, column=0, sticky='w', padx=5, pady=2)
        color_frame = ttk.Frame(parent)
        color_frame.grid(row=row, column=1, padx=5, pady=2)
        self.pick_color_var = tk.StringVar(value='red')
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'black']
        for color in colors:
            ttk.Radiobutton(color_frame, variable=self.pick_color_var, 
                          value=color, width=3).pack(side=tk.LEFT, padx=1)
        row += 1
        
        # 记录号
        ttk.Label(parent, text='当前记录号:').grid(row=row, column=0, sticky='w', padx=5, pady=2)
        self.record_var = tk.IntVar(value=1)
        record_spin = ttk.Spinbox(parent, from_=1, to=100, textvariable=self.record_var, 
                                 width=15)
        record_spin.grid(row=row, column=1, padx=5, pady=2)
        row += 1
        
        # 拾取模式按钮
        ttk.Button(parent, text='进入拾取模式 (P)', width=20,
                  command=lambda: self.main_gui.enter_pick_mode()).grid(row=row, column=0, 
                                                                       columnspan=2, padx=5, pady=10)
    
    def update_pick_word(self):
        """更新活动拾取字"""
        self.params.apick = self.pick_word_var.get()
    
    def create_advanced_tab(self, parent):
        """创建高级参数标签页 - 包含所有32个菜单参数"""
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
        
        # 创建参数控件
        row = 0
        
        # 第1列：显示控制 (1-4)
        col0_frame = self._create_panel_labelframe(scrollable_frame, '显示控制')
        col0_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        self._create_param_row(col0_frame, 0, 1, 'iscale', '缩放模式', 
                               values=['自动(0)', '固定(1)', '变增益(2)'], param_type='int')
        self._create_param_row(col0_frame, 1, 2, 'amp', '振幅', param_type='float')
        self._create_param_row(col0_frame, 2, 3, 'rcor', '距离校正', param_type='float')
        self._create_param_row(col0_frame, 3, 4, 'sf', '缩放因子', param_type='float')
        
        # 第2列：数据处理 (5-8)
        col1_frame = self._create_panel_labelframe(scrollable_frame, '数据处理')
        col1_frame.grid(row=0, column=1, sticky='nsew', padx=5, pady=5)
        self._create_param_row(col1_frame, 0, 5, 'irec', '记录号', param_type='int')
        self._create_param_row(col1_frame, 1, 6, 'itx', '理论走时', 
                              values=['不显示(0)', '显示(1)'], param_type='int')
        # imute: 静校正开关 (0=关闭, !=0=启用短波长静校正)
        self._create_param_row(col1_frame, 2, 7, 'imute', '静校正开关', 
                              values=['关闭(0)', '启用(1)'], param_type='int')
        self._create_param_row(col1_frame, 3, 8, 'iwater', '水层校正', 
                              values=['无(0)', '启用(1)'], param_type='int')
        
        # 第3列：滤波设置 (9-12)
        col2_frame = self._create_panel_labelframe(scrollable_frame, '滤波设置')
        col2_frame.grid(row=0, column=2, sticky='nsew', padx=5, pady=5)
        self._create_param_row(col2_frame, 0, 9, 'ibndps', '带通滤波', 
                              values=['无(0)', '启用(1)', 'FFT显示(-2)'], param_type='int')
        self._create_param_row(col2_frame, 1, 10, 'izerop', '零相位滤波', 
                              values=['否(0)', '是(1)'], param_type='int')
        self._create_param_row(col2_frame, 2, 11, 'freqlo', '低截止频率(Hz)', param_type='float')
        self._create_param_row(col2_frame, 3, 12, 'freqhi', '高截止频率(Hz)', param_type='float')
        
        # 第4列：显示参数 (13-16)
        col3_frame = self._create_panel_labelframe(scrollable_frame, '显示参数')
        col3_frame.grid(row=0, column=3, sticky='nsew', padx=5, pady=5)
        self._create_param_row(col3_frame, 0, 13, 'tlinc', '时间增量(s)', param_type='float')
        self._create_param_row(col3_frame, 1, 14, 'clip', '裁剪值', param_type='float')
        self._create_param_row(col3_frame, 2, 15, 'vred', '折合速度(km/s)', param_type='float')
        self._create_param_row(col3_frame, 3, 16, 'ishade', '阴影填充', 
                              values=['无(0)', '正峰值(>0)', '负峰值(<0)'], param_type='int')
        
        # 第5列：拾取控制 (17-20)
        col4_frame = self._create_panel_labelframe(scrollable_frame, '拾取控制')
        col4_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        self._create_param_row(col4_frame, 0, 17, 'spick', '拾取符号大小', param_type='float')
        self._create_param_row(col4_frame, 1, 18, 'ixaxis', 'X轴类型', 
                              values=['炮检距(-1)', '模型位置(-2)', '方位角(-3)', '修正方位角(-4)', '道号(-5)'], 
                              param_type='int')
        self._create_param_row(col4_frame, 2, 19, 'itype', '数据类型', 
                              values=['全部(0)', '垂直(1)', '径向(2)', '横向(3)', '水听器(4)'], param_type='int')
        self._create_param_row(col4_frame, 3, 20, 'apick', '活动拾取字', param_type='int', min_val=1, max_val=40)
        
        # 第6列：其他参数 (21-24)
        col5_frame = self._create_panel_labelframe(scrollable_frame, '其他参数')
        col5_frame.grid(row=1, column=1, sticky='nsew', padx=5, pady=5)
        self._create_param_row(col5_frame, 0, 21, 'nskip', '跳过的道数', param_type='int')
        self._create_param_row(col5_frame, 1, 22, 'xmm', 'X轴范围(mm)', param_type='float')
        self._create_param_row(col5_frame, 2, 23, 'xmin', 'X轴最小值', param_type='float')
        self._create_param_row(col5_frame, 3, 24, 'xmax', 'X轴最大值', param_type='float')
        
        # 第7列：时间参数 (25-28)
        col6_frame = self._create_panel_labelframe(scrollable_frame, '时间参数')
        col6_frame.grid(row=1, column=2, sticky='nsew', padx=5, pady=5)
        self._create_param_row(col6_frame, 0, 25, 'ndecim', '数据抽取间隔', param_type='int', min_val=1)
        self._create_param_row(col6_frame, 1, 26, 'tmm', '时间轴范围(mm)', param_type='float')
        self._create_param_row(col6_frame, 2, 27, 'tmin', '时间最小值(s)', param_type='float')
        self._create_param_row(col6_frame, 3, 28, 'tmax', '时间最大值(s)', param_type='float')
        
        # 第8列：特殊功能 (29-32) - 这些是按钮
        col7_frame = self._create_panel_labelframe(scrollable_frame, '特殊功能')
        col7_frame.grid(row=1, column=3, sticky='nsew', padx=5, pady=5)
        ttk.Button(col7_frame, text='退出 (Q)', width=20,
                  command=self.main_gui.quit).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(col7_frame, text='原始/前一个图', width=20,
                  command=self.prev_plot).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(col7_frame, text='重绘', width=20,
                  command=self.replot).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(col7_frame, text='自动拾取', width=20,
                  command=self.auto_pick).grid(row=3, column=0, padx=5, pady=5)
        
        # 应用所有参数按钮
        ttk.Button(scrollable_frame, text='应用所有参数', width=30,
                  command=self.apply_all_params).grid(row=2, column=0, columnspan=4, padx=5, pady=10)
    
    def _apply_font_to_widget(self, widget):
        """为组件应用中文字体"""
        try:
            chinese_font = self._get_ui_font(16)
            # 尝试直接设置字体（对于某些组件可能有效）
            try:
                widget.configure(font=chinese_font)
            except:
                pass
        except:
            pass
    
    def _configure_labelframe_font(self, labelframe):
        """配置 LabelFrame 的标题字体"""
        style = ttk.Style()
        chinese_font = self._get_ui_font(16, title_boost=1)
        
        # 方法1: 确保样式已设置
        style.configure('TLabelframe.Label', font=chinese_font)
        
        # 方法2: 创建并使用自定义样式
        try:
            style_name = 'Chinese.TLabelframe'
            style.configure(style_name, font=chinese_font)
            style.configure(f'{style_name}.Label', font=chinese_font)
            style.configure('Chinese.TLabelframe.Label', font=chinese_font)
            labelframe.configure(style=style_name)
        except Exception:
            pass
        
        # 方法3: 尝试直接访问并设置标题标签的字体
        # 注意：LabelFrame 的标题标签可能不是直接子组件
        try:
            # 使用 update_idletasks 确保组件已完全创建
            labelframe.update_idletasks()
            
            # 尝试通过内部组件设置字体
            for widget_id in labelframe.winfo_children():
                try:
                    widget = labelframe.nametowidget(str(widget_id))
                    widget_class = widget.winfo_class()
                    if 'Label' in widget_class:
                        widget.configure(font=chinese_font)
                except:
                    pass
        except Exception:
            pass
        
        # 方法4: 使用 after 延迟设置（确保组件完全创建后）
        try:
            def set_font_later():
                try:
                    style.configure('TLabelframe.Label', font=chinese_font)
                    labelframe.update_idletasks()
                except:
                    pass
            
            # 获取根窗口
            root = labelframe
            while root and not isinstance(root, tk.Tk) and not isinstance(root, tk.Toplevel):
                try:
                    root = root.master
                except:
                    break
            
            if root:
                root.after(100, set_font_later)  # 100ms 后设置字体
        except:
            pass

    def _apply_font_recursively(self, widget):
        """递归刷新子控件字体，确保显式字体控件也同步。"""
        if widget is None:
            return

        normal_font = self._get_ui_font(16)
        try:
            wclass = widget.winfo_class()
        except Exception:
            wclass = ''

        if wclass in (
            'Label', 'Button', 'Entry', 'Text', 'Listbox', 'Menu', 'Menubutton',
            'Checkbutton', 'Radiobutton', 'Message', 'Spinbox',
            'TLabel', 'TButton', 'TEntry', 'TCheckbutton', 'TRadiobutton', 'TSpinbox', 'TMenubutton'
        ):
            try:
                widget.configure(font=normal_font)
            except Exception:
                pass

        if wclass == 'TCombobox':
            try:
                widget.configure(font=normal_font)
            except Exception:
                pass
            try:
                font_name = normal_font[0] if isinstance(normal_font, tuple) else normal_font
                font_size = normal_font[1] if isinstance(normal_font, tuple) and len(normal_font) > 1 else self._get_main_ui_size(16)
                self._apply_combobox_popup_font(widget, font_name, font_size)
            except Exception:
                pass

        try:
            for child in widget.winfo_children():
                self._apply_font_recursively(child)
        except Exception:
            pass

    def _bind_param_hover_help(self, widget, param_name: str):
        """为参数控件绑定悬停说明。"""
        help_text = self._param_help_texts.get(param_name)
        if not help_text or widget is None:
            return

        def _compose_help_text() -> str:
            if not self._shift_pressed:
                return help_text
            detailed = self._param_help_texts_detailed.get(param_name, help_text)
            return f"详细：{detailed}"

        def _show_hover_help():
            if hasattr(self.main_gui, 'update_status'):
                self.main_gui.update_status(f"{param_name}: {_compose_help_text()}")

        def _on_enter(_event=None):
            self._hover_help_param = param_name
            _show_hover_help()

        def _on_leave(_event=None):
            self._hover_help_param = None
            if hasattr(self.main_gui, 'update_status'):
                self.main_gui.update_status()

        def _on_motion(event=None):
            # 兼容某些平台上 Shift 状态只体现在鼠标事件 state 位
            if event is not None and hasattr(event, 'state'):
                self._shift_pressed = bool(event.state & 0x0001)
            if self._hover_help_param == param_name:
                _show_hover_help()

        try:
            widget.bind('<Enter>', _on_enter, add='+')
            widget.bind('<Leave>', _on_leave, add='+')
            widget.bind('<Motion>', _on_motion, add='+')
        except Exception:
            pass

    def _refresh_hover_help_status(self):
        """当 Shift 状态变化时刷新当前悬停说明。"""
        param_name = self._hover_help_param
        if not param_name:
            return
        help_text = self._param_help_texts.get(param_name)
        if not help_text:
            return
        if self._shift_pressed:
            detailed = self._param_help_texts_detailed.get(param_name, help_text)
            content = f"详细：{detailed}"
        else:
            content = help_text
        if hasattr(self.main_gui, 'update_status'):
            self.main_gui.update_status(f"{param_name}: {content}")

    def refresh_ui_fonts(self):
        """刷新工具栏字体，供主窗口动态字号调整时调用。"""
        self.setup_chinese_fonts()
        try:
            if self.header_title_label is not None:
                self.header_title_label.configure(font=self._get_ui_font(16, 'bold', title_boost=1))
            if self.shade_quality_label is not None:
                self.shade_quality_label.configure(font=self._get_ui_font(16))
        except Exception:
            pass

        for lf in list(self._panel_labelframes):
            try:
                if lf.winfo_exists():
                    self._configure_labelframe_font(lf)
            except Exception:
                pass

        self._apply_font_recursively(self.container)
    
    def _create_param_row(self, parent, row, menu_id, param_name, label, 
                          param_type='float', values=None, min_val=None, max_val=None):
        """创建参数行控件
        
        Args:
            parent: 父容器
            row: 行号
            menu_id: 菜单ID (1-32)
            param_name: 参数名称
            label: 显示标签
            param_type: 参数类型 ('int', 'float')
            values: 可选值列表（用于下拉框）
            min_val: 最小值
            max_val: 最大值
        """
        # 标签（使用中文字体）
        label_widget = ttk.Label(parent, text=f'{label}:', font=self._get_ui_font(16))
        label_widget.grid(row=row, column=0, sticky='w', padx=1, pady=2)
        
        current_value = self.params.get_menu_value(menu_id)
        
        if values:
            # 使用下拉框
            var = tk.StringVar()
            # 找到当前值对应的选项
            # 特殊处理 ishade 参数
            if param_name == 'ishade':
                if current_value == 0:
                    var.set('无(0)')
                elif current_value > 0:
                    var.set('正峰值(>0)')
                elif current_value < 0:
                    var.set('负峰值(<0)')
                else:
                    var.set(values[0])
            else:
                current_str = str(current_value)
                for val in values:
                    if f'({current_str})' in val or val.startswith(current_str):
                        var.set(val)
                        break
                else:
                    var.set(values[0])
            
            combo = ttk.Combobox(parent, textvariable=var, values=values, 
                               state='readonly', width=12)
            combo.grid(row=row, column=1, padx=1, pady=2, sticky='ew')
            # 确保 Combobox 使用中文字体
            self._apply_font_to_widget(combo)
            try:
                f = self._get_ui_font(16)
                fn = f[0] if isinstance(f, tuple) else f
                self._apply_combobox_popup_font(combo, fn, 12)
            except Exception:
                pass
            self.param_widgets[param_name] = (combo, var, menu_id, param_type, 'combo')
            self._bind_param_hover_help(combo, param_name)
        elif param_type == 'int' and (min_val is not None or max_val is not None):
            # 使用 Spinbox
            var = tk.IntVar(value=int(current_value))
            spin = ttk.Spinbox(parent, from_=min_val or 0, to=max_val or 1000, 
                             textvariable=var, width=12)
            spin.grid(row=row, column=1, padx=1, pady=2, sticky='ew')
            self._apply_font_to_widget(spin)
            self.param_widgets[param_name] = (spin, var, menu_id, param_type, 'spin')
            self._bind_param_hover_help(spin, param_name)
        else:
            # 使用 Entry
            var = tk.StringVar(value=str(current_value))
            entry = ttk.Entry(parent, textvariable=var, width=12)
            entry.grid(row=row, column=1, padx=1, pady=2, sticky='ew')
            self._apply_font_to_widget(entry)
            self.param_widgets[param_name] = (entry, var, menu_id, param_type, 'entry')
            self._bind_param_hover_help(entry, param_name)
        
        self._bind_param_hover_help(label_widget, param_name)
        
        # 配置列权重和最小宽度
        parent.columnconfigure(0, minsize=80)  # 标签列最小宽度
        parent.columnconfigure(1, weight=1, minsize=120)  # 控件列最小宽度
    
    def prev_plot(self):
        """显示前一个记录"""
        if hasattr(self.main_gui, 'prev_record'):
            self.main_gui.prev_record()
        else:
            self.main_gui.update_status('记录导航功能未初始化')
    
    def replot(self):
        """重绘图形（使用当前编辑的参数重新绘制）"""
        if not hasattr(self.main_gui, 'update_plot'):
            self.main_gui.update_status('重绘功能未初始化')
            return
        
        if not self.main_gui.data_loaded:
            self.main_gui.update_status('请先加载数据文件')
            return
        
        # 清除用户缩放状态（重绘时使用参数中的范围）
        self.main_gui.preserve_zoom = False
        self.main_gui.user_xlim = None
        self.main_gui.user_ylim = None
        
        # 清除对齐状态
        self.main_gui.alignment_active = False
        self.main_gui.alignment_offsets = {}
        self.main_gui.aligned_trace_indices = []
        
        # 恢复初始的范围参数（xmin, xmax, tmin, tmax）
        if hasattr(self.main_gui, 'initial_range_params') and self.main_gui.initial_range_params:
            self.params.xmin = self.main_gui.initial_range_params['xmin']
            self.params.xmax = self.main_gui.initial_range_params['xmax']
            self.params.tmin = self.main_gui.initial_range_params['tmin']
            self.params.tmax = self.main_gui.initial_range_params['tmax']
        
        # 从工具栏控件读取当前值并更新参数（使用编辑后的参数，但不包括范围参数）
        # 范围参数已经在上面恢复了，这里跳过它们
        range_params = {'xmin', 'xmax', 'tmin', 'tmax'}
        for param_name, (widget, var, menu_id, param_type, widget_type) in self.param_widgets.items():
            # 跳过范围参数，它们已经恢复了
            if param_name in range_params:
                continue
                
            try:
                if widget_type == 'combo':
                    # 从下拉框获取值（格式：'标签(值)'）
                    value_str = var.get()
                    # 提取括号中的值
                    if '(' in value_str and ')' in value_str:
                        value_str = value_str.split('(')[1].split(')')[0]
                    
                    # 特殊处理 ishade 参数：'>0' -> 1, '<0' -> -1
                    if param_name == 'ishade':
                        if value_str == '>0' or value_str == '1':
                            value = 1
                        elif value_str == '<0' or value_str == '-1':
                            value = -1
                        else:
                            value = int(value_str) if param_type == 'int' else float(value_str)
                    else:
                        value = int(value_str) if param_type == 'int' else float(value_str)
                elif widget_type == 'spin':
                    value = var.get()
                else:  # entry
                    value_str = var.get()
                    value = int(value_str) if param_type == 'int' else float(value_str)
                
                self.params.set_menu_value(menu_id, value)
            except (ValueError, AttributeError):
                # 参数值无效，静默跳过
                pass
        
        # 确保主窗口的参数对象已更新
        if hasattr(self.main_gui, 'params') and hasattr(self, 'params'):
            self.main_gui.params = self.params
        
        # 更新工具栏控件显示（包括范围参数），确保显示值与参数对象同步
        self.update_widgets_from_params()
        
        # 调用主窗口的重绘方法（会自动绘制剖面和拾取点）
        if hasattr(self.main_gui, 'request_plot_refresh'):
            self.main_gui.request_plot_refresh(delay_ms=20)
        else:
            self.main_gui.update_plot()
        self.main_gui.update_status('图形已重绘（使用当前参数）')
    
    def auto_pick(self):
        """自动拾取"""
        self.main_gui.auto_pick()

    def _shade_quality_label_to_value(self, label: str) -> str:
        mapping = {
            '快速': 'fast',
            '平衡': 'balanced',
            '高质量': 'high',
        }
        return mapping.get(label, 'balanced')

    def _shade_quality_value_to_label(self, value: str) -> str:
        mapping = {
            'fast': '快速',
            'balanced': '平衡',
            'high': '高质量',
        }
        return mapping.get(str(value).strip().lower(), '平衡')

    def on_shade_quality_changed(self, _event=None):
        """阴影质量档位切换回调。"""
        if not hasattr(self, 'params') or self.params is None:
            return
        quality_value = self._shade_quality_label_to_value(self.shade_quality_var.get())
        self.params.shade_quality_preset = quality_value

        if hasattr(self.main_gui, 'plot_manager') and self.main_gui.plot_manager is not None:
            self.main_gui.plot_manager.shade_quality_preset = quality_value

        # 仅在阴影模式下触发重绘，避免无谓刷新
        if hasattr(self.main_gui, 'data_loaded') and self.main_gui.data_loaded:
            if getattr(self.params, 'ishade', 0) != 0 and hasattr(self.main_gui, 'update_plot'):
                if hasattr(self.main_gui, 'request_plot_refresh'):
                    self.main_gui.request_plot_refresh(delay_ms=40)
                else:
                    self.main_gui.update_plot()
            if hasattr(self.main_gui, 'update_status'):
                self.main_gui.update_status(f'阴影质量已切换为：{self.shade_quality_var.get()}')
    
    def apply_all_params(self):
        """应用所有参数"""
        # 从控件读取值并更新参数
        range_param_changed = False
        for param_name, (widget, var, menu_id, param_type, widget_type) in self.param_widgets.items():
            try:
                if widget_type == 'combo':
                    # 从下拉框获取值（格式：'标签(值)'）
                    value_str = var.get()
                    # 提取括号中的值
                    if '(' in value_str and ')' in value_str:
                        value_str = value_str.split('(')[1].split(')')[0]
                    
                    # 特殊处理 ishade 参数：'>0' -> 1, '<0' -> -1
                    if param_name == 'ishade':
                        if value_str == '>0' or value_str == '1':
                            value = 1
                        elif value_str == '<0' or value_str == '-1':
                            value = -1
                        else:
                            value = int(value_str) if param_type == 'int' else float(value_str)
                    else:
                        value = int(value_str) if param_type == 'int' else float(value_str)
                elif widget_type == 'spin':
                    value = var.get()
                else:  # entry
                    value_str = var.get().strip()
                    # 如果控件为空，跳过更新（保持当前params值）
                    if not value_str:
                        continue
                    try:
                        value = int(value_str) if param_type == 'int' else float(value_str)
                    except ValueError:
                        # 如果转换失败，跳过更新
                        continue
                
                old_value = self.params.get_menu_value(menu_id)
                self.params.set_menu_value(menu_id, value)
                if menu_id in (23, 24, 27, 28):
                    try:
                        if float(old_value) != float(value):
                            range_param_changed = True
                    except Exception:
                        range_param_changed = True
            except (ValueError, AttributeError):
                # 参数值无效，静默跳过
                pass

        # 同步阴影质量档位（zplotpy 扩展参数）
        if hasattr(self, 'shade_quality_var'):
            self.params.shade_quality_preset = self._shade_quality_label_to_value(self.shade_quality_var.get())
        if hasattr(self.main_gui, 'plot_manager') and self.main_gui.plot_manager is not None:
            self.main_gui.plot_manager.shade_quality_preset = self.params.shade_quality_preset

        # 用户手动改了范围参数时，取消“保留缩放”状态，
        # 让 update_plot 严格按 xmin/xmax/tmin/tmax 绘制。
        if range_param_changed and hasattr(self.main_gui, 'preserve_zoom'):
            self.main_gui.preserve_zoom = False
            self.main_gui.user_xlim = None
            self.main_gui.user_ylim = None
        
        # 更新绘图
        if hasattr(self.main_gui, 'update_plot'):
            if hasattr(self.main_gui, 'request_plot_refresh'):
                self.main_gui.request_plot_refresh(delay_ms=40)
            else:
                self.main_gui.update_plot()
            try:
                self.main_gui.update_status(
                    "参数已应用 | "
                    f"X窗口=[{float(self.params.xmin):.3f}, {float(self.params.xmax):.3f}] | "
                    f"T窗口=[{float(self.params.tmin):.3f}, {float(self.params.tmax):.3f}]"
                )
            except Exception:
                self.main_gui.update_status('所有参数已应用')
        else:
            self.main_gui.update_status('所有参数已应用')
        
    def apply_display_params(self):
        """应用显示参数"""
        # 显示参数（包括 xmin/xmax/tmin/tmax）统一走 apply_all_params
        self.apply_all_params()
    
    def toggle_filter(self):
        """切换滤波开关"""
        enabled = self.filter_on_var.get()
        # 启用/禁用滤波相关控件
        state = 'normal' if enabled else 'disabled'
        self.freqlo_entry.config(state=state)
        self.freqhi_entry.config(state=state)
        self.npoles_entry.config(state=state)
    
    def apply_filter_params(self):
        """应用滤波参数"""
        try:
            # 带通滤波开关
            self.params.ibndps = 1 if self.filter_on_var.get() else 0
            
            # 零相位滤波
            self.params.izerop = 1 if self.zerophase_var.get() else 0
            
            # 频率范围
            self.params.freqlo = float(self.freqlo_entry.get())
            self.params.freqhi = float(self.freqhi_entry.get())
            
            # 滤波器阶数
            self.params.npoles = int(self.npoles_entry.get())
            
            self.main_gui.update_status('滤波参数已应用')
        except ValueError as e:
            from tkinter import messagebox
            messagebox.showerror('错误', f'参数值无效: {e}')
    
    def update_widgets_from_params(self):
        """从参数对象更新所有控件显示值"""
        # 更新 param_widgets 中的控件
        for param_name, (widget, var, menu_id, param_type, widget_type) in self.param_widgets.items():
            try:
                current_value = self.params.get_menu_value(menu_id)
                
                if widget_type == 'combo':
                    # 更新下拉框
                    try:
                        combo_values = widget.cget('values')
                    except:
                        try:
                            combo_values = widget['values']
                        except:
                            combo_values = []
                    
                    if param_name == 'ishade':
                        # 特殊处理 ishade
                        if current_value == 0:
                            var.set('无(0)')
                        elif current_value > 0:
                            var.set('正峰值(>0)')
                        elif current_value < 0:
                            var.set('负峰值(<0)')
                    else:
                        # 查找匹配的选项
                        current_str = str(current_value)
                        found = False
                        for val in combo_values:
                            if f'({current_str})' in val or val.startswith(current_str):
                                var.set(val)
                                found = True
                                break
                        if not found and combo_values:
                            # 如果找不到匹配，尝试设置第一个选项
                            var.set(combo_values[0])
                elif widget_type == 'spin':
                    # 更新 Spinbox
                    var.set(int(current_value))
                else:  # entry
                    # 更新 Entry
                    var.set(str(current_value))
            except (ValueError, AttributeError, KeyError):
                # 如果更新失败，静默跳过
                pass
        
        # 更新滤波标签页的特殊控件
        if hasattr(self, 'filter_on_var'):
            self.filter_on_var.set(self.params.ibndps != 0)
            self.toggle_filter()  # 更新控件状态
        
        if hasattr(self, 'freqlo_entry'):
            self.freqlo_entry.delete(0, tk.END)
            self.freqlo_entry.insert(0, str(self.params.freqlo))
        
        if hasattr(self, 'freqhi_entry'):
            self.freqhi_entry.delete(0, tk.END)
            self.freqhi_entry.insert(0, str(self.params.freqhi))
        
        if hasattr(self, 'zerophase_var'):
            self.zerophase_var.set(self.params.izerop != 0)
        
        if hasattr(self, 'npoles_entry'):
            self.npoles_entry.delete(0, tk.END)
            self.npoles_entry.insert(0, str(self.params.npoles))
        
        # 更新拾取标签页的特殊控件
        if hasattr(self, 'pick_word_var'):
            self.pick_word_var.set(self.params.apick)
        
        if hasattr(self, 'pick_size_entry'):
            self.pick_size_entry.delete(0, tk.END)
            self.pick_size_entry.insert(0, str(self.params.spick))
        
        # 更新显示标签页的特殊控件
        if hasattr(self, 'xmin_entry'):
            self.xmin_entry.delete(0, tk.END)
            self.xmin_entry.insert(0, str(self.params.xmin))
        
        if hasattr(self, 'xmax_entry'):
            self.xmax_entry.delete(0, tk.END)
            self.xmax_entry.insert(0, str(self.params.xmax))
        
        if hasattr(self, 'tmin_entry'):
            self.tmin_entry.delete(0, tk.END)
            self.tmin_entry.insert(0, str(self.params.tmin))
        
        if hasattr(self, 'tmax_entry'):
            self.tmax_entry.delete(0, tk.END)
            self.tmax_entry.insert(0, str(self.params.tmax))
        
        # 更新阴影质量控件（zplotpy 扩展）
        if hasattr(self, 'shade_quality_var'):
            self.shade_quality_var.set(
                self._shade_quality_value_to_label(
                    getattr(self.params, 'shade_quality_preset', 'balanced')
                )
            )
