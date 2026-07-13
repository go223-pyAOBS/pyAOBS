# imodel.py GUI实现更新说明

## 更新内容

根据用户要求，参考vedit的实现方式，创建了基于Tkinter的GUI版本。

## 主要变化

### 1. 新增文件

- **`imodel_gui.py`**: 基于Tkinter的GUI主文件
  - 参考vedit的架构设计
  - 使用FigureCanvasTkAgg嵌入matplotlib
  - 提供完整的GUI界面

### 2. 架构调整

**之前**：
- 单一文件 `imodel.py`，使用纯matplotlib窗口

**现在**：
- `imodel.py`: 核心功能类（无GUI依赖）
- `imodel_gui.py`: GUI界面（参考vedit）

### 3. 技术实现对比

| 特性 | vedit | imodel_gui.py |
|------|-------|---------------|
| GUI框架 | Tkinter | Tkinter ✅ |
| Matplotlib集成 | FigureCanvasTkAgg | FigureCanvasTkAgg ✅ |
| 工具栏 | NavigationToolbar2Tk | NavigationToolbar2Tk ✅ |
| 侧边栏 | ttk.Frame | ttk.Frame ✅ |
| 菜单栏 | tk.Menu | tk.Menu ✅ |
| 事件处理 | matplotlib事件系统 | matplotlib事件系统 ✅ |

## 参考vedit的关键实现

### 1. 窗口创建

```python
# vedit方式
self.root = tk.Tk()
self.root.title('v.in editor')
self.root.geometry('1000x600+200+30')

# imodel_gui.py
self.root = tk.Tk()
self.root.title('Interactive Velocity Model Viewer')
self.root.geometry('1200x800+100+50')
```

### 2. Matplotlib嵌入

```python
# vedit方式
self.fig = plt.Figure(tight_layout=True)
self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nswe')
self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)

# imodel_gui.py（相同方式）
self.fig = plt.Figure(figsize=(10, 8), tight_layout=True)
self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nswe')
self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
```

### 3. 侧边栏布局

```python
# vedit方式
side_area = ttk.Frame(self)
side_area.grid(row=0, column=1, rowspan=2, padx=(0, 15), pady=(0, 15), sticky='nswe')

# imodel_gui.py
side_frame = ttk.Frame(main_frame)
side_frame.grid(row=0, column=1, rowspan=2, padx=(10, 0), sticky='nswe')
```

### 4. 菜单栏

```python
# vedit方式
menubar = tk.Menu(self)
self.master.config(menu=menubar)
filemenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='File', menu=filemenu)

# imodel_gui.py（相同方式）
menubar = tk.Menu(self.root)
self.root.config(menu=menubar)
filemenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='文件', menu=filemenu)
```

## 功能对比

| 功能 | vedit | imodel_gui.py |
|------|-------|---------------|
| 打开文件 | ✅ | ✅ |
| 保存图形 | ✅ | ✅ |
| 交互式绘图 | ✅ | ✅ |
| 点选择 | ❌ | ✅ |
| 多边形选择 | ❌ | ✅ |
| 剖面提取 | ✅ | ✅ |
| 物性计算 | ❌ | ✅ |
| 重力计算 | ❌ | ✅ |
| 岩石分类 | ❌ | ✅ |

## 使用方式

### GUI方式（推荐）

```python
from pyAOBS.visualization.imodel_gui import main
main()
```

### 编程方式

```python
from pyAOBS.visualization.imodel import InteractiveModelViewer
viewer = InteractiveModelViewer('velocity.grd')
viewer.show()
```

## 优势

1. **统一的用户体验**: 与vedit保持一致的界面风格
2. **更好的集成**: 可以轻松集成到现有工具链
3. **功能完整**: 保留了所有核心功能
4. **易于扩展**: 可以添加更多GUI功能

## 后续改进建议

1. 添加更多vedit风格的控件（滑块、下拉菜单等）
2. 实现多窗口管理（如vedit的速度窗口）
3. 添加配置文件保存/加载功能
4. 实现撤销/重做功能（参考vedit的command_history）
