# imodel_gui 启动指南

## 启动方式

### 方式1: 直接运行Python文件（最简单）

```bash
# Windows
python pyAOBS/visualization/imodel_gui.py

# Linux/Mac
python pyAOBS/visualization/imodel_gui.py
```

### 方式2: 作为模块运行（推荐）

```bash
# Windows
python -m pyAOBS.visualization.imodel_gui

# Linux/Mac
python -m pyAOBS.visualization.imodel_gui
```

### 方式3: 使用启动脚本（最方便）

**Windows**:
```bash
# 双击运行
imodel-gui.bat

# 或在命令行
cd pyAOBS/visualization
imodel-gui.bat
```

**Linux/Mac**:
```bash
# 先添加执行权限
chmod +x imodel-gui.sh

# 然后运行
./imodel-gui.sh
```

### 方式4: 在Python代码中启动

```python
from pyAOBS.visualization.imodel_gui import main

# 启动GUI
main()
```

或者：

```python
from pyAOBS.visualization.imodel_gui import InteractiveModelViewerGUI
import tkinter as tk

# 创建根窗口
root = tk.Tk()
app = InteractiveModelViewerGUI(master=root, grid_file='velocity.grd')
app.run()
```

## 启动时指定模型文件

### 命令行方式

```bash
# 方式1: 修改代码，在main()函数中指定
# 方式2: 启动后通过GUI菜单"文件->打开模型"加载
```

### 代码方式

```python
from pyAOBS.visualization.imodel_gui import InteractiveModelViewerGUI
import tkinter as tk

root = tk.Tk()
# 启动时加载模型
app = InteractiveModelViewerGUI(master=root, grid_file='path/to/velocity.grd')
app.run()
```

## 常见问题

### 1. 找不到模块

**错误**: `ModuleNotFoundError: No module named 'pyAOBS'`

**解决**:
```bash
# 确保在项目根目录运行，或添加路径
export PYTHONPATH=/path/to/pyAOBS:$PYTHONPATH  # Linux/Mac
set PYTHONPATH=D:\path\to\pyAOBS;%PYTHONPATH%  # Windows
```

### 2. Tkinter未安装

**错误**: `ModuleNotFoundError: No module named 'tkinter'`

**解决**:
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# CentOS/RHEL
sudo yum install python3-tkinter

# Mac (通常已预装)
# Windows (通常已预装)
```

### 3. 窗口无法显示

**检查**:
- 确保使用图形界面环境（不是SSH无图形界面）
- 检查DISPLAY环境变量（Linux）
- 确保matplotlib后端设置为TkAgg

## 快速测试

```python
# test_imodel_gui.py
from pyAOBS.visualization.imodel_gui import InteractiveModelViewerGUI
import tkinter as tk

if __name__ == '__main__':
    root = tk.Tk()
    # 不指定grid_file，启动后手动打开
    app = InteractiveModelViewerGUI(master=root)
    app.run()
```

运行：
```bash
python test_imodel_gui.py
```

## 与vedit启动方式对比

| 特性 | vedit | imodel_gui |
|------|-------|------------|
| 启动脚本 | vin-editor.bat/sh | imodel-gui.bat/sh ✅ |
| 模块运行 | python main.py | python -m pyAOBS.visualization.imodel_gui ✅ |
| 直接运行 | ✅ | ✅ |
| 代码导入 | ✅ | ✅ |

## 推荐启动方式

**开发环境**: 使用方式2（模块运行）
```bash
python -m pyAOBS.visualization.imodel_gui
```

**生产环境**: 使用方式3（启动脚本）
```bash
# Windows
imodel-gui.bat

# Linux/Mac
./imodel-gui.sh
```
