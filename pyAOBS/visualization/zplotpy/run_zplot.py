#!/usr/bin/env python
"""
run_zplot.py - 兼容启动脚本（推荐改用模块启动）

推荐命令：
    python -m pyAOBS.visualization.zplotpy.gui
或：
    python -m pyaobs.visualization.zplotpy.gui
"""

try:
    from .gui import main
except ImportError:
    # 兼容直接以脚本方式执行：python run_zplot.py
    from gui import main


if __name__ == '__main__':
    main()
