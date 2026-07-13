#!/bin/bash
# ZPLOT 启动脚本 (Linux/Mac)

cd "$(dirname "$0")"
python -m pyaobs.visualization.zplotpy.gui
