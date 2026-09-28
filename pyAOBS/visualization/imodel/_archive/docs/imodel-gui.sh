#!/bin/bash
# Interactive Velocity Model Viewer GUI启动脚本 (Linux/Mac)
# 参考vedit的启动方式

cd "$(dirname "$0")"
python -m pyAOBS.visualization.imodel_gui "$@"
