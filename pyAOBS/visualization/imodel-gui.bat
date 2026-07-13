@echo off
REM Interactive Velocity Model Viewer GUI启动脚本 (Windows)
REM 参考vedit的启动方式

cd /d %~dp0
python -m pyAOBS.visualization.imodel_gui %*
