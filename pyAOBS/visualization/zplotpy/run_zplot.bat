@echo off
REM ZPLOT 启动脚本 (Windows)

cd /d %~dp0
python -m pyaobs.visualization.zplotpy.gui

pause
