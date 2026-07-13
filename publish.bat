@echo off
REM pyAOBS 发布脚本 (Windows)

echo ==========================================
echo pyAOBS 发布脚本
echo ==========================================

REM 检查是否安装了必要的工具
echo 检查构建工具...
python -m pip install --upgrade build twine --quiet

REM 清理旧的构建文件
echo 清理旧的构建文件...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
for /d %%d in (*.egg-info) do rmdir /s /q %%d

REM 构建包
echo 构建分发包...
python -m build

REM 检查包
echo 检查构建的包...
twine check dist/*

echo.
echo ==========================================
echo 构建完成！
echo ==========================================
echo.
echo 生成的文件：
dir dist
echo.
echo 下一步：
echo 1. 测试安装: pip install dist\pyAOBS-*.whl
echo 2. 上传到测试PyPI: python -m twine upload --repository testpypi dist/*
echo 3. 上传到正式PyPI: python -m twine upload dist/*
echo.
echo 注意：上传需要 PyPI 账号和 API token
echo ==========================================
pause
