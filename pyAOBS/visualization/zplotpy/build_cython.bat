@echo off
REM 编译Cython扩展的批处理脚本（Windows）

echo Building Cython extensions...
python setup_cython.py build_ext --inplace

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Build successful! Cython extensions are ready.
) else (
    echo.
    echo Build failed! Please check the error messages above.
    pause
)
