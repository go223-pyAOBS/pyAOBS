#!/usr/bin/env bash
# pyAOBS 发布脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "pyAOBS 发布脚本"
echo "=========================================="

# 检查是否安装了必要的工具
echo "检查构建工具..."
python -m pip install --upgrade build twine --quiet

# 清理旧的构建文件
echo "清理旧的构建文件..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/

# 构建包
echo "构建分发包..."
python -m build

# 检查包（使用python -m twine避免路径问题）
echo "检查构建的包..."
python -m twine check dist/*

echo ""
echo "=========================================="
echo "构建完成！"
echo "=========================================="
echo ""
echo "生成的文件："
ls -lh dist/
echo ""
echo "下一步："
echo "1. 测试安装: pip install dist/pyAOBS-*.whl"
echo "2. 冒烟: python -c \"import pyAOBS; print(pyAOBS.__version__)\""
echo "3. （可选）测试PyPI: python -m twine upload --repository testpypi dist/*"
echo "4. （可选）正式PyPI: python -m twine upload dist/*"
echo ""
echo "3.0.0rc1 建议先 GitHub Release / tag，确认 GUI 后再上 PyPI。"
echo "注意：上传需要 PyPI 账号和 API token"
echo "=========================================="
