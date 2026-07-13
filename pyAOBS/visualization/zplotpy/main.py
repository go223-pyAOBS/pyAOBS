"""
main.py - 兼容主程序入口（推荐改用 gui 模块入口）
"""

try:
    from .gui import main
except ImportError:
    # 兼容直接以脚本方式执行
    from gui import main


if __name__ == '__main__':
    main()
