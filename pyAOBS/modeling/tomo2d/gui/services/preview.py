"""预览文本辅助（无 UI）。"""

from __future__ import annotations

import shlex
from typing import Any, Callable


def format_resolved_cmdline(args: Any) -> str:
    """将 subprocess 实际使用的 argv 列表格式化为可复制的 shell 风格命令行。"""
    if args is None:
        return ""
    if isinstance(args, str):
        return args
    seq = [str(a) for a in args]
    if not seq:
        return ""
    try:
        return shlex.join(seq)
    except (TypeError, ValueError):
        return " ".join(seq)


def preview_append_resolved_cmdline(text: str, resolve: Callable[[], Any]) -> str:
    """在 Python 调用预览后追加与 subprocess 一致的 argv 命令行。"""
    try:
        cmd = resolve()
        if cmd:
            return f"{text}\n\n# 解析后命令行:\n{format_resolved_cmdline(cmd)}"
        return (
            f"{text}\n\n# 解析后命令行: （必选参数不齐或与「仅打印帮助」路径一致，未拼可执行 argv）"
        )
    except Exception as e:
        return f"{text}\n\n# 解析后命令行: （无法解析: {e}）"
