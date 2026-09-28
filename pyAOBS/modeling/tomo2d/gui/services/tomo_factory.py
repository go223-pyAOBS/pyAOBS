"""TomoAnd 工厂（无 UI）。"""

from __future__ import annotations

from ...tomand import TomoAnd


def get_tomo(bin_path: str | None) -> TomoAnd:
    """``bin_path`` 可为相对路径；``TomoAnd`` 内部会抬成绝对路径。"""
    bp = (bin_path or "").strip() or None
    return TomoAnd(bin_path=bp)
