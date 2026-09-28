"""正反演默认关闭非必要磁盘/终端输出（仅补齐缺失键，不覆盖用户已填）。"""

from __future__ import annotations

from ..state.form_state import FormState

# 缺失时写入的「安静」默认：少写盘、少刷屏；监视靠 -L / status.jsonl
_IO_DEFAULTS: dict[str, object] = {
    # tt_inverse：不写中间射线/残差；少终端；默认只写最终模型（大幅减 smesh I/O）
    "inv.out_level": "",
    "inv.verbose_level": "",
    "inv.print_final_only": True,
    "inv.dws_file": "",
    # tt_forward：默认不写射线文件（很重）；verbose 关
    "fwd.out_ray": "",
    "fwd.verbose_level": "",
    "fwd.out_elements": "",
    "fwd.out_obs_ttime": "",
    "fwd.out_source": "",
    "fwd.out_vgrid": "",
    "fwd.out_diff": "",
}


def ensure_quiet_io_defaults(state: FormState) -> list[str]:
    """
    为尚未出现的键写入低 I/O 默认。

    返回实际写入的键名列表（便于日志/测试）。
    """
    applied: list[str] = []
    for key, val in _IO_DEFAULTS.items():
        if not state.has(key):
            state.set(key, val)
            applied.append(key)
    return applied
