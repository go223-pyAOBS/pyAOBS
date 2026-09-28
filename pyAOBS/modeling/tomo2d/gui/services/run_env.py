"""并行 / 策略环境变量：从表单收集并交给子进程（无 UI）。"""

from __future__ import annotations

import os
from typing import Any

from ..state.form_state import FormState

# FormState 键 → 环境变量名
_BOOL_ENV = (
    ("env.inv_omp", "TOMO2D_INV_OMP"),
    ("env.fwd_omp", "TOMO2D_FWD_OMP"),
    ("env.inv_legacy_baseline", "TOMO2D_INV_LEGACY_BASELINE"),
    ("env.inv_lsqr_precond", "TOMO2D_INV_LSQR_PRECOND"),
    ("env.inv_sens_weight", "TOMO2D_INV_SENS_WEIGHT"),
    ("env.inv_linesearch", "TOMO2D_INV_LINESEARCH"),
    ("env.inv_lm", "TOMO2D_INV_LM"),
    ("env.inv_reuse_forward", "TOMO2D_INV_REUSE_FORWARD"),
    ("env.inv_coarse2fine", "TOMO2D_INV_COARSE2FINE"),
    ("env.inv_diag", "TOMO2D_INV_DIAG"),
    ("env.graph_fs_enum", "TOMO2D_GRAPH_FS_ENUM"),
)

_STR_ENV = (
    ("env.omp_num_threads", "OMP_NUM_THREADS"),
    ("env.inv_reuse_thresh", "TOMO2D_INV_REUSE_THRESH"),
    ("env.inv_lsqr_precond_max", "TOMO2D_INV_LSQR_PRECOND_MAX"),
    ("env.inv_sens_kappa", "TOMO2D_INV_SENS_KAPPA"),
    ("env.inv_c2f_smooth_start", "TOMO2D_INV_C2F_SMOOTH_START"),
    ("env.inv_c2f_smooth_end", "TOMO2D_INV_C2F_SMOOTH_END"),
    ("env.inv_c2f_damp_start", "TOMO2D_INV_C2F_DAMP_START"),
    ("env.inv_c2f_damp_end", "TOMO2D_INV_C2F_DAMP_END"),
    # 相对子进程 cwd；运行包下即为 outputs/status.jsonl
    ("env.inv_status_jsonl_path", "TOMO2D_INV_STATUS_JSONL"),
)

# 布尔关时写入 "0"；开时 "1"。字符串空则不设置（继承系统环境）。
_DEFAULT_BOOL = {
    "env.inv_omp": False,
    "env.fwd_omp": False,
    "env.inv_legacy_baseline": False,
    "env.inv_lsqr_precond": False,
    "env.inv_sens_weight": False,
    "env.inv_linesearch": False,
    "env.inv_lm": False,
    "env.inv_reuse_forward": False,
    "env.inv_coarse2fine": False,
    "env.inv_diag": False,
    "env.graph_fs_enum": True,
}


def ensure_run_env_defaults(state: FormState) -> None:
    """补齐缺失键；线程数默认取当前进程环境（若有）。"""
    if not state.has("env.omp_num_threads"):
        state.set("env.omp_num_threads", os.environ.get("OMP_NUM_THREADS", ""))
    for key, default in _DEFAULT_BOOL.items():
        if not state.has(key):
            # 若进程已设置对应环境变量为非 0，则默认勾选
            env_name = dict(_BOOL_ENV).get(key)
            if env_name and os.environ.get(env_name, "").strip() not in ("", "0"):
                state.set(key, True)
            else:
                state.set(key, default)
    for key, _ in _STR_ENV:
        if key == "env.omp_num_threads":
            continue
        if not state.has(key):
            if key == "env.inv_status_jsonl_path":
                # 默认开启轻量监视通道（空字符串可关闭）
                state.set(key, "outputs/status.jsonl")
            else:
                state.set(key, "")


def collect_run_env(state: FormState) -> dict[str, str]:
    """
    返回子进程应使用的环境变量覆盖表（相对 ``os.environ`` 的增量）。

    - 布尔项：始终写入 ``1`` 或 ``0``（明确开关，避免继承旧 shell 值造成困惑）
    - 字符串项：非空才写入；空则不覆盖（保留系统环境）
    - Legacy baseline 开启时仍写出 reuse/c2f（C 端会忽略），便于日志对照
    """
    ensure_run_env_defaults(state)
    out: dict[str, str] = {}
    for key, env_name in _BOOL_ENV:
        out[env_name] = "1" if state.get_bool(key) else "0"
    for key, env_name in _STR_ENV:
        raw = state.get_str(key)
        if raw:
            out[env_name] = raw
    return out


def format_run_env_preview(env: dict[str, str]) -> str:
    """预览用短摘要。"""
    if not env:
        return ""
    lines = ["# 运行环境变量（子进程）:"]
    for k in sorted(env.keys()):
        lines.append(f"#   {k}={env[k]}")
    return "\n".join(lines)


def format_strategy_flags(env: dict[str, str] | None) -> str:
    """已开的加速/对拍策略（关的不写）。"""
    if not env:
        return ""
    bits: list[str] = []
    if env.get("TOMO2D_INV_LEGACY_BASELINE") == "1":
        bits.append("Legacy")
        if env.get("TOMO2D_INV_DIAG") == "1":
            bits.append("DIAG")
        return " · ".join(bits)
    if env.get("TOMO2D_INV_REUSE_FORWARD") == "1":
        th = (env.get("TOMO2D_INV_REUSE_THRESH") or "").strip()
        bits.append("前向复用" + (f" 阈={th}" if th else ""))
    if env.get("TOMO2D_INV_COARSE2FINE") == "1":
        bits.append("C2F")
    if env.get("TOMO2D_INV_LSQR_PRECOND") == "1":
        mx = (env.get("TOMO2D_INV_LSQR_PRECOND_MAX") or "").strip() or "10"
        bits.append(f"列预条件 maxD={mx}")
    if env.get("TOMO2D_INV_SENS_WEIGHT") == "1":
        kap = (env.get("TOMO2D_INV_SENS_KAPPA") or "").strip() or "10"
        bits.append(f"灵敏度加权 κ={kap}")
    if env.get("TOMO2D_INV_LINESEARCH") == "1":
        bits.append("线搜索")
    if env.get("TOMO2D_INV_LM") == "1":
        bits.append("LM")
    if env.get("TOMO2D_INV_DIAG") == "1":
        bits.append("DIAG")
    if env.get("TOMO2D_GRAPH_FS_ENUM") == "1":
        bits.append("图论FS")
    return " · ".join(bits)


def strategy_env_from_form(data: dict[str, Any] | None) -> dict[str, str]:
    """gui_profile / FormState 字典 → 策略相关环境变量（供 ``format_strategy_flags``）。"""
    if not data:
        return {}
    out: dict[str, str] = {}

    def _on(key: str) -> bool:
        v = data.get(key)
        if v is True or v == 1:
            return True
        if isinstance(v, str) and v.strip().lower() in ("1", "true", "yes", "on"):
            return True
        return False

    def _put_bool(key: str, env_name: str) -> None:
        if key not in data:
            return
        out[env_name] = "1" if _on(key) else "0"

    _put_bool("env.inv_legacy_baseline", "TOMO2D_INV_LEGACY_BASELINE")
    _put_bool("env.inv_reuse_forward", "TOMO2D_INV_REUSE_FORWARD")
    _put_bool("env.inv_coarse2fine", "TOMO2D_INV_COARSE2FINE")
    _put_bool("env.inv_lsqr_precond", "TOMO2D_INV_LSQR_PRECOND")
    _put_bool("env.inv_sens_weight", "TOMO2D_INV_SENS_WEIGHT")
    _put_bool("env.inv_linesearch", "TOMO2D_INV_LINESEARCH")
    _put_bool("env.inv_lm", "TOMO2D_INV_LM")
    _put_bool("env.inv_diag", "TOMO2D_INV_DIAG")
    _put_bool("env.graph_fs_enum", "TOMO2D_GRAPH_FS_ENUM")
    th = str(data.get("env.inv_reuse_thresh") or "").strip()
    if th:
        out["TOMO2D_INV_REUSE_THRESH"] = th
    mx = str(data.get("env.inv_lsqr_precond_max") or "").strip()
    if mx:
        out["TOMO2D_INV_LSQR_PRECOND_MAX"] = mx
    kap = str(data.get("env.inv_sens_kappa") or "").strip()
    if kap:
        out["TOMO2D_INV_SENS_KAPPA"] = kap
    return out


def format_parallel_status_line(env: dict[str, str] | None) -> str:
    """执行日志一行：OMP / 正反演并行开关（不依赖 C++ 是否打印）。"""
    if not env:
        return ""
    thr = (env.get("OMP_NUM_THREADS") or "").strip() or "系统默认"
    inv = env.get("TOMO2D_INV_OMP", "0")
    fwd = env.get("TOMO2D_FWD_OMP", "0")
    bits = [
        f"OMP_NUM_THREADS={thr}",
        f"tt_inverse并行={'开' if inv == '1' else '关'}",
        f"tt_forward并行={'开' if fwd == '1' else '关'}",
    ]
    extra = format_strategy_flags(env)
    if extra:
        bits.append(extra)
    return "并行: " + " · ".join(bits)


def merge_subprocess_env(overrides: dict[str, str] | None) -> dict[str, str] | None:
    """合并进完整 env 字典供 ``subprocess.run(..., env=)``；无覆盖则返回 None。"""
    if not overrides:
        return None
    env = dict(os.environ)
    for k, v in overrides.items():
        env[str(k)] = str(v)
    return env
