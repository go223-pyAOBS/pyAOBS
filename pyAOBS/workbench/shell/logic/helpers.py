"""Pure helpers extracted from Workbench shell."""

from __future__ import annotations

import fnmatch
import os
import shlex
import shutil
from pathlib import Path


def ordered_plugin_ids(plugin_ids: list[str]) -> list[str]:
    preferred = [
        "data.gui",
        "zplotpy.gui",
        "tomo2d.gui",
        "tomo2d.shell",
        "imodel.gui",
        "iphase.gui",
        "petrology.lip.gui",
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for pid in preferred:
        if pid in plugin_ids and pid not in seen:
            ordered.append(pid)
            seen.add(pid)
    for pid in plugin_ids:
        if pid not in seen:
            ordered.append(pid)
            seen.add(pid)
    return ordered


def quote_arg_if_needed(s: str) -> str:
    if not s:
        return s
    if any(ch.isspace() for ch in s) and not (s.startswith('"') and s.endswith('"')):
        return f'"{s}"'
    return s


def with_text_padding(content: str) -> str:
    text = (content or "").strip("\n")
    if not text:
        return "\n\n"
    return text + "\n\n"


def read_tail_text_with_trunc(path: Path, max_chars: int = 12000) -> tuple[str, bool]:
    if not path.exists() or not path.is_file():
        return "(日志文件不存在)", False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"(日志读取失败: {exc})", False
    if len(text) <= max_chars:
        return (text if text else "(空日志)"), False
    return text[-max_chars:], True


def parse_batch_recent_values(values) -> dict[str, str] | None:
    if not values or not isinstance(values, (list, tuple)):
        return None
    if len(values) < 3:
        return None
    return {
        "source_run_id": str(values[0]),
        "status": str(values[1]),
        "new_run_id": str(values[2]),
    }


def parse_name_filter_tokens(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    return [t.strip() for t in text.replace(",", " ").split() if t.strip()]


def match_name_filter(filename: str, tokens: list[str]) -> bool:
    if not tokens:
        return True
    name = filename.lower()
    for tok in tokens:
        if "*" in tok or "?" in tok:
            if fnmatch.fnmatchcase(name, tok):
                return True
        elif tok in name:
            return True
    return False


def looks_like_tx_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".tx") or name.endswith(".txi")


def extract_option_value(tokens: list[str], option: str) -> str | None:
    for idx, tok in enumerate(tokens):
        if tok == option:
            if idx + 1 < len(tokens):
                return tokens[idx + 1]
        if tok.startswith(option) and len(tok) > len(option):
            return tok[len(option) :]
    return None


def build_risk_warnings(executable: str, args_text: str, env: dict[str, str]) -> list[str]:
    risks: list[str] = []
    try:
        tokens = shlex.split(args_text, posix=False)
    except ValueError:
        tokens = []

    if executable in ("tt_inverse", "tt_forward"):
        dv = extract_option_value(tokens, "-DV")
        dq = extract_option_value(tokens, "-DQ")
        if dq is not None:
            if dv is None:
                risks.append("检测到 -DQ 但未提供 -DV，通常会导致阻尼文件不生效。")
            else:
                try:
                    if float(dv) <= 0:
                        risks.append("检测到 -DQ 与 -DV<=0 组合，建议设置 -DV 为正值。")
                except ValueError:
                    risks.append("检测到 -DV 非数值，建议检查参数拼写。")

    if executable == "tt_inverse" and "TOMO2D_INV_OMP" not in env:
        risks.append("未设置 TOMO2D_INV_OMP，默认可能串行。")
    if executable == "tt_forward" and "TOMO2D_FWD_OMP" not in env:
        risks.append("未设置 TOMO2D_FWD_OMP，默认可能串行。")
    if executable in ("tt_inverse", "tt_forward"):
        if "OMP_PLACES" not in env or "OMP_PROC_BIND" not in env:
            risks.append("建议显式设置 OMP_PLACES 与 OMP_PROC_BIND 以获得更稳定并行性能。")
    if executable and shutil.which(executable) is None:
        risks.append(f"当前 PATH 未找到可执行程序：{executable}（若使用绝对路径可忽略）。")
    return risks


def normalize_payload_for_plugin(plugin_id: str, payload: dict[str, str]) -> dict[str, str]:
    if not plugin_id.endswith(".gui"):
        return payload
    normalized = dict(payload)
    normalized["cwd"] = ""
    normalized["inputs_text"] = ""
    return normalized


def should_auto_preflight(plugin_id: str) -> bool:
    force = str(os.environ.get("PYAOBS_WORKBENCH_AUTO_PREFLIGHT", "")).strip().lower()
    if force in {"1", "true", "yes", "on"}:
        return True
    return plugin_id == "tomo2d.shell"


def parse_env_text(env_text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in str(env_text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip()
    return out


def parse_inputs_text(inputs_text: str) -> list[Path]:
    paths: list[Path] = []
    for line in str(inputs_text or "").splitlines():
        line = line.strip()
        if line:
            paths.append(Path(line))
    return paths
