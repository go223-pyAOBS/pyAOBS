"""TOMO2D shell plugin for Workbench node runner."""

from __future__ import annotations

import os
from pathlib import Path
import shlex

from .base import PluginCommandSpec, PluginValidationError


class Tomo2DShellPlugin:
    """
    Minimal TOMO2D plugin.

    It executes user-provided command line fragments under project workspace
    and records the run via RunManager.
    """

    id = "tomo2d.shell"
    name = "TOMO2D 命令节点"
    description = "运行 tt_inverse / tt_forward / gen_smesh 等命令"

    def build_spec(self, project_root: Path, payload: dict) -> PluginCommandSpec:
        executable = str(payload.get("executable", "")).strip()
        args_text = str(payload.get("args", "")).strip()
        node_id = str(payload.get("node_id", "")).strip() or "tomo2d_node"
        cwd_text = str(payload.get("cwd", "")).strip()
        env_text = str(payload.get("env_text", "")).strip()
        inputs_text = str(payload.get("inputs_text", "")).strip()

        if not executable:
            raise PluginValidationError("可执行程序不能为空。")

        command = [executable]
        if args_text:
            command.extend(_split_args(args_text))

        cwd: Path | None = None
        if cwd_text:
            p = Path(cwd_text)
            cwd = p if p.is_absolute() else (project_root / p)
            cwd = cwd.resolve()
            if not cwd.exists():
                raise PluginValidationError(f"工作目录不存在：{cwd}")

        env = _parse_env(env_text)
        inputs = _parse_paths(inputs_text, project_root)

        params = {
            "plugin": self.id,
            "executable": executable,
            "args_text": args_text,
            "cwd": str(cwd) if cwd else "",
        }
        return PluginCommandSpec(
            node_id=node_id,
            command=command,
            cwd=cwd,
            inputs=inputs,
            env=env,
            params=params,
        )


def _split_args(text: str) -> list[str]:
    # On Windows keep backslashes/quotes behavior closer to cmd usage.
    return shlex.split(text, posix=(os.name != "nt"))


def _parse_env(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    if not text:
        return env
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise PluginValidationError(f"环境变量格式错误（需 KEY=VALUE）：{line}")
        k, v = line.split("=", 1)
        key = k.strip()
        if not key:
            raise PluginValidationError(f"环境变量名为空：{line}")
        env[key] = v.strip()
    return env


def _parse_paths(text: str, project_root: Path) -> list[Path]:
    paths: list[Path] = []
    if not text:
        return paths
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line)
        ap = p if p.is_absolute() else (project_root / p)
        paths.append(ap.resolve())
    return paths

