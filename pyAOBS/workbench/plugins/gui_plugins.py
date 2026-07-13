"""Additional GUI launcher plugins for Workbench."""

from __future__ import annotations

import os
from pathlib import Path
import shlex

from .base import PluginCommandSpec, PluginValidationError


class _PythonModuleGuiPlugin:
    """Generic python -m launcher plugin."""

    id = "gui.python_module"
    name = "Python 模块 GUI"
    description = "通过 python -m 启动 GUI 模块"
    module = ""
    default_node = "gui_node"

    def build_spec(self, project_root: Path, payload: dict) -> PluginCommandSpec:
        executable = str(payload.get("executable", "")).strip() or "python"
        args_text = str(payload.get("args", "")).strip()
        node_id = str(payload.get("node_id", "")).strip() or self.default_node
        cwd_text = str(payload.get("cwd", "")).strip()
        env_text = str(payload.get("env_text", "")).strip()
        inputs_text = str(payload.get("inputs_text", "")).strip()

        if not self.module:
            raise PluginValidationError("插件 module 未配置。")

        command = [executable, "-m", self.module]
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
            "launcher": "python_module",
            "module": self.module,
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


class ZplotpyGuiPlugin(_PythonModuleGuiPlugin):
    id = "zplotpy.gui"
    name = "ZPlotPy GUI"
    description = "启动震相拾取 GUI（zplotpy）"
    module = "pyAOBS.workbench.gui_audit_launchers.zplotpy_gui"
    default_node = "zplotpy_gui"


class ImodelGuiPlugin(_PythonModuleGuiPlugin):
    id = "imodel.gui"
    name = "iModel GUI"
    description = "启动速度模型解释 GUI（imodel，Qt / PySide6）"
    module = "pyAOBS.workbench.gui_audit_launchers.imodel_gui"
    default_node = "imodel_gui"


class IphaseGuiPlugin(_PythonModuleGuiPlugin):
    id = "iphase.gui"
    name = "iPhase GUI"
    description = "启动震相分析 GUI（iphase）"
    module = "pyAOBS.workbench.gui_audit_launchers.iphase_gui"
    default_node = "iphase_gui"


class Tomo2dGuiPlugin(_PythonModuleGuiPlugin):
    id = "tomo2d.gui"
    name = "TOMO2D GUI"
    description = "启动 TOMO2D 图形界面"
    module = "pyAOBS.modeling.tomo2d.gui"
    default_node = "tomo2d_gui"


class PetrologyLipGuiPlugin(_PythonModuleGuiPlugin):
    id = "petrology.lip.gui"
    name = "LIP Petrology GUI"
    description = "启动 LIP 地幔熔融解释 GUI（REEBOX + 岩性预设，PySide6）"
    module = "pyAOBS.workbench.gui_audit_launchers.lip_gui"
    default_node = "lip_petrology_gui"


class ObemTsmToSacPlugin(_PythonModuleGuiPlugin):
    id = "processor.obem_tsm_to_sac"
    name = "OBEM TSM -> SAC"
    description = "启动 processors 转换：obem_tsm_to_sac_obspy"
    module = "pyAOBS.processors.raw2sac.obem_tsm_to_sac_obspy"
    default_node = "obem_tsm_to_sac"


class Sac2yPlugin(_PythonModuleGuiPlugin):
    id = "processor.sac2y_v2_1_obspy"
    name = "SAC -> SEGY (sac2y v2.1)"
    description = "启动 processors 转换：sac2y_v2_1_obspy"
    module = "pyAOBS.processors.raw2sac.sac2y_v2_1_obspy"
    default_node = "sac2y_v2_1"


class Raw2SacPlugin(_PythonModuleGuiPlugin):
    id = "processor.raw2sac_v1_1_obspy"
    name = "RAW -> SAC (raw2sac v1.1)"
    description = "启动 processors 转换：raw2sac_v1_1_obspy"
    module = "pyAOBS.processors.raw2sac.raw2sac_v1_1_obspy"
    default_node = "raw2sac_v1_1"


class DataGuiPlugin:
    id = "data.gui"
    name = "idata 数据转换节点"
    description = "启动统一数据转换 UI（idata）"
    default_node = "data_gui"

    def build_spec(self, project_root: Path, payload: dict) -> PluginCommandSpec:
        executable = str(payload.get("executable", "")).strip() or "python"
        args_text = str(payload.get("args", "")).strip()
        node_id = str(payload.get("node_id", "")).strip() or self.default_node
        cwd_text = str(payload.get("cwd", "")).strip()
        env_text = str(payload.get("env_text", "")).strip()
        inputs_text = str(payload.get("inputs_text", "")).strip()

        script_path = Path(__file__).resolve().parents[2] / "processors" / "raw2sac" / "idata.py"
        if not script_path.exists():
            raise PluginValidationError(f"idata 启动脚本不存在：{script_path}")

        command = [executable, str(script_path)]
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
            "launcher": "python_script",
            "script": str(script_path),
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

