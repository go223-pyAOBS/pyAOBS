"""GUI quick-form → run payload (toolkit independent)."""

from __future__ import annotations

from pathlib import Path

from ...petrology_launcher import petrology_obs_path_from_state, petrology_transect_path_from_state
from .helpers import quote_arg_if_needed


def petrology_lip_import_args_from_state(state_path: str | Path) -> list[str]:
    state_path = Path(state_path)
    if not state_path.is_file():
        return []
    args: list[str] = []
    obs_path = petrology_obs_path_from_state(state_path)
    tran_path = petrology_transect_path_from_state(state_path)
    if obs_path:
        args.extend(["--import-obs", quote_arg_if_needed(str(obs_path))])
    if tran_path:
        args.extend(["--import-transect", quote_arg_if_needed(str(tran_path))])
    return args


def apply_gui_quick_form_to_command(
    plugin_id: str,
    *,
    node_id: str,
    gui_form: dict[str, str],
) -> tuple[str, str, str]:
    """
    Return (node_id, args, status_message) for GUI plugins.
    executable is always 'python'; cwd/inputs cleared by caller.
    """
    if plugin_id == "data.shell":
        plugin_id = "data.gui"

    if plugin_id == "tomo2d.shell":
        return node_id, "", "当前是 TOMO2D 插件，请使用模板表单。"

    if plugin_id == "zplotpy.gui":
        nid = node_id.strip() or "OBS_node"
        return nid, "", "zplotpy.gui 启动后请在 GUI 内部加载输入文件。"

    if plugin_id == "imodel.gui":
        nid = str(gui_form.get("imodel_node", "")).strip() or "OBS_node"
        return nid, "", "imodel（Qt）启动后请在 GUI 内部加载模型与辅助文件。"

    if plugin_id == "iphase.gui":
        nid = str(gui_form.get("iphase_node", "")).strip() or "OBS_node"
        return nid, "", "iphase.gui 启动后请在 GUI 内部加载输入文件。"

    if plugin_id == "data.gui":
        nid = node_id.strip() or "OBS_node"
        return nid, "", "data.gui 直接启动 idata UI，无需额外参数。"

    if plugin_id == "tomo2d.gui":
        nid = node_id.strip() or "OBS_node"
        return nid, "", "tomo2d.gui 直接启动 TOMO2D GUI，无需额外参数。"

    if plugin_id == "petrology.lip.gui":
        nid = node_id.strip() or "lip_petrology"
        state_text = str(gui_form.get("petrology_state", "")).strip()
        args_list = petrology_lip_import_args_from_state(state_text) if state_text else []
        args = " ".join(args_list)
        if args_list:
            msg = "petrology.lip.gui 将携带 imodel 观测/沿迹 CSV 启动 LIP Petrology。"
        else:
            msg = (
                "petrology.lip.gui 启动 LIP 地幔熔融 GUI。"
                "可在桥接区指定 gui_state 后重新「应用GUI表单到命令」。"
            )
        return nid, args, msg

    return node_id, "", "当前插件未提供 GUI 快速表单。"


def find_latest_imodel_gui_state(project_root: Path) -> Path | None:
    runs_dir = project_root / "runs"
    if not runs_dir.is_dir():
        return None
    try:
        run_dirs = sorted(runs_dir.iterdir(), key=lambda p: p.name, reverse=True)
    except OSError:
        return None
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        plugin = str((manifest.get("params") or {}).get("plugin", ""))
        if plugin != "imodel.gui":
            continue
        gs = manifest.get("gui_state") or {}
        rel = str(gs.get("state_file", "")).strip()
        if not rel:
            continue
        state_path = (project_root / rel).resolve()
        if state_path.is_file():
            return state_path
    return None


def petrology_bridge_status(state_path: str | Path) -> str:
    state_path = Path(state_path)
    if not state_path.is_file():
        return f"文件不存在: {state_path}"
    obs_path = petrology_obs_path_from_state(state_path)
    tran_path = petrology_transect_path_from_state(state_path)
    parts = [f"gui_state: {state_path.name}"]
    parts.append(f"观测 JSON: {obs_path.name}" if obs_path else "观测 JSON: (无)")
    parts.append(f"沿迹 windows: {tran_path.name}" if tran_path else "沿迹 windows: (无)")
    return "  |  ".join(parts)
