"""GUI quick-form → run payload (toolkit independent)."""

from __future__ import annotations

from pathlib import Path

from ...core.project_layout import (
    PLUGIN_NODE_FORM_KEY,
    iter_runs_dirs,
    resolve_node_id,
    workdir_from_form,
)
from ...petrology_launcher import petrology_obs_path_from_state, petrology_transect_path_from_state
from .helpers import quote_arg_if_needed


def _abs_under_project(path_text: str, project_root: Path | None) -> Path:
    pp = Path(path_text).expanduser()
    if pp.is_absolute() or project_root is None:
        return pp
    return (project_root / pp).resolve()


def _workdir_gui_payload(
    *,
    nid: str,
    work: str,
    project_root: Path | None,
    meta_json: str,
    env_key: str,
    extra_inputs: list[str] | None = None,
    extra_args: str = "",
    opened_msg: str,
    empty_msg: str,
) -> tuple[str, str, str, str, list[str]]:
    """Build argv/env for a Qt GUI that opens ``meta/<name>_project.json`` or a workdir."""
    args_parts: list[str] = []
    env_lines: list[str] = []
    inputs: list[str] = list(extra_inputs or [])
    if work:
        wp = _abs_under_project(work, project_root)
        proj_json = wp / meta_json if wp.is_dir() else wp
        if wp.is_file():
            target = wp
            env_val = str(wp)
        else:
            target = proj_json if proj_json.is_file() else wp
            env_val = str(wp)
        args_parts.append(quote_arg_if_needed(str(target)))
        env_lines.append(f"{env_key}={env_val}")
        if extra_args.strip():
            args_parts.append(extra_args.strip())
        args = " ".join(x for x in args_parts if x).strip()
        return nid, args, opened_msg, "\n".join(env_lines), inputs
    if extra_args.strip():
        args_parts.append(extra_args.strip())
    args = " ".join(x for x in args_parts if x).strip()
    return nid, args, empty_msg, "", inputs


def _workspace_node_id(
    plugin_id: str,
    gui_form: dict[str, str],
    node_id: str,
    work: str,
) -> str:
    key = PLUGIN_NODE_FORM_KEY.get(plugin_id, "")
    named = str(gui_form.get(key, "")).strip() if key else ""
    return resolve_node_id(named, node_id, work_dir=work)


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
    project_root: Path | None = None,
    workspaces: dict[str, str] | None = None,
) -> tuple[str, str, str, str, list[str]]:
    """
    Return ``(node_id, args, status_message, env_extra_text, inputs)``.

    ``env_extra_text`` 为可追加到环境变量框的 ``KEY=VAL`` 行（可空）。
    ``inputs`` 为登记到运行追踪的路径列表。
    空的 ``work_dir`` 会回落到 ``workspaces`` 登记路径（工作台工区）。
    """
    none_extra: tuple[str, list[str]] = ("", [])
    if plugin_id == "data.shell":
        plugin_id = "data.gui"
    ws = workspaces or {}

    if plugin_id == "tomo2d.shell":
        return node_id, "", "当前是 TOMO2D 插件，请使用模板表单。", *none_extra

    if plugin_id == "zplotpy.gui":
        work = workdir_from_form(plugin_id, gui_form, ws)
        nid = _workspace_node_id(plugin_id, gui_form, node_id, work)
        return _workdir_gui_payload(
            nid=nid,
            work=work,
            project_root=project_root,
            meta_json="meta/zplotpy_project.json",
            env_key="PYAOBS_ZPLOTPY_PROJECT",
            extra_args=str(gui_form.get("zplot_extra", "")).strip(),
            opened_msg="zplotpy.gui 将打开指定工区；OBS/炮在 GUI 内选择。",
            empty_msg="zplotpy.gui 启动后请在 GUI 内新建/打开工区。",
        )

    if plugin_id == "imodel.gui":
        work = workdir_from_form(plugin_id, gui_form, ws)
        nid = _workspace_node_id(plugin_id, gui_form, node_id, work)
        model = str(gui_form.get("imodel_model", "")).strip()
        aux = str(gui_form.get("imodel_aux", "")).strip()
        extra = str(gui_form.get("imodel_extra", "")).strip()
        args_parts: list[str] = []
        env_lines: list[str] = []
        inputs: list[str] = []
        root = Path(project_root).resolve() if project_root else None

        def _abs(p: str) -> Path:
            pp = Path(p).expanduser()
            if pp.is_absolute() or root is None:
                return pp
            return (root / pp).resolve()

        if work:
            wp = _abs(work)
            proj_json = wp / "meta" / "imodel_project.json"
            target = proj_json if proj_json.is_file() else wp
            args_parts.append(quote_arg_if_needed(str(target)))
            env_lines.append(f"PYAOBS_IMODEL_PROJECT={wp}")
        if extra:
            args_parts.append(extra)
        if model:
            inputs.append(model)
        if aux:
            inputs.append(aux)
        args = " ".join(x for x in args_parts if x).strip()
        if work:
            msg = "imodel.gui 将打开指定工区；测线/模型在 GUI 内选择。"
        else:
            msg = "imodel（Qt）启动后请在 GUI 内新建/打开工区或加载模型。"
        return nid, args, msg, "\n".join(env_lines), inputs

    if plugin_id == "iphase.gui":
        extra_inputs: list[str] = []
        for key in ("iphase_rin", "iphase_txout"):
            raw = str(gui_form.get(key, "")).strip()
            if raw:
                extra_inputs.append(raw)
        tx_blob = str(gui_form.get("iphase_tx_text", "")).strip()
        if tx_blob:
            extra_inputs.extend(ln.strip() for ln in tx_blob.splitlines() if ln.strip())
        work = workdir_from_form(plugin_id, gui_form, ws)
        nid = _workspace_node_id(plugin_id, gui_form, node_id, work)
        return _workdir_gui_payload(
            nid=nid,
            work=work,
            project_root=project_root,
            meta_json="meta/iphase_project.json",
            env_key="PYAOBS_IPHASE_PROJECT",
            extra_inputs=extra_inputs,
            extra_args=str(gui_form.get("iphase_extra", "")).strip(),
            opened_msg="iphase.gui 将打开指定工区；OBS/炮在 GUI 内选择。",
            empty_msg="iphase.gui 启动后请在 GUI 内新建/打开工区。",
        )

    if plugin_id == "data.gui":
        work = workdir_from_form(plugin_id, gui_form, ws)
        nid = _workspace_node_id(plugin_id, gui_form, node_id, work)
        return _workdir_gui_payload(
            nid=nid,
            work=work,
            project_root=project_root,
            meta_json="meta/idata_project.json",
            env_key="PYAOBS_IDATA_PROJECT",
            extra_args=str(gui_form.get("data_extra", "")).strip(),
            opened_msg="data.gui 将打开指定 idata 工区。",
            empty_msg="data.gui 启动后请在 GUI 内新建/打开工区。",
        )

    if plugin_id == "tomo2d.gui":
        work = workdir_from_form(plugin_id, gui_form, ws)
        nid = _workspace_node_id(plugin_id, gui_form, node_id, work)
        return _workdir_gui_payload(
            nid=nid,
            work=work,
            project_root=project_root,
            meta_json="meta/tomo2d_project.json",
            env_key="PYAOBS_TOMO2D_PROJECT",
            extra_args=str(gui_form.get("tomo_extra", "")).strip(),
            opened_msg="tomo2d.gui 将打开指定工区；反演结果在工区 runs/ 内选择。",
            empty_msg="tomo2d.gui 启动后请在 GUI 内新建/打开工区。",
        )

    if plugin_id == "vedit.gui":
        work = workdir_from_form(plugin_id, gui_form, ws)
        nid = _workspace_node_id(plugin_id, gui_form, node_id, work)
        model = str(gui_form.get("vedit_model", "")).strip()
        extra_inputs = [model] if model else []
        if work:
            return _workdir_gui_payload(
                nid=nid,
                work=work,
                project_root=project_root,
                meta_json="meta/vedit_project.json",
                env_key="PYAOBS_VEDIT_PROJECT",
                extra_inputs=extra_inputs,
                extra_args=str(gui_form.get("vedit_extra", "")).strip(),
                opened_msg="vedit.gui 将打开指定工区（argv / PYAOBS_VEDIT_PROJECT）。",
                empty_msg="vedit.gui 启动后请在 GUI 内打开工区或 v.in。",
            )
        if model:
            mp = _abs_under_project(model, project_root)
            extra = str(gui_form.get("vedit_extra", "")).strip()
            args = quote_arg_if_needed(str(mp))
            if extra:
                args = f"{args} {extra}"
            return nid, args, "vedit.gui 将打开指定 v.in / .edit。", "", extra_inputs
        extra = str(gui_form.get("vedit_extra", "")).strip()
        return nid, extra, "vedit.gui 启动后请在 GUI 内打开工区或 v.in。", "", []

    if plugin_id == "petrology.lip.gui":
        nid = resolve_node_id(node_id)
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
        return nid, args, msg, *none_extra

    return node_id, "", "当前插件未提供 GUI 快速表单。", *none_extra


def find_latest_imodel_gui_state(project_root: Path) -> Path | None:
    run_dirs: list[Path] = []
    for runs_dir in iter_runs_dirs(project_root):
        try:
            run_dirs.extend(p for p in runs_dir.iterdir() if p.is_dir())
        except OSError:
            continue
    run_dirs.sort(key=lambda p: p.name, reverse=True)
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
