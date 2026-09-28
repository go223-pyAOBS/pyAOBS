"""预览文案与运行准备（无 UI；Qt 与脚本共用）。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ...tx2tomo2d import (
    convert_tx_in_to_tomo2d,
    parse_obs_id_spec,
    parse_phase_set,
    parse_tx_in_list,
)
from ..state.form_state import FormState
from .audits import (
    edit_smesh_input_path_audit,
    gen_smesh_preview_path_audit,
    mesh_family_input_path_audit,
    pipeline_step_path_audit,
    stat_smesh_input_path_audit,
    tt_forward_input_path_audit,
    tt_inverse_input_path_audit,
    tx_convert_input_path_audit,
    validate_pipeline_input_files,
)
from .collectors import (
    collect_edit_smesh_args,
    collect_gen_damp_kwargs,
    collect_gen_smesh_kwargs,
    collect_gen_vcorr_kwargs,
    collect_gen_dcorr_kwargs,
    collect_stat_smesh_kwargs,
    collect_tt_forward_args,
    collect_tt_inverse_args,
    format_tt_inverse_flag_summary,
)
from .paths import to_workdir_relative
from .pipeline import (
    build_pipeline_plan,
    pipeline_links,
    resolve_pipeline_step_cmdline,
)
from .preview import format_resolved_cmdline, preview_append_resolved_cmdline
from .run_env import format_run_env_preview, collect_run_env, format_strategy_flags


def _append_run_env(text: str, state: FormState) -> str:
    block = format_run_env_preview(collect_run_env(state))
    if not block:
        return text
    return text + "\n\n" + block


def _tt_inverse_flag_line(state: FormState, kwargs: dict) -> str:
    flag = format_tt_inverse_flag_summary(kwargs)
    strat = format_strategy_flags(collect_run_env(state))
    if strat:
        return f"{flag} | {strat}"
    return flag


def _append_run_env(text: str, state: FormState) -> str:
    block = format_run_env_preview(collect_run_env(state))
    if not block:
        return text
    return text + "\n\n" + block

GEN_SMESH_OUT_REQUIRED = (
    "请先填写「smesh 输出文件」（相对 work_dir）。\n\n"
    "未指定路径时程序不会把网格写入磁盘，本界面不允许在此情况下运行。\n"
    "仅预览参数可不填；独立脚本仍可使用 tomo.gen_smesh(..., out_file=...)。"
)


@dataclass
class PreparedRun:
    title: str
    job: Callable[[], Any]
    preview_text: str = ""
    proc_cwd: Path | None = None
    finally_fn: Callable[[Any, Any], None] | None = None
    notes: list[str] = field(default_factory=list)
    #: 反演准实时监视目标（见 ``inv_monitor.InvMonitorSpec``）
    monitor_spec: Any = None


def _call_preview(name: str, kwargs: dict) -> str:
    body = ",\n".join(f"  {k}={kwargs[k]!r}" for k in sorted(kwargs.keys()))
    return f"tomo.{name}(\n{body}\n)"


def resolve_tx_convert_paths(
    state: FormState, work: Path
) -> tuple[Path, list[Path], Path, Path]:
    """返回 station、tx.in 列表、data_out、geom_out 绝对路径。"""

    def to_abs(pstr: str) -> Path:
        p = Path(pstr).expanduser()
        if p.is_absolute():
            return p.resolve()
        rel = to_workdir_relative(pstr, work, warn_outside=False).value
        return (work / rel).resolve()

    ss = state.get_str("tx.station_lis") or "station.lis"
    raw_tx = state.get_str("tx.tx_in") or "tx.in"
    tx_list = parse_tx_in_list(raw_tx) or ["tx.in"]
    do = state.get_str("tx.data_out") or "ttimes.dat"
    go = state.get_str("tx.geom_out") or "geom.dat"
    return to_abs(ss), [to_abs(p) for p in tx_list], to_abs(do), to_abs(go)


def preview_gen_smesh(state: FormState, work: Path, tomo) -> str:
    kwargs = collect_gen_smesh_kwargs(state)
    text = _call_preview("gen_smesh", kwargs)
    if not str(kwargs.get("out_file") or "").strip():
        text += (
            "\n\n# 未传 out_file：GUI 下点击「运行 gen_smesh」将被禁止；"
            "请填写「smesh 输出文件」或仅在脚本中使用 out_file=。"
        )
    vin = str(kwargs.get("v_in") or "")
    if kwargs.get("vel_opt") == "zelt" and vin and ("/" in vin.replace("\\", "/")):
        text += (
            "\n\n# 注：v_in 表单可含相对子目录；"
            "命令行 -C 只用文件名，运行前会复制到 work_dir 根目录。"
        )
    if kwargs.get("refl_file"):
        text += (
            "\n\n# 注：refl_file (-F) 为**输出**（不必预先存在）；"
            "命令行用文件名，若路径含目录则写完后挪到目标位置。"
        )
    text += "\n\n" + gen_smesh_preview_path_audit(work, kwargs)
    return preview_append_resolved_cmdline(
        text, lambda: tomo.resolve_cmdline_gen_smesh(**kwargs)
    )


def prepare_gen_smesh(state: FormState, work: Path, tomo) -> PreparedRun:
    kwargs = collect_gen_smesh_kwargs(state)
    if not str(kwargs.get("out_file") or "").strip():
        raise ValueError(GEN_SMESH_OUT_REQUIRED)
    base = "运行: tomo.gen_smesh(...)\n" + json.dumps(kwargs, ensure_ascii=False, indent=2)
    return PreparedRun(
        title="gen_smesh",
        job=lambda: tomo.gen_smesh(**kwargs),
        preview_text=preview_append_resolved_cmdline(
            base, lambda: tomo.resolve_cmdline_gen_smesh(**kwargs)
        ),
    )


def preview_tt_forward(state: FormState, work: Path, tomo) -> str:
    smesh, geom, kwargs = collect_tt_forward_args(state)
    if not smesh:
        raise ValueError("tt_forward 需要 smesh")
    pieces = [f"  smesh={smesh!r}"]
    if geom is not None:
        pieces.append(f"  geom={geom!r}")
    pieces.extend(f"  {k}={kwargs[k]!r}" for k in sorted(kwargs.keys()))
    text = (
        "tomo.tt_forward(\n"
        + ",\n".join(pieces)
        + "\n)\n\n"
        + tt_forward_input_path_audit(work, smesh, geom, kwargs)
    )
    text = _append_run_env(text, state)
    return preview_append_resolved_cmdline(
        text, lambda: tomo.resolve_cmdline_tt_forward(smesh=smesh, geom=geom, **kwargs)
    )


def prepare_tt_forward(state: FormState, work: Path, tomo) -> PreparedRun:
    smesh, geom, kwargs = collect_tt_forward_args(state)
    if not smesh:
        raise ValueError("tt_forward 需要 smesh")
    payload = {"smesh": smesh, "geom": geom, **kwargs}
    base = "运行: tomo.tt_forward(...)\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    notes: list[str] = []
    auto_bridge = state.get_bool("gui.auto_fwd_to_inv", False)
    if auto_bridge:
        notes.append(
            "已勾选「合成流程：完成后写入反演空位」→ 将尝试填 inv.mesh / inv.data"
        )

    def _after(res, err):
        if err is not None:
            return
        from .workflow_bridge import apply_fwd_outputs_to_inv, suggest_next_after_forward

        apply_fwd_outputs_to_inv(state, overwrite=False, sync_ray=True, sync_refl=True)
        state.set("gui._fwd_bridge_hint", suggest_next_after_forward(state))

    return PreparedRun(
        title="tt_forward",
        job=lambda: tomo.tt_forward(smesh=smesh, geom=geom, **kwargs),
        preview_text=preview_append_resolved_cmdline(
            base,
            lambda: tomo.resolve_cmdline_tt_forward(smesh=smesh, geom=geom, **kwargs),
        ),
        finally_fn=_after if auto_bridge else None,
        notes=notes,
    )


def preview_gen_damp(state: FormState, work: Path, tomo) -> str:
    kwargs = collect_gen_damp_kwargs(state)
    text = (
        _call_preview("gen_damp", kwargs)
        + "\n\n"
        + mesh_family_input_path_audit(work, kwargs, "gen_damp")
    )
    return preview_append_resolved_cmdline(
        text, lambda: tomo.resolve_cmdline_gen_damp(**kwargs)
    )


def prepare_gen_damp(state: FormState, work: Path, tomo) -> PreparedRun:
    kwargs = collect_gen_damp_kwargs(state)
    base = "运行: tomo.gen_damp(...)\n" + json.dumps(kwargs, ensure_ascii=False, indent=2)
    return PreparedRun(
        title="gen_damp",
        job=lambda: tomo.gen_damp(**kwargs),
        preview_text=preview_append_resolved_cmdline(
            base, lambda: tomo.resolve_cmdline_gen_damp(**kwargs)
        ),
    )


def _simple_vcorr_file_preview(kwargs: dict) -> str:
    from ...simple_vcorr import format_simple_vcorr_from_kwargs

    try:
        body = format_simple_vcorr_from_kwargs(kwargs)
    except ValueError as e:
        return f"# {e}"
    return (
        "# 本模式不调用 gen_vcorr 二进制，直接写出 CorrelationLength2d（2×2 顶/底）:\n"
        + body
    )


def preview_gen_vcorr(state: FormState, work: Path, tomo) -> str:
    kwargs = collect_gen_vcorr_kwargs(state)
    text = (
        _call_preview("gen_vcorr", kwargs)
        + "\n\n"
        + mesh_family_input_path_audit(work, kwargs, "gen_vcorr")
    )
    if kwargs.get("out_file"):
        from .audits import audit_output_target_note

        text += "\n" + audit_output_target_note(
            work, "vcorr 输出文件", str(kwargs["out_file"])
        )
    if str(kwargs.get("mode") or "") == "simple_2x2":
        text += "\n\n" + _simple_vcorr_file_preview(kwargs)
        return text
    return preview_append_resolved_cmdline(
        text, lambda: tomo.resolve_cmdline_gen_vcorr(**kwargs)
    )


def prepare_gen_vcorr(state: FormState, work: Path, tomo) -> PreparedRun:
    kwargs = collect_gen_vcorr_kwargs(state)
    if str(kwargs.get("mode") or "") == "simple_2x2":
        if not str(kwargs.get("out_file") or "").strip():
            raise ValueError("gen_vcorr（simple_2x2）需要填写「vcorr 输出文件」")
        from ...simple_vcorr import format_simple_vcorr_from_kwargs

        format_simple_vcorr_from_kwargs(kwargs)
    base = "运行: tomo.gen_vcorr(...)\n" + json.dumps(kwargs, ensure_ascii=False, indent=2)
    if str(kwargs.get("mode") or "") == "simple_2x2":
        preview = base + "\n\n" + _simple_vcorr_file_preview(kwargs)
        if kwargs.get("out_file"):
            from .audits import audit_output_target_note

            preview += "\n" + audit_output_target_note(
                work, "vcorr 输出文件", str(kwargs["out_file"])
            )
        return PreparedRun(
            title="gen_vcorr",
            job=lambda: tomo.gen_vcorr(**kwargs),
            preview_text=preview,
        )
    return PreparedRun(
        title="gen_vcorr",
        job=lambda: tomo.gen_vcorr(**kwargs),
        preview_text=preview_append_resolved_cmdline(
            base, lambda: tomo.resolve_cmdline_gen_vcorr(**kwargs)
        ),
    )


def preview_gen_dcorr(state: FormState, work: Path, tomo) -> str:
    kwargs = collect_gen_dcorr_kwargs(state)
    text = (
        _call_preview("gen_dcorr", kwargs)
        + "\n\n"
        + mesh_family_input_path_audit(work, kwargs, "gen_dcorr")
    )
    if kwargs.get("vcorr_file"):
        from .audits import audit_path_under_work_dir

        text += "\n" + audit_path_under_work_dir(
            work, "vcorr_file (-V)", str(kwargs["vcorr_file"]), optional=False
        )
    if kwargs.get("out_file"):
        from .audits import audit_output_target_note

        text += "\n" + audit_output_target_note(
            work, "dcorr 输出文件", str(kwargs["out_file"])
        )
    return preview_append_resolved_cmdline(
        text, lambda: tomo.resolve_cmdline_gen_dcorr(**kwargs)
    )


def prepare_gen_dcorr(state: FormState, work: Path, tomo) -> PreparedRun:
    kwargs = collect_gen_dcorr_kwargs(state)
    if not str(kwargs.get("out_file") or "").strip():
        raise ValueError("gen_dcorr 需要填写「dcorr 输出文件」，否则 -CD 文件无法落盘")
    base = "运行: tomo.gen_dcorr(...)\n" + json.dumps(kwargs, ensure_ascii=False, indent=2)
    return PreparedRun(
        title="gen_dcorr",
        job=lambda: tomo.gen_dcorr(**kwargs),
        preview_text=preview_append_resolved_cmdline(
            base, lambda: tomo.resolve_cmdline_gen_dcorr(**kwargs)
        ),
    )


def preview_tt_inverse(state: FormState, work: Path, tomo) -> str:
    mesh, data, kwargs = collect_tt_inverse_args(state)
    if not mesh or not data:
        raise ValueError("tt_inverse 需要 mesh 与 data")
    pieces = [f"  mesh={mesh!r}", f"  data={data!r}"]
    pieces.extend(f"  {k}={kwargs[k]!r}" for k in sorted(kwargs.keys()))
    text = "tomo.tt_inverse(\n" + ",\n".join(pieces) + "\n)"
    text += "\n# 开关摘要: " + _tt_inverse_flag_line(state, kwargs)
    from .refl_stride import peek_refl_stride

    src_refl = state.get_str("inv.refl_file")
    stride = peek_refl_stride(kwargs)
    if src_refl and stride > 1:
        text += (
            f"\n# refl 运行时按步长 {stride} 抽稀（表单仍指向源文件，不另存 *_sN.refl）"
        )
    ub = state.get_bool("inv.use_repro_bundle", True)
    mesh_r, data_r, kw_r = mesh, data, kwargs
    if ub:
        from ...tt_inverse_bundle import bundle_argv_preview_paths

        mesh_r, data_r, kw_r = bundle_argv_preview_paths(mesh, data, kwargs)
        text += (
            "\n\n# 已勾选「可复现运行包」：运行时会新建 work_dir/runs/…/ 作为 run_dir，"
            "输入快照到 inputs/、-L/-O/-K 等到 outputs/，并写 manifest。"
            "下方「解析后命令行」按 proc_cwd=run_dir 拼出（与真实子进程一致；预览不创建目录）。"
            "上表 tomo.tt_inverse(...) 仍为表单原始参数。"
        )
    else:
        text += (
            "\n\n# 未勾选运行包：子进程 cwd = work_dir；"
            "输出项留空则 TomoAnd 不传 -L/-O/-K（与直接运行一致）。"
        )
    text += "\n\n" + tt_inverse_input_path_audit(work, mesh, data, kwargs)
    text = _append_run_env(text, state)
    return preview_append_resolved_cmdline(
        text,
        lambda: tomo.resolve_cmdline_tt_inverse(
            mesh=mesh_r, data=data_r, **dict(kw_r)
        ),
    )


def prepare_tt_inverse(
    state: FormState,
    work: Path,
    tomo,
    *,
    gui_profile: dict | None = None,
) -> PreparedRun:
    mesh, data, kwargs = collect_tt_inverse_args(state)
    if not mesh or not data:
        raise ValueError("tt_inverse 需要 mesh 与 data")
    payload = {"mesh": mesh, "data": data, **kwargs}
    base = "运行: tomo.tt_inverse(...)\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    flag_sum = _tt_inverse_flag_line(state, kwargs)
    base += f"\n\n# 开关摘要: {flag_sum}\n"
    use_bundle = state.get_bool("inv.use_repro_bundle", True)
    mesh_r, data_r, kw_r = mesh, data, kwargs
    proc_cwd: Path | None = None
    finally_fn = None
    notes: list[str] = [f"[tt_inverse] {flag_sum}"]
    monitor_spec = None

    if use_bundle:
        from ...tt_inverse_bundle import (
            build_tt_inverse_bundle,
            finalize_tt_inverse_manifest,
            write_tt_inverse_manifest,
        )
        from .inv_monitor import build_monitor_spec_from_paths

        label = state.get_str("inv.bundle_run_label") or None
        br = build_tt_inverse_bundle(work, mesh, data, kwargs, run_label=label)
        mesh_r, data_r, kw_r = br.mesh, br.data, br.kwargs
        argv = tomo.resolve_cmdline_tt_inverse(mesh=mesh_r, data=data_r, **kw_r)
        if not argv:
            raise ValueError("无法解析 tt_inverse 命令行（运行包模式）")
        write_tt_inverse_manifest(
            manifest_path=br.manifest_path,
            work_dir=work,
            run_dir=br.run_dir,
            executable_resolved=argv[0],
            argv=argv,
            mesh=mesh_r,
            data=data_r,
            kwargs=kw_r,
            inputs_rows=br.inputs_manifest,
            status="running",
            gui_profile=gui_profile,
        )
        proc_cwd = br.run_dir
        tomo.proc_cwd = str(br.run_dir)
        manifest_path = br.manifest_path
        notes.append(f"[tt_inverse] 运行包目录: {br.run_dir}")
        base += f"\n\n# 可复现运行包: proc_cwd = {proc_cwd}\n"

        def finally_fn(res, err):
            finalize_tt_inverse_manifest(manifest_path, result=res, err=err)

        st_path = state.get_str("env.inv_status_jsonl_path") or "outputs/status.jsonl"
        monitor_spec = build_monitor_spec_from_paths(
            cwd=br.run_dir,
            log_file=str(kw_r.get("log_file") or "outputs/tt_inverse.log"),
            out_root=str(kw_r.get("out_root") or "outputs/out"),
            niter=kw_r.get("niter"),
            run_dir=br.run_dir,
            status_jsonl=st_path,
        )
    else:
        from .inv_monitor import build_monitor_spec_from_paths
        from .refl_stride import materialize_refl_for_cwd

        materialize_refl_for_cwd(kwargs, work, work)

        log_f = kwargs.get("log_file")
        out_r = kwargs.get("out_root")
        if log_f or out_r:
            st_path = state.get_str("env.inv_status_jsonl_path") or "outputs/status.jsonl"
            monitor_spec = build_monitor_spec_from_paths(
                cwd=work,
                log_file=str(log_f) if log_f else None,
                out_root=str(out_r) if out_r else None,
                niter=kwargs.get("niter"),
                run_dir=None,
                status_jsonl=st_path,
            )

    kw_copy = dict(kw_r)
    preview = preview_append_resolved_cmdline(
        base,
        lambda: tomo.resolve_cmdline_tt_inverse(mesh=mesh_r, data=data_r, **kw_r),
    )
    return PreparedRun(
        title="tt_inverse",
        job=lambda: tomo.tt_inverse(mesh=mesh_r, data=data_r, **kw_copy),
        preview_text=preview,
        proc_cwd=proc_cwd,
        finally_fn=finally_fn,
        notes=notes,
        monitor_spec=monitor_spec,
    )


def preview_stat_smesh(state: FormState, work: Path, tomo) -> str:
    kwargs = collect_stat_smesh_kwargs(state)
    text = (
        _call_preview("stat_smesh", kwargs)
        + "\n\n"
        + stat_smesh_input_path_audit(work, kwargs)
    )
    return preview_append_resolved_cmdline(
        text, lambda: tomo.resolve_cmdline_stat_smesh(**kwargs)
    )


def prepare_stat_smesh(state: FormState, work: Path, tomo) -> PreparedRun:
    kwargs = collect_stat_smesh_kwargs(state)
    base = "运行: tomo.stat_smesh(...)\n" + json.dumps(kwargs, ensure_ascii=False, indent=2)
    return PreparedRun(
        title="stat_smesh",
        job=lambda: tomo.stat_smesh(**kwargs),
        preview_text=preview_append_resolved_cmdline(
            base, lambda: tomo.resolve_cmdline_stat_smesh(**kwargs)
        ),
    )


def preview_edit_smesh(state: FormState, work: Path, tomo) -> str:
    smesh_file, cmd_type, kwargs = collect_edit_smesh_args(state)
    if not smesh_file:
        raise ValueError("edit_smesh_HHB 需要 smesh_file")
    pieces = [f"  smesh_file={smesh_file!r}", f"  cmd_type={cmd_type!r}"]
    pieces.extend(f"  {k}={kwargs[k]!r}" for k in sorted(kwargs.keys()))
    text = (
        "tomo.edit_smesh(\n"
        + ",\n".join(pieces)
        + "\n)\n\n"
        + edit_smesh_input_path_audit(work, smesh_file, kwargs)
    )
    return preview_append_resolved_cmdline(
        text,
        lambda: tomo.resolve_cmdline_edit_smesh(
            smesh_file=smesh_file, cmd_type=cmd_type, **kwargs
        ),
    )


def prepare_edit_smesh(state: FormState, work: Path, tomo) -> PreparedRun:
    smesh_file, cmd_type, kwargs = collect_edit_smesh_args(state)
    if not smesh_file:
        raise ValueError("edit_smesh_HHB 需要 smesh_file")
    payload = {"smesh_file": smesh_file, "cmd_type": cmd_type, **kwargs}
    base = "运行: tomo.edit_smesh(...)\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    return PreparedRun(
        title="edit_smesh",
        job=lambda: tomo.edit_smesh(
            smesh_file=smesh_file, cmd_type=cmd_type, **kwargs
        ),
        preview_text=preview_append_resolved_cmdline(
            base,
            lambda: tomo.resolve_cmdline_edit_smesh(
                smesh_file=smesh_file, cmd_type=cmd_type, **kwargs
            ),
        ),
    )


def preview_pipeline(state: FormState, work: Path, tomo) -> str:
    """构建 pipeline 预览；可能通过 auto_wire 修改 state。"""
    plan = build_pipeline_plan(state)
    validate_pipeline_input_files(
        plan, pipeline_links(state), state.get_str("work_dir")
    )
    payload = {
        "recipe": state.get_str("pipe.recipe"),
        "auto_wire": state.get_bool("pipe.auto_wire"),
        "links": pipeline_links(state),
        "steps": [{"name": n, **v} for n, v in plan],
    }
    text = "pipeline plan:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    blocks = [text, "\n# ========== 输入路径检查 =========="]
    ub = state.get_bool("inv.use_repro_bundle", True)
    for step_name, spec in plan:
        blocks.append(f"\n### {step_name}\n")
        blocks.append(pipeline_step_path_audit(work, step_name, spec))
        try:
            cmd = resolve_pipeline_step_cmdline(
                tomo, step_name, spec, use_repro_bundle=ub
            )
            if cmd:
                blocks.append(f"\n## {step_name}\n{format_resolved_cmdline(cmd)}")
            elif step_name == "gen_vcorr" and str(
                (spec.get("kwargs") or {}).get("mode") or ""
            ) == "simple_2x2":
                blocks.append(
                    f"\n## {step_name}\n"
                    + _simple_vcorr_file_preview(spec.get("kwargs") or {})
                )
            else:
                blocks.append(f"\n## {step_name}\n（必选参数不齐，未拼 argv）")
        except Exception as e:
            blocks.append(f"\n## {step_name}\n（无法解析: {e}）")
    return _append_run_env("\n".join(blocks), state)


def execute_pipeline_plan(tomo, plan: list[tuple[str, dict]]) -> str:
    """顺序执行 pipeline 各步，返回合并日志（不含运行包增强）。"""
    logs: list[str] = []
    for step_name, spec in plan:
        logs.append(f"== {step_name} ==")
        if step_name == "gen_smesh":
            logs.append(str(tomo.gen_smesh(**spec["kwargs"]) or ""))
        elif step_name == "gen_damp":
            logs.append(str(tomo.gen_damp(**spec["kwargs"]) or ""))
        elif step_name == "gen_vcorr":
            logs.append(str(tomo.gen_vcorr(**spec["kwargs"]) or ""))
        elif step_name == "gen_dcorr":
            logs.append(str(tomo.gen_dcorr(**spec["kwargs"]) or ""))
        elif step_name == "tt_forward":
            logs.append(
                str(
                    tomo.tt_forward(
                        smesh=spec["smesh"],
                        geom=spec.get("geom"),
                        **spec["kwargs"],
                    )
                    or ""
                )
            )
        elif step_name == "tt_inverse":
            logs.append(
                str(
                    tomo.tt_inverse(
                        mesh=spec["mesh"],
                        data=spec["data"],
                        **spec["kwargs"],
                    )
                    or ""
                )
            )
        else:
            raise ValueError(f"未知步骤 {step_name}")
    return "\n".join(logs)


def prepare_pipeline_simple(state: FormState, work: Path, tomo) -> PreparedRun:
    """简单 pipeline（无 tt_inverse 运行包增强）；用于 Qt。"""
    preview = preview_pipeline(state, work, tomo)
    plan = build_pipeline_plan(state)
    recipe = state.get_str("pipe.recipe") or ""
    return PreparedRun(
        title=f"pipeline[{recipe}]" if recipe else "pipeline",
        job=lambda: execute_pipeline_plan(tomo, plan),
        preview_text=preview,
    )


def preview_tx_convert(state: FormState, work: Path) -> str:
    station, txins, dout, gout = resolve_tx_convert_paths(state, work)
    refr = parse_phase_set(state.get_str("tx.refr_phases") or "1")
    refl = parse_phase_set(state.get_str("tx.refl_phases") or "")
    water = parse_phase_set(state.get_str("tx.water_phases") or "")
    mult = parse_phase_set(state.get_str("tx.mult_phases") or "")
    refr_mult = parse_phase_set(state.get_str("tx.refr_mult_phases") or "")
    refl_mult = parse_phase_set(state.get_str("tx.refl_mult_phases") or "")
    psp = parse_phase_set(state.get_str("tx.psp_phases") or "")
    include_obs = parse_obs_id_spec(state.get_str("tx.obs_ids"))
    if len(txins) == 1:
        tx_arg = f"    Path({str(txins[0])!r}),\n"
    else:
        body = ",\n".join(f"        Path({str(p)!r})" for p in txins)
        tx_arg = f"    [\n{body},\n    ],\n"
    obs_kw = ""
    if include_obs is not None:
        obs_kw = f"    include_obs={set(include_obs)!r},\n"
    water_kw = f"    water_phases={set(water)!r},\n" if water else ""
    mult_kw = f"    mult_phases={set(mult)!r},\n" if mult else ""
    refr_mult_kw = (
        f"    refr_mult_phases={set(refr_mult)!r},\n" if refr_mult else ""
    )
    refl_mult_kw = (
        f"    refl_mult_phases={set(refl_mult)!r},\n" if refl_mult else ""
    )
    psp_kw = f"    psp_phases={set(psp)!r},\n" if psp else ""
    if include_obs is None:
        obs_note = "全部"
    elif not include_obs:
        obs_note = "（未选择）"
    else:
        obs_note = ", ".join(str(i) for i in sorted(include_obs))
    extra_phase = ""
    if water:
        extra_phase += f"# 直达水波震相→2: {sorted(water)}\n"
    if mult:
        extra_phase += f"# 水柱多次震相→3: {sorted(mult)}\n"
    if refr_mult:
        extra_phase += f"# 折射台侧多次震相→4: {sorted(refr_mult)}\n"
    if refl_mult:
        extra_phase += f"# 反射台侧多次震相→5: {sorted(refl_mult)}\n"
    if psp:
        extra_phase += f"# 折合 PSP 震相→6: {sorted(psp)}\n"
    text = (
        "from pathlib import Path\n"
        "from pyAOBS.modeling.tomo2d.tx2tomo2d import convert_tx_in_to_tomo2d\n\n"
        "convert_tx_in_to_tomo2d(\n"
        f"    Path({str(station)!r}),\n"
        f"{tx_arg}"
        f"    Path({str(dout)!r}),\n"
        f"    Path({str(gout)!r}),\n"
        f"    refr_phases={set(refr)!r},\n"
        f"    refl_phases={set(refl)!r},\n"
        f"{water_kw}"
        f"{mult_kw}"
        f"{refr_mult_kw}"
        f"{refl_mult_kw}"
        f"{psp_kw}"
        f"{obs_kw}"
        ")\n\n"
        f"# work_dir: {work}\n"
        f"# tx.in 文件数: {len(txins)}\n"
        f"# 折射震相: {sorted(refr) if refr else '（空）'}\n"
        f"# 反射震相: {sorted(refl) if refl else '（空）'}\n"
        f"{extra_phase}"
        f"# 选用 OBS: {obs_note}\n"
        "\n"
        + tx_convert_input_path_audit(work, station, txins, dout, gout)
    )
    return text


def prepare_tx_convert(state: FormState, work: Path) -> PreparedRun:
    station, txins, dout, gout = resolve_tx_convert_paths(state, work)
    if not station.is_file():
        raise ValueError(f"台站文件不存在: {station}")
    if not txins:
        raise ValueError("至少需要一个 tx.in")
    missing = [p for p in txins if not p.is_file()]
    if missing:
        raise ValueError(
            "tx.in 不存在:\n" + "\n".join(str(p) for p in missing)
        )
    refr = parse_phase_set(state.get_str("tx.refr_phases") or "1")
    refl = parse_phase_set(state.get_str("tx.refl_phases") or "")
    water = parse_phase_set(state.get_str("tx.water_phases") or "")
    mult = parse_phase_set(state.get_str("tx.mult_phases") or "")
    refr_mult = parse_phase_set(state.get_str("tx.refr_mult_phases") or "")
    refl_mult = parse_phase_set(state.get_str("tx.refl_mult_phases") or "")
    psp = parse_phase_set(state.get_str("tx.psp_phases") or "")
    include_obs = parse_obs_id_spec(state.get_str("tx.obs_ids"))
    if include_obs is not None and not include_obs:
        raise ValueError(
            "未选择任何 OBS。请在转换页或「预览 tx.in」列表中勾选要转换的台站，或点「全选」。"
        )
    payload = {
        "station": str(station),
        "tx_in": [str(p) for p in txins],
        "data_out": str(dout),
        "geom_out": str(gout),
        "refr_phases": sorted(refr),
        "refl_phases": sorted(refl),
        "include_obs": None if include_obs is None else sorted(include_obs),
    }
    if water:
        payload["water_phases"] = sorted(water)
    if mult:
        payload["mult_phases"] = sorted(mult)
    if refr_mult:
        payload["refr_mult_phases"] = sorted(refr_mult)
    if refl_mult:
        payload["refl_mult_phases"] = sorted(refl_mult)
    if psp:
        payload["psp_phases"] = sorted(psp)
    return PreparedRun(
        title="tx.in→tomo2d",
        job=lambda: convert_tx_in_to_tomo2d(
            station,
            txins,
            dout,
            gout,
            refr_phases=refr,
            refl_phases=refl,
            water_phases=water or None,
            mult_phases=mult or None,
            refr_mult_phases=refr_mult or None,
            refl_mult_phases=refl_mult or None,
            psp_phases=psp or None,
            include_obs=include_obs,
        ),
        preview_text="运行: tx.in→tomo2d\n"
        + json.dumps(payload, ensure_ascii=False, indent=2),
    )
