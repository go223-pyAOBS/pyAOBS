"""正反演流程衔接：正演写回反演、上游填充、射线参数同步、监视就绪检查（无 UI）。"""

from __future__ import annotations

from pathlib import Path

from ..state.form_state import FormState

_RAY_KEYS = (
    "xorder",
    "zorder",
    "clen",
    "nintp",
    "bend_cg_tol",
    "bend_br_tol",
)


def _set_if(
    state: FormState,
    key: str,
    value: str,
    *,
    overwrite: bool,
    notes: list[str],
    label: str,
) -> None:
    val = (value or "").strip()
    if not val:
        return
    cur = state.get_str(key)
    if cur and not overwrite:
        return
    if cur == val:
        return
    state.set(key, val)
    notes.append(f"{label}: {key} ← {val}")


def apply_fwd_outputs_to_inv(
    state: FormState,
    *,
    overwrite: bool = False,
    sync_ray: bool = True,
    sync_refl: bool = True,
) -> list[str]:
    """
    将 tt_forward 产出接到 tt_inverse：
    ``fwd.smesh→inv.mesh``、``fwd.out_ttime→inv.data``；可选同步 -N / -F / -B→-Y / -X→-B / -U / -k。
    """
    notes: list[str] = []
    _set_if(
        state,
        "inv.mesh",
        state.get_str("fwd.smesh"),
        overwrite=overwrite,
        notes=notes,
        label="正演→反演",
    )
    _set_if(
        state,
        "inv.data",
        state.get_str("fwd.out_ttime"),
        overwrite=overwrite,
        notes=notes,
        label="正演→反演",
    )
    if sync_ray:
        # 射线参数以正演页为准（用户点「同步」即期望覆盖）
        notes.extend(sync_ray_params(state, direction="fwd_to_inv", overwrite=True))
    if sync_refl:
        _set_if(
            state,
            "inv.refl_file",
            state.get_str("fwd.refl_file"),
            overwrite=overwrite,
            notes=notes,
            label="正演→反演",
        )
        _set_if(
            state,
            "inv.seafloor_file",
            state.get_str("fwd.seafloor_file"),
            overwrite=overwrite,
            notes=notes,
            label="正演→反演",
        )
        _set_if(
            state,
            "inv.conv_file",
            state.get_str("fwd.conv_file"),
            overwrite=overwrite,
            notes=notes,
            label="正演→反演",
        )
        _set_if(
            state,
            "inv.vsmesh",
            state.get_str("fwd.vsmesh"),
            overwrite=overwrite,
            notes=notes,
            label="正演→反演",
        )
        _set_if(
            state,
            "inv.kappa",
            state.get_str("fwd.kappa"),
            overwrite=overwrite,
            notes=notes,
            label="正演→反演",
        )
        # do_full_refl：仅当 inv 未勾选且 fwd 勾选时写 True（不强制关）
        if state.get_bool("fwd.do_full_refl") and (
            overwrite or not state.get_bool("inv.do_full_refl")
        ):
            if not state.get_bool("inv.do_full_refl"):
                state.set("inv.do_full_refl", True)
                notes.append("正演→反演: inv.do_full_refl ← True")
    return notes


def fill_inv_from_upstream(state: FormState, *, overwrite: bool = False) -> list[str]:
    """用 gen_smesh / pipe 桥接 / 正演输出填充反演常用路径。"""
    notes: list[str] = []
    _set_if(
        state,
        "inv.mesh",
        state.get_str("gen.smesh_out") or state.get_str("pipe.link_smesh") or state.get_str("fwd.smesh"),
        overwrite=overwrite,
        notes=notes,
        label="上游填充",
    )
    _set_if(
        state,
        "inv.data",
        state.get_str("fwd.out_ttime"),
        overwrite=overwrite,
        notes=notes,
        label="上游填充",
    )
    auto_on = bool(
        state.get_str("inv.auto_damp_max_dv") or state.get_str("inv.auto_damp_max_dd")
    )
    damp_src = state.get_str("pipe.link_damp")
    if auto_on and damp_src:
        notes.append("已选自动 -T，跳过填充 -DQ（与固定 -D 互斥）")
    else:
        _set_if(
            state,
            "inv.damp_v_fn",
            damp_src,
            overwrite=overwrite,
            notes=notes,
            label="上游填充",
        )
    _set_if(
        state,
        "inv.smooth_corr_v_fn",
        state.get_str("vcorr.out_file"),
        overwrite=overwrite,
        notes=notes,
        label="上游填充",
    )
    _set_if(
        state,
        "inv.smooth_corr_v_fn",
        state.get_str("pipe.link_vcorr_v"),
        overwrite=overwrite,
        notes=notes,
        label="上游填充",
    )
    _set_if(
        state,
        "inv.smooth_corr_d_fn",
        state.get_str("dcorr.out_file"),
        overwrite=overwrite,
        notes=notes,
        label="上游填充",
    )
    _set_if(
        state,
        "inv.smooth_corr_d_fn",
        state.get_str("pipe.link_dcorr"),
        overwrite=overwrite,
        notes=notes,
        label="上游填充",
    )
    _set_if(
        state,
        "inv.smooth_corr_d_fn",
        state.get_str("pipe.link_vcorr_d"),
        overwrite=overwrite,
        notes=notes,
        label="上游填充",
    )
    _set_if(
        state,
        "inv.refl_file",
        state.get_str("fwd.refl_file"),
        overwrite=overwrite,
        notes=notes,
        label="上游填充",
    )
    _set_if(
        state,
        "inv.seafloor_file",
        state.get_str("fwd.seafloor_file") or state.get_str("gen.seafloor_out"),
        overwrite=overwrite,
        notes=notes,
        label="上游填充",
    )
    _set_if(
        state,
        "inv.conv_file",
        state.get_str("fwd.conv_file"),
        overwrite=overwrite,
        notes=notes,
        label="上游填充",
    )
    _set_if(
        state,
        "inv.vsmesh",
        state.get_str("fwd.vsmesh"),
        overwrite=overwrite,
        notes=notes,
        label="上游填充",
    )
    _set_if(
        state,
        "inv.kappa",
        state.get_str("fwd.kappa"),
        overwrite=overwrite,
        notes=notes,
        label="上游填充",
    )
    _set_if(
        state,
        "fwd.seafloor_file",
        state.get_str("gen.seafloor_out"),
        overwrite=overwrite,
        notes=notes,
        label="上游填充",
    )
    if not notes:
        notes.append(
            "无可填充项（请先填 gen.smesh_out / fwd.out_ttime / pipeline 桥接路径，"
            "或勾选覆盖后重试）"
        )
    return notes


def sync_ray_params(
    state: FormState,
    *,
    direction: str = "fwd_to_inv",
    overwrite: bool = True,
) -> list[str]:
    """同步 fwd/inv 的 -N 射线参数。"""
    notes: list[str] = []
    if direction == "inv_to_fwd":
        src, dst, tag = "inv", "fwd", "反演→正演"
    else:
        src, dst, tag = "fwd", "inv", "正演→反演"
    for k in _RAY_KEYS:
        _set_if(
            state,
            f"{dst}.{k}",
            state.get_str(f"{src}.{k}"),
            overwrite=overwrite,
            notes=notes,
            label=tag,
        )
    return notes


def format_monitor_checklist(state: FormState) -> str:
    """监视/中间模型可读性检查（文本）。"""
    lines: list[str] = ["【反演监视就绪检查】"]
    use_bundle = state.get_bool("inv.use_repro_bundle", True)
    out_level = state.get_str("inv.out_level")
    try:
        ol = int(float(out_level)) if out_level.strip() else 0
    except ValueError:
        ol = -1
    final_only = state.get_bool("inv.print_final_only")
    out_root = state.get_str("inv.out_root")
    log_file = state.get_str("inv.log_file")
    st = state.get_str("env.inv_status_jsonl_path") or "outputs/status.jsonl"

    def ok(flag: bool, good: str, bad: str) -> None:
        lines.append(("✓ " if flag else "✗ ") + (good if flag else bad))

    ok(bool(state.get_str("inv.mesh")), "mesh 已填", "缺少 inv.mesh (-M)")
    ok(bool(state.get_str("inv.data")), "data 已填", "缺少 inv.data (-G)")
    ok(
        use_bundle or bool(out_root),
        "有 -O（运行包默认 outputs/out 或已填 out_root）",
        "未开运行包且 out_root 为空 → 监视难找 smesh",
    )
    ok(
        use_bundle or bool(log_file),
        "有 -L（运行包默认或已填 log_file）",
        "未开运行包且 log_file 为空 → 无 χ² 曲线",
    )
    if final_only:
        lines.append(
            "· print_final_only 开 → 少写 smesh（推荐生产；曲线仍看 -L/jsonl）"
        )
    else:
        lines.append("· print_final_only 关 → 可刷中间 smesh（I/O 更重）")
    if ol >= 2:
        lines.append(f"· out_level={ol} ≥ 2 → 可画残差拟合与抽样射线（I/O 较重）")
    elif ol >= 1:
        lines.append(f"· out_level={ol} ≥ 1 → 可画走时残差；射线需 ≥2")
    else:
        lines.append("· out_level 空或 0 → 不写 .tres/.ray（监视无拟合图）")
    ok(bool(st.strip()), f"status.jsonl: {st}", "未设 status.jsonl（可用默认）")
    if use_bundle:
        lines.append("· 已勾选可复现运行包 → 输出在 runs/…/outputs/")
    return "\n".join(lines)


def suggest_next_after_forward(state: FormState) -> str:
    mesh = state.get_str("inv.mesh")
    data = state.get_str("inv.data")
    return (
        "合成走时已写出并尝试填入反演空位。"
        "请到「6) tt_inverse」核对后反演（反演内部仍会自行正演）。"
        f"  inv.mesh={'已填' if mesh else '空'}，inv.data={'已填' if data else '空'}。"
    )
