"""预览用路径审计与 pipeline 输入文件校验（无 UI）。"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from .collectors import format_tt_inverse_flag_summary


def assert_existing_file(path_value: str, field: str, work_dir: str | Path) -> None:
    if not path_value:
        return
    p = Path(path_value)
    if not p.is_absolute():
        p = Path(str(work_dir).strip() or str(Path.cwd())) / p
    if not p.exists():
        raise ValueError(f"pipeline 文件检查失败: {field} 不存在 -> {p}")
    if not p.is_file():
        raise ValueError(f"pipeline 文件检查失败: {field} 不是文件 -> {p}")


def audit_path_under_work_dir(
    work: Path,
    label: str,
    form_value: str | None,
    *,
    optional: bool = False,
) -> str:
    """单行说明：表单路径相对 work_dir 解析后的绝对路径及是否存在（供预览）。"""
    s = (form_value or "").strip()
    if not s:
        if optional:
            return f"{label}: （未填，不传）"
        return f"{label}: （空）"
    p = Path(s).expanduser()
    try:
        rp = p.resolve() if p.is_absolute() else (work / p).resolve()
    except OSError:
        return f"{label}: {s!r} → 解析失败"
    ok = "✓ 存在"
    bad = "✗ 不存在"
    if rp.is_file():
        st = ok
    elif rp.is_dir():
        st = "✗ 为目录（需要文件）"
    elif rp.exists():
        st = "✗ 非普通文件"
    else:
        st = bad
    return f"{label}: {s!r} → {rp}  {st}"


def audit_output_target_note(work: Path, label: str, form_value: str | None) -> str:
    """输出文件路径：解析目标绝对路径并检查父目录是否存在。"""
    s = (form_value or "").strip()
    if not s:
        return f"{label}: （未填）"
    p = Path(s).expanduser()
    try:
        rp = p.resolve() if p.is_absolute() else (work / p).resolve()
    except OSError:
        return f"{label}（输出）: {s!r} → 解析失败"
    par = rp.parent
    try:
        par_ok = par.is_dir()
    except OSError:
        par_ok = False
    st = "✓ 父目录存在" if par_ok else "✗ 父目录不存在"
    return f"{label}（输出）: {s!r} → {rp}  {st}"


def tt_forward_input_path_audit(
    work: Path, smesh: str, geom: str | None, kwargs: dict
) -> str:
    lines = [
        "# 输入路径检查（子进程 cwd = work_dir，即下方目录）",
        f"#   {work}",
        audit_path_under_work_dir(work, "smesh (-M)", smesh, optional=False),
        audit_path_under_work_dir(work, "geom (-G)", geom or "", optional=True),
    ]
    rf = kwargs.get("refl_file")
    if rf:
        lines.append(
            audit_path_under_work_dir(work, "refl_file (-F)", str(rf), optional=False)
        )
    sf = kwargs.get("seafloor_file")
    if sf:
        lines.append(
            audit_path_under_work_dir(work, "seafloor_file (-B)", str(sf), optional=False)
        )
    conv = kwargs.get("conv_file")
    if conv:
        lines.append(
            audit_path_under_work_dir(work, "conv_file (-X)", str(conv), optional=False)
        )
    vs = kwargs.get("vsmesh")
    if vs:
        lines.append(
            audit_path_under_work_dir(work, "vsmesh (-U)", str(vs), optional=False)
        )
    cf = kwargs.get("clock_file")
    if cf:
        lines.append(
            audit_path_under_work_dir(work, "clock_file (-C)", str(cf), optional=False)
        )
    return "\n".join(lines)


def mesh_family_input_path_audit(work: Path, kwargs: dict, title: str) -> str:
    lines = [
        f"# 输入路径检查（{title}，子进程 cwd = work_dir）",
        f"#   {work}",
    ]
    for key, label in (
        ("v_in", "v_in (-C)"),
        ("x_file", "x_file (-X)"),
        ("z_file", "z_file (-Z)"),
        ("topo_file", "topo_file (-T)"),
    ):
        if kwargs.get(key):
            lines.append(
                audit_path_under_work_dir(work, label, str(kwargs[key]), optional=False)
            )
    # gen_smesh -F 为输出，不要求已存在
    if title == "gen_smesh" and kwargs.get("refl_file"):
        lines.append(
            audit_output_target_note(
                work, "refl_file (-F 输出)", str(kwargs["refl_file"])
            )
        )
    elif kwargs.get("refl_file"):
        lines.append(
            audit_path_under_work_dir(
                work, "refl_file (-F)", str(kwargs["refl_file"]), optional=False
            )
        )
    zd = kwargs.get("zelt_dump_file")
    if zd:
        lines.append(
            audit_output_target_note(work, "zelt_dump (-d 输出)", str(zd))
        )
    if title == "gen_smesh" and kwargs.get("seafloor_out"):
        lines.append(
            audit_output_target_note(
                work, "seafloor_out (-G 输出)", str(kwargs["seafloor_out"])
            )
        )
    return "\n".join(lines)


def gen_smesh_preview_path_audit(work: Path, kwargs: dict) -> str:
    body = mesh_family_input_path_audit(work, kwargs, "gen_smesh")
    out = kwargs.get("out_file")
    return body + "\n" + audit_output_target_note(
        work, "smesh 输出文件", str(out) if out else None
    )


def tt_inverse_input_path_audit(work: Path, mesh: str, data: str, kwargs: dict) -> str:
    lines = [
        "# 输入路径检查（tt_inverse，预览为相对 work_dir；可复现运行包下实际 cwd 为 run_dir）",
        f"#   {work}",
        "# 开关摘要: " + format_tt_inverse_flag_summary(kwargs),
        audit_path_under_work_dir(work, "mesh (-M)", mesh, optional=False),
        audit_path_under_work_dir(work, "data (-G)", data, optional=False),
    ]
    if kwargs.get("refl_file"):
        lines.append(
            audit_path_under_work_dir(
                work, "refl_file (-F)", str(kwargs["refl_file"]), optional=False
            )
        )
    if kwargs.get("seafloor_file"):
        lines.append(
            audit_path_under_work_dir(
                work, "seafloor_file (-Y)", str(kwargs["seafloor_file"]), optional=False
            )
        )
    if kwargs.get("conv_file"):
        lines.append(
            audit_path_under_work_dir(
                work, "conv_file (-B)", str(kwargs["conv_file"]), optional=False
            )
        )
    if kwargs.get("vsmesh"):
        lines.append(
            audit_path_under_work_dir(
                work, "vsmesh (-U)", str(kwargs["vsmesh"]), optional=False
            )
        )
    if kwargs.get("filter_bound_file"):
        lines.append(
            audit_path_under_work_dir(
                work, "filter_bound (-s)", str(kwargs["filter_bound_file"]), optional=False
            )
        )
    elif kwargs.get("apply_filter"):
        lines.append("filter (-s): 开，未指定边界文件 → 用 mesh 海底/地形作上边界")
    sm = kwargs.get("smooth_opts") or {}
    if sm.get("corr_v_fn"):
        lines.append(
            audit_path_under_work_dir(
                work, "smooth_corr_v (-CV)", str(sm["corr_v_fn"]), optional=False
            )
        )
    if sm.get("corr_d_fn"):
        lines.append(
            audit_path_under_work_dir(
                work, "smooth_corr_d (-CD)", str(sm["corr_d_fn"]), optional=False
            )
        )
    dm = kwargs.get("damp_opts") or {}
    if dm.get("damp_v_fn"):
        lines.append(
            audit_path_under_work_dir(
                work, "damp_v_fn (-DQ)", str(dm["damp_v_fn"]), optional=False
            )
        )
    g = kwargs.get("gravity_opts") or {}
    if g.get("grav_file"):
        lines.append(
            audit_path_under_work_dir(
                work, "grav_file (-ZG)", str(g["grav_file"]), optional=False
            )
        )
    cont = g.get("continent")
    if cont and isinstance(cont, (list, tuple)) and len(cont) >= 1 and cont[0]:
        lines.append(
            audit_path_under_work_dir(
                work, "grav_ZC cont_up", str(cont[0]), optional=False
            )
        )
    ou = g.get("ocean_upper")
    if ou and isinstance(ou, (list, tuple)) and len(ou) >= 2:
        if ou[0]:
            lines.append(
                audit_path_under_work_dir(work, "grav_ZU up", str(ou[0]), optional=False)
            )
        if ou[1]:
            lines.append(
                audit_path_under_work_dir(work, "grav_ZU lo", str(ou[1]), optional=False)
            )
    ol = g.get("ocean_lower")
    if ol and isinstance(ol, (list, tuple)) and len(ol) >= 1 and ol[0]:
        lines.append(
            audit_path_under_work_dir(work, "grav_ZL up", str(ol[0]), optional=False)
        )
    sed = g.get("sediment")
    if sed and isinstance(sed, (list, tuple)) and len(sed) >= 2:
        if sed[0]:
            lines.append(
                audit_path_under_work_dir(work, "grav_ZS up", str(sed[0]), optional=False)
            )
        if sed[1]:
            lines.append(
                audit_path_under_work_dir(work, "grav_ZS lo", str(sed[1]), optional=False)
            )
    if g.get("grav_dws"):
        lines.append(audit_output_target_note(work, "grav_dws (-ZK)", str(g["grav_dws"])))
    for py_key, lbl in (
        ("log_file", "log (-L)"),
        ("out_root", "out_root (-O)"),
        ("dws_file", "dws (-K)"),
    ):
        if kwargs.get(py_key):
            lines.append(audit_output_target_note(work, lbl, str(kwargs[py_key])))
    return "\n".join(lines)


def stat_smesh_input_path_audit(work: Path, kwargs: dict) -> str:
    lines = [
        "# 输入路径检查（stat_smesh，子进程 cwd = work_dir）",
        f"#   {work}",
    ]
    for key in sorted(kwargs.keys()):
        if not (key.endswith("_file") or key.endswith("_bound")):
            continue
        v = kwargs[key]
        if v is None or (isinstance(v, str) and not str(v).strip()):
            continue
        lines.append(audit_path_under_work_dir(work, f"{key}", str(v), optional=False))
    return "\n".join(lines)


def edit_smesh_input_path_audit(work: Path, smesh_file: str, kwargs: dict) -> str:
    lines = [
        "# 输入路径检查（edit_smesh_HHB，子进程 cwd = work_dir）",
        f"#   {work}",
        audit_path_under_work_dir(work, "smesh_file", smesh_file, optional=False),
    ]
    for key in sorted(kwargs.keys()):
        if key not in (
            "paste_file",
            "prof_file",
            "remove_bg_file",
            "moho_file",
            "base_file",
            "corr_file",
            "upper_bound",
        ):
            continue
        v = kwargs.get(key)
        if v is None or (isinstance(v, str) and not str(v).strip()):
            continue
        lines.append(audit_path_under_work_dir(work, key, str(v), optional=True))
    return "\n".join(lines)


def tx_convert_input_path_audit(
    work: Path,
    station: Path,
    txin: Path | Sequence[Path],
    dout: Path,
    gout: Path,
) -> str:
    lines = [
        "# 输入路径检查（tx.in→tomo2d；台站与 tx.in 须已存在，输出为将写入的路径）",
        f"#   work_dir = {work}",
    ]

    def one_abs(label: str, p: Path, *, need_file: bool) -> str:
        try:
            rp = p.resolve()
        except OSError:
            return f"{label}: {p} → 解析失败"
        if need_file:
            if rp.is_file():
                st = "✓ 存在"
            elif rp.is_dir():
                st = "✗ 为目录"
            elif rp.exists():
                st = "✗ 非普通文件"
            else:
                st = "✗ 不存在"
        else:
            st = ""
        return f"{label}: {str(p)!r} → {rp}  {st}".rstrip()

    lines.append(one_abs("station_lis", station, need_file=True))
    if isinstance(txin, Path):
        tx_list: list[Path] = [txin]
    else:
        tx_list = [Path(p) for p in txin]
    if len(tx_list) == 1:
        lines.append(one_abs("tx.in", tx_list[0], need_file=True))
    else:
        lines.append(f"# tx.in 共 {len(tx_list)} 个文件：")
        for i, p in enumerate(tx_list, 1):
            lines.append(one_abs(f"  [{i}]", p, need_file=True))
    lines.append(audit_output_target_note(work, "ttimes.dat 输出", str(dout)))
    lines.append(audit_output_target_note(work, "geom.dat 输出", str(gout)))
    return "\n".join(lines)


def pipeline_step_path_audit(work: Path, step_name: str, spec: dict) -> str:
    if step_name == "gen_smesh":
        return gen_smesh_preview_path_audit(work, spec.get("kwargs") or {})
    if step_name == "gen_damp":
        return mesh_family_input_path_audit(work, spec.get("kwargs") or {}, "gen_damp")
    if step_name == "gen_vcorr":
        kw = spec.get("kwargs") or {}
        body = mesh_family_input_path_audit(work, kw, "gen_vcorr")
        if kw.get("out_file"):
            body += "\n" + audit_output_target_note(
                work, "vcorr 输出文件", str(kw["out_file"])
            )
        return body
    if step_name == "gen_dcorr":
        kw = spec.get("kwargs") or {}
        body = mesh_family_input_path_audit(work, kw, "gen_dcorr")
        extra = []
        if kw.get("vcorr_file"):
            extra.append(
                audit_path_under_work_dir(
                    work, "vcorr_file (-V)", str(kw["vcorr_file"]), optional=False
                )
            )
        if kw.get("out_file"):
            extra.append(
                audit_output_target_note(work, "dcorr 输出文件", str(kw["out_file"]))
            )
        return body + (("\n" + "\n".join(extra)) if extra else "")
    if step_name == "tt_forward":
        sm = spec.get("smesh") or ""
        gm = spec.get("geom")
        kw = dict(spec.get("kwargs") or {})
        return tt_forward_input_path_audit(work, str(sm), gm, kw)
    if step_name == "tt_inverse":
        return tt_inverse_input_path_audit(
            work,
            str(spec.get("mesh") or ""),
            str(spec.get("data") or ""),
            dict(spec.get("kwargs") or {}),
        )
    return f"# （未知步骤 {step_name}，跳过路径检查）"


def validate_pipeline_input_files(
    plan: list[tuple[str, dict]],
    links: Mapping[str, str],
    work_dir: str | Path,
) -> None:
    for key, value in links.items():
        if value:
            assert_existing_file(value, f"pipe.link_{key}", work_dir)

    for step_name, spec in plan:
        if step_name == "gen_smesh":
            kw = spec.get("kwargs", {}) or {}
            outf = kw.get("out_file")
            if not str(outf or "").strip():
                raise ValueError(
                    "pipeline 中的 gen_smesh 必须在界面填写「smesh 输出文件」（out_file），"
                    "否则网格无法落盘，下游步骤无法使用。"
                )
            for k in ("v_in", "x_file", "z_file", "topo_file"):
                if k in kw and kw[k] is not None:
                    assert_existing_file(str(kw[k]), f"gen_smesh.{k}", work_dir)
            # refl_file (-F) 为输出，不要求已存在
        elif step_name == "gen_damp":
            kw = spec.get("kwargs", {})
            for k in ("v_in", "x_file", "z_file", "topo_file"):
                if k in kw and kw[k] is not None:
                    assert_existing_file(str(kw[k]), f"gen_damp.{k}", work_dir)
        elif step_name == "gen_vcorr":
            kw = spec.get("kwargs", {}) or {}
            if str(kw.get("mode") or "") == "simple_2x2":
                if not str(kw.get("out_file") or "").strip():
                    raise ValueError(
                        "pipeline 中的 gen_vcorr（simple_2x2）必须填写「vcorr 输出文件」"
                    )
            for k in ("v_in", "x_file", "z_file", "topo_file"):
                if k in kw and kw[k] is not None:
                    assert_existing_file(str(kw[k]), f"gen_vcorr.{k}", work_dir)
        elif step_name == "gen_dcorr":
            kw = spec.get("kwargs", {}) or {}
            if not str(kw.get("out_file") or "").strip():
                raise ValueError("pipeline 中的 gen_dcorr 必须填写「dcorr 输出文件」")
            for k in ("v_in", "vcorr_file", "refl_file"):
                if k in kw and kw[k] is not None:
                    assert_existing_file(str(kw[k]), f"gen_dcorr.{k}", work_dir)
        elif step_name == "tt_forward":
            assert_existing_file(spec.get("smesh") or "", "tt_forward.smesh", work_dir)
            geom = spec.get("geom")
            if geom:
                assert_existing_file(str(geom), "tt_forward.geom", work_dir)
            fkw = spec.get("kwargs", {}) or {}
            refl = fkw.get("refl_file")
            if refl:
                assert_existing_file(str(refl), "tt_forward.refl_file", work_dir)
            convf = fkw.get("conv_file")
            if convf:
                assert_existing_file(str(convf), "tt_forward.conv_file", work_dir)
            vs = fkw.get("vsmesh")
            if vs:
                assert_existing_file(str(vs), "tt_forward.vsmesh", work_dir)
            clk = fkw.get("clock_file")
            if clk:
                assert_existing_file(str(clk), "tt_forward.clock_file", work_dir)
        elif step_name == "stat_smesh":
            skw = spec.get("kwargs", {}) or {}
            mode = skw.get("mode")
            if mode == "list":
                lf = skw.get("list_file")
                if lf:
                    assert_existing_file(str(lf), "stat_smesh.list_file", work_dir)
                if skw.get("cmd_type") == "r" and skw.get("ave_file"):
                    assert_existing_file(str(skw["ave_file"]), "stat_smesh.ave_file", work_dir)
            elif mode == "mesh":
                mf = skw.get("mesh_file")
                if mf:
                    assert_existing_file(str(mf), "stat_smesh.mesh_file", work_dir)
                for k in ("top_bound", "bot_bound", "mid_bound"):
                    p = skw.get(k)
                    if p:
                        assert_existing_file(str(p), f"stat_smesh.{k}", work_dir)
                et = skw.get("exclude_top_bound")
                eb = skw.get("exclude_bot_bound")
                if et:
                    assert_existing_file(str(et), "stat_smesh.exclude_top_bound", work_dir)
                if eb:
                    assert_existing_file(str(eb), "stat_smesh.exclude_bot_bound", work_dir)
        elif step_name == "tt_inverse":
            assert_existing_file(spec.get("mesh") or "", "tt_inverse.mesh", work_dir)
            assert_existing_file(spec.get("data") or "", "tt_inverse.data", work_dir)
            kw = spec.get("kwargs", {}) or {}
            if kw.get("refl_file"):
                assert_existing_file(str(kw["refl_file"]), "tt_inverse.refl_file", work_dir)
            if kw.get("seafloor_file"):
                assert_existing_file(str(kw["seafloor_file"]), "tt_inverse.seafloor_file", work_dir)
            if kw.get("conv_file"):
                assert_existing_file(str(kw["conv_file"]), "tt_inverse.conv_file", work_dir)
            if kw.get("vsmesh"):
                assert_existing_file(str(kw["vsmesh"]), "tt_inverse.vsmesh", work_dir)
            smooth_opts = kw.get("smooth_opts", {}) or {}
            if smooth_opts.get("corr_v_fn"):
                assert_existing_file(
                    str(smooth_opts["corr_v_fn"]), "tt_inverse.smooth_opts.corr_v_fn", work_dir
                )
            if smooth_opts.get("corr_d_fn"):
                assert_existing_file(
                    str(smooth_opts["corr_d_fn"]), "tt_inverse.smooth_opts.corr_d_fn", work_dir
                )
            damp_opts = kw.get("damp_opts", {}) or {}
            if damp_opts.get("damp_v_fn"):
                assert_existing_file(
                    str(damp_opts["damp_v_fn"]), "tt_inverse.damp_opts.damp_v_fn", work_dir
                )
            if kw.get("filter_bound_file"):
                assert_existing_file(
                    str(kw["filter_bound_file"]), "tt_inverse.filter_bound_file", work_dir
                )
            go = kw.get("gravity_opts") or {}
            if go.get("grav_file"):
                assert_existing_file(
                    str(go["grav_file"]), "tt_inverse.gravity_opts.ZG", work_dir
                )
            cont = go.get("continent")
            if cont:
                assert_existing_file(str(cont[0]), "tt_inverse.gravity_opts.ZC", work_dir)
            ou = go.get("ocean_upper")
            if ou:
                assert_existing_file(str(ou[0]), "tt_inverse.gravity_opts.ZU_up", work_dir)
                assert_existing_file(str(ou[1]), "tt_inverse.gravity_opts.ZU_lo", work_dir)
            ol = go.get("ocean_lower")
            if ol:
                assert_existing_file(str(ol[0]), "tt_inverse.gravity_opts.ZL", work_dir)
            sed = go.get("sediment")
            if sed:
                assert_existing_file(str(sed[0]), "tt_inverse.gravity_opts.ZS_up", work_dir)
                assert_existing_file(str(sed[1]), "tt_inverse.gravity_opts.ZS_lo", work_dir)
