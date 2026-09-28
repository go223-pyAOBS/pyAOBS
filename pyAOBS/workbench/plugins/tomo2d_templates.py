"""Template builders for TOMO2D node forms."""

from __future__ import annotations

from pathlib import Path

from pyAOBS.workbench.core.project_layout import node_id_from_work_dir


TEMPLATE_CUSTOM = "custom"
TEMPLATE_INVERSE_STANDARD = "tt_inverse_standard"
TEMPLATE_FORWARD_BASIC = "tt_forward_basic"


def list_tomo2d_templates() -> list[tuple[str, str]]:
    return [
        (TEMPLATE_CUSTOM, "自定义命令"),
        (TEMPLATE_INVERSE_STANDARD, "tt_inverse 标准反演"),
        (TEMPLATE_FORWARD_BASIC, "tt_forward 基础正演"),
    ]


def build_tomo2d_template_payload(template_id: str, fields: dict[str, str]) -> dict[str, str]:
    """
    Build executable/args/cwd/env/inputs payload from structured form fields.

    Returns plain text fields consumed by Tomo2DShellPlugin.
    """
    if template_id == TEMPLATE_CUSTOM:
        return {
            "executable": "",
            "args": "",
            "cwd": "",
            "env_text": "",
            "inputs_text": "",
            "node_id": "",
        }
    if template_id == TEMPLATE_INVERSE_STANDARD:
        return _build_inverse_standard(fields)
    if template_id == TEMPLATE_FORWARD_BASIC:
        return _build_forward_basic(fields)
    raise ValueError(f"Unknown tomo2d template: {template_id}")


def _build_inverse_standard(fields: dict[str, str]) -> dict[str, str]:
    mesh = _required(fields, "mesh_path")
    data = _required(fields, "data_path")
    iterations = fields.get("iterations", "5").strip() or "5"
    sv = fields.get("sv", "100").strip() or "100"
    sd = fields.get("sd", "10").strip() or "10"
    dv = fields.get("dv", "1").strip() or "1"
    dd = fields.get("dd", "20").strip() or "20"
    verbose = fields.get("verbose", "-1").strip() or "-1"
    extra = fields.get("extra_args", "").strip()
    cwd = fields.get("work_dir", "tools/tomo2d").strip() or "tools/tomo2d"
    node_id = fields.get("node_id", "").strip() or node_id_from_work_dir(cwd) or "workspace"

    mesh_arg = _path_for_arg(mesh, cwd)
    data_arg = _path_for_arg(data, cwd)
    args = (
        f"-M{mesh_arg} -G{data_arg} -I{iterations} "
        f"-SV{sv} -SD{sd} -DV{dv} -DD{dd} -V{verbose}"
    )
    args, extra_inputs = _append_interface_args(
        args, fields, cwd, seafloor_flag="Y", include_invert_flags=True
    )
    if extra:
        args += f" {extra}"
    inputs = "\n".join(_path_for_inputs([mesh, data, *extra_inputs], cwd)) + "\n"
    env_text = (
        "TOMO2D_INV_OMP=1\n"
        "OMP_NUM_THREADS=4\n"
        "OMP_PLACES=threads\n"
        "OMP_PROC_BIND=spread\n"
    )
    return {
        "executable": "tt_inverse",
        "args": args,
        "cwd": cwd,
        "env_text": env_text,
        "inputs_text": inputs,
        "node_id": node_id,
    }


def _build_forward_basic(fields: dict[str, str]) -> dict[str, str]:
    mesh = _required(fields, "mesh_path")
    data = _required(fields, "data_path")
    verbose = fields.get("verbose", "-1").strip() or "-1"
    extra = fields.get("extra_args", "").strip()
    cwd = fields.get("work_dir", "tools/tomo2d").strip() or "tools/tomo2d"
    node_id = fields.get("node_id", "").strip() or node_id_from_work_dir(cwd) or "workspace"

    mesh_arg = _path_for_arg(mesh, cwd)
    data_arg = _path_for_arg(data, cwd)
    args = f"-M{mesh_arg} -G{data_arg} -V{verbose}"
    args, extra_inputs = _append_interface_args(
        args, fields, cwd, seafloor_flag="B", include_invert_flags=False
    )
    if extra:
        args += f" {extra}"
    inputs = "\n".join(_path_for_inputs([mesh, data, *extra_inputs], cwd)) + "\n"
    env_text = (
        "TOMO2D_FWD_OMP=1\n"
        "OMP_NUM_THREADS=4\n"
        "OMP_PLACES=threads\n"
        "OMP_PROC_BIND=spread\n"
    )
    return {
        "executable": "tt_forward",
        "args": args,
        "cwd": cwd,
        "env_text": env_text,
        "inputs_text": inputs,
        "node_id": node_id,
    }


def _flag_on(fields: dict[str, str], key: str) -> bool:
    raw = str(fields.get(key, "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _append_interface_args(
    args: str,
    fields: dict[str, str],
    cwd: str,
    *,
    seafloor_flag: str,
    include_invert_flags: bool,
) -> tuple[str, list[str]]:
    """Append optional -Y/-B, -F, and inverse-only -y/-w/-u.

    Empty seafloor/refl keeps icode 0/1 defaults. Invert flags are ignored on
    forward templates.
    """
    extra_inputs: list[str] = []
    seafloor = fields.get("seafloor_path", "").strip()
    refl = fields.get("refl_path", "").strip()
    water_only = include_invert_flags and _flag_on(fields, "invert_water_only")
    crust_only = include_invert_flags and _flag_on(fields, "invert_crust_only")
    freeze_refl = include_invert_flags and _flag_on(fields, "freeze_refl")
    if water_only and crust_only:
        raise ValueError("-y 与 -w 不能同时勾选。")
    if (water_only or crust_only) and not seafloor and not refl:
        raise ValueError("-y / -w 需要海底界面 (-Y/-B) 或反射界面 (-F)。")
    if freeze_refl and not refl:
        raise ValueError("-u 需要同时填写反射界面 (-F)。")
    if seafloor:
        args += f" -{seafloor_flag}{_path_for_arg(seafloor, cwd)}"
        extra_inputs.append(seafloor)
    if refl:
        args += f" -F{_path_for_arg(refl, cwd)}"
        extra_inputs.append(refl)
    if water_only:
        args += " -y"
    if crust_only:
        args += " -w"
    if freeze_refl:
        args += " -u"
    return args, extra_inputs


def _required(fields: dict[str, str], key: str) -> str:
    value = fields.get(key, "").strip()
    if not value:
        raise ValueError(f"模板字段不能为空：{key}")
    return value


def _path_for_arg(path_text: str, cwd_text: str) -> str:
    p = Path(path_text)
    if p.is_absolute():
        return str(p)
    if cwd_text:
        return str(p)
    return str(p)


def _path_for_inputs(paths: list[str], cwd_text: str) -> list[str]:
    out: list[str] = []
    for p in paths:
        pp = Path(p)
        if pp.is_absolute():
            out.append(str(pp))
        elif cwd_text:
            out.append(str(Path(cwd_text) / pp))
        else:
            out.append(str(pp))
    return out

