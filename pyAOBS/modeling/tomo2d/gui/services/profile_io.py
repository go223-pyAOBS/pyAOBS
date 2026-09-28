"""配置 / manifest 加载辅助（无 UI；返回更新字典，由壳层写回控件）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..state.form_state import FormState
from .paths import normalize_file_path_vars, resolve_work_dir, to_workdir_relative


_INV_CLEAR_STR_KEYS = (
    "inv.mesh",
    "inv.data",
    "inv.xorder",
    "inv.zorder",
    "inv.clen",
    "inv.nintp",
    "inv.bend_cg_tol",
    "inv.bend_br_tol",
    "inv.refl_file",
    "inv.seafloor_file",
    "inv.refl_stride",
    "inv.refl_weight",
    "inv.log_file",
    "inv.out_root",
    "inv.out_level",
    "inv.dws_file",
    "inv.crit_chi",
    "inv.lsqr_tol",
    "inv.niter",
    "inv.target_chi2",
    "inv.auto_damp_max_dv",
    "inv.auto_damp_max_dd",
    "inv.smooth_vel",
    "inv.smooth_dep",
    "inv.smooth_corr_v_fn",
    "inv.smooth_corr_d_fn",
    "inv.damp_vel",
    "inv.damp_dep",
    "inv.damp_v_fn",
    "inv.filter_bound_file",
    "inv.bundle_run_label",
    "inv.grav_file",
    "inv.grav_grid",
    "inv.grav_refrange",
    "inv.grav_cont_file",
    "inv.grav_cont_iconv",
    "inv.grav_oceanU_up",
    "inv.grav_oceanU_lo",
    "inv.grav_oceanU_iconv",
    "inv.grav_oceanL_up",
    "inv.grav_oceanL_iconv",
    "inv.grav_sed_up",
    "inv.grav_sed_lo",
    "inv.grav_sed_iconv",
    "inv.grav_deriv",
    "inv.grav_weight",
    "inv.grav_z0",
    "inv.grav_dws",
    "inv.grav_cutoff",
    "inv.verbose_level",
)

_INV_CLEAR_BOOL_KEYS = (
    "inv.do_full_refl",
    "inv.freeze_refl",
    "inv.invert_water_only",
    "inv.invert_crust_only",
    "inv.jumping",
    "inv.print_final_only",
    "inv.apply_filter",
    "inv.smooth_vel_log10",
    "inv.smooth_dep_log10",
)


def json_to_entry_str(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, bool):
        return "1" if val else ""
    if isinstance(val, float):
        if val == int(val):
            return str(int(val))
        return repr(val)
    return str(val).strip()


def tt_inverse_replay_path_to_gui(p: str | None, run_dir: Path) -> str:
    """manifest 内相对路径（inputs/、outputs/）转为绝对路径填入表单。"""
    if p is None:
        return ""
    s = str(p).strip()
    if not s:
        return ""
    path = Path(s.replace("\\", "/"))
    if path.is_absolute():
        try:
            return str(path.resolve())
        except OSError:
            return s
    try:
        return str((run_dir / path).resolve())
    except OSError:
        return str(run_dir / path)


def clear_tt_inverse_form_keys(state: FormState) -> None:
    """清空 tt_inverse 页可编辑项（保留 inv.use_repro_bundle）。"""
    preserve = state.get("inv.use_repro_bundle")
    for key in _INV_CLEAR_STR_KEYS:
        if state.has(key):
            state.set(key, "")
    for key in _INV_CLEAR_BOOL_KEYS:
        if state.has(key):
            state.set(key, False)
    if preserve is not None:
        state.set("inv.use_repro_bundle", preserve)


def apply_tt_inverse_kwargs_updates(kwargs: dict, run_dir: Path) -> dict[str, Any]:
    """将 TomoAnd tt_inverse 的 kwargs 转为表单键更新字典。"""
    u: dict[str, Any] = {}
    for tk_key, py_key in (
        ("inv.xorder", "xorder"),
        ("inv.zorder", "zorder"),
        ("inv.clen", "clen"),
        ("inv.nintp", "nintp"),
        ("inv.bend_cg_tol", "bend_cg_tol"),
        ("inv.bend_br_tol", "bend_br_tol"),
        ("inv.refl_weight", "refl_weight"),
        ("inv.out_level", "out_level"),
        ("inv.crit_chi", "crit_chi"),
        ("inv.lsqr_tol", "lsqr_tol"),
        ("inv.niter", "niter"),
        ("inv.target_chi2", "target_chi2"),
        ("inv.auto_damp_max_dv", "auto_damp_max_dv"),
        ("inv.auto_damp_max_dd", "auto_damp_max_dd"),
    ):
        if py_key in kwargs:
            u[tk_key] = json_to_entry_str(kwargs[py_key])

    if "refl_file" in kwargs:
        u["inv.refl_file"] = tt_inverse_replay_path_to_gui(str(kwargs["refl_file"]), run_dir)
    if "seafloor_file" in kwargs:
        u["inv.seafloor_file"] = tt_inverse_replay_path_to_gui(
            str(kwargs["seafloor_file"]), run_dir
        )
    if "filter_bound_file" in kwargs:
        u["inv.filter_bound_file"] = tt_inverse_replay_path_to_gui(
            str(kwargs["filter_bound_file"]), run_dir
        )
    if "log_file" in kwargs:
        u["inv.log_file"] = tt_inverse_replay_path_to_gui(str(kwargs["log_file"]), run_dir)
    if "out_root" in kwargs:
        u["inv.out_root"] = tt_inverse_replay_path_to_gui(str(kwargs["out_root"]), run_dir)
    if "dws_file" in kwargs:
        u["inv.dws_file"] = tt_inverse_replay_path_to_gui(str(kwargs["dws_file"]), run_dir)

    if kwargs.get("do_full_refl"):
        u["inv.do_full_refl"] = True
    if kwargs.get("freeze_refl"):
        u["inv.freeze_refl"] = True
    if kwargs.get("invert_water_only"):
        u["inv.invert_water_only"] = True
    if kwargs.get("invert_crust_only"):
        u["inv.invert_crust_only"] = True
    if kwargs.get("jumping"):
        u["inv.jumping"] = True
    if kwargs.get("print_final_only"):
        u["inv.print_final_only"] = True
    if kwargs.get("apply_filter") or kwargs.get("filter_bound_file"):
        u["inv.apply_filter"] = True

    sm = kwargs.get("smooth_opts") or {}
    if sm.get("vel") is not None:
        u["inv.smooth_vel"] = json_to_entry_str(sm["vel"])
    if sm.get("dep") is not None:
        u["inv.smooth_dep"] = json_to_entry_str(sm["dep"])
    if sm.get("corr_v_fn"):
        u["inv.smooth_corr_v_fn"] = tt_inverse_replay_path_to_gui(str(sm["corr_v_fn"]), run_dir)
    if sm.get("corr_d_fn"):
        u["inv.smooth_corr_d_fn"] = tt_inverse_replay_path_to_gui(str(sm["corr_d_fn"]), run_dir)
    if sm.get("vel_log10"):
        u["inv.smooth_vel_log10"] = True
    if sm.get("dep_log10"):
        u["inv.smooth_dep_log10"] = True

    dm = kwargs.get("damp_opts") or {}
    if dm.get("damp_v_fn"):
        u["inv.damp_v_fn"] = tt_inverse_replay_path_to_gui(str(dm["damp_v_fn"]), run_dir)
    if dm.get("vel") is not None:
        u["inv.damp_vel"] = json_to_entry_str(dm["vel"])
    if dm.get("dep") is not None:
        u["inv.damp_dep"] = json_to_entry_str(dm["dep"])

    g = kwargs.get("gravity_opts") or {}
    if g.get("grav_file"):
        u["inv.grav_file"] = tt_inverse_replay_path_to_gui(str(g["grav_file"]), run_dir)
    if g.get("grid_spec"):
        u["inv.grav_grid"] = json_to_entry_str(g["grid_spec"])
    if g.get("refrange"):
        u["inv.grav_refrange"] = json_to_entry_str(g["refrange"])
    cont = g.get("continent")
    if cont and isinstance(cont, (list, tuple)) and len(cont) >= 2:
        u["inv.grav_cont_file"] = tt_inverse_replay_path_to_gui(str(cont[0]), run_dir)
        u["inv.grav_cont_iconv"] = json_to_entry_str(cont[1])
    ou = g.get("ocean_upper")
    if ou and isinstance(ou, (list, tuple)) and len(ou) >= 3:
        u["inv.grav_oceanU_up"] = tt_inverse_replay_path_to_gui(str(ou[0]), run_dir)
        u["inv.grav_oceanU_lo"] = tt_inverse_replay_path_to_gui(str(ou[1]), run_dir)
        u["inv.grav_oceanU_iconv"] = json_to_entry_str(ou[2])
    ol = g.get("ocean_lower")
    if ol and isinstance(ol, (list, tuple)) and len(ol) >= 2:
        u["inv.grav_oceanL_up"] = tt_inverse_replay_path_to_gui(str(ol[0]), run_dir)
        u["inv.grav_oceanL_iconv"] = json_to_entry_str(ol[1])
    sed = g.get("sediment")
    if sed and isinstance(sed, (list, tuple)) and len(sed) >= 3:
        u["inv.grav_sed_up"] = tt_inverse_replay_path_to_gui(str(sed[0]), run_dir)
        u["inv.grav_sed_lo"] = tt_inverse_replay_path_to_gui(str(sed[1]), run_dir)
        u["inv.grav_sed_iconv"] = json_to_entry_str(sed[2])
    if g.get("deriv"):
        u["inv.grav_deriv"] = json_to_entry_str(g["deriv"])
    if g.get("weight_grav") is not None:
        u["inv.grav_weight"] = json_to_entry_str(g["weight_grav"])
    if g.get("z0") is not None:
        u["inv.grav_z0"] = json_to_entry_str(g["z0"])
    if g.get("grav_dws"):
        u["inv.grav_dws"] = tt_inverse_replay_path_to_gui(str(g["grav_dws"]), run_dir)
    if g.get("cutoff"):
        u["inv.grav_cutoff"] = json_to_entry_str(g["cutoff"])

    if kwargs.get("verbose"):
        vl = kwargs.get("verbose_level")
        u["inv.verbose_level"] = json_to_entry_str(vl) if vl is not None else "1"
    return u


def merge_python_replay_updates(manifest: dict, source_path: str) -> dict[str, Any]:
    """用 python_replay 生成 inv.* 更新字典。"""
    pr = manifest.get("python_replay")
    if not isinstance(pr, dict):
        return {}
    mesh, data = pr.get("mesh"), pr.get("data")
    if not mesh or not data:
        return {}
    rd = manifest.get("run_dir")
    if rd:
        run_dir = Path(str(rd)).resolve()
    else:
        run_dir = Path(source_path).resolve().parent
    u: dict[str, Any] = {
        "inv.mesh": tt_inverse_replay_path_to_gui(str(mesh), run_dir),
        "inv.data": tt_inverse_replay_path_to_gui(str(data), run_dir),
    }
    u.update(apply_tt_inverse_kwargs_updates(dict(pr.get("kwargs") or {}), run_dir))
    return u


def sync_fwd_smesh_from_inv_if_missing(state: FormState) -> None:
    """若 fwd.smesh 找不到而 inv.mesh 存在，则用 inv.mesh 填 fwd。"""
    if not state.has("fwd.smesh") or not state.has("inv.mesh"):
        return
    work = resolve_work_dir(state.get_str("work_dir"))

    def resolved_file(pstr: str) -> Path | None:
        s = (pstr or "").strip()
        if not s:
            return None
        path = Path(s).expanduser()
        if not path.is_absolute():
            path = (work / s).resolve()
        else:
            path = path.resolve()
        return path if path.is_file() else None

    if resolved_file(state.get_str("fwd.smesh")) is not None:
        return
    im = state.get_str("inv.mesh")
    ip = resolved_file(im)
    if ip is None:
        return
    rel = to_workdir_relative(str(ip), work, warn_outside=False)
    state.set("fwd.smesh", rel.value)


def apply_manifest_to_state(
    state: FormState,
    manifest: dict,
    source_path: str,
    *,
    file_keys: list[str] | None = None,
) -> list[str]:
    """
    将 tt_inverse manifest 合并进 ``state``（原地）。
    返回日志消息；``work_dir`` 同步询问仍由 UI 层处理。
    """
    notes: list[str] = []
    file_keys = list(file_keys or [])
    gp = manifest.get("gui_profile")
    if isinstance(gp, dict) and gp:
        state.apply_mapping(gp)
        notes.append(
            "[manifest] 已从 gui_profile 恢复整界面；随后将解析路径（工作目录 / python_replay）"
        )
        if file_keys:
            normalize_file_path_vars(state, file_keys, warn_outside=False)
        state.update(merge_python_replay_updates(manifest, source_path))
        if file_keys:
            normalize_file_path_vars(state, file_keys, warn_outside=False)
        sync_fwd_smesh_from_inv_if_missing(state)
        if file_keys:
            normalize_file_path_vars(state, file_keys, warn_outside=False)
        return notes

    pr = manifest.get("python_replay")
    if not isinstance(pr, dict):
        raise ValueError("manifest 缺少 gui_profile 与 python_replay，无法加载")
    mesh = pr.get("mesh")
    data_path = pr.get("data")
    if not mesh or not data_path:
        raise ValueError("python_replay 缺少 mesh 或 data")
    rd = manifest.get("run_dir")
    if rd:
        run_dir = Path(str(rd)).resolve()
    else:
        run_dir = Path(source_path).resolve().parent
    clear_tt_inverse_form_keys(state)
    state.set("inv.mesh", tt_inverse_replay_path_to_gui(str(mesh), run_dir))
    state.set("inv.data", tt_inverse_replay_path_to_gui(str(data_path), run_dir))
    state.update(apply_tt_inverse_kwargs_updates(dict(pr.get("kwargs") or {}), run_dir))
    if file_keys:
        normalize_file_path_vars(state, file_keys, warn_outside=False)
    sync_fwd_smesh_from_inv_if_missing(state)
    notes.append("[manifest] 仅从 python_replay 恢复 tt_inverse")
    return notes


def read_profile_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("配置文件根节点须为 JSON 对象")
    return data
