"""从 FormState 收集各流程 kwargs（无 UI）。"""

from __future__ import annotations

from typing import Any

from ..state.form_state import FormState
from .refl_stride import apply_inv_refl_stride_from_state


def to_number(value: str):
    if value == "":
        return None
    try:
        if any(c in value.lower() for c in [".", "e"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def merge_grid_vars_into(
    state: FormState, kwargs: dict, pfx: str, *, grid_opt: str
) -> None:
    """仅合并当前 grid_opt 下界面启用的网格项，忽略灰显框里的残留文本。"""
    keys_by_grid = {
        "uniform": ("nx", "nz", "xmax", "zmax"),
        "variable": ("x_file", "z_file", "topo_file"),
        "zelt": ("dx", "z_file"),
    }
    for key in keys_by_grid.get(grid_opt, ()):
        vk = f"{pfx}{key}"
        if not state.has(vk):
            continue
        raw = state.get_str(vk)
        if not raw:
            continue
        if key.endswith("_file"):
            kwargs[key] = raw
        else:
            kwargs[key] = to_number(raw)


def collect_gen_smesh_kwargs(state: FormState) -> dict:
    kwargs: dict = {
        "vel_opt": state.get_str("gen.vel_opt"),
        "grid_opt": state.get_str("gen.grid_opt"),
    }
    if kwargs["vel_opt"] == "zelt":
        kwargs["grid_opt"] = "zelt"
    path_keys = frozenset({"v_in", "refl_file"})
    vo = kwargs["vel_opt"]
    if vo == "uniform":
        vel_keys = ["v0", "gradient"]
    elif vo == "zelt":
        vel_keys = ["v_in", "ilayer", "refl_layer", "refl_file"]
    else:
        vel_keys = []
    for key in vel_keys:
        raw = state.get_str(f"gen.{key}")
        if raw:
            kwargs[key] = raw if key in path_keys else to_number(raw)
    merge_grid_vars_into(state, kwargs, "gen.", grid_opt=kwargs["grid_opt"])
    for key in ["water_col", "v_water", "v_air"]:
        raw = state.get_str(f"gen.{key}")
        if raw:
            kwargs[key] = to_number(raw)
    if vo == "zelt":
        zd = state.get_str("gen.zelt_dump")
        if zd:
            kwargs["zelt_dump_file"] = zd
        if state.get_bool("gen.hang_sea_surface"):
            kwargs["hang_sea_surface"] = True
        sf_out = state.get_str("gen.seafloor_out")
        if sf_out:
            if not kwargs.get("hang_sea_surface"):
                raise ValueError("gen_smesh：写出海底界面 (-G) 需要先勾选挂海面 topo=0 (-S)")
            kwargs["seafloor_out"] = sf_out
    smesh_out = state.get_str("gen.smesh_out")
    if smesh_out:
        kwargs["out_file"] = smesh_out
    return kwargs


def collect_gen_damp_kwargs(state: FormState) -> dict:
    kwargs: dict = {
        "vel_opt": state.get_str("damp.vel_opt"),
        "grid_opt": state.get_str("damp.grid_opt"),
    }
    if kwargs["vel_opt"] == "zelt":
        kwargs["grid_opt"] = "zelt"
    # -A 数值在 uniform/zelt 下都需要（zelt 的 -C/-F 只划区）
    for key in ("abnormal_damp", "normal_damp"):
        raw = state.get_str(f"damp.{key}")
        if raw:
            kwargs[key] = to_number(raw)
    if kwargs["vel_opt"] == "zelt":
        for key in ("v_in", "ilayer", "top_layer", "bot_layer"):
            raw = state.get_str(f"damp.{key}")
            if raw:
                kwargs[key] = raw if key == "v_in" else to_number(raw)
    merge_grid_vars_into(state, kwargs, "damp.", grid_opt=kwargs["grid_opt"])
    return kwargs


def collect_gen_vcorr_kwargs(state: FormState) -> dict:
    mode = state.get_str("vcorr.mode") or "program"
    if mode == "simple_2x2":
        kwargs: dict = {"mode": "simple_2x2"}
        for key in ("Lht", "Lhb", "Lvt", "Lvb", "xmin", "xmax", "zmin", "zmax"):
            raw = state.get_str(f"vcorr.{key}")
            if raw:
                kwargs[key] = to_number(raw)
        out = state.get_str("vcorr.out_file")
        if out:
            kwargs["out_file"] = out
        return kwargs
    kwargs = {
        "mode": "program",
        "vel_opt": state.get_str("vcorr.vel_opt"),
        "grid_opt": state.get_str("vcorr.grid_opt"),
    }
    if kwargs["vel_opt"] == "zelt":
        kwargs["grid_opt"] = "zelt"
    # -A 数值在 uniform/zelt 下都需要
    for key in ("abnormal_h", "abnormal_v", "normal_h", "normal_v"):
        raw = state.get_str(f"vcorr.{key}")
        if raw:
            kwargs[key] = to_number(raw)
    if kwargs["vel_opt"] == "zelt":
        for key in ("v_in", "ilayer", "top_layer", "bot_layer"):
            raw = state.get_str(f"vcorr.{key}")
            if raw:
                kwargs[key] = raw if key == "v_in" else to_number(raw)
    merge_grid_vars_into(state, kwargs, "vcorr.", grid_opt=kwargs["grid_opt"])
    out = state.get_str("vcorr.out_file")
    if out:
        kwargs["out_file"] = out
    return kwargs


def collect_gen_dcorr_kwargs(state: FormState) -> dict:
    mode = state.get_str("dcorr.mode") or "uniform"
    kwargs: dict = {"mode": mode}
    out = state.get_str("dcorr.out_file")
    if out:
        kwargs["out_file"] = out
    if mode == "uniform":
        for key in ("lh", "xmin", "xmax", "nx"):
            raw = state.get_str(f"dcorr.{key}")
            if raw:
                kwargs[key] = to_number(raw)
    elif mode == "zelt":
        for key in ("abnormal_d", "normal_d", "ilayer", "top_layer", "bot_layer", "dx"):
            raw = state.get_str(f"dcorr.{key}")
            if raw:
                kwargs[key] = to_number(raw)
        vin = state.get_str("dcorr.v_in")
        if vin:
            kwargs["v_in"] = vin
        refl = state.get_str("dcorr.refl_file")
        if refl:
            kwargs["refl_file"] = refl
    elif mode == "from_vcorr":
        vf = state.get_str("dcorr.vcorr_file")
        if vf:
            kwargs["vcorr_file"] = vf
        refl = state.get_str("dcorr.refl_file")
        if refl:
            kwargs["refl_file"] = refl
    return kwargs


def collect_tt_inverse_args(state: FormState) -> tuple[str, str, dict]:
    mesh = state.get_str("inv.mesh")
    data = state.get_str("inv.data")
    kwargs: dict = {}
    for key in ["xorder", "zorder", "clen", "nintp", "bend_cg_tol", "bend_br_tol"]:
        raw = state.get_str(f"inv.{key}")
        if raw:
            kwargs[key] = to_number(raw)
    refl = state.get_str("inv.refl_file")
    if refl:
        kwargs["refl_file"] = refl
    sf = state.get_str("inv.seafloor_file")
    if sf:
        kwargs["seafloor_file"] = sf
    conv = state.get_str("inv.conv_file")
    if conv:
        kwargs["conv_file"] = conv
    vs = state.get_str("inv.vsmesh")
    if vs:
        kwargs["vsmesh"] = vs
    kappa = state.get_str("inv.kappa")
    if kappa:
        kwargs["kappa"] = kappa.strip()
    apply_inv_refl_stride_from_state(state, kwargs)
    rw = state.get_str("inv.refl_weight")
    if rw:
        kwargs["refl_weight"] = to_number(rw)
    if state.get_bool("inv.do_full_refl"):
        kwargs["do_full_refl"] = True
    if state.get_bool("inv.freeze_refl"):
        if not kwargs.get("refl_file"):
            raise ValueError("tt_inverse：冻结界面 (-u) 需要先指定 refl_file (-F)")
        kwargs["freeze_refl"] = True
    yw = state.get_bool("inv.invert_water_only")
    cw = state.get_bool("inv.invert_crust_only")
    if yw and cw:
        raise ValueError("tt_inverse：只反水 (-y) 与只反壳 (-w) 互斥")
    if yw:
        if not kwargs.get("seafloor_file") and not kwargs.get("refl_file"):
            raise ValueError("tt_inverse：只反水 (-y) 需要 seafloor_file (-Y) 或 refl_file (-F)")
        kwargs["invert_water_only"] = True
    if cw:
        if not kwargs.get("seafloor_file") and not kwargs.get("refl_file"):
            raise ValueError("tt_inverse：只反壳 (-w) 需要 seafloor_file (-Y) 或 refl_file (-F)")
        kwargs["invert_crust_only"] = True
    if state.get_bool("inv.jumping"):
        kwargs["jumping"] = True
    if state.get_bool("inv.print_final_only"):
        kwargs["print_final_only"] = True
    apply_s = state.get_bool("inv.apply_filter")
    fb = state.get_str("inv.filter_bound_file")
    if apply_s:
        kwargs["apply_filter"] = True
        if fb:
            kwargs["filter_bound_file"] = fb
    elif fb and not state.has("inv.apply_filter"):
        # 脚本/旧 FormState 只填了文件
        kwargs["filter_bound_file"] = fb
    for key in ["log_file", "out_root", "out_level", "dws_file"]:
        raw = state.get_str(f"inv.{key}")
        if raw:
            kwargs[key] = raw if key != "out_level" else to_number(raw)
    for key in ["crit_chi", "lsqr_tol", "niter", "target_chi2"]:
        raw = state.get_str(f"inv.{key}")
        if raw:
            kwargs[key] = to_number(raw)

    adv = state.get_str("inv.auto_damp_max_dv")
    add = state.get_str("inv.auto_damp_max_dd")
    dv = state.get_str("inv.damp_vel")
    dd = state.get_str("inv.damp_dep")
    dq = state.get_str("inv.damp_v_fn")
    kind = (state.get_str("inv.damp_kind") or "").strip()
    auto_on = bool(adv or add)
    fixed_on = bool(dv or dd or dq)
    # 灰显侧仍可能留着旧字：按 damp_kind 丢掉，避免 -T/-D 同时进命令行
    if kind == "auto":
        fixed_on = False
    elif kind == "fixed":
        auto_on = False
    if auto_on and fixed_on:
        raise ValueError(
            "tt_inverse：自动阻尼 (-TV/-TD) 与固定阻尼 (-DV/-DD/-DQ) 互斥，请清空其中一侧"
        )
    if auto_on:
        if adv:
            kwargs["auto_damp_max_dv"] = to_number(adv)
        if add:
            kwargs["auto_damp_max_dd"] = to_number(add)
    if fixed_on:
        damp: dict = {}
        if dv:
            damp["vel"] = to_number(dv)
        if dd:
            damp["dep"] = to_number(dd)
        if dq:
            damp["damp_v_fn"] = dq
        if damp:
            kwargs["damp_opts"] = damp

    sv = state.get_str("inv.smooth_vel")
    sd = state.get_str("inv.smooth_dep")
    cv = state.get_str("inv.smooth_corr_v_fn")
    cd = state.get_str("inv.smooth_corr_d_fn")
    xv = state.get_bool("inv.smooth_vel_log10")
    xd = state.get_bool("inv.smooth_dep_log10")
    if sv or sd or cv or cd or xv or xd:
        smooth: dict = {}
        if sv:
            smooth["vel"] = sv if ("/" in sv and sv.count("/") >= 2) else to_number(sv)
        if sd:
            smooth["dep"] = sd if ("/" in sd and sd.count("/") >= 2) else to_number(sd)
        if cv:
            smooth["corr_v_fn"] = cv
        if cd:
            smooth["corr_d_fn"] = cd
        if xv:
            smooth["vel_log10"] = True
        if xd:
            smooth["dep_log10"] = True
        kwargs["smooth_opts"] = smooth

    gf = state.get_str("inv.grav_file")
    if gf:
        g: dict[str, Any] = {"grav_file": gf}
        gg = state.get_str("inv.grav_grid")
        grng = state.get_str("inv.grav_refrange")
        if gg:
            g["grid_spec"] = gg
        if grng:
            g["refrange"] = grng
        cf = state.get_str("inv.grav_cont_file")
        if cf:
            ic = state.get_str("inv.grav_cont_iconv")
            if not ic:
                raise ValueError("联合重力：已填 grav_ZC cont_up 时须填 grav_ZC iconv")
            g["continent"] = (cf, int(to_number(ic)))
        uu = state.get_str("inv.grav_oceanU_up")
        ul = state.get_str("inv.grav_oceanU_lo")
        ui = state.get_str("inv.grav_oceanU_iconv")
        if uu or ul or ui:
            if not (uu and ul and ui):
                raise ValueError("联合重力：grav_ZU 三项须同时填写")
            g["ocean_upper"] = (uu, ul, int(to_number(ui)))
        lu = state.get_str("inv.grav_oceanL_up")
        li = state.get_str("inv.grav_oceanL_iconv")
        if lu or li:
            if not (lu and li):
                raise ValueError("联合重力：grav_ZL 两项须同时填写")
            g["ocean_lower"] = (lu, int(to_number(li)))
        su = state.get_str("inv.grav_sed_up")
        sl = state.get_str("inv.grav_sed_lo")
        si = state.get_str("inv.grav_sed_iconv")
        if su or sl or si:
            if not (su and sl and si):
                raise ValueError("联合重力：grav_ZS 三项须同时填写")
            g["sediment"] = (su, sl, int(to_number(si)))
        gd = state.get_str("inv.grav_deriv")
        if gd:
            g["deriv"] = gd
        gw = state.get_str("inv.grav_weight")
        if gw:
            g["weight_grav"] = to_number(gw)
        gz = state.get_str("inv.grav_z0")
        if gz:
            g["z0"] = to_number(gz)
        gk = state.get_str("inv.grav_dws")
        if gk:
            g["grav_dws"] = gk
        gcut = state.get_str("inv.grav_cutoff")
        if gcut:
            g["cutoff"] = gcut
        kwargs["gravity_opts"] = g

    vl = state.get_str("inv.verbose_level")
    if vl:
        val = to_number(vl)
        if isinstance(val, (int, float)):
            if val > 0:
                kwargs["verbose"] = True
                kwargs["verbose_level"] = val
        else:
            raise ValueError(
                "inv.verbose_level 需要数值（0 表示不启用 verbose，>0 表示启用 -V[level]）"
            )

    return mesh, data, kwargs


def collect_tt_forward_args(state: FormState) -> tuple[str, str | None, dict]:
    smesh = state.get_str("fwd.smesh")
    geom = state.get_str("fwd.geom") or None
    kwargs: dict = {}

    refl = state.get_str("fwd.refl_file")
    if refl:
        kwargs["refl_file"] = refl
    sf = state.get_str("fwd.seafloor_file")
    if sf:
        kwargs["seafloor_file"] = sf
    conv = state.get_str("fwd.conv_file")
    if conv:
        kwargs["conv_file"] = conv
    vs = state.get_str("fwd.vsmesh")
    if vs:
        kwargs["vsmesh"] = vs
    kappa = state.get_str("fwd.kappa")
    if kappa:
        kwargs["kappa"] = kappa.strip()

    if state.get_bool("fwd.do_full_refl"):
        kwargs["do_full_refl"] = True

    numeric = {}
    for key in ["xorder", "zorder", "clen", "nintp"]:
        raw = state.get_str(f"fwd.{key}")
        if raw:
            numeric[key] = to_number(raw)
    cg = state.get_str("fwd.bend_cg_tol")
    br = state.get_str("fwd.bend_br_tol")
    if cg:
        numeric["tol1"] = to_number(cg)
    if br:
        numeric["tol2"] = to_number(br)
    kwargs.update(numeric)

    vred = state.get_str("fwd.vred")
    if vred:
        kwargs["vred"] = to_number(vred)

    out_opts = {}
    map_out = {
        "elements": "out_elements",
        "ttime": "out_ttime",
        "obs_ttime": "out_obs_ttime",
        "ray": "out_ray",
        "source": "out_source",
        "vgrid": "out_vgrid",
        "diff": "out_diff",
    }
    for k, ui_key in map_out.items():
        raw = state.get_str(f"fwd.{ui_key}")
        if raw:
            out_opts[k] = raw
    if out_opts:
        kwargs["out_opts"] = out_opts

    sub_keys = ("sub_west", "sub_east", "sub_south", "sub_north", "sub_dx", "sub_dz")
    sub_vals = [state.get_str(f"fwd.{k}") for k in sub_keys]
    if any(sub_vals):
        if not all(sub_vals):
            raise ValueError("tt_forward: vgrid -i 六项 west/east/south/north/dx/dz 须同时填写")
        kwargs["vgrid_subregion"] = tuple(to_number(x) for x in sub_vals)

    cf = state.get_str("fwd.clock_file")
    if cf:
        kwargs["clock_file"] = cf

    if state.get_bool("fwd.graph_only"):
        kwargs["graph_only"] = True
    if state.get_bool("fwd.omit_air_water"):
        kwargs["omit_air_water"] = True

    fvl = state.get_str("fwd.verbose_level")
    if fvl:
        val = to_number(fvl)
        if isinstance(val, (int, float)):
            if val > 0:
                kwargs["verbose"] = True
                kwargs["verbose_level"] = val
        else:
            raise ValueError(
                "fwd.verbose_level 需要数值（0 或留空表示不启用 verbose，>0 表示启用 -V[level]）"
            )

    return smesh, geom, kwargs


def collect_stat_smesh_kwargs(state: FormState) -> dict:
    kwargs: dict = {
        "mode": state.get_str("stat.mode"),
        "cmd_type": state.get_str("stat.cmd_type"),
    }
    for key in [
        "list_file",
        "mesh_file",
        "ave_file",
        "ave_x",
        "window_len",
        "xmin",
        "xmax",
        "dx",
        "top_bound",
        "bot_bound",
        "mid_bound",
        "pt_corr",
        "exclude_top_bound",
        "exclude_bot_bound",
    ]:
        raw = state.get_str(f"stat.{key}")
        if raw:
            if key.endswith("_file") or key.endswith("_bound"):
                kwargs[key] = raw
            else:
                kwargs[key] = raw if key == "pt_corr" else to_number(raw)
    rn = state.get_str("stat.refl_nnodes")
    if rn:
        kwargs["refl_nnodes"] = to_number(rn)
    vr = state.get_str("stat.vrepl")
    if vr:
        kwargs["vrepl"] = to_number(vr)
    for key in ("abs_xmin", "abs_xmax", "exclude_cxmin", "exclude_cxmax"):
        raw = state.get_str(f"stat.{key}")
        if raw:
            kwargs[key] = to_number(raw)
    if state.get_bool("stat.verbose"):
        kwargs["verbose"] = True
    return kwargs


def collect_edit_smesh_args(state: FormState) -> tuple[str, str, dict]:
    smesh_file = state.get_str("edit.smesh_file")
    cmd_type = state.get_str("edit.cmd_type")
    kwargs: dict = {}
    path_keys = {
        "paste_file",
        "prof_file",
        "remove_bg_file",
        "moho_file",
        "base_file",
        "corr_file",
        "upper_bound",
    }
    for key in [
        "paste_file",
        "prof_file",
        "remove_bg_file",
        "h_len",
        "v_len",
        "mx",
        "mz",
        "amp",
        "xmin",
        "xmax",
        "zmin",
        "zmax",
        "x0",
        "z0",
        "Lh",
        "Lv",
        "seed",
        "nrand",
        "N",
        "dx",
        "dz",
        "vel",
        "moho_file",
        "k",
        "base_file",
        "corr_file",
        "upper_bound",
    ]:
        raw = state.get_str(f"edit.{key}")
        if raw:
            kwargs[key] = raw if key in path_keys else to_number(raw)
    return smesh_file, cmd_type, kwargs


def format_tt_inverse_flag_summary(kwargs: dict) -> str:
    """人话摘要：用的是 -T 还是 -D，以及是否开了 -s。供预览 / GUI 日志 / 监视。"""
    auto_dv = kwargs.get("auto_damp_max_dv")
    auto_dd = kwargs.get("auto_damp_max_dd")
    damp = kwargs.get("damp_opts") or {}
    has_fixed = any(damp.get(k) is not None for k in ("vel", "dep", "damp_v_fn"))
    if auto_dv is not None or auto_dd is not None:
        bits = ["自动阻尼 -T"]
        if auto_dv is not None:
            try:
                frac = float(auto_dv) / 100.0
                bits.append(f"-TV {auto_dv}% → 日志 frac={frac:g}")
            except (TypeError, ValueError):
                bits.append(f"-TV {auto_dv}")
        if auto_dd is not None:
            try:
                frac = float(auto_dd) / 100.0
                bits.append(f"-TD {auto_dd}% → 日志 frac={frac:g}")
            except (TypeError, ValueError):
                bits.append(f"-TD {auto_dd}")
        damp_txt = "；".join(bits)
    elif has_fixed:
        bits = ["固定阻尼 -D"]
        if damp.get("vel") is not None:
            bits.append(f"-DV {damp['vel']}")
        if damp.get("dep") is not None:
            bits.append(f"-DD {damp['dep']}")
        if damp.get("damp_v_fn"):
            bits.append(f"-DQ {damp['damp_v_fn']}")
        damp_txt = "；".join(bits)
    else:
        damp_txt = ""
    fb = kwargs.get("filter_bound_file")
    if kwargs.get("apply_filter") or fb:
        if fb:
            s_txt = f"滤波 -s（bound={fb}）"
        else:
            s_txt = "滤波 -s（mesh 地形）"
    else:
        s_txt = ""
    freeze_txt = "冻结界面 -u" if kwargs.get("freeze_refl") else ""
    water_txt = ""
    if kwargs.get("invert_water_only"):
        water_txt = "只反水 -y"
    elif kwargs.get("invert_crust_only"):
        water_txt = "只反壳 -w"
    if kwargs.get("seafloor_file"):
        water_txt = (water_txt + " " if water_txt else "") + "海底 -Y"
    parts = [p for p in (damp_txt, s_txt, freeze_txt, water_txt) if p]
    return " · ".join(parts)
