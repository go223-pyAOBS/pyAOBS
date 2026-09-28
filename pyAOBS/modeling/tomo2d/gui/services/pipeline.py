"""Pipeline 计划构建与步骤命令行解析（无 UI）。"""

from __future__ import annotations

from typing import Any

from ...tomand import TomoAnd
from ..state.form_state import FormState
from .collectors import (
    collect_gen_damp_kwargs,
    collect_gen_smesh_kwargs,
    collect_gen_vcorr_kwargs,
    collect_gen_dcorr_kwargs,
    collect_tt_forward_args,
    collect_tt_inverse_args,
)


def pipeline_links(state: FormState) -> dict[str, str]:
    return {
        "smesh": state.get_str("pipe.link_smesh"),
        "damp": state.get_str("pipe.link_damp"),
        "vcorr_v": state.get_str("pipe.link_vcorr_v"),
        "vcorr_d": state.get_str("pipe.link_vcorr_d"),
        "dcorr": state.get_str("pipe.link_dcorr"),
    }


def auto_wire_pipeline_targets(
    recipe: str,
    payload: dict,
    state: FormState,
    *,
    auto_wire: bool | None = None,
) -> dict:
    """
    将上游生成文件桥接到下游参数：
    - 仅在 auto_wire=True 且目标参数为空时填入
    - 已有显式参数优先，不覆盖
    会原地更新 ``state`` 中对应表单键。
    """
    if auto_wire is None:
        auto_wire = state.get_bool("pipe.auto_wire")
    if not auto_wire:
        return payload
    links = pipeline_links(state)

    if "tt_forward" in recipe and not payload.get("smesh") and links["smesh"]:
        payload["smesh"] = links["smesh"]
        state.set("fwd.smesh", links["smesh"])
    if "tt_inverse" in recipe and not payload.get("mesh") and links["smesh"]:
        payload["mesh"] = links["smesh"]
        state.set("inv.mesh", links["smesh"])

    if "gen_damp" in recipe and links["damp"]:
        damp_opts = payload.get("inv_kwargs", {}).get("damp_opts", {}) or {}
        if "damp_v_fn" not in damp_opts:
            damp_opts["damp_v_fn"] = links["damp"]
            payload["inv_kwargs"]["damp_opts"] = damp_opts
            state.set("inv.damp_v_fn", links["damp"])

    if "gen_vcorr" in recipe:
        vcorr_out = links["vcorr_v"] or state.get_str("vcorr.out_file")
        smooth_opts = payload.get("inv_kwargs", {}).get("smooth_opts", {}) or {}
        changed = False
        if vcorr_out and "corr_v_fn" not in smooth_opts:
            smooth_opts["corr_v_fn"] = vcorr_out
            state.set("inv.smooth_corr_v_fn", vcorr_out)
            if not links["vcorr_v"]:
                state.set("pipe.link_vcorr_v", vcorr_out)
            changed = True
        if links["vcorr_d"] and "corr_d_fn" not in smooth_opts:
            smooth_opts["corr_d_fn"] = links["vcorr_d"]
            state.set("inv.smooth_corr_d_fn", links["vcorr_d"])
            changed = True
        if changed:
            payload["inv_kwargs"]["smooth_opts"] = smooth_opts

    if "gen_dcorr" in recipe:
        dcorr = links.get("dcorr") or state.get_str("dcorr.out_file")
        if dcorr:
            smooth_opts = payload.get("inv_kwargs", {}).get("smooth_opts", {}) or {}
            if "corr_d_fn" not in smooth_opts:
                smooth_opts["corr_d_fn"] = dcorr
                payload.setdefault("inv_kwargs", {})["smooth_opts"] = smooth_opts
                state.set("inv.smooth_corr_d_fn", dcorr)
                if not links.get("dcorr"):
                    state.set("pipe.link_dcorr", dcorr)

    # 正演输出走时 → 反演 data
    if "tt_forward" in recipe and "tt_inverse" in recipe:
        out_tt = state.get_str("fwd.out_ttime")
        if out_tt and not payload.get("data"):
            payload["data"] = out_tt
            state.set("inv.data", out_tt)
        smesh_fwd = payload.get("smesh") or state.get_str("fwd.smesh")
        if smesh_fwd and not payload.get("mesh"):
            payload["mesh"] = smesh_fwd
            state.set("inv.mesh", smesh_fwd)
            state.set("fwd.smesh", smesh_fwd)

    return payload


def build_pipeline_plan(state: FormState) -> list[tuple[str, dict]]:
    recipe = state.get_str("pipe.recipe")
    if recipe == "gen_smesh -> tt_forward":
        gen_kwargs = collect_gen_smesh_kwargs(state)
        smesh, geom, fwd_kwargs = collect_tt_forward_args(state)
        payload = auto_wire_pipeline_targets(
            recipe,
            {"smesh": smesh, "geom": geom, "fwd_kwargs": fwd_kwargs},
            state,
        )
        smesh = payload["smesh"]
        geom = payload["geom"]
        fwd_kwargs = payload["fwd_kwargs"]
        if not smesh:
            raise ValueError("pipeline(gen_smesh -> tt_forward) 要求 tt_forward 的 smesh 已填写")
        return [
            ("gen_smesh", {"kwargs": gen_kwargs}),
            ("tt_forward", {"smesh": smesh, "geom": geom, "kwargs": fwd_kwargs}),
        ]
    if recipe == "gen_smesh -> tt_inverse":
        gen_kwargs = collect_gen_smesh_kwargs(state)
        mesh, data, inv_kwargs = collect_tt_inverse_args(state)
        payload = auto_wire_pipeline_targets(
            recipe,
            {"mesh": mesh, "data": data, "inv_kwargs": inv_kwargs},
            state,
        )
        mesh = payload["mesh"]
        data = payload["data"]
        inv_kwargs = payload["inv_kwargs"]
        if not mesh or not data:
            raise ValueError("pipeline(gen_smesh -> tt_inverse) 要求 tt_inverse 的 mesh/data 已填写")
        return [
            ("gen_smesh", {"kwargs": gen_kwargs}),
            ("tt_inverse", {"mesh": mesh, "data": data, "kwargs": inv_kwargs}),
        ]
    if recipe == "gen_smesh -> gen_damp -> tt_inverse":
        gen_kwargs = collect_gen_smesh_kwargs(state)
        damp_kwargs = collect_gen_damp_kwargs(state)
        mesh, data, inv_kwargs = collect_tt_inverse_args(state)
        payload = auto_wire_pipeline_targets(
            recipe,
            {"mesh": mesh, "data": data, "inv_kwargs": inv_kwargs},
            state,
        )
        mesh = payload["mesh"]
        data = payload["data"]
        inv_kwargs = payload["inv_kwargs"]
        if not mesh or not data:
            raise ValueError(
                "pipeline(gen_smesh -> gen_damp -> tt_inverse) 要求 tt_inverse 的 mesh/data 已填写"
            )
        return [
            ("gen_smesh", {"kwargs": gen_kwargs}),
            ("gen_damp", {"kwargs": damp_kwargs}),
            ("tt_inverse", {"mesh": mesh, "data": data, "kwargs": inv_kwargs}),
        ]
    if recipe == "gen_smesh -> gen_vcorr -> tt_inverse":
        gen_kwargs = collect_gen_smesh_kwargs(state)
        vcorr_kwargs = collect_gen_vcorr_kwargs(state)
        mesh, data, inv_kwargs = collect_tt_inverse_args(state)
        payload = auto_wire_pipeline_targets(
            recipe,
            {"mesh": mesh, "data": data, "inv_kwargs": inv_kwargs},
            state,
        )
        mesh = payload["mesh"]
        data = payload["data"]
        inv_kwargs = payload["inv_kwargs"]
        if not mesh or not data:
            raise ValueError(
                "pipeline(gen_smesh -> gen_vcorr -> tt_inverse) 要求 tt_inverse 的 mesh/data 已填写"
            )
        if str(vcorr_kwargs.get("mode") or "") == "simple_2x2" and not str(
            vcorr_kwargs.get("out_file") or ""
        ).strip():
            raise ValueError("pipeline(…→ gen_vcorr) 要求填写 vcorr 输出文件")
        return [
            ("gen_smesh", {"kwargs": gen_kwargs}),
            ("gen_vcorr", {"kwargs": vcorr_kwargs}),
            ("tt_inverse", {"mesh": mesh, "data": data, "kwargs": inv_kwargs}),
        ]
    if recipe == "gen_smesh -> gen_dcorr -> tt_inverse":
        gen_kwargs = collect_gen_smesh_kwargs(state)
        dcorr_kwargs = collect_gen_dcorr_kwargs(state)
        mesh, data, inv_kwargs = collect_tt_inverse_args(state)
        payload = auto_wire_pipeline_targets(
            recipe,
            {"mesh": mesh, "data": data, "inv_kwargs": inv_kwargs},
            state,
        )
        mesh = payload["mesh"]
        data = payload["data"]
        inv_kwargs = payload["inv_kwargs"]
        if not mesh or not data:
            raise ValueError(
                "pipeline(gen_smesh -> gen_dcorr -> tt_inverse) 要求 tt_inverse 的 mesh/data 已填写"
            )
        if not str(dcorr_kwargs.get("out_file") or "").strip():
            raise ValueError("pipeline(…→ gen_dcorr) 要求填写 dcorr 输出文件")
        return [
            ("gen_smesh", {"kwargs": gen_kwargs}),
            ("gen_dcorr", {"kwargs": dcorr_kwargs}),
            ("tt_inverse", {"mesh": mesh, "data": data, "kwargs": inv_kwargs}),
        ]
    if recipe == "gen_smesh -> tt_forward -> tt_inverse":
        gen_kwargs = collect_gen_smesh_kwargs(state)
        smesh, geom, fwd_kwargs = collect_tt_forward_args(state)
        mesh, data, inv_kwargs = collect_tt_inverse_args(state)
        payload = auto_wire_pipeline_targets(
            recipe,
            {
                "smesh": smesh,
                "geom": geom,
                "fwd_kwargs": fwd_kwargs,
                "mesh": mesh,
                "data": data,
                "inv_kwargs": inv_kwargs,
            },
            state,
        )
        smesh = payload["smesh"]
        geom = payload["geom"]
        fwd_kwargs = payload["fwd_kwargs"]
        mesh = payload["mesh"]
        data = payload["data"]
        inv_kwargs = payload["inv_kwargs"]
        if not smesh:
            raise ValueError(
                "pipeline(gen_smesh -> tt_forward -> tt_inverse) 要求 smesh 已填写"
            )
        if not data and not state.get_str("fwd.out_ttime"):
            raise ValueError(
                "pipeline(…→ tt_inverse) 要求填写 fwd.out_ttime（将作为 inv.data）或 inv.data"
            )
        if not data:
            data = state.get_str("fwd.out_ttime")
            state.set("inv.data", data)
        if not mesh:
            mesh = smesh
            state.set("inv.mesh", mesh)
        return [
            ("gen_smesh", {"kwargs": gen_kwargs}),
            ("tt_forward", {"smesh": smesh, "geom": geom, "kwargs": fwd_kwargs}),
            ("tt_inverse", {"mesh": mesh, "data": data, "kwargs": inv_kwargs}),
        ]
    raise ValueError(f"未知 pipeline recipe: {recipe}")


def resolve_pipeline_step_cmdline(
    tomo: TomoAnd,
    step_name: str,
    spec: dict,
    *,
    use_repro_bundle: bool = True,
) -> Any:
    if step_name == "gen_smesh":
        return tomo.resolve_cmdline_gen_smesh(**spec["kwargs"])
    if step_name == "gen_damp":
        return tomo.resolve_cmdline_gen_damp(**spec["kwargs"])
    if step_name == "gen_vcorr":
        return tomo.resolve_cmdline_gen_vcorr(**spec["kwargs"])
    if step_name == "gen_dcorr":
        return tomo.resolve_cmdline_gen_dcorr(**spec["kwargs"])
    if step_name == "tt_forward":
        return tomo.resolve_cmdline_tt_forward(
            smesh=spec["smesh"], geom=spec.get("geom"), **spec["kwargs"]
        )
    if step_name == "tt_inverse":
        m, d = spec["mesh"], spec["data"]
        kw = dict(spec["kwargs"])
        if use_repro_bundle:
            try:
                from ...tt_inverse_bundle import bundle_argv_preview_paths
            except ImportError:
                from pyAOBS.modeling.tomo2d.tt_inverse_bundle import (
                    bundle_argv_preview_paths,
                )
            m, d, kw = bundle_argv_preview_paths(m, d, kw)
        return tomo.resolve_cmdline_tt_inverse(mesh=m, data=d, **kw)
    return None
