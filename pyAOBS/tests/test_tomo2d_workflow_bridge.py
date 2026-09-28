"""正反演衔接：写回 / 上游填充 / pipeline 配方。"""
from __future__ import annotations

import pytest

from pyAOBS.modeling.tomo2d.gui.services.pipeline import build_pipeline_plan
from pyAOBS.modeling.tomo2d.gui.services.workflow_bridge import (
    apply_fwd_outputs_to_inv,
    fill_inv_from_upstream,
    format_monitor_checklist,
    sync_ray_params,
)
from pyAOBS.modeling.tomo2d.gui.state.form_state import FormState

pytestmark = pytest.mark.unit


def test_apply_fwd_outputs_to_inv_empty_targets() -> None:
    st = FormState(
        {
            "fwd.smesh": "outputs/m.smesh",
            "fwd.out_ttime": "outputs/tt.dat",
            "fwd.xorder": "5",
            "inv.xorder": "4",
            "inv.mesh": "",
            "inv.data": "",
        }
    )
    notes = apply_fwd_outputs_to_inv(st, overwrite=False, sync_ray=True)
    assert st.get_str("inv.mesh") == "outputs/m.smesh"
    assert st.get_str("inv.data") == "outputs/tt.dat"
    assert st.get_str("inv.xorder") == "5"
    assert any("inv.mesh" in n for n in notes)


def test_apply_fwd_syncs_seafloor() -> None:
    st = FormState(
        {
            "fwd.smesh": "a.smesh",
            "fwd.out_ttime": "a.dat",
            "fwd.refl_file": "moho.dat",
            "fwd.seafloor_file": "sf.dat",
            "inv.mesh": "",
            "inv.data": "",
            "inv.refl_file": "",
            "inv.seafloor_file": "",
        }
    )
    notes = apply_fwd_outputs_to_inv(st, overwrite=False, sync_ray=False, sync_refl=True)
    assert st.get_str("inv.refl_file") == "moho.dat"
    assert st.get_str("inv.seafloor_file") == "sf.dat"
    assert any("seafloor" in n for n in notes)


def test_apply_fwd_syncs_conv_and_vsmesh() -> None:
    st = FormState(
        {
            "fwd.smesh": "a.smesh",
            "fwd.out_ttime": "a.dat",
            "fwd.conv_file": "conv.dat",
            "fwd.vsmesh": "vs.smesh",
            "fwd.kappa": "1.73",
            "inv.mesh": "",
            "inv.data": "",
            "inv.conv_file": "",
            "inv.vsmesh": "",
            "inv.kappa": "",
        }
    )
    notes = apply_fwd_outputs_to_inv(st, overwrite=False, sync_ray=False, sync_refl=True)
    assert st.get_str("inv.conv_file") == "conv.dat"
    assert st.get_str("inv.vsmesh") == "vs.smesh"
    assert st.get_str("inv.kappa") == "1.73"
    assert any("conv" in n for n in notes)


def test_fill_inv_from_upstream_seafloor() -> None:
    st = FormState(
        {
            "fwd.seafloor_file": "sf.dat",
            "fwd.refl_file": "moho.dat",
            "inv.seafloor_file": "",
            "inv.refl_file": "",
        }
    )
    fill_inv_from_upstream(st, overwrite=False)
    assert st.get_str("inv.seafloor_file") == "sf.dat"
    assert st.get_str("inv.refl_file") == "moho.dat"


def test_fill_inv_from_gen_seafloor_out() -> None:
    st = FormState(
        {
            "gen.seafloor_out": "outputs/sf.dat",
            "fwd.seafloor_file": "",
            "inv.seafloor_file": "",
        }
    )
    fill_inv_from_upstream(st, overwrite=False)
    assert st.get_str("fwd.seafloor_file") == "outputs/sf.dat"
    assert st.get_str("inv.seafloor_file") == "outputs/sf.dat"


def test_apply_fwd_does_not_overwrite() -> None:
    st = FormState(
        {
            "fwd.smesh": "a.smesh",
            "fwd.out_ttime": "a.dat",
            "inv.mesh": "keep.smesh",
            "inv.data": "keep.dat",
        }
    )
    apply_fwd_outputs_to_inv(st, overwrite=False, sync_ray=False, sync_refl=False)
    assert st.get_str("inv.mesh") == "keep.smesh"
    assert st.get_str("inv.data") == "keep.dat"


def test_fill_and_sync_and_checklist() -> None:
    st = FormState(
        {
            "gen.smesh_out": "outputs/g.smesh",
            "pipe.link_damp": "outputs/damp.dat",
            "fwd.out_ttime": "outputs/tt.dat",
            "fwd.clen": "1.2",
            "inv.use_repro_bundle": True,
            "inv.print_final_only": True,
            "inv.mesh": "outputs/g.smesh",
            "inv.data": "outputs/tt.dat",
        }
    )
    notes = fill_inv_from_upstream(st, overwrite=False)
    assert st.get_str("inv.damp_v_fn") == "outputs/damp.dat"
    assert any("damp" in n for n in notes)
    sync_ray_params(st, direction="fwd_to_inv", overwrite=True)
    assert st.get_str("inv.clen") == "1.2"
    text = format_monitor_checklist(st)
    assert "mesh 已填" in text
    assert "少写 smesh" in text


def test_pipeline_fwd_inv_recipe() -> None:
    st = FormState(
        {
            "pipe.recipe": "gen_smesh -> tt_forward -> tt_inverse",
            "pipe.auto_wire": True,
            "pipe.link_smesh": "outputs/mesh.smesh",
            "gen.smesh_out": "outputs/mesh.smesh",
            "gen.vel_opt": "uniform",
            "gen.grid_opt": "uniform",
            "gen.nx": "10",
            "gen.nz": "10",
            "gen.xmax": "10",
            "gen.zmax": "5",
            "fwd.smesh": "",
            "fwd.geom": "inputs/geom.dat",
            "fwd.out_ttime": "outputs/ttimes.dat",
            "fwd.xorder": "4",
            "fwd.zorder": "4",
            "fwd.clen": "0.8",
            "fwd.nintp": "8",
            "fwd.bend_cg_tol": "1e-4",
            "fwd.bend_br_tol": "1e-5",
            "inv.mesh": "",
            "inv.data": "",
            "inv.niter": "3",
            "inv.xorder": "4",
            "inv.zorder": "4",
            "inv.clen": "0.8",
            "inv.nintp": "8",
            "inv.bend_cg_tol": "1e-4",
            "inv.bend_br_tol": "1e-5",
        }
    )
    plan = build_pipeline_plan(st)
    names = [n for n, _ in plan]
    assert names == ["gen_smesh", "tt_forward", "tt_inverse"]
    assert st.get_str("inv.data") == "outputs/ttimes.dat"
    assert st.get_str("inv.mesh") == "outputs/mesh.smesh"
