import pytest

from pyAOBS.workbench.plugins.tomo2d_templates import (
    TEMPLATE_FORWARD_BASIC,
    TEMPLATE_INVERSE_STANDARD,
    build_tomo2d_template_payload,
)


def _slash(text: str) -> str:
    return text.replace("\\", "/")


def test_build_inverse_template_payload() -> None:
    payload = build_tomo2d_template_payload(
        TEMPLATE_INVERSE_STANDARD,
        {
            "mesh_path": "inputs/mesh.dat",
            "data_path": "inputs/data.dat",
            "iterations": "8",
            "sv": "120",
            "sd": "15",
            "dv": "1",
            "dd": "30",
            "verbose": "-1",
            "extra_args": "-N4/4/0.8/8/0.0001/1e-05",
            "work_dir": "work",
            "node_id": "inv_job",
        },
    )
    assert payload["executable"] == "tt_inverse"
    assert "-I8" in payload["args"]
    assert "-SV120" in payload["args"]
    assert "-N4/4/0.8/8/0.0001/1e-05" in payload["args"]
    assert payload["cwd"] == "work"
    assert payload["node_id"] == "inv_job"
    assert "work/inputs/mesh.dat" in _slash(payload["inputs_text"])
    assert "-Y" not in payload["args"]
    assert " -y" not in payload["args"]
    assert " -w" not in payload["args"]


def test_build_forward_template_payload() -> None:
    payload = build_tomo2d_template_payload(
        TEMPLATE_FORWARD_BASIC,
        {
            "mesh_path": "inputs/mesh.dat",
            "data_path": "inputs/data.dat",
            "verbose": "0",
            "work_dir": "job1",
            "node_id": "fwd_job",
        },
    )
    assert payload["executable"] == "tt_forward"
    assert "-V0" in payload["args"]
    assert payload["cwd"] == "job1"
    assert payload["node_id"] == "fwd_job"
    assert "-B" not in payload["args"]


def test_build_inverse_template_with_seafloor_and_water_only() -> None:
    payload = build_tomo2d_template_payload(
        TEMPLATE_INVERSE_STANDARD,
        {
            "mesh_path": "inputs/mesh.dat",
            "data_path": "inputs/data.dat",
            "seafloor_path": "inputs/sf.dat",
            "invert_water_only": "1",
            "work_dir": "work",
            "node_id": "inv_job",
        },
    )
    assert "-Yinputs/sf.dat" in _slash(payload["args"])
    assert " -y" in payload["args"]
    assert "-B" not in payload["args"]
    assert "work/inputs/sf.dat" in _slash(payload["inputs_text"])


def test_build_forward_template_with_seafloor_and_refl() -> None:
    payload = build_tomo2d_template_payload(
        TEMPLATE_FORWARD_BASIC,
        {
            "mesh_path": "inputs/mesh.dat",
            "data_path": "inputs/data.dat",
            "seafloor_path": "inputs/sf.dat",
            "refl_path": "inputs/moho.dat",
            "invert_water_only": "1",
            "work_dir": "job1",
            "node_id": "fwd_job",
        },
    )
    args = _slash(payload["args"])
    inputs = _slash(payload["inputs_text"])
    assert "-Binputs/sf.dat" in args
    assert "-Finputs/moho.dat" in args
    assert " -y" not in payload["args"]
    assert "job1/inputs/sf.dat" in inputs
    assert "job1/inputs/moho.dat" in inputs


def test_build_inverse_template_freeze_refl() -> None:
    payload = build_tomo2d_template_payload(
        TEMPLATE_INVERSE_STANDARD,
        {
            "mesh_path": "inputs/mesh.dat",
            "data_path": "inputs/data.dat",
            "refl_path": "inputs/moho.dat",
            "freeze_refl": "1",
            "invert_crust_only": "true",
            "work_dir": "work",
        },
    )
    assert "-Finputs/moho.dat" in _slash(payload["args"])
    assert " -u" in payload["args"]
    assert " -w" in payload["args"]


def test_build_inverse_rejects_y_and_w() -> None:
    with pytest.raises(ValueError, match="-y 与 -w"):
        build_tomo2d_template_payload(
            TEMPLATE_INVERSE_STANDARD,
            {
                "mesh_path": "inputs/mesh.dat",
                "data_path": "inputs/data.dat",
                "seafloor_path": "inputs/sf.dat",
                "invert_water_only": "1",
                "invert_crust_only": "1",
            },
        )


def test_build_inverse_water_only_needs_interface() -> None:
    with pytest.raises(ValueError, match="-y / -w"):
        build_tomo2d_template_payload(
            TEMPLATE_INVERSE_STANDARD,
            {
                "mesh_path": "inputs/mesh.dat",
                "data_path": "inputs/data.dat",
                "invert_water_only": "1",
            },
        )


def test_build_inverse_freeze_needs_refl() -> None:
    with pytest.raises(ValueError, match="-u"):
        build_tomo2d_template_payload(
            TEMPLATE_INVERSE_STANDARD,
            {
                "mesh_path": "inputs/mesh.dat",
                "data_path": "inputs/data.dat",
                "freeze_refl": "1",
            },
        )
