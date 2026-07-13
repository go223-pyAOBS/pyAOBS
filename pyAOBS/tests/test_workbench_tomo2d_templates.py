from pyAOBS.workbench.plugins.tomo2d_templates import (
    TEMPLATE_FORWARD_BASIC,
    TEMPLATE_INVERSE_STANDARD,
    build_tomo2d_template_payload,
)


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
    assert "work/inputs/mesh.dat" in payload["inputs_text"]


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

