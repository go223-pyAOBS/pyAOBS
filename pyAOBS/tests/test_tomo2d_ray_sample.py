"""射线抽样解析（监视用）。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pyAOBS.modeling.tomo2d.gui.services.ray_sample import (
    list_inverse_ray_files,
    load_sampled_rays_for_monitor,
    parse_tomo2d_ray_file,
    sample_ray_file_entries,
)

pytestmark = pytest.mark.unit


def test_parse_tomo2d_ray_file_segments(tmp_path: Path) -> None:
    p = tmp_path / "out.ray.1.0"
    p.write_text(
        ">\n"
        "0 0\n"
        "1 1\n"
        "2 0\n"
        ">\n"
        "10 0\n"
        "11 2\n"
        "12 0\n",
        encoding="utf-8",
    )
    segs = parse_tomo2d_ray_file(p)
    assert len(segs) == 2
    assert segs[0][0] == [0.0, 1.0, 2.0]
    assert segs[1][1][1] == pytest.approx(2.0)


def test_sample_and_list_rays(tmp_path: Path) -> None:
    od = tmp_path / "outputs"
    rays = od / "rays"
    rays.mkdir(parents=True)
    for isrc in range(1, 9):
        (rays / f"out.ray.2.{isrc}").write_text(
            ">\n0 0\n1 1\n", encoding="utf-8"
        )
    (rays / "out.ray.1.1").write_text(">\n0 0\n1 0\n", encoding="utf-8")
    entries = list_inverse_ray_files(od / "out")
    assert len(entries) == 9
    picked = sample_ray_file_entries(entries, max_sources=4, iter_prefer=2)
    assert len(picked) == 4
    assert all(it == 2 for _, it, _ in picked)
    groups, note = load_sampled_rays_for_monitor(od / "out", max_sources=3)
    assert "3/3" in note and "iter=2" in note
    assert len(groups) == 3
    assert {isrc for isrc, _ in groups} <= set(range(1, 9))
    assert all(segs for _, segs in groups)
    from pyAOBS.modeling.tomo2d.gui.services.ray_sample import (
        has_inverse_ray_files,
        obs_ray_color,
    )

    assert has_inverse_ray_files(od / "out") is True
    assert has_inverse_ray_files(od / "missing") is False
    assert obs_ray_color(1) != obs_ray_color(2)


def test_infer_out_root_and_load_for_smesh_plot(tmp_path: Path) -> None:
    from pyAOBS.modeling.tomo2d.gui.services.ray_sample import (
        infer_out_root_from_ray_hint,
        load_rays_for_smesh_plot,
        looks_like_ray_name,
    )

    od = tmp_path / "outputs"
    rays = od / "rays"
    rays.mkdir(parents=True)
    for isrc in range(1, 4):
        (rays / f"out.ray.2.{isrc}").write_text(
            ">\n0 0\n1 1\n2 0\n", encoding="utf-8"
        )
    (rays / "out.ray.1.1").write_text(">\n0 0\n1 0\n", encoding="utf-8")
    one = rays / "out.ray.2.1"
    assert looks_like_ray_name(one)
    assert looks_like_ray_name(rays)
    assert not looks_like_ray_name("dws.dat")
    assert infer_out_root_from_ray_hint(one) == od / "out"
    assert infer_out_root_from_ray_hint(rays) == od / "out"
    groups, note = load_rays_for_smesh_plot(one, iter_prefer=2)
    assert "iter=2" in note
    assert len(groups) == 3
    fwd = tmp_path / "fwd.ray"
    fwd.write_text(">\n0 0\n1 1\n>\n2 0\n3 1\n", encoding="utf-8")
    g2, n2 = load_rays_for_smesh_plot(fwd)
    assert len(g2) == 1 and len(g2[0][1]) == 2
    assert "fwd.ray" in n2
