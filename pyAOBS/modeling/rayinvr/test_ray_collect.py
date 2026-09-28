# -*- coding: utf-8 -*-
"""``ray_collect`` 单元测试。"""

from __future__ import annotations

from pyAOBS.modeling.rayinvr.ray_collect import (
    infer_ray_shot_x,
    limit_rays_per_shot,
    missing_shot_xs_for_rays,
    shot_xs_with_rays,
)
from pyAOBS.modeling.vedit.core.phase_filter import (
    ensure_rays_per_shot,
    filter_rays_by_shot,
)


def _ray(x0: float, x1: float, *, phase_id: int = 5) -> dict:
    return {
        "npoints": 2,
        "x": [x0, x1],
        "z": [2.64, 3.0],
        "phase_id": phase_id,
    }


def test_infer_ray_shot_x_receiver_first_obs53():
    """OBS53 长偏移：路径首点为接收端，末点为炮点。"""
    shot_xs = [102.86, 96.859, 84.86]
    ray = _ray(28.5, 102.86)
    assert infer_ray_shot_x(ray, shot_xs) == 102.86


def test_filter_rays_by_shot_uses_endpoints():
    rays = [
        _ray(28.5, 102.86),
        _ray(96.859, 40.0),
        _ray(10.0, 20.0),
    ]
    out = filter_rays_by_shot(rays, shot_xs={102.86, 96.859})
    assert len(out) == 2


def test_limit_rays_per_shot_receiver_first():
    rays = [_ray(30.0 + i * 0.1, 102.86, phase_id=5) for i in range(20)]
    shot_xs = [102.86, 96.859]
    out = limit_rays_per_shot(rays, shot_xs, max_per_shot=5)
    assert len(out) == 5
    assert shot_xs_with_rays(out, shot_xs) == {102.86}


def test_limit_rays_per_shot_stratified_by_phase():
    from pyAOBS.modeling.rayinvr.ray_collect import (
        RAYS_PER_SHOT_MIN_PER_PHASE,
        limit_rays_per_shot_stratified,
    )

    rays = [_ray(30.0 + i * 0.1, 288.735, phase_id=5) for i in range(30)]
    rays += [_ray(40.0 + i * 0.1, 288.735, phase_id=8) for i in range(30)]
    out = limit_rays_per_shot_stratified(
        rays,
        [288.735],
        max_per_shot=10,
        min_per_phase=RAYS_PER_SHOT_MIN_PER_PHASE,
    )
    phases = {int(r.get("phase_id") or 0) for r in out}
    assert 5 in phases and 8 in phases, f"missing phase in display sample: {phases}"
    n5 = sum(1 for r in out if int(r.get("phase_id") or 0) == 5)
    n8 = sum(1 for r in out if int(r.get("phase_id") or 0) == 8)
    assert n5 >= RAYS_PER_SHOT_MIN_PER_PHASE
    assert n8 >= RAYS_PER_SHOT_MIN_PER_PHASE


def test_ensure_rays_per_shot_backfill_after_phase_filter():
    rays = [
        _ray(28.5, 102.86, phase_id=3),
        _ray(28.5, 102.86, phase_id=5),
        _ray(96.859, 40.0, phase_id=5),
    ]
    shot_xs = [102.86, 96.859]

    def _phase(rays, *, phase_ids):
        return [r for r in rays if int(r.get("phase_id") or 0) in phase_ids]

    out = ensure_rays_per_shot(
        rays,
        shot_xs,
        phase_ids={99},
        filter_phase_fn=_phase,
        filter_shot_fn=filter_rays_by_shot,
    )
    assert missing_shot_xs_for_rays(out, shot_xs) == []


if __name__ == "__main__":
    test_infer_ray_shot_x_receiver_first_obs53()
    test_filter_rays_by_shot_uses_endpoints()
    test_limit_rays_per_shot_receiver_first()
    test_limit_rays_per_shot_stratified_by_phase()
    test_ensure_rays_per_shot_backfill_after_phase_filter()
    print("ok")
