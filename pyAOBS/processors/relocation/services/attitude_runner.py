"""调用联合姿态校正；要求有效水深采样。"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from ..bathymetry_sampler import BathymetrySampler
from ..orientation_correction import (
    OrientationCorrectionInput,
    OrientationCorrectionResult,
    OrientationObservation,
    run_orientation_correction,
)
from .depth_sampler import sample_initial_depth_km
from .models import AttitudeSolution, AttitudeUiParams

ProgressCallback = Callable[[int, int, str], None]
DepthSampler = Callable[[float, float], Optional[float]]


class AttitudeRunner:
    """无 Qt 的姿态校正执行器。"""

    def __init__(
        self,
        ui_params: Optional[AttitudeUiParams] = None,
        initial_solution: Optional[AttitudeSolution] = None,
    ):
        self.ui_params = ui_params or AttitudeUiParams()
        self.initial_solution = initial_solution or AttitudeSolution()

    def run(
        self,
        observations: List[OrientationObservation],
        depth_sampler: Optional[DepthSampler],
        *,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Tuple[Optional[OrientationCorrectionResult], str]:
        if not observations:
            return None, "观测列表为空"
        if depth_sampler is None:
            return None, "未检测到可用水深采样（请先加载水深文件），无法执行姿态校正。"

        depth0 = sample_initial_depth_km(observations, depth_sampler)
        if depth0 is None or not np.isfinite(depth0) or depth0 <= 0.0:
            return None, "当前地形无法采样有效水深，无法执行姿态校正。"

        sol = self.initial_solution
        # 未勾选校正倾角：强制 tilt 初值=0，禁止沿用旧解非零 tilt（否则预览会把 Z 与 R 混叠）
        tilt0 = float(sol.tilt_deg) if bool(self.ui_params.correct_tilt) else 0.0
        result = run_orientation_correction(
            OrientationCorrectionInput(
                observations=observations,
                initial_azimuth_deg=float(sol.azimuth_deg),
                initial_tilt_deg=float(tilt0),
                initial_position_correction=sol.position_correction(),
                initial_time_shift_sec=float(self.ui_params.prior_tt_shift_sec),
                depth_sampler=depth_sampler,
                max_iterations=max(1, int(self.ui_params.att_iter)),
                w_tt=max(0.0, float(self.ui_params.att_wtt)),
                w_pol=max(0.0, float(self.ui_params.att_wpol)),
                w_sym=max(0.0, float(self.ui_params.att_wsym)),
                correct_tilt=bool(self.ui_params.correct_tilt),
                progress_callback=progress_callback,
            )
        )
        if not result.success:
            return result, result.message or "姿态校正失败"
        return result, ""

    @staticmethod
    def result_to_solution(result: OrientationCorrectionResult) -> AttitudeSolution:
        dx, dy, dz = result.position_correction
        details = getattr(result, "details", {}) or {}
        prior = float(details.get("prior_time_shift_sec", 0.0))
        final = float(details.get("time_shift_sec", 0.0))
        corr = float(details.get("tt_corr_sec", final - prior))
        return AttitudeSolution(
            azimuth_deg=float(result.azimuth_deg),
            tilt_deg=float(result.tilt_deg),
            dx=float(dx),
            dy=float(dy),
            dz=float(dz),
            prior_tt_shift_sec=prior,
            tt_corr_sec=corr,
            time_shift_sec=final,
        )

    @staticmethod
    def make_depth_sampler_from_bathymetry(
        sampler: BathymetrySampler,
    ) -> DepthSampler:
        def _fn(x: float, y: float) -> Optional[float]:
            try:
                return sampler(float(x), float(y))
            except Exception:
                return None

        return _fn
