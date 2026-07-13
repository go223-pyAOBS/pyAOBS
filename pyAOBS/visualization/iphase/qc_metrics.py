"""
iphase 统计与质量控制

每相位拾取数、每炮拾取数、偏移-到时分布、PPP-PPS 差值拟合与偏离误差等。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from .models import PhaseDataset


@dataclass
class DiffFitStats:
    """正/负偏移距 PPP-PPS 差值拟合统计。"""

    n: int
    offsets: np.ndarray
    t_diff: np.ndarray
    coeffs: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    mean_residual: float
    std_residual: float
    rms: float


def stats_ppp_pps_diff_by_offset(
    pairs: Sequence[tuple[float, float, float]],
    *,
    degree: int = 2,
) -> Dict[str, Any]:
    """
    对正、负偏移距的 PPP-PPS 差值分别统计：拟合曲线、偏离误差。

    pairs: (model_dist, true_offset, t_diff)
    按 true_offset 正负分组，对每组用多项式拟合 t_diff(offset)，计算残差统计。

    Returns:
        {
            "positive": DiffFitStats or None,
            "negative": DiffFitStats or None,
        }
    """
    pos_x, pos_y = [], []
    neg_x, neg_y = [], []
    for md, to, td in pairs:
        if to > 0:
            pos_x.append(to)
            pos_y.append(td)
        elif to < 0:
            neg_x.append(to)
            neg_y.append(td)

    out: Dict[str, Any] = {"positive": None, "negative": None}

    for key, xs, ys in [
        ("positive", pos_x, pos_y),
        ("negative", neg_x, neg_y),
    ]:
        if len(xs) < degree + 2:
            continue
        ox = np.array(xs)
        oy = np.array(ys)
        coeffs = np.polyfit(ox, oy, degree)
        fitted = np.polyval(coeffs, ox)
        residuals = oy - fitted
        mean_r = float(np.mean(residuals))
        std_r = float(np.std(residuals))
        rms = float(np.sqrt(np.mean(residuals**2)))
        out[key] = DiffFitStats(
            n=len(xs),
            offsets=ox,
            t_diff=oy,
            coeffs=coeffs,
            fitted=fitted,
            residuals=residuals,
            mean_residual=mean_r,
            std_residual=std_r,
            rms=rms,
        )
    return out


def phase_counts(dataset: PhaseDataset) -> Dict[int, int]:
    """各相位的拾取数量。"""
    counts: Dict[int, int] = {}
    for s in dataset.shots:
        for p in s.picks:
            counts[p.phase_id] = counts.get(p.phase_id, 0) + 1
    return counts


def shot_counts(dataset: PhaseDataset) -> List[int]:
    """各炮的拾取数量。"""
    return [len(s.picks) for s in dataset.shots]


def summary(dataset: PhaseDataset) -> Dict:
    """汇总统计。"""
    return {
        "n_shots": dataset.n_shots,
        "n_picks": dataset.n_picks,
        "phase_counts": phase_counts(dataset),
        "phase_ids": dataset.phase_ids(),
    }
