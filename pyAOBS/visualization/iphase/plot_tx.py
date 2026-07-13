"""
tx.in 数据可视化

偏移-到时散点图、相位覆盖图等。
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .models import PhaseDataset

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _get_arrays(dataset: PhaseDataset, phase_ids: Optional[Sequence[int]] = None):
    """提取 x, t, phase 数组供绘图使用。"""
    xs, ts, phases = [], [], []
    for s in dataset.shots:
        for p in s.picks:
            if phase_ids is None or p.phase_id in phase_ids:
                xs.append(p.x)
                ts.append(p.t)
                phases.append(p.phase_id)
    return np.array(xs), np.array(ts), np.array(phases)


def plot_offset_time(
    dataset: PhaseDataset,
    phase_ids: Optional[Sequence[int]] = None,
    ax=None,
    **kwargs,
):
    """
    绘制偏移-到时散点图，按相位着色。

    Parameters
    ----------
    dataset : PhaseDataset
    phase_ids : sequence of int, optional
        若指定，仅绘制这些相位；否则绘制全部。
    ax : matplotlib Axes, optional
    **kwargs : 传递给 ax.scatter
    """
    if not HAS_MPL:
        raise ImportError("需要安装 matplotlib")

    xs, ts, phases = _get_arrays(dataset, phase_ids)
    if xs.size == 0:
        if ax is None:
            _, ax = plt.subplots()
        ax.set_xlabel("Model distance (km)")
        ax.set_ylabel("Time (s)")
        return ax

    if ax is None:
        _, ax = plt.subplots()

    unique = np.unique(phases)
    for i, pid in enumerate(unique):
        m = phases == pid
        c = plt.cm.tab10((i % 10) / 9.0)
        ax.scatter(xs[m], ts[m], label=f"Phase {pid}", c=[c], **kwargs)
    ax.set_xlabel("Model distance (km)")
    ax.set_ylabel("Time (s)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.5)
    return ax


def plot_phase_coverage(
    dataset: PhaseDataset,
    phase_ids: Optional[Sequence[int]] = None,
    ax=None,
):
    """绘制各相位在偏移上的分布（直方图或密度）。"""
    if not HAS_MPL:
        raise ImportError("需要安装 matplotlib")

    xs, _, phases = _get_arrays(dataset, phase_ids)
    if xs.size == 0:
        if ax is None:
            _, ax = plt.subplots()
        return ax

    if ax is None:
        _, ax = plt.subplots()

    unique = np.unique(phases)
    for i, pid in enumerate(unique):
        m = phases == pid
        c = plt.cm.tab10((i % 10) / 9.0)
        ax.hist(xs[m], bins=min(30, max(5, m.sum() // 2)), alpha=0.6, label=f"Phase {pid}", color=c)
    ax.set_xlabel("Model distance (km)")
    ax.set_ylabel("Count")
    ax.legend(loc="best")
    return ax


def plot_difference_offset(
    pairs: Sequence[tuple[float, ...]],
    *,
    x_axis: str = "true_offset",
    ax=None,
    **kwargs,
):
    """
    绘制 PPP-PPS 差值-偏移距曲线。

    Parameters
    ----------
    pairs : sequence of (model_dist, true_offset, t_diff)
        model_dist: x (km)，模型距离
        true_offset: x - xshot (km)，有符号真实偏移距
        t_diff: t_PPS - t_PPP (s)
    x_axis : "true_offset" | "model_distance"
        X 轴用真实偏移距还是模型距离
    ax : matplotlib Axes, optional
    **kwargs : 传递给 ax.scatter 或 ax.plot
    """
    if not HAS_MPL:
        raise ImportError("需要安装 matplotlib")

    if not pairs:
        if ax is None:
            _, ax = plt.subplots()
        ax.set_xlabel("Model distance (km)" if x_axis == "model_distance" else "True offset (km), x - xshot")
        ax.set_ylabel("PPS-PPP (s)")
        return ax

    x_idx = 0 if x_axis == "model_distance" else 1
    offsets = [p[x_idx] for p in pairs]
    diffs = [p[2] for p in pairs]
    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(offsets, diffs, **kwargs)
    so = sorted(zip(offsets, diffs))
    if so:
        ox, oy = zip(*so)
        ax.plot(ox, oy, "b-", alpha=0.5, linewidth=1)
    ax.set_xlabel("Model distance (km)" if x_axis == "model_distance" else "True offset (km), x - xshot")
    ax.set_ylabel("PPS-PPP (s)")
    ax.grid(True, alpha=0.5)
    return ax


def plot_difference_offset_with_fit(
    pairs: Sequence[tuple[float, ...]],
    fit_stats: dict[str, Any],
    *,
    x_axis: str = "true_offset",
    ax=None,
):
    """
    绘制 PPP-PPS 差值曲线并叠加拟合线；可选绘制残差分布。

    fit_stats: stats_ppp_pps_diff_by_offset 的返回值。
    """
    if not HAS_MPL:
        raise ImportError("需要安装 matplotlib")

    x_idx = 0 if x_axis == "model_distance" else 1
    if not pairs:
        if ax is None:
            _, ax = plt.subplots()
        ax.set_xlabel("Model distance (km)" if x_axis == "model_distance" else "True offset (km), x - xshot")
        ax.set_ylabel("PPS-PPP (s)")
        return ax

    offsets = [p[x_idx] for p in pairs]
    diffs = [p[2] for p in pairs]
    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(offsets, diffs, c="gray", alpha=0.7, label="data")
    for label, stats in [("positive", fit_stats.get("positive")), ("negative", fit_stats.get("negative"))]:
        if stats is None:
            continue
        ox = stats.offsets
        oy_fit = stats.fitted
        so = sorted(zip(ox, oy_fit))
        if so:
            xp, yp = zip(*so)
            ax.plot(xp, yp, "-", linewidth=2, label=f"{label} fit (n={stats.n}, rms={stats.rms:.4f}s)")
    ax.set_xlabel("Model distance (km)" if x_axis == "model_distance" else "True offset (km), x - xshot")
    ax.set_ylabel("PPS-PPP (s)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.5)
    return ax


def plot_residuals_histogram(
    fit_stats: dict[str, Any],
    *,
    ax=None,
):
    """绘制正/负偏移距拟合残差直方图。"""
    if not HAS_MPL:
        raise ImportError("需要安装 matplotlib")
    if ax is None:
        _, ax = plt.subplots()

    for label, color in [("positive", "C0"), ("negative", "C1")]:
        s = fit_stats.get(label)
        if s is None or len(s.residuals) == 0:
            continue
        ax.hist(s.residuals, bins=min(20, max(5, len(s.residuals) // 2)), alpha=0.6, label=f"{label} (std={s.std_residual:.4f}s)", color=color)
    ax.set_xlabel("Residual (s), obs - fit")
    ax.set_ylabel("Count")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.5)
    return ax
