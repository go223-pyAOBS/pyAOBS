"""Tk `rock_scatter.add_polygon_samples_*` 多边形均匀采样几何（独立于 GUI）。"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from matplotlib.path import Path as MplPath


def polygon_area_xy(vertices: Sequence[Tuple[float, float]]) -> float:
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0


def uniform_polygon_interior_samples_xy(
    polygon: Sequence[Sequence[float]],
    n_samples: int,
    *,
    rng: np.random.Generator | None = None,
) -> List[Tuple[float, float]]:
    """
    与 imodel_gui/rock_scatter.add_polygon_samples_to_scatter 相同：
    bbox 网格 + 多边形内点 + 最远点贪心 + 缺口随机补足。
    """
    if rng is None:
        rng = np.random.default_rng()

    poly_xy = [(float(p[0]), float(p[1])) for p in polygon]
    if len(poly_xy) < 3:
        return []

    poly_x = [p[0] for p in poly_xy]
    poly_z = [p[1] for p in poly_xy]
    x_min, x_max = min(poly_x), max(poly_x)
    z_min, z_max = min(poly_z), max(poly_z)
    poly_path = MplPath(poly_xy)

    poly_area = polygon_area_xy(poly_xy)
    x_range = x_max - x_min
    z_range = z_max - z_min

    if poly_area > 0:
        target_grid_points = n_samples * 5
        grid_size = float(np.sqrt(poly_area / target_grid_points))
        grid_size = max(grid_size, min(x_range, z_range) / 50.0)
    else:
        grid_size = min(x_range, z_range) / 20.0

    n_x = max(int(x_range / grid_size) + 1, 10)
    n_z = max(int(z_range / grid_size) + 1, 10)
    x_grid = np.linspace(x_min, x_max, n_x)
    z_grid = np.linspace(z_min, z_max, n_z)

    grid_points: List[Tuple[float, float]] = []
    for x in x_grid:
        for z in z_grid:
            if poly_path.contains_point((float(x), float(z))):
                grid_points.append((float(x), float(z)))

    if len(grid_points) >= n_samples:
        selected_points: List[Tuple[float, float]] = []
        remaining_points = grid_points.copy()
        if remaining_points:
            idx = int(rng.integers(0, len(remaining_points)))
            selected_points.append(remaining_points.pop(idx))

        while len(selected_points) < n_samples and remaining_points:
            max_min_dist = -1.0
            best_point: Tuple[float, float] | None = None
            best_idx = -1
            for idx, candidate in enumerate(remaining_points):
                min_dist = float("inf")
                for sx, sz in selected_points:
                    d = np.hypot(candidate[0] - sx, candidate[1] - sz)
                    min_dist = min(min_dist, d)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_point = candidate
                    best_idx = idx

            if best_point is not None:
                selected_points.append(best_point)
                remaining_points.pop(best_idx)
            else:
                break

        if len(selected_points) < n_samples and remaining_points:
            n_needed = n_samples - len(selected_points)
            pick = rng.choice(
                len(remaining_points),
                size=min(n_needed, len(remaining_points)),
                replace=False,
            )
            for idx in pick:
                selected_points.append(remaining_points[int(idx)])
        selected_points_final = selected_points[:n_samples]
    else:
        selected_points_final = grid_points.copy()
        remaining_need = n_samples - len(selected_points_final)
        max_attempts = remaining_need * 200 if remaining_need > 0 else 0
        attempts = 0
        while len(selected_points_final) < n_samples and attempts < max_attempts:
            attempts += 1
            x = float(rng.uniform(x_min, x_max))
            z = float(rng.uniform(z_min, z_max))
            if poly_path.contains_point((x, z)):
                too_close = False
                for ex_x, ex_z in selected_points_final:
                    dist = float(np.hypot(x - ex_x, z - ex_z))
                    if dist < grid_size * 0.5:
                        too_close = True
                        break
                if not too_close:
                    selected_points_final.append((x, z))

    return selected_points_final[:n_samples]
