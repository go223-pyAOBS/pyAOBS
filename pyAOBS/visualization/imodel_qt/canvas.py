"""在 Matplotlib Axes 上绘制速度网格（pcolormesh）。"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from pyAOBS.visualization.imodel import ProfileExtractor


def redraw_velocity_axes(
    ax: Axes,
    grid_data: xr.Dataset,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple[object, str, float, float]:
    ax.clear()

    extractor = ProfileExtractor(grid_data)
    vel_name = extractor.velocity_var
    velocity = grid_data[vel_name].values
    x_coords = grid_data.coords[extractor.x_coord].values
    z_coords = grid_data.coords[extractor.z_coord].values

    if vmin is None:
        vmin = float(np.nanmin(velocity))
    if vmax is None:
        vmax = float(np.nanmax(velocity))

    im = ax.pcolormesh(
        x_coords,
        z_coords,
        velocity,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_title("Velocity Model")
    ax.invert_yaxis()
    return im, vel_name, vmin, vmax
