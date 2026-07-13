"""垂直剖面平均 —— 与 Tk `profiles.ProfileMixin.extract_vertical_profile` 核心语义对齐。"""



from __future__ import annotations



from typing import Callable, Optional



import numpy as np

import pandas as pd

import xarray as xr



from pyAOBS.visualization.imodel import ProfileExtractor





def _interp_vp_to_grid(

    profile_depths: np.ndarray,

    profile_vp: np.ndarray,

    reference_depths: np.ndarray,

) -> np.ndarray:

    """线性插值到 reference_depths；区间外为 nan。"""

    d = np.asarray(profile_depths, dtype=float)

    v = np.asarray(profile_vp, dtype=float)

    m = np.isfinite(d) & np.isfinite(v)

    if np.sum(m) < 2:

        return np.full(reference_depths.shape, np.nan, dtype=float)

    d = d[m]

    v = v[m]

    order = np.argsort(d)

    d = d[order]

    v = v[order]

    uniq = np.concatenate(([True], np.diff(d) > 1e-9))

    d = d[uniq]

    v = v[uniq]

    if len(d) < 2:

        return np.full(reference_depths.shape, np.nan, dtype=float)

    return np.interp(

        np.asarray(reference_depths, dtype=float),

        d,

        v,

        left=np.nan,

        right=np.nan,

    )





def averaged_vertical_profile(

    grid_data: xr.Dataset,

    x_min: float,

    x_max: float,

    dx: float,

    basement_depth_fn: Optional[Callable[[float], float]] = None,

) -> pd.DataFrame:

    """

    在 [x_min, x_max] 上按 dx 取样；每条竖线先可做基底深度校正（depth -= basement），

    再插值到统一深度网格上对 Vp 做平均（对齐 Tk profiles 流程）。

    """

    if x_min >= x_max:

        raise ValueError("x_min 必须小于 x_max")

    if dx <= 0:

        raise ValueError("dx 必须大于 0")



    extractor = ProfileExtractor(grid_data)

    x_coord = extractor.x_coord

    z_coord = extractor.z_coord

    velocity_var = extractor.velocity_var



    xs = np.arange(float(x_min), float(x_max) + 0.5 * float(dx), float(dx))

    if xs.size < 1:

        raise ValueError("X 采样为空")



    bd_fn = basement_depth_fn or (lambda _x: 0.0)



    all_profiles: list[pd.DataFrame] = []



    try:

        velocity_data = grid_data[velocity_var]

        z_coords = np.asarray(grid_data.coords[z_coord].values, dtype=float)



        for x in xs:

            try:

                v_slice = velocity_data.sel({x_coord: float(x)}, method="nearest")

                vp_values = np.asarray(v_slice.values, dtype=float).reshape(-1)

                if vp_values.size != z_coords.size:

                    if vp_values.size == 1:

                        vp_values = np.full(z_coords.size, float(vp_values[0]), dtype=float)

                    elif vp_values.size > z_coords.size:

                        vp_values = vp_values[: z_coords.size]

                    else:

                        vp_values = np.pad(

                            vp_values,

                            (0, z_coords.size - vp_values.size),

                            constant_values=np.nan,

                        )

                profile = pd.DataFrame({"depth": z_coords.copy(), "vp": vp_values})

            except Exception:

                profile = extractor.extract_vertical_profile(float(x))



            basement_depth = float(bd_fn(float(x)))

            if basement_depth > 0:

                profile_depths = np.asarray(profile["depth"].values, dtype=float)

                adjusted = profile_depths - basement_depth

                valid_mask = adjusted >= 0

                if np.any(valid_mask):

                    profile = pd.DataFrame(

                        {

                            "depth": adjusted[valid_mask],

                            "vp": np.asarray(profile["vp"].values, dtype=float)[valid_mask],

                        }

                    )



            all_profiles.append(profile)

    except Exception:

        all_profiles = []

        for x in xs:

            profile = extractor.extract_vertical_profile(float(x))

            basement_depth = float(bd_fn(float(x)))

            if basement_depth > 0:

                profile_depths = np.asarray(profile["depth"].values, dtype=float)

                adjusted = profile_depths - basement_depth

                valid_mask = adjusted >= 0

                if np.any(valid_mask):

                    profile = pd.DataFrame(

                        {

                            "depth": adjusted[valid_mask],

                            "vp": np.asarray(profile["vp"].values, dtype=float)[valid_mask],

                        }

                    )

            all_profiles.append(profile)



    if not all_profiles:

        raise RuntimeError("未能提取任何剖面")



    all_depths: list[float] = []

    for profile in all_profiles:

        d = np.asarray(profile["depth"].values, dtype=float)

        d = d[np.isfinite(d)]

        if d.size > 0:

            all_depths.extend(d.tolist())

    if not all_depths:

        raise RuntimeError("剖面中无有效深度")



    all_depths_array = np.asarray(all_depths, dtype=float)

    depth_min = float(np.nanmin(all_depths_array))

    depth_max = float(np.nanmax(all_depths_array))



    first_d = np.asarray(all_profiles[0]["depth"].values, dtype=float)

    first_d = first_d[np.isfinite(first_d)]

    if first_d.size > 1:

        depth_interval = float(np.mean(np.diff(np.sort(first_d))))

    else:

        depth_interval = 0.1

    depth_interval = max(depth_interval, 1e-4)



    reference_depths = np.arange(depth_min, depth_max + 0.5 * depth_interval, depth_interval)



    vp_rows: list[np.ndarray] = []

    for profile in all_profiles:

        d = profile["depth"].values

        v = profile["vp"].values

        if hasattr(d, "values"):

            d = d.values

        if hasattr(v, "values"):

            v = v.values

        d = np.asarray(d, dtype=float)

        v = np.asarray(v, dtype=float)

        vp_interp = _interp_vp_to_grid(d, v, reference_depths)

        vp_rows.append(vp_interp)



    if not vp_rows:

        raise RuntimeError("插值对齐失败")



    vp_matrix = np.asarray(vp_rows, dtype=float)

    averaged_vp = np.nanmean(vp_matrix, axis=0)

    valid_mask = np.isfinite(averaged_vp)

    if not np.any(valid_mask):

        raise RuntimeError("平均后无有效值")



    final_depths = reference_depths[valid_mask]

    final_vp = averaged_vp[valid_mask]

    return pd.DataFrame({"depth": final_depths, "vp": final_vp}).reset_index(drop=True)
