"""GMT CPT → Matplotlib，不依赖 pygmt / GMT 动态库。"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def gmt_cpt_segment_rgb(parts: List[str]) -> List[float] | None:
    """解析一行 CPT 色段：8 列 ``z r g b z r g b``，或 GMT 斜杠 ``z r/g/b z r/g/b``。"""
    if len(parts) == 8:
        try:
            return [float(x) for x in parts]
        except ValueError:
            return None
    if len(parts) == 4 and "/" in parts[1] and "/" in parts[3]:
        try:
            r1, g1, b1 = (float(x) for x in parts[1].split("/"))
            r2, g2, b2 = (float(x) for x in parts[3].split("/"))
            return [float(parts[0]), r1, g1, b1, float(parts[2]), r2, g2, b2]
        except ValueError:
            return None
    return None


def parse_gmt_cpt_for_matplotlib(cpt_file: str) -> Tuple[LinearSegmentedColormap, float, float]:
    """解析 GMT 分段 CPT，返回色标与数据域 ``[z_min, z_max]``。

    ``B`` / ``F`` / ``N`` 接到 matplotlib 的 under / over / bad。
    """
    cpt_rows: List[List[float]] = []
    z_samples: List[float] = []
    under = over = bad = None
    with open(cpt_file, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            key = parts[0].upper()
            if key in "BFN" and len(parts) >= 4:
                try:
                    rgb = (
                        float(parts[1]) / 255.0,
                        float(parts[2]) / 255.0,
                        float(parts[3]) / 255.0,
                    )
                except ValueError:
                    continue
                if key == "B":
                    under = rgb
                elif key == "F":
                    over = rgb
                else:
                    bad = rgb
                continue
            parsed = gmt_cpt_segment_rgb(parts)
            if parsed is None:
                continue
            z1, r1, g1, b1, z2, r2, g2, b2 = parsed
            z_samples.extend((z1, z2))
            cpt_rows.append([z1, r1 / 255.0, g1 / 255.0, b1 / 255.0])
            cpt_rows.append([z2, r2 / 255.0, g2 / 255.0, b2 / 255.0])
    if not cpt_rows:
        raise ValueError(f"CPT 中无有效 8 列色标段: {cpt_file!r}")
    arr = np.array(cpt_rows, dtype=float)
    z_min = float(min(z_samples))
    z_max = float(max(z_samples))
    if z_max <= z_min:
        raise ValueError(f"CPT 数据域无效 z_min={z_min}, z_max={z_max}: {cpt_file!r}")
    z_norm = (arr[:, 0] - z_min) / (z_max - z_min)
    colors = arr[:, 1:]
    pairs: list[tuple[float, tuple[float, float, float]]] = []
    for zv, rgb in zip(z_norm, colors):
        zf = float(zv)
        rgb_t = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
        if pairs and abs(pairs[-1][0] - zf) < 1e-12:
            pairs[-1] = (zf, rgb_t)
        else:
            pairs.append((zf, rgb_t))
    cmap = LinearSegmentedColormap.from_list("custom_gmt_cpt", pairs)
    ext: dict[str, tuple[float, float, float]] = {}
    if under is not None:
        ext["under"] = under
    if over is not None:
        ext["over"] = over
    if bad is not None:
        ext["bad"] = bad
    if ext:
        try:
            cmap = cmap.with_extremes(**ext)
        except Exception:
            if under is not None:
                cmap.set_under(under)
            if over is not None:
                cmap.set_over(over)
            if bad is not None:
                cmap.set_bad(bad)
    return cmap, z_min, z_max
