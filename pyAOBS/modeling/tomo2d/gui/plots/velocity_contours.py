"""速度场上叠加 GMT 风格等值线（按色标选表 + Matplotlib）。

``vp`` / ``vs`` / ``vpvs`` / ``water`` 各有一张内置表（``assets/contour_p``、
``contour_s``、``contour_vpvs``、``contour_water``）。第二列 **A** 标注，**C** 只画线。
画法对齐 ``grdcontour -Wa2 -Wc0.5,-``（A 粗实线红字白底，C 细虚线）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

_CONTOUR_ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"


@dataclass(frozen=True)
class GmtContourLevel:
    """一行：速度值 + 是否标注。"""

    value: float
    annotate: bool


def parse_gmt_contour_text(text: str) -> list[GmtContourLevel]:
    """解析 GMT grdcontour 等值线表。

    每行：``值  A|C  [角度]``。``#`` 与空行忽略；第三列角度不用于 Matplotlib。
    """
    out: list[GmtContourLevel] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(">"):
            continue
        parts = line.split()
        try:
            val = float(parts[0])
        except (TypeError, ValueError):
            continue
        flag = parts[1][:1].upper() if len(parts) > 1 else "C"
        out.append(GmtContourLevel(value=val, annotate=(flag == "A")))
    return out


# 内置表：vp=contour_p，vs=contour_s，vpvs=contourK
_DEFAULT_VP_CONTOUR_TEXT = """
2.5 C
3.0 C 0
4.0 A
5.0 A 0
5.1 C
5.2 C
5.3 C
5.4 C
5.5 C
5.6 C
5.7 C
5.8 C
5.9 C
6.0 A
6.1 C
6.2 C
6.3 C
6.4 C 0
6.5 C
6.6 C
6.7 C
6.8 A
6.9 C
7.0 A
7.1 C
7.8 C
8.1 C
"""

_DEFAULT_VS_CONTOUR_TEXT = """
1.50 C
2.25 C
2.50 A
2.75 C
3.00 A
3.25 C
3.50 A
3.75 C
4.00 A
4.25 C
4.50 A
4.75 C
5.00 A
"""

_DEFAULT_VPVS_CONTOUR_TEXT = """
1.70 A
1.75 C
1.80 A
1.85 C
1.90 A
1.95 C
2.00 A
"""

_DEFAULT_WATER_CONTOUR_TEXT = """
1.35 A
1.36 C
1.37 C
1.38 C
1.39 C
1.40 A
1.41 C
1.42 C
1.43 C
1.44 C
1.45 A
1.46 C
1.47 C
1.48 C
1.49 C
1.50 A
1.51 C
1.52 C
1.53 C
1.54 C
1.55 A
1.56 C
1.57 C
1.58 C
1.59 C
1.60 A
1.61 C
1.62 C
1.63 C
1.64 C
1.65 A
"""


def _contour_text_from_asset(filename: str, fallback: str) -> str:
    """优先读 ``assets/contour_*`` 模板，文件缺失或空则用内置副本。"""
    p = _CONTOUR_ASSET_DIR / filename
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return fallback
    return text if parse_gmt_contour_text(text) else fallback


DEFAULT_VP_CONTOURS: tuple[GmtContourLevel, ...] = tuple(
    parse_gmt_contour_text(_contour_text_from_asset("contour_p", _DEFAULT_VP_CONTOUR_TEXT))
)
DEFAULT_VS_CONTOURS: tuple[GmtContourLevel, ...] = tuple(
    parse_gmt_contour_text(_contour_text_from_asset("contour_s", _DEFAULT_VS_CONTOUR_TEXT))
)
DEFAULT_VPVS_CONTOURS: tuple[GmtContourLevel, ...] = tuple(
    parse_gmt_contour_text(
        _contour_text_from_asset("contour_vpvs", _DEFAULT_VPVS_CONTOUR_TEXT)
    )
)
DEFAULT_WATER_CONTOURS: tuple[GmtContourLevel, ...] = tuple(
    parse_gmt_contour_text(
        _contour_text_from_asset("contour_water", _DEFAULT_WATER_CONTOUR_TEXT)
    )
)

CONTOUR_PREF_KEY = "gui.plot_smesh_contours"


def contours_enabled(state) -> bool:
    """表单未写该键时默认叠加。"""
    if state is None:
        return True
    get_bool = getattr(state, "get_bool", None)
    if callable(get_bool):
        return bool(get_bool(CONTOUR_PREF_KEY, True))
    return True


def set_contours_enabled(state, on: bool) -> None:
    if state is not None and hasattr(state, "set"):
        state.set(CONTOUR_PREF_KEY, "1" if on else "0")


def contours_for_cmap_id(cmap_id: str | None) -> tuple[GmtContourLevel, ...]:
    """色标 ``vp`` / ``vs`` / ``vpvs`` / ``water`` 对应的内置等值线表。"""
    cid = (cmap_id or "vp").strip().lower()
    if cid == "vs":
        return DEFAULT_VS_CONTOURS
    if cid == "vpvs":
        return DEFAULT_VPVS_CONTOURS
    if cid == "water":
        return DEFAULT_WATER_CONTOURS
    return DEFAULT_VP_CONTOURS


def contour_specs_for_state(state):
    """勾选时返回当前色标对应的等值线表；关闭则空序列（不画）。"""
    if not contours_enabled(state):
        return []
    cmap_id = "vp"
    if state is not None:
        try:
            from ..services.smesh_plot_core import get_smesh_cmap_id

            cmap_id = get_smesh_cmap_id(state)
        except Exception:
            cmap_id = "vp"
    return list(contours_for_cmap_id(cmap_id))


def auto_contour_specs(
    lo: float,
    hi: float,
    *,
    target: int = 7,
) -> list[GmtContourLevel]:
    """按数据范围生成等值线（误差 σ、ΔV 等，不能套 Vp 公用表）。"""
    import math

    try:
        a, b = float(lo), float(hi)
    except (TypeError, ValueError):
        return []
    if not math.isfinite(a) or not math.isfinite(b) or b <= a:
        return []
    span = b - a
    raw = span / max(int(target), 1)
    if raw <= 0 or not math.isfinite(raw):
        return []
    exp = math.floor(math.log10(raw))
    frac = raw / (10**exp)
    if frac <= 1.0:
        nice = 1.0
    elif frac <= 2.0:
        nice = 2.0
    elif frac <= 2.5:
        nice = 2.5
    elif frac <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    step = nice * (10**exp)
    ndig = max(0, -int(math.floor(math.log10(step))) + 1)

    def _round(v: float) -> float:
        return round(v, ndig)

    start = _round(math.ceil((a + step * 1e-9) / step) * step)
    values: list[float] = []
    v = start
    guard = 0
    while v < b - step * 1e-9 and guard < 64:
        values.append(_round(v))
        v = _round(v + step)
        guard += 1
    if not values:
        return []

    def _major(val: float) -> bool:
        if val == 0.0:
            return False
        e = math.floor(math.log10(abs(val)))
        mant = abs(val) / (10**e)
        return any(abs(mant - m) < 0.08 for m in (1.0, 2.0, 5.0))

    majors = [_major(x) for x in values]
    if not any(majors):
        majors = [(i % 2) == 0 for i in range(len(values))]
    return [
        GmtContourLevel(value=x, annotate=bool(ann))
        for x, ann in zip(values, majors)
    ]


def velocity_as_nz_nx(data, x, z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``imshow`` / ``contour`` 用 (nz, nx)。

    已是 ``(len(z), len(x))`` 时不转置。方阵 ``nx==nz`` 时旧逻辑会误转，
    等值线会竖过来、和色块对不上。
    """
    arr = np.asarray(data, dtype=float)
    xx = np.asarray(x, dtype=float)
    zz = np.asarray(z, dtype=float)
    nz, nx = int(zz.size), int(xx.size)
    if arr.ndim == 2 and arr.shape == (nz, nx):
        return arr, xx, zz
    if arr.ndim == 2 and arr.shape == (nx, nz):
        return arr.T, xx, zz
    return arr, xx, zz


def _water_contour_pen(specs: Sequence[GmtContourLevel]) -> bool:
    """水层 0.01 表线很密，用细笔以免压住射线。"""
    vals = [float(s.value) for s in specs]
    if len(vals) < 8:
        return False
    return min(vals) >= 1.30 and max(vals) <= 1.70


def overlay_velocity_contours(
    ax,
    data,
    x,
    z,
    specs: Sequence[GmtContourLevel] | None = None,
    *,
    zorder: float = 1,
):
    """叠加等值线。``specs is None`` 用 Vp 表；空序列则不画。"""
    if specs is None:
        specs = DEFAULT_VP_CONTOURS
    if not specs:
        return None, None
    arr, xx, zz = velocity_as_nz_nx(data, x, z)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return None, None
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    a_lv = sorted(
        {s.value for s in specs if s.annotate and vmin <= s.value <= vmax}
    )
    c_lv = sorted(
        {s.value for s in specs if (not s.annotate) and vmin <= s.value <= vmax}
    )
    water = _water_contour_pen(specs)
    lw_c, lw_a, fsz = (0.22, 0.55, 6.5) if water else (0.5, 1.5, 8)
    X, Z = np.meshgrid(xx, zz)
    cs_c = cs_a = None
    if c_lv:
        cs_c = ax.contour(
            X,
            Z,
            arr,
            levels=c_lv,
            colors="k",
            linewidths=lw_c,
            linestyles="--",
            zorder=zorder,
            alpha=0.55 if water else 1.0,
        )
    if a_lv:
        cs_a = ax.contour(
            X,
            Z,
            arr,
            levels=a_lv,
            colors="k",
            linewidths=lw_a,
            linestyles="-",
            zorder=zorder,
            alpha=0.75 if water else 1.0,
        )
        texts = ax.clabel(
            cs_a,
            a_lv,
            fmt=lambda v: f"{v:g}",
            fontsize=fsz,
            inline=True,
            inline_spacing=6 if water else 8,
            colors="#cc0000",
        )
        for t in texts or []:
            t.set_color("#cc0000")
            t.set_bbox(
                dict(
                    boxstyle="round,pad=0.08" if water else "round,pad=0.12",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.72 if water else 0.9,
                )
            )
    return cs_c, cs_a
