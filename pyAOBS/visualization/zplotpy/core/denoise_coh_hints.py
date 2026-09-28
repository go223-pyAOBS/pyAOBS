"""去噪相干门控参数启发式起点（cl 样点数、cb 回混）。

说明（非物理精确反演，仅供交互起点）：
- ``offsets`` 与 ``data_loader`` 中一致，单位为 **千米**（炮检距 ``offsti``）。
- **cl**：按「偏移排序后相邻道间距中位数 Δh」×「假定沿偏移轴视时差梯度（s/km）」估算邻道可走时差，
  再换算为 ``ceil(Δt / dt)``，并限制在 ``lag_max``（界面通常为 8）。
  梯度档位见 ``COHERENCE_DIP_SLOPE_PRESET_ITEMS``（约 6 / 12 / 22 / 35 ms/km）；也可用自定义 ms/km。
- **cb**：主要随 **采样率** 略增——采样越密，时频收缩越容易伤到可分辨细节，倾向略提高回混保留原始。
- **从拾取估梯度**：对某一拾取字，用各道 pick 走时（应与 ``times``/去噪一致 **记录时间**）与 ``offsets``
  （km）；按偏移排序后，对 **相邻两道上的拾取** 计算 ``|Δt|/|Δh|``，以其 **median** 作为填入自定义梯度的默认值，
  并给出 **p75** 供保守调大 lag 参考。不同拾取字代表不同震相时应 **切换拾取字** 分别估算。

**折合速度（vred / vredf）**：当前启发式 **不读取** GUI 折合参数。去噪与三门相干均在同一 ``times``
记录时间采样栅格上逐样点对齐；界面上的折合走时只是把每道在绘图时整体上移/下移（``_compute_display_tshift`` /
``_compute_reduction_tshift``），**不重采样到折合时间轴**。因此 ms/km 描述「记录时间轴上邻道走时变化」；
屏上被 vred **拉平** 的同相轴在样本对齐上未必变缓，仍靠 **Lag(cl)** 在真时栅格上搜邻道时差。

若要将折合并入推荐，需单独 UI 选项并用 **残余** ``dΔt/dh``（依赖相速度与所选 vreg，假设须明确）。
"""

from __future__ import annotations

import math
from typing import Dict, Final, List, Optional, Tuple

import numpy as np


# (界面标签, dip_slope_s_per_km)：沿偏移轴邻道走时差的粗略量级（秒/千米）。
COHERENCE_DIP_SLOPE_PRESET_ITEMS: Tuple[Tuple[str, float], ...] = (
    ("平缓 (~6 ms/km)", 0.006),
    ("中等 (~12 ms/km)", 0.012),
    ("陡 (~22 ms/km)", 0.022),
    ("极陡 (~35 ms/km)", 0.035),
)

CUSTOM_DIP_SLOPE_COMBO_MARKER: Final[str] = "custom_ms_per_km"


def estimate_moveout_slope_s_per_km_from_picks(
    picks_by_trace: Dict[int, float],
    offsets_km: Optional[np.ndarray],
    *,
    dh_floor_km: float = 1e-12,
) -> Tuple[Optional[float], Optional[float], str]:
    """由同一拾取字在多道上的拾取走时，估计沿偏移梯度的局域 ``|dt/dh|``（s/km）。

    方法：将所有 (offset, pick_time) 按偏移排序，对相邻两拾取计算 ``abs(Δt)/abs(Δh)``，
    再对上述斜率样本取 median 与 75 分位。适用于 **整条剖面上同源震相连线近似分段单调** 的情形；
    缺失道或拾取稀疏时段落变少，统计会不稳定。

    Returns
    -------
    median_slope, p75_slope, message
        填入 UI 时宜用 median；若 lag 不足可手工改大到接近 p75。
    """
    if offsets_km is None:
        return None, None, "偏移距不可用。"
    if not picks_by_trace:
        return None, None, "当前拾取字下无拾取。"

    off = np.asarray(offsets_km, dtype=np.float64).reshape(-1)
    rows: List[Tuple[float, float]] = []
    for g_raw, t_raw in picks_by_trace.items():
        gi = int(g_raw)
        if gi < 0 or gi >= int(off.size):
            continue
        hh = float(off[gi])
        tt = float(t_raw)
        if not (math.isfinite(hh) and math.isfinite(tt)):
            continue
        rows.append((hh, tt))

    if len(rows) < 2:
        return None, None, f"该拾取字有效点不足 2（当前 {len(rows)}）。"

    rows.sort(key=lambda r: r[0])
    seg_slopes: List[float] = []
    dup_h_note = False
    for i in range(1, len(rows)):
        dh = float(rows[i][0] - rows[i - 1][0])
        dt = float(abs(rows[i][1] - rows[i - 1][1]))
        adh = abs(dh)
        if adh <= float(dh_floor_km):
            dup_h_note = True
            continue
        seg_slopes.append(dt / adh)

    if not seg_slopes:
        return None, None, "相邻拾取在偏移上等距或过近(|Δh|过小)，跳过了所有坡度段。" + (
            "（存在同偏移多重拾取时也会如此）。" if dup_h_note else ""
        )

    arr = np.asarray(seg_slopes, dtype=np.float64)
    med = float(np.median(arr))
    p75 = float(np.percentile(arr, 75))
    npick = len(rows)
    nseg = int(arr.size)
    warn = ""
    if nseg < 3:
        warn = " 段落较少，median 仅供参考。"
    elif dup_h_note:
        warn = " 部分段落因 Δh≈0 已跳过。"
    detail = (
        f"拾取→梯度：共 {npick} 道拾取合成 {nseg} 段邻域坡度；"
        f"median={med * 1000.0:.2f} ms/km，p75={p75 * 1000.0:.2f} ms/km。"
        f"已可把 median 填入「自定义梯度」后再点「推荐 cl/cb」。{warn}"
    )
    return med, p75, detail.strip()


def median_neighbor_offset_spacing_km(offsets: Optional[np.ndarray]) -> Optional[float]:
    """偏移轴排序后，相邻道间距绝对值的 **中位数**（km）。无效时为 None。"""
    if offsets is None:
        return None
    o = np.asarray(offsets, dtype=np.float64).reshape(-1)
    if o.size < 2:
        return None
    sidx = np.argsort(o)
    sorted_o = o[sidx]
    d = np.abs(np.diff(sorted_o))
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return None
    return float(np.median(d))


def suggest_coherence_blend(dt_s: float) -> float:
    """由采样间隔启发 cb（0~1），再交给界面裁剪到 [0, 0.90]。

    将约 80 Hz→偏低 cb，约 320 Hz→偏高 cb；中间做对数插值。
    """
    dt_s = float(dt_s)
    if not math.isfinite(dt_s) or dt_s <= 0:
        return 0.35
    fs = 1.0 / dt_s
    lo_f, hi_f = 80.0, 320.0
    denom = math.log(hi_f / lo_f)
    if denom <= 0:
        u = 0.5
    else:
        u = (math.log(max(fs, lo_f)) - math.log(lo_f)) / denom
        u = float(min(1.0, max(0.0, u)))
    cb = 0.30 + 0.42 * u
    cb = float(min(0.85, max(0.28, cb)))
    return round(cb, 2)


def suggest_coherence_cb_cl_start(
    dt_s: float,
    offsets: Optional[np.ndarray],
    *,
    lag_max: int = 8,
    dip_slope_s_per_km: float = 0.012,
    dip_margin: float = 1.2,
) -> Tuple[int, float, str]:
    """返回推荐的 ``(cl, cb, 中文说明字符串)``。

    Parameters
    ----------
    dt_s
        采样间隔（秒）。
    offsets
        各道偏移距（km），与 ``loaded['offsets']`` 一致；可为 None。
    lag_max
        cl 上限（与相干对话框一致，默认 8）。
    dip_slope_s_per_km
        假定的邻道走时差异量级：**秒/km**，乘以 Δh(km) 得邻道时差上沿。
        默认 0.012 s/km ≈ **12 ms/km**。
    dip_margin
        安全系数，略放大估算时差后再换算样点。
    """
    dt_s = float(dt_s)
    if not math.isfinite(dt_s) or dt_s <= 0:
        return 2, 0.35, "dt 无效，退回默认 cl=2, cb=0.35。"

    lag_max = int(max(0, min(16, lag_max)))
    md_km = median_neighbor_offset_spacing_km(offsets)

    physics_cap_s = lag_max * dt_s * 0.98

    if md_km is None or not math.isfinite(md_km) or md_km <= 0:
        # 无偏移距：仅用 dt，给出「不少于数个采样」的下沿，避免 cl=0 过快失效
        dt_neighbor_s = min(physics_cap_s, max(dt_s * 3.0, 0.0025))
        mode = "偏移距不可用"
    else:
        dt_geom = float(dip_margin) * abs(float(dip_slope_s_per_km)) * float(md_km)
        floor_s = max(dt_s * 2.0, 0.0015)
        dt_neighbor_s = min(physics_cap_s, max(dt_geom, floor_s))
        mode = f"median Δh≈{md_km:.4g} km"

    cl = int(math.ceil(dt_neighbor_s / dt_s))
    cl = int(min(lag_max, max(0, cl)))

    cb = suggest_coherence_blend(dt_s)

    slope_ms_per_km = float(dip_slope_s_per_km) * 1000.0
    detail = (
        f"启发说明（{mode}）：假定沿偏移轴时差梯度量级≈{slope_ms_per_km:.1f} ms/km × 安全系数 {dip_margin:g} → "
        f"靶邻道时差≈{dt_neighbor_s * 1000.0:.2f} ms → cl={cl}（≤{lag_max}）；"
        f"按采样率启发 cb≈{cb:.2f}（愈密采样略偏高以少伤细节）。"
        "若同相轴仍偏糊或仍过削：增大 cl / 提高 cb。"
    )
    return cl, cb, detail


__all__ = [
    "COHERENCE_DIP_SLOPE_PRESET_ITEMS",
    "CUSTOM_DIP_SLOPE_COMBO_MARKER",
    "estimate_moveout_slope_s_per_km_from_picks",
    "median_neighbor_offset_spacing_km",
    "suggest_coherence_blend",
    "suggest_coherence_cb_cl_start",
]
