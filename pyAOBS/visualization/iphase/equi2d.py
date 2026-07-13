"""
2Dequi 时差模式

根据 diff_pairs 计算 PPS-PPP 时差平均值 ``avg_pps_ppp``，用户提供 PPS/PSS 比值，
将 PSS 映射为等效 PPS、等效 PPP；并按
``t_等效PSP = t_PSS - avg_pps_ppp`` 写入等效 PSP（可选相位号），
供 RAYINVR 正演得到 tx.out。
"""

from __future__ import annotations

import numpy as np

from .models import PhaseDataset, Shot


def build_equi_dataset(
    diff_pairs: list[tuple[float, float, float]],
    ds_pss: PhaseDataset,
    *,
    pps_pss_ratio: float = 1.0,
    phase_ppp: int = 5,
    phase_pps: int = 14,
    phase_pss: int = 24,
    phase_psp: int | None = None,
) -> PhaseDataset:
    """
    2Dequi：根据 PPS-PPP 时差平均值和 PPS/PSS 比值，将 PSS 转换为等效 PPS 和 PPP。
    输出包含等效 PPS、等效 PPP、原始 PSS；若给定 ``phase_psp``，再写入等效 PSP。

    步骤：
    1. 计算 PPS-PPP 时差平均值 avg_pps_ppp（来自 diff_pairs）
    2. PPS等效 = PSS 走时 * (PPS/PSS 比值)
    3. PPP等效 = PPS等效 - avg_pps_ppp
    4. （可选）PSP等效 = PSS 走时 - avg_pps_ppp，相位号为 ``phase_psp``
    5. 生成 PhaseDataset 供写 tx_2Dequiv.in 并正演 tx.out
    """
    if not diff_pairs:
        return PhaseDataset()
    arr = [p[2] for p in diff_pairs if len(p) >= 3]
    if not arr:
        return PhaseDataset()
    avg_pps_ppp = sum(arr) / len(arr)

    out = PhaseDataset()
    for s in ds_pss.shots:
        pss_picks = [p for p in s.picks if p.phase_id == phase_pss]
        if not pss_picks:
            continue
        new_shot = Shot(xshot=s.xshot, tshot=s.tshot, ushot=s.ushot)
        for p in pss_picks:
            t_pps_equiv = p.t * pps_pss_ratio
            t_ppp_equiv = t_pps_equiv - avg_pps_ppp
            new_shot.add_pick(p.x, t_pps_equiv, p.u, phase_pps)
            new_shot.add_pick(p.x, t_ppp_equiv, p.u, phase_ppp)
            new_shot.add_pick(p.x, p.t, p.u, phase_pss)
            if phase_psp is not None:
                t_psp_equiv = float(p.t) - avg_pps_ppp
                new_shot.add_pick(p.x, t_psp_equiv, p.u, int(phase_psp))
        out.shots.append(new_shot)
    return out


def build_equi_dataset_picks_plus_equiv_psp(
    ds_full: PhaseDataset,
    diff_pairs: list[tuple[float, float, float]],
    *,
    phase_ppp: int,
    phase_pps: int,
    phase_pss: int,
    phase_psp: int,
) -> PhaseDataset:
    """
    2Dequi（勾选写等效 PSP）：`tx_2Dequiv.in` 写入与原始文件一致的
   PPP / PPS / PSS / PSP（如有）拾取，并在每个 PSS 点追加
    ``t_PSP_eq = t_PSS - mean(PPS-PPP)``（同一 PSP 相位号，供 RAYINVR 正演）。
    """
    if not diff_pairs:
        return PhaseDataset()
    arr = [float(p[2]) for p in diff_pairs if len(p) >= 3 and np.isfinite(p[2])]
    if not arr:
        return PhaseDataset()
    avg_pps_ppp = float(sum(arr) / len(arr))

    out = PhaseDataset()
    ippp, ipps, ipss, ipsp = int(phase_ppp), int(phase_pps), int(phase_pss), int(phase_psp)
    for s in ds_full.shots:
        new_shot = Shot(xshot=s.xshot, tshot=s.tshot, ushot=s.ushot)
        for p in s.picks:
            pid = int(p.phase_id)
            if pid in (ippp, ipps, ipss, ipsp):
                new_shot.add_pick(float(p.x), float(p.t), float(p.u), pid)
        for p in s.picks:
            if int(p.phase_id) == ipss:
                new_shot.add_pick(float(p.x), float(p.t) - avg_pps_ppp, float(p.u), ipsp)
        if new_shot.picks:
            out.shots.append(new_shot)
    return out
