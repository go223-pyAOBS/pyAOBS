"""
多相位组合

在多个 tx.in 中匹配同一炮点、同一接收点的 PPP/PPS/PSS，
构造 PPS-PPP 与修正 PSP 等派生相位。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .io_tx import read_tx
from .models import PhaseDataset, Pick, Shot


DEFAULT_TOL = 0.001


def compute_ppp_pps_diff_pairs(
    ds_ppp: PhaseDataset,
    ds_pps: PhaseDataset,
    *,
    ip1: int = 5,
    ip2: int = 14,
    tol: float = DEFAULT_TOL,
) -> list[tuple[float, float, float]]:
    """
    匹配 PPP 与 PPS，返回 (model_dist, true_offset, t_diff) 列表。

    model_dist = x（模型距离）；true_offset = x - xshot（有符号真实偏移距）；t_diff = t_PPS - t_PPP。
    """
    pairs: list[tuple[float, float, float]] = []
    for s1 in ds_ppp.shots:
        for p1 in s1.picks_by_phase(ip1):
            x1shot, x1, t1 = s1.xshot, p1.x, p1.t
            for s2 in ds_pps.shots:
                if not _shots_match(x1shot, s1.tshot, s2.xshot, s2.tshot, tol):
                    continue
                p2 = _find_pick_at(s2, x1, ip2, tol)
                if p2 is None:
                    continue
                t2 = p2.t
                model_dist = x1
                true_offset = x1 - x1shot  # 有符号，分正负偏移距
                t_diff = t2 - t1
                pairs.append((model_dist, true_offset, t_diff))
                break
    return pairs


@dataclass
class CombineResult:
    """combine_ppp_pps_pss 的返回结果。"""

    tx1_out: PhaseDataset  # PPS-PPP 相位
    tx2_out: PhaseDataset  # 修正 PSP 相位
    npick_matched: int
    npick_unmatched: int
    nshot: int


def _find_pick_at(
    shot: Shot,
    x: float,
    phase_id: int,
    tol: float = DEFAULT_TOL,
) -> Optional[Pick]:
    for p in shot.picks:
        if p.phase_id == phase_id and abs(p.x - x) < tol:
            return p
    return None


def _shots_match(
    x1: float, t1: float,
    x2: float, t2: float,
    tol: float = DEFAULT_TOL,
) -> bool:
    return abs(x1 - x2) < tol and abs(t1 - t2) < tol


def combine_ppp_pps_pss(
    tx1_path: str,
    tx2_path: str,
    tx3_path: str,
    *,
    ip1: int,
    ip2: int,
    ip3: int,
    ip4: int,
    ip5: int,
    tol: float = DEFAULT_TOL,
) -> CombineResult:
    """
    逻辑：匹配 PPP/PPS/PSS 并构造 PPS-PPP、修正 PSP。

    - tx1.in: PPP 相位 (ip1)
    - tx2.in: PPS 相位 (ip2)
    - tx3.in: PSS 相位 (ip3)
    - 输出1: PPS-PPP，相位号 ip4
    - 输出2: 修正 PSP，相位号 ip5

    匹配条件：同一炮点 (xshot, tshot) 且同一接收点 x。
    同一炮点的多个匹配只写一次炮头。
    """
    ds1 = read_tx(tx1_path)
    ds2 = read_tx(tx2_path)
    ds3 = read_tx(tx3_path)

    out1 = PhaseDataset()
    out2 = PhaseDataset()
    npick_matched = 0
    npick_unmatched = 0
    nshot = 0

    # 同一炮点的多个匹配只写一次炮头；out1/out2 炮点一一对应
    seen_shot_key: set = set()

    for s1 in ds1.shots:
        picks1 = s1.picks_by_phase(ip1)
        if not picks1:
            continue

        for p1 in picks1:
            x1shot, t1shot = s1.xshot, s1.tshot
            x1, t1, u1 = p1.x, p1.t, p1.u
            matched = False

            for s2 in ds2.shots:
                if not _shots_match(x1shot, t1shot, s2.xshot, s2.tshot, tol):
                    continue
                p2 = _find_pick_at(s2, x1, ip2, tol)
                if p2 is None:
                    continue
                x2, t2, u2 = p2.x, p2.t, p2.u
                x2shot, t2shot, u2shot = s2.xshot, s2.tshot, s2.ushot

                for s3 in ds3.shots:
                    if not _shots_match(x2shot, t2shot, s3.xshot, s3.tshot, tol):
                        continue
                    p3 = _find_pick_at(s3, x2, ip3, tol)
                    if p3 is None:
                        continue
                    x3, t3, u3 = p3.x, p3.t, p3.u
                    x3shot, t3shot, u3shot = s3.xshot, s3.tshot, s3.ushot

                    key = (round(x2shot, 6), round(t2shot, 6))
                    if key not in seen_shot_key:
                        seen_shot_key.add(key)
                        out1.shots.append(Shot(x2shot, t2shot, u2shot))
                        out2.shots.append(Shot(x3shot, t3shot, u3shot))
                        nshot += 1

                    out1.shots[-1].add_pick(
                        (x1 + x2) / 2, t2 - t1, u1 + u2, ip4
                    )
                    out2.shots[-1].add_pick(
                        (x2 + x3) / 2, t3 - (t2 - t1), u1 + u2 + u3, ip5
                    )
                    npick_matched += 1
                    matched = True
                    break
                if matched:
                    break
            if not matched:
                npick_unmatched += 1

    return CombineResult(
        tx1_out=out1,
        tx2_out=out2,
        npick_matched=npick_matched,
        npick_unmatched=npick_unmatched,
        nshot=nshot,
    )


def combine_ppp_pps_pss_from_datasets(
    ds1: PhaseDataset,
    ds2: PhaseDataset,
    ds3: PhaseDataset,
    *,
    ip1: int,
    ip2: int,
    ip3: int,
    ip4: int,
    ip5: int,
    tol: float = DEFAULT_TOL,
) -> CombineResult:
    """
    Same as combine_ppp_pps_pss but takes PhaseDataset instead of file paths.
    Useful when all three phases come from a single tx.in file.
    """
    out1 = PhaseDataset()
    out2 = PhaseDataset()
    npick_matched = 0
    npick_unmatched = 0
    nshot = 0
    seen_shot_key: set = set()

    for s1 in ds1.shots:
        picks1 = s1.picks_by_phase(ip1)
        if not picks1:
            continue

        for p1 in picks1:
            x1shot, t1shot = s1.xshot, s1.tshot
            x1, t1, u1 = p1.x, p1.t, p1.u
            matched = False

            for s2 in ds2.shots:
                if not _shots_match(x1shot, t1shot, s2.xshot, s2.tshot, tol):
                    continue
                p2 = _find_pick_at(s2, x1, ip2, tol)
                if p2 is None:
                    continue
                x2, t2, u2 = p2.x, p2.t, p2.u
                x2shot, t2shot, u2shot = s2.xshot, s2.tshot, s2.ushot

                for s3 in ds3.shots:
                    if not _shots_match(x2shot, t2shot, s3.xshot, s3.tshot, tol):
                        continue
                    p3 = _find_pick_at(s3, x2, ip3, tol)
                    if p3 is None:
                        continue
                    x3, t3, u3 = p3.x, p3.t, p3.u
                    x3shot, t3shot, u3shot = s3.xshot, s3.tshot, s3.ushot

                    key = (round(x2shot, 6), round(t2shot, 6))
                    if key not in seen_shot_key:
                        seen_shot_key.add(key)
                        out1.shots.append(Shot(x2shot, t2shot, u2shot))
                        out2.shots.append(Shot(x3shot, t3shot, u3shot))
                        nshot += 1

                    out1.shots[-1].add_pick(
                        (x1 + x2) / 2, t2 - t1, u1 + u2, ip4
                    )
                    out2.shots[-1].add_pick(
                        (x2 + x3) / 2, t3 - (t2 - t1), u1 + u2 + u3, ip5
                    )
                    npick_matched += 1
                    matched = True
                    break
                if matched:
                    break
            if not matched:
                npick_unmatched += 1

    return CombineResult(
        tx1_out=out1,
        tx2_out=out2,
        npick_matched=npick_matched,
        npick_unmatched=npick_unmatched,
        nshot=nshot,
    )
