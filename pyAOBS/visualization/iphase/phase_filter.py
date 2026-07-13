"""
相位筛选（对应 txphase.f）

从 tx.in 中筛选/剔除指定相位，行为与 txphase.f 一致：
- 仅当某炮至少有一条被选拾取时才输出该炮头；
- 保留炮点结构及 format(3f10.3,i10) 输出格式。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .models import PhaseDataset


def select_phases(
    dataset: PhaseDataset,
    phase_ids: Sequence[int],
) -> Tuple[PhaseDataset, Dict[str, int]]:
    """
    保留指定相位的拾取，其余丢弃。

    Returns:
        (筛选后的 PhaseDataset, 统计信息 {"npick": N, "nshot": M})
    """
    ids_set = set(phase_ids)
    out = PhaseDataset()
    for s in dataset.shots:
        kept = [p for p in s.picks if p.phase_id in ids_set]
        if kept:
            from .models import Shot

            new_shot = Shot(xshot=s.xshot, tshot=s.tshot, ushot=s.ushot)
            new_shot.picks = kept
            out.shots.append(new_shot)
    stats = {"npick": out.n_picks, "nshot": out.n_shots}
    return out, stats


def exclude_phases(
    dataset: PhaseDataset,
    phase_ids: Sequence[int],
) -> Tuple[PhaseDataset, Dict[str, int]]:
    """
    剔除指定相位，保留其余拾取。
    """
    return select_phases(
        dataset,
        [pid for pid in dataset.phase_ids() if pid not in set(phase_ids)],
    )
