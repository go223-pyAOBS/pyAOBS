"""
iphase 数据模型

定义 tx.in 文件对应的核心数据结构：Pick、Shot、PhaseDataset。
格式参考RAYINVR format(3f10.3,i10)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence


@dataclass
class Pick:
    """单条相位拾取。"""

    x: float  # 偏移/距离 (km)
    t: float  # 到时 (s)
    u: float  # 不确定度/权重 (s)
    phase_id: int  # 相位编号，必须 > 0

    def __post_init__(self) -> None:
        if self.phase_id <= 0:
            raise ValueError(f"Pick.phase_id 必须 > 0，实际为 {self.phase_id}")

    def to_row(self) -> tuple:
        """返回 (x, t, u, phase_id) 用于写入 tx.in。"""
        return (self.x, self.t, self.u, self.phase_id)


@dataclass
class Shot:
    """单个炮点及其拾取。"""

    xshot: float  # 炮点 X 坐标 (km)
    tshot: float  # 炮点时间 (s)，通常为 -1.0 或 0
    ushot: float  # 炮点权重，通常为 0
    picks: List[Pick] = field(default_factory=list)

    def add_pick(self, x: float, t: float, u: float, phase_id: int) -> None:
        self.picks.append(Pick(x=x, t=t, u=u, phase_id=phase_id))

    def picks_by_phase(self, phase_id: int) -> List[Pick]:
        return [p for p in self.picks if p.phase_id == phase_id]

    def has_phase(self, phase_id: int) -> bool:
        return any(p.phase_id == phase_id for p in self.picks)


@dataclass
class PhaseDataset:
    """
    多炮点 tx 数据集。

    结构：shots[0], shots[1], ... 每个 Shot 包含该炮点的 picks。
    对应 tx.in 的 shot(i=0) → picks(i>0) → shot(i=0) → ... → i=-1。
    """

    shots: List[Shot] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.shots)

    def __iter__(self) -> Iterator[Shot]:
        return iter(self.shots)

    @property
    def n_picks(self) -> int:
        return sum(len(s.picks) for s in self.shots)

    @property
    def n_shots(self) -> int:
        return len(self.shots)

    def phase_ids(self) -> List[int]:
        """返回数据集中出现过的所有相位编号（去重、排序）。"""
        ids: set = set()
        for s in self.shots:
            for p in s.picks:
                ids.add(p.phase_id)
        return sorted(ids)

    def filter_by_phases(self, phase_ids: Sequence[int]) -> "PhaseDataset":
        """返回仅包含指定相位的新 PhaseDataset（不修改自身）。"""
        ids_set = set(phase_ids)
        out = PhaseDataset()
        for s in self.shots:
            kept = [p for p in s.picks if p.phase_id in ids_set]
            if kept:
                new_shot = Shot(xshot=s.xshot, tshot=s.tshot, ushot=s.ushot)
                new_shot.picks = kept
                out.shots.append(new_shot)
        return out

    def exclude_phases(self, phase_ids: Sequence[int]) -> "PhaseDataset":
        """返回剔除指定相位后的新 PhaseDataset。"""
        ids_set = set(phase_ids)
        out = PhaseDataset()
        for s in self.shots:
            kept = [p for p in s.picks if p.phase_id not in ids_set]
            if kept:
                new_shot = Shot(xshot=s.xshot, tshot=s.tshot, ushot=s.ushot)
                new_shot.picks = kept
                out.shots.append(new_shot)
        return out
