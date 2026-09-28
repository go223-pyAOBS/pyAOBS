# -*- coding: utf-8 -*-
"""tx.in OBS 目录：L/R 炮块合并、拾取左右计数、标签格式。

``vedit`` / ``iphase`` / ``tomo2d`` 共用，避免各 GUI 各自解析 ``tx.in`` 语义。

约定（与 iphase HELP / tomo2d 预览一致）：

- 炮头 ``i=0``：``t=±1``、``u≈0`` 表示 OBS 左(``-1``)/右(``+1``) 支
- **同一 ``xshot`` 仅对应一个 OBS**（左右各一块，列表按 x 合并）
- 标签：``{站号}  x=… km  (n=…)  L{n_left}/R{n_right}``（两侧均有拾取时显示 L/R）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .tx_io import TxDataset, TxShotBlock, group_shots_by_obs_x

PathLike = str

# km；与 tomo2d / iphase / tx2tomo2d 炮头匹配一致
X_MATCH_TOL = 0.001


@dataclass
class TxObsGroup:
    """合并 L/R 支后的单个 OBS。"""

    xobs: float
    n_picks: int = 0
    n_left: int = 0
    n_right: int = 0
    n_branches: int = 1
    obs_id: int | None = None
    shot_idxs: list[int] = field(default_factory=list)

    def label(self, *, z: float | None = None, suffix: str = "") -> str:
        return format_obs_catalog_label(
            obs_id=self.obs_id,
            xobs=self.xobs,
            n=self.n_picks,
            n_left=self.n_left,
            n_right=self.n_right,
            z=z,
            suffix=suffix,
        )


def count_lr_picks(
    xobs: float,
    pick_xs: Iterable[float],
    *,
    tol: float = X_MATCH_TOL,
) -> tuple[int, int, int]:
    """``(n_total, n_left, n_right)`` — 相对 ``xobs`` 统计拾取落在左/右侧的数量。"""
    xf = float(xobs)
    n_left = n_right = 0
    n_total = 0
    for px in pick_xs:
        n_total += 1
        x = float(px)
        if x < xf - tol:
            n_left += 1
        elif x > xf + tol:
            n_right += 1
    return n_total, n_left, n_right


def group_tx_dataset_by_obs_x(
    dataset: TxDataset,
    *,
    x_tol: float = X_MATCH_TOL,
) -> list[TxObsGroup]:
    """``TxDataset`` → 按 ``xshot`` 合并的 OBS 列表（含 L/R 拾取计数）。"""
    out: list[TxObsGroup] = []
    for xobs, blocks in group_shots_by_obs_x(dataset, x_tol=x_tol):
        pick_xs = [p.x for b in blocks for p in b.picks]
        n, n_left, n_right = count_lr_picks(xobs, pick_xs, tol=x_tol)
        out.append(
            TxObsGroup(
                xobs=float(xobs),
                n_picks=n,
                n_left=n_left,
                n_right=n_right,
                n_branches=len(blocks),
            )
        )
    return out


def group_arrays_by_obs_x(
    shot_x: Sequence[float],
    rcv_x: Sequence[float],
    *,
    x_decimals: int = 3,
    shot_idx: Sequence[int] | None = None,
) -> list[TxObsGroup]:
    """扁平拾取数组 → OBS 组（tomo2d 预览 / 转换共用）。"""
    import numpy as np

    if len(shot_x) == 0:
        return []
    sx = np.asarray(shot_x, dtype=np.float64)
    rx = np.asarray(rcv_x, dtype=np.float64)
    nd = int(x_decimals)
    xkey = np.round(sx, nd)
    tol = 10.0 ** (-nd)
    sidx = (
        np.asarray(shot_idx, dtype=np.int32)
        if shot_idx is not None
        else np.arange(sx.size, dtype=np.int32)
    )
    out: list[TxObsGroup] = []
    for xobs in sorted(np.unique(xkey)):
        m = xkey == xobs
        xobs_f = float(xobs)
        pick_xs = rx[m]
        n, n_left, n_right = count_lr_picks(xobs_f, pick_xs, tol=tol)
        sids = [int(s) for s in np.unique(sidx[m])]
        out.append(
            TxObsGroup(
                xobs=xobs_f,
                n_picks=n,
                n_left=n_left,
                n_right=n_right,
                n_branches=len(sids),
                shot_idxs=sids,
            )
        )
    return out


def nearest_station_id(
    x: float,
    stations: Sequence[tuple[int, float, float]],
    *,
    tol: float = X_MATCH_TOL,
) -> int | None:
    """``station.lis`` 第一列 OBS 号 ↔ 第二列模型距离 ``xshot``。"""
    best_id: int | None = None
    best_d = float(tol) + 1.0
    xf = float(x)
    for oid, sx, _sz in stations:
        d = abs(xf - float(sx))
        if d <= tol and d < best_d:
            best_d = d
            best_id = int(oid)
    return best_id


def attach_station_ids(
    groups: list[TxObsGroup],
    stations: Sequence[tuple[int, float, float]] | None,
    *,
    tol: float = X_MATCH_TOL,
) -> None:
    """原地写入 ``obs_id``（有 ``station.lis`` 时）。"""
    if not stations:
        return
    for g in groups:
        g.obs_id = nearest_station_id(g.xobs, stations, tol=tol)


def sort_obs_groups(groups: list[TxObsGroup]) -> list[TxObsGroup]:
    """有站号时按站号，否则按 ``xobs``。"""
    return sorted(
        groups,
        key=lambda g: (
            g.obs_id is None,
            int(g.obs_id) if g.obs_id is not None else 0,
            float(g.xobs),
        ),
    )


def format_obs_catalog_label(
    *,
    obs_id: int | None = None,
    xobs: float,
    n: int,
    n_left: int = 0,
    n_right: int = 0,
    z: float | None = None,
    suffix: str = "",
) -> str:
    """预览 / 选择列表共用标签（对齐 iphase / tomo2d）。"""
    head = str(int(obs_id)) if obs_id is not None else "—"
    label = f"{head}  x={float(xobs):.3f} km  (n={int(n)})"
    nl = int(n_left)
    nr = int(n_right)
    if nl > 0 and nr > 0:
        label += f"  L{nl}/R{nr}"
    if z is not None:
        label += f"  z={float(z):.3f}"
    if suffix:
        label += f"  {suffix}"
    return label


def obs_groups_from_tx_path(
    path: PathLike,
    *,
    stations: Sequence[tuple[int, float, float]] | None = None,
    x_tol: float = X_MATCH_TOL,
) -> list[TxObsGroup]:
    """读 ``tx.in`` 并返回合并后的 OBS 组。"""
    from .tx_io import read_tx_file

    groups = group_tx_dataset_by_obs_x(read_tx_file(path), x_tol=x_tol)
    attach_station_ids(groups, stations, tol=x_tol)
    return sort_obs_groups(groups)


def split_tx_dataset_by_obs_x(
    dataset: TxDataset,
    *,
    x_tol: float = X_MATCH_TOL,
) -> list[tuple[float, TxDataset]]:
    """按 ``xshot`` 拆成多个 ``TxDataset`` 子集（L/R 支同 x 合并）。"""
    out: list[tuple[float, TxDataset]] = []
    for xobs, blocks in group_shots_by_obs_x(dataset, x_tol=x_tol):
        out.append((float(xobs), TxDataset(shots=list(blocks))))
    return out


def unique_obs_from_tx_groups(
    path: PathLike,
    *,
    x_tol: float = X_MATCH_TOL,
) -> list[tuple[float, int, int, int, int]]:
    """``[(xobs, n_picks, n_left, n_right, n_branches), ...]`` — ``tx_io`` 兼容扩展。"""
    from .tx_io import read_tx_file

    groups = group_tx_dataset_by_obs_x(read_tx_file(path), x_tol=x_tol)
    return [
        (g.xobs, g.n_picks, g.n_left, g.n_right, g.n_branches) for g in groups
    ]


__all__ = [
    "X_MATCH_TOL",
    "TxObsGroup",
    "attach_station_ids",
    "count_lr_picks",
    "format_obs_catalog_label",
    "group_arrays_by_obs_x",
    "group_tx_dataset_by_obs_x",
    "nearest_station_id",
    "obs_groups_from_tx_path",
    "sort_obs_groups",
    "split_tx_dataset_by_obs_x",
    "unique_obs_from_tx_groups",
]
