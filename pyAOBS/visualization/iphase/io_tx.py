"""
tx.in 文件读写

格式：format(3f10.3,i10)，每行 x, t, u, i
- i=0: 炮点头 (xshot, tshot, ushot)
- i>0: 拾取
- i=-1: 文件结束

底层解析委托 ``pyAOBS.modeling.rayinvr.tx_io``，本模块仅做 PhaseDataset 适配。
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from pyAOBS.modeling.rayinvr.tx_io import (
    TxDataset,
    TxShotBlock,
    read_tx_file,
    write_tx_file,
)

from .models import PhaseDataset, Pick, Shot


def _to_phase_dataset(raw: TxDataset) -> PhaseDataset:
    dataset = PhaseDataset()
    for s in raw.shots:
        shot = Shot(xshot=s.xshot, tshot=s.tshot, ushot=s.ushot)
        for p in s.picks:
            shot.picks.append(
                Pick(x=p.x, t=p.t, u=p.u, phase_id=p.phase_id)
            )
        dataset.shots.append(shot)
    return dataset


def _from_phase_dataset(dataset: PhaseDataset) -> TxDataset:
    out = TxDataset()
    for s in dataset.shots:
        block = TxShotBlock(xshot=s.xshot, tshot=s.tshot, ushot=s.ushot)
        for p in s.picks:
            block.add_pick(p.x, p.t, p.u, p.phase_id)
        out.shots.append(block)
    return out


def phase_dataset_to_tx(dataset: PhaseDataset) -> TxDataset:
    """``PhaseDataset`` → ``TxDataset``（供 ``tx_obs_catalog`` 等共用逻辑）。"""
    return _from_phase_dataset(dataset)


def tx_dataset_to_phase(raw: TxDataset) -> PhaseDataset:
    """``TxDataset`` → ``PhaseDataset``。"""
    return _to_phase_dataset(raw)


def read_tx(path: Union[str, Path]) -> PhaseDataset:
    """
    读取 tx.in 文件，返回 PhaseDataset。

    支持固定宽度 (3f10.3,i10) 和自由格式（空格分隔）解析。
    """
    return _to_phase_dataset(read_tx_file(path))


def write_tx(dataset: PhaseDataset, path: Union[str, Path]) -> None:
    """
    将 PhaseDataset 写入 tx.in 文件。

    使用 format(3f10.3,i10) 保持与 Fortran 工具兼容。
    """
    write_tx_file(_from_phase_dataset(dataset), path)
