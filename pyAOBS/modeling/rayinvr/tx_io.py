"""RAYINVR ``tx.in`` / ``tx.out`` 统一读写。

格式：``format(3f10.3,i10)`` — 每行 ``x, t, u, i``

- ``i == 0``（或其它 ``i <= 0`` 且非 ``-1``）：炮点头；第二列 ``t=±1``、``u=0`` 时常表示
  OBS 左(``-1``)/右(``+1``) 支，**同一 x 仅对应一个 OBS**（左右各一块，拾取需合并）
- ``i > 0``：观测/理论走时点（震相号）
- ``i == -1``：文件结束

供 ``vedit`` / ``zplotpy`` / ``iphase`` / ``tomo2d`` 共用，避免多套解析分叉。

OBS 列表合并与标签格式见 ``rayinvr.tx_obs_catalog``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

PathLike = Union[str, Path]


@dataclass
class TxPick:
    x: float
    t: float
    u: float
    phase_id: int

    def as_tuple(self) -> Tuple[float, float, float, int]:
        return (self.x, self.t, self.u, self.phase_id)


@dataclass
class TxShotBlock:
    xshot: float
    tshot: float = -1.0
    ushot: float = 0.0
    picks: list[TxPick] = field(default_factory=list)

    def add_pick(self, x: float, t: float, u: float, phase_id: int) -> None:
        self.picks.append(TxPick(x=float(x), t=float(t), u=float(u), phase_id=int(phase_id)))


@dataclass
class TxDataset:
    """按炮分组的 tx 数据。"""

    shots: list[TxShotBlock] = field(default_factory=list)

    @property
    def n_shots(self) -> int:
        return len(self.shots)

    @property
    def n_picks(self) -> int:
        return sum(len(s.picks) for s in self.shots)

    def to_iphase_by_shot(self) -> list[dict]:
        """``theory2d_service`` 兼容：``[{shot_x, obs:[(x,t,u,ipf),...]}, ...]``。"""
        return [
            {
                "shot_x": float(s.xshot),
                "obs": [p.as_tuple() for p in s.picks],
            }
            for s in self.shots
        ]

    def to_zplot_dict(self) -> dict:
        """``TheoreticalTravelTimeCalculator`` 兼容字典。"""
        return {
            "shots": [
                {
                    "shot_position": float(s.xshot),
                    "observations": [
                        {
                            "x": float(p.x),
                            "t": float(p.t),
                            "u": float(p.u),
                            "phase": int(p.phase_id),
                        }
                        for p in s.picks
                    ],
                }
                for s in self.shots
            ],
            "total_observations": self.n_picks,
        }

    def to_flat_arrays(self) -> Optional[dict]:
        """扁平数组，形如 ``RayinvrWrapper.get_observed_data()``。"""
        import numpy as np

        xs: list[float] = []
        ts: list[float] = []
        us: list[float] = []
        ph: list[int] = []
        for s in self.shots:
            for p in s.picks:
                xs.append(p.x)
                ts.append(p.t)
                us.append(p.u)
                ph.append(p.phase_id)
        if not xs:
            return None
        return {
            "x": np.asarray(xs, dtype=np.float32),
            "t": np.asarray(ts, dtype=np.float32),
            "u": np.asarray(us, dtype=np.float32),
            "phase": np.asarray(ph, dtype=np.int32),
        }


def format_tx_line(x: float, t: float, u: float, i_val: int) -> str:
    """单行 ``format(3f10.3,i10)``。"""
    return f"{float(x):10.3f}{float(t):10.3f}{float(u):10.3f}{int(i_val):10d}"


def parse_tx_line(raw: str) -> Optional[Tuple[float, float, float, int]]:
    """解析一行；固定列宽优先，失败则空格分隔。空行/非法返回 None。"""
    if not raw or not raw.strip():
        return None
    x = t = u = None
    i_val: Optional[int] = None
    if len(raw) >= 40:
        try:
            x = float(raw[0:10].strip() or 0)
            t = float(raw[10:20].strip() or 0)
            u = float(raw[20:30].strip() or 0)
            i_val = int(raw[30:40].strip() or 0)
        except (ValueError, IndexError):
            x = t = u = i_val = None
    if x is None or t is None or i_val is None:
        parts = raw.split()
        if len(parts) < 4:
            return None
        try:
            x = float(parts[0])
            t = float(parts[1])
            u = float(parts[2])
            i_val = int(float(parts[3]))  # 允许 1.0
        except (ValueError, IndexError):
            return None
    if u is None:
        u = 0.0
    return float(x), float(t), float(u), int(i_val)


def read_tx_file(path: PathLike) -> TxDataset:
    """读取 ``tx.in`` / ``tx.out``。文件不存在则抛 ``FileNotFoundError``。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"tx 文件不存在: {p}")

    ds = TxDataset()
    current: Optional[TxShotBlock] = None

    with open(p, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = parse_tx_line(line.rstrip("\n\r"))
            if parsed is None:
                continue
            x, t, u, i_val = parsed
            if i_val == -1:
                break
            if i_val <= 0:
                current = TxShotBlock(xshot=x, tshot=t, ushot=u)
                ds.shots.append(current)
                continue
            if current is None:
                current = TxShotBlock(xshot=0.0, tshot=-1.0, ushot=0.0)
                ds.shots.append(current)
            current.add_pick(x, t, u, i_val)

    return ds


def shot_branch_side(shot: TxShotBlock) -> int:
    """``i=0`` 炮头：``t=±1`` 且 ``u≈0`` 表示 OBS 左(-1)/右(+1) 支；否则 0。"""
    t = float(shot.tshot)
    u = float(shot.ushot)
    if abs(abs(t) - 1.0) <= 1.0e-6 and abs(u) <= 1.0e-6:
        return -1 if t < 0 else 1
    return 0


def group_shots_by_obs_x(
    dataset: TxDataset,
    *,
    x_tol: float = 0.001,
) -> list[tuple[float, list[TxShotBlock]]]:
    """按 OBS 位置合并 tx 炮块（同 ``xshot`` 的 L/R 支并为一个 OBS）。"""
    groups: list[tuple[float, list[TxShotBlock]]] = []
    for shot in dataset.shots:
        x = float(shot.xshot)
        placed = False
        for i, (gx, blocks) in enumerate(groups):
            if abs(gx - x) <= x_tol:
                blocks.append(shot)
                placed = True
                break
        if not placed:
            groups.append((x, [shot]))
    groups.sort(key=lambda item: item[0])
    return groups


def unique_obs_from_tx(
    path: PathLike,
    *,
    x_tol: float = 0.001,
) -> list[tuple[float, int, int]]:
    """``[(obs_x, n_picks, n_branches), ...]``，已合并 L/R 炮块。"""
    from .tx_obs_catalog import group_tx_dataset_by_obs_x

    groups = group_tx_dataset_by_obs_x(read_tx_file(path), x_tol=x_tol)
    return [(g.xobs, g.n_picks, g.n_branches) for g in groups]


def filter_tx_dataset_by_shot_xs(
    dataset: TxDataset,
    shot_xs: Iterable[float],
    *,
    x_tol: float = 0.001,
) -> TxDataset:
    """仅保留炮头 ``xshot`` 落在 *shot_xs* 内的块（含 L/R 双支）。"""
    xs = [float(x) for x in shot_xs]
    if not xs:
        return dataset
    shots: list[TxShotBlock] = []
    for shot in dataset.shots:
        sx = float(shot.xshot)
        if any(abs(sx - ox) <= x_tol for ox in xs):
            shots.append(shot)
    return TxDataset(shots=shots)


def write_tx_file(dataset: TxDataset, path: PathLike) -> Path:
    """写入固定宽度 tx 文件，末尾带 ``i=-1`` 结束行。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for shot in dataset.shots:
        lines.append(format_tx_line(shot.xshot, shot.tshot, shot.ushot, 0))
        for pick in shot.picks:
            lines.append(format_tx_line(pick.x, pick.t, pick.u, pick.phase_id))
    lines.append(format_tx_line(0.0, 0.0, 0.0, -1))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def write_tx_from_picks(
    path: PathLike,
    *,
    shot_x: float,
    picks: Iterable[Tuple[float, float, float, int]],
    shot_t: float = -1.0,
    shot_u: float = 0.0,
) -> Path:
    """便捷写单炮 tx：``picks`` 为 ``(x, t, u, phase_id)``。"""
    shot = TxShotBlock(xshot=float(shot_x), tshot=float(shot_t), ushot=float(shot_u))
    for x, t, u, phase_id in picks:
        shot.add_pick(x, t, u, phase_id)
    return write_tx_file(TxDataset(shots=[shot]), path)


def validate_tx_file(path: PathLike) -> Tuple[bool, Optional[str]]:
    """校验可读且至少有一个观测点。"""
    p = Path(path)
    if not p.exists():
        return False, f"文件不存在: {p}"
    try:
        ds = read_tx_file(p)
    except Exception as exc:
        return False, f"解析失败: {exc}"
    if ds.n_picks == 0:
        return False, "文件中没有观测走时数据"
    return True, None


def parse_tx_file_by_shot(path: PathLike) -> list[dict]:
    """iphase ``theory2d`` 兼容别名。"""
    return read_tx_file(path).to_iphase_by_shot()


__all__ = [
    "TxPick",
    "TxShotBlock",
    "TxDataset",
    "filter_tx_dataset_by_shot_xs",
    "format_tx_line",
    "parse_tx_line",
    "parse_tx_file_by_shot",
    "read_tx_file",
    "group_shots_by_obs_x",
    "shot_branch_side",
    "unique_obs_from_tx",
    "validate_tx_file",
    "write_tx_file",
    "write_tx_from_picks",
]
