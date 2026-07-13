"""
tx.in 文件读写

格式：format(3f10.3,i10)，每行 x, t, u, i
- i=0: 炮点头 (xshot, tshot, ushot)
- i>0: 拾取
- i=-1: 文件结束

参考 vedit/RAYINVR
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from .models import PhaseDataset, Pick, Shot


def read_tx(path: Union[str, Path]) -> PhaseDataset:
    """
    读取 tx.in 文件，返回 PhaseDataset。

    支持固定宽度 (3f10.3,i10) 和自由格式（空格分隔）解析。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"tx.in 文件不存在: {path}")

    dataset = PhaseDataset()
    current_shot: Shot | None = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip("\n\r")
            if not line or not line.strip():
                continue

            # 尝试固定宽度格式
            if len(line) >= 40:
                try:
                    x = float(line[0:10].strip() or 0)
                    t = float(line[10:20].strip() or 0)
                    u = float(line[20:30].strip() or 0)
                    i_val = int(line[30:40].strip() or 0)
                except (ValueError, IndexError):
                    # 回退到自由格式
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    x, t, u = float(parts[0]), float(parts[1]), float(parts[2])
                    i_val = int(parts[3])
            else:
                parts = line.split()
                if len(parts) < 4:
                    continue
                x, t, u = float(parts[0]), float(parts[1]), float(parts[2])
                i_val = int(parts[3])

            if i_val == -1:
                break
            if i_val == 0:
                current_shot = Shot(xshot=x, tshot=t, ushot=u)
                dataset.shots.append(current_shot)
                continue
            if i_val > 0:
                if current_shot is None:
                    current_shot = Shot(xshot=0.0, tshot=-1.0, ushot=0.0)
                    dataset.shots.append(current_shot)
                current_shot.add_pick(x, t, u, i_val)

    return dataset


def write_tx(dataset: PhaseDataset, path: Union[str, Path]) -> None:
    """
    将 PhaseDataset 写入 tx.in 文件。

    使用 format(3f10.3,i10) 保持与 Fortran 工具兼容。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for shot in dataset.shots:
        lines.append(f"{shot.xshot:10.3f}{shot.tshot:10.3f}{shot.ushot:10.3f}{0:10d}")
        for p in shot.picks:
            lines.append(f"{p.x:10.3f}{p.t:10.3f}{p.u:10.3f}{p.phase_id:10d}")
    lines.append(f"{0.0:10.3f}{0.0:10.3f}{0.0:10.3f}{-1:10d}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
