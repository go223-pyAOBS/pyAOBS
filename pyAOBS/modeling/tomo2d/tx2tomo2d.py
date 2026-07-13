# -*- coding: utf-8 -*-
"""
tx.in → tomo2d 走时/几何文件转换（Fortran ``tx2tomo2d.f`` 的 Python 实现）。

输出 ``ttimes.dat``（tt_inverse 的 data）与 ``geom.dat``（tt_forward 的 geom），
格式与原版 Zelt 程序一致（format 5 / 7）。

折射与反射拾取均写成 **r** 行；第三列整数 **0=折射、1=反射**（垂向坐标固定 0.01）。
**ttimes.dat** 中 r 行末两列为走时 **t** 与误差 **u**（来自 tx.in）；**geom.dat** 中对应 r 行的 t、u 恒为 0。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Set


@dataclass(frozen=True)
class TxConvertStats:
    nshot: int
    ntime: int
    station_rows: int
    data_path: str
    geom_path: str


def parse_phase_set(spec: str) -> Set[int]:
    """解析逗号或空白分隔的震相编号，如 ``1,2,3`` 或 ``11 12``。"""
    out: Set[int] = set()
    for tok in spec.replace(",", " ").split():
        t = tok.strip()
        if not t:
            continue
        out.add(int(float(t)))
    return out


def _fmt_s_line(x: float, z: float, npick: int) -> str:
    """FORMAT 7: ('s',2f10.3,i5)"""
    return f"s{x:10.3f}{z:10.3f}{npick:5d}"


def _fmt_r_line(x: float, z: float, kind: int, t: float, u: float) -> str:
    """FORMAT 5: ('r',2f10.3,i5,2f10.3)；折射 kind=0，反射 kind=1。"""
    return f"r{x:10.3f}{z:10.3f}{kind:5d}{t:10.3f}{u:10.3f}"


def _read_stations(path: Path) -> list[tuple[int, float, float]]:
    rows: list[tuple[int, float, float]] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"station 行至少需要 3 列 (ishot x z): {line!r}")
        ishot = int(float(parts[0]))
        x = float(parts[1])
        z = float(parts[2])
        rows.append((ishot, x, z))
    if not rows:
        raise ValueError(f"台站/炮点文件为空: {path}")
    return rows


def _iter_tx_records(path: Path) -> Iterable[tuple[float, float, float, int]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"tx.in 行至少需要 4 列 (x t u phase): {line!r}")
        yield float(parts[0]), float(parts[1]), float(parts[2]), int(float(parts[3]))


def convert_tx_in_to_tomo2d(
    station_path: Path | str,
    tx_in_path: Path | str,
    data_out_path: Path | str,
    geom_out_path: Path | str,
    *,
    refr_phases: Set[int],
    refl_phases: Set[int],
    x_match_tol: float = 0.001,
) -> TxConvertStats:
    """
    读取 ``station.lis`` 与 ``tx.in``，写出 ``ttimes.dat`` 与 ``geom.dat``。

    与 Fortran 一致：``i<0`` 结束；``i==0`` 为炮点分隔行，用 ``x`` 与台站表中 ``xshot``
    匹配（``abs(x-xshot)<=x_match_tol``）；``i>0`` 为拾取，按震相集合分别累积为折射/反射序列，
    写出时均为 **r** 行：折射第三列 0、反射第三列 1；**data** 写 ``t,u``，**geom** 写 ``0,0``。
    """
    station_path = Path(station_path)
    tx_in_path = Path(tx_in_path)
    data_out_path = Path(data_out_path)
    geom_out_path = Path(geom_out_path)

    stations = _read_stations(station_path)

    # ishot -> (折射拾取, 反射拾取)，保持插入顺序
    from collections import defaultdict

    refr: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    refl: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    npick: dict[int, int] = defaultdict(int)

    current_isw: int | None = None

    for x, t, u, iph in _iter_tx_records(tx_in_path):
        if iph < 0:
            break
        if iph == 0:
            current_isw = None
            for ishot_i, xs, _zs in stations:
                if abs(x - xs) <= x_match_tol:
                    current_isw = ishot_i
                    break
            continue
        if current_isw is None:
            # 与未初始化 isw 的 Fortran 行为相比，显式跳过并避免脏数据
            continue
        matched_line = False
        if iph in refr_phases:
            refr[current_isw].append((x, t, u))
            matched_line = True
        if iph in refl_phases:
            refl[current_isw].append((x, t, u))
            matched_line = True
        if matched_line:
            npick[current_isw] += 1

    z_pick = 0.01

    nshot = 0
    ntime = 0
    for ishot_i, _x, _z in stations:
        if npick[ishot_i] > 0:
            nshot += 1
            ntime += npick[ishot_i]

    data_lines: list[str] = [str(nshot)]
    geom_lines: list[str] = [str(nshot)]

    for ishot_i, xshot, zshot in stations:
        if npick[ishot_i] <= 0:
            continue
        npk = npick[ishot_i]
        zs = zshot + 0.01
        hdr = _fmt_s_line(xshot, zs, npk)
        data_lines.append(hdr)
        geom_lines.append(hdr)
        for px, pt, pu in refr[ishot_i]:
            data_lines.append(_fmt_r_line(px, z_pick, 0, pt, pu))
            geom_lines.append(_fmt_r_line(px, z_pick, 0, 0.0, 0.0))
        for px, pt, pu in refl[ishot_i]:
            data_lines.append(_fmt_r_line(px, z_pick, 1, pt, pu))
            geom_lines.append(_fmt_r_line(px, z_pick, 1, 0.0, 0.0))

    data_out_path.parent.mkdir(parents=True, exist_ok=True)
    geom_out_path.parent.mkdir(parents=True, exist_ok=True)
    data_out_path.write_text("\n".join(data_lines) + "\n", encoding="utf-8")
    geom_out_path.write_text("\n".join(geom_lines) + "\n", encoding="utf-8")

    return TxConvertStats(
        nshot=nshot,
        ntime=ntime,
        station_rows=len(stations),
        data_path=str(data_out_path),
        geom_path=str(geom_out_path),
    )
