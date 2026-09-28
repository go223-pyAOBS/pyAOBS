# -*- coding: utf-8 -*-
"""
tx.in → tomo2d 走时/几何文件转换（Fortran ``tx2tomo2d.f`` 的 Python 实现）。

输出 ``ttimes.dat``（tt_inverse 的 data）与 ``geom.dat``（tt_forward 的 geom），
格式与原版 Zelt 程序一致（format 5 / 7）。

支持**多个 tx.in** 合并写入同一对输出文件（共用一份 station.lis；按炮号累积拾取）。

折射与反射拾取均写成 **r** 行；第三列整数 **0=折射、1=反射**（垂向坐标固定 0.01）。
**ttimes.dat** 中 r 行末两列为走时 **t** 与误差 **u**（来自 tx.in）；**geom.dat** 中对应 r 行的 t、u 恒为 0。
可选 ``water_phases`` / ``mult_phases`` 映射为 raytype 2 / 3，
``refr_mult_phases`` / ``refl_mult_phases`` 映射为 4 / 5，
``psp_phases`` 映射为 6（折合 PSP；默认空，不影响原 0/1 输出）。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Set


@dataclass(frozen=True)
class TxConvertStats:
    nshot: int
    ntime: int
    station_rows: int
    data_path: str
    geom_path: str
    n_tx_files: int = 1


@dataclass(frozen=True)
class TxObsSummary:
    """station.lis 一台站在 tx.in 中的匹配摘要。"""

    obs_id: int
    x: float
    z: float
    n_picks: int = 0
    n_blocks: int = 0


def parse_phase_set(spec: str) -> Set[int]:
    """解析逗号或空白分隔的震相编号，如 ``1,2,3`` 或 ``11 12``。"""
    out: Set[int] = set()
    for tok in spec.replace(",", " ").split():
        t = tok.strip()
        if not t:
            continue
        out.add(int(float(t)))
    return out


def parse_obs_id_spec(spec: str | Sequence[int] | None) -> set[int] | None:
    """解析要转换的 OBS 号。

    ``None`` 表示不过滤（全部）。字符串空 / ``all`` / ``*`` 同义；
    ``none`` / ``-`` 表示空选择；其余按逗号或空白分隔的整数。
    """
    if spec is None:
        return None
    if isinstance(spec, (list, tuple, set, frozenset)):
        return {int(x) for x in spec}
    s = str(spec).strip()
    if not s or s.lower() in ("all", "*", "全部"):
        return None
    if s.lower() in ("none", "-", "无"):
        return set()
    return parse_phase_set(s)


def format_obs_id_spec(ids: Sequence[int] | None, *, catalog_ids: Sequence[int] | None = None) -> str:
    """把 OBS 选择写成表单字符串（``all`` / ``none`` / ``12,30``）。"""
    if ids is None:
        return "all"
    wanted = {int(x) for x in ids}
    if not wanted:
        return "none"
    if catalog_ids is not None and wanted == {int(x) for x in catalog_ids}:
        return "all"
    return ",".join(str(i) for i in sorted(wanted))


def parse_tx_in_list(spec: str | Path | Sequence[str | Path] | None) -> list[str]:
    """
    解析一个或多个 tx.in 路径。

    接受：单路径字符串、``Path``、路径序列，或用换行 / ``;`` / ``|`` 分隔的多路径文本。
    """
    if spec is None:
        return []
    if isinstance(spec, (str, Path)):
        raw = str(spec).strip()
        if not raw:
            return []
        # 单路径且无分隔符
        if "\n" not in raw and ";" not in raw and "|" not in raw:
            return [raw.replace("\\", "/")]
        parts: list[str] = []
        for chunk in raw.replace("|", "\n").replace(";", "\n").splitlines():
            s = chunk.strip().strip('"').strip("'")
            if s:
                parts.append(s.replace("\\", "/"))
        return parts
    out: list[str] = []
    for item in spec:
        out.extend(parse_tx_in_list(item))
    return out


def _fmt_s_line(x: float, z: float, npick: int) -> str:
    """FORMAT 7: ('s',2f10.3,i5)"""
    return f"s{x:10.3f}{z:10.3f}{npick:5d}"


def _fmt_r_line(x: float, z: float, kind: int, t: float, u: float) -> str:
    """FORMAT 5: ('r',2f10.3,i5,2f10.3)；0=折射，1=反射，2=直达水波，3=水柱多次，4=折射台侧，5=反射台侧。"""
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


def read_station_lis(path: Path | str) -> list[tuple[int, float, float]]:
    """读取 ``station.lis``：每行 ``ishot x z``（OBS 号、模型距离、深度，km）。"""
    return _read_stations(Path(path))


def _iter_tx_records(path: Path) -> Iterable[tuple[float, float, float, int]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"tx.in 行至少需要 4 列 (x t u phase): {line!r} @ {path}")
        yield float(parts[0]), float(parts[1]), float(parts[2]), int(float(parts[3]))


def _match_station_id(
    x: float,
    stations: Sequence[tuple[int, float, float]],
    x_match_tol: float,
) -> int | None:
    for ishot_i, xs, _zs in stations:
        if abs(x - xs) <= x_match_tol:
            return int(ishot_i)
    return None


def list_obs_in_tx(
    station_path: Path | str,
    tx_in_path: Path | str | Sequence[Path | str],
    *,
    x_match_tol: float = 0.001,
) -> tuple[list[TxObsSummary], int]:
    """根据 ``station.lis`` 与 tx.in 生成 OBS 摘要（台站表顺序）。

    返回 ``(rows, unmatched_blocks)``：``unmatched_blocks`` 为 phase=0 炮头
    未能匹配任何台站 x 的块数。``n_picks`` 计该 OBS 下全部 ``phase>0`` 行。
    """
    stations = _read_stations(Path(station_path))
    tx_paths = [Path(p) for p in parse_tx_in_list(tx_in_path)]
    n_picks: dict[int, int] = defaultdict(int)
    n_blocks: dict[int, int] = defaultdict(int)
    unmatched = 0
    for txp in tx_paths:
        if not txp.is_file():
            continue
        current: int | None = None
        for x, _t, _u, iph in _iter_tx_records(txp):
            if iph < 0:
                break
            if iph == 0:
                current = _match_station_id(x, stations, x_match_tol)
                if current is None:
                    unmatched += 1
                else:
                    n_blocks[current] += 1
                continue
            if current is not None:
                n_picks[current] += 1
    rows = [
        TxObsSummary(
            obs_id=int(oid),
            x=float(sx),
            z=float(sz),
            n_picks=int(n_picks.get(int(oid), 0)),
            n_blocks=int(n_blocks.get(int(oid), 0)),
        )
        for oid, sx, sz in stations
    ]
    return rows, unmatched


def _accumulate_tx_file(
    tx_in_path: Path,
    stations: list[tuple[int, float, float]],
    *,
    refr_phases: Set[int],
    refl_phases: Set[int],
    water_phases: Set[int] | None = None,
    mult_phases: Set[int] | None = None,
    refr_mult_phases: Set[int] | None = None,
    refl_mult_phases: Set[int] | None = None,
    psp_phases: Set[int] | None = None,
    x_match_tol: float,
    refr: dict[int, list[tuple[float, float, float]]],
    refl: dict[int, list[tuple[float, float, float]]],
    water: dict[int, list[tuple[float, float, float]]],
    mult: dict[int, list[tuple[float, float, float]]],
    refr_mult: dict[int, list[tuple[float, float, float]]],
    refl_mult: dict[int, list[tuple[float, float, float]]],
    psp: dict[int, list[tuple[float, float, float]]],
    npick: dict[int, int],
) -> None:
    water_phases = water_phases or set()
    mult_phases = mult_phases or set()
    refr_mult_phases = refr_mult_phases or set()
    refl_mult_phases = refl_mult_phases or set()
    psp_phases = psp_phases or set()
    current_isw: int | None = None
    for x, t, u, iph in _iter_tx_records(tx_in_path):
        if iph < 0:
            break
        if iph == 0:
            current_isw = _match_station_id(x, stations, x_match_tol)
            continue
        if current_isw is None:
            continue
        matched_line = False
        if iph in refr_phases:
            refr[current_isw].append((x, t, u))
            matched_line = True
        if iph in refl_phases:
            refl[current_isw].append((x, t, u))
            matched_line = True
        if iph in water_phases:
            water[current_isw].append((x, t, u))
            matched_line = True
        if iph in mult_phases:
            mult[current_isw].append((x, t, u))
            matched_line = True
        if iph in refr_mult_phases:
            refr_mult[current_isw].append((x, t, u))
            matched_line = True
        if iph in refl_mult_phases:
            refl_mult[current_isw].append((x, t, u))
            matched_line = True
        if iph in psp_phases:
            psp[current_isw].append((x, t, u))
            matched_line = True
        if matched_line:
            npick[current_isw] += 1


def convert_tx_in_to_tomo2d(
    station_path: Path | str,
    tx_in_path: Path | str | Sequence[Path | str],
    data_out_path: Path | str,
    geom_out_path: Path | str,
    *,
    refr_phases: Set[int],
    refl_phases: Set[int],
    water_phases: Set[int] | None = None,
    mult_phases: Set[int] | None = None,
    refr_mult_phases: Set[int] | None = None,
    refl_mult_phases: Set[int] | None = None,
    psp_phases: Set[int] | None = None,
    x_match_tol: float = 0.001,
    include_obs: Sequence[int] | None = None,
) -> TxConvertStats:
    """
    读取 ``station.lis`` 与一个或多个 ``tx.in``，写出 ``ttimes.dat`` 与 ``geom.dat``。

    多文件时按文件顺序累积到同一炮号下；共用一份台站表。
    与 Fortran 一致：``i<0`` 结束；``i==0`` 为炮点分隔行，用 ``x`` 与台站表中 ``xshot``
    匹配（``abs(x-xshot)<=x_match_tol``）；``i>0`` 为拾取，按震相集合分别累积为折射/反射序列。
    ``include_obs`` 为 ``None`` 时转换全部台站；否则只保留所列 OBS 号（station.lis 第一列）。
    """
    station_path = Path(station_path)
    data_out_path = Path(data_out_path)
    geom_out_path = Path(geom_out_path)

    tx_paths = [Path(p) for p in parse_tx_in_list(tx_in_path)]
    if not tx_paths:
        raise ValueError("至少需要一个 tx.in 路径")
    for p in tx_paths:
        if not p.is_file():
            raise FileNotFoundError(f"tx.in 不存在: {p}")

    stations = _read_stations(station_path)
    if include_obs is not None:
        wanted = {int(i) for i in include_obs}
        if not wanted:
            raise ValueError("未选择任何 OBS")
        stations = [s for s in stations if int(s[0]) in wanted]
        if not stations:
            raise ValueError("所选 OBS 在 station.lis 中均不存在")

    refr: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    refl: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    water: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    mult: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    refr_mult: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    refl_mult: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    psp: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    npick: dict[int, int] = defaultdict(int)

    for txp in tx_paths:
        _accumulate_tx_file(
            txp,
            stations,
            refr_phases=refr_phases,
            refl_phases=refl_phases,
            water_phases=water_phases,
            mult_phases=mult_phases,
            refr_mult_phases=refr_mult_phases,
            refl_mult_phases=refl_mult_phases,
            psp_phases=psp_phases,
            x_match_tol=x_match_tol,
            refr=refr,
            refl=refl,
            water=water,
            mult=mult,
            refr_mult=refr_mult,
            refl_mult=refl_mult,
            psp=psp,
            npick=npick,
        )

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
        for px, pt, pu in water[ishot_i]:
            data_lines.append(_fmt_r_line(px, z_pick, 2, pt, pu))
            geom_lines.append(_fmt_r_line(px, z_pick, 2, 0.0, 0.0))
        for px, pt, pu in mult[ishot_i]:
            data_lines.append(_fmt_r_line(px, z_pick, 3, pt, pu))
            geom_lines.append(_fmt_r_line(px, z_pick, 3, 0.0, 0.0))
        for px, pt, pu in refr_mult[ishot_i]:
            data_lines.append(_fmt_r_line(px, z_pick, 4, pt, pu))
            geom_lines.append(_fmt_r_line(px, z_pick, 4, 0.0, 0.0))
        for px, pt, pu in refl_mult[ishot_i]:
            data_lines.append(_fmt_r_line(px, z_pick, 5, pt, pu))
            geom_lines.append(_fmt_r_line(px, z_pick, 5, 0.0, 0.0))
        for px, pt, pu in psp[ishot_i]:
            data_lines.append(_fmt_r_line(px, z_pick, 6, pt, pu))
            geom_lines.append(_fmt_r_line(px, z_pick, 6, 0.0, 0.0))

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
        n_tx_files=len(tx_paths),
    )
