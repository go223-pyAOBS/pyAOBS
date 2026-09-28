"""走时预览数据：ttimes.dat / tx.in → 统一拾取点；折合与 iphase 一致。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from pyAOBS.modeling.rayinvr.tx_io import TxDataset, read_tx_file


@dataclass(frozen=True)
class TtPick:
    """单条拾取，用于 T–X / 折合预览。"""

    shot_x: float
    rcv_x: float
    t: float
    u: float
    code: int  # ttimes: 0 折射 / 1 反射 / 2 水波 / 3 多次；tx.in: phase_id
    shot_idx: int
    source: str = ""  # 文件名标签（多文件时）


def reduce_traveltime(
    t: np.ndarray | float,
    offset: np.ndarray | float,
    vred: float,
) -> np.ndarray | float:
    """折合：``t' = t - |x-xobs|/vred``；``vred<=0`` 返回真走时（对齐 iphase）。"""
    if vred is None or float(vred) <= 0.0:
        return t
    return t - np.abs(offset) / float(vred)


def picks_to_arrays(picks: Sequence[TtPick]) -> dict[str, np.ndarray]:
    if not picks:
        return {
            "shot_x": np.zeros(0),
            "rcv_x": np.zeros(0),
            "offset": np.zeros(0),
            "t": np.zeros(0),
            "u": np.zeros(0),
            "code": np.zeros(0, dtype=np.int32),
            "shot_idx": np.zeros(0, dtype=np.int32),
        }
    shot_x = np.asarray([p.shot_x for p in picks], dtype=np.float64)
    rcv_x = np.asarray([p.rcv_x for p in picks], dtype=np.float64)
    return {
        "shot_x": shot_x,
        "rcv_x": rcv_x,
        "offset": rcv_x - shot_x,
        "t": np.asarray([p.t for p in picks], dtype=np.float64),
        "u": np.asarray([p.u for p in picks], dtype=np.float64),
        "code": np.asarray([p.code for p in picks], dtype=np.int32),
        "shot_idx": np.asarray([p.shot_idx for p in picks], dtype=np.int32),
    }


def group_obs_by_shot_x(
    arr: dict[str, np.ndarray],
    *,
    x_decimals: int = 3,
    stations: Sequence[tuple[int, float, float]] | None = None,
) -> list[dict]:
    """同一 OBS 左右两支（相同 ``shot_x``、不同 ``shot_idx``）合并为一条。

    ``tx.in`` 里一个台站常写成两个炮块（左支 / 右支），预览列表应按台站位置合并。
    """
    from pyAOBS.modeling.rayinvr.tx_obs_catalog import (
        attach_station_ids,
        group_arrays_by_obs_x,
    )

    if arr["t"].size == 0:
        return []
    tol = 10.0 ** (-int(x_decimals))
    groups = group_arrays_by_obs_x(
        arr["shot_x"],
        arr["rcv_x"],
        x_decimals=x_decimals,
        shot_idx=arr["shot_idx"],
    )
    attach_station_ids(groups, stations, tol=tol)
    out: list[dict] = []
    for g in groups:
        out.append(
            {
                "xobs": float(g.xobs),
                "shot_idxs": list(g.shot_idxs),
                "n": int(g.n_picks),
                "n_left": int(g.n_left),
                "n_right": int(g.n_right),
                "obs_id": g.obs_id,
            }
        )
    return out


def format_obs_catalog_label(rec: dict) -> str:
    """预览 / 转换共用：``OBS号  x=… km  (n=…)  L…/R…``。"""
    from pyAOBS.modeling.rayinvr.tx_obs_catalog import (
        format_obs_catalog_label as _fmt,
    )

    return _fmt(
        obs_id=rec.get("obs_id"),
        xobs=float(rec["xobs"]),
        n=int(rec["n"]),
        n_left=int(rec.get("n_left") or 0),
        n_right=int(rec.get("n_right") or 0),
    )


def build_obs_catalog(
    arr: dict[str, np.ndarray],
    stations: Sequence[tuple[int, float, float]] | None = None,
) -> list[dict]:
    """走时预览与 tx 转换共用的 OBS 列表（按台站号排序）。"""
    groups = group_obs_by_shot_x(arr, stations=stations)
    groups.sort(
        key=lambda g: (
            g.get("obs_id") is None,
            int(g["obs_id"]) if g.get("obs_id") is not None else 0,
            float(g["xobs"]),
        )
    )
    out: list[dict] = []
    for g in groups:
        rec = dict(g)
        rec["label"] = format_obs_catalog_label(rec)
        out.append(rec)
    return out


def phase_display_name(code: int, *, kind: str) -> str:
    """ttimes：0 折射 / 1 反射 / 2 直达水波 / 3 水柱多次；tx.in：震相号。"""
    iph = int(code)
    if kind == "ttimes":
        if iph == 0:
            return "折射"
        if iph == 1:
            return "反射"
        if iph == 2:
            return "直达水波"
        if iph == 3:
            return "水柱多次"
        if iph == 4:
            return "折射台侧多次"
        if iph == 5:
            return "反射台侧多次"
        return "其他"
    return ""


def format_phase_catalog_label(rec: dict, *, kind: str) -> str:
    iph = int(rec["code"])
    n = int(rec["n"])
    name = phase_display_name(iph, kind=kind)
    if name:
        return f"{iph}  {name}  (n={n})"
    return f"{iph}  (n={n})"


def build_phase_catalog(
    arr: dict[str, np.ndarray],
    *,
    kind: str = "tx",
) -> list[dict]:
    """预览窗震相列表：按 code 计数。"""
    codes = np.asarray(arr.get("code", []), dtype=np.int32)
    if codes.size == 0:
        return []
    out: list[dict] = []
    for ph in sorted(np.unique(codes)):
        iph = int(ph)
        rec = {
            "id": iph,
            "code": iph,
            "n": int(np.count_nonzero(codes == ph)),
        }
        rec["label"] = format_phase_catalog_label(rec, kind=kind)
        out.append(rec)
    return out


def load_ttimes_picks(path: str | Path) -> list[TtPick]:
    """解析 tomo2d ``ttimes.dat`` / ``geom.dat``（s/r 同构）。"""
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in raw.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    if not lines:
        raise ValueError(f"文件为空: {p}")
    head = lines[0].split()
    if not head:
        raise ValueError(f"首行为空: {p}")
    try:
        nsrc = int(float(head[0]))
    except (TypeError, ValueError) as e:
        raise ValueError(f"首行应为 nsrc，当前: {lines[0]!r}") from e
    if nsrc <= 0:
        raise ValueError(f"nsrc 无效: {nsrc}")

    picks: list[TtPick] = []
    idx = 1
    tag = p.name
    for ishot in range(nsrc):
        if idx >= len(lines):
            raise ValueError(f"炮 {ishot + 1}/{nsrc} 缺少 s 行")
        sp = lines[idx].split()
        idx += 1
        if len(sp) < 4 or sp[0] != "s":
            raise ValueError(f"期望 s 行，得到: {sp!r}")
        sx = float(sp[1])
        nrcv = int(float(sp[3]))
        for _ in range(nrcv):
            if idx >= len(lines):
                raise ValueError(f"炮 {ishot + 1} 的 r 行不足")
            rp = lines[idx].split()
            idx += 1
            if len(rp) < 6 or rp[0] != "r":
                raise ValueError(f"期望 r 行，得到: {rp!r}")
            rx = float(rp[1])
            code = int(float(rp[3]))
            t = float(rp[4])
            u = float(rp[5])
            picks.append(
                TtPick(
                    shot_x=sx,
                    rcv_x=rx,
                    t=t,
                    u=u,
                    code=code,
                    shot_idx=ishot,
                    source=tag,
                )
            )
    return picks


def tx_dataset_to_picks(ds: TxDataset, *, shot_offset: int = 0, source: str = "") -> list[TtPick]:
    out: list[TtPick] = []
    for i, shot in enumerate(ds.shots):
        sx = float(shot.xshot)
        for pk in shot.picks:
            out.append(
                TtPick(
                    shot_x=sx,
                    rcv_x=float(pk.x),
                    t=float(pk.t),
                    u=float(pk.u),
                    code=int(pk.phase_id),
                    shot_idx=shot_offset + i,
                    source=source,
                )
            )
    return out


def load_tx_in_picks(paths: Iterable[str | Path]) -> list[TtPick]:
    """读取一个或多个 ``tx.in``，炮号跨文件连续编号。"""
    picks: list[TtPick] = []
    shot_off = 0
    for raw in paths:
        p = Path(raw)
        ds = read_tx_file(p)
        part = tx_dataset_to_picks(ds, shot_offset=shot_off, source=p.name)
        picks.extend(part)
        shot_off += ds.n_shots
    if not picks:
        raise ValueError("未读到任何拾取点")
    return picks
