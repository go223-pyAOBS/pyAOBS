# -*- coding: utf-8 -*-
"""道头复制/粘贴、批量填充、条件赋值、offset 检查。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .raw2sac_paths import ensure_raw2sac_on_path
from .segy_dataset import COLUMN_NAMES, SegyDataset

ensure_raw2sac_on_path()
from segy_trace_header import SEGY_TRACE_FIELDS, offset_m_from_xy  # type: ignore  # noqa: E402


def copy_headers(
    ds: SegyDataset,
    rows: Sequence[int],
    fields: Optional[Sequence[str]] = None,
) -> List[Dict[str, int]]:
    names = list(fields) if fields else list(COLUMN_NAMES)
    out: List[Dict[str, int]] = []
    for r in rows:
        th = ds.get_header(int(r))
        out.append({k: int(th.get(k, 0) or 0) for k in names if k in SEGY_TRACE_FIELDS})
    return out


def paste_headers(
    ds: SegyDataset,
    start_row: int,
    block: List[Dict[str, int]],
    fields: Optional[Sequence[str]] = None,
) -> int:
    """从 start_row 起粘贴；块比剩余道长则截断。返回写入行数。"""
    if not block:
        return 0
    names = list(fields) if fields else None
    n = 0
    for i, vals in enumerate(block):
        row = start_row + i
        if row >= ds.ntraces:
            break
        ds.set_header_row(row, vals, fields=names)
        n += 1
    return n


def fill_column(
    ds: SegyDataset,
    field: str,
    value: int,
    rows: Optional[Sequence[int]] = None,
) -> int:
    if field not in SEGY_TRACE_FIELDS:
        raise KeyError(field)
    targets = list(rows) if rows is not None else list(range(ds.ntraces))
    for r in targets:
        ds.set_header_value(int(r), field, int(value))
    return len(targets)


def batch_set_where(
    ds: SegyDataset,
    *,
    match_field: str,
    match_value: int,
    set_field: str,
    set_value: int,
) -> int:
    """当 match_field==match_value 时设置 set_field。"""
    if match_field not in SEGY_TRACE_FIELDS or set_field not in SEGY_TRACE_FIELDS:
        raise KeyError("field not in SEGY_TRACE_FIELDS")
    n = 0
    for i, th in enumerate(ds.headers):
        if int(th.get(match_field, 0) or 0) == int(match_value):
            ds.set_header_value(i, set_field, int(set_value))
            n += 1
    return n


def swap_fields(
    ds: SegyDataset,
    field_a: str,
    field_b: str,
    rows: Optional[Sequence[int]] = None,
) -> int:
    """逐道交换两个道头字段的值。返回修改道数。"""
    if field_a not in SEGY_TRACE_FIELDS or field_b not in SEGY_TRACE_FIELDS:
        raise KeyError("field not in SEGY_TRACE_FIELDS")
    if field_a == field_b:
        return 0
    targets = list(rows) if rows is not None else list(range(ds.ntraces))
    for r in targets:
        i = int(r)
        th = ds.get_header(i)
        va = int(th.get(field_a, 0) or 0)
        vb = int(th.get(field_b, 0) or 0)
        ds.set_header_value(i, field_a, vb)
        ds.set_header_value(i, field_b, va)
    return len(targets)


def check_offset_vs_xy(
    ds: SegyDataset,
    *,
    tol_m: float = 5.0,
    rows: Optional[Sequence[int]] = None,
    max_samples: int = 50,
) -> Dict[str, Any]:
    """
    道头 offset（米，不受 scalco）与 |hypot(sx-gx, sy-gy)|（scalco 后坐标）对比。

    比较用 |offset|（道头可带符号）。返回统计字典，含 mismatches 样本列表。
    """
    targets = list(rows) if rows is not None else list(range(ds.ntraces))
    ok = 0
    bad = 0
    missing_xy = 0
    zero_hdr = 0
    diffs: List[float] = []
    samples: List[str] = []
    max_diff = 0.0
    max_diff_row = -1

    for i in targets:
        th = ds.get_header(i)
        phy = ds.physical_xy(i)
        sx, sy, gx, gy = phy["sx"], phy["sy"], phy["gx"], phy["gy"]
        if abs(sx) + abs(sy) + abs(gx) + abs(gy) < 1e-9:
            missing_xy += 1
            continue
        hdr = int(th.get("offset", 0) or 0)
        hdr_abs = abs(hdr)
        if hdr_abs == 0:
            zero_hdr += 1
        xy = float(offset_m_from_xy(sx, sy, gx, gy))
        d = abs(float(hdr_abs) - xy)
        diffs.append(d)
        if d > max_diff:
            max_diff = d
            max_diff_row = i
        if d <= float(tol_m):
            ok += 1
        else:
            bad += 1
            if len(samples) < int(max_samples):
                samples.append(
                    f"trace={i}  offset={hdr} (|{hdr_abs}|)  "
                    f"xy={xy:.3f}  Δ={d:.3f}  "
                    f"sx,sy=({sx:.3f},{sy:.3f}) gx,gy=({gx:.3f},{gy:.3f})"
                )

    n = len(targets)
    mean_d = (sum(diffs) / len(diffs)) if diffs else 0.0
    # 众数 counit / scalco 提示
    counits = {}
    scalcos = {}
    for i in targets[: min(len(targets), 200)]:
        th = ds.get_header(i)
        c = int(th.get("counit", 0) or 0)
        s = int(th.get("scalco", 0) or 0)
        counits[c] = counits.get(c, 0) + 1
        scalcos[s] = scalcos.get(s, 0) + 1
    return {
        "n": n,
        "ok": ok,
        "bad": bad,
        "missing_xy": missing_xy,
        "zero_hdr": zero_hdr,
        "tol_m": float(tol_m),
        "max_diff": max_diff,
        "max_diff_row": max_diff_row,
        "mean_diff": mean_d,
        "samples": samples,
        "counits": counits,
        "scalcos": scalcos,
    }


def format_offset_check_report(result: Dict[str, Any]) -> str:
    """将 check_offset_vs_xy 结果格式化为可读报告。"""
    n = int(result.get("n", 0))
    ok = int(result.get("ok", 0))
    bad = int(result.get("bad", 0))
    tol = float(result.get("tol_m", 5.0))
    lines = [
        "【offset 检查】道头 offset vs hypot(sx-gx, sy-gy)（坐标已 scalco）",
        f"容差 tol = {tol:g} m（比较 |offset| 与 xy 距离）",
        f"检查道数 = {n}",
        f"一致 (Δ≤tol) = {ok}",
        f"不一致 (Δ>tol) = {bad}",
        f"坐标缺失/全零 = {result.get('missing_xy', 0)}",
        f"道头 offset=0 = {result.get('zero_hdr', 0)}",
        f"平均 |Δ| = {float(result.get('mean_diff', 0)):.3f} m",
        f"最大 |Δ| = {float(result.get('max_diff', 0)):.3f} m"
        + (
            f"  @trace={result.get('max_diff_row')}"
            if int(result.get("max_diff_row", -1)) >= 0
            else ""
        ),
    ]
    counits = result.get("counits") or {}
    scalcos = result.get("scalcos") or {}
    if counits:
        lines.append(
            "counit 分布: "
            + ", ".join(f"{k}:{v}" for k, v in sorted(counits.items()))
        )
    if scalcos:
        lines.append(
            "scalco 分布: "
            + ", ".join(f"{k}:{v}" for k, v in sorted(scalcos.items()))
        )
    if any(int(c) == 2 for c in counits):
        lines.append(
            "注意: counit=2 表示弧秒/地理坐标；若整型实为「度×|scalco|」，"
            "则 xy 距离单位不是米，Δ 会很大——需先统一坐标约定。"
        )
    samples = list(result.get("samples") or [])
    if samples:
        lines.append("--- 不一致样本 ---")
        lines.extend(samples)
        if bad > len(samples):
            lines.append(f"... 另有 {bad - len(samples)} 道未列出")
    elif bad == 0 and n > 0:
        lines.append("全部检查道通过。")
    return "\n".join(lines)


def clipboard_to_tsv(block: List[Dict[str, int]], fields: Sequence[str]) -> str:
    lines = ["\t".join(fields)]
    for row in block:
        lines.append("\t".join(str(int(row.get(f, 0) or 0)) for f in fields))
    return "\n".join(lines)


def tsv_to_block(text: str) -> Tuple[List[str], List[Dict[str, int]]]:
    """解析 TSV（可含表头）。返回 (fields, rows)。"""
    lines = [ln for ln in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n") if ln.strip()]
    if not lines:
        return [], []
    first = lines[0].split("\t")
    if all(tok in SEGY_TRACE_FIELDS or tok == "trace" for tok in first):
        fields = [t for t in first if t in SEGY_TRACE_FIELDS]
        data_lines = lines[1:]
    else:
        # 无表头：按列数对齐 COLUMN_NAMES 前 N 列，或单列
        ncol = len(first)
        fields = list(COLUMN_NAMES[:ncol]) if ncol > 1 else ["offset"]
        data_lines = lines
    rows: List[Dict[str, int]] = []
    for ln in data_lines:
        parts = ln.split("\t")
        d: Dict[str, int] = {}
        for i, name in enumerate(fields):
            if i >= len(parts):
                break
            try:
                d[name] = int(float(parts[i]))
            except ValueError:
                d[name] = 0
        if d:
            rows.append(d)
    return fields, rows
