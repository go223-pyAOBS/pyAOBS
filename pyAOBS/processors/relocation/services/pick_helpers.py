"""三分量共享拾取（无 Qt；对齐 zplotpy _get/_set/_remove_shared_pick）。"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple


def trace_group_key(trace_headers: list, trace_idx: int) -> Optional[Tuple[int, int]]:
    if trace_idx < 0 or trace_idx >= len(trace_headers):
        return None
    th = trace_headers[trace_idx]
    shot = int(getattr(th, "ishoti", 0) or 0)
    rec = int(getattr(th, "ireci", 0) or 0)
    if shot <= 0 or rec <= 0:
        return None
    return shot, rec


def trace_group_indices(trace_headers: list, trace_idx: int) -> List[int]:
    key = trace_group_key(trace_headers, int(trace_idx))
    if key is None:
        return [int(trace_idx)]
    out: List[int] = []
    for i, th in enumerate(trace_headers):
        if int(getattr(th, "ishoti", 0) or 0) == key[0] and int(getattr(th, "ireci", 0) or 0) == key[1]:
            out.append(int(i))
    return out if out else [int(trace_idx)]


def get_shared_pick(pick_manager: Any, trace_headers: list, trace_idx: int, pick_word: int) -> Optional[float]:
    if pick_manager is None:
        return None
    group = trace_group_indices(trace_headers, int(trace_idx))
    val = pick_manager.get_pick(int(trace_idx), int(pick_word))
    if val is not None:
        return float(val)
    for gi in group:
        v = pick_manager.get_pick(int(gi), int(pick_word))
        if v is not None:
            return float(v)
    return None


def set_shared_pick(
    pick_manager: Any, trace_headers: list, trace_idx: int, pick_word: int, t_pick: float
) -> bool:
    if pick_manager is None:
        return False
    group = trace_group_indices(trace_headers, int(trace_idx))
    target = int(trace_idx)
    for gi in group:
        if pick_manager.get_pick(int(gi), int(pick_word)) is not None:
            target = int(gi)
            break
    for gi in group:
        if int(gi) != target:
            pick_manager.remove_pick(int(gi), int(pick_word))
    return bool(pick_manager.add_pick(int(target), float(t_pick), int(pick_word)))


def remove_shared_pick(pick_manager: Any, trace_headers: list, trace_idx: int, pick_word: int) -> bool:
    if pick_manager is None:
        return False
    group = trace_group_indices(trace_headers, int(trace_idx))
    removed = False
    for gi in group:
        removed = bool(pick_manager.remove_pick(int(gi), int(pick_word))) or removed
    return removed
