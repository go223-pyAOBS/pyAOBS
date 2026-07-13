from __future__ import annotations

from collections import deque
from typing import Iterable, List, Tuple

import numpy as np


def _neighbor_offsets(bwconn: int) -> List[Tuple[int, int]]:
    if int(bwconn) == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]


def _binary_dilate(mask: np.ndarray, *, steps: int, bwconn: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    if int(steps) <= 0:
        return out
    offs = _neighbor_offsets(int(bwconn))
    n0, n1 = out.shape
    for _ in range(int(steps)):
        src = out
        dst = src.copy()
        for di, dj in offs:
            i0 = max(0, -di)
            i1 = min(n0, n0 - di)
            j0 = max(0, -dj)
            j1 = min(n1, n1 - dj)
            if i0 >= i1 or j0 >= j1:
                continue
            dst[i0 + di:i1 + di, j0 + dj:j1 + dj] |= src[i0:i1, j0:j1]
        out = dst
    return out


def _keep_large_components(seed: np.ndarray, *, min_area: int, bwconn: int) -> np.ndarray:
    m = np.asarray(seed, dtype=bool)
    n0, n1 = m.shape
    if not np.any(m):
        return np.zeros_like(m, dtype=bool)
    visited = np.zeros_like(m, dtype=bool)
    keep = np.zeros_like(m, dtype=bool)
    offs = _neighbor_offsets(int(bwconn))
    min_sz = int(max(1, min_area))

    true_pos = np.argwhere(m)
    for p in true_pos:
        i = int(p[0])
        j = int(p[1])
        if visited[i, j]:
            continue
        q: deque[Tuple[int, int]] = deque()
        q.append((i, j))
        visited[i, j] = True
        comp: List[Tuple[int, int]] = []
        while q:
            ci, cj = q.popleft()
            comp.append((ci, cj))
            for di, dj in offs:
                ni = ci + di
                nj = cj + dj
                if ni < 0 or ni >= n0 or nj < 0 or nj >= n1:
                    continue
                if visited[ni, nj] or (not m[ni, nj]):
                    continue
                visited[ni, nj] = True
                q.append((ni, nj))
        if len(comp) >= min_sz:
            for ci, cj in comp:
                keep[ci, cj] = True
    return keep


def build_mask_from_tf(
    tf: np.ndarray,
    freq: np.ndarray | None = None,
    *,
    bwconn: int = 8,
    amp_quantile: float = 0.70,
    floor_ratio: float = 0.03,
    min_area: int = 24,
    expand_steps: int = 1,
    keep_strong_quantile: float = 0.95,
) -> np.ndarray:
    """Build morphology-aware mask for TF coefficients."""
    X = np.asarray(tf)
    if X.ndim != 2 or X.size == 0:
        return np.ones_like(X, dtype=np.float64)

    A = np.abs(X).astype(np.float64, copy=False)
    if not np.any(np.isfinite(A)):
        return np.ones_like(A, dtype=np.float64)
    A = np.nan_to_num(A, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    amax = float(np.max(A))
    if amax <= 0.0:
        return np.ones_like(A, dtype=np.float64)

    q = float(min(0.99, max(0.05, amp_quantile)))
    q_strong = float(min(0.999, max(q, keep_strong_quantile)))
    thr = max(float(np.quantile(A, q)), float(max(0.0, floor_ratio)) * amax)
    seed = A >= thr
    if not np.any(seed):
        return np.ones_like(A, dtype=np.float64)

    keep = _keep_large_components(seed, min_area=int(max(1, min_area)), bwconn=int(bwconn))
    if int(expand_steps) > 0 and np.any(keep):
        keep = _binary_dilate(keep, steps=int(expand_steps), bwconn=int(bwconn))

    # Always retain very strong bins to avoid over-suppressing isolated true events.
    strong = A >= float(np.quantile(A, q_strong))
    mask = keep | strong

    # Safety fallback: if mask becomes too sparse, disable morphology this round.
    keep_ratio = float(np.mean(mask))
    if keep_ratio < 0.005:
        return np.ones_like(A, dtype=np.float64)

    return mask.astype(np.float64, copy=False)

