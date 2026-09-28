# -*- coding: utf-8 -*-
"""从 Fortran 内存收集射线路径，保证各 TRAPAR 炮点均有代表射线。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

RAYINVR_STORED_RAY_CAP = 24000  # prayt = pshot2 * prayf (rayinvr.par)
X_SHOT_TOL = 0.001
# vedit 显示：每个 TRAPAR 炮点（台站）背景射线配额（均匀抽样）
RAYS_PER_SHOT_DEFAULT = 24
# 多震相时每 ivray 至少显示的条数（与 theory_ray_link.THEORY_CURVE_DISPLAY_RAYS_MIN 一致）
RAYS_PER_SHOT_MIN_PER_PHASE = 5
# collect_stored_rays：``max_rays_per_shot=-1`` 表示每炮不抽稀，保留全部
RAYS_PER_SHOT_UNLIMITED = -1


def _even_pick(items: Sequence[dict], n: int) -> List[dict]:
    """从列表中均匀抽取至多 ``n`` 条（保留首、尾与中间分布）。"""
    pool = list(items)
    if n <= 0 or not pool:
        return []
    if len(pool) <= n:
        return pool
    if n == 1:
        return [pool[0]]
    idxs: List[int] = []
    seen: set[int] = set()
    for i in range(n):
        j = int(round(i * (len(pool) - 1) / (n - 1)))
        if j not in seen:
            seen.add(j)
            idxs.append(j)
    return [pool[j] for j in idxs]


def limit_rays_per_shot(
    rays: Sequence[dict],
    shot_xs: Sequence[float] | None = None,
    *,
    max_per_shot: int = RAYS_PER_SHOT_DEFAULT,
    x_tol: float = X_SHOT_TOL,
) -> List[dict]:
    """按炮点均匀抽样，限制每个台站显示的射线条数。"""
    if max_per_shot <= 0:
        return list(rays or [])
    items = list(rays or [])
    if not items:
        return []

    targets = [float(x) for x in (shot_xs or [])]
    if targets:
        buckets: dict[float, List[dict]] = {sx: [] for sx in targets}
        for r in items:
            hit = infer_ray_shot_x(r, targets, tol=x_tol)
            if hit is not None:
                buckets[hit].append(r)
        out: List[dict] = []
        for sx in targets:
            out.extend(_even_pick(buckets[sx], max_per_shot))
        return out

    by_x: dict[float, List[dict]] = {}
    for r in items:
        x0 = ray_origin_x(r)
        if x0 is None:
            continue
        key = round(float(x0) / max(x_tol, 1e-9)) * max(x_tol, 1e-9)
        by_x.setdefault(key, []).append(r)
    out = []
    for pool in by_x.values():
        out.extend(_even_pick(pool, max_per_shot))
    return out


def _ray_phase_id(ray: dict) -> int:
    try:
        return int(ray.get("phase_id", ray.get("ivray", 0)) or 0)
    except (TypeError, ValueError):
        return 0


def _limit_one_shot_stratified(
    shot_rays: Sequence[dict],
    *,
    max_per_shot: int,
    min_per_phase: int = RAYS_PER_SHOT_MIN_PER_PHASE,
) -> List[dict]:
    """单炮内多震相分层抽样，各震相至少 ``min_per_phase`` 条（若池中有）。"""
    pool = list(shot_rays or [])
    if max_per_shot <= 0 or not pool:
        return []
    min_ph = max(1, int(min_per_phase))
    phases = sorted({p for p in (_ray_phase_id(r) for r in pool) if p > 0})
    if len(phases) <= 1:
        return _even_pick(pool, max(max_per_shot, min_ph))
    per_ph = max(min_ph, int(max_per_shot // len(phases)))
    seen: set[int] = set()
    out: List[dict] = []
    for ph in phases:
        sub = [r for r in pool if _ray_phase_id(r) == ph]
        for r in _even_pick(sub, min(len(sub), per_ph)):
            rid = id(r)
            if rid not in seen:
                seen.add(rid)
                out.append(r)
    budget = max(max_per_shot, min_ph * len(phases))
    if len(out) < budget:
        rem = [r for r in pool if id(r) not in seen]
        for r in _even_pick(rem, budget - len(out)):
            rid = id(r)
            if rid not in seen:
                seen.add(rid)
                out.append(r)
    return out


def limit_rays_per_shot_stratified(
    rays: Sequence[dict],
    shot_xs: Sequence[float] | None = None,
    *,
    max_per_shot: int = RAYS_PER_SHOT_DEFAULT,
    min_per_phase: int = RAYS_PER_SHOT_MIN_PER_PHASE,
    x_tol: float = X_SHOT_TOL,
) -> List[dict]:
    """按炮点限额；同一炮多震相时按 ivray 分层抽样（背景射线与理论曲线一致）。"""
    if max_per_shot <= 0:
        return list(rays or [])
    items = list(rays or [])
    if not items:
        return []

    targets = [float(x) for x in (shot_xs or [])]
    if targets:
        buckets: dict[float, List[dict]] = {sx: [] for sx in targets}
        for r in items:
            hit = infer_ray_shot_x(r, targets, tol=x_tol)
            if hit is not None:
                buckets[hit].append(r)
        out: List[dict] = []
        for sx in targets:
            out.extend(
                _limit_one_shot_stratified(
                    buckets[sx],
                    max_per_shot=max_per_shot,
                    min_per_phase=min_per_phase,
                )
            )
        return out

    by_x: dict[float, List[dict]] = {}
    for r in items:
        x0 = ray_origin_x(r)
        if x0 is None:
            continue
        key = round(float(x0) / max(x_tol, 1e-9)) * max(x_tol, 1e-9)
        by_x.setdefault(key, []).append(r)
    out: List[dict] = []
    for pool in by_x.values():
        out.extend(
            _limit_one_shot_stratified(
                pool,
                max_per_shot=max_per_shot,
                min_per_phase=min_per_phase,
            )
        )
    return out


def parse_trapar_xshot(rin_path: Path | str) -> List[float]:
    """从 ``r.in`` TRAPAR 读取 ``xshot`` 数组。"""
    return [x for x, _z in parse_trapar_shots(rin_path)]


def parse_trapar_shots(rin_path: Path | str) -> List[Tuple[float, float]]:
    """从 ``r.in`` TRAPAR 读取 ``(xshot, zshot)`` 列表。"""
    p = Path(rin_path)
    if not p.is_file():
        return []
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    m = re.search(
        r"&trapar\b(.*?)^&end",
        text,
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    if not m:
        return []
    block = m.group(1)

    def _read_array(name: str) -> List[float]:
        m2 = re.search(rf"{name}\s*=\s*([^\n]+)", block, re.IGNORECASE)
        if not m2:
            return []
        raw = re.sub(r"&.*", "", m2.group(1))
        out: List[float] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(float(part))
            except ValueError:
                continue
        return out

    xs = _read_array("xshot")
    zs = _read_array("zshot")
    if not xs:
        return []
    if len(zs) < len(xs):
        pad = zs[-1] if zs else 0.0
        zs = list(zs) + [pad] * (len(xs) - len(zs))
    return list(zip(xs, zs[: len(xs)]))


def write_rin_trapar_shots(
    rin_src: Path | str,
    rin_dst: Path | str,
    xs: Sequence[float],
    zs: Sequence[float],
) -> None:
    """把 ``rin_src`` 复制到 ``rin_dst``，并重写 TRAPAR ``xshot/zshot/ishot``。"""
    src = Path(rin_src)
    dst = Path(rin_dst)
    if not src.is_file():
        raise FileNotFoundError(f"缺少 r.in: {src}")
    text = src.read_text(encoding="utf-8", errors="ignore")
    m = re.search(
        r"(&trapar\b.*?^&end)",
        text,
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    if not m:
        raise ValueError(f"r.in 无 TRAPAR namelist: {src}")
    block = m.group(1)
    n = len(xs)
    if n == 0 or len(zs) != n:
        raise ValueError("xshot/zshot 长度须一致且非空")
    ishot_s = "2" if n == 1 else ", ".join("2" for _ in range(n))
    xshot_s = ", ".join(f"{float(x):g}" for x in xs)
    zshot_s = ", ".join(f"{float(z):g}" for z in zs)

    def _repl_line(name: str, value: str, src_block: str) -> str:
        return re.sub(
            rf"^{name}\s*=.*$",
            f"{name} = {value}",
            src_block,
            count=1,
            flags=re.IGNORECASE | re.MULTILINE,
        )

    new_block = _repl_line("xshot", xshot_s, block)
    new_block = _repl_line("zshot", zshot_s, new_block)
    new_block = _repl_line("ishot", ishot_s, new_block)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text.replace(block, new_block), encoding="utf-8")


def read_rin_ximax(rin_path: Path | str) -> Optional[float]:
    """读取 INVPAR ``ximax``（km）；缺省或未找到时返回 ``None``。"""
    p = Path(rin_path)
    if not p.is_file():
        return None
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    m = re.search(
        r"&invpar\b(.*?)^&end",
        text,
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    if not m:
        return None
    m2 = re.search(r"ximax\s*=\s*([^\n,&]+)", m.group(1), re.IGNORECASE)
    if not m2:
        return None
    try:
        return float(m2.group(1).strip())
    except ValueError:
        return None


def max_pick_offset_km(
    tx_path: Path | str,
    shot_xs: Sequence[float] | None = None,
) -> float:
    """``tx.in`` 中相对炮点的最大拾取偏移 |x - xshot|（km）。"""
    from pyAOBS.modeling.rayinvr.tx_io import (
        filter_tx_dataset_by_shot_xs,
        read_tx_file,
    )

    p = Path(tx_path)
    if not p.is_file():
        return 0.0
    try:
        ds = read_tx_file(p)
    except Exception:
        return 0.0
    if shot_xs:
        ds = filter_tx_dataset_by_shot_xs(ds, shot_xs)
    max_off = 0.0
    for block in ds.shots:
        sx = float(block.xshot)
        for pick in block.picks:
            max_off = max(max_off, abs(float(pick.x) - sx))
    return max_off


def ensure_rin_ximax_at_least(
    rin_path: Path | str,
    min_ximax: float,
) -> tuple[Optional[float], Optional[float]]:
    """若 INVPAR ``ximax`` 小于 ``min_ximax`` 则写回；返回 (旧值, 新值)。"""
    p = Path(rin_path)
    if not p.is_file() or min_ximax <= 0:
        return None, None
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None, None
    m = re.search(
        r"(&invpar\b.*?^&end)",
        text,
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    if not m:
        return None, None
    block = m.group(1)
    m2 = re.search(r"^ximax\s*=.*$", block, re.IGNORECASE | re.MULTILINE)
    if not m2:
        return None, None
    old: Optional[float] = None
    try:
        old = float(re.search(r"ximax\s*=\s*([^\n,&]+)", m2.group(0), re.I).group(1))
    except (AttributeError, ValueError):
        pass
    new = float(min_ximax)
    if old is not None and old >= new - 1e-6:
        return old, old
    new_line = f"ximax = {new:g}"
    new_block = re.sub(
        r"^ximax\s*=.*$",
        new_line,
        block,
        count=1,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    p.write_text(text.replace(block, new_block), encoding="utf-8")
    return old, new


def read_rin_invr(rin_path: Path | str) -> Optional[int]:
    """读取 INVPAR ``invr``；缺省或未找到时返回 ``None``。"""
    p = Path(rin_path)
    if not p.is_file():
        return None
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    m = re.search(
        r"&invpar\b(.*?)^&end",
        text,
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    if not m:
        return None
    m2 = re.search(r"^invr\s*=.*$", m.group(1), re.IGNORECASE | re.MULTILINE)
    if not m2:
        return None
    try:
        return int(
            float(re.search(r"invr\s*=\s*([^\n,&]+)", m2.group(0), re.I).group(1))
        )
    except (AttributeError, ValueError):
        return None


def ensure_rin_invr(
    rin_path: Path | str,
    invr: int,
) -> tuple[Optional[int], Optional[int]]:
    """若 INVPAR ``invr`` 与目标值不同则写回；返回 (旧值, 新值)。"""
    p = Path(rin_path)
    new = int(invr)
    if not p.is_file():
        return None, None
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None, None
    m = re.search(
        r"(&invpar\b.*?^&end)",
        text,
        re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    if not m:
        return None, None
    block = m.group(1)
    m2 = re.search(r"^invr\s*=.*$", block, re.IGNORECASE | re.MULTILINE)
    if not m2:
        return None, None
    old: Optional[int] = None
    try:
        old = int(
            float(re.search(r"invr\s*=\s*([^\n,&]+)", m2.group(0), re.I).group(1))
        )
    except (AttributeError, ValueError):
        pass
    if old is not None and old == new:
        return old, old
    new_block = re.sub(
        r"^invr\s*=.*$",
        f"invr = {new}",
        block,
        count=1,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    p.write_text(text.replace(block, new_block), encoding="utf-8")
    return old, new


def ray_origin_x(ray: dict) -> Optional[float]:
    """射线路径起点 x（优先已标 ``shot_x``，否则取 ``x[0]``）。"""
    sx = ray.get("shot_x")
    if sx is not None:
        try:
            v = float(sx)
            if v == v:  # finite
                return v
        except (TypeError, ValueError):
            pass
    rx = ray.get("x")
    if rx is None or len(rx) == 0:
        return None
    try:
        return float(rx[0])
    except (TypeError, ValueError, IndexError):
        return None


def _ray_endpoint_xs(ray: dict) -> List[float]:
    """射线路径两端 x（用于匹配 TRAPAR 炮点）。"""
    rx = ray.get("x")
    if rx is None or len(rx) == 0:
        return []
    try:
        x0 = float(rx[0])
        x1 = float(rx[-1]) if len(rx) > 1 else x0
    except (TypeError, ValueError, IndexError):
        return []
    if abs(x1 - x0) < 1e-15:
        return [x0]
    return [x0, x1]


def infer_ray_shot_x(
    ray: dict,
    shot_xs: Sequence[float],
    *,
    tol: float = X_SHOT_TOL,
) -> Optional[float]:
    """将射线归属到 TRAPAR 炮点（``shot_x`` 或路径任一端点在容差内）。"""
    if not shot_xs:
        return None
    sx = ray.get("shot_x")
    if sx is not None:
        try:
            hit = match_shot_x(float(sx), shot_xs, tol=tol)
            if hit is not None:
                return hit
        except (TypeError, ValueError):
            pass
    best: Optional[float] = None
    best_d = tol + 1.0
    for x0 in _ray_endpoint_xs(ray):
        hit = match_shot_x(x0, shot_xs, tol=tol)
        if hit is None:
            continue
        d = abs(float(x0) - float(hit))
        if d < best_d:
            best_d = d
            best = hit
    return best


def match_shot_x(
    x0: float,
    shot_xs: Sequence[float],
    *,
    tol: float = X_SHOT_TOL,
) -> Optional[float]:
    """将射线起点匹配到 TRAPAR 炮点 x。"""
    best: Optional[float] = None
    best_d = tol + 1.0
    for sx in shot_xs:
        d = abs(float(x0) - float(sx))
        if d <= tol and d < best_d:
            best_d = d
            best = float(sx)
    return best


def tag_ray_shot_x(
    ray: dict,
    shot_xs: Sequence[float],
    *,
    tol: float = X_SHOT_TOL,
) -> dict:
    """为射线写入 ``shot_x``（匹配 TRAPAR xshot）。"""
    hit = infer_ray_shot_x(ray, shot_xs, tol=tol)
    if hit is not None:
        ray = dict(ray)
        ray["shot_x"] = hit
    return ray


def shot_xs_with_rays(
    rays: Sequence[dict],
    shot_xs: Sequence[float],
    *,
    tol: float = X_SHOT_TOL,
) -> set[float]:
    """已有射线的炮点 x 集合。"""
    have: set[float] = set()
    targets = [float(x) for x in shot_xs]
    for r in rays or []:
        hit = infer_ray_shot_x(r, targets, tol=tol)
        if hit is not None:
            have.add(hit)
    return have


def missing_shot_xs_for_rays(
    rays: Sequence[dict],
    shot_xs: Sequence[float],
    *,
    tol: float = X_SHOT_TOL,
) -> List[float]:
    """``shot_xs`` 中尚无射线的炮点。"""
    have = shot_xs_with_rays(rays, shot_xs, tol=tol)
    return [float(x) for x in shot_xs if float(x) not in have]


def _valid_ray(ray: Optional[dict]) -> bool:
    if not ray:
        return False
    n = int(ray.get("npoints") or 0)
    if n <= 0:
        return False
    x, z = ray.get("x"), ray.get("z")
    return (
        x is not None
        and z is not None
        and len(x) == n
        and len(z) == n
    )


def collect_stored_rays(
    wrap: Any,
    *,
    max_rays: int = 0,
    shot_xs: Sequence[float] | None = None,
    min_rays_per_shot: int = 1,
    max_rays_per_shot: int = 0,
    x_tol: float = X_SHOT_TOL,
) -> Tuple[List[dict], str]:
    """从 ``RayinvrWrapper`` 收集射线；必要时按炮点均衡取样。

    Returns
    -------
    rays, note
        ``note`` 非空时表示截断或无射线炮点等警告。
    """
    notes: List[str] = []
    try:
        n_stored = int(wrap.get_ray_count())
    except Exception as exc:
        return [], f"get_ray_count 失败: {exc}"

    if n_stored <= 0:
        return [], "Fortran 未存储射线"

    cap = RAYINVR_STORED_RAY_CAP
    scan_upto = min(n_stored, cap)
    if max_rays <= 0:
        limit = scan_upto
    else:
        limit = min(scan_upto, int(max_rays))

    targets = [float(x) for x in (shot_xs or [])]
    if int(max_rays_per_shot) == RAYS_PER_SHOT_UNLIMITED:
        per_shot_cap = 0
    elif int(max_rays_per_shot) > 0:
        per_shot_cap = int(max_rays_per_shot)
    elif targets:
        per_shot_cap = RAYS_PER_SHOT_DEFAULT
    else:
        per_shot_cap = 0
    buckets: dict[float, List[dict]] = {sx: [] for sx in targets}
    other: List[dict] = []

    for i in range(1, scan_upto + 1):
        try:
            raw = wrap.get_stored_ray(i)
        except Exception:
            continue
        if not _valid_ray(raw):
            continue
        ray = tag_ray_shot_x(raw, targets, tol=x_tol)
        hit = infer_ray_shot_x(ray, targets, tol=x_tol)
        if hit is not None:
            buckets[hit].append(ray)
        else:
            other.append(ray)

    if limit >= scan_upto:
        out: List[dict] = []
        for sx in targets:
            pool = buckets[sx]
            if per_shot_cap > 0:
                out.extend(_even_pick(pool, per_shot_cap))
            else:
                out.extend(pool)
        if not targets:
            out.extend(other)
        missing = [sx for sx in targets if not buckets[sx]]
        if missing:
            notes.append(
                "下列炮点无射线: "
                + ", ".join(f"{x:.3f}" for x in missing)
            )
        if per_shot_cap > 0 and targets:
            n_raw = sum(len(buckets[sx]) for sx in targets)
            if n_raw > len(out):
                notes.append(
                    f"每炮至多 {per_shot_cap} 条射线（均匀抽样，共 {n_raw}→{len(out)}）"
                )
        if n_stored > cap:
            notes.append(f"Fortran 存储 {n_stored} 条，超过上限 {cap}")
        return out, "; ".join(notes)

    # 需截断：每个炮点至少 min_rays_per_shot，再轮询补齐
    notes.append(f"射线收集截断为 {limit}/{n_stored} 条（按炮点均衡）")
    per = max(1, int(min_rays_per_shot))
    picked: List[dict] = []
    picked_ids: set[int] = set()

    def _take(r: dict) -> None:
        rid = id(r)
        if rid in picked_ids:
            return
        picked_ids.add(rid)
        picked.append(r)

    for sx in targets:
        for r in buckets[sx][:per]:
            _take(r)

    idx = 0
    while len(picked) < limit:
        progressed = False
        for sx in targets:
            pool = buckets[sx]
            if idx < len(pool):
                _take(pool[idx])
                progressed = True
                if len(picked) >= limit:
                    break
        if not progressed:
            for r in other:
                _take(r)
                if len(picked) >= limit:
                    break
            break
        idx += 1

    missing = [sx for sx in targets if not buckets[sx]]
    if missing:
        notes.append(
            "下列炮点无射线: " + ", ".join(f"{x:.3f}" for x in missing)
        )
    picked_out = picked[:limit]
    if per_shot_cap > 0 and targets:
        picked_out = limit_rays_per_shot(
            picked_out, targets, max_per_shot=per_shot_cap, x_tol=x_tol
        )
        if len(picked[:limit]) > len(picked_out):
            notes.append(f"每炮至多 {per_shot_cap} 条射线（均匀抽样）")
    return picked_out, "; ".join(notes)


def summarize_r1_out(path: Path | str, *, max_lines: int = 8) -> str:
    """从 ``r1.out`` 提取追踪失败相关行（供 GUI 日志）。"""
    p = Path(path)
    if not p.is_file():
        return ""
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    keys = (
        "outside model",
        "error",
        "warning",
        "failed",
        "abort",
        "***",
        "stop",
    )
    hits: List[str] = []
    for line in text.splitlines():
        low = line.lower()
        if any(k in low for k in keys):
            s = line.strip()
            if s and s not in hits:
                hits.append(s)
        if len(hits) >= max_lines:
            break
    return "; ".join(hits)
