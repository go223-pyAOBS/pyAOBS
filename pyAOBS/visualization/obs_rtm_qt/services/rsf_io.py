# -*- coding: utf-8 -*-
"""轻量 RSF 读写（与 scripts/su_to_shots 一致，避免整模块副作用）。"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np


def parse_rsf_header(path: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta


def resolve_rsf_binary(rsf_path: str, meta: Optional[Dict[str, str]] = None) -> str:
    meta = meta or parse_rsf_header(rsf_path)
    in_path = meta.get("in", "")
    if not in_path:
        raise RuntimeError("%s: missing in=" % rsf_path)
    if os.path.isabs(in_path) and os.path.isfile(in_path):
        return in_path
    cand = os.path.join(os.path.dirname(os.path.abspath(rsf_path)), os.path.basename(in_path))
    if os.path.isfile(cand):
        return cand
    cand2 = os.path.abspath(in_path)
    if os.path.isfile(cand2):
        return cand2
    cand3 = os.path.abspath(rsf_path) + "@"
    if os.path.isfile(cand3):
        return cand3
    # 头文件常写绝对 in=（如 WSL /var/tmp/...）；二进制实际在 .rsf 同目录
    stem_at = os.path.abspath(rsf_path)
    if not stem_at.endswith("@"):
        alt = stem_at + "@"
        if os.path.isfile(alt):
            return alt
    raise RuntimeError(
        "%s: 找不到二进制 in=%s（若作业在 WSL/其他机写出，请把同名 .rsf@ "
        "放到头文件旁，或在同一环境打开工区）" % (rsf_path, in_path)
    )


def read_gather(path: str) -> Tuple[np.ndarray, Dict[str, float]]:
    """读 shot_###.rsf → data (n1=time, n2=trace), meta 含 d1,o1,n1,n2。"""
    meta = parse_rsf_header(path)
    n1, n2 = int(meta["n1"]), int(meta.get("n2", 1))
    d1 = float(meta.get("d1", "0.004"))
    o1 = float(meta.get("o1", "0"))
    bin_path = resolve_rsf_binary(path, meta)
    raw = np.fromfile(bin_path, dtype=np.float32, count=n1 * n2)
    if raw.size != n1 * n2:
        raise RuntimeError(
            "short read %s: got %d want %d" % (path, raw.size, n1 * n2)
        )
    data = raw.reshape((n2, n1)).T.copy()
    return data, {"d1": d1, "o1": o1, "n1": float(n1), "n2": float(n2)}


def read_rsf_slice_n3(
    path: str,
    i3: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    读 3D RSF 的一张 n3 切片（不整卷进内存）。

    约定与 awefd2d 波场一致：n1 最快、n2、n3；返回 (n1, n2) 即 (z, x)。
    ``i3`` 为 None 或负值时取中间帧 ``n3//2``。
    """
    meta = parse_rsf_header(path)
    n1 = int(meta["n1"])
    n2 = int(meta.get("n2", 1))
    n3 = max(int(meta.get("n3", 1)), 1)
    o1 = float(meta.get("o1", 0))
    d1 = float(meta.get("d1", 1))
    o2 = float(meta.get("o2", 0))
    d2 = float(meta.get("d2", 1))
    o3 = float(meta.get("o3", 0))
    d3 = float(meta.get("d3", 1))
    if i3 is None or int(i3) < 0:
        frame = n3 // 2
    else:
        frame = int(i3)
    if frame >= n3:
        raise ValueError("%s: 帧 %d 越界（n3=%d）" % (path, frame, n3))
    bin_path = resolve_rsf_binary(path, meta)
    plane = n1 * n2
    with open(bin_path, "rb") as f:
        f.seek(frame * plane * 4)
        raw = np.fromfile(f, dtype=np.float32, count=plane)
    if raw.size != plane:
        raise RuntimeError(
            "short read %s frame %d: got %d want %d" % (path, frame, raw.size, plane)
        )
    data = raw.reshape((n2, n1)).T.copy()
    return data, {
        "o1": o1,
        "d1": d1,
        "n1": float(n1),
        "o2": o2,
        "d2": d2,
        "n2": float(n2),
        "o3": o3,
        "d3": d3,
        "n3": float(n3),
        "i3": float(frame),
        "t": float(o3 + frame * d3),
    }


def write_gather(
    path: str,
    data: np.ndarray,
    d1: float,
    o1: float = 0.0,
    *,
    label1: str = "Time",
    unit1: str = "s",
    label2: str = "Trace",
    unit2: str = "",
) -> None:
    data = np.asarray(data, dtype=np.float32)
    n1, n2 = data.shape
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    bin_path = path + "@"
    with open(bin_path, "wb") as b:
        for j in range(n2):
            b.write(np.ascontiguousarray(data[:, j]).tobytes())
    with open(path, "w", encoding="utf-8") as h:
        h.write("in=%s\n" % os.path.abspath(bin_path))
        h.write("n1=%d\nd1=%g\no1=%g\n" % (n1, d1, o1))
        h.write("label1=%s\nunit1=%s\n" % (label1, unit1))
        h.write("n2=%d\n" % n2)
        h.write("d2=1\n")
        h.write("o2=0\n")
        h.write("label2=%s\nunit2=%s\n" % (label2, unit2))
        h.write("data_format=native_float\nesize=4\n")


# path → (mtime, table)；换炮/拼图反复查 offset 时避免每次扫整文件
_OFFSETS_TABLE_CACHE: Dict[str, Tuple[float, Dict[int, list]]] = {}


def load_offsets_table(path: str) -> Dict[int, list]:
    """
    offsets.txt → {ishot: [offset_m per iobs, ...]}
    行: ishot iobs offset_m source offset_xy_m
    """
    if not path or not os.path.isfile(path):
        return {}
    try:
        mtime = float(os.path.getmtime(path))
    except OSError:
        mtime = -1.0
    key = os.path.normpath(os.path.abspath(path))
    hit = _OFFSETS_TABLE_CACHE.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    out: Dict[int, list] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a = line.split()
            if len(a) < 3:
                continue
            ishot, iobs = int(a[0]), int(a[1])
            om = float(a[2])
            out.setdefault(ishot, [])
            while len(out[ishot]) <= iobs:
                out[ishot].append(0.0)
            out[ishot][iobs] = om
    _OFFSETS_TABLE_CACHE[key] = (mtime, out)
    return out
