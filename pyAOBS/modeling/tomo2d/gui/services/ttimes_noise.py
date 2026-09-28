"""走时数据 (ttimes/geom 同构) 读写与加噪。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def add_traveltime_noise(
    src: str | Path,
    dst: str | Path,
    *,
    sigma: float,
    seed: int | None = None,
    relative_to_u: bool = False,
) -> int:
    """
    复制 ``src`` 到 ``dst``，对每条 ``r`` 行的走时 ``t`` 加高斯噪声。

    - ``sigma``：绝对噪声标准差（秒）；若 ``relative_to_u`` 则为 ``sigma * u``（u 为该行误差列）。
    - 返回被扰动的 r 行数。
    """
    if sigma < 0:
        raise ValueError("走时噪声 sigma 不能为负")
    text = Path(src).read_text(encoding="utf-8", errors="replace")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    rng = np.random.default_rng(seed)
    out: list[str] = []
    n_r = 0
    for line in lines:
        raw = line
        s = line.strip()
        if not s:
            out.append(raw)
            continue
        parts = s.split()
        if parts[0] == "r" and len(parts) >= 6:
            try:
                x, y = float(parts[1]), float(parts[2])
                code = int(float(parts[3]))
                t, u = float(parts[4]), float(parts[5])
            except (TypeError, ValueError):
                out.append(raw)
                continue
            scale = (sigma * abs(u)) if relative_to_u else sigma
            t2 = t + float(rng.normal(0.0, scale if scale > 0 else 0.0))
            out.append(f"r{x:10.3f}{y:10.3f}{code:5d}{t2:10.3f}{u:10.3f}")
            n_r += 1
        else:
            out.append(raw)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_text("\n".join(out) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")
    return n_r
