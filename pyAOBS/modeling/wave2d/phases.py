# -*- coding: utf-8 -*-
"""震相颜色/折合速度。不改 tomo2d。"""

from __future__ import annotations

from pathlib import Path

# 0 Pg, 1 PmP, 6 PSP 转折, 7 PPS, 8 PSS, 12 PSP-Moho, 13 PSS-Moho
CODES_P = (0, 1)
CODES_S = (6, 7, 8, 12, 13)
CODES_ALL = CODES_P + CODES_S

PHASE_STYLE = {
    0: ("#222222", "0 Pg"),
    1: ("#d95f02", "1 PmP"),
    6: ("#1f77b4", "6 PSP"),
    7: ("#17becf", "7 PPS"),
    8: ("#2ca02c", "8 PSS"),
    12: ("#c44e8a", "12 PSP-Moho"),
    13: ("#8c564b", "13 PSS-Moho"),
}

OBS_XS = (30.0, 50.0, 70.0, 90.0, 110.0)
SHOT_XS = tuple(float(x) for x in range(2, 147, 8))
DX_MIN = 10.0
VRED_P = 8.0
VRED_S = 4.5


def write_geom(path: Path, *, codes=CODES_ALL, obs_xs=OBS_XS) -> int:
    recs_by_obs: dict[float, list[str]] = {}
    nrec = 0
    for ox in obs_xs:
        recs: list[str] = []
        for code in codes:
            for x in SHOT_XS:
                if abs(x - ox) <= DX_MIN:
                    continue
                recs.append(f"r  {x:8.3f}     0.010 {code:4d}     0.000     0.050")
        recs_by_obs[ox] = recs
        nrec += len(recs)
    lines = [str(len(obs_xs))]
    for ox in obs_xs:
        recs = recs_by_obs[ox]
        lines.append(f"s  {ox:8.3f}     2.000 {len(recs):4d}")
        lines.extend(recs)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return nrec
