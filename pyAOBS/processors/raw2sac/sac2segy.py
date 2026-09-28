#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键：SAC + UKOOA → SEGY（炮=sx/sy，OBS=gx/gy）。

等价于调用 sac2y_v2_1_obspy.py，命名与 sac2su / raw2segy 对齐。

用法::

    python sac2segy.py data.sac shots.ukooa sac2y.ini out.segy
"""

from __future__ import annotations

import argparse
from pathlib import Path

from convert_pipeline import sac_to_segy


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SAC → SEGY (shot=sx/sy, OBS=gx/gy)")
    ap.add_argument("sac", type=Path)
    ap.add_argument("ukooa", type=Path)
    ap.add_argument("config", type=Path)
    ap.add_argument("segy", type=Path)
    args = ap.parse_args(argv)
    out = sac_to_segy(args.sac, args.ukooa, args.config, args.segy)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
