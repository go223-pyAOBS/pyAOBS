#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键：OBS RAW → SAC → SEGY（不停 SU）。

用法::

    python raw2segy.py rawfile 1000 256 shots.ukooa sac2y.ini out_dir
"""

from __future__ import annotations

import argparse
from pathlib import Path

from convert_pipeline import raw_to_segy


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RAW → SAC → SEGY")
    ap.add_argument("raw", type=Path)
    ap.add_argument("sps")
    ap.add_argument("tc")
    ap.add_argument("ukooa", type=Path)
    ap.add_argument("config", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--channels", default="all")
    args = ap.parse_args(argv)
    ch = None if args.channels.strip().lower() in ("all", "*") else [
        c.strip() for c in args.channels.split(",") if c.strip()
    ]
    outs = raw_to_segy(
        args.raw, args.sps, args.tc, args.ukooa, args.config, args.out_dir, channels=ch
    )
    for p in outs:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
