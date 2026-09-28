#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键：OBS RAW → SAC(分量) → SEGY → SU。

用法::

    python raw2su.py rawfile 1000 256 shots.ukooa sac2y.ini out_dir
    python raw2su.py rawfile 1000 256 shots.ukooa sac2y.ini out_dir --channels shz,hyd
    python raw2su.py ... --keep-segy
"""

from __future__ import annotations

import argparse
from pathlib import Path

from convert_pipeline import raw_to_su


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RAW → SAC → SEGY → SU")
    ap.add_argument("raw", type=Path)
    ap.add_argument("sps")
    ap.add_argument("tc")
    ap.add_argument("ukooa", type=Path)
    ap.add_argument("config", type=Path, help="sac2y config ini")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument(
        "--channels",
        default="all",
        help="comma list: shx,shy,shz,hyd or all",
    )
    ap.add_argument("--keep-segy", action="store_true")
    ap.add_argument("--no-keep-sac", action="store_true")
    ap.add_argument("--endian", choices=["little", "big"], default="little")
    args = ap.parse_args(argv)
    ch = None if args.channels.strip().lower() in ("all", "*") else [
        c.strip() for c in args.channels.split(",") if c.strip()
    ]
    outs = raw_to_su(
        args.raw,
        args.sps,
        args.tc,
        args.ukooa,
        args.config,
        args.out_dir,
        channels=ch,
        keep_sac=not args.no_keep_sac,
        keep_segy=bool(args.keep_segy),
        endian=args.endian,
    )
    for p in outs:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
