#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键：OBEM TSM → SAC → SEGY。

用法::

    python obem2segy.py obem.ini shots.ukooa sac2y.ini out_dir
    python obem2segy.py obem.ini shots.ukooa sac2y.ini out_dir --skip-obem
"""

from __future__ import annotations

import argparse
from pathlib import Path

from convert_pipeline import obem_to_segy


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="OBEM → SAC → SEGY")
    ap.add_argument("obem_config", type=Path)
    ap.add_argument("ukooa", type=Path)
    ap.add_argument("sac2y_config", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--channels", default="all")
    ap.add_argument(
        "--skip-obem",
        action="store_true",
        help="skip OBEM step; convert existing SAC under config output_path",
    )
    args = ap.parse_args(argv)
    ch = None if args.channels.strip().lower() in ("all", "*") else [
        c.strip() for c in args.channels.split(",") if c.strip()
    ]
    outs = obem_to_segy(
        args.obem_config,
        args.ukooa,
        args.sac2y_config,
        args.out_dir,
        channels=ch,
        skip_obem=bool(args.skip_obem),
    )
    for p in outs:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
