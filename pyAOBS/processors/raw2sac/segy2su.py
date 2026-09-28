#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SEGY → SU（去卷头；IBM→IEEE；默认 little-endian）。

用法::

    python segy2su.py input.segy output.su
    python segy2su.py input.segy output.su --endian little
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Convert SEGY to SU (IEEE float)")
    ap.add_argument("segy", type=Path, help="input SEGY/SGY file")
    ap.add_argument("su", type=Path, help="output SU file")
    ap.add_argument(
        "--endian",
        choices=["little", "big"],
        default="little",
        help="output SU endian (default: little)",
    )
    ap.add_argument(
        "--src-endian",
        choices=["little", "big", "auto"],
        default="auto",
        help="input SEGY trace endian (default: auto probe)",
    )
    args = ap.parse_args(argv)

    # 经 idata 服务（避免 import pyAOBS.processors 拉 pygmt）
    # raw2sac -> processors -> pyAOBS -> repo
    processors = Path(__file__).resolve().parents[1]
    idata_dir = processors / "idata"
    raw2sac_dir = Path(__file__).resolve().parent
    repo_root = processors.parent.parent
    for p in (str(repo_root), str(raw2sac_dir), str(idata_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from gui.services.segy_dataset import convert_segy_to_su

    src_endian = None if args.src_endian == "auto" else args.src_endian
    out = convert_segy_to_su(
        args.segy,
        args.su,
        endian=args.endian,
        src_endian=src_endian,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
