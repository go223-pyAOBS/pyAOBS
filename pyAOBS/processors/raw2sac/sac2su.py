#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键：SAC + UKOOA → SEGY(炮=sx/sy, OBS=gx/gy) → SU。

用法::

    python sac2su.py data.sac shots.ukooa sac2y.ini out.su
    python sac2su.py data.sac shots.ukooa sac2y.ini out.su --keep-segy out.segy
"""

from __future__ import annotations

import argparse
from pathlib import Path

from convert_pipeline import sac_to_su


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SAC → SEGY → SU (shot=sx/sy, OBS=gx/gy)")
    ap.add_argument("sac", type=Path)
    ap.add_argument("ukooa", type=Path)
    ap.add_argument("config", type=Path, help="sac2y config ini")
    ap.add_argument("su", type=Path, help="output .su")
    ap.add_argument("--keep-segy", type=Path, default=None, help="also keep intermediate SEGY")
    ap.add_argument("--endian", choices=["little", "big"], default="little")
    args = ap.parse_args(argv)
    out = sac_to_su(
        args.sac,
        args.ukooa,
        args.config,
        args.su,
        keep_segy=args.keep_segy,
        endian=args.endian,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
