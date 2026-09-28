#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对照 PPP+PSP 同一次联合：同时收回 Vp 与 Vs。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ps_inv"))
sys.path.insert(0, str(HERE.parent / "psp_inv"))
sys.path.insert(0, str(HERE.parent / "ps_fwd"))
sys.path.insert(0, str(HERE.parent / "converse_fwd"))
sys.path.insert(0, str(HERE.parent / "water_fwd"))
sys.path.insert(0, str(HERE.parent / "water_inv"))
from check_ps_inv import plot_ttimes_fit, report_vp  # noqa: E402
from check_psp_inv import report as report_vs  # noqa: E402
from check_water_inv import latest_smesh, parse_picks  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=HERE / "out")
    p.add_argument("--smesh", type=Path, default=None)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()
    rec = args.smesh if args.smesh else latest_smesh(args.out_root)
    if not rec.is_file():
        cands = sorted(HERE.glob("out.smesh.*.*"))
        if not cands:
            print("缺 out.smesh.*", file=sys.stderr)
            return 1
        rec = max(cands, key=lambda q: (int(q.name.split(".")[-2]), int(q.name.split(".")[-1])))
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    rc = 0
    if (HERE / "rec_vp.smesh").is_file():
        rc |= report_vp(HERE / "rec_vp.smesh", HERE / "true_vp.smesh", HERE / "start_vp.smesh")
    rc |= report_vs(rec, HERE / "true_vs.smesh", HERE / "start_vs.smesh")
    import plot_ppp_psp_inv_models as plot

    plot.main()
    if (HERE / "syn_ppp.dat").is_file() and (HERE / "syn_ppp_start.dat").is_file():
        ppp_rec = (
            parse_picks((HERE / "syn_ppp_rec.dat").read_text(encoding="utf-8"))
            if (HERE / "syn_ppp_rec.dat").is_file()
            else None
        )
        plot_ttimes_fit(
            parse_picks((HERE / "syn_ppp.dat").read_text(encoding="utf-8")),
            parse_picks((HERE / "syn_ppp_start.dat").read_text(encoding="utf-8")),
            ppp_rec,
            HERE / "check_inv_vp_ttimes.png",
            show=False,
        )
        print(f"wrote {HERE / 'check_inv_vp_ttimes.png'}")
    if (HERE / "syn_inv.dat").is_file() and (HERE / "syn_start.dat").is_file():
        rec_picks = (
            parse_picks((HERE / "syn_rec.dat").read_text(encoding="utf-8"))
            if (HERE / "syn_rec.dat").is_file()
            else None
        )
        plot_ttimes_fit(
            parse_picks((HERE / "syn_inv.dat").read_text(encoding="utf-8")),
            parse_picks((HERE / "syn_start.dat").read_text(encoding="utf-8")),
            rec_picks,
            HERE / "check_inv_ttimes.png",
            show=False,
        )
        print(f"wrote {HERE / 'check_inv_ttimes.png'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
