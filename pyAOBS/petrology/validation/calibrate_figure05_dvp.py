"""Fit and save Fig.5 F-locus ΔVp calibration coefficients."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.fig5_dvp_cal import (
    FIG5_DVP_CAL_JSON,
    compute_calibration,
    save_fig5_dvp_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate ΔVp vs Fig.5 digitized F-loci")
    parser.add_argument("--json", type=Path, default=FIG5_DVP_CAL_JSON)
    parser.add_argument("--mineral-backend", default="sb1994_fig2ol")
    parser.add_argument("--kd-engine", default="langmuir")
    args = parser.parse_args()

    cal = compute_calibration(
        mineral_backend=args.mineral_backend,
        kd_engine=args.kd_engine,
    )
    save_fig5_dvp_calibration(cal, args.json)

    print(f"Wrote {args.json}")
    for key, track in cal.get("tracks", {}).items():
        coeff = track.get("coeff", [])
        print(f"  P_fc={key} MPa: ΔVp_corr = {coeff[0]:+.4f} + {coeff[1]:+.4f} * F")
        for f, b, c in zip(
            track.get("f_nodes", []),
            track.get("bias_sim_minus_paper", []),
            track.get("correction_at_nodes", []),
        ):
            print(f"    F={f:g}: bias={b:+.4f}  correction={c:+.4f} km/s")


if __name__ == "__main__":
    main()
