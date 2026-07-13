"""
Fig.5 digitized F-locus acceptance (panels a/b @ 100 / 400 MPa).

Requires ``figure05_dvp_calibration.json`` and ``fig5_dvp_calibrate=True``.

Usage::

    py -3.11 petrology/validation/check_fig5_digitized.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.data.load_catalog import load_melt_catalog, oxides_from_record
from petrology.fc.delta_vp import R3_DELTA_VP_WL_KW
from petrology.fc.figure05_digitized import residual_to_f_loci, sigma_delta_vp_at_f
from petrology.fractionation import delta_vp_km_s
from petrology.norm_velocity import norm_velocity_from_record

PANELS = (
    ("a_vlc_100", 100.0),
    ("b_vlc_400", 400.0),
)
F_CHECK = (0.5, 0.6, 0.7, 0.8)


@dataclass
class Check:
    name: str
    value: float
    limit: float
    op: str
    unit: str = " km/s"

    @property
    def ok(self) -> bool:
        if self.op == "<":
            return abs(self.value) < self.limit
        if self.op == ">":
            return self.value > self.limit
        raise ValueError(self.op)

    def line(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        return f"  [{mark}] {self.name}: {self.value:+.4f}{self.unit} {self.op} {self.limit:.4f}{self.unit}"


def run(*, strict: bool = False) -> int:
    from petrology.fc.figure05_digitized import load_figure05_digitized

    data = load_figure05_digitized()
    rows = [r for r in load_melt_catalog() if r.get("include_in_regression")]
    wl_kw = dict(R3_DELTA_VP_WL_KW)
    wl_kw["fig5_dvp_calibrate"] = True  # this check requires the empirical lift
    checks: list[Check] = []

    for pkey, p_fc in PANELS:
        for f_t in F_CHECK:
            dd = []
            for rec in rows:
                bulk = float(
                    norm_velocity_from_record(rec, mineral_backend=wl_kw["mineral_backend"])["vp_km_s"]  # type: ignore[arg-type]
                )
                ox = {k: v for k, v in oxides_from_record(rec).items() if k != "P2O5"}
                d_sim = delta_vp_km_s(
                    f_t,
                    bulk_vp_km_s=bulk,
                    p_fc_mpa=p_fc,
                    engine="wl1990",
                    melt_oxides_wt=ox,
                    **wl_kw,
                )
                bias, _ = residual_to_f_loci(d_sim, bulk + d_sim, pkey, f_t, data=data)  # type: ignore[arg-type]
                if not np.isnan(bias):
                    dd.append(bias)
            if not dd:
                continue
            mean_bias = float(np.mean(dd))
            sigma = sigma_delta_vp_at_f(pkey, f_t, data=data) if f_t in (0.5, 0.8) else (0.05 if not strict else 0.04)
            checks.append(
                Check(
                    f"{pkey} mean bias @ F={f_t:g}",
                    mean_bias,
                    sigma,
                    "<",
                )
            )
            rms = float(np.sqrt(np.mean(np.square(dd))))
            print(f"  (info) {pkey} RMS @ F={f_t:g}: {rms:.4f} km/s")

    print("=== Fig.5 digitized F-locus acceptance (fig5_dvp_calibrate) ===")
    print(f"  catalog n={len(rows)}  wl_kw={wl_kw}")
    print("  (RMS printed for info — catalog spread; pass/fail on mean bias only)")
    for c in checks:
        print(c.line())

    failed = [c for c in checks if not c.ok]
    if failed:
        print(f"\nFig.5: {len(failed)} check(s) failed.")
        return 1
    print(f"\nFig.5: all {len(checks)} checks passed.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(strict=args.strict))


if __name__ == "__main__":
    main()
