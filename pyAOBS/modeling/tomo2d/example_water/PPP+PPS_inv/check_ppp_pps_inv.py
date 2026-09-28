#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对照 PPP+PPS 同一次联合：同时收回 Vp 与台侧盖层 Vs。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ps_inv"))
sys.path.insert(0, str(HERE.parent / "pps_inv"))
sys.path.insert(0, str(HERE.parent / "ps_fwd"))
sys.path.insert(0, str(HERE.parent / "converse_fwd"))
sys.path.insert(0, str(HERE.parent / "water_fwd"))
sys.path.insert(0, str(HERE.parent / "water_inv"))
from check_ps_inv import plot_ttimes_fit, report_vp  # noqa: E402
from check_pps_inv import OBS_HALF, S_Z1, S_Z2, _lid_near_obs  # noqa: E402
from check_water_inv import latest_smesh, parse_picks  # noqa: E402
from make_pps_inv_case import (  # noqa: E402
    H,
    KAPPA_START,
    V_WATER,
    Z_CONV,
    illum_x_range,
    node_stats,
    parse_smesh,
)


def report_vs(rec_path: Path, true_path: Path, start_path: Path, vp_path: Path | None) -> int:
    xs, zs, vrec = parse_smesh(rec_path)
    _, _, vtrue = parse_smesh(true_path)
    _, _, vstart = parse_smesh(start_path)
    x_lo, x_hi = illum_x_range()
    w_m, w_lo, w_hi, n_w = node_stats(
        xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, z_lo=0.0, z_hi=H, z_hi_inclusive=False
    )
    lid_m, lid_lo, lid_hi, n_lid = node_stats(
        xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, z_lo=H + 1e-6, z_hi=Z_CONV
    )
    t_lid, *_ = node_stats(xs, zs, vtrue, x_lo=x_lo, x_hi=x_hi, z_lo=H + 1e-6, z_hi=Z_CONV)
    st_lid, *_ = node_stats(xs, zs, vstart, x_lo=x_lo, x_hi=x_hi, z_lo=H + 1e-6, z_hi=Z_CONV)
    s_m, s_lo, s_hi, n_s = node_stats(
        xs, zs, vrec, x_lo=x_lo, x_hi=x_hi, z_lo=S_Z1, z_hi=S_Z2, z_hi_inclusive=True
    )
    t_s, *_ = node_stats(xs, zs, vtrue, x_lo=x_lo, x_hi=x_hi, z_lo=S_Z1, z_hi=S_Z2, z_hi_inclusive=True)
    st_s, *_ = node_stats(xs, zs, vstart, x_lo=x_lo, x_hi=x_hi, z_lo=S_Z1, z_hi=S_Z2, z_hi_inclusive=True)
    obs_m, *_ = _lid_near_obs(xs, zs, vrec, x_lo, x_hi)
    t_obs, *_ = _lid_near_obs(xs, zs, vtrue, x_lo, x_hi)
    st_obs, *_ = _lid_near_obs(xs, zs, vstart, x_lo, x_hi)
    print(f"recovered  {rec_path.name}  (Vs field, PPS joint)")
    print(f"  water  n={n_w}  mean={w_m:.4f}  [{w_lo:.4f},{w_hi:.4f}]  expect={V_WATER}")
    print(
        f"  lid    n={n_lid}  mean={lid_m:.4f}  [{lid_lo:.4f},{lid_hi:.4f}]  "
        f"true={t_lid:.4f}  start={st_lid:.4f}"
    )
    print(
        f"  lid@OBS  mean={obs_m:.4f}  true={t_obs:.4f}  start={st_obs:.4f}  "
        f"(±{OBS_HALF:g} km)"
    )
    print(
        f"  below  n={n_s}  mean={s_m:.4f}  [{s_lo:.4f},{s_hi:.4f}]  "
        f"true={t_s:.4f}  start={st_s:.4f}"
    )
    ok = True
    if abs(w_m - V_WATER) > 0.02 or abs(w_hi - V_WATER) > 0.08:
        print(f"FAIL water not frozen at {V_WATER}（应加 -Y -w）")
        ok = False
    if vp_path is not None and vp_path.is_file():
        _, _, vvp = parse_smesh(vp_path)
        exp_m, *_ = node_stats(
            xs, zs, [[v / KAPPA_START for v in col] for col in vvp],
            x_lo=x_lo, x_hi=x_hi, z_lo=S_Z1, z_hi=S_Z2, z_hi_inclusive=True,
        )
        print(f"  below vs rec_vp/{KAPPA_START:g}  mean={exp_m:.4f}  (joint reset)")
        if abs(s_m - exp_m) > 0.15:
            print(f"FAIL below-conv Vs moved {abs(s_m - exp_m):.3f} from rec_vp/k (PPS 面下是 P)")
            ok = False
        else:
            print(f"  below Vs stayed at rec_vp/k ({exp_m:.3f} → {s_m:.3f})")
    if abs(obs_m - t_obs) >= abs(st_obs - t_obs) - 1e-6:
        print("FAIL lid-near-OBS Vs did not move toward true")
        ok = False
    elif abs(obs_m - t_obs) > 0.40:
        print(f"FAIL lid-near-OBS mean not within 0.40 of true {t_obs:.3f}")
        ok = False
    else:
        print(f"  lid-near-OBS moved toward true ({st_obs:.3f} → {obs_m:.3f}, true {t_obs:.3f})")
    if ok:
        print(f"OK  PPS 台侧盖层 Vs（联合 PPP，-k{KAPPA_START:g}）")
    return 0 if ok else 1


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
    vp = HERE / "rec_vp.smesh"
    if vp.is_file():
        rc |= report_vp(vp, HERE / "true_vp.smesh", HERE / "start_vp.smesh")
    rc |= report_vs(rec, HERE / "true_vs.smesh", HERE / "start_vs.smesh", vp if vp.is_file() else None)
    import plot_ppp_pps_inv_models as plot

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
