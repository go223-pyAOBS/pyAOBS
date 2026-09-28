#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P1 四震相联合 QC：按相位走时 + 盖层/面下 Vs + Vp 是否被带偏。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FOLDER = HERE / "inv_joint"
sys.path.insert(0, str(HERE / "inv_2d"))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1].parent / "water_inv"))
sys.path.insert(0, str(HERE.parents[1].parent / "ps_fwd"))
import inv_grid as g  # noqa: E402
import make_ppp_psp_inv_case as m2  # noqa: E402
import plot_rugged_inv as pr  # noqa: E402
from check_ps_fwd import draw_ps_rays  # noqa: E402
from check_water_inv import parse_picks  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DCLIM = 0.60
XLO, XHI = 30.0, 70.0
OBS_XS = (30.0, 40.0, 50.0, 60.0, 70.0)
PHASES = ((0, "PPP"), (6, "PSP"), (7, "PPS"), (8, "PSS"))


def mask_crust(xs, zs, vel):
    arr = np.asarray(vel, float).T
    for j, z in enumerate(zs):
        for i, _x in enumerate(xs):
            if z < g.H - 1e-9:
                arr[j, i] = np.nan
    return arr


def load_pair(true_p: Path, start_p: Path, rec_p: Path):
    xs, zs, true = m2.parse_smesh(true_p)
    _, _, start = m2.parse_smesh(start_p)
    _, _, rec = m2.parse_smesh(rec_p)
    return (
        xs,
        zs,
        mask_crust(xs, zs, true),
        mask_crust(xs, zs, start),
        mask_crust(xs, zs, rec),
    )


def ttrms(obs: Path, pred: Path, code: int | None = None) -> float:
    o = parse_picks(obs.read_text(encoding="utf-8"))
    p = parse_picks(pred.read_text(encoding="utf-8"))
    if code is not None:
        o = [x for x in o if x[0] == code]
        p = [x for x in p if x[0] == code]
    key = lambda t: (round(t[4], 3), round(t[1], 3), int(t[0]))
    md = {key(x): x[3] for x in p}
    ds = [md[key(x)] - x[3] for x in o if key(x) in md and math.isfinite(x[3])]
    return math.sqrt(sum(v * v for v in ds) / len(ds)) if ds else float("nan")


def npicks(path: Path, code: int) -> tuple[int, int]:
    recs = parse_picks(path.read_text(encoding="utf-8"))
    got = [x for x in recs if x[0] == code]
    ok = sum(1 for x in got if math.isfinite(x[3]))
    return ok, len(got)


def region_rms(xs, zs, a, b, *, lid: bool | None) -> float:
    s = n = 0
    for i, x in enumerate(xs):
        if not (XLO <= x <= XHI):
            continue
        zi = g.z_conv(x)
        for k, z in enumerate(zs):
            if z < g.H - 1e-9:
                continue
            if lid is True and z >= zi - 1e-9:
                continue
            if lid is False and z < zi - 1e-9:
                continue
            d = a[k, i] - b[k, i]
            if np.isnan(d):
                continue
            s += float(d * d)
            n += 1
    return math.sqrt(s / n) if n else float("nan")


PHASE_COLOR = {0: "#1f77b4", 6: "#2ca02c", 7: "#ff7f0e", 8: "#c44e8a"}


def _pick_map(picks, code):
    out = {}
    for p in picks:
        if p[0] != code or not math.isfinite(p[3]):
            continue
        out[(round(p[4], 3), round(p[1], 3))] = (abs(p[1] - p[4]), p[3])
    return out


def plot_phase_ttimes(obs, start, rec, phases, out_png: Path, title: str) -> None:
    n = len(phases)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7.0), facecolor="w", layout="constrained")
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for j, (code, name) in enumerate(phases):
        color = PHASE_COLOR.get(code, "0.3")
        mo, ms, mr = _pick_map(obs, code), _pick_map(start, code), _pick_map(rec, code)
        keys = sorted(set(mo) & set(ms) & set(mr), key=lambda k: mo[k][0])
        dx = [mo[k][0] for k in keys]
        to = [mo[k][1] for k in keys]
        ts = [ms[k][1] for k in keys]
        tr = [mr[k][1] for k in keys]
        ax, axr = axes[0, j], axes[1, j]
        ax.plot(dx, to, "o", color=color, ms=4, label="观测")
        ax.plot(dx, ts, "--", color=color, lw=1.2, alpha=0.85, label="初值")
        ax.plot(dx, tr, "-", color="0.15", lw=1.4, label="收回")
        r0 = [a - b for a, b in zip(ts, to)]
        r1 = [a - b for a, b in zip(tr, to)]
        rms0 = math.sqrt(sum(v * v for v in r0) / len(r0)) if r0 else float("nan")
        rms1 = math.sqrt(sum(v * v for v in r1) / len(r1)) if r1 else float("nan")
        ax.set_title(f"{name}   {rms0:.3f} → {rms1:.3f} s", fontsize=10)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.set_ylabel("走时 t (s)")
            ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
        axr.axhline(0.0, color="0.45", lw=0.8)
        axr.plot(dx, r0, "s", color=color, ms=3.5, alpha=0.7, label="初值−观测")
        axr.plot(dx, r1, "o", color="0.15", ms=4, label="收回−观测")
        axr.set_xlabel("偏移 dx (km)")
        axr.grid(True, alpha=0.3)
        if j == 0:
            axr.set_ylabel("残差 (s)")
            axr.legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.suptitle(title)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


def _drms(obsf: Path, predf: Path, ca: int, cb: int) -> float:
    o = parse_picks(obsf.read_text(encoding="utf-8"))
    p = parse_picks(predf.read_text(encoding="utf-8"))
    key = lambda t: (round(t[4], 3), round(t[1], 3))

    def by(code, rows):
        return {key(x): x[3] for x in rows if x[0] == code and math.isfinite(x[3])}

    oa, ob, pa, pb = by(ca, o), by(cb, o), by(ca, p), by(cb, p)
    ds = []
    for k in oa:
        if k in ob and k in pa and k in pb:
            ds.append((oa[k] - ob[k]) - (pa[k] - pb[k]))
    return math.sqrt(sum(v * v for v in ds) / len(ds)) if ds else float("nan")


def _corrected_psp_rms(true_picks, pred_picks) -> float:
    """T_PSS − (T_PPS − T_PPP) vs true T_PSP, same (src, rcv.x)."""
    key = lambda t: (round(t[4], 3), round(t[1], 3))

    def by(code, rows):
        return {key(x): x[3] for x in rows if x[0] == code and math.isfinite(x[3])}

    tp, o8, o7, o0 = by(6, true_picks), by(8, pred_picks), by(7, pred_picks), by(0, pred_picks)
    ds = []
    for k, t6 in tp.items():
        if k in o8 and k in o7 and k in o0:
            ds.append((o8[k] - (o7[k] - o0[k])) - t6)
    return math.sqrt(sum(v * v for v in ds) / len(ds)) if ds else float("nan")


def plot_diff_ttimes(obs, start, rec, out_png: Path, pairs=None, title=None) -> None:
    if pairs is None:
        pairs = ((6, 0, "PSP−PPP"), (7, 0, "PPS−PPP"), (8, 6, "PSS−PSP"))
    if title is None:
        title = "C 步走时差拟合    同一炮–台  ΔT = T_a − T_b"
    n = max(len(pairs), 1)
    fig, axes = plt.subplots(2, n, figsize=(3.8 * n, 7.0), facecolor="w", layout="constrained")
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for j, (ca, cb, name) in enumerate(pairs):
        oa, ob = _pick_map(obs, ca), _pick_map(obs, cb)
        sa, sb = _pick_map(start, ca), _pick_map(start, cb)
        ra, rb = _pick_map(rec, ca), _pick_map(rec, cb)
        keys = sorted(set(oa) & set(ob) & set(sa) & set(sb) & set(ra) & set(rb), key=lambda k: oa[k][0])
        dx = [oa[k][0] for k in keys]
        to = [oa[k][1] - ob[k][1] for k in keys]
        ts = [sa[k][1] - sb[k][1] for k in keys]
        tr = [ra[k][1] - rb[k][1] for k in keys]
        ax, axr = axes[0, j], axes[1, j]
        ax.plot(dx, to, "o", color="#1f77b4", ms=4, label="观测 ΔT")
        ax.plot(dx, ts, "--", color="#d62728", lw=1.2, alpha=0.85, label="初值 ΔT")
        ax.plot(dx, tr, "-", color="0.15", lw=1.4, label="收回 ΔT")
        r0 = [a - b for a, b in zip(ts, to)]
        r1 = [a - b for a, b in zip(tr, to)]
        rms0 = math.sqrt(sum(v * v for v in r0) / len(r0)) if r0 else float("nan")
        rms1 = math.sqrt(sum(v * v for v in r1) / len(r1)) if r1 else float("nan")
        ax.set_title(f"{name}   {rms0:.3f} → {rms1:.3f} s", fontsize=10)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.set_ylabel("走时差 ΔT (s)")
            ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
        axr.axhline(0.0, color="0.45", lw=0.8)
        axr.plot(dx, r0, "s", color="#d62728", ms=3.5, alpha=0.7, label="初值−观测")
        axr.plot(dx, r1, "o", color="0.15", ms=4, label="收回−观测")
        axr.set_xlabel("偏移 dx (km)")
        axr.grid(True, alpha=0.3)
        if j == 0:
            axr.set_ylabel("残差 (s)")
            axr.legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.suptitle(title)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


def recs_for_draw(syn_path: Path):
    picks = parse_picks(syn_path.read_text(encoding="utf-8"))
    return [(int(p[0]), abs(p[1] - p[4]), p[3]) for p in picks]


def main() -> int:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--folder", type=Path, default=FOLDER)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--title", default="")
    p.add_argument(
        "--kappa",
        action="store_true",
        help="对照：inv_joint_kappa，Vs 初值=rec_vp/κ",
    )
    p.add_argument(
        "--678",
        dest="vs678",
        choices=("hot", "kappa"),
        default=None,
        help="B 步：冻 Vp，只反 6+7+8",
    )
    p.add_argument(
        "--td",
        action="store_true",
        help="历史 C：inv_678_td_hot（跳过 PSS 绝对走时）",
    )
    p.add_argument(
        "--td078",
        action="store_true",
        help="现场：inv_078_td_hot，PSS 绝对在、无 PSP、PPS−PPP",
    )
    p.add_argument(
        "--td2step",
        action="store_true",
        help="两步：inv_078_td_2step，先 PPS−PPP 收盖层再冻，PSS 只反面下",
    )
    p.add_argument(
        "--tdbelow",
        action="store_true",
        help="同一次 LSQR：inv_078_td_pssbelow，7 写盖层、8 只写面下",
    )
    p.add_argument(
        "--tdtruelid",
        action="store_true",
        help="对照：inv_078_td_truelid，盖层 Vp=真值，面下仍是 PPP",
    )
    p.add_argument(
        "--psptruelid",
        action="store_true",
        help="真盖层 Vp + PPP 面下，只反 PSP：inv_psp_truelid",
    )
    p.add_argument(
        "--ppplid",
        action="store_true",
        help="PPP 第二段：冻面下，只反盖层 Vp，inv_ppp_lid",
    )
    args = p.parse_args()
    phases = PHASES
    folder = args.folder
    if args.ppplid:
        folder = HERE / "inv_ppp_lid"
        phases = ((0, "PPP"),)
    elif args.psptruelid:
        folder = HERE / "inv_psp_truelid"
        phases = ((6, "PSP"),)
    elif args.tdtruelid:
        folder = HERE / "inv_078_td_truelid"
        phases = ((0, "PPP"), (7, "PPS"), (8, "PSS"))
    elif args.tdbelow:
        folder = HERE / "inv_078_td_pssbelow"
        phases = ((0, "PPP"), (7, "PPS"), (8, "PSS"))
    elif args.td2step:
        folder = HERE / "inv_078_td_2step"
        phases = ((0, "PPP"), (7, "PPS"), (8, "PSS"))
    elif args.td078:
        folder = HERE / "inv_078_td_hot"
        phases = ((0, "PPP"), (7, "PPS"), (8, "PSS"))
    elif args.td:
        folder = HERE / "inv_678_td_hot"
        phases = ((0, "PPP"), (6, "PSP"), (7, "PPS"), (8, "PSS"))
    elif args.vs678 == "hot":
        folder = HERE / "inv_678_hot"
    elif args.vs678 == "kappa":
        folder = HERE / "inv_678_kappa"
    elif args.kappa:
        folder = HERE / "inv_joint_kappa"
    if args.vs678:
        phases = ((6, "PSP"), (7, "PPS"), (8, "PSS"))
    out = args.out
    if out is None:
        if args.ppplid:
            out = HERE / "check_ppp_lid.png"
        elif args.psptruelid:
            out = HERE / "check_psp_truelid.png"
        elif args.tdtruelid:
            out = HERE / "check_078_td_truelid.png"
        elif args.tdbelow:
            out = HERE / "check_078_td_pssbelow.png"
        elif args.td2step:
            out = HERE / "check_078_td_2step.png"
        elif args.td078:
            out = HERE / "check_078_td_hot.png"
        elif args.td:
            out = HERE / "check_678_td_hot.png"
        elif args.vs678:
            out = HERE / f"check_678_{args.vs678}.png"
        else:
            out = HERE / ("check_joint_kappa.png" if args.kappa else "check_joint.png")
    title = args.title
    if not title:
        if args.ppplid:
            title = "PPP 分域  ①冻盖层收面下  ②冻面下收盖层    红=偏快 蓝=偏慢"
        elif args.psptruelid:
            title = "真盖层 Vp + PPP 面下  只反 PSP    红=偏快 蓝=偏慢"
        elif args.tdtruelid:
            title = "盖层 Vp=真值  面下 PPP  0+7+8 -td    红=偏快 蓝=偏慢"
        elif args.tdbelow:
            title = "同一次 LSQR  PPS写盖层  PSS只写面下  盖层不冻    红=偏快 蓝=偏慢"
        elif args.td2step:
            title = "两步  PPS−PPP收盖层后冻  PSS只反面下    红=偏快 蓝=偏慢"
        elif args.td078:
            title = "冻 Vp + PPS−PPP  PSS 绝对在、无 PSP    红=偏快 蓝=偏慢"
        elif args.td:
            title = "历史 C  冻 Vp + 走时差  跳过 PSS 绝对    红=偏快 蓝=偏慢"
        elif args.vs678 == "hot":
            title = "B 冻 Vp  6+7+8    Vs 盖层+面下 = rec_vp/κ + 0.50    红=偏快 蓝=偏慢"
        elif args.vs678 == "kappa":
            title = "B 冻 Vp  6+7+8    Vs = rec_vp/κ（无热扰动）    红=偏快 蓝=偏慢"
        elif args.kappa:
            title = "P1 对照  0+6+7+8    Vs = rec_vp/κ（无热扰动）    红=偏快 蓝=偏慢"
        else:
            title = "P1 四震相联合  0+6+7+8    Vs 盖层+面下 = rec_vp/κ + 0.50    红=偏快 蓝=偏慢"

    xs, zs, t_vs, s_vs, r_vs = load_pair(
        folder / "true_vs.smesh", folder / "start_vs.smesh", folder / "rec_vs.smesh"
    )
    _, _, t_vp, s_vp, r_vp = load_pair(
        folder / "true_vp.smesh", folder / "ppp_vp.smesh", folder / "rec_vp.smesh"
    )
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    obs, start_p, rec_p = folder / "syn_inv.dat", folder / "syn_start.dat", folder / "syn_rec.dat"
    rays_rec, recs_rec = pr.parse_rays(folder / "rays_rec.dat"), recs_for_draw(rec_p)
    rays_st, recs_st = pr.parse_rays(folder / "rays_start.dat"), recs_for_draw(start_p)

    print("phase   n_obs  start→rec t RMS")
    for code, name in phases:
        ok, n = npicks(obs, code)
        print(
            f"  {name:3s}({code})  {ok}/{n}   "
            f"{ttrms(obs, start_p, code):.4f} → {ttrms(obs, rec_p, code):.4f}"
        )
    print(
        f"  ALL      {ttrms(obs, start_p):.4f} → {ttrms(obs, rec_p):.4f}"
    )
    orig_vp = None
    if args.ppplid:
        orig_p = folder / "start_vp.smesh"
        st1_p = folder / "syn_stage1.dat"
        if orig_p.is_file():
            _, _, ov = m2.parse_smesh(orig_p)
            orig_vp = mask_crust(xs, zs, ov)
            print(
                f"  Vp orig-true   lid {region_rms(xs, zs, orig_vp, t_vp, lid=True):.4f}  "
                f"below {region_rms(xs, zs, orig_vp, t_vp, lid=False):.4f}  "
                f"crust {region_rms(xs, zs, orig_vp, t_vp, lid=None):.4f}"
            )
        if st1_p.is_file():
            print(
                f"  PPP t  orig→s1→rec  "
                f"{ttrms(obs, start_p):.4f} → {ttrms(obs, st1_p):.4f} → {ttrms(obs, rec_p):.4f}"
            )
        ix = min(range(len(xs)), key=lambda i: abs(xs[i] - 50.0))
        zi = g.z_conv(xs[ix])
        print(f"  lid Vp @ x={xs[ix]:.1f}  z_conv={zi:.2f}")
        for lab, grid in (
            ("true", t_vp),
            ("orig", orig_vp),
            ("s1  ", s_vp),
            ("rec ", r_vp),
        ):
            if grid is None:
                continue
            vals = []
            for k, z in enumerate(zs):
                if g.H - 1e-9 <= z < zi - 1e-9 and np.isfinite(grid[k, ix]):
                    vals.append((z, float(grid[k, ix])))
            if vals:
                print(
                    f"    {lab}  {vals[0][1]:.3f}@{vals[0][0]:.2f} → "
                    f"{vals[-1][1]:.3f}@{vals[-1][0]:.2f}"
                )
    if args.td or args.td078 or args.td2step or args.tdbelow or args.tdtruelid:
        print(
            f"  Δ PPS-PPP  {_drms(obs, start_p, 7, 0):.4f} → {_drms(obs, rec_p, 7, 0):.4f}"
        )
        if args.td:
            print(
                f"  Δ PSP-PPP  {_drms(obs, start_p, 6, 0):.4f} → {_drms(obs, rec_p, 6, 0):.4f}"
            )
            print(
                f"  Δ PSS-PSP  {_drms(obs, start_p, 8, 6):.4f} → {_drms(obs, rec_p, 8, 6):.4f}"
            )
        hold_t, hold_r = folder / "syn_holdout_true.dat", folder / "syn_holdout_rec.dat"
        if hold_t.is_file() and hold_r.is_file():
            print(
                f"  PSP holdout  {ttrms(hold_t, hold_r, 6):.4f}"
            )
            ht = parse_picks(hold_t.read_text(encoding="utf-8"))
            hr = parse_picks(hold_r.read_text(encoding="utf-8"))
            print(
                f"  PSS-(PPS-PPP) vs true PSP  "
                f"{_corrected_psp_rms(ht, ht):.4f} → {_corrected_psp_rms(ht, hr):.4f}"
            )
    ppp_obs, ppp_rec = folder / "syn_ppp.dat", folder / "syn_ppp_rec.dat"
    if ppp_obs.is_file() and ppp_rec.is_file():
        print(f"  PPP freeze-check  rec_vp vs true  {ttrms(ppp_obs, ppp_rec):.4f}")
    lid_vs = folder / "rec_vs_lid.smesh"
    if lid_vs.is_file():
        _, _, lidv = m2.parse_smesh(lid_vs)
        lidv = mask_crust(xs, zs, lidv)
        print(
            f"  Vs lid-step-true  lid {region_rms(xs, zs, lidv, t_vs, lid=True):.4f}  "
            f"below {region_rms(xs, zs, lidv, t_vs, lid=False):.4f}"
        )
        print(
            f"  Vs rec-lid-step   lid {region_rms(xs, zs, r_vs, lidv, lid=True):.4f}  "
            f"below {region_rms(xs, zs, r_vs, lidv, lid=False):.4f}"
        )
    for lab, a, b in (
        ("Vs start-true", s_vs, t_vs),
        ("Vs rec-true  ", r_vs, t_vs),
        ("Vs rec-start ", r_vs, s_vs),
        ("Vp start-true", s_vp, t_vp),
        ("Vp rec-true  ", r_vp, t_vp),
        ("Vp rec-start ", r_vp, s_vp),
    ):
        print(
            f"  {lab}  lid {region_rms(xs, zs, a, b, lid=True):.4f}  "
            f"below {region_rms(xs, zs, a, b, lid=False):.4f}  "
            f"crust {region_rms(xs, zs, a, b, lid=None):.4f}"
        )

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.4), facecolor="w", layout="constrained")
    rows = (
        (
            "Vs",
            s_vs - t_vs,
            r_vs - t_vs,
            r_vs - s_vs,
            (
                f"Vs start−true\nlid {region_rms(xs,zs,s_vs,t_vs,lid=True):.3f}  "
                f"below {region_rms(xs,zs,s_vs,t_vs,lid=False):.3f}",
                f"Vs rec−true\nlid {region_rms(xs,zs,r_vs,t_vs,lid=True):.3f}  "
                f"below {region_rms(xs,zs,r_vs,t_vs,lid=False):.3f}  "
                f"t {ttrms(obs, rec_p):.3f} s",
                "Vs rec−start",
            ),
            (rays_st, recs_st),
            (rays_rec, recs_rec),
        ),
        (
            "Vp",
            s_vp - t_vp,
            r_vp - t_vp,
            r_vp - s_vp,
            (
                (
                    f"Vp ①面下收回−true\nlid {region_rms(xs,zs,s_vp,t_vp,lid=True):.3f}  "
                    f"below {region_rms(xs,zs,s_vp,t_vp,lid=False):.3f}"
                    if args.ppplid
                    else f"Vp start−true (PPP 收回)\ncrust {region_rms(xs,zs,s_vp,t_vp,lid=None):.3f}"
                ),
                f"Vp rec−true\ncrust {region_rms(xs,zs,r_vp,t_vp,lid=None):.3f}",
                (
                    "Vp ②盖层增量（相对第 1 段）"
                    if args.ppplid
                    else "Vp rec−start（联合是否带偏）"
                ),
            ),
            (rays_st, recs_st),
            (rays_rec, recs_rec),
        ),
    )
    last = None
    legend_done = False
    fig_title = title
    for i, (_name, d0, d1, d2, titles, ray0, ray1) in enumerate(rows):
        ray_sets = (ray0, ray1, ray1)
        if i == 0:
            if args.ppplid:
                codes = (0,)
            elif args.psptruelid:
                codes = (6,)
            elif args.td078 or args.td2step or args.tdbelow or args.tdtruelid:
                codes = (7, 8)
            else:
                codes = (6, 7, 8)
        else:
            codes = (0,)
        for ax, grid, panel_title, (rays, recs) in zip(axes[i], (d0, d1, d2), titles, ray_sets):
            last = ax.imshow(
                grid, extent=extent, cmap="RdBu_r", vmin=-DCLIM, vmax=DCLIM,
                aspect="auto", interpolation="nearest", zorder=0,
            )
            if rays and recs:
                draw_ps_rays(
                    ax, rays, recs, thin=True, mark_conv=(ax is axes[i][1]),
                    z_conv=g.z_conv, codes=codes,
                )
                if not legend_done and i == 0 and not args.ppplid:
                    ax.legend(loc="upper right", framealpha=0.88, fontsize=6, ncol=2)
                    legend_done = True
            xs_l = np.linspace(18, 82, 80)
            ax.plot(xs_l, [g.z_conv(x) for x in xs_l], "k-.", lw=0.8, zorder=2)
            ax.plot(xs_l, [g.H] * len(xs_l), "k--", lw=0.6, zorder=2)
            ax.plot(list(OBS_XS), [g.H] * len(OBS_XS), "k^", ms=5, zorder=7)
            ax.set_xlim(18, 82)
            ax.set_ylim(16, 0)
            ax.set_title(panel_title, fontsize=10)
            ax.grid(True, alpha=0.25)
            if ax is axes[i, 0]:
                ax.set_ylabel("深度 (km)")
    fig.colorbar(last, ax=axes, shrink=0.55, label="ΔV (km/s)")
    fig.suptitle(fig_title)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    o_picks = parse_picks(obs.read_text(encoding="utf-8"))
    s_picks = parse_picks(start_p.read_text(encoding="utf-8"))
    r_picks = parse_picks(rec_p.read_text(encoding="utf-8"))
    tt_out = out.with_name(out.stem + "_ttimes.png")
    plot_phase_ttimes(o_picks, s_picks, r_picks, phases, tt_out, title + "    走时拟合")
    if args.td:
        plot_diff_ttimes(o_picks, s_picks, r_picks, HERE / "check_678_td_ttdiff.png")
    elif args.td078 or args.td2step or args.tdbelow or args.tdtruelid:
        if args.tdtruelid:
            dpng = HERE / "check_078_td_truelid_ttdiff.png"
        elif args.tdbelow:
            dpng = HERE / "check_078_td_pssbelow_ttdiff.png"
        elif args.td2step:
            dpng = HERE / "check_078_td_2step_ttdiff.png"
        else:
            dpng = HERE / "check_078_td_ttdiff.png"
        plot_diff_ttimes(
            o_picks,
            s_picks,
            r_picks,
            dpng,
            pairs=((7, 0, "PPS−PPP"),),
            title="PPS−PPP    台侧盖层 S−P（无 PSP）",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
