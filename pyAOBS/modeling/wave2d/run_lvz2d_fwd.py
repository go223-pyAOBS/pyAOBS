# -*- coding: utf-8 -*-
"""lvz2d 真模型弹性正演：OBS 竖力、水中记压力，叠射线 6/12。不改 inv_*。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pyAOBS.modeling.wave2d.elastic2d import _HAS_NUMBA, Gather, propagate, suggest_dt
from pyAOBS.modeling.wave2d.grid import resample_dual
from pyAOBS.modeling.wave2d.io_smesh import parse_pickfile


def _work_default() -> Path:
    return (
        HERE.parent
        / "tomo2d"
        / "example_water"
        / "PPP+PSP_inv2"
        / "psp_p_shoot"
        / "thin2km_rugged"
        / "lvz2d"
        / "inv_612"
    )


def _out_default() -> Path:
    return _work_default().parent / "wave_fwd"


def plot_gather(g: Gather, picks: list, out: Path, codes=(6, 12)) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    tr = np.asarray(g.data, dtype=float)
    peak = np.max(np.abs(tr), axis=1, keepdims=True)
    tr = tr / (peak + 1e-30)
    fig, ax = plt.subplots(figsize=(10.2, 7.2), facecolor="w", layout="constrained")
    dx = float(np.median(np.diff(g.rec_x))) if g.rec_x.size > 1 else 8.0
    scale = 0.42 * dx
    for k, x in enumerate(g.rec_x):
        w = tr[k] * scale
        ax.plot(x + w, g.t, color="0.15", lw=0.45)
        ax.fill_betweenx(g.t, x, x + np.clip(w, 0, None), color="0.15", lw=0, alpha=0.55)
    col = {6: "#1f77b4", 12: "#c44e8a"}
    lab = {6: "射线 6 PSP", 12: "射线 12 Moho"}
    drawn = set()
    sx = g.src_x
    for code, rx, _rz, tt, src in picks:
        if code not in codes:
            continue
        if abs(src - sx) > 0.2:
            continue
        ax.plot(
            rx,
            tt + g.delay,
            "o",
            ms=4.5,
            mew=0.8,
            mfc="none",
            color=col.get(code, "k"),
            label=lab[code] if code not in drawn else None,
        )
        drawn.add(code)
    ax.axvline(sx, color="0.4", ls="--", lw=0.7)
    ax.set_xlabel("炮点 x (km)")
    ax.set_ylabel("时间 (s)")
    ax.set_title(
        f"弹性正演 OBS={sx:.0f} km  vz源  水中压力   "
        f"f0={g.f0:g} Hz  dt={g.dt*1e3:.2f} ms  "
        f"{'numba' if _HAS_NUMBA else 'numpy'}"
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_xlim(0.0, 150.0)
    ax.set_ylim(g.t[-1], 0)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="lvz2d 真模型 2D 弹性正演（独立，不改射线代码）")
    p.add_argument("--work", type=Path, default=_work_default())
    p.add_argument("--out", type=Path, default=_out_default())
    p.add_argument("--obs", type=float, default=50.0)
    p.add_argument("--dx", type=float, default=0.10)
    p.add_argument("--tmax", type=float, default=16.0)
    p.add_argument("--f0", type=float, default=2.5)
    p.add_argument("--quick", action="store_true", help="缩小窗口，便于试跑")
    p.add_argument("--absorb", choices=("pml", "cerjan"), default="pml")
    args = p.parse_args()
    work = args.work
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    dx = 0.16 if args.quick else args.dx
    tmax = 10.0 if args.quick else args.tmax
    xmax = 90.0 if args.quick else None
    zmax = 12.0 if args.quick else None
    model = resample_dual(
        work / "true_vp.smesh",
        work / "true_vs.smesh",
        work / "seafloor.refl",
        dx=dx,
        dz=dx,
        xmax=xmax,
        zmax=zmax,
    )
    dt = suggest_dt(model.vp, model.dx, model.dz)
    picks = parse_pickfile(work / "syn_inv.dat")
    rec_x = sorted({rx for code, rx, _rz, _t, src in picks if abs(src - args.obs) < 0.2})
    rec_x = [x for x in rec_x if model.x[0] + 0.5 <= x <= model.x[-1] - 0.5]
    print(
        f"grid nx={model.x.size} nz={model.z.size} dx={model.dx} "
        f"dt={dt:.4e} nt={int(np.ceil(tmax/dt))+1} "
        f"vp {model.vp.min():.2f}-{model.vp.max():.2f} "
        f"water vs max={model.vs[model.water].max() if model.water.any() else 0:.3f} "
        f"nrec={len(rec_x)} engine={'numba' if _HAS_NUMBA else 'numpy'}"
    )
    g = propagate(
        model,
        src_x=args.obs,
        src_z=2.0,
        rec_x=np.asarray(rec_x, dtype=np.float64),
        rec_z=max(2.0 * dx, 0.20),
        tmax=tmax,
        f0=args.f0,
        dt=dt,
        src_kind="vz",
        absorb=args.absorb,
    )
    tag = f"obs{int(args.obs)}_dx{dx:.2f}_f{args.f0:g}_t{tmax:.0f}"
    np.savez_compressed(
        out / f"gather_{tag}.npz",
        t=g.t,
        rec_x=g.rec_x,
        data=g.data,
        src_x=g.src_x,
        delay=g.delay,
        dt=g.dt,
        f0=g.f0,
    )
    png = out / f"gather_{tag}.png"
    plot_gather(g, picks, png)
    peak = float(np.max(np.abs(g.data)))
    print(f"peak |p|={peak:.4e}  delay={g.delay:.3f}s  wrote {png}")
    if not np.isfinite(peak) or peak < 1e-30:
        print("WARNING: empty gather")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
