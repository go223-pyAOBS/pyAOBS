# -*- coding: utf-8 -*-
"""对照：水中压力（竖力）/ 互易水平质点速度 / 加强转换面后的水平分量。"""

from __future__ import annotations

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

from pyAOBS.modeling.wave2d.elastic2d import Gather, propagate, suggest_dt
from pyAOBS.modeling.wave2d.grid import boost_conv_impedance, resample_dual
from pyAOBS.modeling.wave2d.io_smesh import load_xz, parse_pickfile
from pyAOBS.modeling.wave2d.run_gather_017 import (
    CODES,
    STYLE,
    _density_clim,
    _work_default,
    build_rec_x,
    plot_section,
)


def _panel(ax, g: Gather, picks, *, vred: float, title: str, offset: float) -> None:
    order = np.argsort(g.rec_x)
    xs = np.asarray(g.rec_x, dtype=float)[order]
    data = np.asarray(g.data, dtype=np.float64)[order]
    peak = np.max(np.abs(data), axis=1, keepdims=True)
    data = data / (peak + 1e-30)
    sx = float(g.src_x)
    dt = float(g.dt)
    cols = [g.t - abs(float(x) - sx) / vred for x in xs]
    y = np.arange(min(c[0] for c in cols), max(c[-1] for c in cols) + 0.5 * dt, dt)
    gather = np.zeros((y.size, xs.size), dtype=np.float32)
    for k, tk in enumerate(cols):
        gather[:, k] = np.interp(y, tk, data[k], left=0.0, right=0.0)
    clim = _density_clim(gather, 98.0)
    drec = float(np.median(np.diff(xs)))
    ax.imshow(
        gather,
        extent=(xs[0] - 0.5 * drec, xs[-1] + 0.5 * drec, float(y[-1]), float(y[0])),
        aspect="auto",
        cmap="gray",
        vmin=-clim,
        vmax=clim,
        interpolation="nearest",
        origin="upper",
    )
    drawn = set()
    ys = []
    for code, rx, _rz, tt, src in picks:
        if code not in CODES or abs(src - sx) > 0.2 or not np.isfinite(tt) or tt <= 0:
            continue
        col, lab = STYLE[code]
        ty = tt + g.delay - abs(rx - sx) / vred
        ys.append(ty)
        ax.plot(
            rx, ty, "o", ms=2.2, mew=0.6, mfc="none", color=col,
            label=lab if code not in drawn else None, zorder=5,
        )
        drawn.add(code)
    ax.axvline(sx, color="#f59e0b", ls="--", lw=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(f"t-|x-x0|/{vred:g} (s)")
    ax.set_xlim(sx - offset - 2, sx + offset + 2)
    if ys:
        ax.set_ylim(max(ys) + 1.2, min(ys) - 0.6)
    ax.grid(True, alpha=0.15, color="0.5")
    ax.legend(loc="lower right", fontsize=6, framealpha=0.9, ncol=3)


def main() -> int:
    work = _work_default()
    out = work.parent / "wave_fwd"
    out.mkdir(parents=True, exist_ok=True)
    obs, offset, drec, dx, tmax, f0, vred = 50.0, 80.0, 0.2, 0.10, 22.0, 3.0, 8.0
    base = resample_dual(
        work / "true_vp.smesh", work / "true_vs.smesh", work / "seafloor.refl", dx=dx, dz=dx
    )
    conv = load_xz(work / "conv.refl")
    strong = boost_conv_impedance(base, conv, lid_vs_scale=0.55, below_vs_scale=1.25)
    rec_x = build_rec_x(obs, offset, drec, float(base.x[0]), float(base.x[-1]))
    dt = suggest_dt(base.vp, base.dx, base.dz)
    picks = parse_pickfile(out / "syn_obs50_017.dat")
    print(f"nrec={rec_x.size} dt={dt:.4e} nt={int(np.ceil(tmax/dt))+1}")

    def run(model, kind: str, tag: str) -> Gather:
        print(f"propagate {tag} ...")
        g = propagate(
            model, src_x=obs, src_z=2.0, rec_x=rec_x, rec_z=max(2 * dx, 0.2),
            tmax=tmax, f0=f0, dt=dt, src_kind=kind, absorb="pml",
        )
        np.savez_compressed(
            out / f"gather_{tag}.npz",
            t=g.t, rec_x=g.rec_x, data=g.data, src_x=g.src_x,
            delay=g.delay, dt=g.dt, f0=g.f0,
        )
        return g

    g_vx = run(base, "vx", "obs50_vx_base")
    g_vx_s = run(strong, "vx", "obs50_vx_strong")
    z = np.load(out / "gather_obs50_off80_d0.2_f3.npz")
    g_vz = Gather(
        t=z["t"], rec_x=z["rec_x"], data=z["data"], src_x=float(z["src_x"]),
        src_z=2.0, delay=float(z["delay"]), dt=float(z["dt"]), f0=float(z["f0"]),
    )

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 1, figsize=(12.2, 14.5), facecolor="w", layout="constrained", sharex=True)
    _panel(axes[0], g_vz, picks, vred=vred, offset=offset,
           title="水中压力 · 竖力源（现有）  ≈ OBS 垂直向")
    _panel(axes[1], g_vx, picks, vred=vred, offset=offset,
           title="水中压力 · 水平力源（互易）  ≈ OBS 固体水平质点速度 vx")
    _panel(axes[2], g_vx_s, picks, vred=vred, offset=offset,
           title="同上 · 转换面 Vs 加强（盖层×0.55，面下×1.25）")
    axes[2].set_xlabel("炮点 x (km)")
    fig.suptitle(f"OBS={obs:.0f} km  折合 {vred:g} km/s  C-PML  f0={f0:g} Hz", fontsize=12)
    png = out / "gather_obs50_vx_vs_strong.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    plot_section(g_vx, picks, out / "gather_obs50_vx_base_vred8.png", offset, vred=vred)
    plot_section(g_vx_s, picks, out / "gather_obs50_vx_strong_vred8.png", offset, vred=vred)
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
