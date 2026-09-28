# -*- coding: utf-8 -*-
"""已有弹性道集 + 多震相射线，折合时间显示。不改 inv_*。"""

from __future__ import annotations

import argparse
import math
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

from pyAOBS.modeling.wave2d.elastic2d import Gather
from pyAOBS.modeling.wave2d.io_smesh import parse_pickfile
from pyAOBS.modeling.wave2d.phases import (
    CODES_P,
    CODES_S,
    PHASE_STYLE,
    VRED_P,
    VRED_S,
    write_geom,
)


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


def _load_gather(path: Path) -> Gather:
    z = np.load(path)
    return Gather(
        t=z["t"],
        rec_x=z["rec_x"],
        data=z["data"],
        src_x=float(z["src_x"]),
        src_z=2.0,
        delay=float(z["delay"]),
        dt=float(z["dt"]),
        f0=float(z["f0"]),
    )


def _tred(t: np.ndarray | float, x: float, src_x: float, vred: float) -> np.ndarray | float:
    return t - abs(float(x) - float(src_x)) / vred


def _draw_wiggles(ax, g: Gather, vred: float) -> None:
    tr = np.asarray(g.data, dtype=float)
    peak = np.max(np.abs(tr), axis=1, keepdims=True)
    tr = tr / (peak + 1e-30)
    dx = float(np.median(np.diff(g.rec_x))) if g.rec_x.size > 1 else 8.0
    scale = 0.42 * dx
    for k, x in enumerate(g.rec_x):
        w = tr[k] * scale
        y = _tred(g.t, float(x), g.src_x, vred)
        ax.plot(float(x) + w, y, color="0.15", lw=0.4)
        ax.fill_betweenx(y, float(x), float(x) + np.clip(w, 0, None), color="0.15", lw=0, alpha=0.5)


def _draw_picks(ax, picks, src_x: float, delay: float, vred: float, codes: tuple[int, ...]) -> None:
    drawn: set[int] = set()
    for code, rx, _rz, tt, src in picks:
        if code not in codes or abs(src - src_x) > 0.2:
            continue
        if not math.isfinite(tt) or tt <= 0.0:
            continue
        col, lab = PHASE_STYLE[code]
        ax.plot(
            rx,
            _tred(tt + delay, rx, src_x, vred),
            "o",
            ms=5.0,
            mew=0.9,
            mfc="none",
            color=col,
            label=lab if code not in drawn else None,
            zorder=5,
        )
        drawn.add(code)


def plot_reduced(
    g: Gather,
    picks: list,
    out: Path,
    *,
    vred_p: float = VRED_P,
    vred_s: float = VRED_S,
) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 9.2), facecolor="w", layout="constrained", sharex=True)
    for ax, vred, codes, title in (
        (axes[0], vred_p, CODES_P, f"P  折合 {vred_p:g} km/s   0 Pg  1 PmP"),
        (axes[1], vred_s, CODES_S, f"转换/S  折合 {vred_s:g} km/s   6/7/8/12/13"),
    ):
        _draw_wiggles(ax, g, vred)
        _draw_picks(ax, picks, g.src_x, g.delay, vred, codes)
        ax.axvline(g.src_x, color="0.45", ls="--", lw=0.7)
        ax.set_ylabel(f"t - |x-x0|/{vred:g} (s)")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0.0, 150.0)
        if codes == CODES_P:
            ax.set_ylim(6.5, 2.0)
        else:
            ax.set_ylim(8.0, -1.5)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9, ncol=2)
    axes[1].set_xlabel("炮点 x (km)")
    fig.suptitle(
        f"弹性道集 OBS={g.src_x:.0f} km  叠射线震相   f0={g.f0:g} Hz  "
        f"子波延迟 {g.delay:.2f} s",
        fontsize=12,
    )
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="多震相折合速度对比（不改射线代码）")
    p.add_argument("--work", type=Path, default=_work_default())
    p.add_argument("--out", type=Path, default=_out_default())
    p.add_argument("--npz", type=Path, default=None)
    p.add_argument("--syn", type=Path, default=None)
    p.add_argument("--vred-p", type=float, default=VRED_P)
    p.add_argument("--vred-s", type=float, default=VRED_S)
    p.add_argument("--write-geom-only", action="store_true")
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    geom = out / "geom_phases.dat"
    n = write_geom(geom)
    print(f"wrote {geom} nrec={n}")
    if args.write_geom_only:
        return 0
    npz = args.npz or (out / "gather_obs50_dx0.05_f4_t28.npz")
    if not npz.is_file():
        fallback = out / "gather_obs50_dx0.05_f4.npz"
        if fallback.is_file():
            npz = fallback
    syn = args.syn or (out / "syn_phases.dat")
    if not npz.is_file():
        raise SystemExit(f"missing gather {npz}")
    if not syn.is_file():
        raise SystemExit(f"missing {syn}  (先跑 run_ray_phases.sh)")
    g = _load_gather(npz)
    picks = parse_pickfile(syn)
    png = out / f"gather_obs{int(g.src_x)}_reduced.png"
    plot_reduced(g, picks, png, vred_p=args.vred_p, vred_s=args.vred_s)
    n_ok = sum(1 for c, _x, _z, t, s in picks if math.isfinite(t) and t > 0 and abs(s - g.src_x) < 0.2)
    print(f"OBS={g.src_x:.0f}  picks={n_ok}  wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
