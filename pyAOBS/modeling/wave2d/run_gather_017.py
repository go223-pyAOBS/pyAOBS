# -*- coding: utf-8 -*-
"""OBS 弹性道集：默认 OBS 为源（竖力）、水中记压力，叠 PPP/PmP/PPS。

水柱理论曲线：t=√(Δx²+(nH)²)/vw + delay（n=1,3,5；H=水深）。
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
from matplotlib.figure import Figure
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pyAOBS.modeling.wave2d.elastic2d import _HAS_NUMBA, Gather, propagate, suggest_dt
from pyAOBS.modeling.wave2d.grid import resample_dual
from pyAOBS.modeling.wave2d.io_smesh import parse_pickfile, parse_smesh

CODES = (0, 1, 7)
STYLE = {
    0: ("#e41a1c", "PPP (0 Pg)"),
    1: ("#d95f02", "PmP (1)"),
    7: ("#17becf", "PPS (7)"),
}


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


def build_rec_x(obs: float, offset: float, drec: float, xmin: float, xmax: float) -> np.ndarray:
    lo = max(xmin + 0.5 * drec, obs - offset)
    hi = min(xmax - 0.5 * drec, obs + offset)
    # 对齐到 obs±n*drec，避开 OBS 正上方
    n_lo = int(math.floor((obs - lo) / drec))
    n_hi = int(math.floor((hi - obs) / drec))
    xs = [obs - i * drec for i in range(n_lo, 0, -1)]
    xs += [obs + i * drec for i in range(1, n_hi + 1)]
    return np.asarray(xs, dtype=np.float64)


def write_geom(path: Path, obs: float, rec_x: np.ndarray, codes=CODES) -> int:
    # 射线叠点稀疏一些：每 1 km，减轻 tt_forward
    step = max(1, int(round(1.0 / float(np.median(np.diff(rec_x))))))
    xs = list(rec_x[::step])
    if abs(xs[-1] - rec_x[-1]) > 1e-6:
        xs.append(float(rec_x[-1]))
    lines = ["1", f"s  {obs:8.3f}     2.000 {len(xs) * len(codes):4d}"]
    for code in codes:
        for x in xs:
            lines.append(f"r  {x:8.3f}     0.010 {code:4d}     0.000     0.050")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(xs) * len(codes)


def run_tt_forward(work: Path, out: Path, geom: Path) -> Path:
    bin_dir = HERE.parent / "tomo2d" / "src" / "build-tomo2d"
    syn = out / "syn_obs50_017.dat"
    rays = out / "rays_obs50_017.dat"
    log = out / "fwd_obs50_017.log"
    # WSL 路径
    def wsl(p: Path) -> str:
        s = str(p.resolve()).replace("\\", "/")
        if len(s) >= 2 and s[1] == ":":
            return f"/mnt/{s[0].lower()}{s[2:]}"
        return s

    cmd = [
        "wsl",
        "-e",
        "bash",
        "-lc",
        (
            f'export TOMO2D_FWD_OMP=1 OMP_NUM_THREADS=8; '
            f'cd "{wsl(work)}" && '
            f'"{wsl(bin_dir)}/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh '
            f'-G"{wsl(geom)}" -Xconv.refl -Bseafloor.refl -Fmoho_true.refl -A '
            f'-N8/8/0.8/8/1e-4/1e-5 -R"{wsl(rays)}" '
            f'> "{wsl(syn)}" 2>"{wsl(log)}"'
        ),
    ]
    print("running:", " ".join(cmd[-1][:80]), "...")
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise SystemExit(f"tt_forward failed rc={r.returncode}; see {log}")
    return syn


def _density_clim(data: np.ndarray, pclip: float = 98.0) -> float:
    """对齐 zplot density：色标 = abs 振幅的 pclip 百分位。"""
    flat = np.abs(np.asarray(data, dtype=np.float64).ravel())
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 1.0
    if flat.size > 250_000:
        flat = flat[:: max(1, flat.size // 200_000)]
    clim = float(np.percentile(flat, pclip))
    if not np.isfinite(clim) or clim < 1e-30:
        clim = float(np.max(flat)) if flat.size else 1.0
    return max(clim, 1e-30)


def plot_section(
    g: Gather,
    picks: list,
    out: Path,
    offset: float,
    *,
    pclip: float = 98.0,
    vred: float | None = None,
    trace_norm: bool = True,
    layout: str = "reciprocal",
    water_h: float = 2.0,
    water_v: float = 1.5,
    water_n: tuple[int, ...] = (1, 3, 5),
    tred_max: float | None = 12.0,
) -> None:
    """变密度剖面（zplot density）：灰度 imshow，±pclip。

    vred 非空时纵轴为折合时间 t-|x-x0|/vred；波形按道插值到统一折合网格。
    水柱理论曲线：t = sqrt(Δx² + (nH)²)/vw + Ricker delay（OBS 海底源；n=1,3 为主）。
    vred 图上对同一绝对走时做 t-|x|/vred。
    """
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    order = np.argsort(g.rec_x)
    xs = np.asarray(g.rec_x, dtype=float)[order]
    data = np.asarray(g.data, dtype=np.float64)[order]  # (nrec, nt)
    if trace_norm:
        peak = np.max(np.abs(data), axis=1, keepdims=True)
        data = data / (peak + 1e-30)
    sx = g.src_x
    dt = float(g.dt)

    if vred is not None and vred > 0:
        tred_cols = [g.t - abs(float(x) - sx) / vred for x in xs]
        tred_lo = float(min(c[0] for c in tred_cols))
        tred_hi = float(max(c[-1] for c in tred_cols))
        y = np.arange(tred_lo, tred_hi + 0.5 * dt, dt, dtype=np.float64)
        gather = np.zeros((y.size, xs.size), dtype=np.float32)
        for k, tk in enumerate(tred_cols):
            gather[:, k] = np.interp(y, tk, data[k], left=0.0, right=0.0).astype(
                np.float32
            )
        ylab = f"t - |x-x0|/{vred:g} (s)"
        mode = f"density 折合 {vred:g} km/s"
    else:
        y = np.asarray(g.t, dtype=np.float64)
        gather = data.T.astype(np.float32)
        ylab = "时间 (s)"
        mode = "density"

    clim = _density_clim(gather, pclip=pclip)
    drec = float(np.median(np.diff(xs))) if xs.size > 1 else 0.2
    x0 = float(xs[0]) - 0.5 * drec
    x1 = float(xs[-1]) + 0.5 * drec
    fig = Figure(figsize=(12.5, 8.0), facecolor="w", layout="constrained")
    ax = fig.add_subplot(111)
    ax.imshow(
        gather,
        extent=(x0, x1, float(y[-1]), float(y[0])),
        aspect="auto",
        cmap="gray",
        vmin=-clim,
        vmax=clim,
        interpolation="nearest",
        origin="upper",
        zorder=0,
    )
    drawn: set[int] = set()
    pick_y: list[float] = []
    for code, rx, _rz, tt, src in picks:
        if code not in CODES or abs(src - sx) > 0.2:
            continue
        if not math.isfinite(tt) or tt <= 0:
            continue
        col, lab = STYLE[code]
        ty = tt + g.delay
        if vred is not None and vred > 0:
            ty = ty - abs(rx - sx) / vred
        pick_y.append(ty)
        ax.plot(
            rx,
            ty,
            "o",
            ms=3.4,
            mew=0.9,
            mfc="none",
            color=col,
            label=lab if code not in drawn else None,
            zorder=5,
        )
        drawn.add(code)
    # 水柱理论曲线：OBS 在海底为源，t=sqrt(x^2+(nH)^2)/vw + delay
    # n=1 直达；n=3,5,… 为主要 peg（零偏移 → nH/v）
    if water_h > 0 and water_v > 0:
        off = xs - sx
        for n in water_n:
            t_abs = np.sqrt(off**2 + (n * water_h) ** 2) / water_v + float(g.delay)
            if vred is not None and vred > 0:
                ty = t_abs - np.abs(off) / vred
            else:
                ty = t_abs
            is3 = n == 3
            ax.plot(
                xs,
                ty,
                color="#16a34a" if is3 else "#4ade80",
                ls="-" if is3 else "--",
                lw=1.5 if is3 else 1.0,
                alpha=0.95 if is3 else 0.82,
                zorder=3,
                label=(
                    f"水柱 n={n}: √(x²+({n}H)²)/v +d"
                    if is3 or n in (1, 2)
                    else None
                ),
            )
    ax.axvline(sx, color="#f59e0b", ls="--", lw=0.9, zorder=4)
    ax.set_xlabel("炮点 x (km)")
    ax.set_ylabel(ylab)
    ax.set_title(
        f"弹性剖面 {mode}  OBS源 z=2 km  水中检波  "
        f"±{offset:g} km   PPP/PmP/PPS  f0={g.f0:g} Hz  pclip={pclip:g}"
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
    ax.set_xlim(sx - offset - 2.0, sx + offset + 2.0)
    # 折合时间默认显示到 tred_max（12 s）；绝对时间仍跟数据/拾取
    use_tred_max = (
        tred_max is not None and tred_max > 0 and vred is not None and vred > 0
    )
    if use_tred_max:
        # 折合时间从 0 画到 tred_max（时间向下增大）
        ax.set_ylim(float(tred_max), 0.0)
    elif pick_y:
        y_hi = max(pick_y) + 1.5
        y_lo = min(pick_y) - 0.8
        ax.set_ylim(min(y_hi, float(y[-1])), max(y_lo, float(y[0])))
    else:
        ax.set_ylim(float(y[-1]), float(y[0]))
    ax.grid(True, alpha=0.15, color="0.5")
    fig.savefig(out, dpi=150)


def propagate_water_to_obs(
    model,
    *,
    obs_x: float,
    obs_z: float,
    shot_x: np.ndarray,
    src_z: float,
    tmax: float,
    f0: float,
    dt: float,
    src_kind: str,
    absorb: str,
) -> Gather:
    """浅水多炮 → 单台 OBS：每炮一次正演，道集 x = 炮点。"""
    n = int(shot_x.size)
    traces = None
    t = delay = None
    for k, sx in enumerate(shot_x):
        g1 = propagate(
            model,
            src_x=float(sx),
            src_z=src_z,
            rec_x=np.asarray([obs_x], dtype=np.float64),
            rec_z=obs_z,
            tmax=tmax,
            f0=f0,
            dt=dt,
            src_kind=src_kind,
            absorb=absorb,
        )
        if traces is None:
            traces = np.zeros((n, g1.data.shape[1]), dtype=np.float64)
            t = g1.t
            delay = g1.delay
        traces[k] = g1.data[0]
        if (k + 1) % 10 == 0 or k == 0 or k + 1 == n:
            print(f"  shot {k + 1}/{n}  x={sx:.1f} km", flush=True)
    assert traces is not None and t is not None and delay is not None
    return Gather(
        t=t,
        rec_x=np.asarray(shot_x, dtype=np.float64),
        data=traces,
        src_x=float(obs_x),
        src_z=float(src_z),
        delay=float(delay),
        dt=float(dt),
        f0=float(f0),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, default=_work_default())
    p.add_argument("--out", type=Path, default=_out_default())
    p.add_argument("--obs", type=float, default=50.0)
    p.add_argument("--offset", type=float, default=80.0)
    p.add_argument("--drec", type=float, default=0.2)
    p.add_argument("--dx", type=float, default=0.10)
    p.add_argument("--tmax", type=float, default=22.0)
    p.add_argument("--f0", type=float, default=3.0)
    p.add_argument("--skip-ray", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--vred", type=float, default=8.0, help="折合速度 km/s；<=0 用绝对时间")
    p.add_argument("--pclip", type=float, default=98.0)
    p.add_argument("--absorb", choices=("pml", "cerjan"), default="pml")
    p.add_argument(
        "--layout",
        choices=("water-obs", "reciprocal"),
        default="reciprocal",
        help="reciprocal: OBS竖力→水中压力（默认）；water-obs: 浅水多炮→OBS",
    )
    p.add_argument("--src-z", type=float, default=0.01, help="浅水炮深度 km（water-obs）")
    p.add_argument("--obs-z", type=float, default=2.0, help="OBS 深度 km（源）")
    p.add_argument(
        "--src-kind",
        choices=("expl", "vz", "vx"),
        default="expl",
        help="water-obs 默认爆炸源；reciprocal 用 vz",
    )
    args = p.parse_args()
    res = run_obs_gather(
        work=args.work,
        out=args.out,
        vp=args.work / "true_vp.smesh",
        vs=args.work / "true_vs.smesh",
        seafloor=args.work / "seafloor.refl",
        obs=args.obs,
        offset=args.offset,
        drec=args.drec,
        dx=args.dx,
        tmax=args.tmax,
        f0=args.f0,
        skip_ray=args.skip_ray,
        quick=args.quick,
        vred=args.vred,
        pclip=args.pclip,
        absorb=args.absorb,
        layout=args.layout,
        src_z=args.src_z,
        obs_z=args.obs_z,
        src_kind=args.src_kind,
    )
    return 0 if res.peak > 1e-30 else 1


@dataclass
class GatherOutputs:
    npz: Path
    png: Path
    png_reduced: Path | None
    log: str
    peak: float


def run_obs_gather(
    *,
    work: Path,
    out: Path,
    vp: Path,
    vs: Path,
    seafloor: Path,
    obs: float = 50.0,
    offset: float = 80.0,
    drec: float = 0.2,
    dx: float = 0.10,
    tmax: float = 22.0,
    f0: float = 3.0,
    skip_ray: bool = True,
    quick: bool = False,
    vred: float = 8.0,
    pclip: float = 98.0,
    absorb: str = "pml",
    layout: str = "reciprocal",
    src_z: float = 0.01,
    obs_z: float = 2.0,
    src_kind: str = "expl",
    syn: Path | None = None,
    water_h: float = 2.0,
    water_v: float = 1.5,
    tred_max: float = 12.0,
) -> GatherOutputs:
    """OBS 为源的弹性道集。供 CLI 与 tomo2d GUI 共用。"""
    lines: list[str] = []

    def say(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    for label, path in (("vp", vp), ("vs", vs), ("seafloor", seafloor)):
        if not Path(path).is_file():
            raise FileNotFoundError(f"缺少 {label}: {path}")
    out.mkdir(parents=True, exist_ok=True)
    dx_use = 0.16 if quick else float(dx)
    tmax_use = 12.0 if quick else float(tmax)
    xs, _zs, _v = parse_smesh(vp)
    xmax_m = float(xs[-1])
    model = resample_dual(vp, vs, seafloor, dx=dx_use, dz=dx_use)
    shot_x = build_rec_x(obs, offset, drec, float(model.x[0]), float(model.x[-1]))
    dt = suggest_dt(model.vp, model.dx, model.dz)
    say(
        f"model x={model.x[0]:.1f}-{model.x[-1]:.1f} (smesh xmax={xmax_m:.1f})  "
        f"nx={model.x.size} nz={model.z.size} dx={dx_use}  "
        f"nshot={shot_x.size}  offset L={obs - shot_x[0]:.1f} R={shot_x[-1] - obs:.1f}  "
        f"dt={dt:.4e} nt={int(np.ceil(tmax_use / dt)) + 1}  "
        f"engine={'numba' if _HAS_NUMBA else 'numpy'}"
    )
    if layout == "water-obs":
        say(
            f"layout=water-obs  炮 z≈{src_z:g} km ({src_kind}) → OBS=({obs:g},{obs_z:g})  "
            f"共 {shot_x.size} 炮"
        )
        g = propagate_water_to_obs(
            model,
            obs_x=obs,
            obs_z=obs_z,
            shot_x=shot_x,
            src_z=src_z,
            tmax=tmax_use,
            f0=f0,
            dt=dt,
            src_kind=src_kind,
            absorb=absorb,
        )
        tag = (
            f"obs{int(obs)}_off{int(offset)}_d{drec:.1f}_f{f0:g}"
            f"_waterobs_srcz{src_z:g}"
        )
        rec_z_saved = obs_z
    else:
        rec_z = max(2.0 * dx_use, src_z)
        say(
            f"layout=reciprocal  OBS源 z={obs_z:g} → 水中检波 z={rec_z:g}  "
            f"（走时 ≡ 浅水炮→OBS）"
        )
        g = propagate(
            model,
            src_x=obs,
            src_z=obs_z,
            rec_x=shot_x,
            rec_z=rec_z,
            tmax=tmax_use,
            f0=f0,
            dt=dt,
            src_kind="vz",
            absorb=absorb,
        )
        g = Gather(
            t=g.t,
            rec_x=g.rec_x,
            data=g.data,
            src_x=g.src_x,
            src_z=rec_z,
            delay=g.delay,
            dt=g.dt,
            f0=g.f0,
        )
        tag = f"obs{int(obs)}_off{int(offset)}_d{drec:.1f}_f{f0:g}_recip"
        rec_z_saved = obs_z
    j_src = int(np.argmin(np.abs(model.z - src_z)))
    say(f"shot grid z={model.z[j_src]:.3f} km (requested {src_z:g}); OBS z={obs_z:g}")
    npz = out / f"gather_{tag}.npz"
    np.savez_compressed(
        npz,
        t=g.t,
        rec_x=g.rec_x,
        data=g.data,
        src_x=g.src_x,
        src_z=g.src_z,
        rec_z=rec_z_saved,
        layout=layout,
        delay=g.delay,
        dt=g.dt,
        f0=g.f0,
    )
    geom = out / "geom_obs50_017.dat"
    nray = write_geom(geom, obs, shot_x)
    say(f"wrote {geom} nray={nray}")
    syn_path = Path(syn) if syn else out / "syn_obs50_017.dat"
    picks: list = []
    if not skip_ray:
        syn_path = run_tt_forward(work, out, geom)
        picks = parse_pickfile(syn_path)
    elif syn_path.is_file():
        picks = parse_pickfile(syn_path)
        say(f"overlay {syn_path}")
    else:
        say(f"无射线走时（{syn_path} 不存在），只画波场与水柱曲线")
    plot_kw = dict(
        pclip=pclip,
        layout=layout,
        water_h=water_h,
        water_v=water_v,
        tred_max=None,
    )
    png = out / f"gather_{tag}_017.png"
    plot_section(g, picks, png, offset, vred=None, **plot_kw)
    png_r: Path | None = None
    if vred and vred > 0:
        png_r = out / f"gather_{tag}_017_vred{vred:g}.png"
        plot_section(
            g, picks, png_r, offset, vred=vred, tred_max=tred_max, **{
                k: plot_kw[k] for k in ("pclip", "layout", "water_h", "water_v")
            },
        )
        say(f"wrote {png_r}")
    peak = float(np.max(np.abs(g.data)))
    say(f"peak |p|={peak:.4e}  wrote {npz}  {png}")
    return GatherOutputs(
        npz=npz,
        png=png,
        png_reduced=png_r,
        log="\n".join(lines),
        peak=peak,
    )


if __name__ == "__main__":
    raise SystemExit(main())
