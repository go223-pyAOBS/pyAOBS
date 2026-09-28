#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""共 p 射击 PSP 正演（一维剖面；本工区速度横向均匀，p 守恒）。

不改 tomo2d 图论 / 弯曲。type 6 对照若需要，调用现成 tt_forward。
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
sys.path.insert(0, str(CASE))
sys.path.insert(0, str(CASE.parent / "ps_inv"))
import make_ppp_psp_inv_case as m  # noqa: E402
import plot_ps_inv_models as psp  # noqa: E402

H, Z_CONV, ZMAX = m.H, m.Z_CONV, m.ZMAX
V_WATER, KAPPA = m.V_WATER, m.KAPPA_TRUE
OBS_XS, OBS_Z = m.OBS_XS, m.OBS_Z
SHOT_XS, SHOT_Z = m.SHOT_XS, m.SHOT_Z
P_COLOR, S_COLOR = "#1f77b4", "#c44e8a"


@dataclass(frozen=True)
class Profile1D:
    name: str
    lid_vp: float
    crust_vp: float
    vs0: float
    vs_grad: float
    sed0: float
    sed_grad: float

    @property
    def p_max_p(self) -> float:
        return 0.999 / max(self.lid_vp, 1e-6)

    @property
    def p_max_s(self) -> float:
        return 0.999 / max(self.vs0, 1e-6)

    @property
    def p_min_turn(self) -> float:
        vs_bot = self.vs0 + self.vs_grad * (ZMAX - Z_CONV)
        if vs_bot <= self.vs0:
            return self.p_max_s
        return 1.0 / vs_bot

    @property
    def p_hi(self) -> float:
        return min(self.p_max_p, self.p_max_s)

    @property
    def p_lo(self) -> float:
        return min(self.p_min_turn + 1e-4, self.p_hi - 1e-4)


def profile_slow() -> Profile1D:
    lid, crust = 4.00, 5.00
    return Profile1D(
        name="slow",
        lid_vp=lid,
        crust_vp=crust,
        vs0=crust / KAPPA,
        vs_grad=0.12,
        sed0=1.80,
        sed_grad=(lid - 1.80) / (Z_CONV - H),
    )


def profile_fast() -> Profile1D:
    lid, crust = 4.00, 7.20
    return Profile1D(
        name="fast",
        lid_vp=lid,
        crust_vp=crust,
        vs0=crust / KAPPA,
        vs_grad=0.12,
        sed0=1.80,
        sed_grad=(lid - 1.80) / (Z_CONV - H),
    )


def vp_at(pr: Profile1D, z: float) -> float:
    if z <= H + 1e-12:
        return V_WATER
    if z < Z_CONV:
        return pr.sed0 + pr.sed_grad * (z - H)
    return pr.lid_vp


def vs_at(pr: Profile1D, z: float) -> float:
    if z < Z_CONV:
        return pr.vs0
    return pr.vs0 + pr.vs_grad * (z - Z_CONV)


def _step_xz(z: float, dz: float, p: float, v: float) -> tuple[float, float, float] | None:
    pv = p * v
    if pv >= 0.999:
        return None
    denom = math.sqrt(max(1.0 - pv * pv, 1e-16))
    dx = abs(dz) * pv / denom
    dt = abs(dz) / (v * denom)
    return z + dz, dx, dt


def integrate_no_turn(
    z0: float, z1: float, p: float, vfun, *, dz: float = 0.02
) -> tuple[float, float, list[float], list[float]] | None:
    if abs(z1 - z0) < 1e-9:
        return 0.0, 0.0, [z0], [0.0]
    n = max(8, int(abs(z1 - z0) / dz))
    zs = np.linspace(z0, z1, n + 1)
    x = t = 0.0
    xs = [0.0]
    for a, b in zip(zs[:-1], zs[1:]):
        v = vfun(0.5 * (a + b))
        got = _step_xz(a, b - a, p, v)
        if got is None:
            return None
        _, dx, dt = got
        x += dx
        t += dt
        xs.append(x)
    return x, t, zs.tolist(), xs


def integrate_s_turn(
    pr: Profile1D, p: float, *, n: int = 72
) -> tuple[float, float, float, list[float], list[float]] | None:
    """线性梯度 Vs=vs0+g(z−zconv)：射线为圆。"""
    v0 = pr.vs0
    g = pr.vs_grad
    if g <= 1e-9 or p * v0 >= 0.999:
        return None
    c0 = math.sqrt(max(1.0 - (p * v0) ** 2, 0.0))
    if c0 < 1e-5:
        return None
    r = 1.0 / (p * g)
    x_half = c0 / (p * g)
    z_turn = Z_CONV + (1.0 / p - v0) / g
    if z_turn > ZMAX - 0.05 or z_turn <= Z_CONV + 0.05:
        return None
    t = (2.0 / g) * math.log((1.0 + c0) / (p * v0))
    x_tot = 2.0 * x_half
    zc = Z_CONV - v0 / g
    alpha = math.acos(max(-1.0, min(1.0, p * v0)))
    thetas = np.linspace(-alpha, alpha, n)
    xs = [x_half + r * math.sin(th) for th in thetas]
    zs = [zc + r * math.cos(th) for th in thetas]
    xs[0], zs[0] = 0.0, Z_CONV
    xs[-1], zs[-1] = x_tot, Z_CONV
    return x_tot, t, z_turn, zs, xs


@dataclass
class PspRay:
    xs: list[float]
    zs: list[float]
    p: float
    t: float
    z_turn: float
    offset: float


def trace_psp(pr: Profile1D, p: float) -> tuple[float, float, float, dict] | None:
    down = integrate_no_turn(SHOT_Z, Z_CONV, p, lambda z: vp_at(pr, z))
    up = integrate_no_turn(Z_CONV, OBS_Z, p, lambda z: vp_at(pr, z))
    sleg = integrate_s_turn(pr, p)
    if down is None or up is None or sleg is None:
        return None
    xd, td, zd, xdrel = down
    xu, tu, zu, xurel = up
    xs, ts, z_turn, zs_s, xs_s = sleg
    return xd + xs + xu, td + ts + tu, z_turn, {
        "xd": xd, "xu": xu, "xs": xs, "td": td, "tu": tu, "ts": ts,
        "zd": zd, "xdrel": xdrel, "zu": zu, "xurel": xurel,
        "zs_s": zs_s, "xs_s": xs_s,
    }


def assemble_path(shot_x: float, obs_x: float, pack: dict) -> tuple[list[float], list[float]]:
    sgn = 1.0 if obs_x >= shot_x else -1.0
    xs: list[float] = []
    zs: list[float] = []
    x0 = shot_x
    for xrel, z in zip(pack["xdrel"], pack["zd"]):
        xs.append(x0 + sgn * xrel)
        zs.append(z)
    x1 = xs[-1]
    for xrel, z in zip(pack["xs_s"][1:], pack["zs_s"][1:]):
        xs.append(x1 + sgn * xrel)
        zs.append(z)
    x2 = xs[-1]
    # up-leg table is conv → OBS, xrel from 0
    for xrel, z in zip(pack["xurel"][1:], pack["zu"][1:]):
        xs.append(x2 + sgn * xrel)
        zs.append(z)
    return xs, zs


def match_offset(pr: Profile1D, offset: float) -> PspRay | None:
    if offset < 1.0:
        return None
    p_lo, p_hi = pr.p_lo, pr.p_hi
    if p_hi <= p_lo + 1e-5:
        return None
    ps = np.linspace(p_lo, p_hi, 48)
    rows = []
    for p in ps:
        got = trace_psp(pr, float(p))
        if got is None:
            continue
        x, t, z_turn, pack = got
        rows.append((float(p), x, t, z_turn, pack))
    if len(rows) < 2:
        return None
    best = None
    for (p0, x0, t0, z0, pack0), (p1, x1, t1, z1, pack1) in zip(rows, rows[1:]):
        if (x0 - offset) * (x1 - offset) > 0:
            continue
        if abs(x1 - x0) < 1e-9:
            continue
        a = (offset - x0) / (x1 - x0)
        p = p0 + a * (p1 - p0)
        got = trace_psp(pr, p)
        if got is None:
            continue
        x, t, z_turn, pack = got
        cand = (abs(x - offset), -z_turn, p, t, z_turn, pack, x)
        if best is None or cand < best:
            best = cand
    if best is None:
        # 最近偏移（窗口内可能只有大偏移枝）
        p, x, t, z_turn, pack = min(rows, key=lambda r: abs(r[1] - offset))
        if abs(x - offset) > 4.0:
            return None
        best = (abs(x - offset), -z_turn, p, t, z_turn, pack, x)
    _, _, p, t, z_turn, pack, x = best
    return PspRay([], [], p, t, z_turn, x)


def shoot_fan(pr: Profile1D, shot_x: float, sign: float, n: int = 10) -> list[PspRay]:
    """不绑观测点，把窗口内的 p 全打出去，看续至簇。"""
    out: list[PspRay] = []
    for p in np.linspace(pr.p_lo, pr.p_hi, n):
        got = trace_psp(pr, float(p))
        if got is None:
            continue
        x, t, z_turn, pack = got
        dummy_obs = shot_x + sign * max(x, 1.0)
        xs, zs = assemble_path(shot_x, dummy_obs, pack)
        out.append(PspRay(xs, zs, float(p), t, z_turn, x))
    return out


def shoot_geom(pr: Profile1D, shot_stride: int = 2) -> list[PspRay]:
    rays: list[PspRay] = []
    shots = SHOT_XS[::shot_stride]
    for ox in OBS_XS:
        for sx in shots:
            off = abs(ox - sx)
            hit = match_offset(pr, off)
            if hit is None:
                continue
            got = trace_psp(pr, hit.p)
            if got is None:
                continue
            x, t, z_turn, pack = got
            if abs(x - off) > 2.5:
                continue
            xs, zs = assemble_path(sx, ox, pack)
            rays.append(PspRay(xs, zs, hit.p, t, z_turn, x))
    if len(rays) < 6:
        rays.extend(shoot_fan(pr, 22.0, 1.0, n=9))
        rays.extend(shoot_fan(pr, 78.0, -1.0, n=9))
    return rays


def vp_kw(pr: Profile1D) -> dict:
    return dict(
        sed0=pr.sed0,
        sed_grad=pr.sed_grad,
        crust0=pr.crust_vp,
        crust_grad=KAPPA * pr.vs_grad,
    )


def write_case_files(pr: Profile1D) -> Path:
    d = HERE / pr.name
    d.mkdir(parents=True, exist_ok=True)
    vp = d / "true_vp.smesh"
    mixed = d / "true_mixed.smesh"
    m.write_vp(vp, **vp_kw(pr))
    m.write_mixed_smesh(vp, vp, mixed, below_kappa=KAPPA)
    m.write_seafloor(d / "seafloor.refl")
    m.write_conv(d / "conv.refl")
    m.write_geom(d / "geom_psp.dat", codes=m.CODES)
    return d


def wsl_path(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        return f"/mnt/{s[0].lower()}{s[2:]}"
    return s


def run_tt_forward_type6(pr: Profile1D) -> Path | None:
    d = write_case_files(pr)
    bin_dir = CASE.parents[1] / "src" / "build-tomo2d" / "tt_forward"
    if not bin_dir.is_file() and not (CASE.parents[1] / "src" / "build-tomo2d").is_dir():
        print(f"skip tt_forward ({pr.name}): binary dir missing")
        return None
    rays = d / "rays_graph6.dat"
    syn = d / "syn_graph6.dat"
    n = "-N8/8/0.8/8/1e-4/1e-5"
    cmd = (
        f"cd {wsl_path(d)} && {wsl_path(bin_dir)} "
        f"-Mtrue_mixed.smesh -Ggeom_psp.dat -Xconv.refl -Bseafloor.refl "
        f"{n} -Rrays_graph6.dat > syn_graph6.dat"
    )
    print(f"== tt_forward type 6  {pr.name} ==")
    r = subprocess.run(["wsl", "bash", "-lc", cmd], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        print(f"tt_forward failed ({pr.name}), 只画共 p")
        return None
    if not rays.is_file():
        return None
    print(f"wrote {rays}  {syn}")
    return rays


def _zmax(rays: list) -> float:
    zm = 0.0
    for r in rays:
        zs = r.zs if isinstance(r, PspRay) else r[1]
        if zs:
            zm = max(zm, max(zs))
    return zm


def _draw_bg(ax, mixed: Path, title: str) -> None:
    xs, zs, vel = m.parse_smesh(mixed)
    grid = psp._grid(xs, zs, vel)
    extent = (float(xs[0]), float(xs[-1]), float(zs[-1]), float(zs[0]))
    ax.imshow(
        grid, extent=extent, cmap="RdYlBu_r", vmin=1.2, vmax=5.8,
        aspect="auto", interpolation="nearest", zorder=0,
    )
    ax.axhline(H, color="0.15", ls="--", lw=0.9, zorder=2)
    ax.axhline(Z_CONV, color="0.15", ls="-.", lw=1.0, zorder=2)
    ax.set_xlim(18, 82)
    ax.set_ylim(ZMAX, 0.0)
    ax.set_title(title)
    ax.grid(True, alpha=0.28)


def _draw_shoot(ax, rays: list[PspRay]) -> None:
    for i, ray in enumerate(rays):
        xs, zs = ray.xs, ray.zs
        for j in range(len(xs) - 1):
            s = 0.5 * (zs[j] + zs[j + 1]) > Z_CONV + 1e-3
            ax.plot(
                xs[j : j + 2], zs[j : j + 2],
                color=S_COLOR if s else P_COLOR,
                lw=0.85, alpha=0.8, zorder=3,
                label="P / S（共 p）" if i == 0 and j == 0 else None,
            )
    ax.scatter(list(OBS_XS), [OBS_Z] * len(OBS_XS), marker="^", c="k", s=28, zorder=5, label="OBS")
    ax.scatter(list(SHOT_XS[::4]), [SHOT_Z] * len(SHOT_XS[::4]), marker="o", c="C1", s=10, zorder=5, label="炮")


def _draw_graph(ax, rayfile: Path | None) -> None:
    ax.scatter(list(OBS_XS), [OBS_Z] * len(OBS_XS), marker="^", c="k", s=28, zorder=5, label="OBS")
    if rayfile is None or not rayfile.is_file():
        ax.text(50, 8, "无 type 6 射线文件\n（未改图论，对照未跑）", ha="center", va="center", fontsize=9)
        return
    segs = psp.parse_rays(rayfile)
    zm = 0.0
    for i, (xs, zs) in enumerate(segs):
        zm = max(zm, max(zs) if zs else 0.0)
        ax.plot(xs, zs, color=P_COLOR, lw=0.45, ls="--", alpha=0.55, zorder=3,
                label="type 6 图论" if i == 0 else None)
    ax.set_xlabel("模型距离 (km)")


def plot_compare(
    cases: list[tuple[Profile1D, list[PspRay], Path | None, Path]],
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(len(cases), 2, figsize=(13.4, 4.8 * len(cases)),
                             facecolor="w", layout="constrained", squeeze=False)
    for row, (pr, shoot, graph, mixed) in enumerate(cases):
        vs_lid = pr.vs0
        note = "Vs顶 < 盖层底Vp" if vs_lid < pr.lid_vp else "Vs顶 > 盖层底Vp"
        _draw_bg(
            axes[row, 0], mixed,
            f"{pr.name}  图论 type 6  zmax={_zmax(psp.parse_rays(graph) if graph and graph.is_file() else []):.2f} km\n"
            f"盖层底Vp={pr.lid_vp:.2f}  面下Vs={vs_lid:.2f}  {note}",
        )
        _draw_graph(axes[row, 0], graph)
        _draw_bg(
            axes[row, 1], mixed,
            f"{pr.name}  共 p 射击  n={len(shoot)}  zmax={_zmax(shoot):.2f} km\n"
            f"p∈[{pr.p_lo:.3f},{pr.p_hi:.3f}]  两端 Snell，S 转折"
            + (f"  临界偏移~{np.nanmin([r.offset for r in shoot]):.0f} km" if shoot else ""),
        )
        _draw_shoot(axes[row, 1], shoot)
        axes[row, 0].set_ylabel("深度 (km)")
        axes[row, 1].set_xlabel("模型距离 (km)")
        if row == 0:
            axes[row, 0].legend(loc="upper right", fontsize=7, framealpha=0.9)
            axes[row, 1].legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.suptitle(
        "PSP 正演对照：左=现成 tt_forward type 6（图论+弯曲，未改源码）；右=一维共 p 射击",
        fontsize=12,
    )
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


def plot_xp(pr: Profile1D, out_png: Path) -> None:
    ps = np.linspace(pr.p_lo, pr.p_hi, 40)
    xs, zts = [], []
    for p in ps:
        got = trace_psp(pr, float(p))
        if got is None:
            xs.append(np.nan)
            zts.append(np.nan)
        else:
            xs.append(got[0])
            zts.append(got[2])
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor="w", layout="constrained")
    ax.plot(ps, xs, "C0-o", ms=3, label="X(p) 共 p PSP")
    ax.set_xlabel("p (s/km)")
    ax.set_ylabel("炮检距 (km)")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{pr.name}  X(p)  Vs0={pr.vs0:.2f}  窗口 [{pr.p_lo:.3f},{pr.p_hi:.3f}]")
    ax2 = ax.twinx()
    ax2.plot(ps, zts, "C3--s", ms=3, label="z_turn")
    ax2.set_ylabel("转折深度 (km)")
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"wrote {out_png}")


def report(pr: Profile1D, rays: list[PspRay]) -> None:
    print(
        f"{pr.name}: 盖层底 Vp={pr.lid_vp:.2f}  Vs0={pr.vs0:.2f}  "
        f"p窗口 [{pr.p_lo:.4f},{pr.p_hi:.4f}]  命中 {len(rays)} 条  "
        f"zmax={_zmax(rays):.2f} km"
    )
    if rays:
        ps = [r.p for r in rays]
        print(f"  p [{min(ps):.4f},{max(ps):.4f}]  T [{min(r.t for r in rays):.2f},{max(r.t for r in rays):.2f}] s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-graph6", action="store_true", help="不调用 tt_forward")
    args = ap.parse_args()
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    packed = []
    for pr in (profile_slow(), profile_fast()):
        write_case_files(pr)
        rays = shoot_geom(pr)
        report(pr, rays)
        plot_xp(pr, HERE / f"xp_{pr.name}.png")
        existing = HERE / pr.name / "rays_graph6.dat"
        if args.no_graph6:
            g6 = existing if existing.is_file() else None
        elif existing.is_file():
            g6 = existing
        else:
            g6 = run_tt_forward_type6(pr)
        packed.append((pr, rays, g6, HERE / pr.name / "true_mixed.smesh"))
    plot_compare(packed, HERE / "check_psp_p_shoot.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
