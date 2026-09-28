#!/usr/bin/env python
"""
Compare Vp-angle from slope-derived ray parameter with RAYINVR r1.out.

Inputs:
- r1.out: columns include dist, red.time, f.angle, code
- r.in: read vred from &pltpar (default fallback 8.0)

Method:
1) Recover real travel time: t = red.time + |dist - xshot| / vred
   (matches RAYINVR: reduced t = t_real - |x_receiver - xshot| / vred; xshot from r.in &trapar).
2) For each ray code group, fit t(dist) with LocalLinear and estimate p=dt/ddist.
3) Compute incidence angles from p:
   - code 4.1: theta_v = asin(|p| * Vp)
   - code 5.1: theta_v = asin(|p| * Vs)
   - theta_h = 90 - theta_v           (from horizontal)
4) Compare with abs(f.angle) and (90 - abs(f.angle)).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from pyAOBS.visualization.iphase.theoretical_ppp_pps import fit_ppp_time_curve_local_linear


def parse_vred_from_rin(rin_path: Path, default: float = 8.0) -> float:
    text = rin_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"\bvred\s*=\s*([+-]?\d+(?:\.\d*)?)", text, flags=re.IGNORECASE)
    if not m:
        return float(default)
    try:
        v = float(m.group(1))
    except ValueError:
        return float(default)
    return v if v > 0 else float(default)


def parse_xshot_from_rin(rin_path: Path) -> tuple[float | None, list[float]]:
    """Return (first_xshot_or_none, all_xshot_values_found)."""
    text = rin_path.read_text(encoding="utf-8", errors="ignore")
    vals: list[float] = []
    for m in re.finditer(r"\bxshot\s*=\s*([+-]?\d+(?:\.\d*)?)", text, flags=re.IGNORECASE):
        try:
            vals.append(float(m.group(1)))
        except ValueError:
            continue
    if not vals:
        return None, []
    return vals[0], vals


def _blend(c: str | tuple, *, white: float) -> str:
    """Mix color with white (white in [0,1])."""
    r, g, b, a = mcolors.to_rgba(c)
    r = r * (1.0 - white) + white
    g = g * (1.0 - white) + white
    b = b * (1.0 - white) + white
    return mcolors.to_hex((r, g, b, a))


def _darken(c: str | tuple, *, fac: float) -> str:
    r, g, b, a = mcolors.to_rgba(c)
    r, g, b = r * fac, g * fac, b * fac
    return mcolors.to_hex((r, g, b, a))


def _angle_series_colors(code_idx: int, n_codes: int) -> tuple[str, str, str, str]:
    """Four distinct colors per code for θ_v, θ_h, |f.angle|, 90-|f.angle|."""
    # Spread hues across codes; four sat/lightness levels for curve types
    h = (code_idx / max(1, n_codes) + 0.02 * (code_idx % 3)) % 1.0
    theta_v = mcolors.to_hex(mcolors.hsv_to_rgb((h, 0.82, 0.95)))
    theta_h = mcolors.to_hex(mcolors.hsv_to_rgb(((h + 0.06) % 1.0, 0.72, 0.75)))
    f_h = mcolors.to_hex(mcolors.hsv_to_rgb(((h + 0.12) % 1.0, 0.55, 0.55)))
    f_v = mcolors.to_hex(mcolors.hsv_to_rgb(((h + 0.18) % 1.0, 0.45, 0.42)))
    return theta_v, theta_h, f_h, f_v


def parse_r1_out(r1_path: Path) -> dict[str, np.ndarray]:
    dist_list: list[float] = []
    redt_list: list[float] = []
    fangle_list: list[float] = []
    code_list: list[float] = []

    with r1_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.lower().startswith("shot"):
                continue
            parts = s.split()
            if len(parts) < 9:
                continue
            try:
                f_ang = float(parts[3])
                dist = float(parts[4])
                red_t = float(parts[6])
                code = float(parts[8])
            except ValueError:
                continue
            if np.isfinite(dist) and np.isfinite(red_t) and np.isfinite(f_ang) and np.isfinite(code):
                dist_list.append(dist)
                redt_list.append(red_t)
                fangle_list.append(f_ang)
                code_list.append(code)

    return {
        "dist": np.asarray(dist_list, dtype=float),
        "red_time": np.asarray(redt_list, dtype=float),
        "f_angle": np.asarray(fangle_list, dtype=float),
        "code": np.asarray(code_list, dtype=float),
    }


def estimate_p_from_fit(
    x: np.ndarray,
    t: np.ndarray,
    *,
    window_points: int = 11,
    dense_points: int = 401,
    smooth_half_win: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    if x.size < 5:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    so = np.argsort(x)
    xs = x[so]
    ts = t[so]
    finite = np.isfinite(xs) & np.isfinite(ts)
    xs = xs[finite]
    ts = ts[finite]
    if xs.size < 5:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    xg = np.linspace(float(np.min(xs)), float(np.max(xs)), int(max(101, dense_points)))
    tg = fit_ppp_time_curve_local_linear(
        xs,
        ts,
        xg,
        window_points=int(max(3, window_points | 1)),
        split_by_sign=False,
    )
    vg = np.isfinite(tg)
    xg = xg[vg]
    tg = tg[vg]
    if xg.size < 5:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    if smooth_half_win > 0 and xg.size > (2 * smooth_half_win + 2):
        w = 2 * smooth_half_win + 1
        kernel = np.ones(w, dtype=float) / w
        tg = np.convolve(tg, kernel, mode="same")

    p = np.gradient(tg, xg)
    return xg, p


def velocity_for_ray_code(code_val: float, vp: float, vs: float) -> tuple[float, str]:
    """
    Map RAYINVR ray code (last column in r1.out) to body-wave velocity at receiver.

    Default project rule:
      - 4.1 -> Vp
      - 5.1 -> Vs
    Other codes fall back to Vp with tag 'P?'.
    """
    c = float(code_val)
    if np.isclose(c, 4.1, atol=1e-3):
        return float(vp), "P"
    if np.isclose(c, 5.1, atol=1e-3):
        return float(vs), "S"
    return float(vp), "P?"


def collapse_to_single_valued_t_of_x(
    x: np.ndarray,
    t: np.ndarray,
    *,
    x_round_decimals: int = 3,
    mode: str = "min",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collapse duplicated/near-duplicated x to one t(x), so derivative is stable.

    r1.out can contain multiple rays at almost same dist (same code), which makes
    t(x) non-single-valued and causes slope spikes when fitted/derived directly.
    """
    xv = np.asarray(x, dtype=float)
    tv = np.asarray(t, dtype=float)
    v = np.isfinite(xv) & np.isfinite(tv)
    xv = xv[v]
    tv = tv[v]
    if xv.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    keys = np.round(xv, int(max(0, x_round_decimals)))
    uniq = np.unique(keys)
    xo: list[float] = []
    to: list[float] = []
    for k in uniq:
        m = keys == k
        xk = xv[m]
        tk = tv[m]
        if xk.size == 0:
            continue
        xo.append(float(np.mean(xk)))
        if mode == "median":
            to.append(float(np.median(tk)))
        else:
            # First-arrival style: choose minimum time at same/near same distance.
            to.append(float(np.min(tk)))

    xa = np.asarray(xo, dtype=float)
    ta = np.asarray(to, dtype=float)
    so = np.argsort(xa)
    return xa[so], ta[so]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare slope-based angle with r1.out f.angle")
    parser.add_argument("--r1", type=Path, default=Path("r1.out"), help="Path to r1.out")
    parser.add_argument("--rin", type=Path, default=Path("r.in"), help="Path to r.in")
    parser.add_argument("--vp", type=float, default=1.5, help="Vp (km/s), used for code 4.1")
    parser.add_argument("--vs", type=float, default=1.14, help="Vs (km/s), used for code 5.1")
    parser.add_argument(
        "--xshot",
        type=float,
        default=None,
        help="Shot x (km); default: parse first xshot= in r.in",
    )
    parser.add_argument("--window", type=int, default=11, help="LocalLinear window points")
    parser.add_argument("--smooth", type=int, default=5, help="Half-window for smoothing dense curve")
    parser.add_argument(
        "--collapse-mode",
        choices=["min", "median"],
        default="min",
        help="How to collapse duplicated dist values before fitting",
    )
    parser.add_argument("--code", type=float, default=None, help="Only analyze one code (e.g., 4.1)")
    parser.add_argument("--save", type=Path, default=Path("r1_angle_compare.png"), help="Output figure path")
    args = parser.parse_args()

    r1_path = args.r1.resolve()
    rin_path = args.rin.resolve()
    data = parse_r1_out(r1_path)
    if data["dist"].size == 0:
        raise RuntimeError(f"No valid rows parsed from {r1_path}")

    vred = parse_vred_from_rin(rin_path, default=8.0)
    vp = float(args.vp)
    vs = float(args.vs)
    if args.xshot is not None:
        xshot = float(args.xshot)
        xshot_all: list[float] = [xshot]
    else:
        x_first, xshot_all = parse_xshot_from_rin(rin_path)
        if x_first is None:
            raise RuntimeError(f"Could not parse xshot= from {rin_path}; pass --xshot explicitly.")
        xshot = float(x_first)
        uniq_x = sorted({round(v, 6) for v in xshot_all})
        if len(uniq_x) > 1:
            print(
                f"[WARN] Multiple xshot= found in r.in ({uniq_x}); using first={xshot}. "
                "Use --xshot to override."
            )

    dist = data["dist"]
    red_t = data["red_time"]
    f_angle_abs = np.abs(data["f_angle"])
    code = data["code"]
    offset = np.abs(dist - xshot)
    t_real = red_t + offset / vred

    if args.code is not None:
        m = np.isclose(code, float(args.code), atol=1e-6)
        dist = dist[m]
        red_t = red_t[m]
        t_real = t_real[m]
        f_angle_abs = f_angle_abs[m]
        code = code[m]
        if dist.size == 0:
            raise RuntimeError(f"No rows for code={args.code}")

    unique_codes = np.unique(np.round(code, 3))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)
    ax_t, ax_p, ax_a, ax_d = axes.ravel()

    stats_lines: list[str] = []
    n_codes = int(unique_codes.size)

    for i, c in enumerate(unique_codes):
        mc = np.isclose(code, c, atol=1e-6)
        xd = dist[mc]
        td = t_real[mc]
        rd = red_t[mc]
        fa = f_angle_abs[mc]
        if xd.size < 5:
            continue

        base = plt.cm.tab10(i % 10)
        c_red_sc = _blend(base, white=0.55)
        c_real_sc = base
        c_fit_ln = list(plt.cm.tab10.colors)[(i + 5) % 10]
        c_p = _darken(base, fac=0.75)
        cv, ch, cfh, cfv = _angle_series_colors(i, max(1, n_codes))
        c_dv, c_dh = cv, ch

        # scatter: reduced and real time (raw)
        ax_t.scatter(
            xd, rd, s=14, alpha=0.55, color=c_red_sc, marker="x",
            label=f"code {c:.1f} red.time",
        )
        ax_t.scatter(
            xd, td, s=14, alpha=0.7, edgecolors=c_real_sc, facecolors="none",
            linewidths=0.9, marker="o", label=f"code {c:.1f} t(real)",
        )

        # Collapse duplicated dist points to enforce single-valued t(dist)
        x_fit, t_fit_in = collapse_to_single_valued_t_of_x(
            xd, td, mode=args.collapse_mode
        )
        if x_fit.size < 5:
            continue

        xg, p = estimate_p_from_fit(
            x_fit,
            t_fit_in,
            window_points=int(args.window),
            dense_points=501,
            smooth_half_win=int(args.smooth),
        )
        if xg.size < 5:
            continue

        tg = fit_ppp_time_curve_local_linear(
            x_fit,
            t_fit_in,
            xg,
            window_points=int(max(3, args.window | 1)),
            split_by_sign=False,
        )
        vg = np.isfinite(tg)
        xg = xg[vg]
        p = p[vg]
        tg = tg[vg]
        if xg.size < 5:
            continue

        # interpolate r1 angle onto fit grid
        so_raw = np.argsort(xd)
        x_raw = xd[so_raw]
        fa_raw = fa[so_raw]
        ang_h_r1 = np.interp(xg, x_raw, fa_raw, left=np.nan, right=np.nan)  # horizontal convention
        ang_v_r1 = 90.0 - ang_h_r1

        v_body, wave_tag = velocity_for_ray_code(c, vp, vs)
        if wave_tag == "P?":
            print(f"[WARN] code {c:.3f} not 4.1/5.1; using Vp={vp} for angle (override mapping in script).")

        p_abs = np.abs(p)
        p_max = 1.0 / max(v_body, 1e-6)
        p_abs[p_abs >= (0.999 * p_max)] = np.nan
        pv = np.clip(p_abs * v_body, 0.0, 1.0)
        valid_p = np.isfinite(p_abs) & (p_abs < p_max)
        theta_v_p = np.full_like(p_abs, np.nan, dtype=float)
        theta_v_p[valid_p] = np.degrees(np.arcsin(pv[valid_p]))
        theta_h_p = 90.0 - theta_v_p
        th_lab = "θ_v(P)" if wave_tag == "P" else ("θ_v(S)" if wave_tag == "S" else "θ_v(?)")
        hh_lab = "θ_h(P)" if wave_tag == "P" else ("θ_h(S)" if wave_tag == "S" else "θ_h(?)")

        ax_t.plot(xg, tg, "-", color=c_fit_ln, lw=1.9, alpha=0.95, label=f"code {c:.1f} fit t")
        ax_p.plot(xg, p_abs, "-", color=c_p, lw=1.9, alpha=0.95, label=f"code {c:.1f} |p|")
        ax_a.plot(xg, theta_v_p, "-", color=cv, lw=2.0, alpha=0.95, label=f"code {c:.1f} {th_lab}")
        ax_a.plot(xg, theta_h_p, ":", color=ch, lw=2.0, alpha=0.95, label=f"code {c:.1f} {hh_lab}")
        ax_a.plot(xg, ang_h_r1, "--", color=cfh, lw=1.7, alpha=0.88, label=f"code {c:.1f} |f.angle|")
        ax_a.plot(xg, ang_v_r1, "-.", color=cfv, lw=1.7, alpha=0.88, label=f"code {c:.1f} 90-|f.angle|")

        dv = theta_v_p - ang_v_r1
        dh = theta_h_p - ang_h_r1
        vdv = np.isfinite(dv)
        vdh = np.isfinite(dh)
        if np.any(vdv):
            rms_v = float(np.sqrt(np.mean(dv[vdv] ** 2)))
            med_v = float(np.median(np.abs(dv[vdv])))
        else:
            rms_v = np.nan
            med_v = np.nan
        if np.any(vdh):
            rms_h = float(np.sqrt(np.mean(dh[vdh] ** 2)))
            med_h = float(np.median(np.abs(dh[vdh])))
        else:
            rms_h = np.nan
            med_h = np.nan
        stats_lines.append(
            f"code {c:.1f} ({wave_tag}, V={v_body:.3f}): N={xg.size}, RMS(θv-90+|f|)={rms_v:.2f}°, "
            f"MedAbs={med_v:.2f}°; RMS(θh-|f|)={rms_h:.2f}°, MedAbs={med_h:.2f}°"
        )

        ax_d.plot(xg, dv, "-", color=c_dv, lw=1.7, alpha=0.95, label=f"code {c:.1f} Δθ_v")
        ax_d.plot(xg, dh, ":", color=c_dh, lw=1.5, alpha=0.95, label=f"code {c:.1f} Δθ_h")

    ax_t.set_title("Travel Time from r1.out (red.time and recovered real t)")
    ax_t.set_xlabel("dist (km)")
    ax_t.set_ylabel("time (s)")
    ax_t.grid(True, alpha=0.4)

    ax_p.set_title("|Ray Parameter| from fitted t(dist)")
    ax_p.set_xlabel("dist (km)")
    ax_p.set_ylabel("|p| (s/km)")
    ax_p.grid(True, alpha=0.4)

    ax_a.set_title("Angle comparison (f.angle is from horizontal)")
    ax_a.set_xlabel("dist (km)")
    ax_a.set_ylabel("angle (deg)")
    ax_a.grid(True, alpha=0.4)

    ax_d.set_title("Angle residuals")
    ax_d.set_xlabel("dist (km)")
    ax_d.set_ylabel("computed - r1 (deg)")
    ax_d.axhline(0.0, color="k", lw=1.0, alpha=0.4)
    ax_d.grid(True, alpha=0.4)

    for a in (ax_t, ax_p, ax_a, ax_d):
        h, l = a.get_legend_handles_labels()
        if h:
            a.legend(fontsize=7, ncol=2, loc="best")

    txt = (
        f"r1={r1_path.name}, r.in={rin_path.name}, vred={vred:.3f}, xshot={xshot:.3f} km, "
        f"t_real=red.time+|dist-xshot|/vred; code4.1→Vp={vp:.3f}, code5.1→Vs={vs:.3f}\n"
        + ("\n".join(stats_lines) if stats_lines else "No valid groups")
    )
    fig.text(0.01, 0.01, txt, fontsize=9, family="monospace", va="bottom", ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    out = args.save.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"[OK] Figure saved: {out}")
    print(txt)


if __name__ == "__main__":
    main()
