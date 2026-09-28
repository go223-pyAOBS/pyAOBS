#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""灵敏度加权 ON vs 原 inv_612 / inv_pps_lid(7)。不改原工区文件。"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "inv_2d"))
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT.parents[1].parent / "water_inv"))
import inv_grid as g  # noqa: E402
import make_inv_612 as case612  # noqa: E402
import make_ppp_psp_inv_case as m2  # noqa: E402
from check_joint import mask_crust, ttrms  # noqa: E402
from make_lvz import BEL_LVZ, LID_LVZ  # noqa: E402

g.z_conv = case612.z_conv

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OFF612 = HERE / "inv_612"
ON612 = HERE / "inv_sens_w" / "612"
OFF7 = HERE / "inv_pps_lid"
ON7 = HERE / "inv_sens_w" / "lid7"


def region_rms(xs, zs, a, b, *, lid: bool | None, xlo: float, xhi: float) -> float:
    s = n = 0
    for i, x in enumerate(xs):
        if not (xlo <= x <= xhi):
            continue
        zi = case612.z_conv(x)
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


def _load_xz(path: Path) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        a = ln.split()
        if len(a) >= 2:
            out.append((float(a[0]), float(a[1])))
    return out


def _z_at(xz: list[tuple[float, float]], x: float) -> float:
    return float(np.interp(x, [p[0] for p in xz], [p[1] for p in xz]))


def _moho_rms(rec, true, xlo=10.0, xhi=140.0) -> float:
    s = n = 0
    for x, zt in true:
        if not (xlo <= x <= xhi):
            continue
        d = _z_at(rec, x) - zt
        s += d * d
        n += 1
    return math.sqrt(s / n) if n else float("nan")


def _pair(folder: Path, rec_name: str):
    xs, zs, t_raw = m2.parse_smesh(folder / "true_vs.smesh")
    _, _, s_raw = m2.parse_smesh(folder / "start_vs.smesh")
    _, _, r_raw = m2.parse_smesh(folder / rec_name)
    return xs, zs, t_raw, s_raw, r_raw, mask_crust(xs, zs, t_raw), mask_crust(xs, zs, r_raw)


def _log_flags(path: Path) -> str:
    if not path.is_file():
        return "missing"
    bits = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ln.startswith("# sens_weight") or ln.startswith("# lsqr_precond"):
            bits.append(ln.strip())
        if ln[:1].isdigit():
            break
    return " | ".join(bits) if bits else "no header"


def main() -> int:
    print("=== log flags ===")
    print(f"  612 OFF  {_log_flags(OFF612 / 'inv.log')}")
    print(f"  612 ON   {_log_flags(ON612 / 'inv.log')}")
    print(f"  lid7 OFF {_log_flags(OFF7 / 'inv_7.log')}")
    print(f"  lid7 ON  {_log_flags(ON7 / 'inv.log')}")

    xs, zs, t_raw, s_raw, r_off_raw, t_vs, r_off = _pair(OFF612, "rec_vs.smesh")
    _, _, _, _, r_on_raw, _, r_on = _pair(ON612, "rec_vs.smesh")
    true_m = _load_xz(OFF612 / "moho_true.refl")
    start_m = _load_xz(OFF612 / "moho.refl")
    off_m = _load_xz(OFF612 / "rec_moho.refl")
    on_m = _load_xz(ON612 / "rec_moho.refl")
    xlo, xhi = 10.0, 140.0
    print("\n=== 612  面下 Vs + 莫霍  (冻盖层) ===")
    print(
        f"  start   lid {region_rms(xs, zs, mask_crust(xs, zs, s_raw), t_vs, lid=True, xlo=xlo, xhi=xhi):.4f}"
        f"  below {region_rms(xs, zs, mask_crust(xs, zs, s_raw), t_vs, lid=False, xlo=xlo, xhi=xhi):.4f}"
        f"  moho {_moho_rms(start_m, true_m):.4f} km"
    )
    print(
        f"  OFF     lid {region_rms(xs, zs, r_off, t_vs, lid=True, xlo=xlo, xhi=xhi):.4f}"
        f"  below {region_rms(xs, zs, r_off, t_vs, lid=False, xlo=xlo, xhi=xhi):.4f}"
        f"  moho {_moho_rms(off_m, true_m):.4f} km"
    )
    print(
        f"  ON      lid {region_rms(xs, zs, r_on, t_vs, lid=True, xlo=xlo, xhi=xhi):.4f}"
        f"  below {region_rms(xs, zs, r_on, t_vs, lid=False, xlo=xlo, xhi=xhi):.4f}"
        f"  moho {_moho_rms(on_m, true_m):.4f} km"
    )
    for lab, syn_r, rec_vs in (
        ("OFF", OFF612 / "syn_rec.dat", OFF612 / "rec_vp.smesh"),
        ("ON", ON612 / "syn_rec.dat", ON612 / "rec_vp.smesh"),
    ):
        syn_t = OFF612 / "syn_inv.dat"
        print(
            f"  tt {lab:3s}  6={ttrms(syn_t, syn_r, 6):.4f} s  "
            f"12={ttrms(syn_t, syn_r, 12):.4f} s"
        )
        _, _, vt = m2.parse_smesh(OFF612 / "true_vp.smesh")
        _, _, vr = m2.parse_smesh(rec_vs)
        mx = max(abs(a - b) for col_t, col_r in zip(vt, vr) for a, b in zip(col_t, col_r))
        print(f"  Vp {lab:3s}  max|d|={mx:.4f}")

    xs7, zs7, t7_raw, s7_raw, r7off_raw, t7, r7off = _pair(OFF7, "rec_vs_7.smesh")
    _, _, _, _, r7on_raw, _, r7on = _pair(ON7, "rec_vs.smesh")
    s7 = mask_crust(xs7, zs7, s7_raw)
    xlo7, xhi7 = 15.0, 95.0
    print("\n=== lid PPS7  面上 Vs  (冻面下) ===")
    print(
        f"  start  lid {region_rms(xs7, zs7, s7, t7, lid=True, xlo=xlo7, xhi=xhi7):.4f}"
        f"  below {region_rms(xs7, zs7, s7, t7, lid=False, xlo=xlo7, xhi=xhi7):.4f}"
    )
    print(
        f"  OFF    lid {region_rms(xs7, zs7, r7off, t7, lid=True, xlo=xlo7, xhi=xhi7):.4f}"
        f"  below {region_rms(xs7, zs7, r7off, t7, lid=False, xlo=xlo7, xhi=xhi7):.4f}"
    )
    print(
        f"  ON     lid {region_rms(xs7, zs7, r7on, t7, lid=True, xlo=xlo7, xhi=xhi7):.4f}"
        f"  below {region_rms(xs7, zs7, r7on, t7, lid=False, xlo=xlo7, xhi=xhi7):.4f}"
    )
    syn_t7 = OFF7 / "syn_inv_7.dat"
    print(f"  tt OFF  7={ttrms(syn_t7, OFF7 / 'syn_rec_7.dat', 7):.4f} s")
    print(f"  tt ON   7={ttrms(syn_t7, ON7 / 'syn_rec.dat', 7):.4f} s")

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.2), facecolor="w", layout="constrained")
    dclim_b, dclim_l = 0.80, 0.50
    panels = (
        (axes[0, 0], r_off - t_vs, (xs[0], xs[-1], zs[-1], zs[0]), -dclim_b, dclim_b, 10, 140, 16, "612 OFF  反−真"),
        (axes[0, 1], r_on - t_vs, (xs[0], xs[-1], zs[-1], zs[0]), -dclim_b, dclim_b, 10, 140, 16, "612 ON  反−真"),
        (axes[1, 0], r7off - t7, (xs7[0], xs7[-1], zs7[-1], zs7[0]), -dclim_l, dclim_l, 15, 95, 6.2, "lid7 OFF  反−真"),
        (axes[1, 1], r7on - t7, (xs7[0], xs7[-1], zs7[-1], zs7[0]), -dclim_l, dclim_l, 15, 95, 6.2, "lid7 ON  反−真"),
    )
    last = None
    xs_m = np.linspace(10, 140, 200)
    for i, (ax, arr, extent, vmin, vmax, xlo_p, xhi_p, zhi, title) in enumerate(panels):
        ax.imshow(
            arr,
            extent=extent,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
            zorder=0,
        )
        ax.plot(xs_m, [case612.z_conv(x) for x in xs_m], "k-.", lw=0.7, zorder=2)
        ax.plot(xs_m, [g.H] * len(xs_m), "k--", lw=0.5, zorder=2)
        if i < 2:
            ax.plot([p[0] for p in true_m], [p[1] for p in true_m], "k:", lw=1.1, zorder=3)
            mo = off_m if i == 0 else on_m
            ax.plot([p[0] for p in mo], [p[1] for p in mo], color="#c44e8a", lw=1.3, zorder=4)
            ax.plot(BEL_LVZ["x0"], BEL_LVZ["z0"], "kx", ms=7, mew=1.3, zorder=8)
        else:
            ax.plot(LID_LVZ["x0"], LID_LVZ["z0"], "kx", ms=7, mew=1.3, zorder=8)
        ax.set_xlim(xlo_p, xhi_p)
        ax.set_ylim(zhi, 0)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.25)
        last = ax.images[-1]
        if i % 2 == 0:
            ax.set_ylabel("深度 (km)")
        if i >= 2:
            ax.set_xlabel("x (km)")
    fig.colorbar(last, ax=axes.ravel().tolist(), shrink=0.82, label="ΔVs (km/s)")
    fig.suptitle("灵敏度加权  OFF vs ON   612=面下+莫霍 · lid7=面上  （原工区未改）", fontsize=12)
    out = HERE / "inv_sens_w" / "check_inv_sens_w.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
