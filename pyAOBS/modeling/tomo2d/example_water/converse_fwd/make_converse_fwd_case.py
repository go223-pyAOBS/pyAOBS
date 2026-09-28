#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 tt_forward 折合 PSP（raytype 6）正演工区。

混合慢度：转换面以上是 P（水 + 沉积盖层梯度 Vp），以下是壳幔 Vs 梯度。
OBS 在海底、炮在海面。0 仍是 Fermat 初至；6 须 ``-X``，转换点按整条射线最短选。
"""

from __future__ import annotations

import math
from pathlib import Path

V_WATER = 1.5
# 沉积盖层：海底 1.80 → 转换面 3.15，与面下 Vs0=3.40 对比不大。
VP_SED0 = 1.80
VP_SED_GRAD = 0.45
VS0 = 3.40
VS_GRAD = 0.12  # km/s per km，界面下递增
V_AIR = 0.33
H = 2.0
Z_CONV = 5.0
# 兼容旧名：界面处盖层 Vp。
V_P_CRUST = VP_SED0 + VP_SED_GRAD * (Z_CONV - H)
XMIN, XMAX, DX = 0.0, 100.0, 2.0
ZMAX, DZ = 10.0, 0.2
OBS_X = 50.0
SHOT_Z = 0.01
# 相对台的偏移（km）。短偏移看零偏差，长偏移看 0/6 分叉。
OFFSETS = (0.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0)


def vp_sed(z: float) -> float:
    """盖层 P：海底 VP_SED0，向下线性增加到转换面。"""
    return VP_SED0 + VP_SED_GRAD * max(0.0, min(z, Z_CONV) - H)


def expected_lid_mean() -> float:
    return 0.5 * (vp_sed(H) + vp_sed(Z_CONV))


def t_lid_vertical() -> float:
    """盖层垂直单程走时（线性梯度）。"""
    if VP_SED_GRAD < 1e-12:
        return (Z_CONV - H) / VP_SED0
    return math.log(vp_sed(Z_CONV) / vp_sed(H)) / VP_SED_GRAD


def v_at(z: float, *, vs0: float = VS0, vs_grad: float = VS_GRAD) -> float:
    # 界面结点划到 Vs 侧，过面段按面下 Vs 计时。
    if z >= Z_CONV - 1e-9:
        return vs0 + vs_grad * max(0.0, z - Z_CONV)
    if z > H + 1e-9:
        return vp_sed(z)
    return V_WATER


def _xs() -> list[float]:
    n = int(round((XMAX - XMIN) / DX)) + 1
    return [XMIN + i * DX for i in range(n)]


def _zs() -> list[float]:
    n = int(round(ZMAX / DZ)) + 1
    return [i * DZ for i in range(n)]


def write_smesh(
    path: Path,
    *,
    vs0: float = VS0,
    vs_grad: float = VS_GRAD,
) -> None:
    xs, zs = _xs(), _zs()
    nx, nz = len(xs), len(zs)
    lines = [
        f"{nx} {nz} {V_WATER:.4f} {V_AIR}",
        " ".join(f"{x:.4f}" for x in xs),
        " ".join("0.0000" for _ in xs),
        " ".join(f"{z:.4f}" for z in zs),
    ]
    for _x in xs:
        lines.append(" ".join(f"{v_at(z, vs0=vs0, vs_grad=vs_grad):.4f}" for z in zs))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_seafloor(path: Path) -> None:
    xs = _xs()
    path.write_text("".join(f"{x:.4f} {H:.4f}\n" for x in xs), encoding="utf-8")


def write_conv(path: Path) -> None:
    xs = _xs()
    path.write_text("".join(f"{x:.4f} {Z_CONV:.4f}\n" for x in xs), encoding="utf-8")


def _fmt_s_line(x: float, z: float, npick: int) -> str:
    return f"s{x:10.3f}{z:10.3f}{npick:5d}"


def _fmt_r_line(x: float, z: float, kind: int, t: float, u: float) -> str:
    return f"r{x:10.3f}{z:10.3f}{kind:5d}{t:10.3f}{u:10.3f}"


def write_geom(path: Path, *, codes: tuple[int, ...] = (0, 6)) -> None:
    shots = [OBS_X + dx for dx in OFFSETS]
    nrcv = len(shots) * len(codes)
    lines = ["1", _fmt_s_line(OBS_X, H, nrcv)]
    for kind in codes:
        for x in shots:
            lines.append(_fmt_r_line(x, SHOT_Z, kind, 0.0, 0.0))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def t_head_wave(dx: float) -> float:
    """海面炮→海底台：水柱直达，或沿海底的临界头波（用海底沉积 Vp）。"""
    dx = abs(dx)
    vert = H - SHOT_Z
    t_dir = math.hypot(dx, vert) / V_WATER
    if VP_SED0 <= V_WATER + 1e-9:
        return t_dir
    sini = V_WATER / VP_SED0
    cosi = math.sqrt(max(0.0, 1.0 - sini * sini))
    xcrit = vert * sini / cosi
    if dx <= xcrit + 1e-9:
        return t_dir
    t_hw = vert / (V_WATER * cosi) + (dx - xcrit) / VP_SED0
    return min(t_dir, t_hw)


def t_psp_ref(dx: float, *, vs0: float = VS0) -> float:
    """折合 PSP 参考：两端盖层垂直 P + 界面 Vs0 走偏移（无下潜、无 Snell 斜 P）。"""
    dx = abs(float(dx))
    return (H - SHOT_Z) / V_WATER + 2.0 * t_lid_vertical() + dx / vs0


def write_analytic(path: Path) -> None:
    rows = [
        "# dx_km  t0_s  t6_s  t6_minus_t0_s",
        "# t0 = 水柱直达或海底头波；t6 = 折合 PSP 参考（界面 S，无下潜）",
    ]
    for dx in OFFSETS:
        t0, t6 = t_head_wave(dx), t_psp_ref(dx)
        rows.append(f"{dx:.3f}  {t0:.6f}  {t6:.6f}  {t6 - t0:.6f}")
    rows.append(
        f"# H={H} Zc={Z_CONV} Vw={V_WATER} Vp_sed={VP_SED0}+{VP_SED_GRAD}(z-H) "
        f"Vs0={VS0} zero-offset t6≈{(H - SHOT_Z) / V_WATER + 2.0 * t_lid_vertical():.6f}"
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def compare_syn_to_analytic(
    syn_text: str, *, atol: float = 99.0
) -> list[tuple[int, float, float, float, float]]:
    """解析 tt_forward stdout，返回 (code, dx, t_syn, t_ref, err)。"""
    lines = [ln.strip() for ln in syn_text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("空的正演走时")
    recs: list[tuple[int, float, float, float, float]] = []
    i = 1
    src_x = OBS_X
    while i < len(lines):
        parts = lines[i].split()
        i += 1
        if not parts:
            continue
        if parts[0] == "s":
            src_x = float(parts[1])
            nrcv = int(float(parts[-1]))
            for _ in range(nrcv):
                rp = lines[i].split()
                i += 1
                x = float(rp[1])
                code = int(float(rp[3]))
                t = float(rp[4])
                dx = abs(x - src_x)
                if code == 0:
                    ref = t_head_wave(dx)
                elif code == 6:
                    ref = t_psp_ref(dx)
                else:
                    ref = float("nan")
                recs.append((code, dx, t, ref, t - ref))
        else:
            continue
    bad = [r for r in recs if r[3] == r[3] and abs(r[4]) > atol]
    if bad:
        msg = "; ".join(
            f"code={c} dx={dx:g} syn={ts:.4f} ref={ta:.4f} d={e:.4f}"
            for c, dx, ts, ta, e in bad
        )
        raise AssertionError(f"正演与参考偏差 > {atol} s: {msg}")
    return recs


def main() -> None:
    here = Path(__file__).resolve().parent
    write_smesh(here / "converse.smesh")
    write_seafloor(here / "seafloor.refl")
    write_conv(here / "conv.refl")
    write_geom(here / "geom_conv.dat")
    write_analytic(here / "analytic.txt")
    print(f"wrote {here}")
    print(
        f"  converse.smesh  topo=0  H={H} Zc={Z_CONV}  "
        f"Vw={V_WATER} Vp_sed={VP_SED0}+{VP_SED_GRAD}(z-H) "
        f"Vs={VS0}+{VS_GRAD}*(z-Zc)"
    )
    print("  seafloor.refl  （作图；raytype 6 不需要 -B）")
    print("  conv.refl      （tt_forward -X）")
    print(f"  geom_conv.dat  OBS x={OBS_X} z={H}  codes 0+6  offsets {list(OFFSETS)}")
    print(
        f"  analytic.txt  zero-offset t0={t_head_wave(0):.4f}  "
        f"t6={t_psp_ref(0):.4f}"
    )


if __name__ == "__main__":
    main()
