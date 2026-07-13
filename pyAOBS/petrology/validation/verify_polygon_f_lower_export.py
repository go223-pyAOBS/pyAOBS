"""
核对多边形导出路径：f_lower 是否与顶边几何一致。

复现 GUI → Petrology Export（多边形模式）的核心逻辑，无需启动 Qt。
与 imodel_qt 中 ``_compute_observation`` 多边形分支等价。

用法::

    python petrology/validation/verify_polygon_f_lower_export.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.imodel_bridge.crust_geometry import (
    f_lower_from_lc_top,
    f_lower_from_polygon_at_x,
    f_lower_from_thicknesses,
    polygon_effective_x_range,
    polygon_top_depth_at_x,
)
from petrology.imodel_bridge.imodel_adapter import crust_from_polygon_selection

# 与 test_imodel_bridge_crust 一致
Z_B = 5.0
Z_M = 35.0
H = Z_M - Z_B

# 梯形多边形：x=10 顶边 z=12；全局最浅顶点在 (5,10) 但不在 x=10
POLY = [
    (0.0, 20.0),
    (5.0, 10.0),
    (10.0, 12.0),
    (20.0, 20.0),
    (10.0, 25.0),
]

# 模拟 Profile Extraction X 范围（窄于多边形包络）
PROF_X0, PROF_X1 = 8.0, 18.0


def _flat_b(_x: float) -> float:
    return Z_B


def _flat_m(_x: float) -> float:
    return Z_M


def _vp_grid(_x: float, _z: float) -> float:
    return 7.0


def expected_f_lower_at_x(x_km: float) -> tuple[float, float, float]:
    """手工：H=|M-B|, h_lc=|M-顶边(x)|, f=h_lc/H。"""
    z_top = polygon_top_depth_at_x(POLY, x_km)
    if z_top is None:
        raise ValueError(f"x={x_km} 无顶边")
    h_lc = abs(Z_M - z_top)
    f = f_lower_from_thicknesses(h_whole_km=H, h_lower_km=h_lc)
    return float(z_top), h_lc, f


def simulate_gui_polygon_export(*, x_km: float) -> dict[str, float]:
    """等价于 petrology_export_qt._compute_observation 多边形分支。"""
    ctx = {
        "has_basement_interface": True,
        "has_moho_interface": True,
        "x_min_km": PROF_X0,
        "x_max_km": PROF_X1,
    }
    moho_fn = _flat_m
    top_fn = _flat_b

    # _require_crust_interfaces
    if not ctx["has_basement_interface"] or not ctx["has_moho_interface"]:
        raise RuntimeError("B/M 未设置")
    z_top = float(top_fn(x_km))
    z_bot = float(moho_fn(x_km))
    h = abs(z_bot - z_top)

    eff_lo, eff_hi = polygon_effective_x_range(
        POLY,
        x_min_km=float(ctx["x_min_km"]),
        x_max_km=float(ctx["x_max_km"]),
    )
    x_use = float(np.clip(x_km, eff_lo, eff_hi))
    z_top, z_bot, h = float(top_fn(x_use)), float(moho_fn(x_use)), abs(
        float(moho_fn(x_use)) - float(top_fn(x_use))
    )

    obs = crust_from_polygon_selection(
        POLY,
        grid_vp_fn=_vp_grid,
        h_whole_km=h,
        z_basement_km=z_top,
        z_moho_km=z_bot,
        x_km=x_use,
        x_min_km=float(ctx["x_min_km"]),
        x_max_km=float(ctx["x_max_km"]),
        pt_correct=False,
    )
    return {
        "x_requested": x_km,
        "x_used": x_use,
        "H": h,
        "f_lower_export": obs.f_lower,
        "v_lc": obs.v_lc_km_s,
        "n_samples": float(obs.n_samples or 0),
    }


def main() -> int:
    eff_lo, eff_hi = polygon_effective_x_range(POLY, x_min_km=PROF_X0, x_max_km=PROF_X1)
    print("=" * 60)
    print("多边形 f_lower 核对（模拟 GUI 导出路径）")
    print("=" * 60)
    print(f"B (沉积基底)     z = {Z_B:g} km")
    print(f"M (Moho)         z = {Z_M:g} km")
    print(f"全地壳 H         = |M-B| = {H:g} km")
    print(f"剖面 X 范围      [{PROF_X0:g}, {PROF_X1:g}] km")
    print(f"有效 X 范围      [{eff_lo:g}, {eff_hi:g}] km  (多边形 ∩ 剖面)")
    print()

    test_xs = [8.0, 10.0, 12.0, 15.0, 18.0, 25.0]
    print(f"{'x_req':>6} {'x_use':>6} {'顶边z':>7} {'h_lc':>7} {'f_期望':>8} {'f_导出':>8} {'Δf':>10} {'OK':>4}")
    print("-" * 68)

    all_ok = True
    for x in test_xs:
        try:
            exp_lo, exp_hi = polygon_effective_x_range(
                POLY, x_min_km=PROF_X0, x_max_km=PROF_X1
            )
            x_clip = float(np.clip(x, exp_lo, exp_hi))
            z_edge, h_lc, f_exp = expected_f_lower_at_x(x_clip)
            row = simulate_gui_polygon_export(x_km=x)
            f_got = row["f_lower_export"]
            delta = f_got - f_exp
            ok = abs(delta) < 1e-9
            all_ok = all_ok and ok
            mark = "OK" if ok else "FAIL"
            print(
                f"{x:6.1f} {row['x_used']:6.1f} {z_edge:7.2f} {h_lc:7.2f} "
                f"{f_exp:8.5f} {f_got:8.5f} {delta:+10.2e} {mark:>4}"
            )
        except ValueError as exc:
            print(f"{x:6.1f}   —     —       —        —        —          —   skip ({exc})")
            all_ok = False

    print()
    # 对照：若误用全局最浅顶点 (5,10) 在 x=10 会得到错误 f_lower
    wrong_top = 10.0  # min vertex z, not top edge at x=10
    f_wrong = f_lower_from_thicknesses(h_whole_km=H, h_lower_km=abs(Z_M - wrong_top))
    _, _, f_correct = expected_f_lower_at_x(10.0)
    print("x=10  sanity check:")
    print(f"  顶边几何 f_lower     = {f_correct:.5f}  (z_top=12)")
    print(f"  若误用全局最浅顶点   = {f_wrong:.5f}  (z=10 @ x=5，错误)")
    print(f"  导出 f_lower         = {simulate_gui_polygon_export(x_km=10.0)['f_lower_export']:.5f}")
    print()

    if all_ok:
        print("结论: 导出 f_lower 与 |顶边(x)-M| / |M-B| 一致。")
        return 0
    print("结论: 存在不一致，请检查。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
