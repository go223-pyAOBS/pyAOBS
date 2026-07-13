"""Compare Fig.11 d/h digitized curves vs §4 model predictions."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.active_upwelling import solve_active_upwelling
from petrology.vp_regression import predict_v_bulk_fig11_km_s, predict_v_bulk_km_s

DATA = Path(__file__).resolve().parents[1] / "data"
PANEL_D = DATA / "ScreenShot_2026-07-04_124209_264.txt"
PANEL_H = DATA / "ScreenShot_2026-07-04_124221_433.txt"


def kh95_vp(p_gpa: float, f: float) -> float:
    """KKHS02 eq.(2) / KH95 linear @ 600 MPa, 400 C."""
    return 6.712 + 0.16 * float(p_gpa) + 0.661 * float(f)


def parse_getdata(path: Path) -> dict[str, list[tuple[float, float]]]:
    curves: dict[str, list[tuple[float, float]]] = {}
    key: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("Generated"):
            continue
        m = re.match(r"^([xXb=]+[\w=]+)$", line.replace(" ", ""))
        if m:
            key = line.rstrip()
            curves[key] = []
            continue
        if key is None:
            continue
        parts = line.split()
        if len(parts) >= 2:
            tp, v = float(parts[0]), float(parts[1])
            curves[key].append((tp, v))
    return curves


def interp_v(curve: list[tuple[float, float]], tp: float) -> float:
    xs = np.array([p[0] for p in curve], dtype=float)
    ys = np.array([p[1] for p in curve], dtype=float)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    return float(np.interp(tp, xs, ys))


def model_at(tp: float, *, chi: float = 1.0, b_km: float = 0.0) -> dict[str, float] | None:
    try:
        r = solve_active_upwelling(tp_c=tp, b_km=b_km, chi=chi)
    except (ValueError, Exception):
        return None
    return {
        "eq1": r.vp_bulk_km_s,
        "holbrook": predict_v_bulk_fig11_km_s(r.pbar_gpa, r.fbar),
        "kh95": kh95_vp(r.pbar_gpa, r.fbar),
        "pbar": r.pbar_gpa,
        "fbar": r.fbar,
        "h": r.h_km,
    }


def model_scalar(tp: float, key: str, *, chi: float, b_km: float) -> float:
    m = model_at(tp, chi=chi, b_km=b_km)
    return float("nan") if m is None else float(m[key])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def fit_linear(tp: np.ndarray, v: np.ndarray) -> tuple[float, float]:
    """V = a + b*Tp (Tp in C)."""
    x = np.column_stack([np.ones_like(tp), tp])
    coef, _, _, _ = np.linalg.lstsq(x, v, rcond=None)
    return float(coef[0]), float(coef[1])


def main() -> None:
    d_curves = parse_getdata(PANEL_D)
    h_curves = parse_getdata(PANEL_H)

    print("=== Digitized panel (d): chi=1 vs chi=8, b=0 ===")
    for label in d_curves:
        pts = d_curves[label]
        tp = np.array([p[0] for p in pts])
        v = np.array([p[1] for p in pts])
        a, b = fit_linear(tp, v)
        print(f"  {label}: n={len(pts)}  V=[{v.min():.3f},{v.max():.3f}]  "
              f"linear V={a:.2f}+{b*1000:.5f}*(Tp-1250) approx slope {b*1000:.5f} km/s per 1000C")

    print("\n=== Digitized panel (h): b=0,20,30 chi=1 ===")
    for label in h_curves:
        pts = h_curves[label]
        tp = np.array([p[0] for p in pts])
        v = np.array([p[1] for p in pts])
        print(f"  {label}: n={len(pts)}  V=[{v.min():.3f},{v.max():.3f}]")

    check_tps = [1250.0, 1300.0, 1400.0, 1420.0, 1500.0, 1600.0]

    print("\n=== Panel (d) spread & model comparison at key Tp ===")
    print(f"{'Tp':>6}  {'dig chi1':>9} {'dig chi8':>9} {'spread':>7}  "
          f"{'eq1 c1':>8} {'eq1 c8':>8} {'eq1 spr':>7}  "
          f"{'KH95 c1':>8} {'KH95 c8':>8} {'Hol c1':>8} {'Hol c8':>8}")
    for tp in check_tps:
        v1 = interp_v(d_curves["X=1"], tp)
        v8 = interp_v(d_curves["x=8"], tp)
        m1 = model_at(tp, chi=1.0, b_km=0.0)
        m8 = model_at(tp, chi=8.0, b_km=0.0)
        if m1 is None or m8 is None:
            continue
        print(
            f"{tp:6.0f}  {v1:9.3f} {v8:9.3f} {v1-v8:7.3f}  "
            f"{m1['eq1']:8.3f} {m8['eq1']:8.3f} {m8['eq1']-m1['eq1']:7.3f}  "
            f"{m1['kh95']:8.3f} {m8['kh95']:8.3f} {m1['holbrook']:8.3f} {m8['holbrook']:8.3f}"
        )

    print("\n=== Panel (h) spread & model comparison at key Tp ===")
    print(f"{'Tp':>6}  {'dig b0':>8} {'dig b20':>8} {'dig b30':>8} {'spr0-30':>8}  "
          f"{'eq1 b0':>8} {'eq1 b30':>8} {'eq1 spr':>7}  "
          f"{'KH95 b0':>8} {'KH95 b30':>8} {'Hol b0':>8} {'Hol b30':>8}")
    for tp in check_tps:
        vb0 = interp_v(h_curves["b=0"], tp)
        vb20 = interp_v(h_curves["b=20"], tp)
        vb30 = interp_v(h_curves["b=30"], tp)
        m0 = model_at(tp, chi=1.0, b_km=0.0)
        m30 = model_at(tp, chi=1.0, b_km=30.0)
        eq0 = m0["eq1"] if m0 else float("nan")
        eq30 = m30["eq1"] if m30 else float("nan")
        kh0 = m0["kh95"] if m0 else float("nan")
        kh30 = m30["kh95"] if m30 else float("nan")
        ho0 = m0["holbrook"] if m0 else float("nan")
        ho30 = m30["holbrook"] if m30 else float("nan")
        eqspr = eq30 - eq0 if m0 and m30 else float("nan")
        print(
            f"{tp:6.0f}  {vb0:8.3f} {vb20:8.3f} {vb30:8.3f} {vb0-vb30:8.3f}  "
            f"{eq0:8.3f} {eq30:8.3f} {eqspr:7.3f}  "
            f"{kh0:8.3f} {kh30:8.3f} {ho0:8.3f} {ho30:8.3f}"
        )

    print("\n=== Full-curve RMSE vs digitized (panel d) ===")
    for label, chi in [("X=1", 1.0), ("x=8", 8.0)]:
        pts = d_curves[label]
        tp = np.array([p[0] for p in pts])
        v_obs = np.array([p[1] for p in pts])
        for name, fn in [
            ("eq1", lambda t: model_scalar(t, "eq1", chi=chi, b_km=0.0)),
            ("KH95", lambda t: model_scalar(t, "kh95", chi=chi, b_km=0.0)),
            ("Holbrook", lambda t: model_scalar(t, "holbrook", chi=chi, b_km=0.0)),
        ]:
            v_pred = np.array([fn(t) for t in tp])
            print(f"  {label} vs {name:8s}: RMSE={rmse(v_obs, v_pred):.4f} km/s  "
                  f"bias(pred-obs)={np.mean(v_pred-v_obs):+.4f}")

    print("\n=== Full-curve RMSE vs digitized (panel h) ===")
    for label, b in [("b=0", 0.0), ("b=20", 20.0), ("b=30", 30.0)]:
        if label not in h_curves:
            continue
        pts = h_curves[label]
        tp = np.array([p[0] for p in pts])
        v_obs = np.array([p[1] for p in pts])
        for name, key in [("eq1", "eq1"), ("KH95", "kh95"), ("Holbrook", "holbrook")]:
            v_pred = np.array([model_scalar(t, key, chi=1.0, b_km=b) for t in tp])
            print(f"  {label} vs {name:8s}: RMSE={rmse(v_obs, v_pred):.4f} km/s  "
                  f"bias={np.mean(v_pred-v_obs):+.4f}")

    # Junction check ~1400-1420
    tp_j = 1400.0
    print(f"\n=== Junction diagnostic Tp={tp_j:.0f} C ===")
    d_spread = interp_v(d_curves["X=1"], tp_j) - interp_v(d_curves["x=8"], tp_j)
    h_spread = interp_v(h_curves["b=0"], tp_j) - interp_v(h_curves["b=30"], tp_j)
    v_mean_d = 0.5 * (interp_v(d_curves["X=1"], tp_j) + interp_v(d_curves["x=8"], tp_j))
    print(f"  Digitized (d): mean V={v_mean_d:.3f}  chi1-chi8 spread={d_spread:.3f}")
    print(f"  Digitized (h): b0-b30 spread={h_spread:.3f}")
    m = model_at(tp_j, chi=1.0, b_km=0.0)
    m8 = model_at(tp_j, chi=8.0, b_km=0.0)
    m30 = model_at(tp_j, chi=1.0, b_km=30.0)
    if m and m8:
        print(f"  Code eq1: chi spread={m8['eq1']-m['eq1']:.3f}  V(chi1)={m['eq1']:.3f}")
        print(f"  Code KH95: chi spread={m8['kh95']-m['kh95']:.3f}  V(chi1)={m['kh95']:.3f}")
        print(f"  Code Holbrook: chi spread={m8['holbrook']-m['holbrook']:.3f}  V(chi1)={m['holbrook']:.3f}")

    # Optimal constant bias for eq1 to match digitized chi=1 curve
    pts = d_curves["X=1"]
    tp = np.array([p[0] for p in pts])
    v_obs = np.array([p[1] for p in pts])
    v_eq1 = np.array([model_scalar(t, "eq1", chi=1.0, b_km=0.0) for t in tp])
    bias = float(np.mean(v_obs - v_eq1))
    v_adj = v_eq1 + bias
    print(f"\n=== eq.(1) + constant bias to match digitized chi=1 (d panel) ===")
    print(f"  bias (obs - eq1) = {bias:+.4f} km/s")
    print(f"  RMSE after bias: {rmse(v_obs, v_adj):.4f} km/s")
    tp1400 = 1400.0
    print(
        f"  At Tp=1400 after bias: chi1={model_scalar(tp1400,'eq1',chi=1,b_km=0)+bias:.3f} "
        f"chi8={model_scalar(tp1400,'eq1',chi=8,b_km=0)+bias:.3f} "
        f"spread={(model_scalar(tp1400,'eq1',chi=8,b_km=0)-model_scalar(tp1400,'eq1',chi=1,b_km=0)):.3f}"
    )


if __name__ == "__main__":
    main()
