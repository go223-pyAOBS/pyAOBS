"""Compare Reproduction (Fig.12 linear) vs Modern REEBOX on a shared Tp grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


class _HvpPoint(Protocol):
    tp_c: float
    chi: float
    h_km: float
    vp_bulk_km_s: float


@dataclass(frozen=True)
class TrackCompareRow:
    tp_c: float
    chi: float
    h_modern_km: float
    vp_modern_km_s: float
    h_linear_km: float
    vp_linear_km_s: float
    delta_h_km: float
    delta_vp_km_s: float


def compare_tracks_at_grid(
    modern_points: Sequence[_HvpPoint],
    *,
    vp_bias_km_s: float = 0.0,
    dfdp_per_gpa: float = 0.12,
) -> list[TrackCompareRow]:
    """Pair Modern scan points with Fig.12 linear Step-4 at same (Tp, χ)."""
    if not modern_points:
        return []
    tp_u = sorted({p.tp_c for p in modern_points})
    chi_u = sorted({p.chi for p in modern_points})
    from petrology.active_upwelling import sweep_hvp

    linear_res = sweep_hvp(
        tp_values_c=tp_u,
        chi_values=chi_u,
        b_km=0.0,
        dfdp_per_gpa=dfdp_per_gpa,
        vp_bias_km_s=vp_bias_km_s,
    )
    lin_map = {(r.tp_c, r.chi): r for r in linear_res}
    rows: list[TrackCompareRow] = []
    for m in modern_points:
        lin = lin_map.get((m.tp_c, m.chi))
        if lin is None:
            continue
        rows.append(
            TrackCompareRow(
                tp_c=float(m.tp_c),
                chi=float(m.chi),
                h_modern_km=float(m.h_km),
                vp_modern_km_s=float(m.vp_bulk_km_s),
                h_linear_km=float(lin.h_km),
                vp_linear_km_s=float(lin.vp_bulk_km_s),
                delta_h_km=float(m.h_km - lin.h_km),
                delta_vp_km_s=float(m.vp_bulk_km_s - lin.vp_bulk_km_s),
            )
        )
    return rows


def format_track_compare_table(rows: list[TrackCompareRow], *, max_rows: int = 20) -> str:
    if not rows:
        return "(无配对点 — 请先运行 Tp–χ 扫描)"
    hdr = f"{'Tp':>5} {'χ':>4} {'H_mod':>7} {'Vp_mod':>7} {'H_lin':>7} {'Vp_lin':>7} {'ΔH':>6} {'ΔVp':>7}"
    lines = [hdr, "-" * len(hdr)]
    for r in rows[:max_rows]:
        lines.append(
            f"{r.tp_c:5.0f} {r.chi:4.1f} {r.h_modern_km:7.2f} {r.vp_modern_km_s:7.3f} "
            f"{r.h_linear_km:7.2f} {r.vp_linear_km_s:7.3f} {r.delta_h_km:6.2f} {r.delta_vp_km_s:7.3f}"
        )
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows)")
    dvp = [r.delta_vp_km_s for r in rows]
    dh = [r.delta_h_km for r in rows]
    lines.append("")
    lines.append(f"统计: ΔVp mean={sum(dvp)/len(dvp):+.3f} km/s  ΔH mean={sum(dh)/len(dh):+.2f} km")
    return "\n".join(lines)
