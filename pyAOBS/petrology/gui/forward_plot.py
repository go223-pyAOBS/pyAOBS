"""正演单点结果：五格指标 + 与模式图 (c)(d) 联动的 T–P / F–T 子图。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from matplotlib.figure import Figure

from petrology.validation.plot_active_passive_schematic import (
    build_forward_geometry,
    lithologies_from_col_kwargs,
    plot_schematic_panel_c,
    plot_schematic_panel_d,
)


@dataclass(frozen=True)
class ForwardPlotContext:
    """最近一次正演参数，供「模式图」窗体联动。"""

    tp_c: float
    chi: float
    phi: float
    b_km: float
    melting_engine: str
    lith_kwargs: dict


_LAST_FORWARD_CTX: ForwardPlotContext | None = None


def get_last_forward_context() -> ForwardPlotContext | None:
    return _LAST_FORWARD_CTX


def _metric_cards(col: Any) -> list[tuple[str, str, str]]:
    delta_p = col.p0_gpa - col.pf_gpa
    return [
        (
            "H",
            f"{col.h_km:.1f} km",
            "火成壳厚度：熔融柱几何积分（方程 (10)，Step 4）\n"
            "与观测 H_obs 比较 |ΔH|",
        ),
        (
            "P₀",
            f"{col.p0_gpa:.2f} GPa",
            "起始熔融压力：Tp 绝热线 ∩ 固相线\n"
            "（柱内最深熔融起点）",
        ),
        (
            "Pf",
            f"{col.pf_gpa:.2f} GPa",
            "最浅熔融压力：等熵路径浅端\n"
            f"（ΔP = P₀ − Pf = {delta_p:.2f} GPa）",
        ),
        (
            "Fmax",
            f"{col.f_max:.3f}",
            "最大熔融分数：路径浅端 @ Pf 的 F\n"
            "（模式图 (d) 方点 / 星标）",
        ),
        (
            "F̄",
            f"{col.fbar:.3f}",
            "柱内平均熔融分数：路径加权平均\n"
            f"（P̄ = {col.pbar_gpa:.2f} GPa；用于 H 与 eq.(1) V_bulk）",
        ),
    ]


def plot_forward_result(
    fig: Figure,
    col: Any,
    *,
    tp_c: float,
    chi: float,
    phi: float,
    b_km: float = 0.0,
    melting_engine: str = "reebox",
    lith_kwargs: dict | None = None,
    title_fontsize: float = 13,
    card_fontsize: float = 11,
    note_fontsize: float = 9,
) -> None:
    """五格摘要 + 底部 (c) T–P、(d) F–T（与验证→模式图同源）。"""
    global _LAST_FORWARD_CTX

    lith_kw = dict(lith_kwargs or {})
    _LAST_FORWARD_CTX = ForwardPlotContext(
        tp_c=float(tp_c),
        chi=float(chi),
        phi=float(phi),
        b_km=float(b_km),
        melting_engine=str(melting_engine),
        lith_kwargs=lith_kw,
    )

    fig.clear()
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.15], hspace=0.48, wspace=0.28)
    ax_cards = fig.add_subplot(gs[0, :])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    ax_cards.set_axis_off()
    ax_cards.text(
        0.5,
        0.98,
        f"正演结果 — Tp={tp_c:g}°C   χ={chi:g}   Φ={phi:g}   b={b_km:g} km   engine={melting_engine}",
        transform=ax_cards.transAxes,
        ha="center",
        va="top",
        fontsize=title_fontsize,
        fontweight="bold",
    )

    cards = _metric_cards(col)
    positions = [
        (0.02, 0.42),
        (0.21, 0.42),
        (0.40, 0.42),
        (0.59, 0.42),
        (0.78, 0.42),
    ]
    box_kw = dict(boxstyle="round,pad=0.45", facecolor="#f4f7fb", edgecolor="#8aa4c8", linewidth=1.0)

    for (sym, value, meaning), (x0, y0) in zip(cards, positions):
        ax_cards.text(
            x0,
            y0 + 0.48,
            sym,
            transform=ax_cards.transAxes,
            fontsize=card_fontsize + 1,
            fontweight="bold",
            va="bottom",
            color="#1a3a5c",
        )
        ax_cards.text(
            x0 + 0.02,
            y0 + 0.48,
            value,
            transform=ax_cards.transAxes,
            fontsize=card_fontsize,
            fontweight="600",
            va="bottom",
        )
        ax_cards.text(
            x0,
            y0,
            meaning,
            transform=ax_cards.transAxes,
            fontsize=note_fontsize,
            va="top",
            color="#333333",
            linespacing=1.3,
            bbox=box_kw,
            wrap=True,
        )

    melt = getattr(col, "pooled_melt_wt", {}) or {}
    sio2 = melt.get("SiO2", float("nan"))
    vp = getattr(col, "vp_bulk_eq1_km_s", float("nan"))
    ax_cards.text(
        0.5,
        0.02,
        f"V_bulk(eq.1) = {vp:.3f} km/s  |  熔体 SiO₂ = {sio2:.1f} wt%  |  "
        "下方 (c)(d) 与「验证→模式图」同源",
        transform=ax_cards.transAxes,
        ha="center",
        va="bottom",
        fontsize=note_fontsize,
        color="#555555",
        style="italic",
    )

    rg = build_forward_geometry(
        tp_c=tp_c,
        chi=chi,
        b_km=b_km,
        phi=phi,
        col_kw=lith_kw,
        melting_engine=melting_engine,
        p0_gpa=col.p0_gpa,
        pf_gpa=col.pf_gpa,
        h_km=col.h_km,
    )
    liths = lithologies_from_col_kwargs(lith_kw, phi=phi)

    plot_schematic_panel_c(
        ax_c,
        ref_tp=float(tp_c),
        ref_chi=float(chi),
        col_kw=lith_kw,
        rg=rg,
        phi=float(phi),
        forward_mode=True,
        highlight_pbar_gpa=float(col.pbar_gpa),
    )
    ax_c.set_title(r"(c) $T$–$P$：$P_0$、$P_f$、$\bar P$", fontsize=note_fontsize + 1, pad=6)

    plot_schematic_panel_d(
        ax_d,
        ref_tp=float(tp_c),
        ref_chi=float(chi),
        liths=liths,
        forward_mode=True,
        rg=rg,
        highlight_fbar=float(col.fbar),
        highlight_fmax=float(col.f_max),
    )
    ax_d.set_title(r"(d) $F$–$T$：$F_{\max}$ 与 $\bar F$", fontsize=note_fontsize + 1, pad=6)

    fig.suptitle(f"壳厚 H = {col.h_km:.1f} km", fontsize=title_fontsize - 1, y=0.995)
    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.08, right=0.97)

    # 保留主 ax 引用供 savefig（取 (c) 轴）
    fig._lip_primary_ax = ax_c  # type: ignore[attr-defined]
