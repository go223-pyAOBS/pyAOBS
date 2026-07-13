# -*- coding: utf-8 -*-
"""
tt_inverse ``-L`` 日志解析与反演结果可视化（与 ``inverse.cc`` 数据行 26 列 + 可选重力列一致）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# 列索引（0 起，与 help_docs 中 1–26 对应）
COL_ITER = 0
COL_ISET = 1
COL_REJECTED = 2
COL_RMS_TOT = 3
COL_CHI_TOT = 4
COL_N_PG = 5
COL_RMS_PG = 6
COL_CHI_PG = 7
COL_N_PMP = 8
COL_RMS_PMP = 9
COL_CHI_PMP = 10
COL_W_SV = 13
COL_W_SD = 14
COL_W_DV = 15
COL_W_DD = 16
COL_PRED_CHI = 20
COL_DV_NORM = 21
COL_DD_NORM = 22
COL_LMVH = 23
COL_LMVV = 24
COL_LMD = 25

_MPL_CJK_CONFIGURED = False


def ensure_matplotlib_cjk_font() -> None:
    """
    为 Matplotlib 选择含中日韩字形的无衬线字体，避免 suptitle/坐标轴中文缺字警告。

    仅从 **fontManager 已索引的字体文件** 中识别 CJK（按路径启发式），**不把**一长串可能未安装的
    字体族名写入 ``font.sans-serif``，以免触发大量 ``findfont: ... not found`` 日志。
    若系统未装任何 CJK 字体，则只保留 DejaVu 等回退字体（中文仍可能显示为方块）。
    """
    global _MPL_CJK_CONFIGURED
    if _MPL_CJK_CONFIGURED:
        return
    _MPL_CJK_CONFIGURED = True
    try:
        import matplotlib
        from matplotlib import font_manager as fm

        def _fontfile_has_cjk(path: str) -> bool:
            if not path:
                return False
            pl = path.lower().replace("\\", "/")
            return any(
                k in pl
                for k in (
                    "notosanscjk",
                    "notoserifcjk",
                    "sourcehansans",
                    "source han",
                    "wqy",
                    "wenquanyi",
                    "msyh",
                    "simhei",
                    "simsun",
                    "droidsansfallback",
                    "arphic",
                    "uming",
                    "ukai",
                    "ipaex",
                    "ipagothic",
                    "ipamincho",
                )
            )

        def _cjk_preference_score(fname: str) -> int:
            """越小越优先作为 sans-serif 首选。"""
            pl = fname.lower().replace("\\", "/")
            order = (
                ("notosanscjk", 0),
                ("notoserifcjk", 1),
                ("sourcehansans", 2),
                ("wqy-microhei", 3),
                ("wqy-zenhei", 3),
                ("wqy", 4),
                ("wenquanyi", 4),
                ("droidsansfallback", 5),
                ("msyh", 6),
                ("simhei", 7),
                ("simsun", 8),
                ("arphic", 9),
                ("uming", 10),
                ("ukai", 11),
                ("ipaex", 12),
                ("ipag", 13),
            )
            best = 99
            for key, rank in order:
                if key in pl:
                    best = min(best, rank)
            return best

        # 只使用磁盘上真实存在的字体；按路径判断 CJK，避免对虚构族名调用 findfont。
        families_ordered: List[str] = []
        seen: set[str] = set()
        scored: List[Tuple[int, int, str]] = []
        for idx, font in enumerate(fm.fontManager.ttflist):
            if not _fontfile_has_cjk(font.fname):
                continue
            name = font.name
            if name in seen:
                continue
            seen.add(name)
            scored.append((_cjk_preference_score(font.fname), idx, name))

        scored.sort(key=lambda t: (t[0], t[1]))
        families_ordered = [t[2] for t in scored]

        fallback = ["DejaVu Sans"]
        if families_ordered:
            matplotlib.rcParams["font.sans-serif"] = families_ordered + [
                x for x in fallback if x not in families_ordered
            ]
        # 未检测到任何 CJK 文件：不写入虚构族名列表，避免 findfont 对每个名字告警
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def parse_tt_inverse_log(path: Path) -> List[List[float]]:
    """
    读取日志中非 ``#`` 开头的数值行；每行至少 26 列（联合重力时可有第 27 列）。
    无效行跳过。
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    rows: List[List[float]] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            continue
        if len(vals) < 26:
            continue
        rows.append(vals)
    return rows


def sort_log_rows(rows: Sequence[Sequence[float]]) -> List[List[float]]:
    return sorted(rows, key=lambda r: (r[COL_ITER], r[COL_ISET]))


def last_row_metrics(rows: Sequence[Sequence[float]]) -> Dict[str, float] | None:
    if not rows:
        return None
    r = sort_log_rows(list(rows))[-1]
    rough = abs(r[COL_LMVH]) + abs(r[COL_LMVV]) + abs(r[COL_LMD])
    return {
        "rms_total": r[COL_RMS_TOT],
        "chi_total": r[COL_CHI_TOT],
        "pred_chi": r[COL_PRED_CHI],
        "w_sv": r[COL_W_SV],
        "w_sd": r[COL_W_SD],
        "w_dv": r[COL_W_DV],
        "w_dd": r[COL_W_DD],
        "roughness_sum": rough,
    }


def composite_score(pred_chi: float, roughness: float, weight: float) -> float:
    """
    启发式综合得分：越小越倾向「拟合好且模型不过度振荡」。
    score = pred_χ² × (1 + weight × R)，R 为速度水平/垂向与深度粗糙度之和。
    """
    return float(pred_chi * (1.0 + weight * max(roughness, 0.0)))


def format_params_compact(m: Dict[str, float]) -> str:
    """末步日志列 14–17：平滑权重 s_v/s_d，阻尼 dv/dd（与 help_docs 一致）。"""
    return (
        f"sv={m['w_sv']:.4g} sd={m['w_sd']:.4g} "
        f"dv={m['w_dv']:.4g} dd={m['w_dd']:.4g}"
    )


def _collect_last_metrics_series(
    series: Dict[str, Sequence[Sequence[float]]],
) -> Tuple[List[str], List[Dict[str, float]]]:
    names: List[str] = []
    metrics: List[Dict[str, float]] = []
    for name, rows in series.items():
        m = last_row_metrics(rows)
        if m is None:
            continue
        names.append(name)
        metrics.append(m)
    return names, metrics


def build_figure_multi_param_influence(
    series: Dict[str, Sequence[Sequence[float]]],
    title: str = "",
):
    """
    末步：各反演在日志中记录的平滑/阻尼权重与 pred χ²、粗糙度 R 的关系（每组反演一个点）。
    横轴为对数刻度，便于跨数量级比较不同参数组。
    """
    import matplotlib.pyplot as plt

    ensure_matplotlib_cjk_font()
    names, metrics = _collect_last_metrics_series(series)
    if not names:
        raise ValueError("没有可用的多日志数据")

    param_axes: List[Tuple[str, str]] = [
        ("w_sv", "weight_s_v（平滑·速度）"),
        ("w_sd", "weight_s_d（平滑·深度）"),
        ("w_dv", "w_dv（阻尼·速度）"),
        ("w_dd", "w_dd（阻尼·深度）"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(10.5, 11.0), constrained_layout=True)
    fig.suptitle(
        title or "反演参数影响（末步）：日志列 14–17 与 pred χ² / 粗糙度 R",
        fontsize=11,
    )

    for i, (key, xlab) in enumerate(param_axes):
        w_raw = np.array([float(m[key]) for m in metrics], dtype=float)
        w_plot = np.where(np.isfinite(w_raw) & (w_raw > 0.0), w_raw, np.nan)
        pred = np.array([m["pred_chi"] for m in metrics], dtype=float)
        rough = np.array([m["roughness_sum"] for m in metrics], dtype=float)

        ax_l, ax_r = axes[i, 0], axes[i, 1]
        ax_l.scatter(w_plot, pred, s=72, c=range(len(names)), cmap="tab10", zorder=3)
        ax_r.scatter(w_plot, rough, s=72, c=range(len(names)), cmap="tab10", zorder=3)
        for j, nm in enumerate(names):
            if not np.isfinite(w_plot[j]):
                continue
            ax_l.annotate(
                nm,
                (w_plot[j], pred[j]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6,
            )
            ax_r.annotate(
                nm,
                (w_plot[j], rough[j]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6,
            )
        ax_l.set_xscale("log")
        ax_l.set_ylabel("pred χ² (LSQR)")
        ax_l.set_xlabel(xlab)
        ax_l.grid(True, alpha=0.3)
        ax_l.set_title("pred χ²")

        ax_r.set_xscale("log")
        ax_r.set_ylabel("R = |Lmvh|+|Lmvv|+|Lmd|")
        ax_r.set_xlabel(xlab)
        ax_r.grid(True, alpha=0.3)
        ax_r.set_title("粗糙度 R")

    return fig


def build_figure_multi_summary_table(
    series: Dict[str, Sequence[Sequence[float]]],
    rough_weight: float = 0.001,
    title: str = "",
):
    """末步数值表：便于对照文件名与平滑/阻尼及指标；含综合得分 ``pred_chi*(1+w*R)``（与 Pareto 页同一 *w*）。"""
    import matplotlib.pyplot as plt

    ensure_matplotlib_cjk_font()
    names, metrics = _collect_last_metrics_series(series)
    if not names:
        raise ValueError("没有可用的多日志数据")

    nrows = len(names)
    # 字号较大时行高略增，避免裁切
    fig_h = min(2.6 + nrows * 0.48, 28.0)
    fig, ax = plt.subplots(figsize=(14.5, fig_h))
    ax.axis("off")
    col_labels = [
        "Run",
        "s_v",
        "s_d",
        "dv",
        "dd",
        "pred chi2",
        "R",
        f"score (w={rough_weight:g})",
        "RMS",
    ]
    cell_text: List[List[str]] = []
    for nm, m in zip(names, metrics):
        sc = composite_score(
            float(m["pred_chi"]),
            float(m["roughness_sum"]),
            float(rough_weight),
        )
        cell_text.append(
            [
                nm[:36] + ("…" if len(nm) > 36 else ""),
                f"{m['w_sv']:.5g}",
                f"{m['w_sd']:.5g}",
                f"{m['w_dv']:.5g}",
                f"{m['w_dd']:.5g}",
                f"{m['pred_chi']:.5g}",
                f"{m['roughness_sum']:.5g}",
                f"{sc:.5g}",
                f"{m['rms_total']:.5g}",
            ]
        )
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="upper center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.08, 2.05)
    fig.suptitle(
        title
        or f"末步汇总：平滑/阻尼（列 14–17）与 pred chi2、R、score（w={rough_weight:g}）、RMS",
        fontsize=13,
        y=0.99,
    )
    plt.subplots_adjust(top=0.88, left=0.04, right=0.98, bottom=0.02)
    return fig


def build_figure_single_log(rows: Sequence[Sequence[float]], title: str = ""):
    """单日志：RMS / 卡方 随迭代（横轴为 iteration，行已按 iter、iset 排序）。"""
    import matplotlib.pyplot as plt

    ensure_matplotlib_cjk_font()
    if not rows:
        raise ValueError("日志无有效数据行（需至少一行 ≥26 列数值）")
    s = sort_log_rows(rows)
    it = np.array([r[COL_ITER] for r in s], dtype=float)
    rms = np.array([r[COL_RMS_TOT] for r in s])
    chi0 = np.array([r[COL_CHI_TOT] for r in s])
    pred = np.array([r[COL_PRED_CHI] for r in s])

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    fig.suptitle(title or "tt_inverse 日志：走时残差与卡方随迭代", fontsize=12)

    axes[0].plot(it, rms, "b-o", markersize=4, lw=1.2)
    axes[0].set_ylabel("RMS traveltime (Pg+PmP)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(it, chi0, "g-s", markersize=4, lw=1.2, label="initial χ² (合并)")
    axes[1].plot(it, pred, "m-^", markersize=4, lw=1.2, label="pred χ² (LSQR)")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("χ²")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    return fig


def build_figure_multi_last_bars(
    series: Dict[str, Sequence[Sequence[float]]],
    title: str = "",
):
    """多日志：各次反演**最后一次**数据行的 RMS、initial χ²、pred χ² 柱状对比。"""
    import matplotlib.pyplot as plt

    ensure_matplotlib_cjk_font()
    labels: List[str] = []
    rms_l: List[float] = []
    chi0_l: List[float] = []
    pred_l: List[float] = []
    param_lines: List[str] = []
    for name, rows in series.items():
        m = last_row_metrics(rows)
        if m is None:
            continue
        labels.append(name)
        rms_l.append(m["rms_total"])
        chi0_l.append(m["chi_total"])
        pred_l.append(m["pred_chi"])
        param_lines.append(format_params_compact(m))

    if not labels:
        raise ValueError("没有可用的多日志数据")

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.35), 6.2), constrained_layout=True)
    fig.suptitle(
        title or "各反演末步：RMS 与 χ² 对比（横轴下为平滑/阻尼：sv sd dv dd）",
        fontsize=11,
    )
    ax.bar(x - w, rms_l, width=w, label="RMS (Pg+PmP)", color="steelblue")
    ax.bar(x, chi0_l, width=w, label="initial χ²", color="seagreen")
    ax.bar(x + w, pred_l, width=w, label="pred χ² (LSQR)", color="darkorchid")
    ax.set_xticks(x)
    tick_labels = [f"{lb}\n{p}" for lb, p in zip(labels, param_lines)]
    ax.set_xticklabels(tick_labels, rotation=22, ha="right", fontsize=7)
    ax.set_ylabel("value")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    return fig


def build_figure_multi_overlay(
    series: Dict[str, Sequence[Sequence[float]]],
    title: str = "",
):
    """多日志：RMS、pred χ² 随 iteration 叠画（每文件独立曲线）。"""
    import matplotlib.pyplot as plt

    ensure_matplotlib_cjk_font()
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
    fig.suptitle(title or "多反演：RMS / pred χ² 随 iteration（叠画）", fontsize=12)

    for name, rows in series.items():
        if not rows:
            continue
        s = sort_log_rows(rows)
        it = np.array([r[COL_ITER] for r in s], dtype=float)
        rms = np.array([r[COL_RMS_TOT] for r in s])
        pred = np.array([r[COL_PRED_CHI] for r in s])
        axes[0].plot(it, rms, "-o", markersize=3, lw=1.0, label=name)
        axes[1].plot(it, pred, "-s", markersize=3, lw=1.0, label=name)

    axes[0].set_ylabel("RMS (Pg+PmP)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=7, ncol=2)

    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("pred χ² (LSQR)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=7, ncol=2)

    return fig


def build_figure_multi_pareto_and_score(
    series: Dict[str, Sequence[Sequence[float]]],
    rough_weight: float = 0.001,
    title: str = "",
):
    """
    策略：pred χ² vs 综合粗糙度 (|Lmvh|+|Lmvv|+|Lmd|) 散点；并给出加权得分
    score = pred_χ² × (1 + w×R)，越小越优（启发式，非唯一准则）。
    """
    import matplotlib.pyplot as plt

    ensure_matplotlib_cjk_font()
    names: List[str] = []
    pred_chi: List[float] = []
    rough: List[float] = []
    scores: List[float] = []

    param_lines: List[str] = []
    for name, rows in series.items():
        m = last_row_metrics(rows)
        if m is None:
            continue
        names.append(name)
        param_lines.append(format_params_compact(m))
        pc = m["pred_chi"]
        rg = m["roughness_sum"]
        pred_chi.append(pc)
        rough.append(rg)
        scores.append(composite_score(pc, rg, rough_weight))

    if not names:
        raise ValueError("没有可用的多日志数据")

    n = len(names)
    norm = plt.Normalize(vmin=0, vmax=max(n - 1, 1))
    cmap = plt.get_cmap("tab10")
    point_colors = [cmap(norm(i)) for i in range(n)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    fig.suptitle(
        title or f"反演质量：pred χ²–粗糙度 与 综合得分（w={rough_weight:g}）",
        fontsize=11,
    )

    ax0 = axes[0]
    ax0.scatter(
        rough,
        pred_chi,
        c=range(n),
        cmap=cmap,
        norm=norm,
        s=80,
        zorder=3,
    )
    for i, nm in enumerate(names):
        ax0.annotate(
            f"{nm}\n{param_lines[i]}",
            (rough[i], pred_chi[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=6,
        )
    ax0.set_xlabel("R = |Lmvh|+|Lmvv|+|Lmd|（末步粗糙度）")
    ax0.set_ylabel("pred χ² (LSQR, 末步)")
    ax0.grid(True, alpha=0.3)
    ax0.set_title("Pareto 式：左下区域通常更优（拟合好且更平滑）")

    ax1 = axes[1]
    order = np.argsort(scores)
    xo = np.arange(len(names))
    bar_colors = [point_colors[i] for i in order]
    ax1.barh(xo, [scores[i] for i in order], color=bar_colors, alpha=0.85)
    ax1.set_yticks(xo)
    ax1.set_yticklabels(
        [f"{names[i]}\n{param_lines[i]}" for i in order],
        fontsize=6,
    )
    ax1.set_xlabel(f"score = pred_χ² × (1 + {rough_weight:g} × R)（越小越好）")
    ax1.set_title("综合得分排序（启发式）")
    ax1.grid(True, axis="x", alpha=0.3)

    return fig
