"""参数区、工具栏与输出面板的悬停说明（标签与控件共用同一 tooltip）。"""

from __future__ import annotations

from PySide6.QtWidgets import QGroupBox, QTabWidget, QWidget

from petrology.reference.kkhs02_workflow import (
    TRACK_FIG12_LINEAR_LABEL,
    TRACK_MODERN_LABEL,
)

# 参数标签 → 说明（标签与其输入框共用）
PARAM_LABEL_TIPS: dict[str, str] = {
    "H": (
        "观测壳厚 H (km)，H–Vp 图横轴读图锚点。\n"
        "用于 Step 4 方程 (10) 与 |ΔH| 判据；**不是** 论文 Step 1（Step 1 = 方程 (1) V_bulk）。"
    ),
    "V_LC": (
        "下地壳 / 壳幔边界观测 Vp V_LC (km/s)。\n"
        "用于 Step-2 bulk 速度界与 H–Vp 图叠加；厚壳时配合 ΔVp 竖段读图。"
    ),
    "b": (
        "被动上升附加厚度 b (km)，Langmuir 主动上升柱几何参数。\n"
        "b=0 为纯主动柱；增大 b 相当于增加初始柱顶深度。\n"
        "kinzler_linear 要求 P0 > b/30（约 Tp ≳ 1150 + 3.33·b）；"
        "否则无熔融柱（例如 b=30 km 时需 Tp ≳ 1250°C）。"
    ),
    "Φ": (
        "辉石岩体积分数 Φ（pyroxenite fraction，0–1）。\n"
        "正演与 Tp–χ 扫描时固定此 Φ；若 Φ 不确定，用工具栏「Φ 扫描」逐档试探。"
    ),
    "|ΔH|": (
        "H 匹配容差 (km)。\n"
        "扫描标记可行时要求 |H_model − H_obs| ≤ 此值；可与 bulk 界判据同时启用。"
    ),
    "Tp": (
        "潜在温度 Tp (°C)，用于「正演」单点计算。\n"
        "Tp–χ 扫描使用下方 Tp scan 范围与 step，与此处独立。"
    ),
    "χ": (
        "柱体形状参数 χ（Langmuir / KKHS02 主动上升）。\n"
        "χ 越大柱体越窄；正演单点使用此值，扫描使用 χ list。"
    ),
    "engine": (
        "熔融化学引擎。\n"
        "• reebox — pMELTS + Brown & Lesher (2016) REEBOX 等温线 + 双岩性 RMC\n"
        "• kinzler_linear — Kinzler (1997) 批次熔融线性端元混合（较快）"
    ),
    "Tp scan": (
        "Tp 网格扫描下限 (°C)。\n"
        "与右侧上限、step 共同定义 (Tp, χ) / Φ / 预设对比的 Tp 轴范围。"
    ),
    "step": (
        "Tp 扫描步长 (°C)。\n"
        "步长越小网格越密、耗时越长；REEBOX 常用 25，kinzler 可用 10。"
    ),
    "Tp scan 上限": (
        "Tp 网格扫描上限 (°C)。\n"
        "与 Tp scan 下限、step 共同定义 (Tp, χ) / Φ / 预设对比的 Tp 轴范围。"
    ),
    "χ list": (
        "χ 扫描列表，逗号分隔（如 1,2,4,8,10,12,16）。\n"
        "χ < 1 的项在扫描中自动跳过。"
    ),
    "H–Vp": (
        "H–Vp 左图曲线来源：\n"
        f"• {TRACK_FIG12_LINEAR_LABEL} — 默认；方程 (1) 计算（方法源自 KKHS02 Fig.12）\n"
        f"• {TRACK_MODERN_LABEL} — Step 4 Katz/REEBOX 正演\n"
        "• 双轨 — 计算轨与 Modern 并排对比\n"
        "印刷数字化 Fig.12a 仅 validation/ 保留，不在 GUI。"
    ),
    "ΔVp": (
        "Step-2 读图最大 ΔVp (km/s)。\n"
        "厚壳观测允许 V_bulk 低至 V_LC − ΔVp；例 V_LC=7.0、ΔVp=0.15 → 下界 6.85。"
    ),
    "厚壳 H>": (
        "厚壳阈值 H (km)。\n"
        "当 H_obs ≤ 此值时 Step-2 竖段上界失效（薄壳 / 孔隙壳），读图带仅保留下界。"
    ),
    "backend": (
        "岩性 / 化学后端。\n"
        "• pymelt — pMELTS 岩性库 + REEBOX 双端元（推荐 reebox 引擎）\n"
        "• native — Kinzler 线性端元，不读 preset / per / pyr"
    ),
    "preset": (
        "岩性预设名（如 greenland_kg1）。\n"
        "选中后自动填充橄榄石 / 辉石岩键名与 H₂O；选 (custom) 可手动编辑。"
    ),
    "per": (
        "橄榄石 / 橄榄岩 pMELTS 岩性键（如 katz_lherzolite）。\n"
        "与 pyroxenite 端元按 Φ 混合参与 REEBOX 等温熔融。"
    ),
    "pyr": (
        "辉石岩 / 富集端元 pMELTS 岩性键（如 pertermann_g2）。\n"
        "高 Φ 时熔体成分与 H、Vp 主要受此端元影响。"
    ),
    "H₂O per": (
        "橄榄石端元 bulk H₂O 含量 (wt%)。\n"
        "影响固相线位置与熔融程度；预设会覆盖此值。"
    ),
    "H₂O pyr": (
        "辉石岩端元 bulk H₂O 含量 (wt%)。\n"
        "通常高于橄榄石端元；与 preset 联动。"
    ),
}

# 无独立标签的控件（复选框等）
WIDGET_TIPS: dict[str, str] = {
    "fast_scan": (
        "快速扫描（跳过 BurnMan 精修，推荐）。\n"
        "关闭后启用 top-K CIPW+BurnMan norm Vp 精修，显著变慢但 Vp 界更可靠。\n"
        "精修时 BurnMan 在主线程运行（避免 Segfault）。"
    ),
    "bulk_bounds": (
        "H + Vp bulk 界精修 (norm Vp / REEBOX)。\n"
        "粗网格 melt_proxy/eq.1 → top-K CIPW+BurnMan norm Vp。\n"
        "要求 |ΔH| 与 Step-2 V_bulk 界同时满足；自动关闭快速扫描。"
    ),
    "read_band": (
        "Step 2 读图竖段 (ΔVp)：厚壳时显示 V_bulk ∈ [V_LC−ΔVp, V_LC]。"
    ),
    "fig12_paper_axes": (
        "H–Vp 图论文轴范围 (H 3–35 km, Vp 6.9–7.36 km/s)。\n"
        "取消则轴范围随数据自适应。"
    ),
    "toggle_params": (
        "收起 / 展开顶部参数区，最大化绘图区域。"
    ),
}

GROUP_TIPS: dict[str, str] = {
    "观测锚点": (
        "地震观测读图坐标：H（横轴）、V_LC（纵轴，Step 2 上界）；"
        "几何 b、Φ 与 |ΔH| 容差。"
    ),
    "模型参数": (
        "正演单点 (Tp, χ)、网格扫描范围、熔融引擎、H–Vp 显示与 "
        "Fig.12 读图选项。"
    ),
    "岩性 (reebox)": (
        "pMELTS 双端元与 REEBOX 异源熔融；Φ 为辉石岩体积分数。\n"
        "点击「岩性说明…」查看各 preset / per / pyr 的组成、特点与适用场景。"
    ),
    "输出": (
        "扫描结果表、Step-2 观测读图摘要与运行日志。"
    ),
}

TAB_TIPS: dict[str, str] = {
    "扫描结果": (
        "最近一次 Tp–χ / Φ 扫描的可行点摘要（Tp, χ, H, Vp, ΔH 等）。"
    ),
    "观测读图": (
        "Step-2 bulk 速度界与导入观测 (JSON / transect) 的读图摘要。"
    ),
    "日志": (
        "正演、扫描、导入与导出操作的文本记录。"
    ),
}

TOOLBAR_TIPS: dict[str, str] = {
    "Tp–χ 扫描": (
        "在 Tp × χ 网格上搜索同时满足 H 与 Vp bulk 界的可行模型。\n"
        "何时用：Φ 与岩性已选定，找可行 Tp、χ 区间。主图 H–Vp + 可行热图。"
    ),
    "正演": (
        "单点 (Tp, χ, Φ) 熔融柱正演；主图五格 H、P₀、Pf、Fmax、F̄，"
        "下方 (c) T–P、(d) F–T 与验证→模式图联动。\n"
        "模式图窗内可「按正演参数生成…」刷新完整九面板。"
    ),
    "Φ 扫描": (
        "源区辉石岩比例 Φ 敏感性：对 Φ=0…0.25 各跑一遍 Tp–χ 扫描，"
        "主图绘制 Φ vs Vp excess。\n"
        "何时用：Φ 不确定、需检验「要多少 pyroxenite 才闭合 H 与 V_LC」。"
        "Φ 已确定时优先用 Tp–χ 扫描。"
    ),
    "预设对比": (
        "对多个 pymelt 岩性预设各跑 Tp–χ 扫描并对比可行区。\n"
        "何时用：比较 greenland_kg1 等预设谁更匹配当前 (H, V_LC)。"
    ),
    "轨对比": (
        "Modern REEBOX 正演轨与 Fig.12 线性 Step-4 **计算轨**对比。\n"
        "何时用：已完成 Tp–χ 扫描，需对比现代轨与论文线性轨。"
    ),
    "导入观测": (
        "从 imodel 导出的 CrustObservation JSON 或 transect 窗 CSV 导入 (H, V_LC)。"
    ),
    "公式": (
        "打开 KKHS02 / REEBOX 公式与默认参数查阅（非模态窗口）。"
    ),
    "保存图": (
        "将当前 matplotlib 主图保存为 PNG / PDF / SVG。"
    ),
}

MENU_TIPS: dict[str, str] = {
    "Katz (2003) Fig.1–6…": (
        "预览 Katz (2003) 熔融参数化 Fig.1–6（Table 2；LIP 橄榄石端元基准）。非模态窗。"
    ),
    "导出 Katz Fig.1–6 PNG…": (
        "批量导出 Katz Fig.1–6 至所选目录（petrology/figures/katz2003_fig*.png）。"
    ),
    "经典熔融理论曲线…": (
        "预览 REEBOX 固相线、绝热线、F(P)、F(T) 等经典曲线（不含 H/Vp）。非模态窗。"
    ),
    "主动/被动熔融模式图…": (
        "九面板模式图：柱几何、Tp–P、F–T、H–Vp、χ 与 H、KKHS02 eq.(1) 与 Step-2。"
        "源文件 active_passive_melting_schematic.png。非模态窗。"
    ),
    "分离结晶 Fig.2 / Fig.5…": (
        "Step 3 结晶图件：Fig.2 单路径 Vp/矿物；Fig.5 ΔVp(F) 包络（→ GUI「ΔVp」≈0.15）。"
        "非模态；Ctrl+Shift+X。"
    ),
    "公式与参数查阅…": (
        "浏览 KKHS02 关键公式与默认参数；右侧详情排版与读者指南一致，"
        "可点击上下游链接跳转。"
    ),
    "读者指南…": (
        "先读：Modern 与 Fig.12 共用方程 (1)，熔融侧可不同；"
        "图面一致 ≠ 数值重合。完整工作流说明（非模态，F1）。"
    ),
}


def apply_tip(widget: QWidget, text: str) -> None:
    if text:
        widget.setToolTip(text)


def apply_labelled_param_tip(label_text: str, label: QWidget, widget: QWidget) -> None:
    text = PARAM_LABEL_TIPS.get(label_text, "")
    apply_tip(label, text)
    apply_tip(widget, text)


def apply_widget_tip(widget: QWidget, key: str) -> None:
    apply_tip(widget, WIDGET_TIPS.get(key, ""))


def apply_group_tooltip(group: QGroupBox) -> None:
    apply_tip(group, GROUP_TIPS.get(group.title(), ""))


def apply_tab_tooltips(tabs: QTabWidget) -> None:
    for i in range(tabs.count()):
        label = tabs.tabText(i)
        widget = tabs.widget(i)
        text = TAB_TIPS.get(label, "")
        tabs.setTabToolTip(i, text)
        if widget is not None:
            apply_tip(widget, text)


def apply_action_tip(action, key: str) -> None:
    """工具栏 / 菜单 QAction：toolTip + statusTip 共用说明。"""
    text = TOOLBAR_TIPS.get(key) or MENU_TIPS.get(key) or WIDGET_TIPS.get(key, "")
    if not text:
        return
    action.setToolTip(text)
    action.setStatusTip(text)
