"""KKHS02 四步正演链与 H–Vp 曲线术语（GUI 文档单一来源）。

论文 Step 1–4 指 **正演方法链**（校准 → 反演界 → 结晶 → 上涌），
与 H–Vp 图上「读观测 (H, V_LC)」的横纵锚点 **不是同一编号**。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Kkhs02StepSpec:
    """KKHS02 正演四步之一。"""

    step_id: int
    short_title: str
    purpose: str
    method: str
    inputs: str
    outputs: str
    paper_figs: str
    gui_mapping: str
    relationship: str


KKHS02_STEPS: tuple[Kkhs02StepSpec, ...] = (
    Kkhs02StepSpec(
        step_id=1,
        short_title="Bulk Vp 与方程 (1)",
        purpose=(
            "建立 **平均压力 P̄、平均熔融分数 F̄ → 假想 bulk 速度 V_bulk** 的回归标尺。"
            "V_bulk 表示「若全柱瞬时固结」的 norm-based 端元速度（600 MPa、400 °C），"
            "**不是** 地震观测的下地壳速度 V_LC。"
        ),
        method=(
            "实验/计算熔体 → CIPW 名义矿物 → Hashin–Sutherland 混合 → 回归为方程 (1)："
            "V_bulk = a0 + W_L·poly_b(P̄,F̄) + W_H·poly_c(P̄,F̄)，"
            "其中 W_L、W_H 为乘积 tanh 窗，poly 为二次多项式（系数见 data/korenaga2002_eq1.json；"
            "生产用 b1=−0.55，印刷表 +0.55 为勘误保留值）。"
            "Modern 轨在精修模式下用 BurnMan 直接算 norm Vp，快速扫描仍查同一方程 (1)。"
        ),
        inputs="熔体库（Fig.3）；正演时由 Step 4 给出 (P̄, F̄)。",
        outputs="V_bulk(P̄, F̄) @ 600 MPa、400 °C。",
        paper_figs="§2.2，Fig.3–4a。",
        gui_mapping=(
            "「Tp–χ 扫描」「正演」输出的 Vp；H–Vp 左图纵坐标；"
            "「bulk 界精修」用 BurnMan 与 V_LC 比较。"
        ),
        relationship="← Step 4 提供 (P̄, F̄)；→ Step 2 用 V_bulk 与 V_LC 比较。",
    ),
    Kkhs02StepSpec(
        step_id=2,
        short_title="V_LC 上界与 ΔVp 读图带",
        purpose=(
            "用可直接观测的 **V_LC** 排除不可能的 (Tp, χ, H) 组合："
            "模型 V_bulk 不得高于 V_LC；并给出 bulk 速度 **可读下界**。"
        ),
        method=(
            "上界：V_bulk,model ≤ V_LC（厚壳 H > 15 km 时可靠）。"
            "下界：V_bulk ≥ V_LC − ΔVp,max，其中 ΔVp,max 来自 Step 3（默认 ≈ 0.15 km/s）。"
            "在 H–Vp 图上以 **竖线段** 表示允许区间，**不是** 平移模型曲线。"
        ),
        inputs="观测 V_LC；Step 1 的 V_bulk,model；Step 3 的 ΔVp,max。",
        outputs="bulk 可行区间 [V_LC − ΔVp, V_LC]。",
        paper_figs="§2.3，Fig.5；反演读图 Fig.12a / Fig.15(c)。",
        gui_mapping=(
            "「Step-2 竖段」「ΔVp」「厚壳 H>」；扫描结果「可行」列；"
            "「观测读图」Tab 摘要。"
        ),
        relationship="← Step 1（V_bulk）+ Step 3（ΔVp 幅度）；→ 与 Step 4 的 H 判据联合筛选 Tp、χ。",
    ),
    Kkhs02StepSpec(
        step_id=3,
        short_title="分离结晶 ΔVp(F)",
        purpose="量化 **分离结晶** 使 bulk 相对堆晶岩端元变慢的幅度 ΔVp = V_cumulate − V_bulk。",
        method=(
            "Walker & Daly (1990) 结晶路径 + Langmuir 高压校正；"
            "堆晶与 bulk 均用与 Step 1 相同的 CIPW + HS 链算 Vp。"
        ),
        inputs="Step 1 同源熔体库 / 正演路径上的 F。",
        outputs="ΔVp(F)；F ≈ 0.7–0.8 时 ΔVp,max ≈ 0.15 km/s（Fig.5）。",
        paper_figs="Fig.2（单路径示意），Fig.5（全库统计）。",
        gui_mapping="不单独设按钮；通过「ΔVp」默认值与 Step-2 竖段体现。",
        relationship="← Step 1 物性链；→ 为 Step 2 下界提供 ΔVp,max。",
    ),
    Kkhs02StepSpec(
        step_id=4,
        short_title="主动上涌 → H 与 (P̄, F̄)",
        purpose=(
            "由源区 **Tp、柱几何 χ、被动厚度 b、辉石岩 Φ** 正演熔融柱，"
            "得到壳厚 H 与柱内 (P̄, F̄)，闭合 H–Vp 解释。"
        ),
        method=(
            "**默认「H–Vp 图」= KKHS02 §4 方程 (4)–(11) + 方程 (1)**："
            "(4) T₀=1150+120 P₀；(5) Tp=T₀−20 P₀ ⇒ P₀=(Tp−1150)/100；"
            "(6) F(P)=(∂F/∂P)_S (P₀−P)，α=12%/GPa（§4 / Fig.11–12 标准）；"
            "(7) P_f=(H+b)/30；(8) F̄=∫F dP/∫dP（线性 ⇒ ½ F(P_f)）；"
            "(9) P̄=∫P F dP/∫F dP（线性实现取 (P₀+P_f)/2）；"
            "(10) H=30 χ (P₀−P_f) F̄；(11) 线性时 P_f 闭式。"
            "实现 active_upwelling.solve_active_upwelling → predict_v_bulk_km_s。"
            "**Modern REEBOX**：不用 (4)–(11) 线性闭式，改 Katz/REEBOX 几何算 (H,P̄,F̄)，"
            "再经 **同一方程 (1)** 得 Vp。"
            "图面可与 Fig.12 同布局；数值曲线不必与印刷 Fig.12 重合。"
        ),
        inputs="Tp，χ，b，Φ，engine（reebox / kinzler_linear），岩性 preset。",
        outputs="H_model，P̄，F̄ → Step 1 → V_bulk；与 H_obs、Step 2 界比较。",
        paper_figs="§4，Fig.11–12。",
        gui_mapping=(
            "「正演」「Tp–χ 扫描」「Φ 扫描」；H–Vp 显示 **默认**「H–Vp 图」；"
            "|ΔH| 容差闭合 H_obs。"
        ),
        relationship="→ Step 1（(P̄,F̄)）；H 与 V_bulk 再经 Step 2 与观测闭合。",
    ),
)


# --- H–Vp 曲线：三种来源（勿与论文图号混为一谈） ---

TRACK_MODERN_ID = "modern"
TRACK_FIG12_LINEAR_ID = "fig12_linear"
TRACK_FIG12A_DIGITIZED_ID = "fig12a_digitized"  # validation-only; not a GUI track

TRACK_MODERN_LABEL = "Modern REEBOX 正演轨"
TRACK_FIG12_LINEAR_LABEL = "H–Vp 图"
TRACK_FIG12A_DIGITIZED_LABEL = "Fig.12a digitized（validation 对照）"

TRACK_TERMINOLOGY: tuple[tuple[str, str], ...] = (
    (
        TRACK_FIG12_LINEAR_LABEL,
        "默认显示轨。熔融与厚度用 KKHS02 §4 方程 (4)–(11)"
        "（TK83 固相线 + 线性 F，α≈12%/GPa + Langmuir 柱几何）；"
        "Vp 用 Step 1 方程 (1)。方法来源论文 Fig.12；"
        "实现 active_upwelling / reproduce_fig12。",
    ),
    (
        TRACK_MODERN_LABEL,
        "熔融与厚度用 Katz/pyMelt + REEBOX 柱几何（engine=reebox）；"
        "Vp 仍查同一方程 (1)（快速扫描）；「bulk 界精修」才用 CIPW+BurnMan。"
        "曲线常高于线性轨 — 因 F̄ 通常更大，属预期，非方程 (1) 算错。",
    ),
    (
        TRACK_FIG12A_DIGITIZED_LABEL,
        "印刷 Fig.12a 数值化曲线，仅 validation/ 对照用，不进入 GUI 主流程。",
    ),
)

OBS_ANCHOR_NOTE = (
    "观测锚点 H、V_LC 在 H–Vp 图上分别为横轴、纵轴读图坐标；"
    "它们不对应论文 Step 1 或 Step 4 的编号。"
    "H 用于 |ΔH| 与 §4 方程 (7)(10) 闭合；V_LC 用于 Step 2 bulk 界。"
)

# Modern vs Fig.12：读者最易混淆的一点（读者指南 / 术语共用）
HVP_SHARED_INTERFACE_NOTE = (
    "两条轨共用同一接口：Step 4 给出 (H, P̄, F̄) → Step 1 方程 (1) 给出 V_bulk。"
    "「和 Fig.12 一致」指图面约定（轴、χ 族、等温线、方程 (1)），"
    "不是 Modern 的数值曲线等于印刷 Fig.12。"
)

HVP_TRACK_COMPARE_ROWS: tuple[tuple[str, str, str], ...] = (
    ("固相线 / F(P,T)", "TK83 + 线性 F=α(P₀−P)，α≈12%/GPa", "Katz / pyMelt 真实 F(P,T)"),
    ("壳厚 H 与柱几何", "§4 方程 (7)–(11) 闭式", "REEBOX 三角积分 + 主动权重"),
    ("(P̄, F̄)", "线性柱平均", "RMC / 柱内平均"),
    ("V_bulk（纵轴）", "方程 (1)", "同一方程 (1)"),
    ("GUI 入口", "H–Vp 显示 →「H–Vp 图」", "engine=reebox +「Modern」/ 双轨"),
)


def format_kkhs02_steps_text() -> str:
    """读者指南 / 公式窗可用的四步说明正文。"""
    parts: list[str] = [
        "KKHS02 正演四步（Step 1→4）构成闭合链：",
        "Step 4 几何与熔融 → (P̄, F̄, H) → Step 1 得 V_bulk → Step 3 给出 ΔVp → Step 2 与 V_LC 比较。",
        "",
    ]
    for s in KKHS02_STEPS:
        parts.extend(
            [
                f"Step {s.step_id} — {s.short_title}",
                f"  目的：{s.purpose}",
                f"  方法：{s.method}",
                f"  输入：{s.inputs}",
                f"  输出：{s.outputs}",
                f"  论文：{s.paper_figs}",
                f"  GUI：{s.gui_mapping}",
                f"  衔接：{s.relationship}",
                "",
            ]
        )
    return "\n".join(parts).rstrip()


def format_track_terminology_text() -> str:
    """H–Vp 三种曲线来源说明（纯文本）。"""
    parts: list[str] = [
        HVP_SHARED_INTERFACE_NOTE,
        "",
        "H–Vp 图上可能出现三种不同来源的曲线，名称务必区分：",
        "",
    ]
    for title, body in TRACK_TERMINOLOGY:
        parts.append(f"• {title}")
        parts.append(f"  {body}")
        parts.append("")
    parts.append("对照（熔融侧不同，Vp 接口相同）：")
    parts.append("")
    parts.append(f"{'项目':<14}  {'默认 H–Vp 图（线性）':<36}  {'Modern（Katz）'}")
    parts.append("-" * 72)
    for item, linear, modern in HVP_TRACK_COMPARE_ROWS:
        parts.append(f"{item:<14}  {linear:<36}  {modern}")
    parts.extend(["", OBS_ANCHOR_NOTE])
    return "\n".join(parts)


def format_workflow_summary_short() -> str:
    """公式参考窗顶部的短摘要。"""
    return (
        "Step 1：方程 (1) V_bulk(P̄,F̄)  |  "
        "Step 2：V_LC 上界 + ΔVp 带  |  "
        "Step 3：分离结晶 ΔVp(F)  |  "
        "Step 4：上涌 → H、(P̄,F̄) → 再经 Step 1。"
        " 默认「H–Vp 图」= §4 线性轨 + 方程 (1)；"
        "Modern = Katz 熔融 + 同一方程 (1)（图面同 Fig.12，数值不必重合）。"
    )
