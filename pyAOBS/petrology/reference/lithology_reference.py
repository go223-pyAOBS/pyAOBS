"""岩性 / 预设说明（中文），供 GUI 岩性查阅窗。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from petrology.melting.kinzler1997_batch import HZ_DEP1_WT
from petrology.melting.lithology import G2_BULK_WT
from petrology.melting.pymelt_lithology_adapter import (
    LITHOLOGY_META,
    LITHOLOGY_PRESETS,
    _DEFAULT_DELTA_S,
    _SOURCE_WT,
    enriched_lithology_keys,
    list_lithology_presets,
    list_pymelt_lithology_keys,
    lithology_diagnostics,
    peridotite_lithology_keys,
)

_FAMILY_ZH = {
    "peridotite": "橄榄石端元（方辉橄榄岩）",
    "pyroxenite": "辉石岩端元（富集组分）",
    "eclogite": "榴辉岩端元",
    "hydrous": "含水端元",
    "unknown": "未分类",
}


@dataclass(frozen=True)
class LithologyNote:
    title_zh: str
    traits: str
    when_to_use: str
    reference: str = ""


LITHOLOGY_NOTES: dict[str, LithologyNote] = {
    "katz_lherzolite": LithologyNote(
        "Katz (2003) 方辉橄榄岩",
        "干地幔 **固/液相线** F(P,T) 由 pyMelt 提供；"
        "源区 solid 化学默认 HZ-Dep1（高 MgO、低 SiO₂，**不是熔体**）。"
        "熔体主量由 Kinzler batch 或 pMELTS KLB-1 在 (P,F) 上算出。",
        "一般 LIP 地幔柱正演、Greenland 类 K 剖面；preset 中最常用 per 端元。",
        "Katz et al. 2003; Kinzler 1997 HZ-Dep1",
    ),
    "mckenzie_lherzolite": LithologyNote(
        "McKenzie (1989) 方辉橄榄岩",
        "较早的橄榄岩固相线参数化；ΔS≈250；与 Katz 相比熔融温度略不同。",
        "偏保守/对比试验；lip_eclogite 预设的橄榄石端。",
        "McKenzie & Bickle 1988",
    ),
    "matthews_klb1": LithologyNote(
        "Matthews KLB-1 橄榄岩",
        "基于 KLB-1 成分的 pyMelt 曲线；可与 0.1 wt% H₂O 联用模拟含水地幔。",
        "lip_klb1_hydrous 预设；需模拟源区水时选用。",
        "Matthews et al.; pyMelt klb1",
    ),
    "ball_depleted_mantle": LithologyNote(
        "Ball 亏损地幔",
        "Ball et al. 亏损地幔曲线；ΔS=407 J/kg/K（高于 Katz）。",
        "亏损地幔源区或 Ball 系列对比。",
        "Ball et al.; pyMelt depleted_mantle",
    ),
    "ball_mixed_mantle": LithologyNote(
        "Ball 混合地幔",
        "Ball 混合地幔固/液相线；ΔS=407。",
        "介于亏损与原始地幔之间的源区试验。",
        "Ball et al.; pyMelt mixed_mantle",
    ),
    "ball_primitive_mantle": LithologyNote(
        "Ball 原始地幔",
        "Ball 原始地幔；ΔS=407；lip_ball 预设橄榄石端。",
        "需更「饱满」地幔源、与 G2 辉石岩配对时。",
        "Ball et al.; pyMelt primitive_mantle",
    ),
    "pertermann_g2": LithologyNote(
        "Pertermann G2 辉石岩",
        "经典 G2 富铁镁辉石岩 bulk；固相线比 Katz 橄榄岩低 ~150°C；ΔS≈240。",
        "lip_default 的 pyroxenite 端；Φ>0 时的默认富集端元。",
        "Pertermann & Hirschmann 2003",
    ),
    "matthews_kg1": LithologyNote(
        "Matthews KG-1 辉石岩",
        "Kilauea KG-1 富集辉石岩；更高 Ti、更低 Mg/Si；熔体更硅质/碱质倾向。",
        "Greenland / K 剖面类 LIP（greenland_kg1、lip_kg1）；preset 对比常用。",
        "Matthews et al.; pyMelt kg1",
    ),
    "shorttle_kg1": LithologyNote(
        "Shorttle KG-1",
        "Shorttle 参数化的 KG-1 曲线；ΔS≈380。",
        "与 Matthews KG-1 对照的替代富集端元。",
        "Shorttle et al.; pyMelt kg1",
    ),
    "matthews_eclogite": LithologyNote(
        "Matthews 榴辉岩",
        "高 Ca、低 Mg 榴辉岩固相线；熔融行为与辉石岩不同。",
        "lip_eclogite 预设；深部再循环壳物质 / 高硅实验对比。",
        "Matthews et al.; pyMelt eclogite",
    ),
}

PRESET_NOTES: dict[str, LithologyNote] = {
    "lip_default": LithologyNote(
        "LIP 默认对",
        "Katz 橄榄岩 + G2 辉石岩；REEBOX 文档默认双端元；Φ 控制 G2 体积分数。",
        "无特定剖面约束时的基线；与 native Kinzler 化学近似配对。",
    ),
    "lip_kg1": LithologyNote(
        "K 剖面 KG-1 对",
        "Katz + Matthews KG-1；富集端元更偏 Kilauea 型。",
        "夏威夷 / 洋岛型富集源试验。",
    ),
    "lip_klb1_hydrous": LithologyNote(
        "含水 KLB-1 + KG-1",
        "橄榄石端 0.1 wt% H₂O；固相线降低、同等 Tp 下 F 增大。",
        "源区含水是 H/Vp 闭合的重要自由度。",
    ),
    "lip_eclogite": LithologyNote(
        "McKenzie + 榴辉岩",
        "橄榄石端 McKenzie；富集端 Matthews eclogite。",
        "再循环榴辉岩贡献较大的源区假设。",
    ),
    "lip_ball": LithologyNote(
        "Ball 原始地幔 + G2",
        "Ball primitive mantle + G2 辉石岩。",
        "Ball 系列与 G2 配对的预设对比项。",
    ),
    "greenland_kg1": LithologyNote(
        "Greenland K 剖面（推荐）",
        "Katz + KG-1；GUI 启动默认；与 README / Fig.15 类 Greenland 锚点一致。",
        "Greenland 类 LIP 剖面解释的首选起点；可用「预设对比」与其它 preset 比。",
    ),
}

NATIVE_BACKEND_NOTE = LithologyNote(
    "native 后端",
    "不调用 pyMelt 岩性曲线；橄榄石端 Katz/ Kinzler 线性熔体化学，"
    "辉石岩端 G2 固定 bulk；几何仍为 REEBOX。"
    "速度快，但岩性键 per/pyr 被忽略。",
    "快速试算或无机物性库环境；精细岩性对比请用 backend=pymelt。",
)

# KKHS02 Fig.2 标注的 Kinzler (1997) 多压聚合熔体（文献值，非 solid bulk）
FIG2_KINZLER_MELT_WT: dict[str, float] = {
    "SiO2": 48.2,
    "TiO2": 0.94,
    "Al2O3": 16.4,
    "Cr2O3": 0.12,
    "FeO": 7.96,
    "MgO": 12.5,
    "CaO": 11.4,
    "K2O": 0.07,
    "Na2O": 2.27,
}

SOLID_VS_MELT_NOTE = (
    "岩性说明中的「源区 solid bulk」是 **未熔地幔源岩**（如 Kinzler HZ-Dep1），"
    "不是 Fig.2 / 正演日志里的 **熔体主量**。"
    "熔体由 batch_melt(P,F) 或 pMELTS 网格在每条 (P,F) 上计算；"
    "柱内再按 RMC 聚合为 pooled melt（正演 footer 的 SiO₂）。"
)


def format_composition_wt(source: dict[str, float]) -> str:
    keys = ("SiO2", "TiO2", "Al2O3", "FeO", "MgO", "CaO", "Na2O", "K2O", "H2O")
    parts = [f"{k} {source.get(k, 0.0):.2f}" for k in keys if float(source.get(k, 0.0)) > 0.01]
    return "  ".join(parts) if parts else "（无）"


def kinzler_batch_melt_example(*, p_gpa: float = 1.5, f: float = 0.09) -> dict[str, float]:
    """Kinzler (1997) 等压批熔：由 HZ-Dep1 solid → melt @ (P,F)。"""
    from petrology.melting.kinzler1997_batch import batch_melt_oxides

    return batch_melt_oxides(float(p_gpa), float(f))


def format_melt_comparison_block(*, p_gpa: float = 1.5, f: float = 0.09) -> str:
    """对比 Fig.2 文献熔体与本仓库 Kinzler 计算。"""
    calc = kinzler_batch_melt_example(p_gpa=p_gpa, f=f)
    lit = FIG2_KINZLER_MELT_WT
    keys = ("SiO2", "TiO2", "Al2O3", "FeO", "MgO", "CaO", "Na2O")
    lines = [
        f"KKHS02 Fig.2 标注（Kinzler 多压聚合熔体，F̄≈{f*100:.0f}%，P̄≈{p_gpa:g} GPa）：",
        format_composition_wt(lit),
        "",
        f"本仓库 batch_melt_oxides({p_gpa:g}, {f:g}) 复算：",
        format_composition_wt(calc),
        "",
        "二者应接近（SiO₂ ~48%，Al₂O₃ ~16–17%，MgO ~11–12%），"
        "与源区 solid bulk（MgO ~38%）完全不同。",
    ]
    return "\n".join(lines)


def _safe_diagnostics(key: str, *, h2o_wt: float = 0.0) -> dict[str, Any]:
    try:
        d = lithology_diagnostics(key, p_gpa=2.0, h2o_wt=h2o_wt)
        return {**d, "valid": True}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def describe_lithology_key(key: str, *, h2o_wt: float = 0.0) -> dict[str, Any]:
    """单条 pyMelt 岩性键的说明（供 GUI 详情面板）。"""
    meta = LITHOLOGY_META.get(key)
    note = LITHOLOGY_NOTES.get(key)
    family = meta.family if meta else "unknown"
    source = dict(_SOURCE_WT.get(key, HZ_DEP1_WT if family == "peridotite" else G2_BULK_WT))
    chem_note = (
        "pMELTS KLB-1 网格（pmelts_klb1）"
        if family == "peridotite"
        else "Kinzler batch 或固定 G2 熔体端元"
    )
    diag = _safe_diagnostics(key, h2o_wt=h2o_wt)
    return {
        "id": key,
        "kind": "lithology",
        "family": family,
        "family_zh": _FAMILY_ZH.get(family, family),
        "title": note.title_zh if note else (meta.description if meta else key),
        "description_en": meta.description if meta else "",
        "traits": note.traits if note else "",
        "when_to_use": note.when_to_use if note else "",
        "reference": note.reference if note else "",
        "source_solid_wt": source,
        "composition_text": format_composition_wt(source),
        "composition_role": "源区 solid bulk（批熔输入）" if family == "peridotite" else "富集端元 solid bulk",
        "melt_chemistry": chem_note,
        "delta_s_default": _DEFAULT_DELTA_S.get(key),
        "diagnostics": diag,
        "pymelt_module": key,
        "is_peridotite": family == "peridotite",
    }


def describe_preset(name: str) -> dict[str, Any]:
    """命名 preset 的说明。"""
    if name in ("", "(custom)"):
        return {
            "id": "custom",
            "kind": "preset",
            "title": "自定义 (custom)",
            "traits": "手动选择 per / pyr 与 H₂O；不绑定命名 preset。",
            "when_to_use": "试验非预设端元组合时使用。",
            "peridotite_key": "",
            "pyroxenite_key": "",
        }
    p = LITHOLOGY_PRESETS[name]
    note = PRESET_NOTES.get(name)
    return {
        "id": name,
        "kind": "preset",
        "title": note.title_zh if note else name,
        "traits": note.traits if note else p.description,
        "when_to_use": note.when_to_use if note else "",
        "reference": note.reference if note else "",
        "description_en": p.description,
        "peridotite_key": p.peridotite_key,
        "pyroxenite_key": p.pyroxenite_key,
        "peridotite_h2o_wt": p.peridotite_h2o_wt,
        "pyroxenite_h2o_wt": p.pyroxenite_h2o_wt,
        "phi_note": "源区 Φ 为辉石岩体积分数；REEBOX 按 u₀ 混合两端元熔融曲线。",
    }


def format_lithology_detail_text(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """详情区段落 (标题, 正文)。"""
    sections: list[tuple[str, str]] = []
    if payload.get("kind") == "preset":
        sections.append(("预设说明", payload.get("traits") or ""))
        if payload.get("when_to_use"):
            sections.append(("何时选用", payload["when_to_use"]))
        per = payload.get("peridotite_key") or "—"
        pyr = payload.get("pyroxenite_key") or "—"
        h2o_per = float(payload.get("peridotite_h2o_wt") or 0.0)
        h2o_pyr = float(payload.get("pyroxenite_h2o_wt") or 0.0)
        sections.append(
            (
                "端元组成",
                f"橄榄石端 (per)：{per}"
                + (f"，H₂O={h2o_per:g} wt%" if h2o_per > 0 else "")
                + f"\n辉石岩端 (pyr)：{pyr}"
                + (f"，H₂O={h2o_pyr:g} wt%" if h2o_pyr > 0 else "")
                + "\n"
                + (payload.get("phi_note") or ""),
            )
        )
        if per and per != "—":
            sections.append(("橄榄石端详情", _short_lith_summary(per, h2o_wt=h2o_per)))
        if pyr and pyr != "—":
            sections.append(("辉石岩端详情", _short_lith_summary(pyr, h2o_wt=h2o_pyr)))
        return sections

    sections.append(("类型", payload.get("family_zh") or ""))
    if payload.get("traits"):
        sections.append(("特点", payload["traits"]))
    if payload.get("when_to_use"):
        sections.append(("何时选用", payload["when_to_use"]))
    role = payload.get("composition_role") or "源区 bulk"
    sections.append((role, payload.get("composition_text") or ""))
    if payload.get("is_peridotite"):
        sections.append(("与 KKHS02 Fig.2 熔体对比", format_melt_comparison_block()))
    elif payload.get("melt_chemistry"):
        sections.append(("熔体化学", str(payload["melt_chemistry"])))
    sections.append(("solid 与 melt 区别", SOLID_VS_MELT_NOTE))
    diag = payload.get("diagnostics") or {}
    if diag.get("valid"):
        sections.append(
            (
                "参考 P=2 GPa 诊断",
                f"固相线 Tsol = {diag['tsol_c']:.1f} °C\n"
                f"液相线 Tliq = {diag['tliq_c']:.1f} °C\n"
                f"ΔS = {diag.get('delta_s', payload.get('delta_s_default', float('nan'))):.0f} J/kg/K",
            )
        )
    elif diag.get("error"):
        sections.append(("pyMelt 诊断", f"不可用：{diag['error']}"))
    if payload.get("reference"):
        sections.append(("文献", payload["reference"]))
    return sections


def _short_lith_summary(key: str, *, h2o_wt: float) -> str:
    d = describe_lithology_key(key, h2o_wt=h2o_wt)
    role = d.get("composition_role") or "源区 bulk"
    lines = [
        d.get("title") or key,
        d.get("traits") or "",
        f"{role}：{d.get('composition_text')}",
    ]
    if d.get("is_peridotite"):
        calc = kinzler_batch_melt_example()
        lines.append(f"示例 melt @1.5 GPa F=0.09：{format_composition_wt(calc)}")
    diag = d.get("diagnostics") or {}
    if diag.get("valid"):
        lines.append(f"@2 GPa：Tsol={diag['tsol_c']:.0f}°C  Tliq={diag['tliq_c']:.0f}°C")
    return "\n".join(x for x in lines if x)


def catalog_tree_ids() -> tuple[list[str], list[str], list[str]]:
    return (list_lithology_presets(), peridotite_lithology_keys(), enriched_lithology_keys())
