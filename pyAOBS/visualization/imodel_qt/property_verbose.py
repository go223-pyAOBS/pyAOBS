"""与 Tk InteractionMixin / PropertiesUIMixin 一致的物性日志与分区显示文案。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_zone_label_map() -> Dict[str, str]:
    labels = {
        "water": "海水层 (water)",
        "sediment": "沉积层 (sediment)",
        "deep": "深部地层 (deep)",
    }
    default_path = Path(__file__).resolve().parents[2] / "utils" / "rockdata" / "imodel_priors.json"
    config_path = Path(os.environ.get("PYAOBS_IMODEL_PRIORS_FILE", str(default_path)))
    if not config_path.exists():
        return labels
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        raw_labels = config.get("zone_labels")
        if isinstance(raw_labels, dict):
            for k, v in raw_labels.items():
                key = str(k or "").strip().lower()
                text = str(v or "").strip()
                if key and text:
                    labels[key] = text
    except Exception:
        pass
    return labels


def format_zone_display(zone: str, zone_map: Dict[str, str]) -> str:
    key = str(zone or "").strip().lower()
    return zone_map.get(key, zone if zone else "-")


ROCK_CANDIDATE_GEOLOGY_CN_ORDER: tuple[str, ...] = (
    "岩石属性",
    "岩石分类",
    "变质程度",
    "地质意义",
    "长英质/镁铁质",
    "岩相",
    "SiO₂ (wt%)",
)


def iter_rock_candidate_geology_lines(candidate: dict[str, Any]) -> Iterable[str]:
    """候选岩性条目下输出的中文岩石属性（与 ``get_auxiliary_attributes`` / 样本表一致）。"""
    gx = candidate.get("geology_cn")
    if not isinstance(gx, dict) or not gx:
        return
    for label in ROCK_CANDIDATE_GEOLOGY_CN_ORDER:
        val = gx.get(label)
        if val is None:
            continue
        s = str(val).strip()
        if not s or s.lower() == "nan":
            continue
        yield f"{label}: {s}"


def format_confidence_text(props: dict[str, Any]) -> str:
    try:
        top1 = float(props.get("confidence_top1", 0.0) or 0.0)
    except (TypeError, ValueError):
        top1 = 0.0
    try:
        top2 = float(props.get("confidence_top2", 0.0) or 0.0)
    except (TypeError, ValueError):
        top2 = 0.0
    try:
        gap = float(
            props.get("confidence_gap", max(0.0, top1 - top2)) or 0.0
        )
    except (TypeError, ValueError):
        gap = max(0.0, top1 - top2)
    is_low = bool(props.get("low_confidence", False))
    level = "低置信度" if is_low else "正常"
    return f"{level} (top1={top1:.2%}, top2={top2:.2%}, gap={gap:.2%})"


def iter_point_property_lines(
    x: float,
    z: float,
    density_method: str,
    props: dict[str, Any],
    *,
    zone_label_map: Dict[str, str],
    point_label: str | None = None,
) -> Iterable[str]:
    """点对点选物性后与 Tk InteractionMixin 基本一致的多行日志。"""
    vp = float(props.get("vp", 0.0) or 0.0)
    vs = float(props.get("vs", 0.0) or 0.0)
    density = float(props.get("density", 0.0) or 0.0)
    pressure = float(props.get("pressure", 0.0) or 0.0)
    temperature = float(props.get("temperature", 0.0) or 0.0)
    zone = str(props.get("zone", "deep") or "deep").strip()
    vp_vs_ratio = vp / vs if vs > 0 else None
    rock_type = str(props.get("rock_type", "UNKNOWN"))
    rock_candidates = props.get("rock_candidates", [])
    if not isinstance(rock_candidates, list):
        rock_candidates = []

    yield "\n" + "=" * 50
    if point_label:
        yield point_label
    else:
        yield f"Point: x={x:.2f} km, z={z:.2f} km"
    yield f"  P-wave velocity: {vp:.2f} km/s"
    yield f"  S-wave velocity: {vs:.2f} km/s"
    if vp_vs_ratio is None or zone == "water":
        yield "  Vp/Vs ratio: N/A (fluid)"
    else:
        yield f"  Vp/Vs ratio: {vp_vs_ratio:.2f}"
        if not (1.0 <= float(vp_vs_ratio) <= 3.5):
            yield "  Warning: Vp/Vs is outside expected range [1.0, 3.5]"
    yield f"  Density ({density_method}): {density:.2f} g/cm³"
    yield f"  Pressure: {pressure:.2f} MPa"
    yield f"  Temperature: {temperature:.2f} °C"
    yield f"  Zone: {format_zone_display(zone, zone_label_map)}"

    if rock_candidates:
        yield "  Rock type candidates:"
        for i, candidate in enumerate(rock_candidates, 1):
            cand_type = candidate.get("rock_type", "UNKNOWN")
            cand_prob = candidate.get("probability", 0.0)
            marker = " <-- Most likely" if i == 1 else ""
            yield f"    {i}. {cand_type}: {float(cand_prob):.2%}{marker}"
            for sub in iter_rock_candidate_geology_lines(candidate):
                yield f"       {sub}"
    else:
        yield (
            f"  Rock type: {rock_type} (probability: "
            f"{float(props.get('rock_probability', 0.0) or 0.0):.2%})"
        )

    foma = str(props.get("felsic_or_mafic", "") or "").strip()
    facies = str(props.get("rock_facies", "") or "").strip()
    sio2_raw = props.get("sio2_wt", None)
    if foma:
        yield f"  Felsic/Mafic: {foma}"
    if facies:
        yield f"  Rock facies: {facies}"
    try:
        sio2_val = (
            float(sio2_raw) if sio2_raw is not None and str(sio2_raw).strip() != "" else None
        )
    except (TypeError, ValueError):
        sio2_val = None
    if sio2_val is not None:
        yield f"  SiO2: {sio2_val:.2f} wt.%"

    rc = str(props.get("rock_classification", "") or "").strip()
    mg = str(props.get("metamorphic_grade", "") or "").strip()
    gm = str(props.get("geological_meaning", "") or "").strip()
    if rc:
        yield f"  Rock classification: {rc}"
    if mg:
        yield f"  Metamorphic grade: {mg}"
    if gm:
        yield f"  Geological meaning: {gm}"
    if bool(props.get("zone_prior_applied", False)):
        yield "  Note: zone-first sample prior applied"
    if bool(props.get("rerank_applied", False)):
        yield "  Note: low-confidence rerank applied"
    if bool(props.get("zone_constraint_applied", False)):
        yield "  Note: zone semantic constraint applied"
    yield f"  Confidence: {format_confidence_text(props)}"

    yield f"  Bulk modulus: {float(props.get('bulk_modulus', 0.0) or 0.0):.2f} GPa"
    yield f"  Shear modulus: {float(props.get('shear_modulus', 0.0) or 0.0):.2f} GPa"
    yield "=" * 50


def iter_manual_calculate_lines(
    x: float,
    z: float,
    props: dict[str, Any],
    *,
    zone_label_map: Dict[str, str],
) -> Iterable[str]:
    """对齐 Tk `PropertiesUIMixin.calculate_properties` 的日志主分支。"""
    p = dict(props or {})
    yield ""
    yield f"Property Results (x={x:.2f}, z={z:.2f}):"
    yield f"  P-wave velocity: {float(p.get('vp', 0.0) or 0.0):.2f} km/s"
    yield f"  S-wave velocity: {float(p.get('vs', 0.0) or 0.0):.2f} km/s"
    vs_source = str(p.get("vs_source", "") or "").strip()
    if vs_source:
        yield f"  Vs source: {vs_source}"
    zone = str(p.get("zone", "deep") or "deep").strip()
    try:
        vp_vs_ratio = (
            float(p.get("vp", 0.0) or 0.0) / float(p.get("vs", 0.0) or 0.0)
            if float(p.get("vs", 0.0) or 0.0) > 0
            else None
        )
    except Exception:
        vp_vs_ratio = None
    if vp_vs_ratio is None or zone == "water":
        yield "  Vp/Vs ratio: N/A (fluid)"
    else:
        yield f"  Vp/Vs ratio: {vp_vs_ratio:.2f}"
        if not (1.0 <= float(vp_vs_ratio) <= 3.5):
            yield "  Warning: Vp/Vs is outside expected range [1.0, 3.5]"
    yield f"  Density: {float(p.get('density', 0.0) or 0.0):.2f} g/cm³"
    yield f"  Pressure: {float(p.get('pressure', 0.0) or 0.0):.1f} MPa"
    yield f"  Temperature: {float(p.get('temperature', 0.0) or 0.0):.1f} °C"
    yield f"  Zone: {format_zone_display(zone, zone_label_map)}"

    rock_candidates = p.get("rock_candidates", [])
    if isinstance(rock_candidates, list) and rock_candidates:
        yield "  Rock type candidates:"
        for i, candidate in enumerate(rock_candidates, 1):
            cand_type = candidate.get("rock_type", "UNKNOWN")
            cand_prob = candidate.get("probability", 0.0)
            marker = " <-- Most likely" if i == 1 else ""
            yield f"    {i}. {cand_type}: {float(cand_prob):.2%}{marker}"
            for sub in iter_rock_candidate_geology_lines(candidate):
                yield f"       {sub}"
    else:
        yield (
            f"  Rock type: {p.get('rock_type', 'UNKNOWN')} "
            f"(probability: {float(p.get('rock_probability', 0.0) or 0.0):.2%})"
        )

    foma = str(p.get("felsic_or_mafic", "") or "").strip()
    facies = str(p.get("rock_facies", "") or "").strip()
    sio2_raw = p.get("sio2_wt", None)
    if foma:
        yield f"  Felsic/Mafic: {foma}"
    if facies:
        yield f"  Rock facies: {facies}"
    try:
        sio2_val = (
            float(sio2_raw) if sio2_raw is not None and str(sio2_raw).strip() != "" else None
        )
    except (TypeError, ValueError):
        sio2_val = None
    if sio2_val is not None:
        yield f"  SiO2: {sio2_val:.2f} wt.%"

    rock_classification = str(p.get("rock_classification", "") or "").strip()
    metamorphic_grade = str(p.get("metamorphic_grade", "") or "").strip()
    geological_meaning = str(p.get("geological_meaning", "") or "").strip()
    if rock_classification:
        yield f"  Rock classification: {rock_classification}"
    if metamorphic_grade:
        yield f"  Metamorphic grade: {metamorphic_grade}"
    if geological_meaning:
        yield f"  Geological meaning: {geological_meaning}"
    if bool(p.get("zone_prior_applied", False)):
        yield "  Note: zone-first sample prior applied"
    if bool(p.get("rerank_applied", False)):
        yield "  Note: low-confidence rerank applied"
    if bool(p.get("zone_constraint_applied", False)):
        yield "  Note: zone semantic constraint applied"
    yield f"  Confidence: {format_confidence_text(p)}"

    yield f"  Bulk modulus: {float(p.get('bulk_modulus', 0.0) or 0.0):.2f} GPa"
    yield f"  Shear modulus: {float(p.get('shear_modulus', 0.0) or 0.0):.2f} GPa"


def geology_summary_lines(props: dict[str, Any]) -> List[str]:
    """Current Geological Info 区域（Tk fixed_info_frame）对齐。"""
    zone = format_zone_display(
        str(props.get("zone", "deep") or "deep"),
        load_zone_label_map(),
    )
    rt = str(props.get("rock_classification", "") or props.get("rock_type", "") or "").strip()
    mg = str(props.get("metamorphic_grade", "") or "").strip()
    gm = str(props.get("geological_meaning", "") or "").strip()
    lines = []
    lines.append(zone or "-")
    lines.append(rt or "-")
    lines.append(mg or "-")
    lines.append(gm or "-")
    return lines
