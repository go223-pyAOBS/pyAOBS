"""将 calculate_all_properties 结果格式化为可读文本。

另提供与 imodel Tk 版一致的导出行字典（CSV/JSON）。
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict


def json_compatible(value: Any) -> Any:
    """递归转为 JSON 友好类型（对齐 Tk `PropertiesUIMixin._to_json_compatible`）。"""
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    if isinstance(value, dict):
        return {str(k): json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(v) for v in value]
    return value


def property_result_row_dict(
    x: float,
    z: float,
    props: Dict[str, Any],
    *,
    source: str = "manual_calculate",
) -> Dict[str, Any]:
    """与 `PropertiesUIMixin._record_property_result` 一致的列集合。"""
    props_copy = dict(props or {})
    rock_candidates = props_copy.get("rock_candidates", [])
    if not isinstance(rock_candidates, list):
        rock_candidates = []
    top_prob = 0.0
    if rock_candidates and isinstance(rock_candidates[0], dict):
        try:
            top_prob = float(rock_candidates[0].get("probability", 0.0) or 0.0)
        except (TypeError, ValueError):
            top_prob = 0.0

    vp = props_copy.get("vp")
    vs = props_copy.get("vs")
    vp_vs_ratio = props_copy.get("vp_vs_ratio")
    try:
        fvp = float(vp) if vp is not None else 0.0
        fvs = float(vs) if vs is not None else 0.0
        if vp_vs_ratio is None and fvs > 0:
            vp_vs_ratio = fvp / fvs
    except (TypeError, ValueError):
        pass

    point_id = f"point_{x:.2f}_{z:.2f}"
    row: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "point_id": point_id,
        "source": source,
        "x_km": float(x),
        "z_km": float(z),
        "vp_km_s": json_compatible(props_copy.get("vp")),
        "vs_km_s": json_compatible(props_copy.get("vs")),
        "vp_vs_ratio": json_compatible(vp_vs_ratio),
        "vs_source": str(props_copy.get("vs_source", "")),
        "density_g_cm3": json_compatible(props_copy.get("density")),
        "pressure_mpa": json_compatible(props_copy.get("pressure")),
        "temperature_c": json_compatible(props_copy.get("temperature")),
        "zone": str(props_copy.get("zone", "deep") or "deep"),
        "rock_type": str(props_copy.get("rock_type", "UNKNOWN")),
        "felsic_or_mafic": json_compatible(props_copy.get("felsic_or_mafic")),
        "rock_facies": json_compatible(props_copy.get("rock_facies")),
        "sio2_wt": json_compatible(props_copy.get("sio2_wt")),
        "rock_classification": json_compatible(props_copy.get("rock_classification")),
        "metamorphic_grade": json_compatible(props_copy.get("metamorphic_grade")),
        "geological_meaning": json_compatible(props_copy.get("geological_meaning")),
        "rock_probability": json_compatible(props_copy.get("rock_probability", top_prob)),
        "confidence_level": str(props_copy.get("confidence_level", "unknown")),
        "low_confidence": bool(props_copy.get("low_confidence", False)),
        "zone_prior_applied": bool(props_copy.get("zone_prior_applied", False)),
        "rerank_applied": bool(props_copy.get("rerank_applied", False)),
        "zone_constraint_applied": bool(props_copy.get("zone_constraint_applied", False)),
        "confidence_top1": json_compatible(props_copy.get("confidence_top1", top_prob)),
        "confidence_top2": json_compatible(props_copy.get("confidence_top2", 0.0)),
        "confidence_gap": json_compatible(props_copy.get("confidence_gap", 0.0)),
        "bulk_modulus_gpa": json_compatible(props_copy.get("bulk_modulus")),
        "shear_modulus_gpa": json_compatible(props_copy.get("shear_modulus")),
        "rock_candidates_json": json.dumps(
            json_compatible(rock_candidates),
            ensure_ascii=False,
        ),
    }
    return row


def format_property_summary(props: Dict[str, Any]) -> str:
    lines = []
    x = props.get("_x_disp")
    z = props.get("_z_disp")
    if x is not None and z is not None:
        lines.append(f"坐标: x={float(x):.3f} km, z={float(z):.3f} km")

    vp = props.get("vp")
    vs = props.get("vs")
    if vp is not None:
        lines.append(f"P 波速度: {float(vp):.3f} km/s")
    if vs is not None:
        lines.append(f"S 波速度: {float(vs):.3f} km/s")
    vs_src = str(props.get("vs_source") or "").strip()
    if vs_src:
        lines.append(f"Vs 来源: {vs_src}")

    zone = str(props.get("zone", "deep") or "deep").strip()
    lines.append(f"分区: {zone}")

    rho = props.get("density")
    if rho is not None:
        lines.append(f"密度: {float(rho):.3f} g/cm³")
    pres = props.get("pressure")
    if pres is not None:
        lines.append(f"压力: {float(pres):.1f} MPa")
    temp = props.get("temperature")
    if temp is not None:
        lines.append(f"温度: {float(temp):.1f} °C")

    cands = props.get("rock_candidates")
    if isinstance(cands, list) and cands:
        lines.append("岩性候选:")
        for i, c in enumerate(cands[:6], start=1):
            if isinstance(c, dict):
                rt = c.get("rock_type", "?")
                pr = float(c.get("probability", 0.0) or 0.0)
                lines.append(f"  {i}. {rt}: {pr:.2%}")
    else:
        rt = props.get("rock_type", "UNKNOWN")
        rp = props.get("rock_probability", 0.0)
        try:
            lines.append(f"岩性: {rt} （概率 {float(rp):.2%}）")
        except (TypeError, ValueError):
            lines.append(f"岩性: {rt}")

    for key, label in (
        ("bulk_modulus", "体积模量"),
        ("shear_modulus", "剪切模量"),
    ):
        v = props.get(key)
        if v is not None:
            lines.append(f"{label}: {float(v):.2f} GPa")

    return "\n".join(lines)
