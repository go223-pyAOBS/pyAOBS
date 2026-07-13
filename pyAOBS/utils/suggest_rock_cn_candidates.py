from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


PHRASE_MAP: list[tuple[str, str]] = [
    ("coarsed-grained", "粗粒"),
    ("coarse-grained", "粗粒"),
    ("fine-grained", "细粒"),
    ("retrograde", "退变质"),
    ("foliated", "片理化"),
    ("altered", "蚀变"),
    ("felsic", "长英质"),
    ("mafic", "镁铁质"),
    ("tonalitic", "英云闪长质"),
    ("granodioritic", "花岗闪长质"),
    ("trondhjemitic", "奥长花岗质"),
    ("calc-silicate", "钙硅酸盐"),
    ("meta", "变"),
]


TOKEN_MAP: dict[str, str] = {
    "gneiss": "片麻岩",
    "schist": "片岩",
    "granulite": "麻粒岩",
    "eclogite": "榴辉岩",
    "garnetite": "石榴岩",
    "granite": "花岗岩",
    "granodiorite": "花岗闪长岩",
    "diorite": "闪长岩",
    "tonalite": "英云闪长岩",
    "trondhjemite": "奥长花岗岩",
    "basalt": "玄武岩",
    "gabbro": "辉长岩",
    "anorthosite": "斜长岩",
    "peridotite": "橄榄岩",
    "harzburgite": "方辉橄榄岩",
    "lherzolite": "二辉橄榄岩",
    "dunite": "纯橄榄岩",
    "pyroxenite": "辉石岩",
    "marble": "大理岩",
    "quartzite": "石英岩",
    "greywacke": "杂砂岩",
    "webster": "二辉橄榄岩",
    "dolomite": "白云岩",
    "skarn": "矽卡岩",
    "metapelite": "变泥质岩",
    "metapelitic": "变泥质岩",
    "metagabbro": "变辉长岩",
    "metabasalt": "变玄武岩",
    "metadacite": "变英安岩",
    "metarhyolite": "变流纹岩",
    "orthogneiss": "正片麻岩",
    "paragneiss": "副片麻岩",
    "kinzigite": "金兹岩",
    "albitite": "钠长岩",
    "jadeite": "硬玉岩",
    "garnet": "石榴子石",
    "amphibolite": "角闪岩",
    "granofels": "粒状角岩",
    "qf": "石英长石",
    "qtz": "石英",
    "pl": "斜长石",
    "bi": "黑云母",
    "hb": "角闪石",
    "hbl": "角闪石",
    "cpx": "单斜辉石",
    "opx": "斜方辉石",
    "ga": "石榴子石",
    "gt": "石榴子石",
    "ks": "钾长石",
    "ep": "绿帘石",
    "phlog": "金云母",
    "phlogopite": "金云母",
    "graph": "石墨",
    "pyr": "黄铁矿",
    "py": "辉石",
    "lesz": "莱什",
    "nc": "北卡",
}


ROCK_SUFFIX_PRIORITY = [
    "榴辉岩",
    "麻粒岩",
    "片麻岩",
    "片岩",
    "辉长岩",
    "花岗岩",
    "闪长岩",
    "英云闪长岩",
    "橄榄岩",
    "大理岩",
    "石英岩",
]


def _normalize_en(text: str) -> str:
    s = _clean_text(text).lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s


def _translate_rock_type(rock_type: str) -> tuple[str, str, str]:
    s = _normalize_en(rock_type)

    for src, dst in PHRASE_MAP:
        s = re.sub(rf"\b{re.escape(src)}\b", dst, s)

    parts = re.split(r"[\s\-/(),]+", s)
    cn_parts: list[str] = []
    unknown: list[str] = []
    for part in parts:
        p = _clean_text(part)
        if not p:
            continue
        if re.search(r"[\u4e00-\u9fff]", p):
            cn_parts.append(p)
            continue
        if p in TOKEN_MAP:
            cn_parts.append(TOKEN_MAP[p])
        elif re.fullmatch(r"[0-9.]+", p):
            continue
        else:
            unknown.append(p)

    if not cn_parts:
        return "", "low", "no_token_matched"

    # 去重但保持顺序
    seen = set()
    uniq_parts = []
    for p in cn_parts:
        if p not in seen:
            uniq_parts.append(p)
            seen.add(p)

    suggestion = "-".join(uniq_parts)

    # 尝试把岩石主名放到末尾，便于阅读
    for suffix in ROCK_SUFFIX_PRIORITY:
        if suffix in uniq_parts and not suggestion.endswith(suffix):
            reordered = [x for x in uniq_parts if x != suffix] + [suffix]
            suggestion = "-".join(reordered)
            break

    if unknown:
        confidence = "medium" if len(unknown) <= 2 else "low"
        reason = f"unknown_tokens={','.join(unknown[:6])}"
    else:
        confidence = "high"
        reason = "all_tokens_mapped"

    return suggestion, confidence, reason


def suggest(
    unresolved_csv: Path,
    output_csv: Path,
) -> tuple[int, int]:
    rows = _read_csv(unresolved_csv)
    out_rows: list[dict[str, str]] = []
    high_count = 0
    for row in rows:
        rock_type = _clean_text(row.get("rock_type", ""))
        count = _clean_text(row.get("count", ""))
        suggestion, confidence, reason = _translate_rock_type(rock_type)
        if confidence == "high":
            high_count += 1
        out_rows.append(
            {
                "rock_type": rock_type,
                "count": count,
                "suggested_cn": suggestion,
                "confidence": confidence,
                "reason": reason,
                "accepted": "",
                "final_cn": "",
            }
        )

    _write_csv(
        output_csv,
        out_rows,
        [
            "rock_type",
            "count",
            "suggested_cn",
            "confidence",
            "reason",
            "accepted",
            "final_cn",
        ],
    )
    return len(out_rows), high_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Chinese-name suggestions for unresolved rock types."
    )
    parser.add_argument(
        "--unresolved",
        required=True,
        help="Path to rock_type_cn_unresolved.csv.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Suggestion output CSV path.",
    )
    args = parser.parse_args()

    unresolved_path = Path(args.unresolved).expanduser().resolve()
    if not unresolved_path.exists():
        raise FileNotFoundError(f"Unresolved file not found: {unresolved_path}")

    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else unresolved_path.with_name("rock_type_cn_suggestions.csv")
    )

    total, high_count = suggest(unresolved_path, out_path)
    print(f"suggestions_file={out_path}")
    print(f"suggestions_total={total}")
    print(f"suggestions_high_confidence={high_count}")


if __name__ == "__main__":
    main()
