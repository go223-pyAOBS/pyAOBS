from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _norm_key(value: str) -> str:
    s = _clean_text(value).lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff\-/,() ]+", "", s)
    return s.strip()


def _norm_key_loose(value: str) -> str:
    s = _norm_key(value)
    s = s.replace("-", " ").replace("/", " ").replace(",", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_mapping(rows: list[dict[str, str]]) -> tuple[dict[str, str], dict[str, str]]:
    strict_map: dict[str, str] = {}
    loose_map: dict[str, str] = {}
    for row in rows:
        rock = _clean_text(row.get("rock_type", ""))
        cn = _clean_text(row.get("岩石属性", ""))
        if not rock or not cn:
            continue
        strict_key = _norm_key(rock)
        loose_key = _norm_key_loose(rock)
        # 保留最早出现的映射，避免后续噪声覆盖
        if strict_key and strict_key not in strict_map:
            strict_map[strict_key] = cn
        if loose_key and loose_key not in loose_map:
            loose_map[loose_key] = cn
    return strict_map, loose_map


def fill_cn_names(
    input_csv: Path,
    output_csv: Path,
    unresolved_csv: Path,
) -> tuple[int, int, int]:
    rows = _read_csv(input_csv)
    if not rows:
        _write_csv(output_csv, [], [])
        _write_csv(unresolved_csv, [], ["rock_type", "count"])
        return 0, 0, 0

    fieldnames = list(rows[0].keys())
    if "岩石属性" not in fieldnames:
        fieldnames.append("岩石属性")

    strict_map, loose_map = _build_mapping(rows)

    filled = 0
    unresolved_counter: dict[str, int] = {}
    for row in rows:
        rock = _clean_text(row.get("rock_type", ""))
        cn = _clean_text(row.get("岩石属性", ""))
        if cn or not rock:
            continue

        strict_key = _norm_key(rock)
        loose_key = _norm_key_loose(rock)
        matched = strict_map.get(strict_key) or loose_map.get(loose_key)
        if matched:
            row["岩石属性"] = matched
            filled += 1
        else:
            unresolved_counter[rock] = unresolved_counter.get(rock, 0) + 1

    unresolved_rows = [
        {"rock_type": k, "count": str(v)}
        for k, v in sorted(unresolved_counter.items(), key=lambda x: (-x[1], x[0].lower()))
    ]

    _write_csv(output_csv, rows, fieldnames)
    _write_csv(unresolved_csv, unresolved_rows, ["rock_type", "count"])
    return len(rows), filled, len(unresolved_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill missing Chinese rock names (岩石属性) from existing rock_type mappings."
    )
    parser.add_argument("--input", required=True, help="Input rocks_merged.csv path.")
    parser.add_argument(
        "--output",
        default="",
        help="Output csv path. Default: overwrite input.",
    )
    parser.add_argument(
        "--unresolved",
        default="",
        help="Unresolved list path. Default: <input_dir>/rock_type_cn_unresolved.csv",
    )
    args = parser.parse_args()

    input_csv = Path(args.input).expanduser().resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    output_csv = Path(args.output).expanduser().resolve() if args.output else input_csv
    unresolved_csv = (
        Path(args.unresolved).expanduser().resolve()
        if args.unresolved
        else input_csv.with_name("rock_type_cn_unresolved.csv")
    )

    total, filled, unresolved_unique = fill_cn_names(input_csv, output_csv, unresolved_csv)
    print(f"output={output_csv}")
    print(f"rows_total={total}")
    print(f"rows_filled_cn={filled}")
    print(f"unresolved_unique_rock_type={unresolved_unique}")
    print(f"unresolved_file={unresolved_csv}")


if __name__ == "__main__":
    main()
