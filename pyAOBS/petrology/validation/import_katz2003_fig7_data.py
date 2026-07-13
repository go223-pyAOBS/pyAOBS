"""
Build / inspect Katz (2003) Fig.7 experimental database (P–T–F in anhydrous space).

  py -3.11 petrology/validation/import_katz2003_fig7_data.py --build
  py -3.11 petrology/validation/import_katz2003_fig7_data.py --list
  py -3.11 petrology/validation/import_katz2003_fig7_data.py --stats
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DATA = Path(__file__).resolve().parents[1] / "data" / "katz2003_fig7"
_FIG6_DATA = Path(__file__).resolve().parents[1] / "data" / "katz2003_fig6"
_EXPERIMENTS = _DATA / "experiments.csv"
_JAQUES_T4 = _FIG6_DATA / "jaques_green_1980_pyrolite_table4.csv"
_JAQUES_T5 = _FIG6_DATA / "jaques_green_1980_tinaquillo_table5.csv"
_FIG6_EXP = _FIG6_DATA / "experiments.csv"

_FIELDNAMES = ("id", "p_gpa", "t_c", "f", "study_id", "source_short", "verified", "notes")
_VERIFY_RANK = {"yes": 3, "partial": 2, "no": 1}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(_FIELDNAMES))
        w.writeheader()
        w.writerows(rows)


def _key(row: dict[str, str]) -> tuple[float, float]:
    return (round(float(row["p_gpa"]), 3), round(float(row["t_c"]), 1))


def _merge_rows(*groups: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: dict[tuple[float, float], dict[str, str]] = {}
    for group in groups:
        for row in group:
            k = _key(row)
            existing = merged.get(k)
            if existing is None:
                merged[k] = row
                continue
            r_new = _VERIFY_RANK.get(row.get("verified", ""), 0)
            r_old = _VERIFY_RANK.get(existing.get("verified", ""), 0)
            if r_new > r_old:
                merged[k] = row
    return sorted(merged.values(), key=lambda r: (float(r["p_gpa"]), float(r["t_c"])))


def _from_fig6() -> list[dict[str, str]]:
    if not _FIG6_EXP.is_file():
        return []
    rows: list[dict[str, str]] = []
    for r in _read_csv(_FIG6_EXP):
        rows.append(
            {
                "id": r["id"],
                "p_gpa": r["p_gpa"],
                "t_c": r["t_c"],
                "f": r["f"],
                "study_id": r.get("study_id", ""),
                "source_short": r.get("source_short", ""),
                "verified": r.get("verified", "partial"),
                "notes": r.get("notes", "from katz2003_fig6/experiments.csv"),
            }
        )
    return rows


def _from_jaques(csv_path: Path, study_id: str, id_prefix: str, table_label: str) -> list[dict[str, str]]:
    if not csv_path.is_file():
        return []
    rows: list[dict[str, str]] = []
    for r in _read_csv(csv_path):
        pk = float(r["p_kbar"])
        tc = float(r["t_c"])
        rows.append(
            {
                "id": f"{id_prefix}_{int(pk)}k_{int(tc)}",
                "p_gpa": r["p_gpa"],
                "t_c": r["t_c"],
                "f": r["f"],
                "study_id": study_id,
                "source_short": "Jaques_Green_1980",
                "verified": "partial",
                "notes": f"Jaques & Green (1980) {table_label}; {pk:g} kbar",
            }
        )
    return rows


def build_fig7_experiments() -> Path:
    """Merge Fig.6 seed points + full Jaques Table 4/5 into Fig.7 database."""
    merged = _merge_rows(
        _from_fig6(),
        _from_jaques(_JAQUES_T4, "JG1980_HP", "JG80", "Table 4 pyrolite"),
        _from_jaques(_JAQUES_T5, "JG1980_TQ", "JG80TQ", "Table 5 Tinaquillo"),
    )
    _write_csv(_EXPERIMENTS, merged)
    return _EXPERIMENTS


def list_points() -> None:
    if not _EXPERIMENTS.is_file():
        print(f"Missing {_EXPERIMENTS} — run with --build")
        return
    rows = _read_csv(_EXPERIMENTS)
    print(f"Katz Fig.7 database: {_EXPERIMENTS}  ({len(rows)} points)")
    for r in rows:
        print(
            f"  P={float(r['p_gpa']):g} GPa  T={float(r['t_c']):4.0f} C  "
            f"F={float(r['f']):.2f}  {r.get('study_id', '')}  [{r.get('verified', '')}]"
        )


def print_stats() -> None:
    from petrology.melting.katz2003 import katz2003_fig7_near_hirschmann_stats, load_katz2003_fig7_experiments

    rows = load_katz2003_fig7_experiments()
    if not rows:
        print("No Fig.7 experiments loaded.")
        return
    stats = katz2003_fig7_near_hirschmann_stats(rows, dt_c=10.0)
    print(f"Total experiments: {len(rows)}")
    print(
        f"Within ±{stats['dt_c']:.0f}°C of Hirschmann (2000) solidus: "
        f"{stats['n_near']:.0f} points"
    )
    print(f"  mean F = {stats['mean_f_wt_pct']:.1f} wt%")
    print(f"  F = 0: {stats['n_zero']:.0f} points")
    print("(Katz caption: 29 expts, mean 10 wt%, 5 with no melting — requires full ESI Table S2)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Katz (2003) Fig.7 experimental database")
    parser.add_argument("--build", action="store_true", help="Merge Fig.6 + Jaques into fig7/experiments.csv")
    parser.add_argument("--list", action="store_true", help="List Fig.7 experiments")
    parser.add_argument("--stats", action="store_true", help="Near-Hirschmann solidus statistics")
    args = parser.parse_args()

    if args.build:
        path = build_fig7_experiments()
        rows = _read_csv(path)
        print(f"Wrote {path} ({len(rows)} unique P–T–F points)")
    if args.list:
        list_points()
    if args.stats:
        print_stats()
    if not any((args.build, args.list, args.stats)):
        parser.print_help()


if __name__ == "__main__":
    main()
