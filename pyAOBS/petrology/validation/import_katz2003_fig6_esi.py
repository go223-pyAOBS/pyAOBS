"""
Katz (2003) Fig.6 experimental point table — list, LEPR cross-check, ESI placeholder.

  py -3.11 petrology/validation/import_katz2003_fig6_esi.py --list
  py -3.11 petrology/validation/import_katz2003_fig6_esi.py --check-lepr
  py -3.11 petrology/validation/import_katz2003_fig6_esi.py --esi petrology/data/es2002gc000433.pdf
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_DATA = Path(__file__).resolve().parents[1] / "data" / "katz2003_fig6"
_EXPERIMENTS = _DATA / "experiments.csv"
_STUDIES = _DATA / "studies.csv"
_LEPR_DEFAULT = Path(__file__).resolve().parents[1] / "data" / "ExPetDB_download_LEPR-2007-08-17.xlsx"
_ESI_DEFAULT = Path(__file__).resolve().parents[1] / "data" / "es2002gc000433.pdf"

_PANEL_LABEL = {"a": "0 GPa", "b": "1 GPa", "c": "1.5 GPa", "d": "3 GPa"}
_JAQUES_XLSX = (
    Path(__file__).resolve().parents[1] / "data" / "Pyrolite_equilibrium_melt_compositions_豆包AI生成.xlsx"
)
_JAQUES_TABLE4_CSV = _DATA / "jaques_green_1980_pyrolite_table4.csv"
_JAQUES_TABLE5_CSV = _DATA / "jaques_green_1980_tinaquillo_table5.csv"
_JAQUES_SHEETS = {
    "table4": "Pyrolite 平衡熔体成分",
    "table5": "Tinaquillo lherzolite 平衡熔体成分",
}


def _fig6_panel_for_kbar(p_kbar: float) -> str:
    pg = float(p_kbar) / 10.0
    if abs(pg - 1.0) < 0.05:
        return "b"
    if abs(pg - 1.5) < 0.05:
        return "c"
    return ""


def _coerce_float(val) -> float | None:
    try:
        import pandas as pd

        if pd.isna(val):
            return None
    except ImportError:
        if val is None:
            return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _extract_ptf_from_sheet(df) -> list[dict[str, str]]:
    """Parse Jaques & Green (1980) Table 4/5 layout: P/T/F in rows 1–3."""
    p_row = df.iloc[1, 1:].tolist()
    t_row = df.iloc[2, 1:].tolist()
    f_row = df.iloc[3, 1:].tolist()
    rows: list[dict[str, str]] = []
    for pk, tc, fv in zip(p_row, t_row, f_row):
        pk_f = _coerce_float(pk)
        tc_f = _coerce_float(tc)
        fv_f = _coerce_float(fv)
        if pk_f is None or tc_f is None or fv_f is None:
            continue
        pg = pk_f / 10.0
        rows.append(
            {
                "p_kbar": f"{pk_f:g}",
                "p_gpa": f"{pg:g}",
                "t_c": f"{tc_f:g}",
                "f": f"{fv_f / 100.0:.4f}".rstrip("0").rstrip("."),
                "fig6_panel": _fig6_panel_for_kbar(pk_f),
            }
        )
    return rows


def extract_jaques_tables(xlsx_path: Path | None = None) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return (Table 4 Pyrolite, Table 5 Tinaquillo lherzolite) P–T–F rows."""
    path = xlsx_path or _JAQUES_XLSX
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas required") from exc

    t4 = pd.read_excel(path, sheet_name=_JAQUES_SHEETS["table4"], engine="calamine", header=None)
    t5 = pd.read_excel(path, sheet_name=_JAQUES_SHEETS["table5"], engine="calamine", header=None)
    return _extract_ptf_from_sheet(t4), _extract_ptf_from_sheet(t5)


def _write_ptf_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["p_kbar", "p_gpa", "t_c", "f", "fig6_panel"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def extract_jaques_from_xlsx(xlsx_path: Path | None = None) -> tuple[Path, Path]:
    """Write Table 4/5 CSV from the Jaques xlsx and refresh experiments.csv Jaques rows."""
    t4, t5 = extract_jaques_tables(xlsx_path)
    _write_ptf_csv(_JAQUES_TABLE4_CSV, t4)
    _write_ptf_csv(_JAQUES_TABLE5_CSV, t5)
    n_exp = sync_jaques_experiments()
    print(f"Wrote {_JAQUES_TABLE4_CSV} ({len(t4)} P–T–F points)")
    print(f"Wrote {_JAQUES_TABLE5_CSV} ({len(t5)} P–T–F points)")
    print(f"Updated {_EXPERIMENTS} (+{n_exp} Jaques Fig.6 points on panels b/c)")
    return _JAQUES_TABLE4_CSV, _JAQUES_TABLE5_CSV


def sync_jaques_experiments() -> int:
    """Merge Jaques Table 4/5 (1 & 1.5 GPa) into experiments.csv."""
    specs = (
        (_JAQUES_TABLE4_CSV, "JG1980_HP", "JG80", "Table 4 Hawaiian pyrolite"),
        (_JAQUES_TABLE5_CSV, "JG1980_TQ", "JG80TQ", "Table 5 Tinaquillo lherzolite"),
    )
    new_rows: list[dict[str, str]] = []
    for csv_path, study_id, id_prefix, table_label in specs:
        if not csv_path.is_file():
            continue
        for r in _read_csv(csv_path):
            if r.get("fig6_panel") not in ("b", "c"):
                continue
            pk = float(r["p_kbar"])
            tc = float(r["t_c"])
            new_rows.append(
                {
                    "id": f"{id_prefix}_{int(pk)}k_{int(tc)}",
                    "p_gpa": r["p_gpa"],
                    "t_c": r["t_c"],
                    "f": r["f"],
                    "study_id": study_id,
                    "source_short": "Jaques_Green_1980",
                    "fig6_panel": r["fig6_panel"],
                    "verified": "partial",
                    "notes": f"Jaques & Green (1980) {table_label}; {pk:g} kbar; F=% melt/100",
                }
            )

    if not _EXPERIMENTS.is_file():
        all_rows = new_rows
    else:
        existing = _read_csv(_EXPERIMENTS)
        drop_ids = {s[1] for s in specs}
        all_rows = [r for r in existing if r.get("study_id") not in drop_ids] + new_rows

    fieldnames = [
        "id",
        "p_gpa",
        "t_c",
        "f",
        "study_id",
        "source_short",
        "fig6_panel",
        "verified",
        "notes",
    ]
    with _EXPERIMENTS.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    return len(new_rows)


def _list_jaques_ptf_csv(path: Path, title: str) -> None:
    if not path.is_file():
        print(f"Missing {path}")
        return
    rows = _read_csv(path)
    print(f"{title}: {path}")
    for panel in ("b", "c", ""):
        pts = [r for r in rows if r.get("fig6_panel") == panel]
        if not pts:
            continue
        label = _PANEL_LABEL.get(panel, f"other pressures ({len(pts)} pts)")
        print(f"\n--- {label} ---")
        for r in sorted(pts, key=lambda x: float(x["t_c"])):
            print(
                f"  P={float(r['p_gpa']):g} GPa ({float(r['p_kbar']):g} kbar)  "
                f"T={float(r['t_c']):.0f} C  F={float(r['f']):.2f}"
            )


def list_jaques_tables() -> None:
    _list_jaques_ptf_csv(_JAQUES_TABLE4_CSV, "Jaques & Green (1980) Table 4 (Hawaiian Pyrolite)")
    _list_jaques_ptf_csv(_JAQUES_TABLE5_CSV, "Jaques & Green (1980) Table 5 (Tinaquillo lherzolite)")


def list_jaques_table4() -> None:
    list_jaques_tables()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def list_points() -> None:
    if not _EXPERIMENTS.is_file():
        print(f"Missing {_EXPERIMENTS}")
        return
    rows = _read_csv(_EXPERIMENTS)
    by_panel: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_panel.setdefault(row.get("fig6_panel", "?"), []).append(row)

    print(f"Katz Fig.6 point table: {_EXPERIMENTS}")
    print(f"Total rows: {len(rows)}")
    for panel in ("a", "b", "c", "d"):
        pts = by_panel.get(panel, [])
        label = _PANEL_LABEL.get(panel, panel)
        print(f"\n--- Panel ({panel}) {label} — {len(pts)} points ---")
        for r in sorted(pts, key=lambda x: float(x["t_c"])):
            ver = r.get("verified", "")
            print(
                f"  T={float(r['t_c']):4.0f} C  F={float(r['f']):.3f}  "
                f"{r.get('source_short', '')}  [{ver}]  {r.get('notes', '')[:60]}"
            )

    unverified = sum(1 for r in rows if r.get("verified") != "yes")
    print(f"\nNot verified against Katz ESI: {unverified}/{len(rows)}")


def check_lepr(lepr_path: Path, *, dp_gpa: float = 0.08, dt_c: float = 25.0) -> None:
    if not _EXPERIMENTS.is_file():
        print(f"Missing {_EXPERIMENTS}")
        return
    if not lepr_path.is_file():
        print(f"LEPR file not found: {lepr_path}")
        return

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas required") from exc

    engine = "calamine"
    try:
        exp_df = pd.read_excel(lepr_path, sheet_name="Experiment", engine=engine)
    except Exception as exc:
        raise SystemExit(
            "Reading LEPR xlsx failed (try: py -3.11 -m pip install python-calamine). "
            f"Original error: {exc}"
        ) from exc

    points = _read_csv(_EXPERIMENTS)
    print(f"LEPR Experiment rows: {len(exp_df)}")
    print(f"Cross-check {len(points)} Katz Fig.6 points (T±{dt_c} C, P±{dp_gpa} GPa, liq in Phases)\n")

    for row in points:
        pg = float(row["p_gpa"])
        tc = float(row["t_c"])
        src = row.get("source_short", "")
        mask = (
            exp_df["P (GPa)"].between(pg - dp_gpa, pg + dp_gpa)
            & exp_df["T (C)"].between(tc - dt_c, tc + dt_c)
            & exp_df["Phases"].astype(str).str.contains("liq", case=False, na=False)
        )
        if src:
            mask &= exp_df["Citation"].astype(str).str.contains(
                src.split("_")[0], case=False, na=False
            )
        hits = exp_df.loc[mask, ["Experiment", "T (C)", "P (GPa)", "Phases", "Citation"]].head(3)
        print(f"{row['id']}: P={pg} GPa T={tc} C F={row['f']} ({src})")
        if hits.empty:
            print("  LEPR: no match (check citation keyword or widen tolerance)")
        else:
            for _, h in hits.iterrows():
                print(f"  LEPR: {h['Experiment']}  T={h['T (C)']}  P={h['P (GPa)']}  {h['Phases']}")
        print()


def try_parse_esi(esi_path: Path) -> None:
    if not esi_path.is_file():
        print(
            f"ESI not found: {esi_path}\n"
            "Download Supporting Information from "
            "https://doi.org/10.1029/2002GC000433 "
            "and save as petrology/data/es2002gc000433.pdf"
        )
        return
    try:
        from pypdf import PdfReader
    except ImportError:
        print("Install pypdf to extract ESI text: py -3.11 -m pip install pypdf")
        return

    reader = PdfReader(str(esi_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    out = _DATA / "_esi_extracted.txt"
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {len(text)} chars to {out}")
    if "Summary of Experimental Peridotites" in text:
        print("Found 'Summary of Experimental Peridotites' — manually copy Table S2 into experiments.csv")
    else:
        print("Table title not found in extract; PDF may be scanned — transcribe tables by hand.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Katz (2003) Fig.6 experimental point utilities")
    parser.add_argument(
        "--extract-jaques",
        action="store_true",
        help="Extract Table 4/5 from Jaques xlsx into CSV and sync experiments.csv",
    )
    parser.add_argument(
        "--jaques-xlsx",
        type=Path,
        default=_JAQUES_XLSX,
        help="Jaques & Green (1980) equilibrium melt xlsx (Table 4 + Table 5 sheets)",
    )
    parser.add_argument(
        "--jaques",
        action="store_true",
        help="Print Jaques & Green (1980) Table 4/5 F(T) points",
    )
    parser.add_argument("--list", action="store_true", help="Print experiments.csv by panel")
    parser.add_argument("--check-lepr", action="store_true", help="Match points against LEPR Experiment sheet")
    parser.add_argument("--lepr", type=Path, default=_LEPR_DEFAULT)
    parser.add_argument("--esi", type=Path, default=_ESI_DEFAULT, help="Parse Katz ESI PDF to text")
    args = parser.parse_args()

    if args.extract_jaques:
        extract_jaques_from_xlsx(args.jaques_xlsx)
    if args.jaques:
        list_jaques_tables()
    if args.list:
        list_points()
    if args.check_lepr:
        check_lepr(args.lepr)
    if args.esi and args.esi.is_file():
        try_parse_esi(args.esi)
    elif not any(
        (args.list, args.check_lepr, args.jaques, args.extract_jaques, args.esi.is_file())
    ):
        parser.print_help()


if __name__ == "__main__":
    main()
