"""
Verify ``katz2003.py`` constants against Katz et al. (2003) Table 2 (Excel).

Usage::

    py -3.11 petrology/validation/verify_katz2003_table2.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XLSX = ROOT / "data" / "Table2_岩石学计算参数表_katz.xlsx"


def _norm_param(name: str) -> str:
    s = str(name).strip()
    repl = {
        "\u2081": "1",
        "\u2082": "2",
        "\u2083": "3",
        "\u03b2": "beta",
        "\u03b3": "gamma",
        "\u03c7": "chi",
        "\u03bb": "plam",
        "D\u2082O": "DH2O",
        "\u0394S": "DeltaS",
        "c_P": "c_P",
        "\u03b1\u209b": "alpha_s",
        "\u03b1_f": "alpha_f",
        "\u03c1\u209b": "rho_s",
        "\u03c1_f": "rho_f",
    }
    for old, new in repl.items():
        s = s.replace(old, new)
    s = re.sub(r"\s+", "", s)
    return s


def _excel_si_value(param: str, excel: float, unit: str | None) -> float:
    """Convert Table 2 display values to SI units used in code."""
    u = (unit or "").replace(" ", "")
    if param in ("alpha_s", "alpha_f") and "10" in u and "K" in u:
        return excel * 1.0e-6
    if param in ("rho_s", "rho_f") and "10" in u and "kg" in u:
        return excel * 1.0e3
    return excel


def _code_value(param: str):
    from petrology.melting import katz2003 as k

    mapping: dict[str, tuple[str, int | None] | None] = {
        "A1": ("_SOLIDUS", 0),
        "A2": ("_SOLIDUS", 1),
        "A3": ("_SOLIDUS", 2),
        "B1": ("_LHERZOLITE_LIQUIDUS", 0),
        "B2": ("_LHERZOLITE_LIQUIDUS", 1),
        "B3": ("_LHERZOLITE_LIQUIDUS", 2),
        "C1": ("_LIQUIDUS", 0),
        "C2": ("_LIQUIDUS", 1),
        "C3": ("_LIQUIDUS", 2),
        "r1": ("_R_CPX", 0),
        "r2": ("_R_CPX", 1),
        "beta1": ("_BETA1", None),
        "beta2": ("_BETA2", None),
        "K": ("_PK", None),
        "gamma": ("_GAMMA", None),
        "DH2O": ("_DH2O", None),
        "chi1": ("_CHI1", None),
        "chi2": ("_CHI2", None),
        "plam": ("_PLAM", None),
        "c_P": ("_CP_J_KG", None),
        "alpha_s": ("_ALPHA_S_K", None),
        "alpha_f": ("_ALPHA_F_K", None),
        "rho_s": ("_RHO_S", None),
        "rho_f": ("_RHO_F", None),
        "DeltaS": ("_DELTA_S", None),
    }
    spec = mapping.get(param)
    if spec is None:
        return None
    attr, idx = spec
    if not hasattr(k, attr):
        return None
    val = getattr(k, attr)
    return float(val[idx] if idx is not None else val)


def load_table2_excel(path: Path) -> list[dict[str, object]]:
    df = pd.read_excel(path, sheet_name=0, header=None)
    rows: list[dict[str, object]] = []
    for i in range(2, len(df)):
        raw = df.iloc[i, 0]
        val = df.iloc[i, 2]
        unit = df.iloc[i, 3]
        if pd.isna(raw) or pd.isna(val):
            continue
        param = _norm_param(str(raw))
        rows.append(
            {
                "param": param,
                "raw": str(raw),
                "for": None if pd.isna(df.iloc[i, 1]) else str(df.iloc[i, 1]),
                "excel": float(val),
                "excel_si": _excel_si_value(
                    param,
                    float(val),
                    None if pd.isna(unit) else str(unit),
                ),
                "unit": None if pd.isna(unit) else str(unit),
            }
        )
    return rows


def compare(path: Path) -> dict[str, object]:
    from petrology.melting import katz2003 as k

    excel_rows = load_table2_excel(path)
    comparison: list[dict[str, object]] = []
    ok = mismatch = missing = 0

    for row in excel_rows:
        param = str(row["param"])
        xval = float(row["excel_si"])
        code_val = _code_value(param)
        if code_val is None:
            status = "NOT_IN_CODE"
            missing += 1
        elif abs(code_val - xval) < 1e-9:
            status = "OK"
            ok += 1
        else:
            status = "MISMATCH"
            mismatch += 1
        comparison.append(
            {
                **row,
                "code": code_val,
                "delta": None if code_val is None else code_val - xval,
                "status": status,
            }
        )

    pymelt_params: object = None
    try:
        from petrology.melting.pymelt_lithology_adapter import instantiate_pymelt_lithology

        pm = instantiate_pymelt_lithology("katz_lherzolite")
        pymelt_params = dict(pm.parameters)
    except Exception as exc:
        pymelt_params = {"error": str(exc)}

    sample = {}
    for p in (1.0, 3.0):
        sample[str(p)] = {
            "Ts_C": k.katz2003_solidus(p),
            "T_lherz_C": k.katz2003_lherzolite_liquidus(p),
            "T_liq_C": k.katz2003_liquidus(p),
            "F_cpx_default": k.katz2003_fcpx_out(p_gpa=p),
        }

    return {
        "xlsx": str(path),
        "summary": {"ok": ok, "mismatch": mismatch, "not_in_code": missing},
        "comparison": comparison,
        "pymelt_katz_lherzolite": pymelt_params,
        "default_cpx_mass": k._DEFAULT_CPX_MASS,
        "sample_T_F_at_P": sample,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify katz2003.py vs Table 2 Excel")
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    parser.add_argument("-o", "--out", type=Path, default=None, help="Write JSON report")
    args = parser.parse_args()

    if not args.xlsx.is_file():
        print(f"Missing Table 2 workbook: {args.xlsx}", file=sys.stderr)
        sys.exit(1)

    report = compare(args.xlsx)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")

    s = report["summary"]
    print(f"Table 2 vs katz2003.py: OK={s['ok']}  MISMATCH={s['mismatch']}  NOT_IN_CODE={s['not_in_code']}")
    for row in report["comparison"]:
        if row["status"] != "OK":
            print(f"  {row['param']}: excel={row['excel']} code={row['code']} -> {row['status']}")
    print(f"Default Mcpx in katz2003.py: {report['default_cpx_mass']}")
    print(f"pyMelt katz_lherzolite Mcpx: {report['pymelt_katz_lherzolite'].get('Mcpx', '?')}")

    if s["mismatch"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
