"""Import GetData Fig.5 digitization into figure05_digitized.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.figure05_digitized import (
    FIGURE05_JSON,
    FIGURE05_TXT,
    load_figure05_digitized,
    merge_getdata_fxl_files,
    merge_txt_into_json,
    parse_figure05_txt,
    parse_getdata_fxl_txt,
    save_figure05_digitized,
)

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_DEFAULT_100 = _DATA_DIR / "100Mpa.txt"
_DEFAULT_400 = _DATA_DIR / "400Mpa.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Fig.5 digitization → JSON")
    parser.add_argument(
        "--100Mpa",
        dest="mpa100",
        type=Path,
        default=_DEFAULT_100,
        help="GetData export for panel a (100 MPa VLC)",
    )
    parser.add_argument(
        "--400Mpa",
        dest="mpa400",
        type=Path,
        default=_DEFAULT_400,
        help="GetData export for panel b (400 MPa VLC)",
    )
    parser.add_argument(
        "--txt",
        type=Path,
        default=None,
        help="Optional legacy figure05_digitized.txt (section headers)",
    )
    parser.add_argument("--json", type=Path, default=FIGURE05_JSON)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = load_figure05_digitized(args.json) if args.json.is_file() else load_figure05_digitized()
    merged = base

    fxl_map = {}
    if args.mpa100.is_file():
        curves = parse_getdata_fxl_txt(args.mpa100)
        print(f"{args.mpa100.name}: F-loci {sorted(curves)}")
        for f, pts in sorted(curves.items()):
            print(
                f"  F={f:g}: {len(pts)} raw pts, "
                f"V_LC [{pts[0][0]:.3f}, {pts[-1][0]:.3f}], "
                f"ΔVp [{min(p[1] for p in pts):.3f}, {max(p[1] for p in pts):.3f}]"
            )
        fxl_map["a_vlc_100"] = args.mpa100
    else:
        print(f"Skip panel a — missing {args.mpa100}")

    if args.mpa400.is_file():
        curves = parse_getdata_fxl_txt(args.mpa400)
        print(f"{args.mpa400.name}: F-loci {sorted(curves)}")
        for f, pts in sorted(curves.items()):
            print(
                f"  F={f:g}: {len(pts)} raw pts, "
                f"V_LC [{pts[0][0]:.3f}, {pts[-1][0]:.3f}], "
                f"ΔVp [{min(p[1] for p in pts):.3f}, {max(p[1] for p in pts):.3f}]"
            )
        fxl_map["b_vlc_400"] = args.mpa400
    else:
        print(f"Skip panel b — missing {args.mpa400}")

    if args.dry_run:
        return

    if fxl_map:
        merged = merge_getdata_fxl_files(fxl_map, merged)
        for pkey in fxl_map:
            for fk, block in merged["panels"][pkey].get("f_loci", {}).items():
                fit = block.get("fit") or {}
                if fit:
                    print(
                        f"  {pkey} F={fk}: fit poly{fit.get('degree')} "
                        f"({fit.get('n_inliers')}/{fit.get('n_total')} inliers)"
                    )

    if args.txt and args.txt.is_file():
        sections = parse_figure05_txt(args.txt)
        print(f"Legacy txt: {len(sections)} sections from {args.txt}")
        merged = merge_txt_into_json(sections, merged)
    elif args.txt is None and FIGURE05_TXT.is_file():
        sections = parse_figure05_txt(FIGURE05_TXT)
        if sections:
            merged = merge_txt_into_json(sections, merged)

    save_figure05_digitized(merged, args.json)
    print(f"Wrote {args.json} (provisional={merged['meta'].get('provisional')})")


if __name__ == "__main__":
    main()
