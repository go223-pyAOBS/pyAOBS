"""
Katz et al. (2003) Fig. 1–9 — LIP workflow registry and batch export.

Used by LIP GUI (验证 → Katz 图件) and CLI:

  python petrology/validation/katz2003_workflow.py
  python petrology/validation/katz2003_workflow.py --only katz_fig1,katz_fig7 -o petrology/figures

Per-figure scripts with extra CLI flags remain as ``reproduce_katz2003_fig*.py``;
for batch PNG export without extra options, prefer ``katz2003_workflow.py``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"


@dataclass(frozen=True)
class Katz2003Panel:
    key: str
    tab_title: str
    filename: str
    builder: Callable


def _panels() -> tuple[Katz2003Panel, ...]:
    from petrology.melting import katz2003 as k

    return (
        Katz2003Panel("katz_fig1", "Fig. 1 — T–P 固/液相线", "katz2003_fig1.png", k.build_katz2003_fig1_figure),
        Katz2003Panel("katz_fig2", "Fig. 2 — 无水 F(T)", "katz2003_fig2.png", k.build_katz2003_fig2_figure),
        Katz2003Panel("katz_fig3", "Fig. 3 — 含水固相线", "katz2003_fig3.png", k.build_katz2003_fig3_figure),
        Katz2003Panel("katz_fig4", "Fig. 4 — 含水 F(T)", "katz2003_fig4.png", k.build_katz2003_fig4_figure),
        Katz2003Panel("katz_fig5", "Fig. 5 — F vs H₂O", "katz2003_fig5.png", k.build_katz2003_fig5_figure),
        Katz2003Panel("katz_fig6", "Fig. 6 — 模型 vs 实验（验收）", "katz2003_fig6.png", k.build_katz2003_fig6_figure),
        Katz2003Panel("katz_fig7", "Fig. 7 — 实验 F(P,T)", "katz2003_fig7.png", k.build_katz2003_fig7_figure),
        Katz2003Panel("katz_fig8", "Fig. 8 — 含水标定", "katz2003_fig8.png", k.build_katz2003_fig8_figure),
        Katz2003Panel("katz_fig9", "Fig. 9 — 无水模型对比", "katz2003_fig9.png", k.build_katz2003_fig9_figure),
    )


KATZ2003_PANELS: tuple[Katz2003Panel, ...] = _panels()

KATZ2003_PANEL_KEYS: tuple[str, ...] = tuple(p.key for p in KATZ2003_PANELS)


def panel_by_key(key: str) -> Katz2003Panel:
    for panel in KATZ2003_PANELS:
        if panel.key == key:
            return panel
    raise KeyError(f"Unknown Katz (2003) panel: {key!r}")


def build_katz2003_figure(key: str, **kwargs):
    """Build one Katz (2003) figure by panel key."""
    panel = panel_by_key(key)
    if kwargs:
        return panel.builder(**kwargs)
    return panel.builder()


def export_katz2003_figures(
    out_dir: Path | None = None,
    *,
    keys: tuple[str, ...] | None = None,
    dpi: int = 150,
) -> list[Path]:
    """Write Katz Fig. 1–9 PNGs; returns written paths."""
    import matplotlib.pyplot as plt

    base = Path(out_dir) if out_dir is not None else FIG_DIR
    base.mkdir(parents=True, exist_ok=True)

    selected = KATZ2003_PANELS
    if keys is not None:
        key_set = set(keys)
        selected = tuple(p for p in KATZ2003_PANELS if p.key in key_set)

    written: list[Path] = []
    for panel in selected:
        fig = panel.builder()
        path = base / panel.filename
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Katz (2003) Fig. 1–9 for LIP workflow")
    parser.add_argument("-o", "--out-dir", type=Path, default=FIG_DIR)
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated panel keys (default: all), e.g. katz_fig1,katz_fig6",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--show", action="store_true", help="Show last figure instead of closing")
    args = parser.parse_args()

    keys: tuple[str, ...] | None = None
    if args.only.strip():
        keys = tuple(k.strip() for k in args.only.split(",") if k.strip())

    paths = export_katz2003_figures(args.out_dir, keys=keys, dpi=args.dpi)
    for path in paths:
        print(f"Wrote {path}")

    if args.show and paths:
        import matplotlib.pyplot as plt

        last_key = keys[-1] if keys else KATZ2003_PANEL_KEYS[-1]
        build_katz2003_figure(last_key)
        plt.show()


if __name__ == "__main__":
    main()
