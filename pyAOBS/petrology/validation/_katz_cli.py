"""Shared CLI helpers for ``reproduce_katz2003_fig*.py`` scripts."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = _ROOT / "petrology" / "figures"


def ensure_repo_root_on_path() -> None:
    """Insert repo root on ``sys.path`` so ``petrology`` imports work when run as script."""
    root = str(_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def bootstrap() -> None:
    """Alias for :func:`ensure_repo_root_on_path`."""
    ensure_repo_root_on_path()


def default_fig_path(fig_num: int) -> Path:
    return FIG_DIR / f"katz2003_fig{fig_num}.png"


def parse_float_list(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def write_figure(fig, path: Path, *, show: bool = False, dpi: int = 150) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Wrote {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
