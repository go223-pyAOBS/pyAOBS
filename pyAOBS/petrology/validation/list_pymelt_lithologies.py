"""
Print pyMelt lithology registry and LIP presets.

  python petrology/validation/list_pymelt_lithologies.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    from petrology.melting.pymelt_lithology_adapter import print_lithology_catalog

    print_lithology_catalog()


if __name__ == "__main__":
    main()
