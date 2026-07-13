"""Generate petrology/data/cdat/*.cdat from built-in catalog."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from petrology.fc.cdat_library import build_library, list_samples


def main() -> None:
    paths = build_library(force=True)
    print(f"Wrote {len(list_samples())} samples + all_samples.cdat → {paths['catalog'].parent}")
    for sid, p in sorted(paths.items()):
        if sid in ("all_samples", "catalog"):
            continue
        print(f"  {sid}: {p.name}")


if __name__ == "__main__":
    main()
