# -*- coding: utf-8 -*-
"""子进程入口：运行 RAYINVR 并收集 rays/obs（供 ``service.run_rayinvr_collect`` 调用）。"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path


def _ensure_import_path(explicit_root: Path | None) -> None:
    """把含 ``pyAOBS`` 包的目录加入 ``sys.path``（不依赖 ``python -m`` 解析）。"""
    candidates: list[Path] = []
    if explicit_root is not None:
        candidates.append(Path(explicit_root))
    here = Path(__file__).resolve()
    if len(here.parents) >= 4:
        candidates.append(here.parents[3])
    if len(here.parents) >= 3:
        candidates.append(here.parents[2].parent)
    seen: set[str] = set()
    for p in candidates:
        try:
            s = str(p.resolve())
        except OSError:
            continue
        if not p.is_dir() or s in seen:
            continue
        sys.path.insert(0, s)
        seen.add(s)


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) < 4:
        print("usage: subprocess_collect WD ROOT MAX_RAYS PKL", file=sys.stderr)
        return 2

    wd = Path(args[0])
    root = Path(args[1])
    max_rays = int(args[2])
    if max_rays <= 0:
        max_rays = 24000
    pkl = Path(args[3])

    _ensure_import_path(root)

    from pyAOBS.modeling.rayinvr.service import _wrapper_collect_workdir

    ok, msg, rays, obs, timing = _wrapper_collect_workdir(wd, max_rays=max_rays)
    out: dict = {
        "ok": ok,
        "rays": rays,
        "obs": obs,
        "err": msg,
        "timing": {
            **timing,
            "subprocess_total_s": timing.get("job_total_s", 0.0),
        },
    }

    try:
        pkl.write_bytes(pickle.dumps(out))
    except Exception:
        pass

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
