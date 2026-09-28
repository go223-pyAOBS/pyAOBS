"""QC 结果图：蒙特卡洛平均剖面（非模态，pyqtgraph）。"""

from __future__ import annotations

from pathlib import Path


def open_mc_profile_dialog(profile_txt: str | Path, parent=None):
    path = Path(profile_txt)
    if not path.is_file():
        return None
    zs: list[float] = []
    vm: list[float] = []
    vs: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 3:
            continue
        zs.append(float(parts[0]))
        vm.append(float(parts[1]))
        vs.append(float(parts[2]))
    if not zs:
        return None

    from ..plots.inv_analysis_pg import show_mc_profile_window

    return show_mc_profile_window(
        zs,
        vm,
        vs,
        title="蒙特卡洛 — 平均速度剖面",
        save_dir=str(path.parent),
    )
