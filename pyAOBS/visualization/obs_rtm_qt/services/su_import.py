# -*- coding: utf-8 -*-
"""封装 scripts/su_to_shots.py（子进程导入；工区根无脚本）。"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Callable, List, Optional

from ..project import ObsRtmProject
from .paths import script_path


def build_su_to_shots_cmd(project: ObsRtmProject) -> List[str]:
    from .workdir_layout import SU_WORK

    g = project.geometry
    script = script_path("su_to_shots.py")
    project.ensure_workdir()
    cmd = [
        sys.executable,
        script,
        "--su", project.su_path,
        "--endian", g.endian,
        "--group", g.group,
        "--geom", g.geom,
        "--xy-unit", g.xy_unit,
        "--line-axis", g.line_axis,
        "--outdir", project.path(project.shots_dir),
        "--shots-xz", project.path(project.shots_xz),
        "--obs-xz", project.path(project.obs_xz),
        "--offsets", project.path(project.offsets_txt),
        "--summary", project.path(project.summary),
        "--workdir", project.path(SU_WORK),
        "--zshot", str(g.zshot_km),
        "--zobs-mode", g.zobs_mode,
        "--zobs-const", str(g.zobs_const_km),
    ]
    if g.native:
        cmd.append("--native")
    if g.component:
        cmd.extend(["--component", g.component])
    if g.trid:
        cmd.extend(["--trid", g.trid])
    if g.geom == "offset":
        cmd.extend([
            "--obs-x", str(g.obs_x_km),
            "--offset-sign", str(g.offset_sign),
        ])
    return cmd


def run_su_to_shots(
    project: ObsRtmProject,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> int:
    """在 project.workdir 下运行；成功后写出 shots_xz / obs_xz / summary。"""
    if not project.su_path or not os.path.isfile(project.su_path):
        raise FileNotFoundError("SU 文件不存在: %s" % project.su_path)
    project.ensure_workdir()
    cmd = build_su_to_shots_cmd(project)
    # 几何 txt / shots 路径已由 CLI 指定；cwd=工区便于相对路径
    if log:
        log("$ " + " ".join(cmd))
    env = os.environ.copy()
    proc = subprocess.Popen(
        cmd,
        cwd=project.workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if log:
            log(line.rstrip())
    return int(proc.wait())
