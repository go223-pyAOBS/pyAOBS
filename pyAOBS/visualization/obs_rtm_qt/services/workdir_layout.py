# -*- coding: utf-8 -*-
"""工区完整分层路径约定 + 旧布局迁移。

最终布局::

    meta/                 obs_rtm_project.json
    inputs/raw/           *.su, v.in, shots 备份
    inputs/shots/
    prep/geom/            shots_xz, obs_xz, offsets
    prep/vel/             tomo_vel, ss, rr
    prep/shots_mute/  prep/shots_proc/
    rtm_in/               vel.rsf, bath1d.rsf
    rtm_work/
    diag/  examples/  cache/
"""
from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from ..project import ObsRtmProject

# ---- 相对工区根的规范路径 ----
META_DIR = "meta"
PROJECT_JSON = os.path.join(META_DIR, "obs_rtm_project.json")

INPUTS_RAW = os.path.join("inputs", "raw")
SHOTS_DIR = os.path.join("inputs", "shots")
SHOTS_MUTE = os.path.join("prep", "shots_mute")
SHOTS_PROC = os.path.join("prep", "shots_proc")

GEOM_DIR = os.path.join("prep", "geom")
SHOTS_XZ = os.path.join(GEOM_DIR, "shots_xz.txt")
OBS_XZ = os.path.join(GEOM_DIR, "obs_xz.txt")
OFFSETS = os.path.join(GEOM_DIR, "offsets.txt")
BATH_X = os.path.join(GEOM_DIR, "bath_x.txt")

VEL_DIR = os.path.join("prep", "vel")
TOMO_VEL = os.path.join(VEL_DIR, "tomo_vel.rsf")
SS_RSF = os.path.join(VEL_DIR, "ss.rsf")
RR_RSF = os.path.join(VEL_DIR, "rr.rsf")

RTM_IN = "rtm_in"
VEL_RSF = os.path.join(RTM_IN, "vel.rsf")
BATH1D = os.path.join(RTM_IN, "bath1d.rsf")

DIAG_DIR = "diag"
SUMMARY = os.path.join(DIAG_DIR, "su_summary.txt")

CACHE_DIR = "cache"
SU_WORK = os.path.join(CACHE_DIR, "_su_work")
PREVIEW_CACHE = os.path.join(CACHE_DIR, ".obs_rtm_preview_cache")

LAYOUT_DIRS: Tuple[str, ...] = (
    META_DIR,
    INPUTS_RAW,
    SHOTS_DIR,
    SHOTS_MUTE,
    SHOTS_PROC,
    GEOM_DIR,
    VEL_DIR,
    RTM_IN,
    DIAG_DIR,
    "examples",
    CACHE_DIR,
    "rtm_work",
)


def apply_layout_defaults(project: ObsRtmProject) -> None:
    """把工程字段设为新布局相对路径（迁移后或新建时调用）。"""
    project.shots_dir = SHOTS_DIR
    project.shots_mute_dir = SHOTS_MUTE
    project.shots_proc_dir = SHOTS_PROC
    project.shots_xz = SHOTS_XZ
    project.obs_xz = OBS_XZ
    project.offsets_txt = OFFSETS
    project.summary = SUMMARY
    project.velocity.out_vel = VEL_RSF
    project.velocity.bath_path = BATH_X
    project.rtm.vel_rsf = VEL_RSF
    if not (getattr(project.rtm, "workdir", None) or "").strip():
        project.rtm.workdir = "rtm_work"


def ensure_layout_dirs(project: ObsRtmProject) -> None:
    if not project.workdir:
        raise ValueError("未设置工区目录 workdir")
    os.makedirs(project.workdir, exist_ok=True)
    for rel in LAYOUT_DIRS:
        os.makedirs(os.path.join(project.workdir, rel), exist_ok=True)
    rtm_sub = (getattr(project.rtm, "workdir", None) or "rtm_work").strip() or "rtm_work"
    os.makedirs(os.path.join(project.workdir, rtm_sub), exist_ok=True)


def project_json_path(workdir: str) -> str:
    return os.path.join(workdir, PROJECT_JSON)


def infer_workdir_from_json(json_path: str) -> str:
    """meta/obs_rtm_project.json → 工区根；旧根上 JSON → 其所在目录。"""
    d = os.path.dirname(os.path.abspath(json_path))
    if os.path.basename(d).lower() == META_DIR:
        return os.path.dirname(d)
    return d


def find_project_json(workdir: str) -> Optional[str]:
    for rel in (PROJECT_JSON, "obs_rtm_project.json"):
        p = os.path.join(workdir, rel)
        if os.path.isfile(p):
            return p
    return None


def resolve_rel(
    project: ObsRtmProject,
    preferred: str,
    *legacy: str,
    as_dir: bool = False,
) -> str:
    """返回存在的相对路径（优先新布局）；都不存在则返回 preferred。"""
    checks: Sequence[str] = (preferred,) + legacy
    for rel in checks:
        if not rel:
            continue
        p = project.path(rel)
        ok = os.path.isdir(p) if as_dir else os.path.isfile(p)
        if ok:
            return rel
    return preferred


def abs_path(project: ObsRtmProject, preferred: str, *legacy: str) -> str:
    rel = resolve_rel(project, preferred, *legacy)
    return project.path(rel)


def path_bath1d(project: ObsRtmProject) -> str:
    return abs_path(project, BATH1D, "bath1d.rsf")


def path_tomo_vel(project: ObsRtmProject) -> str:
    return abs_path(project, TOMO_VEL, "tomo_vel.rsf")


def path_ss(project: ObsRtmProject) -> str:
    return abs_path(project, SS_RSF, "ss.rsf")


def path_rr(project: ObsRtmProject) -> str:
    return abs_path(project, RR_RSF, "rr.rsf")


def path_vel_write(project: ObsRtmProject) -> str:
    """写出成像速度的目标路径（始终新布局）。"""
    name = (project.velocity.out_vel or VEL_RSF).strip() or VEL_RSF
    if name in ("vel.rsf", os.path.basename(VEL_RSF)):
        name = VEL_RSF
    return project.path(name) if not os.path.isabs(name) else name


def _fix_rsf_in(path: str) -> None:
    if not path.endswith(".rsf") or not os.path.isfile(path):
        return
    try:
        from ..scripts.su_to_shots import fix_rsf_in_abspath

        fix_rsf_in_abspath(path)
    except Exception:
        pass


def _move_path(src: str, dst: str, log: Optional[Callable[[str], None]]) -> bool:
    if not os.path.exists(src):
        return False
    if os.path.abspath(src) == os.path.abspath(dst):
        return False
    if os.path.isdir(dst):
        try:
            if not os.listdir(dst):
                os.rmdir(dst)
            else:
                return False
        except OSError:
            return False
    elif os.path.exists(dst):
        return False
    parent = os.path.dirname(dst)
    if parent:
        os.makedirs(parent, exist_ok=True)
    shutil.move(src, dst)
    if log:
        log("迁移: %s → %s" % (src, dst))
    return True


def _move_rsf_pair(wd: str, src_rel: str, dst_rel: str, log) -> None:
    src = os.path.join(wd, src_rel)
    dst = os.path.join(wd, dst_rel)
    if _move_path(src, dst, log):
        _move_path(src + "@", dst + "@", log)
        _fix_rsf_in(dst)


def _move_dir(wd: str, src_rel: str, dst_rel: str, log) -> None:
    src = os.path.join(wd, src_rel)
    dst = os.path.join(wd, dst_rel)
    if not os.path.isdir(src):
        return
    if os.path.isdir(dst) and os.listdir(dst):
        # 目标已有内容：逐文件并入
        for name in os.listdir(src):
            _move_path(os.path.join(src, name), os.path.join(dst, name), log)
        try:
            if not os.listdir(src):
                os.rmdir(src)
        except OSError:
            pass
        return
    # 目标不存在或为空目录：整树挪入
    _move_path(src, dst, log)


def needs_legacy_migrate(workdir: str) -> bool:
    if not workdir or not os.path.isdir(workdir):
        return False
    markers = (
        "shots",
        "shots_mute",
        "shots_proc",
        "shots_xz.txt",
        "obs_xz.txt",
        "offsets.txt",
        "vel.rsf",
        "bath1d.rsf",
        "tomo_vel.rsf",
        "obs_rtm_project.json",
        "_su_work",
        ".obs_rtm_preview_cache",
        "su_summary.txt",
        "rr.rsf",
        "ss.rsf",
    )
    return any(os.path.exists(os.path.join(workdir, m)) for m in markers)


def migrate_legacy_workdir(
    project: ObsRtmProject,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> List[str]:
    """将旧工区根散落文件迁入分层目录，并更新 project 字段。"""
    wd = project.workdir
    if not wd or not os.path.isdir(wd):
        return []
    ensure_layout_dirs(project)
    moved: List[str] = []

    def _log(msg: str) -> None:
        moved.append(msg)
        if log:
            log(msg)

    # 目录
    _move_dir(wd, "shots", SHOTS_DIR, _log)
    _move_dir(wd, "shots_mute", SHOTS_MUTE, _log)
    _move_dir(wd, "shots_proc", SHOTS_PROC, _log)
    _move_dir(wd, "shots_raw", INPUTS_RAW, _log)
    _move_dir(wd, "_su_work", SU_WORK, _log)
    _move_dir(wd, ".obs_rtm_preview_cache", PREVIEW_CACHE, _log)
    # 旧 diag 已在根上的检查文件：若仍在根则迁入 diag（已在 diag/ 的跳过）
    for name in (
        "geom_check.txt",
        "offset_sign_check.txt",
        "su_summary.txt",
        "trid_list.txt",
        "trid_components.txt",
        "obs_segy_geometry.txt",
    ):
        _move_path(os.path.join(wd, name), os.path.join(wd, DIAG_DIR, name), _log)

    # 几何
    for src, dst in (
        ("shots_xz.txt", SHOTS_XZ),
        ("obs_xz.txt", OBS_XZ),
        ("offsets.txt", OFFSETS),
        ("bath_x.txt", BATH_X),
    ):
        _move_path(os.path.join(wd, src), os.path.join(wd, dst), _log)

    # 速度
    for src, dst in (
        ("vel.rsf", VEL_RSF),
        ("bath1d.rsf", BATH1D),
        ("tomo_vel.rsf", TOMO_VEL),
        ("ss.rsf", SS_RSF),
        ("rr.rsf", RR_RSF),
    ):
        _move_rsf_pair(wd, src, dst, _log)

    # 工程 JSON
    old_json = os.path.join(wd, "obs_rtm_project.json")
    new_json = os.path.join(wd, PROJECT_JSON)
    _move_path(old_json, new_json, _log)

    # 常见原始输入 → inputs/raw（仅当目标不存在）
    for name in os.listdir(wd):
        low = name.lower()
        if low.endswith(".su") or name == "v.in":
            _move_path(
                os.path.join(wd, name),
                os.path.join(wd, INPUTS_RAW, name),
                _log,
            )

    # 更新字段
    apply_layout_defaults(project)
    relocate_project_paths(project)

    # 修复迁入 RSF 的 in=（shots / rtm_in / prep/vel）
    for root_rel in (SHOTS_DIR, SHOTS_MUTE, SHOTS_PROC, RTM_IN, VEL_DIR):
        root = project.path(root_rel)
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if name.endswith(".rsf") and not name.endswith(".rsf@"):
                _fix_rsf_in(os.path.join(root, name))

    return moved


def relocate_project_paths(project: ObsRtmProject) -> None:
    """把失效的绝对路径（跨机 /mnt、旧根路径）重定位到分层后的文件。"""
    wd = project.workdir
    if not wd:
        return

    def _relocate_abs(path: str) -> str:
        if not path:
            return path
        if os.path.isfile(path):
            return os.path.normpath(path)
        base = os.path.basename(str(path).replace("\\", "/"))
        if not base:
            return path
        for cand in (
            os.path.join(wd, INPUTS_RAW, base),
            os.path.join(wd, TOMO_VEL) if base == "tomo_vel.rsf" else "",
            os.path.join(wd, VEL_RSF) if base == "vel.rsf" else "",
            os.path.join(wd, BATH1D) if base == "bath1d.rsf" else "",
            os.path.join(wd, GEOM_DIR, base),
            os.path.join(wd, base),
        ):
            if cand and os.path.isfile(cand):
                return os.path.normpath(cand)
        return path

    project.su_path = _relocate_abs(project.su_path)
    project.tomo_vel = _relocate_abs(project.tomo_vel)
    if project.tomo_vel and os.path.basename(project.tomo_vel) == "tomo_vel.rsf":
        tp = project.path(TOMO_VEL)
        if os.path.isfile(tp):
            project.tomo_vel = tp
    project.velocity.tomo_path = _relocate_abs(
        getattr(project.velocity, "tomo_path", "") or ""
    )
    project.velocity.zelt_vin_path = _relocate_abs(
        getattr(project.velocity, "zelt_vin_path", "") or ""
    )


def _workdir_looks_populated(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    for rel in (META_DIR, "rtm_work", "inputs", "prep", "rtm_in", "shots"):
        if os.path.isdir(os.path.join(path, rel)):
            return True
    return bool(os.listdir(path))


def coerce_workdir(project: ObsRtmProject, json_path: Optional[str] = None) -> str:
    """把 JSON 里失效的 workdir（如 WSL /mnt/d/...）纠正为本地存在的路径。"""
    import re

    candidates: List[str] = []
    if json_path and os.path.isfile(json_path):
        candidates.append(infer_workdir_from_json(json_path))
    wd = (project.workdir or "").strip()
    if wd:
        candidates.append(wd)
        m = re.match(r"^/mnt/([a-zA-Z])/(.*)$", wd.replace("\\", "/"))
        if m:
            candidates.append(
                m.group(1).upper() + ":\\" + m.group(2).replace("/", "\\")
            )
        # 已是 \\mnt\\d\\... 的 normpath 残骸
        m2 = re.match(r"^[/\\\\]+mnt[/\\\\]+([a-zA-Z])[/\\\\]+(.*)$", wd)
        if m2:
            candidates.append(
                m2.group(1).upper() + ":\\" + m2.group(2).replace("/", "\\")
            )

    for c in candidates:
        c = os.path.normpath(c)
        if _workdir_looks_populated(c):
            project.workdir = c
            return c
    if not wd:
        raise ValueError("未设置工区目录 workdir")
    project.workdir = os.path.normpath(wd)
    return project.workdir


def prepare_workdir(
    project: ObsRtmProject,
    *,
    migrate: bool = True,
    log: Optional[Callable[[str], None]] = None,
    json_path: Optional[str] = None,
) -> None:
    """确保分层目录；可选迁移旧布局；字段对齐新路径。"""
    coerce_workdir(project, json_path=json_path)
    ensure_layout_dirs(project)
    if migrate and needs_legacy_migrate(project.workdir):
        migrate_legacy_workdir(project, log=log)
    else:
        apply_layout_defaults(project)
    relocate_project_paths(project)