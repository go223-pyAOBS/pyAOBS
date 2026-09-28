# -*- coding: utf-8 -*-
"""
偏移作业：Madagascar scons（awefd2d）/ rtm_shot_loop，叠炮与水柱压制预览。
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..project import ObsRtmProject
from .paths import script_path
from .rsf_io import parse_rsf_header, read_rsf_slice_n3
from .velocity import read_vel_rsf, write_vel_rsf

OBS_STACK_MANIFEST = "img_stack_manifest.json"


def resolve_shot_dir(project: ObsRtmProject, use_proc: bool) -> str:
    if use_proc:
        d = project.path(project.shots_proc_dir)
        if os.path.isdir(d) and any(
            n.startswith("shot_") and n.endswith(".rsf") for n in os.listdir(d)
        ):
            return d
    return project.path(project.shots_dir)


def rtm_run_dir(project: ObsRtmProject) -> str:
    """偏移作业目录（相对工区，默认 rtm_work/）：SConstruct + 全部中间/结果 RSF。"""
    d = project.path(getattr(project.rtm, "workdir", None) or "rtm_work")
    os.makedirs(d, exist_ok=True)
    return d


def parse_shot_list(spec: str) -> List[int]:
    """
    解析自选炮号：\"0,5,10-12\" / \"0 5 10-12\" / \"1529;1600\"。
    支持 a-b 闭区间；去重且保持首次出现顺序。
    """
    import re

    text = (spec or "").strip()
    if not text:
        return []
    out: List[int] = []
    seen = set()
    for tok in re.split(r"[,;\s]+", text):
        tok = tok.strip()
        if not tok:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", tok)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            for i in range(a, b + 1):
                if i not in seen:
                    seen.add(i)
                    out.append(i)
            continue
        if not tok.isdigit():
            raise ValueError("无法识别的炮号片段: %r（示例: 0,5,10-12）" % tok)
        i = int(tok)
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def resolve_shot_indices(
    project: ObsRtmProject,
    *,
    n_shots: Optional[int] = None,
) -> List[int]:
    """
    作业实际炮号列表。
    优先 rtm.shot_list；否则 first_shot + max_shot（max_shot<=0 且未给 n_shots 时返回 []，
    由 SConstruct/loop 在运行时展开为「从 first 到末炮」）。
    """
    r = project.rtm
    spec = str(getattr(r, "shot_list", "") or "").strip()
    if spec:
        indices = parse_shot_list(spec)
    else:
        i0 = max(int(getattr(r, "first_shot", 0) or 0), 0)
        n = int(r.max_shot)
        if n > 0:
            indices = list(range(i0, i0 + n))
        elif n_shots is not None and n_shots >= 0:
            indices = list(range(i0, n_shots))
        else:
            return []
    if n_shots is not None and n_shots >= 0:
        indices = [i for i in indices if 0 <= i < n_shots]
    return indices


def shot_selection_mode(project: ObsRtmProject) -> str:
    """list | range"""
    if str(getattr(project.rtm, "shot_list", "") or "").strip():
        return "list"
    return "range"


def describe_shot_selection(project: ObsRtmProject, n_shots: Optional[int] = None) -> str:
    if shot_selection_mode(project) == "list":
        ids = resolve_shot_indices(project, n_shots=n_shots)
        if not ids:
            return "自选炮号（空/越界）"
        if len(ids) <= 8:
            return "自选: %s（%d 炮）" % (",".join(str(i) for i in ids), len(ids))
        return "自选: %s…%s（%d 炮）" % (ids[0], ids[-1], len(ids))
    r = project.rtm
    i0 = max(int(getattr(r, "first_shot", 0) or 0), 0)
    n = int(r.max_shot)
    if n == 1:
        return "连续: 炮 %d（1 炮）" % i0
    if n > 1:
        return "连续: 炮 %d–%d（%d 炮）" % (i0, i0 + n - 1, n)
    return "连续: 炮 %d–末炮" % i0


def describe_obs_source_plan(project: ObsRtmProject) -> str:
    """日志用：OBS 为源作业摘要。obs_xz 缺失时 n_obs=0，标明文件状态。"""
    from .geometry import load_xz_txt

    obs_path = project.path(project.obs_xz) if project.workdir else ""
    obs = load_xz_txt(obs_path) if obs_path else []
    n_obs = len(obs)
    spec = str(getattr(project.rtm, "obs_list", "") or "").strip()
    if spec:
        try:
            ids = parse_shot_list(spec)
        except ValueError:
            ids = []
        obs_txt = "OBS %s（%d 台）" % (
            ",".join(str(i) for i in ids[:12]) + ("…" if len(ids) > 12 else ""),
            len(ids),
        )
    elif n_obs == 0:
        obs_txt = "全部 OBS（0 台，obs_xz 缺失或空）"
    else:
        obs_txt = "全部 OBS（%d 台）" % n_obs
    return "OBS为源(互易) · %s · 检波=%s" % (obs_txt, describe_shot_selection(project))


def find_scons_executable() -> List[str]:
    """返回可直接传给 QProcess/subprocess 的 [program, *prefix_args]。"""
    for name in ("scons", "scons.bat"):
        p = shutil.which(name)
        if p:
            return [p]
    # 常见：python -m SCons
    return [sys.executable, "-m", "SCons"]


def build_scons_cmd(project: ObsRtmProject, sconstruct: str = "SConstruct_obs_rtm") -> List[str]:
    """构建到 img_lap.rsf（与 Flow 产出文件名一致；勿用无后缀的 img_lap）。"""
    cmd = find_scons_executable()
    cmd.extend(["-f", sconstruct, "img_lap.rsf"])
    # 勿包 stdbuf -oL：verb=y 时每时间步刷一行，管道反压会让 GUI 比终端慢很多。
    # 进度靠 SConstruct 的 print(>>>…) + 心跳扫产物文件。
    return cmd


def build_impulse_scons_cmd(
    project: ObsRtmProject, sconstruct: str = "SConstruct_impulse"
) -> List[str]:
    """脉冲 RTM：构建到 img_impulse_lap.rsf（cwd = rtm_work/impulse/）。"""
    cmd = find_scons_executable()
    cmd.extend(["-f", sconstruct, "img_impulse_lap.rsf"])
    return cmd


def _impulse_stamp(
    project: ObsRtmProject, *, iobs: int, ishot: int, it: int, amp: float
) -> dict:
    """脉冲作业指纹：变了才需要重跑 awefd2d。"""
    from .impulse_gather import impulse_nt_jsnap

    r = project.rtm
    g = project.grid
    vp = project.velocity
    nt, jsnap, _ns = impulse_nt_jsnap(project, it)
    vel = project.path(r.vel_rsf)
    if not os.path.isfile(vel):
        alt = project.path(project.velocity.out_vel)
        if os.path.isfile(alt):
            vel = alt
    from .workdir_layout import path_bath1d

    bath = path_bath1d(project)
    try:
        vel_m = int(os.path.getmtime(vel)) if os.path.isfile(vel) else 0
    except OSError:
        vel_m = 0
    try:
        bath_m = int(os.path.getmtime(bath)) if os.path.isfile(bath) else 0
    except OSError:
        bath_m = 0
    fm = 0.5 * (float(r.fmin) + float(r.fmax))
    if fm <= 0:
        fm = max(float(r.fmax), 6.0)
    return {
        "iobs": int(iobs),
        "ishot": int(ishot),
        "it": int(it),
        "amp": round(float(amp), 8),
        # 反传数据：与正传 wav 同 fm 的带限 Ricker（非单点 δ）
        "impulse_wavelet": "ricker",
        "nt": int(nt),
        "jsnap": int(jsnap),
        "nx": int(g.nx),
        "nz": int(g.nz),
        "ox": float(g.ox),
        "oz": float(g.oz),
        "dx": float(g.dx),
        "dz": float(g.dz),
        "dt": float(r.dt),
        "fmin": float(r.fmin),
        "fmax": float(r.fmax),
        "fm": float(fm),
        "nb": int(getattr(r, "nb", 40) or 40),
        "verb": bool(getattr(r, "awefd_verb", False)),
        "smooth_rect": int(getattr(vp, "smooth_rect", 0) or 0),
        "vel_path": os.path.abspath(vel) if vel else "",
        "vel_mtime": vel_m,
        "bath_mtime": bath_m,
    }


def _path_is_under(path: str, root: str) -> bool:
    try:
        ap = os.path.abspath(path)
        ar = os.path.abspath(root)
        if ap == ar:
            return True
        prefix = ar if ar.endswith(os.sep) else ar + os.sep
        # 也认 POSIX 前缀（WSL）
        ap_u, ar_u = ap.replace("\\", "/"), ar.replace("\\", "/")
        return ap.startswith(prefix) or ap_u.startswith(
            ar_u if ar_u.endswith("/") else ar_u + "/"
        )
    except Exception:
        return False


def _wipe_impulse_products(run: str, *, log: Optional[Callable[[str], None]] = None) -> int:
    """
    删 impulse/ 内产物及其 DATAPATH 体。

    绝不能跟 in= 删到工区根的 vel.rsf@（sfwindow 常把 vels 指回父目录速度体）。
    """
    if not run or not os.path.isdir(run):
        return 0
    removed = 0
    skipped_out = 0
    try:
        names = list(os.listdir(run))
    except OSError:
        return 0
    run_abs = os.path.abspath(run)

    def _rm_allowed(path: str) -> None:
        nonlocal removed, skipped_out
        if not path:
            return
        ap = os.path.abspath(path)
        ap_n = ap.replace("\\", "/")
        # 仅：impulse 目录内，或 DATAPATH 路径中带 /impulse/
        ok = _path_is_under(ap, run_abs) or ("/impulse/" in ap_n) or ap_n.rstrip(
            "/"
        ).endswith("/impulse")
        if not ok:
            skipped_out += 1
            return
        try:
            if os.path.isfile(ap) or os.path.islink(ap):
                os.remove(ap)
                removed += 1
            elif os.path.isdir(ap):
                shutil.rmtree(ap)
                removed += 1
        except OSError:
            pass

    for name in names:
        if not name.endswith(".rsf"):
            continue
        hdr = os.path.join(run, name)
        try:
            with open(hdr, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    s = line.strip()
                    if s.startswith("in="):
                        inp = s.split("=", 1)[1].strip().strip("\"'")
                        if inp:
                            _rm_allowed(inp)
                        break
        except OSError:
            pass
    for name in names:
        _rm_allowed(os.path.join(run, name))
    if log:
        msg = "脉冲目录已清理 %d 项（参数变化或强制重算）" % removed
        if skipped_out:
            msg += "；跳过 %d 个目录外 in=（保护 vel.rsf@ 等）" % skipped_out
        log(msg)
    return removed


def ensure_project_bath1d(
    project: ObsRtmProject,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> str:
    """
    保证 bath1d.rsf 数据体存在；缺失则按工区 bath/OBS 重写（不依赖已丢的 .rsf@）。
    """
    from .rsf_io import resolve_rsf_binary
    from .velocity import resolve_bath_1d, write_bath1d_rsf
    from .workdir_layout import BATH1D, path_bath1d

    path = path_bath1d(project)
    try:
        if os.path.isfile(path):
            resolve_rsf_binary(path)
            return path
    except RuntimeError:
        pass

    g = project.grid
    src = str(getattr(project.velocity, "vel_source", "file") or "file").strip().lower()
    # 与生成成像速度一致：一维重建 bath 时不用 v.in 地形
    bath, tag = resolve_bath_1d(
        project, g, log=log, prefer_zelt=(src != "builtin_1d")
    )
    path = project.path(BATH1D)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_bath1d_rsf(path, bath, g.ox, g.dx)
    if log:
        log("已重建 %s（原数据体丢失，按 %s 重写）" % (BATH1D, tag))
    return path


def ensure_project_vel_binary(
    project: ObsRtmProject,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> str:
    """
    保证工区 vel.rsf 数据体存在。

    若头文件 in= 指向的 .rsf@ 丢失，尝试从 rtm_work 的 DATAPATH 备份恢复。
    返回可用的 vel.rsf 头路径。
    """
    from .rsf_io import parse_rsf_header, resolve_rsf_binary

    r = project.rtm
    vel = project.path(r.vel_rsf)
    if not os.path.isfile(vel):
        alt = project.path(project.velocity.out_vel)
        if os.path.isfile(alt):
            vel = alt
    if not os.path.isfile(vel):
        raise FileNotFoundError("缺少 vel.rsf，请先在速度页生成成像速度")

    try:
        resolve_rsf_binary(vel)
        return vel
    except RuntimeError:
        pass

    # 候选备份：正式 RTM 曾 window 出的 vels 体
    candidates: List[str] = []
    work_vels = os.path.join(rtm_run_dir(project), "vels.rsf")
    if os.path.isfile(work_vels):
        try:
            meta = parse_rsf_header(work_vels)
            # 优先 history 里 /var/tmp/.../vels.rsf@；扫全文件找 in=
            with open(work_vels, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            for line in text.splitlines():
                s = line.strip().replace('in("', "in=").replace('")', "")
                if "in=" in s and ".rsf@" in s:
                    # in=/path 或 in("/path"
                    if "in=" in s:
                        p = s.split("in=", 1)[1].strip().strip("\"'")
                        if p.endswith(".rsf@") or p.endswith("@"):
                            candidates.append(p)
        except OSError:
            pass
    dp = (os.environ.get("DATAPATH") or "/var/tmp").rstrip("/") + "/"
    wd_name = os.path.basename(os.path.abspath(project.workdir or "")) or "obs"
    for rel in (
        "obs_rtm_qt/%s/rtm_work/vels.rsf@" % wd_name,
        "obs_rtm_qt/madagascar_obs_rtm/rtm_work/vels.rsf@",
        "madagascar_obs_rtm/rtm_work/vels.rsf@",
        "visualization/obs_rtm_qt/madagascar_obs_rtm/vels.rsf@",
    ):
        candidates.append(os.path.join(dp, rel).replace("\\", "/"))
    # 去重保序
    seen = set()
    uniq: List[str] = []
    for c in candidates:
        c = os.path.abspath(c) if not c.startswith("/var/") else c
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    # write_vel_rsf 约定：二进制为 path + "@"
    dest = os.path.abspath(vel) + "@"

    for src in uniq:
        if not os.path.isfile(src):
            continue
        try:
            shutil.copy2(src, dest)
            # 重写头 in=
            meta = parse_rsf_header(vel)
            with open(vel, "w", encoding="utf-8") as h:
                h.write("in=%s\n" % dest)
                for k in (
                    "n1",
                    "d1",
                    "o1",
                    "label1",
                    "unit1",
                    "n2",
                    "d2",
                    "o2",
                    "label2",
                    "unit2",
                    "data_format",
                    "esize",
                ):
                    if k in meta:
                        h.write("%s=%s\n" % (k, meta[k]))
            resolve_rsf_binary(vel)
            if log:
                log("已恢复丢失的速度体: %s ← %s" % (dest, src))
            return vel
        except (OSError, RuntimeError):
            continue

    raise RuntimeError(
        "vel.rsf 数据体（.rsf@）丢失，无法跑 awefd2d。"
        "请到速度页重新「生成速度」后再试。"
        "（原因：旧版清理 impulse 时误删了目录外的 in= 目标）"
    )


def prepare_impulse_workdir(
    project: ObsRtmProject,
    *,
    iobs: int,
    ishot: int,
    it: int,
    amp: float,
    force_clean: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[str, bool]:
    """
    准备 ``rtm_work/impulse/``（脉冲道集 + SConstruct_impulse）。

    返回 ``(run_dir, reused)``。``reused=True`` 表示参数未变且已有成像，
    与终端手跑一样 scons 会几乎瞬时（Nothing to be done）。
    """
    import json

    from .impulse_gather import impulse_run_dir
    from .sconstruct_gen import generate_sconstruct_impulse_rtm

    project.ensure_workdir()
    # 旧版 wipe 曾误删工区 vel.rsf@ / bath1d.rsf@
    ensure_project_vel_binary(project, log=log)
    ensure_project_bath1d(project, log=log)
    run = impulse_run_dir(project)
    os.makedirs(run, exist_ok=True)
    stamp = _impulse_stamp(
        project, iobs=iobs, ishot=ishot, it=it, amp=amp
    )
    stamp_path = os.path.join(run, "impulse_stamp.json")
    lap = os.path.join(run, "img_impulse_lap.rsf")
    old = None
    if os.path.isfile(stamp_path):
        try:
            with open(stamp_path, "r", encoding="utf-8") as f:
                old = json.load(f)
        except (OSError, ValueError, TypeError):
            old = None
    reuse = (
        not force_clean
        and isinstance(old, dict)
        and old == stamp
        and os.path.isfile(lap)
    )
    if log:
        log("准备脉冲 RTM：输出目录 %s" % run)
    if reuse:
        if log:
            log(
                "参数未变且已有 img_impulse_lap.rsf → 与终端一样目标已是最新。"
                "公平比墙钟请先:\n"
                "  cd %s && scons -c -f SConstruct_impulse && "
                "scons -f SConstruct_impulse img_impulse_lap.rsf" % run
            )
        return run, True

    if old is not None and old != stamp and log:
        log("脉冲参数已变，清理旧波场后重算…")
    elif force_clean and log:
        log("强制清理脉冲目录后重算…")
    _wipe_impulse_products(run, log=log)
    os.makedirs(run, exist_ok=True)

    generate_sconstruct_impulse_rtm(
        project, iobs=iobs, ishot=ishot, it=it, amp=amp, log=log
    )
    try:
        with open(stamp_path, "w", encoding="utf-8") as f:
            json.dump(stamp, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except OSError:
        pass
    if log:
        log(
            "手跑脉冲: cd %s && scons -f SConstruct_impulse img_impulse_lap.rsf"
            % run
        )
        log(
            "公平对比墙钟: cd 同上 && scons -c -f SConstruct_impulse "
            "&& time scons -f SConstruct_impulse img_impulse_lap.rsf"
        )
    return run, False


def build_rtm_loop_cmd(project: ObsRtmProject) -> List[str]:
    r = project.rtm
    g = project.grid
    shot_dir = resolve_shot_dir(project, r.use_shots_proc)
    vel = project.path(r.vel_rsf)
    if not os.path.isfile(vel):
        alt = project.path(project.velocity.out_vel)
        if os.path.isfile(alt):
            vel = alt
    cmd = [
        sys.executable,
        script_path("rtm_shot_loop.py"),
        "--obs", project.path(project.obs_xz),
        "--shots", project.path(project.shots_xz),
        "--shot-dir", shot_dir,
        "--workdir", project.path(r.workdir),
        "--vel", vel,
        "--ox", str(g.ox),
        "--dx", str(g.dx),
        "--oz", str(g.oz),
        "--dz", str(g.dz),
        "--nx", str(g.nx),
        "--nz", str(g.nz),
        "--nt", str(r.nt),
        "--dt", str(r.dt),
        "--fmin", str(r.fmin),
        "--fmax", str(r.fmax),
    ]
    spec = str(getattr(r, "shot_list", "") or "").strip()
    if spec:
        # 校验格式；loop 脚本用 --shot-list
        parse_shot_list(spec)
        cmd.extend(["--shot-list", spec])
    else:
        first = max(int(getattr(r, "first_shot", 0) or 0), 0)
        if first > 0:
            cmd.extend(["--first-shot", str(first)])
        if r.max_shot > 0:
            cmd.extend(["--max-shot", str(r.max_shot)])
    if r.rtm_bin.strip():
        cmd.extend(["--rtm-bin", r.rtm_bin.strip()])
    dry = r.dry_run or (getattr(r, "engine", "") == "dry_run")
    if dry:
        cmd.append("--dry-run")
    return cmd


def build_rtm_run_cmd(project: ObsRtmProject) -> Tuple[str, List[str]]:
    """
    按 engine 返回 (mode, cmd)。
    mode: madagascar | loop
    """
    eng = (getattr(project.rtm, "engine", None) or "madagascar").lower()
    if eng in ("madagascar", "scons", "awefd2d"):
        sc = os.path.join(rtm_run_dir(project), "SConstruct_obs_rtm")
        if not os.path.isfile(sc):
            from .sconstruct_gen import generate_sconstruct_obs_rtm

            generate_sconstruct_obs_rtm(project)
        return "madagascar", build_scons_cmd(project)
    return "loop", build_rtm_loop_cmd(project)


def run_rtm_loop(
    project: ObsRtmProject,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> int:
    """同步跑 rtm_shot_loop（后台线程调用）。"""
    import subprocess

    project.ensure_workdir()
    os.makedirs(project.path(project.rtm.workdir), exist_ok=True)
    # 确保 obs/shots 在工区
    for name in (project.obs_xz, project.shots_xz):
        p = project.path(name)
        if not os.path.isfile(p):
            raise FileNotFoundError("缺少 %s，请先完成数据/几何" % p)
    cmd = build_rtm_loop_cmd(project)
    if log:
        log("$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=project.workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if log:
            log(line.rstrip())
    return int(proc.wait())


def list_shot_images(workdir: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(workdir, "img_*.npy")))
    return [p for p in paths if "stack" not in os.path.basename(p)]


def list_madagascar_img_shots(workdir: str) -> List[int]:
    """目录内成像：优先 img_obs_NNN.rsf；否则兼容旧 img_NNN.rsf。"""
    import re

    out_obs: List[int] = []
    out_shot: List[int] = []
    if not workdir or not os.path.isdir(workdir):
        return out_obs
    skip = {"img_stack", "img_solid", "img_lap"}
    for name in os.listdir(workdir):
        if not name.startswith("img_") or not name.endswith(".rsf"):
            continue
        if name.endswith(".rsf@"):
            continue
        stem = name[: -len(".rsf")]
        if stem in skip:
            continue
        m_obs = re.match(r"img_obs_(\d+)$", stem)
        if m_obs:
            out_obs.append(int(m_obs.group(1)))
            continue
        m = re.match(r"img_(\d+)$", stem)
        if m:
            out_shot.append(int(m.group(1)))
    if out_obs:
        return sorted(out_obs)
    return sorted(out_shot)


def list_rtm_img_shots(project: ObsRtmProject) -> List[int]:
    """优先 rtm_work/，兼容旧工区根目录残留。"""
    shots = list_madagascar_img_shots(rtm_run_dir(project))
    if shots:
        return shots
    return list_madagascar_img_shots(project.workdir)


def list_img_obs_rsf(project: ObsRtmProject) -> List[Tuple[int, str]]:
    """盘上全部单台互易像 ``(obs_id, abs_path)``，按编号排序。"""
    out: List[Tuple[int, str]] = []
    seen = set()
    for base in (rtm_run_dir(project), project.workdir):
        if not base or not os.path.isdir(base):
            continue
        try:
            names = os.listdir(base)
        except OSError:
            continue
        for name in names:
            m = re.match(r"img_obs_(\d+)\.rsf$", name, re.I)
            if not m or name.endswith(".rsf@"):
                continue
            tid = int(m.group(1))
            if tid in seen:
                continue
            path = os.path.join(base, name)
            if os.path.isfile(path):
                seen.add(tid)
                out.append((tid, path))
    out.sort(key=lambda x: x[0])
    return out


def obs_stack_manifest_path(project: ObsRtmProject) -> str:
    return os.path.join(rtm_run_dir(project), OBS_STACK_MANIFEST)


def _file_mtime(path: str) -> float:
    try:
        return float(os.path.getmtime(path))
    except OSError:
        return 0.0


def _obs_stack_fingerprint(project: ObsRtmProject) -> Dict[str, Any]:
    r = project.rtm
    vp = project.velocity
    vel = ""
    for name in (
        getattr(r, "vel_rsf", None),
        getattr(vp, "out_vel", None),
        "rtm_in/vel.rsf",
        "vel.rsf",
    ):
        if not name:
            continue
        p = name if os.path.isabs(str(name)) else project.path(str(name))
        if os.path.isfile(p):
            vel = os.path.normpath(os.path.abspath(p))
            break
    return {
        "vel_rsf": vel,
        "vel_mtime": _file_mtime(vel) if vel else 0.0,
        "fmin": float(getattr(r, "fmin", 0) or 0),
        "fmax": float(getattr(r, "fmax", 0) or 0),
        "nt": int(getattr(r, "nt", 0) or 0),
        "dt": float(getattr(r, "dt", 0) or 0),
        "jsnap": int(getattr(r, "jsnap", 0) or 0),
        "use_shots_proc": bool(getattr(r, "use_shots_proc", True)),
    }


def read_obs_stack_manifest(project: ObsRtmProject) -> Optional[dict]:
    path = obs_stack_manifest_path(project)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def write_obs_stack_manifest(
    project: ObsRtmProject,
    *,
    obs_ids: List[int],
    file_mtimes: Dict[str, float],
) -> str:
    run = rtm_run_dir(project)
    path = obs_stack_manifest_path(project)
    data = {
        "version": 1,
        "obs_ids": [int(i) for i in obs_ids],
        "files": {
            str(int(i)): {
                "name": "img_obs_%03d.rsf" % int(i),
                "mtime": float(file_mtimes.get(str(int(i)), 0.0)),
            }
            for i in obs_ids
        },
        "outputs": {
            "img_stack": _file_mtime(os.path.join(run, "img_stack.rsf")),
            "img_solid": _file_mtime(os.path.join(run, "img_solid.rsf")),
            "img_lap": _file_mtime(os.path.join(run, "img_lap.rsf")),
        },
        "fingerprint": _obs_stack_fingerprint(project),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def obs_stack_staleness(project: ObsRtmProject) -> Tuple[bool, str]:
    """
    叠后是否相对盘上 ``img_obs_*`` 过期。

    返回 ``(stale, reason)``；无单台像时 stale=False（由预览另报缺文件）。
    """
    items = list_img_obs_rsf(project)
    if not items:
        return False, ""
    ids = [i for i, _ in items]
    man = read_obs_stack_manifest(project)
    run = rtm_run_dir(project)
    lap = os.path.join(run, "img_lap.rsf")
    if not os.path.isfile(lap) and not os.path.isfile(
        os.path.join(run, "img_stack.rsf")
    ):
        return True, "尚无 img_lap / img_stack，请叠加已有 OBS 像"
    if man is None:
        return True, "无叠后清单（img_stack_manifest.json），建议重新叠加"
    man_ids = [int(x) for x in (man.get("obs_ids") or [])]
    if sorted(man_ids) != sorted(ids):
        return True, "盘上 OBS 像集合已变（清单 %s → 现有 %s）" % (
            man_ids,
            ids,
        )
    files = man.get("files") or {}
    for tid, path in items:
        key = str(int(tid))
        old = float((files.get(key) or {}).get("mtime") or 0.0)
        now = _file_mtime(path)
        if abs(old - now) > 1e-3:
            return True, "img_obs_%03d 已更新，叠后过期" % int(tid)
    return False, ""


def _image_laplacian(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float64)
    try:
        from scipy import ndimage

        return np.ascontiguousarray(ndimage.laplace(a), dtype=np.float32)
    except Exception:
        z = np.zeros_like(a, dtype=np.float64)
        z[1:-1, 1:-1] = (
            a[:-2, 1:-1]
            + a[2:, 1:-1]
            + a[1:-1, :-2]
            + a[1:-1, 2:]
            - 4.0 * a[1:-1, 1:-1]
        )
        return np.ascontiguousarray(z, dtype=np.float32)


def rebuild_obs_image_stack(
    project: ObsRtmProject,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> str:
    """
    扫盘叠加全部 ``img_obs_*.rsf`` → img_stack / img_solid / img_lap + manifest。

    与本轮 ``obs_list`` 无关，支持分批跑多台 OBS 后汇总。
    返回 ``img_lap.rsf`` 路径。
    """
    items = list_img_obs_rsf(project)
    if not items:
        raise RuntimeError(
            "rtm_work/ 下无 img_obs_NNN.rsf。请先对至少一台 OBS 跑完互易 RTM。"
        )
    run = rtm_run_dir(project)
    acc = None
    meta0 = None
    mtimes: Dict[str, float] = {}
    for tid, path in items:
        img, meta = read_vel_rsf(path)
        img = np.asarray(img, dtype=np.float64)
        if acc is None:
            acc = img.copy()
            meta0 = meta
        else:
            if img.shape != acc.shape:
                raise RuntimeError(
                    "img_obs_%03d 网格 %s 与其它像 %s 不一致，无法叠加"
                    % (tid, img.shape, acc.shape)
                )
            acc += img
        mtimes[str(int(tid))] = _file_mtime(path)
        if log:
            log("stack + img_obs_%03d  shape=%s" % (int(tid), img.shape))
    assert acc is not None and meta0 is not None
    from ..project import GridParams

    g = project.grid
    grid = GridParams(
        ox=float(meta0.get("o2", g.ox)),
        dx=float(meta0.get("d2", g.dx)),
        nx=int(acc.shape[1]),
        oz=float(meta0.get("o1", g.oz)),
        dz=float(meta0.get("d1", g.dz)),
        nz=int(acc.shape[0]),
    )
    stack_f = np.ascontiguousarray(acc, dtype=np.float32)
    stack_path = os.path.join(run, "img_stack.rsf")
    write_vel_rsf(stack_path, stack_f, grid)
    solid = mute_water_column(stack_f, project)
    solid_path = os.path.join(run, "img_solid.rsf")
    write_vel_rsf(solid_path, solid, grid)
    lap = _image_laplacian(solid)
    lap_path = os.path.join(run, "img_lap.rsf")
    write_vel_rsf(lap_path, lap, grid)
    npy_path = os.path.join(run, "img_stack.npy")
    np.save(npy_path, lap)
    ids = [int(i) for i, _ in items]
    man = write_obs_stack_manifest(project, obs_ids=ids, file_mtimes=mtimes)
    if log:
        log(
            "叠全部 OBS 像完成: n=%d ids=%s → img_lap.rsf + %s"
            % (len(ids), ids, os.path.basename(man))
        )
    return lap_path


def format_obs_stack_label(project: ObsRtmProject) -> str:
    """成像源下拉「叠后」文案。"""
    items = list_img_obs_rsf(project)
    if not items:
        return "叠后 (img_lap / stack)"
    ids = [i for i, _ in items]
    stale, _why = obs_stack_staleness(project)
    if len(ids) <= 8:
        id_txt = ",".join(str(i) for i in ids)
    else:
        id_txt = "%s…%s" % (ids[0], ids[-1])
    base = "叠后 (OBS %s · %d台)" % (id_txt, len(ids))
    return base + (" · 过期" if stale else "")


def sync_obs_stack_manifest(
    project: ObsRtmProject,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """scons 叠后已生成时，仅根据盘上 img_obs_* 刷新清单。"""
    items = list_img_obs_rsf(project)
    if not items:
        return None
    mtimes = {str(int(i)): _file_mtime(p) for i, p in items}
    path = write_obs_stack_manifest(
        project,
        obs_ids=[int(i) for i, _ in items],
        file_mtimes=mtimes,
    )
    if log:
        log(
            "叠后清单已更新: OBS %s → %s"
            % ([int(i) for i, _ in items], os.path.basename(path))
        )
    return path


def list_rtm_wfl_ids(project: ObsRtmProject, kind: str = "wflr") -> List[int]:
    """扫描各目录波场编号（不去重目录）；兼容旧调用。"""
    return sorted({tid for _scope, tid in list_rtm_wfl_entries(project, kind)})


def list_rtm_wfl_entries(
    project: ObsRtmProject, kind: str = "wflr"
) -> List[Tuple[str, int]]:
    """
    扫描波场条目 ``(scope, tag)``。

    scope: ``impulse`` | ``formal``（rtm_work 根或工区根）。
    同编号可同时存在两条，供 UI 区分。
    """
    import re

    from .impulse_gather import impulse_run_dir

    prefix = "wflr_" if str(kind).lower().startswith("wflr") else "wfls_"
    out: List[Tuple[str, int]] = []
    seen = set()

    def _scan(base: str, scope: str) -> None:
        if not base or not os.path.isdir(base):
            return
        try:
            names = os.listdir(base)
        except OSError:
            return
        for name in names:
            if not name.endswith(".rsf") or name.endswith(".rsf@"):
                continue
            m = re.match(r"%s(\d+)\.rsf$" % prefix, name, re.I)
            if not m:
                continue
            tid = int(m.group(1))
            key = (scope, tid)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)

    _scan(impulse_run_dir(project), "impulse")
    _scan(rtm_run_dir(project), "formal")
    # 工区根旧散落：仅当 formal 尚无该编号时记入
    wd = project.workdir
    if wd and os.path.isdir(wd) and os.path.abspath(wd) != os.path.abspath(
        rtm_run_dir(project)
    ):
        try:
            for name in os.listdir(wd):
                if not name.endswith(".rsf") or name.endswith(".rsf@"):
                    continue
                m = re.match(r"%s(\d+)\.rsf$" % prefix, name, re.I)
                if not m:
                    continue
                tid = int(m.group(1))
                if ("formal", tid) in seen:
                    continue
                seen.add(("formal", tid))
                out.append(("formal", tid))
        except OSError:
            pass
    out.sort(key=lambda x: (0 if x[0] == "impulse" else 1, x[1]))
    return out


def resolve_wfl_rsf(
    project: ObsRtmProject,
    kind: str,
    tag_id: int,
    *,
    scope: Optional[str] = None,
    prefer_impulse: bool = False,
) -> Optional[str]:
    """
    返回 wflr_NNN.rsf / wfls_NNN.rsf。

    ``scope='impulse'|'formal'`` 时只查对应目录；
    未指定时：``prefer_impulse`` 决定优先顺序（默认正式优先）。
    """
    from .impulse_gather import impulse_run_dir

    prefix = "wflr" if str(kind).lower().startswith("wflr") else "wfls"
    name = "%s_%03d.rsf" % (prefix, int(tag_id))
    imp = impulse_run_dir(project)
    run = rtm_run_dir(project)
    sc = str(scope or "").strip().lower()
    if sc == "impulse":
        bases = (imp,)
    elif sc == "formal":
        bases = (run, project.workdir)
    elif prefer_impulse:
        bases = (imp, run, project.workdir)
    else:
        bases = (run, project.workdir, imp)
    for base in bases:
        if not base:
            continue
        p = os.path.join(base, name)
        if os.path.isfile(p):
            return p
    return None


def wfl_n3(path: str) -> int:
    """波场快照张数 n3。"""
    meta = parse_rsf_header(path)
    return max(int(meta.get("n3", 1)), 1)


def load_wfl_for_preview(
    project: ObsRtmProject,
    *,
    kind: str = "wflr",
    tag_id: int = 0,
    frame: Optional[int] = None,
) -> Tuple[np.ndarray, str, dict]:
    """
    预览一张波场快照。

    返回 (img[nz,nx], title, meta)；``frame`` 为 None/<0 时取中间帧。
    """
    path = resolve_wfl_rsf(project, kind, tag_id)
    if not path:
        prefix = "wflr" if str(kind).lower().startswith("wflr") else "wfls"
        raise FileNotFoundError(
            "无 %s_%03d.rsf（请先跑完 awefd2d；结果在 rtm_work/）"
            % (prefix, int(tag_id))
        )
    img, meta = read_rsf_slice_n3(path, frame)
    i3 = int(meta["i3"])
    n3 = int(meta["n3"])
    t = float(meta["t"])
    prefix = "wflr" if str(kind).lower().startswith("wflr") else "wfls"
    kind_cn = "反传" if prefix == "wflr" else "正传"
    title = "%s波场 %s_%03d · 帧 %d/%d · t≈%.3fs" % (
        kind_cn,
        prefix,
        int(tag_id),
        i3,
        n3,
        t,
    )
    return img, title, meta


def _format_shot_span(shots: List[int]) -> str:
    if not shots:
        return "炮号未知"
    if len(shots) == 1:
        return "炮 %d（单炮）" % shots[0]
    if shots == list(range(shots[0], shots[-1] + 1)):
        return "炮 %d–%d（共 %d 炮）" % (shots[0], shots[-1], len(shots))
    if len(shots) <= 6:
        return "炮 %s（共 %d 炮）" % (",".join(str(s) for s in shots), len(shots))
    return "炮 %s…%s（共 %d 炮）" % (shots[0], shots[-1], len(shots))


def describe_rtm_image_shots(project: ObsRtmProject) -> Tuple[List[int], str]:
    """
    推断当前成像对应哪些炮/OBS。
    优先看 rtm_work img_obs_NNN / img_NNN.rsf；否则用 shot_list / first+max。
    """
    shots = list_rtm_img_shots(project)
    if shots:
        run = rtm_run_dir(project)
        is_obs = any(
            os.path.isfile(os.path.join(run, "img_obs_%03d.rsf" % i))
            or os.path.isfile(os.path.join(project.workdir, "img_obs_%03d.rsf" % i))
            for i in shots
        )
        if is_obs:
            if len(shots) == 1:
                return shots, "OBS %d（单台互易像）" % shots[0]
            return shots, "OBS %s（共 %d 台）" % (
                ",".join(str(s) for s in shots[:8]) + ("…" if len(shots) > 8 else ""),
                len(shots),
            )
        return shots, _format_shot_span(shots)
    if shot_selection_mode(project) == "list":
        seq = resolve_shot_indices(project)
        if seq:
            return seq, _format_shot_span(seq) + " [按自选炮号推断]"
        return [], "自选炮号为空 [按作业参数推断]"
    r = project.rtm
    i0 = max(int(getattr(r, "first_shot", 0) or 0), 0)
    n = int(r.max_shot)
    if n == 1:
        return [i0], _format_shot_span([i0])
    if n > 1:
        seq = list(range(i0, i0 + n))
        return seq, _format_shot_span(seq) + " [按作业参数推断]"
    return [], "全部炮（max_shot=-1）[按作业参数推断]"


def stack_shot_images(
    workdir: str,
    *,
    out_name: str = "img_stack.npy",
    log: Optional[Callable[[str], None]] = None,
) -> str:
    paths = list_shot_images(workdir)
    if not paths:
        raise RuntimeError("rtm_work 下无 img_###.npy（需 --rtm-bin 产出或自行放入）")
    acc = None
    for p in paths:
        a = np.load(p)
        acc = a.astype(np.float64) if acc is None else acc + a.astype(np.float64)
        if log:
            log("stack + %s" % os.path.basename(p))
    assert acc is not None
    out = os.path.join(workdir, out_name)
    np.save(out, acc.astype(np.float32))
    if log:
        log("wrote %s shape=%s" % (out, acc.shape))
    return out


def mute_water_column(
    img: np.ndarray,
    project: ObsRtmProject,
    bath_1d: Optional[np.ndarray] = None,
) -> np.ndarray:
    """z < bath(x) 置零（对齐 SConstruct img_solid）。"""
    g = project.grid
    out = np.asarray(img, dtype=np.float32).copy()
    if out.ndim != 2:
        raise ValueError("成像须为 2D (nz, nx)")
    nz, nx = out.shape
    if bath_1d is None:
        from .velocity import resolve_bath_1d

        bath_1d, _tag = resolve_bath_1d(project, g)
    if len(bath_1d) != nx:
        x_img = g.ox + np.arange(nx) * g.dx
        x_b = g.ox + np.arange(len(bath_1d)) * g.dx
        bath_1d = np.interp(x_img, x_b, np.asarray(bath_1d, float))
    z = g.oz + np.arange(nz) * g.dz
    for ix in range(nx):
        out[z < float(bath_1d[ix]), ix] = 0.0
    return out


def load_image_for_preview(
    project: ObsRtmProject,
    *,
    prefer_stack: bool = True,
    mute_water: bool = True,
    shot_id: Optional[int] = None,
) -> Tuple[np.ndarray, str]:
    """返回 (img, title)。

    ``shot_id`` 给定时优先 ``img_obs_NNN.rsf``（OBS 互易像），否则 ``img_NNN.rsf``；
    否则优先叠后（img_lap / stack），仅当作业只产出 1 张像时回退该像。
    """
    run = rtm_run_dir(project)
    root = project.workdir

    def _mute(img: np.ndarray, title: str) -> Tuple[np.ndarray, str]:
        img = np.asarray(img, dtype=np.float32)
        if mute_water:
            img = mute_water_column(img, project)
            title += " [water muted]"
        return img, title

    # 显式单像预览（OBS 互易像优先）
    if shot_id is not None:
        sid = int(shot_id)
        tag = "%03d" % sid
        for base in (run, root):
            rsf_obs = os.path.join(base, "img_obs_%s.rsf" % tag)
            if os.path.isfile(rsf_obs):
                img, _ = read_vel_rsf(rsf_obs)
                title = "OBS 像 · OBS %d · %s" % (
                    sid,
                    os.path.relpath(rsf_obs, root),
                )
                return _mute(img, title)
            rsf = os.path.join(base, "img_%s.rsf" % tag)
            if os.path.isfile(rsf):
                img, _ = read_vel_rsf(rsf)
                title = "单炮像 · 炮 %d · %s" % (
                    sid,
                    os.path.relpath(rsf, root),
                )
                return _mute(img, title)
            npy = os.path.join(base, "img_%s.npy" % tag)
            if os.path.isfile(npy):
                img = np.load(npy)
                title = "单炮像 · 炮 %d · %s" % (
                    sid,
                    os.path.relpath(npy, root),
                )
                return _mute(img, title)
        raise FileNotFoundError(
            "无成像 img_obs_%s.rsf / img_%s.rsf（请先跑 RTM，结果在 rtm_work/）"
            % (tag, tag)
        )

    shot_ids, shot_txt = describe_rtm_image_shots(project)

    # 仅一张像产出时：直接显示（OBS 互易像优先）
    if len(shot_ids) == 1:
        tag = "%03d" % shot_ids[0]
        for base in (run, root):
            rsf_obs = os.path.join(base, "img_obs_%s.rsf" % tag)
            if os.path.isfile(rsf_obs):
                img, _ = read_vel_rsf(rsf_obs)
                title = "成像 · %s · %s" % (
                    shot_txt,
                    os.path.relpath(rsf_obs, root),
                )
                return _mute(img, title)
            rsf = os.path.join(base, "img_%s.rsf" % tag)
            if os.path.isfile(rsf):
                img, _ = read_vel_rsf(rsf)
                title = "成像 · %s · %s" % (shot_txt, os.path.relpath(rsf, root))
                return _mute(img, title)

    stack = os.path.join(run, "img_stack.npy")
    if prefer_stack and not os.path.isfile(stack):
        try_export_img_lap_npy(project)
    if prefer_stack and os.path.isfile(stack):
        img = np.load(stack)
        src = "rtm_work/img_stack.npy"
        for base in (run, root):
            if os.path.isfile(os.path.join(base, "img_lap.rsf")):
                src = "rtm_work/img_lap.rsf→npy" if base == run else "img_lap.rsf→npy"
                break
        title = "成像叠后 · %s · %s" % (shot_txt, src)
    else:
        paths = list_shot_images(run)
        if not paths:
            exported = try_export_img_lap_npy(project)
            if exported and os.path.isfile(exported):
                img = np.load(exported)
                title = "成像叠后 · %s · %s" % (
                    shot_txt,
                    os.path.relpath(exported, root),
                )
            else:
                for name in ("img_lap.rsf", "img_stack.rsf", "img_solid.rsf"):
                    found = None
                    for base in (run, root):
                        p = os.path.join(base, name)
                        if os.path.isfile(p):
                            found = p
                            break
                    if found:
                        img, _ = read_vel_rsf(found)
                        title = "成像叠后 · %s · %s" % (
                            shot_txt,
                            os.path.relpath(found, root),
                        )
                        break
                else:
                    vel = project.path(project.rtm.vel_rsf)
                    if not os.path.isfile(vel):
                        vel = project.path(project.velocity.out_vel)
                    if os.path.isfile(vel):
                        v, _ = read_vel_rsf(vel)
                        return np.zeros_like(v), "(无成像 — 显示零像占位)"
                    raise FileNotFoundError(
                        "无成像文件；请先跑 Madagascar RTM（结果在 rtm_work/）"
                    )
        else:
            img = None
            for p in paths:
                a = np.load(p)
                img = a if img is None else img + a
            title = "成像叠加 · %s · %d 个 npy" % (shot_txt, len(paths))
    return _mute(img, title)


# SConstruct 中间/结果前缀（不含 vel/rr/shots 等输入）
_RTM_INTERMEDIATE_PREFIXES = (
    "raw_",
    "bp_",
    "mut_",
    "mut_t_",
    "datf_",
    "datb_",
    "dat_adj_",
    "wfls_",
    "wflr_",
    "img_",
    "xs_",
    "zs_",
    "ss_",
)
_RTM_INTERMEDIATE_EXACT = (
    "wav",
    "vels",
    "den",
    "bath2d",
    "img_stack",
    "img_solid",
    "img_lap",
)


def _stem_of_rsf_name(name: str) -> str:
    if name.endswith(".rsf@"):
        return name[: -len(".rsf@")]
    base, ext = os.path.splitext(name)
    if ext in (".rsf", ".hh", ".npy"):
        return base
    return name


# 输入/速度产物 stem：清理工区根旧散落文件时跳过；rtm_work 内同名可清
_ROOT_INPUT_STEMS = frozenset(
    {"vel", "vels", "rr", "bath1d", "ss", "vels_user", "den", "tomo_vel"}
)


def _is_rtm_product_name(name: str, *, in_rtm_work: bool = False) -> bool:
    stem = _stem_of_rsf_name(name)
    if stem.startswith("shot_"):
        return False
    if not in_rtm_work and stem in _ROOT_INPUT_STEMS:
        return False
    if stem in _RTM_INTERMEDIATE_EXACT:
        return True
    if any(stem.startswith(p) for p in _RTM_INTERMEDIATE_PREFIXES):
        return True
    if name in (
        "SConstruct_obs_rtm",
        ".sconsign.dblite",
        "SConstruct_obs_rtm.sconsign.dblite",
    ):
        return True
    if name.endswith(".sconsign.dblite") or name.startswith(".sconsign"):
        return True
    return False


def clean_rtm_intermediates(
    project: ObsRtmProject,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> int:
    """
    清理偏移输出目录 rtm_work/，并顺带清工区根上旧版散落的中间文件。
    保留：rtm_in/、prep/、inputs/、meta/、diag/ 等分层输入。
    """
    wd = project.workdir
    if not wd or not os.path.isdir(wd):
        return 0
    removed = 0
    protect_dirs = frozenset(
        {
            "rtm_in",
            "prep",
            "inputs",
            "meta",
            "diag",
            "examples",
            "cache",
            "shots",
            "shots_mute",
            "shots_proc",
        }
    )

    def _rm_file(path: str) -> None:
        nonlocal removed
        try:
            if os.path.isfile(path):
                os.remove(path)
                removed += 1
        except OSError:
            pass

    def _wipe_dir(root: str, *, aggressive: bool) -> None:
        if not root or not os.path.isdir(root):
            return
        try:
            names = os.listdir(root)
        except OSError:
            return
        for name in names:
            path = os.path.join(root, name)
            if not os.path.isfile(path):
                continue
            if name in ("README.txt", "rtm_gui_grid.txt", "WORKDIR_LAYOUT.txt"):
                continue
            if aggressive:
                low = name.lower()
                if (
                    low.endswith((".rsf", ".rsf@", ".npy", ".hh"))
                    or name.startswith("SConstruct")
                    or "sconsign" in low
                    or _is_rtm_product_name(name)
                ):
                    _rm_file(path)
            elif _is_rtm_product_name(name):
                _rm_file(path)

    run = rtm_run_dir(project)
    _wipe_dir(run, aggressive=True)
    # 兼容：旧版曾把 wav/wfls/img 写在工区根（勿进 protect 子目录）
    try:
        for name in os.listdir(wd):
            if name in protect_dirs or name.startswith("."):
                continue
            path = os.path.join(wd, name)
            if os.path.isfile(path) and _is_rtm_product_name(name):
                _rm_file(path)
    except OSError:
        pass
    _rm_file(os.path.join(wd, "SConstruct_obs_rtm"))

    if log:
        log(
            "已清理 RTM 中间文件 %d 个（输出在 rtm_work/；保留 rtm_in/prep/inputs）"
            % removed
        )
    return removed


def _write_rtm_layout_readme(project: ObsRtmProject, run_dir: str) -> str:
    """在 rtm_work/ 写简短目录说明，方便用户辨认输入/输出。"""
    path = os.path.join(run_dir, "README.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "OBS RTM 输出目录（本文件夹）\n"
            "========================\n"
            "本目录由 GUI 生成 SConstruct，并写入全部中间/成像结果：\n"
            "  SConstruct_obs_rtm, wav.rsf, wfls_*.rsf, img_obs_*.rsf, img_lap.rsf …\n"
            "互易：sou=OBS, rec=炮点\n"
            "\n"
            "输入（相对本目录）：\n"
            "  ../rtm_in/vel.rsf  ../rtm_in/bath1d.rsf\n"
            "  ../inputs/shots/ 或 ../prep/shots_proc/\n"
            "  ../prep/geom/shots_xz.txt  ../prep/geom/obs_xz.txt\n"
            "说明见 ../WORKDIR_LAYOUT.txt\n"
            "\n"
            "手跑（在本目录，OBS 为源互易）：\n"
            "  scons -f SConstruct_obs_rtm img_lap.rsf\n"
            "\n"
            "vel=%s\n"
            "shots=%s\n"
            % (
                project.rtm.vel_rsf,
                resolve_shot_dir(project, project.rtm.use_shots_proc),
            )
        )
    return path


def prepare_scons_workdir(project: ObsRtmProject, log: Optional[Callable[[str], None]] = None) -> str:
    """
    清理并生成 ``rtm_work/SConstruct_obs_rtm``（OBS 为源互易）。
    手跑:  cd 工区/rtm_work && scons -f SConstruct_obs_rtm img_lap.rsf
    返回偏移作业目录绝对路径。
    """
    from .sconstruct_gen import generate_sconstruct_obs_rtm

    project.ensure_workdir()
    ensure_project_vel_binary(project, log=log)
    ensure_project_bath1d(project, log=log)
    run = rtm_run_dir(project)
    if log:
        log("准备 SConstruct（OBS 为源互易）：输出目录 %s" % run)

    # 改 nt/jsnap/流程后旧 wfls/wflr 会导致 n3 mismatch；准备前先清中间结果
    if log:
        log("正在清理 rtm_work/ 中间文件…")
    clean_rtm_intermediates(project, log=log)

    # 旧导入可能 in= 为相对路径；只修「本作业将用的炮」，勿扫全库两千炮
    try:
        from ..scripts.su_to_shots import fix_shot_rsf_headers

        shot_dir = resolve_shot_dir(project, bool(project.rtm.use_shots_proc))
        ids = resolve_shot_indices(project)
        if not ids:
            if log:
                log(
                    "全炮/未限定炮号：跳过批量检查 in= "
                    "（导入时一般已是绝对路径；请用手选或 first+max 限定作业炮）"
                )
        else:
            paths = [
                os.path.join(shot_dir, "shot_%03d.rsf" % int(i))
                for i in ids
                if os.path.isfile(os.path.join(shot_dir, "shot_%03d.rsf" % int(i)))
            ]
            if log:
                log(
                    "作业炮 %d 个 · 目录 %s"
                    % (len(paths), os.path.basename(shot_dir) or shot_dir)
                )
            fix_shot_rsf_headers(paths, log=log)
    except Exception as exc:
        if log:
            log("fix shot in= 跳过: %s" % exc)

    if log:
        log("正在生成 SConstruct_obs_rtm…")
    generate_sconstruct_obs_rtm(project, log=log)
    readme = _write_rtm_layout_readme(project, run)

    tip = os.path.join(run, "rtm_gui_grid.txt")
    g = project.grid
    r = project.rtm
    with open(tip, "w", encoding="utf-8") as f:
        f.write(
            "# GUI grid / Madagascar RTM (OBS-as-source reciprocity)\n"
            "# 输入在 ../ ；本目录为输出与 SConstruct；互易 sou=OBS, rec=炮点\n"
            "engine=%s\n"
            "ox=%g dx=%g nx=%d\n"
            "oz=%g dz=%g nz=%d\n"
            "nt=%d dt=%g fmin=%g fmax=%g jsnap=%d nb=%d\n"
            "vel=../%s\n"
            "shots=%s\n"
            "run: cd rtm_work && scons -f SConstruct_obs_rtm img_lap.rsf\n"
            % (
                getattr(r, "engine", "madagascar"),
                g.ox, g.dx, g.nx, g.oz, g.dz, g.nz,
                r.nt, r.dt, r.fmin, r.fmax,
                int(getattr(r, "jsnap", 40)),
                int(getattr(r, "nb", 40)),
                r.vel_rsf,
                resolve_shot_dir(project, r.use_shots_proc),
            )
        )
    if log:
        log("wrote %s" % tip)
        log("wrote %s（输入在工区根，输出在本目录；互易 sou=OBS）" % readme)
        log(
            "手跑(OBS为源): cd %s && scons -f SConstruct_obs_rtm img_lap.rsf"
            % run
        )
    return run


def try_export_img_lap_npy(project: ObsRtmProject, log: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """若有 img_lap.rsf 等，导出为 rtm_work/img_stack.npy（优先读 rtm_work/）。"""
    run = rtm_run_dir(project)
    out = os.path.join(run, "img_stack.npy")
    for name in ("img_lap.rsf", "img_stack.rsf", "img_solid.rsf"):
        for base in (run, project.workdir):
            src = os.path.join(base, name)
            if not os.path.isfile(src):
                continue
            try:
                img, _ = read_vel_rsf(src)
                np.save(out, np.asarray(img, dtype=np.float32))
                if log:
                    log("exported %s → %s shape=%s" % (src, out, img.shape))
                return out
            except Exception as exc:
                if log:
                    log("export %s failed: %s" % (src, exc))
    return None
