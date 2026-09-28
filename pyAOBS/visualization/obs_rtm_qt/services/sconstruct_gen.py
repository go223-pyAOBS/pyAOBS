# -*- coding: utf-8 -*-
"""
由 GUI 工程参数生成 Madagascar SConstruct（野外 OBS 叠前 RTM）。

互易 / OBS 为源：
  正传 sou=OBS, rec=手选炮点, expl=y
  反传 sou=手选炮点（awefd2d 只在 sou 注入）, rec=OBS, expl=n, adj=y
  道集由各 shot_### 第 iobs 道拼成；每台 OBS：awefd2d × 2

引擎：sfawefd2d（snap 波场 + 零延迟互相关）。

目录约定：
  工区根 = 输入（vel / rr / bath / shots / 坐标）
  rtm_work/ = 本 SConstruct + 预拼 OBS 道集 + 中间与成像结果
  运行：cd rtm_work && scons -f SConstruct_obs_rtm img_lap.rsf
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..project import ObsRtmProject


def _rel_to_run(project: ObsRtmProject, path: str, run_dir: str) -> str:
    """路径相对偏移作业目录（scons cwd = rtm_work）。"""
    ap = os.path.abspath(path)
    rd = os.path.abspath(run_dir)
    try:
        rel = os.path.relpath(ap, rd)
        return rel.replace("\\", "/")
    except Exception:
        return ap.replace("\\", "/")


def _rsf_binary_ok(path: str) -> bool:
    """头文件存在且 in= 数据体可读（避免仅有 .rsf 无 .rsf@ 时 scons Error 1）。"""
    if not path or not os.path.isfile(path):
        return False
    try:
        from .rsf_io import resolve_rsf_binary

        resolve_rsf_binary(path)
        return True
    except Exception:
        return False


def _lines(*parts: str) -> str:
    return "\n".join(parts) + "\n"


def _resolve_shot_indices(project: ObsRtmProject, n_shots: int) -> List[int]:
    from .rtm_job import parse_shot_list

    r = project.rtm
    spec = str(getattr(r, "shot_list", "") or "").strip()
    if spec:
        ids = [i for i in parse_shot_list(spec) if 0 <= int(i) < n_shots]
        if not ids:
            raise ValueError("自选炮号为空或越界，请检查手选道集")
        return [int(i) for i in ids]
    i0 = max(int(getattr(r, "first_shot", 0) or 0), 0)
    n = int(r.max_shot)
    if n > 0:
        return list(range(i0, min(i0 + n, n_shots)))
    return list(range(i0, n_shots))


def _resolve_obs_indices(project: ObsRtmProject, n_obs: int) -> List[int]:
    from .rtm_job import parse_shot_list

    r = project.rtm
    spec = str(getattr(r, "obs_list", "") or "").strip()
    if spec:
        ids = [i for i in parse_shot_list(spec) if 0 <= int(i) < n_obs]
        if not ids:
            raise ValueError("OBS 列表为空或越界（obs_list）")
        return [int(i) for i in ids]
    return list(range(n_obs))


def _prepare_obs_source_inputs(
    project: ObsRtmProject,
    run_dir: str,
    shot_dir: str,
    shot_indices: Sequence[int],
    obs_indices: Sequence[int],
    obs_pts: Sequence[Tuple[float, float]],
    shot_pts: Sequence[Tuple[float, float]],
    *,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[str, List[Tuple[int, str, float, float]]]:
    """
    写入 rec_shots.rsf（炮点作检波）与各 obs_gath_XXX.rsf。
    返回 (rec_shots 相对路径, [(iobs, gather_rel, xo, zo), ...])。
    """
    from .rsf_io import read_gather, write_gather
    from .velocity import write_xz_rsf

    rec_pts = [(float(shot_pts[i][0]), float(shot_pts[i][1])) for i in shot_indices]
    rec_path = os.path.join(run_dir, "rec_shots.rsf")
    write_xz_rsf(rec_path, rec_pts, label2="SHOT_REC")
    rec_rel = _rel_to_run(project, rec_path, run_dir)

    # 预读炮集（同 nt）；缺文件跳过并记入 used
    loaded: List[Tuple[int, np.ndarray, float, float]] = []
    d1 = float(project.rtm.dt)
    o1 = 0.0
    for ishot in shot_indices:
        path = os.path.join(shot_dir, "shot_%03d.rsf" % int(ishot))
        if not os.path.isfile(path):
            if log:
                log("SKIP missing gather: %s" % path)
            continue
        data, meta = read_gather(path)
        d1 = float(meta.get("d1", d1))
        o1 = float(meta.get("o1", o1))
        loaded.append((int(ishot), np.asarray(data, dtype=np.float32), d1, o1))
    if not loaded:
        raise RuntimeError("没有可用的 shot_*.rsf（手选炮在 shots/ 下均未找到）")

    # 若有缺炮，rec_shots 须与实际拼道一致：重写 rec
    if len(loaded) != len(shot_indices):
        used_ids = [h for h, _, _, _ in loaded]
        rec_pts = [(float(shot_pts[i][0]), float(shot_pts[i][1])) for i in used_ids]
        write_xz_rsf(rec_path, rec_pts, label2="SHOT_REC")
        if log:
            log(
                "手选 %d 炮中 %d 个有道集，已按实有炮重写 rec_shots.rsf"
                % (len(shot_indices), len(loaded))
            )

    # 未用 shots_proc 时：拼道阶段用 Python mute（相对 OBS offset / model x）。
    # 勿依赖 SConstruct mutter：obs_gath 轴2 是道序号，x0=OBS 绝对坐标会错。
    prep = project.preprocess
    use_proc = bool(getattr(project.rtm, "use_shots_proc", True))
    bake_mute = (not use_proc) and (
        bool(getattr(prep, "use_mute", False))
        or (
            bool(getattr(prep, "use_poly_mute", False))
            and len(getattr(prep, "poly_points", None) or []) >= 3
        )
    )
    if bake_mute:
        from .preprocess import apply_mute_only, time_axis

    rows: List[Tuple[int, str, float, float]] = []
    for iobs in obs_indices:
        iobs = int(iobs)
        xo, zo = float(obs_pts[iobs][0]), float(obs_pts[iobs][1])
        cols: List[np.ndarray] = []
        model_xs: List[float] = []
        for ishot, data, _d1, _o1 in loaded:
            if data.ndim != 2 or data.shape[1] <= iobs:
                raise RuntimeError(
                    "shot 道数 n2=%d 不足以取 OBS 列 iobs=%d"
                    % (data.shape[1] if data.ndim == 2 else -1, iobs)
                )
            cols.append(np.asarray(data[:, iobs], dtype=np.float32))
            model_xs.append(float(shot_pts[int(ishot)][0]))
        gather = np.column_stack(cols).astype(np.float32, copy=False)
        if bake_mute:
            nt = int(gather.shape[0])
            times = time_axis(nt, d1, o1)
            model_x = np.asarray(model_xs, dtype=float)
            rel_off = model_x - float(xo)
            gather = apply_mute_only(
                gather,
                times,
                rel_off,
                prep,
                x_coords=model_x,
                x_reduce_origin=float(xo),
                skip_poly=False,
            )
        gpath = os.path.join(run_dir, "obs_gath_%03d.rsf" % iobs)
        write_gather(gpath, gather, d1=d1, o1=o1)
        grow_rel = _rel_to_run(project, gpath, run_dir)
        rows.append((iobs, grow_rel, xo, zo))
        if log:
            extra = " ·已预mute" if bake_mute else ""
            log(
                "OBS %d @ (%.3f, %.3f) km · 拼道 %d 炮%s → %s"
                % (iobs, xo, zo, gather.shape[1], extra, os.path.basename(gpath))
            )
    return rec_rel, rows


def generate_sconstruct_obs_rtm(
    project: ObsRtmProject,
    *,
    out_name: str = "SConstruct_obs_rtm",
    log: Optional[Callable[[str], None]] = None,
) -> str:
    """
    写入 ``rtm_work/SConstruct_obs_rtm``（OBS 为源互易），返回绝对路径。
    """
    from .geometry import load_xz_txt
    from .rtm_job import describe_shot_selection, resolve_shot_dir, rtm_run_dir

    project.ensure_workdir()
    run_dir = rtm_run_dir(project)
    r = project.rtm
    g = project.grid
    vp = project.velocity

    vel = project.path(r.vel_rsf)
    if not os.path.isfile(vel):
        alt = project.path(vp.out_vel)
        if os.path.isfile(alt):
            vel = alt
    if not os.path.isfile(vel):
        raise FileNotFoundError("缺少速度 RSF（请先生成 vel.rsf）: %s" % r.vel_rsf)
    if not os.path.isfile(project.path(project.shots_xz)):
        raise FileNotFoundError("缺少 %s" % project.shots_xz)
    if not os.path.isfile(project.path(project.obs_xz)):
        raise FileNotFoundError("缺少 %s" % project.obs_xz)

    shot_pts = load_xz_txt(project.path(project.shots_xz))
    obs_pts = load_xz_txt(project.path(project.obs_xz))
    if not shot_pts:
        raise RuntimeError("shots_xz.txt 无炮点")
    if not obs_pts:
        raise RuntimeError("obs_xz.txt 无 OBS")

    shot_dir = resolve_shot_dir(project, r.use_shots_proc)
    shot_indices = _resolve_shot_indices(project, len(shot_pts))
    obs_indices = _resolve_obs_indices(project, len(obs_pts))

    if log:
        log(
            "OBS 为源（互易）: %d 台 OBS × 2 次 awefd2d；检波=手选 %d 炮"
            % (len(obs_indices), len(shot_indices))
        )

    rec_rel, obs_rows = _prepare_obs_source_inputs(
        project,
        run_dir,
        shot_dir,
        shot_indices,
        obs_indices,
        obs_pts,
        shot_pts,
        log=log,
    )
    if not obs_rows:
        raise RuntimeError("未生成任何 OBS 道集")

    vel_rel = _rel_to_run(project, vel, run_dir)
    from .workdir_layout import path_bath1d

    bath_rel = _rel_to_run(project, path_bath1d(project), run_dir)
    p = project.preprocess
    apply_bp = not bool(r.use_shots_proc)
    # 速度/多边形 mute 已在 _prepare_obs_source_inputs 用 Python 预烘焙（相对 OBS）；
    # SConstruct 不再 mutter（轴2=道序号时 x0=OBS 绝对坐标必错）。
    apply_mute = False
    tmute = float(p.tmute)
    vmute = float(p.vmute)
    mute_tp = float(getattr(p, "mute_tp", 0.15) or 0.0)
    # mutter inner=y 切深（默认）；反选切浅 → inner=n（保留参数仅兼容旧脚本）
    mute_inner = not bool(getattr(p, "vel_mute_invert", False))
    if (
        not bool(r.use_shots_proc)
        and (bool(p.use_mute) or bool(getattr(p, "use_poly_mute", False)))
        and log
    ):
        log(
            "未用 shots_proc：mute 已在拼 obs_gath 时用 Python 预应用"
            "（相对 OBS offset / model x）；SConstruct 不再 mutter"
        )

    fm = 0.5 * (float(r.fmin) + float(r.fmax))
    if fm <= 0:
        fm = max(float(r.fmax), 6.0)
    kt = max(int(1.0 / (fm * float(r.dt))), 1)
    jsnap = max(int(getattr(r, "jsnap", 40) or 40), 1)
    nb = max(int(getattr(r, "nb", 40) or 40), 1)
    smooth = int(vp.smooth_rect) if int(vp.smooth_rect) > 0 else 0
    verb = "y" if bool(getattr(r, "awefd_verb", True)) else "n"
    zmax = g.oz + (g.nz - 1) * g.dz
    xmax = g.ox + (g.nx - 1) * g.dx
    nsnap = max(int(r.nt) // jsnap, 1)
    nt_i = int(r.nt)

    # 对齐 nt：探测第一道 OBS 道集
    shot_n1 = None
    shot_d1 = None
    try:
        from .rsf_io import parse_rsf_header

        probe = os.path.join(run_dir, "obs_gath_%03d.rsf" % obs_rows[0][0])
        if os.path.isfile(probe):
            meta = parse_rsf_header(probe)
            shot_n1 = int(meta["n1"])
            shot_d1 = float(meta.get("d1", r.dt))
    except Exception:
        pass
    if shot_n1 is not None and int(shot_n1) > nt_i:
        match_cmd = "'window n1=%(nt)d' % par"
        match_how = "window(截断)"
    elif shot_n1 is not None and int(shot_n1) < nt_i:
        match_cmd = "'pad n1=%(nt)d' % par"
        match_how = "pad(补零)"
    else:
        match_cmd = "'window n1=%(nt)d' % par"
        match_how = "window"
    if log and shot_n1 is not None and int(shot_n1) != nt_i:
        log(
            "注意: OBS 道集 n1=%d 与 RTM nt=%d 不同 → mut_t 用 %s 对齐到 nt"
            % (shot_n1, nt_i, match_how)
        )
    if log and shot_d1 is not None and abs(float(shot_d1) - float(r.dt)) > 1e-9:
        log(
            "注意: 道集 d1=%g 与 RTM dt=%g 不一致，请核对采样率 fs"
            % (shot_d1, float(r.dt))
        )

    sel_txt = describe_shot_selection(project)
    L: List[str] = []
    L.append("# -*- coding: utf-8 -*-")
    L.append('"""AUTO-GENERATED by pyAOBS obs_rtm_qt — OBS-as-source (reciprocal) RTM."""')
    L.append("# Inputs: ../vel.rsf ../shots*/ + prebuilt obs_gath_*/rec_shots")
    L.append("# Mode: sou=OBS, rec=selected shots; 2×awefd2d per OBS")
    L.append("from rsf.proj import *")
    L.append("import os")
    L.append("import sys")
    L.append("")
    L.append("par = dict(")
    L.append("    oz=%g, dz=%g, nz=%d," % (g.oz, g.dz, int(g.nz)))
    L.append("    ox=%g, dx=%g, nx=%d," % (g.ox, g.dx, int(g.nx)))
    L.append("    ot=0.0, dt=%g, nt=%d," % (float(r.dt), int(r.nt)))
    L.append(
        "    fmin=%g, fmax=%g, fm=%g, kt=%d,"
        % (float(r.fmin), float(r.fmax), fm, kt)
    )
    L.append("    nb=%d, jsnap=%d, vwater=%g," % (nb, jsnap, float(vp.vwater)))
    L.append(
        "    apply_bp=%s, apply_mute=%s,"
        % (str(bool(apply_bp)), str(bool(apply_mute)))
    )
    L.append(
        "    tmute=%g, vmute=%g, mute_tp=%g, mute_inner=%s,"
        % (tmute, vmute, mute_tp, str(bool(mute_inner)))
    )
    L.append("    verb=%r," % verb)
    L.append(")")
    L.append("par['zmax'] = %g" % zmax)
    L.append("par['xmax'] = %g" % xmax)
    L.append(
        "print('OBS-as-source RTM: nx={nx} nz={nz} nt={nt} jsnap={jsnap} "
        "~{nsnap} snaps/OBS n_obs={nobs} n_rec_shots={nrec} verb={verb}'.format("
        "nx=par['nx'], nz=par['nz'], nt=par['nt'], jsnap=par['jsnap'], "
        "nsnap=%d, nobs=%d, nrec=%d, verb=par['verb']), flush=True)"
        % (nsnap, len(obs_rows), len(shot_indices))
    )
    L.append("VEL = %r" % vel_rel)
    L.append("REC_SHOTS = %r  # selected surface shots as receivers" % rec_rel)
    L.append("BATH2D_FROM = %r" % bath_rel)
    L.append("_SMOOTH = %d" % smooth)
    # (iobs, gather_rel, xo, zo)
    L.append("OBS_JOBS = %r" % [(i, grel, xo, zo) for i, grel, xo, zo in obs_rows])
    L.append("")
    L.append("if _SMOOTH > 0:")
    L.append(
        "    Flow('vels', VEL, 'smooth rect1=%d rect2=%d repeat=1' % (_SMOOTH, _SMOOTH))"
    )
    L.append("else:")
    L.append("    Flow('vels', VEL, 'window')")
    L.append("Flow('den', 'vels', 'math output=1')")
    L.append("")
    # 生成时校验数据体；勿仅用 isfile(头文件)（无 .rsf@ 时 sfspray → Error 1）
    if _rsf_binary_ok(path_bath1d(project)):
        L.append(
            "Flow('bath2d', BATH2D_FROM,"
            "     'spray axis=1 n=%(nz)d o=%(oz)g d=%(dz)g | "
            "put label1=Depth unit1=km label2=Distance unit2=km' % par)"
        )
    else:
        L.append("Flow('bath2d', 'vels', 'math output=0')")
        if log:
            log("bath1d 数据体不可用 → bath2d 用 0（无压水柱）")
    L.append("")
    L.append(
        "Flow('wav', None,"
        "     'spike nsp=1 n1=%(nt)d d1=%(dt)g o1=%(ot)g k1=%(kt)d | "
        "ricker1 frequency=%(fm)g | scale axis=1 | transp' % par)"
    )
    L.append("")
    L.append("img_names = []")
    L.append("for iobs, gather, xo, zo in OBS_JOBS:")
    L.append("    tag = '%03d' % int(iobs)")
    L.append(
        "    print('>>> schedule OBS', tag, 'as SOURCE x=%.3f z=%.3f' % (xo, zo), "
        "flush=True)"
    )
    L.append("    Flow('xs_obs_' + tag, None, 'math n1=1 output=%g' % float(xo))")
    L.append(
        "    Flow('zs_obs_' + tag, None, 'math n1=1 output=%g' % max(float(zo), 1e-4))"
    )
    L.append(
        "    Flow('ss_obs_' + tag, ['xs_obs_' + tag, 'zs_obs_' + tag],"
        "         'cat axis=2 space=n ${SOURCES[0]} ${SOURCES[1]} | transp', stdin=0)"
    )
    L.append("    Flow('raw_' + tag, gather, 'window')")
    L.append("    if par['apply_bp']:")
    L.append(
        "        Flow('bp_' + tag, 'raw_' + tag,"
        "             'bandpass fhi=%(fmax)g flo=%(fmin)g' % par)"
    )
    L.append("    else:")
    L.append("        Flow('bp_' + tag, 'raw_' + tag, 'window')")
    L.append("    # mute：由 GUI 拼 obs_gath 时 Python 预应用（相对 OBS）；此处不再 mutter")
    L.append("    if par['apply_mute']:")
    L.append(
        "        Flow('mut_' + tag, 'bp_' + tag,"
        "             'mutter x0=0 half=n t0=%g tp=%g abs=y v0=%g inner=%s' % ("
        "par['tmute'], par.get('mute_tp', 0.15), par['vmute'], "
        "'y' if par.get('mute_inner', True) else 'n'))"
    )
    L.append("    else:")
    L.append("        Flow('mut_' + tag, 'bp_' + tag, 'window')")
    L.append("    Flow('mut_t_' + tag, 'mut_' + tag, %s)" % match_cmd)
    L.append(
        "    print('>>> awefd2d FORWARD OBS', tag, "
        "'(sou=OBS, rec=shots, expl=y)', flush=True)"
    )
    # 正传：单点 OBS 源可用 expl=y
    L.append(
        "    Flow(['datf_' + tag, 'wfls_' + tag],"
        "         ['wav', 'vels', 'den', 'ss_obs_' + tag, REC_SHOTS],"
        "         'awefd2d vel=${SOURCES[1]} den=${SOURCES[2]} "
        "sou=${SOURCES[3]} rec=${SOURCES[4]} wfl=${TARGETS[1]} "
        "verb=%(verb)s free=y expl=y dabc=y nb=%(nb)d snap=y "
        "jsnap=%(jsnap)d jdata=1' % par)"
    )
    L.append("    print('>>> awefd2d FORWARD done OBS', tag, flush=True)")
    # dat: n1=time,n2=nshot → transp 后 n1=ns,n2=nt（awefd2d 非 expl 按此读入）。
    # 勿再 reverse：adj=y 内部已按 (nt-it-1) 倒序注入；再 reverse = 双重反转，
    # 空间上仍从炮点发散，但与正传时间对不齐 → 孔径内互相关接近零。
    L.append("    Flow('dat_adj_' + tag, 'mut_t_' + tag, 'transp')")
    L.append(
        "    print('>>> awefd2d ADJOINT OBS', tag, "
        "'(sou=shots 注入反传, expl=n；awefd2d 始终在 sou 注入)', flush=True)"
    )
    # 关键：sfawefd2d 无论 adj 与否都在 sou 注入、在 rec 抽取。
    # 反传须 sou=检波炮点、expl=n（多道），不能再用 sou=OBS + expl=y。
    L.append(
        "    Flow(['datb_' + tag, 'wflr_' + tag],"
        "         ['dat_adj_' + tag, 'vels', 'den', REC_SHOTS, 'ss_obs_' + tag],"
        "         'awefd2d adj=y vel=${SOURCES[1]} den=${SOURCES[2]} "
        "sou=${SOURCES[3]} rec=${SOURCES[4]} wfl=${TARGETS[1]} "
        "verb=%(verb)s free=y expl=n dabc=y nb=%(nb)d snap=y "
        "jsnap=%(jsnap)d jdata=1' % par)"
    )
    L.append("    print('>>> awefd2d ADJOINT done OBS', tag, flush=True)")
    # adj=y 快照与正传同一 it 索引：I = Σ_t S(t)·Adj(t)，不要再 reverse which=3
    L.append(
        "    Flow('img_obs_' + tag, ['wfls_' + tag, 'wflr_' + tag],"
        "         'add ${SOURCES[1]} mode=p | stack axis=3 norm=n')"
    )
    L.append("    img_names.append('img_obs_' + tag)")
    L.append("")
    L.append("if not img_names:")
    L.append("    raise RuntimeError('no OBS jobs scheduled')")
    L.append("")
    # 分批跑多台 OBS：叠后 = 盘上已有 img_obs_* ∪ 本轮目标（勿只用本轮子集覆盖）
    L.append("import glob as _glob")
    L.append("import os as _os")
    L.append("import re as _re")
    L.append("_disk_obs = []")
    L.append("for _p in _glob.glob('img_obs_*.rsf'):")
    L.append("    if _p.endswith('.rsf@'):")
    L.append("        continue")
    L.append("    _m = _re.match(r'img_obs_(\\d+)\\.rsf$', _os.path.basename(_p))")
    L.append("    if _m:")
    L.append("        _disk_obs.append('img_obs_%03d' % int(_m.group(1)))")
    L.append(
        "_stack_src = sorted("
        "set(_disk_obs) | set(img_names), "
        "key=lambda s: int(s.rsplit('_', 1)[-1]))"
    )
    L.append(
        "print('>>> stack OBS images:', ','.join(_stack_src), "
        "'(disk+this run)', flush=True)"
    )
    L.append("if len(_stack_src) == 1:")
    L.append("    Flow('img_stack', _stack_src[0], 'cp')")
    L.append("else:")
    L.append(
        "    Flow('img_stack', _stack_src, "
        "'add ${SOURCES[1:%d]}' % len(_stack_src))"
    )
    L.append(
        "Flow('img_solid', 'img_stack bath2d',"
        "     \"math b=${SOURCES[1]} output='input*0.5*(1+sign(x1-b))'\")"
    )
    L.append("Flow('img_lap', 'img_solid', 'laplac')")
    L.append("End()")

    out_path = os.path.join(run_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_lines(*L))
    if log:
        log(
            "wrote %s  (OBS-as-source, %s, n_obs=%d, jsnap=%d, bp=%s, mute=%s "
            "t0=%g v0=%g, tmatch=%s)"
            % (
                out_path,
                sel_txt,
                len(obs_rows),
                jsnap,
                apply_bp,
                apply_mute,
                tmute,
                vmute,
                match_how,
            )
        )
    return out_path


def generate_sconstruct_impulse_rtm(
    project: ObsRtmProject,
    *,
    iobs: int,
    ishot: int,
    it: int,
    amp: float,
    out_name: str = "SConstruct_impulse",
    log: Optional[Callable[[str], None]] = None,
) -> str:
    """
    写入 ``rtm_work/impulse/SConstruct_impulse``：单 OBS + 单检波炮 +
    与正传同 ``fm`` 的带限 Ricker 道集（峰值=拾取样点），无带通/mute。
    目标 ``img_impulse_lap.rsf``。
    """
    from .geometry import load_xz_txt
    from .impulse_gather import (
        impulse_nt_jsnap,
        impulse_run_dir,
        write_impulse_obs_gather,
    )

    project.ensure_workdir()
    run_dir = impulse_run_dir(project)
    r = project.rtm
    g = project.grid
    vp = project.velocity

    vel = project.path(r.vel_rsf)
    if not os.path.isfile(vel):
        alt = project.path(vp.out_vel)
        if os.path.isfile(alt):
            vel = alt
    if not os.path.isfile(vel):
        raise FileNotFoundError("缺少速度 RSF（请先生成 vel.rsf）")

    shot_pts = load_xz_txt(project.path(project.shots_xz))
    obs_pts = load_xz_txt(project.path(project.obs_xz))
    if not shot_pts or not obs_pts:
        raise RuntimeError("缺少 shots_xz / obs_xz")

    iobs = int(iobs)
    ishot = int(ishot)
    nt, jsnap, nsnap = impulse_nt_jsnap(project, it)
    gpath, _rec, nt, it_u, amp_u = write_impulse_obs_gather(
        project, iobs=iobs, ishot=ishot, it=it, amp=amp, nt=nt, log=log
    )
    xo, zo = float(obs_pts[iobs][0]), float(obs_pts[iobs][1])
    gather_rel = os.path.basename(gpath)
    rec_rel = "rec_shots.rsf"
    vel_rel = _rel_to_run(project, vel, run_dir)
    from .workdir_layout import path_bath1d

    bath_rel = _rel_to_run(project, path_bath1d(project), run_dir)

    fm = 0.5 * (float(r.fmin) + float(r.fmax))
    if fm <= 0:
        fm = max(float(r.fmax), 6.0)
    kt = max(int(1.0 / (fm * float(r.dt))), 1)
    nb = max(int(getattr(r, "nb", 40) or 40), 1)
    smooth = int(vp.smooth_rect) if int(vp.smooth_rect) > 0 else 0
    verb = "y" if bool(getattr(r, "awefd_verb", True)) else "n"
    zmax = g.oz + (g.nz - 1) * g.dz
    xmax = g.ox + (g.nx - 1) * g.dx
    t0 = float(it_u) * float(r.dt)
    if log:
        log(
            "脉冲加速参数: nt=%d (作业 nt=%d) jsnap=%d → ~%d 张快照；网格 %d×%d"
            % (nt, int(r.nt), jsnap, nsnap, int(g.nx), int(g.nz))
        )

    L: List[str] = []
    L.append("# -*- coding: utf-8 -*-")
    L.append(
        '"""AUTO-GENERATED — impulse RTM '
        '(band-limited Ricker @ pick, matched to forward fm)."""'
    )
    L.append("from rsf.proj import *")
    L.append("import os")
    L.append("")
    L.append("par = dict(")
    L.append("    oz=%g, dz=%g, nz=%d," % (g.oz, g.dz, int(g.nz)))
    L.append("    ox=%g, dx=%g, nx=%d," % (g.ox, g.dx, int(g.nx)))
    L.append("    ot=0.0, dt=%g, nt=%d," % (float(r.dt), int(nt)))
    L.append(
        "    fmin=%g, fmax=%g, fm=%g, kt=%d,"
        % (float(r.fmin), float(r.fmax), fm, kt)
    )
    L.append("    nb=%d, jsnap=%d, vwater=%g," % (nb, jsnap, float(vp.vwater)))
    L.append("    verb=%r," % verb)
    L.append(")")
    L.append("par['zmax'] = %g" % zmax)
    L.append("par['xmax'] = %g" % xmax)
    L.append(
        "print('Impulse RTM: OBS=%d shot=%d t=%.4fs amp=%.4g "
        "nx=%%(nx)d nz=%%(nz)d nt=%%(nt)d jsnap=%%(jsnap)d ~%d snaps' %% par, "
        "flush=True)"
        % (iobs, ishot, t0, amp_u, nsnap)
    )
    L.append("VEL = %r" % vel_rel)
    L.append("REC_SHOTS = %r" % rec_rel)
    L.append("BATH2D_FROM = %r" % bath_rel)
    L.append("GATHER = %r" % gather_rel)
    L.append("_SMOOTH = %d" % smooth)
    L.append("IOBS, XO, ZO = %d, %g, %g" % (iobs, xo, zo))
    L.append("")
    L.append("if _SMOOTH > 0:")
    L.append(
        "    Flow('vels', VEL, 'smooth rect1=%d rect2=%d repeat=1' % (_SMOOTH, _SMOOTH))"
    )
    L.append("else:")
    L.append("    Flow('vels', VEL, 'window')")
    L.append("Flow('den', 'vels', 'math output=1')")
    L.append("")
    if _rsf_binary_ok(path_bath1d(project)):
        L.append(
            "Flow('bath2d', BATH2D_FROM,"
            "     'spray axis=1 n=%(nz)d o=%(oz)g d=%(dz)g | "
            "put label1=Depth unit1=km label2=Distance unit2=km' % par)"
        )
    else:
        L.append("Flow('bath2d', 'vels', 'math output=0')")
        if log:
            log("bath1d 数据体不可用 → bath2d 用 0（无压水柱）")
    L.append("")
    L.append(
        "Flow('wav', None,"
        "     'spike nsp=1 n1=%(nt)d d1=%(dt)g o1=%(ot)g k1=%(kt)d | "
        "ricker1 frequency=%(fm)g | scale axis=1 | transp' % par)"
    )
    L.append("")
    L.append("tag = '%03d' % int(IOBS)")
    L.append("Flow('xs_obs_' + tag, None, 'math n1=1 output=%g' % float(XO))")
    L.append(
        "Flow('zs_obs_' + tag, None, 'math n1=1 output=%g' % max(float(ZO), 1e-4))"
    )
    L.append(
        "Flow('ss_obs_' + tag, ['xs_obs_' + tag, 'zs_obs_' + tag],"
        "     'cat axis=2 space=n ${SOURCES[0]} ${SOURCES[1]} | transp', stdin=0)"
    )
    # 无 bp/mute：直接对齐 nt
    L.append("Flow('mut_t_' + tag, GATHER, 'window n1=%(nt)d' % par)")
    # 勿在 Flow 前 print：那是读 SConstruct 阶段，不是真正开算
    L.append(
        "Flow(['datf_' + tag, 'wfls_' + tag],"
        "     ['wav', 'vels', 'den', 'ss_obs_' + tag, REC_SHOTS],"
        "     'awefd2d vel=${SOURCES[1]} den=${SOURCES[2]} "
        "sou=${SOURCES[3]} rec=${SOURCES[4]} wfl=${TARGETS[1]} "
        "verb=%(verb)s free=y expl=y dabc=y nb=%(nb)d snap=y "
        "jsnap=%(jsnap)d jdata=1' % par)"
    )
    L.append("Flow('dat_adj_' + tag, 'mut_t_' + tag, 'transp')")
    L.append(
        "Flow(['datb_' + tag, 'wflr_' + tag],"
        "     ['dat_adj_' + tag, 'vels', 'den', REC_SHOTS, 'ss_obs_' + tag],"
        "     'awefd2d adj=y vel=${SOURCES[1]} den=${SOURCES[2]} "
        "sou=${SOURCES[3]} rec=${SOURCES[4]} wfl=${TARGETS[1]} "
        "verb=%(verb)s free=y expl=n dabc=y nb=%(nb)d snap=y "
        "jsnap=%(jsnap)d jdata=1' % par)"
    )
    L.append(
        "Flow('img_impulse', ['wfls_' + tag, 'wflr_' + tag],"
        "     'add ${SOURCES[1]} mode=p | stack axis=3 norm=n')"
    )
    L.append(
        "Flow('img_impulse_solid', 'img_impulse bath2d',"
        "     \"math b=${SOURCES[1]} output='input*0.5*(1+sign(x1-b))'\")"
    )
    L.append("Flow('img_impulse_lap', 'img_impulse_solid', 'laplac')")
    L.append("End()")

    out_path = os.path.join(run_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_lines(*L))
    if log:
        log(
            "wrote %s  (impulse OBS=%d shot=%d it=%d amp=%.4g, jsnap=%d)"
            % (out_path, iobs, ishot, it_u, amp_u, jsnap)
        )
    return out_path
