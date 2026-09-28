# -*- coding: utf-8 -*-
"""
2D OBS 叠前 RTM — 起伏海底 + 按炮循环骨架
============================================

算法（每炮 ishot）
------------------
  1) 正传: 海面炮 (xs, zshot) → 源波场 S(x,z,t)
     （存边界 / checkpoint，勿整场落盘）
  2) 反传: 时间反转的 OBS 道集，只在海底点 (xr[i], zr[i]) 注入
           → 检波波场 R(x,z,t)
  3) 成像: I_shot += Σ_t S * R
           （可选 I_shot /= Σ_t S^2 + eps）
  4) 叠炮: I = Σ_shot I_shot → Laplacian / 带通

起伏海底
--------
  每台 OBS 的 z 不同：zr[i] = bath(xr[i])
  速度模型: z < bath(x) → vwater，否则层析
  不能用单一 gzbeg 的规则网格 RTM（除非水深几乎不变）

本脚本角色
----------
  - 读入 obs_xz.txt / shots_xz.txt / 炮集列表
  - 为每炮生成 Madagascar 几何片段（ss_###.asc）与处理命令
  - 若设置了 --rtm-bin，则按炮 subprocess 调用；否则只打印/写脚本

野外炮集约定
------------
  shots/shot_000.rsf ...
    n1 = nt  (时间), n2 = nobs（与 obs_xz.txt 行序一致）

依赖
----
  标准库；可选 numpy。真正传播需自备 RTM 可执行文件或 Madagascar 用户程序。
"""
from __future__ import print_function
import argparse
import os
import subprocess
import sys

try:
    import numpy as np
except ImportError:
    np = None


def load_xz(path):
    pts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            a = line.replace(",", " ").split()
            pts.append((float(a[0]), float(a[1])))
    return pts


def to_index(coord, origin, delta):
    return int(round((coord - origin) / delta))


def write_sou_asc(path, x, z):
    """单炮 awefd2d 风格：两行或供 echo/sfmath 使用的文本。"""
    with open(path, "w") as f:
        f.write("# single shot x z (km)\n")
        f.write("%g %g\n" % (x, z))


def write_rec_asc(path, obs_pts):
    with open(path, "w") as f:
        f.write("# OBS x z (km), one per line; z = seafloor\n")
        for x, z in obs_pts:
            f.write("%g %g\n" % (x, z))


def rsf_cat_ss_cmd(x, z, out_prefix):
    """生成用 Madagascar 做 ss.rsf 的命令（单炮）。"""
    return (
        "echo %(x)g %(z)g | "
        "sfmath output=input n1=2 d1=1 o1=0 | "
        "sfput n1=2 n2=1 d1=1 o1=0 d2=1 o2=0 "
        "label1='xz' label2='shot' > %(out)s.rsf"
        % {"x": x, "z": z, "out": out_prefix}
    )


def build_rr_cmds(obs_pts, out_prefix="rr"):
    """起伏海底 rr：先写坐标，再建议用 Python/m8r 组装；此处给出流程说明命令。"""
    n = len(obs_pts)
    lines = [
        "# Build irregular OBS receivers (variable z)",
        "# Prefer: GUI velocity page → vel.rsf, or sfmath + cat",
        "nobs=%d" % n,
    ]
    for i, (x, z) in enumerate(obs_pts):
        lines.append("# obs[%d] x=%g z=%g" % (i, x, z))
    lines.append("# -> %s.rsf shape [2, nobs]" % out_prefix)
    return "\n".join(lines)


def imaging_pseudocode():
    return """
# -------- per-shot RTM (implement inside your propagator) --------
# S[:,:,:] = 0; R[:,:,:] = 0; I[:,:] = 0
# # forward
# for it in 0..nt-1:
#     inject wavelet at (ix_s, iz_s) into S
#     step_forward(S, vel)
#     save_boundary(S, it)          # or checkpoint
# # backward + image
# for it in nt-1 .. 0:
#     reconstruct_or_reload S at it
#     for iobs in 0..nobs-1:
#         inject data[it, iobs] at (ix_r[iobs], iz_r[iobs]) into R
#     step_forward(R, vel)          # reverse-time via reverse data order
#     I += S * R                    # zero-lag cross-correlation
#     # Illum += S * S
# # I /= (Illum + eps)
# # optional: mute water column I[z < bath(x)] = 0
"""


def stack_images(paths, out_path):
    if np is None:
        print("numpy not available: stack with: sfadd %s > %s" % (" ".join(paths), out_path))
        return
    acc = None
    for p in paths:
        if not os.path.isfile(p):
            print("missing %s, skip" % p)
            continue
        a = np.load(p)
        acc = a if acc is None else acc + a
    if acc is None:
        print("no images to stack")
        return
    np.save(out_path, acc)
    print("stacked -> %s  shape=%s" % (out_path, acc.shape))


def main():
    ap = argparse.ArgumentParser(description="OBS RTM shot-loop skeleton (bathymetry OK)")
    ap.add_argument(
        "--obs",
        default="prep/geom/obs_xz.txt",
        help="OBS x z (km), z=seafloor",
    )
    ap.add_argument(
        "--shots",
        default="prep/geom/shots_xz.txt",
        help="shot x z (km)",
    )
    ap.add_argument(
        "--shot-dir",
        default="inputs/shots",
        help="directory shot_###.rsf",
    )
    ap.add_argument("--workdir", default="rtm_work")
    ap.add_argument("--ox", type=float, default=-10.0)
    ap.add_argument("--dx", type=float, default=0.025)
    ap.add_argument("--oz", type=float, default=0.0)
    ap.add_argument("--dz", type=float, default=0.025)
    ap.add_argument("--nx", type=int, default=2801)
    ap.add_argument("--nz", type=int, default=1601)
    ap.add_argument("--nt", type=int, default=10001)
    ap.add_argument("--dt", type=float, default=0.0015)
    ap.add_argument("--fmin", type=float, default=3.0)
    ap.add_argument("--fmax", type=float, default=8.0)
    ap.add_argument(
        "--vel",
        default="rtm_in/vel.rsf",
        help="migration velocity RSF",
    )
    ap.add_argument("--rtm-bin", default="", help="optional: your_rtm2d executable")
    ap.add_argument("--first-shot", type=int, default=0, help="start shot index")
    ap.add_argument("--max-shot", type=int, default=-1, help="debug: N shots from first-shot")
    ap.add_argument(
        "--shot-list",
        default="",
        help="explicit shot indices, e.g. 0,5,10-12 (overrides first/max)",
    )
    ap.add_argument("--dry-run", action="store_true", help="only write cmds, do not run")
    args = ap.parse_args()

    obs = load_xz(args.obs)
    shots = load_xz(args.shots)
    if (args.shot_list or "").strip():
        # 与 GUI parse_shot_list 一致的简易解析
        import re

        shot_indices = []
        seen = set()
        for tok in re.split(r"[,;\s]+", args.shot_list.strip()):
            tok = tok.strip()
            if not tok:
                continue
            m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", tok)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                if a > b:
                    a, b = b, a
                for i in range(a, b + 1):
                    if i not in seen and 0 <= i < len(shots):
                        seen.add(i)
                        shot_indices.append(i)
            elif tok.isdigit():
                i = int(tok)
                if i not in seen and 0 <= i < len(shots):
                    seen.add(i)
                    shot_indices.append(i)
            else:
                raise SystemExit("bad --shot-list token: %r" % tok)
        if not shot_indices:
            raise SystemExit("empty --shot-list after filter")
    else:
        i0 = max(int(args.first_shot), 0)
        if args.max_shot > 0:
            shot_indices = list(range(i0, min(i0 + args.max_shot, len(shots))))
        else:
            shot_indices = list(range(i0, len(shots)))

    os.makedirs(args.workdir, exist_ok=True)
    write_rec_asc(os.path.join(args.workdir, "rr_all.txt"), obs)

    # 索引检查：起伏 z → 每台不同 iz
    print("=== OBS grid indices (variable iz) ===")
    ixr, izr = [], []
    for i, (x, z) in enumerate(obs):
        ix = to_index(x, args.ox, args.dx)
        iz = to_index(z, args.oz, args.dz)
        ixr.append(ix)
        izr.append(iz)
        ok = 0 <= ix < args.nx and 0 <= iz < args.nz
        print("obs[%02d] x=%7.3f z=%6.3f -> ix=%4d iz=%4d %s" % (
            i, x, z, ix, iz, "OK" if ok else "OUT"))

    cmd_path = os.path.join(args.workdir, "run_shots.sh")
    img_list = []
    with open(cmd_path, "w") as sh:
        sh.write("#!/bin/sh\n# auto-generated OBS RTM shot loop\nset -e\n")
        sh.write(build_rr_cmds(obs) + "\n\n")
        sh.write(imaging_pseudocode() + "\n")

        for ishot in shot_indices:
            xs, zs = shots[ishot]
            tag = "%03d" % ishot
            ss_txt = os.path.join(args.workdir, "ss_%s.txt" % tag)
            write_sou_asc(ss_txt, xs, zs)
            ix_s = to_index(xs, args.ox, args.dx)
            iz_s = max(to_index(zs, args.oz, args.dz), 1)
            gather = os.path.join(args.shot_dir, "shot_%s.rsf" % tag)
            img = os.path.join(args.workdir, "img_%s.npy" % tag)
            img_list.append(img)

            # 预处理：带通（Madagascar 示例命令）
            prep = os.path.join(args.workdir, "shot_%s_bp.rsf" % tag)
            sh.write("\n# -------- shot %s  xs=%g zs=%g  ix=%d iz=%d --------\n" % (
                tag, xs, zs, ix_s, iz_s))
            sh.write("# bandpass + mute (mute picks: use tomography first-arrival)\n")
            sh.write(
                "sfbandpass < %s fhi=%g flo=%g > %s\n"
                % (gather, args.fmax, args.fmin, prep)
            )

            if args.rtm_bin:
                # 约定 CLI（请按你的程序改）:
                #   your_rtm2d vel=.. sou_x=.. sou_z=.. rec_list=.. data=.. img=..
                cmd = (
                    "%s --vel %s --data %s --img %s "
                    "--sou %g %g --rec-file %s "
                    "--ox %g --dx %g --oz %g --dz %g --nx %d --nz %d --nt %d --dt %g"
                    % (
                        args.rtm_bin, args.vel, prep, img,
                        xs, zs, os.path.join(args.workdir, "rr_all.txt"),
                        args.ox, args.dx, args.oz, args.dz,
                        args.nx, args.nz, args.nt, args.dt,
                    )
                )
                sh.write(cmd + "\n")
                if not args.dry_run:
                    print("RUN:", cmd)
                    subprocess.check_call(cmd, shell=True)
            else:
                sh.write(
                    "# TODO: rtm2d vel=%s data=%s sou=(%g,%g) "
                    "rec=rr_all.txt -> %s\n"
                    % (args.vel, prep, xs, zs, img)
                )
                sh.write(
                    "# inject receivers at variable iz: %s\n"
                    % ",".join(str(z) for z in izr)
                )

        sh.write("\n# stack\n# sfadd %s/img_*.rsf > %s/img_stack.rsf\n" % (
            args.workdir, args.workdir))
        sh.write("# sflaplac < img_stack.rsf > img_lap.rsf\n")

    print("\nWrote %s" % cmd_path)
    print(imaging_pseudocode())
    print("Next:")
    print("  1) GUI 速度页生成 rtm_in/vel.rsf + rtm_in/bath1d.rsf")
    print("  2) put gathers in %s/shot_###.rsf" % args.shot_dir)
    print("  3) connect --rtm-bin（或改用 GUI Madagascar scons 主路径）")
    print("  4) stack + laplac; mute water column using bath(x)")

    # 若已有 npy 像，尝试叠
    existing = [p for p in img_list if os.path.isfile(p)]
    if existing:
        stack_images(existing, os.path.join(args.workdir, "img_stack.npy"))


if __name__ == "__main__":
    main()
