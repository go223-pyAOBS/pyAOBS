#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用观测 PSS 减去正演 PSS-PSP，写成 type 6 拾取。

T_corr = T_PSS_obs - (T_PSS - T_PSP)_fwd
正演时差来自 rec_vp + rec_vs_lid（PPP+PPS 收回，面下仍是初值）。
"""

from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from make_lvz import seed_child  # noqa: E402


def parse_picks(text: str):
    recs = []
    src_x = 0.0
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    i = 1
    while i < len(lines):
        parts = lines[i].split()
        i += 1
        if parts[:1] == ["s"]:
            src_x = float(parts[1])
            nrcv = int(float(parts[-1]))
            for _ in range(nrcv):
                rp = lines[i].split()
                i += 1
                recs.append((int(float(rp[3])), float(rp[1]), float(rp[2]), float(rp[4]), src_x))
    return recs


def _pmap(picks, code: int):
    out = {}
    for p in picks:
        if int(p[0]) != code or not math.isfinite(p[3]):
            continue
        out[(round(p[4], 3), round(p[1], 3))] = float(p[3])
    return out

SIG = 0.010
Z_RCV = 0.010


def _seed(dest: Path) -> None:
    lid = HERE / "path_a" / "rec_vs_lid.smesh"
    if not (HERE / "rec_vp.smesh").is_file() or not lid.is_file():
        raise SystemExit("missing rec_vp or path_a/rec_vs_lid")
    seed_child(dest)
    shutil.copyfile(lid, dest / "rec_vs_lid.smesh")
    start_a = HERE / "path_a" / "start_vs.smesh"
    if start_a.is_file():
        shutil.copyfile(start_a, dest / "start_vs.smesh")


def corr_rows(min_dx: float = 0.0):
    true_p = HERE / "syn_true_all.dat"
    fwd_p = HERE / "syn_lidvs_all.dat"
    if not true_p.is_file() or not fwd_p.is_file():
        raise SystemExit("missing syn_true_all.dat or syn_lidvs_all.dat — 先跑 run_fwd_lidvs.sh")
    true = parse_picks(true_p.read_text(encoding="utf-8"))
    fwd = parse_picks(fwd_p.read_text(encoding="utf-8"))
    to, fo = _pmap(true, 8), _pmap(fwd, 8)
    f6 = _pmap(fwd, 6)
    t6 = _pmap(true, 6)
    keys = sorted(set(to) & set(fo) & set(f6))
    by_src: dict[float, list] = {}
    n_skip = 0
    for src, sx in keys:
        dx = abs(sx - src)
        if dx <= min_dx:
            n_skip += 1
            continue
        t_corr = to[(src, sx)] - (fo[(src, sx)] - f6[(src, sx)])
        if not math.isfinite(t_corr):
            n_skip += 1
            continue
        t_true = t6.get((src, sx), float("nan"))
        by_src.setdefault(src, []).append((sx, t_corr, t_true, dx))
    return by_src, n_skip


def write_syn(path: Path, by_src, src_z: float = 2.0) -> int:
    srcs = sorted(by_src)
    lines = [str(len(srcs))]
    n = 0
    for src in srcs:
        recs = sorted(by_src[src], key=lambda t: t[0])
        lines.append(f"s {src:.6g} {src_z:.6g} {len(recs)}")
        for sx, t_corr, _t_true, _dx in recs:
            lines.append(f"r {sx:.6g} {Z_RCV:.6g} 6 {t_corr:.6g} {SIG:.6g}")
            n += 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n


def main() -> int:
    by_all, skip0 = corr_rows(0.0)
    by_far, skip_near = corr_rows(20.0)
    for dest, by_src, tag in (
        (HERE / "path_c", by_all, "all"),
        (HERE / "path_cf", by_far, "dx>20"),
    ):
        _seed(dest)
        n = write_syn(dest / "syn_inv.dat", by_src)
        shutil.copyfile(dest / "syn_inv.dat", dest / "syn_true.dat")
        nsrc = len(by_src)
        print(f"{dest.name}: {tag}  n={n}  nsrc={nsrc}  wrote syn_inv.dat")
    print(f"skip dx<=0: {skip0}  skip dx<=20 for far: {skip_near}")
    # QC vs true PSP
    d_all = []
    d_far = []
    for src, recs in by_all.items():
        for sx, t_corr, t_true, dx in recs:
            if math.isfinite(t_true):
                d_all.append(t_corr - t_true)
                if dx > 20.0:
                    d_far.append(t_corr - t_true)
    def rms(a):
        return (sum(v * v for v in a) / len(a)) ** 0.5 if a else float("nan")
    print(f"T_corr vs true PSP  all RMS {rms(d_all):.4f} s  >20 km {rms(d_far):.4f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
