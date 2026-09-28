#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分层抽 15% 炮台对当「现场能拾到的 PSP」，其余 holdout。

写出 path_s/geom_inv.dat（全体 PSS + 保留 PSP）和 keep_pairs.txt。
不改 true / rec_vp / path_a / path_b。
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from make_lvz import seed_child  # noqa: E402

KEEP_FRAC = 0.15
SEED = 14
BINS = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 99.0))
SIG_PSS = 0.030
SIG_PSP = 0.010
DEST = HERE / "path_s"


def parse_shots(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    header = None
    i = 0
    if lines and lines[0].split()[:1] != ["s"]:
        header = lines[0]
        i = 1
    shots = []
    while i < len(lines):
        parts = lines[i].split()
        if parts[:1] == ["s"]:
            src = (float(parts[1]), float(parts[2]))
            recs = []
            i += 1
            while i < len(lines) and lines[i].split()[:1] == ["r"]:
                rp = lines[i].split()
                recs.append((float(rp[1]), float(rp[2])))
                i += 1
            shots.append((src, recs))
        else:
            i += 1
    return header, shots


def choose_keep(shots, frac: float, seed: int):
    pairs = []
    for isrc, (src, recs) in enumerate(shots):
        for irec, (rx, _rz) in enumerate(recs):
            dx = abs(rx - src[0])
            pairs.append((isrc, irec, src[0], rx, dx))
    rng = random.Random(seed)
    keep = set()
    for lo, hi in BINS:
        idx = [i for i, p in enumerate(pairs) if lo <= p[4] < hi]
        if not idx:
            continue
        n = max(1, int(round(len(idx) * frac)))
        n = min(n, len(idx))
        keep.update(rng.sample(idx, n))
    return pairs, keep


def _rline(x: float, z: float, code: int, t: float, dt: float) -> str:
    return f"r {x:8.3f} {z:9.3f} {code:4d} {t:9.3f} {dt:9.3f}"


def write_geom(path: Path, header, shots, keep_set, *, codes_all, codes_keep, dt_all, dt_keep):
    out = []
    if header is not None:
        out.append(header)
    for isrc, (src, recs) in enumerate(shots):
        rows = []
        for irec, (rx, rz) in enumerate(recs):
            for code in codes_all:
                rows.append(_rline(rx, rz, code, 0.0, dt_all))
            if (isrc, irec) in keep_set:
                for code in codes_keep:
                    rows.append(_rline(rx, rz, code, 0.0, dt_keep))
        out.append(f"s {src[0]:8.3f} {src[1]:9.3f} {len(rows):4d}")
        out.extend(rows)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_keep_table(path: Path, pairs, keep):
    lines = ["# isrc irec src_x rec_x offset keep"]
    for i, (isrc, irec, sx, rx, dx) in enumerate(pairs):
        flag = 1 if i in keep else 0
        lines.append(f"{isrc} {irec} {sx:.3f} {rx:.3f} {dx:.3f} {flag}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rewrite_syn_sigma(path: Path, sig_pss: float = SIG_PSS, sig_psp: float = SIG_PSP) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for ln in lines:
        p = ln.split()
        if p[:1] == ["r"] and len(p) >= 6:
            code = int(float(p[3]))
            dt = sig_psp if code == 6 else sig_pss if code == 8 else float(p[5])
            out.append(_rline(float(p[1]), float(p[2]), code, float(p[4]), dt))
        else:
            out.append(ln)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def seed_path_s() -> None:
    if not (HERE / "rec_vp.smesh").is_file():
        raise SystemExit("missing rec_vp.smesh — 先跑 lvz2d/run_wsl.sh 的 PPP")
    lid = HERE / "path_a" / "rec_vs_lid.smesh"
    if not lid.is_file():
        raise SystemExit(f"missing {lid} — 先跑 path A 盖层步")
    seed_child(DEST)
    shutil.copyfile(lid, DEST / "rec_vs_lid.smesh")
    start_a = HERE / "path_a" / "start_vs.smesh"
    if start_a.is_file():
        shutil.copyfile(start_a, DEST / "start_vs.smesh")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--rewrite-syn":
        p = Path(sys.argv[2]) if len(sys.argv) > 2 else DEST / "syn_inv.dat"
        rewrite_syn_sigma(p)
        print(f"rewrote sigma PSP={SIG_PSP} PSS={SIG_PSS}  {p}")
        return 0

    seed_path_s()
    header, shots = parse_shots(HERE / "geom_ppp.dat")
    pairs, keep = choose_keep(shots, KEEP_FRAC, SEED)
    keep_set = {(pairs[i][0], pairs[i][1]) for i in keep}
    write_keep_table(DEST / "keep_pairs.txt", pairs, keep)
    write_geom(
        DEST / "geom_inv.dat",
        header,
        shots,
        keep_set,
        codes_all=(8,),
        codes_keep=(6,),
        dt_all=SIG_PSS,
        dt_keep=SIG_PSP,
    )
    write_geom(
        DEST / "geom_psp_keep.dat",
        header,
        shots,
        keep_set,
        codes_all=(),
        codes_keep=(6,),
        dt_all=SIG_PSS,
        dt_keep=SIG_PSP,
    )
    hold_set = {
        (pairs[i][0], pairs[i][1])
        for i in range(len(pairs))
        if i not in keep
    }
    write_geom(
        DEST / "geom_psp_hold.dat",
        header,
        shots,
        hold_set,
        codes_all=(),
        codes_keep=(6,),
        dt_all=SIG_PSS,
        dt_keep=SIG_PSP,
    )
    nkeep = len(keep)
    print(
        f"path_s: keep {nkeep}/{len(pairs)} ({100.0 * nkeep / len(pairs):.1f}%)  "
        f"seed={SEED}  stratify offset"
    )
    for lo, hi in BINS:
        nbin = sum(1 for p in pairs if lo <= p[4] < hi)
        nk = sum(1 for i in keep if lo <= pairs[i][4] < hi)
        print(f"  offset [{lo:.0f},{hi:.0f})  {nk}/{nbin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
