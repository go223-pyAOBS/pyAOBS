#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""策略数据：从观测走时抽出震相；正演时差校正 PSS→PSP；拾取优先。

无 matplotlib，可供 WSL python3 调用。
"""

from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path


def parse_shots(text: str):
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    header = None
    i = 0
    if lines and lines[0].split()[:1] != ["s"]:
        header = lines[0]
        i = 1
    shots = []
    while i < len(lines):
        p = lines[i].split()
        if p[:1] != ["s"]:
            i += 1
            continue
        src = (float(p[1]), float(p[2]))
        nrcv = int(float(p[-1]))
        i += 1
        recs = []
        for _ in range(nrcv):
            if i >= len(lines):
                break
            rp = lines[i].split()
            i += 1
            if rp[:1] != ["r"]:
                continue
            recs.append(
                dict(
                    x=float(rp[1]),
                    z=float(rp[2]),
                    code=int(float(rp[3])),
                    t=float(rp[4]),
                    dt=float(rp[5]) if len(rp) > 5 else 0.01,
                )
            )
        shots.append((src, recs))
    return header, shots


def write_shots(path: Path, header, shots) -> int:
    nsrc = 0
    nrec = 0
    kept = []
    for src, recs in shots:
        if not recs:
            continue
        kept.append((src, recs))
        nsrc += 1
        nrec += len(recs)
    if nsrc == 0:
        raise SystemExit(f"no receivers to write {path}")
    out = [header if header is not None else str(nsrc)]
    if header is not None:
        out[0] = str(nsrc)
    else:
        out = [str(nsrc)]
    for src, recs in kept:
        out.append(f"s {src[0]:.6g} {src[1]:.6g} {len(recs)}")
        for r in recs:
            out.append(
                f"r {r['x']:.6g} {r['z']:.6g} {r['code']} {r['t']:.6g} {r['dt']:.6g}"
            )
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return nrec


def filter_codes(src: Path, dst: Path, codes: set[int]) -> int:
    header, shots = parse_shots(src.read_text(encoding="utf-8"))
    out = []
    for src_xz, recs in shots:
        keep = [r for r in recs if r["code"] in codes and math.isfinite(r["t"])]
        out.append((src_xz, keep))
    return write_shots(dst, header, out)


def _key(src_x: float, rec_x: float):
    return (round(src_x, 3), round(rec_x, 3))


def _maps(shots):
    out = {c: {} for c in (0, 6, 7, 8)}
    meta = {}
    for src, recs in shots:
        sx = src[0]
        for r in recs:
            if r["code"] not in out or not math.isfinite(r["t"]):
                continue
            k = _key(sx, r["x"])
            out[r["code"]][k] = r
            meta.setdefault(k, dict(src=src, rec_x=r["x"], rec_z=r["z"]))
    return out, meta


def _rms(a) -> float:
    a = [float(x) for x in a if math.isfinite(x)]
    return math.sqrt(sum(v * v for v in a) / len(a)) if a else float("nan")


def _percentile(xs, q: float) -> float:
    xs = sorted(float(x) for x in xs)
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    t = (len(xs) - 1) * q / 100.0
    i = int(t)
    f = t - i
    if i + 1 >= len(xs):
        return xs[-1]
    return xs[i] * (1.0 - f) + xs[i + 1] * f


def choose_dstar(overlap, pss_dx, thresh: float, min_far: int):
    if len(overlap) >= 8:
        overlap = sorted(overlap, key=lambda t: t[0])
        cands = sorted({round(d, 2) for d, _ in overlap})
        for d0 in cands:
            far = [dl for d, dl in overlap if d + 1e-9 >= d0]
            if len(far) < min_far:
                continue
            rms = _rms(far)
            if rms <= thresh:
                return d0, "overlap-rms", rms, len(far)
        d0 = _percentile([d for d, _ in overlap], 60.0)
        far = [dl for d, dl in overlap if d + 1e-9 >= d0]
        return d0, "overlap-q60", _rms(far), len(far)
    d0 = _percentile(pss_dx, 60.0)
    return d0, "pss-q60", float("nan"), sum(1 for d in pss_dx if d + 1e-9 >= d0)


def build_below(obs_p: Path, fwd_p: Path, dst: Path, report: Path) -> dict:
    thresh = float(os.environ.get("STRATEGY_DSTAR_RMS", "0.06"))
    min_far = int(os.environ.get("STRATEGY_DSTAR_MINN", "8"))
    keep_frac = float(os.environ.get("STRATEGY_PSP_FRAC", "1.0"))
    seed = int(os.environ.get("STRATEGY_PSP_SEED", "14"))
    sig_floor = float(os.environ.get("STRATEGY_CORR_SIG", "0.05"))

    _, oshots = parse_shots(obs_p.read_text(encoding="utf-8"))
    _, fshots = parse_shots(fwd_p.read_text(encoding="utf-8"))
    om, ometa = _maps(oshots)
    fm, _ = _maps(fshots)

    keys = sorted(set(ometa) | set(om[8]) | set(om[6]))
    rng = random.Random(seed)
    kept_pick_keys = set()
    if keep_frac < 1.0 - 1e-12:
        cand = [k for k in keys if k in om[6]]
        nkeep = max(0, int(round(len(cand) * keep_frac)))
        kept_pick_keys = set(rng.sample(cand, min(nkeep, len(cand)))) if cand else set()
    else:
        kept_pick_keys = {k for k in keys if k in om[6]}

    overlap = []
    pss_dx = []
    rows = []
    for k in keys:
        src_x, rec_x = k
        dx = abs(rec_x - src_x)
        if dx <= 1e-6:
            continue
        meta = ometa.get(k)
        if meta is None:
            continue
        pss = om[8].get(k)
        pick = om[6].get(k) if k in kept_pick_keys else None
        if pss is not None:
            pss_dx.append(dx)
        t_corr = float("nan")
        if pss is not None and k in fm[8] and k in fm[6]:
            t_corr = pss["t"] - (fm[8][k]["t"] - fm[6][k]["t"])
        if pick is not None and math.isfinite(t_corr):
            overlap.append((dx, pick["t"] - t_corr))
        rows.append(
            dict(
                k=k,
                src=meta["src"],
                x=meta["rec_x"],
                z=meta["rec_z"],
                dx=dx,
                pick=pick,
                t_corr=t_corr,
            )
        )

    dstar, how, dstar_rms, n_far = choose_dstar(overlap, pss_dx, thresh, min_far)
    far_d = [dl for d, dl in overlap if d + 1e-9 >= dstar]
    sig_corr = max(sig_floor, _rms(far_d) if far_d else sig_floor)

    by_src: dict[tuple[float, float], list] = {}
    n_pick = n_corr = n_skip = 0
    for r in rows:
        src = r["src"]
        if r["pick"] is not None:
            rec = dict(r["pick"])
            rec["code"] = 6
            rec["x"], rec["z"] = r["x"], r["z"]
            by_src.setdefault(src, []).append(rec)
            n_pick += 1
        elif math.isfinite(r["t_corr"]) and r["dx"] + 1e-9 >= dstar:
            by_src.setdefault(src, []).append(
                dict(x=r["x"], z=r["z"], code=6, t=r["t_corr"], dt=sig_corr)
            )
            n_corr += 1
        else:
            n_skip += 1

    shots = [(src, recs) for src, recs in sorted(by_src.items(), key=lambda t: t[0][0])]
    n = write_shots(dst, None, shots)
    lines = [
        f"dstar_km {dstar:.4f}  rule {how}  overlap_n {len(overlap)}  far_n {n_far}",
        f"dstar_rms_s {dstar_rms if math.isfinite(dstar_rms) else float('nan'):.4f}  "
        f"sig_corr_s {sig_corr:.4f}  thresh_s {thresh:.4f}",
        f"psp_frac {keep_frac:g}  n_pick {n_pick}  n_corr {n_corr}  n_skip {n_skip}  n_out {n}",
        f"obs {obs_p}  fwd {fwd_p}  out {dst}",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return dict(dstar=dstar, n_pick=n_pick, n_corr=n_corr, n_out=n)


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(
            "strategy_flow.py filter IN OUT 0,7\n"
            "strategy_flow.py below OBS FWD OUT REPORT"
        )
    cmd = sys.argv[1]
    if cmd == "filter":
        src, dst = Path(sys.argv[2]), Path(sys.argv[3])
        codes = {int(x) for x in sys.argv[4].split(",") if x.strip() != ""}
        n = filter_codes(src, dst, codes)
        print(f"filter codes={sorted(codes)}  n={n}  -> {dst}")
        return 0
    if cmd == "below":
        build_below(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))
        return 0
    raise SystemExit(f"unknown cmd {cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
