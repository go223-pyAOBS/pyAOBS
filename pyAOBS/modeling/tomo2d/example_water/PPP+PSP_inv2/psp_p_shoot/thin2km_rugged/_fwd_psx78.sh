#!/bin/bash
set -euo pipefail
ROOT=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged
BIN=/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/src/build-tomo2d
N=-N8/8/0.8/8/1e-4/1e-5
export TOMO2D_FWD_OMP=1 OMP_NUM_THREADS=8
DUAL="$ROOT/inv_graph6k_hot"
MIX="$ROOT/inv_graph6_hot"
python3 - <<'PY'
from pathlib import Path
root = Path("/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/PPP+PSP_inv2/psp_p_shoot/thin2km_rugged")
src = (root / "inv_graph6k_hot/geom_psp6.dat").read_text(encoding="utf-8").splitlines()
def recode(code):
    out = []
    for ln in src:
        parts = ln.split()
        if parts[:1] == ["r"] and len(parts) >= 6:
            ln = (
                f"r {float(parts[1]):8.3f} {float(parts[2]):9.3f} "
                f"{code:4d} {float(parts[4]):9.3f} {float(parts[5]):9.3f}"
            )
        out.append(ln)
    return "\n".join(out) + "\n"
for code, name in ((6, "psp6"), (7, "pps7"), (8, "pss8")):
    (root / "inv_graph6k_hot" / f"geom_{name}.dat").write_text(recode(code), encoding="utf-8")
    (root / "inv_graph6_hot" / f"geom_{name}.dat").write_text(recode(code), encoding="utf-8")
print("wrote geom_psp6/pps7/pss8")
PY
for code in psp6 pps7 pss8; do
  echo "== dual $code =="
  cd "$DUAL"
  "$BIN/tt_forward" -Mtrue_vp.smesh -Utrue_vs.smesh -Ggeom_${code}.dat \
    -Xconv.refl -Bseafloor.refl "$N" > /tmp/dual_${code}.dat
  echo "== single $code =="
  cd "$MIX"
  "$BIN/tt_forward" -Mtrue_vp.smesh -k1.73 -Ggeom_${code}.dat -Xconv.refl -Bseafloor.refl \
    "$N" > /tmp/mixed_${code}.dat
done
python3 - <<'PY'
from pathlib import Path
import math, sys
sys.path.insert(0, "/mnt/d/python-learn/pyAOBS/pyAOBS/modeling/tomo2d/example_water/water_inv")
from check_water_inv import parse_picks

def stats(a, b):
    o = parse_picks(Path(a).read_text(encoding="utf-8"))
    p = parse_picks(Path(b).read_text(encoding="utf-8"))
    key = lambda t: (round(t[4], 3), round(t[1], 3), int(t[0]))
    md = {key(x): x[3] for x in p}
    ds = [md[key(x)] - x[3] for x in o if key(x) in md]
    mean = sum(ds) / len(ds)
    rms = math.sqrt(sum(v * v for v in ds) / len(ds))
    ts = [x[3] for x in o]
    return len(ds), rms, mean, sum(ts) / len(ts)

for name in ("psp6", "pps7", "pss8"):
    n, r, m, t = stats(f"/tmp/dual_{name}.dat", f"/tmp/mixed_{name}.dat")
    print(f"{name} dual vs mixed: n={n} RMS={r:.6f} s mean={m:+.6f}  dual_mean_t={t:.3f}")
PY
