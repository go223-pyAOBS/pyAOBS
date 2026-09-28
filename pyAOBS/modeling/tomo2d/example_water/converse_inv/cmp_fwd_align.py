from pathlib import Path
import math

HERE = Path(__file__).resolve().parent


def parse_picks(path):
    recs = []
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    i = 1
    src_x = 0.0
    while i < len(lines):
        parts = lines[i].split()
        i += 1
        if not parts:
            continue
        if parts[0] == "s":
            src_x = float(parts[1])
            nrcv = int(float(parts[-1]))
            for _ in range(nrcv):
                rp = lines[i].split()
                i += 1
                recs.append((src_x, float(rp[1]), float(rp[4]), int(float(rp[3]))))
    return recs


def load_rays(p):
    rays, cur = [], []
    for line in Path(p).read_text().splitlines():
        if line.startswith(">"):
            if cur:
                rays.append(cur)
                cur = []
            continue
        a = line.split()
        if len(a) >= 2:
            cur.append((float(a[0]), float(a[1])))
    if cur:
        rays.append(cur)
    return rays


def rms(a, b):
    n = min(len(a), len(b))
    s = sum((a[i][2] - b[i][2]) ** 2 for i in range(n))
    return math.sqrt(s / n) if n else float("nan"), n


def maxz(rays):
    z = 0.0
    for r in rays:
        for _, y in r:
            if y > z:
                z = y
    return z


def mean_maxz(rays, zc=4.5):
    zs = [max((y for _, y in r), default=0) for r in rays]
    deep = [z for z in zs if z > zc]
    return (sum(deep) / len(deep) if deep else 0.0), len(deep)


cv = parse_picks(HERE / "syn_inv.dat")
du = parse_picks(HERE / "syn_psx.dat")
cvs = parse_picks(HERE / "syn_start.dat")
dus = parse_picks(HERE / "syn_psx_start.dat")
r_cv = load_rays(HERE / "rays_cv_true.dat")
r_du = load_rays(HERE / "rays_psx_true.dat")
print("n cv/du", len(cv), len(du))
print("true dual vs converse RMS", rms(du, cv))
print("start dual vs converse RMS", rms(dus, cvs))
print("converse start-true RMS", rms(cvs, cv))
print("dual start-true RMS", rms(dus, du))
print("true abs mean cv, du",
      sum(x[2] for x in cv) / len(cv),
      sum(x[2] for x in du) / len(du))
n = min(len(du), len(cv))
print("true max|dt|", max(abs(du[i][2] - cv[i][2]) for i in range(n)))
print("ray maxz cv/du", maxz(r_cv), maxz(r_du))
print("ray mean-maxz>4.5 cv", mean_maxz(r_cv), "du", mean_maxz(r_du))
print("sample dt (first 8)")
for i in range(min(8, n)):
    print(i, "cv", f"{cv[i][2]:.4f}", "du", f"{du[i][2]:.4f}",
          "d", f"{du[i][2] - cv[i][2]:+.4f}")
