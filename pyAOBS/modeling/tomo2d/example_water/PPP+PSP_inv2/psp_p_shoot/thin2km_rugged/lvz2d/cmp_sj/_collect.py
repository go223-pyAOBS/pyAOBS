from pathlib import Path
import shutil
import sys
import inv_grid as g

d = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent / "joint"
vs = g.latest("out.smesh*.*", d)
vp = g.latest("out.vp.smesh.*.*", d)
shutil.copyfile(vs, d / "rec_vs.smesh")
shutil.copyfile(vp, d / "rec_vp.smesh")
msg = f"{d.name}: rec_vs <- {vs.name}  rec_vp <- {vp.name}"
if list(d.glob("out.refl.*.*")):
    rf = g.latest("out.refl.*.*", d)
    shutil.copyfile(rf, d / "rec_moho.refl")
    msg += f"  rec_moho <- {rf.name}"
print(msg)
