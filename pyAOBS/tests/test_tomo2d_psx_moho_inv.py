# -*- coding: utf-8 -*-
"""12/13 反演核接通：正演 Moho 反射 PSP/PSS，tt_inverse 不再拒绝。"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

H = 2.0
CONV_Z = 4.0
MOHO_Z = 8.0
KAPPA = 1.73
V_WATER = 1.50
VP_LID0 = 1.80
VP_LID1 = 4.00
VP_CRUST0 = 7.20
VP_MANTLE0 = 8.00


def _is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def _bin_dir() -> Path | None:
    env = (os.getenv("PYAOBS_TOMO2D_BIN") or os.getenv("TOMO2D_BIN") or "").strip()
    names = ("tt_forward", "tt_inverse")
    here = Path(__file__).resolve()
    src = here.parents[1] / "modeling" / "tomo2d" / "src"
    cands = []
    if env:
        cands.append(Path(env))
    cands.extend(
        [
            src / "build-tomo2d",
            src / "build-tomo2d" / "Release",
            src / "build",
        ]
    )
    for root in cands:
        if all((root / n).is_file() or (root / f"{n}.exe").is_file() for n in names):
            return root
    return None


def _exe(root: Path, name: str) -> Path:
    for p in (root / name, root / f"{name}.exe"):
        if p.is_file():
            return p
    raise FileNotFoundError(name)


def _win_to_wsl(path: Path) -> str:
    p = path.resolve()
    return "/mnt/" + p.drive.rstrip(":").lower() + p.as_posix()[2:]


def _run(exe: Path, args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    if os.name == "nt" and _is_elf(exe):
        wsl = shutil.which("wsl")
        if not wsl:
            pytest.skip("Linux ELF 二进制，本机没有 wsl")
        qargs = " ".join(f"'{a}'" for a in [_win_to_wsl(exe), *args])
        cmd = [wsl, "-e", "bash", "-lc", f"cd '{_win_to_wsl(cwd)}' && {qargs}"]
        return subprocess.run(cmd, check=False, capture_output=True, text=True)
    return subprocess.run(
        [str(exe), *args], check=False, capture_output=True, text=True, cwd=cwd
    )


def _write_iface(path: Path, z: float, xmax: float = 40.0) -> None:
    path.write_text("".join(f"{x:.4f} {z:.4f}\n" for x in (0.0, xmax / 2, xmax)), encoding="utf-8")


def _write_smesh(path: Path, *, vs: bool) -> None:
    xmin, xmax, dx = 0.0, 40.0, 2.0
    zmin, zmax, dz = 0.0, 12.0, 0.5
    xs = [xmin + i * dx for i in range(int(round((xmax - xmin) / dx)) + 1)]
    zs = [zmin + k * dz for k in range(int(round((zmax - zmin) / dz)) + 1)]
    lines = [f"{len(xs)} {len(zs)} {V_WATER:.4f} 0.3300"]
    lines.append(" ".join(f"{x:.4f}" for x in xs))
    lines.append(" ".join("0.0000" for _ in xs))
    lines.append(" ".join(f"{z:.4f}" for z in zs))
    h_lid = CONV_Z - H
    for _x in xs:
        col = []
        for z in zs:
            if z <= H + 1e-9:
                v = V_WATER
            elif z < CONV_Z - 1e-9:
                vp = VP_LID0 + (VP_LID1 - VP_LID0) * (z - H) / h_lid
                v = vp / KAPPA if vs else vp
            elif z < MOHO_Z - 1e-9:
                vp = VP_CRUST0 + 0.05 * (z - CONV_Z)
                v = vp / KAPPA if vs else vp
            else:
                vp = VP_MANTLE0 + 0.10 * (z - MOHO_Z)
                v = vp / KAPPA if vs else vp
            col.append(f"{v:.4f}")
        lines.append(" ".join(col))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_geom(path: Path) -> None:
    shots = (4.0, 8.0, 32.0, 36.0)
    codes = (12, 13)
    recs = []
    for code in codes:
        for x in shots:
            recs.append(f"r  {x:8.3f}     0.010 {code:4d}     0.000     0.050")
    lines = ["1", f"s    20.000     2.000 {len(recs):4d}"]
    lines.extend(recs)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_corr(path: Path, xmax: float, zmax: float) -> None:
    path.write_text(
        "2 2\n"
        f"0 {xmax:.0f}\n"
        "0.0 0.0\n"
        f"0.0 {zmax:.1f}\n"
        "6.0 6.0\n"
        "6.0 6.0\n"
        "2.0 2.0\n"
        "2.0 2.0\n",
        encoding="utf-8",
    )


@pytest.mark.integration
def test_raytype_12_13_inverse_kernel(tmp_path: Path) -> None:
    root = _bin_dir()
    if root is None:
        pytest.skip("tt_forward/tt_inverse 不在 PATH / PYAOBS_TOMO2D_BIN / src/build-tomo2d")
    fwd = _exe(root, "tt_forward")
    inv = _exe(root, "tt_inverse")

    _write_smesh(tmp_path / "true_vp.smesh", vs=False)
    _write_smesh(tmp_path / "true_vs.smesh", vs=True)
    _write_iface(tmp_path / "seafloor.refl", H)
    _write_iface(tmp_path / "conv.refl", CONV_Z)
    _write_iface(tmp_path / "moho.refl", MOHO_Z)
    _write_geom(tmp_path / "geom.dat")
    _write_corr(tmp_path / "vcorr.dat", 40.0, 12.0)
    _write_corr(tmp_path / "dcorr.dat", 40.0, 12.0)

    nflag = "-N4/4/0.8/8/1e-4/1e-5"
    proc = _run(
        fwd,
        [
            "-Mtrue_vp.smesh",
            "-Utrue_vs.smesh",
            "-Ggeom.dat",
            "-Xconv.refl",
            "-Bseafloor.refl",
            "-Fmoho.refl",
            nflag,
        ],
        tmp_path,
    )
    if proc.returncode != 0:
        pytest.fail(f"tt_forward failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    (tmp_path / "syn.dat").write_text(proc.stdout, encoding="utf-8")
    assert "r" in proc.stdout
    assert "12" in proc.stdout and "13" in proc.stdout

    proc = _run(
        inv,
        [
            "-Mtrue_vp.smesh",
            "-Utrue_vs.smesh",
            "-Gsyn.dat",
            "-Bconv.refl",
            "-Yseafloor.refl",
            "-Fmoho.refl",
            "-w",
            "-k1.73",
            nflag,
            "-I1",
            "-SV20",
            "-SD20",
            "-TV2",
            "-TD0.3",
            "-CVvcorr.dat",
            "-CDdcorr.dat",
            "-Oout",
            "-l",
            "-Linv.log",
            "-V0",
        ],
        tmp_path,
    )
    text = proc.stdout + "\n" + proc.stderr
    assert "not wired in inversion yet" not in text
    if proc.returncode != 0:
        pytest.fail(f"tt_inverse failed ({proc.returncode})\n{text}")
    log = (tmp_path / "inv.log").read_text(encoding="utf-8", errors="replace")
    assert "S-Moho kernel" in text or "S-Moho kernel" in log
    assert any(tmp_path.glob("out.smesh.*"))
