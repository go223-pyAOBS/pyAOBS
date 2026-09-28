"""Killer test: hybrid mesh, joint-Fermat PSP; S below conv when Vs is not much slower than lid P."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

CONV_Z = 3.0
VP_ABOVE0 = 2.20
VP_ABOVE_GRAD = 0.25  # 2.20 → 2.95 at z=3
VS_BELOW0 = 3.50


def _is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def _find_tt_forward() -> Path | None:
    env = (os.getenv("PYAOBS_TOMO2D_BIN") or os.getenv("TOMO2D_BIN") or "").strip()
    names = ("tt_forward", "tt_forward.exe")
    if env:
        root = Path(env)
        for name in names:
            p = root / name
            if p.is_file():
                return p
    here = Path(__file__).resolve()
    src = here.parents[1] / "modeling" / "tomo2d" / "src"
    for cand in (
        src / "build-tomo2d" / "tt_forward",
        src / "build-tomo2d" / "Release" / "tt_forward.exe",
        src / "build" / "tt_forward",
    ):
        if cand.is_file():
            return cand
    found = shutil.which("tt_forward")
    if found:
        return Path(found)
    return None


def _win_to_wsl(path: Path) -> str:
    p = path.resolve()
    return "/mnt/" + p.drive.rstrip(":").lower() + p.as_posix()[2:]


def _run_tt_forward(exe: Path, args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    if os.name == "nt" and _is_elf(exe):
        wsl = shutil.which("wsl")
        if not wsl:
            pytest.skip("tt_forward 是 Linux ELF，本机没有 wsl")
        qargs = " ".join(f"'{a}'" for a in [_win_to_wsl(exe), *args])
        cmd = [wsl, "-e", "bash", "-lc", f"cd '{_win_to_wsl(cwd)}' && {qargs}"]
        return subprocess.run(cmd, check=False, capture_output=True, text=True)
    return subprocess.run(
        [str(exe), *args], check=False, capture_output=True, text=True, cwd=cwd
    )


def _write_hybrid_smesh(path: Path) -> None:
    xmin, xmax, dx = 0.0, 50.0, 1.0
    zmin, zmax, dz = 0.0, 8.0, 0.25
    xs = [xmin + i * dx for i in range(int(round((xmax - xmin) / dx)) + 1)]
    zs = [zmin + k * dz for k in range(int(round((zmax - zmin) / dz)) + 1)]
    nx, nz = len(xs), len(zs)
    lines = [f"{nx} {nz} 1.5 0.33"]
    lines.append(" ".join(f"{x:.4f}" for x in xs))
    lines.append(" ".join("0.0000" for _ in xs))
    lines.append(" ".join(f"{z:.4f}" for z in zs))
    for _x in xs:
        col = []
        for z in zs:
            if z < CONV_Z - 1e-9:
                v = VP_ABOVE0 + VP_ABOVE_GRAD * z
            else:
                v = VS_BELOW0 + 0.15 * (z - CONV_Z)
            col.append(f"{v:.4f}")
        lines.append(" ".join(col))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_conv(path: Path) -> None:
    xs = [0.0, 25.0, 50.0]
    path.write_text("".join(f"{x:.4f} {CONV_Z:.4f}\n" for x in xs), encoding="utf-8")


def _write_geom(path: Path) -> None:
    path.write_text(
        "1\n"
        "s 10.0 0.5 2\n"
        "r 40.0 0.5 0 0.0 0.05\n"
        "r 40.0 0.5 6 0.0 0.05\n",
        encoding="utf-8",
    )


def _parse_stdout_times(text: str) -> dict[int, float]:
    times: dict[int, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "r":
            times[int(float(parts[3]))] = float(parts[4])
    return times


def _parse_ray_blocks(path: Path) -> list[list[tuple[float, float]]]:
    blocks: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s == ">":
            if cur:
                blocks.append(cur)
                cur = []
            continue
        parts = s.split()
        if len(parts) >= 2:
            cur.append((float(parts[0]), float(parts[1])))
    if cur:
        blocks.append(cur)
    return blocks


@pytest.mark.integration
def test_psp_code6_crosses_slow_halfspace(tmp_path: Path) -> None:
    exe = _find_tt_forward()
    if exe is None:
        pytest.skip("tt_forward 不在 PATH / PYAOBS_TOMO2D_BIN / src/build-tomo2d")

    smesh = tmp_path / "hybrid.smesh"
    conv = tmp_path / "conv.dat"
    geom = tmp_path / "geom.dat"
    rays = tmp_path / "rays.dat"
    _write_hybrid_smesh(smesh)
    _write_conv(conv)
    _write_geom(geom)

    proc = _run_tt_forward(
        exe,
        [
            f"-M{smesh.name}",
            f"-G{geom.name}",
            f"-X{conv.name}",
            f"-R{rays.name}",
            "-g",
        ],
        tmp_path,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"tt_forward failed ({proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    times = _parse_stdout_times(proc.stdout)
    assert 0 in times and 6 in times
    # 联合最短时 t6 ≥ 初至；Vs 不慢于盖层时两者可很接近。
    assert times[6] + 1e-3 >= times[0], times

    blocks = _parse_ray_blocks(rays)
    assert len(blocks) >= 2
    z0 = [z for _x, z in blocks[0]]
    z6 = [z for _x, z in blocks[1]]
    assert max(z6) > CONV_Z + 0.2, max(z6)
    n_below = sum(1 for z in z6 if z > CONV_Z + 1e-3)
    assert n_below >= 3, n_below


@pytest.mark.integration
def test_psp_midline_source_reaches_s_leg(tmp_path: Path) -> None:
    """OBS 在测线中部时 limitRange 不能把 psp_istar 抢到网格边。"""
    exe = _find_tt_forward()
    if exe is None:
        pytest.skip("tt_forward 不在 PATH / PYAOBS_TOMO2D_BIN / src/build-tomo2d")

    smesh = tmp_path / "hybrid.smesh"
    conv = tmp_path / "conv.dat"
    geom = tmp_path / "geom.dat"
    _write_hybrid_smesh(smesh)
    _write_conv(conv)
    geom.write_text(
        "1\n"
        "s 25.0 0.5 2\n"
        "r 40.0 0.5 0 0.0 0.05\n"
        "r 40.0 0.5 6 0.0 0.05\n",
        encoding="utf-8",
    )
    proc = _run_tt_forward(
        exe,
        [f"-M{smesh.name}", f"-G{geom.name}", f"-X{conv.name}", "-g"],
        tmp_path,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"tt_forward failed ({proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    times = _parse_stdout_times(proc.stdout)
    assert 0 in times and 6 in times
    assert times[6] + 1e-3 >= times[0], times
