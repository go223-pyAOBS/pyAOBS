# -*- coding: utf-8 -*-
"""转换流水线：串联现有 CLI（不重写读头逻辑）。

典型链路::

    RAW/OBEM → SAC → SEGY(shot=sx/sy, OBS=gx/gy)[ → SU ]

目标可停在 SEGY 或继续到 SU。各步仍调用::

    raw2sac_v1_1_obspy.py / obem_tsm_to_sac_obspy.py
    sac2y_v2_1_obspy.py
    segy2su（idata SegyDataset.export_su）
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, List, Optional, Sequence

RAW2SAC_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable or "python"

# raw2sac 分量扩展名（无点）
RAW_CHANNEL_EXTS = ("shx", "shy", "shz", "hyd")


def _run(cmd: Sequence[str], *, cwd: Optional[Path] = None) -> None:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        capture_output=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}")


def run_raw2sac(raw_file: Path, sps: str, tc: str, *, cwd: Path) -> List[Path]:
    """在 cwd 下运行 raw2sac，返回生成的分量文件路径。"""
    script = RAW2SAC_DIR / "raw2sac_v1_1_obspy.py"
    before = {p.resolve() for p in cwd.iterdir()} if cwd.is_dir() else set()
    cwd.mkdir(parents=True, exist_ok=True)
    _run([PYTHON, str(script), str(raw_file), str(sps), str(tc)], cwd=cwd)
    produced: List[Path] = []
    for p in sorted(cwd.iterdir()):
        if not p.is_file():
            continue
        if p.resolve() in before:
            continue
        name = p.name.lower()
        if any(name.endswith(ext) for ext in RAW_CHANNEL_EXTS):
            produced.append(p)
    # 若 before 已有同名被覆盖，按扩展名再扫一遍
    if not produced:
        for ext in RAW_CHANNEL_EXTS:
            for p in cwd.glob(f"*{ext}"):
                if p.is_file():
                    produced.append(p)
        produced = sorted(set(produced))
    if not produced:
        raise RuntimeError(f"raw2sac produced no channel files in {cwd}")
    return produced


def run_obem2sac(config_file: Path) -> Path:
    """运行 OBEM→SAC；返回配置中的 output_path。"""
    script = RAW2SAC_DIR / "obem_tsm_to_sac_obspy.py"
    _run([PYTHON, str(script), str(config_file)])
    # 解析 output_path
    out = _ini_get(config_file, "output_path")
    if not out:
        raise RuntimeError(f"no output_path in {config_file}")
    return Path(out).expanduser()


def _ini_get(path: Path, key: str) -> str:
    key_l = key.lower()
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        text = path.read_text(encoding="latin-1")
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith(";") or s.startswith("["):
            continue
        if "=" not in s:
            continue
        k, _, v = s.partition("=")
        if k.strip().lower() == key_l:
            return v.strip()
    return ""


def find_sac_files(directory: Path, *, patterns: Sequence[str] = ("*.sac", "*.SAC", "*.sh?", "*.hyd")) -> List[Path]:
    found: List[Path] = []
    for pat in patterns:
        found.extend(directory.glob(pat))
    # raw2sac 扩展名无点：*.shx 等
    for ext in RAW_CHANNEL_EXTS:
        found.extend(directory.glob(f"*{ext}"))
        found.extend(directory.glob(f"*.{ext}"))
    uniq = sorted({p.resolve() for p in found if p.is_file()})
    return [Path(p) for p in uniq]


def filter_channels(paths: Sequence[Path], channels: Optional[Sequence[str]]) -> List[Path]:
    """channels: None/'all' → 全部；否则匹配扩展名或文件名后缀（如 shz, hyd）。"""
    if not channels or (len(channels) == 1 and str(channels[0]).lower() in ("all", "*")):
        return list(paths)
    want = {c.lower().lstrip(".") for c in channels}
    out: List[Path] = []
    for p in paths:
        name = p.name.lower()
        stem_suf = p.suffix.lower().lstrip(".")
        # raw 风格：无点扩展 或 .shz
        tag = stem_suf if stem_suf else name[-3:]
        if tag in want or name.endswith(tuple(want)):
            out.append(p)
    return out


def run_sac2y(sac: Path, ukooa: Path, segy_out: Path, config: Path, *, cwd: Optional[Path] = None) -> Path:
    script = RAW2SAC_DIR / "sac2y_v2_1_obspy.py"
    segy_out.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [PYTHON, str(script), str(sac), str(ukooa), str(segy_out), str(config)],
        cwd=cwd,
    )
    if not segy_out.is_file():
        raise RuntimeError(f"sac2y did not create {segy_out}")
    return segy_out


def run_segy2su(segy: Path, su_out: Path, *, endian: str = "little") -> Path:
    # 经 idata 服务，避免拉 pygmt
    idata_dir = RAW2SAC_DIR.parent / "idata"
    repo_root = RAW2SAC_DIR.parent.parent.parent
    for p in (str(repo_root), str(RAW2SAC_DIR), str(idata_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from gui.services.segy_dataset import convert_segy_to_su

    return convert_segy_to_su(segy, su_out, endian=endian)


def sac_to_su(
    sac: Path,
    ukooa: Path,
    config: Path,
    su_out: Path,
    *,
    keep_segy: Optional[Path] = None,
    endian: str = "little",
) -> Path:
    """SAC → SEGY → SU。"""
    segy = keep_segy if keep_segy is not None else su_out.with_suffix(".segy")
    run_sac2y(sac, ukooa, segy, config)
    out = run_segy2su(segy, su_out, endian=endian)
    if keep_segy is None and segy.exists() and segy.resolve() != out.resolve():
        try:
            segy.unlink()
        except Exception:
            pass
    return out


def raw_to_su(
    raw_file: Path,
    sps: str,
    tc: str,
    ukooa: Path,
    sac2y_config: Path,
    out_dir: Path,
    *,
    channels: Optional[Sequence[str]] = None,
    keep_sac: bool = True,
    keep_segy: bool = False,
    endian: str = "little",
) -> List[Path]:
    """RAW → SAC(分量) → 各分量 SEGY → SU。"""
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work_sac"
    work.mkdir(parents=True, exist_ok=True)
    sac_files = run_raw2sac(Path(raw_file).resolve(), sps, tc, cwd=work)
    sac_files = filter_channels(sac_files, channels)
    if not sac_files:
        raise RuntimeError("no SAC channels matched filter")
    results: List[Path] = []
    for sac in sac_files:
        tag = sac.suffix.lstrip(".") if sac.suffix else sac.name[-3:]
        stem = f"{Path(raw_file).stem}_{tag}"
        su_path = out_dir / f"{stem}.su"
        segy_path = (out_dir / f"{stem}.segy") if keep_segy else None
        results.append(
            sac_to_su(
                sac,
                ukooa,
                sac2y_config,
                su_path,
                keep_segy=segy_path,
                endian=endian,
            )
        )
    if not keep_sac:
        for p in work.iterdir():
            try:
                p.unlink()
            except Exception:
                pass
    return results


def obem_to_su(
    obem_config: Path,
    ukooa: Path,
    sac2y_config: Path,
    out_dir: Path,
    *,
    channels: Optional[Sequence[str]] = None,
    keep_segy: bool = False,
    endian: str = "little",
    skip_obem: bool = False,
) -> List[Path]:
    """OBEM→SAC（或已有 SAC 目录）→ 各 SAC → SU。"""
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if skip_obem:
        sac_dir = Path(_ini_get(obem_config, "output_path") or "").expanduser()
        if not sac_dir.is_dir():
            raise RuntimeError("skip_obem requires valid output_path in obem config")
    else:
        sac_dir = run_obem2sac(obem_config)
    sac_files = filter_channels(find_sac_files(sac_dir), channels)
    if not sac_files:
        raise RuntimeError(f"no SAC files found in {sac_dir}")
    results: List[Path] = []
    for sac in sac_files:
        stem = sac.stem
        su_path = out_dir / f"{stem}.su"
        segy_path = (out_dir / f"{stem}.segy") if keep_segy else None
        results.append(
            sac_to_su(
                sac,
                ukooa,
                sac2y_config,
                su_path,
                keep_segy=segy_path,
                endian=endian,
            )
        )
    return results


def raw_to_segy(
    raw_file: Path,
    sps: str,
    tc: str,
    ukooa: Path,
    sac2y_config: Path,
    out_dir: Path,
    *,
    channels: Optional[Sequence[str]] = None,
) -> List[Path]:
    """RAW → SAC → SEGY（停在 SEGY）。"""
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work_sac"
    work.mkdir(parents=True, exist_ok=True)
    sac_files = filter_channels(
        run_raw2sac(Path(raw_file).resolve(), sps, tc, cwd=work),
        channels,
    )
    results: List[Path] = []
    for sac in sac_files:
        tag = sac.suffix.lstrip(".") if sac.suffix else sac.name[-3:]
        segy = out_dir / f"{Path(raw_file).stem}_{tag}.segy"
        results.append(run_sac2y(sac, ukooa, segy, sac2y_config))
    return results


def obem_to_segy(
    obem_config: Path,
    ukooa: Path,
    sac2y_config: Path,
    out_dir: Path,
    *,
    channels: Optional[Sequence[str]] = None,
    skip_obem: bool = False,
) -> List[Path]:
    """OBEM→SAC（或已有 SAC）→ SEGY。"""
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if skip_obem:
        sac_dir = Path(_ini_get(obem_config, "output_path") or "").expanduser()
        if not sac_dir.is_dir():
            raise RuntimeError("skip_obem requires valid output_path in obem config")
    else:
        sac_dir = run_obem2sac(obem_config)
    sac_files = filter_channels(find_sac_files(sac_dir), channels)
    if not sac_files:
        raise RuntimeError(f"no SAC files found in {sac_dir}")
    results: List[Path] = []
    for sac in sac_files:
        segy = out_dir / f"{sac.stem}.segy"
        results.append(run_sac2y(sac, ukooa, segy, sac2y_config))
    return results


def sac_to_segy(sac: Path, ukooa: Path, config: Path, segy_out: Path) -> Path:
    """SAC → SEGY（炮=sx/sy，OBS=gx/gy）。"""
    return run_sac2y(sac, ukooa, segy_out, config)
