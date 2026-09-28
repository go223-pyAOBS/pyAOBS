# -*- coding: utf-8 -*-
"""DWS（Derivative Weight Sum）遮罩：无覆盖留白，有覆盖按 DWS 透明（越大越实）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

DWS_PREF_KEY = "gui.plot_smesh_dws_mask"
DWS_FILE_PREF_KEY = "gui.plot_smesh_dws_file"
DWS_THRESHOLD = 0.0
# 透明遮罩：正 DWS 取分位后做 log 拉伸。线性会把绝大多数格子洗成几乎看不见。
DWS_ALPHA_P_LO = 10.0
DWS_ALPHA_P_HI = 75.0
DWS_ALPHA_MIN = 0.22
DWS_ALPHA_GAMMA = 0.65


def dws_mask_enabled(state) -> bool:
    """表单未写该键时默认开启。"""
    if state is None:
        return True
    get_bool = getattr(state, "get_bool", None)
    if callable(get_bool):
        return bool(get_bool(DWS_PREF_KEY, True))
    return True


def set_dws_mask_enabled(state, on: bool) -> None:
    if state is not None and hasattr(state, "set"):
        state.set(DWS_PREF_KEY, "1" if on else "0")


def looks_like_dws_name(path: str | Path) -> bool:
    """拖放/粘贴时的文件名启发；「指定 DWS…」不依赖此函数，任意文件名均可。"""
    n = Path(path).name.lower()
    if ".smesh" in n:
        return False
    return "dws" in n


def resolve_readable_dws_path(path: str | Path | None) -> Path | None:
    """规范化 WSL/Windows 路径后，找到当前 Python 能打开的文件。"""
    if path is None:
        return None
    raw = str(path).strip().strip('"')
    if not raw:
        return None
    cands: list[Path] = []
    try:
        from .smesh_plot_core import normalize_dropped_path

        cands.append(normalize_dropped_path(raw))
    except Exception:
        pass
    cands.append(Path(raw))
    seen: set[str] = set()
    for c in cands:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        try:
            if c.is_file() and c.stat().st_size > 0:
                return c.resolve()
        except OSError:
            continue
    return None


def _parse_dws_xyz_array(path: Path) -> np.ndarray | None:
    try:
        arr = np.loadtxt(path, usecols=(0, 1, 2), dtype=float, comments="#")
    except Exception:
        arr = None
    if arr is None or np.size(arr) == 0:
        arr = _parse_dws_xyz_loose(path)
    if arr is None or np.size(arr) == 0:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        return None
    if arr.shape[0] < 3:
        return None
    return np.asarray(arr[:, :3], dtype=float)


def _parse_dws_xyz_loose(path: Path) -> np.ndarray | None:
    rows: list[tuple[float, float, float]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for raw in text.splitlines():
        line = raw.strip().strip("\ufeff")
        if not line or line[0] in "#>%":
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 3:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            continue
    if len(rows) < 3:
        return None
    return np.asarray(rows, dtype=float)


def read_dws_xyz(path: str | Path | None) -> tuple[np.ndarray | None, str]:
    """读取 ``printMaskGrid`` 格式（每行 x z 覆盖权重）。文件名不限。"""
    if path is None or not str(path).strip():
        return None, "未指定 DWS 路径"
    shown = str(path)
    p = resolve_readable_dws_path(path)
    if p is None:
        return None, (
            f"找不到文件（不必叫 dws.dat，任意文件名均可）：\n{shown}"
        )
    arr = _parse_dws_xyz_array(p)
    if arr is None:
        return None, (
            f"已打开 {p.name}，但无法解析为每行 x z 覆盖权重（至少三列数字）。\n{p}"
        )
    return arr, ""


def load_dws_xyz(path: str | Path) -> np.ndarray | None:
    """读取 ``printMaskGrid`` 格式：每行 ``x z dws``。失败返回 None。"""
    arr, _err = read_dws_xyz(path)
    return arr


def write_dws_xyz(path: str | Path, xyz: np.ndarray) -> Path:
    """写出 ``x z dws`` 三列，供结果图与再次统计读取。"""
    arr = np.asarray(xyz, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 3:
        raise ValueError("DWS 须为至少 1×3 的 x z 权重")
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        f.write("# x  z  dws\n")
        for row in arr[:, :3]:
            f.write(f"{row[0]:.6f}  {row[1]:.6f}  {row[2]:.6f}\n")
    return dst


def get_explicit_dws_path(state, work: Path | None = None) -> Path | None:
    """绘制 smesh 用的用户指定 DWS；未指定或文件不存在则 None。"""
    if state is None or not hasattr(state, "get_str"):
        return None
    s = str(state.get_str(DWS_FILE_PREF_KEY) or "").strip().strip('"')
    if not s:
        return None
    p = Path(s)
    cands: list[Path] = [p]
    if work is not None:
        wp = Path(work)
        if not p.is_absolute():
            cands.append(wp / p)
        cands.append(wp / p.name)
        cands.append(wp / "outputs" / "dws" / p.name)
        cands.append(wp / "outputs" / p.name)
    for c in cands:
        hit = resolve_readable_dws_path(c)
        if hit is not None:
            return hit
    return None


def set_explicit_dws_path(
    state, path: str | Path | None, work: Path | None = None
) -> None:
    if state is None or not hasattr(state, "set"):
        return
    if not path:
        state.set(DWS_FILE_PREF_KEY, "")
        return
    raw = str(path)
    if work is not None:
        from .paths import to_workdir_relative

        r = to_workdir_relative(raw, work, warn_outside=False)
        raw = r.value or raw
    state.set(DWS_FILE_PREF_KEY, raw)


def dws_xyz_for_explicit(
    state, path: str | Path | None
) -> np.ndarray | None:
    """仅用用户给出的路径；勾选关闭或路径无效则不遮罩。"""
    if not dws_mask_enabled(state) or not path:
        return None
    return load_dws_xyz(path)


def resolve_plot_dws_for_smesh(
    smesh_path: str | Path | None,
    state,
    work: Path,
    *,
    out_root: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> Path | None:
    """为单个 smesh 找 DWS。

    优先顺序：
    1. 与 smesh **同一目录**（用户自选模型常见：dws 和 smesh 放一起）
    2. GUI tt_inverse 运行包 ``outputs/dws/``（以及旧式 ``outputs/dws.dat``）
    3. 调用方给出的 ``run_dir`` / ``out_root``
    4. 表单 ``inv.dws_file``（最后才用，避免集合里所有模型共用一份）
    """
    files: list[Path] = []
    dirs: list[Path] = []
    seen_f: set[str] = set()
    seen_d: set[str] = set()

    def add_file(p: Path | str | None) -> None:
        if p is None:
            return
        path = Path(p)
        key = str(path)
        if key in seen_f:
            return
        seen_f.add(key)
        files.append(path)

    def add_dir(p: Path | str | None) -> None:
        if p is None:
            return
        path = Path(p)
        key = str(path)
        if key in seen_d:
            return
        seen_d.add(key)
        dirs.append(path)

    def add_outputs_dws(outputs: Path | None) -> None:
        if outputs is None:
            return
        add_file(outputs / "dws" / "dws.dat")
        add_dir(outputs / "dws")
        add_file(outputs / "dws.dat")

    smesh = Path(smesh_path) if smesh_path else None
    if smesh is not None:
        here = smesh.parent
        add_file(here / "dws.dat")
        add_dir(here)
        add_dir(here / "dws")
        if here.name.lower() == "models":
            add_outputs_dws(here.parent)
        elif here.name.lower() == "outputs":
            add_outputs_dws(here)

        # 蒙特卡洛单次：smesh 在 reals/iii/out[/models]，DWS 归类到 reals/iii/dws/
        for anc in (here, *list(here.parents)):
            try:
                parent_name = anc.parent.name
            except Exception:
                break
            if parent_name == "reals":
                add_dir(anc / "dws")
                add_file(anc / "dws.dat")
                add_dir(anc)
                break
            if parent_name == "runs":
                break

        inferred = None
        if run_dir is None:
            try:
                from .result_nav import infer_run_dir_from_smesh

                inferred = infer_run_dir_from_smesh(smesh)
            except Exception:
                inferred = None
        rd = Path(run_dir) if run_dir else inferred
        if rd is not None:
            add_outputs_dws(rd / "outputs")
            add_dir(rd / "outputs")

    if out_root:
        root = Path(out_root)
        add_outputs_dws(root.parent)
        add_outputs_dws(root.parent / "outputs")
    if run_dir:
        rd = Path(run_dir)
        add_outputs_dws(rd / "outputs")
        add_dir(rd / "outputs")
        add_dir(rd / "dws")

    for f in files:
        hit = resolve_readable_dws_path(f)
        if hit is not None:
            return hit
    for d in dirs:
        hit = _pick_dws_in_dir(d)
        if hit is not None:
            return hit

    if state is not None and hasattr(state, "get_str"):
        s = str(state.get_str("inv.dws_file") or "").strip()
        if s:
            p = Path(s)
            full = p if p.is_absolute() else (Path(work) / p)
            for cand in (
                full,
                Path(work) / "outputs" / "dws" / p.name,
                Path(work) / "outputs" / p.name,
            ):
                hit = resolve_readable_dws_path(cand)
                if hit is not None:
                    return hit
    return None


def describe_dws_for_smeshes(
    smesh_paths: list[str | Path],
    state,
    work: Path,
    *,
    out_root: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> list[tuple[Path, Path | None]]:
    """每个 smesh 解析到的 DWS 路径（未找到则为 None）。"""
    out: list[tuple[Path, Path | None]] = []
    for raw in smesh_paths:
        p = Path(raw)
        hit = resolve_plot_dws_for_smesh(
            p, state, work, out_root=out_root, run_dir=run_dir
        )
        out.append((p, hit))
    return out


def format_dws_match_report(
    pairs: list[tuple[Path, Path | None]],
    *,
    work: Path | None = None,
) -> str:
    """给人看的 smesh → DWS 对照。"""
    if not pairs:
        return "（没有 smesh）"
    lines: list[str] = []
    n_hit = sum(1 for _s, d in pairs if d is not None)
    uniq: set[str] = set()
    for _s, d in pairs:
        if d is None:
            continue
        try:
            uniq.add(str(d.resolve()))
        except OSError:
            uniq.add(str(d))
    lines.append(f"找到 DWS {n_hit}/{len(pairs)}（{len(uniq)} 个不同文件）")
    lines.append("优先：与 smesh 同目录，其次该次运行 outputs/dws/。")
    lines.append("")
    wp = Path(work) if work is not None else None
    for smesh, dws in pairs:
        s_show = smesh.name
        try:
            if wp is not None:
                s_show = str(smesh.resolve().relative_to(wp.resolve()))
        except Exception:
            s_show = str(smesh)
        if dws is None:
            lines.append(f"{s_show}")
            lines.append("  → 未找到")
            continue
        d_show = str(dws)
        try:
            if wp is not None:
                d_show = str(dws.resolve().relative_to(wp.resolve()))
        except Exception:
            pass
        lines.append(f"{s_show}")
        lines.append(f"  → {d_show}")
    return "\n".join(lines)


def _pick_dws_in_dir(d: Path) -> Path | None:
    """优先 ``dws.dat``；``dws/`` 目录内也接受其它非重力数据文件。"""
    try:
        if not d.is_dir():
            return resolve_readable_dws_path(d)
    except OSError:
        return None
    preferred = resolve_readable_dws_path(d / "dws.dat")
    if preferred is not None:
        return preferred
    try:
        names = list(d.iterdir())
    except OSError:
        return None
    in_dws_folder = d.name.lower() == "dws"
    named: list[Path] = []
    others: list[Path] = []
    for p in names:
        n = p.name.lower()
        if "grav" in n or ".smesh" in n:
            continue
        if looks_like_dws_name(p):
            named.append(p)
        elif in_dws_folder:
            others.append(p)
    named.sort(key=lambda p: p.name.lower())
    others.sort(key=lambda p: p.name.lower())
    for p in named + others:
        hit = resolve_readable_dws_path(p)
        if hit is not None:
            return hit
    return None


def dws_watch_stamp(
    smesh_path: str | Path | None,
    work: Path,
    *,
    out_root: str | Path | None = None,
    run_dir: str | Path | None = None,
    state=None,
) -> str:
    """监视用指纹：DWS 文件出现/更新后会变（不逐行读）。"""
    hit = resolve_plot_dws_for_smesh(
        smesh_path, state, work, out_root=out_root, run_dir=run_dir
    )
    if hit is not None:
        try:
            st = hit.stat()
            return f"{hit}:{int(st.st_mtime)}:{int(st.st_size)}"
        except OSError:
            return str(hit)
    parts: list[str] = []
    roots: list[Path] = []
    if out_root:
        roots.append(Path(out_root).parent)
    if run_dir:
        roots.append(Path(run_dir) / "outputs")
    if smesh_path:
        roots.append(Path(smesh_path).parent)
        roots.append(Path(smesh_path).parent.parent)
    for base in roots:
        for d in (base / "dws", base):
            try:
                if d.is_dir():
                    st = d.stat()
                    parts.append(f"{d.name}:{int(st.st_mtime)}")
            except OSError:
                continue
    return "none|" + "|".join(parts)


def dws_xyz_for_plot(
    state,
    work: Path,
    smesh_path: str | Path | None,
    *,
    out_root: str | Path | None = None,
    run_dir: str | Path | None = None,
    enabled: bool | None = None,
) -> np.ndarray | None:
    """反演监视 / 模型挑选：勾选遮罩时自动就近找 DWS。"""
    xyz, _path, _note = load_plot_dws(
        state,
        work,
        smesh_path,
        out_root=out_root,
        run_dir=run_dir,
        enabled=enabled,
    )
    return xyz


def _load_dws_xyz_arrays(
    state,
    work: Path,
    smesh_paths: list[str | Path],
    *,
    enabled: bool | None = None,
) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for raw in smesh_paths:
        xyz = dws_xyz_for_plot(state, work, raw, enabled=enabled)
        if xyz is None:
            continue
        arr = np.asarray(xyz, dtype=float)
        if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] >= 3:
            arrays.append(arr[:, :3])
    return arrays


def _align_dws_weight_cols(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    """对齐到第一份 (x,z)，返回 (ref_xyz, weights) 其中 weights 形如 (n, nnode)。"""
    if not arrays:
        return None
    ref = arrays[0]
    cols = [ref[:, 2]]
    for arr in arrays[1:]:
        if arr.shape[0] == ref.shape[0] and np.allclose(
            arr[:, :2], ref[:, :2], atol=1e-3, rtol=0.0
        ):
            cols.append(arr[:, 2])
            continue
        try:
            from scipy.interpolate import NearestNDInterpolator

            interp = NearestNDInterpolator(arr[:, :2], arr[:, 2])
            cols.append(np.asarray(interp(ref[:, 0], ref[:, 1]), dtype=float))
        except Exception:
            continue
    return ref.copy(), np.stack(cols, axis=0)


def union_dws_xyz_for_smeshes(
    state,
    work: Path,
    smesh_paths: list[str | Path],
    *,
    enabled: bool | None = None,
) -> tuple[np.ndarray | None, int]:
    """各模型 DWS 并集：同一 (x, z) 取覆盖最大值（任一处有覆盖即显示）。"""
    aligned = _align_dws_weight_cols(
        _load_dws_xyz_arrays(state, work, smesh_paths, enabled=enabled)
    )
    if aligned is None:
        return None, 0
    ref, cols = aligned
    ref[:, 2] = np.nanmax(cols, axis=0)
    return ref, int(cols.shape[0])


def intersect_dws_xyz_for_smeshes(
    state,
    work: Path,
    smesh_paths: list[str | Path],
    *,
    enabled: bool | None = None,
) -> tuple[np.ndarray | None, int]:
    """差值用交集：两侧都有覆盖才显示；权利用逐点 min（受更弱的一侧限制）。

    缺一侧 DWS 文件时无法判断交集，返回 ``(None, n_loaded)``。
    """
    arrays = _load_dws_xyz_arrays(state, work, smesh_paths, enabled=enabled)
    n = len(arrays)
    if n < 2:
        return None, n
    aligned = _align_dws_weight_cols(arrays)
    if aligned is None:
        return None, n
    ref, cols = aligned
    w = np.array(cols, dtype=float, copy=True)
    w = np.where(np.isnan(w), 0.0, w)
    w = np.where(np.isfinite(w) & (w <= 0.0), 0.0, w)
    ref[:, 2] = np.min(w, axis=0)
    return ref, n


def mean_dws_xyz_from_arrays(
    arrays: list,
) -> tuple[np.ndarray | None, int]:
    """由已读入的 DWS 数组算各点覆盖均值（DWS>0 才计入）。"""
    cleaned: list[np.ndarray] = []
    for xyz in arrays:
        if xyz is None:
            continue
        arr = np.asarray(xyz, dtype=float)
        if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] >= 3:
            cleaned.append(arr[:, :3])
    aligned = _align_dws_weight_cols(cleaned)
    if aligned is None:
        return None, 0
    ref, cols = aligned
    w = np.where(np.isfinite(cols) & (cols <= 0.0), np.nan, cols)
    valid = ~np.isnan(w)
    n = np.sum(valid, axis=0)
    s = np.nansum(w, axis=0)
    m = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    ref[:, 2] = np.where(np.isnan(m), 0.0, m)
    return ref, int(cols.shape[0])


def mean_dws_xyz_for_smeshes(
    state,
    work: Path,
    smesh_paths: list[str | Path],
    *,
    enabled: bool | None = None,
) -> tuple[np.ndarray | None, int]:
    """与均值速度同一套：各点只平均 DWS>0 的成员（无覆盖不拉低）。"""
    return mean_dws_xyz_from_arrays(
        _load_dws_xyz_arrays(state, work, smesh_paths, enabled=enabled)
    )


def load_plot_dws(
    state,
    work: Path,
    smesh_path: str | Path | None,
    *,
    out_root: str | Path | None = None,
    run_dir: str | Path | None = None,
    enabled: bool | None = None,
) -> tuple[np.ndarray | None, Path | None, str]:
    """返回 ``(xyz, path, note)``；勾选关闭或找不到时 xyz 为 None。"""
    on = dws_mask_enabled(state) if enabled is None else bool(enabled)
    if not on:
        return None, None, ""
    hit = resolve_plot_dws_for_smesh(
        smesh_path, state, work, out_root=out_root, run_dir=run_dir
    )
    if hit is None:
        return None, None, "未找到 DWS（tt_inverse -K 在反演结束才写出）"
    xyz, err = read_dws_xyz(hit)
    if xyz is None:
        return None, hit, err or f"读不了 {hit.name}"
    return xyz, hit, ""


def _dws_from_smesh_nodes(
    xx: np.ndarray,
    zz: np.ndarray,
    xyz: np.ndarray,
    mesh: Any,
) -> np.ndarray | None:
    """DWS 与 smesh 同序（i 外 k 内）时，按节点网格取覆盖权重。"""
    xpos = getattr(mesh, "xpos", None)
    zpos = getattr(mesh, "zpos", None)
    topo = getattr(mesh, "topo", None)
    if xpos is None or zpos is None or topo is None:
        return None
    xpos = np.asarray(xpos, dtype=float)
    zpos = np.asarray(zpos, dtype=float)
    topo = np.asarray(topo, dtype=float)
    if xpos.size < 2 or zpos.size < 2 or topo.size != xpos.size:
        return None
    nx, nz = int(xpos.size), int(zpos.size)
    if xyz.shape[0] != nx * nz:
        return None
    dws_grid = np.asarray(xyz[:, 2], dtype=float).reshape(nx, nz)
    gx, gz = np.meshgrid(xx, zz)
    topo_x = np.interp(xx, xpos, topo)
    i = np.abs(xpos[:, np.newaxis] - xx[np.newaxis, :]).argmin(axis=0)
    rel = gz - topo_x[np.newaxis, :]
    k = np.searchsorted(zpos, rel, side="left")
    k = np.clip(k, 1, nz - 1)
    left = k - 1
    nearer_right = np.abs(zpos[k] - rel) < np.abs(zpos[left] - rel)
    k = np.where(nearer_right, k, left)
    dws = dws_grid[np.broadcast_to(i, k.shape), k]
    below = gz >= topo_x[np.newaxis, :]
    dws = np.where(below, dws, np.inf)
    return np.asarray(dws, dtype=float)


def dws_grid_for_plot(
    x: np.ndarray,
    z: np.ndarray,
    dws_xyz: np.ndarray,
    *,
    mesh: Any = None,
) -> np.ndarray | None:
    """把 DWS 落到 ``(nz, nx)`` 绘图网格。水层为 ``+inf``（保持实色）；失败返回 None。"""
    xyz = np.asarray(dws_xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[0] < 3 or xyz.shape[1] < 3:
        return None
    xx = np.asarray(x, dtype=float)
    zz = np.asarray(z, dtype=float)
    dws = None
    if mesh is not None:
        dws = _dws_from_smesh_nodes(xx, zz, xyz, mesh)
    if dws is None:
        try:
            from scipy.interpolate import NearestNDInterpolator
        except ImportError:
            return None
        interp = NearestNDInterpolator(xyz[:, :2], xyz[:, 2])
        gx, gz = np.meshgrid(xx, zz)
        dws = np.asarray(interp(gx, gz), dtype=float)
        if mesh is not None and getattr(mesh, "xpos", None) is not None and getattr(
            mesh, "topo", None
        ) is not None:
            topo = np.interp(
                xx,
                np.asarray(mesh.xpos, dtype=float),
                np.asarray(mesh.topo, dtype=float),
            )
            below = gz >= topo[np.newaxis, :]
            dws = np.where(below, dws, np.inf)
    if dws is None:
        return None
    if dws.shape != (zz.size, xx.size):
        return None
    return np.asarray(dws, dtype=float)


def alpha_from_dws(
    dws: np.ndarray,
    *,
    p_lo: float = DWS_ALPHA_P_LO,
    p_hi: float = DWS_ALPHA_P_HI,
    alpha_min: float = DWS_ALPHA_MIN,
    gamma: float = DWS_ALPHA_GAMMA,
) -> np.ndarray:
    """
    DWS → 不透明度。

    - 水层（``+inf`` / 非有限）：α=1
    - DWS≤0：α=0（无覆盖留白）
    - DWS>0：``log10`` 拉伸到本图正值的 ``p_lo``–``p_hi`` 分位，再映到
      ``[alpha_min, 1]``；``gamma<1`` 略抬中等覆盖。
    """
    grid = np.asarray(dws, dtype=float)
    alpha = np.ones(grid.shape, dtype=float)
    uncovered = np.isfinite(grid) & (grid <= 0.0)
    alpha[uncovered] = 0.0
    pos = np.isfinite(grid) & (grid > 0.0)
    if not np.any(pos):
        return alpha
    vals = grid[pos]
    vlo = float(np.percentile(vals, p_lo))
    vhi = float(np.percentile(vals, p_hi))
    vlo = max(vlo, 1e-30)
    if vhi <= vlo:
        alpha[pos] = 1.0
        return alpha
    t = (np.log10(np.clip(vals, vlo, vhi)) - np.log10(vlo)) / (
        np.log10(vhi) - np.log10(vlo)
    )
    t = np.clip(t, 0.0, 1.0)
    g = float(gamma) if float(gamma) > 0.0 else 1.0
    if g != 1.0:
        t = np.power(t, g)
    floor = float(np.clip(alpha_min, 0.0, 1.0))
    alpha[pos] = floor + (1.0 - floor) * t
    return alpha


def mask_velocity_with_dws(
    data: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    dws_xyz: np.ndarray,
    *,
    mesh: Any = None,
    threshold: float = DWS_THRESHOLD,
) -> np.ndarray:
    """``data`` 为 (nz, nx)。地形以下且 DWS≤threshold 处置 NaN；水层/空气仍着色。"""
    out = np.array(data, dtype=float, copy=True)
    xx = np.asarray(x, dtype=float)
    zz = np.asarray(z, dtype=float)
    if out.ndim != 2 or out.shape != (zz.size, xx.size):
        return out
    dws = dws_grid_for_plot(xx, zz, dws_xyz, mesh=mesh)
    if dws is None:
        return out
    weak = np.isfinite(dws) & (dws <= float(threshold))
    if not np.any(weak):
        return out
    out[weak] = np.nan
    return out


def mask_dataset_velocity(ds: Any, mesh: Any, dws_xyz: np.ndarray) -> Any:
    """返回 velocity 已遮罩的 Dataset 副本。"""
    data = np.asarray(ds["velocity"].values, dtype=float)
    x = np.asarray(ds["x"].values, dtype=float)
    z = np.asarray(ds["z"].values, dtype=float)
    transposed = data.ndim == 2 and data.shape == (x.size, z.size)
    arr = data.T if transposed else data
    masked = mask_velocity_with_dws(arr, x, z, dws_xyz, mesh=mesh)
    out = ds.copy(deep=True)
    if transposed:
        out["velocity"].values[...] = masked.T
    else:
        out["velocity"].values[...] = masked
    return out


def cmap_blank_uncovered(cmap):
    """无覆盖（NaN / mask）处画白，避免沿用 jet 的边缘色。"""
    try:
        cm = cmap.copy()
    except Exception:
        cm = cmap
    try:
        cm.set_bad(color="#ffffff", alpha=1.0)
    except Exception:
        try:
            cm.set_bad("#ffffff")
        except Exception:
            pass
    return cm


def masked_for_imshow(data: np.ndarray) -> np.ma.MaskedArray:
    """imshow 用 masked array，避免 nearest 把 NaN 涂成邻格颜色。"""
    arr = np.asarray(data, dtype=float)
    return np.ma.masked_invalid(arr)
