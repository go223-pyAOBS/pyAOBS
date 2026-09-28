"""tt_inverse 射线文件抽样读取（监视 / 绘制 smesh 共用）。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

RAYS_PREF_KEY = "gui.plot_smesh_rays"
RAYS_ROOT_PREF_KEY = "gui.plot_smesh_ray_root"

_RE_INV_RAY = re.compile(
    r"^(?P<stem>.+)\.ray\.(?P<iter>\d+)\.(?P<isrc>\d+)$", re.IGNORECASE
)

# 与走时预览同系，按 OBS/炮循环着色
_OBS_RAY_PALETTE = (
    "#E60000",
    "#0055FF",
    "#009933",
    "#FF8800",
    "#9900CC",
    "#008888",
    "#FF1493",
    "#886600",
    "#444444",
    "#00AACC",
    "#CC0000",
    "#3366FF",
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
)


def obs_ray_color(isrc: int) -> str:
    """``isrc``（炮/OBS 序号，通常从 1）对应的监视射线颜色。"""
    i = int(isrc) - 1
    if i < 0:
        i = int(isrc)
    return _OBS_RAY_PALETTE[i % len(_OBS_RAY_PALETTE)]


def rays_overlay_enabled(state) -> bool:
    """表单未写该键时默认开启（仍须指定射线路径才会画）。"""
    if state is None:
        return True
    get_bool = getattr(state, "get_bool", None)
    if callable(get_bool):
        return bool(get_bool(RAYS_PREF_KEY, True))
    return True


def set_rays_overlay_enabled(state, on: bool) -> None:
    if state is not None and hasattr(state, "set"):
        state.set(RAYS_PREF_KEY, "1" if on else "0")


def looks_like_ray_name(path: str | Path) -> bool:
    """拖放/粘贴启发：``*.ray.<iter>.<isrc>``、``*.ray``、或 ``rays/`` 目录。"""
    p = Path(path)
    n = p.name.lower()
    if ".smesh" in n or "dws" in n:
        return False
    if n == "rays" or n.endswith(".ray") or ".ray." in n:
        return True
    try:
        if p.is_dir() and any(_RE_INV_RAY.match(c.name) for c in p.iterdir()):
            return True
    except OSError:
        pass
    return False


def parse_inverse_ray_name(path: Path | str) -> tuple[str, int, int] | None:
    """从 ``stem.ray.<iter>.<isrc>`` 解析；失败返回 None。"""
    m = _RE_INV_RAY.match(Path(path).name)
    if not m:
        return None
    return m.group("stem"), int(m.group("iter")), int(m.group("isrc"))


def resolve_ray_hint_path(path: str | Path | None) -> Path | None:
    """规范化后找到可读的射线文件或目录。"""
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
            if c.is_dir():
                return c.resolve()
        except OSError:
            continue
    return None


def _iter_inverse_ray_files_in(d: Path) -> list[Path]:
    out: list[Path] = []
    try:
        if not d.is_dir():
            return out
        for p in d.iterdir():
            if p.is_file() and _RE_INV_RAY.match(p.name):
                out.append(p)
    except OSError:
        return []
    return out


def infer_out_root_from_ray_hint(path: Path | str | None) -> Path | None:
    """从任一 ``.ray`` 文件、``rays/`` 目录或 ``-O`` 前缀推出 ``out_root``。"""
    hit = resolve_ray_hint_path(path)
    if hit is None:
        return None
    if hit.is_file():
        parsed = parse_inverse_ray_name(hit)
        if parsed is None:
            return None
        stem = parsed[0]
        parent = hit.parent
        if parent.name.lower() == "rays":
            return parent.parent / stem
        return parent / stem
    files = _iter_inverse_ray_files_in(hit)
    if not files and (hit / "rays").is_dir():
        files = _iter_inverse_ray_files_in(hit / "rays")
    if not files:
        return None
    parsed = parse_inverse_ray_name(files[0])
    if parsed is None:
        return None
    stem = parsed[0]
    parent = files[0].parent
    if parent.name.lower() == "rays":
        return parent.parent / stem
    return parent / stem


def get_explicit_ray_hint(state, work: Path | None = None) -> Path | None:
    """绘制 smesh 用的用户指定射线路径；未指定则 None。"""
    if state is None or not hasattr(state, "get_str"):
        return None
    s = str(state.get_str(RAYS_ROOT_PREF_KEY) or "").strip().strip('"')
    if not s:
        return None
    p = Path(s)
    cands: list[Path] = [p]
    if work is not None:
        wp = Path(work)
        if not p.is_absolute():
            cands.append(wp / p)
        cands.append(wp / p.name)
        cands.append(wp / "outputs" / "rays" / p.name)
        cands.append(wp / "outputs" / p.name)
        cands.append(wp / "outputs" / "rays")
    for c in cands:
        hit = resolve_ray_hint_path(c)
        if hit is not None:
            return hit
    return None


def set_explicit_ray_hint(
    state, path: str | Path | None, work: Path | None = None
) -> None:
    if state is None or not hasattr(state, "set"):
        return
    if not path:
        state.set(RAYS_ROOT_PREF_KEY, "")
        return
    raw = str(path)
    if work is not None:
        from .paths import to_workdir_relative

        r = to_workdir_relative(raw, work, warn_outside=False)
        raw = r.value or raw
    state.set(RAYS_ROOT_PREF_KEY, raw)


def _downsample_xy(
    xs: list[float], zs: list[float], max_points: int
) -> tuple[list[float], list[float]]:
    if max_points <= 0 or len(xs) <= max_points:
        return xs, zs
    idx = list(range(0, len(xs), max(1, len(xs) // max_points)))
    if idx[-1] != len(xs) - 1:
        idx.append(len(xs) - 1)
    return [xs[i] for i in idx], [zs[i] for i in idx]


def parse_tomo2d_ray_file(
    path: Path | str,
    *,
    max_points_per_ray: int = 250,
) -> list[tuple[list[float], list[float]]]:
    """
    解析 tomo2d 射线文件：``>`` 分隔多条射线，每行 ``x z``
    （``printCurve`` / graph 折线输出）。

    返回 ``[(xs, zs), ...]``；单条过长时均匀抽稀。
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    segments: list[tuple[list[float], list[float]]] = []
    xs: list[float] = []
    zs: list[float] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith(">"):
            if len(xs) >= 2:
                segments.append(_downsample_xy(xs, zs, max_points_per_ray))
            xs, zs = [], []
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            xs.append(float(parts[0]))
            zs.append(float(parts[1]))
        except ValueError:
            continue
    if len(xs) >= 2:
        segments.append(_downsample_xy(xs, zs, max_points_per_ray))
    return segments


def list_inverse_ray_files(out_root: Path | str) -> list[tuple[Path, int, int]]:
    """
    列出 ``{out_root}.ray.<iter>.<isrc>``（扁平或 ``rays/`` 子目录）。
    返回 ``(path, iter, isrc)``，按 iter、isrc 排序。
    """
    root = Path(out_root)
    parent = root.parent if root.parent.as_posix() not in ("", ".") else Path(".")
    stem = root.name
    found: dict[tuple[int, int], Path] = {}
    for d in (parent, parent / "rays"):
        if not d.is_dir():
            continue
        for p in d.glob(f"{stem}.ray.*"):
            m = _RE_INV_RAY.match(p.name)
            if not m:
                continue
            it = int(m.group("iter"))
            isrc = int(m.group("isrc"))
            found[(it, isrc)] = p
    return [(found[k], k[0], k[1]) for k in sorted(found.keys())]


def has_inverse_ray_files(out_root: Path | str) -> bool:
    """是否已有任一 ``{out_root}.ray.<iter>.<isrc>``（找到一个即停）。"""
    root = Path(out_root)
    parent = root.parent if root.parent.as_posix() not in ("", ".") else Path(".")
    stem = root.name
    for d in (parent, parent / "rays"):
        if not d.is_dir():
            continue
        try:
            for p in d.glob(f"{stem}.ray.*"):
                if _RE_INV_RAY.match(p.name):
                    return True
        except OSError:
            continue
    return False


def ray_files_change_stamp(out_root: Path | str) -> str:
    """轻量指纹：``outputs/`` 与 ``outputs/rays/`` 的目录 mtime（不逐文件 stat）。"""
    root = Path(out_root)
    parent = root.parent if root.parent.as_posix() not in ("", ".") else Path(".")
    parts: list[str] = []
    for d in (parent, parent / "rays"):
        if not d.is_dir():
            continue
        try:
            st = d.stat()
            parts.append(f"{d.name}:{int(st.st_mtime)}")
        except OSError:
            continue
    return "|".join(parts)


def _uniform_take(items: list, k: int) -> list:
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return list(items)
    n = len(items)
    idxs = sorted({int(round(i * (n - 1) / (k - 1))) for i in range(k)})
    return [items[i] for i in idxs]


def sample_ray_file_entries(
    entries: Iterable[tuple[Path, int, int]],
    *,
    max_sources: int | None = None,
    iter_prefer: int | None = None,
) -> list[tuple[Path, int, int]]:
    """指定/最大 iter 下的全部炮文件；``max_sources`` 为正时再均匀抽炮。"""
    items = list(entries)
    if not items:
        return []
    if iter_prefer is None:
        iter_prefer = max(it for _, it, _ in items)
    pool = [(p, it, isrc) for p, it, isrc in items if it == iter_prefer]
    if not pool:
        pool = items
    if max_sources is None or int(max_sources) <= 0:
        return pool
    return _uniform_take(pool, int(max_sources))


def load_sampled_rays_for_monitor(
    out_root: Path | str,
    *,
    max_sources: int | None = None,
    iter_prefer: int | None = None,
    max_points_per_ray: int = 120,
    max_segments_total: int = 3000,
    min_segments_per_source: int = 30,
    max_segments_per_source: int = 100,
) -> tuple[list[tuple[int, list[tuple[list[float], list[float]]]]], str]:
    """监视窗用：当前 iter **每个**已写出的 OBS/炮都抽若干条射线。

    返回 ``[(isrc, [(xs, zs), ...]), ...]`` 与状态短注。
    """
    entries = list_inverse_ray_files(out_root)
    picked = sample_ray_file_entries(
        entries, max_sources=max_sources, iter_prefer=iter_prefer
    )
    if not picked:
        return [], "无 .ray 文件（需 -O 且 out_level≥2）"
    nsrc = max(len(picked), 1)
    budget = max(int(max_segments_total), nsrc * int(min_segments_per_source))
    per_src = max(
        int(min_segments_per_source),
        min(int(max_segments_per_source), budget // nsrc),
    )
    groups: list[tuple[int, list[tuple[list[float], list[float]]]]] = []
    for path, _it, isrc in picked:
        try:
            src_segs = parse_tomo2d_ray_file(
                path, max_points_per_ray=max_points_per_ray
            )
        except OSError:
            continue
        src_segs = _uniform_take(src_segs, per_src)
        if src_segs:
            groups.append((int(isrc), src_segs))
    it = picked[0][1]
    n_all = len(picked)
    n_segs = sum(len(s) for _, s in groups)
    note = (
        f"抽样 {len(groups)}/{n_all} 炮 · 每炮≤{per_src} 条"
        f" / iter={it} · 共 {n_segs} 条 · 按 OBS 着色"
    )
    return groups, note


def load_rays_for_smesh_plot(
    hint: Path | str | None,
    *,
    iter_prefer: int | None = None,
    max_points_per_ray: int = 120,
    max_segments_per_source: int = 100,
) -> tuple[list[tuple[int, list[tuple[list[float], list[float]]]]], str]:
    """绘制 smesh：用户指定的 ``.ray`` / ``rays/`` / 单文件正演射线。"""
    if hint is None or not str(hint).strip():
        return [], "未指定射线路径"
    hit = resolve_ray_hint_path(hint)
    if hit is None:
        return [], f"找不到射线文件或目录：\n{hint}"
    root = infer_out_root_from_ray_hint(hit)
    if root is not None and has_inverse_ray_files(root):
        groups, note = load_sampled_rays_for_monitor(
            root,
            iter_prefer=iter_prefer,
            max_points_per_ray=max_points_per_ray,
            max_segments_per_source=max_segments_per_source,
        )
        if groups or iter_prefer is None:
            return groups, note
        return load_sampled_rays_for_monitor(
            root,
            iter_prefer=None,
            max_points_per_ray=max_points_per_ray,
            max_segments_per_source=max_segments_per_source,
        )
    if hit.is_file():
        try:
            segs = parse_tomo2d_ray_file(
                hit, max_points_per_ray=max_points_per_ray
            )
        except OSError as e:
            return [], f"读射线失败：{e}"
        segs = _uniform_take(segs, int(max_segments_per_source))
        if not segs:
            return [], (
                f"已打开 {hit.name}，但没有 '>' 分隔的 x z 折线。\n{hit}"
            )
        note = f"{hit.name} · {len(segs)} 条"
        return [(1, segs)], note
    return [], (
        f"目录里没有 stem.ray.<iter>.<isrc>（需 tt_inverse -o≥2）：\n{hit}"
    )
