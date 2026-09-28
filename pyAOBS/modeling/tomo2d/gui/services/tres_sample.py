"""tt_inverse 走时残差 ``.tres`` 读取（监视拟合图用）。"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

_RE_INV_TRES = re.compile(
    r"^(?P<stem>.+)\.tres\.(?P<iter>\d+)\.(?P<isrc>\d+)$", re.IGNORECASE
)


# 与 ttimes / tt_inverse 一致：0 折射、1 反射；缺列或无法解析为 -1。
RAYTYPE_UNKNOWN = -1
RAYTYPE_REFR = 0
RAYTYPE_REFL = 1


def parse_tomo2d_tres_file(
    path: Path | str,
    *,
    max_points: int = 800,
) -> tuple[list[float], list[float], list[int]]:
    """
    解析 ``{out}.tres.<iter>.<isrc>``。

    每行 ``rcv_x  residual [raytype]``：残差为观测−计算（秒）；
    第三列可选，0=折射、1=反射。有第三列时拟合图优先用它；旧两列 ``raytype=-1``。
    点数过多时均匀抽稀。
    """
    xs: list[float] = []
    rs: list[float] = []
    codes: list[int] = []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return xs, rs, codes
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or s.startswith(">"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            r = float(parts[1])
        except ValueError:
            continue
        code = RAYTYPE_UNKNOWN
        if len(parts) >= 3:
            try:
                code = int(float(parts[2]))
            except ValueError:
                code = RAYTYPE_UNKNOWN
        xs.append(x)
        rs.append(r)
        codes.append(code)
    if max_points > 0 and len(xs) > max_points:
        n = len(xs)
        idxs = sorted({int(round(i * (n - 1) / (max_points - 1))) for i in range(max_points)})
        xs = [xs[i] for i in idxs]
        rs = [rs[i] for i in idxs]
        codes = [codes[i] for i in idxs]
    return xs, rs, codes


def raytypes_by_src_from_data(path: Path | str) -> dict[int, list[int]]:
    """
    读 tomo2d ``-G`` 数据（s/r）：``isrc`` 从 1 起，每炮接收点顺序与 ``.tres`` 行序一致。
    """
    from .tt_plot_data import load_ttimes_picks

    out: dict[int, list[int]] = {}
    try:
        picks = load_ttimes_picks(path)
    except (OSError, ValueError):
        return out
    for p in picks:
        isrc = int(p.shot_idx) + 1
        out.setdefault(isrc, []).append(int(p.code))
    return out


def attach_raytypes_from_data(
    xs: list[float],
    codes: list[int],
    data_codes: list[int] | None,
) -> list[int]:
    """旧两列 ``.tres``：按行序贴上 ``-G`` 的 raytype；长度不一致则保持未知。"""
    if not xs or not data_codes:
        return codes
    if any(c != RAYTYPE_UNKNOWN for c in codes):
        return codes
    if len(data_codes) != len(xs):
        return codes
    return [int(c) for c in data_codes]


def normalize_tres_raytypes(codes: list[int]) -> list[int]:
    """tomo2d 为 0 折射 / 1 反射；若文件只有 1 与 2，则按 1→折射、2→反射。"""
    uniq = {int(c) for c in codes if int(c) != RAYTYPE_UNKNOWN}
    if not uniq or RAYTYPE_REFR in uniq:
        return [int(c) for c in codes]
    if 1 in uniq and 2 in uniq:
        out: list[int] = []
        for c in codes:
            ic = int(c)
            if ic == 1:
                out.append(RAYTYPE_REFR)
            elif ic == 2:
                out.append(RAYTYPE_REFL)
            else:
                out.append(ic)
        return out
    return [int(c) for c in codes]


def find_run_data_file(
    run_dir: Path | str | None = None,
    *,
    out_root: Path | str | None = None,
) -> Path | None:
    """运行包 ``inputs/data.dat``，或从 ``manifest.json`` / ``out_root`` 上推。"""
    roots: list[Path] = []
    if run_dir:
        roots.append(Path(run_dir))
    if out_root:
        root = Path(out_root)
        parent = root.parent
        roots.append(parent)
        if parent.name.lower() == "outputs":
            roots.append(parent.parent)
        if (parent / "manifest.json").is_file():
            roots.append(parent)
        if (parent.parent / "manifest.json").is_file():
            roots.append(parent.parent)
    seen: set[Path] = set()
    for rd in roots:
        try:
            key = rd.resolve()
        except OSError:
            key = rd
        if key in seen:
            continue
        seen.add(key)
        for rel in (
            "inputs/data.dat",
            "inputs/ttimes.dat",
            "data.dat",
            "ttimes.dat",
            "geom.dat",
        ):
            p = rd / rel
            if p.is_file():
                return p
        man = rd / "manifest.json"
        if not man.is_file():
            continue
        try:
            obj = json.loads(man.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(obj, dict):
            continue
        replay = obj.get("python_replay") or {}
        rel = replay.get("data") if isinstance(replay, dict) else None
        if rel:
            p = rd / str(rel)
            if p.is_file():
                return p
        for inp in obj.get("inputs") or []:
            if not isinstance(inp, dict) or inp.get("role") != "data":
                continue
            p = rd / str(inp.get("path_in_run") or "")
            if p.is_file():
                return p
    return None


def out_root_from_inverse_smesh(smesh: Path | str) -> Path:
    """``dir/foo.smesh.3.0`` → ``dir/foo``（与 ``-O`` 前缀一致）。"""
    p = Path(smesh)
    name = p.name
    low = name.lower()
    if ".smesh." in low:
        i = low.index(".smesh.")
        return p.parent / name[:i]
    return p.parent / p.stem


def tres_out_root_candidates(
    *,
    smesh: Path | str | None = None,
    log: Path | str | None = None,
    spec=None,
) -> list[Path]:
    """绘制模型时查找 ``.tres`` 的 -O 候选（运行包、smesh 前缀、所在目录）。"""
    cands: list[Path] = []
    out = getattr(spec, "out_root", None) if spec is not None else None
    if out is not None:
        cands.append(Path(out))
    if smesh is not None:
        sp = Path(smesh)
        cands.append(out_root_from_inverse_smesh(sp))
        cands.append(sp.parent)
    if log is not None:
        cands.append(Path(log).parent)
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in cands:
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def _subsample_tres(
    xs: list[float], rs: list[float], codes: list[int], max_points: int
) -> tuple[list[float], list[float], list[int]]:
    if max_points <= 0 or len(xs) <= max_points:
        return xs, rs, codes
    n = len(xs)
    idxs = sorted({int(round(i * (n - 1) / (max_points - 1))) for i in range(max_points)})
    return [xs[i] for i in idxs], [rs[i] for i in idxs], [codes[i] for i in idxs]


def list_inverse_tres_files(out_root: Path | str) -> list[tuple[Path, int, int]]:
    """
    列出 ``{out_root}.tres.<iter>.<isrc>``（扁平或 ``residuals/``）。
    若 ``out_root`` 是目录，再扫该目录内任意 ``*.tres.<iter>.<isrc>``。
    返回 ``(path, iter, isrc)``，按 iter、isrc 排序。
    """
    root = Path(out_root)
    parent = root.parent if root.parent.as_posix() not in ("", ".") else Path(".")
    stem = root.name
    found: dict[tuple[int, int], Path] = {}

    def _absorb(paths) -> None:
        for p in paths:
            m = _RE_INV_TRES.match(p.name)
            if not m:
                continue
            found[(int(m.group("iter")), int(m.group("isrc")))] = p

    for d in (parent, parent / "residuals"):
        if not d.is_dir():
            continue
        try:
            _absorb(d.glob(f"{stem}.tres.*"))
        except OSError:
            continue
    try:
        if root.is_dir():
            for d in (root, root / "residuals"):
                if not d.is_dir():
                    continue
                _absorb(d.glob("*.tres.*"))
    except OSError:
        pass
    return [(found[k], k[0], k[1]) for k in sorted(found.keys())]


def has_inverse_tres_files(out_root: Path | str) -> bool:
    """是否已有任一 ``{out_root}.tres.<iter>.<isrc>``（含目录内任意前缀）。"""
    return bool(list_inverse_tres_files(out_root))


def tres_files_change_stamp(out_root: Path | str) -> str:
    """轻量指纹：``outputs/`` 与 ``outputs/residuals/`` 的目录 mtime。"""
    root = Path(out_root)
    parent = root.parent if root.parent.as_posix() not in ("", ".") else Path(".")
    parts: list[str] = []
    for d in (parent, parent / "residuals"):
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


def sample_tres_file_entries(
    entries: Iterable[tuple[Path, int, int]],
    *,
    iter_prefer: int | None = None,
) -> tuple[list[tuple[Path, int, int]], int | None]:
    """指定 iter 的全部炮文件；若该轮没有，则退到已有最大 iter。"""
    items = list(entries)
    if not items:
        return [], None
    latest = max(it for _, it, _ in items)
    use = latest if iter_prefer is None else int(iter_prefer)
    pool = [(p, it, isrc) for p, it, isrc in items if it == use]
    if not pool:
        use = latest
        pool = [(p, it, isrc) for p, it, isrc in items if it == use]
    return pool, use


def load_tres_for_monitor(
    out_root: Path | str,
    *,
    iter_prefer: int | None = None,
    max_points_per_src: int = 800,
    run_dir: Path | str | None = None,
    data_file: Path | str | None = None,
) -> tuple[list[tuple[int, list[float], list[float], list[int]]], str]:
    """监视拟合图：当前/指定 iter 各炮的 ``(isrc, xs, residuals, raytypes)`` 与状态短注。"""
    entries = list_inverse_tres_files(out_root)
    picked, used_iter = sample_tres_file_entries(entries, iter_prefer=iter_prefer)
    if not picked:
        return [], "无 .tres（需 -O 且 out_level≥1）"
    data_path = Path(data_file) if data_file else None
    if data_path is not None and not data_path.is_file():
        data_path = None
    if data_path is None:
        data_path = find_run_data_file(run_dir, out_root=out_root)
    codes_by_src = raytypes_by_src_from_data(data_path) if data_path else {}
    n_from_data = 0
    groups: list[tuple[int, list[float], list[float], list[int]]] = []
    n_pts = 0
    n_known = 0
    sum_sq = 0.0
    for path, _it, isrc in picked:
        xs, rs, codes = parse_tomo2d_tres_file(path, max_points=0)
        if not xs:
            continue
        # 首选 .tres 第三列（0=折射、1=反射）；仅全未知时才用旧规则从 -G 补列。
        if any(int(c) != RAYTYPE_UNKNOWN for c in codes):
            codes = normalize_tres_raytypes(codes)
        else:
            attached = attach_raytypes_from_data(xs, codes, codes_by_src.get(int(isrc)))
            if attached != codes:
                n_from_data += 1
            codes = normalize_tres_raytypes(attached)
        xs, rs, codes = _subsample_tres(xs, rs, codes, max_points_per_src)
        groups.append((int(isrc), xs, rs, codes))
        n_pts += len(rs)
        n_known += sum(1 for c in codes if c != RAYTYPE_UNKNOWN)
        for v in rs:
            sum_sq += v * v
    if not groups:
        return [], f"iter={used_iter} 的 .tres 为空"
    rms = math.sqrt(sum_sq / n_pts) if n_pts else 0.0
    fallback = (
        iter_prefer is not None
        and used_iter is not None
        and int(iter_prefer) != int(used_iter)
    )
    if n_known:
        color_note = "上=折射 / 下=反射"
        if n_from_data:
            color_note = f"{color_note}（震相由 -G 补列）"
        else:
            color_note = f"{color_note}（.tres 第三列）"
    else:
        color_note = "旧 .tres 无震相列，按 OBS 着色"
    note = (
        f"残差 {len(groups)} 炮 · {n_pts} 点 · iter={used_iter}"
        f" · RMS={rms:.4g} s · {color_note}"
    )
    if fallback:
        note = f"{note}（模型 iter={iter_prefer} 尚无 .tres，用已有轮）"
    return groups, note


_RE_INV_OUTLIERS = re.compile(
    r"^(?P<stem>.+)\.outliers\.(?P<iter>\d+)\.(?P<iset>\d+)$", re.IGNORECASE
)


def parse_outlier_file(path: Path | str) -> list[tuple[int, int, float, float, int, float, float]]:
    """
    解析 ``{out}.outliers.<iter>.<iset>``。

    每行 ``isrc ircv src_x rcv_x raytype tres_s lin_res``。
    返回上述七元组列表。
    """
    rows: list[tuple[int, int, float, float, int, float, float]] = []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return rows
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 7:
            continue
        try:
            rows.append(
                (
                    int(float(parts[0])),
                    int(float(parts[1])),
                    float(parts[2]),
                    float(parts[3]),
                    int(float(parts[4])),
                    float(parts[5]),
                    float(parts[6]),
                )
            )
        except ValueError:
            continue
    return rows


def list_inverse_outlier_files(out_root: Path | str) -> list[tuple[Path, int, int]]:
    """列出 ``{out}.outliers.<iter>.<iset>``（扁平或 ``residuals/``）。"""
    root = Path(out_root)
    parent = root.parent if root.parent.as_posix() not in ("", ".") else Path(".")
    stem = root.name
    found: dict[tuple[int, int], Path] = {}

    def _absorb(paths) -> None:
        for p in paths:
            m = _RE_INV_OUTLIERS.match(p.name)
            if not m:
                continue
            found[(int(m.group("iter")), int(m.group("iset")))] = p

    for d in (parent, parent / "residuals"):
        if not d.is_dir():
            continue
        try:
            _absorb(d.glob(f"{stem}.outliers.*"))
        except OSError:
            continue
    try:
        if root.is_dir():
            for d in (root, root / "residuals"):
                if not d.is_dir():
                    continue
                _absorb(d.glob("*.outliers.*"))
    except OSError:
        pass
    return [(found[k], k[0], k[1]) for k in sorted(found.keys())]


def load_outliers_for_monitor(
    out_root: Path | str,
    *,
    iter_prefer: int | None = None,
    iset_prefer: int | None = None,
) -> tuple[list[tuple[int, int, float, float, int, float, float]], str]:
    """监视拟合：指定 iter（及可选 iset）被 -R 剔除的点。"""
    entries = list_inverse_outlier_files(out_root)
    if not entries:
        root = Path(out_root)
        parent = root.parent if root.parent.as_posix() not in ("", ".") else Path(".")
        for d in (parent, parent / "residuals"):
            p = d / f"{root.name}.outliers.final"
            if p.is_file():
                rows = parse_outlier_file(p)
                return rows, f"-R 剔除 {len(rows)} 点（.outliers.final）"
        return [], ""
    latest_iter = max(it for _, it, _ in entries)
    use_iter = latest_iter if iter_prefer is None else int(iter_prefer)
    pool = [(p, it, iset) for p, it, iset in entries if it == use_iter]
    if not pool:
        use_iter = latest_iter
        pool = [(p, it, iset) for p, it, iset in entries if it == use_iter]
    if iset_prefer is not None:
        hit = [(p, it, iset) for p, it, iset in pool if iset == int(iset_prefer)]
        if hit:
            pool = hit
    rows: list[tuple[int, int, float, float, int, float, float]] = []
    used_iset: list[int] = []
    for p, _it, iset in pool:
        rows.extend(parse_outlier_file(p))
        used_iset.append(int(iset))
    note = f"-R 剔除 {len(rows)} 点 · iter={use_iter}"
    if used_iset:
        note = f"{note} iset={used_iset[0]}" if len(used_iset) == 1 else f"{note} iset={used_iset}"
    if iter_prefer is not None and int(iter_prefer) != int(use_iter):
        note = f"{note}（模型 iter={iter_prefer} 尚无清单，用已有轮）"
    return rows, note
