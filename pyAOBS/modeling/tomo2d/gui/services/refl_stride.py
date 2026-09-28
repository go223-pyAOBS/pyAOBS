"""tt_inverse 反射面抽稀（GUI 预处理，不是 C++ -F 开关）。

gen_smesh 写出的 refl 点距多为网格 dx；反演时常加大点距，等价于::

    awk '(NR)%N==0{print}' refl > refl_new

本模块按步长 N 取样：保留首点、每隔 N 点、以及末点（避免界面两端被裁掉）。
空或 1 表示不抽稀。步长记在 kwargs ``_refl_stride``，**不**在源文件旁另存
``*_sN.refl``；真正给 C++ ``-F`` 时才写到运行包 ``inputs/`` 或临时目录。
"""

from __future__ import annotations

from pathlib import Path

from ..state.form_state import FormState
from .paths import resolve_existing_file, resolve_work_dir, to_workdir_relative

STRIDE_KEY = "_refl_stride"


def parse_refl_stride(raw: str) -> int:
    s = (raw or "").strip()
    if not s:
        return 1
    try:
        n = int(s)
    except ValueError:
        raise ValueError("refl 抽稀步长须为正整数（空或 1 表示不抽稀）") from None
    if n < 1:
        raise ValueError("refl 抽稀步长须为正整数（空或 1 表示不抽稀）")
    return n


def peek_refl_stride(kwargs: dict | None) -> int:
    if not kwargs:
        return 1
    raw = kwargs.get(STRIDE_KEY)
    if raw is None:
        return 1
    return parse_refl_stride(str(raw))


def pop_refl_stride(kwargs: dict) -> int:
    raw = kwargs.pop(STRIDE_KEY, None)
    if raw is None:
        return 1
    return parse_refl_stride(str(raw))


def estimate_refl_dx(path: Path) -> float | None:
    """从 refl 首列 x 估计点距（相邻差的中位数）。"""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    xs: list[float] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        tok = s.split()[0]
        try:
            xs.append(float(tok))
        except ValueError:
            continue
    if len(xs) < 2:
        return None
    diffs = [abs(b - a) for a, b in zip(xs, xs[1:]) if b != a]
    if not diffs:
        return None
    diffs.sort()
    mid = len(diffs) // 2
    if len(diffs) % 2:
        return diffs[mid]
    return 0.5 * (diffs[mid - 1] + diffs[mid])


def format_dx(dx: float) -> str:
    return f"{dx:.6g}"


def subsample_refl_lines(lines: list[str], stride: int) -> list[str]:
    """保留首点、每隔 stride 点、以及末点。"""
    if stride <= 1 or len(lines) <= 2:
        return list(lines)
    idxs = list(range(0, len(lines), stride))
    last = len(lines) - 1
    if idxs[-1] != last:
        idxs.append(last)
    return [lines[i] for i in idxs]


def write_subsampled_refl(src: Path, dst: Path, stride: int) -> int:
    text = src.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    out_lines = subsample_refl_lines(lines, stride)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(out_lines), encoding="utf-8")
    return len(out_lines)


def copy_or_stride_refl(src: Path, dest: Path, stride: int) -> None:
    """stride≤1 时原样复制；否则把抽稀结果写到 dest（不改 src）。"""
    src_r = src.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if stride <= 1:
        if dest.resolve() != src_r:
            import shutil

            shutil.copy2(src_r, dest)
        return
    write_subsampled_refl(src_r, dest, stride)


def apply_inv_refl_stride(state: FormState, work: Path, kwargs: dict) -> str | None:
    """步长 > 1 时只记下 ``_refl_stride``，``refl_file`` 仍指向源文件。"""
    src_s = str(kwargs.get("refl_file") or "").strip()
    stride = parse_refl_stride(state.get_str("inv.refl_stride"))
    kwargs.pop(STRIDE_KEY, None)
    if not src_s or stride <= 1:
        return None
    try:
        src = resolve_existing_file(src_s, work)
    except FileNotFoundError:
        src = Path(src_s)
        if not src.is_absolute():
            src = (work / src).resolve()
        else:
            src = src.resolve()
    if not src.is_file():
        raise ValueError(f"refl 抽稀步长为 {stride}，但 refl_file 不存在: {src}")
    kwargs[STRIDE_KEY] = stride
    return f"refl 抽稀步长 {stride}（运行时取样，不另存文件）"


def apply_inv_refl_stride_from_state(state: FormState, kwargs: dict) -> str | None:
    work = resolve_work_dir(state.get_str("work_dir"))
    return apply_inv_refl_stride(state, work, kwargs)


def materialize_refl_for_cwd(
    kwargs: dict,
    work: Path,
    cwd: Path,
) -> Path | None:
    """无运行包时：把抽稀结果写到 cwd/.tomo2d_tmp/，改 kwargs['refl_file']。"""
    stride = pop_refl_stride(kwargs)
    src_s = str(kwargs.get("refl_file") or "").strip()
    if not src_s or stride <= 1:
        return None
    src = resolve_existing_file(src_s, work)
    dest = Path(cwd) / ".tomo2d_tmp" / f"refl_stride{src.suffix or '.dat'}"
    write_subsampled_refl(src, dest, stride)
    try:
        rel = dest.resolve().relative_to(Path(cwd).resolve()).as_posix()
    except ValueError:
        rel = to_workdir_relative(str(dest), work, warn_outside=False).value
    kwargs["refl_file"] = rel
    return dest
