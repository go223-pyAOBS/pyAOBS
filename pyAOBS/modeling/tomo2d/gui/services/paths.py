"""工作目录相对化、Zelt 路径 basename、校验（无 UI）。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse

from ..state.form_state import FormState

# gen_smesh.cc：-C<vpath>/<ilayer>、-F<layer>/<rpath> 用 sscanf，vpath/rpath 内不得含 '/'
TOMO2D_ZELT_TOKEN_FILE_KEYS = frozenset(
    {"gen.v_in", "gen.refl_file", "damp.v_in", "vcorr.v_in", "dcorr.v_in"}
)


@dataclass
class PathWarn:
    title: str
    message: str


@dataclass
class PathOpResult:
    value: str
    warnings: list[PathWarn] = field(default_factory=list)


def resolve_work_dir(work_dir: str | None) -> Path:
    """当前 work_dir 的绝对路径（解析失败时退回 cwd）。"""
    wd = (work_dir or "").strip()
    try:
        from .smesh_plot_core import normalize_dropped_path

        base = normalize_dropped_path(wd or ".")
    except Exception:
        base = Path(wd or ".").expanduser()
    try:
        return base.resolve()
    except OSError:
        return Path.cwd()


def _first_log_in_dir(folder: Path) -> Path | None:
    if not folder.is_dir():
        return None
    names = ("tt_inverse.log", "log.all")
    for n in names:
        hit = folder / n
        if hit.is_file():
            return hit
    hits = sorted(p for p in folder.glob("log.all*") if p.is_file())
    if hits:
        return hits[0]
    logs = sorted(p for p in folder.glob("*.log") if p.is_file())
    return logs[0] if logs else None


def _log_all_variant(path: Path) -> Path | None:
    """``log.all`` 不存在时，同目录 ``log.all*``（JobsHHB 常带后缀）。"""
    name = path.name.lower()
    if name != "log.all" and not name.startswith("log.all"):
        return None
    parent = path.parent
    if not parent.is_dir():
        return None
    hits = sorted(p for p in parent.glob("log.all*") if p.is_file())
    if not hits:
        return None
    exact = [p for p in hits if p.name.lower() == "log.all"]
    return (exact or hits)[0]


def resolve_existing_file(path_str: str, work: Path | str | None = None) -> Path:
    """把表单/拖放路径收成当前系统能打开的文件。

    Windows 下 ``/mnt/d/...``、WSL 下 ``D:\\...`` 会互转；相对路径相对 ``work``。
    若写的是目录或 ``…/log.all`` 而实际是 ``log.all.*``，会在同目录补全。
    """
    raw = (path_str or "").strip().strip('"')
    if not raw:
        raise FileNotFoundError("路径为空")
    from .smesh_plot_core import normalize_dropped_path

    work_p: Path | None = None
    if work is not None and str(work).strip():
        try:
            work_p = normalize_dropped_path(str(work))
            if work_p.exists():
                work_p = work_p.resolve()
        except OSError:
            work_p = Path(str(work))

    cands: list[Path] = []
    seen: set[str] = set()

    def add(p: Path | str | None) -> None:
        if p is None or not str(p).strip():
            return
        try:
            q = Path(p)
        except Exception:
            return
        key = str(q).replace("\\", "/").casefold()
        if key in seen:
            return
        seen.add(key)
        cands.append(q)

    add(normalize_dropped_path(raw))
    add(Path(raw).expanduser())
    if work_p is not None:
        p0 = Path(raw).expanduser()
        try:
            posix_abs = raw.startswith("/") or (len(raw) >= 2 and raw[1] == ":")
        except Exception:
            posix_abs = False
        if not posix_abs and not p0.is_absolute():
            add(work_p / p0)
            add(normalize_dropped_path(str(work_p / p0)))

    for p in cands:
        try:
            if p.is_file():
                return p.resolve()
        except OSError:
            continue
    for p in cands:
        try:
            if p.is_dir():
                hit = _first_log_in_dir(p)
                if hit is not None:
                    return hit.resolve()
        except OSError:
            continue
        alt = _log_all_variant(p)
        if alt is not None:
            return alt.resolve()

    shown = str(cands[0]) if cands else raw
    raise FileNotFoundError(f"找不到: {shown}")


def to_workdir_relative(
    path_str: str,
    work: Path,
    *,
    warn_outside: bool = True,
) -> PathOpResult:
    """
    将路径转为相对于 work_dir 的字符串（POSIX 斜杠）。
    已为相对路径时仅规范化；绝对路径且在 work_dir 下则去掉前缀。

    ``work`` 会先 ``resolve()``，避免用未解析的 ``Path('.')`` 相对化时
    误以进程 cwd 为基准，导致预览/运行目录错位。
    """
    raw = (path_str or "").strip()
    if not raw:
        return PathOpResult("")
    try:
        work_r = Path(work).expanduser().resolve()
    except OSError:
        work_r = Path(work).expanduser()
    try:
        from .smesh_plot_core import normalize_dropped_path

        work_r = normalize_dropped_path(work_r)
        try:
            if work_r.exists():
                work_r = work_r.resolve()
        except OSError:
            pass
        raw_n = str(normalize_dropped_path(raw))
    except Exception:
        raw_n = raw
    p = Path(raw_n).expanduser()
    if not p.is_absolute():
        norm = os.path.normpath(raw_n)
        return PathOpResult(Path(norm).as_posix())
    try:
        rp = p.resolve()
    except OSError:
        rp = p
    try:
        return PathOpResult(rp.relative_to(work_r).as_posix())
    except ValueError:
        warns: list[PathWarn] = []
        try:
            rel_s = os.path.relpath(os.fspath(rp), os.fspath(work_r))
        except ValueError:
            # Windows 跨盘符等：无法构造相对路径，保留绝对路径
            if warn_outside:
                warns.append(
                    PathWarn(
                        "路径不在 work_dir 内",
                        "所选路径无法相对当前「工作目录」表示（例如跨盘符）。\n"
                        "已保留绝对路径；请确认底层程序能否解析。\n\n"
                        f"work_dir: {work_r}\n路径: {rp}",
                    )
                )
            return PathOpResult(Path(os.fspath(rp)).as_posix(), warns)
        if warn_outside and rel_s.startswith(".."):
            warns.append(
                PathWarn(
                    "路径不在 work_dir 内",
                    "所选路径不在当前「工作目录」之下。\n"
                    "已存为相对路径（含 ..），请确认底层程序在 work_dir 下运行时能解析。\n\n"
                    f"work_dir: {work_r}\n路径: {rp}",
                )
            )
        return PathOpResult(Path(rel_s).as_posix(), warns)


def postprocess_zelt_file_var(var_key: str, workdir_relative: str) -> PathOpResult:
    """
    表单侧**保留**含目录的相对路径（便于预览/校验）。

    ``-C``/``-F`` 的 sscanf 限制在构造命令行时再取 basename，并在运行前把文件
    stage 到 ``work_dir`` 根目录（见 ``tomand._stage_zelt_token_under_cwd``）。
    """
    if var_key not in TOMO2D_ZELT_TOKEN_FILE_KEYS:
        return PathOpResult(workdir_relative)
    return PathOpResult((workdir_relative or "").strip())


# 表单里存「多路径文本」（换行 / ; / |）的键；规范化时逐条相对化
MULTI_FILE_PATH_KEYS = frozenset({"tx.tx_in"})


def _split_multi_paths(spec: str) -> list[str]:
    """与 tx2tomo2d.parse_tx_in_list 对齐的轻量拆分（避免循环 import）。"""
    raw = (spec or "").strip()
    if not raw:
        return []
    if "\n" not in raw and ";" not in raw and "|" not in raw:
        return [raw]
    parts: list[str] = []
    for chunk in raw.replace("|", "\n").replace(";", "\n").splitlines():
        s = chunk.strip().strip('"').strip("'")
        if s:
            parts.append(s)
    return parts


_FILE_URL_SPLIT = re.compile(r"(?=file:)", re.IGNORECASE)
_LOG_DRIVE_CONCAT = re.compile(r"(?<=\.(?:log|txt|out))(?=[A-Za-z]:[\\/])")


def _strip_file_url(path: str) -> str:
    s = (path or "").strip()
    if not s.lower().startswith("file:"):
        return s
    u = urlparse(s)
    body = unquote(u.path or "")
    if len(body) >= 3 and body[0] == "/" and body[2] == ":":
        body = body[1:]
    return body or s


def split_log_path_list(text: str) -> list[str]:
    """多日志文本 → 路径列表。

    真正的换行 / ``;`` / ``|`` 分隔；也拆开拖放时挤在同一行的多个 ``file://``，
    以及 ``….log`` 紧接另一个盘符路径（``C:\\a.logD:\\b.log``）。
    **不**按空格拆，以免切断含空格的 Windows 路径。界面折行不是换行。
    """
    raw = (text or "").replace("\x00", "\n")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    if not raw.strip():
        return []
    if "file:" in raw.lower():
        pieces = [p.strip() for p in _FILE_URL_SPLIT.split(raw) if p.strip()]
        if len(pieces) > 1:
            raw = "\n".join(pieces)
    raw = _LOG_DRIVE_CONCAT.sub("\n", raw)
    parts: list[str] = []
    seen: set[str] = set()
    for chunk in raw.replace("|", "\n").replace(";", "\n").splitlines():
        s = chunk.strip().strip('"').strip("'")
        if not s or s.startswith("#"):
            continue
        s = _strip_file_url(s)
        key = s.replace("\\", "/").casefold()
        if not s or key in seen:
            continue
        seen.add(key)
        parts.append(s)
    return parts


def normalize_file_path_vars(
    state: FormState,
    file_keys: Iterable[str],
    *,
    warn_outside: bool = False,
) -> list[PathWarn]:
    """将指定「文件路径」键转为相对 work_dir（若可能）。原地更新 state。"""
    work = resolve_work_dir(state.get_str("work_dir"))
    all_warns: list[PathWarn] = []
    for key in file_keys:
        if not state.has(key):
            continue
        cur = state.get_str(key)
        if not cur:
            continue
        if key in MULTI_FILE_PATH_KEYS or (
            ("\n" in cur or ";" in cur or "|" in cur) and key.endswith(".tx_in")
        ):
            pieces = _split_multi_paths(cur)
            norms: list[str] = []
            for piece in pieces:
                rel = to_workdir_relative(piece, work, warn_outside=warn_outside)
                all_warns.extend(rel.warnings)
                zelt = postprocess_zelt_file_var(key, rel.value)
                all_warns.extend(zelt.warnings)
                if zelt.value:
                    norms.append(zelt.value)
            state.set(key, "\n".join(norms))
            continue
        rel = to_workdir_relative(cur, work, warn_outside=warn_outside)
        all_warns.extend(rel.warnings)
        zelt = postprocess_zelt_file_var(key, rel.value)
        all_warns.extend(zelt.warnings)
        state.set(key, zelt.value)
    return all_warns

def validate_work_dir(work_dir: str | None) -> Path:
    work = Path((work_dir or "").strip() or str(Path.cwd())).expanduser()
    try:
        work = work.resolve()
    except OSError as e:
        raise ValueError(f"work_dir 无法解析: {work_dir}") from e
    if not work.exists():
        raise ValueError(f"work_dir 不存在: {work}")
    return work
