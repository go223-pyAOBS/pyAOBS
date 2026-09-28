"""运行包查找、manifest 摘要、相邻模型速度差（无 UI；供模型挑选等使用）。"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .smesh_ops import list_inverse_smesh_files, parse_inverse_smesh_name


@dataclass
class RunSummary:
    run_dir: Path
    manifest_path: Path | None = None
    status: str = ""
    exit_code: int | None = None
    finished_utc: str = ""
    created_utc: str = ""
    out_root_rel: str = "outputs/out"
    kind_counts: dict[str, int] = field(default_factory=dict)
    n_models: int = 0
    note: str = ""
    error: str = ""


def list_tt_inverse_run_dirs(work: Path | str, *, limit: int = 40) -> list[Path]:
    """``work/runs/*`` 中含 manifest 或 outputs 的目录，按 mtime 新→旧。"""
    root = Path(work) / "runs"
    if not root.is_dir():
        return []
    cands: list[tuple[float, Path]] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if not (p / "manifest.json").is_file() and not (p / "outputs").is_dir():
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        cands.append((mtime, p))
    cands.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in cands[: max(1, int(limit))]]


def _is_tt_inverse_run_dir(d: Path) -> bool:
    """是否像一次 tt_inverse 运行包根（避免把工区根因存在 outputs/ 误判进来）。"""
    if not d.is_dir():
        return False
    if (d / "manifest.json").is_file():
        return True
    if d.parent.name == "runs":
        return True
    if (d / "outputs" / "models").is_dir():
        return True
    from .smesh_ops import list_inverse_smesh_files

    for root in (d / "outputs", d):
        try:
            if list_inverse_smesh_files(root):
                return True
        except OSError:
            continue
    return False


def resolve_user_run_dir(path: Path | str) -> Path | None:
    """把用户点的目录或文件收成运行包根。

    可点包根、``outputs/``、``models/``，或包内某个 ``.smesh``。
    """
    p = Path(path).expanduser()
    try:
        p = p.resolve()
    except OSError:
        pass
    hit = infer_run_dir_from_smesh(p)
    if hit is not None:
        return hit
    start = p.parent if p.is_file() else p
    if not start.is_dir():
        return None
    for d in [start, *list(start.parents)[:6]]:
        if _is_tt_inverse_run_dir(d):
            return d
    return None


def _out_root_from_manifest(data: dict[str, Any]) -> str:
    pr = data.get("python_replay") or {}
    kw = pr.get("kwargs") or {}
    out = kw.get("out_root")
    if out:
        return str(out).replace("\\", "/")
    argv = data.get("argv") or []
    for a in argv:
        s = str(a)
        if s.startswith("-O") and len(s) > 2:
            return s[2:].replace("\\", "/")
    return "outputs/out"


def load_run_summary(run_dir: Path | str) -> RunSummary:
    """读取一次 tt_inverse 运行包摘要。"""
    rd = Path(run_dir)
    man = rd / "manifest.json"
    summary = RunSummary(run_dir=rd.resolve() if rd.exists() else rd)
    data: dict[str, Any] = {}
    if man.is_file():
        summary.manifest_path = man
        try:
            data = json.loads(man.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            summary.note = f"manifest 解析失败: {e}"
            data = {}
    summary.created_utc = str(data.get("created_utc") or "")
    pr = data.get("post_run") or {}
    summary.status = str(pr.get("status") or data.get("status") or "")
    summary.finished_utc = str(pr.get("finished_utc") or "")
    summary.error = str(pr.get("error") or "") or ""
    ec = pr.get("exit_code")
    try:
        summary.exit_code = int(ec) if ec is not None else None
    except (TypeError, ValueError):
        summary.exit_code = None
    summary.out_root_rel = _out_root_from_manifest(data) if data else "outputs/out"

    outs = pr.get("output_files") or []
    if outs:
        summary.kind_counts = dict(Counter(str(r.get("kind") or "other") for r in outs))
        summary.n_models = int(summary.kind_counts.get("model", 0))
    else:
        od = rd / "outputs"
        if od.is_dir():
            counts: Counter[str] = Counter()
            for p in od.rglob("*"):
                if not p.is_file():
                    continue
                from ...tt_inverse_bundle import classify_tt_inverse_output_kind

                counts[classify_tt_inverse_output_kind(p.name)] += 1
            summary.kind_counts = dict(counts)
            summary.n_models = int(counts.get("model", 0))
    return summary


def resolve_run_out_root(summary: RunSummary) -> Path:
    rel = summary.out_root_rel or "outputs/out"
    return (summary.run_dir / rel).resolve()


def monitor_spec_for_run(run_dir: Path | str, *, niter: Any = None):
    """一次 ``runs/<包>/`` → 监视/挑选用的规格。"""
    from .inv_monitor import build_monitor_spec_from_paths

    rd = Path(run_dir)
    summary = load_run_summary(rd)
    return build_monitor_spec_from_paths(
        cwd=rd,
        log_file="outputs/tt_inverse.log",
        out_root=summary.out_root_rel or "outputs/out",
        run_dir=rd,
        status_jsonl="outputs/status.jsonl",
        niter=niter,
    )


def format_run_summary_text(s: RunSummary) -> str:
    parts = [f"run: {s.run_dir.name}"]
    if s.status:
        parts.append(f"status={s.status}")
    if s.exit_code is not None:
        parts.append(f"exit={s.exit_code}")
    if s.finished_utc:
        parts.append(f"finished={s.finished_utc}")
    if s.kind_counts:
        kc = ", ".join(f"{k}:{v}" for k, v in sorted(s.kind_counts.items()))
        parts.append(f"files[{kc}]")
    if s.error:
        parts.append(f"error={s.error}")
    if s.note:
        parts.append(s.note)
    return " · ".join(parts)


DiffMode = Literal["abs", "percent"]


@dataclass
class SmeshDiffResult:
    path_a: Path
    path_b: Path
    mode: DiffMode
    xpos: np.ndarray
    zpos: np.ndarray
    topo: np.ndarray
    dv: np.ndarray
    vmin: float
    vmax: float
    mean: float
    std: float


def order_smesh_pair(
    path_a: Path | str, path_b: Path | str
) -> tuple[Path, Path]:
    """按 ``iter.iset`` 排序：前轮为 A、后轮为 B（B−A）。解析失败则保持传入顺序。"""
    pa, pb = Path(path_a), Path(path_b)
    ka = parse_inverse_smesh_name(pa)
    kb = parse_inverse_smesh_name(pb)
    if ka is not None and kb is not None and ka > kb:
        return pb, pa
    return pa, pb


DIFF_VLIM_AUTO_KEY = "gui.diff_vlim_auto"
DIFF_VLIM_ABS_KEY = "gui.diff_vlim_abs"
DIFF_VLIM_PCT_KEY = "gui.diff_vlim_pct"
DEFAULT_DIFF_VLIM_ABS = 0.5
DEFAULT_DIFF_VLIM_PCT = 5.0


def _state_positive_float(state: Any, key: str, default: float) -> float:
    raw = ""
    if state is not None and hasattr(state, "get_str"):
        raw = state.get_str(key, "")
    if not str(raw).strip():
        return float(default)
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(v) or v <= 0:
        return float(default)
    return v


def diff_vlim_is_auto(state: Any) -> bool:
    """未写键时默认固定色标，避免每次对比都被本图 |max| 拉满。"""
    if state is None or not hasattr(state, "get_bool"):
        return False
    return bool(state.get_bool(DIFF_VLIM_AUTO_KEY, False))


def set_diff_vlim_auto(state: Any, on: bool) -> None:
    if state is not None and hasattr(state, "set"):
        state.set(DIFF_VLIM_AUTO_KEY, "1" if on else "0")


def diff_vlim_half_range(state: Any, mode: str) -> float:
    if str(mode) == "percent":
        return _state_positive_float(state, DIFF_VLIM_PCT_KEY, DEFAULT_DIFF_VLIM_PCT)
    return _state_positive_float(state, DIFF_VLIM_ABS_KEY, DEFAULT_DIFF_VLIM_ABS)


def set_diff_vlim_half_range(state: Any, mode: str, half: float) -> None:
    key = DIFF_VLIM_PCT_KEY if str(mode) == "percent" else DIFF_VLIM_ABS_KEY
    default = DEFAULT_DIFF_VLIM_PCT if str(mode) == "percent" else DEFAULT_DIFF_VLIM_ABS
    try:
        v = float(half)
    except (TypeError, ValueError):
        v = float(default)
    if not np.isfinite(v) or v <= 0:
        v = float(default)
    if state is not None and hasattr(state, "set"):
        state.set(key, str(v))


def resolve_diff_colorbar_limits(
    data_peak: float,
    *,
    auto: bool,
    half_range: float,
) -> tuple[float, float]:
    """对称 ΔV 色标。``auto`` 时用数据 |max|，否则固定 ±half_range。"""
    peak = abs(float(data_peak))
    if auto:
        h = peak if peak > 0 else 1e-6
        return -h, h
    try:
        h = float(half_range)
    except (TypeError, ValueError):
        h = 0.0
    if not np.isfinite(h) or h <= 0:
        h = peak if peak > 0 else 1e-6
    return -h, h


def format_diff_vlim_caption(
    data_peak: float, vmin: float, vmax: float, *, auto: bool
) -> str:
    half = max(abs(float(vmin)), abs(float(vmax)))
    peak = abs(float(data_peak))
    if auto:
        return f"色标±{half:.4g}（随数据）"
    note = f"色标±{half:g}"
    if peak > half * (1.0 + 1e-9):
        note += "（超出已饱和）"
    return note


SIGMA_AUTO_PERCENTILE = 96.0
SIGMA_NORM_GAMMA = 0.5


def sigma_data_max(std_v) -> float:
    """误差场有限值的最大值；空则 1e-6。"""
    arr = np.asarray(std_v, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1e-6
    return max(float(np.max(np.abs(finite))), 1e-6)


def sigma_robust_hi(
    std_v,
    *,
    percentile: float = SIGMA_AUTO_PERCENTILE,
) -> float:
    """自动色标上界：正 σ 的高分位，避免边部极值把中间压成一团。"""
    arr = np.asarray(std_v, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1e-6
    peak = max(float(np.max(np.abs(finite))), 1e-6)
    pos = finite[finite > 0.0]
    src = pos if pos.size >= 8 else finite
    try:
        p = float(percentile)
    except (TypeError, ValueError):
        p = SIGMA_AUTO_PERCENTILE
    p = min(max(p, 50.0), 99.9)
    hi = float(np.percentile(src, p))
    if not np.isfinite(hi) or hi <= 0.0:
        return peak
    return min(max(hi, 1e-6), peak)


def resolve_sigma_colorbar_limits(
    data_max: float,
    *,
    auto: bool,
    half_range: float,
    robust_hi: float | None = None,
) -> tuple[float, float]:
    """误差 σ 色标：0 到上界。

    ``auto`` 优先用 ``robust_hi``（分位上界），否则用本图最大；
    固定模式用 ±km/s 的正半幅。
    """
    peak = abs(float(data_max))
    if auto:
        h = peak
        if robust_hi is not None:
            try:
                rh = float(robust_hi)
            except (TypeError, ValueError):
                rh = 0.0
            if np.isfinite(rh) and rh > 0.0:
                h = rh
        if h <= 0:
            h = 1e-6
        return 0.0, h
    try:
        h = float(half_range)
    except (TypeError, ValueError):
        h = 0.0
    if not np.isfinite(h) or h <= 0:
        h = peak if peak > 0 else 1e-6
    return 0.0, h


def format_sigma_vlim_caption(
    vmax: float, data_max: float, *, auto: bool, scale: str = "robust"
) -> str:
    hi = abs(float(vmax))
    peak = abs(float(data_max))
    if str(scale).strip().lower() == "cpt":
        note = f"σ 色标 0–{hi:.4g}（develf）"
        if peak > hi * (1.0 + 1e-9):
            note += "（超出已饱和）"
        return note
    if auto:
        note = f"σ 色标 0–{hi:.4g}（P{SIGMA_AUTO_PERCENTILE:g}）"
        if peak > hi * (1.0 + 1e-9):
            note += "（尾部饱和）"
        return note
    note = f"σ 色标 0–{hi:g}"
    if peak > hi * (1.0 + 1e-9):
        note += "（超出已饱和）"
    return note


def resolve_sigma_cmap_and_limits(
    std_v,
    *,
    auto: bool,
    half_range: float,
) -> tuple[str, float, float, float | None]:
    """误差图：优先 ``assets/develf.cpt`` 及其 0–0.40 分段；否则稳健自动 + 幂次。

    返回 ``(cmap_spec, vmin, vmax, norm_gamma)``。有 CPT 时按文件自身 z 域着色，
    不用幂次拉伸，也不用手写半幅去压色段。
    """
    from .smesh_plot_core import builtin_sigma_cpt_path

    cpt = builtin_sigma_cpt_path()
    if cpt.is_file():
        try:
            from pyAOBS.visualization.gmt_cpt import parse_gmt_cpt_for_matplotlib

            _cmap, z0, z1 = parse_gmt_cpt_for_matplotlib(str(cpt))
            lo, hi = float(z0), float(z1)
        except Exception:
            lo, hi = 0.0, 0.4
        return str(cpt), lo, hi, None
    slo, shi = resolve_sigma_colorbar_limits(
        sigma_data_max(std_v),
        auto=bool(auto),
        half_range=float(half_range),
        robust_hi=sigma_robust_hi(std_v),
    )
    return "YlGnBu_0white", slo, shi, SIGMA_NORM_GAMMA


def compute_smesh_velocity_diff(
    path_a: Path | str,
    path_b: Path | str,
    *,
    mode: DiffMode = "abs",
) -> SmeshDiffResult:
    """
    速度差：``B - A``（abs=km/s）或 ``100*(B-A)/|A|``（percent）。
    网格形状须一致。
    """
    try:
        from pyAOBS.model_building.tomoform import SlownessMesh2D
    except ImportError:  # pragma: no cover
        from pyAOBS.model_building.tomoform import SlownessMesh2D  # type: ignore

    pa, pb = Path(path_a), Path(path_b)
    ma = SlownessMesh2D.from_file(str(pa))
    mb = SlownessMesh2D.from_file(str(pb))
    if ma.vgrid.shape != mb.vgrid.shape:
        raise ValueError(
            f"网格尺寸不一致: {pa.name} {ma.vgrid.shape} vs {pb.name} {mb.vgrid.shape}"
        )
    va = np.asarray(ma.vgrid, dtype=float)
    vb = np.asarray(mb.vgrid, dtype=float)
    if mode == "percent":
        dv = 100.0 * (vb - va) / np.maximum(np.abs(va), 1e-12)
    else:
        dv = vb - va
    finite = dv[np.isfinite(dv)]
    if finite.size == 0:
        raise ValueError("差值场无有效数值")
    peak = float(np.nanmax(np.abs(finite)))
    return SmeshDiffResult(
        path_a=pa,
        path_b=pb,
        mode=mode,
        xpos=np.asarray(ma.xpos, dtype=float),
        zpos=np.asarray(ma.zpos, dtype=float),
        topo=np.asarray(ma.topo, dtype=float),
        dv=dv,
        vmin=-peak,
        vmax=peak,
        mean=float(np.mean(finite)),
        std=float(np.std(finite)),
    )


def resample_vgrid_fields_to_xarray(mesh: Any, fields: list[np.ndarray]) -> list:
    """多个节点场共用同一套绘图 (x, z)，空气/水层填 0。"""
    if not fields:
        return []
    native = tuple(np.asarray(mesh.vgrid).shape)
    arrays: list[np.ndarray] = []
    for field in fields:
        arr = np.asarray(field, dtype=float)
        if arr.shape != native:
            raise ValueError(f"节点场形状与 smesh 不一致: {arr.shape} vs {native}")
        arrays.append(arr)
    x_new, full_zpos = mesh._regular_plot_axes()
    out = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for arr in arrays:
            vg = mesh._vgrid_on_regular(arr, x_new, full_zpos, 0.0, 0.0)
            out.append(
                mesh._regular_velocity_dataset(
                    x_new, full_zpos, vg, v_air=0.0, v_water=0.0
                )
            )
    return out


def resample_vgrid_field_to_xarray(mesh: Any, field: np.ndarray):
    """把节点场 ``(nx, nz)`` 铺到与 ``SlownessMesh2D.to_xarray()`` 相同的绘图网格。

    监视/挑选用的速度图不是原生 ``vgrid``，而是绝对深度规则网格（含空气/水层，
    ``x`` 还可能因 ``arange`` 少 1–2 个点）。差值必须先重采样，不能直接塞进
    ``ds["velocity"]``。空气/水层填 0，避免沿用 ``v_air``/``v_water`` 污染色标。
    """
    return resample_vgrid_fields_to_xarray(mesh, [field])[0]


def default_adjacent_pair(
    out_root: Path | str,
) -> tuple[Path, Path] | None:
    """取最后两个写出模型作为相邻对比对；不足两个则 None。"""
    entries = list_inverse_smesh_files(out_root)
    if len(entries) < 2:
        return None
    return entries[-2][0], entries[-1][0]


def label_smesh_candidate(path: Path) -> str:
    key = parse_inverse_smesh_name(path)
    if key is None:
        return path.name
    return f"iter {key[0]}.{key[1]} · {path.name}"


def infer_run_dir_from_smesh(path: Path | str) -> Path | None:
    """若 smesh 落在某次 tt_inverse 运行包内，返回该 ``run_dir``。"""
    p = Path(path)
    try:
        p = p.resolve()
    except OSError:
        pass
    cur = p.parent if p.suffix or not p.is_dir() else p
    if p.is_file():
        cur = p.parent
    for d in [cur, *list(cur.parents)]:
        if (d / "manifest.json").is_file():
            return d
        if d.parent.name == "runs" and (d / "outputs").is_dir():
            return d
    return None


def find_smesh_for_inverse_log(log_path: Path | str) -> Path:
    """由 tt_inverse ``-L`` 日志定位该次运行**最后写出**的 smesh（非 χ² 最优）。

    查找顺序：
    1. GUI 运行包 ``manifest`` 的 ``-O``（``outputs/out.smesh.*`` 或 ``outputs/models/``）
    2. 日志旁的默认前缀 ``out`` / ``outputs/out``
    3. 日志所在目录当作一次反演文件夹：目录内（及 ``models/``）任意
       ``*.smesh.<iter>.<iset>``；或父目录下 ``{目录名}.smesh.*``（``-O`` 等于该目录路径）
    """
    from .smesh_ops import find_latest_inverse_smesh, list_inverse_smesh_files

    log = Path(log_path)
    rd = infer_run_dir_from_smesh(log)
    if rd is not None:
        spec = monitor_spec_for_run(rd)
        if spec.out_root is not None:
            try:
                return find_latest_inverse_smesh(spec.out_root)
            except FileNotFoundError:
                pass
    parent = log.parent
    for root in (
        parent / "out",
        parent.parent / "out",
        parent / "outputs" / "out",
        parent,
    ):
        try:
            if list_inverse_smesh_files(root):
                return find_latest_inverse_smesh(root)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        "未找到与日志对应的 smesh。"
        "tt_inverse 写出的是 ``{ -O 前缀 }.smesh.<迭代>.<iset>`` "
        f"（可与日志同目录，或在 models/ 下；不限于文件名 ``out.smesh.*``）：{log}"
    )
