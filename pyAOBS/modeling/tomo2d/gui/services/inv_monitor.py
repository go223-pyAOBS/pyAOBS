"""反演准实时监视：解析 -L / status.jsonl、定位最新 smesh（无 UI）。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...tt_inverse_log_analysis import (
    COL_CHI_TOT,
    COL_ITER,
    COL_ISET,
    COL_LMVH,
    COL_LMVV,
    COL_PRED_CHI,
    COL_RMS_TOT,
    COL_W_DD,
    COL_W_DV,
    COL_W_SD,
    COL_W_SV,
    format_tt_inverse_log_header_summary,
    format_tt_inverse_run_params,
    parse_tt_inverse_log,
    parse_tt_inverse_log_header,
)
from .smesh_ops import find_latest_inverse_smesh, list_inverse_smesh_files


def _merge_status_bits(hdr: str, extra: str) -> str:
    """把策略摘要接到日志头后，已出现的词不再重复。"""
    if not extra:
        return hdr
    keep: list[str] = []
    for bit in extra.split(" · "):
        token = bit.split("=")[0].split()[0]
        if token and token in hdr:
            continue
        keep.append(bit)
    if not keep:
        return hdr
    return (hdr + " · " if hdr else "") + " · ".join(keep)


def _strategy_flags_for_monitor(
    spec: InvMonitorSpec, *, hdr_info: dict[str, Any] | None = None
) -> str:
    """-L 头没有 ``# accel`` 时，从运行包 manifest 的 gui_profile 补前向复用/C2F/列预条件。"""
    info = hdr_info or {}
    if (
        info.get("reuse_forward") is not None
        or info.get("coarse2fine") is not None
        or info.get("legacy_baseline") is not None
    ):
        return ""
    run_dir = spec.run_dir
    if run_dir is None:
        return ""
    man = Path(run_dir) / "manifest.json"
    if not man.is_file():
        return ""
    try:
        obj = json.loads(man.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    gp = obj.get("gui_profile")
    if not isinstance(gp, dict):
        return ""
    from .run_env import format_strategy_flags, strategy_env_from_form

    return format_strategy_flags(strategy_env_from_form(gp))


def _replay_param_bits(kwargs: dict[str, Any]) -> list[str]:
    bits: list[str] = []
    niter = kwargs.get("niter")
    if niter is not None:
        bits.append(f"-I{niter}")
    j = kwargs.get("target_chi2")
    if j is not None:
        bits.append(f"-J{j}")
    sm = kwargs.get("smooth_opts") or {}
    if isinstance(sm, dict):
        cv = sm.get("corr_v_fn")
        if cv:
            bits.append(f"-CV {Path(str(cv)).name}")
        cd = sm.get("corr_d_fn")
        if cd:
            bits.append(f"-CD {Path(str(cd)).name}")
    return bits


def collect_run_inversion_params(spec: InvMonitorSpec) -> str:
    """当前运行包的反演参数摘要（-L 头优先，缺项用 manifest 补）。"""
    note = ""
    info: dict[str, Any] = {}
    log_path = spec.resolve_log()
    if log_path is not None:
        try:
            info = parse_tt_inverse_log_header(log_path)
            note = format_tt_inverse_run_params(info)
            extra = _strategy_flags_for_monitor(spec, hdr_info=info)
            if extra:
                note = _merge_status_bits(note, extra)
        except OSError:
            info = {}
    else:
        extra = _strategy_flags_for_monitor(spec)
        if extra:
            note = extra
    replay_bits: list[str] = []
    run_dir = spec.run_dir
    if run_dir is not None:
        man = Path(run_dir) / "manifest.json"
        if man.is_file():
            try:
                obj = json.loads(man.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                obj = {}
            pr = obj.get("python_replay") if isinstance(obj, dict) else None
            kw = pr.get("kwargs") if isinstance(pr, dict) else None
            if isinstance(kw, dict):
                replay_bits = _replay_param_bits(kw)
            if not note:
                gp = obj.get("gui_profile") if isinstance(obj, dict) else None
                if isinstance(gp, dict):
                    from .run_env import format_strategy_flags, strategy_env_from_form

                    note = format_strategy_flags(strategy_env_from_form(gp))
    for bit in replay_bits:
        token = bit.split()[0]
        if token and token in note:
            continue
        note = (note + " · " if note else "") + bit
    return note


@dataclass
class InvMonitorSpec:
    """一次 tt_inverse 运行的监视目标。"""

    log_candidates: list[Path] = field(default_factory=list)
    status_candidates: list[Path] = field(default_factory=list)
    out_root: Path | None = None
    run_dir: Path | None = None
    niter: int | None = None

    def resolve_log(self) -> Path | None:
        for p in self.log_candidates:
            if p is not None and Path(p).is_file():
                return Path(p)
        return None

    def resolve_status(self) -> Path | None:
        for p in self.status_candidates:
            if p is not None and Path(p).is_file():
                return Path(p)
        return None


def parse_status_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """读取 C++ ``TOMO2D_INV_STATUS_JSONL`` 追加的 NDJSON 行。"""
    rows: list[dict[str, Any]] = []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return rows
    for line in text.splitlines():
        s = line.strip()
        if not s or not s.startswith("{"):
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


@dataclass
class InvMonitorSnapshot:
    log_path: Path | None
    n_rows: int
    last_iter: int | None
    last_iset: int | None
    last_chi: float | None
    last_rms: float | None
    last_pred_chi: float | None
    last_rough_v: float | None  # Lmvh+Lmvv
    iters: list[float]
    chi: list[float]
    rms: list[float]
    pred_chi: list[float]
    rough_v: list[float]
    smesh_path: Path | None
    smesh_mtime: float | None
    ray_stamp: str = ""  # 射线文件集合指纹（路径+mtime）
    tres_stamp: str = ""  # 残差 .tres 目录指纹
    status_path: Path | None = None
    status_n: int = 0
    note: str = ""


def _uniq_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in paths:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def build_monitor_spec_from_paths(
    *,
    cwd: Path,
    log_file: str | None,
    out_root: str | None,
    niter: Any = None,
    run_dir: Path | None = None,
    status_jsonl: str | None = "outputs/status.jsonl",
) -> InvMonitorSpec:
    """根据子进程 cwd 与相对路径构造监视规格。"""
    base = Path(cwd).resolve()
    log_name = Path(str(log_file or "tt_inverse.log").replace("\\", "/")).name or "tt_inverse.log"
    log_rel = str(log_file or f"outputs/{log_name}").replace("\\", "/")
    log_cands = _uniq_paths(
        [
            base / log_rel,
            base / "outputs" / log_name,
            base / "outputs" / "logs" / log_name,
        ]
    )

    st_name = "status.jsonl"
    st_rel = (status_jsonl or "").strip().replace("\\", "/")
    st_cands: list[Path] = []
    if st_rel:
        st_name = Path(st_rel).name or st_name
        st_cands.append(base / st_rel)
    st_cands.extend(
        [
            base / "outputs" / st_name,
            base / "outputs" / "logs" / st_name,
        ]
    )
    st_cands = _uniq_paths(st_cands)

    out_rel = str(out_root or "outputs/out").replace("\\", "/")
    out_p = base / out_rel
    ni: int | None
    try:
        ni = int(niter) if niter is not None and str(niter).strip() != "" else None
    except (TypeError, ValueError):
        ni = None
    return InvMonitorSpec(
        log_candidates=log_cands,
        status_candidates=st_cands,
        out_root=out_p,
        run_dir=run_dir.resolve() if run_dir is not None else None,
        niter=ni,
    )


def collect_monitor_snapshot(
    spec: InvMonitorSpec,
    *,
    include_rays: bool = False,
) -> InvMonitorSnapshot:
    """读取 -L / status.jsonl 与最后写出的 smesh（非最优评判）。

    ``include_rays`` 为假时不枚举 ``.ray``（监视默认不叠加射线，避免主线程卡顿）。
    """
    log_path = spec.resolve_log()
    status_path = spec.resolve_status()
    iters: list[float] = []
    chi: list[float] = []
    rms: list[float] = []
    pred: list[float] = []
    rough: list[float] = []
    last_iter = last_iset = None
    last_chi = last_rms = last_pred = last_rv = None
    n_rows = 0
    status_n = 0
    note = ""

    if log_path is not None:
        try:
            rows = parse_tt_inverse_log(log_path)
        except OSError as e:
            rows = []
            note = f"读日志失败: {e}"
        n_rows = len(rows)
        for i, r in enumerate(rows):
            iters.append(float(i + 1))
            chi.append(float(r[COL_CHI_TOT]))
            rms.append(float(r[COL_RMS_TOT]))
            pred.append(float(r[COL_PRED_CHI]))
            rv = float(r[COL_LMVH]) + float(r[COL_LMVV])
            rough.append(rv)
        if rows:
            r = rows[-1]
            last_iter = int(r[COL_ITER])
            last_iset = int(r[COL_ISET])
            last_chi = float(r[COL_CHI_TOT])
            last_rms = float(r[COL_RMS_TOT])
            last_pred = float(r[COL_PRED_CHI])
            last_rv = float(r[COL_LMVH]) + float(r[COL_LMVV])
        try:
            info = parse_tt_inverse_log_header(log_path)
            hdr = format_tt_inverse_log_header_summary(info)
            extra = _strategy_flags_for_monitor(spec, hdr_info=info)
            if extra:
                hdr = _merge_status_bits(hdr, extra)
            if hdr:
                note = (note + " · " if note else "") + hdr
        except OSError:
            pass
    else:
        extra = _strategy_flags_for_monitor(spec)
        if extra:
            note = (note + " · " if note else "") + extra

    # status.jsonl：优先刷新末态；无 -L 时也用于画曲线
    if status_path is not None:
        st_rows = parse_status_jsonl(status_path)
        status_n = len(st_rows)
        if st_rows:
            if n_rows == 0:
                for i, obj in enumerate(st_rows):
                    iters.append(float(i + 1))
                    chi.append(float(obj.get("chi2", float("nan"))))
                    rms.append(float(obj.get("rms", float("nan"))))
                    pred.append(float(obj.get("pred_chi", float("nan"))))
                    rough.append(float(obj.get("rough_v", float("nan"))))
                n_rows = len(st_rows)
            last = st_rows[-1]
            try:
                last_iter = int(last.get("iter", last_iter or 0))
                last_iset = int(last.get("iset", last_iset or 0))
            except (TypeError, ValueError):
                pass
            if last.get("chi2") is not None:
                last_chi = float(last["chi2"])
            if last.get("rms") is not None:
                last_rms = float(last["rms"])
            if last.get("pred_chi") is not None:
                last_pred = float(last["pred_chi"])
            if last.get("rough_v") is not None:
                last_rv = float(last["rough_v"])
            note = (note + " · " if note else "") + f"status.jsonl×{status_n}"
    elif n_rows == 0:
        note = note or "等待 -L / status.jsonl…"

    smesh_path = None
    smesh_mtime = None
    ray_stamp = ""
    tres_stamp = ""
    if spec.out_root is not None:
        try:
            smesh_path = find_latest_inverse_smesh(spec.out_root)
            smesh_mtime = smesh_path.stat().st_mtime
        except (OSError, FileNotFoundError):
            if not note:
                note = "尚无 smesh 写出（需 -O；中间步需未开 -l）"
        try:
            from .ray_sample import ray_files_change_stamp

            if include_rays:
                ray_stamp = ray_files_change_stamp(spec.out_root)
        except Exception:
            ray_stamp = ""
        try:
            from .tres_sample import tres_files_change_stamp

            tres_stamp = tres_files_change_stamp(spec.out_root)
        except Exception:
            tres_stamp = ""

    return InvMonitorSnapshot(
        log_path=log_path,
        n_rows=n_rows,
        last_iter=last_iter,
        last_iset=last_iset,
        last_chi=last_chi,
        last_rms=last_rms,
        last_pred_chi=last_pred,
        last_rough_v=last_rv,
        iters=iters,
        chi=chi,
        rms=rms,
        pred_chi=pred,
        rough_v=rough,
        smesh_path=smesh_path,
        smesh_mtime=smesh_mtime,
        ray_stamp=ray_stamp,
        tres_stamp=tres_stamp,
        status_path=status_path,
        status_n=status_n,
        note=note,
    )


def format_monitor_status(snap: InvMonitorSnapshot, *, niter: int | None = None) -> str:
    parts: list[str] = []
    if snap.last_iter is not None:
        if niter:
            parts.append(f"iter {snap.last_iter}/{niter}")
        else:
            parts.append(f"iter {snap.last_iter}")
        if snap.last_iset is not None:
            parts.append(f"iset {snap.last_iset}")
    if snap.last_chi is not None:
        parts.append(f"χ²={snap.last_chi:.4g}")
    if snap.last_rms is not None:
        parts.append(f"RMS={snap.last_rms:.4g}")
    if snap.smesh_path is not None:
        parts.append(f"smesh={snap.smesh_path.name}")
    if snap.status_n:
        parts.append(f"jsonl={snap.status_n}")
    if snap.note:
        parts.append(snap.note)
    return " · ".join(parts) if parts else "等待数据…"


def progress_fraction(snap: InvMonitorSnapshot, *, niter: int | None = None) -> tuple[int, int]:
    """返回 (current, maximum) 供进度条；maximum=0 表示未知上限。"""
    cur = int(snap.last_iter) if snap.last_iter is not None else 0
    mx = int(niter) if niter and int(niter) > 0 else 0
    if mx > 0 and cur > mx:
        cur = mx
    return cur, mx


@dataclass
class ModelCandidate:
    """一次迭代写出的模型 + 可选过程指标（供手工挑选，非自动最优）。"""

    path: Path
    iter: int
    iset: int
    chi2: float | None = None
    rms: float | None = None
    pred_chi: float | None = None
    rough_v: float | None = None
    w_sv: float | None = None
    w_sd: float | None = None
    w_dv: float | None = None
    w_dd: float | None = None
    run_name: str | None = None
    mtime: float = 0.0


def _metrics_by_iter_iset(spec: InvMonitorSpec) -> dict[tuple[int, int], dict[str, float]]:
    """从 -L / status.jsonl 建立 (iter,iset) → 指标（后者覆盖前者）。"""
    out: dict[tuple[int, int], dict[str, float]] = {}
    log_path = spec.resolve_log()
    if log_path is not None:
        try:
            rows = parse_tt_inverse_log(log_path)
        except OSError:
            rows = []
        for r in rows:
            try:
                key = (int(r[COL_ITER]), int(r[COL_ISET]))
            except (TypeError, ValueError, IndexError):
                continue
            out[key] = {
                "chi2": float(r[COL_CHI_TOT]),
                "rms": float(r[COL_RMS_TOT]),
                "pred_chi": float(r[COL_PRED_CHI]),
                "rough_v": float(r[COL_LMVH]) + float(r[COL_LMVV]),
                "w_sv": float(r[COL_W_SV]),
                "w_sd": float(r[COL_W_SD]),
                "w_dv": float(r[COL_W_DV]),
                "w_dd": float(r[COL_W_DD]),
            }
    st = spec.resolve_status()
    if st is not None:
        for obj in parse_status_jsonl(st):
            try:
                key = (int(obj["iter"]), int(obj["iset"]))
            except (KeyError, TypeError, ValueError):
                continue
            m = out.get(key, {}).copy()
            for src, dst in (
                ("chi2", "chi2"),
                ("rms", "rms"),
                ("pred_chi", "pred_chi"),
                ("rough_v", "rough_v"),
            ):
                if obj.get(src) is not None:
                    try:
                        m[dst] = float(obj[src])
                    except (TypeError, ValueError):
                        pass
            out[key] = m
    return out


def build_model_catalog(spec: InvMonitorSpec) -> list[ModelCandidate]:
    """列出 out_root 下各轮 smesh，并尽量挂上 χ²/RMS 等指标。"""
    if spec.out_root is None:
        return []
    metrics = _metrics_by_iter_iset(spec)
    cands: list[ModelCandidate] = []
    for path, it, iset in list_inverse_smesh_files(spec.out_root):
        m = metrics.get((it, iset), {})
        try:
            mtime = float(path.stat().st_mtime)
        except OSError:
            mtime = 0.0
        cands.append(
            ModelCandidate(
                path=path,
                iter=it,
                iset=iset,
                chi2=m.get("chi2"),
                rms=m.get("rms"),
                pred_chi=m.get("pred_chi"),
                rough_v=m.get("rough_v"),
                w_sv=m.get("w_sv"),
                w_sd=m.get("w_sd"),
                w_dv=m.get("w_dv"),
                w_dd=m.get("w_dd"),
                run_name=spec.run_dir.name if spec.run_dir is not None else None,
                mtime=mtime,
            )
        )
    return cands
