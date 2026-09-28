#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tt_inverse 可复现运行包：在 work_dir/runs/<可读目录名>/ 下创建 inputs/、outputs/，
目录名默认含 mesh/data 文件名主干、**本地**紧凑时间戳与短随机后缀；可选备注插入其中。
快照输入并写入 manifest.json，便于日后对照命令行与哈希复现。

运行结束后将 ``outputs/`` 扁平产物归入 ``models/`` ``residuals/`` ``rays/``
``dws/`` ``logs/`` ``other/``；**不**自动写 final.smesh（选用由用户决定）。
"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import re
import shutil
import copy


SCHEMA_VERSION = 2

# 子进程仍写到 outputs/ 扁平文件；结束后按类型移入子目录（不写 final.smesh）
_KIND_SUBDIR = {
    "model": "models",
    "reflector": "models",
    "residual": "residuals",
    "ray": "rays",
    "dws": "dws",
    "log": "logs",
    "other": "other",
}

_RE_SMESH = re.compile(r"^.+\.smesh\.\d+\.\d+$", re.IGNORECASE)
_RE_REFL = re.compile(r"^.+\.refl\.\d+\.\d+$", re.IGNORECASE)
_RE_TRES = re.compile(r"^.+\.tres\.\d+\.\d+$", re.IGNORECASE)
_RE_OUTLIERS = re.compile(
    r"^.+\.outliers\.(\d+\.\d+|final)$", re.IGNORECASE
)
_RE_RAY = re.compile(r"^.+\.ray\.\d+\.\d+$", re.IGNORECASE)
_RE_RGRAV = re.compile(r"^.+\.rgrav\.\d+$", re.IGNORECASE)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _local_run_dir_timestamp() -> str:
    """运行包目录名用：本机本地日历时间的紧凑串（``%Y%m%dT%H%M%S``，无 Z 后缀）。"""
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _resolve_under_work(work_dir: Path, p: str) -> Path:
    s = (p or "").strip()
    if not s:
        raise ValueError("空路径")
    path = Path(s)
    if path.is_absolute():
        return path.resolve()
    return (work_dir / path).resolve()


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _slug_segment(text: str, max_len: int, *, tail: bool = False) -> str:
    """目录名片段：保留字母数字与 -_，其余转为下划线并压缩。

    ``tail=True`` 时在超长情况下保留**末尾**字符（适合文件名主干，避免 ``ph8_13`` 被截成 ``ph8_1``）。
    """
    s = (text or "").strip()
    if not s:
        return "x"
    out: List[str] = []
    for c in s:
        if c.isalnum() or c in "-_":
            out.append(c)
        else:
            out.append("_")
    t = re.sub(r"_+", "_", "".join(out)).strip("_")
    t = t or "x"
    if len(t) <= max_len:
        return t
    return t[-max_len:] if tail else t[:max_len]


def _slug_path_stem(path_str: str, max_len: int = 20) -> str:
    try:
        stem = Path(str(path_str).replace("\\", "/")).stem
    except Exception:
        stem = ""
    # 路径主干超长时保留后缀，以免尾部版本号/相位号被截断
    return _slug_segment(stem, max_len, tail=True)


def make_tt_inverse_run_dir_name(
    mesh: str,
    data: str,
    *,
    run_label: Optional[str] = None,
    run_id: Optional[str] = None,
    stamp: Optional[str] = None,
    max_total_len: int = 180,
) -> str:
    """
    运行包目录名：``ttinv[_备注]_<mesh 主干>_<data 主干>_<本地时间戳>_<6位hex>``。
    过长时逐步缩短备注与 mesh/data 主干，末尾时间戳与随机后缀始终保留以防碰撞。
    """
    stamp_part = stamp or _local_run_dir_timestamp()
    rid = run_id or secrets.token_hex(3)
    suffix = f"{stamp_part}_{rid}"
    lab_raw = (run_label or "").strip()
    for lm, ld, ll in (
        (48, 48, 32),
        (36, 36, 28),
        (28, 28, 24),
        (20, 20, 20),
        (16, 16, 16),
        (12, 12, 12),
        (8, 8, 0),
    ):
        m = _slug_path_stem(mesh, lm)
        d = _slug_path_stem(data, ld)
        parts = ["ttinv"]
        if ll > 0 and lab_raw:
            parts.append(_slug_segment(lab_raw, ll))
        parts.extend([m, d, suffix])
        name = "_".join(parts)
        if len(name) <= max_total_len:
            return name
    return name[:max_total_len].rstrip("_")


def _outputs_basename(user_val: str, default: str) -> str:
    s = (user_val or "").strip()
    if not s:
        return default
    return Path(s.replace("\\", "/")).name or default


def _bundle_input_rel(role: str, path_str: str) -> str:
    """与 ``add_input_file`` 首次分配名一致：``inputs/{role}{源后缀}``。"""
    ext = Path(str(path_str)).suffix or ""
    return f"inputs/{role}{ext}"


def _apply_bundle_output_path_strings(kw: dict) -> None:
    """将 ``log_file`` / ``out_root`` / ``dws_file`` 及联合重力的 ``grav_dws`` 改到 ``outputs/`` 下（就地修改）。"""
    log_bn = _outputs_basename(str(kw.get("log_file") or ""), "tt_inverse.log")
    kw["log_file"] = f"outputs/{log_bn}"

    out_bn = _outputs_basename(str(kw.get("out_root") or ""), "out")
    kw["out_root"] = f"outputs/{out_bn}"

    dws_raw = str(kw.get("dws_file") or "").strip()
    if dws_raw:
        dws_bn = _outputs_basename(dws_raw, "dws.dat")
    else:
        dws_bn = "dws.dat"
    kw["dws_file"] = f"outputs/{dws_bn}"

    g = kw.get("gravity_opts")
    if not g:
        return
    g = dict(g)
    if g.get("grav_dws"):
        gd_bn = _outputs_basename(str(g["grav_dws"]), "grav_dws.dat")
        g["grav_dws"] = f"outputs/{gd_bn}"
    else:
        g["grav_dws"] = "outputs/grav_dws.dat"
    kw["gravity_opts"] = g


def bundle_argv_preview_paths(
    mesh: str, data: str, kwargs: Mapping[str, Any]
) -> tuple[str, str, dict]:
    """
    仅用于 GUI 预览：模拟 ``runs/<run_dir>/`` 下子进程所见的相对路径 argv，
    不创建目录、不复制文件；与 ``TtInverseBundleBuilder.prepare`` 的路径规则一致。
    """
    kw = copy.deepcopy(dict(kwargs))
    kw.pop("_refl_stride", None)
    mesh_r = _bundle_input_rel("mesh", mesh)
    data_r = _bundle_input_rel("data", data)

    if kw.get("refl_file"):
        kw["refl_file"] = _bundle_input_rel("refl", str(kw["refl_file"]))
    if kw.get("seafloor_file"):
        kw["seafloor_file"] = _bundle_input_rel("seafloor", str(kw["seafloor_file"]))
    if kw.get("filter_bound_file"):
        kw["filter_bound_file"] = _bundle_input_rel("bound", str(kw["filter_bound_file"]))

    smooth = kw.get("smooth_opts")
    if smooth:
        smooth = dict(smooth)
        if smooth.get("corr_v_fn"):
            smooth["corr_v_fn"] = _bundle_input_rel("corr_v", str(smooth["corr_v_fn"]))
        if smooth.get("corr_d_fn"):
            smooth["corr_d_fn"] = _bundle_input_rel("corr_d", str(smooth["corr_d_fn"]))
        kw["smooth_opts"] = smooth

    damp = kw.get("damp_opts")
    if damp and damp.get("damp_v_fn"):
        damp = dict(damp)
        damp["damp_v_fn"] = _bundle_input_rel("damp_v", str(damp["damp_v_fn"]))
        kw["damp_opts"] = damp

    g = kw.get("gravity_opts")
    if g:
        g = dict(g)
        if g.get("grav_file"):
            g["grav_file"] = _bundle_input_rel("grav", str(g["grav_file"]))
        cont = g.get("continent")
        if cont:
            p, iconv = cont
            g["continent"] = (_bundle_input_rel("grav_ZC", str(p)), iconv)
        ou = g.get("ocean_upper")
        if ou:
            up, lo, iconv = ou
            g["ocean_upper"] = (
                _bundle_input_rel("grav_ZU_up", str(up)),
                _bundle_input_rel("grav_ZU_lo", str(lo)),
                iconv,
            )
        ol = g.get("ocean_lower")
        if ol:
            up, iconv = ol
            g["ocean_lower"] = (_bundle_input_rel("grav_ZL", str(up)), iconv)
        sed = g.get("sediment")
        if sed:
            up, lo, iconv = sed
            g["sediment"] = (
                _bundle_input_rel("grav_ZS_up", str(up)),
                _bundle_input_rel("grav_ZS_lo", str(lo)),
                iconv,
            )
        kw["gravity_opts"] = g

    _apply_bundle_output_path_strings(kw)
    return mesh_r, data_r, kw


@dataclass
class TtInverseBundleResult:
    """供 TomoAnd 在 run_dir 下调用 tt_inverse 的路径与 manifest 路径。"""

    run_dir: Path
    mesh: str
    data: str
    kwargs: dict
    manifest_path: Path
    inputs_manifest: List[dict] = field(default_factory=list)


class TtInverseBundleBuilder:
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir.resolve()
        self._src_to_rel: Dict[Path, str] = {}
        self._used_names: set[str] = set()
        self.inputs_rows: List[dict] = []

    def _alloc_name(self, role: str, src: Path) -> str:
        ext = src.suffix or ""
        base = f"{role}{ext}"
        if base not in self._used_names:
            self._used_names.add(base)
            return base
        n = 2
        while True:
            cand = f"{role}_{n}{ext}"
            if cand not in self._used_names:
                self._used_names.add(cand)
                return cand
            n += 1

    def add_input_file(
        self, role: str, path_str: str, run_dir: Path, *, stride: int = 1
    ) -> str:
        full = _resolve_under_work(self.work_dir, path_str)
        if not full.is_file():
            raise FileNotFoundError(
                f"tt_inverse 运行包：找不到输入文件 ({role})\n"
                f"  参数路径: {path_str!r}\n"
                f"  解析为: {full}"
            )
        key = full.resolve()
        if key in self._src_to_rel:
            return self._src_to_rel[key]
        name = self._alloc_name(role, full)
        dest = run_dir / "inputs" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if role == "refl" and stride > 1:
            from .gui.services.refl_stride import write_subsampled_refl

            write_subsampled_refl(full, dest, stride)
            hashed = dest
        else:
            shutil.copy2(full, dest)
            hashed = dest
        rel = f"inputs/{name}"
        self._src_to_rel[key] = rel
        row = {
            "role": role,
            "path_in_run": rel,
            "original_path": str(full),
            "sha256": _sha256_file(hashed),
            "bytes": hashed.stat().st_size,
        }
        if role == "refl" and stride > 1:
            row["refl_stride"] = stride
        self.inputs_rows.append(row)
        return rel

    def prepare(
        self,
        mesh: str,
        data: str,
        kwargs: Mapping[str, Any],
        *,
        run_id: Optional[str] = None,
        run_label: Optional[str] = None,
    ) -> TtInverseBundleResult:
        stamp = _local_run_dir_timestamp()
        rid = run_id or secrets.token_hex(3)
        dir_name = make_tt_inverse_run_dir_name(
            mesh, data, run_label=run_label, run_id=rid, stamp=stamp
        )
        run_dir = self.work_dir / "runs" / dir_name
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "inputs").mkdir(exist_ok=True)
        (run_dir / "outputs").mkdir(exist_ok=True)

        kw = copy.deepcopy(dict(kwargs))
        from .gui.services.refl_stride import pop_refl_stride

        refl_stride = pop_refl_stride(kw)

        mesh_r = self.add_input_file("mesh", mesh, run_dir)
        data_r = self.add_input_file("data", data, run_dir)

        if kw.get("refl_file"):
            kw["refl_file"] = self.add_input_file(
                "refl", str(kw["refl_file"]), run_dir, stride=refl_stride
            )
        if kw.get("seafloor_file"):
            kw["seafloor_file"] = self.add_input_file(
                "seafloor", str(kw["seafloor_file"]), run_dir
            )
        if kw.get("filter_bound_file"):
            kw["filter_bound_file"] = self.add_input_file(
                "bound", str(kw["filter_bound_file"]), run_dir
            )

        smooth = kw.get("smooth_opts")
        if smooth:
            smooth = dict(smooth)
            if smooth.get("corr_v_fn"):
                smooth["corr_v_fn"] = self.add_input_file("corr_v", str(smooth["corr_v_fn"]), run_dir)
            if smooth.get("corr_d_fn"):
                smooth["corr_d_fn"] = self.add_input_file("corr_d", str(smooth["corr_d_fn"]), run_dir)
            kw["smooth_opts"] = smooth

        damp = kw.get("damp_opts")
        if damp and damp.get("damp_v_fn"):
            damp = dict(damp)
            damp["damp_v_fn"] = self.add_input_file("damp_v", str(damp["damp_v_fn"]), run_dir)
            kw["damp_opts"] = damp

        g = kw.get("gravity_opts")
        if g:
            g = dict(g)
            if g.get("grav_file"):
                g["grav_file"] = self.add_input_file("grav", str(g["grav_file"]), run_dir)
            cont = g.get("continent")
            if cont:
                p, iconv = cont
                g["continent"] = (self.add_input_file("grav_ZC", str(p), run_dir), iconv)
            ou = g.get("ocean_upper")
            if ou:
                up, lo, iconv = ou
                g["ocean_upper"] = (
                    self.add_input_file("grav_ZU_up", str(up), run_dir),
                    self.add_input_file("grav_ZU_lo", str(lo), run_dir),
                    iconv,
                )
            ol = g.get("ocean_lower")
            if ol:
                up, iconv = ol
                g["ocean_lower"] = (self.add_input_file("grav_ZL", str(up), run_dir), iconv)
            sed = g.get("sediment")
            if sed:
                up, lo, iconv = sed
                g["sediment"] = (
                    self.add_input_file("grav_ZS_up", str(up), run_dir),
                    self.add_input_file("grav_ZS_lo", str(lo), run_dir),
                    iconv,
                )
            kw["gravity_opts"] = g

        _apply_bundle_output_path_strings(kw)

        manifest_path = run_dir / "manifest.json"
        return TtInverseBundleResult(
            run_dir=run_dir,
            mesh=mesh_r,
            data=data_r,
            kwargs=kw,
            manifest_path=manifest_path,
            inputs_manifest=list(self.inputs_rows),
        )


def write_tt_inverse_manifest(
    *,
    manifest_path: Path,
    work_dir: Path,
    run_dir: Path,
    executable_resolved: str,
    argv: List[str],
    mesh: str,
    data: str,
    kwargs: Mapping[str, Any],
    inputs_rows: List[dict],
    status: str = "planned",
    gui_profile: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    ``gui_profile`` 与 GUI「保存配置」JSON 同结构（写入前经与「保存配置」相同的 work_dir 路径规范化），
    与 argv / ``python_replay`` 中运行包 ``inputs/``、``outputs/`` 路径区分。
    ``python_replay`` 为实际子进程所用的相对 ``run_dir`` 的路径，便于复现与对照。
    """
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "pyaobs.tomo2d.tt_inverse",
        "created_utc": _utc_now_iso(),
        "work_dir": str(work_dir.resolve()),
        "run_dir": str(run_dir.resolve()),
        "proc_cwd_note": "子进程 cwd 应设为 run_dir；命令行中路径均相对 run_dir。",
        "executable": {"name": "tt_inverse", "resolved_path": executable_resolved},
        "argv": argv,
        "inputs": inputs_rows,
        "python_replay": {
            "mesh": mesh,
            "data": data,
            "kwargs": _json_sanitize(dict(kwargs)),
        },
        "post_run": {
            "status": status,
            "exit_code": None,
            "finished_utc": None,
            "error": None,
            "output_files": [],
        },
    }
    if gui_profile is not None:
        manifest["gui_profile"] = _json_sanitize(dict(gui_profile))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def classify_tt_inverse_output_kind(filename: str) -> str:
    """按 tt_inverse 命名约定判定文件类型（非「最优」评判）。"""
    name = Path(filename).name
    if _RE_SMESH.match(name):
        return "model"
    if _RE_REFL.match(name):
        return "reflector"
    if _RE_TRES.match(name) or _RE_RGRAV.match(name) or _RE_OUTLIERS.match(name):
        return "residual"
    if _RE_RAY.match(name):
        return "ray"
    low = name.lower()
    if "dws" in low:
        return "dws"
    if low.endswith(".log") or low.endswith(".jsonl") or low == "status.jsonl":
        return "log"
    return "other"


def classify_tt_inverse_outputs(outputs_dir: Path | str) -> List[dict]:
    """
    将 ``outputs/``（或任意目录）下**直接子文件**按类型移入子目录：

    ``models/`` ``residuals/`` ``rays/`` ``dws/`` ``logs/`` ``other/``

    不创建 final.smesh；选用哪次模型由用户事后决定。
    已位于上述子目录中的文件不重复移动（幂等）。
    返回归类后的条目 ``{path, kind, bytes}``（``path`` 相对 ``outputs_dir``）。
    """
    root = Path(outputs_dir)
    if not root.is_dir():
        return []

    reserved = set(_KIND_SUBDIR.values())
    moved: List[tuple[Path, str]] = []

    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        kind = classify_tt_inverse_output_kind(p.name)
        sub = _KIND_SUBDIR[kind]
        dest_dir = root / sub
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / p.name
        if dest.resolve() == p.resolve():
            moved.append((dest, kind))
            continue
        if dest.exists():
            # 避免覆盖：保留已有目标，删除源仅当内容路径冲突极少见
            dest = dest_dir / f"{p.stem}_{secrets.token_hex(2)}{p.suffix}"
        try:
            shutil.move(str(p), str(dest))
            moved.append((dest, kind))
        except OSError:
            moved.append((p, kind))

    # 已在子目录中的文件一并列入清单
    listed: Dict[Path, str] = {p.resolve(): k for p, k in moved}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        try:
            rel_parts = p.relative_to(root).parts
        except ValueError:
            continue
        if len(rel_parts) >= 2 and rel_parts[0] in reserved:
            key = p.resolve()
            if key not in listed:
                listed[key] = classify_tt_inverse_output_kind(p.name)

    rows: List[dict] = []
    for p_res, kind in sorted(listed.items(), key=lambda kv: str(kv[0])):
        p = Path(p_res)
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            rel = p.name
        try:
            sz = p.stat().st_size
        except OSError:
            sz = -1
        rows.append({"path": rel, "kind": kind, "bytes": sz})
    return rows


def finalize_tt_inverse_manifest(
    manifest_path: Path,
    *,
    result: Any,
    err: Optional[BaseException],
) -> None:
    """在子进程结束后归类 outputs/ 并更新 manifest（工作线程内调用即可）。"""
    path = Path(manifest_path)
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    run_dir = path.parent
    od = run_dir / "outputs"
    classified = classify_tt_inverse_outputs(od) if od.is_dir() else []
    outs: List[dict] = []
    for row in classified:
        outs.append(
            {
                "path": f"outputs/{row['path']}",
                "kind": row["kind"],
                "bytes": row["bytes"],
            }
        )

    finished = _utc_now_iso()
    pr: Dict[str, Any] = {
        "finished_utc": finished,
        "output_files": outs,
        "output_layout": {
            "note": (
                "outputs/ 下按类型分子目录；不自动选定最优模型。"
                "请结合 -L 日志、参数与地质认识后自行选用 models/ 中某次迭代。"
            ),
            "dirs": {
                "models": "smesh / refl（按 iter.iset）",
                "residuals": "tres / rgrav",
                "rays": "ray（需较高 out_level）",
                "dws": "dws / grav_dws",
                "logs": "tt_inverse.log 等",
                "other": "未能识别的文件",
            },
        },
    }
    if err is None and result is not None:
        pr["status"] = "finished"
        pr["exit_code"] = int(getattr(result, "returncode", 0) or 0)
        pr["error"] = None
    else:
        pr["status"] = "failed"
        rc = getattr(err, "returncode", None)
        pr["exit_code"] = int(rc) if rc is not None else None
        pr["error"] = repr(err) if err else None

    data["post_run"] = {**data.get("post_run", {}), **pr}
    data["schema_version"] = SCHEMA_VERSION
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def build_tt_inverse_bundle(
    work_dir: Path | str,
    mesh: str,
    data: str,
    kwargs: Mapping[str, Any],
    *,
    run_label: Optional[str] = None,
) -> TtInverseBundleResult:
    """在工作目录下创建 ``runs/<可读目录名>/``，快照输入并重写 kwargs 中的路径。"""
    return TtInverseBundleBuilder(Path(work_dir)).prepare(
        mesh, data, kwargs, run_label=run_label
    )
