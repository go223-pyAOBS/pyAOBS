"""RAYINVR 公用调用层。

供 ``vedit`` / ``zplotpy`` / ``iphase`` 等复用，避免各处各自 ``chdir`` +
``RayinvrWrapper`` / 原生 ``rayinvr`` 的分叉实现。

约定
----
- 工作目录内需有 ``r.in``；``vfile`` / ``tfile`` 默认 ``v.in`` / ``tx.in``，
  也可由 ``r.in`` namelist 指定。
- ``backend``:
  - ``auto``: 优先 PATH 中的 ``rayinvr`` 可执行文件，否则子进程 Wrapper，
    再否则进程内 Wrapper；若 ``collect_rays``/``collect_obs`` 则强制进程内。
  - ``exe`` / ``wrapper_subprocess`` / ``wrapper_inprocess``
- 射线路径与观测数组来自 ``librayinvr`` 进程内状态；``run_rayinvr_collect``
  默认在**子进程**内运行并尝试收集，避免 GUI 线程 segfault；``tx.out`` 始终从磁盘读取。
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

Backend = Literal["auto", "exe", "wrapper_subprocess", "wrapper_inprocess"]


@dataclass
class RayinvrInputSpec:
    r_file: Path
    t_file: Path
    v_file: Path
    tfile_from_rin: bool = False
    vfile_from_rin: bool = False


@dataclass
class RayinvrResult:
    success: bool
    code: str
    message: str
    working_dir: str
    elapsed_s: float
    ran_forward: bool
    used_existing_txout: bool = False
    missing_inputs: tuple[str, ...] = ()
    tx_out_path: Optional[str] = None
    rays: list = field(default_factory=list)
    obs: Any = None
    backend: str = ""
    wrapper: Any = None  # 仅 wrapper_inprocess 时可能非空，供后续同进程取数


def hash_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_rin_input_files(working_dir: str | Path) -> RayinvrInputSpec:
    """从 ``r.in`` 解析 ``tfile`` / ``vfile``；缺失则回退 ``tx.in`` / ``v.in``。"""
    wd = Path(working_dir)
    r_file = wd / "r.in"
    t_name = "tx.in"
    v_name = "v.in"
    t_from = False
    v_from = False
    if r_file.exists():
        txt = r_file.read_text(encoding="utf-8", errors="ignore")
        m_t = re.search(
            r"\btfile\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([^,\s]+))",
            txt,
            flags=re.IGNORECASE,
        )
        if m_t:
            t_name = (m_t.group(1) or m_t.group(2) or m_t.group(3) or t_name).strip()
            t_from = True
        m_v = re.search(
            r"\bvfile\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([^,\s]+))",
            txt,
            flags=re.IGNORECASE,
        )
        if m_v:
            v_name = (m_v.group(1) or m_v.group(2) or m_v.group(3) or v_name).strip()
            v_from = True
    return RayinvrInputSpec(
        r_file=r_file,
        t_file=(wd / t_name),
        v_file=(wd / v_name),
        tfile_from_rin=t_from,
        vfile_from_rin=v_from,
    )


def validate_rayinvr_inputs(
    working_dir: str | Path,
    *,
    require_tx: bool = True,
) -> tuple[bool, tuple[str, ...], RayinvrInputSpec]:
    """校验工作目录输入。``require_tx=False`` 时允许无观测文件（仅正演路径）。"""
    spec = parse_rin_input_files(working_dir)
    miss: list[str] = []
    if not spec.r_file.exists():
        miss.append(spec.r_file.name)
    if not spec.v_file.exists():
        miss.append(spec.v_file.name)
    if require_tx and not spec.t_file.exists():
        miss.append(spec.t_file.name)
    miss_t = tuple(miss)
    return len(miss_t) == 0, miss_t, spec


_COLLECT_PKL = ".rayinvr_collect.pkl"

# 连续追踪时若不清除，Fortran 可能读到上一轮 p.out / tx.out 等残留
_RAYINVR_STALE_OUTPUTS = (
    "r1.out",
    "r2.out",
    "i.out",
    "p.out",
    "tx.out",
    "tx.out.bak",
    "m.out",
    "vm.out",
    _COLLECT_PKL,
)


def clean_rayinvr_workdir(working_dir: str | Path) -> None:
    """删除工作目录内上一轮 RAYINVR 输出（输入 v.in/r.in/tx.in 不动）。"""
    wd = Path(working_dir)
    if not wd.is_dir():
        return
    for name in _RAYINVR_STALE_OUTPUTS:
        p = wd / name
        try:
            if p.is_file():
                p.unlink()
        except OSError:
            pass


def _copy_if_needed(src: Path, dst: Path) -> None:
    src = Path(src)
    dst = Path(dst)
    if not src.is_file():
        raise FileNotFoundError(f"源文件不存在: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if src.resolve() == dst.resolve():
            return
    except Exception:
        pass
    shutil.copy2(src, dst)


def prepare_rayinvr_workdir(
    working_dir: str | Path,
    *,
    vin_src: str | Path | None = None,
    rin_src: str | Path | None = None,
    tx_src: str | Path | None = None,
    tx_shot_xs: Sequence[float] | None = None,
    trapar_xs: Sequence[float] | None = None,
    trapar_zs: Sequence[float] | None = None,
    forward_tracing: bool = True,
) -> Path:
    """把 ``v.in`` / ``r.in`` / ``tx.in``（或 r.in 指定名）准备到工作目录。

    顺序：先放 ``r.in``（以便解析 vfile/tfile），再按解析结果放置模型与观测。
    *tx_shot_xs* 非空时仅写入这些炮点对应的 ``tx.in`` 块（与 TRAPAR xshot 一致）。
    *trapar_xs* / *trapar_zs* 非空时在**工作副本**上重写 TRAPAR（避免源 r.in 与选中 OBS 不一致）。
    *forward_tracing* 为真时在工作副本设 ``invr=0``（正演/显示追踪；``invr=1`` 在本库+haiti 模型下会在 calmod 段错误）。
    """
    wd = Path(working_dir)
    wd.mkdir(parents=True, exist_ok=True)

    clean_rayinvr_workdir(wd)

    if rin_src is not None:
        _copy_if_needed(Path(rin_src), wd / "r.in")
    elif not (wd / "r.in").exists():
        raise FileNotFoundError(f"工作目录缺少 r.in: {wd / 'r.in'}")

    if trapar_xs is not None and trapar_zs is not None:
        from pyAOBS.modeling.rayinvr.ray_collect import write_rin_trapar_shots

        xs = [float(x) for x in trapar_xs]
        zs = [float(z) for z in trapar_zs]
        if xs and len(xs) == len(zs):
            write_rin_trapar_shots(wd / "r.in", wd / "r.in", xs, zs)

    if forward_tracing:
        from pyAOBS.modeling.rayinvr.ray_collect import ensure_rin_invr

        ensure_rin_invr(wd / "r.in", 0)

    spec = parse_rin_input_files(wd)

    if vin_src is not None:
        _copy_if_needed(Path(vin_src), spec.v_file)
    elif not spec.v_file.exists():
        # 常见情况：目录里有任意名模型，但未指定；要求调用方显式给 vin_src
        raise FileNotFoundError(f"工作目录缺少速度模型: {spec.v_file}")

    if tx_src is not None:
        src = Path(tx_src)
        if tx_shot_xs:
            from pyAOBS.modeling.rayinvr.tx_io import (
                filter_tx_dataset_by_shot_xs,
                read_tx_file,
                write_tx_file,
            )

            ds = filter_tx_dataset_by_shot_xs(
                read_tx_file(src), tx_shot_xs
            )
            if ds.n_picks == 0:
                raise ValueError(
                    f"tx.in 中无与当前 xshot {list(tx_shot_xs)} 匹配的观测块"
                )
            write_tx_file(ds, spec.t_file)
        else:
            _copy_if_needed(src, spec.t_file)

    _auto_adjust_workdir_ximax(wd, tx_shot_xs=tx_shot_xs)

    return wd


def _tx_out_usable(path: Path) -> bool:
    """``tx.out`` 须非空（Fortran 失败时可能留下 0 字节文件）。"""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _auto_adjust_workdir_ximax(
    wd: Path,
    *,
    tx_shot_xs: Sequence[float] | None = None,
) -> None:
    """``invr=1`` 时按拾取偏移抬升工作副本 ``ximax``，避免长偏移台站 0 射线。"""
    from pyAOBS.modeling.rayinvr.ray_collect import (
        ensure_rin_ximax_at_least,
        max_pick_offset_km,
        parse_trapar_xshot,
        read_rin_invr,
    )

    rin = wd / "r.in"
    spec = parse_rin_input_files(wd)
    tx = spec.t_file
    if not rin.is_file() or not tx.is_file():
        return
    if read_rin_invr(rin) == 0:
        return
    xs = list(tx_shot_xs or []) or parse_trapar_xshot(rin)
    max_off = max_pick_offset_km(tx, xs if xs else None)
    if max_off <= 0:
        return
    # RAYINVR：偏导插值要求射线端点距拾取 ≤ ximax（km）
    min_xi = max(20.0, float(max_off) * 1.05)
    ensure_rin_ximax_at_least(rin, min_xi)


def find_rayinvr_executable() -> Optional[str]:
    """在 PATH 中探测原生 ``rayinvr`` 可执行文件。"""
    try:
        subprocess.run(
            ["rayinvr"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        return "rayinvr"
    except FileNotFoundError:
        return None
    except Exception:
        # 能找到但立刻失败（缺输入等）也视为可用
        return "rayinvr"


def _repo_root_for_subprocess() -> Path:
    """``modeling/rayinvr/service.py`` → 含 ``pyAOBS`` 包目录的上一级。"""
    return Path(__file__).resolve().parents[3]


def _subprocess_env(*extra_roots: Path) -> dict[str, str]:
    """子进程 ``PYTHONPATH``：保证 ``import pyAOBS.modeling...`` 可用。"""
    env = dict(os.environ)
    roots: list[str] = []
    here = Path(__file__).resolve()
    for p in (here.parents[3], here.parents[2].parent, *extra_roots):
        try:
            s = str(Path(p).resolve())
        except OSError:
            continue
        if s not in roots and Path(s).is_dir():
            roots.append(s)
    if roots:
        prev = env.get("PYTHONPATH", "")
        merged = os.pathsep.join(roots + ([prev] if prev else []))
        env["PYTHONPATH"] = merged
    return env


_SUBPROCESS_COLLECT_SCRIPT = Path(__file__).resolve().parent / "subprocess_collect.py"


# Fortran ``store_ray`` 上限：``prayt = pshot2 * prayf``（rayinvr.par）
RAYINVR_STORED_RAY_CAP = 24000


def _obs_from_tfile(working_dir: Path) -> Any:
    """从工作目录 ``tfile``（通常 tx.in）读取观测走时数组。"""
    spec = parse_rin_input_files(working_dir)
    if not spec.t_file.is_file():
        return None
    try:
        from pyAOBS.modeling.rayinvr.tx_io import read_tx_file

        return read_tx_file(spec.t_file).to_flat_arrays()
    except Exception:
        return None


def _load_subprocess_collect(pkl: Path) -> tuple[list, Any, str, dict]:
    import pickle

    if not pkl.is_file():
        return [], None, "", {}
    try:
        data = pickle.loads(pkl.read_bytes())
    except Exception as exc:
        return [], None, str(exc), {}
    if not isinstance(data, dict):
        return [], None, "collect pickle 格式无效", {}
    rays = data.get("rays") or []
    obs = data.get("obs")
    err = str(data.get("err") or "")
    timing = data.get("timing") or {}
    if not isinstance(rays, list):
        rays = []
    if not isinstance(timing, dict):
        timing = {}
    return rays, obs, err, timing


def prefer_inline_ray_collect() -> bool:
    """Linux/WSL 下在工作线程内直接调 librayinvr，避免子进程启动开销。"""
    import sys

    if sys.platform == "win32":
        return False
    lib = Path(__file__).resolve().parent / "librayinvr.so"
    return lib.is_file()


@contextlib.contextmanager
def _quiet_fortran_stdio():
    """inline 正演时静音 Fortran ``write(6)`` 与 Python ``print``（WSL 终端不刷屏）。"""
    if sys.platform == "win32":
        yield
        return
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    save_out = os.dup(1)
    save_err = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        try:
            os.dup2(save_out, 1)
            os.dup2(save_err, 2)
        finally:
            os.close(save_out)
            os.close(save_err)
            os.close(devnull_fd)


def _wrapper_collect_workdir(
    working_dir: Path,
    *,
    max_rays: int = 0,
    max_rays_per_shot: int = 0,
) -> tuple[bool, str, list, Any, dict]:
    """进程内运行 RAYINVR 并收集 rays/obs（供子进程脚本与 vedit 工作线程复用）。"""
    from pyAOBS.modeling.rayinvr.ray_collect import (
        RAYINVR_STORED_RAY_CAP,
        RAYS_PER_SHOT_UNLIMITED,
        collect_stored_rays,
        parse_trapar_xshot,
    )
    from pyAOBS.modeling.rayinvr.rayinvr_wrapper import RayinvrWrapper

    wd = Path(working_dir)
    cap = max_rays if max_rays > 0 else RAYINVR_STORED_RAY_CAP
    per_shot = int(max_rays_per_shot)
    if per_shot == 0:
        per_shot = RAYS_PER_SHOT_UNLIMITED

    ok = False
    msg = ""
    rays: list = []
    obs = None
    fortran_s = 0.0
    collect_s = 0.0
    t_job0 = time.perf_counter()
    try:
        wrap = RayinvrWrapper(working_dir=str(wd))
        t_f0 = time.perf_counter()
        with _quiet_fortran_stdio():
            ok = bool(wrap.run_rayinvr())
        fortran_s = time.perf_counter() - t_f0
        if ok:
            shot_xs = parse_trapar_xshot(wd / "r.in")
            t_c0 = time.perf_counter()
            try:
                rays, note = collect_stored_rays(
                    wrap,
                    max_rays=cap,
                    shot_xs=shot_xs or None,
                    max_rays_per_shot=per_shot,
                )
                if note:
                    msg += note + ";"
            except Exception as exc:
                msg += f"rays:{exc};"
            try:
                obs = wrap.get_observed_data()
            except Exception as exc:
                msg += f"obs:{exc};"
            collect_s = time.perf_counter() - t_c0
    except Exception as exc:
        msg = str(exc)
    timing = {
        "fortran_s": float(fortran_s),
        "collect_s": float(collect_s),
        "job_total_s": float(time.perf_counter() - t_job0),
    }
    return ok, msg.strip("; "), rays, obs, timing


def run_wrapper_in_subprocess_collect(
    working_dir: Path,
    *,
    max_rays: int = 0,
    timeout_s: int = 120,
) -> tuple[bool, str, list, Any, dict]:
    """子进程：运行 RAYINVR 并尝试收集 rays/obs（崩溃不影响主进程）。

    结果写入 ``working_dir/.rayinvr_collect.pkl``；``tx.out`` 由 Fortran 写盘。
    返回 ``(ok, msg, rays, obs, timing)``，``timing`` 含 ``fortran_s`` / ``collect_s``。
    """
    repo_root = _repo_root_for_subprocess()
    pkl = working_dir / _COLLECT_PKL
    try:
        if pkl.exists():
            pkl.unlink()
    except OSError:
        pass
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(_SUBPROCESS_COLLECT_SCRIPT),
                str(working_dir),
                str(repo_root),
                str(int(max_rays)),
                str(pkl),
            ],
            cwd=str(working_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            text=True,
            encoding="utf-8",
            errors="ignore",
            env=_subprocess_env(repo_root),
        )
    except subprocess.TimeoutExpired:
        return False, "RayinvrWrapper 子进程超时", [], None, {}
    except Exception as e:
        return False, f"RayinvrWrapper 子进程异常: {e}", [], None, {}

    rays, obs, perr, timing = _load_subprocess_collect(pkl)
    tx_out = working_dir / "tx.out"
    if _tx_out_usable(tx_out):
        msg = perr or ""
        if proc.returncode not in (0, None) and msg:
            msg = f"子进程 code={proc.returncode}; {msg}"
        return True, msg, rays, obs, timing

    out_tail = (proc.stdout or "").strip()
    if out_tail:
        out_tail = out_tail[-300:]
    if proc.returncode == 0:
        return False, "RAYINVR 完成但未生成 tx.out", rays, obs, timing
    detail = f"RayinvrWrapper 子进程失败 code={proc.returncode}"
    if out_tail:
        detail += f"; {out_tail}"
    if perr:
        detail += f"; {perr}"
    return False, detail, rays, obs, timing


def run_wrapper_in_subprocess(
    working_dir: Path, *, timeout_s: int = 120
) -> tuple[bool, str]:
    """子进程执行 ``RayinvrWrapper.run_rayinvr``，避免共享库状态残留。"""
    repo_root = _repo_root_for_subprocess()
    code = (
        "import sys;"
        "from pathlib import Path;"
        "wd=Path(sys.argv[1]);"
        "root=Path(sys.argv[2]);"
        "sys.path.insert(0, str(root));"
        "from pyAOBS.modeling.rayinvr.rayinvr_wrapper import RayinvrWrapper;"
        "ok=bool(RayinvrWrapper(working_dir=str(wd)).run_rayinvr());"
        "raise SystemExit(0 if ok else 2)"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code, str(working_dir), str(repo_root)],
            cwd=str(working_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            text=True,
            encoding="utf-8",
            errors="ignore",
            env=_subprocess_env(repo_root),
        )
    except subprocess.TimeoutExpired:
        return False, "RayinvrWrapper 子进程超时"
    except Exception as e:
        return False, f"RayinvrWrapper 子进程异常: {e}"
    if proc.returncode == 0:
        return True, ""
    msg = (proc.stdout or "").strip()
    if msg:
        msg = msg[-300:]
    detail = f"RayinvrWrapper 子进程失败 code={proc.returncode}"
    if proc.returncode == -11:
        detail += " (SIGSEGV)"
    if msg:
        detail += f"; {msg}"
    if proc.returncode == -11:
        if "velocity model" in msg.lower():
            detail += (
                "；haiti 等复杂模型在 invr=1 时 librayinvr 可能在 calmod 崩溃，"
                "工作副本应已强制 invr=0"
            )
        else:
            detail += "；请查看 cache/rayinvr/p.out / r1.out"
    return False, detail


def _run_exe(working_dir: Path, *, timeout_s: int) -> tuple[bool, str]:
    exe = find_rayinvr_executable()
    if exe is None:
        return False, "未找到 rayinvr 可执行文件"
    try:
        proc = subprocess.run(
            [exe],
            cwd=str(working_dir),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
        if proc.returncode != 0 and not (working_dir / "tx.out").exists():
            return False, f"rayinvr 退出码={proc.returncode}"
        return True, "exe"
    except subprocess.TimeoutExpired:
        return False, f"rayinvr 运行超时({timeout_s}s)"
    except Exception as e:
        return False, f"rayinvr 异常: {e}"


def _run_inprocess(
    working_dir: Path,
    *,
    collect_rays: bool,
    collect_obs: bool,
    max_rays: int,
) -> tuple[bool, str, Any, list, Any]:
    try:
        from pyAOBS.modeling.rayinvr.rayinvr_wrapper import RayinvrWrapper
    except Exception as exc:
        return (
            False,
            f"无法导入 RayinvrWrapper（可能缺少 librayinvr）: {exc}",
            None,
            [],
            None,
        )
    try:
        wrap = RayinvrWrapper(working_dir=str(working_dir))
        ok = bool(wrap.run_rayinvr())
        if not ok:
            return False, "run_rayinvr 返回 False", wrap, [], None
        rays: list = []
        obs = None
        collect_note = ""
        if collect_rays:
            from pyAOBS.modeling.rayinvr.ray_collect import (
                RAYINVR_STORED_RAY_CAP,
                collect_stored_rays,
                parse_trapar_xshot,
            )

            cap = max_rays if max_rays > 0 else RAYINVR_STORED_RAY_CAP
            shot_xs = parse_trapar_xshot(working_dir / "r.in")
            rays, collect_note = collect_stored_rays(
                wrap,
                max_rays=cap,
                shot_xs=shot_xs or None,
            )
        if collect_obs:
            try:
                obs = wrap.get_observed_data()
            except Exception:
                obs = None
        msg = "wrapper_inprocess"
        if collect_note:
            msg = f"{msg}; {collect_note}"
        return True, msg, wrap, rays, obs
    except Exception as exc:
        detail = str(exc)
        # Windows 加载 Linux .so 常见 WinError 193
        if "193" in detail or "WinError" in detail:
            detail += (
                "（当前平台可能无法加载 librayinvr.so；"
                "请在 Linux/WSL 运行，或安装原生 rayinvr 可执行文件）"
            )
        return False, f"RayinvrWrapper 进程内运行失败: {detail}", None, [], None


def _resolve_backend(
    backend: Backend, *, collect_rays: bool, collect_obs: bool
) -> Backend:
    if collect_rays or collect_obs:
        return "wrapper_inprocess"
    if backend != "auto":
        return backend
    if find_rayinvr_executable() is not None:
        return "exe"
    return "wrapper_subprocess"


def run_rayinvr(
    working_dir: str | Path,
    *,
    force_run: bool = True,
    require_tx: bool = True,
    collect_rays: bool = False,
    collect_obs: bool = False,
    max_rays: int = 0,
    timeout_s: int = 120,
    backend: Backend = "auto",
    tx_in_override: str | Path | None = None,
    sync_override_to_tfile: bool = True,
) -> RayinvrResult:
    """在工作目录运行 RAYINVR 正演。

    Parameters
    ----------
    force_run:
        ``False`` 且已有 ``tx.out`` 时直接复用（iphase 缓存语义）。
    require_tx:
        是否强制要求观测 ``tfile`` 存在。
    collect_rays / collect_obs:
        是否在进程内收集射线/观测（会强制 ``wrapper_inprocess``）。
    """
    wd = Path(working_dir)
    tx_out = wd / "tx.out"
    t0 = time.time()

    if tx_in_override is not None and sync_override_to_tfile:
        spec0 = parse_rin_input_files(wd)
        src = Path(tx_in_override)
        if src.exists():
            try:
                _copy_if_needed(src, spec0.t_file)
            except Exception as e:
                return RayinvrResult(
                    success=False,
                    code="sync_tfile_failed",
                    message=f"同步 tfile 失败: {e}",
                    working_dir=str(wd),
                    elapsed_s=time.time() - t0,
                    ran_forward=False,
                    tx_out_path=str(tx_out) if tx_out.exists() else None,
                )

    ok, miss, spec = validate_rayinvr_inputs(wd, require_tx=require_tx)
    if not ok:
        return RayinvrResult(
            success=False,
            code="missing_inputs",
            message=f"缺少输入文件: {', '.join(miss)}",
            working_dir=str(wd),
            elapsed_s=time.time() - t0,
            ran_forward=False,
            missing_inputs=miss,
            tx_out_path=str(tx_out) if tx_out.exists() else None,
        )

    if tx_out.exists() and not force_run and not collect_rays and not collect_obs:
        return RayinvrResult(
            success=True,
            code="ok",
            message="使用现有 tx.out",
            working_dir=str(wd),
            elapsed_s=time.time() - t0,
            ran_forward=False,
            used_existing_txout=True,
            tx_out_path=str(tx_out),
            backend="cache",
        )

    if tx_out.exists() and force_run:
        try:
            tx_out.unlink()
        except Exception:
            pass

    chosen = _resolve_backend(
        backend, collect_rays=collect_rays, collect_obs=collect_obs
    )
    wrap = None
    rays: list = []
    obs = None
    run_ok = False
    msg = ""

    if chosen == "exe":
        run_ok, msg = _run_exe(wd, timeout_s=timeout_s)
        if not run_ok and msg.startswith("未找到"):
            # auto 回退
            run_ok, msg = run_wrapper_in_subprocess(wd, timeout_s=timeout_s)
            chosen = "wrapper_subprocess"
    elif chosen == "wrapper_subprocess":
        run_ok, msg = run_wrapper_in_subprocess(wd, timeout_s=timeout_s)
    else:
        run_ok, msg, wrap, rays, obs = _run_inprocess(
            wd,
            collect_rays=collect_rays,
            collect_obs=collect_obs,
            max_rays=max_rays,
        )
        chosen = "wrapper_inprocess"

    if not run_ok:
        return RayinvrResult(
            success=False,
            code="run_failed",
            message=msg or "RAYINVR 运行失败",
            working_dir=str(wd),
            elapsed_s=time.time() - t0,
            ran_forward=True,
            tx_out_path=str(tx_out) if tx_out.exists() else None,
            backend=chosen,
            wrapper=wrap,
        )

    # 非收集模式：以 tx.out 作为成功判据（与 iphase 一致）；收集模式以 wrapper 成功为准
    if not collect_rays and not collect_obs and not tx_out.exists():
        return RayinvrResult(
            success=False,
            code="txout_missing",
            message="RAYINVR 完成但未生成 tx.out",
            working_dir=str(wd),
            elapsed_s=time.time() - t0,
            ran_forward=True,
            backend=chosen,
            wrapper=wrap,
        )

    return RayinvrResult(
        success=True,
        code="ok",
        message="RAYINVR 正演完成",
        working_dir=str(wd),
        elapsed_s=time.time() - t0,
        ran_forward=True,
        tx_out_path=str(tx_out) if tx_out.exists() else None,
        rays=rays,
        obs=obs,
        backend=chosen,
        wrapper=wrap,
    )


def run_rayinvr_collect(
    working_dir: str | Path,
    *,
    max_rays: int = 0,
    require_tx: bool = False,
    timeout_s: int = 300,
    inline: Optional[bool] = None,
) -> dict:
    """vedit / 交互式绘图便捷接口：子进程运行并返回 ``{rays, obs, working_dir, tx_out_path}``。

    不在 GUI/QThread 进程内调用 ``get_all_rays``，避免 ``librayinvr`` segfault；
    理论走时一律从磁盘 ``tx.out`` 读取。

    多炮时**先合跑**一次子进程；失败或有个别炮点无射线时再**分炮**重试。

    *inline* 为真时在调用线程内直接调 ``librayinvr``（vedit 工作线程推荐，省子进程启动）；
    默认 ``None`` 时在 Linux/WSL 自动 inline，Windows 仍走子进程隔离。
    """
    wd = Path(working_dir)
    if not wd.is_dir():
        raise FileNotFoundError(f"工作目录不存在: {wd}")

    use_inline = prefer_inline_ray_collect() if inline is None else bool(inline)

    from pyAOBS.modeling.rayinvr.ray_collect import (
        missing_shot_xs_for_rays,
        parse_trapar_shots,
    )

    shot_xzs = parse_trapar_shots(wd / "r.in")
    once_kw = dict(
        max_rays=max_rays,
        require_tx=require_tx,
        timeout_s=timeout_s,
        inline=use_inline,
    )
    if len(shot_xzs) <= 1:
        return _run_rayinvr_collect_once(wd, **once_kw)

    shot_xs = [float(x) for x, _ in shot_xzs]
    fallback_reason = ""
    try:
        combined = _run_rayinvr_collect_once(wd, **once_kw)
        missing = missing_shot_xs_for_rays(combined.get("rays") or [], shot_xs)
        if not missing:
            note = str(combined.get("collect_note") or "").strip()
            combined["collect_note"] = (
                f"多炮合跑{('；' + note) if note else ''}"
            )
            combined["backend"] = (
                "wrapper_inline_collect_combined"
                if use_inline
                else "wrapper_subprocess_collect_combined"
            )
            return combined
        miss_s = ", ".join(f"{x:.3f}" for x in missing)
        fallback_reason = f"合跑缺失炮点 {miss_s}"
    except RuntimeError as exc:
        fallback_reason = f"合跑失败: {exc}"

    result = run_rayinvr_collect_per_shot(
        wd,
        shot_xzs=shot_xzs,
        max_rays=max_rays,
        require_tx=require_tx,
        timeout_s=timeout_s,
        inline=use_inline,
    )
    note = str(result.get("collect_note") or "").strip()
    result["collect_note"] = (
        f"{fallback_reason}；{note}" if note else fallback_reason
    )
    return result


def _run_rayinvr_collect_once(
    working_dir: Path,
    *,
    max_rays: int = 0,
    require_tx: bool = False,
    timeout_s: int = 300,
    inline: bool = False,
) -> dict:
    wd = Path(working_dir)
    ok, miss, _spec = validate_rayinvr_inputs(wd, require_tx=require_tx)
    if not ok:
        raise FileNotFoundError(f"缺少输入文件: {', '.join(miss)}")

    tx_out = wd / "tx.out"
    if tx_out.is_file():
        try:
            tx_out.unlink()
        except OSError:
            pass

    t0 = time.time()
    if inline:
        run_ok, msg, rays, obs, timing = _wrapper_collect_workdir(
            wd, max_rays=max_rays
        )
        backend = "wrapper_inline_collect"
    else:
        run_ok, msg, rays, obs, timing = run_wrapper_in_subprocess_collect(
            wd, max_rays=max_rays, timeout_s=timeout_s
        )
        backend = "wrapper_subprocess_collect"
    if not _tx_out_usable(tx_out):
        err = msg or "RAYINVR 运行失败"
        if not run_ok:
            raise RuntimeError(err)
        hint = ""
        if not (rays or []):
            hint = "（tx.out 为空：长偏移 OBS 在 invr=1 时常见 ximax 不足）"
        raise RuntimeError(
            "RAYINVR 未生成有效 tx.out"
            + hint
            + (f"；{msg}" if msg else "")
        )

    if obs is None:
        obs = _obs_from_tfile(wd)

    return {
        "rays": rays or [],
        "obs": obs,
        "working_dir": str(wd),
        "tx_out_path": str(tx_out),
        "elapsed_s": time.time() - t0,
        "timing": dict(timing or {}),
        "backend": backend,
        "collect_note": msg or "",
    }


def run_rayinvr_collect_per_shot(
    working_dir: str | Path,
    *,
    shot_xzs: Sequence[tuple[float, float]],
    max_rays: int = 0,
    require_tx: bool = False,
    timeout_s: int = 300,
    inline: bool = False,
) -> dict:
    """多炮分次运行 RAYINVR 并合并 rays / tx.out（规避合跑 ``illegal reflection`` 整 job 中止）。"""
    wd = Path(working_dir)
    if not wd.is_dir():
        raise FileNotFoundError(f"工作目录不存在: {wd}")
    if len(shot_xzs) <= 1:
        return _run_rayinvr_collect_once(
            wd,
            max_rays=max_rays,
            require_tx=require_tx,
            timeout_s=timeout_s,
            inline=inline,
        )

    ok, miss, spec = validate_rayinvr_inputs(wd, require_tx=require_tx)
    if not ok:
        raise FileNotFoundError(f"缺少输入文件: {', '.join(miss)}")

    from pyAOBS.modeling.rayinvr.ray_collect import write_rin_trapar_shots
    from pyAOBS.modeling.rayinvr.tx_io import read_tx_file, write_tx_file, TxDataset

    rin_template = wd / "r.in"
    vin_src = spec.v_file
    tx_src = spec.t_file if spec.t_file.is_file() else None

    all_rays: list = []
    merged = TxDataset()
    notes = ["多炮分炮追踪"]
    timing = {"fortran_s": 0.0, "collect_s": 0.0, "job_total_s": 0.0}
    t0 = time.time()

    for x, z in shot_xzs:
        tag = f"{float(x):.3f}".replace(".", "p")
        sub = wd / f"_per_shot_{tag}"
        if sub.exists():
            shutil.rmtree(sub)
        sub.mkdir(parents=True, exist_ok=True)
        rin_one = sub / "r.in"
        write_rin_trapar_shots(rin_template, rin_one, [x], [z])
        prepare_rayinvr_workdir(
            sub,
            vin_src=vin_src,
            rin_src=rin_one,
            tx_src=tx_src,
            tx_shot_xs=[float(x)],
        )
        try:
            part = _run_rayinvr_collect_once(
                sub,
                max_rays=max_rays,
                require_tx=require_tx,
                timeout_s=timeout_s,
                inline=inline,
            )
        except RuntimeError as exc:
            notes.append(f"x={float(x):.3f}: {exc}")
            continue
        all_rays.extend(part.get("rays") or [])
        pt = part.get("timing") or {}
        timing["fortran_s"] = float(timing.get("fortran_s", 0.0)) + float(
            pt.get("fortran_s", 0.0)
        )
        timing["collect_s"] = float(timing.get("collect_s", 0.0)) + float(
            pt.get("collect_s", 0.0)
        )
        timing["job_total_s"] = float(timing.get("job_total_s", 0.0)) + float(
            pt.get("job_total_s", 0.0) or pt.get("subprocess_total_s", 0.0)
        )
        txp = part.get("tx_out_path")
        if txp and Path(txp).is_file():
            merged.shots.extend(read_tx_file(txp).shots)

    out_tx = wd / "tx.out"
    if merged.shots:
        write_tx_file(merged, out_tx)
    elif out_tx.is_file():
        try:
            out_tx.unlink()
        except OSError:
            pass

    if not all_rays:
        raise RuntimeError("分炮追踪仍无射线: " + "; ".join(notes))

    obs = _obs_from_tfile(wd)
    return {
        "rays": all_rays,
        "obs": obs,
        "working_dir": str(wd),
        "tx_out_path": str(out_tx) if out_tx.is_file() else None,
        "elapsed_s": time.time() - t0,
        "timing": timing,
        "backend": "wrapper_subprocess_collect_per_shot",
        "collect_note": "; ".join(notes),
    }


__all__ = [
    "Backend",
    "RayinvrInputSpec",
    "RayinvrResult",
    "find_rayinvr_executable",
    "hash_file",
    "parse_rin_input_files",
    "clean_rayinvr_workdir",
    "prepare_rayinvr_workdir",
    "prefer_inline_ray_collect",
    "run_rayinvr",
    "run_rayinvr_collect",
    "run_rayinvr_collect_per_shot",
    "run_wrapper_in_subprocess",
    "run_wrapper_in_subprocess_collect",
    "validate_rayinvr_inputs",
]
