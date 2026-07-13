"""
src_kernel_bridge.py - bridge for zplot src Fortran kernels.

This module compiles and loads selected source kernels from
`visualization/zplotpy/src` and exposes Python-callable methods.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Iterable, List, Optional

import numpy as np

_STAMP_NAME = ".numpy_build_version"
_KERNEL_MODULE_PREFIX = "_zplot_src_"
# 清理编译缓存时保留的非二进制文件
_PRESERVE_IN_BUILD_DIR = {_STAMP_NAME, "zplot.par"}


def _numpy_build_stamp() -> str:
    return str(np.__version__)


def _is_numpy_abi_mismatch(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "compiled using numpy 1.x" in msg
        or "numpy.core.multiarray failed to import" in msg
        or ("numpy 2" in msg and "cannot be run" in msg)
    )


def _purge_kernel_build_artifacts(build_dir: Path) -> None:
    """Remove stale f2py/meson artifacts after NumPy ABI changes."""
    if not build_dir.exists():
        return

    for name in list(sys.modules):
        if name.startswith(_KERNEL_MODULE_PREFIX):
            del sys.modules[name]

    for child in build_dir.iterdir():
        if child.name in _PRESERVE_IN_BUILD_DIR:
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def _filter_compile_args_for_backend(
    compile_extra_args: Iterable[str],
    backend: str,
) -> List[str]:
    """Drop distutils-only f2py flags when using the meson backend."""
    args = list(compile_extra_args)
    if backend != "meson":
        return args

    filtered: List[str] = []
    skip_next = False
    distutils_only = {"--f77flags", "--f90flags", "--fflags", "--opt", "--arch", "--link-lapack_opt"}
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in distutils_only:
            skip_next = True
            continue
        filtered.append(arg)
    return filtered


def _find_extension_module(build_dir: Path, module_name: str) -> Optional[Path]:
    for pattern in (f"{module_name}*.so", f"{module_name}*.pyd"):
        matches = sorted(build_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _ensure_kernel_build_dir_fresh(build_dir: Path) -> None:
    """Rebuild f2py extensions when the active NumPy version changes."""
    build_dir.mkdir(parents=True, exist_ok=True)
    stamp_path = build_dir / _STAMP_NAME
    current = _numpy_build_stamp()
    if stamp_path.exists() and stamp_path.read_text(encoding="utf-8").strip() == current:
        return

    _purge_kernel_build_artifacts(build_dir)
    stamp_path.write_text(current + "\n", encoding="utf-8")


class _F2pyKernelBridgeBase:
    """Shared loader/compiler for zplot Fortran f2py kernels."""

    _build_lock = threading.Lock()
    _dir_fresh_lock = threading.Lock()
    _dir_fresh_checked: set[str] = set()

    def __init__(
        self,
        module_name: str,
        source_file: Path,
        compile_extra_args: Optional[Iterable[str]] = None,
        compile_error_label: str = "src",
    ) -> None:
        self._module_name = module_name
        self._base_dir = Path(__file__).resolve().parent
        self._build_dir = self._base_dir / "_src_kernels"
        self._source_file = source_file
        self._compile_extra_args: List[str] = list(compile_extra_args or [])
        self._compile_error_label = compile_error_label
        self._module = None
        self._ensure_module_loaded()

    @classmethod
    def _ensure_build_dir_fresh_once(cls, build_dir: Path) -> None:
        key = str(build_dir)
        with cls._dir_fresh_lock:
            if key in cls._dir_fresh_checked:
                return
            _ensure_kernel_build_dir_fresh(build_dir)
            cls._dir_fresh_checked.add(key)

    def _ensure_module_loaded(self) -> None:
        if not self._source_file.exists():
            raise RuntimeError(f"未找到内核源码文件: {self._source_file}")

        self._ensure_build_dir_fresh_once(self._build_dir)
        build_dir_str = str(self._build_dir)
        if build_dir_str not in sys.path:
            sys.path.insert(0, build_dir_str)

        try:
            self._module = importlib.import_module(self._module_name)
            return
        except Exception as exc:
            if not _is_numpy_abi_mismatch(exc):
                pass
            else:
                _purge_kernel_build_artifacts(self._build_dir)
                (self._build_dir / _STAMP_NAME).write_text(
                    _numpy_build_stamp() + "\n",
                    encoding="utf-8",
                )

        with self._build_lock:
            self._ensure_build_dir_fresh_once(self._build_dir)
            try:
                self._module = importlib.import_module(self._module_name)
                return
            except Exception:
                pass

            self._compile_module()
            try:
                self._module = importlib.import_module(self._module_name)
            except Exception as exc:
                ext = _find_extension_module(self._build_dir, self._module_name)
                raise RuntimeError(
                    f"编译后仍无法导入 {self._compile_error_label} 内核 "
                    f"({self._module_name}): {exc}\n"
                    f"build_dir={self._build_dir}\n"
                    f"extension={ext}"
                ) from exc

    def _compile_module(self) -> None:
        backend = os.environ.get("PYAOBS_F2PY_BACKEND", "meson").strip().lower()
        if backend not in {"meson", "distutils"}:
            backend = "meson"
        compile_args = _filter_compile_args_for_backend(self._compile_extra_args, backend)
        cmd = [
            sys.executable,
            "-m",
            "numpy.f2py",
            "-c",
            "--backend",
            backend,
            *compile_args,
            "-m",
            self._module_name,
            str(self._source_file),
        ]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            cmd,
            cwd=str(self._build_dir),
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            msg = (
                f"编译 {self._compile_error_label} 内核失败，无法继续（已禁用回退）。\n"
                f"f2py backend: {backend}\n"
                f"命令: {' '.join(cmd)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
            raise RuntimeError(msg)

        ext = _find_extension_module(self._build_dir, self._module_name)
        if ext is None:
            msg = (
                f"编译 {self._compile_error_label} 内核未生成扩展文件 "
                f"({self._module_name}.*)。\n"
                f"f2py backend: {backend}\n"
                f"命令: {' '.join(cmd)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
            raise RuntimeError(msg)


class SrcBandpassKernelBridge(_F2pyKernelBridgeBase):
    """Bridge for Fortran `bndpas` kernel in `src/zplot/bandpass.f`."""

    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parent
        super().__init__(
            module_name="_zplot_src_bandpass",
            source_file=base_dir / "src" / "zplot" / "bandpass.f",
            compile_error_label="src bandpass",
        )

    def apply_bandpass(
        self,
        trace: np.ndarray,
        freqlo: float,
        freqhi: float,
        dt: float,
        npoles: int,
        izerop: int,
    ) -> np.ndarray:
        """Apply Fortran bndpas kernel on one trace."""
        if self._module is None:
            raise RuntimeError("src 内核模块未加载")
        if trace.ndim != 1:
            raise ValueError("bandpass 内核仅支持一维道数据")
        if dt <= 0:
            raise ValueError(f"无效采样间隔 dt={dt}")
        if npoles <= 0:
            raise ValueError(f"无效滤波器阶数 npoles={npoles}")

        work = np.ascontiguousarray(trace.astype(np.float32, copy=True))
        iflag = np.array(0, dtype=np.int32)

        try:
            out = self._module.bndpas(
                np.float32(freqlo),
                np.float32(freqhi),
                np.float32(dt),
                int(npoles),
                int(izerop),
                work,
                iflag,
            )
        except Exception as exc:
            raise RuntimeError(f"调用 src bndpas 内核失败: {exc}") from exc

        # f2py 可能返回 inout 参数，也可能仅原地修改
        if isinstance(out, tuple):
            for item in out:
                if isinstance(item, np.ndarray) and item.ndim == 1:
                    work = np.ascontiguousarray(item.astype(np.float32, copy=False))
                    break

        return work

    def apply_hilbert(self, trace: np.ndarray, mode: int) -> np.ndarray:
        """Apply Fortran `hilbert` in bandpass.f.

        Notes:
        - mode=1 returns phase
        - mode=2 returns envelope
        """
        if self._module is None:
            raise RuntimeError("src 内核模块未加载")
        if trace.ndim != 1:
            raise ValueError("hilbert 内核仅支持一维道数据")
        if mode not in (1, 2):
            raise ValueError(f"hilbert 内核仅支持 mode=1/2，收到 mode={mode}")

        work = np.ascontiguousarray(trace.astype(np.float32, copy=True))
        n = int(work.shape[0])
        try:
            out = self._module.hilbert(work, n, int(mode))
        except Exception as exc:
            raise RuntimeError(f"调用 src hilbert 内核失败: {exc}") from exc

        if isinstance(out, tuple):
            for item in out:
                if isinstance(item, np.ndarray) and item.ndim == 1:
                    work = np.ascontiguousarray(item.astype(np.float32, copy=False))
                    break
        return work


class SrcMiscKernelBridge(_F2pyKernelBridgeBase):
    """Bridge for selected standalone Fortran kernels."""

    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parent
        super().__init__(
            module_name="_zplot_src_vg2",
            source_file=base_dir / "src" / "kernels" / "vg2_kernel.f90",
            compile_error_label="src vg2",
        )

    def apply_vg2(self, trace: np.ndarray, dt: float, tvg: float, pvg: float) -> np.ndarray:
        """Apply Fortran `vg2` variable-gain kernel."""
        if self._module is None:
            raise RuntimeError("src misc 内核模块未加载")
        if trace.ndim != 1:
            raise ValueError("vg2 内核仅支持一维道数据")
        if dt <= 0:
            raise ValueError(f"无效采样间隔 dt={dt}")

        work = np.ascontiguousarray(trace.astype(np.float32, copy=True))
        n = int(work.shape[0])
        try:
            # 与其他内核保持一致：关键字传参，避免 f2py 位置参数解析差异。
            out = self._module.vg2_kernel(
                x=work,
                npts=int(n),
                dts=np.float32(dt),
                tvg=np.float32(tvg),
                pvg=np.float32(pvg),
            )
        except Exception as exc:
            raise RuntimeError(f"调用 src vg2 内核失败: {exc}") from exc

        if isinstance(out, tuple):
            for item in out:
                if isinstance(item, np.ndarray) and item.ndim == 1:
                    work = np.ascontiguousarray(item.astype(np.float32, copy=False))
                    break
        return work


class SrcPickKernelBridge(_F2pyKernelBridgeBase):
    """Bridge for standalone Fortran `pick_kernel`."""

    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parent
        super().__init__(
            module_name="_zplot_src_pick",
            source_file=base_dir / "src" / "kernels" / "pick_kernel.f90",
            compile_error_label="src pick",
        )

    def pick_energy_ratio(
        self,
        trace: np.ndarray,
        start_idx: int,
        end_idx: int,
        nwind: int,
        dt: float,
        min_energy_ratio: float,
    ) -> tuple[int, float, int]:
        """Call Fortran `pick_kernel`; returns (pick_idx, max_ratio, iflag)."""
        if self._module is None:
            raise RuntimeError("src pick 内核模块未加载")
        if trace.ndim != 1:
            raise ValueError("pick 内核仅支持一维道数据")
        if dt <= 0:
            raise ValueError(f"无效采样间隔 dt={dt}")
        if nwind < 1:
            raise ValueError(f"无效窗口长度 nwind={nwind}")

        work = np.ascontiguousarray(trace.astype(np.float32, copy=False))
        npts = int(work.shape[0])
        try:
            # 用关键字传参，避免 f2py 对位置参数顺序推断差异导致 npts 错位。
            out = self._module.pick_kernel(
                seis=work,
                npts=int(npts),
                start_idx=int(start_idx),
                end_idx=int(end_idx),
                nwind=int(nwind),
                dts=np.float32(dt),
                minenratio=np.float32(min_energy_ratio),
            )
        except Exception as exc:
            raise RuntimeError(f"调用 src pick 内核失败: {exc}") from exc

        if not isinstance(out, tuple) or len(out) < 3:
            raise RuntimeError("src pick 内核返回值异常")

        pick_idx = int(out[0])
        max_ratio = float(out[1])
        iflag = int(out[2])
        return pick_idx, max_ratio, iflag


class SrcCrossCorrelationKernelBridge(_F2pyKernelBridgeBase):
    """Bridge for standalone Fortran `crscor_kernel`."""

    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parent
        super().__init__(
            module_name="_zplot_src_crscor",
            source_file=base_dir / "src" / "kernels" / "crscor_kernel.f90",
            compile_error_label="src crscor",
        )

    def cross_correlate(
        self,
        pilot: np.ndarray,
        trace: np.ndarray,
        start_idx: int,
        window_length: int,
        search_range: int,
        hilbert_ratio: float = 1.0,
    ) -> tuple[int, float]:
        """Call Fortran `crscor_kernel`; returns (best_lag, max_corr)."""
        if self._module is None:
            raise RuntimeError("src crscor 内核模块未加载")
        if pilot.ndim != 1 or trace.ndim != 1:
            raise ValueError("crscor 内核仅支持一维数据")
        if window_length < 1:
            raise ValueError(f"无效窗口长度 window_length={window_length}")
        if search_range < 0:
            raise ValueError(f"无效搜索范围 search_range={search_range}")

        pilot_window = np.ascontiguousarray(
            pilot[:window_length].astype(np.float32, copy=False)
        )
        trace_work = np.ascontiguousarray(trace.astype(np.float32, copy=False))
        npts = int(trace_work.shape[0])
        ncrcor = int(pilot_window.shape[0])
        if ncrcor <= 0 or ncrcor > npts:
            return 0, 0.0

        try:
            # 用关键字传参，避免 f2py 对位置参数顺序推断差异导致 npts 错位。
            out = self._module.crscor_kernel(
                pilot=pilot_window,
                trace=trace_work,
                npts=int(npts),
                start_idx=int(start_idx),
                ncrcor=int(ncrcor),
                nlag=int(search_range),
                ratio=np.float32(hilbert_ratio),
            )
        except Exception as exc:
            raise RuntimeError(f"调用 src crscor 内核失败: {exc}") from exc

        if not isinstance(out, tuple) or len(out) < 2:
            raise RuntimeError("src crscor 内核返回值异常")
        return int(out[0]), float(out[1])


class SrcShadeKernelBridge(_F2pyKernelBridgeBase):
    """Bridge for standalone Fortran `build_shade_segments`."""

    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parent
        super().__init__(
            module_name="_zplot_src_shade",
            source_file=base_dir / "src" / "kernels" / "shade_kernel.f90",
            compile_error_label="src shade",
        )

    def build_segments(
        self,
        x_values: np.ndarray,
        t_values: np.ndarray,
        baseline_x: float,
        fill_positive: bool,
        row_step: int = 1,
    ) -> np.ndarray:
        """Return shade segments array with shape (nseg, 2, 2)."""
        if self._module is None:
            raise RuntimeError("src shade 内核模块未加载")
        if x_values.ndim != 1 or t_values.ndim != 1:
            raise ValueError("shade 内核仅支持一维数组")

        n = min(int(x_values.shape[0]), int(t_values.shape[0]))
        if n <= 0:
            return np.empty((0, 2, 2), dtype=np.float64)

        x_work = np.ascontiguousarray(x_values[:n].astype(np.float32, copy=False))
        t_work = np.ascontiguousarray(t_values[:n].astype(np.float32, copy=False))
        try:
            # 用关键字传参，避免 f2py 对位置参数顺序推断差异导致的 npts 错位。
            out = self._module.build_shade_segments(
                x_values=x_work,
                t_values=t_work,
                npts=int(n),
                baseline_x=np.float32(baseline_x),
                fill_positive=int(1 if fill_positive else -1),
                row_step=int(max(1, row_step)),
            )
        except Exception as exc:
            raise RuntimeError(f"调用 src shade 内核失败: {exc}") from exc

        if not isinstance(out, tuple) or len(out) < 5:
            raise RuntimeError("src shade 内核返回值异常")

        sx = np.asarray(out[0], dtype=np.float64)
        st = np.asarray(out[1], dtype=np.float64)
        ex = np.asarray(out[2], dtype=np.float64)
        et = np.asarray(out[3], dtype=np.float64)
        nseg = int(out[4])
        if nseg <= 0:
            return np.empty((0, 2, 2), dtype=np.float64)

        nseg = min(nseg, sx.shape[0], st.shape[0], ex.shape[0], et.shape[0])
        segs = np.empty((nseg, 2, 2), dtype=np.float64)
        segs[:, 0, 0] = sx[:nseg]
        segs[:, 0, 1] = st[:nseg]
        segs[:, 1, 0] = ex[:nseg]
        segs[:, 1, 1] = et[:nseg]
        return segs

