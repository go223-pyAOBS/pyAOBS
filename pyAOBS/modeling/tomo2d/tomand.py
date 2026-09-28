#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tomo2d 命令的 TomoAnd 封装。

完整必选/可选规则见 ``help_docs.TomoHelp.python_wrapper_help()``；
GUI 逐字段说明见 ``param_hints.TOMO2D_GUI_HINTS``。
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .help_docs import TomoHelp


def _resolve_under_proc_cwd(tomo: "TomoAnd", path: Any) -> Path:
    cwd = tomo.proc_cwd
    if not cwd:
        raise FileNotFoundError("未设置 proc_cwd（工作目录）")
    s = str(path).strip()
    if not s:
        raise FileNotFoundError("路径为空")
    p = Path(s).expanduser()
    full = p.resolve() if p.is_absolute() else (Path(cwd) / p).resolve()
    return full


def _assert_readable_under_proc_cwd(tomo: "TomoAnd", role: str, path: Any) -> None:
    """
    gen_smesh 等对 z_file 调用 countLines/open；路径相对于进程 cwd（GUI 下即 work_dir）。
    在启动子进程前检查，避免 C 端仅打印 ``countLines::can't open``。
    """
    if not tomo.proc_cwd or path is None:
        return
    s = str(path).strip()
    if not s:
        return
    try:
        full = _resolve_under_proc_cwd(tomo, s)
    except (OSError, FileNotFoundError) as e:
        raise FileNotFoundError(f"{role}: 路径无效 {s!r} ({e})") from e
    if not full.is_file():
        raise FileNotFoundError(
            f"{role}: 找不到或不是可读文件\n"
            f"  参数: {s!r}\n"
            f"  proc_cwd（子进程工作目录）: {tomo.proc_cwd!r}\n"
            f"  解析为: {full}\n"
            f"  提示: 请确认文件已存在；路径相对界面「工作目录」。"
        )


def _stage_zelt_token_under_cwd(tomo: "TomoAnd", role: str, form_path: Any) -> str:
    """
    ``-C v.in`` 路径段不能含 ``/``。表单可含子目录；运行前把**输入**文件
    复制到 ``proc_cwd`` 根目录（若尚不在该处），返回 basename。
    """
    if form_path is None:
        return ""
    s = str(form_path).strip()
    if not s:
        return ""
    _assert_readable_under_proc_cwd(tomo, role, s)
    src = _resolve_under_proc_cwd(tomo, s)
    base = os.path.basename(str(src).replace("\\", "/"))
    cwd = Path(str(tomo.proc_cwd))
    dest = (cwd / base).resolve()
    if dest != src:
        try:
            shutil.copy2(src, dest)
        except OSError as e:
            raise FileNotFoundError(
                f"{role}: 无法将 {src} 复制到工作目录根以供 -C/-F 使用 → {dest}\n{e}"
            ) from e
    return base


def _prepare_zelt_output_under_cwd(
    tomo: "TomoAnd", role: str, form_path: Any
) -> tuple[str, Path | None]:
    """
    ``gen_smesh -F`` 的 refl_file 是**输出**（ofstream），不必预先存在。
    命令行只能带 basename；返回 (basename, 若表单含目录则运行后应挪到的绝对路径)。
    """
    if form_path is None:
        return "", None
    s = str(form_path).strip()
    if not s:
        return "", None
    if not tomo.proc_cwd:
        raise FileNotFoundError(f"{role}: 未设置 proc_cwd（工作目录）")
    cwd = Path(str(tomo.proc_cwd))
    base = os.path.basename(s.replace("\\", "/"))
    if not base:
        raise ValueError(f"{role}: 无效输出路径 {s!r}")
    desired = Path(s).expanduser()
    try:
        desired = (
            desired.resolve()
            if desired.is_absolute()
            else (cwd / desired).resolve()
        )
    except OSError as e:
        raise FileNotFoundError(f"{role}: 无法解析输出路径 {s!r} ({e})") from e
    try:
        desired.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise FileNotFoundError(
            f"{role}: 无法创建输出父目录 {desired.parent}\n{e}"
        ) from e
    staged = (cwd / base).resolve()
    if staged == desired:
        return base, None
    return base, desired


def _apply_zelt_output_moves(moves: list[tuple[Path, Path]]) -> None:
    for src, dst in moves:
        if not src.is_file():
            continue
        if src.resolve() == dst.resolve():
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        except OSError as e:
            raise FileNotFoundError(
                f"无法将 -F/-G/-d 输出从 {src} 移到 {dst}\n{e}"
            ) from e


def _verify_gen_mesh_family_inputs(tomo: "TomoAnd", kw: Dict[str, Any], context: str) -> None:
    go = kw.get("grid_opt")
    if go == "variable":
        _assert_readable_under_proc_cwd(tomo, f"{context} x_file (-X)", kw.get("x_file"))
        _assert_readable_under_proc_cwd(tomo, f"{context} z_file (-Z)", kw.get("z_file"))
        if kw.get("topo_file"):
            _assert_readable_under_proc_cwd(tomo, f"{context} topo_file (-T)", kw.get("topo_file"))
    elif go == "zelt":
        _assert_readable_under_proc_cwd(tomo, f"{context} z_file (-Z)", kw.get("z_file"))
    vo = kw.get("vel_opt")
    if vo == "zelt" and kw.get("v_in"):
        kw["v_in"] = _stage_zelt_token_under_cwd(
            tomo, f"{context} v.in (-C)", kw.get("v_in")
        )
    # gen_smesh -F 为反射面**输出**文件，不是输入
    if context == "gen_smesh" and kw.get("refl_file"):
        base, final = _prepare_zelt_output_under_cwd(
            tomo, f"{context} refl_file (-F 输出)", kw.get("refl_file")
        )
        kw["refl_file"] = base
        if final is not None:
            cwd = Path(str(tomo.proc_cwd))
            kw.setdefault("_zelt_output_moves", []).append((cwd / base, final))
    if context == "gen_smesh" and kw.get("seafloor_out"):
        base, final = _prepare_zelt_output_under_cwd(
            tomo, f"{context} seafloor_out (-G 输出)", kw.get("seafloor_out")
        )
        kw["seafloor_out"] = base
        if final is not None:
            cwd = Path(str(tomo.proc_cwd))
            kw.setdefault("_zelt_output_moves", []).append((cwd / base, final))



def _zelt_sscanf_token_path(path: Any) -> str:
    """
    gen_smesh / gen_damp / gen_vcorr / gen_dcorr 中 ``-C<vpath>/<ilayer>``、``-F<layer>/<rpath>``
    用 ``sscanf(..., "%[^/]/...", ...)`` 解析：``vpath`` / ``rpath`` 内不能含 ``'/'``，
    否则会在第一个 ``/`` 处被截断（见 gen_smesh.cc）。
    可执行文件在 ``work_dir`` 下启动时，应只传位于该目录下的**文件名**（无目录前缀）。
    """
    if path is None:
        return ""
    s = str(path).strip()
    if not s:
        return s
    return os.path.basename(s.replace("\\", "/"))


def _tomo_glued_path_argv(short_flag: str, path: Any) -> List[str]:
    """
    ``tt_forward`` 路径类选项：一律粘成 ``-M<path>`` 单个 argv。

    C 端（``tt_forward.cc`` / ``tt_inverse.cc``）只读 ``&argv[i][2]``，
    **不**支持 ``-M`` 与路径拆成两个参数；若拆开则路径为空，表现为未带目录。
    仅当路径以 ``-`` 开头时拆开，避免被当成新选项。
    """
    s = str(path).strip() if path is not None else ""
    if not s:
        return [f"-{short_flag}"]
    if s.startswith("-"):
        return [f"-{short_flag}", s]
    return [f"-{short_flag}{s}"]


def validate_tomo2d_geom_data_format(path: Path) -> tuple[int, int]:
    """
    校验 ``SyntheticTraveltimeGenerator2d::read_file``（syngen.cc）所读的 geom / ttimes 文本格式。

    结构：首行 ``nsrc``；每个炮点一行 ``s x y nrcv``，再紧跟 ``nrcv`` 行 ``r x y code t u``。
    若 ``nrcv`` 与实际 ``r`` 行数不符，或 ``nsrc`` 之后仍有多余行，C 端可能未定义行为乃至段错误。
    """
    raw = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in raw.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    if not lines:
        raise ValueError(f"geom 文件为空: {path}")
    head = lines[0].split()
    if not head:
        raise ValueError(f"geom 第 1 行为空: {path}")
    try:
        nsrc = int(float(head[0]))
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"geom 第 1 行应以炮点数 nsrc（数字）开头（syngen.cc）；当前: {lines[0]!r}"
        ) from e
    if nsrc <= 0:
        raise ValueError(f"geom nsrc 无效: {nsrc}")
    idx = 1
    total_r = 0
    for ishot in range(nsrc):
        if idx >= len(lines):
            raise ValueError(f"geom 在解析第 {ishot + 1}/{nsrc} 个炮点时遇到 EOF（缺少 s 行）")
        sp = lines[idx].split()
        line_no = idx + 1
        idx += 1
        if len(sp) < 4 or sp[0] != "s":
            raise ValueError(
                f"geom 第 {line_no} 行应为 s x y nrcv（第 {ishot + 1} 个炮点），当前: {lines[line_no - 1]!r}"
            )
        try:
            nrcv = int(float(sp[3]))
        except (TypeError, ValueError) as e:
            raise ValueError(f"geom 第 {line_no} 行 nrcv 无效: {sp!r}") from e
        if nrcv < 0 or nrcv > 50_000_000:
            raise ValueError(
                f"geom 第 {line_no} 行 nrcv={nrcv} 异常（若极大，多为上一块 r 行数量与声明不符导致读错位）"
            )
        for j in range(nrcv):
            if idx >= len(lines):
                raise ValueError(
                    f"geom 第 {ishot + 1} 个炮点声明 nrcv={nrcv}，但只有 {j} 条 r 行（文件提前结束）"
                )
            rp = lines[idx].split()
            rline_no = idx + 1
            idx += 1
            if len(rp) < 6 or rp[0] != "r":
                raise ValueError(
                    f"geom 第 {rline_no} 行应为 r x y code t u（炮 {ishot + 1} 的第 {j + 1} 条接收），"
                    f"当前: {lines[rline_no - 1]!r}"
                )
        total_r += nrcv
    if idx < len(lines):
        tail = "\n".join(lines[idx : min(idx + 5, len(lines))])
        raise ValueError(
            f"geom 在声明的 {nsrc} 个炮点之后仍有内容（第 {idx + 1} 行起），"
            f"syngen 读完后不应再有 ``s``/``r`` 行，否则易导致段错误。尾部示例:\n{tail}"
        )
    return nsrc, total_r


def _peel_tt_forward_stdout_ttime(kwargs: Dict[str, Any]) -> Optional[str]:
    """
    ``out_opts['ttime']`` 表示把 stdout（``operator<<``，与 tt_inverse -G 同构）落到该文件。

    原生 ``-T`` 走 ``printSynTime``：每炮 ``>`` + ``x t``，反演读首行会报 ``invalid nsrc``。
    若确实要原生 -T，用 ``out_opts['ttime_plot']``。
    """
    out_opts = dict(kwargs.get("out_opts") or {})
    ttime_data = out_opts.pop("ttime", None)
    ttime_plot = out_opts.pop("ttime_plot", None)
    if ttime_plot:
        out_opts["ttime"] = ttime_plot
    if ttime_data is not None or ttime_plot is not None:
        kwargs["out_opts"] = out_opts
    if ttime_data is None:
        return None
    s = str(ttime_data).strip()
    return s or None


class TomoCommandError(RuntimeError):
    """tomo2d 命令执行异常。"""

    def __init__(self, command: List[str], returncode: int, stdout: str, stderr: str):
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        message = (
            f"命令执行失败 (exit={returncode}): {' '.join(command)}\n"
            f"stderr: {stderr.strip() or '<empty>'}"
        )
        super().__init__(message)

class TomoAnd:
    """
    tomo2d 各可执行文件的 Python 封装（拼参、校验、subprocess 调用）。

    参数【必选】/【可选】/【条件必选】/【成组可选】与 GUI 说明一致，完整摘要见
    ``help_docs.TomoHelp.python_wrapper_help()``；逐字段中文说明见 ``param_hints.TOMO2D_GUI_HINTS``。
    """
    _HELP_MAP = {
        "edit_smesh_HHB": TomoHelp.edit_smesh_help,
        "gen_smesh": TomoHelp.gen_smesh_help,
        "gen_damp": TomoHelp.gen_damp_help,
        "gen_vcorr": TomoHelp.gen_vcorr_help,
        "gen_dcorr": TomoHelp.gen_dcorr_help,
        "tt_forward": TomoHelp.tt_forward_help,
        "tt_inverse": TomoHelp.tt_inverse_help,
        "stat_smesh": TomoHelp.stat_smesh_help,
    }

    def __init__(self, bin_path: Optional[str] = None, *, capture_subprocess_output: bool = True):
        """
        初始化TomoAnd类
        
        参数:
            bin_path: 可执行文件所在目录。若为None，将按如下顺序解析:
                1) 环境变量 PYAOBS_TOMO2D_BIN
                2) 环境变量 TOMO2D_BIN
                3) 当前模块所在目录
            capture_subprocess_output: 为 True（默认）时用管道捕获 stdout/stderr。
                若同时设置 ``stream_output_line``，则按行回调（GUI 实时日志）；否则进程结束后才能在
                ``CompletedProcess`` / ``TomoCommandError`` 里查看。设为 False 时子进程继承当前终端。
        """
        self.bin_path = self._resolve_bin_path(bin_path)
        #: 若设置，subprocess 将在此目录下启动（相对路径输入/输出均相对该目录）。GUI 多线程下可避免依赖进程全局 chdir。
        self.proc_cwd: Optional[str] = None
        self.capture_subprocess_output = capture_subprocess_output
        #: 子进程额外环境变量（覆盖 ``os.environ`` 中同名项）；用于 OMP / TOMO2D_* 并行开关。
        self.run_env: Optional[Dict[str, str]] = None
        #: 若设置且 ``capture_subprocess_output``：按行回调 ``(stream, line)``，
        #: ``stream`` 为 ``"stdout"`` / ``"stderr"``（供 GUI 准实时刷日志）。
        self.stream_output_line: Optional[Callable[[str, str], None]] = None

    def _resolve_bin_path(self, bin_path: Optional[str]) -> str:
        if bin_path and str(bin_path).strip():
            return self._absolutize_bin_dir(str(bin_path).strip())
        env_path = os.getenv("PYAOBS_TOMO2D_BIN") or os.getenv("TOMO2D_BIN")
        if env_path:
            return self._absolutize_bin_dir(env_path.strip())
        return os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def _absolutize_bin_dir(path: str) -> str:
        """将 bin 目录转为绝对路径（子进程 cwd=work_dir 时相对路径会失效）。"""
        p = Path(path).expanduser()
        if p.is_absolute():
            try:
                return str(p.resolve())
            except OSError:
                return str(p)
        # 相对：优先 cwd，再试包目录（modeling/tomo2d）
        pkg = Path(__file__).resolve().parent
        for base in (Path.cwd(), pkg, pkg.parent.parent.parent):
            cand = base / p
            try:
                if cand.is_dir() or any(
                    (cand / name).is_file() or (cand / f"{name}.exe").is_file()
                    for name in ("gen_smesh", "tt_forward", "tt_inverse")
                ):
                    return str(cand.resolve())
            except OSError:
                continue
        try:
            return str((Path.cwd() / p).resolve())
        except OSError:
            return str(Path.cwd() / p)

    def _resolve_executable(self, exe_name: str) -> str:
        candidates = []
        if self.bin_path:
            base = os.path.join(self.bin_path, exe_name)
            candidates.extend([base, f"{base}.exe", f"{base}.bat", f"{base}.cmd"])

        which_target = shutil.which(exe_name)
        if which_target:
            candidates.append(which_target)

        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                # 必须绝对路径：subprocess 的 cwd 常为 work_dir
                return os.path.abspath(candidate)

        raise FileNotFoundError(
            f"未找到可执行文件 '{exe_name}'。"
            f"请检查 bin_path='{self.bin_path}' 或将其加入 PATH。"
        )

    def _print_help(self, exe_name: str) -> Optional[str]:
        helper = self._HELP_MAP.get(exe_name)
        if helper is None:
            return None
        help_text = helper()
        print(help_text)
        return help_text

    @staticmethod
    def _require_keys(params: Dict, keys: List[str], context: str) -> None:
        missing = [k for k in keys if k not in params or params[k] is None]
        if missing:
            raise ValueError(f"{context} 缺少必需参数: {', '.join(missing)}")

    @staticmethod
    def _vel_grid_zelt_consistency(vel_opt: str, grid_opt: str, context: str) -> None:
        """
        gen_smesh / gen_damp / gen_vcorr 源码中 readZelt（-C v.in）会同时计入速度与网格选项，
        故 vel_opt='zelt' 与 grid_opt='zelt' 必须同时成立，且不能与 uniform/variable 网格混用。
        """
        if vel_opt == "zelt" and grid_opt != "zelt":
            raise ValueError(
                f"{context}: vel_opt='zelt' 时必须 grid_opt='zelt'（-C 与 -E/-Z 配套，见 gen_smesh.cc）"
            )
        if grid_opt == "zelt" and vel_opt != "zelt":
            raise ValueError(f"{context}: grid_opt='zelt' 时必须 vel_opt='zelt'")

    def _write_stdout_to_file(self, result: Any, out_file: Optional[str]) -> None:
        """可执行文件将主结果写在 stdout 时，可选落盘。"""
        if not out_file or result is None:
            return
        stdout = getattr(result, "stdout", None)
        if stdout is None:
            return
        path = Path(out_file)
        if not path.is_absolute():
            base = self.proc_cwd or os.getcwd()
            path = Path(base) / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(stdout, encoding="utf-8", newline="\n")

    @staticmethod
    def _tt_inverse_sv_sd_arg(val: Union[str, int, float]) -> str:
        """-SV / -SD 后为单值或 min/max/dw 三值（斜杠分隔），与 tt_inverse.cc sscanf 一致。"""
        if isinstance(val, str):
            return val.strip()
        return str(val)

    def _tt_inverse_append_gravity(self, args: List[str], g: Dict[str, Any]) -> None:
        """联合重力 -Z*（tt_inverse.cc case 'Z'）。"""
        if not g.get("grav_file"):
            return
        args.append(f"-ZG{g['grav_file']}")
        self._require_keys(g, ["grid_spec", "refrange"], "tt_inverse(gravity_opts)")
        args.append(f"-ZX{g['grid_spec']}")
        args.append(f"-ZR{g['refrange']}")
        n_domain = 0
        cont = g.get("continent")
        if cont:
            path, iconv = cont
            args.append(f"-ZC{path}/{int(iconv)}")
            n_domain += 1
        ou = g.get("ocean_upper")
        if ou:
            up, lo, iconv = ou
            args.append(f"-ZU{up}/{lo}/{int(iconv)}")
            n_domain += 1
        ol = g.get("ocean_lower")
        if ol:
            up, iconv = ol
            args.append(f"-ZL{up}/{int(iconv)}")
            n_domain += 1
        sed = g.get("sediment")
        if sed:
            up, lo, iconv = sed
            args.append(f"-ZS{up}/{lo}/{int(iconv)}")
            n_domain += 1
        if n_domain == 0:
            raise ValueError(
                "tt_inverse(gravity_opts): 至少指定 continent / ocean_upper / ocean_lower / sediment 之一"
            )
        deriv = g.get("deriv")
        if deriv:
            args.append(f"-ZD{deriv}")
        if g.get("weight_grav") is not None:
            args.append(f"-ZW{g['weight_grav']}")
        if g.get("z0") is not None:
            args.append(f"-ZZ{g['z0']}")
        gd = g.get("grav_dws")
        if gd:
            args.append(f"-ZK{gd}")
        co = g.get("cutoff")
        if co:
            args.append(f"-ZT{co}")

    def _build_grid_args(self, kwargs: Dict, context: str) -> List[str]:
        """
        构建通用网格参数:
        - uniform: -N<nx>/<nz> -D<xmax>/<zmax>
        - variable: -X<x_file> -Z<z_file> [-T<topo_file>]
        - zelt: -E<dx> -Z<z_file>
        """
        args: List[str] = []
        grid_opt = kwargs.get("grid_opt")

        if grid_opt == "uniform":
            self._require_keys(kwargs, ["nx", "nz", "xmax", "zmax"], f"{context}(grid_opt='uniform')")
            args.extend([f"-N{kwargs['nx']}/{kwargs['nz']}", f"-D{kwargs['xmax']}/{kwargs['zmax']}"])
        elif grid_opt == "variable":
            self._require_keys(kwargs, ["x_file", "z_file"], f"{context}(grid_opt='variable')")
            args.extend([f"-X{kwargs['x_file']}", f"-Z{kwargs['z_file']}"])
            if "topo_file" in kwargs and kwargs["topo_file"] is not None:
                args.append(f"-T{kwargs['topo_file']}")
        elif grid_opt == "zelt":
            self._require_keys(kwargs, ["dx", "z_file"], f"{context}(grid_opt='zelt')")
            args.extend([f"-E{kwargs['dx']}", f"-Z{kwargs['z_file']}"])
        else:
            raise ValueError(f"{context} 不支持的 grid_opt: {grid_opt}")

        return args

    def _compose_command(self, exe_name: str, args: Optional[List[str]] = None) -> List[str]:
        """与 subprocess 实际使用的一致：[解析后的可执行文件路径] + 参数。"""
        executable = self._resolve_executable(exe_name)
        return [executable] + [str(arg) for arg in (args or [])]

    def _run_cmd(self, exe_name: str, args: Optional[List[str]] = None, check_only: bool = False):
        """
        运行shell命令的通用函数
        
        参数:
            exe_name: 可执行文件名（例如 gen_smesh）
            args: 参数列表
            check_only: True时仅打印并返回帮助文本
        返回:
            - check_only=True: 帮助文本字符串或None
            - check_only=False: subprocess.CompletedProcess对象
        """
        if check_only:
            return self._print_help(exe_name)

        cmd = self._compose_command(exe_name, args)
        cwd = self.proc_cwd or None
        env = None
        if self.run_env:
            env = dict(os.environ)
            for k, v in self.run_env.items():
                if v is None:
                    env.pop(str(k), None)
                else:
                    env[str(k)] = str(v)

        if (
            self.capture_subprocess_output
            and self.stream_output_line is not None
        ):
            return self._run_cmd_streaming(cmd, cwd=cwd, env=env)

        run_kw: Dict[str, Any] = {
            "shell": False,
            "check": True,
            "text": True,
        }
        if self.capture_subprocess_output:
            run_kw["stdout"] = subprocess.PIPE
            run_kw["stderr"] = subprocess.PIPE
        else:
            run_kw["stdout"] = None
            run_kw["stderr"] = None
        if cwd:
            run_kw["cwd"] = cwd
        if env is not None:
            run_kw["env"] = env
        try:
            return subprocess.run(cmd, **run_kw)
        except subprocess.CalledProcessError as e:
            raise TomoCommandError(
                command=cmd,
                returncode=e.returncode,
                stdout=e.stdout or "",
                stderr=e.stderr or "",
            ) from e

    def _run_cmd_streaming(
        self,
        cmd: List[str],
        *,
        cwd: Optional[str],
        env: Optional[Dict[str, str]],
    ) -> subprocess.CompletedProcess:
        """Popen + 双线程按行泵出 stdout/stderr，供 GUI 实时显示。"""
        import threading

        popen_kw: Dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
        }
        if cwd:
            popen_kw["cwd"] = cwd
        if env is not None:
            popen_kw["env"] = env
        # 多线程 Qt 下 fork 不安全；独立会话也避免子进程 SIGSEGV 看起来像 GUI 崩了
        if os.name != "nt":
            popen_kw["start_new_session"] = True

        try:
            proc = subprocess.Popen(cmd, **popen_kw)
        except OSError as e:
            raise TomoCommandError(
                command=cmd, returncode=-1, stdout="", stderr=str(e)
            ) from e

        out_parts: List[str] = []
        err_parts: List[str] = []
        cb = self.stream_output_line

        def _pump(stream, bucket: List[str], name: str) -> None:
            try:
                while True:
                    line = stream.readline()
                    if line == "":
                        break
                    bucket.append(line)
                    if cb is not None:
                        try:
                            cb(name, line.rstrip("\r\n"))
                        except Exception:
                            pass
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

        t_out = threading.Thread(
            target=_pump, args=(proc.stdout, out_parts, "stdout"), daemon=True
        )
        t_err = threading.Thread(
            target=_pump, args=(proc.stderr, err_parts, "stderr"), daemon=True
        )
        t_out.start()
        t_err.start()
        rc = proc.wait()
        t_out.join(timeout=60)
        t_err.join(timeout=60)
        stdout = "".join(out_parts)
        stderr = "".join(err_parts)
        if rc != 0:
            raise TomoCommandError(
                command=cmd, returncode=rc, stdout=stdout, stderr=stderr
            )
        return subprocess.CompletedProcess(cmd, rc, stdout, stderr)
            
    def _build_edit_smesh_program_args(
        self, smesh_file: Any, cmd_type: Any, kwargs: Dict[str, Any]
    ) -> Optional[List[str]]:
        if smesh_file is None or cmd_type is None:
            return None
        args = [str(smesh_file)]

        if cmd_type == "a":
            cmd_token = "a"
        elif cmd_type == "p":
            self._require_keys(kwargs, ["paste_file"], "edit_smesh(cmd_type='p')")
            cmd_token = f"p{kwargs['paste_file']}"
        elif cmd_type == "P":
            self._require_keys(kwargs, ["prof_file"], "edit_smesh(cmd_type='P')")
            cmd_token = f"P{kwargs['prof_file']}"
        elif cmd_type == "B":
            self._require_keys(kwargs, ["remove_bg_file"], "edit_smesh(cmd_type='B')")
            cmd_token = f"B{kwargs['remove_bg_file']}"
        elif cmd_type == "s":
            self._require_keys(kwargs, ["h_len", "v_len"], "edit_smesh(cmd_type='s')")
            cmd_token = f"s{kwargs['h_len']}/{kwargs['v_len']}"
        elif cmd_type == "rm":
            self._require_keys(kwargs, ["mx", "mz"], "edit_smesh(cmd_type='rm')")
            cmd_token = f"r{kwargs['mx']}/{kwargs['mz']}"
        elif cmd_type == "c":
            self._require_keys(kwargs, ["amp", "h_len", "v_len"], "edit_smesh(cmd_type='c')")
            cmd_token = f"c{kwargs['amp']}/{kwargs['h_len']}/{kwargs['v_len']}"
        elif cmd_type == "d":
            self._require_keys(kwargs, ["amp", "xmin", "xmax", "zmin", "zmax"], "edit_smesh(cmd_type='d')")
            cmd_token = f"d{kwargs['amp']}/{kwargs['xmin']}/{kwargs['xmax']}/{kwargs['zmin']}/{kwargs['zmax']}"
        elif cmd_type == "g":
            self._require_keys(kwargs, ["amp", "x0", "z0", "Lh", "Lv"], "edit_smesh(cmd_type='g')")
            cmd_token = f"g{kwargs['amp']}/{kwargs['x0']}/{kwargs['z0']}/{kwargs['Lh']}/{kwargs['Lv']}"
        elif cmd_type == "l":
            cmd_token = "l"
        elif cmd_type == "R":
            self._require_keys(kwargs, ["seed", "amp", "nrand"], "edit_smesh(cmd_type='R')")
            cmd_token = f"R{kwargs['seed']}/{kwargs['amp']}/{kwargs['nrand']}"
        elif cmd_type == "S":
            self._require_keys(
                kwargs,
                ["seed", "amp", "xmin", "xmax", "dx", "zmin", "zmax", "dz"],
                "edit_smesh(cmd_type='S')",
            )
            cmd_token = (
                f"S{kwargs['seed']}/{kwargs['amp']}/{kwargs['xmin']}/{kwargs['xmax']}/"
                f"{kwargs['dx']}/{kwargs['zmin']}/{kwargs['zmax']}/{kwargs['dz']}"
            )
        elif cmd_type == "G":
            self._require_keys(
                kwargs,
                ["seed", "amp", "N", "xmin", "xmax", "zmin", "zmax"],
                "edit_smesh(cmd_type='G')",
            )
            cmd_token = (
                f"G{kwargs['seed']}/{kwargs['amp']}/{kwargs['N']}/{kwargs['xmin']}/"
                f"{kwargs['xmax']}/{kwargs['zmin']}/{kwargs['zmax']}"
            )
        elif cmd_type == "m":
            self._require_keys(kwargs, ["vel", "moho_file"], "edit_smesh(cmd_type='m')")
            cmd_token = f"m{kwargs['vel']}/{kwargs['moho_file']}"
        elif cmd_type == "b":
            self._require_keys(kwargs, ["k", "base_file"], "edit_smesh(cmd_type='b')")
            cmd_token = f"b{kwargs['k']}/{kwargs['base_file']}"
        else:
            raise ValueError(f"edit_smesh 不支持的 cmd_type: {cmd_type}")

        args.append("-C" + cmd_token)

        if "corr_file" in kwargs and kwargs["corr_file"] is not None:
            args.append(f"-L{kwargs['corr_file']}")
        if "upper_bound" in kwargs and kwargs["upper_bound"] is not None:
            args.append(f"-U{kwargs['upper_bound']}")

        return args

    def resolve_cmdline_edit_smesh(self, smesh_file=None, cmd_type=None, **kwargs) -> Optional[List[str]]:
        kw = dict(kwargs)
        prog = self._build_edit_smesh_program_args(smesh_file, cmd_type, kw)
        if prog is None:
            return None
        return self._compose_command("edit_smesh_HHB", prog)

    def edit_smesh(self, smesh_file=None, cmd_type=None, **kwargs):
        """
        编辑慢度网格文件（可执行名 edit_smesh_HHB，对应 src/edit_smesh_HHB.cc）。

        【必选】smesh_file, cmd_type。缺一则仅打印可执行文件帮助。
        【条件必选】由 cmd_type 决定附加字段；完整 -C 语义、-U/-L、kstart、-Cr 早退等见 ``TomoHelp.edit_smesh_help()``。
        【可选】corr_file（-L）, upper_bound（-U）。
        说明：cmd_type='B' 生成 -CB（removeBG，HHB）；cmd_type='b' 生成 -Cb，随包 HHB 源码无 case 'b'。
        """
        kw = dict(kwargs)
        prog = self._build_edit_smesh_program_args(smesh_file, cmd_type, kw)
        if prog is None:
            return self._run_cmd("edit_smesh_HHB", check_only=True)
        return self._run_cmd("edit_smesh_HHB", args=prog)

    def _build_gen_smesh_program_args(self, kwargs: Dict[str, Any]) -> Optional[List[str]]:
        """构造 gen_smesh 的参数列表（不含 exe）。缺 vel_opt/grid_opt 时返回 None。"""
        if not kwargs.get("vel_opt") or not kwargs.get("grid_opt"):
            return None
        vel_opt = kwargs.get("vel_opt")
        grid_opt = kwargs.get("grid_opt")
        self._vel_grid_zelt_consistency(vel_opt, grid_opt, "gen_smesh")
        args: List[str] = []
        if vel_opt == "uniform":
            self._require_keys(kwargs, ["v0", "gradient"], "gen_smesh(vel_opt='uniform')")
            args.extend([f"-A{kwargs['v0']}", f"-B{kwargs['gradient']}"])
        elif vel_opt == "zelt":
            self._require_keys(kwargs, ["v_in", "ilayer"], "gen_smesh(vel_opt='zelt')")
            args.append(f"-C{_zelt_sscanf_token_path(kwargs['v_in'])}/{kwargs['ilayer']}")
            if "refl_layer" in kwargs or "refl_file" in kwargs:
                self._require_keys(kwargs, ["refl_layer", "refl_file"], "gen_smesh(refl options)")
                args.append(
                    f"-F{kwargs['refl_layer']}/{_zelt_sscanf_token_path(kwargs['refl_file'])}"
                )
        else:
            raise ValueError(f"gen_smesh 不支持的 vel_opt: {vel_opt}")
        args.extend(self._build_grid_args(kwargs, "gen_smesh"))
        if "water_col" in kwargs and kwargs["water_col"] is not None:
            args.append(f"-W{kwargs['water_col']}")
        if "v_water" in kwargs and kwargs["v_water"] is not None:
            args.append(f"-Q{kwargs['v_water']}")
        if "v_air" in kwargs and kwargs["v_air"] is not None:
            args.append(f"-R{kwargs['v_air']}")
        hang = bool(kwargs.get("hang_sea_surface"))
        sf_out = kwargs.get("seafloor_out")
        if sf_out and not hang:
            raise ValueError("gen_smesh: seafloor_out（-G）需要同时勾选 hang_sea_surface（-S）")
        if hang:
            if vel_opt != "zelt":
                raise ValueError("gen_smesh: hang_sea_surface（-S）仅在 vel_opt='zelt' 时有效")
            args.append("-S")
            if sf_out:
                args.append(f"-G{_zelt_sscanf_token_path(sf_out)}")
        zd = kwargs.get("zelt_dump_file")
        if zd:
            if vel_opt != "zelt":
                raise ValueError("gen_smesh: zelt_dump_file（-d）仅在与 vel_opt='zelt' 联用时有效")
            args.append(f"-d{zd}")
        return args

    def resolve_cmdline_gen_smesh(self, **kwargs) -> Optional[List[str]]:
        """与 gen_smesh 将要执行的 subprocess argv 一致（含可执行文件绝对路径）；缺必选时返回 None。"""
        kw = dict(kwargs)
        kw.pop("out_file", None)
        prog = self._build_gen_smesh_program_args(kw)
        if prog is None:
            return None
        return self._compose_command("gen_smesh", prog)

    def gen_smesh(self, **kwargs):
        """
        生成慢度网格文件。

        【必选】vel_opt, grid_opt；缺一则仅打印帮助。
        vel_opt='zelt' 与 grid_opt='zelt' 必须同时成立（与 gen_smesh.cc 一致）。
        【条件必选】uniform→v0,gradient；zelt→v_in,ilayer；若出现 refl_layer 或 refl_file 则二者须成对。
        zelt 时 v_in 表单可含相对子目录；命令行只用 basename，运行前会复制到 ``proc_cwd`` 根目录。
        ``refl_file``（-F）为**输出**反射面（不必预先存在）；命令行用 basename，若表单含目录则写完后挪到目标路径。
        grid: uniform→nx,nz,xmax,zmax；variable→x_file,z_file；zelt→dx,z_file。
        【可选】topo_file（variable）, water_col, v_water, v_air, zelt_dump_file（-d，仅 vel_opt=zelt）,
            hang_sea_surface（-S，仅 zelt：topo 全 0，ilayer 当海底，以上填 v_water、以下取 v.in 绝对深度）,
            seafloor_out（-G，写出 ilayer 海底界面，须同时有 -S；供正演 -B / 反演 -Y）,
            out_file（将 stdout 慢度网格写入该路径；命令行典型用法为 gen_smesh ... > file）。
        """
        kw = dict(kwargs)
        out_file = kw.pop("out_file", None)
        prog = self._build_gen_smesh_program_args(kw)
        if prog is None:
            return self._run_cmd("gen_smesh", check_only=True)
        _verify_gen_mesh_family_inputs(self, kw, "gen_smesh")
        # 若 verify 把 refl 改成 basename，重建 argv 与之对齐
        prog = self._build_gen_smesh_program_args(kw) or prog
        result = self._run_cmd("gen_smesh", args=prog)
        _apply_zelt_output_moves(list(kw.pop("_zelt_output_moves", []) or []))
        self._write_stdout_to_file(result, out_file)
        return result
        
    def gen_damp(self, **kwargs):
        """
        生成阻尼文件。

        【必选】vel_opt, grid_opt（网格分支同 gen_smesh）。
        vel_opt='zelt' 与 grid_opt='zelt' 必须同时成立。
        【必选】abnormal_damp, normal_damp（-A；uniform/zelt 均需要数值）。
        zelt 另须 v_in, ilayer（-C）；若出现 top_layer 或 bot_layer 则须成对（-F）。
        缺 vel_opt/grid_opt 时仅打印帮助。
        【可选】out_file（将 stdout 写入路径）。
        """
        kw = dict(kwargs)
        out_file = kw.pop("out_file", None)
        prog = self._build_gen_damp_program_args(kw)
        if prog is None:
            return self._run_cmd("gen_damp", check_only=True)
        _verify_gen_mesh_family_inputs(self, kw, "gen_damp")
        result = self._run_cmd("gen_damp", args=prog)
        self._write_stdout_to_file(result, out_file)
        return result

    def _build_gen_damp_program_args(self, kwargs: Dict[str, Any]) -> Optional[List[str]]:
        if not kwargs.get("vel_opt") or not kwargs.get("grid_opt"):
            return None
        vel_opt = kwargs.get("vel_opt")
        grid_opt = kwargs.get("grid_opt")
        self._vel_grid_zelt_consistency(vel_opt, grid_opt, "gen_damp")
        # -A：异常/正常阻尼值；zelt 时 -C/-F 只划区，数值仍来自 -A（gen_damp.cc）
        self._require_keys(
            kwargs, ["abnormal_damp", "normal_damp"], "gen_damp(-A)"
        )
        args: List[str] = [
            f"-A{kwargs['abnormal_damp']}/{kwargs['normal_damp']}"
        ]
        if vel_opt == "uniform":
            pass
        elif vel_opt == "zelt":
            self._require_keys(kwargs, ["v_in", "ilayer"], "gen_damp(vel_opt='zelt')")
            args.append(f"-C{_zelt_sscanf_token_path(kwargs['v_in'])}/{kwargs['ilayer']}")
            if "top_layer" in kwargs or "bot_layer" in kwargs:
                self._require_keys(kwargs, ["top_layer", "bot_layer"], "gen_damp(layer bounds)")
                args.append(f"-F{kwargs['top_layer']}/{kwargs['bot_layer']}")
        else:
            raise ValueError(f"gen_damp 不支持的 vel_opt: {vel_opt}")
        args.extend(self._build_grid_args(kwargs, "gen_damp"))
        return args

    def resolve_cmdline_gen_damp(self, **kwargs) -> Optional[List[str]]:
        kw = dict(kwargs)
        kw.pop("out_file", None)
        prog = self._build_gen_damp_program_args(kw)
        if prog is None:
            return None
        return self._compose_command("gen_damp", prog)
        
    def gen_vcorr(self, **kwargs):
        """
        生成速度相关文件。

        mode='simple_2x2'：Python 直接写出 2×2 CorrelationLength2d（顶/底 Lh/Lv），
        不调用 gen_vcorr 二进制。【必选】Lht, Lhb, Lvt, Lvb, xmax, zmax, out_file；
        可选 xmin/zmin/topo（默认 0）。

        mode='program'（默认）：调用 gen_vcorr。
        【必选】vel_opt, grid_opt。vel_opt='zelt' 与 grid_opt='zelt' 必须同时成立。
        【必选】abnormal_h/v、normal_h/v（-A；uniform/zelt 均需要）。
        zelt 另须 v_in, ilayer；-F 层界成对规则同 gen_damp。
        【可选】out_file（将 stdout 写入路径）。
        """
        kw = dict(kwargs)
        mode = kw.pop("mode", None) or "program"
        if mode == "simple_2x2":
            return self._gen_vcorr_simple(kw)
        out_file = kw.pop("out_file", None)
        prog = self._build_gen_vcorr_program_args(kw)
        if prog is None:
            return self._run_cmd("gen_vcorr", check_only=True)
        _verify_gen_mesh_family_inputs(self, kw, "gen_vcorr")
        result = self._run_cmd("gen_vcorr", args=prog)
        self._write_stdout_to_file(result, out_file)
        return result

    def _gen_vcorr_simple(self, kw: Dict[str, Any]) -> str:
        from .simple_vcorr import write_simple_vcorr

        self._require_keys(
            kw,
            ["Lht", "Lhb", "Lvt", "Lvb", "xmax", "zmax", "out_file"],
            "gen_vcorr(simple_2x2)",
        )
        out = kw["out_file"]
        path = Path(str(out))
        if not path.is_absolute():
            base = self.proc_cwd or os.getcwd()
            path = Path(base) / path
        return write_simple_vcorr(path, **{k: v for k, v in kw.items() if k != "out_file"})

    def _build_gen_vcorr_program_args(self, kwargs: Dict[str, Any]) -> Optional[List[str]]:
        if not kwargs.get("vel_opt") or not kwargs.get("grid_opt"):
            return None
        vel_opt = kwargs.get("vel_opt")
        grid_opt = kwargs.get("grid_opt")
        self._vel_grid_zelt_consistency(vel_opt, grid_opt, "gen_vcorr")
        # -A：相关长度；zelt 时 -C/-F 只划区，数值仍来自 -A（gen_vcorr.cc）
        self._require_keys(
            kwargs,
            ["abnormal_h", "abnormal_v", "normal_h", "normal_v"],
            "gen_vcorr(-A)",
        )
        args: List[str] = [
            f"-A{kwargs['abnormal_h']}/{kwargs['abnormal_v']}/{kwargs['normal_h']}/{kwargs['normal_v']}"
        ]
        if vel_opt == "uniform":
            pass
        elif vel_opt == "zelt":
            self._require_keys(kwargs, ["v_in", "ilayer"], "gen_vcorr(vel_opt='zelt')")
            args.append(f"-C{_zelt_sscanf_token_path(kwargs['v_in'])}/{kwargs['ilayer']}")
            if "top_layer" in kwargs or "bot_layer" in kwargs:
                self._require_keys(kwargs, ["top_layer", "bot_layer"], "gen_vcorr(layer bounds)")
                args.append(f"-F{kwargs['top_layer']}/{kwargs['bot_layer']}")
        else:
            raise ValueError(f"gen_vcorr 不支持的 vel_opt: {vel_opt}")
        args.extend(self._build_grid_args(kwargs, "gen_vcorr"))
        return args

    def resolve_cmdline_gen_vcorr(self, **kwargs) -> Optional[List[str]]:
        kw = dict(kwargs)
        mode = kw.pop("mode", None) or "program"
        if mode == "simple_2x2":
            return None
        kw.pop("out_file", None)
        prog = self._build_gen_vcorr_program_args(kw)
        if prog is None:
            return None
        return self._compose_command("gen_vcorr", prog)

    def gen_dcorr(self, **kwargs):
        """
        生成 tt_inverse -CD 用的 1D 反射点相关长度文件（stdout：每行 x Lh）。

        【必选】mode = 'uniform' | 'zelt' | 'from_vcorr'
        uniform：lh, xmin, xmax；可选 nx（默认 2）
        zelt：abnormal_d, normal_d, v_in, ilayer, dx；可选 top_layer/bot_layer、refl_file
        from_vcorr：vcorr_file, refl_file
        【可选】out_file（将 stdout 写入路径）
        """
        kw = dict(kwargs)
        out_file = kw.pop("out_file", None)
        prog = self._build_gen_dcorr_program_args(kw)
        if prog is None:
            return self._run_cmd("gen_dcorr", check_only=True)
        self._verify_gen_dcorr_inputs(kw)
        prog = self._build_gen_dcorr_program_args(kw) or prog
        result = self._run_cmd("gen_dcorr", args=prog)
        self._write_stdout_to_file(result, out_file)
        return result

    def _verify_gen_dcorr_inputs(self, kw: Dict[str, Any]) -> None:
        mode = kw.get("mode")
        if mode == "zelt" and kw.get("v_in"):
            kw["v_in"] = _stage_zelt_token_under_cwd(
                self, "gen_dcorr v.in (-C)", kw.get("v_in")
            )
            if kw.get("refl_file"):
                _assert_readable_under_proc_cwd(
                    self, "gen_dcorr refl (-R)", kw.get("refl_file")
                )
        elif mode == "from_vcorr":
            _assert_readable_under_proc_cwd(
                self, "gen_dcorr vcorr (-V)", kw.get("vcorr_file")
            )
            _assert_readable_under_proc_cwd(
                self, "gen_dcorr refl (-R)", kw.get("refl_file")
            )

    def _build_gen_dcorr_program_args(self, kwargs: Dict[str, Any]) -> Optional[List[str]]:
        mode = kwargs.get("mode")
        if not mode:
            return None
        if mode == "uniform":
            self._require_keys(kwargs, ["lh", "xmin", "xmax"], "gen_dcorr(mode='uniform')")
            args: List[str] = [
                f"-A{kwargs['lh']}",
                f"-D{kwargs['xmin']}/{kwargs['xmax']}",
            ]
            if kwargs.get("nx") not in (None, ""):
                args.append(f"-N{kwargs['nx']}")
            return args
        if mode == "zelt":
            self._require_keys(
                kwargs,
                ["abnormal_d", "normal_d", "v_in", "ilayer", "dx"],
                "gen_dcorr(mode='zelt')",
            )
            args = [
                f"-A{kwargs['abnormal_d']}/{kwargs['normal_d']}",
                f"-C{_zelt_sscanf_token_path(kwargs['v_in'])}/{kwargs['ilayer']}",
                f"-E{kwargs['dx']}",
            ]
            if "top_layer" in kwargs or "bot_layer" in kwargs:
                self._require_keys(
                    kwargs, ["top_layer", "bot_layer"], "gen_dcorr(layer bounds)"
                )
                args.append(f"-F{kwargs['top_layer']}/{kwargs['bot_layer']}")
            if kwargs.get("refl_file"):
                args.extend(_tomo_glued_path_argv("R", kwargs["refl_file"]))
            return args
        if mode == "from_vcorr":
            self._require_keys(
                kwargs, ["vcorr_file", "refl_file"], "gen_dcorr(mode='from_vcorr')"
            )
            args = []
            args.extend(_tomo_glued_path_argv("V", kwargs["vcorr_file"]))
            args.extend(_tomo_glued_path_argv("R", kwargs["refl_file"]))
            return args
        raise ValueError(f"gen_dcorr 不支持的 mode: {mode}")

    def resolve_cmdline_gen_dcorr(self, **kwargs) -> Optional[List[str]]:
        kw = dict(kwargs)
        kw.pop("out_file", None)
        prog = self._build_gen_dcorr_program_args(kw)
        if prog is None:
            return None
        return self._compose_command("gen_dcorr", prog)

        
    def _build_tt_forward_program_args(
        self, smesh: Optional[str], geom: Optional[str], kwargs: Dict[str, Any]
    ) -> Optional[List[str]]:
        if smesh is None:
            return None
        args: List[str] = []
        args.extend(_tomo_glued_path_argv("M", smesh))
        if geom:
            args.extend(_tomo_glued_path_argv("G", geom))
        if "refl_file" in kwargs and kwargs["refl_file"] is not None:
            args.extend(_tomo_glued_path_argv("F", kwargs["refl_file"]))
        if kwargs.get("seafloor_file"):
            args.extend(_tomo_glued_path_argv("B", kwargs["seafloor_file"]))
        if kwargs.get("conv_file"):
            args.extend(_tomo_glued_path_argv("X", kwargs["conv_file"]))
        if kwargs.get("vsmesh"):
            args.extend(_tomo_glued_path_argv("U", kwargs["vsmesh"]))
        if kwargs.get("kappa") is not None:
            args.append(f"-k{kwargs['kappa']}")
        if kwargs.get("do_full_refl"):
            args.append("-A")
        num_keys = ["xorder", "zorder", "clen", "nintp", "tol1", "tol2"]
        if any(k in kwargs for k in num_keys):
            self._require_keys(kwargs, num_keys, "tt_forward(numerical options)")
            args.append(
                f"-N{kwargs['xorder']}/{kwargs['zorder']}/{kwargs['clen']}/"
                f"{kwargs['nintp']}/{kwargs['tol1']}/{kwargs['tol2']}"
            )
        out_opts = kwargs.get("out_opts", {}) or {}
        if "elements" in out_opts and out_opts["elements"] is not None:
            args.extend(_tomo_glued_path_argv("E", out_opts["elements"]))
        if "ttime" in out_opts and out_opts["ttime"] is not None:
            args.extend(_tomo_glued_path_argv("T", out_opts["ttime"]))
        if "obs_ttime" in out_opts and out_opts["obs_ttime"] is not None:
            args.extend(_tomo_glued_path_argv("O", out_opts["obs_ttime"]))
        if "ray" in out_opts and out_opts["ray"] is not None:
            args.extend(_tomo_glued_path_argv("R", out_opts["ray"]))
        if "source" in out_opts and out_opts["source"] is not None:
            args.extend(_tomo_glued_path_argv("S", out_opts["source"]))
        if "vgrid" in out_opts and out_opts["vgrid"] is not None:
            args.extend(_tomo_glued_path_argv("I", out_opts["vgrid"]))
        if "diff" in out_opts and out_opts["diff"] is not None:
            args.extend(_tomo_glued_path_argv("D", out_opts["diff"]))
        sub = kwargs.get("vgrid_subregion")
        if sub is not None:
            if not (isinstance(sub, (list, tuple)) and len(sub) == 6):
                raise ValueError("tt_forward: vgrid_subregion 须为 (west,east,south,north,dx,dz) 六项")
            if out_opts.get("vgrid") is None:
                raise ValueError("tt_forward: 使用 vgrid_subregion（-i）时必须指定 out_opts['vgrid']")
            w, e, s, n, dx, dz = sub
            args.append(f"-i{w}/{e}/{s}/{n}/{dx}/{dz}")
        if kwargs.get("omit_air_water"):
            args.append("-n")
        if "vred" in kwargs and kwargs["vred"] is not None:
            args.append(f"-r{kwargs['vred']}")
        if kwargs.get("graph_only"):
            args.append("-g")
        cf = kwargs.get("clock_file")
        if cf:
            args.extend(_tomo_glued_path_argv("C", cf))
        if kwargs.get("verbose"):
            if "verbose_level" in kwargs and kwargs["verbose_level"] is not None:
                args.append(f"-V{kwargs['verbose_level']}")
            else:
                args.append("-V")
        return args

    def resolve_cmdline_tt_forward(self, *, smesh=None, geom=None, **kwargs) -> Optional[List[str]]:
        kw = dict(kwargs)
        _peel_tt_forward_stdout_ttime(kw)
        prog = self._build_tt_forward_program_args(smesh, geom, kw)
        if prog is None:
            return None
        return self._compose_command("tt_forward", prog)

    def tt_forward(self, smesh=None, geom=None, **kwargs):
        """
        正演走时计算。

        【必选】smesh（-M）；缺则仅打印帮助。
        【可选】geom, refl_file, seafloor_file（-B 海底，水层/台侧多次用；-F 仍是反射面/莫霍）,
        conv_file（-X 转换面，raytype 6/7/8）, vsmesh（-U 独立 Vs，与 -M 同维）,
        kappa（-k，Vp/Vs；可 k 或 k_lid/k_below；有 -U 可不传）,
        do_full_refl（-A：反射贴界面走；不传时远偏移可穿幔成初至，不是精度开关）,
        ``out_opts['ttime']`` 把 **stdout**（与 tt_inverse -G 同构）落到该文件；原生 ``-T``
        是 ``>`` 折合图，请用 ``out_opts['ttime_plot']``。
        若传 geom 且路径可解析为本地文件（相对路径需已设 ``proc_cwd``），运行前会校验结构（首行 nsrc、每炮 s 与 nrcv 条 r、无尾部多余行），
        与 syngen.cc 一致；**正演输出**里 ``r`` 行末两列为合成走时（非 0），勿与 **geom 输入**（常为 0）混淆。
        段错误常见于 **接收点超出 smesh 模型范围** 等与格式无关的问题。
        【成组可选】-N：xorder,zorder,clen,nintp,tol1,tol2 须同时传入且 clen/tol>0，否则勿传该组。
        【可选】graph_only（-g）, clock_file（-C）, omit_air_water（-n，仅全网格 -I 时有效）,
            vgrid_subregion（-i，west/east/south/north/dx/dz，须配合 out_opts vgrid）。
        """
        kw = dict(kwargs)
        ttime_data = _peel_tt_forward_stdout_ttime(kw)
        prog = self._build_tt_forward_program_args(smesh, geom, kw)
        if prog is None:
            return self._run_cmd("tt_forward", check_only=True)
        if geom:
            gs = str(geom).strip()
            if gs:
                gp = Path(gs)
                try:
                    cwd = self.proc_cwd
                    if cwd and not gp.is_absolute():
                        gfp = (Path(cwd) / gp).resolve()
                    elif gp.is_absolute():
                        gfp = gp.resolve()
                    else:
                        gfp = None
                    if gfp is not None and gfp.is_file():
                        validate_tomo2d_geom_data_format(gfp)
                except OSError:
                    pass
        if kw.get("refl_file"):
            _assert_readable_under_proc_cwd(
                self, "tt_forward refl_file (-F 输入)", kw.get("refl_file")
            )
        if kw.get("seafloor_file"):
            _assert_readable_under_proc_cwd(
                self, "tt_forward seafloor_file (-B 输入)", kw.get("seafloor_file")
            )
        if kw.get("conv_file"):
            _assert_readable_under_proc_cwd(
                self, "tt_forward conv_file (-X 转换面)", kw.get("conv_file")
            )
        if kw.get("vsmesh"):
            _assert_readable_under_proc_cwd(
                self, "tt_forward vsmesh (-U)", kw.get("vsmesh")
            )
        old_cap = self.capture_subprocess_output
        old_cb = self.stream_output_line
        if ttime_data:
            self.capture_subprocess_output = True
            if old_cb is not None:
                def _stderr_only(name: str, line: str) -> None:
                    if name != "stdout":
                        old_cb(name, line)

                self.stream_output_line = _stderr_only
        try:
            result = self._run_cmd("tt_forward", args=prog)
        finally:
            self.capture_subprocess_output = old_cap
            self.stream_output_line = old_cb
        if ttime_data:
            stdout = getattr(result, "stdout", None) if result is not None else None
            if not stdout or not str(stdout).strip():
                raise FileNotFoundError(
                    f"tt_forward 未在标准输出写出走时，无法写入 {ttime_data!r}。"
                    "（合成走时在 stdout，不是原生 -T 文件。）"
                )
            self._write_stdout_to_file(result, ttime_data)
            out_path = Path(ttime_data)
            if not out_path.is_absolute():
                base = self.proc_cwd or os.getcwd()
                out_path = Path(base) / out_path
            try:
                validate_tomo2d_geom_data_format(out_path)
            except ValueError as e:
                raise ValueError(
                    "tt_forward 标准输出不是 tt_inverse 可用的走时格式"
                    f"（原生 -T 是 '>' 折合图，读了会 invalid nsrc）。\n{e}"
                ) from e
        return result
        
    def _build_tt_inverse_program_args(
        self, mesh: Optional[str], data: Optional[str], kwargs: Dict[str, Any]
    ) -> Optional[List[str]]:
        if mesh is None or data is None:
            return None
        kwargs.pop("_refl_stride", None)
        args = [f"-M{mesh}", f"-G{data}"]
        num_keys = ["xorder", "zorder", "clen", "nintp", "bend_cg_tol", "bend_br_tol"]
        if any(k in kwargs for k in num_keys):
            self._require_keys(kwargs, num_keys, "tt_inverse(numerical options)")
            args.append(
                f"-N{kwargs['xorder']}/{kwargs['zorder']}/{kwargs['clen']}/"
                f"{kwargs['nintp']}/{kwargs['bend_cg_tol']}/{kwargs['bend_br_tol']}"
            )
        if "refl_file" in kwargs and kwargs["refl_file"] is not None:
            args.append(f"-F{kwargs['refl_file']}")
        if kwargs.get("seafloor_file"):
            args.extend(_tomo_glued_path_argv("Y", kwargs["seafloor_file"]))
        if kwargs.get("conv_file"):
            args.extend(_tomo_glued_path_argv("B", kwargs["conv_file"]))
        if kwargs.get("vsmesh"):
            args.extend(_tomo_glued_path_argv("U", kwargs["vsmesh"]))
        if kwargs.get("kappa") is not None:
            args.append(f"-k{kwargs['kappa']}")
        if kwargs.get("invert_water_only") and kwargs.get("invert_crust_only"):
            raise ValueError("tt_inverse: -y (invert water only) and -w (invert crust only) are mutually exclusive")
        if kwargs.get("invert_water_only"):
            if not kwargs.get("seafloor_file") and not kwargs.get("refl_file"):
                raise ValueError("tt_inverse: -y (invert water only) requires -Y or -F")
            args.append("-y")
        if kwargs.get("invert_crust_only"):
            if not kwargs.get("seafloor_file") and not kwargs.get("refl_file"):
                raise ValueError("tt_inverse: -w (invert crust only) requires -Y or -F")
            args.append("-w")
        if kwargs.get("freeze_refl"):
            if not kwargs.get("refl_file"):
                raise ValueError("tt_inverse: -u (freeze reflector) requires -F")
            args.append("-u")
        if kwargs.get("do_full_refl"):
            args.append("-A")
        if "refl_weight" in kwargs and kwargs["refl_weight"] is not None:
            args.append(f"-W{kwargs['refl_weight']}")
        if kwargs.get("jumping"):
            args.append("-P")
        if kwargs.get("print_final_only"):
            args.append("-l")
        if kwargs.get("apply_filter") or kwargs.get("filter_bound_file"):
            args.append(f"-s{kwargs.get('filter_bound_file') or ''}")
        if "log_file" in kwargs and kwargs["log_file"] is not None:
            args.append(f"-L{kwargs['log_file']}")
        if "out_root" in kwargs and kwargs["out_root"] is not None:
            args.append(f"-O{kwargs['out_root']}")
        if "out_level" in kwargs and kwargs["out_level"] is not None:
            args.append(f"-o{kwargs['out_level']}")
        if "dws_file" in kwargs and kwargs["dws_file"] is not None:
            args.append(f"-K{kwargs['dws_file']}")
        if "crit_chi" in kwargs and kwargs["crit_chi"] is not None:
            args.append(f"-R{kwargs['crit_chi']}")
        if "lsqr_tol" in kwargs and kwargs["lsqr_tol"] is not None:
            args.append(f"-Q{kwargs['lsqr_tol']}")
        if "niter" in kwargs and kwargs["niter"] is not None:
            args.append(f"-I{kwargs['niter']}")
        if "target_chi2" in kwargs and kwargs["target_chi2"] is not None:
            args.append(f"-J{kwargs['target_chi2']}")
        auto_dv = kwargs.get("auto_damp_max_dv")
        auto_dd = kwargs.get("auto_damp_max_dd")
        if auto_dv is not None:
            args.append(f"-TV{auto_dv}")
        if auto_dd is not None:
            args.append(f"-TD{auto_dd}")
        damp_opts = kwargs.get("damp_opts", {}) or {}
        had_auto = auto_dv is not None or auto_dd is not None
        had_fixed = any(
            damp_opts.get(k) is not None for k in ("vel", "dep", "damp_v_fn")
        )
        if had_auto and had_fixed:
            raise ValueError("tt_inverse: 自动阻尼 (-TV/-TD) 与固定阻尼 (-DV/-DD/-DQ) 不能同时使用")
        smooth_opts = kwargs.get("smooth_opts", {}) or {}
        if "vel" in smooth_opts and smooth_opts["vel"] is not None:
            args.append(f"-SV{self._tt_inverse_sv_sd_arg(smooth_opts['vel'])}")
        if smooth_opts.get("vel_log10"):
            args.append("-XV")
        if "dep" in smooth_opts and smooth_opts["dep"] is not None:
            args.append(f"-SD{self._tt_inverse_sv_sd_arg(smooth_opts['dep'])}")
        if smooth_opts.get("dep_log10"):
            args.append("-XD")
        if "corr_v_fn" in smooth_opts and smooth_opts["corr_v_fn"] is not None:
            args.append(f"-CV{smooth_opts['corr_v_fn']}")
        if "corr_d_fn" in smooth_opts and smooth_opts["corr_d_fn"] is not None:
            args.append(f"-CD{smooth_opts['corr_d_fn']}")
        if "vel" in damp_opts and damp_opts["vel"] is not None:
            args.append(f"-DV{damp_opts['vel']}")
        if "dep" in damp_opts and damp_opts["dep"] is not None:
            args.append(f"-DD{damp_opts['dep']}")
        if "damp_v_fn" in damp_opts and damp_opts["damp_v_fn"] is not None:
            args.append(f"-DQ{damp_opts['damp_v_fn']}")
        grav = kwargs.get("gravity_opts", {}) or {}
        if grav.get("grav_file"):
            self._tt_inverse_append_gravity(args, grav)
        if kwargs.get("verbose"):
            if "verbose_level" in kwargs and kwargs["verbose_level"] is not None:
                args.append(f"-V{kwargs['verbose_level']}")
            else:
                args.append("-V")
        return args

    def resolve_cmdline_tt_inverse(self, *, mesh=None, data=None, **kwargs) -> Optional[List[str]]:
        kw = dict(kwargs)
        prog = self._build_tt_inverse_program_args(mesh, data, kw)
        if prog is None:
            return None
        return self._compose_command("tt_inverse", prog)

    def tt_inverse(self, mesh=None, data=None, **kwargs):
        """
        走时反演。

        【必选】mesh（-M）, data（-G）；缺一则仅打印帮助。
        【成组可选】-N：xorder,zorder,clen,nintp,bend_cg_tol,bend_br_tol（规则同 tt_forward）。
        【可选】refl_file, seafloor_file（-Y 海底；与正演 -B 不同，反演 -B 是转换波界面）,
        conv_file（反演 -B，6 折合 PSP / 7/8 钉点；不劫持 0/1）, vsmesh（-U 独立 Vs 初值；有则不再用 Vp/κ 覆盖 Vs）, kappa（-k，6 只反面下并冻盖层；有 7/8 才解冻盖层；反演 -Q 仍是 LSQR）, invert_water_only（-y，冻壳只反水；须 -Y 或 -F）, invert_crust_only（-w，冻水只反壳；须 -Y 或 -F；与 -y 互斥）, freeze_refl（-u，冻结 -F 几何；须同时有 refl_file）, do_full_refl（-A：反射贴界面；不传时远偏移可穿幔）, refl_weight, jumping, print_final_only,
            apply_filter（裸 -s，用 mesh 地形）, filter_bound_file（-s 文件；有文件则不必再传 apply_filter）,
            log_file, out_root, out_level, dws_file, crit_chi, lsqr_tol, niter, target_chi2,
            smooth_opts（含 vel/dep 单值或 min/max/dw 字符串、vel_log10/dep_log10）,
            auto_damp_max_dv / auto_damp_max_dd（-TV/-TD，与固定阻尼互斥）,
            damp_opts, gravity_opts, verbose, verbose_level。
        命令行细节见 ``TomoHelp.tt_inverse_help()``。
        """
        kw = dict(kwargs)
        prog = self._build_tt_inverse_program_args(mesh, data, kw)
        if prog is None:
            return self._run_cmd("tt_inverse", check_only=True)
        if kw.get("refl_file"):
            _assert_readable_under_proc_cwd(
                self, "tt_inverse refl_file (-F 输入)", kw.get("refl_file")
            )
        if kw.get("seafloor_file"):
            _assert_readable_under_proc_cwd(
                self, "tt_inverse seafloor_file (-Y 输入)", kw.get("seafloor_file")
            )
        if kw.get("conv_file"):
            _assert_readable_under_proc_cwd(
                self, "tt_inverse conv_file (-B 转换面)", kw.get("conv_file")
            )
        if kw.get("vsmesh"):
            _assert_readable_under_proc_cwd(
                self, "tt_inverse vsmesh (-U)", kw.get("vsmesh")
            )
        return self._run_cmd("tt_inverse", args=prog)
        
    def _build_stat_smesh_program_args(self, kwargs: Dict[str, Any]) -> Optional[List[str]]:
        if not kwargs.get("mode"):
            return None
        args: List[str] = []
        mode = kwargs.get("mode")
        if mode == "list":
            self._require_keys(kwargs, ["list_file", "cmd_type"], "stat_smesh(mode='list')")
            args.append(f"-L{kwargs['list_file']}")
            if kwargs.get("cmd_type") == "a":
                args.append("-Ca")
            elif kwargs.get("cmd_type") == "r":
                self._require_keys(kwargs, ["ave_file"], "stat_smesh(cmd_type='r')")
                args.append(f"-Cr{kwargs['ave_file']}")
            else:
                raise ValueError(f"stat_smesh(list) 不支持的 cmd_type: {kwargs.get('cmd_type')}")
            rn = kwargs.get("refl_nnodes")
            if rn is not None:
                args.append(f"-R{int(rn)}")
        elif mode == "mesh":
            self._require_keys(kwargs, ["mesh_file", "cmd_type"], "stat_smesh(mode='mesh')")
            args.append(f"-M{kwargs['mesh_file']}")
            if kwargs.get("cmd_type") == "a":
                self._require_keys(kwargs, ["ave_x", "window_len"], "stat_smesh(mesh cmd='a')")
                args.append(f"-Da{kwargs['ave_x']}/{kwargs['window_len']}")
            elif kwargs.get("cmd_type") == "b":
                self._require_keys(kwargs, ["xmin", "xmax", "dx", "window_len"], "stat_smesh(mesh cmd='b')")
                args.append(f"-Db{kwargs['xmin']}/{kwargs['xmax']}/{kwargs['dx']}/{kwargs['window_len']}")
            else:
                raise ValueError(f"stat_smesh(mesh) 不支持的 cmd_type: {kwargs.get('cmd_type')}")
        else:
            raise ValueError(f"stat_smesh 不支持的 mode: {mode}")
        if "top_bound" in kwargs and kwargs["top_bound"] is not None:
            args.append(f"-T{kwargs['top_bound']}")
        if "bot_bound" in kwargs and kwargs["bot_bound"] is not None:
            args.append(f"-B{kwargs['bot_bound']}")
        if "mid_bound" in kwargs and kwargs["mid_bound"] is not None:
            args.append(f"-m{kwargs['mid_bound']}")
        if mode == "mesh" and kwargs.get("cmd_type") == "b":
            self._require_keys(
                kwargs,
                ["top_bound", "bot_bound", "mid_bound"],
                "stat_smesh(mesh -Db 需要顶/底/中界文件)",
            )
        if mode == "mesh":
            pc = kwargs.get("pt_corr")
            if pc:
                args.append(f"-P{pc}")
            if kwargs.get("vrepl") is not None:
                args.append(f"-U{kwargs['vrepl']}")
            ax0 = kwargs.get("abs_xmin")
            ax1 = kwargs.get("abs_xmax")
            if ax0 is not None or ax1 is not None:
                self._require_keys(
                    {"abs_xmin": ax0, "abs_xmax": ax1},
                    ["abs_xmin", "abs_xmax"],
                    "stat_smesh(-X)",
                )
                args.append(f"-X{ax0}/{ax1}")
            ex0 = kwargs.get("exclude_cxmin")
            ex1 = kwargs.get("exclude_cxmax")
            et = kwargs.get("exclude_top_bound")
            eb = kwargs.get("exclude_bot_bound")
            if any(x is not None for x in (ex0, ex1, et, eb)):
                self._require_keys(
                    {
                        "exclude_cxmin": ex0,
                        "exclude_cxmax": ex1,
                        "exclude_top_bound": et,
                        "exclude_bot_bound": eb,
                    },
                    ["exclude_cxmin", "exclude_cxmax", "exclude_top_bound", "exclude_bot_bound"],
                    "stat_smesh(剔除大陆带 -x/-t/-b)",
                )
                args.append(f"-x{ex0}/{ex1}")
                args.append(f"-t{et}")
                args.append(f"-b{eb}")
        if kwargs.get("verbose"):
            args.append("-V")
        return args

    def resolve_cmdline_stat_smesh(self, **kwargs) -> Optional[List[str]]:
        kw = dict(kwargs)
        prog = self._build_stat_smesh_program_args(kw)
        if prog is None:
            return None
        return self._compose_command("stat_smesh", prog)

    def stat_smesh(self, **kwargs):
        """
        慢度网格统计。

        【必选】mode（list|mesh）；缺则仅打印帮助。
        list→list_file, cmd_type（a|r）；cmd_type=r 另须 ave_file。
        mesh→mesh_file, cmd_type（a|b）；a→ave_x,window_len；b→xmin,xmax,dx,window_len。
        【可选】top_bound, bot_bound, mid_bound, verbose,
            refl_nnodes（list 模式 -R）, pt_corr（mesh -P 六段斜杠）, vrepl（-U）,
            abs_xmin/abs_xmax（-X，须成对）,
            exclude_cxmin/exclude_cxmax/exclude_top_bound/exclude_bot_bound（-x/-t/-b，四项须齐）。
        mesh 且 cmd_type=b 时原生要求 -T/-B/-m 均已给出。
        """
        kw = dict(kwargs)
        prog = self._build_stat_smesh_program_args(kw)
        if prog is None:
            return self._run_cmd("stat_smesh", check_only=True)
        return self._run_cmd("stat_smesh", args=prog)

if __name__ == "__main__":
    # 使用示例
    tomo = TomoAnd()
    
    # 不带参数调用gen_smesh，将显示使用说明
    tomo.gen_smesh()
