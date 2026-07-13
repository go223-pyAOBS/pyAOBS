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
from typing import Any, Dict, List, Optional, Union

from .help_docs import TomoHelp


def _assert_readable_under_proc_cwd(tomo: "TomoAnd", role: str, path: Any) -> None:
    """
    gen_smesh 等对 z_file 调用 countLines/open；路径相对于进程 cwd（GUI 下即 work_dir）。
    在启动子进程前检查，避免 C 端仅打印 ``countLines::can't open``。
    """
    cwd = tomo.proc_cwd
    if not cwd or path is None:
        return
    s = str(path).strip()
    if not s:
        return
    p = Path(s)
    try:
        full = (Path(cwd) / p).resolve() if not p.is_absolute() else p.resolve()
    except OSError as e:
        raise FileNotFoundError(f"{role}: 路径无效 {s!r} ({e})") from e
    if not full.is_file():
        raise FileNotFoundError(
            f"{role}: 找不到或不是可读文件\n"
            f"  参数: {s!r}\n"
            f"  proc_cwd（子进程工作目录）: {cwd!r}\n"
            f"  解析为: {full}\n"
            f"  提示: z_file / x_file 等与 v.in 不同，可用相对路径（相对上述目录）或绝对路径；"
            f"请确认文件已存在且路径与界面「工作目录」一致。"
        )


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
        _assert_readable_under_proc_cwd(
            tomo, f"{context} v.in (-C)", _zelt_sscanf_token_path(str(kw["v_in"]))
        )
    if context == "gen_smesh" and vo == "zelt" and kw.get("refl_file"):
        _assert_readable_under_proc_cwd(
            tomo, f"{context} refl_file (-F)", _zelt_sscanf_token_path(str(kw["refl_file"]))
        )


def _zelt_sscanf_token_path(path: Any) -> str:
    """
    gen_smesh / gen_damp / gen_vcorr 中 ``-C<vpath>/<ilayer>``、``-F<layer>/<rpath>``
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


def _tomo_path_should_split_for_argv(s: str) -> bool:
    """
    是否将 ``tt_forward`` 等单字母选项拆成 ``-M`` 与路径两个 argv。

    ``-M/mnt/e/mesh`` 在老版解析里本应等价于 ``mfn=/mnt/e/mesh``；但 POSIX 绝对路径、
    Windows 盘符/UNC、以及以 ``-`` 开头的文件名在部分环境下更易出错，故拆成两参数；
    需使用已支持该语法的 ``tt_forward``（见 modeling/tomo2d/src/tt_forward.cc）。
    """
    if not s:
        return False
    if s.startswith("/"):
        return True
    if len(s) >= 2 and s[1] == ":" and s[0].isalpha():
        return True
    if s.startswith("\\\\"):
        return True
    if s.startswith("\\"):
        return True
    if s.startswith("-"):
        return True
    return False


def _tomo_glued_path_argv(short_flag: str, path: Any) -> List[str]:
    """``tt_forward`` 路径类选项：相对路径保持 ``-Mrel``，否则 ``-M`` + ``path``。"""
    s = str(path).strip() if path is not None else ""
    if not s:
        return [f"-{short_flag}"]
    if _tomo_path_should_split_for_argv(s):
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
            capture_subprocess_output: 为 True（默认）时用管道捕获 stdout/stderr，进程结束后才能在
                ``CompletedProcess`` / ``TomoCommandError`` 里查看；长时间任务若要在终端**实时**看到
                子进程输出（含 ``tt_forward -V`` 打到 stderr 的进展），设为 False，子进程继承当前终端。
        """
        self.bin_path = self._resolve_bin_path(bin_path)
        #: 若设置，subprocess 将在此目录下启动（相对路径输入/输出均相对该目录）。GUI 多线程下可避免依赖进程全局 chdir。
        self.proc_cwd: Optional[str] = None
        self.capture_subprocess_output = capture_subprocess_output

    def _resolve_bin_path(self, bin_path: Optional[str]) -> str:
        if bin_path:
            return bin_path
        env_path = os.getenv("PYAOBS_TOMO2D_BIN") or os.getenv("TOMO2D_BIN")
        if env_path:
            return env_path
        return os.path.dirname(os.path.abspath(__file__))

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
                return candidate

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
        if self.proc_cwd:
            run_kw["cwd"] = self.proc_cwd
        try:
            return subprocess.run(cmd, **run_kw)
        except subprocess.CalledProcessError as e:
            raise TomoCommandError(
                command=cmd,
                returncode=e.returncode,
                stdout=e.stdout or "",
                stderr=e.stderr or "",
            ) from e
            
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
        zelt 时 v_in、refl_file 请传**仅文件名**（无 ``/``），文件须在 ``proc_cwd`` 根目录（与 gen_smesh.cc sscanf 一致）。
        grid: uniform→nx,nz,xmax,zmax；variable→x_file,z_file；zelt→dx,z_file。
        【可选】topo_file（variable）, water_col, v_water, v_air, zelt_dump_file（-d，仅 vel_opt=zelt）,
            out_file（将 stdout 慢度网格写入该路径；命令行典型用法为 gen_smesh ... > file）。
        """
        kw = dict(kwargs)
        out_file = kw.pop("out_file", None)
        prog = self._build_gen_smesh_program_args(kw)
        if prog is None:
            return self._run_cmd("gen_smesh", check_only=True)
        _verify_gen_mesh_family_inputs(self, kw, "gen_smesh")
        result = self._run_cmd("gen_smesh", args=prog)
        self._write_stdout_to_file(result, out_file)
        return result
        
    def gen_damp(self, **kwargs):
        """
        生成阻尼文件。

        【必选】vel_opt, grid_opt（网格分支同 gen_smesh）。
        vel_opt='zelt' 与 grid_opt='zelt' 必须同时成立。
        uniform→abnormal_damp, normal_damp；zelt→v_in, ilayer；
        若出现 top_layer 或 bot_layer 则须成对。缺 vel_opt/grid_opt 时仅打印帮助。
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
        args: List[str] = []
        if vel_opt == "uniform":
            self._require_keys(kwargs, ["abnormal_damp", "normal_damp"], "gen_damp(vel_opt='uniform')")
            args.append(f"-A{kwargs['abnormal_damp']}/{kwargs['normal_damp']}")
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

        【必选】vel_opt, grid_opt。vel_opt='zelt' 与 grid_opt='zelt' 必须同时成立。
        uniform→abnormal_h, abnormal_v, normal_h, normal_v；
        zelt→v_in, ilayer；-F 层界成对规则同 gen_damp。
        【可选】out_file（将 stdout 写入路径）。
        """
        kw = dict(kwargs)
        out_file = kw.pop("out_file", None)
        prog = self._build_gen_vcorr_program_args(kw)
        if prog is None:
            return self._run_cmd("gen_vcorr", check_only=True)
        _verify_gen_mesh_family_inputs(self, kw, "gen_vcorr")
        result = self._run_cmd("gen_vcorr", args=prog)
        self._write_stdout_to_file(result, out_file)
        return result

    def _build_gen_vcorr_program_args(self, kwargs: Dict[str, Any]) -> Optional[List[str]]:
        if not kwargs.get("vel_opt") or not kwargs.get("grid_opt"):
            return None
        vel_opt = kwargs.get("vel_opt")
        grid_opt = kwargs.get("grid_opt")
        self._vel_grid_zelt_consistency(vel_opt, grid_opt, "gen_vcorr")
        args: List[str] = []
        if vel_opt == "uniform":
            self._require_keys(
                kwargs,
                ["abnormal_h", "abnormal_v", "normal_h", "normal_v"],
                "gen_vcorr(vel_opt='uniform')",
            )
            args.append(
                f"-A{kwargs['abnormal_h']}/{kwargs['abnormal_v']}/{kwargs['normal_h']}/{kwargs['normal_v']}"
            )
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
        kw.pop("out_file", None)
        prog = self._build_gen_vcorr_program_args(kw)
        if prog is None:
            return None
        return self._compose_command("gen_vcorr", prog)
        
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
        prog = self._build_tt_forward_program_args(smesh, geom, kw)
        if prog is None:
            return None
        return self._compose_command("tt_forward", prog)

    def tt_forward(self, smesh=None, geom=None, **kwargs):
        """
        正演走时计算。

        【必选】smesh（-M）；缺则仅打印帮助。
        【可选】geom, refl_file, do_full_refl, out_opts 各键, vred, verbose, verbose_level。
        若传 geom 且路径可解析为本地文件（相对路径需已设 ``proc_cwd``），运行前会校验结构（首行 nsrc、每炮 s 与 nrcv 条 r、无尾部多余行），
        与 syngen.cc 一致；**正演输出**里 ``r`` 行末两列为合成走时（非 0），勿与 **geom 输入**（常为 0）混淆。
        段错误常见于 **接收点超出 smesh 模型范围** 等与格式无关的问题。
        【成组可选】-N：xorder,zorder,clen,nintp,tol1,tol2 须同时传入且 clen/tol>0，否则勿传该组。
        【可选】graph_only（-g）, clock_file（-C）, omit_air_water（-n，仅全网格 -I 时有效）,
            vgrid_subregion（-i，west/east/south/north/dx/dz，须配合 out_opts vgrid）。
        """
        kw = dict(kwargs)
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
        return self._run_cmd("tt_forward", args=prog)
        
    def _build_tt_inverse_program_args(
        self, mesh: Optional[str], data: Optional[str], kwargs: Dict[str, Any]
    ) -> Optional[List[str]]:
        if mesh is None or data is None:
            return None
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
        if kwargs.get("do_full_refl"):
            args.append("-A")
        if "refl_weight" in kwargs and kwargs["refl_weight"] is not None:
            args.append(f"-W{kwargs['refl_weight']}")
        if kwargs.get("jumping"):
            args.append("-P")
        if kwargs.get("print_final_only"):
            args.append("-l")
        if "filter_bound_file" in kwargs and kwargs["filter_bound_file"]:
            args.append(f"-s{kwargs['filter_bound_file']}")
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
        【可选】refl_file, do_full_refl, refl_weight, jumping, print_final_only, filter_bound_file,
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
