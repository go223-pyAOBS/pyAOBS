"""
2D theory service (RAYINVR-first).

提供：
1) RAYINVR 输入校验与一键正演；
2) tx.out 按 shot/phase 解析；
3) 基于 PPP/PPS 构建 2D Delta_t 查询函数（默认不外推）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence
import hashlib
import re
import shutil
import subprocess
import sys
import time

import numpy as np

try:
    from ...modeling.rayinvr.rayinvr_wrapper import RayinvrWrapper
except Exception:
    RayinvrWrapper = None  # type: ignore[assignment]


def _find_rayinvr_executable() -> str | None:
    """在 PATH 中查找原生 rayinvr 可执行文件。"""
    try:
        r = subprocess.run(
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
        return "rayinvr"


def _run_wrapper_in_subprocess(working_dir: Path, *, timeout_s: int = 120) -> tuple[bool, str]:
    """
    在子进程中执行 RayinvrWrapper，避免共享库状态残留。
    """
    repo_root = Path(__file__).resolve().parents[3]
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
    return False, f"RayinvrWrapper 子进程失败 code={proc.returncode}; {msg}"


@dataclass
class ForwardRunResult:
    success: bool
    code: str
    message: str
    working_dir: str
    tx_out_path: str | None
    elapsed_s: float
    ran_forward: bool
    used_existing_txout: bool
    missing_inputs: tuple[str, ...] = ()


@dataclass
class ShotDeltaCurve:
    shot_x: float
    x: np.ndarray
    dt: np.ndarray


@dataclass
class Theory2DBundle:
    working_dir: str
    tx_out_path: str
    phase_ppp: int
    phase_pps: int
    input_hash: str
    curves: list[ShotDeltaCurve]
    global_x: np.ndarray
    global_dt: np.ndarray

    @property
    def n_shots(self) -> int:
        return len(self.curves)


@dataclass
class RayinvrInputSpec:
    r_file: Path
    t_file: Path
    v_file: Path
    tfile_from_rin: bool = False
    vfile_from_rin: bool = False


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
    """
    从 r.in 解析 tfile/vfile；若缺失则回退默认 tx.in/v.in。
    """
    wd = Path(working_dir)
    r_file = wd / "r.in"
    t_name = "tx.in"
    v_name = "v.in"
    if r_file.exists():
        txt = r_file.read_text(encoding="utf-8", errors="ignore")
        m_t = re.search(r"\btfile\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([^,\s]+))", txt, flags=re.IGNORECASE)
        if m_t:
            t_name = (m_t.group(1) or m_t.group(2) or m_t.group(3) or t_name).strip()
            t_from = True
        else:
            t_from = False
        m_v = re.search(r"\bvfile\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([^,\s]+))", txt, flags=re.IGNORECASE)
        if m_v:
            v_name = (m_v.group(1) or m_v.group(2) or m_v.group(3) or v_name).strip()
            v_from = True
        else:
            v_from = False
    else:
        t_from = False
        v_from = False
    return RayinvrInputSpec(
        r_file=r_file,
        t_file=(wd / t_name),
        v_file=(wd / v_name),
        tfile_from_rin=t_from,
        vfile_from_rin=v_from,
    )


def validate_rayinvr_inputs(working_dir: str | Path) -> tuple[bool, tuple[str, ...], RayinvrInputSpec]:
    spec = parse_rin_input_files(working_dir)
    miss: list[str] = []
    if not spec.r_file.exists():
        miss.append(str(spec.r_file.name))
    if not spec.v_file.exists():
        miss.append(str(spec.v_file.name))
    if not spec.t_file.exists():
        miss.append(str(spec.t_file.name))
    miss_t = tuple(miss)
    return len(miss_t) == 0, miss_t, spec


def update_rin_shots_from_receivers_and_depth(
    receiver_x_km: Sequence[float],
    xshot_zshot_table_x: np.ndarray | Sequence[float],
    xshot_zshot_table_z: np.ndarray | Sequence[float],
    r_in_path: str | Path,
) -> tuple[bool, str]:
    """
    根据 tx.in 的 xshot 在“xshot,zshot 列表”中查找对应 zshot，并写回 r.in 的 xshot= 与 zshot=。

    - receiver_x_km: tx.in 中每炮的 xshot（模型距离 km）
    - xshot_zshot_table_x, xshot_zshot_table_z: 已加载的 xshot、zshot 两列列表，按 xshot 查找
    - r_in_path: r.in 文件路径

    对每个 tx.in 的 xshot，在表中取 xshot 最接近的一行的 zshot。
    相同 xshot 只保留一次，避免 r.in 中 xshot/zshot 重复。
    返回 (成功与否, 提示信息)。
    """
    r_path = Path(r_in_path)
    if not r_path.exists():
        return False, f"r.in 不存在: {r_path}"
    if len(receiver_x_km) == 0:
        return False, "接收点列表为空"
    tbl_x = np.asarray(xshot_zshot_table_x, dtype=float)
    tbl_z = np.asarray(xshot_zshot_table_z, dtype=float)
    if tbl_x.size == 0 or tbl_z.size == 0 or tbl_x.size != tbl_z.size:
        return False, "xshot/zshot 列表为空或列长不一致"
    # 按首次出现顺序去重，避免同一位置写多遍
    seen: set[float] = set()
    x_list: list[float] = []
    for x in receiver_x_km:
        xf = float(x)
        key = round(xf, 6)
        if key not in seen:
            seen.add(key)
            x_list.append(xf)
    z_list = []
    for x in x_list:
        i = int(np.argmin(np.abs(tbl_x - x)))
        z_list.append(float(tbl_z[i]))
    try:
        text = r_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, f"读取 r.in 失败: {e}"
    # 替换 xshot= 与 zshot= 的数值部分（数字、逗号、空白，不含下一参数名），保留换行
    x_str = ",".join(f"{x:.3f}" for x in x_list)
    z_str = ",".join(f"{z:.3f}" for z in z_list)
    suffix = ",\n           "
    # 用替换函数避免 \1 后接数字被误解析为 \11 等
    new_text = re.sub(
        r"(xshot\s*=\s*)[\d.\s,]+(?=[a-zA-Z&]|$)",
        lambda m: m.group(1) + x_str + suffix,
        text,
    )
    new_text = re.sub(
        r"(zshot\s*=\s*)[\d.\s,]+(?=[a-zA-Z&]|$)",
        lambda m: m.group(1) + z_str + suffix,
        new_text,
    )
    new_text = _ensure_rin_tfile_vfile_quoted(new_text)
    try:
        r_path.write_text(new_text, encoding="utf-8")
    except Exception as e:
        return False, f"写入 r.in 失败: {e}"
    return True, f"已更新 r.in: xshot({len(x_list)}), zshot 来自炮点深度表"


def _extract_tfile_vfile_name(rhs: str) -> str | None:
    """
    从 tfile/vfile 行右侧（= 与行末逗号之间）提取文件名。
    兼容：\"v.in\"、\",v.in\"、\"\",v.in\"、裸 v.in 等损坏写法。
    """
    rhs = rhs.strip()
    if not rhs:
        return None
    m = re.search(r"([\w./-]+\.in)\b", rhs, flags=re.IGNORECASE)
    return m.group(1).replace("\\", "/") if m else None


def _ensure_rin_tfile_vfile_quoted(text: str) -> str:
    """
    确保 r.in 中 tfile= 与 vfile= 的值有双引号，且以逗号结尾（RAYINVR 规则：每个参数必须以逗号结尾）。

    按**整行**重写，避免旧正则 ([^,\\n]+?)(?=\\s*\\S) 在 vfile=\"\" 处提前结束，
    误生成 vfile=\"\",v.in\" 这类损坏。
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    for line in lines:
        # 行末逗号前为“值”部分；贪婪捕获到**最后一个**逗号，以处理 \"\",v.in\" 等多逗号损坏
        m = re.match(
            r"^(\s*\b(?:tfile|vfile)\s*=\s*)(.*)\s*,(\s*(?:\r?\n)?)$",
            line,
            flags=re.IGNORECASE,
        )
        if not m:
            out.append(line)
            continue
        pre, rhs, trail = m.group(1), m.group(2), m.group(3)
        fname = _extract_tfile_vfile_name(rhs)
        if fname is None:
            out.append(line)
            continue
        out.append(f'{pre}"{fname}",{trail}')
    return "".join(out)


def set_rin_tfile(r_in_path: str | Path, tfile_name: str) -> tuple[bool, str]:
    """
    将 r.in 中的 tfile 更新为指定文件名（带双引号、逗号结尾）。
    仅替换 tfile 的值部分，不触动 vfile 等其他参数。
    """
    r_path = Path(r_in_path)
    if not r_path.exists():
        return False, f"r.in 不存在: {r_path}"
    try:
        text = r_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, f"读取 r.in 失败: {e}"
    # 精确匹配 tfile= 后的值（支持 "xxx" 或 xxx 格式），并吃掉行末已有逗号，避免写成 tfile="...",,
    def _repl(m: re.Match) -> str:
        return f'{m.group(1)}"{tfile_name}",'
    new_text = re.sub(
        r"(\btfile\s*=\s*)(?:\"([^\"]*)\"|'([^']*)'|([^,\s\n]+))\s*,?",
        _repl,
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    if new_text == text:
        return False, "未找到 tfile= 行"
    try:
        r_path.write_text(new_text, encoding="utf-8")
    except Exception as e:
        return False, f"写入 r.in 失败: {e}"
    return True, f"已更新 r.in: tfile={tfile_name}"


def parse_pois_from_rin(r_in_path: str | Path) -> tuple[float | None, str]:
    """
    从 r.in 读取 pois= 行，返回 (第一个值, 从第二个值起用逗号拼接的字符串)。
    若未找到或解析失败，返回 (None, "")。
    """
    r_path = Path(r_in_path)
    if not r_path.exists():
        return None, ""
    try:
        text = r_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None, ""
    m = re.search(r"pois\s*=\s*([\d., \t]+)", text, flags=re.IGNORECASE)
    if not m:
        return None, ""
    raw = m.group(1)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    values: list[float] = []
    for p in parts:
        try:
            values.append(float(p))
        except ValueError:
            continue
    if not values:
        return None, ""
    first = values[0]
    rest_str = ",".join(f"{v:.3f}" for v in values[1:])
    return first, rest_str


def parse_pois_full_from_rin(r_in_path: str | Path) -> str:
    """从 r.in 读取 pois= 行，返回完整取值字符串（用于左/右支输入框初始值）。"""
    first, rest = parse_pois_from_rin(r_in_path)
    if first is None:
        return ""
    return f"{first:.3f},{rest}" if rest else f"{first:.3f}"


def write_pois_full_to_rin(r_in_path: str | Path, pois_full_str: str) -> tuple[bool, str]:
    """
    将完整 pois 字符串写回 r.in（用于左支或右支单独写回）。
    pois_full_str 为整行取值，如 "0.5,0.44,0.4"。
    """
    r_path = Path(r_in_path)
    if not r_path.exists():
        return False, f"r.in 不存在: {r_path}"
    parts = [p.strip() for p in pois_full_str.split(",") if p.strip()]
    vals: list[float] = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            return False, f"pois 含非数字: {p!r}"
    if not vals:
        return False, "pois 为空"
    new_val = ",".join(f"{v:.3f}" for v in vals) + ","
    try:
        text = r_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, f"读取 r.in 失败: {e}"
    # 仅匹配 pois 的值，不含换行，避免吃掉 vfile/tfile 行
    new_text = re.sub(
        r"(pois\s*=\s*)[\d., \t]+",
        lambda m: m.group(1) + new_val + "\n           ",
        text,
        flags=re.IGNORECASE,
    )
    if new_text == text:
        return False, "未找到 pois= 行"
    new_text = _ensure_rin_tfile_vfile_quoted(new_text)
    try:
        r_path.write_text(new_text, encoding="utf-8")
    except Exception as e:
        return False, f"写入 r.in 失败: {e}"
    return True, "已写回 pois 到 r.in"


def write_pois_to_rin(
    r_in_path: str | Path,
    first_value: float,
    rest_str: str,
) -> tuple[bool, str]:
    """
    将 pois= 写回 r.in：pois=first_value,<rest_str 解析后的值>,。
    rest_str 为逗号分隔的数字字符串（从第二个值起）。
    """
    r_path = Path(r_in_path)
    if not r_path.exists():
        return False, f"r.in 不存在: {r_path}"
    rest_parts = [p.strip() for p in rest_str.split(",") if p.strip()]
    rest_vals: list[float] = []
    for p in rest_parts:
        try:
            rest_vals.append(float(p))
        except ValueError:
            return False, f"pois 含非数字: {p!r}"
    try:
        text = r_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return False, f"读取 r.in 失败: {e}"
    new_val = f"{first_value:.3f}," + ",".join(f"{v:.3f}" for v in rest_vals) + ","
    new_text = re.sub(
        r"(pois\s*=\s*)[\d., \t]+",
        lambda m: m.group(1) + new_val + "\n           ",
        text,
        flags=re.IGNORECASE,
    )
    if new_text == text:
        return False, "未找到 pois= 行"
    new_text = _ensure_rin_tfile_vfile_quoted(new_text)
    try:
        r_path.write_text(new_text, encoding="utf-8")
    except Exception as e:
        return False, f"写入 r.in 失败: {e}"
    return True, "已写回 pois 到 r.in"


def run_rayinvr_forward(
    working_dir: str | Path,
    *,
    force_run: bool = False,
    tx_in_override: str | Path | None = None,
    sync_override_to_tfile: bool = True,
) -> ForwardRunResult:
    wd = Path(working_dir)
    tx_out = wd / "tx.out"
    t0 = time.time()
    ok, miss, spec = validate_rayinvr_inputs(wd)
    # 可选：把当前打开的 in 文件同步到 r.in 指定的 tfile
    if tx_in_override is not None and sync_override_to_tfile:
        src = Path(tx_in_override)
        if src.exists():
            try:
                spec.t_file.parent.mkdir(parents=True, exist_ok=True)
                # 源与目标相同文件时无需 copy（否则 Windows/WSL 路径混用下可能抛 samefile）
                src_r = src.resolve()
                dst_r = spec.t_file.resolve()
                if src_r != dst_r:
                    shutil.copy2(src, spec.t_file)
                ok, miss, spec = validate_rayinvr_inputs(wd)
            except Exception as e:
                return ForwardRunResult(
                    success=False,
                    code="sync_tfile_failed",
                    message=f"同步 tfile 失败: {e}",
                    working_dir=str(wd),
                    tx_out_path=str(tx_out) if tx_out.exists() else None,
                    elapsed_s=time.time() - t0,
                    ran_forward=False,
                    used_existing_txout=False,
                )
    if not ok:
        return ForwardRunResult(
            success=False,
            code="missing_inputs",
            message=f"缺少输入文件: {', '.join(miss)}",
            working_dir=str(wd),
            tx_out_path=str(tx_out) if tx_out.exists() else None,
            elapsed_s=time.time() - t0,
            ran_forward=False,
            used_existing_txout=False,
            missing_inputs=miss,
        )

    if tx_out.exists() and not force_run:
        return ForwardRunResult(
            success=True,
            code="ok",
            message="使用现有 tx.out",
            working_dir=str(wd),
            tx_out_path=str(tx_out),
            elapsed_s=time.time() - t0,
            ran_forward=False,
            used_existing_txout=True,
        )

    # 运行前删除旧 tx.out，确保全新输出
    if tx_out.exists():
        try:
            tx_out.unlink()
        except Exception:
            pass

    # 优先用原生 rayinvr 可执行文件（subprocess，每次全新进程，无状态残留）
    exe = _find_rayinvr_executable()
    if exe is not None:
        try:
            proc = subprocess.run(
                [exe],
                cwd=str(wd),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=120,
            )
            if proc.returncode != 0 and not tx_out.exists():
                return ForwardRunResult(
                    success=False,
                    code="run_failed",
                    message=f"rayinvr 退出码={proc.returncode}",
                    working_dir=str(wd),
                    tx_out_path=None,
                    elapsed_s=time.time() - t0,
                    ran_forward=True,
                    used_existing_txout=False,
                )
        except subprocess.TimeoutExpired:
            return ForwardRunResult(
                success=False,
                code="run_timeout",
                message="rayinvr 运行超时(120s)",
                working_dir=str(wd),
                tx_out_path=str(tx_out) if tx_out.exists() else None,
                elapsed_s=time.time() - t0,
                ran_forward=True,
                used_existing_txout=False,
            )
        except Exception as e:
            return ForwardRunResult(
                success=False,
                code="run_exception",
                message=f"rayinvr 异常: {e}",
                working_dir=str(wd),
                tx_out_path=str(tx_out) if tx_out.exists() else None,
                elapsed_s=time.time() - t0,
                ran_forward=True,
                used_existing_txout=False,
            )
    else:
        # 回退：在子进程中执行 RayinvrWrapper（方案2）
        ok_run, msg = _run_wrapper_in_subprocess(wd, timeout_s=120)
        if not ok_run:
            return ForwardRunResult(
                success=False,
                code="run_exception",
                message=msg or "RayinvrWrapper 子进程运行失败",
                working_dir=str(wd),
                tx_out_path=str(tx_out) if tx_out.exists() else None,
                elapsed_s=time.time() - t0,
                ran_forward=True,
                used_existing_txout=False,
            )

    if not tx_out.exists():
        return ForwardRunResult(
            success=False,
            code="txout_missing",
            message="RAYINVR 完成但未生成 tx.out",
            working_dir=str(wd),
            tx_out_path=None,
            elapsed_s=time.time() - t0,
            ran_forward=True,
            used_existing_txout=False,
        )

    return ForwardRunResult(
        success=True,
        code="ok",
        message="RAYINVR 正演完成",
        working_dir=str(wd),
        tx_out_path=str(tx_out),
        elapsed_s=time.time() - t0,
        ran_forward=True,
        used_existing_txout=False,
    )


def _parse_tx_file_by_shot(tx_path: str | Path) -> list[dict]:
    """
    读取 tx.in/tx.out（format(3f10.3,i10)）并按 shot 分组。

    解析方式：优先固定列宽（col 0-10, 10-20, 20-30, 30-40），
    若行太短或固定列宽失败则回退 split()。
    """
    p = Path(tx_path)
    shots: list[dict] = []
    cur: dict | None = None
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            raw = line.rstrip("\n\r")
            if not raw.strip():
                continue
            x: float | None = None
            t: float | None = None
            u: float | None = None
            ipf: int | None = None
            if len(raw) >= 40:
                try:
                    x = float(raw[0:10])
                    t = float(raw[10:20])
                    u = float(raw[20:30])
                    ipf = int(raw[30:40])
                except Exception:
                    pass
            if x is None or t is None or ipf is None:
                parts = raw.split()
                if len(parts) < 4:
                    continue
                try:
                    x = float(parts[0])
                    t = float(parts[1])
                    u = float(parts[2])
                    ipf = int(parts[3])
                except Exception:
                    continue
            if u is None:
                u = 0.0
            if ipf == -1:
                break
            if ipf <= 0:
                if cur is not None:
                    shots.append(cur)
                cur = {"shot_x": x, "obs": []}
                continue
            if cur is None:
                cur = {"shot_x": 0.0, "obs": []}
            cur["obs"].append((x, t, u, ipf))
    if cur is not None:
        shots.append(cur)
    return shots


def _phase_curve_in_shot(shot: dict, phase_id: int) -> tuple[np.ndarray, np.ndarray]:
    obs = shot.get("obs", [])
    if not obs:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x, t = [], []
    for xx, tt, _uu, ipf in obs:
        if int(ipf) == int(phase_id):
            x.append(float(xx))
            t.append(float(tt))
    if len(x) < 1:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    xa = np.asarray(x, dtype=float)
    ta = np.asarray(t, dtype=float)
    # 同x取均值，避免多条记录冲突
    keys = np.round(xa, 4)
    uniq = np.unique(keys)
    xo, to = [], []
    for k in uniq:
        m = keys == k
        xo.append(float(np.mean(xa[m])))
        to.append(float(np.mean(ta[m])))
    xoa = np.asarray(xo, dtype=float)
    toa = np.asarray(to, dtype=float)
    so = np.argsort(xoa)
    return xoa[so], toa[so]


def collect_phase_points_from_txout(
    tx_out_path: str | Path,
    *,
    phase_ids: tuple[int, ...] = (5, 14),
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    从 tx.out 收集指定相位的理论点（不要求 PPP/PPS x 重合）。

    Returns
    -------
    dict:
      phase_id -> (x_model_distance, t, shot_x)
    """
    tx_out = Path(tx_out_path)
    if not tx_out.exists():
        return {}
    shots = _parse_tx_file_by_shot(tx_out)
    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    phase_set = {int(p) for p in phase_ids}
    buckets: dict[int, list[tuple[float, float, float]]] = {p: [] for p in phase_set}
    for s in shots:
        sx = float(s.get("shot_x", np.nan))
        for xx, tt, _uu, ipf in s.get("obs", []):
            pid = int(ipf)
            if pid in phase_set and np.isfinite(xx) and np.isfinite(tt):
                buckets[pid].append((float(xx), float(tt), sx))
    for pid, rows in buckets.items():
        if not rows:
            continue
        arr = np.asarray(rows, dtype=float)
        out[pid] = (arr[:, 0], arr[:, 1], arr[:, 2])
    return out


def build_theory2d_bundle_from_txout(
    tx_out_path: str | Path,
    *,
    phase_ppp: int = 5,
    phase_pps: int = 14,
    input_hash: str = "",
) -> Theory2DBundle:
    tx_out = Path(tx_out_path)
    shots = _parse_tx_file_by_shot(tx_out)

    # 诊断：不依赖 shot 分组，直接扫描全文件收集相位点
    raw_ppp_x: list[float] = []
    raw_ppp_t: list[float] = []
    raw_pps_x: list[float] = []
    raw_pps_t: list[float] = []
    with open(tx_out, "r", encoding="utf-8", errors="ignore") as _fdiag:
        for _ln in _fdiag:
            _raw = _ln.rstrip("\n\r")
            if len(_raw) < 40:
                _pts = _raw.split()
                if len(_pts) < 4:
                    continue
                try:
                    _xv = float(_pts[0])
                    _tv = float(_pts[1])
                    _iv = int(_pts[3])
                except Exception:
                    continue
            else:
                try:
                    _xv = float(_raw[0:10])
                    _tv = float(_raw[10:20])
                    _iv = int(_raw[30:40])
                except Exception:
                    _pts = _raw.split()
                    if len(_pts) < 4:
                        continue
                    try:
                        _xv = float(_pts[0])
                        _tv = float(_pts[1])
                        _iv = int(_pts[3])
                    except Exception:
                        continue
            if _iv == phase_ppp:
                raw_ppp_x.append(_xv)
                raw_ppp_t.append(_tv)
            elif _iv == phase_pps:
                raw_pps_x.append(_xv)
                raw_pps_t.append(_tv)

    curves: list[ShotDeltaCurve] = []
    xg_list: list[np.ndarray] = []
    dg_list: list[np.ndarray] = []
    all_x_ppp: list[float] = list(raw_ppp_x)
    all_t_ppp: list[float] = list(raw_ppp_t)
    all_x_pps: list[float] = list(raw_pps_x)
    all_t_pps: list[float] = list(raw_pps_t)
    for s in shots:
        sx = float(s.get("shot_x", 0.0))
        x_ppp, t_ppp = _phase_curve_in_shot(s, phase_ppp)
        x_pps, t_pps = _phase_curve_in_shot(s, phase_pps)
        if x_ppp.size > 0:
            all_x_ppp.extend(x_ppp.tolist())
            all_t_ppp.extend(t_ppp.tolist())
        if x_pps.size > 0:
            all_x_pps.extend(x_pps.tolist())
            all_t_pps.extend(t_pps.tolist())
        if x_ppp.size < 2 or x_pps.size < 2:
            continue

        # 自适应坐标：尝试 x 与 x+shot_x 两种坐标，选重叠更大的组合
        cand_ppp = [x_ppp, x_ppp + sx]
        cand_pps = [x_pps, x_pps + sx]
        best_x1, best_x2 = x_ppp, x_pps
        best_score = -1.0
        for xc1 in cand_ppp:
            for xc2 in cand_pps:
                ov_min_t = max(float(np.min(xc1)), float(np.min(xc2)))
                ov_max_t = min(float(np.max(xc1)), float(np.max(xc2)))
                if ov_max_t <= ov_min_t:
                    continue
                score = (ov_max_t - ov_min_t)
                if score > best_score:
                    best_score = score
                    best_x1, best_x2 = xc1, xc2

        # 首选：同x直接配对（兼容原逻辑）
        k1 = {round(float(x), 4): float(t) for x, t in zip(best_x1, t_ppp)}
        k2 = {round(float(x), 4): float(t) for x, t in zip(best_x2, t_pps)}
        keys = sorted(set(k1) & set(k2))
        if len(keys) >= 2:
            xs = np.asarray([float(k) for k in keys], dtype=float)
            dt = np.asarray([k2[k] - k1[k] for k in keys], dtype=float)
        else:
            # 回退：在重叠区间内插值配对（避免“有相位但x不完全重合”被误判为空）
            ov_min = max(float(np.min(best_x1)), float(np.min(best_x2)))
            ov_max = min(float(np.max(best_x1)), float(np.max(best_x2)))
            if ov_max <= ov_min:
                continue
            xu = np.unique(np.concatenate([best_x1, best_x2]))
            xm = xu[(xu >= ov_min) & (xu <= ov_max)]
            if xm.size < 2:
                continue
            tppp_i = np.interp(xm, best_x1, t_ppp)
            tpps_i = np.interp(xm, best_x2, t_pps)
            xs = xm
            dt = tpps_i - tppp_i

        if xs.size < 2:
            continue
        curves.append(ShotDeltaCurve(shot_x=float(s["shot_x"]), x=xs, dt=dt))
        xg_list.append(xs)
        dg_list.append(dt)

    if xg_list:
        xg = np.concatenate(xg_list)
        dg = np.concatenate(dg_list)
        so = np.argsort(xg)
        xg, dg = xg[so], dg[so]
    else:
        # 分炮都无法配对时，回退到“全局相位点”构建Delta_t（不要求同炮）
        phase_pts = collect_phase_points_from_txout(tx_out, phase_ids=(phase_ppp, phase_pps))
        x_ppp_g, t_ppp_g, s_ppp_g = phase_pts.get(
            int(phase_ppp),
            (np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)),
        )
        x_pps_g, t_pps_g, s_pps_g = phase_pts.get(
            int(phase_pps),
            (np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)),
        )
        if x_ppp_g.size >= 2 and x_pps_g.size >= 2:
            # 全局也做一次坐标自适应：raw vs (x+shot_x)
            cand_ppp = [x_ppp_g, x_ppp_g + np.nan_to_num(s_ppp_g, nan=0.0)]
            cand_pps = [x_pps_g, x_pps_g + np.nan_to_num(s_pps_g, nan=0.0)]
            best_x1, best_x2 = x_ppp_g, x_pps_g
            best_score = -1.0
            for xc1 in cand_ppp:
                for xc2 in cand_pps:
                    ov_min_t = max(float(np.min(xc1)), float(np.min(xc2)))
                    ov_max_t = min(float(np.max(xc1)), float(np.max(xc2)))
                    if ov_max_t <= ov_min_t:
                        continue
                    score = (ov_max_t - ov_min_t)
                    if score > best_score:
                        best_score = score
                        best_x1, best_x2 = xc1, xc2

            so1 = np.argsort(x_ppp_g)
            so2 = np.argsort(x_pps_g)
            x_ppp_g, t_ppp_g = best_x1[so1], t_ppp_g[so1]
            x_pps_g, t_pps_g = best_x2[so2], t_pps_g[so2]
            ov_min = max(float(np.min(x_ppp_g)), float(np.min(x_pps_g)))
            ov_max = min(float(np.max(x_ppp_g)), float(np.max(x_pps_g)))
            if ov_max > ov_min:
                xu = np.unique(np.concatenate([x_ppp_g, x_pps_g]))
                xm = xu[(xu >= ov_min) & (xu <= ov_max)]
                if xm.size >= 2:
                    tppp_i = np.interp(xm, x_ppp_g, t_ppp_g)
                    tpps_i = np.interp(xm, x_pps_g, t_pps_g)
                    xg = xm
                    dg = tpps_i - tppp_i
                else:
                    xg = np.asarray([], dtype=float)
                    dg = np.asarray([], dtype=float)
            else:
                xg = np.asarray([], dtype=float)
                dg = np.asarray([], dtype=float)
        else:
            xg = np.asarray([], dtype=float)
            dg = np.asarray([], dtype=float)

    # 诊断信息附在 input_hash 末尾（可选），方便调试
    diag = (
        f"|shots={len(shots)}"
        f"|raw_ppp={len(raw_ppp_x)}"
        f"|raw_pps={len(raw_pps_x)}"
        f"|curves={len(curves)}"
        f"|global_n={int(xg.size) if hasattr(xg, 'size') else 0}"
    )
    return Theory2DBundle(
        working_dir=str(tx_out.parent),
        tx_out_path=str(tx_out),
        phase_ppp=int(phase_ppp),
        phase_pps=int(phase_pps),
        input_hash=str(input_hash) + diag,
        curves=curves,
        global_x=xg,
        global_dt=dg,
    )


def collect_theory2d_delta_by_branch(
    tx_out_path: str | Path,
    *,
    phase_ppp: int = 5,
    phase_pps: int = 14,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    从 tx.out 按偏移距正负拆成左支(offset<0)、右支(offset>0)的理论 PPS-PPP 曲线。
    用于 2D 双支 pois 编辑：编辑左支后只更新左支结果，编辑右支后只更新右支结果。

    Returns
    -------
    ((left_x, left_dt), (right_x, right_dt))
      左支/右支的 (model_x, delta_t) 数组，可用于 np.interp 查询。
    """
    tx_out = Path(tx_out_path)
    if not tx_out.exists():
        return (np.asarray([]), np.asarray([])), (np.asarray([]), np.asarray([]))
    shots = _parse_tx_file_by_shot(tx_out)
    left_x_list: list[float] = []
    left_dt_list: list[float] = []
    right_x_list: list[float] = []
    right_dt_list: list[float] = []

    for s in shots:
        sx = float(s.get("shot_x", 0.0))
        x_ppp, t_ppp = _phase_curve_in_shot(s, phase_ppp)
        x_pps, t_pps = _phase_curve_in_shot(s, phase_pps)
        if x_ppp.size < 2 or x_pps.size < 2:
            continue
        ov_min = max(float(np.min(x_ppp)), float(np.min(x_pps)))
        ov_max = min(float(np.max(x_ppp)), float(np.max(x_pps)))
        if ov_max <= ov_min:
            continue
        x_common = np.unique(np.concatenate([x_ppp, x_pps]))
        x_common = x_common[(x_common >= ov_min) & (x_common <= ov_max)]
        if x_common.size < 2:
            continue
        t_ppp_i = np.interp(x_common, x_ppp, t_ppp)
        t_pps_i = np.interp(x_common, x_pps, t_pps)
        dt = t_pps_i - t_ppp_i
        offset = x_common - sx
        left_mask = offset < 0
        right_mask = offset > 0
        if np.any(left_mask):
            left_x_list.extend(x_common[left_mask].tolist())
            left_dt_list.extend(dt[left_mask].tolist())
        if np.any(right_mask):
            right_x_list.extend(x_common[right_mask].tolist())
            right_dt_list.extend(dt[right_mask].tolist())

    left_x = np.asarray(left_x_list, dtype=float)
    left_dt = np.asarray(left_dt_list, dtype=float)
    right_x = np.asarray(right_x_list, dtype=float)
    right_dt = np.asarray(right_dt_list, dtype=float)
    if left_x.size > 1:
        so = np.argsort(left_x)
        left_x, left_dt = left_x[so], left_dt[so]
    if right_x.size > 1:
        so = np.argsort(right_x)
        right_x, right_dt = right_x[so], right_dt[so]
    return (left_x, left_dt), (right_x, right_dt)


def _interp_with_coverage(
    x_src: np.ndarray,
    y_src: np.ndarray,
    xq: np.ndarray,
    *,
    allow_extrapolation: bool,
) -> np.ndarray:
    out = np.full_like(xq, np.nan, dtype=float)
    if x_src.size < 2:
        return out
    xmn = float(np.min(x_src))
    xmx = float(np.max(x_src))
    if allow_extrapolation:
        out[:] = np.interp(xq, x_src, y_src, left=y_src[0], right=y_src[-1])
        return out
    m = (xq >= xmn) & (xq <= xmx)
    if np.any(m):
        out[m] = np.interp(xq[m], x_src, y_src)
    return out


def make_delta_query(
    bundle: Theory2DBundle,
    *,
    allow_extrapolation: bool = False,
    shot_tolerance_km: float = 0.05,
) -> Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]:
    """
    返回 shot-aware 查询函数：
      delta(x_query, shot_x_query=None) -> dt_query
    """

    def _query(x_query: np.ndarray, shot_x_query: Optional[np.ndarray] = None) -> np.ndarray:
        xq = np.asarray(x_query, dtype=float)
        out = np.full_like(xq, np.nan, dtype=float)
        if xq.size == 0:
            return out
        if shot_x_query is None or len(bundle.curves) == 0:
            return _interp_with_coverage(
                bundle.global_x, bundle.global_dt, xq, allow_extrapolation=allow_extrapolation
            )

        sq = np.asarray(shot_x_query, dtype=float)
        if sq.shape != xq.shape:
            return _interp_with_coverage(
                bundle.global_x, bundle.global_dt, xq, allow_extrapolation=allow_extrapolation
            )

        shot_vals = np.asarray([c.shot_x for c in bundle.curves], dtype=float)
        for i in range(xq.size):
            if not np.isfinite(xq[i]) or not np.isfinite(sq[i]):
                continue
            j = int(np.argmin(np.abs(shot_vals - sq[i])))
            if abs(float(shot_vals[j] - sq[i])) > shot_tolerance_km:
                # shot 不匹配时回退到全局曲线
                out[i] = _interp_with_coverage(
                    bundle.global_x,
                    bundle.global_dt,
                    np.asarray([xq[i]], dtype=float),
                    allow_extrapolation=allow_extrapolation,
                )[0]
                continue
            c = bundle.curves[j]
            out[i] = _interp_with_coverage(
                c.x,
                c.dt,
                np.asarray([xq[i]], dtype=float),
                allow_extrapolation=allow_extrapolation,
            )[0]
        return out

    return _query

