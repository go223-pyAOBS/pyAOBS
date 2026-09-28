"""绘制 smesh 的共享解析逻辑（无 UI）。"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

from ..state.form_state import FormState

_WSL_UNC_RE = re.compile(
    r"^[/\\]{2}wsl(?:\.localhost|\$)[/\\][^/\\]+[/\\](.*)$",
    re.IGNORECASE,
)
_MNT_DRIVE_RE = re.compile(r"^/mnt/([a-zA-Z])/(.*)$")


def running_in_wsl() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def normalize_dropped_path(path: str | Path) -> Path:
    """把资源管理器 / WSLg 拖来的路径收成当前 Python 能打开的 Path。"""
    s = str(path).strip().strip('"').replace("\x00", "").strip()
    if not s:
        return Path(s)
    if s.lower().startswith("file:"):
        u = urlparse(s)
        host = (u.hostname or "").lower()
        body = unquote(u.path or "")
        if host in ("wsl.localhost", "wsl$") or host.startswith("wsl"):
            parts = [p for p in body.split("/") if p]
            if len(parts) >= 2:
                rest = "/" + "/".join(parts[1:])
                if sys.platform.startswith("linux"):
                    s = rest
                else:
                    s = r"\\wsl.localhost\{}\{}".format(
                        parts[0], rest.lstrip("/").replace("/", "\\")
                    )
            else:
                s = body
        else:
            s = (
                unquote(u.netloc + body)
                if u.netloc and u.netloc.lower() not in ("localhost",)
                else body
            )
            if s.startswith("/") and len(s) >= 3 and s[2] == ":":
                s = s[1:]
    s = unquote(s)
    slash = s.replace("\\", "/")

    if sys.platform.startswith("linux"):
        if len(s) >= 2 and s[1] == ":":
            drive = s[0].lower()
            rest = s[2:].replace("\\", "/").lstrip("/")
            return Path(f"/mnt/{drive}/{rest}")
        m = _WSL_UNC_RE.match(slash)
        if m:
            return Path("/" + m.group(1).replace("\\", "/"))
        if slash.startswith("//wsl.localhost/") or slash.startswith("//wsl$/"):
            parts = [p for p in slash.split("/") if p]
            if len(parts) >= 2:
                return Path("/" + "/".join(parts[1:]))
        return Path(s)

    m = _MNT_DRIVE_RE.match(slash)
    if m:
        return Path(f"{m.group(1).upper()}:/{m.group(2)}")
    return Path(s)


def parse_clipboard_path_text(text: str) -> list[Path]:
    """解析「复制为路径」或多行路径文本。"""
    if not text:
        return []
    out: list[Path] = []
    seen: set[str] = set()
    for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        t = line.strip().strip("\ufeff").strip('"').strip("'").strip()
        if not t or t.startswith("#"):
            continue
        p = normalize_dropped_path(t)
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def read_windows_explorer_clipboard_paths() -> list[Path]:
    """WSL 里读资源管理器 Ctrl+C 的文件列表（Linux Qt 拿不到 CF_HDROP）。"""
    if not running_in_wsl():
        return []
    try:
        import subprocess

        cmd = (
            "$OutputEncoding = [Console]::OutputEncoding = "
            "New-Object System.Text.UTF8Encoding $false; "
            "$f = Get-Clipboard -Format FileDropList -ErrorAction SilentlyContinue; "
            "if ($f) { $f | ForEach-Object { $_.FullName } } "
            "else { Get-Clipboard }"
        )
        r = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", cmd],
            capture_output=True,
            timeout=5,
            check=False,
        )
        raw = r.stdout or b""
        text = raw.decode("utf-8", errors="replace")
        if "\x00" in text:
            text = raw.decode("utf-16le", errors="replace")
        return parse_clipboard_path_text(text)
    except Exception:
        return []


def looks_like_smesh_name(path: str | Path) -> bool:
    """只看文件名（拖入时路径可能尚不能 is_file）。"""
    return ".smesh" in Path(path).name.lower()


def looks_like_grid_name(path: str | Path) -> bool:
    """GMT / NetCDF 速度网格：``.grd`` ``.nc`` ``.nc4``。"""
    return Path(path).suffix.lower() in {".grd", ".nc", ".nc4"}


def looks_like_vin_name(path: str | Path) -> bool:
    """Zelt ``v.in`` / ``*.vin`` / 非标准 RAYINVR 的 ``*.in``。"""
    from pyAOBS.modeling.rayinvr.vin_io import is_vin_path

    return is_vin_path(path)


def looks_like_model_name(path: str | Path) -> bool:
    """绘制模型：smesh、v.in、grd/nc。"""
    return (
        looks_like_smesh_name(path)
        or looks_like_grid_name(path)
        or looks_like_vin_name(path)
    )


def pick_smesh_paths(paths: list[Path]) -> list[Path]:
    """从路径列表里抽出 smesh（保持原顺序、去重）。"""
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        if not looks_like_model_name(p):
            continue
        try:
            key = str(p.resolve()) if p.exists() else str(p)
        except OSError:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def looks_like_interface_name(path: str | Path) -> bool:
    n = Path(path).name.lower()
    if looks_like_model_name(n):
        return False
    if ".refl" in n or n.startswith("refl") or "_refl" in n:
        return True
    stem = Path(path).stem.lower()
    return any(
        k in stem
        for k in ("bathy", "bathymetry", "basement", "moho", "interface", "conv")
    )


def looks_like_smesh(path: str | Path) -> bool:
    """文件名含 ``.smesh``（含 ``out.smesh.1.1``）。"""
    p = Path(path)
    if not looks_like_smesh_name(p):
        return False
    try:
        return p.is_file()
    except OSError:
        return False


def looks_like_interface(path: str | Path) -> bool:
    """反射/界面文本：``*.refl.*``、文件名含 refl，或常见 bathy/moho 等。"""
    p = Path(path)
    try:
        if not p.is_file() or looks_like_smesh(p):
            return False
    except OSError:
        return False
    return looks_like_interface_name(p)


_IFACE_STYLE = (
    ("#dc143c", 1.8, "--"),
    ("#1d4ed8", 1.6, "-"),
    ("#7c3aed", 1.6, "--"),
    ("#0f766e", 1.6, "-"),
)


def load_interface_overlays(paths: list[str | Path] | None) -> list[dict]:
    """读取若干 ``x z`` 界面文件为绘图 overlays。"""
    if not paths:
        return []
    from pyAOBS.model_building.tomoform import load_tomo2d_interface_file

    out: list[dict] = []
    for i, raw in enumerate(paths):
        p = Path(raw)
        rx, rz = load_tomo2d_interface_file(str(p))
        color, lw, ls = _IFACE_STYLE[i % len(_IFACE_STYLE)]
        out.append(
            {
                "x": rx,
                "z": rz,
                "label": p.name,
                "color": color,
                "linewidth": lw,
                "linestyle": ls,
            }
        )
    return out


SMESH_CMAP_KEY = "gui.plot_smesh_cmap"
DEFAULT_SMESH_CMAP_ID = "vp"
# 下拉显示顺序：vpvs / vs / vp / water
BUILTIN_SMESH_CMAPS: tuple[tuple[str, str], ...] = (
    ("vpvs", "scale_vpvs.cpt"),
    ("vs", "scale_s.cpt"),
    ("vp", "scale_p.cpt"),
    ("water", "scale_water.cpt"),
)


def builtin_smesh_cmap_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "assets"


def list_builtin_smesh_cmap_ids() -> tuple[str, ...]:
    return tuple(cid for cid, _fn in BUILTIN_SMESH_CMAPS)


def builtin_smesh_cmap_path(cmap_id: str) -> Path:
    """内置 CPT 路径；未知 id 回退 ``vp``。"""
    want = (cmap_id or DEFAULT_SMESH_CMAP_ID).strip().lower()
    for cid, fn in BUILTIN_SMESH_CMAPS:
        if cid == want:
            return builtin_smesh_cmap_dir() / fn
    return builtin_smesh_cmap_dir() / "scale_p.cpt"


def builtin_scale_p_cpt_path() -> Path:
    """GUI 内置 Vp 色标（与 ``examples/inputs/scale_p.cpt`` 相同）。"""
    return builtin_smesh_cmap_path("vp")


def builtin_sigma_cpt_path() -> Path:
    """误差 σ 色标：Haiti ``develf.cpt``（0–0.40 km/s 分段）。"""
    return builtin_smesh_cmap_dir() / "develf.cpt"


def alias_builtin_smesh_cmap_id(raw: str) -> str | None:
    """把表单值认成 ``vp`` / ``vs`` / ``vpvs`` / ``water``；认不出则返回 None。"""
    s = (raw or "").strip()
    if not s:
        return DEFAULT_SMESH_CMAP_ID
    key = s.lower().replace("\\", "/").split("/")[-1]
    aliases = {
        "vp": "vp",
        "scale_p": "vp",
        "scale_p.cpt": "vp",
        "vs": "vs",
        "scale_s": "vs",
        "scale_s.cpt": "vs",
        "vpvs": "vpvs",
        "scale_vpvs": "vpvs",
        "scale_vpvs.cpt": "vpvs",
        "vpvs1": "vpvs",
        "vpvs1.cpt": "vpvs",
        "vpvs2": "vpvs",
        "vpvs2.cpt": "vpvs",
        "water": "water",
        "scale_water": "water",
        "scale_water.cpt": "water",
    }
    hit = aliases.get(key) or aliases.get(s.lower())
    if hit:
        return hit
    return None


def get_smesh_cmap_id(state: FormState | None) -> str:
    """下拉当前项；无法识别的旧值显示为 ``vp``。"""
    if state is None:
        return DEFAULT_SMESH_CMAP_ID
    raw = state.get_str(SMESH_CMAP_KEY) if hasattr(state, "get_str") else ""
    return alias_builtin_smesh_cmap_id(raw) or DEFAULT_SMESH_CMAP_ID


def set_smesh_cmap_id(state: FormState | None, cmap_id: str) -> None:
    if state is not None and hasattr(state, "set"):
        want = (cmap_id or DEFAULT_SMESH_CMAP_ID).strip().lower()
        if want not in list_builtin_smesh_cmap_ids():
            want = DEFAULT_SMESH_CMAP_ID
        state.set(SMESH_CMAP_KEY, want)


def colorbar_label_for_cmap(cmap_spec: str) -> str:
    """vp / vs / water 用 km/s，vpvs 无量纲。"""
    raw = str(cmap_spec or "").strip()
    cid = alias_builtin_smesh_cmap_id(raw) or alias_builtin_smesh_cmap_id(Path(raw).name)
    if cid == "vpvs":
        return "Vp/Vs"
    return "km/s"


def mask_air_layer_for_plot(
    data,
    x,
    z,
    mesh=None,
    *,
    air_vmax: float = 1.2,
):
    """海面以上（z < topo 或 z < 0）及空气速度设为 NaN，imshow 不着色。

    ``data`` 须为 ``velocity(z, x)``。
    """
    import numpy as np

    out = np.array(data, dtype=float, copy=True)
    zz = np.asarray(z, dtype=float).reshape(-1)
    xx = np.asarray(x, dtype=float).reshape(-1)
    if out.ndim != 2 or out.shape[0] != zz.size or out.shape[1] != xx.size:
        air = np.isfinite(out) & (out < float(air_vmax))
        out[air] = np.nan
        return out
    air = zz[:, None] < -1e-8
    if mesh is not None and getattr(mesh, "xpos", None) is not None:
        topo = np.interp(
            xx,
            np.asarray(mesh.xpos, dtype=float),
            np.asarray(mesh.topo, dtype=float),
        )
        air = air | (zz[:, None] < (topo[None, :] - 1e-8))
    # 空气速度只抹海面以上；海面节点（含 arange 的 -2e-16）保留水速
    air = air | ((zz[:, None] < -1e-8) & np.isfinite(out) & (out < float(air_vmax)))
    out[air] = np.nan
    return out


def cmap_blank_air(cmap):
    """NaN / mask 处全透明，露出坐标轴底色（空气无色）。"""
    try:
        return cmap.with_extremes(bad=(1.0, 1.0, 1.0, 0.0))
    except Exception:
        pass
    cm = cmap.copy() if hasattr(cmap, "copy") else cmap
    try:
        cm.set_bad(color=(1.0, 1.0, 1.0, 0.0))
    except Exception:
        pass
    return cm


def air_axis_zlim(zmin: float, zmax: float) -> tuple[float, float]:
    """深度向下的 ylim，海面以上留一条空白（空气无色）。"""
    return (float(zmax), min(float(zmin), 0.0) - 0.4)


def default_plot_smesh_cmap() -> str:
    """绘制模型的默认色标：内置 ``vp``（scale_p.cpt），缺失时退回 seismic。"""
    p = builtin_smesh_cmap_path(DEFAULT_SMESH_CMAP_ID)
    if p.is_file():
        return str(p)
    return "seismic"


def resolve_plot_smesh_cmap(
    state: FormState,
    work: Path,
    *,
    on_missing_cpt: Callable[[str], None] | None = None,
) -> str:
    """返回 matplotlib 色标名，或存在的 .cpt 绝对路径。

    表单 ``gui.plot_smesh_cmap`` 优先认 ``vp`` / ``vs`` / ``vpvs`` / ``water``
    （及同名内置文件）；空则 ``vp``。仍兼容外部 ``.cpt`` 路径与 matplotlib 色标名。
    """
    fallback = default_plot_smesh_cmap()
    s = (state.get_str(SMESH_CMAP_KEY) or "").strip()
    builtin_id = alias_builtin_smesh_cmap_id(s)
    if builtin_id is not None:
        p = builtin_smesh_cmap_path(builtin_id)
        if p.is_file():
            return str(p)
        return fallback
    p = Path(s)
    if p.suffix.lower() == ".cpt":
        full = p if p.is_absolute() else (work / p)
        try:
            full = full.resolve()
        except OSError:
            full = p
        if full.is_file():
            return str(full)
        msg = f"找不到 CPT 文件，已改用内置 vp 色标：\n{full}"
        if on_missing_cpt is not None:
            on_missing_cpt(msg)
        return fallback
    return s


def resolve_optional_refl_path(
    state: FormState,
    work: Path,
    keys: tuple[str, ...] | None = None,
) -> str | None:
    """表单里已填且存在的 refl；默认看正演/反演页（监视、挑选用）。"""
    for key in keys or (
        "fwd.refl_file",
        "inv.refl_file",
        "fwd.seafloor_file",
        "inv.seafloor_file",
    ):
        s = state.get_str(key)
        if not s:
            continue
        p = Path(s)
        full = p if p.is_absolute() else (work / p)
        if full.is_file():
            return str(full)
    return None


def resolve_plot_refl_for_smesh(
    smesh_path: str | Path,
    state: FormState,
    work: Path,
    *,
    refl_keys: tuple[str, ...] | None = None,
) -> str | None:
    """优先用同轮反演写出的 ``*.refl.<iter>.<iset>``，否则退回给定/默认 -F。"""
    from .smesh_ops import companion_inverse_refl

    hit = companion_inverse_refl(smesh_path)
    if hit is not None:
        return str(hit)
    return resolve_optional_refl_path(state, work, keys=refl_keys)


def _first_existing_smesh(state: FormState, work: Path, keys: tuple[str, ...]) -> Path | None:
    for key in keys:
        guess = state.get_str(key)
        if not guess:
            continue
        gp = Path(guess)
        cand = gp if gp.is_absolute() else (work / gp)
        if cand.is_file():
            return cand
    return None


def _form_file_lookup(
    state: FormState, work: Path, key: str | None
) -> tuple[str, Path | None, Path | None]:
    """返回 ``(表单原文, 已存在路径, 已填但不存在的路径)``。"""
    if not key:
        return "", None, None
    raw = state.get_str(key)
    if not raw:
        return "", None, None
    p = Path(raw)
    cand = p if p.is_absolute() else (work / p)
    try:
        ok = cand.is_file()
    except OSError:
        ok = False
    if ok:
        return raw, cand, None
    return raw, None, cand


@dataclass(frozen=True)
class CmdTabPlotSource:
    """与主窗命令页签顺序一致：该页用于绘制 smesh / 界面的表单字段。"""

    tab_id: str
    nav_name: str
    smesh_key: str | None = None
    smesh_label: str = ""
    refl_key: str | None = None
    refl_label: str = ""


# 顺序必须与 main_window 左侧命令列表 / stack 一致
CMD_TAB_PLOT_SOURCES: tuple[CmdTabPlotSource, ...] = (
    CmdTabPlotSource(
        "gen_smesh",
        "gen_smesh",
        "gen.smesh_out",
        "smesh 输出文件",
        "gen.refl_file",
        "refl_file (-F 输出)",
    ),
    CmdTabPlotSource(
        "tt_forward",
        "tt_forward",
        "fwd.smesh",
        "smesh (-M)",
        "fwd.refl_file",
        "refl_file (-F)",
    ),
    CmdTabPlotSource("gen_damp", "gen_damp"),
    CmdTabPlotSource("gen_vcorr", "gen_vcorr"),
    CmdTabPlotSource("gen_dcorr", "gen_dcorr"),
    CmdTabPlotSource(
        "tt_inverse",
        "tt_inverse",
        "inv.mesh",
        "mesh (-M)",
        "inv.refl_file",
        "refl_file (-F)",
    ),
    CmdTabPlotSource(
        "stat_smesh",
        "stat_smesh",
        "stat.mesh_file",
        "mesh_file (-M)",
    ),
    CmdTabPlotSource(
        "edit_smesh",
        "edit_smesh_HHB",
        "edit.smesh_file",
        "smesh_file",
    ),
    CmdTabPlotSource(
        "pipeline",
        "pipeline",
        "pipe.link_smesh",
        "link_smesh (桥接)",
    ),
    CmdTabPlotSource("tx_convert", "tx.in→tomo2d"),
    CmdTabPlotSource(
        "checkerboard",
        "棋盘格测试",
        "cb.bg_smesh",
        "背景 smesh",
        "cb.refl_file",
        "反射面（有同轮则自动填）",
    ),
    CmdTabPlotSource(
        "monte_carlo",
        "蒙特卡洛",
        "mc.base_mesh",
        "初始/背景 mesh（-M）",
    ),
    CmdTabPlotSource(
        "wave2d",
        "wave2d",
        "wave.vp_smesh",
        "Vp smesh",
        "wave.seafloor",
        "海底 seafloor",
    ),
)


def plot_source_for_tab(tab_id: str | None) -> CmdTabPlotSource | None:
    if not tab_id:
        return None
    for src in CMD_TAB_PLOT_SOURCES:
        if src.tab_id == tab_id:
            return src
    return None


@dataclass
class PlotSmeshLookup:
    """当前命令页签上的 smesh / 界面解析结果（不跨页签回退）。"""

    tab_id: str
    nav_name: str
    smesh_key: str | None
    smesh_label: str
    smesh_raw: str
    smesh_path: Path | None
    smesh_missing: Path | None
    refl_key: str | None
    refl_label: str
    refl_raw: str
    refl_path: Path | None
    refl_missing: Path | None
    browse_start: Path | None

    def missing_smesh_message(self) -> str:
        if self.smesh_key is None:
            return (
                f"当前页签「{self.nav_name}」没有 smesh 字段。\n"
                "请指定要绘制的文件，或先切换到 gen_smesh / tt_forward / "
                "tt_inverse 等含网格路径的页签。"
            )
        if self.smesh_raw and self.smesh_missing is not None:
            return (
                f"当前页签「{self.nav_name}」的「{self.smesh_label}」已填，"
                f"但找不到文件：\n{self.smesh_missing}\n"
                "请指定一个 smesh 文件。"
            )
        return (
            f"当前页签「{self.nav_name}」未填写「{self.smesh_label}」。\n"
            "请指定要绘制的 smesh 文件。"
        )

    def missing_refl_message(self) -> str | None:
        if not self.refl_key or not self.refl_raw or self.refl_path is not None:
            return None
        shown = self.refl_missing if self.refl_missing is not None else self.refl_raw
        return (
            f"当前页签「{self.nav_name}」的「{self.refl_label}」已填，"
            f"但找不到文件：\n{shown}\n"
            "请在图窗点「叠加界面…」指定。"
        )


def lookup_plot_sources_for_tab(
    state: FormState,
    work: Path,
    tab_id: str | None,
) -> PlotSmeshLookup:
    """只看当前页签字段；不回退到其它页的 model.smesh / refl。"""
    src = plot_source_for_tab(tab_id)
    if src is None:
        name = (tab_id or "未知页签").strip() or "未知页签"
        return PlotSmeshLookup(
            tab_id=name,
            nav_name=name,
            smesh_key=None,
            smesh_label="",
            smesh_raw="",
            smesh_path=None,
            smesh_missing=None,
            refl_key=None,
            refl_label="",
            refl_raw="",
            refl_path=None,
            refl_missing=None,
            browse_start=work,
        )
    smesh_raw, smesh_path, smesh_missing = _form_file_lookup(
        state, work, src.smesh_key
    )
    refl_raw, refl_path, refl_missing = _form_file_lookup(
        state, work, src.refl_key
    )
    if smesh_path is not None:
        from .smesh_ops import companion_inverse_refl

        companion = companion_inverse_refl(smesh_path)
        if companion is not None:
            refl_path = companion
            refl_missing = None
    browse = work
    for cand in (smesh_path, smesh_missing, refl_path, refl_missing):
        if cand is not None:
            browse = cand.parent
            break
    return PlotSmeshLookup(
        tab_id=src.tab_id,
        nav_name=src.nav_name,
        smesh_key=src.smesh_key,
        smesh_label=src.smesh_label,
        smesh_raw=smesh_raw,
        smesh_path=smesh_path,
        smesh_missing=smesh_missing,
        refl_key=src.refl_key,
        refl_label=src.refl_label,
        refl_raw=refl_raw,
        refl_path=refl_path,
        refl_missing=refl_missing,
        browse_start=browse,
    )


def guess_smesh_initial_path(
    state: FormState, work: Path, *, tab_id: str | None = None
) -> Path | None:
    """当前命令页签上已存在的 smesh；未给 ``tab_id`` 时不跨页签猜测。"""
    if not tab_id:
        return None
    return lookup_plot_sources_for_tab(state, work, tab_id).smesh_path


def resolve_inv_start_smesh(state: FormState, work: Path) -> Path | None:
    """反演起始网格：优先 ``inv.mesh``（-M），否则正演/生成网格。"""
    return _first_existing_smesh(
        state, work, ("inv.mesh", "fwd.smesh", "gen.smesh_out")
    )


def load_smesh_plot_data(
    smesh_path: str | Path,
    refl_path: str | None,
    *,
    with_xarray: bool = True,
) -> tuple[Any, Any, list | None]:
    """返回 (mesh, xarray Dataset|None, extra_interfaces|None)。"""
    try:
        from pyAOBS.model_building.tomoform import (
            SlownessMesh2D,
            load_tomo2d_interface_file,
        )
    except ImportError:  # pragma: no cover
        from pyAOBS.model_building.tomoform import (  # type: ignore
            SlownessMesh2D,
            load_tomo2d_interface_file,
        )

    mesh = SlownessMesh2D.from_file(str(smesh_path))
    ds = mesh.to_xarray() if with_xarray else None
    extra: list | None = None
    if refl_path:
        rx, rz = load_tomo2d_interface_file(refl_path)
        extra = [
            {
                "x": rx,
                "z": rz,
                "label": Path(refl_path).name,
                "color": "crimson",
                "linewidth": 1.8,
                "linestyle": "--",
            }
        ]
    return mesh, ds, extra


def normalize_velocity_plot_dataset(ds: Any):
    """把任意速度 Dataset 收成 ``velocity(z, x)``，供 imshow 用。"""
    import numpy as np
    import xarray as xr

    data_name = None
    for n in ("velocity", "vel", "vp", "vs", "vpvs", "v"):
        if n in ds.data_vars:
            data_name = n
            break
    if data_name is None:
        for n, da in ds.data_vars.items():
            if int(getattr(da, "ndim", 0)) == 2:
                data_name = n
                break
    if data_name is None:
        raise ValueError("网格没有二维速度变量（velocity / vp / vs / vpvs）")
    da = ds[data_name]
    dims = list(da.dims)
    coords = list(ds.coords) + dims
    x_name = next((c for c in ("x", "lon", "longitude") if c in coords), None)
    z_name = next((c for c in ("z", "y", "depth", "lat", "latitude") if c in coords), None)
    if (x_name is None or z_name is None) and len(dims) == 2:
        z_name, x_name = dims[0], dims[1]
    if x_name is None or z_name is None:
        raise ValueError(f"无法识别 x/z 坐标，现有: {list(ds.coords)} dims={dims}")
    if set(da.dims) == {z_name, x_name}:
        da = da.transpose(z_name, x_name)
    xv = np.asarray(
        ds[x_name].values if x_name in ds.coords else da[x_name].values,
        dtype=float,
    )
    zv = np.asarray(
        ds[z_name].values if z_name in ds.coords else da[z_name].values,
        dtype=float,
    )
    return xr.Dataset(
        data_vars={"velocity": (("z", "x"), np.asarray(da.values, dtype=float))},
        coords={"x": xv, "z": zv},
    )


def _zelt_layer_overlays(zelt) -> list[dict]:
    extra: list[dict] = []
    n = len(getattr(zelt, "depth_nodes", []) or [])
    for i in range(n):
        try:
            rx, rz = zelt.get_layer_geometry(i)
        except Exception:
            continue
        color, lw, ls = _IFACE_STYLE[i % len(_IFACE_STYLE)]
        extra.append(
            {
                "x": rx,
                "z": rz,
                "label": f"layer {i + 1}",
                "color": color,
                "linewidth": lw,
                "linestyle": ls,
            }
        )
    return extra


def load_model_plot_data(
    path: str | Path,
    refl_path: str | None = None,
    *,
    with_xarray: bool = True,
) -> tuple[Any, Any, list | None]:
    """绘制用：smesh / v.in / .grd|.nc → ``(mesh|None, Dataset, overlays)``。"""
    p = Path(path)
    if looks_like_smesh_name(p):
        return load_smesh_plot_data(p, refl_path, with_xarray=with_xarray)
    extra: list[dict] = []
    mesh = None
    from pyAOBS.modeling.rayinvr.vin_io import is_vin_file, load_zelt_model

    if looks_like_vin_name(p) or is_vin_file(p):
        zelt = load_zelt_model(p)
        ds = normalize_velocity_plot_dataset(
            zelt.to_xarray(dx=2.0, dz=0.5)
        ) if with_xarray else None
        extra.extend(_zelt_layer_overlays(zelt))
    elif looks_like_grid_name(p):
        from pyAOBS.visualization.xarray_nc import open_netcdf_like_dataset

        raw = open_netcdf_like_dataset(str(p))
        try:
            names = {str(c).lower() for c in list(raw.coords) + list(raw.dims)}
            if {"lon", "longitude"} & names and {"lat", "latitude"} & names:
                raise ValueError(
                    f"{p.name} 是经纬度平面格网（lon/lat），"
                    "绘制模型需要剖面速度网格（x–z，单位 km）"
                )
            ds = normalize_velocity_plot_dataset(raw) if with_xarray else None
        finally:
            try:
                raw.close()
            except Exception:
                pass
    else:
        raise ValueError(
            f"不支持的模型格式：{p.name}（请打开 smesh、v.in 或 .grd/.nc）"
        )
    if refl_path:
        extra.extend(load_interface_overlays([refl_path]))
    return mesh, ds, extra or None
