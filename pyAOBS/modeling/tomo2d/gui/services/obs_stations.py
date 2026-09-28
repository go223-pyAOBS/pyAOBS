"""station.lis：OBS 号 ↔ 模型距离，供监视标注与台站标记。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..state.form_state import FormState


def _as_work_file(raw: str, work: Path) -> Path | None:
    s = (raw or "").strip()
    if not s:
        return None
    p = Path(s).expanduser()
    full = p if p.is_absolute() else (work / p)
    try:
        if full.is_file():
            return full.resolve()
    except OSError:
        return None
    return None


def parse_station_lis(path: Path | str) -> list[tuple[int, float, float]]:
    """返回 ``[(obs_id, x, z), ...]``。"""
    from pyAOBS.modeling.tomo2d.tx2tomo2d import read_station_lis

    return read_station_lis(path)


def list_tomo2d_sources(path: Path | str) -> list[tuple[int, float, float]]:
    """
    读 ``ttimes.dat`` / ``geom.dat`` 的 s 行，按文件顺序编号。

    返回 ``[(isrc, x, z), ...]``，``isrc`` 从 1 起，与 ``*.ray.*.<isrc>`` / ``*.tres`` 一致。
    """
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = [ln.strip() for ln in raw.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    if not lines:
        return []
    try:
        nsrc = int(float(lines[0].split()[0]))
    except (TypeError, ValueError, IndexError):
        return []
    out: list[tuple[int, float, float]] = []
    idx = 1
    for k in range(nsrc):
        if idx >= len(lines):
            break
        ln = lines[idx]
        idx += 1
        parsed = _parse_s_line(ln)
        if parsed is None:
            break
        sx, sz, nrcv = parsed
        out.append((k + 1, sx, sz))
        idx += max(0, nrcv)
    return out


def _parse_s_line(line: str) -> tuple[float, float, int] | None:
    s = (line or "").strip()
    if not s or s[0] not in "sS":
        return None
    rest = s[1:].strip()
    parts = rest.split()
    if len(parts) >= 2:
        try:
            x = float(parts[0])
            z = float(parts[1])
            nrcv = int(float(parts[2])) if len(parts) >= 3 else 0
        except ValueError:
            return None
        return x, z, nrcv
    if len(s) >= 21:
        try:
            x = float(s[1:11])
            z = float(s[11:21])
            nrcv = int(float(s[21:26])) if len(s) >= 26 else 0
        except ValueError:
            return None
        return x, z, nrcv
    return None


def nearest_obs_id(
    x: float,
    stations: list[tuple[int, float, float]],
    *,
    tol: float = 0.05,
) -> int | None:
    """用模型距离（station.lis 第二列）匹配第一列 OBS 号。"""
    best_id: int | None = None
    best_d = float(tol)
    for oid, sx, _sz in stations:
        d = abs(float(x) - float(sx))
        if d <= best_d:
            best_d = d
            best_id = int(oid)
    return best_id


def match_isrc_to_obs(
    sources: list[tuple[int, float, float]],
    stations: list[tuple[int, float, float]],
    *,
    tol: float = 0.05,
) -> dict[int, int]:
    """``isrc``（数据文件炮序）→ station.lis OBS 号。"""
    out: dict[int, int] = {}
    for isrc, x, _z in sources:
        oid = nearest_obs_id(x, stations, tol=tol)
        if oid is not None:
            out[int(isrc)] = oid
    return out


@dataclass
class ObsContext:
    stations: list[tuple[int, float, float]] = field(default_factory=list)
    isrc_to_obs: dict[int, int] = field(default_factory=dict)
    station_path: Path | None = None
    data_path: Path | None = None

    def obs_id(self, isrc: int) -> int:
        return self.isrc_to_obs.get(int(isrc), int(isrc))

    def label(self, isrc: int) -> str:
        oid = self.isrc_to_obs.get(int(isrc))
        if oid is not None:
            return f"OBS {oid}"
        return f"OBS {isrc}"

    def enrich_from_ray_x(self, isrc: int, x: float, *, tol: float = 0.05) -> None:
        if int(isrc) in self.isrc_to_obs or not self.stations:
            return
        oid = nearest_obs_id(x, self.stations, tol=tol)
        if oid is not None:
            self.isrc_to_obs[int(isrc)] = oid


def resolve_station_lis_path(state: FormState, work: Path) -> Path | None:
    hit = _as_work_file(state.get_str("tx.station_lis"), work)
    if hit is not None:
        return hit
    for rel in ("station.lis", "inputs/station.lis"):
        p = work / rel
        if p.is_file():
            return p.resolve()
    return None


def resolve_inv_geometry_path(state: FormState, work: Path) -> Path | None:
    for key in ("inv.data", "tx.data_out", "fwd.out_ttime", "fwd.geom"):
        hit = _as_work_file(state.get_str(key), work)
        if hit is not None:
            return hit
    return None


def load_obs_context(state: FormState, work: Path) -> ObsContext:
    ctx = ObsContext()
    sp = resolve_station_lis_path(state, work)
    if sp is not None:
        try:
            ctx.stations = parse_station_lis(sp)
            ctx.station_path = sp
        except (OSError, ValueError):
            pass
    dp = resolve_inv_geometry_path(state, work)
    if dp is not None:
        ctx.data_path = dp
        if ctx.stations:
            sources = list_tomo2d_sources(dp)
            ctx.isrc_to_obs = match_isrc_to_obs(sources, ctx.stations)
    return ctx


def plot_obs_x_markers(
    ax,
    stations: list[tuple[int, float, float]],
    *,
    obs_ids: set[int] | None = None,
    y: float = 0.0,
) -> int:
    """在走时拟合图上于 OBS 模型距离处画竖线 + 白圆，便于和残差（接收点 X）对照。"""
    if not stations:
        return 0
    n = 0
    for oid, x, _z in stations:
        if obs_ids is not None and int(oid) not in obs_ids:
            continue
        ax.axvline(float(x), color="#94a3b8", lw=0.55, alpha=0.4, zorder=1.4)
        ax.scatter(
            [float(x)],
            [y],
            s=28,
            facecolors="white",
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
            clip_on=True,
        )
        n += 1
    return n


def plot_obs_markers(
    ax,
    stations: list[tuple[int, float, float]],
    *,
    x_range: tuple[float, float] | None = None,
    label_ids: set[int] | None = None,
) -> int:
    """在速度场上画白圆+黑边 OBS；返回画出的点数。"""
    if not stations:
        return 0
    xs: list[float] = []
    zs: list[float] = []
    ids: list[int] = []
    xmin = xmax = None
    if x_range is not None:
        xmin, xmax = float(min(x_range)), float(max(x_range))
        pad = max(1.0, 0.02 * (xmax - xmin)) if xmax > xmin else 1.0
        xmin -= pad
        xmax += pad
    for oid, x, z in stations:
        if xmin is not None and not (xmin <= float(x) <= xmax):
            continue
        xs.append(float(x))
        zs.append(float(z))
        ids.append(int(oid))
    if not xs:
        return 0
    ax.scatter(
        xs,
        zs,
        s=42,
        facecolors="white",
        edgecolors="black",
        linewidths=0.9,
        zorder=6,
        clip_on=True,
    )
    n = len(ids)
    show_all = n <= 24
    try:
        from matplotlib.patheffects import withStroke

        stroke = [withStroke(linewidth=2.2, foreground="white")]
    except Exception:
        stroke = None
    for oid, x, z in zip(ids, xs, zs):
        if not show_all and label_ids is not None and oid not in label_ids:
            continue
        txt = ax.annotate(
            str(oid),
            (x, z),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            va="bottom",
            fontsize=7,
            color="black",
            zorder=7,
            clip_on=True,
        )
        if stroke:
            txt.set_path_effects(stroke)
    return n
