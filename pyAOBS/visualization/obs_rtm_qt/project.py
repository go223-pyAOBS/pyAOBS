# -*- coding: utf-8 -*-
"""工区 / 会话状态（内存 + 可选 JSON）。"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GeometryParams:
    """与 su_to_shots / obs_segy_geometry 对齐。"""
    geom: str = "offset"  # obs | segy | offset
    group: str = "fldr"
    xy_unit: str = "m"
    line_axis: str = "x"
    obs_x_km: float = 0.0
    offset_sign: float = 1.0
    zshot_km: float = 0.01
    zobs_mode: str = "const"
    zobs_const_km: float = 1.901
    component: str = "hydro"
    trid: str = ""
    endian: str = "little"
    native: bool = True


@dataclass
class PreprocessParams:
    """zplotpy DataProcessor + 多边形 mute + 显示增益。"""
    use_bandpass: bool = True
    freqlo: float = 3.0
    freqhi: float = 15.0
    use_mute: bool = False  # 速度 mute：原始时间轴 tm+|x|/vm
    tmute: float = 0.0
    vmute: float = 6.0  # mute 线速度 km/s（左右分支）
    vel_mute_invert: bool = False  # False=切深(inner)；True=切浅
    mute_tp: float = 0.15  # 边缘余弦过渡(s)，对齐 mutter tp；0=硬切
    # 旧字段：曾误用于 mute 公式；保留仅兼容 JSON，mute 不再读取
    vred: float = 8.0
    # 多边形 mute（与 qt_fast_viewer 一致；x 轴见 poly_x_mode）
    use_poly_mute: bool = False
    poly_invert: bool = False
    poly_x_mode: str = "offset"  # offset | trace
    poly_points: list = None  # type: ignore  # [[x,t], ...]
    # 手选炮集（浏览；未开多边形时驱动「应用选道」与 RTM）
    hand_select: bool = True
    proc_shot_list: str = ""
    # 「应用预处理」：写出/预览定稿 = mute(原始) 后再带通+增益一次
    apply_preprocess: bool = True
    # 显示增益（对齐 zplotpy；与带通一并写入 shots_proc）
    use_gain: bool = True
    iscale: int = 0  # 0自动 / 1固定 / 2变增益
    amp: float = 1.2
    rcor: float = 0.3
    sf: float = 0.0
    tvg: float = 1.0
    pvg: float = 1.0
    clip: float = 0.0  # 硬裁剪；0=关
    dscale: float = 1.0  # wiggle 显示宽度
    display_mode: str = "density"  # density | wiggle | fill+ | fill-
    pclip: float = 98.0  # density 色标百分位
    # 全局显示折合 km/s；0=关。公式 t'=t-|x-xobs|/vred；多边形 mute 预览/定稿与之对齐
    display_vred: float = 8.0

    def __post_init__(self) -> None:
        if self.poly_points is None:
            self.poly_points = []


@dataclass
class GridParams:
    """与 SConstruct / services.geometry 一致：n1=z, n2=x, km。

    默认 dx/dz 与速度页 vin 栅格一致（0.5 / 0.25 km）。
    「由炮点建议网格」只改 ox/nx（及 nz），保留当前 dx/dz。
    """
    oz: float = 0.0
    dz: float = 0.25
    nz: int = 161  # ~0–40 km @ dz=0.25
    ox: float = -400.0
    dx: float = 0.5
    nx: int = 1001  # ~500 km @ dx=0.5


@dataclass
class VelocityParams:
    """层析 / 内置一维 + bath → 成像速度。"""
    tomo_path: str = ""  # 可为 v.in / .grd / .nc / smesh / .rsf
    # 用户选中的原始 Zelt v.in（转换 tomo_vel.rsf 后仍保留，供 Interfaces / 地形 bath）
    zelt_vin_path: str = ""
    bath_path: str = "prep/geom/bath_x.txt"
    flat_bath_km: float = 1.901
    vwater: float = 1.50
    fill_water: bool = True
    smooth_rect: int = 0  # 0=不光滑；>0 盒式光滑
    out_vel: str = "rtm_in/vel.rsf"
    # v.in / smesh 栅格化间距（km）；最终仍可再采样到工区网格
    vin_dx_km: float = 0.5
    vin_dz_km: float = 0.25
    auto_convert_rsf: bool = True  # 非 rsf 自动写成 tomo_vel.rsf
    # 速度来源：file=用户模型；builtin_1d=内置一维（可无 tomo）
    vel_source: str = "file"
    # absolute=按绝对深度 z；subbottom=按海底以下深度（起伏水深更贴沉积）
    v1d_ref: str = "subbottom"
    # 预设：linear_crust | layered_crust
    v1d_preset: str = "linear_crust"
    v1d_v0: float = 2.0  # 海底处岩体速度 km/s（linear）
    v1d_grad: float = 0.5  # 垂向梯度 km/s / km（linear）
    v1d_vmax: float = 8.0  # 速度上限
    # 诊断：把某速度等值线做成有限跳变界面（默认关）
    v1d_iface_enable: bool = False
    v1d_iface_v: float = 6.0  # 等值速度 km/s（与 Contours 档位一致）
    v1d_iface_dv: float = 0.8  # 界面下侧跳变幅度 km/s（可负）
    # v.in Interfaces 着色：B/S/M 层索引（0-based）；None=未选/Auto
    iface_basement: Optional[int] = None
    iface_seafloor: Optional[int] = None
    iface_moho: Optional[int] = None


@dataclass
class RtmParams:
    """叠前 RTM 作业（Madagascar scons / rtm_shot_loop）。"""
    workdir: str = "rtm_work"
    vel_rsf: str = "rtm_in/vel.rsf"
    use_shots_proc: bool = True
    # 用户侧：记录时长 / 采样率；0 表示未单独保存，界面从 nt/dt 回推
    # 注意：勿默认 40/250，否则旧 JSON 仅有 nt 时会被误当成 T=40s
    tmax: float = 0.0  # s；>0 时与 fs 一并写入 JSON
    fs: float = 0.0  # Hz
    nt: int = 10001
    dt: float = 0.004
    fmin: float = 3.0
    fmax: float = 8.0
    first_shot: int = 0  # 起始炮号（与 shot_NNN / shots_xz 行号一致）
    max_shot: int = 1  # 从 first_shot 起跑几炮；-1 = 直到末炮
    # 非空则优先：自选炮号，如 "0,5,10-12"；支持逗号/空格/分号与 a-b 闭区间
    # 互易 RTM：这些炮点作检波（反传注入）
    shot_list: str = ""
    # 空=全部 OBS；否则如 "0" / "0,2"（互易时作震源的 OBS 下标）
    obs_list: str = ""
    # madagascar | custom_bin | dry_run
    engine: str = "madagascar"
    rtm_bin: str = ""
    dry_run: bool = False
    jsnap: int = 80  # awefd2d 波场抽样（越大越省内存；首跑宜偏大）
    nb: int = 40  # 吸收边界厚度
    mute_water_preview: bool = True
    # awefd2d verb=y：刷时间步；GUI 管道+日志刷新会明显拖慢，默认关（看心跳/>>> 行即可）
    awefd_verb: bool = False


@dataclass
class ObsRtmProject:
    """一个偏移工区目录下的状态。"""
    name: str = "untitled"
    workdir: str = ""
    su_path: str = ""
    tomo_vel: str = ""
    shots_dir: str = "inputs/shots"
    shots_mute_dir: str = "prep/shots_mute"
    shots_proc_dir: str = "prep/shots_proc"
    shots_xz: str = "prep/geom/shots_xz.txt"
    obs_xz: str = "prep/geom/obs_xz.txt"
    summary: str = "diag/su_summary.txt"
    offsets_txt: str = "prep/geom/offsets.txt"
    geometry: GeometryParams = field(default_factory=GeometryParams)
    preprocess: PreprocessParams = field(default_factory=PreprocessParams)
    grid: GridParams = field(default_factory=GridParams)
    velocity: VelocityParams = field(default_factory=VelocityParams)
    rtm: RtmParams = field(default_factory=RtmParams)
    notes: str = ""

    def ensure_workdir(self, *, migrate: bool = True) -> str:
        if not self.workdir:
            raise ValueError("未设置工区目录 workdir")
        from .services.workdir_layout import prepare_workdir

        prepare_workdir(self, migrate=migrate)
        return self.workdir

    def path(self, *parts: str) -> str:
        return os.path.join(self.workdir, *parts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ObsRtmProject":
        def _filter(dc_cls, raw: dict):
            names = set(dc_cls.__dataclass_fields__.keys())
            return {k: v for k, v in (raw or {}).items() if k in names}

        g = GeometryParams(**_filter(GeometryParams, d.get("geometry") or {}))
        p = PreprocessParams(**_filter(PreprocessParams, d.get("preprocess") or {}))
        grid = GridParams(**_filter(GridParams, d.get("grid") or {}))
        vel = VelocityParams(**_filter(VelocityParams, d.get("velocity") or {}))
        rtm = RtmParams(**_filter(RtmParams, d.get("rtm") or {}))
        skip = {"geometry", "preprocess", "grid", "velocity", "rtm"}
        kw = {k: v for k, v in d.items() if k not in skip}
        return cls(geometry=g, preprocess=p, grid=grid, velocity=vel, rtm=rtm, **kw)

    def save(self, path: Optional[str] = None) -> str:
        self.ensure_workdir()
        from .services.workdir_layout import project_json_path

        path = path or project_json_path(self.workdir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: str) -> "ObsRtmProject":
        with open(path, "r", encoding="utf-8") as f:
            proj = cls.from_dict(json.load(f))
        from .services.workdir_layout import infer_workdir_from_json

        if not proj.workdir:
            proj.workdir = infer_workdir_from_json(path)
        return proj


def list_shot_rsf(project: ObsRtmProject) -> List[str]:
    d = project.path(project.shots_dir)
    if not os.path.isdir(d):
        return []
    names = sorted(
        n for n in os.listdir(d)
        if n.startswith("shot_") and n.endswith(".rsf") and not n.endswith(".rsf@")
    )
    return [os.path.join(d, n) for n in names]


def list_shot_rsf_by_indices(
    project: ObsRtmProject, indices: List[int]
) -> List[str]:
    """按炮号取 shots/shot_NNN.rsf（保持 indices 顺序；缺文件跳过）。"""
    out: List[str] = []
    for i in indices:
        p = project.path(project.shots_dir, "shot_%03d.rsf" % int(i))
        if os.path.isfile(p):
            out.append(p)
    return out
