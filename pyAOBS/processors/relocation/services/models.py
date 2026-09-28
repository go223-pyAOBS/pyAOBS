"""姿态校正 GUI / 服务层共用数据结构（无 Qt 依赖）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class WaveformSelection:
    """单段 V 选波窗口。"""

    trace_idx: int
    offset: float
    t_display: float
    t_true: float
    pick_word: int = 1

    def key(self) -> Tuple[int, int]:
        return int(self.trace_idx), int(self.pick_word)

    def to_dict(self) -> Dict[str, float]:
        return {
            "trace_idx": float(self.trace_idx),
            "offset": float(self.offset),
            "t_display": float(self.t_display),
            "t_true": float(self.t_true),
            "pick_word": float(self.pick_word),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WaveformSelection":
        return cls(
            trace_idx=int(d.get("trace_idx", -1)),
            offset=float(d.get("offset", 0.0)),
            t_display=float(d.get("t_display", 0.0)),
            t_true=float(d.get("t_true", 0.0)),
            pick_word=int(d.get("pick_word", 1)),
        )


@dataclass
class AttitudeUiParams:
    """姿态校正对话框参数（与 zplotpy 字段兼容）。

    校正截窗预处理（不含增益）：
      默认 rmean + rtrend；可选与主图一致的带通。
    """

    wave_pre: float = 0.30
    wave_post: float = 0.70
    att_iter: int = 4
    # 默认压低走时权重，校正更侧重多炮方位一致性/位置
    att_wtt: float = 0.15
    att_wpol: float = 1.0
    # 非 ppol 窗对称；对齐 ppol 多炮一致性时默认关闭
    att_wsym: float = 0.0
    # 用户预设全局走时 shift（正=观测加走时/变晚，负=减走时/变早）；
    # 作用在观测/拾取侧；反演最优值约等于残差(预测-观测)
    prior_tt_shift_sec: float = 0.0
    # 默认不校正倾角（仅方位/位置/走时）；勾选后才搜索 tilt
    correct_tilt: bool = False
    # 截窗预处理（进反演；无增益）
    use_rmean: bool = True
    use_rtrend: bool = True
    use_bandpass: bool = True
    freqlo: float = 3.0
    freqhi: float = 15.0
    npoles: int = 8
    izerop: bool = True

    def to_dict(self) -> Dict[str, float]:
        return {
            "wave_pre": float(self.wave_pre),
            "wave_post": float(self.wave_post),
            "att_iter": float(self.att_iter),
            "att_wtt": float(self.att_wtt),
            "att_wpol": float(self.att_wpol),
            "att_wsym": float(self.att_wsym),
            "prior_tt_shift_sec": float(self.prior_tt_shift_sec),
            "correct_tilt": 1.0 if self.correct_tilt else 0.0,
            "use_rmean": 1.0 if self.use_rmean else 0.0,
            "use_rtrend": 1.0 if self.use_rtrend else 0.0,
            "use_bandpass": 1.0 if self.use_bandpass else 0.0,
            "freqlo": float(self.freqlo),
            "freqhi": float(self.freqhi),
            "npoles": float(self.npoles),
            "izerop": 1.0 if self.izerop else 0.0,
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "AttitudeUiParams":
        d = d or {}

        def _flag(key: str, default: bool) -> bool:
            if key not in d:
                return bool(default)
            return bool(float(d.get(key, 1.0 if default else 0.0)))

        return cls(
            wave_pre=float(d.get("wave_pre", 0.30)),
            wave_post=float(d.get("wave_post", 0.70)),
            att_iter=int(round(float(d.get("att_iter", 4)))),
            att_wtt=float(d.get("att_wtt", 0.15)),
            att_wpol=float(d.get("att_wpol", 1.0)),
            att_wsym=float(d.get("att_wsym", 0.0)),
            prior_tt_shift_sec=float(d.get("prior_tt_shift_sec", 0.0)),
            correct_tilt=_flag("correct_tilt", False),
            use_rmean=_flag("use_rmean", True),
            use_rtrend=_flag("use_rtrend", True),
            use_bandpass=_flag("use_bandpass", True),
            freqlo=float(d.get("freqlo", 3.0)),
            freqhi=float(d.get("freqhi", 15.0)),
            npoles=int(round(float(d.get("npoles", 8)))),
            izerop=_flag("izerop", True),
        )


@dataclass
class AttitudeSolution:
    """当前姿态解（可作下次反演初值）。

    走时三量（分开存）：
      prior_tt_shift_sec — 用户预置全局走时 shift（观测侧：正加负减）
      tt_corr_sec        — 校正得到的走时增量（观测侧）
      time_shift_sec     — 最终全局偏移 = prior + corr（应用到拾取/显示：t' = t + final）
    """

    azimuth_deg: float = 0.0
    tilt_deg: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    prior_tt_shift_sec: float = 0.0
    tt_corr_sec: float = 0.0
    time_shift_sec: float = 0.0

    def position_correction(self) -> Tuple[float, float, float]:
        return float(self.dx), float(self.dy), float(self.dz)

    def to_dict(self) -> Dict[str, float]:
        return {
            "azimuth_deg": float(self.azimuth_deg),
            "tilt_deg": float(self.tilt_deg),
            "dx": float(self.dx),
            "dy": float(self.dy),
            "dz": float(self.dz),
            "prior_tt_shift_sec": float(self.prior_tt_shift_sec),
            "tt_corr_sec": float(self.tt_corr_sec),
            "time_shift_sec": float(self.time_shift_sec),
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "AttitudeSolution":
        d = d or {}
        prior = float(d.get("prior_tt_shift_sec", 0.0))
        final = float(d.get("time_shift_sec", 0.0))
        if "tt_corr_sec" in d:
            corr = float(d.get("tt_corr_sec", 0.0))
        else:
            # 兼容旧解：仅有 final 时视为纯最终值，corr=final、prior=0
            corr = float(d.get("tt_corr_sec", final - prior))
        return cls(
            azimuth_deg=float(d.get("azimuth_deg", 0.0)),
            tilt_deg=float(d.get("tilt_deg", 0.0)),
            dx=float(d.get("dx", 0.0)),
            dy=float(d.get("dy", 0.0)),
            dz=float(d.get("dz", 0.0)),
            prior_tt_shift_sec=prior,
            tt_corr_sec=corr,
            time_shift_sec=final,
        )


@dataclass
class StackResult:
    """V 段叠加结果（相对中心时间轴）。"""

    tau: List[float] = field(default_factory=list)
    stack: List[float] = field(default_factory=list)
    centers: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tau": list(self.tau),
            "stack": list(self.stack),
            "centers": list(self.centers),
        }


@dataclass
class AutoPickParams:
    """自动拾取参数。"""

    window_length: float = 0.1
    min_energy_ratio: float = 1.5
    search_start: Optional[float] = None
    search_end: Optional[float] = None
    vred: float = 0.0
    pick_word: int = 1
    # 仅对指定道索引拾取；None 表示全部道
    trace_indices: Optional[List[int]] = None


@dataclass
class SessionState:
    """会话侧状态：选波 + 姿态 UI/解（不含波形大数组）。"""

    waveform_selections: List[WaveformSelection] = field(default_factory=list)
    # (trace_idx, pick_word) -> 叠加校正后的真实走时中心
    corrected_ttrue: Dict[Tuple[int, int], float] = field(default_factory=dict)
    attitude_ui: AttitudeUiParams = field(default_factory=AttitudeUiParams)
    attitude_solution: AttitudeSolution = field(default_factory=AttitudeSolution)
    current_apick: int = 1
    terrain_path: Optional[str] = None
    # build_bathymetry_sampler 可用的 UTM meta（points/grid）
    terrain_meta_utm: Optional[Dict[str, Any]] = None
