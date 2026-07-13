"""
parameters.py - ZPLOT 参数管理类

管理所有32个菜单参数，对应原始 zplot 的菜单系统
"""

import os
import math
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ZPlotParameters:
    """ZPLOT 参数类 - 包含所有32个菜单参数"""
    
    # 显示控制参数 (1-4)
    iscale: int = 0          # 缩放模式: 0=自动, 1=固定, 2=变增益（默认贴近 Fortran 自动增益）
    amp: float = 1.0         # 振幅缩放因子
    rcor: float = 0.0        # 距离校正指数
    sf: float = 0.0          # 缩放因子
    
    # 数据处理参数 (5-8)
    irec: int = 1            # 记录号（炮集号）
    itx: int = 0             # 理论走时显示: 0=不显示, 1=显示
    imute: int = 0           # 静校正模式: 0=无, >0=正向, <0=反向
    iwater: int = 0          # 水层校正: 0=无, 1=启用
    
    # 滤波参数 (9-12)
    ibndps: int = 1          # 带通滤波: 0=无, 1=启用, -2=FFT显示（默认启用滤波）
    izerop: int = 1          # 零相位滤波: 0=否, 1=是
    freqlo: float = 3.0      # 低截止频率 (Hz)
    freqhi: float = 15.0     # 高截止频率 (Hz)
    
    # 显示参数 (13-16)
    tlinc: float = 0.1       # 时间增量 (s)
    clip: float = 0.0        # 裁剪值
    vred: float = 6.0        # 折合速度 (km/s)
    ishade: int = 0          # 阴影填充: 0=无, >0=正峰值, <0=负峰值
    
    # 拾取控制参数 (17-20)
    spick: float = 0.5       # 拾取符号大小
    ixaxis: int = -1         # X轴类型: -1=炮检距, -2=模型位置, -3=方位角, -4=修正方位角, -5=道号
    itype: int = 1           # 数据类型过滤: 
                              # 0=全部, 1=垂直(默认), 2=径向, 3=横向, 4=水听器
                              # -1=垂直+径向, -2=径向+横向, -3=径向+水听器, -4=垂直+水听器
    apick: int = 1           # 活动拾取字 (1-40)
    
    # 其他参数 (21-24)
    nskip: int = 0           # 跳过的道数
    xmm: float = 200.0       # X轴范围（毫米）
    xmin: float = 0.0        # X轴最小值
    xmax: float = 100.0      # X轴最大值
    
    # 时间参数 (25-28)
    ndecim: int = 1          # 数据抽取间隔
    tmm: float = 200.0       # 时间轴范围（毫米）
    tmin: float = 0.0        # 时间最小值 (s)
    tmax: float = 20.0       # 时间最大值 (s)
    
    # 特殊功能 (29-32) - 这些是按钮，不是参数
    # quit - 退出（通过菜单实现）
    # orig/prev plot - 原始/前一个图（通过菜单实现）
    # plot - 绘图（自动触发）
    # auto pick - 自动拾取（通过菜单实现）
    
    # 额外参数（从 namelist /pltpar/ 中，对应原Fortran代码的40个参数）
    # 1-5: 基本控制参数
    icomp: int = 999         # 组件选择（默认999=全部）
    iscreen: int = 0         # 屏幕输出: 0=否, 1=是
    iout: int = 0            # 输出控制: 0=无, 1=输出到文件, 2=不进行去直流
    idot: int = 1            # 点显示: 0=否, 1=是
    itemp: int = 0           # 温度/时间相关: 0=否, 1=是
    
    # 6-7: 颜色参数（已存在）
    pickc: list = field(default_factory=lambda: list(range(2, 42)))  # 拾取颜色（40个）
    colour: list = field(default_factory=lambda: [4, 3, 6, 4, 5])  # 颜色设置（5个）
    
    # 8-10: 显示位置参数
    xpick: float = 0.0       # 拾取X偏移（毫米）
    annht: float = 0.0       # 注释高度（毫米）
    space: float = 0.0       # 间距（毫米）
    
    # 11-12: 格式参数
    ndeca: int = 1           # 小数位数
    title: str = ''          # 标题字符串
    
    # 13-15: 显示密度和滤波参数
    overlap: float = 10.0    # 重叠百分比（0-90）
    dens: float = 0.0        # 密度（自动计算）
    npoles: int = 8          # 滤波器阶数
    
    # 16-20: 自适应叠加和能量参数
    tcrcor: float = 0.60     # 互相关窗口长度（秒）
    tlag: float = 0.10       # 时间延迟搜索范围（秒）
    tempwin: float = 1.0     # 临时窗口（秒）
    twinener: float = 0.1    # 时间窗口能量（秒）
    minenratio: float = 10.0 # 最小能量比
    
    # 21-24: 静音和输出参数
    vmute: float = 0.0       # 静音速度（km/s）
    tmute: float = 0.0       # 静音时间（秒）
    nout: int = 0            # 输出点数（0=全部）
    nbout: int = 0            # 输出前点数
    
    # 25-28: 轴和布局参数
    albht: float = 2.5       # 轴标签高度（毫米）
    amenht: float = -1.0     # 菜单高度（毫米，-1=自动）
    orig: float = 12.5       # 原点位置（毫米）
    itrev: int = 0           # 时间反转: 0=否, 1=是
    
    # 29-31: 频率和增益参数
    fmax: float = -1.0       # 最大频率（Hz，-1=自动）
    tvg: float = 1.0         # 时间变化增益
    pvg: float = 1.0         # 功率变化增益
    
    # 32-35: 注释和显示参数
    iann: int = 0            # 注释索引: 0=无, >0=显示
    thick: float = -1.0      # 厚度（-1=自动）
    ndiga: int = -1          # 数字精度（-1=自动）
    shadedc: float = 0.0     # 阴影颜色
    
    # 36-40: 自适应叠加参数（Haibo Huang 2023新增）
    hilbratio: float = 3.0   # Hilbert变换比例因子（相位一致性加权，默认3.0，对应Fortran代码）
    nsi: int = 10            # 叠加迭代次数
    pjgl: int = 3            # Lp范数指数（1=L1, 2=L2, 3=L3）
    stkwb: float = 0.04      # 叠加窗口起始时间（秒）
    txadj: float = 0.0       # 时间X调整（秒）
    
    # 自适应叠加开关（不在40个参数中，但用于控制）
    iadaptive: int = 0       # 自适应叠加开关：0=关闭, 1=启用
    
    # Hilbert变换参数（不在40个参数中，但用于控制）
    ihilbt: int = 0          # Hilbert变换模式: 0=无, 1=输出相位, 2=输出包络
    
    # 叠加窗口长度（不在40个参数中，但用于自适应叠加）
    stkwl: float = 0.5       # 叠加窗口长度（秒，等于tcrcor）
    
    # 差分时间搜索范围（不在40个参数中，但用于自适应叠加）
    dtcw: float = 0.1        # 差分时间搜索范围（秒，等于tlag）
    
    # 渲染质量参数（zplotpy 扩展）
    # 阴影质量档位：fast=快速, balanced=平衡, high=高质量
    shade_quality_preset: str = 'balanced'

    # 去噪参数（processors.denoise）
    denoise_enabled: int = 0          # 去噪开关（0=关闭, 1=开启）
    denoise_ab_raw: int = 1           # A/B对比默认原始A（UI默认不勾选A/B）
    denoise_scope: str = 'rendered'   # 去噪范围：rendered/visible/record
    denoise_f_s: float = 3.0          # 去噪频带下限 (Hz)
    denoise_f_e: float = 20.0         # 去噪频带上限 (Hz)
    denoise_bwconn: int = 8           # 邻域连通性（4 或 8）
    denoise_strength: float = 3.0     # GCV收缩强度（trace路径）
    denoise_workers: int = 1          # section路径并行参数（当前实现预留）
    denoise_return_debug: int = 0     # 是否返回调试时频信息（0/1）
    denoise_return_result: int = 0    # 是否返回 DenoiseResult 包装（0/1）
    denoise_coh_win: int = 11         # 局部semblance窗长（样点，奇数）
    denoise_coh_lag: int = 2          # 局部semblance最大道间时移（样点）
    denoise_coh_thr: float = 0.55     # 相干门控阈值（越高越保守）
    denoise_coh_blend: float = 0.35   # 相干回混上限（0-1）
    denoise_coh_penalty: float = 0.08 # lag轨迹平滑惩罚（越大越平滑）
    denoise_perf_diag: int = 0        # 去噪性能诊断日志开关（0/1）
    denoise_morph_enable: int = 1     # 形态学约束开关（0/1）
    denoise_morph_preset: str = "balanced"  # 形态学预设：conservative/balanced/strong
    denoise_morph_quantile: float = 0.70    # 形态学种子幅值分位阈值
    denoise_morph_min_area: int = 24        # 最小连通域面积（TF像素）
    denoise_morph_expand: int = 1           # 形态学膨胀步数
    denoise_morph_floor_ratio: float = 0.03 # 幅值地板比例（相对全局最大）
    denoise_morph_keep_strong_q: float = 0.95  # 强能量保护分位阈值
    denoise_pick_guidance: int = 0          # 拾取引导 TF 软门控（0/1）
    denoise_pick_wavelet_length: float = 0.19  # 子波长度 T：时间门高斯包络 FWHM（秒）
    denoise_pick_floor: float = 0.12       # 拾取窗外侧 TF 保留比例下限

    def __post_init__(self):
        """根据配置档应用默认参数

        通过环境变量 `ZPLOTPY_PARAM_PROFILE` 控制：
        - `modern`（默认）：保持当前 Python 版本默认体验
        - `fortran`：尽量贴近 Fortran zplot 初始参数
        """
        profile = os.environ.get('ZPLOTPY_PARAM_PROFILE', 'modern').strip().lower()
        if profile == 'fortran':
            self.apply_fortran_profile()

    def apply_fortran_profile(self):
        """应用更接近 Fortran 版本的参数默认值

        注意：
        - 仅覆盖语义一致、低风险的参数。
        - `nskip` 在 Python 实现中语义与 Fortran 不同（Python: 0=不跳道），因此不在此处强制设为 1。
        """
        # 菜单参数（1-28）中的安全对齐项
        self.iscale = 0
        self.amp = 1.0
        self.rcor = 0.0
        self.sf = 0.01

        self.itx = 0
        self.imute = 0
        self.iwater = 0

        self.ibndps = 0
        self.izerop = 1
        self.freqlo = 1.0
        self.freqhi = 20.0

        self.tlinc = 0.0
        self.clip = 0.0
        self.vred = -1.0
        self.ishade = 0

        self.spick = 0.0
        self.ixaxis = 0
        self.itype = 0
        self.apick = 1

        # nskip 保持 Python 语义默认值（0）
        self.ndecim = 1
        self.xmm = 230.0
        self.tmm = 130.0

        # Fortran 额外参数（已在 Python 中使用）
        self.tcrcor = 0.60
        self.tlag = 0.10
        self.npoles = 8
        self.overlap = 10.0
        self.hilbratio = 3.0
        self.nsi = 10
        self.pjgl = 3
        self.stkwb = 0.04
        self.txadj = 0.0
        self.idot = 1
        self.ihilbt = 0
    
    def get_menu_value(self, menu_id: int) -> Any:
        """获取菜单项的值
        
        Args:
            menu_id: 菜单项ID (1-32)
            
        Returns:
            菜单项的值
        """
        menu_map = {
            1: self.iscale, 2: self.amp, 3: self.rcor, 4: self.sf,
            5: self.irec, 6: self.itx, 7: self.imute, 8: self.iwater,
            9: self.ibndps, 10: self.izerop, 11: self.freqlo, 12: self.freqhi,
            13: self.tlinc, 14: self.clip, 15: self.vred, 16: self.ishade,
            17: self.spick, 18: self.ixaxis, 19: self.itype, 20: self.apick,
            21: self.nskip, 22: self.xmm, 23: self.xmin, 24: self.xmax,
            25: self.ndecim, 26: self.tmm, 27: self.tmin, 28: self.tmax,
        }
        return menu_map.get(menu_id)
    
    def set_menu_value(self, menu_id: int, value: Any):
        """设置菜单项的值
        
        Args:
            menu_id: 菜单项ID (1-32)
            value: 要设置的值
        """
        menu_map = {
            1: ('iscale', int), 2: ('amp', float), 3: ('rcor', float), 4: ('sf', float),
            5: ('irec', int), 6: ('itx', int), 7: ('imute', int), 8: ('iwater', int),
            9: ('ibndps', int), 10: ('izerop', int), 11: ('freqlo', float), 12: ('freqhi', float),
            13: ('tlinc', float), 14: ('clip', float), 15: ('vred', float), 16: ('ishade', int),
            17: ('spick', float), 18: ('ixaxis', int), 19: ('itype', int), 20: ('apick', int),
            21: ('nskip', int), 22: ('xmm', float), 23: ('xmin', float), 24: ('xmax', float),
            25: ('ndecim', int), 26: ('tmm', float), 27: ('tmin', float), 28: ('tmax', float),
        }
        
        if menu_id in menu_map:
            attr_name, attr_type = menu_map[menu_id]
            setattr(self, attr_name, attr_type(value))
    
    def get_menu_name(self, menu_id: int) -> str:
        """获取菜单项名称"""
        menu_names = {
            1: 'iscale', 2: 'amp', 3: 'rcor', 4: 'sf',
            5: 'irec', 6: 'itx', 7: 'imute', 8: 'iwater',
            9: 'ibndps', 10: 'izerop', 11: 'freqlo', 12: 'freqhi',
            13: 'tlinc', 14: 'clip', 15: 'vred', 16: 'ishade',
            17: 'spick', 18: 'ixaxis', 19: 'itype', 20: 'pick',
            21: 'nskip', 22: 'xmm', 23: 'xmin', 24: 'xmax',
            25: 'ndecim', 26: 'tmm', 27: 'tmin', 28: 'tmax',
            29: 'quit', 30: 'orig/prev plot', 31: 'plot', 32: 'auto pick',
        }
        return menu_names.get(menu_id, f'menu_{menu_id}')
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            # 32个菜单参数
            'iscale': self.iscale, 'amp': self.amp, 'rcor': self.rcor, 'sf': self.sf,
            'irec': self.irec, 'itx': self.itx, 'imute': self.imute, 'iwater': self.iwater,
            'ibndps': self.ibndps, 'izerop': self.izerop, 'freqlo': self.freqlo, 'freqhi': self.freqhi,
            'tlinc': self.tlinc, 'clip': self.clip, 'vred': self.vred, 'ishade': self.ishade,
            'spick': self.spick, 'ixaxis': self.ixaxis, 'itype': self.itype, 'apick': self.apick,
            'nskip': self.nskip, 'xmm': self.xmm, 'xmin': self.xmin, 'xmax': self.xmax,
            'ndecim': self.ndecim, 'tmm': self.tmm, 'tmin': self.tmin, 'tmax': self.tmax,
            # 40个额外参数
            'icomp': self.icomp, 'iscreen': self.iscreen, 'iout': self.iout, 'idot': self.idot, 'itemp': self.itemp,
            'pickc': self.pickc.copy() if isinstance(self.pickc, list) else self.pickc,
            'colour': self.colour.copy() if isinstance(self.colour, list) else self.colour,
            'xpick': self.xpick, 'annht': self.annht, 'space': self.space,
            'ndeca': self.ndeca, 'title': self.title,
            'overlap': self.overlap, 'dens': self.dens, 'npoles': self.npoles,
            'tcrcor': self.tcrcor, 'tlag': self.tlag, 'tempwin': self.tempwin, 'twinener': self.twinener, 'minenratio': self.minenratio,
            'vmute': self.vmute, 'tmute': self.tmute, 'nout': self.nout, 'nbout': self.nbout,
            'albht': self.albht, 'amenht': self.amenht, 'orig': self.orig, 'itrev': self.itrev,
            'fmax': self.fmax, 'tvg': self.tvg, 'pvg': self.pvg,
            'iann': self.iann, 'thick': self.thick, 'ndiga': self.ndiga, 'shadedc': self.shadedc,
            'hilbratio': self.hilbratio, 'nsi': self.nsi, 'pjgl': self.pjgl, 'stkwb': self.stkwb, 'txadj': self.txadj,
            # 其他控制参数
            'stkwl': self.stkwl, 'dtcw': self.dtcw, 'iadaptive': self.iadaptive, 'ihilbt': self.ihilbt,
            'shade_quality_preset': self.shade_quality_preset,
            # 去噪参数
            'denoise_enabled': self.denoise_enabled,
            'denoise_ab_raw': self.denoise_ab_raw,
            'denoise_scope': self.denoise_scope,
            'denoise_f_s': self.denoise_f_s,
            'denoise_f_e': self.denoise_f_e,
            'denoise_bwconn': self.denoise_bwconn,
            'denoise_strength': self.denoise_strength,
            'denoise_workers': self.denoise_workers,
            'denoise_return_debug': self.denoise_return_debug,
            'denoise_return_result': self.denoise_return_result,
            'denoise_coh_win': self.denoise_coh_win,
            'denoise_coh_lag': self.denoise_coh_lag,
            'denoise_coh_thr': self.denoise_coh_thr,
            'denoise_coh_blend': self.denoise_coh_blend,
            'denoise_coh_penalty': self.denoise_coh_penalty,
            'denoise_perf_diag': self.denoise_perf_diag,
            'denoise_morph_enable': self.denoise_morph_enable,
            'denoise_morph_preset': self.denoise_morph_preset,
            'denoise_morph_quantile': self.denoise_morph_quantile,
            'denoise_morph_min_area': self.denoise_morph_min_area,
            'denoise_morph_expand': self.denoise_morph_expand,
            'denoise_morph_floor_ratio': self.denoise_morph_floor_ratio,
            'denoise_morph_keep_strong_q': self.denoise_morph_keep_strong_q,
            'denoise_pick_guidance': self.denoise_pick_guidance,
            'denoise_pick_wavelet_length': self.denoise_pick_wavelet_length,
            'denoise_pick_floor': self.denoise_pick_floor,
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """从字典加载"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if "denoise_pick_wavelet_length" not in data and "denoise_pick_half_width" in data:
            try:
                sig_o = float(data["denoise_pick_half_width"])
                if math.isfinite(sig_o) and sig_o > 0.0:
                    self.denoise_pick_wavelet_length = sig_o * (2.0 * math.sqrt(2.0 * math.log(2.0)))
            except Exception:
                pass
