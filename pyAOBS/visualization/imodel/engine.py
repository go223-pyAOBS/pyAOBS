"""
imodel.py - 交互式速度模型分析工具

提供交互式界面，用于：
1. 可视化速度模型（grid格式）
2. 鼠标交互（取点、多边形选择）
3. 提取一维速度剖面
4. 计算物性参数（密度、压力、岩性、重力等）
5. 结果可视化和导出

Author: Haibo Huang
Date: 2025
"""

import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector, Button, RadioButtons
from matplotlib.patches import Polygon
import xarray as xr
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union, Callable, Any
import warnings

# 导入pyAOBS模块
try:
    from pyAOBS.visualization.show_model import GridModelVisualizer, GridModelProcessor
    from pyAOBS.utils import classify_velocity_model, SimpleRockClassifier
    from pyAOBS.utils.rocks import Rock, RockProperties
    from pyAOBS.utils import (
        calculate_density,
        calculate_vs,
        calculate_vp_from_vs_brocher,
        correct_velocity,
        calculate_pressure_from_depth,
        calculate_temperature_from_depth
    )
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pyAOBS.visualization.show_model import GridModelVisualizer, GridModelProcessor
    from pyAOBS.utils import classify_velocity_model, SimpleRockClassifier
    from pyAOBS.utils.rocks import Rock, RockProperties
    from pyAOBS.utils import (
        calculate_density,
        calculate_vs,
        calculate_vp_from_vs_brocher,
        correct_velocity,
        calculate_pressure_from_depth,
        calculate_temperature_from_depth
    )


class PointSelector:
    """点选择器 - 鼠标点击选择点"""
    
    def __init__(self, ax, callback: Optional[Callable] = None):
        """初始化点选择器
        
        Args:
            ax: matplotlib axes对象
            callback: 选择点后的回调函数 callback(x, z)
        """
        self.ax = ax
        self.callback = callback
        self.points = []
        self.point_artists = []
        self.text_artists = []  # 存储text对象，便于管理
        
        # 连接鼠标事件
        self.cid_click = ax.figure.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid_key = ax.figure.canvas.mpl_connect('key_press_event', self.onkey)
    
    def onclick(self, event):
        """处理鼠标点击事件"""
        if event.inaxes != self.ax:
            return
        
        if event.button == 1:  # 左键点击
            x, y = event.xdata, event.ydata
            self.add_point(x, y)
        elif event.button == 3:  # 右键点击
            self.remove_nearest_point(event.xdata, event.ydata)
    
    def onkey(self, event):
        """处理键盘事件"""
        if event.key == 'd' or event.key == 'D':  # 按D删除最近的点
            if self.points:
                self.remove_point(-1)
        elif event.key == 'c' or event.key == 'C':  # 按C清除所有点
            self.clear_points()
    
    def add_point(self, x: float, z: float):
        """添加点"""
        self.points.append((x, z))
        
        # 绘制点
        artist = self.ax.plot(x, z, 'ro', markersize=8, picker=5, 
                             label=f'Point {len(self.points)}')[0]
        self.point_artists.append(artist)
        
        # 添加文本标签
        text = self.ax.text(x, z, f'{len(self.points)}', 
                           fontsize=8, color='white', 
                           bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                           ha='center', va='center')
        self.text_artists.append(text)  # 保存text对象引用
        
        self.ax.figure.canvas.draw()
        
        if self.callback:
            self.callback(x, z, len(self.points) - 1)
    
    def remove_point(self, index: int):
        """删除指定索引的点"""
        if 0 <= index < len(self.points):
            self.points.pop(index)
            self.point_artists[index].remove()
            self.point_artists.pop(index)
            # 移除对应的text对象
            if index < len(self.text_artists):
                self.text_artists[index].remove()
                self.text_artists.pop(index)
            self.ax.figure.canvas.draw()
    
    def remove_nearest_point(self, x: float, z: float, threshold: float = 0.5):
        """删除最近的点"""
        if not self.points:
            return
        
        distances = [np.sqrt((px - x)**2 + (pz - z)**2) for px, pz in self.points]
        nearest_idx = np.argmin(distances)
        
        if distances[nearest_idx] < threshold:
            self.remove_point(nearest_idx)
    
    def clear_points(self):
        """清除所有点"""
        for artist in self.point_artists:
            artist.remove()
        self.point_artists.clear()
        self.points.clear()
        # 清除所有text对象
        for text in self.text_artists:
            text.remove()
        self.text_artists.clear()
        self.ax.figure.canvas.draw()
    
    def get_points(self) -> List[Tuple[float, float]]:
        """获取所有点"""
        return self.points.copy()


class ProfileExtractor:
    """剖面提取器 - 从2D/3D模型中提取1D剖面"""
    
    def __init__(self, grid_data: xr.Dataset):
        """初始化剖面提取器
        
        Args:
            grid_data: xarray Dataset，包含速度网格数据
        """
        self.grid_data = grid_data
        
        # 检查速度变量名（自动检测）
        data_vars = list(grid_data.data_vars)
        coords = list(grid_data.coords)
        
        # 优先查找常见的速度变量名
        velocity_candidates = ['velocity', 'v', 'vp', 'v_p', 'vel', 'speed']
        self.velocity_var = None
        
        for var in velocity_candidates:
            if var in data_vars:
                self.velocity_var = var
                break
        
        # 如果没找到，检查是否有z变量（z通常表示速度）
        if self.velocity_var is None:
            # 检查z是坐标还是数据变量
            if 'z' in data_vars:
                self.velocity_var = 'z'
            elif 'z' in coords and len(data_vars) > 0:
                # 如果z是坐标，使用第一个数据变量
                self.velocity_var = data_vars[0]
            else:
                # 默认使用第一个数据变量
                self.velocity_var = data_vars[0] if data_vars else 'velocity'
        
        # 检查坐标名称
        self.x_coord = next((c for c in coords if c.lower() in ['x', 'lon', 'longitude', 'distance']), coords[0] if coords else 'x')
        # z坐标：如果z是坐标，使用它；否则查找其他深度坐标
        if 'z' in coords and self.velocity_var != 'z':
            self.z_coord = 'z'
        else:
            self.z_coord = next((c for c in coords if c.lower() in ['y', 'depth', 't', 'time']), 
                              coords[1] if len(coords) > 1 else 'z')
    
    def extract_vertical_profile(self, x: float, 
                                z_min: Optional[float] = None,
                                z_max: Optional[float] = None) -> pd.DataFrame:
        """提取垂直剖面（固定x，变化z）
        
        Args:
            x: x坐标
            z_min: 最小深度（可选）
            z_max: 最大深度（可选）
        
        Returns:
            DataFrame: 包含depth和vp列
        """
        # 获取z坐标范围
        z_coords = self.grid_data[self.z_coord].values
        
        if z_min is not None:
            z_coords = z_coords[z_coords >= z_min]
        if z_max is not None:
            z_coords = z_coords[z_coords <= z_max]
        
        # 提取速度值
        vp = []
        for z in z_coords:
            try:
                v = self.grid_data[self.velocity_var].sel(
                    {self.x_coord: x, self.z_coord: z}, 
                    method='nearest'
                )
                vp.append(float(v.values))
            except:
                vp.append(np.nan)
        
        return pd.DataFrame({
            'depth': z_coords,
            'vp': vp
        })
    
    def extract_horizontal_profile(self, z: float,
                                  x_min: Optional[float] = None,
                                  x_max: Optional[float] = None) -> pd.DataFrame:
        """提取水平剖面（固定z，变化x）
        
        Args:
            z: 深度
            x_min: 最小x坐标（可选）
            x_max: 最大x坐标（可选）
        
        Returns:
            DataFrame: 包含x和vp列
        """
        # 获取x坐标范围
        x_coords = self.grid_data[self.x_coord].values
        
        if x_min is not None:
            x_coords = x_coords[x_coords >= x_min]
        if x_max is not None:
            x_coords = x_coords[x_coords <= x_max]
        
        # 提取速度值
        vp = []
        for x in x_coords:
            try:
                v = self.grid_data[self.velocity_var].sel(
                    {self.x_coord: x, self.z_coord: z},
                    method='nearest'
                )
                vp.append(float(v.values))
            except:
                vp.append(np.nan)
        
        return pd.DataFrame({
            'x': x_coords,
            'vp': vp
        })
    
    def extract_path_profile(self, path: List[Tuple[float, float]],
                            n_points: Optional[int] = None) -> pd.DataFrame:
        """沿路径提取剖面
        
        Args:
            path: 路径点列表 [(x1, z1), (x2, z2), ...]
            n_points: 插值点数（可选，默认使用路径点数）
        
        Returns:
            DataFrame: 包含distance, x, z, vp列
        """
        if len(path) < 2:
            raise ValueError("路径至少需要2个点")
        
        # 计算路径总长度
        distances = [0.0]
        total_dist = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dz = path[i][1] - path[i-1][1]
            dist = np.sqrt(dx**2 + dz**2)
            total_dist += dist
            distances.append(total_dist)
        
        # 插值点
        if n_points is None:
            n_points = len(path) * 10  # 默认每个线段10个点
        
        dist_interp = np.linspace(0, total_dist, n_points)
        
        # 插值x和z坐标
        x_interp = np.interp(dist_interp, distances, [p[0] for p in path])
        z_interp = np.interp(dist_interp, distances, [p[1] for p in path])
        
        # 提取速度值
        vp = []
        for x, z in zip(x_interp, z_interp):
            try:
                v = self.grid_data[self.velocity_var].sel(
                    {self.x_coord: x, self.z_coord: z},
                    method='nearest'
                )
                vp.append(float(v.values))
            except:
                vp.append(np.nan)
        
        return pd.DataFrame({
            'distance': dist_interp,
            'x': x_interp,
            'z': z_interp,
            'vp': vp
        })


class PropertyCalculator:
    """物性参数计算器"""

    DEFAULT_SEDIMENTARY_KEYWORDS = [
        'sediment', 'sand', 'shale', 'silt', 'mud', 'clastic', 'conglomerate',
        'limestone', 'dolomite', 'chalk', 'marl', 'greywacke', 'graywacke',
        'sedimentary',
        '沉积', '砂岩', '页岩', '泥岩', '灰岩', '白云岩', '杂砂岩',
    ]
    DEFAULT_PRIOR_CONFIG_NAME = 'imodel_priors.json'
    DEFAULT_GEOTHERMAL_GRADIENT_C_PER_KM = 30.0
    DEFAULT_SEAWATER_TEMPERATURE_C = 4.0
    DEFAULT_SEAFLOOR_TEMPERATURE_C = 4.0
    DEFAULT_SEAWATER_DENSITY_G_CM3 = 1.03
    DEFAULT_SEAWATER_PRESSURE_GRADIENT_MPA_PER_KM = 10.1
    SEDIMENT_NON_SEDIMENTARY_HINTS = [
        'ultramafic', 'mafic', 'mantle', 'peridotite', 'serpentin',
        '超镁铁', '镁铁质', '地幔', '橄榄岩', '蛇纹',
    ]
    TOMO2D_SEDIMENT_DENSITY_METHOD_ALIASES = {
        'tomo2d_sediment',
        'sed_tomo2d',
        'tomo2d-sediment',
    }
    
    def __init__(
        self,
        grid_data: xr.Dataset,
        vs_grid_data: Optional[xr.Dataset] = None,
        use_felsic_mafic: bool = False,
        use_rock_facies: bool = False,
        use_sio2: bool = False,
    ):
        """初始化物性计算器
        
        Args:
            grid_data: 主速度网格数据（默认按P波解释）
            vs_grid_data: 可选S波速度网格数据（与grid_data可不同网格，按最近邻取值）
            use_felsic_mafic: 是否启用长英质/镁铁质增强特征
            use_rock_facies: 是否启用岩相增强特征
            use_sio2: 是否启用SiO2增强特征
        """
        self.grid_data = grid_data
        self.vs_grid_data = vs_grid_data
        
        # 自动检测速度变量名
        self._detect_velocity_var()
        
        # 初始化处理器和分类器
        self.processor = GridModelProcessor()
        self.processor.velocity_grid = grid_data

        # 允许通过环境变量开启扩展特征（默认关闭，保持兼容）
        def _env_flag(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return default
            return str(raw).strip().lower() in {"1", "true", "yes", "on"}

        use_felsic_mafic = _env_flag("PYAOBS_IMODEL_USE_FELSIC_MAFIC", use_felsic_mafic)
        use_rock_facies = _env_flag("PYAOBS_IMODEL_USE_ROCK_FACIES", use_rock_facies)
        use_sio2 = _env_flag("PYAOBS_IMODEL_USE_SIO2", use_sio2)
        
        # 初始化岩石分类器（尝试加载数据库，如果失败则使用None）
        # 首先检查 SimpleRockClassifier 是否可用
        if SimpleRockClassifier is None:
            self.rock_classifier = None
            import warnings
            warnings.warn(
                "SimpleRockClassifier is not available (import failed). Rock classification will be disabled.",
                UserWarning,
                stacklevel=2
            )
        else:
            try:
                # 尝试初始化，但不立即访问identifier（避免触发train_classifier）
                self.rock_classifier = SimpleRockClassifier(
                    auto_correct=False,
                    use_felsic_mafic=use_felsic_mafic,
                    use_rock_facies=use_rock_facies,
                    use_sio2=use_sio2,
                )
            except FileNotFoundError as e:
                # 如果数据库文件不存在，使用None，分类时将返回UNKNOWN
                self.rock_classifier = None
                import warnings
                warnings.warn(
                    f"Rock classifier database not found. Rock classification will be disabled. "
                    f"To enable rock classification, please provide a rock database file in utils/rockdata "
                    f"(or pass database_file explicitly). "
                    f"Error: {str(e)}",
                    UserWarning,
                    stacklevel=2
                )
            except Exception as e:
                # 其他异常（如初始化错误）
                self.rock_classifier = None
                import warnings
                import traceback
                error_details = traceback.format_exc()
                warnings.warn(
                    f"Failed to initialize rock classifier: {str(e)}. Rock classification will be disabled.\n"
                    f"Error details: {error_details}",
                    UserWarning,
                    stacklevel=2
                )
        
        # 计算密度网格（缓存）
        self._density_grid = None
        self._density_grid_method = None
        self._prior_config = self._load_prior_config()
        self.sedimentary_keywords = self._load_sedimentary_keywords()
        self.geothermal_gradient_c_per_km = self._load_geothermal_gradient_c_per_km()
        self.seawater_temperature_c = self._load_seawater_temperature_c()
        self.seafloor_temperature_c = self._load_seafloor_temperature_c()
        self.seawater_density_g_cm3 = self._load_seawater_density_g_cm3()
        self.seawater_pressure_gradient_mpa_per_km = self._load_seawater_pressure_gradient_mpa_per_km()
        self.classifier_diagnostics = self._collect_classifier_diagnostics()

    @staticmethod
    def _to_float_safe(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _resolve_prior_config_path(self) -> Path:
        default_path = Path(__file__).resolve().parents[1] / 'utils' / 'rockdata' / self.DEFAULT_PRIOR_CONFIG_NAME
        return Path(os.environ.get('PYAOBS_IMODEL_PRIORS_FILE', str(default_path)))

    def _load_prior_config(self) -> Dict[str, Any]:
        config_path = self._resolve_prior_config_path()
        if not config_path.exists():
            return {}
        try:
            with config_path.open('r', encoding='utf-8') as f:
                raw = json.load(f)
            return raw if isinstance(raw, dict) else {}
        except Exception as exc:
            warnings.warn(f"Failed to read imodel prior config: {config_path} ({exc})")
            return {}

    def _load_sedimentary_keywords(self) -> List[str]:
        """加载沉积层候选关键词（支持外部配置）"""
        keywords = list(self.DEFAULT_SEDIMENTARY_KEYWORDS)
        config = getattr(self, "_prior_config", None) or self._load_prior_config()
        configured = config.get('sedimentary_keywords')
        if not isinstance(configured, list):
            return keywords

        normalized = []
        for kw in configured:
            text = str(kw or '').strip().lower()
            if text:
                normalized.append(text)
        return normalized or keywords

    def _load_seawater_temperature_c(self) -> float:
        config = getattr(self, "_prior_config", None) or self._load_prior_config()
        value = config.get("seawater_temperature_c", self.DEFAULT_SEAWATER_TEMPERATURE_C)
        try:
            out = float(value)
        except (TypeError, ValueError):
            out = self.DEFAULT_SEAWATER_TEMPERATURE_C

        env_raw = os.environ.get("PYAOBS_IMODEL_SEAWATER_TEMPERATURE_C")
        if env_raw is not None and str(env_raw).strip() != "":
            try:
                out = float(env_raw)
            except (TypeError, ValueError):
                pass
        # 合理范围裁剪，避免配置误填导致异常结果
        return max(-2.0, min(40.0, out))

    def _load_geothermal_gradient_c_per_km(self) -> float:
        config = getattr(self, "_prior_config", None) or self._load_prior_config()
        value = config.get("geothermal_gradient_c_per_km", self.DEFAULT_GEOTHERMAL_GRADIENT_C_PER_KM)
        try:
            out = float(value)
        except (TypeError, ValueError):
            out = self.DEFAULT_GEOTHERMAL_GRADIENT_C_PER_KM

        env_raw = os.environ.get("PYAOBS_IMODEL_GEOTHERMAL_GRADIENT_C_PER_KM")
        if env_raw is not None and str(env_raw).strip() != "":
            try:
                out = float(env_raw)
            except (TypeError, ValueError):
                pass
        return max(1.0, min(100.0, out))

    def _load_seafloor_temperature_c(self) -> float:
        config = getattr(self, "_prior_config", None) or self._load_prior_config()
        default_val = float(getattr(self, "seawater_temperature_c", self.DEFAULT_SEAFLOOR_TEMPERATURE_C))
        value = config.get("seafloor_temperature_c", default_val)
        try:
            out = float(value)
        except (TypeError, ValueError):
            out = default_val

        env_raw = os.environ.get("PYAOBS_IMODEL_SEAFLOOR_TEMPERATURE_C")
        if env_raw is not None and str(env_raw).strip() != "":
            try:
                out = float(env_raw)
            except (TypeError, ValueError):
                pass
        return max(-2.0, min(60.0, out))

    def _load_seawater_density_g_cm3(self) -> float:
        config = getattr(self, "_prior_config", None) or self._load_prior_config()
        value = config.get("seawater_density_g_cm3", self.DEFAULT_SEAWATER_DENSITY_G_CM3)
        try:
            out = float(value)
        except (TypeError, ValueError):
            out = self.DEFAULT_SEAWATER_DENSITY_G_CM3

        env_raw = os.environ.get("PYAOBS_IMODEL_SEAWATER_DENSITY_G_CM3")
        if env_raw is not None and str(env_raw).strip() != "":
            try:
                out = float(env_raw)
            except (TypeError, ValueError):
                pass
        return max(0.95, min(1.20, out))

    def _load_seawater_pressure_gradient_mpa_per_km(self) -> float:
        config = getattr(self, "_prior_config", None) or self._load_prior_config()
        value = config.get(
            "seawater_pressure_gradient_mpa_per_km",
            self.DEFAULT_SEAWATER_PRESSURE_GRADIENT_MPA_PER_KM,
        )
        try:
            out = float(value)
        except (TypeError, ValueError):
            out = self.DEFAULT_SEAWATER_PRESSURE_GRADIENT_MPA_PER_KM

        env_raw = os.environ.get("PYAOBS_IMODEL_SEAWATER_PRESSURE_GRADIENT_MPA_PER_KM")
        if env_raw is not None and str(env_raw).strip() != "":
            try:
                out = float(env_raw)
            except (TypeError, ValueError):
                pass
        return max(5.0, min(15.0, out))

    @staticmethod
    def _pick_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
        existing = {str(c).strip().lower(): c for c in df.columns}
        for c in candidates:
            hit = existing.get(str(c).strip().lower())
            if hit is not None:
                return hit
        return None

    @staticmethod
    def _read_table_for_diagnostics(path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()
        if suffix in {'.xlsx', '.xlsm'}:
            return pd.read_excel(path)
        if suffix in {'.csv', '.txt', '.tsv'}:
            sep = '\t' if suffix == '.tsv' else ','
            for enc in ('utf-8-sig', 'utf-8', 'gb18030', 'latin-1'):
                try:
                    return pd.read_csv(path, encoding=enc, sep=sep)
                except Exception:
                    continue
            return pd.read_csv(path, sep=sep)
        return pd.DataFrame()

    def _collect_classifier_diagnostics(self) -> Dict[str, Any]:
        diag: Dict[str, Any] = {
            'enabled': bool(self.rock_classifier is not None),
            'database_path': None,
            'row_count': None,
            'rock_type_count': None,
            'source_top5': [],
            'coverage': {},
            'error': None,
        }
        if self.rock_classifier is None:
            return diag

        db_path_raw = getattr(self.rock_classifier, 'database_file', None)
        db_path = Path(str(db_path_raw)) if db_path_raw else None
        diag['database_path'] = str(db_path) if db_path else None
        if db_path is None or not db_path.exists():
            diag['error'] = 'database_not_found'
            return diag

        try:
            df = self._read_table_for_diagnostics(db_path)
            if df.empty:
                diag['error'] = 'empty_database'
                return diag

            diag['row_count'] = int(len(df))

            rock_col = self._pick_column(df, ['rock_type', 'rock type', 'type', '岩石类型', '岩性'])
            if rock_col is not None:
                rock_series = df[rock_col].astype(str).str.strip()
                rock_series = rock_series[rock_series != '']
                diag['rock_type_count'] = int(rock_series.nunique())

            source_col = self._pick_column(df, ['source', 'reference', 'citation'])
            if source_col is not None:
                src = df[source_col].astype(str).str.strip()
                src = src[(src != '') & (src.str.lower() != 'nan')]
                diag['source_top5'] = [
                    {'source': str(k), 'count': int(v)}
                    for k, v in src.value_counts().head(5).items()
                ]

            for key, aliases in {
                'sio2_wt': ['sio2_wt', 'sio2', 'sio2 wt.%', 'sio2, wt.%'],
                'rock_facies': ['rock_facies', 'rock facies', 'facies'],
                'felsic_or_mafic': ['felsic_or_mafic', 'felsic or mafic'],
            }.items():
                col = self._pick_column(df, aliases)
                if col is None:
                    diag['coverage'][key] = 0.0
                    continue
                series = df[col].astype(str).str.strip()
                valid = (series != '') & (series.str.lower() != 'nan')
                diag['coverage'][key] = float(valid.mean())
        except Exception as exc:
            diag['error'] = str(exc)
        return diag

    _CANDIDATE_GEOLOGY_CN_FIELDS: tuple[tuple[str, str], ...] = (
        ("岩石属性", "rock_attributes_cn"),
        ("岩石分类", "rock_classification"),
        ("变质程度", "metamorphic_grade"),
        ("地质意义", "geological_meaning"),
        ("长英质/镁铁质", "felsic_or_mafic"),
        ("岩相", "rock_facies"),
        ("SiO₂ (wt%)", "sio2_wt"),
    )

    @classmethod
    def _geology_cn_dict_from_aux(cls, aux: Dict[str, Any]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for cn, en in cls._CANDIDATE_GEOLOGY_CN_FIELDS:
            raw = aux.get(en)
            if raw is None:
                continue
            if en == "sio2_wt":
                try:
                    fv = float(raw)
                except (TypeError, ValueError):
                    continue
                if np.isnan(fv):
                    continue
                out[cn] = f"{fv:.2f}"
                continue
            s = str(raw).strip()
            if s and s.lower() != "nan":
                out[cn] = s
        return out

    def _enrich_rock_candidates_geology_cn(
        self, rock_candidates: List[Dict[str, Any]], *, vp: float, vs: float
    ) -> List[Dict[str, Any]]:
        if not rock_candidates:
            return rock_candidates
        clf = getattr(self.rock_classifier, "classifier", None) if self.rock_classifier else None
        enriched: List[Dict[str, Any]] = []
        for c in rock_candidates:
            d = dict(c)
            rt = d.get("rock_type")
            sl = str(rt or "").strip().lower()
            if sl in {"seawater", "marine water", "water"}:
                d["geology_cn"] = {
                    "岩石分类": "水体",
                    "变质程度": "-",
                    "地质意义": "海底面以上海水层（Vp≈1.5 km/s）",
                }
                enriched.append(d)
                continue
            if clf is None or not rt:
                d.setdefault("geology_cn", {})
                enriched.append(d)
                continue
            try:
                aux = clf.get_auxiliary_attributes(str(rt), vp=vp, vs=vs) or {}
                d["geology_cn"] = self._geology_cn_dict_from_aux(aux)
            except Exception:
                d.setdefault("geology_cn", {})
            enriched.append(d)
        return enriched

    def _rerank_low_confidence_candidates(
        self,
        candidates: List[Dict[str, Any]],
        *,
        zone: str,
        vp: float,
        vs: float,
        expected_foma: Optional[str],
        expected_facies: Optional[str],
        expected_sio2: Optional[float],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        if not candidates or len(candidates) < 2:
            return candidates, False
        if self.rock_classifier is None or getattr(self.rock_classifier, 'classifier', None) is None:
            return candidates, False

        zone_key = str(zone or '').strip().lower()
        exp_foma = str(expected_foma or '').strip().lower()
        exp_facies = str(expected_facies or '').strip().lower()
        exp_sio2 = expected_sio2 if isinstance(expected_sio2, (int, float)) else None

        scored: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
        helper = self.rock_classifier.classifier

        for item in candidates:
            row = dict(item)
            rock_name = str(row.get('rock_type', '') or '').strip()
            prob = max(0.0, self._to_float_safe(row.get('probability', 0.0), 0.0))
            aux: Dict[str, Any] = {}
            if rock_name:
                try:
                    aux = helper.get_auxiliary_attributes(rock_name, vp=vp, vs=vs) or {}
                except Exception:
                    aux = {}

            score = prob
            blob = ' '.join([
                rock_name.lower(),
                str(aux.get('rock_classification', '')).lower(),
                str(aux.get('geological_meaning', '')).lower(),
            ])
            if zone_key == 'sediment' and any(k in blob for k in self.sedimentary_keywords):
                score += 0.12
            if zone_key == 'water' and 'seawater' in rock_name.lower():
                score += 0.30
            if zone_key == 'deep' and any(k in blob for k in ['deep', 'crust', 'mantle', '变质', '深部', '下地壳']):
                score += 0.04

            cand_foma = str(aux.get('felsic_or_mafic', '') or '').strip().lower()
            cand_facies = str(aux.get('rock_facies', '') or '').strip().lower()
            cand_sio2_raw = aux.get('sio2_wt')
            cand_sio2 = self._to_float_safe(cand_sio2_raw, np.nan)

            if exp_foma and cand_foma and cand_foma == exp_foma:
                score += 0.03
            if exp_facies and cand_facies and cand_facies == exp_facies:
                score += 0.02
            if exp_sio2 is not None and not np.isnan(cand_sio2):
                diff = abs(cand_sio2 - exp_sio2)
                if diff <= 8.0:
                    score += 0.02

            scored.append((score, row, aux))

        if not scored:
            return candidates, False

        scored.sort(key=lambda x: x[0], reverse=True)
        score_sum = sum(max(0.0, s) for s, _, _ in scored)
        reranked: List[Dict[str, Any]] = []
        for s, row, _ in scored:
            out = dict(row)
            out['probability'] = (max(0.0, s) / score_sum) if score_sum > 0 else 0.0
            reranked.append(out)

        changed = [c.get('rock_type') for c in reranked] != [c.get('rock_type') for c in candidates]
        return reranked, changed

    @staticmethod
    def _norm_name(text: Any) -> str:
        return str(text or '').strip().lower()

    def _is_sedimentary_record(self, rock_type: Any, rock_classification: Any, geological_meaning: Any) -> bool:
        blob = " ".join([
            self._norm_name(rock_type),
            self._norm_name(rock_classification),
            self._norm_name(geological_meaning),
        ])
        return any(k in blob for k in self.sedimentary_keywords)

    def _get_zone_allowed_types(self, zone: str) -> set[str]:
        helper = getattr(self.rock_classifier, 'classifier', None) if self.rock_classifier is not None else None
        ref = getattr(helper, 'rock_database_reference', None)
        if ref is None or not isinstance(ref, pd.DataFrame) or ref.empty or 'rock_type' not in ref.columns:
            return set()

        zone_key = self._norm_name(zone)
        if zone_key == 'water':
            return {'seawater', 'water'}

        allowed: set[str] = set()
        for _, row in ref.iterrows():
            rock = str(row.get('rock_type', '')).strip()
            if not rock:
                continue
            cls = row.get('岩石分类', '')
            meaning = row.get('地质意义', '')
            is_sed = self._is_sedimentary_record(rock, cls, meaning)
            if zone_key == 'sediment' and is_sed:
                allowed.add(self._norm_name(rock))
            elif zone_key == 'deep' and not is_sed and self._norm_name(rock) not in {'seawater', 'water'}:
                allowed.add(self._norm_name(rock))
        return allowed

    def _classify_with_zone_prior(
        self,
        *,
        zone: str,
        vp: float,
        vs: float,
        density: float,
        pressure: float,
        temperature: float,
    ) -> Dict[str, Any] | None:
        if self.rock_classifier is None:
            return None
        helper = getattr(self.rock_classifier, 'classifier', None)
        if helper is None:
            return None

        zone_key = self._norm_name(zone)
        allowed_types = self._get_zone_allowed_types(zone_key)
        if zone_key in {'sediment', 'deep'} and not allowed_types:
            return None

        try:
            vp_corrected = helper.correct_velocity(
                vp,
                pressure=pressure,
                temperature=temperature,
                is_s_wave=False,
                target_pressure=200.0,
                target_temperature=25.0,
            )
            vs_corrected = helper.correct_velocity(
                vs,
                pressure=pressure,
                temperature=temperature,
                is_s_wave=True,
                target_pressure=200.0,
                target_temperature=25.0,
            )
            probs = helper.classify_probabilities_by_features(
                vp=vp_corrected,
                vs=vs_corrected,
                density=density,
                felsic_or_mafic=None,
                rock_facies=None,
                sio2_wt=None,
            )
        except Exception:
            return None

        if not probs:
            return None

        if zone_key in {'sediment', 'deep'}:
            zone_probs = {
                k: v for k, v in probs.items()
                if self._norm_name(k) in allowed_types
            }
            if zone_probs:
                probs = zone_probs
            else:
                return None

        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        if not sorted_probs:
            return None
        top_type, top_prob = sorted_probs[0]
        prob_sum = sum(max(0.0, float(v)) for _, v in sorted_probs[:5])
        candidates = []
        for k, v in sorted_probs[:5]:
            p = max(0.0, float(v))
            p = (p / prob_sum) if prob_sum > 0 else p
            candidates.append({'rock_type': str(k), 'probability': p})

        aux = helper.get_auxiliary_attributes(top_type, vp=vp_corrected, vs=vs_corrected) or {}
        return {
            'rock_type': str(top_type),
            'probability': float(top_prob),
            'all_candidates': candidates,
            'felsic_or_mafic': aux.get('felsic_or_mafic'),
            'rock_facies': aux.get('rock_facies'),
            'sio2_wt': aux.get('sio2_wt'),
            'rock_classification': aux.get('rock_classification'),
            'metamorphic_grade': aux.get('metamorphic_grade'),
            'geological_meaning': aux.get('geological_meaning'),
            'zone_prior_applied': zone_key in {'sediment', 'deep'},
        }

    @staticmethod
    def is_water_zone(vp: float, z: float, seafloor_depth: Optional[float]) -> bool:
        """统一海水层判据：位于海底面以上且 Vp 近似海水速度。"""
        seafloor_z = seafloor_depth if seafloor_depth is not None else 0.0
        return float(z) < float(seafloor_z) and abs(float(vp) - 1.5) <= 0.25

    def compute_total_pressure(
        self,
        z: float,
        *,
        seafloor_depth: Optional[float],
        is_water: bool,
        rock_pressure_gradient: float,
    ) -> float:
        """
        统一压力口径（总压力）：
        - 水层：按海平面到当前点的静水压；
        - 海底以下：海水柱静水压 + 海底以下岩层增量压力。
        """
        z_val = float(z)
        seafloor_z = float(seafloor_depth if seafloor_depth is not None else 0.0)
        water_col_km = max(0.0, seafloor_z)
        hydro_at_seafloor = water_col_km * self.seawater_pressure_gradient_mpa_per_km

        if is_water:
            return max(0.0, z_val) * self.seawater_pressure_gradient_mpa_per_km

        rock_increment = calculate_pressure_from_depth(
            z_val,
            pressure_gradient=rock_pressure_gradient,
            seafloor_depth=seafloor_z,
        )
        return hydro_at_seafloor + float(rock_increment)

    def _apply_zone_semantic_constraints(
        self,
        *,
        zone: str,
        rock_type: str,
        felsic_or_mafic: Any,
        rock_facies: Any,
        sio2_wt: Any,
        rock_classification: Any,
        metamorphic_grade: Any,
        geological_meaning: Any,
    ) -> Tuple[Any, Any, Any, Any, Any, Any, bool]:
        """
        对 zone 进行语义一致性约束，避免“沉积层 + 超镁铁质蚀变解释”这类冲突输出。
        """
        zone_key = str(zone or '').strip().lower()
        changed = False

        if zone_key == 'water':
            if str(rock_classification or '').strip() != '水体':
                rock_classification = '水体'
                changed = True
            if str(metamorphic_grade or '').strip() not in {'', '-'}:
                metamorphic_grade = '-'
                changed = True
            if str(geological_meaning or '').strip() == '':
                geological_meaning = '海底面以上海水层（Vp≈1.5 km/s）'
                changed = True
            return (
                felsic_or_mafic,
                rock_facies,
                sio2_wt,
                rock_classification,
                metamorphic_grade,
                geological_meaning,
                changed,
            )

        if zone_key != 'sediment':
            return (
                felsic_or_mafic,
                rock_facies,
                sio2_wt,
                rock_classification,
                metamorphic_grade,
                geological_meaning,
                changed,
            )

        sediment_blob = ' '.join([
            str(rock_type or '').lower(),
            str(rock_classification or '').lower(),
            str(geological_meaning or '').lower(),
        ])
        has_conflict_hint = any(k in sediment_blob for k in self.SEDIMENT_NON_SEDIMENTARY_HINTS)

        if has_conflict_hint or not str(rock_classification or '').strip():
            if str(rock_classification or '').strip() != '沉积岩':
                rock_classification = '沉积岩'
                changed = True

        foma_text = str(felsic_or_mafic or '').strip().lower()
        if (not foma_text) or any(k in foma_text for k in self.SEDIMENT_NON_SEDIMENTARY_HINTS):
            if str(felsic_or_mafic or '').strip() != 'sedimentary':
                felsic_or_mafic = 'sedimentary'
                changed = True

        facies_text = str(rock_facies or '').strip()
        if not facies_text or facies_text.lower() in {'unknown', 'nan', '-'}:
            rock_facies = 'sedimentary'
            changed = True

        meta_text = str(metamorphic_grade or '').strip().lower()
        if not meta_text or any(k in meta_text for k in ['蛇纹', '变质', 'metamorph', 'serpentin']):
            if str(metamorphic_grade or '').strip() != '-':
                metamorphic_grade = '-'
                changed = True

        meaning_text = str(geological_meaning or '').strip().lower()
        if (not meaning_text) or any(k in meaning_text for k in self.SEDIMENT_NON_SEDIMENTARY_HINTS):
            geological_meaning = '海底面与基底面之间沉积层先验约束'
            changed = True

        return (
            felsic_or_mafic,
            rock_facies,
            sio2_wt,
            rock_classification,
            metamorphic_grade,
            geological_meaning,
            changed,
        )
    
    def _detect_velocity_var(self):
        """自动检测速度变量名"""
        self.velocity_var = self._detect_velocity_var_for_dataset(self.grid_data)
        self.x_coord, self.z_coord = self._detect_coords_for_dataset(self.grid_data)
        self.vs_velocity_var = None
        self.vs_x_coord = None
        self.vs_z_coord = None
        if self.vs_grid_data is not None:
            self.vs_velocity_var = self._detect_velocity_var_for_dataset(self.vs_grid_data)
            self.vs_x_coord, self.vs_z_coord = self._detect_coords_for_dataset(self.vs_grid_data)

    @staticmethod
    def _detect_velocity_var_for_dataset(dataset: xr.Dataset) -> str:
        data_vars = list(dataset.data_vars)
        velocity_candidates = ['velocity', 'v', 'vp', 'v_p', 'vel', 'speed', 'vs', 'v_s']
        for var in velocity_candidates:
            if var in data_vars:
                return var
        if 'z' in data_vars:
            return 'z'
        return data_vars[0] if data_vars else 'velocity'

    @staticmethod
    def _detect_coords_for_dataset(dataset: xr.Dataset) -> Tuple[str, str]:
        coords = list(dataset.coords)
        x_coord = next(
            (c for c in coords if c.lower() in ['x', 'lon', 'longitude', 'distance']),
            coords[0] if coords else 'x',
        )
        z_coord = next(
            (c for c in coords if c.lower() in ['z', 'depth', 'y', 't', 'time']),
            coords[1] if len(coords) > 1 else 'z',
        )
        return x_coord, z_coord

    @staticmethod
    def _sample_dataset_value(
        dataset: xr.Dataset,
        value_var: str,
        x_coord: str,
        z_coord: str,
        x: float,
        z: float,
    ) -> float:
        try:
            return float(dataset[value_var].sel({x_coord: x, z_coord: z}, method='nearest').values)
        except Exception:
            x_coords = np.asarray(dataset[x_coord].values, dtype=float)
            z_coords = np.asarray(dataset[z_coord].values, dtype=float)
            x_idx = int(np.argmin(np.abs(x_coords - x)))
            z_idx = int(np.argmin(np.abs(z_coords - z)))
            # 不能用 values[z,z,x]：numpy 轴顺序取决于 DataArray.dims，
            # 与 (depth, horizontal) 约定无关；isel 才能与坐标名对齐。
            sampled = dataset[value_var].isel({x_coord: x_idx, z_coord: z_idx}).values
            return float(np.asarray(sampled).item())

    def get_vp_vs_at_point(
        self,
        x: float,
        z: float,
        *,
        velocity_method: str = 'brocher',
    ) -> Tuple[float, float, str]:
        """
        获取指定点的Vp/Vs。
        返回: (vp, vs, vs_source) 其中 vs_source 为 'model' 或 'empirical'
        """
        vp = self._sample_dataset_value(
            self.grid_data,
            self.velocity_var,
            self.x_coord,
            self.z_coord,
            x,
            z,
        )
        if self.vs_grid_data is not None and self.vs_velocity_var and self.vs_x_coord and self.vs_z_coord:
            vs = self._sample_dataset_value(
                self.vs_grid_data,
                self.vs_velocity_var,
                self.vs_x_coord,
                self.vs_z_coord,
                x,
                z,
            )
            return vp, vs, 'model'
        vs = calculate_vs(vp, method=velocity_method)
        return vp, vs, 'empirical'

    @classmethod
    def _normalize_density_method(cls, method: str) -> str:
        return str(method or 'gardner').strip().lower()

    @classmethod
    def uses_tomo2d_sediment_density(cls, method: str) -> bool:
        return cls._normalize_density_method(method) in cls.TOMO2D_SEDIMENT_DENSITY_METHOD_ALIASES

    @staticmethod
    def tomo2d_sediment_density_from_vp(vp: float) -> float:
        """
        tomo2d 的沉积层速度-密度关系（jgrav.cc::sed_vel2rho, mode=1）。
        输入/输出单位：vp 为 km/s，rho 为 g/cm^3。
        """
        vp_val = float(vp)
        if vp_val <= 1.5:
            rho = 1.0
        else:
            rho = 1.0 + 1.18 * pow(max(vp_val - 1.5, 0.0), 0.22)
        return min(rho, 2.6)
    
    def calculate_density_grid(self, method: str = 'gardner') -> xr.Dataset:
        """计算密度网格（缓存结果）"""
        method_key = self._normalize_density_method(method)
        if self._density_grid is None or self._density_grid_method != method_key:
            self._density_grid = self.processor.velocity_to_density(method=method)
            self._density_grid_method = method_key
        return self._density_grid
    
    def calculate_all_properties(self, 
                               x: float, 
                               z: float,
                               density_method: str = 'gardner',
                               pressure_gradient: float = 30.0,
                               temperature_gradient: Optional[float] = None,
                               seafloor_depth: Optional[float] = None,
                               basement_depth: Optional[float] = None) -> Dict:
        """计算指定位置的所有物性参数
        
        Args:
            x: x坐标
            z: 深度坐标
            density_method: 密度计算方法
            pressure_gradient: 压力梯度 (MPa/km)
            temperature_gradient: 地温梯度 (°C/km)，None时使用配置默认值
            seafloor_depth: 海底面深度（用于温压起算与海水判别）
            basement_depth: 基底面深度（用于沉积层先验约束）
        
        Returns:
            包含所有物性参数的字典
        """
        # 1. 获取P波、S波速度（优先使用双模型中的真实Vs）
        if not hasattr(self, 'velocity_var'):
            self._detect_velocity_var()
        vp, vs, vs_source = self.get_vp_vs_at_point(x, z)
        
        density_method_key = self._normalize_density_method(density_method)
        use_tomo2d_sed_density = self.uses_tomo2d_sediment_density(density_method_key)

        # 2. 计算分区（Zone-first），供密度与后续分类共用
        # 如果提供了海底面深度，使用它；否则使用简单的检测方法
        if seafloor_depth is not None:
            seafloor_z = seafloor_depth
        else:
            # 简单检测：如果vp < 1.6，假设是海水层，海底面深度为z
            # 否则，假设海底面深度为0（从模型顶部开始）
            seafloor_z = 0.0
            if vp < 1.6:
                # 海水层，假设海底面在当前深度之上
                seafloor_z = z
        
        # 5.0 先确定分区（Zone-first）
        # - 海底面以上且Vp~1.5: 水层
        # - 海底面与基底面之间: 沉积层
        seafloor_z_for_rule = seafloor_depth if seafloor_depth is not None else 0.0
        basement_z_for_rule = basement_depth
        if basement_z_for_rule is not None and basement_z_for_rule <= seafloor_z_for_rule:
            basement_z_for_rule = None

        in_water = self.is_water_zone(vp=vp, z=z, seafloor_depth=seafloor_z_for_rule)
        in_sediment = (
            basement_z_for_rule is not None
            and z >= seafloor_z_for_rule
            and z < basement_z_for_rule
        )
        zone = 'deep'
        if in_water:
            zone = 'water'
        elif in_sediment:
            zone = 'sediment'

        # 3. 计算密度（考虑海水层/沉积层特例）
        if in_water:
            density = self.seawater_density_g_cm3
        elif use_tomo2d_sed_density and in_sediment:
            density = self.tomo2d_sediment_density_from_vp(vp)
        else:
            density_method_for_library = 'gardner' if use_tomo2d_sed_density else density_method_key
            try:
                density_grid = self.calculate_density_grid(method=density_method_for_library)
                density = float(density_grid['density'].sel(
                    {self.x_coord: x, self.z_coord: z}, method='nearest'
                ).values)
            except Exception:
                density = calculate_density(vp, method=density_method_for_library)
                if vp < 1.6:
                    density = self.seawater_density_g_cm3

        # 4. 计算压力和温度（从海底面开始计算）
        temp_gradient_val = (
            self.geothermal_gradient_c_per_km
            if temperature_gradient is None
            else float(temperature_gradient)
        )
        pressure = calculate_pressure_from_depth(z, pressure_gradient=pressure_gradient, seafloor_depth=seafloor_z)
        temperature = calculate_temperature_from_depth(
            z,
            temperature_gradient=temp_gradient_val,
            surface_temperature=self.seafloor_temperature_c,
            seafloor_depth=seafloor_z,
        )

        # 5.1 在 Zone 对应样本空间内进行分类（先样本、后分类）
        # 注意：分类器会自动将速度值校正到标准条件（200MPa, 25°C）
        rock_type = 'UNKNOWN'
        rock_prob = 0.0
        rock_candidates = []
        felsic_or_mafic = None
        rock_facies = None
        sio2_wt = None
        rock_classification = None
        metamorphic_grade = None
        geological_meaning = None
        zone_prior_applied = False

        if in_water:
            rock_type = 'Seawater'
            rock_prob = 1.0
            rock_candidates = [{'rock_type': 'Seawater', 'probability': 1.0}]
            # 海水层采用流体专用物性：Vs=0、海水密度、静水压
            vs = 0.0
            density = self.seawater_density_g_cm3
            water_depth_km = max(0.0, float(z))
            pressure = water_depth_km * self.seawater_pressure_gradient_mpa_per_km
            temperature = self.seawater_temperature_c
            felsic_or_mafic = 'water'
            rock_facies = 'marine water'
            rock_classification = '水体'
            metamorphic_grade = '-'
            geological_meaning = '海底面以上海水层（Vp≈1.5 km/s）'
        else:
            rock_result: Dict[str, Any] | None = None
            try:
                rock_result = self._classify_with_zone_prior(
                    zone=zone,
                    vp=vp,
                    vs=vs,
                    density=density,
                    pressure=pressure,
                    temperature=temperature,
                )
                if rock_result is not None:
                    zone_prior_applied = bool(rock_result.get('zone_prior_applied', False))
            except Exception:
                rock_result = None

            if rock_result is None and self.rock_classifier is not None:
                try:
                    rock_result = self.rock_classifier.classify(
                        vp=vp,
                        vs=vs,
                        density=density,
                        pressure=pressure,
                        temperature=temperature,
                        return_probabilities=True,
                    )
                except Exception as e:
                    warnings.warn(f"Rock classification failed: {e}")
                    rock_result = None

            if isinstance(rock_result, dict):
                rock_type = rock_result.get('rock_type', 'UNKNOWN')
                rock_prob = rock_result.get('probability', 0.0)
                felsic_or_mafic = rock_result.get('felsic_or_mafic')
                rock_facies = rock_result.get('rock_facies')
                sio2_wt = rock_result.get('sio2_wt')
                rock_classification = rock_result.get('rock_classification')
                metamorphic_grade = rock_result.get('metamorphic_grade')
                geological_meaning = rock_result.get('geological_meaning')
                all_candidates = rock_result.get('all_candidates', [])
                if all_candidates:
                    rock_candidates = sorted(
                        all_candidates,
                        key=lambda x: x.get('probability', 0.0),
                        reverse=True,
                    )[:5]
            elif isinstance(rock_result, str):
                rock_type = rock_result
                rock_prob = 1.0

            if zone == 'sediment':
                def _is_sedimentary_name(name: Any) -> bool:
                    s = str(name or '').lower()
                    return any(k in s for k in self.sedimentary_keywords)

                filtered_candidates = [
                    c for c in rock_candidates
                    if _is_sedimentary_name(c.get('rock_type'))
                ]
                if filtered_candidates:
                    rock_candidates = filtered_candidates[:5]
                    rock_type = str(rock_candidates[0].get('rock_type', rock_type))
                    try:
                        rock_prob = float(rock_candidates[0].get('probability', rock_prob) or rock_prob)
                    except (TypeError, ValueError):
                        pass
                elif not _is_sedimentary_name(rock_type):
                    if vp < 2.2:
                        rock_type = 'Unconsolidated sediment'
                    elif vp < 3.8:
                        rock_type = 'Sandstone/Shale'
                    elif vp < 5.2:
                        rock_type = 'Carbonate sedimentary rock'
                    else:
                        rock_type = 'Sedimentary rock'
                    rock_prob = max(float(rock_prob or 0.0), 0.55)
                    rock_candidates = [{'rock_type': rock_type, 'probability': rock_prob}]

                if not rock_classification:
                    rock_classification = '沉积岩'
                if not geological_meaning:
                    geological_meaning = '海底面与基底面之间沉积层先验约束'

            # 非水层总压力 = 海水柱静水压 + 海底以下增量压力
            pressure = self.compute_total_pressure(
                z=float(z),
                seafloor_depth=seafloor_z_for_rule,
                is_water=False,
                rock_pressure_gradient=pressure_gradient,
            )

        # 5.1 置信度评估（用于 GUI 提示与导出）
        top1_prob = 0.0
        top2_prob = 0.0
        if rock_candidates:
            try:
                top1_prob = float(rock_candidates[0].get('probability', 0.0) or 0.0)
            except (TypeError, ValueError, AttributeError):
                top1_prob = 0.0
            if len(rock_candidates) > 1:
                try:
                    top2_prob = float(rock_candidates[1].get('probability', 0.0) or 0.0)
                except (TypeError, ValueError, AttributeError):
                    top2_prob = 0.0
        else:
            try:
                top1_prob = float(rock_prob or 0.0)
            except (TypeError, ValueError):
                top1_prob = 0.0
        confidence_gap = max(0.0, top1_prob - top2_prob)
        # 经验阈值：top1<0.40 或 top1-top2<0.10 视为低置信度
        low_confidence = top1_prob < 0.40 or confidence_gap < 0.10
        confidence_level = 'low' if low_confidence else 'normal'
        rerank_applied = False
        zone_constraint_applied = False

        if low_confidence and len(rock_candidates) > 1:
            reranked, changed = self._rerank_low_confidence_candidates(
                rock_candidates,
                zone=zone,
                vp=vp,
                vs=vs,
                expected_foma=felsic_or_mafic,
                expected_facies=rock_facies,
                expected_sio2=sio2_wt,
            )
            if changed and reranked:
                rock_candidates = reranked[:5]
                rerank_applied = True
                rock_type = str(rock_candidates[0].get('rock_type', rock_type))
                rock_prob = self._to_float_safe(rock_candidates[0].get('probability', rock_prob), rock_prob)
                if self.rock_classifier is not None and getattr(self.rock_classifier, 'classifier', None) is not None:
                    try:
                        aux = self.rock_classifier.classifier.get_auxiliary_attributes(rock_type, vp=vp, vs=vs) or {}
                        felsic_or_mafic = aux.get('felsic_or_mafic', felsic_or_mafic)
                        rock_facies = aux.get('rock_facies', rock_facies)
                        sio2_wt = aux.get('sio2_wt', sio2_wt)
                        rock_classification = aux.get('rock_classification', rock_classification)
                        metamorphic_grade = aux.get('metamorphic_grade', metamorphic_grade)
                        geological_meaning = aux.get('geological_meaning', geological_meaning)
                    except Exception:
                        pass

                try:
                    top1_prob = float(rock_candidates[0].get('probability', 0.0) or 0.0)
                except (TypeError, ValueError, AttributeError):
                    top1_prob = 0.0
                try:
                    top2_prob = float(rock_candidates[1].get('probability', 0.0) or 0.0) if len(rock_candidates) > 1 else 0.0
                except (TypeError, ValueError, AttributeError):
                    top2_prob = 0.0
                confidence_gap = max(0.0, top1_prob - top2_prob)
                low_confidence = top1_prob < 0.40 or confidence_gap < 0.10
                confidence_level = 'low' if low_confidence else 'normal'

        (
            felsic_or_mafic,
            rock_facies,
            sio2_wt,
            rock_classification,
            metamorphic_grade,
            geological_meaning,
            zone_constraint_applied,
        ) = self._apply_zone_semantic_constraints(
            zone=zone,
            rock_type=rock_type,
            felsic_or_mafic=felsic_or_mafic,
            rock_facies=rock_facies,
            sio2_wt=sio2_wt,
            rock_classification=rock_classification,
            metamorphic_grade=metamorphic_grade,
            geological_meaning=geological_meaning,
        )

        rock_candidates = self._enrich_rock_candidates_geology_cn(
            rock_candidates, vp=float(vp), vs=float(vs)
        )

        # 6. 计算弹性参数
        rock = Rock('temp', RockProperties(vp=vp, vs=vs, density=density))
        moduli = rock.calculate_elastic_moduli()
        
        return {
            'x': x,
            'z': z,
            'vp': vp,
            'vs': vs,
            'vs_source': vs_source,
            'vp_vs_ratio': (float(vp) / float(vs)) if float(vs) > 0 else np.nan,
            'density': density,
            'pressure': pressure,
            'temperature': temperature,
            'zone': zone,
            'rock_type': rock_type,
            'rock_probability': rock_prob,
            'felsic_or_mafic': felsic_or_mafic,
            'rock_facies': rock_facies,
            'sio2_wt': sio2_wt,
            'rock_classification': rock_classification,
            'metamorphic_grade': metamorphic_grade,
            'geological_meaning': geological_meaning,
            'rock_candidates': rock_candidates,  # 添加候选列表
            'confidence_level': confidence_level,
            'low_confidence': low_confidence,
            'confidence_top1': top1_prob,
            'confidence_top2': top2_prob,
            'confidence_gap': confidence_gap,
            'rerank_applied': rerank_applied,
            'zone_constraint_applied': zone_constraint_applied,
            'zone_prior_applied': zone_prior_applied,
            'bulk_modulus': moduli['bulk_modulus'],
            'shear_modulus': moduli['shear_modulus'],
            'young_modulus': moduli['young_modulus'],
            'poisson_ratio': rock.poisson_ratio
        }
    
    def calculate_properties_for_profile(self, 
                                        profile: pd.DataFrame,
                                        **kwargs) -> pd.DataFrame:
        """为剖面计算所有物性参数
        
        Args:
            profile: 剖面DataFrame，必须包含x和z列（或depth列）
            **kwargs: 传递给calculate_all_properties的参数
        
        Returns:
            包含所有物性参数的DataFrame
        """
        results = []
        
        # 确定坐标列
        if 'x' in profile.columns and 'z' in profile.columns:
            x_col, z_col = 'x', 'z'
        elif 'x' in profile.columns and 'depth' in profile.columns:
            x_col, z_col = 'x', 'depth'
        else:
            raise ValueError("剖面DataFrame必须包含x和z（或depth）列")
        
        for _, row in profile.iterrows():
            props = self.calculate_all_properties(
                row[x_col], row[z_col], **kwargs
            )
            results.append(props)
        
        return pd.DataFrame(results)


class GravityCalculator:
    """重力异常计算器"""
    
    G = 6.67430e-11  # 万有引力常数 (m³/(kg·s²))
    SI_TO_MGAL = 1.0e5
    
    def calculate_bouguer_anomaly(self,
                                  density_profile: pd.DataFrame,
                                  reference_density: float = 2.67) -> np.ndarray:
        """计算Bouguer重力异常
        
        Args:
            density_profile: 密度剖面DataFrame，包含depth和density列
            reference_density: 参考密度 (g/cm³)
        
        Returns:
            重力异常数组 (mGal)
        """
        depth = density_profile['depth'].values * 1000  # 转换为米
        density = density_profile['density'].values * 1000  # 转换为kg/m³
        ref_density = reference_density * 1000  # 转换为kg/m³
        
        delta_rho = density - ref_density
        
        # Bouguer板公式: Δg = 2πG × Δρ × h
        # 转换为mGal: 1 mGal = 1e-5 m/s²
        gravity_anomaly = 2 * np.pi * self.G * delta_rho * depth * 1e5
        
        return gravity_anomaly
    
    def calculate_free_air_anomaly(self,
                                   depth_profile: np.ndarray,
                                   elevation: float = 0.0) -> np.ndarray:
        """计算自由空气重力异常
        
        Args:
            depth_profile: 深度数组 (km)
            elevation: 海拔高度 (km)
        
        Returns:
            自由空气异常 (mGal)
        """
        # 自由空气校正: 0.3086 mGal/m
        free_air_correction = -0.3086 * elevation * 1000  # 转换为mGal
        
        return np.full_like(depth_profile, free_air_correction)

    @staticmethod
    def _compute_cell_edges(coords: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=float)
        if coords.size == 0:
            return np.asarray([0.0, 1.0], dtype=float)
        if coords.size == 1:
            c = float(coords[0])
            return np.asarray([c - 0.5, c + 0.5], dtype=float)
        mids = 0.5 * (coords[:-1] + coords[1:])
        first = coords[0] - 0.5 * (coords[1] - coords[0])
        last = coords[-1] + 0.5 * (coords[-1] - coords[-2])
        return np.concatenate([[first], mids, [last]])

    def calculate_tomo2d_fft_anomaly(
        self,
        x_coords_km: np.ndarray,
        z_coords_km: np.ndarray,
        density_contrast_kg_m3: np.ndarray,
        significant_mask: np.ndarray,
        x_obs_km: np.ndarray,
        obs_level_km: float = 0.0,
        ref_x_range_km: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        基于 tomo2d(jgrav.cc) 的 2D 频域积分思路计算重力异常。
        """
        x_coords_km = np.asarray(x_coords_km, dtype=float)
        z_coords_km = np.asarray(z_coords_km, dtype=float)
        x_obs_km = np.asarray(x_obs_km, dtype=float)
        density = np.asarray(density_contrast_kg_m3, dtype=float).copy()
        mask = np.asarray(significant_mask, dtype=bool)
        density[~mask] = 0.0

        if density.ndim != 2:
            raise ValueError("density_contrast_kg_m3 必须为二维数组")
        nz, nx = density.shape
        if nx < 2 or z_coords_km.size < 1 or x_obs_km.size < 1:
            return np.zeros_like(x_obs_km, dtype=float)

        x_m = x_coords_km * 1000.0
        z_m = z_coords_km * 1000.0
        x_obs_m = x_obs_km * 1000.0
        z_obs_m = float(obs_level_km) * 1000.0

        dx = float(np.nanmean(np.diff(x_m)))
        if not np.isfinite(dx) or abs(dx) < 1e-9:
            raise ValueError("x 坐标间距无效，无法执行 tomo2d FFT 重力计算")

        if z_m.size >= 2:
            dz = abs(float(np.nanmean(np.diff(z_m))))
        else:
            dz = max(abs(dx), 1.0)
        dz = max(dz, 1.0)

        nfft = 1 << int(np.ceil(np.log2(max(4 * nx, 2))))
        hnfft = nfft // 2
        npad = max(0, (hnfft - nx) // 2)

        rhofft = np.zeros((nz, nfft), dtype=float)
        for k in range(nz):
            row = density[k]
            half = rhofft[k, :hnfft]
            half[:npad] = row[0]
            half[npad:npad + nx] = row
            half[npad + nx:] = row[-1]
            rhofft[k, hnfft:] = half[::-1]

        kx = 2.0 * np.pi * np.fft.rfftfreq(nfft, d=dx)
        coeff = np.zeros_like(kx)
        nz_k = kx > 0.0
        coeff[nz_k] = 2.0 * np.pi * self.G * (1.0 - np.exp(-kx[nz_k] * dz)) / kx[nz_k]
        coeff[0] = 2.0 * np.pi * self.G * dz

        grav_grid = np.zeros(nfft, dtype=float)
        for k in range(nz):
            tmpz = float(z_m[k] - z_obs_m)
            if tmpz < 0.0:
                continue
            atten = np.exp(-kx * tmpz) * coeff
            spec = np.fft.rfft(rhofft[k]) * atten
            grav_grid += np.fft.irfft(spec, n=nfft)

        # 与 jgrav.cc 一致：先在参考区间去均值再采样
        grav_model = grav_grid[npad:npad + nx]
        if ref_x_range_km is None:
            ref_x0, ref_x1 = float(x_coords_km[0]), float(x_coords_km[-1])
        else:
            ref_x0, ref_x1 = float(ref_x_range_km[0]), float(ref_x_range_km[1])
        ref_mask = (x_coords_km >= min(ref_x0, ref_x1)) & (x_coords_km <= max(ref_x0, ref_x1))
        if np.any(ref_mask):
            grav_model = grav_model - float(np.nanmean(grav_model[ref_mask]))
        else:
            grav_model = grav_model - float(np.nanmean(grav_model))

        gravity_obs = np.interp(
            x_obs_m,
            x_m,
            grav_model,
            left=float(grav_model[0]),
            right=float(grav_model[-1]),
        )
        return gravity_obs * self.SI_TO_MGAL

    def calculate_talwani_grid_anomaly(
        self,
        x_coords_km: np.ndarray,
        z_coords_km: np.ndarray,
        density_contrast_kg_m3: np.ndarray,
        significant_mask: np.ndarray,
        x_obs_km: np.ndarray,
        obs_level_km: float = 0.0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """
        将每个有效网格作为矩形体，按 Talwani 公式逐体叠加。
        """
        try:
            from pyAOBS.utils.gravity_talwani import talwani2d_gravity
        except Exception:
            from utils.gravity_talwani import talwani2d_gravity

        x_coords_km = np.asarray(x_coords_km, dtype=float)
        z_coords_km = np.asarray(z_coords_km, dtype=float)
        x_obs_m = np.asarray(x_obs_km, dtype=float) * 1000.0
        obs_level_m = float(obs_level_km) * 1000.0
        density = np.asarray(density_contrast_kg_m3, dtype=float)
        mask = np.asarray(significant_mask, dtype=bool)

        x_edges_m = self._compute_cell_edges(x_coords_km) * 1000.0
        z_edges_m = self._compute_cell_edges(z_coords_km) * 1000.0

        valid_cells: List[Tuple[np.ndarray, np.ndarray, float]] = []
        nz, nx = density.shape
        for i in range(nz):
            for j in range(nx):
                if not mask[i, j]:
                    continue
                x_left = float(min(x_edges_m[j], x_edges_m[j + 1]))
                x_right = float(max(x_edges_m[j], x_edges_m[j + 1]))
                z_top = float(min(z_edges_m[i], z_edges_m[i + 1]))
                z_bottom = float(max(z_edges_m[i], z_edges_m[i + 1]))
                rect_x = np.array([x_left, x_right, x_right, x_left, x_left], dtype=float)
                rect_z = np.array([z_top, z_top, z_bottom, z_bottom, z_top], dtype=float)
                valid_cells.append((rect_x, rect_z, float(density[i, j])))

        gravity_anomaly = np.zeros_like(x_obs_m, dtype=float)
        for obs_idx, x0_m in enumerate(x_obs_m):
            if progress_callback is not None:
                try:
                    progress_callback(obs_idx + 1, len(x_obs_m))
                except Exception:
                    pass
            total_gravity = 0.0
            for rect_x, rect_z, rho in valid_cells:
                try:
                    total_gravity += talwani2d_gravity(
                        rect_x, rect_z, float(x0_m), obs_level_m, rho, check_vertex=False
                    )
                except Exception:
                    continue
            gravity_anomaly[obs_idx] = total_gravity
        return gravity_anomaly


class InteractiveModelViewer:
    """交互式速度模型查看器 - 主类"""
    
    def __init__(self, 
                 grid_file: Optional[str] = None,
                 grid_data: Optional[xr.Dataset] = None,
                 figsize: Tuple[float, float] = (12, 8)):
        """初始化交互式模型查看器
        
        Args:
            grid_file: 速度模型grid文件路径
            grid_data: 或直接提供xarray Dataset
            figsize: 图形大小
        """
        # 加载数据
        if grid_data is not None:
            self.grid_data = grid_data
        elif grid_file is not None:
            self.processor = GridModelProcessor(grid_file=grid_file)
            self.grid_data = self.processor.velocity_grid
        else:
            raise ValueError("必须提供grid_file或grid_data")
        
        # 初始化组件
        self.profile_extractor = ProfileExtractor(self.grid_data)
        self.property_calculator = PropertyCalculator(self.grid_data)
        self.gravity_calculator = GravityCalculator()
        
        # 交互工具
        self.point_selector = None
        self.polygon_selector = None
        
        # 创建图形
        self.fig = plt.figure(figsize=figsize)
        self.ax_main = self.fig.add_subplot(111)
        
        # 存储结果
        self.selected_points = []
        self.selected_polygons = []
        self.profiles = {}
        self.property_results = {}
    
    def plot_model(self, 
                   cmap: str = 'viridis',
                   vmin: Optional[float] = None,
                   vmax: Optional[float] = None,
                   show_colorbar: bool = True):
        """绘制速度模型
        
        Args:
            cmap: 颜色映射
            vmin: 速度最小值
            vmax: 速度最大值
            show_colorbar: 是否显示颜色条
        """
        # 获取速度数据（自动检测速度变量名）
        # 首先尝试从ProfileExtractor获取速度变量名
        if hasattr(self, 'profile_extractor') and hasattr(self.profile_extractor, 'velocity_var'):
            velocity_var = self.profile_extractor.velocity_var
        else:
            # 自动检测
            data_vars = list(self.grid_data.data_vars)
            velocity_candidates = ['velocity', 'v', 'vp', 'v_p', 'vel', 'speed']
            velocity_var = None
            for var in velocity_candidates:
                if var in data_vars:
                    velocity_var = var
                    break
            if velocity_var is None:
                if 'z' in data_vars:
                    velocity_var = 'z'
                else:
                    velocity_var = data_vars[0] if data_vars else 'velocity'
        
        # 使用ProfileExtractor检测到的坐标名
        x_coord = self.profile_extractor.x_coord
        z_coord = self.profile_extractor.z_coord
        
        velocity = self.grid_data[velocity_var].values
        x_coords = self.grid_data.coords[x_coord].values
        z_coords = self.grid_data.coords[z_coord].values
        
        # 确定速度范围
        if vmin is None:
            vmin = np.nanmin(velocity)
        if vmax is None:
            vmax = np.nanmax(velocity)
        
        # 使用pcolormesh代替imshow，可以更好地处理坐标方向
        # pcolormesh会自动根据坐标数组的方向正确显示
        # 参考show_model.py中的实现（304行）
        im = self.ax_main.pcolormesh(
            x_coords,
            z_coords,
            velocity,
            shading='auto',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        
        self.ax_main.set_xlabel('Distance (km)')
        self.ax_main.set_ylabel('Depth (km)')
        self.ax_main.set_title('Velocity Model')
        self.ax_main.invert_yaxis()  # 深度向下增加
        
        if show_colorbar:
            cbar = plt.colorbar(im, ax=self.ax_main)
            cbar.set_label('Velocity (km/s)')
        
        plt.tight_layout()
        return im
    
    def enable_point_selection(self, callback: Optional[Callable] = None):
        """启用点选择功能
        
        Args:
            callback: 选择点后的回调函数 callback(x, z, index)
        """
        self.point_selector = PointSelector(self.ax_main, callback=callback)
        print("点选择已启用：")
        print("  - 左键点击：添加点")
        print("  - 右键点击：删除最近的点")
        print("  - 按D键：删除最后一个点")
        print("  - 按C键：清除所有点")
    
    def enable_polygon_selection(self, callback: Optional[Callable] = None):
        """启用多边形选择功能
        
        Args:
            callback: 选择多边形后的回调函数 callback(vertices)
        """
        def on_select(vertices):
            if len(vertices) >= 3:  # 至少3个点才能形成多边形
                self.selected_polygons.append(vertices)
                # 绘制多边形
                poly = Polygon(vertices, closed=True, fill=False, 
                              edgecolor='red', linewidth=2, alpha=0.7)
                self.ax_main.add_patch(poly)
                self.ax_main.figure.canvas.draw()
                
                if callback:
                    callback(vertices)
        
        self.polygon_selector = PolygonSelector(
            self.ax_main,
            on_select,
            useblit=True,
            props=dict(color='red', linestyle='-', linewidth=2, alpha=0.5)
        )
        print("多边形选择已启用：")
        print("  - 左键点击：添加顶点")
        print("  - 右键点击：完成多边形")
    
    def extract_vertical_profile(self, x: float) -> pd.DataFrame:
        """提取垂直剖面"""
        profile = self.profile_extractor.extract_vertical_profile(x)
        profile_id = f'vertical_x{x:.2f}'
        self.profiles[profile_id] = profile
        return profile
    
    def extract_horizontal_profile(self, z: float) -> pd.DataFrame:
        """提取水平剖面"""
        profile = self.profile_extractor.extract_horizontal_profile(z)
        profile_id = f'horizontal_z{z:.2f}'
        self.profiles[profile_id] = profile
        return profile
    
    def calculate_properties_at_point(self, x: float, z: float) -> Dict:
        """计算指定点的物性参数"""
        props = self.property_calculator.calculate_all_properties(x, z)
        point_id = f'point_{x:.2f}_{z:.2f}'
        self.property_results[point_id] = props
        return props
    
    def calculate_gravity(self, profile: pd.DataFrame,
                         reference_density: float = 2.67) -> pd.DataFrame:
        """计算重力异常"""
        if 'depth' not in profile.columns:
            raise ValueError("剖面必须包含depth列")
        if 'density' not in profile.columns:
            # 如果没有密度，先计算
            profile = self.property_calculator.calculate_properties_for_profile(profile)
        
        gravity = self.gravity_calculator.calculate_bouguer_anomaly(
            profile, reference_density
        )
        
        result = profile.copy()
        result['gravity_anomaly'] = gravity
        return result
    
    def show(self):
        """显示交互式窗口"""
        plt.show()
    
    def save_figure(self, filename: str):
        """保存图形"""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
    
    def export_results(self, output_dir: str):
        """导出所有结果"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 导出剖面
        for name, profile in self.profiles.items():
            profile.to_csv(output_path / f'{name}.csv', index=False)
        
        # 导出物性结果
        if self.property_results:
            props_df = pd.DataFrame(list(self.property_results.values()))
            props_df.to_csv(output_path / 'property_results.csv', index=False)
        
        print(f"结果已导出到: {output_dir}")


# 便捷函数
def interactive_model_viewer(grid_file: str) -> InteractiveModelViewer:
    """创建交互式模型查看器的便捷函数
    
    Args:
        grid_file: 速度模型grid文件路径
    
    Returns:
        InteractiveModelViewer实例
    """
    viewer = InteractiveModelViewer(grid_file=grid_file)
    viewer.plot_model()
    viewer.enable_point_selection()
    viewer.enable_polygon_selection()
    return viewer


if __name__ == '__main__':
    # 使用示例
    import sys
    
    if len(sys.argv) > 1:
        grid_file = sys.argv[1]
    else:
        print("用法: python imodel.py <grid_file>")
        sys.exit(1)
    
    # 创建查看器
    viewer = interactive_model_viewer(grid_file)
    
    # 显示
    viewer.show()
