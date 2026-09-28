"""imodel_gui 各 Tk mixin 共用的依赖（Matplotlib/Tk/pyAOBS/Talwani 可选）。"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Polygon
from matplotlib.widgets import PolygonSelector

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from pyAOBS.model_building.zeltform import ZeltVelocityModel2d
    from pyAOBS.utils import SimpleRockClassifier
    from pyAOBS.utils import (
        calculate_dem_velocity,
        calculate_density,
        calculate_pressure_from_depth,
        calculate_serpentinization_from_water_content,
        calculate_temperature_from_depth,
        calculate_vp_from_serpentinization,
        calculate_vp_from_vs_brocher,
        calculate_vs,
        calculate_vs_from_serpentinization,
        calculate_water_content_from_porosity,
        correct_velocity,
    )
    from pyAOBS.utils.rocks import Rock, RockProperties
    from pyAOBS.visualization.imodel import (
        GravityCalculator,
        PointSelector,
        ProfileExtractor,
        PropertyCalculator,
    )
    from pyAOBS.visualization.show_model import GridModelProcessor
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from pyAOBS.model_building.zeltform import ZeltVelocityModel2d
    from pyAOBS.utils import SimpleRockClassifier
    from pyAOBS.utils import (
        calculate_dem_velocity,
        calculate_density,
        calculate_pressure_from_depth,
        calculate_serpentinization_from_water_content,
        calculate_temperature_from_depth,
        calculate_vp_from_serpentinization,
        calculate_vp_from_vs_brocher,
        calculate_vs,
        calculate_vs_from_serpentinization,
        calculate_water_content_from_porosity,
        correct_velocity,
    )
    from pyAOBS.utils.rocks import Rock, RockProperties
    from pyAOBS.visualization.imodel import (
        GravityCalculator,
        PointSelector,
        ProfileExtractor,
        PropertyCalculator,
    )
    from pyAOBS.visualization.show_model import GridModelProcessor

from .talwani_optional import (
    TALWANI_AVAILABLE,
    calculate_gravity_anomaly,
    convert_density_units,
    talwani2d_gravity,
    talwani2d_gravity_multibody,
    talwani2d_gravity_profile,
)

__all__ = [
    "Any",
    "Callable",
    "Dict",
    "FigureCanvasTkAgg",
    "GravityCalculator",
    "GridModelProcessor",
    "List",
    "NavigationToolbar2Tk",
    "Optional",
    "Path",
    "PointSelector",
    "Polygon",
    "PolygonSelector",
    "ProfileExtractor",
    "PropertyCalculator",
    "Rock",
    "RockProperties",
    "SimpleRockClassifier",
    "TALWANI_AVAILABLE",
    "Tuple",
    "Union",
    "ZeltVelocityModel2d",
    "calculate_dem_velocity",
    "calculate_density",
    "calculate_gravity_anomaly",
    "calculate_pressure_from_depth",
    "calculate_serpentinization_from_water_content",
    "calculate_temperature_from_depth",
    "calculate_vp_from_serpentinization",
    "calculate_vp_from_vs_brocher",
    "calculate_vs",
    "calculate_vs_from_serpentinization",
    "calculate_water_content_from_porosity",
    "convert_density_units",
    "correct_velocity",
    "datetime",
    "filedialog",
    "hashlib",
    "json",
    "matplotlib",
    "messagebox",
    "np",
    "os",
    "pd",
    "pickle",
    "platform",
    "plt",
    "shutil",
    "talwani2d_gravity",
    "talwani2d_gravity_multibody",
    "talwani2d_gravity_profile",
    "tk",
    "ttk",
    "warnings",
    "xr",
]
