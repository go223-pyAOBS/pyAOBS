# -*- coding: utf-8 -*-
"""tomo2d GUI 文件对话框过滤器（避免 *.*/漏匹配 *.smesh.<iter>.<iset>）。"""

from __future__ import annotations

# Qt 文档：用 (*) 表示所有文件；*.* 在非原生对话框上不可移植。
ALL_FILES = "所有文件 (*)"

# 反演输出多为 out.smesh.<iter>.<iset>：末段扩展名是数字，*.smesh 匹配不到。
SMESH_OPEN_FILTERS = (
    f"反演迭代 (*.smesh.*);;"
    f"smesh / 数据 (*.smesh *.dat);;"
    f"{ALL_FILES}"
)
MODEL_OPEN_FILTERS = (
    f"模型 (*.smesh.* *.smesh v.in *.in *.grd *.nc);;"
    f"反演迭代 (*.smesh.*);;"
    f"smesh (*.smesh *.dat);;"
    f"Zelt v.in (v.in *.in *.vin);;"
    f"GMT 网格 (*.grd *.nc);;"
    f"{ALL_FILES}"
)
SMESH_SAVE_FILTERS = f"smesh (*.smesh);;数据 (*.dat);;{ALL_FILES}"

REFL_OPEN_FILTERS = (
    f"界面 / 反射面 (refl* *.refl.* *.refl *.dat *.txt);;"
    f"反演界面 (*.refl.*);;"
    f"refl 开头 (refl*);;"
    f"反射面 (*.refl *.dat *.txt);;"
    f"{ALL_FILES}"
)
REFL_SAVE_FILTERS = f"反射面 (refl* *.refl *.dat *.txt);;{ALL_FILES}"
DWS_OPEN_FILTERS = (
    f"DWS / 文本 (*.dat *.txt);;"
    f"{ALL_FILES}"
)
RAY_OPEN_FILTERS = (
    f"反演射线 (*.ray.*);;"
    f"射线 (*.ray);;"
    f"{ALL_FILES}"
)

DATA_OPEN_FILTERS = f"走时/几何 (*.dat);;{ALL_FILES}"
DATA_SAVE_FILTERS = f"走时/几何 (*.dat);;{ALL_FILES}"

LOG_OPEN_FILTERS = f"log (*.log);;文本 (*.txt);;{ALL_FILES}"
JSON_OPEN_FILTERS = f"JSON (*.json);;{ALL_FILES}"
JSON_PROJECT_OPEN_FILTERS = (
    f"tomo2d project (tomo2d_project.json);;JSON (*.json);;{ALL_FILES}"
)
PNG_SAVE_FILTERS = f"PNG (*.png);;{ALL_FILES}"
TEXT_OPEN_FILTERS = f"文本 (*.txt *.dat *.in);;{ALL_FILES}"

# 字段 key 子串 → (open_filter, save_filter)；未命中则仅 ALL_FILES
_KEY_FILTER_RULES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (
        ("dws_file", "grav_dws", "plot_smesh_dws"),
        DWS_OPEN_FILTERS,
        DWS_OPEN_FILTERS,
    ),
    (
        ("plot_smesh_ray", "out_ray"),
        RAY_OPEN_FILTERS,
        RAY_OPEN_FILTERS,
    ),
    (
        (
            "smesh",
            "inv.mesh",
            "mesh_file",
            "base_mesh",
            "bg_smesh",
            "link_smesh",
        ),
        SMESH_OPEN_FILTERS,
        SMESH_SAVE_FILTERS,
    ),
    (
        ("refl_file", "moho_file", "seafloor"),
        REFL_OPEN_FILTERS,
        REFL_SAVE_FILTERS,
    ),
    (
        (
            "geom",
            "inv.data",
            "out_ttime",
            "out_obs",
            "data_out",
            "geom_out",
            "ttimes",
        ),
        DATA_OPEN_FILTERS,
        DATA_SAVE_FILTERS,
    ),
    (
        ("v_in", "tx_in", "station", ".in"),
        TEXT_OPEN_FILTERS,
        TEXT_OPEN_FILTERS,
    ),
    (
        ("log_file", ".log"),
        LOG_OPEN_FILTERS,
        LOG_OPEN_FILTERS,
    ),
)


def filters_for_form_key(key: str, *, for_save: bool = False) -> str:
    """按表单字段 key 选择合适的打开/保存过滤器。"""
    k = (key or "").lower()
    if "cmap" in k or "cpt" in k:
        return f"色标 (*.cpt *.txt);;{ALL_FILES}"
    for needles, open_f, save_f in _KEY_FILTER_RULES:
        if any(n in k for n in needles):
            return save_f if for_save else open_f
    return ALL_FILES
