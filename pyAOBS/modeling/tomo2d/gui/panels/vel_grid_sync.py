# -*- coding: utf-8 -*-
"""gen_smesh / gen_damp / gen_vcorr 共用的 vel_opt ↔ grid_opt 联动。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from ..widgets.field_form import FieldFormTab


def sync_vel_grid_options(
    tab: "FieldFormTab",
    *,
    vel_key: str,
    grid_key: str,
    uniform_vel_keys: Sequence[str],
    zelt_vel_keys: Sequence[str],
    uniform_grid_keys: Sequence[str],
    variable_grid_keys: Sequence[str],
    zelt_grid_keys: Sequence[str],
    zelt_section: int | None = 1,
    grid_section: int | None = 2,
) -> tuple[str, str]:
    """
    根据 vel_opt / grid_opt 启用参数框；vel_opt=zelt 时强制 grid_opt=zelt 并锁定下拉框。

    返回 (vel_opt, grid_opt)。以 combo 当前显示值为准，避免 state 与控件不一致时误锁。
    """
    vel_combo = tab._combo_keys.get(vel_key)
    grid_combo = tab._combo_keys.get(grid_key)
    if vel_combo is not None:
        vo = str(vel_combo.currentText() or "").strip()
        tab.state.set(vel_key, vo)
    else:
        tab.binder.pull_from_widgets()
        vo = tab.state.get_str(vel_key)

    if vo == "zelt" and grid_combo is not None:
        if grid_combo.currentText() != "zelt":
            grid_combo.blockSignals(True)
            idx = grid_combo.findText("zelt")
            if idx >= 0:
                grid_combo.setCurrentIndex(idx)
            grid_combo.blockSignals(False)
        tab.state.set(grid_key, "zelt")
        grid_combo.setEnabled(False)
        go = "zelt"
    else:
        if grid_combo is not None:
            grid_combo.setEnabled(True)
            go = str(grid_combo.currentText() or "").strip()
            tab.state.set(grid_key, go)
        else:
            tab.binder.pull_from_widgets()
            go = tab.state.get_str(grid_key)

    # help：两侧速度参数都灰显；zelt 时必须点亮 Zelt 输入
    tab.set_enabled_keys(list(uniform_vel_keys), vo == "uniform")
    tab.set_enabled_keys(list(zelt_vel_keys), vo == "zelt")

    if go == "uniform":
        tab.set_enabled_keys(list(uniform_grid_keys), True)
        tab.set_enabled_keys(list(variable_grid_keys), False)
        extra_off = [k for k in zelt_grid_keys if k not in variable_grid_keys]
        if extra_off:
            tab.set_enabled_keys(extra_off, False)
    elif go == "variable":
        tab.set_enabled_keys(list(uniform_grid_keys), False)
        tab.set_enabled_keys(list(variable_grid_keys), True)
        dx_keys = [k for k in zelt_grid_keys if k not in variable_grid_keys]
        if dx_keys:
            tab.set_enabled_keys(dx_keys, False)
    else:  # zelt grid
        tab.set_enabled_keys(list(uniform_grid_keys), False)
        var_only = [k for k in variable_grid_keys if k not in zelt_grid_keys]
        tab.set_enabled_keys(var_only, False)
        tab.set_enabled_keys(list(zelt_grid_keys), True)

    if vo == "zelt" and zelt_section is not None:
        tab.set_section_expanded(zelt_section, True)
    if go != "uniform" and grid_section is not None:
        tab.set_section_expanded(grid_section, True)

    return vo, go
