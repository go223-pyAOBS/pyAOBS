# -*- coding: utf-8 -*-
"""姿态校正迭代诊断（兼容入口；优先走统一页签结果窗）。"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtWidgets import QDialog, QVBoxLayout

from ..orientation_correction import OrientationCorrectionResult, OrientationObservation
from .dialog_utils import show_modeless_dialog
from .orientation_result_plots import build_diagnostics_page, show_orientation_result_bundle


def show_orientation_diagnostics(
    result: OrientationCorrectionResult,
    parent=None,
    observations: Optional[List[OrientationObservation]] = None,
) -> Optional[QDialog]:
    """若有观测，打开统一页签窗；否则仅诊断页。"""
    if observations:
        return show_orientation_result_bundle(observations, result, parent=parent)
    page = build_diagnostics_page(result, parent=None)
    if page is None:
        return None
    dlg = QDialog(parent)
    dlg.setWindowTitle("姿态校正迭代诊断")
    dlg.setModal(False)
    dlg.resize(1100, 740)
    lay = QVBoxLayout(dlg)
    lay.addWidget(page)
    show_modeless_dialog(dlg, activate=True)
    return dlg
