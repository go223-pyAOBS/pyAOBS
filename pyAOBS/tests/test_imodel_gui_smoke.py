"""Smoke tests for imodel Qt package imports (no GUI mainloop)."""

from __future__ import annotations

import unittest

try:
    import numpy as _np_check
except ImportError:
    _np_check = None


def _has_pyside() -> bool:
    try:
        import PySide6  # noqa: F401

        return True
    except ImportError:
        return False


@unittest.skipUnless(_np_check is not None, "numpy required (dev / CI with pyAOBS deps)")
class ImodelPackageSmokeTest(unittest.TestCase):
    def test_imodel_module_import(self):
        import pyAOBS.visualization.imodel as imodel

        self.assertTrue(hasattr(imodel, "ProfileExtractor"))
        self.assertTrue(hasattr(imodel, "PropertyCalculator"))
        self.assertFalse(hasattr(imodel, "main"))

    def test_imodel_gui_module_import(self):
        from pyAOBS.visualization.imodel.gui import main

        self.assertTrue(callable(main))

    def test_legacy_imodel_gui_package_gone(self):
        import importlib

        with self.assertRaises((ModuleNotFoundError, ImportError)):
            importlib.import_module("pyAOBS.visualization.imodel_gui")


@unittest.skipUnless(
    (_np_check is not None) and _has_pyside(),
    "需要 numpy + PySide6（pip install 'pyAOBS[gui-qt]'）",
)
class ImodelGuiWithPySide(unittest.TestCase):
    def test_mainwindow_class_import(self):
        from pyAOBS.visualization.imodel.gui.mainwindow import ImodelQtMainWindow

        self.assertTrue(hasattr(ImodelQtMainWindow, "__init__"))


if __name__ == "__main__":
    unittest.main()
