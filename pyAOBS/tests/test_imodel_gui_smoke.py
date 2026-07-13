"""Smoke tests for imodel Tk GUI package imports (no Tk mainloop)."""

from __future__ import annotations

import unittest

try:
    import numpy as _np_check
except ImportError:
    _np_check = None


@unittest.skipUnless(_np_check is not None, "numpy required (dev / CI with pyAOBS deps)")
class ImodelGuiSmokeTest(unittest.TestCase):
    def test_imodel_gui_module_import(self):
        import pyAOBS.visualization.imodel_gui as ig

        self.assertTrue(hasattr(ig, "InteractiveModelViewerGUI"))
        self.assertTrue(hasattr(ig, "main"))
        self.assertTrue(callable(ig.main))

    def test_viewer_class_mro_contains_mixins(self):
        from pyAOBS.visualization.imodel_gui.viewer import InteractiveModelViewerGUI

        names = {c.__name__ for c in InteractiveModelViewerGUI.__mro__}
        self.assertIn("WorkbenchStateMixin", names)
        self.assertIn("ModelSurfaceMixin", names)
        self.assertIn("PropertiesUIMixin", names)

    def test_deps_symbols(self):
        from pyAOBS.visualization.imodel_gui import deps

        self.assertIsNotNone(deps.np)
        self.assertIsNotNone(deps.plt)
        self.assertIsNotNone(deps.GridModelProcessor)


if __name__ == "__main__":
    unittest.main()
