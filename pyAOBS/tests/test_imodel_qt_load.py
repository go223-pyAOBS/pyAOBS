"""imodel Qt 预览：在无 PySide6 环境下仅测 model_load（不启动 GUI）。"""

from __future__ import annotations

import unittest
from pathlib import Path

try:
    import numpy as _np  # noqa: F401
except ImportError:
    _np = None


def _has_pyside() -> bool:
    try:
        import PySide6  # noqa: F401

        return True
    except ImportError:
        return False


@unittest.skipUnless(_np is not None, "需要 numpy / 完整 pyAOBS 运行环境")
class ImodelQtModelLoadTest(unittest.TestCase):
    def test_is_vin_path(self):
        from pyAOBS.visualization.imodel_qt.model_load import is_vin_path

        self.assertTrue(is_vin_path(Path("v.in")))
        self.assertTrue(is_vin_path(Path("foo.vin")))
        self.assertFalse(is_vin_path(Path("x.grd")))

    def test_is_vin_file_by_content_negative(self):
        import tempfile

        from pyAOBS.visualization.imodel_qt.model_load import is_vin_file_by_content

        with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as f:
            try:
                f.write("not a zelt file\n")
                f.flush()
                self.assertFalse(is_vin_file_by_content(f.name))
            finally:
                Path(f.name).unlink(missing_ok=True)


@unittest.skipUnless(
    (_np is not None) and _has_pyside(),
    "需要 numpy + PySide6（pip install 'pyAOBS[gui-qt]'）",
)
class ImodelQtImportWithPySide(unittest.TestCase):
    def test_qt_package_main_callable(self):
        from pyAOBS.visualization.imodel_qt import main

        self.assertTrue(callable(main))


if __name__ == "__main__":
    unittest.main()
