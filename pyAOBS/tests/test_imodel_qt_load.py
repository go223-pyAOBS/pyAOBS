"""imodel Qt 预览：在无 PySide6 环境下仅测 model_load（不启动 GUI）。"""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

try:
    import numpy as _np  # noqa: F401
except ImportError:
    _np = None

_ROOT = Path(__file__).resolve().parents[1]
_MODEL_LOAD = _ROOT / "visualization" / "imodel" / "gui" / "model_load.py"


def _load_model_load_module():
    """避免经 imodel.__init__ 拉起 show_model/pygmt。"""
    spec = importlib.util.spec_from_file_location("imodel_model_load_under_test", _MODEL_LOAD)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _has_pyside() -> bool:
    try:
        import PySide6  # noqa: F401

        return True
    except ImportError:
        return False


@unittest.skipUnless(_np is not None, "需要 numpy / 完整 pyAOBS 运行环境")
class ImodelQtModelLoadTest(unittest.TestCase):
    def test_is_vin_path(self):
        ml = _load_model_load_module()
        try:
            self.assertTrue(ml.is_vin_path(Path("v.in")))
        except ImportError as e:
            self.skipTest(f"vin_io deps: {e}")
        self.assertTrue(ml.is_vin_path(Path("foo.vin")))
        self.assertFalse(ml.is_vin_path(Path("x.grd")))

    def test_is_smesh_path(self):
        ml = _load_model_load_module()
        self.assertTrue(ml.is_smesh_path(Path("vp.smesh")))
        self.assertTrue(ml.is_smesh_path(Path("out.smesh.8.1")))
        self.assertTrue(ml.is_smesh_path(Path("model.mesh")))
        self.assertFalse(ml.is_smesh_path(Path("v.in")))
        self.assertFalse(ml.is_smesh_path(Path("x.grd")))

    def test_load_smesh_to_dataset(self):
        ml = _load_model_load_module()
        sample = _ROOT / "modeling" / "tomo2d" / "example_water" / "ps_fwd" / "vp.smesh"
        if not sample.is_file():
            self.skipTest(f"missing sample smesh: {sample}")
        try:
            ds, zelt = ml.load_velocity_grid(str(sample))
        except ImportError as e:
            self.skipTest(f"smesh deps: {e}")
        self.assertIsNone(zelt)
        self.assertIn("velocity", ds.data_vars)
        self.assertTrue(ds["velocity"].size > 0)

    def test_is_vin_file_by_content_negative(self):
        import tempfile

        ml = _load_model_load_module()
        fd, name = tempfile.mkstemp(suffix=".txt")
        path = Path(name)
        try:
            os.close(fd)
            path.write_text("not a zelt file\n", encoding="utf-8")
            try:
                self.assertFalse(ml.is_vin_file_by_content(str(path)))
            except ImportError as e:
                self.skipTest(f"vin_io deps: {e}")
        finally:
            path.unlink(missing_ok=True)


@unittest.skipUnless(
    (_np is not None) and _has_pyside(),
    "需要 numpy + PySide6（pip install 'pyAOBS[gui-qt]'）",
)
class ImodelQtImportWithPySide(unittest.TestCase):
    def test_qt_package_main_callable(self):
        from pyAOBS.visualization.imodel.gui import main

        self.assertTrue(callable(main))


if __name__ == "__main__":
    unittest.main()
