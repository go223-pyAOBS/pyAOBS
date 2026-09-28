# -*- coding: utf-8 -*-
"""冒烟：mixin 包可导入、方法无交叉定义、壳类 MRO 完整。"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


MIXINS_DIR = Path(__file__).resolve().parents[1] / "gui" / "mixins"
VIEWER_PATH = Path(__file__).resolve().parents[1] / "gui" / "qt_fast_viewer.py"

EXPECTED = [
    "DenoiseMixin",
    "DenoiseCompareMixin",
    "LocationMapMixin",
    "LocationTerrainMixin",
    "ParamPanelMixin",
    "MuteMixin",
    "WaveopMixin",
    "PickMixin",
    "TravelStaticMixin",
    "ThemeMixin",
    "HelpMixin",
    "DataInfoMixin",
    "StackMixin",
    "BuildUiMixin",
    "FileIoMixin",
    "UiParamsMixin",
    "PlotInteractionMixin",
    "TraceGeomMixin",
    "RenderCoreMixin",
]


def _class_methods(path: Path) -> dict[str, set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            names = {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            out[node.name] = names
    return out


class TestQtViewerMixinsSmoke(unittest.TestCase):
    def test_mixin_package_exports(self) -> None:
        from pyAOBS.visualization.zplotpy.gui import mixins as m

        self.assertEqual(list(m.__all__), EXPECTED)
        for name in EXPECTED:
            self.assertTrue(hasattr(m, name), name)

    def test_no_duplicate_method_names_across_mixins(self) -> None:
        all_methods: dict[str, str] = {}
        overlaps: list[str] = []
        for path in sorted(MIXINS_DIR.glob("*_mixin.py")):
            for cls, methods in _class_methods(path).items():
                for meth in methods:
                    if meth in all_methods:
                        overlaps.append(f"{meth}: {all_methods[meth]} vs {cls}")
                    else:
                        all_methods[meth] = cls
        self.assertEqual(overlaps, [], msg="; ".join(overlaps))

    def test_viewer_shell_bases_match_expected(self) -> None:
        tree = ast.parse(VIEWER_PATH.read_text(encoding="utf-8"))
        cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QtFastViewer"
        )
        base_names = []
        for b in cls.bases:
            if isinstance(b, ast.Name):
                base_names.append(b.id)
            elif isinstance(b, ast.Attribute):
                base_names.append(b.attr)
        self.assertEqual(base_names[:-1], EXPECTED)
        self.assertEqual(base_names[-1], "QMainWindow")

    def test_gain_and_render_items_merged(self) -> None:
        self.assertFalse((MIXINS_DIR / "gain_mixin.py").exists())
        self.assertFalse((MIXINS_DIR / "render_items_mixin.py").exists())
        ui = (MIXINS_DIR / "ui_params_mixin.py").read_text(encoding="utf-8")
        rc = (MIXINS_DIR / "render_core_mixin.py").read_text(encoding="utf-8")
        self.assertIn("def _apply_far_offset_boost", ui)
        self.assertIn("def _ensure_curve_pool", rc)

    def test_no_staticmethod_with_self(self) -> None:
        bad: list[str] = []
        for path in sorted(MIXINS_DIR.glob("*_mixin.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                for item in node.body:
                    if not isinstance(item, ast.FunctionDef):
                        continue
                    has_static = any(
                        (isinstance(d, ast.Name) and d.id == "staticmethod")
                        or (isinstance(d, ast.Attribute) and d.attr == "staticmethod")
                        for d in item.decorator_list
                    )
                    if has_static and item.args.args and item.args.args[0].arg == "self":
                        bad.append(f"{path.name}:{item.name}")
        self.assertEqual(bad, [])

        self.assertTrue((MIXINS_DIR / "denoise_compare_mixin.py").exists())
        self.assertTrue((MIXINS_DIR / "location_terrain_mixin.py").exists())
        dn = (MIXINS_DIR / "denoise_mixin.py").read_text(encoding="utf-8")
        cmp_ = (MIXINS_DIR / "denoise_compare_mixin.py").read_text(encoding="utf-8")
        terr = (MIXINS_DIR / "location_terrain_mixin.py").read_text(encoding="utf-8")
        self.assertNotIn("def _open_denoise_compare_plot", dn)
        self.assertIn("def _open_denoise_compare_plot", cmp_)
        self.assertIn("def ensure_shared_terrain_loaded", terr)


if __name__ == "__main__":
    unittest.main()
