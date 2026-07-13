# -*- coding: utf-8 -*-
"""tx2tomo2d.py 单元测试（直接加载模块，避免经 pyAOBS 包顶层拉取 numpy 等依赖）。"""

import importlib.util
import sys
import unittest
from pathlib import Path


def _load_tx2tomo2d():
    root = Path(__file__).resolve().parent.parent
    path = root / "modeling" / "tomo2d" / "tx2tomo2d.py"
    spec = importlib.util.spec_from_file_location("tx2tomo2d", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tx2tomo2d"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_tx = _load_tx2tomo2d()
convert_tx_in_to_tomo2d = _tx.convert_tx_in_to_tomo2d
parse_phase_set = _tx.parse_phase_set


class TestTx2Tomo2d(unittest.TestCase):
    def test_parse_phase_set(self):
        self.assertEqual(parse_phase_set("1, 2 ,3"), {1, 2, 3})
        self.assertEqual(parse_phase_set("11 12"), {11, 12})
        self.assertEqual(parse_phase_set(""), set())

    def test_convert_minimal(self):
        import tempfile

        d = Path(tempfile.mkdtemp())
        st = d / "station.lis"
        st.write_text("1 100.0 0.5\n", encoding="utf-8")
        txf = d / "tx.in"
        txf.write_text(
            "100.0 -1.0 0.0 0\n"
            "50.0 1.0 0.1 1\n"
            "100.0 1.0 0.0 0\n"
            "60.0 2.0 0.1 1\n"
            "0 0 0 -1\n",
            encoding="utf-8",
        )
        dout = d / "ttimes.dat"
        gout = d / "geom.dat"
        stats = convert_tx_in_to_tomo2d(
            st, txf, dout, gout, refr_phases={1}, refl_phases=set()
        )
        self.assertEqual(stats.nshot, 1)
        self.assertEqual(stats.ntime, 2)
        data = dout.read_text(encoding="utf-8").splitlines()
        self.assertEqual(data[0], "1")
        self.assertTrue(data[1].startswith("s"))
        self.assertEqual(len(data), 4)

    def test_convert_refl_and_refr(self):
        import tempfile

        d = Path(tempfile.mkdtemp())
        st = d / "station.lis"
        st.write_text("7 200.0 1.0\n", encoding="utf-8")
        txf = d / "tx.in"
        txf.write_text(
            "200.0 0.0 0.0 0\n"
            "190.0 3.0 0.05 1\n"
            "195.0 5.0 0.05 11\n"
            "0 0 0 -1\n",
            encoding="utf-8",
        )
        dout = d / "out.dat"
        gout = d / "out.geom"
        convert_tx_in_to_tomo2d(
            st, txf, dout, gout, refr_phases={1}, refl_phases={11}
        )
        r_lines = [ln for ln in dout.read_text(encoding="utf-8").splitlines() if ln.startswith("r")]
        self.assertEqual(len(r_lines), 2)
        self.assertEqual(int(r_lines[0][21:26]), 0)
        self.assertEqual(int(r_lines[1][21:26]), 1)
        g_r = [ln for ln in gout.read_text(encoding="utf-8").splitlines() if ln.startswith("r")]
        self.assertTrue(g_r[0].rstrip().endswith("0.000     0.000"))
        self.assertTrue(g_r[1].rstrip().endswith("0.000     0.000"))

    def test_empty_phases_writes_zero_shots(self):
        import tempfile

        d = Path(tempfile.mkdtemp())
        st = d / "station.lis"
        st.write_text("1 1.0 0.0\n", encoding="utf-8")
        txf = d / "tx.in"
        txf.write_text("1.0 0 0 0\n2.0 1 0.1 9\n0 0 0 -1\n", encoding="utf-8")
        dout = d / "d.dat"
        gout = d / "g.dat"
        stats = convert_tx_in_to_tomo2d(
            st, txf, dout, gout, refr_phases=set(), refl_phases=set()
        )
        self.assertEqual(stats.nshot, 0)
        self.assertEqual(dout.read_text(encoding="utf-8").strip(), "0")


if __name__ == "__main__":
    unittest.main()
