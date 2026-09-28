"""默认 bin_path。"""
from __future__ import annotations

from pathlib import Path

import pytest

from pyAOBS.modeling.tomo2d.gui.services.bin_defaults import default_bin_path

pytestmark = pytest.mark.unit


def test_default_bin_path_points_at_src_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYAOBS_TOMO2D_BIN", raising=False)
    monkeypatch.delenv("TOMO2D_BIN", raising=False)
    p = Path(default_bin_path())
    assert p.name == "build-tomo2d"
    assert p.parent.name == "src"
    assert "tomo2d" in p.as_posix()


def test_env_overrides_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PYAOBS_TOMO2D_BIN", str(tmp_path))
    assert Path(default_bin_path()).resolve() == tmp_path.resolve()
