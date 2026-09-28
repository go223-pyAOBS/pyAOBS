"""安静 I/O 默认。"""
from __future__ import annotations

import pytest

from pyAOBS.modeling.tomo2d.gui.services.io_defaults import ensure_quiet_io_defaults
from pyAOBS.modeling.tomo2d.gui.state.form_state import FormState

pytestmark = pytest.mark.unit


def test_quiet_io_defaults_fill_missing_only() -> None:
    st = FormState({"inv.out_level": "2"})  # 用户已设，勿覆盖
    applied = ensure_quiet_io_defaults(st)
    assert "inv.out_level" not in applied
    assert st.get_str("inv.out_level") == "2"
    assert st.get_bool("inv.print_final_only") is True
    assert st.get_str("inv.verbose_level") == ""
    assert st.get_str("fwd.out_ray") == ""
    assert "inv.print_final_only" in applied
