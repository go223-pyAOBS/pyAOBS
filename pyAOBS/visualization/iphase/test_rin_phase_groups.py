from __future__ import annotations

from .rin_phase_groups import (
    ConversionEvent,
    PhaseGroup,
    ReflectionEvent,
    apply_phase_groups_to_rin_text,
    describe_all_phase_groups,
    describe_phase_group_short_sentence,
    parse_rin_phase_arrays_from_text,
    parse_phase_groups_from_rin_text,
)


SAMPLE_RIN = """
 &trapar  imodf=1, i2pt=1,
           ray= 1.2, 2.1,
           nrbnd=0, 1,
           rbnd=-1,
           ncbnd=1, 1,
           cbnd=0, 3,
           nray=30, 70,
           tfile="tx.in",
           vfile="v.in",
 &end
 &invpar  invr=1,
           ivray=5, 24,
 &end
"""


def test_describe_all_phase_groups_sample():
    out = parse_phase_groups_from_rin_text(SAMPLE_RIN, max_layer=6)
    lines = describe_all_phase_groups(out.groups)
    assert len(lines) == 2
    assert "组001:" in lines[0] and "ivray=5" in lines[0]
    assert "震相5" in lines[0] and "震源S" in lines[0] and "无界面转换" in lines[0]
    assert "组002:" in lines[1] and "nrbnd=1" in lines[1] and "rbnd=-1" in lines[1]
    assert "ivray=24" in lines[1] and "nray=70" in lines[1]
    assert "震相24" in lines[1] and "震源P" in lines[1] and "界面3处P→S" in lines[1]
    assert "1次转换" in lines[1] and "PPS波" in lines[1]


def test_describe_pps_cbnd_1_3_ignores_first_encounter_for_count():
    g = PhaseGroup(
        ray_code=5.1,
        nray=10,
        ivray=1,
        source_wave="P",
        conversions=[ConversionEvent(1), ConversionEvent(3)],
    )
    s = describe_phase_group_short_sentence(g)
    assert "1次转换" in s
    assert "界面3处" in s
    assert "P→S" in s
    assert "PPS波" in s
    assert "不计入转换次数" in s


def test_describe_psp_cbnd_3_10_no_cbnd1_strip():
    """PSP：两次真实转换，不以 cbnd=1 作出射扣减。"""
    g = PhaseGroup(
        ray_code=7.1,
        nray=10,
        ivray=99,
        source_wave="P",
        conversions=[ConversionEvent(3), ConversionEvent(10)],
    )
    s = describe_phase_group_short_sentence(g)
    assert "2次转换" in s
    assert "界面3和10处P→S→P" in s
    assert "PSP波" in s
    assert "不计入转换次数" not in s


def test_describe_psp_cbnd_3_6_two_step_chain():
    g = PhaseGroup(
        ray_code=5.1,
        nray=10,
        ivray=2,
        source_wave="P",
        conversions=[ConversionEvent(3), ConversionEvent(6)],
    )
    s = describe_phase_group_short_sentence(g)
    assert "2次转换" in s
    assert "界面3和6处P→S→P" in s
    assert "PSP波" in s


def test_describe_lonely_cbnd_1_not_pps_pss_complete():
    g = PhaseGroup(ray_code=5.1, nray=10, ivray=1, source_wave="P", conversions=[ConversionEvent(1)])
    s = describe_phase_group_short_sentence(g)
    assert "仅cbnd=1" in s


def test_describe_pss_cbnd_1_10():
    g = PhaseGroup(
        ray_code=7.1,
        nray=10,
        ivray=3,
        source_wave="P",
        conversions=[ConversionEvent(1), ConversionEvent(10)],
    )
    s = describe_phase_group_short_sentence(g)
    assert "1次转换" in s
    assert "界面10处P→S" in s
    assert "PSS波" in s
    assert "不计入转换次数" in s


def test_parse_phase_groups_from_rin_text():
    out = parse_phase_groups_from_rin_text(SAMPLE_RIN, max_layer=6)
    assert len(out.groups) == 2
    g1, g2 = out.groups

    assert g1.ray_code == 1.2
    assert g1.nray == 30
    assert g1.ivray == 5
    assert g1.source_wave == "S"
    assert [c.encounter_idx for c in g1.conversions] == [0]
    assert g1.reflections == []

    assert g2.ray_code == 2.1
    assert g2.source_wave == "P"
    assert [r.boundary for r in g2.reflections] == [1]
    assert [r.direction for r in g2.reflections] == ["up_reflect_down"]
    assert [c.encounter_idx for c in g2.conversions] == [3]


def test_compile_and_apply_roundtrip():
    groups = [
        PhaseGroup(
            name="PsA",
            ray_code=4.2,
            nray=50,
            ivray=24,
            source_wave="P",
            reflections=[],
            conversions=[ConversionEvent(encounter_idx=1)],
        ),
        PhaseGroup(
            name="PsB",
            ray_code=4.2,
            nray=50,
            ivray=24,
            source_wave="P",
            reflections=[ReflectionEvent(boundary=1, direction="up_reflect_down")],
            conversions=[ConversionEvent(encounter_idx=3)],
        ),
    ]
    updated_text, diagnostics = apply_phase_groups_to_rin_text(SAMPLE_RIN, groups)
    assert "ray=4.2, 4.2," in updated_text
    assert "nrbnd=0, 1," in updated_text
    assert "rbnd=-1," in updated_text
    assert "ncbnd=1, 1," in updated_text
    assert "cbnd=1, 3," in updated_text
    assert "nray=50, 50," in updated_text
    assert "ivray=24, 24," in updated_text
    assert any("isrch=1" in d for d in diagnostics)


def test_parse_without_rbnd_cbnd_when_counts_zero():
    rin = """
 &trapar  imodf=1,
           ray= 1.2, 2.1, 3.1,
           nrbnd=0,
           ncbnd=0,
           nray=30,
 &end
 &invpar  invr=1,
           ivray=5,
 &end
"""
    out = parse_phase_groups_from_rin_text(rin)
    assert len(out.groups) == 3
    assert all(len(g.reflections) == 0 for g in out.groups)
    assert all(len(g.conversions) == 0 for g in out.groups)
    assert [g.nray for g in out.groups] == [30, 30, 30]
    assert [g.ivray for g in out.groups] == [5, 5, 5]


def test_parse_rin_phase_arrays_main_and_pool_arrays():
    arr = parse_rin_phase_arrays_from_text(SAMPLE_RIN, parse_path_arrays=True)
    assert arr.ray == [1.2, 2.1]
    assert arr.nrbnd == [0, 1]
    assert arr.rbnd == [-1]
    assert arr.ncbnd == [1, 1]
    assert arr.cbnd == [0, 3]
    assert arr.nray == [30, 70]
    assert arr.ivray == [5, 24]


def test_safe_mode_still_loads_cbnd_pool():
    arr = parse_rin_phase_arrays_from_text(SAMPLE_RIN, parse_path_arrays=False)
    assert arr.cbnd == [0, 3]
    assert arr.rbnd == [-1]


def test_compile_disabled_group_keeps_nray_zero_and_same_length():
    from .rin_phase_groups import compile_phase_groups_to_arrays

    g1 = PhaseGroup(
        enabled=False,
        ray_code=1.2,
        nray=99,
        ivray=5,
        conversions=[ConversionEvent(0)],
    )
    g2 = PhaseGroup(
        enabled=True,
        ray_code=2.1,
        nray=70,
        ivray=24,
        reflections=[ReflectionEvent(1, "up_reflect_down")],
        conversions=[ConversionEvent(3)],
    )
    out = compile_phase_groups_to_arrays([g1, g2])
    assert out["nray"] == [0, 70]
    assert out["ivray"] == [5, 24]
    assert out["cbnd"] == [0, 3]

