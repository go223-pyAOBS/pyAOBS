import json

import numpy as np
import xarray as xr

from pyAOBS.visualization import imodel as imodel_module
from pyAOBS.visualization.imodel import PropertyCalculator


def _build_test_grid() -> xr.Dataset:
    vp = np.array([
        [1.5, 1.6],
        [2.8, 6.4],
    ], dtype=float)
    return xr.Dataset(
        {"velocity": (("z", "x"), vp)},
        coords={"x": np.array([0.0, 1.0]), "z": np.array([-0.1, 1.0])},
    )


def _build_test_vs_grid() -> xr.Dataset:
    vs = np.array([
        [0.0, 0.0],
        [1.6, 3.6],
    ], dtype=float)
    return xr.Dataset(
        {"vs": (("z", "x"), vs)},
        coords={"x": np.array([0.0, 1.0]), "z": np.array([-0.1, 1.0])},
    )


def test_zone_prior_rules_without_classifier(monkeypatch):
    monkeypatch.setattr(imodel_module, "SimpleRockClassifier", None)
    calc = PropertyCalculator(_build_test_grid())

    water = calc.calculate_all_properties(x=0.0, z=-0.1, seafloor_depth=0.0, basement_depth=2.0)
    assert water["zone"] == "water"
    assert water["rock_type"] == "Seawater"
    assert float(water["vs"]) == 0.0
    assert float(water["density"]) == calc.seawater_density_g_cm3
    assert float(water["pressure"]) == 0.0
    assert float(water["shear_modulus"]) == 0.0
    assert float(water["temperature"]) == calc.seawater_temperature_c

    water_mid = calc.calculate_all_properties(x=0.0, z=0.2, seafloor_depth=0.5, basement_depth=2.0)
    assert water_mid["zone"] == "water"
    assert abs(float(water_mid["pressure"]) - 0.2 * calc.seawater_pressure_gradient_mpa_per_km) < 1e-6
    assert float(water_mid["temperature"]) == calc.seawater_temperature_c

    # 海底以下总压力应包含海水柱静水压基线，不应低于海底附近水层压力
    shallow_sed = calc.calculate_all_properties(x=0.0, z=0.6, seafloor_depth=0.5, basement_depth=2.0)
    assert shallow_sed["zone"] in {"sediment", "deep"}
    assert float(shallow_sed["pressure"]) > float(water_mid["pressure"])
    assert abs(
        float(shallow_sed["temperature"]) - (calc.seafloor_temperature_c + calc.geothermal_gradient_c_per_km * 0.1)
    ) < 1e-6

    sediment = calc.calculate_all_properties(x=0.0, z=1.0, seafloor_depth=0.0, basement_depth=2.0)
    assert sediment["zone"] == "sediment"
    assert "sediment" in str(sediment["rock_type"]).lower()

    deep = calc.calculate_all_properties(x=1.0, z=1.0, seafloor_depth=0.0, basement_depth=0.5)
    assert deep["zone"] == "deep"
    assert deep["vs_source"] == "empirical"


def test_dual_vp_vs_model_prefers_vs_grid(monkeypatch):
    monkeypatch.setattr(imodel_module, "SimpleRockClassifier", None)
    calc = PropertyCalculator(_build_test_grid(), vs_grid_data=_build_test_vs_grid())
    result = calc.calculate_all_properties(x=1.0, z=1.0, seafloor_depth=0.0, basement_depth=2.0)
    assert abs(float(result["vp"]) - 6.4) < 1e-6
    assert abs(float(result["vs"]) - 3.6) < 1e-6
    assert result["vs_source"] == "model"
    assert abs(float(result["vp_vs_ratio"]) - (6.4 / 3.6)) < 1e-6


def test_load_sedimentary_keywords_from_config(monkeypatch, tmp_path):
    monkeypatch.setattr(imodel_module, "SimpleRockClassifier", None)
    cfg = tmp_path / "imodel_priors.json"
    cfg.write_text(
        json.dumps({"sedimentary_keywords": ["my_sediment_keyword"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setenv("PYAOBS_IMODEL_PRIORS_FILE", str(cfg))

    calc = PropertyCalculator(_build_test_grid())
    assert calc.sedimentary_keywords == ["my_sediment_keyword"]


def test_low_confidence_rerank_prefers_zone_compatible_candidate(monkeypatch):
    monkeypatch.setattr(imodel_module, "SimpleRockClassifier", None)
    calc = PropertyCalculator(_build_test_grid())

    class DummyClassifier:
        @staticmethod
        def get_auxiliary_attributes(rock_type, vp=None, vs=None):
            if str(rock_type).lower() == "sandstone":
                return {"rock_classification": "沉积岩", "geological_meaning": "沉积层"}
            return {"rock_classification": "火成岩", "geological_meaning": "深部侵入体"}

    class DummySimpleClassifier:
        def __init__(self):
            self.classifier = DummyClassifier()

    calc.rock_classifier = DummySimpleClassifier()
    calc.sedimentary_keywords = ["sediment", "沉积"]
    candidates = [
        {"rock_type": "Basalt", "probability": 0.34},
        {"rock_type": "Sandstone", "probability": 0.33},
    ]

    reranked, changed = calc._rerank_low_confidence_candidates(
        candidates,
        zone="sediment",
        vp=4.0,
        vs=2.2,
        expected_foma=None,
        expected_facies=None,
        expected_sio2=None,
    )
    assert changed is True
    assert reranked[0]["rock_type"] == "Sandstone"
    assert reranked[0]["probability"] >= reranked[1]["probability"]


def test_sediment_zone_constrains_conflicting_aux_fields(monkeypatch):
    class DummySimpleRockClassifier:
        def __init__(self, *args, **kwargs):
            self.database_file = ""
            self.classifier = None

        @staticmethod
        def classify(**kwargs):
            return {
                "rock_type": "Sandstone/Shale",
                "probability": 0.55,
                "all_candidates": [{"rock_type": "Sandstone/Shale", "probability": 0.55}],
                "felsic_or_mafic": "ultramafic",
                "rock_facies": "UNKNOWN",
                "sio2_wt": 50.2,
                "rock_classification": "变质岩（蚀变）",
                "metamorphic_grade": "蛇纹石化",
                "geological_meaning": "超镁铁质岩水化蚀变，代表洋壳或地幔楔",
            }

    monkeypatch.setattr(imodel_module, "SimpleRockClassifier", DummySimpleRockClassifier)
    calc = PropertyCalculator(_build_test_grid())
    result = calc.calculate_all_properties(x=0.0, z=1.0, seafloor_depth=0.0, basement_depth=2.0)

    assert result["zone"] == "sediment"
    assert result["felsic_or_mafic"] == "sedimentary"
    assert result["rock_facies"] == "sedimentary"
    assert result["rock_classification"] == "沉积岩"
    assert result["metamorphic_grade"] == "-"
    assert "沉积层先验约束" in str(result["geological_meaning"])
    assert result["zone_constraint_applied"] is True


def test_zone_first_prior_limits_classification_sample_space(monkeypatch):
    class DummyInnerClassifier:
        def __init__(self):
            import pandas as pd
            self.rock_database_reference = pd.DataFrame(
                [
                    {"rock_type": "Basalt", "岩石分类": "火成岩", "地质意义": "洋壳"},
                    {"rock_type": "Sandstone", "岩石分类": "沉积岩", "地质意义": "沉积层"},
                ]
            )

        @staticmethod
        def correct_velocity(v, **kwargs):
            return float(v)

        @staticmethod
        def classify_probabilities_by_features(**kwargs):
            return {"Basalt": 0.9, "Sandstone": 0.1}

        @staticmethod
        def get_auxiliary_attributes(rock_type, vp=None, vs=None):
            if str(rock_type).lower() == "sandstone":
                return {"rock_classification": "沉积岩", "geological_meaning": "沉积层"}
            return {"rock_classification": "火成岩", "geological_meaning": "洋壳"}

    class DummySimpleRockClassifier:
        def __init__(self, *args, **kwargs):
            self.database_file = ""
            self.classifier = DummyInnerClassifier()

        @staticmethod
        def classify(**kwargs):
            return {
                "rock_type": "Basalt",
                "probability": 0.9,
                "all_candidates": [{"rock_type": "Basalt", "probability": 0.9}],
            }

    monkeypatch.setattr(imodel_module, "SimpleRockClassifier", DummySimpleRockClassifier)
    calc = PropertyCalculator(_build_test_grid())
    result = calc.calculate_all_properties(x=0.0, z=1.0, seafloor_depth=0.0, basement_depth=2.0)
    assert result["zone"] == "sediment"
    assert result["zone_prior_applied"] is True
    assert str(result["rock_type"]).lower() == "sandstone"


def test_load_seawater_temperature_from_config_and_env(monkeypatch, tmp_path):
    monkeypatch.setattr(imodel_module, "SimpleRockClassifier", None)
    cfg = tmp_path / "imodel_priors.json"
    cfg.write_text(
        json.dumps({"seawater_temperature_c": 2.5}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setenv("PYAOBS_IMODEL_PRIORS_FILE", str(cfg))
    calc = PropertyCalculator(_build_test_grid())
    assert abs(calc.seawater_temperature_c - 2.5) < 1e-6

    monkeypatch.setenv("PYAOBS_IMODEL_SEAWATER_TEMPERATURE_C", "1.8")
    calc2 = PropertyCalculator(_build_test_grid())
    assert abs(calc2.seawater_temperature_c - 1.8) < 1e-6


def test_load_seafloor_temperature_from_config_and_env(monkeypatch, tmp_path):
    monkeypatch.setattr(imodel_module, "SimpleRockClassifier", None)
    cfg = tmp_path / "imodel_priors.json"
    cfg.write_text(
        json.dumps({"seafloor_temperature_c": 6.5}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setenv("PYAOBS_IMODEL_PRIORS_FILE", str(cfg))
    calc = PropertyCalculator(_build_test_grid())
    assert abs(calc.seafloor_temperature_c - 6.5) < 1e-6

    monkeypatch.setenv("PYAOBS_IMODEL_SEAFLOOR_TEMPERATURE_C", "8.2")
    calc2 = PropertyCalculator(_build_test_grid())
    assert abs(calc2.seafloor_temperature_c - 8.2) < 1e-6


def test_load_geothermal_gradient_from_config_and_env(monkeypatch, tmp_path):
    monkeypatch.setattr(imodel_module, "SimpleRockClassifier", None)
    cfg = tmp_path / "imodel_priors.json"
    cfg.write_text(
        json.dumps({"geothermal_gradient_c_per_km": 35.0}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setenv("PYAOBS_IMODEL_PRIORS_FILE", str(cfg))
    calc = PropertyCalculator(_build_test_grid())
    assert abs(calc.geothermal_gradient_c_per_km - 35.0) < 1e-6

    monkeypatch.setenv("PYAOBS_IMODEL_GEOTHERMAL_GRADIENT_C_PER_KM", "40.5")
    calc2 = PropertyCalculator(_build_test_grid())
    assert abs(calc2.geothermal_gradient_c_per_km - 40.5) < 1e-6


def test_load_seawater_density_and_pressure_gradient_config(monkeypatch, tmp_path):
    monkeypatch.setattr(imodel_module, "SimpleRockClassifier", None)
    cfg = tmp_path / "imodel_priors.json"
    cfg.write_text(
        json.dumps(
            {
                "seawater_density_g_cm3": 1.028,
                "seawater_pressure_gradient_mpa_per_km": 10.4,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PYAOBS_IMODEL_PRIORS_FILE", str(cfg))
    calc = PropertyCalculator(_build_test_grid())
    assert abs(calc.seawater_density_g_cm3 - 1.028) < 1e-6
    assert abs(calc.seawater_pressure_gradient_mpa_per_km - 10.4) < 1e-6

    monkeypatch.setenv("PYAOBS_IMODEL_SEAWATER_DENSITY_G_CM3", "1.04")
    monkeypatch.setenv("PYAOBS_IMODEL_SEAWATER_PRESSURE_GRADIENT_MPA_PER_KM", "10.7")
    calc2 = PropertyCalculator(_build_test_grid())
    assert abs(calc2.seawater_density_g_cm3 - 1.04) < 1e-6
    assert abs(calc2.seawater_pressure_gradient_mpa_per_km - 10.7) < 1e-6
