"""Talwani 重力相关可选依赖；供 viewer 与各 mixin 共用，避免重复 try/except。"""

try:
    from pyAOBS.utils.gravity_talwani import (
        talwani2d_gravity,
        talwani2d_gravity_profile,
        talwani2d_gravity_multibody,
        calculate_gravity_anomaly,
        convert_density_units,
    )
    TALWANI_AVAILABLE = True
except ImportError:
    try:
        from utils.gravity_talwani import (
            talwani2d_gravity,
            talwani2d_gravity_profile,
            talwani2d_gravity_multibody,
            calculate_gravity_anomaly,
            convert_density_units,
        )

        TALWANI_AVAILABLE = True
    except ImportError:
        TALWANI_AVAILABLE = False
        talwani2d_gravity = None
        talwani2d_gravity_profile = None
        talwani2d_gravity_multibody = None
        calculate_gravity_anomaly = None
        convert_density_units = None
