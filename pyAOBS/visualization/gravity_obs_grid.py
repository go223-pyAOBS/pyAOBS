"""Shim: ``pyAOBS.visualization.gravity_obs_grid`` → ``imodel.gravity_obs_grid``."""

from pyAOBS.visualization.imodel.gravity_obs_grid import *  # noqa: F403
from pyAOBS.visualization.imodel import gravity_obs_grid as _mod

__all__ = [name for name in dir(_mod) if not name.startswith("_")]
