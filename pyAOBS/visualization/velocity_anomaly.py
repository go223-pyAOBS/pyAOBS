"""Shim: ``pyAOBS.visualization.velocity_anomaly`` → ``imodel.velocity_anomaly``."""

from pyAOBS.visualization.imodel.velocity_anomaly import *  # noqa: F403
from pyAOBS.visualization.imodel import velocity_anomaly as _mod

__all__ = [name for name in dir(_mod) if not name.startswith("_")]
