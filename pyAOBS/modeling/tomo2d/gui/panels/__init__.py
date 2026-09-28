"""主窗 panels。"""

from .checkerboard_tab import CheckerboardTab
from .edit_smesh_tab import EditSmeshTab
from .gen_damp_tab import GenDampTab
from .gen_dcorr_tab import GenDcorrTab
from .gen_smesh_tab import GenSmeshTab
from .gen_vcorr_tab import GenVcorrTab
from .log_panel import LogPanel
from .monte_carlo_tab import MonteCarloTab
from .parallel_env_panel import ParallelEnvPanel
from .pipeline_tab import PipelineTab
from .preview_panel import PreviewPanel
from .stat_smesh_tab import StatSmeshTab
from .top_chrome import TopChromePanel
from .tt_forward_tab import TtForwardTab
from .tt_inverse_tab import TtInverseTab
from .tx_convert_tab import TxConvertTab
from .wave2d_tab import Wave2dTab

__all__ = [
    "CheckerboardTab",
    "EditSmeshTab",
    "GenDampTab",
    "GenDcorrTab",
    "GenSmeshTab",
    "GenVcorrTab",
    "LogPanel",
    "MonteCarloTab",
    "ParallelEnvPanel",
    "PipelineTab",
    "PreviewPanel",
    "StatSmeshTab",
    "TopChromePanel",
    "TtForwardTab",
    "TtInverseTab",
    "TxConvertTab",
    "Wave2dTab",
]
