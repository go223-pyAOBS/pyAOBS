"""嵌入 Matplotlib Figure 的非模态工具窗（pyqtgraph 式鼠标，无 NavigationToolbar）。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
except ImportError:  # pragma: no cover
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from pyAOBS.modeling.tomo2d.tt_inverse_log_analysis import ensure_matplotlib_cjk_font
from pyAOBS.utils.mpl_plot_nav import PyqtgraphStyleNav

from ..dialog_utils import show_modeless_dialog


class MplNavCanvas(QWidget):
    """对齐 iphase：滚轮缩放 / 左拖平移 / 右拖连续缩放 / 双击与 Reset 复位到 home。"""

    def __init__(self, fig, parent=None, *, on_right_click=None) -> None:
        super().__init__(parent)
        ensure_matplotlib_cjk_font()
        self.fig = fig
        self.canvas = FigureCanvasQTAgg(fig)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        bar = QHBoxLayout()
        hint = QLabel("滚轮缩放 · 左拖平移 · 右拖缩放 · 双击复位 · 右键菜单")
        hint.setStyleSheet("color: #666; font-size: 11px;")
        self.hint = hint
        bar.addWidget(hint)
        bar.addStretch(1)
        btn_reset = QPushButton("Reset View")
        btn_reset.setToolTip("复位到数据范围（等同双击空白）")
        bar.addWidget(btn_reset)
        lay.addLayout(bar)
        lay.addWidget(self.canvas, stretch=1)

        self._nav = PyqtgraphStyleNav(self.canvas, on_right_click=on_right_click)
        btn_reset.clicked.connect(self._nav.reset_view)
        # 窗口打开时 figure 往往已画好：立即锁 home
        try:
            self.canvas.draw()
            self._nav.remember_home_views()
        except Exception:
            pass

    def remember_current_views(self) -> None:
        self._nav.remember_current_views()

    def remember_home_views(self) -> None:
        self._nav.remember_home_views()

    def reset_view(self) -> None:
        self._nav.reset_view()


class MplFigureWindow(QWidget):
    """独立工具窗：承载 Figure + 导航；顶部可选「保存 PNG」。"""

    def __init__(
        self,
        fig,
        title: str = "Figure",
        *,
        save_dir: str | None = None,
        default_name: str = "figure",
    ) -> None:
        super().__init__(None)
        self.setWindowTitle(title)
        self.resize(960, 640)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._fig = fig
        self._save_dir = save_dir
        self._default_name = default_name

        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        btn_save = QPushButton("保存图像…")
        btn_save.setToolTip("保存为 PNG / JPEG / TIFF / PDF / PS / EPS / SVG")
        btn_save.clicked.connect(self._save_png)
        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        bar.addWidget(btn_save)
        bar.addStretch(1)
        bar.addWidget(btn_close)
        lay.addLayout(bar)
        self.host = MplNavCanvas(fig, self)
        lay.addWidget(self.host, stretch=1)

    def _save_png(self) -> None:
        from .export_figure import save_mpl_figure

        save_mpl_figure(
            self,
            self._fig,
            start_dir=self._save_dir or "",
            default_name=self._default_name,
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
        except Exception:
            pass
        super().closeEvent(event)


def show_mpl_figure(
    fig,
    title: str,
    *,
    save_dir: str | None = None,
    default_name: str = "figure",
) -> MplFigureWindow:
    win = MplFigureWindow(
        fig, title=title, save_dir=save_dir, default_name=default_name
    )
    show_modeless_dialog(win, activate=True)
    return win
