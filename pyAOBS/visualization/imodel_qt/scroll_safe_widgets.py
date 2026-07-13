"""嵌套在 QScrollArea 等可滚动容器内时，避免滚轮误改未聚焦控件。"""

from __future__ import annotations

from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QComboBox


class ComboBoxNoScrollUnlessFocused(QComboBox):
    """无键盘焦点时不处理滚轮，事件向上传递以便外层滚动；聚焦后可滚轮切项。"""

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self.hasFocus():
            event.ignore()
            return
        super().wheelEvent(event)
