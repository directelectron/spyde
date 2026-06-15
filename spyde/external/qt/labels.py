from PySide6 import QtWidgets
from PySide6.QtCore import Signal, Qt

from spyde.external.pyqtgraph.scale_bar import tex_to_html


class ClickableLabel(QtWidgets.QLabel):
    clicked = Signal()

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.rect().contains(
            e.position().toPoint()
        ):
            self.clicked.emit()
        super().mouseReleaseEvent(e)


class EditableLabel(QtWidgets.QWidget):
    editingFinished = Signal(str)

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._raw_text = text

        self._label = ClickableLabel(parent=self)
        self._label.setTextFormat(Qt.TextFormat.RichText)
        self._label.setText(tex_to_html(self._raw_text))

        self._line_edit = QtWidgets.QLineEdit(self)

        self._stack = QtWidgets.QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._stack.addWidget(self._label)
        self._stack.addWidget(self._line_edit)
        self._stack.setCurrentWidget(self._label)

        self._label.clicked.connect(self._start_editing)
        self._line_edit.editingFinished.connect(self._finish_editing)

        # Themed to match the rest of the app (accent hover + shared input look)
        # rather than the old hardcoded bluish tint. Tokens live in qt.style.
        from spyde.qt.style import EDITABLE_LABEL_QSS, INPUT_QSS
        self._label.setObjectName("editableLabelPart")
        self._label.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._label.setStyleSheet(EDITABLE_LABEL_QSS)
        self._line_edit.setStyleSheet(INPUT_QSS)

        self.previous_text = text

    def _start_editing(self):
        self._line_edit.setText(self._raw_text)
        self._stack.setCurrentWidget(self._line_edit)
        self._line_edit.selectAll()
        self._line_edit.setFocus()
        self.previous_text = self._raw_text

    def _finish_editing(self):
        new_text = self._line_edit.text()
        self._raw_text = new_text
        self._label.setText(tex_to_html(new_text))
        self._stack.setCurrentWidget(self._label)
        self.editingFinished.emit(new_text)

    def setText(self, text: str):
        self._raw_text = text
        self._label.setText(tex_to_html(text))

    def text(self) -> str:
        return self._raw_text
