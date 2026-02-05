from __future__ import annotations

from PySide6.QtWidgets import QMessageBox, QWidget


class DialogService:
    def __init__(self, parent: QWidget) -> None:
        self._parent = parent

    def info(self, title: str, message: str) -> None:
        QMessageBox.information(self._parent, str(title), str(message))

    def warn(self, title: str, message: str) -> None:
        QMessageBox.warning(self._parent, str(title), str(message))

    def error(self, title: str, message: str) -> None:
        QMessageBox.critical(self._parent, str(title), str(message))

    def confirm(self, title: str, message: str) -> bool:
        res = QMessageBox.question(self._parent, str(title), str(message))
        return res == QMessageBox.StandardButton.Yes
