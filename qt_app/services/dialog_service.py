from __future__ import annotations

from PySide6.QtWidgets import QMessageBox, QWidget


class DialogService:
    def __init__(self, parent: QWidget) -> None:
        """Implement the `__init__` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        self._parent = parent

    def info(self, title: str, message: str) -> None:
        """Implement the `info` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        QMessageBox.information(self._parent, str(title), str(message))

    def warn(self, title: str, message: str) -> None:
        """Implement the `warn` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        QMessageBox.warning(self._parent, str(title), str(message))

    def error(self, title: str, message: str) -> None:
        """Implement the `error` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        QMessageBox.critical(self._parent, str(title), str(message))

    def confirm(self, title: str, message: str) -> bool:
        """Implement the `confirm` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        res = QMessageBox.question(self._parent, str(title), str(message))
        return res == QMessageBox.StandardButton.Yes
