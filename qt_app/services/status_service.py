from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QTimer


class StatusService:
    def __init__(
        self,
        *,
        set_text: Callable[[str], None],
        set_busy: Callable[[bool], None],
        set_progress: Callable[[int], None],
    ) -> None:
        """Implement the `__init__` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        self._set_text = set_text
        self._set_busy = set_busy
        self._set_progress = set_progress
        self._busy_count = 0

    def set_status(self, text: str) -> None:
        """Implement the `set_status` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        QTimer.singleShot(0, lambda: self._set_text(str(text)))

    def set_busy(self, busy: bool) -> None:
        """Implement the `set_busy` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        def _apply() -> None:
            """Implement the `_apply` behavior for this module.

            Text-only documentation note: modify internal logic here to change behavior.
            """
            if busy:
                self._busy_count += 1
                self._set_busy(True)
                return
            self._busy_count = max(0, self._busy_count - 1)
            if self._busy_count == 0:
                self._set_busy(False)

        QTimer.singleShot(0, _apply)

    def set_progress(self, value: Optional[int]) -> None:
        """Implement the `set_progress` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        def _apply() -> None:
            """Implement the `_apply` behavior for this module.

            Text-only documentation note: modify internal logic here to change behavior.
            """
            if value is None:
                self._set_progress(0)
            else:
                self._set_progress(int(value))

        QTimer.singleShot(0, _apply)
