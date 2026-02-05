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
        self._set_text = set_text
        self._set_busy = set_busy
        self._set_progress = set_progress
        self._busy_count = 0

    def set_status(self, text: str) -> None:
        QTimer.singleShot(0, lambda: self._set_text(str(text)))

    def set_busy(self, busy: bool) -> None:
        def _apply() -> None:
            if busy:
                self._busy_count += 1
                self._set_busy(True)
                return
            self._busy_count = max(0, self._busy_count - 1)
            if self._busy_count == 0:
                self._set_busy(False)

        QTimer.singleShot(0, _apply)

    def set_progress(self, value: Optional[int]) -> None:
        def _apply() -> None:
            if value is None:
                self._set_progress(0)
            else:
                self._set_progress(int(value))

        QTimer.singleShot(0, _apply)
