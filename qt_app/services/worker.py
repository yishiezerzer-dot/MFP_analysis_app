from __future__ import annotations

import logging
import os
import threading
import time
import traceback
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from qt_app.services.status_service import StatusService


_logger = logging.getLogger(__name__)
_GROUP_HANDLES: Dict[str, "WorkerHandle"] = {}
_GROUP_TOKENS: Dict[str, str] = {}
_THREADPOOL_READY = False


def _init_threadpool() -> None:
    global _THREADPOOL_READY
    if _THREADPOOL_READY:
        return
    pool = QThreadPool.globalInstance()
    max_threads_raw = os.getenv("QT_APP_MAX_THREADS", "").strip()
    if max_threads_raw:
        try:
            pool.setMaxThreadCount(max(1, int(max_threads_raw)))
        except Exception:
            pass
    else:
        cpu = os.cpu_count() or 4
        pool.setMaxThreadCount(max(2, min(4, cpu)))
    _THREADPOOL_READY = True


class _WorkerSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


class WorkerHandle:
    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


class _Worker(QRunnable):
    def __init__(self, fn: Callable[[WorkerHandle], Any], handle: WorkerHandle, description: str = "") -> None:
        super().__init__()
        self.fn = fn
        self.handle = handle
        self.description = str(description or "")
        self.signals = _WorkerSignals()

    @Slot()
    def run(self) -> None:
        start = time.perf_counter()
        thread_name = threading.current_thread().name
        try:
            if self.handle.cancelled:
                _logger.info("Worker cancelled before start: %s", self.description)
                self.signals.finished.emit()
                return
            _logger.info("Worker start (%s): %s", thread_name, self.description)
            result = self.fn(self.handle)
            self.signals.result.emit(result)
        except Exception:
            _logger.exception("Worker error (%s): %s", thread_name, self.description)
            self.signals.error.emit(traceback.format_exc())
        finally:
            elapsed = time.perf_counter() - start
            _logger.info("Worker finished (%s): %s (%.3fs)", thread_name, self.description, elapsed)
            self.signals.finished.emit()


def run_in_worker(
    fn: Callable[[WorkerHandle], Any],
    on_result: Optional[Callable[[Any], None]] = None,
    on_error: Optional[Callable[[str], None]] = None,
    on_finished: Optional[Callable[[], None]] = None,
    *,
    description: str = "",
    status: Optional[StatusService] = None,
    group: Optional[str] = None,
    cancel_previous: bool = False,
) -> WorkerHandle:
    _init_threadpool()
    handle = WorkerHandle()
    worker = _Worker(fn, handle, description=description)

    token: Optional[str] = None
    if group:
        token = str(uuid4())
        _GROUP_TOKENS[group] = token
        if cancel_previous:
            prev = _GROUP_HANDLES.get(group)
            if prev is not None:
                prev.cancel()
        _GROUP_HANDLES[group] = handle

    def _is_latest() -> bool:
        if not group:
            return True
        if token is None:
            return False
        return _GROUP_TOKENS.get(group) == token

    def _wrap(callback):
        def _inner(*args, **kwargs):
            if not _is_latest():
                return
            callback(*args, **kwargs)

        return _inner

    if on_result is not None:
        worker.signals.result.connect(_wrap(on_result))
    if on_error is not None:
        worker.signals.error.connect(_wrap(on_error))
    if on_finished is not None:
        worker.signals.finished.connect(_wrap(on_finished))

    if status is not None:
        if description:
            status.set_status(str(description))
        status.set_busy(True)

        def _clear_status() -> None:
            status.set_busy(False)

        worker.signals.finished.connect(_clear_status)

    QThreadPool.globalInstance().start(worker)
    return handle
