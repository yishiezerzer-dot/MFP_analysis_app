from __future__ import annotations

import os
import subprocess
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QMenu,
    QProgressBar,
    QStatusBar,
    QTabWidget,
    QWidget,
)

from qt_app.adapters import DataStudioAdapter, PlateReaderAdapter
from qt_app.services import DialogService, RecentFilesService, StatusService
from qt_app.services.worker import run_in_worker
from qt_app.tabs.data_studio_tab import DataStudioTab
from qt_app.tabs.ftir_tab import FTIRTab
from qt_app.tabs.lcms_tab import LCMSTab
from qt_app.tabs.microscopy_tab import MicroscopyTab
from qt_app.tabs.plate_reader_tab import PlateReaderTab


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MFP lab analysis tool (Qt)")
        self.resize(1400, 850)

        self._recent_menu: QMenu | None = None
        self._status_label = QLabel("Ready")
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)

        self.status_service = StatusService(
            set_text=self._status_label.setText,
            set_busy=self._set_busy,
            set_progress=self._set_progress,
        )
        self.dialog_service = DialogService(self)
        self.recent_files = RecentFilesService(app_name="MFP lab analysis tool")

        self._init_ui()

    def _init_ui(self) -> None:
        tabs = QTabWidget()
        plate_adapter = PlateReaderAdapter(status=self.status_service, dialogs=self.dialog_service)
        data_studio_adapter = DataStudioAdapter()

        tabs.addTab(LCMSTab(self.status_service, self.dialog_service, run_in_worker), "LCMS")
        tabs.addTab(FTIRTab(self.status_service, self.dialog_service, run_in_worker), "FTIR")
        tabs.addTab(PlateReaderTab(self.status_service, self.dialog_service, run_in_worker, plate_adapter), "Plate Reader")
        tabs.addTab(MicroscopyTab(self.status_service, self.dialog_service, run_in_worker), "Microscopy")
        tabs.addTab(DataStudioTab(self.status_service, self.dialog_service, run_in_worker, data_studio_adapter), "Data Studio")
        self.setCentralWidget(tabs)
        self._tabs = tabs

        self._build_menu()
        self._build_status_bar()

    def _build_menu(self) -> None:
        menu = self.menuBar()

        file_menu = menu.addMenu("File")
        open_action = QAction("Open Workspace", self)
        open_action.triggered.connect(self._open_workspace)
        file_menu.addAction(open_action)

        save_action = QAction("Save Workspace", self)
        save_action.triggered.connect(self._save_workspace)
        file_menu.addAction(save_action)

        self._recent_menu = file_menu.addMenu("Recent Workspaces")
        self._refresh_recent_menu()

        reveal_action = QAction("Reveal in Explorer", self)
        reveal_action.triggered.connect(self._reveal_in_explorer)
        file_menu.addAction(reveal_action)

        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = menu.addMenu("View")
        reset_action = QAction("Reset Layout", self)
        reset_action.triggered.connect(self._reset_layout)
        view_menu.addAction(reset_action)

        help_menu = menu.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._about)
        help_menu.addAction(about_action)

    def _build_status_bar(self) -> None:
        bar = QStatusBar()
        bar.addWidget(self._status_label, 1)
        bar.addPermanentWidget(self._progress)
        self.setStatusBar(bar)

    def _set_busy(self, busy: bool) -> None:
        self._progress.setVisible(bool(busy))

    def _set_progress(self, value: int) -> None:
        self._progress.setVisible(True)
        if value <= 0:
            self._progress.setRange(0, 0)
        else:
            self._progress.setRange(0, 100)
            self._progress.setValue(int(value))

    def _set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def _open_workspace(self) -> None:
        tab = getattr(self, "_tabs", None)
        if tab is None:
            self.status_service.set_status("Open Workspace (stub)")
            return
        current = tab.currentWidget()
        if current is not None and hasattr(current, "open_workspace"):
            path = current.open_workspace()
            if path:
                self.recent_files.add_recent(path)
                self._refresh_recent_menu()
            return
        self.status_service.set_status("Open Workspace (stub)")

    def _save_workspace(self) -> None:
        tab = getattr(self, "_tabs", None)
        if tab is None:
            self.status_service.set_status("Save Workspace (stub)")
            return
        current = tab.currentWidget()
        if current is not None and hasattr(current, "save_workspace"):
            path = current.save_workspace()
            if path:
                self.recent_files.add_recent(path)
                self._refresh_recent_menu()
            return
        self.status_service.set_status("Save Workspace (stub)")

    def _reset_layout(self) -> None:
        tab = getattr(self, "_tabs", None)
        if tab is None:
            self.status_service.set_status("Layout reset")
            return
        current = tab.currentWidget()
        if current is not None and hasattr(current, "reset_layout"):
            try:
                current.reset_layout()
                self.status_service.set_status("Layout reset")
                return
            except Exception:
                pass
        self.status_service.set_status("Layout reset")

    def _about(self) -> None:
        self.dialog_service.info("About", "MFP lab analysis tool — Qt preview")

    def _refresh_recent_menu(self) -> None:
        if self._recent_menu is None:
            return
        self._recent_menu.clear()
        items = self.recent_files.list_recent()
        if not items:
            self._recent_menu.addAction(QAction("(empty)", self))
            return
        for path in items:
            action = QAction(path, self)
            action.triggered.connect(lambda _checked=False, p=path: self._open_recent(p))
            self._recent_menu.addAction(action)

    def _open_recent(self, path: str) -> None:
        tab = getattr(self, "_tabs", None)
        if tab is None:
            self.status_service.set_status(f"Open recent (stub): {path}")
            return
        current = tab.currentWidget()
        if current is not None and hasattr(current, "open_workspace_path"):
            current.open_workspace_path(path)
            self.recent_files.add_recent(path)
            self._refresh_recent_menu()
            return
        self.status_service.set_status(f"Open recent (stub): {path}")

    def _reveal_in_explorer(self) -> None:
        tab = getattr(self, "_tabs", None)
        if tab is None:
            return
        current = tab.currentWidget()
        path = None
        if current is not None and hasattr(current, "get_last_workspace_path"):
            path = current.get_last_workspace_path()
        if not path:
            self.status_service.set_status("Reveal in Explorer (no workspace)")
            return
        try:
            p = Path(str(path))
            if p.exists():
                if os.name == "nt":
                    subprocess.Popen(["explorer", "/select,", str(p)])
                else:
                    os.startfile(str(p))
        except Exception:
            self.status_service.set_status("Reveal in Explorer failed")
