from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QFileDialog,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class PlotPanel(QWidget):
    def __init__(
        self,
        *,
        on_status: Optional[Callable[[str], None]] = None,
        on_click: Optional[Callable[[float, float, int, bool], None]] = None,
        on_move: Optional[Callable[[float, float], None]] = None,
        on_release: Optional[Callable[[float, float], None]] = None,
        move_debounce_ms: int = 16,
    ) -> None:
        super().__init__()
        self._on_status = on_status
        self._on_click = on_click
        self._on_move = on_move
        self._on_release = on_release
        self._move_debounce_ms = max(0, int(move_debounce_ms))
        self._pending_move: Optional[Tuple[float, float]] = None
        self._move_timer = QTimer(self)
        self._move_timer.setSingleShot(True)
        self._move_timer.timeout.connect(self._flush_move)

        self._plot = pg.PlotWidget()
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.setMouseEnabled(x=True, y=True)

        self._vline = pg.InfiniteLine(angle=90, movable=False)
        self._hline = pg.InfiniteLine(angle=0, movable=False)
        self._crosshair_enabled = False
        self._items: Dict[str, pg.PlotDataItem] = {}
        self._legend = None
        self._annotations: list[pg.TextItem] = []
        self._region_item: Optional[pg.LinearRegionItem] = None
        self._region_callback: Optional[Callable[[Tuple[float, float]], None]] = None

        self._install_toolbar()
        self._install_mouse_handlers()

    def _install_toolbar(self) -> None:
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)

        btn_reset = QPushButton("Reset View")
        btn_reset.clicked.connect(self._reset_view)
        toolbar.addWidget(btn_reset)

        btn_auto = QPushButton("Autoscale")
        btn_auto.clicked.connect(self._auto_scale)
        toolbar.addWidget(btn_auto)

        self._btn_crosshair = QToolButton()
        self._btn_crosshair.setText("Crosshair")
        self._btn_crosshair.setCheckable(True)
        self._btn_crosshair.toggled.connect(self._toggle_crosshair)
        toolbar.addWidget(self._btn_crosshair)

        btn_export = QPushButton("Export Image")
        btn_export.clicked.connect(self._export_image)
        toolbar.addWidget(btn_export)

        toolbar.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(toolbar)
        layout.addWidget(self._plot, 1)

    def _install_mouse_handlers(self) -> None:
        scene = self._plot.scene()
        scene.sigMouseClicked.connect(self._on_mouse_clicked)
        try:
            scene.sigMouseReleased.connect(self._on_mouse_released)
        except Exception:
            pass
        self._proxy = pg.SignalProxy(scene.sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved)

    def _on_mouse_clicked(self, ev) -> None:
        if ev.double():
            self._reset_view()
        if self._on_click is None:
            return
        pos = ev.scenePos()
        if not self._plot.sceneBoundingRect().contains(pos):
            return
        vb = self._plot.plotItem.vb
        mouse_point = vb.mapSceneToView(pos)
        x = float(mouse_point.x())
        y = float(mouse_point.y())
        button = int(ev.button())
        if button == 1:
            btn = 1
        elif button == 2:
            btn = 2
        elif button == 4:
            btn = 3
        else:
            btn = 0
        self._on_click(x, y, btn, bool(ev.double()))

    def _on_mouse_moved(self, evt) -> None:
        pos = evt[0]
        if not self._plot.sceneBoundingRect().contains(pos):
            return
        vb = self._plot.plotItem.vb
        mouse_point = vb.mapSceneToView(pos)
        x = float(mouse_point.x())
        y = float(mouse_point.y())
        if self._move_debounce_ms:
            self._pending_move = (x, y)
            if not self._move_timer.isActive():
                self._move_timer.start(self._move_debounce_ms)
            return
        self._handle_move(x, y)

    def _flush_move(self) -> None:
        if self._pending_move is None:
            return
        x, y = self._pending_move
        self._pending_move = None
        self._handle_move(x, y)

    def _handle_move(self, x: float, y: float) -> None:
        if self._crosshair_enabled:
            self._vline.setPos(x)
            self._hline.setPos(y)
            if self._on_status:
                self._on_status(f"x={x:.4g}, y={y:.4g}")
        if self._on_move:
            self._on_move(x, y)

    def _on_mouse_released(self, ev) -> None:
        if self._on_release is None:
            return
        pos = ev.scenePos()
        if not self._plot.sceneBoundingRect().contains(pos):
            return
        vb = self._plot.plotItem.vb
        mouse_point = vb.mapSceneToView(pos)
        x = float(mouse_point.x())
        y = float(mouse_point.y())
        self._on_release(x, y)

    def _reset_view(self) -> None:
        self._plot.enableAutoRange()
        self._plot.autoRange()

    def _auto_scale(self) -> None:
        self._plot.autoRange()

    def _toggle_crosshair(self, enabled: bool) -> None:
        self._crosshair_enabled = bool(enabled)
        if self._crosshair_enabled:
            self._plot.addItem(self._vline, ignoreBounds=True)
            self._plot.addItem(self._hline, ignoreBounds=True)
        else:
            self._plot.removeItem(self._vline)
            self._plot.removeItem(self._hline)
            if self._on_status:
                self._on_status("Ready")

    def _export_image(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot Image",
            "",
            "PNG (*.png);;JPG (*.jpg *.jpeg);;TIFF (*.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return
        self.export_image_to_path(path)

    def export_image_to_path(self, path: str) -> None:
        try:
            exporter = ImageExporter(self._plot.plotItem)
            exporter.export(fileName=str(path))
            if self._on_status:
                self._on_status("Image exported")
        except Exception as exc:
            if self._on_status:
                self._on_status(f"Export failed: {exc}")

    def set_grid(self, show: bool, *, alpha: float = 0.2) -> None:
        try:
            self._plot.showGrid(x=bool(show), y=bool(show), alpha=float(alpha))
        except Exception:
            pass

    def enable_region_select(
        self,
        enabled: bool,
        *,
        initial: Optional[Tuple[float, float]] = None,
        on_change: Optional[Callable[[Tuple[float, float]], None]] = None,
    ) -> None:
        if not enabled:
            if self._region_item is not None:
                try:
                    self._plot.removeItem(self._region_item)
                except Exception:
                    pass
            self._region_item = None
            self._region_callback = None
            return

        if self._region_item is None:
            try:
                self._region_item = pg.LinearRegionItem(orientation=pg.LinearRegionItem.Vertical)
                self._region_item.sigRegionChanged.connect(self._on_region_changed)
                self._plot.addItem(self._region_item)
            except Exception:
                self._region_item = None
                return

        self._region_callback = on_change
        if initial is not None and self._region_item is not None:
            try:
                self._region_item.setRegion((float(initial[0]), float(initial[1])))
            except Exception:
                pass

    def _on_region_changed(self) -> None:
        if self._region_item is None or self._region_callback is None:
            return
        try:
            lo, hi = self._region_item.getRegion()
            self._region_callback((float(lo), float(hi)))
        except Exception:
            return

    def clear(self) -> None:
        self._plot.clear()
        self._items.clear()
        self._legend = None
        self._annotations = []
        if self._crosshair_enabled:
            self._plot.addItem(self._vline, ignoreBounds=True)
            self._plot.addItem(self._hline, ignoreBounds=True)

    def clear_annotations(self) -> None:
        for item in list(self._annotations):
            try:
                self._plot.removeItem(item)
            except Exception:
                pass
        self._annotations = []

    def add_annotation(self, x: float, y: float, text: str, *, color: str = "#111111") -> None:
        try:
            item = pg.TextItem(text=str(text), color=color, anchor=(0.5, 1.0))
            item.setPos(float(x), float(y))
            self._plot.addItem(item)
            self._annotations.append(item)
        except Exception:
            pass

    def plot_line(self, x, y, name: str, pen=None) -> None:
        key = str(name)
        item = self._items.get(key)
        if item is None:
            item = self._plot.plot(x, y, name=key, pen=pen)
            self._items[key] = item
            return
        item.setData(x, y)
        if pen is not None:
            item.setPen(pen)

    def plot_scatter(self, x, y, name: str) -> None:
        key = str(name)
        item = self._items.get(key)
        if item is None:
            item = self._plot.plot(x, y, name=key, pen=None, symbol="o", symbolSize=6)
            self._items[key] = item
            return
        item.setData(x, y, pen=None, symbol="o", symbolSize=6)

    def set_title(self, title: str) -> None:
        self._plot.setTitle(str(title))

    def set_labels(self, xlabel: str, ylabel: str) -> None:
        self._plot.setLabel("bottom", str(xlabel))
        self._plot.setLabel("left", str(ylabel))

    def add_legend(self) -> None:
        if self._legend is None:
            self._legend = self._plot.addLegend()
