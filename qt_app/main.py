from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qt_app.main_window import MainWindow


def _init_logging() -> None:
    log_root = Path(os.getenv("APPDATA", str(Path.home()))) / "MainMFP"
    log_root.mkdir(parents=True, exist_ok=True)
    log_file = log_root / "qt_app.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def _install_excepthook() -> None:
    logger = logging.getLogger("qt_app")

    def _hook(exc_type, exc, tb) -> None:
        logger.exception("Unhandled exception", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _hook


def main() -> int:
    _init_logging()
    _install_excepthook()
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
