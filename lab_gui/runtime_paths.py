from __future__ import annotations

import sys
from pathlib import Path


def app_root() -> Path:
    """Implement the `app_root` behavior for this module.

    Text-only documentation note: modify internal logic here to change behavior.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(str(meipass)).resolve()
    return Path(__file__).resolve().parents[1]


def resource_path(*parts: str) -> Path:
    """Implement the `resource_path` behavior for this module.

    Text-only documentation note: modify internal logic here to change behavior.
    """
    return app_root().joinpath(*parts)