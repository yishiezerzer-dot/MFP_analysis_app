from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List


class RecentFilesService:
    def __init__(self, *, app_name: str, max_items: int = 10) -> None:
        """Implement the `__init__` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        self._app_name = str(app_name)
        self._max_items = int(max_items)
        self._paths: List[str] = []
        self._settings_path = self._resolve_settings_path()
        self._load()

    def list_recent(self) -> List[str]:
        """Implement the `list_recent` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        return list(self._paths)

    def add_recent(self, path: str) -> None:
        """Implement the `add_recent` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        p = str(path)
        self._paths = [x for x in self._paths if x != p]
        self._paths.insert(0, p)
        self._paths = self._paths[: self._max_items]
        self._save()

    def clear(self) -> None:
        """Implement the `clear` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        self._paths = []
        self._save()

    def _resolve_settings_path(self) -> Path:
        """Implement the `_resolve_settings_path` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        appdata = os.environ.get("APPDATA") or os.path.expanduser("~")
        base = Path(appdata) / self._app_name
        base.mkdir(parents=True, exist_ok=True)
        return base / "settings.json"

    def _load(self) -> None:
        """Load data required by this function.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        try:
            if not self._settings_path.exists():
                return
            with self._settings_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            items = payload.get("recent_workspaces", [])
            if isinstance(items, list):
                self._paths = [str(p) for p in items if p]
        except Exception:
            self._paths = []

    def _save(self) -> None:
        """Save output/state to persistent storage.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        try:
            payload = {"recent_workspaces": list(self._paths)}
            with self._settings_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            return
