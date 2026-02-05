from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


# Keep this stable; used for %APPDATA%\<APP_SETTINGS_DIRNAME>\settings.json
APP_SETTINGS_DIRNAME = "MFP lab analysis tool"
SETTINGS_FILENAME = "settings.json"


def _appdata_dir() -> Path:
    # Windows: %APPDATA% (Roaming)
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata)

    # Fallbacks (best-effort)
    home = Path.home()
    candidate = home / "AppData" / "Roaming"
    return candidate if candidate.exists() else home


def settings_path() -> Path:
    return _appdata_dir() / APP_SETTINGS_DIRNAME / SETTINGS_FILENAME


def load_settings() -> Dict[str, Any]:
    """Load persistent user settings.

    Returns a dict with at least:
        - fiji_exe_path: str | None
        - last_macro_dir: str | None
        - last_microscopy_dir: str | None
        - microscopy_run_headless: bool
    """
    p = settings_path()
    try:
        if not p.exists() or not p.is_file():
            return {
                "fiji_exe_path": None,
                "last_macro_dir": None,
                "last_microscopy_dir": None,
                "microscopy_run_headless": True,
                "theme": "flatly",
            }
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("settings.json must be an object")
    except Exception:
        return {
            "fiji_exe_path": None,
            "last_macro_dir": None,
            "last_microscopy_dir": None,
            "microscopy_run_headless": True,
            "theme": "flatly",
        }

    out: Dict[str, Any] = {
        "fiji_exe_path": None,
        "last_macro_dir": None,
        "last_microscopy_dir": None,
        "microscopy_run_headless": True,
        "theme": "flatly",
    }

    for k in ("fiji_exe_path", "last_macro_dir", "last_microscopy_dir"):
        v = data.get(k)
        out[k] = None if v in (None, "") else str(v)

    try:
        v = data.get("microscopy_run_headless", True)
        out["microscopy_run_headless"] = bool(v) if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "yes", "on")
    except Exception:
        out["microscopy_run_headless"] = True
    try:
        theme = str(data.get("theme", "flatly") or "flatly").strip().lower()
        if theme in ("dark", "darkly"):
            theme = "darkly"
        elif theme in ("light", "flatly"):
            theme = "flatly"
        out["theme"] = theme
    except Exception:
        out["theme"] = "flatly"
    return out


def save_settings(settings: Dict[str, Any]) -> None:
    """Persist settings to %APPDATA%\\<APP_SETTINGS_DIRNAME>\\settings.json."""
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)

    safe: Dict[str, Any] = {}
    for k in ("fiji_exe_path", "last_macro_dir", "last_microscopy_dir"):
        v = (settings or {}).get(k)
        safe[k] = None if v in (None, "") else str(v)

    try:
        safe["microscopy_run_headless"] = bool((settings or {}).get("microscopy_run_headless", True))
    except Exception:
        safe["microscopy_run_headless"] = True

    try:
        theme = str((settings or {}).get("theme", "flatly") or "flatly").strip().lower()
        if theme in ("dark", "darkly"):
            theme = "darkly"
        elif theme in ("light", "flatly"):
            theme = "flatly"
        safe["theme"] = theme
    except Exception:
        safe["theme"] = "flatly"

    p.write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_imagej_exe_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    s = str(path).strip().strip('"')
    if not s:
        return None
    p = Path(s)
    try:
        if not (p.exists() and p.is_file()):
            return None
    except Exception:
        return None
    if p.suffix.lower() != ".exe":
        return None
    return str(p)


def guess_imagej_initial_dirs() -> list[str]:
    """Common Fiji/ImageJ install locations on Windows."""
    candidates = [
        r"C:\\Fiji.app\\ImageJ-win64.exe",
        r"C:\\Fiji.app\\fiji-windows-x64.exe",
        r"C:\\Program Files\\Fiji.app\\ImageJ-win64.exe",
        r"C:\\Program Files\\ImageJ\\ImageJ.exe",
        r"C:\\Program Files\\ImageJ\\ImageJ64.exe",
    ]

    out: list[str] = []
    for c in candidates:
        try:
            p = Path(c)
            if p.exists():
                out.append(str(p.parent))
        except Exception:
            continue

    # Fall back to C:\ if none exist
    if not out:
        out.append(r"C:\\")
    # Deduplicate while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for d in out:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq
