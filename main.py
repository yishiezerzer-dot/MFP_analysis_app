from __future__ import annotations

import os

from lab_gui.app import main


def _maybe_run_polymer_selfchecks() -> None:
    # Hidden developer hook; no UI changes.
    if str(os.environ.get("LAB_GUI_RUN_POLYMER_SELFTESTS", "")).strip() not in ("1", "true", "True", "yes", "YES"):
        return
    try:
        from lab_gui.lcms_polymer_match import run_polymer_self_checks

        res = run_polymer_self_checks()
        print("[polymer-selfcheck]", res)
    except Exception as exc:
        print("[polymer-selfcheck] failed:", exc)


def _maybe_run_overlay_selfchecks() -> None:
    if str(os.environ.get("LAB_GUI_RUN_OVERLAY_SELFTESTS", "")).strip() not in ("1", "true", "True", "yes", "YES"):
        return
    try:
        from lab_gui.app import run_overlay_self_checks

        res = run_overlay_self_checks()
        print("[overlay-selfcheck]", res)
    except Exception as exc:
        print("[overlay-selfcheck] failed:", exc)


if __name__ == "__main__":
    _maybe_run_polymer_selfchecks()
    _maybe_run_overlay_selfchecks()
    raise SystemExit(main())
