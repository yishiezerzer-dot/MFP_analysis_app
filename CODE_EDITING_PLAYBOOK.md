# Main MFP Code Editing Playbook

This playbook explains how to modify the code safely and keep it understandable for future readers.

## 1) Fast orientation

Read these in order:

1. `main.py` — app bootstrap and optional self-check hooks.
2. `lab_gui/app.py` — main Tk workflow orchestration and persistence helpers.
3. `qt_app/main.py` + `qt_app/main_window.py` — Qt startup and shell.
4. `CODEBASE_EXPLAINER.md` — per-module, per-function purpose + change hints.

## 2) Where to change what

- **Data loading/parsing:** `lab_gui/*_io.py`
- **Domain models/state:** `lab_gui/*_model.py`
- **Analysis logic:** `lab_gui/ftir_analysis.py`, `lab_gui/lcms_polymer_match.py`
- **Tk views/UI behavior:** `lab_gui/*_view.py`, `lab_gui/app.py`
- **Qt parity layer:** `qt_app/tabs/*`, `qt_app/adapters/*`, `qt_app/services/*`
- **Shared visuals:** `lab_gui/ui_theme.py`, `lab_gui/ui_widgets.py`, `qt_app/widgets/*`

## 3) Safe modification pattern

For any feature change:

1. Update IO assumptions first (if input/output format changes).
2. Update model fields and defaults.
3. Update analysis logic.
4. Update UI presentation and controls.
5. Update adapters/services if Qt and Tk contracts differ.
6. Run a smoke flow and verify exported output.

## 4) Commenting/docstring style used

- Module docstring: what the file owns.
- Function docstring: purpose + modification intent.
- Complex block comments: why a branch or heuristic exists.
- Keep comments behavior-accurate; remove stale text quickly.

## 5) Typical customizations

- **Change thresholds:** edit matching/filtering logic in analysis modules and retest edge cases.
- **Add export field:** update model snapshot builder + export writer + reader counterpart.
- **Adjust UI defaults:** update settings loader/saver and view initialization.
- **Add new tab/workflow:** add model/io pair, create tab view, wire into `qt_app/main_window.py` and (if needed) `lab_gui/app.py`.

## 6) Non-behavior documentation rule

When doing “text-only” updates:

- Add/adjust docstrings, comments, and Markdown guides only.
- Do not modify control flow, constants, conditions, or function signatures.
- Verify syntax after edits.
