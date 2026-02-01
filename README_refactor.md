# Refactor notes

This repo was refactored from a single large `gui_analyze_mzml.py` into a small package under `lab_gui/`.

## Entry points

- Run the app (recommended):
  - `py main.py`

## What moved

- Main GUI implementation: `lab_gui/app.py`
- Shared UI widgets (tooltips): `lab_gui/ui_widgets.py`
- Export editor window: `lab_gui/export_editor.py`
- UI-free models:
  - `lab_gui/lcms_model.py`
  - `lab_gui/ftir_model.py`
- UI-free IO/parsing:
  - `lab_gui/lcms_io.py`
  - `lab_gui/ftir_io.py`

## FTIR

`lab_gui/ftir_analysis.py` is used by the GUI for FTIR preprocessing and peak picking.

## Compatibility

This workspace does not include the old `gui_analyze_mzml.py` shim; use `main.py`.
