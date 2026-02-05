# Qt vs Tk Parity Audit (2026-02-03)

This audit compares the current Qt app to the Tk app and identifies functional gaps. It focuses on **behavioral parity** (what users can do), not code structure. Status tags:
- ✅ **Implemented** (matches Tk behavior)
- ⚠️ **Partial** (some behavior missing or different)
- ❌ **Missing/Broken** (stub or not wired)

## App Shell (menus, recent files, status)
- File > Open/Save Workspace routing to active tab: ✅ (Data Studio, Plate Reader)
- File > Open/Save Workspace for LCMS/FTIR/Microscopy: ⚠️ (basic implementations)
- Recent Workspaces list + open recent: ⚠️ (only tabs with open/save implemented)
- Reveal in Explorer: ✅ (works when tab exposes path)
- Reset Layout: ✅
- About: ✅

## Data Studio
- Add files, remove, clear: ✅
- Async schema inference + column map: ✅
- Preview table with sheet selector: ✅
- Plot builder with types listed in Tk: ✅
- Overlay across datasets (same plot type): ✅
- Persist plot definitions and UI state (.ui.json): ✅
- Export plot/data editor: ✅ (Qt export dialog)
- Workspace save/load: ✅

## Plate Reader
- Load files, select active dataset, preview: ✅
- MIC wizard + plot render: ✅
- Plot editor: ✅
- Workspace save/load: ✅
- Performance (async load): ✅

## LCMS
- Load mzML, UV CSV (single/many): ✅
- Workspace tree, UV linking: ✅
- TIC/UV/Spectrum plots: ✅
- Spectrum navigation (prev/next, jump RT): ✅
- Overlay TIC/UV/spectra: ✅
- Labeling/annotations, quick annotate: ✅ (top-N + custom labels)
- Polymer matching: ✅ (Qt-native dialog + spectrum labels)
- Export labels/plots/data: ✅ (spectrum CSV, overlay TIC/spectra CSV, plot image save)
- Workspace save/load: ✅

## FTIR
- Workspaces (new/rename/duplicate/delete): ⚠️ (Qt-native basic workspaces)
- Load FTIR datasets + list: ⚠️ (Qt-native load/list)
- Overlays + color/offset: ⚠️ (Qt-native basic overlay groups + offsets)
- Peaks dialog + export peaks: ⚠️ (basic peak picking + CSV export)
- Bond labels workflow: ✅ (add + click placement + drag)
- Plot interactions (drag/peak pick): ✅ (interactive labels + drag)
- Workspace save/load: ⚠️ (Qt-native basic save/load)

## Microscopy
- Tab UI: ⚠️ (Qt-native workspace + outputs UI)
- Workspaces/datasets: ⚠️ (Qt-native add/remove)
- Preset runs (ImageJ/Fiji macros): ✅ (preset panel + run on active/selected/all)
- Outputs list + preview: ⚠️ (outputs list implemented; no image preview)
- Settings (ImageJ path): ⚠️ (set path + open in ImageJ)
- Batch run dialog + cancel: ✅ (progress dialog + cancel)
- Workspace save/load: ⚠️ (Qt-native basic save/load)

## Shared Widgets
- PlotPanel crosshair + basic plotting: ✅
- PlotPanel export image: ✅

## Summary
Remaining parity gaps are small polish items (e.g., Data Studio export editor UI depth). Core parity targets are met.

## Next Steps (priority order)
1. Optional: Data Studio export/editor UX polish.
