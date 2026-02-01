# mzML analysis code

A small, local mzML analyzer for LC–MS(/MS) data.

## What it produces
Given an `.mzML`, it writes an output folder containing:
- `summary.json`: high-level dataset summary and per-MS-level stats
- `tic.csv`, `bpc.csv`: chromatograms (retention time vs intensity)
- `chromatograms.png`: TIC and BPC plots
- `ms1_mz_hist.png`: MS1 m/z histogram (from aggregated centroided peaks)
- `intensity_hist.png`: log10 intensity histogram (all peaks)
- `spectra.csv` (optional): per-spectrum table (scan id, ms level, RT, peaks count, etc.)

## Install

```powershell
C:/Users/owner/AppData/Local/Programs/Python/Python312/python.exe -m pip install -r requirements.txt
```

## Run

### GUI (interactive chromatogram viewer)

The main entry point for running the analysis GUI is `main.py`:

```powershell
C:/Users/owner/AppData/Local/Programs/Python/Python312/python.exe main.py
```

To avoid a console window entirely, you can run with `pythonw.exe`:

```powershell
C:/Users/owner/AppData/Local/Programs/Python/Python312/pythonw.exe main.py
```

In the viewer:
- Open an mzML file
- Choose polarity (All/Positive/Negative) in the **View** tab
- Click the TIC to display the MS1 spectrum (m/z vs intensity) at the nearest RT
- Use `Graph Settings…` (View tab or Tools menu) to change fonts and axis ranges
- Use `Annotate Peaks…` (Annotate tab or Tools menu) to label spectrum peaks with m/z

Useful extras:
- Navigation: use the **Navigate** tab buttons, or keyboard `Left/Right`, `Home/End`, and “Jump to RT”
- UV chromatogram: `Load UV CSV…`, then click UV peaks to jump to the nearest MS1 RT
- Polymer/reaction IDs: `Polymer Match…` supports optional `−CO2`, optional oxidation `−2H`, optional `2M` dimers, and optional extra adducts
- Edit/delete labels: right-click (or double-click) any label in Spectrum/UV to rename or delete
- Export: `Export Spectrum CSV…` writes a CSV with m/z, intensity, and the currently active labels
- Plot export: `Export…` → `Save TIC/Spectrum/UV Plot…` opens an Export Editor where you can adjust size/fonts/limits, drag/edit/delete labels, optionally number labels with an in-plot table (editable + movable via x/y/w/h), and edit plot/label/table colors

## Notes
- Works best on centroided data. Profile data can be large; use `--max-spectra` to sample.
- If your vendor export stores RT in seconds, you can use `--rt-unit seconds`.

## FTIR

If you use the FTIR features in the GUI (preprocessing + peak picking), `lab_gui/ftir_analysis.py` is required.
