from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


ROOT = Path.cwd().resolve()

datas = []
if (ROOT / "assets").exists():
    datas.append((str(ROOT / "assets"), "assets"))
if (ROOT / "lab_gui" / "macros").exists():
    datas.append((str(ROOT / "lab_gui" / "macros"), "lab_gui/macros"))

datas += collect_data_files("ttkbootstrap", include_py_files=False)

hiddenimports = []
hiddenimports += collect_submodules("pyteomics")
hiddenimports += collect_submodules("openpyxl")


a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MainMFP",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="MainMFP",
)