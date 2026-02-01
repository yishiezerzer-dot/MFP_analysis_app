from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from lab_gui.external_tools import best_effort_close_process_log, run_fiji_macro
from lab_gui.microscopy_tab import MicroscopyView
from lab_gui.settings import load_settings, validate_imagej_exe_path


def main() -> int:
    settings = load_settings()
    exe = validate_imagej_exe_path(settings.get("fiji_exe_path"))
    if not exe:
        print("No valid fiji_exe_path in settings.json")
        return 2

    work = Path(tempfile.mkdtemp(prefix="mfp_microscopy_smoke_"))
    out_dir = work / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic 8-bit image: one bright blob.
    img_path = work / "test.png"
    im = Image.new("L", (128, 128), 0)
    for y in range(40, 88):
        for x in range(50, 78):
            im.putpixel((x, y), 220)
    im.save(img_path)

    params = {
        "threshold_method": "Otsu",
        "manual_min": "50",
        "manual_max": "200",
        "min_size": "10",
        "max_size": "Infinity",
        "circ_min": "0.00",
        "circ_max": "1.00",
        "exclude_edge": True,
        "overlay": True,
    }

    macro_text = MicroscopyView._render_macro(None, preset_id="droplet_count", params=params)
    macro_path = out_dir / "preset.ijm"
    macro_path.write_text(macro_text, encoding="utf-8")

    print("Work dir:", work)
    print("Input image:", img_path)
    print("Macro:", macro_path)

    proc = run_fiji_macro(exe, str(macro_path), str(img_path), str(out_dir), headless=True, log_name="run_log_retry.txt")

    try:
        rc = proc.wait(timeout=180)
    finally:
        best_effort_close_process_log(proc)

    print("Exit code:", rc)

    log_path = out_dir / "run_log_retry.txt"
    if log_path.exists():
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-120:]
        print("--- log tail ---")
        print("\n".join(tail))

    results = out_dir / "results.csv"
    print("results.csv exists:", results.exists(), "size:", (results.stat().st_size if results.exists() else None))

    overlay = out_dir / "overlay.png"
    print("overlay.png exists:", overlay.exists(), "size:", (overlay.stat().st_size if overlay.exists() else None))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
