from __future__ import annotations

import os
import subprocess
import sys
import ctypes
from pathlib import Path
from typing import Dict, Optional, TypedDict


_SUPPORTS_DASH_MACRO_CACHE: Dict[str, bool] = {}


class ImageJEngineInfo(TypedDict):
    engine_type: str
    help_text: str
    supports_headless: bool
    supports_ij2_flag: bool
    supports_dash_macro: bool
    supports_batch: bool


_ENGINE_INFO_CACHE: Dict[str, ImageJEngineInfo] = {}


def _get_imagej_help_text(exe_path: str) -> str:
    """Best-effort: collect help text from different ImageJ/Fiji variants.

    Some classic ImageJ builds use `-help`/`-h` and may print nothing for `--help`.
    """
    exe = str(exe_path or "").strip().strip('"')
    if not exe:
        return ""

    def run_help(args: list[str]) -> str:
        try:
            proc = subprocess.run(args, capture_output=True, text=True, timeout=5, shell=False)
            return ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        except Exception:
            return ""

    # Try a couple of common conventions.
    for flag in ("--help", "-help", "-h"):
        out = run_help([exe, flag])
        if out:
            return out

    return ""


def _supports_dash_macro(exe_path: str) -> bool:
    """Return True if the given ImageJ/Fiji executable supports the `-macro` CLI flag.

    Some distributions (notably the ImageJ2 launcher `ImageJ.exe`) support `--headless`
    but do *not* support Fiji's `-macro` flag; in that case presets would appear to
    "run" yet produce no outputs.
    """
    key = str(exe_path or "").strip().strip('"')
    if not key:
        return False
    cached = _SUPPORTS_DASH_MACRO_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        proc = subprocess.run(
            [key, "--help"],
            capture_output=True,
            text=True,
            timeout=5,
            shell=False,
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        ok = "-macro" in out
    except Exception:
        ok = False

    _SUPPORTS_DASH_MACRO_CACHE[key] = ok
    return ok


def supports_fiji_dash_macro(exe_path: str) -> bool:
    """Public wrapper: whether the executable supports Fiji-style `-macro`."""
    return _supports_dash_macro(exe_path)


def detect_imagej_engine(exe_path: str) -> ImageJEngineInfo:
    """Detect ImageJ/Fiji CLI capabilities by parsing `<exe> --help`.

    Spec (user request):
      - If help contains `--ij2` or `--headless` => engine_type="ij2_launcher"
      - Else => engine_type="ij1_classic"

    We also expose capability flags used to choose the best macro runner.
    """
    key = str(exe_path or "").strip().strip('"')
    if not key:
        return {
            "engine_type": "unknown",
            "help_text": "",
            "supports_headless": False,
            "supports_ij2_flag": False,
            "supports_dash_macro": False,
            "supports_batch": False,
        }

    cached = _ENGINE_INFO_CACHE.get(key)
    if cached is not None:
        return cached

    help_text = _get_imagej_help_text(key)

    supports_headless = "--headless" in help_text
    supports_ij2_flag = "--ij2" in help_text
    # Note: classic ImageJ typically supports `-batch` and/or `-macro`.
    supports_dash_macro = "-macro" in help_text
    supports_batch = "-batch" in help_text

    # Spec: if help contains --ij2 or --headless => ij2_launcher, else ij1_classic
    engine_type = "ij2_launcher" if (supports_ij2_flag or supports_headless) else "ij1_classic"

    out: ImageJEngineInfo = {
        "engine_type": engine_type,
        "help_text": help_text,
        "supports_headless": supports_headless,
        "supports_ij2_flag": supports_ij2_flag,
        "supports_dash_macro": supports_dash_macro,
        "supports_batch": supports_batch,
    }

    _ENGINE_INFO_CACHE[key] = out
    return out


def _try_get_short_path(path: str) -> Optional[str]:
    """Best-effort 8.3 short path on Windows.

    Helps some external tools that struggle with Unicode/long paths.
    Returns None if unavailable.
    """
    if os.name != "nt":
        return None
    try:
        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW  # type: ignore[attr-defined]
    except Exception:
        return None

    try:
        in_path = str(path)
        # First call to get required buffer size
        required = int(GetShortPathNameW(in_path, None, 0))
        if required <= 0:
            return None
        buf = ctypes.create_unicode_buffer(required + 1)
        out_len = int(GetShortPathNameW(in_path, buf, required + 1))
        if out_len <= 0:
            return None
        out = str(buf.value)
        return out if out else None
    except Exception:
        return None


def normalize_path_for_imagej(path: str) -> str:
    """Normalize a path string for passing to ImageJ/Fiji as a CLI argument."""
    s = str(path or "").strip().strip('"')
    try:
        p = Path(s).expanduser().resolve(strict=False)
    except Exception:
        p = Path(s)

    # Some ImageJ/Fiji builds behave better with forward slashes.
    normalized = str(p)
    if os.name == "nt":
        normalized = normalized.replace("\\", "/")

        # If path contains non-ASCII characters, try using short path.
        try:
            normalized.encode("ascii")
        except Exception:
            short = _try_get_short_path(str(p))
            if short:
                normalized = str(short).replace("\\", "/")

    return normalized


def run_fiji_open(exe_path: str, image_path: str) -> subprocess.Popen:
    """Launch Fiji/ImageJ and open an image/file (non-blocking)."""
    img = normalize_path_for_imagej(image_path)
    return subprocess.Popen([str(exe_path), str(img)], shell=False)


def run_fiji_macro(
    exe_path: str,
    macro_path: str,
    image_path: str,
    out_dir: str,
    extra_args: str = "",
    *,
    headless: bool = True,
    log_name: str = "run.log",
) -> subprocess.Popen:
    """Run an ImageJ macro for a single input.

    Macro args are passed as a single string.
    Convention: "input=<path> output=<dir>".

    Writes stdout/stderr to <out_dir>/<log_name>.

    Returns a Popen so the caller can wait/terminate.
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    log_path = outp / str(log_name)

    img = normalize_path_for_imagej(image_path)
    out_norm = normalize_path_for_imagej(out_dir)
    macro_norm = normalize_path_for_imagej(macro_path)
    # Quote values so paths with spaces survive ImageJ macro arg parsing.
    macro_args = f"input=\"{img}\" output=\"{out_norm}\""
    if extra_args and str(extra_args).strip():
        macro_args = macro_args + " " + str(extra_args).strip()

    engine = detect_imagej_engine(str(exe_path))

    cmd = [str(exe_path)]
    # Classic ImageJ (IJ1) path: use -batch (spec requirement).
    if engine.get("engine_type") == "ij1_classic":
        if engine.get("supports_batch"):
            cmd.extend(["-batch", str(macro_norm), str(macro_args)])
        elif engine.get("supports_dash_macro"):
            # Best-effort fallback for unusual IJ1 builds.
            cmd.extend(["-macro", str(macro_norm), str(macro_args)])
        else:
            # Many classic ImageJ builds do not provide rich help output.
            # Try -batch anyway; if it truly doesn't work, the subprocess output will go to run_log.txt.
            cmd.extend(["-batch", str(macro_norm), str(macro_args)])
    else:
        # Fiji/ImageJ2 launcher path: ONLY use -macro when supported.
        # Do not use '--run' for IJM macros; it often triggers SciJava parsing errors.
        if headless and engine.get("supports_headless"):
            cmd.append("--headless")
        if engine.get("supports_ij2_flag"):
            # Keep consistent with spec; does not prevent IJ1 macro execution if the launcher supports it.
            cmd.append("--ij2")
        if engine.get("supports_dash_macro"):
            cmd.extend(["-macro", str(macro_norm), str(macro_args)])
        elif engine.get("supports_batch"):
            cmd.extend(["-batch", str(macro_norm), str(macro_args)])
        else:
            raise RuntimeError(
                "This ImageJ/Fiji executable looks like an ImageJ2 launcher (it has '--headless'/'--ij2') but does not support '-macro' or '-batch'.\n\n"
                "Fix: in Fiji.app, select 'ImageJ-win64.exe' (or classic ImageJ that supports '-batch')."
            )

    # Append a short header so logs for re-runs are readable.
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n" + ("=" * 72) + "\n")
            f.write(f"CMD: {' '.join(cmd)}\n")
            f.write(
                f"ENGINE: {engine.get('engine_type')} batch={engine.get('supports_batch')} macro={engine.get('supports_dash_macro')} headless={engine.get('supports_headless')} ij2={engine.get('supports_ij2_flag')}\n"
            )
    except Exception:
        pass

    log_fh = open(log_path, "a", encoding="utf-8", errors="replace")
    try:
        # Run with cwd set to the executable directory; helps plugin/jar resolution.
        cwd = None
        try:
            cwd = str(Path(str(exe_path)).parent)
        except Exception:
            cwd = None

        p = subprocess.Popen(cmd, shell=False, cwd=cwd, stdout=log_fh, stderr=log_fh)
    except Exception:
        try:
            log_fh.close()
        except Exception:
            pass
        raise

    # Attach for callers who want to close it later.
    setattr(p, "_log_fh", log_fh)
    return p


def best_effort_close_process_log(proc: subprocess.Popen) -> None:
    """Close the log file handle attached by run_fiji_macro (if present)."""
    fh: Optional[object] = getattr(proc, "_log_fh", None)
    if fh is None:
        return
    try:
        fh.close()  # type: ignore[attr-defined]
    except Exception:
        pass
