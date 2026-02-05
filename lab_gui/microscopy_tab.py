from __future__ import annotations

import datetime
import os
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import ttkbootstrap as tb
import tkinter.ttk as ttk_native

ttk = tb
ttk.LabelFrame = ttk_native.LabelFrame
if not hasattr(ttk, "PanedWindow") and hasattr(ttk, "Panedwindow"):
    ttk.PanedWindow = ttk.Panedwindow  # type: ignore[attr-defined]

import pandas as pd

from PIL import Image, ImageTk

from lab_gui.external_tools import best_effort_close_process_log, detect_imagej_engine, run_fiji_macro, run_fiji_open
from lab_gui.microscopy_model import MicroscopyDataset, MicroscopyWorkspace
from lab_gui.settings import guess_imagej_initial_dirs, load_settings, save_settings, validate_imagej_exe_path
from lab_gui.ui_widgets import ToolTip


SUPPORTED_INPUT_EXTS = {
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".czi",
}

SUPPORTED_OUTPUT_EXTS = {
    ".csv",
    ".xlsx",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".txt",
    ".pdf",
}


PRESETS: List[Dict[str, Any]] = [
    {
        "id": "particle_size",
        "name": "Particle size (Analyze Particles)",
        "what": "Thresholds the image, converts to mask, runs Analyze Particles, and saves a CSV table.",
        "mode": "particles",
    },
    {
        "id": "droplet_count",
        "name": "Droplet count",
        "what": "Thresholds and counts droplet-like particles via Analyze Particles (Count + sizes).",
        "mode": "particles",
    },
    {
        "id": "area_fraction",
        "name": "Area/Coverage fraction",
        "what": "Thresholds and computes area fraction (foreground fraction) and writes a one-row CSV.",
        "mode": "area_fraction",
    },
    {
        "id": "quick_qc",
        "name": "Quick QC (saturation + focus proxy)",
        "what": "Computes saturation fraction and a simple edge-based focus proxy; writes a one-row CSV.",
        "mode": "qc",
    },
]


class _BatchRunDialog(tk.Toplevel):
    def __init__(self, parent: tk.Widget, *, title: str, total: int) -> None:
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self._cancel_event = threading.Event()

        body = ttk.Frame(self, padding=10)
        body.grid(row=0, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)

        self._label_var = tk.StringVar(value="Starting…")
        ttk.Label(body, textvariable=self._label_var, wraplength=560, justify="left").grid(row=0, column=0, sticky="w")

        self._pb = ttk.Progressbar(body, mode="determinate", maximum=max(1, int(total)))
        self._pb.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        btns = ttk.Frame(body)
        btns.grid(row=2, column=0, sticky="e", pady=(10, 0))
        cancel = ttk.Button(btns, text="Cancel", command=self._on_cancel)
        cancel.grid(row=0, column=0)

        ToolTip.attach(cancel, "Cancel the batch run (best-effort).")

        try:
            self.transient(parent.winfo_toplevel())
            self.grab_set()
        except Exception:
            pass

        try:
            self.update_idletasks()
            x = int(self.winfo_screenwidth() / 2 - self.winfo_reqwidth() / 2)
            y = int(self.winfo_screenheight() / 2 - self.winfo_reqheight() / 2)
            self.geometry(f"+{x}+{y}")
        except Exception:
            pass

    def _on_cancel(self) -> None:
        self._cancel_event.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def set_status(self, text: str, *, step: Optional[int] = None) -> None:
        try:
            self._label_var.set(str(text))
        except Exception:
            pass
        if step is not None:
            try:
                self._pb["value"] = int(step)
            except Exception:
                pass


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_stem(path_str: str) -> str:
    try:
        p = Path(path_str)
        stem = p.stem
    except Exception:
        stem = "dataset"
    stem = "".join(ch for ch in stem if ch.isalnum() or ch in ("-", "_", " ")).strip()
    return stem or "dataset"


def _safe_dirname(name: str) -> str:
    s = "".join(ch for ch in str(name) if ch.isalnum() or ch in ("-", "_", " ")).strip()
    s = s.replace(" ", "_")
    return s or "workspace"


def discover_outputs(out_dir: str) -> List[Path]:
    return discover_outputs_limited(out_dir)


def discover_outputs_limited(
    out_dir: str,
    *,
    recursive: bool = True,
    max_files: int = 5000,
    max_seconds: float = 2.0,
) -> List[Path]:
    """Discover output files under a dataset output directory.

    This function is UI-safety oriented: it hard-limits time and file count so a very large output
    folder (or a slow filesystem) cannot freeze the Tkinter event loop.
    """
    p = Path(out_dir)
    if not p.exists() or not p.is_dir():
        return []

    start = time.time()
    found: List[Path] = []

    it = p.rglob("*") if recursive else p.glob("*")
    try:
        for f in it:
            if max_seconds > 0 and (time.time() - start) > float(max_seconds):
                break
            if max_files > 0 and len(found) >= int(max_files):
                break
            try:
                if not f.is_file():
                    continue
                if f.suffix.lower() in SUPPORTED_OUTPUT_EXTS:
                    found.append(f)
            except Exception:
                continue
    except Exception:
        return []

    found.sort(key=lambda x: x.name.lower())
    return found


def _tail_text(path: Path, n_lines: int = 40) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        try:
            text = path.read_text(errors="replace")
        except Exception:
            return ""
    lines = text.splitlines()
    if len(lines) <= n_lines:
        return "\n".join(lines)
    return "\n".join(lines[-n_lines:])


class MicroscopyView(ttk.Frame):
    def __init__(self, parent: tk.Widget, app: Any, workspace: Any) -> None:
        super().__init__(parent)
        self.app = app
        self.workspace = workspace

        # Ensure we always have at least one workspace.
        self._ensure_default_workspace()

        # UI state
        self._active_dataset_id: Optional[str] = None
        self._active_output_path: Optional[str] = None

        # Guard to prevent <<TreeviewSelect>> recursion during refreshes.
        self._ds_ignore_select: bool = False

        # Cache outputs count to avoid rescanning every dataset on each UI refresh.
        self._outputs_count_cache: Dict[str, int] = {}

        # Preset runner threading state
        self._preset_thread: Optional[threading.Thread] = None
        self._preset_current_proc: Optional[subprocess.Popen] = None
        self._overlay_photo = None
        self._last_run_log_path: Optional[str] = None

        # Per-dataset run history (in-memory; derived from produced folders)
        self._runs_history: Dict[str, List[Dict[str, Any]]] = {}

        self._build_ui()
        self.refresh_from_workspace(select_first=True)

    # -------------------------- public hooks --------------------------

    def status_text(self) -> str:
        selected = self._selected_dataset_ids()
        nsel = len(selected)

        settings = load_settings()
        exe = validate_imagej_exe_path(settings.get("fiji_exe_path"))
        if not exe:
            return f"Microscopy: {nsel} datasets selected | ImageJ engine MISSING"

        info = detect_imagej_engine(exe)
        caps = []
        if info.get("supports_batch"):
            caps.append("-batch")
        if info.get("supports_dash_macro"):
            caps.append("-macro")
        cap_txt = "/".join(caps) if caps else "no-macro"
        return f"Microscopy: {nsel} datasets selected | ImageJ engine {info.get('engine_type')} ({cap_txt})"

    def refresh_from_workspace(self, *, select_first: bool = False) -> None:
        self._refresh_dataset_tree(preserve_selection=True)
        if select_first:
            try:
                items = list(self._ds_tree.get_children(""))
                if items:
                    self._ds_tree.selection_set(items[0])
                    self._ds_tree.see(items[0])
            except Exception:
                pass
            try:
                self._refresh_workspace_selector()
            except Exception:
                pass
        self._on_dataset_select(None)

    # -------------------------- debug helpers --------------------------

    def _test_macro(self) -> None:
        exe = self._ensure_imagej_path()
        if not exe:
            return

        info = detect_imagej_engine(exe)

        # Pick an output directory that is always writable and easy to find.
        try:
            root = Path(self.app._get_session_root_dir())
        except Exception:
            root = Path.cwd()

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = root / "microscopy" / "_macro_test" / ts
        out_dir.mkdir(parents=True, exist_ok=True)

        macro_path = out_dir / "test_macro.ijm"

        # A tiny macro script that writes a file into output and logs a couple of lines.
        # It accepts output=<dir> but also works if output is missing.
        macro_text = """
function getKV(args, key) {
    k = key + "=";
    i = indexOf(args, k);
    if (i < 0) return "";
    i = i + lengthOf(k);
    if (substring(args, i, i+1) == "\"") {
        j = indexOf(args, "\"", i+1);
        if (j < 0) return substring(args, i+1);
        return substring(args, i+1, j);
    }
    j = indexOf(args, " ", i);
    if (j < 0) j = lengthOf(args);
    return substring(args, i, j);
}

args = getArgument();
out = getKV(args, "output");
if (out == "") out = getDirectory("temp") + "mfp_macro_test";

function logLine(s) {
    File.append(s+"\n", out + "/run_log.txt");
    print(s);
}

File.makeDirectory(out);
logLine("TestMacro: started");
logLine("out=" + out);
File.saveString("ok\n", out + "/test_ok.txt");
logLine("TestMacro: wrote test_ok.txt");
logLine("TestMacro: done");
""".lstrip("\n")

        try:
            macro_path.write_text(macro_text, encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("Test Macro", f"Failed to write macro file:\n{exc}", parent=self.winfo_toplevel())
            return

        # Run the macro. We pass a dummy input as well to keep the argument style consistent.
        dummy_input = str(macro_path)  # not used by the test macro
        try:
            proc = run_fiji_macro(
                exe,
                str(macro_path),
                dummy_input,
                str(out_dir),
                headless=True,
                log_name="run_log.txt",
            )
        except Exception as exc:
            messagebox.showerror("Test Macro", f"Failed to launch macro:\n{exc}", parent=self.winfo_toplevel())
            return

        try:
            rc = proc.wait(timeout=60)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
            rc = None
        finally:
            best_effort_close_process_log(proc)

        ok_path = out_dir / "test_ok.txt"
        log_path = out_dir / "run_log.txt"
        tail = _tail_text(log_path, n_lines=50) if log_path.exists() else "(no log)"

        if ok_path.exists() and (rc in (0, None)):
            messagebox.showinfo(
                "Test Macro",
                f"Success.\n\nEngine: {info.get('engine_type')}\nOutput: {out_dir}\n\nLast log lines:\n{tail}",
                parent=self.winfo_toplevel(),
            )
        else:
            messagebox.showerror(
                "Test Macro",
                f"Failed.\n\nEngine: {info.get('engine_type')}\nExit code: {rc}\nOutput: {out_dir}\n\nLast log lines:\n{tail}",
                parent=self.winfo_toplevel(),
            )

    # -------------------------- settings helpers --------------------------

    def _ensure_imagej_path(self) -> Optional[str]:
        settings = load_settings()
        exe = validate_imagej_exe_path(settings.get("fiji_exe_path"))
        if exe:
            return exe

        if not messagebox.askyesno(
            "ImageJ/Fiji path",
            "ImageJ/Fiji executable path is not set (or is invalid).\n\nSet it now?",
            parent=self.winfo_toplevel(),
        ):
            return None

        return self._set_imagej_path_flow()

    def _set_imagej_path_flow(self) -> Optional[str]:
        settings = load_settings()

        initialdirs = guess_imagej_initial_dirs()
        initialdir = initialdirs[0] if initialdirs else "C:\\"
        if settings.get("fiji_exe_path"):
            try:
                initialdir = str(Path(str(settings.get("fiji_exe_path"))).parent)
            except Exception:
                pass

        path = filedialog.askopenfilename(
            parent=self.winfo_toplevel(),
            title="Select ImageJ/Fiji executable",
            filetypes=[("Executable", "*.exe"), ("All files", "*.*")],
            initialdir=initialdir,
        )
        if not path:
            return None

        exe = validate_imagej_exe_path(path)
        if not exe:
            messagebox.showerror("Invalid path", "Please select a valid .exe file for ImageJ or Fiji.", parent=self.winfo_toplevel())
            return None

        # If the user picked the Fiji launcher, prefer the ImageJ launcher inside the same folder.
        # `fiji-windows-x64.exe` is a GUI launcher and is not reliable for macro execution from our app.
        try:
            p = Path(str(exe))
            if p.name.lower().startswith("fiji") and p.name.lower().endswith(".exe"):
                candidate = p.parent / "ImageJ-win64.exe"
                if candidate.exists() and candidate.is_file():
                    if messagebox.askyesno(
                        "Fiji executable selected",
                        "You selected Fiji's launcher executable. For macro execution, it's more reliable to use:\n\n"
                        f"{candidate}\n\n"
                        "Use this instead?",
                        parent=self.winfo_toplevel(),
                    ):
                        exe2 = validate_imagej_exe_path(str(candidate))
                        if exe2:
                            exe = exe2
        except Exception:
            pass

        info = detect_imagej_engine(exe)
        if not (info.get("supports_batch") or info.get("supports_dash_macro")):
            messagebox.showwarning(
                "ImageJ executable selected",
                "This ImageJ executable does not appear to support macro execution (no -batch or -macro).\n\n"
                "Preset Analysis will not work with this selection. Please choose a different ImageJ/Fiji executable.",
                parent=self.winfo_toplevel(),
            )

        settings["fiji_exe_path"] = exe
        save_settings(settings)

        try:
            self.app._set_status("Fiji path saved")
        except Exception:
            pass
        self._maybe_update_status_bar()
        try:
            self._update_engine_note()
        except Exception:
            pass
        return exe

    # -------------------------- workspace/dataset model --------------------------

    def _get_workspaces(self) -> List[MicroscopyWorkspace]:
        wss = getattr(self.workspace, "microscopy_workspaces", None)
        if isinstance(wss, list):
            return wss
        # Migration/default
        try:
            self.workspace.microscopy_workspaces = []
        except Exception:
            pass
        return []

    def _set_active_workspace_id(self, ws_id: Optional[str]) -> None:
        try:
            self.workspace.active_microscopy_workspace_id = ws_id
        except Exception:
            pass

    def _active_workspace(self) -> Optional[MicroscopyWorkspace]:
        active_id = getattr(self.workspace, "active_microscopy_workspace_id", None)
        wss = self._get_workspaces()
        if active_id:
            for ws in wss:
                if str(ws.id) == str(active_id):
                    return ws
        return wss[0] if wss else None

    def _ensure_default_workspace(self) -> MicroscopyWorkspace:
        wss = self._get_workspaces()
        if wss:
            ws = self._active_workspace() or wss[0]
            self._set_active_workspace_id(ws.id)
            return ws

        ws = MicroscopyWorkspace(id=str(uuid.uuid4()), name="Default")
        try:
            self.workspace.microscopy_workspaces = [ws]
        except Exception:
            pass
        self._set_active_workspace_id(ws.id)
        return ws

    def _refresh_workspace_selector(self) -> None:
        wss = self._get_workspaces()
        names = [str(ws.name) for ws in wss]
        try:
            self._ws_combo["values"] = names
        except Exception:
            pass

        active = self._active_workspace()
        if active is not None:
            try:
                self._ws_combo_var.set(str(active.name))
            except Exception:
                pass
        elif names:
            try:
                self._ws_combo_var.set(str(names[0]))
            except Exception:
                pass

    def _on_workspace_selected(self, _evt=None) -> None:
        name = str(self._ws_combo_var.get() or "").strip()
        if not name:
            return
        for ws in self._get_workspaces():
            if str(ws.name) == name:
                self._set_active_workspace_id(ws.id)
                break
        self._maybe_update_status_bar()

    def _iter_datasets(self) -> List[MicroscopyDataset]:
        out: List[MicroscopyDataset] = []
        for ws in self._get_workspaces():
            out.extend(list(ws.datasets or []))
        return out

    def _find_dataset(self, dataset_id: str) -> Optional[MicroscopyDataset]:
        did = str(dataset_id)
        for ds in self._iter_datasets():
            if str(ds.id) == did:
                return ds
        return None

    def _find_workspace(self, ws_id: str) -> Optional[MicroscopyWorkspace]:
        wid = str(ws_id)
        for ws in self._get_workspaces():
            if str(ws.id) == wid:
                return ws
        return None

    def _dataset_output_dir(self, ws: MicroscopyWorkspace, ds: MicroscopyDataset) -> str:
        # Use the app's session root if available.
        try:
            root = Path(self.app._get_session_root_dir())
        except Exception:
            root = Path.cwd()
        ws_part = _safe_dirname(ws.name) or _safe_dirname(ws.id)
        ds_part = _safe_dirname(_safe_stem(ds.file_path))
        return str(root / "microscopy" / ws_part / ds_part)

    # -------------------------- UI --------------------------

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self, padding=(6, 6, 6, 6))
        toolbar.grid(row=0, column=0, sticky="ew")

        btn_add_ws = ttk.Button(toolbar, text="Add Workspace…", command=self._add_workspace)

        ttk.Label(toolbar, text="Active workspace:").grid(row=0, column=1, padx=(0, 6))
        self._ws_combo_var = tk.StringVar(value="")
        self._ws_combo = ttk.Combobox(toolbar, textvariable=self._ws_combo_var, state="readonly", width=22)
        self._ws_combo.grid(row=0, column=2, padx=(0, 12))
        self._ws_combo.bind("<<ComboboxSelected>>", self._on_workspace_selected)

        btn_load = ttk.Button(toolbar, text="Load Files…", command=self._load_files)
        btn_remove = ttk.Button(toolbar, text="Remove Selected", command=self._remove_selected)
        btn_set_path = ttk.Button(toolbar, text="Set Fiji/ImageJ Path…", command=self._set_imagej_path_flow)
        btn_open = ttk.Button(toolbar, text="Open Selected in Fiji/ImageJ", command=self._open_selected_in_imagej)
        btn_open_out = ttk.Button(toolbar, text="Open Output Folder", command=self._open_output_folder)
        btn_test_macro = ttk.Button(toolbar, text="Test Macro", command=self._test_macro)

        btn_add_ws.grid(row=0, column=0, padx=(0, 10))
        btn_load.grid(row=0, column=3, padx=(0, 8))
        btn_remove.grid(row=0, column=4, padx=(0, 8))
        btn_set_path.grid(row=0, column=5, padx=(0, 8))
        btn_open.grid(row=0, column=6, padx=(0, 8))
        btn_open_out.grid(row=0, column=7, padx=(0, 0))
        btn_test_macro.grid(row=0, column=8, padx=(8, 0))

        ToolTip.attach(btn_add_ws, "Create a new Microscopy workspace.")
        ToolTip.attach(self._ws_combo, "Choose the active workspace.\n\nNew files loaded via 'Load Files…' go into the active workspace.")
        ToolTip.attach(btn_load, "Add microscopy image files into the active workspace.\n\nThis does NOT parse the images. Files are only tracked by path and can be opened in ImageJ/Fiji.")
        ToolTip.attach(btn_remove, "Remove the selected dataset(s) from this app session.\n\nThis does NOT delete the original input files. It also does not delete existing output files unless you remove them manually.")
        ToolTip.attach(btn_set_path, "Choose the ImageJ/Fiji executable (.exe) and save it persistently.\n\nYou only need to do this once; use it again later if you want to switch to a different ImageJ/Fiji installation.")
        ToolTip.attach(btn_open, "Open the selected dataset(s) in Fiji/ImageJ.\n\nEach selected file is launched as a separate open request.")
        ToolTip.attach(btn_open_out, "Open the output folder for the first selected dataset (or the Microscopy root folder if nothing is selected).")
        ToolTip.attach(btn_test_macro, "Run a tiny diagnostic macro and show the log tail.\n\nUse this to debug ImageJ/Fiji path and macro execution.")

        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=2)
        main.add(right, weight=3)

        # Left: dataset list
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        ttk.Label(left, text="Datasets", padding=(0, 0, 0, 6)).grid(row=0, column=0, sticky="w")

        cols = ("workspace", "dataset_name", "file_path", "notes", "outputs_count")
        tree = ttk.Treeview(left, columns=cols, show="headings", selectmode="extended")
        tree.grid(row=1, column=0, sticky="nsew")

        vsb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=tree.yview)
        vsb.grid(row=1, column=1, sticky="ns")
        tree.configure(yscrollcommand=vsb.set)

        tree.heading("workspace", text="Workspace")
        tree.heading("dataset_name", text="Dataset")
        tree.heading("file_path", text="Path")
        tree.heading("notes", text="Notes")
        tree.heading("outputs_count", text="Outputs")

        tree.column("workspace", width=120, stretch=False)
        tree.column("dataset_name", width=160, stretch=False)
        tree.column("file_path", width=380, stretch=True)
        tree.column("notes", width=220, stretch=True)
        tree.column("outputs_count", width=70, stretch=False, anchor="center")

        tree.bind("<<TreeviewSelect>>", self._on_dataset_select)
        self._ds_tree = tree

        # Right: details
        right.columnconfigure(0, weight=1)
        right.rowconfigure(4, weight=1)

        info = ttk.LabelFrame(right, text="Selected dataset", padding=8)
        info.grid(row=0, column=0, sticky="ew")
        info.columnconfigure(1, weight=1)
        info.columnconfigure(2, weight=0)

        self._info_path_var = tk.StringVar(value="")
        self._info_size_var = tk.StringVar(value="")
        self._info_mtime_var = tk.StringVar(value="")
        self._info_outdir_var = tk.StringVar(value="")

        ttk.Label(info, text="Path:").grid(row=0, column=0, sticky="w")
        ttk.Label(info, textvariable=self._info_path_var, wraplength=620, justify="left").grid(row=0, column=1, sticky="w")
        ttk.Label(info, text="File size:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(info, textvariable=self._info_size_var).grid(row=1, column=1, sticky="w", pady=(4, 0))
        ttk.Label(info, text="Modified:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Label(info, textvariable=self._info_mtime_var).grid(row=2, column=1, sticky="w", pady=(4, 0))

        ttk.Label(info, text="Output folder:").grid(row=3, column=0, sticky="w", pady=(4, 0))
        ttk.Label(info, textvariable=self._info_outdir_var, wraplength=620, justify="left").grid(row=3, column=1, sticky="w", pady=(4, 0))
        btn_set_out = ttk.Button(info, text="Set…", command=self._set_output_folder_for_selected)
        btn_set_out.grid(row=3, column=2, sticky="e", padx=(10, 0), pady=(4, 0))
        ToolTip.attach(btn_set_out, "Choose a custom output folder for the selected dataset.\n\nOutputs are discovered from that folder.")

        notesf = ttk.LabelFrame(right, text="Notes", padding=8)
        notesf.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        notesf.columnconfigure(0, weight=1)

        notes = tk.Text(notesf, height=4, wrap="word")
        notes.grid(row=0, column=0, sticky="ew")
        notes.bind("<FocusOut>", self._on_notes_focus_out)
        self._notes_text = notes

        ToolTip.attach(notes, "Notes are stored per dataset inside the workspace JSON session file.")

        # Preset analysis panel
        presetf = ttk.LabelFrame(right, text="Preset Analysis", padding=8)
        presetf.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        presetf.columnconfigure(0, weight=1)
        self._build_preset_panel(presetf)

        outsf = ttk.LabelFrame(right, text="Outputs", padding=8)
        outsf.grid(row=4, column=0, sticky="nsew", pady=(8, 0))
        outsf.columnconfigure(0, weight=1)
        outsf.rowconfigure(1, weight=1)

        out_btns = ttk.Frame(outsf)
        out_btns.grid(row=0, column=0, sticky="ew")

        btn_refresh = ttk.Button(out_btns, text="Refresh Outputs", command=self._refresh_outputs)
        btn_open_out = ttk.Button(out_btns, text="Open Selected Output", command=self._open_selected_output)
        btn_import = ttk.Button(out_btns, text="Import Outputs…", command=self._import_outputs)
        btn_export = ttk.Button(out_btns, text="Export Summary…", command=self._export_summary)

        btn_refresh.pack(side=tk.LEFT)
        btn_open_out.pack(side=tk.LEFT, padx=(8, 0))
        btn_import.pack(side=tk.LEFT, padx=(8, 0))
        btn_export.pack(side=tk.RIGHT)

        ToolTip.attach(btn_refresh, "Scan the selected dataset's output folder for files (CSV/PNG/JPG/TXT/PDF/...).\n\nThis updates the Outputs list and the outputs count shown in the dataset list.")
        ToolTip.attach(btn_open_out, "Open the selected output file with the system default application.")
        ToolTip.attach(btn_import, "Copy existing output files from a chosen folder into this dataset's output folder.\n\nUse this if you ran ImageJ/Fiji outside the app and want the outputs tracked here.")
        ToolTip.attach(btn_export, "Export a simple summary table (Excel) describing selected dataset(s) and their discovered outputs.")

        out_cols = ("name", "ext", "size", "modified")
        out_tree = ttk.Treeview(outsf, columns=out_cols, show="headings", selectmode="browse")
        out_tree.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        out_tree.heading("name", text="File")
        out_tree.heading("ext", text="Type")
        out_tree.heading("size", text="Size")
        out_tree.heading("modified", text="Modified")
        out_tree.column("name", width=360, stretch=True)
        out_tree.column("ext", width=70, stretch=False)
        out_tree.column("size", width=110, stretch=False, anchor="e")
        out_tree.column("modified", width=170, stretch=False)
        out_tree.bind("<<TreeviewSelect>>", self._on_output_select)

        out_vsb = ttk.Scrollbar(outsf, orient=tk.VERTICAL, command=out_tree.yview)
        out_vsb.grid(row=1, column=1, sticky="ns", pady=(8, 0))
        out_tree.configure(yscrollcommand=out_vsb.set)
        self._out_tree = out_tree

        self._refresh_workspace_selector()

    # -------------------------- UI actions --------------------------

    def _add_workspace(self) -> None:
        name = simpledialog.askstring("Add Workspace", "Workspace name:", parent=self.winfo_toplevel())
        if not name or not str(name).strip():
            return
        ws = MicroscopyWorkspace(id=str(uuid.uuid4()), name=str(name).strip())
        wss = self._get_workspaces()
        wss.append(ws)
        self._set_active_workspace_id(ws.id)
        self._refresh_workspace_selector()
        self._refresh_dataset_tree(preserve_selection=False)
        self._maybe_update_status_bar()

    # -------------------------- Preset analysis UI --------------------------

    def _build_preset_panel(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)

        top = ttk.Frame(parent)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        pan = ttk.PanedWindow(top, orient=tk.HORIZONTAL)
        pan.grid(row=0, column=0, columnspan=2, sticky="ew")

        left = ttk.Frame(pan)
        right = ttk.Frame(pan)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        pan.add(left, weight=1)
        pan.add(right, weight=3)

        ttk.Label(left, text="Presets", padding=(0, 0, 0, 6)).grid(row=0, column=0, sticky="w")
        self._preset_list = tk.Listbox(left, height=6, exportselection=False)
        self._preset_list.grid(row=1, column=0, sticky="nsew")
        vsb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self._preset_list.yview)
        vsb.grid(row=1, column=1, sticky="ns")
        self._preset_list.configure(yscrollcommand=vsb.set)
        for p in PRESETS:
            self._preset_list.insert(tk.END, p["name"])
        self._preset_list.bind("<<ListboxSelect>>", self._on_preset_select)

        # Right side: description + parameters + run controls + status + previews
        self._preset_desc_var = tk.StringVar(value="")
        ttk.Label(right, text="What it does:").grid(row=0, column=0, sticky="w")
        ttk.Label(right, textvariable=self._preset_desc_var, wraplength=640, justify="left").grid(row=1, column=0, sticky="ew", pady=(2, 8))

        params = ttk.Frame(right)
        params.grid(row=2, column=0, sticky="ew")
        for c in range(6):
            params.columnconfigure(c, weight=1 if c in (1, 3, 5) else 0)

        # Headless setting
        settings = load_settings()
        self._headless_var = tk.BooleanVar(value=bool(settings.get("microscopy_run_headless", True)))
        headless_cb = ttk.Checkbutton(params, text="Run Fiji headless", variable=self._headless_var, command=self._on_headless_toggle)
        headless_cb.grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 6))
        ToolTip.attach(headless_cb, "When ON: runs Fiji/ImageJ in headless mode (no UI).\nWhen OFF: launches normally so you can see the image.")

        # Engine/capabilities note
        self._engine_note_var = tk.StringVar(value="")
        engine_note = ttk.Label(params, textvariable=self._engine_note_var, wraplength=700, justify="left")
        engine_note.grid(row=0, column=3, columnspan=3, sticky="e", pady=(0, 6))

        # Threshold method
        self._thr_method_var = tk.StringVar(value="Otsu")
        ttk.Label(params, text="Threshold:").grid(row=1, column=0, sticky="w")
        thr = ttk.Combobox(params, textvariable=self._thr_method_var, state="readonly", values=["Otsu", "Yen", "Triangle", "Manual"], width=12)
        thr.grid(row=1, column=1, sticky="w", padx=(6, 10))
        thr.bind("<<ComboboxSelected>>", lambda _e=None: self._update_thr_manual_state())

        self._thr_min_var = tk.StringVar(value="50")
        self._thr_max_var = tk.StringVar(value="200")
        ttk.Label(params, text="Manual min:").grid(row=1, column=2, sticky="w")
        self._thr_min_entry = ttk.Entry(params, textvariable=self._thr_min_var, width=8)
        self._thr_min_entry.grid(row=1, column=3, sticky="w", padx=(6, 10))
        ttk.Label(params, text="Manual max:").grid(row=1, column=4, sticky="w")
        self._thr_max_entry = ttk.Entry(params, textvariable=self._thr_max_var, width=8)
        self._thr_max_entry.grid(row=1, column=5, sticky="w", padx=(6, 0))

        # Size + circularity
        self._min_size_var = tk.StringVar(value="10")
        self._max_size_var = tk.StringVar(value="Infinity")
        ttk.Label(params, text="Min size (px^2):").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(params, textvariable=self._min_size_var, width=10).grid(row=2, column=1, sticky="w", padx=(6, 10), pady=(6, 0))
        ttk.Label(params, text="Max size (px^2):").grid(row=2, column=2, sticky="w", pady=(6, 0))
        ttk.Entry(params, textvariable=self._max_size_var, width=10).grid(row=2, column=3, sticky="w", padx=(6, 10), pady=(6, 0))

        self._circ_min_var = tk.StringVar(value="0.00")
        self._circ_max_var = tk.StringVar(value="1.00")
        ttk.Label(params, text="Circ min:").grid(row=2, column=4, sticky="w", pady=(6, 0))
        ttk.Entry(params, textvariable=self._circ_min_var, width=8).grid(row=2, column=5, sticky="w", padx=(6, 0), pady=(6, 0))
        ttk.Label(params, text="Circ max:").grid(row=3, column=4, sticky="w", pady=(6, 0))
        ttk.Entry(params, textvariable=self._circ_max_var, width=8).grid(row=3, column=5, sticky="w", padx=(6, 0), pady=(6, 0))

        self._exclude_edge_var = tk.BooleanVar(value=True)
        self._overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params, text="Exclude edge particles", variable=self._exclude_edge_var).grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(params, text="Create overlay output", variable=self._overlay_var).grid(row=3, column=3, columnspan=2, sticky="w", pady=(6, 0))

        # Run buttons
        btns = ttk.Frame(right)
        btns.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        btn_run_active = ttk.Button(btns, text="Run on Active Image", command=self._run_preset_on_active)
        btn_run_sel = ttk.Button(btns, text="Run on Selected Images…", command=self._run_preset_on_selected)
        btn_run_all = ttk.Button(btns, text="Run on ALL in Workspace", command=self._run_preset_on_all_in_workspace)
        btn_run_active.grid(row=0, column=0, padx=(0, 8))
        btn_run_sel.grid(row=0, column=1, padx=(0, 8))
        btn_run_all.grid(row=0, column=2, padx=(0, 0))

        # Status/log
        ttk.Label(right, text="Last run log:").grid(row=4, column=0, sticky="w", pady=(10, 0))
        self._preset_log = tk.Text(right, height=5, wrap="word")
        self._preset_log.grid(row=5, column=0, sticky="ew")
        self._preset_log.configure(state="disabled")

        # Runs history + preview
        previews = ttk.Frame(right)
        previews.grid(row=6, column=0, sticky="ew", pady=(10, 0))
        previews.columnconfigure(0, weight=1)
        previews.columnconfigure(1, weight=2)

        histf = ttk.LabelFrame(previews, text="Runs history (active dataset)", padding=6)
        histf.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        histf.columnconfigure(0, weight=1)
        histf.rowconfigure(0, weight=1)
        self._runs_tree = ttk.Treeview(histf, columns=("ts", "preset"), show="headings", height=5, selectmode="browse")
        self._runs_tree.grid(row=0, column=0, sticky="nsew")
        self._runs_tree.heading("ts", text="Timestamp")
        self._runs_tree.heading("preset", text="Preset")
        self._runs_tree.column("ts", width=140, stretch=False)
        self._runs_tree.column("preset", width=160, stretch=True)
        self._runs_tree.bind("<<TreeviewSelect>>", self._on_run_history_select)

        prevf = ttk.LabelFrame(previews, text="Results preview", padding=6)
        prevf.grid(row=0, column=1, sticky="nsew")
        prevf.columnconfigure(0, weight=1)
        prevf.rowconfigure(1, weight=1)

        self._overlay_label = ttk.Label(prevf, text="(overlay preview)")
        self._overlay_label.grid(row=0, column=0, sticky="w")

        self._results_tree = ttk.Treeview(prevf, show="headings", height=6)
        self._results_tree.grid(row=1, column=0, sticky="nsew", pady=(6, 0))

        self._preset_list.selection_set(0)
        self._preset_list.event_generate("<<ListboxSelect>>")
        self._update_thr_manual_state()
        self._update_engine_note()

    def _update_engine_note(self) -> None:
        try:
            settings = load_settings()
            exe = validate_imagej_exe_path(settings.get("fiji_exe_path"))
            if not exe:
                self._engine_note_var.set("ImageJ: not set")
                return

            info = detect_imagej_engine(exe)
            caps = []
            if info.get("supports_batch"):
                caps.append("-batch")
            if info.get("supports_dash_macro"):
                caps.append("-macro")
            cap_txt = "/".join(caps) if caps else "no-macro"

            note = f"ImageJ: {info.get('engine_type')} ({cap_txt})"
            if info.get("engine_type") == "ij1_classic":
                note += " | Fiji-only methods disabled"
            self._engine_note_var.set(note)
        except Exception:
            # Keep UI resilient
            try:
                self._engine_note_var.set("")
            except Exception:
                pass

    def _on_headless_toggle(self) -> None:
        settings = load_settings()
        settings["microscopy_run_headless"] = bool(self._headless_var.get())
        try:
            save_settings(settings)
        except Exception:
            pass

    def _current_preset(self) -> Dict[str, Any]:
        try:
            idxs = self._preset_list.curselection()
            idx = int(idxs[0]) if idxs else 0
        except Exception:
            idx = 0
        idx = max(0, min(idx, len(PRESETS) - 1))
        return PRESETS[idx]

    def _on_preset_select(self, _evt=None) -> None:
        p = self._current_preset()
        self._preset_desc_var.set(str(p.get("what") or ""))
        self._update_thr_manual_state()

    def _update_thr_manual_state(self) -> None:
        is_manual = str(self._thr_method_var.get() or "").strip().lower() == "manual"
        st = "normal" if is_manual else "disabled"
        try:
            self._thr_min_entry.configure(state=st)
            self._thr_max_entry.configure(state=st)
        except Exception:
            pass

    # -------------------------- Preset execution --------------------------

    def _run_preset_on_active(self) -> None:
        did = self._active_dataset_id
        if not did:
            messagebox.showinfo("Preset Analysis", "Select an active dataset first.", parent=self.winfo_toplevel())
            return
        self._run_preset_on_dataset_ids([did])

    def _run_preset_on_selected(self) -> None:
        ids = self._selected_dataset_ids()
        if not ids:
            messagebox.showinfo("Preset Analysis", "Select one or more datasets first.", parent=self.winfo_toplevel())
            return
        self._run_preset_on_dataset_ids(ids)

    def _run_preset_on_all_in_workspace(self) -> None:
        ws = self._active_workspace()
        if ws is None or not (ws.datasets or []):
            messagebox.showinfo("Preset Analysis", "No datasets in the active workspace.", parent=self.winfo_toplevel())
            return
        self._run_preset_on_dataset_ids([str(d.id) for d in (ws.datasets or [])])

    def _gather_preset_params(self) -> Dict[str, Any]:
        # Keep values as strings; macro generation handles formatting.
        return {
            "threshold_method": str(self._thr_method_var.get() or "Otsu"),
            "manual_min": str(self._thr_min_var.get() or ""),
            "manual_max": str(self._thr_max_var.get() or ""),
            "min_size": str(self._min_size_var.get() or ""),
            "max_size": str(self._max_size_var.get() or ""),
            "circ_min": str(self._circ_min_var.get() or ""),
            "circ_max": str(self._circ_max_var.get() or ""),
            "exclude_edge": bool(self._exclude_edge_var.get()),
            "overlay": bool(self._overlay_var.get()),
        }

    def _run_preset_on_dataset_ids(self, dataset_ids: List[str]) -> None:
        if self._preset_thread is not None and self._preset_thread.is_alive():
            messagebox.showwarning("Preset Analysis", "A preset run is already in progress.", parent=self.winfo_toplevel())
            return

        exe = self._ensure_imagej_path()
        if not exe:
            return

        preset = self._current_preset()
        params = self._gather_preset_params()
        headless = bool(self._headless_var.get())

        dlg = _BatchRunDialog(self.winfo_toplevel(), title=f"Running preset: {preset['name']}", total=len(dataset_ids))

        def worker() -> None:
            failures: List[Tuple[str, str]] = []
            completed = 0
            for step, did in enumerate(dataset_ids, start=1):
                if dlg.cancelled:
                    break
                ds = self._find_dataset(did)
                if ds is None:
                    continue
                ws = self._find_workspace(str(getattr(ds, "workspace_id", ""))) or self._active_workspace()
                if ws is None:
                    continue

                in_path = Path(str(ds.file_path))
                if not in_path.exists():
                    failures.append((ds.display_name, "Input file not found"))
                    continue

                run_dir, ts = self._make_run_output_dir(ws, ds, preset["id"])
                try:
                    run_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass

                macro_path = run_dir / "preset.ijm"
                log_path = run_dir / "run_log.txt"
                results_path = run_dir / "results.csv"
                overlay_path = run_dir / "overlay.png"

                self._call_ui(dlg.set_status, f"{ds.display_name} ({step}/{len(dataset_ids)})", step=step - 1)

                # Write macro
                try:
                    macro_text = self._render_macro(preset_id=preset["id"], params=params)
                    macro_path.write_text(macro_text, encoding="utf-8")
                except Exception as exc:
                    failures.append((ds.display_name, f"Failed to write macro: {exc}"))
                    continue

                # Run Fiji/ImageJ
                proc = None
                try:
                    proc = run_fiji_macro(
                        exe,
                        str(macro_path),
                        str(in_path),
                        str(run_dir),
                        headless=headless,
                        log_name="run_log.txt",
                    )
                    self._preset_current_proc = proc
                    rc = proc.wait()
                    best_effort_close_process_log(proc)
                except Exception as exc:
                    rc = 999
                    failures.append((ds.display_name, f"Failed to launch Fiji/ImageJ: {exc}"))

                if dlg.cancelled:
                    try:
                        if proc is not None and proc.poll() is None:
                            proc.terminate()
                    except Exception:
                        pass
                    break

                ok = (rc == 0)
                if ok and (not results_path.exists()):
                    ok = False
                    self._last_run_log_path = str(log_path)
                    failures.append((ds.display_name, f"Run completed but results.csv was not created.\n\nLog: {log_path}"))

                if ok:
                    completed += 1
                else:
                    # log tail
                    tail = _tail_text(log_path, n_lines=60)
                    self._last_run_log_path = str(log_path)
                    failures.append((ds.display_name, f"Run failed (exit={rc}).\n\nLog: {log_path}\n\n{tail}"))

                # Update in-app history + previews (for single runs, show last run; for batch, show last processed)
                entry = {
                    "timestamp": ts,
                    "preset_id": preset["id"],
                    "preset_name": preset["name"],
                    "run_dir": str(run_dir),
                    "results_csv": (str(results_path) if results_path.exists() else ""),
                    "overlay_png": (str(overlay_path) if overlay_path.exists() else ""),
                    "run_log": str(log_path),
                    "status": ("ok" if ok else "failed"),
                }
                self._runs_history.setdefault(str(ds.id), []).append(entry)

                self._call_ui(self._refresh_run_history_for_active)
                self._call_ui(self._load_run_preview_from_entry, entry)
                self._call_ui(self._refresh_outputs)
                self._call_ui(self._refresh_dataset_tree)
                self._call_ui(self._maybe_update_status_bar)

            def finish() -> None:
                try:
                    dlg.set_status(f"Done. Completed {completed}/{len(dataset_ids)}.", step=len(dataset_ids))
                except Exception:
                    pass
                try:
                    dlg.destroy()
                except Exception:
                    pass

                if failures and not dlg.cancelled:
                    # Show first failure details; keep it clear and allow opening the log.
                    name, detail = failures[0]
                    self._show_run_error_popup(title="Preset Analysis failed", header=name, detail=detail)
                elif dlg.cancelled:
                    messagebox.showinfo("Preset Analysis", "Run cancelled (best-effort).", parent=self.winfo_toplevel())
                else:
                    messagebox.showinfo("Preset Analysis", f"Completed {completed} run(s).", parent=self.winfo_toplevel())

            self._call_ui(finish)
            self._preset_current_proc = None

        t = threading.Thread(target=worker, daemon=True)
        self._preset_thread = t
        t.start()

        def poll_cancel() -> None:
            if not bool(getattr(dlg, "winfo_exists", lambda: False)()):
                return
            if dlg.cancelled:
                try:
                    proc = self._preset_current_proc
                    if proc is not None and proc.poll() is None:
                        proc.terminate()
                except Exception:
                    pass
            try:
                dlg.after(250, poll_cancel)
            except Exception:
                pass

        poll_cancel()

    def _make_run_output_dir(self, ws: MicroscopyWorkspace, ds: MicroscopyDataset, preset_id: str) -> Tuple[Path, str]:
        # Base output folder is dataset.output_dir (user-configurable).
        base = Path(str(ds.output_dir or "")).expanduser() if getattr(ds, "output_dir", None) else Path.cwd()
        try:
            base.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        preset_part = _safe_dirname(str(preset_id))
        ws_part = _safe_dirname(str(getattr(ws, "name", "workspace")))
        ds_part = _safe_dirname(str(getattr(ds, "display_name", "dataset")))
        out_dir = base / "MicroscopyResults" / ws_part / ds_part / preset_part / ts
        return out_dir, ts

    def _render_macro(self, *, preset_id: str, params: Dict[str, Any]) -> str:
        # Macro receives arguments: input=<path> output=<dir>
        # We bake parameter values into the macro for robustness.
        thr_method = str(params.get("threshold_method") or "Otsu").strip()
        manual_min = str(params.get("manual_min") or "50").strip()
        manual_max = str(params.get("manual_max") or "200").strip()
        min_size = str(params.get("min_size") or "10").strip()
        max_size = str(params.get("max_size") or "Infinity").strip()
        circ_min = str(params.get("circ_min") or "0.00").strip()
        circ_max = str(params.get("circ_max") or "1.00").strip()
        exclude_edge = "true" if bool(params.get("exclude_edge", True)) else "false"
        overlay = "true" if bool(params.get("overlay", True)) else "false"

        # Escape for IJM strings
        def q(s: str) -> str:
            return s.replace("\\", "\\\\").replace('"', "\\\"")

        thr_method_q = q(thr_method)

        # Note: ImageJ/Fiji CLI `-macro <file> <args>` executes the file as a *script*.
        # If the file only contains a named macro definition (`macro "..." { ... }`) it will not run.
        # So we emit top-level script statements (with helper functions) rather than a named macro.
        common = f"""
    // PresetAnalysis (generated)

    // Robust args parsing: supports quoted values with spaces.
    function getKV(args, key) {{
        k = key + \"=\";
        i = indexOf(args, k);
        if (i < 0) return \"\";
        i = i + lengthOf(k);
        if (substring(args, i, i+1) == \"\\\"\") {{
            j = indexOf(args, \"\\\"\", i+1);
            if (j < 0) return substring(args, i+1);
            return substring(args, i+1, j);
        }}
        j = indexOf(args, \" \" , i);
        if (j < 0) j = lengthOf(args);
        return substring(args, i, j);
}}

args = getArgument();
input = getKV(args, \"input\");
out = getKV(args, \"output\");
if (input==\"\" || out==\"\") {{
    print(\"Missing input/output args\");
    exit(1);
}}

// Helper: log to file
function logLine(s) {{
    File.append(s+\"\\n\", out + \"/run_log.txt\");
    print(s);
}}

File.makeDirectory(out);
logLine(\"Input: \" + input);
logLine(\"Output: \" + out);

setBatchMode(true);
open(input);
title = getTitle();

// Convert to 8-bit if needed
run(\"8-bit\");

// Threshold
method = \"{thr_method_q}\";
if (toLowerCase(method) == \"manual\") {{
    setThreshold(parseFloat(\"{q(manual_min)}\"), parseFloat(\"{q(manual_max)}\"));
    setOption(\"BlackBackground\", true);
    run(\"Convert to Mask\");
}} else {{
    // Auto threshold (dark objects on bright background by default)
    setAutoThreshold(method + \" dark\");
    setOption(\"BlackBackground\", true);
    run(\"Convert to Mask\");
}}

// Ensure binary
run(\"Make Binary\");
"""

        if preset_id in ("particle_size", "droplet_count"):
            # Analyze particles
            return (
                common
                + f"""
    // Measurements
    run(\"Set Measurements...\", \"area mean min perimeter shape feret's redirect=None decimal=3\");

    // Analyze Particles
    edgeOpt = {exclude_edge};
    overlayOpt = {overlay};
    ap = \"size={q(min_size)}-{q(max_size)} circularity={q(circ_min)}-{q(circ_max)}\";
    if (edgeOpt) ap = ap + \" exclude\";
    if (overlayOpt) ap = ap + \" show=Overlay\"; else ap = ap + \" show=Nothing\";
    ap = ap + \" clear\";
    run(\"Analyze Particles...\", ap);

    // Save results
    saveAs(\"Results\", out + \"/results.csv\");
    if (overlayOpt) {{
        // Best-effort overlay snapshot
        selectWindow(title);
        saveAs(\"PNG\", out + \"/overlay.png\");
    }}

    close();
    run(\"Close All\");
    setBatchMode(false);
    logLine(\"Done\");
"""
            )

        if preset_id == "area_fraction":
            return (
                common
                + """
    // Area fraction measurement
    run("Set Measurements...", "area area_fraction redirect=None decimal=6");
    run("Measure");
    saveAs("Results", out + "/results.csv");

    close();
    run("Close All");
    setBatchMode(false);
    logLine("Done");
"""
            )

        if preset_id == "quick_qc":
            # For QC: compute saturation fraction and focus proxy (edge std dev)
            return (
                f"""
// PresetAnalysis (generated)

function getKV(args, key) {{
        k = key + \"=\";
        i = indexOf(args, k);
        if (i < 0) return \"\";
        i = i + lengthOf(k);
        if (substring(args, i, i+1) == \"\\\"\") {{
            j = indexOf(args, \"\\\"\", i+1);
            if (j < 0) return substring(args, i+1);
            return substring(args, i+1, j);
        }}
        j = indexOf(args, \" \" , i);
        if (j < 0) j = lengthOf(args);
        return substring(args, i, j);
}}

args = getArgument();
input = getKV(args, \"input\");
out = getKV(args, \"output\");
if (input==\"\" || out==\"\") {{
    print(\"Missing input/output args\");
    exit(1);
}}

function logLine(s) {{
    File.append(s+\"\\n\", out + \"/run_log.txt\");
    print(s);
}}

File.makeDirectory(out);
setBatchMode(true);
open(input);
run(\"8-bit\");

    // Saturation fraction (pixels at 255)
    getHistogram(values, counts, 256);
    total = 0;
    for (i=0; i<256; i++) total = total + counts[i];
    sat = counts[255];
    sat_frac = (total>0) ? (sat/total) : 0;

    // Focus proxy: edge image std-dev
    run(\"Duplicate...\", \"title=edges\");
    run(\"Find Edges\");
    getStatistics(area, mean, min, max, std);
    focus_proxy = std;
    close();

    // Write CSV manually
    File.saveString(\"saturation_fraction,focus_proxy\\n\" + sat_frac + \",\" + focus_proxy + \"\\n\", out + \"/results.csv\");
    logLine(\"saturation_fraction=\" + sat_frac);
    logLine(\"focus_proxy=\" + focus_proxy);

    close();
    run(\"Close All\");
    setBatchMode(false);
    logLine(\"Done\");
"""
            )

        # Default: fall back to area fraction
        return self._render_macro(preset_id="area_fraction", params=params)

    def _show_run_error_popup(self, *, title: str, header: str, detail: str) -> None:
        win = tk.Toplevel(self.winfo_toplevel())
        win.title(title)
        win.resizable(True, True)
        frm = ttk.Frame(win, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        frm.columnconfigure(0, weight=1)
        ttk.Label(frm, text=header, style="Header.TLabel").grid(row=0, column=0, sticky="w")
        txt = tk.Text(frm, height=10, wrap="word")
        txt.grid(row=1, column=0, sticky="nsew", pady=(8, 8))
        txt.insert("1.0", str(detail))
        txt.configure(state="disabled")

        btns = ttk.Frame(frm)
        btns.grid(row=2, column=0, sticky="e")
        close_btn = ttk.Button(btns, text="Close", command=win.destroy)
        close_btn.grid(row=0, column=1)

        # Open the most recent run log if we know it.
        def open_log() -> None:
            try:
                p = Path(str(self._last_run_log_path or "")).expanduser()
                if str(p) and p.exists():
                    os.startfile(str(p))  # type: ignore[attr-defined]
                    return
            except Exception:
                pass
            messagebox.showinfo(title, "Log file path was not detected in the error text.", parent=win)

        open_btn = ttk.Button(btns, text="Open log", command=open_log)
        open_btn.grid(row=0, column=0, padx=(0, 8))

        try:
            win.transient(self.winfo_toplevel())
            win.grab_set()
        except Exception:
            pass

    def _set_preset_log_text(self, text: str) -> None:
        try:
            self._preset_log.configure(state="normal")
            self._preset_log.delete("1.0", tk.END)
            self._preset_log.insert("1.0", text)
            self._preset_log.configure(state="disabled")
        except Exception:
            pass

    def _load_run_preview_from_entry(self, entry: Dict[str, Any]) -> None:
        # Update log box
        log_path = Path(str(entry.get("run_log") or ""))
        if log_path.exists():
            self._last_run_log_path = str(log_path)
            self._set_preset_log_text(_tail_text(log_path, n_lines=120))
        else:
            self._set_preset_log_text("(no log)")

        # Overlay preview
        overlay = Path(str(entry.get("overlay_png") or ""))
        if overlay.exists():
            try:
                im = Image.open(str(overlay))
                # Keep preview reasonably sized
                max_w, max_h = 520, 260
                im.thumbnail((max_w, max_h))
                img_tk = ImageTk.PhotoImage(im)
                self._overlay_photo = img_tk
                self._overlay_label.configure(image=img_tk, text="")
            except Exception:
                self._overlay_label.configure(text=f"Overlay saved: {overlay.name}")
                self._overlay_photo = None
        else:
            self._overlay_label.configure(text="(overlay preview)", image="")
            self._overlay_photo = None

        # Results CSV preview
        results = Path(str(entry.get("results_csv") or ""))
        self._load_results_preview(results)

    def _load_results_preview(self, csv_path: Path) -> None:
        # Clear
        try:
            self._results_tree.delete(*self._results_tree.get_children(""))
        except Exception:
            pass

        if not csv_path.exists():
            try:
                self._results_tree["columns"] = ("info",)
                self._results_tree.heading("info", text="Info")
                self._results_tree.insert("", "end", values=("(results.csv not found)",))
            except Exception:
                pass
            return

        try:
            df = pd.read_csv(str(csv_path))
        except Exception as exc:
            try:
                self._results_tree["columns"] = ("info",)
                self._results_tree.heading("info", text="Info")
                self._results_tree.insert("", "end", values=(f"Failed to read CSV: {exc}",))
            except Exception:
                pass
            return

        if df is None or df.empty:
            try:
                self._results_tree["columns"] = ("info",)
                self._results_tree.heading("info", text="Info")
                self._results_tree.insert("", "end", values=("(empty results)",))
            except Exception:
                pass
            return

        cols = [str(c) for c in list(df.columns)[:12]]
        try:
            self._results_tree["columns"] = tuple(cols)
            for c in cols:
                self._results_tree.heading(c, text=c)
                self._results_tree.column(c, width=110, stretch=True)
        except Exception:
            pass

        for _idx, row in df.head(50).iterrows():
            vals = []
            for c in cols:
                try:
                    v = row[c]
                    vals.append("" if pd.isna(v) else str(v))
                except Exception:
                    vals.append("")
            try:
                self._results_tree.insert("", "end", values=tuple(vals))
            except Exception:
                pass

    def _refresh_run_history_for_active(self) -> None:
        try:
            for iid in list(self._runs_tree.get_children("")):
                self._runs_tree.delete(iid)
        except Exception:
            pass

        did = self._active_dataset_id
        if not did:
            return
        items = list(self._runs_history.get(str(did), []))
        for e in items[-20:]:
            try:
                self._runs_tree.insert("", "end", values=(str(e.get("timestamp", "")), str(e.get("preset_name", ""))), tags=(str(e.get("run_dir", "")),))
            except Exception:
                continue

    def _on_run_history_select(self, _evt=None) -> None:
        try:
            sel = self._runs_tree.selection()
            if not sel:
                return
            iid = sel[0]
            vals = self._runs_tree.item(iid, "values")
            ts = str(vals[0]) if vals else ""
            did = self._active_dataset_id
            if not did:
                return
            entries = self._runs_history.get(str(did), [])
            for e in entries:
                if str(e.get("timestamp", "")) == ts:
                    self._load_run_preview_from_entry(e)
                    return
        except Exception:
            pass

    def _load_files(self) -> None:
        ws = self._ensure_default_workspace()

        settings = load_settings()
        initialdir = settings.get("last_microscopy_dir")
        if not initialdir:
            try:
                initialdir = str(Path.home())
            except Exception:
                initialdir = ""

        paths = filedialog.askopenfilenames(
            parent=self.winfo_toplevel(),
            title="Select microscopy files",
            initialdir=initialdir,
            filetypes=[
                ("Microscopy files", "*.tif *.tiff *.png *.jpg *.jpeg *.czi"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return

        try:
            settings["last_microscopy_dir"] = str(Path(str(paths[0])).parent)
            save_settings(settings)
        except Exception:
            pass

        for p in paths:
            pth = str(p)
            try:
                suffix = Path(pth).suffix.lower()
                if suffix and suffix not in SUPPORTED_INPUT_EXTS:
                    # still allow, but warn later
                    pass
            except Exception:
                pass

            ds = MicroscopyDataset(
                id=str(uuid.uuid4()),
                display_name=_safe_stem(pth),
                file_path=str(Path(pth).expanduser().resolve(strict=False)),
                workspace_id=str(ws.id),
                created_at=_utc_now_iso(),
                notes="",
                output_dir="",  # set below
                last_macro_run=None,
            )
            ds.output_dir = self._dataset_output_dir(ws, ds)
            try:
                Path(ds.output_dir).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            ws.datasets.append(ds)
            # New datasets start with 0 outputs.
            try:
                self._outputs_count_cache[str(ds.id)] = 0
            except Exception:
                pass

        self._refresh_dataset_tree(preserve_selection=False)
        self._refresh_outputs()
        self._maybe_update_status_bar()

    def _remove_selected(self) -> None:
        ids = self._selected_dataset_ids()
        if not ids:
            return
        if not messagebox.askyesno(
            "Remove datasets",
            f"Remove {len(ids)} selected dataset(s) from this session?\n\nInput files will NOT be deleted.",
            parent=self.winfo_toplevel(),
        ):
            return

        # Remove from their workspaces
        for did in ids:
            ds = self._find_dataset(did)
            if ds is None:
                continue
            ws = self._find_workspace(ds.workspace_id)
            if ws is None:
                continue
            try:
                ws.datasets = [d for d in (ws.datasets or []) if str(d.id) != str(did)]
            except Exception:
                pass

        self._active_dataset_id = None
        self._refresh_dataset_tree(preserve_selection=False)
        self._on_dataset_select(None)
        self._maybe_update_status_bar()

    def _open_selected_in_imagej(self) -> None:
        exe = self._ensure_imagej_path()
        if not exe:
            return

        ids = self._selected_dataset_ids()
        if not ids:
            messagebox.showinfo("Open in Fiji/ImageJ", "Select one or more datasets first.", parent=self.winfo_toplevel())
            return

        for did in ids:
            ds = self._find_dataset(did)
            if ds is None:
                continue
            try:
                if not Path(ds.file_path).exists():
                    continue
                run_fiji_open(exe, ds.file_path)
            except Exception as exc:
                messagebox.showerror("Open in Fiji/ImageJ", f"Failed to open:\n{ds.file_path}\n\n{exc}", parent=self.winfo_toplevel())
                return

        try:
            self.app._set_status(f"Opened {len(ids)} dataset(s) in Fiji/ImageJ")
        except Exception:
            pass

    def _set_output_folder_for_selected(self) -> None:
        did = self._active_dataset_id
        ds = self._find_dataset(did) if did else None
        if ds is None:
            messagebox.showinfo("Set Output Folder", "Select a single dataset first.", parent=self.winfo_toplevel())
            return

        initialdir = ""
        try:
            if ds.output_dir:
                initialdir = str(Path(ds.output_dir))
        except Exception:
            initialdir = ""

        folder = filedialog.askdirectory(parent=self.winfo_toplevel(), title="Choose output folder", initialdir=initialdir or None)
        if not folder:
            return

        try:
            Path(folder).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        ds.output_dir = str(Path(folder).expanduser())
        self._set_dataset_info(ds)
        self._refresh_outputs()
        self._maybe_update_status_bar()

    def _open_output_folder(self) -> None:
        ids = self._selected_dataset_ids()
        if ids:
            ds = self._find_dataset(ids[0])
            if ds is not None:
                p = Path(ds.output_dir)
                try:
                    p.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                self._open_folder(str(p))
                return

        # Fallback to microscopy root
        try:
            root = Path(self.app._get_session_root_dir()) / "microscopy"
        except Exception:
            root = Path.cwd() / "microscopy"
        try:
            root.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self._open_folder(str(root))

    def _open_folder(self, path: str) -> None:
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
        except Exception as exc:
            messagebox.showerror("Open folder", f"Failed to open folder:\n\n{exc}", parent=self.winfo_toplevel())

    def _refresh_outputs(self) -> None:
        # Only show outputs for a single active dataset.
        did = self._active_dataset_id
        ds = self._find_dataset(did) if did else None

        # Clear output tree
        try:
            for iid in list(self._out_tree.get_children("")):
                self._out_tree.delete(iid)
        except Exception:
            pass

        if ds is None:
            self._active_output_path = None
            return

        files = discover_outputs_limited(ds.output_dir, recursive=True, max_files=5000, max_seconds=2.0)

        count = int(len(files))
        try:
            self._outputs_count_cache[str(ds.id)] = count
        except Exception:
            pass

        # Update just this row's outputs_count (avoid rebuilding the dataset tree).
        try:
            if self._ds_tree.exists(str(ds.id)):
                self._ds_tree.set(str(ds.id), "outputs_count", count)
        except Exception:
            pass
        for f in files:
            try:
                stat = f.stat()
                size = stat.st_size
                mtime = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            except Exception:
                size = 0
                mtime = ""
            self._out_tree.insert("", "end", iid=str(f), values=(f.name, f.suffix.lower().lstrip("."), f"{size:,}", mtime))

        # Count is updated above; no full refresh here.

    def _import_outputs(self) -> None:
        ds = self._find_dataset(self._active_dataset_id) if self._active_dataset_id else None
        if ds is None:
            messagebox.showinfo("Import Outputs", "Select a single dataset first.", parent=self.winfo_toplevel())
            return

        src = filedialog.askdirectory(parent=self.winfo_toplevel(), title="Select folder containing output files")
        if not src:
            return

        srcp = Path(src)
        if not srcp.exists() or not srcp.is_dir():
            return

        dstp = Path(ds.output_dir)
        dstp.mkdir(parents=True, exist_ok=True)

        copied = 0
        for f in srcp.rglob("*"):
            try:
                if not f.is_file():
                    continue
                if f.suffix.lower() not in SUPPORTED_OUTPUT_EXTS:
                    continue
                shutil.copy2(str(f), str(dstp / f.name))
                copied += 1
            except Exception:
                continue

        self._refresh_outputs()
        self._refresh_dataset_tree()
        messagebox.showinfo("Import Outputs", f"Imported {copied} file(s) into the dataset output folder.", parent=self.winfo_toplevel())

    def _export_summary(self) -> None:
        ids = self._selected_dataset_ids()
        if not ids:
            messagebox.showinfo("Export Summary", "Select one or more datasets first.", parent=self.winfo_toplevel())
            return

        rows: List[Dict[str, Any]] = []
        for did in ids:
            ds = self._find_dataset(did)
            if ds is None:
                continue
            ws = self._find_workspace(ds.workspace_id)
            ws_name = ws.name if ws is not None else ""

            outs = discover_outputs(ds.output_dir)
            if not outs:
                rows.append(
                    {
                        "workspace": ws_name,
                        "dataset": ds.display_name,
                        "input_path": ds.file_path,
                        "output_dir": ds.output_dir,
                        "notes": ds.notes,
                        "last_macro_run": ds.last_macro_run,
                        "output_file": "",
                        "output_type": "",
                    }
                )
            else:
                for f in outs:
                    rows.append(
                        {
                            "workspace": ws_name,
                            "dataset": ds.display_name,
                            "input_path": ds.file_path,
                            "output_dir": ds.output_dir,
                            "notes": ds.notes,
                            "last_macro_run": ds.last_macro_run,
                            "output_file": f.name,
                            "output_type": f.suffix.lower(),
                        }
                    )

        df = pd.DataFrame(rows)

        path = filedialog.asksaveasfilename(
            parent=self.winfo_toplevel(),
            title="Export Microscopy Summary",
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv"), ("All files", "*.*")],
            initialfile="microscopy_summary.xlsx",
        )
        if not path:
            return

        try:
            p = Path(path)
            if p.suffix.lower() == ".csv":
                df.to_csv(str(p), index=False)
            else:
                df.to_excel(str(p), index=False)
        except Exception as exc:
            messagebox.showerror("Export Summary", f"Failed to export:\n\n{exc}", parent=self.winfo_toplevel())
            return

        messagebox.showinfo("Export Summary", f"Exported summary:\n\n{path}", parent=self.winfo_toplevel())

    def _open_selected_output(self) -> None:
        p = self._active_output_path
        if not p:
            return
        try:
            os.startfile(str(p))  # type: ignore[attr-defined]
        except Exception as exc:
            messagebox.showerror("Open output", f"Failed to open file:\n\n{exc}", parent=self.winfo_toplevel())

    # -------------------------- events --------------------------

    def _on_dataset_select(self, _evt) -> None:
        if bool(getattr(self, "_ds_ignore_select", False)):
            return
        ids = self._selected_dataset_ids()
        self._active_dataset_id = ids[0] if len(ids) == 1 else None

        # Notes: only editable for single selection
        try:
            self._notes_text.configure(state="normal" if self._active_dataset_id else "disabled")
        except Exception:
            pass

        if self._active_dataset_id is None:
            self._set_dataset_info(None)
            try:
                self._notes_text.delete("1.0", "end")
            except Exception:
                pass
            self._refresh_outputs()
            self._refresh_run_history_for_active()
            try:
                self._load_results_preview(Path(""))
            except Exception:
                pass
            try:
                self._overlay_label.configure(text="(overlay preview)", image="")
                self._overlay_photo = None
            except Exception:
                pass
            self._maybe_update_status_bar()
            return

        ds = self._find_dataset(self._active_dataset_id)
        self._set_dataset_info(ds)

        # Load notes
        try:
            self._notes_text.delete("1.0", "end")
            self._notes_text.insert("1.0", "" if ds is None else str(ds.notes or ""))
        except Exception:
            pass

        self._refresh_outputs()
        self._refresh_run_history_for_active()
        self._maybe_update_status_bar()

    def _on_notes_focus_out(self, _evt) -> None:
        ds = self._find_dataset(self._active_dataset_id) if self._active_dataset_id else None
        if ds is None:
            return
        try:
            text = self._notes_text.get("1.0", "end-1c")
        except Exception:
            return
        ds.notes = str(text)
        # Update just the notes cell (avoid a full refresh).
        try:
            if self._ds_tree.exists(str(ds.id)):
                preview = (ds.notes or "").strip().replace("\n", " ")[:60]
                self._ds_tree.set(str(ds.id), "notes", preview)
        except Exception:
            pass

    def _on_output_select(self, _evt) -> None:
        self._active_output_path = None
        try:
            sel = list(self._out_tree.selection())
        except Exception:
            sel = []
        if not sel:
            return
        self._active_output_path = str(sel[0])

    # -------------------------- internals --------------------------

    def _selected_dataset_ids(self) -> List[str]:
        try:
            return [str(x) for x in (self._ds_tree.selection() or [])]
        except Exception:
            return []

    def _refresh_dataset_tree(self, preserve_selection: bool = True) -> None:
        sel = self._selected_dataset_ids() if preserve_selection else []

        self._ds_ignore_select = True
        try:
            for iid in list(self._ds_tree.get_children("")):
                self._ds_tree.delete(iid)
        except Exception:
            pass

        # Rebuild
        for ws in self._get_workspaces():
            for ds in (ws.datasets or []):
                out_count = int(self._outputs_count_cache.get(str(ds.id), 0))

                self._ds_tree.insert(
                    "",
                    "end",
                    iid=str(ds.id),
                    values=(ws.name, ds.display_name, ds.file_path, (ds.notes or "").strip().replace("\n", " ")[:60], out_count),
                )

        if sel:
            try:
                # Filter to existing
                existing = set(self._ds_tree.get_children(""))
                keep = [s for s in sel if s in existing]
                if keep:
                    self._ds_tree.selection_set(keep)
                    self._ds_tree.see(keep[0])
            except Exception:
                pass

        self._ds_ignore_select = False

    def _set_dataset_info(self, ds: Optional[MicroscopyDataset]) -> None:
        if ds is None:
            self._info_path_var.set("")
            self._info_size_var.set("")
            self._info_mtime_var.set("")
            try:
                self._info_outdir_var.set("")
            except Exception:
                pass
            return

        self._info_path_var.set(str(ds.file_path))
        try:
            self._info_outdir_var.set(str(ds.output_dir or ""))
        except Exception:
            pass
        p = Path(ds.file_path)
        try:
            st = p.stat()
            self._info_size_var.set(f"{st.st_size:,} bytes")
            self._info_mtime_var.set(datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"))
        except Exception:
            self._info_size_var.set("(missing)")
            self._info_mtime_var.set("(missing)")

    def _call_ui(self, fn, *args, **kwargs) -> None:
        try:
            self.after(0, lambda: fn(*args, **kwargs))
        except Exception:
            pass

    def _maybe_update_status_bar(self) -> None:
        try:
            self.app._update_status_by_tab()
        except Exception:
            pass
