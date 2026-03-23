from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import tkinter as tk

import ttkbootstrap as tb
import tkinter.ttk as ttk_native

from lab_gui.ai_assistant import AIAssistant
from lab_gui.ai_context import build_ai_context
from lab_gui.ui_theme import style_primary, style_secondary


ttk: Any = tb
ttk.LabelFrame = ttk_native.LabelFrame  # type: ignore[attr-defined]


class AIPanel(ttk.Frame):
    """Read-only in-app AI assistant panel.

    The first implementation is intentionally safe:
    - no action execution
    - no dataset mutation
    - no autonomous behavior
    """

    def __init__(self, parent: tk.Widget, app: Any, workspace: Any, *, assistant: Optional[AIAssistant] = None) -> None:
        """Implement the `__init__` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        super().__init__(parent)
        self.app = app
        self.workspace = workspace
        self.assistant = assistant or AIAssistant()

        self._status_var = tk.StringVar(value="Ready")
        self._mode_var = tk.StringVar(value=self.assistant.mode_hint())
        self._include_context_var = tk.BooleanVar(value=True)

        self._history_text: Optional[tk.Text] = None
        self._input_text: Optional[tk.Text] = None
        self._ask_btn: Optional[Any] = None

        self._build_ui()

    def status_text(self) -> str:
        """Return a short status string for the app status bar."""
        try:
            return str(self._status_var.get())
        except Exception:
            return ""

    def _build_ui(self) -> None:
        """Build and return composed application state.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, style="ShellPanel.TFrame", padding=(14, 12))
        header.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=0)

        ttk.Label(header, text="AI Assistant", style="SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self._mode_var, style="AppEyebrow.TLabel").grid(row=0, column=1, sticky="e")
        ttk.Label(
            header,
            text="Ask about app functionality, module purpose, and lab-analysis concepts. This panel is read-only.",
            style="CardHint.TLabel",
            wraplength=900,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        body = ttk.LabelFrame(self, text="Conversation", padding=8, style="Card.TLabelframe")
        body.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        body.rowconfigure(1, weight=0)

        hist_wrap = ttk.Frame(body)
        hist_wrap.grid(row=0, column=0, sticky="nsew")
        hist_wrap.columnconfigure(0, weight=1)
        hist_wrap.rowconfigure(0, weight=1)

        hist = tk.Text(hist_wrap, wrap="word", state="disabled", height=18)
        hist.grid(row=0, column=0, sticky="nsew")
        hist_sb = ttk.Scrollbar(hist_wrap, orient=tk.VERTICAL, command=hist.yview)
        hist_sb.grid(row=0, column=1, sticky="ns")
        hist.configure(yscrollcommand=hist_sb.set)
        self._history_text = hist

        controls = ttk.Frame(body)
        controls.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        controls.columnconfigure(0, weight=1)

        input_box = tk.Text(controls, wrap="word", height=5)
        input_box.grid(row=0, column=0, columnspan=3, sticky="ew")
        self._input_text = input_box

        cb = ttk.Checkbutton(
            controls,
            text="Include current tab context",
            variable=self._include_context_var,
        )
        cb.grid(row=1, column=0, sticky="w", pady=(8, 0))

        ask_btn = ttk.Button(controls, text="Ask", command=self._on_submit)
        ask_btn.grid(row=1, column=1, sticky="e", padx=(8, 0), pady=(8, 0))
        style_primary(ask_btn)
        self._ask_btn = ask_btn

        clear_btn = ttk.Button(controls, text="Clear", command=self._on_clear)
        clear_btn.grid(row=1, column=2, sticky="e", padx=(8, 0), pady=(8, 0))
        style_secondary(clear_btn)

        try:
            self.bind_all("<Control-Return>", lambda _e: self._on_submit(), add=True)
        except Exception:
            pass

        self._append_message("Assistant", "Ready. Ask a question about the app or the active module.")

    def _on_submit(self) -> None:
        """Implement the `_on_submit` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        self._mode_var.set(self.assistant.mode_hint())
        question = self._current_input().strip()
        if not question:
            self._status_var.set("Enter a question")
            return

        self._set_busy(True)
        self._append_message("You", question)

        include_ctx = bool(self._include_context_var.get())
        context_payload = None
        if include_ctx:
            ctx = build_ai_context(self.app, self.workspace)
            context_payload = ctx.to_prompt_dict()

        try:
            reply = self.assistant.ask(
                question,
                context=context_payload,
                include_context=include_ctx,
            )
            text = reply.text
            if reply.is_mock:
                text += "\n\n(source: demo mode)"
                if reply.error:
                    text += f"\n(note: live model unavailable: {reply.error})"
            self._append_message("Assistant", text)
            self._status_var.set("Replied")
        except Exception as exc:
            self._append_message("Assistant", f"I am unsure due to an internal error: {exc}")
            self._status_var.set("Error")
        finally:
            self._set_busy(False)
            self._clear_input()

    def _on_clear(self) -> None:
        """Implement the `_on_clear` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        hist = self._history_text
        if hist is None:
            return
        hist.configure(state="normal")
        hist.delete("1.0", tk.END)
        hist.configure(state="disabled")
        self._status_var.set("Cleared")

    def _append_message(self, speaker: str, text: str) -> None:
        """Implement the `_append_message` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        hist = self._history_text
        if hist is None:
            return

        now = datetime.now().strftime("%H:%M")
        hist.configure(state="normal")
        hist.insert(tk.END, f"[{now}] {speaker}:\n{text.strip()}\n\n")
        hist.see(tk.END)
        hist.configure(state="disabled")

    def _set_busy(self, is_busy: bool) -> None:
        """Implement the `_set_busy` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        btn = self._ask_btn
        if btn is None:
            return
        try:
            btn.configure(state=("disabled" if is_busy else "normal"))
        except Exception:
            pass
        self._status_var.set("Thinking..." if is_busy else "Ready")

    def _current_input(self) -> str:
        """Implement the `_current_input` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        w = self._input_text
        if w is None:
            return ""
        try:
            return str(w.get("1.0", tk.END) or "")
        except Exception:
            return ""

    def _clear_input(self) -> None:
        """Implement the `_clear_input` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        w = self._input_text
        if w is None:
            return
        try:
            w.delete("1.0", tk.END)
        except Exception:
            pass


# TODO: Move model calls to a background worker once UX requirements for cancellation are defined.
# TODO: Add structured read-only tool-calling hooks (context inspectors) after safety review.
