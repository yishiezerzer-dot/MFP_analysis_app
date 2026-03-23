from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Protocol


SYSTEM_PROMPT = (
    "A lab software assistant that helps explain app functionality, module purpose, and analysis concepts clearly.\n"
    "You support FTIR, LCMS, plate reader, and general lab-analysis concepts.\n"
    "Use only the provided context and user message.\n"
    "If information is missing or uncertain, explicitly say you are unsure.\n"
    "Do not claim to have executed analysis, clicked UI controls, modified files, or changed datasets.\n"
    "Do not invent app state details."
)


class LLMClientProtocol(Protocol):
    """Minimal provider protocol for future LLM client integrations."""

    def generate_reply(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str,
        timeout_seconds: float,
    ) -> str:
        """Implement the `generate_reply` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        ...


@dataclass
class AIAssistantResponse:
    """Assistant reply payload used by the UI panel."""

    text: str
    is_mock: bool = False
    used_context: bool = False
    model: str = ""
    error: Optional[str] = None


class AIAssistant:
    """Read-only assistant orchestration layer for the app.

    This class builds a constrained prompt from user input plus optional app context,
    and delegates text generation to an injected LLM client.
    """

    def __init__(
        self,
        *,
        llm_client: Optional[LLMClientProtocol] = None,
        model: str = "gpt-4.1-mini",
        api_key_env_var: str = "OPENAI_API_KEY",
        timeout_seconds: float = 20.0,
    ) -> None:
        """Implement the `__init__` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        self._llm_client = llm_client
        self._model = str(model or "gpt-4.1-mini")
        self._api_key_env_var = str(api_key_env_var or "OPENAI_API_KEY")
        self._timeout_seconds = float(timeout_seconds)

    def ask(
        self,
        user_message: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
        include_context: bool = True,
    ) -> AIAssistantResponse:
        """Answer a user question using optional read-only app context."""
        question = str(user_message or "").strip()
        if not question:
            return AIAssistantResponse(
                text="Please enter a question.",
                is_mock=True,
                used_context=bool(include_context and context),
                model=self._model,
            )

        used_context = bool(include_context and context)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._build_user_prompt(question, context if used_context else None),
            },
        ]

        if self._llm_client is None:
            return self._mock_response(question, context=context if used_context else None, used_context=used_context)

        try:
            text = str(
                self._llm_client.generate_reply(
                    messages,
                    model=self._model,
                    timeout_seconds=self._timeout_seconds,
                )
            ).strip()
            if not text:
                raise RuntimeError("Empty model response")
            return AIAssistantResponse(text=text, is_mock=False, used_context=used_context, model=self._model)
        except Exception as exc:
            fallback = self._mock_response(question, context=context if used_context else None, used_context=used_context)
            fallback.error = str(exc)
            return fallback

    def mode_hint(self) -> str:
        """Return a small UI hint describing whether live model mode is available."""
        if self._llm_client is None:
            return "Mode: Demo mode"
        client_hint = getattr(self._llm_client, "mode_hint", None)
        if callable(client_hint):
            try:
                return str(client_hint())
            except Exception:
                pass
        return "Mode: Live model"

    def _has_api_key_configured(self) -> bool:
        """Implement the `_has_api_key_configured` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        try:
            return bool(str(os.environ.get(self._api_key_env_var, "")).strip())
        except Exception:
            return False

    def _build_user_prompt(self, question: str, context: Optional[Mapping[str, Any]]) -> str:
        """Build and return composed application state.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        if not context:
            return f"User question:\n{question}\n\nNo app context was provided for this request."

        parts: List[str] = [f"User question:\n{question}", "", "Read-only app context:"]
        for key in ("active_module", "module_summary"):
            val = context.get(key)
            if val:
                parts.append(f"- {key}: {val}")

        files = context.get("loaded_filenames")
        if isinstance(files, list) and files:
            parts.append("- loaded_filenames: " + ", ".join(str(x) for x in files[:12]))

        parts.append("")
        parts.append("Answer with clear, practical guidance. If unsure, say so.")
        return "\n".join(parts)

    def _mock_response(
        self,
        question: str,
        *,
        context: Optional[Mapping[str, Any]],
        used_context: bool,
    ) -> AIAssistantResponse:
        """Implement the `_mock_response` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        module_name = "the app"
        summary = ""
        if isinstance(context, Mapping):
            module_name = str(context.get("active_module") or module_name)
            summary = str(context.get("module_summary") or "").strip()

        text = (
            "[Demo mode] No live LLM is configured yet, so this is a safe mock response.\n\n"
            f"You asked: {question}\n"
            f"Current focus: {module_name}.\n"
        )
        if summary:
            text += f"Context summary: {summary}\n"
        text += (
            "\nI can help explain what this module does, which file is active, and how common FTIR/LCMS/plate-reader "
            "analysis concepts work. I may be missing details, so I will say when I am unsure."
        )

        return AIAssistantResponse(
            text=text,
            is_mock=True,
            used_context=used_context,
            model=self._model,
        )


# TODO: Add provider-specific adapters (for example OpenAI client wrappers) here.
# TODO: Add optional structured outputs once read-only tool-calling is introduced.
