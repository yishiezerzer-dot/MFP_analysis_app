from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, cast


class OpenAIChatClient:
    """Minimal OpenAI chat adapter implementing the assistant protocol.

    This adapter is intentionally simple and read-only:
    - it sends chat messages to a model
    - it returns plain assistant text only
    - it does not perform tool calling or state-changing actions
    """

    def __init__(self, *, api_key_env_var: str = "OPENAI_API_KEY") -> None:
        """Implement the `__init__` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        self._api_key_env_var = str(api_key_env_var or "OPENAI_API_KEY")
        self._api_key = str(os.environ.get(self._api_key_env_var, "") or "").strip() or None
        self._client = None

        if not self._api_key:
            return

        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError(f"OpenAI SDK is not available: {exc}") from exc

        self._client = OpenAI(api_key=self._api_key)

    def mode_hint(self) -> str:
        """Return a short UI label for the active provider mode."""
        if self._client is None:
            return f"Mode: Demo mode (missing {self._api_key_env_var})"
        return "Mode: Live model (OpenAI)"

    def generate_reply(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str,
        timeout_seconds: float,
    ) -> str:
        """Generate a text-only assistant reply from OpenAI chat completions."""
        if self._client is None:
            raise RuntimeError(f"{self._api_key_env_var} is not configured")

        try:
            response = self._client.chat.completions.create(
                model=str(model),
                messages=cast(Any, messages),
                timeout=float(timeout_seconds),
            )
        except Exception as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

        try:
            text: Optional[str] = response.choices[0].message.content
        except Exception:
            text = None

        return str(text or "").strip()
