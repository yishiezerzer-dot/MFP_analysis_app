from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Dict, List


class OllamaChatClient:
    """Minimal Ollama chat adapter implementing the assistant protocol.

    This adapter targets a local Ollama server and returns assistant text only.
    It is read-only and does not support tool-calling.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        endpoint: str = "/api/chat",
    ) -> None:
        """Implement the `__init__` behavior for this module.

        Text-only documentation note: modify internal logic here to change behavior.
        """
        url = str(base_url or os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).strip().rstrip("/")
        ep = str(endpoint or "/api/chat").strip()
        if not ep.startswith("/"):
            ep = "/" + ep
        self._chat_url = f"{url}{ep}"

    def mode_hint(self) -> str:
        """Return a short UI label for the active provider mode."""
        return "Mode: Live model (Ollama local)"

    def generate_reply(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str,
        timeout_seconds: float,
    ) -> str:
        """Generate a text-only assistant reply from Ollama chat API."""
        payload = {
            "model": str(model),
            "messages": messages,
            "stream": False,
        }

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._chat_url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=float(timeout_seconds)) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        try:
            data = json.loads(raw)
            message = data.get("message") or {}
            text = message.get("content")
        except Exception as exc:
            raise RuntimeError(f"Ollama response parse failed: {exc}") from exc

        return str(text or "").strip()
