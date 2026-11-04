"""Explicit, provider-configured LLM fallback for unresolved claims."""

from __future__ import annotations

import json
from typing import Any

import requests

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT_SECONDS


class ConfiguredLLMVerifier:
    """Use an OpenAI-compatible chat-completions endpoint when configured.

    No endpoint or key means the fallback is deliberately unavailable.  This
    prevents a silent dependency on a developer's local Ollama process.
    """

    def __init__(self) -> None:
        self.base_url = LLM_BASE_URL
        self.api_key = LLM_API_KEY
        self.model_name = LLM_MODEL

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model_name)

    def explain(self, claim: str, verdict: str, passages: list[str]) -> str | None:
        """Explain a verdict already decided, from the retrieved passages alone."""
        if not self.configured:
            return None

        evidence = "\n\n".join(f"Passage {i}: {p[:1500]}" for i, p in enumerate(passages, 1))
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model_name,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "A fact-check verdict on a health claim has already been decided. Explain it to a "
                            "member of the public in at most three plain sentences. Use only what the passages "
                            "say and add no facts of your own, and do not mention passage numbers. If the passages "
                            "do not settle the claim, say so. "
                            "Do not give medical advice. Return JSON only: {\"explanation\": \"...\"}."
                        ),
                    },
                    {"role": "user", "content": f"Claim: {claim}\nVerdict: {verdict}\n\n{evidence}"},
                ],
            },
            timeout=LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        result = json.loads(response.json()["choices"][0]["message"]["content"])
        return str(result.get("explanation", "")).strip() or None

    def verify(self, claim_text: str) -> dict[str, Any] | None:
        if not self.configured:
            return None

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model_name,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Classify a health claim as TRUE, FALSE, MIXED, or "
                            "UNVERIFIED. Return JSON only with verdict, confidence "
                            "(0 to 1), and explanation. Do not give medical advice."
                        ),
                    },
                    {"role": "user", "content": claim_text},
                ],
            },
            timeout=LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        result = json.loads(content)
        verdict = str(result.get("verdict", "UNVERIFIED")).upper()
        if verdict not in {"TRUE", "FALSE", "MIXED", "UNVERIFIED"}:
            verdict = "UNVERIFIED"
        confidence = float(result.get("confidence", 0.0))
        return {
            "verdict": verdict,
            "confidence": min(max(confidence, 0.0), 1.0),
            "explanation": str(result.get("explanation", "No explanation returned.")),
            "source": f"Configured LLM ({self.model_name})",
        }
