"""NLI judge backed by the vLLM-served LLM, with a local NLI model as fallback."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict

import requests

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

LABELS = ("SUPPORTS", "REFUTES", "NEI")
# Share of rows the judge got right when it predicted each label on 2,259 labelled
# claim/evidence pairs (evaluation/baselines/step4_llm_judge). Interim confidences
# until the calibrator is fitted on the golden set.
LABEL_PRECISION = {"SUPPORTS": 0.58, "REFUTES": 0.79, "NEI": 0.55}
# Prompt used until round 2 (evaluation/baselines/round2); kept so the two can be compared.
SYSTEM_V1 = (
    "You judge whether an evidence passage supports or refutes a health claim. "
    "Use only the passage. Answer SUPPORTS if the passage backs the claim, REFUTES if it contradicts the claim, "
    "NEI if the passage does not settle the claim. Return JSON: {\"label\": \"SUPPORTS|REFUTES|NEI\"}."
)
SYSTEM = (
    "You judge whether an evidence passage supports or refutes a health claim. Use only the passage.\n"
    "SUPPORTS: the passage states or directly implies the claim.\n"
    "REFUTES: the passage is about the same subject and directly contradicts the claim.\n"
    "NEI: anything else, including a passage about a different topic, condition, population or "
    "outcome, or one that is related but neither confirms nor contradicts the claim. "
    "A passage that fails to mention the claim is NEI, not REFUTES.\n"
    "Return JSON: {\"label\": \"SUPPORTS|REFUTES|NEI\"}."
)


def _result(label: str, claim: str, evidence_text: str) -> Dict[str, Any]:
    conf = LABEL_PRECISION[label]
    rest = (1.0 - conf) / 2
    probs = {k: (conf if k == label else rest) for k in LABELS}
    return {
        "verdict": label,
        "confidence": conf,
        "label_probabilities": probs,
        "supports_probability": probs["SUPPORTS"],
        "refutes_probability": probs["REFUTES"],
        "claim": claim,
        "evidence": evidence_text[:200],
    }


class LLMNLIJudge:
    def __init__(self, base_url: str = LLM_BASE_URL, api_key: str = LLM_API_KEY,
                 model: str = LLM_MODEL, timeout: float = 60.0, system_prompt: str = SYSTEM):
        self.base_url, self.api_key, self.model, self.timeout = base_url, api_key, model, timeout
        self.system_prompt = system_prompt

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)

    def verify(self, claim: str, evidence_text: str) -> Dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model, "temperature": 0, "max_tokens": 20,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Claim: {claim}\n\nPassage: {evidence_text[:3000]}"},
                ],
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        label = str(json.loads(response.json()["choices"][0]["message"]["content"]).get("label", "")).upper()
        if label not in LABELS:
            raise ValueError(f"unexpected NLI label from LLM: {label!r}")
        return _result(label, claim, evidence_text)


class FallbackNLI:
    """Use the LLM judge; on any failure use the local NLI model, built on first need."""

    def __init__(self, primary, fallback_factory: Callable[[], Any]):
        self.primary = primary
        self._fallback_factory = fallback_factory
        self._fallback = None
        self.fallback_calls = 0

    def verify(self, claim: str, evidence_text: str) -> Dict[str, Any]:
        try:
            return self.primary.verify(claim, evidence_text)
        except Exception:
            if self._fallback is None:
                self._fallback = self._fallback_factory()
            self.fallback_calls += 1
            return self._fallback.verify(claim, evidence_text)
