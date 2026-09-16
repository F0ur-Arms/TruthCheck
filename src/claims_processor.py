"""Phase 3: Claim Understanding, Medical Routing Gate & Claim Decomposition.

Implements:
1. Medical Routing Gate (Fact-Checkable Claim vs Personal Medical Advice).
2. Structured Claim Decomposition according to the TruthCheck v2 Canonical Schema.
3. Language identification & English Gloss generation.
4. Fallback heuristics when an LLM API endpoint is unconfigured or fails.
"""

from __future__ import annotations

import json
import re

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.llm_fallback import ConfiguredLLMVerifier


class SubClaim(BaseModel):
    claim_id: str = Field(default="c1")
    canonical_claim: str
    subject: Optional[str] = None
    intervention: Optional[str] = None
    population: Optional[str] = None
    population_confidence: str = Field(default="unspecified")
    outcome: Optional[str] = None
    relationship: str = Field(default="causal")  # causal, correlational, therapeutic
    direction: str = Field(default="harm")       # harm, benefit, neutral
    certainty_in_source_text: str = Field(default="assertive")
    scope: str = Field(default="general")


class CanonicalClaimRepresentation(BaseModel):
    original_text: str
    language: str = Field(default="en")
    script: str = Field(default="latin")
    normalized_text: str
    english_gloss: str
    claim_type: str = Field(default="causal_claim")
    route: str = Field(default="fact_check")     # "fact_check" or "medical_advice"
    claims: List[SubClaim] = Field(default_factory=list)
    safety_response: Optional[str] = None


ROUTING_PROMPT = """You are a medical safety classifier for a healthcare fact-checking engine.

Analyze the user input and determine whether it is:
1. "fact_check": A general, checkable factual claim or general question about health, medicine, diet, or physiology.
   Examples: "Does warm water improve digestion?", "Protein causes kidney failure", "Turmeric cures cancer".
2. "medical_advice": A personal medical question, personal symptom query, self-diagnosis, or personal treatment request.
   Examples: "I have stage 3 CKD, should I stop drinking whey protein?", "My child has fever 102, what medicine should I give?", "I have chest pain, how to cure it at home?".

Return a JSON object with keys:
- "route": "fact_check" or "medical_advice"
- "reason": brief explanation
"""

DECOMPOSITION_PROMPT = """You are an expert medical claim extraction and decomposition engine.

Analyze the given input sentence and perform structured claim decomposition:
1. Detect language (english, hinglish, hindi_devanagari) and script (latin, devanagari).
2. Generate a clear English gloss suitable for medical database searching.
3. Classify claim type (e.g. causal_claim, therapeutic_claim, myth_assertion).
4. Decompose into one or more atomic sub-claims. Each sub-claim must specify:
   - canonical_claim (clear English atomic statement)
   - subject (e.g. "whey protein", "warm water")
   - intervention (e.g. "protein supplementation")
   - population (stated target population, or null if unstated/general)
   - outcome (e.g. "kidney damage", "digestion")
   - direction ("harm", "benefit", "neutral")

Return valid JSON with the schema:
{
  "language": "...",
  "script": "...",
  "english_gloss": "...",
  "claim_type": "...",
  "claims": [
    {
      "claim_id": "c1",
      "canonical_claim": "...",
      "subject": "...",
      "intervention": "...",
      "population": null,
      "outcome": "...",
      "direction": "..."
    }
  ]
}
"""


class ClaimsProcessor:
    def __init__(self, llm_verifier: Optional[ConfiguredLLMVerifier] = None) -> None:
        self.llm = llm_verifier or ConfiguredLLMVerifier()

    def process_query(self, raw_text: str, cleaned_text: str) -> CanonicalClaimRepresentation:
        """Process input text through the medical routing gate and claim decomposition engine."""

        # 1. Try LLM Routing & Decomposition if configured
        if self.llm.configured:
            try:
                # Step 1: Medical Routing Gate
                route_res = self._call_llm_json(ROUTING_PROMPT, cleaned_text)
                route = route_res.get("route", "fact_check")
                
                if route == "medical_advice":
                    safety_msg = self._build_safety_response(raw_text)
                    return CanonicalClaimRepresentation(
                        original_text=raw_text,
                        language=self._detect_basic_language(cleaned_text),
                        script="latin",
                        normalized_text=cleaned_text,
                        english_gloss=cleaned_text,
                        claim_type="personal_medical_question",
                        route="medical_advice",
                        claims=[],
                        safety_response=safety_msg
                    )

                # Step 2: Decompose Claim
                decomp_res = self._call_llm_json(DECOMPOSITION_PROMPT, cleaned_text)
                sub_claims = []
                for idx, c_dict in enumerate(decomp_res.get("claims", []), 1):
                    sub_claims.append(
                        SubClaim(
                            claim_id=c_dict.get("claim_id", f"c{idx}"),
                            canonical_claim=c_dict.get("canonical_claim", cleaned_text),
                            subject=c_dict.get("subject"),
                            intervention=c_dict.get("intervention"),
                            population=c_dict.get("population"),
                            outcome=c_dict.get("outcome"),
                            direction=c_dict.get("direction", "harm")
                        )
                    )

                if not sub_claims:
                    sub_claims = [SubClaim(canonical_claim=decomp_res.get("english_gloss", cleaned_text))]

                return CanonicalClaimRepresentation(
                    original_text=raw_text,
                    language=decomp_res.get("language", "en"),
                    script=decomp_res.get("script", "latin"),
                    normalized_text=cleaned_text,
                    english_gloss=decomp_res.get("english_gloss", cleaned_text),
                    claim_type=decomp_res.get("claim_type", "causal_claim"),
                    route="fact_check",
                    claims=sub_claims
                )
            except Exception:
                # LLM failed, fallback to heuristic processing
                pass

        # 2. Heuristic Rule-Based Processing (Fallback)
        return self._heuristic_processing(raw_text, cleaned_text)

    def _heuristic_processing(self, raw_text: str, cleaned_text: str) -> CanonicalClaimRepresentation:
        """Deterministic heuristic processing when LLM API is disabled or fails."""
        # Check heuristic medical advice triggers
        advice_triggers = [
            r"\bi have\b", r"\bmy child\b", r"\bmy doctor\b", r"\bshould i\b",
            r"\bwhat medicine\b", r"\bhow to cure my\b", r"\bcan i take\b",
            r"\bmera\b", r"\bmeri\b", r"\bmujhe\b", r"\bkya mai\b"
        ]
        text_lower = raw_text.lower()
        is_advice = any(re.search(pat, text_lower) for pat in advice_triggers)

        if is_advice:
            return CanonicalClaimRepresentation(
                original_text=raw_text,
                language=self._detect_basic_language(cleaned_text),
                script="latin",
                normalized_text=cleaned_text,
                english_gloss=cleaned_text,
                claim_type="personal_medical_question",
                route="medical_advice",
                claims=[],
                safety_response=self._build_safety_response(raw_text)
            )

        # Basic decomposition heuristic (split on 'and', 'aur')
        sub_claim_texts = [s.strip() for s in re.split(r"\b(?:and|aur|evam)\b", cleaned_text) if s.strip()]
        if not sub_claim_texts:
            sub_claim_texts = [cleaned_text]

        sub_claims = [
            SubClaim(
                claim_id=f"c{idx+1}",
                canonical_claim=stmt
            )
            for idx, stmt in enumerate(sub_claim_texts)
        ]

        return CanonicalClaimRepresentation(
            original_text=raw_text,
            language=self._detect_basic_language(cleaned_text),
            script="latin",
            normalized_text=cleaned_text,
            english_gloss=cleaned_text,
            claim_type="general_assertion",
            route="fact_check",
            claims=sub_claims
        )

    def _call_llm_json(self, system_prompt: str, user_text: str) -> Dict[str, Any]:
        """Helper to invoke configured LLM with JSON payload."""
        import requests
        headers = {"Authorization": f"Bearer {self.llm.api_key}"}
        payload = {
            "model": self.llm.model_name,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ]
        }
        res = requests.post(
            f"{self.llm.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=15
        )
        res.raise_for_status()
        content = res.json()["choices"][0]["message"]["content"]
        return json.loads(content)

    def _detect_basic_language(self, text: str) -> str:
        hinglish_words = {"paani", "pet", "khali", "subah", "garam", "bimari", "ilaj", "nuksan", "se", "hai", "ka", "ki", "ke"}
        words = set(re.findall(r"\w+", text.lower()))
        if words.intersection(hinglish_words):
            return "hinglish"
        return "en"

    def _build_safety_response(self, query: str) -> str:
        return (
            "⚠️ Medical Safety Advice Notice: "
            "This query appears to be a personal medical advice request or specific symptom question. "
            "TruthCheck is an automated medical fact-checking engine and cannot provide personal medical diagnoses, "
            "prescriptions, or individualized treatment plans. Please consult a licensed doctor, dietitian, or qualified healthcare professional. "
            "If this is an emergency, seek urgent medical attention immediately."
        )
