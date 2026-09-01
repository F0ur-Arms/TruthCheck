"""Neural token-level language tagging for Hindi-English code-mixed text."""

from __future__ import annotations

import re
from collections.abc import Mapping

from .scripts import char_script, iter_tokens
from .types import TokenTag

URL_PATTERN = re.compile(r"^(?:https?://|www\.)", re.IGNORECASE)
NUMBER_UNIT_PATTERN = re.compile(r"^\d+(?:[.,]\d+)?(?:[a-zA-Z%]+)?$")


class HinglishLIDClassifier:
    """Lazy local MuRIL token-classification inference; no word lists."""

    def __init__(self, model_name: str = "PhysicsWallahAI/muril-hinglish-lid", *, local_files_only: bool = False) -> None:
        self.model_name, self.local_files_only, self._pipeline = model_name, local_files_only, None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install multilingual/requirements.txt to use neural Hinglish LID.") from exc
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=self.local_files_only)
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, local_files_only=self.local_files_only
        )
        self._pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer,
                                  aggregation_strategy="first")

    def classify(self, text: str) -> Mapping[tuple[int, int], tuple[str, float]]:
        self._load()
        return {(item["start"], item["end"]): (str(item["entity_group"]), float(item["score"]))
                for item in self._pipeline(text)}


class TokenLanguageDetector:
    """Tags each token. Every Latin Hindi/English choice is model-derived."""

    def __init__(self, classifier: HinglishLIDClassifier | None = None) -> None:
        self.classifier = classifier or HinglishLIDClassifier()

    def tag(self, text: str) -> list[TokenTag]:
        # Do not load the Hinglish model for pure Devanagari/non-Latin claims.
        # Script assignment for those tokens is deterministic Unicode metadata.
        labels = self.classifier.classify(text) if any(char_script(char) == "Latin" for char in text if char.isalpha()) else {}
        result: list[TokenTag] = []
        for token, start, end in iter_tokens(text):
            if URL_PATTERN.match(token):
                result.append(TokenTag(token, start, end, "entity", 1.0)); continue
            if NUMBER_UNIT_PATTERN.match(token):
                result.append(TokenTag(token, start, end, "number_unit", 1.0)); continue
            script = next((char_script(char) for char in token if char.isalpha()), "Common")
            if script == "Devanagari":
                result.append(TokenTag(token, start, end, "hi_Deva", 1.0)); continue
            if script in {"Bengali", "Gujarati", "Gurmukhi", "Kannada", "Malayalam", "Odia", "Tamil", "Telugu", "Arabic"}:
                result.append(TokenTag(token, start, end, "other_Indic", 1.0)); continue
            if script != "Latin":
                result.append(TokenTag(token, start, end, "other", 1.0)); continue
            overlapping = [value for (left, right), value in labels.items() if left < end and right > start]
            if not overlapping:
                raise RuntimeError(f"Neural LID returned no label for token {token!r}.")
            label, confidence = max(overlapping, key=lambda value: value[1])
            normalized = str(label).casefold().replace("label_", "")
            if normalized in {"hin", "hi", "hindi", "hi_latn"}:
                output_label = "hi_Latn"
            elif normalized in {"eng", "en", "english"}:
                output_label = "en"
            else:
                raise RuntimeError(f"Unsupported LID label from configured model: {label!r}")
            result.append(TokenTag(token, start, end, output_label, confidence))
        return result
