"""Orchestrates tagging, transliteration, glossing, and safety validation."""

from __future__ import annotations

from dataclasses import dataclass

from .gloss_generator import IndicTrans2Backend, TranslationBackend
from .medical_entity_guard import MedicalEntityGuard
from .linguistic_scorer import DevanagariAwareScorer
from .scripts import detect_script, language_code_for_script
from .semantic_preservation import SemanticPreserver
from .token_lang_detector import HinglishLIDClassifier, TokenLanguageDetector
from .transliteration import QwenHinglishTransliterator, Transliterator
from .types import MultilingualClaim


@dataclass(frozen=True)
class MultilingualProcessorConfig:
    hinglish_lid_model: str = "PhysicsWallahAI/muril-hinglish-lid"
    indictrans_local_files_only: bool = False
    neural_models_local_files_only: bool = False
    transliteration_language: str = "hi"
    enable_linguistic_score: bool = True


class MultilingualProcessor:
    """Standalone processor; no existing TruthCheck modules are imported."""

    def __init__(
        self,
        config: MultilingualProcessorConfig | None = None,
        *,
        detector: TokenLanguageDetector | None = None,
        transliterator: Transliterator | None = None,
        translator: TranslationBackend | None = None,
        semantic_preserver: SemanticPreserver | None = None,
        entity_guard: MedicalEntityGuard | None = None,
        scorer: DevanagariAwareScorer | None = None,
    ) -> None:
        self.config = config or MultilingualProcessorConfig()
        self.detector = detector or TokenLanguageDetector(HinglishLIDClassifier(
            self.config.hinglish_lid_model, local_files_only=self.config.neural_models_local_files_only
        ))
        self.transliterator = transliterator or QwenHinglishTransliterator(
            local_files_only=self.config.neural_models_local_files_only
        )
        self.translator = translator or IndicTrans2Backend(
            local_files_only=self.config.indictrans_local_files_only
        )
        self.semantic_preserver = semantic_preserver or SemanticPreserver(
            local_files_only=self.config.neural_models_local_files_only
        )
        # Medical entity extraction now uses token-level language tags (no external model).
        # Enable it by default since it has zero dependencies and high accuracy.
        self.entity_guard = entity_guard or MedicalEntityGuard()
        self.scorer = scorer
        if self.scorer is None and self.config.enable_linguistic_score:
            self.scorer = DevanagariAwareScorer(local_files_only=self.config.neural_models_local_files_only)

    def process(self, text: str) -> MultilingualClaim:
        if not text or not text.strip():
            raise ValueError("A non-empty claim is required.")

        tags = self.detector.tag(text)
        script = detect_script(text)
        warnings: list[str] = []
        canonical = self._canonicalize(text, tags, warnings)
        source_language = self._source_language(script, tags)

        # English has no derived representation to protect. Return before
        # loading translation models, which keeps this component fast.
        if source_language == "eng_Latn" and not any(tag.label == "hi_Latn" for tag in tags):
            claim = MultilingualClaim(
                original_text=text,
                script=script,
                source_language=source_language,
                token_tags=tags,
                canonical_indic_text=canonical,
                english_gloss=text,
                transformation_confidence=1.0,
                warnings=warnings,
            )
            self._add_linguistic_score(claim)
            return claim

        if self.entity_guard is not None:
            try:
                # Extract medical entities from token tags (high-confidence source)
                # Falls back to pattern matching if tags aren't available.
                entities = sorted(self.entity_guard.protected_entities(canonical, tags))
            except Exception as exc:
                entities = []
                warnings.append(f"Medical entity extraction failed: {type(exc).__name__}: {exc}")
        else:
            entities = []
        claim = MultilingualClaim(
            original_text=text,
            script=script,
            source_language=source_language,
            token_tags=tags,
            protected_entities=entities,
            canonical_indic_text=canonical,
            warnings=warnings,
        )

        try:
            gloss = self.translator.translate(canonical, source_language, "eng_Latn")
        except Exception as exc:
            claim.warnings.append(f"Translation unavailable: {type(exc).__name__}: {exc}")
            self._add_linguistic_score(claim)
            return claim

        validation = self.semantic_preserver.validate(text, gloss)
        claim.validation = validation
        claim.transformation_confidence = validation.confidence
        if validation.accepted:
            claim.english_gloss = gloss
        else:
            claim.warnings.extend(validation.warnings)
        self._add_linguistic_score(claim)
        return claim

    def _canonicalize(self, text: str, tags, warnings: list[str]) -> str:
        if not any(tag.label == "hi_Latn" for tag in tags):
            return text
        try:
            sentence_method = getattr(self.transliterator, "transliterate_text", None)
            if callable(sentence_method):
                return sentence_method(text, tags, self.config.transliteration_language)
        except Exception as exc:
            warnings.append(f"Transliteration unavailable: {type(exc).__name__}: {exc}")
            return text

        # Adapter fallback for older injected implementations.
        output: list[str] = []
        previous_end = 0
        for tag in tags:
            # Preserve original whitespace exactly enough for sentence translation.
            if output and tag.start > previous_end:
                output.append(" ")
            token = tag.token
            if tag.label == "hi_Latn":
                try:
                    token = self.transliterator.transliterate(token, self.config.transliteration_language)
                except Exception as exc:
                    warnings.append(f"Transliteration unavailable: {type(exc).__name__}: {exc}")
            output.append(token)
            previous_end = tag.end
        return "".join(output).strip()

    def _add_linguistic_score(self, claim: MultilingualClaim) -> None:
        if self.scorer is None:
            return
        try:
            claim.linguistic_score = self.scorer.calculate_score_detailed(claim.original_text)
        except Exception as exc:
            claim.warnings.append(f"Linguistic scoring unavailable: {type(exc).__name__}: {exc}")

    @staticmethod
    def _source_language(script: str, tags) -> str:
        if any(tag.label == "hi_Latn" for tag in tags):
            return "hin_Deva"
        if script == "Mixed" and any(tag.label == "hi_Deva" for tag in tags):
            return "hin_Deva"
        return language_code_for_script(script)
