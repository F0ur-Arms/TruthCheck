"""Contract tests plus opt-in local neural model acceptance checks.

Run lightweight checks:
    python -m unittest multilingual.test_translation -v

Run live local-model checks (downloads free weights on first run):
    $env:RUN_REAL_TRANSLATION='1'; python -m unittest multilingual.test_translation -v
"""

from __future__ import annotations

import os
import unittest

from .gloss_generator import IndicTrans2Backend
from .processor import MultilingualProcessor, MultilingualProcessorConfig
from .semantic_preservation import SemanticPreserver
from .scripts import iter_tokens
from .token_lang_detector import HinglishLIDClassifier, TokenLanguageDetector
from .transliteration import QwenHinglishTransliterator
from .types import TokenTag, TransformationValidation


class _StaticDetector:
    def __init__(self, tags_by_text: dict[str, list[TokenTag]]) -> None:
        self.tags_by_text = tags_by_text

    def tag(self, text: str) -> list[TokenTag]:
        return self.tags_by_text[text]


class _FailingTransliterator:
    def transliterate_text(self, text, tags, language="hi") -> str:
        raise RuntimeError("transliteration backend failed")


class _RecordingTranslator:
    def __init__(self, output: str = "English translation") -> None:
        self.output = output
        self.calls: list[tuple[str, str, str]] = []

    def translate(self, text, source_language, target_language="eng_Latn") -> str:
        self.calls.append((text, source_language, target_language))
        return self.output


class _AcceptingSemanticPreserver:
    def validate(self, source, translated) -> TransformationValidation:
        return TransformationValidation(True, 0.99)


HINGLISH_TRANSLITERATION_FIXTURES = [
    ("protein kidney ke liye kharab hai", ["protein", "kidney"], "protein kidney के लिए खराब है"),
    ("haldi cancer ko 100% cure karti hai", ["cancer", "100", "cure"], "हल्दी cancer को 100% cure करती है"),
    ("doctor se bina pooche metformin 500mg band mat karo", ["doctor", "metformin", "500mg"], "doctor से बिना पूछे metformin 500mg बंद मत करो"),
    ("subah khali pet nimbu pani diabetes ko theek karta hai", ["diabetes"], "सुबह खाली पेट नींबू पानी diabetes को ठीक करता है"),
    ("kuch studies kehte hain ki green tea inflammation kam kar sakti hai", ["studies", "green", "tea", "inflammation"], "कुछ studies कहते हैं कि green tea inflammation कम कर सकती है"),
    ("ye miracle remedy turant share karo!!!", ["miracle", "remedy", "share"], "ये miracle remedy तुरंत share करो!!!"),
    ("vitamin D deficiency ke liye blood test karwana useful ho sakta hai", ["vitamin", "D", "deficiency", "blood", "test", "useful"], "vitamin D deficiency के लिए blood test करवाना useful हो सकता है"),
    ("garlic sabhi viruses ko khatam kar deta hai", ["garlic", "viruses"], "garlic सभी viruses को खत्म कर देता है"),
    ("pregnancy mein iron supplement doctor ki salah se lena chahiye", ["pregnancy", "iron", "supplement", "doctor"], "pregnancy में iron supplement doctor की सलाह से लेना चाहिए"),
    ("cold milk acidity ko permanently cure karta hai", ["cold", "milk", "acidity", "permanently", "cure"], "cold milk acidity को permanently cure करता है"),
]


class ContractTests(unittest.TestCase):
    def test_token_tag_preserves_source_offsets(self) -> None:
        tag = TokenTag("protein", 0, 7, "en", 0.99)
        self.assertEqual("protein", "protein kidney ke liye kharab hai"[tag.start:tag.end])

    def test_semantic_guard_rejects_empty_gloss_without_loading_model(self) -> None:
        result = SemanticPreserver().validate("हल्दी कैंसर का इलाज नहीं करती", "")
        self.assertFalse(result.accepted)
        self.assertEqual(result.confidence, 0.0)

    def test_devanagari_token_keeps_combining_marks_and_offsets(self) -> None:
        text = "हल्दी कैंसर का इलाज नहीं करती।"
        tokens = list(iter_tokens(text))
        self.assertEqual([token for token, _, _ in tokens], ["हल्दी", "कैंसर", "का", "इलाज", "नहीं", "करती", "।"])
        self.assertTrue(all(text[start:end] == token for token, start, end in tokens))

    def test_failed_hinglish_transliteration_never_reaches_translation(self) -> None:
        text = "kharab hai"
        translator = _RecordingTranslator()
        processor = MultilingualProcessor(
            MultilingualProcessorConfig(enable_linguistic_score=False),
            detector=_StaticDetector({text: [TokenTag("kharab", 0, 6, "hi_Latn", 0.99),
                                            TokenTag("hai", 7, 10, "hi_Latn", 0.99)]}),
            transliterator=_FailingTransliterator(),
            translator=translator,
            semantic_preserver=_AcceptingSemanticPreserver(),
        )

        claim = processor.process(text)

        self.assertIsNone(claim.canonical_indic_text)
        self.assertIsNone(claim.english_gloss)
        self.assertEqual(translator.calls, [])
        self.assertTrue(any(warning.startswith("Transliteration unavailable:") for warning in claim.warnings))

    def test_non_english_gloss_is_rejected_after_semantic_acceptance(self) -> None:
        source = "हल्दी कैंसर का इलाज नहीं करती"
        gloss = "हल्दी कैंसर का इलाज नहीं करती"
        processor = MultilingualProcessor(
            MultilingualProcessorConfig(enable_linguistic_score=False),
            detector=_StaticDetector({source: [TokenTag("हल्दी", 0, 5, "hi_Deva", 1.0)],
                                      gloss: [TokenTag("हल्दी", 0, 5, "hi_Deva", 1.0)]}),
            translator=_RecordingTranslator(gloss),
            semantic_preserver=_AcceptingSemanticPreserver(),
        )

        claim = processor.process(source)

        self.assertIsNone(claim.english_gloss)
        self.assertIsNotNone(claim.validation)
        self.assertFalse(claim.validation.accepted)
        self.assertIn("Derived gloss was rejected because it is not English.", claim.warnings)

    def test_hinglish_fixture_outputs_are_devanagari_and_preserve_protected_tokens(self) -> None:
        for source, protected, expected in HINGLISH_TRANSLITERATION_FIXTURES:
            with self.subTest(source=source):
                self.assertTrue(any("\u0900" <= character <= "\u097f" for character in expected))
                self.assertTrue(all(
                    QwenHinglishTransliterator._contains_protected_token(expected, token)
                    for token in protected
                ))

    def test_transliterator_cleans_wrappers_and_uses_case_insensitive_boundaries(self) -> None:
        output = QwenHinglishTransliterator._clean_generated_text(
            '```text\n  "protein kidney के लिए खराब है"  \n```'
        )
        self.assertEqual(output, "protein kidney के लिए खराब है")
        self.assertTrue(QwenHinglishTransliterator._contains_protected_token("Doctor से पूछें", "doctor"))
        self.assertFalse(QwenHinglishTransliterator._contains_protected_token("doctors से पूछें", "doctor"))

    def test_hindi_span_reassembly_never_sends_protected_tokens_to_generation(self) -> None:
        text = "doctor se bina pooche metformin 500mg band mat karo"
        tags = [
            TokenTag("doctor", 0, 6, "en", 0.99),
            TokenTag("se", 7, 9, "hi_Latn", 0.99),
            TokenTag("bina", 10, 14, "hi_Latn", 0.99),
            TokenTag("pooche", 15, 21, "hi_Latn", 0.99),
            TokenTag("metformin", 22, 31, "en", 0.99),
            TokenTag("500mg", 32, 37, "number_unit", 1.0),
            TokenTag("band", 38, 42, "hi_Latn", 0.99),
            TokenTag("mat", 43, 46, "hi_Latn", 0.99),
            TokenTag("karo", 47, 51, "hi_Latn", 0.99),
        ]
        spans: list[str] = []
        result = QwenHinglishTransliterator._reassemble_hindi_spans(
            text, tags, lambda span: spans.append(span) or {"se bina pooche": "से बिना पूछे", "band mat karo": "बंद मत करो"}[span]
        )
        self.assertEqual(spans, ["se bina pooche", "band mat karo"])
        self.assertEqual(result, "doctor से बिना पूछे metformin 500mg बंद मत करो")
        self.assertTrue(QwenHinglishTransliterator._contains_protected_token(result, "doctor"))
        self.assertTrue(QwenHinglishTransliterator._contains_protected_token(result, "metformin"))
        self.assertTrue(QwenHinglishTransliterator._contains_protected_token(result, "500mg"))

    def test_transliteration_prompt_contains_hinglish_examples(self) -> None:
        system_prompt = QwenHinglishTransliterator._messages("claim", ["doctor"])[0]["content"]
        self.assertIn("protein kidney के लिए खराब है", system_prompt)
        self.assertIn("doctor से बिना पूछे metformin 500mg बंद मत करो", system_prompt)
        self.assertIn("garlic सभी viruses को खत्म कर देता है", system_prompt)
        self.assertIn("सुबह खाली पेट नींबू पानी diabetes को ठीक करता है", system_prompt)


@unittest.skipUnless(os.getenv("RUN_REAL_TRANSLATION") == "1", "Set RUN_REAL_TRANSLATION=1 after model setup.")
class LiveModelAcceptanceTests(unittest.TestCase):
    def test_hinglish_token_lid(self) -> None:
        tags = TokenLanguageDetector(HinglishLIDClassifier()).tag("protein kidney ke liye kharab hai")
        self.assertEqual([tag.token for tag in tags], ["protein", "kidney", "ke", "liye", "kharab", "hai"])
        self.assertTrue(all(tag.label in {"en", "hi_Latn"} for tag in tags))

    def test_hindi_to_english_translation(self) -> None:
        translated = IndicTrans2Backend(local_files_only=False).translate("हल्दी कैंसर का इलाज नहीं करती।", "hin_Deva")
        print(f"Hindi: हल्दी कैंसर का इलाज नहीं करती।\nEnglish gloss: {translated}")
        self.assertTrue(translated.strip())


if __name__ == "__main__":
    unittest.main(verbosity=2)
