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
from .semantic_preservation import SemanticPreserver
from .scripts import iter_tokens
from .token_lang_detector import HinglishLIDClassifier, TokenLanguageDetector
from .types import TokenTag


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
