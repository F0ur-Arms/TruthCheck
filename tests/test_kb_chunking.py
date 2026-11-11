import unittest

from src.RAGpipeline.kb_chunking import MAX_WORDS, file_title, line_passages, window_passages

HEADER = ["SOURCE: CDC", "TITLE: Dengue", "URL: https://example.org/dengue", "─" * 60, ""]
SENTENCE = "Dengue spreads to people through the bite of an infected mosquito in warm wet areas."


class KbChunkingTests(unittest.TestCase):
    def test_lines_mode_keeps_one_passage_per_line(self):
        lines = HEADER + ["What is dengue?", SENTENCE, "- Reported cases: 1"]
        self.assertEqual(line_passages(lines), [SENTENCE])

    def test_window_groups_short_lines_and_keeps_the_heading_with_its_text(self):
        lines = HEADER + ["What is dengue?"] + [SENTENCE] * 3 + ["- High fever", "- Rash"]
        passages = window_passages(lines, file_title(lines))
        self.assertEqual(len(passages), 1)
        self.assertTrue(passages[0].startswith("Dengue: What is dengue? Dengue spreads"))
        self.assertIn("High fever Rash", passages[0])

    def test_window_starts_a_new_passage_at_a_heading_once_text_is_long_enough(self):
        lines = HEADER + ["Symptoms"] + [SENTENCE] * 4 + ["Treatment"] + [SENTENCE] * 4
        passages = window_passages(lines, "Dengue")
        self.assertEqual(len(passages), 2)
        self.assertTrue(passages[1].startswith("Dengue: Treatment"))

    def test_window_passages_stay_bounded(self):
        long_line = " ".join(["word"] * (3 * MAX_WORDS))
        passages = window_passages(HEADER + [long_line, SENTENCE * 20], "Dengue")
        self.assertTrue(all(len(p.split()) <= MAX_WORDS + 10 for p in passages))

    def test_no_header_text_or_fragments_become_passages(self):
        passages = window_passages(HEADER + ["- 1", SENTENCE], file_title(HEADER))
        self.assertFalse(any(p.startswith(("SOURCE", "URL")) for p in passages))
        self.assertTrue(all(len(p.split()) >= 5 for p in passages))


if __name__ == "__main__":
    unittest.main()
