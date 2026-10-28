import unittest

from src.sentence_selector import select_sentences, split_sentences


def overlap_scorer(pairs):
    return [len(set(c.lower().split()) & set(s.lower().split())) for c, s in pairs]


class SentenceSelectorTests(unittest.TestCase):
    def test_split(self):
        self.assertEqual(split_sentences("A b. C d? E f."), ["A b.", "C d?", "E f."])

    def test_keeps_top_k_in_original_order(self):
        text = "Vitamin C supports immunity. Rain fell in May. Zinc helps immunity too. Cats sleep a lot."
        out = select_sentences("vitamin c immunity zinc", text, overlap_scorer, k=2)
        self.assertEqual(out, "Vitamin C supports immunity. Zinc helps immunity too.")

    def test_short_text_unchanged(self):
        self.assertEqual(select_sentences("x", "One sentence only.", overlap_scorer, k=2), "One sentence only.")


if __name__ == "__main__":
    unittest.main()
