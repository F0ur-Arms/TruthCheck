# import json
# import os

# class FactVerifier:
#     def __init__(self, kb_path="data/verified_facts.json"):
#         # Look for the data file relative to the project root
#         self.kb_path = kb_path
#         self.kb = self._load_kb()

#     def _load_kb(self):
#         if not os.path.exists(self.kb_path):
#             # Fallback for testing if file doesn't exist yet
#             return [
#                 {
#                     "claim_subject": "warm water",
#                     "verdict": "TRUE",
#                     "scientific_truth": "Warm water aids digestion and blood circulation.",
#                     "risk_level": "Low"
#                 },
#                 {
#                     "claim_subject": "papaya seeds",
#                     "verdict": "FALSE",
#                     "scientific_truth": "No clinical evidence supports papaya seeds as a safe or effective abortion method; it can be toxic.",
#                     "risk_level": "High"
#                 }
#             ]
#         with open(self.kb_path, 'r', encoding='utf-8') as f:
#             return json.load(f)

#     def verify(self, extracted_triple):
#         """
#         Matches triple against the Knowledge Base.
#         """
#         subj = extracted_triple['subject'].lower()
#         obj = extracted_triple['object'].lower()
        
#         # Check for matches in our KB
#         for entry in self.kb:
#             keyword = entry["claim_subject"].lower()
#             # If the keyword exists in either the subject or object of the triple
#             if keyword in subj or keyword in obj:
#                 return {
#                     "match_found": True,
#                     "verdict": entry["verdict"],
#                     "truth": entry["scientific_truth"],
#                     "risk_level": entry["risk_level"]
#                 }
        
#         return {
#             "match_found": False,
#             "verdict": "UNVERIFIED",
#             "truth": "This claim is not in our verified medical database yet.",
#             "risk_level": "Medium"
#         }

# if __name__ == "__main__":
#     # Test Objective 4
#     verifier = FactVerifier()
#     test_triple = {"subject": "drinking warm water", "relation": "improves", "object": "digestion"}
    
#     result = verifier.verify(test_triple)
#     print(f"Triple: {test_triple['subject']} -> {test_triple['relation']}")
#     print(f"Verdict: {result['verdict']}")
#     print(f"Evidence: {result['truth']}")

import json
import os
import re

# ---------------------------------------------------------------------------
# FUZZY MATCHING HELPERS
# We are NOT using any external fuzzy library (like fuzzywuzzy / rapidfuzz)
# to keep dependencies minimal. Instead we implement two simple but effective
# techniques that cover the real variation in health claim language:
#
#   1. Token Overlap (Jaccard-style)
#      Splits both strings into word sets and measures overlap.
#      Handles word order variation and partial phrases well.
#      e.g. "warm water drinking" vs "drinking warm water" → high overlap
#
#   2. Substring containment
#      Checks if the KB keyword is fully contained inside the claim text.
#      Handles cases like "warm water" inside "warm lemon water".
#
# A match is accepted if EITHER method crosses its threshold.
# ---------------------------------------------------------------------------

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def token_overlap_score(text_a, text_b):
    """
    Jaccard-style token overlap between two strings.
    Returns a float between 0.0 and 1.0.

    Example:
        "warm water digestion" vs "warm water"
        intersection = {"warm", "water"} = 2
        union        = {"warm", "water", "digestion"} = 3
        score        = 2/3 = 0.67
    """
    tokens_a = set(normalize(text_a).split())
    tokens_b = set(normalize(text_b).split())

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union        = tokens_a | tokens_b

    return len(intersection) / len(union)


def is_substring_match(keyword, text, threshold=0.0):
    """
    Checks if the normalized keyword is fully contained within the normalized text.
    e.g. keyword="warm water", text="drinking warm water on empty stomach" → True
    """
    return normalize(keyword) in normalize(text)


def fuzzy_match_score(keyword, claim_text):
    """
    Master matching function. Combines token overlap + substring containment
    into a single confidence score (0.0 to 1.0).

    Strategy:
    - If keyword is a substring of claim text → strong signal, score = 0.85
    - Token overlap above threshold → use that score
    - Take the MAX of both methods
    """
    overlap  = token_overlap_score(keyword, claim_text)
    substr   = is_substring_match(keyword, claim_text)

    substr_score = 0.85 if substr else 0.0

    return max(overlap, substr_score)

STRONG_MATCH_THRESHOLD = 0.65
WEAK_MATCH_THRESHOLD   = 0.35


class FactVerifier:
    def __init__(self, kb_path="TruthCheck/data/verified_facts.json"):
        self.kb_path = kb_path
        self.kb = self._load_kb()

    def _load_kb(self):

        if not os.path.exists(self.kb_path):
            print(f"[FactVerifier] Warning: KB not found at {self.kb_path}. Using fallback KB.")
            return [
                {
                    "claim_subject": "warm water",
                    "verdict": "TRUE",
                    "scientific_truth": "Warm water aids digestion and blood circulation.",
                    "risk_level": "Low"
                },
                {
                    "claim_subject": "papaya seeds",
                    "verdict": "FALSE",
                    "scientific_truth": "No clinical evidence supports papaya seeds as a safe abortion method; it can be toxic.",
                    "risk_level": "High"
                },
                {
                    "claim_subject": "turmeric",
                    "verdict": "MIXED",
                    "scientific_truth": "Turmeric has anti-inflammatory properties but does not cure cancer. Overconsumption can cause liver issues.",
                    "risk_level": "Medium"
                },
                {
                    "claim_subject": "lemon water",
                    "verdict": "TRUE",
                    "scientific_truth": "Lemon water supports hydration and provides vitamin C but has no proven detox effects.",
                    "risk_level": "Low"
                },
            ]

        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                kb = json.load(f)
            print(f"[FactVerifier] Loaded {len(kb)} entries from {self.kb_path}")
            return kb
        except Exception as e:
            print(f"[FactVerifier] Error loading KB: {e}. Using fallback KB.")
            return []

    def _build_claim_text(self, triple):
        """
        Combines subject + object into a single string for matching.
        We match against both because the health entity could appear in either.

        e.g. triple = {subject: "drinking warm water", object: "digestion"}
             claim_text = "drinking warm water digestion"
        """
        subj = triple.get('subject', '')
        obj  = triple.get('object',  '')
        return f"{subj} {obj}".strip()

    def verify(self, extracted_triple):
        """
        Matches an extracted triple against the knowledge base using fuzzy matching.

        Returns a dict with:
            match_found   (bool)
            verdict       (str)  : TRUE / FALSE / MIXED / UNVERIFIED
            truth         (str)  : explanation
            risk_level    (str)  : Low / Medium / High
            match_score   (float): confidence of the match (0.0 to 1.0)
            matched_entry (str)  : which KB keyword was matched (for transparency)
        """
        claim_text = self._build_claim_text(extracted_triple)

        best_score = 0.0
        best_entry = None

        # Score every KB entry and keep the best match
        for entry in self.kb:
            keyword = entry.get("claim_subject", "")
            if not keyword:
                continue

            score = fuzzy_match_score(keyword, claim_text)

            if score > best_score:
                best_score = score
                best_entry = entry

        # ---------------------------------------------------------------------------
        # STRONG MATCH — high confidence, use KB verdict directly
        # ---------------------------------------------------------------------------
        if best_score >= STRONG_MATCH_THRESHOLD and best_entry:
            return {
                "match_found":   True,
                "verdict":       best_entry["verdict"],
                "truth":         best_entry["scientific_truth"],
                "risk_level":    best_entry["risk_level"],
                "match_score":   round(best_score, 2),
                "matched_entry": best_entry["claim_subject"],
                "match_type":    "STRONG"
            }

        # ---------------------------------------------------------------------------
        # WEAK MATCH — partial match found, return verdict but flag it
        # Adds a caveat to the truth explanation so downstream knows confidence
        # is lower. Does not override the verdict — RiskEngine handles weighting.
        # ---------------------------------------------------------------------------
        if best_score >= WEAK_MATCH_THRESHOLD and best_entry:
            return {
                "match_found":   True,
                "verdict":       best_entry["verdict"],
                "truth":         f"[Partial Match — verify manually] {best_entry['scientific_truth']}",
                "risk_level":    best_entry["risk_level"],
                "match_score":   round(best_score, 2),
                "matched_entry": best_entry["claim_subject"],
                "match_type":    "WEAK"
            }

        # ---------------------------------------------------------------------------
        # NO MATCH — claim not in KB
        # UNVERIFIED score in RiskEngine is 0.25 (mild caution, not a penalty)
        # ---------------------------------------------------------------------------
        return {
            "match_found":   False,
            "verdict":       "UNVERIFIED",
            "truth":         "This claim could not be matched against our verified medical database.",
            "risk_level":    "Medium",
            "match_score":   round(best_score, 2),
            "matched_entry": None,
            "match_type":    "NONE"
        }


# ---------------------------------------------------------------------------
# Standalone testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    verifier = FactVerifier()

    test_triples = [
        # Exact match
        {"subject": "warm water",          "relation": "improves",  "object": "digestion"},
        # Variation — old code would miss this
        {"subject": "hot water",            "relation": "aids",      "object": "digestion"},
        # Variation — old code would miss this
        {"subject": "drinking warm water",  "relation": "improves",  "object": "gut health"},
        # Partial match
        {"subject": "warm lemon water",     "relation": "helps",     "object": "metabolism"},
        # Dangerous claim
        {"subject": "papaya seeds",         "relation": "causes",    "object": "abortion"},
        # Mixed verdict
        {"subject": "turmeric",             "relation": "cures",     "object": "cancer"},
        # No match — should return UNVERIFIED
        {"subject": "cow urine",            "relation": "treats",    "object": "diabetes"},
    ]

    print(f"\n{'Claim':<45} | {'Verdict':<12} | {'Score':<6} | {'Type':<8} | Matched")
    print("-" * 95)

    for triple in test_triples:
        claim_str = f"{triple['subject']} → {triple['object']}"
        result = verifier.verify(triple)
        print(
            f"{claim_str:<45} | "
            f"{result['verdict']:<12} | "
            f"{result['match_score']:<6} | "
            f"{result['match_type']:<8} | "
            f"{result['matched_entry']}"
        )