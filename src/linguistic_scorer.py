# import re

# class LinguisticScorer:
#     def __init__(self):
#         # Keywords commonly found in Indian health misinformation and viral forwards
#         self.clickbait_triggers = [
#             "miracle", "secret", "shocking", "must share", 
#             "whatsapp", "doctors don't want", "guaranteed",
#             "forwarded", "urgent", "warning", "magic remedy",
#             "100%", "hidden truth"
#         ]

#     def calculate_score(self, text):
#         """
#         Analyzes text style to return a risk score between 0.0 and 1.0.
#         """
#         score = 0.0
#         if not text:
#             return score

#         # 1. SHOUTING CHECK (Uppercase Ratio)
#         # If more than 25% of the text is uppercase, it's likely sensationalist.
#         upper_count = sum(1 for c in text if c.isupper())
#         total_chars = len(text.replace(" ", ""))
#         if total_chars > 0:
#             upper_ratio = upper_count / total_chars
#             if upper_ratio > 0.3:
#                 score += 0.4

#         # 2. SENSATIONAL PUNCTUATION (Repeated Marks)
#         # Scientific text rarely uses !!! or ???
#         if re.search(r'!{2,}', text) or re.search(r'\?{2,}', text):
#             score += 0.3

#         # 3. CLICKBAIT KEYWORD MATCHING
#         text_lower = text.lower()
#         found_triggers = [word for word in self.clickbait_triggers if word in text_lower]
#         if found_triggers:
#             # Add 0.1 for each trigger found, maxing out at 0.3
#             score += min(len(found_triggers) * 0.1, 0.3)

#         return round(min(score, 1.0), 2)

# if __name__ == "__main__":
#     scorer = LinguisticScorer()
    
#     # Test cases
#     samples = [
#         "Drinking warm water improves digestion.",
#         "SHOCKING MIRACLE REMEDY!!! TURMERIC CURES CANCER!!! MUST SHARE!!!",
#         "This is a secret doctors don't want you to know about obesity."
#     ]

#     print(f"{'Sample Text':<40} | {'Risk Score'}")
#     print("-" * 55)
#     for s in samples:
#         score = scorer.calculate_score(s)
#         print(f"{s[:38]:<40} | {score}")

import re


# ---------------------------------------------------------------------------
# CLICKBAIT & SENSATIONALISM TRIGGERS
#
# Organized into weighted tiers instead of a flat list.
# Tier 1 (HIGH, +0.35): Strong misinformation signals — rarely appear in
#         legitimate health content. e.g. "doctors don't want you to know"
# Tier 2 (MEDIUM, +0.20): Common in viral forwards and WhatsApp health myths.
#         Not always misinformation but strong co-occurring signals.
# Tier 3 (LOW, +0.10): Weak signals — appear in both real and fake content.
#         Only meaningful when combined with other signals.
#
# Indian-context terms are intentionally included since the pipeline
# targets Indian health misinformation specifically.
# ---------------------------------------------------------------------------

TIER1_HIGH = {
    # Conspiracy / anti-establishment
    "doctors don't want",
    "doctors hate",
    "they don't want you to know",
    "hidden truth",
    "big pharma",
    "government is hiding",
    "suppressed cure",
    # Absolute cure claims
    "cures cancer",
    "cures diabetes",
    "cures all",
    "permanent cure",
    "100% cure",
    # Dangerous urgency
    "must share immediately",
    "share before it's deleted",
    "forward to everyone",
    "going viral",
}

TIER2_MEDIUM = {
    # Sensational framing
    "miracle", "magic remedy", "magic cure",
    "shocking", "unbelievable", "mind blowing",
    "ancient secret", "ancient remedy",
    "guaranteed", "100%", "zero side effects",
    "no side effects", "completely safe",
    # WhatsApp / forward culture signals
    "whatsapp", "forwarded message", "must share",
    "please share", "share with family",
    "viral remedy", "trending cure",
    # Indian-context terms
    "desi nuskha", "gharelu nuskha", "ayurvedic secret",
    "generations old remedy", "daadi ka nuskha",
    # Urgency without conspiracy
    "urgent", "warning", "act now", "limited time",
    "breaking", "exclusive",
}

TIER3_LOW = {
    # Weak sensationalism
    "secret", "revealed", "exposed",
    "natural remedy", "home remedy",
    "try this", "works instantly",
    "proven", "scientifically proven",   # often misused in misinformation
    "detox", "cleanse", "flush toxins",
    "boosts immunity",                    # overused claim post-COVID
    "superfood", "wonder food",
    "empty stomach",                      # common in Indian health forwards
    "khali pet",                          # Hinglish version of empty stomach
}


class LinguisticScorer:
    def __init__(self):
        # Pre-compile regex patterns for performance
        # These catch repeated punctuation like !!! or ???
        self._exclamation_pattern = re.compile(r'!{2,}')
        self._question_pattern    = re.compile(r'\?{2,}')
        # ALL CAPS words (3+ chars to avoid abbreviations like "OK", "BP")
        self._caps_word_pattern   = re.compile(r'\b[A-Z]{3,}\b')

    # ---------------------------------------------------------------------------
    # INDIVIDUAL SIGNAL SCORERS
    # Each returns a float contribution to the final score.
    # Kept separate so they're easy to test, tune, or disable individually.
    # ---------------------------------------------------------------------------

    def _score_uppercase(self, text):
        """
        Detects shouting / emphasis via uppercase ratio.

        Old code: threshold was 0.30 (30% of ALL chars uppercase)
        Problem:  A single word like "SHOCKING" in a normal sentence
                  doesn't move the needle — you'd need most of the text
                  to be caps before it triggered.

        Fix 1: Lower ratio threshold to 0.15
        Fix 2: Also count ALL-CAPS words directly (catches "SHOCKING CURE!!!"
               even in otherwise lowercase text)
        """
        score = 0.0

        # Method 1: Overall uppercase character ratio
        upper_count = sum(1 for c in text if c.isupper())
        total_chars = len(text.replace(" ", ""))

        if total_chars > 0:
            upper_ratio = upper_count / total_chars
            if upper_ratio > 0.50:
                score += 0.35    # Majority caps — very strong signal
            elif upper_ratio > 0.15:
                score += 0.20    # Partial caps — moderate signal

        # Method 2: Count individual ALL-CAPS words
        caps_words = self._caps_word_pattern.findall(text)
        if len(caps_words) >= 3:
            score += 0.20        # Multiple caps words — strong signal
        elif len(caps_words) >= 1:
            score += 0.10        # Single caps word — weak signal

        return min(score, 0.40)  # Cap this component at 0.40

    def _score_punctuation(self, text):
        """
        Detects sensational punctuation — !!! ??? etc.
        Scientific / factual text virtually never uses these.

        Old code gave flat 0.3 for any repeated mark.
        New: scaled by count — more occurrences = stronger signal.
        """
        score = 0.0

        exclamations = len(self._exclamation_pattern.findall(text))
        questions    = len(self._question_pattern.findall(text))

        total_marks = exclamations + questions

        if total_marks >= 3:
            score += 0.30
        elif total_marks == 2:
            score += 0.20
        elif total_marks == 1:
            score += 0.10

        return score

    def _score_clickbait_keywords(self, text):
        """
        Tiered keyword matching against TIER1/TIER2/TIER3 lists.

        Old code: flat list of 13 words, all weighted equally at 0.1 each.
        New: 3 tiers with different weights, much larger vocabulary,
             includes Indian-context and Hinglish terms.

        Scoring:
        - Each Tier1 match: +0.35 (capped at 0.35 total from this tier)
        - Each Tier2 match: +0.20 (capped at 0.30 total from this tier)
        - Each Tier3 match: +0.10 (capped at 0.20 total from this tier)
        """
        text_lower = text.lower()
        score = 0.0

        tier1_hits = [kw for kw in TIER1_HIGH   if kw in text_lower]
        tier2_hits = [kw for kw in TIER2_MEDIUM if kw in text_lower]
        tier3_hits = [kw for kw in TIER3_LOW    if kw in text_lower]

        score += min(len(tier1_hits) * 0.35, 0.35)
        score += min(len(tier2_hits) * 0.20, 0.30)
        score += min(len(tier3_hits) * 0.10, 0.20)

        return min(score, 0.60)  # Cap keyword component at 0.60

    def _score_length_pattern(self, text):
        """
        NEW signal — was completely missing before.

        Very short health claims (under 6 words) are often from WhatsApp
        forwards stripped of context: "Turmeric cures cancer. Share!!"
        Extremely long ones can be spam-padded misinformation.

        This is a weak signal on its own — only adds 0.05-0.10.
        """
        word_count = len(text.split())

        if word_count <= 5:
            return 0.10   # suspiciously short health claim
        if word_count >= 80:
            return 0.05   # unusually long — possible padding

        return 0.0

    # ---------------------------------------------------------------------------
    # MASTER SCORER
    # ---------------------------------------------------------------------------

    def calculate_score(self, text):
        """
        Combines all signals into a final risk score between 0.0 and 1.0.

        Also returns a breakdown dict so RiskEngine / report can show
        which signals contributed — useful for explaining verdicts.

        Args:
            text (str): Raw input sentence (before cleaning — we want
                        original caps, punctuation for style analysis)

        Returns:
            float: risk score 0.0 to 1.0
        """
        if not text or not text.strip():
            return 0.0

        upper_score    = self._score_uppercase(text)
        punct_score    = self._score_punctuation(text)
        keyword_score  = self._score_clickbait_keywords(text)
        length_score   = self._score_length_pattern(text)

        total = upper_score + punct_score + keyword_score + length_score

        return round(min(total, 1.0), 2)

    def calculate_score_detailed(self, text):
        """
        Same as calculate_score but returns full breakdown.
        Use this in main pipeline if you want per-signal visibility in reports.
        """
        if not text or not text.strip():
            return {"score": 0.0, "breakdown": {}}

        upper_score   = self._score_uppercase(text)
        punct_score   = self._score_punctuation(text)
        keyword_score = self._score_clickbait_keywords(text)
        length_score  = self._score_length_pattern(text)

        total = round(min(upper_score + punct_score + keyword_score + length_score, 1.0), 2)

        return {
            "score": total,
            "breakdown": {
                "uppercase_signal":  round(upper_score,   2),
                "punctuation_signal": round(punct_score,  2),
                "keyword_signal":    round(keyword_score, 2),
                "length_signal":     round(length_score,  2),
            }
        }


# ---------------------------------------------------------------------------
# Standalone testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    scorer = LinguisticScorer()

    test_cases = [
        # Should be LOW
        "Drinking warm water on an empty stomach improves digestion.",
        "Exercise regularly to maintain a healthy weight.",
        # Should be MEDIUM
        "This natural remedy is a proven detox cleanse for your body.",
        "Boosts immunity naturally with this ancient remedy.",
        # Should be HIGH
        "SHOCKING MIRACLE!!! Doctors don't want you to know this secret cure!!!",
        "TURMERIC CURES CANCER COMPLETELY!!! Must share with family!!!",
        # Indian context
        "Subah khali pet ye gharelu nuskha try karo — 100% guaranteed results!",
        # Edge cases
        "",                           # empty
        "OK",                         # too short
        "Share before deleted!!!",    # short + urgent
    ]

    print(f"\n{'Text':<60} | Score  | Breakdown")
    print("-" * 110)

    for text in test_cases:
        result = scorer.calculate_score_detailed(text)
        breakdown_str = " | ".join(
            f"{k.replace('_signal','').replace('_',' ')}={v}"
            for k, v in result["breakdown"].items()
        ) if result["breakdown"] else "—"
        display = (text[:57] + "...") if len(text) > 60 else text
        print(f"{display:<60} | {result['score']:<6} | {breakdown_str}")