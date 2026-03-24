import re

# ─────────────────────────────────────────────────────────────────────────────
# TIER 1 — HIGH RISK (conspiracy, absolute cure claims, dangerous urgency)
# ─────────────────────────────────────────────────────────────────────────────
TIER1_HIGH = {
    "doctors don't want", "doctors hate", "they don't want you to know",
    "hidden truth", "big pharma", "government is hiding", "suppressed cure",
    "cures cancer", "cures diabetes", "cures all", "permanent cure", "100% cure",
    "must share immediately", "share before it's deleted",
    "forward to everyone", "going viral",
}

# ─────────────────────────────────────────────────────────────────────────────
# TIER 2 — MEDIUM RISK (sensational framing, WhatsApp culture, urgency)
# ─────────────────────────────────────────────────────────────────────────────
TIER2_MEDIUM = {
    "miracle", "magic remedy", "magic cure", "shocking", "unbelievable",
    "mind blowing", "ancient secret", "ancient remedy", "guaranteed",
    "zero side effects", "no side effects", "completely safe",
    "whatsapp", "forwarded message", "must share", "please share",
    "share with family", "viral remedy", "trending cure",
    "desi nuskha", "gharelu nuskha", "ayurvedic secret",
    "generations old remedy", "daadi ka nuskha",
    "urgent", "act now", "limited time", "breaking", "exclusive",
}

# ─────────────────────────────────────────────────────────────────────────────
# TIER 3 — LOW RISK (weak sensationalism, overused health buzzwords)
# ─────────────────────────────────────────────────────────────────────────────
TIER3_LOW = {
    "secret", "revealed", "exposed", "natural remedy", "home remedy",
    "try this", "works instantly", "proven", "scientifically proven",
    "detox", "cleanse", "flush toxins", "boosts immunity",
    "superfood", "wonder food", "empty stomach", "khali pet",
}

# ─────────────────────────────────────────────────────────────────────────────
# ABSOLUTE QUANTIFIERS
# Real medical text never makes absolute claims. These words signal
# misinformation regardless of the health topic.
# ─────────────────────────────────────────────────────────────────────────────
ABSOLUTE_QUANTIFIERS = {
    "always", "never", "completely", "totally", "instantly", "immediately",
    "permanently", "forever", "100 percent", "definitely", "certainly",
    "undoubtedly", "absolutely", "will cure", "will prevent", "will cause",
    "eliminates completely", "reverses completely",
}

# ─────────────────────────────────────────────────────────────────────────────
# HEDGING PHRASES
# Presence of hedging = credible framing → reduces style score.
# Real scientific writing always hedges causal claims.
# ─────────────────────────────────────────────────────────────────────────────
HEDGING_PHRASES = {
    "may", "might", "could", "suggests", "evidence suggests",
    "some studies", "research indicates", "according to",
    "in some cases", "potentially", "possibly", "appears to",
    "is associated with", "linked to", "preliminary evidence",
    "consult a doctor", "speak to a professional", "limited evidence",
    "further research needed", "not conclusive",
}

# ─────────────────────────────────────────────────────────────────────────────
# VAGUE AUTHORITY
# "Studies show", "doctors say" with no specifics = fake credibility signal.
# ─────────────────────────────────────────────────────────────────────────────
VAGUE_AUTHORITY = {
    "doctors say", "scientists say", "experts say", "researchers say",
    "studies show", "studies prove", "science says", "doctors confirm",
    "experts confirm", "scientists confirm", "research shows",
    "research proves", "doctors recommend", "experts recommend",
    "they say", "everyone knows",
}

# ─────────────────────────────────────────────────────────────────────────────
# PSEUDOSCIENCE TERMS
# Scientific-sounding language used to fake mechanism explanations.
# "microwave alters food dna" — "alters dna" fires here.
# ─────────────────────────────────────────────────────────────────────────────
PSEUDOSCIENCE_TERMS = {
    "alters dna", "destroys dna", "mutates dna", "affects dna", "damages dna",
    "electromagnetic", "pineal gland", "third eye", "alkaline body",
    "acidic blood", "oxygenate blood", "detoxify blood", "toxic blood",
    "negative energy", "positive energy", "vibration frequency",
    "activates genes", "rewires brain", "resets metabolism",
    "kills cancer cells", "starves cancer", "alkalizes body",
    "heavy metal detox", "removes toxins from blood",
}

# ─────────────────────────────────────────────────────────────────────────────
# CAUSAL OVERCLAIM PATTERN
# Regex: catches ANY "cures X", "causes X", "prevents X" bare assertion.
# Broader than TIER1 keywords — covers any disease/condition after the verb.
# ─────────────────────────────────────────────────────────────────────────────
_CAUSAL_VERBS = (
    r"cures?|causes?|prevents?|treats?|heals?|kills?|destroys?|"
    r"reverses?|eliminates?|boosts?|damages?|blocks?|melts?|whitens?|"
    r"cleans?|flushes?|purifies?|removes?|repairs?|restores?"
)
CAUSAL_OVERCLAIM_PATTERN = re.compile(
    rf"\b({_CAUSAL_VERBS})\b\s+\w+",
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────────────────────
# CAUSAL CHAIN PATTERN
# Two-hop causal claim — stronger misinformation signal than single hop.
# "microwave alters food dna causing cancer"
# "stress affects hormones leading to diabetes"
# ─────────────────────────────────────────────────────────────────────────────
CAUSAL_CHAIN_PATTERN = re.compile(
    r"\b\w+\s+(alters?|changes?|affects?|disrupts?|damages?|blocks?)\s+\w+"
    r"\s+(caus(?:es?|ing)|lead(?:s|ing)\s+to|result(?:s|ing)\s+in|trigger(?:s|ing))\s+\w+",
    re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────────────────
# LINGUISTIC SCORER
# ─────────────────────────────────────────────────────────────────────────────

class LinguisticScorer:
    """
    Rule-based sensationalism and misinformation style scorer.

    Measures 9 independent signals across 4 dimensions:

    Dimension 1 — Presentation signals:
        uppercase, punctuation, tiered keywords

    Dimension 2 — Claim structure signals:
        absolute quantifiers, hedging absence, causal overclaim,
        causal chain (two-hop)

    Dimension 3 — Credibility signals:
        vague authority, pseudoscience terms

    Dimension 4 — Structural signals:
        length pattern

    Note on pretrained models:
        Health-specific models (PUBHEALTH/health_fact) require claim+evidence
        pairs and are not suitable for standalone sentence scoring.
        Clickbait models (trained on headlines) actively hurt scores on calm
        misinformation like "microwave alters food dna causing cancer".
        Rule-based scoring is more reliable for this domain and input format.
    """

    def __init__(self):
        self._exclamation_pattern = re.compile(r'!{2,}')
        self._question_pattern    = re.compile(r'\?{2,}')
        self._caps_word_pattern   = re.compile(r'\b[A-Z]{3,}\b')

    # ─────────────────────────────────────────────────────────────────────────
    # DIMENSION 1 — PRESENTATION SIGNALS
    # ─────────────────────────────────────────────────────────────────────────

    def _score_uppercase(self, text):
        """Uppercase ratio + ALL-CAPS word count."""
        score = 0.0
        upper_count = sum(1 for c in text if c.isupper())
        total_chars = len(text.replace(" ", ""))

        if total_chars > 0:
            ratio = upper_count / total_chars
            if ratio > 0.50:
                score += 0.35
            elif ratio > 0.15:
                score += 0.20

        caps_words = self._caps_word_pattern.findall(text)
        if len(caps_words) >= 3:
            score += 0.20
        elif len(caps_words) >= 1:
            score += 0.10

        return min(score, 0.40)

    def _score_punctuation(self, text):
        """Repeated punctuation — !!! ??? etc."""
        exclamations = len(self._exclamation_pattern.findall(text))
        questions    = len(self._question_pattern.findall(text))
        total        = exclamations + questions

        if total >= 3:   return 0.30
        if total == 2:   return 0.20
        if total == 1:   return 0.10
        return 0.0

    def _score_tiered_keywords(self, text):
        """Tiered keyword matching across conspiracy / sensational / mild lists."""
        text_lower = text.lower()
        t1 = sum(1 for kw in TIER1_HIGH   if kw in text_lower)
        t2 = sum(1 for kw in TIER2_MEDIUM if kw in text_lower)
        t3 = sum(1 for kw in TIER3_LOW    if kw in text_lower)

        score  = min(t1 * 0.35, 0.35)
        score += min(t2 * 0.20, 0.30)
        score += min(t3 * 0.10, 0.20)
        return min(score, 0.60)

    # ─────────────────────────────────────────────────────────────────────────
    # DIMENSION 2 — CLAIM STRUCTURE SIGNALS
    # ─────────────────────────────────────────────────────────────────────────

    def _score_absolute_quantifiers(self, text):
        """
        Absolute language that real medical writing never uses.
        "This completely cures diabetes" — "completely" fires.
        +0.15 per hit, capped at 0.30.
        """
        text_lower = text.lower()
        hits = sum(1 for q in ABSOLUTE_QUANTIFIERS if q in text_lower)
        return min(hits * 0.15, 0.30)

    def _score_hedging_absence(self, text):
        """
        Causal claim without hedging = suspicious.
        Causal claim with hedging = credible (negative contribution).

        Returns:
            +0.20 : causal verb present, no hedging → bare assertion
            -0.10 : causal verb present, hedging found → credible framing
             0.0  : no causal verb → neutral
        """
        text_lower  = text.lower()
        has_causal  = bool(CAUSAL_OVERCLAIM_PATTERN.search(text))
        has_hedging = any(h in text_lower for h in HEDGING_PHRASES)

        if not has_causal:
            return 0.0
        return -0.10 if has_hedging else 0.20

    def _score_causal_overclaim(self, text):
        """
        Regex catches any bare "cures X", "causes X", "melts X" etc.
        Covers everything in TIER1 plus all other disease/condition combos.
        +0.10 per match, capped at 0.20.
        """
        matches = CAUSAL_OVERCLAIM_PATTERN.findall(text)
        return min(len(matches) * 0.10, 0.20)

    def _score_causal_chain(self, text):
        """
        Two-hop causal claims — stronger signal than single hop.
        "microwave alters food dna causing cancer" → fires.
        "stress affects hormones leading to diabetes" → fires.
        +0.20 per match, capped at 0.20.
        """
        matches = CAUSAL_CHAIN_PATTERN.findall(text)
        return min(len(matches) * 0.20, 0.20)

    # ─────────────────────────────────────────────────────────────────────────
    # DIMENSION 3 — CREDIBILITY SIGNALS
    # ─────────────────────────────────────────────────────────────────────────

    def _score_vague_authority(self, text):
        """
        Unnamed authority with no specifics — fake credibility.
        "Studies show", "doctors say" → +0.15 each, capped at 0.20.
        """
        text_lower = text.lower()
        hits = sum(1 for v in VAGUE_AUTHORITY if v in text_lower)
        return min(hits * 0.15, 0.20)

    def _score_pseudoscience(self, text):
        """
        Scientific-sounding mechanism language used incorrectly.
        "alters dna", "alkalizes body", "pineal gland" → +0.20 each, capped 0.30.
        Catches: "microwave alters food dna causing cancer" via "alters dna".
        """
        text_lower = text.lower()
        hits = sum(1 for p in PSEUDOSCIENCE_TERMS if p in text_lower)
        return min(hits * 0.20, 0.30)

    # ─────────────────────────────────────────────────────────────────────────
    # DIMENSION 4 — STRUCTURAL SIGNALS
    # ─────────────────────────────────────────────────────────────────────────

    def _score_length_pattern(self, text):
        """
        Very short claims are often stripped WhatsApp forwards.
        Very long claims may be spam-padded misinformation.
        Weak signal — only 0.05-0.10.
        """
        wc = len(text.split())
        if wc <= 5:  return 0.10
        if wc >= 80: return 0.05
        return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # MASTER SCORER
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_all_signals(self, text):
        """Runs all 9 signals and returns dict of values."""
        return {
            "uppercase":     self._score_uppercase(text),
            "punctuation":   self._score_punctuation(text),
            "keywords":      self._score_tiered_keywords(text),
            "absolute":      self._score_absolute_quantifiers(text),
            "hedging":       self._score_hedging_absence(text),
            "causal":        self._score_causal_overclaim(text),
            "causal_chain":  self._score_causal_chain(text),
            "authority":     self._score_vague_authority(text),
            "pseudoscience": self._score_pseudoscience(text),
            "length":        self._score_length_pattern(text),
        }

    def calculate_score(self, text):
        """
        Final style score: 0.0 (credible/neutral) → 1.0 (highly sensational).

        Args:
            text (str): raw sentence — pass original before cleaning so
                        caps and punctuation signals are preserved.

        Returns:
            float
        """
        if not text or not text.strip():
            return 0.0

        signals = self._compute_all_signals(text)
        total   = sum(signals.values())
        return round(max(0.0, min(total, 1.0)), 4)

    def calculate_score_detailed(self, text):
        """
        Same as calculate_score but returns full per-signal breakdown.
        Use this in pipeline for richer logging.

        Returns:
            dict: {"score": float, "breakdown": {signal: value, ...}}
        """
        if not text or not text.strip():
            return {"score": 0.0, "breakdown": {}}

        signals = self._compute_all_signals(text)
        total   = round(max(0.0, min(sum(signals.values()), 1.0)), 4)

        return {
            "score":     total,
            "breakdown": {k: round(v, 4) for k, v in signals.items()},
        }


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scorer = LinguisticScorer()

    test_cases = [
        # Expected LOW
        ("LOW",    "Evidence suggests that vitamin D may support immune function."),
        ("LOW",    "Exercise regularly to maintain a healthy weight."),
        ("LOW",    "Some studies indicate that green tea is associated with reduced inflammation."),
        # Expected MEDIUM — unhedged causal, calm language
        ("MEDIUM", "Tea bags cause brain fog."),
        ("MEDIUM", "Laptops on lap cause infertility."),
        ("MEDIUM", "Cold milk cures acidity."),
        ("MEDIUM", "Doctors say onions clean viruses from the air."),
        # Expected HIGH — pseudoscience + causal chain
        ("HIGH",   "Microwave alters food dna causing cancer."),
        ("HIGH",   "Stress affects hormones leading to diabetes."),
        ("HIGH",   "SHOCKING MIRACLE!!! Doctors don't want you to know this secret cure!!!"),
        ("HIGH",   "TURMERIC CURES CANCER COMPLETELY!!! Must share with family!!!"),
        ("HIGH",   "This ancient remedy instantly cures diabetes permanently — guaranteed!"),
        ("HIGH",   "Scientists confirm this completely destroys all viruses immediately."),
        # Hinglish
        ("HIGH",   "Subah khali pet ye gharelu nuskha try karo — 100% guaranteed results!"),
        # Hedged — should score lower than unhedged equivalent
        ("LOW",    "Some studies suggest turmeric may have anti-inflammatory properties."),
    ]

    print(f"\n{'Expected':<8} {'Score':<7}  Text")
    print("─" * 80)
    for expected, text in test_cases:
        score   = scorer.calculate_score(text)
        display = (text[:65] + "...") if len(text) > 68 else text
        print(f"{expected:<8} {score:<7}  {display}")

    print("\n--- Detailed breakdown: 'Microwave alters food dna causing cancer.' ---")
    result = scorer.calculate_score_detailed("Microwave alters food dna causing cancer.")
    for k, v in result['breakdown'].items():
        bar = "█" * int(v * 20) if v > 0 else "░" * int(abs(v) * 20)
        print(f"  {k:<16} : {v:>6}  {bar}")
    print(f"  {'TOTAL':<16} : {result['score']:>6}")