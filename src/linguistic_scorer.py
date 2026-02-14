import re

class LinguisticScorer:
    def __init__(self):
        # Keywords commonly found in Indian health misinformation and viral forwards
        self.clickbait_triggers = [
            "miracle", "secret", "shocking", "must share", 
            "whatsapp", "doctors don't want", "guaranteed",
            "forwarded", "urgent", "warning", "magic remedy",
            "100%", "hidden truth"
        ]

    def calculate_score(self, text):
        """
        Analyzes text style to return a risk score between 0.0 and 1.0.
        """
        score = 0.0
        if not text:
            return score

        # 1. SHOUTING CHECK (Uppercase Ratio)
        # If more than 25% of the text is uppercase, it's likely sensationalist.
        upper_count = sum(1 for c in text if c.isupper())
        total_chars = len(text.replace(" ", ""))
        if total_chars > 0:
            upper_ratio = upper_count / total_chars
            if upper_ratio > 0.3:
                score += 0.4

        # 2. SENSATIONAL PUNCTUATION (Repeated Marks)
        # Scientific text rarely uses !!! or ???
        if re.search(r'!{2,}', text) or re.search(r'\?{2,}', text):
            score += 0.3

        # 3. CLICKBAIT KEYWORD MATCHING
        text_lower = text.lower()
        found_triggers = [word for word in self.clickbait_triggers if word in text_lower]
        if found_triggers:
            # Add 0.1 for each trigger found, maxing out at 0.3
            score += min(len(found_triggers) * 0.1, 0.3)

        return round(min(score, 1.0), 2)

if __name__ == "__main__":
    scorer = LinguisticScorer()
    
    # Test cases
    samples = [
        "Drinking warm water improves digestion.",
        "SHOCKING MIRACLE REMEDY!!! TURMERIC CURES CANCER!!! MUST SHARE!!!",
        "This is a secret doctors don't want you to know about obesity."
    ]

    print(f"{'Sample Text':<40} | {'Risk Score'}")
    print("-" * 55)
    for s in samples:
        score = scorer.calculate_score(s)
        print(f"{s[:38]:<40} | {score}")