
import joblib
import os


class RiskEngine:
    def __init__(
        self,
        model_path="models/language/baseline_lr_model.pkl",
        vec_path="models/language/tfidf_vectorizer.pkl",
    ):
        #changed wts
        self.BASE_FACT_WEIGHT  = 0.60   # fact verdict is the strongest signal
        self.BASE_ML_WEIGHT    = 0.25   # ML provides statistical second opinion
        self.BASE_STYLE_WEIGHT = 0.15   # style/sensationalism is weakest signal alone

        #tfidf
        if os.path.exists(model_path) and os.path.exists(vec_path):
            try:
                self.ml_model   = joblib.load(model_path)
                self.vectorizer = joblib.load(vec_path)
                self.has_ml     = True
                print("[RiskEngine] ML model loaded successfully.")
            except Exception as e:
                print(f"[RiskEngine] Warning: Failed to load ML model — {e}")
                self.has_ml = False
        else:
            print("[RiskEngine] Warning: ML model not found. Using heuristic-only mode.")
            self.has_ml = False

        #if not weigh rest more
        if not self.has_ml:
            total = self.BASE_FACT_WEIGHT + self.BASE_STYLE_WEIGHT
            self.fact_weight  = round(self.BASE_FACT_WEIGHT  / total, 4)
            self.style_weight = round(self.BASE_STYLE_WEIGHT / total, 4)
            self.ml_weight    = 0.0
        else:
            self.fact_weight  = self.BASE_FACT_WEIGHT
            self.ml_weight    = self.BASE_ML_WEIGHT
            self.style_weight = self.BASE_STYLE_WEIGHT

    #verducts
    VERDICT_SCORES = {
        "TRUE":       0.0,   # confirmed safe
        "MIXED":      0.4,   # partially true — moderate concern
        "FALSE":      1.0,   # confirmed dangerous
        "UNVERIFIED": 0.25,  # unknown — slight caution, not a penalty
    }

    #riks
    THRESHOLD_HIGH   = 0.65   # lowered from 0.75 — catches more genuine risks
    THRESHOLD_MEDIUM = 0.40   # lowered from 0.50 — UNVERIFIED claims land here

    def get_ml_probability(self, text):
        if not self.has_ml:
            return 0.0  # Fixed: was 0.5 before — caused phantom +0.15 on every score

        try:
            vec_text = self.vectorizer.transform([text])
            # predict_proba returns [prob_real, prob_fake] — we want prob_fake
            return float(self.ml_model.predict_proba(vec_text)[0][1])
        except Exception as e:
            print(f"[RiskEngine] ML prediction failed: {e}")
            return 0.0

    def calculate_risk(self, original_text, linguistic_score, fact_check_result):
        # --- 1. Fact Score ---
        verdict = fact_check_result.get("verdict", "UNVERIFIED")
        fact_score = self.VERDICT_SCORES.get(verdict, self.VERDICT_SCORES["UNVERIFIED"])

        # --- 2. ML Score ---
        ml_score = self.get_ml_probability(original_text)

        # --- 3. Weighted Final Score ---
        final_score = (
            (self.fact_weight  * fact_score)     +
            (self.ml_weight    * ml_score)        +
            (self.style_weight * linguistic_score)
        )
        final_score = round(min(final_score, 1.0), 2)

        # --- 4. Label ---
        if final_score >= self.THRESHOLD_HIGH:
            label = "CRITICAL / HIGH RISK"
        elif final_score >= self.THRESHOLD_MEDIUM:
            label = "WARNING / MODERATE RISK"
        else:
            label = "SAFE / LOW RISK"

        breakdown = {
            "fact_verdict":   verdict,
            "fact_impact":    round(fact_score  * self.fact_weight,  2),
            "ml_impact":      round(ml_score    * self.ml_weight,    2),
            "style_impact":   round(linguistic_score * self.style_weight, 2),
            "weights_used": {
                "fact":  self.fact_weight,
                "ml":    self.ml_weight,
                "style": self.style_weight,
            },
            "ml_available": self.has_ml,
        }

        return {
            "score":     final_score,
            "label":     label,
            "breakdown": breakdown,
        }

#test
if __name__ == "__main__":
    engine = RiskEngine()

    print("\n--- RiskEngine Test ---")
    print(f"ML Available : {engine.has_ml}")
    print(f"Weights Used : fact={engine.fact_weight}, ml={engine.ml_weight}, style={engine.style_weight}")
    print()

    test_cases = [
        {
            "label": "TRUE claim, neutral style",
            "text": "Drinking warm water improves digestion.",
            "style": 0.0,
            "fact":  {"verdict": "TRUE"}
        },
        {
            "label": "FALSE claim, sensational style",
            "text": "TURMERIC CURES CANCER COMPLETELY!!!",
            "style": 0.9,
            "fact":  {"verdict": "FALSE"}
        },
        {
            "label": "UNVERIFIED claim, neutral style",
            "text": "Eating papaya on empty stomach detoxifies liver.",
            "style": 0.1,
            "fact":  {"verdict": "UNVERIFIED"}
        },
        {
            "label": "MIXED claim, moderate style",
            "text": "Lemon water helps with weight loss to some extent.",
            "style": 0.3,
            "fact":  {"verdict": "MIXED"}
        },
        {
            "label": "UNVERIFIED claim, high sensationalism",
            "text": "SECRET REMEDY!!! Doctors don't want you to know this!!!",
            "style": 0.8,
            "fact":  {"verdict": "UNVERIFIED"}
        },
    ]

    for case in test_cases:
        result = engine.calculate_risk(case["text"], case["style"], case["fact"])
        print(f"Scenario : {case['label']}")
        print(f"Score    : {result['score']}  →  {result['label']}")
        print(f"Breakdown: {result['breakdown']}")
        print()