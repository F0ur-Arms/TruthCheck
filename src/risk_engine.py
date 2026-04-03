import joblib
import os


class RiskEngine:
    def __init__(
        self,
        model_path="models/language/baseline_lr_model.pkl",
        vec_path="models/language/tfidf_vectorizer.pkl",
    ):
        self.BASE_FACT_WEIGHT = 0.60
        self.BASE_ML_WEIGHT = 0.25
        self.BASE_STYLE_WEIGHT = 0.15

        if os.path.exists(model_path) and os.path.exists(vec_path):
            try:
                self.ml_model = joblib.load(model_path)
                self.vectorizer = joblib.load(vec_path)
                self.has_ml = True
                print("[RiskEngine] ML model loaded successfully.")
            except Exception as e:
                print(f"[RiskEngine] Warning: Failed to load ML model - {e}")
                self.has_ml = False
        else:
            print("[RiskEngine] Warning: ML model not found. Using heuristic-only mode.")
            self.has_ml = False

        if not self.has_ml:
            total = self.BASE_FACT_WEIGHT + self.BASE_STYLE_WEIGHT
            self.fact_weight = round(self.BASE_FACT_WEIGHT / total, 4)
            self.style_weight = round(self.BASE_STYLE_WEIGHT / total, 4)
            self.ml_weight = 0.0
        else:
            self.fact_weight = self.BASE_FACT_WEIGHT
            self.ml_weight = self.BASE_ML_WEIGHT
            self.style_weight = self.BASE_STYLE_WEIGHT

    VERDICT_SCORES = {
        "TRUE": 0.0,
        "PARTIALLY TRUE": 0.45,
        "MIXED": 0.45,
        "FALSE": 1.0,
        "UNVERIFIED": 0.30,
    }

    MATCH_RELIABILITY = {
        "STRONG": 1.0,
        "WEAK": 0.7,
        "NONE": 0.35,
    }

    SOURCE_RISK_OFFSETS = {
        "HIGH": 0.08,
        "MEDIUM": 0.03,
        "LOW": -0.03,
    }

    THRESHOLD_HIGH = 0.65
    THRESHOLD_MEDIUM = 0.40

    def get_ml_probability(self, text):
        if not self.has_ml:
            return 0.0

        try:
            vec_text = self.vectorizer.transform([text])
            return float(self.ml_model.predict_proba(vec_text)[0][1])
        except Exception as e:
            print(f"[RiskEngine] ML prediction failed: {e}")
            return 0.0

    def _calibrate_fact_score(self, verdict, fact_check_result):
        fact_score = self.VERDICT_SCORES.get(verdict, self.VERDICT_SCORES["UNVERIFIED"])

        match_type = str(fact_check_result.get("match_type", "NONE")).upper()
        base_reliability = self.MATCH_RELIABILITY.get(match_type, self.MATCH_RELIABILITY["NONE"])

        nli_conf = fact_check_result.get("nli_confidence")
        if nli_conf is None:
            reliability = base_reliability
        else:
            confidence_factor = min(max(0.5 + (float(nli_conf) / 2.0), 0.55), 1.0)
            reliability = round(base_reliability * confidence_factor, 4)

        neutral_score = self.VERDICT_SCORES["UNVERIFIED"]
        calibrated_score = (
            (fact_score * reliability) +
            (neutral_score * (1.0 - reliability))
        )

        source_risk = str(fact_check_result.get("risk_level", "")).upper()
        calibrated_score += self.SOURCE_RISK_OFFSETS.get(source_risk, 0.0)

        rag_adjustment = 0.0
        rag_verdict = fact_check_result.get("rag_verdict")
        rag_conf = fact_check_result.get("rag_confidence")
        if rag_verdict and rag_conf is not None and float(rag_conf) >= 0.80:
            if rag_verdict == "REFUTES":
                rag_adjustment = 0.12
            elif rag_verdict == "SUPPORTS":
                rag_adjustment = -0.08

        calibrated_score = round(
            min(max(calibrated_score + rag_adjustment, 0.0), 1.0),
            4,
        )

        return calibrated_score, reliability, rag_adjustment, nli_conf, match_type

    def calculate_risk(self, original_text, linguistic_score, fact_check_result):
        verdict = str(fact_check_result.get("verdict", "UNVERIFIED")).upper()
        calibrated_fact_score, reliability, rag_adjustment, nli_conf, match_type = (
            self._calibrate_fact_score(verdict, fact_check_result)
        )

        ml_score = self.get_ml_probability(original_text)

        effective_style_score = linguistic_score
        if verdict == "TRUE" and reliability >= 0.80:
            effective_style_score = min(linguistic_score, 0.30)
        elif verdict in {"PARTIALLY TRUE", "MIXED"} and reliability >= 0.75:
            effective_style_score = min(linguistic_score, 0.45)

        final_score = (
            (self.fact_weight * calibrated_fact_score) +
            (self.ml_weight * ml_score) +
            (self.style_weight * effective_style_score)
        )
        final_score = round(min(final_score, 1.0), 2)

        if final_score >= self.THRESHOLD_HIGH:
            label = "CRITICAL / HIGH RISK"
        elif final_score >= self.THRESHOLD_MEDIUM:
            label = "WARNING / MODERATE RISK"
        else:
            label = "SAFE / LOW RISK"

        breakdown = {
            "fact_verdict": verdict,
            "fact_impact": round(calibrated_fact_score * self.fact_weight, 2),
            "ml_impact": round(ml_score * self.ml_weight, 2),
            "style_impact": round(effective_style_score * self.style_weight, 2),
            "match_type": match_type,
            "fact_reliability": reliability,
            "nli_confidence": nli_conf,
            "rag_adjustment": round(rag_adjustment, 2),
            "weights_used": {
                "fact": self.fact_weight,
                "ml": self.ml_weight,
                "style": self.style_weight,
            },
            "ml_available": self.has_ml,
        }

        return {
            "score": final_score,
            "label": label,
            "breakdown": breakdown,
        }


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
            "fact": {"verdict": "TRUE", "match_type": "STRONG", "nli_confidence": 0.95, "risk_level": "Low"},
        },
        {
            "label": "FALSE claim, sensational style",
            "text": "TURMERIC CURES CANCER COMPLETELY!!!",
            "style": 0.9,
            "fact": {"verdict": "FALSE", "match_type": "STRONG", "nli_confidence": 0.97, "risk_level": "High"},
        },
        {
            "label": "UNVERIFIED claim, neutral style",
            "text": "Eating papaya on empty stomach detoxifies liver.",
            "style": 0.1,
            "fact": {"verdict": "UNVERIFIED", "match_type": "NONE"},
        },
        {
            "label": "PARTIALLY TRUE claim, moderate style",
            "text": "Kadha cures fever completely.",
            "style": 0.3,
            "fact": {"verdict": "PARTIALLY TRUE", "match_type": "WEAK", "nli_confidence": 0.82, "risk_level": "Medium"},
        },
    ]

    for case in test_cases:
        result = engine.calculate_risk(case["text"], case["style"], case["fact"])
        print(f"Scenario : {case['label']}")
        print(f"Score    : {result['score']} -> {result['label']}")
        print(f"Breakdown: {result['breakdown']}")
        print()
