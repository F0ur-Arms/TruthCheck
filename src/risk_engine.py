import joblib
import os

class RiskEngine:
    def __init__(self, model_path='models/language/baseline_lr_model.pkl', 
                 vec_path='models/language/tfidf_vectorizer.pkl'):
        # Updated Weights: Fact is still king, but ML provides a critical "second opinion"
        self.fact_weight = 0.5
        self.ml_weight = 0.3
        self.style_weight = 0.2
        
        # Load the ML artifacts you just saved
        if os.path.exists(model_path) and os.path.exists(vec_path):
            self.ml_model = joblib.load(model_path)
            self.vectorizer = joblib.load(vec_path)
            self.has_ml = True
        else:
            print("Warning: ML model not found. Using heuristic-only mode.")
            self.has_ml = False

    def get_ml_probability(self, text):
        """Predicts the probability of the text being 'Fake' using the ML model."""
        if not self.has_ml: return 0.5
        vec_text = self.vectorizer.transform([text])
        # Returns [prob_real, prob_fake] -> we want prob_fake
        return self.ml_model.predict_proba(vec_text)[0][1]

    def calculate_risk(self, original_text, linguistic_score, fact_check_result):
        # 1. Map Fact Verdict
        verdict_scores = {"FALSE": 1.0, "MIXED": 0.5, "TRUE": 0.0, "UNVERIFIED": 0.4}
        fact_score = verdict_scores.get(fact_check_result['verdict'], 0.4)
        
        # 2. Get Statistical Score (ML)
        ml_score = self.get_ml_probability(original_text)
        
        # 3. Calculate Weighted Final Score
        final_score = (self.fact_weight * fact_score) + \
                      (self.ml_weight * ml_score) + \
                      (self.style_weight * linguistic_score)
        
        # 4. Final Labeling Logic
        if final_score >= 0.75:
            level = "CRITICAL / HIGH RISK"
        elif 0.5 <= final_score < 0.75:
            level = "WARNING / MODERATE RISK"
        else:
            level = "SAFE / LOW RISK"
            
        return {
            "score": round(final_score, 2),
            "label": level,
            "breakdown": {
                "fact_impact": round(fact_score * self.fact_weight, 2),
                "ml_impact": round(ml_score * self.ml_weight, 2),
                "style_impact": round(linguistic_score * self.style_weight, 2)
            }
        }

if __name__ == "__main__":
    engine = RiskEngine()
    
    # Example 1: A FALSE claim written in SHOUTY style
    print(f"Scenario 1: {engine.calculate_risk(1.0, {'verdict': 'FALSE'})}")
    
    # Example 2: A TRUE claim written in NEUTRAL style
    print(f"Scenario 2: {engine.calculate_risk(0.0, {'verdict': 'TRUE'})}")
    
    # Example 3: An UNVERIFIED claim written in SHOUTY style
    print(f"Scenario 3: {engine.calculate_risk(0.8, {'verdict': 'UNVERIFIED'})}")