class RiskEngine:
    def __init__(self):
        # Weights for the final score
        self.fact_weight = 0.7
        self.style_weight = 0.3

    def calculate_risk(self, linguistic_score, fact_check_result):
        """
        Combines style and fact-check into a final danger score.
        """
        # Step 1: Map the Verdict to a numeric value
        # FALSE is maximum risk, TRUE is zero risk, UNVERIFIED is cautious
        verdict_scores = {
            "FALSE": 1.0,
            "MIXED": 0.5,
            "TRUE": 0.0,
            "UNVERIFIED": 0.4
        }
        
        fact_score = verdict_scores.get(fact_check_result['verdict'], 0.4)
        
        # Step 2: Apply Weighted Average
        final_score = (self.fact_weight * fact_score) + (self.style_weight * linguistic_score)
        
        # Step 3: Assign a Label
        if final_score >= 0.7:
            level = "HIGH RISK / LIKELY FAKE"
        elif 0.4 <= final_score < 0.7:
            level = "MODERATE RISK / UNVERIFIED"
        else:
            level = "LOW RISK / LIKELY SAFE"
            
        return {
            "score": round(final_score, 2),
            "label": level
        }

if __name__ == "__main__":
    engine = RiskEngine()
    
    # Example 1: A FALSE claim written in SHOUTY style
    print(f"Scenario 1: {engine.calculate_risk(1.0, {'verdict': 'FALSE'})}")
    
    # Example 2: A TRUE claim written in NEUTRAL style
    print(f"Scenario 2: {engine.calculate_risk(0.0, {'verdict': 'TRUE'})}")
    
    # Example 3: An UNVERIFIED claim written in SHOUTY style
    print(f"Scenario 3: {engine.calculate_risk(0.8, {'verdict': 'UNVERIFIED'})}")