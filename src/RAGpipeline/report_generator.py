class ReportGenerator:
    def __init__(self):
        # Thresholds for risk levels
        self.risk_mapping = {
            "SUPPORTS": {"label": "SAFE", "score_boost": 0.1},
            "REFUTES": {"label": "HIGH RISK / MISINFORMATION", "score_boost": 0.9},
            "NEI": {"label": "UNCERTAIN / UNVERIFIED", "score_boost": 0.5}
        }

    def generate_final_report(self, original_input, nli_result, linguistic_score):
        """
        Synthesizes all pipeline outputs into a single report.
        linguistic_score: 0.0 (formal) to 1.0 (clickbait/sensational)
        """
        verdict = nli_result['verdict']
        nli_conf = nli_result['confidence']
        evidence = nli_result['evidence']
        
        # Calculate a weighted risk score
        # (NLI verdict carries more weight than the 'style' of the text)
        base_risk = self.risk_mapping[verdict]['score_boost']
        final_risk_score = (base_risk * 0.7) + (linguistic_score * 0.3)
        
        # Create the human-friendly explanation
        explanation = self._craft_explanation(verdict, evidence, original_input)

        return {
            "input": original_input,
            "verdict": self.risk_mapping[verdict]['label'],
            "risk_score": round(final_risk_score, 2),
            "explanation": explanation,
            "source_fragment": evidence
        }

    def _craft_explanation(self, verdict, evidence, original_input):
        """Creates a simple summary. For Hinglish context, you can wrap this in a translator."""
        if verdict == "REFUTES":
            return f"This claim seems incorrect. Reliable sources state: '{evidence}'"
        elif verdict == "SUPPORTS":
            return f"This is likely true. According to health guidelines: '{evidence}'"
        else:
            return "We couldn't find enough scientific evidence to confirm or deny this claim."

if __name__ == "__main__":
    # Example usage after NLI is done
    gen = ReportGenerator()
    dummy_nli = {
        "verdict": "REFUTES",
        "confidence": 0.98,
        "evidence": "Clinical trials show no evidence that high doses of Vitamin C prevent viral infections."
    }
    
    report = gen.generate_final_report(
        original_input="Vitamin C prevents all colds instantly!",
        nli_result=dummy_nli,
        linguistic_score=0.85 # High sensationalism
    )
    
    print("--- TRUTHCHECK FINAL REPORT ---")
    print(f"VERDICT: {report['verdict']}")
    print(f"RISK: {report['risk_score']}")
    print(f"WHY: {report['explanation']}")