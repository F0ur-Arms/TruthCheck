import json
import os

class FactVerifier:
    def __init__(self, kb_path="data/verified_facts.json"):
        # Look for the data file relative to the project root
        self.kb_path = kb_path
        self.kb = self._load_kb()

    def _load_kb(self):
        if not os.path.exists(self.kb_path):
            # Fallback for testing if file doesn't exist yet
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
                    "scientific_truth": "No clinical evidence supports papaya seeds as a safe or effective abortion method; it can be toxic.",
                    "risk_level": "High"
                }
            ]
        with open(self.kb_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def verify(self, extracted_triple):
        """
        Matches triple against the Knowledge Base.
        """
        subj = extracted_triple['subject'].lower()
        obj = extracted_triple['object'].lower()
        
        # Check for matches in our KB
        for entry in self.kb:
            keyword = entry["claim_subject"].lower()
            # If the keyword exists in either the subject or object of the triple
            if keyword in subj or keyword in obj:
                return {
                    "match_found": True,
                    "verdict": entry["verdict"],
                    "truth": entry["scientific_truth"],
                    "risk_level": entry["risk_level"]
                }
        
        return {
            "match_found": False,
            "verdict": "UNVERIFIED",
            "truth": "This claim is not in our verified medical database yet.",
            "risk_level": "Medium"
        }

if __name__ == "__main__":
    # Test Objective 4
    verifier = FactVerifier()
    test_triple = {"subject": "drinking warm water", "relation": "improves", "object": "digestion"}
    
    result = verifier.verify(test_triple)
    print(f"Triple: {test_triple['subject']} -> {test_triple['relation']}")
    print(f"Verdict: {result['verdict']}")
    print(f"Evidence: {result['truth']}")