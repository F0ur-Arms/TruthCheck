import spacy
from src.preprocessor import HinglishMapper
from src.lifestyle_ner import build_lifestyle_ner
from src.refine_extractor import extract_triples
from src.linguistic_scorer import LinguisticScorer
from src.verifier import FactVerifier
from src.risk_engine import RiskEngine

class TruthCheckPipeline:
    def __init__(self):
        print("--- Initializing TruthCheck Engine ---")
        self.mapper = HinglishMapper()
        self.scorer = LinguisticScorer()
        self.verifier = FactVerifier()
        self.engine = RiskEngine()
        
        # Load NLP model and add custom Lifestyle NER
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp = build_lifestyle_ner(self.nlp)
        print("--- Pipeline Ready ---\n")

    def analyze_query(self, raw_text):
        # 1. Preprocess Hinglish
        clean_text = self.mapper.clean_text(raw_text)
        
        # 2. LINGUISTIC STYLE (on raw text to catch !!!)
        style_score = self.scorer.calculate_score(raw_text)
        
        # 3. EXTRACTION (Pass CLEAN text here)
        # The clean_text is now "Morning empty stomach drinking warm water improves digestion"
        triples = extract_triples(clean_text, self.nlp)
        
        final_reports = []
        
        if not triples:
            return {"error": "Could not identify a clear health claim in the text."}

        for triple in triples:
            # 4. Fact Verification
            fact_result = self.verifier.verify(triple)
            
            # 5. Risk Calculation
            risk = self.engine.calculate_risk(style_score, fact_result)
            
            report = {
                "input": raw_text,
                "extracted_claim": f"{triple['subject']} -> {triple['relation']} -> {triple['object']}",
                "verdict": fact_result['verdict'],
                "explanation": fact_result['truth'],
                "risk_score": risk['score'],
                "risk_level": risk['label']
            }
            final_reports.append(report)
            
        return final_reports

if __name__ == "__main__":
    pipeline = TruthCheckPipeline()
    
    # Test with a common Indian health myth
    user_input = "Subah khali pet drinking warm water improves digestion!!!"
    
    results = pipeline.analyze_query(user_input)
    
    for r in results:
        print(f"REPORT FOR: '{r['input']}'")
        print(f"Claim: {r['extracted_claim']}")
        print(f"Verdict: {r['verdict']}")
        print(f"Risk: {r['risk_score']} ({r['risk_level']})")
        print(f"Scientific Truth: {r['explanation']}")
        print("-" * 50)