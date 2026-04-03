import spacy
from src.preprocessor import HinglishMapper
from src.lifestyle_ner import build_lifestyle_ner
from src.refine_extractor import extract_triples
from src.linguistic_scorer import LinguisticScorer
from src.verifier import FactVerifier
from src.risk_engine import RiskEngine
from src.RAGpipeline.knowledge_manager import KnowledgeManager
from src.RAGpipeline.nli_verifier import NLIVerifier

CLAIM_CUE_WORDS = {
    "cure", "cures", "prevent", "prevents", "cause", "causes", "reduce", "reduces",
    "improve", "improves", "help", "helps", "boost", "boosts", "increase", "increases",
    "decrease", "decreases", "treat", "treats", "heal", "heals", "protect", "protects",
    "risk", "safe", "unsafe", "harm", "harms", "benefit", "benefits", "affect", "affects",
    "digestion", "immunity", "cancer", "diabetes", "blood pressure", "covid", "fever",
}

NON_CLAIM_PHRASES = {
    "please share", "must share", "share this", "forward this", "forward to everyone",
    "subscribe now", "click here", "watch this", "breaking news", "urgent alert",
}


class TruthCheckPipeline:
    def __init__(self):
        print("--- Initializing TruthCheck Engine ---")

        self.mapper = HinglishMapper()
        self.scorer = LinguisticScorer()
        self.engine = RiskEngine()

        # --- RAG COMPONENTS ---
        self.kb_manager = KnowledgeManager()
        self.kb_manager.load_and_index("data/medical_kb")
        self.kb_manager.load_verified_facts("data/verified_facts.json")
        self.nli_judge = NLIVerifier()
        self.verifier = FactVerifier(
            kb_manager=self.kb_manager,
            nli_verifier=self.nli_judge,
        )
        # ----------------------

        self.nlp = spacy.load("en_core_web_sm")
        self.nlp = build_lifestyle_ner(self.nlp)

        print("--- Pipeline Ready ---\n")

    def _is_checkable_claim(self, sentence_span, clean_sent):
        word_count = len(clean_sent.split())
        if word_count < 3:
            return False

        lower_sent = clean_sent.lower()
        if any(phrase in lower_sent for phrase in NON_CLAIM_PHRASES):
            return False

        if any(cue in lower_sent for cue in CLAIM_CUE_WORDS):
            return True

        if any(ent.label_ in {"DIET_HABIT", "BODY_SYSTEM", "HABIT", "INGREDIENT"} for ent in sentence_span.ents):
            return True

        return any(token.pos_ in {"VERB", "AUX", "ADJ"} for token in sentence_span if token.is_alpha)

    def _format_extracted_claims(self, triples, fallback_claim):
        if not triples:
            return [fallback_claim]

        formatted = []
        seen = set()
        for triple in triples:
            text = f"{triple['subject']} -> {triple['relation']} -> {triple['object']}"
            if text not in seen:
                seen.add(text)
                formatted.append(text)
        return formatted or [fallback_claim]

    def analyze_query(self, raw_text):
        doc = self.nlp(raw_text)
        final_reports = []

        for sent in doc.sents:
            sentence_text = sent.text.strip()
            if not sentence_text:
                continue

            clean_sent = self.mapper.clean_text(sentence_text)
            if not self._is_checkable_claim(sent, clean_sent):
                continue

            style_result = self.scorer.calculate_score_detailed(sentence_text)
            style_score = style_result["score"]
            triples = extract_triples(clean_sent, self.nlp)
            claim_text = clean_sent or sentence_text
            fact_result = self.verifier.verify(claim_text)
            evidence_list = self.kb_manager.retrieve_evidence(claim_text)
            source_used = "Verified Facts Semantic Match"

            if evidence_list:
                rag_evidence = evidence_list[0]
                rag_result = self.nli_judge.verify(claim_text, rag_evidence)
                fact_result["rag_evidence"] = rag_result["evidence"]
                fact_result["rag_verdict"] = rag_result["verdict"]
                fact_result["rag_confidence"] = rag_result["confidence"]

                if (
                    fact_result["verdict"] == "UNVERIFIED"
                    and rag_result["confidence"] >= 0.90
                    and rag_result["verdict"] in {"SUPPORTS", "REFUTES"}
                ):
                    fact_result["verdict"] = "TRUE" if rag_result["verdict"] == "SUPPORTS" else "FALSE"

                fact_result["truth"] = (
                    fact_result["truth"]
                    + " || "
                    + f"RAG Evidence: {rag_result['evidence']} "
                    + f"| RAG Verdict: {rag_result['verdict']} "
                    + f"(confidence={rag_result['confidence']})"
                )
                source_used = "Verified Facts Semantic Match + Vector RAG"

            risk = self.engine.calculate_risk(sentence_text, style_score, fact_result)
            extracted_claims = self._format_extracted_claims(triples, claim_text)

            report = {
                "input": sentence_text,
                "claim_text": claim_text,
                "extracted_claim": extracted_claims[0],
                "extracted_claims": extracted_claims,
                "verdict": fact_result['verdict'],
                "explanation": fact_result['truth'],
                "source": source_used,
                "style_score": style_score,
                "style_breakdown": style_result["breakdown"],
                "match_type": fact_result.get("match_type"),
                "match_score": fact_result.get("match_score"),
                "rag_verdict": fact_result.get("rag_verdict"),
                "rag_confidence": fact_result.get("rag_confidence"),
                "risk_score": risk['score'],
                "risk_level": risk['label']
            }

            final_reports.append(report)

        return final_reports if final_reports else [{"error": "No clear health claims found."}]

# --- EVALUATION HELPERS ---

x=0
def get_binary_prediction(reports, risk_threshold=0.5):
    """
    Aggregates TruthCheck reports into a single binary label.
    0 = Real/Safe, 1 = Fake/Misinformation
    """
    global x
    if not reports or "error" in reports[0]:
        x+=1
        return 1  # Safety first: flag as 1 if the engine can't parse it
    
    # 1. Check for hard refutations (Label 1)
    for r in reports:
        if r['verdict'] == "FALSE" or "RAG Verdict: REFUTES" in r['explanation']:
            return 1
            
    # 2. Check Risk Scores (Label 1)
    max_risk = max([r.get('risk_score', 0) for r in reports])
    if max_risk > risk_threshold:
        return 1
        
    # 3. Check for positive verification (Label 0)
    for r in reports:
        if r['verdict'] == "TRUE" or "RAG Verdict: SUPPORTS" in r['explanation']:
            return 0
            
    # 4. Neutral/Unverified fallback
    return 1 if max_risk > 0.3 else 0

if __name__ == "__main__":
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, confusion_matrix, classification_report
    )

    print("=" * 60)
    print("  TRUTHCHECK ENGINE: BTP EVALUATION MODE")
    print("=" * 60)

    # 1. Initialize
    pipeline = TruthCheckPipeline()
    test_data_path = "data/final_health_claims.csv"

    if not os.path.exists(test_data_path):
        print(f"CRITICAL ERROR: CSV not found at {test_data_path}")
    else:
        # 2. Load Dataset
        df = pd.read_csv(test_data_path)
        print(f"[Dataset] Loaded {len(df)} claims for testing.")

        y_true = df['label'].astype(int).tolist()
        y_pred = []

        # 3. Run Inference
        print("[Inference] Analyzing claims (NER + RAG Verification)...")
        for i, row in df.iterrows():
            claim_text = row['claim']
            raw_reports = pipeline.analyze_query(claim_text)
            if i==0:
                print(raw_reports)
            prediction = get_binary_prediction(raw_reports)
            y_pred.append(prediction)

            if (i + 1) % 5 == 0:
                print(f"            Processed {i + 1}/{len(df)}...")

        # 4. Calculate Stats
        # Note: 'weighted' is used to match the multi-class style of your baseline
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # 5. Print Results in Requested Format
        print(f"\n{'─'*50}")
        print(f"  TruthCheck (NER + RAG) — Results")
        print(f"{'─'*50}")
        print(f"  Accuracy   : {acc:.4f}")
        print(f"  Precision  : {prec:.4f}")
        print(f"  Recall     : {rec:.4f}")
        print(f"  F1 Score   : {f1:.4f}")
        print(f"\n{classification_report(y_true, y_pred, target_names=['Real (0)', 'Fake (1)'])}")

        # 6. Plot Confusion Matrix using Matplotlib/Seaborn
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Predicted Real', 'Predicted Fake'],
            yticklabels=['Actual Real', 'Actual Fake']
        )
        plt.title('Confusion Matrix: TruthCheck NER + RAG Pipeline')
        plt.ylabel('Ground Truth')
        plt.xlabel('Pipeline Prediction')
        
        # Save the plot for your BTP report
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/truthcheck_confusion_matrix.png")
        print("\n[Plot] Confusion matrix saved to 'results/truthcheck_confusion_matrix.png'")
        
        plt.show()

    print("=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)
    print(x)
