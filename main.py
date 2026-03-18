import spacy
from src.preprocessor import HinglishMapper
from src.lifestyle_ner import build_lifestyle_ner
from src.refine_extractor import extract_triples
from src.linguistic_scorer import LinguisticScorer
from src.verifier import FactVerifier
from src.risk_engine import RiskEngine
from src.RAGpipeline.knowledge_manager import KnowledgeManager
from src.RAGpipeline.nli_verifier import NLIVerifier


class TruthCheckPipeline:
    def __init__(self):
        print("--- Initializing TruthCheck Engine ---")

        self.mapper = HinglishMapper()
        self.scorer = LinguisticScorer()
        self.verifier = FactVerifier()
        self.engine = RiskEngine()

        # --- RAG COMPONENTS ---
        self.kb_manager = KnowledgeManager()
        self.kb_manager.load_and_index("TruthCheck/data/medical_kb/")
        self.nli_judge = NLIVerifier()
        # ----------------------

        self.nlp = spacy.load("en_core_web_sm")
        self.nlp = build_lifestyle_ner(self.nlp)

        print("--- Pipeline Ready ---\n")

    def analyze_query(self, raw_text):
        doc = self.nlp(raw_text)
        final_reports = []

        for sent in doc.sents:
            sentence_text = sent.text.strip()

            clean_sent = self.mapper.clean_text(sentence_text)
            style_score = self.scorer.calculate_score(sentence_text)
            triples = extract_triples(clean_sent, self.nlp)

            if not triples:
                continue

            for triple in triples:

                # -----------------------------
                # STEP 1: Local JSON Verification
                # -----------------------------
                fact_result = self.verifier.verify(triple)
                source_used = "Local JSON Knowledge Base"

                # -----------------------------
                # STEP 2: ALWAYS RUN RAG
                # -----------------------------
                triple_str = f"{triple['subject']} {triple['relation']} {triple['object']}"
                evidence_list = self.kb_manager.retrieve_evidence(triple_str)

                rag_info = None

                if evidence_list:
                    rag_result = self.nli_judge.verify(triple_str, evidence_list[0])

                    rag_info = (
                        f"RAG Evidence: {rag_result['evidence']} "
                        f"| RAG Verdict: {rag_result['verdict']} "
                        f"(confidence={rag_result['confidence']})"
                    )

                # Attach RAG explanation (do not override JSON verdict)
                if rag_info:
                    fact_result['truth'] = fact_result['truth'] + " || " + rag_info
                    source_used = source_used + " + Vector RAG"

                # -----------------------------
                # STEP 3: Risk Calculation
                # -----------------------------
                risk = self.engine.calculate_risk(sentence_text, style_score, fact_result)

                # -----------------------------
                # STEP 4: Build Report
                # -----------------------------
                report = {
                    "input": sentence_text,
                    "extracted_claim": f"{triple['subject']} -> {triple['relation']} -> {triple['object']}",
                    "verdict": fact_result['verdict'],
                    "explanation": fact_result['truth'],
                    "source": source_used,
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
    test_data_path = "TruthCheck/data/final_health_claims.csv"

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