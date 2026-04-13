import spacy
import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from src.preprocessor import HinglishMapper
from src.linguistic_scorer import LinguisticScorer
from src.verifiernew import FactVerifier  # Now with built-in FAISS
from src.risk_engine import RiskEngine
from src.RAGpipeline.nli_verifier import NLIVerifier  # Only NLI from RAG
from src.RAGpipeline.knowledge_manager import KnowledgeManager

# ─────────────────────────────────────────────────────────────────────────────
# ORIGINAL LOGGER
# ─────────────────────────────────────────────────────────────────────────────
LOG_PATH = r"C:\Users\Shivam Kumar\frenemy\TruthCheck\logs.txt"

def init_log():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write(f"  NEW RUN (Built-in FAISS + NLI verifier) — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

def log(msg=""):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# NEW TARGETED DEBUG LOGGER (Scorer & Verifier Only)
# ─────────────────────────────────────────────────────────────────────────────
DEBUG_LOG_PATH = r"C:\Users\Shivam Kumar\frenemy\TruthCheck\src\loginput.txt"

def init_debug_log():
    os.makedirs(os.path.dirname(DEBUG_LOG_PATH), exist_ok=True)
    # Using 'w' to overwrite and give us a fresh clean log each run
    with open(DEBUG_LOG_PATH, "w", encoding="utf-8") as f:
        f.write(f"DEBUG RUN: SCORER & VERIFIER ISOLATION — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 100 + "\n")

def debug_log(msg=""):
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
class TruthCheckPipeline:
    def __init__(self):
        print("--- Initializing TruthCheck Engine ---")

        self.mapper = HinglishMapper()
        self.scorer = LinguisticScorer()
        self.engine = RiskEngine()

        # ── Step 1: NLI model — loaded once, shared ───────────────────────────
        print("--- Loading NLI model (shared) ---")
        self.nli_judge = NLIVerifier()

        # ── Step 2: FactVerifier — NOW WITH BUILT-IN FAISS ────────────────────
        # No more KnowledgeManager dependency!
        print("--- Initializing FactVerifier with built-in FAISS ---")
        self.verifier = FactVerifier(
            facts_path="TruthCheck/data/final_verifiedfacts.json",
            nli_verifier=self.nli_judge,
            embedding_model='all-MiniLM-L6-v2'
        )
        #rag retrieval
                # ── RAG RETRIEVAL SETUP ─────────────────────────
        self.kb_manager = KnowledgeManager()
        self.kb_manager.load_and_index("TruthCheck/data/medical_kb/")
        # ── Step 3: spaCy ─────────────────────────────────────────────────────
        self.nlp = spacy.load("en_core_web_sm")

        print("--- Pipeline Ready ---\n")

    def analyze_query(self, raw_text, row_index=None, true_label=None):
        # Step 1: Clean the sentences
        clean_text = self.mapper.clean_text(raw_text)

        # Step 2: Sentence segmentation using spacy 
        doc = self.nlp(clean_text)
        final_reports = []

        # Logs
        log(f"{'─' * 70}")
        log(f"ROW #{row_index}  INPUT : {raw_text}")
        log(f"              CLEANED: {clean_text}")
        log(f"{'─' * 70}")
        debug_log(f"\n[ROW {row_index} | TRUE LABEL: {true_label}]")
        debug_log(f"CLAIM: {raw_text}")

        # Step 3: Loop through sentences
        for sent_idx, sent in enumerate(doc.sents, 1):
            sentence = sent.text.strip()
            if not sentence:
                continue

            log(f"\n  [SENTENCE {sent_idx}] {sentence}")

            # Signal 1: Style scoring
            style_score = self.scorer.calculate_score(sentence)
            log(f"  [STYLE SCORE] {style_score}")

            # Signal 2: Fact-based score (now using built-in FAISS)
            fact_result = self.verifier.verify(sentence)
            # ── RAG: Retrieve evidence ──────────────────────
            evidence_list = self.kb_manager.retrieve_evidence(sentence)

            if evidence_list:
                top_evidence = evidence_list[0]

                rag_result = self.nli_judge.verify(sentence, top_evidence)

                # Attach RAG outputs
                fact_result["rag_evidence"] = top_evidence   # ✅ ADD THIS
                fact_result["rag_verdict"] = rag_result["verdict"]
                fact_result["rag_confidence"] = rag_result["confidence"]

                # Optional: upgrade verdict if KB is unsure
                if (
                    fact_result["verdict"] == "UNVERIFIED"
                    and rag_result["confidence"] >= 0.85
                    and rag_result["verdict"] in {"SUPPORTS", "REFUTES"}
                ):
                    fact_result["verdict"] = (
                        "TRUE" if rag_result["verdict"] == "SUPPORTS" else "FALSE"
                    )

                # Append explanation
                fact_result["truth"] += (
                    f" || RAG: {rag_result['verdict']} "
                    f"(conf={rag_result['confidence']})"
                )

                source_used = "KB + RAG"
            else:
                fact_result["rag_verdict"] = None
                fact_result["rag_confidence"] = None
                source_used = "Local JSON Knowledge Base (Built-in FAISS)"

            log(f"  [KB VERIFIER]")
            log(f"    Match Found   : {fact_result.get('match_found', False)}")
            log(f"    Matched Entry : {fact_result.get('matched_entry', 'None')}")
            log(f"    Match Type    : {fact_result.get('match_type', 'N/A')}")
            log(f"    Match Score   : {fact_result.get('match_score', 0.0)}")
            log(f"    KB Verdict    : {fact_result.get('verdict')}")
            log(f"    NLI Verdict   : {fact_result.get('nli_verdict')}")
            log(f"    NLI Confidence: {fact_result.get('nli_confidence')}")
            log(f"    Truth         : {fact_result.get('truth', '')[:120]}")

            # Use signal1 + signal2 to get risk
            risk = self.engine.calculate_risk(sentence, style_score, fact_result)

            log(f"  [RISK ENGINE]")
            log(f"    Final Score : {risk['score']}")
            log(f"    Risk Label  : {risk['label']}")
            bd = risk['breakdown']
            log(f"    Breakdown:")
            log(f"      fact_verdict : {bd['fact_verdict']}")
            log(f"      fact_impact  : {bd['fact_impact']}  (weight={bd['weights_used']['fact']})")
            log(f"      ml_impact    : {bd['ml_impact']}    (weight={bd['weights_used']['ml']})")
            log(f"  [RAG] Verdict: {rag_result['verdict']} | Confidence: {rag_result['confidence']}")
            log(f"  [RAG] Evidence: {fact_result.get('rag_evidence')[:]}")
            log(f"      style_impact : {bd['style_impact']} (weight={bd['weights_used']['style']})")
            log(f"    Source Used : {source_used}")

            report = {
                "input": sentence,
                "verdict": fact_result['verdict'],
                "nli_verdict": fact_result.get('nli_verdict'),
                "rag_verdict": fact_result.get('rag_verdict'),   
                "rag_confidence": fact_result.get('rag_confidence'),  
                "explanation": fact_result['truth'],
                "source": source_used,
                "risk_score": risk['score'],
                "risk_level": risk['label'],
            }
            final_reports.append(report)

        return final_reports if final_reports else [{"error": "No sentences found."}]


# ─────────────────────────────────────────────────────────────────────────────
# BINARY PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
x = 0

def get_binary_prediction(reports, risk_threshold=0.5):
    """
    0 = Real/Safe
    1 = Fake/Misinformation
    """
    if not reports or "error" in reports[0]:
        return 1  # fallback: treat as risky

    # ─────────────────────────────────────────
    # 1. FACT VERIFIER (HIGHEST PRIORITY)
    # ─────────────────────────────────────────
    for r in reports:
        if r['verdict'] == "FALSE":
            return 1
        if r['verdict'] == "TRUE":
            return 0

    # ─────────────────────────────────────────
    # 2. RAG VERDICT (SECONDARY)
    # ─────────────────────────────────────────
    for r in reports:
        if r.get('rag_verdict') == "REFUTES":
            return 1
        if r.get('rag_verdict') == "SUPPORTS":
            return 0

    # ─────────────────────────────────────────
    # 3. RISK SCORE (FALLBACK)
    # ─────────────────────────────────────────
    max_risk = max(r.get('risk_score', 0) for r in reports)
    return 1 if max_risk > risk_threshold else 0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    init_log()
    init_debug_log()
    log("CONFIGURATION: Built-in FAISS mode + NLI-backed FactVerifier\n")

    print("=" * 60)
    print("  TRUTHCHECK ENGINE: BUILT-IN FAISS + NLI MODE")
    print("=" * 60)

    pipeline = TruthCheckPipeline()
    test_data_path = "TruthCheck/data/final_health_claims.csv"

    if not os.path.exists(test_data_path):
        print(f"CRITICAL ERROR: CSV not found at {test_data_path}")
        log(f"CRITICAL ERROR: CSV not found at {test_data_path}")
    else:
        df = pd.read_csv(test_data_path)
        # Shuffle full dataset (recommended)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"[Dataset] Loaded {len(df)} claims for testing.")
        log(f"Dataset: {test_data_path} | Rows: {len(df)}\n")

        y_true = df['label'].astype(int).tolist()
        y_pred = []

        print("[Inference] Analyzing claims...")
        for i, row in df.iterrows():
            claim_text = row['claim']
            true_label = int(row['label'])

            reports = pipeline.analyze_query(claim_text, row_index=i + 1, true_label=true_label)
            prediction = get_binary_prediction(reports)
            y_pred.append(prediction)

            correct = "✅" if prediction == true_label else "❌"
            log(f"  [RESULT] Predicted={prediction} | True={true_label} {correct}")
            log("")

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(df)}...")

        # ── Metrics ───────────────────────────────────────────────────────────
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        metrics_str = (
            f"\n{'=' * 60}\n"
            f"  EVALUATION RESULTS\n"
            f"{'=' * 60}\n"
            f"  Accuracy  : {acc:.4f}\n"
            f"  Precision : {prec:.4f}\n"
            f"  Recall    : {rec:.4f}\n"
            f"  F1 Score  : {f1:.4f}\n"
            f"\n{classification_report(y_true, y_pred, target_names=['Real (0)', 'Fake (1)'])}\n"
            f"  Unparseable claims (error fallback): {x}\n"
        )
        print(metrics_str)
        log(metrics_str)

    print("=" * 60)
    print(f"  DONE | Unparseable: {x} | Main Log: {LOG_PATH}")
    print(f"  DEBUG LOG SAVED TO: {DEBUG_LOG_PATH}")
    print("=" * 60)