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
from src.verifier import FactVerifier
from src.risk_engine import RiskEngine
from src.RAGpipeline.knowledge_manager import KnowledgeManager
from src.RAGpipeline.nli_verifier import NLIVerifier


# ─────────────────────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────────────────────
LOG_PATH = r"C:\Users\Shivam Kumar\frenemy\TruthCheck\logs.txt"

def init_log():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write(f"  NEW RUN (full-sentence + NLI-backed verifier) — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

def log(msg=""):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
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

        # ── Step 1: NLI model — loaded once, shared by verifier + RAG ─────────
        print("--- Loading NLI model (shared) ---")
        self.nli_judge = NLIVerifier()

        # ── Step 2: KnowledgeManager — MUST come before FactVerifier ──────────
        self.kb_manager = KnowledgeManager()
        self.kb_manager.load_and_index("TruthCheck/data/medical_kb/")
        self.kb_manager.load_verified_facts("TruthCheck/data/verified_facts.json")

        # ── Step 3: FactVerifier — receives kb_manager + nli ──────────────────
        self.verifier = FactVerifier(
            kb_manager=self.kb_manager,
            nli_verifier=self.nli_judge
        )

        # ── Step 4: spaCy ─────────────────────────────────────────────────────
        self.nlp = spacy.load("en_core_web_sm")

        print("--- Pipeline Ready ---\n")

    def analyze_query(self, raw_text, row_index=None):
        clean_text = self.mapper.clean_text(raw_text)
        doc        = self.nlp(clean_text)
        final_reports = []

        log(f"{'─' * 70}")
        log(f"ROW #{row_index}  INPUT : {raw_text}")
        log(f"              CLEANED: {clean_text}")
        log(f"{'─' * 70}")

        for sent_idx, sent in enumerate(doc.sents, 1):
            sentence = sent.text.strip()
            if not sentence:
                continue

            log(f"\n  [SENTENCE {sent_idx}] {sentence}")

            # ── Style score ───────────────────────────────────────────────────
            style_score = self.scorer.calculate_score(sentence)
            log(f"  [STYLE SCORE] {style_score}")

            # ── Fact verification — full sentence, NLI-backed ─────────────────
            fact_result = self.verifier.verify(sentence)
            source_used = "Local JSON Knowledge Base"

            log(f"  [KB VERIFIER]")
            log(f"    Match Found   : {fact_result.get('match_found', False)}")
            log(f"    Matched Entry : {fact_result.get('matched_entry', 'None')}")
            log(f"    Match Type    : {fact_result.get('match_type', 'N/A')}")
            log(f"    Match Score   : {fact_result.get('match_score', 0.0)}")
            log(f"    KB Verdict    : {fact_result.get('verdict')}")
            log(f"    NLI Verdict   : {fact_result.get('nli_verdict')}")
            log(f"    NLI Confidence: {fact_result.get('nli_confidence')}")
            log(f"    Truth         : {fact_result.get('truth', '')[:120]}")

            # ── RAG + NLI — full sentence as claim ────────────────────────────
            evidence_list = self.kb_manager.retrieve_evidence(sentence)
            rag_info = None

            log(f"  [RAG]")
            if not evidence_list:
                log(f"    ⚠ No evidence retrieved from medical KB")
            else:
                log(f"    Evidence retrieved: {len(evidence_list)} passage(s)")
                log(f"    Top evidence: {evidence_list[0][:150]}")

                rag_result = self.nli_judge.verify(sentence, evidence_list[0])

                log(f"  [NLI — RAG]")
                log(f"    Verdict    : {rag_result['verdict']}")
                log(f"    Confidence : {rag_result['confidence']}")
                log(f"    Evidence   : {rag_result['evidence'][:150]}")

                rag_info = (
                    f"RAG Evidence: {rag_result['evidence']} "
                    f"| RAG Verdict: {rag_result['verdict']} "
                    f"(confidence={rag_result['confidence']})"
                )

            if rag_info:
                fact_result['truth'] = fact_result['truth'] + " || " + rag_info
                source_used = source_used + " + Vector RAG"

            # ── Risk calculation ──────────────────────────────────────────────
            risk = self.engine.calculate_risk(sentence, style_score, fact_result)

            log(f"  [RISK ENGINE]")
            log(f"    Final Score : {risk['score']}")
            log(f"    Risk Label  : {risk['label']}")
            bd = risk['breakdown']
            log(f"    Breakdown:")
            log(f"      fact_verdict : {bd['fact_verdict']}")
            log(f"      fact_impact  : {bd['fact_impact']}  (weight={bd['weights_used']['fact']})")
            log(f"      ml_impact    : {bd['ml_impact']}    (weight={bd['weights_used']['ml']})")
            log(f"      style_impact : {bd['style_impact']} (weight={bd['weights_used']['style']})")
            log(f"    Source Used : {source_used}")

            report = {
                "input":       sentence,
                "verdict":     fact_result['verdict'],
                "nli_verdict": fact_result.get('nli_verdict'),
                "explanation": fact_result['truth'],
                "source":      source_used,
                "risk_score":  risk['score'],
                "risk_level":  risk['label'],
            }
            final_reports.append(report)

        return final_reports if final_reports else [{"error": "No sentences found."}]


# ─────────────────────────────────────────────────────────────────────────────
# BINARY PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
x = 0

def get_binary_prediction(reports, risk_threshold=0.5):
    global x

    if not reports or "error" in reports[0]:
        x += 1
        log(f"  [PREDICTION] ⚠ No reports — defaulting to 1 (Fake). Error count: {x}")
        return 1

    # 1. Hard KB verdicts
    for r in reports:
        if r['verdict'] == "FALSE":
            log(f"  [PREDICTION] → 1 (Fake) via KB verdict=FALSE")
            return 1
        if r['verdict'] == "TRUE":
            log(f"  [PREDICTION] → 0 (Real) via KB verdict=TRUE")
            return 0

    # 2. RAG verdicts
    for r in reports:
        if "RAG Verdict: REFUTES" in r.get('explanation', ''):
            log(f"  [PREDICTION] → 1 (Fake) via RAG REFUTES")
            return 1
        if "RAG Verdict: SUPPORTS" in r.get('explanation', ''):
            log(f"  [PREDICTION] → 0 (Real) via RAG SUPPORTS")
            return 0

    # 3. Risk score fallback
    max_risk   = max(r.get('risk_score', 0) for r in reports)
    prediction = 1 if max_risk > risk_threshold else 0
    log(f"  [PREDICTION] → {prediction} via risk score (max_risk={max_risk}, threshold={risk_threshold})")
    return prediction


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    init_log()
    log("CONFIGURATION: full-sentence mode + NLI-backed FactVerifier\n")

    print("=" * 60)
    print("  TRUTHCHECK ENGINE: NLI-BACKED FULL-SENTENCE MODE")
    print("=" * 60)

    pipeline = TruthCheckPipeline()
    test_data_path = "TruthCheck/data/final_health_claims.csv"

    if not os.path.exists(test_data_path):
        print(f"CRITICAL ERROR: CSV not found at {test_data_path}")
        log(f"CRITICAL ERROR: CSV not found at {test_data_path}")
    else:
        df   = pd.read_csv(test_data_path)
        real = df[df['label'] == 0].head(500)
        fake = df[df['label'] == 1].head(500)
        df   = pd.concat([real, fake]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"[Dataset] Loaded {len(df)} claims for testing.")
        log(f"Dataset: {test_data_path} | Rows: {len(df)}\n")

        y_true = df['label'].astype(int).tolist()
        y_pred = []

        print("[Inference] Analyzing claims...")
        for i, row in df.iterrows():
            claim_text = row['claim']
            true_label = int(row['label'])

            reports    = pipeline.analyze_query(claim_text, row_index=i + 1)
            prediction = get_binary_prediction(reports)
            y_pred.append(prediction)

            correct = "✅" if prediction == true_label else "❌"
            log(f"  [RESULT] Predicted={prediction} | True={true_label} {correct}")
            log("")

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(df)}...")

        # ── Metrics ───────────────────────────────────────────────────────────
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)

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
    print(f"  DONE | Unparseable: {x} | Log: {LOG_PATH}")
    print("=" * 60)