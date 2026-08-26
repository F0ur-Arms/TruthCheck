import sys
import os
import time

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import spacy
import pandas as pd
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from src.preprocessor import HinglishMapper
from src.linguistic_scorer import LinguisticScorer
from src.verifiernew import FactVerifier
from src.risk_engine import RiskEngine
from src.RAGpipeline.nli_verifier import NLIVerifier
from src.RAGpipeline.knowledge_manager import KnowledgeManager
from src.claims_processor import ClaimsProcessor, SubClaim
from src.hybrid_retriever import MultiLaneQueryGenerator, HybridRetriever, Reranker
from src.evidence_quality import EvidenceQualityScorer, ClaimEvidenceSummary
from src.calibration import ConfidenceCalibrator
from src.operating_mode import (
    decide_mode,
    detect_harm_signal_placeholder,
    should_escalate_to_deep,
    escalation_reason,
)
from src.source_tier import tier_from_filename, tier_from_passage_text
from config import DEBUG_LOG_PATH, EMBEDDING_MODEL, EVALUATION_DATASET, FACTS_JSON, KB_PATH, LOG_PATH
from src.refine_extractor import extract_triples
# ORIGINAL LOGGER
# ─────────────────────────────────────────────────────────────────────────────
LOG_PATH = str(LOG_PATH)

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
DEBUG_LOG_PATH = str(DEBUG_LOG_PATH)

def init_debug_log():
    os.makedirs(os.path.dirname(DEBUG_LOG_PATH), exist_ok=True)
    # Using 'w' to overwrite and give us a fresh clean log each run
    with open(DEBUG_LOG_PATH, "w", encoding="utf-8") as f:
        f.write(f"DEBUG RUN: SCORER & VERIFIER ISOLATION — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 100 + "\n")

def debug_log(msg=""):
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Calibrated verdict taxonomy → binary eval mapping
_FAKE_VERDICTS = frozenset({"FALSE", "MISLEADING", "MOSTLY_FALSE"})
_REAL_VERDICTS = frozenset({"SUPPORTED", "MOSTLY_SUPPORTED"})

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
        print("--- Initializing FactVerifier with built-in FAISS ---")
        self.verifier = FactVerifier(
            facts_path=str(FACTS_JSON),
            nli_verifier=self.nli_judge,
            embedding_model=EMBEDDING_MODEL,
        )

        # ── RAG / hybrid retrieval setup ─────────────────────────────────────
        self.kb_manager = KnowledgeManager()
        self.kb_manager.load_and_index(str(KB_PATH))
        self.query_generator = MultiLaneQueryGenerator()
        self.reranker = Reranker()
        self.hybrid_retriever = HybridRetriever(
            passages=self.kb_manager.passages,
            dense_retriever_fn=self.kb_manager.dense_search,
        )
        self.evidence_scorer = EvidenceQualityScorer()
        self.calibrator = ConfidenceCalibrator()

        # ── Step 3: spaCy & Claims Processor ─────────────────────────────────
        self.nlp = spacy.load("en_core_web_sm")
        self.claims_processor = ClaimsProcessor()

        print("--- Pipeline Ready ---\n")

    def _sub_claims_for_query(self, claim_rep, clean_text):
        if claim_rep.claims:
            return claim_rep.claims
        doc = self.nlp(clean_text)
        return [
            SubClaim(claim_id=f"c{idx}", canonical_claim=sent.text.strip())
            for idx, sent in enumerate(doc.sents, 1)
            if sent.text.strip()
        ]

    def _kb_truth_from_tier1(self, tier1_result):
        truth = tier1_result.get("truth") or ""
        if " [NLI:" in truth:
            return truth.split(" [NLI:")[0].strip()
        if truth.startswith("Unverified:"):
            return ""
        return truth.strip()

    def _build_fast_evidence_summary(self, tier1_result, sub_claim):
        kb_truth = self._kb_truth_from_tier1(tier1_result)
        if not kb_truth:
            kb_truth = tier1_result.get("matched_entry") or sub_claim.canonical_claim
        pq = self.evidence_scorer.score_passage(
            kb_truth,
            claim_subject=sub_claim.subject,
            claim_outcome=sub_claim.outcome,
            target_population=sub_claim.population,
        )
        nli_conf = tier1_result.get("nli_confidence")
        if nli_conf is None:
            nli_conf = 0.5
        return ClaimEvidenceSummary(
            best_tier="guideline",
            tier_diversity=["guideline"],
            agreement_ratio=1.0,
            contradiction_count=0,
            support_count=1,
            mean_quality_score=pq.overall_quality_score,
        ), [float(nli_conf)]

    def _run_hybrid_evidence_pass(
        self,
        sub_claim,
        top_per_lane,
        log_prefix="",
    ):
        lanes = self.query_generator.generate_query_lanes(
            sub_claim.canonical_claim,
            subject=sub_claim.subject,
            outcome=sub_claim.outcome,
        )
        log(f"  {log_prefix}[HYBRID] Query lanes: {list(lanes.keys())}")
        for lane_name, lane_query in lanes.items():
            log(f"  {log_prefix}[HYBRID]   lane={lane_name}: {lane_query[:120]}")

        candidates = self.hybrid_retriever.retrieve_hybrid(lanes, top_per_lane=top_per_lane)
        reranked = self.reranker.rerank(
            sub_claim.canonical_claim,
            candidates,
            top_k=min(top_per_lane, len(candidates) or top_per_lane),
        )

        scored_passages = []
        nli_support_probs = []
        for rank, item in enumerate(reranked, 1):
            passage = item["passage"]
            passage_index = item.get("passage_index")
            nli_result = self.nli_judge.verify(sub_claim.canonical_claim, passage)
            source_file = (
                self.kb_manager.source_for_index(passage_index)
                if passage_index is not None
                else ""
            )
            source_tier = tier_from_filename(source_file)
            if source_tier == "journalism" and not source_file:
                source_tier = tier_from_passage_text(passage)

            pq = self.evidence_scorer.score_passage(
                passage,
                claim_subject=sub_claim.subject,
                claim_outcome=sub_claim.outcome,
                target_population=sub_claim.population,
            )
            scored_passages.append({
                "passage": passage,
                "passage_index": passage_index,
                "source_tier": source_tier,
                "source_file": source_file,
                "nli_verdict": nli_result["verdict"],
                "quality_score": pq.overall_quality_score,
                "supports_probability": nli_result["supports_probability"],
                "refutes_probability": nli_result["refutes_probability"],
                "rerank_score": item.get("rerank_score"),
            })
            nli_support_probs.append(nli_result["supports_probability"])
            log(
                f"  {log_prefix}[HYBRID]   rerank#{rank} idx={passage_index} "
                f"tier={source_tier} nli={nli_result['verdict']} "
                f"sup_p={nli_result['supports_probability']:.3f} "
                f"passage={passage[:100]}"
            )

        evidence_summary = self.evidence_scorer.summarize_evidence(scored_passages)
        return evidence_summary, scored_passages, nli_support_probs, lanes

    def analyze_query(self, raw_text, row_index=None, true_label=None):
        t0 = time.perf_counter()
        clean_text = self.mapper.clean_text(raw_text)

        log(f"{'─' * 70}")
        log(f"ROW #{row_index}  INPUT : {raw_text}")
        log(f"              CLEANED: {clean_text}")
        log(f"{'─' * 70}")
        debug_log(f"\n[ROW {row_index} | TRUE LABEL: {true_label}]")
        debug_log(f"CLAIM: {raw_text}")

        claim_rep = self.claims_processor.process_query(raw_text, clean_text)
        if claim_rep.route == "medical_advice":
            log("  [ROUTE GATE] Personal Medical Advice detected -> Bypassing fact check")
            return [{
                "input": raw_text,
                "verification_text": raw_text,
                "claim_triples": [],
                "verdict": "NOT_A_FACT_CHECK",
                "route": "medical_advice",
                "safety_response": claim_rep.safety_response,
                "explanation": claim_rep.safety_response,
                "source": "Medical Safety Advice Gate",
                "risk_score": 1.0,
                "legacy_risk_score": 1.0,
                "risk_level": "High (Medical Advice Request)",
                "operating_mode": "fast",
                "escalated": False,
                "latency_sec": round(time.perf_counter() - t0, 4),
            }]

        harm_signal = detect_harm_signal_placeholder(raw_text)
        sub_claims = self._sub_claims_for_query(claim_rep, clean_text)
        final_reports = []

        for sub_idx, sub_claim in enumerate(sub_claims, 1):
            canonical = sub_claim.canonical_claim
            triples = extract_triples(canonical, self.nlp)
            verification_text = canonical
            if triples:
                first = triples[0]
                verification_text = f"{first['subject']} {first['relation']} {first['object']}"

            log(f"\n  [SUB-CLAIM {sub_idx}] {canonical}")
            log(f"  [CLAIM TRIPLES] {triples}")

            style_score = self.scorer.calculate_score(canonical)
            log(f"  [STYLE SCORE] {style_score}")

            tier1_result = self.verifier.tier1_lookup(canonical)
            mode = decide_mode(tier1_result)
            operating_mode = mode
            escalated = False

            log(f"  [MODE] decided: {mode} (match_type={tier1_result.get('match_type')}, "
                f"nli_conf={tier1_result.get('nli_confidence')})")
            if harm_signal:
                log("  [MODE] harm_signal_placeholder=True")

            fact_result = self.verifier.verify(verification_text)

            if mode == "fast":
                evidence_summary, nli_probs = self._build_fast_evidence_summary(tier1_result, sub_claim)
                log("  [MODE] fast path — Tier-1 evidence only, skipping hybrid retrieval")
            else:
                top_per_lane = 5
                evidence_summary, scored_passages, nli_probs, lanes = self._run_hybrid_evidence_pass(
                    sub_claim, top_per_lane=top_per_lane, log_prefix=""
                )
                if should_escalate_to_deep(evidence_summary, harm_signal):
                    reason = escalation_reason(evidence_summary, harm_signal)
                    log(f"  [MODE] escalated standard -> deep: reason={reason}")
                    escalated = True
                    operating_mode = "deep"
                    top_per_lane = 15
                    evidence_summary, scored_passages, nli_probs, lanes = self._run_hybrid_evidence_pass(
                        sub_claim, top_per_lane=top_per_lane, log_prefix="[DEEP] "
                    )

            has_population_split = bool(
                sub_claim.population
                and sub_claim.population_confidence not in ("unspecified", "", None)
            )
            calibrated = self.calibrator.calibrate(
                evidence_summary=evidence_summary,
                top_nli_probabilities=nli_probs,
                population_match_score=0.75,
                has_population_split=has_population_split,
            )

            log(f"  [CALIBRATION] verdict={calibrated.verdict} "
                f"confidence={calibrated.calibrated_confidence:.4f}")

            log(f"  [KB VERIFIER]")
            log(f"    Match Found   : {fact_result.get('match_found', False)}")
            log(f"    Matched Entry : {fact_result.get('matched_entry', 'None')}")
            log(f"    Match Type    : {fact_result.get('match_type', 'N/A')}")
            log(f"    Match Score   : {fact_result.get('match_score', 0.0)}")
            log(f"    KB Verdict    : {fact_result.get('verdict')}")
            log(f"    NLI Verdict   : {fact_result.get('nli_verdict')}")
            log(f"    NLI Confidence: {fact_result.get('nli_confidence')}")
            log(f"    Truth         : {fact_result.get('truth', '')[:120]}")

            risk = self.engine.calculate_risk(canonical, style_score, fact_result)

            log(f"  [RISK ENGINE]")
            log(f"    Final Score : {risk['score']}")
            log(f"    Risk Label  : {risk['label']}")
            bd = risk['breakdown']
            log(f"    Breakdown:")
            log(f"      fact_verdict : {bd['fact_verdict']}")
            log(f"      fact_impact  : {bd['fact_impact']}  (weight={bd['weights_used']['fact']})")
            log(f"      ml_impact    : {bd['ml_impact']}    (weight={bd['weights_used']['ml']})")
            log(f"      style_impact : {bd['style_impact']} (weight={bd['weights_used']['style']})")
            log(f"  [MODE] operating_mode={operating_mode} escalated={escalated}")

            latency_sec = round(time.perf_counter() - t0, 4)

            report = {
                "input": canonical,
                "verification_text": verification_text,
                "claim_triples": triples,
                "verdict": calibrated.verdict,
                "calibrated_confidence": calibrated.calibrated_confidence,
                "legacy_kb_verdict": fact_result.get("verdict"),
                "nli_verdict": fact_result.get("nli_verdict"),
                "explanation": calibrated.explanation_summary,
                "source": fact_result.get("source") or "TruthCheck v2 Pipeline",
                "risk_score": risk["score"],
                "legacy_risk_score": risk["score"],
                "risk_level": risk["label"],
                "operating_mode": operating_mode,
                "escalated": escalated,
                "latency_sec": latency_sec,
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
        return 1

    for r in reports:
        verdict = r.get("verdict", "")
        if verdict in _FAKE_VERDICTS or verdict == "FALSE":
            return 1
        if verdict in _REAL_VERDICTS or verdict == "TRUE":
            return 0

    max_risk = max(r.get("legacy_risk_score", r.get("risk_score", 0)) for r in reports)
    return 1 if max_risk > risk_threshold else 0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    init_log()
    init_debug_log()
    log("CONFIGURATION: Built-in FAISS mode + NLI-backed FactVerifier + gated hybrid retrieval\n")

    print("=" * 60)
    print("  TRUTHCHECK ENGINE: BUILT-IN FAISS + NLI MODE")
    print("=" * 60)

    pipeline = TruthCheckPipeline()
    test_data_path = str(EVALUATION_DATASET)

    if not os.path.exists(test_data_path):
        print(f"CRITICAL ERROR: CSV not found at {test_data_path}")
        log(f"CRITICAL ERROR: CSV not found at {test_data_path}")
    else:
        df = pd.read_csv(test_data_path)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"[Dataset] Loaded {len(df)} claims for testing.")
        log(f"Dataset: {test_data_path} | Rows: {len(df)}\n")

        y_true = df["label"].astype(int).tolist()
        y_pred = []
        mode_counts = {"fast": 0, "standard": 0, "deep": 0}
        mode_latencies = {"fast": [], "standard": [], "deep": []}
        mode_correct = {"fast": 0, "standard": 0, "deep": 0}
        mode_totals = {"fast": 0, "standard": 0, "deep": 0}

        print("[Inference] Analyzing claims...")
        for i, row in df.iterrows():
            claim_text = row["claim"]
            true_label = int(row["label"])

            reports = pipeline.analyze_query(claim_text, row_index=i + 1, true_label=true_label)
            prediction = get_binary_prediction(reports)
            y_pred.append(prediction)

            claim_mode = reports[0].get("operating_mode", "standard") if reports else "standard"
            claim_latency = sum(r.get("latency_sec", 0) for r in reports if "error" not in r)
            mode_counts[claim_mode] = mode_counts.get(claim_mode, 0) + 1
            mode_latencies.setdefault(claim_mode, []).append(claim_latency)
            mode_totals[claim_mode] = mode_totals.get(claim_mode, 0) + 1
            if prediction == true_label:
                mode_correct[claim_mode] = mode_correct.get(claim_mode, 0) + 1

            correct = "✅" if prediction == true_label else "❌"
            log(f"  [RESULT] mode={claim_mode} Predicted={prediction} | True={true_label} {correct}")
            log("")

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(df)}...")

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        total_claims = len(df)
        mode_split_lines = []
        for m in ("fast", "standard", "deep"):
            count = mode_counts.get(m, 0)
            pct = 100.0 * count / total_claims if total_claims else 0
            lats = mode_latencies.get(m, [])
            avg_lat = sum(lats) / len(lats) if lats else 0.0
            acc_m = mode_correct.get(m, 0) / mode_totals[m] if mode_totals.get(m) else 0.0
            mode_split_lines.append(
                f"  {m:8s}: {count:4d} claims ({pct:5.1f}%) | "
                f"accuracy={acc_m:.4f} | avg_latency={avg_lat:.3f}s"
            )

        metrics_str = (
            f"\n{'=' * 60}\n"
            f"  EVALUATION RESULTS\n"
            f"{'=' * 60}\n"
            f"  Accuracy  : {acc:.4f}\n"
            f"  Precision : {prec:.4f}\n"
            f"  Recall    : {rec:.4f}\n"
            f"  F1 Score  : {f1:.4f}\n"
            f"\n  Mode traffic split & per-mode metrics:\n"
            + "\n".join(mode_split_lines)
            + f"\n\n{classification_report(y_true, y_pred, target_names=['Real (0)', 'Fake (1)'])}\n"
            f"  Unparseable claims (error fallback): {x}\n"
        )
        print(metrics_str)
        log(metrics_str)

    print("=" * 60)
    print(f"  DONE | Unparseable: {x} | Main Log: {LOG_PATH}")
    print(f"  DEBUG LOG SAVED TO: {DEBUG_LOG_PATH}")
    print("=" * 60)
