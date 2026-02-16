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


if __name__ == "__main__":
    pipeline = TruthCheckPipeline()

    user_input = (
        "Subah khali pet drinking warm water improves digestion and boosts metabolism. "
        "Many people also believe it helps in detoxifying the body and improving skin health. "
        "However, some individuals simply enjoy starting their day with a glass of warm water out of habit. "
        "It is a common routine in many households and is often recommended by elders. "
        "Whether or not someone chooses to follow this practice usually depends on personal preference and lifestyle."
    )

    results = pipeline.analyze_query(user_input)

    for r in results:
        print(f"REPORT FOR: '{r['input']}'")
        print(f"Claim: {r['extracted_claim']}")
        print(f"Verdict: {r['verdict']}")
        print(f"Source: {r['source']}")
        print(f"Risk: {r['risk_score']} ({r['risk_level']})")
        print(f"Scientific Truth: {r['explanation']}")
        print("-" * 60)
