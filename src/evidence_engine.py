"""Retrieval + NLI evidence engine shared by the legacy pipeline and the LangGraph nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.evidence_quality import ClaimEvidenceSummary, EvidenceQualityScorer
from src.hybrid_retriever import HybridRetriever, MultiLaneQueryGenerator, Reranker
from src.source_tier import tier_from_filename, tier_from_passage_text


@dataclass
class EvidenceResult:
    lanes: Dict[str, str]
    scored_passages: List[Dict[str, Any]]
    nli_support_probs: List[float]
    summary: ClaimEvidenceSummary


class EvidenceEngine:
    def __init__(self, kb_manager, hybrid_retriever, reranker, nli_judge, evidence_scorer=None,
                 query_generator=None, min_relevance: Optional[float] = None):
        self.kb_manager = kb_manager
        # Passages the reranker scores below this are dropped before NLI, so "nothing relevant
        # retrieved" ends as INSUFFICIENT_EVIDENCE instead of being judged against off-topic text.
        self.min_relevance = min_relevance
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.nli_judge = nli_judge
        self.evidence_scorer = evidence_scorer or EvidenceQualityScorer()
        self.query_generator = query_generator or MultiLaneQueryGenerator()

    @classmethod
    def from_defaults(cls) -> "EvidenceEngine":
        import os
        from config import KB_PATH
        from src.RAGpipeline.knowledge_manager import KnowledgeManager
        from src.RAGpipeline.llm_nli import FallbackNLI, LLMNLIJudge
        from src.RAGpipeline.nli_verifier import NLIVerifier

        kb_manager = KnowledgeManager()
        kb_manager.load_and_index(str(KB_PATH))
        retriever = HybridRetriever(
            passages=kb_manager.passages,
            dense_retriever_fn=kb_manager.dense_search,
        )

        # TRUTHCHECK_NLI_BACKEND: llm (LLM judge, local NLI as fallback), local (local NLI only),
        # auto (llm when an LLM endpoint is configured, else local). The LLM judge scored 65% on
        # the labelled evidence rows against 40-45% for the dedicated NLI models.
        backend = os.environ.get("TRUTHCHECK_NLI_BACKEND", "auto").lower()
        # TRUTHCHECK_NLI_PROMPT: v1 (default; with the relevance gate it matches v2 on balance and
        # answers more claims, round 4) or v2 (off-topic text is NEI, for use without the gate).
        from src.RAGpipeline.llm_nli import SYSTEM, SYSTEM_V1
        prompt = SYSTEM_V1 if os.environ.get("TRUTHCHECK_NLI_PROMPT", "v1").lower() == "v1" else SYSTEM
        judge = LLMNLIJudge(system_prompt=prompt)
        if backend == "local" or (backend == "auto" and not judge.configured):
            nli = NLIVerifier()
        else:
            from config import NLI_FALLBACK_MODEL
            nli = FallbackNLI(judge, lambda: NLIVerifier(NLI_FALLBACK_MODEL))
        from config import MIN_RELEVANCE
        min_relevance = float(os.environ.get("TRUTHCHECK_MIN_RELEVANCE", MIN_RELEVANCE))
        return cls(kb_manager, retriever, Reranker(), nli, min_relevance=min_relevance or None)

    def make_lanes(self, canonical_claim: str, subject: Optional[str] = None,
                   outcome: Optional[str] = None) -> Dict[str, str]:
        return self.query_generator.generate_query_lanes(canonical_claim, subject=subject, outcome=outcome)

    def retrieve(self, canonical_claim: str, lanes: Dict[str, str], top_per_lane: int = 5) -> List[Dict[str, Any]]:
        candidates = self.hybrid_retriever.retrieve_hybrid(lanes, top_per_lane=top_per_lane)
        reranked = self.reranker.rerank(
            canonical_claim,
            candidates,
            top_k=min(top_per_lane, len(candidates) or top_per_lane),
        )
        if self.min_relevance is None:
            return reranked
        return [p for p in reranked if (p.get("rerank_score") or 0.0) >= self.min_relevance]

    def score(self, canonical_claim: str, reranked: List[Dict[str, Any]], subject: Optional[str] = None,
              outcome: Optional[str] = None, population: Optional[str] = None):
        scored_passages: List[Dict[str, Any]] = []
        nli_support_probs: List[float] = []
        for item in reranked:
            passage = item["passage"]
            passage_index = item.get("passage_index")
            nli_result = self.nli_judge.verify(canonical_claim, passage)
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
                claim_subject=subject,
                claim_outcome=outcome,
                target_population=population,
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
        return scored_passages, nli_support_probs, self.evidence_scorer.summarize_evidence(scored_passages)

    def gather(self, canonical_claim: str, subject: Optional[str] = None, outcome: Optional[str] = None,
               population: Optional[str] = None, top_per_lane: int = 5) -> EvidenceResult:
        lanes = self.make_lanes(canonical_claim, subject, outcome)
        reranked = self.retrieve(canonical_claim, lanes, top_per_lane)
        scored, probs, summary = self.score(canonical_claim, reranked, subject, outcome, population)
        return EvidenceResult(lanes, scored, probs, summary)
