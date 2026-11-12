"""Retrieval + NLI evidence engine shared by the legacy pipeline and the LangGraph nodes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.evidence_quality import ClaimEvidenceSummary, EvidenceQualityScorer
from src.hybrid_retriever import HybridRetriever, MultiLaneQueryGenerator, Reranker
from src.source_tier import source_url, tier_from_filename, tier_from_passage_text


@dataclass
class EvidenceResult:
    lanes: Dict[str, str]
    scored_passages: List[Dict[str, Any]]
    nli_support_probs: List[float]
    summary: ClaimEvidenceSummary


class EvidenceEngine:
    def __init__(self, kb_manager, hybrid_retriever, reranker, nli_judge, evidence_scorer=None,
                 query_generator=None, min_relevance: Optional[float] = None, excluded_sources=()):
        self.kb_manager = kb_manager
        # Passages the reranker scores below this are dropped before NLI, so "nothing relevant
        # retrieved" ends as INSUFFICIENT_EVIDENCE instead of being judged against off-topic text.
        self.min_relevance = min_relevance
        self.excluded_sources = set(excluded_sources)
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.nli_judge = nli_judge
        self.evidence_scorer = evidence_scorer or EvidenceQualityScorer()
        self.query_generator = query_generator or MultiLaneQueryGenerator()

    @classmethod
    def from_defaults(cls) -> "EvidenceEngine":
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
        # answers more claims, round 4), v2 (off-topic text is NEI, for use without the gate) or v3
        # (less ready to answer NEI; judge tests only so far, evaluation/baselines/round9_judge_prompt) or v4
        # (v3 + named-event and yes/no-question rules from the round 19 error analysis).
        from src.RAGpipeline.llm_nli import SYSTEM, SYSTEM_V1, SYSTEM_V3, SYSTEM_V4
        prompt = {"v1": SYSTEM_V1, "v2": SYSTEM, "v3": SYSTEM_V3, "v4": SYSTEM_V4}[os.environ.get("TRUTHCHECK_NLI_PROMPT", "v1").lower()]
        judge = LLMNLIJudge(system_prompt=prompt)
        if backend == "local" or (backend == "auto" and not judge.configured):
            nli = NLIVerifier()
        else:
            from config import NLI_FALLBACK_MODEL
            nli = FallbackNLI(judge, lambda: NLIVerifier(NLI_FALLBACK_MODEL))
        from config import MIN_RELEVANCE
        min_relevance = float(os.environ.get("TRUTHCHECK_MIN_RELEVANCE", MIN_RELEVANCE))
        # TRUTHCHECK_EXCLUDE_SOURCES: comma-separated knowledge-base files never to use as evidence;
        # a trailing * matches a prefix (e.g. "toi_*" turns off every Times of India file).
        excluded = [f for f in os.environ.get("TRUTHCHECK_EXCLUDE_SOURCES", "").split(",") if f]
        return cls(kb_manager, retriever, Reranker(), nli, min_relevance=min_relevance or None,
                   excluded_sources=excluded)

    def _excluded(self, source_file: Optional[str]) -> bool:
        """An entry ending in "*" excludes every file with that prefix (e.g. "toi_*")."""
        if not source_file:
            return False
        return any(source_file.startswith(e[:-1]) if e.endswith("*") else source_file == e
                   for e in self.excluded_sources)

    def make_lanes(self, canonical_claim: str, subject: Optional[str] = None,
                   outcome: Optional[str] = None) -> Dict[str, str]:
        # TRUTHCHECK_QUERY_LANES: v1 (default, templated lanes) or v2 (claim, subject, subject+outcome,
        # content words; see MultiLaneQueryGenerator.generate_plain_lanes).
        if os.environ.get("TRUTHCHECK_QUERY_LANES", "v1").lower() == "v2" and hasattr(
                self.query_generator, "generate_plain_lanes"):
            return self.query_generator.generate_plain_lanes(canonical_claim, subject=subject, outcome=outcome)
        return self.query_generator.generate_query_lanes(canonical_claim, subject=subject, outcome=outcome)

    def retrieve(self, canonical_claim: str, lanes: Dict[str, str], top_per_lane: int = 5) -> List[Dict[str, Any]]:
        # Wider retrieval is opt-in: TRUTHCHECK_RETRIEVAL_POOL candidates per lane, TRUTHCHECK_RERANK_KEEP judged.
        keep = int(os.environ.get("TRUTHCHECK_RERANK_KEEP", top_per_lane))
        top_per_lane = int(os.environ.get("TRUTHCHECK_RETRIEVAL_POOL", top_per_lane))
        candidates = self.hybrid_retriever.retrieve_hybrid(lanes, top_per_lane=top_per_lane)
        if self.excluded_sources:
            candidates = [c for c in candidates if c.get("passage_index") is None
                          or not self._excluded(self.kb_manager.source_for_index(c["passage_index"]))]
        reranked = self.reranker.rerank(
            canonical_claim,
            candidates,
            top_k=min(keep, len(candidates) or keep),
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
                source_tier=source_tier,
            )
            scored_passages.append({
                "passage": passage,
                "passage_index": passage_index,
                "source_tier": source_tier,
                "source_file": source_file,
                "source_url": source_url(source_file),
                "nli_verdict": nli_result["verdict"],
                "quality_score": pq.overall_quality_score,
                "supports_probability": nli_result["supports_probability"],
                "refutes_probability": nli_result["refutes_probability"],
                "rerank_score": item.get("rerank_score"),
            })
            nli_support_probs.append(nli_result["supports_probability"])
        combined = self._judge_combined(canonical_claim, scored_passages)
        if combined:
            scored_passages.append(combined)
            nli_support_probs.append(combined["supports_probability"])
        return scored_passages, nli_support_probs, self.evidence_scorer.summarize_evidence(scored_passages)

    def _judge_combined(self, canonical_claim: str, scored: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """TRUTHCHECK_NLI_COMBINE=k: when every passage alone was NEI, judge the k most relevant together.

        A page is split into passages, so the answer can be spread over neighbouring chunks that each
        look inconclusive. The extra judgment is added only when it is SUPPORTS or REFUTES, so claims
        with no usable evidence are unchanged. It cites the most relevant of the passages it read and
        lists all of them in combined_from.
        """
        k = int(os.environ.get("TRUTHCHECK_NLI_COMBINE", "0") or 0)
        if k < 2 or len(scored) < 2 or any(p["nli_verdict"] != "NEI" for p in scored):
            return None
        top = sorted(scored, key=lambda p: p.get("rerank_score") or 0.0, reverse=True)[:k]
        budget = 3000 // len(top)  # the judge reads at most 3,000 characters
        text = "\n\n".join(p["passage"][:budget] for p in top)
        result = self.nli_judge.verify(canonical_claim, text)
        if result["verdict"] == "NEI":
            return None
        lead = top[0]
        return {
            "passage": text,
            "passage_index": lead["passage_index"],
            "source_tier": lead["source_tier"],
            "source_file": lead["source_file"],
            "source_url": lead["source_url"],
            "nli_verdict": result["verdict"],
            "quality_score": sum(p["quality_score"] for p in top) / len(top),
            "supports_probability": result["supports_probability"],
            "refutes_probability": result["refutes_probability"],
            "rerank_score": lead.get("rerank_score"),
            "combined_from": [p["source_file"] for p in top],
        }

    def gather(self, canonical_claim: str, subject: Optional[str] = None, outcome: Optional[str] = None,
               population: Optional[str] = None, top_per_lane: int = 5) -> EvidenceResult:
        lanes = self.make_lanes(canonical_claim, subject, outcome)
        reranked = self.retrieve(canonical_claim, lanes, top_per_lane)
        scored, probs, summary = self.score(canonical_claim, reranked, subject, outcome, population)
        return EvidenceResult(lanes, scored, probs, summary)
