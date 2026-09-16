"""Stage-Isolated Retrieval Evaluation Harness.

Evaluates multi-lane hybrid retrieval and cross-encoder reranking quality:
1. Recall@K (5, 10, 20), MRR, and NDCG@K per query lane (support, contradiction, guideline, population, india_context).
2. NDCG lift before vs after Cross-Encoder reranking.
"""

from __future__ import annotations

import math
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field

from src.hybrid_retriever import MultiLaneQueryGenerator, HybridRetriever, Reranker


class RetrievalLaneMetrics(BaseModel):
    lane: str
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    mrr: float
    ndcg_at_10: float


class RetrievalEvalReport(BaseModel):
    lane_metrics: List[RetrievalLaneMetrics]
    overall_mrr: float
    reranking_ndcg_before: float
    reranking_ndcg_after: float
    reranking_lift_percent: float


class RetrievalEvaluator:
    """Evaluates multi-lane hybrid retrieval and cross-encoder reranking."""

    def evaluate_dataset(
        self,
        eval_items: List[Dict[str, Any]],
        corpus_passages: List[str]
    ) -> RetrievalEvalReport:
        lanes = ["support", "contradiction", "guideline", "population", "india_context"]
        lane_recalls_5 = {l: [] for l in lanes}
        lane_recalls_10 = {l: [] for l in lanes}
        lane_recalls_20 = {l: [] for l in lanes}
        lane_mrrs = {l: [] for l in lanes}
        lane_ndcgs = {l: [] for l in lanes}

        all_mrrs = []
        ndcg_before_list = []
        ndcg_after_list = []

        retriever = HybridRetriever(passages=corpus_passages)
        reranker = Reranker()

        for item in eval_items:
            claim = item["claim"]
            gold_passages = set(item.get("relevant_passages", []))
            if not gold_passages:
                continue

            query_lanes = MultiLaneQueryGenerator.generate_query_lanes(claim)

            # Evaluate per lane
            for lane_name, query_str in query_lanes.items():
                lane_results = retriever.retrieve_hybrid({lane_name: query_str}, top_per_lane=20)
                passages_retrieved = [r["passage"] for r in lane_results]

                # Compute metrics for this lane
                r5 = self._recall_at_k(passages_retrieved, gold_passages, k=5)
                r10 = self._recall_at_k(passages_retrieved, gold_passages, k=10)
                r20 = self._recall_at_k(passages_retrieved, gold_passages, k=20)
                mrr = self._mrr(passages_retrieved, gold_passages)
                ndcg = self._ndcg_at_k(passages_retrieved, gold_passages, k=10)

                lane_recalls_5[lane_name].append(r5)
                lane_recalls_10[lane_name].append(r10)
                lane_recalls_20[lane_name].append(r20)
                lane_mrrs[lane_name].append(mrr)
                lane_ndcgs[lane_name].append(ndcg)

            # Evaluate combined candidates before vs after reranking
            combined_candidates = retriever.retrieve_hybrid(query_lanes, top_per_lane=10)
            before_passages = [c["passage"] for c in combined_candidates]
            ndcg_b = self._ndcg_at_k(before_passages, gold_passages, k=10)

            reranked = reranker.rerank(claim, combined_candidates, top_k=10)
            after_passages = [r["passage"] for r in reranked]
            ndcg_a = self._ndcg_at_k(after_passages, gold_passages, k=10)

            ndcg_before_list.append(ndcg_b)
            ndcg_after_list.append(ndcg_a)
            all_mrrs.append(self._mrr(after_passages, gold_passages))

        # Aggregate per-lane metrics
        lane_reports = []
        for lane_name in lanes:
            r5_avg = sum(lane_recalls_5[lane_name]) / max(1, len(lane_recalls_5[lane_name]))
            r10_avg = sum(lane_recalls_10[lane_name]) / max(1, len(lane_recalls_10[lane_name]))
            r20_avg = sum(lane_recalls_20[lane_name]) / max(1, len(lane_recalls_20[lane_name]))
            mrr_avg = sum(lane_mrrs[lane_name]) / max(1, len(lane_mrrs[lane_name]))
            ndcg_avg = sum(lane_ndcgs[lane_name]) / max(1, len(lane_ndcgs[lane_name]))

            lane_reports.append(RetrievalLaneMetrics(
                lane=lane_name,
                recall_at_5=round(r5_avg, 4),
                recall_at_10=round(r10_avg, 4),
                recall_at_20=round(r20_avg, 4),
                mrr=round(mrr_avg, 4),
                ndcg_at_10=round(ndcg_avg, 4)
            ))

        mean_mrr = sum(all_mrrs) / max(1, len(all_mrrs))
        avg_ndcg_b = sum(ndcg_before_list) / max(1, len(ndcg_before_list))
        avg_ndcg_a = sum(ndcg_after_list) / max(1, len(ndcg_after_list))
        lift = ((avg_ndcg_a - avg_ndcg_b) / max(1e-6, avg_ndcg_b)) * 100.0

        return RetrievalEvalReport(
            lane_metrics=lane_reports,
            overall_mrr=round(mean_mrr, 4),
            reranking_ndcg_before=round(avg_ndcg_b, 4),
            reranking_ndcg_after=round(avg_ndcg_a, 4),
            reranking_lift_percent=round(lift, 2)
        )

    def _recall_at_k(self, retrieved: List[str], gold: set, k: int) -> float:
        top_k = retrieved[:k]
        hits = sum(1 for p in top_k if p in gold)
        return hits / max(1, len(gold))

    def _mrr(self, retrieved: List[str], gold: set) -> float:
        for idx, p in enumerate(retrieved, start=1):
            if p in gold:
                return 1.0 / idx
        return 0.0

    def _ndcg_at_k(self, retrieved: List[str], gold: set, k: int) -> float:
        top_k = retrieved[:k]
        dcg = 0.0
        for idx, p in enumerate(top_k, start=1):
            rel = 1.0 if p in gold else 0.0
            dcg += rel / math.log2(idx + 1)

        idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, min(k, len(gold)) + 1))
        if idcg == 0:
            return 0.0
        return dcg / idcg
