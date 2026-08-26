"""Phase 4: Hybrid Retrieval & Reranking Module.

Implements:
1. Multi-lane Adversarial Query Generation (support, contradiction, guideline, population, India-context).
2. Hybrid Retrieval (BM25 + Dense FAISS) with Reciprocal Rank Fusion (RRF).
3. Cross-Encoder Reranking (BAAI/bge-reranker-v2-m3 with CrossEncoder fallback).
"""

from __future__ import annotations

import re
import numpy as np
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from config import RERANKER_MODEL


class MultiLaneQueryGenerator:
    """Generates 5 adversarial query variants per claim to prevent confirmation bias."""

    @staticmethod
    def generate_query_lanes(claim_text: str, subject: Optional[str] = None, outcome: Optional[str] = None) -> Dict[str, str]:
        text_clean = claim_text.strip()
        sub = subject or text_clean
        out = outcome or "health safety evidence"

        return {
            "support": f"{text_clean}",
            "contradiction": f"{sub} does not cause {out} safe healthy normal evidence",
            "guideline": f"{sub} {out} clinical guideline WHO ICMR NIH advice",
            "population": f"{sub} {out} CKD patients elderly pregnant children guideline",
            "india_context": f"{sub} {out} ICMR NIN FSSAI Indian diet guidelines",
        }


class HybridRetriever:
    """Combines BM25 lexical search with FAISS dense vector search using Reciprocal Rank Fusion."""

    def __init__(self, passages: List[str], dense_retriever_fn, k: int = 60):
        self.passages = passages
        self.dense_retriever_fn = dense_retriever_fn
        self.k = k

        # Tokenize passages for BM25
        tokenized_corpus = [re.findall(r"\w+", p.lower()) for p in passages]
        self.bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def retrieve_hybrid(self, query_lanes: Dict[str, str], top_per_lane: int = 15) -> List[Dict[str, Any]]:
        if not self.passages:
            return []

        rrf_scores: Dict[int, float] = {}
        passage_lane_map: Dict[int, List[str]] = {}

        for lane_name, query_str in query_lanes.items():
            # 1. Lexical BM25 ranking
            tokenized_query = re.findall(r"\w+", query_str.lower())
            if self.bm25 and tokenized_query:
                bm25_scores = self.bm25.get_scores(tokenized_query)
                bm25_indices = np.argsort(bm25_scores)[::-1][:top_per_lane]
                for rank, idx in enumerate(bm25_indices):
                    idx = int(idx)
                    rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (self.k + rank + 1))
                    passage_lane_map.setdefault(idx, []).append(lane_name)

            # 2. Dense Vector FAISS ranking
            try:
                dense_hits = self.dense_retriever_fn(query_str, top_k=top_per_lane)
                for rank, hit in enumerate(dense_hits):
                    idx = None
                    if isinstance(hit, dict):
                        idx = hit.get("index")
                        if idx is None and "passage_index" in hit:
                            idx = hit["passage_index"]
                    elif isinstance(hit, (list, tuple)) and len(hit) >= 2:
                        idx = int(hit[0])
                    elif isinstance(hit, int):
                        idx = hit
                    elif isinstance(hit, str) and hit in self.passages:
                        # Legacy string-only hits (deprecated — prefer index tuples)
                        idx = self.passages.index(hit)

                    if idx is not None and 0 <= idx < len(self.passages):
                        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (self.k + rank + 1))
                        passage_lane_map.setdefault(idx, []).append(lane_name)
            except Exception:
                pass

        # Sort candidate indices by RRF score
        sorted_candidates = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

        results = []
        for idx in sorted_candidates:
            results.append({
                "passage_index": idx,
                "passage": self.passages[idx],
                "rrf_score": round(rrf_scores[idx], 5),
                "query_lanes": list(set(passage_lane_map.get(idx, [])))
            })

        return results


class Reranker:
    """Reranks candidate evidence passages against the canonical claim using Cross-Encoder scoring."""

    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model_name = model_name
        self.cross_encoder = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
            # The pipeline must remain usable in offline CI/runtime environments.
            # If the model is absent locally, the existing lexical fallback below
            # is deliberately used instead of attempting a network download.
            self.cross_encoder = CrossEncoder(self.model_name, local_files_only=True)
            print(f"[Reranker] Loaded CrossEncoder model: {self.model_name}")
        except Exception as e:
            print(f"[Reranker] Warning: Could not load CrossEncoder {self.model_name} ({e}). Using lexical fallback.")
            self.cross_encoder = None

    def rerank(self, claim_text: str, candidate_passages: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not candidate_passages:
            return []

        passages_text = [item["passage"] for item in candidate_passages]

        if self.cross_encoder:
            pairs = [[claim_text, p] for p in passages_text]
            scores = self.cross_encoder.predict(pairs)
            for idx, item in enumerate(candidate_passages):
                item["rerank_score"] = float(scores[idx])
        else:
            # Simple term overlap fallback if cross encoder isn't available
            claim_words = set(re.findall(r"\w+", claim_text.lower()))
            for item in candidate_passages:
                p_words = set(re.findall(r"\w+", item["passage"].lower()))
                overlap = len(claim_words.intersection(p_words))
                item["rerank_score"] = float(overlap) / (len(claim_words) + 1.0)

        # Sort by rerank score
        reranked = sorted(candidate_passages, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return reranked[:top_k]
