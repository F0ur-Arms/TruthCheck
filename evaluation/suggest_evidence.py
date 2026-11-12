"""Suggest knowledge-base passages for each golden-set claim, to speed up annotation.

For every row of a golden-set CSV, the pipeline's own retrieval (query lanes, BM25 + dense search,
cross-encoder rerank) finds the most relevant knowledge-base passages and writes them to a CSV the
annotator reads beside golden_v1.csv. These are SUGGESTIONS: no verdict is formed, nothing is written
into the golden set, and the annotator decides whether a passage settles the claim (GUIDELINES.md
rules 2 and 3). The relevance gate is off here, so weak matches show up with a low score instead of
being hidden.

Runs on a CPU (no LLM needed). It refuses to run if the cross-encoder cannot be loaded, because the
reranker's lexical fallback would give poor suggestions without saying so.

usage: python evaluation/suggest_evidence.py GOLDEN.csv OUT.csv [--top 3]
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import KB_PATH
from src.evidence_engine import EvidenceEngine
from src.hybrid_retriever import HybridRetriever, MultiLaneQueryGenerator, Reranker
from src.RAGpipeline.knowledge_manager import KnowledgeManager
from src.source_tier import source_url

COLUMNS = ["id", "language", "claim_text", "rank", "rerank_score", "source_file", "source_url", "passage"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("out")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    with open(args.golden, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))[: args.limit]

    # Window passages (~90 words, page title in front) read better as suggestions than single lines.
    os.environ.setdefault("TRUTHCHECK_KB_CHUNKING", "window")
    kb = KnowledgeManager()
    kb.load_and_index(str(KB_PATH))
    if not kb.kb_loaded or not kb.passages:
        sys.exit("knowledge base did not load")
    reranker = Reranker()
    if reranker.cross_encoder is None:
        sys.exit("cross-encoder not loaded; refusing to write suggestions from the lexical fallback")
    retriever = HybridRetriever(passages=kb.passages, dense_retriever_fn=kb.dense_search)
    engine = EvidenceEngine(kb, retriever, reranker, nli_judge=None, min_relevance=None,
                            query_generator=MultiLaneQueryGenerator())
    os.environ.setdefault("TRUTHCHECK_QUERY_LANES", "v2")
    os.environ.setdefault("TRUTHCHECK_RETRIEVAL_POOL", "20")
    os.environ["TRUTHCHECK_RERANK_KEEP"] = str(args.top)

    t0 = time.time()
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for n, row in enumerate(rows, 1):
            claim = row["claim_text"]
            hits = engine.retrieve(claim, engine.make_lanes(claim), top_per_lane=args.top)
            for rank, hit in enumerate(hits, 1):
                source = kb.source_for_index(hit["passage_index"]) if hit.get("passage_index") is not None else ""
                writer.writerow({"id": row["id"], "language": row.get("language", ""), "claim_text": claim,
                                 "rank": rank, "rerank_score": f"{hit.get('rerank_score') or 0:.3f}",
                                 "source_file": source, "source_url": source_url(source) or "",
                                 "passage": hit["passage"]})
            if n % 25 == 0:
                f.flush()
                print(f"{n}/{len(rows)} claims ({time.time() - t0:.0f}s)", flush=True)
    print(f"wrote suggestions for {len(rows)} claims to {args.out}")


if __name__ == "__main__":
    main()
