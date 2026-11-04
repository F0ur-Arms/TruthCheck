"""Why did the pipeline answer INSUFFICIENT_EVIDENCE?  Sorts a sample of those claims by cause.

Each claim's pipeline passages (top 5, gate ignored) and a deeper pool (top_per_lane=--deep, best 10
after reranking) go to the judge. The claim lands in the first category that fits:

  scored_undecided  a passage above the gate got a clear label, yet the verdict stayed insufficient
  gate_blocked      a passage in the pipeline's top 5 got a clear label but scored below the gate
  retrieval_missed  a clear label only from the deeper pool, outside the pipeline's top 5
  no_answer_found   no clear label anywhere: content probably missing (or the judge is too cautious)

"Clear label" means SUPPORTS or REFUTES; each is checked against the claim's gold label, which shows
whether recovering these claims would be safe. review.csv holds claims and passages for a human check.
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MIN_RELEVANCE
from src.evidence_engine import EvidenceEngine

CLEAR = {"SUPPORTS", "REFUTES"}
# Predictions use 0 = true claim, 1 = false claim (src/binary_scoring.py).
CORRECT_LABEL = {0: "SUPPORTS", 1: "REFUTES"}
PIPELINE_TOP = 5
DEEP_KEEP = 10


def categorise(top5, deep):
    """Returns (category, deciding passage) for judged passages sorted by rerank score."""
    for p in top5:
        if p["label"] in CLEAR and p["score"] >= MIN_RELEVANCE:
            return "scored_undecided", p
    for p in top5:
        if p["label"] in CLEAR:
            return "gate_blocked", p
    for p in deep:
        if p["label"] in CLEAR:
            return "retrieval_missed", p
    return "no_answer_found", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", help="merged_predictions.json from run_graph_eval")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--deep", type=int, default=50, help="top_per_lane for the deeper pool")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    records = json.load(open(args.predictions))
    pool = [r for r in records if r["verdict"] == "INSUFFICIENT_EVIDENCE"]
    sample = random.Random(args.seed).sample(pool, min(args.n, len(pool)))
    print(f"{len(sample)} sampled from {len(pool)} INSUFFICIENT_EVIDENCE of {len(records)} claims", flush=True)

    engine = EvidenceEngine.from_defaults()

    def judge(claim, passages):
        def one(item):
            verdict = engine.nli_judge.verify(claim, item["passage"])["verdict"]
            return {"label": verdict, "score": item.get("rerank_score") or 0.0, "passage": item["passage"],
                    "source": engine.kb_manager.source_for_index(item.get("passage_index"))
                    if item.get("passage_index") is not None else ""}
        with ThreadPoolExecutor(6) as ex:
            return list(ex.map(one, passages))

    rows = []
    for i, rec in enumerate(sample, 1):
        claim = rec["claim_text"]
        lanes = engine.make_lanes(claim)
        shallow = engine.hybrid_retriever.retrieve_hybrid(lanes, top_per_lane=PIPELINE_TOP)
        top5 = engine.reranker.rerank(claim, shallow, top_k=min(PIPELINE_TOP, len(shallow) or PIPELINE_TOP))
        seen = {p["passage"] for p in top5}
        wide = engine.hybrid_retriever.retrieve_hybrid(lanes, top_per_lane=args.deep)
        wide = [p for p in wide if p["passage"] not in seen]
        deep = engine.reranker.rerank(claim, wide, top_k=min(DEEP_KEEP, len(wide) or DEEP_KEEP))

        top5_j, deep_j = judge(claim, top5), judge(claim, deep)
        category, passage = categorise(top5_j, deep_j)
        best = max((p["score"] for p in top5_j + deep_j), default=0.0)
        rows.append({"idx": rec["idx"], "origin": rec.get("origin"), "claim": claim, "true": rec["true"],
                     "category": category, "best_score": best,
                     "deciding": passage, "top5": top5_j, "deep": deep_j,
                     "deciding_correct": None if passage is None else passage["label"] == CORRECT_LABEL[rec["true"]]})
        if i % 10 == 0:
            print(f"processed {i}/{len(sample)}", flush=True)
            json.dump(rows, open(os.path.join(args.out, "rows.json"), "w"))

    json.dump(rows, open(os.path.join(args.out, "rows.json"), "w"))
    with open(os.path.join(args.out, "review.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "origin", "gold_true_claim", "claim", "label", "rerank_score", "source", "passage",
                    "human_says_kb_answers_claim"])
        for r in rows:
            p = r["deciding"] or max(r["top5"] + r["deep"], key=lambda x: x["score"], default=None)
            w.writerow([r["category"], r["origin"], r["true"] == 0, r["claim"],
                        p and p["label"], p and round(p["score"], 3), p and p["source"], p and p["passage"][:1500], ""])

    n = len(rows)
    print(f"\nlocal-model fallback judgments: {getattr(engine.nli_judge, 'fallback_calls', 'n/a')}")
    print(f"cause of INSUFFICIENT_EVIDENCE (n={n})")
    by = defaultdict(list)
    for r in rows:
        by[r["category"]].append(r)
    for cat in ("scored_undecided", "gate_blocked", "retrieval_missed", "no_answer_found"):
        items = by[cat]
        checked = [r["deciding_correct"] for r in items if r["deciding_correct"] is not None]
        right = f"  clear label matches gold {sum(checked)}/{len(checked)}" if checked else ""
        print(f"  {cat:18s} {len(items):4d} ({len(items) / max(n, 1):.1%}){right}")
    buckets = Counter(min(int(r["best_score"] * 5), 4) for r in rows)
    print("best rerank score per claim:",
          {f"{b / 5:.1f}-{(b + 1) / 5:.1f}": buckets[b] for b in range(5)})
    print("by origin:", {k: dict(Counter(r["category"] for r in rows if r["origin"] == k))
                         for k in sorted({str(r["origin"]) for r in rows})})


if __name__ == "__main__":
    main()
