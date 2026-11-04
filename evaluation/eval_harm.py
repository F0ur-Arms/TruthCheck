"""Scores the harm check on hand-labelled claims: the LLM's judgement against the keyword score.

A missed harm is the costly error, so misses are listed in full. A claim routed as a personal
medical-advice question counts as caught, because it gets the safety response.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.claims_processor import ClaimsProcessor
from src.risk_engine_v2 import TwoAxisRiskEngine, has_cessation_language


def report(name, rows, pred_key):
    harmful = [r for r in rows if r["high_harm"] == 1]
    safe = [r for r in rows if r["high_harm"] == 0]
    caught = sum(r[pred_key] for r in harmful)
    alarms = sum(r[pred_key] for r in safe)
    print(f"{name:9s} caught {caught}/{len(harmful)} harmful ({caught / max(len(harmful), 1):.1%})   "
          f"false alarms {alarms}/{len(safe)} ({alarms / max(len(safe), 1):.1%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="evaluation/golden_set/harm_labels_draft.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    processor = ClaimsProcessor()
    if not processor.llm.configured:
        sys.exit("no LLM endpoint configured: the harm judgement would not be tested")
    engine = TwoAxisRiskEngine()

    def score(row):
        rep = processor.process_query(row["claim"])
        judged = [c.high_harm for c in rep.claims]
        keyword = any(
            (a := engine.evaluate_risk(c.canonical_claim, 0.9)).has_hard_override or a.harm_axis >= engine.harm_threshold
            for c in rep.claims
        )
        return {**row, "route": rep.route, "sub_claims": [c.canonical_claim for c in rep.claims], "judged": judged,
                "no_judgement": rep.route != "medical_advice" and not any(j is not None for j in judged),
                "llm": bool(rep.route == "medical_advice" or any(judged) or has_cessation_language(row["claim"])),
                "keywords": keyword or has_cessation_language(row["claim"])}

    records = pd.read_csv(args.csv).to_dict("records")
    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(score, records))
    json.dump(rows, open(args.out, "w"), ensure_ascii=False, indent=1)

    print(f"claims {len(rows)}  decomposition fallbacks {processor.fallback_count}  "
          f"no LLM judgement {sum(r['no_judgement'] for r in rows)}  "
          f"medical-advice route {sum(r['route'] == 'medical_advice' for r in rows)}")
    report("LLM", rows, "llm")
    report("keywords", rows, "keywords")
    by_language = defaultdict(list)
    for r in rows:
        by_language[r["language"]].append(r)
    for language, items in sorted(by_language.items()):
        report(f"LLM {language}"[:9], items, "llm")
    print("\nmissed harmful claims:")
    for r in rows:
        if r["high_harm"] == 1 and not r["llm"]:
            print(f"  [{r['language']}] {r['claim']}  -> {r['sub_claims']}")
    print("\nfalse alarms:")
    for r in rows:
        if r["high_harm"] == 0 and r["llm"]:
            print(f"  [{r['language']}] {r['claim']}  route={r['route']}")


if __name__ == "__main__":
    main()
