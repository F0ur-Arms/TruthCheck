"""Compare the routing prompts on a golden set: does each send claims to fact_check and only personal
questions to medical_advice?

Expected route: medical_advice for rows whose verdict is NOT_A_FACT_CHECK (personal medical-advice
requests), fact_check for every other row. Prints, per prompt and language, how many claims each prompt
wrongly sends to medical_advice (they then get no fact-check) and how many advice requests it wrongly
treats as claims. Needs the LLM server (TRUTHCHECK_LLM_BASE_URL / _MODEL).

usage: python evaluation/test_router.py GOLDEN.csv [--workers 6] [--out rows.json]
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.claims_processor import ROUTING_PROMPT, ROUTING_PROMPT_V2, ClaimsProcessor

PROMPTS = {"v1": ROUTING_PROMPT, "v2": ROUTING_PROMPT_V2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out")
    args = ap.parse_args()
    with open(args.golden, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    proc = ClaimsProcessor()
    if not proc.llm.configured:
        sys.exit("LLM not configured")

    def route(job):
        name, row = job
        try:
            return proc._call_llm_json(PROMPTS[name], row["claim_text"]).get("route", "fact_check")
        except Exception as exc:
            return f"ERROR:{type(exc).__name__}"

    results = {}
    with ThreadPoolExecutor(args.workers) as pool:
        for name in PROMPTS:
            results[name] = list(pool.map(route, [(name, r) for r in rows]))

    out = []
    for name, routes in results.items():
        errors = sum(r.startswith("ERROR") for r in routes)
        print(f"\nprompt {name} (errors {errors})")
        print("  language   claims sent to advice   advice requests sent to fact_check")
        for lang in sorted({r["language"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["language"] == lang]
            claims = [i for i in idx if rows[i]["expected_verdict"] != "NOT_A_FACT_CHECK"]
            advice = [i for i in idx if rows[i]["expected_verdict"] == "NOT_A_FACT_CHECK"]
            wrong_claims = sum(routes[i] == "medical_advice" for i in claims)
            wrong_advice = sum(routes[i] != "medical_advice" for i in advice)
            print(f"  {lang:9}  {wrong_claims:4} of {len(claims):4}             {wrong_advice:3} of {len(advice):3}")
    for i, r in enumerate(rows):
        out.append({"id": r["id"], "language": r["language"], "gold": r["expected_verdict"],
                    **{name: results[name][i] for name in results}, "claim_text": r["claim_text"]})
    changed = Counter((o["v1"], o["v2"]) for o in out if o["v1"] != o["v2"])
    print("\nroute changes v1 -> v2:", dict(changed))
    if args.out:
        json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
