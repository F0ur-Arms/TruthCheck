"""Score the LangGraph pipeline on a labelled claim set, sharded like main.py.

Default set: data/final_health_claims.csv (label 1 = fake). With --public, the TRUE/FALSE rows of
golden_v1.csv from the cleaner public datasets are used instead (FALSE = 1). --details writes the
sub-claim verdicts and the passages behind each answer for error analysis. With --golden PATH, a
golden set in the GUIDELINES.md format is scored instead: SUPPORTED / SUPPORTED_WITH_CAVEATS rows count
as true and FALSE / MOSTLY_FALSE / MISLEADING rows as false; rows that expect an abstention are left
out. --languages hi,hinglish keeps only those languages, and each record keeps its language.
"""

import argparse
import re
import json
import os
import sys
import time

os.environ.setdefault("TRUTHCHECK_CHECKPOINT_DB", ":memory:")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import EVALUATION_DATASET
import graph.nodes as nodes
from graph.workflow import create_truthcheck_graph
from src.binary_scoring import get_binary_prediction, score_with_abstain

PUBLIC_CSV = "evaluation/golden_set/golden_v1.csv"
# COVID-Fact and raw_claims.json fail the style-artifact check (evaluation/artifact_check.py).
PUBLIC_ORIGINS = "scifact_dataset,healthver_dataset,pubhealth_dataset,ai4bharat_dataset"


# Golden-set verdicts with a true/false meaning; INSUFFICIENT_EVIDENCE, MIXED_EVIDENCE and
# NOT_A_FACT_CHECK rows expect an abstention and are not part of the binary score.
GOLD_FAKE = {"FALSE", "MOSTLY_FALSE", "MISLEADING"}
GOLD_REAL = {"SUPPORTED", "SUPPORTED_WITH_CAVEATS"}


def load(args) -> pd.DataFrame:
    if args.golden:
        # keep_default_na=False: the file uses the string "NA" for empty fields, not missing values.
        df = pd.read_csv(args.golden, keep_default_na=False, encoding="utf-8-sig")
        df = df[df["expected_verdict"].isin(GOLD_FAKE | GOLD_REAL)]
        if args.languages:
            df = df[df["language"].str.lower().isin(re.split(r"[,:]", args.languages.lower()))]
        df = pd.DataFrame({"claim": df["claim_text"], "label": df["expected_verdict"].isin(GOLD_FAKE).astype(int),
                           "origin": df["claim_origin"], "language": df["language"].str.lower(),
                           "row_id": df["id"]})
    elif args.public:
        df = pd.read_csv(PUBLIC_CSV)
        df = df[df["expected_verdict"].isin(["TRUE", "FALSE"]) & df["claim_origin"].isin(re.split(r"[,:]", args.origins))]
        df = pd.DataFrame({"claim": df["claim_text"], "label": (df["expected_verdict"] == "FALSE").astype(int),
                           "origin": df["claim_origin"], "row_id": df["id"]})
    else:
        df = pd.read_csv(EVALUATION_DATASET)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def details(out: dict) -> dict:
    return {
        "claims": [
            {
                "claim": r["canonical_claim"],
                "verdict": r["calibrated_verdict"]["verdict"],
                "confidence": r["calibrated_verdict"]["calibrated_confidence"],
                "passages": [
                    {k: p.get(k) for k in ("source_file", "nli_verdict", "rerank_score")} | {"text": p["passage"][:300]}
                    for p in r["scored_passages"]
                ],
            }
            for r in out.get("claim_results") or []
        ],
        "risk_reasons": (out.get("risk_assessment") or {}).get("review_reasons"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--origins", default=PUBLIC_ORIGINS)
    ap.add_argument("--golden", help="golden-set CSV with the 8 verdicts (GUIDELINES.md); scores its true/false rows")
    ap.add_argument("--languages", help="with --golden: only these languages, e.g. hi,hinglish")
    ap.add_argument("--details", action="store_true")
    args = ap.parse_args()
    i, n = (int(x) for x in args.shard.split("/"))

    df = load(args).iloc[i::n]
    if args.limit:
        df = df.head(args.limit)

    app = create_truthcheck_graph()
    records = []
    t0 = time.time()
    for idx, row in df.iterrows():
        cfg = {"configurable": {"thread_id": f"eval-{idx}"}}
        out = {}
        fallback_before = getattr(nodes.get_engine().nli_judge, "fallback_calls", 0)
        decomp_before = nodes.claims_processor.fallback_count
        for attempt in range(2):
            try:
                out = app.invoke({"raw_input": str(row["claim"])}, config=cfg)
                if out.get("__interrupt__"):
                    verdict, held = None, True
                    out = app.get_state(cfg).values
                else:
                    verdict, held = (out.get("final_response") or {}).get("verdict"), False
                break
            except Exception as exc:
                verdict, held = f"ERROR:{type(exc).__name__}", False
                # A GPU out-of-memory error says nothing about the claim; free the cache and retry once
                # on a fresh thread so it is not scored as an abstention.
                if attempt == 0 and "OutOfMemory" in type(exc).__name__:
                    import torch
                    torch.cuda.empty_cache()
                    cfg = {"configurable": {"thread_id": f"eval-{idx}-retry"}}
                    continue
                break
        pred = get_binary_prediction([{"verdict": verdict}]) if verdict and not verdict.startswith("ERROR") else None
        rec = {"idx": int(idx), "true": int(row["label"]), "pred": pred, "verdict": verdict, "held": held,
               "nli_fallback_calls": getattr(nodes.get_engine().nli_judge, "fallback_calls", 0) - fallback_before,
               "decomposition_fallbacks": nodes.claims_processor.fallback_count - decomp_before}
        for key in ("origin", "language"):
            if key in row:
                rec[key] = row[key]
        if args.details:
            rec["claim_text"] = str(row["claim"])
            rec.update(details(out))
        records.append(rec)
        if len(records) % 25 == 0:
            print(f"Processed {len(records)}/{len(df)} ({time.time() - t0:.0f}s)", flush=True)
            json.dump(records, open(args.out, "w"))

    json.dump(records, open(args.out, "w"))
    print(json.dumps(score_with_abstain([r["true"] for r in records], [r["pred"] for r in records]), indent=2))


if __name__ == "__main__":
    main()
