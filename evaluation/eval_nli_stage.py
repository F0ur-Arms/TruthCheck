"""Stage-isolated NLI evaluation: given the gold evidence quote, does the NLI model reach the gold label?

Rows come from golden_v1.csv that carry a gold_evidence_quote. TRUE -> SUPPORTS, FALSE -> REFUTES,
UNCERTAIN -> NEI. Rows labelled PARTLY_TRUE or MISLEADING are skipped (no single NLI label).
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.RAGpipeline.nli_verifier import NLIVerifier

GOLD = {"TRUE": "SUPPORTS", "FALSE": "REFUTES", "UNCERTAIN": "NEI"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="evaluation/golden_set/golden_v1.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=None, help="HF model id; default is config.NLI_MODEL")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["gold_evidence_quote"].notna() & df["expected_verdict"].isin(GOLD)]
    df = df[df["gold_evidence_quote"].astype(str).str.strip() != ""]
    if args.limit:
        df = df.head(args.limit)

    nli = NLIVerifier(args.model) if args.model else NLIVerifier()
    rows = []
    for _, r in df.iterrows():
        res = nli.verify(str(r["claim_text"]), str(r["gold_evidence_quote"]))
        rows.append({"id": r["id"], "origin": r["claim_origin"], "gold": GOLD[r["expected_verdict"]],
                     "pred": res["verdict"], "conf": res["confidence"]})
        if len(rows) % 200 == 0:
            print(f"{len(rows)}/{len(df)}", flush=True)

    json.dump(rows, open(args.out, "w"))
    labels = ["SUPPORTS", "REFUTES", "NEI"]
    print(f"rows: {len(rows)}  accuracy: {sum(x['gold'] == x['pred'] for x in rows) / len(rows):.4f}")
    cm = Counter((x["gold"], x["pred"]) for x in rows)
    print("gold \\ pred " + " ".join(f"{p:>9}" for p in labels))
    for g in labels:
        print(f"{g:>11} " + " ".join(f"{cm[(g, p)]:>9}" for p in labels))
    by = defaultdict(list)
    for x in rows:
        by[x["origin"]].append(x["gold"] == x["pred"])
    for k, v in sorted(by.items()):
        print(f"  {k:<22} n={len(v):<5} acc={sum(v) / len(v):.3f}")


if __name__ == "__main__":
    main()
