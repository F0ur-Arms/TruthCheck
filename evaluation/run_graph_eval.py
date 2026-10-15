"""Score the LangGraph pipeline on the labelled claim set, sharded like main.py."""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TRUTHCHECK_CHECKPOINT_DB", ":memory:")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import EVALUATION_DATASET
from graph.workflow import create_truthcheck_graph
from src.binary_scoring import get_binary_prediction, score_with_abstain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    i, n = (int(x) for x in args.shard.split("/"))

    df = pd.read_csv(EVALUATION_DATASET).sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.iloc[i::n]
    if args.limit:
        df = df.head(args.limit)

    app = create_truthcheck_graph()
    records = []
    t0 = time.time()
    for idx, row in df.iterrows():
        cfg = {"configurable": {"thread_id": f"eval-{idx}"}}
        try:
            out = app.invoke({"raw_input": str(row["claim"])}, config=cfg)
            if out.get("__interrupt__"):
                verdict, held = None, True
            else:
                verdict, held = (out.get("final_response") or {}).get("verdict"), False
        except Exception as exc:
            verdict, held = f"ERROR:{type(exc).__name__}", False
        pred = get_binary_prediction([{"verdict": verdict}]) if verdict and not verdict.startswith("ERROR") else None
        records.append({"idx": int(idx), "true": int(row["label"]), "pred": pred, "verdict": verdict, "held": held})
        if len(records) % 25 == 0:
            print(f"Processed {len(records)}/{len(df)} ({time.time() - t0:.0f}s)", flush=True)
            json.dump(records, open(args.out, "w"))

    json.dump(records, open(args.out, "w"))
    print(json.dumps(score_with_abstain([r["true"] for r in records], [r["pred"] for r in records]), indent=2))


if __name__ == "__main__":
    main()
