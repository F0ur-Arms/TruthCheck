"""Same stage test as eval_nli_stage.py, with the vLLM-served LLM as the judge."""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

GOLD = {"TRUE": "SUPPORTS", "FALSE": "REFUTES", "UNCERTAIN": "NEI"}
BASE = os.environ.get("TRUTHCHECK_LLM_BASE_URL", "http://cb-gpu-server:8000/v1")
MODEL = os.environ.get("TRUTHCHECK_LLM_MODEL", "qwen2.5-7b-awq")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.RAGpipeline.llm_nli import LLMNLIJudge

JUDGE = LLMNLIJudge(base_url=BASE, api_key="none", model=MODEL)


def judge(item):
    try:
        res = JUDGE.verify(*item)
        return res["verdict"], res["confidence"]
    except Exception:
        return "ERROR", 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="evaluation/golden_set/golden_v1.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["gold_evidence_quote"].notna() & df["expected_verdict"].isin(GOLD)]
    df = df[df["gold_evidence_quote"].astype(str).str.strip() != ""]
    items = list(zip(df["claim_text"].astype(str), df["gold_evidence_quote"].astype(str)))
    with ThreadPoolExecutor(args.workers) as ex:
        preds = list(ex.map(judge, items))

    rows = [{"id": i, "origin": o, "gold": GOLD[v], "pred": p[0], "conf": p[1]}
            for i, o, v, p in zip(df["id"], df["claim_origin"], df["expected_verdict"], preds)]
    json.dump(rows, open(args.out, "w"))
    labels = ["SUPPORTS", "REFUTES", "NEI"]
    print(f"rows: {len(rows)}  accuracy: {sum(x['gold'] == x['pred'] for x in rows) / len(rows):.4f}  "
          f"errors: {sum(x['pred'] == 'ERROR' for x in rows)}")
    cm = Counter((x["gold"], x["pred"]) for x in rows)
    print("gold \\ pred " + " ".join(f"{p:>9}" for p in labels))
    for g in labels:
        print(f"{g:>11} " + " ".join(f"{cm[(g, p)]:>9}" for p in labels))
    by = defaultdict(list)
    for x in rows:
        by[x["origin"]].append(x["gold"] == x["pred"])
    for k, v in sorted(by.items()):
        print(f"  {k:<22} n={len(v):<5} acc={sum(v) / len(v):.3f}")
    for lo, hi in ((0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
        sel = [x for x in rows if lo <= x["conf"] < hi]
        if sel:
            print(f"  conf {lo:.1f}-{min(hi, 1):.1f}: n={len(sel):<5} acc={sum(x['gold'] == x['pred'] for x in sel) / len(sel):.3f}")


if __name__ == "__main__":
    main()
