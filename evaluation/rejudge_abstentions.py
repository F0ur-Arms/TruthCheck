"""Re-judge the evidence behind a run's INSUFFICIENT_EVIDENCE claims with another judge setting.

A quick estimate of how much a judge change (prompt v3, judging the top passages together) would
recover, without re-running retrieval: it reuses the passages a --details run saved for each
abstained claim, restores their full text from the knowledge-base passage cache, and asks the LLM
judge again. A claim counts as answered when its sub-claims' new labels point one way (SUPPORTS =
true, REFUTES = false) and as still abstained when they are NEI or conflict. This approximates the
calibrator, so treat the result as an estimate and confirm a promising setting with a full run.

With --scope answered it instead re-judges the passages behind the claims the run DID answer and
reports how many answers the change keeps, withdraws or flips, to test a judge fix for wrong answers.

usage: python evaluation/rejudge_abstentions.py RUN/merged_predictions.json KB/kb_passages.<key>.txt
           --prompt v3 [--combine 3] [--top 5] [--min-rerank 0.0] [--workers 8] [--scope answered] --out rows.json
"""

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.RAGpipeline.llm_nli import SYSTEM, SYSTEM_V1, SYSTEM_V3, SYSTEM_V4, LLMNLIJudge

PROMPTS = {"v1": SYSTEM_V1, "v2": SYSTEM, "v3": SYSTEM_V3, "v4": SYSTEM_V4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("passages")
    ap.add_argument("--prompt", default="v3", choices=sorted(PROMPTS))
    ap.add_argument("--combine", type=int, default=0, help="also judge the top k passages together")
    ap.add_argument("--top", type=int, default=5, help="passages judged per sub-claim, by rerank score")
    ap.add_argument("--min-rerank", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scope", default="abstained", choices=["abstained", "answered"],
                    help="abstained: can the change recover INSUFFICIENT_EVIDENCE claims; answered: which of the "
                         "run's answers would it keep, withdraw or flip")
    args = ap.parse_args()

    recs = json.load(open(args.predictions, encoding="utf-8"))
    full = {}
    with open(args.passages, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                full.setdefault(line[:300], line)

    if args.scope == "answered":
        targets = [r for r in recs if r["pred"] is not None and any(c.get("passages") for c in r.get("claims", []))]
    else:
        targets = [r for r in recs if r["pred"] is None and r["verdict"] == "INSUFFICIENT_EVIDENCE"
                   and any(c.get("passages") for c in r.get("claims", []))]
    judge = LLMNLIJudge(system_prompt=PROMPTS[args.prompt])

    jobs = []  # (record index, sub-claim index, claim text, evidence text)
    for ti, r in enumerate(targets):
        for ci, c in enumerate(r["claims"]):
            ps = sorted(c.get("passages", []), key=lambda p: p.get("rerank_score") or 0.0, reverse=True)
            ps = [p for p in ps if (p.get("rerank_score") or 0.0) >= args.min_rerank][:args.top]
            texts = [full.get(p["text"][:300], p["text"]) for p in ps]
            jobs += [(ti, ci, c["claim"], t) for t in texts]
            if args.combine >= 2 and len(texts) >= 2:
                top = texts[:args.combine]
                budget = 3000 // len(top)
                jobs.append((ti, ci, c["claim"], "\n\n".join(t[:budget] for t in top)))
    print(f"{len(targets)} {args.scope} claims with passages, {len(jobs)} judgments", flush=True)

    def run(job):
        try:
            return judge.verify(job[2], job[3])["verdict"]
        except Exception as exc:
            return f"ERROR:{type(exc).__name__}"

    with ThreadPoolExecutor(args.workers) as pool:
        labels = list(pool.map(run, jobs))

    sub = {}
    for (ti, ci, _, _), label in zip(jobs, labels):
        sub.setdefault((ti, ci), Counter())[label] += 1
    rows, errors = [], sum(l.startswith("ERROR") for l in labels)
    for ti, r in enumerate(targets):
        dirs = set()
        for ci in range(len(r["claims"])):
            c = sub.get((ti, ci), Counter())
            if c["SUPPORTS"] or c["REFUTES"]:
                dirs.add(0 if c["SUPPORTS"] > c["REFUTES"] else 1 if c["REFUTES"] > c["SUPPORTS"] else None)
        dirs.discard(None)
        pred = dirs.pop() if len(dirs) == 1 else None
        rows.append({"idx": r["idx"], "origin": r.get("origin"), "true": r["true"], "old_pred": r["pred"],
                     "new_pred": pred, "claim_text": r.get("claim_text")})
    json.dump(rows, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"prompt={args.prompt} combine={args.combine} top={args.top} scope={args.scope}  judge errors={errors}")
    if args.scope == "answered":
        report_answered(recs, rows)
        return

    got = [x for x in rows if x["new_pred"] is not None]
    recall = {c: sum(x["new_pred"] == c for x in got if x["true"] == c) / max(1, sum(x["true"] == c for x in got))
              for c in (0, 1)}
    print(f"recovered {len(got)} of {len(targets)} abstained claims; correct {sum(x['new_pred'] == x['true'] for x in got)}"
          f" ({sum(x['new_pred'] == x['true'] for x in got) / max(1, len(got)):.1%}); "
          f"true claims called true {recall[0]:.1%}, false claims called false {recall[1]:.1%}")
    base = sum(r["pred"] is not None for r in recs)
    base_ok = sum(r["pred"] == r["true"] for r in recs if r["pred"] is not None)
    new_ok = base_ok + sum(x["new_pred"] == x["true"] for x in got)
    print(f"estimated run: coverage {base / len(recs):.1%} -> {(base + len(got)) / len(recs):.1%}, "
          f"answered accuracy {base_ok / max(1, base):.1%} -> {new_ok / max(1, base + len(got)):.1%}")
    by = Counter(x["origin"] for x in got)
    print("recovered by origin:", dict(by))


def report_answered(recs, rows):
    """How the run's answered claims fare under the new judge: kept, withdrawn (abstain) or flipped."""
    moves = Counter()
    for x in rows:
        old_ok = x["old_pred"] == x["true"]
        new = "abstain" if x["new_pred"] is None else ("same" if x["new_pred"] == x["old_pred"] else "flipped")
        moves[(("right" if old_ok else "wrong"), new)] += 1
    for (was, now), n in sorted(moves.items()):
        print(f"  {was:5} answer -> {now:8} {n}")
    kept = [x for x in rows if x["new_pred"] is not None]
    n = len(recs)
    base_ok = sum(r["pred"] == r["true"] for r in recs if r["pred"] is not None)
    new_ok = sum(x["new_pred"] == x["true"] for x in kept)
    recall = {c: sum(x["new_pred"] == c for x in kept if x["true"] == c) / max(1, sum(x["true"] == c for x in kept))
              for c in (0, 1)}
    print(f"estimated run (answered claims only): coverage {len(rows) / n:.1%} -> {len(kept) / n:.1%}, "
          f"answered accuracy {base_ok / max(1, len(rows)):.1%} -> {new_ok / max(1, len(kept)):.1%}, "
          f"balanced {(recall[0] + recall[1]) / 2:.3f}")


if __name__ == "__main__":
    main()
