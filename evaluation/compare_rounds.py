"""Compare two --details runs on the same claims: coverage change and which verdicts flipped.

Lists true claims that the new run calls false (the most harmful error) together with the passages
behind the new verdict, and counts the source files those passages came from, so a knowledge-base
batch that misleads the judge can be found and excluded with TRUTHCHECK_EXCLUDE_SOURCES.

usage: python evaluation/compare_rounds.py OLD/merged_predictions.json NEW/merged_predictions.json [--show 25]
"""

import argparse
import json
from collections import Counter


def stats(recs):
    ans = [r for r in recs if r["pred"] is not None]
    rec = {c: sum(r["pred"] == c for r in ans if r["true"] == c) / max(1, sum(r["true"] == c for r in ans))
           for c in (0, 1)}
    return (f"answered {len(ans)}/{len(recs)} ({len(ans) / max(1, len(recs)):.1%})  "
            f"acc {sum(r['pred'] == r['true'] for r in ans) / max(1, len(ans)):.1%}  "
            f"balanced {(rec[0] + rec[1]) / 2:.3f}  true called true {rec[0]:.1%}  false called false {rec[1]:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--show", type=int, default=25)
    args = ap.parse_args()
    old = {r["idx"]: r for r in json.load(open(args.old, encoding="utf-8"))}
    new = {r["idx"]: r for r in json.load(open(args.new, encoding="utf-8"))}
    common = sorted(set(old) & set(new))
    print(f"claims in both runs: {len(common)}")
    print("old:", stats([old[i] for i in common]))
    print("new:", stats([new[i] for i in common]))

    moves = Counter()
    for i in common:
        a, b = old[i]["pred"], new[i]["pred"]
        moves[(("abstain" if a is None else ("ok" if a == old[i]["true"] else "wrong")),
               ("abstain" if b is None else ("ok" if b == new[i]["true"] else "wrong")))] += 1
    print("old -> new:", {f"{k[0]}->{k[1]}": v for k, v in sorted(moves.items()) if k[0] != k[1]})

    harmful = [i for i in common if new[i]["true"] == 0 and new[i]["pred"] == 1 and old[i]["pred"] != 1]
    files = Counter()
    print(f"\ntrue claims the new run calls false and the old one did not: {len(harmful)}")
    for i in harmful:
        r = new[i]
        decisive = [p for c in r.get("claims", []) for p in c.get("passages", []) if p.get("nli_verdict") == "REFUTES"]
        files.update(p.get("source_file") for p in decisive)
        if args.show:
            args.show -= 1
            print(f"- [{r.get('origin')}] {r.get('claim_text', '')[:110]}")
            for p in decisive[:2]:
                print(f"    REFUTES from {p.get('source_file')} (rerank {p.get('rerank_score', 0):.2f}): {p.get('text', '')[:140]}")
    print("\nsource files behind those REFUTES judgments:", files.most_common(15))

    gained = [i for i in common if old[i]["pred"] is None and new[i]["pred"] is not None]
    src = Counter(p.get("source_file") for i in gained for c in new[i].get("claims", []) for p in c.get("passages", [])
                  if p.get("nli_verdict") in ("SUPPORTS", "REFUTES"))
    print(f"\nnewly answered claims: {len(gained)}, correct {sum(new[i]['pred'] == new[i]['true'] for i in gained)}")
    print("source files behind newly answered claims:", src.most_common(15))


if __name__ == "__main__":
    main()
