import glob
import json
import sys
from collections import Counter

from src.binary_scoring import score_with_abstain

d = sys.argv[1]
recs = [r for f in sorted(glob.glob(f"{d}/shard*.json")) for r in json.load(open(f, encoding="utf-8"))]

print(json.dumps(score_with_abstain([r["true"] for r in recs], [r["pred"] for r in recs]), indent=2))

held = [r for r in recs if r.get("held")]
if held:
    rest = [r for r in recs if not r.get("held")]
    print(f"held for human review: {len(held)} ({len(held) / len(recs):.1%}); scores excluding them:")
    print(json.dumps(score_with_abstain([r["true"] for r in rest], [r["pred"] for r in rest]), indent=2))

decomp = sum(1 for r in recs if r.get("decomposition_fallbacks"))
if decomp:
    print(f"WARNING: {decomp} claims used the decomposition heuristic after an LLM failure")
fb = [r for r in recs if r.get("nli_fallback_calls")]
if fb:
    print(f"WARNING: {len(fb)} claims used the local NLI fallback ({sum(r['nli_fallback_calls'] for r in fb)} judgments)")
print("verdicts:", Counter(r["verdict"] for r in recs).most_common())

# Coverage alone can be bought by calling true claims false, so report both classes' recall.
answered = [r for r in recs if r["pred"] is not None]
recall = {c: (sum(r["pred"] == c for r in answered if r["true"] == c) /
              max(1, sum(r["true"] == c for r in answered))) for c in (0, 1)}
print(f"answered balanced accuracy: {(recall[0] + recall[1]) / 2:.3f}  "
      f"(true claims called true {recall[0]:.3f}, false claims called false {recall[1]:.3f})")
errors = sum(1 for r in recs if str(r["verdict"]).startswith("ERROR"))
if errors:
    print(f"WARNING: {errors} claims ended in an error and count as abstentions")
for key in ("origin", "language"):
    if not any(key in r for r in recs):
        continue
    print(f"by {key}:")
    for origin in sorted({str(r.get(key)) for r in recs}):
        rows = [r for r in recs if str(r.get(key)) == origin]
        ans = [r for r in rows if r["pred"] is not None]
        acc = sum(r["pred"] == r["true"] for r in ans) / len(ans) if ans else 0.0
        print(f"  {origin:20} n={len(rows):5} coverage={len(ans) / len(rows):.3f} answered_acc={acc:.3f}")
json.dump(recs, open(f"{d}/merged_predictions.json", "w"))
