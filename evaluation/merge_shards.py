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

fb = [r for r in recs if r.get("nli_fallback_calls")]
if fb:
    print(f"WARNING: {len(fb)} claims used the local NLI fallback ({sum(r['nli_fallback_calls'] for r in fb)} judgments)")
print("verdicts:", Counter(r["verdict"] for r in recs).most_common())
json.dump(recs, open(f"{d}/merged_predictions.json", "w"))
