import glob, json, sys
from collections import Counter
from src.binary_scoring import score_with_abstain

d = sys.argv[1]
recs = [r for f in sorted(glob.glob(f"{d}/shard*.json")) for r in json.load(open(f, encoding="utf-8"))]
y_true = [r["true"] for r in recs]
y_pred = [r["pred"] for r in recs]
print(json.dumps(score_with_abstain(y_true, y_pred), indent=2))
print("verdicts:", Counter(r["verdict"] for r in recs).most_common())
json.dump(recs, open(f"{d}/merged_predictions.json", "w"))
