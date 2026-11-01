"""Check new KB files before committing: format, size, and overlap with evaluation claims.

Flags a file when it holds 50%+ of a golden_v1 claim's 5-word sequences, or a gold evidence quote
verbatim, because the system could then look the answer up and inflate scores. Reworded copies are
not detected. Also flags near-duplicate files.

usage: python scripts/kb/check_kb.py [FILENAME_PREFIX ...]   (default: every file in SOURCES.csv)
"""
import csv
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
KB = "data/medical_kb"
N = 5
BENCHMARK = {"scifact_dataset", "healthver_dataset", "pubhealth_dataset", "ai4bharat_dataset"}
prefixes = tuple(sys.argv[1:])


def tokens(s):
    return re.sub(r"[^\w ]+", " ", str(s).lower()).split()


rows = list(csv.DictReader(open(os.path.join(KB, "SOURCES.csv"), encoding="utf-8")))
if prefixes:
    rows = [r for r in rows if r["filename"].startswith(prefixes)]
print("csv rows", len(rows), "unique files", len({r["filename"] for r in rows}), "unique file+url", len({(r["filename"], r["url"]) for r in rows}))
texts = {}
for r in rows:
    lines = open(os.path.join(KB, r["filename"]), encoding="utf-8").read().split("\n")
    ok = lines[0].startswith("SOURCE:") and lines[1].startswith("TITLE:") and lines[2].startswith("URL:") and lines[3].startswith("─")
    if not ok:
        print("BAD HEADER", r["filename"])
    texts[r["filename"]] = "\n".join(lines[5:])
wc = {f: len(t.split()) for f, t in texts.items()}
print("words min/median/max", min(wc.values()), sorted(wc.values())[len(wc) // 2], max(wc.values()), "total", sum(wc.values()))
print("under 150 words:", [(f, w) for f, w in wc.items() if w < 150])
print("by prefix", Counter(re.match(r"[a-z]+", f).group() for f in texts))

grams = defaultdict(set)
for f, t in texts.items():
    w = tokens(t)
    for i in range(len(w) - N + 1):
        grams[" ".join(w[i:i + N])].add(f)
g = pd.read_csv("evaluation/golden_set/golden_v1.csv", low_memory=False)
hits = defaultdict(list)
for _, r in g.iterrows():
    for field, need in (("claim_text", 0.5), ("gold_evidence_quote", 0.9)):
        if not isinstance(r[field], str):
            continue
        w = tokens(r[field])
        gs = [" ".join(w[i:i + N]) for i in range(len(w) - N + 1)]
        if len(gs) < 3:
            continue
        for f, n in Counter(f for x in gs for f in grams.get(x, ())).items():
            if n / len(gs) >= need:
                hits[f].append((round(n / len(gs), 2), r.id, r.claim_origin, field, r.expected_verdict, r[field][:90]))
print("files overlapping golden_v1:", len(hits))
for f, h in sorted(hits.items(), key=lambda x: -len(x[1])):
    bench = sum(x[2] in BENCHMARK for x in h)
    print(f"  {f} rows={len(h)} benchmark={bench} {sorted(h, reverse=True)[:2]}")

sh = {f: {" ".join(tokens(t)[i:i + 8]) for i in range(0, max(1, len(tokens(t)) - 8), 4)} for f, t in texts.items()}
files = list(sh)
dups = []
for i, a in enumerate(files):
    for b in files[i + 1:]:
        o = len(sh[a] & sh[b]) / max(1, min(len(sh[a]), len(sh[b])))
        if o > 0.4:
            dups.append((round(o, 2), a, b))
print("near-duplicate pairs:", len(dups), sorted(dups, reverse=True)[:15])
