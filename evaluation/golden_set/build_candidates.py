"""Create a deterministic, balanced v0 review queue from labelled claims."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "final_health_claims.csv"
OUTPUT = Path(__file__).with_name("golden_candidates_v0.jsonl")
PER_CLASS = 125


def main():
    rows_by_label = {0: [], 1: []}
    with SOURCE.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label = int(row["label"])
            if label in rows_by_label and row["claim"].strip():
                rows_by_label[label].append(row["claim"].strip())

    # The source order is stable; stride sampling keeps the candidate queue
    # broad without silently treating its labels as independently audited truth.
    candidates = []
    for label, expected_class in ((0, "FACTUAL"), (1, "MISINFORMATION")):
        population = rows_by_label[label]
        stride = max(1, len(population) // PER_CLASS)
        selected = population[::stride][:PER_CLASS]
        for index, claim in enumerate(selected, 1):
            candidates.append({
                "id": f"v0-{label}-{index:03d}",
                "original_text": claim,
                "expected_class": expected_class,
                "language": "en",
                "source": "data/final_health_claims.csv",
                "source_label": label,
                "review_status": "candidate",
                "reviewer": None,
                "evidence_urls": [],
                "notes": "Requires independent evidence review before becoming golden data."
            })

    with OUTPUT.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    print(f"Wrote {len(candidates)} review candidates to {OUTPUT}")


if __name__ == "__main__":
    main()
