"""Append claims from public health fact-checking datasets to golden_v1.csv.

Rows use the same structure as the existing ai4bharat-derived rows. Sources
(download and extract each into its own directory first):

  SciFact     https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz
  HealthVer   https://github.com/sarrouti/HealthVer (data/healthver_{train,dev,test}.csv)
  COVID-Fact  https://github.com/asaakyan/covidfact (COVIDFACT_dataset.jsonl)
  PUBHEALTH   https://github.com/neemakot/Health-Fact-Checking (PUBHEALTH/{train,dev,test}.tsv)

PubMed/MedRAG is an evidence corpus without claims or labels, so it is not imported.

Evaluation splits (test, dev) are used before train so the golden set stays out
of data a model is likely to be trained on. Sampling is seeded and stratified by
label, and every claim is checked against golden_v1.csv and the rule-1 files in
GUIDELINES.md before it is appended.
"""

import argparse
import collections
import csv
import json
import random
import re
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
GOLD = Path(__file__).with_name("golden_v1.csv")
OVERLAP_FILES = [
    ROOT / "data" / "final_health_claims.csv",
    ROOT / "data" / "final_verifiedfacts.json",
    Path(__file__).with_name("golden_candidates_v0.jsonl"),
]
SEED = 13

POPULATION_RULES = [
    (r"\b(pregnan|miscarr|fetus|foetus|fetal|breastfeed|placenta|maternal)\w*", "pregnant women"),
    (r"\b(newborn|neonat\w*|baby|babies|infant\w*|child|children|kids?|toddler|teen\w*|adolescent\w*|pediatric)\b",
     "children"),
    (r"\bdiabet\w*", "diabetics"),
    (r"\b(elderly|older adults|seniors?|aged \d+ and older)\b", "elderly"),
    (r"\bcancer patients?\b", "cancer patients"),
    (r"\b(women|menopaus\w*)\b", "women"),
]


def population_for(text):
    for pattern, population in POPULATION_RULES:
        if re.search(pattern, text, re.I):
            return population
    return "general population"


def domain(url):
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else (host or "NA")


def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()


# Each loader yields dicts: claim, verdict, label, split, source, quote, url, title, extra_note.

def load_scifact(root):
    data = Path(root) / "data"
    corpus = {}
    with (data / "corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            corpus[str(doc["doc_id"])] = doc
    verdicts = {"SUPPORT": "TRUE", "CONTRADICT": "FALSE"}
    for split in ("dev", "train"):  # test labels are hidden
        with (data / f"claims_{split}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                claim = json.loads(line)
                labels = {ev["label"] for evs in claim["evidence"].values() for ev in evs}
                if len(labels) > 1:
                    continue
                if labels:
                    label = labels.pop()
                    doc_id, evs = next(iter(claim["evidence"].items()))
                    doc = corpus[doc_id]
                    quote = " ".join(doc["abstract"][i] for i in evs[0]["sentences"])
                else:
                    label = "NEI"
                    doc_id = str(claim["cited_doc_ids"][0]) if claim["cited_doc_ids"] else ""
                    doc = corpus.get(doc_id, {"title": "", "abstract": []})
                    quote = ""
                yield {
                    "claim": clean(claim["claim"]), "verdict": verdicts.get(label, "UNCERTAIN"),
                    "label": label, "split": split, "source": "pubmed", "quote": clean(quote),
                    "url": "NA", "title": clean(doc["title"]),
                    "extra_note": f"SciFact claim id {claim['id']}, corpus doc_id {doc_id}",
                }


def load_healthver(root):
    verdicts = {"Supports": "TRUE", "Refutes": "FALSE", "Neutral": "UNCERTAIN"}
    for split in ("test", "dev", "train"):
        pairs = collections.defaultdict(list)
        with (Path(root) / f"healthver_{split}.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                pairs[clean(row["claim"])].append(row)
        for claim, rows in pairs.items():
            labels = {r["label"] for r in rows}
            # Labels are per claim-evidence pair; only claims whose pairs agree get a claim-level verdict.
            if len(labels) != 1:
                continue
            label = labels.pop()
            yield {
                "claim": claim, "verdict": verdicts[label], "label": label, "split": split,
                "source": "cord-19", "quote": clean(rows[0]["evidence"]), "url": "NA",
                "title": clean(rows[0]["question"]),
                "extra_note": f"HealthVer ids {','.join(r['id'] for r in rows)}; "
                              f"{len(rows)} evidence pair(s), all labelled {label}",
            }


def load_covidfact(root):
    verdicts = {"SUPPORTED": "TRUE", "REFUTED": "FALSE"}
    with (Path(root) / "COVIDFACT_dataset.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            note = f"COVID-Fact flair: {item.get('flair', '')}"
            if item["label"] == "REFUTED":
                note += "; refuted claims in COVID-Fact are automatically generated counter-claims"
            yield {
                "claim": clean(item["claim"]), "verdict": verdicts[item["label"]], "label": item["label"],
                "split": "all", "source": domain(item["gold_source"]),
                "quote": clean(" ".join(item["evidence"][:2])), "url": item["gold_source"] or "NA",
                "title": "", "extra_note": note,
            }


# PUBHEALTH mixes in politics and general news; keep only medical items.
PUBHEALTH_MEDICAL_SUBJECT = re.compile(
    r"\b(health|medical|medicine|coronavirus|covid-19|cancer|virus outbreak|vaccin\w*|diseases?|"
    r"nutrition|diet|drugs?|mental health)\b", re.I)
PUBHEALTH_POLITICAL_SUBJECT = re.compile(
    r"\b(politics|donald trump|punditfact|abortion|elections?|guns|immigration|crime|"
    r"entertainment|business)\b", re.I)
MEDICAL_TERMS = re.compile(
    r"\b(vaccin|cancer|diabet|insulin|vitamin|blood|heart|covid|coronavirus|virus|infect|drug|dose|"
    r"pregnan|bab(y|ies)|infant|autism|brain|immun|disease|cure|treat|therap|symptom|supplement|"
    r"diet|food|sugar|salt|cholesterol|liver|kidney|lung|skin|antibiotic|dementia|alzheim|stroke|"
    r"pain|sleep|measles|flu|steroid|hormone|weight|obes|smok|alcohol|aspirin|chemo|tumou?r|"
    r"surgery|screening|patients?|doctors?|hospital|medic|health|clinical|trial|study)\w*", re.I)


def is_medical_pubhealth(row):
    subjects = row["subjects"] or ""
    if PUBHEALTH_POLITICAL_SUBJECT.search(subjects):
        return False
    if subjects.strip():
        return bool(PUBHEALTH_MEDICAL_SUBJECT.search(subjects))
    return bool(MEDICAL_TERMS.search(row["claim"] or ""))


def load_pubhealth(root):
    verdicts = {"true": "TRUE", "false": "FALSE", "mixture": "PARTLY_TRUE", "unproven": "UNCERTAIN"}
    base = Path(root) / "PUBHEALTH" if (Path(root) / "PUBHEALTH").is_dir() else Path(root)
    csv.field_size_limit(10 ** 8)
    for split in ("test", "dev", "train"):
        with (base / f"{split}.tsv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["label"] not in verdicts or not is_medical_pubhealth(row):
                    continue
                urls = [u for u in re.split(r"[,\s]+", row["sources"] or "") if u.startswith("http")]
                yield {
                    "claim": clean(row["claim"]), "verdict": verdicts[row["label"]], "label": row["label"],
                    "split": split, "source": "pubhealth", "quote": clean(row["explanation"]),
                    "url": urls[0] if urls else "NA", "title": "",
                    "extra_note": f"PUBHEALTH claim_id {row['claim_id']}; fact checkers: "
                                  f"{clean(row['fact_checkers']) or 'NA'}; subjects: {clean(row['subjects']) or 'NA'}",
                }


DATASETS = {
    "scifact": ("SF", "scifact_dataset", "SciFact", load_scifact),
    "healthver": ("HV", "healthver_dataset", "HealthVer", load_healthver),
    "covidfact": ("CF", "covidfact_dataset", "COVID-Fact", load_covidfact),
    "pubhealth": ("PH", "pubhealth_dataset", "PUBHEALTH", load_pubhealth),
}


class Deduper:
    """Near-duplicate check (token Jaccard) backed by an inverted index."""

    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.docs = []
        self.index = collections.defaultdict(list)

    @staticmethod
    def tokens(text):
        return frozenset(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())

    def add(self, text):
        toks = self.tokens(text)
        if not toks:
            return
        doc_id = len(self.docs)
        self.docs.append(toks)
        for tok in toks:
            self.index[tok].append(doc_id)

    def contains(self, text):
        toks = self.tokens(text)
        if not toks:
            return True
        # Very common tokens only narrow the candidate set; similarity uses all tokens.
        candidates = set()
        for tok in toks:
            postings = self.index.get(tok, ())
            if len(postings) < 5000:
                candidates.update(postings)
        for doc_id in candidates:
            other = self.docs[doc_id]
            if len(toks & other) / len(toks | other) >= self.threshold:
                return True
        return False


def load_existing(gold_rows):
    dedup = Deduper()
    for row in gold_rows:
        dedup.add(row["claim_text"])
    with OVERLAP_FILES[0].open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            dedup.add(row["claim"])
    for item in json.loads(OVERLAP_FILES[1].read_text(encoding="utf-8")):
        dedup.add(item.get("claim_subject", ""))
        dedup.add(item.get("claim_text", ""))
    for line in OVERLAP_FILES[2].read_text(encoding="utf-8").splitlines():
        dedup.add(json.loads(line)["original_text"])
    return dedup


def sample(records, per_dataset, rng):
    """Stratified by verdict, preferring earlier (evaluation) splits."""
    by_verdict = collections.defaultdict(list)
    for rec in records:
        by_verdict[rec["verdict"]].append(rec)
    split_rank = {"test": 0, "dev": 1, "all": 1, "train": 2}
    for recs in by_verdict.values():
        rng.shuffle(recs)
        recs.sort(key=lambda r: split_rank[r["split"]])
    chosen, queues = [], [collections.deque(v) for _, v in sorted(by_verdict.items())]
    while len(chosen) < per_dataset and any(queues):
        for queue in queues:
            if queue and len(chosen) < per_dataset:
                chosen.append(queue.popleft())
    return chosen


def to_row(rec, row_id, origin, dataset_name):
    quote = rec["quote"] or "NA"
    notes = {
        "legacy_claim_subject": "",
        "legacy_scientific_truth": rec["quote"],
        "legacy_risk_level": "Medium",
        "legacy_medical_domain": "health_claim",
        "legacy_source_title": rec["title"] or dataset_name,
        "legacy_evidence_type": "expert_consensus",
        "legacy_evidence_summary": rec["quote"],
        "conversion_note": f"Generated from {dataset_name} dataset ({rec['split']} split); "
                           f"original label: {rec['label']}; {rec['extra_note']}",
    }
    return {
        "id": row_id, "claim_text": rec["claim"], "language": "English", "script": "Latin",
        "claim_type": "health_claim", "population": population_for(rec["claim"]),
        "expected_verdict": rec["verdict"], "gold_evidence_source": rec["source"],
        "gold_evidence_quote": quote, "evidence_url": rec["url"], "in_kb": "NA",
        "claim_origin": origin, "annotator": "NA", "second_annotator": "NA",
        "second_verdict": "NA", "status": "needs_evidence_review",
        "notes": json.dumps(notes, ensure_ascii=False),
    }


def main():
    parser = argparse.ArgumentParser()
    for name in DATASETS:
        parser.add_argument(f"--{name}", help=f"directory holding the extracted {name} release")
    parser.add_argument("--per-dataset", type=int, default=250)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with GOLD.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        gold_rows = list(reader)
    dedup = load_existing(gold_rows)
    rng = random.Random(SEED)

    new_rows = []
    for name, (prefix, origin, dataset_name, loader) in DATASETS.items():
        root = getattr(args, name)
        if not root:
            continue
        candidates, skipped = [], collections.Counter()
        for rec in loader(root):
            if not 15 <= len(rec["claim"]) <= 400:
                skipped["length"] += 1
            elif dedup.contains(rec["claim"]):
                skipped["duplicate"] += 1
            else:
                candidates.append(rec)
                dedup.add(rec["claim"])  # also dedups within and across new datasets
        chosen = sample(candidates, args.per_dataset, rng)
        next_id = 1 + max((int(r["id"][2:]) for r in gold_rows
                           if re.fullmatch(prefix + r"\d{6}", r["id"])), default=0)
        for offset, rec in enumerate(chosen):
            new_rows.append(to_row(rec, f"{prefix}{next_id + offset:06d}", origin, dataset_name))
        verdicts = collections.Counter(r["verdict"] for r in chosen)
        splits = collections.Counter(r["split"] for r in chosen)
        print(f"{dataset_name:10} candidates={len(candidates):5} chosen={len(chosen):4} "
              f"verdicts={dict(verdicts)} splits={dict(splits)} skipped={dict(skipped)}")

    if args.dry_run or not new_rows:
        print(f"Dry run: {len(new_rows)} rows not written")
        return
    with GOLD.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=columns).writerows(new_rows)
    print(f"Appended {len(new_rows)} rows to {GOLD} (now {len(gold_rows) + len(new_rows)} rows)")


if __name__ == "__main__":
    main()
