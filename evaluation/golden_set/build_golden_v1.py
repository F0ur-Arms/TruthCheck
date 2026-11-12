"""Build golden_v1.csv from the curated set plus the Indian fact-check and gap-filling sources.

Inputs
  final_goldenv1_fixed.csv           curated rows (English, and Hinglish/Hindi translations)
  sources/indian_factchecks_v1.csv   real Hindi claims from Vishvas News / Newschecker, plus Hinglish
                                     rewrites of some of them; verdicts mapped from the publisher rating
  sources/gap_claims_draft.csv       self-written claims for categories GUIDELINES.md asks for
                                     (treatment cessation, dosage, medical-advice requests); the verdicts
                                     there are proposals that a human annotator must confirm (rule 3)
  KB registry (SOURCES.csv)          to set in_kb / gold_evidence_source when the source article is in the KB

The public-dataset candidate pool that used to be golden_v1.csv is now
evaluation/external/public_candidates_v1.csv, as the progress report asked.
Values are normalised to GUIDELINES.md: language en/hi/hinglish, script latin/devanagari,
population general/pregnancy/children/elderly/diabetes/hypertension/other:<text>, and the file is
written without a byte-order mark.

usage: python evaluation/golden_set/build_golden_v1.py [--kb-sources data/medical_kb/SOURCES.csv]
"""

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
COLUMNS = ["id", "claim_text", "language", "script", "claim_type", "population", "expected_verdict",
           "gold_evidence_source", "gold_evidence_quote", "evidence_url", "in_kb", "claim_origin",
           "annotator", "second_annotator", "second_verdict", "status", "notes"]
LANG = {"english": "en", "en": "en", "hindi": "hi", "hi": "hi", "hinglish": "hinglish"}
POPULATION = {"general population": "general", "general": "general", "diabetics": "diabetes",
              "diabetes": "diabetes", "pregnant women": "pregnancy", "pregnancy": "pregnancy",
              "children": "children", "elderly": "elderly", "hypertension": "hypertension"}
CLAIM_TYPES = {"general", "population_specific", "treatment_cessation", "dosage", "food_remedy", "vaccine",
               "medical_advice_request"}
# Publisher ratings on the Indian fact-checks -> the 8 verdicts. "Half true" on Vishvas is used for claims
# with a grain of truth stretched to a wrong conclusion, which GUIDELINES.md calls MISLEADING.
PUBLISHER_VERDICT = {"False": "FALSE", "Half true": "MISLEADING", "Missing Context": "MISLEADING",
                     "Partly False": "MOSTLY_FALSE"}
VACCINE = re.compile(r"वैक्सीन|टीका|टीके|vaccin|\btika\b|\bteeka\b", re.I)
CESSATION = re.compile(r"band kar|chhod|छोड़|बंद कर|stop (taking|your)", re.I)
REMEDY = re.compile(r"काढ़ा|kadha|juice|जूस|तेल|\btel\b|नींबू|nimbu|हल्दी|haldi|लहसुन|lehsun|अदरक|adrak|शहद|shahad|"
                    r"मेथी|methi|दालचीनी|dalchini|पत्त|patte|पानी|paani|दूध|doodh|खाने से|khane se|पीने से|pine se|"
                    r"चूना|chuna|गेहूं|gehu|नारियल|nariyal|केला|kela|प्याज|pyaz", re.I)


def population(value):
    v = (value or "").strip()
    if v.lower() in POPULATION:
        return POPULATION[v.lower()]
    if v.startswith("other:"):
        return v
    return f"other:{v.lower()}" if v and v.upper() != "NA" else "general"


def norm(text):
    """Claim text without punctuation or case, keeping letters in any script (Devanagari matras too)."""
    kept = "".join(c if not unicodedata.category(c)[0] in "PSZ" else " " for c in str(text).lower())
    return " ".join(kept.split())


def curated_rows():
    rows = []
    with open(HERE / "final_goldenv1_fixed.csv", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            r = {k: (r.get(k) or "") for k in COLUMNS}
            r["language"] = LANG.get(r["language"].lower(), r["language"].lower())
            r["script"] = r["script"].lower()
            r["population"] = population(r["population"])
            if r["claim_type"] not in CLAIM_TYPES:
                r["claim_type"] = "population_specific" if r["population"] != "general" else "general"
            if r["status"] == "needs_evidence_review":
                r["status"] = "draft"
            rows.append(r)
    return rows


def indian_rows(kb_files):
    rows = []
    with open(HERE / "sources" / "indian_factchecks_v1.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            notes = json.loads(r["notes"])
            # Hindi rows say "publisher rating: X"; their Hinglish rewrites say "... rating: X".
            rating = re.search(r"rating: (.+)$", notes.get("conversion_note", ""))
            rating = rating.group(1).strip() if rating else ""
            verdict = PUBLISHER_VERDICT.get(rating, r["expected_verdict"])
            text = r["claim_text"]
            pop = population(r["population"])
            if VACCINE.search(text):
                ctype = "vaccine"
            elif CESSATION.search(text):
                ctype = "treatment_cessation"
            elif pop != "general":
                ctype = "population_specific"
            elif REMEDY.search(text):
                ctype = "food_remedy"
            else:
                ctype = "general"
            kb_file = kb_files.get(r["evidence_url"].rstrip("/"))
            notes.update({"publisher_rating": rating,
                          "verdict_note": f"mapped from publisher rating '{rating}' to {verdict}; "
                                          "a human annotator should confirm it against the article"})
            rows.append({
                **{k: r.get(k, "") for k in COLUMNS},
                "language": LANG[r["language"].lower()], "script": r["script"].lower(),
                "claim_type": ctype, "population": pop, "expected_verdict": verdict,
                "gold_evidence_source": kb_file or "NONE", "in_kb": "yes" if kb_file else "no",
                # The Hinglish rows are rewrites of the real Hindi claims, not forwards found in the wild.
                "claim_origin": r["claim_origin"] + ("_hinglish_rewrite" if r["language"] == "Hinglish" else ""),
                "annotator": "NA", "second_annotator": "NA", "second_verdict": "NA", "status": "draft",
                "notes": json.dumps(notes, ensure_ascii=False),
            })
    return rows


def gap_rows():
    rows = []
    with open(HERE / "sources" / "gap_claims_draft.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            advice = r["claim_type"] == "medical_advice_request"
            note = ("asks for personal medical advice, so NOT_A_FACT_CHECK by definition" if advice else
                    "proposed verdict only: a human annotator must confirm it and add a KB source and quote")
            rows.append({
                "id": r["id"], "claim_text": r["claim_text"], "language": r["language"], "script": r["script"],
                "claim_type": r["claim_type"], "population": population(r["population"]),
                "expected_verdict": r["proposed_verdict"], "gold_evidence_source": "NONE",
                "gold_evidence_quote": "", "evidence_url": "NA", "in_kb": "no" if advice else "NA",
                "claim_origin": "self_written", "annotator": "NA", "second_annotator": "NA",
                "second_verdict": "NA", "status": "draft",
                "notes": json.dumps({"written_to_fill": "GUIDELINES.md category gap", "verdict_note": note},
                                    ensure_ascii=False),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb-sources", default=str(ROOT / "data" / "medical_kb" / "SOURCES.csv"))
    args = ap.parse_args()
    kb_files = {}
    if Path(args.kb_sources).exists():
        with open(args.kb_sources, encoding="utf-8", newline="") as f:
            kb_files = {r["url"].rstrip("/"): r["filename"] for r in csv.DictReader(f)}
    else:
        print(f"WARNING: {args.kb_sources} not found; in_kb is 'no' for every Indian fact-check row")

    rows, seen, dropped = [], set(), []
    for r in curated_rows() + indian_rows(kb_files) + gap_rows():
        key = norm(r["claim_text"])
        if key in seen:
            dropped.append(r["id"])
            continue
        seen.add(key)
        rows.append(r)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids"
    with open(HERE / "golden_v1.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to golden_v1.csv; dropped duplicate claims: {dropped or 'none'}")


if __name__ == "__main__":
    main()
