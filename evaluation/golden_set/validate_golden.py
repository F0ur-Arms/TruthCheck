"""Check golden_v1.csv against GUIDELINES.md. Exits non-zero if the hard rules fail."""

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VERDICTS = {"SUPPORTED", "SUPPORTED_WITH_CAVEATS", "MIXED_EVIDENCE", "MISLEADING",
            "MOSTLY_FALSE", "FALSE", "INSUFFICIENT_EVIDENCE", "NOT_A_FACT_CHECK"}
LANGS = {"en", "hi", "hinglish", "other"}


def norm(text) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def reference_claims() -> dict:
    out = {}
    eval_csv = ROOT / "data" / "final_health_claims.csv"
    if eval_csv.exists():
        out["final_health_claims.csv"] = {norm(c) for c in pd.read_csv(eval_csv)["claim"]}
    facts = ROOT / "data" / "final_verifiedfacts.json"
    if facts.exists():
        out["final_verifiedfacts.json"] = {
            norm(f.get("claim_text") or f.get("claim") or "") for f in json.load(open(facts, encoding="utf-8"))
        }
    cand = ROOT / "evaluation" / "golden_set" / "golden_candidates_v0.jsonl"
    if cand.exists():
        out["golden_candidates_v0.jsonl"] = {
            norm(json.loads(l)["original_text"]) for l in open(cand, encoding="utf-8") if l.strip()
        }
    return out


def main(path: str) -> int:
    df = pd.read_csv(path)
    n = len(df)
    problems, warnings = [], []

    bad = df[~df["expected_verdict"].isin(VERDICTS)]
    if len(bad):
        problems.append(f"{len(bad)}/{n} rows have a verdict outside the 8-verdict taxonomy: "
                        f"{bad['expected_verdict'].value_counts().to_dict()}")

    lang = df["language"].fillna("").str.lower().map({"english": "en", "hindi": "hi"}).fillna(df["language"])
    if n and (lang == "en").mean() > 0.6:
        problems.append(f"language mix: {(lang == 'en').mean():.0%} English, "
                        f"{(lang == 'hinglish').mean():.0%} Hinglish, {(lang == 'hi').mean():.0%} Hindi "
                        "(target about 40/30/20)")

    needs_ev = df["expected_verdict"].isin(VERDICTS - {"INSUFFICIENT_EVIDENCE", "NOT_A_FACT_CHECK"})
    no_quote = df["gold_evidence_quote"].isna() | (df["gold_evidence_quote"].astype(str).str.strip() == "")
    if (needs_ev & no_quote).sum():
        problems.append(f"{(needs_ev & no_quote).sum()} verdict rows have no gold_evidence_quote")

    unreviewed = (~df["status"].isin(["labeled", "double_checked", "final"])).sum()
    if unreviewed:
        problems.append(f"{unreviewed}/{n} rows are not labeled yet (status={df['status'].value_counts().to_dict()})")
    if df["annotator"].isna().all():
        problems.append("no annotator recorded on any row")

    dups = df["claim_text"].map(norm).duplicated().sum()
    if dups:
        warnings.append(f"{dups} duplicate claim texts")

    ids = df["id"]
    if ids.duplicated().any():
        problems.append(f"{ids.duplicated().sum()} duplicate ids")

    norm_claims = df["claim_text"].map(norm)
    for name, ref in reference_claims().items():
        overlap = norm_claims.isin(ref).sum()
        if overlap:
            problems.append(f"{overlap} claims overlap {name} (rule 1: no overlap)")

    shared = df["second_verdict"].notna().sum()
    if shared < 0.2 * n:
        warnings.append(f"only {shared} rows double-annotated (target 20% = {int(0.2 * n)})")
    if (df["claim_type"] == "health_claim").all():
        warnings.append("claim_type is 'health_claim' on every row; guidelines expect the specific types")

    print(f"{path}: {n} rows")
    for p in problems:
        print("FAIL ", p)
    for w in warnings:
        print("WARN ", w)
    if not problems:
        print("OK: all hard rules pass")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "evaluation/golden_set/golden_v1.csv")))
