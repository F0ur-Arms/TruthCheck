"""Second annotation for golden_v1.csv (GUIDELINES.md, "Double annotation").

make   Write a blind sheet for the second annotator: every 5th row of golden_v1.csv (115 of 574) with
       only id, claim_text and language, so neither the first verdict nor the notes (which hold
       publisher ratings and source summaries) are visible. Refuses to overwrite a sheet that exists.

merge  Copy second_verdict and second_annotator from the filled sheet into golden_v1.csv, matched by
       id, and report agreement and Cohen's kappa on the rows both annotators have labelled. Nothing
       is written if any check fails: unknown or duplicate ids, a verdict outside the 8 labels, a
       verdict without an annotator name, the same person as the first annotator, or a sheet value
       that would replace a different second_verdict already in golden_v1.csv (use --force for that).

kappa  Only report agreement and kappa for golden_v1.csv as it is.

usage: python evaluation/golden_set/second_annotation.py make  [--out SHEET]
       python evaluation/golden_set/second_annotation.py merge SHEET [--dry-run] [--force]
       python evaluation/golden_set/second_annotation.py kappa
"""

import argparse
import csv
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
GOLDEN = HERE / "golden_v1.csv"
DEFAULT_SHEET = HERE / "second_annotation_sheet.csv"
VERDICTS = {"SUPPORTED", "SUPPORTED_WITH_CAVEATS", "MIXED_EVIDENCE", "MISLEADING", "MOSTLY_FALSE", "FALSE",
            "INSUFFICIENT_EVIDENCE", "NOT_A_FACT_CHECK"}
LABELLED = {"labeled", "double_checked", "final"}
SHEET_COLUMNS = ["id", "claim_text", "language", "second_verdict", "second_evidence_source",
                 "second_evidence_quote", "second_annotator", "second_notes"]
EMPTY = {"", "NA", "N/A", "NONE", "NAN"}


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def blank(value):
    return (value or "").strip().upper() in EMPTY


def make(args):
    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} already exists; it may hold the second annotator's work. Move it away first.")
    _, rows = read_csv(GOLDEN)
    shared = [r for i, r in enumerate(rows) if i % 5 == 0]
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        for r in shared:
            writer.writerow({"id": r["id"], "claim_text": r["claim_text"], "language": r["language"]})
    print(f"wrote {len(shared)} of {len(rows)} rows ({len(shared) / len(rows):.0%}) to {out}")
    print("Give the second annotator this file and GUIDELINES.md only, not golden_v1.csv.")


def kappa_report(rows):
    pairs = [(r["expected_verdict"], r["second_verdict"].strip()) for r in rows
             if r.get("status") in LABELLED and r["expected_verdict"] in VERDICTS
             and r.get("second_verdict", "").strip() in VERDICTS]
    with_second = sum(1 for r in rows if r.get("second_verdict", "").strip() in VERDICTS)
    print(f"rows with a second verdict: {with_second}; of those, first annotation finished (status labeled+): {len(pairs)}")
    if not pairs:
        print("kappa needs rows that both annotators have labelled; first-annotator rows still in draft do not count.")
        return
    n = len(pairs)
    agree = sum(a == b for a, b in pairs)
    first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(first[k] * second[k] for k in VERDICTS) / (n * n)
    kappa = 1.0 if expected == 1 else (agree / n - expected) / (1 - expected)
    print(f"agreement {agree}/{n} ({agree / n:.1%}), Cohen's kappa {kappa:.3f}")
    disagreements = Counter((a, b) for a, b in pairs if a != b)
    if disagreements:
        print("disagreements (first -> second):", dict(disagreements.most_common()))
        print("Discuss each one, put the agreed verdict in expected_verdict and the reason in notes.")


def merge(args):
    columns, rows = read_csv(GOLDEN)
    for col in ("second_verdict", "second_annotator"):
        if col not in columns:
            sys.exit(f"golden_v1.csv has no {col} column")
    by_id = {r["id"]: r for r in rows}
    _, sheet = read_csv(args.sheet)
    problems, updates = [], []
    seen = Counter(s["id"] for s in sheet)
    problems += [f"duplicate id in sheet: {i}" for i, c in seen.items() if c > 1]
    for s in sheet:
        rid, verdict, who = s["id"].strip(), s.get("second_verdict", "").strip(), s.get("second_annotator", "").strip()
        if rid not in by_id:
            problems.append(f"{rid}: not in golden_v1.csv")
            continue
        if blank(verdict):
            continue
        row = by_id[rid]
        if s.get("claim_text", "").strip() != row["claim_text"].strip():
            problems.append(f"{rid}: claim_text differs from golden_v1.csv (sheet out of date?)")
        if verdict not in VERDICTS:
            problems.append(f"{rid}: second_verdict {verdict!r} is not one of the 8 verdicts")
        if blank(who):
            problems.append(f"{rid}: second_verdict without second_annotator")
        elif not blank(row.get("annotator")) and who.lower() == row["annotator"].strip().lower():
            problems.append(f"{rid}: second annotator is the same person as the first ({who})")
        old = row.get("second_verdict", "").strip()
        if not blank(old) and old != verdict and not args.force:
            problems.append(f"{rid}: would replace second_verdict {old} with {verdict} (use --force)")
        updates.append((rid, verdict, who))
    if problems:
        print(f"Not merged; {len(problems)} problem(s):")
        for p in problems[:50]:
            print("  -", p)
        sys.exit(1)
    for rid, verdict, who in updates:
        by_id[rid]["second_verdict"], by_id[rid]["second_annotator"] = verdict, who
    print(f"{len(updates)} second verdicts ready to merge")
    kappa_report(rows)
    if args.dry_run or not updates:
        print("golden_v1.csv not changed" + (" (dry run)" if args.dry_run else " (nothing to merge)"))
        return
    fd, tmp = tempfile.mkstemp(dir=HERE, suffix=".csv")
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, GOLDEN)
    print(f"merged into {GOLDEN}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--out", default=str(DEFAULT_SHEET))
    g = sub.add_parser("merge")
    g.add_argument("sheet")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--force", action="store_true")
    sub.add_parser("kappa")
    args = ap.parse_args()
    if args.cmd == "make":
        make(args)
    elif args.cmd == "merge":
        merge(args)
    else:
        kappa_report(read_csv(GOLDEN)[1])


if __name__ == "__main__":
    main()
