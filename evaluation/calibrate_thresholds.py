"""Check how well the pipeline's confidence matches its accuracy, and pick an answer threshold honestly.

Reads merged_predictions.json from a --details run (run_graph_eval.py). A claim's confidence is the
lowest sub-claim confidence, as graph/nodes.py reports it. For the claims the run answered it prints:

- a reliability table (confidence band -> share correct) and the expected calibration error (ECE);
- a threshold sweep: answering only claims at or above a confidence, with coverage, accuracy and
  balanced accuracy at each step;
- a threshold CHOSEN on one random half of the claims (the highest coverage whose accuracy reaches
  --target) and REPORTED on the other half, so the threshold is not tuned on the claims it is scored on.

A threshold can only withhold answers, never add them. The fit is meaningful on a reviewed golden set
(report Step 5); on the public benchmark it is only a dry run, because those labels are not reviewed.

usage: python evaluation/calibrate_thresholds.py RUN/merged_predictions.json [--target 0.90] [--seed 13]
"""

import argparse
import json
import random


def confidence(rec):
    values = [c.get("confidence") for c in rec.get("claims", []) if c.get("confidence") is not None]
    return min(values) if values else None


def stats(rows, n_total):
    if not rows:
        return {"answered": 0, "coverage": 0.0, "accuracy": 0.0, "balanced": 0.0}
    rec = {c: sum(r["pred"] == c for r in rows if r["true"] == c) / max(1, sum(r["true"] == c for r in rows))
           for c in (0, 1)}
    return {"answered": len(rows), "coverage": len(rows) / n_total,
            "accuracy": sum(r["pred"] == r["true"] for r in rows) / len(rows), "balanced": (rec[0] + rec[1]) / 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--target", type=float, default=0.90, help="accuracy the chosen threshold must reach")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    recs = json.load(open(args.predictions, encoding="utf-8"))
    answered = [dict(r, conf=confidence(r)) for r in recs if r["pred"] is not None]
    missing = sum(r["conf"] is None for r in answered)
    answered = [r for r in answered if r["conf"] is not None]
    if missing:
        print(f"WARNING: {missing} answered claims have no sub-claim confidence (run without --details?) and are skipped")
    if not answered:
        raise SystemExit("no answered claims with a confidence")

    print(f"{len(recs)} claims, {len(answered)} answered with a confidence")
    print("\nreliability (answered claims)")
    print("  confidence      claims  correct   mean conf")
    ece = 0.0
    for lo in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
        hi = {0.0: 0.5, 0.5: 0.6, 0.6: 0.7, 0.7: 0.8, 0.8: 0.9, 0.9: 1.01}[lo]
        band = [r for r in answered if lo <= r["conf"] < hi]
        if not band:
            continue
        acc = sum(r["pred"] == r["true"] for r in band) / len(band)
        mean = sum(r["conf"] for r in band) / len(band)
        ece += len(band) / len(answered) * abs(acc - mean)
        print(f"  {lo:.1f}-{min(hi, 1.0):.1f}      {len(band):6}   {acc:6.1%}     {mean:.3f}")
    print(f"expected calibration error (ECE): {ece:.3f}  (0 = confidence equals accuracy)")

    thresholds = sorted({round(r["conf"], 2) for r in answered})
    print("\nthreshold sweep (all claims)")
    print("  answer if conf >=  answered  coverage  accuracy  balanced")
    for t in [0.0] + [x for x in thresholds if x > 0][:: max(1, len(thresholds) // 12)]:
        s = stats([r for r in answered if r["conf"] >= t], len(recs))
        print(f"  {t:17.2f}  {s['answered']:8}  {s['coverage']:8.1%}  {s['accuracy']:8.1%}  {s['balanced']:.3f}")

    rng = random.Random(args.seed)
    ids = sorted({r["idx"] for r in recs})
    rng.shuffle(ids)
    half = set(ids[: len(ids) // 2])
    fit = [r for r in answered if r["idx"] in half]
    test = [r for r in answered if r["idx"] not in half]
    n_fit, n_test = len(half), len(ids) - len(half)
    best = None
    for t in [0.0] + thresholds:
        s = stats([r for r in fit if r["conf"] >= t], n_fit)
        if s["answered"] and s["accuracy"] >= args.target and (best is None or s["coverage"] > best[1]["coverage"]):
            best = (t, s)
    print(f"\nthreshold for accuracy >= {args.target:.0%}, chosen on half A ({n_fit} claims), reported on half B ({n_test})")
    if best is None:
        print("  no threshold reaches the target on half A")
        return
    t, s = best
    held = stats([r for r in test if r["conf"] >= t], n_test)
    base = stats(test, n_test)
    print(f"  threshold {t:.2f}: half A coverage {s['coverage']:.1%} accuracy {s['accuracy']:.1%}")
    print(f"  half B without threshold: coverage {base['coverage']:.1%} accuracy {base['accuracy']:.1%} balanced {base['balanced']:.3f}")
    print(f"  half B with threshold:    coverage {held['coverage']:.1%} accuracy {held['accuracy']:.1%} balanced {held['balanced']:.3f}")


if __name__ == "__main__":
    main()
