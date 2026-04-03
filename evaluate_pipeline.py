import argparse
import json
from collections import Counter

import pandas as pd
import spacy
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from src.lifestyle_ner import build_lifestyle_ner
from src.preprocessor import HinglishMapper
from src.refine_extractor import extract_triples
from src.risk_engine import RiskEngine


def try_load_full_pipeline():
    try:
        from main import TruthCheckPipeline, get_binary_prediction

        pipeline = TruthCheckPipeline()
        return pipeline, get_binary_prediction, None
    except Exception as exc:
        return None, None, exc


def compute_metrics(y_true, y_pred, title):
    return {
        "title": title,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_weighted": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "recall_weighted": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Real (0)", "Fake (1)"],
            zero_division=0,
        ),
    }


def build_lightweight_components():
    mapper = HinglishMapper()
    nlp = spacy.load("en_core_web_sm")
    nlp = build_lifestyle_ner(nlp)
    engine = RiskEngine()
    return mapper, nlp, engine


def evaluate_claims(df, sample_size=None):
    if sample_size is not None:
        df = df.head(sample_size).copy()

    y_true = df["label"].astype(int).tolist()
    mapper, nlp, engine = build_lightweight_components()
    full_pipeline, prediction_fn, pipeline_error = try_load_full_pipeline()

    extraction_success = 0
    extraction_fail = 0
    triple_count_distribution = Counter()
    heuristic_preds = []
    pipeline_preds = []
    verifier_available = full_pipeline is not None
    verifier_report_count = 0
    verifier_verdicts = Counter()

    for claim in df["claim"].astype(str):
        cleaned = mapper.clean_text(claim)
        triples = extract_triples(cleaned, nlp)
        triple_count_distribution[len(triples)] += 1

        if triples:
            extraction_success += 1
        else:
            extraction_fail += 1

        style_score = full_pipeline.scorer.calculate_score(claim) if full_pipeline else 0.0

        if triples:
            heuristic_fact = {"verdict": "UNVERIFIED", "match_type": "NONE"}
            if full_pipeline:
                reports = full_pipeline.analyze_query(claim)
                pipeline_preds.append(prediction_fn(reports))
                if reports and "error" not in reports[0]:
                    verifier_report_count += len(reports)
                    for report in reports:
                        verifier_verdicts[report.get("verdict", "UNKNOWN")] += 1
                else:
                    verifier_verdicts["PIPELINE_ERROR"] += 1
            else:
                reports = []

            if not full_pipeline:
                risk = engine.calculate_risk(claim, style_score, heuristic_fact)
                heuristic_preds.append(1 if risk["score"] >= 0.5 else 0)
        else:
            if full_pipeline:
                pipeline_preds.append(1)
                verifier_verdicts["NO_TRIPLE"] += 1
            else:
                risk = engine.calculate_risk(claim, style_score, {"verdict": "UNVERIFIED", "match_type": "NONE"})
                heuristic_preds.append(1 if risk["score"] >= 0.5 else 0)

    results = {
        "rows_evaluated": len(df),
        "extraction": {
            "success": extraction_success,
            "fail": extraction_fail,
            "coverage": round(extraction_success / len(df), 4) if len(df) else 0.0,
            "triple_count_distribution": dict(sorted(triple_count_distribution.items())),
        },
        "verifier": {
            "available": verifier_available,
            "error": None if verifier_available else str(pipeline_error),
            "report_count": verifier_report_count,
            "verdict_distribution": dict(verifier_verdicts),
        },
        "metrics": [],
    }

    if pipeline_preds:
        results["metrics"].append(compute_metrics(y_true, pipeline_preds, "Full pipeline final prediction"))

    if heuristic_preds:
        results["metrics"].append(compute_metrics(y_true, heuristic_preds, "Heuristic fallback final prediction"))

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate TruthCheck stage by stage on health claims.")
    parser.add_argument("--data", default="data/final_health_claims.csv", help="CSV with claim,label columns")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional number of rows to evaluate")
    parser.add_argument("--output", default=None, help="Optional JSON file to save results")
    args = parser.parse_args()

    df = pd.read_csv(args.data).dropna(subset=["claim", "label"])
    results = evaluate_claims(df, sample_size=args.sample_size)

    print("\n=== Extraction ===")
    print(json.dumps(results["extraction"], indent=2))

    print("\n=== Verifier ===")
    print(json.dumps(results["verifier"], indent=2))

    for metric in results["metrics"]:
        print(f"\n=== {metric['title']} ===")
        print(
            json.dumps(
                {
                    k: v for k, v in metric.items()
                    if k not in {"classification_report"}
                },
                indent=2,
            )
        )
        print(metric["classification_report"])

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved evaluation results to {args.output}")


if __name__ == "__main__":
    main()
