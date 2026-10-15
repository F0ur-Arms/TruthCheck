"""Standardized Section 56 Evaluation Report Generator.

Generates the mandatory Section 56 evaluation report comparing current system performance
against the Phase 0 frozen baseline, reporting accuracy, F1, calibration (ECE/Brier),
safety metrics, and latency deltas.
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, Optional
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE0_BASELINE_FILE = ROOT / "evaluation" / "baselines" / "phase0" / "portable-baseline-utf8.stdout.log"


REQUIRED_METRICS = ("accuracy", "f1_score")
OPTIONAL_METRICS = ("brier_score", "ece", "dangerous_fn_rate", "leakage_rate", "latency_ms", "latency_p95_ms")


def load_phase0_baseline(path: Path = PHASE0_BASELINE_FILE) -> Dict[str, float]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    out = {}
    for key, label in (("accuracy", "Accuracy"), ("f1", "F1 Score")):
        m = re.search(rf"{label}\s*:\s*([0-9.]+)", text)
        if not m:
            raise ValueError(f"Could not find '{label}' in Phase 0 baseline log {path}")
        out[key] = float(m.group(1))
    return out


def _fmt(value, spec=".4f", suffix=""):
    return "not measured" if value is None else f"{format(value, spec)}{suffix}"


def generate_section56_report(
    change_summary: str,
    architecture_why: str,
    files_changed: list[str],
    behavior_desc: str,
    tests_summary: str,
    metrics: Dict[str, Any],
    per_language_accuracy: Optional[Dict[str, float]] = None,
    limitations: str = "None recorded.",
    next_decision: str = "None recorded.",
) -> str:
    """Render the report from measured metrics only.

    Raises ValueError if required metrics are missing. Optional metrics that were
    not measured are printed as "not measured" rather than defaulted.
    """
    missing = [k for k in REQUIRED_METRICS if metrics.get(k) is None]
    if missing:
        raise ValueError(f"Missing required metrics: {missing}. Run the evaluation first and pass its output.")

    baseline = load_phase0_baseline()
    accuracy = metrics["accuracy"]
    f1_score = metrics["f1_score"]
    m = {k: metrics.get(k) for k in OPTIONAL_METRICS}

    lang_lines = (
        "\n".join(f"- {lang}: {acc:.4f}" for lang, acc in per_language_accuracy.items())
        if per_language_accuracy
        else "- not measured (needs the multilingual golden set)"
    )

    report = f"""# TruthCheck v2 — Section 56 Architecture Execution & Benchmark Report

## 1. CHANGE
{change_summary}

## 2. WHY
{architecture_why}

## 3. FILES
""" + "\n".join(f"- `{f}`" for f in files_changed) + f"""

## 4. BEHAVIOR
{behavior_desc}

## 5. TESTS
{tests_summary}

## 6. RESULTS
- Accuracy: {accuracy:.4f}
- F1-Score: {f1_score:.4f}
- Brier Score (Calibration): {_fmt(m["brier_score"])}
- ECE (Expected Calibration Error): {_fmt(m["ece"])}
- Average Latency: {_fmt(m["latency_ms"], ".1f", " ms")}

## 7. BASELINE DELTA (vs Phase 0 Frozen Baseline)
- Accuracy: {baseline["accuracy"]:.4f} -> {accuracy:.4f} (Delta: {accuracy - baseline["accuracy"]:+.4f})
- F1: {baseline["f1"]:.4f} -> {f1_score:.4f} (Delta: {f1_score - baseline["f1"]:+.4f})

## 8. MULTILINGUAL DELTA
{lang_lines}

## 9. SAFETY DELTA
- Dangerous False-Negative Rate: {_fmt(m["dangerous_fn_rate"], ".2%")}
- Medical-Advice Leakage Rate: {_fmt(m["leakage_rate"], ".2%")}

## 10. LATENCY
- Average Request Latency: {_fmt(m["latency_ms"], ".1f", " ms")}
- p95: {_fmt(m["latency_p95_ms"], ".1f", " ms")}

## 11. KNOWN LIMITATIONS
{limitations}

## 12. NEXT DECISION
{next_decision}
"""
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render the Section 56 report from a measured metrics JSON file.")
    parser.add_argument("--metrics", required=True, help="JSON with accuracy, f1_score and optional brier_score, ece, ...")
    parser.add_argument("--summary", required=True, help="Short description of the change")
    parser.add_argument("--why", default="")
    parser.add_argument("--files", nargs="*", default=[])
    parser.add_argument("--behavior", default="")
    parser.add_argument("--tests", default="")
    args = parser.parse_args()

    data = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    print(
        generate_section56_report(
            change_summary=args.summary,
            architecture_why=args.why,
            files_changed=args.files,
            behavior_desc=args.behavior,
            tests_summary=args.tests,
            metrics=data,
            per_language_accuracy=data.get("per_language_accuracy"),
        )
    )
