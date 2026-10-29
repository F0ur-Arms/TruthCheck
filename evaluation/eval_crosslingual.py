"""Cross-lingual consistency of the NLI judge.

Pairs each translated (Hinglish / romanized Hindi) row of a golden CSV with its English original
(same evidence quote and label) and judges three versions against the same evidence: the English
claim, the translated claim as written, and the translated claim after the pipeline's
HinglishMapper normalisation. Reports accuracy per version and how often the translated versions
agree with the English one.

With --pipeline it also runs the claims through the pipeline's LLM routing/decomposition first and
judges the resulting canonical claims, as the graph does: English, translated (decomposition sees the
normaliser output, the current graph behaviour) and translated with the raw text sent to the LLM.
"""

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessor import HinglishMapper
from src.RAGpipeline.llm_nli import SYSTEM, SYSTEM_V1, LLMNLIJudge

_TO_VERDICT = {"SUPPORTS": "SUPPORTED", "REFUTES": "FALSE", "NEI": "INSUFFICIENT_EVIDENCE"}
_SUPPORT = {"SUPPORTED", "MOSTLY_SUPPORTED", "SUPPORTED_WITH_CAVEATS"}
_REFUTE = {"FALSE", "MOSTLY_FALSE", "MISLEADING"}


def combine(labels: list) -> str:
    """Combine per-sub-claim judge labels the way the graph combines sub-claim verdicts."""
    from graph.nodes import aggregate_verdict
    if not labels or "ERROR" in labels:
        return "ERROR" if "ERROR" in labels else "NEI"
    v = aggregate_verdict([_TO_VERDICT[l] for l in labels])
    return "SUPPORTS" if v in _SUPPORT else "REFUTES" if v in _REFUTE else "NEI"

GOLD = {"TRUE": "SUPPORTS", "FALSE": "REFUTES", "UNCERTAIN": "NEI"}


def build_pairs(df: pd.DataFrame) -> list:
    en = df[df["language"] == "English"]
    tr = df[df["language"].isin(["hinglish", "hindi"]) & df["gold_evidence_quote"].notna()
            & df["expected_verdict"].isin(GOLD)]
    index = {}
    for _, r in en[en["gold_evidence_quote"].notna()].iterrows():
        index.setdefault((r["gold_evidence_quote"].strip(), r["expected_verdict"]), []).append(r)
    pairs = []
    for _, r in tr.iterrows():
        match = index.get((r["gold_evidence_quote"].strip(), r["expected_verdict"]), [])
        if len(match) == 1:
            pairs.append({"id": r["id"], "language": r["language"], "gold": GOLD[r["expected_verdict"]],
                          "evidence": r["gold_evidence_quote"], "en": match[0]["claim_text"],
                          "translated": r["claim_text"]})
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", choices=["v1", "v2"], default="v1")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pipeline", action="store_true", help="also judge the pipeline's decomposed claims")
    args = ap.parse_args()

    pairs = build_pairs(pd.read_csv(args.csv))
    mapper = HinglishMapper()
    for p in pairs:
        p["mapped"] = mapper.clean_text(p["translated"])

    base_url = os.environ.get("TRUTHCHECK_LLM_BASE_URL", "http://cb-gpu-server:8000/v1")
    for attempt in range(80):
        try:
            if requests.get(f"{base_url}/models", timeout=5).ok:
                break
        except requests.RequestException:
            pass
        time.sleep(15)
    else:
        sys.exit("LLM server not reachable after 20 minutes")

    judge = LLMNLIJudge(base_url=base_url,
                        api_key="none", model=os.environ.get("TRUTHCHECK_LLM_MODEL", "qwen2.5-7b-awq"),
                        system_prompt=SYSTEM_V1 if args.prompt == "v1" else SYSTEM)

    def run(item):
        try:
            return judge.verify(*item)["verdict"]
        except Exception:
            return "ERROR"

    with ThreadPoolExecutor(args.workers) as ex:
        for version in ("en", "translated", "mapped"):
            labels = list(ex.map(run, [(p[version], p["evidence"]) for p in pairs]))
            for p, lab in zip(pairs, labels):
                p[f"pred_{version}"] = lab

    versions = ["en", "translated", "mapped"]
    fallbacks = {}
    if args.pipeline:
        from src.claims_processor import ClaimsProcessor
        settings = {
            "en_pipeline": lambda p: (p["en"], mapper.clean_text(p["en"])),
            "translated_pipeline": lambda p: (p["translated"], p["mapped"]),
            "translated_pipeline_raw": lambda p: (p["translated"], p["translated"]),
        }
        for name, make_input in settings.items():
            cp = ClaimsProcessor()

            def decompose_and_judge(p, cp=cp, make_input=make_input):
                rep_ = cp.process_query(*make_input(p))
                if rep_.route == "medical_advice":
                    return "NEI", ["<medical_advice>"]
                claims = [c.canonical_claim for c in rep_.claims]
                return combine([run((c, p["evidence"])) for c in claims]), claims

            with ThreadPoolExecutor(args.workers) as ex2:
                results = list(ex2.map(decompose_and_judge, pairs))
            for p, (lab, claims) in zip(pairs, results):
                p[f"pred_{name}"] = lab
                p[f"claims_{name}"] = claims
            fallbacks[name] = cp.fallback_count
            versions.append(name)

    json.dump(pairs, open(args.out, "w"), ensure_ascii=False)
    n = len(pairs)
    errors = sum(p[f"pred_{v}"] == "ERROR" for p in pairs for v in versions)
    if errors > 0.05 * len(versions) * n:
        sys.exit(f"{errors} of {len(versions) * n} judge calls failed; results not reported")
    print(f"pairs: {n} {dict(Counter(p['language'] for p in pairs))}  prompt: {args.prompt}")
    for v in versions:
        ref = "pred_en_pipeline" if v.startswith("translated_pipeline") else "pred_en"
        acc = sum(p[f"pred_{v}"] == p["gold"] for p in pairs) / n
        err = sum(p[f"pred_{v}"] == "ERROR" for p in pairs)
        agree = sum(p[f"pred_{v}"] == p[ref] for p in pairs) / n
        extra = f"  decomposition fallbacks {fallbacks[v]}" if v in fallbacks else ""
        print(f"  {v:24} accuracy {acc:.3f}  agrees with English ({ref[5:]}) {agree:.3f}  errors {err}{extra}")
    for lang in sorted({p["language"] for p in pairs}):
        sub = [p for p in pairs if p["language"] == lang]
        print(f"  {lang:10} n={len(sub)}: EN acc {sum(p['pred_en'] == p['gold'] for p in sub) / len(sub):.3f}, "
              f"translated acc {sum(p['pred_translated'] == p['gold'] for p in sub) / len(sub):.3f}, "
              f"agreement {sum(p['pred_translated'] == p['pred_en'] for p in sub) / len(sub):.3f}")
    flips = Counter((p["pred_en"], p["pred_translated"]) for p in pairs if p["pred_en"] != p["pred_translated"])
    print("  label changes EN -> translated:", flips.most_common(6))


if __name__ == "__main__":
    main()
