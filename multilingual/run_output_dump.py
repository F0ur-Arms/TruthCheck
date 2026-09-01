"""Run representative multilingual claims and write their full outputs to JSONL.

Usage from the TruthCheck repository root:
    python -m multilingual.run_output_dump
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .linguistic_scorer import DevanagariAwareScorer
from .processor import MultilingualProcessor, MultilingualProcessorConfig


SAMPLE_CLAIMS = [
    "Turmeric cures cancer completely.",
    "Some studies suggest that turmeric may have anti-inflammatory properties.",
    "Drinking 500ml of water every morning prevents diabetes.",
    "Doctors do not want you to know that garlic eliminates all viruses!",
    "Metformin 500mg should not be stopped without medical advice.",
    "Vitamin D supplements may help people with a deficiency.",
    "A detox drink removes toxins from the blood instantly.",
    "Exercise and a balanced diet can support cardiovascular health.",
    "हल्दी कैंसर का इलाज नहीं करती।",
    "यह चमत्कारी नुस्खा मधुमेह को हमेशा के लिए ठीक करता है।",
    "कुछ अध्ययनों में विटामिन डी और प्रतिरक्षा के बीच संबंध पाया गया है।",
    "पैरासिटामोल 500mg बुखार कम करने के लिए इस्तेमाल की जाती है।",
    "तुरंत शेयर करें, डॉक्टर आपको यह राज़ नहीं बताएंगे!",
    "संतुलित आहार और नियमित व्यायाम स्वास्थ्य के लिए उपयोगी हो सकते हैं।",
    "नींबू पानी कोरोना को रोकता है।",
    "protein kidney ke liye kharab hai",
    "haldi cancer ko 100% cure karti hai",
    "doctor se bina pooche metformin 500mg band mat karo",
    "subah khali pet nimbu pani diabetes ko theek karta hai",
    "kuch studies kehte hain ki green tea inflammation kam kar sakti hai",
    "ye miracle remedy turant share karo!!!",
    "vitamin D deficiency ke liye blood test karwana useful ho sakta hai",
    "garlic sabhi viruses ko khatam kar deta hai",
    "pregnancy mein iron supplement doctor ki salah se lena chahiye",
    "cold milk acidity ko permanently cure karta hai",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write multilingual results for 25 sample claims.")
    parser.add_argument("--offline", action="store_true", help="Use cached model weights only.")
    parser.add_argument("--no-style-score", action="store_true", help="Skip the mDeBERTa style scorer.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N samples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MultilingualProcessorConfig(
        indictrans_local_files_only=args.offline,
        neural_models_local_files_only=args.offline,
    )
    processor = MultilingualProcessor(config if not args.no_style_score else MultilingualProcessorConfig(
        indictrans_local_files_only=args.offline,
        neural_models_local_files_only=args.offline,
        enable_linguistic_score=False,
    ))
    claims = SAMPLE_CLAIMS[:args.limit] if args.limit is not None else SAMPLE_CLAIMS
    output_path = Path(__file__).with_name("output_dump.jsonl")

    with output_path.open("w", encoding="utf-8") as stream:
        for index, text in enumerate(claims, start=1):
            record: dict[str, object] = {"id": index, "input": text}
            try:
                record["multilingual_claim"] = processor.process(text).to_dict()
            except Exception as exc:
                record["processing_error"] = f"{type(exc).__name__}: {exc}"
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(claims)}] {text}")

    print(f"\nWrote {len(claims)} result record(s) to: {output_path}")


if __name__ == "__main__":
    main()
