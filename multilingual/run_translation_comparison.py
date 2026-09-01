"""Compare the existing dictionary normaliser with the new neural route.

Usage from the TruthCheck repository root:
    python -m multilingual.run_translation_comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.preprocessor import HinglishMapper

from .processor import MultilingualProcessor, MultilingualProcessorConfig
from .run_output_dump import SAMPLE_CLAIMS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write old-versus-new multilingual translations.")
    parser.add_argument("--offline", action="store_true", help="Use cached model weights only.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N samples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    claims = SAMPLE_CLAIMS[:args.limit] if args.limit is not None else SAMPLE_CLAIMS
    old_mapper = HinglishMapper()
    processor = MultilingualProcessor(MultilingualProcessorConfig(
        indictrans_local_files_only=args.offline,
        neural_models_local_files_only=args.offline,
        enable_linguistic_score=False,
    ))
    output_path = Path(__file__).with_name("translation_comparison_dump.jsonl")

    with output_path.open("w", encoding="utf-8") as stream:
        for index, text in enumerate(claims, start=1):
            old_translation = old_mapper.clean_text(text)
            try:
                result = processor.process(text)
                new_translation = result.english_gloss or ""
            except Exception:
                # Keep exactly the requested fields. An empty new value means
                # the neural safety gate withheld an untrusted gloss.
                new_translation = ""
            stream.write(json.dumps({
                "input": text,
                "old_translated": old_translation,
                "new_translated": new_translation,
            }, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(claims)}] {text}")

    print(f"\nWrote {len(claims)} comparison record(s) to: {output_path}")


if __name__ == "__main__":
    main()
