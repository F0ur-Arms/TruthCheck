import csv
import re
from pathlib import Path


DEFAULT_HINGLISH_REPLACEMENTS = {
    "khali pet": "empty stomach",
    "subah": "morning",
    "raat": "night",
    "garam pani": "warm water",
    "garam paani": "warm water",
    "haldi": "turmeric",
    "kadha": "herbal decoction",
    "giloy": "giloy",
    "nuskha": "home remedy",
    "gharelu": "homemade",
    "pet": "stomach",
    "bukhar": "fever",
    "sardi": "cold",
    "khansi": "cough",
    "motapa": "obesity",
    "vajan": "weight",
    "charbi": "fat",
    "kam": "reduce",
    "badhata": "increases",
    "badhati": "increases",
    "ghatata": "reduces",
    "ghatati": "reduces",
    "theek": "cure",
    "ilaaj": "treatment",
}


class HinglishMapper:
    def __init__(self, lexicon_path="data/hinglish_dict.csv"):
        self.lexicon_path = Path(lexicon_path)
        replacements = self._load_replacements()
        self._replacement_patterns = [
            (re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE), target)
            for src, target in sorted(
                replacements.items(),
                key=lambda item: len(item[0]),
                reverse=True,
            )
        ]

    def _load_replacements(self):
        replacements = dict(DEFAULT_HINGLISH_REPLACEMENTS)

        if not self.lexicon_path.exists():
            return replacements

        with self.lexicon_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                source = (row.get("hinglish") or "").strip().lower()
                target = (row.get("english") or "").strip().lower()
                if source and target:
                    replacements[source] = target

        return replacements

    def _apply_phrase_replacements(self, text):
        normalized = text
        for pattern, replacement in self._replacement_patterns:
            normalized = pattern.sub(replacement, normalized)
        return normalized

    def clean_text(self, text):
        """
        Cleans noisy Hinglish text and applies lexicon-driven phrase normalization
        so extraction and retrieval see more stable English health claims.
        """
        text = text.lower().strip()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = text.replace("-", " ")
        text = self._apply_phrase_replacements(text)
        text = re.sub(r"[^\w\s,.!?]", " ", text)
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
