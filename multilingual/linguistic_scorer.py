"""Neural multilingual sensationalism scoring with no language rules."""

from __future__ import annotations


class DevanagariAwareScorer:
    _LABELS = ("sensational or manipulative health misinformation framing", "neutral evidence-based health claim")

    def __init__(self, model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", *, local_files_only: bool = False) -> None:
        self.model_name, self.local_files_only, self._pipeline = model_name, local_files_only, None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install multilingual/requirements.txt to use neural scoring.") from exc
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=self.local_files_only)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, local_files_only=self.local_files_only
        )
        self._pipeline = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

    def calculate_score_detailed(self, text: str) -> dict[str, object]:
        self._load()
        result = self._pipeline(text, candidate_labels=list(self._LABELS), multi_label=False)
        scores = dict(zip(result["labels"], result["scores"]))
        score = float(scores[self._LABELS[0]])
        return {"score": round(score, 4), "breakdown": {"neural_sensationalism": round(score, 4)}}

    def calculate_score(self, text: str) -> float:
        return float(self.calculate_score_detailed(text)["score"])
