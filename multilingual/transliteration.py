"""Fully local neural Roman-Hindi → Devanagari transliteration."""

from __future__ import annotations

from typing import Protocol, Sequence

from .types import TokenTag


class Transliterator(Protocol):
    def transliterate(self, token: str, language: str = "hi") -> str: ...

    def transliterate_text(self, text: str, tags: Sequence[TokenTag], language: str = "hi") -> str: ...


class QwenHinglishTransliterator:
    """Open local neural transliteration, with no term dictionary or paid API.

    The previous IndicXlit package requires legacy Fairseq and is not usable on
    Windows. This instruction model runs locally.

    It is deliberately called once for the entire code-mixed claim, rather
    than once per word.  Word-at-a-time generation loses grammar and was the
    cause of outputs such as ``कार्टा हाई``.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", *, local_files_only: bool = False,
                 device: str | None = None) -> None:
        self.model_name, self.local_files_only, self.device = model_name, local_files_only, device
        self._model = self._tokenizer = self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install multilingual/requirements.txt to use neural transliteration.") from exc
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=self.local_files_only)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, local_files_only=self.local_files_only).to(self.device)
        self._model.eval(); self._torch = torch

    def transliterate(self, token: str, language: str = "hi") -> str:
        """Compatibility method for callers that truly have one Hindi token."""
        return self.transliterate_text(token, (), language)

    def transliterate_text(self, text: str, tags: Sequence[TokenTag], language: str = "hi") -> str:
        if language != "hi":
            raise RuntimeError(f"No configured neural transliterator for language {language!r}.")
        self._load()
        protected = [tag.token for tag in tags if tag.label in {"en", "number_unit", "entity"}]
        messages = [
            {"role": "system", "content": (
                "You convert code-mixed Hindi-English claims into Hindi in Devanagari. "
                "Transliterate only the Hindi words written in Roman letters; do not translate them. "
                "Keep every English word, medical name, number, URL, and punctuation exactly unchanged. "
                "Return only the completed sentence, with no explanation."
            )},
            {"role": "user", "content": f"Protected exact tokens: {protected}\nClaim: {text}"},
        ]
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            output = self._model.generate(**inputs, do_sample=False, max_new_tokens=max(32, len(text) * 3))
        value = self._tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        if not value:
            raise RuntimeError("Neural transliterator returned empty output.")
        if tags and any(tag.label == "hi_Latn" for tag in tags) and not any("\u0900" <= char <= "\u097f" for char in value):
            raise RuntimeError("Neural transliterator did not return Devanagari for Romanized Hindi.")
        # The model is not allowed to rewrite protected source tokens.  Reject
        # instead of passing a silently altered medicine name or dose onward.
        cursor = 0
        for token in protected:
            found = value.find(token, cursor)
            if found < 0:
                raise RuntimeError(f"Neural transliterator changed protected token {token!r}.")
            cursor = found + len(token)
        return value
