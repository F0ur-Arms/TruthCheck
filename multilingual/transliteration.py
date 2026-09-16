"""Fully local neural Roman-Hindi → Devanagari transliteration."""

from __future__ import annotations

import re
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
        # A low-confidence English label is often a Roman-Hindi false positive
        # (for example ``nimbu``).  Only preserve high-confidence English terms;
        # doses and URLs remain protected regardless of confidence.
        protected = [
            tag.token for tag in tags
            if tag.label in {"number_unit", "entity"}
            or (tag.label == "en" and tag.confidence >= 0.90)
        ]
        value = self._transliterate_hindi_spans(text, tags)
        value = self._clean_generated_text(value)
        if not value:
            raise RuntimeError("Neural transliterator returned empty output.")
        if tags and any(tag.label == "hi_Latn" for tag in tags) and not any("\u0900" <= char <= "\u097f" for char in value):
            raise RuntimeError("Neural transliterator did not return Devanagari for Romanized Hindi.")
        # Protected tokens never enter a model prompt: the original text is
        # reassembled around each generated Hindi span. Verify that invariant
        # anyway before returning a canonical claim.
        for token in protected:
            if not self._contains_protected_token(value, token):
                raise RuntimeError(f"Neural transliterator changed protected token {token!r}.")
        return value

    def _transliterate_hindi_spans(self, text: str, tags: Sequence[TokenTag]) -> str:
        """Generate only Roman-Hindi spans and preserve all other source text."""
        return self._reassemble_hindi_spans(text, tags, self._transliterate_hindi_span)

    @staticmethod
    def _reassemble_hindi_spans(text: str, tags: Sequence[TokenTag], transliterate_span) -> str:
        """Apply ``transliterate_span`` only to Hindi-tagged source spans."""
        output: list[str] = []
        cursor = 0
        index = 0
        while index < len(tags):
            tag = tags[index]
            if tag.label != "hi_Latn":
                index += 1
                continue

            start = tag.start
            end = tag.end
            index += 1
            while index < len(tags) and tags[index].label == "hi_Latn":
                end = tags[index].end
                index += 1

            output.append(text[cursor:start])
            output.append(transliterate_span(text[start:end]))
            cursor = end
        output.append(text[cursor:])
        return "".join(output)

    def _transliterate_hindi_span(self, text: str) -> str:
        messages = self._messages(text, ())
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            output = self._model.generate(**inputs, do_sample=False, max_new_tokens=max(32, len(text) * 3))
        return self._clean_generated_text(
            self._tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        )

    @staticmethod
    def _messages(text: str, protected: Sequence[str]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": """You transliterate code-mixed Hinglish health claims into Hindi written in Devanagari.

Rules:
1. Transliterate Romanized Hindi words into Devanagari: `ke liye` -> `के लिए`, `kharab` -> `खराब`, `hai` -> `है`, and `theek` -> `ठीक`.
2. Do not translate the Hindi words into English.
3. Keep every Protected Token exactly unchanged in Latin script. Protected Tokens include English medical words, drug names, dosage numbers/units, URLs, and explicitly protected English terms such as `doctor`, `metformin`, `500mg`, `protein`, and `kidney`.
4. Keep punctuation unchanged.
5. Output only the transliterated sentence: no Markdown fences, explanation, label, or quotation marks.

Examples:
Protected: ['protein', 'kidney'] | Input: protein kidney ke liye kharab hai -> protein kidney के लिए खराब है
Protected: ['doctor', 'metformin', '500mg'] | Input: doctor se bina pooche metformin 500mg band mat karo -> doctor से बिना पूछे metformin 500mg बंद मत करो
Protected: ['garlic', 'viruses'] | Input: garlic sabhi viruses ko khatam kar deta hai -> garlic सभी viruses को खत्म कर देता है
Protected: ['diabetes'] | Input: subah khali pet nimbu pani diabetes ko theek karta hai -> सुबह खाली पेट नींबू पानी diabetes को ठीक करता है"""},
            {"role": "user", "content": (
                f"Protected Tokens: {list(protected)}\n"
                f"Input: {text}"
            )},
        ]

    @staticmethod
    def _clean_generated_text(value: str) -> str:
        """Remove presentation wrappers without changing the actual claim."""
        value = value.strip()
        value = re.sub(r"^```(?:[A-Za-z0-9_-]+)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1].strip()
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _contains_protected_token(value: str, token: str) -> bool:
        """Match a protected token case-insensitively, but never as a substring."""
        return re.search(rf"(?<!\w){re.escape(token)}(?!\w)", value, flags=re.IGNORECASE) is not None
