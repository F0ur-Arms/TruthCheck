"""Local IndicTrans2 derived-English-gloss backend."""

from __future__ import annotations

from typing import Protocol


class TranslationBackend(Protocol):
    def translate(self, text: str, source_language: str, target_language: str = "eng_Latn") -> str: ...


class _PurePythonIndicProcessor:
    """Small Windows-safe subset of IndicProcessor used for inference.

    IndicTrans2 needs language tags before tokenisation. The official toolkit
    provides richer normalisation and entity maps, but its current release uses
    a Cython extension and explicitly does not support Windows. This fallback
    supplies the required tag format while the package's safety layer preserves
    numbers and URLs after generation and checks neural semantic similarity.
    """

    @staticmethod
    def preprocess_batch(texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        # The downloaded IndicTrans2 tokenizer expects plain source/target
        # language tags, for example ``hin_Deva eng_Latn ...``.
        return [f"{src_lang} {tgt_lang} {text}" for text in texts]

    @staticmethod
    def postprocess_batch(texts: list[str], lang: str) -> list[str]:
        return texts


class IndicTrans2Backend:
    """Lazy local inference wrapper for AI4Bharat IndicTrans2.

    No hosted API is used.  The Hugging Face checkpoint may be cached locally
    first or downloaded by Transformers when ``local_files_only`` is False.
    """

    def __init__(
        self,
        model_name: str = "ai4bharat/indictrans2-indic-en-dist-200M",
        *,
        local_files_only: bool = False,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.device = device
        self._model = self._tokenizer = self._processor = self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional runtime backend
            raise RuntimeError(
                "IndicTrans2 dependencies are unavailable. Install multilingual/requirements.txt."
            ) from exc

        try:  # Prefer upstream preprocessing where its native extension works.
            from IndicTransToolkit.processor import IndicProcessor
            processor = IndicProcessor(inference=True)
        except ImportError:
            processor = _PurePythonIndicProcessor()

        chosen_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, local_files_only=self.local_files_only
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, trust_remote_code=True, local_files_only=self.local_files_only
        ).to(chosen_device)
        self._model.eval()
        self._processor = processor
        self._torch = torch
        self.device = chosen_device

    def translate(self, text: str, source_language: str, target_language: str = "eng_Latn") -> str:
        if source_language == target_language:
            return text
        self._load()
        prepared = self._processor.preprocess_batch([text], src_lang=source_language, tgt_lang=target_language)
        inputs = self._tokenizer(
            prepared, truncation=True, padding=True, return_tensors="pt", return_attention_mask=True
        ).to(self.device)
        with self._torch.no_grad():
            # The upstream checkpoint's cached remote model predates the
            # Transformers cache-object change. Disabling KV caching avoids its
            # ``NoneType.shape`` failure on current Windows installations.
            generated = self._model.generate(
                **inputs, use_cache=False, min_length=0, max_length=256, num_beams=5
            )
        decoded = self._tokenizer.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self._processor.postprocess_batch(decoded, lang=target_language)[0]
