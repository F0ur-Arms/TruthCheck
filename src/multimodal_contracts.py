"""Multimodal Ingestion Contracts & Interfaces for OCR (Image) and ASR (Audio).

Provides stable data contracts and confidence-gated routing for external OCR and ASR model outputs.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class ImageIngestionResult(BaseModel):
    raw_input_id: str
    extracted_text: str
    visual_context: Optional[str] = None
    language: str = Field(default="auto")
    script: str = Field(default="auto")
    ocr_confidence: float = Field(default=1.0)
    extraction_method: str = Field(default="paddle_ocr")  # "paddle_ocr", "tesseract", "vlm_fallback"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AudioIngestionResult(BaseModel):
    raw_input_id: str
    transcript: str
    segments: List[Dict[str, Any]] = Field(default_factory=list)
    language: str = Field(default="hi")
    asr_confidence: float = Field(default=1.0)
    asr_model: str = Field(default="sarvam_saaras_v3")  # "sarvam_saaras_v3", "indic_whisper"
    timestamps: List[float] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MultimodalIngestionManager:
    """Manages confidence-gated OCR and ASR routing."""

    def __init__(self, ocr_confidence_threshold: float = 0.65, asr_confidence_threshold: float = 0.60):
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.asr_confidence_threshold = asr_confidence_threshold

    def process_image(
        self,
        raw_input_id: str,
        first_pass_text: str,
        first_pass_confidence: float,
        vlm_fallback_fn=None
    ) -> ImageIngestionResult:
        if first_pass_confidence >= self.ocr_confidence_threshold or not vlm_fallback_fn:
            return ImageIngestionResult(
                raw_input_id=raw_input_id,
                extracted_text=first_pass_text,
                ocr_confidence=first_pass_confidence,
                extraction_method="paddle_ocr"
            )

        # Confidence gate triggered -> invoke VLM fallback
        vlm_res = vlm_fallback_fn(raw_input_id)
        return ImageIngestionResult(
            raw_input_id=raw_input_id,
            extracted_text=vlm_res.get("extracted_text", first_pass_text),
            visual_context=vlm_res.get("visual_context"),
            ocr_confidence=vlm_res.get("confidence", 0.85),
            extraction_method="vlm_fallback"
        )

    def process_audio(
        self,
        raw_input_id: str,
        transcript: str,
        asr_confidence: float,
        segments: Optional[List[Dict[str, Any]]] = None,
        model_name: str = "sarvam_saaras_v3"
    ) -> AudioIngestionResult:
        needs_review = asr_confidence < self.asr_confidence_threshold
        return AudioIngestionResult(
            raw_input_id=raw_input_id,
            transcript=transcript,
            segments=segments or [],
            asr_confidence=asr_confidence,
            asr_model=model_name,
            metadata={"needs_asr_review": needs_review}
        )
