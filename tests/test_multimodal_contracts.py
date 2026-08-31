import unittest
from src.multimodal_contracts import MultimodalIngestionManager, ImageIngestionResult, AudioIngestionResult


class MultimodalContractTests(unittest.TestCase):
    def setUp(self):
        self.manager = MultimodalIngestionManager(ocr_confidence_threshold=0.70)

    def test_high_confidence_ocr_uses_first_pass(self):
        res = self.manager.process_image(
            raw_input_id="img-101",
            first_pass_text="Curd at night causes cold",
            first_pass_confidence=0.85
        )
        self.assertEqual(res.extraction_method, "paddle_ocr")
        self.assertEqual(res.extracted_text, "Curd at night causes cold")

    def test_low_confidence_ocr_triggers_vlm_fallback(self):
        def mock_vlm(img_id):
            return {
                "extracted_text": "Clean text from noisy screenshot",
                "visual_context": "Infographic showing dietary protein recommendation",
                "confidence": 0.90
            }

        res = self.manager.process_image(
            raw_input_id="img-102",
            first_pass_text="noisy txt",
            first_pass_confidence=0.45,
            vlm_fallback_fn=mock_vlm
        )
        self.assertEqual(res.extraction_method, "vlm_fallback")
        self.assertEqual(res.extracted_text, "Clean text from noisy screenshot")

    def test_audio_ingestion_contract(self):
        res = self.manager.process_audio(
            raw_input_id="aud-201",
            transcript="Subah garam pani peene se weight loss hota hai",
            asr_confidence=0.92,
            model_name="sarvam_saaras_v3"
        )
        self.assertEqual(res.asr_model, "sarvam_saaras_v3")
        self.assertFalse(res.metadata["needs_asr_review"])


if __name__ == "__main__":
    unittest.main()
