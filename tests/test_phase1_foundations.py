from config import BASE_DIR, FACTS_JSON, KB_PATH, LOG_PATH
from src.llm_fallback import ConfiguredLLMVerifier
from src.risk_engine import RiskEngine


def test_runtime_paths_are_repository_relative():
    assert FACTS_JSON.is_file()
    assert KB_PATH.is_dir()
    assert LOG_PATH.parent == BASE_DIR


def test_domain_mismatched_ml_signal_is_disabled():
    engine = RiskEngine()
    assert engine.has_ml is False
    assert engine.ml_weight == 0.0


def test_llm_fallback_requires_explicit_configuration():
    verifier = ConfiguredLLMVerifier()
    assert verifier.verify("A claim without configured LLM credentials") is None
