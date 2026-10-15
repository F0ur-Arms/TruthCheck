from src.evidence_quality import EvidenceQualityScorer


class FakeEngine:
    """Deterministic stand-in for EvidenceEngine: no models, fixed NLI outcome."""

    def __init__(self, nli_verdict="SUPPORTS", supports_p=0.95, refutes_p=0.02, quality=0.9):
        self.nli_verdict = nli_verdict
        self.supports_p = supports_p
        self.refutes_p = refutes_p
        self.quality = quality
        self.scorer = EvidenceQualityScorer()
        self.scored_claims = []

    def make_lanes(self, claim, subject=None, outcome=None):
        return {"support": claim}

    def retrieve(self, claim, lanes, top_per_lane=5):
        return [{"passage": f"evidence about {claim}", "passage_index": i, "rerank_score": 1.0 - i / 10}
                for i in range(3)]

    def score(self, claim, reranked, subject=None, outcome=None, population=None):
        self.scored_claims.append(claim)
        scored = [{
            "passage": r["passage"], "passage_index": r["passage_index"], "source_tier": "who",
            "source_file": "who_x.txt", "nli_verdict": self.nli_verdict, "quality_score": self.quality,
            "supports_probability": self.supports_p, "refutes_probability": self.refutes_p,
            "rerank_score": r["rerank_score"],
        } for r in reranked]
        return scored, [self.supports_p] * len(scored), self.scorer.summarize_evidence(scored)
