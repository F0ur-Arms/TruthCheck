import json
import os


def normalize_verdict(verdict):
    """Maps dataset verdict variants into a stable internal label set."""
    if verdict is None:
        return "UNVERIFIED"

    normalized = str(verdict).strip().upper()
    alias_map = {
        "MIXED": "PARTIALLY TRUE",
        "PARTIAL": "PARTIALLY TRUE",
        "PARTIALLY_TRUE": "PARTIALLY TRUE",
    }
    return alias_map.get(normalized, normalized)

# ─────────────────────────────────────────────────────────────────────────────
# VERDICT TABLE
#
# Maps (kb_verdict, nli_verdict) → final claim verdict.
#
# kb_verdict  = truth value stored in verified_facts.json for that entry
# nli_verdict = what DistilRoBERTa says about claim vs scientific_truth
#
# Examples:
#   KB: "warm water aids digestion"  verdict=TRUE
#   Claim: "warm water improves digestion"
#   NLI: SUPPORTS → (TRUE, SUPPORTS) → TRUE  ✅
#
#   KB: "warm water aids digestion"  verdict=TRUE
#   Claim: "warm water does NOT help digestion"
#   NLI: REFUTES → (TRUE, REFUTES) → FALSE  ✅
#
#   KB: "No evidence supports cow urine curing cancer"  verdict=FALSE
#   Claim: "cow urine cures cancer"
#   NLI: REFUTES → (FALSE, REFUTES) → FALSE  ✅
#   Claim: "cow urine does not cure cancer"
#   NLI: SUPPORTS → (FALSE, SUPPORTS) → FALSE  ✅
#   (both map to FALSE because the KB entry is a refutation statement —
#    any claim about cow urine curing cancer is false regardless)
# ─────────────────────────────────────────────────────────────────────────────

VERDICT_TABLE = {
    ("TRUE",  "SUPPORTS"): "TRUE",
    ("TRUE",  "REFUTES"):  "FALSE",
    ("TRUE",  "NEI"):      "UNVERIFIED",
    ("FALSE", "SUPPORTS"): "FALSE",
    ("FALSE", "REFUTES"):  "FALSE",
    ("FALSE", "NEI"):      "UNVERIFIED",
    ("PARTIALLY TRUE", "SUPPORTS"): "PARTIALLY TRUE",
    ("PARTIALLY TRUE", "REFUTES"):  "PARTIALLY TRUE",
    ("PARTIALLY TRUE", "NEI"):      "UNVERIFIED",
}

# ─────────────────────────────────────────────────────────────────────────────
# DISTANCE THRESHOLDS
#
# L2 distance from all-MiniLM-L6-v2 (same model used in KnowledgeManager).
# These are empirically reasonable starting values — tune after first run.
#
# Distance = 0.0  → identical sentences
# Distance < 1.0  → strong semantic match
# Distance 1.0-1.5 → weak / partial match
# Distance > 1.5  → semantically unrelated, return UNVERIFIED
# ─────────────────────────────────────────────────────────────────────────────

STRONG_MATCH_DISTANCE = 1.0
WEAK_MATCH_DISTANCE   = 1.5


# ─────────────────────────────────────────────────────────────────────────────
# FACT VERIFIER
# ─────────────────────────────────────────────────────────────────────────────

class FactVerifier:
    def __init__(self, kb_manager, nli_verifier=None):
        """
        Args:
            kb_manager   : KnowledgeManager instance.
                           Must have load_verified_facts() called before this.
            nli_verifier : NLIVerifier instance injected from pipeline.
                           If None, falls back to KB verdict directly (no
                           negation detection).
        """
        self.kb  = kb_manager
        self.nli = nli_verifier

        if self.nli is None:
            print("[FactVerifier] ⚠  No NLI injected — KB verdict only, no negation detection.")
        else:
            print("[FactVerifier] ✅ Semantic search + NLI active — negation-aware.")

    # ─────────────────────────────────────────────────────────────────────────
    # VERIFY
    # ─────────────────────────────────────────────────────────────────────────

    def verify(self, claim_text):
        """
        Full verification pipeline:

          Step 1 — Semantic search
                   KnowledgeManager.find_best_fact() embeds the claim with
                   all-MiniLM-L6-v2 and finds the nearest KB entry via FAISS.
                   No manual keyword lists, no lemmatization, no stop words.

          Step 2 — NLI verdict
                   DistilRoBERTa compares claim against scientific_truth.
                   Detects negation, paraphrase, contradiction automatically.

          Step 3 — 2D verdict lookup
                   (kb_verdict, nli_verdict) → final verdict.

        Args:
            claim_text (str): full sentence to verify

        Returns:
            dict with verdict, truth, match metadata
        """
        # ── Step 1: semantic search ───────────────────────────────────────────
        best_entry, distance = self.kb.find_best_fact(claim_text)

        # ── No meaningful match ───────────────────────────────────────────────
        if best_entry is None or distance > WEAK_MATCH_DISTANCE:
            return {
                "match_found":    False,
                "verdict":        "UNVERIFIED",
                "truth":          "This claim could not be matched against our verified medical database.",
                "risk_level":     "Medium",
                "match_score":    round(float(distance), 4) if distance != float('inf') else None,
                "matched_entry":  None,
                "match_type":     "NONE",
                "nli_verdict":    None,
                "nli_confidence": None,
            }

        match_type = "STRONG" if distance <= STRONG_MATCH_DISTANCE else "WEAK"

        # ── Step 2 + 3: NLI + verdict lookup ─────────────────────────────────
        if self.nli is not None:
            nli_result  = self.nli.verify(claim_text, best_entry["scientific_truth"])
            nli_verdict = nli_result["verdict"]
            nli_conf    = nli_result["confidence"]

            kb_verdict = normalize_verdict(best_entry["verdict"])
            verdict = VERDICT_TABLE.get(
                (kb_verdict, nli_verdict),
                "UNVERIFIED"
            )

            truth_str = (
                f"{best_entry['scientific_truth']} "
                f"[NLI: claim {nli_verdict} the KB fact "
                f"(confidence={nli_conf})]"
            )
            if match_type == "WEAK":
                truth_str = f"[Partial match — verify manually] {truth_str}"

            return {
                "match_found":    True,
                "verdict":        verdict,
                "truth":          truth_str,
                "risk_level":     best_entry.get("risk_level", "Medium"),
                "match_score":    round(float(distance), 4),
                "matched_entry":  best_entry.get("claim_subject"),
                "match_type":     match_type,
                "kb_verdict":     kb_verdict,
                "nli_verdict":    nli_verdict,
                "nli_confidence": nli_conf,
            }

        # ── Fallback: no NLI — return KB verdict directly ─────────────────────
        # No negation detection in this mode — cow urine cures cancer and
        # cow urine does not cure cancer both return the same KB verdict.
        # Only use this mode for testing / when NLI is unavailable.
        print(f"[DEBUG] claim='{claim_text[:40]}' dist={distance:.4f} entry='{best_entry.get('claim_subject')}'")
        return {
            "match_found":    True,
            "verdict":        normalize_verdict(best_entry["verdict"]),
            "truth":          best_entry["scientific_truth"],
            "risk_level":     best_entry.get("risk_level", "Medium"),
            "match_score":    round(float(distance), 4),
            "matched_entry":  best_entry.get("claim_subject"),
            "match_type":     match_type,
            "kb_verdict":     normalize_verdict(best_entry["verdict"]),
            "nli_verdict":    None,
            "nli_confidence": None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST
# Run from project root: python -m src.verifier
# Requires KnowledgeManager and NLIVerifier to be importable.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(".")

    from src.RAGpipeline.knowledge_manager import KnowledgeManager
    from src.RAGpipeline.nli_verifier import NLIVerifier

    print("=" * 65)
    print("  FACT VERIFIER — STANDALONE TEST")
    print("=" * 65)

    # Init
    km = KnowledgeManager()
    km.load_and_index("data/medical_kb")
    km.load_verified_facts("data/verified_facts.json")

    nli      = NLIVerifier()
    verifier = FactVerifier(kb_manager=km, nli_verifier=nli)

    test_claims = [
        "eating sand improves bone density.",
        # Should be FALSE
        "lemon water detoxifies the liver.",
    ]

    print(f"\n{'Claim':<50} {'Verdict':<12} {'NLI':<10} {'Type':<8} Dist")
    print("─" * 95)

    for claim in test_claims:
        r = verifier.verify(claim)
        print(
            f"{claim:<50} "
            f"{r['verdict']:<12} "
            f"{str(r['nli_verdict']):<10} "
            f"{r['match_type']:<8} "
            f"{r['match_score']}"
        )
