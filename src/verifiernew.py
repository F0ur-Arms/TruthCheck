import itertools
import os
import re
import requests
import spacy
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import DEBUG_LOG_PATH as CONFIG_DEBUG_LOG_PATH
from config import EMBEDDING_MODEL, FACTS_JSON
from src.llm_fallback import ConfiguredLLMVerifier
from src.verification.normalization import normalize_verdict

DEBUG_LOG_PATH = str(CONFIG_DEBUG_LOG_PATH)

def debug_log(msg=""):
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# VERDICT TABLE (for KB + NLI interaction)
# ─────────────────────────────────────────────────────────────────────────────
VERDICT_TABLE = {
    ("TRUE",  "SUPPORTS"): "TRUE",
    ("TRUE",  "REFUTES"):  "FALSE",
    ("TRUE",  "NEI"):      "UNVERIFIED",
    ("FALSE", "SUPPORTS"): "FALSE",
    ("FALSE", "REFUTES"):  "FALSE",
    ("FALSE", "NEI"):      "UNVERIFIED",
    ("MIXED", "SUPPORTS"): "MIXED",
    ("MIXED", "REFUTES"):  "MIXED",
    ("MIXED", "NEI"):      "UNVERIFIED",
}

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
STRONG_MATCH_DISTANCE   = 0.82   # FAISS distance for "strong" KB match
NLI_MIN_CONFIDENCE      = 0.65   # NLI confidence threshold

# LLM-specific thresholds
LLM_CONFIDENCE_THRESHOLD = 0.6   # Minimum hybrid confidence to trust LLM
VOTING_SAMPLES = 3               # Number of voting samples

# ─────────────────────────────────────────────────────────────────────────────
# Deprecated local fallback configuration.  The live FactVerifier uses
# ConfiguredLLMVerifier; this legacy class is retained temporarily only to keep
# historical debugging code readable and has no executable local endpoint.
# ─────────────────────────────────────────────────────────────────────────────
OLLAMA_URL = None
OLLAMA_MODEL = "deprecated"

# ─────────────────────────────────────────────────────────────────────────────
# CACHE FILES FOR FAISS INDEX
# ─────────────────────────────────────────────────────────────────────────────
FACTS_INDEX_FILE = "verified_facts.faiss"
FACTS_CACHE_FILE = "verified_facts_cache.json"


# ─────────────────────────────────────────────────────────────────────────────
# BUILT-IN KNOWLEDGE MANAGER (NO RAG DEPENDENCY)
# ─────────────────────────────────────────────────────────────────────────────
class BuiltInKnowledgeManager:
    """
    Standalone FAISS-based knowledge manager.
    Caches index to avoid rebuilding on every run.
    """
    
    def __init__(self, model_name=EMBEDDING_MODEL):
        print(f"[KB] Loading embedding model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, local_files_only=True)
        self.facts_index = None
        self.facts_entries = []
        self.facts_path = None
        
    def _get_cache_paths(self, facts_path):
        """Get paths for cached index and metadata."""
        base_dir = os.path.dirname(facts_path)
        model_key = re.sub(r"[^a-z0-9]+", "_", self.model_name.lower()).strip("_")
        index_path = os.path.join(base_dir, f"verified_facts.{model_key}.faiss")
        cache_path = os.path.join(base_dir, f"verified_facts.{model_key}.cache.json")
        return index_path, cache_path
    
    def _load_cache_metadata(self, cache_path):
        """Load cached metadata (file hash, entry count)."""
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None
    
    def _save_cache_metadata(self, cache_path, facts_path, num_entries):
        """Save cache metadata."""
        # Use file modification time as simple "hash"
        file_mtime = os.path.getmtime(facts_path)
        
        metadata = {
            "file_mtime": file_mtime,
            "num_entries": num_entries,
            "facts_path": facts_path
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def _is_cache_valid(self, cache_metadata, facts_path, current_facts):
        """Check if cached index is still valid."""
        if cache_metadata is None:
            return False
        
        # Check if file was modified
        current_mtime = os.path.getmtime(facts_path)
        if cache_metadata.get("file_mtime") != current_mtime:
            print("[KB] Facts file modified — rebuilding index")
            return False
        
        # Check if entry count matches
        if cache_metadata.get("num_entries") != len(current_facts):
            print("[KB] Entry count mismatch — rebuilding index")
            return False
        
        return True
    
    def load_verified_facts(self, facts_path=str(FACTS_JSON)):
        """
        Load verified facts and build/load FAISS index with caching.
        
        Caching logic:
        - If facts.json unchanged AND cached index exists → load from cache
        - Otherwise → rebuild index and save cache
        """
        self.facts_path = facts_path
        
        if not os.path.exists(facts_path):
            print(f"[KB] verified_facts.json not found at {facts_path}")
            self.facts_index = None
            self.facts_entries = []
            return
        
        # Load facts from JSON
        with open(facts_path, 'r', encoding='utf-8') as f:
            facts = json.load(f)
        
        self.facts_entries = facts
        
        # Get cache paths
        index_path, cache_path = self._get_cache_paths(facts_path)
        
        # Try to load from cache
        cache_metadata = self._load_cache_metadata(cache_path)
        
        if (cache_metadata and 
            os.path.exists(index_path) and 
            self._is_cache_valid(cache_metadata, facts_path, facts)):
            
            # Load cached index
            try:
                self.facts_index = faiss.read_index(index_path)
                print(f"[KB] ✅ Loaded cached FAISS index ({self.facts_index.ntotal} facts)")
                print(f"[KB] ✅ Using cached index from: {index_path}")
                return
            except Exception as e:
                print(f"[KB] ⚠ Failed to load cache: {e} — rebuilding")
        
        # Build fresh index
        print(f"[KB] 🔧 Building FAISS index for {len(facts)} facts...")
        
        # Embed scientific_truth of each entry
        premises = [entry["scientific_truth"] for entry in facts]
        embeddings = self.model.encode(premises, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.facts_index = faiss.IndexFlatL2(dimension)
        self.facts_index.add(embeddings.astype('float32'))
        
        # Save to cache
        try:
            faiss.write_index(self.facts_index, index_path)
            self._save_cache_metadata(cache_path, facts_path, len(facts))
            print(f"[KB] ✅ FAISS index cached to: {index_path}")
            print(f"[KB] ✅ Metadata cached to: {cache_path}")
        except Exception as e:
            print(f"[KB] ⚠ Failed to save cache: {e}")
        
        print(f"[KB] ✅ Indexed {len(facts)} verified facts")
    
    def find_best_fact(self, claim_text):
        """
        Find the most semantically similar fact to the claim.
        Returns (best_entry, distance). Lower distance = better match.
        """
        if self.facts_index is None or not self.facts_entries:
            return None, float('inf')
        
        vec = self.model.encode([claim_text]).astype('float32')
        distances, indices = self.facts_index.search(vec, 1)
        
        best_idx = indices[0][0]
        best_dist = distances[0][0]
        best_entry = self.facts_entries[best_idx]
        
        return best_entry, best_dist


# ─────────────────────────────────────────────────────────────────────────────
# LLM VERIFIER (TIER 2) — WITH VERBALIZED + VOTING CONFIDENCE
# ─────────────────────────────────────────────────────────────────────────────
class LLMVerifier:
    """
    Calls a local Ollama model to fact-check a health claim.
    Implements:
    1. Verbalized confidence (asks model for confidence score)
    2. Self-consistency voting (multiple samples with temperature)
    3. Hybrid confidence formula
    """

    def find_best_fact_with_hybrid_confidence(self, claim_text: str, n_samples=VOTING_SAMPLES):
        """
        Use hybrid confidence: verbalized + voting.
        
        Args:
            claim_text: The claim to verify
            n_samples: Number of voting samples (default: 3)
        
        Returns:
            (entry_dict, hybrid_confidence) or (None, inf) on failure
        """
        print(f"[Tier 2 - LLM] Evaluating with hybrid confidence (n={n_samples})...")
        
        # ── STEP 1: Get verbalized confidence (single call, temp=0.1) ──────
        verbalized_entry, verbalized_conf = self._get_verbalized_confidence(claim_text)
        
        if verbalized_entry is None:
            return None, float("inf")
        
        # ── STEP 2: Get voting confidence (multiple calls, temp=0.7) ───────
        voting_verdict, voting_conf, vote_distribution = self._get_voting_confidence(
            claim_text, 
            n_samples
        )
        
        # ── STEP 3: Calculate hybrid confidence ────────────────────────────
        hybrid_confidence = (0.3 * verbalized_conf) + (0.7 * voting_conf)
        
        print(f"[Tier 2 - LLM] Verbalized Conf: {verbalized_conf:.2%}")
        print(f"[Tier 2 - LLM] Voting Conf: {voting_conf:.2%} (votes: {vote_distribution})")
        print(f"[Tier 2 - LLM] Hybrid Conf: {hybrid_confidence:.2%}")
        
        # ── DEBUG LOG ──────────────────────────────────────────────────────
        debug_log("\n" + "="*80)
        debug_log("[LLM HYBRID CONFIDENCE]")
        debug_log(f"CLAIM: {claim_text}")
        debug_log(f"Verbalized Verdict: {verbalized_entry['verdict']} (Conf: {verbalized_conf:.2%})")
        debug_log(f"Voting Verdict: {voting_verdict} (Conf: {voting_conf:.2%})")
        debug_log(f"Vote Distribution: {vote_distribution}")
        debug_log(f"Hybrid Confidence: {hybrid_confidence:.2%}")
        debug_log("="*80)
        
        # ── Use voting verdict as final (majority wins) ────────────────────
        final_entry = {
            "explanation": verbalized_entry["explanation"],
            "verdict": voting_verdict,  # Use majority vote
            "source": f"LLM ({OLLAMA_MODEL}) - Hybrid UQ",
            "claim_subject": "LLM Assessment",
            "risk_level": "Medium",
            "verbalized_confidence": verbalized_conf,
            "voting_confidence": voting_conf,
            "hybrid_confidence": hybrid_confidence,
            "vote_distribution": vote_distribution,
        }
        
        return final_entry, hybrid_confidence

    def _get_verbalized_confidence(self, claim_text: str):
        """
        Ask the model for explicit confidence score.
        Uses temperature=0.1 for consistency.
        """
        prompt = (
            f"Health claim: {claim_text}\n\n"
            "Task: Verify this claim using your medical knowledge.\n\n"
            "Reply in exactly 3 lines:\n"
            "Verdict: TRUE or FALSE or UNVERIFIED\n"
            "Confidence: [0.0-1.0] (how certain are you?)\n"
            "Reason: one sentence explaining why\n"
        )
        
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temp for consistency
                        "num_predict": 80,
                    },
                },
                timeout=60,
            )
            response.raise_for_status()
            raw_text = response.json().get("response", "").strip()
            
            verdict, confidence, reason = self._parse_llm_response_with_confidence(raw_text)
            
            entry = {
                "explanation": reason,
                "verdict": verdict,
                "source": f"LLM ({OLLAMA_MODEL})",
            }
            
            return entry, confidence
            
        except requests.exceptions.ConnectionError:
            print("[Tier 2 - LLM] ⚠ Cannot connect to Ollama. Is it running?")
            return None, float("inf")
        except Exception as e:
            print(f"[Tier 2 - LLM] Error in verbalized confidence: {e}")
            return None, float("inf")

    def _get_voting_confidence(self, claim_text: str, n_samples: int):
        """
        Self-consistency voting: run multiple queries and count agreements.
        Uses temperature=0.7 for diversity.
        
        Returns:
            (majority_verdict, confidence, vote_distribution)
        """
        prompt = (
            f"Health claim: {claim_text}\n\n"
            "Reply in exactly 2 lines:\n"
            "Verdict: TRUE or FALSE or UNVERIFIED\n"
            "Reason: one sentence\n"
        )
        
        votes = {"TRUE": 0, "FALSE": 0, "UNVERIFIED": 0}
        reasons = []
        
        for i in range(n_samples):
            try:
                response = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,  # Higher for diversity
                            "num_predict": 60,
                        },
                    },
                    timeout=60,
                )
                response.raise_for_status()
                raw_text = response.json().get("response", "").strip()
                
                verdict, _, reason = self._parse_llm_response_with_confidence(raw_text)
                votes[verdict] += 1
                reasons.append(reason)
                
                print(f"[Voting {i+1}/{n_samples}] {verdict}")
                
            except Exception as e:
                print(f"[Voting] Sample {i+1} failed: {e}")
                # Count as UNVERIFIED on failure
                votes["UNVERIFIED"] += 1
        
        # Calculate results
        total_votes = sum(votes.values())
        if total_votes == 0:
            return "UNVERIFIED", 0.0, votes
        
        majority_verdict = max(votes, key=votes.get)
        voting_confidence = votes[majority_verdict] / total_votes
        
        return majority_verdict, voting_confidence, votes

    @staticmethod
    def _parse_llm_response_with_confidence(text: str):
        """
        Extract Verdict, Confidence, and Reason from LLM response.
        
        Returns:
            (verdict, confidence, reason)
        """
        verdict = "UNVERIFIED"
        confidence = 0.5  # Default medium uncertainty
        reason = text.strip()
        
        # Extract verdict
        verdict_match = re.search(
            r"verdict\s*[:\-]\s*(TRUE|FALSE|UNVERIFIED)", 
            text, 
            re.IGNORECASE
        )
        if verdict_match:
            verdict = verdict_match.group(1).upper()
        else:
            # Keyword fallback
            lower = text.lower()
            if "false" in lower or "incorrect" in lower or "not true" in lower:
                verdict = "FALSE"
            elif "true" in lower or "correct" in lower or "accurate" in lower:
                verdict = "TRUE"
        
        # Extract confidence
        conf_match = re.search(
            r"confidence\s*[:\-]\s*(0?\.\d+|1\.0|0|1)", 
            text, 
            re.IGNORECASE
        )
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
                confidence = min(max(confidence, 0.0), 1.0)  # Clamp to [0,1]
            except ValueError:
                confidence = 0.5
        
        # Extract reason
        reason_match = re.search(
            r"reason\s*[:\-]\s*(.+)", 
            text, 
            re.IGNORECASE | re.DOTALL
        )
        if reason_match:
            reason_raw = reason_match.group(1).strip()
            reason = reason_raw.split("\n")[0].strip()
        
        return verdict, confidence, reason


# ─────────────────────────────────────────────────────────────────────────────
# FACT VERIFIER (MAIN CLASS)
# ─────────────────────────────────────────────────────────────────────────────
class FactVerifier:
    def __init__(
        self,
        facts_path="data/final_verifiedfacts.json",
        nli_verifier=None,
        embedding_model=EMBEDDING_MODEL
    ):
        """
        Initialize FactVerifier with built-in knowledge management.
        
        Args:
            facts_path: Path to verified_facts.json
            nli_verifier: Optional NLI verifier for alignment checking
            embedding_model: SentenceTransformer model name
        """
        # Built-in KB manager (no RAG dependency)
        self.kb = BuiltInKnowledgeManager(model_name=embedding_model)
        self.kb.load_verified_facts(facts_path)
        
        self.nli = nli_verifier
        self.llm = ConfiguredLLMVerifier()

        print("[FactVerifier] Loading SpaCy for NER/Keyword filtering...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("[FactVerifier] ⚠ SpaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None

        if self.nli is None:
            print("[FactVerifier] ⚠ No NLI injected — KB verdict only, no negation detection.")
        else:
            print("[FactVerifier] ✅ Two-Tier Verifier Active (KB + LLM with Hybrid UQ)")

    # ─────────────────────────────────────────────────────────────────────────
    # HELPER: ENTITY / NOUN OVERLAP (GATE 2 for weak KB matches)
    # ─────────────────────────────────────────────────────────────────────────
    def _has_topic_overlap(self, claim, text):
        """Check if claim and KB entry share key entities/nouns."""
        if not self.nlp:
            return True
        
        doc1 = self.nlp(claim.lower())
        doc2 = self.nlp(text.lower())
        
        tokens1 = {t.lemma_ for t in doc1 if not t.is_stop and t.is_alpha and len(t) > 2}
        tokens2 = {t.lemma_ for t in doc2 if not t.is_stop and t.is_alpha and len(t) > 2}
        
        overlap = tokens1.intersection(tokens2)
        print(f"[Gate 2] Token overlap: {overlap}")
        
        return len(overlap) > 0

    # ─────────────────────────────────────────────────────────────────────────
    # TIER-1 PROBE (mode gate input — no LLM fallback)
    # ─────────────────────────────────────────────────────────────────────────
    def tier1_lookup(self, claim_text):
        """Run Tier-1 FAISS + NLI gates only; never falls through to LLM."""
        best_entry, distance = self.kb.find_best_fact(claim_text)
        if best_entry is None:
            return self._unverified_response(float("inf"), "No KB entry found")
        return self._run_local_gates(claim_text, best_entry, distance)

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN VERIFY METHOD (TWO-TIER CASCADE)
    # ─────────────────────────────────────────────────────────────────────────
    def verify(self, claim_text):
        """
        Two-tier verification cascade:
        
        TIER 1: Local Knowledge Base (FAISS + NLI)
        ↓ (if no strong match)
        TIER 2: LLM Fallback (Verbalized + Voting Confidence)
        """
        
        # ══════════════════════════════════════════════════════════════════════
        # TIER 1: LOCAL KNOWLEDGE BASE
        # ══════════════════════════════════════════════════════════════════════
        print(f"[FactVerifier] TIER 1: Searching Local KB...")
        best_entry, distance = self.kb.find_best_fact(claim_text)
        
        print(f"[Tier 1 - KB] Best match distance: {distance:.4f}")
        
        # Check if we have a strong enough match
        if distance <= STRONG_MATCH_DISTANCE:
            print(f"[Tier 1 - KB] Strong match found (distance ≤ {STRONG_MATCH_DISTANCE})")
            result = self._run_local_gates(claim_text, best_entry, distance)
            
            # If gates pass, return KB result
            if result['match_found']:
                print(f"[Tier 1 - KB] ✅ ACCEPTED | Verdict: {result['verdict']}")
                return result
            else:
                print(f"[Tier 1 - KB] ❌ REJECTED by gates | Falling to Tier 2...")
        else:
            print(f"[Tier 1 - KB] No strong match (distance={distance:.4f} > {STRONG_MATCH_DISTANCE})")
        
        # ══════════════════════════════════════════════════════════════════════
        # TIER 2: LLM FALLBACK (WITH HYBRID CONFIDENCE)
        # ══════════════════════════════════════════════════════════════════════
        print("[FactVerifier] TIER 2: configured LLM fallback...")
        try:
            llm_entry = self.llm.verify(claim_text)
        except Exception as exc:
            return self._unverified_response(float("inf"), f"Tier 2 LLM request failed: {exc}")
        hybrid_conf = llm_entry["confidence"] if llm_entry else 0.0
        
        if llm_entry is None:
            # A deployment has not configured a hosted LLM fallback.
            return self._unverified_response(
                float("inf"), 
                "Tier 1 failed and the configured LLM fallback is unavailable."
            )
        
        # Check if hybrid confidence is high enough
        if hybrid_conf >= LLM_CONFIDENCE_THRESHOLD:
            print(f"[Tier 2 - LLM] ✅ ACCEPTED | Verdict: {llm_entry['verdict']} | Conf: {hybrid_conf:.2%}")
            return self._build_llm_result(claim_text, llm_entry, hybrid_conf)
        else:
            print(f"[Tier 2 - LLM] ⚠ LOW CONFIDENCE | Hybrid: {hybrid_conf:.2%} < {LLM_CONFIDENCE_THRESHOLD:.2%}")
            return self._unverified_response(
                hybrid_conf,
                f"LLM uncertain (hybrid confidence: {hybrid_conf:.2%})"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # TIER 1: LOCAL KB GATE RUNNER (NO CROSSENCODER)
    # ─────────────────────────────────────────────────────────────────────────
    def _run_local_gates(self, claim_text, best_entry, distance):
        """
        Two-gate validation for KB matches:
        Gate 1: FAISS distance (already passed if we're here)
        Gate 2: Topic overlap (for weak matches)
        Gate 3: NLI verdict alignment (if available)
        """
        kb_truth = best_entry["scientific_truth"]  # Changed from "explanation"
        match_type = "STRONG" if distance <= STRONG_MATCH_DISTANCE else "WEAK"
        
        print(f"[Gate 1] FAISS Distance: {distance:.4f} → {match_type} match")
        
        # ── GATE 2: Topic Overlap (only for weak matches) ─────────────────────
        if match_type == "WEAK":
            if not self._has_topic_overlap(claim_text, kb_truth):
                print(f"[Gate 2] ❌ FAILED: No topic overlap")
                return self._unverified_response(
                    distance, 
                    "Weak KB match with no shared entities"
                )
            print(f"[Gate 2] ✅ PASSED: Topic overlap found")
        
        # ── GATE 3: NLI Verdict (if available) ────────────────────────────────
        if self.nli is not None:
            nli_result = self.nli.verify(claim_text, kb_truth)
            nli_verdict = nli_result["verdict"]
            nli_conf = nli_result["confidence"]
            
            print(f"[Gate 3] NLI Verdict: {nli_verdict} (Conf: {nli_conf:.2%})")
            
            # For weak matches, require high NLI confidence
            if match_type == "WEAK" and nli_conf < NLI_MIN_CONFIDENCE:
                print(f"[Gate 3] ❌ FAILED: Weak match + low NLI confidence")
                return self._unverified_response(
                    distance,
                    f"Weak KB match with low NLI confidence ({nli_conf:.2%})"
                )
            
            print(f"[Gate 3] ✅ PASSED: NLI alignment confirmed")
            
            # Combine KB verdict + NLI verdict using table
            final_verdict = VERDICT_TABLE.get(
                (normalize_verdict(best_entry["verdict"]), nli_verdict), 
                "UNVERIFIED"
            )
            
            truth_str = f"{kb_truth} [NLI: {nli_verdict}]"
            
            return {
                "match_found":    True,
                "verdict":        final_verdict,
                "truth":          truth_str,
                "risk_level":     best_entry.get("risk_level", "Medium"),
                "match_score":    round(float(distance), 4),
                "matched_entry":  best_entry.get("claim_subject"),
                "match_type":     match_type,
                "source":         "Tier 1 (Local KB)",
                "nli_verdict":    nli_verdict,
                "nli_confidence": nli_conf,
            }
        
        # No NLI available — use KB verdict directly
        print(f"[Gate 3] NLI not available — using KB verdict directly")
        
        return {
            "match_found":    True,
            "verdict":        best_entry["verdict"],
            "truth":          kb_truth,
            "risk_level":     best_entry.get("risk_level", "Medium"),
            "match_score":    round(float(distance), 4),
            "matched_entry":  best_entry.get("claim_subject"),
            "match_type":     match_type,
            "source":         "Tier 1 (Local KB)",
            "nli_verdict":    None,
            "nli_confidence": None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # TIER 2: LLM RESULT BUILDER
    # ─────────────────────────────────────────────────────────────────────────
    def _build_llm_result(self, claim_text, llm_entry, hybrid_conf):
        """Build result dict from LLM output."""
        return {
            "match_found":    True,
            "verdict":        llm_entry["verdict"],
            "truth":          f"{llm_entry['explanation']} (Source: {llm_entry['source']})",
            "risk_level":     llm_entry.get("risk_level", "Medium"),
            "match_score":    hybrid_conf,  # Use hybrid confidence as match score
            "matched_entry":  llm_entry.get("claim_subject"),
            "match_type":     "LLM",
            "source":         f"Tier 2 ({llm_entry['source']})",
            "nli_verdict":    None,
            "nli_confidence": None,
            # LLM-specific fields
            "verbalized_confidence": llm_entry.get("verbalized_confidence"),
            "voting_confidence":     llm_entry.get("voting_confidence"),
            "hybrid_confidence":     hybrid_conf,
            "vote_distribution":     llm_entry.get("vote_distribution"),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # UNVERIFIED RESPONSE HELPER
    # ─────────────────────────────────────────────────────────────────────────
    def _unverified_response(self, distance, reason):
        """Return standardized unverified response."""
        return {
            "match_found":    False,
            "verdict":        "UNVERIFIED",
            "truth":          f"Unverified: {reason}",
            "risk_level":     "Medium",
            "match_score":    round(float(distance), 4) if distance != float("inf") else None,
            "matched_entry":  None,
            "match_type":     "NONE",
            "source":         None,
            "nli_verdict":    None,
            "nli_confidence": None,
        }


if __name__ == "__main__":
    print("="*80)
    print("  FACT VERIFIER - BUILT-IN FAISS (NO RAG DEPENDENCY)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Tier 1: Local KB (FAISS threshold: {STRONG_MATCH_DISTANCE})")
    print(f"  Tier 2: LLM ({OLLAMA_MODEL})")
    print(f"    - Voting samples: {VOTING_SAMPLES}")
    print(f"    - Hybrid formula: 0.3×verbalized + 0.7×voting")
    print(f"    - Acceptance threshold: {LLM_CONFIDENCE_THRESHOLD:.0%}")
    print(f"\n  NOTE: CrossEncoder removed, FAISS index cached")
    print("="*80)
