import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json 
from config import EMBEDDING_MODEL, FACTS_JSON, KB_PATH

CACHE_FILE = "processed_cache.txt"   # tracks which files are already embedded
INDEX_FILE = "kb_index.faiss"        # saved FAISS index on disk
PASSAGES_FILE = "kb_passages.txt"    # saved passages (one per line, pipe-separated)


class KnowledgeManager:
    def __init__(self, model_name=EMBEDDING_MODEL):
        print(f"--- Loading Embedding Model: {model_name} ---")
        self.model = SentenceTransformer(model_name, local_files_only=True)
        self.index = None
        self.passages = []
        self.passage_sources: list[str] = []
        self.facts_index = None
        self.facts_entries = []
        self.kb_loaded = False
        self.folder_path = None  # set during load_and_index
        self.cache_key = "".join(c if c.isalnum() else "_" for c in model_name.lower()).strip("_")

    # ─────────────────────────────────────────────────────────────────────────
    # CACHE HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _cache_path(self):
        return os.path.join(self.folder_path, f"processed_cache.{self.cache_key}.txt")

    def _index_path(self):
        return os.path.join(self.folder_path, f"kb_index.{self.cache_key}.faiss")

    def _passages_path(self):
        return os.path.join(self.folder_path, f"kb_passages.{self.cache_key}.txt")

    def _passage_sources_path(self):
        return os.path.join(self.folder_path, f"kb_passage_sources.{self.cache_key}.txt")

    def _load_cache(self) -> set:
        """Returns set of filenames already processed."""
        path = self._cache_path()
        if not os.path.exists(path):
            return set()
        with open(path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())

    def _write_cache(self, filenames: list[str]):
        """Persist processed files only after their index batch is durable."""
        with open(self._cache_path(), "w", encoding="utf-8") as f:
            for filename in filenames:
                f.write(filename + "\n")

    def _save_index(self):
        """Save FAISS index to disk."""
        faiss.write_index(self.index, self._index_path())

    def _load_index(self) -> bool:
        """Load FAISS index from disk. Returns True if successful."""
        path = self._index_path()
        if not os.path.exists(path):
            return False
        self.index = faiss.read_index(path)
        return True

    def _save_passages(
        self,
        new_passages: list,
        new_sources: list | None = None,
        append: bool = True,
    ):
        """Save passages (and their parallel source filenames) to disk."""
        mode = "a" if append else "w"
        with open(self._passages_path(), mode, encoding="utf-8") as f:
            for p in new_passages:
                f.write(p.replace("\n", " ").strip() + "\n")
        if new_sources:
            with open(self._passage_sources_path(), mode, encoding="utf-8") as f:
                for src in new_sources:
                    f.write(src + "\n")

    def _load_passages(self) -> list:
        """Load all previously saved passages from disk."""
        path = self._passages_path()
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _load_passage_sources(self) -> list:
        """Load source filename per passage (parallel to passages list)."""
        path = self._passage_sources_path()
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN LOAD + INDEX
    # ─────────────────────────────────────────────────────────────────────────

    def load_and_index(self, folder_path="data/medical_kb"):
        self.folder_path = folder_path
        print(f"\n🔍 Knowledge Base folder : {folder_path}")

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"⚠️  Created {folder_path}. Add medical .txt files there.")
            return

        # ── Step 1: Load existing index + passages from disk ─────────────────
        already_processed = self._load_cache()
        self.passages = self._load_passages()
        self.passage_sources = self._load_passage_sources()
        index_loaded = self._load_index()

        # Source provenance is required by Phase 5.  A legacy or interrupted
        # cache cannot safely be resumed, because its index rows no longer have
        # a trustworthy filename mapping.  Rebuild it as one consistent batch.
        rebuild_for_sources = bool(
            self.passages
            and len(self.passage_sources) != len(self.passages)
        )
        if rebuild_for_sources:
            print("⚠️  Rebuilding KB index: source sidecar is missing or incomplete")
            already_processed = set()
            self.passages = []
            self.passage_sources = []
            self.index = None
            index_loaded = False

        if index_loaded and self.passages:
            print(f"✅ Loaded existing FAISS index ({self.index.ntotal} vectors)")
            print(f"✅ Loaded {len(self.passages)} cached passages")
        else:
            # No saved index — will build fresh
            index_loaded = False
            self.passages = []
            self.index = None

        # ── Step 2: Find unprocessed .txt files ───────────────────────────────
        all_txt_files = [
            f for f in os.listdir(folder_path)
            if f.endswith(".txt")
            and not f.startswith((
                "processed_cache.",
                "kb_passages.",
                "kb_passage_sources.",
            ))
        ]

        new_files = [f for f in all_txt_files if f not in already_processed]
        skipped   = len(all_txt_files) - len(new_files)

        print(f"\n📂 Total .txt files  : {len(all_txt_files)}")
        print(f"   ⏭️  Already cached  : {skipped}")
        print(f"   🆕 New to process  : {len(new_files)}")

        if not new_files and index_loaded and self.passages:
            print("\n✅ Nothing new to process — KB is up to date.\n")
            self.kb_loaded = True
            return

        # ── Step 3: Extract passages from new files ───────────────────────────
        new_passages = []
        new_sources = []

        for filename in new_files:
            filepath = os.path.join(folder_path, filename)
            print(f"   📖 Processing: {filename}")

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f.readlines()]

                filtered = [
                    line for line in lines
                    if len(line) >= 20
                    and not line.isupper()          # skip ALL-CAPS section headers
                    and not line.startswith("SOURCE:")
                    and not line.startswith("URL:")
                    and not line.startswith("TITLE:")
                    and "─" not in line             # skip separator lines
                ]

                print(f"      ✅ {len(filtered)} passages extracted")
                new_passages.extend(filtered)
                new_sources.extend([filename] * len(filtered))

            except Exception as e:
                print(f"      ❌ Error reading {filename}: {e}")

        if not new_passages:
            if self.passages:
                print("\n⚠️  No new passages found but existing KB is loaded.\n")
                self.kb_loaded = True
            else:
                print("\n⚠️  No passages found at all. RAG disabled.\n")
            return

        # ── Step 4: Embed new passages ────────────────────────────────────────
        print(f"\n🔧 Embedding {len(new_passages)} new passages...")
        new_embeddings = self.model.encode(new_passages, show_progress_bar=True)

        # ── Step 5: Add to FAISS index ────────────────────────────────────────
        dimension = new_embeddings.shape[1]

        if self.index is None:
            # Build fresh index
            self.index = faiss.IndexFlatL2(dimension)

        self.index.add(np.array(new_embeddings).astype('float32'))
        self.passages.extend(new_passages)
        self.passage_sources.extend(new_sources)

        # ── Step 6: Save updated index + passages to disk ─────────────────────
        self._save_index()
        self._save_passages(
            new_passages,
            new_sources,
            append=not rebuild_for_sources,
        )
        self._write_cache(sorted(already_processed.union(new_files)))

        print(f"\n✅ FAISS index saved  : {self._index_path()}")
        print(f"✅ Passages saved     : {self._passages_path()}")
        print(f"✅ Total vectors in index : {self.index.ntotal}")
        print(f"✅ Total passages in KB   : {len(self.passages)}\n")

        self.kb_loaded = True

    # ─────────────────────────────────────────────────────────────────────────
    # RETRIEVE
    # ─────────────────────────────────────────────────────────────────────────

    def retrieve_evidence(self, query_triple, top_k=2):
        hits = self.dense_search(query_triple, top_k=top_k)
        return [hit["passage"] for hit in hits]

    def dense_search(self, query: str, top_k: int = 5) -> list[dict]:
        """FAISS dense search returning passage index + text (no string round-trip)."""
        if self.index is None or not self.kb_loaded:
            return []

        query_vector = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)
        hits = []
        for i in indices[0]:
            if i < len(self.passages):
                hits.append({"index": int(i), "passage": self.passages[i]})
        return hits

    def source_for_index(self, passage_index: int) -> str:
        if 0 <= passage_index < len(self.passage_sources):
            return self.passage_sources[passage_index]
        return ""
    def load_verified_facts(self, facts_path=str(FACTS_JSON)):
        """
        Builds a separate FAISS index for verified_facts.json.
        Uses the same MiniLM model already loaded — no extra memory.
        """
        if not os.path.exists(facts_path):
            print(f"[KnowledgeManager] ⚠ verified_facts.json not found at {facts_path}")
            self.facts_index   = None
            self.facts_entries = []
            return

        with open(facts_path, 'r', encoding='utf-8') as f:
            facts = json.load(f)

        # Embed scientific_truth of each entry
        premises   = [entry["scientific_truth"] for entry in facts]
        embeddings = self.model.encode(premises, show_progress_bar=False)

        self.facts_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.facts_index.add(embeddings.astype('float32'))
        self.facts_entries = facts

        print(f"[KnowledgeManager] ✅ Loaded {len(facts)} verified facts into FAISS index")

    def find_best_fact(self, claim_text):
        """
        Semantically finds the most relevant KB entry for a claim.
        Returns (best_entry, distance). Lower distance = better match.
        """
        if self.facts_index is None or not self.facts_entries:
            return None, float('inf')

        vec               = self.model.encode([claim_text]).astype('float32')
        distances, indices = self.facts_index.search(vec, 1)

        best_idx   = indices[0][0]
        best_dist  = distances[0][0]
        best_entry = self.facts_entries[best_idx]

        return best_entry, best_dist
# ─────────────────────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    km = KnowledgeManager()
    km.load_and_index(str(KB_PATH))

    if km.kb_loaded:
        test_queries = [
            "warm water improves digestion",
            "turmeric cures cancer",
            "vitamin C prevents all infections",
            "hot water detoxes body",
        ]
        print("=" * 60)
        print("RETRIEVAL TEST")
        print("=" * 60)
        for query in test_queries:
            evidence = km.retrieve_evidence(query, top_k=1)
            print(f"\n📋 QUERY  : {query}")
            print(f"✅ MATCH  : {evidence[0][:120] if evidence else 'None'}")
