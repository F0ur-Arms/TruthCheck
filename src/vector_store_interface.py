"""Abstract Vector Store Interface for FAISS / pgvector / Qdrant portability.

Provides a unified interface so vector search can be abstracted away from specific vector storage backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class VectorStoreInterface(ABC):
    """Abstract base class for vector store backends."""

    @abstractmethod
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Add dense vectors and associated metadata items to the index."""
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for top_k nearest neighbors given a query vector."""
        pass

    @abstractmethod
    def save(self, destination_path: str) -> None:
        """Persist vector index and metadata to disk/storage."""
        pass

    @abstractmethod
    def load(self, source_path: str) -> bool:
        """Load vector index and metadata from storage."""
        pass


class FAISSVectorStoreAdapter(VectorStoreInterface):
    """FAISS-backed vector store implementation."""

    def __init__(self, dimension: int = 1024):
        import faiss
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product / Cosine similarity (normalized)
        self.metadata_store: List[Dict[str, Any]] = []

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        if len(vectors) != len(metadata):
            raise ValueError("Length of vectors and metadata must match.")
        
        vectors_float32 = np.array(vectors, dtype=np.float32)
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(vectors_float32, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_vectors = vectors_float32 / norms

        self.index.add(normalized_vectors)
        self.metadata_store.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            return []

        q_vec = np.array([query_vector], dtype=np.float32)
        q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True)
        q_norm[q_norm == 0] = 1.0
        q_vec = q_vec / q_norm

        scores, indices = self.index.search(q_vec, min(top_k, self.index.ntotal))
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and idx < len(self.metadata_store):
                item = dict(self.metadata_store[idx])
                item["score"] = float(score)
                results.append(item)
        return results

    def save(self, destination_path: str) -> None:
        import faiss
        import json
        faiss.write_index(self.index, f"{destination_path}.faiss")
        with open(f"{destination_path}.meta.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata_store, f, ensure_ascii=False, indent=2)

    def load(self, source_path: str) -> bool:
        import faiss
        import json
        import os
        faiss_path = f"{source_path}.faiss"
        meta_path = f"{source_path}.meta.json"
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            return False

        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata_store = json.load(f)
        return True


class PGVectorStoreAdapter(VectorStoreInterface):
    """PostgreSQL + pgvector vector store implementation placeholder."""

    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string
        self.memory_mock: List[Tuple[np.ndarray, Dict[str, Any]]] = []

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        for vec, meta in zip(vectors, metadata):
            self.memory_mock.append((vec, meta))

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        # Mock search calculation for unit testing without live PostgreSQL
        results = []
        for vec, meta in self.memory_mock:
            score = float(np.dot(query_vector, vec) / (np.linalg.norm(query_vector) * np.linalg.norm(vec) + 1e-9))
            item = dict(meta)
            item["score"] = score
            results.append(item)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def save(self, destination_path: str) -> None:
        pass

    def load(self, source_path: str) -> bool:
        return True
