import unittest
import numpy as np
from src.vector_store_interface import FAISSVectorStoreAdapter, PGVectorStoreAdapter


class VectorStoreTests(unittest.TestCase):
    def test_faiss_adapter_add_and_search(self):
        store = FAISSVectorStoreAdapter(dimension=4)
        vectors = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype=np.float32)
        meta = [
            {"passage": "passage 1"},
            {"passage": "passage 2"},
            {"passage": "passage 3"}
        ]

        store.add_vectors(vectors, meta)
        query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        results = store.search(query, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["passage"], "passage 1")

    def test_pgvector_adapter_mock_search(self):
        store = PGVectorStoreAdapter()
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        meta = [{"passage": "p1"}, {"passage": "p2"}]
        store.add_vectors(vectors, meta)

        query = np.array([0.0, 0.9], dtype=np.float32)
        res = store.search(query, top_k=1)
        self.assertEqual(res[0]["passage"], "p2")


if __name__ == "__main__":
    unittest.main()
