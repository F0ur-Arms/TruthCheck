import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class FactVectorStore:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Load a lightweight embedding model
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []

    def build_index(self, json_file):
        """Converts JSON facts into a searchable Vector Database."""
        with open(json_file, 'r') as f:
            facts = json.load(f)
        
        # We combine the claim and the truth for a rich embedding
        sentences = [f"{item['triple'][0]} {item['triple'][1]} {item['triple'][2]}" for item in facts]
        self.metadata = facts # Keep original data to return later
        
        # Convert text to vectors
        embeddings = self.model.encode(sentences)
        
        # Create the FAISS index (using L2 distance for similarity)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        print(f"✅ Indexed {len(sentences)} facts for RAG.")

    def search(self, query, top_k=1):
        """Finds the most semantically similar fact."""
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vec).astype('float32'), top_k)
        
        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.metadata[idx])
        return results

if __name__ == "__main__":
    store = FactVectorStore()
    store.build_index('data/verified_facts.json')
    
    # Test semantic search: Even if the words don't match exactly
    test_query = "Is turmeric good for health?"
    match = store.search(test_query)
    print(f"Query: {test_query}\nTop Match: {match}")