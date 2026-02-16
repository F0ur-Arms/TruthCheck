import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class KnowledgeManager:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # 1. Load the embedding model (Lightweight for CPU)
        print(f"--- Loading Embedding Model: {model_name} ---")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.passages = []
        self.kb_loaded = False  # Track if KB is successfully loaded

    def load_and_index(self, folder_path="TruthCheck/data/medical_kb/"):
        """Reads all .txt files in a folder and creates a searchable index."""
        
        # Debug: Print the path being checked
        print(f"🔍 Looking for medical KB in: {folder_path}")
        print(f"🔍 Current working directory: {os.getcwd()}")
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"⚠️ Created {folder_path}. Please add medical text files there.")
            return

        all_text = []
        txt_files_found = 0
        
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                txt_files_found += 1
                filepath = os.path.join(folder_path, filename)
                print(f"📖 Reading: {filepath}")
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f.readlines()]
                        
                        print(f"   📄 Total lines in file: {len(lines)}")
                        
                        # Filter: Remove empty lines and section headers (all uppercase)
                        # Keep lines that are substantial facts (> 20 chars and not all uppercase)
                        filtered_lines = []
                        for line in lines:
                            # Skip empty lines
                            if len(line) == 0:
                                continue
                            # Skip section headers (all uppercase like "GENERAL HEALTH")
                            if line.isupper():
                                continue
                            # Skip very short lines (likely incomplete)
                            if len(line) < 20:
                                continue
                            # Keep the rest
                            filtered_lines.append(line)
                        
                        print(f"   ✅ Extracted {len(filtered_lines)} medical facts")
                        
                        if filtered_lines and len(all_text) < 5:
                            # Show first few as sample
                            print(f"   📝 Sample: {filtered_lines[0][:60]}...")
                        
                        all_text.extend(filtered_lines)
                        
                except Exception as e:
                    print(f"   ❌ Error reading {filename}: {e}")

        if txt_files_found == 0:
            print(f"⚠️ No .txt files found in {folder_path}")
            return

        if not all_text:
            print("⚠️ No medical data found to index. RAG will be disabled.")
            return

        self.passages = all_text
        self.kb_loaded = True
        print(f"\n✅ Successfully loaded {len(self.passages)} medical passages")
        print(f"🔧 Now building FAISS index...")

        # 2. Convert text to vectors
        embeddings = self.model.encode(self.passages, show_progress_bar=True)
        
        # 3. Build the FAISS Index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        print("✅ Knowledge Base Index Ready\n")

    def retrieve_evidence(self, query_triple, top_k=2):
        """
        Uses the Triple string (e.g., 'turmeric water cures cancer') 
        to find the most relevant scientific evidence.
        """
        if self.index is None or not self.kb_loaded:
            return []  # Return empty list if KB not loaded

        # Convert triple to vector
        query_vector = self.model.encode([query_triple]).astype('float32')
        
        # Search index
        distances, indices = self.index.search(query_vector, top_k)
        
        results = [self.passages[i] for i in indices[0]]
        return results


if __name__ == "__main__":
    # Test Run
    print("="*60)
    print("TESTING KNOWLEDGE MANAGER")
    print("="*60 + "\n")
    
    km = KnowledgeManager()
    km.load_and_index("data/medical_kb/")
    
    if km.kb_loaded:
        print("\n" + "="*60)
        print("TESTING RETRIEVAL")
        print("="*60)
        
        # Test queries
        test_queries = [
            "warm water improves blood circulation",
            "hot water kills COVID virus",
            "turmeric cures cancer",
            "vitamin C prevents all infections",
            "drinking water helps digestion"
        ]
        
        for query in test_queries:
            evidence = km.retrieve_evidence(query, top_k=1)
            print(f"\n📋 QUERY: {query}")
            if evidence:
                print(f"✅ TOP MATCH: {evidence[0]}")
            else:
                print("❌ No evidence found")
                
        print("\n" + "="*60)
        print(f"✅ Test Complete! KB has {len(km.passages)} passages indexed")
        print("="*60)
    else:
        print("\n⚠️ Knowledge base was not loaded successfully.")
        print("Please check:")
        print("  1. Does data/medical_kb/ directory exist?")
        print("  2. Does it contain .txt files?")
        print("  3. Do the .txt files have content?")